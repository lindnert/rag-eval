import asyncio
import re
import statistics
import time
from datetime import datetime

import aiohttp

from rag.llm_config import (
    LLAMACPP_GEN_BASE_URL,
    LLAMACPP_RAG_CONCURRENCY,
    LLAMACPP_RAG_ENABLE_THINKING,
    LLAMACPP_RAG_MAX_TOKENS,
    LLAMACPP_RAG_MODEL,
    LLAMACPP_RAG_TEMPERATURE,
    LLAMACPP_RAG_TOP_LOGPROBS,
    LLAMACPP_RAG_TOP_P,
    RAG_K,
    RAG_SC_GEN_MEAN_LOGPROB_THRESHOLD,
    RAG_SC_GEN_MIN_LOGPROB_THRESHOLD,
    RAG_SC_HYDE_MAX_TOKENS,
    RAG_SC_RETRIEVAL_BEST_THRESHOLD,
    RAG_SC_RETRIEVAL_SPREAD_THRESHOLD,
    RAG_SC_SCORE_DIRECTION,
)
from retrieval import QUERY_PREFIX, build_vectorstore

VARIANTS = ("no_rag", "rag", "rag_sc")
_RAG_VARIANTS = {"rag", "rag_sc"}

_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore


SYSTEM_PROMPT_RAG = (
    "Du bist ein hilfreicher Ernährungsberater, der evidenzbasierte Empfehlungen gibt. "
    "Antworte kurz und prägnant (max 3-4 Absätze). Verwende nur Informationen aus dem Kontext."
)

SYSTEM_PROMPT_NO_RAG = (
    "Du bist ein hilfreicher Ernährungsberater, der evidenzbasierte Empfehlungen gibt. "
    "Antworte kurz und prägnant (max 3-4 Absätze)."
)

# Stricter prompt used by the rag_sc regen path when generation triggers fire.
# Kept deliberately short and unambiguous: earlier multi-step versions ("first
# identify, then write…") sent the 4B thinking model into multi-thousand-token
# meta-loops about how to interpret the instructions instead of answering.
SYSTEM_PROMPT_RAG_STRICT = (
    "Du bist ein evidenzbasierter Ernährungsberater. "
    "Verwende ausschließlich Informationen aus dem Kontext. "
    "Wenn der Kontext die Frage nicht beantwortet, sage dies in einem Satz und höre auf. "
    "Andernfalls antworte direkt und in höchstens 3 Absätzen. "
    "Beginne sofort mit der Antwort."
)

# HyDE: a short, plausible draft answer used only for re-embedding/retrieval.
SYSTEM_PROMPT_HYDE = (
    "Du bist ein Ernährungsberater. Schreibe eine kurze, plausible Beispiel-Antwort "
    "auf die folgende Frage (3-5 Sätze). Diese hypothetische Antwort dient nur dazu, "
    "passende Quellen zu finden — sachliche Korrektheit ist nicht erforderlich."
)

# Qwen3 emits its reasoning as <think>…</think> at the start of content when
# enable_thinking=true; strip it so the stored answer isn't polluted.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text):
    return _THINK_BLOCK.sub("", text or "").lstrip()


def _retrieval_correction_triggers(scores):
    """Return list of fired triggers (with values), empty if retrieval looks fine."""
    if not scores:
        return ["empty_scores"]
    triggers = []
    lo, hi = min(scores), max(scores)
    if RAG_SC_SCORE_DIRECTION == "lower":
        if lo > RAG_SC_RETRIEVAL_BEST_THRESHOLD:
            triggers.append(f"lowest={lo:.3f}>{RAG_SC_RETRIEVAL_BEST_THRESHOLD}")
    else:
        if hi < RAG_SC_RETRIEVAL_BEST_THRESHOLD:
            triggers.append(f"highest={hi:.3f}<{RAG_SC_RETRIEVAL_BEST_THRESHOLD}")
    spread = hi - lo
    if spread > RAG_SC_RETRIEVAL_SPREAD_THRESHOLD:
        triggers.append(f"spread={spread:.3f}>{RAG_SC_RETRIEVAL_SPREAD_THRESHOLD}")
    return triggers


def _logprob_stats(logprobs):
    """Return {'mean','min','max'} of a non-empty logprob list, or None."""
    if not logprobs:
        return None
    return {
        "mean": sum(logprobs) / len(logprobs),
        "median": statistics.median(logprobs),
        "min": min(logprobs),
        "max": max(logprobs),
    }


def _generation_correction_triggers(logprobs):
    if not logprobs:
        return ["empty_logprobs"]
    stats = _logprob_stats(logprobs)
    mean_lp, min_lp = stats["mean"], stats["min"]
    triggers = []
    if mean_lp < RAG_SC_GEN_MEAN_LOGPROB_THRESHOLD:
        triggers.append(f"mean={mean_lp:.3f}<{RAG_SC_GEN_MEAN_LOGPROB_THRESHOLD}")
    if min_lp < RAG_SC_GEN_MIN_LOGPROB_THRESHOLD:
        triggers.append(f"min={min_lp:.3f}<{RAG_SC_GEN_MIN_LOGPROB_THRESHOLD}")
    return triggers


def _merge_and_rerank(ctx_orig, retr_scores_orig, ctx_hyde, retr_scores_hyde, k=RAG_K):
    """Union both retrievals, dedupe by context text, keep best score per doc,
    sort by direction, truncate to k."""
    pool = {}
    for context, score in list(zip(ctx_orig, retr_scores_orig)) + list(zip(ctx_hyde, retr_scores_hyde)):
        existing = pool.get(context)
        if existing is None:
            pool[context] = score
        elif RAG_SC_SCORE_DIRECTION == "lower":
            pool[context] = min(existing, score)
        else:
            pool[context] = max(existing, score)
    items = sorted(
        pool.items(),
        key=lambda kv_pair: kv_pair[1],
        reverse=(RAG_SC_SCORE_DIRECTION == "higher"),
    )[:k]
    return [c for c, _ in items], [s for _, s in items]


def build_user_prompt(query, contexts):
    if not contexts:
        return f"Frage: {query}"
    lines = [f"Frage: {query}", "", "Kontextinformationen:"]
    for i, c in enumerate(contexts, start=1):
        lines.append(f"{i}. {c}")
    return "\n".join(lines)


def _retrieve(vectorstore, query, k=RAG_K):
    # QUERY_PREFIX is "" for bge-m3 but kept for symmetry with E5/Qwen3.
    docs_with_scores = vectorstore.similarity_search_with_score(
        QUERY_PREFIX + query, k=k
    )
    contexts = [doc.page_content for doc, _ in docs_with_scores]
    scores = [float(score) for _, score in docs_with_scores]
    return contexts, scores


async def _generate(
    session, messages, enable_thinking=LLAMACPP_RAG_ENABLE_THINKING, max_tokens=None
):
    payload = {
        "model": LLAMACPP_RAG_MODEL,
        "messages": messages,
        "temperature": LLAMACPP_RAG_TEMPERATURE,
        "top_p": LLAMACPP_RAG_TOP_P,
        "max_tokens": max_tokens if max_tokens is not None else LLAMACPP_RAG_MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": LLAMACPP_RAG_TOP_LOGPROBS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    answer = ""
    gen_logprobs = []
    prompt_tokens = 0
    gen_tokens = 0
    try:
        async with session.post(
            f"{LLAMACPP_GEN_BASE_URL}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as response:
            parsed = await response.json()
            if "error" in parsed:
                answer = f"[LLAMACPP ERROR] {parsed['error']}"
            else:
                choice = parsed["choices"][0]
                # Recent llama.cpp builds split Qwen3's <think>…</think> out of
                # `content` into a separate `reasoning_content` field. Re-attach
                # it so _strip_thinking can handle both layouts uniformly.
                msg = choice["message"]
                reasoning = msg.get("reasoning_content") or ""
                content = msg.get("content") or ""
                answer = f"<think>{reasoning}</think>{content}" if reasoning else content
                lp = choice.get("logprobs") or {}
                content = lp.get("content") or []
                gen_logprobs = [
                    tok["logprob"]
                    for tok in content
                    if tok.get("logprob") is not None
                ]
                usage = parsed.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                gen_tokens = usage.get("completion_tokens", 0) or 0
    except aiohttp.ClientError as exc:
        answer = f"[LLAMACPP HTTP ERROR] {exc}"
    except Exception as exc:
        answer = f"[LLAMACPP ERROR] {exc}"
    return answer, gen_logprobs, prompt_tokens, gen_tokens


async def _generate_hyde(session, sem, query):
    """Draft a short hypothetical answer used to seed a second retrieval pass."""
    payload = {
        "model": LLAMACPP_RAG_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_HYDE},
            {"role": "user", "content": query},
        ],
        "temperature": LLAMACPP_RAG_TEMPERATURE,
        "top_p": LLAMACPP_RAG_TOP_P,
        "max_tokens": RAG_SC_HYDE_MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem:
        try:
            async with session.post(
                f"{LLAMACPP_GEN_BASE_URL}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                parsed = await response.json()
                if "error" in parsed:
                    return ""
                return parsed["choices"][0]["message"].get("content") or ""
        except Exception:
            return ""


async def process_single_query(
    session, query, pipeline_id, metadata, variant, sem, total, progress_counter, start_time
):
    loop = asyncio.get_event_loop()
    contexts = []
    retrieval_scores = []
    sc_metadata = None  # only populated for variant == "rag_sc"

    if variant in _RAG_VARIANTS:
        vectorstore = _get_vectorstore()
        try:
            contexts, retrieval_scores = await loop.run_in_executor(
                None, lambda: _retrieve(vectorstore, query)
            )
        except Exception as exc:
            return {
                "pipeline_id": pipeline_id,
                **(metadata or {}),
                "query": query,
                "variant": variant,
                "answer": f"[RETRIEVAL ERROR] {exc}",
                "contexts": [],
                "retrieval_scores": [],
                "gen_logprob_stats": None,
            }

    # --- SC retrieval pass (budget=1) ----------------------------------------
    if variant == "rag_sc":
        sc_metadata = {
            "retrieval_correction_triggers": _retrieval_correction_triggers(retrieval_scores),
            "retrieval_retried_count": 0,
            "generation_correction_triggers": [],
            "generation_retried_count": 0,
        }
        if sc_metadata["retrieval_correction_triggers"]:
            hyde_text = await _generate_hyde(session, sem, query)
            if hyde_text.strip():
                try:
                    hyde_ctx, hyde_retrieval_scores = await loop.run_in_executor(
                        None, lambda: _retrieve(vectorstore, hyde_text)
                    )
                    sc_metadata["hyde_retrieval_fake_answer"] = hyde_text
                    sc_metadata["original_contexts"] = list(contexts)
                    sc_metadata["original_retrieval_scores"] = list(retrieval_scores)
                    contexts, retrieval_scores = _merge_and_rerank(
                        contexts, retrieval_scores, hyde_ctx, hyde_retrieval_scores
                    )
                    sc_metadata["retrieval_retried_count"] += 1
                except Exception as exc:
                    sc_metadata["retrieval_error"] = str(exc)

    system_prompt = SYSTEM_PROMPT_RAG if variant in _RAG_VARIANTS else SYSTEM_PROMPT_NO_RAG
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(query, contexts)},
    ]

    async with sem:
        task_start = time.time()
        answer, gen_logprobs, prompt_tokens, gen_tokens = await _generate(
            session, messages
        )

        # --- SC generation pass (budget=1) -----------------------------------
        if sc_metadata is not None:
            sc_metadata["generation_correction_triggers"] = _generation_correction_triggers(
                gen_logprobs
            )
            if sc_metadata["generation_correction_triggers"]:
                sc_metadata["original_answer"] = answer
                sc_metadata["original_gen_logprob_stats"] = _logprob_stats(gen_logprobs)
                strict_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_RAG_STRICT},
                    {"role": "user", "content": build_user_prompt(query, contexts)},
                ]
                # Thinking is disabled on regen: on Qwen3.5-4B-Q4 the reasoning
                # trace consistently loops on self-doubt ("Wait, let me check
                # again…") and either burns the full budget or leaves an empty
                # post-think answer. The stricter system prompt alone is what
                # actually improves the regen output.
                regen_answer, regen_lp, p2, g2 = await _generate(
                    session,
                    strict_messages,
                    enable_thinking=False,
                )
                answer = regen_answer
                gen_logprobs = regen_lp
                prompt_tokens += p2
                gen_tokens += g2
                sc_metadata["generation_retried_count"] += 1

        task_time = time.time() - task_start

        progress_counter["prompt_tokens"] += prompt_tokens
        progress_counter["gen_tokens"] += gen_tokens
        progress_counter["done"] += 1
        done = progress_counter["done"]
        elapsed = time.time() - start_time
        rate = elapsed / done if done > 0 else 0
        remaining = (total - done) * rate

        sc_tag = ""
        if sc_metadata is not None:
            sc_tag = (
                f" sc(ret={sc_metadata['retrieval_retried_count']},"
                f"gen={sc_metadata['generation_retried_count']})"
            )

        print(
            f"  [{done:>{len(str(total))}}/{total}]  "
            f"Q#{pipeline_id + 1} [{variant}]{sc_tag} done in {task_time:.1f}s  "
            f"prompt={prompt_tokens} gen={gen_tokens}  "
            f"Rate: {rate:.2f} s/task  ETA: {remaining:.1f}s",
            flush=True,
        )

    result = {
        "pipeline_id": pipeline_id,
        # Spread the dataset-supplied metadata (id, summary, difficulty, …) at
        # top level. Pipeline-owned keys below override any colliding names.
        **(metadata or {}),
        "query": query,
        "variant": variant,
        "answer": answer,
        "contexts": contexts,
        "retrieval_scores": retrieval_scores,
        "gen_logprob_stats": _logprob_stats(gen_logprobs),
    }
    if sc_metadata is not None:
        result["sc_metadata"] = sc_metadata
    return result


async def run_rag_pipeline_async(indexed_items):
    # indexed_items: iterable of (pipeline_id, query_text, metadata_dict).
    # The caller hands out *global* pipeline ids so they survive shard slicing
    # and can be re-merged in order. `metadata` is opaque to the pipeline; its
    # keys are spread into each result row at top level.
    indexed_items = list(indexed_items)
    _get_vectorstore()  # warm before fan-out so all tasks share one instance
    sem = asyncio.Semaphore(LLAMACPP_RAG_CONCURRENCY)
    start_time = time.time()
    total_tasks = len(indexed_items) * len(VARIANTS)
    progress_counter = {"done": 0, "prompt_tokens": 0, "gen_tokens": 0}

    print(f"\n{'=' * 80}")
    print(
        f"Starting RAG batch: {len(indexed_items)} queries × {len(VARIANTS)} variants "
        f"= {total_tasks} tasks  (concurrency={LLAMACPP_RAG_CONCURRENCY})"
    )
    print(f"Variants: {VARIANTS}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n", flush=True)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_single_query(
                session, q, pid, meta, v, sem, total_tasks, progress_counter, start_time
            )
            for pid, q, meta in indexed_items
            for v in VARIANTS
        ]
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    prompt_tokens = progress_counter["prompt_tokens"]
    gen_tokens = progress_counter["gen_tokens"]
    wall_gen_rate = gen_tokens / total_time if total_time > 0 else 0.0

    print(f"{'=' * 80}")
    print(f"Done — total: {total_time:.1f}s ({total_time / 60:.1f}m)")
    print(
        f"Tokens — prompt: {prompt_tokens}, gen: {gen_tokens}  "
        f"({wall_gen_rate:.1f} tok/s wall-clock gen)"
    )
    print(f"{'=' * 80}\n", flush=True)

    return results
