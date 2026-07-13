import asyncio
import json
import re
import statistics
import time
from datetime import datetime

import aiohttp

from rag.llm_config import (
    LLAMACPP_GEN_API_KEY,
    LLAMACPP_GEN_BASE_URL,
    LLAMACPP_RAG_CONCURRENCY,
    LLAMACPP_RAG_ENABLE_THINKING,
    LLAMACPP_RAG_MAX_TOKENS,
    LLAMACPP_RAG_MODEL,
    LLAMACPP_RAG_TEMPERATURE,
    LLAMACPP_RAG_TOP_LOGPROBS,
    LLAMACPP_RAG_TOP_P,
    RAG_HYBRID_ALPHA,
    RAG_K,
    RAG_SC_GEN_MEAN_LOGPROB_THRESHOLD,
    RAG_SC_GEN_MIN_LOGPROB_THRESHOLD,
    RAG_SC_HYDE_MAX_TOKENS,
    RAG_SC_REGEN_MAX_TOKENS,
    RAG_SC_RETRIEVAL_BEST_THRESHOLD,
    RAG_SC_RETRIEVAL_SPREAD_THRESHOLD,
    RAG_SC_SCORE_DIRECTION,
)
from common.constants import RAG_LANG, REJECTION_ANSWER

VARIANTS = ("no_rag", "rag", "rag_sc")
_RAG_VARIANTS = {"rag", "rag_sc"}

# Bearer auth for the generation endpoint, added to every chat-completions POST.
# Empty when LLAMACPP_GEN_API_KEY is unset (local llama-server) → no header sent;
# populated when pointing at an authenticated node (e.g. Ollama). Built once so
# both _generate and _generate_hyde share it.
_GEN_HEADERS = (
    {"Authorization": f"Bearer {LLAMACPP_GEN_API_KEY}"} if LLAMACPP_GEN_API_KEY else {}
)

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        # Imported lazily (not at module top) so the generation helpers in this
        # module can be imported without the retrieval stack (rank_bm25, faiss,
        # …) installed — e.g. by test_single_query.py, which injects fixed
        # contexts and never retrieves. Any real rag/rag_sc run still calls this
        # during warm-up, so the retriever is built exactly as before.
        from retrieval import build_hybrid_retriever

        _retriever = build_hybrid_retriever(RAG_HYBRID_ALPHA)
    return _retriever

# --- Language-dependent prompt bundle --------------------------------------
# All model-facing prompt text is selected by RAG_LANG (see common.constants) so
# a single `export RAG_LANG=de` flips the system prompts, the user-prompt labels,
# and the abstention string coherently for a run. German runs answer the
# (English) dataset queries in German. The field labels 'Food information' and
# 'User profile' stay English in every bundle because they appear verbatim in the
# NGQA query text (see dataset/NGQA/NGQA.py) — the prompt points the model at the
# labels as they actually occur, not a translation of them.


def _en_bundle():
    rejection = (
        "If neither the question nor the context provides the information needed to "
        "answer — not even partially — respond with exactly this sentence and add "
        "nothing else: "
        f"\"{REJECTION_ANSWER}\" "
    )
    sources = (
        "You have two sources of information: the question (including for example "
        "'Food information' and 'User profile'), which may itself state "
        "relevant facts, and the section labelled \"Context\", which holds retrieved "
        "passages. Use both together. "
    )
    return {
        "no_rag": (
            "You are a helpful nutrition advisor who gives evidence-based recommendations. "
            "Answer briefly and concisely."
        ),
        "rag": (
            "You are a helpful nutrition advisor who gives evidence-based recommendations. "
            + sources
            + "If relevant information is available — even if it only partially covers the "
            "question — answer briefly and concisely, grounded in that material. "
            + rejection
        ),
        "rag_strict": (
            "You are a helpful nutrition advisor who gives evidence-based recommendations. "
            + sources
            + "If relevant information is available — even if it only partially covers the "
            "question — answer briefly and concisely. State only what the question and the "
            "context support, and make no claim they do not back up; if they support only "
            "part of an answer, give that part and leave the rest unanswered. "
            + rejection
            + "Judge once whether the information suffices and what your answer is, then "
            "state it directly. Do not reconsider or re-weigh — your first reasoned "
            "conclusion is final. When the evidence is mixed, name the trade-off in one "
            "sentence and stop. "
        ),
        # HyDE: a short, plausible draft answer used only for re-embedding/retrieval.
        "hyde": (
            "You are a nutrition advisor. Write a short, plausible example answer to the "
            "following question (3-5 sentences). Factual correctness is not required."
        ),
        "question_label": "Question",
        "context_label": "Context",
    }


def _de_bundle():
    rejection = (
        "Wenn weder die Frage noch der Kontext die zur Beantwortung nötigen "
        "Informationen liefert — nicht einmal teilweise —, antworte mit exakt "
        "diesem Satz und füge nichts hinzu: "
        f"\"{REJECTION_ANSWER}\" "
    )
    sources = (
        "Dir stehen zwei Informationsquellen zur Verfügung: die Frage "
        "(einschließlich zum Beispiel 'Food information' und 'User profile'), die "
        "selbst relevante Fakten enthalten kann, und der Abschnitt mit der "
        "Bezeichnung \"Kontext\", der die abgerufenen Passagen enthält. Nutze beide "
        "zusammen. "
    )
    return {
        "no_rag": (
            "Du bist ein hilfreicher Ernährungsberater, der evidenzbasierte Empfehlungen gibt. "
            "Antworte kurz und prägnant."
        ),
        "rag": (
            "Du bist ein hilfreicher Ernährungsberater, der evidenzbasierte Empfehlungen gibt. "
            + sources
            + "Wenn relevante Informationen verfügbar sind — auch wenn sie die Frage nur "
            "teilweise abdecken —, antworte kurz und prägnant auf Grundlage dieses "
            "Materials. "
            + rejection
        ),
        "rag_strict": (
            "Du bist ein hilfreicher Ernährungsberater, der evidenzbasierte Empfehlungen gibt. "
            + sources
            + "Wenn relevante Informationen verfügbar sind — auch wenn sie die Frage nur "
            "teilweise abdecken —, antworte kurz und prägnant. Nenne nur, was die Frage und "
            "der Kontext stützen, und triff keine Aussage, die sie nicht belegen; wenn sie "
            "nur einen Teil einer Antwort stützen, gib diesen Teil und lasse den Rest offen. "
            + rejection
            + "Entscheide einmal, ob die Informationen ausreichen und wie deine Antwort "
            "lautet, und formuliere sie dann direkt. Überdenke oder gewichte nichts noch "
            "einmal — deine erste begründete Schlussfolgerung ist endgültig. Wenn die Belege "
            "gemischt sind, benenne den Zielkonflikt in einem Satz und höre auf. "
        ),
        # HyDE: a short, plausible draft answer used only for re-embedding/retrieval.
        "hyde": (
            "Du bist ein Ernährungsberater. Schreibe eine kurze, plausible Beispiel-Antwort "
            "auf die folgende Frage (3-5 Sätze). Sachliche Korrektheit ist nicht erforderlich."
        ),
        "question_label": "Frage",
        "context_label": "Kontext",
    }


_BUNDLES = {"en": _en_bundle, "de": _de_bundle}
_bundle = _BUNDLES[RAG_LANG]()

SYSTEM_PROMPT_RAG = _bundle["rag"]
SYSTEM_PROMPT_NO_RAG = _bundle["no_rag"]
SYSTEM_PROMPT_RAG_STRICT = _bundle["rag_strict"]
SYSTEM_PROMPT_HYDE = _bundle["hyde"]
_QUESTION_LABEL = _bundle["question_label"]
_CONTEXT_LABEL = _bundle["context_label"]

# Qwen3 emits its reasoning as <think>…</think> at the start of content when
# enable_thinking=true; strip it so the stored answer isn't polluted.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text):
    return _THINK_BLOCK.sub("", text or "").lstrip()


# Captures the *inner* reasoning of <think>…</think> blocks (vs _THINK_BLOCK,
# which strips them). Used to surface the rag_sc regen's reasoning trace so the
# thinking-looped (empty-answer) case is inspectable without re-running.
_THINK_CAPTURE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_thinking(text):
    if not text:
        return ""
    return "\n".join(m.strip() for m in _THINK_CAPTURE.findall(text)).strip()


# --- Conservative abstention detection -------------------------------------
# The model is told to emit REJECTION_ANSWER verbatim, but in practice it
# paraphrases by a word or two ("The retrieved context …" vs "The provided
# context …") or appends an explanation after the rejection sentence. We still want
# those counted as one rejection outcome, without misclassifying a substantive
# answer that merely notes the context is partly insufficient somewhere in the
# middle. So we look only at the *first sentence* and accept it only if it is a
# near-exact (≤2 word edits) match of REJECTION_ANSWER.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _words(text):
    return _WORD_RE.findall((text or "").lower())


def _word_levenshtein(a, b):
    """Token-level edit distance between two word lists."""
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,                 # deletion
                cur[j - 1] + 1,              # insertion
                prev[j - 1] + (wa != wb),    # substitution
            ))
        prev = cur
    return prev[-1]


_REJECTION_WORDS = _words(REJECTION_ANSWER)

# Anchor tokens that a near-rejection first sentence must contain: a negation
# and an "information" noun. Unioned across EN and DE so one code path serves
# both language bundles — the language-specific edit-distance gate below already
# forces a near-exact match to the active REJECTION_ANSWER, so the anchors are
# only a cheap guard against an affirmative near-miss ("… contains enough
# information …") and don't need to be language-partitioned. ("keine" is the
# negation in the German rejection sentence, not "nicht".)
_NEGATION_ANCHORS = {"not", "nicht", "keine"}
_INFORMATION_ANCHORS = {"information", "informationen"}


def _is_near_rejection(stripped, max_word_diff=2):
    """True if the answer *opens* with (a near-paraphrase of) REJECTION_ANSWER.

    Conservative by design — matches only when the first sentence is within
    ``max_word_diff`` word-level edits of REJECTION_ANSWER, so it catches:
      - exact emissions (distance 0),
      - one/two-word paraphrases ("The retrieved context …"),
      - REJECTION_ANSWER followed by an appended explanation (first sentence
        is the rejection),
    but not an answer that merely mentions mid-text that the context is
    incomplete. The negation/information anchor tokens guard against an
    affirmative near-miss ("… contains enough information …") matching on edit
    distance alone.
    """
    first = _SENTENCE_SPLIT.split(stripped, maxsplit=1)[0]
    words = _words(first)
    word_set = set(words)
    if not (word_set & _NEGATION_ANCHORS) or not (word_set & _INFORMATION_ANCHORS):
        return False
    return _word_levenshtein(words, _REJECTION_WORDS) <= max_word_diff


def _finalize_answer(answer, *, is_rag):
    """Map a raw completion onto either a clean answer or the canonical
    rejection. Returns (final_answer, rejection_reason | None).

    Both RAG variants share the same abstention string so rejection is one
    countable outcome, via two paths:

      - "model_rejected": the model abstained — its answer *opens* with the
        canonical REJECTION_ANSWER, allowing a one/two-word paraphrase and any
        appended explanation (see _is_near_rejection). This is the only
        rejection signal the thinking-off `rag` baseline has.
      - "empty": no answer text survives after the <think> block is stripped.
        This is what a runaway rag_sc regen looks like — the loop spends the
        whole budget *thinking* and never emits a real text response. Note we
        key on empty content, NOT on truncation: a generation that hit
        max_tokens but did write an answer is a long answer, not an abstention.

    no_rag has no context to be "insufficient", so it never abstains here.
    """
    stripped = _strip_thinking(answer).strip()
    if not is_rag:
        return stripped, None
    if not stripped:
        return REJECTION_ANSWER, "empty"
    if _is_near_rejection(stripped):
        return REJECTION_ANSWER, "model_rejected"
    return stripped, None


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
        return f"{_QUESTION_LABEL}: {query}"
    lines = [f"{_QUESTION_LABEL}: {query}", "", f"{_CONTEXT_LABEL}:"]
    for i, c in enumerate(contexts, start=1):
        lines.append(f"{i}. {c}")
    return "\n".join(lines)


def _retrieve(retriever, query, k=RAG_K):
    # The hybrid retriever handles QUERY_PREFIX and dense+BM25 fusion internally.
    docs_with_scores = retriever.search_with_score(query, k=k)
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
    finish_reason = None
    try:
        async with session.post(
            f"{LLAMACPP_GEN_BASE_URL}/chat/completions",
            json=payload,
            headers=_GEN_HEADERS,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as response:
            # Force UTF-8. aiohttp's response.json() sniffs the charset from the
            # Content-Type header; when llama.cpp omits one it can guess Latin-1
            # and mojibake UTF-8 German (ü -> Ã¼). json.loads on raw bytes always
            # decodes UTF-8 per RFC 8259.
            parsed = json.loads(await response.read())
            if "error" in parsed:
                answer = f"[LLAMACPP ERROR] {parsed['error']}"
            else:
                choice = parsed["choices"][0]
                # "length" => the model hit max_tokens without emitting a stop
                # token, i.e. it kept reasoning/searching and never converged.
                finish_reason = choice.get("finish_reason")
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
    return answer, gen_logprobs, prompt_tokens, gen_tokens, finish_reason


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
                headers=_GEN_HEADERS,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                # Force UTF-8 (see note on the generation call above).
                parsed = json.loads(await response.read())
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
        retriever = _get_retriever()
        try:
            contexts, retrieval_scores = await loop.run_in_executor(
                None, lambda: _retrieve(retriever, query)
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
                        None, lambda: _retrieve(retriever, hyde_text)
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
        answer, gen_logprobs, prompt_tokens, gen_tokens, finish_reason = await _generate(
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
                # Thinking is re-enabled on the regen so the model can actually
                # reason over the context, with a larger budget so a genuine
                # trace+answer finishes. When Qwen3.5-4B-Q4 instead loops on
                # self-doubt ("Wait, let me check again…") it burns the whole
                # budget thinking and emits no answer text — _finalize_answer
                # catches that empty result and substitutes REJECTION_ANSWER.
                regen_answer, regen_lp, p2, g2, finish_reason = await _generate(
                    session,
                    strict_messages,
                    enable_thinking=True,
                    max_tokens=RAG_SC_REGEN_MAX_TOKENS,
                )
                # Surface the regen's reasoning trace so the thinking-looped
                # (empty-answer) case is inspectable without re-running.
                sc_metadata["regen_thinking"] = _extract_thinking(regen_answer)
                answer = regen_answer
                gen_logprobs = regen_lp
                prompt_tokens += p2
                gen_tokens += g2
                sc_metadata["generation_retried_count"] += 1

        # Normalise the final completion: an empty (thinking-looped) generation
        # and the model's own abstention both collapse onto the exact
        # REJECTION_ANSWER so rejection is a single countable outcome downstream.
        answer, rejection_reason = _finalize_answer(
            answer, is_rag=variant in _RAG_VARIANTS
        )
        if sc_metadata is not None:
            # finish_reason of whatever produced the final answer (regen if it
            # ran, else the first pass). An "empty" rejection paired with
            # "length" here is the regen-looped-on-thinking case.
            sc_metadata["finish_reason"] = finish_reason
            if rejection_reason is not None:
                sc_metadata["rejection_reason"] = rejection_reason

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
        "rejected": answer == REJECTION_ANSWER,
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
    _get_retriever()  # warm before fan-out so all tasks share one instance
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
