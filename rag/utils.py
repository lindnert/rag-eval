import asyncio
import time
from datetime import datetime

import aiohttp

from rag.llm_config import (
    LLAMACPP_GEN_BASE_URL,
    LLAMACPP_RAG_CONCURRENCY,
    LLAMACPP_RAG_MAX_TOKENS,
    LLAMACPP_RAG_MODEL,
    LLAMACPP_RAG_TEMPERATURE,
    LLAMACPP_RAG_TOP_LOGPROBS,
    LLAMACPP_RAG_TOP_P,
    RAG_K,
)
from retrieval import QUERY_PREFIX, build_vectorstore

_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore


SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Ernährungsberater, der evidenzbasierte Empfehlungen gibt. "
    "Antworte kurz und prägnant (max 3-4 Absätze). Verwende nur Informationen aus dem Kontext."
)


def build_user_prompt(query, contexts):
    lines = [f"Frage: {query}"]
    if contexts:
        lines.append("")
        lines.append("Kontextinformationen:")
        for i, c in enumerate(contexts, start=1):
            lines.append(f"{i}. {c}")
    else:
        lines.append("")
        lines.append("Hinweis: Kein Kontext verfügbar.")
    return "\n".join(lines)


def _retrieve(vectorstore, query, k=RAG_K):
    # QUERY_PREFIX is "" for bge-m3 but kept for symmetry with E5/Qwen3.
    docs_with_scores = vectorstore.similarity_search_with_score(
        QUERY_PREFIX + query, k=k
    )
    contexts = [doc.page_content for doc, _ in docs_with_scores]
    scores = [float(score) for _, score in docs_with_scores]
    return contexts, scores


async def process_single_query(
    session, query, sem, idx, total, progress_counter, start_time
):
    loop = asyncio.get_event_loop()
    vectorstore = _get_vectorstore()

    try:
        contexts, retrieval_scores = await loop.run_in_executor(
            None, lambda: _retrieve(vectorstore, query)
        )
    except Exception as exc:
        return {
            "query": query,
            "answer": f"[RETRIEVAL ERROR] {exc}",
            "contexts": [],
            "retrieval_scores": [],
            "gen_logprobs": [],
            "variant": "rag",
        }

    async with sem:
        query_start = time.time()
        payload = {
            "model": LLAMACPP_RAG_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, contexts)},
            ],
            "temperature": LLAMACPP_RAG_TEMPERATURE,
            "top_p": LLAMACPP_RAG_TOP_P,
            "max_tokens": LLAMACPP_RAG_MAX_TOKENS,
            "logprobs": True,
            "top_logprobs": LLAMACPP_RAG_TOP_LOGPROBS,
            "stream": False,
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
                    answer = choice["message"]["content"]
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

        query_time = time.time() - query_start
        progress_counter["prompt_tokens"] += prompt_tokens
        progress_counter["gen_tokens"] += gen_tokens
        progress_counter["done"] += 1
        done = progress_counter["done"]
        elapsed = time.time() - start_time
        rate = elapsed / done if done > 0 else 0
        remaining = (total - done) * rate

        print(
            f"  [{done:>{len(str(total))}}/{total}]  "
            f"Query #{idx + 1} done in {query_time:.1f}s  "
            f"prompt={prompt_tokens} gen={gen_tokens}  "
            f"Rate: {rate:.2f} s/q  ETA: {remaining:.1f}s",
            flush=True,
        )

    return {
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "retrieval_scores": retrieval_scores,
        "gen_logprobs": gen_logprobs,
        "variant": "rag",
    }


async def run_rag_pipeline_async(queries):
    _get_vectorstore()  # warm before fan-out so all tasks share one instance
    sem = asyncio.Semaphore(LLAMACPP_RAG_CONCURRENCY)
    start_time = time.time()
    total_queries = len(queries)
    progress_counter = {"done": 0, "prompt_tokens": 0, "gen_tokens": 0}

    print(f"\n{'=' * 80}")
    print(
        f"Starting RAG batch: {total_queries} queries "
        f"(concurrency={LLAMACPP_RAG_CONCURRENCY})"
    )
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n", flush=True)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_single_query(
                session, q, sem, i, total_queries, progress_counter, start_time
            )
            for i, q in enumerate(queries)
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
