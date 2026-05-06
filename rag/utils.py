import os
import aiohttp
import asyncio
import time
from datetime import datetime
import requests

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_RAG_MODEL = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e2b")
OLLAMA_RAG_MODEL_TEMPERATURE = float(os.getenv("OLLAMA_RAG_MODEL_TEMPERATURE", "0.1"))
OLLAMA_RAG_MODEL_TOP_P = float(os.getenv("OLLAMA_RAG_MODEL_TOP_P", "0.95"))
OLLAMA_RAG_MODEL_TOP_K = int(os.getenv("OLLAMA_RAG_MODEL_TOP_K", "64"))
OLLAMA_RAG_MODEL_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", 8))
OLLAMA_EVAL_MODEL = os.getenv("OLLAMA_EVAL_MODEL", "qwen3.5:2b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_CONTEXT_LENGTH = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "8192"))
FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "richtlinien", "faiss_index")

_retriever = None


def _get_retriever(index_dir=FAISS_INDEX_DIR, k=3):
    global _retriever
    if _retriever is None:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        _retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return _retriever


def build_prompt(query, contexts):
    prompt_lines = [
        "Du bist ein hilfreicher Ernährungsberater, der evidenzbasierte Empfehlungen gibt.",
        "Antworte kurz und prägnant (max 3-4 Absätze). Verwende nur Informationen aus dem Kontext.",
        "",
        f"Frage: {query}",
    ]

    if contexts:
        prompt_lines.append("")
        prompt_lines.append("Kontextinformationen:")
        for index, context in enumerate(contexts, start=1):
            prompt_lines.append(f"{index}. {context}")
    else:
        prompt_lines.append("")
        prompt_lines.append("Hinweis: Kein Kontext verfügbar.")
        
    prompt_lines.append("")
    prompt_lines.append("Antwort:")

    return "\n".join(prompt_lines)


def parse_ollama_response(response_json):
    if "response" in response_json:
        return response_json["response"]
    if "error" in response_json:
        raise ValueError(f"Ollama error: {response_json['error']}")
    raise ValueError(f"Unexpected Ollama response shape: {response_json}")

# outdated function, not used in current pipeline but kept for reference
"""async def generate_llm_answer_async(session, query, contexts):
    prompt = build_prompt(query, contexts)
    payload = {
        "model": OLLAMA_RAG_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": OLLAMA_CONTEXT_LENGTH,
            "temperature": OLLAMA_RAG_MODEL_TEMPERATURE,
            "top_p": OLLAMA_RAG_MODEL_TOP_P,
            "top_k": OLLAMA_RAG_MODEL_TOP_K,
        }
    }

    try:
        async with session.post(OLLAMA_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as response:
            parsed = await response.json()
            return parse_ollama_response(parsed)
    except aiohttp.ClientError as exc:
        return f"[OLLAMA HTTP ERROR] {exc}"
    except Exception as exc:
        return f"[OLLAMA ERROR] {exc}"

"""

# outdated function, not used in current pipeline but kept for reference
""" def run_rag_pipeline(query):
    retriever = _get_retriever()
    retrieved_docs = [doc.page_content for doc in retriever.invoke(query)]

    answer = generate_llm_answer(query, retrieved_docs)

    return {
        "query": query,
        "answer": answer,
        "contexts": retrieved_docs,
    }
"""
async def process_single_query(session, query, retriever, sem, idx, total, progress_counter, start_time):
    loop = asyncio.get_event_loop()

    # Run blocking retrieval in a thread so it doesn't block the event loop
    try:
        retrieved_docs = await loop.run_in_executor(
            None,
            lambda: [doc.page_content for doc in retriever.invoke(query)]
        )
    except Exception as exc:
        return {"query": query, "answer": f"[RETRIEVAL ERROR] {exc}", "contexts": []}

    # Semaphore limits how many LLM requests are in-flight at once
    async with sem:
        query_start = time.time()

        prompt = build_prompt(query, retrieved_docs)
        payload = {
        "model": OLLAMA_RAG_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": OLLAMA_CONTEXT_LENGTH,
            "temperature": OLLAMA_RAG_MODEL_TEMPERATURE,
            "top_p": OLLAMA_RAG_MODEL_TOP_P,
            "top_k": OLLAMA_RAG_MODEL_TOP_K,
            }
        }
        prompt_tokens = 0
        gen_tokens = 0
        prompt_eval_duration_s = 0.0
        eval_duration_s = 0.0
        try:
            async with session.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                parsed = await response.json()
                answer = parse_ollama_response(parsed)
                prompt_tokens = parsed.get("prompt_eval_count", 0) or 0
                gen_tokens = parsed.get("eval_count", 0) or 0
                prompt_eval_duration_s = (parsed.get("prompt_eval_duration", 0) or 0) / 1e9
                eval_duration_s = (parsed.get("eval_duration", 0) or 0) / 1e9
        except aiohttp.ClientError as exc:
            answer = f"[OLLAMA HTTP ERROR] {exc}"
        except Exception as exc:
            answer = f"[OLLAMA ERROR] {exc}"

        query_time = time.time() - query_start

        progress_counter["prompt_tokens"] += prompt_tokens
        progress_counter["gen_tokens"] += gen_tokens
        progress_counter["prompt_eval_duration_s"] += prompt_eval_duration_s
        progress_counter["eval_duration_s"] += eval_duration_s

        gen_rate = gen_tokens / eval_duration_s if eval_duration_s > 0 else 0.0
        print(
            f"    Query #{idx+1} tokens — prompt: {prompt_tokens}, gen: {gen_tokens}  "
            f"({gen_rate:.1f} tok/s gen)",
            flush=True,
        )

        progress_counter["done"] += 1
        done = progress_counter["done"]
        elapsed = time.time() - start_time
        rate = elapsed / done if done > 0 else 0
        remaining = (total - done) * rate

        print(
            f"  [{done:>{len(str(total))}}/{total}]  "
            f"Query #{idx+1} answered in {query_time:.1f}s  Rate: {rate:.2f} s/q |  "
            f"Total time elapsed: {elapsed:.1f}s  ETA: {remaining:.1f}s",
            flush=True,
        )

        print(requests.get("http://127.0.0.1:11434/api/ps").json(), flush=True)

    return {"query": query, "answer": answer, "contexts": retrieved_docs}

async def run_rag_pipeline_async(queries):
    retriever = _get_retriever()
    sem = asyncio.Semaphore(OLLAMA_RAG_MODEL_CONCURRENCY)

    start_time = time.time()
    total_queries = len(queries)
    progress_counter = {
        "done": 0,
        "prompt_tokens": 0,
        "gen_tokens": 0,
        "prompt_eval_duration_s": 0.0,
        "eval_duration_s": 0.0,
    }

    print(f"\n{'='*80}")
    print(f"Starting batch processing: {total_queries} queries with batch_size={OLLAMA_RAG_MODEL_CONCURRENCY}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n", flush=True)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_single_query(session, q, retriever, sem, i, total_queries, progress_counter, start_time)
            for i, q in enumerate(queries)
            ]
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    prompt_tokens = progress_counter["prompt_tokens"]
    gen_tokens = progress_counter["gen_tokens"]
    total_tokens = prompt_tokens + gen_tokens
    prompt_eval_s = progress_counter["prompt_eval_duration_s"]
    eval_s = progress_counter["eval_duration_s"]

    # Per-model rates (sum of model-time, ignores concurrency overlap)
    model_prompt_rate = prompt_tokens / prompt_eval_s if prompt_eval_s > 0 else 0.0
    model_gen_rate = gen_tokens / eval_s if eval_s > 0 else 0.0
    # Wall-clock throughput (reflects concurrency benefit)
    wall_gen_rate = gen_tokens / total_time if total_time > 0 else 0.0
    wall_total_rate = total_tokens / total_time if total_time > 0 else 0.0

    print(f"{'='*80}")
    print(f"Processing completed!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Total queries: {total_queries}")
    print(f"Tokens — prompt: {prompt_tokens}, generated: {gen_tokens}, total: {total_tokens}")
    print(f"Per-model rate (single-stream): prompt {model_prompt_rate:.1f} tok/s | gen {model_gen_rate:.1f} tok/s")
    print(f"Wall-clock throughput (with concurrency): gen {wall_gen_rate:.1f} tok/s | total {wall_total_rate:.1f} tok/s")
    print(f"{'='*80}\n", flush=True)

    return results