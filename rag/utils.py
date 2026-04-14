import json
import os
import urllib.request
import urllib.error
import aiohttp
import asyncio
import time
from datetime import datetime

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_RAG_MODEL = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e2b")
OLLAMA_EVAL_MODEL = os.getenv("OLLAMA_EVAL_MODEL", "qwen3.5:2b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_CONTEXT_LENGTH = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32000"))
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
    return response_json["response"]


def generate_llm_answer(query, contexts):
    prompt = build_prompt(query, contexts)
    payload = {
        "model": OLLAMA_RAG_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
        "num_ctx": OLLAMA_CONTEXT_LENGTH
    }
    }
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            return parse_ollama_response(parsed)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        return f"[OLLAMA HTTP ERROR] {exc.code}: {error_body}"
    except Exception as exc:
        return f"[OLLAMA ERROR] {exc}"


async def generate_llm_answer_async(session, query, contexts):
    prompt = build_prompt(query, contexts)
    payload = {
        "model": OLLAMA_RAG_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": OLLAMA_CONTEXT_LENGTH
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


def run_rag_pipeline(query):
    retriever = _get_retriever()
    retrieved_docs = [doc.page_content for doc in retriever.invoke(query)]

    answer = generate_llm_answer(query, retrieved_docs)

    return {
        "query": query,
        "answer": answer,
        "contexts": retrieved_docs,
    }


async def run_rag_pipeline_batch_async(queries, batch_size=10):
    """
    Process multiple queries asynchronously in batches.

    Args:
        queries: List of query strings
        batch_size: Number of concurrent requests (3 recommended for T4 GPU)

    Returns:
        List of results, one per query
    """
    retriever = _get_retriever()
    results = []
    start_time = time.time()
    total_queries = len(queries)
    processed_count = 0

    print(f"\n{'='*80}")
    print(f"Starting batch processing: {total_queries} queries with batch_size={batch_size}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n", flush=True)

    async with aiohttp.ClientSession() as session:
        for batch_num, i in enumerate(range(0, len(queries), batch_size), 1):
            batch = queries[i:i + batch_size]
            batch_start = time.time()
            tasks = []

            print(f"[Batch {batch_num}] Processing {len(batch)} queries ({i+1}-{min(i+len(batch), total_queries)}/{total_queries})...", flush=True)

            for query_idx, query in enumerate(batch, 1):
                retrieved_docs = [doc.page_content for doc in retriever.invoke(query)]
                task = generate_llm_answer_async(session, query, retrieved_docs)
                tasks.append((query, retrieved_docs, task))

            answers = await asyncio.gather(*[t[2] for t in tasks])

            for (query, contexts, _), answer in zip(tasks, answers):
                processed_count += 1
                results.append({
                    "query": query,
                    "answer": answer,
                    "contexts": contexts,
                })

            batch_time = time.time() - batch_start
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            remaining = (total_queries - processed_count) / rate if rate > 0 else 0

            print(f"  ✓ Batch {batch_num} completed in {batch_time:.1f}s")
            print(f"  Progress: {processed_count}/{total_queries} queries ({processed_count*100//total_queries}%)")
            print(f"  Elapsed: {elapsed:.1f}s | Rate: {rate:.2f} queries/s | ETA: {remaining:.1f}s\n", flush=True)

    total_time = time.time() - start_time
    print(f"{'='*80}")
    print(f"Batch processing completed!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n", flush=True)

    return results