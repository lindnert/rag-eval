import json
import os
import urllib.request
import urllib.error

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_RAG_MODEL = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e2b")
OLLAMA_EVAL_MODEL = os.getenv("OLLAMA_EVAL_MODEL", "qwen3.5:2b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_CONTEXT_LENGTH = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "110000"))
FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "retrieval", "richtlinien", "faiss_index")

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
        "Nutze die folgenden Dokumente als Kontext, falls sie relevant sind.",
        "",
        f"Frage: {query}",
    ]

    if contexts:
        prompt_lines.append("")
        prompt_lines.append("Kontextinformationen:")
        for index, context in enumerate(contexts, start=1):
            prompt_lines.append(f"{index}. {context}")

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


def run_rag_pipeline(query):
    retriever = _get_retriever()
    retrieved_docs = [doc.page_content for doc in retriever.invoke(query)]

    answer = generate_llm_answer(query, retrieved_docs)

    return {
        "query": query,
        "answer": answer,
        "contexts": retrieved_docs,
    }