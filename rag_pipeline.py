import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHUNKS_PATH = str(Path(__file__).resolve().parent / "richtlinien" / "all_chunks.json")


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
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
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


def build_retriever(chunks_path=CHUNKS_PATH, k=3):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def run_rag_pipeline(query, retriever):
    retrieved_docs = [doc.page_content for doc in retriever.invoke(query)]

    answer = generate_llm_answer(query, retrieved_docs)

    return {
        "query": query,
        "answer": answer,
        "contexts": retrieved_docs,
    }