import json
import os
import urllib.request
import urllib.error
from langchain_text_splitters import CharacterTextSplitter   
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS   
from retrieval.docs import DEFAULT_DOCUMENTS

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


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
    if isinstance(response_json, dict):
        if "output" in response_json:
            output = response_json["output"]
            if isinstance(output, list) and output:
                first = output[0]
                if isinstance(first, dict):
                    return first.get("content") or first.get("text") or str(first)
                return str(first)

        if "choices" in response_json:
            choices = response_json["choices"]
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    return first.get("text") or first.get("message") or str(first)
                return str(first)

        if "text" in response_json:
            return response_json["text"]

    return json.dumps(response_json, ensure_ascii=False)


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


def build_retriever(documents, k=3):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = []
    metadatas = []

    for i, doc in enumerate(documents):
        for chunk in splitter.split_text(doc):
            texts.append(chunk)
            metadatas.append({"source": f"doc_{i}"})

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def run_rag_pipeline(query):
    documents = DEFAULT_DOCUMENTS
    retriever = build_retriever(documents)
    retrieved_docs = [doc.page_content for doc in retriever.invoke(query)]

    answer = generate_llm_answer(query, retrieved_docs)

    return {
        "query": query,
        "answer": answer,
        "contexts": retrieved_docs,
    }