import json
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS

from retrieval.embeddings import PASSAGE_PREFIX, get_embeddings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = str(_PROJECT_ROOT / "richtlinien" / "all_chunks.json")
FAISS_INDEX_DIR = os.getenv(
    "FAISS_INDEX_DIR",
    str(_PROJECT_ROOT / "richtlinien" / "faiss_index_e5_llamacpp"),
)


def build_retriever(chunks_path=CHUNKS_PATH, index_dir=FAISS_INDEX_DIR, k=3):
    embeddings = get_embeddings(show_progress_bar=not os.path.exists(index_dir))

    if os.path.exists(index_dir):
        vectorstore = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
    else:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        texts = [PASSAGE_PREFIX + c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vectorstore.save_local(index_dir)

    return vectorstore.as_retriever(search_kwargs={"k": k})
