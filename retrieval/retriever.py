import json
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from retrieval.embeddings import PASSAGE_PREFIX, get_embeddings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = str(_PROJECT_ROOT / "richtlinien" / "all_chunks.json")
FAISS_INDEX_DIR = os.getenv(
    "FAISS_INDEX_DIR",
    str(_PROJECT_ROOT / "richtlinien" / "faiss_index_bge_m3_cosine"),
)


def build_vectorstore(chunks_path=CHUNKS_PATH, index_dir=FAISS_INDEX_DIR):
    embeddings = get_embeddings(show_progress_bar=not os.path.exists(index_dir))

    if os.path.exists(index_dir):
        return FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            normalize_L2=True,
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [PASSAGE_PREFIX + c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # MAX_INNER_PRODUCT + normalize_L2 = cosine similarity. bge-m3 already
    # emits unit vectors, but normalize_L2 keeps the metric well-defined if
    # the embedding model is later swapped for an unnormalised one.
    vectorstore = FAISS.from_texts(
        texts,
        embeddings,
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )
    vectorstore.save_local(index_dir)
    return vectorstore


def build_retriever(chunks_path=CHUNKS_PATH, index_dir=FAISS_INDEX_DIR, k=3):
    return build_vectorstore(chunks_path, index_dir).as_retriever(
        search_kwargs={"k": k}
    )
