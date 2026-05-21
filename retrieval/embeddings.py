import os

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

EMB_BATCH_SIZE = int(os.getenv("EMB_BATCH_SIZE", "32"))

# Embedding model prefixes — must stay consistent between indexing and querying.
# Current model (multilingual-e5-base) was trained with "passage: " / "query: ".
# For models that don't use prefixes (e.g. bge-m3), set these to "".
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

LLAMACPP_EMB_BASE_URL = os.getenv("LLAMACPP_EMB_BASE_URL", "http://127.0.0.1:8081/v1")
LLAMACPP_EMB_MODEL = os.getenv(
    "LLAMACPP_EMB_MODEL",
    "dinab/multilingual-e5-base-Q4_K_M-GGUF",
)


def get_embeddings(show_progress_bar: bool = False) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=LLAMACPP_EMB_MODEL,
        base_url=LLAMACPP_EMB_BASE_URL,
        api_key=SecretStr("sk-no-key-required"),
        check_embedding_ctx_length=False,
        tiktoken_enabled=False,
        chunk_size=EMB_BATCH_SIZE,
        timeout=120,
        show_progress_bar=show_progress_bar,
    )
