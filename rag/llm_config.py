import os

LLAMACPP_GEN_BASE_URL = os.getenv("LLAMACPP_GEN_BASE_URL", "http://localhost:8080/v1")
LLAMACPP_RAG_MODEL = os.getenv(
    "LLAMACPP_RAG_MODEL",
    "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
)

# Qwen3-family chat template emits <think>…</think> blocks by default; we
# disable thinking on the baseline so gen_logprobs reflect only the answer.
# Step 4 (self-correction) will flip this on for the rag_sc regen.
LLAMACPP_RAG_ENABLE_THINKING = (
    os.getenv("LLAMACPP_RAG_ENABLE_THINKING", "false").lower() == "true"
)

LLAMACPP_RAG_TEMPERATURE = float(os.getenv("LLAMACPP_RAG_TEMPERATURE", "0.1"))
LLAMACPP_RAG_TOP_P = float(os.getenv("LLAMACPP_RAG_TOP_P", "0.95"))

# Safety cap on completion length, not a target — the system prompt asks for
# 3-4 paragraphs (~500-800 tokens). Keeps room for the self-correcting variant,
# which may emit short reasoning before the final answer.
LLAMACPP_RAG_MAX_TOKENS = int(os.getenv("LLAMACPP_RAG_MAX_TOKENS", "2048"))

# top_logprobs=5 mirrors the OpenAI default; we only actually consume the
# chosen-token logprob, but keeping the top-k around makes future analysis
# (e.g. token-level entropy) cheap to add without re-running generation.
LLAMACPP_RAG_TOP_LOGPROBS = int(os.getenv("LLAMACPP_RAG_TOP_LOGPROBS", "5"))

LLAMACPP_RAG_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "6"))

RAG_K = int(os.getenv("RAG_K", "3"))
