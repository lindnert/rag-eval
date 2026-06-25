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

LLAMACPP_RAG_TEMPERATURE = float(os.getenv("LLAMACPP_RAG_TEMPERATURE", "0.0"))
LLAMACPP_RAG_TOP_P = float(os.getenv("LLAMACPP_RAG_TOP_P", "1.0"))

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

# ---------------------------------------------------------------------------
# Self-correction (rag_sc variant). Thresholds are placeholders — tune once
# we have logprob and retrieval-score distributions from the baseline runs.
# ---------------------------------------------------------------------------

# Direction of FAISS similarity_search_with_score values:
#   "lower"  → L2 distance, smaller = more similar (langchain default)
#   "higher" → inner-product / cosine, larger = more similar
RAG_SC_SCORE_DIRECTION = os.getenv("RAG_SC_SCORE_DIRECTION", "higher").lower()

# Trigger T: the *best* score in the retrieved set is itself unacceptable.
#   lower=better → fire if min(scores) > threshold
#   higher=better → fire if max(scores) < threshold
RAG_SC_RETRIEVAL_BEST_THRESHOLD = float(
    os.getenv("RAG_SC_RETRIEVAL_BEST_THRESHOLD", "0.73")
)

# Trigger Δ: high spread between best and worst → noise mixed in.
RAG_SC_RETRIEVAL_SPREAD_THRESHOLD = float(
    os.getenv("RAG_SC_RETRIEVAL_SPREAD_THRESHOLD", "0.04")
)

# Trigger U: mean token logprob below this → low overall confidence.
RAG_SC_GEN_MEAN_LOGPROB_THRESHOLD = float(
    os.getenv("RAG_SC_GEN_MEAN_LOGPROB_THRESHOLD", "-0.3")
)

# Trigger V: some token's logprob below this → at least one very uncertain step.
RAG_SC_GEN_MIN_LOGPROB_THRESHOLD = float(
    os.getenv("RAG_SC_GEN_MIN_LOGPROB_THRESHOLD", "-3.0")
)

# HyDE draft is only used to embed-and-retrieve; doesn't need to be long.
RAG_SC_HYDE_MAX_TOKENS = int(os.getenv("RAG_SC_HYDE_MAX_TOKENS", "512"))
