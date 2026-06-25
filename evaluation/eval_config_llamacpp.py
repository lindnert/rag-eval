import os

LLAMACPP_EVAL_MODEL = os.getenv(
    "LLAMACPP_EVAL_MODEL",
    "unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL",
)
LLAMACPP_EVAL_EMBEDDINGS_MODEL = os.getenv(
    "LLAMACPP_EVAL_EMBEDDINGS_MODEL",
    "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
)
LLAMACPP_GEN_BASE_URL = os.getenv("LLAMACPP_GEN_BASE_URL", "http://localhost:8080/v1")
LLAMACPP_EMB_BASE_URL = os.getenv("LLAMACPP_EMB_BASE_URL", "http://localhost:8081/v1")

LLAMACPP_TEMPERATURE = float(os.getenv("LLAMACPP_TEMPERATURE", "0.0"))
LLAMACPP_NUM_PREDICT = int(os.getenv("LLAMACPP_NUM_PREDICT", "4500"))
LLAMACPP_TOP_P = float(os.getenv("LLAMACPP_TOP_P", "1.0"))
LLAMACPP_REPEAT_PENALTY = float(os.getenv("LLAMACPP_REPEAT_PENALTY", "1.1"))
LLAMACPP_REPEAT_LAST_N = int(os.getenv("LLAMACPP_REPEAT_LAST_N", "16"))
LLAMACPP_CONTEXT_LENGTH = int(os.getenv("LLAMACPP_CONTEXT_LENGTH", "32768"))

JSON_SYSTEM_PROMPT = (
    "You are a RAG evaluation assistant. Your task is to evaluate the quality of a generated answer based on the question, the answer, and the retrieved contexts. "
    "Follow the user's instructions and return your answer as a single JSON object that matches the schema given in the prompt. "
    "Do not wrap the JSON in markdown code fences and do not add commentary before or after it. "
)
