import os

OLLAMA_EVAL_MODEL = os.getenv("OLLAMA_EVAL_MODEL", "qwen3.5:4b")
OLLAMA_EVAL_EMBEDDINGS_MODEL = os.getenv("OLLAMA_EVAL_EMBEDDINGS_MODEL", "qwen3-embedding:0.6b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "8192"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.90"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
OLLAMA_CONTEXT_LENGTH = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "16384"))
JSON_SYSTEM_PROMPT = (
    "You are a RAG evaluation assistant. Your task is to evaluate the quality of a generated answer based on the question, the answer, and the retrieved contexts. "
    "Follow the user's instructions and return your answer as a single JSON object that matches the schema given in the prompt. "
    "Do not wrap the JSON in markdown code fences and do not add commentary before or after it. "
)
