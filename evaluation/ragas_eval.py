import os
import re
import shutil
import subprocess
from typing import cast

from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import SystemMessage, HumanMessage
from ragas.llms import LangchainLLMWrapper

OLLAMA_EVAL_MODEL = os.getenv("OLLAMA_EVAL_MODEL", "qwen3.5:4b")
OLLAMA_EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDINGS_MODEL", "qllama/multilingual-e5-base:q4_k_m")
OLLAMA_CONTEXT_LENGTH = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "8192"))

JSON_SYSTEM_PROMPT = (
    "Follow the user's instructions and return your answer as a single JSON object "
    "that matches the schema given in the prompt. "
    "Do not wrap the JSON in markdown code fences and do not add commentary before or after it. "
    "/no_think"
)

print(f"[ragas_eval] OLLAMA_EVAL_MODEL = {OLLAMA_EVAL_MODEL}", flush=True)

EVAL_DEBUG_LLM = os.getenv("EVAL_DEBUG_LLM", "1") == "1"
_call_counter = {"n": 0}
_retry_state = {"last_primary_n": 0, "retry_idx": 0}
_diag_state = {"printed": False}

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)
_RETRY_PROMPT_MARKER = "The output string did not satisfy"


def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    m = _CODE_FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def _print_gpu_diagnostics():
    """Print nvidia-smi and `ollama ps` once, after the first real LLM call."""
    print("\n========== GPU / Ollama diagnostics (after first call) ==========", flush=True)
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True, timeout=10,
            )
            print(out.stdout, flush=True)
            if out.stderr:
                print(out.stderr, flush=True)
        except Exception as e:
            print(f"[diag] nvidia-smi failed: {e}", flush=True)
    else:
        print("[diag] nvidia-smi not on PATH", flush=True)

    if shutil.which("ollama"):
        try:
            out = subprocess.run(
                ["ollama", "ps"],
                capture_output=True, text=True, timeout=10,
            )
            print(out.stdout, flush=True)
            if out.stderr:
                print(out.stderr, flush=True)
        except Exception as e:
            print(f"[diag] ollama ps failed: {e}", flush=True)
    else:
        print("[diag] ollama not on PATH", flush=True)
    print("========== end diagnostics ==========\n", flush=True)


# format="json" enforces Ollama's structured-output mode at the server side.
# Combined with the system prompt + ragas's schema-in-prompt this yields
# strictly valid JSON without code fences in nearly all cases.
base_llm = ChatOllama(
    model=OLLAMA_EVAL_MODEL,
    base_url="http://localhost:11434",
    num_ctx=OLLAMA_CONTEXT_LENGTH,
    format="json",
)


class RagasJSONWrapper:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, prompt, **kwargs):
        messages = [
            SystemMessage(content=JSON_SYSTEM_PROMPT),
            HumanMessage(content=str(prompt)),
        ]
        response = self.llm.invoke(messages).content or "{}"
        response = _strip_code_fences(response)

        if EVAL_DEBUG_LLM:
            _call_counter["n"] += 1
            n = _call_counter["n"]
            prompt_str = str(prompt)

            is_retry = _RETRY_PROMPT_MARKER in prompt_str[:300]
            if is_retry:
                _retry_state["retry_idx"] += 1
                retry_label = f"RETRY #{_retry_state['retry_idx']} of primary call #{_retry_state['last_primary_n']}"
            else:
                _retry_state["last_primary_n"] = n
                _retry_state["retry_idx"] = 0
                retry_label = "PRIMARY (retry 0)"

            print(f"\n--- ragas LLM call #{n} | {retry_label} ---", flush=True)
            print(f"[PROMPT len_chars={len(prompt_str)}]", flush=True)
            print(f"[PROMPT FULL]\n{prompt_str}", flush=True)
            print(f"[RESPONSE len_chars={len(response)}]\n{response}", flush=True)
            print(f"--- end call #{n} ---\n", flush=True)

        if not _diag_state["printed"]:
            _diag_state["printed"] = True
            _print_gpu_diagnostics()

        return response

    async def agenerate(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def generate_prompt(self, prompts, **kwargs):
        outputs = []
        for p in prompts:
            text = self.generate(p, **kwargs)
            outputs.append([Generation(text=text)])
        return LLMResult(generations=outputs)

    async def agenerate_prompt(self, prompts, **kwargs):
        return self.generate_prompt(prompts, **kwargs)


wrapped_llm = RagasJSONWrapper(base_llm)
ragas_llm = LangchainLLMWrapper(wrapped_llm)


embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBEDDINGS_MODEL,
    base_url="http://localhost:11434"
)


def run_ragas(sample):
    dataset = Dataset.from_dict({
        "question": [sample["query"]],
        "answer": [sample["answer"]],
        "contexts": [sample["contexts"]],
    })

    try:
        result = cast(
            EvaluationResult,
            evaluate(
                dataset,
                llm=ragas_llm,
                embeddings=embeddings,
                metrics=[Faithfulness(), AnswerRelevancy()],
                return_executor=False,
            ),
        )

        return {
            "ragas_faithfulness": result["faithfulness"][0] if result["faithfulness"][0] == result["faithfulness"][0] else None,
            "ragas_answer_relevancy": result["answer_relevancy"][0] if result["answer_relevancy"][0] == result["answer_relevancy"][0] else None,
        }

    except Exception as e:

        return {
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
            "ragas_error": str(e)
        }
