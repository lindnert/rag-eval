import asyncio
import contextvars
import os
import re
import shutil
import subprocess
from typing import cast


from utils import _prompt_to_text
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.dataset_schema import EvaluationResult
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import SystemMessage, HumanMessage
from ragas.llms import LangchainLLMWrapper

from evaluation.eval_config import (
    OLLAMA_EVAL_MODEL,
    OLLAMA_EVAL_EMBEDDINGS_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TOP_P,
    OLLAMA_CONTEXT_LENGTH,
    JSON_SYSTEM_PROMPT,
)

RAGAS_TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "900"))

print(f"[ragas_eval] OLLAMA_EVAL_MODEL = {OLLAMA_EVAL_MODEL}", flush=True)

EVAL_DEBUG_LLM = os.getenv("EVAL_DEBUG_LLM", "1") == "1"
RAGAS_CONCURRENCY = int(os.getenv("RAGAS_CONCURRENCY", "2"))
print(f"RAGAS_CONCURRENCY is set to {RAGAS_CONCURRENCY}", flush=True)
# Tag debug prints with the sample index so interleaved concurrent output is readable.
_current_sample_idx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_current_sample_idx", default=None
)
_call_counter = {"n": 0}
_retry_state = {"last_primary_n": 0, "retry_idx": 0}
_prompt_store: dict[int, str] = {}
_diag_state = {"printed": False, "concurrent_printed": False}

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)
_RETRY_PROMPT_MARKER = "The output string did not satisfy"


def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    m = _CODE_FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def _print_gpu_diagnostics(label="after first call"):
    """Print nvidia-smi and `ollama ps`."""
    print(f"\n========== GPU / Ollama diagnostics ({label}) ==========", flush=True)
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


def _build_base_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_EVAL_MODEL,
        base_url="http://localhost:11434",
        num_ctx=OLLAMA_CONTEXT_LENGTH,
        num_predict=OLLAMA_NUM_PREDICT,
        disable_streaming=True,
        temperature=OLLAMA_TEMPERATURE,
        top_p=OLLAMA_TOP_P,
        reasoning=False,
    )


def _build_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EVAL_EMBEDDINGS_MODEL,
        base_url="http://localhost:11434",
    )


class RagasJSONWrapper:
    def __init__(self, llm):
        self.llm = llm

    def _log_prompt(self, prompt):
        if not EVAL_DEBUG_LLM:
            return None
        _call_counter["n"] += 1
        n = _call_counter["n"]
        prompt_str = _prompt_to_text(prompt)
        sample_idx = _current_sample_idx.get()
        sample_tag = f"sample={sample_idx}" if sample_idx is not None else "sample=?"

        is_retry = _RETRY_PROMPT_MARKER in prompt_str[:300]
        if is_retry:
            _retry_state["retry_idx"] += 1
            retry_label = f"RETRY #{_retry_state['retry_idx']} of primary call #{_retry_state['last_primary_n']}"
        else:
            _retry_state["last_primary_n"] = n
            _retry_state["retry_idx"] = 0
            retry_label = "PRIMARY (retry 0)"

        # Stash prompt; it will be printed together with the response when the
        # call completes (single self-contained block per call).
        _prompt_store[n] = prompt_str

        # Once we've kicked off enough calls that concurrency should be saturated,
        # snapshot the GPU under load (single shot).
        if not _diag_state["concurrent_printed"] and n >= RAGAS_CONCURRENCY:
            _diag_state["concurrent_printed"] = True
            _print_gpu_diagnostics(label=f"under load, after {n} calls launched")

        return (n, sample_tag, retry_label)

    def _log_response(self, ctx, response=None, error=None):
        if not EVAL_DEBUG_LLM or ctx is None:
            return
        n, sample_tag, retry_label = ctx
        prompt_str = _prompt_store.pop(n, "")
        prompt_section = (
            f"[PROMPT len_chars={len(prompt_str)}]\n"
            f"[PROMPT FULL]\n{prompt_str}\n"
        )
        header = f"ragas LLM call #{n} | {sample_tag} | {retry_label}"
        if error is not None:
            block = (
                f"\n--- {header} | COMPLETED (ERROR) ---\n"
                f"{prompt_section}"
                f"[ERROR] {type(error).__name__}: {error}\n"
                f"--- end call #{n} ({sample_tag}) ---\n"
            )
        else:
            block = (
                f"\n--- {header} | COMPLETED ---\n"
                f"{prompt_section}"
                f"[RESPONSE len_chars={len(response or '')}]\n{response}\n"
                f"--- end call #{n} ({sample_tag}) ---\n"
            )
        print(block, flush=True)

        if not _diag_state["printed"]:
            _diag_state["printed"] = True
            _print_gpu_diagnostics()

    async def agenerate(self, prompt, **kwargs):
        messages = [
            SystemMessage(content=JSON_SYSTEM_PROMPT),
            HumanMessage(content=_prompt_to_text(prompt)),
        ]
        ctx = self._log_prompt(prompt)
        try:
            result = await self.llm.ainvoke(messages)
            response = (result.content if result is not None else None) or "{}"
            response = _strip_code_fences(response)
        except Exception as e:
            self._log_response(ctx, error=e)
            raise
        self._log_response(ctx, response=response)
        return response

    async def agenerate_prompt(self, prompts, **kwargs):
        texts = await asyncio.gather(*(self.agenerate(p, **kwargs) for p in prompts))
        return LLMResult(generations=[[Generation(text=t)] for t in texts])


def run_ragas(sample):
    # When EVAL_DEBUG_LLM, ask ragas to raise on per-metric failures instead of
    # silently emitting NaN — otherwise null scores have no visible cause.
    raise_exceptions = EVAL_DEBUG_LLM

    dataset = Dataset.from_dict({
        "question": [sample["query"]],
        "answer": [sample["answer"]],
        "contexts": [sample["contexts"]],
    })

    # Build fresh clients per-call. ragas.evaluate() spins up its own short-lived
    # event loop internally; reusing module-level ChatOllama/OllamaEmbeddings
    # across calls leaves their lazy httpx.AsyncClient bound to a closed loop,
    # producing "Event loop is closed" / "Event bound to a different event loop".
    ragas_llm = LangchainLLMWrapper(RagasJSONWrapper(_build_base_llm()))
    embeddings = _build_embeddings()

    try:
        result = cast(
            EvaluationResult,
            evaluate(
                dataset,
                llm=ragas_llm,
                embeddings=embeddings,
                metrics=[Faithfulness(), AnswerRelevancy()],
                return_executor=False,
                raise_exceptions=raise_exceptions,
                run_config=RunConfig(timeout=RAGAS_TIMEOUT, max_retries=3, max_wait=60),
            ),
        )

        faith = result["faithfulness"][0]
        relev = result["answer_relevancy"][0]
        return {
            "ragas_faithfulness": faith if faith == faith else None,
            "ragas_answer_relevancy": relev if relev == relev else None,
        }

    except Exception as e:
        sample_idx = _current_sample_idx.get()
        print(
            f"[ragas_eval] run_ragas FAILED for sample={sample_idx}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return {
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
            "ragas_error": f"{type(e).__name__}: {e}",
        }


async def arun_ragas(sample, semaphore: asyncio.Semaphore | None = None, idx: int | None = None):
    """Async wrapper around run_ragas. ragas.evaluate is sync internally, so we
    offload it to a thread; multiple threads in flight let Ollama serve the
    requests in parallel (requires OLLAMA_NUM_PARALLEL >= concurrency).

    `idx` is propagated to the worker thread via a ContextVar so debug prints
    can be tagged with the originating sample index.
    """
    if idx is not None:
        _current_sample_idx.set(idx)
    if semaphore is None:
        return await asyncio.to_thread(run_ragas, sample)
    async with semaphore:
        return await asyncio.to_thread(run_ragas, sample)


async def arun_ragas_batch(samples, concurrency: int | None = None, on_done=None):
    """Run ragas for many samples concurrently. Returns scores in input order.

    on_done: optional callback `on_done(idx, sample, scores)` invoked as each
    sample finishes (useful for progress logging / partial saves).
    """
    if concurrency is None:
        concurrency = RAGAS_CONCURRENCY
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list = [None] * len(samples)

    async def _one(i, s):
        # Each task gets its own context (asyncio.Task copies the parent
        # context) so per-task ContextVar.set() is isolated.
        scores = await arun_ragas(s, semaphore, idx=i)
        results[i] = scores
        if on_done is not None:
            on_done(i, s, scores)
        return scores

    await asyncio.gather(*(_one(i, s) for i, s in enumerate(samples)))
    return results


def run_ragas_batch(samples, concurrency: int | None = None, on_done=None):
    """Sync entry point for the async batch runner."""
    return asyncio.run(arun_ragas_batch(samples, concurrency=concurrency, on_done=on_done))
