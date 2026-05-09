import asyncio
import contextvars
import json
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from evaluation.utils import _prompt_to_text, _strip_code_fences, print_gpu_diagnostics
from evaluation.eval_config import (
    OLLAMA_EVAL_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TOP_P,
    OLLAMA_CONTEXT_LENGTH,
    JSON_SYSTEM_PROMPT,
)

EVAL_DEBUG_LLM = os.getenv("EVAL_DEBUG_LLM", "1") == "1"
DEEPEVAL_CONCURRENCY = int(os.getenv("DEEPEVAL_CONCURRENCY", "4"))
print(f"[deepeval_eval] OLLAMA_EVAL_MODEL = {OLLAMA_EVAL_MODEL}", flush=True)
print(f"[deepeval_eval] DEEPEVAL_CONCURRENCY = {DEEPEVAL_CONCURRENCY}", flush=True)

_current_sample_idx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_deepeval_sample_idx", default=None
)
_call_counter = {"n": 0}
_diag_state = {"printed": False, "concurrent_printed": False}


def _coerce_to_schema(text: str, schema):
    """Parse `text` as JSON and instantiate the given Pydantic schema.

    DeepEval metrics call generate(prompt, schema=SomePydanticModel) and access
    fields on the returned object (result.truths, result.verdicts, result.reason).
    Returning raw JSON text caused KeyErrors like 'truths' / 'verdicts' / 'reason'
    because DeepEval fell back to dict access on a response shaped for a different
    step.
    """
    cleaned = _strip_code_fences(text or "").strip() or "{}"
    try:
        data = json.loads(cleaned)
    except Exception as e:
        raise ValueError(
            f"LLM returned non-JSON for schema {schema.__name__}: {e}\n--RAW--\n{text}"
        )
    try:
        if isinstance(data, dict):
            return schema(**data)
        return schema.model_validate(data)
    except Exception as e:
        raise ValueError(
            f"LLM JSON did not match schema {schema.__name__}: {e}\n"
            f"--PARSED-- {data}\n--RAW-- {text}"
        )


def _log_call(prompt_str, schema, response_text=None, error=None):
    if not EVAL_DEBUG_LLM:
        return
    _call_counter["n"] += 1
    n = _call_counter["n"]
    # Once enough calls have launched that concurrency should be saturated,
    # snapshot the GPU under load (single shot).
    if not _diag_state["concurrent_printed"] and n >= DEEPEVAL_CONCURRENCY:
        _diag_state["concurrent_printed"] = True
        print_gpu_diagnostics(label=f"deepeval under load, after {n} calls launched")
    sample_idx = _current_sample_idx.get()
    sample_tag = f"sample={sample_idx}" if sample_idx is not None else "sample=?"
    schema_name = getattr(schema, "__name__", "NO_SCHEMA")
    header = f"deepeval LLM call #{n} | {sample_tag} | schema={schema_name}"
    if error is not None:
        block = (
            f"\n--- {header} | COMPLETED (ERROR) ---\n"
            f"[PROMPT len_chars={len(prompt_str)}]\n[PROMPT FULL]\n{prompt_str}\n"
            f"[ERROR] {type(error).__name__}: {error}\n"
            f"--- end call #{n} ({sample_tag}) ---\n"
        )
    else:
        block = (
            f"\n--- {header} | COMPLETED ---\n"
            f"[PROMPT len_chars={len(prompt_str)}]\n[PROMPT FULL]\n{prompt_str}\n"
            f"[RESPONSE len_chars={len(response_text or '')}]\n{response_text}\n"
            f"--- end call #{n} ({sample_tag}) ---\n"
        )
    print(block, flush=True)

    if not _diag_state["printed"]:
        _diag_state["printed"] = True
        print_gpu_diagnostics(label="deepeval after first call")


class OllamaWrapper(DeepEvalBaseLLM):
    def __init__(self, llm):
        self.llm = llm

    def load_model(self):
        return self.llm

    def _messages(self, prompt):
        return [
            SystemMessage(content=JSON_SYSTEM_PROMPT),
            HumanMessage(content=_prompt_to_text(prompt)),
        ]

    def generate(self, prompt, schema=None, **kwargs):
        prompt_str = _prompt_to_text(prompt)
        try:
            response = self.llm.invoke(self._messages(prompt)).content or ""
        except Exception as e:
            _log_call(prompt_str, schema, error=e)
            raise
        response = _strip_code_fences(response)
        _log_call(prompt_str, schema, response_text=response)
        if schema is not None:
            return _coerce_to_schema(response, schema)
        return response

    async def a_generate(self, prompt, schema=None, **kwargs):
        prompt_str = _prompt_to_text(prompt)
        try:
            result = await self.llm.ainvoke(self._messages(prompt))
            response = (result.content if result is not None else "") or ""
        except Exception as e:
            _log_call(prompt_str, schema, error=e)
            raise
        response = _strip_code_fences(response)
        _log_call(prompt_str, schema, response_text=response)
        if schema is not None:
            return _coerce_to_schema(response, schema)
        return response

    def get_model_name(self):
        return f"ollama-{OLLAMA_EVAL_MODEL}"


def _build_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_EVAL_MODEL,
        base_url="http://localhost:11434",
        format="json",
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_NUM_PREDICT,
        top_p=OLLAMA_TOP_P,
        num_ctx=OLLAMA_CONTEXT_LENGTH,
        disable_streaming=True,
        reasoning=False,
    )


def _build_metrics(model):
    return (
        FaithfulnessMetric(model=model, async_mode=True),
        AnswerRelevancyMetric(model=model, async_mode=True),
    )


def run_deepeval(sample):
    """Sync single-sample entry point (kept for compatibility)."""
    test_case = LLMTestCase(
        input=sample["query"],
        actual_output=sample["answer"],
        retrieval_context=sample["contexts"],
    )
    ollama_model = OllamaWrapper(_build_llm())
    faithfulness, relevance = _build_metrics(ollama_model)
    try:
        faithfulness.measure(test_case)
        relevance.measure(test_case)
        return {
            "deepeval_faithfulness": faithfulness.score,
            "deepeval_relevance": relevance.score,
        }
    except Exception as e:
        return {
            "deepeval_faithfulness": None,
            "deepeval_relevance": None,
            "deepeval_error": f"{type(e).__name__}: {e}",
        }


async def arun_deepeval(sample, semaphore: asyncio.Semaphore | None = None, idx: int | None = None):
    if idx is not None:
        _current_sample_idx.set(idx)

    async def _go():
        test_case = LLMTestCase(
            input=sample["query"],
            actual_output=sample["answer"],
            retrieval_context=sample["contexts"],
        )
        # Fresh client per sample: same event-loop / httpx.AsyncClient hazard
        # already documented in ragas_eval.py.
        ollama_model = OllamaWrapper(_build_llm())
        faithfulness, relevance = _build_metrics(ollama_model)
        try:
            await faithfulness.a_measure(test_case)
            await relevance.a_measure(test_case)
            return {
                "deepeval_faithfulness": faithfulness.score,
                "deepeval_relevance": relevance.score,
            }
        except Exception as e:
            print(
                f"[deepeval_eval] arun_deepeval FAILED sample={idx}: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            return {
                "deepeval_faithfulness": None,
                "deepeval_relevance": None,
                "deepeval_error": f"{type(e).__name__}: {e}",
            }

    if semaphore is None:
        return await _go()
    async with semaphore:
        return await _go()


async def arun_deepeval_batch(samples, concurrency: int | None = None, on_done=None):
    if concurrency is None:
        concurrency = DEEPEVAL_CONCURRENCY
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list = [None] * len(samples)

    async def _one(i, s):
        scores = await arun_deepeval(s, semaphore, idx=i)
        results[i] = scores
        if on_done is not None:
            on_done(i, s, scores)
        return scores

    await asyncio.gather(*(_one(i, s) for i, s in enumerate(samples)))
    return results


def run_deepeval_batch(samples, concurrency: int | None = None, on_done=None):
    return asyncio.run(arun_deepeval_batch(samples, concurrency=concurrency, on_done=on_done))
