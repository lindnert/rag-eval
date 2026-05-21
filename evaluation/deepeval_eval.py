import asyncio
import contextvars
import json
import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric

from evaluation.utils import _prompt_to_text, _strip_code_fences, print_gpu_diagnostics
from evaluation.eval_config_llamacpp import (
    LLAMACPP_EVAL_MODEL,
    LLAMACPP_GEN_BASE_URL,
    LLAMACPP_TEMPERATURE,
    LLAMACPP_NUM_PREDICT,
    LLAMACPP_TOP_P,
    LLAMACPP_REPEAT_PENALTY,
    LLAMACPP_REPEAT_LAST_N,
    JSON_SYSTEM_PROMPT,
)

EVAL_DEBUG_LLM = os.getenv("EVAL_DEBUG_LLM", "1") == "1"
DEEPEVAL_CONCURRENCY = int(os.getenv("DEEPEVAL_CONCURRENCY", "6"))
print(f"[deepeval_eval] LLAMACPP_EVAL_MODEL = {LLAMACPP_EVAL_MODEL}", flush=True)
print(f"[deepeval_eval] DEEPEVAL_CONCURRENCY = {DEEPEVAL_CONCURRENCY}", flush=True)

_current_sample_idx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_deepeval_sample_idx", default=None
)
_call_counter = {"n": 0}
_diag_state = {"printed": False, "concurrent_printed": False}


def _coerce_to_schema(text: str, schema):
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


class LlamaCppWrapper(DeepEvalBaseLLM):
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
        return f"llamacpp-{LLAMACPP_EVAL_MODEL}"


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLAMACPP_EVAL_MODEL,
        base_url=LLAMACPP_GEN_BASE_URL,
        api_key=SecretStr("sk-no-key-required"),
        temperature=LLAMACPP_TEMPERATURE,
        top_p=LLAMACPP_TOP_P,
        max_completion_tokens=LLAMACPP_NUM_PREDICT,
        streaming=False,
        model_kwargs={"response_format": {"type": "json_object"}},
        extra_body={
            "repeat_penalty": LLAMACPP_REPEAT_PENALTY,
            "repeat_last_n": LLAMACPP_REPEAT_LAST_N,
        },
    )


def _build_metrics(model):
    return (
        FaithfulnessMetric(model=model, async_mode=True),
        AnswerRelevancyMetric(model=model, async_mode=True),
        ContextualRelevancyMetric(model=model, async_mode=True),
    )


def run_deepeval(sample):
    test_case = LLMTestCase(
        input=sample["query"],
        actual_output=sample["answer"],
        retrieval_context=sample["contexts"],
    )
    eval_model = LlamaCppWrapper(_build_llm())
    faithfulness, relevance, contextual_relevance = _build_metrics(eval_model)
    try:
        faithfulness.measure(test_case)
        relevance.measure(test_case)
        contextual_relevance.measure(test_case)
        return {
            "deepeval_faithfulness": faithfulness.score,
            "deepeval_relevance": relevance.score,
            "deepeval_contextual_relevance": contextual_relevance.score,
        }
    except Exception as e:
        return {
            "deepeval_faithfulness": None,
            "deepeval_relevance": None,
            "deepeval_contextual_relevance": None,
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
        eval_model = LlamaCppWrapper(_build_llm())
        faithfulness, relevance, contextual_relevance = _build_metrics(eval_model)
        try:
            await asyncio.gather(
                faithfulness.a_measure(test_case),
                relevance.a_measure(test_case),
                contextual_relevance.a_measure(test_case),
            )
            return {
                "deepeval_faithfulness": faithfulness.score,
                "deepeval_relevance": relevance.score,
                "deepeval_contextual_relevance": contextual_relevance.score,
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
                "deepeval_contextual_relevance": None,
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

    t0 = time.perf_counter()
    await asyncio.gather(*(_one(i, s) for i, s in enumerate(samples)))
    elapsed = time.perf_counter() - t0
    n = len(samples)
    per = elapsed / n if n else 0.0
    print(
        f"[deepeval_eval] TOTAL deepeval time: {elapsed:.1f}s "
        f"({elapsed/60:.2f} min) over {n} samples ({per:.2f}s/sample)",
        flush=True,
    )
    return results


def run_deepeval_batch(samples, concurrency: int | None = None, on_done=None):
    return asyncio.run(arun_deepeval_batch(samples, concurrency=concurrency, on_done=on_done))
