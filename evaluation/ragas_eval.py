import asyncio
import contextvars
import os
from typing import cast


from evaluation.utils import _prompt_to_text, _strip_code_fences, print_gpu_diagnostics as _print_gpu_diagnostics
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.dataset_schema import EvaluationResult
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr
from ragas.llms import LangchainLLMWrapper

from evaluation.eval_config_llamacpp import (
    LLAMACPP_EVAL_MODEL,
    LLAMACPP_EVAL_EMBEDDINGS_MODEL,
    LLAMACPP_GEN_BASE_URL,
    LLAMACPP_EMB_BASE_URL,
    LLAMACPP_TEMPERATURE,
    LLAMACPP_NUM_PREDICT,
    LLAMACPP_TOP_P,
    LLAMACPP_REPEAT_PENALTY,
    LLAMACPP_REPEAT_LAST_N,
    JSON_SYSTEM_PROMPT,
)

RAGAS_TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "900"))

print(f"[ragas_eval] LLAMACPP_EVAL_MODEL = {LLAMACPP_EVAL_MODEL}", flush=True)

EVAL_DEBUG_LLM = os.getenv("EVAL_DEBUG_LLM", "1") == "1"
RAGAS_CONCURRENCY = int(os.getenv("RAGAS_CONCURRENCY", "4"))
print(f"RAGAS_CONCURRENCY is set to {RAGAS_CONCURRENCY}", flush=True)
_current_sample_idx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_current_sample_idx", default=None
)
_call_counter = {"n": 0}
_retry_state = {"last_primary_n": 0, "retry_idx": 0}
_prompt_store: dict[int, str] = {}
_diag_state = {"printed": False, "concurrent_printed": False}

_RETRY_PROMPT_MARKER = "The output string did not satisfy"


def _build_base_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLAMACPP_EVAL_MODEL,
        base_url=LLAMACPP_GEN_BASE_URL,
        api_key=SecretStr("sk-no-key-required"),
        temperature=LLAMACPP_TEMPERATURE,
        top_p=LLAMACPP_TOP_P,
        max_completion_tokens=LLAMACPP_NUM_PREDICT,
        streaming=False,
        extra_body={
            "repeat_penalty": LLAMACPP_REPEAT_PENALTY,
            "repeat_last_n": LLAMACPP_REPEAT_LAST_N,
        },
    )


def _build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=LLAMACPP_EVAL_EMBEDDINGS_MODEL,
        base_url=LLAMACPP_EMB_BASE_URL,
        api_key=SecretStr("sk-no-key-required"),
        check_embedding_ctx_length=False,
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

        _prompt_store[n] = prompt_str

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
    raise_exceptions = EVAL_DEBUG_LLM

    dataset = Dataset.from_dict({
        "question": [sample["query"]],
        "answer": [sample["answer"]],
        "contexts": [sample["contexts"]],
    })

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
    if idx is not None:
        _current_sample_idx.set(idx)
    if semaphore is None:
        return await asyncio.to_thread(run_ragas, sample)
    async with semaphore:
        return await asyncio.to_thread(run_ragas, sample)


async def arun_ragas_batch(samples, concurrency: int | None = None, on_done=None):
    if concurrency is None:
        concurrency = RAGAS_CONCURRENCY
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list = [None] * len(samples)

    async def _one(i, s):
        scores = await arun_ragas(s, semaphore, idx=i)
        results[i] = scores
        if on_done is not None:
            on_done(i, s, scores)
        return scores

    await asyncio.gather(*(_one(i, s) for i, s in enumerate(samples)))
    return results


def run_ragas_batch(samples, concurrency: int | None = None, on_done=None):
    return asyncio.run(arun_ragas_batch(samples, concurrency=concurrency, on_done=on_done))
