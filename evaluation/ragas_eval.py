import asyncio
import contextvars
import os
from typing import cast

from evaluation.utils import (
    _prompt_to_text,
    _strip_code_fences,
    print_gpu_diagnostics as _print_gpu_diagnostics,
)
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevance,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import embedding_factory, BaseRagasEmbedding
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr

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

print(f"[ragas_eval] LLAMACPP_EVAL_MODEL = {LLAMACPP_EVAL_MODEL}", flush=True)

EVAL_DEBUG_LLM = os.getenv("EVAL_DEBUG_LLM", "1") == "1"
RAGAS_CONCURRENCY = int(os.getenv("RAGAS_CONCURRENCY", "6"))  # match gen server slot count
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


def _build_embeddings():
    # Native llama-server on LLAMACPP_EMB_BASE_URL (started in run_eval.sh
    # with --parallel 1, so n_seq_max=1 and n_ctx is honored exactly).
    # Collections metrics require the modern BaseRagasEmbedding interface,
    # which embedding_factory produces from an AsyncOpenAI client.
    client = AsyncOpenAI(
        api_key="sk-no-key-required",
        base_url=LLAMACPP_EMB_BASE_URL,
    )
    return cast(
        BaseRagasEmbedding,
        embedding_factory(
            "openai",
            model=LLAMACPP_EVAL_EMBEDDINGS_MODEL,
            client=client,
        ),
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


_NO_REASON_FALLBACK = "no reasoning provided by ragas"


def _score_value(result):
    # Collections metrics return a MetricResult with .value; be defensive in
    # case a numeric is returned directly.
    val = getattr(result, "value", result)
    try:
        return float(val) if val is not None and val == val else None
    except (TypeError, ValueError):
        return None


def _score_reason(result) -> str:
    reason = getattr(result, "reason", None)
    if reason is None or (isinstance(reason, str) and not reason.strip()):
        return _NO_REASON_FALLBACK
    return reason


async def _score_one(sample: dict) -> dict:
    ragas_llm = LangchainLLMWrapper(RagasJSONWrapper(_build_base_llm()))
    embeddings = _build_embeddings()

    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=embeddings)
    context_relevance = ContextRelevance(llm=ragas_llm)

    user_input = sample["query"]
    response = sample["answer"]
    retrieved_contexts = sample["contexts"]

    try:
        faith_res, relev_res, ctx_res = await asyncio.gather(
            faithfulness.ascore(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            ),
            answer_relevancy.ascore(
                user_input=user_input,
                response=response,
            ),
            context_relevance.ascore(
                user_input=user_input,
                retrieved_contexts=retrieved_contexts,
            ),
        )
        return {
            "ragas_faithfulness": _score_value(faith_res),
            "ragas_faithfulness_reason": _score_reason(faith_res),
            "ragas_answer_relevancy": _score_value(relev_res),
            "ragas_answer_relevancy_reason": _score_reason(relev_res),
            "ragas_context_relevance": _score_value(ctx_res),
            "ragas_context_relevance_reason": _score_reason(ctx_res),
        }
    except Exception as e:
        import traceback
        sample_idx = _current_sample_idx.get()
        print(
            f"[ragas_eval] _score_one FAILED for sample={sample_idx}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        traceback.print_exc()
        return {
            "ragas_faithfulness": None,
            "ragas_faithfulness_reason": _NO_REASON_FALLBACK,
            "ragas_answer_relevancy": None,
            "ragas_answer_relevancy_reason": _NO_REASON_FALLBACK,
            "ragas_context_relevance": None,
            "ragas_context_relevance_reason": _NO_REASON_FALLBACK,
            "ragas_error": f"{type(e).__name__}: {e}",
        }


def run_ragas(sample):
    return asyncio.run(_score_one(sample))


async def arun_ragas(sample, semaphore: asyncio.Semaphore | None = None, idx: int | None = None):
    if idx is not None:
        _current_sample_idx.set(idx)
    if semaphore is None:
        return await _score_one(sample)
    async with semaphore:
        return await _score_one(sample)


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
