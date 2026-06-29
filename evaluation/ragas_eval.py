import asyncio
import contextvars
import os
import sys
import threading
import types
from typing import cast

# ragas 0.4.3 eagerly imports `langchain_community.chat_models.vertexai`,
# which was removed in langchain-community 0.4.x (Vertex now lives in
# langchain-google-vertexai). Inject a shim module so the import resolves
# before ragas is loaded. We don't actually call ChatVertexAI anywhere.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertex_shim = types.ModuleType("langchain_community.chat_models.vertexai")
    try:
        from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    except Exception:  # pragma: no cover — fall back to a stub if pkg missing
        class _ChatVertexAI:  # type: ignore[no-redef]
            pass
    _vertex_shim.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertex_shim

import math

from evaluation.utils import (
    _prompt_to_text,
    _strip_code_fences,
    print_gpu_diagnostics as _print_gpu_diagnostics,
    NO_RAG_SENTINEL,
    REJECTED_SENTINEL,
)
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.dataset_schema import EvaluationResult
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._faithfulness import FaithfulnesswithHHEM
from ragas.metrics._nv_metrics import AnswerAccuracy
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
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
RAGAS_CONCURRENCY = int(os.getenv("RAGAS_CONCURRENCY", "6"))  # match gen server slot count
print(f"RAGAS_CONCURRENCY is set to {RAGAS_CONCURRENCY}", flush=True)
_current_sample_idx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_current_sample_idx", default=None
)
_call_counter = {"n": 0}
_retry_state = {"last_primary_n": 0, "retry_idx": 0}
_prompt_store: dict[int, str] = {}
_diag_state = {"printed": False, "concurrent_printed": False}

HHEM_DEVICE = os.getenv("HHEM_DEVICE", "cpu") # fallback
HHEM_BATCH_SIZE = int(os.getenv("HHEM_BATCH_SIZE", "16"))


class FaithfulnesswithHHEMPerChunk(FaithfulnesswithHHEM):
    """HHEM faithfulness with per-(claim, chunk) scoring and max-aggregation.

    Despite HHEM-2.1-Open having no 512-token input limit (as the old version
    did - https://huggingface.co/vectara/hallucination_evaluation_model)
    it was trained on short sequences. The stock ragas implementation
    concatenates all retrieved contexts into a single premise;

    Instead: score each claim against each chunk individually (each pair
    fits comfortably in 512 tokens), take the MAX across chunks per claim
    (a claim is supported iff any single chunk supports it), then MEAN
    over claims. The final averaging step matches the stock metric.
    """

    name: str = "faithfulness_with_hhem"

    async def _ascore(self, row, callbacks) -> float:
        import numpy as np

        assert self.llm is not None, "LLM is not set"
        statements = await self._create_statements(row, callbacks)
        statements = statements.statements
        if not statements:
            return float("nan")

        contexts = row.get("retrieved_contexts") or []
        if not contexts:
            return float("nan")

        pairs: list[tuple[str, str]] = []
        owners: list[int] = []
        for claim_idx, claim in enumerate(statements):
            for chunk in contexts:
                pairs.append((chunk, claim))
                owners.append(claim_idx)

        raw_scores: list[float] = []
        for batch in self._create_batch(pairs):
            preds = self.nli_classifier.predict(batch).cpu().detach()
            raw_scores.extend(preds.tolist())

        per_claim_max = [-1.0] * len(statements)
        for s, owner in zip(raw_scores, owners):
            if s > per_claim_max[owner]:
                per_claim_max[owner] = s

        rounded = [round(s) for s in per_claim_max]
        return float(np.mean(rounded))


_hhem_metric: FaithfulnesswithHHEMPerChunk | None = None
_hhem_lock = threading.Lock()  # guards lazy init across run_ragas worker threads

_RETRY_PROMPT_MARKER = "The output string did not satisfy"

def _get_hhem_metric() -> FaithfulnesswithHHEMPerChunk:
    global _hhem_metric
    if _hhem_metric is None:
        with _hhem_lock:
            if _hhem_metric is None:
                print(f"[ragas_eval] loading HHEM on {HHEM_DEVICE} (batch_size={HHEM_BATCH_SIZE}, per-chunk max-agg)", flush=True)
                _hhem_metric = FaithfulnesswithHHEMPerChunk(device=HHEM_DEVICE, batch_size=HHEM_BATCH_SIZE)
    return _hhem_metric


# Eagerly load HHEM at import time so the first batch of concurrent samples
# doesn't race on lazy init (and so download/load failures surface before eval starts).
_get_hhem_metric()


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


def _build_embeddings() -> Embeddings:
    # Native llama-server on LLAMACPP_EMB_BASE_URL (started in run_eval.sh
    # with --parallel 1, so n_seq_max=1 and n_ctx is honored exactly).
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
    has_contexts = bool(sample.get("contexts"))
    # Abstentions: relevancy of the canonical REJECTION_ANSWER to the question
    # is meaningless, so we skip it and report REJECTED_SENTINEL. Faithfulness /
    # HHEM still run (a rejection grounded in the contexts is a valid signal).
    is_rejected = bool(sample.get("rejected"))
    # AnswerAccuracy compares the response against the gold reference answer.
    # All loaders surface `reference_answer` at the top level of the row; for
    # MEDQA it is the canonical REJECTION_ANSWER, so a well-behaved abstention
    # scores high and a confident hallucination scores low. Guard against rows
    # with no reference so the metric is only requested when it can be scored.
    reference_answer = sample.get("reference_answer")
    has_reference_answer = reference_answer is not None and str(reference_answer).strip() != ""

    dataset = Dataset.from_dict({
        "question": [sample["query"]],
        "answer": [sample["answer"]],
        "contexts": [sample["contexts"]],
        # Legacy ragas HF column name that maps to the schema `reference` field
        # (same backward-compat mapping that makes question/answer/contexts work).
        "ground_truth": [reference_answer if has_reference_answer else ""],
    })

    ragas_llm = LangchainLLMWrapper(RagasJSONWrapper(_build_base_llm()))
    embeddings = _build_embeddings()

    try:
        # Faithfulness + HHEM require retrieved contexts; for the no_rag
        # variant we only run AnswerRelevancy and report the rest as None.
        # AnswerRelevancy is dropped entirely for rejected rows.
        metrics = []
        if has_contexts:
            metrics += [Faithfulness(), _get_hhem_metric()]
        if not is_rejected:
            metrics.append(AnswerRelevancy(strictness=3))
        # AnswerAccuracy is independent of contexts and of rejection: scoring an
        # abstention against the (REJECTION_ANSWER) gold is exactly the correctness
        # signal we want, so it runs whenever a reference answer is available.
        if has_reference_answer:
            metrics.append(AnswerAccuracy())

        # Nothing left to score (only reachable for a rejected row that has no
        # contexts and no reference; no_rag never rejects, so it always runs
        # relevancy).
        if not metrics:
            return {
                "ragas_faithfulness": None,
                "ragas_answer_relevancy": REJECTED_SENTINEL,
                "ragas_faithfulness_with_hhem": None,
                "ragas_answer_accuracy": None,
            }

        result = cast(
            EvaluationResult,
            evaluate(
                dataset,
                llm=ragas_llm,
                embeddings=embeddings,
                metrics=metrics,
                return_executor=False,
                raise_exceptions=raise_exceptions,
                run_config=RunConfig(timeout=RAGAS_TIMEOUT, max_retries=3, max_wait=60),
            ),
        )

        # ragas returns NaN for rows it couldn't score (e.g. empty claims
        # list); surface those as None so the output stays valid JSON. Rejected
        # rows didn't run relevancy at all → REJECTED_SENTINEL.
        if is_rejected:
            relev = REJECTED_SENTINEL
        else:
            relev_raw = result["answer_relevancy"][0]
            relev = None if math.isnan(relev_raw) else relev_raw

        if has_contexts:
            faith_raw = result["faithfulness"][0]
            faith_hhem_raw = result["faithfulness_with_hhem"][0]
            faith_out = None if math.isnan(faith_raw) else faith_raw
            faith_hhem_out = None if math.isnan(faith_hhem_raw) else faith_hhem_raw
        else:
            faith_out = NO_RAG_SENTINEL
            faith_hhem_out = NO_RAG_SENTINEL

        # AnswerAccuracy is keyed "nv_accuracy" in ragas. None when no reference
        # was available (metric not requested) or when ragas returned NaN.
        if has_reference_answer:
            acc_raw = result["nv_accuracy"][0]
            acc_out = None if math.isnan(acc_raw) else acc_raw
        else:
            acc_out = None

        return {
            "ragas_faithfulness": faith_out,
            "ragas_answer_relevancy": relev,
            "ragas_faithfulness_with_hhem": faith_hhem_out,
            "ragas_answer_accuracy": acc_out,
        }

    except Exception as e:
        import traceback
        sample_idx = _current_sample_idx.get()
        print(
            f"[ragas_eval] run_ragas FAILED for sample={sample_idx}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        traceback.print_exc()
        return {
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
            "ragas_faithfulness_with_hhem": None,
            "ragas_answer_accuracy": None,
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
