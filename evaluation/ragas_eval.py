import asyncio
import contextvars
import logging
import os
import re
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
from ragas.metrics._answer_correctness import AnswerCorrectness
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

RAGAS_TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "300"))

print(f"[ragas_eval] LLAMACPP_EVAL_MODEL = {LLAMACPP_EVAL_MODEL}", flush=True)

EVAL_DEBUG_LLM = os.getenv("EVAL_DEBUG_LLM", "1") == "1"
# Concurrency is NESTED, and the product is what lands on the llama-server slots:
#   RAGAS_CONCURRENCY samples in flight  x  RAGAS_MAX_WORKERS metric jobs each,
# where a "job" is one metric on one row (this dataset is always 1 row, so a job
# == a metric). The server exposes GEN_PARALLEL slots (3 by default, see
# slurm/run_eval.sh), so the product should not exceed it.
#
# It badly did: 6 samples x ragas' DEFAULT max_workers=16 put ~30 metric jobs on
# 3 slots. RunConfig.timeout is a per-job WALL-CLOCK budget (ragas wraps each
# metric's _ascore in asyncio.wait_for), so that queue wait — not the model — is
# what blew the deadline, and it killed metrics in order of how many sequential
# LLM calls they need: answer_correctness (3 calls) failed 98% of the time,
# faithfulness (2) 81%, while single-call answer_relevancy and the local-NLI
# faithfulness_with_hhem survived ~96%. Every progress bar read exactly
# [05:00<00:00] because all jobs start together and time out together.
#
# NOTE this bounds concurrent JOBS, not concurrent CALLS: a single metric may
# still fan out internally (AnswerRelevancy(strictness=3) gathers its 3 question
# generations in agenerate_prompt), so 3 jobs can mean up to ~9 in-flight calls.
RAGAS_CONCURRENCY = int(os.getenv("RAGAS_CONCURRENCY", "3"))  # = gen server slot count
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "1"))  # metrics run serially per sample
print(f"RAGAS_CONCURRENCY is set to {RAGAS_CONCURRENCY}, "
      f"RAGAS_MAX_WORKERS to {RAGAS_MAX_WORKERS} "
      f"({RAGAS_CONCURRENCY * RAGAS_MAX_WORKERS} concurrent metric jobs)", flush=True)
_current_sample_idx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_current_sample_idx", default=None
)
_call_counter = {"n": 0}
_retry_state = {"last_primary_n": 0, "retry_idx": 0}
_prompt_store: dict[int, str] = {}
_diag_state = {"printed": False, "concurrent_printed": False}

HHEM_DEVICE = os.getenv("HHEM_DEVICE", "cpu") # fallback
HHEM_BATCH_SIZE = int(os.getenv("HHEM_BATCH_SIZE", "16"))


# --- Per-metric failure capture ----------------------------------------------
# The trade-off ragas gives you is a bad one out of the box:
#
#   raise_exceptions=True  -> the exception propagates out of evaluate(), so ONE
#       metric's failure discards every other metric already scored for that
#       sample. A domino, and the reason we stopped using it.
#   raise_exceptions=False -> ragas records NaN for just the failed metric and
#       lets the rest finish (what we want), but the reason only ever appears as
#       a log line: `Exception raised in Job[i]: Type(msg)` on the
#       `ragas.executor` logger. Nothing reaches the result file, which is how
#       one run silently lost 2388 metric cells with no ragas_error to show.
#
# This handler takes the third option: keep raise_exceptions=False and tee those
# log records into a per-thread buffer, so run_ragas can attach the reason to the
# specific metric that failed. Job index maps to the metrics list positionally —
# ragas submits one job per metric per row (`for metric in metrics`) and this
# dataset is always a single row, so Job[i] is metrics[i].
_JOB_ERROR_RE = re.compile(r"Exception raised in Job\[(\d+)\]:\s*(\w+)\((.*)\)\s*$", re.S)
_job_error_local = threading.local()


def _job_error_buf() -> dict[int, str]:
    """Captured {job_index: "ExcType: message"} for the CURRENT thread.

    Thread-local because samples are evaluated concurrently (arun_ragas hands
    run_ragas to asyncio.to_thread), and ragas logs the failure from the same
    thread that runs evaluate() — so each sample sees only its own errors.
    """
    buf = getattr(_job_error_local, "buf", None)
    if buf is None:
        buf = _job_error_local.buf = {}
    return buf


class _JobErrorCapture(logging.Handler):
    """Tees ragas' swallowed per-job exceptions into the thread-local buffer."""

    def emit(self, record):
        try:
            m = _JOB_ERROR_RE.search(record.getMessage())
        except Exception:  # never let logging break the eval
            return
        if m:
            idx, exc_name, exc_msg = int(m.group(1)), m.group(2), m.group(3).strip()
            _job_error_buf()[idx] = f"{exc_name}: {exc_msg}" if exc_msg else exc_name


_ragas_executor_logger = logging.getLogger("ragas.executor")
_ragas_executor_logger.addHandler(_JobErrorCapture())
# ERROR passes the default WARNING threshold, but be explicit: if anything lowers
# the root level later, the capture must not go quiet.
if _ragas_executor_logger.level > logging.ERROR:
    _ragas_executor_logger.setLevel(logging.ERROR)

# ragas metric.name -> the key we publish it under, so a captured failure is
# labelled with the same name the result file and the analysis use.
_METRIC_NAME_TO_KEY = {
    "faithfulness": "ragas_faithfulness",
    "faithfulness_with_hhem": "ragas_faithfulness_with_hhem",
    "answer_relevancy": "ragas_answer_relevancy",
    "nv_accuracy": "ragas_answer_accuracy",
    "answer_correctness": "ragas_answer_correctness",
}


def _collect_metric_errors(metrics) -> dict[str, str]:
    """Map this thread's captured job errors onto metric keys, then clear them."""
    buf = _job_error_buf()
    out = {}
    for idx, msg in buf.items():
        if 0 <= idx < len(metrics):
            name = getattr(metrics[idx], "name", f"job_{idx}")
            out[_METRIC_NAME_TO_KEY.get(name, name)] = msg
        else:  # index outside this sample's metric list — keep it rather than drop
            out[f"job_{idx}"] = msg
    buf.clear()
    return out


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

# Grammar-level JSON enforcement gate. This marker is auto-appended by ragas to
# *every* PydanticPrompt (pydantic_prompt.py _generate_output_signature) and to
# its FixOutputFormat reprompt, but is ABSENT from the nv_metrics rating prompts
# (AnswerAccuracy), which end in "The rating is:" and are parsed by scanning the
# reply for a bare digit 0-4. We enable llama-server's json_object response
# format ONLY when this marker is present, so the structured metrics
# (Faithfulness, AnswerRelevancy, AnswerCorrectness) and their format-fix
# retries are grammar-locked to valid JSON, while a global json_object -- which
# would wrap nv_accuracy's rating in an object whose key digits corrupt its
# process_score scan -- is avoided. Note this is enforcement, not instruction:
# JSON_SYSTEM_PROMPT already *asks* for JSON on every call (soft, ignorable);
# response_format makes the server unable to emit syntactically invalid JSON.
_JSON_PROMPT_MARKER = "Please return the output in a JSON format that complies with"
_JSON_RESPONSE_FORMAT = {"type": "json_object"}

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


def _reference_context_ids(sample) -> list | None:
    """Gold context chunk ids for a synthetic sample, else None.

    Synthetic goldens carry the guideline chunks they were generated from in
    dataset_metadata.context_chunks, each with a `chunk_id` in the retriever's
    id space (see dataset/synthetic/build_contexts.py). No other dataset has
    gold contexts, so id-based context scoring only applies to synthetic rows.
    """
    dm = sample.get("dataset_metadata") or {}
    chunks = dm.get("context_chunks") or []
    ids = [c.get("chunk_id") for c in chunks if c.get("chunk_id") is not None]
    return ids or None


def _id_based_context_scores(sample) -> dict:
    """ID-based context precision/recall for synthetic samples (gold ids known).

    Pure set/rank arithmetic over retrieved-vs-gold chunk ids — no LLM, no
    embeddings, deterministic. The three fields are:

    - recall    = |gold ∩ retrieved| / |gold|      (rank-agnostic)
    - precision = |gold ∩ retrieved| / |retrieved| (rank-agnostic)
      Both are the exact formulas ragas' IDBasedContextRecall /
      IDBasedContextPrecision use; computed inline so the cheap deterministic
      metrics don't ride the LLM `evaluate()` path.
    - ap        = Average Precision over the retrieved ids in rank order,
      mirroring ragas' _calculate_average_precision. This is the rank-aware
      signal (the flat precision above ignores order); averaged across samples
      it is MAP@k.

    Every field is None when the sample has no gold ids (non-synthetic) or no
    retrieved ids (no_rag / retrieval error / result files predating the
    chunk-id wiring), so the key set stays stable across all rows.
    """
    none_scores = {
        "ragas_id_context_recall": None,
        "ragas_id_context_precision": None,
        "ragas_id_context_ap": None,
    }
    reference_ids = _reference_context_ids(sample)
    retrieved_ids = sample.get("retrieved_context_ids")
    if not reference_ids or not retrieved_ids:
        return none_scores

    # Compare as strings so int/str id representations always match (ragas does
    # the same). Recall/precision are set-based; AP keeps retrieval rank order.
    ref_set = {str(r) for r in reference_ids}
    ret_ids = [str(r) for r in retrieved_ids]
    ret_set = set(ret_ids)

    inter = ref_set & ret_set
    recall = len(inter) / len(ref_set)
    precision = len(inter) / len(ret_set)

    rel = [1 if rid in ref_set else 0 for rid in ret_ids]
    hits = sum(rel)
    if hits == 0:
        ap = 0.0
    else:
        ap = sum(
            (sum(rel[: i + 1]) / (i + 1)) * rel[i] for i in range(len(rel))
        ) / hits

    return {
        "ragas_id_context_recall": recall,
        "ragas_id_context_precision": precision,
        "ragas_id_context_ap": ap,
    }


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
        prompt_text = _prompt_to_text(prompt)
        messages = [
            SystemMessage(content=JSON_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ]
        # Grammar-lock structured-metric calls (and their fix-format retries) to
        # valid JSON; leave nv_accuracy's free-text rating prompt untouched.
        llm = self.llm
        if _JSON_PROMPT_MARKER in prompt_text:
            llm = llm.bind(response_format=_JSON_RESPONSE_FORMAT)
        ctx = self._log_prompt(prompt)
        try:
            result = await llm.ainvoke(messages)
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
    # Per-metric isolation. With raise_exceptions=True a single metric's failure
    # aborts the whole evaluate() and discards the OTHER metrics this sample
    # already scored — e.g. AnswerCorrectness running away to the token cap
    # (LengthFinishReasonError) or blowing the RAGAS_TIMEOUT watchdog would also
    # wipe a valid faithfulness/relevancy/accuracy. raise_exceptions=False lets
    # ragas record NaN for just the failed metric (surfaced as None by the
    # NaN-guards below) while the rest survive. EVAL_DEBUG_LLM still controls the
    # verbose prompt/response logging; it no longer makes failures fatal.
    raise_exceptions = False
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

    # ID-based context precision/recall (synthetic-only, deterministic, no LLM).
    # Computed once here and merged into every return so the key set is stable.
    id_ctx_scores = _id_based_context_scores(sample)

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
        # AnswerAccuracy and AnswerCorrectness are independent of contexts and of
        # rejection: scoring an abstention against the (REJECTION_ANSWER) gold is
        # exactly the correctness signal we want, so they run whenever a reference
        # answer is available. AnswerAccuracy is the NVIDIA dual-judge metric;
        # AnswerCorrectness is the classic factual-F1 + semantic-similarity metric
        # (needs both the LLM and embeddings, both already passed to evaluate()).
        if has_reference_answer:
            metrics.append(AnswerAccuracy())
            metrics.append(AnswerCorrectness())

        # Nothing left to score (only reachable for a rejected row that has no
        # contexts and no reference; no_rag never rejects, so it always runs
        # relevancy).
        if not metrics:
            return {
                "ragas_faithfulness": None,
                "ragas_answer_relevancy": REJECTED_SENTINEL,
                "ragas_faithfulness_with_hhem": None,
                "ragas_answer_accuracy": None,
                "ragas_answer_correctness": None,
                **id_ctx_scores,
                "ragas_metric_errors": None,
            }

        # Drop anything a previous sample left on this thread, so the errors we
        # collect after evaluate() belong to THIS sample's jobs only.
        _job_error_buf().clear()

        result = cast(
            EvaluationResult,
            evaluate(
                dataset,
                llm=ragas_llm,
                embeddings=embeddings,
                metrics=metrics,
                return_executor=False,
                raise_exceptions=raise_exceptions,
                run_config=RunConfig(
                    timeout=RAGAS_TIMEOUT,
                    max_retries=2,
                    max_wait=60,
                    # Without this ragas defaults to 16, firing every metric of
                    # this sample at the server at once; see RAGAS_MAX_WORKERS.
                    max_workers=RAGAS_MAX_WORKERS,
                ),
            ),
        )

        # Why any metric below came back NaN, per metric, harvested from the log
        # records ragas emits when raise_exceptions=False swallows a job. Empty
        # when everything scored. A NaN with no entry here is a legitimate
        # un-scorable row (e.g. no claims extracted), not a failure.
        metric_errors = _collect_metric_errors(metrics)
        if metric_errors:
            print(f"[ragas_eval] sample={_current_sample_idx.get()} metric failures: "
                  f"{metric_errors}", flush=True)

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

        # AnswerAccuracy is keyed "nv_accuracy" and AnswerCorrectness
        # "answer_correctness" in ragas. Both are None when no reference was
        # available (metrics not requested) or when ragas returned NaN.
        if has_reference_answer:
            acc_raw = result["nv_accuracy"][0]
            acc_out = None if math.isnan(acc_raw) else acc_raw
            corr_raw = result["answer_correctness"][0]
            corr_out = None if math.isnan(corr_raw) else corr_raw
        else:
            acc_out = None
            corr_out = None

        return {
            "ragas_faithfulness": faith_out,
            "ragas_answer_relevancy": relev,
            "ragas_faithfulness_with_hhem": faith_hhem_out,
            "ragas_answer_accuracy": acc_out,
            "ragas_answer_correctness": corr_out,
            **id_ctx_scores,
            # {metric_key: "ExcType: msg"} for the metrics ragas gave up on, or
            # None when the whole sample scored. Distinct from `ragas_error`
            # below, which marks the whole-sample failure path.
            "ragas_metric_errors": metric_errors or None,
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
            "ragas_answer_correctness": None,
            **id_ctx_scores,
            # Whatever per-metric reasons were captured before the whole sample
            # blew up — often the first domino, so keep them next to ragas_error.
            "ragas_metric_errors": _collect_metric_errors(locals().get("metrics") or []) or None,
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
