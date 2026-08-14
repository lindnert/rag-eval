"""Everything about the EVALUATED results, and the bridge to the pipeline signals.

There are two analysis entry points, and this is one of them:

  - ``analysis.rag_analysis``: the *raw* pipeline signals (generation confidence,
    hybrid retrieval scores, self-correction, how often the pipeline abstained)
    with no notion of quality.
  - ``analysis.eval_analysis`` (this file): everything that touches a metric or a
    judge — loading the evaluated JSON, the summary tables, metric validation,
    the paired variant comparisons, and the cross-cutting questions that need the
    pipeline signals alongside the metrics.

The primitives in the first two sections below — ``load``, ``metric_summary`` /
``means_by``, ``compare_variants``, ``health_report``, ``drop_eval_errors``, the
reason mining — used to live in a third module, ``analysis.analysis``. They read
metrics and judge prose like everything else here, so they were folded in;
``analysis.plots`` imports them from this module now.

Abstentions are split across the two modules on purpose. *Counting* them is a
pipeline fact and lives in ``rag_analysis`` (``abstention_summary`` and friends,
and the one shared ``_abstained`` detector this module imports). How the METRICS
behave on them is a scoring question and lives here: ``abstention_adjusted``
(every metric over all rows vs answered-only, so a low mean can be split into
"bad answers" vs "often refused to answer"), ``classify_metrics``'s
``na_rejected`` status, ``metric_agreement`` dropping abstentions from the
faithfulness pairs, and ``extremes_profile``'s abstain rate.

It answers six things, in the order __main__ runs them:

  0. What is in the cohort, and what has to come out of it. ``describe_cohort``
     before and after ``drop_eval_errors``, which excludes rows where a
     *technical failure* — a generation crash, a RAGAS raise, a DeepEval judge
     crash — rather than a weak answer produced the score; plus
     ``health_report``'s blunt "is any metric empty or constant?" checks. These
     run on the RAW frame, ahead of every exclusion: run afterwards they would be
     reporting on data the exclusion had already removed.

  1. Metric-computation sanity. Every metric cell is one of: genuinely SCORED, a
     legitimate NOT-APPLICABLE (``no_rag`` has no context, so no faithfulness/
     context metric; an abstention is not scored for answer relevancy; the
     ID-context metrics need reference contexts, i.e. the synthetic set only), or
     an actual ERROR (a RAGAS scorer raised, a DeepEval judge crashed). Reported
     per dataset x variant so the not-applicable cells are never miscounted as
     failures. ``prepare`` then nulls the RAGAS cells on RAGAS-errored rows so
     those rows drop out of RAGAS aggregates while their independent DeepEval
     scores survive.

  2. Metric validation, in two parts.

     (a) Is each metric discriminative on its own? ``metric_distribution`` gives
     min / q25 / median / q75 / max, spread, distinct-value count and the 0/1 rail
     fractions per metric — overall and per dataset x variant, since a metric can
     spread over one dataset and collapse on another. A metric whose IQR sits on a
     rail ranks nothing, and its mean in the results chapter means nothing either,
     so this comes before every aggregate below.

     (b) Do metrics that measure the same thing agree?
     ``ragas_answer_relevancy`` vs ``deepeval_relevance``; the three faithfulness
     metrics against each other — overall and per dataset / per variant
     (correlation, bias, agreement-within-tolerance). Plus
     ``deepeval_reason_consistency``: does each DeepEval score match the number
     its OWN ``*_reason`` prose states (an automatable internal-consistency check;
     semantic quality still needs eyeballing, for which ``analysis.reason_hits``
     samples rows).

  2c. How the variants actually compare — the results-chapter spine.
     ``compare_variants`` pairs each question's score under two variants and runs
     a Wilcoxon signed-rank test with a rank-biserial effect size and a bootstrap
     CI; ``means_by`` and ``abstention_adjusted`` give the flat and
     abstention-split means; ``decile_breakdown`` names the worst-scoring tenth
     of queries. Read AFTER 2a: a mean difference on a metric that ranks nothing
     is not a finding.

  3. Worst / best queries (the cross-link, moved here from ``rag_analysis``). On
     the rag+eval join, the bottom and top 10% by EVERY headline metric, each row
     carrying its pipeline signals (retrieval score, confidence, re-retrieval
     gain, abstention), with the dataset/variant split and signal means — to see
     which queries the system does worst/best on and whether a pattern explains it.

  4. Signal -> metric. Is the confidence gain from no_rag -> rag_sc reflected in
     the metrics (paired per-id deltas)? Do the faithfulness / context metrics
     track the hybrid retrieval score, and the HyDE re-retrieval gain?

Running from the console
------------------------
Run as a module from the repo root (the ``analysis.`` / ``common.`` imports need
the package on the path, so ``python analysis/eval_analysis.py`` will NOT work):

    python -m analysis.eval_analysis                       # newest of each prefix
    python -m analysis.eval_analysis EVAL_FILE             # explicit eval file
    python -m analysis.eval_analysis EVAL_FILE RAG_FILE    # explicit both

The two files are told apart by their name prefix (``evaluated_results`` vs
``rag_results``), so they may be passed in either order. With no RAG file the
matching one is found from the eval name's ``_from_<stamp>``
(see ``common.results_io``), falling back to the newest RAG file.

Every artifact goes to ``analysis/out/<eval-stem>/`` — the tables, the full
console report, the joined base table ``linked/`` (one row per id x variant,
pipeline signals + every evaluated metric) and the figures. Anything derived from
BOTH files is named for both, so a cross-linked plot says which rag run it was
paired with; see ``analysis.paths``. The figures themselves live in
``analysis.plots``.
"""

import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common.results_io import EVAL_PREFIX, RAG_PREFIX, latest_results
from analysis import analysis as ev
from analysis import paths
from analysis import plots
from analysis import rag_analysis as ra

# --- Metric applicability (which metrics a cell is even supposed to carry) ----
# Answer-only metrics need just the answer + reference, so they apply to EVERY
# variant (no_rag included). The two relevancy metrics are the exception: an
# abstention makes no claim to be relevant to the question, so it is left
# unscored ("rejected - ...").
ANSWER_METRICS = {"ragas_scores.ragas_answer_relevancy",
                  "ragas_scores.ragas_answer_accuracy",
                  "ragas_scores.ragas_answer_correctness",
                  "deepeval_scores.deepeval_relevance"}
RELEVANCY_METRICS = {"ragas_scores.ragas_answer_relevancy",
                     "deepeval_scores.deepeval_relevance"}
# Faithfulness + context-relevance need retrieved context, so they apply to
# rag / rag_sc only ("no rag - no retrieved contexts" on no_rag).
CONTEXT_METRICS = {"ragas_scores.ragas_faithfulness",
                   "ragas_scores.ragas_faithfulness_with_hhem",
                   "deepeval_scores.deepeval_faithfulness",
                   "deepeval_scores.deepeval_contextual_relevance"}
# The faithfulness subset, which is ALSO dropped on abstentions, for the same
# reason as relevancy: these score the ANSWER's claims against the context, and an
# abstention makes no claims, so what the scorer returns is its convention for the
# empty case rather than a measurement — see ``faithfulness_on_abstentions``.
# ``deepeval_contextual_relevance`` is deliberately not here: it scores the
# retrieved context against the QUESTION, which an abstention does not touch.
FAITHFULNESS_METRICS = {"ragas_scores.ragas_faithfulness",
                        "ragas_scores.ragas_faithfulness_with_hhem",
                        "deepeval_scores.deepeval_faithfulness"}
# ID-context retrieval metrics additionally need a gold reference-context set,
# which only the synthetic guideline questions carry -> everywhere else null is
# expected, not an error.
IDCTX_METRICS = {"ragas_scores.ragas_id_context_recall",
                 "ragas_scores.ragas_id_context_precision",
                 "ragas_scores.ragas_id_context_ap"}
CLASSIFIED_METRICS = sorted(ANSWER_METRICS | CONTEXT_METRICS | IDCTX_METRICS)

# Comparable metric pairs for validation. ``family`` groups them and flags the
# faithfulness ones, which are ill-defined on abstentions (so those rows can be
# dropped from the comparison).
METRIC_PAIRS = [
    ("relevancy: ragas vs deepeval", "ragas_scores.ragas_answer_relevancy",
     "deepeval_scores.deepeval_relevance", "relevancy"),
    ("faithfulness: ragas vs hhem", "ragas_scores.ragas_faithfulness",
     "ragas_scores.ragas_faithfulness_with_hhem", "faithfulness"),
    ("faithfulness: ragas vs deepeval", "ragas_scores.ragas_faithfulness",
     "deepeval_scores.deepeval_faithfulness", "faithfulness"),
    ("faithfulness: hhem vs deepeval", "ragas_scores.ragas_faithfulness_with_hhem",
     "deepeval_scores.deepeval_faithfulness", "faithfulness"),
]

# DeepEval metrics whose prose reason states the score, for the self-consistency check.
DEEPEVAL_REASON_METRICS = [
    ("deepeval_scores.deepeval_faithfulness",
     "deepeval_scores.deepeval_faithfulness_reason"),
    ("deepeval_scores.deepeval_contextual_relevance",
     "deepeval_scores.deepeval_contextual_relevance_reason"),
    ("deepeval_scores.deepeval_relevance",
     "deepeval_scores.deepeval_relevance_reason"),
]
_SCORE_RE = re.compile(r"score is\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

# Headline metrics ranked for the worst/best mining.
EXTREME_METRICS = [
    "ragas_scores.ragas_answer_correctness", "ragas_scores.ragas_answer_relevancy",
    "ragas_scores.ragas_answer_accuracy", "ragas_scores.ragas_faithfulness",
    "ragas_scores.ragas_faithfulness_with_hhem", "deepeval_scores.deepeval_faithfulness",
    "deepeval_scores.deepeval_relevance",
]
# Context/faithfulness metrics tested against retrieval quality.
RETRIEVAL_METRICS = [
    "ragas_scores.ragas_faithfulness", "ragas_scores.ragas_faithfulness_with_hhem",
    "deepeval_scores.deepeval_faithfulness", "deepeval_scores.deepeval_contextual_relevance",
    "ragas_scores.ragas_id_context_precision", "ragas_scores.ragas_id_context_recall",
]
# Pipeline signals shown alongside the worst/best rows.
DISPLAY_SIGNALS = ["retrieval_best", "gen_logprob_stats.mean", "reretrieval_gain", "rejected"]
NUM_SIGNALS = ["retrieval_best", "gen_logprob_stats.mean", "reretrieval_gain"]

_NAN = float("nan")


# --- (0) Evaluator-failure exclusion + blunt health checks -------------------

# Narrow, technical-failure patterns for DECIDING to drop a row — the evaluator
# LLM crashed, not "the answer was weak". Deliberately much stricter than
# FAILURE_PATTERNS (which is broad on purpose, for *eyeballing* leads): a
# DeepEval reason that legitimately says "no relevant context" is a real low
# score, not a failure, so it must NOT match here.
#
# Word boundaries are not decoration here. Unanchored, ``exception`` matches
# "the retrieval context is EXCEPTIONally helpful" — a judge praising the context
# — and that one substring was enough to discard two perfectly good rows
# (llmdrs_45/rag_sc, mmlu_37/rag_sc) from the 2026-08-12 run. ``rate ?limit`` has
# the same hazard inside "accuRATE LIMITs". Anchor anything that is also an
# English word fragment; ``outputparser`` still catches OutputParserException.
EVAL_ERROR_PATTERNS = [
    r"\bexception\b", r"\btraceback\b", r"failed to parse", r"parse error",
    r"outputparser", r"json ?decode", r"could ?n[o']t parse",
    r"\brate ?limit", r"timed ?out", r"\btimeout\b", r"api error",
    r"error (?:generating|calling|during|while)", r"\[llamacpp",
]


def _ragas_error_mask(df):
    c = "ragas_scores.ragas_error"
    if c in df:
        return df[c].notna() & df[c].astype(str).str.strip().ne("")
    return pd.Series(False, index=df.index)


def _nonempty(s):
    """Mask of cells holding actual text — not NaN, not blank."""
    t = s.astype("string")
    return t.notna() & t.str.strip().ne("")


def _deepeval_error_cells(df):
    """``{metric_col: mask}`` — rows where DeepEval recorded a crash for that
    metric, read from its ``deepeval_<metric>_error`` field.

    Cell-level on purpose. A crashed metric is one missing value, exactly like
    RAGAS's per-metric failures in ``ragas_metric_errors``; the row's other
    DeepEval metrics are independent of it and stay.
    """
    out = {}
    for c in df.columns:
        if not (c.startswith("deepeval_scores.") and c.endswith("_error")):
            continue
        metric = c[: -len("_error")]
        if metric in df:
            out[metric] = _nonempty(df[c])
    return out


def _deepeval_error_mask(df):
    """Rows to DROP for a DeepEval failure: the judge crashed and left the row with
    no usable DeepEval score at all.

    One crashed metric out of three is NOT a reason to drop the row — that is a
    cell failure (``_deepeval_error_cells``), handled the way every RAGAS
    per-metric failure is: the value is NaN, ``classify_metrics`` calls it
    ``error``, the row's other metrics survive. Dropping a whole row over one
    crashed cell would be a policy this module applies to no other evaluator.

    ``deepeval_*_error`` has not always existed. On a frame without those columns
    the original heuristic still applies: scan the judge's ``*_reason`` prose for
    ``EVAL_ERROR_PATTERNS``. It is a fallback, not a second opinion — where the
    fields DO exist the scrape was strictly worse than reading them (on the
    2026-08-12 run it found 0 of the 1 real crash and invented 2 that were only
    the word "exceptionally"), so it gets no vote once the evaluator states the
    answer itself.
    """
    cells = _deepeval_error_cells(df)
    if any(c.startswith("deepeval_scores.") and c.endswith("_error")
           for c in df.columns):
        if not cells:
            return pd.Series(False, index=df.index)
        crashed = pd.DataFrame(cells).any(axis=1)
        metrics = [c for c in score_cols(df) if c.startswith("deepeval_scores.")]
        usable = (pd.DataFrame({m: pd.to_numeric(df[m], errors="coerce").notna()
                                for m in metrics}).any(axis=1)
                  if metrics else pd.Series(False, index=df.index))
        return crashed & ~usable

    rx = re.compile("|".join(EVAL_ERROR_PATTERNS), re.IGNORECASE)
    cols = [c for c in df.columns
            if c.startswith("deepeval_scores.") and c.endswith("_reason")]
    m = pd.Series(False, index=df.index)
    for c in cols:
        t = df[c].astype("string").fillna("")
        m = m | (t.str.strip().ne("") & t.str.contains(rx))
    return m


def _generation_error_mask(df):
    """Rows whose answer text is a generation crash rather than an answer."""
    if "answer" not in df:
        return pd.Series(False, index=df.index)
    return df["answer"].fillna("").str.contains(r"\[LLAMACPP", regex=True)


def drop_eval_errors(df, mask_metric_level=False):
    """Exclude rows where a *technical failure* — not a low score — produced the
    metric, and report how many and of what type were dropped.

    Three failure signals, one per mask above: a generation failure (``answer``
    contains ``[LLAMACPP ...]``, so no real answer exists and the whole row is
    meaningless), a RAGAS failure (the row-level ``ragas_error``, which nulls the
    whole RAGAS block), or a DeepEval crash that left the row without a single
    usable score.

    All three are deliberately ROW-level catastrophes. Per-metric crashes —
    ``ragas_metric_errors.<metric>``, ``deepeval_<metric>_error`` — are not here
    and must not be: they cost one cell, the rest of the row is sound, and
    ``classify_metrics`` already reports them as ``error`` while the NaN keeps them
    out of that one metric's aggregates. Expect this to drop far fewer rows than
    ``metric_error_report`` shows error cells; the two count different things.

    Returns ``(clean_df, report)``. ``report`` is a dict with: ``n_before`` /
    ``n_after`` / ``n_dropped``; ``by_type`` (rows flagged per failure signal —
    can overlap); ``by_dataset`` and ``by_variant`` (dropped counts); and
    ``by_dataset_variant`` (the crosstab of what was dropped).

    ``mask_metric_level=False`` (default) drops the whole row on any failure — a
    clean uniform cohort for the paired tables. It is not implemented to mask per
    metric; the flag documents the alternative, which is to null only the failed
    evaluator's columns and keep the other evaluator's valid scores. ``prepare``
    below does precisely that for the RAGAS half, so __main__ runs it afterwards
    as a defensive second pass.
    """
    if mask_metric_level:
        raise NotImplementedError(
            "metric-level masking not implemented; call with mask_metric_level=False "
            "(whole-row drop). See docstring for the trade-off.")

    gen = _generation_error_mask(df)
    ragas = _ragas_error_mask(df)
    deepeval = _deepeval_error_mask(df)

    bad = gen | ragas | deepeval
    dropped = df[bad]
    report = {
        "n_before": len(df),
        "n_after": int((~bad).sum()),
        "n_dropped": int(bad.sum()),
        "by_type": pd.Series(
            {"generation": int(gen.sum()),
             "ragas": int(ragas.sum()),
             "deepeval": int(deepeval.sum())}, dtype="int64"),
        "by_dataset": (dropped["source_dataset"].value_counts()
                       if "source_dataset" in dropped else pd.Series(dtype="int64")),
        "by_variant": (dropped["variant"].astype("object").value_counts()
                       if "variant" in dropped else pd.Series(dtype="int64")),
        "by_dataset_variant": (ev.describe_cohort(dropped) if len(dropped)
                               else pd.DataFrame()),
    }
    return df[~bad].copy(), report


def health_report(df, source=None):
    """Quick 'is something broken?' checks. Prints; ``source`` (the results
    path/name it ran on) only adds the provenance header lines.

    Surfaces the failure modes that silently poison aggregates: generation errors
    baked into the answer text, RAGAS scorer errors, metrics that never produced
    a value, degenerate (constant) metrics, and faithfulness scored on abstentions
    (ill-defined — an abstention makes no claim to be faithful to a context).

    Deliberately blunt, and deliberately run BEFORE ``drop_eval_errors``: the
    per-cell version of the same question, which separates a real error from a
    legitimate not-applicable, is ``metric_error_report`` below.

    Returns the flags dict. The printed lines are saved along with the rest of the
    run's console output by ``analysis.paths.capture``, which __main__ wraps around
    everything — this function does not write its own file.
    """
    flags = {}
    lines = []
    emit = lines.append

    emit(f"health_report: {len(df)} rows")
    if source is not None:
        emit(f"  source: {source}")
        emit(f"  generated: {datetime.now().isoformat(timespec='seconds')}")

    gen_err = _generation_error_mask(df)
    flags["answer_generation_errors"] = int(gen_err.sum())
    emit(f"  answer generation errors ([LLAMACPP ...]): {int(gen_err.sum())}")

    ragas_err = _ragas_error_mask(df)
    flags["ragas_errors"] = int(ragas_err.sum())
    emit(f"  ragas scorer errors: {int(ragas_err.sum())}")

    empty, degenerate = [], []
    for m in ev.metric_cols(df):
        s = pd.to_numeric(df[m], errors="coerce")
        if s.notna().sum() == 0:
            empty.append(m)
        elif s.nunique(dropna=True) == 1:
            degenerate.append((m, s.dropna().iloc[0]))
    for m in empty:
        emit(f"  [BROKEN] {m}: no numeric values at all")
    for m, v in degenerate:
        emit(f"  [DEGENERATE] {m}: constant = {v}")
    flags["empty_metrics"] = empty
    flags["degenerate_metrics"] = degenerate

    rej = ra._abstained(df)
    # score_cols, not metric_cols: the latter still carries the *_verdicts.* tallies,
    # and "n_verdicts scored on 199 abstained rows" is a check on a claim count, not
    # on a faithfulness score.
    faith_cols = [c for c in score_cols(df) if "faithful" in c]
    for c in faith_cols:
        scored = int(pd.to_numeric(df.loc[rej, c], errors="coerce").notna().sum())
        if scored:
            emit(f"  [CHECK] {c} scored on {scored} abstained rows in the raw file "
                 f"(faithfulness is ill-defined on abstentions; those cells are "
                 f"excluded downstream — see the na_rejected column below)")
    flags["faithfulness_on_rejections"] = {
        c: int(pd.to_numeric(df.loc[rej, c], errors="coerce").notna().sum())
        for c in faith_cols
    }

    print("\n".join(lines))
    return flags


# --- (1) Metric-computation sanity -------------------------------------------


def classify_metrics(df):
    """Long table (id, variant, source_dataset, metric, status) tagging every
    metric cell as one of: ``scored`` / ``na_no_context`` (context metric on
    no_rag) / ``na_no_reference`` (ID-context metric off the synthetic set) /
    ``na_rejected`` (relevancy OR faithfulness metric on an abstention — both
    score the answer, and an abstention is not one) / ``error`` (applicable but
    missing — a scorer raised or the value never landed).

    The status is derived from variant + dataset + the ``rejected`` flag, NOT from
    the sentinel strings, so it is robust to wording changes. This is the single
    source of truth for the error report.
    """
    rej = ra._abstained(df)
    is_norag = df["variant"].astype(str).eq("no_rag")
    is_synth = (df["source_dataset"].eq("synthetic_guidelines")
                if "source_dataset" in df else pd.Series(False, index=df.index))

    frames = []
    for m in CLASSIFIED_METRICS:
        if m not in df:
            continue
        val = pd.to_numeric(df[m], errors="coerce")
        # Narrowest reason first, broadest last, because each mask overwrites the
        # one before: a faithfulness cell on a no_rag row reads na_no_context (no
        # context existed at all), not na_rejected, even if that row also abstained.
        status = pd.Series("scored", index=df.index, dtype="object")
        if m in RELEVANCY_METRICS or m in FAITHFULNESS_METRICS:
            status = status.mask(rej, "na_rejected")
        if m in CONTEXT_METRICS or m in IDCTX_METRICS:
            status = status.mask(is_norag, "na_no_context")
        if m in IDCTX_METRICS:
            status = status.mask(~is_synth, "na_no_reference")
        # Cells still marked "scored" are the ones a value is EXPECTED for:
        # present => scored, absent => error (a RAGAS raise, if any, is the why).
        expected = status.eq("scored")
        status = status.mask(expected & val.isna(), "error")
        frames.append(pd.DataFrame({
            "id": df["id"], "variant": df["variant"].astype(str),
            "source_dataset": df.get("source_dataset"), "metric": m, "status": status,
        }))
    if not frames:  # no metric columns at all (e.g. a raw rag file passed by mistake)
        return pd.DataFrame(columns=["id", "variant", "source_dataset", "metric", "status"])
    return pd.concat(frames, ignore_index=True)


def metric_error_reasons(df):
    """Long table ``(id, variant, source_dataset, metric, error, error_type)`` of
    every crash the evaluators recorded per metric cell.

    Both evaluators write the exception next to the metric it killed — RAGAS in
    ``ragas_scores.ragas_metric_errors.<metric>``, DeepEval in
    ``deepeval_scores.<metric>_error`` — so an ``error`` cell in
    ``classify_metrics`` almost always has a stated cause. ``error_type`` is the
    exception class (the text before the first colon), which is what separates a
    fixable config problem from model behaviour: ``AssertionError: LLM is not set``
    means the metric was never wired up, ``LengthFinishReasonError`` means the
    judge ran out of output budget.

    Empty when the file predates the error fields, or when ``analysis.load`` has
    coerced them away — which it did until ``error_cols`` was carved out of
    ``metric_cols``.
    """
    keys = [c for c in ("id", "variant", "source_dataset") if c in df]
    frames = []
    for c in ev.error_cols(df):
        # Skip the two fields that are not per-metric: the ``ragas_metric_errors``
        # container itself, and the row-level ``ragas_error`` (whose failure kills
        # the whole RAGAS block and is reported on its own, above).
        if c.endswith("ragas_metric_errors") or c == "ragas_scores.ragas_error":
            continue
        if ".ragas_metric_errors." in c:
            metric = "ragas_scores." + c.split(".ragas_metric_errors.")[1]
        elif c.endswith("_error"):
            metric = c[: -len("_error")]
        else:
            continue
        hit = _nonempty(df[c])
        if not hit.any():
            continue
        sub = df.loc[hit, keys].copy()
        if "variant" in sub:  # str, matching classify_metrics, so the two tables join
            sub["variant"] = sub["variant"].astype(str)
        sub["metric"] = metric
        sub["error"] = df.loc[hit, c].astype(str)
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=keys + ["metric", "error", "error_type"])
    out = pd.concat(frames, ignore_index=True)
    out["error_type"] = out["error"].str.split(":").str[0].str.strip()
    return out


def faithfulness_on_abstentions(df):
    """``(table, info)``: what each faithfulness scorer put on the abstained rows.

    The evidence behind the ``na_rejected`` cells the status table shows for the
    faithfulness metrics, so the exclusion is argued in the report rather than
    asserted. Faithfulness asks whether the ANSWER's claims are entailed by the
    context; an abstention makes no claims, so "all zero claims are entailed" is
    vacuously true and each scorer is free to invent its own convention for the
    empty case. They do, and they disagree almost maximally — on the 2026-07-29 run
    RAGAS scores 98.5% of abstentions 0.0 (a refusal counted as a hallucination,
    wrong even on RAGAS's own definition) while HHEM and DeepEval score 83-95% of
    the SAME rows 1.0. Since the refusal text is identical across those rows, none
    of that spread can be a property of the run.

    One row per faithfulness metric: how many abstained cells carried a value,
    their mean, the share on each rail, and the answered-row mean for contrast.
    ``info["n_texts"]`` counts the distinct abstention answer strings — a handful
    means the rows really are canonical refusals; many would mean ``_abstained``
    is catching partial answers, which DO carry claims and must not be discarded
    wholesale, and the exclusion would have to be reconsidered.

    Run this on the RAW frame: ``prepare`` nulls exactly these cells, so afterwards
    there is nothing left to describe.
    """
    rej = ra._abstained(df)
    cols = sorted(c for c in FAITHFULNESS_METRICS if c in df)
    rows = {}
    for m in cols:
        v = pd.to_numeric(df[m], errors="coerce")
        a, b = v[rej].dropna(), v[~rej].dropna()
        rows[m.split(".")[-1]] = {
            "n_abstained": len(a),
            "mean_abstained": a.mean() if len(a) else _NAN,
            "frac_0": float((a == 0).mean()) if len(a) else _NAN,
            "frac_1": float((a == 1).mean()) if len(a) else _NAN,
            "n_answered": len(b),
            "mean_answered": b.mean() if len(b) else _NAN,
        }
    tab = pd.DataFrame(rows).T
    if len(tab):
        tab["n_abstained"] = tab["n_abstained"].astype(int)
        tab["n_answered"] = tab["n_answered"].astype(int)
        tab.index.name = "metric"

    texts = None
    if "answer" in df:
        texts = df.loc[rej, "answer"].fillna("").str.strip()
    info = {
        "n_abstained_rows": int(rej.sum()),
        "n_texts": int(texts.nunique()) if texts is not None else None,
        # The gap between the most and least generous convention, on identical text.
        "spread": (float(tab["mean_abstained"].max() - tab["mean_abstained"].min())
                   if len(tab) and tab["mean_abstained"].notna().any() else _NAN),
    }
    return tab, info


def metric_error_report(df, cls=None, source=None):
    """Prints the metric-computation sanity report: how many metric values are
    genuinely errors vs legitimately not applicable, broken down by dataset x
    variant. Returns a flags dict. ``source`` only adds the provenance header
    lines — the whole console report is saved by ``analysis.paths.capture``.

    Sections: RAGAS scorer errors (count, error-type, dataset x variant);
    DeepEval judge crashes (count, dataset x variant); the per-metric status
    breakdown (scored / na_* / error); and the actionable list of error cells by
    dataset x variant and by metric.
    """
    cls = classify_metrics(df) if cls is None else cls
    flags, lines = {}, []
    emit = lines.append

    emit(f"metric_error_report: {len(df)} rows, "
         f"{df['id'].nunique() if 'id' in df else '?'} ids")
    if source is not None:
        emit(f"  source: {source}")
        emit(f"  generated: {pd.Timestamp.now().isoformat(timespec='seconds')}")

    # RAGAS scorer errors (the whole ragas block is nulled on the row).
    rerr = _ragas_error_mask(df)
    flags["ragas_error_rows"] = int(rerr.sum())
    emit(f"\n  RAGAS scorer errors: {int(rerr.sum())} rows "
         f"(their entire ragas block is null and drops from ragas aggregates)")
    if rerr.any():
        etype = (df.loc[rerr, "ragas_scores.ragas_error"].astype(str)
                 .str.split(":").str[0].str.strip().value_counts())
        emit("    by error type:")
        for k, v in etype.items():
            emit(f"      {k}: {v}")
        emit("    by dataset x variant:")
        emit(_indent(pd.crosstab(df.loc[rerr, "source_dataset"],
                                 df.loc[rerr, "variant"].astype(str)).to_string()))

    # DeepEval crashes severe enough to cost the whole row (a crash that killed only
    # one metric is a cell, and shows up in the error table at the end instead).
    derr = _deepeval_error_mask(df)
    flags["deepeval_error_rows"] = int(derr.sum())
    emit(f"\n  DeepEval judge crashes leaving the row with no usable score: "
         f"{int(derr.sum())} rows")
    if derr.any():
        emit(_indent(pd.crosstab(df.loc[derr, "source_dataset"],
                                 df.loc[derr, "variant"].astype(str)).to_string()))

    # Per-metric status breakdown. Every cell carries exactly ONE na_* reason — the
    # broadest applicable one, since the masks overwrite in order (an ID-context cell
    # on a non-synthetic no_rag row reads na_no_reference, not na_no_context). That
    # split makes the individual columns hard to read against an expected count, so
    # na_total sums them: error + na_total + scored == total, per metric.
    present = set(cls["metric"])
    piv = (cls.pivot_table(index="metric", columns="status", values="id",
                           aggfunc="count", fill_value=0)
           .reindex([m for m in CLASSIFIED_METRICS if m in present]))
    # Left to right in the order the table is read: the one column worth acting on
    # (error) first, then WHY the rest were legitimately not scored, how many that is,
    # then what was scored and the row count it all has to add up to. The na_* reasons
    # run in mask-application order, so the broadest comes first; a reason added later
    # and not listed here still prints, after the known ones.
    na_order = ["na_no_context", "na_no_reference", "na_rejected"]
    na_cols = ([c for c in na_order if c in piv.columns]
               + sorted(c for c in piv.columns
                        if str(c).startswith("na_") and c not in na_order))
    piv = piv.reindex(columns=["error"] + na_cols + ["scored"], fill_value=0)
    piv.insert(1 + len(na_cols), "na_total",
               piv[na_cols].sum(axis=1).astype(int) if na_cols else 0)
    piv["total"] = piv["error"] + piv["na_total"] + piv["scored"]
    piv.index = [m.split(".")[-1] for m in piv.index]
    # The corner label is the pivot's ``columns.name`` ("status"), which sits directly
    # above a column of metric names and reads as if it labelled them. Name it for what
    # is under it.
    piv.columns.name = "metric"
    flags["status_by_metric"] = piv
    emit("\n  per-metric cell status — the actionable column first, then why a cell was "
         "NOT scored and the totals (error | na_* reasons | na_total | scored | total); "
         "each cell gets ONE na reason, the broadest one that applies, and "
         "error + na_total + scored == total:")
    emit(_indent(piv.to_string()))

    # Immediately after the table because it is the evidence for one column of it:
    # why the faithfulness metrics carry na_rejected rather than a score.
    fa, fa_info = faithfulness_on_abstentions(df)
    flags["faithfulness_on_abstentions"] = fa
    flags["abstention_texts"] = fa_info["n_texts"]
    if len(fa) and fa_info["n_abstained_rows"]:
        emit(f"\n  why faithfulness is na_rejected: what the scorers had put on the "
             f"{fa_info['n_abstained_rows']} abstained rows")
        if fa_info["n_texts"] is not None:
            emit(f"    those rows use {fa_info['n_texts']} distinct answer string(s), i.e. "
                 f"they are textually near-identical, so any spread below is the scorer's "
                 f"convention for an answer with no claims — not a property of this run:")
        emit(_indent(fa.round(3).to_string(), 4))
        if fa_info["spread"] == fa_info["spread"]:  # not NaN
            emit(f"    the conventions disagree by {fa_info['spread']:.3f} on that identical "
                 f"text (a refusal is zero claims: RAGAS reads that as 0 = hallucinated, "
                 f"HHEM/DeepEval as 1 = nothing false asserted). Keeping the cells would "
                 f"report the choice of scorer as faithfulness, and would let a system raise "
                 f"its score by abstaining more; they are excluded from every mean, "
                 f"correlation, paired test and figure below. The abstention RATE is reported "
                 f"separately, at the top of this report.")

    # The actionable bit: which cells are true errors, and — since both evaluators
    # record the exception next to the metric it killed — WHY. Note this counts
    # CELLS, not rows: a cell error costs one metric on one row, which is why 75 of
    # them can coexist with 2 dropped rows. The two sets need not overlap at all.
    err = cls[cls["status"] == "error"]
    flags["error_cells"] = int(len(err))
    emit(f"\n  error cells (applicable but missing): {len(err)} "
         f"— cells, not rows; each costs ONE metric on one row and is excluded from "
         f"that metric only, so every metric has its own effective n (the 'scored' "
         f"column above)")
    if len(err):
        emit("    by dataset x variant:")
        emit(_indent(pd.crosstab(err["source_dataset"], err["variant"]).to_string()))
        emit("    by metric:")
        emit(_indent(err["metric"].map(lambda m: m.split(".")[-1])
                     .value_counts().to_string()))

        # The recorded cause, joined back onto the error cells. An error type that
        # is a config fault (AssertionError: LLM is not set) is fixable by rerunning
        # the evaluation; one that is model behaviour (LengthFinishReasonError) is
        # not, and has to be reported as a coverage limit instead.
        reasons = metric_error_reasons(df)
        flags["error_reasons"] = reasons
        if len(reasons):
            keys = [c for c in ("id", "variant", "metric") if c in err]
            known = err.merge(reasons, on=keys, how="left", suffixes=("", "_r"))
            explained = known["error_type"].notna()
            emit(f"    recorded cause ({int(explained.sum())} of {len(err)} explained "
                 f"by the evaluator's own error field):")
            emit(_indent(pd.crosstab(
                known.loc[explained, "metric"].map(lambda m: m.split(".")[-1]),
                known.loc[explained, "error_type"]).to_string(), 6))
            if (~explained).any():
                emit(f"      {int((~explained).sum())} error cell(s) with no recorded "
                     f"cause — the value simply never landed")
        else:
            emit("    no recorded cause available: this file predates the per-metric "
                 "error fields (ragas_metric_errors / deepeval_*_error)")

    print("\n".join(lines))
    return flags


def prepare(df):
    """Return ``(clean_df, report)`` with the two cell-level exclusions applied.

    1. RAGAS metric cells nulled on RAGAS-errored rows, so those rows leave RAGAS
       aggregates while keeping their independent DeepEval scores. (The metric-level
       masking ``analysis.drop_eval_errors`` deliberately punts on.) They are already
       NaN after ``analysis.load``; this makes the exclusion explicit and defensive.
    2. Faithfulness cells nulled on abstentions, matching the ``na_rejected`` status
       ``classify_metrics`` gives the same cells. This is the one place the exclusion
       has to happen: everything downstream — distributions, means by variant, paired
       comparisons, agreement, deciles, every figure — reads this frame, so nulling
       here is what makes "excluded everywhere" true rather than a claim repeated per
       call site. ``faithfulness_on_abstentions`` (run on the RAW frame, before this)
       is the diagnosis; this is the treatment.

    Reports what it touched, in both cases.
    """
    clean = df.copy()
    rerr = _ragas_error_mask(clean)
    ragas_cols = [c for c in ev.metric_cols(clean) if c.startswith("ragas_scores.")]
    clean.loc[rerr, ragas_cols] = np.nan

    rej = ra._abstained(clean)
    faith_cols = sorted(c for c in FAITHFULNESS_METRICS if c in clean)
    n_faith_cells = 0
    if faith_cols and rej.any():
        n_faith_cells = int(clean.loc[rej, faith_cols].notna().to_numpy().sum())
        clean.loc[rej, faith_cols] = np.nan

    report = {
        "n_ragas_error_rows": int(rerr.sum()),
        "by_dataset_variant": (pd.crosstab(clean.loc[rerr, "source_dataset"],
                                           clean.loc[rerr, "variant"].astype(str))
                               if rerr.any() else pd.DataFrame()),
        "n_abstained_rows": int(rej.sum()),
        "n_faithfulness_cells_masked": n_faith_cells,
        "faithfulness_cols_masked": faith_cols,
    }
    return clean, report


# --- (2a) Metric validation: is each metric discriminative on its own? -------

def score_cols(df):
    """The real 0-1 score columns — see ``analysis.score_cols``, which is now the
    single definition.

    Kept as a name here because this module and ``plots`` both need the answer and
    kept diverging when each filtered ``metric_cols`` for itself: the error fields
    were dropped here but not in the rail plot, which drew twelve verdict tallies
    as if they were metrics.
    """
    return ev.score_cols(df)


def metric_distribution(df, by=None, metrics=None):
    """Per-metric location and spread — ``ev.metric_summary`` overall, or once per
    ``by`` group with the group as the leading index level.

    This is the "does this metric separate anything?" table that must be read
    before any mean in the results chapter is trusted. Beyond mean/std it carries
    min / q25 / median / q75 / max, ``n_unique`` and the 0/1 rail fractions, so the
    three failure modes are visible at a glance: a metric pinned at one rail
    (frac_one ~ 1), a metric whose IQR is 0 while min < max (all mass on one value,
    signal only in a thin tail), and a metric emitting too few distinct levels to
    rank queries at all.

    Grouping matters because discriminativeness is not a property of the metric
    alone: a metric can spread nicely over NGQA and collapse to 1.0 on MMLU, and a
    single pooled row hides that. ``by=["source_dataset", "variant"]`` is the
    default cut used by ``__main__``.

    Note on ``coverage`` within a group: it is scored-rows / group-rows, so the
    legitimately not-applicable cells (faithfulness on ``no_rag``, the ID-context
    metrics off the synthetic set) read as low coverage here by design. Which
    zeros are expected and which are real failures is ``metric_error_report``'s
    job, not this table's.
    """
    metrics = metrics or score_cols(df)
    if by is None:
        return ev.metric_summary(df, metrics)
    keys = [by] if isinstance(by, str) else list(by)
    frames = {}
    for g, sub in df.groupby(keys, observed=True):
        if not len(sub):
            continue
        frames[g if isinstance(g, tuple) else (g,)] = ev.metric_summary(sub, metrics)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames.values(), keys=frames.keys(),
                    names=keys + ["metric"]).reset_index()
    for k in keys:
        out[k] = out[k].astype(str)
    return out.set_index(keys + ["metric"])


def means_by(df, by="source_dataset", metrics=None):
    """Mean of every metric grouped by a column (default ``source_dataset``).

    Rows = groups, columns = metrics. Also appends an ``n`` column (rows per
    group) so a low group mean built on a handful of queries is obvious. Pass
    ``by="variant"`` for the per-variant table, or a list for a crosstab.

    The flat companion to ``metric_distribution``: read that one first, since a
    mean here is only interpretable for a metric that spreads.
    """
    metrics = metrics or score_cols(df)
    sub = df.copy()
    sub[metrics] = sub[metrics].apply(pd.to_numeric, errors="coerce")
    g = sub.groupby(by, observed=True)[metrics].mean()
    g.insert(0, "n", sub.groupby(by, observed=True).size())
    return g


# --- (2b) Metric validation: do comparable metrics agree? --------------------

def _corr(x, y):
    """(n, pearson, spearman) over the rows where both are numeric."""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n <= 2:
        return n, _NAN, _NAN
    return n, x[m].corr(y[m]), x[m].corr(y[m], method="spearman")


def metric_agreement(df, pairs=METRIC_PAIRS, by=None, tol=0.2,
                     drop_rejections_for_faith=True):
    """How closely each comparable metric pair agrees, overall or per ``by`` group.

    For every pair, on the rows where BOTH are numeric: ``n``; ``pearson`` /
    ``spearman`` (do they rank queries the same way?); ``mean_diff`` = first minus
    second (systematic bias, e.g. DeepEval scoring higher); ``frac_within`` (share
    of rows agreeing to within ``tol``). Read mean_diff and frac_within together:
    mean_diff averages signed gaps, so disagreements in both directions cancel and
    only frac_within exposes them. Low correlation with a high frac_within means
    both hug the same rail (agree by default, not by signal). Faithfulness pairs
    drop abstentions by default (faithfulness is ill-defined on a non-answer).

    ``by="source_dataset"`` / ``by="variant"`` gives the per-group table.
    """
    rej = ra._abstained(df)

    def one(sub, sub_rej, group):
        rows = []
        for name, a, b, family in pairs:
            if a not in sub or b not in sub:
                continue
            x = pd.to_numeric(sub[a], errors="coerce")
            y = pd.to_numeric(sub[b], errors="coerce")
            keep = x.notna() & y.notna()
            if family == "faithfulness" and drop_rejections_for_faith:
                keep = keep & ~sub_rej
            n = int(keep.sum())
            diff = x[keep] - y[keep]
            pear = spear = _NAN
            if n > 2:
                pear = x[keep].corr(y[keep])
                spear = x[keep].corr(y[keep], method="spearman")
            rows.append({
                "pair": name, "group": group, "family": family, "n": n,
                "pearson": pear, "spearman": spear,
                "mean_diff": diff.mean() if n else _NAN,
                "frac_within": float((diff.abs() <= tol).mean()) if n else _NAN,
            })
        return pd.DataFrame(rows)

    if by is None:
        return one(df, rej, "overall")
    frames = [one(sub, rej.loc[sub.index], str(g))
              for g, sub in df.groupby(by, observed=True)]
    cols = ["pair", "group", "family", "n", "pearson", "spearman",
            "mean_diff", "frac_within"]
    return pd.concat(frames, ignore_index=True)[cols]


def deepeval_reason_consistency(df, tol=0.011):
    """Does each DeepEval score match the number its own ``*_reason`` prose states?

    DeepEval reasons open with "The score is X.XX because …"; this extracts that
    X.XX (rounded to 2 dp in the prose) and compares it to the recorded score.
    Returns ``(summary, mismatches)``: per metric ``n_scored`` / ``n_reason_number``
    (prose carried a parseable number) / ``n_no_number`` / ``n_match`` (|Δ| ≤ tol)
    / ``n_mismatch`` / ``match_rate``; and the individual mismatching rows.

    A high match_rate means the prose is internally consistent with the field —
    necessary but not sufficient for the reason to be a *good* justification
    (semantic quality still needs manual review; ``analysis.reason_hits`` samples
    rows for that). A mismatch is a genuine red flag: the judge's own explanation
    contradicts the number it emitted.
    """
    rows, mism = [], []
    for mcol, rcol in DEEPEVAL_REASON_METRICS:
        if mcol not in df or rcol not in df:
            continue
        val = pd.to_numeric(df[mcol], errors="coerce")
        scored = val.notna()
        stated = pd.to_numeric(
            df[rcol].astype("string").str.extract(_SCORE_RE, expand=False), errors="coerce")
        has_num = scored & stated.notna()
        delta = (val - stated).abs()
        match = has_num & (delta <= tol)
        bad = has_num & (delta > tol)
        rows.append({
            "metric": mcol.split(".")[-1],
            "n_scored": int(scored.sum()),
            "n_reason_number": int(has_num.sum()),
            "n_no_number": int((scored & ~stated.notna()).sum()),
            "n_match": int(match.sum()),
            "n_mismatch": int(bad.sum()),
            "match_rate": round(match.sum() / has_num.sum(), 3) if has_num.sum() else _NAN,
        })
        for i in df.index[bad]:
            mism.append({
                "id": df.at[i, "id"], "variant": str(df.at[i, "variant"]),
                "source_dataset": df.at[i, "source_dataset"],
                "metric": mcol.split(".")[-1],
                "actual": round(float(val.at[i]), 3),
                "stated": round(float(stated.at[i]), 3),
                "reason": str(df.at[i, rcol])[:160],
            })
    return pd.DataFrame(rows), pd.DataFrame(mism)


# --- (2c) Judge prose: is the reason text signalling failure? ----------------

# Phrases that, when they appear in a judge's *_reason prose or in ragas_error,
# usually mean the score is a scorer/judge failure rather than a real rating of
# the answer. Deliberately broad; treat the output as leads to eyeball, not a
# verdict. Word boundaries keep "error" from firing inside unrelated words.
# The narrow counterpart used to DECIDE exclusions is EVAL_ERROR_PATTERNS.
FAILURE_PATTERNS = [
    r"no context", r"no relevant", r"not enough", r"insufficient",
    r"could ?n[o']t", r"unable to", r"can ?not", r"can['’]t",
    r"invalid json", r"failed to parse", r"parse error", r"\berror\b",
    r"\bn/?a\b", r"not applicable", r"no answer", r"\bempty\b", r"exception",
]


def reason_cols(df):
    """The judge prose worth mining: the DeepEval ``*_reason`` fields plus
    ``ragas_scores.ragas_error``.

    Deliberately excludes ``sc_metadata.*`` fields (``rejection_reason``,
    ``finish_reason``) — those are pipeline metadata explaining the model's own
    behaviour, not an evaluator justifying a score, so failure-phrase matching
    there is meaningless (a rejection reason legitimately says "insufficient").
    """
    cols = [c for c in df.columns
            if c.startswith("deepeval_scores.") and c.endswith("_reason")]
    if "ragas_scores.ragas_error" in df:
        cols.append("ragas_scores.ragas_error")
    return cols


def mine_reasons(df, patterns=FAILURE_PATTERNS, by=None):
    """Count failure-phrase hits in each reason/error column.

    Returns one row per reason column with ``n_nonempty`` (rows that carry any
    prose) and ``n_hits`` / ``hit_rate`` (share of those matching a failure
    phrase). A high hit_rate on a metric's reason column means that metric's
    judge is failing often, so its scores are untrustworthy. Pass ``by`` (e.g.
    ``"source_dataset"``) to get the hit_rate broken down per group instead.
    """
    rx = re.compile("|".join(patterns), re.IGNORECASE)
    cols = reason_cols(df)

    def _stats(sub):
        out = {}
        for c in cols:
            text = sub[c].astype("string").fillna("").str.strip()
            nonempty = text.ne("")
            hits = nonempty & text.str.contains(rx)
            n = int(nonempty.sum())
            out[c] = {
                "n_nonempty": n,
                "n_hits": int(hits.sum()),
                "hit_rate": round(hits.sum() / n, 3) if n else float("nan"),
            }
        return pd.DataFrame(out).T

    if by is None:
        return _stats(df)
    return (df.groupby(by, observed=True)
              .apply(lambda g: _stats(g)["hit_rate"], include_groups=False))


def reason_hits(df, patterns=FAILURE_PATTERNS, cols=None, id_col="id"):
    """Long table of the individual rows whose reason prose matches a failure
    phrase — (id, variant, column, snippet) — for eyeballing what broke.

    The manual-review sampler behind ``deepeval_reason_consistency``: that one
    automates 'does the prose state the same number as the field', this one hands
    you the prose itself, since whether a reason is a *good* justification is not
    automatable.
    """
    rx = re.compile("|".join(patterns), re.IGNORECASE)
    cols = cols or reason_cols(df)
    keep = [c for c in (id_col, "variant", "source_dataset") if c in df]
    records = []
    for c in cols:
        text = df[c].astype("string").fillna("")
        hit = text.str.strip().ne("") & text.str.contains(rx)
        for _, row in df.loc[hit].iterrows():
            rec = {k: row[k] for k in keep}
            rec["column"] = c
            rec["snippet"] = str(row[c])[:160]
            records.append(rec)
    return pd.DataFrame(records)


# --- (2d) How the variants compare -------------------------------------------
# The paired test itself is ``analysis.analysis.compare_variants`` (shared with
# plots.variant_effect_forest); what follows are the cuts that only belong to
# the report.

def abstention_adjusted(df, metrics=None, by=None):
    """Every metric reported twice: over all rows vs over answered rows only.

    Splits a low headline mean into its two causes — genuinely poor answers vs a
    high abstention rate. ``mean_all`` counts abstentions (via the shared
    ``rag_analysis._abstained`` flag) as scored; ``mean_answered`` excludes them.
    A large positive ``delta`` (answered >> all) means the metric is mostly
    measuring how often the system abstained, not answer quality. With ``by`` set,
    returns the answered-only means per group.

    This is the metrics-side abstention question. How OFTEN the pipeline abstained
    is ``rag_analysis.abstention_summary``.
    """
    metrics = metrics or score_cols(df)
    answered = ~ra._abstained(df)

    if by is not None:
        sub = df[answered].copy()
        sub[metrics] = sub[metrics].apply(pd.to_numeric, errors="coerce")
        return sub.groupby(by, observed=True)[metrics].mean()

    rows = {}
    for m in metrics:
        s_all = pd.to_numeric(df[m], errors="coerce")
        s_ans = pd.to_numeric(df.loc[answered, m], errors="coerce")
        rows[m] = {
            "n_all": int(s_all.notna().sum()),
            "mean_all": s_all.mean(),
            "n_answered": int(s_ans.notna().sum()),
            "mean_answered": s_ans.mean(),
            "delta": s_ans.mean() - s_all.mean(),
        }
    return pd.DataFrame(rows).T


def decile_breakdown(df, metric, n_bins=10, id_col="id"):
    """Split queries into ``n_bins`` equal-size bins by ``metric`` and list the
    IDs in each — decile 1 is the worst-scoring tenth (systematic weak points).

    Bins are cut on the *rank* (not the raw value) so metrics that pile up at
    0.0 / 1.0 still divide into equal-count groups instead of collapsing on
    duplicate quantile edges. Rows where the metric is NaN are dropped, so this
    reflects only scored queries.

    The ``ids`` cell tags each id with its ``variant`` when that column exists
    (e.g. ``mmlu_72[rag_sc]``). This matters because the pool is *unpaired*: a
    question scored under all three variants contributes up to three rows, so the
    same id can appear several times in one decile — the tag says which variant
    each is (``mmlu_72[no_rag]`` being in the worst tenth is expected; ``[rag_sc]``
    there is the interesting failure). For the "does rag_sc lift the *same*
    question?" story use a *paired* view instead (``compare_variants`` /
    ``plots.slopegraph``); pass a single-variant slice here
    (``df[df.variant=="rag_sc"]``) to read one variant's distribution cleanly, or
    the full frame to find questions that score badly across *every* variant
    (intrinsically hard items / bad golds).
    """
    s = pd.to_numeric(df[metric], errors="coerce")
    keep = [id_col] + (["variant"] if "variant" in df and id_col != "variant" else [])
    sub = df.loc[s.notna(), keep].copy()
    sub["value"] = s[s.notna()].to_numpy()
    if sub.empty:
        return pd.DataFrame(columns=["n", "mean", "min", "max", "ids"])
    if "variant" in sub:
        sub["_label"] = sub[id_col].astype(str) + "[" + sub["variant"].astype(str) + "]"
    else:
        sub["_label"] = sub[id_col].astype(str)
    ranks = sub["value"].rank(method="first")
    sub["decile"] = pd.qcut(ranks, min(n_bins, len(sub)), labels=False) + 1
    return sub.groupby("decile").agg(
        n=("value", "size"),
        mean=("value", "mean"),
        min=("value", "min"),
        max=("value", "max"),
        ids=("_label", lambda x: list(x)),
    )


# --- (3) Worst / best queries on the rag+eval join ---------------------------

def extremes(linked, metric, frac=0.01, signals=None):
    """The bottom and top ``frac`` of scored rows by ``metric``, with the pipeline
    signals attached. Returns ``(worst, best, k, n_scored)``; ``k`` is the count in
    each tail. Rows where the metric is NaN (not applicable / errored) are dropped
    first, so ``n_scored`` is the honest denominator.
    """
    signals = DISPLAY_SIGNALS if signals is None else signals
    scored = linked.dropna(subset=[metric])
    k = max(1, math.ceil(frac * len(scored)))
    cols = [c for c in ["id", "variant", "source_dataset", metric] + signals
            if c in scored.columns]
    worst = scored.nsmallest(k, metric)[cols]
    best = scored.nlargest(k, metric)[cols]
    return worst, best, k, len(scored)


def extremes_profile(linked, metric, frac=0.10, signals=None):
    """Signal means for the worst / best / middle tail on ``metric`` — the compact
    'is there a pattern?' table. Rows = worst/rest/best; cols = ``n``, the metric
    mean, each numeric signal mean, and the abstention rate. A worst tail with a
    markedly lower ``retrieval_best`` or higher abstention rate than the best tail
    is a systematic-difficulty signal.
    """
    signals = NUM_SIGNALS if signals is None else signals
    scored = linked.dropna(subset=[metric]).copy()
    if scored.empty:
        return pd.DataFrame()
    k = max(1, math.ceil(frac * len(scored)))
    rank = scored[metric].rank(method="first")
    scored["_grp"] = np.where(rank <= k, "worst",
                              np.where(rank > len(scored) - k, "best", "rest"))
    scored["_abstain"] = ra._abstained(scored).astype(float)
    agg = {"n": (metric, "size"), "metric_mean": (metric, "mean"),
           "abstain_rate": ("_abstain", "mean")}
    for s in signals:
        if s in scored:
            agg[s] = (s, "mean")
    return scored.groupby("_grp").agg(**agg).reindex(["worst", "rest", "best"])


# --- (4) Signal -> metric ----------------------------------------------------

def logprob_correlation(df, metric="ragas_scores.ragas_answer_correctness",
                        logprob_col="gen_logprob_stats.mean", by="variant"):
    """Correlate generation confidence (mean token logprob) with a quality
    metric, overall and per group.

    A healthy pipeline shows a *positive* correlation: more-confident generations
    score better. A flat or negative correlation means the logprob signal isn't
    tracking quality — the self-correction trigger keyed on it is mis-calibrated.
    Reports both Pearson (linear) and Spearman (monotone) on the rows where both
    values exist. ``plots.logprob_scatter`` draws it.

    This reads a metric, which is why it lives here rather than in
    ``rag_analysis`` with the rest of the confidence tables.
    """
    def _corr_one(sub):
        x = pd.to_numeric(sub[logprob_col], errors="coerce")
        y = pd.to_numeric(sub[metric], errors="coerce")
        m = x.notna() & y.notna()
        n = int(m.sum())
        return pd.Series({
            "n": n,
            "pearson": x[m].corr(y[m]) if n > 2 else _NAN,
            "spearman": x[m].corr(y[m], method="spearman") if n > 2 else _NAN,
        })

    rows = {"overall": _corr_one(df)}
    if by in df:
        for g, sub in df.groupby(by, observed=True):
            rows[str(g)] = _corr_one(sub)
    return pd.DataFrame(rows).T


def paired_delta(df, x_col, y_col, a, b):
    """Per-id paired change in ``x_col`` and ``y_col`` between variants ``a`` and
    ``b``, and how they relate. Pivots on id, forms Δx = a−b and Δy = a−b over ids
    scored in both, and returns ``n``, mean Δx / Δy, pearson / spearman of the two
    deltas, ``frac_conf_up`` (share with Δx>0) and ``frac_both_up``.

    With ``x_col = gen_logprob_stats.mean`` this is the 'is the confidence gain from
    b to a reflected in the metric?' test: a positive correlation says the queries
    that got more confident also scored better.
    """
    tmp = pd.DataFrame({
        "id": df["id"], "variant": df["variant"].astype(str),
        "x": pd.to_numeric(df[x_col], errors="coerce"),
        "y": pd.to_numeric(df[y_col], errors="coerce"),
    })
    wx = tmp.pivot_table(index="id", columns="variant", values="x", observed=True)
    wy = tmp.pivot_table(index="id", columns="variant", values="y", observed=True)
    if not ({a, b} <= set(wx.columns)) or not ({a, b} <= set(wy.columns)):
        return pd.Series({"n": 0})
    dx, dy = wx[a] - wx[b], wy[a] - wy[b]
    m = dx.notna() & dy.notna()
    n = int(m.sum())
    return pd.Series({
        "n": n,
        "mean_dx": dx[m].mean() if n else _NAN,
        "mean_dy": dy[m].mean() if n else _NAN,
        "pearson": dx[m].corr(dy[m]) if n > 2 else _NAN,
        "spearman": dx[m].corr(dy[m], method="spearman") if n > 2 else _NAN,
        "frac_conf_up": float((dx[m] > 0).mean()) if n else _NAN,
        "frac_both_up": float(((dx[m] > 0) & (dy[m] > 0)).mean()) if n else _NAN,
    })


def signal_vs_metric(df, signal, metrics=RETRIEVAL_METRICS, by=None, rows_mask=None):
    """Correlation of a pipeline ``signal`` (e.g. ``retrieval_best`` or
    ``reretrieval_gain``) with each of ``metrics``, overall or per ``by`` group.

    ``rows_mask`` restricts the rows first (e.g. only rag/rag_sc rows that HAVE a
    retrieval score, or only re-retried rag_sc rows). Returns one row per metric
    (x group) with ``n`` / ``pearson`` / ``spearman`` — a positive correlation
    means the metric reflects the retrieval signal.
    """
    sub = df if rows_mask is None else df[rows_mask]

    def one(s, group):
        out = []
        for m in metrics:
            if m not in s:
                continue
            n, pear, spear = _corr(s[signal], s[m])
            out.append({"metric": m.split(".")[-1], "group": group,
                        "n": n, "pearson": pear, "spearman": spear})
        return pd.DataFrame(out)

    if by is None:
        return one(sub, "overall")
    return pd.concat([one(g, str(k)) for k, g in sub.groupby(by, observed=True)],
                     ignore_index=True)


# --- helpers -----------------------------------------------------------------

def _indent(text, n=4):
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


def _rag_for_eval(eval_path):
    """The RAG file matching an eval file: from the eval name's ``_from_<stamp>``,
    else the newest RAG file (empty string if none)."""
    stem = Path(eval_path).stem
    if "_from_" in stem:
        stamp = stem.split("_from_", 1)[1]
        cands = sorted(Path("results").glob(f"{RAG_PREFIX}_{stamp}.json"))
        if cands:
            return str(cands[-1])
    try:
        return latest_results(RAG_PREFIX)
    except FileNotFoundError:
        return ""


def _split_args(argv):
    """Sort the positional paths into (eval_path, rag_path) by filename prefix, so
    the two can be given in either order. Unrecognised names fall back to
    positional (eval first, the documented order)."""
    eval_p = rag_p = None
    for a in argv:
        base = Path(a).name
        if base.startswith(EVAL_PREFIX):
            eval_p = a
        elif base.startswith(RAG_PREFIX):
            rag_p = a
        elif eval_p is None:
            eval_p = a
        elif rag_p is None:
            rag_p = a
    return eval_p, rag_p


if __name__ == "__main__":
    import sys
    import warnings

    # A correlation over a constant group (e.g. every faithfulness score = 1.0 in
    # one cell) is legitimately undefined and shows up as NaN below; mute the
    # numpy/scipy noise so the report stays readable.
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    try:
        from scipy.stats import ConstantInputWarning
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
    except ImportError:
        pass

    # Both files may be given in either order; they are told apart by name prefix.
    eval_path, rag_path = _split_args(sys.argv[1:])
    if eval_path is None:
        eval_path = latest_results(EVAL_PREFIX)
    if rag_path is None:
        rag_path = _rag_for_eval(eval_path)

    d = ev.load(eval_path)
    if not ev.metric_cols(d):
        sys.exit(f"{eval_path} has no ragas_scores.*/deepeval_scores.* columns — that "
                 f"looks like a raw rag file. Pass the evaluated_results_*.json "
                 f"(order does not matter).")
    # Everything printed below is collected and saved to this run's
    # reports/ folder, then echoed to the console.
    with paths.capture(eval_path, "eval_analysis_report"):
        print(f"{eval_path}: {len(d)} rows, {d['id'].nunique()} ids, "
              f"variants={list(d['variant'].cat.categories)}")
        print(f"writing every artifact to {paths.rel(paths.run_dir(eval_path))}")
        print(ra._abstained(d).groupby(d["variant"], observed=True)
              .mean().round(3).to_string())

        # (0) cohort, diagnosis, exclusion ------------------------------------------
        # Order matters: everything that DIAGNOSES the file runs on the raw frame,
        # and only then is the frame narrowed. Run the other way round, the reports
        # would be describing data the exclusion had already removed.
        print("\n=== cohort before exclusion (source_dataset x variant) ===")
        print(ev.describe_cohort(d).to_string())

        print("\n=== broken-signal checks (raw frame) ===")
        health_report(d, source=eval_path)

        print("\n=== metric-computation sanity (raw frame) ===")
        cls = classify_metrics(d)
        metric_error_report(d, cls, source=eval_path)

        d, drop_report = drop_eval_errors(d)
        print(f"\n=== dropped {drop_report['n_dropped']} evaluator-failure rows "
              f"({drop_report['n_before']} -> {drop_report['n_after']}) ===")
        print("by failure type (rows can overlap):")
        print(drop_report["by_type"].to_string())
        if drop_report["n_dropped"]:
            print("by dataset:")
            print(drop_report["by_dataset"].to_string())
            print("by variant:")
            print(drop_report["by_variant"].to_string())
            print("\n=== cohort after exclusion (what is actually analysed) ===")
            print(ev.describe_cohort(d).to_string())

        # Defensive second pass. `drop_eval_errors` above already removes every
        # RAGAS-errored row outright, so this normally reports 0 — it is kept so a
        # future change to the exclusion policy (masking per evaluator instead of
        # dropping the row, as its docstring describes) cannot silently leave
        # RAGAS-errored cells in the aggregates.
        d, prep = prepare(d)
        if prep["n_ragas_error_rows"]:
            print(f"\nprepared for analysis: on the {prep['n_ragas_error_rows']} rows whose RAGAS "
                  f"scorer raised, every ragas_scores.* value is set to NaN, so those rows no "
                  f"longer contribute to any RAGAS mean/correlation below. The rows themselves "
                  f"are kept — their DeepEval scores are independent of the RAGAS failure and "
                  f"stay in the DeepEval numbers.")
        else:
            print("\nprepared for analysis: no RAGAS-errored rows survived the exclusion "
                  "above, so nothing to mask (individual missing metric cells are still "
                  "excluded per metric).")
        if prep["n_faithfulness_cells_masked"]:
            print(f"prepared for analysis: {prep['n_faithfulness_cells_masked']} faithfulness "
                  f"cells on the {prep['n_abstained_rows']} abstained rows set to NaN "
                  f"({', '.join(c.split('.')[-1] for c in prep['faithfulness_cols_masked'])}), "
                  f"so from here on faithfulness is measured on answered rows only. The "
                  f"reason is in the raw-frame report above; the abstention rate itself is "
                  f"reported separately and is unaffected.")

        # (2a) metric distribution / discriminativeness -----------------------------
        print("\n=== per-metric distribution (is the metric discriminative?) ===")
        print("read before any mean below: median/IQR at a rail, or a tiny n_unique, "
              "means the metric separates nothing and its mean is not interpretable.")
        dist = metric_distribution(d)
        print(dist.round(3).to_string())
        dist.to_csv(paths.table(eval_path, "eval_metric_distribution"))

        print("\n--- per dataset x variant (a metric can discriminate on one and not "
              "another; low coverage here is often legitimate — see the report above) ---")
        dist_g = metric_distribution(d, by=["source_dataset", "variant"])
        print(dist_g.round(3).to_string())
        dist_g.to_csv(paths.table(
            eval_path, "eval_metric_distribution_by_dataset_variant"))

        # (2b) metric validation ----------------------------------------------------
        print("\n=== metric agreement (comparable metrics, overall) ===")
        ag = metric_agreement(d)
        print(ag.round(3).to_string(index=False))
        ag.to_csv(paths.table(eval_path, "eval_metric_agreement"), index=False)
        for by in ("source_dataset", "variant"):
            print(f"\n--- agreement by {by} ---")
            tab = metric_agreement(d, by=by)
            print(tab.round(3).to_string(index=False))
            tab.to_csv(paths.table(eval_path, f"eval_metric_agreement_by_{by}"),
                       index=False)

        print("\n=== DeepEval reason vs recorded score (internal consistency) ===")
        rc, rc_mism = deepeval_reason_consistency(d)
        print(rc.to_string(index=False))
        rc.to_csv(paths.table(eval_path, "eval_reason_consistency"), index=False)
        if len(rc_mism):
            print(f"\n{len(rc_mism)} mismatches (reason states a different number than the field):")
            print(rc_mism.head(20).to_string(index=False))
            rc_mism.to_csv(paths.table(eval_path, "eval_reason_mismatches"),
                           index=False)

        print("\n=== reason-string mining (failure-phrase hit rate per judge) ===")
        reasons = mine_reasons(d)
        print(reasons.to_string())
        reasons.to_csv(paths.table(eval_path, "reason_mining"))

        # (2d) how the variants compare ---------------------------------------------
        for by in ("source_dataset", "variant"):
            print(f"\n=== metric means by {by} ===")
            tbl = means_by(d, by=by)
            print(tbl.round(3).to_string())
            tbl.to_csv(paths.table(eval_path, f"means_by_{by}"))

        print("\n=== paired variant comparisons (Wilcoxon signed-rank) ===")
        print("the results-chapter spine; only meaningful for a metric that the "
              "distribution table above shows actually spreads.")
        for metric in ("ragas_scores.ragas_answer_correctness",
                       "deepeval_scores.deepeval_relevance"):
            for a, b in (("rag", "no_rag"), ("rag_sc", "rag")):
                cmp = ev.compare_variants(d, metric, a=a, b=b)
                print(f"\n{metric}: {a} vs {b}")
                print(cmp.round(3).to_string())
                cmp.to_csv(paths.table(
                    eval_path, f"compare_{a}_vs_{b}_{metric.split('.')[-1]}"))

        print("\n=== abstention-adjusted metrics (all rows vs answered only) ===")
        print("a large positive delta means the metric is mostly measuring how often "
              "the system abstained, not how good its answers are.")
        print("the faithfulness rows necessarily show delta 0 and n_all == n_answered: "
              "their abstained cells were excluded upstream, so both columns are already "
              "the answered-only mean.")
        adj = abstention_adjusted(d)
        print(adj.round(3).to_string())
        adj.to_csv(paths.table(eval_path, "abstention_adjusted"))

        print("\n=== decile drill-down (decile 1 = worst) ===")
        for metric in ("ragas_scores.ragas_answer_correctness",
                       "deepeval_scores.deepeval_relevance"):
            dec = decile_breakdown(d, metric)
            if dec.empty:
                print(f"{metric}: no scored rows")
                continue
            print(f"\n{metric}:")
            print(dec[["n", "mean", "min", "max"]].round(3).to_string())
            print(f"  worst-decile ids: {dec.iloc[0]['ids']}")
            dec.to_csv(paths.table(eval_path, f"deciles_{metric.split('.')[-1]}"))

        print("\n=== logprob vs answer correctness (want positive) ===")
        corr = logprob_correlation(d)
        print(corr.round(3).to_string())
        corr.to_csv(paths.table(eval_path, "logprob_correlation"))

        # Headline figures: each one is a table above that only becomes an argument
        # once you can see its shape.
        print("\n=== headline figures ===")
        plots.save_all({
            "fig_variant_effects": lambda: plots.variant_effect_forest(d),
            "fig_metric_rails": lambda: plots.metric_rail_plot(d),
            "fig_dataset_variant_correctness": lambda: plots.dataset_variant_heatmap(
                d, "ragas_scores.ragas_answer_correctness"),
            "fig_dataset_variant_relevance": lambda: plots.dataset_variant_heatmap(
                d, "deepeval_scores.deepeval_relevance"),
            "fig_metric_agreement": lambda: plots.metric_agreement_dots(ag),
        }, eval_path)

        # Figures that need only the evaluated file. The three eval_* ones used to
        # sit inside the cross-link block below and were silently skipped whenever
        # no rag file was found, although not one of them touches the joined frame.
        print("\n=== exploratory figures (evaluated results) ===")
        plots.save_all({
            "boxplot_faithfulness": lambda ax: plots.metric_boxplot(
                d, "ragas_scores.ragas_faithfulness", ax=ax),
            "coverage": lambda ax: plots.coverage_violin(d, ax=ax),
            "rejection": lambda ax: plots.rejection_bars(d, ax=ax),
            "slope_relevance": lambda ax: plots.slopegraph(
                d, "deepeval_scores.deepeval_relevance", ax=ax),
            "logprob_vs_correctness": lambda ax: plots.logprob_scatter(d, ax=ax),
            "eval_agreement_relevancy": lambda ax: plots.ragas_vs_deepeval(
                d, "ragas_scores.ragas_answer_relevancy",
                "deepeval_scores.deepeval_relevance", ax=ax),
            "eval_agreement_faithfulness": lambda ax: plots.ragas_vs_deepeval(
                d, "ragas_scores.ragas_faithfulness",
                "deepeval_scores.deepeval_faithfulness", ax=ax),
            "eval_confidence_delta_vs_correctness": lambda ax: plots.delta_scatter(
                d, "gen_logprob_stats.mean", "ragas_scores.ragas_answer_correctness",
                "rag_sc", "no_rag", ax=ax),
        }, eval_path)

        # (3) cross-link + worst/best ----------------------------------------------
        import os
        if rag_path and os.path.exists(rag_path):
            linked = ra.link_eval(ra.load(rag_path), d)
            print(f"\n=== linked rag+eval: {len(linked)} rows shared (over id x variant) ===")
            print(f"  rag file:  {rag_path}")
            print(f"  eval file: {eval_path}")
            if len(linked):
                # Everything below is derived from BOTH files, so both name it.
                both = [eval_path, rag_path]
                base = paths.linked(both, "linked")
                linked.to_parquet(base, index=False)
                print(f"  wrote {paths.rel(base)}  ({len(linked)} rows x "
                      f"{linked.shape[1]} cols) — base table")

                print("\n=== worst / best 10% per metric (with pipeline signals) ===")
                for metric in EXTREME_METRICS:
                    if metric not in linked:
                        continue
                    worst, best, k, n = extremes(linked, metric)
                    label = metric.split(".")[-1]
                    if not n:
                        print(f"\n{label}: no scored rows")
                        continue
                    print(f"\n--- {label}: worst {k} of {n} scored ---")
                    print(worst.round(3).to_string(index=False))
                    print(f"  worst by dataset: {worst['source_dataset'].value_counts().to_dict()}")
                    print(f"  worst by variant: {worst['variant'].value_counts().to_dict()}")
                    print(f"--- {label}: best {k} of {n} scored ---")
                    print(best.round(3).to_string(index=False))
                    print(f"  best by dataset: {best['source_dataset'].value_counts().to_dict()}")
                    print(f"  best by variant: {best['variant'].value_counts().to_dict()}")
                    prof = extremes_profile(linked, metric)
                    if len(prof):
                        print("  signal profile (worst / rest / best):")
                        print(_indent(prof.round(3).to_string()))

                # (4) signal -> metric -------------------------------------------------
                print("\n=== is the confidence gain reflected in the metrics? "
                      "(paired per-id Δ, want positive) ===")
                rows = {}
                for a, b in (("rag", "no_rag"), ("rag_sc", "no_rag"), ("rag_sc", "rag")):
                    for metric in ("ragas_scores.ragas_answer_correctness",
                                   "deepeval_scores.deepeval_relevance"):
                        key = f"Δconf vs Δ{metric.split('.')[-1]}: {a}-{b}"
                        rows[key] = paired_delta(d, "gen_logprob_stats.mean", metric, a, b)
                print("\npaired deltas (confidence change vs metric change):")
                print(pd.DataFrame(rows).T.round(3).to_string())

                print("\n=== do faithfulness/context metrics reflect retrieval scores? ===")
                has_retr = linked["variant"].astype(str).isin(["rag", "rag_sc"])
                rv = signal_vs_metric(linked, "retrieval_best", by="variant", rows_mask=has_retr)
                print("retrieval_best vs metric, by variant (want positive):")
                print(rv.round(3).to_string(index=False))
                rv.to_csv(paths.table(both, "eval_retrieval_vs_metric"), index=False)
                if "reretrieval_gain" in linked:
                    sc = (linked["variant"].astype(str).eq("rag_sc")
                          & linked["reretrieval_gain"].notna())
                    if sc.any():
                        print("\nHyDE re-retrieval gain vs metric (rag_sc rows, want positive):")
                        gv = signal_vs_metric(linked, "reretrieval_gain", rows_mask=sc)
                        print(gv.round(3).to_string(index=False))
                        gv.to_csv(paths.table(both, "eval_reretrieval_gain_vs_metric"),
                                  index=False)

                # The only figure that actually reads the joined frame, so the only
                # one named for both files.
                print("\n=== figures (linked rag+eval) ===")
                plots.save_all({
                    "eval_retrieval_vs_faithfulness": lambda ax: plots.retrieval_metric_scatter(
                        linked, "deepeval_scores.deepeval_faithfulness", ax=ax),
                }, both)
        else:
            print(f"\n(no RAG file at {rag_path!r}; cross-link, worst/best and signal->metric skipped)")
