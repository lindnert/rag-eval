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
     context metric; an abstention is not scored for answer relevancy, for
     faithfulness, or — off ``ABSTENTION_SCORED_DATASETS`` — for the two
     reference metrics; the ID-context metrics need reference contexts, i.e.
     the synthetic set only), or
     an actual ERROR (a RAGAS scorer raised, a DeepEval judge crashed). Reported
     per dataset x variant so the not-applicable cells are never miscounted as
     failures. ``prepare`` then nulls the RAGAS cells on RAGAS-errored rows so
     those rows drop out of RAGAS aggregates while their independent DeepEval
     scores survive.

  1b. Dataset-specific audits, each asking whether some property of the QUESTIONS
     rather than of the pipeline is driving the numbers above: the NGQA conflict
     structure, the LLMDRS answer leakage, and the cross-lingual cell.

     The cross-lingual block is the one language analysis in the run.
     ``crosslingual_frame`` narrows to the synthetic reference contexts — the only
     cell generated in both German and English — and from there
     ``retrieval_language_routing`` asks whether the retriever answers a German
     question from German documents, ``gold_recall_by_language`` what that routing
     costs the ID-based retrieval metrics, ``answer_language_audit`` whether the
     answer came back in the language it was asked in, ``language_contrast`` how
     the two arms score (unpaired, stratified by context), ``language_variant_effects``
     whether retrieval buys the same thing in both, and ``scorer_direction_conflict``
     whether the scorers of one construct even agree on which language scored
     higher. Read that last one first: where they do not, no language claim in
     that family can be made without naming its scorer.

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

import csv
import json
import math
import re
from typing import NamedTuple
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
# empty case rather than a measurement — see ``metrics_on_abstentions``.
# ``deepeval_contextual_relevance`` is deliberately not here: it scores the
# retrieved context against the QUESTION, which an abstention does not touch.
FAITHFULNESS_METRICS = {"ragas_scores.ragas_faithfulness",
                        "ragas_scores.ragas_faithfulness_with_hhem",
                        "deepeval_scores.deepeval_faithfulness"}
# The two metrics that grade the answer against a REFERENCE answer. They are
# dropped on abstentions too, but — unlike relevancy and faithfulness — not
# everywhere: see ABSTENTION_SCORED_DATASETS.
#
# On an abstention the comparison has no answer on its side, so the scorer grades
# a refusal against a gold answer and necessarily returns ~0. Left in, that 0 is
# indistinguishable from a wrong answer, and the metric's mean silently becomes a
# mixture of "how good are the answers" and "how often did it decline to give
# one" — the second question already has its own number (the abstention rate),
# and mixing them is what makes a cautious system look incompetent.
REFERENCE_METRICS = {"ragas_scores.ragas_answer_accuracy",
                     "ragas_scores.ragas_answer_correctness"}
# The exception: datasets that exist to test whether the system declines. MedQA is
# clinical, i.e. outside the nutrition scope the system is built for, so abstaining
# there is the behaviour under test rather than a missing answer — the score on
# those rows is a measurement we want, and dropping them would delete the evidence.
# Everywhere else abstaining is a non-answer to a question that was in scope.
ABSTENTION_SCORED_DATASETS = {"medqa"}
# The variants whose abstention rate is a measurement at all. ``no_rag`` gets a
# system prompt with no abstention clause and is short-circuited before the
# rejection check in ``rag.utils._finalize_answer`` ("no_rag has no context to be
# 'insufficient', so it never abstains here"), so its 0% abstention rate on the
# probe is a property of the prompt, not a finding about the model.
ABSTAINING_VARIANTS = ("rag", "rag_sc")

# --- multiple-choice flattening ---------------------------------------------
# MMLU and MEDQA ship as multiple-choice items; both loaders keep only the stem
# and the correct option's TEXT, dropping the distractors, so the pipeline is
# asked an open-ended question. A stem that pointed at its option list ("which of
# the following …") still points at it afterwards, at nothing.
MC_DERIVED_DATASETS = ("mmlu", "medqa")
# The pointer phrases actually present in the run. `which of these`,
# `all of the above` and lettered `A) B)` blocks are matched too but occur zero
# times — kept so the flag stays valid if the sample is redrawn.
MC_POINTER_RE = re.compile(
    r"of the following"
    r"|which (one )?of these"
    r"|which statements? (about|is|are)"
    r"|the (factors|treatments|options|statements|choices) below"
    r"|statements? (is|are) (true|correct|false)"
    r"|(both|all|none) of (the )?(above|options|statements)"
    r"|^\s*[a-e][).:]\s", re.IGNORECASE | re.MULTILINE)
# The pointer alone is a poor predictor: 43 of 50 MEDQA stems carry one and none
# of them break, because a clinical vignette (median ~740 chars: age, vitals,
# labs, imaging) is still answerable once the options are gone. An MMLU stem
# (median ~85 chars) is often nothing BUT the pointer — "Which of the following
# statements about proteolysis is correct?" has no proposition left in it. So the
# flag that matters is pointer AND short stem, i.e. not self-contained. The
# threshold sits in the empty gap between the two datasets' length distributions;
# it separates them, it is not tuned against any score.
MC_SELF_CONTAINED_CHARS = 300
# A gold that is itself an index into the deleted option list ("All of the
# above"). Unscoreable by construction: no open-ended answer can match it.
MC_GOLD_POINTER_RE = re.compile(
    r"all of (the )?(above|options)|all options given"
    r"|both of the statements|none of the (above|options)"
    r"|options given (is|are) correct", re.IGNORECASE)
# The answer-side symptom, and the only part of this audit that is a direct
# count rather than an inference: the model replying "please provide the list of
# statements" instead of answering. Deliberately narrow — it must name the
# missing material, so ordinary clinical prose ("the first-line choice is …")
# does not match.
MC_OPTION_REQUEST_RE = re.compile(
    r"please (provide|share|list|specify) the (list|options|specific|multiple)"
    r"|you (have not|haven'?t|did not|didn'?t) (provide|provided|share|shared|list|listed)"
    r"|provide the (list|options|specific|multiple)"
    r"|(specific|the) (options|statements|choices) (you|to choose|are|were|is)"
    r"|without the (options|list|specific)"
    r"|cannot (select|identify) (the|which|it)"
    r"|list of (statements|options|factors|choices)"
    r"|options to choose"
    r"|question was cut off", re.IGNORECASE)
# Manual adjudication of the rows that asked for the options, still volunteered
# substantive content, and were graded 0.0 on answer_accuracy anyway. Read by
# hand on 2026-08-25 against the run below; keyed by id so a redrawn sample just
# prints fewer notes instead of printing stale ones. The point of keeping it is
# that the group is NOT uniform — most of the zeros are real misses caused by
# the missing options, but not all of them are, and the report should not claim
# otherwise in either direction.
MC_ADJUDICATED = {
    "mmlu_212": "0.00 correct. Gold is the FALSE statement (starch -> insulin -> "
                "motility); the answer lists true principles plus three guessed "
                "false ones, none of them the gold. Disjoint.",
    "mmlu_106": "0.00 correct on the string (gold 'Parental anxiety' never "
                "appears), but the GOLD is itself doubtful: parental anxiety is "
                "commonly reported as a risk factor for eating disorders, so the "
                "item was only answerable via the deleted distractor set.",
    "mmlu_197": "0.00 too harsh. Gold 'Have been unchanged'; the answer opens "
                "'rates ... have stabilized or declined'. Stabilised ~ unchanged, "
                "so the gold claim is present but hedged across two mutually "
                "exclusive outcomes, which is plausibly why the judge withheld "
                "credit. Judge severity on a degenerate input, not a model error.",
}

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
    # The one pair whose two members come from the same library. Both grade the
    # answer against the same reference answer and differ only in how: correctness
    # decomposes into claims and scores an F1 over them, accuracy asks a judge for
    # a single 0/2/4 verdict rescaled to [0,1]. So this pair is not "do two
    # libraries agree" but "does the decomposition buy anything the verdict does
    # not" — and both numbers are load-bearing in the results chapter, which is
    # reason enough to know whether they are one metric or two.
    #
    # Abstentions are NOT dropped here. They are already gone from every dataset
    # but ABSTENTION_SCORED_DATASETS (nulled upstream in ``prepare``), and on that
    # dataset the score on a refusal is the measurement under test, not noise.
    # Beware the rail effect that leaves: both scorers reward the same refusals,
    # so on medqa they can agree strongly while sharing no signal about answers.
    # Named without the family prefix the other pairs carry: those prefixes exist to
    # say WHICH construct two differently-named scorers are being compared on, and
    # here both members already name it. "Reference: RAGAS Correctness vs Accuracy"
    # also invites reading "Accuracy" as some other library's.
    ("ragas correctness vs ragas accuracy",
     "ragas_scores.ragas_answer_correctness",
     "ragas_scores.ragas_answer_accuracy", "reference"),
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
# Which metrics get a paired variant test, and against which variants.
#
# The pairs differ per metric and not by preference: faithfulness and context
# relevance are only defined where there are retrieved contexts, so pairing them
# against ``no_rag`` yields an empty intersection. Listing the pairs explicitly
# keeps that a documented property of the metric rather than a silent empty
# table. All three faithfulness scorers are here on purpose -- run side by side
# on the same pairs, they say whether a variant effect is a property of the
# system or of one judge.
#
# The list is every metric that CAN be paired, in METRIC_ORDER, rather than a
# selection: it drives both this table and plots.paired_comparison_plot, and a
# metric left out here is a comparison the results chapter never sees. That is not
# hypothetical -- answer_accuracy is the one reference metric on which rag_sc beats
# rag, and while it was missing the table said self-correction bought no answer
# quality at all.
#
# That is why the three ragas_id_context_* rows are here despite pairing 68
# questions against the 250-370 above them. They need a gold reference-context set,
# so they exist for the synthetic guideline questions only -- and they are the ONLY
# retrieval measure in the run with a ground truth. Dropping them for thin n left
# deepeval_contextual_relevance as the sole voice on whether self-correction
# retrieves better, and it happens to be the one that says yes. Their n is on the
# figure and in the table; a reader can discount them, but not if they are absent.
VARIANT_COMPARISONS = (
    ("ragas_scores.ragas_answer_correctness", (("rag", "no_rag"), ("rag_sc", "rag"))),
    ("ragas_scores.ragas_answer_accuracy", (("rag", "no_rag"), ("rag_sc", "rag"))),
    ("ragas_scores.ragas_answer_relevancy", (("rag", "no_rag"), ("rag_sc", "rag"))),
    ("deepeval_scores.deepeval_relevance", (("rag", "no_rag"), ("rag_sc", "rag"))),
    ("ragas_scores.ragas_faithfulness", (("rag_sc", "rag"),)),
    ("ragas_scores.ragas_faithfulness_with_hhem", (("rag_sc", "rag"),)),
    ("deepeval_scores.deepeval_faithfulness", (("rag_sc", "rag"),)),
    ("deepeval_scores.deepeval_contextual_relevance", (("rag_sc", "rag"),)),
    ("ragas_scores.ragas_id_context_precision", (("rag_sc", "rag"),)),
    ("ragas_scores.ragas_id_context_recall", (("rag_sc", "rag"),)),
    ("ragas_scores.ragas_id_context_ap", (("rag_sc", "rag"),)),
)

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


def _abstention_excluded(df):
    """Rows whose ``REFERENCE_METRICS`` cells do not count: an abstention on a
    dataset where abstaining is NOT the behaviour under test.

    The one place that rule is expressed. ``classify_metrics`` uses it to label the
    cells ``na_rejected`` and ``prepare`` to null them, so the report's census and
    the frame everything downstream reads can never disagree about which cells
    those are. Without ``source_dataset`` (a frame that never carried it) it falls
    back to every abstention, which is the conservative direction: it drops cells
    rather than scoring ones that may not belong.
    """
    rej = ra._abstained(df)
    if "source_dataset" not in df:
        return rej
    return rej & ~df["source_dataset"].astype(str).isin(ABSTENTION_SCORED_DATASETS)


def classify_metrics(df):
    """Long table (id, variant, source_dataset, metric, status) tagging every
    metric cell as one of: ``scored`` / ``na_no_context`` (context metric on
    no_rag) / ``na_no_reference`` (ID-context metric off the synthetic set) /
    ``na_rejected`` (an answer-grading metric on an abstention — relevancy and
    faithfulness everywhere, the two reference metrics everywhere except
    ``ABSTENTION_SCORED_DATASETS``) / ``error`` (applicable but missing — a scorer
    raised or the value never landed).

    The status is derived from variant + dataset + the ``rejected`` flag, NOT from
    the sentinel strings, so it is robust to wording changes. This is the single
    source of truth for the error report.
    """
    rej = ra._abstained(df)
    rej_ref = _abstention_excluded(df)
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
        if m in REFERENCE_METRICS:  # same status, dataset-dependent mask
            status = status.mask(rej_ref, "na_rejected")
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


def _metric_error_cells(df, metric):
    """Boolean mask of rows where the evaluator recorded an exception for one metric.

    Both evaluators write per-cell errors, in different shapes (see
    ``metric_error_reasons``), so the column is looked up rather than derived.
    """
    for col in (f"ragas_scores.ragas_metric_errors.{metric.split('.')[-1]}",
                f"{metric}_error"):
        if col in df:
            return _nonempty(df[col])
    return pd.Series(False, index=df.index)


def error_length_profile(df, metrics=None, by="source_dataset", text_col="answer",
                         min_errors=1):
    """Is a metric's failure rate a function of how long the answer was?

    The metric-status table says HOW MANY cells a scorer failed to produce and
    ``metric_error_reasons`` says which exception it raised; neither says whether the
    failures are a random subset of the rows or a systematically different one. That
    difference decides how the gap is reported: an exception that fires uniformly
    costs precision, while one that fires on the longest answers censors the cohort,
    and a mean computed over what is left is a mean over the short answers.

    ``LengthFinishReasonError`` is the case that motivated this — the judge's own
    completion hits its token cap, which is a property of how much text it was asked
    to decompose, so the failed cells should be the long ones. Reporting the two
    medians side by side is what turns "40 cells are missing" into a statement about
    which 40.

    One row per (metric, ``by``-group) with at least ``min_errors`` failures, so the
    table lists only the cells where there is something to explain. Columns:
    ``n_attempted`` (rows where the metric either landed a value or recorded an
    exception — the scorer's own denominator, which is NOT the applicable-cell count
    in the status table: a cell can be skipped as ``na_*`` without either); ``n_scored``
    / ``n_error`` / ``error_rate``; ``error_types`` (the exception classes, with counts
    where there is more than one); and the ``text_col`` length in characters —
    ``len_med_scored`` against ``len_med_error``, with the means and the errored p90
    alongside, since a median pair can hide a heavy tail.

    Run on the RAW frame: ``prepare`` nulls metric cells, which would move rows out of
    ``n_scored`` for reasons that have nothing to do with the scorer failing.
    """
    if text_col not in df:
        return pd.DataFrame()
    metrics = metrics or [c for c in ev.metric_cols(df)
                          if _metric_error_cells(df, c).any()]
    length = df[text_col].fillna("").astype(str).str.len()
    by = [by] if isinstance(by, str) else list(by or [])
    rows = []
    for metric in sorted(set(metrics)):
        err = _metric_error_cells(df, metric)
        if metric not in df or not err.any():
            continue
        val = df[metric]
        # "Attempted" is value-or-exception. A cell with neither was never tried (the
        # evaluators skip whole metric families on a rejected row), and counting those
        # into the denominator would report a skip as a success.
        scored = _nonempty(val) if val.dtype == object else val.notna()
        scored = scored & ~err
        attempted = scored | err
        groups = ([(name, g) for name, g in df.groupby(by, observed=True)]
                  if by else [((), df)])
        for name, g in groups:
            name = name if isinstance(name, tuple) else (name,)
            idx = g.index
            e, s = err.loc[idx], scored.loc[idx]
            n_err, n_ok = int(e.sum()), int(s.sum())
            if n_err < min_errors:
                continue
            n_att = int(attempted.loc[idx].sum())
            types = _error_types(df, metric, idx[e])
            le, ls = length.loc[idx[e]], length.loc[idx[s]]
            rows.append(dict(zip(by, name), **{
                "metric": metric.split(".")[-1],
                "n_attempted": n_att, "n_scored": n_ok, "n_error": n_err,
                "error_rate": n_err / n_att if n_att else _NAN,
                "error_types": types,
                "len_med_scored": float(ls.median()) if n_ok else _NAN,
                "len_med_error": float(le.median()) if n_err else _NAN,
                "len_mean_scored": float(ls.mean()) if n_ok else _NAN,
                "len_mean_error": float(le.mean()) if n_err else _NAN,
                "len_p90_error": float(le.quantile(0.9)) if n_err else _NAN,
            }))
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    return (out.set_index(["metric"] + by).sort_index() if by
            else out.set_index("metric").sort_index())


def _error_types(df, metric, idx):
    """The exception classes behind a set of failed cells, as one printable string —
    bare when there is only one, ``"Type xN"`` joined by ``+`` when there are several.
    """
    for col in (f"ragas_scores.ragas_metric_errors.{metric.split('.')[-1]}",
                f"{metric}_error"):
        if col in df:
            vc = (df.loc[idx, col].astype(str).str.split(":").str[0].str.strip()
                  .value_counts())
            if not len(vc):
                return ""
            return (vc.index[0] if len(vc) == 1
                    else " + ".join(f"{k} x{v}" for k, v in vc.items()))
    return ""


# The three metric families ``classify_metrics`` marks ``na_rejected``, in the order
# the report argues them. Grouped rather than merged because the three exclusions
# rest on DIFFERENT evidence, and one table showing all three side by side is what
# makes that visible: relevancy was never scored at all (the evaluators skip it on a
# rejected row, so the exclusion is bookkeeping for a decision already taken
# upstream); faithfulness WAS scored and the scorers disagree with each other about
# what a refusal means; the reference metrics were scored, agree, and are excluded
# anyway — because what they agree on is that a refusal is a wrong answer, which is
# true and is the abstention rate rather than an answer-quality measurement.
ABSTENTION_EXCLUDED_FAMILIES = (
    ("relevancy", RELEVANCY_METRICS),
    ("faithfulness", FAITHFULNESS_METRICS),
    ("reference", REFERENCE_METRICS),
)


def _verdict_stats(df, metric, mask):
    """``(n_idk, frac_idk)`` over ``mask`` from the DeepEval verdict counts a metric
    persists, or ``(NaN, NaN)`` for metrics that keep none.

    DeepEval records ``{n_verdicts, yes, no, idk}`` per cell (see the 2026-08-10
    change in ``evaluation/deepeval_eval.py``), and ``idk`` is the load-bearing one:
    its verdict prompt reserves "no" for a DIRECT contradiction, so an unsupported
    claim comes back "idk", and whether that counts as faithful is the whole
    difference between the score before and after ``penalize_ambiguous_claims``. On
    an abstention it is also the mechanism — a refusal yields one claim, which is
    unverifiable, so the metric's value there is really a count of shrugs.
    """
    n_col, idk_col = f"{metric}_verdicts.n_verdicts", f"{metric}_verdicts.idk"
    if n_col not in df or idk_col not in df:
        return _NAN, _NAN
    total = pd.to_numeric(df.loc[mask, n_col], errors="coerce").sum()
    idk = pd.to_numeric(df.loc[mask, idk_col], errors="coerce").sum()
    # No verdicts at all is "the metric never ran here", not "it ran and shrugged
    # zero times" — a metric skipped on abstentions would otherwise print n_idk = 0
    # and read as evidence of something.
    if not total:
        return _NAN, _NAN
    return float(idk), float(idk / total)


def metrics_on_abstentions(df, families=ABSTENTION_EXCLUDED_FAMILIES):
    """``(table, info)``: what each scorer put on the abstained rows, per family.

    The evidence behind every ``na_rejected`` cell the status table shows, so each
    exclusion is argued in the report rather than asserted. The three families are
    excluded for three different reasons and the table is meant to be read family by
    family — see ``ABSTENTION_EXCLUDED_FAMILIES``.

    Faithfulness is the one where the exclusion is forced. It asks whether the
    ANSWER's claims are entailed by the context; an abstention makes no claims, so
    "all zero claims are entailed" is vacuously true and each scorer invents its own
    convention for the empty case. They do, and they disagree almost maximally: some
    score a refusal 0.0 (counting it as a hallucination, wrong even on their own
    definition) and some score it 1.0. Since the refusal text is identical across
    those rows, none of that spread can be a property of the run.

    Which scorer takes which side is NOT stable, so read it off the table rather than
    remembering it. On the 2026-07-29 run RAGAS scored 98.5% of abstentions 0.0 while
    HHEM and DeepEval scored 83-95% of the same rows 1.0; on 2026-08-12, with
    ``penalize_ambiguous_claims`` enabled (2026-08-10), DeepEval extracts a single
    claim from a refusal, returns "idk" on it 89% of the time and now scores 95% of
    those rows 0.0 — the RAGAS side. Only HHEM is still on 1.0. The exclusion this
    table justifies is unaffected either way: that is the point of excluding rather
    than picking a convention.

    Columns, per metric: ``n_abstained`` (abstained cells that carried a value — 0
    means the evaluator never scored them, which is itself the justification);
    ``n_excluded`` (cells the rule actually nulls, smaller than ``n_abstained``
    exactly where ``ABSTENTION_SCORED_DATASETS`` applies); ``mean_excluded`` and the
    two rail shares, over those nulled cells and not over every abstained one — the
    difference matters only for the reference family, where the two cohorts sit at
    opposite rails and their pooled mean describes neither; ``mean_kept``, the same
    metric on the abstained cells that survive (NaN wherever the exclusion is
    run-wide); ``n_idk`` / ``frac_idk`` from the DeepEval verdict counts, NaN for
    metrics that keep none; and the answered-row mean for contrast.

    ``info["n_texts"]`` counts the distinct abstention answer strings — a handful
    means the rows really are canonical refusals; many would mean ``_abstained`` is
    catching partial answers, which DO carry claims and must not be discarded
    wholesale, and the exclusion would have to be reconsidered. ``info["spread"]``,
    ``reads_zero`` and ``reads_one`` describe the faithfulness family only, since
    they are about the disagreement that family alone exhibits.

    Run this on the RAW frame: ``prepare`` nulls exactly these cells, so afterwards
    there is nothing left to describe.
    """
    rej = ra._abstained(df)
    rej_ref = _abstention_excluded(df)
    rows = {}
    for family, metrics in families:
        # The reference family is the one whose exclusion is dataset-dependent, so
        # it is the one whose n_excluded differs from n_abstained.
        excl = rej_ref if family == "reference" else rej
        for m in sorted(c for c in metrics if c in df):
            v = pd.to_numeric(df[m], errors="coerce")
            # The mean and the rails describe the cells the rule NULLS, not every
            # abstained cell. They are the same set for two of the three families;
            # for ``reference`` they are not, and pooling the kept ones in produced a
            # mean of 0.489 on a metric whose exclusion rests on it being ~0 — an
            # average over a cohort that is ~0 on four datasets and ~1 on the fifth,
            # describing neither.
            a = v[excl].dropna()
            kept = v[rej & ~excl].dropna()
            b = v[~rej].dropna()
            n_idk, frac_idk = _verdict_stats(df, m, rej)
            rows[m.split(".")[-1]] = {
                "family": family,
                "n_abstained": int((rej & v.notna()).sum()),
                "n_excluded": len(a),
                "mean_excluded": a.mean() if len(a) else _NAN,
                "frac_0": float((a == 0).mean()) if len(a) else _NAN,
                "frac_1": float((a == 1).mean()) if len(a) else _NAN,
                "mean_kept": kept.mean() if len(kept) else _NAN,
                "n_idk": n_idk,
                "frac_idk": frac_idk,
                "n_answered": len(b),
                "mean_answered": b.mean() if len(b) else _NAN,
            }
    tab = pd.DataFrame(rows).T
    if len(tab):
        for c in ("n_abstained", "n_excluded", "n_answered"):
            tab[c] = tab[c].astype(int)
        tab.index.name = "metric"

    texts = None
    if "answer" in df:
        texts = df.loc[rej, "answer"].fillna("").str.strip()
    # WHICH scorer takes which convention is read off the table, never assumed. It is
    # not a stable property of the libraries: DeepEval scored these rows 1.0 until
    # ``penalize_ambiguous_claims`` was turned on (2026-08-10), after which its lone
    # extracted claim comes back "idk" on a refusal and it scores 0.0 — the same side
    # as RAGAS. A sentence naming the groups in prose was correct when written and
    # wrong two weeks later, which is why this is computed.
    # Faithfulness only: the disagreement these three keys describe is that family's,
    # and pooling the other two in would compare scorers that were never asked the
    # same question about the row.
    faith = (tab[tab["family"] == "faithfulness"] if len(tab)
             else pd.DataFrame(columns=["mean_excluded"]))
    mid = faith["mean_excluded"] if len(faith) else pd.Series(dtype=float)
    info = {
        "n_abstained_rows": int(rej.sum()),
        "n_texts": int(texts.nunique()) if texts is not None else None,
        # The gap between the most and least generous convention, on identical text.
        "spread": (float(mid.max() - mid.min()) if len(faith) and mid.notna().any()
                   else _NAN),
        "reads_zero": sorted(mid.index[mid < 0.5]),
        "reads_one": sorted(mid.index[mid > 0.5]),
    }
    return tab, info


def probe_non_abstentions(df, datasets=ABSTENTION_SCORED_DATASETS,
                          variants=ABSTAINING_VARIANTS, max_chars=400):
    """The out-of-scope probe rows the system ANSWERED instead of declining — one
    row each, with the query, the answer and the signals, for reading by hand.

    MEDQA is loaded as a rejection probe with the refusal string as its gold
    (``dataset/MEDQA/loader.py``), so every one of these rows scores ~0 on both
    reference metrics whatever it says, and no aggregate can tell an out-of-scope
    hallucination from a question the corpus turned out to cover. Only the rows
    themselves can, and there are few enough of them to look at.

    ``no_rag`` is excluded (``ABSTAINING_VARIANTS``): it is never offered the
    abstention instruction, so its rows are answered by construction and would
    swamp the handful that are a decision. ``retrieval_best`` and
    ``ctx_relevance`` (the DeepEval context judge, which scores retrieval against
    the QUESTION and is therefore defined on abstentions too) are the two columns
    to read first: high on both means the nutrition corpus really does cover the
    question, which makes the refusal gold wrong for that row rather than the
    answer.

    Run on the RAW frame — ``prepare`` nulls the faithfulness cells of abstentions.
    """
    if "source_dataset" not in df or "variant" not in df:
        return pd.DataFrame()
    probe = df[df["source_dataset"].astype(str).isin(datasets)]
    answered = probe[probe["variant"].astype(str).isin(variants)
                     & ~ra._abstained(probe)]
    if not len(answered):
        return pd.DataFrame()

    cols = [c for c in ("id", "variant", "lang", "retrieval_best", "retrieval_average",
                        "deepeval_scores.deepeval_contextual_relevance",
                        "ragas_scores.ragas_faithfulness",
                        "deepeval_scores.deepeval_faithfulness",
                        "ragas_scores.ragas_answer_correctness",
                        "gen_logprob_stats.mean",
                        "dataset_metadata.original_medqa_answer",
                        "query", "answer") if c in answered]
    # Short names, except the logprob mean: stripped to its last component it
    # becomes a column called "mean" sitting among six other numeric columns.
    out = answered[cols].rename(columns=lambda c: (
        "logprob_mean" if c == "gen_logprob_stats.mean" else c.split(".")[-1]))
    for c in ("query", "answer"):
        if c in out:
            out[c] = out[c].map(
                lambda t: re.sub(r"\s+", " ", str(t)).strip()[:max_chars])
    return out.sort_values(["id", "variant"])


def mc_flatten_audit(df, datasets=MC_DERIVED_DATASETS,
                     self_contained_chars=MC_SELF_CONTAINED_CHARS):
    """One row per QUESTION of the multiple-choice-derived datasets, flagged for
    the scars left by dropping the option list.

    Question level, not row level: "the stem points at an option list" is a
    property of the query, and the three variant rows that share a query are one
    observation measured three times, not three observations.

    ``pointer``          the stem still refers to the deleted options.
    ``short_stem``       nothing much left besides the pointer.
    ``dangling``         both — the query is not answerable as written. This is
                         the flag to use; ``pointer`` alone over-flags MEDQA,
                         whose vignettes survive option removal intact.
    ``visual_ref``       refers to an image the pipeline was never given.
    ``gold_pointer``     the GOLD is an option index ("All of the above"), so the
                         item is unscoreable however the system answers.
    """
    if not {"source_dataset", "query", "id"} <= set(df):
        return pd.DataFrame()
    q = df[df["source_dataset"].astype(str).isin(datasets)]
    q = q.drop_duplicates(["source_dataset", "id"])
    if not len(q):
        return pd.DataFrame()

    query = q["query"].astype(str)
    gold = q["reference_answer"].astype(str) if "reference_answer" in q else pd.Series("", index=q.index)
    out = pd.DataFrame({
        "dataset": q["source_dataset"].astype(str),
        "id": q["id"].astype(str),
        "chars": query.str.len(),
        "pointer": query.str.contains(MC_POINTER_RE),
        "short_stem": query.str.len() < self_contained_chars,
        # Exam items also carry figures, dropped by the same loaders. Bare
        # ``figure``/``image`` is safe here because the flag is computed on the
        # MC datasets only, where those words never occur in prose.
        "visual_ref": query.str.contains(
            r"\b(?:is shown|as shown|arrows?|photograph|photo|figure|image"
            r"|micrograph|radiograph|exhibit)\b", case=False, regex=True),
        "gold_pointer": gold.str.contains(MC_GOLD_POINTER_RE),
    })
    out["dangling"] = out["pointer"] & out["short_stem"]
    return out.sort_values(["dataset", "id"]).reset_index(drop=True)


def option_request_rows(df, max_chars=220):
    """The rows where the model answered "please provide the list of options"
    (or "you did not provide the image") instead of answering — the measurable
    consequence of the flattening, and the only part of this audit that is a
    direct count rather than an inference.

    These are NOT abstentions: they carry none of the rejection strings, so the
    abstention detector counts them as answers and every reference metric grades
    a request for clarification against a nutrition fact.

    The group is bimodal, so ``noncommittal`` splits it: ``ragas_answer_relevancy``
    = 0.0 EXACTLY is RAGAS's own noncommittal detector firing, i.e. the row is a
    bare request with nothing to grade. The rest asked for the options and then
    volunteered an answer anyway, and land in the ordinary 0.5-0.8 band. Averaging
    the two together mixes "the metric returned a structural zero" with "the model
    answered and was graded", which are different events; report the split.

    Run on the RAW frame: ``prepare`` nulls cells these rows still occupy.
    """
    if "answer" not in df:
        return pd.DataFrame()
    hit = df[df["answer"].astype(str).str.contains(MC_OPTION_REQUEST_RE)]
    if not len(hit):
        return pd.DataFrame()
    cols = [c for c in ("id", "source_dataset", "variant", "lang",
                        "ragas_scores.ragas_answer_relevancy",
                        "ragas_scores.ragas_answer_correctness",
                        "ragas_scores.ragas_answer_accuracy",
                        "deepeval_scores.deepeval_relevance",
                        "query", "answer") if c in hit]
    out = hit[cols].rename(columns=lambda c: c.split(".")[-1])
    rel = "ragas_answer_relevancy"
    out["noncommittal"] = (pd.to_numeric(out[rel], errors="coerce").eq(0.0)
                           if rel in out else False)
    for c in ("query", "answer"):
        if c in out:
            out[c] = out[c].map(
                lambda t: re.sub(r"\s+", " ", str(t)).strip()[:max_chars])
    tail = [c for c in ("query", "answer") if c in out]
    out = out[[c for c in out.columns if c not in tail] + tail]
    return out.sort_values(["source_dataset", "variant", "id"])


def option_request_contrast(df, degenerate, metrics=sorted(ANSWER_METRICS)):
    """Within each affected (dataset, variant) cell: the option-request rows
    against the OTHER rows of the same cell.

    Deliberately a within-cell contrast, not a cross-variant one. Comparing these
    rows to ``rag``/``rag_sc`` answers the wrong question, because the variants
    also differ in whether they may abstain at all; holding the variant fixed
    isolates what the missing option list did.

    Reports mean, median, the count of exact 0.0 scores on each side, and a
    Mann-Whitney U with rank-biserial. Rank-based because these distributions are
    not shifted normals but two clusters (see ``option_request_rows``). One test
    per metric within a single cell, so no multiplicity correction is applied
    here — but n on the flagged side is small and the group is defined by a regex
    over the answer text, so this is a descriptive contrast and not a randomised
    one. The counts and the zero-tallies are the evidence; the p-value only says
    the two clusters are not a coincidence.
    """
    from scipy.stats import mannwhitneyu

    if not len(degenerate) or "variant" not in df:
        return pd.DataFrame()
    bad = set(zip(degenerate["source_dataset"], degenerate["variant"], degenerate["id"]))
    rows = []
    for ds, var in sorted(set(zip(degenerate["source_dataset"], degenerate["variant"]))):
        cell = df[(df["source_dataset"].astype(str) == ds)
                  & (df["variant"].astype(str) == var)]
        is_bad = pd.Series([k in bad for k in zip(cell["source_dataset"],
                                                  cell["variant"], cell["id"])],
                           index=cell.index)
        for m in (m for m in metrics if m in df):
            a = pd.to_numeric(cell.loc[is_bad, m], errors="coerce").dropna()
            b = pd.to_numeric(cell.loc[~is_bad, m], errors="coerce").dropna()
            rec = {"dataset": ds, "variant": var, "metric": m.split(".")[-1],
                   "n_flag": len(a), "mean_flag": a.mean(), "med_flag": a.median(),
                   "zeros_flag": int((a == 0).sum()),
                   "n_rest": len(b), "mean_rest": b.mean(), "med_rest": b.median(),
                   "zeros_rest": int((b == 0).sum()),
                   "delta": a.mean() - b.mean(),
                   "rank_biserial": np.nan, "p": np.nan}
            if len(a) >= 3 and len(b) >= 3 and (a.nunique() > 1 or b.nunique() > 1):
                u, p = mannwhitneyu(a, b, alternative="two-sided")
                rec["rank_biserial"] = 2 * u / (len(a) * len(b)) - 1
                rec["p"] = p
            rows.append(rec)
    return pd.DataFrame(rows)


def mc_flatten_impact(df, degenerate, metrics=sorted(ANSWER_METRICS)):
    """What the degenerate rows do to the cells they sit in: each affected
    (dataset, variant) scored with them and without.

    The point of the table is the ``delta`` column. These rows are an artifact of
    dataset preparation, so whatever they subtract is not a property of the
    system, and a cell whose delta is large should be reported with the
    sensitivity stated rather than as a result.
    """
    if not len(degenerate) or "variant" not in df:
        return pd.DataFrame()
    keys = set(zip(degenerate["source_dataset"], degenerate["variant"]))
    bad = set(zip(degenerate["source_dataset"], degenerate["variant"], degenerate["id"]))
    cols = [m for m in metrics if m in df]
    rows = []
    for ds, var in sorted(keys):
        cell = df[(df["source_dataset"].astype(str) == ds)
                  & (df["variant"].astype(str) == var)]
        is_bad = pd.Series([k in bad for k in zip(cell["source_dataset"],
                                                  cell["variant"], cell["id"])],
                           index=cell.index)
        keep = cell[~is_bad]
        rec = {"dataset": ds, "variant": var, "n": len(cell),
               "n_degenerate": len(cell) - len(keep)}
        for m in cols:
            with_, without = cell[m].mean(), keep[m].mean()
            rec[m.split(".")[-1]] = with_
            rec[f"{m.split('.')[-1]}_excl"] = without
            rec[f"{m.split('.')[-1]}_delta"] = without - with_
        rows.append(rec)
    return pd.DataFrame(rows)


# --- LLMDRS answer leakage ---------------------------------------------------
# LLMDRS items are free-text patient profiles, and their closing sections
# ("Additional Information", the diet history) are written by the same hand as
# the gold recommendation. On some of them that prose stops describing the
# patient and starts prescribing: "Bakytzhan should consume smaller, more
# frequent meals", "his diet requires modifications, including reducing
# saturated fats, choosing healthier fats, and incorporating high fibre foods".
# The question therefore already contains part of the answer it is graded
# against, and a system that merely paraphrases the prompt can collect
# reference-metric credit it did not earn.
#
# Keyed by id, so a redrawn sample silently flags fewer questions instead of
# flagging the wrong ones. This is a property of the DATASET, not of the run,
# which is why the list is a constant here rather than a detector: the leak is a
# shift in the prose's illocutionary force (describing vs. prescribing), and no
# keyword rule separates "she avoids smoke" from "she should avoid smoke"
# reliably enough to carry a caveat in the results chapter.
#
# ADJUDICATION (all 50 stems read against their own gold, 2026-08-26). The
# membership test is narrow on purpose: the stem must state recommendation
# CONTENT, not merely that a recommendation exists. "Her GP suggested she make
# some adjustments" is the question; "limiting sodium intake, increasing fluid
# intake" is the answer. Three of the ten were found by the sweep rather than by
# the initial read, and one initially-flagged item did not survive it:
#
#   +llmdrs_7   "Aida should adopt a balanced, calorie-controlled diet,
#               including whole grains, lean proteins, and an increased intake
#               of fruits and vegetables" — the most explicit stem in the set.
#   +llmdrs_47  shares an eight-word phrase with its own gold's closing summary:
#               "a well-balanced diet with reduced-fat intake" + exercise.
#   +llmdrs_49  states the gold's recommendations #1 and #2 outright ("limiting
#               sodium intake, increasing fluid intake"); tightest leak here.
#   -llmdrs_25  REJECTED. The GP "suggested she make some adjustments" and names
#               none, and the diet history it carries (fried eggs, butter,
#               deep-fried dishes) is the ANTI-pattern the gold argues against —
#               a foil, not a leak. Its gold-vocabulary overlap is 45th of 50.
ANSWER_LEAK_DATASET = "llmdrs"
ANSWER_LEAK_IDS = tuple(f"llmdrs_{n}" for n in
                        (6, 7, 8, 9, 10, 11, 47, 48, 49, 50))

# A SECOND, weaker leak mechanism, kept out of the set above because pooling the
# two would be a category error. Here the stem prescribes nothing; it describes a
# patient who is already compliant, and the gold then endorses that description
# rather than replacing it — llmdrs_29's gold opens "Nurgul is already eating a
# balanced diet ... Continue to prioritize these nutrient-rich foods". The credit
# a model can collect is real but partial (that gold spends most of its length on
# content the stem does not carry: omega-3s, antioxidants, hydration targets),
# which is why llmdrs_29 sits 47th of 50 on vocabulary overlap while belonging in
# the caveat. Reported as a sensitivity row, never merged into the primary set.
ANSWER_LEAK_COMPLIANT_IDS = ("llmdrs_29",)

# Function words dropped before measuring query/gold vocabulary overlap. Short
# and deliberately generic: the point is to stop the overlap being dominated by
# English glue, not to build a stemmer. Tokens of <= 2 characters go too.
_LEAK_STOPWORDS = frozenset("""
a an the and or of to in for with on at is are was were be been being has have
had he she it they them her his their as that this these those not no if then
than by from into out up down over under more most some such can could should
would may might will shall do does did about also which who whom while when
where there here own very just each both any all other same
""".split())


def _content_tokens(text):
    """Lower-cased alphabetic tokens of length > 2 that are not function words."""
    return {w for w in re.findall(r"[a-z]+", str(text).lower())
            if len(w) > 2 and w not in _LEAK_STOPWORDS}


def answer_leak_flag(df, ids=ANSWER_LEAK_IDS, dataset=ANSWER_LEAK_DATASET):
    """Boolean Series over ``df``: does this row's question already state part of
    its own gold answer? False everywhere outside ``dataset``."""
    if "id" not in df:
        return pd.Series(False, index=df.index)
    flag = df["id"].astype(str).isin(set(ids))
    if dataset is not None and "source_dataset" in df:
        flag &= df["source_dataset"].astype(str) == str(dataset)
    # Named, because it is used as a crosstab axis: an unnamed boolean Series
    # inherits ``id`` from the column it was built off and labels the table with it.
    return flag.rename("answer_leak")


def answer_leak_overlap(df, ids=ANSWER_LEAK_IDS, dataset=ANSWER_LEAK_DATASET):
    """One row per LLMDRS QUESTION, with the share of the gold answer's
    vocabulary that already appears in the prompt.

    The manipulation check on the hand-made flag: if the nine ids really do carry
    answer text, they should sit above the rest on a measure that knows nothing
    about the annotation. ``overlap`` is |content words of query ∩ content words
    of gold| / |content words of gold| — a recall of the gold's vocabulary, so a
    long profile is not rewarded for its length the way an F1 would be.

    It is a WEAK check in one specific direction, and the report says so: every
    LLMDRS profile shares vocabulary with its recommendation by construction (the
    foods, the diagnosis, the patient's name), so the floor is well above zero
    and the flagged items are expected to sit higher, not alone. Read it as
    corroboration of the flag's direction, never as a substitute for it.
    """
    need = {"id", "query", "reference_answer"}
    if not need <= set(df):
        return pd.DataFrame()
    q = df if dataset is None or "source_dataset" not in df else \
        df[df["source_dataset"].astype(str) == str(dataset)]
    q = q.drop_duplicates("id")
    if not len(q):
        return pd.DataFrame()
    rows = []
    for _, r in q.iterrows():
        gold = _content_tokens(r["reference_answer"])
        rows.append({
            "id": str(r["id"]),
            "leak": str(r["id"]) in set(ids),
            "query_chars": len(str(r["query"])),
            "gold_chars": len(str(r["reference_answer"])),
            "n_gold_tokens": len(gold),
            "overlap": (len(_content_tokens(r["query"]) & gold) / len(gold)
                        if gold else _NAN),
        })
    return pd.DataFrame(rows).sort_values("overlap", ascending=False)


def _bootstrap_diff_ci(a, b, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the UNPAIRED difference of means ``a - b``.

    The two groups are different questions, so each is resampled independently —
    unlike ``analysis._bootstrap_ci``, which resamples one vector of paired
    differences.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not len(a) or not len(b):
        return _NAN, _NAN
    rng = np.random.default_rng(seed)
    da = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    db = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    lo, hi = np.quantile(da - db, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _holm(pvals):
    """Holm-Bonferroni adjusted p-values, NaNs passed through.

    Holm rather than Bonferroni because it is uniformly more powerful at the same
    family-wise error rate, and the family here is small enough that the
    difference decides whether the one nominally significant cell survives.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, _NAN)
    idx = np.flatnonzero(np.isfinite(p))
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(p[i])))
        out[i] = running
    return out


def answer_leak_contrast(df, metrics=None, level="variant",
                         ids=ANSWER_LEAK_IDS, dataset=ANSWER_LEAK_DATASET):
    """Within LLMDRS: the nine answer-leaking questions against the other 41.

    Run on the PREPARED frame, so the contrast is computed over exactly the cells
    the results chapter reports — evaluator failures dropped, faithfulness and
    reference metrics already nulled on abstentions. A leak-vs-rest table built
    on the raw frame would compare cells the thesis never averages.

    Two levels, because they answer two different questions and have different
    independence properties:

    ``level="variant"``   one test per metric x variant. Each question
                          contributes ONE row to each test, so the samples are
                          independent, and the variant is held fixed — the same
                          reason ``option_request_contrast`` is a within-cell
                          contrast. This is the table for the appendix.
    ``level="question"``  one test per metric, each question contributing the
                          mean of its variant scores. Pooling the raw rows
                          instead would triple n by counting every question three
                          times; averaging first keeps one observation per
                          question and buys the power the per-variant cells lack.
                          Note the cohort shifts slightly per metric: a question
                          whose rag_sc row abstained is averaged over the
                          variants that were scored.

    Reports mean, median, the mean difference with a bootstrap CI, and a
    Mann-Whitney U with rank-biserial (rank-based: these are bounded scores,
    several of them pinned to a rail, and 9-vs-41 is far too small to lean on a
    normal approximation). ``p_holm`` corrects across the rows of the returned
    table — one family, corrected once, rather than a per-call correction the
    caller has to remember to apply.

    Read the effect sizes and the n columns, not the p-values: nine questions
    cannot support a hypothesis test that would survive on its own, and the
    purpose of the table is to bound the size of a possible contamination, not to
    establish one.
    """
    from scipy.stats import mannwhitneyu

    if "source_dataset" not in df or "variant" not in df:
        return pd.DataFrame()
    sub = df[df["source_dataset"].astype(str) == str(dataset)]
    if not len(sub):
        return pd.DataFrame()
    if metrics is None:
        metrics = [m for m in plots.order_metrics(sorted(ANSWER_METRICS | CONTEXT_METRICS))
                   if m in sub and sub[m].notna().any()]
    leak = answer_leak_flag(sub, ids=ids, dataset=dataset)

    rows = []
    for m in metrics:
        if m not in sub:
            continue
        if level == "question":
            wide = sub.assign(_v=pd.to_numeric(sub[m], errors="coerce")) \
                      .groupby("id", observed=True)["_v"].mean().dropna()
            groups = [("pooled", wide[wide.index.isin(set(ids))],
                       wide[~wide.index.isin(set(ids))])]
        else:
            groups = []
            for var in [v for v in ("no_rag", "rag", "rag_sc")
                        if (sub["variant"].astype(str) == v).any()]:
                cell = sub["variant"].astype(str) == var
                vals = pd.to_numeric(sub.loc[cell, m], errors="coerce")
                groups.append((var, vals[leak[cell]].dropna(),
                               vals[~leak[cell]].dropna()))
        for var, a, b in groups:
            if not len(a) or not len(b):
                continue
            lo, hi = _bootstrap_diff_ci(a, b)
            rec = {"metric": m.split(".")[-1], "variant": var,
                   "n_leak": len(a), "mean_leak": a.mean(), "med_leak": a.median(),
                   "n_rest": len(b), "mean_rest": b.mean(), "med_rest": b.median(),
                   "delta": a.mean() - b.mean(), "ci_low": lo, "ci_high": hi,
                   "rank_biserial": _NAN, "p": _NAN}
            if len(a) >= 3 and len(b) >= 3 and (a.nunique() > 1 or b.nunique() > 1):
                u, p = mannwhitneyu(a, b, alternative="two-sided")
                rec["rank_biserial"] = 2 * u / (len(a) * len(b)) - 1
                rec["p"] = float(p)
            rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = _holm(out["p"])
    return out


def _tex_num(v, nd=3, sign=False):
    """One number for the LaTeX table: fixed decimals, en-dash for a missing one.

    Wrapped in math mode so a negative value prints a real minus sign — a table of
    effect sizes typeset with ASCII hyphens is the classic giveaway of a
    pasted-in console dump. ``sign=True`` keeps the ``+`` on positive differences,
    so the direction of every ``delta`` is readable without hunting for the ones
    that lack a minus.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"${v:{'+' if sign else ''}.{nd}f}$"


def _tex_p(v):
    """A p-value for the LaTeX table, with a floor rather than 0.000."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return "$<0.001$" if v < 0.001 else f"${v:.3f}$"


def answer_leak_latex(contrast, label="tab:llmdrs-answer-leak", caption=None,
                      style="full"):
    """``answer_leak_contrast`` as a ``booktabs`` table, ready to paste into the
    appendix.

    Emitted as a file rather than copied out of the console so the appendix table
    is reproducible from the run: re-analysing the same results file rewrites it,
    and a number in the thesis can be traced to the row that produced it.

    The metric name is printed once per group instead of on every row (no
    ``multirow`` dependency — the table needs ``booktabs`` and nothing else), and
    the CI shares a cell with the mean difference, which is the only way ten
    columns fit a portrait page. A contrast with one row per metric — the
    ``level="question"`` cut, whose ``variant`` is ``pooled`` throughout — drops
    the variant column and the group rules with it, since neither separates
    anything there.

    ``style="simple"`` prints the two group means, their difference and a
    p-value, and nothing else. It is the version to put in front of a reader who
    should not have to learn three statistics to read one caveat: the bootstrap
    CI, the rank-biserial effect size and the Holm column all answer questions
    that only arise once the table is being defended, and none of them changes
    what this one says. The multiplicity warning the dropped Holm column carried
    moves into the caption, where it costs a sentence instead of a column — it is
    not discarded, because the whole point of the table is that a lone p below
    0.05 among eight tests is not a finding.
    """
    if not len(contrast):
        return ""
    simple = style == "simple"
    by_variant = contrast["variant"].nunique() > 1
    # n from the widest row rather than hard-coded: abstentions and scorer
    # failures thin individual cells, so the group sizes belong to the data.
    n_leak = int(contrast["n_leak"].max())
    n_rest = int(contrast["n_rest"].max())
    if caption is None:
        caption = (f"LLMDRS questions whose prompt already states part of the gold "
                   f"recommendation ($n={n_leak}$) against the remaining LLMDRS "
                   f"questions ($n={n_rest}$), per metric and variant. "
                   "$\\Delta$ is the difference "
                   "in means (leaking $-$ rest) with a percentile bootstrap 95\\% "
                   "CI; $r_{\\mathrm{rb}}$ is the rank-biserial effect size of a "
                   "two-sided Mann-Whitney $U$ test; $p_{\\mathrm{Holm}}$ corrects "
                   "across all rows of the table.")
    if simple:
        # The two n's stay, folded into one column. They are counts, not a
        # statistic to explain, and without them a reader takes the caption's
        # group sizes to hold on every row — which they do not: a scorer failure
        # or an abstention thins individual metrics (correctness loses one
        # question on each side), and a mean over 9 presented as a mean over 10
        # is the kind of quiet error a simplified table exists to avoid.
        head = ("Metric & " + ("Variant & " if by_variant else "")
                + r"$n$ (leak/rest) & Leaking & Rest & $\Delta$ & $p$ \\")
        spec = ("ll" if by_variant else "l") + "crrrr"
    else:
        head = ("Metric & " + ("Variant & " if by_variant else "")
                + r"$n_{\mathrm{leak}}$ & $\bar{x}_{\mathrm{leak}}$ & "
                  r"$n_{\mathrm{rest}}$ & $\bar{x}_{\mathrm{rest}}$ & "
                  r"$\Delta$ [95\% CI] & "
                  r"$r_{\mathrm{rb}}$ & $p$ & $p_{\mathrm{Holm}}$ \\")
        spec = ("ll" if by_variant else "l") + "rrrrlrrr"
    lines = ["% requires \\usepackage{booktabs}",
             "\\begin{table}[htbp]", "  \\centering",
             f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
             "  \\footnotesize",
             f"  \\begin{{tabular}}{{{spec}}}", "    \\toprule",
             f"    {head}", "    \\midrule"]
    prev = None
    for _, r in contrast.iterrows():
        if by_variant and prev is not None and r["metric"] != prev:
            lines.append("    \\addlinespace")
        name = plots.metric_label(r["metric"]) if r["metric"] != prev else ""
        prev = r["metric"]
        var = f"{plots.variant_label(r['variant'])} & " if by_variant else ""
        if simple:
            lines.append(
                f"    {name} & {var}{int(r['n_leak'])}/{int(r['n_rest'])} & "
                f"{_tex_num(r['mean_leak'])} & {_tex_num(r['mean_rest'])} & "
                f"{_tex_num(r['delta'], sign=True)} & {_tex_p(r['p'])} \\\\")
            continue
        delta = (f"{_tex_num(r['delta'], sign=True)} [{_tex_num(r['ci_low'])}, "
                 f"{_tex_num(r['ci_high'])}]")
        lines.append(
            f"    {name} & {var}{int(r['n_leak'])} & {_tex_num(r['mean_leak'])} & "
            f"{int(r['n_rest'])} & {_tex_num(r['mean_rest'])} & {delta} & "
            f"{_tex_num(r['rank_biserial'])} & {_tex_p(r['p'])} & "
            f"{_tex_p(r['p_holm'])} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# --- NGQA conflict structure -------------------------------------------------
# NGQA asks "is this food healthy for THIS user". The graph answers it with
# ``contradict`` edges (condition -> nutrient tag), and ``dataset/NGQA/NGQA.py``
# turns those into the reference answer: any edge -> "not recommended because
# ...", no edge -> "appears suitable because ...". So ``has_conflict`` is the
# single structural property the reference text hinges on, and it is carried on
# every row as ``dataset_metadata.has_conflict``.
#
# The second axis is NOT in the results file. NGQA's own ``difficulty`` label is
# essentially "how many nutrient tags are relevant, and do they point one way or
# both ways" — the relevant subset lives in the CSV's ``answer_medium`` column,
# and it is a strict subset of the food's tags. Deriving it from the query text
# instead does not work: the full tag list is mixed on almost every food, which
# collapses 50 of the 53 consistent-evidence questions into the mixed group. So
# the split needs the CSV, and every function here degrades to an empty frame
# when it is absent rather than silently substituting the wrong axis.
NGQA_CSV = Path(__file__).resolve().parents[1] / "dataset" / "NGQA" / "NGQA.csv"
# Tags that count against a food. Note ``low_protein``: NGQA's graph uses it as an
# OFFENDING tag (it contradicts a high-protein, weight-gain or opioid-misuse
# profile), while ``NGQA.py``'s ``_build_gold`` classifies it as favourable via
# ``t.startswith('low_')`` and cites it as a reason a food is suitable. The two
# readings disagree on 65% of the conflict-free samples. This list follows the
# graph, because the graph is what ``has_conflict`` is computed from.
NGQA_UNFAVOURABLE_TAGS = frozenset({
    "high_sodium", "high_calorie", "high_cholesterol", "high_sugar",
    "high_saturated_fat", "high_carb", "high_fat",
    "low_protein", "low_fiber", "low_calorie"})
NGQA_GROUPS = ("A_no_conflict", "B_conflict_consistent", "C_conflict_mixed")


def ngqa_relevant_tags(path=NGQA_CSV):
    """``{id: {tag, ...}}`` — the tags NGQA itself says decide each question.

    Read from the CSV's ``answer_medium`` column, which is the gold answer to
    NGQA's medium task ("which nutrient tags determine whether this food is
    healthy for the user"). ``process_csv`` keys samples on the CSV row index
    after skipping rows with an empty required field, so the same skip has to be
    replayed here or every id shifts.

    Returns an empty dict if the CSV is not present; callers treat that as "this
    axis is unavailable" rather than as an error.
    """
    path = Path(path)
    if not path.exists():
        return {}
    required = ["question_hard", "answer_hard", "node_list", "edge_list", "difficulty"]
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if any(not (row.get(c) or "").strip() for c in required):
                continue
            out[f"ngqa_{i}"] = {t.strip() for t in
                                (row.get("answer_medium") or "").split(",") if t.strip()}
    return out


def ngqa_conflict_groups(df, tags=None):
    """Label every NGQA row ``A_no_conflict`` / ``B_conflict_consistent`` /
    ``C_conflict_mixed``; a Series aligned to ``df.index``, NaN off NGQA.

    The three groups separate the two things NGQA's ``difficulty`` label conflates:

    * A vs B is conflict PRESENCE at a matched number of relevant tags. This is
      the contrast that matters, and it is large.
    * C vs B is evidence MIXING at a matched number of conflicts — the food has
      favourable and unfavourable properties at once and the verdict has to weigh
      them. NGQA's "hard" tier is 94% this case.

    Deliberately not a three-tier "difficulty" ordering. Reporting easy/medium/hard
    straight makes hard look EASIEST (accuracy 0.69 vs 0.47 on medium), because
    every hard question carries a conflict while easy/medium are 50/50 and the
    model's verdict is negative on ~95% of rows either way. That is a label-balance
    artifact meeting a response bias, not a difficulty effect, so the grouping
    holds conflict fixed instead.
    """
    if "source_dataset" not in df:
        return pd.Series(np.nan, index=df.index, dtype="object")
    tags = ngqa_relevant_tags() if tags is None else tags
    is_ngqa = df["source_dataset"].astype(str).eq("ngqa")
    conflict = df.get("dataset_metadata.has_conflict")
    if not len(tags) or conflict is None:
        return pd.Series(np.nan, index=df.index, dtype="object")

    def label(idx):
        if not is_ngqa.loc[idx]:
            return np.nan
        if not bool(conflict.loc[idx]):
            return NGQA_GROUPS[0]
        t = tags.get(str(df.at[idx, "id"]))
        if t is None:
            return np.nan
        bad, good = t & NGQA_UNFAVOURABLE_TAGS, t - NGQA_UNFAVOURABLE_TAGS
        return NGQA_GROUPS[2] if (bad and good) else NGQA_GROUPS[1]

    return pd.Series([label(i) for i in df.index], index=df.index, dtype="object")


def ngqa_conflict_contrast(df, groups=None, metrics=sorted(REFERENCE_METRICS)):
    """Reference-metric means per NGQA group, with the B-A and C-B contrasts.
    One row per metric, pooled over the pipeline variants.

    Pooling is at QUESTION level — the three variants of a question are averaged
    first, then tested at n = questions. Stacking the variant rows instead
    triples every question and reports a C-B p of 0.02 where the question-level
    test says 0.28; the means are identical either way, only the significance is
    fabricated.

    Mann-Whitney because the groups are not shifted normals:
    ``ragas_answer_accuracy`` is a five-level judge grid and group A piles up on 0.
    """
    from scipy.stats import mannwhitneyu

    groups = ngqa_conflict_groups(df) if groups is None else groups
    sub = df[groups.notna()].copy()
    if not len(sub):
        return pd.DataFrame()
    sub["_group"] = groups[groups.notna()]
    cols = [m for m in metrics if m in sub]
    sub = sub.groupby(["id", "_group"], observed=True)[cols].mean().reset_index()

    rows = []
    for m in cols:
        g = {k: pd.to_numeric(sub.loc[sub["_group"] == k, m],
                              errors="coerce").dropna() for k in NGQA_GROUPS}
        rec = {"metric": m.split(".")[-1]}
        for short, key in zip("ABC", NGQA_GROUPS):
            rec[f"n_{short}"] = len(g[key])
        for short, key in zip("ABC", NGQA_GROUPS):
            rec[short] = g[key].mean() if len(g[key]) else np.nan
        for lo, hi, name in (("A", "B", "B_minus_A"), ("B", "C", "C_minus_B")):
            a, b = g[NGQA_GROUPS["ABC".index(hi)]], g[NGQA_GROUPS["ABC".index(lo)]]
            rec[name] = a.mean() - b.mean() if len(a) and len(b) else np.nan
            rec[f"p_{name}"] = np.nan
            if len(a) >= 3 and len(b) >= 3 and pd.concat([a, b]).nunique() > 1:
                rec[f"p_{name}"] = mannwhitneyu(a, b, alternative="two-sided")[1]
        rows.append(rec)
    return pd.DataFrame(rows)


# --- Cross-lingual cell: German vs English reference questions ---------------
# The one place in the run where QUESTION LANGUAGE is a manipulated factor.
#
# `dataset/synthetic` generates two context sets, and only one of them exists in
# both languages (see its README, "Hybrid design"):
#
#   cell         n Q   question   system prompt   gold answer   gold chunks
#   enQ_condC     30      en           en             en         3x en
#   enQ_refC      17      en           en             en         1x de + 2x en
#   deQ_refC      21      DE           DE             DE         1x de + 2x en
#
# There is no `deQ_condC`: the condition contexts are English clinical-guideline
# prose bound to an English persona, and no German counterpart was generated.
# So question language is crossed with CONTEXT TYPE, not with context language,
# and the only language-comparable cell is the reference one — same five
# contexts, same life-stage bands, same styling profiles, same gold content, two
# question languages. Everything here is therefore restricted to
# ``context_type == "reference"``. Pooling the condition rows in would compare
# German reference-table questions against English clinical-condition questions
# and report a dataset difference as a language effect.
#
# Two properties of that cell decide every test below:
#
#   - The two language arms are NOT paired. Each language got its own generation
#     pass over the same context, so `..._de_000` and `..._en_000` ask about
#     different nutrients. Every contrast is therefore unpaired (Mann-Whitney),
#     never a signed-rank test over matched ids.
#   - `context_id` is the stratum. It pins the life-stage band AND the styling
#     profile (each context runs one profile), so a language difference that is
#     really a "technical vs lay framing" difference shows up as a discrepancy
#     between the raw and the stratified delta.
CROSSLINGUAL_DATASET = "synthetic_guidelines"
CROSSLINGUAL_CONTEXT_TYPE = "reference"
CROSSLINGUAL_STRATUM = "dataset_metadata.context_id"
# German first everywhere, and every contrast below is oriented de - en, so the
# sign of a delta reads the same way in every table and figure.
CROSSLINGUAL_LANGS = ("de", "en")

# The retriever's corpus. `retrieval/hybrid.py` attaches the FAISS position as
# `chunk_id` (see its comment at the `_metadatas` lookup), and that position IS
# the index into this file, which is what lets a retrieved id be resolved back to
# the language of the chunk it names. Read lazily and cached: it is 2548 records
# of guideline text, and only two fields of each are ever wanted.
CHUNKS_PATH = Path(__file__).resolve().parents[1] / "richtlinien" / "all_chunks.json"
_chunk_lang_cache = {}

# The two reference-value tables the synthetic reference contexts are built from,
# matched on a substring of the source filename. They are the content-equivalent
# pair — same nutrients, age bands aligned from the oldest end — that makes a
# controlled routing measure possible. (The IOM filename's misspelling is the
# corpus's, not a typo here.)
REFERENCE_TABLE_MARKERS = ("DGE-Referenzwerte", "US_intstitute_of_Medicine")

# How a daily reference value is written in each source, and how an answer that
# followed that source tends to write it. DGE prints "4,0 µg/Tag", IOM prints
# "2.4 µg/d" — so the decimal separator and the per-day denominator are a
# PROVENANCE signal as much as a localisation one, and an answer that mixes them
# ("1,7 mg/d") is showing which table it read and which language it was asked in
# at the same time. Both patterns are searched in every answer regardless of its
# language, since the interesting rows are exactly the ones that use the other
# language's convention.
_DEC_COMMA = re.compile(r"\d,\d")
_DEC_POINT = re.compile(r"\d\.\d")
_PER_DAY_DE = re.compile(r"/\s*Tag\b|pro\s+Tag\b|täglich", re.IGNORECASE)
_PER_DAY_EN = re.compile(r"/\s*d\b|/\s*day\b|per\s+day\b|daily\b", re.IGNORECASE)

# A dosed quantity: a number with a nutrition unit attached. It must carry a unit,
# so ages ("65 Jahre") and bare prose numbers are never compared against a table.
#
# The unit list and the two optional groups are all driven by what the two source
# documents actually print — verified by scanning every token following a number
# in the 121 DGE and 130 IOM chunks:
#
#   DGE   mg/Tag (526)  µg/Tag (198)  % der Energie (184)  kcal/Tag (65)
#         g/kg KG/Tag (28)  µg-RAE/Tag (28)  ml/Tag (28)  g/Tag (12)
#   IOM   mg/d (850)  µg/d (548)  g/d (212)  L/d (22)  g/kg/d (20)
#
# Three consequences, each a group below:
#
#   - ``/kg`` MUST be captured, because without it "1,4 g/kg KG/Tag" reads as
#     "1.4 g" and would match a g/Tag value in the other table. Protein per
#     kilogram of body weight and protein per day are different measurements, and
#     silently conflating them manufactures grounding that is not there. It is
#     kept as part of the unit key, so the two can never match each other.
#   - RANGES are DGE-only and common (54 of them: "20-40 µg/Tag", "45-50 % der
#     Energie"). Both endpoints are emitted as separate quantities, so an answer
#     quoting either end of a published range counts as grounded in it.
#   - The QUALIFIER ("-RAE", "-NÄ") is matched and discarded. DGE writes Vitamin A
#     as "µg-RAE/Tag" where IOM writes plain "µg/d" for the same retinol-activity
#     quantity, so keeping the qualifier in the key would make the two tables look
#     as though they never agree on vitamin A.
#
# Units are normalised only WITHIN a dimension and only where the two documents
# genuinely differ in scale (litres to millilitres). Milligrams are deliberately
# NOT converted to micrograms: both documents already use the same mass
# conventions, so a conversion would buy nothing and would risk matching 1 g
# against a 1000 mg that names a different nutrient.
_QUANTITY = re.compile(
    r"(\d{1,4}(?:[.,]\d{1,3})?)"                        # value / range start
    r"(?:\s*[-–]\s*(\d{1,4}(?:[.,]\d{1,3})?))?"         # optional range end
    r"\s*(µg|μg|mcg|mg|kcal|kJ|ml|mL|g|l|L|%)"          # unit (mg before g)
    r"(?:\s*-\s*[A-Za-zÄÖÜäöüαβ]{1,4})?"                # optional -RAE / -NÄ
    r"(\s*/\s*kg)?",                                     # per-kilogram basis
    re.IGNORECASE)

# Which nutrient a number is about. Built from the label vocabulary of the two
# tables themselves — every string before the colon on a "- <Nutrient> (<age,
# sex>): <value> <unit>" line, 47 distinct labels in DGE and 39 in IOM — plus the
# surface forms the ANSWERS use for the same nutrients, which the tables do not
# contain: German compounds ("Selenzufuhr", "Eisenzufuhr"), the bare vitamin
# numbers ("B12-Referenzwert"), and the chemical names a lay answer reaches for
# ("Cobalamin", "Retinol").
#
# The canonical key is what makes the two tables comparable at all: DGE says
# "Eisen" and IOM says "Iron" for the same row of the same nutrient, and without
# a shared key no cross-language provenance test can be written. DGE's qualified
# variants collapse onto the base nutrient — "Zink bei hoher Phytatzufuhr" and
# "Energie bei PAL 1,4" are zinc and energy — since the qualifier selects a row,
# not a different substance.
NUTRIENT_ALIASES = {
    "vitamin_a": ("vitamin a", "retinol"),
    "vitamin_b6": ("vitamin b6", "b6", "pyridoxin"),
    "vitamin_b12": ("vitamin b12", "b12", "cobalamin"),
    "vitamin_c": ("vitamin c", "ascorbin"),
    "vitamin_d": ("vitamin d", "calciferol"),
    "vitamin_e": ("vitamin e", "tocopherol"),
    "vitamin_k": ("vitamin k", "phyllochinon"),
    "thiamin": ("thiamin",),
    "riboflavin": ("riboflavin",),
    "niacin": ("niacin",),
    "folate": ("folat",),
    "biotin": ("biotin",),
    "pantothenic_acid": ("pantothensäure", "pantothenic"),
    "choline": ("cholin",),
    "calcium": ("calcium", "kalzium"),
    "phosphorus": ("phosphor",),
    "magnesium": ("magnesium",),
    "iron": ("eisen", "iron"),
    "zinc": ("zink", "zinc"),
    "iodine": ("jod", "iodine"),
    "selenium": ("selen",),
    "copper": ("kupfer", "copper"),
    "manganese": ("mangan",),
    "chromium": ("chrom",),
    "molybdenum": ("molybdän", "molybdenum"),
    "fluoride": ("fluorid", "fluoride"),
    "sodium": ("natrium", "sodium"),
    "potassium": ("kalium", "potassium"),
    "chloride": ("chlorid", "chloride"),
    "boron": ("boron",),
    "nickel": ("nickel",),
    "vanadium": ("vanadium",),
    "protein": ("protein", "eiweiß"),
    "water": ("wasser", "water"),
    "fiber": ("ballaststoff", "fiber", "fibre"),
    "carbohydrate": ("kohlenhydrat", "carbohydrate"),
    "fat": ("gesamtfett", "fat"),
    # The three fatty-acid classes are DGE-only rows (IOM has no MUFA/PUFA/SFA
    # line), and they are the reason this list is checked against the tables
    # rather than written from memory: without them a "> 10 % der Energie" on a
    # MUFA line has no nutrient of its own and inherits the previous bullet's,
    # which silently filed a fat percentage under iron.
    # "gesättigte" is a substring of "ungesättigte"; the leading \b on every alias
    # is what keeps the saturated pattern from firing inside the unsaturated one.
    "mufa": ("einfach ungesättigte", "monounsaturated"),
    "pufa": ("mehrfach ungesättigte", "polyunsaturated"),
    "saturated_fat": ("gesättigte fettsäuren", "saturated"),
    "linoleic_acid": ("linolsäure", "linoleic"),
    "alpha_linolenic_acid": ("linolensäure", "linolenic"),
    "energy": ("energie", "energy", "kalorien"),
    "alcohol": ("alkohol", "alcohol"),
    "epa_dha": ("epa", "dha"),
}
# Longest surface form first, so "vitamin b12" claims the span before the bare
# "b12" alias can, and "linolensäure" before "linolsäure".
_NUTRIENT_SURFACE = [
    (re.compile(r"\b" + re.escape(s)), s, k)
    for s, k in sorted(((s, k) for k, ss in NUTRIENT_ALIASES.items() for s in ss),
                       key=lambda t: -len(t[0]))
]

# Where a quantity's nutrient is looked for: one table line, or one sentence.
_SEGMENT = re.compile(r"[\n;]+|(?<=[.!?])\s+")
_HYPHEN = re.compile(r"[-–—]")
# "45-50 % der Energie" names Energie next to the number; blanked before mentions
# are located, or every macronutrient percentage is tagged as energy.
_ENERGY_BASIS = re.compile(r"%\s*(?:der\s+energie|of\s+energy|energy)")

# Function words that decide the language of a generated answer. Deliberately
# tiny and hand-checked for cross-language collisions rather than imported from a
# language-id package: this run has exactly two languages, the texts are
# paragraph-length, and adding a dependency to answer a binary question would put
# a model nobody has validated between the results and the thesis.
#
# Every word that exists in BOTH languages is excluded, which is why some obvious
# ones are missing: "was" (de: what), "will" (de: wants), "in", "so", "man", "am"
# and "all" are English stopwords that are also German, and each would score for
# the wrong side on the language it does not belong to.
_DE_MARKERS = frozenset("""
der die das den dem des ein eine einen einer eines und oder aber nicht ist sind
war waren wird werden wurde kann koennen können soll sollte sollten muss müssen
für von mit zum zur auf aus nach über unter durch bei seit gegen ohne als wie
auch noch schon nur sehr mehr weniger etwa täglich pro ich sie er es wir ihre
ihnen sein seine haben hat hatte dass diese dieser dieses jeder alle liegt
beträgt empfohlene empfohlenen zufuhr
""".split())
_EN_MARKERS = frozenset("""
the and but not is are were will would can could should must for from with about
after over under through since against without how also still only very more
less daily per you he she it we they their his her have has had that this these
those each of to at on by recommended intake allowance
""".split())
# Umlauts and eszett are decisive on their own: no English word carries one, and
# a German answer of any length carries several.
_DE_CHARS = "äöüß"


def chunk_corpus(path=CHUNKS_PATH):
    """``{chunk_id: {"lang", "source", "text"}}`` for the retriever's corpus.

    Read once and cached. Everything the cross-lingual block needs from the
    corpus goes through here: the language of a retrieved chunk (routing), the
    document it came from (which reference table), and its text (whether a number
    in the answer actually appears in the evidence).

    Returns an empty dict if the corpus file is absent, so a checkout without
    ``richtlinien/`` degrades to "no routing table" instead of raising half way
    through the report.
    """
    key = str(path)
    if key not in _chunk_lang_cache:
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except (OSError, ValueError):
            _chunk_lang_cache[key] = {}
            return _chunk_lang_cache[key]
        _chunk_lang_cache[key] = {
            i: {"lang": (c.get("metadata") or {}).get("lang"),
                "source": (c.get("metadata") or {}).get("source", ""),
                "text": c.get("text", "")}
            for i, c in enumerate(chunks)
        }
    return _chunk_lang_cache[key]


def chunk_languages(path=CHUNKS_PATH):
    """``({chunk_id: lang}, corpus language mix)``.

    The mix is a Series of shares over the whole corpus, and it is not a
    footnote: German is 121 of 2548 chunks (4.7%), so "the retriever returned
    German context" means something entirely different for a German query than
    the same sentence would in a balanced corpus.

    Read the caveat on ``retrieval_language_routing`` before quoting a number
    against this base rate, though: the whole-corpus mix is the wrong denominator
    for a question whose relevant documents are not a random sample of the corpus.
    ``reference_table_routing`` is the controlled version.
    """
    corpus = chunk_corpus(path)
    langs = {i: c["lang"] for i, c in corpus.items()}
    mix = (pd.Series(list(langs.values())).value_counts(normalize=True)
           if langs else pd.Series(dtype=float))
    return langs, mix


def reference_table_pool(path=CHUNKS_PATH):
    """``({chunk_id: lang}, German share of the pool)`` over the TWO aligned
    reference-value tables, and nothing else.

    This is the control the naive routing number needs. ``dataset/synthetic``
    builds every reference context from one German DGE slice plus the English IOM
    tables for the SAME life-stage band, chosen to maximise shared nutrients (see
    its README) — so within this pool the two languages are near-duplicates by
    construction: same nutrients, same age bands, different authority and
    different language.

    That makes the pool's German share the right denominator for "did the
    retriever choose by language?". It is close to a coin flip (121 DGE chunks
    against 130 IOM, 48%), whereas the whole corpus is 4.7% German — and the gap
    between those two denominators is exactly the confound: reference-table
    questions retrieve reference tables because reference tables are relevant,
    which inflates the German share for a German question no matter how
    language-blind the retriever is. Measured inside the pool, topical relevance
    is held and only the choice of authority is left.
    """
    corpus = chunk_corpus(path)
    pool = {i: c["lang"] for i, c in corpus.items()
            if any(m in c["source"] for m in REFERENCE_TABLE_MARKERS)}
    if not pool:
        return {}, _NAN
    share = sum(1 for v in pool.values() if v == "de") / len(pool)
    return pool, share


def _text_language(text):
    """``"de"`` / ``"en"`` / ``"unknown"`` for one generated answer.

    Marker-word counts plus umlauts, which is enough for a two-language decision
    over paragraph-length text and reports ``unknown`` rather than guessing when
    the evidence is thin (under five words) or tied. The report prints the
    ``unknown`` count, so a detector that starts failing announces itself instead
    of quietly relabelling rows.
    """
    words = re.findall(r"[a-zA-ZäöüÄÖÜß]+", str(text).lower())
    if len(words) < 5:
        return "unknown"
    de = sum(w in _DE_MARKERS for w in words)
    de += sum(any(ch in w for ch in _DE_CHARS) for w in words)
    en = sum(w in _EN_MARKERS for w in words)
    if de == en:
        return "unknown"
    return "de" if de > en else "en"


def crosslingual_frame(df):
    """The reference-context rows of the synthetic set, with the language columns
    every table below reads. Empty frame when the cell is absent or single-language.

    Derived per row, all named without a dot so they never collide with a
    ``dataset_metadata.*`` field:

      ``ctx_frac_de`` / ``ctx_frac_match``  share of the RETRIEVED chunks that are
          German, and that are in the question's own language. The second is the
          routing measure; the first is what it is measured against.
      ``ctx_major_lang``                    the language of the majority of the
          retrieved chunks, and ``ctx_lang_match`` whether that is the question's.
      ``gold_recall_de`` / ``_en`` / ``_all``  share of the row's GOLD chunks that
          were retrieved, split by the gold chunk's own language. Each reference
          gold set is one German DGE slice plus two English IOM tables, so a
          retriever that stays in one language cannot exceed 1/3 or 2/3 on
          ``gold_recall_all`` however good its ranking is — the split columns are
          what separate "missed the evidence" from "could not cross the language".
      ``answer_lang`` / ``answer_lang_match``  did the answer come back in the
          language it was asked in.
      ``answer_chars``                      answer length, because a length
          difference between the arms is a confound for every judge-scored metric
          and has to be visible next to them.

    Abstentions keep their ``answer_lang`` (the canonical rejection string is
    itself translated), which is the point: an abstention in the wrong language
    is still a language failure.
    """
    need = {"source_dataset", "lang", "variant"}
    if not need <= set(df) or "dataset_metadata.context_type" not in df:
        return pd.DataFrame()
    out = df[(df["source_dataset"].astype(str) == CROSSLINGUAL_DATASET)
             & (df["dataset_metadata.context_type"].astype(str)
                == CROSSLINGUAL_CONTEXT_TYPE)].copy()
    if not len(out) or out["lang"].nunique() < 2:
        return pd.DataFrame()

    langs, _ = chunk_languages()

    def _retrieved(ids):
        return [int(i) for i in (ids if isinstance(ids, (list, tuple)) else [])]

    got = out.get("retrieved_context_ids", pd.Series(index=out.index, dtype=object))
    got = got.apply(_retrieved)
    out["ctx_frac_de"] = [np.mean([langs.get(i) == "de" for i in ids]) if ids else _NAN
                          for ids in got]
    out["ctx_frac_match"] = [
        np.mean([langs.get(i) == q for i in ids]) if ids else _NAN
        for ids, q in zip(got, out["lang"].astype(str))
    ]
    out["ctx_major_lang"] = np.where(out["ctx_frac_de"].isna(), None,
                                     np.where(out["ctx_frac_de"] > 0.5, "de", "en"))
    out["ctx_lang_match"] = out["ctx_major_lang"] == out["lang"].astype(str)

    gold = out.get("dataset_metadata.context_chunks",
                   pd.Series(index=out.index, dtype=object))
    for lg in (*CROSSLINGUAL_LANGS, None):
        col = f"gold_recall_{lg or 'all'}"
        vals = []
        for chunks, ids in zip(gold, got):
            g = [int(c["chunk_id"]) for c in (chunks or [])
                 if lg is None or c.get("lang") == lg]
            vals.append(len(set(g) & set(ids)) / len(g) if g else _NAN)
        out[col] = vals

    out["answer_lang"] = out.get("answer", "").apply(_text_language)
    out["answer_lang_match"] = out["answer_lang"] == out["lang"].astype(str)
    out["answer_chars"] = out.get("answer", "").astype(str).str.len()
    return out


def crosslingual_cohort(frame):
    """``context_id`` x ``styling_profile`` x language counts, at QUESTION level.

    The table to read before any contrast below, for two reasons. It shows that
    every context carries both languages — so no stratum drops out of the
    stratified test — and it shows that styling profile is a property of the
    CONTEXT (each runs one profile), which is why ``context_id`` is the stratum
    and ``styling_profile`` is not a second one.
    """
    if not len(frame):
        return pd.DataFrame()
    q = frame.drop_duplicates("id")
    keys = [k for k in (CROSSLINGUAL_STRATUM, "dataset_metadata.styling_profile")
            if k in q]
    if not keys:
        return pd.DataFrame()
    idx = [q[k].astype(str).rename(k.split(".")[-1]) for k in keys]
    return pd.crosstab(idx, q["lang"].astype(str), margins=True, margins_name="All")


def retrieval_language_routing(frame, variants=("rag", "rag_sc")):
    """Does the retriever answer a German question from German documents?

    One row per question language x variant, over the rows that retrieved at all:

      ``frac_de_ctx``     share of retrieved chunks that are German;
      ``frac_match``      share in the QUESTION's language — the routing measure;
      ``corpus_share``    that language's share of the whole corpus;
      ``enrichment``      ``frac_match / corpus_share``. The number that makes the
                          effect legible: retrieval indifferent to language sits
                          at 1.0 by construction, whatever the corpus mix is, so a
                          German arm at 18x and an English arm at 1.0 say that only
                          one of them is being routed.
      ``frac_major_match``  share of ROWS (not chunks) whose retrieved context is
                          majority the question's language.

    Read ``enrichment`` and ``corpus_share`` together and the asymmetry stops
    being surprising: English is 95% of the corpus, so an English query cannot be
    enriched by more than 1.05x however hard it is routed, and the English row is
    a floor check rather than a comparison. It is the German row that carries the
    finding.
    """
    if not len(frame):
        return pd.DataFrame()
    _, mix = chunk_languages()
    sub = frame[frame["variant"].astype(str).isin(variants)]
    sub = sub[sub["ctx_frac_de"].notna()]
    if not len(sub):
        return pd.DataFrame()
    rows = []
    for (lg, var), g in sub.groupby([sub["lang"].astype(str),
                                     sub["variant"].astype(str)], observed=True):
        share = float(mix.get(lg, _NAN))
        match = g["ctx_frac_match"].mean()
        rows.append({
            "lang": lg, "variant": var, "n_rows": len(g),
            "frac_de_ctx": g["ctx_frac_de"].mean(),
            "frac_match": match,
            "corpus_share": share,
            "enrichment": match / share if share else _NAN,
            "frac_major_match": g["ctx_lang_match"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["lang", "variant"])


def gold_recall_by_language(frame, variants=("rag", "rag_sc")):
    """Gold-chunk recall per question language x variant, split by the language of
    the gold chunk.

    The consequence of the routing table for the ID-based retrieval metrics. Every
    reference gold set is language-mixed (one German chunk, two English), so a
    language-locked retriever is capped: ``gold_recall_all`` cannot pass 1/3 for a
    German-only context and 2/3 for an English-only one. Reporting
    ``ragas_id_context_recall`` across the two arms without this split reads that
    ceiling as a retrieval quality difference.

    ``n_gold_de`` / ``n_gold_en`` are carried so the ceiling is visible in the
    table rather than asserted in the prose around it.
    """
    if not len(frame):
        return pd.DataFrame()
    sub = frame[frame["variant"].astype(str).isin(variants)]
    cols = [f"gold_recall_{k}" for k in (*CROSSLINGUAL_LANGS, "all")]
    cols = [c for c in cols if c in sub]
    if not len(sub) or not cols:
        return pd.DataFrame()
    gold = sub.get("dataset_metadata.context_chunks")
    for lg in CROSSLINGUAL_LANGS:
        sub = sub.assign(**{f"n_gold_{lg}": [
            sum(1 for c in (chunks or []) if c.get("lang") == lg) for chunks in gold]})
    keep = [f"n_gold_{lg}" for lg in CROSSLINGUAL_LANGS] + cols
    idm = [c for c in sub.columns if "id_context" in c]
    out = sub.groupby([sub["lang"].astype(str), sub["variant"].astype(str)],
                      observed=True)[keep + idm].mean()
    out.index.names = ["lang", "variant"]
    return out


def reference_table_routing(frame, variants=("rag", "rag_sc")):
    """Routing measured INSIDE the reference-table pool — the confound-controlled
    twin of ``retrieval_language_routing``.

    The naive measure compares the German share of what was retrieved against
    German's 4.7% share of the whole corpus, and that comparison is confounded:
    these questions ask for nutrient reference values, the two reference-value
    tables are the relevant documents, and one of the two is German. A German
    share far above 4.7% is therefore the expected result of a perfectly
    language-blind retriever that is merely good at its job.

    So restrict to the retrieved chunks that came from either table and ask which
    one was chosen. Inside that pool the two are near-duplicates — same nutrients,
    aligned age bands, different authority and language — and the pool is 48%
    German, so a language-blind retriever lands near 0.48 in BOTH arms.

    Columns: ``frac_from_pool`` (how much of the retrieved context came from the
    two tables at all, i.e. whether the question was on-topic for this control),
    ``frac_de_in_pool`` (the controlled routing measure), ``pool_share`` (0.48)
    and ``lift`` against it. The distance between the two arms' ``frac_de_in_pool``
    is the language effect with topical relevance held.
    """
    if not len(frame):
        return pd.DataFrame()
    corpus = chunk_corpus()
    pool, share = reference_table_pool()
    if not pool:
        return pd.DataFrame()
    sub = frame[frame["variant"].astype(str).isin(variants)]
    if not len(sub):
        return pd.DataFrame()
    rows = []
    for (lg, var), g in sub.groupby([sub["lang"].astype(str),
                                     sub["variant"].astype(str)], observed=True):
        ids = [int(i) for lst in g["retrieved_context_ids"] for i in (lst or [])]
        if not ids:
            continue
        in_pool = [i for i in ids if i in pool]
        n_de = sum(1 for i in in_pool if pool[i] == "de")
        frac_de = n_de / len(in_pool) if in_pool else _NAN
        rows.append({
            "lang": lg, "variant": var, "n_rows": len(g), "n_chunks": len(ids),
            "frac_from_pool": len(in_pool) / len(ids),
            "n_in_pool": len(in_pool),
            "frac_de_in_pool": frac_de,
            "pool_share": share,
            "lift": frac_de / share if share else _NAN,
        })
    return pd.DataFrame(rows).sort_values(["lang", "variant"])


def _dosed_quantities(text):
    """The set of ``(value, unit)`` pairs a text states, normalised so a German
    and an English statement of the same quantity compare equal.

    Normalising is the whole point: DGE writes ``4,0 µg/Tag`` and IOM writes
    ``2.4 µg/d``, so without it a German answer could never be matched against an
    English table and every cross-language comparison would return "not found"
    for reasons of punctuation. What is normalised:

      - the decimal separator, comma and point to one value;
      - microgram across its three spellings (``µg``, ``μg`` — different
        codepoints — and ``mcg``);
      - litres to millilitres, the one place the two documents use different
        scales for the same dimension (IOM ``0.7 L/d``, DGE ``620 ml/Tag``);
      - the ``-RAE`` / ``-NÄ`` qualifier, dropped, since DGE carries it where IOM
        does not for the same retinol-activity quantity.

    What is deliberately NOT normalised is the per-kilogram basis: ``g/kg`` stays
    a distinct unit from ``g``, so a protein-per-body-weight value can never match
    a protein-per-day one. A range emits both of its endpoints.
    """
    return {(v, u) for _, v, u in nutrient_quantities(text)}


def _match_values(match):
    """``(value, unit)`` for each endpoint of one ``_QUANTITY`` match."""
    lo, hi, unit, per_kg = match.groups()
    u = unit.lower().replace("μ", "µ").replace("mcg", "µg")
    scale = 1000.0 if u == "l" else 1.0
    if u == "l":
        u = "ml"
    if per_kg:
        u += "/kg"
    out = []
    for value in (lo, hi):
        if not value:
            continue
        try:
            out.append((round(float(value.replace(",", ".")) * scale, 3), u))
        except ValueError:
            continue
    return out


def _nutrient_mentions(segment):
    """``[(position, canonical nutrient)]`` for every nutrient named in one
    already-normalised segment.

    Longest surface form first, and a span already claimed by a longer name is not
    re-matched — otherwise "Vitamin B12" would also register the bare "b12" alias
    at an offset two characters to the right, and a quantity sitting between two
    nutrients could be assigned to the phantom.

    Each alias is anchored with ``\\b`` at its START but deliberately not at its
    end. That is what makes German compounding work: "Selenzufuhr" and
    "Eisenzufuhr" name selenium and iron and must match, while "Kreisen" must not
    match "eisen" — a leading word boundary buys the first without the second.
    """
    hits, taken = [], []
    for pattern, surface, canonical in _NUTRIENT_SURFACE:
        for m in pattern.finditer(segment):
            i = m.start()
            if any(a <= i < b for a, b in taken):
                continue
            taken.append((i, i + len(surface)))
            hits.append((i, canonical))
    return sorted(hits)


def text_nutrients(text):
    """The canonical nutrients a text names, in order of first mention."""
    seen = []
    for _, canonical in _nutrient_mentions(_HYPHEN.sub(" ", str(text).lower())):
        if canonical not in seen:
            seen.append(canonical)
    return seen


def nutrient_quantities(text, fallback=None):
    """``{(nutrient, value, unit)}`` — every quantity the text states, each tagged
    with the nutrient it is about.

    Value and unit alone are not an identity. Both tables are dense grids of
    numbers, so "700 µg" occurs in each of them for DIFFERENT nutrients, and a
    provenance test keyed on the number alone credits a match that never happened.
    Tagging the nutrient is what makes "this answer quotes the German table's
    vitamin A figure" a checkable statement rather than a coincidence of digits.

    Each quantity is assigned the nutrient named NEAREST to it within its own
    segment (lines, and sentences inside a line), which resolves both shapes the
    data comes in: a table line names one nutrient and one value, and an answer
    sentence names the nutrient a few words from the number. A segment with a
    quantity but no nutrient inherits the last one named — the shape of "Die
    Vitamin-D-Referenzwerte unterscheiden sich nicht; der Schätzwert liegt bei
    20 µg/Tag", where the nutrient is in the first clause and the number in the
    second.

    ``fallback`` seeds that carry-over, and the caller passes the QUESTION's
    nutrient: some answers never name what they are about ("die empfohlene Zufuhr
    für Männer bei 950 µg-RAE/Tag"), and the question always does. Without it
    those quantities are untaggable and would silently drop out of the provenance
    table; with it they are attributed to what was asked. When the question names
    more than one nutrient the caller passes ``None`` rather than guessing, and
    the quantity is emitted with a nutrient of ``None`` so it can be counted as
    unattributed instead of being matched wrongly.

    Two length-preserving normalisations, and both have to preserve length because
    mentions are located in one string and quantities in the other while positions
    are compared between them: hyphens become spaces (so "Vitamin-D-Referenzwerte"
    names vitamin D) and the "% der Energie" / "% of energy" basis is blanked. The
    second is not cosmetic — that phrase names "Energie" right beside the number,
    which on a "Gesamtfett ... 45-50 % der Energie" line beats the actual nutrient
    on proximity and tags a fat value as energy.
    """
    out, last = set(), fallback
    low = str(text).lower()
    for segment in _SEGMENT.split(low):
        if not segment.strip():
            continue
        normalised = _HYPHEN.sub(" ", segment)
        normalised = _ENERGY_BASIS.sub(
            lambda m: "%" + " " * (len(m.group(0)) - 1), normalised)
        mentions = (_nutrient_mentions(normalised)
                    if len(normalised) == len(segment) else [])
        # Quantities come from the UNnormalised segment: hyphen-to-space would
        # turn the range "20-40 µg/Tag" into two unrelated tokens and lose its
        # lower endpoint.
        for m in _QUANTITY.finditer(segment):
            nutrient = (min(mentions, key=lambda h: abs(h[0] - m.start()))[1]
                        if mentions else last)
            for value, unit in _match_values(m):
                out.add((nutrient, value, unit))
        if mentions:
            last = mentions[-1][1]
    return out


def quantity_notation_audit(frame, variants=("rag", "rag_sc")):
    """Does a German answer use German NOTATION for its numbers?

    Answering in the right language is not the same as localising the values, and
    for a reference-value system the second is what a user acts on: "5,5 mg/Tag"
    and "5.5 mg/day" are the same quantity written in two conventions, and only
    one of them is the convention a German reader parses correctly at a glance.
    ``answer_language_audit`` cannot see this — the prose can be flawless German
    around a number written the American way.

    Per language arm, over answered rows that state at least one quantity:
    the share of answers using a decimal comma / decimal point, and the share
    using a German ("/Tag", "pro Tag", "täglich") / English ("/d", "/day",
    "daily") per-day denominator. An answer can count in both denominator columns
    if it uses both.

    Read the OFF-DIAGONAL cells, not the diagonal. They are not merely
    localisation slips: DGE and IOM print their values in their own conventions,
    so a German answer carrying "/d" is usually one that read the English table,
    and the notation is reporting the answer's provenance. ``quantity_provenance``
    tests that directly.
    """
    if not len(frame):
        return pd.DataFrame()
    sub = frame[frame["variant"].astype(str).isin(variants)
                & ~ra._abstained(frame)]
    if not len(sub):
        return pd.DataFrame()
    rows = []
    for lg, g in sub.groupby(sub["lang"].astype(str), observed=True):
        answers = [str(a) for a in g["answer"] if _dosed_quantities(a)]
        if not answers:
            continue
        n = len(answers)
        rows.append({
            "lang": lg, "n_answers_with_a_quantity": n,
            "dec_comma": sum(bool(_DEC_COMMA.search(a)) for a in answers) / n,
            "dec_point": sum(bool(_DEC_POINT.search(a)) for a in answers) / n,
            "per_day_de": sum(bool(_PER_DAY_DE.search(a)) for a in answers) / n,
            "per_day_en": sum(bool(_PER_DAY_EN.search(a)) for a in answers) / n,
        })
    return pd.DataFrame(rows)


def quantity_provenance(frame, variants=("rag", "rag_sc")):
    """Where did each number in the answer come from — the German table, the
    English one, both, or nowhere?

    The question this cell was built to ask. Each reference context pairs a DGE
    slice with the IOM tables for the same life-stage band, and the two DISAGREE
    on many nutrients (Vitamin B12 for men 65+: DGE 4.0 µg/Tag, IOM 2.4 µg/d;
    Vitamin E: 8 against 15 mg). So a question about one nutrient and one age band
    has two defensible answers, and which one the user gets depends on which table
    was retrieved — which, per ``reference_table_routing``, depends on the language
    they asked in.

    One row per answered row that states a quantity, counting its quantities:

    Every comparison is on ``(nutrient, value, unit)`` triples, never on the number
    alone. Both tables are dense grids, so the same figure recurs across unrelated
    nutrients — matching on value alone credits "the answer quotes DGE's vitamin A"
    to an answer that merely happened to say 700 of something else. Tagging the
    nutrient is what makes each column below a checkable claim; the cost is that a
    quantity whose nutrient cannot be identified is reported separately rather than
    guessed at.

    Attribution is against the RETRIEVED context (the ``from_*`` columns), not the
    gold one. Retrieval reaches the gold chunks on well under a third of rows, so
    an answer's number usually comes from a reference-table chunk that is not the
    gold chunk; scored against gold alone those all land in "neither", which tells
    you nothing about which authority the user was given. The ``gold_*`` columns
    keep that stricter view alongside.

    These are ATTRIBUTION columns, not correctness ones: they say which document
    supplied the number, never that the number is right for the age band asked
    about. Auditing the 90 tagged quantities in the 20260812 run against the
    life-stage band of the supplying chunk, exactly ONE (a German iron value) came
    from a different band. The rest of the non-gold matches were the same band:
    German from further slices of the same DGE age-group page (the page spans
    several chunks and the gold context carries only one), English entirely from
    the IOM RDA/AI table where the gold context pairs DGE with the IOM *EAR* table
    for the same life-stage group — a different statistic, so the figures differ
    legitimately. Use ``gold_dge_only + gold_iom_only + gold_both`` for the
    correctness view (29/58 German, 13/32 English) and read the ``from_*`` columns
    only as routing evidence.

    One row per answered row that states a quantity, counting its quantities:

      ``from_dge`` / ``from_iom``  the nutrient AND its value appear in a retrieved
          chunk from exactly one of the two reference tables — the authority the
          answer actually followed.
      ``from_both``   a value the two tables share, so no authority is being
          chosen; excluded from ``frac_dge_when_they_differ``.
      ``from_other``  taken from a retrieved document that is neither table (EFSA,
          the NIH fact sheets, the national guidelines).
      ``gold_dge_only`` / ``gold_iom_only``  the same test against the gold
          context's two tables. These are the quantities on which the two
          authorities differ, and the count says which one the answer followed.
      ``gold_both``   the same nutrient carries the same value in BOTH gold tables, so
          the answer is not choosing between them and the quantity carries no
          routing signal. Excluded from ``frac_dge_when_they_differ`` for that
          reason. Usually this is genuine agreement (vitamin D is 20 µg in each),
          but the match is on nutrient and value and NOT on the life-stage row: a
          German thiamin value for women 51-65 also lands here if the English
          table happens to print the same figure for adolescents. So ``both`` is
          an upper bound on agreement between the authorities. It only decides how
          many quantities are set aside, never the direction of the fraction.
      ``gold_neither``  the pair is in neither gold table. Usually the
          answer is quoting a chunk that was retrieved instead of the gold one,
          which is what ``in_retrieved`` separates out.
      ``unattributed``  the quantity could not be tied to a nutrient by the answer
          or by its question, so it is counted but excluded from every provenance
          column above. Reported rather than hidden: it is the coverage figure for
          this whole table, and a high value would mean the provenance columns
          describe a minority of the numbers.
      ``in_retrieved`` / ``ungrounded``  found / not found in the context actually
          RETRIEVED for that row, again nutrient-matched. This is the grounding
          check, and the one language-neutral faithfulness measure in the analysis:
          a quantity nobody retrieved is unsupported by definition, no judge and no
          NLI model is involved, and both arms are held to the same standard. Read
          it against the faithfulness scorers that disagree about the languages —
          a hard check saying the arms are equally grounded is evidence about the
          SCORERS.

    Beware the direction of the grounding check's error: a number can be correct
    and count as ungrounded (the model knew it without retrieving it), and it can
    be wrong and count as grounded (the right nutrient copied from the wrong age
    band — nutrient and value are matched, the life-stage row is not).
    """
    if not len(frame):
        return pd.DataFrame()
    corpus = chunk_corpus()
    if not corpus:
        return pd.DataFrame()
    sub = frame[frame["variant"].astype(str).isin(variants)
                & ~ra._abstained(frame)]
    rows = []
    for _, r in sub.iterrows():
        # The question's nutrient seeds the answer's tagging: some answers give a
        # value without ever naming what it is a value OF, and the question always
        # names it. Only when the question names exactly one — otherwise there is
        # nothing to disambiguate with and the quantity stays unattributed.
        asked = text_nutrients(r.get("query", ""))
        aq = nutrient_quantities(r.get("answer", ""),
                                 fallback=asked[0] if len(asked) == 1 else None)
        if not aq:
            continue
        gold_de, gold_en = set(), set()
        for c in (r.get("dataset_metadata.context_chunks") or []):
            text = corpus.get(int(c["chunk_id"]), {}).get("text", "")
            (gold_de if c.get("lang") == "de" else gold_en).update(
                nutrient_quantities(text))
        # The RETRIEVED context, split by which document each chunk came from.
        # This is the attribution that matters: retrieval reaches the gold chunks
        # on well under a third of rows, so an answer's number usually comes from
        # a reference-table chunk that is correct but not the gold one, and
        # attributing only against the gold context files most of them under
        # "neither" — true, and useless.
        retr_dge, retr_iom, retr_other = set(), set(), set()
        for i in (r.get("retrieved_context_ids") or []):
            chunk = corpus.get(int(i), {})
            source = chunk.get("source", "")
            quantities = nutrient_quantities(chunk.get("text", ""))
            if REFERENCE_TABLE_MARKERS[0] in source:
                retr_dge |= quantities
            elif REFERENCE_TABLE_MARKERS[1] in source:
                retr_iom |= quantities
            else:
                retr_other |= quantities
        retrieved = retr_dge | retr_iom | retr_other
        # A quantity with no nutrient can be compared against nothing, so it is
        # held out of every provenance column rather than counted as a miss —
        # "not found in the German table" and "we could not tell what it is" are
        # different statements and only one of them is about the system.
        tagged = {q for q in aq if q[0] is not None}
        rows.append({
            "id": r["id"], "lang": str(r["lang"]), "variant": str(r["variant"]),
            "n_qty": len(aq),
            "unattributed": len(aq) - len(tagged),
            # against the RETRIEVED context — the primary attribution
            "from_dge": len(tagged & retr_dge - retr_iom),
            "from_iom": len(tagged & retr_iom - retr_dge),
            "from_both": len(tagged & retr_dge & retr_iom),
            "from_other": len(tagged & retr_other - retr_dge - retr_iom),
            "ungrounded": len(tagged - retrieved),
            "in_retrieved": len(tagged & retrieved),
            # against the GOLD context — kept as the stricter secondary view
            "gold_dge_only": len(tagged & gold_de - gold_en),
            "gold_iom_only": len(tagged & gold_en - gold_de),
            "gold_both": len(tagged & gold_de & gold_en),
            "gold_neither": len(tagged - gold_de - gold_en),
        })
    return pd.DataFrame(rows)


def quantity_provenance_summary(prov):
    """``quantity_provenance`` totalled per language: which authority's numbers
    each arm ended up quoting, and how many of its numbers were grounded.

    ``frac_dge_when_they_differ`` is the headline: over the quantities that
    distinguish the two tables, the share taken from the German one. A retriever
    and a reader that were both indifferent to language would put the two arms at
    the same value here; the distance between them is the language effect on WHICH
    REFERENCE VALUE A USER IS TOLD, which is the practical consequence the score
    tables cannot express.
    """
    if not len(prov):
        return pd.DataFrame()
    g = prov.groupby("lang")[["n_qty", "unattributed",
                              "from_dge", "from_iom", "from_both", "from_other",
                              "in_retrieved", "ungrounded",
                              "gold_dge_only", "gold_iom_only", "gold_both",
                              "gold_neither"]].sum()
    tagged = g["n_qty"] - g["unattributed"]
    differ = g["from_dge"] + g["from_iom"]
    g["n_tagged"] = tagged
    g["n_distinguishing"] = differ
    g["frac_dge_when_they_differ"] = np.where(differ > 0, g["from_dge"] / differ,
                                              _NAN)
    gold_differ = g["gold_dge_only"] + g["gold_iom_only"]
    g["frac_dge_gold_basis"] = np.where(gold_differ > 0,
                                        g["gold_dge_only"] / gold_differ, _NAN)
    # Over the TAGGED quantities, not all of them: an untaggable number was never
    # tested for grounding, so counting it as ungrounded would report a limit of
    # the nutrient tagger as a hallucination by the system.
    g["frac_ungrounded"] = np.where(tagged > 0, g["ungrounded"] / tagged, _NAN)
    return g


def answer_language_audit(frame):
    """Did the answer come back in the language it was asked in? Counts per
    question language x variant x detected answer language.

    A cheap check that is worth its space precisely when it finds nothing: language
    drift (a German question answered in English because the corpus and the model's
    training data are English-heavy) is the failure this cell exists to catch, and
    a results chapter that reports German scores without it cannot say whether a low
    score is a bad German answer or an English one.

    ``unknown`` is the detector declining rather than a third language; a non-zero
    count there is a reason to read those rows, not to average them in.
    """
    if not len(frame) or "answer_lang" not in frame:
        return pd.DataFrame()
    return pd.crosstab([frame["lang"].astype(str), frame["variant"].astype(str)],
                       frame["answer_lang"], margins=True, margins_name="All")


def _stratified_delta(values, is_de, strata):
    """Stratum-weighted mean difference (de - en), strata with only one language
    dropped. Weighted by stratum size so a five-question context does not count as
    much as a ten-question one."""
    diffs, weights = [], []
    for s in pd.unique(strata):
        k = strata == s
        a, b = values[k & is_de], values[k & ~is_de]
        if len(a) and len(b):
            diffs.append(a.mean() - b.mean())
            weights.append(len(a) + len(b))
    if not diffs:
        return _NAN
    return float(np.average(diffs, weights=weights))


def _stratified_perm_p(values, is_de, strata, n_perm=5000, seed=0):
    """Two-sided p for ``_stratified_delta`` by permuting the language label WITHIN
    each stratum.

    The right test for this cell. The arms are unpaired but not unstructured: both
    languages were generated from the same five contexts, and a context fixes the
    life-stage band and the styling profile. Shuffling language within a context
    holds both of those and asks only whether the label carries anything — which a
    Mann-Whitney over the pooled rows cannot do, and which no normal approximation
    should be trusted for at 21 vs 17.

    The ``+1`` in numerator and denominator is the standard Monte-Carlo correction:
    the observed assignment is one of the permutations, so a p of exactly 0 is not
    an outcome the test can produce.
    """
    obs = _stratified_delta(values, is_de, strata)
    if not np.isfinite(obs):
        return _NAN, obs
    rng = np.random.default_rng(seed)
    idx = np.arange(len(values))
    null = []
    for _ in range(n_perm):
        perm = is_de.copy()
        for s in pd.unique(strata):
            k = idx[strata == s]
            perm[k] = rng.permutation(is_de[k])
        null.append(_stratified_delta(values, perm, strata))
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not len(null):
        return _NAN, obs
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (len(null) + 1)
    return float(p), obs


def language_contrast(frame, metrics=None, level="variant", n_perm=5000):
    """German questions against English ones, per metric.

    Run on the PREPARED frame, so the contrast is computed over exactly the cells
    the results chapter reports — evaluator failures dropped, faithfulness and
    reference metrics already nulled on abstentions.

    Two levels, as in ``answer_leak_contrast`` and for the same reasons:

    ``level="variant"``   one test per metric x variant. Each question contributes
                          one row to each test, and the variant is held fixed.
    ``level="question"``  one test per metric, each question entering once as the
                          mean of its variant scores. Pooling the raw rows instead
                          would triple n on three correlated observations of the
                          same question.

    Columns: the two group means and medians, the mean difference ``delta``
    (de - en) with an unpaired bootstrap CI, a Mann-Whitney rank-biserial and
    p, and Holm across the rows of the returned table.

    Then the two columns that make this cell interpretable rather than merely
    tested. ``delta_strat`` is the same difference computed within ``context_id``
    and pooled, and ``p_perm`` is its within-stratum permutation p. Read
    ``delta`` against ``delta_strat`` FIRST: if they agree the difference is not
    the styling profile or the life-stage band (each context runs one profile, so
    those are what stratification removes), and only then is the unstratified
    p worth reading at all.
    """
    from scipy.stats import mannwhitneyu

    if not len(frame):
        return pd.DataFrame()
    if metrics is None:
        metrics = [m for m in plots.order_metrics(
            sorted(ANSWER_METRICS | CONTEXT_METRICS))
            if m in frame and frame[m].notna().any()]
    strat_col = CROSSLINGUAL_STRATUM if CROSSLINGUAL_STRATUM in frame else None

    rows = []
    for m in metrics:
        if m not in frame:
            continue
        if level == "question":
            keys = ["id"] + [k for k in ("lang", strat_col) if k]
            q = (frame.assign(_v=pd.to_numeric(frame[m], errors="coerce"))
                 .groupby(keys, observed=True)["_v"].mean().dropna().reset_index())
            cells = [("pooled", q)]
        else:
            keys = [k for k in ("lang", strat_col) if k]
            cells = []
            for var in [v for v in ev.VARIANT_ORDER
                        if (frame["variant"].astype(str) == v).any()]:
                g = frame[frame["variant"].astype(str) == var]
                g = g.assign(_v=pd.to_numeric(g[m], errors="coerce"))
                cells.append((var, g[g["_v"].notna()][keys + ["_v"]]))
        for var, cell in cells:
            lang = cell["lang"].astype(str)
            a = cell.loc[lang == "de", "_v"]
            b = cell.loc[lang == "en", "_v"]
            if not len(a) or not len(b):
                continue
            lo, hi = _bootstrap_diff_ci(a, b)
            rec = {"metric": m.split(".")[-1], "variant": var,
                   "n_de": len(a), "mean_de": a.mean(), "med_de": a.median(),
                   "n_en": len(b), "mean_en": b.mean(), "med_en": b.median(),
                   "delta": a.mean() - b.mean(), "ci_low": lo, "ci_high": hi,
                   "rank_biserial": _NAN, "p": _NAN,
                   "delta_strat": _NAN, "p_perm": _NAN}
            if len(a) >= 3 and len(b) >= 3 and (a.nunique() > 1 or b.nunique() > 1):
                u, p = mannwhitneyu(a, b, alternative="two-sided")
                rec["rank_biserial"] = 2 * u / (len(a) * len(b)) - 1
                rec["p"] = float(p)
                if strat_col:
                    p_perm, delta_s = _stratified_perm_p(
                        cell["_v"].to_numpy(dtype=float),
                        (lang == "de").to_numpy(),
                        cell[strat_col].astype(str).to_numpy(),
                        n_perm=n_perm)
                    rec["delta_strat"], rec["p_perm"] = delta_s, p_perm
            rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = _holm(out["p"])
        out["p_perm_holm"] = _holm(out["p_perm"])
    return out


def language_variant_effects(frame, metrics=None,
                             comparisons=(("rag", "no_rag"), ("rag_sc", "rag"))):
    """``compare_variants`` run inside each language arm: does retrieval buy the
    same thing in German as in English?

    The interaction the per-metric contrast cannot show. A language difference in
    the LEVEL of a metric is confounded with everything that differs between two
    sets of questions; a language difference in the EFFECT of adding retrieval is
    paired within each question, so the question itself cancels. That makes this
    the more defensible half of the language analysis even though it has the
    smaller headline.

    Read ``n_pairs`` and ``n_nontied`` before the p-values: 21 and 17 questions
    split three ways leaves cells where an effect of identical size is significant
    in one arm and not the other purely on n, and reporting that as "retrieval
    helps German but not English" would be an artifact of the split.
    """
    if not len(frame):
        return pd.DataFrame()
    if metrics is None:
        metrics = [m for m in plots.order_metrics(sorted(REFERENCE_METRICS))
                   if m in frame and frame[m].notna().any()]
    rows = {}
    for lg in CROSSLINGUAL_LANGS:
        arm = frame[frame["lang"].astype(str) == lg]
        if not len(arm):
            continue
        for m in metrics:
            for a, b in comparisons:
                cmp = ev.compare_variants(arm, m, a=a, b=b)
                if "overall" not in cmp.index:
                    continue
                rows[(lg, m.split(".")[-1], f"{a}_vs_{b}")] = cmp.loc["overall"]
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).T
    out.index.names = ["lang", "metric", "comparison"]
    return out


def context_language_mismatch(frame, metrics=None, variants=("rag", "rag_sc")):
    """The rows whose retrieved context is NOT majority in the question's language,
    listed one by one rather than averaged.

    Because the retriever routes by language, this cell is tiny and nobody
    designed it: it is what is left over when routing fails. On this run that is a
    handful of German questions served English tables, and one English question
    served a German one — far too few for a cell mean, which is exactly why the
    function returns the rows instead of a groupby.

    It is worth listing anyway, because it is the only place where the ANSWER and
    the PREMISE are in different languages, and that is the condition under which
    an NLI-based faithfulness scorer is being asked to do something it was not
    built for. A scorer whose score collapses on these rows while the
    reference-graded metrics on the same rows do not is telling you about itself.
    Treat it as a case series to read, never as a tested contrast.
    """
    if not len(frame) or "ctx_lang_match" not in frame:
        return pd.DataFrame()
    sub = frame[frame["variant"].astype(str).isin(variants)
                & frame["ctx_major_lang"].notna()
                & ~frame["ctx_lang_match"]]
    if not len(sub):
        return pd.DataFrame()
    if metrics is None:
        metrics = [m for m in plots.order_metrics(
            sorted(ANSWER_METRICS | CONTEXT_METRICS)) if m in sub]
    cols = ["id", "variant", "lang", "ctx_major_lang", "ctx_frac_de",
            "retrieval_best"] + list(metrics)
    out = sub[[c for c in cols if c in sub]].copy()
    out.insert(4, "abstained", ra._abstained(sub).to_numpy())
    return out.rename(columns={m: m.split(".")[-1] for m in metrics}) \
              .sort_values(["lang", "id", "variant"])


def scorer_direction_conflict(contrast, tol=0.02):
    """Within each metric family, do the scorers even agree on WHICH language
    scored higher?

    The measurement-side result, computed rather than left to the reader to spot
    across eight rows of a contrast table. One row per family (the sets defined at
    the top of this module — faithfulness, relevancy, reference), listing each
    member's German-minus-English delta and a verdict:

      ``agree``           two or more members move by more than ``tol`` and all
                          point the same way;
      ``CONFLICT``        at least two point in OPPOSITE directions, both by more
                          than ``tol``. The family has no scorer-independent answer
                          in this run, and "language X is more faithful" cannot be
                          written without naming the scorer that says so;
      ``uncorroborated``  exactly one member moves and the rest are flat. Not a
                          conflict, but not agreement either — nothing confirms it,
                          which is worth distinguishing from a family where two
                          scorers independently found the same thing;
      ``no effect``       no member moves by more than ``tol``, so there is nothing
                          for them to agree or disagree about.

    ``tol`` exists so a family is not called into conflict by two scorers sitting
    either side of zero at 0.003 and -0.001. It is a threshold on the EFFECT, not
    on significance: a conflict between two large deltas is a finding whether or
    not either survives a correction at these n, and a conflict between two
    negligible ones is not a finding whichever way their p-values fall.
    """
    if not len(contrast):
        return pd.DataFrame()
    families = {
        "faithfulness": FAITHFULNESS_METRICS,
        "relevancy": RELEVANCY_METRICS,
        "reference": REFERENCE_METRICS,
    }
    rows = []
    for name, cols in families.items():
        short = {c.split(".")[-1] for c in cols}
        sub = contrast[contrast["metric"].isin(short)]
        if len(sub) < 2:
            continue
        deltas = {m: float(d) for m, d in zip(sub["metric"], sub["delta"])
                  if np.isfinite(d)}
        moved = {m: d for m, d in deltas.items() if abs(d) > tol}
        signs = {np.sign(d) for d in moved.values()}
        if not moved:
            verdict = "no effect"
        elif len(signs) > 1:
            verdict = "CONFLICT"
        elif len(moved) == 1:
            verdict = "uncorroborated"
        else:
            verdict = "agree"
        rows.append({
            "family": name,
            "n_scorers": len(deltas),
            "n_moved": len(moved),
            "verdict": verdict,
            "favours": "-" if len(signs) != 1 else ("de" if signs == {1.0} else "en"),
            "deltas": ", ".join(f"{m} {d:+.3f}" for m, d in deltas.items()),
        })
    return pd.DataFrame(rows)


def language_contrast_latex(contrast, label="tab:crosslingual-de-en", caption=None,
                            style="simple"):
    """``language_contrast`` as a ``booktabs`` table for the results chapter.

    ``style="simple"`` is the one to print: metric, the two group means, the
    difference, its stratified twin and the permutation p. That is the argument —
    how big is the gap, does it survive holding the context fixed, and could the
    labels have produced it by chance. ``style="full"`` adds the bootstrap CI, the
    rank-biserial, the Mann-Whitney p and both Holm columns, for the version to
    produce if a number is ever challenged.

    Emitted as a file rather than copied from the console so a table in the thesis
    is reproducible from the run that produced it.
    """
    if not len(contrast):
        return ""
    simple = style == "simple"
    by_variant = contrast["variant"].nunique() > 1
    n_de = int(contrast["n_de"].max())
    n_en = int(contrast["n_en"].max())
    if caption is None:
        caption = (
            f"German ($n={n_de}$) against English ($n={n_en}$) questions on the "
            f"synthetic reference-context cell, the only cell generated in both "
            f"languages. The two arms are separate generation passes over the same "
            f"five contexts, so the questions are not matched and every test is "
            f"unpaired. $\\Delta$ is the difference in means (German $-$ English); "
            f"$\\Delta_{{\\mathrm{{strat}}}}$ is the same difference computed within "
            f"context and pooled, which holds the life-stage band and the styling "
            f"profile fixed; $p_{{\\mathrm{{perm}}}}$ permutes the language label "
            f"within context. Read $\\Delta$ against "
            f"$\\Delta_{{\\mathrm{{strat}}}}$: agreement is what licenses reading the "
            f"gap as a language effect rather than a framing one.")
    if simple:
        head = ("Metric & " + ("Variant & " if by_variant else "")
                + r"$n$ (de/en) & German & English & $\Delta$ & "
                  r"$\Delta_{\mathrm{strat}}$ & $p_{\mathrm{perm}}$ \\")
        spec = ("ll" if by_variant else "l") + "crrrrr"
    else:
        head = ("Metric & " + ("Variant & " if by_variant else "")
                + r"$n_{\mathrm{de}}$ & $\bar{x}_{\mathrm{de}}$ & "
                  r"$n_{\mathrm{en}}$ & $\bar{x}_{\mathrm{en}}$ & "
                  r"$\Delta$ [95\% CI] & $\Delta_{\mathrm{strat}}$ & "
                  r"$r_{\mathrm{rb}}$ & $p$ & $p_{\mathrm{Holm}}$ & "
                  r"$p_{\mathrm{perm}}$ \\")
        spec = ("ll" if by_variant else "l") + "rrrrlrrrrr"
    lines = ["% requires \\usepackage{booktabs}",
             "\\begin{table}[htbp]", "  \\centering",
             f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
             "  \\footnotesize",
             f"  \\begin{{tabular}}{{{spec}}}", "    \\toprule",
             f"    {head}", "    \\midrule"]
    prev = None
    for _, r in contrast.iterrows():
        if by_variant and prev is not None and r["metric"] != prev:
            lines.append("    \\addlinespace")
        name = plots.metric_label(r["metric"]) if r["metric"] != prev else ""
        prev = r["metric"]
        var = f"{plots.variant_label(r['variant'])} & " if by_variant else ""
        if simple:
            lines.append(
                f"    {name} & {var}{int(r['n_de'])}/{int(r['n_en'])} & "
                f"{_tex_num(r['mean_de'])} & {_tex_num(r['mean_en'])} & "
                f"{_tex_num(r['delta'], sign=True)} & "
                f"{_tex_num(r['delta_strat'], sign=True)} & "
                f"{_tex_p(r['p_perm'])} \\\\")
            continue
        delta = (f"{_tex_num(r['delta'], sign=True)} [{_tex_num(r['ci_low'])}, "
                 f"{_tex_num(r['ci_high'])}]")
        lines.append(
            f"    {name} & {var}{int(r['n_de'])} & {_tex_num(r['mean_de'])} & "
            f"{int(r['n_en'])} & {_tex_num(r['mean_en'])} & {delta} & "
            f"{_tex_num(r['delta_strat'], sign=True)} & "
            f"{_tex_num(r['rank_biserial'])} & {_tex_p(r['p'])} & "
            f"{_tex_p(r['p_holm'])} & {_tex_p(r['p_perm'])} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# The DeepEval metrics that persist a per-cell ``{n_verdicts, yes, no, idk}`` tally.
# ``deepeval_contextual_relevance`` is listed even though it currently tallies zeros
# on every row: its metric object exposes ``verdicts_list`` (grouped per context)
# rather than ``verdicts``, and the persistence helper in
# ``evaluation/deepeval_eval.py`` reads only the latter. Keeping it in the list means
# the table reports "this metric kept no verdicts" instead of silently omitting it.
VERDICT_METRICS = ("deepeval_faithfulness", "deepeval_relevance",
                   "deepeval_contextual_relevance")

VERDICT_COHORTS = ("answered", "abstained")


def _verdict_frame(df, metric):
    """The four tally columns of one metric as a numeric frame, or ``None``.

    Rows whose ``n_verdicts`` is 0 or missing are dropped: they mean the metric never
    ran on that row (the evaluators skip answer relevancy on a refusal), which is a
    different statement from "it ran and returned nothing", and counting them in would
    deflate every fraction below by an arbitrary amount.
    """
    cols = {k: f"{_score_col(metric)}_verdicts.{k}"
            for k in ("n_verdicts", "yes", "no", "idk")}
    if not all(c in df for c in cols.values()):
        return None
    out = pd.DataFrame({k: pd.to_numeric(df[c], errors="coerce")
                        for k, c in cols.items()}, index=df.index)
    return out[out["n_verdicts"].fillna(0) > 0]


def verdict_shape(df, metrics=VERDICT_METRICS, by=None, cohorts=VERDICT_COHORTS):
    """What the DeepEval judges actually returned, at the CLAIM level, per cohort.

    ``metrics_on_abstentions`` reports ``frac_idk`` on the abstained rows only, which
    is enough to justify excluding them but not enough to support any claim about the
    run: the abstentions are a few hundred near-identical refusals, so a shrug rate
    measured on them describes one sentence rather than the data. This is the same
    tally on the ANSWERED rows — the cohort every mean, correlation and paired test in
    this report is computed over — with the abstained cohort kept alongside as the
    contrast.

    Why the claim level rather than the score level: DeepEval's verdict prompt reserves
    "no" for a claim the context DIRECTLY contradicts, so a claim the context merely
    fails to mention comes back "idk". With ``penalize_ambiguous_claims=True`` (enabled
    2026-08-10) an "idk" contributes 0 exactly like a "no", so the score is
    ``yes / n_verdicts`` and ``frac_idk`` is precisely the share of the score that turns
    on that convention rather than on any contradiction having been found. RAGAS's NLI
    prompt has no third option — a statement is supported or it is not — so this column
    is the mechanical difference between two metrics that carry the same name. See
    ``idk_counterfactual`` for what the convention does to their agreement.

    Columns: ``n_rows`` (rows on which the metric returned at least one verdict);
    ``n_verdicts`` and ``verdicts_per_row`` (the claim decomposition — a refusal yields
    one claim, a real answer several); ``yes`` / ``no`` / ``idk`` counts and their
    shares of ``n_verdicts``; ``rows_any_idk`` / ``rows_all_idk``, the ROW-level view of
    the same thing (a metric can have a modest ``frac_idk`` and still shrug at least
    once on most rows, and it is rows that the reported mean averages over);
    ``pooled_score`` = ``frac_yes``, the claim-weighted score under the current
    convention; and ``pooled_if_idk_yes`` = ``frac_yes + frac_idk``, the same quantity
    under the convention in force before 2026-08-10. The gap between those two is the
    entire effect of the setting.

    Runs on either frame: ``prepare`` nulls score columns, not the ``*_verdicts.*``
    tallies, so the abstained cohort is still countable afterwards. ``by`` adds a
    breakdown (e.g. ``"source_dataset"`` or ``"variant"``) and is applied within each
    cohort, never across them.
    """
    rej = ra._abstained(df)
    by = [by] if isinstance(by, str) else list(by or [])
    rows = []
    for metric in metrics:
        tal = _verdict_frame(df, metric)
        if tal is None:
            continue
        for cohort in cohorts:
            mask = rej if cohort == "abstained" else ~rej
            sub = tal[mask.reindex(tal.index, fill_value=False)]
            groups = ([(name, g) for name, g in sub.groupby(
                          [df.loc[sub.index, c] for c in by], observed=True)]
                      if by and len(sub) else [((), sub)])
            for name, g in groups:
                name = name if isinstance(name, tuple) else (name,)
                n_rows, n_v = len(g), float(g["n_verdicts"].sum())
                if not n_rows:
                    continue
                y, n, i = (float(g[k].sum()) for k in ("yes", "no", "idk"))
                rows.append(dict(zip(by, name), **{
                    "metric": metric, "cohort": cohort,
                    "n_rows": n_rows, "n_verdicts": int(n_v),
                    "verdicts_per_row": n_v / n_rows,
                    "yes": int(y), "no": int(n), "idk": int(i),
                    "frac_yes": y / n_v if n_v else _NAN,
                    "frac_no": n / n_v if n_v else _NAN,
                    "frac_idk": i / n_v if n_v else _NAN,
                    "rows_any_idk": float((g["idk"] > 0).mean()),
                    "rows_all_idk": float((g["idk"] >= g["n_verdicts"]).mean()),
                    "pooled_score": y / n_v if n_v else _NAN,
                    "pooled_if_idk_yes": (y + i) / n_v if n_v else _NAN,
                }))
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    out["metric"] = pd.Categorical(out["metric"], list(metrics), ordered=True)
    out["cohort"] = pd.Categorical(out["cohort"], list(cohorts), ordered=True)
    return out.set_index(["metric", "cohort"] + by).sort_index()


def empty_verdict_metrics(df, metrics=VERDICT_METRICS):
    """Metrics whose tally columns exist but are 0 on every row — the evaluator
    persisted no verdicts at all. Reported rather than dropped, because "no verdicts
    were kept" is a fact about the harness (``verdicts_list``, see ``VERDICT_METRICS``)
    that a reader of the table would otherwise have to infer from an absence.
    """
    out = []
    for metric in metrics:
        cols = [f"{_score_col(metric)}_verdicts.{k}"
                for k in ("n_verdicts", "yes", "no", "idk")]
        if not all(c in df for c in cols):
            continue
        if not float(pd.to_numeric(df[cols[0]], errors="coerce").fillna(0).sum()):
            out.append(metric)
    return out


def idk_counterfactual(df, metric="deepeval_faithfulness",
                       against="ragas_scores.ragas_faithfulness",
                       exclude_abstentions=True):
    """What crediting "idk" as supported would do to one metric's level AND to its
    agreement with a comparable metric. Returns ``(table, info)``.

    The point of the split. ``penalize_ambiguous_claims`` is usually argued as a level
    effect — it lowers the score — but the level is not what makes two faithfulness
    metrics comparable; the RANKING is. This recomputes the score per row under the
    other convention, straight from the persisted tallies
    (``(yes + idk) / n_verdicts`` against ``yes / n_verdicts``), and correlates both
    with ``against``. If the two conventions rank rows the same way, the setting is a
    calibration choice and nothing more; if they do not, then the same answer decomposed
    into the same claims is being scored on two different constructs, and a correlation
    between the two libraries is partly reporting the convention.

    ``info["recomputed_matches"]`` is the guard: the fraction of rows where
    ``yes / n_verdicts`` reproduces the persisted score to 1e-6. It should be ~1.0. If
    it is not, the tallies and the score did not come from the same metric object and
    every number here is void — read that before reading the table.
    """
    score_col = _score_col(metric)
    tal = _verdict_frame(df, metric)
    info = {"n_rows": 0, "recomputed_matches": _NAN}
    if tal is None or not len(tal) or score_col not in df:
        return pd.DataFrame(), info
    idx = tal.index
    if exclude_abstentions:
        idx = idx[~ra._abstained(df).reindex(idx, fill_value=False)]
        tal = tal.loc[idx]
    if not len(tal):
        return pd.DataFrame(), info
    strict = tal["yes"] / tal["n_verdicts"]
    lenient = (tal["yes"] + tal["idk"]) / tal["n_verdicts"]
    stored = pd.to_numeric(df.loc[idx, score_col], errors="coerce")
    both = stored.notna()
    info["n_rows"] = int(len(tal))
    info["recomputed_matches"] = (
        float((strict[both] - stored[both]).abs().le(1e-6).mean()) if both.any() else _NAN)
    peer = against.split(".")[-1]
    rows = []
    for label, s in (("as scored (idk penalised)", strict),
                     ("counterfactual (idk credited)", lenient)):
        row = {"convention": label, "n_rows": int(len(s)),
               "mean": float(s.mean()), "frac_0": float((s == 0).mean()),
               "frac_1": float((s == 1).mean())}
        if against in df:
            n, pear, spear = _corr(s, pd.to_numeric(df.loc[idx, against], errors="coerce"))
            row.update({"n_paired": n, f"pearson_vs_{peer}": pear,
                        f"spearman_vs_{peer}": spear})
        rows.append(row)
    return pd.DataFrame(rows), info


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
         "each cell gets ONE status, so the columns partition the row count: "
         "error + na_total + scored == total:")
    emit("    'error' means APPLICABLE but missing. Applicability is decided first, so a "
         "cell that was never expected to carry a value keeps its na_* reason even if the "
         "evaluator also recorded an exception on it — that exception is not a failure of "
         "this run and is counted separately, under 'exceptions on cells that were not "
         "applicable' below.")
    emit(_indent(piv.to_string()))

    # Immediately after the table because it is the evidence for one column of it:
    # why the answer-grading metrics carry na_rejected rather than a score.
    fa, fa_info = metrics_on_abstentions(df)
    flags["metrics_on_abstentions"] = fa
    flags["abstention_texts"] = fa_info["n_texts"]
    if len(fa) and fa_info["n_abstained_rows"]:
        emit(f"\n  why the answer-grading metrics are na_rejected: what the scorers had "
             f"put on the {fa_info['n_abstained_rows']} abstained rows")
        if fa_info["n_texts"] is not None:
            emit(f"    those rows use {fa_info['n_texts']} distinct answer string(s), i.e. "
                 f"they are textually near-identical, so any spread below is the scorer's "
                 f"convention for an answer with no claims — not a property of this run:")
        emit(_indent(fa.round(3).to_string(), 4))
        emit(f"    n_abstained = abstained cells that carried a value; n_excluded = cells "
             f"the exclusion actually nulls (smaller only where "
             f"{'/'.join(sorted(ABSTENTION_SCORED_DATASETS))} keeps them); mean_excluded / "
             f"frac_0 / frac_1 describe the NULLED cells and mean_kept the surviving ones, "
             f"which is not the same average wherever those two differ; n_idk / frac_idk = "
             f"DeepEval verdicts on those rows that were neither supported nor "
             f"contradicted (the 'idk' rail its score now penalises) — measured HERE on "
             f"the abstained rows only, which are near-identical refusals; the same "
             f"tally on the answered rows is in the 'verdict shape' table further "
             f"down, and that is the one to quote for anything about the run.")

        # One paragraph per family, because the three exclusions rest on different
        # evidence and a single sentence covering all three would be true of none.
        rel = fa[fa["family"] == "relevancy"]
        if len(rel):
            scored = int(rel["n_abstained"].sum())
            if scored == 0:
                emit(f"    RELEVANCY ({', '.join(rel.index)}): n_abstained is 0 — these "
                     f"were never scored on an abstention in the first place. Both "
                     f"evaluators skip them on a rejected row, so na_rejected here records "
                     f"a decision already taken at scoring time rather than one taken by "
                     f"this analysis; there is no value to argue about. The reason it was "
                     f"taken upstream: relevancy asks whether the answer addresses the "
                     f"question, and a refusal declines to address it, which is a different "
                     f"thing from addressing it badly.")
            else:
                emit(f"    RELEVANCY ({', '.join(rel.index)}): {scored} abstained cells "
                     f"carried a value despite the evaluators' skip — unexpected, since "
                     f"they should be unscored on a rejected row. Worth checking the "
                     f"is_rejected gate in the evaluator before trusting these.")

        if fa_info["spread"] == fa_info["spread"]:  # not NaN
            # Which scorer sits on which rail comes from the table, not from prose:
            # the groupings change with scorer config (see metrics_on_abstentions).
            split = " / ".join(
                part for part in (
                    (f"{', '.join(fa_info['reads_zero'])} read it as 0 = hallucinated"
                     if fa_info["reads_zero"] else ""),
                    (f"{', '.join(fa_info['reads_one'])} as 1 = nothing false asserted"
                     if fa_info["reads_one"] else ""))
                if part)
            emit(f"    FAITHFULNESS: the conventions disagree by {fa_info['spread']:.3f} on "
                 f"that identical text (a refusal is zero claims: {split}). Keeping the "
                 f"cells would report the choice of scorer as faithfulness, and would let a "
                 f"system raise its score by abstaining more; they are excluded from every "
                 f"mean, correlation, paired test and figure below. The abstention RATE is "
                 f"reported separately, at the top of this report.")

        ref = fa[fa["family"] == "reference"]
        if len(ref) and ref["n_abstained"].sum():
            kept_n = int((ref["n_abstained"] - ref["n_excluded"]).max())
            fmt = lambda s: ", ".join(f"{v:.3f}" for v in s)  # noqa: E731
            emit(f"    REFERENCE ({', '.join(ref.index)}): the opposite case — these WERE "
                 f"scored on abstentions, and what they scored is the argument for dropping "
                 f"them. On the nulled cells they average {fmt(ref['mean_excluded'])}, "
                 f"against {fmt(ref['mean_answered'])} on answered rows: grading a refusal "
                 f"against a gold answer returns ~0 by construction, and that 0 is "
                 f"indistinguishable from a wrong answer. Left in, the metric's mean "
                 f"silently becomes a mixture of answer quality and abstention rate — the "
                 f"second of which already has its own number, and a system could raise the "
                 f"metric by answering less. This is the one exclusion that is not run-wide: "
                 f"{kept_n} cells are KEPT on "
                 f"{'/'.join(sorted(ABSTENTION_SCORED_DATASETS))} (mean_kept "
                 f"{fmt(ref['mean_kept'])}), where declining is the behaviour under test and "
                 f"the score on a refusal is the measurement wanted. Note how far mean_kept "
                 f"sits from mean_excluded: the same metric on the same kind of text, "
                 f"scored against a gold answer that says to refuse instead of one that "
                 f"does not. That gap is why these two metrics have a different cohort rule "
                 f"per dataset and must be read per dataset, never pooled.")

    # The actionable bit: which cells are true errors, and — since both evaluators
    # record the exception next to the metric it killed — WHY. Note this counts
    # CELLS, not rows: a cell error costs one metric on one row, which is why 75 of
    # them can coexist with 2 dropped rows. The two sets need not overlap at all.
    err = cls[cls["status"] == "error"]
    flags["error_cells"] = int(len(err))
    # Every exception either evaluator recorded, tagged with the status the cell
    # ended up with. The join is what separates the two populations below: an
    # exception on an ``error`` cell cost a value the analysis wanted, one on an
    # ``na_*`` cell killed a value nothing would have used.
    reasons = metric_error_reasons(df)
    flags["error_reasons"] = reasons
    join_keys = [c for c in ("id", "variant", "metric") if c in cls and c in reasons]
    tagged = (reasons.merge(cls[join_keys + ["status"]], on=join_keys, how="left")
              if len(reasons) and join_keys else reasons.assign(status=None))

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
        if len(reasons):
            explained = tagged["status"].eq("error")
            emit(f"    recorded cause ({int(explained.sum())} of {len(err)} explained "
                 f"by the evaluator's own error field):")
            emit(_indent(pd.crosstab(
                tagged.loc[explained, "metric"].map(lambda m: m.split(".")[-1]),
                tagged.loc[explained, "error_type"]).to_string(), 6))
            missing = len(err) - int(explained.sum())
            if missing > 0:
                emit(f"      {missing} error cell(s) with no recorded cause — the value "
                     f"simply never landed")

            # WHICH cells failed, not just how many. An exception that fires on the
            # longest answers censors the cohort, so the surviving mean describes the
            # short ones; one that fires uniformly only costs n. The two medians are
            # what tells those apart, and the claim "the failures are the long answers"
            # has to be read off a table rather than asserted from the exception name.
            lp = error_length_profile(df)
            flags["error_length_profile"] = lp
            if len(lp):
                emit("")
                emit(f"    which cells failed — answer length (characters) on "
                     f"the scored vs the errored cells of the same metric x dataset:")
                emit(_indent(lp.round(3).to_string(), 6))
                emit(f"      n_attempted = value landed OR exception recorded, which is "
                     f"the scorer's own denominator and not the applicable-cell count "
                     f"above (a cell can be skipped as na_* without either). Read "
                     f"len_med_error against len_med_scored: a gap means the missing "
                     f"cells are a length-selected subset, so that metric's remaining "
                     f"mean is computed over the shorter answers and its coverage "
                     f"limit belongs in the text.")
        else:
            emit("    no recorded cause available: this file predates the per-metric "
                 "error fields (ragas_metric_errors / deepeval_*_error)")

    # The other half of the exception census, and the reason the 'error' column above
    # is smaller than the number of exceptions in the file. Reported as its own block
    # rather than as a column in the status table on purpose: it is a SUBSET of cells
    # already counted under na_*, so putting it in that table would break the one
    # property that makes the table checkable (its columns partition the row count),
    # and a reader would have to be told which columns may be added and which may not.
    shadowed = tagged[tagged["status"].astype("string").str.startswith("na_").fillna(False)] \
        if len(tagged) else tagged
    flags["shadowed_error_cells"] = int(len(shadowed))
    if len(shadowed):
        emit(f"\n  exceptions on cells that were not applicable: {len(shadowed)} "
             f"— recorded by the evaluator, NOT counted as errors above, and not part of "
             f"any total (these cells are already counted under their na_* reason). The "
             f"metric was never going to be used on those rows, so the crash costs "
             f"nothing; it is listed because it is evidence about the scorer's config, "
             f"and because {len(err)} + {len(shadowed)} is the number of exceptions in "
             f"the file:")
        emit(_indent(pd.crosstab(
            shadowed["metric"].map(lambda m: m.split(".")[-1]),
            [shadowed["status"], shadowed["error_type"]]).to_string(), 4))

    print("\n".join(lines))
    return flags


def prepare(df):
    """Return ``(clean_df, report)`` with the three cell-level exclusions applied.

    1. RAGAS metric cells nulled on RAGAS-errored rows, so those rows leave RAGAS
       aggregates while keeping their independent DeepEval scores. (The metric-level
       masking ``analysis.drop_eval_errors`` deliberately punts on.) They are already
       NaN after ``analysis.load``; this makes the exclusion explicit and defensive.
    2. Faithfulness cells nulled on abstentions, matching the ``na_rejected`` status
       ``classify_metrics`` gives the same cells. This is the one place the exclusion
       has to happen: everything downstream — distributions, means by variant, paired
       comparisons, agreement, deciles, every figure — reads this frame, so nulling
       here is what makes "excluded everywhere" true rather than a claim repeated per
       call site. ``metrics_on_abstentions`` (run on the RAW frame, before this)
       is the diagnosis; this is the treatment.
    3. The two ``REFERENCE_METRICS`` nulled on abstentions as well — but only off
       ``ABSTENTION_SCORED_DATASETS`` (``_abstention_excluded``). Same reasoning as
       (2) with one difference that matters: this exclusion is NOT uniform, so the
       affected metrics end up with a different effective cohort per dataset. MedQA
       keeps its abstained rows and every other dataset does not, which means an
       accuracy/correctness mean pooled across datasets is a mean over cohorts
       defined differently — read those two metrics per dataset, and read the
       abstention rate alongside them.

    Reports what it touched in all three cases.
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

    rej_ref = _abstention_excluded(clean)
    ref_cols = sorted(c for c in REFERENCE_METRICS if c in clean)
    n_ref_cells = 0
    if ref_cols and rej_ref.any():
        n_ref_cells = int(clean.loc[rej_ref, ref_cols].notna().to_numpy().sum())
        clean.loc[rej_ref, ref_cols] = np.nan

    report = {
        "n_ragas_error_rows": int(rerr.sum()),
        "by_dataset_variant": (pd.crosstab(clean.loc[rerr, "source_dataset"],
                                           clean.loc[rerr, "variant"].astype(str))
                               if rerr.any() else pd.DataFrame()),
        "n_abstained_rows": int(rej.sum()),
        "n_faithfulness_cells_masked": n_faith_cells,
        "faithfulness_cols_masked": faith_cols,
        "n_reference_cells_masked": n_ref_cells,
        "reference_cols_masked": ref_cols,
        "n_abstained_rows_reference": int(rej_ref.sum()),
        # The abstentions deliberately KEPT — the rows that separate this exclusion
        # from the blanket one, and the number to quote when the two metrics'
        # per-dataset cohorts are questioned.
        "abstentions_kept": (pd.Series(rej & ~rej_ref).groupby(
            clean["source_dataset"].astype(str), observed=True).sum()
            .pipe(lambda s: s[s > 0]) if "source_dataset" in clean else pd.Series(dtype=int)),
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
    # The grouping itself is ``ev.metric_summary_by`` — the same call
    # ``plots.metric_rail_grid`` makes, so this table and that figure cannot drift.
    return ev.metric_summary_by(df, by, metrics)


# The range each score can take BY DEFINITION, read off the scorer's source rather
# than off the data: ``(lo, hi, basis, definition)``. The point is that "every score
# I observed is in [0, 1]" is not the same statement as "this score is a [0, 1]
# score", and only the second one licenses reporting a metric as a percentage or
# feeding it to a test that assumes a bounded scale.
#
# Three of these are similarities, and ragas leaves all three UNCLIPPED:
# ``SemanticSimilarity`` returns the raw dot product of two L2-normalised
# embeddings (``_answer_similarity.py``; ``threshold`` is None here, so no
# binarisation), and ``ResponseRelevancy`` returns the mean cosine between the
# question and the questions it regenerates from the answer
# (``_answer_relevance.py``). A cosine is in [-1, 1], so those metrics have a
# negative floor whatever the observed minimum happens to be.
#
# ``ragas_answer_accuracy`` is NOT one of them despite the name suggesting a
# similarity: ragas's ``AnswerAccuracy`` (``_nv_metrics.py``) asks two independently
# prompted judges for a rating in {0, 2, 4}, maps each to {0, 0.5, 1.0} and averages
# them, so its support is the five-point grid {0, 0.25, 0.5, 0.75, 1.0} and it cannot
# leave [0, 1] by construction.
SCORE_BOUNDS = {
    "ragas_scores.ragas_faithfulness": (
        0.0, 1.0, "LLM ratio",
        "supported statements / statements, one NLI verdict per statement"),
    "ragas_scores.ragas_answer_relevancy": (
        -1.0, 1.0, "cosine",
        "mean cosine(question, 3 questions regenerated from the answer), "
        "multiplied by 0 if every regeneration is judged noncommittal"),
    "ragas_scores.ragas_faithfulness_with_hhem": (
        0.0, 1.0, "NLI probability",
        "mean HHEM-2.1 entailment probability over the answer's statements"),
    "ragas_scores.ragas_answer_accuracy": (
        0.0, 1.0, "judge grid",
        "mean of two independent LLM ratings in {0, 2, 4}, rescaled by /4 - "
        "a 5-level grid, not a similarity"),
    "ragas_scores.ragas_answer_correctness": (
        -0.25, 1.0, "cosine (25%)",
        "0.75 x statement-level F1 + 0.25 x cosine(answer, reference); the "
        "cosine term is raw, so the composite floor is 0.75x0 + 0.25x(-1)"),
    "ragas_scores.ragas_id_context_recall": (
        0.0, 1.0, "set overlap", "gold chunk ids retrieved / gold chunk ids"),
    "ragas_scores.ragas_id_context_precision": (
        0.0, 1.0, "set overlap", "gold chunk ids retrieved / chunks retrieved"),
    "ragas_scores.ragas_id_context_ap": (
        0.0, 1.0, "set overlap", "average precision over the ranked chunk ids"),
    "deepeval_scores.deepeval_faithfulness": (
        0.0, 1.0, "LLM ratio",
        "yes verdicts / claims (penalize_ambiguous_claims=True, so an idk scores 0)"),
    "deepeval_scores.deepeval_relevance": (
        0.0, 1.0, "LLM ratio", "relevant statements / statements in the answer"),
    "deepeval_scores.deepeval_contextual_relevance": (
        0.0, 1.0, "LLM ratio", "relevant statements / statements in the contexts"),
}

# What an unlisted metric is assumed to be. Stated rather than silently defaulted,
# because a new metric that is in fact a similarity would otherwise be audited
# against the wrong floor and pass.
DEFAULT_SCORE_BOUND = (0.0, 1.0, "assumed", "not declared in SCORE_BOUNDS")


def score_range_audit(df, metrics=None, bounds=None):
    """Did any score leave the range its own definition allows, and how close did it
    come? One row per metric, on the RAW numeric cells.

    Motivation. Several of these metrics are cosine similarities that ragas does not
    clip (see ``SCORE_BOUNDS``), so their true floor is negative even though every
    table in this report shows them between 0 and 1. That matters in two directions.
    A negative value that DID occur is not a low score, it is a value outside the
    scale the results chapter describes, and it silently drags a mean below what the
    metric's own documentation says is possible. A negative value that did NOT occur
    is a fact worth stating rather than assuming: the audit turns "presumably fine"
    into a count.

    Columns: ``basis`` (what the number is - a cosine, a ratio of LLM verdicts, a
    grid of judge ratings), ``can_be_negative`` (the whole question, decided by the
    definition and not by the sample); the declared ``lo`` / ``hi`` against the
    observed ``min`` / ``max``; ``n_below`` / ``n_above`` / ``n_negative`` (violations
    - all three should be 0); ``margin_to_lo``, the distance from the observed
    minimum to the declared floor, which is the only column that says how NEAR a
    negative value the run came rather than merely that none appeared; and
    ``n_unique``, which separates a continuous score from a discrete grid that
    ``score_levels`` should be read for instead.

    Run on the raw frame. ``prepare`` nulls the abstained cells, and an out-of-range
    value on a row that is later excluded is still a scorer fact worth knowing.
    """
    bounds = SCORE_BOUNDS if bounds is None else bounds
    metrics = metrics or score_cols(df)
    rows = []
    for metric in metrics:
        if metric not in df:
            continue
        val = pd.to_numeric(df[metric], errors="coerce").dropna()
        if val.empty:
            continue
        lo, hi, basis, _ = bounds.get(metric, DEFAULT_SCORE_BOUND)
        rows.append({
            "metric": metric.split(".")[-1], "basis": basis,
            "can_be_negative": lo < 0, "n_scored": len(val),
            "lo": lo, "hi": hi,
            "min": float(val.min()), "max": float(val.max()),
            "n_below": int((val < lo - 1e-9).sum()),
            "n_above": int((val > hi + 1e-9).sum()),
            "n_negative": int((val < 0).sum()),
            "margin_to_lo": float(val.min() - lo),
            "n_unique": int(val.nunique()),
        })
    out = pd.DataFrame(rows)
    return out.set_index("metric") if len(out) else out


def score_definitions(metrics=None, bounds=None):
    """``{short metric name: definition}`` for the metrics in ``bounds`` - the legend
    that makes ``score_range_audit`` readable without opening the scorer's source.
    Kept out of the table itself so the numeric columns stay printable.
    """
    bounds = SCORE_BOUNDS if bounds is None else bounds
    keys = list(bounds) if metrics is None else [m for m in metrics if m in bounds]
    return {k.split(".")[-1]: bounds[k][3] for k in keys}


def score_levels(df, metrics=None, by=None, max_levels=8):
    """The full value-frequency table for the metrics that emit only a handful of
    distinct values - the honest form of a "distribution" for a discrete score.

    ``metric_distribution`` reports mean, quartiles and the two rail fractions, which
    is the right summary for a continuous metric and a misleading one for a grid: the
    mean of ``ragas_answer_accuracy`` sits near the middle of its five levels while
    almost nothing is scored there, and a quartile of a five-point grid is just one
    of the five points. This lists every level with its count instead, so a bimodal
    metric reads as bimodal.

    Metrics are auto-selected as those with at most ``max_levels`` distinct values,
    so the table covers whatever grids a run happens to contain. ``by`` (e.g.
    ``"source_dataset"``, ``"variant"``, or a list) breaks each metric down within
    the group, with ``frac`` normalised inside the group so rows are comparable
    across groups of different sizes.
    """
    cand = metrics or score_cols(df)
    by = [by] if isinstance(by, str) else list(by or [])
    rows = []
    for metric in cand:
        if metric not in df:
            continue
        val = pd.to_numeric(df[metric], errors="coerce")
        if val.dropna().empty or (metrics is None and val.nunique() > max_levels):
            continue
        groups = ([(name, g) for name, g in val.groupby(
                      [df[c] for c in by], observed=True)]
                  if by else [((), val)])
        for name, g in groups:
            name = name if isinstance(name, tuple) else (name,)
            g = g.dropna()
            if g.empty:
                continue
            vc = g.value_counts().sort_index()
            for level, n in vc.items():
                rows.append(dict(zip(by, name), **{
                    "metric": metric.split(".")[-1], "value": float(level),
                    "n": int(n), "frac": float(n) / len(g),
                }))
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    return out.set_index(["metric"] + by + ["value"]).sort_index()


OUT_OF_RANGE_BIN = "outside range"

DEFAULT_SCORE_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def score_histogram(df, metrics=None, bins=DEFAULT_SCORE_BINS, by=None, min_levels=9):
    """The binned distribution of the CONTINUOUS scores, with the two rails split out
    as rows of their own.

    The complement of ``score_levels``: that one lists every value of a metric whose
    support is a handful of points, this one bins the metrics that emit too many
    distinct values to list. Between them every score column in the run is covered.

    Why the rails get their own rows rather than falling into the first and last bin.
    An exact 0.0 or 1.0 is usually produced by a different mechanism than the values
    around it, so pooling it into ``(0.0, 0.1]`` averages two things together.
    ``ragas_answer_relevancy`` is the clearest case: its score is
    ``mean cosine(...) * int(not all_noncommittal)``, so an exact 0.0 is the
    noncommittal GATE firing — the judge decided the answer evades the question and
    multiplied the similarity away — and not a cosine that happened to be zero. Those
    rows are a refusal-detection event, and reading them as "relevance near zero"
    both misdescribes them and hides how far the real distribution's floor actually
    sits above zero.

    Columns: ``n``, ``frac`` (within the metric, or within the ``by`` group), and
    ``cum_frac``, which is what a sentence like "90% of rows score below 0.8" is read
    off. Bin edges are right-closed, so ``(0.5, 0.6]`` includes 0.6.

    Metrics are auto-selected as those with at least ``min_levels`` distinct values.
    Pass ``metrics`` explicitly to bin one regardless.
    """
    cand = metrics or score_cols(df)
    by = [by] if isinstance(by, str) else list(by or [])
    edges = list(bins)
    labels = ([f"= {edges[0]:g}"]
              + [f"({edges[k]:g}, {edges[k + 1]:g}]" for k in range(len(edges) - 1)]
              + [f"= {edges[-1]:g}"])
    rows = []
    for metric in cand:
        if metric not in df:
            continue
        val = pd.to_numeric(df[metric], errors="coerce")
        if val.dropna().empty or (metrics is None and val.nunique() < min_levels):
            continue
        groups = ([(name, g) for name, g in val.groupby(
                      [df[c] for c in by], observed=True)]
                  if by else [((), val)])
        for name, g in groups:
            name = name if isinstance(name, tuple) else (name,)
            g = g.dropna()
            if g.empty:
                continue
            # The interior is everything strictly between the rails; the rails are
            # counted by equality so a bin never claims a row twice.
            lo_hits, hi_hits = (g == edges[0]), (g == edges[-1])
            interior = g[~(lo_hits | hi_hits)]
            counts = {lab: 0 for lab in labels}
            counts[labels[0]] = int(lo_hits.sum())
            counts[labels[-1]] = int(hi_hits.sum())
            n_out = 0
            if len(interior):
                # No ``include_lowest``: the interior is strictly inside the rails by
                # construction, so a value in (0, 0.1] already lands in the first bin
                # and widening it would only invent a left edge below ``lo``.
                cut = pd.cut(interior, bins=edges, right=True, labels=labels[1:-1])
                for lab, n in cut.value_counts().items():
                    counts[lab] += int(n)
                # Anything ``cut`` could not place is outside [lo, hi] — impossible for
                # a well-behaved metric, which is exactly why it gets a visible row
                # instead of silently dropping out of the fractions.
                n_out = int(cut.isna().sum())
            cum = 0
            for lab in labels + ([OUT_OF_RANGE_BIN] if n_out else []):
                n = n_out if lab == OUT_OF_RANGE_BIN else counts[lab]
                cum += n
                rows.append(dict(zip(by, name), **{
                    "metric": metric.split(".")[-1], "bin": lab, "n": n,
                    "frac": n / len(g), "cum_frac": cum / len(g),
                }))
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    out["bin"] = pd.Categorical(out["bin"], labels + [OUT_OF_RANGE_BIN],
                                ordered=True)
    return out.set_index(["metric"] + by + ["bin"]).sort_index()


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


def _pinned(s):
    """Share of ``s`` sitting on its single most common value — 1.0 when the metric
    returned one number for the whole group, which is when a correlation with it stops
    meaning anything.
    """
    return float(s.value_counts(normalize=True).iloc[0]) if len(s) else _NAN


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

    ``pin_a`` / ``pin_b`` (share of the paired rows sitting on that metric's single
    most common value) and ``nuniq_a`` / ``nuniq_b`` are the guard on the two
    correlation columns; a/b are the first and second metric of the pair, the same
    order ``mean_diff`` subtracts in. A correlation needs variance in BOTH variables,
    so once one of them is on one value for most of the group, r is carried by the
    remaining handful of rows and is close to uninterpretable however tight its
    decimals look. The signature to watch for is a near-zero r beside a HIGH
    ``frac_within`` and a ``pin_*`` near 1: that is not two metrics disagreeing, it is
    one metric having nothing to say. Read it before attributing a low r to the
    metrics measuring different things.

    ``by="source_dataset"`` / ``by="variant"`` gives the per-group table. ``by`` may
    also be a LIST of columns for the crossed table (``["source_dataset",
    "variant"]``), in which case each key additionally gets its own column beside
    the joined ``group`` label — a figure faceting on both needs them separately,
    and re-splitting a joined string is how a dataset called "a / b" would break it.

    Beware what the crossed table costs: every split divides the same paired rows,
    and a correlation over a handful of them is noise with a decimal point. The
    ``n`` column is not decoration here.
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
                "pin_a": _pinned(x[keep]), "pin_b": _pinned(y[keep]),
                "nuniq_a": int(x[keep].nunique()), "nuniq_b": int(y[keep].nunique()),
            })
        return pd.DataFrame(rows)

    if by is None:
        return one(df, rej, "overall")
    keys = [by] if isinstance(by, str) else list(by)
    frames = []
    for g, sub in df.groupby(keys[0] if len(keys) == 1 else keys, observed=True):
        vals = g if isinstance(g, tuple) else (g,)
        frame = one(sub, rej.loc[sub.index], " / ".join(str(v) for v in vals))
        # Only when crossed: on a single key ``group`` already IS that key's value,
        # and a duplicate column would change every existing table and CSV.
        if len(keys) > 1:
            for k, v in zip(keys, vals):
                frame[k] = str(v)
        frames.append(frame)
    cols = ["pair", "group", "family", "n", "pearson", "spearman",
            "mean_diff", "frac_within", "pin_a", "pin_b", "nuniq_a",
            "nuniq_b"] + (keys if len(keys) > 1 else [])
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


# --- (2e) Judge-prose failure taxonomy ---------------------------------------
# ``mine_reasons`` above asks a blunt question: does this prose contain a phrase
# that usually means the scorer broke? What follows asks a sharper one: given
# that the judge DID score the row, is its stated justification defensible?
#
# The blunt version is no longer in the report, and the reason is instructive.
# Its top hit on faithfulness was ``can ?not`` at 90 of 108 matches -- "cannot be
# verified", which is the correct wording of a correct ``idk`` verdict. A phrase
# list cannot separate a judge describing a failure from a judge failing.
# ``mine_reasons`` / ``reason_hits`` stay available for ad-hoc use.
#
# The taxonomy is derived from the prompt chains, not invented. Each DeepEval
# metric runs its own chain, and the chains differ in which row fields ever reach
# the model:
#
#   deepeval_faithfulness         truths(context) + claims(answer), then
#                                 verdicts(claims, truths), then reason(score,
#                                 contradictions). The QUESTION appears in none
#                                 of the four prompts, and the reason step sees
#                                 neither the question NOR the answer.
#   deepeval_contextual_relevance verdicts(input, context), reason(input,
#                                 statements). The ANSWER is never shown -- it
#                                 scores retrieval, not generation.
#   deepeval_relevance            statements(answer), verdicts(input, statements),
#                                 reason(input, statements). The CONTEXT is never
#                                 shown.
#
# Every class below follows from one of those gaps, so each is gated to the
# metric whose chain can actually produce it: "the judge ignored the question" is
# a finding on faithfulness and a guaranteed false positive on the other two.
DEEPEVAL_JUDGE_SEES = {
    "deepeval_faithfulness": {"question": False, "context": True, "answer": True},
    "deepeval_contextual_relevance": {"question": True, "context": True, "answer": False},
    "deepeval_relevance": {"question": True, "context": False, "answer": True},
}

# The FINAL step of each chain is blinder still, and it is the step whose output
# is the only thing persisted. ``generate_reason`` is a summariser over strings
# the verdict step already produced, so it re-reads none of the row:
#
#   faithfulness         reason(score, contradictions)                  -- sees
#                        neither the question, the answer, nor the context.
#   contextual_relevance reason(score, input, irrelevancies, statements)
#   relevance            reason(score, input, irrelevancies)            -- the
#                        answer it is describing is not in the prompt.
#
# Two classes below are gated on THIS table rather than the one above:
# ``speculative`` (prose guessing at an answer the step never received) and
# ``meta_quote`` (prose quoting a verdict's *reason* as if it were source text).
DEEPEVAL_REASON_SEES = {
    "deepeval_faithfulness": {"question": False, "context": False, "answer": False},
    "deepeval_contextual_relevance": {"question": True, "context": True, "answer": False},
    "deepeval_relevance": {"question": True, "context": False, "answer": False},
}

_REASON_PREFIX_RE = re.compile(r"^\s*the score is\s*[0-9.]+\s*because\s*", re.IGNORECASE)
# "likely provided", "an unspecified food" -- the judge guessing at text it was
# never shown. Gated to the metrics whose reason step is blind to the ANSWER it
# is describing (see ``DEEPEVAL_REASON_SEES``): faithfulness and relevance.
_SPECULATIVE_RE = re.compile(
    r"\b(?:likely|presumably|appears to (?:have|be)|seems to (?:have|be|discuss)"
    r"|may have (?:stated|claimed)|an? unspecified|the \w+ in question|it is implied)\b",
    re.IGNORECASE)
# The judge penalising the answer for a referent that exists only in the QUESTION.
#
# The phrase list alone is not enough and the first version of this detector was
# wrong because of it: keyed on "this food", it fired on NGQA and nowhere else,
# which looks like a dataset finding and is really a vocabulary artifact. LLMDRS
# has the identical failure in different words ("the context does not mention
# Shoakram", "does not confirm Svetlana's treatment plan"), because its patient
# profile lives in the query too. The entity test below is what actually
# generalises; the phrases only cover the case where the judge names no referent
# at all because it could not see one.
_UNRESOLVED_REFERENT_RE = re.compile(
    r"no specific question was provided|unspecified question|refers to .?this question"
    r"|about the context'?s? (?:ability|sufficiency)|meta-statement"
    r"|\ban? unspecified \w+|\bthe \w+ in question\b"
    r"|does not (?:specify|identify) (?:which|the specific|a specific)\b",
    re.IGNORECASE)
# Capitalised names, acronyms and quoted spans -- the referents a denial clause
# can be about. Filtered against the query and the context, never used alone.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
# Words that are capitalised because a sentence or a clause started, or because
# they are the judge's own vocabulary -- never a referent from the question.
_ENTITY_STOPWORDS = frozenset({
    "the", "this", "that", "these", "those", "context", "input", "output",
    "additionally", "however", "while", "furthermore", "although", "statement",
    "score", "reason", "retrieval", "claim", "user", "answer", "specifically",
    "for", "instance", "example", "and", "but", "because", "since", "json",
})
# Citations to a source index that does not exist. Two forms occur: the spelled
# "Context 5" / "Contradiction 1" and the compact "(C3)". The compact form is
# matched case-sensitively and without a space so that a quoted "Vitamin C 90
# mg/d" cannot be read as a citation to passage 90.
_CITATION_WORD_RE = re.compile(r"\b(?:context|contradiction)\s*#?\s*(\d{1,2})\b",
                               re.IGNORECASE)
_CITATION_SHORT_RE = re.compile(r"(?<![A-Za-z])C(\d{1,2})\b")
_CONTRADICTION_RE = re.compile(
    r"contradict|incorrectly (?:stated|states|applied)|is (?:incorrect|wrong)"
    r"|does not (?:provide|mention|contain|state)|not (?:present|in) the (?:retrieval )?context"
    r"|unfaithful|unsupported", re.IGNORECASE)
_ATTRIBUTION_RE = re.compile(
    r"context[^.]{0,25}?(?:states?|says?|specifies|provides|mentions|indicates)",
    re.IGNORECASE)
_DENIAL_RE = re.compile(
    r"not (?:found|present|mentioned|listed|provided)"
    r"|does not (?:provide|mention|contain|state|specify|list|identify|discuss|include)"
    r"|absent from", re.IGNORECASE)
# A quoted span that opens like a verdict's own reasoning rather than like source
# text. ``ContextualRelevancyTemplate.generate_reason`` hands the model two lists
# under headings that read alike -- "Reasons for why the retrieval context is
# irrelevant" (which are the *reason* fields of the "no" verdicts, meta-text) and
# "Statement in the retrieval context that is relevant" (actual context text) --
# and then instructs it to "quote data provided in the reasons for irrelevancy
# and relevant statements". Quoting a reason as though it were a retrieved
# passage is the predictable result.
#
# Two patterns, because verdict reasoning is recognisable either by how it opens
# ("The context states ...", "This is a general dietary guideline ...") or by
# vocabulary no retrieved passage would contain: a nutrition guideline never
# refers to "the input". Anchoring on the opening alone missed a fifth of them.
_META_QUOTE_START_RE = re.compile(
    r"^\s*(?:the (?:context|input|statement|retrieval context|actual output)"
    r"|this (?:statement|is an?)|these statements|the reasons? for irrelevanc"
    r"|does not |it does not )", re.IGNORECASE)
_META_QUOTE_BODY_RE = re.compile(
    r"\bthe input\b|\bis not relevant to\b|\bdoes not address\b"
    r"|\bnot relevant to (?:assessing|the)\b|reasons? for irrelevanc"
    r"|\bthis statement\b|\bthe retrieval context\b", re.IGNORECASE)
# A number with a unit: the only span in judge prose that can be compared across
# answer / context / reason without semantic matching.
_QUANTITY_RE = re.compile(
    r"\d[\d.,]*\s*(?:µg|μg|mcg|mg|kcal|ml|IU|g|%)\s*(?:-\s*RAE)?"
    r"\s*(?:/\s*(?:Tag|day|d|kg/d|kg/day)|\s*(?:pro Tag|per day|daily))?",
    re.IGNORECASE)
# Nouns too generic to prove a false denial: the context mentioning "sodium"
# somewhere does not refute "the context gives no sodium VALUE for this dish".
_GENERIC_DENIAL_TERMS = frozenset({
    "the food", "this food", "the patient", "the context", "food items",
    "calories", "sugar", "sodium", "protein", "cholesterol", "fat", "fiber",
    "salt", "carbohydrates", "saturated fat", "added sugars", "protein content",
    "nutritional information",
})


def _reason_body(series):
    """Judge prose with its "The score is X.XX because" preamble removed.

    The preamble carries the score, which ``deepeval_reason_consistency`` already
    checks; leaving it in would let its digits be read as a quantity or a
    citation by the detectors below.
    """
    return (series.astype("string").fillna("")
            .str.replace(_REASON_PREFIX_RE, "", regex=True))


def _norm_text(text):
    """Casefolded, punctuation-stripped, single-spaced -- for substring tests
    between prose and source text that should ignore quoting and hyphenation."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower())).strip()


def _norm_quantity(text):
    """A quantity reduced to the form in which "850 ug-RAE/Tag" and
    "850 ug-RAE/day" are the same value.

    Deliberately collapses the German and English unit words: a judge treating
    that pair as a contradiction is exactly the failure this is meant to catch,
    so the normaliser has to see through it.
    """
    t = str(text).lower().replace("µ", "u").replace("μ", "u")
    t = re.sub(r"\bmcg\b", "ug", t)
    t = re.sub(r"\b(?:pro\s+)?tag\b", "day", t)
    t = re.sub(r"\bt(?:ä|ae)glich\b", "day", t)
    t = re.sub(r"\b(?:daily|per\s+day)\b", "day", t)
    t = re.sub(r"\bd\b", "day", t)
    t = re.sub(r"(\d),(\d)", r"\1.\2", t)
    return re.sub(r"[^a-z0-9.]", "", t)


def _quantities(text):
    return {q for q in (_norm_quantity(m.group(0))
                        for m in _QUANTITY_RE.finditer(str(text)))
            if len(q) > 2}


def _cited_indices(text):
    idx = {int(m.group(1)) for m in _CITATION_WORD_RE.finditer(text)}
    idx |= {int(m.group(1)) for m in _CITATION_SHORT_RE.finditer(text)}
    return idx


# Apostrophe-safe quote extraction. A naive ``'([^']+)'`` treats the apostrophes
# in "the patient's needs ... Adelina's diet" as a matching pair and yields a
# fragment starting mid-word, which then reads as a quoted statement that is
# nowhere in the corpus. The lookarounds require the delimiters not to sit
# against a letter, which is true of a real quotation and false of a possessive.
def _quote_patterns(lo, hi):
    return (rf'"([^"]{{{lo},{hi}}})"',
            f"“([^”]{{{lo},{hi}}})”",
            rf"(?<![A-Za-z])'([^']{{{lo},{hi}}})'(?![A-Za-z])")


def _quoted_terms(text):
    """Concrete multi-word spans the prose sets apart -- quoted or parenthesised.

    Single words are dropped on purpose: a false denial has to be provable, and
    only a specific multi-word term ("choline-rich eggs", "flour tortillas")
    makes "the context does not mention X" checkable by substring.
    """
    out = []
    for pattern in _quote_patterns(6, 80):
        out.extend(m.group(1) for m in re.finditer(pattern, text))
    for m in re.finditer(r"\(([^)]{6,120})\)", text):
        out.extend(part.strip() for part in m.group(1).split(","))
    terms = []
    for raw in out:
        norm = _norm_text(raw.strip(" .,;:"))
        if len(norm) >= 6 and len(norm.split()) >= 2 and norm not in _GENERIC_DENIAL_TERMS:
            terms.append(norm)
    return terms


def _contexts_text(df):
    """One normalised blob of retrieved context per row (empty where none)."""
    if "contexts" not in df:
        return pd.Series("", index=df.index, dtype="object")
    return df["contexts"].map(
        lambda cs: _norm_text(" ".join(cs)) if isinstance(cs, (list, tuple)) else "")


def _n_contexts(df):
    if "contexts" not in df:
        return pd.Series(0, index=df.index, dtype="int64")
    return df["contexts"].map(lambda cs: len(cs) if isinstance(cs, (list, tuple)) else 0)


# Each detector takes (df, body, score) and returns a boolean Series on df.index.
# ``body`` is the prose without its score preamble; ``score`` the recorded metric.

def _detect_speculative(df, body, score):
    return body.str.contains(_SPECULATIVE_RE)


def _quoted_spans(text):
    """Every quoted span, whole and untrimmed -- for asking what KIND of string
    the judge quoted, where ``_quoted_terms`` asks whether a term is present."""
    spans = []
    for pattern in _quote_patterns(12, 300):
        spans.extend(m.group(1) for m in re.finditer(pattern, text))
    return spans


def _denied_entities(clause):
    """Candidate referents named in a denial clause: quoted terms, proper nouns,
    acronyms. Normalised, stopworded, and only ever meaningful once tested
    against the question and the context by ``_detect_question_blind``."""
    raw = list(_quoted_terms(clause))
    raw += _PROPER_NOUN_RE.findall(clause)
    raw += _ACRONYM_RE.findall(clause)
    out = []
    for term in raw:
        norm = _norm_text(term)
        if len(norm) >= 4 and not all(w in _ENTITY_STOPWORDS for w in norm.split()):
            out.append(norm)
    return out


def _detect_question_blind(df, body, score):
    """The judge faults the answer for a referent that the QUESTION supplies and
    the context legitimately lacks.

    Faithfulness compares claims against retrieved text with the question absent
    from every prompt, so an answer that correctly reasons about the patient,
    dish or condition named in the query looks ungrounded. The test is
    positional, not lexical: an entity named inside a denial clause, present in
    this row's question, absent from this row's contexts. That holds whatever the
    entity is -- "Lebkuchen" on NGQA, "Shoakram" on LLMDRS -- which a phrase list
    does not.
    """
    if "query" not in df:
        return body.str.contains(_UNRESOLVED_REFERENT_RE)
    query = df["query"].fillna("").map(_norm_text)
    ctx = _contexts_text(df)
    flags = []
    for text, q, blob in zip(body, query, ctx):
        hit = bool(_UNRESOLVED_REFERENT_RE.search(text))
        if not hit and q:
            hit = any(term in q and term not in blob
                      for clause in _denial_clauses(text)
                      for term in _denied_entities(clause))
        flags.append(hit)
    return pd.Series(flags, index=df.index)


def _detect_meta_quote(df, body, score):
    """The reason quotes a verdict's own reasoning as though it were retrieved text.

    A reader who trusts the quotation marks comes away believing the passages
    contain sentences like "The context states this applies to 'Children and
    adolescents,' but the input profile describes a 20-year-old adult" -- which
    is the judge's earlier commentary, not a guideline. See ``_META_QUOTE_RE``
    for why the template invites it.
    """
    def _is_meta(span):
        return bool(_META_QUOTE_START_RE.match(span)
                    or _META_QUOTE_BODY_RE.search(span))

    return pd.Series(
        [any(_is_meta(q) for q in _quoted_spans(t)) for t in body],
        index=df.index)


def _detect_fabricated_citation(df, body, score):
    n_ctx = _n_contexts(df)
    return pd.Series(
        [bool(_cited_indices(t) - set(range(1, max(int(n), 1) + 1)))
         for t, n in zip(body, n_ctx)],
        index=df.index)


def _detect_self_refuting(df, body, score):
    """The judge calls the answer contradictory while quoting the answer's own
    value as the one the context gives.

    Requires all three of: a sub-perfect score, contradiction language, and a
    quantity that the reason attributes to the context and the answer also
    states. Any two of those can co-occur innocently; together they mean the
    reason refutes itself.
    """
    if "answer" not in df:
        return pd.Series(False, index=df.index)
    answers = df["answer"].fillna("")
    flags = []
    for text, ans, sc in zip(body, answers, score):
        flags.append(bool(
            pd.notna(sc) and sc < 1.0
            and _CONTRADICTION_RE.search(text)
            and _ATTRIBUTION_RE.search(text)
            and (_quantities(text) & _quantities(ans))))
    return pd.Series(flags, index=df.index)


def _denial_clauses(text):
    """The clauses of a reason that actually assert an absence.

    Scoping matters more than it looks. A contextual-relevance reason routinely
    quotes context statements it found RELEVANT in one clause and denies
    something else in another; matching the denial phrase against terms drawn
    from the whole reason would flag every one of those as a false denial. Only
    a term inside the denying clause is evidence about what was denied.
    """
    return [c for c in re.split(r"[.;]|,\s+(?:and|or|but)\s+", text) if _DENIAL_RE.search(c)]


def _detect_false_denial(df, body, score):
    """The judge states a concrete term is absent from the context while the
    context contains it verbatim.

    Restricted to multi-word terms the prose itself sets apart (see
    ``_quoted_terms``) and to the clause doing the denying (see
    ``_denial_clauses``), so neither a generic noun appearing somewhere in the
    passages nor a correctly quoted statement elsewhere in the reason can be
    mistaken for a refutation.
    """
    ctx = _contexts_text(df)
    flags = []
    for text, blob in zip(body, ctx):
        hit = False
        if blob:
            hit = any(term in blob
                      for clause in _denial_clauses(text)
                      for term in _quoted_terms(clause))
        flags.append(hit)
    return pd.Series(flags, index=df.index)


class JudgeErrorClass(NamedTuple):
    """One failure mode visible in a DeepEval judge's stated reason.

    ``metrics`` gates the class to the metrics whose prompt chain can produce it
    (see ``DEEPEVAL_JUDGE_SEES``). ``precision`` records how the counts should be
    read: an ``exact`` class is decidable from the row alone, a ``candidate``
    class is a regex lead that needs eyeballing before it is quoted as an error.

    ``stage`` is the column that decides what a hit is worth, and it exists
    because every class here is detected in the same place -- the reason string --
    while only some of them are ABOUT the scoring:

      ``verdict``  the prose describes why claims were faulted, so it is evidence
                   about the yes/no/idk verdicts that produced the score. A hit
                   means the score itself is suspect.
      ``reason``   the prose is an artifact of the summarising step that runs
                   AFTER ``_calculate_score`` (``self.score = ...`` on the line
                   above ``self.reason = ...`` in all three metrics). A hit
                   CANNOT have moved the score. It is a documentation defect: it
                   makes the string unsafe to quote, and nothing more.

    Reporting the two together as "judge errors" overstates the second kind.
    """
    name: str
    metrics: tuple
    detect: object
    precision: str
    stage: str
    description: str


# Ordered verdict-stage first: those are the hits that bear on a score. The
# reason-stage classes below them are prose defects on a string generated after
# the score was fixed, and are reported so the prose is not mined as evidence.
JUDGE_ERROR_CLASSES = (
    JudgeErrorClass(
        "question_blind", ("deepeval_faithfulness",), _detect_question_blind,
        "candidate", "verdict",
        "Faults the answer for a referent the question supplies and the context "
        "lacks; the faithfulness chain never receives the question."),
    JudgeErrorClass(
        "self_refuting", ("deepeval_faithfulness",), _detect_self_refuting,
        "exact", "verdict",
        "Calls the answer contradictory while quoting the answer's own value as "
        "the context's correct one."),
    # Faithfulness only, and not for want of trying: on contextual relevance the
    # same detector fires on correct reasons. That judge's job IS to call
    # retrieved content irrelevant, and it says so with the same words ("the
    # context does not provide dietary recommendations") while correctly quoting
    # a statement that is present. Absence and irrelevance are one vocabulary and
    # two claims, and only faithfulness penalises the ANSWER for the first.
    JudgeErrorClass(
        "false_denial", ("deepeval_faithfulness",),
        _detect_false_denial, "candidate", "verdict",
        "States a concrete multi-word term is absent from the context, which "
        "contains it verbatim."),
    JudgeErrorClass(
        "speculative", ("deepeval_faithfulness", "deepeval_relevance"),
        _detect_speculative, "candidate", "reason",
        "Guesses at the answer's content ('likely provided'); both reason steps "
        "are blind to the answer they describe. Score already fixed."),
    JudgeErrorClass(
        # Faithfulness only. Contextual relevance's reason prompt contains no
        # numbered list of anything -- just the input, two flat string lists and
        # the score -- so there is no index for the model to miscount, and it
        # scored 0 of 536 there. Carrying an unreachable class would pad the
        # table with a row that can only ever read zero.
        "fabricated_citation", ("deepeval_faithfulness",),
        _detect_fabricated_citation, "exact", "reason",
        "Numbers its contradiction strings as if they were passages, citing an "
        "index beyond the contexts that exist. Score already fixed."),
    # Contextual relevance only. Removed from answer relevance on the evidence:
    # it never fired there, and that metric's reason step is handed a different
    # mix of strings, so a shared class would imply a shared mechanism that is
    # not there.
    #
    # Kept deliberately mild. The template hands the model a list of verdict
    # REASONS and a list of context STATEMENTS and tells it to quote from both,
    # so quoting a reason is compliance, not confusion, and the prose usually
    # attributes it ("as noted by the reason that ..."). A minority genuinely
    # relabels reasons as context content. Either way the score was fixed before
    # this string existed: the only real cost is that these quoted spans cannot
    # be cited as evidence of what was retrieved.
    JudgeErrorClass(
        "meta_quote", ("deepeval_contextual_relevance",),
        _detect_meta_quote, "candidate", "reason",
        "Quotes a verdict's own reasoning inside quotation marks; usually "
        "attributed as such, sometimes passed off as retrieved text."),
)


def _reason_col(metric):
    return f"deepeval_scores.{metric}_reason"


def _score_col(metric):
    return f"deepeval_scores.{metric}"


def judge_error_classes(metric):
    """The taxonomy entries that apply to one DeepEval metric."""
    return tuple(c for c in JUDGE_ERROR_CLASSES if metric in c.metrics)


def judge_error_flags(df, metric="deepeval_faithfulness", exclude_abstentions=True):
    """Boolean flag per row x error class for one DeepEval metric.

    Returns ``(flags, mask)``: a DataFrame with one column per applicable class,
    and the mask of rows eligible at all -- metric scored, prose present, and (by
    default) not an abstention, since an abstention's faithfulness is not a
    judgement about an answer and is excluded from the analysis cohort anyway.

    Every table below is built from this one function, so adding a failure mode
    means adding one ``JUDGE_ERROR_CLASSES`` entry and nothing else.
    """
    scol, rcol = _score_col(metric), _reason_col(metric)
    if scol not in df or rcol not in df:
        return pd.DataFrame(index=df.index), pd.Series(False, index=df.index)

    score = pd.to_numeric(df[scol], errors="coerce")
    prose = df[rcol].astype("string").fillna("")
    mask = score.notna() & prose.str.strip().ne("")
    if exclude_abstentions:
        mask &= ~ra._abstained(df)

    body = _reason_body(df[rcol])
    flags = pd.DataFrame(index=df.index)
    for cls in judge_error_classes(metric):
        flags[cls.name] = cls.detect(df, body, score).fillna(False) & mask
    return flags, mask


def judge_error_summary(df, metrics=None, exclude_abstentions=True):
    """Per metric x error class: how often the judge's stated reason is indefensible.

    ``n_eligible`` is the cohort each rate is over. ``mean_score_flagged`` vs
    ``mean_score_rest`` shows which way the class pulls the metric -- a class that
    only ever fires at 0.00 is a one-directional bias, not noise. ``precision``
    carries the class's exact/candidate status, so a table in the appendix can say
    which counts are decidable and which are leads.
    """
    metrics = metrics or [m for m in DEEPEVAL_JUDGE_SEES
                          if _score_col(m) in df and _reason_col(m) in df]
    rows = []
    for metric in metrics:
        flags, mask = judge_error_flags(df, metric, exclude_abstentions)
        if not len(flags.columns):
            continue
        score = pd.to_numeric(df[_score_col(metric)], errors="coerce")
        n_elig = int(mask.sum())
        for cls in judge_error_classes(metric):
            hit = flags[cls.name]
            rest = mask & ~hit
            rows.append({
                "metric": metric,
                "error_class": cls.name,
                "stage": cls.stage,
                "affects_score": cls.stage == "verdict",
                "precision": cls.precision,
                "n_eligible": n_elig,
                "n_flagged": int(hit.sum()),
                "rate": round(hit.sum() / n_elig, 4) if n_elig else _NAN,
                "mean_score_flagged":
                    round(float(score[hit].mean()), 3) if hit.any() else _NAN,
                "mean_score_rest":
                    round(float(score[rest].mean()), 3) if rest.any() else _NAN,
            })
    return pd.DataFrame(rows)


def judge_error_by(df, metric="deepeval_faithfulness", by="source_dataset",
                   exclude_abstentions=True, as_rate=True):
    """The taxonomy broken down by group -- the table that says whether a failure
    mode is a property of the judge or of one dataset's task shape.

    Rows are the groups, columns the error classes, plus ``n_eligible`` and
    ``any_class``. With ``as_rate`` the cells are shares of that group's eligible
    rows, otherwise raw counts. A class concentrated in a single dataset is a
    task-shape artifact and belongs in that dataset's caveats rather than in a
    general claim about the judge.
    """
    flags, mask = judge_error_flags(df, metric, exclude_abstentions)
    if not len(flags.columns) or by not in df:
        return pd.DataFrame()
    grp = df[by].astype("object")
    out = pd.DataFrame({"n_eligible": mask.groupby(grp, observed=True).sum()})
    for name in flags.columns:
        out[name] = flags[name].groupby(grp, observed=True).sum()
    out["any_class"] = flags.any(axis=1).groupby(grp, observed=True).sum()
    out = out[out["n_eligible"] > 0]
    if as_rate:
        cols = [c for c in out.columns if c != "n_eligible"]
        out[cols] = out[cols].div(out["n_eligible"], axis=0).round(3)
    return out.astype({"n_eligible": "int64"})


def judge_error_rows(df, metric="deepeval_faithfulness", classes=None,
                     exclude_abstentions=True, max_chars=400, id_col="id"):
    """Long evidence table -- one row per (row, error class) hit, with the prose.

    This is what an appendix quotes: a reader has to be able to check the
    classification, and no summary rate lets them. ``max_chars`` truncates the
    prose only; the score and the paired RAGAS faithfulness travel with it, so a
    hit can be read against the metric it disagrees with.
    """
    flags, _ = judge_error_flags(df, metric, exclude_abstentions)
    if not len(flags.columns):
        return pd.DataFrame()
    wanted = [c for c in flags.columns if classes is None or c in classes]
    keep = [c for c in (id_col, "source_dataset", "variant", "lang") if c in df]
    scol, rcol = _score_col(metric), _reason_col(metric)
    score = pd.to_numeric(df[scol], errors="coerce")
    peer = "ragas_scores.ragas_faithfulness"
    peer_val = pd.to_numeric(df[peer], errors="coerce") if peer in df else None
    records = []
    for name in wanted:
        for i in df.index[flags[name]]:
            rec = {k: df.at[i, k] for k in keep}
            rec["error_class"] = name
            rec[metric] = score.at[i]
            if peer_val is not None:
                rec["ragas_faithfulness"] = peer_val.at[i]
            rec["reason"] = str(df.at[i, rcol])[:max_chars]
            records.append(rec)
    out = pd.DataFrame(records)
    return out.sort_values(["error_class"] + keep) if len(out) else out


PEER_FAITHFULNESS = ("ragas_scores.ragas_faithfulness",
                     "ragas_scores.ragas_faithfulness_with_hhem")


def peer_scores_on_flagged(df, metric="deepeval_faithfulness",
                           peers=PEER_FAITHFULNESS, stages=("verdict",),
                           exclude_abstentions=True):
    """What the OTHER metrics said about the rows where this judge got it wrong.

    Restricted by default to ``stage="verdict"`` classes, because those are the
    only ones where the flagged prose is evidence about a scoring decision; a
    reason-stage artifact says nothing about whether the score was right, so
    pooling it in would dilute the comparison with rows that are fine.

    The reliability argument this supports: if a second and a third scorer, one
    of them not an LLM at all, rate these rows near 1.0 while the flagged judge
    rates them near 0.0, the disagreement localises to the flagged judge rather
    than to the answers. ``frac_one`` is the sharpest column for that -- it counts
    how often a peer called the very same answer FULLY grounded.

    ``__unflagged__`` is the control: the gap between a class row and it is the
    claim, not the class row alone.
    """
    flags, mask = judge_error_flags(df, metric, exclude_abstentions)
    if not len(flags.columns):
        return pd.DataFrame()
    wanted = [c.name for c in judge_error_classes(metric)
              if c.stage in stages and c.name in flags.columns]
    if not wanted:
        return pd.DataFrame()
    cols = [_score_col(metric)] + [p for p in peers if p in df]
    vals = {c: pd.to_numeric(df[c], errors="coerce") for c in cols}
    any_hit = flags[wanted].any(axis=1) & mask

    def _stats(name, sel):
        row = {"group": name, "n": int(sel.sum())}
        for c in cols:
            v = vals[c][sel].dropna()
            short = c.split(".")[-1]
            row[f"{short}_n"] = len(v)
            row[f"{short}_mean"] = round(float(v.mean()), 3) if len(v) else _NAN
            row[f"{short}_frac_one"] = round(float((v == 1.0).mean()), 3) if len(v) else _NAN
            row[f"{short}_frac_zero"] = round(float((v == 0.0).mean()), 3) if len(v) else _NAN
        return row

    rows = [_stats(name, flags[name] & mask) for name in wanted]
    rows.append(_stats("__any_verdict_stage__", any_hit))
    rows.append(_stats("__unflagged__", mask & ~any_hit))
    rows.append(_stats("__all_eligible__", mask))
    return pd.DataFrame(rows)


def variant_flip(df, metric, a="rag", b="rag_sc", threshold=None,
                 exclude_abstentions=True, id_col="id"):
    """Paired 2x2: for questions scored under BOTH variants, who moved and which way.

    A difference in two group rates does not say whether the same questions
    behave differently -- it can be two disjoint sets of rows drifting. Pairing on
    ``id_col`` and crosstabbing the outcome under ``a`` against the outcome under
    ``b`` shows the discordant cells directly, and an exact McNemar test says
    whether the imbalance between them survives the sample size.

    ``threshold`` picks what "outcome" means. Given a number, a row counts as a
    hit when the metric is BELOW it (so the table reads low-score vs not). Given
    ``None``, the outcome is "this row tripped any judge-error class", which is
    the flagged/clean cut. Returns ``(table, stats)``.
    """
    if threshold is None:
        flags, mask = judge_error_flags(df, metric, exclude_abstentions)
        if not len(flags.columns):
            return pd.DataFrame(), {}
        hit, label = flags.any(axis=1), "flagged"
    else:
        val = pd.to_numeric(df[_score_col(metric)], errors="coerce")
        mask = val.notna()
        if exclude_abstentions:
            mask &= ~ra._abstained(df)
        hit, label = val < threshold, f"below_{threshold:g}"

    sub = df.loc[mask, [id_col, "variant"]].copy()
    sub["hit"] = hit[mask]
    sub["variant"] = sub["variant"].astype(str)
    sub = sub[sub["variant"].isin([a, b])]
    piv = sub.pivot_table(index=id_col, columns="variant", values="hit",
                          aggfunc="max").dropna()
    if a not in piv or b not in piv or piv.empty:
        return pd.DataFrame(), {}

    tab = pd.crosstab(piv[a].astype(bool), piv[b].astype(bool))
    tab = tab.reindex(index=[False, True], columns=[False, True], fill_value=0)
    tab.index = pd.Index([f"{a} clean", f"{a} {label}"], name="")
    tab.columns = pd.Index([f"{b} clean", f"{b} {label}"], name="")

    only_a = int(tab.iloc[1, 0])
    only_b = int(tab.iloc[0, 1])
    stats = {"metric": metric, "outcome": label, "n_paired": int(len(piv)),
             f"{a}_rate": round(float(piv[a].mean()), 3),
             f"{b}_rate": round(float(piv[b].mean()), 3),
             f"only_{a}": only_a, f"only_{b}": only_b,
             "n_discordant": only_a + only_b}
    if only_a + only_b:
        try:
            from scipy.stats import binomtest
            pval = float(binomtest(only_a, only_a + only_b, 0.5).pvalue)
            # Rounding to 4dp turns a decisive p into a bare "0.0", which reads
            # like a missing value rather than a result.
            stats["mcnemar_p"] = round(pval, 4) if pval >= 1e-4 else f"{pval:.1e}"
        except ImportError:
            stats["mcnemar_p"] = _NAN
    else:
        stats["mcnemar_p"] = _NAN
    return tab, stats


def score_rail_counts(df, metrics=None, by=("source_dataset", "variant"),
                      exclude_abstentions=True):
    """How many rows of each group sit exactly on 0.0 and exactly on 1.0.

    ``metric_distribution`` already reports ``frac_zero`` / ``frac_one``, but a
    fraction is the wrong unit for a claim like "the judge returns 0.00 on 258 of
    300 NGQA retrieval rows": the sentence names a count, so the table backing it
    should print one. This also excludes abstentions by default, which
    ``metric_distribution`` deliberately does not.

    A rail count is the bluntest statement of non-discriminativeness there is: a
    metric with 86% of its mass on one rail is not ranking those rows, whatever
    its mean says.
    """
    metrics = metrics or [c for c in score_cols(df) if c.startswith("deepeval_scores.")]
    rows = []
    keep = df[~ra._abstained(df)] if exclude_abstentions else df
    by = [by] if isinstance(by, str) else list(by)
    groups = ([(name, sub) for name, sub in keep.groupby(by, observed=True)]
              if by else [((), keep)])
    for name, sub in groups:
        name = name if isinstance(name, tuple) else (name,)
        for metric in metrics:
            if metric not in sub:
                continue
            val = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if val.empty:
                continue
            n = len(val)
            n_zero, n_one = int((val == 0.0).sum()), int((val == 1.0).sum())
            rows.append(dict(zip(by, name), **{
                "metric": metric.split(".")[-1], "n_scored": n,
                "n_zero": n_zero, "n_one": n_one,
                "frac_zero": round(n_zero / n, 3), "frac_one": round(n_one / n, 3),
                "n_between": n - n_zero - n_one,
                "mean": round(float(val.mean()), 3),
            }))
    out = pd.DataFrame(rows)
    return out.set_index(by + ["metric"]).sort_index() if len(out) and by else out


def judge_error_divergence(df, metric="deepeval_faithfulness",
                           against="ragas_scores.ragas_faithfulness",
                           exclude_abstentions=True):
    """Does each error class explain the disagreement between two metrics?

    Per class: the two means over the flagged rows, how the pair orders there
    (``n_lower`` counts rows where the DeepEval metric sits below its
    counterpart), and the correlation recomputed with the class REMOVED.

    ``delta_r`` is the honest test. If dropping a class barely moves ``r``, that
    class is not what drives the two metrics apart, however indefensible its
    individual reasons are -- a distinction worth making before a handful of
    quotable errors gets promoted into an explanation of a correlation.
    """
    scol = _score_col(metric)
    if scol not in df or against not in df:
        return pd.DataFrame()
    flags, mask = judge_error_flags(df, metric, exclude_abstentions)
    x = pd.to_numeric(df[scol], errors="coerce")
    y = pd.to_numeric(df[against], errors="coerce")
    mask = mask & x.notna() & y.notna()
    if not mask.any():
        return pd.DataFrame()
    _, base_r, base_rho = _corr(x[mask], y[mask])

    def _row(name, hit, r_without, rho_without):
        return {
            "error_class": name, "n": int(hit.sum()),
            "mean_metric": round(float(x[hit].mean()), 3),
            "mean_against": round(float(y[hit].mean()), 3),
            "n_lower": int((x[hit] < y[hit]).sum()),
            "n_higher": int((x[hit] > y[hit]).sum()),
            "n_equal": int((x[hit] == y[hit]).sum()),
            "pearson_without": round(r_without, 4),
            "delta_pearson": round(r_without - base_r, 4),
            "spearman_without": round(rho_without, 4),
            "delta_spearman": round(rho_without - base_rho, 4),
        }

    rows = [_row("__baseline__", mask, base_r, base_rho)]
    subsets = [(name, flags[name] & mask) for name in flags.columns]
    subsets.append(("__all_flagged__", flags.any(axis=1) & mask))
    for name, hit in subsets:
        if not hit.any():
            continue
        _, r_without, rho_without = _corr(x[mask & ~hit], y[mask & ~hit])
        rows.append(_row(name, hit, r_without, rho_without))
    return pd.DataFrame(rows)


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

def round_keeping_pvalues(df, ndigits=3, p_cols=("wilcoxon_p",)):
    """``df.round(n)`` that does not flatten a decisive p-value into ``0.0``.

    Rounding the whole frame to 3 dp turns 4e-07 into 0.0, which reads as a
    missing value in the very column where the magnitude is the finding. The
    named columns are rendered in scientific notation below 1e-4 instead.
    """
    out = df.round(ndigits)
    for col in p_cols:
        if col in df:
            out[col] = [
                "" if pd.isna(v) else (f"{v:.1e}" if abs(v) < 1e-4 else f"{round(v, 4):g}")
                for v in pd.to_numeric(df[col], errors="coerce")
            ]
    return out


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
        err_flags = metric_error_report(d, cls, source=eval_path)
        # The one table in that report worth having as a file: it is the evidence for
        # every "metric X could not be scored on N of these rows" sentence, and the
        # length columns are what make such a sentence a coverage limit rather than a
        # footnote about n.
        lp = err_flags.get("error_length_profile")
        if lp is not None and len(lp):
            lp.to_csv(paths.table(eval_path, "error_length_profile"))

        # (0d) is the scale what the results chapter says it is? --------------------
        # Still the RAW frame, deliberately: this audits what the SCORERS emitted, so
        # a value outside its range must be caught before any exclusion can hide it,
        # and the level counts below cover the abstained cells that `prepare` nulls.
        print("\n=== score range audit: does any metric leave its declared range? ===")
        print("raw frame, before any exclusion. Read before the distribution "
              "table further down, because it is what licenses "
              "reading these numbers as [0, 1] scores at all. Three of the ragas "
              "metrics are cosine similarities that the library does not clip, so "
              "their definitional floor is negative (can_be_negative=True) however "
              "the sample happens to look; n_below / n_above / n_negative are the "
              "violations and must all be 0, and margin_to_lo says how close the run "
              "actually came to the floor.")
        audit = score_range_audit(d)
        if len(audit):
            print(audit.round(4).to_string())
            audit.to_csv(paths.table(eval_path, "score_range_audit"))
            bad = audit[(audit["n_below"] + audit["n_above"]) > 0]
            print(f"\nout-of-range cells: {int(bad[['n_below', 'n_above']].to_numpy().sum())}"
                  + (f" — in {', '.join(bad.index)}; those means are computed over a "
                     f"scale wider than the one reported" if len(bad)
                     else " — every score is inside its own definition's range."))
            print("\nwhat each number is:")
            for name, definition in score_definitions().items():
                print(f"  {name:32s} {definition}")

        print("\n--- discrete metrics: every level with its count ---")
        print("raw frame too, so the abstained rows are still in these counts - for "
              "ragas_answer_accuracy that is the point, since an abstention scored "
              "against a MEDQA gold that is itself a refusal is a legitimate 1.0. "
              "mean and quartiles describe a continuous score; for a metric whose "
              "support is a handful of points they invent values it never takes. "
              "ragas_answer_accuracy is the case that matters: it is the mean of two "
              "judge ratings in {0, 2, 4}/4, so 0.25 and 0.75 can ONLY arise from the "
              "two judges disagreeing, and its mean lands between two levels that are "
              "themselves nearly empty.")
        lev = score_levels(d)
        if len(lev):
            print(lev.round(3).to_string())
            lev.to_csv(paths.table(eval_path, "score_levels"))
            for by in ("source_dataset", "variant"):
                tab = score_levels(d, by=by)
                if not len(tab):
                    continue
                print(f"\n--- the same levels by {by} ---")
                print(tab.round(3).to_string())
                tab.to_csv(paths.table(eval_path, f"score_levels_by_{by}"))



        print("")
        print("--- continuous metrics: binned, with the exact rails split out ---")
        print("the complement of the table above; between them every score column "
              "is covered. An exact 0.0 or 1.0 gets its own row because it is "
              "usually a different mechanism from the values around it: every 0.0 "
              "in ragas_answer_relevancy is the noncommittal GATE firing (the score "
              "is mean-cosine x int(not all_noncommittal), so the judge zeroed a "
              "similarity it had already computed), not a low similarity. Read "
              "cum_frac to see where a metric's effective floor really sits — a "
              "cosine-based score does not use the bottom half of [0, 1].")
        hist = score_histogram(d)
        if len(hist):
            print(hist.round(4).to_string())
            hist.to_csv(paths.table(eval_path, "score_histogram"))
            for by in ("source_dataset", "variant"):
                tab = score_histogram(d, by=by)
                if not len(tab):
                    continue
                tab.to_csv(paths.table(eval_path, f"score_histogram_by_{by}"))
            print("")
            print(f"the by-dataset and by-variant cuts are written to "
                  f"score_histogram_by_*.csv rather than printed — 12 bins x "
                  f"{hist.index.get_level_values('metric').nunique()} metrics x 5 "
                  f"datasets does not read as a console table.")

        # Still the raw frame: these rows are the ones an exclusion would be most
        # likely to remove, and they are the evidence for what the probe's
        # abstention rate actually measures.
        print("\n=== out-of-scope probe: the rows that did NOT abstain ===")
        probe_rows = probe_non_abstentions(d)
        probe_all = d[d["source_dataset"].astype(str).isin(ABSTENTION_SCORED_DATASETS)]
        n_elig = int(probe_all["variant"].astype(str).isin(ABSTAINING_VARIANTS).sum())
        print(f"{'/'.join(sorted(ABSTENTION_SCORED_DATASETS))} is a rejection probe whose "
              f"gold IS the refusal string, so an answered row scores ~0 on both "
              f"reference metrics whatever it says. {len(probe_rows)} of the {n_elig} "
              f"rows that could abstain answered instead "
              f"({probe_rows['id'].nunique() if len(probe_rows) else 0} distinct "
              f"questions); the {len(probe_all) - n_elig} no_rag rows are excluded "
              f"because that variant is never given the abstention instruction.")
        if len(probe_rows):
            print(probe_rows.drop(columns=[c for c in ("query", "answer")
                                           if c in probe_rows])
                  .round(3).to_string(index=False))
            probe_rows.to_csv(paths.table(eval_path, "probe_non_abstentions"),
                              index=False)
            print("\nread these by hand — high retrieval_best AND contextual_relevance "
                  "means the corpus does cover the question (clinical-nutrition "
                  "guidelines are written about disease states), so the refusal gold is "
                  "wrong for that row rather than the answer being a hallucination:")
            for _, r in probe_rows.iterrows():
                print(f"\n  [{r['id']} / {r['variant']}] gold: "
                      f"{r.get('original_medqa_answer', '?')}")
                print(_indent(f"Q: {r['query']}", 4))
                print(_indent(f"A: {r['answer']}", 4))

        # Also the raw frame, and for the same reason: the degenerate rows below
        # are scored rows that should never have been scored, and any exclusion
        # applied first would hide how many there were.
        print("\n=== multiple-choice flattening: questions that lost their options ===")
        mc = mc_flatten_audit(d)
        if len(mc):
            prev = mc.groupby("dataset")[["pointer", "short_stem", "dangling",
                                          "visual_ref", "gold_pointer"]].sum()
            prev.insert(0, "questions", mc.groupby("dataset").size())
            prev.insert(1, "median_chars", mc.groupby("dataset")["chars"].median())
            print(f"{'/'.join(MC_DERIVED_DATASETS)} are multiple-choice sets whose "
                  f"loaders keep the stem and the correct option's text and drop the "
                  f"distractors. A stem that referred to its options still refers to "
                  f"them, at nothing. Counted over distinct questions:")
            print(prev.to_string())
            print(f"\n`pointer` alone is not the useful flag — read `dangling` "
                  f"(pointer AND stem < {MC_SELF_CONTAINED_CHARS} chars). A clinical "
                  f"vignette stays answerable without its options; a bare "
                  f"'which of the following statements is correct?' does not. The "
                  f"median-stem-length column is why the two datasets behave "
                  f"differently despite carrying the same phrase.")
            mc.to_csv(paths.table(eval_path, "mc_flatten_audit"), index=False)
            if int(mc["gold_pointer"].sum()):
                gp = mc.loc[mc["gold_pointer"], ["dataset", "id"]]
                print(f"\n{len(gp)} questions have a gold that is itself an option "
                      f"index ('All of the above'), so no open-ended answer can match "
                      f"it and the item is unscoreable however the system behaves: "
                      f"{', '.join(gp['id'])}")

        degenerate = option_request_rows(d)
        print(f"\n{len(degenerate)} answers ask for the material the loader dropped — "
              f"the option list, or the figure an exam item referred to — instead of "
              f"answering. These are not abstentions: they carry no rejection string, "
              f"so the abstention detector counts them as answers and both reference "
              f"metrics grade a request for clarification against a nutrition fact.")
        if len(degenerate):
            by = degenerate.groupby(["source_dataset", "variant"]).size()
            print(by.to_string())
            degenerate.to_csv(paths.table(eval_path, "mc_option_requests"), index=False)
            print(degenerate.drop(columns=[c for c in ("query", "answer")
                                           if c in degenerate])
                  .round(3).to_string(index=False))
            nc = int(degenerate["noncommittal"].sum())
            print(f"\nThe group is bimodal, so read it split rather than averaged: "
                  f"{nc} of {len(degenerate)} score ragas_answer_relevancy = 0.0 "
                  f"EXACTLY, which is RAGAS's noncommittal detector firing on a bare "
                  f"request with nothing to grade. The other {len(degenerate) - nc} "
                  f"score in the ordinary band; most of those asked for the options "
                  f"and then volunteered an answer anyway. The flag is the metric's "
                  f"verdict rather than a check on the text, so it is not exact — a "
                  f"bare request can still clear zero — but the mean over the whole "
                  f"group mixes 'the metric returned a structural zero' with 'the "
                  f"model answered and was graded', which are different events.")
            for _, r in degenerate.iterrows():
                tag = "noncommittal" if r["noncommittal"] else "hedged"
                print(f"\n  [{r['id']} / {r['variant']} / {tag}]")
                print(_indent(f"Q: {r['query']}", 4))
                print(_indent(f"A: {r['answer']}", 4))

            contrast = option_request_contrast(d, degenerate)
            if len(contrast):
                print("\n--- within each affected cell: these rows vs the rest ---")
                print(contrast.round(3).to_string(index=False))
                contrast.to_csv(paths.table(eval_path, "mc_option_request_contrast"),
                                index=False)
                print("Within-cell on purpose: the variant is held fixed, because the "
                      "variants also differ in whether they may abstain at all, so a "
                      "cross-variant comparison would answer a different question.\n"
                      "Expect the three answer metrics to disagree, and note WHICH "
                      "way. answer_relevancy separates hardest and is effectively a "
                      "detector for this failure rather than a measurement of it. "
                      "answer_accuracy is close to a floor: a judge shown a request "
                      "for clarification against a gold fact has no partial credit to "
                      "award. answer_correctness moves least and typically reaches no "
                      "exact zeros at all — it carries a token-overlap component, and "
                      "these answers echo the stem's vocabulary back while asking for "
                      "the options, so they collect lexical credit the answer has not "
                      "earned. Correctness is therefore the metric LEAST able to "
                      "detect this artifact; if it is cited as a robustness check "
                      "behind accuracy, say so there.")

            adj = {k: v for k, v in MC_ADJUDICATED.items()
                   if k in set(degenerate["id"])}
            if adj:
                print(f"\n--- manual adjudication of zero-scored hedged rows "
                      f"({len(adj)} read by hand) ---")
                print("Not all of the accuracy zeros are the same thing, and the "
                      "aggregate should not be described as if they were:")
                for k, v in sorted(adj.items()):
                    print(_indent(f"{k}: {v}", 4))
                print(_indent(
                    "Net: the accuracy floor on these rows is mostly a real miss "
                    "caused by the missing options, with a thin layer of judge "
                    "severity on top. Re-crediting the over-harsh one does not "
                    "change the aggregate, which is the reason to state the "
                    "caveat rather than to correct the scores.", 4))

            impact = mc_flatten_impact(d, degenerate)
            if len(impact):
                print("\n--- what those rows cost the cell they sit in ---")
                print(impact.round(3).to_string(index=False))
                impact.to_csv(paths.table(eval_path, "mc_flatten_impact"), index=False)
                print("`_excl` recomputes the cell without them, `_delta` is the "
                      "difference. They are a dataset-preparation artifact, so a cell "
                      "with a large delta should be reported with the sensitivity "
                      "stated rather than as a property of the system. Note which "
                      "variant they land in: the abstention clause gives rag/rag_sc a "
                      "sanctioned exit from an unanswerable question and no_rag has "
                      "none, so the artifact penalises the baseline arm specifically "
                      "and inflates the apparent gain from retrieval.")

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
        if prep["n_reference_cells_masked"]:
            kept = prep["abstentions_kept"]
            print(f"prepared for analysis: {prep['n_reference_cells_masked']} reference-metric "
                  f"cells on {prep['n_abstained_rows_reference']} of the "
                  f"{prep['n_abstained_rows']} abstained rows set to NaN "
                  f"({', '.join(c.split('.')[-1] for c in prep['reference_cols_masked'])}), "
                  f"so these grade the answers the system did give. Grading a refusal "
                  f"against a gold answer returns ~0, which is indistinguishable from a "
                  f"wrong answer and would report the abstention rate as an accuracy.")
            if len(kept):
                print(f"  kept: "
                      f"{', '.join(f'{k} ({int(v)} abstained rows)' for k, v in kept.items())} "
                      f"— abstaining there is the behaviour under test, not a missing answer. "
                      f"These two metrics therefore have a different cohort rule per dataset: "
                      f"read them per dataset, not pooled.")

        ngqa_groups = ngqa_conflict_groups(d)
        if ngqa_groups.notna().any():
            print("\n\n=== NGQA: conflict structure ===")
            counts = (d.loc[ngqa_groups.notna()]
                      .assign(_g=ngqa_groups[ngqa_groups.notna()])
                      .drop_duplicates("id")["_g"].value_counts().reindex(NGQA_GROUPS))
            print("questions per group:")
            print(counts.to_string())
            print("\nA = no contradict edge, so the reference reads 'appears suitable'.\n"
                  "B = conflict, and the relevant tags all point the same way.\n"
                  "C = conflict, and the food is favourable AND unfavourable at once —\n"
                  "    NGQA's 'hard' tier is 94% this case.\n"
                  "Read this INSTEAD of NGQA's easy/medium/hard label. Reporting the tiers\n"
                  "straight makes hard look easiest, because every hard question carries a\n"
                  "conflict while easy/medium are 50/50 and the model returns a negative\n"
                  "verdict either way — label balance meeting a response bias, not difficulty.")

            contrast = ngqa_conflict_contrast(d, ngqa_groups)
            if len(contrast):
                print("\n--- reference metrics by group (pooled over variants, "
                      "question level) ---")
                print(contrast.round(3).to_string(index=False))
                contrast.to_csv(paths.table(eval_path, "ngqa_conflict_contrast"),
                                index=False)
                print("\nB-A is conflict presence at a matched tag count, and it is the "
                      "large effect. C-B is evidence mixing at a matched conflict count, "
                      "and it is small: the model does not weigh a trade-off and lose, it "
                      "omits the unfavourable tag from a list that reads as favourable.\n"
                      "Two caveats belong with any NGQA number quoted from this run. "
                      "First, `is_healthy_agrees_with_csv_answer` is False on half of "
                      "group C: there the graph-derived reference says 'not recommended' "
                      "while NGQA's own answer_hard says 'Yes', and it is the graph one "
                      "that is scored — a system reproducing NGQA's published answer is "
                      "marked wrong. Second, `low_protein` is the offending tag on a large "
                      "share of the conflicts, and the rule behind most of them "
                      "(opioid_misuse) appears nowhere in the knowledge base, so no "
                      "retriever can recover it. Both are properties of the reference, "
                      "not of the pipeline.")

        # (1c) The cross-lingual cell: German vs English -----------------------------
        # On the PREPARED frame, for the same reason as the LLMDRS block below: the
        # question is whether the numbers the results chapter reports differ by
        # language, so it has to be computed over exactly the cells that chapter
        # averages.
        xl = crosslingual_frame(d)
        if len(xl):
            # Question counts, not row counts: the cell is three variants deep, and
            # every n quoted in the prose below is a number of QUESTIONS.
            xl_q_lang = xl.drop_duplicates("id")["lang"].astype(str)
            n_de = int(xl_q_lang.eq("de").sum())
            n_en = int(xl_q_lang.eq("en").sum())
            print(f"\n\n=== cross-lingual cell: {n_de} German vs {n_en} English "
                  f"questions on the same five contexts ===")
            print(f"Question language is a manipulated factor in exactly one place in "
                  f"this run. The synthetic set has three cells — enQ_condC (English "
                  f"question, English persona, three English guideline chunks), "
                  f"enQ_refC and deQ_refC (same five reference contexts, each one "
                  f"German DGE slice plus the English IOM tables for the same "
                  f"life-stage band) — and only the REFERENCE pair exists in both "
                  f"languages. There is no deQ_condC. So question language is crossed "
                  f"with context TYPE, not with context language, and everything "
                  f"below is restricted to context_type == "
                  f"'{CROSSLINGUAL_CONTEXT_TYPE}': pooling the condition rows in "
                  f"would set German reference-table questions against English "
                  f"clinical-condition ones and report a dataset difference as a "
                  f"language effect.")
            print(f"\nIn the German arm the question, the system prompt AND the gold "
                  f"answer are German; the retrievable corpus and the gold context "
                  f"are the same for both arms. Two properties of the cell decide "
                  f"every test: the arms are NOT paired (each language got its own "
                  f"generation pass over the same contexts, so ..._de_000 and "
                  f"..._en_000 ask about different nutrients), and context_id is the "
                  f"stratum, because it pins the life-stage band and the styling "
                  f"profile at once.")

            cohort = crosslingual_cohort(xl)
            if len(cohort):
                print("\n--- the cell, at question level ---")
                print(cohort.to_string())
                cohort.to_csv(paths.table(eval_path, "crosslingual_cohort"))
                print("Every context carries both languages, so no stratum drops out "
                      "of the stratified test below. Note that styling_profile is a "
                      "property of the CONTEXT — each context runs one profile — "
                      "which is why context_id is the stratum and styling is not a "
                      "second one.")

            routing = retrieval_language_routing(xl)
            if len(routing):
                _, mix = chunk_languages()
                de_share = float(mix.get("de", float("nan")))
                print("\n--- does the retriever answer a German question from German "
                      "documents? ---")
                print(routing.round(3).to_string(index=False))
                routing.to_csv(paths.table(eval_path, "crosslingual_routing"),
                               index=False)
                print(f"\nRead `enrichment`, not `frac_match`. German is {de_share:.1%} "
                      f"of the {len(chunk_languages()[0])}-chunk corpus, so a German "
                      f"arm retrieving in German most of the time is a large "
                      f"departure from the base rate, while an English arm doing the "
                      f"same is at 1.0 by construction and can only ever be a floor "
                      f"check. Retrieval indifferent to language sits at 1.0 for both "
                      f"whatever the corpus mix is.")
                print("The consequence is a SCOPE limit, and it cuts the other way "
                      "from the scores below: a German question is effectively "
                      "answered out of the German partition of the corpus. For a "
                      "reference-value question that partition happens to be exactly "
                      "the right document (DGE-Referenzwerte), which is why the "
                      "German arm does well here; for any German question the DGE "
                      "tables do not cover there is almost no corpus to retrieve "
                      "from, and this cell cannot measure that case because no German "
                      "condition questions were generated.")
                print("\nBUT the number above is CONFOUNDED, and should not be quoted "
                      "on its own. These questions ask for nutrient reference values, "
                      "the two reference-value tables are the relevant documents, and "
                      "one of the two is German — so a German share far above 4.7% is "
                      "also what a perfectly language-blind retriever would produce "
                      "by being good at its job. The controlled version follows.")

            pool_routing = reference_table_routing(xl)
            if len(pool_routing):
                _, pool_share = reference_table_pool()
                print("\n--- the same question, asked inside the reference-table pool ---")
                print(pool_routing.round(3).to_string(index=False))
                pool_routing.to_csv(
                    paths.table(eval_path, "crosslingual_routing_controlled"),
                    index=False)
                print(f"\nRestricted to the retrieved chunks that came from one of the "
                      f"two aligned reference tables — DGE (German) and IOM (English), "
                      f"which the synthetic contexts pair on the SAME life-stage band "
                      f"and the same nutrients. Inside that pool the two are "
                      f"near-duplicates differing in authority and language, and the "
                      f"pool is {pool_share:.0%} German, so a language-blind retriever "
                      f"lands near {pool_share:.2f} in BOTH arms. Topical relevance is "
                      f"held; only the choice of authority is left.")
                print("Read `frac_from_pool` first — it says both arms really are "
                      "asking the two tables, so the control applies — and then the "
                      "distance between the two arms' `frac_de_in_pool`. That "
                      "distance is the language effect on retrieval with the "
                      "confound removed, and it is the number to quote rather than "
                      "the enrichment above.")

            gold = gold_recall_by_language(xl)
            if len(gold):
                print("\n--- gold-chunk recall, split by the gold chunk's own language ---")
                print(gold.round(3).to_string())
                gold.to_csv(paths.table(eval_path, "crosslingual_gold_recall"))
                print("The price of that routing, paid by the ID-based retrieval "
                      "metrics. Every reference gold set is language-mixed (one "
                      "German chunk, two English), so a language-locked retriever is "
                      "CAPPED: 1/3 if it stays German, 2/3 if it stays English, "
                      "however well it ranks. gold_recall_all reproduces "
                      "ragas_id_context_recall exactly, which is the check that this "
                      "split is describing the same quantity the metric reports — so "
                      "quoting that metric across the two arms without this table "
                      "reads a ceiling as a retrieval-quality difference.")
                print("A second caveat belongs with any absolute value here: the gold "
                      "set names three SPECIFIC chunks, and the corpus holds many "
                      "near-duplicate reference-table chunks that answer the question "
                      "equally well. These recalls are therefore a lower bound on "
                      "retrieval quality, and are only comparable BETWEEN the arms "
                      "because both are bounded the same way.")

            alang = answer_language_audit(xl)
            if len(alang):
                print("\n--- did the answer come back in the language it was asked in? ---")
                print(alang.to_string())
                alang.to_csv(paths.table(eval_path, "crosslingual_answer_language"))
                off = int(xl["answer_lang_match"].eq(False).sum())
                unk = int(xl["answer_lang"].eq("unknown").sum())
                print(f"{len(xl) - off} of {len(xl)} rows answered in the question's "
                      f"language ({off} did not, {unk} undecidable). Worth its space "
                      f"precisely when it finds nothing: language drift — a German "
                      f"question answered in English because both the corpus and the "
                      f"model's training data are English-heavy — is the failure this "
                      f"cell exists to catch, and without the check a low German "
                      f"score cannot be told apart from an English answer. Detected "
                      f"by marker words and umlauts rather than a language-id "
                      f"package; `unknown` is the detector declining, and those rows "
                      f"are to be read rather than averaged.")

            notation = quantity_notation_audit(xl)
            if len(notation):
                print("\n--- are the NUMBERS written in the right convention? ---")
                print(notation.round(3).to_string(index=False))
                notation.to_csv(paths.table(eval_path, "crosslingual_notation"),
                                index=False)
                print("Answering in the right language is not the same as localising "
                      "the values, and for a reference-value system the second is "
                      "what a user acts on: '5,5 mg/Tag' and '5.5 mg/day' are the "
                      "same quantity in two conventions, and the language audit above "
                      "cannot see the difference — the prose can be flawless German "
                      "around a number written the American way.")
                print("Read the off-diagonal cells. They are not merely localisation "
                      "slips: DGE prints '4,0 µg/Tag' and IOM prints '2.4 µg/d', so a "
                      "German answer carrying '/d' is usually one that read the "
                      "ENGLISH table, and the notation is reporting where the number "
                      "came from. The next table tests that directly.")

            prov = quantity_provenance(xl)
            prov_sum = quantity_provenance_summary(prov)
            if len(prov_sum):
                print("\n--- which table's NUMBERS did each arm end up quoting? ---")
                print(prov_sum.round(3).to_string())
                prov.to_csv(paths.table(eval_path, "crosslingual_quantity_provenance"),
                            index=False)
                prov_sum.to_csv(
                    paths.table(eval_path, "crosslingual_quantity_provenance_summary"))
                print("The consequence the score tables cannot express. Each reference "
                      "context pairs a DGE slice with the IOM tables for the same "
                      "life-stage band, and the two DISAGREE on many nutrients — "
                      "Vitamin B12 for men 65+ is 4.0 µg/Tag in DGE and 2.4 µg/d in "
                      "IOM, Vitamin E is 8 against 15 mg. So one nutrient and one age "
                      "band have two defensible answers, and "
                      "`frac_dge_when_they_differ` says which one the user is told. "
                      "The distance between the two arms on that column is the "
                      "language effect on the ADVICE, not on a metric.")
                print("Every comparison is on (nutrient, value, unit), never on the "
                      "number alone: both tables are dense grids in which the same "
                      "figure recurs across unrelated nutrients, so a value-only "
                      "match would credit 'quotes DGE's vitamin A' to an answer that "
                      "merely said 700 of something else. `unattributed` is the "
                      "coverage figure — quantities whose nutrient could not be "
                      "identified from the answer or its question, held out of every "
                      "column rather than counted as a miss.")
                print("The `from_*` columns attribute against the RETRIEVED context, "
                      "not the gold one, and that is the primary view: retrieval "
                      "reaches the gold chunks on under a third of rows, so an "
                      "answer's number usually comes from a reference-table chunk "
                      "that is correct but not the gold one. Attributed against gold "
                      "those land in `gold_neither`, which is true and uninformative. "
                      "`from_both` is a value the two tables share, carrying no "
                      "routing signal and excluded from the fraction; `from_other` is "
                      "a value taken from a retrieved document that is neither table. "
                      "The `gold_*` columns keep the stricter view for comparison.")
                print("`ungrounded` is the count of TAGGED quantities appearing in NO "
                      "retrieved chunk — a hard check with no judge and no NLI model "
                      "in it, holding both arms to exactly the same standard. Read it "
                      "beside the faithfulness scorers that disagree about the two "
                      "languages: if it says the arms are equally grounded, that is "
                      "evidence about the SCORERS. Two limits on it, both in the "
                      "conservative direction. It bounds hallucination from above and "
                      "does not measure accuracy — a number can be right and "
                      "unretrieved, or grounded but copied from the wrong age band, "
                      "since nutrient and value are matched and the life-stage ROW is "
                      "not. And it over-reports: two DGE iron rows print as "
                      "'Prämenop ausal 16 Postmeno pausal 14 mg/Tag', a PDF-extraction "
                      "artifact in which the first value carries no unit of its own, "
                      "so a correct answer quoting it is scored ungrounded. Those two "
                      "lines are the only ones in either table with that shape, and "
                      "they account for every ungrounded quantity in this run.")

            abst = pd.crosstab(xl["lang"].astype(str),
                               xl["variant"].astype(str), values=ra._abstained(xl),
                               aggfunc="mean")
            if len(abst):
                print("\n--- abstention rate by language x variant ---")
                print(abst.round(3).to_string())
                print("Not a result on its own — it is why the n columns below are "
                      f"not {n_de} and {n_en} on every row. An abstained row carries "
                      "no faithfulness and no reference score, so each metric is "
                      "compared over the rows it was actually computed on.")
                length = xl.groupby([xl["lang"].astype(str),
                                     xl["variant"].astype(str)],
                                    observed=True)["answer_chars"].mean()
                print("\nmean answer length (characters), the standing confound for "
                      "every judge-scored metric below:")
                print(length.round(0).to_string())

            xl_q = language_contrast(xl, level="question")
            if len(xl_q):
                print("\n--- per metric: German against English, one observation per "
                      "question ---")
                print(round_keeping_pvalues(
                    xl_q, p_cols=("p", "p_holm", "p_perm", "p_perm_holm"))
                    .to_string(index=False))
                xl_q.to_csv(paths.table(eval_path, "crosslingual_contrast_question"),
                            index=False)
                print("\nRead `delta` against `delta_strat` FIRST, before any p. "
                      "delta_strat is the same difference computed within context and "
                      "pooled, so it holds the life-stage band and the styling "
                      "profile fixed; where the two agree, the gap is not the framing. "
                      "p_perm permutes the language label within context and is the "
                      "test this cell can actually support — the arms are unpaired but "
                      "not unstructured, and no normal approximation should be trusted "
                      "at these n. Both Holm columns correct across the rows of this "
                      "table only.")
                sig = xl_q[xl_q["p_perm_holm"] < 0.05]
                print(f"{len(xl_q)} metrics tested, {len(sig)} significant after Holm "
                      f"on the permutation p"
                      + (f": {', '.join(sig['metric'])}." if len(sig) else ".")
                      + f" With {n_de} vs {n_en} questions that is the expected "
                      f"outcome for anything but a large effect, so the effect sizes "
                      f"and their CIs are the readable part of this table and the "
                      f"p-values are the guard on them.")

            xl_v = language_contrast(xl, level="variant")
            if len(xl_v):
                print("\n--- the same contrast per metric x variant (variant held "
                      "fixed) ---")
                print(round_keeping_pvalues(
                    xl_v, p_cols=("p", "p_holm", "p_perm", "p_perm_holm"))
                    .to_string(index=False))
                xl_v.to_csv(paths.table(eval_path, "crosslingual_contrast_variant"),
                            index=False)
                print("A separate correction family, so its Holm columns are over "
                      "these rows only. Rows with a blank p are metrics welded to the "
                      "same rail in both languages — there is nothing to rank, and "
                      "the delta of exactly 0 is a fact about the metric rather than "
                      "about the languages.")

            eff = language_variant_effects(xl)
            if len(eff):
                print("\n--- does retrieval buy the same thing in German as in "
                      "English? (paired within each arm) ---")
                print(round_keeping_pvalues(eff).to_string())
                eff.to_csv(paths.table(eval_path, "crosslingual_variant_effects"))
                print("The more defensible half of this analysis, and the one to "
                      "quote if only one is quoted. A language difference in the "
                      "LEVEL of a metric is confounded with everything that differs "
                      "between two sets of questions; a language difference in the "
                      "EFFECT of adding retrieval is paired within each question, so "
                      "the question cancels. Read n_pairs and n_nontied before the "
                      "p-values: split three ways, these n's leave cells where an "
                      "effect of the same size is significant in one arm and not the "
                      "other on n alone, and reporting that as 'retrieval helps "
                      "German but not English' would be an artifact of the split.")

            conflict = scorer_direction_conflict(xl_q) if len(xl_q) else pd.DataFrame()
            if len(conflict):
                print("\n--- within each family, do the scorers agree on WHICH "
                      "language scored higher? ---")
                print(conflict.to_string(index=False))
                conflict.to_csv(paths.table(eval_path, "crosslingual_scorer_conflict"),
                                index=False)
                bad = conflict[conflict["verdict"] == "CONFLICT"]
                if len(bad):
                    print(f"\n{'1 family has' if len(bad) == 1 else f'{len(bad)} families have'} "
                          f"no scorer-independent "
                          f"answer in this run: {', '.join(bad['family'])}. Two "
                          f"scorers of the SAME construct, on the SAME rows, rank the "
                          f"two languages in opposite directions by comparable "
                          f"margins. The conclusion that follows is not 'German is "
                          f"more/less faithful' — it is that a language claim in this "
                          f"family cannot be made without naming the scorer behind "
                          f"it, and that a pooled cross-lingual faithfulness number "
                          f"is partly reporting a choice of library.")
                    print("The mechanism to state alongside it: the two disagreeing "
                          "scorers are not the same kind of instrument. HHEM is a "
                          "fixed NLI cross-encoder scoring a claim against a premise, "
                          "so a German claim against a German premise and a German "
                          "claim against an English one are different tasks for it; "
                          "the DeepEval and RAGAS judges are the system LLM prompted "
                          "in English. The mismatch rows listed below are where that "
                          "shows most directly.")
                else:
                    print("\nNo family is in conflict: every scorer of a shared "
                          "construct points the same way, which is what licenses "
                          "quoting a single direction per family above.")

            ag_lang = metric_agreement(xl, by="lang")
            if len(ag_lang):
                print("\n--- do the scorers agree with each other equally well in "
                      "both languages? ---")
                print(ag_lang.round(3).to_string(index=False))
                ag_lang.to_csv(paths.table(eval_path, "crosslingual_metric_agreement"),
                               index=False)
                print("This is the measurement-side question, and on this run it is "
                      "the one with the clearest answer. Compare each pair's "
                      "mean_diff between the two language rows: a pair whose gap is "
                      "small in one language and large in the other is not measuring "
                      "the same construct in both, and any pooled statement about "
                      "'faithfulness by language' is then partly reporting which "
                      "scorer was used. Read it beside the contrast tables above — "
                      "where two faithfulness scorers put the languages in OPPOSITE "
                      "orders, the honest conclusion is that this run cannot say "
                      "which language is more faithful, not that one of them is.")

            mismatch = context_language_mismatch(xl)
            if len(mismatch):
                print(f"\n--- the {len(mismatch)} rows served context in the OTHER "
                      f"language ---")
                print(mismatch.round(3).to_string(index=False))
                mismatch.to_csv(paths.table(eval_path, "crosslingual_ctx_mismatch"),
                                index=False)
                print("Listed rather than averaged, on purpose: because the retriever "
                      "routes by language this cell is what is left over when routing "
                      "fails, nobody designed it, and it is far too small for a mean. "
                      "It is worth reading because it is the only place where the "
                      "ANSWER and the PREMISE are in different languages — the "
                      "condition under which an NLI-based faithfulness scorer is "
                      "being asked to do something it was not built for. A scorer "
                      "that collapses on these rows while the reference-graded "
                      "metrics on the SAME rows do not is telling you about itself, "
                      "not about the answer.")

            tex = language_contrast_latex(xl_q, style="simple")
            if tex:
                tex_path = paths.table(eval_path, "crosslingual_contrast.tex")
                tex_path.write_text(tex, encoding="utf-8")
                full_path = paths.table(eval_path, "crosslingual_contrast_full.tex")
                full_path.write_text(
                    language_contrast_latex(
                        xl_v, label="tab:crosslingual-de-en-full", style="full"),
                    encoding="utf-8")
                print(f"\nresults-chapter LaTeX (booktabs): the question-level table "
                      f"is {paths.rel(tex_path)}; the full per-metric x variant "
                      f"table, with bootstrap CIs, effect sizes and both Holm "
                      f"columns, is {paths.rel(full_path)}.")

            print("\n=== cross-lingual figures ===")
            plots.save_all({
                "fig_language_routing": lambda: plots.language_routing_bars(routing),
                "fig_language_contrast": lambda: plots.language_contrast_forest(xl_q),
            }, eval_path)

        # (1d) LLMDRS answer leakage ------------------------------------------------
        # On the PREPARED frame on purpose, unlike the two dataset audits above:
        # those count rows that should never have been scored, so they have to run
        # before any exclusion. This one asks whether the numbers the results
        # chapter reports are inflated on those questions, so it has to be computed
        # over exactly the cells that chapter averages.
        ov = answer_leak_overlap(d)
        leak_rows = answer_leak_flag(d)
        if len(ov) and int(leak_rows.sum()):
            n_leak_q = int(ov["leak"].sum())
            print(f"\n=== {ANSWER_LEAK_DATASET.upper()} answer leakage: do the "
                  f"{n_leak_q} questions that state part of their own answer score "
                  f"better? ===")
            print(f"{n_leak_q} of the {len(ov)} {ANSWER_LEAK_DATASET.upper()} stems "
                  f"({', '.join(ANSWER_LEAK_IDS)}) close with prescriptive prose — "
                  f"'X should consume smaller, more frequent meals', 'his diet "
                  f"requires modifications, including reducing saturated fats' — so "
                  f"the question already contains part of the recommendation it is "
                  f"graded against, and a system that paraphrases the prompt can "
                  f"collect reference-metric credit it did not earn. Read by hand "
                  f"over all {len(ov)} stems against their own golds; the "
                  f"adjudication is a constant in this module. The bar is that the "
                  f"stem states recommendation CONTENT: naming that a "
                  f"recommendation exists ('her GP suggested some adjustments') is "
                  f"the question, naming what it says ('limiting sodium intake, "
                  f"increasing fluid intake') is the answer.")
            if ANSWER_LEAK_COMPLIANT_IDS:
                print(f"\n{', '.join(ANSWER_LEAK_COMPLIANT_IDS)} is held OUT of that "
                      f"set and tested separately below: there the stem prescribes "
                      f"nothing but describes an already-compliant patient, and the "
                      f"gold endorses that description instead of replacing it "
                      f"('Nurgul is already eating a balanced diet ... continue to "
                      f"prioritize these'). Real, but a different mechanism and a "
                      f"partial one, so pooling it into the set above would average "
                      f"two things that are not the same event.")

            print("\n--- manipulation check: gold-vocabulary recall of the prompt ---")
            ov_sum = ov.groupby("leak")[["overlap", "query_chars", "gold_chars"]] \
                       .agg(["count", "mean", "median"])
            print(ov_sum.round(3).to_string())
            ov.to_csv(paths.table(eval_path, "answer_leak_overlap"), index=False)
            print("`overlap` is |content words of query ∩ content words of gold| / "
                  "|content words of gold|, and it knows nothing about the "
                  "annotation. The flagged questions sit above the rest on it, which "
                  "corroborates the flag's DIRECTION and no more: every profile "
                  "shares vocabulary with its own recommendation by construction, so "
                  "the floor is far from zero and the two groups overlap heavily. "
                  "Note query_chars in the same table — the flagged stems are also "
                  "the longer ones, so length is a live confound in everything below "
                  "and a difference cannot be attributed to the leak alone.")

            llm = d[d["source_dataset"].astype(str) == ANSWER_LEAK_DATASET]
            tab = (pd.crosstab(llm["variant"].astype(str), answer_leak_flag(llm),
                               values=ra._abstained(llm), aggfunc="mean")
                   if "variant" in llm else pd.DataFrame())
            if len(tab):
                print("\n--- abstention rate, leaking vs rest (the cohorts differ) ---")
                print(tab.rename(columns={False: "rest", True: "leaking"})
                      .round(3).to_string())
                print(f"Not a result in itself — it is the reason the n columns "
                      f"below are not {n_leak_q} and {len(ov) - n_leak_q} "
                      f"everywhere. An abstained row carries no "
                      "faithfulness and no reference score, so each metric is "
                      "compared over the rows it was actually computed on.")

            leak_tab = answer_leak_contrast(d)
            if len(leak_tab):
                print(f"\n--- per metric x variant: the {n_leak_q} against the "
                      f"other {len(ov) - n_leak_q} ---")
                print(round_keeping_pvalues(leak_tab, p_cols=("p", "p_holm"))
                      .to_string(index=False))
                leak_tab.to_csv(paths.table(eval_path, "answer_leak_contrast"),
                                index=False)
                worst = leak_tab.loc[leak_tab["p"].idxmin()] if leak_tab["p"].notna().any() else None
                n_sig = int((leak_tab["p_holm"] < 0.05).sum())
                print(f"\n{len(leak_tab)} tests, {n_sig} of them significant after "
                      f"Holm correction across the table"
                      + (f" (smallest raw p: {worst['metric']} / {worst['variant']}, "
                         f"p={worst['p']:.3f}, p_holm={worst['p_holm']:.3f}, "
                         f"rank-biserial {worst['rank_biserial']:+.3f})"
                         if worst is not None else "") + ".")
                print(f"Within-variant on purpose: each question contributes one row "
                      f"to each test, so the two samples are independent, and the "
                      f"variant is held fixed. Read the effect sizes and the n "
                      f"columns, not the p-values — {n_leak_q} questions cannot "
                      f"support a hypothesis test on their own, and the table exists "
                      f"to BOUND a possible contamination rather than to establish "
                      f"one.")

            pooled = answer_leak_contrast(d, level="question")
            if len(pooled):
                print("\n--- one observation per question (mean over its variants) ---")
                print(round_keeping_pvalues(pooled, p_cols=("p", "p_holm"))
                      .to_string(index=False))
                pooled.to_csv(paths.table(eval_path, "answer_leak_contrast_pooled"),
                              index=False)
                print("The higher-powered cut of the same data: averaging a "
                      "question's variant scores keeps n at one per question, where "
                      "pooling the raw rows would count each question three times "
                      "and inflate n threefold on correlated observations. It is a "
                      "separate correction family, so p_holm here is over these rows "
                      "only.")

            # Sensitivity: does the verdict turn on where the second mechanism was
            # put? Run the question-level cut again with the already-compliant
            # stem folded in. If the two agree, the adjudication boundary is not
            # load-bearing and the caveat can be stated without defending it.
            if ANSWER_LEAK_COMPLIANT_IDS:
                widened = answer_leak_contrast(
                    d, level="question",
                    ids=tuple(ANSWER_LEAK_IDS) + tuple(ANSWER_LEAK_COMPLIANT_IDS))
                if len(widened):
                    print(f"\n--- sensitivity: the same cut with "
                          f"{', '.join(ANSWER_LEAK_COMPLIANT_IDS)} folded in "
                          f"({len(ANSWER_LEAK_IDS) + len(ANSWER_LEAK_COMPLIANT_IDS)} "
                          f"vs the rest) ---")
                    print(round_keeping_pvalues(widened, p_cols=("p", "p_holm"))
                          .to_string(index=False))
                    widened.to_csv(
                        paths.table(eval_path, "answer_leak_contrast_widened"),
                        index=False)
                    print("Read it against the table above, not on its own: the "
                          "question is only whether the conclusion moves when the "
                          "boundary between the two leak mechanisms does.")

            # Two LaTeX tables, and the SIMPLE one is the appendix table. The
            # thesis needs a reader to take one point from this — the leaking
            # questions do not score better — and the full table asks them to
            # learn a bootstrap CI, a rank-biserial and a Holm correction to get
            # there. The full per-variant table is still written, as the thing to
            # produce if the caveat is ever challenged.
            tex = answer_leak_latex(
                pooled, label="tab:llmdrs-answer-leak", style="simple",
                caption=(
                    f"Do the LLMDRS questions whose prompt already states part of "
                    f"the gold recommendation score better than the rest? Each "
                    f"question enters once, as the mean of its scores across the "
                    f"three variants. Leaking / Rest are the group means, $\\Delta$ "
                    f"their difference, and $p$ comes from a two-sided "
                    f"Mann-Whitney $U$ test. Eight tests are reported, so a single "
                    f"$p$ below $0.05$ is expected by chance alone: after "
                    f"correcting for multiple comparisons none of them is "
                    f"significant."))
            if tex:
                tex_path = paths.table(eval_path, "answer_leak_contrast.tex")
                tex_path.write_text(tex, encoding="utf-8")
                full_tex = answer_leak_latex(
                    leak_tab, label="tab:llmdrs-answer-leak-full")
                full_path = paths.table(eval_path,
                                        "answer_leak_contrast_full.tex")
                full_path.write_text(full_tex, encoding="utf-8")
                print(f"\nappendix-ready LaTeX (booktabs): the simplified "
                      f"question-level table is {paths.rel(tex_path)}; the full "
                      f"per-metric x variant table, with effect sizes, bootstrap "
                      f"CIs and the Holm column, is {paths.rel(full_path)}.")

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
        print("pin_a / pin_b are the guard on pearson/spearman: the share of the "
              "paired rows on that metric's most common value (a/b = first/second "
              "in the pair name). A near-zero r next to a HIGH frac_within and a "
              "pin_* near 1 is range restriction, not disagreement — one metric "
              "returned essentially one number, so there is nothing for the other "
              "to correlate with and r rests on the few rows that moved.")
        ag = metric_agreement(d)
        print(ag.round(3).to_string(index=False))
        ag.to_csv(paths.table(eval_path, "eval_metric_agreement"), index=False)
        ag_by = {}
        for by in ("source_dataset", "variant"):
            print(f"\n--- agreement by {by} ---")
            ag_by[by] = tab = metric_agreement(d, by=by)
            print(tab.round(3).to_string(index=False))
            tab.to_csv(paths.table(eval_path, f"eval_metric_agreement_by_{by}"),
                       index=False)
        print("\n--- agreement by source_dataset x variant ---")
        print("read the n column first: this splits the paired rows fifteen ways, and "
              "the cells where it lands in single digits carry a correlation that is "
              "noise with a decimal point.")
        ag_by["dataset_variant"] = tab = metric_agreement(
            d, by=["source_dataset", "variant"])
        print(tab.round(3).to_string(index=False))
        tab.to_csv(paths.table(eval_path, "eval_metric_agreement_by_dataset_variant"),
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

        # (2e) judge-prose failure taxonomy -----------------------------------------
        # Abstentions are excluded throughout: their faithfulness is not a
        # judgement about an answer, and they are dropped from the analysis
        # cohort anyway.
        print("\n=== judge-error taxonomy (is the judge's stated reason defensible?) ===")
        print("one block per metric, because the classes differ between them: each is "
              "gated to the metric whose prompt chain can actually produce it. A "
              "'candidate' rate is a regex lead to eyeball, an 'exact' one is "
              "decidable from the row alone.")
        for m in DEEPEVAL_JUDGE_SEES:
            for label, table in (("chain", DEEPEVAL_JUDGE_SEES),
                                 ("reason step", DEEPEVAL_REASON_SEES)):
                sees = table[m]
                print(f"  {m} ({label}): sees "
                      + (", ".join(k for k, v in sees.items() if v) or "-")
                      + " | blind to "
                      + (", ".join(k for k, v in sees.items() if not v) or "-"))
        jerr = judge_error_summary(d)
        jerr.to_csv(paths.table(eval_path, "judge_error_summary"), index=False)

        for metric in DEEPEVAL_JUDGE_SEES:
            if not judge_error_classes(metric) or _score_col(metric) not in d:
                continue
            short = metric.replace("deepeval_", "")
            print(f"\n########## {metric} ##########")
            print(jerr[jerr["metric"] == metric].drop(columns="metric")
                  .to_string(index=False))
            for by in ("source_dataset", "variant"):
                tab = judge_error_by(d, metric, by=by)
                if not len(tab):
                    continue
                print(f"\n--- {metric}: error rate by {by} ---")
                print(tab.to_string())
                tab.to_csv(paths.table(eval_path, f"judge_error_{short}_by_{by}"))
            # rag vs rag_sc on the SAME questions. Printed twice: once on the
            # flagged/clean cut, once on the metric itself, because a rate that
            # improves between two groups and a rate that improves on the same
            # questions are different claims.
            for thr, note in ((None, "judge-error flag"), (0.5, "metric < 0.5")):
                tab, st = variant_flip(d, metric, threshold=thr)
                if not len(tab):
                    continue
                print(f"\n--- {metric}: rag vs rag_sc on the same questions "
                      f"({note}) ---")
                print(tab.to_string())
                print("  " + "  ".join(f"{k}={v}" for k, v in st.items()
                                       if k not in ("metric", "outcome")))

            ev_rows = judge_error_rows(d, metric)
            if len(ev_rows):
                ev_rows.to_csv(paths.table(eval_path, f"judge_error_rows_{short}"),
                               index=False)
                print(f"\n{len(ev_rows)} flagged reasons written to "
                      f"{paths.rel(paths.table(eval_path, f'judge_error_rows_{short}'))}")
            # Faithfulness is the only one of the three with a comparable metric to
            # be divergent FROM, so this cut belongs inside its block rather than
            # after all three, where it would read as a property of the taxonomy.
            if metric == "deepeval_faithfulness":
                peer = peer_scores_on_flagged(d, metric)
                if len(peer):
                    print(f"\n--- {metric}: what the other faithfulness scorers "
                          f"said about the score-affecting error rows ---")
                    print("verdict-stage classes only, abstentions excluded. Read "
                          "each row against __unflagged__: a peer near 1.0 where "
                          "this judge is near 0.0 localises the disagreement.")
                    print(peer.to_string(index=False))
                    peer.to_csv(paths.table(eval_path, "judge_error_peer_scores"),
                                index=False)

                div = judge_error_divergence(d, metric)
                if len(div):
                    print(f"\n--- {metric}: does any class explain the gap to "
                          f"ragas_faithfulness? ---")
                    print("delta_pearson is the test: a class whose removal barely "
                          "moves r is not what drives the two metrics apart, however "
                          "indefensible its individual reasons are.")
                    print(div.to_string(index=False))
                    div.to_csv(paths.table(eval_path, "judge_error_divergence"),
                               index=False)

        print("\n=== verdict shape: what the DeepEval judges returned per claim ===")
        print("the taxonomy above asks whether the judge's PROSE is defensible; this "
              "asks what its verdicts were, one row per metric x cohort. Read frac_idk "
              "first: DeepEval reserves 'no' for a direct contradiction, so an "
              "unsupported claim comes back 'idk', and with penalize_ambiguous_claims "
              "an idk scores 0 exactly like a no. pooled_score (= frac_yes) is the "
              "claim-weighted score as computed; pooled_if_idk_yes is the same number "
              "under the pre-2026-08-10 convention.")
        vshape = verdict_shape(d)
        if len(vshape):
            print(vshape.round(3).to_string())
            vshape.to_csv(paths.table(eval_path, "verdict_shape"))
            for by in ("source_dataset", "variant"):
                tab = verdict_shape(d, by=by, cohorts=("answered",))
                if not len(tab):
                    continue
                print(f"\n--- verdict shape on ANSWERED rows by {by} ---")
                print(tab.round(3).to_string())
                tab.to_csv(paths.table(eval_path, f"verdict_shape_by_{by}"))
        empty = empty_verdict_metrics(d)
        if empty:
            print(f"\nno verdicts persisted at all for: {', '.join(empty)} — the "
                  f"evaluator reads the metric's `verdicts` attribute, and "
                  f"ContextualRelevancyMetric exposes `verdicts_list` (grouped per "
                  f"context) instead. The SCORES for those metrics are unaffected "
                  f"(DeepEval computes them internally); only this claim-level view of "
                  f"them is missing, so no idk rate can be quoted for them.")

        for metric, peer in (("deepeval_faithfulness",
                              "ragas_scores.ragas_faithfulness"),):
            cf, cf_info = idk_counterfactual(d, metric, against=peer)
            if not len(cf):
                continue
            print(f"\n--- {metric}: does the idk convention drive the gap to "
                  f"{peer.split('.')[-1]}? ---")
            print(f"per-row score recomputed from the persisted tallies under both "
                  f"conventions, abstentions excluded, n = {cf_info['n_rows']}. "
                  f"Sanity check: yes/n_verdicts reproduces the stored score on "
                  f"{cf_info['recomputed_matches']:.1%} of rows — if that is not ~100% "
                  f"the rest of this table is void.")
            print(cf.round(3).to_string(index=False))
            cf.to_csv(paths.table(eval_path, f"idk_counterfactual_{metric}"),
                      index=False)
            print("the correlation columns are the ones that matter: a level shift is "
                  "a calibration choice, a rank change means the two libraries are "
                  "scoring different constructs and any pooled correlation between "
                  "them is partly reporting the convention.")


        print("\n=== score rails: how much of each metric sits exactly on 0 or 1 ===")
        print("counts, abstentions excluded - the unit a claim like 'the judge "
              "returns 0.00 on N of M rows' is actually made in.")
        rails = score_rail_counts(d)
        print(rails.to_string())
        rails.to_csv(paths.table(eval_path, "score_rail_counts"))

        # (2d) how the variants compare ---------------------------------------------
        for by in ("source_dataset", "variant"):
            print(f"\n=== metric means by {by} ===")
            tbl = means_by(d, by=by)
            print(tbl.round(3).to_string())
            tbl.to_csv(paths.table(eval_path, f"means_by_{by}"))

        print("\n=== paired variant comparisons (Wilcoxon signed-rank) ===")
        print("the results-chapter spine; only meaningful for a metric that the "
              "distribution table above shows actually spreads.")
        print("compare n_pairs down the block before comparing effects: the three "
              "faithfulness metrics are answered-only (their abstained cells were "
              "nulled upstream), while contextual relevance scores RETRIEVAL and is "
              "therefore defined on abstentions too, so it pairs a larger cohort.")
        for metric, pairs in VARIANT_COMPARISONS:
            if metric not in d:
                continue
            for a, b in pairs:
                cmp = ev.compare_variants(d, metric, a=a, b=b)
                print(f"\n{metric}: {a} vs {b}")
                print(round_keeping_pvalues(cmp).to_string())
                cmp.to_csv(paths.table(
                    eval_path, f"compare_{a}_vs_{b}_{metric.split('.')[-1]}"))

        print("\n=== abstention-adjusted metrics (all rows vs answered only) ===")
        print("a large positive delta means the metric is mostly measuring how often "
              "the system abstained, not how good its answers are.")
        print("the faithfulness rows necessarily show delta 0 and n_all == n_answered: "
              "their abstained cells were excluded upstream, so both columns are already "
              "the answered-only mean.")
        print(f"the two reference metrics "
              f"({', '.join(c.split('.')[-1] for c in sorted(REFERENCE_METRICS))}) are the "
              f"same story with one exception: their abstained cells were excluded upstream "
              f"too, EXCEPT on {'/'.join(sorted(ABSTENTION_SCORED_DATASETS))}. Whatever delta "
              f"they still show is therefore that dataset's abstentions alone, not a "
              f"run-wide answered-vs-all difference.")
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
            # The paired-comparison table above, drawn from the same
            # VARIANT_COMPARISONS list so the two can never disagree. It carries the
            # Wilcoxon p and the rank-biserial effect size, which is what makes the
            # small mean differences readable: on a metric-units axis a real effect
            # on a compressed metric and no effect at all look identical.
            #
            # This replaced plots.variant_effect_forest, which drew the same mean
            # differences off its own metric list — so the chapter had one figure and
            # one table that silently compared different sets of metrics.
            "fig_paired_comparisons": lambda: plots.paired_comparison_plot(
                d, VARIANT_COMPARISONS),
            "fig_metric_rails": lambda: plots.metric_rail_plot(d),
            # The same figure faceted: the pooled rails say whether a metric can
            # discriminate at all, these say where. They are the `per dataset x
            # variant` table above, drawn from the same metric_summary_by call.
            "fig_metric_rails_by_dataset": lambda: plots.metric_rail_grid(
                d, "source_dataset"),
            "fig_metric_rails_by_variant": lambda: plots.metric_rail_grid(
                d, "variant"),
            "fig_metric_rails_dataset_variant": lambda: plots.metric_rail_grid(
                d, ["source_dataset", "variant"]),
            "fig_dataset_variant_correctness": lambda: plots.dataset_variant_heatmap(
                d, "ragas_scores.ragas_answer_correctness"),
            "fig_dataset_variant_relevance": lambda: plots.dataset_variant_heatmap(
                d, "deepeval_scores.deepeval_relevance"),
            "fig_metric_agreement": lambda: plots.metric_agreement_dots(ag),
            # The same pairs faceted: the pooled dots say whether two comparable
            # metrics agree, these say where — a pair that holds on one dataset and
            # collapses on another is one construct with a domain, not two broken
            # judges, and only the facets separate those readings.
            "fig_metric_agreement_by_dataset": lambda: plots.metric_agreement_grid(
                ag_by["source_dataset"], "source_dataset"),
            # Crossed, like the rail grid: whether a pair's agreement is a property of
            # the questions or of the pipeline that answered them is only visible when
            # both are on the figure at once. Read against the n's — fifteen cells over
            # the same paired rows is thin in places, deliberately shown rather than
            # smoothed away.
            "fig_metric_agreement_dataset_variant": lambda: plots.metric_agreement_grid(
                ag_by["dataset_variant"], ["source_dataset", "variant"]),
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
