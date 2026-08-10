"""Deep-dive on the EVALUATED results, and the bridge to the pipeline signals.

Three layers already exist and this module sits on top of the other two:

  - ``analysis.rag_analysis``: the *raw* pipeline signals (generation confidence,
    hybrid retrieval scores, self-correction) with no notion of quality.
  - ``analysis.analysis``: the evaluated-results primitives (``load`` +
    numeric coercion, ``health_report``, ``metric_summary`` / ``means_by``,
    ``compare_variants``, reason mining, the plots).
  - ``analysis.eval_analysis`` (this file): the cross-cutting questions that need
    the metrics — and, for the linking parts, the pipeline signals — together.

It answers four things, in the order __main__ runs them:

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


# --- (1) Metric-computation sanity -------------------------------------------

def _ragas_error_mask(df):
    c = "ragas_scores.ragas_error"
    if c in df:
        return df[c].notna() & df[c].astype(str).str.strip().ne("")
    return pd.Series(False, index=df.index)


def _deepeval_error_mask(df):
    """Rows whose DeepEval prose signals a judge *crash* (not a low score), using
    the narrow ``analysis.EVAL_ERROR_PATTERNS`` — DeepEval has no error field."""
    rx = re.compile("|".join(ev.EVAL_ERROR_PATTERNS), re.IGNORECASE)
    cols = [c for c in df.columns
            if c.startswith("deepeval_scores.") and c.endswith("_reason")]
    m = pd.Series(False, index=df.index)
    for c in cols:
        t = df[c].astype("string").fillna("")
        m = m | (t.str.strip().ne("") & t.str.contains(rx))
    return m


def classify_metrics(df):
    """Long table (id, variant, source_dataset, metric, status) tagging every
    metric cell as one of: ``scored`` / ``na_no_context`` (context metric on
    no_rag) / ``na_no_reference`` (ID-context metric off the synthetic set) /
    ``na_rejected`` (relevancy metric on an abstention) / ``error`` (applicable
    but missing — a scorer raised or the value never landed).

    The status is derived from variant + dataset + the ``rejected`` flag, NOT from
    the sentinel strings, so it is robust to wording changes. This is the single
    source of truth for the error report.
    """
    rej = ev._rejection_mask(df)
    is_norag = df["variant"].astype(str).eq("no_rag")
    is_synth = (df["source_dataset"].eq("synthetic_guidelines")
                if "source_dataset" in df else pd.Series(False, index=df.index))

    frames = []
    for m in CLASSIFIED_METRICS:
        if m not in df:
            continue
        val = pd.to_numeric(df[m], errors="coerce")
        status = pd.Series("scored", index=df.index, dtype="object")
        if m in CONTEXT_METRICS or m in IDCTX_METRICS:
            status = status.mask(is_norag, "na_no_context")
        if m in IDCTX_METRICS:
            status = status.mask(~is_synth, "na_no_reference")
        if m in RELEVANCY_METRICS:
            status = status.mask(rej, "na_rejected")
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

    # DeepEval judge crashes (only visible in the reason prose).
    derr = _deepeval_error_mask(df)
    flags["deepeval_error_rows"] = int(derr.sum())
    emit(f"\n  DeepEval judge crashes (reason prose matches an error pattern): "
         f"{int(derr.sum())} rows")
    if derr.any():
        emit(_indent(pd.crosstab(df.loc[derr, "source_dataset"],
                                 df.loc[derr, "variant"].astype(str)).to_string()))

    # Per-metric status breakdown. Every cell carries exactly ONE na_* reason — the
    # broadest applicable one, since the masks overwrite in order (an ID-context cell
    # on a non-synthetic no_rag row reads na_no_reference, not na_no_context). That
    # split makes the individual columns hard to read against an expected count, so
    # na_total sums them: scored + na_total + error == rows, per metric.
    piv = (cls.pivot_table(index="metric", columns="status", values="id",
                           aggfunc="count", fill_value=0)
           .reindex(CLASSIFIED_METRICS))
    na_cols = [c for c in piv.columns if str(c).startswith("na_")]
    if na_cols:
        piv.insert(len(piv.columns), "na_total", piv[na_cols].sum(axis=1))
    piv.index = [m.split(".")[-1] for m in piv.index]
    flags["status_by_metric"] = piv
    emit("\n  per-metric cell status (scored | na_* reasons | na_total | error); "
         "each cell gets ONE na reason, the broadest one that applies:")
    emit(_indent(piv.to_string()))

    # The actionable bit: which cells are true errors.
    err = cls[cls["status"] == "error"]
    flags["error_cells"] = int(len(err))
    emit(f"\n  error cells (applicable but missing): {len(err)}")
    if len(err):
        emit("    by dataset x variant:")
        emit(_indent(pd.crosstab(err["source_dataset"], err["variant"]).to_string()))
        emit("    by metric:")
        emit(_indent(err["metric"].map(lambda m: m.split(".")[-1])
                     .value_counts().to_string()))

    print("\n".join(lines))
    return flags


def prepare(df):
    """Return ``(clean_df, report)`` with RAGAS metric cells nulled on RAGAS-errored
    rows, so those rows leave RAGAS aggregates while keeping their independent
    DeepEval scores. (The metric-level masking ``analysis.drop_eval_errors``
    deliberately punts on.) They are already NaN after ``analysis.load``; this makes
    the exclusion explicit and defensive, and reports what it touched.
    """
    clean = df.copy()
    rerr = _ragas_error_mask(clean)
    ragas_cols = [c for c in ev.metric_cols(clean) if c.startswith("ragas_scores.")]
    clean.loc[rerr, ragas_cols] = np.nan
    report = {
        "n_ragas_error_rows": int(rerr.sum()),
        "by_dataset_variant": (pd.crosstab(clean.loc[rerr, "source_dataset"],
                                           clean.loc[rerr, "variant"].astype(str))
                               if rerr.any() else pd.DataFrame()),
    }
    return clean, report


# --- (2a) Metric validation: is each metric discriminative on its own? -------

def score_cols(df):
    """``ev.metric_cols`` minus the bookkeeping fields that are not scores.

    ``metric_cols`` keeps everything under ``ragas_scores.`` / ``deepeval_scores.``
    that is not prose, which sweeps in three things that are not metrics:
    ``ragas_metric_errors[.<metric>]``, the ``deepeval_*_error`` flags, and the
    ``deepeval_*_verdicts.{n_verdicts,yes,no,idk}`` tallies. In a distribution
    table the first two would only add all-NaN rows — and truncating
    ``ragas_metric_errors.ragas_answer_correctness`` to its last segment collides
    with the real metric of that name — while the verdict counts are integer
    tallies on a different scale entirely, whose min/max/median next to a 0-1
    score would be nonsense. Whether a metric errored is
    ``metric_error_report``'s question; what the verdicts were is an input to the
    score, not a score.
    """
    return [c for c in ev.metric_cols(df)
            if "metric_errors" not in c
            and "_verdicts." not in c
            and not c.endswith(("_error", "_verdicts"))]


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
    rej = ev._rejection_mask(df)

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
    scored["_abstain"] = ev._rejection_mask(scored).astype(float)
    agg = {"n": (metric, "size"), "metric_mean": (metric, "mean"),
           "abstain_rate": ("_abstain", "mean")}
    for s in signals:
        if s in scored:
            agg[s] = (s, "mean")
    return scored.groupby("_grp").agg(**agg).reindex(["worst", "rest", "best"])


# --- (4) Signal -> metric ----------------------------------------------------

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
        print("\n=== cohort (source_dataset x variant) ===")
        print(ev.describe_cohort(d).to_string())

        # (1) metric-computation sanity + prepare -----------------------------------
        print("\n=== metric-computation sanity ===")
        cls = classify_metrics(d)
        metric_error_report(d, cls, source=eval_path)

        d, prep = prepare(d)
        if prep["n_ragas_error_rows"]:
            print(f"\nprepared for analysis: on the {prep['n_ragas_error_rows']} rows whose RAGAS "
                  f"scorer raised, every ragas_scores.* value is set to NaN, so those rows no "
                  f"longer contribute to any RAGAS mean/correlation below. The rows themselves "
                  f"are kept — their DeepEval scores are independent of the RAGAS failure and "
                  f"stay in the DeepEval numbers.")
        else:
            print("\nprepared for analysis: no row-level RAGAS scorer errors to mask "
                  "(individual missing metric cells are still excluded per metric).")

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

        # Figures that need only the evaluated file. These used to sit inside the
        # cross-link block below and were silently skipped whenever no rag file was
        # found, although not one of them touches the joined frame.
        print("\n=== figures (evaluated results) ===")
        plots.save_all({
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
                print("logprob vs correctness within each variant (want positive):")
                print(ev.logprob_correlation(
                    d, metric="ragas_scores.ragas_answer_correctness").round(3).to_string())
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
