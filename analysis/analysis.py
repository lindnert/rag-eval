"""Shared primitives for the evaluated RAG results: loading, and the two summary
tables that more than one module needs.

    from analysis.analysis import load, metric_summary, compare_variants
    from analysis import plots
    df = load("results/evaluated_results_YYYYMMDD_HHMMSS.json")
    plots.metric_boxplot(df, "ragas_scores.ragas_faithfulness")

``load`` flattens the nested JSON with pandas and coerces the metric columns to
numeric (the mixed-dtype sentinel strings such as "no rag - no retrieved
contexts" and nulls become NaN, which the aggregates and plots skip). Everything
else is a plain DataFrame you can slice however you like.

This module is a LIBRARY. It has no ``__main__`` and writes no files.
--------------------------------------------------------------------
There are two analysis entry points and neither is this one:

    python -m analysis.rag_analysis  RAG_FILE              # raw pipeline signals
    python -m analysis.eval_analysis EVAL_FILE RAG_FILE    # everything evaluated

What lives here is only what the other modules build on: loading, the variant
ordering, the retrieval-score reducers (``rag_analysis.load`` calls the same three
on the raw results — one definition of "the mean of the top-k scores", or the two
frames start disagreeing), the cohort crosstab, the per-metric summary and the
paired variant comparison. The analyses that used to sit
alongside them — the ``__main__`` report, ``health_report``, ``drop_eval_errors``,
the reason mining, ``means_by``, ``decile_breakdown``, ``abstention_adjusted``,
``logprob_correlation`` — moved into ``analysis.eval_analysis``, since each is a
question about metrics or judges rather than a primitive.

Two things deliberately do NOT live here:

  - the figures, which are all in ``analysis.plots``;
  - the abstention flag. There is one detector, ``rag_analysis._abstained``,
    called by both analysis modules and by ``plots``. ``load`` used to cache a
    duplicate of it as an ``is_rejection`` column; that copy is gone, because two
    definitions of "did this row abstain" is exactly one too many.

Nothing here imports the other analysis modules, which is what keeps the import
graph acyclic while ``plots`` imports this module and ``eval_analysis`` imports
both.
"""

import json

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # scipy is optional; only compare_variants needs it
    wilcoxon = None

VARIANT_ORDER = ["no_rag", "rag", "rag_sc"]


# --- Loading -----------------------------------------------------------------

def error_cols(df):
    """Names of the evaluator's error fields: RAGAS's row-level ``ragas_error``,
    its per-metric ``ragas_metric_errors[.<metric>]``, and DeepEval's per-metric
    ``deepeval_<metric>_error``.

    They sit under the same ``*_scores.`` prefixes as the metrics but hold exception
    TEXT, so they must stay out of ``load``'s numeric coercion. They did not: every
    message was silently coerced to NaN, which is how a file that records exactly
    *why* a metric cell is missing produced a report that could only say *that* it
    was missing. ``eval_analysis.metric_error_reasons`` reads them.
    """
    return [
        c for c in df.columns
        if c.startswith(("ragas_scores.", "deepeval_scores."))
        and ("metric_errors" in c or c.endswith("_error"))
    ]


def metric_cols(df):
    """Every numeric column the evaluators wrote (RAGAS + DeepEval), excluding
    prose ``*_reason`` fields and the string-valued ``error_cols``.

    Numeric is not the same as *a score*: this still includes DeepEval's
    ``*_verdicts.{n_verdicts,yes,no,idk}`` claim tallies. Use ``score_cols`` for
    anything that summarises, ranks or plots metrics on a 0-1 scale.
    """
    err = set(error_cols(df))
    return [
        c for c in df.columns
        if c.startswith(("ragas_scores.", "deepeval_scores."))
        and not c.endswith("_reason")
        and c not in err
    ]


def score_cols(df):
    """``metric_cols`` minus the bookkeeping columns that are not scores: the
    ``deepeval_*_verdicts.{n_verdicts,yes,no,idk}`` tallies.

    The verdict counts are unbounded integers on their own scale — how many claims
    the judge extracted and how it voted on them — so they are an INPUT to a score,
    not a score. Mixed into a 0-1 summary they are nonsense in a specific way that
    looks plausible: a "median" of 3 claims sits in a column of 0-1 medians, and a
    row with one verdict reads as 100% "pinned at the 1.0 rail" in
    ``plots.metric_rail_plot``, which is where twelve of these first showed up
    masquerading as degenerate metrics.

    This is the default for ``metric_summary`` and every caller that asks the
    question "how does this metric behave"; ``metric_cols`` remains for callers
    that genuinely mean every numeric evaluator column.
    """
    return [c for c in metric_cols(df)
            if "_verdicts." not in c and not c.endswith("_verdicts")]


def best_score(scores):
    """Top (max) retrieval score of a row, or NaN when there are none."""
    if isinstance(scores, (list, tuple)) and len(scores):
        return max(scores)
    return float("nan")


def avg_score(scores):
    """Mean retrieval score over a row's top-k chunks, or NaN when there are none.
    The whole-context view next to ``best_score``'s single-chunk one: a context can
    lead with one strong hit and pad the rest, and only this notices."""
    if isinstance(scores, (list, tuple)) and len(scores):
        return sum(scores) / len(scores)
    return float("nan")


def spread_score(scores):
    """Top-k spread (max - min) of a row's retrieval scores; this is the quantity
    the rag_sc 'spread>threshold' correction trigger keys on. NaN if <2 scores."""
    if isinstance(scores, (list, tuple)) and len(scores) >= 2:
        return max(scores) - min(scores)
    return float("nan")


def load(path):
    """Flatten evaluated results JSON to a tidy DataFrame with numeric metrics.

    Nested dicts become dot-named columns (e.g. ``ragas_scores.ragas_faithfulness``,
    ``gen_logprob_stats.mean``). Metric columns are coerced to float so the
    sentinel strings and nulls turn into NaN. ``retrieval_best`` /
    ``retrieval_average`` are derived from ``retrieval_scores`` with the same
    reducers ``rag_analysis.load`` uses, so the column means one thing on both sides.

    Abstentions are not flagged here — call ``rag_analysis._abstained(df)``, the
    single detector, on the frame this returns.

    The evaluator's error fields (``error_cols``) are deliberately NOT coerced:
    they carry the exception text that explains every missing metric cell, and
    ``to_numeric`` would turn each message into the same NaN as a cell that simply
    was not applicable.
    """
    with open(path, "r", encoding="utf-8") as f:
        df = pd.json_normalize(json.load(f))

    cols = metric_cols(df)
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    if "variant" in df:
        df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER,
                                       ordered=True)
    if "retrieval_scores" in df:
        df["retrieval_best"] = df["retrieval_scores"].apply(best_score)
        df["retrieval_average"] = df["retrieval_scores"].apply(avg_score)

    return df


def _order(df):
    """The variants present, in canonical order — the ``order`` / ``hue_order``
    every grouped plot and table sorts by."""
    return [v for v in VARIANT_ORDER if v in df["variant"].unique()]


# --- Cohort ------------------------------------------------------------------

def describe_cohort(df):
    """Counts of what you're analysing: ``source_dataset`` x ``variant`` crosstab
    with row/column totals, so 'how many datapoints of each type' is one table.

    Run it before and after ``eval_analysis.drop_eval_errors`` to see exactly what
    the exclusion cost. Falls back to a single-axis count if either column is
    missing.
    """
    if "source_dataset" in df and "variant" in df:
        return pd.crosstab(df["source_dataset"], df["variant"].astype("object"),
                           margins=True, margins_name="All")
    axis = "source_dataset" if "source_dataset" in df else "variant"
    if axis in df:
        return df[axis].value_counts().rename_axis(axis).to_frame("n")
    return pd.DataFrame({"n": [len(df)]}, index=["all"])


# --- Per-metric summary ------------------------------------------------------

def metric_summary(df, metrics=None):
    """Per-metric overview built to expose *broken* metrics, not just central
    tendency.

    One row per metric with:
      - ``n`` / ``coverage``: how many rows actually carry a numeric value.
        A near-zero coverage means the metric was (almost) never computed —
        the evaluation step for it is broken or was skipped.
      - ``mean`` / ``std`` / ``min`` / ``max``: the usual location/spread.
      - ``q25`` / ``median`` / ``q75``: the quantiles. They are what separate a
        *discriminative* metric from a skewed one that the mean flatters: a
        median of 1.0 with a mean of 0.8 means the metric says "perfect" for
        most queries and only a tail carries any signal. ``q75 - q25`` (the IQR)
        collapsing to 0 while min < max is exactly that pile-up.
      - ``n_unique``: distinct scored values. A metric that only ever emits a
        handful of levels cannot rank ~1000 queries however good its mean looks.
      - ``frac_zero`` / ``frac_one``: share of scored rows pinned at the 0.0 /
        1.0 rails. A metric that is ~all-0 or ~all-1 (std ~ 0) is degenerate —
        often a scorer default firing on every row rather than a real signal.

    ``eval_analysis.metric_distribution`` wraps this to add the per-group cut;
    ``plots.metric_rail_plot`` draws it. Both leave ``metrics`` unset most of the
    time, so the default has to be ``score_cols``, not ``metric_cols`` — the
    verdict tallies are not on a 0-1 scale and every column here assumes they are.
    """
    metrics = metrics or score_cols(df)
    n_total = len(df)
    rows = {}
    for m in metrics:
        s = pd.to_numeric(df[m], errors="coerce")
        n = int(s.notna().sum())
        rows[m] = {
            "n": n,
            "coverage": round(n / n_total, 3) if n_total else float("nan"),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "q25": s.quantile(0.25) if n else float("nan"),
            "median": s.median(),
            "q75": s.quantile(0.75) if n else float("nan"),
            "max": s.max(),
            "n_unique": int(s.nunique()),
            "frac_zero": round((s == 0).sum() / n, 3) if n else float("nan"),
            "frac_one": round((s == 1).sum() / n, 3) if n else float("nan"),
        }
    return pd.DataFrame(rows).T


def metric_summary_by(df, by, metrics=None):
    """``metric_summary`` computed once per ``by`` group and stacked into one frame
    indexed by the group keys plus ``metric``.

    The grouped cut is the one that decides whether a metric is usable: a metric can
    spread over NGQA and collapse to 1.0 on MMLU, and faithfulness cannot be scored
    on ``no_rag`` at all — pooled, all of that averages into one innocuous-looking
    row. Every group is summarised over the SAME ``metrics`` list (the caller's, or
    ``score_cols`` of the whole frame), so a metric absent from one group still gets
    a row there with ``n = 0`` rather than silently dropping out and leaving two
    groups that cannot be compared row by row.

    Group keys are cast to ``str``: they are labels here, and a Categorical index
    level would carry the full category list into every downstream join and reindex.

    Lives here rather than in ``eval_analysis`` because two callers need it and each
    had started writing its own: ``eval_analysis.metric_distribution`` prints it and
    ``plots.metric_rail_grid`` draws it, and a figure whose numbers are computed by
    different code than the table it illustrates is a figure you cannot trust.
    """
    keys = [by] if isinstance(by, str) else list(by)
    metrics = metrics or score_cols(df)
    frames = {}
    for g, sub in df.groupby(keys, observed=True):
        if not len(sub):
            continue
        frames[g if isinstance(g, tuple) else (g,)] = metric_summary(sub, metrics)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames.values(), keys=frames.keys(),
                    names=keys + ["metric"]).reset_index()
    for k in keys:
        out[k] = out[k].astype(str)
    return out.set_index(keys + ["metric"])


# --- Paired 'A beats B' variant comparison -----------------------------------

def _bootstrap_ci(diffs, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of ``diffs``."""
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def compare_variants(df, metric, a="rag", b="no_rag", by=None, n_boot=2000):
    """Paired 'does variant ``a`` beat variant ``b``' on one metric, matched per id.

    The core results-chapter test. Pairs each question's ``a`` and ``b`` score
    (pivot on ``id``, keep ids present in both), then reports on the paired
    differences ``a - b``:
      - ``n_pairs`` (and ``n_unpaired`` dropped for lacking a partner);
      - ``mean_a`` / ``mean_b`` / ``mean_diff`` / ``median_diff``;
      - ``frac_a_gt_b``: share of questions where a strictly beat b;
      - ``wilcoxon_p``: two-sided Wilcoxon signed-rank p-value (non-parametric,
        the right test for these bounded, non-normal scores);
      - ``rank_biserial``: matched-pairs effect size in [-1, 1] (magnitude, since a
        p-value alone doesn't say *how much*); positive favours ``a``;
      - ``ci_low`` / ``ci_high``: bootstrap 95% CI on the mean difference.

    ``by="source_dataset"`` adds one row per dataset beneath the ``overall`` row —
    the per-dataset twin that often carries the interesting finding. Ties (zero
    differences) are dropped from the signed-rank test per Wilcoxon convention;
    an all-tie or empty pairing yields NaN stats rather than raising.

    Read it after ``eval_analysis.metric_distribution``: a significant difference
    on a metric that ranks nothing is not a finding. ``plots.variant_effect_forest``
    draws a panel of these.
    """
    def _one(sub):
        wide = sub.pivot_table(index="id", columns="variant", values=metric,
                               observed=True, aggfunc="mean")
        have = [v for v in (a, b) if v in wide.columns]
        if len(have) < 2:
            return pd.Series({"n_pairs": 0, "n_unpaired": len(wide),
                              "note": f"missing variant(s): {set((a, b)) - set(have)}"})
        pair = wide[[a, b]].dropna()
        n = len(pair)
        diffs = (pair[a] - pair[b]).to_numpy()
        nonzero = diffs[diffs != 0]
        row = {
            "n_pairs": n,
            "n_unpaired": int(len(wide) - n),
            "mean_a": pair[a].mean(),
            "mean_b": pair[b].mean(),
            "mean_diff": diffs.mean() if n else float("nan"),
            "median_diff": float(np.median(diffs)) if n else float("nan"),
            "frac_a_gt_b": float((diffs > 0).mean()) if n else float("nan"),
        }
        if wilcoxon is None or len(nonzero) == 0:
            row["wilcoxon_p"] = float("nan")
            row["rank_biserial"] = 0.0 if len(nonzero) == 0 else float("nan")
        else:
            row["wilcoxon_p"] = float(wilcoxon(nonzero)[1])
            ranks = pd.Series(np.abs(nonzero)).rank().to_numpy()
            t_plus = ranks[nonzero > 0].sum()
            t_minus = ranks[nonzero < 0].sum()
            total = t_plus + t_minus
            row["rank_biserial"] = float((t_plus - t_minus) / total) if total else 0.0
        lo, hi = _bootstrap_ci(diffs, n_boot=n_boot)
        row["ci_low"], row["ci_high"] = lo, hi
        return pd.Series(row)

    rows = {"overall": _one(df)}
    if by is not None and by in df:
        for g, sub in df.groupby(by, observed=True):
            rows[str(g)] = _one(sub)
    return pd.DataFrame(rows).T
