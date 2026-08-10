"""Lightweight loading and summary tables for evaluated RAG results.

    from analysis.analysis import load, metric_summary, compare_variants
    from analysis import plots
    df = load("evaluated_results_YYYYMMDD.json")
    plots.metric_boxplot(df, "ragas_scores.ragas_faithfulness")

`load` flattens the nested JSON with pandas and coerces the metric columns to
numeric (the mixed-dtype sentinel strings such as "no rag - no retrieved
contexts" and nulls become NaN, which plotting skips). Everything else is a
plain DataFrame you can slice and plot however you like.

The figures that used to live here are in ``analysis.plots`` — one module for
every figure the analysis produces, so the three analysis modules keep only the
tables and statistics.

Running from the console
------------------------
Run as a module from the repo root (the ``analysis.`` / ``common.`` imports need
the package on the path, so ``python analysis/analysis.py`` will NOT work):

    python -m analysis.analysis                       # uses DEFAULT_PATH
    python -m analysis.analysis results/evaluated_results_20260701_112445.json

This runs the whole __main__ report: cohort crosstab, evaluator-failure
exclusion, per-metric summary, paired Wilcoxon comparisons, reason mining,
logprob correlation, deciles, and the PNGs. Every artifact goes to
``analysis/out/<results-stem>/`` (tables/, figures/, reports/), named for the
results file it came from — see ``analysis.paths``.
"""

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from analysis import paths

try:
    from scipy.stats import wilcoxon
except ImportError:  # scipy is optional; only compare_variants needs it
    wilcoxon = None

# Canonical abstention strings (per language). We import from the dependency-free
# source so the analysis module doesn't pull in torch/transformers. REJECTION_ANSWERS
# holds every language variant; the single REJECTION_ANSWER depends on RAG_LANG, which
# defaults to "en" — too narrow to match a multilingual run's German abstentions, so
# `load` prefers the authoritative `rejected` field and only falls back to matching
# against *all* language strings.
from common.constants import REJECTION_ANSWERS

VARIANT_ORDER = ["no_rag", "rag", "rag_sc"]

# Default results file used when the script is run without a path argument.
DEFAULT_PATH = "results/evaluated_results_20260701_112445.json"


def metric_cols(df):
    """Names of the numeric score columns (RAGAS + DeepEval), excluding prose
    ``*_reason`` fields and the ``ragas_error`` string."""
    return [
        c for c in df.columns
        if c.startswith(("ragas_scores.", "deepeval_scores."))
        and not c.endswith("_reason")
        and c != "ragas_scores.ragas_error"
    ]


def load(path):
    """Flatten evaluated results JSON to a tidy DataFrame with numeric metrics.

    Nested dicts become dot-named columns (e.g. ``ragas_scores.ragas_faithfulness``,
    ``gen_logprob_stats.mean``). Metric columns are coerced to float so sentinel
    strings / nulls turn into NaN. A few convenience columns are added.
    """
    with open(path, "r", encoding="utf-8") as f:
        df = pd.json_normalize(json.load(f))

    cols = metric_cols(df)
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    if "variant" in df:
        df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    if "retrieval_scores" in df:
        df["retrieval_best"] = df["retrieval_scores"].apply(lambda s: max(s) if s else None)

    # Abstention flag. Prefer the pipeline's own language-agnostic `rejected`
    # boolean; only fall back to string-matching the answer text — and then
    # against *every* language's rejection string, since a run is multilingual
    # (matching one language's string would miss the others).
    if "rejected" in df:
        df["is_rejection"] = df["rejected"].fillna(False).astype(bool)
    elif "answer" in df:
        rej_strings = {s.strip() for s in REJECTION_ANSWERS.values()}
        df["is_rejection"] = df["answer"].fillna("").str.strip().isin(rej_strings)

    return df


def _order(df):
    return [v for v in VARIANT_ORDER if v in df["variant"].unique()]


# --- Tabular summaries: means, per-dataset means, decile drill-down ----------

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
    """
    metrics = metrics or metric_cols(df)
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


def means_by(df, by="source_dataset", metrics=None):
    """Mean of every metric grouped by a column (default ``source_dataset``).

    Rows = groups, columns = metrics. Also appends an ``n`` column (rows per
    group) so a low group mean built on a handful of queries is obvious. Pass
    ``by="variant"`` for the per-variant table, or a list for a crosstab.
    """
    metrics = metrics or metric_cols(df)
    sub = df.copy()
    sub[metrics] = sub[metrics].apply(pd.to_numeric, errors="coerce")
    g = sub.groupby(by, observed=True)[metrics].mean()
    g.insert(0, "n", sub.groupby(by, observed=True).size())
    return g


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
    ``slopegraph``); pass a single-variant slice here (``df[df.variant=="rag_sc"]``)
    to read one variant's distribution cleanly, or the full frame to find questions
    that score badly across *every* variant (intrinsically hard items / bad golds).
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
    out = sub.groupby("decile").agg(
        n=("value", "size"),
        mean=("value", "mean"),
        min=("value", "min"),
        max=("value", "max"),
        ids=("_label", lambda x: list(x)),
    )
    return out


def health_report(df, source=None):
    """Quick 'is something broken?' checks. Prints; ``source`` (the results
    path/name it ran on) only adds the provenance header lines.

    Surfaces the failure modes that silently poison aggregates: generation
    errors baked into the answer text, RAGAS scorer errors, metrics that never
    produced a value, degenerate (constant) metrics, and faithfulness scored on
    abstentions (ill-defined — an abstention makes no claim to be faithful to).

    Returns the flags dict. The printed lines are saved along with the rest of
    the run's console output by ``analysis.paths.capture``, which the __main__
    block wraps around everything — this function no longer writes its own file.
    """
    flags = {}
    lines = []

    def emit(s):
        lines.append(s)

    n = len(df)
    emit(f"health_report: {n} rows")
    if source is not None:
        emit(f"  source: {source}")
        emit(f"  generated: {datetime.now().isoformat(timespec='seconds')}")

    if "answer" in df:
        gen_err = df["answer"].fillna("").str.contains(r"\[LLAMACPP", regex=True)
        flags["answer_generation_errors"] = int(gen_err.sum())
        emit(f"  answer generation errors ([LLAMACPP ...]): {int(gen_err.sum())}")

    err_col = "ragas_scores.ragas_error"
    if err_col in df:
        ragas_err = df[err_col].notna() & df[err_col].astype(str).str.strip().ne("")
        flags["ragas_errors"] = int(ragas_err.sum())
        emit(f"  ragas scorer errors: {int(ragas_err.sum())}")

    empty, degenerate = [], []
    for m in metric_cols(df):
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

    reject = df["rejected"] if "rejected" in df else df.get("is_rejection")
    faith_cols = [c for c in metric_cols(df) if "faithful" in c]
    if reject is not None and faith_cols:
        for c in faith_cols:
            scored = pd.to_numeric(df.loc[reject.fillna(False), c], errors="coerce").notna().sum()
            if scored:
                emit(f"  [CHECK] {c} scored on {int(scored)} abstained rows "
                     f"(faithfulness is ill-defined on abstentions)")
        flags["faithfulness_on_rejections"] = {
            c: int(pd.to_numeric(df.loc[reject.fillna(False), c], errors="coerce").notna().sum())
            for c in faith_cols
        }

    print("\n".join(lines))
    return flags


def _rejection_mask(df):
    """Boolean 'this row abstained' series, preferring the pipeline's own
    ``rejected`` flag and falling back to the string-match ``is_rejection``."""
    if "rejected" in df:
        return df["rejected"].fillna(False).astype(bool)
    if "is_rejection" in df:
        return df["is_rejection"].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


# --- (1) Reason-string mining -----------------------------------------------

# Phrases that, when they appear in a judge's *_reason prose or in ragas_error,
# usually mean the score is a scorer/judge failure rather than a real rating of
# the answer. Deliberately broad; treat the output as leads to eyeball, not a
# verdict. Word boundaries keep "error" from firing inside unrelated words.
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
    phrase — (id, variant, column, snippet) — for eyeballing what broke."""
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


# --- (3) Logprob correlation with answer quality ----------------------------

def logprob_correlation(df, metric="ragas_scores.ragas_answer_correctness",
                        logprob_col="gen_logprob_stats.mean", by="variant"):
    """Correlate generation confidence (mean token logprob) with a quality
    metric, overall and per group.

    A healthy pipeline shows a *positive* correlation: more-confident
    generations score better. A flat or negative correlation means the logprob
    signal isn't tracking quality — the self-correction trigger keyed on it is
    mis-calibrated. Reports both Pearson (linear) and Spearman (monotone) on the
    rows where both values exist.
    """
    def _corr(sub):
        x = pd.to_numeric(sub[logprob_col], errors="coerce")
        y = pd.to_numeric(sub[metric], errors="coerce")
        m = x.notna() & y.notna()
        n = int(m.sum())
        return pd.Series({
            "n": n,
            "pearson": x[m].corr(y[m]) if n > 2 else float("nan"),
            "spearman": x[m].corr(y[m], method="spearman") if n > 2 else float("nan"),
        })

    rows = {"overall": _corr(df)}
    if by in df:
        for g, sub in df.groupby(by, observed=True):
            rows[str(g)] = _corr(sub)
    return pd.DataFrame(rows).T


# The scatter of this correlation is ``plots.logprob_scatter``.


# --- (5) Abstention-adjusted metrics ----------------------------------------

def abstention_adjusted(df, metrics=None, by=None):
    """Every metric reported twice: over all rows vs over answered rows only.

    Splits a low headline mean into its two causes — genuinely poor answers vs a
    high abstention rate. ``mean_all`` counts abstentions (via the pipeline's
    ``rejected`` flag) as scored; ``mean_answered`` excludes them. A large
    positive ``delta`` (answered ≫ all) means the metric is mostly measuring how
    often the system abstained, not answer quality. With ``by`` set, returns the
    answered-only means per group.
    """
    metrics = metrics or metric_cols(df)
    answered = ~_rejection_mask(df)

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


# --- Cohort description & evaluator-failure exclusion ------------------------

def describe_cohort(df):
    """Counts of what you're analysing: ``source_dataset`` x ``variant`` crosstab
    with row/column totals, so 'how many datapoints of each type' is one table.

    Run it before and after ``drop_eval_errors`` to see exactly what the exclusion
    cost. Falls back to a single-axis count if either column is missing.
    """
    if "source_dataset" in df and "variant" in df:
        ct = pd.crosstab(df["source_dataset"],
                         df["variant"].astype("object"),
                         margins=True, margins_name="All")
        return ct
    axis = "source_dataset" if "source_dataset" in df else "variant"
    if axis in df:
        return df[axis].value_counts().rename_axis(axis).to_frame("n")
    return pd.DataFrame({"n": [len(df)]}, index=["all"])


# Narrow, technical-failure patterns for DECIDING to drop a row — the evaluator
# LLM crashed, not "the answer was weak". Deliberately much stricter than
# FAILURE_PATTERNS (which is broad on purpose, for *eyeballing* leads): a
# DeepEval reason that legitimately says "no relevant context" is a real low
# score, not a failure, so it must NOT match here.
EVAL_ERROR_PATTERNS = [
    r"exception", r"traceback", r"failed to parse", r"parse error",
    r"outputparser", r"json ?decode", r"could ?n[o']t parse",
    r"rate ?limit", r"timed ?out", r"\btimeout\b", r"api error",
    r"error (?:generating|calling|during|while)", r"\[llamacpp",
]


def drop_eval_errors(df, mask_metric_level=False):
    """Exclude rows where a *technical failure* — not a low score — produced the
    metric, and report how many and of what type were dropped.

    Three failure signals:
      - generation failure: ``answer`` contains ``[LLAMACPP ...]`` (no real
        answer exists, so the whole row is meaningless -> always dropped);
      - RAGAS failure: ``ragas_scores.ragas_error`` is non-empty (the RAGAS judge
        raised, e.g. an OutputParserException);
      - DeepEval failure: any ``deepeval_scores.*_reason`` matches the narrow
        ``EVAL_ERROR_PATTERNS`` (DeepEval has no error field, so the judge crash
        only shows up in its prose).

    Returns ``(clean_df, report)``. ``report`` is a dict with: ``n_before`` /
    ``n_after`` / ``n_dropped``; ``by_type`` (rows flagged per failure signal —
    can overlap); ``by_dataset`` and ``by_variant`` (dropped counts); and
    ``by_dataset_variant`` (the crosstab of what was dropped).

    ``mask_metric_level=False`` (default) drops the whole row on any failure — a
    clean uniform cohort for paired tables. It is not implemented to mask per
    metric yet; the flag is a placeholder documenting the alternative (mask only
    the failed evaluator's columns, keeping the other evaluator's valid scores).
    """
    if mask_metric_level:
        raise NotImplementedError(
            "metric-level masking not implemented; call with mask_metric_level=False "
            "(whole-row drop). See docstring for the trade-off.")

    idx = df.index
    gen = (df["answer"].fillna("").str.contains(r"\[LLAMACPP", regex=True)
           if "answer" in df else pd.Series(False, index=idx))

    err_col = "ragas_scores.ragas_error"
    ragas = (df[err_col].notna() & df[err_col].astype(str).str.strip().ne("")
             if err_col in df else pd.Series(False, index=idx))

    rx = re.compile("|".join(EVAL_ERROR_PATTERNS), re.IGNORECASE)
    de_cols = [c for c in df.columns
               if c.startswith("deepeval_scores.") and c.endswith("_reason")]
    deepeval = pd.Series(False, index=idx)
    for c in de_cols:
        txt = df[c].astype("string").fillna("")
        deepeval = deepeval | (txt.str.strip().ne("") & txt.str.contains(rx))

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
        "by_dataset_variant": (describe_cohort(dropped) if len(dropped)
                               else pd.DataFrame()),
    }
    return df[~bad].copy(), report


# --- (b) Paired 'A beats B' variant comparison -------------------------------

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
            row["wilcoxon_p"] = float("nan") if wilcoxon is not None else float("nan")
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


if __name__ == "__main__":
    import sys

    # Imported here, not at module level: plots imports this module back, and a
    # top-level import would make that a cycle.
    from analysis import plots

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    d = load(path)

    # Everything printed below is collected and saved to this run's
    # reports/ folder, then echoed to the console.
    with paths.capture(path, "analysis_report"):
        print(f"{path}: {len(d)} rows, {d['id'].nunique()} ids, "
              f"variants={list(d['variant'].cat.categories)}")
        print(f"writing every artifact to {paths.rel(paths.run_dir(path))}")
        print(d.groupby("variant", observed=True)["is_rejection"].mean().round(3).to_string())

        # --- Cohort + evaluator-failure exclusion ----------------------------
        print("\n=== cohort before exclusion (source_dataset x variant) ===")
        print(describe_cohort(d).to_string())

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
        print(describe_cohort(d).to_string())

        # --- Tabular summaries: print to console and save as CSV -------------
        print("\n=== per-metric summary (coverage / mean / degeneracy) ===")
        summary = metric_summary(d)
        print(summary.round(3).to_string())
        summary.to_csv(paths.table(path, "metric_summary"))

        for by in ("source_dataset", "variant"):
            print(f"\n=== metric means by {by} ===")
            tbl = means_by(d, by=by)
            print(tbl.round(3).to_string())
            tbl.to_csv(paths.table(path, f"means_by_{by}"))

        print("\n=== broken-signal checks ===")
        health_report(d, source=path)

        # --- Paired 'A beats B' comparisons (the results-chapter spine) ------
        print("\n=== paired variant comparisons (Wilcoxon signed-rank) ===")
        for metric in ("ragas_scores.ragas_answer_correctness",
                       "deepeval_scores.deepeval_relevance"):
            for a, b in (("rag", "no_rag"), ("rag_sc", "rag")):
                cmp = compare_variants(d, metric, a=a, b=b)
                print(f"\n{metric}: {a} vs {b}")
                print(cmp.round(3).to_string())
                cmp.to_csv(paths.table(
                    path, f"compare_{a}_vs_{b}_{metric.split('.')[-1]}"))

        print("\n=== reason-string mining (failure-phrase hit rate per judge) ===")
        reasons = mine_reasons(d)
        print(reasons.to_string())
        reasons.to_csv(paths.table(path, "reason_mining"))

        print("\n=== logprob vs answer correctness (want positive) ===")
        corr = logprob_correlation(d)
        print(corr.round(3).to_string())
        corr.to_csv(paths.table(path, "logprob_correlation"))

        print("\n=== abstention-adjusted metrics (all rows vs answered only) ===")
        adj = abstention_adjusted(d)
        print(adj.round(3).to_string())
        adj.to_csv(paths.table(path, "abstention_adjusted"))

        # Decile drill-down: worst-scoring tenth of queries per metric, with IDs.
        print("\n=== decile drill-down (decile 1 = worst) ===")
        for metric in ("ragas_scores.ragas_answer_correctness",
                       "deepeval_scores.deepeval_relevance"):
            dec = decile_breakdown(d, metric)
            if dec.empty:
                print(f"{metric}: no scored rows")
                continue
            print(f"\n{metric}:")
            print(dec[["n", "mean", "min", "max"]].round(3).to_string())
            worst = dec.iloc[0]
            print(f"  worst-decile ids: {worst['ids']}")
            dec.to_csv(paths.table(path, f"deciles_{metric.split('.')[-1]}"))

        print("\n=== figures ===")
        plots.save_all({
            "boxplot_faithfulness": lambda ax: plots.metric_boxplot(
                d, "ragas_scores.ragas_faithfulness", ax=ax),
            "coverage": lambda ax: plots.coverage_violin(d, ax=ax),
            "rejection": lambda ax: plots.rejection_bars(d, ax=ax),
            "slope_relevance": lambda ax: plots.slopegraph(
                d, "deepeval_scores.deepeval_relevance", ax=ax),
            "ragas_vs_deepeval_faithfulness": lambda ax: plots.ragas_vs_deepeval(
                d, "ragas_scores.ragas_faithfulness",
                "deepeval_scores.deepeval_faithfulness", ax=ax),
            "logprob_vs_correctness": lambda ax: plots.logprob_scatter(d, ax=ax),
        }, path)
