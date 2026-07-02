"""Lightweight loading and plotting for evaluated RAG results.

    from analysis.analysis import load, coverage_violin, metric_boxplot, \
        rejection_bars, ragas_vs_deepeval
    df = load("evaluated_results_YYYYMMDD.json")
    metric_boxplot(df, "ragas_scores.ragas_faithfulness")

`load` flattens the nested JSON with pandas and coerces the metric columns to
numeric (the mixed-dtype sentinel strings such as "no rag - no retrieved
contexts" and nulls become NaN, which plotting skips). Everything else is a
plain DataFrame you can slice and plot however you like.
"""

import json
import re

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # seaborn is optional; functions fall back to matplotlib
    sns = None

# Canonical abstention string. rag.utils re-exports this, but we import from the
# dependency-free source so the analysis module doesn't pull in torch/transformers.
from common.constants import REJECTION_ANSWER

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
    if "answer" in df:
        df["is_rejection"] = df["answer"].fillna("").str.strip().eq(REJECTION_ANSWER)

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
            "max": s.max(),
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
    reflects only scored queries. Filter ``df`` first (e.g. one variant) when
    the metric only makes sense within a variant.
    """
    s = pd.to_numeric(df[metric], errors="coerce")
    sub = df.loc[s.notna(), [id_col]].copy()
    sub["value"] = s[s.notna()].to_numpy()
    if sub.empty:
        return pd.DataFrame(columns=["n", "mean", "min", "max", "ids"])
    ranks = sub["value"].rank(method="first")
    sub["decile"] = pd.qcut(ranks, min(n_bins, len(sub)), labels=False) + 1
    out = sub.groupby("decile").agg(
        n=("value", "size"),
        mean=("value", "mean"),
        min=("value", "min"),
        max=("value", "max"),
        ids=(id_col, lambda x: list(x)),
    )
    return out


def health_report(df):
    """Print quick 'is something broken?' checks and return a flags dict.

    Surfaces the failure modes that silently poison aggregates: generation
    errors baked into the answer text, RAGAS scorer errors, metrics that never
    produced a value, degenerate (constant) metrics, and faithfulness scored on
    abstentions (ill-defined — an abstention makes no claim to be faithful to).
    """
    flags = {}
    n = len(df)
    print(f"health_report: {n} rows")

    if "answer" in df:
        gen_err = df["answer"].fillna("").str.contains(r"\[LLAMACPP", regex=True)
        flags["answer_generation_errors"] = int(gen_err.sum())
        print(f"  answer generation errors ([LLAMACPP ...]): {int(gen_err.sum())}")

    err_col = "ragas_scores.ragas_error"
    if err_col in df:
        ragas_err = df[err_col].notna() & df[err_col].astype(str).str.strip().ne("")
        flags["ragas_errors"] = int(ragas_err.sum())
        print(f"  ragas scorer errors: {int(ragas_err.sum())}")

    empty, degenerate = [], []
    for m in metric_cols(df):
        s = pd.to_numeric(df[m], errors="coerce")
        if s.notna().sum() == 0:
            empty.append(m)
        elif s.nunique(dropna=True) == 1:
            degenerate.append((m, s.dropna().iloc[0]))
    for m in empty:
        print(f"  [BROKEN] {m}: no numeric values at all")
    for m, v in degenerate:
        print(f"  [DEGENERATE] {m}: constant = {v}")
    flags["empty_metrics"] = empty
    flags["degenerate_metrics"] = degenerate

    reject = df["rejected"] if "rejected" in df else df.get("is_rejection")
    faith_cols = [c for c in metric_cols(df) if "faithful" in c]
    if reject is not None and faith_cols:
        for c in faith_cols:
            scored = pd.to_numeric(df.loc[reject.fillna(False), c], errors="coerce").notna().sum()
            if scored:
                print(f"  [CHECK] {c} scored on {int(scored)} abstained rows "
                      f"(faithfulness is ill-defined on abstentions)")
        flags["faithfulness_on_rejections"] = {
            c: int(pd.to_numeric(df.loc[reject.fillna(False), c], errors="coerce").notna().sum())
            for c in faith_cols
        }
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


def logprob_scatter(df, metric="ragas_scores.ragas_answer_correctness",
                    logprob_col="gen_logprob_stats.mean", ax=None):
    """Scatter of mean token logprob vs a quality metric, coloured by variant."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    x = pd.to_numeric(df[logprob_col], errors="coerce")
    y = pd.to_numeric(df[metric], errors="coerce")
    m = x.notna() & y.notna()
    if sns is not None:
        sns.scatterplot(x=x[m], y=y[m], hue=df.loc[m, "variant"],
                        hue_order=_order(df), ax=ax)
    else:
        ax.scatter(x[m], y[m], alpha=0.6)
    ax.set_xlabel(logprob_col)
    ax.set_ylabel(metric)
    ax.set_title("Generation confidence vs answer quality")
    return ax


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


def metric_boxplot(df, metric, by="variant", ax=None):
    """Distribution of a metric column across variants (or any category)."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    order = _order(df) if by == "variant" else None
    if sns is not None:
        sns.boxplot(data=df, x=by, y=metric, order=order, ax=ax)
    else:
        groups = order or list(df[by].dropna().unique())
        ax.boxplot([df.loc[df[by] == g, metric].dropna() for g in groups], labels=groups)
        ax.set_xlabel(by)
        ax.set_ylabel(metric)
    ax.set_title(f"{metric} by {by}")
    return ax


def coverage_violin(df, ax=None):
    """Best hybrid retrieval score per record, split by dataset (coverage proxy).

    NOTE: retrieval_scores are hybrid dense+BM25 ranking scores, not pure cosine,
    so read this as relative KB coverage per dataset rather than absolute similarity.
    """
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    sub = df.dropna(subset=["retrieval_best"])
    if sns is not None:
        sns.violinplot(data=sub, x="source_dataset", y="retrieval_best", cut=0, ax=ax)
    else:
        cats = list(sub["source_dataset"].unique())
        ax.violinplot([sub.loc[sub["source_dataset"] == c, "retrieval_best"] for c in cats])
        ax.set_xticks(range(1, len(cats) + 1))
        ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_title("Best retrieval score by dataset (hybrid, KB-coverage proxy)")
    ax.tick_params(axis="x", rotation=30)
    return ax


def rejection_bars(df, ax=None):
    """Fraction of answers that are the abstention string, per variant."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    rate = df.groupby("variant", observed=True)["is_rejection"].mean().reindex(_order(df))
    rate.plot.bar(ax=ax)
    ax.set_ylabel("rejection rate")
    ax.set_ylim(0, 1)
    ax.set_title("Abstention rate by variant")
    return ax


def slopegraph(df, metric, ax=None, sample=None):
    """Per-id lines across no_rag -> rag -> rag_sc for one metric."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    wide = df.pivot_table(index="id", columns="variant", values=metric, observed=True)
    wide = wide.reindex(columns=_order(df)).dropna(how="all")
    if sample:
        wide = wide.sample(min(sample, len(wide)), random_state=0)
    for _, row in wide.iterrows():
        ax.plot(range(len(wide.columns)), row.values, color="gray", alpha=0.3, marker="o")
    ax.plot(range(len(wide.columns)), wide.mean().values, color="crimson", marker="o", lw=2, label="mean")
    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels(list(wide.columns))
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per id across variants")
    ax.legend()
    return ax


def ragas_vs_deepeval(df, ragas_col, deepeval_col, ax=None):
    """Scatter of a RAGAS metric against a comparable DeepEval metric.

    Faithfulness is only meaningful on answered rows (an abstention makes no
    domain claim), so those are dropped when comparing faithfulness columns.
    """
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    sub = df
    if "faithful" in ragas_col and "is_rejection" in df:
        sub = df[~df["is_rejection"]]
    sub = sub.dropna(subset=[ragas_col, deepeval_col])
    if sns is not None:
        sns.scatterplot(data=sub, x=ragas_col, y=deepeval_col, hue="variant",
                        hue_order=_order(df), ax=ax)
    else:
        ax.scatter(sub[ragas_col], sub[deepeval_col], alpha=0.6)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)  # y = x reference
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("RAGAS vs DeepEval")
    return ax


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    d = load(path)
    print(f"{path}: {len(d)} rows, {d['id'].nunique()} ids, "
          f"variants={list(d['variant'].cat.categories)}")
    print(d.groupby("variant", observed=True)["is_rejection"].mean().round(3).to_string())

    # --- Tabular summaries: print to console and save as CSV -----------------
    print("\n=== per-metric summary (coverage / mean / degeneracy) ===")
    summary = metric_summary(d)
    print(summary.round(3).to_string())
    summary.to_csv("analysis/metric_summary.csv")

    for by in ("source_dataset", "variant"):
        print(f"\n=== metric means by {by} ===")
        table = means_by(d, by=by)
        print(table.round(3).to_string())
        table.to_csv(f"analysis/means_by_{by}.csv")

    print("\n=== broken-signal checks ===")
    health_report(d)

    print("\n=== reason-string mining (failure-phrase hit rate per judge) ===")
    reasons = mine_reasons(d)
    print(reasons.to_string())
    reasons.to_csv("analysis/reason_mining.csv")

    print("\n=== logprob vs answer correctness (want positive) ===")
    corr = logprob_correlation(d)
    print(corr.round(3).to_string())
    corr.to_csv("analysis/logprob_correlation.csv")

    print("\n=== abstention-adjusted metrics (all rows vs answered only) ===")
    adj = abstention_adjusted(d)
    print(adj.round(3).to_string())
    adj.to_csv("analysis/abstention_adjusted.csv")

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
        dec.to_csv(f"analysis/deciles_{metric.split('.')[-1]}.csv")

    # Render the standard plots next to this script.
    figs = {
        "boxplot_faithfulness": lambda ax: metric_boxplot(d, "ragas_scores.ragas_faithfulness", ax=ax),
        "coverage": lambda ax: coverage_violin(d, ax=ax),
        "rejection": lambda ax: rejection_bars(d, ax=ax),
        "slope_relevance": lambda ax: slopegraph(d, "deepeval_scores.deepeval_relevance", ax=ax),
        "ragas_vs_deepeval_faithfulness": lambda ax: ragas_vs_deepeval(
            d, "ragas_scores.ragas_faithfulness", "deepeval_scores.deepeval_faithfulness", ax=ax),
        "logprob_vs_correctness": lambda ax: logprob_scatter(d, ax=ax),
    }
    for name, fn in figs.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        fn(ax)
        fig.savefig(f"analysis/{name}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote analysis/{name}.png")
