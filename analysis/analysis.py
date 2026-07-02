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


def load(path):
    """Flatten evaluated results JSON to a tidy DataFrame with numeric metrics.

    Nested dicts become dot-named columns (e.g. ``ragas_scores.ragas_faithfulness``,
    ``gen_logprob_stats.mean``). Metric columns are coerced to float so sentinel
    strings / nulls turn into NaN. A few convenience columns are added.
    """
    with open(path, "r", encoding="utf-8") as f:
        df = pd.json_normalize(json.load(f))

    metric_cols = [
        c for c in df.columns
        if c.startswith(("ragas_scores.", "deepeval_scores."))
        and not c.endswith("_reason")
        and c != "ragas_scores.ragas_error"
    ]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce")

    if "variant" in df:
        df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    if "retrieval_scores" in df:
        df["retrieval_best"] = df["retrieval_scores"].apply(lambda s: max(s) if s else None)
    if "answer" in df:
        df["is_rejection"] = df["answer"].fillna("").str.strip().eq(REJECTION_ANSWER)

    return df


def _order(df):
    return [v for v in VARIANT_ORDER if v in df["variant"].unique()]


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

    # Render the standard plots next to this script.
    figs = {
        "boxplot_faithfulness": lambda ax: metric_boxplot(d, "ragas_scores.ragas_faithfulness", ax=ax),
        "coverage": lambda ax: coverage_violin(d, ax=ax),
        "rejection": lambda ax: rejection_bars(d, ax=ax),
        "slope_relevance": lambda ax: slopegraph(d, "deepeval_scores.deepeval_relevance", ax=ax),
        "ragas_vs_deepeval_faithfulness": lambda ax: ragas_vs_deepeval(
            d, "ragas_scores.ragas_faithfulness", "deepeval_scores.deepeval_faithfulness", ax=ax),
    }
    for name, fn in figs.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        fn(ax)
        fig.savefig(f"analysis/{name}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote analysis/{name}.png")
