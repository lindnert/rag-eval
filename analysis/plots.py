"""Every figure the analysis produces, in one place.

The plotting functions used to be spread across the three analysis modules —
``analysis.analysis`` (the evaluated-metric plots), ``analysis.rag_analysis`` (the
pipeline-signal plots) and ``analysis.eval_analysis`` (the two cross-link
scatters) — each with its own savefig loop and its own filename convention. They
are collected here so there is exactly one place to look for "what figures exist"
and one place to change how they all look.

Each function draws onto an ``ax`` you pass (or makes its own) and returns it, so
they compose into subplots as easily as they render standalone. ``save_all``
drives the whole set for a run and writes each PNG to the path
``analysis.paths`` derives, so a figure is always named for the results file(s)
behind it.

The functions are grouped by what they plot:

  - PIPELINE SIGNALS (raw, pre-evaluation): generation confidence and hybrid
    retrieval scores by variant / dataset / self-correction stage, and the
    per-id HyDE re-retrieval slope.
  - EVALUATED METRICS: metric distributions by variant, KB-coverage proxy,
    abstention rate, the per-id slopegraph across variants, and RAGAS-vs-DeepEval
    agreement.
  - THE LINK BETWEEN THEM: retrieval score vs a quality metric, and the paired
    per-id confidence-change vs metric-change scatter.

Data shaping stays in the modules that own it (``rag_analysis.retrieval_stage_long``
and friends); this module only draws. That is why it imports both — and why the
three modules import IT only inside their ``__main__`` blocks, which keeps the
import graph acyclic.
"""

import matplotlib.pyplot as plt
import pandas as pd

from analysis import analysis as ev
from analysis import rag_analysis as ra
from analysis import paths

try:
    import seaborn as sns
except ImportError:  # seaborn is optional; every function falls back to matplotlib
    sns = None


# --- shared helpers ----------------------------------------------------------

def _boxplot(df, x, y, ax=None, order=None):
    """Seaborn boxplot with a matplotlib fallback, NaNs in ``y`` dropped first."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    sub = df.dropna(subset=[y])
    if sns is not None:
        sns.boxplot(data=sub, x=x, y=y, order=order, ax=ax)
    else:
        groups = order or list(sub[x].dropna().unique())
        ax.boxplot([sub.loc[sub[x] == g, y].dropna() for g in groups], labels=groups)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    return ax


# --- Pipeline signals (raw results, pre-evaluation) --------------------------

def confidence_boxplot(df, by="variant", ax=None):
    """Generation confidence (mean token logprob) by variant (or any category)."""
    order = ra._order(df) if by == "variant" else None
    ax = _boxplot(df, by, "gen_logprob_stats.mean", ax=ax, order=order)
    ax.set_title(f"Generation confidence (mean logprob) by {by}")
    return ax


def confidence_by_dataset(df, ax=None):
    """Mean-logprob distribution per dataset, split by variant."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    if sns is not None:
        sns.boxplot(data=df, x="source_dataset", y="gen_logprob_stats.mean",
                    hue="variant", hue_order=ra._order(df), ax=ax)
    else:
        ax = _boxplot(df, "source_dataset", "gen_logprob_stats.mean", ax=ax)
    ax.set_title("Generation confidence by dataset and variant")
    ax.tick_params(axis="x", rotation=30)
    return ax


def confidence_stage_boxplot(df, ax=None):
    """Generation confidence across no_rag / rag / rag_sc-orig / rag_sc-final stages.
    The rag_sc_orig box appears only once a run has retried generation."""
    long = ra.confidence_stage_long(df)
    order = [s for s in ra.CONF_STAGE_ORDER if s in long["stage"].unique()]
    ax = _boxplot(long, "stage", "value", ax=ax, order=order)
    ax.set_title("Generation confidence (mean logprob) by stage")
    return ax


def retrieval_stage_boxplot(df, ax=None):
    """Best retrieval score across rag / rag_sc-orig / rag_sc-final stages."""
    long = ra.retrieval_stage_long(df)
    order = [s for s in ra.STAGE_ORDER if s in long["stage"].unique()]
    ax = _boxplot(long, "stage", "value", ax=ax, order=order)
    ax.set_title("Best retrieval score by stage (hybrid dense+BM25)")
    return ax


def retrieval_by_dataset(df, ax=None, col="retrieval_best"):
    """Best retrieval score per dataset, split by retrieval variant: rag, the rag_sc
    average, and the HyDE-retried rag_sc rows before/after re-retrieval — the plot of
    ``rag_analysis.retrieval_by_dataset_variant`` (the two hyde boxes cover that
    subset only)."""
    ax = ax or plt.subplots(figsize=(9, 5))[1]
    long = ra.retrieval_variant_long(df, col=col)
    order = [v for v in ra.RETRIEVAL_VARIANT_ORDER if v in set(long["variant"])]
    if sns is not None:
        sns.boxplot(data=long, x="source_dataset", y="value",
                    hue="variant", hue_order=order, ax=ax)
    else:
        ax = _boxplot(long, "source_dataset", "value", ax=ax)
    ax.set_ylabel(col)
    ax.set_title("Best retrieval score by dataset and variant")
    ax.tick_params(axis="x", rotation=30)
    return ax


def sc_retrieval_slope(df, ax=None):
    """Per-id lines from original -> re-retrieved best score, for the rag_sc rows
    whose retrieval was actually re-run (shows whether HyDE re-retrieval helped)."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    sc = ra._hyde_rows(df).dropna(subset=["retrieval_best", "retrieval_best_orig"]).copy()
    for _, row in sc.iterrows():
        improved = row["retrieval_best"] >= row["retrieval_best_orig"]
        ax.plot([0, 1], [row["retrieval_best_orig"], row["retrieval_best"]],
                color="seagreen" if improved else "crimson", alpha=0.4, marker="o")
    if len(sc):
        ax.plot([0, 1], [sc["retrieval_best_orig"].mean(), sc["retrieval_best"].mean()],
                color="black", marker="o", lw=2.5, label="mean")
        ax.legend()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["original", "re-retrieved"])
    ax.set_ylabel("best retrieval score")
    ax.set_title("rag_sc HyDE re-retrieval: score before vs after")
    return ax


# --- Evaluated metrics -------------------------------------------------------

def metric_boxplot(df, metric, by="variant", ax=None):
    """Distribution of a metric column across variants (or any category)."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    order = ev._order(df) if by == "variant" else None
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
    """Fraction of answers that abstained, per variant.

    Routes through ``analysis._rejection_mask`` (the single source of truth: the
    pipeline's ``rejected`` flag, falling back to ``is_rejection``) rather than
    reading a raw column, so it can never diverge from the other abstention
    consumers.
    """
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    rate = (ev._rejection_mask(df).groupby(df["variant"], observed=True)
            .mean().reindex(ev._order(df)))
    rate.plot.bar(ax=ax)
    ax.set_ylabel("rejection rate")
    ax.set_ylim(0, 1)
    ax.set_title("Abstention rate by variant")
    return ax


def slopegraph(df, metric, ax=None, sample=None):
    """Per-id lines across no_rag -> rag -> rag_sc for one metric."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    wide = df.pivot_table(index="id", columns="variant", values=metric, observed=True)
    wide = wide.reindex(columns=ev._order(df)).dropna(how="all")
    if sample:
        wide = wide.sample(min(sample, len(wide)), random_state=0)
    for _, row in wide.iterrows():
        ax.plot(range(len(wide.columns)), row.values, color="gray", alpha=0.3, marker="o")
    ax.plot(range(len(wide.columns)), wide.mean().values,
            color="crimson", marker="o", lw=2, label="mean")
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
                        hue_order=ev._order(df), ax=ax)
    else:
        ax.scatter(sub[ragas_col], sub[deepeval_col], alpha=0.6)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)  # y = x reference
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("RAGAS vs DeepEval")
    return ax


# --- Signal -> metric (the cross-link) ---------------------------------------

def logprob_scatter(df, metric="ragas_scores.ragas_answer_correctness",
                    logprob_col="gen_logprob_stats.mean", ax=None):
    """Scatter of mean token logprob vs a quality metric, coloured by variant."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    x = pd.to_numeric(df[logprob_col], errors="coerce")
    y = pd.to_numeric(df[metric], errors="coerce")
    m = x.notna() & y.notna()
    if sns is not None:
        sns.scatterplot(x=x[m], y=y[m], hue=df.loc[m, "variant"],
                        hue_order=ev._order(df), ax=ax)
    else:
        ax.scatter(x[m], y[m], alpha=0.6)
    ax.set_xlabel(logprob_col)
    ax.set_ylabel(metric)
    ax.set_title("Generation confidence vs answer quality")
    return ax


def retrieval_metric_scatter(linked, metric, retrieval_col="retrieval_best", ax=None):
    """Best retrieval score vs a context/faithfulness metric, coloured by variant."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    x = pd.to_numeric(linked[retrieval_col], errors="coerce")
    y = pd.to_numeric(linked[metric], errors="coerce")
    m = x.notna() & y.notna()
    if sns is not None and "variant" in linked:
        sns.scatterplot(x=x[m], y=y[m], hue=linked.loc[m, "variant"], ax=ax)
    else:
        ax.scatter(x[m], y[m], alpha=0.6)
    ax.set_xlabel(retrieval_col)
    ax.set_ylabel(metric.split(".")[-1])
    ax.set_title(f"retrieval score vs {metric.split('.')[-1]}")
    return ax


def delta_scatter(df, x_col, y_col, a, b, ax=None):
    """Per-id Δx vs Δy between variants ``a`` and ``b`` (quadrant reference lines)."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    tmp = pd.DataFrame({
        "id": df["id"], "variant": df["variant"].astype(str),
        "x": pd.to_numeric(df[x_col], errors="coerce"),
        "y": pd.to_numeric(df[y_col], errors="coerce"),
    })
    wx = tmp.pivot_table(index="id", columns="variant", values="x", observed=True)
    wy = tmp.pivot_table(index="id", columns="variant", values="y", observed=True)
    dx, dy = wx[a] - wx[b], wy[a] - wy[b]
    m = dx.notna() & dy.notna()
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.scatter(dx[m], dy[m], alpha=0.6)
    ax.set_xlabel(f"Δ {x_col.split('.')[-1]} ({a}−{b})")
    ax.set_ylabel(f"Δ {y_col.split('.')[-1]} ({a}−{b})")
    ax.set_title("confidence change vs metric change")
    return ax


# --- Rendering ---------------------------------------------------------------

def save_all(figs, source, figsize=(7, 5), dpi=200):
    """Render ``{name: fn(ax)}`` and write each to ``analysis/out/<run>/figures/``.

    Replaces the savefig loop each of the three ``__main__`` blocks used to carry.
    ``source`` is the results file the figures came from (or ``[primary, secondary]``
    for the cross-linked ones), and lands in every filename via ``analysis.paths``.

    A figure that raises is reported and skipped rather than killing the rest of
    the run: a plot needing a column this particular results file lacks (the
    rag_sc_orig stages on a run that never retried generation, say) should not
    cost you the other twelve.
    """
    written = []
    for name, fn in figs.items():
        fig, ax = plt.subplots(figsize=figsize)
        try:
            fn(ax)
        except Exception as e:  # noqa: BLE001 — one bad figure must not abort the set
            plt.close(fig)
            print(f"  [SKIP] {name}: {type(e).__name__}: {e}")
            continue
        p = paths.figure(source, name)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
        print(f"  wrote {paths.rel(p)}")
    return written
