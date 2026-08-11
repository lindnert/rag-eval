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
  - THE HEADLINE THESIS FIGURES: five plots, each one a summary table that only
    becomes an argument once you can see its shape — the paired variant-effect
    forest plot, the metric-rail (discriminativeness) plot, the dataset x variant
    heatmap, abstention by dataset, and metric agreement. These build their own
    Figure rather than drawing onto a supplied ax, and are the ones written in
    PDF as well as PNG for ``\\includegraphics``.

Everything above that group is exploratory: quick looks for working out what
happened in a run, not material meant to go into the thesis as-is.

Data shaping stays in the modules that own it (``rag_analysis.retrieval_stage_long``
and friends); this module only draws. That is why it imports both — and why the
three modules import IT only inside their ``__main__`` blocks, which keeps the
import graph acyclic.
"""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from analysis import analysis as ev
from analysis import rag_analysis as ra
from analysis import paths

try:
    import seaborn as sns
except ImportError:  # seaborn is optional; every function falls back to matplotlib
    sns = None


# --- Palette -----------------------------------------------------------------
# Validated with the dataviz reference validator (light surface, all-pairs):
# lightness band, chroma floor, CVD separation (worst ΔE 9.2 deutan) and
# normal-vision floor (worst ΔE 24.0) all PASS for the three categorical slots.
# Aqua sits below 3:1 contrast on the light surface, so any chart using slot 3
# owes the reader visible labels (the "relief rule") — which is why the bar
# charts below direct-label every bar.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical hues in FIXED slot order — assigned by entity, never cycled and
# never reordered by rank, so "rag is orange" stays true across every figure.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
VARIANT_COLORS = {"no_rag": CATEGORICAL[0], "rag": CATEGORICAL[1],
                  "rag_sc": CATEGORICAL[2]}

# Sequential: ONE hue, light -> dark, for magnitude (the heatmap).
SEQUENTIAL_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                    "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("thesis_blues", SEQUENTIAL_STEPS)

# Diverging: two hues that read as opposite + a NEUTRAL midpoint. Used wherever
# the sign is the message (an effect above/below zero, a +/- correlation). The
# neutral gray is not a third category — it means "not distinguishable from
# nothing", which is exactly what a CI spanning zero says.
POS = "#2a78d6"
NEG = "#e34948"
NEUTRAL = INK_MUTED

# Metric names that do not survive the generic prettifier.
_LABEL_OVERRIDES = {
    "ragas_faithfulness_with_hhem": "RAGAS faithfulness (HHEM)",
    "ragas_id_context_ap": "RAGAS id-context AP",
    "ragas_id_context_precision": "RAGAS id-context precision",
    "ragas_id_context_recall": "RAGAS id-context recall",
}

# The answer-quality metrics the results chapter leads with, in reading order.
HEADLINE_METRICS = [
    "ragas_scores.ragas_answer_correctness",
    "ragas_scores.ragas_answer_accuracy",
    "ragas_scores.ragas_answer_relevancy",
    "deepeval_scores.deepeval_relevance",
    "ragas_scores.ragas_faithfulness",
    "ragas_scores.ragas_faithfulness_with_hhem",
    "deepeval_scores.deepeval_faithfulness",
]

# The two comparisons the thesis argument rests on: does retrieval help, and does
# self-correction add anything on top of it?
DEFAULT_COMPARISONS = [("rag", "no_rag"), ("rag_sc", "rag")]


def metric_label(col):
    """``ragas_scores.ragas_answer_correctness`` -> ``RAGAS answer correctness``.

    Raw column names are fine in a console table and wrong on a thesis axis, so
    every figure below labels through this.
    """
    name = str(col).split(".")[-1]
    if name in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[name]
    for prefix, tag in (("ragas_", "RAGAS "), ("deepeval_", "DeepEval ")):
        if name.startswith(prefix):
            return tag + name[len(prefix):].replace("_", " ")
    return name.replace("_", " ")


def apply_style():
    """Print-oriented matplotlib defaults: recessive solid hairline grid, no top
    or right spine, muted axis ink, no legend frame."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",          # solid hairline — never dashed
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


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

    Routes through ``rag_analysis._abstained`` (the single detector: the
    pipeline's ``rejected`` flag, falling back to a match against every
    language's canonical rejection string) rather than reading a raw column, so
    it can never diverge from the other abstention consumers.
    """
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    rate = (ra._abstained(df).groupby(df["variant"], observed=True)
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
    domain claim), so those are dropped when comparing faithfulness columns —
    via ``rag_analysis._abstained``, matching what ``eval_analysis``'s
    ``metric_agreement`` drops for the same reason.
    """
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    sub = df
    if "faithful" in ragas_col:
        sub = df[~ra._abstained(df)]
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


# --- The headline thesis figures ---------------------------------------------
# These five each replace a table that currently exists only as a CSV, and each
# is the figure for a question the results chapter has to answer. Unlike the
# exploratory plots above they build and return their own Figure, because they
# need control of the layout (long metric labels, colorbars, panel splits);
# render them with ``save`` rather than ``save_all``.

def variant_effect_forest(df, metrics=None, comparisons=None):
    """THE headline figure: does retrieval help, and does self-correction add more?

    One row per metric, one panel per comparison. The dot is the paired mean
    difference ``a − b`` over the questions answered under both variants; the bar
    is its 95% bootstrap CI — so the reader sees direction, magnitude AND
    uncertainty at once, which the four ``compare_*.csv`` tables cannot convey.

    Colour is the SIGN, not the series: blue where the CI clears zero on the
    positive side (``a`` genuinely beats ``b``), red where it clears it on the
    negative side, neutral gray where the CI spans zero and the honest reading is
    "no detectable difference". The paired n is printed per row because a
    difference over 90 pairs and one over 1100 are not the same evidence.

    Metrics whose pairing is empty are dropped rather than drawn as a blank row:
    faithfulness does not exist on ``no_rag``, so it simply has no rag-vs-no_rag
    comparison to make.
    """
    apply_style()
    metrics = [m for m in (metrics or HEADLINE_METRICS) if m in df]
    comparisons = list(comparisons or DEFAULT_COMPARISONS)

    panels = []
    for a, b in comparisons:
        rows = {}
        for m in metrics:
            r = ev.compare_variants(df, m, a=a, b=b).loc["overall"]
            if r.get("n_pairs"):
                rows[m] = {"diff": r["mean_diff"], "lo": r["ci_low"],
                           "hi": r["ci_high"], "n": int(r["n_pairs"])}
        panels.append(((a, b), rows))

    # ONE shared row index across every panel. The panels have different metric
    # sets — faithfulness has no rag-vs-no_rag comparison, because no_rag has no
    # context to be faithful to — and with a shared y-axis, per-panel positions
    # would silently print each panel's marks against the other's labels.
    shown = [m for m in metrics if any(m in rows for _, rows in panels)]
    ypos = {m: len(shown) - 1 - i for i, m in enumerate(shown)}  # first at the top

    fig, axes = plt.subplots(
        1, len(panels), sharey=True,
        figsize=(4.6 * len(panels) + 1.6, 0.5 * len(shown) + 2.2))
    axes = np.atleast_1d(axes)

    for ax, ((a, b), rows) in zip(axes, panels):
        ax.axvline(0, color=AXIS, lw=1)
        ax.grid(axis="y", visible=False)
        for m in shown:
            yi = ypos[m]
            r = rows.get(m)
            if r is None:
                ax.annotate("not applicable", (0.5, yi),
                            xycoords=("axes fraction", "data"),
                            ha="center", va="center", fontsize=7.5,
                            color=INK_MUTED, style="italic")
                continue
            spans_zero = r["lo"] <= 0 <= r["hi"]
            color = NEUTRAL if spans_zero else (POS if r["diff"] > 0 else NEG)
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=color, lw=2,
                    solid_capstyle="butt", zorder=2)
            ax.plot([r["diff"]], [yi], marker="o", ms=8, zorder=3,
                    color=SURFACE if spans_zero else color,
                    markeredgecolor=color, markeredgewidth=2)
            ax.annotate(f"n={r['n']}", (1.0, yi), xycoords=("axes fraction", "data"),
                        xytext=(5, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=7.5, color=INK_MUTED,
                        annotation_clip=False)
        ax.set_ylim(-0.6, len(shown) - 0.4)
        ax.set_yticks([ypos[m] for m in shown])
        ax.set_yticklabels([metric_label(m) for m in shown])
        ax.set_title(f"{a}  −  {b}")
        ax.set_xlabel("paired mean difference (95% CI)")

    handles = [
        Line2D([], [], color=POS, lw=2, marker="o", ms=7,
               label="improvement (CI clears zero)"),
        Line2D([], [], color=NEG, lw=2, marker="o", ms=7,
               label="regression (CI clears zero)"),
        Line2D([], [], color=NEUTRAL, lw=2, marker="o", ms=7,
               markerfacecolor=SURFACE, markeredgewidth=2,
               label="CI spans zero — no detectable difference"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Effect of retrieval and of self-correction, per metric",
                 x=0.008, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 0.96, 0.96), w_pad=3.5)
    return fig


def metric_rail_plot(df, metrics=None, min_scored=1):
    """Which metrics actually discriminate, and which are pinned to a rail.

    Each metric is one diverging stacked bar centred on its SPREAD: the gray
    middle is the share of scored rows strictly between 0 and 1 — the part of the
    metric that can rank anything — and the two arms are the mass welded to the
    rails, 0.0 to the left, 1.0 to the right. Sorted by spread, so the metrics
    that separate queries rise to the top and the degenerate ones sink.

    This is the figure behind the metric-validation argument. A metric that is
    90% one colour is not measuring your system, and its mean elsewhere in the
    chapter cannot be read as quality — which is invisible in a table of means
    and unmissable here.
    """
    apply_style()
    s = ev.metric_summary(df, metrics)
    s = s[s["n"] >= min_scored].copy()
    s["frac_mid"] = 1.0 - s["frac_zero"].fillna(0) - s["frac_one"].fillna(0)
    s = s.sort_values("frac_mid", ascending=True)  # bottom-up = worst first

    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(s) + 2.0))
    y = np.arange(len(s))
    gap = 0.004  # a true surface gap between segments, not a drawn border

    mid = s["frac_mid"].to_numpy()
    z = s["frac_zero"].fillna(0).to_numpy()
    o = s["frac_one"].fillna(0).to_numpy()
    ax.barh(y, mid, left=-mid / 2, height=0.62, color=GRID, zorder=2)
    ax.barh(y, np.maximum(z - gap, 0), left=-mid / 2 - z, height=0.62,
            color=NEG, zorder=2)
    ax.barh(y, np.maximum(o - gap, 0), left=mid / 2 + gap, height=0.62,
            color=POS, zorder=2)

    for yi, zi, mi, oi, n in zip(y, z, mid, o, s["n"]):
        if zi >= 0.06:
            ax.text(-mi / 2 - zi / 2, yi, f"{zi:.0%}", ha="center", va="center",
                    fontsize=7.5, color=SURFACE, zorder=3)
        if oi >= 0.06:
            ax.text(mi / 2 + oi / 2, yi, f"{oi:.0%}", ha="center", va="center",
                    fontsize=7.5, color=SURFACE, zorder=3)
        ax.annotate(f"n={int(n)}", (1.0, yi), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    ha="left", fontsize=7.5, color=INK_MUTED, annotation_clip=False)

    ax.axvline(0, color=SURFACE, lw=0)
    ax.set_yticks(y)
    ax.set_yticklabels([metric_label(m) for m in s.index])
    ax.set_xlabel("share of scored rows  ←  pinned at 0.0    ·    spread    ·    pinned at 1.0  →")
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["100%", "50%", "0", "50%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.set_title("Do the metrics discriminate? Mass on the 0/1 rails vs. spread between")
    handles = [
        Line2D([], [], color=NEG, lw=7, label="scored exactly 0.0"),
        Line2D([], [], color=GRID, lw=7, label="strictly between (the usable signal)"),
        Line2D([], [], color=POS, lw=7, label="scored exactly 1.0"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    return fig


def dataset_variant_heatmap(df, metric, ax=None):
    """Where the system works and where it does not: mean of one metric over the
    dataset x variant grid.

    Fifteen cells is past the point where grouped bars stay readable, so this is
    a grid on a single sequential hue (light = low, dark = high). Every cell is
    annotated, both because a colour scale alone is not an accessible encoding
    and because the exact means matter here.

    The row label carries the number of QUESTIONS in the dataset (unique ids, not
    rows — each question contributes one row per variant): a dataset of 85 and one
    of 450 must not read as equally strong evidence.
    """
    apply_style()
    sub = df.copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    piv = sub.pivot_table(index="source_dataset", columns="variant",
                          values=metric, observed=True, aggfunc="mean")
    piv = piv.reindex(columns=[v for v in ev.VARIANT_ORDER if v in piv.columns])
    counts = sub.groupby("source_dataset", observed=True)["id"].nunique()

    fig, ax = plt.subplots(figsize=(1.5 * len(piv.columns) + 4.2,
                                    0.52 * len(piv) + 2.4))
    vals = piv.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (0, 1)
    im = ax.imshow(vals, cmap=BLUES, vmin=lo, vmax=hi, aspect="auto")
    ax.grid(visible=False)

    # A surface-coloured gap between cells, the heatmap form of the 2px spacer —
    # adjacent fills should never touch.
    for k in range(1, vals.shape[0]):
        ax.axhline(k - 0.5, color=SURFACE, lw=2.5)
    for k in range(1, vals.shape[1]):
        ax.axvline(k - 0.5, color=SURFACE, lw=2.5)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "–", ha="center", va="center",
                        color=INK_MUTED, fontsize=9)
                continue
            # Flip the ink once the cell is dark enough to swallow dark text.
            norm = (v - lo) / (hi - lo) if hi > lo else 0.5
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if norm > 0.55 else INK)

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(list(piv.columns))
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{d}  (n={int(counts.get(d, 0))})" for d in piv.index])
    ax.set_title(f"{metric_label(metric)} by dataset and variant")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.tight_layout()
    return fig


def abstention_grouped_bars(df, variants=("rag", "rag_sc")):
    """How often the system refuses instead of answering, per dataset.

    Abstention is a reliability behaviour, not a failure: refusing when the
    corpus does not cover a question is the intended response, and the gap
    between datasets is the finding. ``no_rag`` is excluded because it has no
    context to call insufficient and cannot abstain by construction — plotting
    its structural zero would imply a comparison that does not exist.

    Every bar is direct-labelled, which is also what discharges the contrast
    relief rule the palette check flagged for slot 3.
    """
    apply_style()
    sub = df[df["variant"].astype(str).isin(variants)]
    tab = ra.abstention_summary(sub, by=["source_dataset", "variant"],
                                with_total=False)
    if not len(tab):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title("no abstention data in this file")
        return fig

    rate = tab["abstention_rate"].unstack("variant")
    rate = rate.reindex(columns=[v for v in variants if v in rate.columns])
    n = tab["n"].unstack("variant").reindex(columns=rate.columns)

    fig, ax = plt.subplots(figsize=(1.6 * len(rate) + 3.4, 4.8))
    x = np.arange(len(rate))
    width = 0.8 / max(len(rate.columns), 1)
    for k, v in enumerate(rate.columns):
        off = (k - (len(rate.columns) - 1) / 2) * width
        color = VARIANT_COLORS.get(str(v), CATEGORICAL[k])
        vals = rate[v].to_numpy(dtype=float)
        ax.bar(x + off, vals, width * 0.92, label=str(v), color=color, zorder=2)
        for xi, val in zip(x + off, vals):
            if not np.isfinite(val):
                continue
            # A label only goes inside the bar when it fits; a tall bar would
            # otherwise push its label off the top of the axes.
            inside = val > 0.90
            ax.text(xi, val - 0.02 if inside else val + 0.015, f"{val:.0%}",
                    ha="center", va="top" if inside else "bottom", fontsize=7.5,
                    color=SURFACE if inside else INK_SECONDARY, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={int(n.loc[d].max())})" for d in rate.index])
    ax.set_ylabel("share of answers that abstained")
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", visible=False)
    ax.set_title("Abstention rate by dataset and variant", pad=26)
    # Between title and plot, so it can never sit on top of a bar.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=len(rate.columns))
    fig.tight_layout()
    return fig


def metric_agreement_dots(ag, stats=("spearman", "pearson")):
    """Do metrics that claim to measure the same thing actually agree?

    One row per comparable metric pair, one marker per correlation statistic
    (circle = Spearman, the rank correlation these bounded non-normal scores
    call for; square = Pearson, shown for completeness). Colour is again the
    sign: blue for positive agreement, red for a pair that ranks queries in
    OPPOSITE directions.

    A red row is a genuine result, not a plotting artefact — two judges scoring
    the same construct and disagreeing means at least one of them is not
    measuring what its name says, and no mean built on it is safe.

    Takes the table from ``eval_analysis.metric_agreement`` rather than computing
    it, which keeps this module off the import cycle that would create.
    """
    apply_style()
    tbl = ag[ag["group"] == "overall"] if "group" in ag else ag
    tbl = tbl.reset_index(drop=True)
    if not len(tbl):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title("no comparable metric pairs in this file")
        return fig

    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(tbl) + 2.4))
    y = np.arange(len(tbl))[::-1]
    markers = {"spearman": "o", "pearson": "s"}
    # The two statistics often land on the same value, so they are nudged apart
    # rather than hidden behind one another. The row rule between them (the y
    # grid, on for this chart) is what still ties both to their label.
    offsets = np.linspace(0.13, -0.13, len(stats)) if len(stats) > 1 else [0.0]
    ax.axvline(0, color=AXIS, lw=1)

    for yi, row in zip(y, tbl.itertuples()):
        for stat, dy in zip(stats, offsets):
            v = getattr(row, stat, np.nan)
            if v is None or not np.isfinite(v):
                continue
            ax.plot([v], [yi + dy], marker=markers.get(stat, "o"), ms=9,
                    color=POS if v > 0 else NEG, zorder=3, ls="none",
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"n={int(row.n)}", (1.0, yi), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    ha="left", fontsize=7.5, color=INK_MUTED, annotation_clip=False)

    ax.set_ylim(-0.6, len(tbl) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(list(tbl["pair"]))
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("correlation between the two metrics")
    ax.grid(axis="y", visible=True)
    ax.set_title("Do comparable metrics agree? (negative = they rank queries oppositely)",
                 pad=26)
    handles = [Line2D([], [], color=INK_MUTED, marker=markers[s], ls="none", ms=8,
                      label=s.capitalize()) for s in stats if s in markers]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.005),
              ncol=len(handles))
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    return fig


# --- Rendering ---------------------------------------------------------------

# PNG to look at, PDF to \includegraphics — LaTeX wants vector, and a 200-dpi
# raster is visibly soft next to body text in print.
FORMATS = ("png", "pdf")


def save(fig, source, name, formats=FORMATS, dpi=200, quiet=False):
    """Write one already-built Figure to ``analysis/out/<run>/figures/`` in each
    requested format, and close it. Returns the paths written."""
    written = []
    for ext in formats:
        p = paths.figure(source, f"{name}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    if not quiet:
        print(f"  wrote {paths.rel(written[0])}"
              f"{f' (+{len(written) - 1} more format)' if len(written) > 1 else ''}")
    return written


def save_all(figs, source, figsize=(7, 5), formats=FORMATS, dpi=200):
    """Render ``{name: fn(ax)}`` and write each to ``analysis/out/<run>/figures/``.

    Replaces the savefig loop each of the three ``__main__`` blocks used to carry.
    ``source`` is the results file the figures came from (or ``[primary, secondary]``
    for the cross-linked ones), and lands in every filename via ``analysis.paths``.

    Each value may also be a zero-argument callable returning its own Figure —
    that is how the headline figures above are rendered, since they lay out their
    own panels and colorbars rather than drawing onto a single supplied ax.

    A figure that raises is reported and skipped rather than killing the rest of
    the run: a plot needing a column this particular results file lacks (the
    rag_sc_orig stages on a run that never retried generation, say) should not
    cost you the other twelve.
    """
    written = []
    for name, fn in figs.items():
        # Which form is this? Read the signature rather than calling and catching
        # TypeError — a TypeError raised *inside* a figure would otherwise be
        # misread as "this one wants an ax".
        wants_ax = len(inspect.signature(fn).parameters) > 0
        fig = None
        try:
            if wants_ax:
                fig, ax = plt.subplots(figsize=figsize)
                fn(ax)
            else:
                fig = fn()
        except Exception as e:  # noqa: BLE001 — one bad figure must not abort the set
            if fig is not None:
                plt.close(fig)
            print(f"  [SKIP] {name}: {type(e).__name__}: {e}")
            continue
        written.extend(save(fig, source, name, formats=formats, dpi=dpi))
    return written
