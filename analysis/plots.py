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

Data shaping stays in the modules that own it (``rag_analysis.retrieval_variant_long``
and friends); this module only draws. That is why it imports both — and why the
three modules import IT only inside their ``__main__`` blocks, which keeps the
import graph acyclic.
"""

import inspect

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
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

# The ordered fills for a STACKED composition, light -> dark by "how much of the thing".
# Its own ramp rather than steps of SEQUENTIAL_STEPS, which spans L* 89->24 and left its
# two dark ends reading as one navy in thin segments. This spans L* ~89->15 and widens
# the steps as it darkens (~19, ~22, ~33), since equal lightness steps are NOT equally
# visible — the eye separates pale blues far better than dark ones.
ORDERED_FILLS = ["#cde2fb", "#74acee", "#2461bd", "#09214a"]
_ORDERED_RAMP = LinearSegmentedColormap.from_list("thesis_ordered", ORDERED_FILLS)


# A light/dark PAIR of the one hue, for a figure that shows exactly two related things
# side by side. Taken from inside the ramp rather than from its ends: the extremes exist
# so a 2%-tall stacked segment still separates, while two free-standing bars only have to
# be told apart — at full contrast they stop reading as two steps of one scale.
PAIRED_STEPS = [SEQUENTIAL_STEPS[1], SEQUENTIAL_STEPS[5]]


def _ordered_fills(n):
    """``n`` fills spanning ``ORDERED_FILLS``' ramp, light -> dark.

    Interpolated rather than picked, so a composition of five buckets is drawn on the
    same scale as one of four; at n=4 it returns the anchors themselves.
    """
    if n <= 1:
        return ORDERED_FILLS[:1]
    return [to_hex(_ORDERED_RAMP(i / (n - 1))) for i in range(n)]

# Diverging: two hues that read as opposite + a NEUTRAL midpoint. Used wherever
# the sign is the message (an effect above/below zero, a +/- correlation). The
# neutral gray is not a third category — it means "not distinguishable from
# nothing", which is exactly what a CI spanning zero says.
POS = "#2a78d6"
NEG = "#e34948"
NEUTRAL = INK_MUTED

# Metric names that do not survive the generic prettifier. ``deepeval_relevance`` is
# here because the column name is the odd one out, not the metric: it is DeepEval's
# answer-relevancy scorer, the counterpart to ``ragas_answer_relevancy``, and calling
# it "relevance" in a figure that also carries a CONTEXTUAL relevance metric invites
# exactly the confusion the pairing is meant to resolve.
_LABEL_OVERRIDES = {
    "ragas_faithfulness_with_hhem": "RAGAS Faithfulness HHEM",
    "ragas_id_context_ap": "RAGAS Context AP",
    "ragas_id_context_precision": "RAGAS Context Precision",
    "ragas_id_context_recall": "RAGAS Context Recall",
    "deepeval_relevance": "DeepEval Answer Relevancy",
}

# Every metric in the order the results chapter reads them, grouped by what they
# measure: answer quality, relevancy, faithfulness, then retrieval quality — and
# within each group the RAGAS scorers before their DeepEval counterpart. The
# retrieval group leads with deepeval_contextual_relevance because it is the one
# retrieval metric scored on all 735 rag/rag_sc rows; the three ragas_id_context_*
# below it need a gold reference-context set and exist for the 68 synthetic
# questions only. Figures that list metrics use this rather than sorting by value,
# so a metric sits in the same place in every figure and the reader can compare
# two of them side by side without re-reading the axis.
METRIC_ORDER = [
    "ragas_scores.ragas_answer_correctness",
    "ragas_scores.ragas_answer_accuracy",
    "ragas_scores.ragas_answer_relevancy",
    "deepeval_scores.deepeval_relevance",
    "ragas_scores.ragas_faithfulness",
    "ragas_scores.ragas_faithfulness_with_hhem",
    "deepeval_scores.deepeval_faithfulness",
    "deepeval_scores.deepeval_contextual_relevance",
    "ragas_scores.ragas_id_context_precision",
    "ragas_scores.ragas_id_context_recall",
    "ragas_scores.ragas_id_context_ap",
]

# The answer-quality metrics the results chapter leads with — the leading prefix of
# METRIC_ORDER, up to and including the faithfulness family. The retrieval-quality
# metrics are excluded on purpose: they exist for the 68 synthetic questions only,
# so they cannot appear next to metrics with n ~ 1000 without implying they carry
# the same weight.
HEADLINE_METRICS = METRIC_ORDER[:7]


def order_metrics(metrics):
    """``metrics`` sorted into ``METRIC_ORDER``, with anything unlisted appended in
    its original order — a new metric shows up at the bottom rather than vanishing.
    """
    rank = {m: i for i, m in enumerate(METRIC_ORDER)}
    return sorted(metrics, key=lambda m: (rank.get(m, len(rank)), list(metrics).index(m)))

# The two comparisons the thesis argument rests on: does retrieval help, and does
# self-correction add anything on top of it?
DEFAULT_COMPARISONS = [("rag", "no_rag"), ("rag_sc", "rag")]


def metric_label(col):
    """``ragas_scores.ragas_answer_correctness`` -> ``RAGAS Answer Correctness``.

    Raw column names are fine in a console table and wrong on a thesis axis, so
    every figure below labels through this.

    Title case, word by word rather than ``str.title()``, because the parts that are
    already capitalised are the ones ``str.title()`` destroys: the scorer names are
    brandmarks (RAGAS, DeepEval) and HHEM is an acronym, and "Ragas Faithfulness
    Hhem" is not a name anyone would recognise.
    """
    name = str(col).split(".")[-1]
    if name in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[name]
    for prefix, tag in (("ragas_", "RAGAS "), ("deepeval_", "DeepEval ")):
        if name.startswith(prefix):
            return tag + _titlecase(name[len(prefix):])
    return _titlecase(name)


def _titlecase(name):
    """``answer_correctness`` -> ``Answer Correctness``, leaving words that are
    already all-caps alone."""
    return " ".join(w if w.isupper() else w.capitalize()
                    for w in str(name).replace("_", " ").split())


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
        # The tick MARKS stay muted — they are scale furniture. The tick LABELS carry
        # the category names on most figures here (variants, datasets, metrics), so
        # they take the same ink as an axis label and stay readable at print size.
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# --- shared helpers ----------------------------------------------------------

# What the pooled group is called wherever a per-dataset figure also carries the whole
# run. One spelling, because the reader meets it on four figures side by side.
_POOLED_LABEL = "all datasets"

# The reading order for datasets, fixed for EVERY figure: the three exam-style
# benchmarks, then the two applied sets. Fixed rather than per-figure (by size, by
# effect) so a reader who has learnt one chart's x axis has learnt them all and can
# carry a dataset's position from one figure to the next — and so two runs stay
# comparable column for column. Anything not listed keeps its own order at the end,
# which is what makes a new dataset show up rather than silently vanish.
DATASET_ORDER = ["medqa", "mmlu", "llmdrs", "ngqa", "synthetic_guidelines"]

# Display names: the data keeps the pipeline's own identifiers, the figure prints how
# the thesis writes them. The four benchmarks get their published capitalisation;
# ``synthetic_guidelines`` is only shortened, since that label is the widest on every
# per-dataset chart in the set and the questions are ours, not a named benchmark.
DATASET_LABELS = {"medqa": "MedQA", "mmlu": "MMLU", "llmdrs": "LLMDRS",
                  "ngqa": "NGQA", "synthetic_guidelines": "synthetic"}


def dataset_label(name):
    """The name a dataset goes by on an axis (``synthetic_guidelines`` -> ``synthetic``)."""
    return DATASET_LABELS.get(str(name), str(name))


# Display names for the variants: the data keeps the pipeline's own identifiers, the
# figure prints how the thesis writes them. Callers opt in through ``variant_label``,
# so a figure whose axis is meant to show the literal column value still can.
VARIANT_LABELS = {"no_rag": "Baseline LLM", "rag": "Naive RAG", "rag_sc": "SC-RAG"}


def variant_label(name):
    """The name a variant goes by in prose (``rag_sc`` -> ``RAG-SC``)."""
    return VARIANT_LABELS.get(str(name), str(name))


def dataset_order(names):
    """``names`` in ``DATASET_ORDER``, with the pooled group first and unknown datasets
    appended in their own order rather than dropped."""
    names = list(dict.fromkeys(str(n) for n in names))
    pooled = [n for n in names if n == _POOLED_LABEL]
    known = [d for d in DATASET_ORDER if d in names]
    return pooled + known + [n for n in names
                             if n not in DATASET_ORDER and n != _POOLED_LABEL]


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


def _with_pooled_dataset(df, col="source_dataset"):
    """``(df with every row repeated once more under the pooled label, axis order)``.

    How a distribution plot gets its ``all datasets`` group. That group is the SAME rows
    a second time, not a sixth dataset — which is why every figure drawing it puts it
    first and behind a rule: it is the reference the datasets are read against.
    """
    out = pd.concat([df, df.assign(**{col: _POOLED_LABEL})], ignore_index=True)
    out[col] = out[col].astype(str)
    return out, dataset_order(out[col].unique())


def _label_dataset_ticks(ax, order, axis="x"):
    """Retick a categorical dataset axis with ``dataset_label`` names, and rule the
    pooled group off from the datasets it pools."""
    ticks = range(len(order))
    labels = [dataset_label(d) for d in order]
    if axis == "x":
        ax.set_xticks(ticks), ax.set_xticklabels(labels), ax.set_xlabel("")
    else:
        ax.set_yticks(ticks), ax.set_yticklabels(labels), ax.set_ylabel("")
    if order and order[0] == _POOLED_LABEL:
        (ax.axvline if axis == "x" else ax.axhline)(0.5, color=AXIS, lw=0.8, zorder=1)


def _ink_on(color, flip_below=0.32):
    """``INK`` or ``SURFACE``, whichever a label can be read in on ``color``.

    Measured from the fill's relative luminance rather than from its position in a
    palette list, so a hand-picked colour is handled as correctly as a ramp step.
    """
    r, g, b = to_rgb(color)
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return SURFACE if 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] < flip_below \
        else INK


# --- Pipeline signals (raw results, pre-evaluation) --------------------------

def confidence_boxplot(df, by="variant", ax=None):
    """Generation confidence (mean token logprob) by variant (or any category)."""
    order = ra._order(df) if by == "variant" else None
    ax = _boxplot(df, by, "gen_logprob_stats.mean", ax=ax, order=order)
    ax.set_title(f"Generation confidence (mean logprob) by {by}")
    return ax


def confidence_by_dataset(df, ax=None):
    """Mean-logprob distribution per dataset, split by variant, the pooled run first."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    sub, order = _with_pooled_dataset(df)
    if sns is not None:
        sns.boxplot(data=sub, x="source_dataset", y="gen_logprob_stats.mean", order=order,
                    hue="variant", hue_order=ra._order(df), ax=ax)
    else:
        ax = _boxplot(sub, "source_dataset", "gen_logprob_stats.mean", ax=ax, order=order)
    _label_dataset_ticks(ax, order)
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
    """Best retrieval score across the four retrieval groups of
    ``rag_analysis.retrieval_variant_long`` — the picture of the table
    ``retrieval_by_variant`` prints, box for box.

    It shapes its data through that one function rather than a stage-specific twin so
    no box can be built over a different cohort than its neighbour: ``rag`` and
    ``rag_sc`` are whole variants over the same ids, and ``rag_sc_hyde_orig`` ->
    ``rag_sc_hyde_final`` is a MATCHED pair over only the rows that re-retrieved. The
    before/after gap is readable only across that pair; comparing the orig box to the
    all-rows ``rag_sc`` box would mix a change in score with a change in who is in it.
    """
    long = ra.retrieval_variant_long(df)
    order = [v for v in ra.RETRIEVAL_VARIANT_ORDER if v in set(long["variant"])]
    ax = _boxplot(long, "variant", "value", ax=ax, order=order)
    # n on the tick, because the whole point of the four groups is that two of them
    # are a subset: the reader should not have to look the sizes up in the table.
    n = long.groupby("variant", observed=True).size()
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{v}\n(n={int(n.get(v, 0))})" for v in order])
    ax.set_xlabel("")
    ax.set_ylabel("hybrid dense+BM25 score")
    ax.set_title("Retrieval Scores by variant")
    return ax


def sc_retrieval_gain_by_dataset(df):
    """Best retrieval score before vs after the HyDE re-retrieval, per dataset — the
    picture of ``rag_analysis.sc_retrieval_gain(df, by="source_dataset")``.

    This is ``sc_retrieval_slope`` cut by dataset, and it is deliberately built on ONE
    cohort: the rag_sc rows whose retrieval was actually re-run, measured twice. It
    replaced a four-hue per-dataset boxplot (rag / rag_sc / rag_sc_hyde_orig /
    rag_sc_hyde_final) in which two boxes of every group spanned all rag_sc rows and two
    spanned only the retried subset, so neighbouring boxes differed in who was in them as
    well as in what they scored — twenty boxes to answer one question.

    Dropping the ``rag`` box costs nothing: rag_sc re-uses the first retrieval verbatim,
    so on these ids rag's score IS the "before" end of the arrow (verified on the
    2026-07-31 run: max |difference| = 0). The full four-way breakdown is still printed
    as ``retrieval_by_dataset_variant``.

    Rows run top to bottom in ``DATASET_ORDER`` — the same order, and the same leading
    pooled group, as every other per-dataset figure, so a dataset keeps its position
    across the set. The pooled row is the number ``sc_retrieval_gain(df)`` reports for
    the whole cohort.

    Title and footnote are not drawn — they belong in the document's own caption, and
    this is what they said: retrieval score before vs after HyDE re-retrieval, by
    dataset. Only the rag_sc rows whose retrieval was re-run, so both ends of an arrow
    come from the same rows; the plain rag scores are identical to the 'before' end on
    these ids. On a score-merge file the mean cannot fall — read the 'improved' share on
    the tick, not the direction. The x-axis label, the before/after end labels and the
    per-row deltas stay ON the figure: they say what the arrows are, not what to make
    of them.
    """
    apply_style()
    tab = ra.sc_retrieval_gain(df, by="source_dataset")
    if not len(tab):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title("Retrieval score before vs after HyDE re-retrieval")
        ax.text(0.5, 0.5, "no re-retrieved rows in this file", ha="center", va="center",
                color=INK_MUTED)
        ax.set_axis_off()
        return fig

    rows = pd.concat([ra.sc_retrieval_gain(df).rename(index={"all": _POOLED_LABEL}),
                      tab.loc[dataset_order(tab.index)]])
    # Pooled row on top, then a gap and a rule, then the datasets running downwards.
    ys = np.append([len(rows) - 0.4], np.arange(len(rows) - 1)[::-1])

    fig, ax = plt.subplots(figsize=(8.2, 0.52 * len(rows) + 2.2))
    for y, (_, r) in zip(ys, rows.iterrows()):
        delta = r["delta"]
        # Sign is the message, so it gets the diverging pair; a file where re-retrieval
        # cost score must not read the same as one where it gained.
        color = POS if delta > 0 else NEG if delta < 0 else NEUTRAL
        ax.annotate("", xy=(r["mean_final"], y), xytext=(r["mean_orig"], y),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.2,head_length=0.45",
                                    color=color, lw=2.2, shrinkA=4, shrinkB=0))
        ax.plot(r["mean_orig"], y, marker="o", ms=7, mfc=SURFACE, mec=color, mew=1.6,
                zorder=3)
        ax.text(max(r["mean_orig"], r["mean_final"]) + 0.004, y, f"{delta:+.3f}",
                va="center", ha="left", fontsize=9, color=INK)

    # n and "how many rows moved at all" ride on the tick, because a mean shift says
    # nothing about how widely it was shared — ngqa's small gain is a third of its rows
    # not moving, not every row moving a little.
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{dataset_label(name)}\n{int(r['n'])} rows · "
                        f"{r['frac_improved']:.0%} improved" for name, r in rows.iterrows()])
    ax.axhline((ys[0] + ys[1]) / 2, color=AXIS, lw=0.8)

    ends = rows[["mean_orig", "mean_final"]].to_numpy(dtype=float)
    lo, hi = float(np.nanmin(ends)), float(np.nanmax(ends))
    pad = 0.12 * (hi - lo) or 0.02
    ax.set_xlim(lo - pad, hi + 2.4 * pad)       # right margin holds the delta labels
    ax.set_ylim(-0.6, ys[0] + 0.85)
    ax.set_xlabel("mean best retrieval score (hybrid dense+BM25)")
    ax.grid(axis="y", visible=False)
    # Direct-labelled on the pooled arrow instead of a legend. Each label is anchored
    # OUTWARD from its own end rather than centred over it, so it cannot collide with
    # its partner however short that arrow turns out to be.
    top = rows.iloc[0]
    for x, text, ha in ((top["mean_orig"], "before ", "right"),
                        (top["mean_final"], " after HyDE", "left")):
        ax.text(x, ys[0] + 0.42, text, ha=ha, fontsize=8, color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def sc_displacement_bars(df):
    """How much of the original context the HyDE re-retrieval pushed out, per dataset —
    ``mean_dropped`` from ``rag_analysis.sc_context_displacement(df, by="source_dataset")``.

    The companion to ``sc_retrieval_gain_by_dataset``, on the same 307-row cohort: that
    figure shows what re-retrieval did to the SCORE (which on a score-merge file cannot
    fall), this one shows what it did to the CHUNKS, which is the half that can go wrong.

    One measure, not two: the wholesale-replacement count is the tail of this same
    distribution and mostly re-ranks the datasets the same way, so plotting it beside the
    mean cost a second panel to say it again. It stays in the table
    (``rows_all_replaced`` and its share), where the two rows it actually separates —
    medqa and mmlu, equal means, 32% vs 41% wholesale — can be read off directly.

    Bars run in ``DATASET_ORDER`` behind the pooled group, as everywhere else.

    Title and footnote are not drawn — they belong in the document's own caption, and
    this is what they said: what HyDE re-retrieval displaced from the context, by
    dataset. Only the rag_sc rows whose retrieval was re-run, counted against their own
    dataset (n on the tick); the first bar pools the same rows. How often the drop took
    the WHOLE context is in the companion table — possible under the score merge only,
    since the rrf merge always keeps the question's top chunk. The dashed line and its
    label stay ON the figure: they are the ceiling the bars are read against, and
    without them "2.7 chunks dropped" is a number rather than 90% of the context.
    """
    apply_style()
    tab = ra.sc_context_displacement(df, by="source_dataset")
    if not len(tab):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title("What HyDE re-retrieval displaced from the context")
        ax.text(0.5, 0.5, "no re-retrieved rows with recorded context ids in this file",
                ha="center", va="center", color=INK_MUTED)
        ax.set_axis_off()
        return fig

    pooled = [i for i in tab.index if i == "ALL"]
    order = dataset_order([i for i in tab.index if i != "ALL"])
    tab = tab.loc[pooled + order]
    xs = np.append(np.zeros(len(pooled)), np.arange(len(order)) + len(pooled) + 0.6)

    # The ceiling: a row can only lose the chunks it retrieved. Constant (3) in every run
    # so far, so it is drawn as the reference the bars are read against — without it
    # "2.7 chunks dropped" is a number, with it it is 90% of the context.
    n_orig = ra.sc_context_displacement(df)["n_orig"]
    ceiling = float(n_orig.iloc[0]) if n_orig.nunique() == 1 else float(n_orig.max())

    # Mid-ramp: a single series carries no comparison, so it does not owe a lighter
    # partner room, and the labels ride above the bars where the fill never reaches them.
    fig, ax = plt.subplots(figsize=(1.35 * len(xs) + 3.0, 4.8))
    ax.bar(xs, tab["mean_dropped"].to_numpy(dtype=float), 0.68, color=SEQUENTIAL_STEPS[3],
           zorder=2)
    for x, v in zip(xs, tab["mean_dropped"]):
        ax.text(x, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK,
                zorder=3)
    ax.axhline(ceiling, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(xs[-1] + 0.55, ceiling + 0.03, f"all {ceiling:.0f} — nothing of the "
            "original context left", ha="right", va="bottom", fontsize=8, color=INK_MUTED)
    ax.set_ylim(0, ceiling * 1.16)
    ax.set_ylabel("mean number of swapped chunks")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    if len(pooled):
        ax.axvline((xs[0] + xs[1]) / 2, color=AXIS, lw=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{_POOLED_LABEL if i == 'ALL' else dataset_label(i)}\n(n={int(tab.loc[i, 'n'])})"
         for i in tab.index])
    fig.tight_layout()
    return fig


def _spread(values, gap, lo=None, hi=None):
    """``values`` nudged apart until consecutive ones are at least ``gap`` apart,
    each staying as close to where it started as that allows.

    The label-collision pass for a figure that direct-labels many close-together
    points. Two sweeps: up from the bottom, then down from the top, which is what
    keeps the result inside ``[lo, hi]`` — the upward sweep alone pushes the whole
    stack past the ceiling as soon as the points need more room than the axis has.
    ``values`` must already be sorted; that is the caller's ordering, not something
    to impose here, because the labels have to keep their pairing with the data.
    """
    out = np.asarray(values, dtype=float).copy()
    if gap <= 0 or len(out) < 2:
        return out
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    if hi is not None and out[-1] > hi:
        out[-1] = hi
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(out[i], out[i + 1] - gap)
    if lo is not None and out[0] < lo:
        # More labels than the axis can hold at this gap: pack from the floor up and
        # let them touch, rather than silently drawing some outside the frame.
        out = lo + np.arange(len(out)) * gap
    return out


def sc_retrieval_slope(df, ax=None, bands=10, show_rows=False):
    """Original -> re-retrieved best score for the rag_sc rows whose retrieval was
    actually re-run, as one slope per band of ``bands`` equal-sized groups of the
    STARTING score (does HyDE re-retrieval help, and where).

    The claim is not "re-retrieval helps on average" — one number says that — but
    that it helps MOST where the first retrieval was weakest. So the figure is a
    binned one, not a per-row one:

      - The 307 individual rows are NOT drawn (``show_rows`` brings them back as
        hairlines). At this density no line can be followed across the gap, and the
        bundle hides the low-start rows it exists to show. Each band's mean slope is
        drawn instead — ~20 rows per line at the default, enough that a line is not
        one lucky query and fine enough that the fan has shape.
      - Every band is drawn in ONE colour. Colour has nothing left to say here: the
        bands are already ordered by where they start, so a ramp over the starting
        score only restates the y position, and spending the channel on it left the
        pooled mean with no ink of its own to be drawn in. Sign is not encoded
        either — the old rule painted a falling line red, and in this run nothing
        falls at all (re-retrieval keeps the better context by construction), so
        that channel was reserved for a category that cannot occur.
      - Each band carries its own mean gain, printed at its left end where the bands
        are still separated. They converge on the right — that convergence IS the
        result, and it is also why nothing can be labelled there.

    Tall and narrow on purpose. The slope of a line is read as an angle, and the
    angle is set by the aspect the data is drawn at: the same +0.09 that is a shrug
    on a wide axis is unmistakable on a narrow one.
    """
    apply_style()
    sc = ra._hyde_rows(df).dropna(subset=["retrieval_best", "retrieval_best_orig"]).copy()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 9.4))
    fig = ax.figure
    if not len(sc):
        ax.text(0.5, 0.5, "no re-retrieved rag_sc rows in this file",
                ha="center", va="center", color=INK_MUTED)
        return fig

    orig = pd.to_numeric(sc["retrieval_best_orig"], errors="coerce")
    new = pd.to_numeric(sc["retrieval_best"], errors="coerce")
    gain = new - orig

    # Quantiles of the starting score, not fixed cuts: the band edges are a property
    # of this run's retrieval, and hard-coded ones would silently stop splitting the
    # data on a run whose scores sit elsewhere. ``duplicates`` guards a degenerate
    # column (every row the same score) rather than raising mid-figure.
    k = int(min(bands, orig.nunique()))
    if k >= 2:
        band = pd.qcut(orig, k, labels=False, duplicates="drop").astype(int)
        k = int(band.max()) + 1
    else:
        band, k = pd.Series(0, index=orig.index), 1
    # One hue for every band, from the middle of the ramp: dark enough to hold a 2.2pt
    # line on the page, light enough that the pooled mean reads as darker than all of
    # them wherever it crosses one.
    band_color = SEQUENTIAL_STEPS[3]

    # A halo in the page colour, so a slope stays one continuous line where it crosses
    # its neighbours instead of dissolving into them.
    halo = [pe.Stroke(linewidth=4.6, foreground=SURFACE), pe.Normal()]

    if show_rows:
        for i in orig.index:
            ax.plot([0, 1], [orig[i], new[i]], color=band_color, alpha=0.25, lw=0.8,
                    solid_capstyle="round", zorder=2)

    starts = []
    for b in range(k):
        m = band == b
        y0, y1 = orig[m].mean(), new[m].mean()
        ax.plot([0, 1], [y0, y1], color=band_color, lw=2.2, marker="o", ms=4.4,
                markeredgecolor=SURFACE, markeredgewidth=0.9, path_effects=halo,
                zorder=4)
        starts.append((y0, f"{gain[m].mean():+.3f}"))

    # The pooled mean is now the only thing colour has to distinguish, so it gets the
    # page's darkest ink and the top of the stack: black against blue separates in
    # print, in greyscale and for every kind of colour vision, which no second blue
    # would. Squared caps and a wider halo so it stays a single unbroken bar across
    # the ten lines it cuts through.
    ax.plot([0, 1], [orig.mean(), new.mean()], color=INK, lw=3.6, marker="o", ms=7.0,
            markeredgecolor=SURFACE, markeredgewidth=1.4, solid_capstyle="butt",
            path_effects=[pe.Stroke(linewidth=7.4, foreground=SURFACE), pe.Normal()],
            zorder=6)
    ax.annotate(f"all rows  {gain.mean():+.3f}", (1, new.mean()), xytext=(11, 0),
                textcoords="offset points", ha="left", va="center", fontsize=8.5,
                fontweight="semibold", color=INK, zorder=8)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["original", "re-retrieved"])
    # Room for the labels at both ends, and — since the data still spans x 0..1 —
    # every inch of margin also steepens the slopes.
    ax.set_xlim(-0.30, 1.34)
    ax.set_ylabel("best retrieval score")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()

    # The per-band labels go on LAST, and only once the axes has its final size: at 30
    # bands the starting scores in the middle sit closer together than a line of type
    # is tall, so where a label has to be pushed off its band a leader line says which
    # band it belongs to. Done in data coordinates, hence the draw() first.
    fig.canvas.draw()
    lo, hi = ax.get_ylim()
    ax_pt = ax.get_window_extent().height * 72.0 / fig.dpi
    gap = (hi - lo) * 9.0 / ax_pt if ax_pt else 0.0
    placed = _spread([y for y, _ in starts], gap, lo, hi)
    for (y_band, text), y_text in zip(starts, placed):
        ax.annotate(text, (0, y_band), xytext=(-0.085, y_text), textcoords="data",
                    ha="right", va="center", fontsize=7.5, color=INK_SECONDARY,
                    zorder=8, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6, shrinkA=2,
                                    shrinkB=3))
    return fig


def sc_retrieval_gain_scatter(df):
    """The re-retrieval gain against the score it started from — one dot per row, no
    bins. The same rows as ``sc_retrieval_slope``, read as a difference.

    This is the figure that carries "the weakest retrievals gained the most" with
    nothing in between the reader and the rows: x is where the row started, y is what
    re-retrieval added, and the claim is the downward tilt. The slopegraph and any
    banded version of this both have to choose cut points, and a reader is entitled
    to wonder whether the staircase survives a different choice — here there is
    nothing to choose.

    Two things worth reading off it besides the trend:

      - Nothing is below zero. Re-retrieval never returned a worse best chunk in this
        run, so the horizontal rule is a floor, not a midline.
      - The rows sitting exactly ON that floor gained nothing at all — HyDE came back
        with the same top chunk. They are a fifth of the set, and they are NOT spread
        evenly: 5% of the lowest-starting quartile against 38% of the highest. So the
        flat right-hand end of the trend is not "small gains everywhere", it is
        "often no gain at all", and no banded version of this figure can say that —
        a band mean of +0.02 reads the same either way.

    Read with one caveat: part of any such gradient is mechanical. The score is
    bounded, so a row starting at 0.78 has less headroom than one starting at 0.55,
    and re-measuring the same context with noise would produce a weak version of this
    tilt on its own. The effect here is far larger than that accounts for, but the
    figure cannot separate the two by itself.
    """
    apply_style()
    sc = ra._hyde_rows(df).dropna(subset=["retrieval_best", "retrieval_best_orig"]).copy()
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    if not len(sc):
        ax.text(0.5, 0.5, "no re-retrieved rag_sc rows in this file",
                ha="center", va="center", color=INK_MUTED)
        return fig

    x = pd.to_numeric(sc["retrieval_best_orig"], errors="coerce").to_numpy(dtype=float)
    y = (pd.to_numeric(sc["retrieval_best"], errors="coerce").to_numpy(dtype=float) - x)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    ax.plot(x, y, ls="none", marker="o", ms=4.6, alpha=0.42, color=SEQUENTIAL_STEPS[3],
            markeredgecolor="none", zorder=3)

    handles = [Line2D([], [], ls="none", marker="o", ms=5.5, alpha=0.6,
                      color=SEQUENTIAL_STEPS[3],
                      label=f"one re-retrieved chunk set (n={len(x)})")]
    if len(x) > 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, intercept + slope * xs, color=INK, lw=2.2, zorder=4,
                path_effects=[pe.Stroke(linewidth=4.6, foreground=SURFACE), pe.Normal()])
        r = float(np.corrcoef(x, y)[0, 1])
        handles.append(Line2D([], [], color=INK, lw=2.2,
                              label=f"least-squares fit  ·  slope {slope:+.2f}  ·  "
                                    f"r = {r:+.2f}"))
    # The count of rows sitting exactly on the floor is NOT printed here — it is
    # thesis-caption material, and the docstring above carries the number.
    ax.set_xlabel("best retrieval score before re-retrieval")
    ax.set_ylabel("gain in best retrieval score")
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=1,
              fontsize=8.5, handlelength=2.0)
    fig.tight_layout()
    return fig


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
    sub, order = _with_pooled_dataset(df.dropna(subset=["retrieval_best"]))
    if sns is not None:
        sns.violinplot(data=sub, x="source_dataset", y="retrieval_best", order=order,
                       cut=0, ax=ax)
    else:
        ax.violinplot([sub.loc[sub["source_dataset"] == c, "retrieval_best"]
                       for c in order], positions=range(len(order)))
    _label_dataset_ticks(ax, order)
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
    # Top of the rect is 1.0: the reserve above it was for the suptitle this figure no
    # longer draws, and left in it is just a band of blank paper in the PDF.
    fig.tight_layout(rect=(0, 0.05, 0.96, 1.0), w_pad=3.5)
    return fig


def _fmt_p(p):
    """A p-value as a figure annotation: three decimals, or ``p<0.001``.

    Scientific notation (``2.6e-06``) is right in the console table and wrong on a
    figure — nobody compares exponents across rows at a glance, and the only claim
    the annotation has to carry is "far below the threshold".
    """
    if not np.isfinite(p):
        return "p=n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


# Metric pairs that measure the same thing under a condition this run happens to
# satisfy, and are therefore drawn as one row.
#
# ID-based context precision and recall are |gold ∩ retrieved| over |retrieved| and
# over |gold| respectively (evaluation/ragas_eval.py). Retrieval returns k=3 and
# every synthetic golden was built from exactly 3 chunks, so the two denominators
# are equal on all 136 scored rows and the metrics are algebraically identical --
# two rows of it would be one piece of evidence presented as two.
#
# The condition is a property of the RUN, not of the metrics: change k or the
# golden builder and they separate again. So this is a request to collapse, checked
# against the numbers at draw time by _collapse_equivalent, and silently ignored
# when they no longer agree.
EQUIVALENT_METRICS = [
    ("ragas_scores.ragas_id_context_precision", "ragas_scores.ragas_id_context_recall"),
]

# The stats that have to match before two metrics are called one row. Deliberately
# the whole visible row -- both panels, both counts -- and not just the mean.
_EQUIV_COLS = ("n_pairs", "n_nontied", "mean_diff", "ci_low", "ci_high",
               "wilcoxon_p", "rank_biserial")


def _collapse_equivalent(rows, pairs=EQUIVALENT_METRICS):
    """``[(metric, stats)]`` -> ``[(label, stats)]``, merging the equivalent pairs.

    A pair merges only if every stat in ``_EQUIV_COLS`` agrees, so the merge cannot
    quietly hide a run where the two metrics came apart: there the check fails and
    both rows are drawn, which is the outcome that wants looking at.
    """
    stats = dict(rows)
    merged_away, labels = set(), {}
    for keep, drop in pairs:
        if keep not in stats or drop not in stats:
            continue
        a, b = stats[keep], stats[drop]
        same = all(np.isclose(float(a[c]), float(b[c]), rtol=0, atol=1e-9,
                              equal_nan=True) for c in _EQUIV_COLS)
        if same:
            merged_away.add(drop)
            # "RAGAS Context Precision/Recall" — the shared prefix is not repeated.
            labels[keep] = f"{metric_label(keep)}/{metric_label(drop).split()[-1]}"
    return [(labels.get(m, metric_label(m)), r)
            for m, r in rows if m not in merged_away]


def paired_comparison_plot(df, comparisons, alpha=0.05):
    """The ``paired variant comparisons (Wilcoxon signed-rank)`` table, drawn.

    One row per (metric, variant pair) in ``comparisons`` — the same
    ``eval_analysis.VARIANT_COMPARISONS`` the table is printed from, so the figure
    and the table can never drift apart — and two panels per row, because the
    table's two effect columns answer different questions and are on different
    scales:

      - LEFT: the paired mean difference ``a − b`` with its 95% bootstrap CI, in
        the metric's own units. This is what "how much better" means to a reader,
        but it is only comparable down the column for metrics that share a scale.
      - RIGHT: the rank-biserial correlation, the matched-pairs effect size that
        belongs to the Wilcoxon test the section is named for. It is scale-free
        and bounded to [-1, 1], so it IS comparable down the column — a 0.06 mean
        gain on contextual relevance and a 0.078 one on RAGAS faithfulness are the
        same size in metric units and very different as effects.

    Each panel is coloured by ITS OWN verdict, which is the honest way to draw two
    statistics that can disagree: the left panel by whether the bootstrap CI
    clears zero, the right by whether the Wilcoxon p clears ``alpha``. Where a row
    is grey on one side and coloured on the other, the mean and the ranks are
    telling different stories and the row deserves the reader's attention rather
    than a single merged verdict.

    Rows are grouped by COMPARISON rather than kept in the table's metric-major
    order. Within a group every row is the same question asked of a different
    judge ("does self-correction change anything?"), which is exactly the
    comparison the eye should be making; interleaving the two comparisons invites
    reading a rag-vs-no_rag effect against a rag_sc-vs-rag one. Metric order
    inside a group follows ``comparisons``.

    Each row is annotated ``n = ranked pairs / all pairs``. The two differ by the
    ties, which the signed-rank test discards, and on a rail-pinned metric they
    differ by almost everything — so the second number sizes the pairing and the
    FIRST one sizes the test.

    Metric pairs listed in ``EQUIVALENT_METRICS`` share a row when their statistics
    actually coincide — see ``_collapse_equivalent``. The table upstream still
    prints them separately, on purpose: there the duplication IS the evidence that
    the two are the same measurement on this run, whereas on a figure it would read
    as two independent findings agreeing.

    Supersedes ``variant_effect_forest``, which draws the same mean differences off
    its own metric list and without the significance or effect-size columns; the
    two disagreeing about which metrics get compared is exactly how a comparison
    ends up in one figure and missing from the table.
    """
    apply_style()

    # Group the (metric, pair) rows by comparison, keeping the caller's metric
    # order inside each group. dict preserves first-seen order, so the groups come
    # out in the order the comparisons are first mentioned.
    groups = {}
    for metric, pairs in comparisons:
        if metric not in df:
            continue
        for a, b in pairs:
            r = ev.compare_variants(df, metric, a=a, b=b).loc["overall"]
            if not r.get("n_pairs"):
                continue
            groups.setdefault((a, b), []).append((metric, r))
    if not groups:
        raise ValueError("no comparison in `comparisons` has a non-empty pairing")

    # Every row of every group shares ONE pair of axes, with the groups separated by
    # a header slot rather than by a subplot boundary. Stacked subplots would be the
    # obvious layout and are the wrong one: their per-panel padding is a fixed number
    # of points, so a two-row group and a six-row group end up with different row
    # heights and different bar thicknesses, and bar thickness is the one thing on
    # this figure that must NOT carry meaning.
    HEADER, GAP = 0.85, 0.9
    slot, placed, headers = 0.0, [], []
    for gi, ((a, b), rows) in enumerate(groups.items()):
        if gi:
            slot += GAP
        # "SC-RAG vs. Naive RAG", never the reverse: the header names a and b in the
        # order the arithmetic subtracts them, so the reader maps a positive bar onto
        # the first name. Naming the pair the other way round silently inverts every
        # row in the group.
        headers.append((slot, f"{variant_label(a)}  vs.  {variant_label(b)}", gi > 0))
        slot += HEADER
        for label, r in _collapse_equivalent(rows):
            placed.append((-slot, label, r))
            slot += 1.0
    bottom = -(slot - 1.0)

    fig, (ax_diff, ax_eff) = plt.subplots(
        1, 2, sharey=True, figsize=(11.5, 0.42 * slot + 1.6))

    for ax in (ax_diff, ax_eff):
        ax.axvline(0, color=AXIS, lw=1)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(bottom - 0.6, 0.6)
    ax_diff.set_yticks([y for y, _, _ in placed])
    ax_diff.set_yticklabels([label for _, label, _ in placed])
    ax_eff.set_xlim(-1.05, 1.05)

    for y, sep_label, draw_rule in headers:
        if draw_rule:
            for ax in (ax_diff, ax_eff):
                ax.axhline(-y + GAP / 2, color=GRID, lw=1, zorder=1)
        ax_diff.annotate(sep_label, (0.0, -y), xycoords=("axes fraction", "data"),
                         va="center", ha="left", fontsize=10.5, color=INK_SECONDARY)

    for y, _, r in placed:
        spans_zero = r["ci_low"] <= 0 <= r["ci_high"]
        c = NEUTRAL if spans_zero else (POS if r["mean_diff"] > 0 else NEG)
        ax_diff.plot([r["ci_low"], r["ci_high"]], [y, y], color=c, lw=2,
                     solid_capstyle="butt", zorder=2)
        ax_diff.plot([r["mean_diff"]], [y], marker="o", ms=8, zorder=3,
                     color=SURFACE if spans_zero else c,
                     markeredgecolor=c, markeredgewidth=2)

        p, rb = r["wilcoxon_p"], r["rank_biserial"]
        sig = np.isfinite(p) and p < alpha
        ce = NEUTRAL if not sig else (POS if rb > 0 else NEG)
        # A bar, not a dot: the rank-biserial is a magnitude measured FROM zero, and
        # the length IS the quantity. Hollow where the test does not reject, so a
        # large-but-unsupported effect cannot be mistaken for a finding.
        ax_eff.barh(y, rb, height=0.5, zorder=2, color=ce if sig else SURFACE,
                    edgecolor=ce, linewidth=1.6)
        # BOTH counts, because on a rail-pinned metric they are worlds apart and
        # only the first one is the evidence behind this panel: DeepEval Relevance
        # pairs 273 questions and the signed-rank test ranks 26 of them. Annotating
        # n_pairs alone — as this figure first did — invites reading a large
        # rank-biserial off 26 pairs as if it stood on 273.
        ax_eff.annotate(f"{_fmt_p(p)}   n={int(r['n_nontied'])}/{int(r['n_pairs'])}",
                        (1.0, y), xycoords=("axes fraction", "data"),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=7.5, color=INK_MUTED,
                        annotation_clip=False)

    # "first − second", not "a − b": the group headers now name the variants, so the
    # letters no longer appear anywhere on the figure for these to refer back to.
    ax_diff.set_xlabel("paired mean difference, first − second (95% CI)")
    ax_eff.set_xlabel("rank-biserial effect size\n"
                      "(Wilcoxon signed-rank; n = ranked pairs / all pairs)")

    handles = [
        Line2D([], [], color=POS, lw=2, marker="o", ms=7,
               label="first named variant wins"),
        Line2D([], [], color=NEG, lw=2, marker="o", ms=7,
               label="second named variant wins"),
        Line2D([], [], color=NEUTRAL, lw=2, marker="o", ms=7,
               markerfacecolor=SURFACE, markeredgewidth=2,
               label=f"no detectable difference (CI spans zero / p ≥ {alpha:g})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01))
    # The legend reserve is a fixed 0.75 INCH converted to a fraction, not a fixed
    # fraction: the figure grows with the metric count, and 7% of a tall figure is a
    # band of blank paper the same legend did not need when it was short.
    # Right margin pulled in to 0.9: the p / n annotations hang outside the right
    # panel's axes, and tight_layout does not measure text it was told not to clip.
    fig.tight_layout(rect=(0, 0.75 / fig.get_figheight(), 0.9, 1.0), w_pad=1.2)
    return fig


def _rail_segments(ax, y, s, height=0.62, gap=0.004):
    """Draw one diverging 0.0 / spread / 1.0 stack per row of ``s`` at positions
    ``y``, and return the ``(zero, mid, one)`` shares as arrays.

    ``s`` is a ``metric_summary`` frame already in draw order. The composition is
    of SCORED rows, so it always sums to 1 — except where nothing was scored at
    all: ``frac_zero``/``frac_one`` are NaN there, and a plain ``fillna(0)`` would
    make ``mid`` come out as 1.0 and paint an empty cell as a metric with perfect
    spread. Rows with ``n == 0`` are therefore drawn as nothing, and the caller
    marks them.

    Shared by ``metric_rail_plot`` and ``metric_rail_grid`` so the pooled figure
    and its per-group panels are the same geometry down to the segment gap — two
    copies of this arithmetic is exactly how a panel stops being comparable to the
    headline figure without anyone noticing.

    ``gap`` is a true surface gap between segments rather than a drawn border, so
    the arms stay separable when one of them is a sliver.
    """
    n = pd.to_numeric(s["n"], errors="coerce").fillna(0).to_numpy()
    scored = n > 0
    z = np.where(scored, pd.to_numeric(s["frac_zero"], errors="coerce").fillna(0), 0.0)
    o = np.where(scored, pd.to_numeric(s["frac_one"], errors="coerce").fillna(0), 0.0)
    mid = np.where(scored, 1.0 - z - o, 0.0)
    ax.barh(y, mid, left=-mid / 2, height=height, color=GRID, zorder=2)
    ax.barh(y, np.maximum(z - gap, 0), left=-mid / 2 - z, height=height,
            color=NEG, zorder=2)
    ax.barh(y, np.maximum(o - gap, 0), left=mid / 2 + gap, height=height,
            color=POS, zorder=2)
    return z, mid, o


def metric_rail_plot(df, metrics=None, min_scored=1):
    """Which metrics actually discriminate, and which are pinned to a rail.

    Each metric is one diverging stacked bar centred on its SPREAD: the gray
    middle is the share of scored rows strictly between 0 and 1 — the part of the
    metric that can rank anything — and the two arms are the mass welded to the
    rails, 0.0 to the left, 1.0 to the right.

    Rows run in ``METRIC_ORDER``, not sorted by spread. Sorting would put the
    sharpest metric on top and read as a ranking, which is the wrong claim for a
    validation figure: the point is not which metric wins but whether each one can
    separate anything at all, and that is legible from the bar itself. A fixed
    order also means this figure and every other metric list in the chapter can be
    read against each other row by row.

    This is the figure behind the metric-validation argument. A metric that is
    90% one colour is not measuring your system, and its mean elsewhere in the
    chapter cannot be read as quality — which is invisible in a table of means
    and unmissable here.
    """
    apply_style()
    s = ev.metric_summary(df, metrics)
    s = s[s["n"] >= min_scored].copy()
    # barh draws index 0 at the bottom, so reverse: first in METRIC_ORDER = top row.
    s = s.reindex(order_metrics(list(s.index))[::-1])

    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(s) + 2.0))
    y = np.arange(len(s))
    z, mid, o = _rail_segments(ax, y, s)

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
        Line2D([], [], color=GRID, lw=7, label="between 0.0 and 1.0 (exclusive)"),
        Line2D([], [], color=POS, lw=7, label="scored exactly 1.0"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    return fig


def _group_label(value, key):
    """The name a grouping value goes by on a panel — display names for the two keys
    that have them, the raw value for anything else."""
    if key == "variant":
        return variant_label(value)
    if key == "source_dataset":
        return dataset_label(value)
    return str(value)


def _group_order(values, key):
    """Panel order for one grouping level: the chapter's dataset order, the pipeline
    order for variants (no_rag -> rag -> rag_sc), alphabetical for anything else."""
    vals = list(dict.fromkeys(str(v) for v in values))
    if key == "variant":
        known = [v for v in ev.VARIANT_ORDER if v in vals]
        return known + [v for v in vals if v not in known]
    if key == "source_dataset":
        return dataset_order(vals)
    return sorted(vals)


def metric_rail_grid(df, by: "str | list[str]" = "source_dataset", metrics=None,
                     fontscale=1.3):
    """``metric_rail_plot`` as small multiples — one panel per group, so you can see
    WHERE a metric discriminates instead of only whether it does on average.

    This is the picture of the ``per dataset x variant`` block in the report, and it
    exists because that table is 165 rows of numbers whose message is a shape. Pass
    ``by="source_dataset"`` or ``by="variant"`` for a single row of panels, or
    ``by=["source_dataset", "variant"]`` for the full grid: the FIRST key varies
    across columns and the second down rows, so the variant contrast — the one the
    thesis argues about — is read vertically inside one dataset, against a fixed
    metric row.

    Every panel carries every metric in ``METRIC_ORDER``, present or not, and the
    bars are the same geometry as the pooled figure (``_rail_segments``), so panels
    are readable against each other and against the headline plot row by row. Two
    things a single panel cannot say on its own are therefore annotated:

      - the per-metric ``n`` at the right of each bar. A composition is a share of
        SCORED rows, and "100% pinned at 1.0" off 2 rows is not the same statement as
        the same bar off 350. In a per-group cut those n's fall by an order of
        magnitude, which is precisely when the shares start to lie.
      - a muted dash at the centre where a metric has no scored rows at all, so an
        empty row reads as "not scored here" rather than "no spread". Most of those
        are legitimate — faithfulness needs a retrieved context, the ``id_context_*``
        metrics need a gold reference set — and the report's error table, not this
        figure, is what separates those from real failures.

    ``fontscale`` is the readability lever, and it moves type size rather than canvas
    size on purpose. At the full grid this is 165 bars, and whoever reads it reads it
    at some fixed width — a thesis page, a screen — so growing the canvas there makes
    the type SMALLER relative to everything else, which is the opposite of the fix.
    This grows the type, and with it only the geometry type actually occupies: the
    metric-name column on the left, and the row pitch once the labels would otherwise
    touch. The bars keep their width, so the canvas grows far less than the type does
    and the ratio that decides legibility improves. Raise it further if the grid
    still reads small.
    """
    apply_style()
    keys = [by] if isinstance(by, str) else list(by)
    if len(keys) > 2:
        raise ValueError(f"metric_rail_grid takes at most 2 grouping keys, got {keys}")
    s = ev.metric_summary_by(df, keys, metrics)
    if s.empty:
        raise ValueError(f"no rows to group by {keys}")

    # barh draws index 0 at the bottom, so reverse: first in METRIC_ORDER = top row.
    mets = order_metrics(
        list(dict.fromkeys(s.index.get_level_values("metric"))))[::-1]
    cols = _group_order(s.index.get_level_values(keys[0]), keys[0])
    rows = _group_order(s.index.get_level_values(keys[1]), keys[1]) if len(keys) > 1 \
        else [None]
    sizes = {tuple(str(v) for v in (g if isinstance(g, tuple) else (g,))): int(n)
             for g, n in df.groupby(keys, observed=True).size().items()}

    # Every size on this figure derives from ``fontscale`` so the proportions hold
    # at any setting: the four type sizes directly, and the two lengths that have to
    # follow type — the left label column, and the row pitch, which only starts
    # growing once a line of type no longer fits in the 0.24in default.
    fs = float(fontscale)
    pt_tick, pt_n, pt_head, pt_foot = 8.5 * fs, 7.8 * fs, 10.0 * fs, 8.5 * fs
    row_h = max(0.255, 0.0175 * pt_tick)
    fig, axes = plt.subplots(
        len(rows), len(cols), sharex=True, squeeze=False,
        figsize=(2.30 * len(cols) + 2.1 * fs,
                 row_h * len(mets) * len(rows) + 1.20 + 0.55 * fs))
    y = np.arange(len(mets))

    for ri, rv in enumerate(rows):
        for ci, cv in enumerate(cols):
            ax = axes[ri][ci]
            key = (cv,) if rv is None else (cv, rv)
            try:
                sub = s.loc[key].reindex(mets)
            except KeyError:  # a combination that never occurs (empty panel)
                sub = pd.DataFrame(float("nan"), index=mets, columns=s.columns)
            _rail_segments(ax, y, sub, height=0.70)

            n = pd.to_numeric(sub["n"], errors="coerce").fillna(0).to_numpy()
            for yi, ni in zip(y, n):
                if ni > 0:
                    # INK_SECONDARY, not INK_MUTED: this n is what stops a bar being
                    # read as a share when it is a share of three rows, so it has to
                    # survive being glanced past — it is data, not furniture.
                    ax.annotate(f"{int(ni)}", (1.0, yi),
                                xycoords=("axes fraction", "data"), xytext=(4, 0),
                                textcoords="offset points", va="center", ha="left",
                                fontsize=pt_n, color=INK_SECONDARY,
                                annotation_clip=False)
                else:
                    ax.plot([0], [yi], marker="_", ms=6, color=AXIS, zorder=2)

            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-0.6, len(mets) - 0.4)
            ax.set_yticks(y)
            ax.set_yticklabels([metric_label(m) for m in mets] if ci == 0 else [],
                               fontsize=pt_tick)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=pt_tick)
            ax.grid(axis="y", visible=False)
            ax.set_xticks([-1, 0, 1])
            ax.set_xticklabels(["100%", "0", "100%"])
            if ri == 0:
                ax.set_title(_group_label(cv, keys[0]), fontsize=pt_head)
            if ci == 0 and rv is not None:
                # INK, overriding axes.labelcolor: this is not an axis label, it is the
                # row header, and it names a group exactly as the column titles above
                # do. The two carry the same weight in the grid, so they take the same
                # ink — matplotlib only calls it a ylabel because that is where it sits.
                ax.set_ylabel(_group_label(rv, keys[1]), fontsize=pt_head,
                              color=INK, labelpad=8)
            # Rows in the group, so a per-metric n beside each bar can be read as
            # coverage. Drawn on the top band only, where it reads as part of the
            # column header. Every variant sees the same question set, so down a
            # column this number is the same three times over and repeating it costs
            # three lines of vertical room to say one thing. If a grouping is ever
            # added where the column total does vary by row, this needs revisiting.
            if ri == 0:
                ax.text(1.0, 1.0, f"n={sizes.get(key, 0)}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=pt_n, color=INK_MUTED)

    handles = [
        Line2D([], [], color=NEG, lw=7, label="scored exactly 0.0"),
        Line2D([], [], color=GRID, lw=7, label="strictly between (the usable signal)"),
        Line2D([], [], color=POS, lw=7, label="scored exactly 1.0"),
        Line2D([], [], color=AXIS, lw=0, marker="_", ms=8, label="not scored in this group"),
    ]
    # The legend is placed in INCHES converted to figure fractions, not in fractions
    # directly: this figure is 4.6in tall grouped one way and 10in tall grouped
    # another, and a fixed bottom margin that clears the legend on the tall one eats
    # a panel row on the short one. The top of the rect is 1.0 — the band that used
    # to hold the suptitle is not reserved now that the figure does not draw one, and
    # neither is the band that used to hold the axis caption ("share of scored rows
    # ← pinned at 0.0 · spread · pinned at 1.0 →, grey number = scored rows for that
    # metric"), which is captioned in the thesis text instead.
    h = fig.get_size_inches()[1]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.10 / h), fontsize=pt_foot)
    # ``w_pad`` has to clear the per-metric n's, which hang off the right edge of each
    # panel as unclipped annotations tight_layout does not measure — but only just.
    # Every point of gap beyond that is width spent on nothing, and on a figure read
    # at a fixed page width, width spent on nothing is what makes the type look small.
    fig.tight_layout(rect=(0, (0.30 + 0.09 * fs) / h, 0.995, 1.0),
                     w_pad=1.2, h_pad=0.5)
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

    Rows run in ``DATASET_ORDER`` under a pooled ``all datasets`` row — the same
    leading reference every per-dataset figure carries. That row is the mean over the
    ROWS, not the mean of the five dataset means, so an uneven split cannot skew it.
    """
    apply_style()
    sub = df.copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    piv = sub.pivot_table(index="source_dataset", columns="variant",
                          values=metric, observed=True, aggfunc="mean")
    piv = piv.reindex(columns=[v for v in ev.VARIANT_ORDER if v in piv.columns])
    counts = sub.groupby("source_dataset", observed=True)["id"].nunique()
    piv.index, counts.index = piv.index.astype(str), counts.index.astype(str)
    piv.loc[_POOLED_LABEL] = sub.groupby("variant", observed=True)[metric].mean()
    counts.loc[_POOLED_LABEL] = sub["id"].nunique()
    piv = piv.loc[dataset_order(piv.index)]

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
    ax.set_yticklabels([f"{dataset_label(d)}  (n={int(counts.get(d, 0))})"
                        for d in piv.index])
    if piv.index[0] == _POOLED_LABEL:
        # Drawn INTO the surface spacer above, so the pooled row reads as the reference
        # rather than as a sixth dataset. Thin enough to leave the gap either side.
        ax.axhline(0.5, color=AXIS, lw=1.5)
    ax.set_title(f"{metric_label(metric)} by dataset and variant")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.outline.set_visible(False)
    # Same split as the axis ticks: muted marks, readable labels.
    cbar.ax.tick_params(color=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    fig.tight_layout()
    return fig


def abstention_grouped_bars(df, variants=("rag", "rag_sc")):
    """How often the system refuses instead of answering, per dataset and for the run.

    Abstention is a reliability behaviour, not a failure: refusing when the
    corpus does not cover a question is the intended response, and the gap
    between datasets is the finding. ``no_rag`` is excluded because it has no
    context to call insufficient and cannot abstain by construction — plotting
    its structural zero would imply a comparison that does not exist.

    The pair takes ``PAIRED_STEPS`` rather than the categorical ``VARIANT_COLORS`` slots:
    ``rag -> rag_sc`` is a progression (the same pipeline plus a correction loop), not
    two unrelated entities, and these are the same two steps the abstention box figures
    give ``answered``/``abstained``, so the three abstention figures read as one family.
    Note the trade: on this figure rag is no longer orange, so it does not colour-match
    the variant-effect plots.

    Every bar is direct-labelled, which is also what discharges the contrast
    relief rule the palette check flagged for slot 3.

    No title — it belongs in the document's own caption, and it said: abstention rate by
    dataset and variant. The legend, the ``share of abstained answers`` axis, the ``n=``
    ticks and the per-bar percentages stay, since those are what the bars are rather
    than what to make of them.
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
    rate.index, n.index = rate.index.astype(str), n.index.astype(str)

    # The pooled group leads, as the reference the datasets are read against. It is
    # computed over the ROWS, not averaged over the five dataset rates — the datasets
    # differ in size by a factor of three, so the mean of the rates is not the run's
    # rate. Being the same rows again, it sits past a gap and a rule.
    pooled = ra.abstention_summary(sub, by="variant", with_total=False)
    rate.loc[_POOLED_LABEL] = pooled["abstention_rate"].reindex(rate.columns)
    n.loc[_POOLED_LABEL] = pooled["n"].reindex(rate.columns)
    order = dataset_order(rate.index)
    rate, n = rate.loc[order], n.loc[order]

    fig, ax = plt.subplots(figsize=(1.6 * len(rate) + 3.4, 4.8))
    x = np.append([0.0], np.arange(len(rate) - 1) + 1.6)
    width = 0.8 / max(len(rate.columns), 1)
    fills = (PAIRED_STEPS if len(rate.columns) == 2
             else _ordered_fills(len(rate.columns)))
    for k, v in enumerate(rate.columns):
        off = (k - (len(rate.columns) - 1) / 2) * width
        color = fills[k]
        vals = rate[v].to_numpy(dtype=float)
        ax.bar(x + off, vals, width * 0.92, label=variant_label(v), color=color, zorder=2)
        for xi, val in zip(x + off, vals):
            if not np.isfinite(val):
                continue
            # A label only goes inside the bar when it fits; a tall bar would
            # otherwise push its label off the top of the axes.
            inside = val > 0.90
            ax.text(xi, val - 0.02 if inside else val + 0.015, f"{val:.0%}",
                    ha="center", va="top" if inside else "bottom", fontsize=7.5,
                    color=_ink_on(color) if inside else INK_SECONDARY, zorder=3)

    ax.axvline((x[0] + x[1]) / 2, color=AXIS, lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{dataset_label(d)}\n(n={int(n.loc[d].max())})"
                        for d in rate.index])
    ax.set_xlim(-0.7, x[-1] + 0.7)
    ax.set_ylabel("share of abstained answers")
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", visible=False)
    # Above the plot, so it can never sit on top of a bar — it kept the title company
    # there and stays put now that the title is gone.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=len(rate.columns))
    fig.tight_layout()
    return fig


# Answered vs abstained is a split WITHIN a variant, not a new entity, so it gets a
# light/dark pair of one hue instead of two categorical slots — which also leaves
# "rag is orange" intact on a figure whose x axis is the variants themselves. The same
# pair as ``abstention_grouped_bars``, from the one constant, so the three abstention
# figures cannot drift apart.
STATUS_COLORS = dict(zip(("answered", "abstained"), PAIRED_STEPS))

# ``signal -> (column, y label, title, caption)``. One figure EACH, not two panels
# of one: both are full-width thesis figures, and side by side they shrink past
# readable in print. Both columns come from ``rag_analysis.ABSTENTION_SIGNAL_COLS``,
# whose table prints the same means.
_BOX_CAPTION = ("Boxes: median, quartiles, 5th–95th percentile; diamond = mean. "
                "One dot per row, jittered horizontally. ")
ABSTENTION_SIGNALS = {
    # Title and caption are empty on purpose: this figure is captioned in the thesis
    # text instead. What used to be printed here was "Retrieval scores when the system
    # abstained vs answered" plus _BOX_CAPTION and "An abstained box below the answered
    # one means refusals concentrate where the retrieved context was weak."
    "retrieval": (
        "retrieval_average", "mean retrieval score of all three chunks per instance", "", ""),
    # Blank for the same reason. What used to be printed here was "Generation
    # confidence when the system abstained vs answered" plus _BOX_CAPTION and "The
    # higher abstained boxes reflect the short formulaic refusal string, whose tokens
    # are trivially predictable — not model certainty about the question."
    "confidence": (
        "gen_logprob_stats.mean", "mean generation token logprob", "", ""),
}


def abstention_signal_boxes(df, signal="retrieval", variants=("rag", "rag_sc")):
    """Does one of the pipeline's own signals explain WHEN it refused? Per variant,
    the rows that answered and the rows that abstained as two distributions.

    The picture of ``rag_analysis.abstention_signals``, which prints the same means.
    Distributions rather than bars, because the claim is about which rows abstain,
    and two cohorts can differ in mean while overlapping almost completely — a gap
    you can only be shown, not told.

    ``signal`` picks one entry of ``ABSTENTION_SIGNALS``, and the two read in
    opposite directions:

      - ``retrieval``: an abstained box sitting BELOW the answered one is the
        intended behaviour — the system refuses when the corpus does not cover the
        question. ``retrieval_average`` (not ``retrieval_best``) is plotted because
        it judges the whole context: one lucky chunk can lift a best score the model
        then sees surrounded by noise. For rag_sc these are the post-re-retrieval
        scores, i.e. the context the answer was actually written from.
      - ``confidence``: the abstained box sits HIGHER (closer to 0), and that is NOT
        the model being sure of itself. The canonical refusal is a short formulaic
        sentence whose tokens are trivially predictable, so mean logprob measures
        string predictability here, not certainty about the question. It matters
        because the generation-correction trigger keys on exactly this number, which
        is why an abstaining first pass almost never trips a regeneration.

    ``no_rag`` is excluded by default: with no context to call insufficient it cannot
    abstain, so it has no abstained cohort to compare against.
    """
    apply_style()
    col, ylabel, title, caption = ABSTENTION_SIGNALS[signal]
    sub = df[df["variant"].astype(str).isin(variants)].copy()
    sub["_variant"] = sub["variant"].astype(str)
    sub["_status"] = np.where(ra._abstained(sub), "abstained", "answered")
    order = [v for v in variants if v in set(sub["_variant"])]
    statuses = ["answered", "abstained"]

    if not len(sub) or not order:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title("no rag / rag_sc rows in this file")
        return fig

    fig, ax = plt.subplots(figsize=(1.9 * len(order) + 2.8, 4.9))
    vals = (pd.to_numeric(sub[col], errors="coerce") if col in sub
            else pd.Series(np.nan, index=sub.index))
    for k, st in enumerate(statuses):
        data, pos, means, tops = [], [], [], []
        for i, v in enumerate(order):
            s = vals[(sub["_variant"] == v) & (sub["_status"] == st)].dropna()
            if not len(s):
                continue
            data.append(s.to_numpy())
            pos.append(i + (k - 0.5) * 0.34)
            means.append(s.mean())
            tops.append(np.percentile(s, 95))
        if not data:
            continue
        # The raw rows behind each box, jittered. Not decoration: the abstained
        # logprob cohorts pile up so tightly at 0 that their box collapses to a
        # line and disappears — the points are the only thing that still shows a
        # cohort is there, and they show WHERE it piles up.
        rng = np.random.default_rng(0)
        for x, s in zip(pos, data):
            ax.plot(x + rng.uniform(-0.09, 0.09, len(s)), s, ls="none", marker="o",
                    ms=2.4, alpha=0.35, color=STATUS_COLORS[st], zorder=1)
        bp = ax.boxplot(data, positions=pos, widths=0.28, patch_artist=True,
                        showfliers=False, whis=(5, 95), zorder=2)
        for box in bp["boxes"]:
            box.set(facecolor=STATUS_COLORS[st], edgecolor=INK_SECONDARY, linewidth=0.8)
        for part in ("whiskers", "caps"):
            for line in bp[part]:
                line.set(color=INK_SECONDARY, linewidth=0.8)
        # The median rule has to stay legible on both fills, so it takes the
        # surface colour on the dark box and ink on the light one.
        for line in bp["medians"]:
            line.set(color=SURFACE if st == "abstained" else INK, linewidth=1.6)
        # The mean is what the printed table reports; marking it keeps the
        # figure and abstention_signals readable against each other.
        ax.plot(pos, means, ls="none", marker="D", ms=4.5, color=INK,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
        # Above the upper cap rather than beside the diamond: a value printed on
        # top of the dark fill is unreadable, and a near-degenerate box (the
        # abstained logprobs, which pile up at 0) leaves no room inside anyway.
        for x, m, t in zip(pos, means, tops):
            ax.annotate(f"{m:.2f}", (x, t), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=INK_SECONDARY, zorder=4)

    # Headroom for those labels: offset text does not enter the autoscale.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.10 * (hi - lo))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([
        "{}\n{} answered / {} abstained".format(
            variant_label(v),
            int(((sub["_variant"] == v) & (sub["_status"] == "answered")).sum()),
            int(((sub["_variant"] == v) & (sub["_status"] == "abstained")).sum()))
        for v in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(ylabel)
    # An empty ``title`` means the caller captions this one in prose; the missing-column
    # note is a diagnostic, not decoration, so it is drawn either way.
    if col not in sub:
        ax.set_title(f"{col} not in this file", pad=26)
    elif title:
        ax.set_title(title, pad=26)
    ax.grid(axis="x", visible=False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STATUS_COLORS[s],
                             edgecolor=INK_SECONDARY, lw=0.8, label=s)
               for s in statuses]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2)
    if caption:
        fig.text(0.01, 0.005, caption, ha="left", fontsize=7.5, color=INK_MUTED,
                 wrap=True)
    fig.tight_layout(rect=(0, 0.06 if caption else 0, 1, 1))
    return fig


def _composition_bars(ct, title, caption, empty_note, colors=None, min_label_share=0.05):
    """One 100%-stacked bar per dataset — the shared body of every composition figure
    in this module.

    ``ct`` is a ``group x bucket`` count table in the ``sc_retry_breakdown`` shape:
    buckets already in stack order (bottom to top, "least of the thing" first) and an
    ``ALL`` row pooling every row. Each of these started life as a single donut over the
    whole run, and each turned out to be an average over datasets that behave nothing
    alike — so the ring became the leading ``all datasets`` bar and the datasets got
    their own, which is the comparison the figure is actually for.

    Bars are normalised to each group's OWN rows, because the datasets differ in size by
    a factor of three (``n`` rides on the tick) and raw counts would mostly redraw that.
    Bucket order is the caller's, never by size: in a 100%-stacked bar only the bottom
    and top segments share a baseline across bars and can be compared by eye, and the
    "nothing fired" / "everything fired" buckets — the two worth comparing — land there.
    The middle ones are read from their labels. Bars run in ``DATASET_ORDER``, the same
    order every per-dataset figure uses.

    Segments below ``min_label_share`` are left unlabelled rather than overprinted; their
    exact counts are in the table behind the figure.

    ``title`` and ``caption`` may be empty (or ``None``) to draw neither, for a figure
    whose explanation belongs in the document's own ``\\caption`` instead of burnt into
    the PNG. That covers those two texts and nothing else: the legend, the y-axis label,
    the ``n=`` tick labels and the in-bar percentages are what makes the bars readable at
    all and are always drawn. Nothing is reserved for what is not drawn — a dropped
    caption gives its strip back to the bars rather than leaving a blank band at the foot
    of the figure. ``empty_note`` is unaffected: it is what the figure says when there is
    no data, and is the whole content in that case.
    """
    apply_style()
    totals = ct.sum(axis=1) if len(ct) and len(ct.columns) else pd.Series(dtype="int64")
    ct = ct[totals > 0] if len(totals) else ct.iloc[0:0]
    if not len(ct):
        fig, ax = plt.subplots(figsize=(7, 3))
        if title:
            ax.set_title(title)
        ax.text(0.5, 0.5, empty_note, ha="center", va="center", color=INK_MUTED)
        ax.set_axis_off()
        return fig

    buckets = list(ct.columns)
    colors = colors or _ordered_fills(len(buckets))
    share = ct.div(ct.sum(axis=1), axis=0)
    order = dataset_order([i for i in share.index if i != "ALL"])
    pooled = [i for i in share.index if i == "ALL"]
    share, totals = share.loc[pooled + order], totals.loc[pooled + order]
    # The pooled bar leads, as the reference the datasets are read against — but it is
    # the same rows counted again, so a gap and a rule keep it out of their sequence.
    xs = np.append(np.zeros(len(pooled)), np.arange(len(order)) + len(pooled) + 0.6)

    fig, ax = plt.subplots(figsize=(1.35 * len(xs) + 3.0, 5.4))
    bottom = np.zeros(len(xs))
    for bucket, color in zip(buckets, colors):
        vals = share[bucket].to_numpy(dtype=float)
        ax.bar(xs, vals, 0.68, bottom=bottom, color=color, label=str(bucket), zorder=2)
        ink = _ink_on(color)
        for x, v, b in zip(xs, vals, bottom):
            if v >= min_label_share:
                ax.text(x, b + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color=ink, zorder=3)
        bottom += vals

    if len(pooled):
        ax.axvline((xs[0] + xs[1]) / 2, color=AXIS, lw=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{_POOLED_LABEL if i == 'ALL' else dataset_label(i)}\n(n={int(totals[i])})"
         for i in share.index])
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    # "share of n", not "share of the dataset's rag_sc rows": n is on every tick right
    # below the bar it belongs to, so the axis can point at it instead of restating it.
    ax.set_ylabel("share of n")
    ax.grid(axis="x", visible=False)
    if title:
        # The pad clears the legend, which sits just above the axes.
        ax.set_title(title, pad=26)
    # Reversed, so the legend reads left to right in the order the stack reads top to
    # bottom — a legend in bucket order would run against every bar in the figure.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower left", bbox_to_anchor=(0, 1.005),
              ncol=len(labels))
    if caption:
        fig.text(0.01, 0.012,
                 f"{caption}\nBars are each dataset's own rag_sc rows, not the run; "
                 f"segments under {min_label_share:.0%} are left unlabelled. The first "
                 "bar pools the same rows and is the run-wide composition.",
                 ha="left", fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.07 if caption else 0, 1, 1))
    return fig


def retry_kind_bars(df):
    """Which correction budgets each rag_sc row spent, per dataset —
    ``rag_analysis.sc_retry_breakdown(df, by="source_dataset")``.

    ``RETRY_KINDS`` is already ordered by how much correction the row spent, so it is
    the stack order as it stands. The two single-trigger kinds are NOT ranked against
    each other — they take adjacent steps of the ramp rather than implying that a
    generation retry is "more" than a retrieval one.

    Title and caption are empty, as in ``trigger_combination_bars``: the figure carries
    only what is read off it and the words belong in the document's own caption. What
    that caption has to say: retrieval correction = HyDE re-retrieval, generation
    correction = strict-prompt regeneration, ``both`` = a row that spent each once, and
    ``none`` = a row that ran the plain rag pipeline unchanged. Bars are each dataset's
    own rag_sc rows (n on the tick), the first bar pools them all, and segments under 5%
    are left unlabelled.
    """
    return _composition_bars(
        ra.sc_retry_breakdown(df, by="source_dataset").reindex(
            columns=ra.RETRY_KINDS, fill_value=0),
        "", "", "no rag_sc rows in this file")


def trigger_combination_bars(df, stage="retrieval"):
    """Which SET of ``stage``'s thresholds tripped, per dataset —
    ``rag_analysis.trigger_combinations(df, stage, by="source_dataset")``.

    The companion table answers "how many rows fired only ``highest``, only ``spread``,
    or both"; the bars answer the part-to-whole question behind it, per dataset: of each
    cohort, how much of the correction budget did each threshold claim on its own, and
    how often did they agree.

    Buckets arrive in ``trigger_combination_order`` — nothing fired, one threshold, then
    several — so the ramp means "how much tripped" and the two ends of the stack are the
    two things worth comparing across datasets.

    Title and caption are deliberately empty, so the figure carries only what has to be
    read off it (legend, axis, ``n=`` per bar, segment shares) and the words go in the
    document's own caption. What that caption has to say, since it is no longer on the
    figure: each rag_sc row counts ONCE, in the bucket for the SET of thresholds it
    tripped, so the segments partition the cohort — ``none`` means the correction never
    fired, and a row tripping both thresholds is one row in ``highest & spread`` here but
    two firings in the per-trigger table. Bars are each dataset's own rag_sc rows (n on
    the tick), the first bar pools them all, and segments under 5% are left unlabelled.
    """
    return _composition_bars(
        ra.trigger_combinations(df, stage=stage, by="source_dataset"),
        "", "", f"no {stage} trigger data in this file")


# Words in a ``METRIC_PAIRS`` name that are not ordinary words: two brandmarks, an
# acronym, and the join, which stays lowercase because "RAGAS Vs DeepEval" reads as
# a typo. Everything else title-cases, same rule as ``metric_label``.
_PAIR_WORDS = {"ragas": "RAGAS", "deepeval": "DeepEval", "hhem": "HHEM", "vs": "vs"}


def _pair_label(name):
    """``faithfulness: ragas vs hhem`` -> ``Faithfulness: RAGAS vs HHEM``.

    The pair names are written once in ``eval_analysis.METRIC_PAIRS``, where they are
    also table keys and want to stay greppable lowercase — so the display form is
    derived here rather than stored, exactly as metric column names are.
    """
    out = []
    for w in str(name).split():
        bare, tail = (w[:-1], w[-1]) if w[-1:] in ":,." else (w, "")
        out.append(_PAIR_WORDS.get(bare.lower(), bare.capitalize()) + tail)
    return " ".join(out)


def metric_agreement_grid(ag, key="source_dataset", stats=("spearman", "pearson"),
                          fontscale=1.15):
    """``metric_agreement_dots``, one panel per group: does the agreement hold everywhere?

    The pooled figure answers whether two comparable metrics agree at all. This one
    answers where, which is the question that decides what a disagreement means. A
    pair that correlates in one dataset and not another is not two broken judges —
    it is one construct that only survives on some kinds of question, and the
    faceting is what separates those two readings.

    Takes the per-group table from ``eval_analysis.metric_agreement(d, by=...)``, so
    the figure and the table in the report cannot drift apart. ``key`` names the
    column(s) it was grouped on: one key gives a row of panels, and a pair of keys
    (``["source_dataset", "variant"]``, matching ``metric_rail_grid``) gives the full
    grid — first key across, second down. The crossed table has to have been built
    with the same list, since that is what puts the two keys in their own columns.

    A row with no dot carries a tick on the zero line instead, so an UNDEFINED
    correlation never reads as "zero agreement". Undefined has two causes and the
    ``n`` beside the tick separates them: no n at all means the pair had no rows
    here, while a tick sitting next to a large n means one of the two metrics was
    CONSTANT across those rows — every value identical, so there is no ranking to
    correlate. The second is a finding, not a gap: it is what a metric looks like
    when a whole cohort pins it to one value.

    That ``n`` is a paired n — rows where BOTH metrics scored — and on the crossed
    grid it is the figure's most important number: splitting five datasets three ways
    leaves some cells with a handful of rows, and a correlation over a handful of
    rows is noise that happens to have a decimal point.

    Carries only its legend, its axis and its group headers; the words go in the
    document's caption. The legend states the shape, the colour and the dash, so all
    the caption still owes the reader is the number: it is the paired rows behind
    that row's dots — rows where both metrics scored — and beside a dash it is what
    says whether the correlation is undefined for want of rows or for want of
    variance.
    """
    apply_style()
    tbl = ag[ag["group"] != "overall"] if "group" in ag else ag
    keys = [key] if isinstance(key, str) else list(key)
    pairs = list(dict.fromkeys(tbl["pair"]))
    # One key reads the ``group`` column, which IS that key's value; two keys read the
    # per-key columns the crossed table carries, rather than splitting ``group`` back
    # apart on its separator — a dataset whose name contained one would break that.
    crossed = len(keys) > 1 and all(k in tbl for k in keys)
    cols = _group_order(tbl[keys[0]] if crossed else tbl["group"], keys[0])
    rows = _group_order(tbl[keys[1]], keys[1]) if crossed else [None]
    if not len(tbl) or not pairs or not cols:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title("no per-group metric agreement in this file")
        return fig

    def cell(rv, cv):
        if not crossed:
            return tbl[tbl["group"].astype(str) == str(cv)]
        return tbl[(tbl[keys[0]].astype(str) == str(cv))
                   & (tbl[keys[1]].astype(str) == str(rv))]

    # Same sizing contract as ``metric_rail_grid``: type size is the readability
    # lever, and only the geometry that type occupies follows it — the pair-name
    # column on the left, and the row pitch, which here also has to hold two markers
    # nudged apart vertically rather than one line of text.
    fs = float(fontscale)
    pt_tick, pt_n, pt_head, pt_foot = 8.5 * fs, 7.8 * fs, 10.0 * fs, 8.5 * fs
    row_h = max(0.42, 0.045 * pt_tick)
    fig, axes = plt.subplots(
        len(rows), len(cols), sharex=True, sharey=True, squeeze=False,
        figsize=(1.95 * len(cols) + 2.6 * fs,
                 row_h * len(pairs) * len(rows) + 1.25 + 0.55 * fs))
    y = np.arange(len(pairs))[::-1]
    pos = dict(zip(pairs, y))
    markers = {"spearman": "o", "pearson": "s"}
    # Two statistics that usually land close together, nudged apart rather than
    # hidden behind one another — the row rule still ties both to their label.
    offsets = np.linspace(0.15, -0.15, len(stats)) if len(stats) > 1 else [0.0]

    for ri, rv in enumerate(rows):
        for ci, cv in enumerate(cols):
            ax = axes[ri][ci]
            ax.axvline(0, color=AXIS, lw=1)
            for row in cell(rv, cv).itertuples():
                yi = pos.get(row.pair)
                if yi is None:
                    continue
                n = int(row.n) if np.isfinite(row.n) else 0
                drawn = False
                for stat, dy in zip(stats, offsets):
                    v = getattr(row, stat, np.nan)
                    if v is None or not np.isfinite(v):
                        continue
                    ax.plot([v], [yi + dy], marker=markers.get(stat, "o"),
                            ms=7.5 * fs, color=POS if v > 0 else NEG, zorder=3,
                            ls="none", markeredgecolor=SURFACE, markeredgewidth=1.4)
                    drawn = True
                if not drawn:
                    ax.plot([0], [yi], marker="_", ms=9, color=INK_SECONDARY,
                            zorder=4)
                if n:
                    # INK_SECONDARY, not INK_MUTED: a correlation over three rows and
                    # one over three hundred are the same dot, and this is the only
                    # thing separating them — data, not furniture.
                    ax.annotate(f"{n}", (1.0, yi), xycoords=("axes fraction", "data"),
                                xytext=(4, 0), textcoords="offset points",
                                va="center", ha="left", fontsize=pt_n,
                                color=INK_SECONDARY, annotation_clip=False)

            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-0.6, len(pairs) - 0.4)
            ax.set_yticks(y)
            # Only the first column labels, and the others are left ALONE rather than
            # given an empty list: ``sharey`` shares one formatter, so blanking it on
            # a later panel blanks the first panel too. Sharing already hides them.
            if ci == 0:
                ax.set_yticklabels([_pair_label(p) for p in pairs], fontsize=pt_tick)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=pt_tick)
            ax.set_xticks([-1, 0, 1])
            ax.set_xticklabels(["\N{MINUS SIGN}1", "0", "+1"])
            ax.grid(axis="y", visible=True)
            if ri == 0:
                ax.set_title(_group_label(cv, keys[0]), fontsize=pt_head)
            if ci == 0 and rv is not None:
                # INK, overriding axes.labelcolor: this is the row header naming a
                # group, exactly as the column titles do, so it takes the same ink.
                # It sits outside the pair names, which are the y ticks.
                ax.set_ylabel(_group_label(rv, keys[1]), fontsize=pt_head, color=INK,
                              labelpad=8)
            if ri == len(rows) - 1 and ci == len(cols) // 2:
                ax.set_xlabel("correlation between the two metrics", fontsize=pt_tick)

    # The legend carries all three of the figure's encodings, not just the shape one:
    # marker = which statistic, colour = the sign, dash = no correlation exists. The
    # colour swatches are plain line segments rather than markers on purpose — giving
    # them a circle or a square would collide with the two shapes that mean something
    # else. What the dash means is deliberately "undefined" and not "not scored": two
    # of the cells it appears on carry n=50 and n=150 and were scored in full, and it
    # is there because one of the two metrics came back constant, which leaves a
    # correlation with a zero denominator. The ``n`` beside the dash is what separates
    # that from the cells with genuinely no rows.
    handles = [Line2D([], [], color=INK_SECONDARY, marker=markers[s], ls="none",
                      ms=7.5 * fs, label=s.capitalize())
               for s in stats if s in markers]
    handles += [
        Line2D([], [], color=POS, lw=5, label="positive correlation"),
        Line2D([], [], color=NEG, lw=5, label="negative correlation"),
        Line2D([], [], color=INK_SECONDARY, marker="_", ls="none", ms=9,
               label="undefined"),
    ]
    # Legend placed in INCHES converted to figure fractions: this figure's height
    # follows the pair count, and a fixed bottom fraction that clears the legend on a
    # four-pair table eats a row on a ten-pair one.
    h = fig.get_size_inches()[1]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.10 / h), fontsize=pt_foot)
    # ``w_pad`` has to clear the per-row n's, which hang off the right edge of each
    # panel as unclipped annotations tight_layout does not measure.
    fig.tight_layout(rect=(0, (0.30 + 0.09 * fs) / h, 0.995, 1.0),
                     w_pad=1.2, h_pad=0.5)
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
                    ha="left", fontsize=7.5, color=INK_SECONDARY,
                    annotation_clip=False)

    ax.set_ylim(-0.6, len(tbl) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([_pair_label(p) for p in tbl["pair"]])
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("correlation between the two metrics")
    ax.grid(axis="y", visible=True)
    ax.set_title("Do comparable metrics agree? (negative = they rank queries oppositely)",
                 pad=26)
    handles = [Line2D([], [], color=INK_SECONDARY, marker=markers[s], ls="none", ms=8,
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
