"""Loading + summary tables for the raw RAG pipeline results (pre-evaluation).

    from analysis.rag_analysis import load, logprob_summary
    from analysis import plots
    from common.results_io import RAG_PREFIX, latest_results
    df = load(latest_results(RAG_PREFIX))     # or load("results/rag_results_<ts>.json")
    logprob_summary(df)                       # confidence per variant
    retrieval_by_dataset_variant(df)          # retrieval per dataset x variant (+HyDE split)
    plots.retrieval_stage_boxplot(df)         # rag vs rag_sc-orig vs rag_sc-final

This is the companion to ``analysis.analysis`` (which works on the *evaluated*
results). Here we look only at signals the pipeline emits itself, before any
RAGAS/DeepEval scoring:

  - generation confidence: ``gen_logprob_stats.mean`` (mean token logprob; higher,
    i.e. closer to 0, = more confident), across no_rag / rag / rag_sc and datasets.
  - hybrid retrieval scores: ``retrieval_scores`` (dense+BM25), summarised as the
    best score and the top-k spread, across rag / rag_sc and datasets. The
    per-variant and per-dataset tables (``retrieval_variant_summary`` /
    ``retrieval_by_dataset_variant``) split rag_sc into its overall average and the
    subset whose retrieval HyDE actually re-ran, scored before and after that.
  - self-correction: for rag_sc the HyDE re-retrieval means each row carries both
    the ORIGINAL retrieval (``sc_metadata.original_retrieval_scores``) and the
    FINAL one, so we can measure whether re-retrieval actually raised the scores.
  - truncation: ``finish_reason == "length"`` marks a generation cut off at
    max_tokens, split by ``truncation_summary`` into runaway thinking loops (no
    answer text) and genuinely truncated answers. Runs made before finish_reason
    was hoisted to top level only carry it for rag_sc rows; ``truncation_coverage``
    reports how many rows actually have one.
  - abstentions: ``rejected`` marks an answer the pipeline replaced with the
    canonical "the context does not cover this" rejection, ``rejection_reason``
    says why (the model abstained, vs nothing surviving a runaway regen).
    ``abstention_summary`` / ``abstention_by_dataset_variant`` count them,
    ``abstention_signals`` shows what retrieval and confidence looked like on
    those rows, ``abstention_transitions`` whether self-correction shifted WHICH
    queries get refused. no_rag never abstains by construction.
  - pass-through sanity: on the rag_sc rows where NEITHER correction fired,
    rag_sc is by construction the same pipeline as plain rag, so those rows should
    reproduce their rag twin. ``sc_retry_breakdown`` counts them,
    ``sc_passthrough_agreement`` / ``sc_passthrough_means`` /
    ``sc_passthrough_diffs`` compare them against rag (see below).

PASS-THROUGH CHECK (rag_sc without any retry vs rag). A rag_sc row whose
``retrieval_retried_count`` and ``generation_retried_count`` are both 0 ran the
exact same code path as the rag row for that id: same retriever and query (so the
same contexts), and the same system prompt, user prompt and sampling parameters
(so the same answer). Generation is greedy (``LLAMACPP_RAG_TEMPERATURE=0.0``,
``top_p=1.0``), so the two answers should be byte-identical. Retrieval is
deterministic and does come out identical; the answers do NOT always, because
llama.cpp's batching makes even greedy decoding numerically non-reproducible
across requests — a divergence here measures that run-to-run noise, NOT an effect
of self-correction. Read the three functions together: ``..._agreement`` for how
often the twins match, ``..._means`` for whether the averages still line up (they
must, if the divergences are noise rather than bias), ``..._diffs`` for the
concrete diverging pairs to eyeball. Any difference in the CONTEXTS, by contrast,
would be a real bug — the untouched path never re-retrieves.

CAVEAT — the generation-retry confidence comparison is CONFOUNDED. Runs since
2026-07-23 do trip the generation-correction trigger (~36% of rag_sc rows), so
``sc_metadata.original_gen_logprob_stats`` exists and ``sc_generation_gain`` /
the ``rag_sc_orig`` stage have data. But the regenerated pass is not the same
kind of generation as the first one (``rag/utils.py`` _generate call):

  - the first pass runs with ``enable_thinking=False`` (the default), the regen
    with ``enable_thinking=True``, and the mean is taken over ALL completion
    tokens — so the regen's mean includes its reasoning trace, which is
    exploratory and higher-entropy than a final declarative answer;
  - the regen adds ``repeat_penalty=1.1``, which lowers the logits of exactly
    the tokens the model would otherwise rate highest;
  - the regen uses the stricter ``rag_strict`` system prompt.

All three push the mean logprob down, which is why the delta is negative on
literally every retried row (0 of 134 improved) rather than on some. Read
``sc_generation_gain`` as "the regen produces a different, lower-logprob token
distribution", NOT as "regenerating made the model less certain". The same
confound leaks into per-variant confidence: rag_sc pools thinking and
non-thinking rows, so its logprob mean is not comparable to rag / no_rag.
(The ordering itself is correct — ``original_gen_logprob_stats`` is captured
before the regen call and the top-level stats after it.)

Running from the console
------------------------
Run as a module from the repo root (the ``analysis.`` import in __main__ needs the
package on the path, so ``python analysis/rag_analysis.py`` will NOT work):

    python -m analysis.rag_analysis                      # newest rag_results_<ts>.json
    python -m analysis.rag_analysis results/rag_results_20260723_141500.json
    python -m analysis.rag_analysis RAG_FILE EVAL_FILE   # explicit eval file to link

The optional 2nd argument is the evaluated-results JSON to cross-link against.
Omitted, it defaults to the eval run built from the rag file in use — eval
outputs are named ``evaluated_results_<evalts>_from_<ragts>.json``, so the
matching pair is found by the rag stamp and the (id x variant) overlap is
complete. Only when no eval was run over that rag file does it fall back to the
newest eval file, which may overlap partially or not at all.

This runs the whole __main__ report, in three passes: what went WRONG (structural
health check, token-cap truncation, the pass-through check on untouched rag_sc
rows, and — not a failure, but the other way a row comes back without an answer —
the abstentions); what the pipeline MEASURED (confidence and retrieval summaries,
self-correction retry rates, the retrieval- AND generation-trigger tallies); and
whether correction PAID OFF (HyDE re-retrieval and generation-retry gains) —
last, because that is where ``analysis.eval_analysis`` takes over. Then the PNGs.
Every artifact goes to ``analysis/out/<rag-stem>/`` (tables/, figures/,
reports/), named for the results file it came from — see ``analysis.paths``. The
figures themselves live in ``analysis.plots``.

The cross-link with the evaluated results — the ``link_eval`` join, the joined
``analysis/linked_<rag_stem>.parquet`` base table, and the worst/best-query
mining — lives in ``analysis.eval_analysis`` (which imports ``link_eval`` /
``top_n`` from here). Run ``python -m analysis.eval_analysis`` for that.
"""

import difflib
import json
import re
from datetime import datetime

import pandas as pd

from analysis import paths
from common.results_io import RAG_PREFIX, latest_results

VARIANT_ORDER = ["no_rag", "rag", "rag_sc"]
# Retrieval "stages": plain rag, plus rag_sc before/after the HyDE re-retrieval.
STAGE_ORDER = ["rag", "rag_sc_orig", "rag_sc_final"]
# Confidence "stages": every variant, plus rag_sc before/after a generation retry.
CONF_STAGE_ORDER = ["no_rag", "rag", "rag_sc_orig", "rag_sc_final"]
# Retrieval "variants" for the per-dataset tables: the two variants that retrieve at
# all, plus the HyDE-retried rag_sc cohort split before/after its re-retrieval. The
# hyde_* groups are a SUBSET of the rag_sc rows re-cut, not extra rows.
RETRIEVAL_VARIANT_ORDER = ["rag", "rag_sc", "rag_sc_hyde_orig", "rag_sc_hyde_final"]
# Hence only these two are disjoint and may be pooled into the per-dataset "all" row.
_POOLED_RETRIEVAL_VARIANTS = ("rag", "rag_sc")

# The two self-correction retry counters (0 = that correction never fired on the row).
RETRIEVAL_RETRY_COL = "sc_metadata.retrieval_retried_count"
GENERATION_RETRY_COL = "sc_metadata.generation_retried_count"

# The trigger lists behind those counters. Both hold strings like ``"highest=0.71<0.73"``
# / ``"mean=-0.42<-0.35"``, so the same tallies work on either stage — pick one with the
# ``stage`` argument of ``trigger_reasons`` / ``trigger_combinations``.
TRIGGER_COLS = {
    "retrieval": "sc_metadata.retrieval_correction_triggers",
    "generation": "sc_metadata.generation_correction_triggers",
}

# How the final generation stopped, and why an answer collapsed onto the canonical
# rejection. The pipeline now writes both at TOP LEVEL for every variant; older result
# files only have them under sc_metadata, i.e. for rag_sc rows alone. ``load`` folds
# the legacy location into these canonical names, so downstream code reads one column
# and both file layouts work — but on an old file the columns are still populated for
# rag_sc rows only, which is what ``truncation_coverage`` reports.
FINISH_REASON_COL = "finish_reason"
REJECTION_REASON_COL = "rejection_reason"
_LEGACY_REASON_COLS = {
    FINISH_REASON_COL: "sc_metadata.finish_reason",
    REJECTION_REASON_COL: "sc_metadata.rejection_reason",
}


def _best(scores):
    """Top (max) retrieval score of a row, or NaN when there are none."""
    if isinstance(scores, (list, tuple)) and len(scores):
        return max(scores)
    return float("nan")


def _spread(scores):
    """Top-k spread (max - min) of a row's retrieval scores; this is the quantity
    the rag_sc 'spread>threshold' correction trigger keys on. NaN if <2 scores."""
    if isinstance(scores, (list, tuple)) and len(scores) >= 2:
        return max(scores) - min(scores)
    return float("nan")


def load(path):
    """Flatten raw RAG results JSON to a tidy DataFrame with convenience columns.

    Adds: ordered ``variant`` categorical; ``retrieval_best`` / ``retrieval_spread``
    (from the final ``retrieval_scores``); ``retrieval_best_orig`` /
    ``retrieval_spread_orig`` (from ``sc_metadata.original_retrieval_scores``, i.e.
    pre re-retrieval, meaningful only for rag_sc); numeric SC retry counts; and
    canonical ``finish_reason`` / ``rejection_reason`` columns (folded from the legacy
    ``sc_metadata.*`` location when reading a pre-hoist result file).
    """
    with open(path, "r", encoding="utf-8") as f:
        df = pd.json_normalize(json.load(f))

    if "variant" in df:
        df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)

    df["retrieval_best"] = df["retrieval_scores"].apply(_best)
    df["retrieval_spread"] = df["retrieval_scores"].apply(_spread)

    orig = "sc_metadata.original_retrieval_scores"
    if orig in df:
        df["retrieval_best_orig"] = df[orig].apply(_best)
        df["retrieval_spread_orig"] = df[orig].apply(_spread)

    # Pre-regeneration generation confidence, present only once a run actually
    # trips the generation-correction trigger (json_normalize creates the column
    # only if some row has sc_metadata.original_gen_logprob_stats). When absent,
    # the confidence-stage helpers just omit the rag_sc_orig stage.
    og = "sc_metadata.original_gen_logprob_stats.mean"
    if og in df:
        df["gen_logprob_mean_orig"] = pd.to_numeric(df[og], errors="coerce")

    for c in (RETRIEVAL_RETRY_COL, GENERATION_RETRY_COL):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fold the legacy sc_metadata copies of finish_reason / rejection_reason onto the
    # canonical top-level names, so both old and new result files expose one column.
    # New files carry the top-level value for every variant; old ones only have the
    # sc_metadata copy (rag_sc rows). Where both exist they agree, and the top-level
    # value wins — the sc_metadata copy only ever fills gaps.
    for canonical, legacy in _LEGACY_REASON_COLS.items():
        if legacy not in df:
            continue
        if canonical in df:
            df[canonical] = df[canonical].where(df[canonical].notna(), df[legacy])
        else:
            df[canonical] = df[legacy]

    return df


def _order(df):
    return [v for v in VARIANT_ORDER if v in df["variant"].unique()]


# --- Health check ------------------------------------------------------------

# Fields every raw-results row should carry. Missing ones point at a pipeline
# writer bug, not a data-quality issue — so they are checked separately.
REQUIRED_FIELDS = ["id", "variant", "source_dataset", "answer",
                   "retrieval_scores", "gen_logprob_stats.mean"]


def rag_health_report(df, source=None):
    """Structural sanity check on the *raw* pipeline results (companion to
    ``analysis.health_report``). Prints; ``source`` only adds the provenance
    header lines — the whole console report is saved by ``analysis.paths.capture``.

    Verifies the things that must hold before any downstream analysis is trusted:
      - required fields present (missing => pipeline writer bug);
      - row counts per variant;
      - empty / blank answers, and ``[LLAMACPP ...]`` generation errors;
      - retrieval sanity: ``rag`` and ``rag_sc`` rows should HAVE a retrieval score,
        ``no_rag`` rows should NOT (retrieval was skipped) — either violation is flagged;
      - generation confidence present per variant;
      - rag_sc self-correction metadata present.

    Returns a flags dict.
    """
    flags = {}
    lines = []

    def emit(s):
        lines.append(s)

    n = len(df)
    emit(f"rag_health_report: {n} rows")
    if source is not None:
        emit(f"  source: {source}")
        emit(f"  generated: {datetime.now().isoformat(timespec='seconds')}")

    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    flags["missing_fields"] = missing
    emit(f"  missing required fields: {len(missing)}{f' {missing}' if missing else ''} "
         f"(of: {', '.join(REQUIRED_FIELDS)})")

    if "variant" in df:
        counts = df["variant"].astype("object").value_counts().to_dict()
        flags["rows_per_variant"] = counts
        emit(f"  rows per variant: {counts}")

    if "answer" in df:
        ans = df["answer"].fillna("").astype(str)
        blank = ans.str.strip().eq("")
        gen_err = ans.str.contains(r"\[LLAMACPP", regex=True)
        flags["blank_answers"] = int(blank.sum())
        flags["answer_generation_errors"] = int(gen_err.sum())
        emit(f"  blank/empty answers: {int(blank.sum())}")
        emit(f"  answer generation errors ([LLAMACPP ...]): {int(gen_err.sum())}")

    # Retrieval presence must match the variant: rag/rag_sc need a score, no_rag must not.
    if "variant" in df and "retrieval_best" in df:
        has_retr = df["retrieval_best"].notna()
        need = df["variant"].isin(["rag", "rag_sc"])
        missing_retr = int((need & ~has_retr).sum())
        stray_retr = int((~need & has_retr).sum())  # no_rag rows that carry a score
        flags["rag_rows_missing_retrieval"] = missing_retr
        flags["no_rag_rows_with_retrieval"] = stray_retr
        emit(f"  rag/rag_sc rows missing a retrieval score: {missing_retr}")
        emit(f"  no_rag rows unexpectedly carrying a retrieval score: {stray_retr}")

    if "variant" in df and "gen_logprob_stats.mean" in df:
        lp = pd.to_numeric(df["gen_logprob_stats.mean"], errors="coerce")
        by_v = df.assign(_m=lp.isna()).groupby("variant", observed=True)["_m"].sum()
        missing_lp = {str(k): int(v) for k, v in by_v.items() if v}
        flags["rows_missing_logprob"] = missing_lp
        emit(f"  rows missing generation confidence, per variant: "
             f"{missing_lp if missing_lp else 0}")

    # Generations cut off at max_tokens, split into runaway thinking loops (no
    # answer text survived) and answers that were written but lost their tail.
    have_fr, n_rows = truncation_coverage(df)
    if have_fr:
        is_len = df[FINISH_REASON_COL].astype(str) == "length"
        rej = (df[REJECTION_REASON_COL].astype(str) if REJECTION_REASON_COL in df
               else pd.Series("", index=df.index))
        loops = int((is_len & (rej == "empty")).sum())
        trunc = int((is_len & (rej != "empty")).sum())
        flags["generations_hit_token_cap"] = int(is_len.sum())
        flags["thinking_loops"] = loops
        flags["answers_truncated"] = trunc
        emit(f"  generations cut off at the token cap (finish_reason=length): "
             f"{int(is_len.sum())} (finish_reason covers {coverage_note(df)})")
        emit(f"    thinking loops (cut off with no answer text): {loops}")
        emit(f"    answers truncated (answer written, tail lost): {trunc}")
    else:
        emit(f"  finish_reason not recorded in this file (0 of {n_rows} rows) "
             f"— no rag_sc rows, or an older pipeline")

    sc_cols = [c for c in df.columns if c.startswith("sc_metadata.")]
    flags["sc_metadata_present"] = bool(sc_cols)
    emit(f"  self-correction metadata columns: "
         f"{len(sc_cols)}{'' if sc_cols else ' (none — rag_sc SC signals unavailable)'}")

    print("\n".join(lines))
    return flags


# --- Tabular summaries -------------------------------------------------------

# The stats every summary table reports for a numeric column, in one place so the
# confidence and retrieval tables stay column-for-column comparable.
_STAT_AGGS = {"n": "count", "mean": "mean", "std": "std",
              "median": "median", "min": "min", "max": "max"}


def _stats_by(df, by, col, dropna=True):
    """``_STAT_AGGS`` for numeric ``col``, grouped by ``by``.

    ``dropna`` first drops rows without a value, so groups that exist only as NaN
    (e.g. no_rag under a retrieval column) disappear instead of showing up as n=0.
    """
    g = df.assign(_v=pd.to_numeric(df[col], errors="coerce"))
    if dropna:
        g = g.dropna(subset=["_v"])
    return g.groupby(by, observed=True)["_v"].agg(**_STAT_AGGS)


def _int_counts(out, cols=("n",)):
    """Cast count columns back to int, in place-ish, and return the frame.

    Any table assembled row-wise from Series (``pd.DataFrame(rows)``, ``.to_frame().T``,
    ``groupby.apply``) comes out all-float, which prints a row count as ``50.0``. Only
    complete columns are cast, so a table with a gap comes back untouched rather than
    raising.
    """
    for c in cols:
        if c in out and out[c].notna().all():
            out[c] = out[c].astype(int)
    return out


def _with_pooled_rows(base, pooled, order=None, level="variant", label="all",
                      int_cols=("n",)):
    """Interleave a per-(source_dataset, group) stats table with one pooled row per
    dataset, appended after that dataset's group rows.

    ``base`` is indexed by (source_dataset, group), ``pooled`` by source_dataset
    alone. The pooled row is recomputed over rows, not averaged from the group
    means, so uneven group sizes don't misweight it. Shared by
    ``logprob_by_dataset_variant`` and ``retrieval_by_dataset_variant`` so both
    tables read the same way.

    ``int_cols`` are cast back to int on the way out: stacking the rows as Series
    upcasts every column to float, which turns a row COUNT into ``50.0``. Only
    columns that are complete (no NaN) are cast, so a table with a missing group
    still comes back rather than raising.
    """
    rows, idx = [], []
    for ds in base.index.get_level_values(0).unique():
        block = base.xs(ds, level=0)
        keys = [k for k in (order or list(block.index)) if k in block.index]
        for key in keys:
            rows.append(block.loc[key])
            idx.append((str(ds), str(key)))
        if pooled is not None and ds in pooled.index:
            rows.append(pooled.loc[ds])
            idx.append((str(ds), label))
    out = pd.DataFrame(rows).reset_index(drop=True)
    out.index = pd.MultiIndex.from_tuples(idx, names=["source_dataset", level])
    return _int_counts(out, int_cols)


def logprob_summary(df, by="variant", col="gen_logprob_stats.mean"):
    """Generation-confidence stats (mean token logprob) grouped by ``by``.

    ``by="variant"`` answers 'is the model more/less confident with vs without
    retrieval?'; ``by=["source_dataset", "variant"]`` breaks that down per dataset.
    Values are negative; closer to 0 = more confident.
    """
    out = _stats_by(df, by, col, dropna=False)
    if by == "variant":
        out = out.reindex(_order(df))
    return out


def logprob_by_dataset_variant(df, with_all=True, col="gen_logprob_stats.mean"):
    """Confidence per (source_dataset, variant), with an ``all`` row appended after
    each dataset's variant rows (pooled over the three variants of that dataset).

    Same stats as ``logprob_summary``. The ``all`` row's ``mean``/``std``/… are
    recomputed over every row of the dataset (a pooled figure, not the mean of the
    three per-variant means), so uneven variant counts don't misweight it.
    """
    base = logprob_summary(df, by=["source_dataset", "variant"], col=col)
    if not with_all:
        return base
    return _with_pooled_rows(base, logprob_summary(df, by="source_dataset", col=col),
                             order=VARIANT_ORDER)


def retrieval_summary(df, by="variant", col="retrieval_best"):
    """Hybrid retrieval-score stats grouped by ``by`` (rows without retrieval —
    i.e. no_rag — drop out, since ``retrieval_best`` is NaN there).

    Plain per-variant view: rag vs rag_sc, where rag_sc is its FINAL score, HyDE
    re-retrieval included. Use ``retrieval_variant_summary`` to see that rag_sc
    average alongside the re-retrieved cohort before/after HyDE.
    """
    out = _stats_by(df, by, col)
    if by == "variant":
        out = out.reindex([v for v in _order(df) if v in out.index])
    return out


def _hyde_rows(df):
    """The rag_sc rows whose retrieval the HyDE correction actually re-ran.

    A missing retry-counter column (a file without sc_metadata) means 'never fired',
    so this returns no rows rather than silently treating every rag_sc row as retried.
    """
    sc = df[df["variant"] == "rag_sc"]
    if RETRIEVAL_RETRY_COL not in sc:
        return sc.iloc[0:0]
    return sc[pd.to_numeric(sc[RETRIEVAL_RETRY_COL], errors="coerce").fillna(0) > 0]


def retrieval_variant_long(df, col="retrieval_best", col_orig="retrieval_best_orig"):
    """Long-form retrieval scores tagged by retrieval variant, HyDE cohort split out.

    Four groups (``RETRIEVAL_VARIANT_ORDER``), the input to the retrieval tables and
    the per-dataset boxplot:

      - ``rag`` — every plain rag row;
      - ``rag_sc`` — every rag_sc row, at its FINAL score. This is the rag_sc average
        that is directly comparable to ``rag``: same ids, whole variant;
      - ``rag_sc_hyde_orig`` — ONLY the rows whose retrieval was re-run, at their
        first-retrieval score (``col_orig``);
      - ``rag_sc_hyde_final`` — those same rows after the HyDE re-retrieval.

    The two ``hyde`` groups are a subset of the ``rag_sc`` rows re-cut, not extra
    rows, so they must not be pooled together with it (see
    ``_POOLED_RETRIEVAL_VARIANTS``). Read ``rag_sc_hyde_orig`` -> ``rag_sc_hyde_final``
    as the paired before/after that ``sc_retrieval_gain`` quantifies as a delta.

    ``col``/``col_orig`` switch the measured quantity: the defaults give the best
    score, ``"retrieval_spread"`` / ``"retrieval_spread_orig"`` the top-k spread.
    """
    frames = []

    def _tag(sub, name, values):
        return pd.DataFrame({"id": sub["id"], "source_dataset": sub["source_dataset"],
                             "variant": name,
                             "value": pd.to_numeric(values, errors="coerce")})

    if col in df:
        for v in ("rag", "rag_sc"):
            sub = df[df["variant"] == v]
            if len(sub):
                frames.append(_tag(sub, v, sub[col]))
        hyde = _hyde_rows(df)
        if len(hyde) and col_orig in hyde:
            frames.append(_tag(hyde, "rag_sc_hyde_orig", hyde[col_orig]))
            frames.append(_tag(hyde, "rag_sc_hyde_final", hyde[col]))
    if not frames:
        return pd.DataFrame(columns=["id", "source_dataset", "variant", "value"])

    out = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    present = [v for v in RETRIEVAL_VARIANT_ORDER if v in set(out["variant"])]
    out["variant"] = pd.Categorical(out["variant"], categories=present, ordered=True)
    return out


def retrieval_variant_summary(df, col="retrieval_best", col_orig="retrieval_best_orig"):
    """Retrieval-score stats per retrieval variant: rag, the rag_sc average, and the
    HyDE-retried cohort before/after re-retrieval (see ``retrieval_variant_long``).

    The retrieval analogue of ``logprob_summary(df)``, extended with the HyDE split.
    """
    long = retrieval_variant_long(df, col=col, col_orig=col_orig)
    if not len(long):
        return pd.DataFrame(columns=list(_STAT_AGGS))
    return _stats_by(long, "variant", "value")


def retrieval_by_dataset_variant(df, with_all=True, col="retrieval_best",
                                 col_orig="retrieval_best_orig"):
    """Retrieval scores per (source_dataset, retrieval variant) — the retrieval
    counterpart of ``logprob_by_dataset_variant``.

    Per dataset: ``rag``, the ``rag_sc`` average (final scores, all its rows), then
    ``rag_sc_hyde_orig`` / ``rag_sc_hyde_final`` — the rows where HyDE re-retrieval
    was triggered, scored before and after it. With ``with_all`` an ``all`` row
    closes each dataset, pooled over that dataset's rag + rag_sc rows only: the
    hyde groups are those same rag_sc rows re-cut and would double-count.
    """
    long = retrieval_variant_long(df, col=col, col_orig=col_orig)
    if not len(long):
        return pd.DataFrame(columns=list(_STAT_AGGS))
    base = _stats_by(long, ["source_dataset", "variant"], "value")
    if not with_all:
        return base
    pooled = _stats_by(long[long["variant"].isin(_POOLED_RETRIEVAL_VARIANTS)],
                       "source_dataset", "value")
    return _with_pooled_rows(base, pooled, order=RETRIEVAL_VARIANT_ORDER)


def trigger_summary(df, by="source_dataset", with_total=True):
    """How often the rag_sc self-correction fired, grouped by ``by``.

    ``retrieval_retry_rate`` / ``generation_retry_rate`` are the share of rag_sc
    rows whose retrieval / generation was re-run. (Generation retry is 0 in the
    current runs.) With ``with_total`` an ``ALL`` row is appended: ``n`` is the sum
    of rag_sc rows across every group and the rates are recomputed over all of them
    (a pooled rate, i.e. total fired / total rows — NOT the mean of the per-group
    rates, which would misweight uneven group sizes).
    """
    sc = df[df["variant"] == "rag_sc"].copy()
    sc["retrieval_fired"] = sc.get(RETRIEVAL_RETRY_COL, 0).fillna(0) > 0
    sc["generation_fired"] = sc.get(GENERATION_RETRY_COL, 0).fillna(0) > 0
    g = sc.groupby(by, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "retrieval_retry_rate": g["retrieval_fired"].mean().round(3),
        "generation_retry_rate": g["generation_fired"].mean().round(3),
    })
    if with_total and len(sc):
        out.loc["ALL"] = {
            "n": len(sc),
            "retrieval_retry_rate": round(sc["retrieval_fired"].mean(), 3),
            "generation_retry_rate": round(sc["generation_fired"].mean(), 3),
        }
        out["n"] = out["n"].astype(int)
    return out


def _trigger_col(stage):
    """Column holding the trigger list of ``stage`` ("retrieval" or "generation")."""
    try:
        return TRIGGER_COLS[stage]
    except KeyError:
        raise ValueError(
            f"stage must be one of {sorted(TRIGGER_COLS)}, got {stage!r}") from None


def _trigger_names(trigs):
    """The bare names of one row's triggers: ``["highest=0.71<0.73"] -> ["highest"]``."""
    return [str(t).split("=", 1)[0].strip() for t in (trigs or [])]


def trigger_reasons(df, stage="retrieval"):
    """Count which self-correction triggers fired across all rag_sc rows.

    The pipeline logs triggers like ``"highest=0.71<0.73"`` / ``"spread=0.05>0.04"`` for
    retrieval and ``"mean=-0.42<-0.35"`` / ``"min=-6.1<-5.0"`` for generation; this
    tallies them by their leading name so you can see which threshold does the work.
    ``stage`` picks which correction to look at (see ``TRIGGER_COLS``).
    """
    sc = df[df["variant"] == "rag_sc"]
    col = _trigger_col(stage)
    counts = {}
    if col in sc:
        for trigs in sc[col]:
            for name in _trigger_names(trigs):
                counts[name] = counts.get(name, 0) + 1
    return pd.Series(counts, dtype="int64").sort_values(ascending=False)


def trigger_combinations(df, stage="retrieval"):
    """Per rag_sc row, which SET of ``stage``'s correction triggers fired.

    Complements ``trigger_reasons`` (which counts total firings per trigger name):
    here each rag_sc row is bucketed by its *combination* of triggers, so you can
    read off how many rows fired ONLY ``highest``, ONLY ``spread``, or BOTH
    (``highest & spread``) — and likewise ``mean`` / ``min`` on the generation stage.
    Rows where no trigger fired are bucketed as ``none``.
    """
    sc = df[df["variant"] == "rag_sc"]
    col = _trigger_col(stage)
    counts = {}
    if col in sc:
        for trigs in sc[col]:
            names = sorted(set(_trigger_names(trigs)))
            key = " & ".join(names) if names else "none"
            counts[key] = counts.get(key, 0) + 1
    return pd.Series(counts, dtype="int64").sort_values(ascending=False)


def truncation_coverage(df):
    """``(rows carrying a finish_reason, total rows)`` — the honest denominator for
    ``truncation_summary``. Equal on runs made since finish_reason was hoisted to top
    level; on older files the first number is just the rag_sc row count, since back
    then only sc_metadata carried one."""
    have = int(df[FINISH_REASON_COL].notna().sum()) if FINISH_REASON_COL in df else 0
    return have, len(df)


def coverage_note(df):
    """One-line, honest description of WHICH rows carry a finish_reason — the caveat
    that belongs next to any truncation figure. Distinguishes the two reasons a row
    can lack one: an old result file (only rag_sc recorded it) versus a generation
    that never completed (those rows carry a ``[LLAMACPP ...]`` answer)."""
    have, n = truncation_coverage(df)
    if have == n:
        return f"all {n} rows"
    if "variant" in df and FINISH_REASON_COL in df:
        covered = set(df.loc[df[FINISH_REASON_COL].notna(), "variant"].astype(str))
        if covered <= {"rag_sc"}:
            return (f"only the {have} rag_sc rows of {n} — this file predates hoisting "
                    f"finish_reason to top level")
    return (f"{have} of {n} rows; the other {n - have} recorded none "
            f"(generation never completed)")


def truncation_summary(df, by="source_dataset", with_total=True):
    """How often generation hit the token cap instead of stopping on its own.

    ``finish_reason == "length"`` means the model never emitted a stop token and was
    cut off at max_tokens. Two very different failures hide behind that one value,
    split apart here by whether any answer text survived:

      - ``thinking_loop``: cut off with NO answer text (``rejection_reason == "empty"``)
        — the runaway rag_sc regen that burns its whole budget reasoning and never
        answers. The pipeline substitutes the canonical rejection, so downstream it
        looks like an ordinary abstention; this count is what unmasks it.
      - ``answer_truncated``: cut off but an answer WAS written — a genuine answer that
        ran past the cap and lost its tail.

    ``length_rate`` is the share of the group's rows that were cut off. With
    ``with_total`` an ``ALL`` row is appended, pooled over every row (total cut off /
    total rows, NOT the mean of the per-group rates, which would misweight uneven
    groups) — the same convention as ``trigger_summary``.

    Coverage caveat: ``n`` counts only rows that carry a finish_reason. On a current
    run that is every row; on one made before finish_reason was hoisted to top level
    it is the rag_sc rows alone — pair this with ``truncation_coverage``.

    Takes whatever rows you hand it, so filter first if the variants aren't comparable:
    the __main__ report passes rag_sc only, since single-pass no_rag/rag rows cannot
    loop and would deflate ``length_rate``.
    """
    cols = ["n", "length", "length_rate", "thinking_loop", "answer_truncated"]
    if FINISH_REASON_COL not in df:
        return pd.DataFrame(columns=cols)
    sub = df[df[FINISH_REASON_COL].notna()].copy()
    if not len(sub):
        return pd.DataFrame(columns=cols)

    sub["_length"] = sub[FINISH_REASON_COL].astype(str) == "length"
    rej = (sub[REJECTION_REASON_COL] if REJECTION_REASON_COL in sub
           else pd.Series("", index=sub.index))
    empty = rej.astype(str) == "empty"
    sub["_thinking_loop"] = sub["_length"] & empty
    sub["_answer_truncated"] = sub["_length"] & ~empty

    def _agg(g):
        return pd.Series({
            "n": len(g),
            "length": int(g["_length"].sum()),
            "length_rate": round(g["_length"].mean(), 3),
            "thinking_loop": int(g["_thinking_loop"].sum()),
            "answer_truncated": int(g["_answer_truncated"].sum()),
        })

    out = sub.groupby(by, observed=True).apply(_agg, include_groups=False)
    if with_total:
        out.loc["ALL"] = _agg(sub)
    for c in ("n", "length", "thinking_loop", "answer_truncated"):
        out[c] = out[c].astype(int)
    return out


# --- Abstentions -------------------------------------------------------------

# Why an answer collapsed onto the canonical rejection string (see
# ``rag.utils._finalize_answer``):
#   model_rejected — the model itself abstained: its answer opens with the canonical
#                    rejection. The only abstention signal the thinking-off ``rag``
#                    variant has, and the intended behaviour — saying "the context does
#                    not cover this" instead of inventing an answer.
#   empty          — nothing survived stripping the <think> block: the runaway rag_sc
#                    regen that spends its whole budget reasoning. Counted here because
#                    downstream it IS an abstention, but it is really the truncation
#                    failure of ``truncation_summary`` wearing an abstention's clothes.
# no_rag has no context to call insufficient, so it never abstains: a 0 there is
# structural, not a finding about the variant.
ABSTENTION_REASONS = ["model_rejected", "empty"]

# Signals compared between the abstained and the answered rows, as
# ``{display name: column}``. gen_tokens is included to expose the length difference
# behind the logprob one (see the caveat on ``abstention_signals``).
ABSTENTION_SIGNAL_COLS = {
    "retrieval_best": "retrieval_best",
    "retrieval_spread": "retrieval_spread",
    "gen_logprob_mean": "gen_logprob_stats.mean",
    "gen_tokens": "gen_logprob_stats.n",
}


def _abstained(df):
    """Boolean 'this row abstained', from the pipeline's own ``rejected`` flag, falling
    back to the presence of a ``rejection_reason``.

    Deliberately never string-matches the answer: a run is multilingual, so the
    canonical rejection has one form per language and matching only the English one
    would silently miss the German abstentions (the same trap
    ``analysis.analysis._rejection_mask`` documents)."""
    if "rejected" in df:
        return df["rejected"].fillna(False).astype(bool)
    if REJECTION_REASON_COL in df:
        return df[REJECTION_REASON_COL].notna()
    return pd.Series(False, index=df.index)


def abstention_summary(df, by="variant", with_total=True):
    """How often the pipeline abstained instead of answering, grouped by ``by``.

    Columns: ``n`` (rows in the group), ``abstained`` / ``abstention_rate``, then the
    same abstentions split by ``rejection_reason`` — ``model_rejected`` (the intended
    behaviour) vs ``empty`` (a runaway regen, i.e. a failure; see
    ``ABSTENTION_REASONS``). The two reason columns sum to ``abstained`` on any file
    that records a reason.

    ``by`` takes anything a groupby does: ``"variant"``, ``"source_dataset"``,
    ``["source_dataset", "variant"]``, or a column you attach at the call site (the
    __main__ report passes ``retry_kind``). With ``with_total`` an ``ALL`` row is
    appended, pooled over rows (total abstained / total rows, NOT the mean of the
    per-group rates) — single-level ``by`` only, since a MultiIndex has no one place
    to put it; use ``abstention_by_dataset_variant`` for the two-level table.
    """
    cols = ["n", "abstained", "abstention_rate"] + ABSTENTION_REASONS
    if not len(df):
        return pd.DataFrame(columns=cols)

    sub = df.copy()
    sub["_abstained"] = _abstained(df)
    reason = (sub[REJECTION_REASON_COL].astype("string") if REJECTION_REASON_COL in sub
              else pd.Series(pd.NA, index=sub.index, dtype="string"))
    for r in ABSTENTION_REASONS:
        sub[f"_{r}"] = reason.eq(r).fillna(False)

    def _agg(g):
        out = {"n": len(g), "abstained": int(g["_abstained"].sum()),
               "abstention_rate": round(g["_abstained"].mean(), 3)}
        out.update({r: int(g[f"_{r}"].sum()) for r in ABSTENTION_REASONS})
        return pd.Series(out)

    out = sub.groupby(by, observed=True).apply(_agg, include_groups=False)
    if with_total and isinstance(by, str):
        out.loc["ALL"] = _agg(sub)
    for c in ["n", "abstained"] + ABSTENTION_REASONS:
        out[c] = out[c].astype(int)
    return out[cols]


def abstention_by_dataset_variant(df, with_all=True):
    """Abstentions per (source_dataset, variant), with an ``all`` row closing each
    dataset — the abstention counterpart of ``logprob_by_dataset_variant``.

    Same columns as ``abstention_summary``, every count of which is handed to
    ``_with_pooled_rows`` as an ``int_col`` (the table is counts end to end and reads
    badly as ``50.0``). The ``all`` row is pooled over that dataset's rows rather than
    averaged from the per-variant rates. Feed it rag/rag_sc rows only if you want that
    pooled rate to mean something: no_rag cannot abstain, so leaving it in drags every
    ``all`` row down by a third for a structural reason (which is what the __main__
    report does).
    """
    base = abstention_summary(df, by=["source_dataset", "variant"], with_total=False)
    if not with_all or not len(base):
        return base
    pooled = abstention_summary(df, by="source_dataset", with_total=False)
    return _with_pooled_rows(base, pooled, order=VARIANT_ORDER,
                             int_cols=["n", "abstained"] + ABSTENTION_REASONS)


def abstention_signals(df, by="variant", cols=None):
    """What the pipeline's own signals looked like when it abstained: the mean of each
    signal per group, split into the rows that ANSWERED and the rows that ABSTAINED.

    Indexed by (group, ``answered`` / ``abstained``); columns are ``n`` plus the mean of
    each of ``cols`` (default ``ABSTENTION_SIGNAL_COLS``). ``retrieval_best`` is the one
    that validates the behaviour: an abstained cohort scoring markedly lower than the
    answered one says abstention tracks weak retrieval, which is what it is for.

    CAVEAT — ``gen_logprob_mean`` is NOT a confidence reading on abstained rows. The
    canonical rejection is a short formulaic sentence whose tokens are trivially
    predictable, so abstentions come out looking MORE confident (mean closer to 0) than
    real answers; that number measures string predictability, not the model's certainty
    about the question. ``gen_tokens`` shows the length gap behind it. This has a real
    knock-on effect: the generation-correction trigger keys on exactly this mean, so an
    abstaining first pass almost never trips a regeneration — which is why the retried
    cohorts in ``abstention_summary(..., by="retry_kind")`` abstain less by selection,
    not by rescue.
    """
    cols = ABSTENTION_SIGNAL_COLS if cols is None else cols
    if not len(df):
        return pd.DataFrame(columns=["n"] + list(cols))

    sub = df.copy()
    sub["_status"] = pd.Categorical(
        _abstained(df).map({True: "abstained", False: "answered"}),
        categories=["answered", "abstained"], ordered=True)
    present = {name: c for name, c in cols.items() if c in sub}
    for name, c in present.items():
        sub[f"_{name}"] = pd.to_numeric(sub[c], errors="coerce")

    keys = ([by] if isinstance(by, str) else list(by)) + ["_status"]
    g = sub.groupby(keys, observed=True)
    out = pd.DataFrame({"n": g.size(),
                        **{name: g[f"_{name}"].mean() for name in present}})
    out.index = out.index.set_names(keys[:-1] + ["status"])
    return out


def abstention_transitions(df, a="rag_sc", b="rag"):
    """Per id, WHO abstained: variant ``a``, variant ``b``, both, or neither.

    The rates alone cannot tell these apart — rag_sc abstaining 3 points more often
    than rag could be a handful of new abstentions, or a hundred new ones against
    ninety rescued. Counted over the ids that carry both variants: ``n_ids``,
    ``both_answered``, ``only_<a>`` (a abstained where b answered — the corrections
    made it give up), ``only_<b>`` (the reverse — rescued), ``both_abstained``, and
    ``net`` = only_a − only_b, the change in abstention count the rate gap summarises.
    """
    tmp = pd.DataFrame({"id": df["id"], "variant": df["variant"].astype(str),
                        "_a": _abstained(df).astype(float)})
    w = tmp.pivot_table(index="id", columns="variant", values="_a", aggfunc="first")
    if not {a, b} <= set(w.columns):
        return pd.Series({"n_ids": 0})
    m = w[a].notna() & w[b].notna()
    x, y = w.loc[m, a] > 0.5, w.loc[m, b] > 0.5
    return pd.Series({
        "n_ids": int(m.sum()),
        "both_answered": int((~x & ~y).sum()),
        f"only_{a}": int((x & ~y).sum()),
        f"only_{b}": int((y & ~x).sum()),
        "both_abstained": int((x & y).sum()),
        "net": int((x & ~y).sum() - (y & ~x).sum()),
    })


# A recorded trigger, e.g. ``"highest=0.710<0.73"`` — value, comparison, threshold.
_TRIGGER_VALUE_RE = re.compile(
    r"^(?P<name>[A-Za-z_]+)=(?P<value>-?[0-9.]+)(?P<op>[<>])(?P<threshold>-?[0-9.]+)$")

# Columns carried by the false-refusal table, in print order. ``query`` last: it is the
# wide one, and the point of the table is to read the questions by hand.
FALSE_REFUSAL_COLS = ["id", "source_dataset", "variant", "lang", "retrieval_best",
                     "retrieval_spread", "gen_logprob_stats.mean", "query"]


def trigger_threshold(df, name="highest", stage="retrieval"):
    """The threshold a named trigger was evaluated against IN THIS RUN, read back out
    of the recorded trigger strings (``"highest=0.710<0.73"`` -> ``0.73``).

    Taken from the data rather than imported from ``rag.llm_config``, so re-analysing an
    old results file uses the threshold that file was actually produced with instead of
    whatever the environment happens to say today. Returns ``None`` if no row fired that
    trigger — the value is only ever written on a firing row, so there is nothing to
    read; pass an explicit threshold in that case.
    """
    sc = df[df["variant"] == "rag_sc"]
    col = _trigger_col(stage)
    if col not in sc:
        return None
    for trigs in sc[col]:
        for t in (trigs or []):
            m = _TRIGGER_VALUE_RE.match(str(t).strip())
            if m and m.group("name") == name:
                return float(m.group("threshold"))
    return None


def false_refusals(df, threshold=None, variants=("rag", "rag_sc"), exclude_empty=True):
    """The abstentions that had GOOD retrieval: rows that refused although their best
    retrieval score cleared the self-correction threshold.

    An abstention on a question the corpus does not cover is the pipeline working as
    designed, and those rows dominate the counts (medqa, mmlu). This is the complement
    — the set worth reading by hand, because the pipeline's OWN threshold judged the
    retrieved context adequate (that score is exactly the level at which no HyDE
    re-retrieval fires) and the model refused anyway. Every row here is either a
    retrieval hit that was topically wrong, or an over-cautious refusal.

    ``threshold=None`` reads the run's own value via ``trigger_threshold``; pass a float
    to override or to sweep the cut. ``exclude_empty`` drops the ``empty`` abstentions,
    which are runaway-regen truncations rather than refusals (``truncation_summary``
    owns those). Returns the rows sorted by ``retrieval_best`` descending — the
    best-supported refusal, i.e. the least defensible one, first — and an empty frame
    when no threshold can be determined.
    """
    cols = [c for c in FALSE_REFUSAL_COLS if c in df]
    if threshold is None:
        threshold = trigger_threshold(df)
    if threshold is None or "retrieval_best" not in df:
        return pd.DataFrame(columns=cols)

    sub = df[df["variant"].astype(str).isin(variants)]
    keep = _abstained(sub) & (pd.to_numeric(sub["retrieval_best"], errors="coerce") >= threshold)
    if exclude_empty and REJECTION_REASON_COL in sub:
        keep = keep & sub[REJECTION_REASON_COL].astype("string").ne("empty").fillna(True)
    return sub[keep].sort_values("retrieval_best", ascending=False)[cols]


def abstention_transition_rows(df, a="rag_sc", b="rag", flips_only=True):
    """The per-id table behind ``abstention_transitions``: the ids that abstain in one
    variant but not the other, with both sides' signals so you can see WHY.

    ``flip`` is ``only_<a>`` (a gave up where b answered) or ``only_<b>`` (the reverse);
    with ``flips_only=False`` the agreeing ids come too, as ``both`` / ``neither``.
    ``hyde_delta`` is a's final minus original best retrieval score, so the obvious
    suspicion about a ``only_rag_sc`` row — the re-retrieval swapped good context for
    bad and the model gave up — can be checked on the row instead of assumed. NaN there
    means a never re-retrieved that id, which rules the explanation out entirely.

    Sorted worst ``hyde_delta`` first within each flip. This is the eyeball list; the
    counts are ``abstention_transitions``.
    """
    def _side(v, suffix, extra=()):
        s = df[df["variant"].astype(str) == v]
        out = pd.DataFrame({"id": s["id"], f"abstained{suffix}": _abstained(s)})
        if "retrieval_best" in s:
            out[f"retrieval_best{suffix}"] = pd.to_numeric(s["retrieval_best"], errors="coerce")
        if "gen_logprob_stats.mean" in s:
            out[f"logprob{suffix}"] = pd.to_numeric(s["gen_logprob_stats.mean"], errors="coerce")
        for c in extra:
            if c in s:
                out[c if c in ("source_dataset", "lang", "query") else f"{c}{suffix}"] = s[c]
        if {"retrieval_best", "retrieval_best_orig"} <= set(s.columns) and suffix == f"_{a}":
            out["hyde_delta"] = (pd.to_numeric(s["retrieval_best"], errors="coerce")
                                 - pd.to_numeric(s["retrieval_best_orig"], errors="coerce"))
        return out

    left = _side(a, f"_{a}", extra=("source_dataset", "lang", REJECTION_REASON_COL, "query"))
    right = _side(b, f"_{b}")
    if not len(left) or not len(right):
        return pd.DataFrame(columns=["id", "flip"])

    out = left.merge(right, on="id", how="inner")
    x, y = out[f"abstained_{a}"], out[f"abstained_{b}"]
    flip = pd.Series("neither", index=out.index, dtype="object")
    flip[x & ~y] = f"only_{a}"
    flip[~x & y] = f"only_{b}"
    flip[x & y] = "both"
    out.insert(1, "flip", flip)
    if flips_only:
        out = out[out["flip"].isin([f"only_{a}", f"only_{b}"])]
    sort_cols = ["flip"] + (["hyde_delta"] if "hyde_delta" in out else [])
    return out.sort_values(sort_cols).reset_index(drop=True)


def sc_retrieval_deltas(df, retried_only=True):
    """Row-level HyDE before/after: one row per rag_sc row with its original and final
    best retrieval score, the ``delta``, and whether that row ended up abstaining.

    ``sc_retrieval_gain`` averages exactly these deltas and reports ``frac_improved``;
    this is the table underneath, for the question a mean cannot answer — WHICH queries
    gained, and by how much. Sorted smallest delta first. ``retried_only`` keeps the
    rows HyDE actually re-ran; the rest have delta 0 by construction and would pad the
    table with non-events.

    WHAT THE DELTA MEANS DEPENDS ON WHICH MERGE PRODUCED THE FILE
    (``sc_metadata.merge_strategy``, absent on files written before 2026-08):

      - ``score`` (the old merge): a negative delta is arithmetically impossible.
        ``rag.utils._merge_by_score`` pooled both retrievals and kept the top-k by
        score, so the final best is ``max(orig, hyde)``. Do NOT read "0 rows got
        worse" off such a file as evidence that re-retrieval never hurts — the number
        cannot express that. Worse, the two sides were scored against different
        queries (question vs pseudo-answer), so the delta is not a like-for-like
        comparison at all.
      - ``rrf`` (current): the delta is identically ZERO, and this table is useless on
        such a file. Survivors are rescored under the original question, and the RRF
        tie-break always keeps the chunk that question ranked first — which held the
        highest question-score in the corpus — so ``max(final) == max(orig)`` on every
        row. Confirmed on the 2026-08-01 run: 0 of 368 rows differ.

    So NEITHER merge lets the best score say anything about re-retrieval. Under ``rrf``
    the effect shows up in the other two statistics instead: ``retrieval_spread`` widens
    (the HyDE chunk scores lower on the question's scale — that gap IS the recall/
    relevance trade, 0.039 -> 0.124 on the 2026-08-01 run), and
    ``sc_context_displacement`` counts how many original chunks the HyDE ones pushed
    out.

    Either way the score cannot show that re-retrieval HELPED. That question needs the
    gold context ids (synthetic subset) or the answer metrics — see
    ``analysis.eval_analysis``.
    """
    cols = ["id", "source_dataset", "lang", "retrieval_best_orig", "retrieval_best",
            "delta", "abstained", "query"]
    sc = (_hyde_rows(df) if retried_only else df[df["variant"] == "rag_sc"])
    if not len(sc) or "retrieval_best_orig" not in sc:
        return pd.DataFrame(columns=cols)
    out = sc.dropna(subset=["retrieval_best", "retrieval_best_orig"]).copy()
    out["delta"] = (pd.to_numeric(out["retrieval_best"], errors="coerce")
                    - pd.to_numeric(out["retrieval_best_orig"], errors="coerce"))
    out["abstained"] = _abstained(out)
    return out[[c for c in cols if c in out]].sort_values("delta").reset_index(drop=True)


def sc_context_displacement(df, retried_only=True, by=None):
    """How much of the ORIGINAL retrieved context each re-retrieved row kept.

    The score deltas cannot show whether re-retrieval hurt (see
    ``sc_retrieval_deltas``); this can, because it counts what actually reaches the
    model. Per row: ``n_orig`` / ``n_final`` chunk ids, ``kept`` (in both), ``dropped``
    = n_orig − kept, ``drop_rate``, plus whether the row abstained.
    ``dropped == n_orig`` means the question's own chunks were replaced wholesale.

    This is the metric that separates the two merge strategies
    (``sc_metadata.merge_strategy``): the old ``score`` merge could evict every
    original chunk and did so on 121 of 307 re-retrieved rows in the 2026-07-31 run;
    the ``rrf`` merge breaks its rank ties toward the question, so the question's
    top-ranked chunk always survives and ``dropped < n_orig`` holds by construction.
    Run this on a file from each to quantify the change.

    With ``by`` (e.g. ``"source_dataset"``) returns group means of the counts plus an
    ``ALL`` row instead of the row-level table.
    """
    cols = ["id", "source_dataset", "lang", "n_orig", "n_final", "kept", "dropped",
            "drop_rate", "abstained", "query"]
    orig_col = "sc_metadata.original_context_ids"
    sc = (_hyde_rows(df) if retried_only else df[df["variant"] == "rag_sc"])
    if not len(sc) or orig_col not in sc or "retrieved_context_ids" not in sc:
        return pd.DataFrame(columns=cols)

    def _as_set(v):
        return set(v) if isinstance(v, (list, tuple)) else set()

    out = sc.copy()
    o, f = out[orig_col].map(_as_set), out["retrieved_context_ids"].map(_as_set)
    out["n_orig"] = o.map(len)
    out["n_final"] = f.map(len)
    out["kept"] = [len(a & b) for a, b in zip(o, f)]
    out["dropped"] = out["n_orig"] - out["kept"]
    out["drop_rate"] = (out["dropped"] / out["n_orig"].where(out["n_orig"] > 0)).round(3)
    out["abstained"] = _abstained(out)
    out = out[[c for c in cols if c in out]]
    if by is None:
        return (out.sort_values(["dropped", "drop_rate"], ascending=False)
                .reset_index(drop=True))

    def _agg(g):
        return pd.Series({
            "n": len(g),
            "mean_dropped": g["dropped"].mean(),
            "mean_drop_rate": g["drop_rate"].mean(),
            "rows_all_replaced": int((g["dropped"] == g["n_orig"]).sum())
            if "n_orig" in g else 0,
            "abstention_rate": g["abstained"].mean(),
        })

    grouped = out.groupby(by, observed=True).apply(_agg, include_groups=False)
    grouped.loc["ALL"] = _agg(out)
    return _int_counts(grouped, ("n", "rows_all_replaced"))


def sc_retrieval_gain(df, by=None, retried_only=True):
    """Did the HyDE re-retrieval actually raise retrieval scores?

    Compares ``retrieval_best_orig`` (first retrieval) with ``retrieval_best``
    (post-HyDE) on rag_sc rows. By default restricts to rows that were actually
    re-retrieved (``retrieval_retried_count > 0``); otherwise the untouched rows
    would dilute the delta with zeros. Reports mean before/after, mean delta, and
    the fraction of rows that improved.

    ONLY MEANINGFUL ON A ``score``-MERGE FILE, and even there it cannot go negative.
    On an ``rrf``-merge file every column is 0 by construction (the question's own top
    chunk always survives and is rescored to its original value), so read
    ``sc_context_displacement`` and the spread instead. ``sc_retrieval_deltas`` spells
    out both cases and is the row-level table behind this one.

    The same before/after pair, as distributions rather than a delta, is the
    ``rag_sc_hyde_orig`` / ``rag_sc_hyde_final`` split in ``retrieval_variant_long``.
    """
    sc = (_hyde_rows(df) if retried_only else df[df["variant"] == "rag_sc"]).copy()
    sc = sc.dropna(subset=["retrieval_best", "retrieval_best_orig"])
    sc["delta"] = sc["retrieval_best"] - sc["retrieval_best_orig"]

    def _agg(g):
        return pd.Series({
            "n": len(g),
            "mean_orig": g["retrieval_best_orig"].mean(),
            "mean_final": g["retrieval_best"].mean(),
            "delta": g["delta"].mean(),
            "frac_improved": (g["delta"] > 0).mean(),
        })

    if by is None:
        return _int_counts(_agg(sc).to_frame("all").T)
    return _int_counts(sc.groupby(by, observed=True).apply(_agg, include_groups=False))


def retrieval_stage_long(df):
    """Long-form retrieval scores tagged by stage (rag / rag_sc_orig / rag_sc_final)
    so a single boxplot or groupby compares plain rag against rag_sc before and
    after re-retrieval. Rows without a retrieval score are dropped."""
    frames = []
    rag = df[df["variant"] == "rag"]
    frames.append(pd.DataFrame({"id": rag["id"], "source_dataset": rag["source_dataset"],
                                "stage": "rag", "value": rag["retrieval_best"]}))
    sc = df[df["variant"] == "rag_sc"]
    if "retrieval_best_orig" in df:
        frames.append(pd.DataFrame({"id": sc["id"], "source_dataset": sc["source_dataset"],
                                    "stage": "rag_sc_orig", "value": sc["retrieval_best_orig"]}))
    frames.append(pd.DataFrame({"id": sc["id"], "source_dataset": sc["source_dataset"],
                                "stage": "rag_sc_final", "value": sc["retrieval_best"]}))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    stages = [s for s in STAGE_ORDER if s in out["stage"].unique()]
    out["stage"] = pd.Categorical(out["stage"], categories=stages, ordered=True)
    return out


def confidence_stage_long(df, col="gen_logprob_stats.mean"):
    """Long-form generation confidence tagged by stage: no_rag / rag / rag_sc_final,
    plus rag_sc_orig (the pre-regeneration logprob) when a run has actually retried
    generation. This is the confidence analogue of ``retrieval_stage_long`` and is
    the thing you plot to compare 'rag_sc retry' logprobs against the initial ones.

    Falls back to just the three variant stages when ``gen_logprob_mean_orig`` is
    absent (no generation retry in the data yet)."""
    frames = []
    for v in VARIANT_ORDER:
        sub = df[df["variant"] == v]
        stage = "rag_sc_final" if v == "rag_sc" else v
        frames.append(pd.DataFrame({"id": sub["id"], "source_dataset": sub["source_dataset"],
                                    "stage": stage,
                                    "value": pd.to_numeric(sub[col], errors="coerce")}))
    if "gen_logprob_mean_orig" in df:
        sc = df[df["variant"] == "rag_sc"]
        frames.append(pd.DataFrame({"id": sc["id"], "source_dataset": sc["source_dataset"],
                                    "stage": "rag_sc_orig", "value": sc["gen_logprob_mean_orig"]}))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    stages = [s for s in CONF_STAGE_ORDER if s in out["stage"].unique()]
    out["stage"] = pd.Categorical(out["stage"], categories=stages, ordered=True)
    return out


def sc_generation_gain(df, by=None, retried_only=True):
    """Did regenerating change generation confidence?

    The logprob analogue of ``sc_retrieval_gain``: compares ``gen_logprob_mean_orig``
    (first generation) with ``gen_logprob_stats.mean`` (post-regeneration) on rag_sc
    rows, restricted by default to rows whose generation was actually retried. Returns
    an empty frame (with a note) when no generation retry exists in the data.

    NOT a like-for-like comparison — see the CAVEAT in the module docstring. The
    regen runs with thinking enabled, a repeat penalty and a stricter prompt, so
    its mean logprob is measured over a different token distribution and comes
    out lower by construction (``frac_improved`` is 0.0, not near it). Use this
    to quantify HOW different the two passes are, not to claim the correction
    hurt the model's certainty; ``sc_retrieval_gain`` has no such confound.
    """
    if "gen_logprob_mean_orig" not in df:
        return pd.DataFrame(columns=["n", "mean_orig", "mean_final", "delta", "frac_improved"])
    sc = df[df["variant"] == "rag_sc"].copy()
    if retried_only and "sc_metadata.generation_retried_count" in sc:
        sc = sc[sc["sc_metadata.generation_retried_count"].fillna(0) > 0]
    sc = sc.dropna(subset=["gen_logprob_stats.mean", "gen_logprob_mean_orig"])
    sc["delta"] = pd.to_numeric(sc["gen_logprob_stats.mean"], errors="coerce") - sc["gen_logprob_mean_orig"]

    def _agg(g):
        return pd.Series({
            "n": len(g),
            "mean_orig": g["gen_logprob_mean_orig"].mean(),
            "mean_final": pd.to_numeric(g["gen_logprob_stats.mean"], errors="coerce").mean(),
            "delta": g["delta"].mean(),
            "frac_improved": (g["delta"] > 0).mean(),
        })

    if by is None:
        return _int_counts(_agg(sc).to_frame("all").T)
    return _int_counts(sc.groupby(by, observed=True).apply(_agg, include_groups=False))


# --- rag_sc without any retry vs plain rag (pass-through check) ---------------

# The four ways a rag_sc row can come out of the self-correction, by which of the
# two budgets it spent. "none" is the pass-through cohort compared against rag below.
RETRY_KINDS = ["none", "retrieval_only", "generation_only", "both"]

_RETRY_COLS = (RETRIEVAL_RETRY_COL, GENERATION_RETRY_COL)

# Row fields compared between the paired rag_sc and rag rows. Suffixed ``_sc`` /
# ``_rag`` on the joined frame.
_PAIR_COLS = ["answer", "rejected", "contexts", "retrieved_context_ids",
              "retrieval_scores", "retrieval_best", "retrieval_spread",
              "gen_logprob_stats.mean"]


def _retry_kind(df):
    """Per rag_sc row, which corrections fired: one of ``RETRY_KINDS``.

    Reads the two retry counters; a missing counter column counts as 'did not fire'
    (the pipeline always writes both for rag_sc, so that only happens on a file with
    no sc_metadata at all)."""
    idx = df.index
    zero = pd.Series(0.0, index=idx)
    ret = (pd.to_numeric(df[_RETRY_COLS[0]], errors="coerce").fillna(0)
           if _RETRY_COLS[0] in df else zero) > 0
    gen = (pd.to_numeric(df[_RETRY_COLS[1]], errors="coerce").fillna(0)
           if _RETRY_COLS[1] in df else zero) > 0
    kind = pd.Series("none", index=idx, dtype="object")
    kind[ret & ~gen] = "retrieval_only"
    kind[~ret & gen] = "generation_only"
    kind[ret & gen] = "both"
    return kind


def sc_retry_breakdown(df, by=None):
    """How many rag_sc rows fired NEITHER correction, exactly one, or both.

    ``trigger_summary`` gives the two retry rates separately, which cannot answer
    'how many rows were left completely untouched' — the two rates overlap. This
    buckets each rag_sc row by its *combination* instead, so the ``none`` count is
    the size of the pass-through cohort that ``sc_passthrough_agreement`` then
    compares against plain rag.

    Without ``by``: one row per kind with ``n`` and ``share`` (of all rag_sc rows).
    With ``by`` (e.g. ``"source_dataset"``): counts per group, ``ALL`` row appended.
    """
    sc = df[df["variant"] == "rag_sc"]
    kind = _retry_kind(sc)
    if by is None:
        out = kind.value_counts().reindex(RETRY_KINDS, fill_value=0).to_frame("n")
        out["share"] = (out["n"] / len(sc)).round(3) if len(sc) else float("nan")
        out.index.name = "retry_kind"
        return out
    ct = pd.crosstab(sc[by], kind).reindex(columns=RETRY_KINDS, fill_value=0)
    ct.columns.name = "retry_kind"
    ct.loc["ALL"] = ct.sum()
    return ct


def _as_tuple(v):
    """List-like -> tuple (hashable, comparable); anything else (NaN, None) -> None."""
    return tuple(v) if isinstance(v, (list, tuple)) else None


def _norm_text(s):
    """Answer text normalised for comparison: collapse all whitespace runs to single
    spaces and strip. Two answers differing only in line wrapping count as equal."""
    return " ".join(str(s).split())


def _same_scores(a, b, tol):
    """Element-wise float comparison of two retrieval-score lists within ``tol``."""
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return False
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def sc_untouched_pairs(df, score_tol=1e-9):
    """Join every *untouched* rag_sc row (no retrieval and no generation retry) to
    its plain ``rag`` twin for the same id, and flag where the two differ.

    This is the row-level table behind the pass-through check described in the
    module docstring: those rag_sc rows ran the identical pipeline to rag, so each
    pair *should* agree on everything. Returns one row per paired id with the raw
    values from both sides (``*_sc`` / ``*_rag``) plus:

      - ``same_context_ids`` / ``same_context_set``: retrieved chunk ids, in order
        and as a set (a set match with an order mismatch would mean re-ranking
        differed, which the untouched path should never do);
      - ``same_contexts``: the context *texts*, in order;
      - ``same_retrieval_scores``: hybrid scores, within ``score_tol``;
      - ``same_answer`` (exact string) / ``same_answer_norm`` (whitespace-collapsed)
        / ``answer_similarity`` (difflib ratio in [0, 1] on the normalised text, so
        'differs by one clause' is distinguishable from 'a different answer');
      - ``same_rejected``: both abstained or both answered;
      - ``logprob_delta``: rag_sc minus rag mean token logprob.

    ids present as rag_sc-untouched but missing a rag row are dropped by the inner
    join; ``sc_passthrough_agreement`` reports how many that was.
    """
    sc = df[df["variant"] == "rag_sc"]
    sc = sc[_retry_kind(sc) == "none"]
    rag = df[df["variant"] == "rag"]

    cols = [c for c in _PAIR_COLS if c in df.columns]
    left_keep = ["id"] + (["source_dataset"] if "source_dataset" in df else []) + cols
    left = sc[left_keep].drop_duplicates("id")
    right = rag[["id"] + cols].drop_duplicates("id")
    out = left.merge(right, on="id", how="inner", suffixes=("_sc", "_rag"))

    def _pairwise(col, fn):
        a, b = f"{col}_sc", f"{col}_rag"
        if a not in out or b not in out:
            return None
        return [fn(x, y) for x, y in zip(out[a], out[b])]

    ids = _pairwise("retrieved_context_ids",
                    lambda a, b: (_as_tuple(a) is not None and _as_tuple(a) == _as_tuple(b)))
    if ids is not None:
        out["same_context_ids"] = ids
        out["same_context_set"] = _pairwise(
            "retrieved_context_ids",
            lambda a, b: (_as_tuple(a) is not None
                          and set(_as_tuple(a)) == set(_as_tuple(b) or ())))
    ctx = _pairwise("contexts",
                    lambda a, b: (_as_tuple(a) is not None and _as_tuple(a) == _as_tuple(b)))
    if ctx is not None:
        out["same_contexts"] = ctx
    sco = _pairwise("retrieval_scores", lambda a, b: _same_scores(a, b, score_tol))
    if sco is not None:
        out["same_retrieval_scores"] = sco

    if {"answer_sc", "answer_rag"} <= set(out.columns):
        a_sc = out["answer_sc"].fillna("").astype(str)
        a_rag = out["answer_rag"].fillna("").astype(str)
        n_sc = a_sc.map(_norm_text)
        n_rag = a_rag.map(_norm_text)
        out["same_answer"] = (a_sc == a_rag).tolist()
        out["same_answer_norm"] = (n_sc == n_rag).tolist()
        out["answer_similarity"] = [difflib.SequenceMatcher(None, x, y).ratio()
                                    for x, y in zip(n_sc, n_rag)]
        out["answer_chars_sc"] = n_sc.str.len()
        out["answer_chars_rag"] = n_rag.str.len()
    if {"rejected_sc", "rejected_rag"} <= set(out.columns):
        out["same_rejected"] = (out["rejected_sc"].fillna(False).astype(bool)
                                == out["rejected_rag"].fillna(False).astype(bool)).tolist()
    lp = "gen_logprob_stats.mean"
    if {f"{lp}_sc", f"{lp}_rag"} <= set(out.columns):
        out["logprob_delta"] = (pd.to_numeric(out[f"{lp}_sc"], errors="coerce")
                                - pd.to_numeric(out[f"{lp}_rag"], errors="coerce"))
    return out


# The agreement flags, in report order, mapped to the rate column they produce.
_AGREEMENT_FLAGS = [
    ("same_context_ids", "frac_same_context_ids"),
    ("same_context_set", "frac_same_context_set"),
    ("same_contexts", "frac_same_contexts"),
    ("same_retrieval_scores", "frac_same_retrieval_scores"),
    ("same_answer", "frac_same_answer"),
    ("same_answer_norm", "frac_same_answer_norm"),
    ("same_rejected", "frac_same_rejected"),
]


def sc_passthrough_agreement(df, by=None, pairs=None, with_total=True):
    """Do the untouched rag_sc rows reproduce their plain rag twin? — the rates.

    Aggregates ``sc_untouched_pairs`` into one row (or one per ``by`` group) with
    ``n`` pairs, the ``frac_same_*`` share for each compared field, plus
    ``mean_answer_similarity`` and ``mean_abs_logprob_delta`` to size the
    disagreement where it isn't exact.

    Expected reading: the retrieval fields are 1.0 — the untouched path never
    re-retrieves, so anything below 1.0 there is a bug worth chasing. The answer
    fields are ≤ 1.0 in practice despite greedy decoding, because llama.cpp is not
    bit-reproducible across requests; ``mean_answer_similarity`` near 1.0 with a few
    non-identical rows is that noise, a low similarity would not be.

    With ``by`` an ``ALL`` row is appended (``with_total``), pooled over every pair
    rather than averaging the per-group rates.
    """
    pairs = sc_untouched_pairs(df) if pairs is None else pairs
    flags = [(f, name) for f, name in _AGREEMENT_FLAGS if f in pairs.columns]

    def _agg(g):
        row = {"n": len(g)}
        for f, name in flags:
            row[name] = round(g[f].mean(), 3) if len(g) else float("nan")
        if "answer_similarity" in g:
            row["mean_answer_similarity"] = round(g["answer_similarity"].mean(), 3)
        if "logprob_delta" in g:
            row["mean_abs_logprob_delta"] = round(g["logprob_delta"].abs().mean(), 4)
        return pd.Series(row)

    if by is None or by not in pairs:
        out = _agg(pairs).to_frame("all").T
    else:
        out = pairs.groupby(by, observed=True).apply(_agg, include_groups=False)
        if with_total and len(pairs):
            out.loc["ALL"] = _agg(pairs)
    out["n"] = out["n"].astype(int)
    return out


# Averages compared side by side on the pass-through cohort: (label, sc column,
# rag column). Everything here is a per-row number the two variants should agree on.
_PASSTHROUGH_MEANS = [
    ("retrieval_best", "retrieval_best_sc", "retrieval_best_rag"),
    ("retrieval_spread", "retrieval_spread_sc", "retrieval_spread_rag"),
    ("gen_logprob_mean", "gen_logprob_stats.mean_sc", "gen_logprob_stats.mean_rag"),
    ("rejected_rate", "rejected_sc", "rejected_rag"),
    ("answer_chars", "answer_chars_sc", "answer_chars_rag"),
]


def sc_passthrough_means(df, pairs=None):
    """The averages half of the pass-through check: rag_sc vs rag means over the
    SAME untouched ids.

    One row per quantity with ``n`` / ``mean_rag`` / ``mean_rag_sc`` / ``delta``
    (rag_sc − rag) / ``mean_abs_delta`` (the paired per-row gap, which does not
    cancel out the way ``delta`` does). Since these are the same pipeline run twice,
    ``delta`` should sit at ~0: a delta near zero with a non-zero
    ``mean_abs_delta`` is symmetric run-to-run noise, a systematically signed delta
    would mean the two paths are not equivalent after all.
    """
    pairs = sc_untouched_pairs(df) if pairs is None else pairs
    rows = {}
    for label, c_sc, c_rag in _PASSTHROUGH_MEANS:
        if c_sc not in pairs or c_rag not in pairs:
            continue
        # astype(float): `rejected` is boolean, and bool - bool is not subtraction.
        s = pd.to_numeric(pairs[c_sc], errors="coerce").astype(float)
        r = pd.to_numeric(pairs[c_rag], errors="coerce").astype(float)
        m = s.notna() & r.notna()
        rows[label] = {
            "n": int(m.sum()),
            "mean_rag": r[m].mean(),
            "mean_rag_sc": s[m].mean(),
            "delta": (s[m] - r[m]).mean(),
            "mean_abs_delta": (s[m] - r[m]).abs().mean(),
        }
    return pd.DataFrame(rows).T


def sc_passthrough_diffs(df, pairs=None, n=10, width=200, on="same_answer_norm"):
    """The concrete examples: untouched pairs whose answers did NOT match, least
    similar first, with both answer texts truncated to ``width`` chars.

    ``on="same_answer_norm"`` ignores pure whitespace/wrapping differences (the
    default — those are not interesting); pass ``on="same_answer"`` for strict
    byte equality, or ``on="same_contexts"`` to list retrieval divergences (which
    should be empty).
    """
    pairs = sc_untouched_pairs(df) if pairs is None else pairs
    if on not in pairs.columns:
        return pd.DataFrame()
    bad = pairs[~pairs[on].astype(bool)].copy()
    if "answer_similarity" in bad:
        bad = bad.sort_values("answer_similarity")
    keep = [c for c in ("id", "source_dataset", "answer_similarity",
                        "logprob_delta", "rejected_rag", "rejected_sc")
            if c in bad.columns]
    out = bad.head(n)[keep + ["answer_rag", "answer_sc"]].copy()
    for c in ("answer_rag", "answer_sc"):
        out[c] = out[c].fillna("").astype(str).map(_norm_text).str.slice(0, width)
    return out


# --- Cross-linking with the evaluated results --------------------------------

# Pipeline-side signals carried over onto the evaluated metrics for joint analysis.
_LINK_FEATURES = [
    "gen_logprob_stats.mean", "retrieval_best", "retrieval_spread",
    "retrieval_best_orig", "retrieval_spread_orig",
    "sc_metadata.retrieval_retried_count", "sc_metadata.generation_retried_count",
]


def link_eval(rag_df, eval_df):
    """Join pipeline signals (this file) onto evaluated metrics, per (id, variant).

    Brings the retrieval/confidence/SC-retry columns from the raw results — suffixed
    ``_rag`` where a name also exists on the evaluated side — alongside every RAGAS/
    DeepEval score, so you can ask e.g. 'does a bigger HyDE re-retrieval gain predict
    a higher answer-correctness score?'. Also adds ``reretrieval_gain`` (final minus
    original best retrieval score). Inner join, so only ids present in both files
    survive — use rag + evaluated files from the *same* run for a full overlap.
    """
    feats = [c for c in _LINK_FEATURES if c in rag_df.columns]
    left = rag_df[["id", "variant"] + feats].copy()
    if {"retrieval_best", "retrieval_best_orig"} <= set(left.columns):
        left["reretrieval_gain"] = left["retrieval_best"] - left["retrieval_best_orig"]
    left["variant"] = left["variant"].astype(str)

    right = eval_df.copy()
    right["variant"] = right["variant"].astype(str)
    return left.merge(right, on=["id", "variant"], how="inner", suffixes=("_rag", ""))


def top_n(df, by, n=10, ascending=True, cols=None):
    """The ``n`` rows ranked by column ``by`` (ascending=worst-first by default).

    ``n`` is just a parameter — pass any count. Returns a compact view (id, variant,
    source_dataset, the ranking column) unless ``cols`` overrides the displayed set.
    Handy on a ``link_eval`` frame: ``top_n(linked, "ragas_scores.ragas_answer_correctness")``
    for the worst-scored queries, or ``by="reretrieval_gain", ascending=False`` for the
    biggest re-retrieval movers.
    """
    show = cols or [c for c in ("id", "variant", "source_dataset", by) if c in df.columns]
    return df.sort_values(by, ascending=ascending).head(n)[show]


if __name__ == "__main__":
    import sys

    # Imported here, not at module level: plots imports this module back, and a
    # top-level import would make that a cycle.
    from analysis import plots

    path = sys.argv[1] if len(sys.argv) > 1 else latest_results(RAG_PREFIX)
    d = load(path)

    # Everything printed below is collected and saved to this run's
    # reports/ folder, then echoed to the console.
    with paths.capture(path, "rag_analysis_report"):
        # These are the TOTAL rows loaded (no cross-linking yet). Every id should carry
        # every variant, so a complete file has n_ids x n_variants rows; `missing` counts
        # the (id, variant) combinations absent from that full grid and should be 0.
        n_ids = d["id"].nunique()
        present_variants = [v for v in d["variant"].cat.categories
                            if v in set(d["variant"].dropna())]
        full_grid = n_ids * len(present_variants)
        missing = full_grid - len(d.drop_duplicates(["id", "variant"]))
        print(f"{path}: {len(d)} total rows = {n_ids} ids x {len(present_variants)} "
              f"variants {present_variants} (full grid {full_grid}); "
              f"{missing} id x variant combos missing (0 = every id has every variant)")

        print(f"writing every artifact to {paths.rel(paths.run_dir(path))}")

        print("\n=== rag health check ===")
        rag_health_report(d, source=path)

        # --- Error analyses: what went WRONG in the run, before any results are read --
        print("\n=== generations cut off at the token cap, by dataset (rag_sc only) ===")
        print(f"finish_reason covers {coverage_note(d)}; n below counts those rows only.")
        print("Restricted to rag_sc: only its regeneration can loop, so pooling in the "
              "single-pass no_rag/rag rows would just dilute length_rate.")
        print("length = hit max_tokens; thinking_loop = cut off with no answer text "
              "(runaway regen); answer_truncated = answer written but tail lost.")
        _trunc = truncation_summary(d[d["variant"] == "rag_sc"])
        if len(_trunc):
            print(_trunc.to_string())
        else:
            print("no finish_reason in this file")

        # Pass-through check: rag_sc rows that fired NEITHER correction ran the identical
        # pipeline to rag, so any divergence is noise (answers) or a bug (contexts). The
        # per-correction breakdown of all rag_sc rows sits further down, with the triggers.
        pairs = sc_untouched_pairs(d)
        n_untouched = int((_retry_kind(d[d["variant"] == "rag_sc"]) == "none").sum())
        print(f"\n=== pass-through check: the {n_untouched} rag_sc rows that fired NEITHER "
              f"correction, vs their rag twin ===")
        print(f"{len(pairs)} of them have a rag row for the same id "
              f"({n_untouched - len(pairs)} unpaired). These rows ran the identical pipeline "
              f"to rag (same retrieval, same prompts, greedy decoding), so every field below "
              f"should match; answer mismatches are llama.cpp run-to-run noise, context "
              f"mismatches would be a bug.")
        if len(pairs):
            print("\nagreement rates:")
            print(sc_passthrough_agreement(d, pairs=pairs).to_string())
            print("\n--- same, per dataset ---")
            print(sc_passthrough_agreement(d, by="source_dataset", pairs=pairs).to_string())

            print("\naverages over the same ids (delta should be ~0):")
            print(sc_passthrough_means(d, pairs=pairs).round(4).to_string())

            diffs = sc_passthrough_diffs(d, pairs=pairs)
            print(f"\nconcrete diverging answers ({int((~pairs['same_answer_norm']).sum())} of "
                  f"{len(pairs)}), least similar first:")
            print(diffs.round(3).to_string(index=False) if len(diffs)
                  else "none — every untouched rag_sc answer reproduced its rag twin")

            ctx_diffs = sc_passthrough_diffs(d, pairs=pairs, on="same_contexts")
            if len(ctx_diffs):
                print(f"\n[CHECK] {len(ctx_diffs)} untouched pairs retrieved DIFFERENT contexts "
                      f"— the untouched path never re-retrieves, so this is a bug:")
                print(ctx_diffs[[c for c in ctx_diffs.columns
                                 if c not in ("answer_rag", "answer_sc")]].to_string(index=False))

            _p = paths.table(path, "rag_sc_passthrough_pairs")
            pairs.to_csv(_p, index=False)
            print(f"\nwrote {paths.rel(_p)} (one row per untouched pair)")
        else:
            print("no untouched rag_sc rows with a rag twin in this file — nothing to compare")

        # Abstentions. NOT an error — refusing on insufficient context is the behaviour the
        # pipeline is built to have — but it belongs beside the error analyses, because it
        # is the other way a row comes back without a real answer, and because the `empty`
        # column below is exactly the thinking-loop failure counted above, wearing an
        # abstention's clothes.
        print("\n=== abstentions (answer collapsed onto the canonical rejection) ===")
        # no_rag is excluded from the count and signal tables below: it has no context to
        # call insufficient, so it cannot abstain by construction. Leaving it in would not
        # add a 0 row, it would drag every pooled figure (the ALL row, each dataset's `all`
        # row, the ratios in the signal table) down by a third for a structural reason.
        _abst = d[d["variant"] != "no_rag"]
        print("rag / rag_sc only — no_rag has no context to call insufficient, so it never "
              "abstains and would only deflate the pooled rates.")
        print("model_rejected = the model itself abstained (the intended behaviour); "
              "empty = nothing survived stripping <think>, i.e. the runaway regen above.")
        print(abstention_summary(_abst).to_string())
        print("\n--- same, per dataset x variant (all = rag + rag_sc pooled per dataset) ---")
        print(abstention_by_dataset_variant(_abst).to_string())

        print("\n--- same, per language x variant ---")
        print("The run is multilingual (one prompt language per query), so a refusal rate "
              "that differs by language would be a retrieval-language problem, not a topic "
              "one. Check n before reading anything into it — the two subsets are far apart "
              "in size.")
        print(abstention_summary(_abst, by=["lang", "variant"], with_total=False).to_string())

        print("\n--- what the pipeline's own signals looked like when it abstained ---")
        print("retrieval_best lower on the abstained rows = abstention tracks weak "
              "retrieval, which is what it is for. gen_logprob_mean is NOT a confidence "
              "reading here: the canonical rejection is a short formulaic sentence, so its "
              "tokens are trivially predictable (gen_tokens shows the length gap behind it).")
        print(abstention_signals(_abst).round(3).to_string())

        # The false-refusal set: the abstentions the retrieval score does NOT excuse.
        _thr = trigger_threshold(d)
        _fr = false_refusals(d)
        _n_abst = int(_abstained(_abst).sum())
        print(f"\n--- false refusals: abstained although retrieval_best >= {_thr} "
              f"({len(_fr)} of {_n_abst} abstentions) ---")
        print(f"{_thr} is the run's OWN retrieval threshold, read back out of the recorded "
              f"trigger strings: at or above it the pipeline judged the context good enough "
              f"not to re-retrieve. So these rows are not 'the corpus does not cover it' — "
              f"they are a topical retrieval miss or an over-cautious model, and they are "
              f"the abstentions worth reading by hand. Best-supported (least defensible) "
              f"first; `empty` rows excluded, those are truncations, not refusals.")
        if len(_fr):
            _show = [c for c in _fr.columns if c != "query"]
            print(_fr[_show].round(3).to_string(index=False))
            _p = paths.table(path, "rag_false_refusals")
            _fr.to_csv(_p, index=False)
            print(f"wrote {paths.rel(_p)} ({len(_fr)} rows, with the query text)")
        else:
            print("none — every abstention came with retrieval below the threshold")

        print("\n--- did self-correction change WHO abstains? (per id, rag_sc vs rag) ---")
        print("only_rag_sc = the corrections made it give up where plain rag answered; "
              "only_rag = the reverse. net is the change in abstention count.")
        print(abstention_transitions(d).to_string())

        # Which ids those are, and whether HyDE explains them. hyde_delta = rag_sc's final
        # minus original retrieval score: negative means the re-retrieval genuinely fetched
        # worse context for that query, NaN means it never re-retrieved (so it cannot be
        # the cause).
        _flips = abstention_transition_rows(d)
        if len(_flips):
            _newly = _flips[_flips["flip"] == "only_rag_sc"]
            _worse = int((_newly["hyde_delta"] < 0).sum()) if "hyde_delta" in _newly else 0
            _untouched = int(_newly["hyde_delta"].isna().sum()) if "hyde_delta" in _newly else 0
            print(f"\nthe {len(_flips)} ids that flipped, worst hyde_delta first "
                  f"(of the {len(_newly)} newly refusing: {_worse} lost retrieval score to "
                  f"HyDE, {_untouched} never re-retrieved at all):")
            print(_flips[[c for c in _flips.columns if c != "query"]].round(3).to_string(index=False))
            _p = paths.table(path, "rag_abstention_flips")
            _flips.to_csv(_p, index=False)
            print(f"wrote {paths.rel(_p)} ({len(_flips)} rows, with the query text)")

        print("\n--- rag_sc abstention rate by which corrections fired ---")
        print("Read as selection, not effect: the generation trigger keys on the mean "
              "logprob, which an abstention maximises, so an abstaining first pass hardly "
              "ever trips a regeneration — the retried cohorts are pre-filtered to rows "
              "that answered, they were not rescued by the retry.")
        _sc = d[d["variant"] == "rag_sc"]
        _by_kind = abstention_summary(_sc.assign(retry_kind=_retry_kind(_sc)), by="retry_kind")
        print(_by_kind.reindex([k for k in RETRY_KINDS + ["ALL"] if k in _by_kind.index])
              .to_string())

        print("\n=== retrieval best-score by variant ===")
        print("rag_sc = all its rows at their final score (the average); rag_sc_hyde_orig/"
              "_final = only the rows where HyDE re-retrieval was triggered, before vs after.")
        print(retrieval_variant_summary(d).round(3).to_string())
        print("\n=== retrieval best-score by dataset x variant "
              "(all = rag + rag_sc pooled per dataset) ===")
        print(retrieval_by_dataset_variant(d).round(3).to_string())

        print("\n=== generation confidence (mean logprob) by variant ===")
        print(logprob_summary(d).round(3).to_string())
        print("\n=== generation confidence by dataset x variant "
              "(all = variants pooled per dataset) ===")
        print(logprob_by_dataset_variant(d).round(3).to_string())

        print("\n=== self-correction retry rates by dataset ===")
        print(trigger_summary(d).to_string())
        print("\n=== which retrieval triggers fired (all rag_sc rows) ===")
        print("total firings per trigger (a row can fire more than one):")
        print(trigger_reasons(d).to_string())
        print("\nretrieval per trigger combination (only-highest / only-spread / both):")
        print(trigger_combinations(d).to_string())

        print("\n=== which generation triggers fired (all rag_sc rows) ===")
        print("total firings per trigger (a row can fire more than one):")
        print(trigger_reasons(d, stage="generation").to_string())
        print("\ngeneration per trigger combination (only-mean / only-min / both):")
        print(trigger_combinations(d, stage="generation").to_string())

        print("\n=== rag_sc rows by which corrections fired ===")
        print(sc_retry_breakdown(d).to_string())
        print("\n--- same, per dataset ---")
        print(sc_retry_breakdown(d, by="source_dataset").to_string())

        # --- Did the corrections pay off? The hand-over to analysis.eval_analysis ---
        # Last, because these are the only blocks that ask whether self-correction
        # IMPROVED anything; eval_analysis picks the same question back up against the
        # quality metrics (`reretrieval_gain` vs metric, paired confidence deltas).
        print("\n=== HyDE re-retrieval gain (retried rag_sc rows only) ===")
        print(sc_retrieval_gain(d).round(3).to_string())
        print("\n--- same, per dataset ---")
        print(sc_retrieval_gain(d, by="source_dataset").round(3).to_string())

        # The gain above cannot answer "did re-retrieval make anything worse" — on a
        # `score`-merge file the delta cannot go negative, on an `rrf`-merge file it cannot
        # go positive (sc_retrieval_deltas explains both). What DID change is which chunks
        # reached the model, which is what displacement counts.
        _disp = sc_context_displacement(d)
        print(f"\n--- what re-retrieval did to the CONTEXT SET ({len(_disp)} re-retrieved "
              f"rows) ---")
        if len(_disp):
            _merge = (d.get("sc_metadata.merge_strategy", pd.Series(dtype="object"))
                      .dropna().unique())
            print(f"merge strategy in this file: "
                  f"{', '.join(map(str, _merge)) if len(_merge) else 'score (pre-2026-08 file)'}")
            print(f"{int((_disp['dropped'] > 0).sum())} rows lost at least one original "
                  f"chunk; {int((_disp['dropped'] == _disp['n_orig']).sum())} lost ALL of "
                  f"them (impossible under the rrf merge, which always keeps the question's "
                  f"top chunk); mean drop rate {_disp['drop_rate'].mean():.3f}.")
            print(sc_context_displacement(d, by="source_dataset").round(3).to_string())
            _p = paths.table(path, "rag_hyde_context_displacement")
            _disp.to_csv(_p, index=False)
            print(f"wrote {paths.rel(_p)} ({len(_disp)} rows with the query text, "
                  f"most displaced first)")
            _deltas = sc_retrieval_deltas(d)
            if len(_deltas):
                _p = paths.table(path, "rag_hyde_retrieval_deltas")
                _deltas.to_csv(_p, index=False)
                print(f"wrote {paths.rel(_p)} (per-row score deltas, smallest first "
                      f"— read sc_retrieval_deltas' docstring first)")
        else:
            print("no re-retrieved rows with recorded context ids in this file")

        print("\n=== generation-retry confidence gain (retried rag_sc rows only) ===")
        gen_gain = sc_generation_gain(d)
        print(gen_gain.round(3).to_string() if len(gen_gain)
              else "no generation retries in this file (sc_metadata.original_gen_logprob_stats absent)")

        # Both gains are raw pipeline signals — whether they translate into better
        # answers is the cross-link with the evaluated results (link_eval join,
        # linked_*.parquet base table, worst/best-query mining), which now lives in
        # analysis.eval_analysis; `link_eval` / `top_n` remain here as the join helpers
        # it imports. Run `python -m analysis.eval_analysis` for that step.

        print("\n=== figures ===")
        plots.save_all({
            "rag_confidence_by_variant": lambda ax: plots.confidence_boxplot(d, ax=ax),
            "rag_confidence_by_dataset": lambda ax: plots.confidence_by_dataset(d, ax=ax),
            "rag_confidence_by_stage": lambda ax: plots.confidence_stage_boxplot(d, ax=ax),
            "rag_retrieval_by_stage": lambda ax: plots.retrieval_stage_boxplot(d, ax=ax),
            "rag_retrieval_by_dataset": lambda ax: plots.retrieval_by_dataset(d, ax=ax),
            "rag_sc_reretrieval_slope": lambda ax: plots.sc_retrieval_slope(d, ax=ax),
        }, path)
