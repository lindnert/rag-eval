"""Loading + plotting for the raw RAG pipeline results (pre-evaluation).

    from analysis.rag_analysis import load, logprob_summary, retrieval_stage_boxplot
    from common.results_io import RAG_PREFIX, latest_results
    df = load(latest_results(RAG_PREFIX))     # or load("results/rag_results_<ts>.json")
    logprob_summary(df)                       # confidence per variant
    retrieval_by_dataset_variant(df)          # retrieval per dataset x variant (+HyDE split)
    retrieval_stage_boxplot(df)               # rag vs rag_sc-orig vs rag_sc-final

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
rows); what the pipeline MEASURED (confidence and retrieval summaries,
self-correction retry rates, the retrieval- AND generation-trigger tallies); and
whether correction PAID OFF (HyDE re-retrieval and generation-retry gains) —
last, because that is where ``analysis.eval_analysis`` takes over. Then the PNGs.
It writes the ``rag_*.png`` files and a timestamped ``rag_health_*.txt`` into
``analysis/``.

The cross-link with the evaluated results — the ``link_eval`` join, the joined
``analysis/linked_<rag_stem>.parquet`` base table, and the worst/best-query
mining — lives in ``analysis.eval_analysis`` (which imports ``link_eval`` /
``top_n`` from here). Run ``python -m analysis.eval_analysis`` for that.
"""

import difflib
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.results_io import RAG_PREFIX, latest_results

try:
    import seaborn as sns
except ImportError:  # seaborn is optional; functions fall back to matplotlib
    sns = None

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


def rag_health_report(df, source=None, out_dir="analysis"):
    """Structural sanity check on the *raw* pipeline results; prints, and writes a
    timestamped file when ``source`` is given (companion to ``analysis.health_report``).

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

    text = "\n".join(lines)
    print(text)
    if source is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(str(source)).stem
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        p = Path(out_dir) / f"rag_health_{stem}_{ts}.txt"
        p.write_text(text, encoding="utf-8")
        print(f"wrote {p}")
        flags["report_path"] = str(p)
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


def _with_pooled_rows(base, pooled, order=None, level="variant", label="all"):
    """Interleave a per-(source_dataset, group) stats table with one pooled row per
    dataset, appended after that dataset's group rows.

    ``base`` is indexed by (source_dataset, group), ``pooled`` by source_dataset
    alone. The pooled row is recomputed over rows, not averaged from the group
    means, so uneven group sizes don't misweight it. Shared by
    ``logprob_by_dataset_variant`` and ``retrieval_by_dataset_variant`` so both
    tables read the same way.
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
    return out


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


def sc_retrieval_gain(df, by=None, retried_only=True):
    """Did the HyDE re-retrieval actually raise retrieval scores?

    Compares ``retrieval_best_orig`` (first retrieval) with ``retrieval_best``
    (post-HyDE) on rag_sc rows. By default restricts to rows that were actually
    re-retrieved (``retrieval_retried_count > 0``); otherwise the untouched rows
    would dilute the delta with zeros. Reports mean before/after, mean delta, and
    the fraction of rows that improved.

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
        return _agg(sc).to_frame("all").T
    return sc.groupby(by, observed=True).apply(_agg, include_groups=False)


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
        return _agg(sc).to_frame("all").T
    return sc.groupby(by, observed=True).apply(_agg, include_groups=False)


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


# --- Plots -------------------------------------------------------------------

def _boxplot(df, x, y, ax=None, order=None):
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


def confidence_boxplot(df, by="variant", ax=None):
    """Generation confidence (mean token logprob) by variant (or any category)."""
    order = _order(df) if by == "variant" else None
    ax = _boxplot(df, by, "gen_logprob_stats.mean", ax=ax, order=order)
    ax.set_title(f"Generation confidence (mean logprob) by {by}")
    return ax


def confidence_by_dataset(df, ax=None):
    """Mean-logprob distribution per dataset, split by variant."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    if sns is not None:
        sns.boxplot(data=df, x="source_dataset", y="gen_logprob_stats.mean",
                    hue="variant", hue_order=_order(df), ax=ax)
    else:
        ax = _boxplot(df, "source_dataset", "gen_logprob_stats.mean", ax=ax)
    ax.set_title("Generation confidence by dataset and variant")
    ax.tick_params(axis="x", rotation=30)
    return ax


def retrieval_stage_boxplot(df, ax=None):
    """Best retrieval score across rag / rag_sc-orig / rag_sc-final stages."""
    long = retrieval_stage_long(df)
    order = [s for s in STAGE_ORDER if s in long["stage"].unique()]
    ax = _boxplot(long, "stage", "value", ax=ax, order=order)
    ax.set_title("Best retrieval score by stage (hybrid dense+BM25)")
    return ax


def confidence_stage_boxplot(df, ax=None):
    """Generation confidence across no_rag / rag / rag_sc-orig / rag_sc-final stages.
    The rag_sc_orig box appears only once a run has retried generation."""
    long = confidence_stage_long(df)
    order = [s for s in CONF_STAGE_ORDER if s in long["stage"].unique()]
    ax = _boxplot(long, "stage", "value", ax=ax, order=order)
    ax.set_title("Generation confidence (mean logprob) by stage")
    return ax


def retrieval_by_dataset(df, ax=None, col="retrieval_best"):
    """Best retrieval score per dataset, split by retrieval variant: rag, the rag_sc
    average, and the HyDE-retried rag_sc rows before/after re-retrieval — the plot of
    ``retrieval_by_dataset_variant`` (the two hyde boxes cover that subset only)."""
    ax = ax or plt.subplots(figsize=(9, 5))[1]
    long = retrieval_variant_long(df, col=col)
    order = [v for v in RETRIEVAL_VARIANT_ORDER if v in set(long["variant"])]
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
    sc = _hyde_rows(df).dropna(subset=["retrieval_best", "retrieval_best_orig"]).copy()
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


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else latest_results(RAG_PREFIX)
    d = load(path)
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

    print("\n=== rag health check (also written to a timestamped file) ===")
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

        pairs.to_csv("analysis/rag_sc_passthrough_pairs.csv", index=False)
        print("\nwrote analysis/rag_sc_passthrough_pairs.csv (one row per untouched pair)")
    else:
        print("no untouched rag_sc rows with a rag twin in this file — nothing to compare")

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

    print("\n=== generation-retry confidence gain (retried rag_sc rows only) ===")
    gen_gain = sc_generation_gain(d)
    print(gen_gain.round(3).to_string() if len(gen_gain)
          else "no generation retries in this file (sc_metadata.original_gen_logprob_stats absent)")

    # Both gains are raw pipeline signals — whether they translate into better
    # answers is the cross-link with the evaluated results (link_eval join,
    # linked_*.parquet base table, worst/best-query mining), which now lives in
    # analysis.eval_analysis; `link_eval` / `top_n` remain here as the join helpers
    # it imports. Run `python -m analysis.eval_analysis` for that step.

    figs = {
        "rag_confidence_by_variant": lambda ax: confidence_boxplot(d, ax=ax),
        "rag_confidence_by_dataset": lambda ax: confidence_by_dataset(d, ax=ax),
        "rag_confidence_by_stage": lambda ax: confidence_stage_boxplot(d, ax=ax),
        "rag_retrieval_by_stage": lambda ax: retrieval_stage_boxplot(d, ax=ax),
        "rag_retrieval_by_dataset": lambda ax: retrieval_by_dataset(d, ax=ax),
        "rag_sc_reretrieval_slope": lambda ax: sc_retrieval_slope(d, ax=ax),
    }
    for name, fn in figs.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        fn(ax)
        fig.savefig(f"analysis/{name}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote analysis/{name}.png")
