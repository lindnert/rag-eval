"""Where every generated artifact goes: one folder per analysed results file, and
every filename says which results file(s) produced it.

Every table, figure and report the analysis scripts write is derived from one or
two results JSONs, so the output folder is keyed by the primary file's stem::

    analysis/out/<results-stem>/
        figures/   *.png       the plots
        tables/    *.csv       the summary tables
        reports/   *.txt       the full console report (see ``capture``)
        linked/    *.parquet   the joined rag+eval base table

That layout fixes the two problems the flat ``analysis/`` directory had. Outputs
from different runs no longer overwrite each other silently. And re-analysing the
SAME file overwrites in place instead of appending yet another timestamped
report, which is what grew the old directory to 52 ``*_health_*.txt`` files for 9
actual runs. The timestamp still goes INSIDE each report, so "when was this
generated" is not lost — only the filename pileup is.

THE FILENAME CARRIES THE SOURCE TOO, deliberately duplicating what the folder
already says::

    out/evaluated_results_20260729_070055/figures/coverage__eval-20260729_070055.png

The redundancy is the point: a figure only lives in that folder until it is
dragged into a thesis, a slide or a LaTeX ``\\includegraphics``, and from then on
the filename is the only provenance it has. A directory of ``coverage.png``,
``rejection.png`` … is unusable six months later; ``coverage__eval-20260729_070055``
still identifies its run. Tags are compacted (``evaluated_results_`` -> ``eval-``,
``rag_results_`` -> ``rag-``) to keep the deep OneDrive paths clear of the Windows
260-character limit.

A figure built from BOTH files — anything on the linked rag+eval frame — is named
for both, primary first::

    paths.figure([eval_path, rag_path], "eval_retrieval_vs_faithfulness")
    -> .../figures/eval_retrieval_vs_faithfulness__eval-20260729_070055__rag-20260801_181144.png

so the pairing behind a cross-linked plot is readable off the file itself. The
folder is always keyed on the FIRST source; ``analysis.eval_analysis`` passes its
eval file first, since that is its primary input.

Call sites use the four named helpers, which supply the conventional extension::

    paths.table(src, "metric_summary")   -> .../tables/metric_summary__<tag>.csv
    paths.figure(src, "coverage")        -> .../figures/coverage__<tag>.png
    paths.report(src, "health")          -> .../reports/health__<tag>.txt
    paths.linked(src, "linked")          -> .../linked/linked__<tag>.parquet

The whole tree is gitignored (``analysis/out/``): everything under it is
reproducible by re-running the scripts over the JSON in ``results/``.
"""

import io
import sys
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from pathlib import Path

# Resolved from THIS FILE rather than the cwd, so the scripts write to the same
# place no matter where they are launched from. (They are documented to run from
# the repo root, but nothing enforces that, and a stray cwd silently scattering
# output into a second `analysis/out` is exactly the mess this module exists to
# prevent.)
OUT_ROOT = Path(__file__).resolve().parent / "out"

# The four artifact kinds, each its own subfolder, with the extension assumed when
# a caller passes a bare name.
KINDS = {
    "figures": ".png",
    "tables": ".csv",
    "reports": ".txt",
    "linked": ".parquet",
}

# Long results-file prefixes shortened in the filename tag. Order matters only in
# that each stem matches at most one of these.
_TAGS = (("evaluated_results", "eval"), ("rag_results", "rag"))

# Separates the artifact name from the tag, and the tags from each other. Double
# underscore so it stays visually distinct from the single underscores inside both
# the artifact names and the timestamps.
SEP = "__"


def _sources(source):
    """Normalise one path or a sequence of paths to a list."""
    if isinstance(source, (str, Path)):
        return [source]
    return list(source)


def run_key(source):
    """The folder name for a results file: its stem, in full.

    ``results/rag_results_20260801_181144.json`` -> ``rag_results_20260801_181144``.
    Given several sources, the FIRST one keys the folder. Idempotent on a bare
    stem, so passing either the path or the key works.
    """
    return Path(str(_sources(source)[0])).stem


def short_key(source):
    """One results file compacted for use inside a filename.

    ``evaluated_results_20260729_070055`` -> ``eval-20260729_070055``;
    ``rag_results_20260801_181144``       -> ``rag-20260801_181144``.
    An eval file named ``..._from_<ragstamp>`` keeps that suffix, so it already
    identifies its own rag file. Any other name is passed through unchanged.
    """
    stem = Path(str(source)).stem
    for prefix, tag in _TAGS:
        if stem.startswith(prefix):
            return f"{tag}-{stem[len(prefix):].lstrip('_')}"
    return stem


def source_tag(source):
    """The filename suffix identifying every source behind an artifact, primary
    first: ``eval-20260729_070055__rag-20260801_181144``."""
    return SEP.join(short_key(s) for s in _sources(source))


def run_dir(source, kind=None, create=True):
    """The output folder for ``source``, or one of its ``kind`` subfolders.

    Creates the folder by default, so callers can hand the result straight to
    ``to_csv`` / ``savefig`` without a mkdir dance.
    """
    d = OUT_ROOT / run_key(source)
    if kind is not None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}, got {kind!r}")
        d = d / kind
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def out_path(source, kind, name, create=True):
    """Full path for one artifact: ``<run>/<kind>/<name>__<source tag><ext>``.

    ``name`` may omit the extension, in which case the one conventional for
    ``kind`` is appended. An explicit extension in ``name`` is preserved (e.g.
    ``"fig.pdf"``), and the tag is still inserted before it.
    """
    suffix = Path(name).suffix or KINDS[kind]
    stem = Path(name).stem
    filename = f"{stem}{SEP}{source_tag(source)}{suffix}"
    return run_dir(source, kind, create=create) / filename


def figure(source, name):
    """Path for a ``.png`` figure of this run."""
    return out_path(source, "figures", name)


def table(source, name):
    """Path for a ``.csv`` table of this run."""
    return out_path(source, "tables", name)


def report(source, name):
    """Path for a ``.txt`` report of this run."""
    return out_path(source, "reports", name)


def linked(source, name):
    """Path for a ``.parquet`` base table of this run."""
    return out_path(source, "linked", name)


def rel(path):
    """``path`` relative to the repo root when it sits under it, else unchanged —
    for printing short, clickable paths in the console reports."""
    p = Path(path)
    try:
        return p.relative_to(OUT_ROOT.parent.parent)
    except ValueError:
        return p


@contextmanager
def capture(source, name):
    """Collect everything printed inside the block, save it to this run's
    ``reports/<name>__<source tag>.txt``, then echo it to the console.

    The analysis scripts print far more than they used to save — cohort
    crosstabs, per-metric distributions, paired Wilcoxon tables, worst/best-query
    listings — but only the health-check section was ever written to disk, so the
    rest existed as scrollback and nothing else. This saves the whole report,
    which is what you actually want to reread when writing a run up.

    Written in UTF-8 (the transcripts carry ``Δ``, em dashes and German query
    text) and truncated on each run, so re-analysing a file replaces its report
    rather than adding another. A crash still saves whatever ran before it.
    """
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            yield
    finally:
        text = buf.getvalue()
        path = report(source, name)
        stamp = datetime.now().isoformat(timespec="seconds")
        srcs = ", ".join(str(s) for s in _sources(source))
        path.write_text(f"# generated: {stamp}\n# source: {srcs}\n\n{text}",
                        encoding="utf-8")
        _echo(text)
        print(f"\nfull report written to {rel(path)}")


def _echo(text):
    """Print ``text``, surviving a console that cannot encode all of it.

    The reports carry ``Δ``, ``≫``, em dashes and German query text; a Windows
    console defaulting to cp1252 raises UnicodeEncodeError on the first of those.
    The FILE is always written in full UTF-8 — only this console copy degrades,
    substituting the handful of characters the terminal cannot represent, because
    losing the whole run's output to a terminal encoding quirk is far worse than
    seeing a few '?' in the scrollback.
    """
    try:
        print(text, end="")
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"), end="")
