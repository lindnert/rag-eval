"""Finding the newest run file in the flat ``results/`` tree.

Each pipeline writes exactly ONE timestamped file per run — there is no
``*_latest.json`` copy. A copy could not record provenance: an eval noting "I
read rag_results_latest.json" said nothing, since the next RAG run overwrote
that name with different data. "Latest" is resolved from the stamps instead,
and the eval output carries the stamp of the RAG file it consumed:

    results/rag_results_20260723_141500.json
    results/evaluated_results_20260723_190000_from_20260723_141500.json

The shell submitter in slurm/run_eval.sh resolves the same "newest" file with a
glob + ``sort | tail -1`` (filenames sort by timestamp) so it needs no venv.
"""

import glob
import os

RAG_PREFIX = "rag_results"
EVAL_PREFIX = "evaluated_results"


def latest_results(prefix, directory=None):
    """Newest ``<prefix>_<timestamp>.json`` under ``directory``.

    The ``_[0-9]*`` glob skips names without a timestamp (a leftover
    ``*_latest.json`` from before this scheme): they carry no run identity, and
    lexically "latest" would otherwise sort above every digit and always win."""
    directory = directory or os.environ.get("RESULTS_DIR", "results")
    matches = sorted(glob.glob(os.path.join(directory, f"{prefix}_[0-9]*.json")))
    if not matches:
        raise FileNotFoundError(
            f"no {prefix}_<timestamp>.json in {directory} — "
            f"run the pipeline first, or pass an explicit path"
        )
    return matches[-1]
