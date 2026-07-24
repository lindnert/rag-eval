import argparse
import glob
import json
import os
import re
import shutil
from datetime import datetime

from common.json_io import dump as dump_json

# Per-kind merge configuration. Both flows write shard files named
# shard_<jobid>_<taskid>.json under a kind-specific subdir of RESULTS_DIR.
CONFIGS = {
    "eval": {
        "shard_subdir": "_shards",
        "output_prefix": "evaluated_results",
    },
    "rag": {
        "shard_subdir": "_shards_rag",
        "output_prefix": "rag_results",
    },
}


def _stamp(path):
    """How a source results file is identified inside a derived filename.

    Normally its YYYYMMDD_HHMMSS run stamp. A file named by hand (pointed at via
    RAG_RESULTS_FILE) may not carry one, so fall back to its stem minus the
    ``rag_results_`` prefix — sanitised, since it becomes part of a filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_(\d{8}_\d{6})", stem)
    if m:
        return m.group(1)
    label = re.sub(r"^rag_results_?", "", stem)
    label = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-")
    return label or None


def _sort_key(kind):
    # RAG output has three rows per query (one per variant) and uses
    # `pipeline_id` (global 0..N-1 enumerate index assigned before sharding)
    # as the stable cross-shard ordering key. Sample-level `id` from the
    # dataset is preserved separately on each row but isn't used for sorting.
    # Eval output has one row per id, so plain id-sort is enough.
    if kind == "rag":
        from rag.utils import VARIANTS

        order = {v: i for i, v in enumerate(VARIANTS)}
        return lambda r: (r["pipeline_id"], order.get(r.get("variant", ""), 99))
    return lambda r: r["id"]


def merge(kind, results_dir=None):
    cfg = CONFIGS[kind]
    results_dir = results_dir or os.environ.get("RESULTS_DIR", "results")

    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("MERGE_JOB_ID")
    shard_root = os.path.join(results_dir, cfg["shard_subdir"])
    shard_dir = (
        os.path.join(shard_root, job_id)
        if job_id
        else max(glob.glob(os.path.join(shard_root, "*")), key=os.path.getmtime)
    )

    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.json")))
    shards = [p for p in shards if not p.endswith(".partial.json")]

    merged = []
    for p in shards:
        with open(p, encoding="utf-8") as f:
            merged.extend(json.load(f))
    merged.sort(key=_sort_key(kind))

    # One timestamped file per run, no *_latest.json copy — see common/results_io.
    # An eval run additionally records WHICH rag_results file it consumed by
    # appending that file's stamp: evaluated_results_<evalts>_from_<ragts>.json.
    # RAG_RESULTS_FILE is resolved once by the login-node submitter in
    # slurm/run_eval.sh and exported down through the shards to this merge job.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{cfg['output_prefix']}_{ts}"
    if kind == "eval":
        src = os.environ.get("RAG_RESULTS_FILE")
        src_stamp = _stamp(src) if src else None
        if src_stamp:
            name += f"_from_{src_stamp}"
        else:
            print("WARNING: RAG_RESULTS_FILE unset — output records no source file. "
                  "Submit via ./slurm/run_eval.sh (or export it) to keep provenance.")
    out = os.path.join(results_dir, f"{name}.json")
    dump_json(merged, out)

    print(f"Merged {len(shards)} shards → {len(merged)} samples → {out}")

    shutil.rmtree(shard_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=tuple(CONFIGS))
    args = parser.parse_args()
    merge(args.kind)
