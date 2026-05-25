import argparse
import glob
import json
import os
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(results_dir, f"{cfg['output_prefix']}_{ts}.json")
    latest = os.path.join(results_dir, f"{cfg['output_prefix']}_latest.json")
    for path in (out, latest):
        dump_json(merged, path)

    print(f"Merged {len(shards)} shards → {len(merged)} samples → {out}")

    shutil.rmtree(shard_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=tuple(CONFIGS))
    args = parser.parse_args()
    merge(args.kind)
