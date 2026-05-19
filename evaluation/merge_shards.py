import glob, json, os
import shutil
from datetime import datetime

results_dir = os.environ.get("RESULTS_DIR", "results")

job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("MERGE_JOB_ID")
shard_dir = os.path.join(results_dir, "_shards", job_id) if job_id else \
            max(glob.glob(os.path.join(results_dir, "_shards", "*")), key=os.path.getmtime)

shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.json")))
shards = [p for p in shards if not p.endswith(".partial.json")]

merged = []
for p in shards:
    with open(p, encoding="utf-8") as f:
        merged.extend(json.load(f))
merged.sort(key=lambda r: r["id"])

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = os.path.join(results_dir, f"evaluated_results_{ts}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
with open(os.path.join(results_dir, "evaluated_results_latest.json"), "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
print(f"Merged {len(shards)} shards → {len(merged)} samples → {out}")

shutil.rmtree(shard_dir)
