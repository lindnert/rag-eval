#!/bin/bash
#SBATCH --job-name=rag-venv
#SBATCH --comment="RAG venv"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --partition=All
## Re-run when requirements.txt changes.

set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${WORKDIR}"

rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Adjustments previously done in run_eval.sh / run_rag.sh
sed -i '/pywin32/d' requirements.txt
sed -i 's/==/>=/g' requirements.txt

pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Pre-warm HF cache for Ragas FaithfulnesswithHHEM (Vectara HHEM-2.1-Open).
# All run_*.sh scripts export HF_HOME=${WORKDIR}/.hf_cache, so populating it
# here means the eval array shards hit a warm cache instead of racing on
# the same download.
# ---------------------------------------------------------------------------
export HF_HOME="${WORKDIR}/.hf_cache"
mkdir -p "${HF_HOME}"
echo "Pre-warming HF cache at ${HF_HOME} with vectara/hallucination_evaluation_model..."
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="vectara/hallucination_evaluation_model")
print(f"HHEM snapshot at: {path}")
PY

echo "Done. Venv at: ${WORKDIR}/.venv"
