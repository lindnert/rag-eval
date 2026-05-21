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
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${WORKDIR}"

rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Adjustments previously done in run_eval.sh / run_rag.sh
sed -i '/pywin32/d' requirements.txt
sed -i 's/==/>=/g' requirements.txt

pip install -r requirements.txt

echo "Done. Venv at: ${WORKDIR}/.venv"
