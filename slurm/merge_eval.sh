#!/bin/bash
#SBATCH --job-name=rag-eval-merge
#SBATCH --time=00:10:00
#SBATCH --comment="RAG Evaluation merge shards"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=All
#SBATCH --chdir=/home/l/lindnerti/rag-eval
#SBATCH --output=/home/l/lindnerti/rag-eval/logs/merge.%j.out
set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
export RESULTS_DIR="${WORKDIR}/results"
source "${WORKDIR}/.venv/bin/activate"

python -m evaluation.merge_shards
