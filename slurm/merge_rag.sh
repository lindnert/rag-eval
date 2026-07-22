#!/bin/bash
#SBATCH --job-name=rag-merge
#SBATCH --time=00:10:00
#SBATCH --comment="RAG pipeline merge shards"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=All
#SBATCH --chdir=/home/l/lindnerti/rag-eval
#SBATCH --output=/home/l/lindnerti/rag-eval/logs/rag.merge.%j.out
#SBATCH --error=/home/l/lindnerti/rag-eval/logs/rag.merge.%j.err
set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
# Inherit RESULTS_DIR from the run job (propagated via --export=ALL) so the
# merge writes into the same tree the shards were written to; fall back to the
# flat results/ tree for a manual run.
export RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/results}"
echo "Merging RAG shards from RESULTS_DIR=${RESULTS_DIR}"
source "${WORKDIR}/.venv/bin/activate"

ARRAY_ID="${MERGE_JOB_ID:?MERGE_JOB_ID not set}"

echo "==== Array job ${ARRAY_ID} — combined stdout ===="
for f in "${WORKDIR}/logs/rag.${ARRAY_ID}_"*.out; do
  [ -e "$f" ] || continue
  echo
  echo "---- ${f} ----"
  cat "$f"
done

{
  echo "==== Array job ${ARRAY_ID} — combined stderr ===="
  for f in "${WORKDIR}/logs/rag.${ARRAY_ID}_"*.err; do
    [ -e "$f" ] || continue
    echo
    echo "---- ${f} ----"
    cat "$f"
  done
} | tee /dev/stderr

echo "==== Merge step ===="
python -m common.merge_shards --kind=rag

# Merge succeeded (set -e would have aborted otherwise) — drop per-task logs.
echo "==== Cleaning up per-task logs ===="
rm -f "${WORKDIR}/logs/rag.${ARRAY_ID}_"*.out \
      "${WORKDIR}/logs/rag.${ARRAY_ID}_"*.err
