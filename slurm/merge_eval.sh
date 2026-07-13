#!/bin/bash
#SBATCH --job-name=rag-eval-merge
#SBATCH --time=00:10:00
#SBATCH --comment="RAG Evaluation merge shards"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=All
#SBATCH --chdir=/home/l/lindnerti/rag-eval
#SBATCH --output=/home/l/lindnerti/rag-eval/logs/eval.merge.%j.out
#SBATCH --error=/home/l/lindnerti/rag-eval/logs/eval.merge.%j.err
set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
# Inherit the language-specific RESULTS_DIR / RAG_LANG from the eval job
# (propagated via --export=ALL) so the merge writes into the same tree the
# shards were written to; fall back to the English default for a manual run.
export RAG_LANG="${RAG_LANG:-en}"
export RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/results/${RAG_LANG}}"
echo "Merging RAG_LANG=${RAG_LANG} from RESULTS_DIR=${RESULTS_DIR}"
source "${WORKDIR}/.venv/bin/activate"

ARRAY_ID="${MERGE_JOB_ID:?MERGE_JOB_ID not set}"

echo "==== Array job ${ARRAY_ID} — combined stdout ===="
for f in "${WORKDIR}/logs/eval.${ARRAY_ID}_"*.out; do
  [ -e "$f" ] || continue
  echo
  echo "---- ${f} ----"
  cat "$f"
done

{
  echo "==== Array job ${ARRAY_ID} — combined stderr ===="
  for f in "${WORKDIR}/logs/eval.${ARRAY_ID}_"*.err; do
    [ -e "$f" ] || continue
    echo
    echo "---- ${f} ----"
    cat "$f"
  done
} | tee /dev/stderr

echo "==== Merge step ===="
python -m common.merge_shards --kind=eval

# Merge succeeded (set -e would have aborted otherwise) — drop per-task logs.
echo "==== Cleaning up per-task logs ===="
rm -f "${WORKDIR}/logs/eval.${ARRAY_ID}_"*.out \
      "${WORKDIR}/logs/eval.${ARRAY_ID}_"*.err
