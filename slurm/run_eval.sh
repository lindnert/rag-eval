#!/bin/bash
#SBATCH --job-name=rag-eval
#SBATCH --comment="RAG Evaluation pipeline"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.lindner@campus.lmu.de
#SBATCH --chdir=/home/l/lindnerti/rag-eval
#SBATCH --output=/home/l/lindnerti/rag-eval/logs/eval.%A_%a.%N.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --partition=NvidiaAll
#SBATCH --array=0-14

## This script is dual-mode (see the SLURM_JOB_ID branch below).
## Run it directly on the login node — it submits the array job(s) for you and
## reads each language's RAG results from results/<lang>/:
##   ./slurm/run_eval.sh          # both languages (default)
##   ./slurm/run_eval.sh de       # one language (en | de)
## Direct sbatch still works (defaults to RAG_LANG=en):
##   RAG_LANG=de sbatch slurm/run_eval.sh
##   sbatch --dependency=afterok:<rag_jobid> --export=ALL,RAG_LANG=de slurm/run_eval.sh
## Optionally override input: RAG_RESULTS_FILE=/path/to/rag_results_YYYYMMDD.json ./slurm/run_eval.sh de
## Override shards per language: ARRAY_MAX=9 ./slurm/run_eval.sh   # 10 shards

set -euo pipefail

# ---------------------------------------------------------------------------
# Dual-mode entry point (mirrors slurm/run_rag.sh).
#   - On the LOGIN NODE (no SLURM_JOB_ID) this acts as a *submitter*: it
#     sbatch-es one array job per language and exits. Languages come from the
#     args (default: both).
#   - Under SLURM (SLURM_JOB_ID set) it falls through to the worker body and
#     evaluates the single $RAG_LANG it was submitted with, reading that
#     language's rag_results from results/<lang>/.
# Array size respects a 30-job cap: 14 shards/lang when both queue together
# (14 + 14 + 2 dependency-held merge jobs = 30), 15 for a single language.
# Override with ARRAY_MAX=<n> (array is 0..n, i.e. n+1 shards).
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
  LANGS=("$@")
  if [ ${#LANGS[@]} -eq 0 ]; then
    LANGS=(en de)
  fi
  for lang in "${LANGS[@]}"; do
    if [ "${lang}" != "en" ] && [ "${lang}" != "de" ]; then
      echo "ERROR: unknown language '${lang}' (expected 'en' or 'de')" >&2
      exit 1
    fi
  done
  if [ -z "${ARRAY_MAX:-}" ]; then
    if [ ${#LANGS[@]} -ge 2 ]; then ARRAY_MAX=13; else ARRAY_MAX=14; fi
  fi
  # Resolve absolute paths so submission is independent of the cwd the user
  # launched from. This script lives in <repo>/slurm/, so REPO_ROOT is its
  # parent dir. cd there before sbatch so the array job's SLURM_SUBMIT_DIR
  # (which the worker body uses as WORKDIR) is pinned to the repo root.
  SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "${REPO_ROOT}"
  for lang in "${LANGS[@]}"; do
    echo "Submitting eval RAG_LANG=${lang}  --array=0-${ARRAY_MAX} ($((ARRAY_MAX + 1)) shards) from ${REPO_ROOT}"
    sbatch --array="0-${ARRAY_MAX}" --export=ALL,RAG_LANG="${lang}" "${SELF}"
  done
  exit 0
fi

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
ERR_LOG="${WORKDIR}/logs/eval.${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.$(hostname).err"
exec 2> >(tee -a "${ERR_LOG}" >&2)

# Language of the run: selects which results/<lang>/ tree we read the RAG
# results from and write evaluated_results into. The evaluation Python itself
# is language-agnostic (abstention travels in the `rejected` flag and
# `reference_answer` already baked into the RAG results).
export RAG_LANG="${RAG_LANG:-en}"
export RESULTS_DIR="${WORKDIR}/results/${RAG_LANG}"
mkdir -p "${RESULTS_DIR}" "${WORKDIR}/logs"
echo "RAG_LANG=${RAG_LANG}  RESULTS_DIR=${RESULTS_DIR}"

# log the merge job as a dependent job that runs after all shards are done; if this is the last shard (task_id=0), it will trigger the merge job.
if [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ]; then
  sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
         --export=ALL,MERGE_JOB_ID=${SLURM_ARRAY_JOB_ID} \
         "${WORKDIR}/slurm/merge_eval.sh"
fi

PYTHON_BIN="$(command -v python3.12 || command -v python3)"

echo "==== Eval job ${SLURM_JOB_ID} on $(hostname) ===="
echo "Start: $(date)"
nvidia-smi || true

# ---------------------------------------------------------------------------
# 1. Activate pre-built venv (see slurm/build_venv.sh)
# ---------------------------------------------------------------------------
if [ ! -d "${WORKDIR}/.venv" ]; then
  echo "ERROR: venv not found at ${WORKDIR}/.venv — run slurm/build_venv.sh on the login node" >&2
  exit 1
fi
source "${WORKDIR}/.venv/bin/activate"

# ---------------------------------------------------------------------------
# 2. Download GGUFs from Hugging Face
# ---------------------------------------------------------------------------
export HF_HOME="${WORKDIR}/.hf_cache"
export LLAMA_MODELS_DIR="${WORKDIR}/.llamacpp_models"
mkdir -p "${HF_HOME}" "${LLAMA_MODELS_DIR}"

# Run Ragas FaithfulnesswithHHEM on CPU — the GPU is fully occupied by the
# llama-server gen/emb instances, and HHEM (184M params) is fast enough on CPU.
export HHEM_DEVICE="${HHEM_DEVICE:-cpu}"

GEN_REPO="${LLAMACPP_GEN_REPO:-unsloth/gemma-4-E2B-it-GGUF}"
GEN_FILE="${LLAMACPP_GEN_FILE:-gemma-4-E2B-it-UD-Q4_K_XL.gguf}"
EMB_REPO="${LLAMACPP_EMB_REPO:-Qwen/Qwen3-Embedding-0.6B-GGUF}"
EMB_FILE="${LLAMACPP_EMB_FILE:-Qwen3-Embedding-0.6B-Q8_0.gguf}"

echo "Downloading GGUFs (cached in ${HF_HOME})..."
# Use the Python API directly — the `hf` CLI leaks a click.exceptions.Exit(0)
# on success in some typer/click version combos, which trips `set -e`.
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${GEN_REPO}', filename='${GEN_FILE}', local_dir='${LLAMA_MODELS_DIR}/gen')"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${EMB_REPO}', filename='${EMB_FILE}', local_dir='${LLAMA_MODELS_DIR}/emb')"
GEN_PATH="${LLAMA_MODELS_DIR}/gen/${GEN_FILE}"
EMB_PATH="${LLAMA_MODELS_DIR}/emb/${EMB_FILE}"
echo "Gen GGUF: ${GEN_PATH}"
echo "Emb GGUF: ${EMB_PATH}"

# ---------------------------------------------------------------------------
# 3. Build native llama-server (once, cached) — needed because llama_cpp.server
#    (Python wrapper) doesn't expose --parallel for gen and hardcodes
#    n_seq_max=256 for embeddings → KV-cache OOM. Native binary fixes both.
# ---------------------------------------------------------------------------
LLAMACPP_TAG="${LLAMACPP_TAG:-master}"
LLAMACPP_BIN_DIR="${WORKDIR}/.llamacpp_bin/${LLAMACPP_TAG}"
LLAMACPP_SERVER="${LLAMACPP_BIN_DIR}/llama-server"

# CUDA 13 toolkit installed to user home (see README/setup notes).
# The compute node's driver supports CUDA 13; we just need the runtime libs
# on LD_LIBRARY_PATH so the dynamically-linked llama-server can find them.
CUDA_HOME="${CUDA_HOME:-$HOME/cuda-13.0}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [ ! -x "${LLAMACPP_SERVER}" ]; then
  echo "ERROR: llama-server not found at ${LLAMACPP_SERVER}." >&2
  echo "Build it once on the login node — see slurm/build_llama_server.sh" >&2
  exit 1
fi
echo "llama-server: ${LLAMACPP_SERVER}"
echo "CUDA_HOME=${CUDA_HOME}, LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# ---------------------------------------------------------------------------
# 4. Start gen (8080) + emb (8081) servers, each configured independently.
# ---------------------------------------------------------------------------
export LLAMACPP_GEN_HOST="127.0.0.1"
export LLAMACPP_GEN_PORT=8080
export LLAMACPP_EMB_PORT=8081
export LLAMACPP_GEN_BASE_URL="http://${LLAMACPP_GEN_HOST}:${LLAMACPP_GEN_PORT}/v1"
export LLAMACPP_EMB_BASE_URL="http://${LLAMACPP_GEN_HOST}:${LLAMACPP_EMB_PORT}/v1"

CONTEXT_LENGTH="${LLAMACPP_CONTEXT_LENGTH:-32768}"
GEN_PARALLEL="${LLAMACPP_GEN_PARALLEL:-6}"
EMB_CONTEXT_LENGTH="${LLAMACPP_EMB_CONTEXT_LENGTH:-2048}"
echo "GEN_PARALLEL=${GEN_PARALLEL}, GEN_CTX=${CONTEXT_LENGTH}, EMB_CTX=${EMB_CONTEXT_LENGTH}"

pkill -f "llama-server" || true
pkill -f "llama_cpp.server" || true
sleep 2

# Generation server: real concurrent slots via --parallel + --cont-batching.
# -fa = flash-attn; -ctk/-ctv q8_0 = q8_0 KV cache (mirrors OLLAMA_KV_CACHE_TYPE).
stdbuf -oL -eL "${LLAMACPP_SERVER}" \
  --model "${GEN_PATH}" \
  --host "${LLAMACPP_GEN_HOST}" --port "${LLAMACPP_GEN_PORT}" \
  --ctx-size "${CONTEXT_LENGTH}" \
  --n-gpu-layers -1 \
  --parallel "${GEN_PARALLEL}" \
  --cont-batching \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  2>&1 | stdbuf -oL sed 's/^/[GEN] /' &
GEN_PID=$!

# Embedding server: --parallel 1 keeps n_seq_max=1, so n_ctx is honored exactly
# (the bug we hit in llama_cpp.server doesn't exist in the native binary).
stdbuf -oL -eL "${LLAMACPP_SERVER}" \
  --model "${EMB_PATH}" \
  --host "${LLAMACPP_GEN_HOST}" --port "${LLAMACPP_EMB_PORT}" \
  --ctx-size "${EMB_CONTEXT_LENGTH}" \
  --n-gpu-layers -1 \
  --parallel 1 \
  --embeddings \
  2>&1 | stdbuf -oL sed 's/^/[EMB] /' &
EMB_PID=$!

trap 'kill ${GEN_PID} ${EMB_PID} 2>/dev/null || true; pkill -f "llama-server" 2>/dev/null || true' EXIT

# Wait for both servers. Native llama-server exposes /v1/models once loaded.
wait_ready() {
  local url="$1"
  local name="$2"
  for i in $(seq 1 600); do
    if curl -sf "${url}/v1/models" >/dev/null; then
      echo "${name} ready after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: ${name} did not become ready" >&2
  return 1
}

wait_ready "http://${LLAMACPP_GEN_HOST}:${LLAMACPP_GEN_PORT}" "llama-server (gen)"
wait_ready "http://${LLAMACPP_GEN_HOST}:${LLAMACPP_EMB_PORT}" "llama-server (emb)"

# ---------------------------------------------------------------------------
# 5. Warm-up requests
# ---------------------------------------------------------------------------
echo "Warming up models with dummy requests..."
curl -sf "http://${LLAMACPP_GEN_HOST}:${LLAMACPP_GEN_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Sag Hallo in einem Wort."}],"max_tokens":8,"temperature":0}' >/dev/null
curl -sf "http://${LLAMACPP_GEN_HOST}:${LLAMACPP_EMB_PORT}/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input":"warmup"}' >/dev/null
echo "Both models warmed up"

echo "GPU status immediately after warm-up (models should be resident on GPU):"
nvidia-smi || true

# ---------------------------------------------------------------------------
# 5. Run evaluation pipeline
# ---------------------------------------------------------------------------
echo "==== Evaluation pipeline ===="
echo "Reading RAG results from: ${RAG_RESULTS_FILE:-${RESULTS_DIR}/rag_results_latest.json}"

# Background sampler: log compact GPU utilization + memory every 30 s while the
# eval runs. Helps confirm whether llama.cpp actually offloaded layers to CUDA.
GPU_LOG="${WORKDIR}/logs/gpu_sample.${SLURM_JOB_ID:-local}.log"
(
  while true; do
    {
      echo "---- $(date '+%F %T') ----"
      nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total \
                 --format=csv,noheader,nounits
      nvidia-smi --query-compute-apps=pid,process_name,used_memory \
                 --format=csv,noheader,nounits
    } >> "${GPU_LOG}" 2>&1
    sleep 30
  done
) &
GPU_SAMPLER_PID=$!
trap 'kill ${GPU_SAMPLER_PID} 2>/dev/null || true; kill ${GEN_PID} ${EMB_PID} 2>/dev/null || true; pkill -f "llama-server" 2>/dev/null || true' EXIT
echo "GPU sampler PID=${GPU_SAMPLER_PID}, logging to ${GPU_LOG}"

export EVAL_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export EVAL_SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
export EVAL_SHARD_TAG="${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"

python -m evaluation.eval_pipeline

kill ${GPU_SAMPLER_PID} 2>/dev/null || true

echo "GPU status after evaluation:"
nvidia-smi || true
echo "---- GPU sample log (${GPU_LOG}) ----"
cat "${GPU_LOG}" || true

echo "==== Done at $(date) ===="
echo "Results in: ${RESULTS_DIR}"
ls -lh "${RESULTS_DIR}"
