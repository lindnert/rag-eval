#!/bin/bash
#SBATCH --job-name=rag
#SBATCH --comment="RAG pipeline (llama.cpp)"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.lindner@campus.lmu.de
#SBATCH --chdir=/home/l/lindnerti/rag-eval
#SBATCH --output=/home/l/lindnerti/rag-eval/logs/rag.%A_%a.%N.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --partition=NvidiaAll
#SBATCH --exclude=adakit
#SBATCH --exclusive
#SBATCH --array=0-2

## Submit with: sbatch slurm/run_rag.sh
## Override array size (= number of shards / nodes):
##   sbatch --array=0-5 slurm/run_rag.sh    # 6 shards
##   sbatch --array=0   slurm/run_rag.sh    # single shard (no sharding)

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
ERR_LOG="${WORKDIR}/logs/rag.${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.$(hostname).err"
exec 2> >(tee -a "${ERR_LOG}" >&2)

export RESULTS_DIR="${WORKDIR}/results"
mkdir -p "${RESULTS_DIR}" "${WORKDIR}/logs"

# First-firing task schedules the merge job; --dependency=afterok holds it
# until all shards in the array finish successfully.
if [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ]; then
  sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
         --export=ALL,MERGE_JOB_ID=${SLURM_ARRAY_JOB_ID} \
         "${WORKDIR}/slurm/merge_rag.sh"
fi

PYTHON_BIN="$(command -v python3.12 || command -v python3)"

echo "==== RAG job ${SLURM_JOB_ID} on $(hostname) ===="
echo "Start: $(date)"
nvidia-smi || true
cat /proc/driver/nvidia/version || true

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

GEN_REPO="${LLAMACPP_GEN_REPO:-unsloth/Qwen3.5-4B-GGUF}"
GEN_FILE="${LLAMACPP_GEN_FILE:-Qwen3.5-4B-UD-Q4_K_XL.gguf}"

# Family-distinct from the gemma judge in evaluation/, avoids self-reference bias.
# Use the Python API directly — the `hf` CLI leaks a click.exceptions.Exit(0)
# on success in some typer/click version combos, which trips `set -e`.
echo "Downloading gen GGUF (cached in ${LLAMA_MODELS_DIR}/gen)..."
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${GEN_REPO}', filename='${GEN_FILE}', local_dir='${LLAMA_MODELS_DIR}/gen')"
GEN_PATH="${LLAMA_MODELS_DIR}/gen/${GEN_FILE}"
echo "Gen GGUF: ${GEN_PATH}"

# Tell rag/llm_config.py which model identifier to send in /v1/chat/completions.
# (llama-server reports this string under /v1/models for the loaded model.)
export LLAMACPP_RAG_MODEL="${LLAMACPP_RAG_MODEL:-unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL}"

# Embedding model is pulled by llama-server via -hf (cached in HF_HOME);
# must match what built the FAISS index in slurm/build_faiss_index.sh.

# ---------------------------------------------------------------------------
# 3. Locate native llama-server (built once via slurm/build_llama_server.sh)
# ---------------------------------------------------------------------------
LLAMACPP_TAG="${LLAMACPP_TAG:-master}"
LLAMACPP_BIN_DIR="${WORKDIR}/.llamacpp_bin/${LLAMACPP_TAG}"
LLAMACPP_SERVER="${LLAMACPP_BIN_DIR}/llama-server"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda-13.0}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [ ! -x "${LLAMACPP_SERVER}" ]; then
  echo "ERROR: llama-server not found at ${LLAMACPP_SERVER}." >&2
  echo "Build it once on the login node — see slurm/build_llama_server.sh" >&2
  exit 1
fi
echo "llama-server: ${LLAMACPP_SERVER}"

echo "---- nvidia-ml diagnostic ----"
ls -l /lib/x86_64-linux-gnu/libnvidia-ml.so* 2>/dev/null || true
ls -l ${CUDA_HOME}/lib64/libnvidia-ml* 2>/dev/null || true
# `|| true` so an empty grep (or SIGPIPE from head) doesn't trip pipefail.
{ LD_DEBUG=libs "${LLAMACPP_SERVER}" --version 2>&1 | grep -i nvidia-ml | head -20; } || true
echo "------------------------------"

# ---------------------------------------------------------------------------
# 4. Start gen (8080) + emb (8081) servers
# ---------------------------------------------------------------------------
export LLAMACPP_GEN_HOST="127.0.0.1"
export LLAMACPP_GEN_PORT=8080
export LLAMACPP_EMB_PORT=8081
export LLAMACPP_GEN_BASE_URL="http://${LLAMACPP_GEN_HOST}:${LLAMACPP_GEN_PORT}/v1"
export LLAMACPP_EMB_BASE_URL="http://${LLAMACPP_GEN_HOST}:${LLAMACPP_EMB_PORT}/v1"

# Match the embedding model used at index build time (bge-m3, CLS pooling, no prefixes).
export LLAMACPP_EMB_MODEL="${LLAMACPP_EMB_MODEL:-lm-kit/bge-m3-gguf:Q4_K_M}"

CONTEXT_LENGTH="${LLAMACPP_CONTEXT_LENGTH:-16384}"
GEN_PARALLEL="${LLAMACPP_GEN_PARALLEL:-6}"
echo "GEN_PARALLEL=${GEN_PARALLEL}, GEN_CTX=${CONTEXT_LENGTH}"

pkill -f "llama-server" || true
sleep 2

# Generation server.
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

# Embedding server — same config as slurm/build_faiss_index.sh so query and
# passage embeddings stay in the same space.
stdbuf -oL -eL "${LLAMACPP_SERVER}" \
  -hf "${LLAMACPP_EMB_MODEL}" \
  --host "${LLAMACPP_GEN_HOST}" --port "${LLAMACPP_EMB_PORT}" \
  -c 1024 \
  -b 1024 -ub 1024 \
  --n-gpu-layers -1 \
  --parallel 1 \
  --embeddings --pooling cls \
  2>&1 | stdbuf -oL sed 's/^/[EMB] /' &
EMB_PID=$!

trap 'kill ${GEN_PID} ${EMB_PID} 2>/dev/null || true; pkill -f "llama-server" 2>/dev/null || true' EXIT

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
# 5. Warm-up
# ---------------------------------------------------------------------------
echo "Warming up models..."
curl -sf "http://${LLAMACPP_GEN_HOST}:${LLAMACPP_GEN_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Sag Hallo in einem Wort."}],"max_tokens":8,"temperature":0}' >/dev/null
curl -sf "http://${LLAMACPP_GEN_HOST}:${LLAMACPP_EMB_PORT}/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input":"warmup"}' >/dev/null
echo "Both models warmed up"
nvidia-smi || true

# ---------------------------------------------------------------------------
# 6. Run RAG pipeline
# ---------------------------------------------------------------------------
echo "==== RAG pipeline ===="

export RAG_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export RAG_SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
export RAG_SHARD_TAG="${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Shard ${RAG_SHARD_INDEX}/${RAG_SHARD_COUNT} (tag=${RAG_SHARD_TAG})"

python -m rag.rag_pipeline

echo "==== Done at $(date) ===="
echo "Results in: ${RESULTS_DIR}"
ls -lh "${RESULTS_DIR}"
