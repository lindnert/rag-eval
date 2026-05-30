#!/bin/bash
#SBATCH --job-name=faiss
#SBATCH --comment="Rebuild FAISS index with llama-server embeddings"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.lindner@campus.lmu.de
#SBATCH --chdir=/home/l/lindnerti/rag-eval
#SBATCH --output=/home/l/lindnerti/rag-eval/logs/build_faiss.%j.%N.out
#SBATCH --error=/home/l/lindnerti/rag-eval/logs/build_faiss.%j.%N.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --partition=NvidiaAll
#SBATCH --exclude=adakit
#SBATCH --exclusive


## Submit with: sbatch slurm/build_faiss_index.sh

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${WORKDIR}/logs"

echo "==== Build FAISS job ${SLURM_JOB_ID} on $(hostname) ===="
echo "Start: $(date)"
nvidia-smi || true

# ---------------------------------------------------------------------------
# 1. Activate pre-built venv
# ---------------------------------------------------------------------------
if [ ! -d "${WORKDIR}/.venv" ]; then
  echo "ERROR: venv not found at ${WORKDIR}/.venv — run slurm/build_venv.sh on the login node" >&2
  exit 1
fi
source "${WORKDIR}/.venv/bin/activate"

# ---------------------------------------------------------------------------
# 2. CUDA runtime libs (same setup as run_eval.sh)
# ---------------------------------------------------------------------------
CUDA_HOME="${CUDA_HOME:-$HOME/cuda-13.0}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

LLAMACPP_TAG="${LLAMACPP_TAG:-master}"
LLAMACPP_BIN_DIR="${WORKDIR}/.llamacpp_bin/${LLAMACPP_TAG}"
LLAMACPP_SERVER="${LLAMACPP_BIN_DIR}/llama-server"

if [ ! -x "${LLAMACPP_SERVER}" ]; then
  echo "ERROR: llama-server not found at ${LLAMACPP_SERVER}." >&2
  echo "Build it once on the login node — see slurm/build_llama_server.sh" >&2
  exit 1
fi
echo "llama-server: ${LLAMACPP_SERVER}"

# ---------------------------------------------------------------------------
# 3. Start embedding server (multilingual-e5-base Q4_K_M, mean pooling)
# ---------------------------------------------------------------------------
export HF_HOME="${WORKDIR}/.hf_cache"
mkdir -p "${HF_HOME}"

EMB_HOST="127.0.0.1"
EMB_PORT="${LLAMACPP_EMB_PORT:-8081}"
export LLAMACPP_EMB_BASE_URL="http://${EMB_HOST}:${EMB_PORT}/v1"
export LLAMACPP_EMB_MODEL="${LLAMACPP_EMB_MODEL:-lm-kit/bge-m3-gguf:Q4_K_M}"

pkill -f "llama-server" || true
sleep 2

# bge-m3: encoder-only (no KV cache), trained up to 8192 tokens but we only
# need ~500-token chunks → -c 1024 leaves headroom while keeping VRAM down.
# -ub 1024 keeps physical batch above the largest expected chunk so pooling
# happens in a single forward pass.
stdbuf -oL -eL "${LLAMACPP_SERVER}" \
  -hf lm-kit/bge-m3-gguf:Q4_K_M \
  --host "${EMB_HOST}" --port "${EMB_PORT}" \
  -c 1024 \
  -b 1024 -ub 1024 \
  --n-gpu-layers -1 \
  --parallel 1 \
  --embeddings --pooling cls \
  2>&1 | stdbuf -oL sed 's/^/[EMB] /' &
EMB_PID=$!
trap 'kill ${EMB_PID} 2>/dev/null || true; pkill -f "llama-server" 2>/dev/null || true' EXIT

# Wait for readiness
for i in $(seq 1 300); do
  if curl -sf "http://${EMB_HOST}:${EMB_PORT}/v1/models" >/dev/null; then
    echo "embedding server ready after ${i}s"
    break
  fi
  sleep 1
done

# Warm-up
curl -sf "http://${EMB_HOST}:${EMB_PORT}/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input":"warmup"}' >/dev/null
echo "Embedding server warmed up"
nvidia-smi || true

# ---------------------------------------------------------------------------
# 4. Build the index via preprocessing.utils.build_retriever()
# ---------------------------------------------------------------------------
export FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-${WORKDIR}/richtlinien/faiss_index_bge_m3_cosine}"
echo "Writing index to ${FAISS_INDEX_DIR}"

# build_retriever() short-circuits if the dir exists, so refuse to silently no-op.
if [ -d "${FAISS_INDEX_DIR}" ]; then
  echo "ERROR: ${FAISS_INDEX_DIR} already exists. Remove it or set FAISS_INDEX_DIR to a fresh path." >&2
  exit 1
fi

python -c "from retrieval import build_retriever; build_retriever()"

echo "==== Done at $(date) ===="
ls -lh "${FAISS_INDEX_DIR}"
