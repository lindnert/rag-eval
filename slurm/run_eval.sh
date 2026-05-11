#!/bin/bash
#SBATCH --job-name=rag-eval
#SBATCH --comment="RAG Evaluation pipeline"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.lindner@campus.lmu.de
#SBATCH --chdir=/home/l/lindnerti/rag-eval
#SBATCH --output=/home/l/lindnerti/rag-eval/logs/eval.%j.%N.out
#SBATCH --error=/home/l/lindnerti/rag-eval/logs/eval.%j.%N.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --partition=NvidiaAll

## Submit with: sbatch slurm/run_eval.sh
## Optionally: sbatch --dependency=afterok:<rag_jobid> slurm/run_eval.sh
## Optionally override input: RAG_RESULTS_FILE=/path/to/rag_results_YYYYMMDD.json sbatch --export=ALL slurm/run_eval.sh

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
export RESULTS_DIR="${WORKDIR}/results"
mkdir -p "${RESULTS_DIR}" "${WORKDIR}/logs"

PYTHON_BIN="$(command -v python3.12 || command -v python3)"

echo "==== Eval job ${SLURM_JOB_ID} on $(hostname) ===="
echo "Start: $(date)"
nvidia-smi || true

# ---------------------------------------------------------------------------
# 1. Python venv + dependencies
# ---------------------------------------------------------------------------
if [ ! -d "${WORKDIR}/.venv" ]; then
  "${PYTHON_BIN}" -m venv "${WORKDIR}/.venv"
fi
source "${WORKDIR}/.venv/bin/activate"

sed -i '/pywin32/d' requirements.txt
sed -i 's/==/>=/g' requirements.txt

pip install --upgrade pip
pip install -r requirements.txt

# llama-cpp-python with prebuilt CUDA wheels (abetlen wheel index).
# Node driver is CUDA 13; cu125 wheel runs on it (driver is backward-compatible
# with older CUDA runtimes — cu13 wheels are not yet published).
LLAMA_CPP_PY_VERSION="${LLAMA_CPP_PY_VERSION:-0.3.16}"
pip install --upgrade --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125 \
  "llama-cpp-python[server]>=${LLAMA_CPP_PY_VERSION}"
pip install --upgrade huggingface_hub

# ---------------------------------------------------------------------------
# 2. Download GGUFs from Hugging Face
# ---------------------------------------------------------------------------
export HF_HOME="${WORKDIR}/.hf_cache"
export LLAMA_MODELS_DIR="${WORKDIR}/.llamacpp_models"
mkdir -p "${HF_HOME}" "${LLAMA_MODELS_DIR}"

GEN_REPO="${LLAMACPP_GEN_REPO:-unsloth/gemma-4-E2B-it-GGUF}"
GEN_FILE="${LLAMACPP_GEN_FILE:-gemma-4-E2B-it-UD-Q4_K_XL.gguf}"
EMB_REPO="${LLAMACPP_EMB_REPO:-Qwen/Qwen3-Embedding-0.6B-GGUF}"
EMB_FILE="${LLAMACPP_EMB_FILE:-Qwen3-Embedding-0.6B-Q8_0.gguf}"

echo "Downloading GGUFs (cached in ${HF_HOME})..."
hf download "${GEN_REPO}" "${GEN_FILE}" --local-dir "${LLAMA_MODELS_DIR}/gen"
hf download "${EMB_REPO}" "${EMB_FILE}" --local-dir "${LLAMA_MODELS_DIR}/emb"
GEN_PATH="${LLAMA_MODELS_DIR}/gen/${GEN_FILE}"
EMB_PATH="${LLAMA_MODELS_DIR}/emb/${EMB_FILE}"
echo "Gen GGUF: ${GEN_PATH}"
echo "Emb GGUF: ${EMB_PATH}"

# ---------------------------------------------------------------------------
# 3. Start llama-cpp-python servers: generation (8080) + embeddings (8081)
# ---------------------------------------------------------------------------
export LLAMACPP_GEN_HOST="127.0.0.1"
export LLAMACPP_GEN_PORT=8080
export LLAMACPP_EMB_PORT=8081
export LLAMACPP_GEN_BASE_URL="http://${LLAMACPP_GEN_HOST}:${LLAMACPP_GEN_PORT}/v1"
export LLAMACPP_EMB_BASE_URL="http://${LLAMACPP_GEN_HOST}:${LLAMACPP_EMB_PORT}/v1"

# Mirrors the previous Ollama hyperparameter settings (see eval_config_llamacpp.py).
CONTEXT_LENGTH="${LLAMACPP_CONTEXT_LENGTH:-16384}"
GEN_PARALLEL="${LLAMACPP_GEN_PARALLEL:-1}"
echo "LLAMACPP_GEN_PARALLEL is set to ${GEN_PARALLEL}"

pkill -f "llama_cpp.server" || true
sleep 2

# Generation server: flash-attn + q8_0 KV cache to mirror OLLAMA_FLASH_ATTENTION / OLLAMA_KV_CACHE_TYPE.
# llama-cpp-python type_k/type_v: 8 = GGML_TYPE_Q8_0.
stdbuf -oL -eL python -m llama_cpp.server \
  --model "${GEN_PATH}" \
  --host "${LLAMACPP_GEN_HOST}" --port "${LLAMACPP_GEN_PORT}" \
  --n_ctx "${CONTEXT_LENGTH}" \
  --n_gpu_layers -1 \
  --flash_attn true \
  --type_k 8 --type_v 8 \
  --n_threads_batch "${GEN_PARALLEL}" \
  2>&1 | stdbuf -oL sed 's/^/[GEN] /' &
GEN_PID=$!

# Embedding server: --embedding enables /v1/embeddings
stdbuf -oL -eL python -m llama_cpp.server \
  --model "${EMB_PATH}" \
  --host "${LLAMACPP_GEN_HOST}" --port "${LLAMACPP_EMB_PORT}" \
  --embedding true \
  --n_gpu_layers -1 \
  2>&1 | stdbuf -oL sed 's/^/[EMB] /' &
EMB_PID=$!

trap 'kill ${GEN_PID} ${EMB_PID} 2>/dev/null || true; pkill -f "llama_cpp.server" 2>/dev/null || true' EXIT

# Wait for both servers to load. llama-cpp-python exposes /v1/models once the
# model is loaded; /health is not guaranteed across versions, so we poll that.
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

wait_ready "http://${LLAMACPP_GEN_HOST}:${LLAMACPP_GEN_PORT}" "llama-cpp-python (gen)"
wait_ready "http://${LLAMACPP_GEN_HOST}:${LLAMACPP_EMB_PORT}" "llama-cpp-python (emb)"

# ---------------------------------------------------------------------------
# 4. Warm-up requests
# ---------------------------------------------------------------------------
echo "Warming up models with dummy requests..."
curl -sf "http://${LLAMACPP_GEN_HOST}:${LLAMACPP_GEN_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Sag Hallo in einem Wort."}],"max_tokens":8,"temperature":0}' >/dev/null
curl -sf "http://${LLAMACPP_GEN_HOST}:${LLAMACPP_EMB_PORT}/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input":"warmup"}' >/dev/null
echo "Models warmed up"

# ---------------------------------------------------------------------------
# 5. Run evaluation pipeline
# ---------------------------------------------------------------------------
echo "==== Evaluation pipeline ===="
echo "Reading RAG results from: ${RAG_RESULTS_FILE:-${RESULTS_DIR}/rag_results_latest.json}"
python -m evaluation.eval_pipeline

echo "GPU status after evaluation:"
nvidia-smi || true

echo "==== Done at $(date) ===="
echo "Results in: ${RESULTS_DIR}"
ls -lh "${RESULTS_DIR}"
