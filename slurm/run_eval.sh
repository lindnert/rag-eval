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
# Node driver is CUDA 13 — forward-compatible with cu124 wheels.
# (cu125 does NOT exist on the abetlen index; with `--upgrade` + `>=` pip
#  silently fell back to the CPU wheel on PyPI, leaving GPU at 0%.)
# Pin EXACTLY and use --index-url so PyPI is only fallback for non-llama deps.
# Already-installed wheels are a pip no-op ("Requirement already satisfied").
LLAMA_CPP_PY_VERSION="${LLAMA_CPP_PY_VERSION:-0.3.23}"
pip install --no-cache-dir \
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
  --extra-index-url https://pypi.org/simple \
  "llama-cpp-python==${LLAMA_CPP_PY_VERSION}"
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
# 3. Build native llama-server (once, cached) — needed because llama_cpp.server
#    (Python wrapper) doesn't expose --parallel for gen and hardcodes
#    n_seq_max=256 for embeddings → KV-cache OOM. Native binary fixes both.
# ---------------------------------------------------------------------------
LLAMACPP_TAG="${LLAMACPP_TAG:-master}"
LLAMACPP_BIN_DIR="${WORKDIR}/.llamacpp_bin/${LLAMACPP_TAG}"
LLAMACPP_SERVER="${LLAMACPP_BIN_DIR}/llama-server"

if [ ! -x "${LLAMACPP_SERVER}" ]; then
  echo "Building llama-server (${LLAMACPP_TAG}) — one-time, cached at ${LLAMACPP_BIN_DIR}"
  SRC_DIR="${WORKDIR}/.llamacpp_src/${LLAMACPP_TAG}"
  rm -rf "${SRC_DIR}"
  git clone --depth 1 --branch "${LLAMACPP_TAG}" https://github.com/ggml-org/llama.cpp "${SRC_DIR}"
  # CUDA 12.0 supports gcc 10-12 only:
  #   - gcc < 10 hits a <type_traits> incompatibility under C++17
  #   - gcc > 12 is rejected by /usr/include/crt/host_config.h
  # Pick highest supported version present on the node.
  BUILD_CXX=""
  for cand in g++-12 g++-11 g++-10; do
    if command -v "${cand}" >/dev/null 2>&1; then
      BUILD_CXX="$(command -v "${cand}")"
      BUILD_CC="$(command -v "${cand/g++/gcc}")"
      break
    fi
  done
  if [ -z "${BUILD_CXX}" ]; then
    echo "ERROR: no compatible gcc found (need gcc-10, -11, or -12 for CUDA 12.0)" >&2
    exit 1
  fi
  echo "Using CC=${BUILD_CC}, CXX=${BUILD_CXX}"
  CC="${BUILD_CC}" CXX="${BUILD_CXX}" cmake -S "${SRC_DIR}" -B "${SRC_DIR}/build" \
    -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_HOST_COMPILER="${BUILD_CXX}"
  cmake --build "${SRC_DIR}/build" --config Release -j --target llama-server
  mkdir -p "${LLAMACPP_BIN_DIR}"
  cp "${SRC_DIR}/build/bin/llama-server" "${LLAMACPP_BIN_DIR}/"
  # Bundle shared libs next to the binary (avoids LD_LIBRARY_PATH gymnastics).
  cp "${SRC_DIR}/build/bin/"*.so "${LLAMACPP_BIN_DIR}/" 2>/dev/null || true
  rm -rf "${SRC_DIR}"
fi
echo "llama-server: ${LLAMACPP_SERVER}"

# ---------------------------------------------------------------------------
# 4. Start gen (8080) + emb (8081) servers, each configured independently.
# ---------------------------------------------------------------------------
export LLAMACPP_GEN_HOST="127.0.0.1"
export LLAMACPP_GEN_PORT=8080
export LLAMACPP_EMB_PORT=8081
export LLAMACPP_GEN_BASE_URL="http://${LLAMACPP_GEN_HOST}:${LLAMACPP_GEN_PORT}/v1"
export LLAMACPP_EMB_BASE_URL="http://${LLAMACPP_GEN_HOST}:${LLAMACPP_EMB_PORT}/v1"

# Gen: n_ctx is total across slots → 32768/4 = 8192 per slot, matches NUM_PREDICT.
CONTEXT_LENGTH="${LLAMACPP_CONTEXT_LENGTH:-32768}"
GEN_PARALLEL="${LLAMACPP_GEN_PARALLEL:-4}"
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
  -fa \
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

python -m evaluation.eval_pipeline

kill ${GPU_SAMPLER_PID} 2>/dev/null || true

echo "GPU status after evaluation:"
nvidia-smi || true
echo "---- GPU sample log (${GPU_LOG}) ----"
cat "${GPU_LOG}" || true

echo "==== Done at $(date) ===="
echo "Results in: ${RESULTS_DIR}"
ls -lh "${RESULTS_DIR}"
