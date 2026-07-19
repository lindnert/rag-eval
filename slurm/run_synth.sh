#!/bin/bash
#SBATCH --job-name=synth-gen
#SBATCH --comment="Synthetic dataset generation (deepeval)"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.lindner@campus.lmu.de
#SBATCH --chdir=/home/l/lindnerti/rag-eval
#SBATCH --output=/home/l/lindnerti/rag-eval/logs/synth.%j.%N.out
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --partition=NvidiaAll

## Single-node job — NO array, NO shard merge. ~150 goldens don't warrant the
## fleet, quality does: this runs the largest model that fits one 8 GB GPU
## (8B-class Q4; override LLAMACPP_GEN_REPO / LLAMACPP_GEN_FILE for another).
## Pilot first: SYNTH_MAX_CONTEXTS=6 sbatch slurm/run_synth.sh
## Escape hatch: if pilot quality disappoints, the Python is endpoint-agnostic
## — point LLAMACPP_GEN_BASE_URL at an Ollama/llama-server /v1 endpoint on the
## 12 GB node (branch test/new-node-larger-models) and run
## `python -m dataset.synthetic.generate_synthetic` there directly.
##
## Prerequisites (login node):
##   1. dataset/synthetic/contexts_mixed.json + personas.json committed
##      (python -m dataset.synthetic.build_contexts — needs the cosine index)
##   2. .venv built (slurm/build_venv.sh), llama-server built
##      (slurm/build_llama_server.sh)
##
## Resume: goldens_<pass>.json files are kept; a requeued job skips finished
## passes and the validation step reuses cached faithfulness scores.

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
ERR_LOG="${WORKDIR}/logs/synth.${SLURM_JOB_ID:-local}.$(hostname).err"
mkdir -p "${WORKDIR}/logs"
exec 2> >(tee -a "${ERR_LOG}" >&2)

echo "==== Synth job ${SLURM_JOB_ID:-local} on $(hostname) ===="
echo "Start: $(date)"
nvidia-smi || true

source "${WORKDIR}/.venv/bin/activate"

# ---------------------------------------------------------------------------
# Model download (generator — critic is the same server at temp 0).
# 8 GB VRAM budget: E4B/8B-class Q4 (~4-5 GB weights) + q8_0 KV fits easily;
# a Q6_K quant of the same model also fits if you want a quality bump.
# ---------------------------------------------------------------------------
export HF_HOME="${WORKDIR}/.hf_cache"
export LLAMA_MODELS_DIR="${WORKDIR}/.llamacpp_models"
mkdir -p "${HF_HOME}" "${LLAMA_MODELS_DIR}"

GEN_REPO="${LLAMACPP_GEN_REPO:-unsloth/gemma-4-E4B-it-GGUF}"
GEN_FILE="${LLAMACPP_GEN_FILE:-gemma-4-E4B-it-UD-Q4_K_XL.gguf}"

echo "Downloading ${GEN_REPO}/${GEN_FILE} (cached in ${HF_HOME})..."
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${GEN_REPO}', filename='${GEN_FILE}', local_dir='${LLAMA_MODELS_DIR}/gen')"
GEN_PATH="${LLAMA_MODELS_DIR}/gen/${GEN_FILE}"

# ---------------------------------------------------------------------------
# llama-server (gen only — no embedding server needed for synthesis)
# ---------------------------------------------------------------------------
LLAMACPP_TAG="${LLAMACPP_TAG:-master}"
LLAMACPP_SERVER="${WORKDIR}/.llamacpp_bin/${LLAMACPP_TAG}/llama-server"
CUDA_HOME="${CUDA_HOME:-$HOME/cuda-13.0}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
if [ ! -x "${LLAMACPP_SERVER}" ]; then
  echo "ERROR: llama-server not found at ${LLAMACPP_SERVER} — see slurm/build_llama_server.sh" >&2
  exit 1
fi

export LLAMACPP_GEN_HOST="127.0.0.1"
export LLAMACPP_GEN_PORT=8080
export LLAMACPP_GEN_BASE_URL="http://${LLAMACPP_GEN_HOST}:${LLAMACPP_GEN_PORT}/v1"
export LLAMACPP_EVAL_MODEL="${GEN_REPO}:${GEN_FILE%.gguf}"
export SYNTH_GENERATOR_TAG="${SYNTH_GENERATOR_TAG:-${GEN_FILE%.gguf}}"

# --ctx-size is TOTAL and is split across --parallel slots ->
# 24576/3 = 8192 tokens per slot, plenty for 3-chunk contexts (~1.5k tokens)
# + synthesizer templates.
CONTEXT_LENGTH="${LLAMACPP_CONTEXT_LENGTH:-24576}"
GEN_PARALLEL="${LLAMACPP_GEN_PARALLEL:-3}"
export SYNTH_MAX_CONCURRENT="${SYNTH_MAX_CONCURRENT:-${GEN_PARALLEL}}"
echo "GEN_PARALLEL=${GEN_PARALLEL}, CTX=${CONTEXT_LENGTH} (per slot: $((CONTEXT_LENGTH / GEN_PARALLEL)))"

pkill -f "llama-server" || true
sleep 2

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
trap 'kill ${GEN_PID} 2>/dev/null || true; pkill -f "llama-server" 2>/dev/null || true' EXIT

for i in $(seq 1 600); do
  if curl -sf "${LLAMACPP_GEN_BASE_URL%/v1}/v1/models" >/dev/null; then
    echo "llama-server ready after ${i}s"; break
  fi
  if [ "$i" = "600" ]; then echo "ERROR: llama-server not ready" >&2; exit 1; fi
  sleep 1
done

curl -sf "${LLAMACPP_GEN_BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":8,"temperature":0}' >/dev/null
echo "Warm-up done. GPU status:"
nvidia-smi || true

# ---------------------------------------------------------------------------
# Generate + validate
# ---------------------------------------------------------------------------
export DEEPEVAL_TELEMETRY_OPT_OUT=YES
export SYNTH_OUTPUT_DIR="${SYNTH_OUTPUT_DIR:-results/synthetic}"
mkdir -p "${WORKDIR}/${SYNTH_OUTPUT_DIR}"

echo "==== Generation ===="
python -m dataset.synthetic.generate_synthetic

echo "==== Validation ===="
python -m dataset.synthetic.validate_synthetic

echo "==== Done at $(date) ===="
ls -lh "${WORKDIR}/${SYNTH_OUTPUT_DIR}"
