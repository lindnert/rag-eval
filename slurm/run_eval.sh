#!/bin/bash
#SBATCH --job-name=rag-eval
#SBATCH --comment="RAG Evaluation pipeline"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.lindner@campus.lmu.de
#SBATCH --chdir=/home/l/lindnerti/rag-eval
#SBATCH --output=/home/l/lindnerti/rag-eval/logs/eval.%A_%a.%N.out
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --partition=NvidiaAll
# Default array for a bare `sbatch slurm/run_eval.sh`; the login-node submitter
# below overrides it with --array=0-${ARRAY_MAX} (see the job-accounting note).
#SBATCH --array=0-14

## Two ways to start it (this script is dual-mode — see the SLURM_JOB_ID branch
## below). It reads the RAG results from results/rag_results_latest.json and
## writes evaluated_results into results/. Run it AFTER the RAG run (including
## its merge) has finished — see the job-accounting note.
##
##   1. Login-node launcher (recommended):  ./slurm/run_eval.sh
##      SLURM_JOB_ID is unset, so the script runs as a *submitter*: it sbatch-es
##      the array job for you and exits. Pick the shard count with the ARRAY_MAX
##      env var — it becomes --array=0-${ARRAY_MAX} (default 14 → 15 shards) and
##      overrides the #SBATCH --array header:
##        ARRAY_MAX=9 ./slurm/run_eval.sh        # 10 shards
##      Point it at a specific RAG results file with RAG_RESULTS_FILE:
##        RAG_RESULTS_FILE=/path/to/rag_results_YYYYMMDD.json ./slurm/run_eval.sh
##
##   2. Plain sbatch:  sbatch slurm/run_eval.sh
##      SLURM sets SLURM_JOB_ID, so the submitter branch is skipped and the
##      worker body runs directly as the array job defined by the #SBATCH
##      --array=0-14 header (15 shards). ARRAY_MAX is NOT read on this path —
##      override the size on the command line, and optionally chain it to start
##      automatically once the RAG merge job completes:
##        sbatch --array=0-9 slurm/run_eval.sh                    # 10 shards
##        sbatch --dependency=afterok:<rag_merge_jobid> slurm/run_eval.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Dual-mode entry point (mirrors slurm/run_rag.sh).
#   - On the LOGIN NODE (no SLURM_JOB_ID) this acts as a *submitter*: it
#     sbatch-es the array job and exits. The command-line --array here
#     overrides the #SBATCH --array header directive above (the header only
#     applies to a bare `sbatch`).
#   - Under SLURM (SLURM_JOB_ID set) it falls through to the worker body and
#     evaluates the RAG results read from the flat results/ tree.
#
# Job accounting (matters for the cluster's per-user submit cap, ~30 jobs):
#   * The array submits ARRAY_MAX+1 independent tasks (default 15: indices
#     0..14); SLURM counts each array task as one job.
#   * The first task to start schedules ONE merge job, held by
#     --dependency=afterok until every shard succeeds — a pending dependency
#     job still counts against the cap.
#   => one eval run = 15 shards + 1 merge = 16 jobs. Run this after the RAG
#      run finishes so the two (16 each) don't queue together and exceed 30.
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
  ARRAY_MAX="${ARRAY_MAX:-14}"
  # Resolve absolute paths so submission is independent of the cwd the user
  # launched from. This script lives in <repo>/slurm/, so REPO_ROOT is its
  # parent dir. cd there before sbatch so the array job's SLURM_SUBMIT_DIR
  # (which the worker body uses as WORKDIR) is pinned to the repo root.
  SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "${REPO_ROOT}"
  echo "Submitting eval --array=0-${ARRAY_MAX} ($((ARRAY_MAX + 1)) shards) from ${REPO_ROOT}"
  sbatch --array="0-${ARRAY_MAX}" --export=ALL "${SELF}"
  exit 0
fi

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
ERR_LOG="${WORKDIR}/logs/eval.${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.$(hostname).err"
exec 2> >(tee -a "${ERR_LOG}" >&2)

# Read the RAG results from and write evaluated_results into the flat results/
# tree. The evaluation Python is language-agnostic (abstention travels in the
# `rejected` flag and `reference_answer` already baked into the RAG results).
export RAG_LANG="${RAG_LANG:-en}"
export RESULTS_DIR="${WORKDIR}/results"
mkdir -p "${RESULTS_DIR}" "${WORKDIR}/logs"
echo "RAG_LANG=${RAG_LANG}  RESULTS_DIR=${RESULTS_DIR}"

# First-firing task (task_id=0) schedules the merge job; --dependency=afterok
# holds it until all shards in the array finish successfully.
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

GEN_REPO="${LLAMACPP_GEN_REPO:-unsloth/gemma-4-E4B-it-GGUF}"
GEN_FILE="${LLAMACPP_GEN_FILE:-gemma-4-E4B-it-UD-Q4_K_XL.gguf}"
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

# llama.cpp divides the total --ctx-size across the --parallel slots, so the
# *per-request* budget is CONTEXT_LENGTH/GEN_PARALLEL (padded up to a multiple of
# 256). At 32768/6 that is only 5632 tokens/slot — and RAGAS's AnswerCorrectness
# metric can exceed it: when the small gemma judge emits malformed JSON, RAGAS's
# recursive output-format-repair loop re-embeds the broken output each retry and
# inflates the prompt past 6k tokens, tripping an "exceeds context size" 400.
# Parallel=4 gives 32768/4 = 8192 tokens/slot with headroom; it is VRAM-neutral
# (total KV cache is sized by CONTEXT_LENGTH, not the slot count) so it does not
# worsen the nodes that already fail to load at higher memory pressure.
CONTEXT_LENGTH="${LLAMACPP_CONTEXT_LENGTH:-30000}"
GEN_PARALLEL="${LLAMACPP_GEN_PARALLEL:-3}"
# 4096 (was 2048). RAGAS embeds each string separately — AnswerCorrectness's
# semantic-similarity half embeds the generated answer and the ground truth,
# AnswerRelevancy the question — so this has to hold the longest SINGLE input,
# not their sum. The binding one is a full-budget rag_sc regen answer
# (RAG_SC_REGEN_MAX_TOKENS = 3072), slightly above the longest dataset question
# (~3000). Kept wider than run_rag.sh's embedding window (3584) because that
# ceiling is a config knob we may raise, whereas the RAG side is bounded by the
# corpus; eval is also the pipeline with VRAM to spare.
EMB_CONTEXT_LENGTH="${LLAMACPP_EMB_CONTEXT_LENGTH:-4096}"
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
# -b/-ub track EMB_CONTEXT_LENGTH: a pooled embedding input has to fit in one
# physical batch, and the default n_ubatch of 512 would reject anything longer
# regardless of how large --ctx-size is.
stdbuf -oL -eL "${LLAMACPP_SERVER}" \
  --model "${EMB_PATH}" \
  --host "${LLAMACPP_GEN_HOST}" --port "${LLAMACPP_EMB_PORT}" \
  --ctx-size "${EMB_CONTEXT_LENGTH}" \
  -b "${EMB_CONTEXT_LENGTH}" -ub "${EMB_CONTEXT_LENGTH}" \
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
