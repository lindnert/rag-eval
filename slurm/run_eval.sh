#!/bin/bash
#SBATCH --job-name=rag-eval
#SBATCH --comment="RAG Evaluation pipeline"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.lindner@campus.lmu.de
#SBATCH --chdir=/home/l/lindnerti/rag-eval
#SBATCH --output=/home/l/lindnerti/rag-eval/logs/eval.%A_%a.%N.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --partition=NvidiaAll
# Default array for a bare `sbatch slurm/run_eval.sh`; the login-node submitter
# below overrides it with --array=0-${ARRAY_MAX} (see the job-accounting note).
#SBATCH --array=0-12
# balance out workload across shards (should be not dividable by 3 variants) + 1 stage + 1 merge job


## Two ways to start it (this script is dual-mode — see the SLURM_JOB_ID branch
## below). It reads the NEWEST results/rag_results_<timestamp>.json and writes
## results/evaluated_results_<evalts>_from_<ragts>.json — the trailing stamp is
## the RAG file it consumed, so every eval file names its own input. Run it
## AFTER the RAG run (including its merge) has finished — see the job-accounting
## note.
##
##   1. Login-node launcher (recommended):  ./slurm/run_eval.sh
##      SLURM_JOB_ID is unset, so the script runs as a *submitter*: it sbatch-es
##      the array job for you and exits. Pick the shard count with the ARRAY_MAX
##      env var — it becomes --array=0-${ARRAY_MAX} (default 14 → 15 shards) and
##      overrides the #SBATCH --array header:
##        ARRAY_MAX=9 ./slurm/run_eval.sh        # 10 shards
##      Override the newest-file default with RAG_RESULTS_FILE:
##        RAG_RESULTS_FILE=/path/to/rag_results_YYYYMMDD.json ./slurm/run_eval.sh
##
##   2. Plain sbatch:  sbatch slurm/run_eval.sh
##      SLURM sets SLURM_JOB_ID, so the submitter branch is skipped and the
##      worker body runs directly as the array job defined by the #SBATCH
##      --array=0-14 header (15 shards). Note that RAG_RESULTS_FILE is NOT
##      resolved on this path, so each shard picks the newest RAG file on its
##      own and the merged filename carries no _from_<ragts> stamp — use the
##      launcher (or export RAG_RESULTS_FILE yourself) to keep that provenance.
##      ARRAY_MAX is NOT read on this path either —
##      override the size on the command line, and optionally chain it to start
##      automatically once the RAG merge job completes:
##        sbatch --array=0-9 slurm/run_eval.sh                    # 10 shards
##        sbatch --dependency=afterok:<rag_merge_jobid> slurm/run_eval.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Model GGUFs served by the eval llama-servers. Declared here — above the
# dual-mode split — so the login-node submitter and the array workers resolve
# the SAME repo/file, and the submitter can pre-stage them (see stage_models).
# ---------------------------------------------------------------------------
GEN_REPO="${LLAMACPP_GEN_REPO:-unsloth/gemma-4-E4B-it-GGUF}"
GEN_FILE="${LLAMACPP_GEN_FILE:-gemma-4-E4B-it-UD-Q4_K_XL.gguf}"
EMB_REPO="${LLAMACPP_EMB_REPO:-Qwen/Qwen3-Embedding-0.6B-GGUF}"
EMB_FILE="${LLAMACPP_EMB_FILE:-Qwen3-Embedding-0.6B-Q8_0.gguf}"

# Download both GGUFs into the shared model dir under $1 (a work dir) and set
# HF_HOME / LLAMA_MODELS_DIR / GEN_PATH / EMB_PATH for the caller. Idempotent:
# hf_hub_download is a no-op on a warm cache. The submitter calls this ONCE
# before fanning out the array so the workers hit a warm cache instead of all
# racing to download a brand-new file into shared NFS at the same time — the
# cold-cache race that fails a shard with a "failed to open GGUF file (No such
# file or directory)" model-load error the first time a new model is used.
# Uses the Python API directly — the `hf` CLI leaks a click.exceptions.Exit(0)
# on success in some typer/click combos, which trips `set -e`.
stage_models() {
  local work="$1"
  export HF_HOME="${work}/.hf_cache"
  export LLAMA_MODELS_DIR="${work}/.llamacpp_models"
  mkdir -p "${HF_HOME}" "${LLAMA_MODELS_DIR}"
  echo "Staging GGUFs (cached in ${HF_HOME})..."
  python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${GEN_REPO}', filename='${GEN_FILE}', local_dir='${LLAMA_MODELS_DIR}/gen')"
  python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${EMB_REPO}', filename='${EMB_FILE}', local_dir='${LLAMA_MODELS_DIR}/emb')"
  GEN_PATH="${LLAMA_MODELS_DIR}/gen/${GEN_FILE}"
  EMB_PATH="${LLAMA_MODELS_DIR}/emb/${EMB_FILE}"
  # Fail fast with a clear message instead of letting llama-server die later with
  # the cryptic "failed to open GGUF file" above (a partial/absent download).
  if [ ! -s "${GEN_PATH}" ] || [ ! -s "${EMB_PATH}" ]; then
    echo "ERROR: model staging incomplete (GEN_PATH=${GEN_PATH}, EMB_PATH=${EMB_PATH})" >&2
    return 1
  fi
}

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
#   * The submitter first launches ONE stage job (downloads the model GGUFs to
#     shared NFS) and holds the array behind it with --dependency=afterok.
#   * The array submits ARRAY_MAX+1 independent tasks (default 15: indices
#     0..14); SLURM counts each array task as one job.
#   * The first task to start schedules ONE merge job, held by
#     --dependency=afterok until every shard succeeds — a pending dependency
#     job still counts against the cap.
#   => one eval run = 1 stage + 15 shards + 1 merge = 17 jobs. Run this after the
#      RAG run finishes so the two (17 each) don't queue together and exceed 30.
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
  ARRAY_MAX="${ARRAY_MAX:-12}"
  # balance out workload across shards (should be not dividable by 3 variants) + 1 stage + 1 merge job
  # Resolve absolute paths so submission is independent of the cwd the user
  # launched from. This script lives in <repo>/slurm/, so REPO_ROOT is its
  # parent dir. cd there before sbatch so the array job's SLURM_SUBMIT_DIR
  # (which the worker body uses as WORKDIR) is pinned to the repo root.
  SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cd "${REPO_ROOT}"
  # Resolve the RAG results file ONCE, here, and export it so the shards and the
  # merge job all read the same run even if a new RAG run lands mid-eval. The
  # merge names its output after this file's stamp
  # (evaluated_results_<evalts>_from_<ragts>.json), which is how an eval file
  # says which RAG run produced it. Filenames embed YYYYMMDD_HHMMSS and so sort
  # chronologically; the [0-9] glob skips any unstamped leftover file.
  if [ -z "${RAG_RESULTS_FILE:-}" ]; then
    RAG_RESULTS_FILE="$(ls -1 "${REPO_ROOT}"/results/rag_results_[0-9]*.json 2>/dev/null | sort | tail -1)"
    if [ -z "${RAG_RESULTS_FILE}" ]; then
      echo "ERROR: no results/rag_results_<timestamp>.json found — run the RAG pipeline first," >&2
      echo "       or point at one explicitly: RAG_RESULTS_FILE=/path/to/file.json $0" >&2
      exit 1
    fi
  fi
  export RAG_RESULTS_FILE
  echo "RAG results: ${RAG_RESULTS_FILE}"
  # Download the model GGUFs in a dedicated one-off job (NOT on the login node),
  # then hold the array behind it with afterok so every shard starts against a
  # warm NFS cache and none cold-race. STAGE_ONLY=1 runs this same script in
  # stage-and-exit mode (see the worker body); --array=0 collapses the #SBATCH
  # --array header to a single task; --time is short since it only downloads.
  # No GPU is needed to download, so the stage job runs on the CPU partition
  # `All` (always available) by default; override with STAGE_PARTITION.
  STAGE_PARTITION="${STAGE_PARTITION:-All}"
  STAGE_ARGS=(--parsable --job-name=eval-stage --array=0 --time=00:40:00
              --partition="${STAGE_PARTITION}" --export=ALL,STAGE_ONLY=1)
  STAGE_JOBID=$(sbatch "${STAGE_ARGS[@]}" "${SELF}")
  echo "Submitted eval-stage job ${STAGE_JOBID} (downloads GGUFs to shared NFS)"
  echo "Submitting eval --array=0-${ARRAY_MAX} ($((ARRAY_MAX + 1)) shards), held on afterok:${STAGE_JOBID}"
  sbatch --array="0-${ARRAY_MAX}" --dependency="afterok:${STAGE_JOBID}" --export=ALL "${SELF}"
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

# STAGE-ONLY invocation: launched by the login-node submitter as a dedicated
# download job that the real array waits on (afterok). Its only purpose is to
# populate the shared NFS model cache so the shards never cold-race — no merge
# job, no GPU, no servers. Activate the venv, stage, and exit.
if [ "${STAGE_ONLY:-0}" = "1" ]; then
  if [ ! -d "${WORKDIR}/.venv" ]; then
    echo "ERROR: venv not found at ${WORKDIR}/.venv — run slurm/build_venv.sh on the login node" >&2
    exit 1
  fi
  source "${WORKDIR}/.venv/bin/activate"
  stage_models "${WORKDIR}"
  echo "stage-only: models staged to ${LLAMA_MODELS_DIR}, exiting"
  exit 0
fi

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
# 2. Stage GGUFs (warm-cache hit — the submitter pre-staged them; this just
#    resolves GEN_PATH/EMB_PATH and re-verifies they're present on this node).
# ---------------------------------------------------------------------------
# Run Ragas FaithfulnesswithHHEM on CPU — the GPU is fully occupied by the
# llama-server gen/emb instances, and HHEM (184M params) is fast enough on CPU.
export HHEM_DEVICE="${HHEM_DEVICE:-cpu}"

# Production eval: EVAL_DEBUG_LLM=0 disables the verbose per-LLM-call prompt
# logging. It no longer affects error handling — ragas evaluate() now always runs
# with raise_exceptions=False so a single metric that fails to parse (small judge
# → occasional bad JSON) degrades THAT metric to None instead of nulling the whole
# sample; ragas_eval captures the per-metric reason separately. Set =1 to debug.
export EVAL_DEBUG_LLM="${EVAL_DEBUG_LLM:-0}"

# RAGAS scorer budget and concurrency. These were previously left to the code
# defaults and were invisible here, which hid the cause of a mass metric failure:
# ragas applies RAGAS_TIMEOUT as a per-metric WALL-CLOCK deadline that includes
# time spent queueing on the llama-server slots, so the concurrency below has to
# stay within GEN_PARALLEL or every multi-call metric times out at once.
#   RAGAS_CONCURRENCY x RAGAS_MAX_WORKERS  <=  GEN_PARALLEL (see below)
export RAGAS_TIMEOUT="${RAGAS_TIMEOUT:-300}"        # seconds, per metric per sample
export RAGAS_CONCURRENCY="${RAGAS_CONCURRENCY:-3}"  # samples evaluated in parallel
export RAGAS_MAX_WORKERS="${RAGAS_MAX_WORKERS:-1}"  # metrics in parallel per sample

# DeepEval runs in a separate phase against the SAME gen server, so it is bound
# by the same slot budget: its semaphore caps in-flight SAMPLES, and each sample
# fans out to three metrics, so anything above GEN_PARALLEL just queues. The code
# default is 6 — double the slots — which is why this is pinned here rather than
# left implicit (the same mistake that hid the RAGAS timeout cascade).
export DEEPEVAL_CONCURRENCY="${DEEPEVAL_CONCURRENCY:-3}"  # samples in parallel

# Judge model id the eval client sends in /v1/chat/completions and records for
# provenance. DERIVED from GEN_REPO + the quant tag parsed out of GEN_FILE
# (unsloth convention: repo ends in -GGUF, file is <stem>-<quant>.gguf) so it
# cannot drift from the loaded weights the way the stale eval_config default did
# (it said E2B while the server loads E4B). An explicit env override still wins;
# the local single-model llama-server ignores the field regardless.
_stem="${GEN_REPO##*/}"; _stem="${_stem%-GGUF}"
_quant="${GEN_FILE%.gguf}"; _quant="${_quant#${_stem}-}"
export LLAMACPP_EVAL_MODEL="${LLAMACPP_EVAL_MODEL:-${GEN_REPO}:${_quant}}"

stage_models "${WORKDIR}"
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
# Set by the login-node submitter; on a plain `sbatch` it is unset and
# eval_pipeline falls back to resolving the newest RAG file itself.
echo "Reading RAG results from: ${RAG_RESULTS_FILE:-<newest results/rag_results_*.json>}"

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
