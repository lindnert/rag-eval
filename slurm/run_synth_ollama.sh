#!/bin/bash
# Synthetic dataset generation against the EXTERNAL Ollama node (12 GB GPU) —
# NOT a SLURM job. The node is HTTP-only (OpenAI-compatible endpoint + Bearer
# key, no shell access), so this script runs wherever the repo and a Python
# env live — local Git Bash or the cluster login node — and only the
# chat-completions calls leave the machine. No llama-server, no GPU here.
#
# Config comes from the repo .env (gitignored; same vars as rag/llm_config.py):
#   LLAMACPP_GEN_BASE_URL   https://<host>/ollama/v1
#   LLAMACPP_GEN_API_KEY    Bearer token (falls back to OLLAMA_API_KEY)
#
# Usage (from anywhere; the script cd's to the repo root):
#   bash slurm/run_synth_ollama.sh                                # gemma4:e4b
#   SYNTH_MAX_CONTEXTS=6 bash slurm/run_synth_ollama.sh           # pilot
#   SYNTH_OLLAMA_MODEL=gemma4:12b bash slurm/run_synth_ollama.sh  # escalate
#
# Output goes to a per-run dir dataset/synthetic/generated_<model>_<timestamp>/
# so re-runs never collide or overwrite an earlier dataset. To RESUME an
# interrupted run instead, pass its existing dir as SYNTH_OUTPUT_DIR — finished
# goldens_<pass>.json files inside it are skipped.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate  # Windows Git Bash layout
fi

# Python (synth_llm.py) also loads .env itself; sourcing it here too gives the
# script access to the URL/key for the preflight check below.
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${LLAMACPP_GEN_BASE_URL:?set LLAMACPP_GEN_BASE_URL=https://<host>/ollama/v1 in .env}"
export LLAMACPP_GEN_API_KEY="${LLAMACPP_GEN_API_KEY:-${OLLAMA_API_KEY:-}}"
: "${LLAMACPP_GEN_API_KEY:?set LLAMACPP_GEN_API_KEY (or OLLAMA_API_KEY) in .env}"

MODEL="${SYNTH_OLLAMA_MODEL:-gemma4:e4b}"
MODEL_SLUG="${MODEL//[:\/]/-}"
export LLAMACPP_EVAL_MODEL="${MODEL}"
export SYNTH_GENERATOR_TAG="${SYNTH_GENERATOR_TAG:-ollama-${MODEL_SLUG}}"
# Ollama queues requests beyond its server-side parallelism, so higher values
# wouldn't crash — but 2 keeps the queue shallow on a single-GPU box we don't
# control (and that others may be using).
export SYNTH_MAX_CONCURRENT="${SYNTH_MAX_CONCURRENT:-2}"
# Each run gets its own timestamped dir so a re-run never overwrites or resumes
# a previous run's dataset. To deliberately RESUME/extend an interrupted run
# instead, point SYNTH_OUTPUT_DIR at its existing dir:
#   SYNTH_OUTPUT_DIR=dataset/synthetic/generated_gemma4-e4b_20260720_143000 bash slurm/run_synth_ollama.sh
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
export SYNTH_OUTPUT_DIR="${SYNTH_OUTPUT_DIR:-dataset/synthetic/generated_${MODEL_SLUG}_${RUN_STAMP}}"
export DEEPEVAL_TELEMETRY_OPT_OUT=YES
mkdir -p "${SYNTH_OUTPUT_DIR}"

echo "Endpoint    : ${LLAMACPP_GEN_BASE_URL}"
echo "Model       : ${MODEL}"
echo "Concurrency : ${SYNTH_MAX_CONCURRENT}"
echo "Output      : ${SYNTH_OUTPUT_DIR}"

# Preflight: endpoint reachable and the model tag actually exists on the node.
MODELS_JSON=$(curl -sf --max-time 30 \
  -H "Authorization: Bearer ${LLAMACPP_GEN_API_KEY}" \
  "${LLAMACPP_GEN_BASE_URL}/models") || {
  echo "ERROR: cannot reach ${LLAMACPP_GEN_BASE_URL}/models (check URL/key/VPN)" >&2
  exit 1
}
if ! printf '%s' "${MODELS_JSON}" | grep -q "\"${MODEL}\""; then
  echo "ERROR: model '${MODEL}' not found on the node. Available:" >&2
  printf '%s\n' "${MODELS_JSON}" | python -c "import json,sys; [print(' -', m['id']) for m in json.load(sys.stdin)['data']]" >&2
  exit 1
fi
echo "Preflight OK — '${MODEL}' is available on the node."

echo "==== Generation ===="
python -m dataset.synthetic.generate_synthetic

echo "==== Validation ===="
python -m dataset.synthetic.validate_synthetic

echo "==== Done at $(date) ===="
ls -lh "${SYNTH_OUTPUT_DIR}"
