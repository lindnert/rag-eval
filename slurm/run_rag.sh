#!/bin/bash
#SBATCH --job-name=rag
#SBATCH --comment="RAG pipeline"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.lindner@campus.lmu.de
#SBATCH --chdir=/home/l/lindnerti/rag-eval
#SBATCH --output=/home/l/lindnerti/rag-eval/logs/rag.%j.%N.out
#SBATCH --error=/home/l/lindnerti/rag-eval/logs/rag.%j.%N.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --partition=NvidiaAll

## Submit with: sbatch slurm/run_rag.sh

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
export RESULTS_DIR="${WORKDIR}/results"
mkdir -p "${RESULTS_DIR}" "${WORKDIR}/logs"

PYTHON_BIN="$(command -v python3.12 || command -v python3)"

echo "==== RAG job ${SLURM_JOB_ID} on $(hostname) ===="
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

# ---------------------------------------------------------------------------
# 2. Install Ollama into $HOME
# ---------------------------------------------------------------------------
OLLAMA_DIR="${HOME}/.local/ollama"
mkdir -p "${OLLAMA_DIR}/bin"
export PATH="${OLLAMA_DIR}/bin:${PATH}"
export OLLAMA_MODELS="${WORKDIR}/.ollama_models"
mkdir -p "${OLLAMA_MODELS}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama into ${OLLAMA_DIR} ..."
  curl -fsSL https://github.com/ollama/ollama/releases/download/v0.23.1/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst
  zstd -d /tmp/ollama.tar.zst -o /tmp/ollama.tar
  tar -xf /tmp/ollama.tar -C "${OLLAMA_DIR}"
fi

# ---------------------------------------------------------------------------
# 3. Start Ollama server
# ---------------------------------------------------------------------------
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_NUM_PARALLEL=10
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_KEEP_ALIVE=-1

pkill -f "ollama serve" || true
sleep 2
ollama serve > "${WORKDIR}/logs/ollama_rag_${SLURM_JOB_ID:-local}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill ${OLLAMA_PID} 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
  if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null; then
    echo "Ollama ready after ${i}s"
    break
  fi
  sleep 1
done

# ---------------------------------------------------------------------------
# 4. Pull + warm models needed for RAG
# ---------------------------------------------------------------------------
ollama pull gemma3:4b
ollama pull nomic-embed-text

echo "Warming up models with dummy requests..."
ollama ps

curl -sf "http://${OLLAMA_HOST}/api/generate" \
  -d '{"model":"gemma3:4b","prompt":"Sag Hallo in einem Wort.","stream":false,"options":{"num_ctx":6000}}' >/dev/null
curl -sf "http://${OLLAMA_HOST}/api/embeddings" \
  -d '{"model":"nomic-embed-text","prompt":"warmup"}' >/dev/null
echo "Models warmed up"

echo "Current Ollama status:"
ollama ps

# ---------------------------------------------------------------------------
# 5. Run RAG pipeline
# ---------------------------------------------------------------------------
echo "==== RAG pipeline ===="
python -m rag.rag_pipeline

echo "GPU status after RAG:"
ollama ps

echo "==== Done at $(date) ===="
echo "Results in: ${RESULTS_DIR}"
ls -lh "${RESULTS_DIR}"
