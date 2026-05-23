#!/bin/bash
#SBATCH --job-name=rag-server
#SBATCH --comment="RAG llama-cpp server build"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --partition=All
## Build llama-server once on the LOGIN node (e.g. citrin).
## The compute nodes only have CUDA 12.0 toolkit + gcc-9/13 — neither pair is
## usable together. The login node has gcc-13 + the user-installed CUDA 13
## toolkit at $HOME/cuda-13.0, which works.
##
## $HOME is shared with the compute nodes, so the resulting binary at
## $WORKDIR/.llamacpp_bin/$LLAMACPP_TAG/llama-server is visible to SLURM jobs.
##
## Re-run only when bumping LLAMACPP_TAG.

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMACPP_TAG="${LLAMACPP_TAG:-master}"
LLAMACPP_BIN_DIR="${WORKDIR}/.llamacpp_bin/${LLAMACPP_TAG}"
SRC_DIR="${WORKDIR}/.llamacpp_src/${LLAMACPP_TAG}"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda-13.0}"
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  echo "ERROR: nvcc not found at ${CUDA_HOME}/bin/nvcc" >&2
  echo "Install CUDA 13 toolkit:" >&2
  echo "  wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.run" >&2
  echo "  sh cuda_13.0.0_580.65.06_linux.run --toolkit --toolkitpath=\$HOME/cuda-13.0 --silent --override" >&2
  exit 1
fi

CC="${CC:-/usr/bin/gcc-13}"
CXX="${CXX:-/usr/bin/g++-13}"
echo "Building llama-server (${LLAMACPP_TAG})"
echo "  CUDA_HOME=${CUDA_HOME}"
echo "  CC=${CC}, CXX=${CXX}"
echo "  Output: ${LLAMACPP_BIN_DIR}/llama-server"

rm -rf "${SRC_DIR}" "${LLAMACPP_BIN_DIR}"
git clone --depth 1 --branch "${LLAMACPP_TAG}" https://github.com/ggml-org/llama.cpp "${SRC_DIR}"

# Build for the compute node's GPU explicitly. The login node has no GPU,
# so -arch=native (llama.cpp default) falls back to a generic Maxwell-era
# target and skips Turing-specific kernels.
#   - RTX 2060 SUPER = sm_75 (Turing)
CUDA_ARCH="${CUDA_ARCH:-75}"
echo "  CUDA_ARCH=${CUDA_ARCH}"

CUDA_HOME="${CUDA_HOME}" \
PATH="${CUDA_HOME}/bin:${PATH}" \
LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}" \
CC="${CC}" CXX="${CXX}" \
cmake -S "${SRC_DIR}" -B "${SRC_DIR}/build" \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="${CXX}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,\$ORIGIN -Wl,-rpath,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,\$ORIGIN -Wl,-rpath,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"

LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}" \
LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" \
cmake --build "${SRC_DIR}/build" --config Release -j --target llama-server

mkdir -p "${LLAMACPP_BIN_DIR}"
cp "${SRC_DIR}/build/bin/llama-server" "${LLAMACPP_BIN_DIR}/"
# Copy all shared libs (including versioned ones like libggml-cuda.so.0).
# Use -a to preserve symlinks (libfoo.so → libfoo.so.0 → libfoo.so.0.x.y).
cp -a "${SRC_DIR}/build/bin/"*.so* "${LLAMACPP_BIN_DIR}/" 2>/dev/null || true

echo "Done. Binary at: ${LLAMACPP_BIN_DIR}/llama-server"
ls -lh "${LLAMACPP_BIN_DIR}/"
