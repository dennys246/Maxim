#!/bin/bash
# Build llama-cpp-python with CUDA 12.8 support for Blackwell (sm_120) GPUs.
# Run from WSL Ubuntu-24.04 with the maxim-env Python venv.
set -e

VENV_PIP="${VENV_PIP:-/mnt/d/Envs/WSL/maxim-env/bin/pip}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
LLAMA_VERSION="${LLAMA_VERSION:-0.3.20}"

export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120"
export FORCE_CMAKE=1

echo "Building llama-cpp-python $LLAMA_VERSION for sm_120 (Blackwell)..."
echo "  CUDACXX=$CUDACXX"
echo "  CMAKE_ARGS=$CMAKE_ARGS"
echo ""

"$VENV_PIP" install \
    --upgrade \
    --force-reinstall \
    --no-cache-dir \
    --no-binary=llama-cpp-python \
    "llama-cpp-python==$LLAMA_VERSION"
