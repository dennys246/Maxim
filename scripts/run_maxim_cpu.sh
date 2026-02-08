#!/bin/bash
# Run Maxim with CPU-only TensorFlow
# Use this script if you encounter GPU compatibility issues or want to force CPU mode

# Force TensorFlow to use CPU only
export CUDA_VISIBLE_DEVICES=""

# Enable LLM if llama-cpp-python is installed
if python -c "import llama_cpp" 2>/dev/null; then
    export MAXIM_LLM_ENABLED=1
    export MAXIM_LLM_PROFILE="smollm-1.7b-instruct"
    echo "LLM enabled: smollm-1.7b-instruct"
fi

echo "╔════════════════════════════════════════════════════════╗"
echo "║        Running Maxim in CPU-only mode                  ║"
echo "║  GPU acceleration is intentionally disabled            ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

maxim "$@"
