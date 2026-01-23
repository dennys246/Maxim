#!/bin/bash
# Download models for Maxim
#
# Usage:
#   ./scripts/download_models.sh           # Download default LLM + TTS
#   ./scripts/download_models.sh --llm     # Download just LLM
#   ./scripts/download_models.sh --tts     # Download just TTS
#   ./scripts/download_models.sh --list    # List available models
#   ./scripts/download_models.sh --help    # Show help

set -e

# Change to repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Set PYTHONPATH to include src
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Use python3 or python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: Python not found"
    exit 1
fi

# Run the download module
exec "$PYTHON" -m maxim.models.download "$@"
