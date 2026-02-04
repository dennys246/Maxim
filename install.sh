#!/bin/bash
# Maxim Installation Script

set -e  # Exit on error

echo "===== Maxim Installation ====="
echo ""

# Ask about optional video processing support
echo "Do you want to install video processing support (PyAV)?"
echo "This requires FFmpeg system libraries and may need compilation."
echo ""
read -p "Install video support? [y/N]: " -n 1 -r
echo ""

INSTALL_VIDEO=false
if [[ $REPLY =~ ^[Yy]$ ]]; then
    INSTALL_VIDEO=true

    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "linux" ]]; then
        echo "Detected Linux system"

        # Check if running in WSL
        if grep -qi microsoft /proc/version 2>/dev/null; then
            echo "Running in WSL"
        fi

        echo ""
        echo "Installing system dependencies (FFmpeg libraries)..."
        echo "This requires sudo privileges."
        echo ""

        sudo apt update
        sudo apt install -y \
            libavformat-dev \
            libavcodec-dev \
            libavdevice-dev \
            libavutil-dev \
            libavfilter-dev \
            libswscale-dev \
            libswresample-dev \
            pkg-config

        echo ""
        echo "System dependencies installed successfully!"

    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Detected macOS"

        if ! command -v brew &> /dev/null; then
            echo "Error: Homebrew is not installed. Please install it first:"
            echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi

        echo ""
        echo "Installing system dependencies (FFmpeg, Cairo, GObject)..."
        brew install ffmpeg pkg-config cairo gobject-introspection

        echo ""
        echo "System dependencies installed successfully!"

    else
        echo "Warning: Unsupported OS type: $OSTYPE"
        echo "Please manually install FFmpeg development libraries for your system."
        echo ""
    fi
else
    echo "Skipping video processing support (can be added later with: pip install -e '.[video]')"
    echo ""

    # Still need Cairo/GObject for reachy-mini[gstreamer] on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Installing required macOS dependencies (Cairo, GObject)..."
        if command -v brew &> /dev/null; then
            brew install cairo gobject-introspection pkg-config
        else
            echo "Warning: Homebrew not found. Please install cairo and gobject-introspection manually."
        fi
    fi
fi

# Enable Metal acceleration for llama-cpp-python on Apple Silicon
if [[ "$OSTYPE" == "darwin"* ]]; then
    export CMAKE_ARGS="-DLLAMA_METAL=on"
fi

echo ""
echo "Installing Python package (core dependencies + LLM support)..."
if [ "$INSTALL_VIDEO" = true ]; then
    pip install -e '.[video]'
else
    pip install -e .
fi

echo ""
echo "Downloading base LLM model (SmolLM 1.7B - ~1.1GB)..."
echo "This is optimized for CPU and recommended for getting started."
echo ""

# Download the default LLM model
python -m maxim.models.download --llm smollm-1.7b-instruct

echo ""
echo "Model downloaded successfully!"

# Create default LLM config if it doesn't exist
LLM_CONFIG_PATH="data/util/llm.json"
if [ ! -f "$LLM_CONFIG_PATH" ]; then
    echo ""
    echo "Creating default LLM configuration..."
    mkdir -p "$(dirname "$LLM_CONFIG_PATH")"
    cat > "$LLM_CONFIG_PATH" << 'EOF'
{
  "enabled": true,
  "profile": "smollm-1.7b-instruct",
  "max_tokens": 512,
  "temperature": 0.0,
  "quantization": "Q4_K_M",
  "prompts_dir": "data/prompts",
  "profiles": {
    "smollm-1.7b-instruct": {
      "backend": "llama_cpp",
      "model_path": "data/models/LLM/SmolLM-1.7B-Instruct.Q4_K_M.gguf",
      "prompt_style": "chatml",
      "stop": ["<|im_end|>", "</s>"],
      "n_ctx": 4096
    },
    "mistral-7b-instruct-v0.2": {
      "backend": "llama_cpp",
      "model_path": "data/models/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
      "prompt_style": "mistral_instruct",
      "stop": ["</s>"],
      "n_ctx": 4096
    },
    "phi-3-mini-4k-instruct": {
      "backend": "llama_cpp",
      "model_path": "data/models/LLM/Phi-3-mini-4k-instruct.Q4_K_M.gguf",
      "prompt_style": "phi3",
      "stop": ["<|end|>", "<|endoftext|>"],
      "n_ctx": 4096
    }
  },
  "mode_response_config": {
    "observe": {
      "max_response_tokens": 128,
      "context_window_tokens": 512,
      "response_format": "minimal"
    },
    "sleep": {
      "max_response_tokens": 64,
      "context_window_tokens": 256,
      "response_format": "minimal"
    },
    "exploration": {
      "max_response_tokens": 256,
      "context_window_tokens": 1024,
      "response_format": "brief"
    },
    "live": {
      "max_response_tokens": 512,
      "context_window_tokens": 2048,
      "response_format": "conversational"
    },
    "active-assistance": {
      "max_response_tokens": 768,
      "context_window_tokens": 2048,
      "response_format": "detailed"
    },
    "reflection": {
      "max_response_tokens": 1024,
      "context_window_tokens": 3072,
      "response_format": "detailed"
    },
    "train": {
      "max_response_tokens": 512,
      "context_window_tokens": 2048,
      "response_format": "conversational"
    },
    "research": {
      "max_response_tokens": 2048,
      "context_window_tokens": 4096,
      "response_format": "academic"
    }
  }
}
EOF
    echo "Default LLM config created at $LLM_CONFIG_PATH"
    echo "You can customize this file to change models or settings."
else
    echo ""
    echo "Existing LLM config found at $LLM_CONFIG_PATH (preserved)"
fi

echo ""
echo "===== Installation Complete ====="
echo ""
echo "Installed components:"
if [ "$INSTALL_VIDEO" = true ]; then
    echo "  - FFmpeg system libraries"
    echo "  - Video processing support (PyAV)"
fi
echo "  - Maxim core dependencies"
echo "  - LLM support (llama-cpp-python)"
echo "  - SmolLM 1.7B model (~1.1GB)"
echo ""
echo "To get started, run:"
echo "  maxim"
echo ""
echo "The LLM is enabled by default and will run on CPU (slower but compatible with all GPUs)."
echo "For faster LLM inference, consider downloading a larger model with GPU support:"
echo "  python -m maxim.models.download --list"
echo "  python -m maxim.models.download --llm <model-name>"
echo ""
echo "For TTS support:"
echo "  pip install -e '.[tts]'"
echo "  python -m maxim.models.download --tts"
