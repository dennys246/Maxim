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
        echo "Installing system dependencies (FFmpeg)..."
        brew install ffmpeg pkg-config

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
fi

echo ""
echo "Installing Python package (core dependencies)..."
pip install -e .

echo ""
echo "Installing LLM support..."
if [ "$INSTALL_VIDEO" = true ]; then
    pip install -e '.[llm,video]'
else
    pip install -e '.[llm]'
fi

echo ""
echo "Downloading base LLM model (SmolLM 1.7B - ~1.1GB)..."
echo "This is optimized for CPU and recommended for getting started."
echo ""

# Check if download script exists
if [ -f "scripts/download_models.sh" ]; then
    chmod +x scripts/download_models.sh
    ./scripts/download_models.sh --llm smollm-1.7b-instruct --enable
else
    echo "Warning: Model download script not found at scripts/download_models.sh"
    echo "You'll need to download models manually."
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
echo "  - SmolLM 1.7B base model (~1.1GB)"
echo ""
echo "To get started, run:"
echo "  maxim"
echo ""
echo "To use the LLM:"
echo "  export MAXIM_LLM_ENABLED=1"
echo "  maxim --language-model smollm-1.7b"
echo ""
echo "For more models, run:"
echo "  ./scripts/download_models.sh --list"
echo ""
echo "For video processing (if not installed):"
echo "  pip install -e '.[video]'"
echo ""
echo "For TTS support:"
echo "  pip install -e '.[tts]'"
echo "  ./scripts/download_models.sh --tts --enable"
