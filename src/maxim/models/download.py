"""Model download utilities for Maxim.

Provides easy downloading of LLM and TTS models from HuggingFace.

Usage:
    # Download default LLM model
    python -m maxim.models.download --llm

    # Download specific LLM model
    python -m maxim.models.download --llm smollm-1.7b-instruct

    # Download TTS model
    python -m maxim.models.download --tts

    # Download all models
    python -m maxim.models.download --all

    # List available models
    python -m maxim.models.download --list
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve
from urllib.error import URLError

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

LLM_MODELS: dict[str, dict[str, Any]] = {
    "smollm-1.7b-instruct": {
        "description": "SmolLM 1.7B - Small, fast, good for CPU (recommended for limited hardware)",
        "size_gb": 1.1,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/QuantFactory/SmolLM-1.7B-Instruct-GGUF/resolve/main/SmolLM-1.7B-Instruct.Q4_K_M.gguf",
        "filename": "SmolLM-1.7B-Instruct.Q4_K_M.gguf",
    },
    "smollm2-1.7b-instruct": {
        "description": "SmolLM2 1.7B - Improved SmolLM, small and efficient",
        "size_gb": 1.0,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "filename": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    },
    "phi-2": {
        "description": "Microsoft Phi-2 2.7B - Compact but capable",
        "size_gb": 1.8,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "filename": "phi-2.Q4_K_M.gguf",
    },
    "gemma-2-2b-it": {
        "description": "Google Gemma 2 2B Instruct - Small and efficient",
        "size_gb": 1.6,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
    },
    "mistral-7b-instruct-v0.2": {
        "description": "Mistral 7B Instruct v0.2 - High quality, needs more RAM",
        "size_gb": 4.4,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    },
    "llama-3-8b-instruct": {
        "description": "Meta Llama 3 8B Instruct - Excellent quality, needs GPU or lots of RAM",
        "size_gb": 4.9,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    },
    "qwen2-7b-instruct": {
        "description": "Alibaba Qwen2 7B Instruct - Strong multilingual support",
        "size_gb": 4.4,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf",
        "filename": "qwen2-7b-instruct-q4_k_m.gguf",
    },
    "qwen2.5-14b-instruct": {
        "description": "Alibaba Qwen2.5 14B Instruct - Excellent instruction following, 32K context",
        "size_gb": 8.5,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "filename": "Qwen2.5-14B-Instruct.Q4_K_M.gguf",
    },
}

TTS_MODELS: dict[str, dict[str, Any]] = {
    "en_US-lessac-medium": {
        "description": "English US - Lessac voice (medium quality, recommended)",
        "size_mb": 75,
        "files": [
            {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
                "filename": "en_US-lessac-medium.onnx",
            },
            {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
                "filename": "en_US-lessac-medium.onnx.json",
            },
        ],
    },
    "en_US-amy-medium": {
        "description": "English US - Amy voice (female, medium quality)",
        "size_mb": 75,
        "files": [
            {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
                "filename": "en_US-amy-medium.onnx",
            },
            {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
                "filename": "en_US-amy-medium.onnx.json",
            },
        ],
    },
    "en_GB-alan-medium": {
        "description": "English UK - Alan voice (British male)",
        "size_mb": 75,
        "files": [
            {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
                "filename": "en_GB-alan-medium.onnx",
            },
            {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
                "filename": "en_GB-alan-medium.onnx.json",
            },
        ],
    },
}

VISION_MODELS: dict[str, dict[str, Any]] = {
    "rtmdet-m": {
        "description": "RTMDet-m - 80-class COCO object detection (Apache 2.0, ~49.4 mAP)",
        "size_mb": 100,
        "url": "https://huggingface.co/ziq/rtm/resolve/main/rtmdet-m.onnx",
        "filename": "rtmdet-m.onnx",
    },
    "rtmpose-m": {
        "description": "RTMPose-m - 17-keypoint COCO pose estimation (Apache 2.0, ~75.8 AP)",
        "size_mb": 55,
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        "filename": "rtmpose-m.onnx",
        "archive": True,  # .zip containing .onnx file
    },
}

# Default models
DEFAULT_LLM = "smollm-1.7b-instruct"
DEFAULT_TTS = "en_US-lessac-medium"
DEFAULT_VISION = "rtmdet-m"

# Default paths
DEFAULT_LLM_DIR = "data/models/LLM"
DEFAULT_TTS_DIR = "data/models/tts"
DEFAULT_VISION_DIR = "data/models/YOLO"


# ─────────────────────────────────────────────────────────────────────────────
# Download Functions
# ─────────────────────────────────────────────────────────────────────────────


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Progress callback for urlretrieve."""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        bar_length = 40
        filled = int(bar_length * percent // 100)
        bar = "=" * filled + "-" * (bar_length - filled)
        sys.stdout.write(f"\r  [{bar}] {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()
        if downloaded >= total_size:
            sys.stdout.write("\n")
            sys.stdout.flush()


def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file with progress indicator.

    Args:
        url: URL to download from.
        dest_path: Destination path.
        desc: Description for logging.

    Returns:
        True if download succeeded, False otherwise.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  Already exists: {dest_path}")
        return True

    print(f"  Downloading: {desc or dest_path.name}")
    print(f"  From: {url}")

    try:
        urlretrieve(url, dest_path, reporthook=_progress_hook)
        print(f"  Saved to: {dest_path}")
        return True
    except URLError as e:
        print(f"  Download failed: {e}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        # Clean up partial download
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_llm(
    model_name: str = DEFAULT_LLM,
    models_dir: str | Path = DEFAULT_LLM_DIR,
) -> bool:
    """Download an LLM model.

    Args:
        model_name: Name of the model to download.
        models_dir: Directory to save the model.

    Returns:
        True if download succeeded, False otherwise.
    """
    if model_name not in LLM_MODELS:
        print(f"Unknown LLM model: {model_name}")
        print(f"Available models: {', '.join(LLM_MODELS.keys())}")
        return False

    model_info = LLM_MODELS[model_name]
    models_dir = Path(models_dir)

    print(f"\nDownloading LLM: {model_name}")
    print(f"  {model_info['description']}")
    print(f"  Size: ~{model_info['size_gb']} GB")

    dest_path = models_dir / model_info["filename"]
    return download_file(model_info["url"], dest_path, model_info["filename"])


def download_tts(
    model_name: str = DEFAULT_TTS,
    models_dir: str | Path = DEFAULT_TTS_DIR,
) -> bool:
    """Download a TTS model.

    Args:
        model_name: Name of the model to download.
        models_dir: Directory to save the model.

    Returns:
        True if all files downloaded successfully, False otherwise.
    """
    if model_name not in TTS_MODELS:
        print(f"Unknown TTS model: {model_name}")
        print(f"Available models: {', '.join(TTS_MODELS.keys())}")
        return False

    model_info = TTS_MODELS[model_name]
    models_dir = Path(models_dir)

    print(f"\nDownloading TTS: {model_name}")
    print(f"  {model_info['description']}")
    print(f"  Size: ~{model_info['size_mb']} MB")

    success = True
    for file_info in model_info["files"]:
        dest_path = models_dir / file_info["filename"]
        if not download_file(file_info["url"], dest_path, file_info["filename"]):
            success = False

    return success


def download_vision(
    model_name: str = DEFAULT_VISION,
    models_dir: str | Path = DEFAULT_VISION_DIR,
) -> bool:
    """Download a vision model (RTMDet or RTMPose ONNX).

    Args:
        model_name: Name of the model to download.
        models_dir: Directory to save the model.

    Returns:
        True if download succeeded, False otherwise.
    """
    if model_name not in VISION_MODELS:
        print(f"Unknown vision model: {model_name}")
        print(f"Available models: {', '.join(VISION_MODELS.keys())}")
        return False

    model_info = VISION_MODELS[model_name]
    models_dir = Path(models_dir)

    print(f"\nDownloading Vision: {model_name}")
    print(f"  {model_info['description']}")
    print(f"  Size: ~{model_info['size_mb']} MB")

    if model_info.get("archive"):
        # Download zip, extract the .onnx file
        import tempfile
        import zipfile

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        # Remove any stale temp file so download_file doesn't skip it
        tmp_path.unlink(missing_ok=True)

        if not download_file(model_info["url"], tmp_path, f"{model_name} (archive)"):
            tmp_path.unlink(missing_ok=True)
            return False

        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                # Find the .onnx file inside
                onnx_files = [n for n in zf.namelist() if n.endswith(".onnx")]
                if not onnx_files:
                    print("  No .onnx file found in archive")
                    return False
                for onnx_name in onnx_files:
                    dest = models_dir / model_info["filename"]
                    with zf.open(onnx_name) as src, open(dest, "wb") as dst:
                        import shutil

                        shutil.copyfileobj(src, dst)
                    print(f"  Extracted: {dest}")
            return True
        except Exception as e:
            print(f"  Extract failed: {e}")
            return False
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        dest_path = models_dir / model_info["filename"]
        return download_file(model_info["url"], dest_path, model_info["filename"])


def list_models() -> None:
    """Print available models."""
    print("\n=== Available LLM Models ===")
    print(f"(Default: {DEFAULT_LLM})\n")

    for name, info in LLM_MODELS.items():
        marker = " [recommended]" if name == DEFAULT_LLM else ""
        print(f"  {name}{marker}")
        print(f"    {info['description']}")
        print(f"    Size: ~{info['size_gb']} GB")
        print()

    print("\n=== Available TTS Models ===")
    print(f"(Default: {DEFAULT_TTS})\n")

    for name, info in TTS_MODELS.items():
        marker = " [recommended]" if name == DEFAULT_TTS else ""
        print(f"  {name}{marker}")
        print(f"    {info['description']}")
        print(f"    Size: ~{info['size_mb']} MB")
        print()

    print("\n=== Available Vision Models ===")
    print("(Default: all)\n")

    for name, info in VISION_MODELS.items():
        print(f"  {name}")
        print(f"    {info['description']}")
        print(f"    Size: ~{info['size_mb']} MB")
        print()


def check_models(
    llm_dir: str | Path = DEFAULT_LLM_DIR,
    tts_dir: str | Path = DEFAULT_TTS_DIR,
    vision_dir: str | Path = DEFAULT_VISION_DIR,
) -> dict[str, bool]:
    """Check which models are already downloaded.

    Returns:
        Dict mapping model names to whether they exist.
    """
    llm_dir = Path(llm_dir)
    tts_dir = Path(tts_dir)
    vision_dir = Path(vision_dir)

    status = {}

    for name, info in LLM_MODELS.items():
        path = llm_dir / info["filename"]
        status[f"llm:{name}"] = path.exists()

    for name, info in TTS_MODELS.items():
        # TTS model is complete if all files exist
        all_exist = all((tts_dir / f["filename"]).exists() for f in info["files"])
        status[f"tts:{name}"] = all_exist

    for name, info in VISION_MODELS.items():
        path = vision_dir / info["filename"]
        status[f"vision:{name}"] = path.exists()

    return status


def print_status(
    llm_dir: str | Path = DEFAULT_LLM_DIR,
    tts_dir: str | Path = DEFAULT_TTS_DIR,
    vision_dir: str | Path = DEFAULT_VISION_DIR,
) -> None:
    """Print status of downloaded models."""
    status = check_models(llm_dir, tts_dir, vision_dir)

    print("\n=== Model Status ===\n")

    print("LLM Models:")
    for name in LLM_MODELS:
        key = f"llm:{name}"
        icon = "[x]" if status.get(key) else "[ ]"
        print(f"  {icon} {name}")

    print("\nTTS Models:")
    for name in TTS_MODELS:
        key = f"tts:{name}"
        icon = "[x]" if status.get(key) else "[ ]"
        print(f"  {icon} {name}")

    print("\nVision Models:")
    for name in VISION_MODELS:
        key = f"vision:{name}"
        icon = "[x]" if status.get(key) else "[ ]"
        print(f"  {icon} {name}")

    print()


def enable_llm_config(
    model_name: str = DEFAULT_LLM,
    config_path: str | Path = "data/util/llm.json",
) -> bool:
    """Enable LLM in config and set the model.

    Args:
        model_name: Model to set as active.
        config_path: Path to llm.json config.

    Returns:
        True if config was updated, False otherwise.
    """
    import json

    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)

        config["enabled"] = True
        config["profile"] = model_name

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Updated {config_path}:")
        print("  enabled: true")
        print(f"  profile: {model_name}")
        return True

    except Exception as e:
        print(f"Failed to update config: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download models for Maxim (LLM, TTS, and Vision)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --llm                      # Download default LLM (smollm-1.7b)
  %(prog)s --llm mistral-7b-instruct-v0.2  # Download Mistral 7B
  %(prog)s --tts                      # Download default TTS voice
  %(prog)s --vision                   # Download vision models (RTMDet + RTMPose)
  %(prog)s --all                      # Download default LLM + TTS + Vision
  %(prog)s --list                     # List available models
  %(prog)s --status                   # Show downloaded models
  %(prog)s --enable                   # Enable LLM in config
""",
    )

    parser.add_argument(
        "--llm",
        nargs="?",
        const=DEFAULT_LLM,
        metavar="MODEL",
        help=f"Download LLM model (default: {DEFAULT_LLM})",
    )
    parser.add_argument(
        "--tts",
        nargs="?",
        const=DEFAULT_TTS,
        metavar="MODEL",
        help=f"Download TTS model (default: {DEFAULT_TTS})",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Download vision models (RTMDet + RTMPose ONNX, Apache 2.0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download default LLM, TTS, and Vision models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status of downloaded models",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Enable LLM in config after download",
    )
    parser.add_argument(
        "--llm-dir",
        type=str,
        default=DEFAULT_LLM_DIR,
        help=f"LLM models directory (default: {DEFAULT_LLM_DIR})",
    )
    parser.add_argument(
        "--tts-dir",
        type=str,
        default=DEFAULT_TTS_DIR,
        help=f"TTS models directory (default: {DEFAULT_TTS_DIR})",
    )
    parser.add_argument(
        "--vision-dir",
        type=str,
        default=DEFAULT_VISION_DIR,
        help=f"Vision models directory (default: {DEFAULT_VISION_DIR})",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_models()
        return 0

    # Handle --status
    if args.status:
        print_status(args.llm_dir, args.tts_dir, args.vision_dir)
        return 0

    # Handle --all
    if args.all:
        args.llm = DEFAULT_LLM
        args.tts = DEFAULT_TTS
        args.vision = True

    # Track what was downloaded
    downloaded_llm = None

    # Download LLM
    if args.llm:
        if download_llm(args.llm, args.llm_dir):
            downloaded_llm = args.llm
            print(f"\nLLM model ready: {args.llm}")
        else:
            print(f"\nFailed to download LLM: {args.llm}")
            return 1

    # Download TTS
    if args.tts:
        if download_tts(args.tts, args.tts_dir):
            print(f"\nTTS model ready: {args.tts}")
        else:
            print(f"\nFailed to download TTS: {args.tts}")
            return 1

    # Download Vision models
    if args.vision:
        for vname in VISION_MODELS:
            if download_vision(vname, args.vision_dir):
                print(f"\nVision model ready: {vname}")
            else:
                print(f"\nFailed to download vision model: {vname}")
                return 1

    # Enable LLM in config if requested
    if args.enable and downloaded_llm:
        print()
        enable_llm_config(downloaded_llm)

    # Show help if no action specified
    if not (args.llm or args.tts or args.vision or args.all or args.list or args.status):
        parser.print_help()
        return 0

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
