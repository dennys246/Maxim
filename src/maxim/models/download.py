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
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

# expected_bytes is the authoritative size for download verification. When set,
# downloaded files must match exactly or download_file treats them as corrupted
# and deletes them. Leave None for profiles where the upstream size has not
# been verified yet; the download path skips the size check in that case.
LLM_MODELS: dict[str, dict[str, Any]] = {
    "smollm-135m-instruct": {
        "description": (
            "SmolLM 135M - Tiniest LLM in the registry (~90 MB). "
            "Great for CI smoke tests, constrained hardware, and download "
            "pipeline verification."
        ),
        "size_gb": 0.09,
        # Pinned Q4_K_M from QuantFactory. expected_bytes left None because
        # the upstream has not been byte-verified yet; the download path
        # falls through on the size check and relies on streamed integrity.
        "expected_bytes": None,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/QuantFactory/SmolLM-135M-Instruct-GGUF/resolve/main/SmolLM-135M-Instruct.Q4_K_M.gguf",
        "filename": "SmolLM-135M-Instruct.Q4_K_M.gguf",
    },
    "smollm-1.7b-instruct": {
        "description": "SmolLM 1.7B - Small, fast, good for CPU (recommended for limited hardware)",
        "size_gb": 1.1,
        "expected_bytes": 1055609344,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/QuantFactory/SmolLM-1.7B-Instruct-GGUF/resolve/main/SmolLM-1.7B-Instruct.Q4_K_M.gguf",
        "filename": "SmolLM-1.7B-Instruct.Q4_K_M.gguf",
    },
    "smollm2-1.7b-instruct": {
        "description": "SmolLM2 1.7B - Improved SmolLM, small and efficient",
        "size_gb": 1.0,
        "expected_bytes": None,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "filename": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    },
    "phi-2": {
        "description": "Microsoft Phi-2 2.7B - Compact but capable",
        "size_gb": 1.8,
        "expected_bytes": 1789239136,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "filename": "phi-2.Q4_K_M.gguf",
    },
    "gemma-2-2b-it": {
        "description": "Google Gemma 2 2B Instruct - Small and efficient",
        "size_gb": 1.6,
        "expected_bytes": 1708582752,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
    },
    "mistral-7b-instruct-v0.2": {
        "description": "Mistral 7B Instruct v0.2 - High quality, needs more RAM",
        "size_gb": 4.4,
        "expected_bytes": 4368439584,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    },
    "llama-3-8b-instruct": {
        "description": "Meta Llama 3 8B Instruct - Excellent quality, needs GPU or lots of RAM",
        "size_gb": 4.9,
        "expected_bytes": 4920734272,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    },
    "qwen2-7b-instruct": {
        "description": "Alibaba Qwen2 7B Instruct - Strong multilingual support",
        "size_gb": 4.4,
        "expected_bytes": 4683071264,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf",
        "filename": "qwen2-7b-instruct-q4_k_m.gguf",
    },
    "qwen2.5-14b-instruct": {
        "description": "Alibaba Qwen2.5 14B Instruct - Excellent instruction following, 32K context",
        "size_gb": 8.5,
        "expected_bytes": 8988110976,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "filename": "Qwen2.5-14B-Instruct.Q4_K_M.gguf",
    },
    "qwen2.5-32b-instruct": {
        "description": (
            "Alibaba Qwen2.5 32B Instruct - The leader-grade default for "
            "48 GB+ Apple Silicon or 24 GB+ VRAM. 32K context."
        ),
        "size_gb": 19.9,
        # expected_bytes left None — upstream size not byte-verified yet;
        # the download path falls through on the size check and relies
        # on streamed integrity, same as the smollm-135m entry above.
        "expected_bytes": None,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "filename": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
    },
    "llama-3.1-70b-instruct": {
        "description": (
            "Meta Llama 3.1 70B Instruct - True large-model territory. "
            "Comfortable on 64 GB+ unified memory; borderline on 48 GB."
        ),
        "size_gb": 42.5,
        "expected_bytes": None,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        "filename": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
    },
    "mixtral-8x7b-instruct": {
        "description": (
            "Mistral Mixtral-8x7B Instruct v0.1 - MoE (8 experts, top-2). "
            "All experts must reside in memory; ~26 GB Q4_K_M weights."
        ),
        "size_gb": 26.4,
        "expected_bytes": None,
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf",
        "filename": "Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf",
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

# Default paths — resolved lazily via helpers below
_DEFAULT_LLM_DIR: str | None = None
_DEFAULT_TTS_DIR: str | None = None
_DEFAULT_VISION_DIR: str | None = None


def _llm_dir() -> str:
    global _DEFAULT_LLM_DIR
    if _DEFAULT_LLM_DIR is None:
        from maxim.utils.paths import model_dir

        _DEFAULT_LLM_DIR = str(model_dir() / "LLM")
    return _DEFAULT_LLM_DIR


def _tts_dir() -> str:
    global _DEFAULT_TTS_DIR
    if _DEFAULT_TTS_DIR is None:
        from maxim.utils.paths import model_dir

        _DEFAULT_TTS_DIR = str(model_dir() / "tts")
    return _DEFAULT_TTS_DIR


def _vision_dir() -> str:
    global _DEFAULT_VISION_DIR
    if _DEFAULT_VISION_DIR is None:
        from maxim.utils.paths import model_dir

        _DEFAULT_VISION_DIR = str(model_dir() / "YOLO")
    return _DEFAULT_VISION_DIR


# Backward-compatible module-level names (lazy property via __getattr__)
def __getattr__(name: str) -> str:
    if name == "DEFAULT_LLM_DIR":
        return _llm_dir()
    if name == "DEFAULT_TTS_DIR":
        return _tts_dir()
    if name == "DEFAULT_VISION_DIR":
        return _vision_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def download_file(
    url: str,
    dest_path: Path,
    desc: str = "",
    *,
    expected_bytes: int | None = None,
) -> bool:
    """Download a file atomically with progress indicator.

    Writes to ``{dest_path}.partial``, verifies size if ``expected_bytes``
    is provided, then ``os.replace()`` atomically to ``dest_path``. A
    partial file is cleaned up on ANY failure path, including
    ``KeyboardInterrupt`` — the previous implementation left a corrupt
    file at the final name on ``URLError`` and on Ctrl+C, which then
    passed ``profile_has_local_file`` on subsequent runs and crashed the
    loader in a retry loop.

    Args:
        url: URL to download from.
        dest_path: Final destination path (the atomic rename target).
        desc: Optional description used in the progress print.
        expected_bytes: If provided, the downloaded file must match this
            byte count exactly or the download is rejected. Sourced from
            ``LLM_MODELS[<profile>]["expected_bytes"]``. ``None`` skips
            the size check (legacy profiles that have not been verified
            against HuggingFace yet).

    Returns:
        True if the file exists at ``dest_path`` after the call (either
        newly downloaded or already present). False on any failure, with
        no residual ``.partial`` file left behind.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  Already exists: {dest_path}")
        return True

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".partial")
    # Clean up any stale partial from a prior crashed run before starting.
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError as e:
            print(f"  Warning: could not remove stale partial {tmp_path.name}: {e}")

    print(f"  Downloading: {desc or dest_path.name}")
    print(f"  From: {url}")

    from maxim.utils import http as _http

    try:
        _http.download_to_file(
            url,
            tmp_path,
            progress_hook=_progress_hook,
            # Large models can take 10+ minutes over slow links — generous budget.
            timeout=_http.TimeoutPolicy(connect_s=10.0, read_s=300.0, total_s=3600.0),
        )
    except BaseException as e:
        # BaseException catches KeyboardInterrupt (which Exception doesn't)
        # so Ctrl+C mid-download cleans up before exiting. Re-raise after
        # cleanup so the caller's interrupt handling still fires.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        if isinstance(e, KeyboardInterrupt):
            print("  Download interrupted — cleaned up partial file")
            raise
        if isinstance(e, _http.HTTPError):
            print(f"  Download failed: {e.fix_hint}")
            return False
        print(f"  Error: {e}")
        return False

    # Verify size before the atomic rename. A truncated download (network
    # drop that urlretrieve didn't notice, HF returning a partial response,
    # etc.) would otherwise land at the final path and pass future
    # profile_has_local_file checks — leading to cryptic load failures.
    if expected_bytes is not None:
        actual_bytes = tmp_path.stat().st_size
        if actual_bytes != expected_bytes:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            print(f"  Download size mismatch: got {actual_bytes} bytes, expected {expected_bytes}. File rejected.")
            return False

    try:
        import os as _os

        _os.replace(tmp_path, dest_path)
    except OSError as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        print(f"  Rename failed: {e}")
        return False

    print(f"  Saved to: {dest_path}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Auto-download (P5: peer_leader_flexibility_plan)
# ─────────────────────────────────────────────────────────────────────────────


def _prompt_yes_no_with_timeout(question: str, timeout_s: float = 30.0) -> bool | None:
    """Prompt the user for a yes/no answer with a wall-clock timeout.

    Returns True/False on a real answer, None on timeout. Used by
    :func:`ensure_available` so an unattended terminal doesn't deadlock
    a Maxim startup waiting for input that will never come.

    POSIX uses ``select`` on stdin. Windows uses a thread + Event so the
    main thread can be interrupted by Ctrl+C without ``msvcrt`` getting
    in the way.
    """
    import sys

    print(question, end="", flush=True)
    if sys.platform == "win32":
        import threading

        result: dict[str, str] = {}
        done = threading.Event()

        def _reader() -> None:
            try:
                result["line"] = sys.stdin.readline()
            except Exception:
                result["line"] = ""
            finally:
                done.set()

        threading.Thread(target=_reader, daemon=True, name="download-prompt").start()
        if not done.wait(timeout=timeout_s):
            print(" [timeout]")
            return None
        response = (result.get("line") or "").strip().lower()
    else:
        import select

        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if not ready:
            print(" [timeout]")
            return None
        response = sys.stdin.readline().strip().lower()
    return response in ("y", "yes")


def _auto_download_enabled() -> bool:
    """Whether auto-download is enabled via env (set by --auto-download)."""
    raw = os.environ.get("MAXIM_AUTO_DOWNLOAD_MODELS", "").strip().lower()
    return raw in ("1", "true", "t", "yes", "y", "on")


def _soft_budget_gb() -> float | None:
    """Read MAXIM_DATA_BUDGET_GB env var, return None if unset / invalid."""
    raw = os.environ.get("MAXIM_DATA_BUDGET_GB", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
        return value if value > 0 else None
    except ValueError:
        return None


def ensure_available(
    profile_name: str,
    *,
    auto: bool | None = None,
    interactive: bool | None = None,
    logger: Any | None = None,
) -> bool:
    """Make sure the GGUF for ``profile_name`` is on disk; download if not.

    The single entry point used by ``build_primary_router`` (P5) and any
    other code path that needs to guarantee a profile's local file exists
    before it tries to load it. Composes the F0.1–F0.5 building blocks:
    storage preflight, advisory file lock, atomic download with size
    verification.

    Args:
        profile_name: Profile key from ``LLM_MODELS``.
        auto: Override the env-derived auto-download flag. ``True`` skips
            the prompt unconditionally; ``None`` (default) reads
            ``MAXIM_AUTO_DOWNLOAD_MODELS`` via :func:`_auto_download_enabled`.
        interactive: Override the tty check. ``None`` (default) checks
            ``sys.stdin.isatty()``.
        logger: Logger for warnings; ``print`` for user-facing messages.

    Returns:
        True iff the file is on disk after this call (already-present or
        newly downloaded). False on any failure — caller should fall
        through to the next-smaller profile via tier re-walk.
    """
    import sys

    from maxim.runtime.llm_server import profile_has_local_file
    from maxim.utils.process_lock import LockContended, file_lock
    from maxim.utils.paths import data_home
    from maxim.utils.storage import can_download, format_report, report_storage

    if profile_has_local_file(profile_name):
        return True

    if profile_name not in LLM_MODELS:
        msg = (
            f"Auto-download skipped: profile '{profile_name}' is not in the LLM_MODELS "
            f"registry. Custom profiles must be downloaded manually — point "
            f"~/.maxim/config/llm.json at an existing GGUF file."
        )
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)
        return False

    info = LLM_MODELS[profile_name]
    # User profiles added via ``maxim model add`` don't have a
    # known size_gb; profile_loader.py injects ``size_gb: None`` for
    # them. ``.get(key, default)`` only returns the default when the
    # key is ABSENT, not when the value is None — so guard against
    # both. 0.0 disables the soft-budget pre-check (acceptable for
    # user profiles where the operator chose the GGUF deliberately).
    raw_size = info.get("size_gb")
    size_gb = float(raw_size) if raw_size is not None else 0.0

    ok, reason = can_download(
        size_gb,
        headroom_gb=2.0,
        soft_budget_gb=_soft_budget_gb(),
    )
    if not ok:
        report = report_storage()
        print(f"\n  Cannot download '{profile_name}' ({size_gb:.1f} GB): {reason}\n")
        print(format_report(report))
        print("\n  Free up space (delete unused models with `maxim --delete-model NAME`) or")
        print("  raise MAXIM_DATA_BUDGET_GB and retry.\n")
        return False

    auto_flag = _auto_download_enabled() if auto is None else bool(auto)
    is_tty = (sys.stdin.isatty() if interactive is None else bool(interactive)) if sys.stdin else False

    if not auto_flag:
        if is_tty:
            question = (
                f"\n  Maxim wants to download '{profile_name}' (~{size_gb:.1f} GB) "
                f"from HuggingFace.\n  Disk: {reason}\n  Proceed? [y/N] (30s timeout) "
            )
            answer = _prompt_yes_no_with_timeout(question, timeout_s=30.0)
            if answer is None:
                print(
                    f"\n  Auto-download cancelled (timeout). Run with --auto-download "
                    f"to skip the prompt, or download manually:\n"
                    f"    python -m maxim.models.download --llm {profile_name}\n"
                )
                return False
            if not answer:
                print(
                    f"\n  Auto-download declined. Run manually when ready:\n"
                    f"    python -m maxim.models.download --llm {profile_name}\n"
                )
                return False
        else:
            print(
                f"\n  Profile '{profile_name}' is not downloaded and stdin is not a tty.\n"
                f"  Either pass --auto-download (or set MAXIM_AUTO_DOWNLOAD_MODELS=1)\n"
                f"  or run manually:\n"
                f"    python -m maxim.models.download --llm {profile_name}\n"
            )
            return False

    # Acquire the download lock so two concurrent `maxim` invocations don't
    # race on the same target file.
    lock_path = data_home() / "util" / "download.lock"
    try:
        with file_lock(lock_path):
            # Re-check inside the lock — another process may have just
            # finished downloading the same profile while we were prompting.
            if profile_has_local_file(profile_name):
                return True
            return download_llm(profile_name)
    except LockContended:
        print(
            f"\n  Another maxim process is downloading models. Wait for it to finish, "
            f"or kill it and retry.\n  (lock file: {lock_path})\n"
        )
        return False


def download_llm(
    model_name: str = DEFAULT_LLM,
    models_dir: str | Path | None = None,
) -> bool:
    """Download an LLM model.

    Args:
        model_name: Name of the model to download.
        models_dir: Directory to save the model.

    Returns:
        True if download succeeded, False otherwise.
    """
    if models_dir is None:
        models_dir = _llm_dir()
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
    return download_file(
        model_info["url"],
        dest_path,
        model_info["filename"],
        expected_bytes=model_info.get("expected_bytes"),
    )


def delete_llm(
    model_name: str,
    models_dir: str | Path | None = None,
) -> bool:
    """Delete a downloaded LLM model to free disk space.

    Args:
        model_name: Name of the model to delete.
        models_dir: Directory where models are stored.

    Returns:
        True if the file was deleted, False if not found or on error.
    """
    if models_dir is None:
        models_dir = _llm_dir()
    models_dir = Path(models_dir)

    # Try the download registry first (exact filename)
    if model_name in LLM_MODELS:
        dest_path = models_dir / LLM_MODELS[model_name]["filename"]
        if dest_path.is_file():
            size_gb = dest_path.stat().st_size / (1024**3)
            dest_path.unlink()
            print(f"Deleted {model_name}: {dest_path.name} ({size_gb:.1f} GB freed)")
            return True

    # Fallback: try resolving via build_model_path (handles case-insensitive matching)
    try:
        from maxim.models.language.config import load_llm_config

        cfg = load_llm_config(profile_override=model_name)
        model_path = Path(getattr(cfg, "model_path", "") or "")
        if model_path.is_file():
            size_gb = model_path.stat().st_size / (1024**3)
            model_path.unlink()
            print(f"Deleted {model_name}: {model_path.name} ({size_gb:.1f} GB freed)")
            return True
    except Exception:
        pass

    print(f"Model not found on disk: {model_name}")
    return False


def download_tts(
    model_name: str = DEFAULT_TTS,
    models_dir: str | Path | None = None,
) -> bool:
    """Download a TTS model.

    Args:
        model_name: Name of the model to download.
        models_dir: Directory to save the model.

    Returns:
        True if all files downloaded successfully, False otherwise.
    """
    if models_dir is None:
        models_dir = _tts_dir()
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
    models_dir: str | Path | None = None,
) -> bool:
    """Download a vision model (RTMDet or RTMPose ONNX).

    Args:
        model_name: Name of the model to download.
        models_dir: Directory to save the model.

    Returns:
        True if download succeeded, False otherwise.
    """
    if models_dir is None:
        models_dir = _vision_dir()
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
    llm_dir: str | Path | None = None,
    tts_dir: str | Path | None = None,
    vision_dir: str | Path | None = None,
) -> dict[str, bool]:
    """Check which models are already downloaded.

    Returns:
        Dict mapping model names to whether they exist.
    """
    llm_dir = Path(llm_dir or _llm_dir())
    tts_dir = Path(tts_dir or _tts_dir())
    vision_dir = Path(vision_dir or _vision_dir())

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
    llm_dir: str | Path | None = None,
    tts_dir: str | Path | None = None,
    vision_dir: str | Path | None = None,
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
    config_path: str | Path | None = None,
) -> bool:
    """Enable LLM in config and set the model.

    Args:
        model_name: Model to set as active.
        config_path: Path to llm.json config.

    Returns:
        True if config was updated, False otherwise.
    """
    import json

    if config_path is None:
        # ALWAYS write to the user config (2026-08-03 fix): resolve_config
        # falls back to the BUNDLED template path when ~/.maxim/config/
        # llm.json is absent, so the pre-fix code would open package-shipped
        # data for writing (PermissionError on an installed wheel, or worse,
        # silent mutation of bundled defaults). The template is read-only
        # SEED content; the user config is the write target.
        from maxim.utils.paths import resolve_config, user_config

        config_path = user_config() / "llm.json"
        if not config_path.exists():
            try:
                template = resolve_config("llm.json")
                if Path(template) != config_path and Path(template).exists():
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    config_path.write_text(Path(template).read_text(encoding="utf-8"), encoding="utf-8")
            except FileNotFoundError:
                pass
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
        default=None,
        help="LLM models directory (default: ~/.maxim/models/LLM)",
    )
    parser.add_argument(
        "--tts-dir",
        type=str,
        default=None,
        help="TTS models directory (default: ~/.maxim/models/tts)",
    )
    parser.add_argument(
        "--vision-dir",
        type=str,
        default=None,
        help="Vision models directory (default: ~/.maxim/models/YOLO)",
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
