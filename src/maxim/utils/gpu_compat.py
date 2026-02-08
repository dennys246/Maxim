"""GPU compatibility detection and workarounds.

Detects Blackwell GPUs (RTX 50 series) which have GStreamer/CUDA
incompatibilities and sets appropriate environment variables.

This module should be imported BEFORE any CUDA/GStreamer libraries
to ensure environment variables are set correctly.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass
class GPUCompatState:
    """GPU compatibility detection results."""

    blackwell_detected: bool = False
    original_cuda_devices: str | None = None


# Module-level state (set once at import time)
_compat_state: GPUCompatState | None = None


def detect_blackwell() -> GPUCompatState:
    """Detect Blackwell GPU and configure environment.

    Blackwell GPUs (RTX 50 series) have issues with NVENC/NVDEC in GStreamer
    that cause segfaults. This function detects them and sets environment
    variables to disable CUDA/GStreamer hardware acceleration.

    Returns:
        GPUCompatState with detection results.
    """
    global _compat_state
    if _compat_state is not None:
        return _compat_state

    _compat_state = GPUCompatState()

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            gpu_names = result.stdout.strip().lower()
            if "rtx 50" in gpu_names or "5080" in gpu_names or "5090" in gpu_names:
                _compat_state.blackwell_detected = True
                _compat_state.original_cuda_devices = os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "0"
                )

                # CRITICAL: Force CPU-only mode to avoid GStreamer/GLib segfaults
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ.setdefault("REACHY_MEDIA_BACKEND", "default")
                os.environ.setdefault("GST_CUDA_NO_CUDA", "1")
    except Exception:
        pass  # nvidia-smi not found, no NVIDIA GPU

    return _compat_state


def is_gpu_available() -> bool:
    """Check if any GPU (CUDA or MPS) is available.

    Returns:
        True if CUDA or MPS GPU is available, False otherwise.
    """
    try:
        import torch
    except ImportError:
        return False

    try:
        if torch.cuda.is_available():
            return True

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and getattr(mps, "is_available", None):
            return bool(mps.is_available())
    except Exception:
        return False

    return False


def is_blackwell_detected() -> bool:
    """Check if a Blackwell GPU was detected.

    Returns:
        True if Blackwell GPU was detected at startup.
    """
    if _compat_state is None:
        detect_blackwell()
    return _compat_state.blackwell_detected if _compat_state else False


def get_original_cuda_devices() -> str | None:
    """Get the original CUDA_VISIBLE_DEVICES before Blackwell workaround.

    This allows subprocesses (like Whisper) to use GPU even when main
    process has CUDA disabled.

    Returns:
        Original CUDA_VISIBLE_DEVICES value, or None if not modified.
    """
    if _compat_state is None:
        detect_blackwell()
    return _compat_state.original_cuda_devices if _compat_state else None


def env_flag(name: str, default: bool = False) -> bool:
    """Parse an environment variable as a boolean flag.

    Accepts: 1, true, t, yes, y, on for True
    Accepts: 0, false, f, no, n, off for False

    Args:
        name: Environment variable name.
        default: Default value if not set or unparseable.

    Returns:
        Boolean value.
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    value = str(raw).strip().lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def is_connection_error(error: object) -> bool:
    """Check if an error indicates a connection failure.

    Used to detect when robot connection needs to be re-established.

    Args:
        error: Exception or error message.

    Returns:
        True if error indicates connection failure.
    """
    if error is None:
        return False

    message = str(error).strip().lower()
    if not message:
        return False

    # Common connection error patterns
    if "lost connection" in message:
        return True
    if "disconnected" in message:
        return True
    if "timeout" in message or "timed out" in message:
        return True
    if "connection" in message and any(
        term in message for term in ("refused", "reset", "broken", "closed")
    ):
        return True

    # Dynamixel/serial communication errors (rustypot panics)
    if "channel closed" in message:
        return True
    if "panicexception" in message:
        return True
    if "assertion failed" in message and "buffer" in message:
        return True
    if "flush serial" in message:
        return True

    return False


__all__ = [
    "GPUCompatState",
    "detect_blackwell",
    "is_gpu_available",
    "is_blackwell_detected",
    "get_original_cuda_devices",
    "env_flag",
    "is_connection_error",
]
