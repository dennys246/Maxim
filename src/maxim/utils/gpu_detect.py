"""Blackwell GPU detection and GStreamer guards.

Detects RTX 50-series (Blackwell) GPUs and applies GStreamer CUDA guards
to prevent reachy-mini media-pipeline segfaults.  Results are cached in
``_MAXIM_BLACKWELL_CHECKED`` env var for the shell session.

This module is imported lazily — call :func:`ensure_blackwell_guards` from
``cli.main()`` or wherever GPU detection is needed.  It does NOT run at
import time.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)

_BLACKWELL_CACHE_ENV = "_MAXIM_BLACKWELL_CHECKED"

_detected: bool | None = None  # None = not yet checked
_detected_lock = threading.Lock()


def _is_hide_cuda() -> bool:
    return os.environ.get("MAXIM_BLACKWELL_HIDE_CUDA", "").strip().lower() in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    )


def _apply_guards(*, hide_cuda: bool) -> None:
    """Apply GStreamer guards, and optionally hide CUDA from all libraries."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["GST_CUDA_NO_CUDA"] = "1"
    os.environ.setdefault("REACHY_MEDIA_BACKEND", "default")
    if hide_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def ensure_blackwell_guards() -> bool:
    """Detect Blackwell GPU and apply guards if needed.  Idempotent.

    Thread-safe: double-checked locking pattern.
    Returns True if a Blackwell GPU was detected.
    """
    global _detected

    # Fast path — no lock once detection is complete
    if _detected is not None:
        return _detected

    with _detected_lock:
        # Re-check under lock (another thread may have completed detection)
        if _detected is not None:
            return _detected

        hide_cuda = _is_hide_cuda()
        cached = os.environ.get(_BLACKWELL_CACHE_ENV)

        if cached == "yes":
            _detected = True
            _apply_guards(hide_cuda=hide_cuda)
            return True

        if cached == "no":
            _detected = False
            return False

        # No cache — run detection
        _detected = False
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=0.5,
            )
            if result.returncode == 0:
                gpu_names = result.stdout.strip().lower()
                if "rtx 50" in gpu_names or "5080" in gpu_names or "5090" in gpu_names:
                    _detected = True
                    _apply_guards(hide_cuda=hide_cuda)
                    os.environ[_BLACKWELL_CACHE_ENV] = "yes"
                    if hide_cuda:
                        print(
                            "Blackwell GPU detected - CPU-only mode (MAXIM_BLACKWELL_HIDE_CUDA=1)",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            "Blackwell GPU detected - GStreamer CUDA disabled, "
                            "LLM CUDA kept (set MAXIM_BLACKWELL_HIDE_CUDA=1 for CPU-only)",
                            file=sys.stderr,
                        )
                else:
                    os.environ[_BLACKWELL_CACHE_ENV] = "no"
            else:
                os.environ[_BLACKWELL_CACHE_ENV] = "no"
        except subprocess.TimeoutExpired:
            os.environ[_BLACKWELL_CACHE_ENV] = "no"
        except Exception:
            pass  # nvidia-smi not found

        return _detected


def is_blackwell_detected() -> bool:
    """Check if Blackwell was detected (without re-running detection)."""
    if _detected is None:
        return ensure_blackwell_guards()
    return _detected
