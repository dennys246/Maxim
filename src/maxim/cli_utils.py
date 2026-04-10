"""CLI utility functions — argument normalization, GPU detection, memory cleanup.

Extracted from cli.py for single-responsibility decomposition.
These are stateless helpers with no lifecycle dependencies.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any


def normalize_epoch_value(value: object) -> int:
    """Convert any input to a non-negative int epoch count."""
    try:
        epochs = int(value)
    except Exception:
        return 0
    return epochs if epochs > 0 else 0


def normalize_args(args: argparse.Namespace) -> None:
    """Coerce and validate parsed CLI arguments, setting env vars as needed."""
    audio_raw = str(getattr(args, "audio", "true")).strip().lower()
    if audio_raw in ("1", "true", "t", "yes", "y", "on"):
        args.audio = True
    elif audio_raw in ("0", "false", "f", "no", "n", "off"):
        args.audio = False
    else:
        raise SystemExit(f"Invalid --audio value: {args.audio!r} (expected True/False)")

    # --interactive: "auto" (default), "on", or "off"
    interactive_raw = str(getattr(args, "interactive", "auto")).strip().lower()
    if interactive_raw not in ("auto", "on", "off"):
        raise SystemExit(f"Invalid --interactive value: {args.interactive!r} (expected auto/on/off)")

    if str(getattr(args, "mode", "active")).strip().lower() == "sleep":
        args.audio = True
    args.epochs = normalize_epoch_value(getattr(args, "epochs", 0))

    language_model = getattr(args, "language_model", None)
    if language_model is not None:
        from maxim.models.language.config import list_llm_profiles, normalize_llm_profile

        selected = normalize_llm_profile(language_model)
        if selected:
            available = list_llm_profiles()
            if available and selected not in available:
                print(f"\n  Unknown model: '{language_model}'\n")
                print("  Run 'maxim --list-models' to see all available models.\n")
                from difflib import get_close_matches

                close = get_close_matches(language_model.lower(), [a.lower() for a in available], n=3, cutoff=0.4)
                if close:
                    print(f"  Did you mean: {', '.join(close)}?\n")
                raise SystemExit(1)
            os.environ["MAXIM_LLM_PROFILE"] = selected
            # Validate API key early for cloud models
            from maxim.models.language.config import _BUILTIN_PROFILES

            profile = _BUILTIN_PROFILES.get(selected, {})
            if profile.get("cloud"):
                api_key_env = profile.get("api_key_env", "")
                if api_key_env and not os.environ.get(api_key_env):
                    raise SystemExit(
                        f"Error: {language_model} requires {api_key_env}.\n  Fix: export {api_key_env}=<your-key>"
                    )
            # Persist across sessions so the user doesn't need --llm every time
            try:
                from maxim.runtime.lane_backends import _write_persisted_model

                _write_persisted_model(selected)
            except Exception:
                pass
        args.language_model = selected

    # ── Cloud provider CLI flags ──────────────────────────────────────────
    cloud_fallback = getattr(args, "cloud_fallback", None)
    cloud_lane = getattr(args, "cloud_lane", None)
    cloud_budget = getattr(args, "cloud_budget", None)

    if cloud_fallback or cloud_lane:
        from maxim.models.language.config import _BUILTIN_PROFILES, normalize_llm_profile

        for label, model_name in [
            ("--cloud-fallback", cloud_fallback),
            ("--cloud-lane", cloud_lane[1] if cloud_lane else None),
        ]:
            if model_name is None:
                continue
            resolved = normalize_llm_profile(model_name)
            profile = _BUILTIN_PROFILES.get(resolved, {})
            if not profile.get("cloud"):
                raise SystemExit(
                    f"{label} {model_name!r} is not a cloud profile. "
                    f"Use a cloud model like claude-sonnet, claude-haiku, gpt-4o, gpt-4o-mini."
                )
            api_key_env = profile.get("api_key_env", "")
            if api_key_env and not os.environ.get(api_key_env):
                raise SystemExit(
                    f"{label} {model_name!r} requires {api_key_env} to be set.\n  export {api_key_env}=<your-key>"
                )

        if cloud_fallback:
            resolved = normalize_llm_profile(cloud_fallback)
            os.environ["MAXIM_CLOUD_FALLBACK_MODEL"] = resolved
        if cloud_lane:
            lane_name, model_name = cloud_lane
            resolved = normalize_llm_profile(model_name)
            os.environ[f"MAXIM_CLOUD_LANE_{lane_name.upper()}_MODEL"] = resolved
        if cloud_budget is not None:
            os.environ["MAXIM_CLOUD_SESSION_BUDGET"] = str(cloud_budget)

        os.environ.setdefault("MAXIM_LLM_CLOUD_ENABLED", "1")
        os.environ.setdefault("MAXIM_LLM_REDACTION_POLICY", "standard")
        current_max = int(os.environ.get("MAXIM_MAX_CLOUD_LANES", "0"))
        needed = sum(1 for x in [cloud_fallback, cloud_lane] if x)
        if current_max < needed:
            os.environ["MAXIM_MAX_CLOUD_LANES"] = str(needed)

    segmentation_model = getattr(args, "segmentation_model", None)
    if segmentation_model is not None:
        from maxim.models.vision.registry import list_engines, normalize_engine_name

        selected = normalize_engine_name(segmentation_model) or "rtm"
        available = list_engines()
        if available and selected not in available:
            opts = ", ".join(available)
            raise SystemExit(f"Unknown --segmentation-model {segmentation_model!r}. Available: {opts}")
        os.environ["MAXIM_SEGMENTATION_MODEL"] = selected
        args.segmentation_model = selected


def gpu_available() -> bool:
    """Check whether a CUDA or MPS GPU is available."""
    try:
        import torch
    except Exception:
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


def check_gpu_status(logger: logging.Logger) -> None:
    """Check and log GPU availability for TensorFlow and PyTorch."""
    from maxim.utils import gpu_detect

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible == "":
        logger.warning('GPU acceleration disabled (CUDA_VISIBLE_DEVICES="")')
        logger.info("Running in CPU-only mode")
        return

    tf_gpus: list[Any] = []
    tf_gpu_info: list[str] = []
    tf_on_cpu = False
    try:
        import tensorflow as tf

        tf_gpus = tf.config.get_visible_devices("GPU")
        if not tf_gpus and gpu_detect.is_blackwell_detected():
            tf_on_cpu = True
        elif tf_gpus:
            for gpu in tf_gpus:
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    gpu_name = gpu_details.get("device_name", "Unknown GPU")
                    tf_gpu_info.append(gpu_name)
                except Exception:
                    tf_gpu_info.append(str(gpu).split(":")[-1].rstrip("'"))
    except Exception:
        pass

    torch_gpus = 0
    torch_gpu_info: list[str] = []
    try:
        import torch

        if torch.cuda.is_available():
            torch_gpus = torch.cuda.device_count()
            for i in range(torch_gpus):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    torch_gpu_info.append(f"{gpu_name} ({gpu_mem:.1f} GB)")
                except Exception:
                    torch_gpu_info.append(f"GPU {i}")
    except Exception:
        pass

    if torch_gpus:
        logger.info("GPU acceleration enabled (PyTorch)")
        logger.info(f"   PyTorch detected {torch_gpus} GPU(s):")
        for i, info in enumerate(torch_gpu_info):
            logger.info(f"     [{i}] {info}")
        if tf_on_cpu:
            logger.info("   TensorFlow: CPU mode (Blackwell GPU not supported by TF 2.20)")
        elif tf_gpus:
            logger.info(f"   TensorFlow detected {len(tf_gpus)} GPU(s)")
    elif tf_gpus:
        logger.info("GPU acceleration enabled (TensorFlow)")
        logger.info(f"   TensorFlow detected {len(tf_gpus)} GPU(s):")
        for i, info in enumerate(tf_gpu_info):
            logger.info(f"     [{i}] {info}")
    else:
        if gpu_detect.is_blackwell_detected():
            logger.info("Running in CPU-only mode (Blackwell GPU workaround)")
            logger.info("   RTX 5080/5090 detected but CUDA disabled to avoid GStreamer crash")
            logger.info("   See github_issue.md for details on this reachy_mini SDK issue")
        else:
            logger.warning("No GPU detected - running in CPU-only mode")
            logger.info("   For GPU support, ensure:")
            logger.info("   - NVIDIA drivers are installed (570+)")
            logger.info("   - CUDA-compatible GPU is available")


def configure_cpu_fallback_model(logger: logging.Logger, home_dir: str = "data") -> None:
    """Configure a smaller LLM model for CPU-only inference when no GPU is available."""
    import json

    llm_config_path = os.path.join(home_dir, "util", "llm.json")
    using_llama_cpp = False
    try:
        if os.path.exists(llm_config_path):
            with open(llm_config_path) as f:
                llm_cfg = json.load(f)
            profile_name = llm_cfg.get("profile", "")
            profiles = llm_cfg.get("profiles", {})
            if profile_name in profiles:
                using_llama_cpp = profiles[profile_name].get("backend") == "llama_cpp"
    except Exception:
        pass

    if using_llama_cpp:
        logger.info("Using llama.cpp backend with native Metal GPU support")
        return

    logger.warning(
        "No GPU detected; falling back to smaller model (smollm-1.7b-instruct) "
        "with CPU inference. Performance may be reduced."
    )
    os.environ.setdefault("MAXIM_LLM_PROFILE", "smollm-1.7b-instruct")
    os.environ.setdefault("MAXIM_LLM_N_GPU_LAYERS", "0")


def reexec_with_mode(args: argparse.Namespace, *, mode: str) -> None:
    """Restart the process in a new operational mode via os.execv."""
    mode = str(mode or "").strip().lower()
    if not mode:
        return

    audio_flag = bool(getattr(args, "audio", True))
    if mode == "sleep":
        audio_flag = True

    epochs_value = normalize_epoch_value(getattr(args, "epochs", 0))
    argv = [
        sys.executable,
        "-m",
        "maxim.cli",
        "--robot-name",
        str(getattr(args, "robot_name", "reachy_mini")),
        "--home-dir",
        str(getattr(args, "home_dir", "data")),
        "--timeout",
        str(float(getattr(args, "timeout", 30.0) or 30.0)),
        "--epochs",
        str(epochs_value),
        "--verbosity",
        str(int(getattr(args, "verbosity", 1) or 1)),
        "--mode",
        mode,
        "--audio",
        "true" if audio_flag else "false",
        "--audio_len",
        str(float(getattr(args, "audio_len", 5.0) or 5.0)),
        "--interactive",
        "true" if bool(getattr(args, "interactive", True)) else "false",
    ]
    language_model = str(getattr(args, "language_model", "") or "").strip()
    if language_model:
        argv.extend(["--language-model", language_model])
    segmentation_model = str(getattr(args, "segmentation_model", "") or "").strip()
    if segmentation_model:
        argv.extend(["--segmentation-model", segmentation_model])
    memory_path = str(getattr(args, "memory_path", "") or "").strip()
    if memory_path:
        argv.extend(["--memory-path", memory_path])
    if bool(getattr(args, "reset", False)):
        argv.append("--reset")
    if bool(getattr(args, "enable_embeddings", False)):
        argv.append("--enable-embeddings")
    os.execv(sys.executable, argv)


# Memory file paths for --clear-memory (relative to data_home / ~/.maxim/)
MEMORY_PATHS = {
    "focus": "util/focus_learner.json",
    "bounds": "util/workspace_bounds.json",
    "escalation": "util/escalation_learning.json",
    "fear": "util/fear_learning.json",
    "threshold": "util/adaptive_thresholds.json",
    "nac": "util/nac_state.json",
    "scn": "util/scn_state.json",
    "hippo": "util/hippocampus.json",
    "pain": "util/pain_detector.json",
    "semantic": "util/semantic_embeddings.npz",
    "statistician": "util/statistician_state.json",
    "atl": "util/atl_state.json",
    "cross_layer": "util/cross_layer_graph.json",
    "planning": "planning",
}


def clear_memory(memory_types: str, home_dir: str | None = None) -> dict[str, bool]:
    """Clear persistent memory files.

    Args:
        memory_types: Comma-separated memory types or 'all'.
        home_dir: Base data directory (deprecated, uses ~/.maxim/ by default).

    Returns:
        Dict mapping memory type to success (True if cleared, False if not found).
    """
    from pathlib import Path
    from maxim.utils.paths import resolve_user_state

    results: dict[str, bool] = {}

    if memory_types == "all":
        types_to_clear = list(MEMORY_PATHS.keys())
    else:
        types_to_clear = [t.strip().lower() for t in memory_types.split(",")]

    for mem_type in types_to_clear:
        if mem_type not in MEMORY_PATHS:
            print(f"Unknown memory type: {mem_type}", file=sys.stderr)
            print(f"Available types: {', '.join(MEMORY_PATHS.keys())}", file=sys.stderr)
            results[mem_type] = False
            continue

        rel_path = MEMORY_PATHS[mem_type]
        if home_dir is not None:
            full_path = Path(home_dir) / rel_path
        else:
            full_path = resolve_user_state(rel_path)

        if full_path.exists():
            try:
                full_path.unlink()
                results[mem_type] = True
                print(f"  Cleared: {mem_type} ({full_path})")
            except Exception as e:
                print(f"  Failed to clear {mem_type}: {e}", file=sys.stderr)
                results[mem_type] = False
        else:
            results[mem_type] = False
            print(f"  Not found: {mem_type} ({full_path})")

    return results


def clear_python_cache(base_dir: str | None = None) -> int:
    """Clear Python bytecode cache (__pycache__ directories and .pyc files).

    Returns the number of cache directories removed.
    """
    import shutil
    from pathlib import Path

    if base_dir is None:
        base_dir = str(Path(__file__).parent)

    base_path = Path(base_dir)
    removed = 0

    for cache_dir in base_path.rglob("__pycache__"):
        try:
            shutil.rmtree(cache_dir)
            removed += 1
        except Exception:
            pass

    for pyc_file in base_path.rglob("*.pyc"):
        try:
            pyc_file.unlink()
        except Exception:
            pass

    return removed
