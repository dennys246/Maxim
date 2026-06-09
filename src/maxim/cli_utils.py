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


# Cloud backend type → (import name, pyproject extra). Used to validate that an
# explicitly-requested cloud model's SDK is installed BEFORE any worker thread
# starts. The openai-compatible providers (gemini/groq/together/fireworks/
# mistral/deepseek) all route through the ``openai`` SDK.
_CLOUD_BACKEND_DEPENDENCY: dict[str, tuple[str, str]] = {
    "anthropic": ("anthropic", "llm-anthropic"),
    "claude": ("anthropic", "llm-anthropic"),
    "openai": ("openai", "llm-openai"),
    "openai_compatible": ("openai", "llm-openai"),
    "openai_compat": ("openai", "llm-openai"),
}


def _missing_backend_dependency(backend_type: str) -> tuple[str, str] | None:
    """Return ``(import_name, extra)`` if a cloud backend's SDK is missing.

    Returns ``None`` when the backend has no optional dependency (or the
    dependency is already importable). Used by :func:`apply_cli_overrides` to
    fail loudly + synchronously on a requested-but-missing cloud SDK.
    """
    from maxim.utils.optional_deps import optional_dependency_available

    dep = _CLOUD_BACKEND_DEPENDENCY.get(str(backend_type).strip().lower())
    if dep is None:
        return None
    import_name, extra = dep
    if optional_dependency_available(import_name):
        return None
    return import_name, extra


# Default for --persona / --mode when neither flag is passed. Preserves
# the historical CLI behavior (the orchestrator used "adversarial" as
# the silent default before the persona deprecation cycle started in 0.9).
# Once --persona is hard-removed in 1.1 and Stages 3-5 of
# docs/plans/persona_cleanup_and_mode_transition.md migrate dispatch off
# persona names entirely, this constant goes with it.
_DEFAULT_SIM_PERSONA = "adversarial"


def _resolve_persona_mode(args: argparse.Namespace) -> None:
    """Resolve --mode / --persona precedence + emit the persona deprecation warning.

    Stage 1 of docs/plans/persona_cleanup_and_mode_transition.md:
    --mode is the new flag, --persona is deprecated in 0.9 and removed
    in 1.1 (matches the C4-C6 timing contract in v1_refinement.md
    Section 5). Both flags are accepted through 1.0; --mode wins on
    conflict.

    After this runs, args.sim_persona is always set (the rest of the CLI
    treats sim_persona as the canonical dispatch slot — Stages 3-5 will
    migrate dispatch off persona names entirely).
    """
    has_persona = hasattr(args, "sim_persona")
    has_mode = hasattr(args, "sim_mode")
    persona_val = getattr(args, "sim_persona", None)
    mode_val = getattr(args, "sim_mode", None)

    if has_mode and has_persona:
        if persona_val != mode_val:
            print(
                f"WARNING: --sim-mode {mode_val!r} and --persona {persona_val!r} both "
                f"specified; using --sim-mode {mode_val!r}. --persona is deprecated and "
                f"will be removed in 1.1.",
                file=sys.stderr,
            )
        args.sim_persona = mode_val
    elif has_mode:
        args.sim_persona = mode_val
    elif has_persona:
        import warnings

        msg = (
            f"--persona / --sim-persona is deprecated and will be removed in 1.1. "
            f"Use --sim-mode {persona_val!r} instead. "
            f"See docs/plans/persona_cleanup_and_mode_transition.md."
        )
        # Stderr line for human visibility (DeprecationWarning is silenced
        # by Python's default warning filter outside __main__).
        print(f"DeprecationWarning: {msg}", file=sys.stderr)
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
        # args.sim_persona already equals persona_val.
    else:
        args.sim_persona = _DEFAULT_SIM_PERSONA


def normalize_args(args: argparse.Namespace) -> None:
    """Coerce and validate parsed CLI arguments, setting env vars as needed."""
    _resolve_persona_mode(args)

    audio_raw = str(getattr(args, "audio", "true")).strip().lower()
    if audio_raw in ("1", "true", "t", "yes", "y", "on"):
        args.audio = True
    elif audio_raw in ("0", "false", "f", "no", "n", "off"):
        args.audio = False
    else:
        raise SystemExit(f"Invalid --audio value: {args.audio!r} (expected True/False)")

    _raw_interactive = getattr(args, "interactive", None)
    if _raw_interactive is None:
        # None means "auto" — leave the resolved boolean to downstream
        # display config (sim path) and default to True for live runs.
        args.interactive = True
    else:
        interactive_raw = str(_raw_interactive).strip().lower()
        if interactive_raw in ("1", "true", "t", "yes", "y", "on"):
            args.interactive = True
        elif interactive_raw in ("0", "false", "f", "no", "n", "off"):
            args.interactive = False
        else:
            raise SystemExit(f"Invalid --interactive value: {args.interactive!r} (expected True/False)")

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
                # Fail loudly + synchronously when the cloud SDK is missing,
                # before any worker thread starts (otherwise it degrades into a
                # swallowed warning + _llm_unavailable fallback — the
                # 2026-06-05 cloud-dispatch incident).
                _missing = _missing_backend_dependency(profile.get("backend", ""))
                if _missing is not None:
                    _import_name, _extra = _missing
                    raise SystemExit(
                        f"Error: {language_model} needs the {_import_name!r} package, which is "
                        f"not installed.\n  Fix: pip install 'pymaxim[{_extra}]'"
                    )
            # Persist across sessions so the user doesn't need --llm every time
            try:
                from maxim.runtime.lane_backends import _write_persisted_model

                _write_persisted_model(selected)
            except Exception:
                pass
        args.language_model = selected

    # ── --auto-download flag (peer_leader_flexibility_plan P5) ──────────
    # Forwards to MAXIM_AUTO_DOWNLOAD_MODELS so ensure_available() picks
    # it up without plumbing a new arg through every layer.
    if getattr(args, "auto_download", False):
        os.environ["MAXIM_AUTO_DOWNLOAD_MODELS"] = "1"

    # ── --llm-n-ctx override (peer_leader_flexibility_plan P4c) ──────────
    # Forwards to MAXIM_LLM_N_CTX so the auto-spawn path in lane_backends
    # picks it up without plumbing a new arg through every layer.
    llm_n_ctx = getattr(args, "llm_n_ctx", None)
    if llm_n_ctx is not None:
        try:
            n_ctx_int = int(llm_n_ctx)
        except (TypeError, ValueError):
            raise SystemExit(f"Invalid --llm-n-ctx value: {llm_n_ctx!r} (expected integer)")
        if n_ctx_int <= 0:
            raise SystemExit(f"--llm-n-ctx must be positive, got {n_ctx_int}")
        os.environ["MAXIM_LLM_N_CTX"] = str(n_ctx_int)

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
            # Fail loudly + synchronously (main thread, before any worker
            # threads spin up) when the cloud backend's optional SDK is not
            # installed. Without this the missing package surfaces only as a
            # swallowed worker warning and the run completes with cost=$0 and
            # every action an _llm_unavailable fallback — the 2026-06-05
            # cloud-dispatch incident. SystemExit gives a clean CLI message
            # (no traceback) consistent with the checks above.
            _missing = _missing_backend_dependency(profile.get("backend", ""))
            if _missing is not None:
                _import_name, _extra = _missing
                raise SystemExit(
                    f"{label} {model_name!r} needs the {_import_name!r} package, which is not "
                    f"installed.\n  Fix: pip install 'pymaxim[{_extra}]'"
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


_CLOUD_API_KEY_TO_PROFILE: tuple[tuple[str, str], ...] = (
    # Priority order for default cloud profile selection. First key
    # present wins. Profiles match the bundled cloud profile catalog
    # in models/language/config.py — anything renamed there must be
    # mirrored here.
    ("ANTHROPIC_API_KEY", "claude-sonnet"),
    ("OPENAI_API_KEY", "gpt-4o"),
    ("GOOGLE_API_KEY", "gemini-2.5-flash"),
    ("GROQ_API_KEY", "groq-llama3-70b"),
    ("TOGETHER_API_KEY", "together-llama3-70b"),
    ("FIREWORKS_API_KEY", "fireworks-llama3-70b"),
    ("MISTRAL_API_KEY", "mistral-large"),
    ("DEEPSEEK_API_KEY", "deepseek-chat"),
)


def configure_cloud_solo_auto_detect(logger: logging.Logger) -> None:
    """C7a of config_unification.md: solo + cloud key auto-detect.

    When the operator has a cloud API key set (Anthropic, OpenAI,
    Google, Groq, Together, Fireworks, Mistral, or DeepSeek) AND
    no local LLM profile is configured AND the resolved role is solo
    AND no peer-routing URL is set, implicitly enable the cloud-LLM
    gates so ``maxim`` "just works" with bare-API-key configuration.

    Sets ONLY env vars that are not already set (operator overrides
    always win). Each implicit set logs INFO so the auto-config is
    visible.

    Does NOT fire when:
    - Role is leader or peer (cloud-as-leader-serving-peers is C7b,
      deferred until ``mesh_usage_accounting.md`` ships)
    - ``MAXIM_LLM_PROFILE`` is already set (operator picked a model)
    - ``MAXIM_LANE_LARGE_REMOTE_URL`` is set (routing to a leader)
    - No cloud API key env vars are present

    This pairs with the larger architectural framing pinned in
    config_unification.md C7a discussion: role and LLM-source are
    independent axes; ``solo`` means "alone, not in a mesh" — it
    doesn't imply "local LLM". The auto-detect just removes the
    seven-flag incantation operators previously needed for cloud-only
    setup.
    """
    role = os.environ.get("MAXIM_ROLE", "").strip().lower()
    if role and role != "solo":
        return

    # Operator already picked a local model — respect it
    if os.environ.get("MAXIM_LLM_PROFILE", "").strip():
        return

    # Routing to a peer leader — that's a different mode, skip
    if os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL", "").strip():
        return

    # Find available cloud keys
    available: list[tuple[str, str]] = []
    for env_name, profile in _CLOUD_API_KEY_TO_PROFILE:
        if os.environ.get(env_name, "").strip():
            available.append((env_name, profile))

    if not available:
        return

    # Pick the highest-priority available profile
    chosen_env, chosen_profile = available[0]
    n_keys_present = len(available)

    # Apply the implicit defaults — only if not already set
    def _setdefault_with_log(name: str, value: str) -> None:
        if not os.environ.get(name, "").strip():
            os.environ[name] = value
            logger.info(
                "C7a auto-detect (solo + %s present): %s=%s",
                chosen_env,
                name,
                value,
            )

    _setdefault_with_log("MAXIM_LLM_ENABLED", "1")
    _setdefault_with_log("MAXIM_LLM_CLOUD_ENABLED", "1")
    _setdefault_with_log("MAXIM_MAX_CLOUD_LANES", str(min(n_keys_present, 3)))
    _setdefault_with_log("MAXIM_LLM_REDACTION_POLICY", "standard")
    _setdefault_with_log("MAXIM_CLOUD_SESSION_BUDGET", "5.0")
    _setdefault_with_log("MAXIM_LLM_PROFILE", chosen_profile)


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
    print(
        "  INFO: No GPU detected — using smollm-1.7b-instruct (CPU mode).\n"
        "  For better results, set a cloud API key (e.g. export ANTHROPIC_API_KEY=...)\n"
        "  or install GPU support: pip install 'pymaxim[llm-torch]'",
        file=sys.stderr,
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


# Memory targets for --clear-memory, as glob patterns relative to data_home / ~/.maxim/.
#
# Two persistence layouts coexist by design:
#   - Global default-network / bridge learners write ONE shared file under util/.
#   - Per-agent bio-systems (Hippocampus, NAc, SCN, ATL, AngularGyrus) are persisted
#     per agent under agents/<agent_id>/ by AgentFactory / build_bio_stack. Clearing a
#     bio-system globs across every agent dir so the bio-system is wiped everywhere;
#     legacy flat paths are included so pre-0.9 installs are cleaned up too.
# A pattern containing "*" is glob-expanded; every match is removed (each is printed).
MEMORY_PATHS: dict[str, tuple[str, ...]] = {
    # Global learners (one shared file under util/)
    "focus": ("util/focus_learner.json",),
    "bounds": ("util/workspace_bounds.json",),
    "escalation": ("util/escalation_learning.json",),
    "fear": ("util/fear_learning.json",),
    "threshold": ("util/adaptive_thresholds.json",),
    "pain": ("util/pain_detector.json",),
    "statistician": ("util/statistician_state.json",),
    "cross_layer": ("util/cross_layer_graph.json",),
    "semantic": ("util/semantic_embeddings.npz",),
    # Per-agent bio-systems (agents/<id>/<file>.json) + legacy fallbacks
    "hippo": (
        "agents/*/hippocampus.json",
        "memory/hippocampus.json",
        "memory/memories.json",
        "util/hippocampus.json",
    ),
    "nac": ("agents/*/nac.json", "util/nac_state.json"),
    "scn": ("agents/*/scn.json", "util/scn_state.json"),
    "atl": ("agents/*/atl.json", "util/atl_state.json"),
    "angular_gyrus": ("agents/*/angular_gyrus.json",),
    # Planning artifacts (directory)
    "planning": ("planning",),
}


def clear_memory(memory_types: str, home_dir: str | None = None) -> dict[str, bool]:
    """Clear persistent memory files/directories.

    Each memory type expands to one or more glob patterns under ``~/.maxim/``
    (see :data:`MEMORY_PATHS`). Per-agent bio-systems are matched across every
    ``agents/<id>/`` directory; every matched file or directory is removed.

    Args:
        memory_types: Comma-separated memory types or 'all'.
        home_dir: Base data directory (deprecated, uses ~/.maxim/ by default).

    Returns:
        Dict mapping memory type to success (True if at least one target was
        cleared, False if none were found).
    """
    import shutil
    from pathlib import Path
    from maxim.utils.paths import data_home

    results: dict[str, bool] = {}

    if memory_types == "all":
        types_to_clear = list(MEMORY_PATHS.keys())
    else:
        types_to_clear = [t.strip().lower() for t in memory_types.split(",")]

    base = Path(home_dir) if home_dir is not None else data_home()

    for mem_type in types_to_clear:
        if mem_type not in MEMORY_PATHS:
            print(f"Unknown memory type: {mem_type}", file=sys.stderr)
            print(f"Available types: {', '.join(MEMORY_PATHS.keys())}", file=sys.stderr)
            results[mem_type] = False
            continue

        # Expand each pattern relative to base; "*" globs across agent dirs.
        matched: list[Path] = []
        for pattern in MEMORY_PATHS[mem_type]:
            if "*" in pattern:
                matched.extend(sorted(base.glob(pattern)))
            else:
                candidate = base / pattern
                if candidate.exists():
                    matched.append(candidate)

        if not matched:
            results[mem_type] = False
            print(f"  Not found: {mem_type}")
            continue

        cleared_any = False
        for path in matched:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                cleared_any = True
                print(f"  Cleared: {mem_type} ({path})")
            except Exception as e:
                print(f"  Failed to clear {mem_type}: {e}", file=sys.stderr)
        results[mem_type] = cleared_any

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
