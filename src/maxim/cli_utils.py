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

from maxim.utils.paths import AGENT_HOME_PATTERNS


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


# Default flow-shape label when --sim-mode is not passed. The persona
# system (and the --persona/--sim-persona flags) was hard-removed in 1.1
# per docs/plans/archive/persona_cleanup_and_mode_transition.md Stages
# 3-5; --sim-mode is a free-form label recorded in reports/logs, while
# flow behavior is driven by the dispatch path (campaign YAML,
# --research, benchmark, ...), exactly as it always was.
_DEFAULT_SIM_MODE = "generative"


def _resolve_sim_mode(args: argparse.Namespace) -> None:
    """Ensure args.sim_mode is always set (default: "generative")."""
    if not hasattr(args, "sim_mode") or getattr(args, "sim_mode", None) in (None, ""):
        args.sim_mode = _DEFAULT_SIM_MODE


def normalize_args(args: argparse.Namespace) -> None:
    """Coerce and validate parsed CLI arguments, setting env vars as needed."""
    _resolve_sim_mode(args)

    audio_raw = str(getattr(args, "audio", "true")).strip().lower()
    if audio_raw in ("1", "true", "t", "yes", "y", "on"):
        args.audio = True
    elif audio_raw in ("0", "false", "f", "no", "n", "off"):
        args.audio = False
    else:
        raise SystemExit(f"Invalid --audio value: {args.audio!r} (expected True/False)")

    _raw_interactive = getattr(args, "interactive", None)
    # Recorded BEFORE coercion: afterwards args.interactive is a bool and
    # "unset" (auto) is indistinguishable from an explicit true.
    args.interactive_explicit = _raw_interactive is not None
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
            from maxim.models.language.config import get_profile

            profile = get_profile(selected) or {}
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
        from maxim.models.language.config import get_profile, normalize_llm_profile

        for label, model_name in [
            ("--cloud-fallback", cloud_fallback),
            ("--cloud-lane", cloud_lane[1] if cloud_lane else None),
        ]:
            if model_name is None:
                continue
            resolved = normalize_llm_profile(model_name)
            profile = get_profile(resolved) or {}
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
    always win). Where ``config.json`` sets the field, it exports THAT
    value rather than its own default: the cloud gates are read only from
    the environment at runtime, so skipping would silently drop the
    operator's ``maxim config set`` value (#1030). Each implicit set logs
    INFO so the auto-config is visible.

    Does NOT fire when:
    - Role is leader or peer (cloud-as-leader-serving-peers is C7b,
      deferred until ``mesh_usage_accounting.md`` ships)
    - ``MAXIM_LLM_PROFILE`` is already set (operator picked a model)
    - ``MAXIM_LANE_LARGE_REMOTE_URL`` is set (routing to a leader)
    - No cloud API key env vars are present
    - ``config.json`` cannot be read (it may pin a local model; #1030)
    - ``config.json`` sets ``cloud.enabled`` to false (the documented
      off-switch)
    - ``config.json`` sets ``llm.profile`` to a local or unrecognised
      profile (``maxim config set llm.profile …`` is the operator picking
      a model too, #1030). A configured CLOUD profile still gets the
      cloud gates, under its own name.

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

    # An unreadable config.json must not read as "nothing configured": the operator may have pinned a local
    # model in it, and firing would pick a billed cloud one instead (#1030). resolve_setting swallows a load
    # failure and falls to defaults, so load it here first.
    from maxim.exceptions import ConfigurationError
    from maxim.runtime.config_loader import get_config

    try:
        get_config()
    except (ConfigurationError, OSError) as exc:
        logger.warning("C7a auto-detect skipped: config.json could not be read (%s)", exc)
        return

    if _config_value("cloud.enabled") is False:
        logger.info("C7a auto-detect skipped: config.json sets cloud.enabled=false")
        return
    # The operator picked a model in config.json (#1030): a local or unrecognised profile stands; a cloud one
    # still gets the gates below, under its own name rather than the first key's default.
    configured = _config_value("llm.profile")
    if configured is not None and _registered_cloud_flag(str(configured)) is not True:
        logger.info("C7a auto-detect skipped: config.json sets llm.profile=%s (not a cloud profile)", configured)
        return

    # Pick the highest-priority available profile
    chosen_env, chosen_profile = available[0]
    n_keys_present = len(available)

    # Apply the implicit defaults — only if the environment does not set them, and config.json's own value where
    # it sets one (the runtime reads these gates from the environment only).
    def _setdefault_with_log(name: str, field_path: str, default: str) -> None:
        if os.environ.get(name, "").strip():
            return
        configured_value = _config_value(field_path)
        value = default if configured_value is None else _env_text(configured_value)
        os.environ[name] = value
        logger.info(
            "C7a auto-detect (solo + %s present): %s=%s%s",
            chosen_env,
            name,
            value,
            "" if configured_value is None else " (from config.json)",
        )

    _setdefault_with_log("MAXIM_LLM_ENABLED", "llm.enabled", "1")
    _setdefault_with_log("MAXIM_LLM_CLOUD_ENABLED", "cloud.enabled", "1")
    _setdefault_with_log("MAXIM_MAX_CLOUD_LANES", "cloud.max_lanes", str(min(n_keys_present, 3)))
    _setdefault_with_log("MAXIM_LLM_REDACTION_POLICY", "cloud.redaction_policy", "standard")
    _setdefault_with_log("MAXIM_CLOUD_SESSION_BUDGET", "cloud.session_budget_usd", "5.0")
    _setdefault_with_log("MAXIM_LLM_PROFILE", "llm.profile", chosen_profile)


def _env_text(value) -> str:
    """A config value as its env-var spelling (booleans as 1/0, as the env coercers read them)."""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _config_value(field_path: str):
    """The field's value when ``config.json`` sets it (source ``config``), else None. The environment is read by
    the caller; a schema default is not an operator's choice."""
    from maxim.runtime.config_loader import resolve_setting

    value, source = resolve_setting(field_path)
    if source != "config" or value is None or (isinstance(value, str) and not value.strip()):
        return None
    return value


def _registered_cloud_flag(profile: str) -> bool | None:
    """The profile's own ``cloud`` field when it is registered (builtin or ``maxim model add``), else None."""
    from maxim.models.language.config import get_profile, normalize_llm_profile

    entry = get_profile(normalize_llm_profile(profile))
    return None if entry is None else bool(entry.get("cloud", False))


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
    """Restart the process in a new mode via os.execv.

    Two vocabularies (#829): an OPERATIONAL name (passive / active / singularity) re-execs with the SAME
    run mode and ``--operational-mode <name>``; any other name is a RUN mode (``--mode <name>``) and
    keeps the current ``--operational-mode``. Before, an operational name went to ``--mode``, which
    argparse rejected (exit 2).
    """
    from maxim.modes.definitions import OPERATIONAL_MODES

    mode = str(mode or "").strip().lower()
    if not mode:
        return
    operational = str(getattr(args, "operational_mode", "") or "").strip().lower() or None
    if mode in OPERATIONAL_MODES:
        operational = mode
        mode = str(getattr(args, "mode", "") or "exploration").strip().lower()

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
    if operational:
        argv.extend(["--operational-mode", operational])
    os.execv(sys.executable, argv)


# Memory targets for --clear-memory, as glob patterns relative to the data home (``utils/paths.py::data_home``).
#
# Two persistence layouts coexist by design:
#   - Global default-network / bridge learners write ONE shared file under util/.
#   - An agent's stores persist together in its AGENT HOME. What counts as an agent's state is defined once,
#     ``runtime/agent_factory.py::AGENT_STATE_FILES`` plus its ``memory_hub/`` (the definition ``create.agent``
#     refuses on and the #932 migration moves); ``_AGENT_FILES`` below splits exactly that by memory type, and a
#     test pins the two together. The agent homes are every directory a writer points a bio stack at:
#     ``agents/<id>/`` (AgentFactory, ``maxim.create.agent``), ``memory/`` (``build_bio_stack(persistence_dir=
#     user_memory())``: the Reachy runtime and ``maxim hive merge``), ``orchestrator/`` (the sim orchestrator) and
#     ``sessions/`` (``maxim.run()``'s agent before #932), declared once as ``utils/paths.py::AGENT_HOME_PATTERNS``.
#     Clearing a type clears it in EVERY agent home (#1337).
# A pattern containing "*" is glob-expanded; every match is removed (each is printed).
AGENT_HOMES: tuple[str, ...] = AGENT_HOME_PATTERNS

_AGENT_FILES: dict[str, tuple[str, ...]] = {
    "hippo": ("hippocampus.json", "hippocampus.json.backup", "memories.json"),
    "nac": ("nac.json", "nac.meta.json"),  # nac.meta.json: `hive merge`'s record beside the merged NAc
    "ec": ("ec.json",),
    "scn": ("scn.json",),
    "atl": ("atl.json",),
    "angular_gyrus": ("angular_gyrus.json",),
    "cerebellum": ("cerebellum.json",),
    "memory_hub": ("memory_hub",),
}

_GLOBAL_PATHS: dict[str, tuple[str, ...]] = {
    # Global learners (one shared file under util/)
    "focus": ("util/focus_learner.json",),
    "bounds": ("util/learned_bounds.json", "util/workspace_bounds.json"),  # spatial/bounds_learner.py; legacy
    "escalation": ("util/escalation_learning.json",),
    "fear": ("util/fear_learning.json",),
    "threshold": ("util/adaptive_thresholds.json",),
    "pain": ("util/pain_detector.json",),
    "statistician": ("util/statistician_state.json",),
    "significance": ("util/significance_weights.json",),  # agents/exec_agent.py
    "short_term": ("short_term_memory",),  # agents/exec_agent.py's staging files that feed the hippocampus
    "cross_layer": ("util/cross_layer_graph.json",),
    "semantic": ("util/semantic_embeddings.npz",),
    "tool_index": ("memory/tool_index.json",),  # the learned tool index (tools/learned_index.py)
    # Legacy flat bio-system files (pre-0.9)
    "hippo": ("util/hippocampus.json",),
    "nac": ("util/nac_state.json",),
    "scn": ("util/scn_state.json",),
    "atl": ("util/atl_state.json",),
    # Planning artifacts (directory)
    "planning": ("planning",),
}

MEMORY_PATHS: dict[str, tuple[str, ...]] = {}
for _type in (*_AGENT_FILES, *_GLOBAL_PATHS):
    if _type not in MEMORY_PATHS:
        MEMORY_PATHS[_type] = tuple(
            f"{home}/{name}" for home in AGENT_HOMES for name in _AGENT_FILES.get(_type, ())
        ) + _GLOBAL_PATHS.get(_type, ())
del _type

# D2: memory types that MUST be invalidated together. NAc's ``reward_bias`` /
# ``cluster_reward_bias`` are keyed by EC node ids, so clearing one half leaves
# the other referring to nodes that will never be re-allocated — persistence
# that LOOKS like it works while the biases silently dangle (the reasoning is
# stated at ``runtime/bio_stack.py``'s EC construction). Before this existed
# there was no ``ec`` key at all, so ``--clear-memory all`` wiped NAc and left
# EC behind on every run.
MEMORY_PAIRS: tuple[frozenset[str], ...] = (frozenset({"nac", "ec"}),)


def expand_paired_types(requested: list[str]) -> list[str]:
    """Add the partner of any memory type that may not be cleared alone.

    Order is preserved and each type appears once; a partner is appended
    directly after the type that pulled it in so the printed output reads in
    the order the operator will see it happen.
    """
    out: list[str] = []
    for mem_type in requested:
        if mem_type not in out:
            out.append(mem_type)
        for pair in MEMORY_PAIRS:
            if mem_type in pair:
                for partner in sorted(pair - {mem_type}):
                    if partner not in out:
                        out.append(partner)
    return out


def _types_to_clear(memory_types: str) -> tuple[list[str], list[str]]:
    """The memory types ``memory_types`` names (pairs expanded, D2), and the names that are not memory types."""
    if memory_types.strip().lower() == "all":
        return list(MEMORY_PATHS), []
    requested = [t.strip().lower() for t in memory_types.split(",") if t.strip()]
    types = expand_paired_types(requested)
    for added in [t for t in types if t not in requested]:
        partner = next(sorted(pair - {added})[0] for pair in MEMORY_PAIRS if added in pair)
        print(f"  Also clearing: {added} (must be invalidated with {partner} — see MEMORY_PAIRS)", file=sys.stderr)
    unknown = [t for t in types if t not in MEMORY_PATHS]
    return [t for t in types if t in MEMORY_PATHS], unknown


def plan_clear_memory(types: list[str], bases: list[Any]) -> dict[str, list[Any]]:
    """Every existing path each memory type matches under each base, in order, each path once."""
    from pathlib import Path

    plan: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for mem_type in types:
        matched: list[Path] = []
        for base in bases:
            for pattern in MEMORY_PATHS[mem_type]:
                candidates = sorted(Path(base).glob(pattern)) if "*" in pattern else [Path(base) / pattern]
                for p in candidates:
                    if os.path.lexists(p) and p not in seen:
                        seen.add(p)
                        matched.append(p)
        plan[mem_type] = matched
    return plan


def _delete_planned(plan: dict[str, list[Any]]) -> dict[str, bool]:
    """Remove every planned path; True per type when at least one of its paths was removed."""
    import shutil

    results: dict[str, bool] = {}
    for mem_type, paths in plan.items():
        if not paths:
            results[mem_type] = False
            print(f"  Not found: {mem_type}")
            continue
        cleared_any = False
        for path in paths:
            try:
                if path.is_dir() and not path.is_symlink():  # a symlinked store is unlinked, never followed
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError as e:
                print(f"  Failed to clear {mem_type} ({path}): {e}", file=sys.stderr)
                continue
            cleared_any = True
            print(f"  Cleared: {mem_type} ({path})")
        results[mem_type] = cleared_any
    return results


def clear_memory(memory_types: str, home_dir: str | None = None) -> dict[str, bool]:
    """Clear persistent memory files/directories, without asking (the CLI asks: :func:`run_clear_memory`).

    Each memory type expands to one or more glob patterns under the data home
    (see :data:`MEMORY_PATHS`). An agent's stores are matched in every agent home
    (:data:`AGENT_HOMES`); every matched file or directory is removed.

    Args:
        memory_types: Comma-separated memory types or 'all'. Types listed in
            :data:`MEMORY_PAIRS` cannot be cleared alone — requesting one pulls
            in its partner (D2), because clearing half a pair leaves dangling
            cross-references.
        home_dir: Base data directory (default: the data home, ``~/.maxim`` or ``MAXIM_DATA_HOME``).

    Returns:
        Dict mapping memory type to success (True if at least one target was
        cleared, False if none were found). May contain types the caller did
        NOT request: a paired type pulls in its partner (see
        :data:`MEMORY_PAIRS`), and the partner appears in this dict too.
    """
    from pathlib import Path
    from maxim.utils.paths import data_home

    types, unknown = _types_to_clear(memory_types)
    for mem_type in unknown:
        print(f"Unknown memory type: {mem_type}", file=sys.stderr)
        print(f"Available types: {', '.join(MEMORY_PATHS.keys())}", file=sys.stderr)
    base = Path(home_dir) if home_dir is not None else data_home()
    results = _delete_planned(plan_clear_memory(types, [base]))
    results.update(dict.fromkeys(unknown, False))
    return results


def explicit_flag(raw_argv: Any, flag: str) -> bool:
    """Whether ``flag`` was passed explicitly, as ``flag X`` or ``flag=X`` (an abbreviation is not explicit)."""
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in (raw_argv or []))


def run_clear_memory(
    memory_types: str,
    *,
    home_dir: str,
    home_dir_explicit: bool,
    assume_yes: bool,
    stdin_is_tty: bool | None = None,
    input_fn: Any = input,
) -> int:
    """``maxim --clear-memory``: list what will be deleted, confirm, delete; the exit code (#1337).

    It clears the data home, and ``home_dir``'s stores too when the operator named ``--home-dir`` (the robot
    path's ``MaximAgent`` persists under it, default ``./data``). Unnamed, a ``./data`` holding memory is only
    warned about, with its paths. It asks y/N at a terminal; off one it refuses unless ``assume_yes``.
    Returns 0 when done or nothing matched, 1 when declined or a deletion failed, 2 for an unknown type or a
    refusal without ``--yes`` (nothing is deleted then).
    """
    from pathlib import Path
    from maxim.utils.paths import data_home

    types, unknown = _types_to_clear(memory_types)
    if unknown or not types:
        print(f"Unknown memory type(s): {', '.join(unknown) or repr(memory_types)}", file=sys.stderr)
        print(f"Available types: {', '.join(MEMORY_PATHS.keys())}", file=sys.stderr)
        return 2
    # Both absolute, so every printed path says where it is
    home = data_home().expanduser().resolve()
    bases = [home]
    other = Path(home_dir).expanduser().resolve()
    if home_dir_explicit and other != home:
        bases.append(other)
    plan = plan_clear_memory(types, bases)
    if not home_dir_explicit and other != home:
        unnamed = [p for paths in plan_clear_memory(types, [other]).values() for p in paths]
        if unnamed:
            print(
                f"Warning: {other} also holds memory (the robot path's --home-dir); not clearing it:",
                file=sys.stderr,
            )
            for p in unnamed:
                print(f"    {p}", file=sys.stderr)
            print(f"  To clear it too, pass --home-dir {home_dir}", file=sys.stderr)
    planned = [p for paths in plan.values() for p in paths]
    # The listing and the question go to stderr: with stdout redirected, a prompt there would wait unseen.
    print(f"Clearing memory ({memory_types}) under {', '.join(str(b) for b in bases)}", file=sys.stderr)
    if not planned:
        print("Nothing to clear.", file=sys.stderr)
        return 0
    for p in planned:
        print(f"    {p}", file=sys.stderr)
    if not assume_yes:
        if not (sys.stdin.isatty() if stdin_is_tty is None else stdin_is_tty):
            print(
                f"Refusing to delete {len(planned)} path(s) without confirmation: re-run with --yes.", file=sys.stderr
            )
            return 2
        print(f"Delete these {len(planned)} path(s)? [y/N] ", end="", file=sys.stderr, flush=True)
        try:
            answer = input_fn("")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled; nothing deleted.", file=sys.stderr)
            return 1
    results = _delete_planned(plan)
    failed = sum(1 for p in planned if os.path.lexists(p))
    print(f"Cleared {sum(results.values())}/{len(results)} memory type(s).")
    return 1 if failed else 0


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
