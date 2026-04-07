from __future__ import annotations

# CRITICAL: Detect Blackwell GPU and force CPU-only mode BEFORE any other imports
# Blackwell GPUs (RTX 50 series, sm_120 / compute capability 12.0) cause GStreamer crashes:
#
# The reachy_mini SDK uses WebRTC/GStreamer for video streaming, and GLib.MainLoop
# crashes in native code when CUDA is active on Blackwell GPUs. This appears to be
# a driver/GStreamer incompatibility that cannot be worked around.
#
# Solution: Force CPU-only mode by hiding all GPUs from CUDA
# See: github_issue.md for details on reporting this to pollen-robotics
import os
import subprocess
import sys

_blackwell_detected = False

# Performance optimization: Check cache first to avoid subprocess on every startup
# Cache is stored in env var (persists for shell session) or temp file
_BLACKWELL_CACHE_ENV = "_MAXIM_BLACKWELL_CHECKED"
_cached_result = os.environ.get(_BLACKWELL_CACHE_ENV)

# Blackwell CUDA policy: by default, keep CUDA visible for LLM backends
# (llama-cpp-python, torch) and only disable CUDA inside GStreamer (GST_CUDA_NO_CUDA=1)
# to avoid the reachy-mini media-pipeline segfault.
#
# Opt-out: set MAXIM_BLACKWELL_HIDE_CUDA=1 to hide CUDA from the entire process
# (CPU-only mode). Use this if you hit a GStreamer crash despite the GST guard.
_hide_cuda = os.environ.get("MAXIM_BLACKWELL_HIDE_CUDA", "").strip().lower() in (
    "1",
    "true",
    "t",
    "yes",
    "y",
    "on",
)


def _apply_blackwell_guards(*, hide_cuda: bool) -> None:
    """Apply GStreamer guards, and optionally hide CUDA from all libraries."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["GST_CUDA_NO_CUDA"] = "1"
    os.environ.setdefault("REACHY_MEDIA_BACKEND", "default")
    if hide_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


if _cached_result == "yes":
    # Cached: Blackwell was detected previously
    _blackwell_detected = True
    _apply_blackwell_guards(hide_cuda=_hide_cuda)
elif _cached_result == "no":
    # Cached: No Blackwell GPU, skip detection
    pass
else:
    # No cache - run detection (with shorter 500ms timeout)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=0.5,  # Reduced from 2s to 500ms
        )
        if result.returncode == 0:
            gpu_names = result.stdout.strip().lower()
            if "rtx 50" in gpu_names or "5080" in gpu_names or "5090" in gpu_names:
                _blackwell_detected = True
                _apply_blackwell_guards(hide_cuda=_hide_cuda)
                os.environ[_BLACKWELL_CACHE_ENV] = "yes"  # Cache for next run
                if _hide_cuda:
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
                os.environ[_BLACKWELL_CACHE_ENV] = "no"  # Cache: no Blackwell
        else:
            os.environ[_BLACKWELL_CACHE_ENV] = "no"  # No nvidia-smi or no GPU
    except subprocess.TimeoutExpired:
        os.environ[_BLACKWELL_CACHE_ENV] = "no"  # Timeout = assume no NVIDIA GPU
    except Exception:
        pass  # nvidia-smi not found, no NVIDIA GPU

# Import everything else
import argparse
import logging
import time
from collections.abc import Sequence

from maxim.cli_parser import _build_parser
from maxim.utils.data_management import build_home
from maxim.utils.logging import configure_logging, log_exception


def _normalize_epoch_value(value: object) -> int:
    try:
        epochs = int(value)
    except Exception:
        return 0
    return epochs if epochs > 0 else 0


def _normalize_args(args: argparse.Namespace) -> None:
    audio_raw = str(getattr(args, "audio", "true")).strip().lower()
    if audio_raw in ("1", "true", "t", "yes", "y", "on"):
        args.audio = True
    elif audio_raw in ("0", "false", "f", "no", "n", "off"):
        args.audio = False
    else:
        raise SystemExit(f"Invalid --audio value: {args.audio!r} (expected True/False)")

    interactive_raw = str(getattr(args, "interactive", "true")).strip().lower()
    if interactive_raw in ("1", "true", "t", "yes", "y", "on"):
        args.interactive = True
    elif interactive_raw in ("0", "false", "f", "no", "n", "off"):
        args.interactive = False
    else:
        raise SystemExit(f"Invalid --interactive value: {args.interactive!r} (expected True/False)")

    if str(getattr(args, "mode", "active")).strip().lower() == "sleep":
        args.audio = True
    args.epochs = _normalize_epoch_value(getattr(args, "epochs", 0))

    language_model = getattr(args, "language_model", None)
    if language_model is not None:
        from maxim.models.language.router import list_llm_profiles, normalize_llm_profile

        selected = normalize_llm_profile(language_model)
        if selected:
            available = list_llm_profiles()
            if available and selected not in available:
                opts = ", ".join(available)
                # Check if they're trying to use a cloud provider
                cloud_hints = {
                    "claude": ("ANTHROPIC_API_KEY", "pip install anthropic"),
                    "anthropic": ("ANTHROPIC_API_KEY", "pip install anthropic"),
                    "sonnet": ("ANTHROPIC_API_KEY", "pip install anthropic"),
                    "opus": ("ANTHROPIC_API_KEY", "pip install anthropic"),
                    "haiku": ("ANTHROPIC_API_KEY", "pip install anthropic"),
                    "gpt": ("OPENAI_API_KEY", "pip install openai"),
                    "openai": ("OPENAI_API_KEY", "pip install openai"),
                }
                hint = ""
                for keyword, (env_var, install) in cloud_hints.items():
                    if keyword in language_model.lower():
                        has_key = bool(os.environ.get(env_var))
                        hint = (
                            f"\n\n  To use cloud models ({keyword}):\n"
                            f"    1. {install}\n"
                            f"    2. export {env_var}=<your-key>\n"
                            f"    3. Add the profile to data/util/llm.json"
                        )
                        if has_key:
                            hint += f"\n\n  {env_var} is set, but the profile '{language_model}' isn't in llm.json."
                        break
                raise SystemExit(f"Unknown --language-model {language_model!r}. Available: {opts}{hint}")
            os.environ["MAXIM_LLM_PROFILE"] = selected
        args.language_model = selected

    # ── Cloud provider CLI flags ──────────────────────────────────────────
    cloud_fallback = getattr(args, "cloud_fallback", None)
    cloud_lane = getattr(args, "cloud_lane", None)
    cloud_budget = getattr(args, "cloud_budget", None)

    if cloud_fallback or cloud_lane:
        from maxim.models.language.config import _BUILTIN_PROFILES, normalize_llm_profile

        # Resolve and validate each cloud model reference
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

        # Auto-enable cloud dispatch and set sensible defaults
        os.environ.setdefault("MAXIM_LLM_CLOUD_ENABLED", "1")
        os.environ.setdefault("MAXIM_LLM_REDACTION_POLICY", "standard")
        current_max = int(os.environ.get("MAXIM_MAX_CLOUD_LANES", "0"))
        needed = sum(1 for x in [cloud_fallback, cloud_lane] if x)
        if current_max < needed:
            os.environ["MAXIM_MAX_CLOUD_LANES"] = str(needed)

    segmentation_model = getattr(args, "segmentation_model", None)
    if segmentation_model is not None:
        from maxim.models.vision.registry import list_segmentation_models, normalize_segmentation_model

        selected = normalize_segmentation_model(segmentation_model) or "rtm"
        available = list_segmentation_models()
        if available and selected not in available:
            opts = ", ".join(available)
            raise SystemExit(f"Unknown --segmentation-model {segmentation_model!r}. Available: {opts}")
        os.environ["MAXIM_SEGMENTATION_MODEL"] = selected
        args.segmentation_model = selected


def _gpu_available() -> bool:
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


def _check_gpu_status(logger: logging.Logger) -> None:
    """Check and log GPU availability for TensorFlow and PyTorch.

    Logs detailed information about:
    - Whether GPUs are detected
    - GPU names and memory
    - CPU fallback status
    """
    import os

    # Check if GPU is globally disabled
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible == "":
        logger.warning('GPU acceleration disabled (CUDA_VISIBLE_DEVICES="")')
        logger.info("Running in CPU-only mode")
        return

    # Check TensorFlow GPU (may be empty if Blackwell detected)
    tf_gpus = []
    tf_gpu_info = []
    tf_on_cpu = False
    try:
        import tensorflow as tf

        # Check visible devices (respects set_visible_devices configuration)
        tf_gpus = tf.config.get_visible_devices("GPU")
        if not tf_gpus and _blackwell_detected:
            tf_on_cpu = True  # TensorFlow GPU disabled for Blackwell
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

    # Check PyTorch GPU
    torch_gpus = 0
    torch_gpu_info = []
    try:
        import torch

        if torch.cuda.is_available():
            torch_gpus = torch.cuda.device_count()
            for i in range(torch_gpus):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                    torch_gpu_info.append(f"{gpu_name} ({gpu_mem:.1f} GB)")
                except Exception:
                    torch_gpu_info.append(f"GPU {i}")
    except Exception:
        pass

    # Log status
    if torch_gpus:
        logger.info("✅ GPU acceleration enabled (PyTorch)")
        logger.info(f"   PyTorch detected {torch_gpus} GPU(s):")
        for i, info in enumerate(torch_gpu_info):
            logger.info(f"     [{i}] {info}")

        if tf_on_cpu:
            logger.info("   TensorFlow: CPU mode (Blackwell GPU not supported by TF 2.20)")
        elif tf_gpus:
            logger.info(f"   TensorFlow detected {len(tf_gpus)} GPU(s)")
    elif tf_gpus:
        logger.info("✅ GPU acceleration enabled (TensorFlow)")
        logger.info(f"   TensorFlow detected {len(tf_gpus)} GPU(s):")
        for i, info in enumerate(tf_gpu_info):
            logger.info(f"     [{i}] {info}")
    else:
        if _blackwell_detected:
            logger.info("Running in CPU-only mode (Blackwell GPU workaround)")
            logger.info("   RTX 5080/5090 detected but CUDA disabled to avoid GStreamer crash")
            logger.info("   See github_issue.md for details on this reachy_mini SDK issue")
        else:
            logger.warning("No GPU detected - running in CPU-only mode")
            logger.info("   For GPU support, ensure:")
            logger.info("   - NVIDIA drivers are installed (570+)")
            logger.info("   - CUDA-compatible GPU is available")


def _configure_cpu_fallback_model(logger: logging.Logger, home_dir: str = "data") -> None:
    """Configure a smaller LLM model for CPU-only inference when no GPU is available.

    Sets environment variables to use SmolLM 1.7B with CPU inference, which is
    suitable for systems with limited resources (e.g., 24GB unified memory).

    Note: Skips fallback if using llama_cpp backend, which has native Metal support on macOS.
    """
    import json
    import os

    # Check if we're using llama_cpp backend (has native Metal support on macOS)
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
        # llama.cpp has native Metal support on macOS - no fallback needed
        logger.info("Using llama.cpp backend with native Metal GPU support")
        return

    logger.warning(
        "No GPU detected; falling back to smaller model (smollm-1.7b-instruct) "
        "with CPU inference. Performance may be reduced."
    )
    os.environ.setdefault("MAXIM_LLM_PROFILE", "smollm-1.7b-instruct")
    os.environ.setdefault("MAXIM_LLM_N_GPU_LAYERS", "0")


def _reexec_with_mode(args: argparse.Namespace, *, mode: str) -> None:
    mode = str(mode or "").strip().lower()
    if not mode:
        return

    audio_flag = bool(getattr(args, "audio", True))
    if mode == "sleep":
        audio_flag = True

    epochs_value = _normalize_epoch_value(getattr(args, "epochs", 0))
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


def _clear_python_cache(base_dir: str | None = None) -> int:
    """Clear Python bytecode cache (__pycache__ directories and .pyc files).

    Returns the number of cache directories removed.
    """
    import shutil
    from pathlib import Path

    if base_dir is None:
        # Default to src/maxim directory
        base_dir = str(Path(__file__).parent)

    base_path = Path(base_dir)
    removed = 0

    # Remove __pycache__ directories
    for cache_dir in base_path.rglob("__pycache__"):
        try:
            shutil.rmtree(cache_dir)
            removed += 1
        except Exception:
            pass

    # Remove any stray .pyc files
    for pyc_file in base_path.rglob("*.pyc"):
        try:
            pyc_file.unlink()
        except Exception:
            pass

    return removed


# Memory file paths for --clear-memory
MEMORY_PATHS = {
    "focus": "data/util/focus_learner.json",
    "bounds": "data/util/workspace_bounds.json",
    "escalation": "data/util/escalation_learning.json",
    "fear": "data/util/fear_learning.json",
    "threshold": "data/util/adaptive_thresholds.json",
    "nac": "data/util/nac_state.json",
    "scn": "data/util/scn_state.json",
    "hippo": "data/util/hippocampus.json",
    "pain": "data/util/pain_detector.json",
    "semantic": "data/util/semantic_embeddings.npz",  # Phase 4 semantic embeddings
    "statistician": "data/util/statistician_state.json",
    "atl": "data/util/atl_state.json",
    "cross_layer": "data/util/cross_layer_graph.json",
    "planning": "data/planning",
}


def _clear_memory(memory_types: str, home_dir: str = "data") -> dict[str, bool]:
    """Clear persistent memory files.

    Args:
        memory_types: Comma-separated memory types or 'all'.
        home_dir: Base data directory.

    Returns:
        Dict mapping memory type to success (True if cleared, False if not found).
    """
    from pathlib import Path

    results: dict[str, bool] = {}

    # Parse types
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

        # Get path relative to home_dir
        rel_path = MEMORY_PATHS[mem_type]
        # Extract just the filename portion (after data/util/)
        if rel_path.startswith("data/"):
            rel_path = rel_path[5:]  # Remove "data/" prefix
        full_path = Path(home_dir) / rel_path

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


def main(argv: Sequence[str] | None = None) -> int:
    from maxim.utils.last_run import (
        should_save,
        save_last_run,
        load_last_run,
        load_all_runs,
        clear_last_run,
        format_all_runs,
    )

    raw_argv = list(argv) if argv is not None else sys.argv[1:]

    # Stage A observability: print loud warning if trace flags are active so
    # users don't leave them on accidentally (log volume + request-id exposure).
    from maxim.models.language.mesh_trace import print_startup_warning_if_enabled

    print_startup_warning_if_enabled()

    # Subcommand dispatch — intercepts positional subcommands before argparse.
    if raw_argv and raw_argv[0] == "tunnel":
        from maxim.tunnel import run_tunnel_subcommand

        return run_tunnel_subcommand(raw_argv[1:])
    if raw_argv and raw_argv[0] == "doctor":
        from maxim.doctor import run_doctor_subcommand

        return run_doctor_subcommand(raw_argv[1:])
    if raw_argv and raw_argv[0] == "peer":
        # `peer connect/show/forget` go to the peer config module;
        # `peer test` is a diagnostic that lives with `doctor`.
        peer_action = raw_argv[1] if len(raw_argv) > 1 else ""
        if peer_action in ("connect", "show", "forget", "update", "restart", "llm", "version", "logs"):
            from maxim.peer import run_peer_connect_subcommand

            return run_peer_connect_subcommand(raw_argv[1:])
        from maxim.doctor import run_peer_subcommand

        return run_peer_subcommand(raw_argv[1:])

    # Handle --last, --show-last, --clear-last before full parse
    if "--show-last" in raw_argv:
        print(format_all_runs())
        return 0
    if "--clear-last" in raw_argv:
        if clear_last_run():
            print("  Saved runs cleared.")
        else:
            print("  No saved runs to clear.")
        return 0
    if "--last" in raw_argv:
        # Parse just --last N
        idx_pos = raw_argv.index("--last")
        n = 1
        if idx_pos + 1 < len(raw_argv):
            try:
                n = int(raw_argv[idx_pos + 1])
            except ValueError:
                pass
        last = load_last_run(n)
        if last is None:
            runs = load_all_runs()
            if not runs:
                print("  No saved runs. Run a simulation first.")
            else:
                print(f"  No run at index {n}. Available:")
                print(format_all_runs())
            return 1
        print(f"  Re-running: {last['command']}")
        raw_argv = last["args"]

    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    _normalize_args(args)

    # ── Early leader proxy bootstrap ─────────────────────────────────
    # Start the LeaderProxy as early as possible so it's always reachable
    # via the Cloudflare tunnel, regardless of what mode maxim enters
    # (sim, interactive, agentic, etc.). This ensures `maxim peer update`,
    # `maxim peer restart`, `maxim peer logs`, and `maxim peer version`
    # always work. The proxy is a daemon thread — it dies with the process.
    detected_role = None
    try:
        from maxim.runtime.leader_mode import detect_role

        detected_role = detect_role()
        if detected_role.role == "leader":
            import os as _os

            _os.environ.setdefault("MAXIM_ALLOW_REMOTE_UPDATE", "1")
            from maxim.runtime.leader_proxy import start_leader_proxy

            _api_key = None
            try:
                from maxim.tunnel.keys import read_key

                _api_key = read_key()
            except Exception:
                print("[leader-boot] WARNING: could not read API key — proxy will run without auth")
            _proxy = start_leader_proxy(api_key=_api_key, bind_host=detected_role.bind_host)
            if _proxy is None:
                print("[leader-boot] WARNING: LeaderProxy failed to start (port in use?)")
    except Exception as _e:
        import traceback as _tb

        print(f"[leader-boot] WARNING: early proxy boot failed: {_e}")
        _tb.print_exc()

    # Save this invocation if it's meaningful
    if should_save(raw_argv):
        save_last_run(raw_argv)

    # Clear Python cache if requested
    if bool(getattr(args, "clear_cache", False)):
        removed = _clear_python_cache()
        print(f"Cleared {removed} __pycache__ director{'y' if removed == 1 else 'ies'}.", file=sys.stderr)

    # Clear memory if requested
    clear_memory = getattr(args, "clear_memory", None)
    if clear_memory is not None:
        print(f"Clearing memory ({clear_memory})...")
        results = _clear_memory(clear_memory, args.home_dir)
        cleared = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"Cleared {cleared}/{total} memory file(s).")
        return 0  # Exit after clearing

    # Scenario generation if requested
    gen_description = getattr(args, "generate_simulation", None)
    if gen_description is not None:
        from pathlib import Path
        from maxim.simulation.simulation_generator import generate_scenario

        output_path = getattr(args, "output", None)
        output = Path(output_path) if output_path else None
        llm_profile = str(getattr(args, "language_model", "") or "").strip() or None

        try:
            yaml_str = generate_scenario(gen_description, output_path=output, llm_profile=llm_profile)
            if output is None:
                print(yaml_str)
        except Exception as e:
            print(f"Error generating scenario: {e}")
            sys.exit(1)
        sys.exit(0)

    # ── Top-level --benchmark command ─────────────────────────────────
    _benchmark_arg = getattr(args, "benchmark", None)
    if _benchmark_arg is not None:
        from maxim.simulation.benchmark import BenchmarkRunner

        models_raw = getattr(args, "models", None)
        if not models_raw:
            print("  Error: --models is required for --benchmark")
            print("  Example: maxim --benchmark tier1 --models mistral-7b,qwen2.5-14b")
            sys.exit(1)

        models = [m.strip() for m in models_raw.split(",") if m.strip()]
        campaign = getattr(args, "campaign", None)
        if not campaign:
            # Default campaign suite based on tier
            _tier_campaigns = {
                "tier1": "scenarios/benchmarks/cognitive_suite.yaml",
                "tier2": "scenarios/benchmarks/biosystem_suite.yaml",
                "tier3": "scenarios/benchmarks/embodiment_suite.yaml",
                "all": "scenarios/benchmarks/cognitive_suite.yaml",
            }
            campaign = _tier_campaigns.get(_benchmark_arg, _tier_campaigns["all"])

        runner = BenchmarkRunner(
            models=models,
            suite_path=campaign,
            runs=getattr(args, "runs", 1) or 1,
            output_dir=getattr(args, "benchmark_output", "data/benchmarks"),
            baseline_path=getattr(args, "baseline", None),
            persona=getattr(args, "sim_persona", "campaign") or "campaign",
            max_turns=50,
            response_timeout=60.0,
            debug=bool(getattr(args, "debug", "")),
        )

        report = runner.run()
        print(report.summary_table())

        report_dir = runner.save_report(report)
        print(f"\n  Report saved: {report_dir}\n")

        if getattr(args, "write_paper", False):
            print("  Generating comparative paper...")
            paper_path = runner.write_paper(report, report_dir)
            if paper_path:
                print(f"  Paper saved: {paper_path}\n")
            else:
                print("  Paper generation failed (LLM unavailable)\n")

        all_passed = all(mr.passed for mr in report.results.values())
        sys.exit(0 if all_passed else 1)

    # Validate simulation-only flags aren't used without --sim agent/research
    sim_path = getattr(args, "sim", None)
    _sim_mode = str(sim_path).strip().lower() if sim_path is not None else ""
    _is_sim_mode = _sim_mode in ("agent", "research")
    if not _is_sim_mode:
        if getattr(args, "sim_goal", None) is not None:
            print("Error: --goal / --sim-goal requires --sim agent or --sim research.")
            print('  Usage: maxim --sim agent --goal "test safety" --persona adversarial')
            print('         maxim --sim research --goal "hippocampal recall" --campaign <yaml>')
            sys.exit(1)
        if getattr(args, "sim_persona", "adversarial") != "adversarial":
            print("Error: --persona / --sim-persona requires --sim agent (simulation mode).")
            print('  Usage: maxim --sim agent --goal "test safety" --persona adversarial')
            sys.exit(1)
        if getattr(args, "resume_sim", None) is not None:
            print("Error: --resume-sim requires --sim agent (simulation mode).")
            print('  Usage: maxim --sim agent --goal "continue" --resume-sim SESSION_ID')
            sys.exit(1)

    # ── DM Campaign mode (--dm flag) ────────────────────────────────────────
    dm_path = getattr(args, "dm", None)
    if dm_path is not None:
        from pathlib import Path

        from maxim.simulation.dm_schema import load_campaign, validate_campaign
        from maxim.simulation.orchestrator import start_simulation_mode

        dm_path = Path(dm_path)
        if not dm_path.exists():
            print(f"Error: campaign file not found: {dm_path}")
            sys.exit(1)

        campaign = load_campaign(dm_path)
        errors = validate_campaign(campaign)
        if errors:
            print(f"Campaign validation failed ({len(errors)} errors):")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)

        _debug_raw = getattr(args, "debug", None)
        debug = bool(_debug_raw)
        result = start_simulation_mode(
            goal=f"dm:{campaign.name}",
            persona="dungeon_master",
            debug=debug,
            no_sim_env=bool(getattr(args, "no_sim_env", False)),
            sandbox_backend=getattr(args, "sandbox_backend", "auto"),
            dm_campaign=campaign,
        )
        sys.exit(0 if result.finish_reason != "error" else 1)

    # Simulation mode if requested — runs full agentic pipeline with fake percepts
    if sim_path is not None:
        import json as _json
        from pathlib import Path

        from maxim.simulation.scenario_source import ScenarioSource
        from maxim.simulation.sinks import RecordingSink
        from maxim.simulation.validation import validate_expectations

        # Parse --debug subsystem selection
        _debug_raw = getattr(args, "debug", None)
        _debug_all = _debug_raw == "all"
        _debug_subs = set()
        if _debug_raw and _debug_raw != "all":
            _debug_subs = {s.strip().lower() for s in _debug_raw.split(",")}
        if _debug_all or "hippo" in _debug_subs:
            os.environ["MAXIM_HIPPO_TRACE"] = "1"
        if _debug_all or "nac" in _debug_subs:
            os.environ["MAXIM_NAC_TRACE"] = "1"
        if _debug_all or "atl" in _debug_subs:
            os.environ["MAXIM_ATL_TRACE"] = "1"

        # -- Detect sim mode from --sim value ---------------------------------
        _sim_val = str(sim_path).strip()
        _sim_val_lower = _sim_val.lower()

        # New unified detection:
        # - "agent", "research", "benchmark" → legacy aliases
        # - ends with .yaml/.yml → YAML campaign/scenario
        # - "interactive" or empty → REPL
        # - anything else → goal string for generative campaign
        _is_yaml = _sim_val.endswith((".yaml", ".yml"))
        _is_legacy_agent = _sim_val_lower == "agent"
        _is_legacy_research = _sim_val_lower == "research"
        _is_legacy_benchmark = _sim_val_lower == "benchmark"
        _is_interactive = _sim_val_lower == "interactive"
        _is_dm = _sim_val_lower == "dm"
        _is_goal_string = not (
            _is_yaml or _is_legacy_agent or _is_legacy_research or _is_legacy_benchmark or _is_interactive or _is_dm
        )

        # If --research flag is set with a goal string, use research mode
        _wants_research = getattr(args, "research", False)
        # If --goal is set explicitly, it overrides the --sim value as goal
        _explicit_goal = getattr(args, "sim_goal", None)

        # ── Generative campaign mode (new default for goal strings) ──
        if _is_goal_string and not _is_legacy_agent:
            goal = _explicit_goal or _sim_val
            if _wants_research:
                # Generative + research report
                from maxim.simulation.research_orchestrator import start_research_mode

                campaign = getattr(args, "campaign", None)
                language_model = str(getattr(args, "language_model", "") or "").strip() or None
                result = start_research_mode(
                    goal=goal,
                    campaign=campaign,
                    language_model=language_model,
                    aut_model=getattr(args, "aut_model", None),
                    debug=bool(_debug_raw),
                    sandbox_backend=getattr(args, "sandbox_backend", "auto"),
                )
                sys.exit(0 if result.review_verdict != "reject" else 1)
            else:
                # Pure generative campaign — use the orchestrator with generative runner
                # For now, delegate to start_simulation_mode with the goal
                # The generative runner will be wired in as the default persona
                from maxim.simulation.orchestrator import start_simulation_mode

                persona = getattr(args, "sim_persona", "campaign")
                debug = bool(_debug_raw)
                resume_sim = getattr(args, "resume_sim", None)

                result = start_simulation_mode(
                    goal=goal,
                    persona=persona,
                    debug=debug,
                    resume_session=resume_sim,
                    continuous=bool(getattr(args, "continuous", False)),
                    no_sim_env=bool(getattr(args, "no_sim_env", False)),
                    sandbox_backend=getattr(args, "sandbox_backend", "auto"),
                    sandbox_image=getattr(args, "sandbox_image", "python:3.12-slim"),
                    sandbox_network=getattr(args, "sandbox_network", "none"),
                    aut_model=getattr(args, "aut_model", None),
                )
                sys.exit(0 if result.finish_reason != "error" else 1)

        # ── DM campaign mode ──
        if _is_dm:
            # Campaign path via --campaign or --sim-goal
            campaign_path = getattr(args, "campaign", None) or getattr(args, "sim_goal", None)
            if not campaign_path:
                print("Error: --sim dm requires a campaign YAML path")
                print("Usage: maxim --sim dm --campaign scenarios/campaigns/heist_v1.yaml")
                print("   or: maxim --sim dm --goal scenarios/campaigns/heist_v1.yaml")
                sys.exit(1)
            # Handle glob patterns (campaign flag may be a list)
            if isinstance(campaign_path, list):
                campaign_path = campaign_path[0]

            from maxim.simulation.dm_schema import load_campaign, validate_campaign
            from maxim.simulation.orchestrator import start_simulation_mode

            campaign = load_campaign(campaign_path)
            errors = validate_campaign(campaign)
            if errors:
                print(f"Campaign validation failed ({len(errors)} errors):")
                for e in errors:
                    print(f"  - {e}")
                sys.exit(1)

            debug = bool(_debug_raw)
            result = start_simulation_mode(
                goal=f"dm:{campaign.name}",
                persona="dungeon_master",
                debug=debug,
                no_sim_env=bool(getattr(args, "no_sim_env", False)),
                sandbox_backend=getattr(args, "sandbox_backend", "auto"),
                dm_campaign=campaign,
            )
            sys.exit(0 if result.finish_reason != "error" else 1)

        # ── Legacy: agent mode (deprecated alias) ──
        _sim_agent = _is_legacy_agent
        if _sim_agent:
            from maxim.simulation.orchestrator import start_simulation_mode

            goal = getattr(args, "sim_goal", None) or "test the agent's capabilities"
            persona = getattr(args, "sim_persona", "adversarial")
            debug = bool(_debug_raw)
            resume_sim = getattr(args, "resume_sim", None)

            result = start_simulation_mode(
                goal=goal,
                persona=persona,
                debug=debug,
                resume_session=resume_sim,
                continuous=bool(getattr(args, "continuous", False)),
                no_sim_env=bool(getattr(args, "no_sim_env", False)),
                sandbox_backend=getattr(args, "sandbox_backend", "auto"),
                sandbox_image=getattr(args, "sandbox_image", "python:3.12-slim"),
                sandbox_network=getattr(args, "sandbox_network", "none"),
                aut_model=getattr(args, "aut_model", None),
            )
            sys.exit(0 if result.finish_reason != "error" else 1)

        # Check for benchmark mode (multi-model comparison)
        _sim_benchmark = str(sim_path).strip().lower() == "benchmark"
        if _sim_benchmark:
            from maxim.simulation.benchmark import BenchmarkRunner

            models_raw = getattr(args, "models", None)
            if not models_raw:
                print("  Error: --models is required for --sim benchmark")
                print("  Example: maxim --sim benchmark --models mistral-7b,qwen2.5-14b --campaign suite.yaml")
                sys.exit(1)

            models = [m.strip() for m in models_raw.split(",") if m.strip()]
            campaign = getattr(args, "campaign", None)
            if not campaign:
                print("  Error: --campaign is required for --sim benchmark")
                print(
                    "  Example: maxim --sim benchmark --models mistral-7b --campaign scenarios/benchmarks/cognitive_suite.yaml"
                )
                sys.exit(1)

            runner = BenchmarkRunner(
                models=models,
                suite_path=campaign,
                runs=getattr(args, "runs", 1) or 1,
                output_dir=getattr(args, "benchmark_output", "data/benchmarks"),
                baseline_path=getattr(args, "baseline", None),
                persona=getattr(args, "sim_persona", "campaign") or "campaign",
                max_turns=50,
                response_timeout=60.0,
                debug=bool(_debug_raw),
            )

            report = runner.run()
            print(report.summary_table())

            # Save report
            report_dir = runner.save_report(report)
            print(f"\n  Report saved: {report_dir}\n")

            # Optional: generate comparative paper
            if getattr(args, "write_paper", False):
                print("  Generating comparative paper...")
                paper_path = runner.write_paper(report, report_dir)
                if paper_path:
                    print(f"  Paper saved: {paper_path}\n")
                else:
                    print("  Paper generation failed (LLM unavailable)\n")

            # Exit with 0 if all models passed, 1 if any failed
            all_passed = all(mr.passed for mr in report.results.values())
            sys.exit(0 if all_passed else 1)

        # Check for research mode (legacy alias — deprecated, use --research flag)
        _sim_research = _is_legacy_research
        if _sim_research:
            from maxim.simulation.research_orchestrator import start_research_mode

            goal = getattr(args, "sim_goal", None) or "investigate the research question"
            debug = bool(_debug_raw)
            campaign = getattr(args, "campaign", None)
            language_model = str(getattr(args, "language_model", "") or "").strip() or None

            result = start_research_mode(
                goal=goal,
                campaign=campaign,
                language_model=language_model,
                aut_model=getattr(args, "aut_model", None),
                debug=debug,
                sandbox_backend=getattr(args, "sandbox_backend", "auto"),
            )
            sys.exit(0 if result.review_verdict != "reject" else 1)

        # ── Auto-detect DM campaigns from YAML ──
        # If --sim points to a YAML with a top-level 'campaign:' key, treat as DM
        if _is_yaml and not _is_dm:
            _yaml_path = Path(sim_path).resolve()
            if _yaml_path.exists():
                try:
                    import yaml as _yaml

                    with open(_yaml_path) as _yf:
                        _raw = _yaml.safe_load(_yf)
                    if isinstance(_raw, dict) and "campaign" in _raw and "encounters" in _raw:
                        _is_dm = True
                        # Re-route to DM handler
                        from maxim.simulation.dm_schema import load_campaign, validate_campaign
                        from maxim.simulation.orchestrator import start_simulation_mode

                        dm_campaign = load_campaign(_yaml_path)
                        errors = validate_campaign(dm_campaign)
                        if errors:
                            print(f"Campaign validation failed ({len(errors)} errors):")
                            for e in errors:
                                print(f"  - {e}")
                            sys.exit(1)

                        debug = bool(_debug_raw)
                        result = start_simulation_mode(
                            goal=f"dm:{dm_campaign.name}",
                            persona="dungeon_master",
                            debug=debug,
                            no_sim_env=bool(getattr(args, "no_sim_env", False)),
                            sandbox_backend=getattr(args, "sandbox_backend", "auto"),
                            dm_campaign=dm_campaign,
                        )
                        sys.exit(0 if result.finish_reason != "error" else 1)
                except Exception:
                    pass  # Not a DM campaign — fall through to normal YAML handling

        # Check for interactive mode
        if _is_interactive:
            scenario_files = []  # No files — interactive REPL handles everything
        else:
            sim_path = Path(sim_path).resolve()  # Resolve to absolute before CWD change
            if sim_path.is_dir():
                scenario_files = sorted(sim_path.glob("*.yaml")) + sorted(sim_path.glob("*.yml"))
            else:
                scenario_files = [sim_path]

        if not scenario_files and not _is_interactive:
            print(f"No scenario files found at {sim_path}")
            sys.exit(1)

        # Force agentic mode with supervised autonomy for simulation
        args.mode = "agentic"
        if not getattr(args, "autonomy", None) or args.autonomy == "planning":
            args.autonomy = "supervised"  # Let agent attempt actions; FearAgent + policy gate
        # Use a reasonable step limit so scenarios don't run forever
        if not getattr(args, "epochs", None):
            args.epochs = 200

        # Create a temporary sandbox directory within the workspace
        # All filesystem operations are confined here, destroyed after the run
        if not _is_interactive:
            # Single/batch scenario mode: set up sandbox and load scenario
            import tempfile

            sim_workspace = Path(getattr(args, "home_dir", "data")) / "sim_sandbox"
            sim_workspace.mkdir(parents=True, exist_ok=True)
            sim_tmpdir = Path(
                tempfile.mkdtemp(
                    prefix=f"sim_{time.strftime('%Y%m%d_%H%M%S')}_",
                    dir=str(sim_workspace),
                )
            )
            args._sim_original_cwd = os.getcwd()
            import atexit

            atexit.register(lambda cwd=args._sim_original_cwd: os.chdir(cwd))
            os.chdir(str(sim_tmpdir))
            _sim_debug = bool(getattr(args, "debug", False) or getattr(args, "sim_debug", False))
            if _sim_debug:
                print(f"  Simulation sandbox: {sim_tmpdir}")

            # (all_results / any_failed removed — unused; batch scenario
            # result aggregation is handled by ScenarioRunner directly)

            from maxim.simulation.sim_logger import enable_sim_logging, disable_sim_logging

            sim_log_path = str(sim_workspace / f"sim_log_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
            enable_sim_logging(log_path=sim_log_path, debug=_sim_debug)

            for scenario_file in scenario_files:
                if _sim_debug:
                    print(f"\nRunning scenario: {scenario_file.name}")
                    print(f"  Loading full agentic pipeline (autonomy={args.autonomy})...")

                source = ScenarioSource(scenario_file)
                sink = RecordingSink()

                args._sim_source = source
                args._sim_sink = sink
                args._sim_scenario_file = scenario_file
                args._sim_tmpdir = sim_tmpdir
                break  # Process one scenario, let the main loop handle it

        # Fall through to the main loop with mode="agentic"
        # The agentic block will detect args._sim_source and wire it in

    # Architecture audit if requested
    if getattr(args, "audit_architecture", False):
        from maxim.utils.audit import audit_architecture

        violations = audit_architecture()
        if violations:
            print(f"Found {len(violations)} architecture violations:")
            for v in violations:
                print(f"  {v.file}:{v.line} — {v.rule} (imports {v.imported_module})")
            sys.exit(1)
        else:
            print("No architecture violations found.")
            sys.exit(0)

    build_home(args.home_dir)
    mode = str(getattr(args, "mode", "active")).strip().lower()
    while True:
        run_id = time.strftime("%Y-%m-%d_%H%M%S")
        log_path = os.path.join(args.home_dir, "logs", f"reachy_log_{run_id}.log")

        configure_logging(args.verbosity, log_file=log_path, force=True)
        logger = logging.getLogger("maxim")

        maxim = None
        try:
            epochs_value = _normalize_epoch_value(getattr(args, "epochs", 0))
            epochs_label = "unlimited" if epochs_value <= 0 else str(epochs_value)
            logger.info(
                "Starting Maxim (robot_name=%s, home_dir=%s, timeout=%.1fs, epochs=%s, mode=%s, log=%s)",
                args.robot_name,
                args.home_dir,
                float(args.timeout),
                epochs_label,
                mode,
                log_path,
            )

            # Check and log GPU status
            _check_gpu_status(logger)

            if mode == "agentic":
                if not _gpu_available():
                    _configure_cpu_fallback_model(logger, args.home_dir)

                from maxim.agents import MaximAgent
                from maxim.agents.autonomy import (
                    AutonomyController,
                    AutonomyLevel,
                    SafetyConstraints,
                    SupervisionPolicy,
                )
                from maxim.agents.llm_worker import LLMWorker
                from maxim.environment import ReachyEnv
                from maxim.runtime import (
                    build_decision_engine,
                    build_executor,
                    build_evaluators,
                    build_memory,
                    build_state,
                    build_tool_registry,
                )
                from maxim.runtime.agent_loop import run_agentic_loop
                from maxim.utils.structured_logging import configure_agentic_verbosity

                # Configure agentic verbosity
                # Default to --verbosity if --agentic-verbosity not specified
                agentic_verbosity = getattr(args, "agentic_verbosity", None)
                if agentic_verbosity is None:
                    agentic_verbosity = int(getattr(args, "verbosity", 1))
                else:
                    agentic_verbosity = int(agentic_verbosity)
                # Agentic console output is ON by default (unless --no-agentic-console)
                agentic_console = not bool(getattr(args, "no_agentic_console", False))
                configure_agentic_verbosity(
                    verbosity=agentic_verbosity,
                    console_output=agentic_console,
                )
                logger.info(
                    "Agentic verbosity: %d (console=%s)",
                    agentic_verbosity,
                    agentic_console,
                )

                # Determine memory persistence path
                memory_path = getattr(args, "memory_path", None)
                if memory_path is None:
                    memory_path = os.path.join(args.home_dir, "memory", "memories.json")

                # Get LLM profile
                llm_profile = str(getattr(args, "language_model", "") or "").strip()
                if not llm_profile:
                    llm_profile = "mistral-7b-instruct-v0.2"

                # Create the agentic agent
                agentic_agent = MaximAgent(
                    llm_profile=llm_profile,
                    memory_persistence_path=memory_path,
                    data_folder=args.home_dir,
                    enable_embeddings=bool(getattr(args, "enable_embeddings", False)),
                    reset_on_startup=bool(getattr(args, "reset", False)),
                )

                # Set up ResponseOutput for LLM responses
                from pathlib import Path
                from maxim.utils.response_output import ResponseOutput

                sandbox_path = Path(args.home_dir) / "sandbox"
                tts_engine = None
                speaker_fn = None

                # Set up TTS if enabled
                if getattr(args, "tts", False):
                    try:
                        from maxim.models.audio.tts import TTSEngine

                        tts_model = str(getattr(args, "tts_model", "en_US-lessac-medium"))
                        tts_engine = TTSEngine(model_name=tts_model)
                        if tts_engine.is_available:
                            logger.info("TTS enabled with model: %s", tts_model)
                        else:
                            logger.warning("TTS model not found, TTS will be disabled")
                            tts_engine = None
                    except Exception as e:
                        logger.warning("Failed to initialize TTS: %s", e)
                        tts_engine = None

                response_output = ResponseOutput(
                    sandbox_path=sandbox_path,
                    logger=logger,
                    tts_engine=tts_engine,
                    speaker_fn=speaker_fn,
                )

                # Check if internet access is enabled (default: True unless --no-internet)
                internet_enabled = not bool(getattr(args, "no_internet", False))

                # Create internet policy getter for tool registry
                def get_internet_policy():
                    from maxim.utils.internet_access import InternetAccessPolicy

                    return InternetAccessPolicy(enabled=internet_enabled)

                # Only pass policy getter if internet is enabled
                internet_policy_getter = get_internet_policy if internet_enabled else None

                # Build comms stack if enabled (--comms flag or MAXIM_COMMS_ENABLED env)
                comms_enabled = bool(getattr(args, "comms", False)) or os.environ.get(
                    "MAXIM_COMMS_ENABLED", ""
                ).lower() in ("1", "true", "yes")
                gateway = None
                if comms_enabled:
                    from maxim.runtime.bootstrap import build_comms_stack

                    gateway, _conv_manager = build_comms_stack(
                        bus=agentic_agent._bus,
                    )

                # Build MemoryHub with Hippocampus for episodic memory
                _cli_memory_hub = None
                _cli_hippocampus = None
                try:
                    from maxim.integration.memory_hub import MemoryHub
                    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
                    from maxim.decisions.nac import NAc
                    from maxim.similarity.ec import EntorhinalCortex
                    from maxim.time.scn import SCN

                    _cli_hippocampus = Hippocampus(
                        config=HippocampusConfig(
                            persistence_path=memory_path,
                        )
                    )
                    _cli_nac = NAc()
                    _cli_scn = SCN()
                    _cli_ec = EntorhinalCortex()
                    _cli_memory_hub = MemoryHub(
                        hippocampus=_cli_hippocampus,
                        scn=_cli_scn,
                        nac=_cli_nac,
                        ec=_cli_ec,
                    )
                    agentic_agent.wire_memory_hub(_cli_memory_hub)
                    logger.info("MemoryHub + Hippocampus + NAc + SCN + EC wired to MaximAgent")
                except Exception as e:
                    logger.warning("Failed to create MemoryHub: %s", e)

                # Use active mode in simulation so agent can read/write in sandbox
                _operational_mode = "active" if getattr(args, "sim", None) is not None else "passive"
                registry = build_tool_registry(
                    response_output=response_output,
                    internet_policy_getter=internet_policy_getter,
                    gateway=gateway,
                    operational_mode=_operational_mode,
                )
                executor = build_executor(registry)

                # Wrap executor with FearAgent safety gating (independent of DefaultNetwork)
                from maxim.agents.fear_agent import FearAgent
                from maxim.runtime.fear_gate import FearGatedExecutor

                fear_agent = FearAgent(
                    llm=agentic_agent._llm if hasattr(agentic_agent, "_llm") else None,
                )
                executor = FearGatedExecutor(executor, fear_agent)
                logger.info("FearGatedExecutor active — all tool calls reviewed by FearAgent")

                decision_engine = build_decision_engine()
                env = ReachyEnv(data_dir=args.home_dir)
                # In simulation mode, don't limit state steps (grace period handles termination)
                _state_max = 0 if getattr(args, "sim", None) is not None else epochs_value
                state = build_state(max_steps=_state_max)
                memory = build_memory()
                evaluators = build_evaluators()

                # Set up autonomy controller
                autonomy_level_str = str(getattr(args, "autonomy", "planning")).lower()
                initial_level = AutonomyLevel(autonomy_level_str)

                # Build allowed tools set based on internet policy (uses internet_enabled from above)
                allowed_tools = {
                    "read_file",
                    "write_file",  # Requires confirmation (see requires_confirmation below)
                    "focus_interests",
                    "track_target",
                    "maxim_command",
                    "mode_switch",
                    "speak",
                    "respond",
                    "list_directory",  # Allow directory listing
                    "glob",  # Pattern-based file search
                    "bash",  # Shell command execution (requires MAXIM_ALLOW_BASH=1)
                }

                # Add internet tools if enabled
                if internet_enabled:
                    allowed_tools.add("internet_search")
                    allowed_tools.add("http_fetch")

                # Add comms tools if gateway is available
                if gateway is not None:
                    allowed_tools.add("send_message")
                    allowed_tools.add("call_user")

                # Configure supervision policy with sensible defaults
                supervision_policy = SupervisionPolicy(
                    allowed_tools=allowed_tools,
                    forbidden_tools={"execute_file", "delete_file"},
                    min_confidence_autonomous=0.7,
                    requires_confirmation={"write_file", "bash"},  # bash requires confirmation for safety
                )

                autonomy_controller = AutonomyController(
                    initial_level=initial_level,
                    safety_constraints=SafetyConstraints(),
                    supervision_policy=supervision_policy,
                )

                # Set timed autonomy if specified
                autonomy_duration = getattr(args, "autonomy_duration", None)
                if autonomy_duration and initial_level == AutonomyLevel.AUTONOMOUS:
                    autonomy_controller.set_level(
                        AutonomyLevel.AUTONOMOUS,
                        f"CLI: timed autonomy for {autonomy_duration}s",
                        duration_seconds=autonomy_duration,
                    )

                # Set up LLM worker — LLM lives inside ExecAgent, not MaximAgent
                llm_worker = None
                llm_router = None
                # Trigger lazy LLM + router init via ExecAgent
                if hasattr(agentic_agent, "exec_agent"):
                    agentic_agent.exec_agent._ensure_llm()
                    llm_router = agentic_agent.exec_agent._ensure_router()
                if llm_router is not None:
                    # Check model file exists before warming up
                    model_path = str(getattr(llm_router.cfg, "model_path", "")).strip()
                    if model_path and not os.path.exists(model_path):
                        logger.info("Model not found at %s — attempting download...", model_path)
                        try:
                            from maxim.models.download import download_llm, LLM_MODELS

                            profile = str(
                                getattr(llm_router.cfg, "profile", "") or getattr(llm_router.cfg, "model_base", "")
                            ).strip()
                            if profile and profile in LLM_MODELS:
                                print(f"  Downloading LLM model: {profile}...")
                                if download_llm(profile):
                                    print(f"  Download complete: {profile}")
                                else:
                                    print("  Download failed. Run: ./scripts/download_models.sh --llm --enable")
                            else:
                                print("  Model not found. Run: ./scripts/download_models.sh --llm --enable")
                        except Exception as e:
                            print(f"  Auto-download failed: {e}")
                            print("  Run: ./scripts/download_models.sh --llm --enable")

                    if hasattr(llm_router, "warmup"):
                        llm_router.warmup()
                    # In simulation mode, wait for LLM to be fully loaded before starting
                    _is_sim = getattr(args, "sim", None) is not None
                    if _is_sim and hasattr(llm_router, "wait_ready"):
                        logger.info("Waiting for LLM to load (simulation mode)...")
                        if llm_router.wait_ready(timeout=120.0):
                            logger.info("LLM ready")
                        else:
                            logger.warning("LLM failed to load — simulation will use fallback responses")
                            print("  WARNING: LLM failed to load. Simulation will not produce real responses.")
                            print("  Ensure model is downloaded: ./scripts/download_models.sh --llm --enable")
                    logger.info("LLM router initialized: %s", llm_router)
                    llm_worker = LLMWorker(
                        llm_router,
                        n_ctx=getattr(llm_router, "n_ctx", 4096),
                        token_counter=(
                            llm_router.get_token_counter() if hasattr(llm_router, "get_token_counter") else None
                        ),
                    )
                    llm_worker.start()

                # Store internet access in state (uses internet_enabled from above)
                state.data["internet_access"] = internet_enabled
                state.data["autonomy_level"] = initial_level.value

                # Wire communication gateway if available
                if gateway is not None:
                    agentic_agent.wire_communication(gateway=gateway)
                    logger.info("Communication gateway wired")

                logger.info(
                    "Starting MaximAgent (memory_path=%s, embeddings=%s, reset=%s, autonomy=%s, internet=%s, comms=%s)",
                    memory_path,
                    bool(getattr(args, "enable_embeddings", False)),
                    bool(getattr(args, "reset", False)),
                    initial_level.value,
                    internet_enabled,
                    gateway is not None,
                )

                # Check for interactive simulation REPL
                if (
                    getattr(args, "sim", None) is not None
                    and str(getattr(args, "sim", "")).strip().lower() == "interactive"
                ):
                    from maxim.simulation.interactive import run_interactive_sim
                    from maxim.proprioception.pain_bus import (
                        PainBus as SimPainBus,
                        create_pain_memory_subscriber,
                    )

                    _sim_pain_bus = SimPainBus()
                    if _cli_hippocampus is not None:
                        _sim_pain_bus.subscribe(create_pain_memory_subscriber(_cli_hippocampus))

                    try:
                        run_interactive_sim(
                            agentic_agent,
                            env,
                            state,
                            memory,
                            decision_engine,
                            executor,
                            autonomy_controller=autonomy_controller,
                            llm_worker=llm_worker,
                            hippocampus=_cli_hippocampus,
                            memory_hub=_cli_memory_hub,
                            evaluators=evaluators,
                            pain_bus=_sim_pain_bus,
                            llm_profile=llm_profile,
                            sim_workspace=Path(getattr(args, "home_dir", "data")) / "sim_sandbox",
                            debug=bool(getattr(args, "debug", False) or getattr(args, "sim_debug", False)),
                        )
                    finally:
                        if llm_worker:
                            llm_worker.stop()
                    return 0

                # Check for simulation mode — wire percept_source and action_sink
                sim_source = getattr(args, "_sim_source", None)
                sim_sink = getattr(args, "_sim_sink", None)

                # In sim mode, create a PainBus for headless pain routing
                # (DefaultNetwork may not exist, so we need our own)
                if sim_source is not None:
                    from maxim.proprioception.pain_bus import (
                        PainBus as SimPainBus,
                        create_pain_memory_subscriber,
                    )

                    _sim_pain_bus = SimPainBus()
                    if _cli_hippocampus is not None:
                        _sim_pain_bus.subscribe(create_pain_memory_subscriber(_cli_hippocampus))
                        logger.info("Sim PainBus wired to hippocampus for pain memory capture")

                    args._sim_pain_bus = _sim_pain_bus
                    args._sim_hippo = _cli_hippocampus

                try:
                    run_agentic_loop(
                        agentic_agent,
                        env,
                        state,
                        memory,
                        decision_engine,
                        executor,
                        autonomy_controller=autonomy_controller,
                        llm_worker=llm_worker,
                        hippocampus=_cli_hippocampus,
                        memory_hub=_cli_memory_hub,
                        evaluators=evaluators,
                        max_steps=epochs_value,
                        run_id=run_id,
                        percept_source=sim_source,
                        action_sink=sim_sink,
                        pain_bus=getattr(args, "_sim_pain_bus", None),
                    )
                finally:
                    if llm_worker:
                        llm_worker.stop()

                # If simulation mode, validate expectations and report
                if sim_source is not None and sim_sink is not None:
                    import json as _json
                    from maxim.simulation.validation import validate_expectations

                    scenario_file = getattr(args, "_sim_scenario_file", None)
                    hippo = getattr(args, "_sim_hippo", None)

                    results = validate_expectations(
                        expectations=sim_source.expectations,
                        sink=sim_sink,
                        hippocampus=hippo,
                        emitted_tags=sim_source.emitted_tags,
                    )

                    passed = all(r.passed for r in results)
                    scenario_name = sim_source.definition.name

                    print(f"\n{'PASS' if passed else 'FAIL'}: {scenario_name}")
                    for r in results:
                        status = "PASS" if r.passed else "FAIL"
                        desc = r.expectation.description or r.expectation.type
                        print(f"  [{status}] {desc}")
                        if not r.passed and r.detail:
                            print(f"         {r.detail}")
                    print(f"\nActions recorded: {len(sim_sink.actions)}")
                    for a in sim_sink.actions[:20]:
                        tag = "[BLOCKED]" if a.blocked else ("[OK]" if a.result_success else "[FAIL]")
                        print(f"  {tag} {a.tool_name}")

                    report_path = getattr(args, "sim_report", None)
                    if report_path:
                        report = {
                            "scenario": scenario_name,
                            "passed": passed,
                            "action_count": len(sim_sink.actions),
                            "expectations_met": [
                                r.expectation.description or r.expectation.type for r in results if r.passed
                            ],
                            "expectations_failed": [
                                f"{r.expectation.description or r.expectation.type}: {r.detail}"
                                for r in results
                                if not r.passed
                            ],
                        }
                        with open(report_path, "w") as f:
                            _json.dump(report, f, indent=2)
                        print(f"Report written to {report_path}")

                    # Save sim log and disable logging
                    saved_log = disable_sim_logging()
                    if saved_log:
                        print(f"  Simulation log saved: {saved_log}")

                    # Clean up simulation sandbox (but preserve the log)
                    sim_tmpdir = getattr(args, "_sim_tmpdir", None)
                    if sim_tmpdir is not None:
                        import shutil

                        original_cwd = getattr(args, "_sim_original_cwd", "/")
                        os.chdir(original_cwd)
                        try:
                            shutil.rmtree(str(sim_tmpdir))
                            print(f"  Sandbox cleaned up: {sim_tmpdir}")
                        except Exception as e:
                            print(f"  Warning: failed to clean sandbox: {e}")

                    return 0 if passed else 1

                return 0


            from maxim.conscience.selfy import Maxim

            audio_enabled = bool(getattr(args, "audio", True))
            if mode == "sleep":
                audio_enabled = True

            maxim = Maxim(
                robot_name=args.robot_name,
                home_dir=args.home_dir,
                timeout=args.timeout,
                epochs=epochs_value,
                verbosity=args.verbosity,
                mode=mode,
                audio=audio_enabled,
                audio_len=float(getattr(args, "audio_len", 5.0) or 5.0),
                interactive=bool(getattr(args, "interactive", True)),
            )

            if mode == "sleep":
                logger.info("Maxim sleeping (audio-only).")
                # Mark as already sleeping so no movement occurs on startup
                maxim.sleeping = True
                maxim.sleep(home_dir=args.home_dir, run_id=run_id)
            else:
                logger.info("✅ Maxim lives!")
                maxim.live(home_dir=args.home_dir, run_id=run_id)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user (Ctrl+C).")
            break
        except Exception as e:
            log_exception(
                logger,
                e,
                verbosity=getattr(args, "verbosity", 0),
                message="❌ Maxim stopped",
            )
            break
        finally:
            if maxim is not None:
                try:
                    maxim.shutdown()
                except Exception:
                    pass

        requested = getattr(maxim, "requested_mode", None) if maxim is not None else None
        if not requested:
            break
        requested = str(requested).strip().lower()
        if requested == "shutdown":
            logger.info("Shutdown requested.")
            break
        if requested in ("sleep", "live", "agentic", "passive", "active", "singularity"):
            logger.info("Switching mode: %s -> %s", mode, requested)
            delay_s = 0.0
            try:
                delay_s = float(os.getenv("MAXIM_MODE_SWITCH_DELAY_S", "1.5") or 0.0)
            except Exception:
                delay_s = 1.5
            if delay_s > 0:
                logger.info("Waiting %.1fs before reconnect...", delay_s)
                time.sleep(delay_s)
            try:
                _reexec_with_mode(args, mode=requested)
            except Exception as e:
                logger.warning("Failed to restart Maxim for mode switch (%s); continuing in-process.", e)
                mode = requested
                continue
        logger.warning("Ignoring unknown requested_mode=%r", requested)
        break

    return 0


life = main


if __name__ == "__main__":
    raise SystemExit(main())
