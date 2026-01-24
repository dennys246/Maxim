from __future__ import annotations

# CRITICAL: Detect Blackwell GPU and hide CUDA BEFORE any other imports
# This must happen at module load time to prevent TensorFlow from initializing CUDA
import os
import subprocess
import sys

_blackwell_detected = False
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=2
    )
    if result.returncode == 0:
        gpu_names = result.stdout.strip().lower()
        if 'rtx 50' in gpu_names or '5080' in gpu_names or '5090' in gpu_names:
            _blackwell_detected = True
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            # Print to stderr since logging not yet configured
            print("⚠️  Blackwell GPU detected - CUDA disabled before imports", file=sys.stderr)
except Exception:
    pass

# NOW import everything else (TensorFlow will see no GPUs if Blackwell detected)
import argparse
import logging
import time
from collections.abc import Sequence

from maxim.utils.data_management import build_home
from maxim.utils.logging import configure_logging, log_exception


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maxim")
    parser.add_argument(
        "--robot-name",
        default=os.environ.get("MAXIM_ROBOT_NAME", "reachy_mini"),
        help="Reachy Mini daemon robot_name / zenoh namespace (default: $MAXIM_ROBOT_NAME or 'reachy_mini').",
    )
    parser.add_argument(
        "--home-dir",
        default="data",
        help="Reachy Mini home directory to save run artifacts (audio/videos/images/transcript/logs) (default: 'data').",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the Zenoh connection (default: 30).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Epochs to run Maxim for (0 = unlimited).",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=warnings/errors, 1=info, 2=debug.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="exploration",
        choices=["live", "train", "reflection", "sleep", "agentic", "exploration"],
        help="Run mode: exploration (novelty-driven active discovery; DEFAULT), sleep (audio-only, no movement), live (no training), train (update MotorCortex), agentic (full perception-memory-goal architecture), reflection (introspection and memory consolidation).",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="true",
        help="Record + transcribe audio (True/False).",
    )
    parser.add_argument(
        "--audio_len",
        type=float,
        default=5.0,
        help="Seconds per transcription chunk (default: 5.0).",
    )
    parser.add_argument(
        "--language-model",
        type=str,
        default=None,
        help="LLM profile name (overrides data/util/llm.json and $MAXIM_LLM_PROFILE).",
    )
    parser.add_argument(
        "--segmentation-model",
        type=str,
        default=None,
        help="Vision segmentation model (default: YOLO8).",
    )
    parser.add_argument(
        "--interactive",
        type=str,
        default="true",
        help="Enable interactive terminal input for keyword actions (True/False).",
    )
    parser.add_argument(
        "--memory-path",
        type=str,
        default=None,
        help="Path for memory persistence (agentic mode). Default: {home_dir}/memory/memories.json",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset memory on startup (agentic mode).",
    )
    parser.add_argument(
        "--enable-embeddings",
        action="store_true",
        help="Enable embedding-based memory similarity (requires sentence-transformers).",
    )
    parser.add_argument(
        "--autonomy",
        type=str,
        default="planning",
        choices=["planning", "supervised", "autonomous"],
        help="Initial autonomy level: planning (propose only), supervised (act within bounds), autonomous (full agency).",
    )
    parser.add_argument(
        "--autonomy-duration",
        type=float,
        default=None,
        help="Duration in seconds for timed autonomy (only applies to autonomous level).",
    )
    parser.add_argument(
        "--internet-access",
        action="store_true",
        help="Enable internet access for search and fetch tools.",
    )
    parser.add_argument(
        "--agentic-verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Agentic logging verbosity: 0=quiet, 1=normal (goals/tools), 2=verbose (+perception/memory), 3=debug (+loop).",
    )
    parser.add_argument(
        "--agentic-console",
        action="store_true",
        help="Print agentic events to console in real-time.",
    )
    # ─────────────────────────────────────────────────────────────────────────
    # TTS (Text-to-Speech) arguments
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable text-to-speech for spoken responses (requires piper-tts).",
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        default="en_US-lessac-medium",
        help="TTS voice model name (default: en_US-lessac-medium).",
    )
    # ─────────────────────────────────────────────────────────────────────────
    # Exploration mode arguments
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--explore",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Start in exploration mode with optional focus (e.g., --explore 'kitchen objects').",
    )
    parser.add_argument(
        "--exploration-duration",
        type=float,
        default=None,
        help="Duration in seconds for exploration session (default: unlimited).",
    )
    parser.add_argument(
        "--exploration-autonomy",
        type=str,
        default="supervised",
        choices=["supervised", "autonomous"],
        help="Autonomy level for exploration: supervised (default) or autonomous.",
    )
    parser.add_argument(
        "--exploration-allow-internet",
        action="store_true",
        help="Allow internet search and fetch during exploration.",
    )
    parser.add_argument(
        "--exploration-allow-scripts",
        action="store_true",
        help="Allow writing and executing Python analysis scripts during exploration.",
    )
    parser.add_argument(
        "--exploration-allow-training",
        action="store_true",
        help="Allow model training during exploration (requires GPU).",
    )
    parser.add_argument(
        "--resume-session",
        type=str,
        default=None,
        help="Resume a previous exploration session by ID.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List available exploration sessions and exit.",
    )
    return parser


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

    if str(getattr(args, "mode", "exploration")).strip().lower() == "sleep":
        args.audio = True
    args.epochs = _normalize_epoch_value(getattr(args, "epochs", 0))

    # Handle --explore shortcut: sets mode to exploration
    explore_focus = getattr(args, "explore", None)
    if explore_focus is not None:
        args.mode = "exploration"
        # Store sanitized focus (empty string means general exploration)
        args.exploration_focus = str(explore_focus).strip() if explore_focus else ""

    language_model = getattr(args, "language_model", None)
    if language_model is not None:
        from maxim.models.language.router import list_llm_profiles, normalize_llm_profile

        selected = normalize_llm_profile(language_model)
        if selected:
            available = list_llm_profiles()
            if available and selected not in available:
                opts = ", ".join(available)
                raise SystemExit(f"Unknown --language-model {language_model!r}. Available: {opts}")
            os.environ["MAXIM_LLM_PROFILE"] = selected
        args.language_model = selected

    segmentation_model = getattr(args, "segmentation_model", None)
    if segmentation_model is not None:
        from maxim.models.vision.registry import list_segmentation_models, normalize_segmentation_model

        selected = normalize_segmentation_model(segmentation_model) or "YOLO8"
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

    # Check if GPU is intentionally disabled (including Blackwell auto-detection)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible == "":
        if _blackwell_detected:
            logger.warning("⚠️  Blackwell GPU detected - TensorFlow CUDA disabled")
            logger.info("   (Transcription worker will use CPU-only CTranslate2)")
        else:
            logger.warning("⚠️  GPU acceleration disabled (CUDA_VISIBLE_DEVICES=\"\")")
        logger.info("Running in CPU-only mode")
        return

    # Check TensorFlow GPU
    tf_gpus = []
    tf_gpu_info = []
    try:
        import tensorflow as tf

        if not blackwell_detected:
            tf_gpus = tf.config.list_physical_devices('GPU')
            if tf_gpus:
                for gpu in tf_gpus:
                    try:
                        # Get GPU details
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        gpu_name = gpu_details.get('device_name', 'Unknown GPU')
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
    if tf_gpus or torch_gpus:
        logger.info("✅ GPU acceleration enabled")

        if tf_gpus:
            logger.info(f"   TensorFlow detected {len(tf_gpus)} GPU(s):")
            for i, info in enumerate(tf_gpu_info):
                logger.info(f"     [{i}] {info}")

        if torch_gpus:
            logger.info(f"   PyTorch detected {torch_gpus} GPU(s):")
            for i, info in enumerate(torch_gpu_info):
                logger.info(f"     [{i}] {info}")
    else:
        logger.warning("⚠️  No GPU detected - running in CPU-only mode")
        logger.info("   For GPU support, ensure:")
        logger.info("   - NVIDIA drivers are installed (570+)")
        logger.info("   - CUDA-compatible GPU is available")
        logger.info("   - tensorflow[and-cuda] is installed")


def _configure_cpu_fallback_model(logger: logging.Logger) -> None:
    """Configure a smaller LLM model for CPU-only inference when no GPU is available.

    Sets environment variables to use SmolLM 1.7B with CPU inference, which is
    suitable for systems with limited resources (e.g., 24GB unified memory).
    """
    import os

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _normalize_args(args)

    build_home(args.home_dir)
    mode = str(getattr(args, "mode", "exploration")).strip().lower()
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
                    _configure_cpu_fallback_model(logger)

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
                    run_agent_loop,
                )
                from maxim.runtime.agent_loop import run_agentic_loop
                from maxim.utils.structured_logging import configure_agentic_verbosity

                # Configure agentic verbosity
                agentic_verbosity = int(getattr(args, "agentic_verbosity", 1))
                agentic_console = bool(getattr(args, "agentic_console", False))
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

                registry = build_tool_registry(response_output=response_output)
                executor = build_executor(registry)
                decision_engine = build_decision_engine()
                env = ReachyEnv(data_dir=args.home_dir)
                state = build_state(max_steps=epochs_value)
                memory = build_memory()
                evaluators = build_evaluators()

                # Set up autonomy controller
                autonomy_level_str = str(getattr(args, "autonomy", "planning")).lower()
                initial_level = AutonomyLevel(autonomy_level_str)

                # Configure supervision policy with sensible defaults
                supervision_policy = SupervisionPolicy(
                    allowed_tools={
                        "read_file",
                        "focus_interests",
                        "track_target",
                        "maxim_command",
                        "mode_switch",
                        "speak",
                        "respond",
                    },
                    forbidden_tools={"execute_file", "delete_file"},
                    min_confidence_autonomous=0.7,
                    requires_confirmation={"write_file"},
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

                # Set up LLM worker
                llm_worker = None
                if hasattr(agentic_agent, "_llm") and agentic_agent._llm is not None:
                    llm_worker = LLMWorker(agentic_agent._llm)
                    llm_worker.start()

                # Store internet access in state
                internet_access = bool(getattr(args, "internet_access", False))
                state.data["internet_access"] = internet_access
                state.data["autonomy_level"] = initial_level.value

                logger.info(
                    "Starting MaximAgent (memory_path=%s, embeddings=%s, reset=%s, autonomy=%s, internet=%s)",
                    memory_path,
                    bool(getattr(args, "enable_embeddings", False)),
                    bool(getattr(args, "reset", False)),
                    initial_level.value,
                    internet_access,
                )

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
                        evaluators=evaluators,
                        max_steps=epochs_value,
                        run_id=run_id,
                    )
                finally:
                    if llm_worker:
                        llm_worker.stop()
                return 0

            # ─────────────────────────────────────────────────────────────────
            # Exploration mode - uses full Maxim with live camera + agentic brain
            # ─────────────────────────────────────────────────────────────────
            if mode == "exploration":
                # Handle --list-sessions
                if bool(getattr(args, "list_sessions", False)):
                    from maxim.modes.exploration import ExplorationSession

                    sessions_dir = os.path.join(args.home_dir, "exploration_sessions")
                    sessions = ExplorationSession.list_sessions(sessions_dir)
                    if not sessions:
                        print("No exploration sessions found.")
                    else:
                        print(f"Found {len(sessions)} exploration session(s):")
                        for session_id in sessions:
                            print(f"  - {session_id}")
                    return 0

                if not _gpu_available():
                    _configure_cpu_fallback_model(logger)

                from maxim.modes.exploration import (
                    AdversarialFocusValidator,
                    ExplorationConstraints,
                    ExplorationPolicy,
                    ExplorationSession,
                )

                # Build exploration policy from CLI args
                exploration_policy = ExplorationPolicy(
                    require_gpu_for_agentic=True,
                    allow_internet=bool(getattr(args, "exploration_allow_internet", False)),
                    allow_scripts=bool(getattr(args, "exploration_allow_scripts", False)),
                    allow_training=bool(getattr(args, "exploration_allow_training", False)),
                )

                # Validate focus text if provided
                exploration_focus = str(getattr(args, "exploration_focus", "") or "").strip()
                if exploration_focus:
                    validator = AdversarialFocusValidator()
                    is_valid, reason = validator.validate(exploration_focus)
                    if not is_valid:
                        logger.error("Invalid exploration focus: %s", reason)
                        return 1

                # Handle session resume
                resume_session_id = getattr(args, "resume_session", None)
                sessions_dir = os.path.join(args.home_dir, "exploration_sessions")
                session: ExplorationSession | None = None

                if resume_session_id:
                    session = ExplorationSession.load(sessions_dir, resume_session_id)
                    if session is None:
                        logger.error("Session %s not found.", resume_session_id)
                        return 1
                    logger.info("Resuming exploration session: %s", resume_session_id)
                    # Override focus from session if not provided
                    if not exploration_focus and session.focus:
                        exploration_focus = session.focus
                else:
                    # Create new session
                    session = ExplorationSession(
                        focus=exploration_focus,
                        policy=exploration_policy,
                        constraints=ExplorationConstraints(),
                    )
                    session.save(sessions_dir)
                    logger.info("Created exploration session: %s", session.session_id)

                logger.info(
                    "Starting exploration mode (focus=%r, session=%s, internet=%s, scripts=%s, training=%s)",
                    exploration_focus or "(general)",
                    session.session_id,
                    exploration_policy.allow_internet,
                    exploration_policy.allow_scripts,
                    exploration_policy.allow_training,
                )

                # Use the full Maxim class with live camera - exploration runs as "live" mode
                # with exploration context stored for the agentic runtime
                from maxim.conscience.selfy import Maxim

                audio_enabled = bool(getattr(args, "audio", True))

                maxim = Maxim(
                    robot_name=args.robot_name,
                    home_dir=args.home_dir,
                    timeout=args.timeout,
                    epochs=epochs_value,
                    verbosity=args.verbosity,
                    mode="exploration",  # Use exploration mode for novelty-driven discovery
                    audio=audio_enabled,
                    audio_len=float(getattr(args, "audio_len", 5.0) or 5.0),
                    interactive=bool(getattr(args, "interactive", True)),
                )

                # Store exploration context in Maxim's state for the agentic runtime
                # This will be picked up by _start_agentic_runtime() in selfy.py
                maxim._exploration_mode = True
                maxim._exploration_focus = exploration_focus
                maxim._exploration_session_id = session.session_id
                maxim._exploration_policy = exploration_policy.to_dict()

                try:
                    logger.info("✅ Maxim exploration mode active!")
                    maxim.live(home_dir=args.home_dir, run_id=run_id)
                finally:
                    # Save session state
                    if session:
                        session.save(sessions_dir)
                    try:
                        maxim.shutdown()
                    except Exception:
                        pass

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
        if requested in ("sleep", "reflection", "train", "live", "agentic", "exploration"):
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
