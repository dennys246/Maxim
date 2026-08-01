import os
import threading

# NOTE: Import-time side effects (mp.set_start_method, GPU detection,
# PYOPENGL_PLATFORM) moved to _setup_hardware_env() — called from
# Maxim.__init__() so that `import maxim.embodied_runtime.selfy` has no
# subprocess or env-mutation side effects.

from maxim.utils.thread_manager import ThreadRegistry
from maxim.utils.response_config import (
    load_key_responses,
    load_phrase_responses,
)
from maxim.modes.state_manager import StateManager
from maxim.runtime.capabilities import RuntimeCapabilities, detect_compute_resources

import time
import atexit
import logging
from typing import Any

import numpy as np

from maxim.motion.movement import load_actions, load_movement_thresholds, load_poses
from maxim.utils.data_management import CLIInputLogger, VisionEventLogger
from maxim.utils.logging import configure_logging, log_swallowed_exception, warn
from maxim.utils.plotting import preflight_matplotlib_fonts, preload_matplotlib_fonts

from maxim.inference.observation import (
    DEFAULT_CLASS_WEIGHTS,
    NoveltyTracker,
)
from maxim.models.vision.registry import build_segmentation_model

# Hardware abstraction layer for multi-robot support
from maxim.hardware import RobotRegistry
from maxim.hardware.reachy import ReachyMiniController
from maxim.hardware.simulation import SimulatedController

# Module-level state — populated lazily by _setup_hardware_env()
_blackwell_detected = False
_original_cuda_devices: str | None = None
_hardware_env_ready = False


def _setup_hardware_env() -> None:
    """Apply hardware environment guards (idempotent).

    - Sets multiprocessing start method to 'spawn' (avoids GStreamer segfaults)
    - Detects Blackwell GPU and applies GStreamer CUDA guards
    - Sets PYOPENGL_PLATFORM=egl for headless OpenGL

    Called from Maxim.__init__() — NOT at import time.
    """
    global _blackwell_detected, _original_cuda_devices, _hardware_env_ready
    if _hardware_env_ready:
        return

    # 1. Multiprocessing start method (must be before any pool/process creation)
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    # 2. Blackwell GPU detection + GStreamer guards
    from maxim.utils.gpu_compat import detect_blackwell

    gpu_state = detect_blackwell()
    _blackwell_detected = gpu_state.blackwell_detected
    _original_cuda_devices = gpu_state.original_cuda_devices

    # 3. OpenGL platform for headless rendering
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    _hardware_env_ready = True


# Mixin classes (compartmentalized from this file)
from maxim.embodied_runtime.connection import ConnectionMixin
from maxim.embodied_runtime.vision_stream import VisionStreamMixin
from maxim.embodied_runtime.agentic_runtime import AgenticRuntimeMixin
from maxim.embodied_runtime.movement import MovementMixin
from maxim.embodied_runtime.input_handlers import InputHandlerMixin
from maxim.embodied_runtime.media_loop import MediaLoopMixin

# Initialize global robot registry with controller types
_robot_registry = RobotRegistry()
_robot_registry.register_controller_type("reachy_mini", ReachyMiniController)
_robot_registry.register_controller_type("simulated", SimulatedController)


class Maxim(InputHandlerMixin, ConnectionMixin, MovementMixin, VisionStreamMixin, AgenticRuntimeMixin, MediaLoopMixin):
    """
    A class for orchestracting models and agents with Reachy-Mini's.
    """

    def __init__(
        self,
        robot_name: str = "reachy_mini",
        timeout: float = 30.0,
        media_backend: str = "default",  # avoid WebRTC/GStreamer if signalling is down
        home_dir: str = "",
        epochs: int | None = None,
        *,
        verbosity: int = 0,
        verbose: bool = False,
        mode: str = "exploration",
        train: bool | None = None,
        audio: bool = True,
        audio_len: float = 5.0,
        interactive: bool = True,
        novelty_tracker: "NoveltyTracker | None" = None,
        simulation: bool = False,
        robot_id: str | None = None,
    ):
        # Apply hardware env guards (mp start method, GPU detection, OpenGL)
        _setup_hardware_env()

        self.verbosity = int(verbosity or 0)
        if verbose and self.verbosity <= 0:
            self.verbosity = 1
        self.verbose = self.verbosity > 0

        if self.verbose:
            configure_logging(self.verbosity)

        self.log = logging.getLogger("maxim.Maxim")

        self.alive = True
        self._closed = False
        self._woke_up = False
        self.sleeping = False  # Track if Reachy is already in sleep pose

        # Central thread registry for coordinated shutdown
        # All threads created by Maxim should be registered here
        self._thread_registry = ThreadRegistry()

        # Hardware abstraction layer
        self._simulation = bool(simulation)
        self._robot_id = robot_id
        self._robot: ReachyMiniController | SimulatedController | None = None

        self.name = robot_name or os.getenv("MAXIM_ROBOT_NAME", "reachy_mini")
        if _blackwell_detected:
            self.log.info("Blackwell GPU detected - GStreamer hardware acceleration disabled")
        self.log.info("Connecting to Reachy Mini '%s'...", self.name)
        self._connect_kwargs = {
            "robot_name": self.name,
            "connection_mode": "network",
            "spawn_daemon": False,
            "use_sim": False,
            "timeout": float(timeout),
            "media_backend": media_backend,
        }
        self._media_lock: threading.Lock | None = None
        self._reconnect_lock = threading.Lock()
        self._last_reconnect_ts = 0.0
        self._reconnect_cooldown_s = 20.0
        self._reconnect_window_s = 5.0
        self._reconnect_thresholds = {"motor": 3, "video": 5, "audio": 5}
        self._connection_failures = {
            "motor": {"count": 0, "last_ts": 0.0},
            "video": {"count": 0, "last_ts": 0.0},
            "audio": {"count": 0, "last_ts": 0.0},
        }
        self.start = time.time()
        self.duration = 1.0
        if not home_dir:
            from maxim.utils.paths import data_home

            home_dir = str(data_home())
        self.home_dir = home_dir

        # Load Matplotlib before Reachy/GStreamer so ft2font binds to stable libs.
        preload_matplotlib_fonts(
            cache_dir=os.path.join(self.home_dir, "matplotlib"),
            logger=self.log,
        )

        self.current_epoch = 0
        self._set_epochs(epochs)
        mode = str(mode or "exploration").strip().lower()
        if train is not None:
            mode = "train" if bool(train) else "live"
        self.mode = mode
        self.train = self.mode == "train"

        self.observation_period = 1
        self.audio = bool(audio)
        try:
            self.audio_len = float(audio_len)
        except (TypeError, ValueError) as e:
            log_swallowed_exception(e, operation="parse_audio_len", context={"audio_len": audio_len})
            self.audio_len = 5.0
        if self.audio_len <= 0:
            self.audio_len = 5.0

        self.video_fps = 20.0

        self.interactive = bool(interactive)

        # Salience boost classes: these get 2x priority in perception scoring.
        # All 80 COCO classes are always detected; this controls attention priority.
        self.interests = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        # Novelty tracking for attention prioritization
        # Novel objects get higher priority, familiar objects decay over time
        # Can be shared with DefaultNetwork for unified tracking
        if novelty_tracker is not None:
            self.novelty_tracker = novelty_tracker
        else:
            self.novelty_tracker = NoveltyTracker(
                focus_decay_seconds=10.0,  # Time for novelty to decay while focused
                recovery_seconds=20.0,  # Time for novelty to recover when not focused
                max_novelty=2.0,  # Novelty boost for new objects
                min_novelty=0.5,  # Minimum novelty for familiar objects
            )
        # Class weights for attention (default gives slight preference to people)
        self.class_weights = dict(DEFAULT_CLASS_WEIGHTS)

        self.actions = load_actions()
        self.poses = load_poses()
        self.movement_thresholds = load_movement_thresholds()
        self._head_max_step = {}
        try:
            head_cfg = self.movement_thresholds.get("head") if isinstance(self.movement_thresholds, dict) else None
            if isinstance(head_cfg, dict) and isinstance(head_cfg.get("max_step"), dict):
                self._head_max_step = dict(head_cfg.get("max_step") or {})
        except (KeyError, TypeError, AttributeError) as e:
            log_swallowed_exception(e, operation="load_head_max_step")
            self._head_max_step = {}

        # Runtime capabilities — updated after robot connection attempt.
        self._capabilities = RuntimeCapabilities()
        _has_gpu, _gpu_type, _vram_gb, _ram_gb = detect_compute_resources()
        self._capabilities.has_gpu = _has_gpu
        self._capabilities.gpu_type = _gpu_type
        self._capabilities.vram_gb = _vram_gb
        self._capabilities.ram_gb = _ram_gb

        # Connect to robot using hardware abstraction layer.
        # Supports both real Reachy Mini hardware and simulation mode.
        robot_type = "simulated" if self._simulation else "reachy_mini"
        effective_robot_id = self._robot_id or self.name

        # Allow env var to override timeout for faster headless startup
        effective_timeout = float(os.environ.get("MAXIM_ROBOT_TIMEOUT", str(timeout)))

        self.log.info("Connecting to Reachy Mini '%s'...", effective_robot_id)

        # Operator-declared connection config (host, connection_mode,
        # tunnel, ...) comes from ~/.maxim/robots.yaml — the same file the
        # body wiring already reads. Pre-fix (2026-07-31) this path built
        # the config inline and silently ignored the file, so the connect
        # error's own advice ("set host: <ip> in robots.yaml") did nothing:
        # the controller fell back to mDNS and failed on hosts where .local
        # resolution is blocked. Declared keys win over the inline defaults;
        # wiring-layer keys (body, audio_localization) are filtered to the
        # constructor's signature by connect_robot.
        if not self._simulation:
            _connect_config = {
                "robot_name": self.name,
                "media_backend": media_backend,
            }
            try:
                from maxim.hardware.config import load_robots_config, resolve_connection_config

                _connect_config = resolve_connection_config(
                    load_robots_config(),
                    effective_robot_id,
                    defaults=_connect_config,
                )
            except Exception as e:  # config load is best-effort; never block startup
                self.log.debug("robots.yaml connection config not loaded: %s", e)
        else:
            _connect_config = {
                "video_resolution": (640, 480),
                "simulate_delays": False,
            }

        # Use the global registry to connect
        self._robot = _robot_registry.connect_robot(
            robot_id=effective_robot_id,
            robot_type=robot_type,
            config=_connect_config,
            timeout=effective_timeout,
            set_primary=True,
        )

        if self._robot is None:
            self.log.warning("No robot connected — running in headless mode")
        else:
            self._capabilities.has_robot = True
            self._capabilities.has_motor = True
            self._capabilities.has_vision = True
            self._capabilities.has_audio = True
            self._capabilities.robot_type = robot_type

        # On Blackwell GPUs, don't start recording - it will crash in GStreamer
        self._recording_started = False
        if self._robot is not None:
            if not _blackwell_detected and not self._simulation:
                self.log.info("Connected. Starting recording...")
                try:
                    self._robot.start_recording()
                    self._recording_started = True
                except Exception as e:
                    self.log.warning("Failed to start recording: %s", e)
            elif self._simulation:
                # Simulation always starts recording
                self._robot.start_recording()
                self._recording_started = True
                self.log.info("Connected to simulated robot.")
            else:
                self.log.info("Connected. (Recording disabled for Blackwell GPU compatibility)")

        self.x = 0.01
        self.y = 0.01
        self.z = 0.01

        self.roll = 0.01
        self.pitch = 0.01
        self.yaw = 0.01
        self.body_yaw = 0.0  # Body rotation (separate from head yaw)
        self._last_turn_around_time = 0.0  # Cooldown tracking for turn_around

        centered = None
        try:
            centered = getattr(self, "poses", {}).get("centered")
        except (AttributeError, TypeError) as e:
            log_swallowed_exception(e, operation="get_centered_pose")
            centered = None
        if isinstance(centered, (list, tuple)) and len(centered) >= 6:
            try:
                self.x = float(centered[0])
                self.y = float(centered[1])
                self.z = float(centered[2])
                self.roll = float(centered[3])
                self.pitch = float(centered[4])
                self.yaw = float(centered[5])
            except (TypeError, ValueError, IndexError) as e:
                log_swallowed_exception(e, operation="parse_centered_pose", context={"centered": centered})

        self._default_head_pose = {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "roll": float(self.roll),
            "pitch": float(self.pitch),
            "yaw": float(self.yaw),
        }

        self._training_paused = threading.Event()
        self._observation_lock = threading.Lock()

        self.key_responses = load_key_responses(self.log)
        self.phrase_responses = load_phrase_responses(self.log)
        self._voice_agentic_enabled = False
        self._phrase_last_trigger_ts: dict[str, float] = {}
        self._outcome_code = 0
        self._last_action_event_id: str | None = None
        self._last_transcript_event: dict | None = None
        self.requested_mode: str | None = None
        # New architecture state tracking via StateManager
        self._state_manager = StateManager()
        self._agentic_stop_event: threading.Event | None = None
        self._agentic_thread: threading.Thread | None = None
        self._agentic_agent = None
        self._agentic_state = None
        self._protocol_registry = None  # Set by _start_agentic_runtime
        self._workspace_limit_override: dict[str, float] | None = None  # Set by ProtocolRegistry
        self._cli_logger: CLIInputLogger | None = None
        self._vision_event_logger: VisionEventLogger | None = None
        self._vision_event_thread: threading.Thread | None = None
        self._vision_event_stop_event: threading.Event | None = None
        self._vision_event_last_frame_ts: float | None = None
        self.vision_events_path: str | None = None

        self.movement_model = None
        self.segmenter = None
        self._segmenter_model: str | None = None
        self.motor_history: list[dict] = []

        atexit.register(self.shutdown)

    # ─────────────────────────────────────────────────────────────────────────
    # Protocol dispatch (called by phrase_responses)
    # ─────────────────────────────────────────────────────────────────────────

    def _protocol_activate(self, name: str) -> None:
        """Called by phrase_responses to activate a protocol."""
        if self._protocol_registry is not None:
            result = self._protocol_registry.activate(name)
            self.log.info("Protocol activation: %s", result)

    def _protocol_deactivate(self, name: str) -> None:
        """Called by phrase_responses to deactivate a protocol."""
        if self._protocol_registry is not None:
            result = self._protocol_registry.deactivate(name)
            self.log.info("Protocol deactivation: %s", result)

    # ─────────────────────────────────────────────────────────────────────────
    # Hardware Properties (backward compatibility + abstraction)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def mini(self) -> Any:
        """Get the underlying ReachyMini SDK instance.

        DEPRECATED: Use self._robot (RobotController) for new code.
        This property is provided for backward compatibility during migration.

        Returns:
            ReachyMini SDK instance if using real hardware, None otherwise.
        """
        if self._robot is None:
            return None
        # ReachyMiniController exposes the SDK instance via .mini property
        return getattr(self._robot, "mini", None)

    @property
    def robot(self) -> "ReachyMiniController | SimulatedController | None":
        """Get the robot controller (new abstraction layer).

        Use this for new code that should work with any robot type.
        """
        return self._robot

    def get_frame(self) -> np.ndarray | None:
        """Get the current video frame from the robot.

        Works with both real hardware and simulation.

        Returns:
            BGR image as numpy array, or None if not available.
        """
        if self._robot is None:
            return None

        video_stream = self._robot.get_video_stream()
        if video_stream is not None:
            return video_stream.get_frame_nonblocking()

        # Fallback to direct SDK access for backward compatibility
        mini = self.mini
        if mini is not None:
            try:
                return mini.media.get_frame()
            except Exception:
                return None

        return None

    def get_audio_sample(self) -> np.ndarray | None:
        """Get an audio sample from the robot.

        Works with both real hardware and simulation.

        Returns:
            Audio samples as numpy array, or None if not available.
        """
        if self._robot is None:
            return None

        audio_stream = self._robot.get_audio_stream()
        if audio_stream is not None:
            return audio_stream.get_sample(timeout=0.1)

        # Fallback to direct SDK access for backward compatibility
        mini = self.mini
        if mini is not None:
            try:
                return mini.media.get_audio_sample()
            except Exception:
                return None

        return None

    def push_audio_sample(self, samples: np.ndarray) -> bool:
        """Push audio samples to the robot speaker.

        Works with both real hardware and simulation.

        Args:
            samples: Audio samples as numpy array.

        Returns:
            True if samples were pushed successfully.
        """
        if self._robot is None:
            return False

        audio_stream = self._robot.get_audio_stream()
        if audio_stream is not None:
            return audio_stream.push_sample(samples)

        # Fallback to direct SDK access for backward compatibility
        mini = self.mini
        if mini is not None:
            try:
                mini.media.push_audio_sample(samples)
                return True
            except Exception:
                return False

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # State Properties (delegate to StateManager)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def operational_mode(self) -> str:
        """Current operational mode (passive/active/singularity)."""
        return self._state_manager.operational_mode

    @property
    def processing_state(self) -> str:
        """Current processing state (awake/sleep)."""
        return self._state_manager.processing_state

    @property
    def current_strategy(self) -> str:
        """Current strategy (observe/explore/research/assist/reflect/learn)."""
        return self._state_manager.strategy

    # ─────────────────────────────────────────────────────────────────────────
    # Core Utility
    # ─────────────────────────────────────────────────────────────────────────

    def _repo_root(self) -> str:
        try:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        except Exception:
            return os.getcwd()

    def _set_epochs(self, epochs: int | None) -> None:
        try:
            value = int(epochs) if epochs is not None else 0
        except Exception:
            value = 0
        self.epochs = value if value > 0 else None

    # ─────────────────────────────────────────────────────────────────────────
    # Mode Request Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _request_mode(self, mode: str) -> None:
        requested = str(mode or "").strip().lower()
        if not requested:
            return

        current = str(getattr(self, "mode", "") or "").strip().lower()
        if requested != "shutdown" and requested == current:
            return

        try:
            self.log.info("Mode switch requested (%s -> %s).", current or None, requested)
        except Exception:
            pass

        self.requested_mode = requested

        # Only stop the live loop for shutdown mode
        # Other mode changes should just update the mode without stopping
        if requested == "shutdown":
            ev = getattr(self, "_live_stop_event", None)
            if ev is not None:
                try:
                    if int(getattr(self, "verbosity", 0) or 0) >= 2:
                        self.log.debug("Stopping live loop for shutdown")
                    ev.set()
                except Exception:
                    pass

    def request_shutdown(self) -> None:
        self._request_mode("shutdown")

    def request_sleep(self) -> None:
        """Enter sleep processing state (minimal processing, keyword monitoring)."""
        self._state_manager.request_sleep()

    def request_wake(self) -> None:
        """Wake from sleep processing state."""
        self._state_manager.request_wake()

    # Operational mode switch methods (called by phrase responses)
    def request_mode_passive(self) -> None:
        self._state_manager.request_mode_passive()

    def request_mode_active(self) -> None:
        self._state_manager.request_mode_active()

    def request_mode_singularity(self) -> None:
        self._state_manager.request_mode_singularity()

    def update_interests(
        self,
        add: list[int] | None = None,
        remove: list[int] | None = None,
    ) -> None:
        """Update interest class IDs for salience boosting.

        Interest classes receive higher salience scores in the perception pipeline,
        making them more likely to be noticed and acted upon. All COCO classes are
        always detected; this controls prioritization, not visibility.
        """
        updated = set(int(v) for v in (getattr(self, "interests", []) or []) if v is not None)
        if add:
            updated.update(int(v) for v in add if v is not None)
        if remove:
            updated.difference_update(int(v) for v in remove if v is not None)

        self.interests = sorted(updated)

        agent = getattr(self, "_agentic_agent", None)
        if agent is not None and hasattr(agent, "update_interests"):
            try:
                agent.update_interests(add=add, remove=remove)
            except Exception as e:
                warn("Failed to update agent interests: %s", e, logger=self.log)

        state = getattr(self, "_agentic_state", None)
        if state is not None:
            try:
                data = getattr(state, "data", None)
                if isinstance(data, dict):
                    runtime = data.get("maxim_runtime")
                    if isinstance(runtime, dict):
                        runtime["interests"] = list(self.interests)
            except Exception:
                pass

        try:
            self.log.info("Updated interests: %s", self.interests)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Core Lifecycle Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_segmenter(self, *, force: bool = False, model_name: str | None = None) -> None:
        if not force and getattr(self, "segmenter", None) is not None:
            return

        seg_model = str(model_name or os.getenv("MAXIM_SEGMENTATION_MODEL", "rtm") or "rtm").strip() or "rtm"
        self.log.info("Loading vision models (%s seg+pose)...", seg_model)
        # Preflight matplotlib font cache in a subprocess to avoid hard crashes on Linux/WSL.
        preflight_ok = preflight_matplotlib_fonts(
            cache_dir=os.path.join(self.home_dir, "matplotlib"),
            logger=self.log,
        )
        if not preflight_ok:
            raise RuntimeError("Matplotlib font preflight failed; see README troubleshooting.")
        try:
            self.segmenter = build_segmentation_model(seg_model, pose_model=True)  # Visual segmentation + pose model
            self._segmenter_model = seg_model
        except Exception as e:
            warn("Failed to load segmentation model '%s': %s (falling back to rtm)", seg_model, e, logger=self.log)
            self.segmenter = build_segmentation_model("rtm", pose_model=True)
            self._segmenter_model = "rtm"

    def awaken(self, vision: bool = True, motor: bool = True, audio: bool = True, wake_up: bool = True):
        if wake_up:
            # Wake up Reachy before model init to avoid loading while asleep.
            self.log.info("Waking up Reachy...")

            # On Blackwell GPUs, skip wake_up() entirely due to GStreamer/GLib crash
            # The crash happens in native GLib.MainLoop.run() code in the WebRTC thread
            # Bug report: https://github.com/pollen-robotics/reachy_mini/issues
            if _blackwell_detected and not self._simulation:
                self.log.warning(
                    "Blackwell GPU detected - skipping wake_up() due to GStreamer incompatibility. "
                    "Robot should already be in usable state. See: https://github.com/pollen-robotics/reachy_mini/issues"
                )
                # Don't call wake_up() - the robot is already connected and motors are accessible
            elif self._robot is not None:
                self._robot.wake_up()
            self.sleeping = False  # Mark as awake so next sleep() will move to sleep pose
            self._woke_up = True

            # On Blackwell GPUs, start recording AFTER wake_up() to avoid
            # GStreamer/WebRTC threading issues that cause segfaults
            if _blackwell_detected and not self._simulation and not getattr(self, "_recording_started", False):
                time.sleep(0.3)  # Brief pause to let motor commands complete
                try:
                    if self._robot is not None:
                        self._robot.start_recording()
                    self._recording_started = True
                    self.log.info("Recording started (delayed for Blackwell GPU)")
                except Exception as e:
                    self.log.warning("Failed to start recording: %s", e)

        # Load models
        if vision:
            self._ensure_segmenter()

        if motor and self.movement_model is None:
            try:
                from maxim.models.movement.motor_cortex import LayerScale, MotorCortex
                from maxim.utils import config as motor_config
                import tensorflow as tf

                self.log.info("Initializing motor cortex...")

                cfg = motor_config.build(motor_config.DEFAULT_SAVE_ROOT)

                # Try GPU first, fallback to CPU if JIT compilation fails (e.g., RTX 5080/Blackwell)
                try:
                    self.movement_model = MotorCortex(cfg)
                    self.log.info("Motor cortex initialized on GPU")
                except (RuntimeError, tf.errors.InternalError) as gpu_err:
                    self.log.warning(f"GPU initialization failed ({type(gpu_err).__name__}), falling back to CPU mode")
                    self.log.warning("This is expected on RTX 5080/Blackwell GPUs with current TensorFlow")

                    # Force CPU mode
                    with tf.device("/CPU:0"):
                        self.movement_model = MotorCortex(cfg)
                    self.log.info("Motor cortex initialized on CPU")

                checkpoint_path = getattr(cfg, "checkpoint_path", None)
                legacy_checkpoint_path = None
                if checkpoint_path and not os.path.exists(checkpoint_path):
                    try:
                        legacy_checkpoint_path = (
                            motor_config.LEGACY_SAVE_ROOT / motor_config.DEFAULT_CHECKPOINT_FILENAME
                        ).as_posix()
                    except Exception:
                        legacy_checkpoint_path = None

                load_path = None
                for candidate in (checkpoint_path, legacy_checkpoint_path):
                    if candidate and os.path.exists(candidate):
                        load_path = candidate
                        break

                if load_path:
                    try:
                        import keras

                        if load_path != checkpoint_path:
                            self.log.info("Loading legacy motor checkpoint: %s", load_path)
                        else:
                            self.log.info("Loading motor checkpoint: %s", load_path)
                        loaded = keras.models.load_model(
                            load_path,
                            custom_objects={
                                "LayerScale": LayerScale,
                                "MotorCortex": MotorCortex,
                                "motor_cortex": MotorCortex,
                            },
                        )
                        self.movement_model.model = loaded
                    except Exception as e:
                        self.log.warning("Failed to load motor checkpoint '%s': %s", load_path, e)
                else:
                    self.log.info("No motor checkpoint found; starting fresh.")
            except Exception as e:
                self.movement_model = None
                self.log.warning("Motor cortex unavailable: %s", e)

        return

    # ─────────────────────────────────────────────────────────────────────────
    # Thread Registry for Coordinated Shutdown
    # ─────────────────────────────────────────────────────────────────────────

    def register_thread(self, name: str, thread: threading.Thread) -> None:
        """Register a thread for coordinated shutdown tracking."""
        self._thread_registry.register(name, thread)

    def unregister_thread(self, name: str) -> None:
        """Remove a thread from the registry."""
        self._thread_registry.unregister(name)

    def stop_all_threads(self, timeout: float = 10.0) -> list[str]:
        """Stop all registered threads with force-kill fallback."""
        return self._thread_registry.stop_all(timeout=timeout)

    def shutdown(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True

        # Stop all registered threads first
        failed = self.stop_all_threads(timeout=5.0)
        if failed:
            self.log.warning("Some threads did not stop: %s", failed)

        self._stop_agentic_runtime(timeout=10.0)

        try:
            training_logger = getattr(self, "_training_logger", None)
            if training_logger is not None:
                training_logger.stop(timeout=2.0)
        except Exception:
            pass
        self._training_logger = None
        try:
            cli_logger = getattr(self, "_cli_logger", None)
            if cli_logger is not None:
                cli_logger.stop(timeout=2.0)
        except Exception:
            pass
        self._cli_logger = None

        # Persist the motor cortex state (best-effort; never blocks shutdown).
        try:
            movement_model = getattr(self, "movement_model", None)
            if movement_model is not None:
                cfg = getattr(movement_model, "config", None)
                checkpoint_path = getattr(cfg, "checkpoint_path", None) if cfg is not None else None
                save_dir = getattr(cfg, "save_dir", None) if cfg is not None else None

                if not checkpoint_path:
                    try:
                        from maxim.utils import config as motor_config

                        checkpoint_path = (
                            motor_config.DEFAULT_SAVE_ROOT / motor_config.DEFAULT_CHECKPOINT_FILENAME
                        ).as_posix()
                        save_dir = save_dir or motor_config.DEFAULT_SAVE_ROOT.as_posix()
                    except Exception:
                        checkpoint_path = os.path.join(self.home_dir, "models", "motor_cortex.keras")

                os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                to_save = getattr(movement_model, "model", movement_model)
                if hasattr(to_save, "save"):
                    try:
                        to_save.save(checkpoint_path)
                        self.log.info("Saved motor model: %s", checkpoint_path)
                    except Exception as e:
                        self.log.warning("Failed to save motor model to '%s': %s", checkpoint_path, e)

                history = getattr(self, "motor_history", None)
                if history is not None:
                    if save_dir:
                        history_path = os.path.join(str(save_dir).rstrip("/"), "motor_cortex_history.json")
                    else:
                        history_path = os.path.join(
                            os.path.dirname(checkpoint_path) or ".", "motor_cortex_history.json"
                        )

                    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
                    payload = {
                        "time": time.time(),
                        "checkpoint_path": checkpoint_path,
                        "train_step": int(getattr(movement_model, "_train_step", 0) or 0),
                        "records": history,
                    }
                    from maxim.utils.atomic_io import atomic_write_json
                    from maxim.utils.format_version import with_format_version

                    atomic_write_json(str(history_path), with_format_version(payload))
                    try:
                        num_records = len(history)
                    except Exception:
                        num_records = 0
                    self.log.info("Saved motor history: %s (%d records)", history_path, num_records)

                    try:
                        from maxim.utils.plotting import update_motor_cortex_loss_plot
                        from maxim.utils.plotting import update_motor_cortex_pixel_error_plot

                        update_motor_cortex_loss_plot(history, save_dir=save_dir)
                        update_motor_cortex_pixel_error_plot(history, save_dir=save_dir)
                    except Exception as e:
                        self.log.warning("Failed to write motor plots: %s", e)
        except Exception as e:
            self.log.warning("Failed to save motor artifacts: %s", e)

        if getattr(self, "_woke_up", False):
            requested = str(getattr(self, "requested_mode", "") or "").strip().lower()
            if requested not in ("reflection", "exploration", "live", "train", "agentic"):
                try:
                    if self._robot is not None:
                        self._robot.goto_sleep()
                except Exception as e:
                    warn("Failed to send Reachy to sleep: %s", e, logger=getattr(self, "log", None))

        # Stop recording data
        try:
            if self._robot is not None:
                self._robot.stop_recording()
        except Exception as e:
            warn("Failed to stop recording: %s", e, logger=getattr(self, "log", None))

        # Release the camera + any OpenCV resources
        self._release_media()

        # Disconnect from robot via registry (handles cleanup)
        try:
            if self._robot is not None:
                robot_id = getattr(self._robot, "robot_id", None)
                if robot_id:
                    _robot_registry.disconnect_robot(robot_id)
                self._robot = None
        except Exception as e:
            warn("Failed to disconnect robot: %s", e, logger=getattr(self, "log", None))

        return


if __name__ == "__main__":
    conscience = Maxim()
