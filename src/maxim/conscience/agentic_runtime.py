"""Agentic runtime lifecycle mixin for Maxim.

Handles starting, stopping, and managing the agentic runtime thread,
including LLM worker setup, tool registry, autonomy controller,
DefaultNetwork, CaptureManager, and all related subsystems.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Optional

from maxim.utils.gpu_compat import is_gpu_available
from maxim.utils.logging import warn


class AgenticRuntimeMixin:
    """Mixin providing agentic runtime lifecycle management for the Maxim class."""

    def _start_agentic_runtime(self, *, use_capture_manager: bool = True) -> None:
        """Start the agentic runtime.

        Args:
            use_capture_manager: If True, use CaptureManager for direct frame access (Phase 3).
                                If False, fall back to JSONL-based vision event stream.
        """
        existing = getattr(self, "_agentic_thread", None)
        if existing is not None and getattr(existing, "is_alive", lambda: False)():
            return

        # Allow CPU-only operation (e.g., when CUDA is hidden for Blackwell GPUs)
        # Note: llama_cpp backend has native Metal support on macOS, so skip this fallback
        # when using llama_cpp - it will use Metal GPU acceleration automatically
        if not is_gpu_available():
            # Check if we're using llama_cpp backend (has native Metal support)
            llm_config_path = os.path.join(str(getattr(self, "home_dir", "data") or "data"), "util", "llm.json")
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
                self.log.info("Using llama.cpp backend with native Metal GPU support")
            else:
                cuda_hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
                if cuda_hidden:
                    self.log.info("GPU hidden for compatibility - agentic runtime will use CPU (slower)")
                else:
                    self.log.warning("No GPU available - agentic runtime will use CPU (slower)")

                # Configure CPU-friendly model defaults (only for non-llama_cpp backends)
                os.environ.setdefault("MAXIM_LLM_PROFILE", "smollm-1.7b-instruct")
                os.environ.setdefault("MAXIM_LLM_N_GPU_LAYERS", "0")
                self.log.info("Using CPU-friendly LLM: smollm-1.7b-instruct")

        try:
            from maxim.agents import MaximAgent
            from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SupervisionPolicy
            from maxim.agents.llm_worker import LLMWorker
            from maxim.environment import ReachyEnv
            from maxim.runtime import (
                CaptureManager,
                build_decision_engine,
                build_evaluators,
                build_executor,
                build_memory,
                build_state,
                build_tool_registry,
            )
            from maxim.runtime.bootstrap import build_default_network
            from maxim.runtime.agent_loop import run_agentic_loop
        except Exception as e:
            warn("Agentic runtime unavailable: %s", e, logger=self.log)
            return

        stop_event = threading.Event()
        self._agentic_stop_event = stop_event

        # Phase 3: Initialize CaptureManager for direct frame access
        capture_manager = None
        if use_capture_manager:
            try:
                capture_manager = CaptureManager(
                    maxim=self,
                    target_fps=float(getattr(self, "video_fps", 10.0) or 10.0),
                    enable_segmentation=True,
                )
                self._capture_manager = capture_manager
            except Exception as e:
                warn("Failed to create CaptureManager: %s (falling back to JSONL)", e, logger=self.log)
                capture_manager = None

        agent = MaximAgent(
            interests=list(getattr(self, "interests", []) or []),
            data_folder=str(getattr(self, "home_dir", "data") or "data"),
            capture_manager=capture_manager,
        )
        env = ReachyEnv(repo_root=os.getcwd(), data_dir=str(getattr(self, "home_dir", "data") or "data"))
        state = build_state(max_steps=1_000_000)
        try:
            state.data["maxim_runtime"] = {
                "mode": getattr(self, "mode", None),
                "interests": list(getattr(self, "interests", []) or []),
            }
        except Exception:
            pass
        self._agentic_agent = agent
        self._agentic_state = state
        self._state_manager.set_agent(agent)

        # Extract AgentBus early (needed for comms stack and Default Network)
        agent_bus = getattr(agent, "_bus", None)

        # Create NAc early (needed for comms stack and Default Network)
        nac = None
        try:
            from maxim.decisions.nac import NAc
            nac = NAc()
            self._nac = nac
            self.log.debug("NAc created for causal learning")
        except Exception as e:
            warn("Failed to create NAc: %s", e, logger=self.log)

        # Propagate exploration mode context if set by CLI
        if bool(getattr(self, "_exploration_mode", False)):
            state.data["exploration_mode"] = True
            state.data["exploration_focus"] = str(getattr(self, "_exploration_focus", "") or "")
            state.data["exploration_session_id"] = str(getattr(self, "_exploration_session_id", "") or "")
            state.data["exploration_policy"] = getattr(self, "_exploration_policy", {}) or {}
            state.data["mode"] = "exploration"  # Override mode for MemoryAgent
            # Also update maxim_runtime for consistency
            if isinstance(state.data.get("maxim_runtime"), dict):
                state.data["maxim_runtime"]["mode"] = "exploration"

        memory = build_memory()
        decision_engine = build_decision_engine()

        # Set up ResponseOutput for LLM responses
        from pathlib import Path
        from maxim.utils.response_output import ResponseOutput

        sandbox_path = Path(self.home_dir) / "sandbox"
        tts_engine = None
        speaker_fn = None

        # Check if TTS is enabled (set via environment or config)
        if os.environ.get("MAXIM_TTS_ENABLED", "").lower() in ("1", "true", "yes"):
            try:
                from maxim.models.audio.tts import TTSEngine
                from maxim.utils.audio import _check_local_audio

                tts_model = os.environ.get("MAXIM_TTS_MODEL", "en_US-lessac-medium")
                tts_engine = TTSEngine(model_name=tts_model)
                if tts_engine.is_available:
                    speaker_fn = self.speak  # Uses Reachy speaker with local fallback
                    self.log.info("TTS enabled with model: %s", tts_model)

                    # Check local audio availability for fallback
                    prefer_local = os.environ.get("MAXIM_TTS_LOCAL", "").lower() in ("1", "true", "yes")
                    if prefer_local:
                        self.log.info("TTS using local audio (MAXIM_TTS_LOCAL=1)")
                    elif _check_local_audio():
                        self.log.info("Local audio available as fallback for remote connections")
                else:
                    self.log.warning("TTS model not found, TTS disabled")
                    tts_engine = None
            except Exception as e:
                self.log.warning("Failed to initialize TTS: %s", e)
                tts_engine = None

        response_output = ResponseOutput(
            sandbox_path=sandbox_path,
            logger=self.log,
            tts_engine=tts_engine,
            speaker_fn=speaker_fn,
        )

        # Check if internet access is allowed (from exploration policy or default to True)
        exploration_policy_dict = getattr(self, "_exploration_policy", {}) or {}
        allow_internet = exploration_policy_dict.get("allow_internet", True)

        # Create internet policy getter for tool registry
        def get_internet_policy():
            from maxim.utils.internet_access import InternetAccessPolicy
            return InternetAccessPolicy(enabled=allow_internet)

        # Only pass policy getter if internet is allowed
        internet_policy_getter = get_internet_policy if allow_internet else None

        # Build comms stack if enabled (MAXIM_COMMS_ENABLED env)
        gateway = None
        if os.environ.get("MAXIM_COMMS_ENABLED", "").lower() in ("1", "true", "yes"):
            try:
                from maxim.runtime.bootstrap import build_comms_stack
                gateway, _conv_manager = build_comms_stack(
                    bus=agent_bus,
                    nac=nac,
                )
            except Exception as e:
                warn("Failed to build comms stack: %s", e, logger=self.log)

        registry = build_tool_registry(
            maxim=self,
            response_output=response_output,
            internet_policy_getter=internet_policy_getter,
            gateway=gateway,
        )
        executor = build_executor(registry)
        evaluators = build_evaluators()

        # --- Protocol system ---
        try:
            from maxim.skills.registry import ProtocolRegistry
            from maxim.skills.tools import RunProtocolTool, StopProtocolTool, ListProtocolsTool

            self._protocol_registry = ProtocolRegistry(
                maxim=self,
                tool_registry=registry,
            )

            # Register built-in protocols
            try:
                from maxim.skills.protocols.shredder_segmenter import ShredderSegmenterProtocol
                duration_env = os.getenv("SHREDDER_DURATION_MINUTES", "0")
                try:
                    duration_min = float(duration_env)
                except ValueError:
                    duration_min = 0.0
                interval_env = os.getenv("SHREDDER_HEALTH_INTERVAL", "30")
                try:
                    health_interval = float(interval_env)
                except ValueError:
                    health_interval = 30.0
                self._protocol_registry.register(ShredderSegmenterProtocol(
                    shredder_api_url=os.getenv("SHREDDER_API_URL"),
                    shredder_license_id=os.getenv("SHREDDER_LICENSE_ID"),
                    shredder_api_key=os.getenv("SHREDDER_API_KEY"),
                    shredder_site_id=os.getenv("SHREDDER_SITE_ID"),
                    duration_minutes=duration_min,
                    health_endpoint_url=os.getenv("SHREDDER_HEALTH_URL", ""),
                    health_interval_seconds=health_interval,
                ))
            except Exception as e:
                warn("ShredderSegmenterProtocol not available: %s", e, logger=self.log)

            # Register protocol management tools
            registry.register(RunProtocolTool(self._protocol_registry))
            registry.register(StopProtocolTool(self._protocol_registry))
            registry.register(ListProtocolsTool(self._protocol_registry))

            # Register activation + stop phrases for all protocols (permanent)
            for proto in self._protocol_registry._protocols.values():
                self._protocol_registry._register_phrases(proto.name, proto)

        except Exception as e:
            warn("Failed to initialize protocol system: %s", e, logger=self.log)

        run_id = getattr(self, "run_id", None) or time.strftime("%Y-%m-%d_%H%M%S")

        # Build allowed tools set based on policy (uses allow_internet from above)
        allowed_tools = {
            "read_file",
            "focus_interests",
            "track_target",
            "maxim_command",
            "mode_switch",
            "speak",
            "respond",
            "list_directory",
        }

        # Add internet tools if allowed
        if allow_internet:
            allowed_tools.add("internet_search")
            allowed_tools.add("http_fetch")

        # Add comms tools if gateway is available
        if gateway is not None:
            allowed_tools.add("send_message")
            allowed_tools.add("call_user")

        # Set up autonomy controller with sensible defaults for live mode
        supervision_policy = SupervisionPolicy(
            allowed_tools=allowed_tools,
            forbidden_tools={"execute_file", "delete_file"},
            min_confidence_autonomous=0.7,
            requires_confirmation={"write_file"},
        )
        autonomy_controller = AutonomyController(
            initial_level=AutonomyLevel.SUPERVISED,
            supervision_policy=supervision_policy,
        )

        # Create LLM worker for handling user questions
        llm_worker = None
        try:
            from maxim.models.language.router import LLMRouter, load_llm_config

            llm_config = load_llm_config()
            if llm_config.enabled:
                llm_router = LLMRouter(llm_config)
                # Start warming up the LLM in background (reduces first-request latency)
                llm_router.warmup()
                llm_worker = LLMWorker(
                    llm=llm_router,
                    stale_threshold_s=5.0,
                    n_ctx=llm_router.n_ctx,
                    token_counter=llm_router.get_token_counter(),
                )
                llm_worker.start()
                self.log.info("LLM worker started for user responses")
            else:
                self.log.debug("LLM disabled in config, responses will use fallback")
        except Exception as e:
            warn("Failed to create LLM worker: %s", e, logger=self.log)
            llm_worker = None

        self._llm_worker = llm_worker

        # Wire communication gateway if available
        if gateway is not None:
            agent.wire_communication(gateway=gateway, nac=nac)
            self.log.info("Communication gateway wired")

        # Create FearAgent for safety gating
        fear_agent = None
        try:
            from maxim.agents.fear_agent import FearAgent
            fear_agent = FearAgent(llm=None)  # No LLM needed for basic safety checks
            self._fear_agent = fear_agent
            self.log.debug("FearAgent created for DN safety gating")
        except Exception as e:
            warn("Failed to create FearAgent: %s", e, logger=self.log)

        # Build Default Network for reactive behaviors
        default_network = None
        try:
            default_network = build_default_network(
                maxim=self,
                bus=agent_bus,
                fear_agent=fear_agent,
                nac=nac,
                frame_size=(640, 480),
            )
            if default_network is not None:
                self._default_network = default_network
                self.log.info(
                    "DefaultNetwork built (bus=%s, fear_agent=%s)",
                    "connected" if agent_bus else "none",
                    "enabled" if fear_agent else "none",
                )
        except Exception as e:
            warn("Failed to build DefaultNetwork: %s", e, logger=self.log)
            default_network = None

        # Start capture manager or fall back to vision event stream
        if capture_manager is not None:
            try:
                capture_manager.start()
                self.log.info("CaptureManager started for direct frame access")
            except Exception as e:
                warn("Failed to start CaptureManager: %s", e, logger=self.log)
                capture_manager = None

        # Fall back to JSONL-based stream if no capture manager
        if capture_manager is None:
            self._start_vision_event_stream()

        def _on_step(ctx: dict) -> None:
            tool_result = ctx.get("tool_result")
            action = ctx.get("action") if isinstance(ctx.get("action"), dict) else None
            goal = ctx.get("goal")
            decision = ctx.get("decision") if isinstance(ctx.get("decision"), dict) else None

            output_preview = None
            output_size = None
            try:
                if tool_result is not None:
                    out = getattr(tool_result, "output", None)
                    if isinstance(out, str):
                        output_size = len(out)
                        output_preview = out[:160]
                    elif isinstance(out, dict):
                        output_preview = {k: out[k] for k in list(out)[:6]}
            except Exception:
                output_preview = None

            record = {
                "kind": "agentic_action",
                "event_id": uuid.uuid4().hex,
                "time": float(time.time()),
                "run_id": run_id,
                "agent_name": getattr(agent, "agent_name", getattr(agent, "name", None)),
                "goal": goal,
                "action": action,
                "score": decision.get("score") if isinstance(decision, dict) else None,
                "success": getattr(tool_result, "success", None) if tool_result is not None else None,
                "error": getattr(tool_result, "error", None) if tool_result is not None else None,
                "output_size": output_size,
                "output_preview": output_preview,
                "outcome_code": int(getattr(self, "_outcome_code", 0) or 0),
                "voice_agentic_enabled": bool(getattr(self, "_voice_agentic_enabled", False)),
            }
            self._log_event(record)

        def _worker() -> None:
            try:
                run_agentic_loop(
                    agent,
                    env,
                    state,
                    memory,
                    decision_engine,
                    executor,
                    autonomy_controller=autonomy_controller,
                    llm_worker=llm_worker,
                    default_network=default_network,
                    evaluators=evaluators,
                    max_steps=0,  # Unlimited
                    run_id=run_id,
                    stop_event=stop_event,
                    on_step=_on_step,
                    idle_sleep_s=0.1,
                    target_hz=10.0,  # 10 Hz for responsive CLI handling
                    protocol_registry=self._protocol_registry,
                )
            except Exception as e:
                warn("Agentic runtime loop failed: %s", e, logger=self.log)
            finally:
                # Clean up LLM worker
                if llm_worker is not None:
                    try:
                        llm_worker.stop()
                    except Exception:
                        pass

        t = threading.Thread(target=_worker, name="maxim.agentic", daemon=True)
        self._agentic_thread = t
        t.start()

    def _stop_agentic_runtime(self, *, timeout: float = 10.0) -> None:
        """Stop the agentic runtime with graceful shutdown and force-kill fallback.

        Shutdown sequence:
        1. Signal all threads to stop via events
        2. Wait for graceful shutdown with timeout
        3. Force-terminate any threads that didn't respond

        Args:
            timeout: Total timeout for the entire shutdown sequence (default 10s).
                     Individual components get proportional timeouts.
        """
        import ctypes

        # Track threads that need to be stopped
        threads_to_stop: list[tuple[str, threading.Thread | None]] = []

        # Phase 1: Signal all stop events first (non-blocking)
        try:
            ev = getattr(self, "_agentic_stop_event", None)
            if ev is not None:
                ev.set()
        except Exception:
            pass

        capture_manager = getattr(self, "_capture_manager", None)
        if capture_manager is not None:
            try:
                # Just set the stop event, don't wait yet
                stop_ev = getattr(capture_manager, "_stop_event", None)
                if stop_ev is not None:
                    stop_ev.set()
            except Exception:
                pass

        default_network = getattr(self, "_default_network", None)
        if default_network is not None:
            try:
                # Signal DN to stop
                default_network._running = False
                bg_tasks = getattr(default_network, "_background_tasks", None)
                if bg_tasks is not None:
                    bg_tasks._running = False
                    stop_ev = getattr(bg_tasks, "_stop_event", None)
                    if stop_ev is not None:
                        stop_ev.set()
            except Exception:
                pass

        # Collect all threads
        t = getattr(self, "_agentic_thread", None)
        if t is not None:
            threads_to_stop.append(("agentic", t))

        if capture_manager is not None:
            for attr, name in [("_frame_thread", "capture.frame"),
                               ("_segmentation_thread", "capture.segmentation"),
                               ("_audio_thread", "capture.audio")]:
                thread = getattr(capture_manager, attr, None)
                if thread is not None:
                    threads_to_stop.append((name, thread))
            # Send poison pills to unblock queues
            try:
                seg_queue = getattr(capture_manager, "_segmentation_queue", None)
                if seg_queue is not None:
                    seg_queue.put_nowait(None)
            except Exception:
                pass

        if default_network is not None:
            dn_thread = getattr(default_network, "_thread", None)
            if dn_thread is not None:
                threads_to_stop.append(("default_network", dn_thread))
            bg_tasks = getattr(default_network, "_background_tasks", None)
            if bg_tasks is not None:
                bg_thread = getattr(bg_tasks, "_thread", None)
                if bg_thread is not None:
                    threads_to_stop.append(("background_tasks", bg_thread))

        # Phase 2: Wait for threads to stop gracefully (with per-thread timeout)
        per_thread_timeout = timeout / max(len(threads_to_stop), 1)
        still_alive: list[tuple[str, threading.Thread]] = []

        for name, thread in threads_to_stop:
            if thread is not None and thread.is_alive():
                try:
                    thread.join(timeout=per_thread_timeout)
                    if thread.is_alive():
                        still_alive.append((name, thread))
                        self.log.warning("Thread '%s' did not stop gracefully", name)
                except Exception:
                    still_alive.append((name, thread))

        # Phase 3: Force-terminate threads that didn't respond
        # This uses ctypes to raise SystemExit in the thread
        for name, thread in still_alive:
            try:
                if thread.is_alive():
                    thread_id = thread.ident
                    if thread_id is not None:
                        # Raise SystemExit in the thread
                        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_ulong(thread_id),
                            ctypes.py_object(SystemExit)
                        )
                        if res == 0:
                            self.log.warning("Invalid thread id for '%s'", name)
                        elif res > 1:
                            # Reset if more than one thread was affected
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_ulong(thread_id), None
                            )
                        else:
                            self.log.info("Force-terminated thread '%s'", name)
                        # Give it a moment to terminate
                        thread.join(timeout=0.5)
            except Exception as e:
                self.log.warning("Failed to force-terminate thread '%s': %s", name, e)

        # Deactivate all protocols before cleanup
        if hasattr(self, "_protocol_registry") and self._protocol_registry is not None:
            for proto in list(self._protocol_registry._active.values()):
                self._protocol_registry.deactivate(proto.name)
            self._protocol_registry = None

        # Phase 4: Cleanup references
        self._agentic_thread = None
        self._agentic_stop_event = None
        self._agentic_agent = None
        self._agentic_state = None
        self._state_manager.set_agent(None)

        # Stop capture manager (will be fast since threads already stopped/terminated)
        if capture_manager is not None:
            try:
                capture_manager.stop(timeout=1.0)
            except Exception:
                pass

        # Stop Default Network (will be fast since threads already stopped/terminated)
        if default_network is not None:
            try:
                default_network.stop()
            except Exception:
                pass
        self._default_network = None
        self._capture_manager = None

        # Clear FearAgent reference
        self._fear_agent = None

        self._stop_vision_event_stream(timeout=2.0)

    def wake_up_agentic(self) -> None:
        """Wake up Reachy and transition to exploration mode.

        Called when the wake word "maxim" is detected. This:
        1. Wakes up the robot motors
        2. Switches mode from sleep to exploration
        3. Starts the agentic runtime if available
        """
        try:
            self._voice_agentic_enabled = True
        except Exception:
            pass

        # Wake up the robot
        try:
            mini = getattr(self, "mini", None)
            if mini is not None:
                self._enqueue_motor(mini.wake_up)
                self._woke_up = True
                self.sleeping = False  # Mark as no longer sleeping
        except Exception as e:
            warn("Failed to wake up Reachy: %s", e, logger=self.log)

        # Switch to exploration mode if currently in sleep mode
        # This triggers a live loop restart with vision=True
        current_mode = str(getattr(self, "mode", "") or "").strip().lower()
        if current_mode == "sleep":
            try:
                self.log.info("Waking up from sleep -> requesting exploration mode")
                self._request_mode("exploration")
            except Exception:
                pass

        self._start_agentic_runtime()
        agentic_thread = getattr(self, "_agentic_thread", None)
        if agentic_thread is None or not getattr(agentic_thread, "is_alive", lambda: False)():
            try:
                self._voice_agentic_enabled = False
            except Exception:
                pass
