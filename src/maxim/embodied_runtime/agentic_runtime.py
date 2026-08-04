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
from typing import Any

from maxim.utils.gpu_compat import is_gpu_available
from maxim.utils.logging import warn


def _compute_target_hz(capabilities) -> float:
    """Adapt agentic loop frequency to available hardware.

    Motor control needs real-time updates (30 Hz).
    Vision processing without motors runs at 10 Hz.
    Headless mode uses 2 Hz — LLM inference cycles, event-driven.
    """
    if getattr(capabilities, "has_motor", False):
        return 30.0
    if getattr(capabilities, "has_vision", False):
        return 10.0
    return 2.0


class AgenticRuntimeMixin:
    """Mixin providing agentic runtime lifecycle management for the Maxim class."""

    def _resolve_body_wiring(self, pain_bus: Any, nac: Any) -> "tuple[str | None, Any]":
        """Resolve an optional SEM body to wire into ``build_executor``.

        Returns ``(entity_ref, component_registry)`` when ``robots.yaml``
        declares a body (``config.body``) for this robot AND the pain_bus +
        nac that ``build_executor``'s embodiment path requires are present;
        otherwise ``(None, None)`` (bodiless — the historical default).

        Body-wiring is opt-in via the declaration seam; see
        ``hardware.config.resolve_body_ref``. A *declared but unresolvable*
        body is operator error — logged loudly — but does NOT crash the live
        robot: we fall back to bodiless so the runtime stays up.
        """
        try:
            from maxim.hardware.config import load_robots_config, resolve_body_ref

            robots_cfg = load_robots_config()
            robot_id = getattr(self, "_robot_id", None) or getattr(self, "name", None)
            robot_config = robots_cfg.get(robot_id)
            if robot_config is None:
                # No exact id match. Fall back to the primary ONLY when it is
                # unambiguous — a single robot, or one explicitly marked
                # primary. The runtime name (e.g. "reachy_mini") routinely
                # differs from the yaml key (e.g. "primary"), so this fallback
                # is load-bearing for the default single-robot config. But do
                # NOT adopt the FIRST of several unmarked robots: in a genuine
                # multi-robot file that would wire a *foreign* robot's body
                # onto this runtime. Ambiguous → bodiless + warn.
                has_explicit_primary = any(r.primary for r in robots_cfg.robots)
                if len(robots_cfg.robots) == 1 or has_explicit_primary:
                    robot_config = robots_cfg.get_primary()
                elif robots_cfg.robots:
                    self.log.warning(
                        "robots.yaml has %d robots, none matching id=%r and none "
                        "marked primary; not guessing a body. Running bodiless.",
                        len(robots_cfg.robots),
                        robot_id,
                    )
            # Stash for sibling gates that need the same free-form config
            # dict (the DoA feed's audio_localization opt-out, Stage 2).
            self._resolved_robot_config = robot_config
            body_ref = resolve_body_ref(robot_config)
        except Exception as e:  # config load is best-effort; never block startup
            self.log.debug("body-ref resolution skipped: %s", e)
            return None, None

        if body_ref is None:
            return None, None

        # A body was declared but the embodiment path needs both a pain_bus
        # (Embodiment._publish_pain emits through it) and a nac (failures need
        # a bridge for NAc learning). Surface the gap rather than silently
        # dropping the declared body.
        if pain_bus is None or nac is None:
            self.log.warning(
                "robots.yaml declares body=%r but it cannot be wired (pain_bus=%s, nac=%s). Running bodiless.",
                body_ref,
                pain_bus is not None,
                nac is not None,
            )
            return None, None

        try:
            from maxim.embodiment.component_registry import ComponentRegistry

            registry = ComponentRegistry()
            if not registry.has(body_ref):
                self.log.warning(
                    "robots.yaml declares body=%r but no such SEM component was found. Running bodiless.",
                    body_ref,
                )
                return None, None
            return body_ref, registry
        except Exception as e:
            self.log.warning("Failed to build ComponentRegistry for body=%r: %s. Running bodiless.", body_ref, e)
            return None, None

    def _maybe_start_doa_feed(self, executor: Any, stop_event: "threading.Event") -> None:
        """Start the live DoA → azimuth feed when three gates all pass.

        Stage 2 of live_audio_orient_wiring.md — capability-driven, no env
        var:

        1. the robot seam yields a reader (``get_doa_reader``, probed via
           ``getattr`` so pre-seam plugins keep working);
        2. the wired body declares an ``azimuth`` sensor; and
        3. robots.yaml's free-form config dict does not opt out
           (``config: {audio_localization: false}``).

        Absent any of the three: no thread, no log spam — a robot without a
        mic behaves exactly as today. The thread follows the CaptureManager
        pattern (shared ``stop_event``; joined in ``_stop_agentic_runtime``).
        The percept sink is wired separately (Stage 3) once the loop's
        adapter exists — this feed alone makes ``listen`` return live
        direction, the azimuth drive breach, and body_state renderable.
        """
        # Clear any prior session's feed state FIRST (pre-merge review fold):
        # a stale _doa_sim_adapter could replay last session's bearing into a
        # restarted loop whose gates fail this time, and a dead _doa_thread
        # reference is join-sweep noise.
        self._doa_feed = None
        self._doa_thread = None
        self._doa_sim_adapter = None
        try:
            emb = getattr(executor, "embodiment", None)
            root = getattr(emb, "root", None)
            sensors = getattr(root, "sensors", None) or {}
            if "azimuth" not in sensors:
                return  # bodiless, or a body without sound localization
            robot_config = getattr(self, "_resolved_robot_config", None)
            cfg_dict = getattr(robot_config, "config", None) or {}
            opt_out = cfg_dict.get("audio_localization")
            # Accept YAML false AND plausible hand-edits ("false", "no", ...).
            if opt_out is False or str(opt_out).strip().lower() in ("false", "0", "no", "off"):
                self.log.info("DoA feed disabled via robots.yaml (audio_localization: false)")
                return
            robot = getattr(self, "_robot", None)
            get_reader = getattr(robot, "get_doa_reader", None)
            reader = get_reader() if callable(get_reader) else None
            if reader is None:
                return  # capability absent — exactly a robot without a mic
            from maxim.embodiment.audio_localization import DoAFeed
            from maxim.runtime.sim_adapter import NullSimulationAdapter

            # Stage 3: the feed's percept lane. A caller-held
            # NullSimulationAdapter (is_sim_mode stays False) carries each
            # fresh audio percept into agent_loop §1.16's side-channel;
            # run_agentic_loop receives it via the sim_adapter= kwarg.
            adapter = NullSimulationAdapter()
            self._doa_sim_adapter = adapter

            # Attention weights for the emitted audio percepts, from the same
            # free-form robots.yaml config dict as the other audio keys. The
            # defaults (0.5/0.3) sit AT or BELOW every > 0.5 escalation gate:
            # sound is passively perceived but NEVER reaches the LLM — the
            # plan's passive-by-default decision. Raising audio_salience
            # above 0.5 makes speech escalate via §1.16 B1 (forces an LLM
            # submission), which is the only cognition-path trigger on a
            # no_media, no-typed-input live session. Clamped to [0, 1];
            # malformed values fall back to the passive defaults with a
            # WARNING (silent clamping would hide misconfiguration).
            def _attention_weight(key: str, default: float) -> float:
                raw = cfg_dict.get(key)
                if raw is None:
                    return default
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    self.log.warning("robots.yaml %s=%r is not a number — using %s", key, raw, default)
                    return default
                if not (0.0 <= v <= 1.0):
                    self.log.warning("robots.yaml %s=%r outside [0, 1] — using %s", key, raw, default)
                    return default
                return v

            feed = DoAFeed(
                reader,
                emb,
                stop_event=stop_event,
                percept_sink=adapter.carry_percept,
                agent_id=getattr(self, "agent_id", "reachy"),
                salience=_attention_weight("audio_salience", 0.5),
                novelty=_attention_weight("audio_novelty", 0.3),
                # Capture-frame stamp: the head yaw each reading was taken
                # in, so consumers (focus_on_sound) compute a stable
                # absolute target instead of re-applying a head-relative
                # delta to a pose that has since moved.
                head_yaw_provider=lambda: float(getattr(self, "yaw", 0.0) or 0.0),
                # Body stamp (sem_motor_binding.md): once SEM turns really
                # rotate the body, consumers need the capture-time body
                # yaw to correct targets across body rotation. maxim.
                # body_yaw is degrees, synced from joint index 0.
                body_yaw_provider=lambda: float(getattr(self, "body_yaw", 0.0) or 0.0),
            )
            thread = threading.Thread(target=feed.run, name="doa-feed", daemon=True)
            self._doa_feed = feed
            self._doa_thread = thread
            thread.start()
            self.log.info("DoA feed started (audio_localization capability + azimuth sensor)")
        except Exception as e:
            warn("DoA feed not started: %s", e, logger=self.log)

    def _start_agentic_runtime(self, *, use_capture_manager: bool = True) -> None:
        """Start the agentic runtime.

        Args:
            use_capture_manager: If True, use CaptureManager for direct frame access (Phase 3).
                                If False, fall back to JSONL-based vision event stream.
        """
        existing = getattr(self, "_agentic_thread", None)
        if existing is not None and getattr(existing, "is_alive", lambda: False)():
            return

        _t_boot = time.time()

        # Apply CPU/Metal fallback for the LLM backend before importing it
        self._apply_llm_backend_fallback()

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

        self.log.info("Bootstrap: config + imports (%.1fms)", (time.time() - _t_boot) * 1000)

        stop_event = threading.Event()
        self._agentic_stop_event = stop_event

        # Phase 3: Initialize CaptureManager for direct frame access
        capture_manager = None
        if use_capture_manager and hasattr(self, "_capabilities") and self._capabilities.has_vision:
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

        _t_mem = time.time()
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

        # Build the full bio-pipeline via build_bio_stack (Wave 3,
        # biosystem_unification). Single call replaces ~50 lines of
        # NAc + Hippocampus + SCN + EC + ATL + AngularGyrus + MemoryHub
        # + PainBus construction. DefaultNetwork is constructed later
        # (after FearAgent exists) using bio.pain_bus and bio.nac.
        bio = None
        nac = None
        memory_hub = None
        try:
            from maxim.runtime.bio_stack import build_bio_stack
            from maxim.utils.paths import user_memory

            # ``getattr(self, "agent_id", "reachy")`` — Reachy is currently
            # the sole consumer of AgenticRuntimeMixin and never sets
            # ``self.agent_id`` in production, so the fallback ``"reachy"``
            # is the literal value, not a fallback in the conventional
            # sense. The dynamic pattern matches the pre-existing call site
            # at line 318+ in this file; if a future multi-Reachy or
            # multi-installation deployment needs distinct identities,
            # set ``self.agent_id`` from configuration before bootstrap.
            bio = build_bio_stack(
                persistence_dir=user_memory(),
                agent_id=getattr(self, "agent_id", "reachy"),
            )
            nac = bio.nac
            memory_hub = bio.memory_hub
            self._nac = nac
            self._memory_hub = memory_hub
            agent.wire_memory_hub(memory_hub)
            self.log.info("BioStack created and wired (hippocampus + NAc + SCN + EC + ATL + MemoryHub + PainBus)")
        except Exception as e:
            warn("Failed to create BioStack: %s", e, logger=self.log)

        self.log.info("Bootstrap: memory systems initialized (%.1fms)", (time.time() - _t_mem) * 1000)

        # Create NumericalWorkspace for agent math operations
        try:
            from maxim.math.ips import IPS
            from maxim.math.workspace import NumericalWorkspace

            ips = IPS()
            numerical_workspace = NumericalWorkspace(ips=ips)
            self._numerical_workspace = numerical_workspace
            self.log.debug("NumericalWorkspace created")
        except Exception as e:
            warn("Failed to create NumericalWorkspace: %s", e, logger=self.log)
            numerical_workspace = None

        # Create SkillMatcher for semantic skill prompt matching
        try:
            from maxim.runtime.skill_matcher import SkillMatcher

            data_dir = str(getattr(self, "home_dir", "data") or "data")
            skills_dir = os.path.join(data_dir, "skills")
            agent_skills_dir = os.path.join(data_dir, "agent_skills")

            # Use EC's neural embedder if available
            embedder = None
            if memory_hub is not None:
                ec_obj = getattr(memory_hub, "ec", None)
                if ec_obj is not None:
                    embedder = getattr(ec_obj, "_neural_embedder", None)

            self._skill_matcher = SkillMatcher(
                skills_dir=skills_dir,
                agent_skills_dir=agent_skills_dir,
                embedder=embedder,
                memory_hub=memory_hub,
            )
            self.log.debug(
                "SkillMatcher created (embedder=%s)",
                "enabled" if embedder else "disabled",
            )
        except Exception as e:
            warn("Failed to create SkillMatcher: %s", e, logger=self.log)

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

        # Wire memory systems into the decision engine for adaptive planning.
        # When memory_hub is available, AdaptivePlanner queries all subsystems
        # (EC, NAc, Hippocampus, ConceptContextBuilder) before proposing plans.
        _de_kwargs: dict = {}
        if memory_hub is not None:
            _de_kwargs["nac"] = getattr(memory_hub, "nac", None)
            _de_kwargs["hippocampus"] = getattr(memory_hub, "hippocampus", None)
            _de_kwargs["ec"] = getattr(memory_hub, "ec", None)
            _de_kwargs["atl"] = getattr(memory_hub, "atl", None)
            _de_kwargs["concept_context_builder"] = getattr(memory_hub, "_concept_context_builder", None)
        decision_engine = build_decision_engine(**_de_kwargs)

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

        _t_tools = time.time()
        # Pass maxim=None when headless so robot tools get no-op stubs
        _maxim_for_tools = self if (hasattr(self, "_capabilities") and self._capabilities.has_robot) else None
        registry = build_tool_registry(
            maxim=_maxim_for_tools,
            response_output=response_output,
            internet_policy_getter=internet_policy_getter,
            gateway=gateway,
            state_manager=self._state_manager,
        )
        # --- Learned Tool Index ---
        # Keyword-weighted hashtable for tool relevance scoring in prompts.
        # Constructed BEFORE build_executor so we can pass it as a
        # constructor arg — `build_executor` forwards it to the
        # ToolPainBridge for keyword-weight updates on tool outcomes.
        tool_index = None
        try:
            from maxim.tools.learned_index import LearnedToolIndex

            tool_index = LearnedToolIndex()
            for tool in registry.list():
                tool_obj = registry.get(tool) if isinstance(tool, str) else tool
                if tool_obj is not None:
                    tool_index.register_tool(tool_obj)
            from maxim.utils.paths import user_memory as _user_memory

            tool_index.load(str(_user_memory() / "tool_index.json"))
            self._tool_index = tool_index
            self.log.debug("LearnedToolIndex: %s", tool_index.stats())
        except Exception as e:
            warn("Failed to create LearnedToolIndex: %s", e, logger=self.log)

        # --- Executor + ToolPainBridge ---
        # F4 migration: switched from legacy pain_detector subscription
        # to canonical pain_bus path. bio.pain_bus is constructed by
        # build_bio_stack (Wave 3) with hippocampus + NAc subscribers
        # already wired.
        _pain_bus = bio.pain_bus if bio is not None else None

        # Optional SEM body (Track 1 of embodiment_runtime_wiring.md). The
        # Reachy runtime historically loaded no body, so executor.embodiment
        # was None and the per-iteration drift tick had nothing to advance.
        # When robots.yaml declares one (config.body — the [declaration]
        # seam) AND we have the pain_bus + nac that build_executor's
        # embodiment path requires, wire it so the body is live: drives drift,
        # evaluate_failures() publishes pain, and self_effect/entity tools are
        # available. Opt-in by design — absent a declaration this is a no-op
        # and behavior is byte-identical to before.
        _body_ref, _body_registry = self._resolve_body_wiring(_pain_bus, nac)

        # SEM motor binding (sem_motor_binding.md Phase 1): when a body is
        # declared AND a real robot controller is connected, the orient
        # modulator's affordances dispatch REAL body turns (head riding
        # along) instead of stub success. Sim/headless: factory stays None
        # → SpecModulator stub semantics, byte-identical.
        _motor_factory = None
        if _body_ref is not None:
            try:
                from maxim.hardware.reachy.motor_backend import make_reachy_orient_factory
                from maxim.tools.reachy import _get_robot_from_registry

                _motor_robot = _get_robot_from_registry(None, self)
                if _motor_robot is not None and _motor_robot.is_connected():
                    # Bind ONLY when an azimuth MEASUREMENT stream will exist
                    # (the same gates the DoA feed uses: reader present +
                    # no audio_localization opt-out). Without the feed
                    # owning ``azimuth``, the modeled azimuth self_effect
                    # applies against a sensor pinned at set-point 0 —
                    # every REAL turn then books −1 relief, the phantom
                    # credit mill's mirror image (review fold F2).
                    _cfg_dict = getattr(getattr(self, "_resolved_robot_config", None), "config", None) or {}
                    _opt_out_raw = _cfg_dict.get("audio_localization")
                    _audio_opted_out = _opt_out_raw is False or str(_opt_out_raw).strip().lower() in (
                        "false",
                        "0",
                        "no",
                        "off",
                    )
                    _get_reader = getattr(_motor_robot, "get_doa_reader", None)
                    _reader_ok = callable(_get_reader) and _get_reader() is not None
                    if _audio_opted_out or not _reader_ok:
                        self.log.info(
                            "SEM motor binding skipped: no azimuth measurement stream "
                            "(audio_localization opt-out or no DoA reader) — modeled "
                            "azimuth credit would mis-punish real turns"
                        )
                    else:
                        _motor_factory = make_reachy_orient_factory(_motor_robot, maxim=self)
            except Exception as e:
                self.log.warning("SEM motor binding unavailable: %s", e)

        if nac is not None:
            executor = build_executor(
                registry,
                pain_bus=_pain_bus,
                nac=nac,
                hippocampus=memory_hub.hippocampus if memory_hub else None,
                scn=memory_hub.scn if memory_hub else None,
                tool_index=tool_index,
                distributor=bio.distributor if bio is not None else None,
                agent_id=getattr(self, "agent_id", "reachy"),
                entity_ref=_body_ref,
                component_registry=_body_registry,
                modulator_factory=_motor_factory,
            )
            if _motor_factory is not None:
                self.log.info("SEM motor binding active: 'orient' affordances dispatch real body turns")
            # Review fold E1: the LearnedToolIndex was populated BEFORE
            # build_executor, so the SEM affordance tools it just registered
            # are invisible to the passive-mode FILTERED prompt renderer
            # (which partitions the index's own keyword universe). Register
            # them now — idempotent, preserves learned weights.
            if tool_index is not None:
                try:
                    from maxim.embodiment.tool_bridge import always_active_sem_tools

                    for _sem_tool in always_active_sem_tools(registry):
                        tool_index.register_tool(_sem_tool)
                except Exception as e:
                    self.log.warning("SEM tool-index registration failed: %s", e)
            self._tool_pain_bridge = executor._tool_pain_bridge
            self.log.debug(
                "ToolPainBridge wired via build_executor (pain_bus=%s, body=%s)",
                _pain_bus is not None,
                _body_ref,
            )
        else:
            executor = build_executor(registry, pain_bus=None)
            self._tool_pain_bridge = None

        # Layer 3a (Track 1 of embodiment_runtime_wiring.md): route the
        # executor's Embodiment into the memory hub so memory_agent's
        # format_body_state_for_prompt populates StructuredContext.body_state
        # (the prompt half — body_state section + Acting Coach Layers 2+4).
        # The Reachy runtime builds its agent directly and so bypasses the
        # AgentFactory seam (_maybe_wire_body_state); replicate its gate here.
        # Only fires when a body was actually wired (executor.embodiment
        # present) AND the same body_state_prompt_enabled() flag is set, so the
        # live-robot prompt is unchanged by default (opt-in, consistent with
        # the Exp 44 ablation seam).
        try:
            from maxim.integration.memory_hub import body_state_prompt_enabled

            _exec_embodiment = getattr(executor, "embodiment", None)
            if _exec_embodiment is not None and memory_hub is not None and body_state_prompt_enabled():
                memory_hub.embodiment = _exec_embodiment
                self.log.debug("body_state wiring: memory_hub.embodiment set (Layer 3a)")
        except Exception as e:
            self.log.debug("body_state wiring skipped: %s", e)

        # Stage 2 (live_audio_orient_wiring.md): live DoA → azimuth feed.
        # Triple-gated inside; a no-op for bodiless / mic-less / opted-out.
        self._maybe_start_doa_feed(executor, stop_event)

        evaluators = build_evaluators()

        # Register MathTool if NumericalWorkspace and AngularGyrus are available
        if numerical_workspace is not None and memory_hub is not None:
            try:
                from maxim.tools.math_tool import MathTool

                math_ips = numerical_workspace._ips
                math_ag = getattr(memory_hub, "angular_gyrus", None)
                if math_ag is not None:
                    math_tool = MathTool(
                        ips=math_ips,
                        angular_gyrus=math_ag,
                        workspace=numerical_workspace,
                    )
                    registry.register(math_tool)
                    self.log.debug("MathTool registered")
            except Exception as e:
                warn("Failed to register MathTool: %s", e, logger=self.log)

        self.log.info("Bootstrap: tool registry built (%.1fms)", (time.time() - _t_tools) * 1000)

        # --- Provenance system ---
        self._provenance_collector = None
        try:
            from maxim.provenance.collector import ProvenanceCollector
            from maxim.provenance.types import ProvenanceVerbosity

            prov_verbosity = int(
                os.getenv(
                    "MAXIM_PROVENANCE_VERBOSITY",
                    str(getattr(self, "provenance_verbosity", 1)),
                )
            )
            verbosity = ProvenanceVerbosity(min(prov_verbosity, 2))
            collector = ProvenanceCollector(verbosity=verbosity)

            prov_persist = os.getenv("MAXIM_PROVENANCE_PERSIST", "1") != "0"
            if prov_persist:
                from maxim.provenance.store import ProvenanceStore

                data_dir = str(getattr(self, "home_dir", "data") or "data")
                collector._store = ProvenanceStore(
                    base_dir=os.path.join(data_dir, "provenance"),
                )

            agent.wire_provenance(collector)
            self._provenance_collector = collector

            from maxim.tools.explain import ExplainTool

            registry.register(ExplainTool(collector))
        except Exception as e:
            warn("Failed to initialize provenance system: %s", e, logger=self.log)

        # --- Introspection tools ---
        # Expose biological subsystems as read-only LLM-callable tools
        try:
            from maxim.tools.introspection import (
                MemoryRecallTool,
                PredictOutcomeTool,
                CausalLinksTool,
                PainHistoryTool,
                TemporalPatternsTool,
                EnergyStatusTool,
                ConceptQueryTool,
                SceneSummaryTool,
                SimilaritySearchTool,
                SystemStatsTool,
            )

            if memory_hub is not None:
                registry.register(MemoryRecallTool(hippocampus=memory_hub.hippocampus))
                registry.register(SimilaritySearchTool(ec=memory_hub.ec))
                registry.register(TemporalPatternsTool(scn=memory_hub.scn))
                if memory_hub.atl is not None:
                    registry.register(ConceptQueryTool(atl=memory_hub.atl))

            if nac is not None:
                registry.register(PredictOutcomeTool(nac=nac))
                registry.register(CausalLinksTool(nac=nac))

            pain_detector = getattr(self, "_pain_detector", None)
            fear_agent = locals().get("fear_agent")
            if pain_detector is not None or fear_agent is not None:
                registry.register(
                    PainHistoryTool(
                        pain_detector=pain_detector,
                        fear_agent=fear_agent,
                    )
                )

            energy_tracker = getattr(self, "_energy_tracker", None)
            if energy_tracker is not None:
                registry.register(EnergyStatusTool(energy_tracker=energy_tracker))

            # Scene tools only when vision subsystems available
            default_network = getattr(self, "_default_network", None)
            if default_network is not None:
                _salience = getattr(default_network, "_salience_network", None)
                _attention = getattr(default_network, "_attention_network", None)
                if _salience is not None or _attention is not None:
                    registry.register(
                        SceneSummaryTool(
                            salience_network=_salience,
                            attention_network=_attention,
                        )
                    )

            # System stats always available (works with whatever subsystems exist)
            registry.register(
                SystemStatsTool(
                    hippocampus=memory_hub.hippocampus if memory_hub else None,
                    nac=nac,
                    ec=memory_hub.ec if memory_hub else None,
                    atl=memory_hub.atl if memory_hub else None,
                    energy_tracker=energy_tracker,
                    pain_detector=pain_detector,
                    significance_learner=getattr(self, "_significance_learner", None),
                )
            )

            self.log.info("Introspection tools registered")
        except Exception as e:
            warn("Failed to register introspection tools: %s", e, logger=self.log)

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
                self._protocol_registry.register(
                    ShredderSegmenterProtocol(
                        shredder_api_url=os.getenv("SHREDDER_API_URL"),
                        shredder_license_id=os.getenv("SHREDDER_LICENSE_ID"),
                        shredder_api_key=os.getenv("SHREDDER_API_KEY"),
                        shredder_site_id=os.getenv("SHREDDER_SITE_ID"),
                        duration_minutes=duration_min,
                        health_endpoint_url=os.getenv("SHREDDER_HEALTH_URL", ""),
                        health_interval_seconds=health_interval,
                    )
                )
            except Exception as e:
                warn("ShredderSegmenterProtocol not available: %s", e, logger=self.log)

            # Register protocol management tools
            registry.register(RunProtocolTool(self._protocol_registry))
            registry.register(StopProtocolTool(self._protocol_registry))
            registry.register(ListProtocolsTool(self._protocol_registry))

            # Register activation + stop phrases for all protocols (permanent)
            for proto in self._protocol_registry._protocols.values():
                self._protocol_registry._register_phrases(proto.name, proto)

            # Wire ProtocolRegistry to MemoryAgent for skill name injection (A7.1)
            agent.memory._protocol_registry = self._protocol_registry

            # Wire skill registry to ConceptContextBuilder for discovery (A7.5)
            if memory_hub and memory_hub._concept_context_builder:
                skills = self._protocol_registry.all_skills()
                memory_hub._concept_context_builder.set_skill_registry(skills)

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
            "math",
        }

        # Add internet tools if allowed
        if allow_internet:
            allowed_tools.add("internet_search")
            allowed_tools.add("http_fetch")

        # Add comms tools if gateway is available
        if gateway is not None:
            allowed_tools.add("send_message")
            allowed_tools.add("call_user")

        # The agent's OWN body is not a mode privilege (sem_motor_binding.md
        # Phase 1): the body's ALWAYS-ACTIVE affordances (the orient turn_*
        # family — reflexive vocabulary, not goal-gated actions) execute
        # without a confirmation stop — the live session that motivated
        # this saw the LLM's correct body-turn strategy die at "requires
        # approval" (and the live confirmation path can deadlock).
        # Deliberately NOT via the frozen ALWAYS_ALLOWED_TOOLS set — this
        # is per-session, derived from the actually-wired body. Narrowed to
        # always_active per the review fold (the unfiltered ~30-tool set
        # was a broader confirmation-free grant than the capability needs).
        try:
            from maxim.embodiment.tool_bridge import always_active_sem_tools

            for _sem_tool in always_active_sem_tools(registry):
                allowed_tools.add(_sem_tool.name)
        except Exception as e:
            self.log.warning("SEM allowed-tools union failed: %s", e)

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
        _t_llm = time.time()
        llm_worker = None
        lane_backend_manager = None
        try:
            from maxim.models.language.router import LLMRouter, load_llm_config
            from maxim.runtime.lane_backends import build_primary_router

            # Build infer-lane LLM via the shared multi-LLM factory. Handles
            # capability-driven profile selection (Phase 2), env overrides
            # (Phase 4), and gate enforcement (Phase 3+4). One place owns this.
            caps = getattr(self, "_capabilities", None)
            llm_router, lane_backend_manager = build_primary_router(
                capabilities=caps,
                logger=self.log,
            )
            if llm_router is None:
                llm_config = load_llm_config()
                if llm_config.enabled:
                    llm_router = LLMRouter(llm_config)

            if llm_router is not None and llm_router.enabled():
                # Start warming up the LLM in background (reduces first-request latency)
                llm_router.warmup()
                llm_worker = LLMWorker(
                    llm=llm_router,
                    stale_threshold_s=5.0,
                    n_ctx=llm_router.n_ctx,
                    token_counter=llm_router.get_token_counter(),
                    tool_index=tool_index,
                )
                # Embodied identity (2026-08-01 live deep-dive fold): the
                # live path never set is_embodied, so a robot WITH a wired
                # SEM body was prompted as "You are Maxim, a robot
                # assistant." — the exact string prompt_builder documents
                # as causing respond-loops on 14B-class models. Mirror the
                # cli.py / orchestrator producers: embodied identity
                # whenever the executor carries a body. (Acting-coach
                # wiring stays a separate, flag-gated decision.)
                if getattr(executor, "embodiment", None) is not None:
                    llm_worker.is_embodied = True
                llm_worker.start()
                self.log.info("LLM worker started for user responses")
            else:
                self.log.debug("LLM disabled in config, responses will use fallback")
        except Exception as e:
            warn("Failed to create LLM worker: %s", e, logger=self.log)
            llm_worker = None

        self._llm_worker = llm_worker
        self._lane_backend_manager = lane_backend_manager
        self.log.info("Bootstrap: LLM worker ready (%.1fms)", (time.time() - _t_llm) * 1000)

        # Share LLM backend with ExecAgent to avoid loading a second model
        # (cleanup #3: double LLM load fix)
        if llm_worker is not None and hasattr(agent, "exec_agent"):
            exec_agent = agent.exec_agent
            if exec_agent._router is None:
                exec_agent._router = llm_router
            if exec_agent._llm_worker is None:
                exec_agent._llm_worker = llm_worker
                exec_agent._owns_llm_worker = False

        # Wire communication gateway if available
        if gateway is not None:
            agent.wire_communication(gateway=gateway, nac=nac)
            self.log.info("Communication gateway wired")

        # Create FearAgent for safety gating (both DN movement and tool calls)
        fear_agent = None
        try:
            from maxim.agents.fear_agent import FearAgent
            from maxim.runtime.fear_gate import FearGatedExecutor

            # Use LLM router if available for deeper code analysis
            llm_for_fear = getattr(agent, "_llm", None)
            fear_agent = FearAgent(llm=llm_for_fear)
            self._fear_agent = fear_agent

            # Wrap executor with FearAgent gating (independent of DefaultNetwork)
            executor = FearGatedExecutor(executor, fear_agent)
            self.log.info("FearGatedExecutor active — all tool calls reviewed by FearAgent")
        except Exception as e:
            warn("Failed to create FearAgent/FearGatedExecutor: %s", e, logger=self.log)

        # Build Default Network for reactive behaviors.
        # PainBus is already constructed inside build_bio_stack (Wave 3) —
        # use bio.pain_bus directly. This replaces the separate build_pain_bus
        # call that existed pre-Wave-3 for Gap B closure.
        default_network = None
        has_robot = hasattr(self, "_capabilities") and self._capabilities.has_robot
        if not has_robot:
            self.log.info("Headless mode: DefaultNetwork builds without motor control")
        try:
            default_network = build_default_network(
                nac=nac,
                maxim=self if has_robot else None,
                bus=agent_bus,
                pain_bus=bio.pain_bus if bio is not None else None,
                fear_agent=fear_agent,
                frame_size=(640, 480),
                # Capability truth: no camera → no idle visual exploration
                # (the phantom look_at generator; 2026-08-01 fold).
                has_vision=bool(getattr(getattr(self, "_capabilities", None), "has_vision", True)),
            )
            if default_network is not None:
                self._default_network = default_network
                self.log.info(
                    "DefaultNetwork built (bus=%s, fear_agent=%s, pain_bus=%s)",
                    "connected" if agent_bus else "none",
                    "enabled" if fear_agent else "none",
                    "injected" if bio is not None else "none",
                )
        except Exception as e:
            warn("Failed to build DefaultNetwork: %s", e, logger=self.log)
            default_network = None

        # Wire MemoryHub bridges to DefaultNetwork subsystems
        if memory_hub is not None and default_network is not None:
            try:
                memory_hub.connect(
                    spatial=getattr(default_network, "_spatial_map", None),
                    attention=getattr(default_network, "_attention_network", None),
                    salience=getattr(default_network, "_salience_network", None),
                    fear_agent=fear_agent,
                    novelty_tracker=getattr(default_network, "_novelty_tracker", None),
                )
                self.log.info("MemoryHub bridges connected to DefaultNetwork")
            except Exception as e:
                warn("Failed to connect MemoryHub bridges: %s", e, logger=self.log)
        elif memory_hub is not None:
            # DN not available but we can still create core bridges (planning, escalation, fear)
            try:
                memory_hub.connect(fear_agent=fear_agent)
                self.log.info("MemoryHub core bridges connected (no DefaultNetwork)")
            except Exception as e:
                warn("Failed to connect MemoryHub core bridges: %s", e, logger=self.log)

        # PainBus → Hippocampus wiring is handled by build_bio_stack
        # (Wave 3). The bio stack's PainBus already has hippocampus + NAc
        # subscribers wired. DN receives the injected bus as a consumer.

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

            # Learned tool index: record surfaced-but-unused decay signal
            if tool_index is not None and action and goal:
                tool_name = action.get("tool_name", "")
                goal_text = str(goal)
                # surfaced_tools comes from the LLMRequest (set by prompt builder)
                surfaced = []
                if decision and isinstance(decision, dict):
                    plan = decision.get("plan")
                    if hasattr(plan, "planning_context") and plan.planning_context:
                        surfaced = getattr(plan.planning_context, "_surfaced_tools", [])
                # Also check state for surfaced tools from last prompt build
                if not surfaced:
                    surfaced = state.data.pop("_surfaced_tools", [])
                if surfaced and tool_name and goal_text:
                    try:
                        tool_index.record_surfaced_but_unused(goal_text, surfaced, tool_name)
                    except Exception:
                        pass

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
                    hippocampus=memory_hub.hippocampus if memory_hub else None,
                    memory_hub=memory_hub,
                    evaluators=evaluators,
                    max_steps=0,  # Unlimited
                    run_id=run_id,
                    stop_event=stop_event,
                    on_step=_on_step,
                    idle_sleep_s=0.1,
                    target_hz=_compute_target_hz(self._capabilities) if hasattr(self, "_capabilities") else 10.0,
                    protocol_registry=self._protocol_registry,
                    # Stage 3 (live_audio_orient_wiring.md): the DoA feed's
                    # adapter, when the feed started; None → loop builds its
                    # own NullSimulationAdapter, byte-identical to before.
                    sim_adapter=getattr(self, "_doa_sim_adapter", None),
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

        self.log.info("Bootstrap: total startup (%.1fms)", (time.time() - _t_boot) * 1000)

        t = threading.Thread(target=_worker, name="maxim.agentic", daemon=True)
        self._agentic_thread = t
        t.start()

    def _apply_llm_backend_fallback(self) -> None:
        """Apply CPU/Metal fallback for the LLM backend when no GPU is available.

        - On systems with native GPU: no-op.
        - On macOS with llama.cpp configured: log "Metal is fine" and no-op
          (llama.cpp uses Metal automatically).
        - Otherwise: switch the default profile to ``smollm-1.7b-instruct`` and
          force ``MAXIM_LLM_N_GPU_LAYERS=0`` so the runtime doesn't try to load
          a model it can't run.
        """
        if is_gpu_available():
            return

        # Check whether the configured profile uses llama.cpp (Metal-friendly)
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
        except Exception as e:
            self.log.debug("Could not inspect LLM config for backend fallback: %s", e)

        if using_llama_cpp:
            self.log.info("Using llama.cpp backend with native Metal GPU support")
            return

        cuda_hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        if cuda_hidden:
            self.log.info("GPU hidden for compatibility - agentic runtime will use CPU (slower)")
        else:
            self.log.warning("No GPU available - agentic runtime will use CPU (slower)")

        os.environ.setdefault("MAXIM_LLM_PROFILE", "smollm-1.7b-instruct")
        os.environ.setdefault("MAXIM_LLM_N_GPU_LAYERS", "0")
        self.log.info("Using CPU-friendly LLM: smollm-1.7b-instruct")

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

        # DoA feed poll thread (Stage 2) — rides _agentic_stop_event (already
        # set in Phase 1); gated_azimuth's stop_event makes the join fast.
        doa_thread = getattr(self, "_doa_thread", None)
        if doa_thread is not None:
            threads_to_stop.append(("doa.feed", doa_thread))

        if capture_manager is not None:
            for attr, name in [
                ("_frame_thread", "capture.frame"),
                ("_segmentation_thread", "capture.segmentation"),
                ("_audio_thread", "capture.audio"),
            ]:
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
                            ctypes.c_ulong(thread_id), ctypes.py_object(SystemExit)
                        )
                        if res == 0:
                            self.log.warning("Invalid thread id for '%s'", name)
                        elif res > 1:
                            # Reset if more than one thread was affected
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
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

        # Clean up MemoryHub reference
        # (session end + consolidation handled by agent_loop.py on_session_end)
        hub = getattr(self, "_memory_hub", None)
        if hub is not None:
            # Shut down concept extractor worker thread if running
            extractor = getattr(hub, "_concept_extractor", None)
            if extractor is not None:
                try:
                    extractor.shutdown()
                except Exception:
                    pass
        self._memory_hub = None

        # Persist learned tool index
        tool_index = getattr(self, "_tool_index", None)
        if tool_index is not None:
            try:
                from maxim.utils.paths import user_memory as _user_memory

                tool_index.save(str(_user_memory() / "tool_index.json"))
            except Exception as e:
                warn("Failed to save tool index: %s", e, logger=self.log)
        self._tool_index = None
        self._tool_pain_bridge = None

        # Clear provenance collector reference
        self._provenance_collector = None

        # Clear FearAgent reference
        self._fear_agent = None

        # Clear NumericalWorkspace and SkillMatcher references
        self._numerical_workspace = None
        self._skill_matcher = None

        # Stop LLM backends and kill the auto-spawned server so it releases
        # VRAM immediately rather than lingering until the atexit handler fires
        # (which may not run if shutdown hangs elsewhere).
        mgr = getattr(self, "_lane_backend_manager", None)
        if mgr is not None:
            try:
                mgr.unload_all()
            except Exception:
                pass
        self._lane_backend_manager = None

        try:
            from maxim.runtime.lane_backends import stop_active_spawner

            stop_active_spawner()
        except Exception:
            pass

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
