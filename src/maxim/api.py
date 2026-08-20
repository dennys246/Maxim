"""Public verb-based API for pymaxim.

Verb-based functions that map directly to user intent:

    import maxim

    maxim.configure(verbosity=2)
    maxim.run(model="mistral-7b")
    maxim.imagine(goal="test safety")
    maxim.connect("reachy_mini")
    maxim.diagnose()
    maxim.observe("memory")

Each function is a thin facade over existing internals.  Heavy imports
are deferred to call time so ``import maxim`` stays fast regardless of
which optional dependencies are installed.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.doctor.checks import CheckResult
    from maxim.doctor.platform_detect import PlatformInfo
    from maxim.hardware.controller import RobotController
    from maxim.integration.recall import CuratedRecall
    from maxim.session import Session

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# configure
# ─────────────────────────────────────────────────────────────────────────────


def configure(
    *,
    verbosity: int = 1,
    log_file: str | None = None,
    debug: str | None = None,
    show: str | None = None,
    display: str | None = None,
    interactive: str | None = None,
) -> None:
    """Configure Maxim runtime settings.

    Call this before other verbs to control logging output.  Safe to call
    multiple times -- subsequent calls update the active configuration.

    Args:
        verbosity: 0=quiet, 1=normal, 2=verbose, 3=debug.
        log_file: Path to log file (None = stdout only).
        debug: Subsystem trace filter.  Comma-separated list of subsystem
            names (e.g. ``"hippo"``, ``"nac"``, ``"hippo,nac"``).
            ``None`` means no subsystem tracing; pass ``"all"`` for everything.
        show: Channel filter for simulation output.  Comma-separated:
            ``"bio"`` (hippocampus/NAc/SCN/ATL/pain/fear),
            ``"exec"`` (tool execution/LLM), ``"sim"`` (percepts/scenes),
            ``"memory"``, ``"safety"``, ``"all"`` (default).
            Also settable via ``$MAXIM_SHOW_CHANNELS`` env var.

        display: Output detail level.  ``"clean"`` (narrative only, default),
            ``"bio"`` (+ memory/learning annotations), ``"debug"``
            (+ full system traces).
        interactive: Input prompting.  ``"on"`` (prompt for user input),
            ``"off"`` (headless — no prompts, use policy defaults).

    Example::

        maxim.configure(display="bio")       # Show bio annotations
        maxim.configure(display="clean")     # Narrative only (default)
        maxim.configure(interactive="off")   # Headless — no prompts
    """
    # ── Validate inputs ────────────────────────────────────────────────
    if not isinstance(verbosity, int) or not (0 <= verbosity <= 3):
        logger.warning(
            "verbosity=%r is outside 0-3 range; clamping to nearest bound. (0=quiet, 1=normal, 2=verbose, 3=debug)",
            verbosity,
        )
        verbosity = max(0, min(3, int(verbosity)))

    _VALID_SHOW_CHANNELS = {"bio", "exec", "sim", "memory", "safety", "all"}
    if show is not None:
        unknown = {ch.strip().lower() for ch in show.split(",") if ch.strip().lower() not in _VALID_SHOW_CHANNELS}
        if unknown:
            logger.warning(
                "Unknown show channel(s): %s. Valid: %s",
                ", ".join(sorted(unknown)),
                ", ".join(sorted(_VALID_SHOW_CHANNELS)),
            )

    _VALID_DEBUG_SUBSYSTEMS = {"hippo", "nac", "atl", "ec", "scn", "pain", "fear", "default_net", "all"}
    if debug is not None:
        unknown = {t.strip().lower() for t in debug.split(",") if t.strip().lower() not in _VALID_DEBUG_SUBSYSTEMS}
        if unknown:
            logger.warning(
                "Unknown debug subsystem(s): %s. Valid: %s",
                ", ".join(sorted(unknown)),
                ", ".join(sorted(_VALID_DEBUG_SUBSYSTEMS)),
            )

    # ── Apply configuration ──────────────────────────────────────────
    from maxim.utils.logging import configure_logging
    from maxim.utils.structured_logging import configure_agentic_verbosity

    configure_logging(verbosity, log_file=log_file, force=True)
    configure_agentic_verbosity(verbosity, console_output=verbosity >= 2)

    # Display tier and interactive mode
    if display is not None:
        from maxim.simulation.sim_logger import set_display_tier

        set_display_tier(display)

    if interactive is not None:
        from maxim.simulation.sim_logger import set_interactive_mode

        set_interactive_mode(interactive)

    # Channel filter for simulation output (legacy — --display is preferred)
    if show is not None:
        from maxim.simulation.sim_logger import set_show_channels

        set_show_channels(show)

    # Subsystem trace env vars (read by individual subsystems at runtime)
    if debug is not None:
        traces = [t.strip().lower() for t in debug.split(",")]
        all_traces = "all" in traces
        for subsystem in ("hippo", "nac", "atl", "ec", "scn", "pain", "fear", "default_net"):
            env_key = f"MAXIM_{subsystem.upper()}_TRACE"
            os.environ[env_key] = "1" if (all_traces or subsystem in traces) else ""


# ─────────────────────────────────────────────────────────────────────────────
# Model validation + discovery
# ─────────────────────────────────────────────────────────────────────────────


_API_DEFAULT_MODEL = "mistral-7b"


def _restore_env(saved: dict[str, str | None]) -> None:
    """Restore env vars from a snapshot. ``None`` means the key was absent."""
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _resolve_model(model: str) -> str:
    """Return the effective model, checking persisted choice when caller
    passed the default.  Allows ``maxim.run()`` (no explicit model) to
    reuse the last ``--llm`` / ``maxim.run(model=...)`` selection.
    """
    if model != _API_DEFAULT_MODEL:
        # User explicitly chose — persist for next session
        try:
            from maxim.runtime.lane_backends import _write_persisted_model

            from maxim.models.language.config import normalize_llm_profile

            _write_persisted_model(normalize_llm_profile(model) or model)
        except Exception as e:
            logger.debug("Could not persist model preference: %s", e)
        return model
    # Default was used — check for a persisted preference
    try:
        from maxim.runtime.lane_backends import _read_persisted_model

        persisted = _read_persisted_model()
        if persisted:
            return persisted
    except Exception as e:
        logger.debug("Could not read persisted model preference: %s", e)
    return model


def _validate_model(model: str) -> None:
    """Validate that a model profile is available, with actionable errors.

    Raises ``ConfigurationError`` with install/export instructions if the
    model is unknown, the required SDK is missing, or the API key is unset.
    """
    from maxim.exceptions import ConfigurationError
    from maxim.models.language.config import _BUILTIN_PROFILES, normalize_llm_profile

    canonical = normalize_llm_profile(model)
    profile = _BUILTIN_PROFILES.get(canonical)

    if profile is None:
        from maxim.models.language.config import list_llm_profiles

        available = list_llm_profiles()
        raise ConfigurationError(
            f"Unknown model '{model}'. Available profiles:\n"
            f"  {', '.join(available[:15])}\n"
            f"Run: maxim --list-models  (or maxim.list_models() from Python)"
        )

    # Cloud models: check SDK + API key
    if profile.get("cloud"):
        backend = profile.get("backend", "")
        api_key_env = profile.get("api_key_env", "")

        if backend == "anthropic":
            try:
                import anthropic  # noqa: F401
            except ImportError:
                raise ConfigurationError(
                    f"Model '{model}' requires the Anthropic SDK.\n  Fix: pip install 'pymaxim[llm-anthropic]'"
                )
        elif backend == "openai":
            try:
                import openai  # noqa: F401
            except ImportError:
                raise ConfigurationError(
                    f"Model '{model}' requires the OpenAI SDK.\n  Fix: pip install 'pymaxim[llm-openai]'"
                )

        if api_key_env and not os.environ.get(api_key_env):
            raise ConfigurationError(
                f"Model '{model}' requires {api_key_env} to be set.\n  Fix: export {api_key_env}=<your-key>"
            )


@dataclass
class ModelInfo:
    """Information about an available LLM profile."""

    name: str
    backend: str
    cloud: bool = False
    api_key_env: str = ""
    context_length: int = 0
    downloaded: bool = False
    ready: bool = False

    def __repr__(self) -> str:
        status = "ready" if self.ready else ("downloaded" if self.downloaded else "not downloaded")
        return f"ModelInfo({self.name!r}, backend={self.backend!r}, [{status}])"


def list_models() -> dict[str, list[ModelInfo]]:
    """Return available models grouped by type.

    Returns a dict with ``"local"`` and ``"cloud"`` keys, each containing
    a list of :class:`ModelInfo` objects.  The ``downloaded`` and ``ready``
    fields indicate whether local models have their GGUF on disk and
    whether cloud models have their API key configured.

    Example::

        models = maxim.list_models()
        for m in models["local"]:
            status = "✓" if m.downloaded else "✗"
            print(f"  {status} {m.name}")
        for m in models["cloud"]:
            status = "✓" if m.ready else "needs key"
            print(f"  {status} {m.name}")
    """
    from maxim.models.language.config import _BUILTIN_PROFILES
    from maxim.runtime.lane_backends import _profile_has_local_file

    local: list[ModelInfo] = []
    cloud: list[ModelInfo] = []

    for name, profile in sorted(_BUILTIN_PROFILES.items()):
        is_cloud = bool(profile.get("cloud"))
        env = profile.get("api_key_env", "")
        if is_cloud:
            key_set = bool(os.environ.get(env)) if env else False
            info = ModelInfo(
                name=name,
                backend=profile.get("backend", "unknown"),
                cloud=True,
                api_key_env=env,
                context_length=profile.get("n_ctx", 0),
                downloaded=True,  # cloud models don't need downloads
                ready=key_set,
            )
            cloud.append(info)
        else:
            has_file = _profile_has_local_file(name)
            info = ModelInfo(
                name=name,
                backend=profile.get("backend", "unknown"),
                cloud=False,
                context_length=profile.get("n_ctx", 0),
                downloaded=has_file,
                ready=has_file,
            )
            local.append(info)

    return {"local": local, "cloud": cloud}


def download_model(name: str) -> bool:
    """Download a local LLM model by profile name.

    Args:
        name: Model profile name (e.g. ``"mistral-7b"``, ``"qwen2.5-14b-instruct"``).

    Returns:
        ``True`` if the download succeeded, ``False`` otherwise.

    Example::

        maxim.download_model("qwen2.5-14b-instruct")
    """
    from maxim.models.language.config import normalize_llm_profile

    canonical = normalize_llm_profile(name) or name
    try:
        from maxim.models.download import download_llm

        return download_llm(canonical)
    except Exception as e:
        logger.error("Failed to download model %r: %s", canonical, e)
        return False


def delete_model(name: str) -> bool:
    """Delete a downloaded local LLM model to free disk space.

    Args:
        name: Model profile name (e.g. ``"mistral-7b"``, ``"qwen2.5-14b-instruct"``).

    Returns:
        ``True`` if the file was deleted, ``False`` if not found.

    Example::

        models = maxim.list_models()
        for m in models["local"]:
            if m.downloaded and m.name != "smollm-1.7b-instruct":
                maxim.delete_model(m.name)
    """
    from maxim.models.language.config import normalize_llm_profile

    canonical = normalize_llm_profile(name) or name
    try:
        from maxim.models.download import delete_llm

        return delete_llm(canonical)
    except Exception as e:
        logger.error("Failed to delete model %r: %s", canonical, e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# run
# ─────────────────────────────────────────────────────────────────────────────


def run(
    model: str = _API_DEFAULT_MODEL,
    *,
    goal: str | None = None,
    headless: bool = True,
    robot: str | None = None,
    home_dir: str | None = None,
    verbosity: int = 1,
    learning: bool = True,
    auto_curate: bool = False,
    curate_genre: str = "fantasy",
) -> None:
    """Run Maxim's agentic cycle.

    Bootstraps the full agent pipeline (LLM router, memory systems,
    planning, tools, safety) and enters the main loop.  Blocks until
    the user interrupts (Ctrl+C) or the goal is completed.

    Bio-learning (episodic memory, causal learning, pain attribution)
    is **enabled by default**.  Memories persist to ``~/.maxim/sessions/``
    across sessions.  Pass ``learning=False`` to disable.

    Args:
        model: LLM profile name (e.g. ``"mistral-7b"``, ``"claude-sonnet"``).
        goal: Optional goal string.  If provided, the agent works toward
            it with a utility prompt.  If ``None``, enters interactive mode.
        headless: If ``True`` (default), run without robot hardware.
        robot: Robot type to connect (e.g. ``"reachy_mini"``).  Requires
            the corresponding package to be installed.
        home_dir: Data/persistence directory (default ``~/.maxim``).
        verbosity: Logging verbosity (0-3).
        learning: Enable bio-learning (episodic memory, causal learning,
            pain attribution).  Default ``True``.  Pass ``False`` to run
            without persistent learning.
        auto_curate: If True, run auto-curation before startup to fill
            genre/category coverage gaps via the Asset Foundry.
        curate_genre: Genre for auto-curation (default ``"fantasy"``).

    Raises:
        maxim.exceptions.ConfigurationError: If the requested model
            is not available (missing files or API key).
    """
    model = _resolve_model(model)
    _validate_model(model)

    import threading

    from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SupervisionPolicy
    from maxim.agents.llm_worker import LLMWorker
    from maxim.agents.maxim_agent import MaximAgent
    from maxim.runtime.agent_loop import run_agentic_loop
    from maxim.runtime.bootstrap import (
        build_decision_engine,
        build_environment,
        build_evaluators,
        build_executor,
        build_memory,
        build_state,
        build_tool_registry,
    )
    from maxim.runtime.lane_backends import build_primary_router

    configure(verbosity=verbosity)

    # Pre-startup auto-curation
    if auto_curate:
        try:
            from maxim.simulation.foundry import auto_curate as _auto_curate

            _auto_curate(genre=curate_genre)
        except Exception as e:
            logger.warning("Auto-curation failed: %s", e)
            print(f"  WARNING: Auto-curation failed ({e})", file=sys.stderr)

    effective_home = os.path.expanduser(home_dir or "~/.maxim")
    os.makedirs(effective_home, exist_ok=True)

    # ── LLM router ───────────────────────────────────────────────────────
    _saved_env = {k: os.environ.get(k) for k in ("MAXIM_LLM_ENABLED", "MAXIM_LLM_PROFILE")}
    os.environ.setdefault("MAXIM_LLM_ENABLED", "1")
    os.environ["MAXIM_LLM_PROFILE"] = model  # explicit model= must win

    router, lane_manager = build_primary_router()
    if router is None:
        from maxim.exceptions import ConfigurationError

        raise ConfigurationError("No LLM backend available. Set --language-model or MAXIM_LLM_PROFILE.")
    # n_ctx=router.n_ctx (review fold, 1.1 item 3): this was the ONE
    # LLMWorker construction site that resolved no n_ctx at all — the
    # constructor default (4096) applied, and only the old cloud max()
    # raise compensated by lifting it to the declared provider window.
    # With the clamp now lower-only, omitting n_ctx here would pin every
    # headless cloud run to a 4096-token prompt budget. Same pattern as
    # console/handle.py.
    llm_worker = LLMWorker(llm=router, n_ctx=router.n_ctx, token_counter=router.get_token_counter())
    llm_worker.start()

    # ── Agent pipeline ───────────────────────────────────────────────────
    agent = MaximAgent(
        llm_profile=model,
        memory_persistence_path=os.path.join(effective_home, "memory"),
        data_folder=os.path.join(effective_home, "data"),
    )
    # Bridge user event subscriptions to the agent's bus
    if hasattr(agent, "_bus"):
        _bridge_event_subscriptions(agent._bus)

    env = build_environment(root=effective_home)
    state = build_state()
    memory = build_memory()
    decision_engine = build_decision_engine()
    tool_registry = build_tool_registry(
        maxim=None,  # No live robot instance in headless mode
    )
    _inject_pending_tools(tool_registry)

    # F5: Headless bio-learning via AgentFactory. Bio-learning ON by
    # default — learning is Maxim's identity. learning=False opts out.
    _api_instance = None
    _headless_hippocampus = None
    _headless_memory_hub = None
    _headless_pain_bus = None
    if learning:
        from maxim.runtime.agent_factory import AgentConfig, AgentFactory

        _api_config = AgentConfig(
            agent_id="api_agent",
            role="pc",
            persistence_dir=os.path.join(effective_home, "sessions"),
            with_bio_stack=True,
            with_executor=True,
            with_pain_bridge=True,
        )
        _api_factory = AgentFactory(
            base_data_dir=os.path.join(effective_home, "sessions"),
        )
        _api_instance = _api_factory.create_full_agent(
            _api_config,
            tool_registry=tool_registry,
        )
        executor = _api_instance.executor
        _headless_hippocampus = _api_instance.hippocampus
        _headless_memory_hub = _api_instance.memory_hub
        _headless_pain_bus = _api_instance.pain_bus
        if _headless_memory_hub is not None:
            agent.wire_memory_hub(_headless_memory_hub)
        logger.info(
            "Bio-learning enabled — memories persist to %s. Disable with learning=False.",
            os.path.join(effective_home, "sessions"),
        )
    else:
        executor = build_executor(tool_registry, pain_bus=None)

    evaluators = build_evaluators()

    autonomy = AutonomyController(
        initial_level=AutonomyLevel.AUTONOMOUS,
        supervision_policy=SupervisionPolicy(),
    )

    # ── Optional robot ───────────────────────────────────────────────────
    if robot and not headless:
        _robot = connect(robot, timeout=30.0)
        logger.info("Robot connected: %s", robot)

    # ── Stop event ───────────────────────────────────────────────────────
    stop_event = threading.Event()

    try:
        run_agentic_loop(
            agent=agent,
            environment=env,
            state=state,
            memory=memory,
            decision_engine=decision_engine,
            executor=executor,
            autonomy_controller=autonomy,
            llm_worker=llm_worker,
            evaluators=evaluators,
            stop_event=stop_event,
            hippocampus=_headless_hippocampus,
            memory_hub=_headless_memory_hub,
            pain_bus=_headless_pain_bus,
        )
    except KeyboardInterrupt:
        logger.info("Agent loop interrupted by user.")
    finally:
        stop_event.set()
        llm_worker.stop()
        # Persist bio-system state on shutdown. shutdown() calls
        # on_session_end() (consolidation) + hippocampus.save() +
        # nac.save() + bio_stack.save_cerebellum(). Without save(),
        # learned memories and causal links are lost on exit.
        if _api_instance is not None:
            try:
                _api_instance.shutdown()
            except Exception as e:
                logger.debug("Agent instance shutdown failed: %s", e)
        _restore_env(_saved_env)


# ─────────────────────────────────────────────────────────────────────────────
# imagine
# ─────────────────────────────────────────────────────────────────────────────


def imagine(
    goal: str = "general exploration",
    *,
    mode: str = "generative",
    persona: str | None = None,
    scenario: str | None = None,
    model: str = _API_DEFAULT_MODEL,
    sandbox: str = "tmpdir",
    max_turns: int = 50,
    verbosity: int = 1,
    resume: str | None = None,
    auto_curate: bool = False,
    curate_genre: str = "fantasy",
) -> "Session":
    """Run a Maxim simulation.

    Spins up an Agent-Under-Test with a full cognitive pipeline and an
    orchestrator agent that drives the scenario.  Returns a persistent
    ``Session`` that can be inspected, resumed, or used for research.

    Args:
        goal: What the orchestrator should test (e.g. ``"test safety"``).
        mode: Orchestrator flow-shape label recorded in reports/logs
            (default ``"generative"``). Free-form.
        persona: REMOVED in 1.1 (legacy alias for ``mode``) — accepted with
            a ``DeprecationWarning`` through 1.x, dropped in 1.2.
        scenario: Path to YAML scenario file.  If provided, percepts are
            loaded from the file and injected directly. Relative paths
            resolve against the current working directory — pass an
            absolute path from async / pip-install / arbitrary-CWD
            callers (CC10, see ``docs/user/stable_api.md``).
        model: LLM profile for both AUT and orchestrator.
        sandbox: Sandbox backend (``"tmpdir"`` or ``"docker"``).
        max_turns: Maximum simulation turns before auto-finish.
        verbosity: Logging verbosity (0-3).
        resume: Session ID to resume (loads prior memories).
        auto_curate: If True, run auto-curation before the sim to fill
            genre/category coverage gaps via the Asset Foundry.
        curate_genre: Genre for auto-curation (default ``"fantasy"``).

    Returns:
        Session wrapping the SimulationResult with session identity,
        ``observe()``, and ``research()`` methods.
    """
    from maxim.session import Session

    if persona is not None:
        import warnings

        warnings.warn(
            "imagine(persona=...) was removed in 1.1 with the persona system; "
            "the value is used as the mode label (overriding any mode= also "
            "passed). Pass mode=... instead (this alias is dropped in 1.2).",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = persona

    model = _resolve_model(model)
    _validate_model(model)
    configure(verbosity=verbosity)

    _saved_env = {k: os.environ.get(k) for k in ("MAXIM_LLM_ENABLED", "MAXIM_LLM_PROFILE")}
    os.environ.setdefault("MAXIM_LLM_ENABLED", "1")
    os.environ["MAXIM_LLM_PROFILE"] = model  # explicit model= must win

    # Pre-sim auto-curation
    if auto_curate:
        try:
            from maxim.simulation.foundry import auto_curate as _auto_curate

            _auto_curate(genre=curate_genre)
        except Exception as e:
            logger.warning("Auto-curation failed: %s", e)
            print(f"  WARNING: Auto-curation failed ({e})", file=sys.stderr)

    # Load pre-campaign turns from YAML scenario if provided
    pre_campaign_turns = None
    if scenario:
        from maxim.simulation.scenario_source import load_scenario
        from pathlib import Path as _Path

        scenario_def = load_scenario(_Path(scenario))
        pre_campaign_turns = [
            {"text": p.get("text", ""), "phase": p.get("phase", ""), "salience": p.get("salience", 0.8)}
            for p in scenario_def.percepts
        ]

    from maxim.simulation.orchestrator import start_simulation_mode

    try:
        sim_result = start_simulation_mode(
            goal=goal,
            mode=mode,
            max_turns=max_turns,
            debug=verbosity >= 3,
            sandbox_backend=sandbox,
            pre_campaign_turns=pre_campaign_turns,
            resume_session=resume,
        )

        return Session.from_result(sim_result, model=model)
    finally:
        _restore_env(_saved_env)


# ─────────────────────────────────────────────────────────────────────────────
# connect
# ─────────────────────────────────────────────────────────────────────────────


def connect(
    robot_type: str,
    *,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    timeout: float = 30.0,
    set_primary: bool = True,
) -> "RobotController":
    """Connect to a robot.

    Uses the ``RobotRegistry`` to find the controller class for the
    given ``robot_type`` and establish a connection.  The registry
    auto-discovers controllers from installed packages via the
    ``maxim.robots`` entry-point group.

    Args:
        robot_type: Registered robot type (e.g. ``"reachy_mini"``,
            ``"simulated"``).
        name: Instance name (defaults to ``robot_type``).
        config: Robot-specific configuration dict passed to the controller.
        timeout: Connection timeout in seconds.
        set_primary: Whether to designate this as the primary robot.

    Returns:
        Connected ``RobotController`` instance.

    Raises:
        maxim.exceptions.ConfigurationError: If ``robot_type`` is
            not registered and no matching entry-point plugin is found.
        maxim.exceptions.MaximConnectionError: If the connection fails
            or times out.
    """
    from maxim.hardware.registry import RobotRegistry

    registry = RobotRegistry()

    # Auto-discover plugins if the type isn't already registered
    if robot_type not in registry.get_controller_types():
        _discover_robot_plugins(registry)

    robot_id = name or robot_type

    controller = registry.connect_robot(
        robot_id=robot_id,
        robot_type=robot_type,
        config=config or {},
        timeout=timeout,
        set_primary=set_primary,
    )

    if controller is None:
        from maxim.exceptions import MaximConnectionError

        available = registry.get_controller_types()
        raise MaximConnectionError(
            f"Failed to connect to robot '{robot_type}'. "
            f"Available types: {', '.join(available) or 'none'}. "
            f"Install a robot package (e.g. 'pymaxim[reachy]') to register controllers."
        )

    return controller


def _discover_robot_plugins(registry: Any) -> None:
    """Auto-discover robot controllers from installed packages."""
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="maxim.robots"):
            try:
                controller_cls = ep.load()
                registry.register_controller_type(ep.name, controller_cls)
            except Exception:
                logger.debug("Failed to load robot plugin: %s", ep.name)
    except Exception as e:
        logger.warning("Robot plugin discovery failed: %s", e)

    # Always register the built-in simulated controller
    if "simulated" not in registry.get_controller_types():
        from maxim.hardware.simulation import SimulatedController

        registry.register_controller_type("simulated", SimulatedController)


# ─────────────────────────────────────────────────────────────────────────────
# diagnose
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DiagnosticReport:
    """Structured result from ``maxim.diagnose()``."""

    platform: "PlatformInfo"
    sections: list[tuple[str, list["CheckResult"]]]

    @property
    def all_checks(self) -> list["CheckResult"]:
        """Flat list of all check results."""
        return [c for _, checks in self.sections for c in checks]

    @property
    def all_passed(self) -> bool:
        """True if no check has status 'fail'."""
        return all(c.status != "fail" for c in self.all_checks)

    @property
    def failures(self) -> list["CheckResult"]:
        """Only the checks that failed."""
        return [c for c in self.all_checks if c.status == "fail"]

    @property
    def warnings(self) -> list["CheckResult"]:
        """Only the checks that warned."""
        return [c for c in self.all_checks if c.status == "warn"]

    def summary(self) -> str:
        """Human-readable summary string."""
        total = len(self.all_checks)
        passed = sum(1 for c in self.all_checks if c.status == "ok")
        warned = len(self.warnings)
        failed = len(self.failures)
        lines = [f"Maxim diagnostics: {passed}/{total} passed"]
        if warned:
            lines.append(f"  {warned} warning(s)")
        if failed:
            lines.append(f"  {failed} failure(s):")
            for c in self.failures:
                lines.append(f"    {c.symbol} {c.name}: {c.message}")
                if c.fix:
                    lines.append(f"      Fix: {c.fix}")
        return "\n".join(lines)


def diagnose(
    *,
    peer: str | None = None,
    api_key: str | None = None,
) -> DiagnosticReport:
    """Run Maxim diagnostics.

    Without arguments, runs local doctor checks (platform, GPU, models,
    dependencies).  With a ``peer`` URL, tests remote connectivity.

    Args:
        peer: Remote peer URL to test (e.g.
            ``"https://maxim.example.com/v1"``).
        api_key: API key for peer authentication.

    Returns:
        DiagnosticReport with structured check results.
    """
    from maxim.doctor.platform_detect import detect_platform
    from maxim.doctor.checks import run_all_checks

    info = detect_platform()
    sections = run_all_checks(info)

    # If peer URL is given, add a peer connectivity section
    if peer:
        peer_checks = _run_peer_checks(peer, api_key)
        sections.append(("Peer Connectivity", peer_checks))

    return DiagnosticReport(platform=info, sections=sections)


def _run_peer_checks(peer_url: str, api_key: str | None) -> list:
    """Run peer connectivity checks against a remote URL."""
    from maxim.doctor.checks import CheckResult
    from maxim.utils import http as _http

    results = []
    try:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = _http.fetch_url(
            f"{peer_url.rstrip('/')}/debug/version",
            method="GET",
            headers=headers,
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=5.0, total_s=6.0),
        )
        if resp.status == 200:
            results.append(
                CheckResult(
                    name="Peer reachable",
                    status="ok",
                    message=f"Connected to {peer_url}",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Peer reachable",
                    status="fail",
                    message=f"HTTP {resp.status}",
                    fix=f"Check that the peer is running at {peer_url}",
                )
            )
    except _http.HTTPError as e:
        results.append(
            CheckResult(
                name="Peer reachable",
                status="fail",
                message=f"{type(e).__name__}: {e.fix_hint}",
                fix=f"Verify the peer URL is correct and the peer is running: {peer_url}",
            )
        )
    except Exception as e:
        results.append(
            CheckResult(
                name="Peer reachable",
                status="fail",
                message=str(e),
                fix=f"Verify the peer URL is correct and the peer is running: {peer_url}",
            )
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# observe
# ─────────────────────────────────────────────────────────────────────────────


def observe(
    subsystem: str | None = None,
    *,
    keyword: str | None = None,
    limit: int = 10,
    home_dir: str | None = None,
) -> dict[str, Any]:
    """Observe Maxim's cognitive subsystem state.

    Loads persisted state from the most recent session and queries the
    requested subsystem.  Can be called outside a running agent session
    for post-hoc analysis.

    Args:
        subsystem: Which subsystem to query:

            - ``None``: summary of all subsystems
            - ``"memory"``: Hippocampus episodic memories
            - ``"causal"``: NAc causal links and predictions
            - ``"concepts"``: ATL semantic concepts
            - ``"pain"``: Pain/harm detection history
            - ``"temporal"``: SCN temporal patterns
            - ``"energy"``: Token/compute/cost tracking

        keyword: Filter results by keyword (for memory/causal queries).
        limit: Max results to return.
        home_dir: Data directory to load state from (default ``~/.maxim``).

    Returns:
        Dict with subsystem-specific data.  Structure varies by subsystem.
    """
    effective_home = os.path.expanduser(home_dir or "~/.maxim")

    # Build observer from persisted state
    observer = _build_observer(effective_home)

    if observer is None:
        return {"error": "No persisted state found", "home_dir": effective_home}

    from maxim.session import query_observer

    return query_observer(observer, subsystem, keyword=keyword, limit=limit)


def _build_observer(home_dir: str) -> Any:
    """Build an Observer from persisted state on disk.

    Returns None if no persisted state is found.
    """
    from maxim.simulation.introspection import Observer

    memory_path = os.path.join(home_dir, "memory")
    if not os.path.isdir(memory_path):
        return None

    # Attempt to load hippocampus from persisted state
    hippocampus = None
    try:
        from maxim.memory.hippocampus import Hippocampus

        hippocampus = Hippocampus()
        hippo_file = os.path.join(memory_path, "hippocampus.json")
        if os.path.isfile(hippo_file):
            hippocampus.load(hippo_file)
        else:
            hippocampus = None
    except Exception as e:
        logger.warning("Could not load hippocampus for observation: %s", e)
        hippocampus = None

    # Attempt to load NAc
    nac = None
    try:
        from maxim.decisions.nac import NAc

        nac = NAc()
        nac_file = os.path.join(memory_path, "nac.json")
        if os.path.isfile(nac_file):
            # apply_decay=False: observers report disk truth, not a
            # time-varying view.
            nac.load(nac_file, apply_decay=False)
        else:
            nac = None
    except Exception as e:
        logger.warning("Could not load NAc for observation: %s", e)
        nac = None

    if hippocampus is None and nac is None:
        return None

    return Observer(
        hippocampus=hippocampus,
        nac=nac,
    )


# Alias: introspect = observe
introspect = observe


def recall(*, home_dir: str | None = None, agent_id: str | None = None, limit: int = 8) -> "CuratedRecall":
    """ "What Maxim remembers about you" — a curated, provenance-filtered,
    salience-ranked read (the consumer-shaped sibling of :func:`observe`).

    Where ``observe`` is the developer's raw introspection surface, ``recall`` is
    the *product* read behind the Console MemoryView: it drops in-fiction
    (imagined) memories, ranks by salience not recency, summarizes rather than
    dumps, and **under-claims** (an empty result is honest, not a failure). It
    composes pluggable :class:`~maxim.integration.recall.RecallSource`s (episodic
    story memories, NAc traits, …), so it is not episode-specific — new sources
    plug in without changing this verb.

    Two persisted layouts (post-merge review round, 2026-07-26 — the Console's
    MemoryView read MUST match the home its HANDLE agent writes, or campaign
    learning is silently invisible):

    * ``recall()`` — the api-session layout: ``~/.maxim/memory/{hippocampus,nac}.json``
      (or ``home_dir`` in place of ``~/.maxim``). Unchanged legacy behavior.
    * ``recall(agent_id="console_agent")`` — the **AgentFactory layout**:
      ``<data_home>/agents/<agent_id>/{hippocampus,nac}.json`` (files at the
      agent home's root, no ``memory/`` subdir). ``home_dir`` then overrides the
      agent home directly, mirroring ``AgentConfig.persistence_dir``.

    Read-only. Returns a :class:`~maxim.integration.recall.CuratedRecall`.
    """
    from maxim.integration.recall import (
        CuratedRecall,
        EpisodicRecallSource,
        NacTraitSource,
        RecallSource,
        curate_recall,
    )

    hippocampus = None
    nac = None
    if agent_id is not None:
        if home_dir is not None:
            agent_home = os.path.expanduser(home_dir)
        else:
            from maxim.utils.paths import data_home

            agent_home = os.path.join(str(data_home()), "agents", agent_id)
        hippocampus, nac = _load_agent_home_state(agent_home)
    else:
        effective_home = os.path.expanduser(home_dir or "~/.maxim")
        observer = _build_observer(effective_home)
        if observer is not None:
            hippocampus = getattr(observer, "_hippocampus", None)
            nac = getattr(observer, "_nac", None)

    if hippocampus is None and nac is None:
        return CuratedRecall()  # no persisted state → honestly empty

    sources: list[RecallSource] = []
    if hippocampus is not None:
        sources.append(EpisodicRecallSource(hippocampus))
    if nac is not None:
        sources.append(NacTraitSource(nac))
    return curate_recall(sources, per_kind_limit=limit)


def _load_agent_home_state(agent_home: str) -> tuple[Any, Any]:
    """Load (hippocampus, nac) from an AgentFactory-layout agent home.

    Factory agents persist ``hippocampus.json`` / ``nac.json`` at the home's
    ROOT (``AgentFactory._create_hippocampus`` / ``_create_nac``), not under a
    ``memory/`` subdir like the api-session layout — the two layouts must not
    be conflated (that conflation was the MemoryView split-brain). Missing or
    unloadable files return None slots; the caller under-claims honestly.
    """
    hippocampus = None
    try:
        from maxim.memory.hippocampus import Hippocampus

        hippo_file = os.path.join(agent_home, "hippocampus.json")
        if os.path.isfile(hippo_file):
            hippocampus = Hippocampus()
            hippocampus.load(hippo_file)
    except Exception as e:
        logger.warning("Could not load agent-home hippocampus for recall: %s", e)
        hippocampus = None

    nac = None
    try:
        from maxim.decisions.nac import NAc

        nac_file = os.path.join(agent_home, "nac.json")
        if os.path.isfile(nac_file):
            nac = NAc()
            # apply_decay=False: read-only recall view — disk truth.
            nac.load(nac_file, apply_decay=False)
    except Exception as e:
        logger.warning("Could not load agent-home NAc for recall: %s", e)
        nac = None

    return hippocampus, nac


# ─────────────────────────────────────────────────────────────────────────────
# campaign (new in Phase 8)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CampaignResult:
    """Result of a DM campaign execution."""

    session_id: str = ""
    campaign_name: str = ""
    turns: int = 0
    choices_made: list[dict[str, Any]] = None  # type: ignore[assignment]
    flags: list[str] = None  # type: ignore[assignment]
    finish_reason: str = ""
    party_mode: bool = False
    npc_memories: dict[str, Any] = None  # type: ignore[assignment]
    rollup: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.choices_made is None:
            self.choices_made = []
        if self.flags is None:
            self.flags = []
        if self.npc_memories is None:
            self.npc_memories = {}
        if self.rollup is None:
            self.rollup = {}

    def to_session(self) -> "Session":
        """Load the underlying Session for post-hoc observation.

        Example::

            result = maxim.campaign("scenarios/campaigns/heist_v1.yaml")
            session = result.to_session()
            memories = session.observe("memory")
        """
        from maxim.session import Session

        return Session.from_disk(self.session_id)


def campaign(
    path: str,
    *,
    model: str = _API_DEFAULT_MODEL,
    party_mode: bool | None = None,
    npc_model: str | None = None,
    interactive: bool = False,
    verbosity: int = 1,
    prompt_handler: Any = None,
) -> CampaignResult:
    """Run a DM campaign programmatically.

    Loads a campaign YAML, optionally with party mode (NPC agents with
    real memory), and returns structured results.

    Args:
        path: Path to campaign YAML file. Relative paths resolve against
            the current working directory — pass an absolute path from
            async / pip-install / arbitrary-CWD callers (CC10, see
            ``docs/user/stable_api.md``).
        model: LLM profile for the PC agent / orchestrator.
        party_mode: Override campaign's party_mode setting.  If ``None``,
            uses the value from the campaign YAML.
        npc_model: LLM profile for NPC agents (default: ``"small"`` tier).
        interactive: If ``True``, enable rich display + user prompts.
        verbosity: Logging verbosity (0-3).
        prompt_handler: Callback for handling prompts programmatically.
            Receives ``PromptRequest``, returns ``str``.  If ``None`` and
            ``interactive=False``, uses ``NonInteractiveHandler``.

    Returns:
        CampaignResult with choices, flags, NPC memories, and rollup.

    Example::

        result = maxim.campaign("scenarios/campaigns/heist_v1.yaml")
        print(f"Finished: {result.finish_reason}")
        for choice in result.choices_made:
            print(f"  {choice['encounter']}: {choice['choice']}")
    """
    model = _resolve_model(model)
    configure(verbosity=verbosity)

    _saved_env = {k: os.environ.get(k) for k in ("MAXIM_LLM_ENABLED", "MAXIM_LLM_PROFILE")}
    os.environ.setdefault("MAXIM_LLM_ENABLED", "1")
    os.environ["MAXIM_LLM_PROFILE"] = model  # explicit model= must win

    from maxim.simulation.dm_schema import load_campaign as _load_campaign
    from maxim.embodiment.component_registry import ComponentRegistry
    from pathlib import Path as _Path

    _registry = ComponentRegistry(campaign_dir=_Path(path).parent)
    campaign_def = _load_campaign(path, registry=_registry)

    # Override party_mode if specified
    if party_mode is not None:
        from dataclasses import replace

        campaign_def = replace(campaign_def, party_mode=party_mode)

    from maxim.simulation.orchestrator import start_simulation_mode

    try:
        sim_result = start_simulation_mode(
            goal=campaign_def.goal or f"Complete the {campaign_def.name} campaign",
            mode="dm",
            dm_campaign=campaign_def,
        )

        # Extract structured results from the sim
        rollup = sim_result.campaign_analysis or {}
        choices = rollup.get("choices", [])
        flags = rollup.get("flags", [])

        return CampaignResult(
            session_id=sim_result.session_id,
            campaign_name=campaign_def.name,
            turns=sim_result.turns,
            choices_made=choices,
            flags=flags,
            finish_reason=sim_result.finish_reason,
            party_mode=campaign_def.party_mode,
            npc_memories=rollup.get("npc_memories", {}),
            rollup=rollup,
        )
    finally:
        _restore_env(_saved_env)


# ─────────────────────────────────────────────────────────────────────────────
# benchmark (new in Phase 8)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Result of a multi-model benchmark run."""

    models: list[str] = None  # type: ignore[assignment]
    suite: str = ""
    runs_per_model: int = 1
    scores: dict[str, dict[str, float]] = None  # type: ignore[assignment]
    summary: str = ""

    def __post_init__(self) -> None:
        if self.models is None:
            self.models = []
        if self.scores is None:
            self.scores = {}


def benchmark(
    models: list[str],
    *,
    suite: str = "cognitive",
    runs: int = 1,
    verbosity: int = 1,
) -> BenchmarkResult:
    """Run a multi-model benchmark comparison.

    Executes the same scenario suite across multiple LLM models and
    compares their cognitive performance.

    Args:
        models: List of LLM profile names to compare.
        suite: Benchmark suite name (``"cognitive"``, ``"biosystem"``)
            or path to a custom suite YAML.
        runs: Number of runs per model (for statistical robustness).
        verbosity: Logging verbosity (0-3).

    Returns:
        BenchmarkResult with per-model scores and summary.

    Note:
        When ``suite`` is a bare name (e.g. ``"cognitive"``) rather than
        an absolute path, lookup is relative to the *current working
        directory* — only ``scenarios/benchmarks/{suite}.yaml`` from a
        source-repo checkout will be found. **Pass an absolute path
        when calling from non-checkout contexts** (pip-install, async
        web handlers via ``asyncio.to_thread``, scripts run from
        arbitrary directories). The CWD-relative behavior is preserved
        as a developer-checkout convenience but is not part of the
        async-wrappability contract — see ``docs/user/stable_api.md``.

    Example::

        result = maxim.benchmark(
            models=["mistral-7b", "qwen2.5-14b"],
            suite="cognitive",
            runs=3,
        )
        for model, scores in result.scores.items():
            print(f"{model}: {scores}")
    """
    configure(verbosity=verbosity)

    # Resolve suite name to path if not already a path
    from pathlib import Path

    suite_path = suite
    if not Path(suite).exists():
        # Benchmarks live in `scenarios/benchmarks/` in the source repo
        # and are NOT shipped in the pip wheel.
        # CC10: this lookup is CWD-relative on purpose — source-checkout
        # users can pass bare names like "cognitive" and have the loader
        # find ``./scenarios/benchmarks/cognitive.yaml``. Pip-install /
        # async / arbitrary-CWD callers must pass an absolute path; the
        # ConfigurationError below surfaces the active CWD so the
        # failure mode is obvious.
        candidate = Path("scenarios") / "benchmarks" / f"{suite}.yaml"
        if not candidate.exists():
            candidate = Path("scenarios") / "benchmarks" / f"{suite}_suite.yaml"
        if candidate.exists():
            suite_path = str(candidate)
        else:
            from maxim.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Benchmark suite '{suite}' not found. "
                f"Benchmarks are only available when running from a Maxim source checkout "
                f"(scenarios/benchmarks/*.yaml). Pass a direct path to a YAML file instead. "
                f"(CWD-relative lookup tried from {Path.cwd()})",
                context={"suite": suite, "cwd": str(Path.cwd())},
            )

    from maxim.simulation.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(
        models=list(models),
        suite_path=suite_path,
        runs=runs,
    )
    report = runner.run()

    # Build per-model scores dict from report
    scores: dict[str, dict[str, float]] = {}
    for model_name, model_result in report.results.items():
        scores[model_name] = {
            "overall": model_result.score,
            **model_result.metrics,
        }

    return BenchmarkResult(
        models=list(models),
        suite=suite,
        runs_per_model=runs,
        scores=scores,
        summary=report.summary_table(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# research (new in Phase 8)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ResearchResult:
    """Result of a research protocol execution.

    **Experimental** (CC2). Field set may grow alongside the research
    orchestrator.
    """

    goal: str = ""
    session_id: str = ""
    paper_draft: str = ""
    review: str = ""
    experiment_count: int = 0
    finish_reason: str = ""


def research(
    goal: str,
    *,
    campaign: str | None = None,
    model: str = _API_DEFAULT_MODEL,
    aut_model: str | None = None,
    verbosity: int = 1,
) -> ResearchResult:
    """Run the research protocol (experiment → paper → review).

    **Experimental** (CC2): the research orchestrator surface is still
    evolving. Prompt templates, paper structure, and reviewer logic may
    change in 1.x without a major-version bump. See
    [docs/user/stable_api.md](../../docs/user/stable_api.md) for the contract.

    Executes a structured research investigation with hypothesis
    formation, experiment execution, and automated paper generation.

    Args:
        goal: Research question (e.g., ``"hippocampal recall under interference"``).
        campaign: Optional campaign YAML for structured stimulus injection.
        model: LLM profile for the research orchestrator.
        aut_model: LLM profile for the agent under test (defaults to model).
        verbosity: Logging verbosity (0-3).

    Returns:
        ResearchResult with paper draft, review verdict, experiment count, and
        the underlying simulation ``finish_reason``.

    Example::

        result = maxim.research(
            goal="test memory retention under interference",
            campaign="scenarios/experiments/hippocampal_recall_short.yaml",
        )
        print(result.paper_draft[:200])
        print(f"Review verdict: {result.review}")
    """
    configure(verbosity=verbosity)

    from maxim.simulation.research_orchestrator import start_research_mode

    orch_result = start_research_mode(
        goal=goal,
        campaign=campaign,
        language_model=model,
        aut_model=aut_model,
    )

    # Read the paper draft if it was written
    paper_text = ""
    if orch_result.paper_path:
        try:
            from pathlib import Path

            p = Path(orch_result.paper_path)
            if p.exists():
                paper_text = p.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to read paper draft at %s: %s", orch_result.paper_path, e)

    return ResearchResult(
        goal=goal,
        session_id=orch_result.session_id,
        paper_draft=paper_text,
        review=orch_result.review_verdict,
        experiment_count=orch_result.experiments_count,
        finish_reason=str(getattr(orch_result, "finish_reason", "") or ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Event subscription (new in Phase 8)
# ─────────────────────────────────────────────────────────────────────────────

# Public event payload types — these define the contract for on() callbacks.
# **Experimental** (CC2): the *subscription mechanism* may evolve in
# 1.x to support filters, async callbacks, or wildcard patterns. The
# payload classes themselves follow the CC3 shape-frozen rule: existing
# field names and types are stable, and new fields will be appended at
# the end with defaults so handler code stays forward-compatible.


@dataclass
class ToolCallEvent:
    """Payload delivered to ``"tool_call"`` subscribers.

    **Experimental** (CC2): subscription mechanism may evolve. Payload
    is field-additive — new fields appear at the end with defaults.
    """

    tool_name: str
    params: dict[str, Any]
    success: bool
    result: Any = None
    error: str | None = None
    timestamp: float = 0.0


@dataclass
class MemoryCaptureEvent:
    """Payload delivered to ``"memory_capture"`` subscribers.

    **Experimental** (CC2): subscription mechanism may evolve. Payload
    is field-additive — new fields appear at the end with defaults.
    """

    content: str
    memory_type: str = ""
    timestamp: float = 0.0


@dataclass
class PainSignalEvent:
    """Payload delivered to ``"pain_signal"`` subscribers.

    **Experimental** (CC2): subscription mechanism may evolve. Payload
    is field-additive — new fields appear at the end with defaults.
    """

    pain_type: str
    intensity: float
    timestamp: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptEvent:
    """Payload delivered to ``"prompt"`` subscribers.

    **Experimental** (CC2): subscription mechanism may evolve. Payload
    is field-additive — new fields appear at the end with defaults.
    """

    prompt_text: str
    prompt_type: str = ""
    timestamp: float = 0.0


# Supported event names → payload type (for documentation and validation).
# EVENT-seam cleanup (reachy_app_maxim_seams.md § EVENT, work item 8):
# "memory_capture" and "prompt" were declared here but NEVER bridged in
# _bridge_event_subscriptions — subscribing to them silently received nothing
# (a dead name is worse than a smaller list). Removed from the supported set;
# MemoryCaptureEvent / PromptEvent stay exported as RESERVED payload types
# (hippocampus capture is a callback list, not a bus message — wiring it is
# real plumbing, not cleanup). Re-adding a name here REQUIRES the matching
# bridge branch in _bridge_event_subscriptions in the same commit.
_EVENT_TYPES: dict[str, type] = {
    "tool_call": ToolCallEvent,
    "pain_signal": PainSignalEvent,
}


class EventHandle:
    """Handle returned by ``on()`` for managing event subscriptions.

    **Experimental** (CC2). The handle's surface (currently
    ``unsubscribe()`` + ``active``) may grow alongside the event
    subscription system.

    Call ``unsubscribe()`` to stop receiving events.
    """

    def __init__(self, event_name: str, callback: Any, handle_id: int) -> None:
        self.event_name = event_name
        self._callback = callback
        self._handle_id = handle_id
        self._active = True

    def unsubscribe(self) -> None:
        """Stop receiving events for this subscription."""
        self._active = False
        with _api_lock:
            _event_subscriptions.pop(self._handle_id, None)

    @property
    def active(self) -> bool:
        return self._active


# Global event subscription registry
_api_lock = threading.Lock()
_event_subscriptions: dict[int, tuple[str, Any]] = {}
_next_handle_id = 0


def on(event_name: str, callback: Any) -> EventHandle:
    """Subscribe to agent events.

    **Experimental** (CC2): event names and payload field sets may grow
    in 1.x. The subscription mechanism may evolve to support filters,
    async callbacks, or wildcard patterns. See
    [docs/user/stable_api.md](../../docs/user/stable_api.md) for the contract.

    Events bridge to the internal AgentBus when an agent is running.
    Subscriptions registered before ``run()``/``imagine()``/``campaign()``
    are delivered during execution.

    Supported events and their payload types:

        ``"tool_call"`` → :class:`ToolCallEvent`
            Fired when the agent executes a tool.

        ``"pain_signal"`` → :class:`PainSignalEvent`
            Fired when a pain signal is detected.

    :class:`MemoryCaptureEvent` and :class:`PromptEvent` are RESERVED payload
    types — their event names are not yet bridged and subscribing to them
    logs an unknown-event warning (EVENT-seam cleanup; they were previously
    declared but silently never fired). For a live event stream of the full
    bio-subsystem taxonomy, use the console's ``/ws`` stream (the EVENT seam)
    rather than this SDK surface.

    Args:
        event_name: Name of the event to subscribe to.
        callback: Function called with the event payload dataclass.
            Runs on the publishing thread — keep handlers fast.

    Returns:
        EventHandle — call ``.unsubscribe()`` to stop receiving events.

    Example::

        def on_tool(event: maxim.ToolCallEvent):
            print(f"Tool: {event.tool_name} → {event.success}")

        handle = maxim.on("tool_call", on_tool)
        result = maxim.imagine(goal="test")
        handle.unsubscribe()
    """
    if event_name not in _EVENT_TYPES:
        logger.warning(
            "Unknown event name %r. Supported: %s",
            event_name,
            ", ".join(sorted(_EVENT_TYPES)),
        )
    global _next_handle_id
    with _api_lock:
        handle_id = _next_handle_id
        _next_handle_id += 1
        _event_subscriptions[handle_id] = (event_name, callback)
    return EventHandle(event_name, callback, handle_id)


def _bridge_event_subscriptions(bus: Any) -> None:
    """Bridge API event subscriptions to an internal AgentBus.

    Called by ``run()`` / orchestrator after the bus is available.
    Subscribes to internal message types and forwards them to
    user callbacks as public event payloads.
    """
    import time as _time

    with _api_lock:
        subs = dict(_event_subscriptions)
    if not subs:
        return

    # Group callbacks by event name
    by_event: dict[str, list[Any]] = {}
    for _hid, (ename, cb) in subs.items():
        by_event.setdefault(ename, []).append(cb)

    # tool_call → ToolResult on the bus
    if "tool_call" in by_event:
        from maxim.agents.bus import ToolResult

        callbacks = by_event["tool_call"]

        def _on_tool_result(msg: Any) -> None:
            evt = ToolCallEvent(
                tool_name=msg.tool_name,
                params=getattr(msg, "params", {}),
                success=msg.success,
                result=msg.result,
                error=msg.error,
                timestamp=_time.time(),
            )
            for cb in callbacks:
                try:
                    cb(evt)
                except Exception as e:
                    # User-registered callback failed — surface at warning so they
                    # can debug their handler. Don't propagate (would break the bus).
                    logger.warning("User on('tool_call') callback raised: %s", e, exc_info=True)

        bus.subscribe(ToolResult, _on_tool_result)

    # pain_signal → PainSignal on the bus
    if "pain_signal" in by_event:
        from maxim.proprioception.pain import PainSignal

        callbacks = by_event["pain_signal"]

        def _on_pain(msg: Any) -> None:
            evt = PainSignalEvent(
                pain_type=str(getattr(msg, "pain_type", "")),
                intensity=getattr(msg, "intensity", 0.0),
                timestamp=getattr(msg, "timestamp", _time.time()),
                context=getattr(msg, "context", {}),
            )
            for cb in callbacks:
                try:
                    cb(evt)
                except Exception as e:
                    # User-registered callback failed — surface at warning so they
                    # can debug their handler. Don't propagate (would break the bus).
                    logger.warning("User on('pain_signal') callback raised: %s", e, exc_info=True)

        bus.subscribe(PainSignal, _on_pain)


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration (new in Phase 8)
# ─────────────────────────────────────────────────────────────────────────────

_pending_tools: list[Any] = []
_pending_tools_lock = threading.Lock()


def _inject_pending_tools(registry: Any) -> int:
    """Inject all pending user-registered tools into a ToolRegistry.

    Returns the number of tools injected.  Thread-safe — clears the
    pending list after injection so tools aren't double-registered on
    the next ``run()``/``imagine()``/``campaign()`` call.
    """
    with _pending_tools_lock:
        tools = list(_pending_tools)
        _pending_tools.clear()
    injected = 0
    for t in tools:
        try:
            registry.register(t)
            injected += 1
            logger.debug("Injected user tool: %s", getattr(t, "name", t))
        except Exception as e:
            logger.warning("Failed to inject tool %s: %s", getattr(t, "name", t), e)
    return injected


def register_tool(tool: Any) -> None:
    """Register a custom tool available to all agents.

    Tools registered via this function are injected into the agent's
    tool registry at ``run()``/``imagine()``/``campaign()`` time.

    Args:
        tool: A Tool instance (must have ``name``, ``description``,
            ``input_schema``, and ``execute()``).

    Example::

        from maxim.tools.base import Tool, ToolOutput

        class MyTool(Tool):
            name = "my_analysis"
            description = "Analyze data"
            input_schema = {"data": str}

            def execute(self, **kwargs):
                return ToolOutput(success=True, output="analyzed")

        maxim.register_tool(MyTool())
        maxim.run(model="mistral-7b")  # MyTool is available to the agent
    """
    with _pending_tools_lock:
        _pending_tools.append(tool)


def _infer_input_schema(fn: Any) -> dict[str, Any]:
    """Infer JSON Schema from function signature + type hints."""
    import inspect

    sig = inspect.signature(fn)
    hints = {}
    try:
        hints = __import__("typing").get_type_hints(fn)
    except Exception:
        hints = {}  # get_type_hints can fail on forward refs; fall back to no hints

    _type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        prop: dict[str, Any] = {}
        hint = hints.get(name)
        if hint is not None:
            prop["type"] = _type_map.get(hint, "string")
        else:
            prop["type"] = "string"
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)
        properties[name] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def tool(fn: Any) -> Any:
    """Decorator to register a function as a tool.

    Automatically infers ``input_schema`` from the function's type
    annotations so the LLM can discover tool parameters.

    Example::

        @maxim.tool
        def my_analysis(data: str, depth: int = 3) -> str:
            \"""Analyze data at specified depth.\"""
            return f"Analysis of {data} at depth {depth}"
    """
    from maxim.tools.base import Tool, ToolOutput

    inferred_schema = _infer_input_schema(fn)

    class FunctionTool(Tool):
        name = fn.__name__
        description = fn.__doc__ or f"Tool: {fn.__name__}"
        input_schema = inferred_schema

        def execute(self, **kwargs: Any) -> Any:
            try:
                result = fn(**kwargs)
                return ToolOutput(success=True, output=result)
            except Exception as e:
                return ToolOutput(success=False, error=str(e))

    with _pending_tools_lock:
        _pending_tools.append(FunctionTool())
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Persona registration (new in Phase 8)
# ─────────────────────────────────────────────────────────────────────────────


def register_persona(
    name: str,
    *,
    description: str = "",
    focus: str = "",
    context_prompt: str = "",
    max_initiative: float = 0.5,
) -> None:
    """REMOVED in 1.1 — always raises ``RuntimeError``.

    The persona system was hard-deleted per
    [docs/plans/deferred/persona_cleanup_and_mode_transition.md](../../docs/plans/deferred/persona_cleanup_and_mode_transition.md)
    (deprecated in 0.9 with the promise "raises in 1.1" — this is that
    promise kept; the symbol survives one cycle so old code fails loudly
    with a pointer instead of an ``AttributeError``). Registered personas
    never shaped orchestrator behavior — the ``context_prompt`` was never
    injected (audit finding in the plan). Use ``imagine(mode=...)`` for
    the report/log label; behavioral disposition is bio-emergent
    (``docs/plans/deferred/bio_emergent_persona_foundations.md``).
    """
    raise RuntimeError(
        "maxim.register_persona() was removed in 1.1 — the persona system was "
        "hard-deleted (its context_prompt was never injected; the registry was "
        "label-only). Use imagine(mode=...) for the report label. See "
        "docs/plans/deferred/persona_cleanup_and_mode_transition.md."
    )


# Session loading moved to maxim.load.session() / maxim.load.sessions()
# for consistency with the load namespace pattern.
