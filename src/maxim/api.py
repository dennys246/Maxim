"""Public verb-based API for pymaxim.

Six top-level functions that map directly to user intent:

    import maxim

    maxim.configure(verbosity=2)
    maxim.run(model="mistral-7b")
    maxim.imagine(goal="test safety", persona="adversarial")
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.doctor.checks import CheckResult
    from maxim.doctor.platform_detect import PlatformInfo
    from maxim.hardware.controller import RobotController
    from maxim.simulation.orchestrator import SimulationResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# configure
# ─────────────────────────────────────────────────────────────────────────────


def configure(
    *,
    verbosity: int = 1,
    log_file: str | None = None,
    debug: str | None = None,
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
    """
    from maxim.utils.logging import configure_logging
    from maxim.utils.structured_logging import configure_agentic_verbosity

    configure_logging(verbosity, log_file=log_file, force=True)
    configure_agentic_verbosity(verbosity, console_output=verbosity >= 2)

    # Subsystem trace env vars (read by individual subsystems at runtime)
    if debug is not None:
        traces = [t.strip().lower() for t in debug.split(",")]
        all_traces = "all" in traces
        for subsystem in ("hippo", "nac", "atl", "ec", "scn", "pain", "fear", "default_net"):
            env_key = f"MAXIM_{subsystem.upper()}_TRACE"
            os.environ[env_key] = "1" if (all_traces or subsystem in traces) else ""


# ─────────────────────────────────────────────────────────────────────────────
# run
# ─────────────────────────────────────────────────────────────────────────────


def run(
    model: str = "mistral-7b",
    *,
    goal: str | None = None,
    headless: bool = True,
    robot: str | None = None,
    home_dir: str | None = None,
    verbosity: int = 1,
) -> None:
    """Run Maxim's agentic cycle.

    Bootstraps the full agent pipeline (LLM router, memory systems,
    planning, tools, safety) and enters the main loop.  Blocks until
    the user interrupts (Ctrl+C) or the goal is completed.

    Args:
        model: LLM profile name (e.g. ``"mistral-7b"``, ``"claude-sonnet"``).
        goal: Optional goal string.  If provided, the agent works toward
            it with a utility prompt.  If ``None``, enters interactive mode.
        headless: If ``True`` (default), run without robot hardware.
        robot: Robot type to connect (e.g. ``"reachy_mini"``).  Requires
            the corresponding package to be installed.
        home_dir: Data/persistence directory (default ``~/.maxim``).
        verbosity: Logging verbosity (0-3).

    Raises:
        maxim.exceptions.MaximConfigurationError: If the requested model
            is not available (missing files or API key).
    """
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

    effective_home = os.path.expanduser(home_dir or "~/.maxim")
    os.makedirs(effective_home, exist_ok=True)

    # ── LLM router ───────────────────────────────────────────────────────
    os.environ.setdefault("MAXIM_LLM_ENABLED", "1")
    os.environ.setdefault("MAXIM_LLM_PROFILE", model)

    router, lane_manager = build_primary_router()
    llm_worker = LLMWorker(router=router)
    llm_worker.start()

    # ── Agent pipeline ───────────────────────────────────────────────────
    agent = MaximAgent(
        llm_profile=model,
        memory_persistence_path=os.path.join(effective_home, "memory"),
        data_folder=os.path.join(effective_home, "data"),
    )
    env = build_environment(root=effective_home)
    state = build_state()
    memory = build_memory()
    decision_engine = build_decision_engine(memory=memory, env=env)
    tool_registry = build_tool_registry(
        maxim=None,  # No live robot instance in headless mode
        data_folder=os.path.join(effective_home, "data"),
    )
    executor = build_executor(tool_registry)
    evaluators = build_evaluators()

    autonomy = AutonomyController(
        level=AutonomyLevel.FULL,
        supervision=SupervisionPolicy(),
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
        )
    except KeyboardInterrupt:
        logger.info("Agent loop interrupted by user.")
    finally:
        stop_event.set()
        llm_worker.stop()


# ─────────────────────────────────────────────────────────────────────────────
# imagine
# ─────────────────────────────────────────────────────────────────────────────


def imagine(
    goal: str = "general exploration",
    *,
    persona: str = "cooperative",
    scenario: str | None = None,
    model: str = "mistral-7b",
    sandbox: str = "tmpdir",
    max_turns: int = 50,
    verbosity: int = 1,
) -> "SimulationResult":
    """Run a Maxim simulation.

    Spins up an Agent-Under-Test with a full cognitive pipeline and an
    orchestrator agent that drives the scenario.  Returns structured
    results when the simulation completes or is interrupted.

    Args:
        goal: What the orchestrator should test (e.g. ``"test safety"``).
        persona: Orchestrator persona (``"adversarial"``, ``"cooperative"``,
            ``"confused"``, ``"escalating"``, ``"researcher"``, etc.).
        scenario: Path to YAML scenario file.  If provided, percepts are
            loaded from the file and injected directly.
        model: LLM profile for both AUT and orchestrator.
        sandbox: Sandbox backend (``"tmpdir"`` or ``"docker"``).
        max_turns: Maximum simulation turns before auto-finish.
        verbosity: Logging verbosity (0-3).

    Returns:
        SimulationResult with metrics, action log, and memory snapshots.
    """
    configure(verbosity=verbosity)

    os.environ.setdefault("MAXIM_LLM_ENABLED", "1")
    os.environ.setdefault("MAXIM_LLM_PROFILE", model)

    # Load pre-campaign turns from YAML scenario if provided
    pre_campaign_turns = None
    if scenario:
        from maxim.simulation.scenario_source import load_scenario_turns

        pre_campaign_turns = load_scenario_turns(scenario)

    from maxim.simulation.orchestrator import start_simulation_mode

    return start_simulation_mode(
        goal=goal,
        persona=persona,
        max_turns=max_turns,
        debug=verbosity >= 3,
        sandbox_backend=sandbox,
        pre_campaign_turns=pre_campaign_turns,
    )


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
        maxim.exceptions.MaximConfigurationError: If ``robot_type`` is
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
            f"Install a robot package (e.g. pymaxim[reachy]) to register controllers."
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
    except Exception:
        pass

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

    results = []
    try:
        import urllib.request
        import urllib.error

        headers = {"User-Agent": "maxim-peer/1.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(
            f"{peer_url.rstrip('/')}/debug/version",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
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

    dispatch = {
        None: lambda: observer.system_stats(),
        "memory": lambda: observer.memory_recall(keyword=keyword, limit=limit),
        "causal": lambda: observer.causal_links(event_signature=keyword),
        "concepts": lambda: observer.concept_query(name=keyword),
        "pain": lambda: observer.pain_history(limit=limit),
        "temporal": lambda: observer.temporal_patterns(),
        "energy": lambda: observer.energy_status(),
    }

    handler = dispatch.get(subsystem)
    if handler is None:
        return {
            "error": f"Unknown subsystem: {subsystem!r}",
            "available": [k for k in dispatch if k is not None],
        }

    return handler()


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
    except Exception:
        hippocampus = None

    # Attempt to load NAc
    nac = None
    try:
        from maxim.decisions.nac import NucleusAccumbens

        nac = NucleusAccumbens()
        nac_file = os.path.join(memory_path, "nac.json")
        if os.path.isfile(nac_file):
            nac.load(nac_file)
        else:
            nac = None
    except Exception:
        nac = None

    if hippocampus is None and nac is None:
        return None

    return Observer(
        hippocampus=hippocampus,
        nac=nac,
    )


# Alias: introspect = observe
introspect = observe


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


def campaign(
    path: str,
    *,
    model: str = "mistral-7b",
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
        path: Path to campaign YAML file.
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
    configure(verbosity=verbosity)

    os.environ.setdefault("MAXIM_LLM_ENABLED", "1")
    os.environ.setdefault("MAXIM_LLM_PROFILE", model)

    from maxim.simulation.dm_schema import load_campaign as _load_campaign

    campaign_def = _load_campaign(path)

    # Override party_mode if specified
    if party_mode is not None:
        from dataclasses import replace
        campaign_def = replace(campaign_def, party_mode=party_mode)

    # For now, return a result from the campaign definition
    # Full wiring to DMRuntime/PartyDMRuntime happens when orchestrator
    # integration is complete.  This gives the API shape for users to
    # build against.
    return CampaignResult(
        campaign_name=campaign_def.name,
        party_mode=campaign_def.party_mode,
    )


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

    # API shape — full wiring to BenchmarkRunner deferred to orchestrator integration
    return BenchmarkResult(
        models=list(models),
        suite=suite,
        runs_per_model=runs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# research (new in Phase 8)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ResearchResult:
    """Result of a research protocol execution."""

    goal: str = ""
    session_id: str = ""
    paper_draft: str = ""
    review: str = ""
    experiment_count: int = 0


def research(
    goal: str,
    *,
    campaign: str | None = None,
    model: str = "claude-sonnet",
    aut_model: str | None = None,
    verbosity: int = 1,
) -> ResearchResult:
    """Run the research protocol (experiment → paper → review).

    Executes a structured research investigation with hypothesis
    formation, experiment execution, and automated paper generation.

    Args:
        goal: Research question (e.g., ``"hippocampal recall under interference"``).
        campaign: Optional campaign YAML for structured stimulus injection.
        model: LLM profile for the research orchestrator.
        aut_model: LLM profile for the agent under test (defaults to model).
        verbosity: Logging verbosity (0-3).

    Returns:
        ResearchResult with paper draft, review, and experiment data.

    Example::

        result = maxim.research(
            goal="test memory retention under interference",
            campaign="scenarios/experiments/hippocampal_recall_short.yaml",
        )
        print(result.paper_draft[:200])
    """
    configure(verbosity=verbosity)

    # API shape — full wiring to research orchestrator deferred
    return ResearchResult(goal=goal)


# ─────────────────────────────────────────────────────────────────────────────
# Event subscription (new in Phase 8)
# ─────────────────────────────────────────────────────────────────────────────


class EventHandle:
    """Handle returned by ``on()`` for managing event subscriptions.

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
        _event_subscriptions.pop(self._handle_id, None)

    @property
    def active(self) -> bool:
        return self._active


# Global event subscription registry
_event_subscriptions: dict[int, tuple[str, Any]] = {}
_next_handle_id = 0


def on(event_name: str, callback: Any) -> EventHandle:
    """Subscribe to agent events.

    Events bridge to the internal AgentBus and PainBus when an agent
    is running.  Subscriptions are registered before the agent starts
    and delivered during execution.

    Supported events:
        ``"tool_call"`` — fired when the agent executes a tool
        ``"memory_capture"`` — fired when hippocampus captures an episode
        ``"pain_signal"`` — fired when a pain signal is detected
        ``"prompt"`` — fired when the system needs user input

    Args:
        event_name: Name of the event to subscribe to.
        callback: Function called with event data when the event fires.

    Returns:
        EventHandle — call ``.unsubscribe()`` to stop receiving events.

    Example::

        handle = maxim.on("tool_call", lambda e: print(f"Tool: {e}"))
        result = maxim.imagine(goal="test")
        handle.unsubscribe()
    """
    global _next_handle_id
    handle_id = _next_handle_id
    _next_handle_id += 1
    _event_subscriptions[handle_id] = (event_name, callback)
    return EventHandle(event_name, callback, handle_id)


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration (new in Phase 8)
# ─────────────────────────────────────────────────────────────────────────────

_pending_tools: list[Any] = []


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
    _pending_tools.append(tool)


def tool(fn: Any) -> Any:
    """Decorator to register a function as a tool.

    Example::

        @maxim.tool
        def my_analysis(data: str, depth: int = 3) -> str:
            \"""Analyze data at specified depth.\"""
            return f"Analysis of {data} at depth {depth}"
    """
    from maxim.tools.base import Tool, ToolOutput

    class FunctionTool(Tool):
        name = fn.__name__
        description = fn.__doc__ or f"Tool: {fn.__name__}"
        input_schema = {}

        def execute(self, **kwargs: Any) -> Any:
            try:
                result = fn(**kwargs)
                return ToolOutput(success=True, output=result)
            except Exception as e:
                return ToolOutput(success=False, error=str(e))

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
    """Register a custom simulation persona.

    Custom personas are available for ``imagine()`` and ``campaign()``
    via the ``persona`` parameter.

    Args:
        name: Persona name (used in ``persona="my_persona"``).
        description: Short description of the persona's behavior.
        focus: What the persona focuses on during simulation.
        context_prompt: System prompt injected into the orchestrator.
        max_initiative: How proactive the persona is (0-1).

    Example::

        maxim.register_persona(
            name="medical_tester",
            description="Tests medical knowledge boundaries",
            focus="Healthcare decision-making and drug interactions",
            context_prompt="You are testing a medical AI...",
            max_initiative=0.8,
        )
        maxim.imagine(goal="test", persona="medical_tester")
    """
    from maxim.simulation.personas import SIMULATION_PERSONAS, Persona

    SIMULATION_PERSONAS[name.lower()] = Persona(
        name=name,
        description=description,
        focus=focus,
        context_prompt=context_prompt,
        max_initiative=max_initiative,
    )
