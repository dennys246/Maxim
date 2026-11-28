from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal

from maxim.environment.filesystem_env import FileSystemEnv
from maxim.evaluation.agent_eval import AgentEvaluator
from maxim.evaluation.plan_eval import PlanEvaluator
from maxim.evaluation.tool_eval import ToolExecutionEvaluator
from maxim.memory import InMemoryMemory
from maxim.planning.constraints import ConstraintSet
from maxim.planning.decision_engine import DecisionEngine
from maxim.planning.planning import TaskPlanner
from maxim.planning.policy import DefaultPolicy
from maxim.runtime.executor import Executor
from maxim.runtime.tool_policy import ToolPolicy, require_policy
from maxim.runtime.state import RuntimeState
from maxim.tools.code_tools import CodeSearchTool, RunTestsTool
from maxim.tools.git_tools import GitDiffTool, GitCommitTool
from maxim.tools.filesystem import (
    EditFileTool,
    ExecuteFileTool,
    ReadFileTool,
    WriteFileTool,
    GlobTool,
    BashTool,
    RequestDirectoryChangeTool,
)
from maxim.tools.registry import ToolRegistry
from maxim.utils.filesystem_policy import (
    ensure_workspace_exists,
    get_effective_cwd,
    get_mode_filesystem_config,
)

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController
    from maxim.agents.bus import AgentBus
    from maxim.agents.fear_agent import FearAgent
    from maxim.bridges.tool_pain_bridge import ToolPainBridge
    from maxim.decisions.nac import NAc
    from maxim.default_network import DefaultNetwork, DefaultNetworkConfig
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.proprioception.pain import PainDetector
    from maxim.proprioception.pain_bus import PainBus
    from maxim.utils.filesystem_policy import FilesystemPolicy
    from maxim.utils.sandbox_executor import SandboxExecutor
    from maxim.utils.output_watcher import OutputWatcher
    from maxim.utils.response_output import ResponseOutput

logger = logging.getLogger(__name__)


def build_tool_registry(
    *,
    internet_launch_enabled: bool,
    maxim: object | None = None,
    autonomy_controller: AutonomyController | None = None,
    filesystem_policy: FilesystemPolicy | None = None,
    sandbox_executor: SandboxExecutor | None = None,
    output_watcher: OutputWatcher | None = None,
    response_output: ResponseOutput | None = None,
    operational_mode: str = "passive",
    gateway: object | None = None,
    allowed_dirs_override: list[str] | None = None,
    state_manager: object | None = None,
    prompt_handler: Any = None,
) -> ToolRegistry:
    """Build the tool registry with mode-based filesystem containment.

    Filesystem containment by mode:
    - passive: Restricted to .maxim_workspace folder within CWD
    - active: Can read/write within CWD, can request directory change
    - singularity: Full filesystem access

    Args:
        maxim: Live legacy Maxim runtime or ``RobotController`` context.
            Controller-only contexts register only controller-backed tools;
            legacy capture/vision/DoA tools require the full Maxim runtime.
        internet_launch_enabled: REQUIRED. The launch cap on internet access (``--no-internet``, an
            exploration policy's ``allow_internet``). When True the builder registers the internet
            tools with the ONE live policy getter (``live_internet_policy_getter``: the operator's
            persisted policy plus the runtime toggle, read per request); when False it registers none.
            Required so a caller states the decision, and the builder, not the caller, builds the
            getter (#832): no caller can hand the tools a bare policy.
        autonomy_controller: Autonomy level controller.
        filesystem_policy: Instance-level filesystem policy.
        sandbox_executor: Sandbox code executor.
        output_watcher: Output watcher for monitoring.
        response_output: Response output handler.
        operational_mode: Current operational mode (passive, active, singularity).
        allowed_dirs_override: If set, overrides mode-based allowed_dirs.
            Used by simulation to confine tools to the sandbox tmpdir.

    Returns:
        Configured ToolRegistry with appropriate containment.
    """
    registry = ToolRegistry()

    # Get mode-based filesystem configuration
    cwd = get_effective_cwd()
    mode_config = get_mode_filesystem_config(operational_mode, cwd)

    # The `.maxim_workspace/` scaffold lives under the containment root: an
    # override root when one is given, the CWD otherwise. Before this the
    # scaffold was ALWAYS created under the CWD, even when every filesystem
    # tool was scoped elsewhere by `allowed_dirs_override` — so a caller that
    # had deliberately kept the process CWD out of the agent's reach (the
    # console handle; a read-only-root deployment) still wrote into it on the
    # first registry build (sandbox-launch, 2026-09-03).
    workspace_base = allowed_dirs_override[0] if allowed_dirs_override else cwd
    if operational_mode == "passive":
        workspace_path = ensure_workspace_exists(workspace_base)
        logger.info("Passive mode: filesystem restricted to workspace: %s", workspace_path)
    elif operational_mode == "active":
        # Also ensure workspace exists (for any tools that need it)
        ensure_workspace_exists(workspace_base)
        logger.info("Active mode: filesystem restricted to %s", workspace_base)
    else:
        logger.info("Singularity mode: full filesystem access")

    # Get allowed_dirs — override takes priority (e.g., simulation sandbox)
    if allowed_dirs_override is not None:
        allowed_dirs = [os.path.realpath(d) for d in allowed_dirs_override]
        logger.info("Filesystem override: restricted to %s", allowed_dirs)
    else:
        allowed_dirs = mode_config.allowed_dirs if mode_config.allowed_dirs else None

    # Register filesystem tools with mode-based containment
    registry.register(ReadFileTool(allowed_dirs=allowed_dirs))
    registry.register(WriteFileTool(allowed_dirs=allowed_dirs))
    registry.register(EditFileTool(allowed_dirs=allowed_dirs))
    registry.register(ExecuteFileTool(allowed_dirs=allowed_dirs))
    registry.register(GlobTool(allowed_dirs=allowed_dirs))
    registry.register(BashTool(allowed_dirs=allowed_dirs))
    registry.register(CodeSearchTool(allowed_dirs=allowed_dirs))
    registry.register(RunTestsTool(allowed_dirs=allowed_dirs))
    registry.register(GitDiffTool(allowed_dirs=allowed_dirs))
    registry.register(GitCommitTool(allowed_dirs=allowed_dirs))

    # Register directory change tool (only enabled for active/singularity modes)
    registry.register(
        RequestDirectoryChangeTool(
            can_change=mode_config.can_request_directory_change,
        )
    )

    # Register display + interaction tools. These respect the global
    # interactive mode via sim_logger.should_prompt, so registering them
    # unconditionally is safe — when interactive is off, the tool returns
    # a "disabled" message without touching the handler.
    from maxim.tools.display import DisplayModeTool, RequestInteractionTool, SetSceneTool

    registry.register(DisplayModeTool())
    registry.register(RequestInteractionTool(prompt_handler=prompt_handler))
    registry.register(SetSceneTool())

    # Register Reachy robot tools
    if maxim is not None:
        try:
            from maxim.tools.reachy import (
                FocusInterestsTool,
                FocusOnSoundTool,
                MaximCommandTool,
                MoveTool,
                NoveltyTrackTool,
                TrackTargetTool,
            )

            from maxim.hardware.controller import RobotController

            controller_only = isinstance(maxim, RobotController)
            if controller_only:
                # RobotController exposes safe direct motion, but not the
                # legacy runtime's capture manager, detector, DoA feed, or
                # command surface. Do not advertise tools guaranteed to fail.
                registry.register(MoveTool(maxim))
            else:
                registry.register(FocusInterestsTool(maxim))
                registry.register(FocusOnSoundTool(maxim))  # Closed-loop audio orienting (no numeric params)
                registry.register(MaximCommandTool(maxim))
                registry.register(MoveTool(maxim))  # Direct head movement control
                registry.register(TrackTargetTool(maxim))
                registry.register(NoveltyTrackTool(maxim))
        except ImportError:
            logger.debug("Reachy tools not available (optional dependency)")
        except Exception as e:
            logger.warning("Failed to register Reachy tools: %s", e)
    else:
        # Register no-op stubs for observation-only mode (no live Maxim instance)
        from maxim.tools.reachy_stubs import (
            NoOpFocusInterestsTool,
            NoOpFocusOnSoundTool,
            NoOpMaximCommandTool,
            NoOpMoveTool,
            NoOpNoveltyTrackTool,
            NoOpTrackTargetTool,
        )

        registry.register(NoOpFocusInterestsTool())
        registry.register(NoOpFocusOnSoundTool())  # Audio orienting (no-op)
        registry.register(NoOpMaximCommandTool())
        registry.register(NoOpMoveTool())  # Direct head movement control (no-op)
        registry.register(NoOpTrackTargetTool())
        registry.register(NoOpNoveltyTrackTool())

    # Register mode switch tool
    if autonomy_controller is not None:
        try:
            from maxim.tools.mode_switch import ModeSwitchTool, AutonomyLevelTool

            # Mode switch requires callbacks - provide defaults if maxim not available
            def get_mode() -> str:
                # the operator's launch grant is the current capability when set (#829)
                granted = getattr(maxim, "launch_operational_mode", None) if maxim is not None else None
                if isinstance(granted, str) and granted:
                    return granted
                if maxim is not None and hasattr(maxim, "mode"):
                    return str(getattr(maxim, "mode", "observe"))
                return "observe"

            def set_mode(mode: str) -> bool:
                """False when there is no runtime to apply a mode to (the tool then reports failure)."""
                if maxim is not None and hasattr(maxim, "requested_mode"):
                    setattr(maxim, "requested_mode", mode)
                    return True
                return False

            registry.register(
                ModeSwitchTool(
                    get_current_mode=get_mode,
                    set_mode=set_mode,
                    autonomy_controller=autonomy_controller,
                )
            )
            registry.register(AutonomyLevelTool(autonomy_controller))
        except ImportError:
            logger.debug("Mode switch tools not available (optional dependency)")
        except Exception as e:
            logger.warning("Failed to register mode switch tools: %s", e)

    # Register sleep tool
    if state_manager is not None:
        try:
            from maxim.tools.sleep import SleepTool

            registry.register(SleepTool(state_manager=state_manager))
        except ImportError:
            logger.debug("Sleep tool not available (optional dependency)")
        except Exception as e:
            logger.warning("Failed to register sleep tool: %s", e)

    # Register internet tools (only under the launch cap; the getter is always the live one)
    from maxim.utils.internet_access import live_internet_policy_getter

    internet_policy_getter = live_internet_policy_getter(internet_launch_enabled)
    if internet_policy_getter is not None:
        try:
            from maxim.tools.internet_search import InternetSearchTool, InternetAccessTool
            from maxim.tools.http_fetch import HttpFetchTool
            from maxim.utils.content_safety import check_content_safety

            registry.register(
                InternetSearchTool(
                    get_internet_policy=internet_policy_getter,
                )
            )
            registry.register(
                HttpFetchTool(
                    get_internet_policy=internet_policy_getter,
                    content_safety_checker=check_content_safety,
                )
            )
            registry.register(InternetAccessTool())
        except ImportError:
            logger.debug("Internet tools not available (optional dependency)")
        except Exception as e:
            logger.warning("Failed to register internet tools: %s", e)

    # Register sandbox tools (if policy and executor provided)
    if filesystem_policy is not None and sandbox_executor is not None:
        try:
            from maxim.tools.sandbox import build_sandbox_tools

            sandbox_tools = build_sandbox_tools(
                policy=filesystem_policy,
                executor=sandbox_executor,
                watcher=output_watcher,
                autonomy_controller=autonomy_controller,
            )
            for tool in sandbox_tools:
                registry.register(tool)
        except ImportError:
            logger.debug("Sandbox tools not available (optional dependency)")
        except Exception as e:
            logger.warning("Failed to register sandbox tools: %s", e)

    # Register response tools (if response_output provided)
    if response_output is not None:
        try:
            from maxim.tools.response import RespondTool, SpeakTool

            registry.register(RespondTool(response_output))
            registry.register(SpeakTool(response_output))
        except ImportError:
            logger.debug("Response tools not available (optional dependency)")
        except Exception as e:
            logger.warning("Failed to register response tools: %s", e)

    # Register communication tools (if gateway provided)
    if gateway is not None:
        try:
            from maxim.tools.comms import CallUserTool, SendMessageTool

            registry.register(SendMessageTool(gateway))
            registry.register(CallUserTool(gateway))
        except ImportError:
            logger.debug("Communication tools not available (optional dependency)")
        except Exception as e:
            logger.warning("Failed to register communication tools: %s", e)

    return registry


def _check_build_executor_args(
    *,
    pain_bus: "PainBus | None",
    pain_detector: "PainDetector | None",
    entity_ref: str | None,
    component_registry: "ComponentRegistry | None",
    nac: "NAc | None",
    modulator_factory: Any,
) -> None:
    """``build_executor``'s fail-fast preconditions, run BEFORE any construction (moved out of it, #1315 item 3)."""
    if pain_bus is not None and pain_detector is not None:
        raise ValueError(
            "build_executor: pain_bus AND pain_detector were both provided. "
            "ToolPainBridge.__init__ uses pain_bus if set and silently ignores "
            "pain_detector — pick one."
        )
    if entity_ref is not None and component_registry is None:
        raise ValueError(
            f"build_executor: entity_ref={entity_ref!r} was provided but "
            "component_registry is None. Pass a ComponentRegistry instance "
            "or leave entity_ref as None."
        )
    if entity_ref is not None and pain_bus is None:
        raise ValueError(
            f"build_executor: entity_ref={entity_ref!r} was provided but "
            "pain_bus is None. Embodiment._publish_pain emits through the "
            "bus; pass a PainBus instance."
        )
    if entity_ref is not None and nac is None:
        raise ValueError(
            f"build_executor: entity_ref={entity_ref!r} was provided but "
            "nac is None. Embodiment failures need a bridge for NAc "
            "attribution — pass an NAc instance."
        )
    if (pain_bus is not None or pain_detector is not None) and nac is None:
        raise ValueError(
            "build_executor: pain_bus or pain_detector was provided but nac "
            "is None. Subscribing the bridge without NAc has no effect — "
            "pass an NAc instance or set both subscription sources to None."
        )
    if modulator_factory is not None and entity_ref is None:
        raise ValueError(
            "build_executor: modulator_factory was provided but entity_ref "
            "is None — there is no entity tree to attach backends to. "
            "A silently-ignored factory is a virtual body that lies success "
            "(sem_motor_binding.md); pass entity_ref or drop the factory."
        )


def build_executor(
    tool_registry: ToolRegistry,
    *,
    pain_bus: "PainBus | None",
    policy: "ToolPolicy",
    nac: "NAc | None" = None,
    hippocampus: Any = None,
    scn: Any = None,
    pain_detector: "PainDetector | None" = None,
    tool_index: Any = None,
    entity_ref: str | None = None,
    component_registry: "ComponentRegistry | None" = None,
    cerebellum: Any = None,
    entity_map: Any = None,
    distributor: Any = None,
    agent_id: str = "",
    modulator_factory: Any = None,
) -> Executor:
    """Build an Executor with an explicit ToolPainBridge decision.

    The canonical agent-construction site. Folds in the bridge wiring
    that previously lived in ``runtime/embodiment_bootstrap.py``, and
    enforces the bridge invariant at the constructor level: every
    caller must make an explicit ``pain_bus=`` decision.

    Three identical bug instances (sem_execution_hook Stages 1, 2, 2c)
    showed that "remember to call the helper after build_executor" is
    not a strong enough invariant — five of six call sites in
    ``src/maxim/`` were either wrong, implicit, or relied on
    helper-discipline. This signature makes the failure mode a
    ``TypeError`` instead of a silent no-op. See
    ``docs/plans/archive/executor_bootstrap_unification.md``.

    Args:
        tool_registry: Registry of tools the executor can dispatch.
        pain_bus: REQUIRED keyword (no default). Pass a live
            ``PainBus`` to subscribe the bridge to bus-published
            pain signals, or ``None`` to skip subscription. Note:
            this controls SUBSCRIPTION only — bridge construction
            is gated on ``nac`` (see below). The structural intent:
            forgetting to make an explicit pain-bus decision at the
            call site is a TypeError.
        nac: NAc instance. **Constructing the bridge is gated on
            this, not on `pain_bus`/`pain_detector`.** Pass a real
            NAc to get a bridge wired for ``record_tool_complete`` /
            ``record_tool_embodiment_failure`` direct attribution
            (the Stage 1 primary path); pass ``None`` to skip the
            bridge entirely (sandbox executors, headless tests).
            Subscription via ``pain_bus``/``pain_detector`` is the
            secondary out-of-band attribution path; both can be
            ``None`` even when ``nac`` is set, in which case the
            bridge exists for direct attribution only.
        hippocampus: Optional ``Hippocampus`` for reflexion memory
            storage. Forwarded to the bridge.
        scn: Optional ``SCN`` for temporal signature registration.
        pain_detector: Optional legacy subscription path for the
            bridge. Pre-PainBus call sites (Reachy runtime) wire
            tool-error signals through ``PainDetector.add_pain_callback``.
            Mutually exclusive with ``pain_bus``.
        tool_index: Optional ``LearnedToolIndex`` for keyword-weight
            updates on tool outcomes.
        entity_ref: Optional SEM component reference (e.g.,
            ``"weapons/rusty_sword"``). When provided, loads the
            entity, wraps it in ``Embodiment``, and registers
            affordance tools into ``tool_registry``. Requires
            ``component_registry`` AND ``pain_bus`` AND ``nac``.
        component_registry: Required iff ``entity_ref`` is set.
        cerebellum: Optional ``Cerebellum`` for forward-model training.
        policy: REQUIRED ``ToolPolicy`` (#1315 item 3, V3): what the executor may dispatch. ``None`` is a
            ``TypeError``, never an allow-all.
        modulator_factory: Optional ``attach_backends``-shaped factory
            (``(entity, mod_name, spec_modulator) -> backend | None``)
            attaching LIVE hardware backends to spec-declared modulators
            before tool generation (sem_motor_binding.md Phase 1 — real
            body turns on live Reachy). ``None`` (every sim/headless
            caller) keeps stub semantics byte-identical. Requires
            ``entity_ref`` (there is no entity tree to attach to
            otherwise — passing it alone raises).

    Returns:
        Unwrapped inner ``Executor`` with the bridge attached when
        ``nac`` was set (otherwise no bridge). When ``entity_ref``
        was provided, the loaded ``Embodiment`` is also stored on
        ``Executor.embodiment`` as a declared field so callers can
        reference it without re-instantiation.

    Raises:
        ValueError: ``pain_bus`` AND ``pain_detector`` are both set
            (ambiguous subscription path); or ``entity_ref`` is set
            but ``component_registry``, ``pain_bus``, or ``nac`` is
            missing; or a subscription source is set but ``nac`` is
            None.
        ComponentNotFoundError: ``entity_ref`` does not resolve in
            the registry.

    Wrapping-order invariant: the returned ``executor`` is the
    unwrapped inner ``Executor``. Wrapping layers (``FearGatedExecutor``,
    ``PainInterceptorExecutor``, ``AnticipatoryPainExecutor``) MUST
    be applied AFTER this call. Wrapping the result and then trying
    to attach a bridge to the wrapper would silently restore the
    Stage 1 no-op bug. The bridge lives on the inner Executor
    because that is the object that runs ``tool.run`` and reads
    ``self._tool_pain_bridge``.
    """
    # ── Fail-fast precondition checks (run BEFORE any construction) ──
    _check_build_executor_args(
        pain_bus=pain_bus,
        pain_detector=pain_detector,
        entity_ref=entity_ref,
        component_registry=component_registry,
        nac=nac,
        modulator_factory=modulator_factory,
    )
    require_policy(policy, where="build_executor")

    # ── Bridge construction (gated on nac, not subscription) ─────────
    # The bridge's PRIMARY value is direct attribution via
    # `record_tool_complete` / `record_tool_embodiment_failure`
    # (Stage 1 root-cause fix). Subscription via pain_bus/pain_detector
    # is the SECONDARY out-of-band path. So the bridge is constructed
    # whenever NAc is available; subscription is layered on top if the
    # caller provided a source. Pre-fold the gating was inverted —
    # callers wanting direct-attribution-only had to pass a no-op
    # PainDetector to trick the constructor (cross-confirmed C2/C3
    # finding from the pre-merge review round).
    bridge: "ToolPainBridge | None" = None
    if nac is not None:
        # Local import — `bridges/tool_pain_bridge` pulls substrate
        # components we don't want to pay for on opt-out call sites.
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        bridge = ToolPainBridge(
            nac=nac,
            pain_detector=pain_detector,
            scn=scn,
            hippocampus=hippocampus,
            tool_index=tool_index,
            pain_bus=pain_bus,
            distributor=distributor,
            agent_id=agent_id,
        )

    # ── Optional embodiment loading (BEFORE Executor construction) ──
    # Loading the embodiment first lets us pass it to Executor.__init__
    # as a declared field, avoiding a post-construction attribute stash.
    embodiment: "Embodiment | None" = None
    entity = None
    if entity_ref is not None:
        # Imports here so non-embodiment call sites don't pay the cost.
        from maxim.embodiment.body import Embodiment

        # Resolver errors propagate to the caller with their original
        # exception type + message — `ComponentNotFoundError` includes
        # a sorted list of available refs so the user can spot a typo.
        entity = component_registry.instantiate(entity_ref)  # type: ignore[union-attr]

        # SEM motor binding (sem_motor_binding.md Phase 1): attach live
        # hardware backends to spec-declared modulators BEFORE tool
        # generation. None (the default, every sim/headless caller) keeps
        # SpecModulator's stub semantics byte-identical. The factory is
        # threaded through the canonical builder — not attached post-hoc
        # by the caller — per the push-silent-no-ops-into-types lesson:
        # a forgotten attach is a virtual body that LIES success.
        if modulator_factory is not None:
            from maxim.embodiment.spec import attach_backends

            attach_backends(entity, modulator_factory=modulator_factory)

        embodiment = Embodiment(entity, pain_bus=pain_bus, agent_id=agent_id)

        # Post-construction backend hookup: an attached motor backend
        # (a) declares which sensors it world-owns — its MEASURED readback
        # is then the single writer and the modeled self_effect on those
        # keys is filtered (the same guard mechanism the DoAFeed uses for
        # ``azimuth``); (b) receives the Embodiment wrapper so its
        # world-set goes through the canonical ``world_set_axis`` API.
        if modulator_factory is not None:
            from maxim.embodiment.spec import SpecModulator

            for ent in entity.walk():
                for mod in ent.modulators.values():
                    backend = getattr(mod, "_backend", None) if isinstance(mod, SpecModulator) else None
                    if backend is None:
                        continue
                    for sensor_name in getattr(backend, "world_owned_sensors", ()) or ():
                        embodiment.live_world_set_sensors.add(str(sensor_name))
                    bind = getattr(backend, "bind_embodiment", None)
                    if callable(bind):
                        bind(embodiment)

    # D79 fix (b): the executor is constructed WITH its generation
    # collaborators as declared fields, and the initial generation runs
    # through the SAME seam acquisition regeneration uses
    # (Executor.generate_entity_tools) — the pre-fix shape generated here
    # with a hand-threaded collaborator list and never stashed
    # `entity_map`/`cerebellum`, so Mechanism-B acquisition was a silent
    # no-op through this canonical builder (the takes-but-does-not-stash
    # class, third miss — the counting rule's structural answer).
    executor = Executor(
        tool_registry=tool_registry,
        pain_detector=pain_detector,
        tool_pain_bridge=bridge,
        policy=policy,
        embodiment=embodiment,
        cerebellum=cerebellum,
        entity_map=entity_map,
    )

    if entity is not None:
        generated = executor.generate_entity_tools(entity)
        logger.info(
            "build_executor: Embodiment loaded from %r — %d affordance tools registered",
            entity_ref,
            len(generated),
        )

    if bridge is not None:
        logger.info(
            "build_executor: ToolPainBridge wired (entity_ref=%s)",
            entity_ref or "<none>",
        )

    return executor


def build_decision_engine(
    *,
    nac: "NAc | None" = None,
    hippocampus: Any = None,
    ec: Any = None,
    atl: Any = None,
    concept_context_builder: Any = None,
    retrieval_orchestrator: Any = None,
    llm: Any = None,
) -> DecisionEngine:
    """Build a DecisionEngine with the best available planner and policy.

    When memory systems are provided, uses AdaptivePlanner (deep memory
    integration) and AdaptivePolicy (6-dimension scoring).  Falls back to
    TaskPlanner + DefaultPolicy when no memory systems are available.
    """
    has_memory = any(x is not None for x in (nac, hippocampus, ec))

    if has_memory:
        from maxim.planning.adaptive_planner import AdaptivePlanner
        from maxim.planning.adaptive_policy import AdaptivePolicy

        planner: Any = AdaptivePlanner(
            nac=nac,
            hippocampus=hippocampus,
            ec=ec,
            atl=atl,
            concept_context_builder=concept_context_builder,
            retrieval_orchestrator=retrieval_orchestrator,
            llm=llm,
        )
        policy: Any = AdaptivePolicy(nac=nac)
    else:
        planner = TaskPlanner()
        policy = DefaultPolicy()

    return DecisionEngine(planner, policy, constraints=[ConstraintSet()])


def build_environment(*, root: str | None = None) -> FileSystemEnv:
    return FileSystemEnv(root or os.getcwd())


def build_state(*, max_steps: int = 100) -> RuntimeState:
    return RuntimeState(max_steps=max_steps)


def build_memory() -> InMemoryMemory:
    return InMemoryMemory()


def build_evaluators() -> list:
    return [AgentEvaluator(), PlanEvaluator(), ToolExecutionEvaluator()]


def build_comms_stack(
    *,
    agent: Any,
    nac: object | None = None,
    goal_agent: object | None = None,
    autonomy_controller: object | None = None,
    mode_controller: object | None = None,
) -> tuple[object | None, object | None]:
    """Build communication stack if Twilio env vars are configured.

    Returns ``(gateway, conv_manager)`` on success, ``(None, None)`` if
    credentials are missing or dependencies are not installed.

    Takes the AGENT, never a bus (#1481 D1): inbound messages are published on ``agent._bus``, the one bus the run's
    loop watches (``loop_external.inbound_comms_subscription``), so comms can never be wired to a bus no loop reads.
    An agent without a bus raises ``ConfigurationError``.
    """
    bus = getattr(agent, "_bus", None)
    if bus is None:
        from maxim.exceptions import ConfigurationError

        raise ConfigurationError("build_comms_stack: the agent has no bus, so no loop would see its inbound messages")
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_number = os.environ.get("TWILIO_FROM_NUMBER", "")

    if not all([account_sid, auth_token, from_number]):
        logger.debug(
            "Twilio env vars not set (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER) — comms stack disabled"
        )
        return None, None

    try:
        from maxim.comms.conversation import ConversationManager
        from maxim.comms.gateway import CommunicationGateway
        from maxim.comms.channels.twilio_channel import TwilioChannel
        from maxim.comms.api import start_api_server
    except ImportError as exc:
        logger.warning(
            'Comms dependencies missing (%s). Install with: pip install "maxim[comms]"',
            exc,
        )
        return None, None

    try:
        conv_manager = ConversationManager(bus=bus, nac=nac)
        gateway = CommunicationGateway(bus=bus)

        twilio_config = {
            "account_sid": account_sid,
            "auth_token": auth_token,
            "from_number": from_number,
            "voice_enabled": os.environ.get("TWILIO_VOICE_ENABLED", "").lower() in ("1", "true", "yes"),
        }
        channel = TwilioChannel(twilio_config, conv_manager, gateway)
        gateway.register_channel("twilio", channel)

        host = os.environ.get("MAXIM_COMMS_HOST", "127.0.0.1")
        try:
            port = int(os.environ.get("MAXIM_COMMS_PORT", "5000"))
        except (ValueError, TypeError):
            port = 5000
        start_api_server(
            bus=bus,
            gateway=gateway,
            goal_agent=goal_agent,
            autonomy_controller=autonomy_controller,
            mode_controller=mode_controller,
            host=host,
            port=port,
        )
        logger.info(
            "Comms stack started (Twilio from=%s, API on %s:%d)",
            from_number,
            host,
            port,
        )
        return gateway, conv_manager
    except Exception:
        logger.exception("Failed to build comms stack")
        return None, None


def build_default_network(
    *,
    nac: "NAc | None",
    maxim: object | None = None,
    bus: "AgentBus | None" = None,
    pain_bus: "PainBus | None" = None,
    fear_agent: "FearAgent | None" = None,
    config: "DefaultNetworkConfig | None" = None,
    config_path: str | None = None,
    frame_size: tuple[int, int] = (640, 480),
    has_vision: bool = True,
) -> "DefaultNetwork | None":
    """Build the Default Network for reactive behaviors.

    This is the canonical DefaultNetwork construction site (Wave 2 of
    biosystem_unification). ``nac`` is a REQUIRED keyword-only
    argument — forgetting it is a ``TypeError``, not a silent no-op.
    Pass ``None`` to explicitly opt out of pain detection + causal
    learning (the core learning subsystem DN provides).

    The Layer 4 → Layer 5 upgrade (see
    ``docs/plans/archive/default_network_unification.md`` Gap A): the previous
    version had all-optional parameters and swallowed all exceptions
    into ``None``. Post-upgrade, ``nac`` is required, and only import
    failures produce a ``None`` return (optional dep not installed).
    Config and type errors propagate to the caller.

    ``pain_bus`` injection closes Gap B from
    ``docs/plans/archive/pain_bus_unification.md``: when provided, DN uses the
    injected bus (which should already have hippocampus + NAc
    subscribers from ``build_pain_bus``) instead of constructing its
    own internally. This inverts DN from bus constructor to bus
    consumer, closing the split-subscriber-ownership problem where
    hippocampus was wired externally at ``agentic_runtime.py:719``.

    ``maxim=None`` is the headless/sim opt-out — DN still provides
    pain detection and novelty tracking without motor control. The
    previous version returned ``None`` early on ``maxim=None``, which
    was wrong for sim mode (sim needs DN for pain detection but not
    motor). Post-upgrade, DN constructs normally with ``maxim=None``;
    motor methods gracefully no-op.

    Args:
        nac: NAc instance for causal learning. Required keyword-only.
            Pass ``None`` for explicit opt-out. Drives pain detection
            via PainCircuitBridge + focus learner.
        maxim: Maxim instance for motor control. ``None`` = headless/sim.
        bus: AgentBus for publishing messages.
        pain_bus: Injected PainBus. When provided, DN uses this instead
            of constructing its own. Should have hippocampus + NAc
            subscribers already wired via ``build_pain_bus``.
        fear_agent: FearAgent for movement gating within DN.
        config: Pre-built DefaultNetworkConfig. When provided, takes
            precedence over ``config_path``.
        config_path: Path to YAML config file. Ignored if ``config``
            is provided. Uses default if neither is specified.
        frame_size: Video frame dimensions for peripheral calculations.

    Returns:
        DefaultNetwork instance, or ``None`` if the default_network
        package is not installed (optional dep).
    """
    try:
        from maxim.default_network import (
            DefaultNetwork,
            DefaultNetworkConfig,
            load_dn_config,
            create_behaviors_from_config,
        )
        from maxim.default_network.arbiter import ArbiterConfig
        from maxim.default_network.gate import GateConfig
    except ImportError as e:
        import logging

        logging.getLogger(__name__).warning(
            "DefaultNetwork not available (optional dep): %s",
            e,
        )
        return None

    # Build config — explicit config takes precedence over config_path.
    if config is None:
        dn_config = load_dn_config(config_path)
        network_config = DefaultNetworkConfig(
            enabled=dn_config.enabled,
            update_hz=dn_config.update_hz,
            publish_actions=dn_config.publish_actions,
            fear_gate_enabled=dn_config.fear_gate_enabled,
            arbiter=ArbiterConfig(
                hysteresis_bonus=dn_config.arbiter.hysteresis_bonus,
                min_switch_interval=dn_config.arbiter.min_switch_interval,
                score_threshold=dn_config.arbiter.score_threshold,
            ),
            gate=GateConfig(
                novelty_threshold=dn_config.gate.novelty_threshold,
                salience_threshold=dn_config.gate.salience_threshold,
                anomaly_threshold=dn_config.gate.anomaly_threshold,
                adaptive=dn_config.gate.adaptive,
            ),
        )
    else:
        network_config = config
        dn_config = None  # no YAML loaded, skip behavior creation from config

    # Capability truth (2026-08-01 live-smoke fold): without a camera the
    # DN's idle visual exploration fabricates gaze targets from the
    # spatial-map grid — it fires PRECISELY when detections are empty,
    # which under a no_media backend is every tick. Each phantom look_at
    # then dies in the SDK ("Camera is not initialized"), pollutes
    # FocusLearner intent-tracking, and gets blamed for unrelated drive
    # pain via context-similarity attribution. No vision → no visual
    # exploration, applied to BOTH config branches (a caller's explicit
    # config cannot re-enable a behavior the hardware cannot perform).
    if not has_vision and getattr(network_config, "idle_exploration_enabled", False):
        import dataclasses as _dc
        import logging as _logging

        network_config = _dc.replace(network_config, idle_exploration_enabled=False)
        _logging.getLogger(__name__).info("DN idle visual exploration disabled (no camera on this robot)")

    # Create the network — maxim=None is valid (headless/sim mode).
    dn = DefaultNetwork(
        maxim=maxim,
        bus=bus,
        config=network_config,
        fear_agent=fear_agent,
        nac=nac,
        pain_bus=pain_bus,
    )

    # Create behaviors from config and add them (only when loaded from YAML).
    if dn_config is not None:
        behaviors = create_behaviors_from_config(
            dn_config.behaviors,
            novelty_tracker=dn.novelty_tracker,
            frame_size=frame_size,
            bounds_learner=dn._bounds_learner,
        )
        for behavior in behaviors:
            dn._behaviors.append(behavior)

    return dn


# ── The one sanctioned caller of the agent loop (docs/plans/loop_stack_builder.md) ───────────────────────────────────

# Every ``run_agentic_loop`` parameter, classified. STACK parameters are the entry point's capability and identity:
# ``run_loop_stack`` takes each as a REQUIRED keyword, so a site that forgets one gets a TypeError, never a silently
# less capable loop (the ``build_executor(pain_bus=...)`` lesson). RUN parameters are run configuration and keep the
# loop's own defaults. ``tests/unit/test_run_loop_stack.py`` asserts the keys equal the loop's signature exactly, so a
# new loop parameter fails CI until it is classified here.
LOOP_PARAMETERS: dict[str, Literal["stack", "run"]] = {
    "agent": "stack",
    "environment": "stack",
    "state": "stack",
    "memory": "stack",
    "decision_engine": "stack",
    "executor": "stack",
    "autonomy_controller": "stack",
    "llm_worker": "stack",
    "default_network": "stack",
    "hippocampus": "stack",
    "memory_hub": "stack",
    "evaluators": "stack",
    "max_steps": "run",
    "run_id": "run",
    "stop_event": "run",
    "on_step": "run",
    "on_event": "run",
    "idle_sleep_s": "run",
    "persist_every_n_steps": "run",
    "target_hz": "run",
    "context_pool_config": "run",
    "use_tool_prompting": "run",
    "protocol_registry": "stack",
    "percept_source": "stack",
    "response_policy": "stack",
    "action_sink": "stack",
    "pain_bus": "stack",
    "imagination_trigger": "stack",
    "bio_enrichment_pipeline": "stack",
    "thought_gate": "stack",
    "aut_mode": "stack",
    "substrate_telemetry": "stack",
    "substrate_policy": "stack",
    "substrate_action_gate": "stack",
    "consolidation": "stack",
    "planning_liveness": "run",
    "sim_adapter": "stack",
    "end_bound": "run",
    "state_home": "stack",
}


def run_loop_stack(
    *,
    agent: Any,
    environment: Any,
    state: Any,
    memory: Any,
    decision_engine: Any,
    executor: Any,
    autonomy_controller: Any,
    llm_worker: Any,
    hippocampus: Any,
    memory_hub: Any,
    pain_bus: Any,
    bio_enrichment_pipeline: Any,
    thought_gate: Any,
    default_network: Any,
    imagination_trigger: Any,
    evaluators: Any,
    protocol_registry: Any,
    sim_adapter: Any,
    aut_mode: str,
    substrate_telemetry: Any,
    substrate_policy: Any,
    substrate_action_gate: Any,
    percept_source: Any,
    action_sink: Any,
    response_policy: Any,
    state_home: Any,
    consolidation: Any,
    acting_coach: Any,
    is_embodied: bool,
    max_steps: int = 0,
    run_id: str | None = None,
    stop_event: Any | None = None,
    on_step: Any | None = None,
    on_event: Any | None = None,
    idle_sleep_s: float = 0.05,
    persist_every_n_steps: int = 10,
    target_hz: float = 30.0,
    context_pool_config: dict[str, Any] | None = None,
    use_tool_prompting: bool = True,
    planning_liveness: bool = False,
    end_bound: Any | None = None,
) -> None:
    """Run the agent loop with a stack the caller states in full: the ONE sanctioned ``run_agentic_loop`` caller.

    Every STACK keyword in ``LOOP_PARAMETERS`` is required, with no default: a site states each subsystem it wires,
    and each one it leaves out (``None``), so a forgotten component is a TypeError, not a silently less capable entry
    point. RUN keywords keep the loop's defaults. ``scripts/lint_loop_stack.py`` fails any other ``run_agentic_loop``
    reference outside its ledger of unmigrated sites, and any call of this function through an alias, a ``partial``
    or a ``**`` splat, so a reviewer sees every stated ``None`` at the call.

    It builds nothing and mutates nothing: each site keeps constructing its own agent, worker and per-run objects, on
    the thread and at the time it always has. Before the loop it checks, raising ``ValueError`` on a mismatch:

    - **the hub, both ways:** ``agent.memory_hub is memory_hub`` (``None``/``None`` passes). A hub handed to the loop
      but never wired, or a wired agent handed ``memory_hub=None``, raises. This one detects a real wiring mismatch.
    - **the acting coach and embodiment:** ``acting_coach`` must be ``llm_worker.acting_coach`` (by identity) and
      ``is_embodied`` must equal ``llm_worker.is_embodied``; with ``llm_worker=None`` they must be ``None`` and
      ``False``. These are STATEMENTS, not detection: they make a site write down the worker state it built, and
      cannot tell whether it should be embodied.

    ``run_agentic_loop`` is looked up on its module when called (imported here for import cost; there is no cycle),
    so a test that patches ``agent_loop.run_agentic_loop`` still reaches its stub. Every argument is passed through
    unchanged, BY KEYWORD (the six positionals too), so a ``def run_agentic_loop(**kwargs)`` stub works. Design and owner decisions: ``docs/plans/loop_stack_builder.md``.
    """
    if agent.memory_hub is not memory_hub:
        raise ValueError(
            "run_loop_stack: the agent's memory hub is not the hub handed to the loop "
            f"(agent holds {type(agent.memory_hub).__name__}, the loop gets {type(memory_hub).__name__}); "
            "wire the agent with wire_memory_hub(hub) before the call, or pass the hub it holds"
        )
    if llm_worker is None:
        if acting_coach is not None or is_embodied is not False:
            raise ValueError(
                "run_loop_stack: llm_worker=None has no coach and is not embodied: pass acting_coach=None, "
                "is_embodied=False"
            )
    else:
        if llm_worker.acting_coach is not acting_coach:
            raise ValueError(
                "run_loop_stack: acting_coach= is not the coach the worker holds (llm_worker.acting_coach)"
            )
        if llm_worker.is_embodied != is_embodied:
            raise ValueError(
                f"run_loop_stack: is_embodied={is_embodied!r}, but the worker says {llm_worker.is_embodied!r}"
            )

    from maxim.runtime import agent_loop

    agent_loop.run_agentic_loop(  # every argument by keyword, the positionals too: a keyword-only stub still works
        agent=agent,
        environment=environment,
        state=state,
        memory=memory,
        decision_engine=decision_engine,
        executor=executor,
        autonomy_controller=autonomy_controller,
        llm_worker=llm_worker,
        default_network=default_network,
        hippocampus=hippocampus,
        memory_hub=memory_hub,
        evaluators=evaluators,
        max_steps=max_steps,
        run_id=run_id,
        stop_event=stop_event,
        on_step=on_step,
        on_event=on_event,
        idle_sleep_s=idle_sleep_s,
        persist_every_n_steps=persist_every_n_steps,
        target_hz=target_hz,
        context_pool_config=context_pool_config,
        use_tool_prompting=use_tool_prompting,
        protocol_registry=protocol_registry,
        percept_source=percept_source,
        response_policy=response_policy,
        action_sink=action_sink,
        pain_bus=pain_bus,
        imagination_trigger=imagination_trigger,
        bio_enrichment_pipeline=bio_enrichment_pipeline,
        thought_gate=thought_gate,
        aut_mode=aut_mode,
        substrate_telemetry=substrate_telemetry,
        substrate_policy=substrate_policy,
        substrate_action_gate=substrate_action_gate,
        consolidation=consolidation,
        planning_liveness=planning_liveness,
        sim_adapter=sim_adapter,
        end_bound=end_bound,
        state_home=state_home,
    )
