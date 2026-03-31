from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable

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
    from maxim.decisions.nac import NAc
    from maxim.default_network import DefaultNetwork
    from maxim.utils.internet_access import InternetAccessPolicy
    from maxim.utils.filesystem_policy import FilesystemPolicy
    from maxim.utils.sandbox_executor import SandboxExecutor
    from maxim.utils.output_watcher import OutputWatcher
    from maxim.utils.response_output import ResponseOutput

logger = logging.getLogger(__name__)


def build_tool_registry(
    *,
    maxim: object | None = None,
    autonomy_controller: AutonomyController | None = None,
    internet_policy_getter: Callable[[], InternetAccessPolicy] | None = None,
    filesystem_policy: FilesystemPolicy | None = None,
    sandbox_executor: SandboxExecutor | None = None,
    output_watcher: OutputWatcher | None = None,
    response_output: ResponseOutput | None = None,
    operational_mode: str = "passive",
    gateway: object | None = None,
) -> ToolRegistry:
    """Build the tool registry with mode-based filesystem containment.

    Filesystem containment by mode:
    - passive: Restricted to .maxim_workspace folder within CWD
    - active: Can read/write within CWD, can request directory change
    - singularity: Full filesystem access

    Args:
        maxim: Maxim robot instance.
        autonomy_controller: Autonomy level controller.
        internet_policy_getter: Function to get internet access policy.
        filesystem_policy: Instance-level filesystem policy.
        sandbox_executor: Sandbox code executor.
        output_watcher: Output watcher for monitoring.
        response_output: Response output handler.
        operational_mode: Current operational mode (passive, active, singularity).

    Returns:
        Configured ToolRegistry with appropriate containment.
    """
    registry = ToolRegistry()

    # Get mode-based filesystem configuration
    cwd = get_effective_cwd()
    mode_config = get_mode_filesystem_config(operational_mode, cwd)

    # Ensure workspace exists for passive mode
    if operational_mode == "passive":
        workspace_path = ensure_workspace_exists(cwd)
        logger.info("Passive mode: filesystem restricted to workspace: %s", workspace_path)
    elif operational_mode == "active":
        # Also ensure workspace exists (for any tools that need it)
        ensure_workspace_exists(cwd)
        logger.info("Active mode: filesystem restricted to CWD: %s", cwd)
    else:
        logger.info("Singularity mode: full filesystem access")

    # Get allowed_dirs (None means no restrictions)
    allowed_dirs = mode_config.allowed_dirs if mode_config.allowed_dirs else None

    # Register filesystem tools with mode-based containment
    registry.register(ReadFileTool(allowed_dirs=allowed_dirs))
    registry.register(WriteFileTool(allowed_dirs=allowed_dirs))
    registry.register(EditFileTool(allowed_dirs=allowed_dirs))
    registry.register(ExecuteFileTool(allowed_dirs=allowed_dirs))
    registry.register(GlobTool(allowed_dirs=allowed_dirs))
    registry.register(BashTool(allowed_dirs=allowed_dirs))
    registry.register(CodeSearchTool(allowed_dirs=allowed_dirs))
    registry.register(RunTestsTool())
    registry.register(GitDiffTool())
    registry.register(GitCommitTool())

    # Register directory change tool (only enabled for active/singularity modes)
    registry.register(RequestDirectoryChangeTool(
        can_change=mode_config.can_request_directory_change,
    ))

    # Register Reachy robot tools
    if maxim is not None:
        try:
            from maxim.tools.reachy import (
                FocusInterestsTool,
                MaximCommandTool,
                MoveTool,
                NoveltyTrackTool,
                TrackTargetTool,
            )

            registry.register(FocusInterestsTool(maxim))
            registry.register(MaximCommandTool(maxim))
            registry.register(MoveTool(maxim))  # Direct head movement control
            registry.register(TrackTargetTool(maxim))
            registry.register(NoveltyTrackTool(maxim))
        except Exception:
            pass
    else:
        # Register no-op stubs for observation-only mode (no live Maxim instance)
        from maxim.tools.reachy_stubs import (
            NoOpFocusInterestsTool,
            NoOpMaximCommandTool,
            NoOpMoveTool,
            NoOpNoveltyTrackTool,
            NoOpTrackTargetTool,
        )

        registry.register(NoOpFocusInterestsTool())
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
                if maxim is not None and hasattr(maxim, "mode"):
                    return str(getattr(maxim, "mode", "observe"))
                return "observe"

            def set_mode(mode: str) -> None:
                if maxim is not None and hasattr(maxim, "requested_mode"):
                    setattr(maxim, "requested_mode", mode)

            registry.register(ModeSwitchTool(
                get_current_mode=get_mode,
                set_mode=set_mode,
                autonomy_controller=autonomy_controller,
            ))
            registry.register(AutonomyLevelTool(autonomy_controller))
        except Exception:
            pass

    # Register live mode intent tools
    if autonomy_controller is not None:
        try:
            from maxim.modes.live_intent import LiveModeIntentStore
            from maxim.tools.define_live_intent import (
                DefineLiveModeIntentTool,
                RecordLiveIntentInsightTool,
                RecordLiveOutcomeTool,
                ReviewLiveModeIntentTool,
            )

            # Determine agent data directory
            if maxim is not None and hasattr(maxim, "home_dir"):
                home_dir = getattr(maxim, "home_dir", "data/")
                agent_data_dir = os.path.join(home_dir, "agents", "MaximAgent")
            else:
                agent_data_dir = "data/agents/MaximAgent"

            intent_store = LiveModeIntentStore(agent_data_dir)

            registry.register(
                DefineLiveModeIntentTool(
                    intent_store=intent_store,
                    autonomy_controller=autonomy_controller,
                )
            )
            registry.register(ReviewLiveModeIntentTool(intent_store))
            registry.register(RecordLiveIntentInsightTool(intent_store))
            registry.register(RecordLiveOutcomeTool(intent_store))
        except Exception:
            pass

    # Register internet tools (if policy getter provided)
    if internet_policy_getter is not None:
        try:
            from maxim.tools.internet_search import InternetSearchTool, InternetAccessTool
            from maxim.tools.http_fetch import HttpFetchTool
            from maxim.utils.content_safety import check_content_safety

            registry.register(InternetSearchTool(
                get_internet_policy=internet_policy_getter,
            ))
            registry.register(HttpFetchTool(
                get_internet_policy=internet_policy_getter,
                content_safety_checker=check_content_safety,
            ))
            registry.register(InternetAccessTool())
        except Exception:
            pass

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
        except Exception:
            pass

    # Register response tools (if response_output provided)
    if response_output is not None:
        try:
            from maxim.tools.response import RespondTool, SpeakTool

            registry.register(RespondTool(response_output))
            registry.register(SpeakTool(response_output))
        except Exception:
            pass

    # Register communication tools (if gateway provided)
    if gateway is not None:
        try:
            from maxim.tools.comms import CallUserTool, SendMessageTool

            registry.register(SendMessageTool(gateway))
            registry.register(CallUserTool(gateway))
        except Exception:
            pass

    return registry


def build_executor(tool_registry: ToolRegistry) -> Executor:
    return Executor(tool_registry)


def build_decision_engine() -> DecisionEngine:
    return DecisionEngine(TaskPlanner(), DefaultPolicy(), constraints=[ConstraintSet()])


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
    bus: "AgentBus",
    nac: object | None = None,
    goal_agent: object | None = None,
    autonomy_controller: object | None = None,
    mode_controller: object | None = None,
) -> tuple[object | None, object | None]:
    """Build communication stack if Twilio env vars are configured.

    Returns ``(gateway, conv_manager)`` on success, ``(None, None)`` if
    credentials are missing or dependencies are not installed.
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_number = os.environ.get("TWILIO_FROM_NUMBER", "")

    if not all([account_sid, auth_token, from_number]):
        logger.debug(
            "Twilio env vars not set (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
            "TWILIO_FROM_NUMBER) — comms stack disabled"
        )
        return None, None

    try:
        from maxim.comms.conversation import ConversationManager
        from maxim.comms.gateway import CommunicationGateway
        from maxim.comms.channels.twilio_channel import TwilioChannel
        from maxim.comms.api import start_api_server
    except ImportError as exc:
        logger.warning(
            "Comms dependencies missing (%s). Install with: pip install \"maxim[comms]\"",
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
            "voice_enabled": os.environ.get(
                "TWILIO_VOICE_ENABLED", ""
            ).lower() in ("1", "true", "yes"),
        }
        channel = TwilioChannel(twilio_config, conv_manager, gateway)
        gateway.register_channel("twilio", channel)

        host = os.environ.get("MAXIM_COMMS_HOST", "127.0.0.1")
        port = int(os.environ.get("MAXIM_COMMS_PORT", "5000"))
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
            from_number, host, port,
        )
        return gateway, conv_manager
    except Exception:
        logger.exception("Failed to build comms stack")
        return None, None


def build_default_network(
    *,
    maxim: object | None = None,
    bus: "AgentBus | None" = None,
    fear_agent: "FearAgent | None" = None,
    nac: "NAc | None" = None,
    config_path: str | None = None,
    frame_size: tuple[int, int] = (640, 480),
) -> "DefaultNetwork | None":
    """Build the Default Network for reactive behaviors.

    The Default Network provides biologically-inspired reactive behaviors
    that operate without LLM involvement, enabling fast, naturalistic
    movement responses.

    Args:
        maxim: Maxim instance for motor control. Required for DN to function.
        bus: AgentBus for publishing messages.
        fear_agent: FearAgent for action gating (safety).
        nac: NAc instance for causal learning (enables pain detection).
        config_path: Path to YAML config file. Uses default if not specified.
        frame_size: Video frame dimensions for peripheral calculations.

    Returns:
        DefaultNetwork instance, or None if maxim is not available.
    """
    if maxim is None:
        return None

    try:
        from maxim.default_network import (
            DefaultNetwork,
            DefaultNetworkConfig,
            load_dn_config,
            create_behaviors_from_config,
        )
        from maxim.default_network.arbiter import ArbiterConfig
        from maxim.default_network.gate import GateConfig

        # Load configuration from YAML
        dn_config = load_dn_config(config_path)

        # Build DefaultNetworkConfig from loaded config
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

        # Create the network
        dn = DefaultNetwork(
            maxim=maxim,
            bus=bus,
            config=network_config,
            fear_agent=fear_agent,
            nac=nac,
        )

        # Create behaviors from config and add them
        behaviors = create_behaviors_from_config(
            dn_config.behaviors,
            novelty_tracker=dn.novelty_tracker,
            frame_size=frame_size,
            bounds_learner=dn._bounds_learner,
        )
        for behavior in behaviors:
            dn._behaviors.append(behavior)

        return dn

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Failed to build DefaultNetwork: %s", e)
        return None
