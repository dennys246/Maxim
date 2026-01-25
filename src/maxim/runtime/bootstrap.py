from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable

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
from maxim.tools.filesystem import ExecuteFileTool, ReadFileTool, WriteFileTool
from maxim.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController
    from maxim.utils.internet_access import InternetAccessPolicy
    from maxim.utils.filesystem_policy import FilesystemPolicy
    from maxim.utils.sandbox_executor import SandboxExecutor
    from maxim.utils.output_watcher import OutputWatcher
    from maxim.utils.response_output import ResponseOutput


def build_tool_registry(
    *,
    maxim: object | None = None,
    autonomy_controller: AutonomyController | None = None,
    internet_policy_getter: Callable[[], InternetAccessPolicy] | None = None,
    filesystem_policy: FilesystemPolicy | None = None,
    sandbox_executor: SandboxExecutor | None = None,
    output_watcher: OutputWatcher | None = None,
    response_output: ResponseOutput | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ExecuteFileTool())

    # Register Reachy robot tools
    if maxim is not None:
        try:
            from maxim.tools.reachy import (
                FocusInterestsTool,
                MaximCommandTool,
                NoveltyTrackTool,
                TrackTargetTool,
            )

            registry.register(FocusInterestsTool(maxim))
            registry.register(MaximCommandTool(maxim))
            registry.register(TrackTargetTool(maxim))
            registry.register(NoveltyTrackTool(maxim))
        except Exception:
            pass
    else:
        # Register no-op stubs for observation-only mode (no live Maxim instance)
        from maxim.tools.reachy_stubs import (
            NoOpFocusInterestsTool,
            NoOpMaximCommandTool,
            NoOpNoveltyTrackTool,
            NoOpTrackTargetTool,
        )

        registry.register(NoOpFocusInterestsTool())
        registry.register(NoOpMaximCommandTool())
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
