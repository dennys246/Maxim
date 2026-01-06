from __future__ import annotations

import os

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


def build_tool_registry(*, maxim: object | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ExecuteFileTool())
    if maxim is not None:
        try:
            from maxim.tools.reachy import FocusInterestsTool, MaximCommandTool

            registry.register(FocusInterestsTool(maxim))
            registry.register(MaximCommandTool(maxim))
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
