"""MaximAgent: Composite agent embodying the full agentic architecture.

Integrates:
- PerceptionAgent: Observes and classifies inputs
- MemoryAgent: Maintains salient context (CORE EXPLORATION FOCUS)
- ExecAgent: Proposes goals via LLM
- AgenticGoalAgent: Holds and executes goals with sub-goal chaining

Root goal: "Understand reality and help people."
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from maxim.agents.base import Agent
from maxim.agents.bus import AgentBus, ToolResult
from maxim.agents.perception_agent import PerceptionAgent
from maxim.agents.memory_agent import MemoryAgent
from maxim.agents.exec_agent import ExecAgent
from maxim.agents.agentic_goal_agent import AgenticGoalAgent

if TYPE_CHECKING:
    from maxim.runtime.capture import CaptureManager


class MaximAgent(Agent):
    """
    Composite agent embodying the full Maxim agentic architecture.

    Root goal: "Understand reality and help people."

    Components:
    - PerceptionAgent: Observes and classifies inputs
    - MemoryAgent: Maintains salient context (CORE EXPLORATION FOCUS)
    - ExecAgent: Proposes goals via LLM
    - AgenticGoalAgent: Holds and executes goals with sub-goal chaining

    Data Flow:
    1. ReachyEnv.observe() → raw observation
    2. PerceptionAgent → Percept (normalized, classified, salience/novelty)
    3. MemoryAgent → updates salient memory, builds StructuredContext
    4. ExecAgent → proposes ProposedGoal based on context + LLM
    5. AgenticGoalAgent → accepts/rejects goal, executes via tools
    6. ToolResult → fed back to MemoryAgent for outcome tracking
    """

    agent_name = "maxim"
    state_name = "MaximAgent"

    ROOT_GOAL = "Understand reality and help people."

    def __init__(
        self,
        *,
        name: str | None = None,
        enabled: bool = True,
        interests: list[int] | None = None,
        llm_profile: str = "mistral-7b-instruct-v0.2",
        quantization: str = "Q4_K_M",
        memory_persistence_path: str | None = None,
        data_folder: str | None = None,
        enable_embeddings: bool = False,
        reset_on_startup: bool = False,
        capture_manager: CaptureManager | None = None,
    ) -> None:
        super().__init__(name=name, enabled=enabled)

        # Create the message bus
        self._bus = AgentBus()

        # Default persistence path if not specified
        if memory_persistence_path is None and data_folder:
            memory_persistence_path = os.path.join(data_folder, "memory", "memories.json")

        # Store capture manager reference (Phase 3)
        self._capture_manager = capture_manager

        # Create agents in dependency order
        self.perception = PerceptionAgent(
            self._bus,
            interests=set(interests or []),
            capture_manager=capture_manager,
        )

        self.memory = MemoryAgent(
            self._bus,
            max_short_term=100,
            max_long_term=500,
            salience_threshold=0.2,
            persistence_path=memory_persistence_path,
            enable_embeddings=enable_embeddings,
            reset_on_startup=reset_on_startup,
        )

        self.exec_agent = ExecAgent(
            self._bus,
            self.memory,
            llm_profile=llm_profile,
            quantization=quantization,
        )

        self.goal = AgenticGoalAgent(
            self._bus,
            memory_agent=self.memory,
        )

        # Throttle control
        self._last_intent_time = 0.0
        self._min_intent_interval = 0.1  # 10 Hz max
        self._last_log_time = 0.0
        self._log_interval = 1.0  # Log at most once per second

    def on_start(self, **kwargs: Any) -> None:
        """Start all component agents."""
        self.perception.on_start(**kwargs)
        self.memory.on_start(**kwargs)
        self.exec_agent.on_start(**kwargs)
        self.goal.on_start(**kwargs)

    def on_stop(self, **kwargs: Any) -> None:
        """Stop all component agents."""
        self.goal.on_stop(**kwargs)
        self.exec_agent.on_stop(**kwargs)
        self.memory.on_stop(**kwargs)
        self.perception.on_stop(**kwargs)
        self._bus.clear()

    def propose_intent(self, state: Any, memory: Any, **kwargs: Any) -> dict[str, Any] | None:
        """
        Run the full perception-memory-goal pipeline.

        1. Perception processes observation
        2. Memory updates and builds context
        3. ExecAgent proposes goals (async)
        4. GoalAgent returns current goal as intent
        """
        now = time.time()

        # Throttle to avoid spinning too fast
        elapsed = now - self._last_intent_time
        if elapsed < self._min_intent_interval:
            return None  # Let agent loop handle idle sleep

        self._last_intent_time = now

        # Step 1: Perception (publishes Percept to bus)
        self.perception.propose_intent(state, memory, **kwargs)

        # Step 2: ExecAgent checks for proposals needed
        exec_result = self.exec_agent.propose_intent(state, memory, **kwargs)

        # Step 3: GoalAgent returns pending tool call
        goal_result = self.goal.propose_intent(state, memory, **kwargs)
        if goal_result:
            self._log_status("goal_active", goal_result)
            return goal_result

        # Fall back to exec result (startup, file change, default focus)
        if exec_result:
            self._log_status("exec_result", exec_result)
            return exec_result

        # Log idle status periodically
        self._log_status("idle", None)
        return None

    def _log_status(self, status: str, result: Any) -> None:
        """Log status at most once per second."""
        now = time.time()
        if now - self._last_log_time >= self._log_interval:
            self._last_log_time = now
            if status == "idle":
                self.log.debug("Idle - no goal pending")
            elif status == "goal_active":
                goal = result.get("goal") if result else None
                if goal is None:
                    goal = {}
                tool = goal.get("tool_name") if isinstance(goal, dict) else str(goal)
                self.log.info("Goal: %s", tool)
            elif status == "exec_result":
                source = result.get("source", "unknown") if result else "unknown"
                self.log.info("Action from %s", source)

    def publish_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        success: bool,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Publish tool result for feedback."""
        tr = ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=success,
            result=result,
            error=error,
        )
        self._bus.publish(tr)

    def report_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        success: bool,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Alias for publish_tool_result for compatibility."""
        self.publish_tool_result(tool_call_id, tool_name, success, result, error)

    def update_interests(
        self, add: list[int] | None = None, remove: list[int] | None = None
    ) -> None:
        """Update object class interests for salience computation."""
        self.perception.update_interests(add=add, remove=remove)

    def record_cli_input(self, command: str) -> None:
        """Record a CLI input for context (flows to abstraction stream)."""
        self.memory.add_cli_input(command)

    def get_current_goal(self) -> str:
        """Get description of current goal."""
        goal = self.goal.get_current_goal()
        if goal:
            return goal.description
        return self.ROOT_GOAL

    def get_pending_sub_goals(self) -> list[str]:
        """Get pending sub-goals for current goal."""
        goal = self.goal.get_current_goal()
        if goal and goal.sub_goals:
            return [
                sg.description
                for sg in goal.sub_goals
                if sg.status.name not in ("COMPLETED", "SKIPPED")
            ]
        return []

    def get_current_goal_id(self) -> str | None:
        """Get ID of current goal."""
        goal = self.goal.get_current_goal()
        return goal.id if goal else None

    def cancel_goal(self, reason: str = "Cancelled") -> bool:
        """Cancel the current goal."""
        return self.goal.cancel_current_goal(reason)

    def get_mode(self) -> str:
        """Get the current operating mode."""
        ctx = self.memory.build_context()
        return ctx.mode

    def get_context_summary(self) -> dict[str, Any]:
        """Get a summary of current context for debugging."""
        ctx = self.memory.build_context()
        return {
            "mode": ctx.mode,
            "active_goal": ctx.active_goal,
            "sub_goals_pending": len(ctx.active_goal_sub_goals),
            "recent_percepts": len(ctx.recent_percepts),
            "relevant_memories": len(ctx.relevant_memories),
            "detected_objects": len(ctx.detected_objects),
            "detected_people": len(ctx.detected_people),
            "detected_speech": len(ctx.detected_speech),
            "cli_inputs": len(ctx.cli_inputs),
        }


# Backwards-compatibility alias
AgenticMaximAgent = MaximAgent
