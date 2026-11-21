"""AgenticGoalAgent: Enhanced goal management with sub-goal dependencies.

Holds and executes goals with priority-based preemption and robust sub-goal chaining.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from maxim.agents.base import Agent
from maxim.agents.bus import (
    AgentBus,
    GoalAccepted,
    GoalCompleted,
    GoalPriority,
    ProposedGoal,
    SubGoal,
    SubGoalStatus,
    ToolCall,
    ToolResult,
)
from maxim.agents.memory_agent import MemoryAgent
from maxim.utils.structured_logging import log_structured


class AgenticGoalAgent(Agent):
    """
    Holds and executes the current goal with robust sub-goal management.

    Responsibilities:
    - Accept/reject proposed goals based on priority
    - Track current goal state
    - Convert goals to tool calls
    - Report goal completion
    - Handle goal chaining with dependencies
    - Support failure recovery strategies

    Goal hierarchy:
    - CRITICAL always preempts
    - Higher priority preempts lower
    - Same priority: current goal continues

    Goal chaining:
    - Sub-goals execute based on dependencies
    - Parent goal completes when all sub-goals complete
    - Failure strategies: RETRY, SKIP, ABORT_PARENT, ESCALATE
    """

    agent_name = "agentic_goal"

    ROOT_GOAL = "Understand reality and help people."

    def __init__(
        self,
        bus: AgentBus,
        memory_agent: MemoryAgent | None = None,
        *,
        plan_manager: Any = None,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        self._bus = bus
        self._memory = memory_agent
        self._plan_manager = plan_manager

        # Current goal state
        self._current_goal: ProposedGoal | None = None
        self._pending_tool_call: ToolCall | None = None

        # Subscribe to messages
        self._bus.subscribe(ProposedGoal, self._on_proposed_goal)
        self._bus.subscribe(ToolResult, self._on_tool_result)

    def _should_accept(self, proposal: ProposedGoal) -> bool:
        """Decide whether to accept a proposed goal."""
        # Always accept if no current goal
        if self._current_goal is None:
            return True

        # CRITICAL always preempts
        if proposal.priority == GoalPriority.CRITICAL:
            return True

        # Higher priority preempts (lower enum value = higher priority)
        if proposal.priority.value < self._current_goal.priority.value:
            return True

        # Check expiration
        if self._current_goal.expires_at and time.time() > self._current_goal.expires_at:
            return True

        return False

    def _on_proposed_goal(self, proposal: ProposedGoal) -> None:
        """Handle incoming goal proposal."""
        if not self._should_accept(proposal):
            return

        # If we're preempting, mark old goal as incomplete
        if self._current_goal is not None:
            self._complete_goal(
                success=False,
                error=f"Preempted by higher priority goal: {proposal.description}",
            )

        # Accept the new goal
        self._current_goal = proposal
        self._bus.publish(GoalAccepted(goal_id=proposal.id, timestamp=time.time()))

        # Update memory with goal info
        if self._memory:
            sub_goal_descs = [sg.description for sg in proposal.sub_goals]
            self._memory.set_active_goal(proposal.id, proposal.description)
            self._memory.set_active_sub_goals(sub_goal_descs)

        # Log to abstraction stream
        log_structured(
            self.log,
            logging.INFO,
            "goal_accepted",
            {
                "goal_id": proposal.id,
                "description": proposal.description,
                "priority": proposal.priority.name,
                "sub_goals": len(proposal.sub_goals),
            },
        )

        # Create initial tool call
        self._schedule_next_action()

    def _schedule_next_action(self) -> None:
        """Determine and schedule the next action for current goal."""
        if self._current_goal is None:
            return

        goal = self._current_goal

        # If goal has sub-goals, work through them
        if goal.sub_goals:
            next_sub = goal.get_next_executable()
            if next_sub:
                next_sub.status = SubGoalStatus.IN_PROGRESS
                next_sub.started_at = time.time()
                next_sub.attempts += 1
                self._pending_tool_call = ToolCall(
                    id=f"tc-{next_sub.id}",
                    tool_name=next_sub.tool_name,
                    params=next_sub.tool_params,
                    goal_id=goal.id,
                )
                return
            elif goal.is_complete():
                self._complete_goal(success=True, result="All sub-goals completed")
                return
            elif goal.all_failed():
                self._complete_goal(success=False, error="Critical sub-goals failed")
                return
            # No sub-goal ready - wait for dependencies
            return

        # Simple goal with direct tool call
        if goal.tool_name:
            self._pending_tool_call = ToolCall(
                id=f"tc-{goal.id}",
                tool_name=goal.tool_name,
                params=goal.tool_params,
                goal_id=goal.id,
            )

    def _on_tool_result(self, result: ToolResult) -> None:
        """Handle tool completion published on the bus."""
        self.close_tool_call(result.tool_call_id, result.success, result.result, result.error)

    def close_tool_call(self, tool_call_id: str, success: bool, result: Any = None, error: str | None = None) -> bool:
        """Close the pending tool call with the fate of the tool that ran (or raised): the sub-goal policy, then the
        next action or the goal's end. Returns False when ``tool_call_id`` is not the pending call (already closed).

        The agent loop's agent-fallback path (§3) calls this directly (#1414): nothing publishes a ``ToolResult``
        today, and publishing one from §3 alone would wake ``ToolResult``'s dormant subscribers for a biased slice of
        calls and book one failure up to three times (owner decision 2026-11-21: direct close; the bus producer
        belongs to every dispatcher at once, #1727). A refused call is :meth:`refuse_tool_call`'s.
        """
        if self._current_goal is None or self._pending_tool_call is None:
            return False

        if tool_call_id != self._pending_tool_call.id:
            return False

        self._pending_tool_call = None
        goal = self._current_goal

        # Find the sub-goal that was executing (if any)
        current_sub: SubGoal | None = None
        for sg in goal.sub_goals:
            if sg.status == SubGoalStatus.IN_PROGRESS:
                current_sub = sg
                break

        if current_sub:
            if success:
                current_sub.status = SubGoalStatus.COMPLETED
                current_sub.result = result
                current_sub.completed_at = time.time()

                log_structured(
                    self.log,
                    logging.DEBUG,
                    "sub_goal_completed",
                    {
                        "sub_goal_id": current_sub.id,
                        "description": current_sub.description,
                    },
                )
            else:
                should_continue = goal.handle_sub_goal_failure(current_sub, error or "Unknown error")
                if not should_continue:
                    self._complete_goal(
                        success=False,
                        error=f"Sub-goal failed: {current_sub.error}",
                    )
                    return True

                log_structured(
                    self.log,
                    logging.WARNING,
                    "sub_goal_failed",
                    {
                        "sub_goal_id": current_sub.id,
                        "error": current_sub.error,
                        "will_retry": current_sub.should_retry(),
                    },
                )

            self._continue_goal(goal)
        else:
            # Simple goal (no sub-goals)
            self._complete_goal(
                success=success,
                result=result,
                error=error,
            )
        return True

    def _complete_goal(
        self,
        success: bool,
        result: Any = None,
        error: str | None = None,
        *,
        refused: bool = False,
    ) -> None:
        """Complete the current goal (``refused``: autonomy refused its tool call, #1414)."""
        if self._current_goal is None:
            return

        completed = GoalCompleted(
            goal_id=self._current_goal.id,
            success=success,
            result=result,
            error=error,
            refused=refused,
        )

        log_structured(
            self.log,
            logging.INFO if success else logging.WARNING,
            "goal_completed",
            {
                "goal_id": self._current_goal.id,
                "success": success,
                "error": error,
            },
        )

        self._bus.publish(completed)
        self._current_goal = None
        self._pending_tool_call = None

        if self._memory:
            self._memory.set_active_goal(None, None)
            self._memory.set_active_sub_goals([])

        # Delegate phase advancement to PlanManager if a plan is active
        if self._plan_manager and self._plan_manager.active_plan:
            try:
                self._plan_manager.advance_phase(success, result, error)
            except Exception:
                pass  # PlanManager failure should not block goal completion

    def get_current_goal(self) -> ProposedGoal | None:
        """Get the current active goal."""
        return self._current_goal

    def get_pending_tool_call(self) -> ToolCall | None:
        """Get the pending tool call."""
        return self._pending_tool_call

    def propose_intent(self, state: Any, memory: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Return pending tool call as intent."""
        if self._pending_tool_call:
            tool_call = self._pending_tool_call
            return {
                "goal": {
                    "tool_name": tool_call.tool_name,
                    "params": tool_call.params,
                },
                "goal_id": tool_call.goal_id,
                "tool_call_id": tool_call.id,
                "confidence": 1.0,
                "source": "agentic_goal_agent",
            }
        return None

    def report_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        success: bool,
        result: Any = None,
        error: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Report a tool result back to the goal agent."""
        tr = ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=success,
            result=result,
            error=error,
            params=params or {},
        )
        self._bus.publish(tr)

    def _continue_goal(self, goal: ProposedGoal) -> None:
        """After a sub-goal's call closed and the goal goes on: memory's remaining sub-goals, then the next action."""
        if self._memory:
            remaining = [
                sg.description
                for sg in goal.sub_goals
                if sg.status not in (SubGoalStatus.COMPLETED, SubGoalStatus.SKIPPED)
            ]
            self._memory.set_active_sub_goals(remaining)
        self._schedule_next_action()

    def refuse_tool_call(self, tool_call_id: str, reason: str) -> bool:
        """The loop refused this agent's pending tool call (autonomy, #1414): close it, without a retry.

        A refusal is not a tool result: the tool never ran, so it does not go through ``ToolResult`` (whose
        subscribers book tool successes and failures). The sub-goal policy for a refusal is
        ``ProposedGoal.handle_sub_goal_refusal`` (only SKIP goes on); a refused call with no sub-goal ends the goal.
        Returns False when ``tool_call_id`` is not the pending call (already closed).
        """
        goal, pending = self._current_goal, self._pending_tool_call  # one read of each: ExecAgent's thread writes them
        if pending is None or pending.id != tool_call_id:
            return False
        self._pending_tool_call = None
        current = next((sg for sg in goal.sub_goals if sg.status == SubGoalStatus.IN_PROGRESS), None) if goal else None
        if goal is not None and current is not None and goal.handle_sub_goal_refusal(current, reason):
            self._continue_goal(goal)
            return True
        self._complete_goal(success=False, error=f"Refused: {reason}", refused=True)
        return True

    def cancel_current_goal(self, reason: str = "Cancelled by user") -> bool:
        """Cancel the current goal."""
        if self._current_goal is None:
            return False

        self._complete_goal(success=False, error=reason)
        return True

    def on_stop(self, **kwargs: Any) -> None:
        """Clean up on stop."""
        # Complete any in-progress goal
        if self._current_goal:
            self._complete_goal(success=False, error="Agent stopped")

        self._bus.unsubscribe(ProposedGoal, self._on_proposed_goal)
        self._bus.unsubscribe(ToolResult, self._on_tool_result)
