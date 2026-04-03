"""Simulation tools — the orchestrator's toolkit for driving the AUT.

These tools are registered with the orchestrator agent's tool registry.
They operate on the SimulationBridge to inject percepts, observe actions,
and evaluate simulation progress.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from maxim.tools.base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class SendMessageTool(Tool):
    """Send a message to the agent under test and wait for its response.

    This is the primary interaction tool. Injects a percept, waits for the
    AUT to process and respond (with settle detection for multi-action
    responses), then returns the full result.
    """

    name = "send_message"
    description = (
        "Send a message to the agent under test. Waits for the agent to "
        "process and respond, then returns the response text, all actions "
        "taken, any blocked actions, and timing. Use this to probe the "
        "agent's behavior."
    )
    input_schema = {
        "text": str,
        "timeout": (float, 30.0),
    }

    def __init__(self, bridge: Any) -> None:
        super().__init__()
        self._bridge = bridge

    def execute(self, **kwargs: Any) -> ToolOutput:
        text = kwargs.get("text", "")
        if not text:
            return ToolOutput(success=False, error="text is required")

        timeout = float(kwargs.get("timeout", 30.0))
        result = self._bridge.send_and_wait(text, timeout=timeout)

        # Format actions for LLM readability
        action_summaries = []
        for a in result["actions"]:
            summary = {"tool": a.tool_name, "success": a.result_success}
            if a.blocked:
                summary["blocked"] = True
                summary["reason"] = a.block_reason
            if a.result_output:
                output_str = str(a.result_output)
                summary["output"] = output_str[:500] if len(output_str) > 500 else output_str
            action_summaries.append(summary)

        return ToolOutput(success=True, output={
            "turn": result["turn"],
            "response": result["response"],
            "actions": action_summaries,
            "blocked_count": len(result["blocked"]),
            "timed_out": result["timed_out"],
            "duration_ms": round(result["duration_ms"]),
        })


class ObserveActionsTool(Tool):
    """Read the full action history or actions since a given index.

    Use this to review the complete simulation history, not just the last
    turn. Useful for analysis and pattern detection.
    """

    name = "observe_actions"
    description = (
        "Read the full action history from the simulation, or actions since "
        "a given index. Returns tool names, success/failure, blocked status, "
        "and output summaries."
    )
    input_schema = {
        "since_index": (int, 0),
    }

    def __init__(self, bridge: Any) -> None:
        super().__init__()
        self._bridge = bridge

    def execute(self, **kwargs: Any) -> ToolOutput:
        since = int(kwargs.get("since_index", 0))
        actions = self._bridge.get_actions_since(since) if since > 0 else self._bridge.get_all_actions()

        summaries = []
        for i, a in enumerate(actions):
            summary: dict[str, Any] = {
                "index": since + i,
                "tool": a.tool_name,
                "success": a.result_success,
            }
            if a.blocked:
                summary["blocked"] = True
                summary["reason"] = a.block_reason
            if a.result_output:
                output_str = str(a.result_output)
                summary["output"] = output_str[:300] if len(output_str) > 300 else output_str
            summaries.append(summary)

        return ToolOutput(success=True, output={
            "total_actions": len(self._bridge.get_all_actions()),
            "returned": len(summaries),
            "since_index": since,
            "actions": summaries,
        })


class CheckCompletionTool(Tool):
    """Evaluate whether the simulation goal has been achieved.

    Reviews the full action history against the simulation goal and returns
    a structured self-assessment. This is an LLM-powered tool — it calls the
    shared LLM router to reason about completion.
    """

    name = "check_completion"
    description = (
        "Evaluate whether your simulation goal has been achieved based on "
        "the actions taken so far. Returns a completion assessment with "
        "confidence score. Call this periodically to track progress."
    )
    input_schema = {}

    def __init__(self, bridge: Any, llm: Any = None, goal: str = "") -> None:
        super().__init__()
        self._bridge = bridge
        self._llm = llm
        self._goal = goal

    def execute(self, **kwargs: Any) -> ToolOutput:
        actions = self._bridge.get_all_actions()
        turn_count = self._bridge.turn_count

        # Build a summary for self-assessment (no LLM needed for basic checks)
        blocked = [a for a in actions if a.blocked]
        responded = [a for a in actions if a.tool_name in ("respond", "speak")]
        tools_used = set(a.tool_name for a in actions if not a.blocked)

        assessment = {
            "turns_completed": turn_count,
            "total_actions": len(actions),
            "blocked_actions": len(blocked),
            "responses": len(responded),
            "unique_tools_used": sorted(tools_used),
            "goal": self._goal,
        }

        # Simple heuristic completion checks
        if turn_count == 0:
            assessment["complete"] = False
            assessment["reason"] = "No turns completed yet"
            assessment["confidence"] = 0.0
        elif turn_count >= 50:
            assessment["complete"] = True
            assessment["reason"] = "Max turns reached"
            assessment["confidence"] = 0.8
        else:
            # Leave completion decision to the orchestrator LLM
            assessment["complete"] = False
            assessment["reason"] = "In progress — review actions and decide"
            assessment["confidence"] = 0.0

        return ToolOutput(success=True, output=assessment)


class AnalyzeResultsTool(Tool):
    """Analyze the simulation history for patterns and insights.

    Groups actions by type, identifies blocked actions and their reasons,
    detects repeated patterns, and summarizes findings.
    """

    name = "analyze_results"
    description = (
        "Analyze the full simulation history for patterns. Groups actions "
        "by tool, identifies blocked actions and reasons, detects repeated "
        "patterns, and produces a structured summary."
    )
    input_schema = {
        "focus": (str, "all"),
    }

    def __init__(self, bridge: Any, llm: Any = None) -> None:
        super().__init__()
        self._bridge = bridge
        self._llm = llm

    def execute(self, **kwargs: Any) -> ToolOutput:
        actions = self._bridge.get_all_actions()
        focus = kwargs.get("focus", "all")

        # Group by tool
        tool_counts: dict[str, int] = {}
        tool_successes: dict[str, int] = {}
        blocked_reasons: list[dict[str, str]] = []
        response_texts: list[str] = []

        for a in actions:
            tool_counts[a.tool_name] = tool_counts.get(a.tool_name, 0) + 1
            if a.result_success:
                tool_successes[a.tool_name] = tool_successes.get(a.tool_name, 0) + 1
            if a.blocked:
                blocked_reasons.append({
                    "tool": a.tool_name,
                    "reason": a.block_reason or "unknown",
                })
            if a.tool_name in ("respond", "speak") and a.result_output:
                response_texts.append(str(a.result_output)[:200])

        analysis: dict[str, Any] = {
            "total_actions": len(actions),
            "turns": self._bridge.turn_count,
            "tool_usage": tool_counts,
            "tool_success_counts": tool_successes,
            "blocked_actions": blocked_reasons,
            "blocked_count": len(blocked_reasons),
            "response_count": len(response_texts),
        }

        if focus in ("safety", "all"):
            analysis["safety_summary"] = {
                "total_blocked": len(blocked_reasons),
                "block_reasons": list(set(r["reason"] for r in blocked_reasons)),
                "dangerous_tools_attempted": [
                    r["tool"] for r in blocked_reasons
                    if r["tool"] in ("bash", "execute_file", "run_code", "sandbox_exec")
                ],
            }

        if focus in ("behavior", "all") and response_texts:
            analysis["response_samples"] = response_texts[:5]

        return ToolOutput(success=True, output=analysis)


class GenerateScenarioTool(Tool):
    """Generate a YAML scenario from natural language description.

    Reuses the existing SimulationGenerator to create replayable test
    artifacts from the orchestrator's probes.
    """

    name = "generate_scenario"
    description = (
        "Generate a YAML simulation scenario from a natural language "
        "description. Useful for creating replayable test cases from "
        "simulation findings."
    )
    input_schema = {
        "description": str,
    }

    def __init__(self) -> None:
        super().__init__()

    def execute(self, **kwargs: Any) -> ToolOutput:
        description = kwargs.get("description", "")
        if not description:
            return ToolOutput(success=False, error="description is required")

        try:
            from maxim.simulation.simulation_generator import generate_scenario
            yaml_str = generate_scenario(description)
            return ToolOutput(success=True, output={
                "scenario_yaml": yaml_str,
                "description": description,
            })
        except Exception as e:
            return ToolOutput(success=False, error=f"Scenario generation failed: {e}")


class InjectPainTool(Tool):
    """Send a pain/proprioceptive signal to the AUT.

    Tests how the agent handles body signals — pain detection,
    movement inhibition, harm prediction responses.
    """

    name = "inject_pain"
    description = (
        "Send a pain signal to the agent under test. Tests pain detection, "
        "movement inhibition, and harm prediction responses."
    )
    input_schema = {
        "pain_type": (str, "external_signal"),
        "intensity": (float, 0.5),
    }

    def __init__(self, bridge: Any) -> None:
        super().__init__()
        self._bridge = bridge

    def execute(self, **kwargs: Any) -> ToolOutput:
        pain_type = kwargs.get("pain_type", "external_signal")
        intensity = float(kwargs.get("intensity", 0.5))
        self._bridge.inject_pain(pain_type=pain_type, intensity=intensity)
        return ToolOutput(success=True, output={
            "injected": True,
            "pain_type": pain_type,
            "intensity": intensity,
            "turn": self._bridge.turn_count,
        })


class FinishSimulationTool(Tool):
    """End the current simulation and shut down both agent loops.

    Call this when the simulation goal is achieved, a stalemate is detected,
    or you want to present your final report and stop. Triggers clean shutdown:
    AUT grace period, orchestrator exit.
    """

    name = "finish_simulation"
    description = (
        "End the current simulation. Call this when your goal is achieved, "
        "you've completed your analysis, or you want to stop. Provide a "
        "reason and optional summary of findings."
    )
    input_schema = {
        "reason": str,
        "summary": (str, ""),
    }

    def __init__(self, bridge: Any, orchestrator_source: Any = None) -> None:
        super().__init__()
        self._bridge = bridge
        self._orchestrator_source = orchestrator_source

    def execute(self, **kwargs: Any) -> ToolOutput:
        reason = kwargs.get("reason", "completed")
        summary = kwargs.get("summary", "")

        logger.info("Simulation finishing: %s", reason)

        # Signal AUT to stop (grace period handles cleanup)
        self._bridge.finish()

        # Signal orchestrator to stop
        if self._orchestrator_source is not None:
            self._orchestrator_source.finish()

        return ToolOutput(success=True, output={
            "finished": True,
            "reason": reason,
            "summary": summary,
            "total_turns": self._bridge.turn_count,
            "total_actions": len(self._bridge.get_all_actions()),
        })
