from __future__ import annotations

import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

from maxim.tools.base import ToolOutput
from maxim.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from maxim.bridges.tool_pain_bridge import ToolPainBridge
    from maxim.proprioception.pain import PainDetector


# ── Tool alias map ────────────────────────────────────────────────────────
# LLMs (especially small ones) hallucinate tool names from their training
# data rather than using the registered tool list.  This map silently
# redirects common hallucinations to the correct registered tool.
#
# How to expand: add entries mapping the hallucinated name (lowercase) to
# the registered tool name.  The executor normalises the incoming name to
# lowercase before lookup, so casing variations are handled automatically.
#
# See also: docs/troubleshooting/tool_aliases.md
TOOL_ALIASES: dict[str, str] = {
    # Memory / recall → memory_recall
    "remember": "memory_recall",
    "recall": "memory_recall",
    "recall_memory": "memory_recall",
    "search_memory": "memory_recall",
    # Speech / dialogue → say
    "speech_recognition": "say",
    "speechrecognition": "say",
    "speech": "say",
    "dialogue": "say",
    "talk": "say",
    # NLP / analysis → think
    "natural_language_processing": "think",
    "nlp": "think",
    "nlp_extractor": "think",
    "nlp_understanding": "think",
    "reflection": "think",
    "analyze_text": "think",
    "research": "think",
    # Dialogue parsing → think
    "dialogue_parser": "think",
    "dialogueparser": "think",
    "parse_dialogue": "think",
    # Internet search → memory_recall (in sim, there's no internet)
    "internet_search": "memory_recall",
    "web_search": "memory_recall",
    # Inspection / observation → examine
    "inspect": "examine",
    "look": "examine",
    "observe": "examine",
    "look_at": "examine",
    "investigate": "examine",
}


class Executor:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        pain_detector: "PainDetector | None" = None,
        tool_pain_bridge: "ToolPainBridge | None" = None,
    ) -> None:
        self.registry = tool_registry
        self._pain_detector = pain_detector
        self._tool_pain_bridge = tool_pain_bridge
        self._lock = threading.Lock()
        # (tool_name, start_time, invocation_id) or None
        self._running: tuple[str, float, str] | None = None
        # Track alias redirects for experiment analysis
        self.alias_redirects: list[tuple[str, str]] = []
        # Tool usage tracking (Phase 5c)
        self._tools_attempted: list[str] = []
        self._tools_succeeded: list[str] = []
        self._tools_hallucinated: list[str] = []
        self._consecutive_failures: int = 0

    def execute(self, action: dict[str, Any]) -> ToolOutput:
        """Execute a tool action, returning raw ToolOutput.

        The caller (agent loop) is responsible for converting this to a
        bus ToolResult (agents.bus.ToolResult) with tool_call_id/tool_name/params
        before publishing on the bus.

        If the requested tool name is not registered but matches an entry
        in TOOL_ALIASES, the request is silently redirected to the correct
        tool.  This is logged and tracked in ``self.alias_redirects`` for
        experiment analysis.
        """
        tool_name = action.get("tool_name")
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        if not isinstance(tool_name, str) or not tool_name:
            return ToolOutput(success=False, error=f"Invalid action: {action!r}")

        self._tools_attempted.append(tool_name)

        # ── Alias resolution ─────────────────────────────────────────
        original_name = tool_name
        if tool_name not in self.registry._tools:
            alias_target = TOOL_ALIASES.get(tool_name.lower())
            if alias_target and alias_target in self.registry._tools:
                import logging

                logging.getLogger(__name__).info(
                    "Tool alias: %s → %s",
                    tool_name,
                    alias_target,
                )
                self.alias_redirects.append((tool_name, alias_target))
                tool_name = alias_target
                # Update the action so downstream (bus, hippocampus) sees
                # the real tool name
                action = {**action, "tool_name": tool_name}

        invocation_id = str(uuid.uuid4())

        with self._lock:
            self._running = (tool_name, time.time(), invocation_id)

        if self._tool_pain_bridge is not None:
            self._tool_pain_bridge.record_tool_start(tool_name, invocation_id, context={"params": params})

        try:
            tool = self.registry.get(tool_name)
        except KeyError:
            with self._lock:
                self._running = None
            self._tools_hallucinated.append(original_name)
            self._consecutive_failures += 1
            error_msg = f"Tool not registered: {tool_name!r}."
            suggestions = self.registry.find_similar(original_name, limit=3)
            if suggestions:
                error_msg += f" Did you mean: {', '.join(suggestions)}?"
            # Phase 5d: proactive tool list after repeated failures
            if self._consecutive_failures >= 2:
                available = sorted(self.registry.list())
                error_msg += f" Available tools: {', '.join(available)}."
            else:
                error_msg += (
                    " Only use tools from the Available Tools list."
                    " Use 'memory_recall' to remember, 'say' to speak aloud,"
                    " 'think' to reason."
                )
            result = ToolOutput(success=False, error=error_msg)
            self._report_failure(tool_name, invocation_id, result, params)
            return result

        try:
            result = tool.run(**params)
        except Exception as e:
            with self._lock:
                self._running = None
            result = ToolOutput(success=False, error=f"Tool {tool_name!r} execution failed: {e}")
            self._report_failure(tool_name, invocation_id, result, params)
            return result

        with self._lock:
            self._running = None

        if result.success:
            self._tools_succeeded.append(tool_name)
            self._consecutive_failures = 0
            if self._tool_pain_bridge is not None:
                self._tool_pain_bridge.record_tool_complete(tool_name, invocation_id, success=True)
        else:
            self._report_failure(tool_name, invocation_id, result, params)

        return result

    def _report_failure(
        self,
        tool_name: str,
        invocation_id: str,
        result: ToolOutput,
        params: dict[str, Any],
    ) -> None:
        """Report a tool failure to pain detector and bridge."""
        if self._pain_detector is not None:
            from maxim.agents.bus import ToolErrorKind

            self._pain_detector.record_tool_error(
                tool_name=tool_name,
                error=result.error or "unknown",
                error_kind=result.error_kind or ToolErrorKind.EXTERNAL_FAILURE,
                context={
                    "params": params,
                    "invocation_id": invocation_id,
                    "metadata": result.metadata,
                },
            )

    def get_last_rpe(self) -> float:
        """Get RPE magnitude from the most recent tool execution.

        Returns the Rescorla-Wagner prediction error from NAc, which
        reflects how surprising the tool outcome was.  High RPE signals
        that hippocampus should boost salience for this memory.
        """
        if self._tool_pain_bridge is not None:
            return self._tool_pain_bridge._last_rpe
        return 0.0

    def tool_usage_stats(self) -> dict[str, Any]:
        """Get tool usage statistics for experiment analysis."""
        return {
            "tools_attempted": list(self._tools_attempted),
            "tools_succeeded": list(self._tools_succeeded),
            "tools_hallucinated": list(self._tools_hallucinated),
            "alias_redirects": [(orig, target) for orig, target in self.alias_redirects],
            "total_attempts": len(self._tools_attempted),
            "total_successes": len(self._tools_succeeded),
            "total_hallucinated": len(self._tools_hallucinated),
            "hallucination_rate": (
                len(self._tools_hallucinated) / len(self._tools_attempted) if self._tools_attempted else 0.0
            ),
        }

    def get_running_tool(self) -> tuple[str, float, str] | None:
        """Get the currently running tool info.

        Returns:
            Tuple of (tool_name, start_time, invocation_id) or None.
        """
        with self._lock:
            return self._running
