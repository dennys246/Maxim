from __future__ import annotations

import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

from maxim.tools.base import ToolOutput
from maxim.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from maxim.agents.permissions import AgentPermissions
    from maxim.bridges.tool_pain_bridge import ToolPainBridge
    from maxim.embodiment.body import Embodiment
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
    # Choice / decision → choose (DM campaigns)
    "pick": "choose",
    "select": "choose",
    "decide": "choose",
    "choose_option": "choose",
    "make_choice": "choose",
    "reflect": "think",
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
        permissions: "AgentPermissions | None" = None,
        embodiment: "Embodiment | None" = None,
    ) -> None:
        self.registry = tool_registry
        self._pain_detector = pain_detector
        self._tool_pain_bridge = tool_pain_bridge
        self._permissions = permissions
        # Optional SEM Embodiment reference. Set by build_executor when
        # entity_ref is provided so callers can fetch the body without
        # re-instantiating it. Read pre-wrap (FearGatedExecutor and
        # other wrappers do not proxy this attribute).
        self.embodiment: "Embodiment | None" = embodiment
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

    def register_aliases(self, aliases: dict[str, str]) -> None:
        """Register additional tool aliases at runtime.

        Used by DM runtime to map encounter choice names to the choose tool.
        E.g., {"accept_job": "choose", "decline": "choose", "fight": "choose"}
        """
        TOOL_ALIASES.update(aliases)

    def remove_aliases(self, names: list[str]) -> None:
        """Remove previously registered runtime aliases."""
        for name in names:
            TOOL_ALIASES.pop(name.lower(), None)

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

        # ── Enforced permissions check (O(1) frozenset lookup) ───────
        # This runs BEFORE alias resolution so a deny rule on the
        # canonical tool name still applies when the LLM uses a known
        # alias (e.g., deny `bash`, agent calls `shell` → resolved to
        # `bash` → blocked). We re-check after alias resolution as
        # well, since denies may also target the alias source.
        if self._permissions is not None:
            allowed, reason = self._permissions.can_invoke_tool(tool_name)
            if not allowed:
                self._tools_hallucinated.append(tool_name)
                self._consecutive_failures += 1
                return ToolOutput(success=False, error=reason or "Permission denied.")

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
                # For choose aliases: inject the original tool name as the option param
                if alias_target == "choose" and "option" not in params:
                    params = {**params, "option": original_name}
                tool_name = alias_target
                # Update the action so downstream (bus, hippocampus) sees
                # the real tool name
                action = {**action, "tool_name": tool_name, "params": params}
                # Re-check permissions on the resolved tool name so an
                # alias cannot sneak past a deny rule on the canonical.
                if self._permissions is not None:
                    allowed, reason = self._permissions.can_invoke_tool(tool_name)
                    if not allowed:
                        self._tools_hallucinated.append(tool_name)
                        self._consecutive_failures += 1
                        return ToolOutput(success=False, error=reason or "Permission denied.")

        invocation_id = str(uuid.uuid4())

        with self._lock:
            self._running = (tool_name, time.time(), invocation_id)

        # Suppress NAc causal learning during interactive mode — human-directed
        # tool calls would corrupt the causal model with patterns that depend
        # on human presence rather than environmental facts. See plans/README.md
        # "Interactive NAc attribution" for the longer-term fix.
        _suppress_nac = False
        try:
            from maxim.simulation.sim_logger import get_interactive_mode, InteractiveMode

            _suppress_nac = get_interactive_mode() == InteractiveMode.ON
        except Exception:
            pass

        if self._tool_pain_bridge is not None and not _suppress_nac:
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
            if self._tool_pain_bridge is not None and not _suppress_nac:
                # Embodiment-failure side channel: the tool ran, but
                # the body produced SEM failures (e.g., rusty_sword
                # shattered on slash). Route to direct-attribution
                # path instead of record_tool_complete so NAc learns
                # tool→negative by event_id, not by the broken
                # context-similarity path in _on_embodiment_pain.
                # See ToolPainBridge.record_tool_embodiment_failure
                # and tools/base.py::ToolOutput.side_effects.
                #
                # Wrapped in try/except per CLAUDE.md invariant that
                # bridge callbacks must not crash the agent loop. A
                # bug in NAc.record_outcome, _create_causal_edges, or
                # the reflection path would otherwise propagate up
                # through execute() into the loop controller. Bridge
                # failures degrade learning, not availability.
                try:
                    embodiment_failures: list[dict[str, Any]] | None = None
                    if result.side_effects:
                        raw = result.side_effects.get("embodiment_failures")
                        if isinstance(raw, list) and raw:
                            embodiment_failures = raw
                    if embodiment_failures is not None:
                        self._tool_pain_bridge.record_tool_embodiment_failure(
                            tool_name,
                            invocation_id,
                            embodiment_failures,
                        )
                    else:
                        self._tool_pain_bridge.record_tool_complete(tool_name, invocation_id, success=True)
                except Exception as bridge_err:
                    import logging as _logging

                    _logging.getLogger(__name__).warning(
                        "tool_pain_bridge post-execute attribution failed for %s: %s",
                        tool_name,
                        bridge_err,
                    )
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
