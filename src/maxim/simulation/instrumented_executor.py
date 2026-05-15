"""InstrumentedExecutor — wraps Executor to record all actions to a sink.

Captures every tool execution (success, failure, and autonomy rejections)
as ActionRecords in a RecordingSink. Transparently wraps an existing
Executor without changing its interface.

Stage 0b (release_0_9_1.md) telemetry: each record carries
``agent_id`` / ``session_id`` from the ``utils/http.py::current_context``
ContextVar (bound at the sim orchestrator entry) and a best-effort
``entity_class`` derived from the action's params. The fields default
to ``None`` when context isn't bound (e.g., unit tests, headless API),
so the producer never raises.
"""

from __future__ import annotations

import time
from typing import Any

from maxim.simulation.sinks import ActionRecord, ActionSink
from maxim.tools.base import ToolOutput
from maxim.utils.http import current_context


def _derive_entity_class(tool_name: str, params: dict[str, Any]) -> str | None:
    """Best-effort entity-class extraction for Stage 0b telemetry.

    Roy-3 analysis normalizes pain-aversion counts per entity_class —
    knowing the agent encountered "food" 50 times but felt pain 3 times
    matters very differently from encountering "food" 5 times and feeling
    pain 3 times. The exact derivation is fuzzy because Maxim's tool
    surface mixes verb-only tools (``respond``, ``examine``) with
    entity-bound tools (``infant_humanoid_pick_up``, ``sense_food_source``).

    Heuristics (lowest-cost-to-highest):
    1. ``params["entity_class"]`` — explicit caller override (highest fidelity).
    2. ``params["target"]`` / ``params["entity"]`` / ``params["object"]`` —
       the conventional param names entity-binding tools use.
    3. Tool-name prefix split — ``infant_humanoid_pick_up`` → ``infant_humanoid``;
       ``sense_food_source`` → ``food`` (heuristic: skip leading verb token).

    Returns ``None`` when nothing in the action surface looks
    entity-bound (e.g., ``respond``, ``examine``, sleep tools). The
    field is best-effort metadata, never load-bearing for correctness;
    Roy-3 analysis aggregates with ``None`` skipped from
    exposure-count normalization.
    """
    if not isinstance(params, dict):
        return None
    # 1. Explicit caller override.
    explicit = params.get("entity_class")
    if isinstance(explicit, str) and explicit:
        return explicit
    # 2. Conventional param names.
    for key in ("target", "entity", "object"):
        val = params.get(key)
        if isinstance(val, str) and val:
            return val
    # 3. Tool-name heuristic. Strip leading verb tokens.
    if "_" in tool_name:
        parts = tool_name.split("_")
        # Common verb prefixes that tools start with — these are NOT
        # the entity class.
        verb_prefixes = {"sense", "use", "do", "get", "set", "make", "go", "look", "examine"}
        if parts and parts[0].lower() in verb_prefixes:
            remainder = "_".join(parts[1:])
            if remainder:
                # ``sense_food_source`` → ``food_source`` after strip;
                # further trim trailing role tokens like ``_source`` /
                # ``_target`` so the bucket reads as ``food``.
                role_suffixes = {"source", "target", "object"}
                tail_parts = remainder.split("_")
                while tail_parts and tail_parts[-1].lower() in role_suffixes:
                    tail_parts.pop()
                if tail_parts:
                    return "_".join(tail_parts)
                return remainder
    return None


class InstrumentedExecutor:
    """Wraps an Executor to record all actions to an ActionSink.

    Drop-in replacement for Executor — same execute() interface.
    All calls are forwarded to the wrapped executor, and results
    are recorded in the sink.

    Example:
        sink = RecordingSink()
        instrumented = InstrumentedExecutor(real_executor, sink)
        result = instrumented.execute({"tool_name": "read_file", "params": {...}})
        # result is the real ToolOutput
        # sink.actions now contains the ActionRecord
    """

    def __init__(self, executor: Any, sink: ActionSink) -> None:
        self._executor = executor
        self._sink = sink

    def _telemetry_fields(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Pull Stage 0b telemetry (agent_id, session_id, entity_class)
        off the bound RequestContext + tool action."""
        ctx = current_context()
        return {
            "agent_id": ctx.agent_id if ctx is not None else None,
            "session_id": ctx.session_id if ctx is not None else None,
            "entity_class": _derive_entity_class(tool_name, params),
        }

    def execute(self, action: dict[str, Any]) -> ToolOutput:
        """Execute a tool action and record the result."""
        tool_name = action.get("tool_name", "unknown")
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}

        result = self._executor.execute(action)

        # Detect FearAgent blocks from metadata
        metadata = getattr(result, "metadata", None) or {}
        is_blocked = metadata.get("fear_agent_blocked", False)

        self._sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name=tool_name,
                tool_args=params,
                result_success=result.success,
                result_output=result.output,
                result_error=result.error,
                blocked=is_blocked,
                block_reason=result.error if is_blocked else None,
                **self._telemetry_fields(tool_name, params),
            )
        )

        return result

    def record_block(self, tool_name: str, reason: str, params: dict[str, Any] | None = None) -> None:
        """Record that an action was blocked (e.g., by FearAgent or autonomy)."""
        params = params or {}
        self._sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name=tool_name,
                tool_args=params,
                blocked=True,
                block_reason=reason,
                **self._telemetry_fields(tool_name, params),
            )
        )

    # Forward all other attributes to the wrapped executor
    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)
