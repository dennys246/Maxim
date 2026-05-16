"""ActionSink protocol and implementations for capturing tool outputs.

The ActionSink captures every tool execution (including FearAgent blocks)
for post-run validation in scenario testing.

Adapter contract (CC8, 2026-04-29 audit)
---------------------------------------

The protocol's two members — ``record(action)`` and the ``actions``
property — are the public adapter contract. A non-sim adapter (e.g. a
Minecraft/Mineflayer integration that wants to mirror agent actions
into the world) can implement these without orchestrator, scenario, or
narrative-phase coupling.

What the protocol does NOT assume:

- **No orchestrator dependency.** The executor calls ``record()`` after
  every tool execution; the sink decides what to do with the record
  (store, forward, drop). Validation/scenario logic lives elsewhere.
- **No conversational-turn assumption.** Records arrive as the executor
  produces them — possibly clustered, possibly sparse. The protocol
  does not promise turn alignment.
- **No tool-registry coupling.** ``ActionRecord.tool_name`` is an
  opaque string from the sink's perspective; the sink does not need
  access to the registry.

Sink implementations may freely impose additional structure (e.g.
``RecordingSink``'s 10k-action ring buffer with compression) — but the
protocol minimum is just the two members below.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ActionRecord:
    """Captured output action from the agent pipeline.

    SHAPE-FROZEN at 1.0 (CC3) — appending optional fields at the end
    with sensible defaults is the only allowed evolution. Required
    fields, type changes, and field reorderings are major-version
    bumps. The fields are observability-focused; no isolation-hygiene
    rule applies, so an ``extra`` escape hatch is unnecessary —
    add purpose-specific fields here as new telemetry needs surface.

    Stage 0b additions (release_0_9_1.md): ``agent_id`` /
    ``session_id`` thread through from the ``RequestContext`` ContextVar
    bound at the sim orchestrator entry; ``entity_class`` carries the
    target entity classification (food, weapon, body-part, etc.) for
    pain-aversion exposure normalization in Roy-3 analysis. All three
    are optional ``None`` defaults so existing test fixtures and
    third-party ``ActionSink`` implementations keep working.
    """

    timestamp: float
    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    result_success: bool = False
    result_output: Any = None
    result_error: str | None = None
    blocked: bool = False
    block_reason: str | None = None
    # Stage 0b (release_0_9_1.md): per-record agent + session attribution.
    # Populated by InstrumentedExecutor from utils/http.py::current_context().
    agent_id: str | None = None
    session_id: str | None = None
    # Stage 0b: entity classification for exposure-count normalization.
    # Best-effort — derived from tool_args target where present.
    entity_class: str | None = None


@runtime_checkable
class ActionSink(Protocol):
    """Captures tool outputs and motor commands."""

    def record(self, action: ActionRecord) -> None:
        """Record an executed action."""
        ...

    @property
    def actions(self) -> list[ActionRecord]:
        """All recorded actions in order."""
        ...


class RecordingSink:
    """Stores all actions for post-run validation.

    Bounded to MAX_ACTIONS. When the limit is reached, the oldest half of
    actions are compressed (detail fields stripped, only tool_name + success
    + timestamp retained) to free memory while preserving the audit trail.
    """

    MAX_ACTIONS = 10_000

    def __init__(self, max_actions: int | None = None) -> None:
        self._actions: list[ActionRecord] = []
        self._lock = threading.Lock()
        self._max = max_actions or self.MAX_ACTIONS

    def record(self, action: ActionRecord) -> None:
        with self._lock:
            self._actions.append(action)
            if len(self._actions) > self._max:
                self._compress_oldest()

    def _compress_oldest(self) -> None:
        """Compress the oldest half of actions — strip heavy fields, keep summary.

        Called under self._lock. Replaces full ActionRecords with lightweight
        copies (tool_args and result_output cleared) to free memory while
        preserving the audit trail of what tools ran and whether they succeeded.
        """
        half = len(self._actions) // 2
        compressed = []
        for rec in self._actions[:half]:
            compressed.append(
                ActionRecord(
                    timestamp=rec.timestamp,
                    tool_name=rec.tool_name,
                    tool_args={},  # Drop heavy args
                    result_success=rec.result_success,
                    result_output=None,  # Drop heavy output
                    result_error=rec.result_error[:100] if rec.result_error else None,
                    blocked=rec.blocked,
                    block_reason=rec.block_reason,
                    # Stage 0b telemetry fields are kept through compression —
                    # they're tiny and the whole point of 0b is post-hoc
                    # attribution analysis across the full action stream,
                    # which would break if compressed records dropped them.
                    agent_id=rec.agent_id,
                    session_id=rec.session_id,
                    entity_class=rec.entity_class,
                )
            )
        self._actions = compressed + self._actions[half:]

    @property
    def actions(self) -> list[ActionRecord]:
        with self._lock:
            return list(self._actions)

    def find_blocked(self, tool_pattern: str | None = None, reason_contains: str | None = None) -> list[ActionRecord]:
        """Find actions that were blocked by FearAgent."""
        import re

        results = []
        for action in self.actions:
            if not action.blocked:
                continue
            if tool_pattern and not re.search(tool_pattern, action.tool_name):
                continue
            if reason_contains and (
                action.block_reason is None or reason_contains.lower() not in action.block_reason.lower()
            ):
                continue
            results.append(action)
        return results

    def record_noop(self, reason: str = "") -> None:
        """Record a deliberation NoOp marker (Stage 4, F2 fix).

        The bridge watches ``len(self.actions)`` for growth.  A NoOp
        deliberation tick records this marker so the bridge sees
        activity and doesn't timeout.  The bridge's text extraction
        only looks for "respond"/"speak" tool_name, so this marker
        doesn't produce response text.
        """
        import time as _time

        self.record(
            ActionRecord(
                timestamp=_time.time(),
                tool_name="_deliberation_noop",
                result_success=True,
                result_output=reason or "deliberation tick — no externalized output",
            )
        )

    def find_actions(self, tool: str | None = None, output_matches: str | None = None) -> list[ActionRecord]:
        """Find actions matching criteria."""
        import re

        results = []
        for action in self.actions:
            if action.blocked:
                continue
            if tool and action.tool_name != tool:
                continue
            if output_matches:
                output_str = str(action.result_output or "")
                if not re.search(output_matches, output_str, re.IGNORECASE):
                    continue
            results.append(action)
        return results
