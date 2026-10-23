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

import dataclasses
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

    Per-action credit (``rpe``, ``pain``, ``nac_failure_booked``): copied
    by ``InstrumentedExecutor`` off the ``ToolOutput`` the executor stamped
    (the surprise, the pain, and the signature the tool-pain bridge booked a
    failure under), so ``actions.jsonl`` 1.2 and the console's run history can
    show what each action was credited with. ``None`` on synthetic records
    (blocks, markers) and for actions that carried none.

    ``utterance`` (#1261): the words a speech tool delivered, filled by
    ``InstrumentedExecutor`` from the call's own argument named by the tool's
    declared ``Tool.speech_params`` (``respond``, ``speak``, ``say``, and an
    affordance whose YAML declares ``speech_param``), when the call succeeded. ``result_output`` is
    the tool's RECEIPT (``RespondTool`` returns ``{"delivered": True, "mode":
    "cli"}``), so every reader that wants what the agent SAID (the bridge's turn
    ``response``, ``report.response_texts``, ``analyze_results``) reads this
    field, never ``result_output`` and never a list of speech-tool names
    (mechanization backlog M54 in ``docs/plans/outstanding.md``). ``None`` for
    non-speech tools, failed or blocked calls, and markers. Not written to
    ``actions.jsonl``: ``params`` already holds the message.
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
    # Per-action credit, from the executor's ToolOutput stamp (actions.jsonl 1.2).
    rpe: float | None = None
    pain: float | None = None
    nac_failure_booked: str | None = None
    # #1261: the words a successful speech call delivered (InstrumentedExecutor).
    utterance: str | None = None


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

    @classmethod
    def from_serialized(cls, actions: list[dict[str, Any]]) -> RecordingSink:
        """A sink holding the action history a finished run serialized (``SimulationResult.actions``), so
        post-run validation sees what the run did (#1310). Keys that are not ``ActionRecord`` fields are
        dropped; a record without ``timestamp`` or ``tool_name`` is a ``KeyError``, never silently skipped."""
        names = {f.name for f in dataclasses.fields(ActionRecord)}
        sink = cls(max_actions=max(len(actions), 1))
        for a in actions:
            sink.record(
                ActionRecord(
                    timestamp=float(a["timestamp"]),
                    tool_name=str(a["tool_name"]),
                    **{k: v for k, v in a.items() if k in names and k not in ("timestamp", "tool_name")},
                )
            )
        return sink

    def record(self, action: ActionRecord) -> None:
        with self._lock:
            self._actions.append(action)
            if len(self._actions) > self._max:
                self._compress_oldest()

    def _compress_oldest(self) -> None:
        """Compress the oldest half of actions — strip heavy fields, keep summary.

        Called under self._lock. Replaces full ActionRecords with lightweight
        copies (tool_args, result_output and utterance cleared) to free memory
        while preserving the audit trail of what tools ran and whether they
        succeeded.
        """
        half = len(self._actions) // 2
        compressed = []
        for rec in self._actions[:half]:
            # ``replace`` keeps every other field, so a field added later survives compression unless it is
            # dropped here. Kept: the Stage 0b telemetry (tiny, and post-hoc attribution reads it across the
            # whole stream) and the per-action credit (the run history reads it per action).
            compressed.append(
                dataclasses.replace(
                    rec,
                    tool_args={},  # Drop heavy args
                    result_output=None,  # Drop heavy output
                    result_error=rec.result_error[:100] if rec.result_error else None,
                    utterance=None,  # The words are the heavy half of the args they came from: dropped with them.
                )
            )
        self._actions = compressed + self._actions[half:]

    @property
    def actions(self) -> list[ActionRecord]:
        with self._lock:
            return list(self._actions)

    def find_blocked(self, tool_pattern: str | None = None, reason_contains: str | None = None) -> list[ActionRecord]:
        """Find blocked actions: by FearAgent, or by autonomy at a turn's rejection cap (#1413, ``block_reason``
        "Rejected by autonomy: ..."). ``reason_contains`` tells them apart."""
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
        reads only ``utterance``, which a marker never carries, so this
        marker doesn't produce response text.
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
