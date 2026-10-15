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

    **DO NOT consume this field from any substrate write path** (NAc,
    EC, ATL, Hippocampus, PainBus). It exists for Roy-3 post-hoc
    exposure-count normalization and the Roy harness's per-class
    plotting. Substrate consumers must derive entity identity from
    the percept text + EC pattern completion, NEVER from this field.
    The bio-fidelity guardrail in the bio-lens review: this field is
    walled off from the substrate so it can stay a best-effort
    heuristic without contaminating the 1.0 thesis ("substrate carries
    cognition; language is I/O").

    **Strict opt-in derivation:** ships explicit-param-only at 0.9.1
    after the pre-merge review caught the verb-strip heuristic
    producing noisy buckets on non-entity tools (``get_status`` →
    ``"status"``, ``set_entity_sensor`` → ``"entity_sensor"``,
    ``do_something_clever`` → ``"something_clever"``). Roy-3
    normalization explicitly skips ``None``, so being conservative is
    strictly safer than producing wrong buckets — silent miscount is
    worse than missing data.

    Heuristics in priority order:
    1. ``params["entity_class"]`` — explicit caller override.
    2. ``params["target"]`` / ``params["entity"]`` / ``params["object"]`` —
       the conventional param names entity-binding tools use.

    Returns ``None`` when neither (1) nor (2) is present, including
    for tools whose name suggests an entity binding but didn't pass
    one through params (``infant_humanoid_pick_up`` with no target →
    None). The field is best-effort metadata.

    TODO (1.1): replace this opt-in heuristic with a declared
    ``Tool.entity_class: str | None`` field on the Tool ABC, so tool
    authors can opt their tools into Roy-3 attribution explicitly
    without participating in this derivation logic at all. Tracks
    the same surface as ``feedback_two_identity_schemes.md`` — the
    substrate already uses tool-name AND EC-cluster identity for one
    concept; declared ``entity_class`` would be a third explicit
    handle that tooling can rely on.
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
    # No verb-strip path: pre-merge review showed it produced noise
    # on non-entity tools that Roy-3 normalization would silently
    # mis-attribute. Future work tracked in the docstring TODO.
    return None


def _utterance(speech_params: tuple[str, ...], params: dict[str, Any], success: bool) -> str | None:
    """The words a speech call delivered: the first non-empty of its tool's ``speech_params`` when the call
    succeeded, else ``None``.

    ``speech_params`` is ``Tool.speech_params`` of the tool the call RAN (#1261), so an aliased call is judged
    as that tool, and a tool that is not speech declares ``()``. A failed call said nothing (``RespondTool``
    refuses a blank message), so it is ``None`` too.
    """
    if not speech_params or not success:
        return None
    for key in speech_params:
        value = params.get(key)
        if value:
            return str(value)
    return None


class InstrumentedExecutor:
    """Wraps an Executor to record all actions to an ActionSink.

    Drop-in replacement for Executor — same execute() interface.
    All calls are forwarded to the wrapped executor, and results
    are recorded in the sink.

    The wrapped executor must provide ``resolve_tool_name`` and ``registry`` (directly, or forwarded by a
    wrapper's ``__getattr__``): the recorder asks them which tool a call runs and whether it is speech (#1261).

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

    def _speech_params(self, tool_name: Any) -> tuple[str, ...]:
        """``Tool.speech_params`` of the tool a call named *tool_name* RUNS (#1261), ``()`` when that is no tool.

        The executor's own alias resolution names it (``talk`` runs as ``say``, so its words are speech), and
        the executor's registry holds it; ``tool_name`` on the record still keeps the raw name. A non-string
        name is refused by dispatch as an invalid action, and an unregistered one by dispatch too: neither ran,
        so neither is speech.
        """
        if not isinstance(tool_name, str):
            return ()
        # One lookup (``find``), not a membership check then ``get``: the DM thread may deregister the tool in
        # between, and a ``KeyError`` here would lose the action before dispatch. A tool gone by now is no
        # speech; dispatch reports it.
        tool = self._executor.registry.find(self._executor.resolve_tool_name(tool_name))
        return tuple(tool.speech_params) if tool is not None else ()

    def execute(self, action: dict[str, Any]) -> ToolOutput:
        """Execute a tool action and record the result."""
        tool_name = action.get("tool_name", "unknown")
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
        # Asked BEFORE dispatch, which may deregister the tool (an entity released mid-call).
        speech_params = self._speech_params(tool_name)

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
                rpe=result.rpe,
                pain=result.pain,
                nac_failure_booked=result.nac_failure_booked,
                utterance=_utterance(speech_params, params, bool(result.success)),
                **self._telemetry_fields(tool_name, params),
            )
        )

        return result

    def record_block(self, tool_name: str, reason: str, params: dict[str, Any] | None = None) -> None:
        """Record that an action was blocked (e.g., by FearAgent or autonomy).

        No ``utterance``: a blocked speech call never delivered its words (#1261).
        """
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
