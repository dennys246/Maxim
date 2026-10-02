"""Cross-component in-flight LLM call registry.

Single source of truth for "is there an LLM call in flight right now?"
across the codebase. One consumer queries it today (the others are planned):
- :mod:`maxim.simulation.orchestrator` :func:`_stall_detector` — suppresses
  spurious "REPEATED STALL" nudges during legitimate inference (the bug
  this module exists to close — see [docs/plans/deferred/stall_detector_timeout_awareness.md]).
- :class:`maxim.runtime.heartbeat.HeartbeatMonitor` (Stage 2, pending) —
  will suppress agent-loop-idle warnings when an LLM call is the reason
  for silence. Tracked in the same plan doc.
- (planned, not wired) operator diagnostics: no ``maxim doctor`` row reads
  the registry today.

The registered ``tier`` is the request's LANE (``PLANNING_LANE`` for every
planning call — ``maxim.agents.llm_types.PLANNING_LANE``), and each entry records whether the call STREAMS: only a
streaming call reports bytes, so only it can be judged byte-silent (#1042).

The instrumentation site is
:meth:`maxim.models.language.router.LLMRouter._complete_text_locked` —
ONE :func:`register_call_start` at dispatch entry, ONE
:func:`register_call_end` in ``try/finally`` at dispatch exit. The wrap
covers the provider-fallback loop so the registry sees one continuous
in-flight window across provider retries. Wrapping at the per-backend
layer (``complete_with_usage``) is **incorrect** — it creates a gap
between provider-A end and provider-B start during which
:func:`any_call_in_flight` returns False, allowing spurious stall nudges
to fire mid-failover.

Byte-arrival updates flow from per-backend ``_stream_response`` chunk
loops via :func:`register_byte_received` reading the active call_id from
a :class:`contextvars.ContextVar`. ``LLMWorker._call_llm_with_timeout``
already uses ``contextvars.copy_context()`` before dispatching to the
worker pool, so the ContextVar set in :func:`register_call_start`
propagates into the same worker thread that runs the stream loop. PR
#320's TTFT keepalive frames (``: keepalive\\n\\n``) arrive as
bytes-on-wire and are tracked by the same path, letting the stall
detector distinguish "call alive but slow" from "connection wedged."

**CLAUDE.md "Mutable globals + module extraction" lesson:** consumers
MUST import this module (``from maxim.runtime import llm_call_registry``)
or use the published functions; do NOT re-import ``_registry`` or
``_active_call_id`` by name from another module. Python binds those names
by value at import time and the two namespaces will diverge silently.
The published surface (``register_call_start`` / ``register_call_end`` /
``register_byte_received`` / ``any_call_in_flight`` /
``oldest_byte_silence_s`` / ``current_call_id``) is the only safe
read-path.

**[engineering] SHAPE-FROZEN at 1.0 (CC3):** :class:`_InFlightCall` is
shape-frozen. An ``extra: dict`` escape hatch was rejected because this
dataclass is the registry's internal value type with no caller-extensible
metadata; new fields must widen the registry's published API (snapshot
or query signatures), not slip in via ``extra``. Adding optional fields
with defaults at the end is non-breaking. Adding required fields or
reordering existing fields is a major-version bump for downstream
consumers of the snapshot API.
"""

from __future__ import annotations

import contextvars
import dataclasses
import threading
import time
import uuid
from dataclasses import dataclass

__all__ = [
    "non_streaming_calls",
    "any_call_in_flight",
    "current_call_id",
    "oldest_byte_silence_s",
    "register_byte_received",
    "register_call_end",
    "register_call_start",
]


@dataclass(frozen=True)
class _InFlightCall:
    """SHAPE-FROZEN at 1.0 (CC3): registry value type.

    Fields are read positionally by tests and by snapshot helpers; adding
    new fields at the end with defaults is the only safe expansion.
    """

    call_id: str
    tier: str  # the request's LANE ("large"/"medium"/"small"/"unknown") — never a cost-budget tier
    started_at: float
    last_byte_at: float
    # Only a STREAMING call reports bytes (register_byte_received from the stream loops): a non-streaming call's
    # "byte silence" would just be its age, so it never feeds the wedged-connection verdict (#1042).
    streams: bool = False
    # The time this call was ALLOWED (the worker's effective timeout, a per-request override included); ``None`` when
    # the caller did not say. A non-streaming call is judged wedged by age against it (#1042).
    allowed_s: float | None = None


_registry: dict[str, _InFlightCall] = {}
_lock = threading.RLock()

# Per-call ContextVar Token saved at register_call_start so register_call_end
# can ``reset()`` back to the parent dispatch's call_id rather than blindly
# clearing. Keyed by call_id. Stack semantics prevent nested dispatches from
# stranding the outer call's ContextVar (the architecture review's C2 bug).
_entry_tokens: dict[str, contextvars.Token] = {}

# Per-thread / per-async-context active call id. Set by ``register_call_start``
# and consumed by per-backend stream loops via ``current_call_id()`` so the
# byte-arrival instrumentation doesn't need to thread call_id through every
# backend method signature. ContextVar inherits across asyncio tasks and is
# thread-local under threading — both correct for our usage (router lock
# serializes per-router, but multiple router instances can coexist).
_active_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("maxim_active_llm_call_id", default=None)


def current_call_id() -> str | None:
    """Return the active call_id from contextvar, or None if no LLM call
    is in flight on the current thread / async task."""
    return _active_call_id.get()


# Defense against SIGKILL / unwrapped exit paths that skip register_call_end:
# entries older than this auto-expire on read. Generous floor (covers
# worst-case big-model timeouts at 600s + 2x safety) so legitimate
# in-flight calls never get filtered out by the staleness check.
_STALE_ENTRY_TTL_S = 1800.0


def register_call_start(*, tier: str, streams: bool = False, allowed_s: float | None = None) -> str:
    """Register the start of an LLM dispatch.

    Sets ``_active_call_id`` ContextVar so per-backend stream loops can
    call :func:`register_byte_received` without threading call_id through
    every signature. Returns the ``call_id`` to be passed to
    :func:`register_call_end`. Always wrap in ``try/finally`` so
    :func:`register_call_end` runs even on exception paths:

    .. code-block:: python

        call_id = register_call_start(tier="large")
        try:
            return self._dispatch_inner(...)
        finally:
            register_call_end(call_id)

    **Nested-dispatch safety:** stores the previous ContextVar Token on
    the registry entry, so :func:`register_call_end` can ``.reset()`` to
    the parent dispatch's call_id rather than blindly clearing. Without
    this, an inner dispatch ending before the outer one would strand the
    outer's ``register_byte_received`` calls as silent no-ops.
    """
    cid = str(uuid.uuid4())
    now = time.time()
    # Set ContextVar FIRST so a query thread that catches the registry
    # mid-mutation never sees a registered call without an active cid.
    token = _active_call_id.set(cid)
    _entry_tokens[cid] = token  # remembered for stack-style reset on end
    with _lock:
        _registry[cid] = _InFlightCall(
            call_id=cid, tier=tier, started_at=now, last_byte_at=now, streams=streams, allowed_s=allowed_s
        )
    return cid


def register_call_end(call_id: str) -> None:
    """Remove an entry from the registry and reset the ContextVar to the
    parent dispatch's call_id (or None if this was the outermost).

    Silently no-ops if the ``call_id`` was already removed (e.g., by
    staleness GC on a long-running but crashed registration).
    """
    with _lock:
        _registry.pop(call_id, None)
    # Reset the ContextVar via the saved Token — restores the parent
    # dispatch's call_id (or None if this was the outermost). Without the
    # stack semantic, an inner end would clear the contextvar and the
    # outer dispatch's stream loop would silently no-op on byte arrivals.
    token = _entry_tokens.pop(call_id, None)
    if token is not None:
        try:
            _active_call_id.reset(token)
        except ValueError:
            # Token was from a different Context; happens if the call
            # was registered on a different thread/task than this end.
            # Safe to fall through — contextvar was thread-local on that
            # other thread and we don't need to touch ours.
            pass


def register_byte_received(call_id: str | None = None) -> None:
    """Update ``last_byte_at`` for the given call (or the active call from
    contextvar if ``call_id`` is None).

    Called from per-backend streaming chunk loops including chunks that
    are PR #320 TTFT keepalive frames (``: keepalive\\n\\n``). Silently
    no-ops if no call_id is resolvable (e.g., call already ended but a
    late chunk arrived; harmless).
    """
    cid = call_id if call_id is not None else _active_call_id.get()
    if cid is None:
        return
    now = time.time()
    with _lock:
        cur = _registry.get(cid)
        if cur is None:
            return
        _registry[cid] = dataclasses.replace(cur, last_byte_at=now)  # every other field carried (streams incl.)


def _live_entries_snapshot(now: float) -> list[_InFlightCall]:
    """Snapshot live entries (filtered by staleness) without holding the
    lock through caller logic. Internal helper."""
    with _lock:
        return [v for v in _registry.values() if now - v.started_at < _STALE_ENTRY_TTL_S]


def any_call_in_flight(*, tier: str | None = None) -> bool:
    """True if any non-stale LLM call is in flight.

    When ``tier`` is provided, filters to that lane tier only — the
    orchestrator's stall detector uses this to suppress nudges only when
    a call is in flight at the same tier the orchestrator's own calls
    use (resolved via :func:`maxim.runtime.function_router.FunctionRouter.resolve`).
    """
    now = time.time()
    for v in _live_entries_snapshot(now):
        if tier is None or v.tier == tier:
            return True
    return False


def oldest_byte_silence_s(*, tier: str | None = None) -> float | None:
    """Seconds since the most recent byte across in-flight calls.

    Used by the stall detector's wedged-connection safety net: distinguish
    "call is alive, bytes flowing (real tokens or keepalives)" from "call
    registered but no bytes for >N seconds → connection is wedged."

    Returns ``None`` when no in-flight call matches. Returns the
    *minimum* byte-silence across the filtered set (the call with the
    most recent activity — if ANY call is recently active, the orchestrator
    should keep suppressing).

    Only STREAMING calls count: a non-streaming call reports no bytes, so its silence is unknown, never evidence of
    a wedged connection (#1042). ``None`` also when only non-streaming calls are in flight.
    """
    now = time.time()
    candidates = [
        now - v.last_byte_at for v in _live_entries_snapshot(now) if v.streams and (tier is None or v.tier == tier)
    ]
    return min(candidates) if candidates else None


def non_streaming_calls(*, tier: str | None = None) -> list[tuple[float, float | None]]:
    """``(age_s, allowed_s)`` of every NON-streaming call in flight. Such a call reports no bytes, so its age —
    against the time it was allowed — is the only evidence it has: the stall detector judges it wedged once it has
    outlived that time (#1042)."""
    now = time.time()
    return [
        (now - v.started_at, v.allowed_s)
        for v in _live_entries_snapshot(now)
        if not v.streams and (tier is None or v.tier == tier)
    ]


def _reset_for_tests() -> None:
    """Test-only: clear the registry, entry-token map, and active call_id.
    NOT public API."""
    with _lock:
        _registry.clear()
    _entry_tokens.clear()
    _active_call_id.set(None)
