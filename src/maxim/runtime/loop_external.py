"""Inbound comms reach the live prompt, framed (#1481; design: docs/plans/inbound_comms_1481_approach.md).

A message sent to Maxim over SMS, voice or ``POST /api/command`` is published on the agent's bus as a ``comms:*``
``Percept``. The live loop reads ``InMemoryMemory``, which is not on that bus, so before this module the message was
accepted, logged and never shown to the model. Here:

- :class:`ExternalInbox` holds those messages: bounded, de-duplicated by Twilio ``MessageSid``, the sender replaced by
  a per-session alias (owner decision 2026-10-28: no phone number ever reaches a prompt or a log).
- :func:`inbound_comms_subscription` subscribes it on the loop agent's bus for exactly the run's lifetime. The
  wrapper enters it inside the run's session claim (A's grant; H's bounded-stop PR 0).
- The inbox is the ``InboundFeed`` the request carries. The prompt builder reads :meth:`ExternalInbox.held` when it
  BUILDS the prompt, records which messages survived the fit whole, and dispatch acknowledges them only when the
  model's turn is ACTED ON (``tool_dispatch.execute_and_learn``). A message leaves only then, or after
  :data:`RENDER_LIMIT` turns showed it without acting (logged).
- :func:`submit_tick_context` is the loop's submit block, moved here whole (A's grant: the ``Try`` that submits and
  releases the carried inputs in its ``finally`` moves, it never splits), with the feed attached.

A leaf: it never imports ``agent_loop`` (the loop's import-direction rule).
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# How many messages are held at once; beyond it the oldest is dropped, with a WARNING (owner decision 2026-10-28).
MAX_HELD = 20
# Characters kept per message (the prompt builder also caps the section in tokens).
MAX_CHARS = 2000
# Turns that may show a message without acting on it before it is evicted (delta1 N3): a message whose content keeps
# the model from producing a usable reply cannot recur forever.
RENDER_LIMIT = 3
# Twilio MessageSids remembered for de-duplication of webhook retries.
_SEEN_SIDS = 200
# What the frame rule says the inbound frames hold (owner decision 2026-10-28: /api/command is untrusted too).


@dataclass
class InboundEntry:
    """One held message. ``alias`` stands in for the sender; the raw sender never leaves the inbox."""

    id: int
    alias: str
    channel: str
    content: str
    received_at: float
    renders: int = field(default=0)


class ExternalInbox:
    """The loop's held inbound comms. Every access holds ``_lock``; the bus handler does O(1) work under it and never
    publishes or logs through the bus while holding it (delta C1)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: deque[InboundEntry] = deque()
        self._ids = itertools.count(1)
        self._aliases: dict[str, str] = {}
        self._sids: deque[str] = deque(maxlen=_SEEN_SIDS)
        self._closed = False

    # -- intake (the bus handler; runs on the publisher's thread) ---------------------------------------------------

    def on_percept(self, percept: Any) -> None:
        """Keep a ``comms:*`` percept. Never raises: ``AgentBus.publish`` logs a handler's exception only at DEBUG,
        so a raise here would lose the message invisibly. Malformed input is a validated early return, logged."""
        source = getattr(percept, "source", None)
        if not isinstance(source, str) or not source.startswith("comms:"):
            return
        channel = source[len("comms:") :] or "unknown"
        content = getattr(percept, "content", None)
        if not isinstance(content, str) or not content.strip():
            logger.warning("Inbound %s message ignored: no text content", channel)
            return
        context = getattr(percept, "context", None)
        sender = str(getattr(context, "sender", None) or "unknown")
        metadata = getattr(percept, "metadata", None)
        sid = metadata.get("message_sid") if isinstance(metadata, dict) else None
        if len(content) > MAX_CHARS:
            content = f"{content[:MAX_CHARS]}\n[... truncated {len(content) - MAX_CHARS} chars]"
        dropped: InboundEntry | None = None
        with self._lock:
            if self._closed:
                closed = True
            else:
                closed = False
                if isinstance(sid, str) and sid:
                    if sid in self._sids:
                        return  # a Twilio webhook retry of a message already held or shown
                    self._sids.append(sid)
                alias = self._aliases.setdefault(sender, f"sender {len(self._aliases) + 1}")
                self._held.append(InboundEntry(next(self._ids), alias, channel, content, time.time()))
                if len(self._held) > MAX_HELD:
                    dropped = self._held.popleft()
        if closed:
            logger.warning("Inbound %s message arrived after its loop ended; not shown", channel)
        elif dropped is not None:
            logger.warning(
                "Inbound message from %s (%s) dropped unshown: more than %d held",
                dropped.alias,
                dropped.channel,
                MAX_HELD,
            )

    # -- the InboundFeed the request carries ------------------------------------------------------------------------

    def held(self) -> list[InboundEntry]:
        """The messages held NOW, oldest first (read when a prompt is built, never at submit: delta1 N4)."""
        with self._lock:
            return [InboundEntry(e.id, e.alias, e.channel, e.content, e.received_at, e.renders) for e in self._held]

    def note_rendered(self, ids: list[int]) -> None:
        """One turn showed these messages whole. After :data:`RENDER_LIMIT` such turns without acting on them, a
        message is evicted, logged (the WARNING is emitted outside the lock: delta2 P6)."""
        if not ids:
            return
        wanted = set(ids)
        evicted: list[InboundEntry] = []
        with self._lock:
            for entry in list(self._held):
                if entry.id in wanted:
                    entry.renders += 1
                    if entry.renders >= RENDER_LIMIT:
                        self._held.remove(entry)
                        evicted.append(entry)
        for entry in evicted:
            logger.warning(
                "Inbound message from %s (%s) evicted: shown in %d turns, none acted on",
                entry.alias,
                entry.channel,
                RENDER_LIMIT,
            )

    def ack(self, ids: list[int]) -> None:
        """A turn that showed these messages whole was acted on: they leave the inbox. Idempotent."""
        if not ids:
            return
        wanted = set(ids)
        with self._lock:
            kept = [e for e in self._held if e.id not in wanted]
            self._held.clear()
            self._held.extend(kept)

    def continuation(self) -> _ContinuationFeed:
        """The feed a deliberation cycle 2+ carries (A's grant 4): the same held messages and acknowledgement, but it
        does not count a render, so :data:`RENDER_LIMIT` counts turns, never cycles."""
        return _ContinuationFeed(self)

    def close(self) -> None:
        with self._lock:
            self._closed = True


class _ContinuationFeed:
    def __init__(self, inbox: ExternalInbox) -> None:
        self._inbox = inbox

    def held(self) -> list[InboundEntry]:
        return self._inbox.held()

    def note_rendered(self, ids: list[int]) -> None:
        return None  # a cycle of the same turn: the turn already counted

    def ack(self, ids: list[int]) -> None:
        self._inbox.ack(ids)

    def continuation(self) -> _ContinuationFeed:
        return self


@contextlib.contextmanager
def inbound_comms_subscription(agent: Any) -> Iterator[ExternalInbox | None]:
    """Hold the run's inbound comms for exactly the run's lifetime. Subscribes on ``agent._bus`` (the bus the comms
    stack publishes on: ``bootstrap.build_comms_stack(agent=)`` reads the same attribute), yields the inbox, and on
    every exit unsubscribes and closes it. ``__exit__`` runs inside the stop budget (bounded_stop.md §4): it never
    blocks and never raises over an in-flight exception. An agent without a bus yields None."""
    from maxim.agents.bus import Percept

    bus = getattr(agent, "_bus", None)
    if bus is None or not callable(getattr(bus, "subscribe", None)) or not callable(getattr(bus, "unsubscribe", None)):
        # Checked BEFORE subscribing: a bus that cannot unsubscribe would raise at exit, over the run's own exception.
        yield None
        return
    inbox = ExternalInbox()
    bus.subscribe(Percept, inbox.on_percept)
    try:
        yield inbox
    finally:
        inbox.close()  # sets a flag under the lock: cannot raise
        try:
            bus.unsubscribe(Percept, inbox.on_percept)
        except Exception as exc:  # noqa: BLE001 - never raise over the run's own exit (bounded_stop.md §4)
            logger.warning("Inbound comms subscription not removed at run end: %s", exc)


def _release_unsent_deferred_inputs(processed_cli_inputs: Any, deferred: list[str], submitted: bool) -> None:
    """A submit that failed (queue full, or raised) must not lose the inputs it carried: they become unprocessed
    again and ride the next submit."""
    if submitted:
        return
    for c in deferred:
        try:
            processed_cli_inputs.remove(c)
        except ValueError:
            continue  # already evicted from the bounded deque: it reads as unprocessed anyway


def planning_submit_shared_kwargs(
    *,
    mode_info: Any,
    internet_access: bool,
    internet_policy_summary: str,
    available_tools: Any,
    tool_descriptions: Any,
    context_pool_text: str,
    agent_states: Any,
    recent_outcomes: Any,
    use_tool_prompting: bool,
    conversation_history_text: str,
    is_sleeping: bool,
    protocol_context: str,
) -> dict[str, Any]:
    """The ``submit_context`` arguments the loop's §6 submit and its deliberation cycles 2+ pass IDENTICALLY (A's grant,
    Rule T option 1). The two submits must stay in step: a value that must be re-read per call (``autonomy_level``,
    which can change while deliberation waits) or that differs between them (the triggering input, the carried inputs,
    the feed, ``deliberation_available``, ``failed_tools``, prefetch) stays explicit at each call site, never here."""
    return {
        "mode": mode_info,
        "internet_access": internet_access,
        "internet_policy_summary": internet_policy_summary,
        "available_tools": available_tools,
        "tool_descriptions": tool_descriptions,
        "context_pool_text": context_pool_text,
        "agent_states": agent_states,
        "recent_outcomes": recent_outcomes,
        "use_tool_prompting": use_tool_prompting,
        "conversation_history_text": conversation_history_text,
        "is_sleeping": is_sleeping,
        "protocol_context": protocol_context,
    }


def submit_tick_context(
    llm_worker: Any,
    *,
    context: Any,
    shared: dict[str, Any],
    autonomy_level: Any,
    _failed_tools: Any,
    triggering_input: str,
    pending_modification: Any,
    prefetch_context_text: str,
    skip_exploration: bool,
    deliberation_available: bool,
    processed_cli_inputs: Any,
    _deferred_inputs: list[str],
    inbound_feed: Any,
) -> bool:
    """The loop's §6 submit (moved whole from ``_run_agentic_loop_body``, A's grant 1), with the run's inbound feed
    attached (#1481). The ``finally`` stays the one place that releases the carried inputs; the inbound messages are
    never released here: they are acknowledged at dispatch, when the turn is acted on."""
    submitted = False
    try:
        submitted = llm_worker.submit_context(
            context=context,
            autonomy_level=autonomy_level,
            failed_tools=_failed_tools,
            triggering_input=triggering_input,
            pending_modification=pending_modification,
            prefetch_context=prefetch_context_text,
            skip_exploration=skip_exploration,
            deliberation_available=deliberation_available,
            deferred_inputs=_deferred_inputs,
            inbound_feed=inbound_feed,
            **shared,
        )
    finally:  # a refused or raising submit must not lose the inputs it carried
        _release_unsent_deferred_inputs(processed_cli_inputs, _deferred_inputs, submitted)
    return submitted
