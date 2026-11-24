"""WorkingMemorySet: Exec-owned active-reference layer over recent cognitive events.

Replaces the scattered deque-backed surfaces (recent_percepts, recent_outcomes,
detected_speech, cli_inputs, conversation_history) with a single typed write
interface.  Parallel bio-system queries (knowledge_context, causal_context,
valence_context, motor_programs, body_state) are NOT part of this surface —
they stay in MemoryAgent._run_parallel_memory_queries().

PerceptTraceBuffer (Dormant since 2026-09-24; never constructed in production)
was meant to keep live-decay activations for NAc learning; NAc keeps its own
eligibility trace instead. WMS entries are point-in-time snapshots (F5).

Thread-safe: a threading.Lock serialises add(); queries snapshot the deque
under the lock.  Producers (MemoryAgent, PainBus, ReactionBus,
Hippocampus.recall, ThinkTool, …) call add() from different threads (F3); the
planned PTB producer was never wired.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from maxim.memory.text import Provenance, shown_text


class WorkingMemoryKind(str, Enum):
    """Kind of entry in the working-memory set."""

    PERCEPT = "percept"
    OUTCOME = "outcome"
    REACTION = "reaction"
    PAIN = "pain"
    ACTIVATION = "activation"  # planned PerceptTraceBuffer emission; never wired (the buffer is Dormant)
    THOUGHT = "thought"  # ThinkTool output
    RECALL = "recall"  # Hippocampus.recall() result
    CONVERSATION = "conversation"  # user/assistant turn


@dataclass(frozen=True, slots=True)
class WMEntry:
    """A single entry in the WorkingMemorySet.

    Frozen so entries are safe to hand out without copies.  ``content``
    may be a dict or string. It is NOT display-ready: it holds raw text, some of it from outside the agent, so a
    prompt shows an entry only through ``prompt_text()``, which renders it by its ``provenance`` (#1519).
    Runtime-only (never persisted, never sent), so the CC3 forward-compat rule does not apply.
    """

    kind: WorkingMemoryKind
    ref: str | None  # episode_id, percept_id, reaction_id, thought_id, or None
    content: Any  # dict or string payload (raw: render through prompt_text)
    timestamp: float
    tick: int  # monotonic add-order counter (for decay / ordering)
    salience: float = 0.0
    agent_id: str | None = None  # multi-agent scoping
    goal_tag: str | None = None  # active goal when this entry was created
    # Where the content came from, set at ``add`` (required there). EXTERNAL by default: fail-closed.
    provenance: Provenance = Provenance.EXTERNAL

    def prompt_text(self, limit: int = 100) -> str:
        """This entry as text a prompt may show ("" when it says nothing), by its provenance (#1519), cut to ``limit``.
        An OUTCOME shows its tool and status as the agent's own bookkeeping, and its error text and goal framed (tool
        and goal text can carry outside words); only an OUTCOME reads its keys that way, so another entry's
        ``action`` key is never shown as trusted structure. Any other entry shows its ``text``, ``content`` or a
        THOUGHT's ``enrichment`` (or the string itself) by ``provenance``: the operator's words defanged, outside
        text framed, the agent's own cut with no partial frame left."""
        content = self.content
        if isinstance(content, dict):
            tool = (
                (content.get("tool_name") or content.get("action", ""))
                if self.kind is WorkingMemoryKind.OUTCOME
                else ""
            )
            if tool:
                status = "succeeded"
                if not content.get("success", True):
                    error = shown_text(
                        str(content.get("error") or ""), Provenance.EXTERNAL, label="tool_error", limit=limit
                    )
                    status = f"failed: {error}" if error else "failed"
                goal = shown_text(str(content.get("goal") or ""), Provenance.EXTERNAL, label="goal", limit=limit)
                return f"{tool} {status}" + (f" (goal: {goal})" if goal else "")
            text = content.get("text") or content.get("content") or content.get("enrichment") or ""
            source = content.get("source")
        elif isinstance(content, str):
            text, source = content, None
        else:
            return ""
        label = f"{self.kind.value}_{source}" if isinstance(source, str) and source else self.kind.value
        return shown_text(str(text), self.provenance, label=label, limit=limit)


class WorkingMemorySet:
    """Exec-owned active-reference layer over recent cognitive events.

    Bounded deque with monotonic tick counter.  Agent-scoped.

    Single write interface: ``add()``.  All recent-context producers route
    through this method.  No code outside this module maintains a parallel
    buffer for prompt-assembly context.

    Construction order invariant: ExecAgent must be constructed BEFORE
    MemoryAgent in every entry path so that producers can receive the
    ``working_memory`` ref via dependency injection.
    """

    # EMA smoothing coefficient for valence signals.
    # ema = (1 - alpha) * ema + alpha * signal.value
    # 0.1 means recent signals shift the EMA gradually.
    VALENCE_EMA_ALPHA: float = 0.1

    def __init__(self, *, agent_id: str, capacity: int = 64) -> None:
        self._agent_id = agent_id
        self._capacity = capacity
        self._buf: deque[WMEntry] = deque(maxlen=capacity)
        self._tick = 0
        self._lock = threading.Lock()
        # Valence EMA keyed by WMEntry.tick (monotonic, never None).
        # Pruned on deque eviction to prevent unbounded growth.
        self._valence_ema: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Write interface
    # ------------------------------------------------------------------

    def add(
        self,
        kind: WorkingMemoryKind,
        *,
        ref: str | None = None,
        content: Any,
        salience: float = 0.0,
        agent_id: str | None = None,
        goal_tag: str | None = None,
        provenance: Provenance,
    ) -> WMEntry:
        """Append a new entry.  Thread-safe.

        ``provenance`` is REQUIRED (#1519): every producer states where its content came from, so forgetting is a
        ``TypeError``, never a raw outside text in a prompt (CLAUDE.md "push silent-no-op invariants into types").
        Returns the created entry (useful for testing / provenance).
        """
        if not isinstance(provenance, Provenance):
            raise TypeError(f"WorkingMemorySet.add: provenance must be a Provenance, not {provenance!r}")
        with self._lock:
            # Prune valence EMA for evicted entry (deque drops leftmost)
            if len(self._buf) == self._capacity:
                evicted = self._buf[0]
                self._valence_ema.pop(evicted.tick, None)

            entry = WMEntry(
                kind=kind,
                ref=ref,
                content=content,
                timestamp=time.time(),
                tick=self._tick,
                salience=salience,
                agent_id=agent_id or self._agent_id,
                goal_tag=goal_tag,
                provenance=provenance,
            )
            self._buf.append(entry)
            self._tick += 1
            return entry

    def receive_valence(self, tick: int, value: float) -> None:
        """Accumulate a valence signal on the entry at *tick*.

        Uses exponential moving average: ``ema = (1 - α) * ema + α * value``.
        Bounded in [-1, +1].  Thread-safe.

        Args:
            tick: The ``WMEntry.tick`` to modulate (monotonic, never None).
            value: Signal value in [-1.0, +1.0].  Source-blind — consumers
                do NOT need to know where it came from.
        """
        with self._lock:
            alpha = self.VALENCE_EMA_ALPHA
            current = self._valence_ema.get(tick, 0.0)
            updated = (1.0 - alpha) * current + alpha * value
            self._valence_ema[tick] = max(-1.0, min(1.0, updated))

    def get_effective_salience(self, entry: WMEntry) -> float:
        """Compute effective salience incorporating valence EMA.

        Positive EMA → boosts salience (reinforcement).
        Negative EMA → also boosts salience via abs() (survival — you
        want to REMEMBER what hurt you).
        """
        ema = self._valence_ema.get(entry.tick, 0.0)
        return entry.salience + abs(ema)

    # ------------------------------------------------------------------
    # Query interface (all return snapshots — safe outside the lock)
    # ------------------------------------------------------------------

    def recent(self, limit: int = 20) -> list[WMEntry]:
        """Most recent *limit* entries, oldest-first."""
        with self._lock:
            items = list(self._buf)
        return items[-limit:]

    def by_kind(self, kinds: set[WorkingMemoryKind], limit: int = 20) -> list[WMEntry]:
        """Entries matching any of *kinds*, most-recent first, capped at *limit*."""
        with self._lock:
            items = list(self._buf)
        matched = [e for e in reversed(items) if e.kind in kinds]
        return matched[:limit]

    def since_tick(self, tick: int) -> list[WMEntry]:
        """All entries added at or after *tick*, oldest-first."""
        with self._lock:
            items = list(self._buf)
        return [e for e in items if e.tick >= tick]

    def top_by_salience(
        self,
        kinds: set[WorkingMemoryKind],
        limit: int = 6,
        min_salience: float = 0.0,
    ) -> list[WMEntry]:
        """Entries matching *kinds* sorted by effective salience desc.

        Effective salience = base salience + abs(valence EMA).
        Returns up to *limit* entries with ``base salience >= min_salience``.
        Higher effective salience first; ties broken by higher tick (more recent).
        """
        with self._lock:
            items = list(self._buf)
            ema_snapshot = dict(self._valence_ema)

        def _effective(e: WMEntry) -> float:
            ema = ema_snapshot.get(e.tick, 0.0)
            return e.salience + abs(ema)

        matched = [e for e in items if e.kind in kinds and e.salience >= min_salience]
        matched.sort(key=lambda e: (_effective(e), e.tick), reverse=True)
        return matched[:limit]

    def for_agent(self, agent_id: str) -> list[WMEntry]:
        """Entries scoped to *agent_id*, oldest-first."""
        with self._lock:
            items = list(self._buf)
        return [e for e in items if e.agent_id == agent_id]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def current_tick(self) -> int:
        with self._lock:
            return self._tick

    def clear(self) -> None:
        """Drop all entries (session end)."""
        with self._lock:
            self._buf.clear()
            # tick is NOT reset — monotonic across clears
