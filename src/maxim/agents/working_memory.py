"""WorkingMemorySet: Exec-owned active-reference layer over recent cognitive events.

Replaces the scattered deque-backed surfaces (recent_percepts, recent_outcomes,
detected_speech, cli_inputs, conversation_history) with a single typed write
interface.  Parallel bio-system queries (knowledge_context, causal_context,
valence_context, motor_programs, body_state) are NOT part of this surface —
they stay in MemoryAgent._run_parallel_memory_queries().

PerceptTraceBuffer keeps its own internal ring buffer (NAc learning uses the
live-decay activations).  WMS entries are point-in-time snapshots — different
audiences, different semantics (F5).

Thread-safe: a threading.Lock serialises add(); queries snapshot the deque
under the lock.  9 producers (MemoryAgent, PainBus, ReactionBus, PTB,
Hippocampus.recall, ThinkTool, …) call add() from different threads (F3).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any


class WorkingMemoryKind(str, Enum):
    """Kind of entry in the working-memory set."""

    PERCEPT = "percept"
    OUTCOME = "outcome"
    REACTION = "reaction"
    PAIN = "pain"
    ACTIVATION = "activation"  # PerceptTraceBuffer emission
    THOUGHT = "thought"  # ThinkTool output
    RECALL = "recall"  # Hippocampus.recall() result
    CONVERSATION = "conversation"  # user/assistant turn


@dataclass(frozen=True, slots=True)
class WMEntry:
    """A single entry in the WorkingMemorySet.

    Frozen so entries are safe to hand out without copies.  ``content``
    may be a dict or string — always display-ready for prompt assembly.
    """

    kind: WorkingMemoryKind
    ref: str | None  # episode_id, percept_id, reaction_id, or None
    content: Any  # dict or string payload (display-ready)
    timestamp: float
    tick: int  # monotonic add-order counter (for decay / ordering)
    salience: float = 0.0
    agent_id: str | None = None  # multi-agent scoping


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

    def __init__(self, *, agent_id: str, capacity: int = 64) -> None:
        self._agent_id = agent_id
        self._capacity = capacity
        self._buf: deque[WMEntry] = deque(maxlen=capacity)
        self._tick = 0
        self._lock = threading.Lock()

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
    ) -> WMEntry:
        """Append a new entry.  Thread-safe.

        Returns the created entry (useful for testing / provenance).
        """
        with self._lock:
            entry = WMEntry(
                kind=kind,
                ref=ref,
                content=content,
                timestamp=time.time(),
                tick=self._tick,
                salience=salience,
                agent_id=agent_id or self._agent_id,
            )
            self._buf.append(entry)
            self._tick += 1
            return entry

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
