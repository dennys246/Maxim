"""ProvenanceCollector — two-tier, thread-safe, session-aware provenance collection.

Tier 1 (Cycle Traces): Tied to run_id, tracks perception->outcome.
Tier 2 (Activity Log): Background operations, no run_id required.

All dict access protected by lock for thread safety (bus callbacks,
worker threads, and main loop can all call simultaneously).

Session-aware: all traces and activities are tagged with session_id
for cross-run persistence and concept lineage queries.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from maxim.provenance.types import (
    PipelineStage,
    ProvenanceEntry,
    ProvenanceRef,
    ProvenanceTrace,
    ProvenanceVerbosity,
)

if TYPE_CHECKING:
    from maxim.provenance.store import ProvenanceStore

logger = logging.getLogger(__name__)


class ProvenanceCollector:
    """Two-tier provenance collection with thread safety and session identity."""

    def __init__(
        self,
        verbosity: ProvenanceVerbosity = ProvenanceVerbosity.COMPACT,
        max_traces: int = 100,
        max_activities: int = 500,
        session_id: str | None = None,
    ) -> None:
        self.verbosity = verbosity
        self.session_id = session_id or str(uuid4())
        self._lock = threading.Lock()
        self._traces: dict[str, ProvenanceTrace] = {}
        self._activities: list[ProvenanceEntry] = []
        self._max_traces = max_traces
        self._max_activities = max_activities
        self._store: ProvenanceStore | None = None

    # ---- Tier 1: Cycle Traces (run_id required) ----

    def begin_trace(self, run_id: str) -> ProvenanceTrace | None:
        """Start a new trace for an agent cycle.

        Returns None when verbosity=OFF (zero allocation).
        """
        if self.verbosity == ProvenanceVerbosity.OFF:
            return None
        trace = ProvenanceTrace(
            trace_id=str(uuid4()),
            run_id=run_id,
            session_id=self.session_id,
            started_at=time.time(),
        )
        with self._lock:
            self._traces[run_id] = trace
            self._evict_old_traces()
        return trace

    def record(
        self,
        run_id: str,
        stage: PipelineStage,
        component: str,
        action: str,
        sources: list[ProvenanceRef] | None = None,
        confidence: float = 1.0,
        **metadata: Any,
    ) -> None:
        """Record a cycle-bound provenance entry.

        No-op if verbosity=OFF or no trace exists for run_id.
        """
        if self.verbosity == ProvenanceVerbosity.OFF:
            return
        with self._lock:
            trace = self._traces.get(run_id)
        if trace is None:
            logger.debug(
                "Provenance: no trace for run_id=%s (stage=%s, component=%s)",
                run_id[:16],
                stage.value,
                component,
            )
            return
        trace.add(stage, component, action, sources, confidence, **metadata)

    def complete_trace(self, run_id: str) -> ProvenanceTrace | None:
        """Mark trace complete, persist if store is wired."""
        with self._lock:
            trace = self._traces.get(run_id)
        if trace:
            trace.complete()
            if self._store:
                self._store.write_trace(trace)
                trace._persisted = True
        return trace

    def get_trace(self, run_id: str) -> ProvenanceTrace | None:
        with self._lock:
            return self._traces.get(run_id)

    def recent_traces(self, limit: int = 10) -> list[ProvenanceTrace]:
        with self._lock:
            completed = [t for t in self._traces.values() if t.completed]
        return sorted(
            completed,
            key=lambda t: t.started_at,
            reverse=True,
        )[:limit]

    # ---- Tier 2: Activity Log (no run_id needed) ----

    def log_activity(
        self,
        stage: PipelineStage,
        component: str,
        action: str,
        sources: list[ProvenanceRef] | None = None,
        confidence: float = 1.0,
        **metadata: Any,
    ) -> None:
        """Log a background activity (not tied to a cycle trace).

        For: ConceptExtractor, ConceptGrounder, Hippocampus sleep,
        NAc causal learning — operations that run asynchronously
        and don't belong to a specific agent cycle.
        """
        if self.verbosity == ProvenanceVerbosity.OFF:
            return
        entry = ProvenanceEntry(
            timestamp=time.time(),
            stage=stage,
            component=component,
            action=action,
            sources=sources or [],
            confidence=confidence,
            metadata=metadata,
        )
        with self._lock:
            self._activities.append(entry)
            if len(self._activities) > self._max_activities:
                self._activities = self._activities[-self._max_activities :]
        # Persist immediately if store is wired
        if self._store:
            self._store.write_activity(entry, self.session_id)

    def recent_activities(
        self,
        limit: int = 20,
        stage: PipelineStage | None = None,
    ) -> list[ProvenanceEntry]:
        """Return recent background activities, optionally filtered."""
        with self._lock:
            entries = list(self._activities)
        if stage:
            entries = [e for e in entries if e.stage == stage]
        return entries[-limit:]

    # ---- Session lifecycle ----

    def on_session_end(self) -> dict[str, Any]:
        """Flush all pending data and write session summary.

        Called from MemoryHub.on_session_end() hook.
        Collects trace refs under lock, releases lock, then writes
        outside — avoids holding lock during file I/O.
        """
        if self._store and self.verbosity > ProvenanceVerbosity.OFF:
            # Collect under lock, write outside
            with self._lock:
                unpersisted = [
                    t
                    for t in self._traces.values()
                    if t.completed and not t._persisted
                ]
                stats: dict[str, Any] = {
                    "session_id": self.session_id,
                    "total_traces": len(self._traces),
                    "completed_traces": sum(
                        1 for t in self._traces.values() if t.completed
                    ),
                    "total_activities": len(self._activities),
                }
            # Write outside lock
            for trace in unpersisted:
                self._store.write_trace(trace)
                trace._persisted = True
            self._store.write_session_summary(self.session_id, stats)
            self._store.close()
            return stats
        return {}

    # ---- Internal ----

    def _evict_old_traces(self) -> None:
        """Must be called with lock held."""
        if len(self._traces) > self._max_traces:
            oldest = sorted(
                self._traces,
                key=lambda k: self._traces[k].started_at,
            )
            for k in oldest[: len(self._traces) - self._max_traces]:
                del self._traces[k]
