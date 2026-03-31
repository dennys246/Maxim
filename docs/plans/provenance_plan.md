# System-Wide Provenance & Traceability Plan

> **Status:** **IMPLEMENTED.** All phases P1-P8 complete.
>
> **Revision:** v6 — post-A7 audit fixes (two passes). Pass 1: per-trace
> lock + `_persisted` flag on ProvenanceTrace, `on_session_end()` releases
> lock before I/O, `_session_id` key mismatch fixed, config uses env var /
> getattr, `open` type fixed, `build_context()` uses Tier 2, registry line
> corrected to 301. Pass 2: P3a pool access ordering fix, P3b merged
> duplicate DECISION entries, P3f moved out of rwlock, P5 "current" vs
> "recent" differentiated, P6 `close()` separated from summary write,
> P2b cleanup added, `compare=False` on internal fields.
> Carries forward v5: ExplainTool via `registry`, `complete_trace(run_id)`
> as parameter, MemoryHub dependency. Carries forward v4: cross-run
> persistence, communicable export, session lifecycle, begin_trace OFF
> fix, sleep() dict fix.

---

## Problem

Maxim's bio systems are a black box. Decision reasoning, concept
formation, pattern completion, causal learning, and consolidation all
happen silently. Users have no way to ask "why did Maxim do that?" or
trace a decision back to the memories, concepts, and predictions that
informed it.

**Cross-run gap:** Even if provenance is collected within a session,
it's lost on shutdown. Concepts learned in session 1, grounded in
session 3, and used for decisions in session 7 have no traceable
lineage. Users can't answer "why does Maxim know about kitchens?" or
"when did it learn that navigating to the kitchen usually succeeds?"

The building blocks exist — `Decision.reasoning`, `LLMProposal.citations`,
`ConceptProvenance`, `CausalLink`, `PredictedOutcome.source_episode_id`,
`CrossLayerEdge` metadata — but they're scattered across 10+ files with
no unified way to collect, connect, persist, or surface them.

---

## Goal

A lightweight, opt-in provenance system that:

1. **Collects** decision traces across the agent pipeline without
   modifying existing hot paths
2. **Connects** traces to their source memories, concepts, and
   predictions with `layer:id` references (leveraging ID-text
   foundation from A7.0a-b)
3. **Persists** traces and activities across sessions with session
   identity, enabling cross-run concept lineage queries
4. **Surfaces** traces to users at configurable verbosity levels
5. **Exports** session reports in structured (JSON) and human-readable
   (markdown) formats for communicability

---

## Naming Standard

Shared with A7 bio-context plan (`docs/plans/bio-skill-integration.md`):

| Concept | Convention | Examples |
|---|---|---|
| Layer names | lowercase snake_case | `"hippocampus"`, `"atl"`, `"angular_gyrus"`, `"nac"`, `"ips"`, `"scn"` |
| Component names | lowercase snake_case | `"concept_extractor"`, `"memory_agent"`, `"llm_worker"`, `"nac"` |
| Pipeline stages | `PipelineStage` enum | `PERCEPTION`, `RECALL`, `DECISION`, `ACTION`, `OUTCOME`, `FORMATION`, `ENRICHMENT`, `LEARNING`, `CONSOLIDATION` |
| Reference format | `layer:id_prefix label` | `` `atl:c7f2a1b3` kitchen (location, confidence: 0.82) `` |
| Verbosity levels | `ProvenanceVerbosity` enum | `OFF` (0), `COMPACT` (1), `VERBOSE` (2) |
| Session ID | UUID4, truncated to 12 chars in display | `session:a1b2c3d4e5f6` |

---

## Verbosity Levels

Integrates with the existing structured logging verbosity system
(structured_logging.py: QUIET=0, NORMAL=1, VERBOSE=2, DEBUG=3):

| Level | Enum | What's Shown | When |
|---|---|---|---|
| **0** | `OFF` | Nothing. Zero overhead. | Production, battery-conscious |
| **1** | `COMPACT` | Stage summaries with `layer:id` refs. One line per stage. | Normal operation, quick debugging |
| **2** | `VERBOSE` | Full detail: alternatives, AG properties, timestamps, latencies, cross-layer edges. | Deep debugging, development |

```python
class ProvenanceVerbosity(IntEnum):
    OFF = 0       # No provenance collection or output
    COMPACT = 1   # Stage summaries + source references
    VERBOSE = 2   # Full metadata, alternatives, timing
```

**Configuration:** Set via environment variable or Maxim instance
attribute (codebase uses `getattr`/env pattern, not YAML):

```bash
MAXIM_PROVENANCE_VERBOSITY=1   # 0=off, 1=compact, 2=verbose
MAXIM_PROVENANCE_PERSIST=1     # 0=disable, 1=enable (default)
```

Defaults: `max_traces=100`, `max_activities=500` (hardcoded in
ProvenanceCollector, overridable via constructor).

**Guard pattern** (replaces `if self._collector:`):

```python
if self._collector and self._collector.verbosity >= ProvenanceVerbosity.COMPACT:
    self._collector.record(...)

# For verbose-only entries (AG properties, alternatives):
if self._collector and self._collector.verbosity >= ProvenanceVerbosity.VERBOSE:
    self._collector.record(
        ..., alternatives=proposal.next_actions,
        ag_properties=stats,
    )
```

---

## Architecture: Two-Tier Collection + Session Awareness

### Two Tiers

Previous plan tried to tie ALL recording to `run_id`. But background
operations (concept extraction, grounding, sleep consolidation, NAc
learning) are NOT part of a single agent cycle — they happen
asynchronously. Forcing them into a run_id lookup silently fails.

**Tier 1: Cycle Traces (tied to run_id)** — operations within an
agent cycle: perception, recall, decision, action, outcome.

**Tier 2: Activity Log (no run_id)** — background operations:
concept extraction, grounding, sleep consolidation, NAc learning.

### Session Awareness

Both tiers are tagged with a `session_id` generated once per Maxim
startup. This enables:

- **Cross-run queries:** "Show all sessions where kitchen was used"
- **Concept lineage:** "When was kitchen first learned? How has its
  confidence evolved?"
- **Session comparison:** "What changed between session A and B?"

```
Session 1 (session_id: abc123)
├── Trace: cycle_001 → learned "kitchen" (atl:f47ac...)
├── Activity: concept_extractor → extracted kitchen
├── Activity: nac → causal: navigate→arrival (V=0.3)
└── Summary: 12 cycles, 5 concepts, 3 causal links

Session 5 (session_id: def456)
├── Trace: cycle_042 → used "kitchen" for navigation decision
│   └── ProvenanceRef links back to atl:f47ac...
├── Activity: concept_grounder → grounded kitchen with 8 AG props
├── Activity: nac → causal: navigate→arrival (V=0.72, n=14)
└── Summary: 8 cycles, 2 new concepts, kitchen confidence: 0.82

Query: "Why does Maxim know about kitchens?"
→ Scans sessions.json manifest → loads abc123, def456 traces
→ Shows: learned in session 1, grounded in session 5, used 14 times
```

---

## What Already Exists

| Mechanism | File | What It Tracks |
|---|---|---|
| `Decision.reasoning` | types.py:145 | Why Maxim decided to act |
| `Decision.alternatives_considered` | types.py:146 | What else was considered |
| `Decision.confidence` | types.py:148 | How sure Maxim was |
| `LLMProposal.reasoning` | llm_types.py:161 | LLM's stated reasoning |
| `LLMProposal.citations` | llm_types.py:164 | Sources the LLM cited |
| `LLMProposal.strategy_used` | llm_types.py:160 | Which strategy drove the decision |
| `ConceptProvenance` | semantic_types.py:23-29 | How a concept was acquired (4 types) |
| `CausalLink` | causal_link.py:103 | Event→Outcome with Rescorla-Wagner learning |
| `PredictedOutcome.source_episode_id` | types.py:104 | Which episode predicted this outcome |
| `CrossLayerEdge.metadata` | cross_layer.py:56 | Extensible dict on every edge |
| `Concept.memory_refs` | semantic_types.py:272 | Which episodes reference each concept |
| `ToolResult` | bus.py:621 | Tool execution success/failure/params |
| `log_agentic()` | structured_logging.py | Structured event logging with verbosity |
| `explain_reasoning` strategy | strategies.py:141 | Already registered, not wired |

**Depends on A7.0a-b:** `to_context_dict()` must include record `id`
fields so ProvenanceRef can link `layer:id` to human-readable text.

---

## Architecture Constraints (from code audit)

1. **`run_id` lifecycle:** Generated once per cycle in
   `agent_loop.py:399`. Available in MemoryAgent via
   `_begin_memory_formation(run_id=...)` (memory_agent.py:318).
   NOT passed to `propose_intent()`.

2. **`EpisodicMemory.run_id`** (types.py:382) — field on every episode.
   This IS our per-cycle trace key.

3. **Bus events lack `run_id`:** `Percept`, `ToolResult`,
   `GoalCompleted` have NO `run_id` field. Bus subscriptions can't
   auto-correlate events to traces.

4. **`prompt_builder.py` is module-level functions** — NOT a class.
   Can't hold collector state.

5. **Background workers run on separate threads:** ConceptExtractor
   worker thread, WorkerPool review/record lanes. These don't have
   run_id context.

6. **`_traces` dict needs thread safety:** Bus callbacks, worker
   threads, and main loop can all access simultaneously.

7. **All persistence uses atomic writes:** `.tmp` + `os.replace()`.
   ProvenanceStore must follow this pattern.

8. **ExecAgent has no `register_tool()`:** Tools are registered on
   `ToolRegistry` (tools/registry.py:15). ExplainTool must be added
   to the registry directly, not via ExecAgent.

9. **MemoryHub.on_session_end()** is the canonical shutdown hook
   (agent_loop.py:2460). Provenance must flush and persist here.

10. **`wire_memory_hub()` is now called.** MemoryHub is created in
    `agentic_runtime.py` (after NAc), wired via
    `agent.wire_memory_hub(memory_hub)`, and passed to
    `run_agentic_loop(memory_hub=, hippocampus=)`. Stored as
    `self._memory_hub` on the Maxim instance.

11. **`_current_run_id` does NOT exist** on MemoryAgent. `run_id` is
    a parameter to `_begin_memory_formation(percept, run_id)` and
    `_complete_forming_memory(run_id, outcome)` — not stored as an
    attribute. The forming pool is keyed by run_id:
    `self._forming_pool[run_id] = entry`.

12. **`_tool_registry` does NOT exist** on MaximAgent. The
    `ToolRegistry` is created as a local variable `registry` in
    `agentic_runtime.py:301` via `build_tool_registry()` and
    passed to the executor. It's not stored on MaximAgent.

13. **No central config dict.** Agentic runtime uses
    `getattr(self, "field", default)` on the Maxim instance and
    env vars — NOT a config.get() pattern. Provenance config
    should follow this: `MAXIM_PROVENANCE_VERBOSITY` env var
    or `self.provenance_verbosity` attribute.

---

## Implementation Phases

### P1. Core Types (~90 lines)

**New file:** `src/maxim/provenance/__init__.py`
**New file:** `src/maxim/provenance/types.py`

```python
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class PipelineStage(Enum):
    """Canonical pipeline stages for provenance tracking."""
    PERCEPTION = "perception"
    RECALL = "recall"
    DECISION = "decision"
    ACTION = "action"
    OUTCOME = "outcome"
    FORMATION = "formation"
    ENRICHMENT = "enrichment"
    LEARNING = "learning"
    CONSOLIDATION = "consolidation"


class ProvenanceVerbosity(IntEnum):
    """Provenance output verbosity levels."""
    OFF = 0       # No collection or output
    COMPACT = 1   # Stage summaries + source references
    VERBOSE = 2   # Full metadata, alternatives, timing


@dataclass
class ProvenanceRef:
    """Reference to a specific memory/concept/record.

    Uses the layer:id convention shared with A7 BioContext output.
    After A7.0a-b, all records include their ID in to_context_dict(),
    so refs can always be resolved to text.
    """
    layer: str          # "atl", "hippocampus", "angular_gyrus", "nac"
    id: str             # Record ID (UUID)
    label: str          # Human-readable: "kitchen (location)"
    confidence: float = 1.0

    def __str__(self) -> str:
        return (
            f"`{self.layer}:{self.id[:8]}` {self.label} "
            f"(confidence: {self.confidence:.2f})"
        )

    def short(self) -> str:
        """Compact format for verbosity=1."""
        return f"`{self.layer}:{self.id[:8]}` {self.label}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceRef:
        return cls(
            layer=data["layer"],
            id=data["id"],
            label=data["label"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ProvenanceEntry:
    """Single step in a provenance trace."""
    timestamp: float
    stage: PipelineStage
    component: str      # "concept_extractor", "memory_agent", etc.
    action: str         # "resolved 3 concepts", "proposed navigate_to"
    sources: list[ProvenanceRef] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "stage": self.stage.value,
            "component": self.component,
            "action": self.action,
            "sources": [r.to_dict() for r in self.sources],
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceEntry:
        return cls(
            timestamp=data["timestamp"],
            stage=PipelineStage(data["stage"]),
            component=data["component"],
            action=data["action"],
            sources=[ProvenanceRef.from_dict(s) for s in data.get("sources", [])],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProvenanceTrace:
    """Full trace of a single agent cycle (perception→outcome).

    Thread-safe: add() and complete() are protected by a per-trace lock
    because record() releases the collector lock before calling trace.add().
    """
    trace_id: str
    run_id: str
    session_id: str
    started_at: float
    entries: list[ProvenanceEntry] = field(default_factory=list)
    completed: bool = False
    _persisted: bool = field(default=False, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(
        self,
        stage: PipelineStage,
        component: str,
        action: str,
        sources: list[ProvenanceRef] | None = None,
        confidence: float = 1.0,
        **metadata: Any,
    ) -> None:
        with self._lock:
            self.entries.append(ProvenanceEntry(
                timestamp=time.time(),
                stage=stage,
                component=component,
                action=action,
                sources=sources or [],
                confidence=confidence,
                metadata=metadata,
            ))

    def complete(self) -> None:
        with self._lock:
            self.completed = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "trace",
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "completed": self.completed,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceTrace:
        trace = cls(
            trace_id=data["trace_id"],
            run_id=data["run_id"],
            session_id=data.get("session_id", "unknown"),
            started_at=data["started_at"],
            completed=data.get("completed", True),
        )
        trace.entries = [
            ProvenanceEntry.from_dict(e) for e in data.get("entries", [])
        ]
        return trace
```

**Changes from v3:**
- `ProvenanceTrace` gains `session_id` field
- `to_dict()` adds `"type": "trace"` discriminator (for mixed JSONL)
- All types gain `from_dict()` classmethod (needed for cross-run loading)
- `ProvenanceRef` gains `to_dict()` / `from_dict()` (needed for serialization)
- `ProvenanceEntry` gains `to_dict()` / `from_dict()`

---

### P2. ProvenanceCollector — Two-Tier, Thread-Safe, Session-Aware (~140 lines)

**New file:** `src/maxim/provenance/collector.py`

```python
import logging
import threading
import time
from uuid import uuid4

from maxim.provenance.types import (
    PipelineStage, ProvenanceEntry, ProvenanceRef,
    ProvenanceTrace, ProvenanceVerbosity,
)

logger = logging.getLogger(__name__)


class ProvenanceCollector:
    """Two-tier provenance collection with thread safety and session identity.

    Tier 1 (Cycle Traces): Tied to run_id, tracks perception→outcome.
    Tier 2 (Activity Log): Background operations, no run_id required.

    All dict access protected by lock for thread safety (bus callbacks,
    worker threads, and main loop can all call simultaneously).

    Session-aware: all traces and activities are tagged with session_id
    for cross-run persistence and concept lineage queries.
    """

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
        self._store: "ProvenanceStore | None" = None

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
        **metadata,
    ) -> None:
        """Record a cycle-bound provenance entry. No-op if verbosity=OFF
        or no trace exists for run_id."""
        if self.verbosity == ProvenanceVerbosity.OFF:
            return
        with self._lock:
            trace = self._traces.get(run_id)
        if trace is None:
            logger.debug(
                "Provenance: no trace for run_id=%s (stage=%s, component=%s)",
                run_id[:16], stage.value, component,
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
            completed, key=lambda t: t.started_at, reverse=True,
        )[:limit]

    # ---- Tier 2: Activity Log (no run_id needed) ----

    def log_activity(
        self,
        stage: PipelineStage,
        component: str,
        action: str,
        sources: list[ProvenanceRef] | None = None,
        confidence: float = 1.0,
        **metadata,
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
                self._activities = self._activities[-self._max_activities:]
        # Persist immediately if store is wired
        if self._store:
            self._store.write_activity(entry, self.session_id)

    def recent_activities(
        self, limit: int = 20, stage: PipelineStage | None = None,
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

        Called from MemoryHub.on_session_end() or wire_provenance() hook.
        Returns session statistics.

        Collects trace refs under lock, releases lock, then writes
        outside — avoids holding lock during file I/O.
        """
        if self._store and self.verbosity > ProvenanceVerbosity.OFF:
            # Collect under lock, write outside
            with self._lock:
                unpersisted = [
                    t for t in self._traces.values()
                    if t.completed and not t._persisted
                ]
                stats = {
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
            for k in oldest[:len(self._traces) - self._max_traces]:
                del self._traces[k]
```

**Changes from v3:**
- `session_id` added — generated once per Maxim startup, tags all output
- `begin_trace()` returns `None` when OFF (not a dummy allocation)
- Callers updated: `if self._collector: trace = self._collector.begin_trace(run_id)` — trace may be None
- `complete_trace()` calls `write_trace()` not `save()` (renamed for clarity)
- `log_activity()` persists immediately if store is wired
- `on_session_end()` method — flush + session summary
- Store API renamed: `write_trace()`, `write_activity()`, `write_session_summary()`

---

### P2b. Collector Wiring + Session Lifecycle (~45 lines)

**Modifies:** `src/maxim/agents/maxim_agent.py`

Add `wire_provenance()` following the `wire_memory_hub()` pattern
(maxim_agent.py:169):

```python
def wire_provenance(self, collector: "ProvenanceCollector") -> None:
    """Wire ProvenanceCollector for decision tracing.

    MemoryHub is now created and wired in agentic_runtime.py (A7.0c).
    The hub is accessible via self.memory._memory_hub (set by
    wire_memory_hub). Both Tier 1 (cycle traces) and Tier 2
    (background activity via hub components) are functional.
    """
    self._collector = collector
    self.memory._collector = collector

    # Wire to MemoryHub internals for Tier 2 (activity log)
    # hub is set by A7.0c (store MemoryHub on Maxim → exposed here)
    # Until MemoryHub integration lands, this block is inert.
    hub = getattr(self.memory, "_memory_hub", None)
    if hub is not None:
        # Store collector on hub for session lifecycle
        hub._collector = collector
        # Hippocampus (for sleep consolidation)
        hub.hippocampus._collector = collector
        # ConceptExtractor (for concept formation)
        if hub._concept_extractor:
            hub._concept_extractor._collector = collector
        # ConceptGrounder (for AG enrichment)
        if hub._concept_grounder:
            hub._concept_grounder._collector = collector
        # NAc (for causal learning)
        if hub.nac:
            hub.nac._collector = collector

```

**Called from agentic_runtime.py** (after `registry = build_tool_registry(...)` at line 301):

```python
from maxim.provenance.collector import ProvenanceCollector
from maxim.provenance.types import ProvenanceVerbosity

# Config via env var or Maxim instance attribute (no central config dict)
prov_verbosity = int(os.getenv(
    "MAXIM_PROVENANCE_VERBOSITY",
    str(getattr(self, "provenance_verbosity", 1)),
))
verbosity = ProvenanceVerbosity(min(prov_verbosity, 2))
collector = ProvenanceCollector(verbosity=verbosity)

# Wire persistence (default on unless explicitly disabled)
prov_persist = os.getenv("MAXIM_PROVENANCE_PERSIST", "1") != "0"
if prov_persist:
    from maxim.provenance.store import ProvenanceStore
    data_dir = str(getattr(self, "home_dir", "data") or "data")
    collector._store = ProvenanceStore(
        base_dir=os.path.join(data_dir, "provenance"),
    )

# Wire collector to agent and its sub-components
agent.wire_provenance(collector)
self._provenance_collector = collector

# Register ExplainTool on the ToolRegistry
# (registry is a local variable in agentic_runtime.py:301,
# NOT stored on MaximAgent — must register here, not in wire_provenance)
from maxim.tools.explain import ExplainTool
registry.register(ExplainTool(collector))
```

**Cleanup in `_stop_agentic_runtime()`** (after `self._memory_hub = None`):

```python
self._provenance_collector = None
```

**Session lifecycle hook in MemoryHub.on_session_end():**

```python
# In MemoryHub.on_session_end(), after all consolidation and saves:
if hasattr(self, "_collector") and self._collector:
    try:
        self._collector.on_session_end()
    except Exception as e:
        logger.warning("Provenance session end failed: %s", e)
```

**Trace lifecycle tied to MemoryAgent formation:**

```python
# In MemoryAgent._begin_memory_formation(self, percept, run_id):
if self._collector:
    self._collector.begin_trace(run_id)

# In MemoryAgent._complete_forming_memory(self, run_id, outcome):
# run_id is a parameter — NOT stored as self._current_run_id (which doesn't exist)
if self._collector:
    self._collector.complete_trace(run_id)
```

---

### P3. Instrument Existing Components (~90 lines total)

All instrumentation guarded by `if self._collector and self._collector.verbosity >= COMPACT:`.

#### P3a. MemoryAgent — Tier 1 (recall + predictions) (~15 lines)

**Modifies:** `src/maxim/agents/memory_agent.py`

MemoryAgent has `run_id` from `_begin_memory_formation()`. Access
forming entries via `self._forming_pool[run_id]` — NOT bare variables:

```python
# In _begin_memory_formation(), after self._forming_pool[run_id] = entry (line ~398):
# NOTE: must be AFTER pool assignment — entry isn't in pool during pattern completion.
if self._collector and self._collector.verbosity >= ProvenanceVerbosity.COMPACT:
    if entry.predicted_outcomes:
        refs = [
            ProvenanceRef("hippocampus", p.source_episode_id,
                           f"{p.tool} (success={p.success})", p.confidence)
            for p in entry.predicted_outcomes if p.source_episode_id
        ]
        self._collector.record(
            run_id, PipelineStage.RECALL, "pattern_completer",
            f"{len(refs)} predicted outcomes", sources=refs,
        )
```

**Note:** Concept context is built in `build_context()` (line ~957),
not during formation. `build_context()` has no `run_id` parameter and
doesn't need one — concept context provenance uses `log_activity()`
(Tier 2) since context building isn't tied to a specific cycle.

#### P3b. MemoryAgent — Tier 1 (decision via proposal) (~15 lines)

**Modifies:** `src/maxim/agents/memory_agent.py`

In `_update_forming_decision(self, run_id, decision)` — uses the
`decision` parameter directly (NOT a bare `memory` variable):

```python
# In _update_forming_decision(), after entry.record.decision = decision:
if self._collector and self._collector.verbosity >= ProvenanceVerbosity.COMPACT:
    action = decision.intent.get("action", "none") if decision.intent else "none"
    metadata: dict[str, Any] = {"reasoning": decision.reasoning}
    # Verbose-only: include alternatives in same entry (not a separate record)
    if self._collector.verbosity >= ProvenanceVerbosity.VERBOSE:
        metadata["alternatives"] = decision.alternatives_considered or []
    self._collector.record(
        run_id, PipelineStage.DECISION, "memory_agent",
        f"Action: {action} (confidence: {decision.confidence:.2f})",
        confidence=decision.confidence,
        **metadata,
    )
```

#### P3c. MemoryAgent — Tier 1 (outcome) (~10 lines)

In `_complete_forming_memory(self, run_id, outcome)` — uses the
`outcome` parameter and accesses entry via `_forming_pool`:

```python
# In _complete_forming_memory(), after entry.record.outcome = outcome:
if self._collector and self._collector.verbosity >= ProvenanceVerbosity.COMPACT:
    entry = self._forming_pool.get(run_id)
    record = entry.record if entry else None
    duration = record.action.execution_time_ms if record and record.action else 0
    self._collector.record(
        run_id, PipelineStage.OUTCOME, "memory_agent",
        f"Success={outcome.success}, duration={duration:.0f}ms",
    )
```

#### P3d. ConceptExtractor — Tier 2 (background activity) (~10 lines)

**Modifies:** `src/maxim/memory/concept_extractor.py`

After `_process_capture()` extracts concepts. Uses `log_activity()`
(NOT `record()`) because this runs on the worker thread:

```python
if self._collector and self._collector.verbosity >= ProvenanceVerbosity.COMPACT:
    refs = [
        ProvenanceRef("atl", cid, f"{name} ({cat})")
        for cid, name, cat in concept_ids
    ]
    self._collector.log_activity(
        PipelineStage.FORMATION, "concept_extractor",
        f"Extracted {len(refs)} concepts from hpc:{memory_id[:8]}",
        sources=refs,
    )
```

#### P3e. ConceptGrounder — Tier 2 (background activity) (~10 lines)

**Modifies:** `src/maxim/memory/concept_grounder.py`

```python
if self._collector and self._collector.verbosity >= ProvenanceVerbosity.COMPACT:
    prop_count = sum(1 for v in stats.values() if isinstance(v, dict))
    self._collector.log_activity(
        PipelineStage.ENRICHMENT, "concept_grounder",
        f"Grounded atl:{concept.id[:8]} ({concept.name}) "
        f"with {prop_count} AG properties",
        sources=[ProvenanceRef("atl", concept.id, concept.name)],
    )
```

#### P3f. Hippocampus sleep — Tier 2 (background activity) (~10 lines)

**Modifies:** `src/maxim/memory/hippocampus.py`

**Fixed:** `sleep()` returns `dict[str, int]` with keys "compressed",
"removed", "preserved", "promoted" — NOT lists. Use dict values.

**Fixed (v6):** Instrumentation goes in `sleep()` (public wrapper),
NOT `_sleep()` — because `_sleep()` runs under `_rwlock.write()` and
provenance I/O would block all hippocampus reads.

```python
# In sleep() (line 1647), AFTER the rwlock is released:
# i.e.: results = self._sleep(strategy)  # inside lock
#        <instrumentation here>           # outside lock
#        return results
if hasattr(self, "_collector") and self._collector and \
   self._collector.verbosity >= ProvenanceVerbosity.COMPACT:
    self._collector.log_activity(
        PipelineStage.CONSOLIDATION, "hippocampus",
        f"Promoted {results['promoted']}, compressed {results['compressed']}, "
        f"removed {results['removed']}",
    )
```

#### P3g. NAc causal learning — Tier 2 (background activity) (~10 lines)

**Modifies:** `src/maxim/decisions/nac.py`

```python
# After causal link update:
if hasattr(self, "_collector") and self._collector and \
   self._collector.verbosity >= ProvenanceVerbosity.COMPACT:
    self._collector.log_activity(
        PipelineStage.LEARNING, "nac",
        f"Causal: {link.event_signature} → {link.outcome_signature} "
        f"(V={link.predicted_value:.2f}, n={link.observation_count})",
        sources=[ProvenanceRef("nac", link.id, link.event_signature)],
    )
```

---

### P4. Trace Rendering (~80 lines)

**New file:** `src/maxim/provenance/render.py`

Four render functions — same data, different contexts:

```python
def render_trace(
    trace: ProvenanceTrace,
    verbosity: ProvenanceVerbosity = ProvenanceVerbosity.COMPACT,
) -> str:
    """Render a single cycle trace as markdown."""
    ...

def render_activities(
    activities: list[ProvenanceEntry],
    verbosity: ProvenanceVerbosity = ProvenanceVerbosity.COMPACT,
) -> str:
    """Render background activity log as markdown."""
    ...

def render_summary(
    traces: list[ProvenanceTrace],
    activities: list[ProvenanceEntry] | None = None,
) -> str:
    """Render multi-cycle summary table with optional activity digest."""
    ...

def render_session_report(
    traces: list[ProvenanceTrace],
    activities: list[ProvenanceEntry],
    session_id: str,
    verbosity: ProvenanceVerbosity = ProvenanceVerbosity.VERBOSE,
) -> str:
    """Full session report for export/sharing."""
    ...
```

#### Verbosity=1 (COMPACT) output:

```markdown
## Decision Trace [2026-03-16_143022]

**perception** → 3 objects, salience=0.82
**recall** → 2 concepts:
  - `atl:c7f2a1b3` kitchen (location)
  - `atl:e3b9d4f1` navigate (action)
**recall** → 1 prediction:
  - `hpc:a1c8f0e2` navigate (success=True)
**decision** → navigate_to (confidence: 0.9)
**outcome** → success, 1.2s
```

#### Verbosity=2 (VERBOSE) output:

```markdown
## Decision Trace [2026-03-16_143022] session:a1b2c3d4e5f6

**perception** [14:30:22.103] → 3 objects, salience=0.82
**recall** [14:30:22.115, +12ms] → 2 concepts:
  - `atl:c7f2a1b3` kitchen (location, confidence: 0.82, episodes: 14)
  - `atl:e3b9d4f1` navigate (action, confidence: 0.71, episodes: 8)
**recall** [14:30:22.118, +3ms] → 1 prediction:
  - `hpc:a1c8f0e2` navigate (success=True, confidence: 0.85)
**decision** [14:30:22.340, +222ms] → navigate_to target=kitchen
  Strategy: assist | Confidence: 0.9
  Reasoning: "Kitchen previously had target object with high confidence"
  Alternatives: 2 considered
    - observe (confidence: 0.4)
    - wait (confidence: 0.2)
**outcome** [14:30:23.540, +1200ms] → success, duration=1200ms

### Background Activity
**formation** [14:30:23.545] concept_extractor → Extracted 3 concepts from hpc:a1c8f0e2
  - `atl:c7f2a1b3` kitchen (location)
  - `atl:e3b9d4f1` navigate (action)
  - `atl:d8e2f1a3` skill:patrol (skill_execution)
**enrichment** [14:30:23.890] concept_grounder → Grounded atl:c7f2a1b3 (kitchen) with 5 AG properties
**learning** [14:30:24.010] nac → Causal: navigate→arrival (V=0.72, n=14)
```

#### Session report output (for export):

```markdown
# Session Report: a1b2c3d4e5f6
**Started:** 2026-03-16 14:30:00 | **Duration:** 42m 18s

## Summary
- **Cycles:** 12 completed
- **Success rate:** 83% (10/12)
- **Concepts used:** kitchen, navigate, slope, mug, patrol
- **New concepts learned:** slope_gradient (AG grounded)
- **Causal links updated:** navigate→arrival (V: 0.3 → 0.72)

## Cycle Details
| # | Goal | Outcome | Key Concepts | Duration |
|---|---|---|---|---|
| 1 | navigate to kitchen | success | kitchen, navigate | 1.2s |
| 2 | observe slope | success | slope, terrain | 0.8s |
...

## Background Activity
- concept_extractor: 15 concepts extracted
- concept_grounder: 4 concepts enriched with AG properties
- nac: 3 causal links updated
- hippocampus: 2 promoted, 5 compressed, 3 removed

## Concept Evolution
| Concept | Start Confidence | End Confidence | Episodes | Change |
|---|---|---|---|---|
| kitchen | 0.65 | 0.82 | 14 → 18 | +4 episodes, grounded |
| navigate | 0.50 | 0.71 | 8 → 12 | +4 episodes |
```

---

### P5. User-Facing Surface (~80 lines)

**New file:** `src/maxim/tools/explain.py`

ExplainTool receives the collector in `__init__()`, same pattern as
tools that need external references:

```python
class ExplainTool(Tool):
    """Surface provenance trace for the current or recent decision."""

    name = "explain"
    description = (
        "Show why Maxim made a decision — what memories, concepts, "
        "and predictions informed it. Supports: current cycle, recent "
        "cycles, session summary, concept history, and session export."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "'current' | 'recent' | 'summary' | 'export' | "
                    "'concept:<name>' | run_id"
                ),
                "default": "current",
            },
            "verbosity": {
                "type": "integer",
                "description": "Detail level: 1=compact, 2=verbose",
                "default": 1,
            },
        },
    }

    def __init__(self, collector: "ProvenanceCollector") -> None:
        self._collector = collector

    def execute(self, query: str = "current", verbosity: int = 1, **kw) -> Any:
        from maxim.provenance.render import (
            render_activities, render_session_report,
            render_summary, render_trace,
        )
        v = ProvenanceVerbosity(min(verbosity, 2))

        if query == "summary":
            traces = self._collector.recent_traces(limit=20)
            activities = self._collector.recent_activities(limit=50)
            return render_summary(traces, activities)

        if query == "export":
            traces = self._collector.recent_traces(limit=100)
            activities = self._collector.recent_activities(limit=200)
            return render_session_report(
                traces, activities, self._collector.session_id, v,
            )

        if query.startswith("concept:"):
            concept_name = query[len("concept:"):]
            return self._query_concept_history(concept_name)

        if query == "current":
            # Try in-progress trace first, fall back to most recent completed
            with self._collector._lock:
                in_progress = [
                    t for t in self._collector._traces.values()
                    if not t.completed
                ]
            if in_progress:
                trace = max(in_progress, key=lambda t: t.started_at)
            else:
                traces = self._collector.recent_traces(limit=1)
                trace = traces[0] if traces else None
        elif query == "recent":
            traces = self._collector.recent_traces(limit=1)
            trace = traces[0] if traces else None
        else:
            trace = self._collector.get_trace(query)

        if trace is None:
            return "No provenance trace available."

        result = render_trace(trace, verbosity=v)

        # In verbose mode, include recent background activities
        if v >= ProvenanceVerbosity.VERBOSE:
            activities = self._collector.recent_activities(limit=10)
            if activities:
                result += "\n" + render_activities(activities, verbosity=v)

        return result

    def _query_concept_history(self, concept_name: str) -> str:
        """Query cross-run concept history via ProvenanceStore."""
        if not self._collector._store:
            return "Provenance persistence not enabled."
        results = self._collector._store.query_concept(concept_name)
        if not results:
            return f"No provenance records found for concept '{concept_name}'."

        lines = [f"## Concept History: {concept_name}\n"]
        for r in results:
            session = r.get("session_id", "?")[:12]
            ts = r.get("timestamp", 0)
            action = r.get("action", "")
            component = r.get("component", "")
            lines.append(
                f"- **{component}** [session:{session}] {action}"
            )
        return "\n".join(lines)
```

**Registration** — via ToolRegistry in wire_provenance() (see P2b).

**Also wire `explain_reasoning` strategy** (strategies.py:141) —
when active, MemoryAgent automatically appends compact traces to
LLM context.

---

### P6. Session-Aware Provenance Store (~120 lines)

**New file:** `src/maxim/provenance/store.py`

Replaces v3's naive append-only JSONL with session-aware, crash-safe
persistence. Follows the codebase's existing atomic write pattern.

```python
import json
import logging
import os
import time
from pathlib import Path

from maxim.provenance.types import (
    PipelineStage, ProvenanceEntry, ProvenanceRef, ProvenanceTrace,
)

logger = logging.getLogger(__name__)


class ProvenanceStore:
    """Session-aware, crash-safe provenance persistence.

    Directory structure:
        data/provenance/
        ├── sessions.json         # Manifest of all sessions
        ├── {session_id}.jsonl    # Per-session traces + activities
        └── {session_id}.jsonl    # ...one file per session

    Each JSONL line is either:
    - {"type": "trace", ...}     — completed cycle trace
    - {"type": "activity", ...}  — background activity entry
    - {"type": "summary", ...}   — session summary (written on shutdown)

    Uses atomic writes for the sessions manifest (tmp + os.replace).
    JSONL appends use flush + fsync for crash safety.
    """

    def __init__(self, base_dir: str = "data/provenance") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._base_dir / "sessions.json"
        self._current_file: Any = None  # TextIOWrapper when open
        self._current_session_id: str | None = None

    def _ensure_session_file(self, session_id: str):
        """Open or reuse the JSONL file for this session."""
        if self._current_session_id == session_id and self._current_file:
            return
        if self._current_file:
            self._current_file.close()
        path = self._base_dir / f"{session_id}.jsonl"
        self._current_file = open(path, "a", encoding="utf-8")
        self._current_session_id = session_id

    def write_trace(self, trace: ProvenanceTrace) -> None:
        """Persist a completed cycle trace."""
        try:
            self._ensure_session_file(trace.session_id)
            line = json.dumps(trace.to_dict(), default=str) + "\n"
            self._current_file.write(line)
            self._current_file.flush()
        except Exception as e:
            logger.warning("Failed to persist trace: %s", e)

    def write_activity(
        self, entry: ProvenanceEntry, session_id: str,
    ) -> None:
        """Persist a background activity entry."""
        try:
            self._ensure_session_file(session_id)
            data = entry.to_dict()
            data["type"] = "activity"
            data["session_id"] = session_id
            line = json.dumps(data, default=str) + "\n"
            self._current_file.write(line)
            self._current_file.flush()
        except Exception as e:
            logger.warning("Failed to persist activity: %s", e)

    def write_session_summary(
        self, session_id: str, stats: dict,
    ) -> None:
        """Write session summary and update manifest.

        Does NOT close the file — late-arriving log_activity() calls
        may still append. Call close() explicitly when fully done.
        """
        # Write summary line to session JSONL
        try:
            self._ensure_session_file(session_id)
            summary = {
                "type": "summary",
                "session_id": session_id,
                "ended_at": time.time(),
                **stats,
            }
            self._current_file.write(
                json.dumps(summary, default=str) + "\n"
            )
            self._current_file.flush()
        except Exception as e:
            logger.warning("Failed to write session summary: %s", e)

        # Update manifest (atomic write)
        self._update_manifest(session_id, stats)

    def close(self) -> None:
        """Close the current session file. Safe to call multiple times."""
        if self._current_file:
            try:
                self._current_file.close()
            except Exception:
                pass
            self._current_file = None
            self._current_session_id = None

    def _update_manifest(
        self, session_id: str, stats: dict,
    ) -> None:
        """Atomically update sessions.json manifest."""
        try:
            manifest = self._load_manifest()
            manifest[session_id] = {
                "ended_at": time.time(),
                "file": f"{session_id}.jsonl",
                **stats,
            }
            tmp = self._manifest_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, default=str)
            os.replace(tmp, self._manifest_path)
        except Exception as e:
            logger.warning("Failed to update manifest: %s", e)

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return {}

    # ---- Cross-run queries ----

    def load_session(self, session_id: str) -> list[dict]:
        """Load all records (traces + activities) from a session."""
        path = self._base_dir / f"{session_id}.jsonl"
        if not path.exists():
            return []
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def load_recent_sessions(self, limit: int = 10) -> list[str]:
        """Return session IDs ordered by most recent."""
        manifest = self._load_manifest()
        sessions = sorted(
            manifest.items(),
            key=lambda kv: kv[1].get("ended_at", 0),
            reverse=True,
        )
        return [sid for sid, _ in sessions[:limit]]

    def query_concept(
        self, concept_name: str, max_sessions: int = 20,
    ) -> list[dict]:
        """Find all provenance records mentioning a concept across sessions.

        Scans recent session files for entries whose sources or actions
        mention the concept name. Returns records with session context.
        """
        results = []
        for session_id in self.load_recent_sessions(max_sessions):
            records = self.load_session(session_id)
            for record in records:
                if self._record_mentions_concept(record, concept_name):
                    record["session_id"] = session_id
                    results.append(record)
        return results

    def query(
        self,
        concept_name: str | None = None,
        tool_name: str | None = None,
        success: bool | None = None,
        max_sessions: int = 20,
    ) -> list[dict]:
        """General query across sessions."""
        results = []
        for session_id in self.load_recent_sessions(max_sessions):
            records = self.load_session(session_id)
            for record in records:
                if record.get("type") == "summary":
                    continue
                if self._matches(record, concept_name, tool_name, success):
                    record["session_id"] = session_id
                    results.append(record)
        return results

    @staticmethod
    def _record_mentions_concept(record: dict, concept_name: str) -> bool:
        """Check if a record mentions a concept by name."""
        # Check action text
        if concept_name.lower() in record.get("action", "").lower():
            return True
        # Check sources
        for source in record.get("sources", []):
            if concept_name.lower() in source.get("label", "").lower():
                return True
        # Check entries (for traces)
        for entry in record.get("entries", []):
            if concept_name.lower() in entry.get("action", "").lower():
                return True
            for source in entry.get("sources", []):
                if concept_name.lower() in source.get("label", "").lower():
                    return True
        return False

    @staticmethod
    def _matches(record, concept_name, tool_name, success):
        entries = record.get("entries", [])
        # For activity-type records, wrap as single entry
        if record.get("type") == "activity":
            entries = [record]
        for entry in entries:
            if concept_name:
                if any(concept_name.lower() in s.get("label", "").lower()
                       for s in entry.get("sources", [])):
                    return True
            if tool_name and tool_name in entry.get("action", ""):
                return True
            if success is not None and f"Success={success}" in entry.get("action", ""):
                return True
        return False
```

**Changes from v3:**
- Per-session JSONL files instead of monolithic file
- `sessions.json` manifest with atomic writes (tmp + os.replace)
- Activities persisted alongside traces (same JSONL, `"type": "activity"`)
- Session summaries written on shutdown
- `query_concept()` — cross-session concept lineage queries
- `load_session()` / `load_recent_sessions()` — session browsing
- Crash-safe: flush after each write, atomic manifest updates

---

### P7. LLM Context Integration (~25 lines)

**Modifies:** `src/maxim/agents/bus.py`, `src/maxim/agents/memory_agent.py`

Add `provenance_context` to StructuredContext (bus.py). It has a
default value so it's not a breaking change:

```python
# In StructuredContext dataclass:
provenance_context: str = ""  # Compact trace markdown (NEW)
```

In MemoryAgent.build_context():

```python
# After building existing context fields:
if self._collector and self._collector.verbosity >= ProvenanceVerbosity.COMPACT:
    from maxim.provenance.render import render_trace
    recent = self._collector.recent_traces(limit=3)
    if recent:
        ctx.provenance_context = "\n".join(
            render_trace(t, verbosity=ProvenanceVerbosity.COMPACT)
            for t in recent
        )
```

Prompt builder functions read `ctx.provenance_context` when non-empty
— no refactoring to a class needed.

---

### P8. Communicable Export (~40 lines)

**Modifies:** `src/maxim/tools/explain.py` (ExplainTool.execute, query="export")
**Modifies:** `src/maxim/provenance/render.py` (render_session_report)

The `explain` tool's `query="export"` mode generates a full session
report in markdown. This can be:

1. **Displayed live** in the CLI / web interface
2. **Saved to file** by the user (copy-paste or redirect)
3. **Structured as JSON** for downstream consumption

```python
# In ExplainTool.execute(), query="export":
if query == "export":
    traces = self._collector.recent_traces(limit=100)
    activities = self._collector.recent_activities(limit=200)
    return render_session_report(
        traces, activities, self._collector.session_id, v,
    )

# For structured JSON export:
if query == "export_json":
    traces = self._collector.recent_traces(limit=100)
    activities = self._collector.recent_activities(limit=200)
    return json.dumps({
        "session_id": self._collector.session_id,
        "traces": [t.to_dict() for t in traces],
        "activities": [a.to_dict() for a in activities],
    }, indent=2, default=str)
```

**Cross-session export** — load from store:

```python
if query == "history":
    if not self._collector._store:
        return "Provenance persistence not enabled."
    sessions = self._collector._store.load_recent_sessions(limit=10)
    lines = ["## Session History\n"]
    manifest = self._collector._store._load_manifest()
    for sid in sessions:
        info = manifest.get(sid, {})
        traces = info.get("completed_traces", "?")
        lines.append(
            f"- `session:{sid[:12]}` — {traces} cycles"
        )
    return "\n".join(lines)
```

---

## Implementation Order

| Step | What | Effort | Dependencies |
|---|---|---|---|
| P1 | PipelineStage, ProvenanceVerbosity, ProvenanceRef/Entry/Trace + serialization | Small (~90 lines) | A7.0a-b (ID-text) |
| P2 | ProvenanceCollector (two-tier, thread-safe, session-aware) | Small (~140 lines) | P1 |
| P2b | wire_provenance() + session lifecycle hooks | Small (~45 lines) | P2 |
| P3a-g | Instrument 5 components (3 Tier 1 + 4 Tier 2) | Medium (~90 lines) | P2b |
| P4 | Trace rendering (compact/verbose/summary/session report) | Small (~80 lines) | P1 |
| P5 | ExplainTool + concept history query | Small (~80 lines) | P2, P4, P6 |
| P6 | ProvenanceStore (session-aware, crash-safe, cross-run queries) | Medium (~120 lines) | P1 |
| P7 | StructuredContext + MemoryAgent integration | Small (~25 lines) | P2, P4 |
| P8 | Communicable export (JSON + markdown session reports) | Small (~40 lines) | P4, P5, P6 |

**Total:** ~710 lines production + ~500 lines tests.
**Order:** P1 → P2/P2b → P6 → P3 (parallelize) → P4 → P5/P7/P8 (parallel)

---

## How Provenance Integrates with Bio-Skills

After A7.0a-b adds IDs to all outputs, provenance and bio-skills
share the same ID-text linking system:

```
┌──────────────────────────────────────────────────┐
│  BioContext.resolve_concepts()                    │
│  Returns: {"id": "f47ac...", "name": "kitchen"}  │
└──────────┬───────────────────────────────────────┘
           │
           ├──── BioContext.ground_concept(c["id"])
           │     Records: log_activity(ENRICHMENT, ...)
           │     Persisted: {session_id}.jsonl
           │
           ├──── ProvenanceRef("atl", "f47ac...", "kitchen")
           │     Rendered: `atl:f47ac10b` kitchen (location)
           │
           └──── Cross-run lineage:
                 explain query="concept:kitchen"
                 → Scans all sessions for atl:f47ac...
                 → Shows: learned session 1, grounded session 5
```

The `id` field in concept dicts is the same ID used in ProvenanceRef.
A user reading a trace can follow any `layer:id` reference back to
the source record. Cross-session queries use the same ID to trace
concept evolution over time.

---

## Cross-Run Persistence Architecture

```
data/provenance/
├── sessions.json               # Manifest (atomic writes)
│   {
│     "a1b2c3d4-...": {
│       "ended_at": 1710600138.5,
│       "completed_traces": 12,
│       "total_activities": 45,
│       "file": "a1b2c3d4-....jsonl"
│     },
│     ...
│   }
│
├── a1b2c3d4-....jsonl          # Session 1
│   {"type":"trace", "session_id":"a1b2c3d4-...", ...}
│   {"type":"activity", "session_id":"a1b2c3d4-...", ...}
│   {"type":"summary", "session_id":"a1b2c3d4-...", ...}
│
└── e5f6g7h8-....jsonl          # Session 2
    ...
```

**Lifecycle:**
1. `agentic_runtime._start_agentic_runtime()` → creates collector with
   new session_id → wires ProvenanceStore
2. Each cycle → `begin_trace()` / `record()` / `complete_trace()` →
   trace written to `{session_id}.jsonl`
3. Background ops → `log_activity()` → activity written to same file
4. `MemoryHub.on_session_end()` → `collector.on_session_end()` →
   writes session summary + updates manifest
5. Next startup → new session_id → new JSONL file → manifest grows

**Cross-run queries:**
- `query_concept("kitchen")` → scans recent session JSOMLs for matching refs
- `load_session(session_id)` → load all records from a specific session
- `load_recent_sessions(10)` → get session IDs for browsing

---

## Key Design Decisions

1. **Two-tier collection solves run_id propagation.** Cycle traces
   use `run_id` (available in MemoryAgent). Background activities
   use `log_activity()` (no run_id needed). No silent failures.

2. **Thread-safe by default.** `self._lock` protects `_traces` and
   `_activities`. Bus callbacks, worker threads, and main loop can
   all record simultaneously without crashes.

3. **Verbosity is a first-class concept.** `ProvenanceVerbosity.OFF`
   means zero overhead (all guards short-circuit, `begin_trace()`
   returns None). `COMPACT` gives quick debugging. `VERBOSE` gives
   full detail. Matches existing structured logging verbosity system.

4. **Decision recording stays in MemoryAgent.** Instead of
   instrumenting LLMWorker (which lacks run_id), record the decision
   when MemoryAgent processes the proposal result. Same data,
   correct context.

5. **Debug logging on missed traces.** `collector.record()` logs a
   DEBUG message when no trace exists for a run_id, instead of
   silently no-oping. Makes incomplete provenance discoverable.

6. **ID-text linking via A7.0a-b.** ProvenanceRef carries both
   `id` and `label`. After A7.0a-b, all source records include
   their ID in `to_context_dict()`. Same ID flows through concept
   resolution → bio operations → provenance trace → user output.

7. **Wire via `wire_provenance()`.** Follows `wire_memory_hub()`
   pattern. Reaches into MemoryHub to wire Hippocampus,
   ConceptExtractor, ConceptGrounder, NAc — no wiring gaps.

8. **Session-aware persistence.** Each Maxim run gets a UUID
   session_id. Per-session JSONL files enable cross-run queries
   without scanning a monolithic file. Manifest uses atomic writes.

9. **ExplainTool registered in agentic_runtime.** Not via ExecAgent
   (which has no `register_tool()`) and not via MaximAgent (which has
   no `_tool_registry`). The `registry` local variable in
   `agentic_runtime.py:301` is the only place with ToolRegistry access.

10. **Activities persisted immediately.** `log_activity()` writes to
    JSONL on each call (flush, no fsync — balanced crash safety vs
    performance). Traces are persisted on `complete_trace()`.

---

## Bugs Fixed from v3

| Bug | Fix |
|---|---|
| `begin_trace()` allocates dummy trace when OFF | Returns `None` instead. Callers guard with `if trace:` |
| `sleep()` returns `dict[str, int]`, not lists | P3f uses `results["promoted"]` not `len(promoted)` |
| ExplainTool registered via nonexistent `exec_agent.register_tool()` | Uses `tool_registry.register()` directly |
| ProvenanceStore uses unsafe `open(path, "a")` | Per-session JSONL with flush; manifest uses atomic tmp+replace |
| Activity log lost on shutdown | `on_session_end()` flushes + `log_activity()` persists immediately |
| No session identity for cross-run queries | `session_id` on collector, traces, activities, and JSONL filenames |
| `ProvenanceTrace.to_dict()` has no `from_dict()` | All types gain `from_dict()` classmethods |
| P2b: `self._tool_registry` doesn't exist on MaximAgent | ExplainTool registered via `registry` local var in agentic_runtime.py:301, not in wire_provenance() |
| P2b: `self._current_run_id` doesn't exist on MemoryAgent | `complete_trace(run_id)` uses the `run_id` parameter from `_complete_forming_memory(run_id, outcome)` |
| P2b: wire_provenance() assumes wire_memory_hub() was called | MemoryHub now created and wired in agentic_runtime.py (A7.0c done). Tier 2 fully functional |
| P2: `record()` race condition on `trace.add()` outside lock | Per-trace lock on ProvenanceTrace — `trace.add()` and `trace.complete()` self-protect |
| P2: `on_session_end()` double-writes completed traces | `_persisted` flag on trace; `on_session_end()` only writes unpersisted traces |
| P2: `on_session_end()` holds lock during file I/O | Collect trace refs under lock, release lock, then write outside |
| P3a-c: References nonexistent variables (`concept_context`, `memory`, `predictions`) | Access via `self._forming_pool[run_id]` and method parameters (`decision`, `outcome`) |
| P2b/P3: `_collector` dynamically injected with no `__init__` declaration | Added `self._collector = None` to MemoryAgent, Hippocampus, ConceptExtractor, ConceptGrounder, NAc |
| A7.0d: `_resolve_bio_systems()` looks for `_hippocampus` etc. on Maxim | Resolves via `_memory_hub` (MemoryHub public attrs) — **IMPLEMENTED** in protocol.py |
| A7.5: `ProtocolRegistry.all_skills()` doesn't exist | **IMPLEMENTED** — `all_skills()` method added to ProtocolRegistry |

### Additional bugs fixed in v6

| Bug | Fix |
|---|---|
| P2b: `config.get("provenance")` — no central config dict exists | Uses `MAXIM_PROVENANCE_VERBOSITY` env var / `getattr(self, ...)` |
| P2b: `registry` referenced at line 213 | Corrected to line 301 (`build_tool_registry()` call) |
| P6: `_current_file: open \| None` — `open` is a function, not a type | Changed to `Any` with comment |
| P6: `query_concept()` sets `record["_session_id"]`, ExplainTool reads `record["session_id"]` | Normalized to `"session_id"` (no underscore prefix) |
| P1: ProvenanceTrace has no per-trace lock | Added `_lock: threading.Lock` field, wrapped `add()` and `complete()` |
| P1: ProvenanceTrace has no `_persisted` flag | Added `_persisted: bool = False` field, set in `complete_trace()` and `on_session_end()` |
| P7: `build_context()` line reference ~999 | Corrected to line ~957 |
| P3a: Concept context provenance "using most recent run_id from _forming_pool" | `build_context()` has no run_id; concept context provenance uses Tier 2 `log_activity()` instead |
| P3a: `_forming_pool.get(run_id)` called before entry added to pool | Moved instrumentation after `self._forming_pool[run_id] = entry`; use local `entry` var directly |
| P3b: Two `record()` calls create duplicate DECISION entries | Merged into single `record()` with conditional metadata dict |
| P3f: Provenance inside `_sleep()` holds hippocampus write lock during I/O | Moved to `sleep()` wrapper, after rwlock is released |
| P5: `query="current"` and `query="recent"` were identical | "current" now checks in-progress traces first; "recent" returns most recent completed |
| P6: `write_session_summary()` closes file prematurely | Removed close from summary; added explicit `close()` method called by `on_session_end()` |
| P2b: `_provenance_collector` not cleaned up in `_stop_agentic_runtime()` | Added `self._provenance_collector = None` to cleanup |
| P1: `_persisted` and `_lock` fields participate in dataclass `__eq__` | Added `compare=False` to both fields |

---

## Tests Needed

### Core Types (P1)
- PipelineStage has all 9 stages, ProvenanceVerbosity has 3 levels
- ProvenanceRef.__str__() and .short() format correctly
- ProvenanceRef.to_dict() / from_dict() round-trip
- ProvenanceEntry.to_dict() / from_dict() round-trip
- ProvenanceTrace.add/complete/to_dict/from_dict work correctly
- ProvenanceTrace includes session_id in to_dict()

### Collector (P2)
- ProvenanceCollector is thread-safe (concurrent begin_trace + record)
- Collector.begin_trace() returns None when verbosity=OFF
- Collector.record() logs DEBUG when trace not found (not silent)
- Collector.record() is no-op when verbosity=OFF
- Collector.log_activity() stores entries without run_id
- Collector.log_activity() persists immediately if store wired
- Collector._evict_old_traces() removes oldest under lock
- Collector.on_session_end() writes summary and flushes

### Instrumentation (P3)
- Tier 1: MemoryAgent records recall/decision/outcome with correct run_id
- Tier 1: P3a recall instrumentation runs AFTER pool assignment (not before)
- Tier 1: P3b decision records alternatives in metadata dict, not separate entry
- Tier 2: ConceptExtractor logs formation activity
- Tier 2: ConceptGrounder logs enrichment activity
- Tier 2: Hippocampus logs consolidation OUTSIDE rwlock (in sleep(), not _sleep())
- Tier 2: NAc logs learning activity (uses hasattr guard)

### Rendering (P4)
- Compact render: one line per stage, layer:id refs, no metadata
- Verbose render: timestamps, latencies, alternatives, session_id
- Summary render: tabular multi-cycle overview with activity digest
- Session report: full export with concept evolution table

### ExplainTool (P5)
- ExplainTool returns compact/verbose trace based on verbosity param
- ExplainTool returns summary when query="summary"
- ExplainTool includes activities in verbose mode
- ExplainTool returns session export when query="export"
- ExplainTool returns concept history when query="concept:kitchen"
- ExplainTool handles missing store gracefully
- ExplainTool query="current" returns in-progress trace if one exists
- ExplainTool query="recent" returns most recent completed trace
- ExplainTool query="current" falls back to recent when no in-progress trace

### Store (P6)
- ProvenanceStore creates per-session JSONL files
- write_trace() appends to correct session file
- write_activity() appends with type discriminator
- write_session_summary() writes summary + updates manifest atomically
- query_concept() finds matches across sessions
- query() filters by concept_name/tool_name/success
- load_session() / load_recent_sessions() return correct data
- Manifest survives crash (atomic writes)
- Store.close() is safe to call multiple times
- Store.close() called by collector.on_session_end() after summary write

### Integration (P7, P8)
- StructuredContext.provenance_context populated by MemoryAgent
- Verbosity guard pattern: OFF skips, COMPACT records summaries, VERBOSE records all
- export_json produces valid JSON with traces and activities
- Session history lists recent sessions from manifest
