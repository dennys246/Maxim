# Memory Unification Plan

Eliminates the `MemoryItem` / `MemoryRecord` type split by making
`MemoryRecord` the universal base class and replacing `MemoryItem` with
a thin agent-level wrapper. Introduces staged memory formation where
episodic memories are constructed incrementally through the pipeline
and held in active working memory during formation, with a pattern
completion hook that ATL can wire into for predictive context.

**Scope:** `MemoryItem` is used in only 3 files (`bus.py`, `__init__.py`,
`memory_agent.py`). The refactor is contained.

---

## Problem

### Two parallel memory representations

Maxim has two memory type hierarchies that don't share an inheritance chain:

```
MemoryRecord (ABC) — types.py
  ├── EpisodicMemory        (Hippocampus)
  ├── MathMemory             (Angular Gyrus)
  ├── SemanticMemory/Concept (ATL)
  └── CompressedRecord       (compressed forms)

MemoryItem — bus.py (standalone dataclass)
  └── Used only by MemoryAgent
```

`MemoryRecord` is well-structured: `id`, `timestamp`, access tracking,
`long_term`, `touch()`, `to_dict()` / `from_dict()`. Every memory layer
(Hippocampus, AG, ATL) stores `MemoryRecord` subclasses.

`MemoryItem` is a grab-bag: `content: Any`, `salience: float`,
`keywords: set[str]`, `tier: MemoryTier`. It duplicates fields that
already exist on EpisodicMemory (salience lives in `perception.salience`,
objects live in `perception.detected_objects`) but in unstructured form.

### Consequences

1. **MemoryAgent can't access structured data.** When MemoryAgent builds
   context, it works with `MemoryItem.content` (raw dicts). It can't
   read `perception.detected_objects` or `action.execution_time_ms`
   because those are `EpisodicMemory` fields, and MemoryAgent doesn't
   know about `EpisodicMemory`.

2. **No piggybacking for concept grounding.** The ATL concept system
   needs loaded `EpisodicMemory` objects to compute numerical properties
   via AG. MemoryAgent has memories loaded but in the wrong type. It must
   re-load them from Hippocampus — redundant I/O for no reason.

3. **MemoryAgent and Hippocampus are fully disconnected.** MemoryAgent
   doesn't import or reference Hippocampus. They're coordinated by
   MaximAgent/MemoryHub but never share data. Percepts are stored
   twice: once as `MemoryItem` in MemoryAgent, once as `EpisodicMemory`
   in Hippocampus.

4. **New memory types can't flow through MemoryAgent.** If AG computes
   a `MathMemory` result relevant to the current context, MemoryAgent
   can't include it in its working set — it only knows `MemoryItem`.
   Same for ATL `Concept` records.

5. **Duplicate field maintenance.** When MemoryAgent creates a
   `MemoryItem` from a percept, it manually extracts salience, keywords,
   and content into flat fields. These are already structured in the
   `Perception`, `Context`, `Decision`, `Action`, `Outcome` sub-objects
   that Hippocampus stores. Any new field added to EpisodicMemory must
   be separately mirrored into MemoryItem's content dict.

6. **No memory formation model.** Currently memories spring into existence
   as complete objects. There's no representation of a memory being
   *formed* — a state where perception and context are known but the
   decision, action, and outcome haven't happened yet. This prevents
   pattern-matching against similar past experiences during decision-making.

### What MemoryItem provides that MemoryRecord doesn't

| MemoryItem field | Equivalent in MemoryRecord/EpisodicMemory | Status |
|------------------|-------------------------------------------|--------|
| `content: Any` | Entire EpisodicMemory structure | Redundant — the record IS the content |
| `salience: float` | `EpisodicMemory.perception.salience` | Redundant for episodic; useful as agent-level ranking score |
| `decay_rate: float` | Not in MemoryRecord | Agent-level concern (working memory management) |
| `keywords: set[str]` | Extractable from EpisodicMemory fields | Agent-level cache (search optimization) |
| `embedding: list[float] \| None` | Not in MemoryRecord | Agent-level concern (similarity search) |
| `tier: MemoryTier` | `MemoryRecord.long_term: bool` | Partially redundant; tier is more expressive |
| `associations: list[str]` | Not in MemoryRecord | Agent-level concern (graph links) |
| `source: str` | Not in MemoryRecord directly | Agent-level metadata |
| `last_accessed: float` | `MemoryRecord.accessed_at` | Fully redundant |
| `access_count: int` | `MemoryRecord.access_count` | Fully redundant |
| `_cached_memory_id` | `MemoryRecord.id` | Fully redundant |

**Conclusion:** ~50% of MemoryItem's fields are redundant with MemoryRecord.
The remaining ~50% are agent-level concerns (ranking, decay, caching) that
belong in a wrapper, not a base class.

---

## Prerequisite: MemoryRecord.touch() Thread Safety (U0)

`MemoryRecord.touch()` performs `self.access_count += 1`, which is a
read-modify-write operation. With shared references between MemoryAgent
and Hippocampus, two threads could call `touch()` concurrently and lose
an increment.

**Fix:** Add a `threading.Lock` to `MemoryRecord.touch()`.

```python
@dataclass
class MemoryRecord(ABC):
    # ... existing fields ...

    _touch_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def touch(self) -> None:
        """Update access tracking (called on recall). Thread-safe."""
        with self._touch_lock:
            self.accessed_at = time.time()
            self.access_count += 1
```

**Cost:** ~1μs per acquire/release. Negligible compared to any I/O,
LLM call, or even dict lookup. You'd need millions of concurrent
touches per second to notice. This is the correct solution — robust
with zero practical performance impact.

**Serialization note:** The lock is excluded from `init`, `repr`, and
`compare`. It's not serialized — `to_dict()` doesn't include it, and
`from_dict()` gets a fresh lock via `default_factory`. No persistence
changes needed.

---

## Prerequisite: MemoryRecord.keywords() ABC Method (U0)

The original plan had `WorkingMemoryEntry._extract_keywords()` with an
`isinstance` chain that grows with every new memory type. This doesn't
scale. Instead, push keyword extraction into `MemoryRecord` as an ABC
method — each subclass knows its own searchable fields.

```python
@dataclass
class MemoryRecord(ABC):
    # ... existing fields ...

    @abstractmethod
    def keywords(self) -> set[str]:
        """Return searchable keywords for this record.

        Each subclass extracts keywords from its own structured fields.
        Called lazily by WorkingMemoryEntry and cached.
        """
        ...
```

**Subclass implementations:**

```python
# EpisodicMemory
def keywords(self) -> set[str]:
    kw: set[str] = set()
    kw.update(self.perception.detected_objects)
    kw.update(self.perception.detected_people)
    if self.context.active_goal:
        kw.add(self.context.active_goal)
    if self.action.tool_name:
        kw.add(self.action.tool_name)
    if self.perception.cli_input:
        kw.update(self.perception.cli_input.lower().split())
    return kw

# MathMemory
def keywords(self) -> set[str]:
    return {self.name, self.domain}

# SemanticMemory / Concept
def keywords(self) -> set[str]:
    return {self.name, self.category}

# CompressedMemory
def keywords(self) -> set[str]:
    kw: set[str] = set()
    if self.goal:
        kw.add(self.goal)
    if self.tool_name:
        kw.add(self.tool_name)
    return kw
```

`WorkingMemoryEntry` then simply delegates:

```python
@property
def keywords(self) -> set[str]:
    if self._keywords is None:
        self._keywords = self.record.keywords()
    return self._keywords
```

Zero isinstance checks. New memory types implement `keywords()` once,
and they automatically work everywhere — WorkingMemoryEntry, search,
context building.

### Prerequisite: MemoryRecord.to_context_dict() ABC Method (U0)

Same pattern as `keywords()` — push context extraction into the record
so `build_context()` doesn't need isinstance chains to access
type-specific fields:

```python
@dataclass
class MemoryRecord(ABC):
    # ... existing fields ...

    @abstractmethod
    def to_context_dict(self) -> dict[str, Any]:
        """Return a dict representation suitable for LLM context.

        Each subclass formats its own structured data for inclusion
        in StructuredContext. Unlike to_dict() (which is for persistence),
        this returns a human/LLM-readable summary.
        """
        ...
```

**Subclass implementations:**

```python
# EpisodicMemory
def to_context_dict(self) -> dict[str, Any]:
    d: dict[str, Any] = {
        "type": "episodic",
        "timestamp": self.timestamp,
    }
    if self.perception.detected_objects:
        d["objects"] = self.perception.detected_objects
    if self.perception.detected_people:
        d["people"] = self.perception.detected_people
    if self.perception.cli_input:
        d["user_input"] = self.perception.cli_input
    if self.context.active_goal:
        d["goal"] = self.context.active_goal
    if self.action.tool_name:
        d["action"] = self.action.tool_name
    if self.outcome.success is not None:
        d["success"] = self.outcome.success
    return d

# MathMemory
def to_context_dict(self) -> dict[str, Any]:
    return {
        "type": "math",
        "name": self.name,
        "domain": self.domain,
        "verbal": self.verbal,
        "confidence": self.confidence,
    }

# SemanticMemory / Concept
def to_context_dict(self) -> dict[str, Any]:
    return {
        "type": "concept",
        "name": self.name,
        "category": self.category,
    }

# CompressedMemory
def to_context_dict(self) -> dict[str, Any]:
    d: dict[str, Any] = {"type": "compressed", "timestamp": self.timestamp}
    if self.goal:
        d["goal"] = self.goal
    if self.tool_name:
        d["action"] = self.tool_name
    d["success"] = self.success
    return d
```

`build_context()` then becomes:

```python
for entry in relevant_memories:
    context_items.append(entry.record.to_context_dict())
```

No isinstance chains for context building OR keyword extraction.
New memory types implement two methods (`keywords()`, `to_context_dict()`)
and plug in everywhere automatically.

---

## Design

### MemoryTier: Enriched Working Memory Phases

The current `MemoryTier` is a simple SHORT_TERM/LONG_TERM enum that
maps 1:1 to `MemoryRecord.long_term: bool` — it adds no value.

Replace with a richer enum that captures the full lifecycle of a memory
in working memory, including the formation phase:

```python
class MemoryTier(Enum):
    """Working memory lifecycle phase.

    Tracks where a memory sits in the formation → active → long-term
    pipeline. Used by MemoryAgent for eviction policy, context building,
    and formation protection.

    Consolidation is NOT a tier — it's an active process that promotes
    memories through these tiers and eventually into Hippocampus
    long-term storage. A consolidated memory gets marked long_term in
    Hippocampus and evicted from working memory (it's safely stored).
    """

    FORMING = "forming"
    # Memory is being constructed through the pipeline.
    # Perception and context are set, decision/action/outcome pending.
    # PROTECTED from eviction — cannot be decayed or removed.
    # Pattern matching against similar past episodes occurs here.

    WORKING = "working"
    # Fully formed memory actively relevant to the current context.
    # Recently completed pipeline cycle, still in active use.
    # Subject to salience-based ranking but not decay eviction.

    SHORT_TERM = "short"
    # Complete memory in the short-term buffer.
    # Subject to normal decay. Oldest entries evicted when buffer full.
    # Can be promoted to LONG_TERM based on access count or salience.

    LONG_TERM = "long"
    # Promoted memory resistant to eviction.
    # Survives buffer pressure. Still subject to age-based eviction
    # if not accessed within max_age_seconds.
    # Consolidation process may mark record.long_term = True in
    # Hippocampus and then evict from working memory.
```

**Tier transitions:**

```
FORMING → WORKING    (pipeline completes: outcome received)
WORKING → SHORT_TERM (_begin_memory_formation() called for next cycle —
                       all WORKING entries in the pool transition to SHORT_TERM)
SHORT_TERM → LONG_TERM (access_count >= threshold OR salience >= threshold)
LONG_TERM → evicted   (consolidation marks record.long_term in Hippocampus,
                        entry removed from working memory — safely persisted)
```

**WORKING → SHORT_TERM trigger:** When `_begin_memory_formation()` starts
a new pipeline cycle, it sweeps the forming pool and transitions any
WORKING entries to SHORT_TERM (moving them into `_short_term` deque).
This ensures WORKING entries don't accumulate indefinitely — they get
exactly one cycle of protection after completion before entering normal
decay.

**Eviction rules by tier:**

| Tier | Evictable? | Condition |
|------|------------|-----------|
| FORMING | Never | Protected until pipeline completes |
| WORKING | Never | Active context, always retained |
| SHORT_TERM | Yes | Buffer full (oldest first) or salience decayed below threshold |
| LONG_TERM | Yes | Not accessed within `max_age_seconds`, OR consolidation process actively moves it out |

**Consolidation as an active process:**

Consolidation is triggered during sleep or idle periods. It examines
LONG_TERM entries and decides which to persist permanently:

1. Marks `record.long_term = True` in Hippocampus
2. Forms associative graph edges (if not already formed)
3. Evicts the entry from MemoryAgent's working set
4. The memory now lives solely in Hippocampus, recallable on demand

This means working memory naturally clears itself: memories flow
FORMING → WORKING → SHORT_TERM → LONG_TERM → consolidated out.
The agent's working set stays bounded while important memories
accumulate in Hippocampus.

**Integration points:**

- `_process_percept()`: creates entry with `tier=FORMING`
- `_on_pipeline_complete()`: transitions FORMING → WORKING
- `_on_new_cycle()`: transitions WORKING → SHORT_TERM
- `_promote()`: transitions SHORT_TERM → LONG_TERM
- `_consolidate()`: marks LONG_TERM records in Hippocampus, evicts from working memory
- `_evict()`: skips FORMING and WORKING entries entirely
- `build_context()`: prioritizes FORMING/WORKING entries (current episode)

### WorkingMemoryEntry: thin wrapper around any MemoryRecord

Replace `MemoryItem` with a generic wrapper that holds any `MemoryRecord`
subclass plus agent-level working memory metadata:

```python
@dataclass
class WorkingMemoryEntry(Generic[T]):
    """Agent-level wrapper around a MemoryRecord for working memory.

    Holds the actual structured record (EpisodicMemory, MathMemory,
    Concept, etc.) plus agent-level metadata for working memory
    management: salience ranking, decay, lifecycle phase.

    The record itself is the canonical type stored in its memory layer
    (Hippocampus, AG, ATL). During FORMING phase, the record's
    decision/action/outcome fields are populated incrementally as the
    pipeline progresses.
    """

    record: T  # The actual MemoryRecord subclass
    salience: float = 0.5  # Agent-level ranking score
    decay_rate: float = 0.1
    source: str = "percept"  # "percept", "goal_outcome", "user_feedback", "inference", "ag_grounding"
    tier: MemoryTier = MemoryTier.SHORT_TERM

    # Pattern completion results (populated during FORMING phase)
    predicted_outcomes: list[PredictedOutcome] | None = None  # Similar past outcomes
    prediction_confidence: float = 0.0  # How confident the pattern match is

    # Cached for search performance (extracted lazily from record)
    _keywords: set[str] | None = field(default=None, repr=False)
    _embedding: list[float] | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def timestamp(self) -> float:
        return self.record.timestamp

    @property
    def is_protected(self) -> bool:
        """FORMING and WORKING entries cannot be evicted."""
        return self.tier in (MemoryTier.FORMING, MemoryTier.WORKING)

    def touch(self) -> None:
        """Delegate access tracking to the underlying record. Thread-safe."""
        self.record.touch()

    @property
    def keywords(self) -> set[str]:
        """Lazily extract keywords from the underlying record."""
        if self._keywords is None:
            self._keywords = self.record.keywords()
        return self._keywords

    def invalidate_keywords(self) -> None:
        """Clear keyword cache after record mutation (e.g., FORMING → complete)."""
        self._keywords = None

    def should_promote(
        self, access_threshold: int = 3, salience_threshold: float = 0.7
    ) -> bool:
        """Check if this entry should be promoted to long-term."""
        if self.tier != MemoryTier.SHORT_TERM:
            return False
        return (
            self.record.access_count >= access_threshold
            or self.salience >= salience_threshold
        )

    def should_evict(self, max_age_seconds: float) -> bool:
        """Check if this entry should be evicted from working memory."""
        if self.is_protected:
            return False
        if self.tier == MemoryTier.SHORT_TERM:
            return False  # SHORT_TERM eviction is buffer-based, not age-based
        # LONG_TERM: age-based eviction (consolidation handles active removal)
        age = time.time() - self.record.accessed_at
        return age > max_age_seconds

    def current_salience(self) -> float:
        """Compute decayed salience based on age. FORMING entries don't decay."""
        if self.is_protected:
            return self.salience
        age = time.time() - self.timestamp
        return self.salience * math.exp(-self.decay_rate * age / 60.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize wrapper + record for persistence."""
        return {
            "record_type": type(self.record).__name__,
            "record": self.record.to_dict(),
            "salience": self.salience,
            "decay_rate": self.decay_rate,
            "source": self.source,
            "tier": self.tier.value,
            "predicted_outcomes": (
                [p.to_dict() for p in self.predicted_outcomes]
                if self.predicted_outcomes else None
            ),
            "prediction_confidence": self.prediction_confidence,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        record_registry: dict[str, type[MemoryRecord]] | None = None,
    ) -> WorkingMemoryEntry:
        """Deserialize wrapper + record from persistence.

        Args:
            data: Serialized dict from to_dict().
            record_registry: Maps type names to classes for deserialization.
                Defaults to built-in types (EpisodicMemory, MathMemory, etc.).
        """
        from maxim.math.angular_gyrus import MathMemory
        from maxim.memory.atl import SemanticMemory

        registry = record_registry or {
            "EpisodicMemory": EpisodicMemory,
            "MathMemory": MathMemory,
            "SemanticMemory": SemanticMemory,
            "CompressedMemory": CompressedMemory,
        }

        record_type_name = data["record_type"]
        record_cls = registry.get(record_type_name)
        if record_cls is None:
            raise ValueError(f"Unknown record type: {record_type_name}")

        record = record_cls.from_dict(data["record"])

        return cls(
            record=record,
            salience=data.get("salience", 0.5),
            decay_rate=data.get("decay_rate", 0.1),
            source=data.get("source", "unknown"),
            tier=MemoryTier(data.get("tier", "short")),
            predicted_outcomes=(
                [PredictedOutcome.from_dict(p) for p in data["predicted_outcomes"]]
                if data.get("predicted_outcomes") else None
            ),
            prediction_confidence=data.get("prediction_confidence", 0.0),
        )
```

---

## Staged Memory Formation

### The pipeline lifecycle of an EpisodicMemory

Currently, `EpisodicMemory` is constructed all at once by `agent_loop.py`
after tool execution via `capture_from_loop_async()`. All five sub-objects
(Perception, Context, Decision, Action, Outcome) are populated in a single
call. This means the memory doesn't exist in working memory *while the
pipeline is processing it*.

After unification, EpisodicMemory is created incrementally and held in
active working memory throughout:

```
Pipeline Stage         EpisodicMemory State              Tier
─────────────────────────────────────────────────────────────────
1. Percept arrives     Perception ✓  Context ✓           FORMING
                       Decision ∅    Action ∅  Outcome ∅
                       → Graph chaining for pattern completion (ATL plan)
                       → Predicted outcomes attached to entry

2. Decision made       Perception ✓  Context ✓           FORMING
                       Decision ✓    Action ∅  Outcome ∅

3. Action executes     Perception ✓  Context ✓           FORMING
                       Decision ✓    Action ✓  Outcome ∅

4. Outcome received    Perception ✓  Context ✓           FORMING → WORKING
                       Decision ✓    Action ✓  Outcome ✓
                       → Invalidate keyword cache (new fields populated)

5. New cycle begins    (previous entry)                   WORKING → SHORT_TERM
```

**Key property:** FORMING entries are eviction-protected. The memory
persists in active working memory throughout the entire pipeline, including
during action execution and outcome processing. It cannot be decayed out
or bumped by buffer pressure.

### Pattern completion hook

Pattern completion (graph chaining through ATL → Hippocampus → AG → future
layers) is defined in the ATL concept plan, not here. This plan provides
the infrastructure that pattern completion plugs into:

- `PredictedOutcome` and `MathContextEntry` dataclasses — typed contract for predictions (defined in `types.py`)
- `WorkingMemoryEntry.predicted_outcomes: list[PredictedOutcome]` — stores results
- `WorkingMemoryEntry.prediction_confidence` — how reliable the prediction is
- `_forming_pool` — where FORMING entries live while awaiting completion
- `_pattern_completion_fn: Callable[[EpisodicMemory], list[PredictedOutcome]] | None`
  — optional callable set by ATL/MemoryHub wiring

MemoryAgent calls the hook during formation if wired:

```python
# Pattern completion is an optional hook, implemented in ATL concept plan
if self._pattern_completion_fn:
    entry.predicted_outcomes = self._pattern_completion_fn(episodic)
    if entry.predicted_outcomes:
        entry.prediction_confidence = self._compute_prediction_confidence(
            entry.predicted_outcomes
        )
```

See [atl_concept_memory_plan.md](atl_concept_memory_plan.md) Phase A5 for
the graph chaining implementation that provides this hook.

### Context construction from agentic state

`_begin_memory_formation` constructs `Context` from the current agentic
state available on the bus, rather than requiring it as a parameter:

```python
def _begin_memory_formation(self, percept: Percept, run_id: str) -> WorkingMemoryEntry:
    """Create a FORMING EpisodicMemory from percept + current agentic state."""
    now = time.time()

    # Sweep WORKING entries from pool → SHORT_TERM
    self._flush_working_to_short_term()

    # Build Context from current agentic state (available on bus)
    context = Context(
        active_goal=self._bus.current_goal if self._bus else None,
        active_mode=self._bus.current_mode if self._bus else "observe",
        fear_level=self._bus.fear_level if self._bus else 0.0,
    )

    episodic = EpisodicMemory(
        id=str(uuid4()),
        timestamp=now,
        run_id=run_id,
        perception=Perception(
            detected_objects=percept.detected_objects,
            detected_people=percept.detected_people,
            salience=percept.salience,
            novelty=percept.novelty,
            cli_input=percept.raw_transcript_text,
            observations=percept.metadata,
        ),
        context=context,
        # Decision, Action, Outcome left as defaults (empty)
    )

    entry = WorkingMemoryEntry(
        record=episodic,
        salience=percept.salience,
        source="percept",
        tier=MemoryTier.FORMING,
    )

    # Pattern completion hook (implemented in ATL concept plan)
    if self._pattern_completion_fn:
        entry.predicted_outcomes = self._pattern_completion_fn(episodic)
        if entry.predicted_outcomes:
            entry.prediction_confidence = self._compute_prediction_confidence(
                entry.predicted_outcomes
            )

    self._forming_pool[run_id] = entry
    return entry

def _flush_working_to_short_term(self) -> None:
    """Transition WORKING entries in the pool to SHORT_TERM, move to _short_term."""
    to_remove = []
    for run_id, entry in self._forming_pool.items():
        if entry.tier == MemoryTier.WORKING:
            entry.tier = MemoryTier.SHORT_TERM
            self._short_term.appendleft(entry)  # Move to short-term buffer
            to_remove.append(run_id)
    for run_id in to_remove:
        del self._forming_pool[run_id]

def _compute_prediction_confidence(self, predictions: list[PredictedOutcome]) -> float:
    """Compute confidence from success rate, action consistency, and sample size.

    High confidence requires: high success rate, low action diversity,
    AND sufficient sample size. A single matching episode cannot produce
    high confidence regardless of outcome.
    """
    if not predictions:
        return 0.0

    n = len(predictions)

    # Success rate
    successes = sum(1 for p in predictions if p.success)
    success_rate = successes / n

    # Action consistency: what fraction used the most common action?
    action_counts: dict[str, int] = {}
    for p in predictions:
        action_counts[p.tool] = action_counts.get(p.tool, 0) + 1
    most_common_count = max(action_counts.values()) if action_counts else 0
    consistency = most_common_count / n

    # Sample size dampening: confidence is low until ~5 matching episodes
    sample_factor = min(n / 5.0, 1.0)

    return success_rate * consistency * sample_factor
```

### Entry lifecycle: single-track ownership

FORMING/WORKING entries live **only** in `_forming_pool`. They do NOT
exist in `_short_term` or `_long_term` simultaneously. When an entry
transitions to SHORT_TERM, it moves out of the pool and into
`_short_term`. This makes ownership explicit:

```
_forming_pool  →  _short_term  →  _long_term
(FORMING/WORKING)  (SHORT_TERM)    (LONG_TERM)
```

`_get_relevant_memories()` combines all three sources when building
context, but each entry exists in exactly one location at any time.

### Filling in Decision, Action, Outcome

As the pipeline progresses, the FORMING entry's record gets updated.
Methods accept `run_id` to look up the entry in the forming pool:

```python
def _update_forming_decision(self, run_id: str, decision: Decision) -> None:
    """Fill in the decision on a FORMING episodic memory."""
    entry = self._forming_pool.get(run_id)
    if entry is None or entry.tier != MemoryTier.FORMING:
        return
    assert isinstance(entry.record, EpisodicMemory)
    entry.record.decision = decision
    entry.invalidate_keywords()

def _update_forming_action(self, run_id: str, action: Action) -> None:
    """Fill in the action on a FORMING episodic memory."""
    entry = self._forming_pool.get(run_id)
    if entry is None or entry.tier != MemoryTier.FORMING:
        return
    assert isinstance(entry.record, EpisodicMemory)
    entry.record.action = action
    entry.invalidate_keywords()

def _complete_forming_memory(self, run_id: str, outcome: Outcome) -> None:
    """Fill in the outcome and transition FORMING → WORKING.

    Entry stays in _forming_pool as WORKING until next cycle sweeps it
    into _short_term via _flush_working_to_short_term().
    """
    entry = self._forming_pool.get(run_id)
    if entry is None or entry.tier != MemoryTier.FORMING:
        return
    assert isinstance(entry.record, EpisodicMemory)
    entry.record.outcome = outcome
    entry.tier = MemoryTier.WORKING
    entry.invalidate_keywords()
```

**Coexistence with agent_loop capture:** MemoryAgent does NOT persist
to Hippocampus directly. `agent_loop.py` continues calling
`capture_from_loop_async()` independently — this preserves all
Hippocampus capture hooks (association formation, consolidation candidate
queueing, immediate promotion). The two systems maintain separate
EpisodicMemory instances representing the same event: one ephemeral
in working memory, one persisted in Hippocampus. This avoids duplicate
entries and keeps the Hippocampus capture machinery untouched.

**Mutability contract:** EpisodicMemory fields are mutable *only* during
the FORMING phase (one pipeline cycle). Once the entry transitions to
WORKING, the record is effectively immutable — only `touch()` mutates
access tracking. Since MemoryAgent's EpisodicMemory is a separate
instance from Hippocampus's copy, there are no shared-reference
mutation concerns during FORMING.

### How pattern completion feeds into decisions

The `predicted_outcomes` on a FORMING entry are available to ExecAgent
when it builds context for the LLM:

```python
# In build_context(), FORMING entries contribute prediction context:
for entry in self._get_forming_and_working():
    if entry.tier == MemoryTier.FORMING and entry.predicted_outcomes:
        context["pattern_predictions"] = {
            "similar_past_outcomes": [
                {"tool": p.tool, "success": p.success, "goal": p.goal,
                 "math_context": [m.to_dict() for m in p.math_context]
                     if p.math_context else None}
                for p in entry.predicted_outcomes
            ],
            "prediction_confidence": entry.prediction_confidence,
        }
```

This gives the LLM a "here's what happened in similar situations" signal
*during decision-making*, enabling it to make more informed choices. The
prediction confidence tells it how reliable the historical pattern is.

---

## What changes in MemoryAgent

MemoryAgent's internal storage changes from `MemoryItem` to
`WorkingMemoryEntry`:

```python
# Before:
self._short_term: deque[MemoryItem] = deque(maxlen=max_short_term)
self._long_term: list[MemoryItem] = []

# After:
self._short_term: deque[WorkingMemoryEntry] = deque(maxlen=max_short_term)
self._long_term: list[WorkingMemoryEntry] = []
self._forming_pool: dict[str, WorkingMemoryEntry] = {}  # Keyed by run/pipeline ID
```

The `_forming_pool` (pool working memory) tracks all FORMING and WORKING
entries by their pipeline/run ID. This supports concurrent pipeline
cycles (e.g., vision and CLI percepts arriving simultaneously). Each
entry is keyed by a unique run ID so multiple formation processes can
coexist without clobbering each other.

When a new pipeline cycle begins (`_begin_memory_formation()`), it:
1. Sweeps the pool for WORKING entries → transitions them to SHORT_TERM
   and moves them into `_short_term`
2. Creates a new FORMING entry in the pool

When a pipeline completes (`_complete_forming_memory()`), it transitions
the entry from FORMING → WORKING within the pool. It stays in the pool
until the next cycle sweeps it out.

### What changes in MemoryAgent's helper classes

**SalientMemoryStore** (memory_agent.py):
- `_memories: dict[str, MemoryItem]` -> `dict[str, WorkingMemoryEntry]`
- `add()`, `query()`, `build_associations()` accept `WorkingMemoryEntry`
- Keyword extraction delegates to `entry.keywords` (lazy, cached via record)

**MemoryAssociationGraph** (memory_agent.py):
- `DependencyGraph[MemoryItem]` -> `DependencyGraph[WorkingMemoryEntry]`
- `add_memory()`, `associate()`, `query_associations()` accept entries

**build_context()** (memory_agent.py):
- `_get_relevant_memories()` returns `list[WorkingMemoryEntry]`
- Can now access `entry.record` for structured data
- FORMING entries contribute pattern predictions to context
- ConceptGrounder can piggyback: `entry.record` IS an `EpisodicMemory`

### Hippocampus integration (coexistence model)

MemoryAgent and Hippocampus coexist as parallel consumers of the same
pipeline events. They maintain separate `EpisodicMemory` instances:

- **MemoryAgent:** Creates its own `EpisodicMemory` during FORMING,
  fills in stages, uses it for working memory and pattern completion.
  This instance is ephemeral — it lives in the forming pool and
  short-term buffer, subject to normal decay and eviction.

- **agent_loop.py + Hippocampus:** Continues calling
  `capture_from_loop_async()` after tool execution, creating its own
  `EpisodicMemory` for persistent storage. All capture hooks
  (association formation, consolidation candidates, immediate promotion)
  fire as before — zero changes to Hippocampus.

**Why coexistence over shared reference:** Shared references would
require MemoryAgent to replace `capture_from_loop_async()`, which
means reimplementing all Hippocampus capture hooks. That's high-risk
refactoring for marginal gain. Coexistence gives MemoryAgent structured
types in working memory (the real win) without touching the capture
pipeline. The cost is two EpisodicMemory instances per event — ~2KB
each, negligible.

**Future optimization:** Once the system is stable, shared references
can be introduced by having MemoryAgent pass its completed FORMING
record to Hippocampus's `capture_record()` instead of agent_loop
creating a separate one. This is a clean optimization, not a
prerequisite.

### MemoryAgent gets memory system references

Currently MemoryAgent doesn't know about Hippocampus. After
unification, MemoryHub wires the pattern completion hook:

```python
# In MemoryHub or MaximAgent.wire_memory_hub():
memory_agent.connect_hippocampus(hippocampus)

# Pattern completion hook — implemented in ATL concept plan (Phase A5)
memory_agent.set_pattern_completion_fn(atl.pattern_complete)
```

This enables:
- Pattern completion during FORMING via the `_pattern_completion_fn` hook
- ConceptGrounder to access episodes through MemoryAgent's working set
- `build_context()` to work with structured types via `to_context_dict()`
- Graph chaining (ATL → Hippocampus → AG) lives in the ATL concept plan

### StructuredContext improvements

With `to_context_dict()` on MemoryRecord, `build_context()` needs zero
isinstance checks:

```python
# Before (parsing MemoryItem.content dicts):
detected_objects = []
for mem in relevant_memories:
    if isinstance(mem.content, dict):
        detected_objects.extend(mem.content.get("detected_objects", []))

# After (polymorphic context extraction):
context_items = []
for entry in relevant_memories:
    context_items.append(entry.record.to_context_dict())
```

Every record type formats itself for LLM context. No isinstance chains
for keyword extraction OR context building. New memory types implement
`keywords()` and `to_context_dict()` and plug in everywhere.

---

## Multi-Type Working Memory

With `WorkingMemoryEntry` being generic, MemoryAgent can hold more than
just episodic memories in its working set:

| Record Type | When it enters working memory | Tier | Example |
|-------------|-------------------------------|------|---------|
| `EpisodicMemory` | Every perception cycle | FORMING → WORKING → SHORT_TERM | "Saw mug on table" |
| `MathMemory` | AG computes a relevant result | SHORT_TERM | "Mug grasp time: 310ms mean" |
| `Concept` | ATL concept recalled for context | SHORT_TERM | "Mug: object, related to kitchen" |

This is the foundation for the ATL concept plan's `get_concept_context()`.
When ConceptGrounder grounds a concept, the resulting stats can be wrapped
in a `WorkingMemoryEntry[MathMemory]` and added to the working set — the
LLM gets both the episodic memories AND the mathematical characterization
in a unified context.

```python
# After concept grounding produces a MathMemory for "mug:execution_time_ms":
math_entry = WorkingMemoryEntry(
    record=math_memory,
    salience=0.6,
    source="ag_grounding",
    tier=MemoryTier.SHORT_TERM,
)
self._add_memory(math_entry)
```

---

## Piggybacking and Scalability

### What piggybacking actually provides

After unification, MemoryAgent's working set contains actual
`EpisodicMemory` objects (not `MemoryItem` copies). When ConceptGrounder
needs episodes linked to a concept, it can check the working set first.

**Honest framing:** For recently-seen concepts, some linked episodes will
be in the working set — these come for free. For established concepts with
hundreds of linked episodes, most will NOT be in the working set and must
be loaded from Hippocampus. The fallback (Hippocampus lookup by ID) is
an in-memory dict lookup — trivially fast. The real win is not I/O
savings but **type unification**: ConceptGrounder receives
`EpisodicMemory` objects either way, without needing a separate loading
path.

### Scalability win

The architectural win scales with system complexity:

1. **New memory types plug in with zero MemoryAgent changes.** Define a
   `MemoryRecord` subclass with `keywords()`, wrap in
   `WorkingMemoryEntry[NewType]`, done. No isinstance chains, no
   content dict parsing, no keyword extraction logic to add.

2. **Single context-building path.** Every record type flows through the
   same `build_context()` pipeline. Structured field access replaces
   content dict key guessing. Type safety catches errors at development
   time, not runtime.

3. **Cross-type queries become natural.** "Find all working memories
   related to the current goal" works across episodic, math, and concept
   records because they all share `keywords()` on `MemoryRecord`.

4. **Formation pipeline is type-agnostic.** While only `EpisodicMemory`
   uses staged FORMING today, the tier system works for any record type
   that might need incremental construction in the future.

```python
def get_concept_context(self, percept: Percept) -> list[dict]:
    relevant = self._get_relevant_memories()

    for name in self._extract_percept_names(percept):
        concept = self._atl.recall(limit=1, name=name.lower())
        if not concept:
            continue

        concept_refs = concept.memory_refs.get("hippocampus", set())

        # Check working set first (free)
        linked_episodes = [
            entry.record for entry in relevant
            if isinstance(entry.record, EpisodicMemory)
            and entry.record.id in concept_refs
        ]

        # Supplement from Hippocampus for established concepts
        if len(linked_episodes) < 5:
            extra_ids = concept_refs - {ep.id for ep in linked_episodes}
            for eid in list(extra_ids)[:15]:
                ep = self._hippocampus.get(eid)
                if isinstance(ep, EpisodicMemory):
                    linked_episodes.append(ep)

        stats = self._concept_grounder.ground_concept(concept, linked_episodes)
        ...
```

---

## Persistence and Serialization

### WorkingMemoryEntry persistence

`WorkingMemoryEntry` supports full serialization via `to_dict()` and
`from_dict()`. The wrapper serializes its own agent-level metadata
(salience, decay_rate, source, tier, predictions) alongside the
underlying record's serialized form.

**Record type registry:** Deserialization needs to know which
`MemoryRecord` subclass to instantiate. `from_dict()` accepts an
optional `record_registry` mapping type names to classes, defaulting
to built-in types. This allows future memory types to register without
modifying `WorkingMemoryEntry`.

### What gets persisted and when

| Component | Persistence | Storage | Recovery |
|-----------|-------------|---------|----------|
| `record: T` | Full serialization via `record.to_dict()` | JSON file (same as Hippocampus/AG/ATL) | `record_cls.from_dict()` via registry |
| `salience`, `decay_rate`, `source` | Serialized with wrapper | Working memory state file | Direct restoration |
| `tier: MemoryTier` | Serialized as enum value | Working memory state file | `MemoryTier(value)` |
| `predicted_outcomes` | Serialized if present | Working memory state file | Direct restoration |
| `_keywords` cache | NOT serialized | Recomputed lazily | `record.keywords()` on first access |
| `_embedding` cache | NOT serialized | Recomputed if needed | Embedding model re-run |
| `_touch_lock` | NOT serialized | Fresh lock on deserialization | `default_factory=threading.Lock` |

### MemoryAgent save/load

MemoryAgent serializes its full working memory state:

```python
def save_state(self, path: str) -> None:
    """Persist working memory state for session recovery."""
    state = {
        "version": "2.0",  # New version for unified format
        "short_term": [e.to_dict() for e in self._short_term],
        "long_term": [e.to_dict() for e in self._long_term],
        "forming_pool": {
            run_id: e.to_dict()
            for run_id, e in self._forming_pool.items()
        },
    }
    # Atomic write: write to temp file, then rename
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, path)

def load_state(self, path: str) -> None:
    """Restore working memory state from persistence."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        state = json.load(f)

    version = state.get("version", "1.0")
    if version == "1.0":
        # Legacy MemoryItem format — skip or migrate
        return

    for item in state.get("short_term", []):
        entry = WorkingMemoryEntry.from_dict(item)
        self._short_term.append(entry)
    for item in state.get("long_term", []):
        entry = WorkingMemoryEntry.from_dict(item)
        self._long_term.append(entry)

    # FORMING entries from a crashed session → transition to SHORT_TERM
    # (the pipeline didn't complete, but partial data is still valuable)
    for run_id, forming_data in state.get("forming_pool", {}).items():
        entry = WorkingMemoryEntry.from_dict(forming_data)
        entry.tier = MemoryTier.SHORT_TERM  # Can't resume FORMING
        self._short_term.appendleft(entry)
```

### Recovery semantics

- **Clean shutdown:** All entries serialized, forming pool saved with
  current state. On reload, any FORMING/WORKING entries transition to
  SHORT_TERM (can't resume a half-complete pipeline).

- **Crash recovery:** Atomic write (`tmp` + `os.replace`) ensures the
  state file is never partially written. If crash occurs before write
  completes, the previous state file is intact.

- **Version migration:** `version: "2.0"` distinguishes from legacy
  `MemoryItem`-based state files. Legacy files are skipped (Hippocampus
  has the canonical records anyway).

- **Hippocampus as source of truth:** If MemoryAgent's state file is
  lost entirely, it starts fresh. Hippocampus retains all completed
  episodic memories. MemoryAgent rebuilds its working set naturally
  as new percepts arrive and old records are recalled.

---

## Migration Path

### What changes

| File | Change |
|------|--------|
| `types.py` | Add `keywords()` and `to_context_dict()` abstract methods to `MemoryRecord`. Implement in `EpisodicMemory`, `CompressedMemory`. Add `threading.Lock` to `touch()`. Add `PredictedOutcome` and `MathContextEntry` dataclasses. |
| `bus.py` | Keep `MemoryItem` temporarily as deprecated alias. Add `WorkingMemoryEntry`. Enrich `MemoryTier` enum with FORMING/WORKING values. |
| `memory_agent.py` | Replace `MemoryItem` usage with `WorkingMemoryEntry`. Add staged formation methods. Update helper classes. Add `connect_hippocampus()`. Update `save_state()`/`load_state()`. |
| `__init__.py` | Export `WorkingMemoryEntry` alongside (then instead of) `MemoryItem`. |
| `memory_hub.py` | Wire `memory_agent.connect_hippocampus(hippocampus)` in setup. |
| `maxim_agent.py` | Pass Hippocampus reference to MemoryAgent during wiring. Update pipeline to call formation stage methods. |
| `agent_loop.py` | Route decision/action/outcome events to MemoryAgent's staged formation methods. agent_loop continues calling `capture_from_loop_async()` independently (coexistence). |
| `angular_gyrus.py` | Add `keywords()` and `to_context_dict()` to `MathMemory`. |
| `atl.py` | Add `keywords()` and `to_context_dict()` to `SemanticMemory`. |

### What doesn't change

- `Hippocampus` — untouched (already works with `EpisodicMemory`,
  `capture_record()` accepts pre-built records)
- `AngularGyrus` core — untouched (only `MathMemory` gets `keywords()`)
- `ATL` core — untouched (only `SemanticMemory` gets `keywords()`)
- All bridges, tools, runtime — they don't use `MemoryItem`

### MathMemory — no changes needed (except keywords())

MathMemory already extends `MemoryRecord`. It has `id`, `timestamp`,
access tracking, `to_dict()` / `from_dict()`, and structured domain
fields (`name`, `category`, `domain`, `verbal`, `code`, `inputs`,
`outputs`). It does NOT need to extend `MemoryItem` — that would be
going backwards (structured type inheriting from grab-bag type).

After unification, MathMemory naturally flows through the system:
- AG stores `MathMemory` (already works)
- `WorkingMemoryEntry[MathMemory]` can enter MemoryAgent's working set
- `build_context()` can extract structured math data
- ConceptGrounder stores `MathMemory` with `QUANTIFIES` edges (ATL plan)

Only changes: add `keywords() -> set[str]` returning `{self.name, self.domain}`
and `to_context_dict()` returning name, domain, verbal, confidence.

---

## Implementation Order

| Phase | Effort | What |
|-------|--------|------|
| U0. MemoryRecord prerequisites | Medium | Add `threading.Lock` to `touch()`. Add `keywords()` and `to_context_dict()` ABC methods. Implement in EpisodicMemory, MathMemory, SemanticMemory, CompressedMemory. |
| U1. WorkingMemoryEntry + MemoryTier | Medium | Generic wrapper in bus.py with serialization. Enriched MemoryTier enum (FORMING/WORKING/SHORT_TERM/LONG_TERM). Tier transition + consolidation eviction logic. |
| U2. MemoryAgent core swap | Medium | Replace MemoryItem with WorkingMemoryEntry in _short_term/_long_term. Add _forming_pool dict. Update _add_memory(), eviction (respect is_protected), promotion. |
| U3. Staged formation + pattern hook | Medium | _begin_memory_formation() creates FORMING entry with Perception+Context from bus. _update_forming_decision/action(), _complete_forming_memory() fill in stages. _pattern_completion_fn hook (implementation in ATL plan). _compute_prediction_confidence() with action consistency + sample dampening. |
| U4. Helper class updates | Small | SalientMemoryStore, MemoryAssociationGraph accept WorkingMemoryEntry. |
| U5. Hippocampus wiring + pattern hook | Medium | connect_hippocampus() for piggybacking, set_pattern_completion_fn() for pattern completion. Coexistence model — MemoryAgent doesn't persist to Hippocampus. ATL registers Hippocampus+AG via register_layer() (ATL concept plan). |
| U6. build_context() polymorphic access | Medium | Replace content dict parsing with to_context_dict(). Add pattern prediction context for FORMING entries. |
| U7. Persistence | Medium | WorkingMemoryEntry.to_dict()/from_dict(). MemoryAgent save_state()/load_state() v2.0 with atomic writes. Forming pool serialization. Legacy migration. Crash recovery (FORMING → SHORT_TERM on reload). |
| U8. Pipeline integration | Large | Update maxim_agent.py propose_intent() to call formation stages with run_id. Update agent_loop.py to route decision/action/outcome events to MemoryAgent. Coexistence: agent_loop continues capture_from_loop_async independently. |
| U9. MemoryItem deprecation | Small | Deprecated alias, migration notes, remove after downstream cleanup. |

**U0** is the prerequisite — thread safety, keyword and context polymorphism.
**U1-U2** are the core type swap. **U3** is staged formation with the
pattern completion hook (graph chaining implementation lives in ATL plan).
**U4-U5** wire Hippocampus and the hook. **U6-U7** deliver polymorphic
context and resilient persistence. **U8** integrates with the actual agent
pipeline (high risk — touches critical runtime code). **U9** cleans up.

Total estimated: ~600 lines changed in production code, ~400 lines new
(WorkingMemoryEntry, staged formation, persistence, pipeline hooks),
~500 lines of tests.

---

## Testing

- **Unit: `test_memory_record_abc.py`**
  - EpisodicMemory.keywords() extracts objects, people, goal, tool, input
  - MathMemory.keywords() returns name and domain
  - SemanticMemory.keywords() returns name and category
  - CompressedMemory.keywords() returns goal and tool_name
  - EpisodicMemory.to_context_dict() includes type, objects, goal, action, success
  - MathMemory.to_context_dict() includes name, domain, verbal, confidence
  - SemanticMemory.to_context_dict() includes name, category
  - CompressedMemory.to_context_dict() includes goal, action, success
  - MemoryRecord.touch() is thread-safe (concurrent touch test)

- **Unit: `test_working_memory_entry.py`**
  - Wraps EpisodicMemory, MathMemory, SemanticMemory correctly
  - id, timestamp delegate to underlying record
  - tier tracks lifecycle phase (FORMING → WORKING → SHORT_TERM → ...)
  - is_protected returns True for FORMING and WORKING
  - touch() delegates to underlying record (thread-safe)
  - keywords delegates to record.keywords() (lazy, cached)
  - invalidate_keywords() clears cache on record mutation
  - current_salience() doesn't decay for FORMING/WORKING entries
  - should_promote() only promotes SHORT_TERM entries
  - should_evict() never evicts FORMING/WORKING entries
  - to_dict() / from_dict() round-trip correctly
  - from_dict() with unknown record type raises ValueError

- **Unit: `test_memory_tier.py`**
  - All 4 tier values exist and serialize correctly
  - Tier transitions follow expected order
  - Eviction rules respect tier protection
  - _flush_working_to_short_term() transitions WORKING entries out of pool

- **Unit: `test_staged_formation.py`**
  - _begin_memory_formation creates FORMING entry with Perception+Context
  - Decision, Action, Outcome fields start empty (defaults)
  - _update_forming_decision fills in Decision by run_id, invalidates keywords
  - _update_forming_action fills in Action by run_id, invalidates keywords
  - _complete_forming_memory fills in Outcome, transitions to WORKING
  - FORMING entry survives eviction sweep
  - Concurrent pipeline cycles: multiple entries in forming pool
  - _flush_working_to_short_term clears WORKING entries on new cycle

- **Unit: `test_pattern_completion_hook.py`**
  - _pattern_completion_fn is None by default (no hook)
  - set_pattern_completion_fn() wires callable
  - Hook called during _begin_memory_formation if wired
  - predicted_outcomes populated from hook return value
  - _compute_prediction_confidence factors in action consistency
  - _compute_prediction_confidence applies sample_factor dampening (n/5)
  - High consistency + high success + sufficient samples = high confidence
  - N=1 confidence capped at 0.2 by sample factor
  - Graph chaining tests live in ATL concept plan (test_graph_chaining.py)

- **Unit: `test_memory_agent_unified.py`**
  - _process_percept creates FORMING entry (not complete MemoryItem)
  - _short_term and _long_term hold WorkingMemoryEntry objects
  - _get_relevant_memories returns WorkingMemoryEntry list
  - Keyword search works across record types (via record.keywords())
  - Promotion from SHORT_TERM to LONG_TERM preserves record
  - Eviction skips FORMING and WORKING entries

- **Unit: `test_persistence.py`**
  - save_state() writes atomic temp file + rename
  - load_state() restores short_term, long_term, forming_pool
  - FORMING entries on reload transition to SHORT_TERM
  - Legacy v1.0 state files are skipped gracefully
  - Corrupt state file doesn't crash (graceful fallback)
  - Round-trip: save → load → entries match original

- **Integration: `test_memory_unification.py`**
  - Full pipeline: percept → FORMING → decision → action → outcome → WORKING
  - Coexistence: agent_loop captures to Hippocampus independently
  - MemoryAgent and Hippocampus maintain separate EpisodicMemory instances
  - build_context() uses to_context_dict() — no isinstance chains
  - build_context() includes pattern predictions for FORMING entries
  - Pattern completion hook wired and producing predictions
  - MathMemory enters working set via WorkingMemoryEntry
  - Concept enters working set via WorkingMemoryEntry
  - Crash recovery: interrupted session restores correctly
  - Concurrent pipelines: multiple FORMING entries coexist in pool

---

## Architecture Notes

### Why WorkingMemoryEntry wraps (not extends) MemoryRecord

A wrapper preserves the canonical type. When you access `entry.record`,
you get the actual `EpisodicMemory` that Hippocampus stores — not a
subclass with extra fields that would need separate serialization.
The wrapper adds agent-level metadata (salience ranking, decay, tier)
without polluting the storage type.

If `WorkingMemoryEntry` extended `MemoryRecord`, every memory type would
need to know about agent-level concerns (decay_rate, tier). That breaks
separation — Hippocampus shouldn't know about working memory tiers.

### Why not just use MemoryRecord directly in MemoryAgent

MemoryRecord is the storage type. Working memory management needs
agent-level metadata: how salient is this memory right now? How fast
should it decay? Is it still forming? Should it be promoted? These
are runtime concerns, not storage concerns. The wrapper keeps them
separate.

### Why coexistence works

MemoryAgent and Hippocampus maintain separate EpisodicMemory instances.
No shared mutable state, no concurrent mutation concerns. Each system
manages its own copy independently:

1. MemoryAgent's copy is mutable during FORMING, immutable after WORKING
2. Hippocampus's copy is created by `capture_from_loop_async()` with all
   capture hooks (associations, consolidation candidates)
3. `touch()` on MemoryAgent's copy is thread-safe (`threading.Lock`)
4. The ~2KB memory cost per duplicate is negligible vs. the complexity
   cost of shared references across two subsystems

### Why staged formation instead of all-at-once

The current system constructs EpisodicMemory after the full pipeline
completes. This means the memory doesn't exist in working memory while
it's being formed. Staged formation provides:

1. **Eviction protection:** The current episode can't be bumped from
   working memory by buffer pressure during processing.
2. **Pattern completion:** Similar past episodes inform the current
   decision before it happens — predictive context.
3. **Biological fidelity:** Working memory holds the current experience
   while it's happening, exactly like biological episodic encoding.
4. **Active persistence:** The memory stays in working memory during
   action execution and outcome processing (the user's key requirement).

### Relationship to ATL concept plan

This unification is a **prerequisite for efficient concept grounding**
(ATL plan Phase A3). After unification:
- ConceptGrounder's `ground_concept()` accepts optional pre-loaded episodes
- `get_concept_context()` filters MemoryAgent's working set by concept refs
- Only supplements from Hippocampus for established concepts with large ref sets
- The real win is type unification (single code path), not I/O savings

### Relationship to future memory types

Every new memory type follows the same pattern:
1. Define record class extending `MemoryRecord` with `keywords()` and
   `to_context_dict()` methods
2. Create memory layer class implementing `MemoryLayer` protocol
3. Records enter MemoryAgent via `WorkingMemoryEntry[NewRecordType]`
4. `build_context()` uses `record.to_context_dict()` — no isinstance chains
5. ATL concepts link to records via `memory_refs[layer_name]`
6. Graph chaining (ATL plan Phase A5) picks up new layers automatically
   via registry-based iteration over `concept.memory_refs`

The unification establishes this pattern. Adding a new memory system
requires: one `MemoryRecord` subclass, one `MemoryLayer`, and
registering the layer name in the ATL graph chaining registry.
Everything else (working memory, context building, pattern completion,
persistence) works automatically.
