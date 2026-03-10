# Memory System

The memory module provides episodic memory storage and retrieval through a biologically-inspired Hippocampus architecture.

## Overview

Maxim's memory system is modeled after the mammalian hippocampus and entorhinal cortex (EC):

- **Hippocampus**: Episodic memory storage with associative indexing
- **SCN Integration**: Temporal rhythm indexing for time-based retrieval
- **EC Similarity**: Multi-modal similarity with Phase 4 neural semantic embeddings

## Components

| Component | File | Purpose |
|-----------|------|---------|
| `Hippocampus` | `hippocampus.py` | Core episodic memory graph |
| `StateStore` | `state_store.py` | State snapshot caching |
| `RWLock` | `rwlock.py` | Reader-writer lock for concurrent access |
| `MemoryStrategy` | `strategies.py` | Retention scoring algorithms |

---

## Hippocampus

The Hippocampus is the central memory store for episodic memories.

### Memory Types

| Type | Class | Contents |
|------|-------|----------|
| Perception | `Perception` | Sensory input (vision, audio) |
| Action | `Action` | Executed actions and parameters |
| Decision | `Decision` | Planning decisions and rationale |
| Outcome | `Outcome` | Action results and feedback |
| Context | `Context` | Environmental state |

### Capturing Memories

```python
from maxim.memory import Hippocampus, HippocampusConfig, Perception

config = HippocampusConfig(
    max_nodes=10_000,
    persistence_path="data/util/hippocampus.json",
    indexed_keys=frozenset({"goal", "tool", "object", "person"}),
)

hippo = Hippocampus(config)

# Capture a perception
perception = Perception(
    timestamp=time.time(),
    modality="vision",
    content={"objects": ["person", "book"], "confidence": 0.95},
    salience=0.8,
    novelty=0.6,
)
memory_id = hippo.capture(perception)
```

### Querying Memories

```python
# Recall by filters (goal, tool, success, mode, time range)
memories = hippo.recall(goal="find_book")

# Recall by time range (parameters on recall())
recent = hippo.recall(
    time_after=time.time() - 3600,  # Last hour
    time_before=time.time(),
)

# Get by ID (O(1))
memory = hippo.get(memory_id)

# Similarity-based recall (perception overlap)
similar = hippo.recall_similar(
    perception=current_perception,
    limit=10,
    threshold=0.5,
)

# Associative recall (spreading activation through graph)
associated = hippo.recall_associated(
    memory_id=memory_id,
    max_depth=3,
    decay=0.5,
)
```

### Configuration

```python
@dataclass
class HippocampusConfig:
    max_nodes: int = 10_000
    persistence_path: str | None = None
    indexed_keys: frozenset[str] = frozenset({"goal", "tool", "object", "person", "success", "mode"})

    # Sleep consolidation
    enable_sleep_consolidation: bool = True
    max_age_without_access: float = 7 * 24 * 3600  # 1 week
    compression_age: float = 24 * 3600             # 1 day
    retention_threshold: float = 0.3
    compression_threshold: float = 0.6

    # Long-term memory
    immediate_promotion_salience: float = 0.95
    immediate_promotion_novelty: float = 0.95
```

---

## Memory Strategies

Retention strategies determine which memories to keep, compress, or remove.

### Available Strategies

| Strategy | Key Factor | Use Case |
|----------|------------|----------|
| `AccessBasedStrategy` | Recency + frequency of access | General use |
| `ImportanceBasedStrategy` | Salience + novelty scores | High-value retention |
| `TemporalAwareStrategy` | Age-weighted scoring | Decay over time |
| `CompositeStrategy` | Weighted combination | Custom policies |

### Usage

```python
from maxim.memory import (
    AccessBasedStrategy,
    ImportanceBasedStrategy,
    CompositeStrategy,
)

# Single strategy
strategy = AccessBasedStrategy()

# Composite with weights
strategy = CompositeStrategy([
    (AccessBasedStrategy(), 0.4),
    (ImportanceBasedStrategy(), 0.6),
])

# Apply to hippocampus
hippo = Hippocampus(config, strategy=strategy)
```

---

## Sleep Consolidation

During sleep mode, the Hippocampus performs memory consolidation:

1. **Prune**: Remove low-retention memories below threshold
2. **Compress**: Convert detailed memories to `CompressedMemory`
3. **Promote**: Move high-value memories to long-term storage

```python
# Trigger sleep consolidation (compress, remove, promote)
hippo.sleep()

# Or trigger promotion-only consolidation
hippo.consolidate()

# Consolidation runs automatically during sleep mode
# See modes/definitions.py for sleep mode behavior
```

### Compressed Memories

Old memories are compressed to save space:

```python
@dataclass
class CompressedMemory:
    """Compact representation of old memories."""
    memory_id: str
    original_type: str           # "perception", "action", etc.
    timestamp: float
    summary_embedding: list[float]  # Semantic embedding
    key_facts: dict[str, Any]    # Preserved important fields
    access_count: int
    last_accessed: float
```

---

## StateStore

Caches state snapshots for fast retrieval during planning.

```python
from maxim.memory import StateStore

store = StateStore(max_entries=1000)

# Cache state
state = {"position": (45, 10), "objects": ["book", "pen"]}
store.set("current_view", state)

# Retrieve
cached = store.get("current_view")
```

---

## Concurrency

The Hippocampus uses RWLock for thread-safe access:

```python
from maxim.memory import RWLock

lock = RWLock()

# Multiple readers
with lock.read_lock():
    memories = hippo.query(goal="example")

# Exclusive writer
with lock.write_lock():
    hippo.capture(new_memory)
```

---

## Persistence

Hippocampus persists to JSON:

```python
# Save manually
hippo.save("data/util/hippocampus.json")

# Load on init if path exists
hippo = Hippocampus(HippocampusConfig(
    persistence_path="data/util/hippocampus.json"
))

# Auto-save on shutdown
# Handled by runtime shutdown hooks
```

### File Format

```json
{
  "version": "3.0",
  "saved_at": 1707235200.0,
  "memories": [...],
  "indices": {...},
  "metadata": {
    "total_captures": 1234,
    "compressions": 56
  }
}
```

Clear with: `maxim --clear-memory hippo`

---

## Integration Points

| System | Integration |
|--------|-------------|
| **SCN** | Temporal indexing for time-based queries |
| **NAc** | Causal learning from episodic sequences |
| **SalienceMemoryBridge** | Updates salience from memory patterns |
| **SpatialMemoryBridge** | Enriches spatial map with memory |
| **EC Similarity** | Phase 4 neural semantic embeddings for deep similarity queries |

### Semantic Embedding (Phase 4)

When semantic similarity is enabled, the EC uses neural embeddings for deep semantic understanding:

```python
from maxim.similarity import ECConfig, EntorhinalCortex

# Enable Phase 4 semantic embeddings
ec = EntorhinalCortex(ECConfig(
    enable_semantic=True,
    semantic_model="all-MiniLM-L6-v2",  # 80MB model
    async_embedding=True,  # Non-blocking capture
))

# Semantic queries work across synonyms
results = ec.find_semantic("find the coffee mug", k=10)
# "cup" memories will match with high similarity
```

Semantic embeddings are automatically generated when memories are captured (via capture callback) and persist across sessions in `data/util/semantic_embeddings.npz`.

See [bridges.md](bridges.md) for bridge documentation.

---

## Memory Hierarchy

Maxim maintains multiple memory tiers with different lifespans and retrieval modes:

```
Percept Buffer → Working Notes → Short-Term Memory → Long-Term Episodic → Semantic Concepts
  (prompt)        (prompt)        (MemoryAgent)       (Hippocampus)       (ATL: objects,
                                                                           people, goals,
                                                                           relationships)
```

| Tier | Retrieval | Contents | Lifespan |
|------|-----------|----------|----------|
| Percept buffer | Always present (last N) | Recent percepts, outcomes | Minutes |
| Working notes | Always present (file read) | Deliberately pinned context | Until LLM removes it |
| MemoryAgent working memory | Association + activation | WorkingMemoryEntry wrappers | Session (decays/promotes) |
| Hippocampus (long-term) | Similarity recall | Consolidated episodic memories | Permanent (retention-scored) |
| ATL (semantic) | Concept lookup + activation | Concepts, relationships, causal patterns | Permanent |
| Plan system | Always present when active | Structured goals/phases | Until plan completes |

**Working notes** (`notes/context.md`) give the LLM a persistent scratchpad — always in the prompt, edited via `write_file`. Unlike similarity-based Hippocampus recall, working notes survive regardless of what the LLM is currently perceiving.

**StructuredContext** (built by MemoryAgent each cycle) assembles these tiers into a single object consumed by ExecAgent for goal proposal. Fields include `relevant_memories`, `working_notes`, `workspace_files`, `knowledge_context`, and `plan_progress`.

## WorkingMemoryEntry and Staged Formation

MemoryAgent wraps all memory records in `WorkingMemoryEntry[T]` — a generic
wrapper that holds any `MemoryRecord` subclass plus agent-level metadata
(salience, decay_rate, tier, predicted outcomes).

### Memory Tier Lifecycle

```
FORMING → WORKING → SHORT_TERM → LONG_TERM → consolidated out
```

- **FORMING**: Created at percept time, filled incrementally during pipeline. Eviction-protected.
- **WORKING**: Pipeline complete, awaiting next cycle sweep. Eviction-protected.
- **SHORT_TERM**: Normal decay and eviction. Promotes to LONG_TERM on high access/salience.
- **LONG_TERM**: Age-based eviction. Consolidation marks records in Hippocampus.

### Staged Formation

EpisodicMemory is constructed incrementally and held in active working memory
throughout the pipeline:

1. **Percept arrives** → `_begin_memory_formation()` creates FORMING entry with Perception+Context
2. **Decision made** → `_update_forming_decision()` fills in Decision
3. **Action executes** → `_update_forming_action()` fills in Action
4. **Outcome received** → `_complete_forming_memory()` fills Outcome, transitions to WORKING
5. **New cycle** → `_flush_working_to_short_term()` sweeps WORKING → SHORT_TERM

### Pattern Completion Hook

Optional `_pattern_completion_fn` callable (wired by ATL) provides predictive
context during FORMING. Returns `list[PredictedOutcome]` with typed fields
(`tool`, `success`, `goal`, `confidence`, `math_context`, `source_episode_id`).

### Type Contracts

- `PredictedOutcome`: Typed prediction from graph chaining (defined in `types.py`)
- `MathContextEntry`: Per-concept math enrichment data (defined in `types.py`)
- `WorkingMemoryEntry[T]`: Generic wrapper (defined in `bus.py`)

### Coexistence with Hippocampus

MemoryAgent and Hippocampus maintain separate EpisodicMemory instances.
`agent_loop.py` continues calling `capture_from_loop_async()` independently.
The two systems coexist — MemoryAgent gets structured types in working memory,
Hippocampus retains all capture hooks (associations, consolidation, promotion).

## Memory Flow

```
Perception/Action/Decision
          ↓
    Hippocampus.capture()
          ↓
    ┌─────┴─────┐
    │  Indexing │
    ├───────────┤
    │ Hash keys │ → O(1) lookup
    │ SCN bins  │ → Temporal query
    │ EC embed  │ → Semantic search
    └───────────┘
          ↓
    StateStore (cache)
          ↓
    Sleep Consolidation
    ├── Prune low-retention
    ├── Compress old memories
    └── Promote high-value
```
