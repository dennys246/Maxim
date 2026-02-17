# Bridges System

The bridges module provides cross-system integration, enabling bidirectional learning between the Hippocampus memory system and external perception/decision/action systems.

## Overview

Bridges serve as integration points that:

1. **Enrich** external systems with historical context from memory
2. **Update** memory based on new observations and outcomes
3. **Enable** learning across system boundaries
4. **Maintain** consistency between distributed state

## Available Bridges

| Bridge | Systems Connected | Purpose |
|--------|-------------------|---------|
| `SpatialMemoryBridge` | Hippocampus ↔ SpatialMap ↔ AttentionNetwork | Spatial memory integration |
| `SalienceMemoryBridge` | Hippocampus ↔ EC ↔ SalienceNetwork | Salience-based memory |
| `PlanHistoryBridge` | Hippocampus ↔ NAc | Plan template retrieval |
| `EscalationLearningBridge` | Hippocampus ↔ SCN/NAc | Learned escalation thresholds |
| `FearCircuitBridge` | Hippocampus ↔ NAc ↔ EC | Learned risk patterns |
| `PainCircuitBridge` | PainDetector ↔ NAc | Movement pain learning |
| `EnergyCircuitBridge` | EnergyRegistry ↔ NAc | Energy cost learning |
| `CommunicationBridge` | Comms ↔ Hippocampus | Communication-aware memory |
| `MathBridge` | AngularGyrus ↔ Hippocampus | Math pattern learning |

---

## SpatialMemoryBridge

Connects the Hippocampus to the SpatialMap and AttentionNetwork for location-based memory.

```python
from maxim.bridges import SpatialMemoryBridge

bridge = SpatialMemoryBridge(
    hippocampus=hippo,
    spatial_map=spatial_map,
    attention_network=attention,
)

# When gazing at a location
bridge.record_gaze(yaw=45.0, pitch=10.0, objects=["book", "cup"])

# Query memories at location
memories = bridge.query_location(yaw=45.0, pitch=10.0, radius=10.0)

# Get spatially-relevant memories for current view
relevant = bridge.get_relevant_for_view(current_fov=(30, 20, 60, 40))
```

### Features

- Records object locations in memory
- Retrieves memories by spatial proximity
- Updates attention based on memory-rich locations
- Persists spatial indices

---

## SalienceMemoryBridge

Connects Hippocampus to the Entorhinal Cortex (EC) similarity system and SalienceNetwork. With Phase 4 semantic embeddings enabled, provides deep semantic similarity for queries like "find mug" matching memories about "cup".

```python
from maxim.bridges import SalienceMemoryBridge

bridge = SalienceMemoryBridge(
    hippocampus=hippo,
    salience_network=salience,
    similarity_threshold=0.7,
)

# Enrich salience with memory context
bridge.enrich_salience(detections)

# Store salient observations
bridge.record_salient_observation(
    object_class="person",
    salience=0.9,
    context={"location": (45, 10)},
)

# Query similar past observations
similar = bridge.query_similar_observations(
    object_class="person",
    context=current_context,
)
```

### Features

- Boosts salience for memory-recognized objects
- Stores high-salience observations
- Semantic similarity queries (Phase 4: neural embeddings for "cup" ≈ "mug")
- Novelty adjustment based on memory
- Async embedding for non-blocking memory capture

---

## PlanHistoryBridge

Retrieves plan templates from memory for similar goals.

```python
from maxim.bridges import PlanHistoryBridge, PlanTemplate

bridge = PlanHistoryBridge(
    hippocampus=hippo,
    nac=nac,
)

# Query for similar plans
templates = bridge.get_templates_for_goal(
    goal="find_object",
    context={"target": "book"},
)

for template in templates:
    print(f"Plan: {template.steps}")
    print(f"Success rate: {template.success_rate:.2f}")

# Record completed plan
bridge.record_plan_outcome(
    goal="find_object",
    steps=["scan_room", "approach", "pick_up"],
    outcome="success",
    duration=45.0,
)
```

### PlanTemplate

```python
@dataclass
class PlanTemplate:
    goal: str
    steps: list[str]
    success_rate: float
    avg_duration: float
    context_match: float
    memory_ids: list[str]
```

---

## EscalationLearningBridge

Learns when to escalate from reactive (DN) to deliberative (LLM) processing.

```python
from maxim.bridges import EscalationLearningBridge

bridge = EscalationLearningBridge(
    hippocampus=hippo,
    scn=scn,
    nac=nac,
    persist_path="data/util/escalation_learning.json",
)

# Query learned threshold for current context
threshold = bridge.get_threshold(
    goal_type="exploration",
    temporal_context=current_temporal_sig,
)

# Record escalation outcome
bridge.record_escalation(
    goal_type="exploration",
    escalated=True,
    outcome="helpful",  # Was escalation useful?
    context={"novelty": 0.8},
)
```

### Features

- Learns goal-specific escalation thresholds
- Temporal context awareness (time of day, etc.)
- Persists learned thresholds
- Adapts based on escalation outcomes

### Persistence

```json
{
  "version": 1,
  "saved_at": "2024-02-06T12:00:00Z",
  "thresholds": {
    "exploration": {
      "base": 0.6,
      "temporal_adjustments": {
        "morning": -0.1,
        "evening": 0.05
      }
    }
  }
}
```

---

## FearCircuitBridge

Connects the FearAgent to learned risk patterns in NAc.

```python
from maxim.bridges import FearCircuitBridge

bridge = FearCircuitBridge(
    fear_agent=fear_agent,
    nac=nac,
    hippocampus=hippo,
    persist_path="data/util/fear_learning.json",
)

# Query risk for action
risk = bridge.predict_risk(
    action_type="movement",
    action_signature="look_at:dy=90",
    context={"current_position": (0, 0)},
)

if risk.level > 0.5:
    print(f"High risk: {risk.reason}")

# Record outcome after action
bridge.record_outcome(
    action_type="movement",
    action_signature="look_at:dy=90",
    outcome="safe",  # or "harmful"
)
```

### Features

- Learns action-risk associations
- Integrates with FearAgent review
- Queries similar past experiences
- Persists learned risk patterns

### Risk Prediction Flow

```
Action Request
      ↓
FearCircuitBridge.predict_risk()
      ├── NAc.predict() (learned patterns)
      ├── Hippocampus.query() (similar episodes)
      └── HarmRegistry.predict() (predictive harm)
      ↓
Combined Risk Assessment
      ↓
FearAgent.review_action()
```

---

## PainCircuitBridge

Connects pain detection to NAc for movement learning.

```python
from maxim.bridges.pain_bridge import PainCircuitBridge

bridge = PainCircuitBridge(
    nac=nac,
    pain_detector=pain_detector,
)

# Record action start
bridge.record_action_start(
    action_signature="look_at:dy=90:dp=30",
    context={"position": (0, 0)},
)

# Pain detector fires automatically during movement
# Bridge handles the learning

# Predict if action will cause pain
should_gate, reason = bridge.should_gate_action(
    action_signature="look_at:dy=85",
)

if should_gate:
    print(f"Predicted pain: {reason}")
```

### Learning Flow

```
1. record_action_start("look_at:dy=90")
         ↓
2. Movement executes, PainDetector fires
         ↓
3. _on_pain() callback:
   → NAc.record_outcome(NEGATIVE)
         ↓
4. Future similar action:
   → should_gate_action() returns True
```

---

## EnergyCircuitBridge

Connects energy tracking to NAc for cost-aware decisions.

```python
from maxim.bridges.energy_bridge import EnergyCircuitBridge

bridge = EnergyCircuitBridge(
    energy_registry=get_global_registry(),
    nac=nac,
)

# Record action energy cost
bridge.record_action_energy(
    action_type="llm",
    action_signature="large_generation",
    energy_cost=1500.0,
)

# Predict energy cost for action
predicted_cost = bridge.predict_energy_cost(
    action_type="llm",
    action_signature="large_generation",
)
```

---

## Bridge Pattern

All bridges follow a consistent pattern:

```python
class ExampleBridge:
    def __init__(
        self,
        hippocampus: Hippocampus,
        external_system: ExternalSystem,
        persist_path: str | None = None,
        auto_save_interval: float = 60.0,
    ):
        self._hippo = hippocampus
        self._external = external_system
        self.persist_path = persist_path
        self._last_save_time = time.time()
        self.auto_save_interval = auto_save_interval

        # Auto-load on init
        if persist_path and os.path.exists(persist_path):
            self.load(persist_path)

    def save(self, path: str | None = None) -> None:
        """Persist bridge state."""
        ...

    def load(self, path: str) -> None:
        """Load bridge state."""
        ...

    def _maybe_auto_save(self) -> None:
        """Auto-save if interval elapsed."""
        if self.persist_path and time.time() - self._last_save_time > self.auto_save_interval:
            self.save(self.persist_path)
            self._last_save_time = time.time()
```

---

## Persistence

All bridges with learning state persist to JSON:

| Bridge | Persist Path | Contents |
|--------|--------------|----------|
| EscalationLearningBridge | `data/util/escalation_learning.json` | Learned thresholds |
| FearCircuitBridge | `data/util/fear_learning.json` | Risk patterns |
| PainCircuitBridge | *(no own persistence — pain learning persisted via NAc)* | Pain associations |
| MemoryHub (semantic) | `data/util/semantic_embeddings.npz` | Phase 4 neural embeddings |

Clear with: `maxim --clear-memory escalation,fear,semantic`

---

## Design Philosophy

Bridges embody several key principles:

1. **Separation of Concerns**: Memory, decisions, and actions remain modular
2. **Bidirectional Flow**: Information flows both ways
3. **Learning at Boundaries**: Learning happens at integration points
4. **Graceful Degradation**: Systems work without bridges, just less intelligently
5. **Persistence**: Learned associations survive restarts

---

## MemoryHub (Coordinator)

The MemoryHub coordinates all bridges and manages semantic embedding lifecycle:

```python
from maxim.integration import MemoryHub

hub = MemoryHub(
    hippocampus=hippo,
    scn=scn,
    nac=nac,
    ec=EntorhinalCortex(ECConfig(enable_semantic=True)),
)

# Session lifecycle manages embedding persistence
hub.on_session_start()  # Loads embeddings from disk

# Semantic queries delegate to EC
if hub.semantic_enabled:
    results = hub.find_semantic("find coffee mug", k=10, threshold=0.5)
    # Returns memories about "cup", "mug", "coffee container", etc.

hub.on_session_end()  # Saves embeddings to disk
```

### Semantic Capture Flow

```
Memory Capture
      ↓
Hippocampus.capture()
      ↓
_on_memory_captured callback (registered by MemoryHub)
      ↓
NeuralSemanticLSH.schedule_embedding() (async)
      ↓
[Background Thread]
      ↓
EmbeddingStore.set(memory_id, embedding, hash)
      ↓
Semantic queries work
```

---

## Adding New Bridges

To add a new bridge:

1. Create `src/maxim/bridges/new_bridge.py`
2. Follow the bridge pattern (init, save, load, auto-save)
3. Export from `src/maxim/bridges/__init__.py`
4. Add persistence path to `MEMORY_PATHS` in `cli.py`
5. Document in this file

```python
class NewBridge:
    def __init__(
        self,
        hippocampus: Hippocampus,
        other_system: OtherSystem,
        persist_path: str | None = "data/util/new_bridge.json",
    ):
        ...
```
