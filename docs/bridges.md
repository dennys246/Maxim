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
| `PlanHistoryBridge` | Hippocampus ↔ NAc ↔ EC | Plan template retrieval |
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
    ec=ec,
    attention=attention,  # optional
)

# On session start, restore priors from memory
restored = bridge.on_session_start()

# Boost attention for current goal
boosts = bridge.boost_attention_for_goal("find mug")
for position, boost in boosts:
    attention.apply_boost(position, boost)

# Get most likely positions for an object class
positions = bridge.get_likely_positions("mug", top_k=5)
# Returns list of (grid_u, grid_v, probability)

# Record successful find
bridge.record_success(
    object_class="mug",
    position=(320, 240),
    goal="find mug",
)

# Record failed search
bridge.record_failure(object_class="mug", position=(100, 100))

# Add cold-start prior
bridge.add_location_prior("mug", "center", probability=0.5)
```

### Features

- Location priors: Learn where object classes typically appear
- Attention boosting: Direct attention to historically successful positions
- Memory enrichment: Record spatial context with episodic memories
- Associative recall: Expands spatial coverage via Hippocampus associative graph
- Health tracking: Auto-disables after repeated errors

---

## SalienceMemoryBridge

Connects Hippocampus to the Entorhinal Cortex (EC) similarity system and SalienceNetwork. With Phase 4 semantic embeddings enabled, provides deep semantic similarity for queries like "find mug" matching memories about "cup".

```python
from maxim.bridges import SalienceMemoryBridge

bridge = SalienceMemoryBridge(
    hippocampus=hippo,
    ec=ec,
    salience_network=salience_network,
)

# On session start, restore interaction history
bridge.on_session_start()

# Enrich salience scores with interaction history
enriched = bridge.enrich_salience(
    detections=[{"label": "mug", "salience": 0.5}],
    goal="find the mug",
)
# enriched[0]["salience"] might be 0.75 if mug has positive history

# Get interaction history for an object class
record = bridge.get_interaction_history("mug")

# Get success rate for an object class
rate = bridge.get_success_rate("mug")  # Returns 0-1 (0.5 if no history)

# Record an interaction outcome
bridge.record_interaction(
    object_class="mug",
    success=True,
    goal="find the mug",
)
```

### Features

- Interaction history: Track success/failure with object classes
- Salience boosting: Increase salience for objects with positive history
- Goal-aware salience: Boost objects relevant to current goal
- Associative recall: Enriches history from Hippocampus associative graph
- Health tracking: Auto-disables after repeated errors

---

## PlanHistoryBridge

Retrieves plan templates from memory for similar goals.

```python
from maxim.bridges import PlanHistoryBridge, PlanTemplate

bridge = PlanHistoryBridge(
    hippocampus=hippo,
    nac=nac,
    ec=ec,
)

# Find templates for current goal
templates = bridge.get_plan_templates("find the mug")

for template in templates:
    print(f"Goal: {template.goal}")
    print(f"Tools: {template.tool_sequence}")
    rate = template.success_count / (template.success_count + template.failure_count)
    print(f"Success rate: {rate:.1%}")

# Predict success for a potential plan
prediction = bridge.get_predicted_success(
    goal="find mug",
    tool_sequence=["look_around", "grasp"],
)

# Get historical success rate for a tool
rate, sample_count = bridge.get_tool_success_rate("look_around")

# Record outcome after execution
bridge.record_plan_outcome(
    goal="find mug",
    tool_sequence=["look_around", "grasp"],
    success=True,
    execution_time_ms=1500,
)
```

### PlanTemplate

```python
@dataclass
class PlanTemplate:
    memory_id: str
    goal: str
    tool_sequence: list[str]
    success_count: int
    failure_count: int
    avg_execution_time_ms: float
    last_used: float
    similarity: float  # Similarity to current goal
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
)
# persist_path defaults to "data/util/escalation_learning.json"
# auto_save_interval defaults to 60.0 seconds

# Query learned threshold for current context
threshold = bridge.get_threshold(
    goal="find mug",
    novelty=0.6,
    salience=0.7,
)

# Or use the convenience method
should, reason = bridge.should_escalate(
    goal="find mug",
    novelty=0.6,
    salience=0.7,
)

# Record escalation outcome
bridge.record_outcome(
    goal="find mug",
    escalated=True,
    success=True,  # Was escalation useful?
    novelty=0.6,
    salience=0.7,
)
```

### Features

- Per-goal thresholds: Different thresholds for different goal types (search, navigation, manipulation, etc.)
- Temporal adjustment: SCN-aware thresholds that vary by time of day
- Outcome learning: Lower thresholds after escalation helped, raise after unnecessary
- Associative enrichment: Queries Hippocampus associative graph via `seed_memory_ids`
- Persists learned thresholds with auto-save

### Persistence

```json
{
  "version": 1,
  "saved_at": 1707220800.0,
  "thresholds": {
    "search:-1": {
      "goal_type": "search",
      "hour_bin": -1,
      "base_threshold": 0.65,
      "adjustment": -0.1,
      "samples": 12,
      "successes": 8,
      "last_updated": 1707220800.0
    }
  },
  "records": [],
  "config": {
    "default_novelty_threshold": 0.7,
    "default_salience_threshold": 0.6,
    "learning_rate": 0.1
  }
}
```

---

## FearCircuitBridge

Memory-informed safety assessment. Learns from historical outcomes to improve risk assessment accuracy.

```python
from maxim.bridges import FearCircuitBridge

bridge = FearCircuitBridge(
    hippocampus=hippo,
    nac=nac,
    ec=ec,
)
# persist_path defaults to "data/util/fear_learning.json"
# auto_save_interval defaults to 60.0 seconds

# Get learned risk adjustment (-0.3 to +0.3)
adjustment = bridge.get_risk_adjustment(
    category="code_execution",
    pattern="subprocess",
    context="trusted_source",
)

# Determine if action should be blocked (memory-informed)
should_block, reason = bridge.should_block(
    category="code_execution",
    severity="medium",
    pattern="subprocess",
)

# Record outcome after action
bridge.record_outcome(
    category="code_execution",
    pattern="subprocess",
    was_blocked=True,
    actual_harm=False,  # Was a false positive
    severity="medium",
    action_type="code_review",
)

# Analysis
fp_rate = bridge.get_false_positive_rate("code_execution")
patterns = bridge.get_patterns_to_review(fp_threshold=0.5)
```

### Features

- Learns risk adjustments from false positive / true positive outcomes
- Integrates with NAc causal inference for harm probability
- Associative graph enrichment via `seed_memory_ids` in `get_risk_adjustment()`
- Persists learned risk patterns with auto-save
- Category-level and pattern-level statistics

### Risk Assessment Flow

```
Action Request
      ↓
FearCircuitBridge.should_block()
      ├── get_risk_adjustment() (learned patterns + associative graph)
      ├── NAc.predict_outcome() (causal inference)
      └── Severity scoring with learned adjustment
      ↓
Combined Risk Score (base + adjustment * nac_factor)
      ↓
Block if score >= 0.65
```

---

## PainCircuitBridge

Connects pain detection to NAc for movement learning.

```python
from maxim.bridges.pain_bridge import PainCircuitBridge, PainBridgeConfig

config = PainBridgeConfig(
    enable_learning=True,
    enable_predictive_harm=True,       # Physics-based prediction
    enable_joint_limit_prediction=True, # Joint limit prediction
    angular_velocity_threshold=100.0,   # deg/sec
    yaw_limit=45.0,
    pitch_limit=30.0,
)

bridge = PainCircuitBridge(
    nac=nac,
    pain_detector=pain_detector,
    config=config,  # optional, defaults are sensible
)

# Record action start (returns event_id)
event_id = bridge.record_action_start(
    action_signature="look_at:dy=90:dp=30",
    context={"position": (0, 0)},
    target_yaw=90.0,
    target_pitch=30.0,
)

# Pain detector fires automatically during movement
# Bridge handles the learning via _on_pain() callback

# Record successful completion (positive feedback to NAc)
bridge.record_action_complete(success=True)

# Two-tier prediction before next similar movement
should_gate, reason = bridge.should_gate_action(
    action_signature="look_at:dy=85:dp=5",
    duration=0.3,
)
if should_gate:
    print(f"Movement gated: {reason}")

# Get combined pain risk score (0-1)
risk = bridge.get_pain_risk("look_at:dy=85:dp=5")
```

### Two-Tier Prediction

```
1. Predictive harm (Tier 1): Physics-based, zero latency
   → MovementHarmPredictor (velocity analysis)
   → JointLimitHarmPredictor (workspace bounds)

2. Learned prediction (Tier 2): NAc-based
   → Queries causal links from past pain events

should_gate_action() checks both tiers.
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
from maxim.bridges.energy_bridge import EnergyCircuitBridge, EnergyBridgeConfig

config = EnergyBridgeConfig(
    enable_learning=True,
    high_energy_valence_threshold=3.0,  # > 3 = NEGATIVE
    low_energy_valence_threshold=0.5,   # < 0.5 = POSITIVE
)

bridge = EnergyCircuitBridge(
    nac=nac,
    registry=energy_registry,
    config=config,  # optional
)

# Track actual energy for an action
event_id = bridge.record_action_start(
    action_signature="large_generation",
    action_type="llm",
)
# ... action executes, energy signals accumulate ...
total_energy = bridge.record_action_end(event_id)

# Predict energy cost before execution
predicted = bridge.predict_energy(
    action_signature="large_generation",
    action_type="llm",
)

# Check if action should be gated due to high energy
should_gate, reason = bridge.should_gate_action(
    action_signature="large_generation",
    action_type="llm",
)

# Get energy context string for LLM prompts
context = bridge.get_energy_context_for_llm()
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

### Consolidation Wiring

MemoryHub also wires the consolidation pipeline during `wire_memory_hub()`:

- Creates `SimilarityIndex` instances (context_index + percept_index)
- Wires LSH into `AssociationIndex` for O(1) similarity lookups
- Wires `ExecAgent.wire_staging()` for acute staging after goal completion
- Wires LSH indices into Hippocampus for `recall_deep` seeding

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
