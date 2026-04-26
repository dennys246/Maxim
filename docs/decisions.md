# Decisions System

The decisions module provides causal inference and outcome prediction through the Nucleus Accumbens (NAc), enabling the robot to learn action-outcome relationships and predict results before acting.

## Overview

The NAc (Nucleus Accumbens) is inspired by the brain's reward prediction system. It:

1. **Observes** action-outcome pairs during execution
2. **Learns** causal relationships via temporal difference learning
3. **Predicts** outcomes for future actions
4. **Enables** proactive decision-making based on expected results

## Components

| Component | File | Purpose |
|-----------|------|---------|
| `NAc` | `nac.py` | Causal inference engine |
| `CausalLink` | `causal_link.py` | Event-outcome relationship |
| `OutcomePrediction` | `causal_link.py` | Prediction result |
| `Valence` | `causal_link.py` | Outcome quality (positive/neutral/negative) |
| `AdaptivePlanner` | `planning/adaptive_planner.py` | Goal decomposition using NAc predictions |

## Runtime Integration

NAc learns from two sources at runtime:

1. **Tool outcomes** — every tool execution in the agent loop (`runtime/agent_loop.py`) calls `nac.observe()` via `_record_outcome()`. This is how NAc learns "tool X in context Y → success/failure."

2. **Pain events** — the PainBus publishes pain signals to a NAc subscriber (`proprioception/pain_bus.py:create_pain_nac_subscriber`). This is how NAc learns "action X → pain" for avoidance.

NAc predictions are surfaced in the LLM prompt via `StructuredContext.causal_context` (built by `MemoryAgent._build_causal_context()`). The LLM sees learned expectations like "stealth past guard → success (confidence=0.7)" before making decisions.

In the Agent Mesh, NAc links can be shared between peers via `CausalLinkProvider`/`CausalLinkReceiver` with trust-level transfer discounts.

---

## Valence

Outcomes are classified by valence:

```python
from maxim.decisions import Valence

class Valence(Enum):
    POSITIVE = "positive"   # Desirable outcome
    NEUTRAL = "neutral"     # No strong preference
    NEGATIVE = "negative"   # Undesirable outcome
```

---

## CausalLink

Represents a learned relationship between an event and its outcome:

```python
@dataclass
class CausalLink:
    event_type: str           # "tool", "movement", "speech"
    event_signature: str      # "internet_search", "look_at:dy=45"
    outcome_type: str         # "tool_result", "pain_signal"
    outcome_signature: str    # "success", "excessive_velocity"
    outcome_valence: Valence  # POSITIVE, NEUTRAL, NEGATIVE

    # Temporal relationship
    delta: TemporalDelta      # Time between event and outcome

    # Learning state
    strength: float           # 0.0-1.0, learned via Rescorla-Wagner
    confidence: float         # How reliable is this link?
    observation_count: int    # Number of observations

    # Context matching
    context: dict[str, Any]   # Conditions when this link applies
```

---

## NAc (Nucleus Accumbens)

The core causal inference engine.

### Configuration

```python
from maxim.decisions import NAc, NACConfig

config = NACConfig(
    max_links=10000,                     # Maximum causal links
    min_confidence_threshold=0.3,        # Min confidence for predictions
    decay_interval_hours=24.0,           # Decay unused links
    context_similarity_threshold=0.5,    # Min context match
    temporal_window_seconds=300.0,       # Max event-outcome delay
    base_learning_rate=0.2,              # Rescorla-Wagner α
    enable_hippocampus_queries=True,     # Query memory for similar cases
)

nac = NAc(config)
```

### Observing Outcomes

Record action-outcome pairs for learning:

```python
# Tool execution succeeded
nac.observe(
    event_type="tool",
    event_signature="internet_search",
    outcome_type="tool_result",
    outcome_signature="success_with_results",
    outcome_valence=Valence.POSITIVE,
    delta_seconds=2.3,
    context={"mode": "exploration", "query_type": "factual"},
)

# Movement caused pain
nac.observe(
    event_type="movement",
    event_signature="look_at:dy=90:dp=30",
    outcome_type="proprioception",
    outcome_signature="excessive_velocity",
    outcome_valence=Valence.NEGATIVE,
    delta_seconds=0.5,
    context={"current_yaw": 0, "target_yaw": 90},
)
```

### Predicting Outcomes

Query predictions before acting:

```python
from maxim.decisions import OutcomePrediction

prediction = nac.predict(
    event_type="movement",
    event_signature="look_at:dy=85:dp=25",
    context={"current_yaw": 5, "target_yaw": 90},
)

if prediction:
    print(f"Predicted outcome: {prediction.predicted_outcome}")
    print(f"Expected valence: {prediction.predicted_valence}")
    print(f"Confidence: {prediction.confidence:.2f}")

    if prediction.predicted_valence == Valence.NEGATIVE:
        print("Consider alternative action")
```

### OutcomePrediction

```python
@dataclass
class OutcomePrediction:
    event_signature: str          # What was queried
    predicted_outcome: str        # Expected outcome signature
    predicted_valence: Valence    # POSITIVE/NEUTRAL/NEGATIVE
    confidence: float             # 0.0-1.0
    supporting_links: int         # Number of observations
    context_match: float          # How well context matches
```

---

## Learning Algorithm

NAc uses **Rescorla-Wagner learning** for stable convergence:

```
ΔV = α(λ - V)
```

Where:
- `α` = learning rate (default 0.2)
- `λ` = actual outcome valence
- `V` = current link strength

### Decay

Unused links decay over time:

```python
# Links not observed within decay_interval_hours lose strength
# This prevents stale predictions from old experiences
```

---

## Context Matching

Predictions are context-aware:

```python
# Context fields compared:
context = {
    "mode": "exploration",     # Current operating mode
    "time_of_day": "morning",  # Temporal context
    "goal": "find_object",     # Current goal
    ...
}

# Links with similar context are weighted higher
```

---

## Integration Points

| System | Integration |
|--------|-------------|
| **Hippocampus** | Query similar episodes for inference |
| **SCN** | Temporal context for when patterns apply. Oscillator feedback (B2): event-type phase tracking → anticipatory credit pre-activates eligibility traces for predicted-imminent events |
| **FearAgent** | Gate actions with negative predictions |
| **PainCircuitBridge** | Learn from movement pain signals |
| **EscalationLearningBridge** | Learn escalation thresholds |
| **ExecAgent Contemplation** | Learn when plan critique+refine improves outcomes; auto-tune contemplation gates |
| **SignificanceWeightLearner** | RPE magnitude is the top-weighted significance heuristic (0.35) for memory staging |
| **ConsolidationOrchestrator** | NAc corroboration (has event→outcome pattern strengthened?) contributes 0.20 of wave score |

### Example: Pain Learning Flow

```
1. Movement executes: look_at(dy=90)
         ↓
2. PainDetector fires: EXCESSIVE_VELOCITY
         ↓
3. NAc.observe():
   - event: "look_at:dy=90"
   - outcome: "excessive_velocity"
   - valence: NEGATIVE
         ↓
4. CausalLink created/strengthened
         ↓
5. Next time similar movement requested:
   - NAc.predict("look_at:dy=85")
   - Returns: NEGATIVE, confidence=0.7
         ↓
6. FearAgent gates or softens action
```

### Example: Contemplation Learning Flow

```
1. ExecAgent contemplates a complex plan (critique → refine)
         ↓
2. Goal executes, GoalCompleted fires on bus
         ↓
3. ExecAgent._on_goal_completed():
   - NAc.observe():
     event: "contemplation:refined"
     outcome: goal success/failure
     valence: POSITIVE/NEGATIVE
         ↓
4. CausalLink created/strengthened
         ↓
5. _adaptive_thresholds() queries NAc:
   - High refined success rate → lower confidence_threshold (contemplate more)
   - Low refined success rate → raise confidence_threshold (contemplate less)
```

---

## Persistence

NAc state persists to JSON:

```python
# Default path
persist_path = "~/.maxim/util/nac_state.json"

# Save/load handled by runtime
nac.save(persist_path)
nac.load(persist_path)
```

Clear with: `maxim --clear-memory nac`

---

## Reward-Modulated Recognition (P2)

NAc now modulates substrate recognition via per-node reward biases. When a Reaction (positive or negative) fires, NAc credits recently-active ATL nodes and adjusts EC's pattern completion threshold for those nodes.

### Per-node reward bias

```python
# Credit a node after a positive reaction
nac.credit_node(agent_id="agent-1", node_id="atl-node-abc", reward=1.0)

# Check current bias
bias = nac.reward_bias("agent-1", "atl-node-abc")

# Get EC threshold overrides for pattern completion
overrides = nac.get_threshold_overrides("agent-1")
# → {"atl-node-abc": 0.25}  (base 0.40 - bias)
```

### Eligibility traces

When a percept completes to a node, NAc tracks that activation. When a reward arrives, all eligible nodes receive credit proportional to their activation strength.

```python
# Automatic — called by LinguisticEncoder on pattern completion
nac.update_eligibility("agent-1", "node-abc", activation=0.85)

# When reward arrives, distribute to all eligible nodes
credited = nac.distribute_reward("agent-1", reward=1.0)
# → [("node-abc", 0.6), ("node-def", 0.4)]
```

### Decay

Reward biases and eligibility traces decay over time to prevent runaway recognition widening.

```python
nac.decay_reward_biases()  # Exponential decay with tau from config
nac.decay_eligibility(factor=0.9)  # Per-tick decay
```

### Configuration

```python
config = NACConfig(
    reward_bias_alpha=0.15,      # How much each reward strengthens bias
    reward_bias_decay_tau=50.0,  # Decay timescale (ticks)
    max_reward_bias=0.20,        # Cap on threshold reduction
)
```

### Reward Distribution (SEM Learning Loop)

`NAc.distribute_reward(agent_id, reward)` distributes reward across eligible nodes via `credit_node()`. Eligibility traces are set by `update_eligibility()` when percepts complete to substrate nodes. The ReactionBus subscriber in `build_bio_stack` maps reactions to rewards:
- `Valence.NEGATIVE` -- reward = -intensity (clamps to 0 in credit_node -- bias only widens)
- `Valence.POSITIVE` -- reward = +intensity (widens EC recognition radius)

`get_threshold_overrides(agent_id)` returns the per-node bias map for EC to use during `pattern_complete`.

---

## Biological Inspiration

| Biological | Maxim Equivalent |
|------------|------------------|
| Nucleus Accumbens | NAc class |
| Dopamine signals | Valence (reward/punishment) |
| Temporal difference | Rescorla-Wagner learning |
| Reward prediction | predict() method |
| Behavioral inhibition | FearAgent gating |
| Reward-modulated plasticity | Per-node reward bias (P2) |
| Eligibility traces | Activation-weighted credit assignment (P2) |

The goal is proactive, experience-based decision making that learns from consequences rather than requiring explicit programming of every scenario.
