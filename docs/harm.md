# Harm Prediction System

The harm module provides predictive harm detection, enabling proactive risk mitigation BEFORE actions are executed.

## Overview

The harm prediction system:

1. **Analyzes** action parameters before execution
2. **Predicts** potential harmful outcomes
3. **Enables** FearAgent to gate risky actions
4. **Registers** domain-specific harm predictors

This is distinct from pain detection (which is reactive) - harm prediction is proactive.

## Components

| Component | File | Purpose |
|-----------|------|---------|
| `HarmRegistry` | `registry.py` | Central coordinator for predictors |
| `HarmPredictor` | `predictor.py` | Base predictor interface |
| `MovementHarmPredictor` | `movement.py` | Velocity/acceleration risks |
| `JointLimitHarmPredictor` | `joint_limit.py` | Position limit risks |

---

## HarmCategory

Categories of potential harm:

```python
from maxim.harm import HarmCategory

class HarmCategory(Enum):
    # Physical/Motor
    MOVEMENT_VELOCITY = "movement_velocity"      # Excessive movement speed
    MOVEMENT_ACCELERATION = "movement_acceleration"  # Sudden speed changes
    MOVEMENT_THRASHING = "movement_thrashing"    # Rapid direction reversals
    MOTOR_STALL = "motor_stall"                  # Position unreachable
    JOINT_LIMIT = "joint_limit"                  # Near workspace boundaries
    # Computational
    LLM_TIMEOUT = "llm_timeout"                  # Predicted slow LLM response
    MEMORY_EXHAUSTION = "memory_exhaustion"      # Memory resource limits
    CONTEXT_OVERFLOW = "context_overflow"        # Context window limits
    # Social/Interaction
    USER_FRUSTRATION = "user_frustration"        # User experience harm
    REPEATED_FAILURE = "repeated_failure"        # Repeated failed attempts
    CONVERSATION_ABANDON = "conversation_abandon"  # Conversation dropped
    # Sensory
    TRACKING_LOSS = "tracking_loss"              # Lost visual tracking
    AUDIO_DISTORTION = "audio_distortion"        # Audio quality issues
    # Task/Goal
    TOOL_FAILURE = "tool_failure"                # Tool execution failure
    GOAL_UNREACHABLE = "goal_unreachable"        # Goal cannot be achieved
    # Generic
    UNKNOWN = "unknown"
```

---

## HarmPrediction

Represents a predicted harm outcome:

```python
@dataclass
class HarmPrediction:
    category: HarmCategory
    intensity: float       # 0.0-1.0 severity
    confidence: float      # 0.0-1.0 confidence in prediction
    reason: str            # Human-readable explanation
    source: str            # Which predictor generated this
    action_type: str = ""  # The action type that was evaluated
    action_params: dict = field(default_factory=dict)
    mitigation: str = ""   # Suggested mitigation action
    metadata: dict = field(default_factory=dict)

    # Computed properties:
    # risk_score -> intensity * confidence
    # should_gate -> risk_score >= 0.3
```

---

## HarmRegistry

Central coordinator for all harm predictors.

### Setup

```python
from maxim.harm import (
    HarmRegistry,
    MovementHarmPredictor,
    JointLimitHarmPredictor,
    get_global_registry,
)

# Create and configure
registry = HarmRegistry()
registry.register(MovementHarmPredictor())
registry.register(JointLimitHarmPredictor())

# Or use global singleton
registry = get_global_registry()
```

### Predicting Harm

```python
# Get worst (highest risk) prediction for an action
prediction = registry.predict_worst(
    action_type="movement",
    action_params={
        "action_signature": "look_at:dy=90:dp=30",
        "duration": 0.2,
    },
)

if prediction and prediction.intensity > 0.5:
    print(f"High risk: {prediction.reason}")
    print(f"Mitigation: {prediction.mitigation}")

    # Decide whether to proceed, modify, or abort

# Quick gating check
should_gate, reason = registry.should_gate(
    action_type="movement",
    action_params={...},
    threshold=0.3,
)
```

### Aggregated Predictions

When multiple predictors fire:

```python
from maxim.harm import AggregatedHarmPrediction

# All predictions from all registered predictors
result = registry.predict_all(
    action_type="movement",
    action_params={...},
)

# Aggregated result
print(f"Max intensity: {result.max_intensity}")
print(f"Max risk score: {result.max_risk_score}")
print(f"Should gate: {result.should_gate}")
print(f"Reasons: {result.reasons}")
```

---

## MovementHarmPredictor

Predicts harm from movement velocity and acceleration.

### Configuration

```python
from maxim.harm import MovementHarmPredictor, MovementHarmConfig

config = MovementHarmConfig(
    angular_velocity_threshold=100.0,       # deg/sec
    translation_velocity_threshold=50.0,    # mm/sec
    default_duration=0.3,                   # seconds
    intensity_scale_factor=0.005,           # excess velocity -> intensity
    confidence_base=0.8,                    # high confidence in physics-based prediction
)

predictor = MovementHarmPredictor(config)
```

### Prediction Logic

```python
# Given: look_at(dy=90, dp=30) in 0.2 seconds
# Angular delta = sqrt(90^2 + 30^2) ~ 95 deg
# Velocity = 95 / 0.2 = 474 deg/sec (exceeds 100 threshold)

prediction = predictor.predict(
    action_type="movement",
    action_params={
        "action_signature": "look_at:dy=90:dp=30",
        "duration": 0.2,
    },
)

# Returns: HarmPrediction(
#   category=MOVEMENT_VELOCITY,
#   intensity=min(1.0, (474-100)*0.005) = 1.0,
#   reason="Expected angular velocity 474 deg/s exceeds threshold 100 deg/s ...",
#   source="movement",
#   mitigation="Increase duration to 1.05s for safe velocity"
# )
```

---

## JointLimitHarmPredictor

Predicts harm from approaching joint position limits.

### Configuration

```python
from maxim.harm import JointLimitHarmPredictor, JointLimitConfig

config = JointLimitConfig(
    # Joint limits (degrees, conservative defaults)
    yaw_limit=45.0,
    pitch_limit_up=30.0,
    pitch_limit_down=-30.0,
    y_limit=20.0,             # mm
    z_limit_up=20.0,          # mm
    z_limit_down=-20.0,       # mm

    # Risk thresholds (fraction of limit)
    warning_threshold=0.7,    # 70% of limit = warning
    danger_threshold=0.9,     # 90% of limit = danger

    # Learning integration
    use_learned_bounds=True,

    # Confidence
    confidence_base=0.75,
)

predictor = JointLimitHarmPredictor(config)

# With WorkspaceBoundsLearner for learned limits
predictor = JointLimitHarmPredictor(
    config=config,
    bounds_learner=workspace_bounds_learner,
)
```

### Prediction Logic

```python
# Given: target_yaw = 42 (93% of 45 limit -> danger zone)
prediction = predictor.predict(
    action_type="movement",
    action_params={
        "target_yaw": 42.0,
        "target_pitch": 25.0,
    },
)

# Returns: HarmPrediction(
#   category=JOINT_LIMIT,
#   intensity=0.79,
#   reason="Movement near joint limits: yaw 42.0° at 93% of limit",
#   source="joint_limit",
#   mitigation="Reduce movement to yaw=31.5°, pitch=18.8°"
# )
```

---

## Custom Predictors

Create custom harm predictors for new domains:

```python
from maxim.harm import HarmPredictor, HarmPrediction, HarmCategory

class SpeechHarmPredictor(HarmPredictor):
    """Predicts harm from speech actions."""

    name = "speech"
    categories = {HarmCategory.USER_FRUSTRATION}

    def __init__(self, max_volume: float = 0.8):
        self.max_volume = max_volume

    def predict(self, action_type: str, action_params: dict,
                context: dict | None = None) -> HarmPrediction | None:
        if action_type != "speech":
            return None
        volume = action_params.get("volume", 0.5)

        if volume > self.max_volume:
            return HarmPrediction(
                category=HarmCategory.USER_FRUSTRATION,
                intensity=(volume - self.max_volume) / (1.0 - self.max_volume),
                confidence=0.7,
                reason=f"Volume {volume:.1f} may be too loud",
                source=self.name,
                action_type=action_type,
                action_params=action_params,
                mitigation=f"Reduce volume to {self.max_volume}",
            )

        return None

# Register with registry
registry.register(SpeechHarmPredictor())
```

---

## Integration with FearAgent

The FearAgent uses harm predictions for action gating:

```python
from maxim.harm import get_global_registry
from maxim.agents import FearAgent

# In FearAgent.review_action()
def review_action(self, action_type, action_params):
    # Query harm predictors
    harm_registry = get_global_registry()
    should_gate, reason = harm_registry.should_gate(
        action_type, action_params, threshold=0.3,
    )

    if should_gate:
        # Get detailed prediction for review
        prediction = harm_registry.predict_worst(action_type, action_params)
        findings.append(Finding(
            category=prediction.category,
            description=prediction.reason,
            severity=prediction.intensity,
        ))

    return ReviewResult(...)
```

---

## Data Flow

```
Action Request
      ↓
HarmRegistry.predict_all()
      ↓
┌─────────────────────────────────┐
│  MovementHarmPredictor          │
│  JointLimitHarmPredictor        │
│  [Custom predictors...]         │
└─────────────────────────────────┘
      ↓
AggregatedHarmPrediction
      ↓
FearAgent.review_action()
      ↓
Gate / Modify / Allow
```

---

## Harm vs Pain

| Aspect | Harm Prediction | Pain Detection |
|--------|-----------------|----------------|
| Timing | BEFORE action | DURING/AFTER action |
| Purpose | Prevention | Learning |
| System | `maxim.harm` | `maxim.proprioception` |
| Integration | FearAgent gating | NAc causal learning |

Both systems work together:
1. Harm prediction gates obviously dangerous actions
2. Pain detection catches unexpected harm
3. NAc learns from pain to improve future harm predictions

---

## Biological Inspiration

| Biological | Maxim Equivalent |
|------------|------------------|
| Nociceptors (prospective) | HarmPredictor |
| Pain anticipation | predict_all() / predict_worst() |
| Withdrawal reflex | Action gating |
| Learned avoidance | NAc integration |

The harm system enables Maxim to anticipate and avoid harm before it occurs, rather than only learning from painful experiences.
