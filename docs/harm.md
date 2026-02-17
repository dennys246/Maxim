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
```

---

## HarmPrediction

Represents a predicted harm outcome:

```python
@dataclass
class HarmPrediction:
    category: HarmCategory
    intensity: float       # 0.0-1.0 severity
    reason: str           # Human-readable explanation
    confidence: float     # How confident is this prediction
    suggested_mitigation: str | None  # How to reduce risk
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
# Query before executing action
prediction = registry.predict_harm(
    action_type="movement",
    action_params={
        "action_signature": "look_at:dy=90:dp=30",
        "duration": 0.2,
        "current_position": (0, 0),
        "target_position": (90, 30),
    },
)

if prediction.intensity > 0.5:
    print(f"High risk: {prediction.reason}")
    print(f"Mitigation: {prediction.suggested_mitigation}")

    # Decide whether to proceed, modify, or abort
```

### Aggregated Predictions

When multiple predictors fire:

```python
from maxim.harm import AggregatedHarmPrediction

# All predictions from all registered predictors
all_predictions = registry.predict_all_harm(
    action_type="movement",
    action_params={...},
)

# Aggregated result
print(f"Max intensity: {all_predictions.max_intensity}")
print(f"Categories: {all_predictions.categories}")
print(f"Should gate: {all_predictions.should_gate}")
```

---

## MovementHarmPredictor

Predicts harm from movement velocity and acceleration.

### Configuration

```python
from maxim.harm import MovementHarmPredictor, MovementHarmConfig

config = MovementHarmConfig(
    # Velocity thresholds (degrees/second)
    safe_velocity=50.0,
    warning_velocity=100.0,
    dangerous_velocity=200.0,

    # Acceleration thresholds (degrees/second²)
    safe_acceleration=100.0,
    warning_acceleration=300.0,

    # Duration factors
    min_safe_duration=0.2,  # seconds
)

predictor = MovementHarmPredictor(config)
```

### Prediction Logic

```python
# Given: look_at(dy=90) in 0.2 seconds
# Velocity = 90 / 0.2 = 450 deg/sec (DANGEROUS)

prediction = predictor.predict(
    action_params={
        "target_delta": 90,
        "duration": 0.2,
    }
)

# Returns: HarmPrediction(
#   category=PHYSICAL_DAMAGE,
#   intensity=0.85,
#   reason="Velocity 450 deg/s exceeds safe limit (200)",
#   suggested_mitigation="Increase duration to 0.9 seconds"
# )
```

---

## JointLimitHarmPredictor

Predicts harm from approaching joint position limits.

### Configuration

```python
from maxim.harm import JointLimitHarmPredictor, JointLimitConfig

config = JointLimitConfig(
    # Joint limits (degrees)
    yaw_limits=(-90.0, 90.0),
    pitch_limits=(-45.0, 45.0),
    roll_limits=(-30.0, 30.0),

    # Warning margins
    soft_limit_margin=10.0,   # Warning zone
    hard_limit_margin=5.0,    # Danger zone
)

predictor = JointLimitHarmPredictor(config)
```

### Prediction Logic

```python
# Given: target_yaw = 85 (close to limit of 90)
prediction = predictor.predict(
    action_params={
        "target_position": (85, 0, 0),  # yaw, pitch, roll
    }
)

# Returns: HarmPrediction(
#   category=PHYSICAL_DAMAGE,
#   intensity=0.6,
#   reason="Target yaw 85° is within 5° of limit (90°)",
#   suggested_mitigation="Target yaw 80° instead"
# )
```

---

## Custom Predictors

Create custom harm predictors for new domains:

```python
from maxim.harm import HarmPredictor, HarmPrediction, HarmCategory

class SpeechHarmPredictor(HarmPredictor):
    """Predicts harm from speech actions."""

    action_type = "speech"

    def __init__(self, max_volume: float = 0.8):
        self.max_volume = max_volume

    def predict(self, action_params: dict) -> HarmPrediction | None:
        volume = action_params.get("volume", 0.5)

        if volume > self.max_volume:
            return HarmPrediction(
                category=HarmCategory.SOCIAL_HARM,
                intensity=(volume - self.max_volume) / (1.0 - self.max_volume),
                reason=f"Volume {volume:.1f} may be too loud",
                confidence=0.7,
                suggested_mitigation=f"Reduce volume to {self.max_volume}",
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
    prediction = harm_registry.predict_harm(action_type, action_params)

    if prediction and prediction.intensity > 0.5:
        # Add to review findings
        findings.append(Finding(
            category=DangerCategory.PHYSICAL_DAMAGE,
            description=prediction.reason,
            severity=RiskLevel.from_intensity(prediction.intensity),
        ))

    return ReviewResult(...)
```

---

## Data Flow

```
Action Request
      ↓
HarmRegistry.predict_harm()
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
| Pain anticipation | predict_harm() |
| Withdrawal reflex | Action gating |
| Learned avoidance | NAc integration |

The harm system enables Maxim to anticipate and avoid harm before it occurs, rather than only learning from painful experiences.
