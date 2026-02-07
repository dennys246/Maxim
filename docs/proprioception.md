# Proprioception System

The proprioception module provides body awareness and pain detection, enabling the robot to learn from and avoid harmful movement patterns.

## Overview

Proprioception in Maxim mirrors biological proprioception - the sense of body position and movement. This module tracks:

1. **Position and velocity** of head/body components
2. **Error patterns** between commanded and actual positions
3. **Pain signals** from aversive movement patterns

## Components

| Component | File | Purpose |
|-----------|------|---------|
| `FocusLearner` | `focus_learner.py` | Learns optimal gain parameters via Rescorla-Wagner |
| `MovementTracker` | `movement_tracker.py` | Tracks position history and computes velocity |
| `PainDetector` | `pain.py` | Detects aversive movement patterns |

---

## FocusLearner

The FocusLearner uses **Rescorla-Wagner learning** to adaptively adjust movement gains based on tracking error:

```
ΔV = α(λ - V)
```

Where:
- `α` = learning rate (0.0-1.0)
- `λ` = target value (ideal gain that minimizes error)
- `V` = current gain estimate

### Key Features

1. **Error-Based Learning**: Adjusts gains based on actual vs expected positions
2. **Bounded Convergence**: Gains stay within safe operational limits
3. **Persistence**: Saves learned gains to disk for cross-session learning

### Configuration

```python
from maxim.proprioception import FocusLearner, FocusLearnerConfig

config = FocusLearnerConfig(
    min_yaw_gain=0.1,
    max_yaw_gain=3.0,
    min_pitch_gain=0.1,
    max_pitch_gain=3.0,
    learning_rate=0.15,           # Rescorla-Wagner α
    error_scaling_factor=0.01,    # Converts pixels to gain delta
    sample_threshold=5,           # Min samples before adjusting
    persist_path="data/util/focus_learner.json",
)

learner = FocusLearner(config)
```

### Usage

```python
# Record tracking sample
learner.record_sample(
    expected_yaw=45.0,
    actual_yaw=42.5,
    expected_pitch=10.0,
    actual_pitch=11.2,
)

# Get current gains
yaw_gain, pitch_gain = learner.get_gains()

# Apply to movement calculation
commanded_yaw = error_yaw * yaw_gain
```

### Persistence

FocusLearner automatically saves learned gains:

```python
# Manual save
learner.save("data/util/focus_learner.json")

# Load existing gains
learner.load("data/util/focus_learner.json")
```

---

## MovementTracker

Tracks position history and computes movement metrics for pain detection.

### Key Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `angular_velocity` | deg/sec | Combined yaw + pitch rate |
| `translation_velocity` | mm/sec | Combined x + y + z rate |
| `angular_acceleration` | deg/sec² | Rate of velocity change |
| `direction_reversals` | count | Sign changes in velocity |

### Usage

```python
from maxim.proprioception import MovementTracker, MovementSample

tracker = MovementTracker(window_seconds=2.0, sample_limit=100)

# Record position updates
metrics = tracker.record_position(
    yaw=45.0,
    pitch=10.0,
    x=0.0, y=0.0, z=0.0,
    roll=0.0,
)

if metrics:
    print(f"Angular velocity: {metrics.angular_velocity:.1f} deg/sec")
```

---

## PainDetector

Detects pain signals from movement metrics using configurable thresholds.

### Pain Types

| Type | Trigger | Description |
|------|---------|-------------|
| `EXCESSIVE_VELOCITY` | Angular velocity > threshold | Too-fast movement |
| `DIRECTION_THRASHING` | Rapid reversals | Back-and-forth oscillation |
| `SUSTAINED_STRAIN` | Prolonged near-limit | Extended uncomfortable position |

### Configuration

```python
from maxim.proprioception import PainDetector, PainConfig

config = PainConfig(
    angular_velocity_pain=100.0,     # deg/sec threshold
    translation_velocity_pain=50.0,  # mm/sec threshold
    reversal_pain_threshold=3,       # Reversals in window
    pain_scaling_factor=0.01,        # Velocity → intensity
    pain_cooldown_seconds=0.5,       # Min time between signals
)

detector = PainDetector(config)
```

### Usage

```python
# Record position and check for pain
pain_signal = detector.record_position(
    yaw=45.0, pitch=10.0,
    x=0.0, y=0.0, z=0.0, roll=0.0,
)

if pain_signal:
    print(f"Pain: {pain_signal.pain_type.value}")
    print(f"Intensity: {pain_signal.intensity:.2f}")

# Register callback for pain events
def on_pain(signal):
    logger.warning(f"Pain detected: {signal.pain_type}")

detector.add_pain_callback(on_pain)
```

---

## Integration with NAc

Pain signals integrate with the NAc (Nucleus Accumbens) for causal learning:

```
Position Updates → MovementTracker → PainDetector → PainSignal
                                          ↓
                                    PainCircuitBridge
                                    ├── NAc.record_outcome() (learning)
                                    └── NAc.predict() (avoidance)
```

See [bridges.md](bridges.md) for PainCircuitBridge details.

---

## Persistence Files

| Component | Default Path | Contents |
|-----------|--------------|----------|
| FocusLearner | `data/util/focus_learner.json` | Learned gains, sample history |
| PainDetector | `data/util/pain_detector.json` | Pain thresholds, history |

Clear with: `maxim --clear-memory focus,pain`

---

## Biological Inspiration

The proprioception system mirrors several biological mechanisms:

| Biological | Maxim Equivalent |
|------------|------------------|
| Muscle spindles | Position tracking |
| Golgi tendon organs | Velocity/acceleration sensing |
| Pain receptors | PainDetector thresholds |
| Cerebellar adaptation | Rescorla-Wagner learning |

The goal is naturalistic, adaptive movement that learns from experience rather than requiring manual tuning.
