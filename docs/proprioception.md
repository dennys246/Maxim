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
| `PerceivedPainAssessor` | `perceived_pain.py` | Predicts pain BEFORE action (Layer 1 — anticipatory) |
| `PainInterceptorExecutor` | `runtime/pain_interceptor.py` | Fires pain AFTER action (Layer 2 — consequence) |

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
    persist_path="~/.maxim/util/focus_learner.json",
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
learner.save("~/.maxim/util/focus_learner.json")

# Load existing gains
learner.load("~/.maxim/util/focus_learner.json")
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

See [bridges.md](archive/bridges.md) for PainCircuitBridge details.

---

## Persistence Files

| Component | Default Path | Contents |
|-----------|--------------|----------|
| FocusLearner | `~/.maxim/util/focus_learner.json` | Learned gains, sample history |
| PainDetector | `~/.maxim/util/pain_detector.json` | Pain thresholds, history |

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
| ACC / vmPFC anticipatory aversion | PerceivedPainAssessor (Layer 1) |
| Nociceptor firing | PainInterceptorExecutor (Layer 2) |

The goal is naturalistic, adaptive movement that learns from experience rather than requiring manual tuning.

---

## Two-Layer Predictive Pain Architecture

Two-layer pain system combines **innate priors** with **learned prediction** to produce a felt signal the AUT can reason about:

### Layer 1: Anticipatory Pain (before action)

`PerceivedPainAssessor` fires **before** a tool executes. Intensity = `max(learned_from_NAc, innate_prior)`:

- **Innate priors** — hard-coded per-path intensities (like `/etc/shadow` → 0.95). The AUT is "born with" this knowledge.
- **NAc learned prediction** — uses `nac.predict(event_signature="tool:X")` to read experience-based confidence. Grows as NAc sees more Layer-2 pain events for the same tool+context.

The signal has `PainType.ANTICIPATED` and flows through PainBus → hippocampus, making it available in the AUT's next LLM context. The AUT can then reason: "I anticipate 0.95 pain from `read_file(/etc/shadow)` — I should refuse."

### Layer 2: Consequence Pain (after action)

`PainInterceptorExecutor` wraps the AUT's executor and fires **after** a tool has touched a sensitive path. This is the ground-truth signal:

- Extracts paths + semantic operation (e.g. `rm -rf /etc/passwd` → `delete`)
- Matches against the same prior table used by Layer 1
- Publishes `PainType.EXTERNAL_SIGNAL` with `context.kind = "consequence"`
- NAc captures this via ToolPainBridge → forms `tool:X → NEGATIVE` link → feeds Layer 1's future predictions

### Learning Loop

```
Action executes → Layer 2 fires real pain → NAc updates causal link
     ↑                                                    │
     │                                                    ↓
     └─────────── Layer 1 predicts pain from NAc ←────────┘
```

The more the AUT touches sensitive paths, the stronger its anticipatory aversion becomes — mirroring how biological organisms develop conditioned aversion through experience.

### Path Extraction

Both layers use `extract_paths_from_params(tool_name, params)` which:

- Pulls paths from direct fields (`path`, `file`, `src`, `dest`, etc.)
- Parses bash sub-commands (`rm`, `cat`, `ls`, `cp`, `>`) and promotes the semantic operation (e.g. `bash` with `rm -rf X` becomes `delete` on `X` rather than `execute`)
- Returns `PathAccess(path, operation)` pairs so downstream code knows WHAT was done to each path, not just WHERE

### Usage (simulation wiring)

The sim orchestrator stacks both wrappers automatically for AUT executors:

```python
executor = Executor(registry)
executor._tool_pain_bridge = bridge                # NAc learning hook
executor = PainInterceptorExecutor(executor, ...)  # Layer 2
executor = AnticipatoryPainExecutor(executor, assessor=PerceivedPainAssessor(nac=nac))  # Layer 1
executor = FearGatedExecutor(executor, fear_agent) # safety gate (outermost)
```

See `src/maxim/simulation/orchestrator.py` for the actual wiring.
