# Default Network

The Default Network (DN) is a biologically-inspired reactive behavior layer that handles routine visual behaviors without requiring LLM deliberation.

## Overview

The Default Network provides:

1. **Reactive Behaviors**: Fast, pre-programmed responses to stimuli
2. **Thalamic Gating**: Filters percepts before escalating to deliberation
3. **Priority Arbitration**: Selects winning behavior with hysteresis
4. **Naturalistic Movement**: Human-like saccade-fixate dynamics

## Architecture

```
Visual Input → SalienceNetwork → ThalamicGate
                     ↓                 ↓
              AttentionNetwork    Escalate to LLM?
                     ↓                 ↓
              PriorityArbiter ← FilteredPercept
                     ↓
              Winning Behavior
                     ↓
              Motor Commands
```

## Components

| Component | File | Purpose |
|-----------|------|---------|
| `DefaultNetwork` | `network.py` | Main coordinator |
| `ThalamicGate` | `gate.py` | Percept filtering and escalation |
| `PriorityArbiter` | `arbiter.py` | Behavior selection |
| `Behaviors` | `behaviors/` | Individual behavior modules |

---

## DefaultNetwork

The main coordinator that orchestrates all DN components.

### Configuration

```python
from maxim.default_network import DefaultNetwork, DefaultNetworkConfig

config = DefaultNetworkConfig(
    # Gate thresholds
    novelty_threshold=0.5,
    salience_threshold=0.6,

    # Arbiter settings
    switch_hysteresis=0.15,

    # Update rates
    attention_update_hz=10.0,
    behavior_update_hz=30.0,
)

dn = DefaultNetwork(config)
```

### Main Loop

```python
# Process a frame
action = dn.process_frame(
    frame=camera_frame,
    detections=yolo_detections,
    current_position=(yaw, pitch),
)

if action:
    # Execute the winning behavior's action
    motor.execute(action)
```

---

## ThalamicGate

Filters percepts to determine what requires deliberative attention.

### Adaptive Thresholds

The gate uses adaptive thresholds that learn from experience:

```python
from maxim.default_network import ThalamicGate, GateConfig, AdaptiveThresholdConfig

gate_config = GateConfig(
    base_novelty_threshold=0.5,
    base_salience_threshold=0.6,
    adaptation_rate=0.1,
)

adaptive_config = AdaptiveThresholdConfig(
    min_novelty=0.3,
    max_novelty=0.9,
    min_salience=0.3,
    max_salience=0.9,
    adapt_interval_seconds=60.0,
    persist_path="data/util/adaptive_thresholds.json",
)

gate = ThalamicGate(gate_config, adaptive_config)
```

### Filtering

```python
# Filter a percept
result = gate.filter(
    percept=detection,
    novelty_score=0.7,
    salience_score=0.8,
)

if result.should_escalate:
    # Send to LLM for deliberation
    agent.process(result.percept)
else:
    # Handle reactively in DN
    dn.handle_locally(result.percept)
```

---

## PriorityArbiter

Selects the winning behavior with hysteresis to prevent rapid switching.

### Configuration

```python
from maxim.default_network import PriorityArbiter, ArbiterConfig

config = ArbiterConfig(
    switch_threshold=0.15,      # Minimum advantage to switch
    decay_rate=0.95,            # Per-frame decay
    history_weight=0.3,         # Weight of recent history
)

arbiter = PriorityArbiter(config)
```

### Arbitration

```python
# Get proposals from all behaviors
proposals = [b.propose(state) for b in behaviors]

# Arbiter selects winner
winner = arbiter.arbitrate(proposals)

# Winner persists until outcompeted by switch_threshold
```

---

## Behaviors

Reactive behavior modules that propose actions based on current state.

### Available Behaviors

| Behavior | Priority | Trigger |
|----------|----------|---------|
| `StartleResponse` | 100 | Sudden large motion |
| `SocialAttention` | 80 | Face/person detected |
| `OrientingResponse` | 70 | Novel object appears |
| `MotionTracking` | 60 | Moving object detected |
| `IdleScan` | 30 | No interesting stimuli |
| `Microsaccades` | 20 | During fixation |
| `ReturnToCenter` | 10 | Been looking away too long |

### Behavior Interface

```python
from maxim.default_network import Behavior, BehaviorState

class CustomBehavior(Behavior):
    name = "custom"
    priority = 50

    def update(self, state: BehaviorState) -> None:
        """Update internal state based on current perception."""
        self._activation = self._compute_activation(state)

    def propose(self, state: BehaviorState) -> ActionProposal | None:
        """Propose an action if activated."""
        if self._activation < self.activation_threshold:
            return None

        return ActionProposal(
            behavior=self.name,
            priority=self.priority * self._activation,
            target_position=(target_yaw, target_pitch),
            duration=0.3,
        )
```

### Behavior Configuration

```yaml
# data/util/dn_config.yaml
behaviors:
  orienting:
    enabled: true
    activation_threshold: 0.4
    priority: 70

  social:
    enabled: true
    activation_threshold: 0.3
    priority: 80
    face_boost: 1.5

  idle_scan:
    enabled: true
    interval_seconds: 3.0
    scan_range: 45.0
```

---

## Specific Behaviors

### OrientingResponse

Quickly orients to novel stimuli:

```python
from maxim.default_network import OrientingResponse

orienting = OrientingResponse(
    activation_threshold=0.4,
    novelty_weight=0.6,
    salience_weight=0.4,
)
```

### SocialAttention

Prioritizes faces and people:

```python
from maxim.default_network import SocialAttention

social = SocialAttention(
    face_priority=1.5,    # Boost for faces
    person_priority=1.2,  # Boost for people
    gaze_duration=2.0,    # How long to maintain contact
)
```

### StartleResponse

Rapid response to sudden stimuli:

```python
from maxim.default_network import StartleResponse

startle = StartleResponse(
    motion_threshold=50.0,   # Pixels/frame to trigger
    cooldown_seconds=1.0,    # Min time between startles
    response_magnitude=0.8,  # Movement intensity
)
```

### IdleScan

Exploratory scanning when nothing is interesting:

```python
from maxim.default_network import IdleScan

idle = IdleScan(
    scan_interval=3.0,    # Seconds between scans
    scan_range=45.0,      # Degrees to scan
    prefer_unexplored=True,
)
```

---

## Background Tasks

DN manages background operations via throttled task manager:

```python
from maxim.default_network import BackgroundTaskManager, ThrottleConfig

config = ThrottleConfig(
    salience_update_hz=5.0,
    attention_update_hz=10.0,
    behavior_update_hz=30.0,
)

manager = BackgroundTaskManager(config)
manager.start()
```

---

## Movement Utilities

Helper functions for computing movements:

```python
from maxim.default_network import (
    compute_dynamic_duration,
    compute_opposite_position,
    get_quadrant,
    suggest_exploration_direction,
)

# Dynamic duration based on distance
duration = compute_dynamic_duration(
    current=(0, 0),
    target=(45, 20),
    min_duration=0.1,
    max_duration=0.5,
)

# Get exploration direction
direction = suggest_exploration_direction(
    gaze_history=history,
    attention_map=attention,
)
```

---

## Integration

The Default Network integrates with:

| System | Integration |
|--------|-------------|
| **SalienceNetwork** | Object-level salience scores |
| **AttentionNetwork** | Spatial attention tracking |
| **GazeController** | Human-like saccade dynamics |
| **Hippocampus** | Memory-informed behavior |
| **NAc** | Learned preferences |

---

## Persistence

Adaptive thresholds persist across sessions:

```python
# Auto-saved every 60 seconds when thresholds adapt
# Path: data/util/adaptive_thresholds.json
```

Clear with: `maxim --clear-memory threshold`

---

## Biological Inspiration

| Biological | Maxim Equivalent |
|------------|------------------|
| Default Mode Network | DefaultNetwork coordinator |
| Thalamus | ThalamicGate filtering |
| Superior colliculus | OrientingResponse |
| FEF (frontal eye fields) | GazeController |
| Amygdala | StartleResponse, SocialAttention |

The DN provides fast, naturalistic behavior without the latency of LLM deliberation, similar to how biological default mode networks handle routine cognition.
