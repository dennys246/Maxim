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
from maxim.default_network.arbiter import ArbiterConfig
from maxim.default_network.gate import GateConfig, AdaptiveThresholdConfig

config = DefaultNetworkConfig(
    enabled=True,
    update_hz=30.0,
    auto_release_timeout=5.0,       # Max inhibition duration before auto-release
    publish_actions=True,
    fear_gate_enabled=True,

    # Sub-component configs
    arbiter=ArbiterConfig(),
    gate=GateConfig(),
    adaptive_threshold=AdaptiveThresholdConfig(),

    # Feature flags
    gaze_salience_enabled=True,
    idle_exploration_enabled=True,
    spatial_map_enabled=True,
    bounds_learning_enabled=True,
    attention_network_enabled=True,
    salience_network_enabled=True,
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
    novelty_threshold=0.7,
    salience_threshold=0.6,
    anomaly_threshold=0.7,
    safety_velocity_threshold=200.0,    # pixels/frame for rapid approach
    max_recent_percepts=100,
    adaptive=True,                       # Enable adaptive thresholding
)

adaptive_config = AdaptiveThresholdConfig(
    base_novelty_threshold=0.7,
    base_salience_threshold=0.6,
    min_threshold=0.3,
    max_threshold=0.95,
    adaptation_rate=0.1,
    escalation_rate_target=0.05,         # Target 5% escalation rate
    window_seconds=60.0,
    adaptation_interval=5.0,             # How often to adapt
    persist_path="~/.maxim/util/adaptive_thresholds.json",
    auto_save_interval=60.0,
)

gate = ThalamicGate(gate_config, adaptive_config=adaptive_config)
```

### Filtering

```python
# Evaluate a percept for escalation
result = gate.evaluate(percept)

if result.escalated:
    # Create filtered percept for the deliberative layer
    filtered = gate.create_filtered_percept(percept, result)
    # Send to LLM for deliberation
    agent.process(filtered)
else:
    # Handle reactively in DN (reason will be "routine")
    dn.handle_locally(percept)

# Set active goal for relevance matching
gate.set_active_goal("find the red cup")

# Force escalation for specific tracks
gate.add_attention_lock(track_id=42)
```

---

## PriorityArbiter

Selects the winning behavior with hysteresis to prevent rapid switching.

### Configuration

```python
from maxim.default_network import PriorityArbiter, ArbiterConfig

config = ArbiterConfig(
    hysteresis_bonus=0.1,       # Priority bonus for current behavior to prevent rapid switching
    min_switch_interval=0.3,    # Minimum seconds between behavior switches
    score_threshold=0.1,        # Minimum effective score to act
)

arbiter = PriorityArbiter(config)
```

### Arbitration

```python
# Get proposals from all behaviors
proposals = [b.evaluate(detections, state) for b in behaviors]
proposals = [p for p in proposals if p is not None]

# Arbiter selects winner (applies hysteresis + cooldown)
winner = arbiter.select(proposals)

if winner:
    execute_action(winner)

# Force a behavior switch (bypasses cooldown)
arbiter.force_switch("startle")

# Check stability
print(f"Current: {arbiter.current_behavior}")
print(f"Stability: {arbiter.behavior_stability:.2f}")
```

---

## Behaviors

Reactive behavior modules that propose actions based on current state.

### Available Behaviors

| Behavior | Priority | Trigger |
|----------|----------|---------|
| `StartleResponse` | 0.95 | Sudden peripheral appearance |
| `SocialAttention` | 0.9 | Face/person detected |
| `OrientingResponse` | 0.8 | Novel object appears |
| `MotionTracking` | 0.7 | Moving object detected |
| `TurnAround` | 0.3 | Head at yaw limit with interesting edge detection |
| `IdleScan` | 0.2 | No interesting stimuli after timeout |
| `ReturnToCenter` | 0.2 | Head drifted beyond threshold |
| `Microsaccades` | 0.1 | During prolonged fixation |

### Behavior Interface

```python
from maxim.default_network import Behavior, BehaviorState, ActionProposal

class CustomBehavior(Behavior):
    name = "custom"
    base_priority = 0.5
    cooldown_seconds = 0.5

    def evaluate(
        self,
        detections: list[dict],
        state: BehaviorState,
    ) -> ActionProposal | None:
        """Evaluate detections and propose an action if triggered."""
        if not self.can_activate():
            return None

        # Analyze detections and state...
        return self._create_proposal(
            action_type="look_at",
            target=(target_u, target_v),
            priority_scale=1.0,
            confidence=0.8,
        )
```

### Behavior Configuration

```yaml
# ~/.maxim/config/default_network.yaml
default_network:
  enabled: true
  update_hz: 30.0
  behaviors:
    orienting:
      enabled: true
      priority: 0.8
      novelty_threshold: 1.2
      min_confidence: 0.4
      cooldown_seconds: 0.5
    social:
      enabled: true
      priority: 0.9
      prefer_faces: true
      tracking_hysteresis: 0.1
    turn_around:
      enabled: true
      priority: 0.3
      yaw_threshold: 0.85
      edge_threshold: 0.15
      turn_angle: 90.0
      cooldown_seconds: 10.0
  arbiter:
    hysteresis_bonus: 0.1
    min_switch_interval: 0.3
  gate:
    novelty_threshold: 0.7
    salience_threshold: 0.6
    adaptive: true
```

---

## Specific Behaviors

### OrientingResponse

Quickly orients to novel stimuli:

```python
from maxim.default_network import OrientingResponse

orienting = OrientingResponse(
    novelty_tracker=novelty_tracker,
    novelty_threshold=1.2,
    min_confidence=0.4,
)
orienting.base_priority = 0.8
orienting.cooldown_seconds = 0.5
```

### SocialAttention

Prioritizes faces and people:

```python
from maxim.default_network import SocialAttention

social = SocialAttention(
    prefer_faces=True,
    tracking_hysteresis=0.1,
)
social.base_priority = 0.9
social.cooldown_seconds = 0.2
```

### StartleResponse

Rapid response to sudden peripheral appearances:

```python
from maxim.default_network import StartleResponse

startle = StartleResponse(
    peripheral_threshold=0.7,
    appearance_window=0.3,
    min_confidence=0.5,
    frame_size=(640, 480),
)
startle.base_priority = 0.95
startle.cooldown_seconds = 2.0
```

### TurnAround

Rotates the body when the head is at its yaw limit and there is something interesting beyond what the head can see:

```python
from maxim.default_network.behaviors.turn_around import TurnAround

turn = TurnAround(
    yaw_threshold=0.85,       # Trigger at 85% of yaw limit
    edge_threshold=0.15,      # Detection within 15% of frame edge
    turn_angle=90.0,          # Degrees to rotate body
    base_duration=5.0,        # Seconds for the turn
    duration_jitter=1.0,      # Random +/- seconds for natural feel
    max_yaw=55.0,             # Maximum yaw angle (workspace limit)
    image_width=640,
)
turn.cooldown_seconds = 10.0  # Don't turn too frequently
```

The turn is slow and deliberate (5 +/- 1 seconds) to appear natural. Only horizontal yaw limits trigger this behavior -- pitch limits do not, since body rotation is around the vertical axis.

### IdleScan

Exploratory scanning when nothing is interesting:

```python
from maxim.default_network import IdleScan

idle = IdleScan(
    idle_timeout=5.0,     # Seconds before scanning starts
)
idle.base_priority = 0.2
idle.cooldown_seconds = 0.5
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
# Path: ~/.maxim/util/adaptive_thresholds.json
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
