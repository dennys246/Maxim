# Attention System

The attention module manages WHERE the robot attends in visual space, providing spatial attention tracking and gaze control.

## Overview

The attention system provides:

1. **Spatial Attention Tracking**: Grid-based attention map
2. **Gaze History**: Temporal tracking with inhibition-of-return
3. **Human-like Gaze**: Saccade-fixate dynamics
4. **Scene Context**: Detect scene changes for exploration

## Components

| Component | File | Purpose |
|-----------|------|---------|
| `AttentionNetwork` | `attention_network.py` | Grid-based spatial attention |
| `GazeHistory` | `gaze_history.py` | Temporal gaze tracking |
| `SalienceMap` | `salience_map.py` | Unified spatial salience |
| `GazeController` | `gaze_controller.py` | Saccade-fixate dynamics |
| `SceneContextDetector` | `scene_context.py` | Scene change detection |

---

## AttentionNetwork

Grid-based spatial attention tracking over the visual field.

### Configuration

```python
from maxim.attention import AttentionNetwork, AttentionConfig

config = AttentionConfig(
    grid_size=10,                    # Cells per dimension (grid_size x grid_size)
    image_width=640,                 # Expected image width in pixels
    image_height=480,                # Expected image height in pixels
    dwell_decay_seconds=2.0,         # Time for dwell/visit salience to decay
    reachability_decay_seconds=60.0, # Time for failure penalties to decay
    min_dwell_seconds=0.5,           # Minimum time to stay at a position
    default_safe_range=(0.25, 0.25, 0.75, 0.75),  # Normalized coords
)

attention = AttentionNetwork(config)
```

### Attention Cells

Each grid cell tracks attention state:

```python
@dataclass
class AttentionCell:
    grid_u: int
    grid_v: int
    visit_count: int = 0       # Total visits
    success_count: int = 0     # Successful gaze movements
    failure_count: int = 0     # Failed movements (IK failures)
    last_visit: float = 0.0    # Timestamp of last visit
    last_success: float = 0.0  # Timestamp of last success
    last_failure: float = 0.0  # Timestamp of last failure
    total_dwell: float = 0.0   # Cumulative dwell time
```

### Usage

```python
# Record a gaze at pixel coordinates (u, v)
attention.record_gaze(position=(320, 240), success=True, duration=0.5)

# Query reachability and visit salience for a position
reach = attention.get_reachability(position=(320, 240))
salience = attention.get_visit_salience(position=(320, 240))

# Check if we should move to a target
should, reason = attention.should_move(target=(400, 300))

# Get next attention target (weighted random selection)
target = attention.get_next_target(avoid_current=True, prefer_unexplored=True)

# Query attention grid with filters
cells = attention.query(min_reachability=0.5, min_salience=0.3)

# Get context string for LLM consumption
context = attention.to_context_str()
```

### Inhibition of Return (IOR)

Recently visited positions have low visit salience, which recovers exponentially
over `dwell_decay_seconds`:

```python
# After recording a gaze, visit salience at that position drops
attention.record_gaze(position=(320, 240), success=True)

# Visit salience is low immediately after (recently visited)
salience = attention.get_visit_salience(position=(320, 240))
print(f"Salience: {salience:.2f}")  # Close to 0

# Salience recovers over time via exponential decay
# salience = 1.0 - exp(-time_since / dwell_decay_seconds)
```

---

## GazeHistory

Temporal tracking of gaze positions.

### Configuration

```python
from maxim.attention import GazeHistory, GazeHistoryConfig

config = GazeHistoryConfig(
    decay_seconds=2.0,          # Time for salience to recover (~63%)
    min_salience_to_move=0.3,   # Minimum salience required to move
    spatial_sigma=80.0,         # Gaussian spatial similarity (pixels)
    max_history_size=50,        # Maximum positions to track
    dwell_time_seconds=0.5,     # Minimum time at position before moving
)

history = GazeHistory(config)
```

### GazePosition

```python
@dataclass
class GazePosition:
    u: float                # Image x coordinate
    v: float                # Image y coordinate
    timestamp: float        # When gaze occurred (default: time.time())
```

### Usage

```python
# Record gaze position (u, v image coordinates)
history.record_gaze(position=(320, 240))

# Compute salience for a potential target (0=recently visited, 1=novel)
salience = history.get_salience(target=(400, 300))

# Check if a movement should be executed (considers salience + dwell time)
should, salience, reason = history.should_move(target=(400, 300))
if should:
    # Execute movement, then record
    history.record_gaze(position=(400, 300))

# Query current state
pos = history.get_current_position()    # Current gaze position
elapsed = history.time_since_last_move() # Seconds since last move
size = history.history_size              # Number of positions in history
```

---

## SalienceMap

Unified spatial salience combining all attention factors.

### Configuration

```python
from maxim.attention import SalienceMap, SalienceMapConfig

config = SalienceMapConfig(
    grid_width=16,              # Horizontal cells
    grid_height=12,             # Vertical cells
    frame_width=640.0,          # Image width in pixels
    frame_height=480.0,         # Image height in pixels

    # Weight factors (should roughly sum to 1.0)
    novelty_weight=0.25,        # Object novelty
    social_weight=0.35,         # Face/person presence
    movement_weight=0.15,       # Object motion
    unexplored_weight=0.15,     # Unvisited areas
    center_bias_weight=0.10,    # Center preference

    inhibition_decay_seconds=1.5,  # Visited location recovery time
    social_radius_cells=2,         # Person influence spread
    min_salience_floor=0.1,        # Minimum salience for any cell
    temperature_default=1.0,       # Default sampling temperature
)

salience_map = SalienceMap(config)
```

### Salience Cells

```python
@dataclass
class SalienceCell:
    # Component contributions (before weighting)
    novelty: float = 0.0
    social: float = 0.0
    movement: float = 0.0
    unexplored: float = 1.0    # Start as unexplored
    center_bias: float = 0.0

    # Temporal state
    last_visited: float = 0.0
    visit_count: int = 0

    # Computed total
    total_salience: float = 0.0
```

### Usage

```python
# Update with current perception
salience_map.update(
    detections=[
        {"bbox_xyxy": (100, 50, 200, 150), "class_id": 0, "track_id": 1},
        {"bbox_xyxy": (300, 200, 350, 280), "class_id": 41, "track_id": 2},
    ],
    gaze_history=gaze_history,
    novelty_tracker=novelty_tracker,
    movement_detector=movement_detector,
)

# Get most salient location (winner-take-all)
peak = salience_map.get_peak_target()  # Returns (x, y) pixels
print(f"Peak at ({peak[0]:.0f}, {peak[1]:.0f})")

# Sample probabilistically for natural variety
target = salience_map.sample_target(temperature=0.8, avoid_current=True)

# Get salience at a specific position
sal = salience_map.get_salience_at(position=(320, 240))

# Get detailed breakdown for a position
info = salience_map.get_cell_info(position=(320, 240))
# {'novelty': 0.5, 'social': 0.0, 'movement': 0.1, ...}

# Get top N most salient targets
top = salience_map.get_top_n_targets(n=5)  # [((x, y), salience), ...]
```

---

## GazeController

Human-like saccade-fixate dynamics.

### Configuration

```python
from maxim.attention import GazeController, GazeControllerConfig

config = GazeControllerConfig(
    # Fixation dynamics (human average ~250-350ms)
    min_fixation_ms=200.0,              # Minimum fixation duration
    max_fixation_ms=800.0,              # Maximum fixation duration
    mean_fixation_ms=350.0,             # Mean fixation duration
    fixation_std_ms=100.0,              # Standard deviation

    # Saccade dynamics
    saccade_speed_multiplier=2.0,       # Faster than normal tracking
    saccade_threshold_pixels=30.0,      # Min pixel distance for saccade

    # Exploration mode
    exploration_trigger_seconds=2.0,    # Idle time before exploring
    exploration_speed_multiplier=0.5,   # Slow, leisurely scanning

    # Interruption
    interrupt_salience_ratio=1.5,       # 50% more salient to interrupt
    sample_temperature=0.8,             # Sampling temperature
)

controller = GazeController(salience_map=salience_map, config=config)
```

### Gaze States

```python
from maxim.attention import GazeState

class GazeState(Enum):
    FIXATING = "fixating"       # Holding gaze on target
    SACCADING = "saccading"     # Rapid movement to new target
    EXPLORING = "exploring"    # Slow scan when nothing salient
```

### Usage

```python
# Update each frame - returns GazeCommand or None
cmd = controller.update(current_gaze=(320, 240), dt=0.033)

if cmd is not None:
    if cmd.action_type == "saccade":
        # Rapid movement to cmd.target at cmd.speed_multiplier
        motor.move(cmd.target, speed=cmd.speed_multiplier)
    elif cmd.action_type == "explore":
        # Slow exploratory movement
        motor.move(cmd.target, speed=cmd.speed_multiplier)

# Force a saccade to a specific target
cmd = controller.update(force_target=(400, 300))

# Query current state
print(f"State: {controller.state}")       # GazeState enum
print(f"Target: {controller.current_target}")

# Note external activity (resets exploration timer)
controller.note_activity()

# Get statistics
stats = controller.get_stats()
```

---

## SceneContextDetector

Detects scene changes for exploration mode.

### Configuration

```python
from maxim.attention import SceneContextDetector, SceneContextConfig

config = SceneContextConfig(
    change_threshold=0.4,              # Fraction of scene change to trigger (0-1)
    min_detections_for_comparison=2,   # Minimum detections for valid comparison
    scene_stability_seconds=2.0,       # Time before scene considered stable
    histogram_memory=5,                # Number of class histograms to remember
    position_change_threshold=30.0,    # Head yaw change (degrees) to trigger scan
)

detector = SceneContextDetector(config)
```

### Scene Snapshots

```python
@dataclass
class SceneSnapshot:
    timestamp: float
    class_histogram: dict[int, int]  # class_id -> count
    total_detections: int
    track_ids: set[int]
    head_yaw: float | None = None
    head_pitch: float | None = None
```

### Usage

```python
# Update with current detections and head position
is_new_scene = detector.update(
    detections=yolo_detections,
    head_yaw=current_yaw,
    head_pitch=current_pitch,
)

if is_new_scene:
    print("Scene changed! Triggering exploration...")

# Query state
print(f"Scene age: {detector.get_scene_age():.1f}s")
print(f"Scanning: {detector.is_scanning()}")

# Force a scene change (e.g., after robot turn_around)
detector.force_scene_change()

# Get statistics
stats = detector.get_stats()
```

---

## Integration

The attention system integrates with:

| System | Integration |
|--------|-------------|
| **SalienceNetwork** | Object-level salience scores |
| **DefaultNetwork** | Behavior gaze targets |
| **Hippocampus** | Spatial memory indexing |
| **GazeHistory** | Inhibition of return |

### Flow

```
Visual Input
     ↓
YOLO Detections → SalienceNetwork (what)
     ↓                    ↓
AttentionNetwork ← SalienceMap (where)
     ↓
GazeHistory (when)
     ↓
GazeController → Motor Commands
```

---

## Biological Inspiration

| Biological | Maxim Equivalent |
|------------|------------------|
| Superior colliculus | AttentionNetwork |
| Frontal eye fields | GazeController |
| Parietal cortex | SalienceMap |
| Inhibition of return | GazeHistory IOR |
| Saccades/fixations | GazeState machine |

The attention system enables naturalistic, human-like gaze behavior that explores efficiently while attending to relevant stimuli.
