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
    grid_rows=10,           # Vertical divisions
    grid_cols=15,           # Horizontal divisions
    decay_rate=0.95,        # Per-frame decay
    boost_amount=0.3,       # Attention boost on gaze
    inhibition_radius=2,    # Cells to inhibit around gaze
)

attention = AttentionNetwork(config)
```

### Attention Cells

Each grid cell tracks attention state:

```python
@dataclass
class AttentionCell:
    row: int
    col: int
    attention: float        # 0.0-1.0 current attention
    last_visited: float     # Timestamp of last gaze
    visit_count: int        # Total visits
    inhibition: float       # Temporary inhibition (IOR)
```

### Usage

```python
# Update attention based on gaze
attention.gaze_at(row=5, col=7)

# Query attention map
cell = attention.get_cell(row=5, col=7)
print(f"Attention: {cell.attention:.2f}")

# Find highest attention cells
top_cells = attention.get_top_k(k=5)

# Get unexplored regions
unexplored = attention.get_unexplored(threshold=0.1)

# Decay all attention (call each frame)
attention.decay()
```

### Inhibition of Return (IOR)

Recently viewed locations are temporarily inhibited:

```python
# After gazing at (5, 7), nearby cells are inhibited
attention.gaze_at(row=5, col=7)

# Cell (5, 8) now has inhibition
neighbor = attention.get_cell(row=5, col=8)
print(f"Inhibition: {neighbor.inhibition:.2f}")  # > 0

# Inhibition decays over time
attention.decay_inhibition()
```

---

## GazeHistory

Temporal tracking of gaze positions.

### Configuration

```python
from maxim.attention import GazeHistory, GazeHistoryConfig

config = GazeHistoryConfig(
    max_history=100,        # Maximum positions to track
    decay_window=30.0,      # Seconds before positions "fade"
)

history = GazeHistory(config)
```

### GazePosition

```python
@dataclass
class GazePosition:
    yaw: float              # Degrees
    pitch: float            # Degrees
    timestamp: float        # When gaze occurred
    duration: float         # How long fixated
    target_type: str        # "object", "person", "random"
```

### Usage

```python
# Record gaze positions
history.record(
    yaw=45.0,
    pitch=10.0,
    duration=0.5,
    target_type="person",
)

# Query recent gaze
recent = history.get_recent(seconds=5.0)

# Check if position was recently viewed
is_recent = history.was_recently_viewed(yaw=45.0, pitch=10.0, threshold=5.0)

# Get gaze statistics
stats = history.get_stats()
print(f"Average fixation: {stats['avg_duration']:.2f}s")
print(f"Most common target: {stats['most_common_type']}")
```

---

## SalienceMap

Unified spatial salience combining all attention factors.

### Configuration

```python
from maxim.attention import SalienceMap, SalienceMapConfig

config = SalienceMapConfig(
    grid_rows=10,
    grid_cols=15,

    # Weight factors
    object_weight=0.4,      # Object presence
    motion_weight=0.3,      # Motion detection
    novelty_weight=0.2,     # Novelty score
    ior_weight=0.1,         # Inhibition of return (negative)
)

salience_map = SalienceMap(config)
```

### Salience Cells

```python
@dataclass
class SalienceCell:
    row: int
    col: int
    salience: float         # Combined salience score
    components: dict        # Individual contributions
    # {
    #   "object": 0.8,
    #   "motion": 0.2,
    #   "novelty": 0.5,
    #   "ior": -0.3
    # }
```

### Usage

```python
# Update with detections
salience_map.update_from_detections(
    detections=[
        {"bbox": (100, 50, 200, 150), "class": "person", "confidence": 0.9},
        {"bbox": (300, 200, 350, 280), "class": "cup", "confidence": 0.7},
    ],
    frame_size=(640, 480),
)

# Get peak salience location
peak = salience_map.get_peak()
print(f"Peak at ({peak.row}, {peak.col}): {peak.salience:.2f}")

# Get next gaze target
target = salience_map.suggest_gaze_target(
    current_position=(0, 0),
    avoid_recent=True,
)
```

---

## GazeController

Human-like saccade-fixate dynamics.

### Configuration

```python
from maxim.attention import GazeController, GazeControllerConfig

config = GazeControllerConfig(
    # Saccade parameters
    saccade_velocity=400.0,    # Degrees/second
    min_saccade_angle=5.0,     # Minimum saccade size
    max_saccade_angle=60.0,    # Maximum saccade size

    # Fixation parameters
    fixation_duration_mean=0.3,  # Average fixation time
    fixation_duration_std=0.1,   # Variation
    fixation_jitter=1.0,         # Microsaccade amplitude

    # State machine
    saccade_cooldown=0.05,     # Min time between saccades
)

controller = GazeController(config)
```

### Gaze States

```python
from maxim.attention import GazeState

class GazeState(Enum):
    FIXATING = "fixating"       # Holding position
    SACCADING = "saccading"     # Rapid eye movement
    PURSUING = "pursuing"       # Smooth pursuit of motion
    MICROSACCADING = "microsaccading"  # Small fixation movements
```

### Usage

```python
# Request gaze to target
command = controller.gaze_to(target_yaw=45.0, target_pitch=10.0)

# Update each frame
state, movement = controller.update(dt=0.033)  # 30 FPS

if state == GazeState.SACCADING:
    # Execute rapid movement
    motor.move_fast(movement.yaw, movement.pitch)
elif state == GazeState.FIXATING:
    # Apply microsaccades for natural look
    motor.move_slow(movement.yaw, movement.pitch)

# Check if gaze has arrived
if controller.has_arrived():
    print("Fixation complete")
```

---

## SceneContextDetector

Detects scene changes for exploration mode.

### Configuration

```python
from maxim.attention import SceneContextDetector, SceneContextConfig

config = SceneContextConfig(
    change_threshold=0.3,     # Fraction of changed pixels
    object_change_weight=0.6, # Weight for object changes
    layout_change_weight=0.4, # Weight for layout changes
    cooldown_seconds=5.0,     # Min time between scene changes
)

detector = SceneContextDetector(config)
```

### Scene Snapshots

```python
@dataclass
class SceneSnapshot:
    timestamp: float
    objects: list[dict]       # Detected objects
    layout_hash: str          # Spatial arrangement hash
    dominant_colors: list     # Color histogram
```

### Usage

```python
# Update with current frame
is_new_scene = detector.update(
    detections=yolo_detections,
    frame=camera_frame,
)

if is_new_scene:
    print("Scene changed! Triggering exploration...")
    # Reset attention, explore new environment

# Get scene summary
snapshot = detector.get_current_snapshot()
print(f"Objects: {[obj['class'] for obj in snapshot.objects]}")
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
