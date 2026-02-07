# Salience System

The salience module manages WHAT objects are salient in the visual scene, providing object-level salience tracking distinct from spatial attention.

## Overview

The salience system provides:

1. **Object-level Salience**: Track individual objects by identity
2. **Novelty Detection**: Identify new or unusual objects
3. **Interest Matching**: Boost salience for goal-relevant objects
4. **Motion Detection**: Track moving objects for attention

## Components

| Component | File | Purpose |
|-----------|------|---------|
| `SalienceNetwork` | `salience_network.py` | Object salience tracking |
| `ThreadSafeNoveltyTracker` | `novelty.py` | Thread-safe novelty scoring |
| `MovementDetector` | `movement_detector.py` | Motion-based salience |

---

## SalienceNetwork

Object-level salience tracking with novelty and interest matching.

### Configuration

```python
from maxim.salience import SalienceNetwork, SalienceConfig

config = SalienceConfig(
    # Object tracking
    max_tracked_objects=50,
    object_timeout=5.0,       # Seconds before object "forgotten"
    iou_threshold=0.5,        # IoU for object matching

    # Salience weights
    confidence_weight=0.3,    # Detection confidence
    novelty_weight=0.25,      # Novelty score
    size_weight=0.15,         # Object size
    motion_weight=0.2,        # Movement amount
    interest_weight=0.1,      # Goal relevance

    # Novelty
    novelty_decay=0.95,       # Per-frame novelty decay
    habituation_rate=0.1,     # How fast objects become "normal"
)

network = SalienceNetwork(config)
```

### TrackedObject

Each detected object is tracked:

```python
@dataclass
class TrackedObject:
    object_id: str            # Unique identifier
    class_name: str           # "person", "cup", etc.
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float         # Detection confidence
    last_seen: float          # Timestamp

    # Salience components
    salience: float           # Combined salience score
    novelty: float            # How novel (0=habituated, 1=new)
    interest: float           # Goal relevance
    motion: float             # Movement amount
```

### Usage

```python
# Update with detections
tracked = network.update(
    detections=[
        {"bbox": (100, 50, 200, 150), "class": "person", "confidence": 0.9},
        {"bbox": (300, 200, 350, 280), "class": "cup", "confidence": 0.7},
    ],
    current_time=time.time(),
)

# Get most salient object
if tracked:
    top = max(tracked, key=lambda o: o.salience)
    print(f"Most salient: {top.class_name} ({top.salience:.2f})")

# Query by class
people = network.get_objects_by_class("person")

# Get object by ID
obj = network.get_object("obj_123")
```

### Interest Matching

Boost salience for goal-relevant objects:

```python
# Set current interests (from goal system)
network.set_interests({
    "person": 1.0,     # High interest in people
    "phone": 0.8,      # Looking for phone
    "cup": 0.2,        # Low interest
})

# Objects matching interests get boosted salience
# A phone will have higher salience than a cup with same novelty
```

### Novelty Decay

Objects become less novel over time (habituation):

```python
# First sighting: novelty = 1.0
obj = network.get_object("obj_123")
print(f"Novelty: {obj.novelty:.2f}")  # 1.0

# After repeated observations
# Novelty decays: 1.0 → 0.9 → 0.81 → ...
network.update(...)
obj = network.get_object("obj_123")
print(f"Novelty: {obj.novelty:.2f}")  # 0.7
```

---

## ThreadSafeNoveltyTracker

Thread-safe wrapper for novelty scoring across multiple threads.

### Usage

```python
from maxim.salience import ThreadSafeNoveltyTracker

tracker = ThreadSafeNoveltyTracker()

# Record observation (thread-safe)
novelty = tracker.observe("person")  # First time: high novelty

# Get current novelty for class
novelty = tracker.get_novelty("person")

# Reset novelty for a class (object left and returned)
tracker.reset("person")
```

### Implementation

```python
class ThreadSafeNoveltyTracker:
    """Thread-safe novelty tracking per object class."""

    def __init__(
        self,
        habituation_rate: float = 0.1,
        recovery_rate: float = 0.01,
    ):
        self._observations: dict[str, int] = {}
        self._lock = threading.RLock()

    def observe(self, object_class: str) -> float:
        """Record observation, return current novelty."""
        with self._lock:
            count = self._observations.get(object_class, 0) + 1
            self._observations[object_class] = count
            return self._compute_novelty(count)

    def _compute_novelty(self, observation_count: int) -> float:
        """Novelty decays with repeated observations."""
        return math.exp(-self._habituation_rate * observation_count)
```

---

## MovementDetector

Tracks object motion for salience boosting.

### Configuration

```python
from maxim.salience import MovementDetector, MovementConfig

config = MovementConfig(
    # Motion thresholds (pixels/frame)
    min_motion=2.0,           # Ignore tiny movements
    significant_motion=10.0,  # Moderate motion
    rapid_motion=30.0,        # Fast motion (high salience)

    # Temporal filtering
    smoothing_window=3,       # Frames to average
    motion_decay=0.9,         # Per-frame decay
)

detector = MovementDetector(config)
```

### Usage

```python
# Update with current detections
motion_scores = detector.update(
    detections=[
        {"object_id": "obj_1", "center": (150, 100)},
        {"object_id": "obj_2", "center": (320, 240)},
    ],
)

# Get motion for specific object
motion = detector.get_motion("obj_1")
print(f"Motion: {motion:.2f} pixels/frame")

# Get moving objects
moving = detector.get_moving_objects(threshold=5.0)
for obj_id, motion in moving:
    print(f"{obj_id}: {motion:.2f}")
```

---

## Integration with Attention

Salience (what) feeds into Attention (where):

```python
from maxim.salience import SalienceNetwork
from maxim.attention import SalienceMap

salience_network = SalienceNetwork(config)
salience_map = SalienceMap(map_config)

# Update salience network with detections
tracked = salience_network.update(detections)

# Feed into spatial salience map
for obj in tracked:
    row, col = map_bbox_to_grid(obj.bbox)
    salience_map.add_salience(
        row=row,
        col=col,
        amount=obj.salience,
        source="object",
    )
```

---

## Salience vs Attention

| Aspect | Salience | Attention |
|--------|----------|-----------|
| Scope | Objects (what) | Locations (where) |
| Tracking | By object identity | By grid cell |
| Module | `maxim.salience` | `maxim.attention` |
| Key class | `SalienceNetwork` | `AttentionNetwork` |

Both systems work together:
1. SalienceNetwork tracks object salience
2. SalienceMap combines with spatial attention
3. AttentionNetwork tracks gaze history
4. GazeController executes movements

---

## Integration Points

| System | Integration |
|--------|-------------|
| **DefaultNetwork** | Behavior salience thresholds |
| **ThalamicGate** | Escalation decisions |
| **Hippocampus** | Memory salience storage |
| **SalienceMemoryBridge** | Memory-informed salience |

---

## Data Flow

```
YOLO Detections
       ↓
SalienceNetwork
├── Confidence score
├── Novelty (ThreadSafeNoveltyTracker)
├── Motion (MovementDetector)
├── Interest matching
└── Combined salience
       ↓
TrackedObject list (sorted by salience)
       ↓
SalienceMap (spatial distribution)
       ↓
ThalamicGate (escalation decision)
```

---

## Biological Inspiration

| Biological | Maxim Equivalent |
|------------|------------------|
| IT cortex (object recognition) | SalienceNetwork |
| Habituation | Novelty decay |
| Priming | Interest matching |
| MT/V5 (motion) | MovementDetector |
| Ventral attention | Object-based salience |

The salience system enables Maxim to identify what's important in the scene, while the attention system manages where to look.
