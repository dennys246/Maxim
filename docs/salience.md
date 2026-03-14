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
    max_tracked_objects=100,          # Max objects before LRU eviction
    min_confidence=0.3,              # Minimum detection confidence to track

    # Salience decay (time-based, not weight-based)
    novelty_decay_seconds=30.0,      # Time for novelty to decay to ~37%
    recency_decay_seconds=5.0,       # Time for recency salience to decay

    # Interest matching
    interest_labels={"person", "face"},  # Labels that get boosted salience
    interest_boost=1.5,              # Multiplier for interest-matched objects
    min_salience_threshold=0.2,      # Minimum salience to consider important
)

network = SalienceNetwork(config)
```

### TrackedObject

Each detected object is tracked:

```python
@dataclass
class TrackedObject:
    track_id: Any             # Unique tracking identifier
    class_id: int             # Object class ID
    label: str                # "person", "cup", etc.
    confidence: float         # Detection confidence
    first_seen: float         # Timestamp of first detection
    last_seen: float          # Timestamp of last detection
    times_seen: int           # Number of observations
    position_u: float         # Bbox center x
    position_v: float         # Bbox center y
    bbox_w: float             # Bounding box width
    bbox_h: float             # Bounding box height
    interest_matched: bool    # Whether label matches interest_labels
```

### Usage

```python
# Update with detections (bbox format: x, y, w, h)
network.update_from_detections(
    detections=[
        {"track_id": 1, "class_id": 0, "label": "person",
         "bbox": (100, 50, 100, 100), "confidence": 0.9},
        {"track_id": 2, "class_id": 41, "label": "cup",
         "bbox": (300, 200, 50, 80), "confidence": 0.7},
    ],
    timestamp=time.time(),
)

# Get most salient objects
top = network.get_top_salient(n=5)
if top:
    best = top[0]
    print(f"Most salient: {best['label']} ({best['salience']:.2f})")

# Query with filters
people = network.query(labels={"person"})

# Get salience for a specific track ID
salience = network.get_salience(track_id=1)

# Get context string for LLM consumption
context = network.to_context_str()
```

### Interest Matching

Interest matching is label-based, configured via `SalienceConfig.interest_labels`:

```python
# Configure interest labels at construction time
config = SalienceConfig(
    interest_labels={"person", "face", "phone"},  # Labels to boost
    interest_boost=1.5,                           # Salience multiplier
)
network = SalienceNetwork(config)

# Objects with matching labels automatically get boosted salience
# A "person" detection will have salience multiplied by interest_boost
# Query only interest-matched objects:
interest_objs = network.query(interest_only=True)
```

### Novelty Decay

Novelty decays exponentially based on time since first seen (not per-frame):

```python
# Novelty for unknown objects = 1.0
print(network.get_novelty(999))  # 1.0 (never seen)

# After detection, novelty decays over novelty_decay_seconds
# Formula: exp(-time_known / novelty_decay_seconds)
# With default 30s decay: after 30s novelty ~= 0.37, after 60s ~= 0.14
novelty = network.get_novelty(track_id=1)
```

---

## ThreadSafeNoveltyTracker

Thread-safe wrapper around the underlying `NoveltyTracker` (from `maxim.inference.segment_vision`).
This is **not** a standalone tracker -- it delegates all novelty logic to a `NoveltyTracker`
instance and adds `threading.Lock`-based synchronization for safe concurrent access from
the perception thread, Default Network thread, etc.

### Usage

```python
from maxim.inference.segment_vision import NoveltyTracker
from maxim.salience import ThreadSafeNoveltyTracker

# Wrap an existing tracker, or let it create one internally
tracker = ThreadSafeNoveltyTracker(NoveltyTracker())

# Get novelty for a track_id (thread-safe)
novelty = tracker.get_novelty(track_id=42)

# Mark a track_id as seen
tracker.update(track_id=42)

# Class-aware update (drives class-level habituation)
tracker.update_with_class(track_id=42, class_id=0)

# Class-level novelty (rare categories score higher)
class_nov = tracker.get_class_novelty(class_id=0)

# Batch operations (single lock acquisition)
tracker.update_batch([1, 2, 3])
scores = tracker.get_novelty_batch([1, 2, 3])

# Focus management (resets novelty to max, starts slow decay)
tracker.focus(track_id=42)

# Atomic get-and-focus
novelty = tracker.focus_and_get_novelty(track_id=42)

# Sensitization modulation callback
tracker.set_modulation_lookup(lambda class_id: 1.2)
```

### Key Properties

```python
tracker.focus_decay_seconds   # Time for novelty to decay while focused (default 10s)
tracker.recovery_seconds      # Time for novelty to recover when not focused (default 20s)
tracker.max_novelty           # Score for never-seen objects (default 2.0)
tracker.min_novelty           # Floor for frequently-focused objects (default 0.5)
tracker.sensitization_ceiling # Max multiplier for sensitized classes (default 1.5)
tracker.tracked_count         # Number of track IDs currently tracked
```

---

## MovementDetector

Tracks object motion for salience boosting.

### Configuration

```python
from maxim.salience import MovementDetector, MovementConfig

config = MovementConfig(
    velocity_normalization=50.0,   # Pixels/frame for max score (1.0)
    decay_seconds=0.5,             # How fast score decays when stopped
    min_movement_threshold=3.0,    # Ignore movements below this (pixels)
    max_entries=200,               # Max tracked objects before cleanup
    peripheral_boost=0.3,          # Extra salience for peripheral motion
    peripheral_threshold=0.6,      # Fraction from center = peripheral
)

detector = MovementDetector(config)
```

### Usage

```python
# Update with detections (needs track_id and bbox_xyxy or bbox)
scores = detector.update(
    detections=[
        {"track_id": 1, "bbox_xyxy": (100, 50, 200, 150)},
        {"track_id": 2, "bbox_xyxy": (300, 200, 350, 280)},
    ],
    frame_center=(320, 240),  # Optional, defaults to frame center
)
# scores: {1: 0.3, 2: 0.3}  (new objects start at 0.3)

# Get movement score for a specific track ID
score = detector.get_movement_score(track_id=1)
print(f"Movement: {score:.2f}")

# Get top movers (track_id, score) sorted by score
top = detector.get_top_movers(n=5)
for track_id, score in top:
    print(f"{track_id}: {score:.2f}")

# Set frame dimensions for peripheral calculations
detector.set_frame_size(1280, 720)
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
salience_network.update_from_detections(detections)

# Feed top salient objects into spatial salience map
for obj in salience_network.get_top_salient(n=10):
    row, col = map_position_to_grid(obj["position"])
    salience_map.add_salience(
        row=row,
        col=col,
        amount=obj["salience"],
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
