# Time System

The time module provides temporal rhythm indexing through the SCN (Suprachiasmatic Nucleus), enabling time-aware memory retrieval and pattern detection.

## Overview

The time system provides:

1. **Temporal Signatures**: Multi-scale time encoding
2. **Rhythm Indexing**: Fast queries by time of day, day of week, etc.
3. **Pattern Detection**: Identify recurring temporal patterns
4. **Memory Integration**: Time-aware memory retrieval

## Components

| Component | File | Purpose |
|-----------|------|---------|
| `SCN` | `scn.py` | Temporal rhythm indexing |
| `TemporalSignature` | `temporal_signature.py` | Multi-scale time encoding |
| `circular_distance` | `temporal_signature.py` | Cyclic time comparison |

---

## TemporalSignature

Encodes a timestamp at multiple time scales.

### Creating Signatures

```python
from maxim.time import TemporalSignature
import time

# From current time
sig = TemporalSignature.from_timestamp(time.time())

# From datetime
from datetime import datetime
sig = TemporalSignature.from_datetime(datetime.now())

# Access components
print(f"Hour: {sig.hour}")        # 0-23
print(f"Day of week: {sig.day}")  # 0=Monday, 6=Sunday
print(f"Week of month: {sig.week}")  # 0-3
print(f"Month: {sig.month}")      # 0-11
```

### Comparing Signatures

```python
from maxim.time import TemporalSignature, circular_distance

sig1 = TemporalSignature(hour=23, day=0, week=0, month=0)
sig2 = TemporalSignature(hour=1, day=0, week=0, month=0)

# Circular distance handles wrap-around
# 23:00 to 01:00 is 2 hours, not 22 hours
distance = circular_distance(sig1.hour, sig2.hour, period=24)
print(f"Hour distance: {distance}")  # 2

# Full signature similarity
similarity = sig1.similarity(sig2)
print(f"Similarity: {similarity:.2f}")
```

### Binning

Signatures convert to bin indices for indexing:

```python
hour_bin, day_bin, week_bin, month_bin = sig.to_bins()
# (9, 2, 1, 5)  # 9am, Wednesday, week 2, June
```

---

## SCN (Suprachiasmatic Nucleus)

The SCN maintains binned indices for fast temporal queries.

### Index Structure

| Index | Bins | Purpose |
|-------|------|---------|
| Circadian | 24 hourly | Time of day patterns |
| Weekly | 7 daily | Day of week patterns |
| Monthly | 4 weekly | Week of month patterns |
| Annual | 12 monthly | Seasonal patterns |

### Registration

```python
from maxim.time import SCN, TemporalSignature

scn = SCN()

# Register a memory with its temporal signature
memory_id = "mem_123"
sig = TemporalSignature.from_timestamp(memory.timestamp)
scn.register(memory_id, sig)
```

### Querying

```python
# Query by hour (circadian)
morning_memories = scn.query_hour(9)   # 9am memories
evening_memories = scn.query_hour(18)  # 6pm memories

# Query by day of week
monday_memories = scn.query_day(0)    # Monday
weekend_memories = scn.query_days([5, 6])  # Sat+Sun

# Query by week of month
first_week = scn.query_week(0)

# Query by month
summer_memories = scn.query_months([5, 6, 7])  # Jun-Aug

# Combined queries
tuesday_mornings = scn.query_hour(9) & scn.query_day(1)
```

### Pattern Detection

Find recurring temporal patterns:

```python
# Find rhythmic patterns
patterns = scn.find_rhythmic_patterns(min_occurrences=5)

for pattern in patterns:
    print(f"Pattern: {pattern.description}")
    print(f"  Frequency: {pattern.frequency}")
    print(f"  Memory IDs: {pattern.memory_ids[:5]}...")

# Example output:
# Pattern: Morning routine (8-9am weekdays)
#   Frequency: 23 occurrences
#   Memory IDs: ['mem_1', 'mem_5', 'mem_12', ...]
```

### Temporal Priors

Cold start handling with temporal priors:

```python
# Register priors for expected patterns
scn.add_prior("work_hours", hours=range(9, 18), days=range(0, 5))
scn.add_prior("sleep_hours", hours=list(range(22, 24)) + list(range(0, 7)))

# Query with prior boosting
memories = scn.query_with_priors("work_hours")
```

---

## Integration with Hippocampus

SCN integrates with Hippocampus for temporal memory queries:

```python
from maxim.memory import Hippocampus
from maxim.time import SCN

hippo = Hippocampus(config)
scn = SCN()

# When capturing memories
memory_id = hippo.capture(perception)
sig = TemporalSignature.from_timestamp(perception.timestamp)
scn.register(memory_id, sig)

# When querying
memory_ids = scn.query_hour(current_hour)
memories = [hippo.get(mid) for mid in memory_ids]
```

---

## Persistence

SCN state persists to JSON:

```python
# Save
scn.save("data/util/scn_state.json")

# Load
scn.load("data/util/scn_state.json")
```

### File Format

```json
{
  "version": 1,
  "saved_at": "2024-02-06T12:00:00Z",
  "circadian_bins": {
    "9": ["mem_1", "mem_5", "mem_12"],
    "10": ["mem_2", "mem_8"],
    ...
  },
  "weekly_bins": {...},
  "monthly_bins": {...},
  "annual_bins": {...},
  "signatures": {...},
  "priors": {...}
}
```

Clear with: `maxim --clear-memory scn`

---

## Memory Footprint

For 10,000 memories:
- 47 total bins (24 + 7 + 4 + 12)
- ~213 memories per bin average
- ~500KB total for indices (memory IDs are shared references)

---

## Use Cases

### Time-Appropriate Responses

```python
# Get current temporal context
current_sig = TemporalSignature.from_timestamp(time.time())

# Find similar times from history
similar_times = scn.query_hour(current_sig.hour)
recent_actions = [hippo.get(mid) for mid in similar_times]

# Use for context-aware behavior
if current_sig.hour in range(22, 24) or current_sig.hour in range(0, 7):
    # Night behavior: quieter, slower movements
    ...
```

### Pattern-Based Predictions

```python
# What typically happens at this time?
patterns = scn.find_patterns_at(current_sig)

for pattern in patterns:
    print(f"Expected: {pattern.typical_action}")
    print(f"Confidence: {pattern.confidence:.2f}")
```

### Temporal Anomaly Detection

```python
# Is current activity unusual for this time?
expected_memories = scn.query_temporal(current_sig)
expected_types = Counter(m.type for m in expected_memories)

if current_action_type not in expected_types:
    print(f"Unusual activity for {current_sig.hour}:00")
```

---

## Biological Inspiration

| Biological | Maxim Equivalent |
|------------|------------------|
| Suprachiasmatic Nucleus | SCN class |
| Circadian rhythms | Hourly bins |
| Weekly patterns | Daily bins |
| Seasonal adaptation | Monthly bins |
| Zeitgebers (time cues) | TemporalSignature |

The SCN enables Maxim to develop time-appropriate behavior patterns, similar to how biological circadian rhythms regulate activity throughout the day.
