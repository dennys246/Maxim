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
| `BoundedBin` | `scn.py` | Capacity-managed time bin with significance-based eviction |
| `BinEntry` | `scn.py` | Single entry in a bounded bin (memory_id, significance, timestamp) |
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

# From current time (convenience)
sig = TemporalSignature.now()

# Access components (all normalized to 0.0-1.0 phase values)
print(f"Circadian phase: {sig.circadian_phase}")  # 0.0-1.0 (midnight=0, noon=0.5)
print(f"Weekly phase: {sig.weekly_phase}")         # 0.0-1.0 (Monday 00:00=0)
print(f"Monthly phase: {sig.monthly_phase}")       # 0.0-1.0 (1st=0, ~15th=0.5)
print(f"Annual phase: {sig.annual_phase}")         # 0.0-1.0 (Jan 1=0, July 1≈0.5)
print(f"Timestamp: {sig.timestamp}")               # Unix timestamp (absolute reference)
```

### Comparing Signatures

```python
from maxim.time import TemporalSignature, circular_distance

sig1 = TemporalSignature.from_timestamp(ts1)
sig2 = TemporalSignature.from_timestamp(ts2)

# Circular distance handles wrap-around on 0.0-1.0 phase values
# e.g., phase 0.95 is close to phase 0.05 (distance = 0.1, not 0.9)
distance = circular_distance(sig1.circadian_phase, sig2.circadian_phase)
print(f"Circadian distance: {distance}")  # 0.0-0.5

# Full signature similarity (weighted across all four phases)
similarity = sig1.similarity(sig2)
print(f"Similarity: {similarity:.2f}")

# Custom weights: (circadian, weekly, monthly, annual)
similarity = sig1.similarity(sig2, weights=(2.0, 1.0, 0.5, 0.5))
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

### BoundedBin (Capacity Management)

Each SCN time bin is a `BoundedBin` with a configurable max capacity (default 200). When a bin is full, the least significant entry from the older half is evicted to make room for more significant new entries.

```python
from maxim.time.scn import BoundedBin

bb = BoundedBin(max_size=200)
bb.add("mem_1", significance=0.9)  # High significance
bb.add("mem_2", significance=0.1)  # Low significance — evicted first when full

# Set-compatible interface
"mem_1" in bb       # True
len(bb)             # 2
set(bb)             # {"mem_1", "mem_2"}
bb & other_set      # Intersection
```

BoundedBin is backward-compatible with v1/v2 persistence via `from_list()` which accepts both `list[dict]` (v3) and `list[str]` (v2) formats.

### Registration

```python
from maxim.time import SCN, TemporalSignature

scn = SCN()

# Register a memory with its temporal signature and significance
memory_id = "mem_123"
sig = TemporalSignature.from_timestamp(memory.timestamp)
scn.register(memory_id, sig, significance=0.8)
```

### Querying

```python
# Query by hour (circadian)
morning_memories = scn.query_hour(9)   # 9am memories
evening_memories = scn.query_hour(18)  # 6pm memories

# Query by day of week
monday_memories = scn.query_day(0)    # Monday
saturday_memories = scn.query_day(5)  # Saturday

# Query by week of month
first_week = scn.query_week_of_month(0)

# Query by month
june_memories = scn.query_month(5)  # June (0=Jan, 11=Dec)

# Combined queries (set intersection via BoundedBin & operator)
tuesday_mornings = scn.query_hour(9) & scn.query_day(1)

# Or use query_intersection for multi-criteria matching
tuesday_mornings = scn.query_intersection(hour=9, day=1)
```

### Pattern Detection

Find recurring temporal patterns:

```python
# Find rhythmic patterns
# Returns dict mapping rhythm type to list of (bin_id, count) tuples
patterns = scn.find_rhythmic_patterns(min_occurrences=5)

for rhythm_type, bins in patterns.items():
    for bin_id, count in bins:
        print(f"  {rhythm_type} bin {bin_id}: {count} occurrences")

# Example output:
#   circadian bin 9: 23 occurrences    (9am)
#   circadian bin 17: 18 occurrences   (5pm)
#   weekly bin 0: 12 occurrences       (Monday)
```

### Temporal Priors

Cold start handling with temporal priors:

```python
# Register priors for expected patterns (one hour bin at a time)
scn.add_temporal_prior("morning_greeting", hour_bin=8)
scn.add_temporal_prior("morning_greeting", hour_bin=9)
scn.add_temporal_prior("evening_wind_down", hour_bin=21)
```

---

## Coupled Oscillator Network

The SCN includes an optional Kuramoto-inspired coupled oscillator network for temporal rhythm learning and anticipatory prediction.  Enabled by default in production via `build_bio_stack`.

### Architecture

Four oscillators represent temporal scales: circadian (daily), weekly, monthly, annual.  The coupling matrix learns which scales co-activate via Hebbian plasticity:

```
dθ_i/dt = ω_i + (K/N) Σ_j W[i][j] * sin(θ_j - θ_i)    (Kuramoto dynamics)
ΔW[i][j] = η * cos(θ_i - θ_j)                             (Hebbian learning)
```

### Event-Type Phase Tracking (B2: Anticipatory Credit)

The oscillator tracks per-event-type circadian phases.  When the same event type fires repeatedly at a consistent time, the oscillator learns the association and can predict imminence:

```python
scn = SCN()
scn.enable_oscillator()

# Events are recorded automatically via TemporalCreditDistributor.record_event()
# Manual observation:
sig = TemporalSignature.now()
scn.observe_event("tool:sword_slash", sig)

# Query which events are predicted to be imminent
imminent = scn.get_anticipatory_signatures(min_imminence=0.5)
# {"tool:sword_slash": 0.83}  — the slash event is expected soon
```

### Anticipatory Pre-Activation

The `TemporalCreditDistributor` uses oscillator predictions to prime the system:

```python
# Called once per tick (before distribute):
dist.anticipatory_pre_activate(agent_id)

# When reward arrives, pre-activated traces are credited normally:
dist.distribute(agent_id, reward=1.0)
```

Anticipation primes NAc eligibility traces for events predicted to be imminent.  When the predicted event actually fires and a reward arrives, the pre-activated trace is credited through the normal fast-decay path.  This closes the SCN→NAc feedback loop.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coupling_strength` | 0.1 | Global K in Kuramoto model |
| `learning_rate` | 0.01 | Hebbian learning rate η |
| `weight_decay` | 0.999 | Per-observation weight decay |
| `max_event_phases` | 50 | Ring buffer cap per event signature |
| `anticipatory_weight` | 0.2 | Credit weight for anticipatory pre-activation |

### Analysis

```python
# Kuramoto order parameter (synchronization measure)
coherence = scn.phase_coherence()  # 0.0-1.0, None if disabled

# Coupling strength between oscillators
coupling = scn.coupling_strength(0, 1)  # circadian-weekly coupling

# Temporal anomaly score
anomaly = scn.temporal_anomaly_score(sig)  # 0.0-1.0

# Predict next occurrence of a circadian phase
hours = scn.predict_next_occurrence(target_hour=14.0)  # hours until 2pm
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
scn.save("~/.maxim/util/scn_state.json")

# Load
scn.load("~/.maxim/util/scn_state.json")
```

### File Format (v3.0)

```json
{
  "version": "3.0",
  "circadian_bins": {
    "9": [
      {"memory_id": "mem_1", "significance": 0.8, "registered_at": 1704103200.0},
      {"memory_id": "mem_5", "significance": 0.6, "registered_at": 1704106800.0}
    ],
    "10": [...]
  },
  "weekly_bins": {...},
  "monthly_bins": {...},
  "annual_bins": {...},
  "signatures": {...},
  "priors": {...}
}
```

v3.0 stores `BoundedBin` entries with significance scores. Loading supports v1.0, v2.0 (plain string IDs), and v3.0 formats.

Clear with: `maxim --clear-memory scn`

---

## Integration with Consolidation

SCN plays a central role in the consolidation pipeline:

1. **Acute staging**: After each goal, `ExecAgent._evaluate_staging()` registers the memory's temporal signature in SCN with its significance score
2. **Wave scoring**: `ConsolidationOrchestrator._compute_wave_score()` queries `scn.query_similar_time()` for temporal recurrence — memories at the same time of day score higher
3. **Promotion**: When a staged memory is promoted, it is registered in SCN with its final wave score as significance
4. **Chronic staging**: `find_chronic_candidates()` queries SCN bounded bins (last 24h of circadian bins, same weekday, similar time tolerance) to find recurring patterns
5. **Eviction**: BoundedBin prevents unbounded growth — low-significance memories are evicted when bins reach capacity

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
current_sig = TemporalSignature.now()

# Find similar times from history (using bin index from phase)
hour_bin, day_bin, week_bin, month_bin = current_sig.to_bins()
similar_times = scn.query_hour(hour_bin)
recent_actions = [hippo.get(mid) for mid in similar_times]

# Use for context-aware behavior
if current_sig.circadian_phase > 0.917 or current_sig.circadian_phase < 0.292:
    # Night behavior (roughly 10pm-7am): quieter, slower movements
    ...
```

### Pattern-Based Predictions

```python
# What typically happens at this time?
# Use query_similar_time to find memories near a given temporal signature
similar_memories = scn.query_similar_time(current_sig, tolerance=1)

for mid in similar_memories:
    memory = hippo.get(mid)
    print(f"Similar time activity: {memory}")
```

### Temporal Anomaly Detection

```python
# Is current activity unusual for this time?
# Use the oscillator-based anomaly score (if oscillator is enabled)
anomaly = scn.temporal_anomaly_score(current_sig)
if anomaly is not None and anomaly > 0.7:
    hour_bin, _, _, _ = current_sig.to_bins()
    print(f"Unusual activity for hour bin {hour_bin}")

# Or check bin populations manually
expected_memories = scn.query_similar_time(current_sig, tolerance=1)
expected_types = Counter(hippo.get(mid).type for mid in expected_memories)

if current_action_type not in expected_types:
    hour_bin, _, _, _ = current_sig.to_bins()
    print(f"Unusual activity for hour {hour_bin}:00")
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
