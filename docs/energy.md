# Energy Tracking System

The energy module provides LLM resource expenditure monitoring.

## Overview

Maxim tracks energy consumption to enable:

1. **Energy-aware decisions**: Avoid expensive actions when resources are low
2. **Cost learning**: Associate actions with their energy costs
3. **Budget gating**: Resource budgeting over time (e.g., imagination energy gate)
4. **Optimization**: Identify high-cost operations for efficiency improvements

> **Note (post-cradle update):** Interoceptive drive signals (hunger, fatigue,
> stamina recovery) are now handled by the **drive protocol** in
> `embodiment/sem.py` (`HomeostaticDriveSpec` / `EntropicDriveSpec`), not the
> energy module. `EnergyReactionBridge` and `MovementEnergyTracker` were removed
> as dead code (zero callers) during the cradle sensorimotor update.

## Components

| Component | File | Purpose |
|-----------|------|---------|
| `EnergyRegistry` | `registry.py` | Central coordinator for all trackers |
| `EnergyTracker` | `tracker.py` | Base tracker interface |
| `LLMEnergyTracker` | `llm_tracker.py` | Token and compute costs |
| `EnergySignal` | `signal.py` | Energy event representation |

---

## Energy Types

```python
from maxim.energy import EnergyType

class EnergyType(Enum):
    LLM_TOKENS = "llm_tokens"           # Token-based energy (input + output)
    LLM_LATENCY = "llm_latency"         # Time waiting for LLM response
    LLM_COST = "llm_cost"               # USD-normalized cost signal
    COMPUTE_TIME = "compute_time"       # General CPU/GPU compute time
    MOTOR_COMMAND = "motor_command"     # Energy to execute movement
    MOTOR_CURRENT = "motor_current"    # Actual motor current draw
    VISION_INFERENCE = "vision_inference" # Vision model inference
    AUDIO_PROCESSING = "audio_processing" # Audio transcription/TTS
    ATTENTION = "attention"             # Cognitive attention/focus cost
    MEMORY_ACCESS = "memory_access"    # Memory retrieval cost
```

---

## EnergyRegistry

Central coordinator that aggregates all energy trackers.

### Setup

```python
from maxim.energy import (
    EnergyRegistry,
    LLMEnergyTracker,
    MovementEnergyTracker,
    get_global_registry,
)

# Create and configure registry
registry = EnergyRegistry()
registry.register(LLMEnergyTracker())
registry.register(MovementEnergyTracker())

# Or use global singleton
registry = get_global_registry()
```

### Recording Energy

```python
# Record LLM usage via the LLMEnergyTracker
llm_tracker = registry.get_tracker("llm")
signal = llm_tracker.record(
    input_tokens=500,
    output_tokens=150,
    model="claude-3-haiku",
    latency_ms=1200,
)

# Record movement via the MovementEnergyTracker
move_tracker = registry.get_tracker("movement")
signal = move_tracker.record(
    delta_yaw=45.0,
    delta_pitch=10.0,
    duration_s=0.3,
    movement_type="head",
)

# Record a generic energy signal directly
from maxim.energy import EnergySignal, EnergyType

signal = EnergySignal(
    energy_type=EnergyType.VISION_INFERENCE,
    amount=2.5,
    source="vision",
    context={"model": "rtmdet"},
)
registry.record_signal(signal)
```

### Querying Energy

```python
# Get summary of all energy usage (per-tracker stats + budgets)
summary = registry.get_summary(window_seconds=60.0)
print(f"LLM energy: {summary['trackers']['llm']['total_energy']:.2f}")
print(f"Movement energy: {summary['trackers']['movement']['total_energy']:.2f}")

# Check budget state
if registry.is_low_energy("llm"):
    print("Low LLM budget - consider using smaller model")

# Get tracker-specific stats
llm_tracker = registry.get_tracker("llm")
llm_stats = llm_tracker.get_llm_stats()
print(f"Total tokens: {llm_stats['total_tokens']}")
```

---

## LLMEnergyTracker

Tracks language model inference costs.

### Configuration

```python
from maxim.energy import LLMEnergyTracker, LLMEnergyConfig

config = LLMEnergyConfig(
    # Energy per token (relative energy units)
    input_token_cost=0.001,       # 1000 input tokens = 1 energy
    output_token_cost=0.003,      # 333 output tokens = 1 energy

    # Latency cost (opportunity cost of waiting)
    latency_cost_per_second=0.1,  # 10 seconds = 1 energy

    # Model-specific energy multipliers (larger = more energy)
    model_multipliers={
        "claude-3-haiku": 0.5,
        "claude-3-sonnet": 1.0,
        "claude-3-opus": 2.0,
        "gpt-4o-mini": 0.4,
        "local": 0.2,
    },

    # Default multiplier for unknown models
    default_multiplier=1.0,
)

tracker = LLMEnergyTracker(config)
```

### Usage

```python
# Record an LLM call
signal = tracker.record(
    input_tokens=500,
    output_tokens=150,
    model="claude-3-haiku",
    latency_ms=1200,
    context={"prompt_type": "planning"},
)

print(f"Inference cost: {signal.amount:.2f}")

# Get LLM-specific stats
stats = tracker.get_llm_stats()
print(f"Total tokens: {stats['total_tokens']}")
print(f"Call count: {stats['call_count']}")
print(f"Avg latency: {stats['avg_latency_ms']:.1f}ms")

# Check token budget
budget = tracker.get_token_budget_status(budget_tokens=100000)
print(f"Token budget used: {budget['percentage']:.1f}%")
```

---

## MovementEnergyTracker

Tracks motor activity energy consumption.

### Configuration

```python
from maxim.energy import MovementEnergyTracker, MovementEnergyConfig

config = MovementEnergyConfig(
    # Distance costs
    angular_energy_per_degree=0.02,      # 50 degrees = 1 energy
    translation_energy_per_mm=0.05,      # 20 mm = 1 energy

    # Speed affects energy (faster = more expensive)
    speed_multiplier_base=1.0,           # Baseline for normal speed
    fast_speed_threshold=100.0,          # deg/sec threshold for "fast"
    fast_speed_multiplier=1.5,           # Multiplier when above threshold

    # Duration cost (motor hold time)
    duration_cost_per_second=0.1,        # Holding position has cost

    # Component multipliers
    antenna_energy_multiplier=0.3,       # Antennas are lighter
    body_rotation_multiplier=2.0,        # Body rotation is heavier
)

tracker = MovementEnergyTracker(config)
```

### Usage

```python
# Record a head movement
signal = tracker.record_head_movement(
    delta_yaw=45.0,
    delta_pitch=10.0,
    duration_s=0.5,
)

# Record from an action signature
signal = tracker.record_from_signature(
    "look_at:dy=90:dp=30",
    duration_s=0.3,
)

# Record body rotation (turn_around)
signal = tracker.record_body_rotation(angle=90.0, duration_s=5.0)

# Get movement stats
stats = tracker.get_movement_stats()
print(f"Total angular distance: {stats['total_angular_distance']}°")
print(f"Movement count: {stats['movement_count']}")
```

---

## EnergySignal

Represents a single energy expenditure event.

```python
from maxim.energy import EnergySignal, EnergyType
import time

signal = EnergySignal(
    energy_type=EnergyType.LLM_TOKENS,
    amount=3.5,                # Normalized energy units
    timestamp=time.time(),
    source="llm",              # What produced this
    duration_ms=1200.0,        # How long the activity took
    context={                  # Optional metadata
        "model": "claude-3-haiku",
        "input_tokens": 500,
        "output_tokens": 150,
    },
)
```

---

## Custom Trackers

Create custom energy trackers for new resource types:

```python
from dataclasses import dataclass
from maxim.energy import EnergyTracker, EnergyConfig, EnergySignal, EnergyType
from typing import Any
import time

@dataclass
class AudioEnergyConfig(EnergyConfig):
    cost_per_second: float = 10.0
    tts_multiplier: float = 2.0
    stt_multiplier: float = 1.5

class AudioEnergyTracker(EnergyTracker):
    name = "audio"
    energy_types = {EnergyType.AUDIO_PROCESSING}

    def __init__(self, config: AudioEnergyConfig | None = None):
        super().__init__(config or AudioEnergyConfig())
        self._audio_config = config or AudioEnergyConfig()

    def record(self, duration_seconds: float = 0.0,
               mode: str = "tts", **kwargs: Any) -> EnergySignal:
        energy = duration_seconds * self._audio_config.cost_per_second
        if mode == "tts":
            energy *= self._audio_config.tts_multiplier
        else:
            energy *= self._audio_config.stt_multiplier

        signal = EnergySignal(
            energy_type=EnergyType.AUDIO_PROCESSING,
            amount=energy,
            timestamp=time.time(),
            source=self.name,
            duration_ms=duration_seconds * 1000,
            context={"mode": mode},
        )
        self._record_signal(signal)
        return signal
```

---

## Integration with NAc

Energy costs can be integrated with causal learning:

```python
from maxim.decisions import NAc, Valence
from maxim.energy import get_global_registry

# After high-cost operation
registry = get_global_registry()
llm_tracker = registry.get_tracker("llm")
signal = llm_tracker.record(input_tokens=500, output_tokens=150, model="claude-3-haiku")

if signal.amount > threshold:
    # Record as negative outcome for learning
    nac.observe(
        event_type="llm",
        event_signature="large_generation",
        outcome_valence=Valence.NEGATIVE,
        context={"energy": energy},
    )
```

---

## Energy Budgeting

The `EnergyBudget` dataclass (in `signal.py`) tracks available energy per domain with passive regeneration:

```python
from maxim.energy.signal import EnergyBudget, EnergyType

budget = EnergyBudget(
    domain=EnergyType.LLM_TOKENS,
    total_capacity=1000.0,
    current_level=1000.0,
    recharge_rate=10.0,      # Energy recovered per second
)

# Attempt to spend energy (returns False if insufficient)
if budget.spend(estimated_cost):
    result = llm.generate(prompt)
else:
    result = fallback_response()

# Check budget state
print(f"Energy: {budget.percentage:.1f}%")
print(f"Low: {budget.is_low}")        # True if < 25%
print(f"Critical: {budget.is_critical}")  # True if < 10%
```

The `EnergyRegistry` manages per-domain budgets automatically via `EnergyRegistryConfig.budget_configs`.

---

## Monitoring

Query energy state for monitoring:

```python
# Get all energy stats
summary = registry.get_summary(window_seconds=60.0)

print(f"Uptime: {summary['uptime_seconds']}s")
print(f"Total signals: {summary['total_signals']}")
print(f"Total window energy: {summary['total_window_energy']:.2f}")

for tracker_name, stats in summary["trackers"].items():
    print(f"{tracker_name}:")
    print(f"  Total: {stats['total_energy']:.2f}")
    print(f"  Rate: {stats['rate_per_second']:.4f}/s")

for domain, budget in summary["budgets"].items():
    print(f"{domain} budget: {budget['percentage']:.1f}% remaining")
```

---

## Biological Inspiration

| Biological | Maxim Equivalent |
|------------|------------------|
| ATP/metabolic energy | Energy units |
| Fatigue | Budget depletion |
| Energy conservation | Cost-aware decisions |
| Hunger/motivation | Energy-based action selection |

The energy system enables Maxim to make resource-aware decisions, similar to how biological systems balance energy expenditure with goal achievement.
