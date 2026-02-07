# Energy Tracking System

The energy module provides unified resource expenditure monitoring across all Maxim subsystems.

## Overview

Maxim tracks energy consumption to enable:

1. **Energy-aware decisions**: Avoid expensive actions when resources are low
2. **Cost learning**: Associate actions with their energy costs
3. **Fatigue simulation**: Resource budgeting over time
4. **Optimization**: Identify high-cost operations for efficiency improvements

## Components

| Component | File | Purpose |
|-----------|------|---------|
| `EnergyRegistry` | `registry.py` | Central coordinator for all trackers |
| `EnergyTracker` | `tracker.py` | Base tracker interface |
| `LLMEnergyTracker` | `llm_tracker.py` | Token and compute costs |
| `MovementEnergyTracker` | `movement_tracker.py` | Motor activity costs |
| `EnergySignal` | `signal.py` | Energy event representation |

---

## Energy Types

```python
from maxim.energy import EnergyType

class EnergyType(Enum):
    LLM = "llm"           # Token generation, inference
    COMPUTE = "compute"   # CPU/GPU cycles
    MOVEMENT = "movement" # Motor activity
    AUDIO = "audio"       # Speech synthesis, transcription
    NETWORK = "network"   # HTTP requests, data transfer
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
# Record LLM usage
registry.record_llm_usage(
    input_tokens=500,
    output_tokens=150,
    model="mistral-7b",
    latency_ms=1200,
)

# Record movement
registry.record_movement(
    joint="yaw",
    delta_degrees=45.0,
    duration_ms=300,
    velocity=150.0,
)

# Record generic energy signal
from maxim.energy import EnergySignal, EnergyType

signal = EnergySignal(
    energy_type=EnergyType.NETWORK,
    amount=1024,  # bytes
    source="http_fetch",
    context={"url": "https://example.com"},
)
registry.record(signal)
```

### Querying Energy

```python
# Get summary of all energy usage
summary = registry.get_summary()
print(f"LLM energy: {summary['llm']['total_energy']:.2f}")
print(f"Movement energy: {summary['movement']['total_energy']:.2f}")

# Get recent signals
signals = registry.get_recent_signals(
    energy_type=EnergyType.LLM,
    since_seconds=60,
)

# Get total by type
llm_total = registry.get_total(EnergyType.LLM)
movement_total = registry.get_total(EnergyType.MOVEMENT)
```

---

## LLMEnergyTracker

Tracks language model inference costs.

### Configuration

```python
from maxim.energy import LLMEnergyTracker, LLMEnergyConfig

config = LLMEnergyConfig(
    # Energy per token (model-specific)
    input_token_cost=1.0,
    output_token_cost=3.0,

    # Latency costs
    latency_cost_per_ms=0.01,

    # Model-specific multipliers
    model_costs={
        "smollm-1.7b": 0.5,
        "mistral-7b": 1.0,
        "llama3-8b": 1.2,
    },
)

tracker = LLMEnergyTracker(config)
```

### Usage

```python
# Record inference
energy = tracker.record_inference(
    input_tokens=500,
    output_tokens=150,
    model="mistral-7b",
    latency_ms=1200,
)

print(f"Inference cost: {energy:.2f}")

# Get cumulative stats
stats = tracker.get_stats()
print(f"Total tokens: {stats['total_tokens']}")
print(f"Total inferences: {stats['inference_count']}")
```

---

## MovementEnergyTracker

Tracks motor activity energy consumption.

### Configuration

```python
from maxim.energy import MovementEnergyTracker, MovementEnergyConfig

config = MovementEnergyConfig(
    # Base cost per degree of movement
    cost_per_degree=1.0,

    # Velocity multiplier (faster = more energy)
    velocity_multiplier=0.5,

    # Joint-specific costs
    joint_costs={
        "yaw": 1.0,
        "pitch": 1.2,
        "roll": 0.8,
    },
)

tracker = MovementEnergyTracker(config)
```

### Usage

```python
# Record movement
energy = tracker.record_movement(
    joint="yaw",
    delta_degrees=45.0,
    duration_ms=300,
    velocity=150.0,
)

# Get movement stats
stats = tracker.get_stats()
print(f"Total degrees moved: {stats['total_degrees']}")
print(f"Movement count: {stats['movement_count']}")
```

---

## EnergySignal

Represents a single energy expenditure event.

```python
from maxim.energy import EnergySignal, EnergyType
import time

signal = EnergySignal(
    energy_type=EnergyType.LLM,
    amount=750.0,              # Energy units
    timestamp=time.time(),
    source="llm_agent",        # What produced this
    context={                  # Optional metadata
        "model": "mistral-7b",
        "tokens": 500,
    },
)
```

---

## Custom Trackers

Create custom energy trackers for new resource types:

```python
from maxim.energy import EnergyTracker, EnergyConfig, EnergyType

@dataclass
class AudioEnergyConfig(EnergyConfig):
    cost_per_second: float = 10.0
    tts_multiplier: float = 2.0
    stt_multiplier: float = 1.5

class AudioEnergyTracker(EnergyTracker):
    energy_type = EnergyType.AUDIO

    def __init__(self, config: AudioEnergyConfig | None = None):
        self.config = config or AudioEnergyConfig()
        self._total = 0.0

    def record_synthesis(self, duration_seconds: float) -> float:
        energy = duration_seconds * self.config.cost_per_second
        energy *= self.config.tts_multiplier
        self._total += energy
        return energy

    def record_transcription(self, duration_seconds: float) -> float:
        energy = duration_seconds * self.config.cost_per_second
        energy *= self.config.stt_multiplier
        self._total += energy
        return energy

    def get_total(self) -> float:
        return self._total
```

---

## Integration with NAc

Energy costs can be integrated with causal learning:

```python
from maxim.decisions import NAc, Valence
from maxim.energy import get_global_registry

# After high-cost operation
registry = get_global_registry()
energy = registry.record_llm_usage(...)

if energy > threshold:
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

Implement energy budgets for constrained operation:

```python
class EnergyBudget:
    def __init__(self, max_energy: float):
        self.max_energy = max_energy
        self.current = 0.0

    def can_afford(self, estimated_cost: float) -> bool:
        return self.current + estimated_cost <= self.max_energy

    def record(self, actual_cost: float) -> None:
        self.current += actual_cost

    def reset(self) -> None:
        self.current = 0.0

# Usage
budget = EnergyBudget(max_energy=10000)

if budget.can_afford(estimated_llm_cost):
    result = llm.generate(prompt)
    budget.record(actual_cost)
else:
    # Use cheaper alternative
    result = fallback_response()
```

---

## Monitoring

Query energy state for monitoring:

```python
# Get all energy stats
summary = registry.get_summary()

for energy_type, stats in summary.items():
    print(f"{energy_type}:")
    print(f"  Total: {stats['total_energy']:.2f}")
    print(f"  Recent (1min): {stats['recent_energy']:.2f}")
    print(f"  Peak: {stats['peak_energy']:.2f}")
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
