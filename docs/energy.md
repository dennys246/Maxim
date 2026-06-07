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
    LLM_TOKENS = "llm_tokens"             # Token-based energy (input + output)
    LLM_LATENCY = "llm_latency"           # Time waiting for LLM response
    LLM_COST = "llm_cost"                 # USD-normalized cost signal
    MOTOR_COMMAND = "motor_command"       # Energy to execute movement
    VISION_INFERENCE = "vision_inference" # Vision model inference
```

---

## EnergyRegistry

Central coordinator that aggregates all energy trackers.

### Setup

```python
from maxim.energy import (
    EnergyRegistry,
    LLMEnergyTracker,
    get_global_registry,
)

# Create and configure registry
registry = EnergyRegistry()
registry.register(LLMEnergyTracker())

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
class VisionEnergyConfig(EnergyConfig):
    cost_per_frame: float = 1.0
    hires_multiplier: float = 2.5

class CustomVisionTracker(EnergyTracker):
    name = "custom_vision"
    energy_types = {EnergyType.VISION_INFERENCE}

    def __init__(self, config: VisionEnergyConfig | None = None):
        super().__init__(config or VisionEnergyConfig())
        self._vcfg = config or VisionEnergyConfig()

    def record(self, frame_count: int = 1,
               hires: bool = False, **kwargs: Any) -> EnergySignal:
        energy = frame_count * self._vcfg.cost_per_frame
        if hires:
            energy *= self._vcfg.hires_multiplier

        signal = EnergySignal(
            energy_type=EnergyType.VISION_INFERENCE,
            amount=energy,
            timestamp=time.time(),
            source=self.name,
            context={"frame_count": frame_count, "hires": hires},
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
