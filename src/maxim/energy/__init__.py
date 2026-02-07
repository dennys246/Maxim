"""Energy tracking module - Monitors resource expenditure across subsystems.

Provides a unified framework for tracking energy consumption in different
domains: LLM tokens, compute time, motor activity, and (optionally) actual
power draw from hardware telemetry.

This enables:
1. Energy-aware decision making (avoid expensive actions when resources are low)
2. Learning associations between actions and their energy costs
3. Fatigue simulation and resource budgeting

Example:
    from maxim.energy import EnergyRegistry, LLMEnergyTracker

    # Create registry and register trackers
    registry = EnergyRegistry()
    registry.register(LLMEnergyTracker())

    # Record energy expenditure
    registry.record_llm_usage(
        input_tokens=500,
        output_tokens=150,
        model="claude-3-haiku",
        latency_ms=1200,
    )

    # Query energy state
    summary = registry.get_summary()
    print(f"Total LLM energy: {summary['llm']['total_energy']:.2f}")
"""

from maxim.energy.signal import EnergySignal, EnergyType
from maxim.energy.tracker import EnergyTracker, EnergyConfig
from maxim.energy.llm_tracker import LLMEnergyTracker, LLMEnergyConfig
from maxim.energy.movement_tracker import MovementEnergyTracker, MovementEnergyConfig
from maxim.energy.registry import EnergyRegistry, get_global_registry

__all__ = [
    # Core types
    "EnergyType",
    "EnergySignal",
    # Base tracker
    "EnergyTracker",
    "EnergyConfig",
    # LLM tracker
    "LLMEnergyTracker",
    "LLMEnergyConfig",
    # Movement tracker
    "MovementEnergyTracker",
    "MovementEnergyConfig",
    # Registry
    "EnergyRegistry",
    "get_global_registry",
]
