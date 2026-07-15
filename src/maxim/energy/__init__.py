"""Energy tracking module - Monitors resource expenditure across subsystems.

Provides a unified framework for tracking energy consumption in different
domains: LLM tokens, compute time, and (optionally) actual power draw from
hardware telemetry.

This enables:
1. Energy-aware decision making (avoid expensive actions when resources are low)
2. Learning associations between actions and their energy costs
3. Resource budgeting

Note: the energy reaction bridge and movement energy tracker classes were
removed in the cradle sensorimotor update.  Interoceptive drive signals
(hunger, fatigue, stamina recovery) are now handled by the drive protocol in
``embodiment.sem.HomeostaticDriveSpec`` / ``EntropicDriveSpec``.

Example:
    from maxim.energy import EnergyRegistry, LLMEnergyTracker

    # Create registry and register trackers
    registry = EnergyRegistry()
    registry.register(LLMEnergyTracker())

    # Query energy state
    summary = registry.get_summary()
    print(f"Total LLM energy: {summary['llm']['total_energy']:.2f}")
"""

from maxim.energy.signal import EnergySignal, EnergyType
from maxim.energy.tracker import EnergyTracker, EnergyConfig
from maxim.energy.llm_tracker import LLMEnergyTracker, LLMEnergyConfig
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
    # Registry
    "EnergyRegistry",
    "get_global_registry",
]
