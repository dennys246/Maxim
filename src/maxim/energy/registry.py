"""Energy registry - Central coordinator for energy tracking.

Aggregates energy signals from multiple domain-specific trackers
and provides unified statistics, budgets, and learning integration.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from maxim.energy.signal import EnergyBudget, EnergySignal, EnergyType
from maxim.energy.tracker import EnergyTracker

logger = logging.getLogger(__name__)

# Global registry instance
_global_registry: EnergyRegistry | None = None
_global_lock = threading.Lock()


@dataclass
class EnergyRegistryConfig:
    """Configuration for the energy registry.

    Attributes:
        enable_budgets: Whether to track energy budgets.
        enable_learning: Whether to integrate with NAc.
        high_energy_threshold: Threshold for "high energy" warnings.
        budget_configs: Per-domain budget configurations.
    """

    enable_budgets: bool = True
    enable_learning: bool = True
    high_energy_threshold: float = 5.0  # Warn when single signal > 5 units

    # Budget configurations per domain
    budget_configs: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "llm": {"capacity": 1000.0, "recharge_rate": 10.0},
            "movement": {"capacity": 500.0, "recharge_rate": 5.0},
            "vision": {"capacity": 200.0, "recharge_rate": 2.0},
        }
    )


class EnergyRegistry:
    """Central registry for energy tracking.

    Manages multiple energy trackers, aggregates statistics,
    and provides integration points for NAc learning.

    Example:
        registry = EnergyRegistry()

        # Register trackers
        registry.register(LLMEnergyTracker())

        # Get unified summary
        summary = registry.get_summary()

        # Check if action should be gated due to low energy
        if registry.is_low_energy("llm"):
            print("Low LLM budget - consider using smaller model")
    """

    def __init__(self, config: EnergyRegistryConfig | None = None) -> None:
        """Initialize the energy registry.

        Args:
            config: Registry configuration. Uses defaults if None.
        """
        self.config = config or EnergyRegistryConfig()
        self._trackers: dict[str, EnergyTracker] = {}
        self._budgets: dict[str, EnergyBudget] = {}
        self._lock = threading.RLock()
        self._callbacks: list[callable] = []

        # Initialize budgets if enabled
        if self.config.enable_budgets:
            self._init_budgets()

        # Accumulated stats
        self._start_time = time.time()
        self._total_signals = 0

    def _init_budgets(self) -> None:
        """Initialize energy budgets from config."""
        for domain, cfg in self.config.budget_configs.items():
            self._budgets[domain] = EnergyBudget(
                domain=EnergyType.LLM_TOKENS,  # Will be overwritten by tracker
                total_capacity=cfg.get("capacity", 100.0),
                current_level=cfg.get("capacity", 100.0),
                recharge_rate=cfg.get("recharge_rate", 1.0),
            )

    def register(self, tracker: EnergyTracker) -> None:
        """Register an energy tracker.

        Args:
            tracker: The tracker to register.
        """
        with self._lock:
            self._trackers[tracker.name] = tracker
            logger.debug("Registered energy tracker: %s", tracker.name)

    def unregister(self, name: str) -> EnergyTracker | None:
        """Unregister a tracker by name.

        Args:
            name: Tracker name to remove.

        Returns:
            The removed tracker, or None if not found.
        """
        with self._lock:
            return self._trackers.pop(name, None)

    def get_tracker(self, name: str) -> EnergyTracker | None:
        """Get a tracker by name.

        Args:
            name: Tracker name.

        Returns:
            The tracker, or None if not found.
        """
        return self._trackers.get(name)

    def record_signal(self, signal: EnergySignal) -> None:
        """Record an energy signal and update budgets.

        Called by trackers or directly when signals are generated.

        Args:
            signal: The energy signal to record.
        """
        with self._lock:
            self._total_signals += 1

            # Update budget for the signal's domain
            source = signal.source
            if source in self._budgets:
                self._budgets[source].spend(signal.amount)

            # Check for high energy warning
            if signal.amount >= self.config.high_energy_threshold:
                logger.warning(
                    "High energy expenditure: %s %.2f units (%s)",
                    signal.energy_type.value,
                    signal.amount,
                    signal.source,
                )

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.warning("Energy callback error: %s", e)

    def add_callback(self, callback: callable) -> None:
        """Register a callback for energy signals.

        Args:
            callback: Function(EnergySignal) to call.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: callable) -> None:
        """Remove a previously registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_summary(self, window_seconds: float = 60.0) -> dict[str, Any]:
        """Get aggregated summary across all trackers.

        Args:
            window_seconds: Time window for statistics.

        Returns:
            Dict with per-tracker stats and totals.
        """
        with self._lock:
            summary = {
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "total_signals": self._total_signals,
                "trackers": {},
                "budgets": {},
            }

            # Collect per-tracker stats
            total_energy = 0.0
            for name, tracker in self._trackers.items():
                stats = tracker.get_window_stats(window_seconds)
                summary["trackers"][name] = stats
                total_energy += stats.get("total_energy", 0.0)

            summary["total_window_energy"] = round(total_energy, 4)

            # Collect budget states
            for name, budget in self._budgets.items():
                summary["budgets"][name] = {
                    "current": round(budget.current_level, 2),
                    "capacity": budget.total_capacity,
                    "percentage": round(budget.percentage, 1),
                    "is_low": budget.is_low,
                    "is_critical": budget.is_critical,
                }

            return summary

    def get_budget(self, domain: str) -> EnergyBudget | None:
        """Get budget for a domain.

        Args:
            domain: Domain name (e.g., "llm", "movement").

        Returns:
            EnergyBudget if exists, None otherwise.
        """
        return self._budgets.get(domain)

    def is_low_energy(self, domain: str) -> bool:
        """Check if a domain has low energy.

        Args:
            domain: Domain to check.

        Returns:
            True if energy < 25%, False otherwise.
        """
        budget = self._budgets.get(domain)
        return budget.is_low if budget else False

    def is_critical_energy(self, domain: str) -> bool:
        """Check if a domain has critical energy.

        Args:
            domain: Domain to check.

        Returns:
            True if energy < 10%, False otherwise.
        """
        budget = self._budgets.get(domain)
        return budget.is_critical if budget else False

    def get_energy_recommendation(self) -> dict[str, Any]:
        """Get recommendations based on current energy state.

        Returns:
            Dict with recommendations for energy-aware decisions.
        """
        recommendations = {
            "warnings": [],
            "suggestions": [],
        }

        for name, budget in self._budgets.items():
            if budget.is_critical:
                recommendations["warnings"].append(f"{name} energy is critical ({budget.percentage:.0f}%)")
            elif budget.is_low:
                recommendations["warnings"].append(f"{name} energy is low ({budget.percentage:.0f}%)")

        # Domain-specific suggestions
        llm_budget = self._budgets.get("llm")
        if llm_budget and llm_budget.is_low:
            recommendations["suggestions"].append("Consider using a smaller/faster model (haiku instead of sonnet)")

        movement_budget = self._budgets.get("movement")
        if movement_budget and movement_budget.is_low:
            recommendations["suggestions"].append("Reduce movement frequency or use smaller movements")

        return recommendations

    def clear_all(self) -> None:
        """Clear all trackers and reset budgets."""
        with self._lock:
            for tracker in self._trackers.values():
                tracker.clear()

            # Reset budgets to full
            for budget in self._budgets.values():
                budget.current_level = budget.total_capacity
                budget.last_update = time.time()

            self._total_signals = 0
            self._start_time = time.time()


def get_global_registry() -> EnergyRegistry:
    """Get the global energy registry instance.

    Creates one if it doesn't exist.

    Returns:
        The global EnergyRegistry.
    """
    global _global_registry
    with _global_lock:
        if _global_registry is None:
            _global_registry = EnergyRegistry()
        return _global_registry


def set_global_registry(registry: EnergyRegistry) -> None:
    """Set the global energy registry.

    Args:
        registry: Registry to set as global.
    """
    global _global_registry
    with _global_lock:
        _global_registry = registry


__all__ = [
    "EnergyRegistry",
    "EnergyRegistryConfig",
    "get_global_registry",
    "set_global_registry",
]
