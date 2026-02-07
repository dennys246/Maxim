"""Energy bridge - Connects energy tracking to NAc causal learning.

Enables the system to:
1. Learn which actions/prompts are energy-expensive
2. Predict energy costs before execution
3. Make energy-aware decisions (prefer cheaper alternatives)

This parallels the PainCircuitBridge but for energy costs instead
of aversive pain signals.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from maxim.energy.signal import EnergySignal, EnergyType
from maxim.energy.registry import EnergyRegistry

if TYPE_CHECKING:
    from maxim.decisions.nac import NAc
    from maxim.decisions.causal_link import Valence

logger = logging.getLogger(__name__)


@dataclass
class EnergyBridgeConfig:
    """Configuration for energy-NAc integration.

    Attributes:
        enable_learning: Whether to record energy outcomes to NAc.
        high_energy_valence_threshold: Energy above this is NEGATIVE.
        low_energy_valence_threshold: Energy below this is POSITIVE.
        prediction_threshold: Confidence threshold for predictions.
        energy_weight: Weight for energy in composite outcomes.
    """

    enable_learning: bool = True
    high_energy_valence_threshold: float = 3.0   # > 3 energy = negative
    low_energy_valence_threshold: float = 0.5    # < 0.5 energy = positive
    prediction_threshold: float = 0.4
    energy_weight: float = 0.3                   # Weight relative to other outcomes


class EnergyCircuitBridge:
    """Connects energy tracking to NAc for causal learning.

    Learning flow:
    1. Action starts → record_action_start(signature)
    2. Energy signals accumulate during execution
    3. Action ends → record_action_end() reports to NAc

    Prediction flow:
    1. Before action → predict_energy(signature)
    2. Returns predicted energy cost based on learned patterns
    3. System can gate or modify action based on prediction

    Example:
        bridge = EnergyCircuitBridge(nac, registry)

        # Before executing an LLM call
        prediction = bridge.predict_energy("llm:planning:long")
        if prediction and prediction > 5.0:
            print("High energy predicted - consider shorter prompt")

        # Track actual energy
        event_id = bridge.record_action_start("llm:planning:long")
        # ... LLM executes ...
        bridge.record_action_end(event_id)  # Reports energy to NAc
    """

    def __init__(
        self,
        nac: "NAc | None" = None,
        registry: EnergyRegistry | None = None,
        config: EnergyBridgeConfig | None = None,
    ) -> None:
        """Initialize the energy bridge.

        Args:
            nac: NAc instance for causal learning.
            registry: Energy registry to track signals.
            config: Bridge configuration.
        """
        self._nac = nac
        self._registry = registry
        self.config = config or EnergyBridgeConfig()

        # Track pending actions
        self._pending_actions: dict[str, _PendingAction] = {}

        # Register callback for energy signals
        if self._registry:
            self._registry.add_callback(self._on_energy_signal)

        # Energy prediction cache (action_signature -> predicted_energy)
        self._prediction_cache: dict[str, float] = {}
        self._cache_ttl = 60.0  # Cache for 1 minute

    def record_action_start(
        self,
        action_signature: str,
        action_type: str = "general",
        context: dict[str, Any] | None = None,
    ) -> str:
        """Record that an action is starting.

        Args:
            action_signature: Unique identifier for the action.
            action_type: Category of action ("llm", "movement", etc.).
            context: Additional context.

        Returns:
            event_id for tracking this action.
        """
        event_id = f"{action_type}:{action_signature}:{time.time()}"

        self._pending_actions[event_id] = _PendingAction(
            event_id=event_id,
            action_signature=action_signature,
            action_type=action_type,
            start_time=time.time(),
            context=context or {},
            accumulated_energy=0.0,
        )

        # Register with NAc if available
        if self._nac and self.config.enable_learning:
            self._nac.record_event(
                event_type=action_type,
                event_signature=action_signature,
                context=context or {},
            )

        return event_id

    def record_action_end(
        self,
        event_id: str,
        additional_energy: float = 0.0,
    ) -> float | None:
        """Record that an action has completed.

        Reports accumulated energy to NAc for learning.

        Args:
            event_id: The event ID from record_action_start.
            additional_energy: Extra energy to add (if known).

        Returns:
            Total energy for this action, or None if not found.
        """
        pending = self._pending_actions.pop(event_id, None)
        if pending is None:
            return None

        # Calculate total energy
        total_energy = pending.accumulated_energy + additional_energy
        duration = time.time() - pending.start_time

        # Determine valence based on energy
        valence = self._energy_to_valence(total_energy)

        # Report to NAc
        if self._nac and self.config.enable_learning:
            from maxim.decisions.causal_link import Valence as ValenceEnum

            self._nac.record_outcome(
                event_type=pending.action_type,
                event_id=pending.action_signature,
                outcome_valence=valence,
                context={
                    "energy": round(total_energy, 4),
                    "duration_s": round(duration, 3),
                    "action_signature": pending.action_signature,
                    **pending.context,
                },
            )

            logger.debug(
                "Energy outcome: %s = %.2f energy (%s)",
                pending.action_signature,
                total_energy,
                valence.value if hasattr(valence, 'value') else valence,
            )

        return total_energy

    def _energy_to_valence(self, energy: float) -> "Valence":
        """Convert energy amount to outcome valence.

        High energy → NEGATIVE (want to avoid)
        Low energy → POSITIVE (efficient)
        Medium energy → NEUTRAL
        """
        from maxim.decisions.causal_link import Valence

        if energy >= self.config.high_energy_valence_threshold:
            return Valence.NEGATIVE
        elif energy <= self.config.low_energy_valence_threshold:
            return Valence.POSITIVE
        else:
            return Valence.NEUTRAL

    def _on_energy_signal(self, signal: EnergySignal) -> None:
        """Handle incoming energy signals.

        Accumulates energy for any pending actions that match.
        """
        # Add energy to all pending actions of matching type
        for pending in self._pending_actions.values():
            # Match by source or energy type
            if signal.source == pending.action_type or signal.source in pending.action_signature:
                pending.accumulated_energy += signal.amount

    def predict_energy(
        self,
        action_signature: str,
        action_type: str = "general",
        context: dict[str, Any] | None = None,
    ) -> float | None:
        """Predict energy cost for an action.

        Uses NAc learned associations to predict energy.

        Args:
            action_signature: Action to predict.
            action_type: Category of action.
            context: Additional context.

        Returns:
            Predicted energy, or None if no prediction available.
        """
        if not self._nac:
            return None

        # Check cache first
        cache_key = f"{action_type}:{action_signature}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        # Query NAc for prediction
        prediction = self._nac.predict(
            event_type=action_type,
            event_signature=action_signature,
            context=context,
        )

        if prediction is None:
            return None

        # Extract energy from prediction context
        if hasattr(prediction, 'context') and isinstance(prediction.context, dict):
            predicted_energy = prediction.context.get("energy")
            if predicted_energy is not None:
                # Cache the prediction
                self._prediction_cache[cache_key] = predicted_energy
                return predicted_energy

        # If no energy in context, infer from valence
        from maxim.decisions.causal_link import Valence

        if hasattr(prediction, 'predicted_valence'):
            if prediction.predicted_valence == Valence.NEGATIVE:
                return self.config.high_energy_valence_threshold * 1.5
            elif prediction.predicted_valence == Valence.POSITIVE:
                return self.config.low_energy_valence_threshold * 0.5

        return None

    def should_gate_action(
        self,
        action_signature: str,
        action_type: str = "general",
        energy_threshold: float | None = None,
    ) -> tuple[bool, str]:
        """Check if action should be gated due to predicted high energy.

        Args:
            action_signature: Action to check.
            action_type: Category of action.
            energy_threshold: Override threshold (uses config default if None).

        Returns:
            Tuple of (should_gate, reason).
        """
        threshold = energy_threshold or self.config.high_energy_valence_threshold * 2

        predicted = self.predict_energy(action_signature, action_type)
        if predicted is None:
            return False, ""

        if predicted > threshold:
            return True, f"Predicted energy {predicted:.1f} exceeds threshold {threshold:.1f}"

        # Also check current budget
        if self._registry:
            budget = self._registry.get_budget(action_type)
            if budget and budget.is_low:
                if predicted > budget.current_level * 0.5:
                    return True, f"Low budget ({budget.percentage:.0f}%) and predicted energy {predicted:.1f}"

        return False, ""

    def get_energy_context_for_llm(self) -> str:
        """Generate energy context to include in LLM prompts.

        Returns:
            String describing current energy state for LLM awareness.
        """
        if not self._registry:
            return ""

        summary = self._registry.get_summary(window_seconds=300)  # 5 min window

        lines = ["[Energy Status]"]

        # Budget status
        for domain, budget in summary.get("budgets", {}).items():
            pct = budget.get("percentage", 100)
            if pct < 50:
                lines.append(f"- {domain}: {pct:.0f}% energy remaining")

        # Recent usage
        for tracker, stats in summary.get("trackers", {}).items():
            total = stats.get("total_energy", 0)
            if total > 0:
                lines.append(f"- Recent {tracker} energy: {total:.1f} units")

        # Recommendations
        rec = self._registry.get_energy_recommendation()
        if rec.get("suggestions"):
            lines.append("Suggestions:")
            for s in rec["suggestions"][:2]:  # Max 2 suggestions
                lines.append(f"  - {s}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        self._prediction_cache.clear()


@dataclass
class _PendingAction:
    """Internal tracking for an in-progress action."""

    event_id: str
    action_signature: str
    action_type: str
    start_time: float
    context: dict[str, Any]
    accumulated_energy: float = 0.0


__all__ = [
    "EnergyCircuitBridge",
    "EnergyBridgeConfig",
]
