"""Central registry for harm predictors.

Aggregates predictions from multiple subsystem predictors and provides
a unified interface for querying harm predictions.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from maxim.harm.predictor import (
    AggregatedHarmPrediction,
    HarmPrediction,
    HarmPredictor,
)

logger = logging.getLogger(__name__)


class HarmRegistry:
    """Central registry for harm predictors.

    Manages registration of predictors and provides methods to query
    all relevant predictors for a given action.

    Example:
        registry = HarmRegistry()

        # Register predictors
        registry.register(MovementHarmPredictor(velocity_threshold=100.0))
        registry.register(ComputationalHarmPredictor(timeout_threshold=30.0))

        # Query all predictors
        result = registry.predict_all(
            action_type="movement",
            action_params={"action_signature": "look_at:dy=90"},
        )

        if result.should_gate:
            print(f"Gating action: {result.reasons}")
    """

    def __init__(self) -> None:
        """Initialize the harm registry."""
        self._predictors: list[HarmPredictor] = []
        self._stats = {
            "total_queries": 0,
            "total_predictions": 0,
            "predictions_by_category": {},
            "query_time_ms_avg": 0.0,
        }
        self._query_times: list[float] = []

    def register(self, predictor: HarmPredictor) -> None:
        """Register a harm predictor.

        Args:
            predictor: The predictor to register.
        """
        self._predictors.append(predictor)
        logger.info(
            "Registered harm predictor: %s (categories=%s)",
            predictor.name,
            [c.value for c in predictor.categories],
        )

    def unregister(self, predictor: HarmPredictor) -> bool:
        """Unregister a harm predictor.

        Args:
            predictor: The predictor to remove.

        Returns:
            True if removed, False if not found.
        """
        if predictor in self._predictors:
            self._predictors.remove(predictor)
            logger.info("Unregistered harm predictor: %s", predictor.name)
            return True
        return False

    def predict_all(
        self,
        action_type: str,
        action_params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> AggregatedHarmPrediction:
        """Query all relevant predictors and aggregate results.

        Args:
            action_type: Type of action being evaluated.
            action_params: Parameters for the action.
            context: Optional context about current state.

        Returns:
            Aggregated prediction from all predictors.
        """
        start_time = time.perf_counter()
        predictions: list[HarmPrediction] = []

        for predictor in self._predictors:
            if not predictor.can_predict(action_type):
                continue

            try:
                prediction = predictor.predict(action_type, action_params, context)
                if prediction is not None:
                    predictions.append(prediction)
                    self._record_prediction(prediction)
            except Exception as e:
                logger.warning(
                    "Harm predictor %s failed: %s",
                    predictor.name,
                    e,
                )

        # Record timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._query_times.append(elapsed_ms)
        if len(self._query_times) > 100:
            self._query_times = self._query_times[-100:]
        self._stats["query_time_ms_avg"] = sum(self._query_times) / len(self._query_times)
        self._stats["total_queries"] += 1

        return AggregatedHarmPrediction(
            predictions=predictions,
            action_type=action_type,
            action_params=action_params,
        )

    def predict_worst(
        self,
        action_type: str,
        action_params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> HarmPrediction | None:
        """Get the worst (highest risk) prediction for an action.

        Convenience method that returns just the most severe prediction.

        Args:
            action_type: Type of action being evaluated.
            action_params: Parameters for the action.
            context: Optional context about current state.

        Returns:
            The highest-risk prediction, or None if no harm predicted.
        """
        result = self.predict_all(action_type, action_params, context)
        return result.worst

    def should_gate(
        self,
        action_type: str,
        action_params: dict[str, Any],
        context: dict[str, Any] | None = None,
        threshold: float = 0.3,
    ) -> tuple[bool, str]:
        """Check if an action should be gated.

        Args:
            action_type: Type of action being evaluated.
            action_params: Parameters for the action.
            context: Optional context about current state.
            threshold: Risk score threshold for gating (default 0.3).

        Returns:
            Tuple of (should_gate, reason).
        """
        result = self.predict_all(action_type, action_params, context)

        if not result.predictions:
            return False, ""

        worst = result.worst
        if worst and worst.risk_score >= threshold:
            return True, worst.reason

        return False, ""

    def get_risk_score(
        self,
        action_type: str,
        action_params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> float:
        """Get the overall risk score for an action (0-1).

        Args:
            action_type: Type of action being evaluated.
            action_params: Parameters for the action.
            context: Optional context about current state.

        Returns:
            Maximum risk score from all predictors.
        """
        result = self.predict_all(action_type, action_params, context)
        return result.max_risk_score

    def _record_prediction(self, prediction: HarmPrediction) -> None:
        """Record a prediction for statistics."""
        self._stats["total_predictions"] += 1
        cat = prediction.category.value
        if cat not in self._stats["predictions_by_category"]:
            self._stats["predictions_by_category"][cat] = 0
        self._stats["predictions_by_category"][cat] += 1

    @property
    def predictor_count(self) -> int:
        """Number of registered predictors."""
        return len(self._predictors)

    @property
    def predictors(self) -> list[HarmPredictor]:
        """List of registered predictors (read-only copy)."""
        return list(self._predictors)

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            **self._stats,
            "predictor_count": self.predictor_count,
            "predictor_names": [p.name for p in self._predictors],
        }


# Global registry instance (optional - subsystems can create their own)
_global_registry: HarmRegistry | None = None


def get_global_registry() -> HarmRegistry:
    """Get or create the global harm registry.

    For systems that want a single shared registry.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = HarmRegistry()
    return _global_registry


__all__ = ["HarmRegistry", "get_global_registry"]
