"""Predicts tool execution harm using NAc learned associations."""

from __future__ import annotations

from typing import Any

from maxim.decisions.nac import NAc
from maxim.harm.predictor import HarmCategory, HarmPrediction, HarmPredictor


class ToolHarmPredictor(HarmPredictor):
    """Predicts tool failure risk using learned NAc associations.

    Queries NAc for negative causal links associated with a given tool
    and returns a HarmPrediction when the learned failure risk exceeds
    a threshold.

    Example:
        predictor = ToolHarmPredictor(nac=nac)
        prediction = predictor.predict(
            action_type="tool_call",
            action_params={"tool_name": "web_search"},
        )
        if prediction and prediction.should_gate:
            print(f"Tool risky: {prediction.reason}")
    """

    name: str = "tool_harm"
    categories = {HarmCategory.TOOL_FAILURE}

    def __init__(self, nac: NAc) -> None:
        self._nac = nac

    def can_predict(self, action_type: str) -> bool:
        """Only handles tool_call actions."""
        return action_type == "tool_call"

    def predict(
        self,
        action_type: str,
        action_params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> HarmPrediction | None:
        """Predict harm from a tool call based on learned failure patterns.

        Args:
            action_type: Must be "tool_call".
            action_params: Must contain "tool_name".
            context: Optional context.

        Returns:
            HarmPrediction if learned failure risk is above threshold, None otherwise.
        """
        tool_name = action_params.get("tool_name", "")
        negative_links = self._nac.get_negative_outcomes(f"tool:{tool_name}")
        if not negative_links:
            return None

        worst = max(
            negative_links,
            key=lambda link: link.confidence * (1 - link.predicted_value),
        )
        risk = worst.confidence * (1 - worst.predicted_value)
        if risk < 0.3:
            return None

        return HarmPrediction(
            category=HarmCategory.TOOL_FAILURE,
            intensity=risk,
            confidence=worst.confidence,
            reason=(
                f"Tool '{tool_name}' has learned failure rate "
                f"(value={worst.predicted_value:.2f}, "
                f"obs={worst.observation_count})"
            ),
            source=self.name,
            action_type=action_type,
            action_params=action_params,
            mitigation="Consider alternative tool or verify preconditions",
            metadata={
                "tool_name": tool_name,
                "predicted_value": worst.predicted_value,
                "observation_count": worst.observation_count,
            },
        )


__all__ = ["ToolHarmPredictor"]
