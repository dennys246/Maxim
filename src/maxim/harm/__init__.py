"""Predictive harm detection system.

Provides a general framework for predicting harmful outcomes BEFORE
actions are executed, enabling proactive gating and risk mitigation.

Subsystems register harm predictors that analyze action parameters
and return risk assessments. The FearAgent queries these predictors
to make gating decisions.

Example:
    from maxim.harm import HarmRegistry, MovementHarmPredictor

    # Create registry and register predictors
    registry = HarmRegistry()
    registry.register(MovementHarmPredictor(velocity_threshold=100.0))

    # Before executing an action
    prediction = registry.predict_harm(
        action_type="movement",
        action_params={"action_signature": "look_at:dy=90:dp=30", "duration": 0.3},
    )

    if prediction.intensity > 0.5:
        print(f"High risk: {prediction.reason}")
"""

from maxim.harm.predictor import (
    AggregatedHarmPrediction,
    HarmCategory,
    HarmPrediction,
    HarmPredictor,
)
from maxim.harm.registry import HarmRegistry, get_global_registry
from maxim.harm.movement import MovementHarmPredictor, MovementHarmConfig
from maxim.harm.joint_limit import JointLimitHarmPredictor, JointLimitConfig
from maxim.harm.tool_predictor import ToolHarmPredictor

__all__ = [
    # Core types
    "HarmCategory",
    "HarmPrediction",
    "HarmPredictor",
    "AggregatedHarmPrediction",
    # Registry
    "HarmRegistry",
    "get_global_registry",
    # Concrete predictors - Movement
    "MovementHarmPredictor",
    "MovementHarmConfig",
    # Concrete predictors - Joint limits
    "JointLimitHarmPredictor",
    "JointLimitConfig",
    # Concrete predictors - Tool failure
    "ToolHarmPredictor",
]
