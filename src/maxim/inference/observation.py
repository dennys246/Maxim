"""
Public observation entrypoints.

This module re-exports the observation routines from smaller modules to keep the
implementation modular while preserving the public API.

Phase 2 note: motor_cortex_control is deprecated for new code.
Use the agentic runtime for movement decisions instead.
"""

from maxim.inference.observation_face import face_observation
from maxim.inference.motor_cortex_observation import motor_cortex_control
from maxim.inference.observation_passive import (
    display_detections,
    passive_listening,
    passive_observation,
)

__all__ = [
    "display_detections",
    "face_observation",
    "motor_cortex_control",  # Deprecated: use agentic runtime
    "passive_listening",
    "passive_observation",
]
