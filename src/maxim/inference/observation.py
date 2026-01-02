"""
Public observation entrypoints.

This module re-exports the observation routines from smaller modules to keep the
implementation modular while preserving the public API.
"""

from maxim.inference.observation_face import face_observation
from maxim.inference.motor_cortex_observation import motor_cortex_control
from maxim.inference.observation_passive import passive_listening, passive_observation

__all__ = [
    "face_observation",
    "motor_cortex_control",
    "passive_listening",
    "passive_observation",
]
