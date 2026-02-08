"""Simulated robot implementation for testing.

Provides a SimulatedController that mimics robot behavior
without requiring actual hardware.
"""

from maxim.hardware.simulation.controller import SimulatedController
from maxim.hardware.simulation.streams import MockAudioStream, MockVideoStream

__all__ = [
    "SimulatedController",
    "MockVideoStream",
    "MockAudioStream",
]
