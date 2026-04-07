"""Embodiment subsystem — Sensor-Entity-Modulator (SEM) protocol.

Provides composable hardware abstraction via three protocols:
- Entity: a physical thing (joint, camera, wheel, sword, NPC)
- Sensor: reads state from an entity (angle, durability, trust)
- Modulator: changes state of an entity (rotate, slash, speak)

Entities compose into trees. Auto-generates agent tools, Cerebellum
model keys, ATL concepts, and pain triggers from the SEM graph.
"""

from maxim.embodiment.sem import (
    AffordanceSchema,
    Entity,
    Modulator,
    ModulatorResult,
    Sensor,
    SensorReading,
)

__all__ = [
    "AffordanceSchema",
    "Entity",
    "Modulator",
    "ModulatorResult",
    "Sensor",
    "SensorReading",
]
