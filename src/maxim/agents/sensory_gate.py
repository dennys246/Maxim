"""Entity-modulated percept filtering via SensoryGate.

Filters percepts through an entity's sensory capacity. Each entity can
have a ``perception`` child entity with acuity sensors (sight_acuity,
hearing_acuity, etc.) that define how well it perceives each modality.

Percepts below the acuity threshold are dropped or reduced in salience.
This mirrors how real sensory organs work — the retina shapes photons
through receptor density, adaptation, and gating before they reach
the brain.

Example::

    from maxim.agents.sensory_gate import SensoryGate

    gate = SensoryGate(perception_entity)
    filtered = gate.modulate(percept)
    if filtered is None:
        print("Entity cannot perceive this (blind/deaf)")
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.agents.modality import SensoryModality, SensoryTag

log = logging.getLogger(__name__)

# Maps SensoryModality → sensor name on the perception bundle entity.
# Each sensor is a ratio [0, 1] where 0 = no capacity, 1 = perfect.
MODALITY_SENSOR_MAP: dict[SensoryModality, str] = {
    SensoryModality.SIGHT: "sight_acuity",
    SensoryModality.SOUND: "hearing_acuity",
    SensoryModality.SMELL: "smell_acuity",
    SensoryModality.TOUCH: "pain_sensitivity",
    SensoryModality.INTEROCEPTION: "intuition",
}

# Acuity at or below this threshold → percept dropped entirely
ACUITY_FLOOR: float = 0.05


class SensoryGate:
    """Filters percepts through an entity's sensory capacity.

    The entity's perception bundle sensors define acuity per modality.
    Percepts below the acuity threshold are dropped or reduced in salience.

    NARRATIVE and ABSTRACT percepts pass through unmodulated — they're
    meta-information, not sensory input.

    Parameters
    ----------
    perception_entity : Entity
        The entity (or child entity) that holds acuity sensors.
        Typically ``character.find("perception")`` or the character itself
        if acuity sensors are on the root.
    """

    def __init__(self, perception_entity: Any) -> None:
        self._perception = perception_entity

    def modulate(self, percept: Any) -> Any | None:
        """Filter a percept through entity sensory capacity.

        Parameters
        ----------
        percept : Percept
            The incoming percept (must have a ``sensory`` attribute).

        Returns
        -------
        Percept | None
            Modified percept with ``perceived_intensity`` set, or None
            if the entity can't perceive this modality (acuity ≤ 0.05).
            Legacy percepts (``sensory is None``) pass through unchanged.
        """
        sensory = getattr(percept, "sensory", None)
        if sensory is None:
            return percept  # Legacy percept — pass through

        modality = sensory.modality
        # Meta-percepts aren't filtered by senses
        if modality in (SensoryModality.NARRATIVE, SensoryModality.ABSTRACT):
            return percept

        sensor_name = MODALITY_SENSOR_MAP.get(modality)
        if sensor_name is None:
            return percept  # Unknown modality — pass through

        # Read the entity's acuity for this modality
        sensors = getattr(self._perception, "sensors", {})
        sensor = sensors.get(sensor_name)
        if sensor is None:
            return percept  # No sensor for this modality — pass through unmodulated

        try:
            reading = sensor.read()
            acuity = float(reading.value)
        except Exception:
            return percept  # Sensor read failed — pass through

        # Gate: drop percepts the entity can't perceive
        if acuity <= ACUITY_FLOOR:
            log.debug(
                "SensoryGate: dropped %s percept (acuity=%.3f for %s)",
                modality.value,
                acuity,
                sensor_name,
            )
            try:
                from maxim.simulation.sim_logger import sim_sensory

                sim_sensory(modality.value, sensory.entity_source or "unknown", acuity, dropped=True)
            except Exception:
                pass
            return None

        # Modulate: perceived intensity = raw intensity * acuity
        perceived = sensory.intensity * acuity

        # Build new sensory tag with modulation results
        new_tag = SensoryTag(
            modality=sensory.modality,
            submodality=sensory.submodality,
            spatial_source=sensory.spatial_source,
            intensity=sensory.intensity,
            entity_source=sensory.entity_source,
            perceived_intensity=perceived,
            modulated_by=f"{self._perception.full_path}.{sensor_name}"
            if hasattr(self._perception, "full_path")
            else sensor_name,
        )

        # Modulate the percept's salience by acuity
        new_salience = (percept.salience or 0.5) * acuity

        # Return a modified copy
        # Percept is a dataclass — copy all fields and override
        try:
            from dataclasses import fields as dc_fields

            field_vals = {f.name: getattr(percept, f.name) for f in dc_fields(percept)}
            field_vals["sensory"] = new_tag
            field_vals["salience"] = new_salience
            return percept.__class__(**field_vals)
        except Exception:
            # Fallback: just update in place if dataclass copy fails
            percept.sensory = new_tag
            percept.salience = new_salience
            return percept

    def get_acuity(self, modality: SensoryModality) -> float:
        """Read the current acuity for a modality.

        Returns 1.0 if no sensor exists (unmodulated).
        """
        sensor_name = MODALITY_SENSOR_MAP.get(modality)
        if sensor_name is None:
            return 1.0
        sensors = getattr(self._perception, "sensors", {})
        sensor = sensors.get(sensor_name)
        if sensor is None:
            return 1.0
        try:
            return float(sensor.read().value)
        except Exception:
            return 1.0

    def is_blind(self) -> bool:
        """Check if the entity effectively can't see."""
        return self.get_acuity(SensoryModality.SIGHT) <= ACUITY_FLOOR

    def is_deaf(self) -> bool:
        """Check if the entity effectively can't hear."""
        return self.get_acuity(SensoryModality.SOUND) <= ACUITY_FLOOR
