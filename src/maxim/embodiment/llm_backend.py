"""LLM and narrative backends for SEM sensors and modulators.

Provides four backend classes:
- ``LLMSensor`` / ``LLMModulator`` — for robotic embodiment (ATL-grounded)
- ``NarrativeSensor`` / ``NarrativeModulator`` — for virtual entities
  in campaigns/simulations (swords, NPCs, environments)

All backends use the LLMRouter for inference and inject ATL body-part
concepts for grounding.  The Cerebellum (Phase 1a) will wrap these
to cache predictions and reduce LLM calls.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from maxim.embodiment.sem import (
    AffordanceSchema,
    Entity,
    ModulatorResult,
    SensorReading,
)

log = logging.getLogger(__name__)


class LLMSensor:
    """Sensor backed by LLM inference for novel/simulated situations.

    Reads entity state by asking the LLM to predict sensor values
    given the current context.  Used when no hardware sensor exists
    (simulation mode) or when Cerebellum has no confident model.
    """

    def __init__(
        self,
        entity: Entity,
        sensor_name: str,
        unit: str,
        schema: dict[str, Any],
        *,
        llm_router: Any = None,
        atl: Any = None,
    ) -> None:
        self._entity = entity
        self._name = sensor_name
        self._unit = unit
        self._schema = schema
        self._llm = llm_router
        self._atl = atl
        self._last_value: Any = schema.get("initial", 0.0)

    @property
    def name(self) -> str:
        return self._name

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def reading_schema(self) -> dict[str, Any]:
        return self._schema

    def read(self) -> SensorReading:
        """Return the last known/predicted sensor value.

        Note: LLM-predicted values are updated by the modulator after
        actions, not on every read (LLM called once per action, not per tick).
        """
        return SensorReading(
            sensor_name=self._name,
            entity_name=self._entity.name,
            value=self._last_value,
            unit=self._unit,
            timestamp=time.time(),
        )

    def update_value(self, value: Any) -> None:
        """Update the cached sensor value (called by modulator/cerebellum)."""
        self._last_value = value


class LLMModulator:
    """Modulator backed by LLM inference for novel/simulated actions.

    On execute():
    1. Build prompt with entity state + ATL body concepts + action
    2. Call LLMRouter for predicted sensor outcomes
    3. Update entity's sensor values from prediction
    4. Return structured result

    The Cerebellum (Phase 1a) will wrap this to cache predictions.
    """

    def __init__(
        self,
        entity: Entity,
        modulator_name: str,
        affordances: dict[str, AffordanceSchema],
        *,
        llm_router: Any = None,
        atl: Any = None,
    ) -> None:
        self._entity = entity
        self._name = modulator_name
        self._affordances = affordances
        self._llm = llm_router
        self._atl = atl

    @property
    def name(self) -> str:
        return self._name

    @property
    def affordances(self) -> dict[str, AffordanceSchema]:
        return self._affordances

    def execute(self, affordance: str, params: dict[str, Any]) -> ModulatorResult:
        """Execute an affordance via LLM prediction."""
        if affordance not in self._affordances:
            return ModulatorResult(
                success=False,
                modulator_name=self._name,
                entity_name=self._entity.name,
                affordance=affordance,
                params=params,
                error=f"Unknown affordance: {affordance}",
            )

        predicted_changes = self._predict_outcome(affordance, params)

        # Apply predicted changes to entity sensors
        for sensor_name, new_value in predicted_changes.items():
            sensor = self._entity.sensors.get(sensor_name)
            if sensor is not None and hasattr(sensor, "update_value"):
                sensor.update_value(new_value)
            # Also update vital metrics
            self._entity.vital_metrics[sensor_name] = float(new_value)

        return ModulatorResult(
            success=True,
            modulator_name=self._name,
            entity_name=self._entity.name,
            affordance=affordance,
            params=params,
            metadata={"predicted_changes": predicted_changes},
        )

    def _predict_outcome(
        self,
        affordance: str,
        params: dict[str, Any],
    ) -> dict[str, float]:
        """Predict sensor changes from executing this affordance.

        If LLMRouter is available, calls it with ATL-grounded context.
        Otherwise, uses simple heuristic predictions.
        """
        if self._llm is not None:
            return self._predict_via_llm(affordance, params)
        return self._predict_heuristic(affordance, params)

    def _predict_via_llm(
        self,
        affordance: str,
        params: dict[str, Any],
    ) -> dict[str, float]:
        """Call LLM for percept prediction with ATL context."""
        # Build context from entity state + ATL concepts
        current_state = {}
        for sname, sensor in self._entity.sensors.items():
            schema = sensor.reading_schema
            if schema.get("type") in ("float", "int"):
                try:
                    reading = sensor.read()
                    current_state[sname] = float(reading.value)
                except Exception:
                    pass

        atl_context = ""
        if self._atl is not None:
            try:
                concepts = self._atl.recall(
                    category="body_part",
                    query=self._entity.name,
                    limit=3,
                )
                if concepts:
                    atl_context = "\n".join(f"- {c.name}: {c.definition}" for c in concepts)
            except Exception:
                pass

        prompt = (
            f"Entity: {self._entity.name} ({self._entity.entity_type})\n"
            f"Current state: {current_state}\n"
            f"Action: {affordance}({params})\n"
        )
        if atl_context:
            prompt += f"\nRelevant knowledge:\n{atl_context}\n"
        prompt += "\nPredict the sensor values after this action. Return JSON: {sensor_name: new_value, ...}"

        try:
            result = self._llm.generate_json(prompt, max_tokens=200)
            if isinstance(result, dict):
                return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
        except Exception as e:
            log.debug("LLM prediction failed for %s.%s: %s", self._entity.name, affordance, e)

        return self._predict_heuristic(affordance, params)

    def _predict_heuristic(
        self,
        affordance: str,
        params: dict[str, Any],
    ) -> dict[str, float]:
        """Simple heuristic predictions when LLM is not available.

        Applies basic physics-like rules based on entity type and affordance.
        """
        changes: dict[str, float] = {}

        for sname, sensor in self._entity.sensors.items():
            schema = sensor.reading_schema
            if schema.get("type") not in ("float", "int"):
                continue
            try:
                reading = sensor.read()
                current = float(reading.value)
            except Exception:
                continue

            rng = schema.get("range", [0, 1])
            lo, hi = rng[0], rng[1]

            # Apply affordance-specific heuristics
            if affordance.startswith("rotate") and sname == "angle":
                degrees = float(params.get("degrees", 0))
                changes[sname] = max(lo, min(hi, current + degrees))
            elif affordance in ("sharpen",) and sname == "sharpness":
                changes[sname] = min(hi, current + 0.1)
            elif affordance in ("repair",) and sname == "durability":
                changes[sname] = min(hi, current + 0.2)
            elif affordance in ("slash", "punch") and sname == "durability":
                changes[sname] = max(lo, current - 0.05)
            elif affordance in ("slash", "punch") and sname == "strain":
                changes[sname] = min(hi, current + 0.1)
            elif affordance in ("threaten",) and sname == "trust":
                changes[sname] = max(lo, current - 0.2)
            elif affordance in ("offer_payment",) and sname == "trust":
                changes[sname] = min(hi, current + 0.15)

        return changes


# ---------------------------------------------------------------------------
# Narrative backends (virtual entities: swords, NPCs, environments)
# ---------------------------------------------------------------------------


class NarrativeSensor(LLMSensor):
    """Sensor for virtual entities in campaigns/simulations.

    Reads from internal state model (vital_metrics on Entity).
    Functionally identical to LLMSensor but with narrative-flavored
    identity for logging/debugging.
    """

    def read(self) -> SensorReading:
        """Read from entity's vital_metrics (narrative world model)."""
        val = self._entity.vital_metrics.get(self._name, self._last_value)
        self._last_value = val
        return SensorReading(
            sensor_name=self._name,
            entity_name=self._entity.name,
            value=val,
            unit=self._unit,
            timestamp=time.time(),
        )


class NarrativeModulator(LLMModulator):
    """Modulator for virtual entities in campaigns/simulations.

    Same mechanics as LLMModulator but generates narrative-flavored
    prompts and applies changes to the entity's vital_metrics.
    """

    def _predict_via_llm(
        self,
        affordance: str,
        params: dict[str, Any],
    ) -> dict[str, float]:
        """Call LLM with narrative context for virtual entity actions."""
        current_state = dict(self._entity.vital_metrics)

        prompt = (
            f"In this narrative world, the entity '{self._entity.name}' "
            f"(a {self._entity.entity_type}) has current state: {current_state}\n"
            f"The action '{affordance}' is performed with params: {params}\n"
            f"What are the new sensor values after this action? "
            f"Return JSON: {{sensor_name: new_value, ...}}"
        )

        if self._llm is not None:
            try:
                result = self._llm.generate_json(prompt, max_tokens=200)
                if isinstance(result, dict):
                    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
            except Exception as e:
                log.debug(
                    "Narrative LLM prediction failed for %s.%s: %s",
                    self._entity.name,
                    affordance,
                    e,
                )

        return self._predict_heuristic(affordance, params)
