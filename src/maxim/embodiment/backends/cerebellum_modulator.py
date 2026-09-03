"""CerebellumModulator — wraps Cerebellum prediction with LLM fallback.

On ``execute()``:
1. Check Cerebellum confidence for (entity, affordance, params)
2. If confident: return predicted sensor readings (no LLM call)
3. If not confident: delegate to fallback modulator (typically LLMModulator)
4. Train Cerebellum on the fallback result

This slots into the SEM protocol — the Entity doesn't know or care
whether its modulator is backed by Cerebellum, LLM, rules, or hardware.
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.embodiment.cerebellum import Cerebellum
from maxim.embodiment.sem import (
    AffordanceSchema,
    Entity,
    ModulatorResult,
)

log = logging.getLogger(__name__)


class CerebellumModulator:
    """Modulator that uses Cerebellum for prediction, with LLM fallback.

    Parameters
    ----------
    entity : Entity
        The entity this modulator is attached to.
    modulator_name : str
        Name of the modulator (e.g., "motor").
    affordances : dict[str, AffordanceSchema]
        Affordance schemas from the spec.
    cerebellum : Cerebellum
        The Cerebellum instance for forward model lookup/training.
    fallback : Modulator-like, optional
        Fallback modulator to use when Cerebellum has no confident model.
        Typically an ``LLMModulator``.
        If None, returns stub results when Cerebellum can't predict.
    sensor_ranges : dict, optional
        ``{sensor_name: (lo, hi)}`` for param bucketing.
    """

    __slots__ = (
        "_entity",
        "_name",
        "_affordances",
        "_cerebellum",
        "_fallback",
        "_sensor_ranges",
        "_reaction_bus",
    )

    def __init__(
        self,
        entity: Entity,
        modulator_name: str,
        affordances: dict[str, AffordanceSchema],
        cerebellum: Cerebellum,
        *,
        fallback: Any = None,
        sensor_ranges: dict[str, tuple[float, float]] | None = None,
        reaction_bus: Any = None,
    ) -> None:
        self._entity = entity
        self._name = modulator_name
        self._affordances = affordances
        self._cerebellum = cerebellum
        self._fallback = fallback
        self._sensor_ranges = sensor_ranges
        self._reaction_bus = reaction_bus

    @property
    def name(self) -> str:
        return self._name

    @property
    def affordances(self) -> dict[str, AffordanceSchema]:
        return self._affordances

    def execute(self, affordance: str, params: dict[str, Any]) -> ModulatorResult:
        """Execute an affordance, using Cerebellum prediction if confident.

        Flow:
        1. Build model key from entity + modulator + affordance + bucketed params
        2. Ask Cerebellum for prediction
        3. If confident prediction exists:
           - Apply predicted sensor changes to entity
           - Return result with predicted_changes metadata
        4. If no confident prediction:
           - Delegate to fallback modulator (LLM/rules/hardware)
           - Train Cerebellum on the fallback result
        """
        if affordance not in self._affordances:
            result = ModulatorResult(
                success=False,
                modulator_name=self._name,
                entity_name=self._entity.name,
                affordance=affordance,
                params=params,
                error=f"Unknown affordance: {affordance}",
            )
            self._emit_failure_reaction(affordance, 0.3)
            return result

        # Try Cerebellum prediction first
        predicted = self._cerebellum.predict(
            self._entity,
            self._name,
            affordance,
            params,
            self._sensor_ranges,
        )

        if predicted is not None:
            # Cerebellum is confident — use cached prediction
            self._apply_predictions(predicted)
            self._emit_success_reaction(affordance, min(0.3, 0.1 + 0.05 * len(predicted)))
            return ModulatorResult(
                success=True,
                modulator_name=self._name,
                entity_name=self._entity.name,
                affordance=affordance,
                params=params,
                metadata={
                    "source": "cerebellum",
                    "predicted_changes": predicted,
                },
            )

        # No confident model — fall back
        self._cerebellum.record_llm_fallback()

        if self._fallback is None:
            return ModulatorResult(
                success=True,
                modulator_name=self._name,
                entity_name=self._entity.name,
                affordance=affordance,
                params=params,
                metadata={"source": "stub_no_fallback"},
            )

        # Execute via fallback (LLM, rules, hardware)
        result = self._fallback.execute(affordance, params)

        # Train Cerebellum on the result
        if result.success:
            actual = self._collect_actual_readings()
            if actual:
                key = self._cerebellum.make_key(
                    self._entity,
                    self._name,
                    affordance,
                    params,
                    self._sensor_ranges,
                )
                self._cerebellum.observe(key, actual)
                log.debug(
                    "Cerebellum trained: %s.%s.%s (obs=%d)",
                    self._entity.name,
                    self._name,
                    affordance,
                    self._cerebellum._models[key].observations,
                )

        # Emit failure Reaction if the fallback execution failed
        if not result.success:
            self._emit_failure_reaction(affordance, min(0.5, 0.3 + 0.1))

        # Tag result with source
        if result.metadata is None:
            result = ModulatorResult(
                success=result.success,
                modulator_name=result.modulator_name,
                entity_name=result.entity_name,
                affordance=result.affordance,
                params=result.params,
                error=result.error,
                metadata={"source": "fallback"},
            )

        return result

    def _emit_failure_reaction(self, affordance: str, intensity: float) -> None:
        """Emit a Reaction(kind="pain") when a modulator execution fails.

        SEM facilitates reactions via Cerebellum mediation — the modulator
        itself doesn't know about the Reaction type; this wrapper handles
        the translation from "execution failed" to "evaluative signal."
        """
        if self._reaction_bus is None:
            return
        try:
            import time

            from maxim.decisions.causal_link import Valence
            from maxim.reactions.types import WORLD_AGENT_ID, Reaction, ReactionContext, TraceSnapshot

            reaction = Reaction(
                kind="pain",
                intensity=intensity,
                valence=Valence.NEGATIVE,
                timestamp=time.time(),
                source=f"cerebellum:{self._entity.name}.{self._name}.{affordance}",
                context=ReactionContext(
                    bindings={"entity_path": TraceSnapshot(percept_id=self._entity.name)},
                    agent_id=WORLD_AGENT_ID,
                ),
            )
            self._reaction_bus.publish(reaction)
        except Exception as e:
            log.debug("CerebellumModulator: failed to emit reaction: %s", e)

    def _emit_success_reaction(self, affordance: str, intensity: float) -> None:
        """Emit a Reaction(kind="reward") when Cerebellum confidently predicts.

        Mirrors ``_emit_failure_reaction`` but with positive valence.
        Intensity is deliberately lower than failure intensity (negativity
        bias: failures are more informative than routine successes).
        """
        if self._reaction_bus is None:
            return
        try:
            import time

            from maxim.decisions.causal_link import Valence
            from maxim.reactions.types import WORLD_AGENT_ID, Reaction, ReactionContext, TraceSnapshot

            reaction = Reaction(
                kind="reward",
                intensity=intensity,
                valence=Valence.POSITIVE,
                timestamp=time.time(),
                source=f"cerebellum:{self._entity.name}.{self._name}.{affordance}",
                context=ReactionContext(
                    bindings={"entity_path": TraceSnapshot(percept_id=self._entity.name)},
                    agent_id=WORLD_AGENT_ID,
                ),
            )
            self._reaction_bus.publish(reaction)
        except Exception as e:
            log.debug("CerebellumModulator: failed to emit success reaction: %s", e)

    def _apply_predictions(self, predicted: dict[str, float]) -> None:
        """Apply Cerebellum predictions to entity sensors/vital_metrics."""
        for sensor_name, new_value in predicted.items():
            sensor = self._entity.sensors.get(sensor_name)
            if sensor is not None and hasattr(sensor, "update_value"):
                sensor.update_value(new_value)
            self._entity.vital_metrics[sensor_name] = float(new_value)

    def _collect_actual_readings(self) -> dict[str, float]:
        """Collect current scalar sensor readings for training."""
        actual: dict[str, float] = {}
        for sname, sensor in self._entity.sensors.items():
            schema = sensor.reading_schema
            if schema.get("type") not in ("float", "int"):
                continue
            try:
                reading = sensor.read()
                actual[sname] = float(reading.value)
            except Exception:
                pass
        return actual


def cerebellum_modulator_factory(
    cerebellum: Cerebellum,
    fallback_factory: Any = None,
    reaction_bus: Any = None,
) -> Any:
    """Dormant since 2026-05-26: the entire Cerebellum backend path is
    infrastructure ready for SEM wiring but not yet invoked from any
    runtime startup path. Grep shows only test callers in
    ``tests/unit/test_producer_protocols.py``. Awaits a new experiment
    that earns the SEM-wiring path. See
    ``docs/plans/archive/reaction_bus_unification.md`` for the gap audit. Do not
    delete (per CLAUDE.md Principle 2) — the ``reaction_bus`` parameter
    was preemptively added to close Gap A so the next caller wires it
    correctly.

    Create a factory function for ``attach_backends()``.

    Usage::

        from maxim.embodiment.spec import attach_backends

        factory = cerebellum_modulator_factory(
            cerebellum,
            fallback_factory=llm_mod_factory,
            reaction_bus=pain_bus.reaction_bus,
        )
        attach_backends(root, modulator_factory=factory)

    Parameters
    ----------
    cerebellum : Cerebellum
        Shared Cerebellum instance.
    fallback_factory : callable, optional
        ``(entity, mod_name, spec_mod) -> Modulator`` for fallback.
        If None, CerebellumModulator runs without fallback.
    reaction_bus : ReactionBus or None, optional
        ``ReactionBus`` for modulator failure reactions. When
        ``None``, ``CerebellumModulator._emit_failure_reaction``
        silently no-ops — every modulator failure signal is dropped.
        Pre-audit (``docs/plans/archive/reaction_bus_unification.md`` Gap A)
        this parameter did not exist on the factory and was ALWAYS
        ``None`` in production. Pass ``pain_bus.reaction_bus`` (or
        the standalone ReactionBus from ``build_reaction_bus``) to
        close the gap. See
        ``docs/plans/archive/reaction_bus_unification.md`` for the full
        audit.

    Returns
    -------
    callable
        Factory function matching ``attach_backends()`` signature.
    """

    def factory(entity: Entity, mod_name: str, spec_mod: Any) -> CerebellumModulator:
        fallback = None
        if fallback_factory is not None:
            fallback = fallback_factory(entity, mod_name, spec_mod)

        # Collect sensor ranges from entity for param bucketing
        sensor_ranges: dict[str, tuple[float, float]] = {}
        for sname, sensor in entity.sensors.items():
            schema = sensor.reading_schema
            if "range" in schema:
                sensor_ranges[sname] = tuple(schema["range"])  # type: ignore[arg-type]

        return CerebellumModulator(
            entity=entity,
            modulator_name=mod_name,
            affordances=spec_mod.affordances,
            cerebellum=cerebellum,
            fallback=fallback,
            sensor_ranges=sensor_ranges,
            reaction_bus=reaction_bus,
        )

    return factory
