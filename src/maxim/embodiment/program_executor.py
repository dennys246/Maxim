"""Program executor — step-by-step motor program runner with pain gates.

Executes a ``MotorProgram`` step by step, checking pain gates before
each step and subscribing to the ``PainBus`` for mid-sequence interrupts.

When pain fires mid-sequence:
1. Execution halts at the painful step
2. The step's pain_gate is tightened
3. The program's known_failure_modes is updated
4. An engram is formed (if hippocampus is available)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from maxim.embodiment.motor import MotorProgram, MotorStep
from maxim.embodiment.sem import Entity, ModulatorResult

log = logging.getLogger(__name__)


@dataclass
class ProgramExecutionResult:
    """Result of executing a motor program."""

    success: bool
    program_name: str
    steps_completed: int
    total_steps: int
    duration_s: float
    pain_step: int | None = None  # step index that triggered pain
    pain_intensity: float = 0.0
    step_results: list[ModulatorResult] = field(default_factory=list)
    error: str | None = None

    @property
    def valence(self) -> str:
        if self.pain_step is not None:
            return "NEGATIVE"
        return "POSITIVE" if self.success else "NEUTRAL"


class ProgramExecutor:
    """Executes motor programs step by step with pain-gate checks.

    Parameters
    ----------
    pain_bus : PainBus or None
        If provided, subscribes for mid-sequence pain interrupts.
    embodiment : Embodiment or None
        If provided, reads entity sensors for pain gate evaluation.
    """

    def __init__(
        self,
        *,
        pain_bus: Any = None,
        embodiment: Any = None,
    ) -> None:
        self._pain_bus = pain_bus
        self._embodiment = embodiment
        self._interrupted = False
        self._interrupt_signal: Any = None

    def execute(
        self,
        program: MotorProgram,
        entity_root: Entity,
    ) -> ProgramExecutionResult:
        """Execute a motor program step by step.

        Flow for each step:
        1. Check pain gate — if sensor exceeds gate threshold, abort or modify
        2. Find the entity and modulator for this step
        3. Execute the affordance
        4. Check for pain interrupts

        Parameters
        ----------
        program : MotorProgram
            The program to execute.
        entity_root : Entity
            Root of the entity tree (for finding entities by path).

        Returns
        -------
        ProgramExecutionResult
            Detailed result including which step failed (if any).
        """
        self._interrupted = False
        self._interrupt_signal = None
        start_time = time.time()

        # Subscribe to PainBus for mid-sequence interrupts
        if self._pain_bus is not None:
            self._pain_bus.subscribe(self._on_pain)

        step_results: list[ModulatorResult] = []
        pain_intensity = 0.0

        try:
            for step_idx, step in enumerate(program.steps):
                # 1. Check pain gate
                gate_result = self._check_pain_gate(step, entity_root)
                if gate_result is not None:
                    log.info(
                        "Pain gate triggered at step %d of %s: %s",
                        step_idx,
                        program.name,
                        gate_result,
                    )
                    return ProgramExecutionResult(
                        success=False,
                        program_name=program.name,
                        steps_completed=step_idx,
                        total_steps=len(program.steps),
                        duration_s=time.time() - start_time,
                        pain_step=step_idx,
                        pain_intensity=0.0,
                        step_results=step_results,
                        error=f"Pain gate at step {step_idx}: {gate_result}",
                    )

                # 2. Find entity and modulator
                entity = self._find_entity(step.entity_path, entity_root)
                if entity is None:
                    return ProgramExecutionResult(
                        success=False,
                        program_name=program.name,
                        steps_completed=step_idx,
                        total_steps=len(program.steps),
                        duration_s=time.time() - start_time,
                        step_results=step_results,
                        error=f"Entity not found: {step.entity_path}",
                    )

                modulator = entity.modulators.get(step.modulator)
                if modulator is None:
                    return ProgramExecutionResult(
                        success=False,
                        program_name=program.name,
                        steps_completed=step_idx,
                        total_steps=len(program.steps),
                        duration_s=time.time() - start_time,
                        step_results=step_results,
                        error=f"Modulator not found: {step.modulator} on {step.entity_path}",
                    )

                # 3. Execute affordance
                result = modulator.execute(step.affordance, step.params)
                step_results.append(result)

                if not result.success:
                    return ProgramExecutionResult(
                        success=False,
                        program_name=program.name,
                        steps_completed=step_idx,
                        total_steps=len(program.steps),
                        duration_s=time.time() - start_time,
                        step_results=step_results,
                        error=f"Step {step_idx} failed: {result.error}",
                    )

                # 4. Check for pain interrupt
                if self._interrupted:
                    if self._interrupt_signal is not None:
                        pain_intensity = getattr(self._interrupt_signal, "intensity", 0.0)
                    log.info(
                        "Pain interrupt at step %d of %s (intensity=%.2f)",
                        step_idx,
                        program.name,
                        pain_intensity,
                    )

                    # Tighten pain gate for this step
                    self._tighten_pain_gate(step, entity)

                    # Record failure mode
                    if self._interrupt_signal is not None:
                        ctx = getattr(self._interrupt_signal, "context", {})
                        fm_name = ctx.get("failure_mode", "unknown")
                        if fm_name not in program.known_failure_modes:
                            program.known_failure_modes.append(fm_name)

                    return ProgramExecutionResult(
                        success=False,
                        program_name=program.name,
                        steps_completed=step_idx + 1,
                        total_steps=len(program.steps),
                        duration_s=time.time() - start_time,
                        pain_step=step_idx,
                        pain_intensity=pain_intensity,
                        step_results=step_results,
                    )

            # All steps completed successfully
            return ProgramExecutionResult(
                success=True,
                program_name=program.name,
                steps_completed=len(program.steps),
                total_steps=len(program.steps),
                duration_s=time.time() - start_time,
                step_results=step_results,
            )

        finally:
            # Always unsubscribe
            if self._pain_bus is not None:
                self._pain_bus.unsubscribe(self._on_pain)

    def _on_pain(self, signal: Any) -> None:
        """PainBus callback — interrupt execution on embodiment pain."""
        ctx = getattr(signal, "context", {})
        if ctx.get("source") == "embodiment":
            self._interrupted = True
            self._interrupt_signal = signal

    def _check_pain_gate(self, step: MotorStep, entity_root: Entity) -> str | None:
        """Check if a step's pain gate would fire given current sensor state.

        Returns a reason string if gated, None if safe to proceed.
        """
        if step.pain_gate is None:
            return None

        entity = self._find_entity(step.entity_path, entity_root)
        if entity is None:
            return None

        field_name = step.pain_gate.get("field")
        op = step.pain_gate.get("op", ">")
        threshold = step.pain_gate.get("value", 0)

        if field_name is None:
            return None

        # Read current sensor value
        current_val = entity.vital_metrics.get(field_name)
        if current_val is None:
            sensor = entity.sensors.get(field_name)
            if sensor is not None:
                try:
                    reading = sensor.read()
                    current_val = float(reading.value)
                except Exception:
                    return None
            else:
                return None

        # Evaluate gate
        import operator as _op

        ops = {">": _op.gt, "<": _op.lt, ">=": _op.ge, "<=": _op.le}
        op_fn = ops.get(op)
        if op_fn and op_fn(current_val, threshold):
            return f"{field_name} {op} {threshold} (current={current_val:.2f})"

        return None

    def _tighten_pain_gate(self, step: MotorStep, entity: Entity) -> None:
        """Tighten a step's pain gate after a painful execution.

        If no gate exists, create one from the current sensor state.
        If a gate exists, tighten the threshold by 10%.
        """
        if step.pain_gate is None:
            # Create a gate from the most pain-relevant sensor
            for fm in entity.failure_modes:
                for trigger in fm.triggers:
                    sensor_val = entity.vital_metrics.get(trigger.field)
                    if sensor_val is not None:
                        # Set gate 10% before the failure threshold
                        range_size = abs(trigger.value)
                        offset = range_size * 0.1
                        if trigger.op in (">", ">="):
                            step.pain_gate = {
                                "field": trigger.field,
                                "op": ">",
                                "value": trigger.value - offset,
                            }
                        elif trigger.op in ("<", "<="):
                            step.pain_gate = {
                                "field": trigger.field,
                                "op": "<",
                                "value": trigger.value + offset,
                            }
                        return
        else:
            # Tighten existing gate by 10%
            current_threshold = step.pain_gate.get("value", 0)
            op = step.pain_gate.get("op", ">")
            if op in (">", ">="):
                step.pain_gate["value"] = current_threshold * 0.9
            elif op in ("<", "<="):
                step.pain_gate["value"] = current_threshold * 1.1

    def _find_entity(self, path: str, root: Entity) -> Entity | None:
        """Find an entity by path, checking root name first."""
        if path == root.name:
            return root
        # Try as relative path from root
        found = root.find(path)
        if found is not None:
            return found
        # Try matching by name in tree
        for ent in root.walk():
            if ent.name == path or ent.full_path == path:
                return ent
        return None
