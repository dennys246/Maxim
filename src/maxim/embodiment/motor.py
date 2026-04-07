"""Motor programs — learned SEM action sequences + bidirectional registry.

A motor program is an ordered list of SEM affordance invocations that
achieves a named outcome.  Programs are discovered by the Cerebellum
when the agent repeats the same SEM sequence for the same goal 3+ times.

The ``ProgramRegistry`` indexes programs in three directions:
- By goal → "I want to reach forward" → matching programs
- By entity type → "I'm holding a sword" → programs involving swords
- By affordance → "I want to slash" → programs with slash steps

This enables bidirectional prompt injection: query an entity and get
its programs, or query an action and get entities that can do it.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MotorStep:
    """One step in a motor program — a single SEM affordance invocation."""

    entity_path: str
    modulator: str
    affordance: str
    params: dict[str, Any] = field(default_factory=dict)
    expected_duration_s: float = 1.0
    expected_sensor_state: dict[str, float] | None = None
    pain_gate: dict[str, Any] | None = None

    @property
    def sem_key(self) -> tuple[str, str, str]:
        """The SEM triple identifying this step's structure (ignoring params)."""
        return (self.entity_path, self.modulator, self.affordance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_path": self.entity_path,
            "modulator": self.modulator,
            "affordance": self.affordance,
            "params": dict(self.params),
            "expected_duration_s": self.expected_duration_s,
            "expected_sensor_state": dict(self.expected_sensor_state) if self.expected_sensor_state else None,
            "pain_gate": dict(self.pain_gate) if self.pain_gate else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotorStep:
        return cls(
            entity_path=data["entity_path"],
            modulator=data["modulator"],
            affordance=data["affordance"],
            params=data.get("params", {}),
            expected_duration_s=data.get("expected_duration_s", 1.0),
            expected_sensor_state=data.get("expected_sensor_state"),
            pain_gate=data.get("pain_gate"),
        )


@dataclass
class MotorProgram:
    """A learned sequence of SEM actions that achieves a named outcome."""

    name: str
    goal_signature: str
    steps: list[MotorStep] = field(default_factory=list)
    confidence: float = 0.0
    total_executions: int = 0
    total_successes: int = 0
    avg_duration_s: float = 0.0
    last_used: float = 0.0
    context_requirements: dict[str, Any] = field(default_factory=dict)
    known_failure_modes: list[str] = field(default_factory=list)
    source: str = "discovered"  # "discovered" | "simulation"

    @property
    def structure_key(self) -> tuple[tuple[str, str, str], ...]:
        """Ordered SEM triples — identifies the program structure (ignoring params)."""
        return tuple(step.sem_key for step in self.steps)

    @property
    def entity_types(self) -> set[str]:
        """Unique entity paths involved in this program."""
        return {step.entity_path for step in self.steps}

    @property
    def affordance_names(self) -> set[str]:
        """Unique affordance names in this program."""
        return {step.affordance for step in self.steps}

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_successes / self.total_executions

    def record_execution(self, success: bool, duration_s: float) -> None:
        """Update stats after an execution."""
        self.total_executions += 1
        if success:
            self.total_successes += 1
        self.last_used = time.time()
        # Running average of duration
        if self.avg_duration_s == 0:
            self.avg_duration_s = duration_s
        else:
            self.avg_duration_s = 0.8 * self.avg_duration_s + 0.2 * duration_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "goal_signature": self.goal_signature,
            "steps": [s.to_dict() for s in self.steps],
            "confidence": self.confidence,
            "total_executions": self.total_executions,
            "total_successes": self.total_successes,
            "avg_duration_s": self.avg_duration_s,
            "last_used": self.last_used,
            "context_requirements": dict(self.context_requirements),
            "known_failure_modes": list(self.known_failure_modes),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotorProgram:
        return cls(
            name=data["name"],
            goal_signature=data["goal_signature"],
            steps=[MotorStep.from_dict(s) for s in data.get("steps", [])],
            confidence=data.get("confidence", 0.0),
            total_executions=data.get("total_executions", 0),
            total_successes=data.get("total_successes", 0),
            avg_duration_s=data.get("avg_duration_s", 0.0),
            last_used=data.get("last_used", 0.0),
            context_requirements=data.get("context_requirements", {}),
            known_failure_modes=data.get("known_failure_modes", []),
            source=data.get("source", "discovered"),
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        """Format for prompt injection."""
        return {
            "name": self.name,
            "goal": self.goal_signature,
            "confidence": self.confidence,
            "success_rate": self.success_rate,
            "executions": self.total_executions,
            "steps": [f"{s.entity_path}.{s.affordance}({s.params})" for s in self.steps],
            "risks": self.known_failure_modes,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Sequence observation (pre-crystallization tracking)
# ---------------------------------------------------------------------------


@dataclass
class _ObservedSequence:
    """Tracks repeated goal → SEM-sequence observations before crystallization."""

    goal: str
    structure: tuple[tuple[str, str, str], ...]
    param_observations: list[list[dict[str, Any]]]  # per-step param lists
    outcome_count: int = 0
    success_count: int = 0
    total_duration: float = 0.0

    def add_observation(
        self,
        step_params: list[dict[str, Any]],
        success: bool,
        duration_s: float,
    ) -> None:
        self.param_observations.append(step_params)
        self.outcome_count += 1
        if success:
            self.success_count += 1
        self.total_duration += duration_s

    def averaged_params(self) -> list[dict[str, Any]]:
        """Average numeric params across observations for crystallization."""
        if not self.param_observations:
            return []

        n_steps = len(self.structure)
        result: list[dict[str, Any]] = []

        for step_idx in range(n_steps):
            merged: dict[str, Any] = {}
            counts: dict[str, int] = {}

            for obs in self.param_observations:
                if step_idx >= len(obs):
                    continue
                for k, v in obs[step_idx].items():
                    if isinstance(v, (int, float)):
                        merged[k] = merged.get(k, 0.0) + v
                        counts[k] = counts.get(k, 0) + 1
                    elif k not in merged:
                        merged[k] = v  # keep first non-numeric value

            for k in list(merged.keys()):
                if k in counts and counts[k] > 0 and isinstance(merged[k], (int, float)):
                    merged[k] = merged[k] / counts[k]

            result.append(merged)
        return result


# ---------------------------------------------------------------------------
# Program Registry — triple-indexed for bidirectional queries
# ---------------------------------------------------------------------------


class ProgramRegistry:
    """Triple-indexed registry for motor programs.

    Indexes programs by:
    - Goal signature → "what programs achieve this goal?"
    - Entity path → "what programs involve this entity?"
    - Affordance name → "what programs use this action?"

    This enables bidirectional prompt injection:
    - Query "sword" → get slash, parry, throw programs
    - Query "slash" → get sword, axe, arm programs
    """

    def __init__(self) -> None:
        self._programs: dict[str, MotorProgram] = {}  # name → program
        self._by_goal: dict[str, set[str]] = defaultdict(set)
        self._by_entity: dict[str, set[str]] = defaultdict(set)
        self._by_affordance: dict[str, set[str]] = defaultdict(set)
        self._by_entity_type: dict[str, set[str]] = defaultdict(set)

        # Pre-crystallization tracking
        self._observed: dict[str, _ObservedSequence] = {}  # obs_key → tracking
        self._crystallization_threshold: int = 3

    # -- registration -------------------------------------------------------

    def register(self, program: MotorProgram) -> None:
        """Register a motor program and index it."""
        self._programs[program.name] = program
        self._by_goal[program.goal_signature].add(program.name)
        for step in program.steps:
            self._by_entity[step.entity_path].add(program.name)
            self._by_affordance[step.affordance].add(program.name)
            # Index by entity type (last segment of path as heuristic)
            entity_type = step.entity_path.rsplit(".", 1)[-1]
            self._by_entity_type[entity_type].add(program.name)

    def remove(self, name: str) -> MotorProgram | None:
        """Remove a program from all indexes."""
        program = self._programs.pop(name, None)
        if program is None:
            return None

        self._by_goal[program.goal_signature].discard(name)
        for step in program.steps:
            self._by_entity[step.entity_path].discard(name)
            self._by_affordance[step.affordance].discard(name)
            entity_type = step.entity_path.rsplit(".", 1)[-1]
            self._by_entity_type[entity_type].discard(name)

        return program

    # -- queries ------------------------------------------------------------

    def get(self, name: str) -> MotorProgram | None:
        return self._programs.get(name)

    def find_by_goal(self, goal: str) -> list[MotorProgram]:
        """Find programs matching a goal signature."""
        names = self._by_goal.get(goal, set())
        return [self._programs[n] for n in names if n in self._programs]

    def find_by_entity(self, entity_path: str) -> list[MotorProgram]:
        """Find programs involving a specific entity.

        Query "sword" → get all programs that have sword in any step.
        """
        names: set[str] = set()
        # Exact match
        names.update(self._by_entity.get(entity_path, set()))
        # Entity type match (e.g., "sword" matches "rusty_sword" entity type)
        names.update(self._by_entity_type.get(entity_path, set()))
        # Partial path match (e.g., "shoulder" matches "arm.shoulder")
        for path, prog_names in self._by_entity.items():
            if path.endswith(f".{entity_path}") or path == entity_path:
                names.update(prog_names)
        return [self._programs[n] for n in names if n in self._programs]

    def find_by_affordance(self, affordance: str) -> list[MotorProgram]:
        """Find programs that use a specific affordance.

        Query "slash" → get programs for sword, axe, arm, etc.
        """
        names = self._by_affordance.get(affordance, set())
        return [self._programs[n] for n in names if n in self._programs]

    def find_related(self, query: str) -> list[MotorProgram]:
        """Find programs matching by goal, entity, or affordance.

        Unified search — tries all three indexes and deduplicates.
        """
        seen: set[str] = set()
        results: list[MotorProgram] = []

        for prog in self.find_by_goal(query):
            if prog.name not in seen:
                seen.add(prog.name)
                results.append(prog)
        for prog in self.find_by_entity(query):
            if prog.name not in seen:
                seen.add(prog.name)
                results.append(prog)
        for prog in self.find_by_affordance(query):
            if prog.name not in seen:
                seen.add(prog.name)
                results.append(prog)

        return sorted(results, key=lambda p: p.confidence, reverse=True)

    def all_programs(self) -> list[MotorProgram]:
        return list(self._programs.values())

    # -- sequence observation + crystallization -----------------------------

    def observe_sequence(
        self,
        goal: str,
        steps: list[tuple[str, str, str, dict[str, Any]]],
        success: bool,
        duration_s: float,
    ) -> MotorProgram | None:
        """Observe an action sequence. Crystallize if seen 3+ times.

        Parameters
        ----------
        goal : str
            Goal signature this sequence achieved.
        steps : list of (entity_path, modulator, affordance, params)
            The ordered SEM actions.
        success : bool
            Whether the sequence achieved the goal.
        duration_s : float
            Total execution duration.

        Returns
        -------
        MotorProgram or None
            Newly crystallized program, or None if not enough observations.
        """
        structure = tuple((ep, mod, aff) for ep, mod, aff, _ in steps)
        obs_key = f"{goal}|{'->'.join(f'{ep}.{mod}.{aff}' for ep, mod, aff in structure)}"

        if obs_key not in self._observed:
            self._observed[obs_key] = _ObservedSequence(
                goal=goal,
                structure=structure,
                param_observations=[],
            )

        obs = self._observed[obs_key]
        step_params = [params for _, _, _, params in steps]
        obs.add_observation(step_params, success, duration_s)

        if obs.outcome_count >= self._crystallization_threshold and obs.success_count > 0:
            return self._crystallize(obs_key, obs)

        return None

    def _crystallize(self, obs_key: str, obs: _ObservedSequence) -> MotorProgram:
        """Create a MotorProgram from accumulated observations."""
        averaged = obs.averaged_params()

        motor_steps: list[MotorStep] = []
        for i, (ep, mod, aff) in enumerate(obs.structure):
            params = averaged[i] if i < len(averaged) else {}
            motor_steps.append(
                MotorStep(
                    entity_path=ep,
                    modulator=mod,
                    affordance=aff,
                    params=params,
                    expected_duration_s=obs.total_duration / max(1, obs.outcome_count * len(obs.structure)),
                )
            )

        # Generate a descriptive name
        entities = "_".join(dict.fromkeys(ep.rsplit(".", 1)[-1] for ep, _, _ in obs.structure))
        affordances = "_".join(dict.fromkeys(aff for _, _, aff in obs.structure))
        name = f"{entities}_{affordances}"

        # Avoid name collisions
        base_name = name
        counter = 1
        while name in self._programs:
            name = f"{base_name}_{counter}"
            counter += 1

        program = MotorProgram(
            name=name,
            goal_signature=obs.goal,
            steps=motor_steps,
            confidence=obs.success_count / obs.outcome_count,
            total_executions=obs.outcome_count,
            total_successes=obs.success_count,
            avg_duration_s=obs.total_duration / obs.outcome_count,
            last_used=time.time(),
        )

        self.register(program)
        del self._observed[obs_key]

        return program

    # -- persistence --------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        return {
            "programs": {name: prog.to_dict() for name, prog in self._programs.items()},
        }

    def import_state(self, data: dict[str, Any]) -> None:
        self._programs.clear()
        self._by_goal.clear()
        self._by_entity.clear()
        self._by_affordance.clear()
        self._by_entity_type.clear()
        for name, prog_data in data.get("programs", {}).items():
            program = MotorProgram.from_dict(prog_data)
            self.register(program)

    def stats(self) -> dict[str, Any]:
        return {
            "program_count": len(self._programs),
            "pending_observations": len(self._observed),
            "goal_index_size": len(self._by_goal),
            "entity_index_size": len(self._by_entity),
            "affordance_index_size": len(self._by_affordance),
        }


# ---------------------------------------------------------------------------
# Entity state similarity (scalar-only, pain-proximity weighted)
# ---------------------------------------------------------------------------


def entity_state_similarity(
    remembered: dict[str, dict[str, float]],
    current: dict[str, dict[str, float]],
    sensor_ranges: dict[str, dict[str, tuple[float, float]]],
    pain_thresholds: dict[str, dict[str, float]] | None = None,
) -> float:
    """Compare scalar sensor states between remembered and current context.

    Returns 0.0 (completely different) to 1.0 (identical).
    Only compares sensors that exist in both snapshots.
    Non-scalar sensors (frames, audio) must be excluded by caller.

    Parameters
    ----------
    remembered : dict
        ``{entity_path: {sensor_name: value}}`` from engram.
    current : dict
        ``{entity_path: {sensor_name: value}}`` from live sensors.
    sensor_ranges : dict
        ``{entity_path: {sensor_name: (lo, hi)}}`` from YAML spec.
    pain_thresholds : dict, optional
        ``{entity_path: {sensor_name: threshold_value}}``.
        Pain-proximate sensors get 2x weight.
    """
    similarities: list[float] = []
    weights: list[float] = []

    for entity_path in remembered:
        if entity_path not in current:
            continue
        for sensor_name, remembered_val in remembered[entity_path].items():
            if sensor_name not in current[entity_path]:
                continue
            current_val = current[entity_path][sensor_name]
            lo, hi = sensor_ranges.get(entity_path, {}).get(sensor_name, (0, 1))
            range_size = max(hi - lo, 1e-6)
            sim = 1.0 - min(abs(current_val - remembered_val) / range_size, 1.0)
            similarities.append(sim)

            weight = 1.0
            if pain_thresholds:
                threshold = pain_thresholds.get(entity_path, {}).get(sensor_name)
                if threshold is not None:
                    proximity = 1.0 - min(abs(current_val - threshold) / range_size, 1.0)
                    if proximity > 0.8:
                        weight = 2.0
            weights.append(weight)

    if not similarities:
        return 0.0
    return sum(s * w for s, w in zip(similarities, weights)) / sum(weights)
