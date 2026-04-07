"""Tests for motor programs (Phase 1b).

Covers:
- MotorStep / MotorProgram dataclasses and serialization
- ProgramRegistry: register, remove, triple-indexed queries
- Sequence observation and crystallization (3+ repeats)
- Param averaging across observations
- Bidirectional queries (by entity, affordance, goal)
- Program executor with pain gates
- Pain gate tightening after failure
- Entity state similarity
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from maxim.embodiment.cerebellum import Cerebellum
from maxim.embodiment.motor import (
    MotorProgram,
    MotorStep,
    ProgramRegistry,
    entity_state_similarity,
)
from maxim.embodiment.program_executor import ProgramExecutor
from maxim.embodiment.sem import Entity
from maxim.embodiment.spec import load_spec

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "embodiment"


# ---------------------------------------------------------------------------
# MotorStep / MotorProgram
# ---------------------------------------------------------------------------


class TestMotorStep:
    def test_sem_key(self):
        step = MotorStep("arm.shoulder", "motor", "rotate_angle", {"degrees": 45})
        assert step.sem_key == ("arm.shoulder", "motor", "rotate_angle")

    def test_serialization(self):
        step = MotorStep("arm.shoulder", "motor", "rotate_angle", {"degrees": 45})
        data = step.to_dict()
        step2 = MotorStep.from_dict(data)
        assert step2.entity_path == step.entity_path
        assert step2.affordance == step.affordance
        assert step2.params == step.params


class TestMotorProgram:
    def test_structure_key(self):
        prog = MotorProgram(
            name="reach",
            goal_signature="reach_forward",
            steps=[
                MotorStep("arm.shoulder", "motor", "rotate_angle"),
                MotorStep("arm.elbow", "motor", "rotate_angle"),
            ],
        )
        assert prog.structure_key == (
            ("arm.shoulder", "motor", "rotate_angle"),
            ("arm.elbow", "motor", "rotate_angle"),
        )

    def test_entity_types(self):
        prog = MotorProgram(
            name="reach",
            goal_signature="reach",
            steps=[
                MotorStep("arm.shoulder", "motor", "rotate_angle"),
                MotorStep("arm.elbow", "motor", "rotate_angle"),
            ],
        )
        assert prog.entity_types == {"arm.shoulder", "arm.elbow"}

    def test_affordance_names(self):
        prog = MotorProgram(
            name="combat",
            goal_signature="attack",
            steps=[
                MotorStep("sword", "combat", "slash"),
                MotorStep("shield", "defense", "parry"),
            ],
        )
        assert prog.affordance_names == {"slash", "parry"}

    def test_success_rate(self):
        prog = MotorProgram(name="x", goal_signature="y")
        assert prog.success_rate == 0.0
        prog.record_execution(True, 1.0)
        prog.record_execution(True, 1.0)
        prog.record_execution(False, 1.0)
        assert abs(prog.success_rate - 2 / 3) < 0.01

    def test_serialization(self):
        prog = MotorProgram(
            name="reach",
            goal_signature="reach_forward",
            steps=[MotorStep("arm.shoulder", "motor", "rotate_angle", {"degrees": 45})],
            confidence=0.8,
            total_executions=10,
            total_successes=8,
        )
        data = prog.to_dict()
        prog2 = MotorProgram.from_dict(data)
        assert prog2.name == "reach"
        assert prog2.confidence == 0.8
        assert len(prog2.steps) == 1

    def test_prompt_dict(self):
        prog = MotorProgram(
            name="reach",
            goal_signature="reach_forward",
            steps=[MotorStep("arm.shoulder", "motor", "rotate_angle", {"degrees": 45})],
            confidence=0.82,
        )
        d = prog.to_prompt_dict()
        assert d["name"] == "reach"
        assert d["confidence"] == 0.82
        assert len(d["steps"]) == 1


# ---------------------------------------------------------------------------
# ProgramRegistry
# ---------------------------------------------------------------------------


class TestProgramRegistry:
    def test_register_and_get(self):
        reg = ProgramRegistry()
        prog = MotorProgram(name="reach", goal_signature="reach_forward")
        reg.register(prog)
        assert reg.get("reach") is prog

    def test_remove(self):
        reg = ProgramRegistry()
        prog = MotorProgram(name="reach", goal_signature="reach_forward")
        reg.register(prog)
        removed = reg.remove("reach")
        assert removed is prog
        assert reg.get("reach") is None

    def test_find_by_goal(self):
        reg = ProgramRegistry()
        reg.register(
            MotorProgram(
                name="reach1",
                goal_signature="reach_forward",
                steps=[MotorStep("arm.shoulder", "motor", "rotate_angle")],
            )
        )
        reg.register(
            MotorProgram(
                name="reach2",
                goal_signature="reach_forward",
                steps=[MotorStep("arm.elbow", "motor", "rotate_angle")],
            )
        )
        reg.register(
            MotorProgram(
                name="attack",
                goal_signature="attack_enemy",
                steps=[MotorStep("sword", "combat", "slash")],
            )
        )
        results = reg.find_by_goal("reach_forward")
        assert len(results) == 2
        assert all(p.goal_signature == "reach_forward" for p in results)

    def test_find_by_entity(self):
        reg = ProgramRegistry()
        reg.register(
            MotorProgram(
                name="swing_sword",
                goal_signature="attack",
                steps=[MotorStep("rusty_sword", "combat", "slash")],
            )
        )
        reg.register(
            MotorProgram(
                name="throw_sword",
                goal_signature="throw",
                steps=[MotorStep("rusty_sword", "combat", "throw")],
            )
        )
        reg.register(
            MotorProgram(
                name="punch",
                goal_signature="attack",
                steps=[MotorStep("right_arm.fist", "combat", "punch")],
            )
        )
        results = reg.find_by_entity("rusty_sword")
        assert len(results) == 2
        names = {p.name for p in results}
        assert "swing_sword" in names
        assert "throw_sword" in names

    def test_find_by_affordance(self):
        reg = ProgramRegistry()
        reg.register(
            MotorProgram(
                name="sword_slash",
                goal_signature="attack",
                steps=[MotorStep("sword", "combat", "slash")],
            )
        )
        reg.register(
            MotorProgram(
                name="axe_slash",
                goal_signature="attack",
                steps=[MotorStep("axe", "combat", "slash")],
            )
        )
        reg.register(
            MotorProgram(
                name="punch",
                goal_signature="attack",
                steps=[MotorStep("arm", "combat", "punch")],
            )
        )
        results = reg.find_by_affordance("slash")
        assert len(results) == 2
        names = {p.name for p in results}
        assert "sword_slash" in names
        assert "axe_slash" in names

    def test_find_related_deduplicates(self):
        reg = ProgramRegistry()
        prog = MotorProgram(
            name="sword_slash",
            goal_signature="slash",
            steps=[MotorStep("sword", "combat", "slash")],
            confidence=0.8,
        )
        reg.register(prog)
        # "slash" matches by goal AND affordance — should not duplicate
        results = reg.find_related("slash")
        assert len(results) == 1

    def test_crystallization(self):
        reg = ProgramRegistry()
        steps = [
            ("arm.shoulder", "motor", "rotate_angle", {"degrees": 45.0}),
            ("arm.elbow", "motor", "rotate_angle", {"degrees": 30.0}),
        ]
        # First 2 observations — no crystallization
        assert reg.observe_sequence("reach_forward", steps, True, 2.0) is None
        assert reg.observe_sequence("reach_forward", steps, True, 1.8) is None
        # 3rd observation — crystallize
        result = reg.observe_sequence("reach_forward", steps, True, 2.2)
        assert result is not None
        assert isinstance(result, MotorProgram)
        assert result.goal_signature == "reach_forward"
        assert len(result.steps) == 2

    def test_crystallization_averages_params(self):
        reg = ProgramRegistry()
        for degrees in [40.0, 44.0, 46.0]:
            steps = [("arm.shoulder", "motor", "rotate_angle", {"degrees": degrees})]
            reg.observe_sequence("reach", steps, True, 1.0)
        # Should have crystallized with averaged degrees
        programs = reg.find_by_goal("reach")
        assert len(programs) == 1
        avg = programs[0].steps[0].params["degrees"]
        assert abs(avg - (40 + 44 + 46) / 3) < 0.1

    def test_persistence(self):
        reg = ProgramRegistry()
        reg.register(
            MotorProgram(
                name="reach",
                goal_signature="reach_forward",
                steps=[MotorStep("arm.shoulder", "motor", "rotate_angle", {"degrees": 45})],
                confidence=0.8,
            )
        )
        state = reg.export_state()
        reg2 = ProgramRegistry()
        reg2.import_state(state)
        assert reg2.get("reach") is not None
        assert reg2.get("reach").confidence == 0.8


# ---------------------------------------------------------------------------
# Program executor
# ---------------------------------------------------------------------------


class TestProgramExecutor:
    def _build_arm_entity(self) -> Entity:
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")
        return spec.root_entity

    def test_successful_execution(self):
        root = self._build_arm_entity()
        program = MotorProgram(
            name="safe_move",
            goal_signature="move",
            steps=[
                MotorStep("shoulder", "motor", "rotate_angle", {"degrees": 45, "speed": 1.0}),
                MotorStep("elbow", "motor", "rotate_angle", {"degrees": 30, "speed": 1.0}),
            ],
        )
        executor = ProgramExecutor()
        result = executor.execute(program, root)
        assert result.success
        assert result.steps_completed == 2
        assert result.pain_step is None

    def test_entity_not_found(self):
        root = self._build_arm_entity()
        program = MotorProgram(
            name="bad_move",
            goal_signature="move",
            steps=[MotorStep("nonexistent", "motor", "rotate_angle", {"degrees": 45})],
        )
        executor = ProgramExecutor()
        result = executor.execute(program, root)
        assert not result.success
        assert "Entity not found" in (result.error or "")

    def test_pain_gate_blocks(self):
        root = self._build_arm_entity()
        shoulder = root.find("shoulder")
        assert shoulder is not None
        shoulder.vital_metrics["angle"] = 176  # Past gate

        program = MotorProgram(
            name="dangerous_move",
            goal_signature="move",
            steps=[
                MotorStep(
                    "shoulder",
                    "motor",
                    "rotate_angle",
                    {"degrees": 180, "speed": 1.0},
                    pain_gate={"field": "angle", "op": ">", "value": 170},
                ),
            ],
        )
        executor = ProgramExecutor()
        result = executor.execute(program, root)
        assert not result.success
        assert result.pain_step == 0
        assert "Pain gate" in (result.error or "")

    def test_pain_interrupt(self):
        root = self._build_arm_entity()
        pain_bus = MagicMock()

        # Simulate pain firing during execution
        def subscribe_and_fire(callback):
            # Immediately fire a pain signal
            signal = MagicMock()
            signal.intensity = 0.8
            signal.context = {"source": "embodiment", "failure_mode": "overextension"}
            callback(signal)

        pain_bus.subscribe = subscribe_and_fire
        pain_bus.unsubscribe = MagicMock()

        program = MotorProgram(
            name="painful_move",
            goal_signature="move",
            steps=[
                MotorStep("shoulder", "motor", "rotate_angle", {"degrees": 90, "speed": 1.0}),
            ],
        )
        executor = ProgramExecutor(pain_bus=pain_bus)
        result = executor.execute(program, root)
        assert not result.success
        assert result.pain_step == 0
        assert result.pain_intensity == 0.8
        pain_bus.unsubscribe.assert_called()

    def test_pain_gate_tightening(self):
        root = self._build_arm_entity()
        step = MotorStep(
            "shoulder",
            "motor",
            "rotate_angle",
            {"degrees": 90, "speed": 1.0},
            pain_gate={"field": "angle", "op": ">", "value": 170},
        )
        original_threshold = step.pain_gate["value"]

        # Simulate tightening
        executor = ProgramExecutor()
        shoulder = root.find("shoulder")
        assert shoulder is not None
        executor._tighten_pain_gate(step, shoulder)

        # Gate should be tighter (lower threshold for > operator)
        assert step.pain_gate["value"] < original_threshold


# ---------------------------------------------------------------------------
# Entity state similarity
# ---------------------------------------------------------------------------


class TestEntityStateSimilarity:
    def test_identical_states(self):
        state = {"arm.shoulder": {"angle": 90.0, "temp": 30.0}}
        ranges = {"arm.shoulder": {"angle": (0, 180), "temp": (20, 80)}}
        sim = entity_state_similarity(state, state, ranges)
        assert sim == 1.0

    def test_completely_different(self):
        remembered = {"arm.shoulder": {"angle": 0.0}}
        current = {"arm.shoulder": {"angle": 180.0}}
        ranges = {"arm.shoulder": {"angle": (0, 180)}}
        sim = entity_state_similarity(remembered, current, ranges)
        assert sim == 0.0

    def test_partial_similarity(self):
        remembered = {"arm.shoulder": {"angle": 45.0}}
        current = {"arm.shoulder": {"angle": 90.0}}
        ranges = {"arm.shoulder": {"angle": (0, 180)}}
        sim = entity_state_similarity(remembered, current, ranges)
        assert 0.8 > sim > 0.5  # 25% of range apart → ~0.75

    def test_missing_entity(self):
        remembered = {"arm.shoulder": {"angle": 45.0}}
        current = {"arm.elbow": {"angle": 90.0}}
        ranges = {"arm.shoulder": {"angle": (0, 180)}}
        sim = entity_state_similarity(remembered, current, ranges)
        assert sim == 0.0

    def test_pain_proximity_weighting(self):
        remembered = {"arm.shoulder": {"angle": 170.0, "temp": 30.0}}
        current = {"arm.shoulder": {"angle": 172.0, "temp": 31.0}}
        ranges = {"arm.shoulder": {"angle": (0, 180), "temp": (20, 80)}}
        thresholds = {"arm.shoulder": {"angle": 175.0}}

        sim_weighted = entity_state_similarity(remembered, current, ranges, thresholds)
        sim_unweighted = entity_state_similarity(remembered, current, ranges)
        # The angle is near its pain threshold, so it gets 2x weight
        # Both are very similar, but weighting should slightly change the result
        assert sim_weighted > 0.9
        assert sim_unweighted > 0.9

    def test_empty_states(self):
        sim = entity_state_similarity({}, {}, {})
        assert sim == 0.0


# ---------------------------------------------------------------------------
# Cerebellum motor program integration
# ---------------------------------------------------------------------------


class TestCerebellumMotorPrograms:
    def test_observe_and_crystallize(self):
        cb = Cerebellum()
        steps = [
            ("arm.shoulder", "motor", "rotate_angle", {"degrees": 45.0}),
            ("arm.elbow", "motor", "rotate_angle", {"degrees": 30.0}),
        ]
        assert cb.observe_action_sequence("reach", steps, True, 1.0) is None
        assert cb.observe_action_sequence("reach", steps, True, 1.0) is None
        prog = cb.observe_action_sequence("reach", steps, True, 1.0)
        assert prog is not None
        assert prog.goal_signature == "reach"

    def test_find_by_goal(self):
        cb = Cerebellum()
        cb.programs.register(
            MotorProgram(
                name="reach",
                goal_signature="reach_forward",
                steps=[MotorStep("arm.shoulder", "motor", "rotate_angle")],
                confidence=0.8,
            )
        )
        results = cb.find_programs_for_goal("reach_forward")
        assert len(results) == 1

    def test_find_by_entity(self):
        cb = Cerebellum()
        cb.programs.register(
            MotorProgram(
                name="sword_swing",
                goal_signature="attack",
                steps=[MotorStep("sword", "combat", "slash")],
                confidence=0.9,
            )
        )
        results = cb.find_programs_for_entity("sword")
        assert len(results) == 1

    def test_find_by_affordance(self):
        cb = Cerebellum()
        cb.programs.register(
            MotorProgram(
                name="sword_swing",
                goal_signature="attack",
                steps=[MotorStep("sword", "combat", "slash")],
                confidence=0.9,
            )
        )
        results = cb.find_programs_for_affordance("slash")
        assert len(results) == 1

    def test_find_related(self):
        cb = Cerebellum()
        cb.programs.register(
            MotorProgram(
                name="sword_slash",
                goal_signature="attack",
                steps=[MotorStep("sword", "combat", "slash")],
                confidence=0.9,
            )
        )
        cb.programs.register(
            MotorProgram(
                name="axe_slash",
                goal_signature="attack",
                steps=[MotorStep("axe", "combat", "slash")],
                confidence=0.7,
            )
        )
        # Query "slash" should find both (by affordance)
        results = cb.find_related_programs("slash")
        assert len(results) == 2
        # Query "attack" should find both (by goal)
        results = cb.find_related_programs("attack")
        assert len(results) == 2
        # Query "sword" should find one (by entity)
        results = cb.find_related_programs("sword")
        assert len(results) == 1

    def test_cleanup_program(self):
        cb = Cerebellum()
        cb.programs.register(
            MotorProgram(
                name="reach",
                goal_signature="reach_forward",
                steps=[MotorStep("arm.shoulder", "motor", "rotate_angle")],
            )
        )
        cb.cleanup_program("reach")
        assert cb.programs.get("reach") is None

    def test_persistence_includes_programs(self):
        cb = Cerebellum()
        cb.programs.register(
            MotorProgram(
                name="reach",
                goal_signature="reach_forward",
                steps=[MotorStep("arm.shoulder", "motor", "rotate_angle", {"degrees": 45})],
                confidence=0.8,
            )
        )
        state = cb.export_state()
        assert "programs" in state

        cb2 = Cerebellum()
        cb2.import_state(state)
        assert cb2.programs.get("reach") is not None
        assert cb2.programs.get("reach").confidence == 0.8

    def test_stats_includes_programs(self):
        cb = Cerebellum()
        cb.programs.register(
            MotorProgram(
                name="reach",
                goal_signature="reach",
                steps=[],
            )
        )
        stats = cb.stats()
        assert stats["programs"] == 1
