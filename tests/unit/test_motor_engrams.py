"""Tests for motor engrams (Phase 1b).

Covers:
- Engram salience computation
- should_form_engram thresholds
- Engram formation decision logic
- Entity state snapshot filtering
- Graph node naming
- MotorEngram relevance scoring
"""

from __future__ import annotations

from maxim.embodiment.engrams import (
    EngramConfig,
    MotorEngram,
    compute_engram_salience,
    is_motor_engram_tool,
    program_graph_node,
    should_form_engram,
    snapshot_entity_states,
)


# ---------------------------------------------------------------------------
# Salience computation
# ---------------------------------------------------------------------------


class TestEngramSalience:
    def test_painful_outcome_high_salience(self):
        s = compute_engram_salience("NEGATIVE", 0.8, pain_intensity=0.7)
        assert s >= 0.7

    def test_surprising_outcome(self):
        s = compute_engram_salience("POSITIVE", 0.5, rpe_magnitude=0.6)
        assert s >= 0.6

    def test_low_confidence_novelty(self):
        s = compute_engram_salience("POSITIVE", 0.1)
        assert s > 0.3  # novelty from low confidence

    def test_routine_success_low_salience(self):
        s = compute_engram_salience("POSITIVE", 0.9)
        assert s < 0.2  # high confidence, no pain, no RPE

    def test_negative_boost(self):
        s_pos = compute_engram_salience("POSITIVE", 0.5, pain_intensity=0.4)
        s_neg = compute_engram_salience("NEGATIVE", 0.5, pain_intensity=0.4)
        assert s_neg > s_pos  # negative outcomes are more salient

    def test_capped_at_one(self):
        s = compute_engram_salience("NEGATIVE", 0.0, pain_intensity=1.0, rpe_magnitude=1.0)
        assert s <= 1.0


# ---------------------------------------------------------------------------
# Formation decision
# ---------------------------------------------------------------------------


class TestShouldFormEngram:
    def test_pain_triggers_formation(self):
        assert should_form_engram("NEGATIVE", pain_intensity=0.5) is True

    def test_rpe_triggers_formation(self):
        assert should_form_engram("POSITIVE", rpe_magnitude=0.5) is True

    def test_novelty_triggers_formation(self):
        assert should_form_engram("POSITIVE", novelty=0.8) is True

    def test_low_confidence_triggers_formation(self):
        assert should_form_engram("POSITIVE", program_confidence=0.1) is True

    def test_routine_success_no_formation(self):
        assert (
            should_form_engram(
                "POSITIVE",
                pain_intensity=0.0,
                rpe_magnitude=0.0,
                novelty=0.1,
                program_confidence=0.9,
            )
            is False
        )

    def test_custom_config(self):
        cfg = EngramConfig(pain_threshold=0.8, rpe_threshold=0.8, novelty_threshold=0.95)
        # Pain below custom threshold + high confidence → no formation
        assert should_form_engram("NEGATIVE", pain_intensity=0.5, program_confidence=0.9, config=cfg) is False
        # Pain above custom threshold → formation
        assert should_form_engram("NEGATIVE", pain_intensity=0.9, program_confidence=0.9, config=cfg) is True


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshotEntityStates:
    def test_scalar_values(self):
        readings = {
            "arm.shoulder": {"angle": 90.0, "temp": 30.0},
            "arm.elbow": {"angle": 45.0},
        }
        result = snapshot_entity_states(readings)
        assert result["arm.shoulder"]["angle"] == 90.0
        assert result["arm.elbow"]["angle"] == 45.0

    def test_excludes_non_scalar(self):
        readings = {
            "head.camera": {"frame": [1, 2, 3], "fps": 30.0},
        }
        result = snapshot_entity_states(readings)
        assert "fps" in result.get("head.camera", {})
        assert "frame" not in result.get("head.camera", {})

    def test_handles_sensor_reading_objects(self):
        class FakeReading:
            value = 42.0

        readings = {"arm": {"angle": FakeReading()}}
        result = snapshot_entity_states(readings)
        assert result["arm"]["angle"] == 42.0

    def test_empty(self):
        result = snapshot_entity_states({})
        assert result == {}


# ---------------------------------------------------------------------------
# Graph node naming
# ---------------------------------------------------------------------------


class TestGraphNodeNaming:
    def test_program_node(self):
        assert program_graph_node("reach_forward") == "cerebellum:program:reach_forward"

    def test_is_motor_engram_tool(self):
        assert is_motor_engram_tool("motor_program:reach_forward") is True
        assert is_motor_engram_tool("respond") is False
        assert is_motor_engram_tool("read_shoulder_angle") is False


# ---------------------------------------------------------------------------
# MotorEngram
# ---------------------------------------------------------------------------


class TestMotorEngram:
    def test_relevance(self):
        e = MotorEngram(
            memory_id="abc",
            program_name="reach",
            activation=0.8,
            context_similarity=0.9,
        )
        assert abs(e.relevance - 0.72) < 0.01

    def test_low_relevance(self):
        e = MotorEngram(
            memory_id="abc",
            program_name="reach",
            activation=0.2,
            context_similarity=0.3,
        )
        assert e.relevance < 0.1
