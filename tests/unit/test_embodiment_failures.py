"""Tests for composed failures + persistence (Phase 2).

Covers:
- Compound failure triggers (all mode)
- Composed failure modes (composes field)
- Persistent failures with recovery
- Failure state export/import
- Save/load to JSON file
- ToolPainBridge embodiment pain handling
- Active failure tracking in stats
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from maxim.embodiment.body import Embodiment
from maxim.embodiment.sem import Entity, FailureMode, FailureTrigger

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "embodiment"


# ---------------------------------------------------------------------------
# Composed failures
# ---------------------------------------------------------------------------


class TestComposedFailures:
    def test_compound_trigger_all(self):
        """Tennis elbow: both strain AND fatigue must exceed thresholds."""
        fm = FailureMode(
            name="tennis_elbow",
            composes=["strain", "fatigue"],
            triggers=[
                FailureTrigger("strain", ">", 0.6, 0.5),
                FailureTrigger("fatigue", ">", 0.5, 0.5),
            ],
            trigger_mode="all",
            pain_intensity=0.5,
        )
        # Only strain high → no fire
        assert fm.evaluate({"strain": 0.7, "fatigue": 0.3}) is False
        # Only fatigue high → no fire
        assert fm.evaluate({"strain": 0.4, "fatigue": 0.6}) is False
        # Both high → fire
        assert fm.evaluate({"strain": 0.7, "fatigue": 0.6}) is True

    def test_composed_labels(self):
        fm = FailureMode(
            name="tennis_elbow",
            composes=["strain", "fatigue"],
            triggers=[FailureTrigger("strain", ">", 0.6)],
        )
        assert fm.composes == ["strain", "fatigue"]

    def test_persistent_with_recovery(self):
        fm = FailureMode(
            name="overheating",
            triggers=[FailureTrigger("temperature", ">", 70, 0.6)],
            persistent=True,
            recovery_condition=FailureTrigger("temperature", "<", 40),
            pain_intensity=0.6,
        )
        # Fire
        assert fm.evaluate({"temperature": 75}) is True
        assert fm.active is True

        # Still active — temperature dropped but not below recovery threshold
        assert fm.evaluate({"temperature": 55}) is True
        assert fm.active is True

        # Recovered
        assert fm.evaluate({"temperature": 35}) is False
        assert fm.active is False

    def test_embodiment_evaluates_composed(self):
        """Composed failure fires through the Embodiment runtime."""
        ent = Entity("elbow", "joint")
        ent.sensors["strain"] = _stub_sensor("strain", 0.7)
        ent.sensors["fatigue"] = _stub_sensor("fatigue", 0.6)
        ent.vital_metrics["strain"] = 0.7
        ent.vital_metrics["fatigue"] = 0.6
        ent.failure_modes.append(
            FailureMode(
                name="tennis_elbow",
                composes=["strain", "fatigue"],
                triggers=[
                    FailureTrigger("strain", ">", 0.6, 0.5),
                    FailureTrigger("fatigue", ">", 0.5, 0.5),
                ],
                trigger_mode="all",
                pain_intensity=0.5,
            )
        )

        pain_bus = MagicMock()
        emb = Embodiment(ent, pain_bus=pain_bus)
        events = emb.evaluate_failures()
        assert any(e.failure_name == "tennis_elbow" for e in events)
        pain_bus.publish.assert_called()

        # Verify composition metadata in pain signal
        signal = pain_bus.publish.call_args[0][0]
        assert signal.context["failure_mode"] == "tennis_elbow"
        assert signal.context["composes"] == ["strain", "fatigue"]


# ---------------------------------------------------------------------------
# Failure persistence
# ---------------------------------------------------------------------------


class TestFailurePersistence:
    def test_export_active_failures(self):
        ent = Entity("elbow", "joint")
        ent.vital_metrics["strain"] = 0.7
        fm = FailureMode(
            name="strain_overload",
            triggers=[FailureTrigger("strain", ">", 0.6, 0.6)],
            persistent=True,
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.evaluate_failures()  # Fire the failure

        state = emb.export_failure_state()
        assert len(state["active_failures"]) == 1
        assert state["active_failures"][0]["failure_name"] == "strain_overload"
        assert state["active_failures"][0]["persistent"] is True

    def test_export_no_active(self):
        ent = Entity("elbow", "joint")
        ent.vital_metrics["strain"] = 0.3
        fm = FailureMode(
            name="strain_overload",
            triggers=[FailureTrigger("strain", ">", 0.6, 0.6)],
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.evaluate_failures()

        state = emb.export_failure_state()
        assert len(state["active_failures"]) == 0

    def test_import_restores_active(self):
        ent = Entity("elbow", "joint")
        fm = FailureMode(
            name="overheating",
            triggers=[FailureTrigger("temperature", ">", 70, 0.6)],
            persistent=True,
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.import_failure_state(
            {
                "active_failures": [
                    {
                        "entity_path": "elbow",
                        "failure_name": "overheating",
                        "persistent": True,
                        "last_fired": 1000.0,
                    },
                ],
            }
        )

        assert fm.active is True
        assert fm.last_fired == 1000.0

    def test_save_load_roundtrip(self, tmp_path):
        ent = Entity("shoulder", "joint")
        ent.vital_metrics["angle"] = 180
        fm = FailureMode(
            name="overextension",
            triggers=[FailureTrigger("angle", ">", 175, 0.8)],
            persistent=True,
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.evaluate_failures()

        path = str(tmp_path / "failures.json")
        emb.save_failures(path)

        # New embodiment, load state
        ent2 = Entity("shoulder", "joint")
        fm2 = FailureMode(
            name="overextension",
            triggers=[FailureTrigger("angle", ">", 175, 0.8)],
            persistent=True,
        )
        ent2.failure_modes.append(fm2)
        emb2 = Embodiment(ent2)
        loaded = emb2.load_failures(path)
        assert loaded is True
        assert fm2.active is True

    def test_failure_history_in_export(self):
        ent = Entity("shoulder", "joint")
        ent.vital_metrics["angle"] = 180
        fm = FailureMode(
            name="overextension",
            triggers=[FailureTrigger("angle", ">", 175, 0.8)],
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.evaluate_failures()

        state = emb.export_failure_state()
        assert len(state["failure_history"]) == 1
        assert state["failure_history"][0]["failure_name"] == "overextension"

    def test_active_failures_in_stats(self):
        ent = Entity("shoulder", "joint")
        ent.vital_metrics["angle"] = 180
        fm = FailureMode(
            name="overextension",
            triggers=[FailureTrigger("angle", ">", 175, 0.8)],
            persistent=True,
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.evaluate_failures()
        stats = emb.stats()
        assert stats["active_failures"] == 1


# ---------------------------------------------------------------------------
# ToolPainBridge embodiment integration
# ---------------------------------------------------------------------------


class TestToolPainBridgeEmbodiment:
    def test_embodiment_pain_routed(self):
        """Embodiment pain signals are routed to _on_embodiment_pain."""
        from maxim.proprioception.pain import PainSignal, PainType

        nac = MagicMock()
        nac.record_outcome_full = MagicMock(return_value=[])

        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        bridge = ToolPainBridge(nac=nac)

        signal = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.8,
            timestamp=1000.0,
            context={
                "source": "embodiment",
                "entity": "arm.shoulder",
                "entity_type": "joint",
                "failure_mode": "overextension",
                "composes": [],
                "sensor_readings": {"angle": 176},
            },
        )
        bridge._on_pain(signal)

        # Should have called record_outcome_full on NAc
        nac.record_outcome_full.assert_called_once()
        call_kwargs = nac.record_outcome_full.call_args
        assert call_kwargs[1]["outcome_type"] == "embodiment_failure"
        assert "arm.shoulder" in call_kwargs[1]["outcome_signature"]
        assert call_kwargs[1]["context"]["failure_mode"] == "overextension"

    def test_composed_metadata_passed(self):
        """Composition metadata flows through to NAc."""
        from maxim.proprioception.pain import PainSignal, PainType

        nac = MagicMock()
        nac.record_outcome_full = MagicMock(return_value=[])

        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        bridge = ToolPainBridge(nac=nac)

        signal = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.5,
            timestamp=1000.0,
            context={
                "source": "embodiment",
                "entity": "arm.elbow",
                "entity_type": "joint",
                "failure_mode": "tennis_elbow",
                "composes": ["strain", "fatigue"],
                "sensor_readings": {"strain": 0.7, "fatigue": 0.6},
            },
        )
        bridge._on_pain(signal)

        call_kwargs = nac.record_outcome_full.call_args
        assert call_kwargs[1]["context"]["composes"] == ["strain", "fatigue"]

    def test_tool_pain_still_works(self):
        """Normal tool pain signals still route correctly."""
        from maxim.proprioception.pain import PainSignal, PainType

        nac = MagicMock()
        nac.record_outcome = MagicMock(return_value=[])

        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        bridge = ToolPainBridge(nac=nac)

        # Non-embodiment signal should NOT call record_outcome_full
        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.5,
            timestamp=1000.0,
            context={"tool_name": "read_file", "invocation_id": "123"},
        )
        bridge._on_pain(signal)
        nac.record_outcome_full.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_sensor(name: str, value: float):
    """Create a minimal SpecSensor for testing."""
    from maxim.embodiment.spec import SpecSensor

    return SpecSensor(
        _name=name,
        _entity_name="test",
        _unit="ratio",
        _schema={"type": "float", "range": [0, 1]},
        _initial=value,
    )
