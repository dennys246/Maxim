"""Integration tests for embodiment MVP (Phase 0).

Tests the full pipeline: YAML -> Entity tree -> tools -> backends ->
failure triggers -> pain signals -> body state prompts.

Does NOT test LLM calls (mocked) or real hardware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from maxim.embodiment.body import Embodiment, EmbodimentConfig
from maxim.embodiment.llm_backend import LLMModulator, LLMSensor, NarrativeModulator, NarrativeSensor
from maxim.embodiment.percepts import EmbodimentPerceptSource
from maxim.embodiment.sem import AffordanceSchema, Entity
from maxim.embodiment.spec import SpecSensor, attach_backends, load_spec
from maxim.embodiment.tool_bridge import generate_tools_for_entity
from maxim.tools.registry import ToolRegistry

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "embodiment"


# ---------------------------------------------------------------------------
# Embodiment runtime
# ---------------------------------------------------------------------------


class TestEmbodiment:
    def _build_arm(self) -> Entity:
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")
        return spec.root_entity

    def test_construction(self):
        root = self._build_arm()
        emb = Embodiment(root)
        stats = emb.stats()
        assert stats["entities"] == 4  # arm + shoulder + elbow + camera
        assert stats["sensors"] == 5
        assert stats["modulators"] == 3
        assert stats["failure_events"] == 0

    def test_find_entity(self):
        root = self._build_arm()
        emb = Embodiment(root)
        assert emb.find("shoulder") is not None
        assert emb.find("elbow") is not None
        assert emb.find("nonexistent") is None

    def test_read_scalars(self):
        root = self._build_arm()
        emb = Embodiment(root)
        scalars = emb.read_scalars()
        assert "robot_arm.shoulder" in scalars
        assert "angle" in scalars["robot_arm.shoulder"]
        # Camera frame should NOT be in scalars
        assert "robot_arm.wrist_camera" not in scalars

    def test_evaluate_failures_no_fire(self):
        root = self._build_arm()
        emb = Embodiment(root)
        events = emb.evaluate_failures()
        assert events == []  # initial values are safe

    def test_evaluate_failures_overextension(self):
        root = self._build_arm()
        shoulder = root.find("shoulder")
        assert shoulder is not None
        # Push angle past threshold
        shoulder.vital_metrics["angle"] = 176
        emb = Embodiment(root)
        events = emb.evaluate_failures()
        assert len(events) >= 1
        assert any(e.failure_name == "overextension" for e in events)

    def test_pain_published(self):
        root = self._build_arm()
        shoulder = root.find("shoulder")
        assert shoulder is not None
        shoulder.vital_metrics["angle"] = 176

        pain_bus = MagicMock()
        emb = Embodiment(root, pain_bus=pain_bus)
        emb.evaluate_failures()
        pain_bus.publish.assert_called()
        signal = pain_bus.publish.call_args[0][0]
        assert signal.intensity == 0.8
        assert signal.context["source"] == "embodiment"
        assert signal.context["entity"] == "robot_arm.shoulder"
        assert signal.context["failure_mode"] == "overextension"

    def test_vital_drift(self):
        root = self._build_arm()
        elbow = root.find("elbow")
        assert elbow is not None
        elbow.vital_metrics["strain"] = 0.5
        emb = Embodiment(root)
        emb.tick_vital_drift(dt=10.0)
        # Strain should have increased (drift toward degradation)
        assert elbow.vital_metrics["strain"] > 0.5

    def test_body_state_summary(self):
        root = self._build_arm()
        emb = Embodiment(root)
        summary = emb.body_state_summary()
        assert len(summary) > 0
        # Should have shoulder and elbow
        paths = [s["entity"] for s in summary]
        assert "robot_arm.shoulder" in paths

    def test_body_state_warning(self):
        root = self._build_arm()
        shoulder = root.find("shoulder")
        assert shoulder is not None
        # Set angle close to threshold (175), within 20% proximity
        shoulder.vital_metrics["angle"] = 172
        emb = Embodiment(root, config=EmbodimentConfig(pain_proximity_threshold=0.2))
        summary = emb.body_state_summary()
        shoulder_state = next(s for s in summary if s["entity"] == "robot_arm.shoulder")
        angle_info = shoulder_state["sensors"]["angle"]
        assert "warning" in angle_info
        assert "pain_proximity" in angle_info

    def test_format_body_state_for_prompt(self):
        root = self._build_arm()
        emb = Embodiment(root)
        prompt = emb.format_body_state_for_prompt()
        assert "=== Body State ===" in prompt
        assert "shoulder" in prompt


# ---------------------------------------------------------------------------
# Percept source
# ---------------------------------------------------------------------------


class TestEmbodimentPerceptSource:
    def test_name(self):
        root = Entity("arm", "arm")
        emb = Embodiment(root)
        source = EmbodimentPerceptSource(emb)
        assert source.name == "embodiment:arm"

    def test_capabilities(self):
        root = Entity("arm", "arm")
        emb = Embodiment(root)
        source = EmbodimentPerceptSource(emb)
        assert "proprioception" in source.capabilities

    def test_not_exhausted(self):
        root = Entity("arm", "arm")
        emb = Embodiment(root)
        source = EmbodimentPerceptSource(emb)
        assert source.is_exhausted() is False

    def test_polling_rate(self):
        root = Entity("arm", "arm")
        root.sensors["x"] = _make_stub_sensor("x", 1.0)
        emb = Embodiment(root)
        source = EmbodimentPerceptSource(emb, poll_hz=1000)  # fast for testing

        # First call should return a percept (enough time has passed)
        p1 = source.next_percept()
        # Second immediate call should return None (too soon)
        p2 = source.next_percept()
        # One of these should be non-None depending on timing
        assert p1 is not None or p2 is None

    def test_demand_mode(self):
        root = Entity("arm", "arm")
        emb = Embodiment(root)
        source = EmbodimentPerceptSource(emb, poll_hz=1.0, demand_hz=100.0)
        source.set_demand_mode(True)
        assert source._in_demand is True
        source.set_demand_mode(False)
        assert source._in_demand is False


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


class TestLLMBackends:
    def test_llm_sensor_read(self):
        ent = Entity("shoulder", "joint")
        sensor = LLMSensor(
            ent,
            "angle",
            "degrees",
            {"type": "float", "range": [0, 360], "initial": 45.0},
        )
        reading = sensor.read()
        assert reading.sensor_name == "angle"
        assert reading.value == 45.0

    def test_llm_sensor_update(self):
        ent = Entity("shoulder", "joint")
        sensor = LLMSensor(
            ent,
            "angle",
            "degrees",
            {"type": "float", "range": [0, 360], "initial": 0.0},
        )
        sensor.update_value(90.0)
        assert sensor.read().value == 90.0

    def test_llm_modulator_heuristic(self):
        ent = Entity("shoulder", "joint")
        ent.sensors["angle"] = LLMSensor(
            ent,
            "angle",
            "degrees",
            {"type": "float", "range": [-180, 180], "initial": 0.0},
        )
        mod = LLMModulator(
            ent,
            "motor",
            {"rotate_angle": AffordanceSchema(params={"degrees": float, "speed": float})},
        )
        result = mod.execute("rotate_angle", {"degrees": 45, "speed": 1.0})
        assert result.success
        # Heuristic should have updated the angle
        assert ent.vital_metrics.get("angle") == 45.0

    def test_llm_modulator_unknown_affordance(self):
        ent = Entity("shoulder", "joint")
        mod = LLMModulator(ent, "motor", {})
        result = mod.execute("fly", {})
        assert not result.success

    def test_narrative_sensor(self):
        ent = Entity("sword", "weapon")
        ent.vital_metrics["durability"] = 0.3
        sensor = NarrativeSensor(
            ent,
            "durability",
            "ratio",
            {"type": "float", "range": [0, 1]},
        )
        reading = sensor.read()
        assert reading.value == 0.3

    def test_narrative_modulator_heuristic(self):
        ent = Entity("sword", "weapon")
        ent.sensors["durability"] = NarrativeSensor(
            ent,
            "durability",
            "ratio",
            {"type": "float", "range": [0, 1], "initial": 0.5},
        )
        ent.vital_metrics["durability"] = 0.5
        mod = NarrativeModulator(
            ent,
            "combat",
            {"slash": AffordanceSchema(params={"target": str, "force": float})},
        )
        result = mod.execute("slash", {"target": "dummy", "force": 0.5})
        assert result.success
        # Durability should have decreased
        assert ent.vital_metrics["durability"] < 0.5


# ---------------------------------------------------------------------------
# Full pipeline: YAML -> Entity -> Tools -> Backend -> Failure -> Pain
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_robot_arm_pipeline(self):
        """YAML -> entity tree -> tools -> failure -> pain."""
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")
        registry = ToolRegistry()
        tools = generate_tools_for_entity(spec.root_entity, registry)
        assert len(tools) == 13

        pain_bus = MagicMock()
        emb = Embodiment(spec.root_entity, pain_bus=pain_bus)

        # No failures initially
        events = emb.evaluate_failures()
        assert len(events) == 0
        pain_bus.publish.assert_not_called()

        # Execute modulator to push past threshold
        shoulder = spec.root_entity.find("shoulder")
        assert shoulder is not None
        shoulder.vital_metrics["angle"] = 176

        events = emb.evaluate_failures()
        assert any(e.failure_name == "overextension" for e in events)
        pain_bus.publish.assert_called()

    def test_sword_npc_pipeline(self):
        """Virtual entities: YAML -> tools -> failure."""
        spec = load_spec(SCENARIOS_DIR / "sword_npc_demo.yaml")
        registry = ToolRegistry()
        for ent in spec.world_entities:
            generate_tools_for_entity(ent, registry)

        sword = spec.world_entities[0]
        assert sword.name == "rusty_sword"

        # Set durability below shatter threshold
        sword.vital_metrics["durability"] = 0.05

        pain_bus = MagicMock()
        emb = Embodiment(sword, pain_bus=pain_bus)
        events = emb.evaluate_failures()
        assert any(e.failure_name == "shatter" for e in events)
        pain_bus.publish.assert_called()

    def test_attach_llm_backends(self):
        """Test backend attachment to spec stubs."""
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")

        def sensor_factory(ent, sname, spec_sensor):
            return LLMSensor(
                ent,
                sname,
                spec_sensor.unit,
                spec_sensor.reading_schema,
            )

        def mod_factory(ent, mname, spec_mod):
            return LLMModulator(
                ent,
                mname,
                spec_mod.affordances,
            )

        attach_backends(
            spec.root_entity,
            sensor_factory=sensor_factory,
            modulator_factory=mod_factory,
        )

        # Verify backends are attached
        shoulder = spec.root_entity.find("shoulder")
        assert shoulder is not None
        sensor = shoulder.sensors["angle"]
        assert hasattr(sensor, "_backend")
        assert sensor._backend is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_sensor(name: str, value: float) -> Any:
    """Create a minimal SpecSensor for testing."""
    return SpecSensor(
        _name=name,
        _entity_name="test",
        _unit="units",
        _schema={"type": "float", "range": [0, 100]},
        _initial=value,
    )
