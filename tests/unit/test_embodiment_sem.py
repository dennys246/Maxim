"""Tests for the SEM protocol, YAML loading, and auto-tool generation.

Covers:
- Entity tree construction and traversal
- Sensor/Modulator protocol compliance
- FailureTrigger evaluation
- FailureMode composition (any/all)
- YAML spec loading (robot arm, sword, NPC)
- Auto-tool generation and naming
- Tool name collision detection and progressive prefixing
- Entity.read_scalar_sensors() filtering
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from maxim.embodiment.sem import (
    BASE_FAILURE_MODES,
    AffordanceSchema,
    Entity,
    FailureMode,
    FailureTrigger,
    Modulator,
    ModulatorResult,
    Sensor,
    SensorReading,
)
from maxim.embodiment.spec import load_spec
from maxim.embodiment.tool_bridge import (
    EntitySenseTool,
    ModulatorAffordanceTool,
    SensorReadTool,
    generate_tools_for_entity,
)
from maxim.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "embodiment"


class StubSensor:
    """Minimal Sensor implementation for testing."""

    def __init__(self, name: str, unit: str = "degrees", value: float = 0.0):
        self._name = name
        self._unit = unit
        self._value = value
        self._schema: dict[str, Any] = {"type": "float", "range": [0, 360]}

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
        return SensorReading(
            sensor_name=self._name,
            entity_name="test",
            value=self._value,
            unit=self._unit,
            timestamp=time.time(),
        )


class StubModulator:
    """Minimal Modulator implementation for testing."""

    def __init__(self, name: str, affordances: dict[str, AffordanceSchema] | None = None):
        self._name = name
        self._affordances = affordances or {
            "rotate_angle": AffordanceSchema(
                params={"degrees": float, "speed": float},
                description="Rotate the joint",
            ),
        }
        self.calls: list[tuple[str, dict]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def affordances(self) -> dict[str, AffordanceSchema]:
        return self._affordances

    def execute(self, affordance: str, params: dict[str, Any]) -> ModulatorResult:
        self.calls.append((affordance, params))
        return ModulatorResult(
            success=True,
            modulator_name=self._name,
            entity_name="test",
            affordance=affordance,
            params=params,
        )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_stub_sensor_is_sensor(self):
        s = StubSensor("angle")
        assert isinstance(s, Sensor)

    def test_stub_modulator_is_modulator(self):
        m = StubModulator("motor")
        assert isinstance(m, Modulator)

    def test_sensor_reading_frozen(self):
        r = SensorReading("angle", "shoulder", 45.0, "degrees", time.time())
        with pytest.raises(AttributeError):
            r.value = 90.0  # type: ignore[misc]

    def test_modulator_result_frozen(self):
        r = ModulatorResult(True, "motor", "shoulder", "rotate", {})
        with pytest.raises(AttributeError):
            r.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Entity tree
# ---------------------------------------------------------------------------


class TestEntity:
    def test_basic_construction(self):
        e = Entity("shoulder", "joint")
        assert e.name == "shoulder"
        assert e.entity_type == "joint"
        assert e.parent is None
        assert e.children == []

    def test_parent_child(self):
        parent = Entity("arm", "arm")
        child = Entity("elbow", "joint", parent=parent)
        assert child.parent is parent
        assert child in parent.children

    def test_full_path(self):
        root = Entity("arm", "arm")
        shoulder = Entity("shoulder", "joint", parent=root)
        motor = Entity("motor", "actuator", parent=shoulder)
        assert motor.full_path == "arm.shoulder.motor"

    def test_walk(self):
        root = Entity("arm", "arm")
        Entity("shoulder", "joint", parent=root)
        Entity("elbow", "joint", parent=root)
        names = [e.name for e in root.walk()]
        assert names == ["arm", "shoulder", "elbow"]

    def test_find(self):
        root = Entity("arm", "arm")
        shoulder = Entity("shoulder", "joint", parent=root)
        Entity("motor", "actuator", parent=shoulder)
        assert root.find("shoulder") is shoulder
        assert root.find("shoulder.motor") is not None
        assert root.find("nonexistent") is None

    def test_read_all_sensors(self):
        e = Entity("shoulder", "joint")
        e.sensors["angle"] = StubSensor("angle", value=45.0)
        e.sensors["temp"] = StubSensor("temp", unit="celsius", value=25.0)
        readings = e.read_all_sensors()
        assert len(readings) == 2
        assert readings["angle"].value == 45.0
        assert readings["temp"].value == 25.0

    def test_read_scalar_sensors_excludes_frames(self):
        e = Entity("head", "head_unit")
        e.sensors["angle"] = StubSensor("angle", value=10.0)
        frame_sensor = StubSensor("frame", unit="rgb_frame", value=None)
        frame_sensor._schema = {"type": "ndarray", "shape": [480, 640, 3]}
        e.sensors["frame"] = frame_sensor
        scalars = e.read_scalar_sensors()
        assert "angle" in scalars
        assert "frame" not in scalars

    def test_repr(self):
        e = Entity("shoulder", "joint")
        e.sensors["angle"] = StubSensor("angle")
        r = repr(e)
        assert "shoulder" in r
        assert "joint" in r


# ---------------------------------------------------------------------------
# Failure triggers
# ---------------------------------------------------------------------------


class TestFailureTrigger:
    def test_greater_than(self):
        t = FailureTrigger(field="angle", op=">", value=175, pain=0.8)
        assert t.evaluate(176) is True
        assert t.evaluate(175) is False
        assert t.evaluate(170) is False

    def test_less_than(self):
        t = FailureTrigger(field="durability", op="<", value=0.1, pain=0.6)
        assert t.evaluate(0.05) is True
        assert t.evaluate(0.1) is False
        assert t.evaluate(0.5) is False

    def test_greater_equal(self):
        t = FailureTrigger(field="strain", op=">=", value=0.8, pain=0.5)
        assert t.evaluate(0.8) is True
        assert t.evaluate(0.79) is False

    def test_equal(self):
        t = FailureTrigger(field="count", op="==", value=5, pain=0.3)
        assert t.evaluate(5) is True
        assert t.evaluate(4) is False

    def test_invalid_op(self):
        t = FailureTrigger(field="x", op="!=", value=1, pain=0.1)
        assert t.evaluate(2) is False  # unknown op never fires


class TestFailureMode:
    def test_any_mode(self):
        fm = FailureMode(
            name="overextension",
            triggers=[
                FailureTrigger("angle", ">", 175, 0.8),
                FailureTrigger("strain", ">", 0.8, 0.6),
            ],
            trigger_mode="any",
        )
        assert fm.evaluate({"angle": 180, "strain": 0.5}) is True
        assert fm.evaluate({"angle": 170, "strain": 0.9}) is True
        assert fm.evaluate({"angle": 170, "strain": 0.5}) is False

    def test_all_mode(self):
        fm = FailureMode(
            name="tennis_elbow",
            triggers=[
                FailureTrigger("strain", ">", 0.6, 0.5),
                FailureTrigger("fatigue", ">", 0.5, 0.5),
            ],
            trigger_mode="all",
        )
        assert fm.evaluate({"strain": 0.7, "fatigue": 0.6}) is True
        assert fm.evaluate({"strain": 0.7, "fatigue": 0.4}) is False

    def test_persistent_mode(self):
        fm = FailureMode(
            name="overheating",
            triggers=[FailureTrigger("temperature", ">", 70, 0.6)],
            persistent=True,
            recovery_condition=FailureTrigger("temperature", "<", 40),
        )
        # Fire
        assert fm.evaluate({"temperature": 75}) is True
        assert fm.active is True
        # Still active even below trigger
        assert fm.evaluate({"temperature": 65}) is True
        # Recover
        assert fm.evaluate({"temperature": 35}) is False
        assert fm.active is False

    def test_base_modes_vocabulary(self):
        assert "overextension" in BASE_FAILURE_MODES
        assert "overheating" in BASE_FAILURE_MODES
        assert "strain" in BASE_FAILURE_MODES
        assert "fatigue" in BASE_FAILURE_MODES
        assert "impact" in BASE_FAILURE_MODES
        assert "exhaustion" in BASE_FAILURE_MODES
        assert len(BASE_FAILURE_MODES) == 6


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestSpecLoading:
    def test_load_robot_arm(self):
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")
        assert spec.name == "robot_arm"
        root = spec.root_entity
        assert root.entity_type == "arm"
        assert len(root.children) == 3

        shoulder = root.find("shoulder")
        assert shoulder is not None
        assert "angle" in shoulder.sensors
        assert "temperature" in shoulder.sensors
        assert "motor" in shoulder.modulators
        assert len(shoulder.failure_modes) == 3

        elbow = root.find("elbow")
        assert elbow is not None
        assert "strain" in elbow.sensors
        assert len(elbow.failure_modes) == 2

        camera = root.find("wrist_camera")
        assert camera is not None
        assert "frame" in camera.sensors
        assert "lifecycle" in camera.modulators

    def test_load_sword_npc(self):
        spec = load_spec(SCENARIOS_DIR / "sword_npc_demo.yaml")
        assert len(spec.world_entities) == 2

        sword = spec.world_entities[0]
        assert sword.name == "rusty_sword"
        assert sword.entity_type == "weapon"
        assert "durability" in sword.sensors
        assert "combat" in sword.modulators
        assert "maintenance" in sword.modulators
        assert any(fm.name == "shatter" for fm in sword.failure_modes)

        ferryman = spec.world_entities[1]
        assert ferryman.name == "grim_ferryman"
        assert ferryman.entity_type == "npc"
        assert "trust" in ferryman.sensors
        assert "social" in ferryman.modulators
        assert "physical" in ferryman.modulators

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_spec("/nonexistent/path.yaml")

    def test_load_baseline(self):
        spec = load_spec(SCENARIOS_DIR / "embodiment_baseline.yaml")
        assert spec.root_entity is not None
        shoulder = spec.root_entity.find("shoulder")
        assert shoulder is not None

    def test_initial_vital_metrics(self):
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")
        shoulder = spec.root_entity.find("shoulder")
        assert shoulder is not None
        assert shoulder.vital_metrics.get("angle") == 0
        assert shoulder.vital_metrics.get("temperature") == 25

    def test_sensor_reading_from_spec(self):
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")
        shoulder = spec.root_entity.find("shoulder")
        assert shoulder is not None
        reading = shoulder.sensors["angle"].read()
        assert reading.sensor_name == "angle"
        assert reading.unit == "degrees"
        assert reading.value == 0  # initial value

    def test_modulator_execution_from_spec(self):
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")
        shoulder = spec.root_entity.find("shoulder")
        assert shoulder is not None
        result = shoulder.modulators["motor"].execute(
            "rotate_angle",
            {"degrees": 45, "speed": 1.0},
        )
        assert result.success is True
        assert result.affordance == "rotate_angle"

    def test_unknown_affordance_fails(self):
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")
        shoulder = spec.root_entity.find("shoulder")
        assert shoulder is not None
        result = shoulder.modulators["motor"].execute("fly", {})
        assert result.success is False
        assert "Unknown affordance" in (result.error or "")


# ---------------------------------------------------------------------------
# Tool generation
# ---------------------------------------------------------------------------


class TestToolGeneration:
    def test_robot_arm_tools(self):
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")
        registry = ToolRegistry()
        tools = generate_tools_for_entity(spec.root_entity, registry)

        tool_names = [t.name for t in tools]

        # Shoulder: sense + read_angle + read_temperature + rotate_angle + brake
        assert "sense_shoulder" in tool_names
        assert "read_shoulder_angle" in tool_names
        assert "read_shoulder_temperature" in tool_names
        assert "shoulder_rotate_angle" in tool_names
        assert "shoulder_brake" in tool_names

        # Elbow: sense + read_angle + read_strain + rotate_angle
        assert "sense_elbow" in tool_names
        assert "read_elbow_angle" in tool_names
        assert "read_elbow_strain" in tool_names
        assert "elbow_rotate_angle" in tool_names

        # Camera: sense + read_frame + capture_frame + restart
        assert "sense_wrist_camera" in tool_names
        assert "read_wrist_camera_frame" in tool_names
        assert "wrist_camera_capture_frame" in tool_names
        assert "wrist_camera_restart" in tool_names

    def test_tool_count(self):
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")
        registry = ToolRegistry()
        tools = generate_tools_for_entity(spec.root_entity, registry)
        # 3 entities with sensors -> 3 sense tools
        # shoulder: 2 sensors + 2 affordances = 4
        # elbow: 2 sensors + 1 affordance = 3
        # camera: 1 sensor + 2 affordances = 3
        # arm root has no sensors/modulators = 0
        # Total: 3 (sense) + 5 (reads) + 5 (affordances) = 13
        assert len(tools) == 13

    def test_sword_npc_tools(self):
        spec = load_spec(SCENARIOS_DIR / "sword_npc_demo.yaml")
        registry = ToolRegistry()

        all_tools = []
        for ent in spec.world_entities:
            all_tools.extend(generate_tools_for_entity(ent, registry))

        tool_names = [t.name for t in all_tools]
        assert "rusty_sword_slash" in tool_names
        assert "rusty_sword_parry" in tool_names
        assert "rusty_sword_sharpen" in tool_names
        assert "read_rusty_sword_durability" in tool_names
        assert "grim_ferryman_speak" in tool_names
        assert "grim_ferryman_threaten" in tool_names
        assert "grim_ferryman_punch" in tool_names
        assert "read_grim_ferryman_trust" in tool_names

    def test_collision_detection(self):
        """Two entities with same name should get prefixed."""
        registry = ToolRegistry()

        robot1 = Entity("robot1", "robot")
        shoulder1 = Entity("shoulder", "joint", parent=robot1)
        shoulder1.sensors["angle"] = StubSensor("angle")
        shoulder1.modulators["motor"] = StubModulator("motor")

        robot2 = Entity("robot2", "robot")
        shoulder2 = Entity("shoulder", "joint", parent=robot2)
        shoulder2.sensors["angle"] = StubSensor("angle")
        shoulder2.modulators["motor"] = StubModulator("motor")

        tools1 = generate_tools_for_entity(robot1, registry)
        tools2 = generate_tools_for_entity(robot2, registry)

        names1 = {t.name for t in tools1}
        names2 = {t.name for t in tools2}

        # First robot gets short names
        assert "shoulder_rotate_angle" in names1
        assert "read_shoulder_angle" in names1

        # Second robot gets prefixed names
        assert "robot2_shoulder_rotate_angle" in names2
        assert "robot2_read_shoulder_angle" in names2

        # No collisions
        assert names1.isdisjoint(names2)

    def test_sensor_read_tool_execution(self):
        e = Entity("shoulder", "joint")
        sensor = StubSensor("angle", value=45.0)
        e.sensors["angle"] = sensor
        tool = SensorReadTool(e, sensor, "read_shoulder_angle")
        result = tool.run()
        assert result.success
        assert result.output["value"] == 45.0
        assert result.output["unit"] == "degrees"

    def test_modulator_tool_execution(self):
        e = Entity("shoulder", "joint")
        mod = StubModulator("motor")
        e.modulators["motor"] = mod
        schema = AffordanceSchema(params={"degrees": float, "speed": float})
        tool = ModulatorAffordanceTool(
            e,
            mod,
            "rotate_angle",
            schema,
            "shoulder_rotate_angle",
        )
        result = tool.run(degrees=45.0, speed=1.0)
        assert result.success
        assert mod.calls == [("rotate_angle", {"degrees": 45.0, "speed": 1.0})]

    def test_entity_sense_tool_execution(self):
        e = Entity("shoulder", "joint")
        e.sensors["angle"] = StubSensor("angle", value=90.0)
        e.sensors["temp"] = StubSensor("temp", unit="celsius", value=30.0)
        tool = EntitySenseTool(e, "sense_shoulder")
        result = tool.run()
        assert result.success
        assert "angle" in result.output
        assert result.output["angle"]["value"] == 90.0

    def test_tools_registered_in_registry(self):
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")
        registry = ToolRegistry()
        generate_tools_for_entity(spec.root_entity, registry)
        assert "shoulder_rotate_angle" in registry.list()
        assert registry.get("shoulder_rotate_angle") is not None
