"""Tests for the v1.0 multi-robot extensibility surface.

These tests pin down the additive changes that opened up Maxim's hardware
layer to non-Reachy robots:

1. ``MotionTarget.extras`` — robot-specific joints (Reachy antennas, Atlas
   arms, Spot legs) live in a generic dict instead of as named fields on
   the abstract dataclass.
2. ``RobotCapabilities.custom`` — robot-specific capability strings that
   don't fit the generic ``MotionCapability`` / ``StreamCapability`` enums.
3. ``maxim.robots`` entry-point discovery — third-party plugins can
   register controllers without modifying core code.
4. Reachy Mini SEM asset (``bodies/reachy_mini.yaml``) loads through the
   component registry.
5. Reachy controller registration is now optional (peers + non-Reachy
   environments can skip the SDK install).
"""

from __future__ import annotations


# ─── MotionTarget.extras ──────────────────────────────────────────────────


class TestMotionTargetExtras:
    def test_extras_defaults_to_empty_dict(self):
        from maxim.hardware.controller import MotionTarget

        target = MotionTarget()
        assert target.extras == {}

    def test_extras_holds_reachy_antennas(self):
        from maxim.hardware.controller import MotionTarget

        target = MotionTarget(
            head_pitch=0.2,
            extras={"antenna_left": 0.5, "antenna_right": -0.5},
        )
        assert target.extras["antenna_left"] == 0.5
        assert target.extras["antenna_right"] == -0.5
        assert target.head_pitch == 0.2

    def test_extras_holds_atlas_arm_joints(self):
        from maxim.hardware.controller import MotionTarget

        target = MotionTarget(
            extras={
                "left_arm_yaw": 1.0,
                "left_arm_pitch": -0.3,
                "right_arm_yaw": -1.0,
                "right_arm_pitch": 0.3,
                "gripper_left": 0.7,
                "gripper_right": 0.7,
            },
        )
        assert len(target.extras) == 6
        assert target.extras["gripper_left"] == 0.7

    def test_no_antenna_fields_on_abstract_class(self):
        """Antennas were promoted out of MotionTarget — they live in extras now."""
        from dataclasses import fields
        from maxim.hardware.controller import MotionTarget

        field_names = {f.name for f in fields(MotionTarget)}
        assert "antenna_left" not in field_names
        assert "antenna_right" not in field_names
        # Generic fields are still there
        assert "head_yaw" in field_names
        assert "extras" in field_names

    def test_extras_independent_per_instance(self):
        """The default_factory keeps extras dicts isolated between instances."""
        from maxim.hardware.controller import MotionTarget

        a = MotionTarget()
        b = MotionTarget()
        a.extras["foo"] = 1.0
        assert "foo" not in b.extras


# ─── RobotCapabilities.custom ─────────────────────────────────────────────


class TestRobotCapabilitiesCustom:
    def test_custom_defaults_to_empty(self):
        from maxim.hardware.capabilities import RobotCapabilities

        caps = RobotCapabilities(robot_type="test", robot_id="t1")
        assert caps.custom == frozenset()

    def test_has_custom_returns_false_when_absent(self):
        from maxim.hardware.capabilities import RobotCapabilities

        caps = RobotCapabilities(robot_type="test", robot_id="t1")
        assert caps.has_custom("antenna_left") is False

    def test_custom_capabilities_advertised(self):
        from maxim.hardware.capabilities import RobotCapabilities

        caps = RobotCapabilities(
            robot_type="reachy_mini",
            robot_id="r1",
            custom=frozenset({"antenna_left", "antenna_right"}),
        )
        assert caps.has_custom("antenna_left") is True
        assert caps.has_custom("antenna_right") is True
        assert caps.has_custom("foot_pressure") is False

    def test_reachy_mini_default_caps_use_custom(self):
        """Reachy's antennas migrated from MotionCapability.ANTENNA → custom."""
        from maxim.hardware.capabilities import REACHY_MINI_CAPABILITIES, MotionCapability

        # The legacy ANTENNA enum value is gone
        assert not hasattr(MotionCapability, "ANTENNA")
        # But Reachy still advertises its antennas via custom
        assert REACHY_MINI_CAPABILITIES.has_custom("antenna_left")
        assert REACHY_MINI_CAPABILITIES.has_custom("antenna_right")

    def test_motion_capability_has_arms_and_legs(self):
        """Generic enums for humanoid + quadruped joints."""
        from maxim.hardware.capabilities import MotionCapability

        assert hasattr(MotionCapability, "ARMS")
        assert hasattr(MotionCapability, "GRIPPER")
        assert hasattr(MotionCapability, "LEGS")
        assert hasattr(MotionCapability, "WHEELS")


# ─── Entry-point discovery ────────────────────────────────────────────────


class TestEntryPointDiscovery:
    """The RobotRegistry singleton auto-discovers ``maxim.robots`` plugins."""

    def test_registry_singleton_exposes_discovery_method(self):
        from maxim.hardware.registry import RobotRegistry

        registry = RobotRegistry()
        # The discovery method exists and is callable
        assert hasattr(registry, "_discover_entry_point_plugins")
        assert callable(registry._discover_entry_point_plugins)

    def test_discovery_handles_no_plugins_silently(self):
        """In a fresh test env, no plugins registered — must not crash."""
        from maxim.hardware.registry import RobotRegistry

        registry = RobotRegistry()
        # Calling discovery a second time is idempotent (no exceptions)
        registry._discover_entry_point_plugins()

    def test_discovery_skips_non_controller_subclasses(self, caplog, monkeypatch):
        """If an entry point yields the wrong type, log warning + skip."""
        import logging
        from importlib.metadata import EntryPoint
        from maxim.hardware.registry import RobotRegistry

        # Build a fake entry point that resolves to ``int`` (not a RobotController)
        fake_ep = EntryPoint(name="bogus", value="builtins:int", group="maxim.robots")

        def fake_eps(group=None, **_kw):
            class _Fake:
                def __iter__(self):
                    return iter([fake_ep])

            return _Fake()

        monkeypatch.setattr("importlib.metadata.entry_points", fake_eps)
        registry = RobotRegistry()
        with caplog.at_level(logging.WARNING, logger="maxim.hardware.registry"):
            registry._discover_entry_point_plugins()
        assert any("did not yield a RobotController subclass" in r.message for r in caplog.records)
        # ``bogus`` should NOT have been registered
        assert "bogus" not in registry.get_controller_types()


# ─── Reachy Mini SEM asset ────────────────────────────────────────────────


class TestReachyMiniSemAsset:
    """The Reachy Mini SEM template ships as a seed component."""

    def test_loads_via_component_registry(self):
        from maxim.embodiment.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        spec = registry.get("bodies/reachy_mini")
        assert spec is not None
        assert spec["component"]["name"] == "reachy_mini"
        assert "robot" in spec["component"]["tags"]
        assert "embodied" in spec["component"]["tags"]

    def test_has_expected_sensors(self):
        from maxim.embodiment.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        spec = registry.get("bodies/reachy_mini")
        sensors = spec["entity"]["sensors"]
        for required in (
            "head_yaw",
            "head_pitch",
            "head_roll",
            "body_yaw",
            "antenna_left",
            "antenna_right",
            "battery",
            "motor_temperature",
        ):
            assert required in sensors, f"missing sensor: {required}"

    def test_has_expected_modulators(self):
        from maxim.embodiment.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        spec = registry.get("bodies/reachy_mini")
        modulators = spec["entity"]["modulators"]
        assert "motion" in modulators
        assert "expression" in modulators
        assert "capture" in modulators
        assert "lifecycle" in modulators
        assert "look_at" in modulators["motion"]["affordances"]
        assert "antenna_alert" in modulators["expression"]["affordances"]

    def test_has_failure_modes(self):
        from maxim.embodiment.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        spec = registry.get("bodies/reachy_mini")
        failures = spec["entity"]["failure_modes"]
        names = {f["name"] for f in failures}
        # The robot can model running hot, low battery, lost camera/mic, drift
        assert "thermal_throttling" in names
        assert "low_battery" in names
        assert "camera_lost" in names

    def test_can_instantiate_as_entity(self):
        """End-to-end: registry → Entity instance."""
        from maxim.embodiment.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        entity = registry.instantiate("bodies/reachy_mini", name="test_reachy")
        assert entity is not None
        assert entity.name == "test_reachy"


# ─── get_doa_reader capability seam (live_audio_orient_wiring.md Stage 1) ──


class TestDoAReaderSeam:
    """The one new Stage-1 surface: a NON-abstract, default-``None``
    ``RobotController.get_doa_reader``. Non-abstract because the ABC is
    contract-frozen at 12 abstract methods — a 13th breaks every third-party
    ``maxim.robots`` plugin."""

    def _minimal_controller(self):
        """A concrete controller that implements ONLY the frozen abstract
        surface — i.e. a third-party plugin compiled before Stage 1."""
        from maxim.hardware.controller import RobotController

        class _Minimal(RobotController):
            @property
            def robot_type(self):
                return "minimal"

            def connect(self, timeout=30.0):
                return True

            def disconnect(self):
                pass

            def goto_target(self, target):
                return True

            def look_at_pixel(self, target):
                return True

            def get_current_pose(self):
                return {}

            def wake_up(self):
                return True

            def goto_sleep(self):
                return True

            def start_recording(self):
                return True

            def stop_recording(self):
                return True

            def get_video_stream(self):
                return None

            def get_audio_stream(self):
                return None

        return _Minimal("minimal")

    def test_get_doa_reader_is_not_abstract(self):
        """A plugin that never heard of DoA still instantiates cleanly."""
        controller = self._minimal_controller()
        assert controller.robot_type == "minimal"

    def test_default_returns_none(self):
        assert self._minimal_controller().get_doa_reader() is None

    def test_getattr_probe_discovers_an_override(self):
        """Call sites probe via getattr so pre-Stage-1 plugins keep working."""
        controller = self._minimal_controller()

        def scripted():
            return (1.57, True)

        controller.get_doa_reader = scripted  # type: ignore[method-assign]
        probe = getattr(controller, "get_doa_reader", None)
        assert probe is not None
        assert probe() == (1.57, True)

    def test_simulated_controller_scripted_reader_end_to_end(self):
        """SimulatedController pins the seam without hardware: injected
        scripted reader is served while connected, None otherwise."""
        from maxim.hardware.simulation.controller import SimulatedController

        readings = iter([(0.0, True), (3.14159, False)])

        def scripted():
            return next(readings, None)

        controller = SimulatedController(doa_reader=scripted)
        # Not connected yet → capability gated off.
        assert controller.get_doa_reader() is None
        assert controller.connect()
        reader = controller.get_doa_reader()
        assert reader is not None
        assert reader() == (0.0, True)
        assert reader() == (3.14159, False)
        controller.disconnect()
        assert controller.get_doa_reader() is None

    def test_simulated_controller_without_reader_has_no_capability(self):
        from maxim.hardware.simulation.controller import SimulatedController

        controller = SimulatedController()
        controller.connect()
        assert controller.get_doa_reader() is None

    def test_reachy_capabilities_advertise_audio_localization(self):
        """Stage 1c: the capability is advertised via RobotCapabilities.custom
        (the zero-schema-change channel, same as the antenna joints)."""
        from maxim.hardware.capabilities import REACHY_MINI_CAPABILITIES

        assert REACHY_MINI_CAPABILITIES.has_custom("audio_localization")

    def test_abstract_method_count_still_frozen_at_twelve(self):
        """get_doa_reader must never be promoted to @abstractmethod — the
        ABC's abstract surface is contract-frozen for third-party plugins."""
        from maxim.hardware.controller import RobotController

        abstract = getattr(RobotController, "__abstractmethods__", frozenset())
        assert "get_doa_reader" not in abstract
        assert len(abstract) == 12, (
            f"RobotController abstract surface changed: {sorted(abstract)} — "
            "it is contract-frozen at 12 (docs/user/extension_api.md)"
        )
