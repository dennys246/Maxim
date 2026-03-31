"""Unit tests for RuntimeCapabilities dataclass."""

from __future__ import annotations

from maxim.runtime.capabilities import RuntimeCapabilities


class TestRuntimeCapabilitiesDefaults:
    """Verify all defaults are safe/headless."""

    def test_all_bools_false(self):
        caps = RuntimeCapabilities()
        assert caps.has_robot is False
        assert caps.has_gpu is False
        assert caps.has_vision is False
        assert caps.has_audio is False
        assert caps.has_motor is False
        assert caps.has_display is False
        assert caps.has_network is False

    def test_optional_fields_none_or_empty(self):
        caps = RuntimeCapabilities()
        assert caps.robot_type is None
        assert caps.gpu_type is None
        assert caps.connected_devices == []


class TestRuntimeCapabilitiesUpdate:
    """Verify fields can be set and read back."""

    def test_set_robot_and_motor(self):
        caps = RuntimeCapabilities(has_robot=True, has_motor=True)
        assert caps.has_robot is True
        assert caps.has_motor is True
        # Other bools remain False
        assert caps.has_gpu is False
        assert caps.has_vision is False

    def test_set_robot_type(self):
        caps = RuntimeCapabilities(
            has_robot=True, robot_type="mycobot_280"
        )
        assert caps.robot_type == "mycobot_280"

    def test_connected_devices(self):
        caps = RuntimeCapabilities(
            connected_devices=["camera_0", "mic_1"]
        )
        assert len(caps.connected_devices) == 2
        assert "camera_0" in caps.connected_devices


class TestHeadlessCapabilities:
    """Verify headless (all-defaults) config has no hardware."""

    def test_headless_no_robot(self):
        caps = RuntimeCapabilities()
        assert caps.has_robot is False
        assert caps.has_motor is False
        assert caps.robot_type is None

    def test_headless_no_devices(self):
        caps = RuntimeCapabilities()
        assert caps.connected_devices == []

    def test_mutate_after_creation(self):
        caps = RuntimeCapabilities()
        caps.has_gpu = True
        caps.gpu_type = "cuda"
        assert caps.has_gpu is True
        assert caps.gpu_type == "cuda"
