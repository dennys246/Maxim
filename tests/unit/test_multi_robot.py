"""Unit tests for multi-robot support."""

from __future__ import annotations

import pytest

from maxim.hardware import (
    MotionTarget,
    PixelTarget,
    RobotConfig,
    RobotRegistry,
    RobotsConfig,
    load_robots_config,
)
from maxim.hardware.simulation import SimulatedController


@pytest.fixture
def registry():
    """Create a fresh registry with simulated controllers."""
    RobotRegistry.reset_instance()
    reg = RobotRegistry()
    reg.register_controller_type("simulated", SimulatedController)
    yield reg
    reg.disconnect_all()
    RobotRegistry.reset_instance()


class TestRobotsConfig:
    """Test robot configuration loading."""

    def test_robot_config_from_dict(self):
        """Can create RobotConfig from dict."""
        data = {
            "type": "simulated",
            "primary": True,
            "config": {"video_resolution": [640, 480]},
        }

        config = RobotConfig.from_dict("test_robot", data)

        assert config.robot_id == "test_robot"
        assert config.robot_type == "simulated"
        assert config.primary is True
        assert config.config == {"video_resolution": [640, 480]}

    def test_robots_config_from_dict(self):
        """Can create RobotsConfig from dict."""
        data = {
            "robots": {
                "primary": {
                    "type": "simulated",
                    "primary": True,
                    "config": {},
                },
                "secondary": {
                    "type": "simulated",
                    "primary": False,
                    "config": {},
                },
            }
        }

        config = RobotsConfig.from_dict(data)

        assert len(config.robots) == 2
        assert config.get_primary().robot_id == "primary"

    def test_robots_config_empty(self):
        """Empty config has no robots."""
        config = RobotsConfig()

        assert len(config.robots) == 0
        assert config.get_primary() is None


class TestMultiRobotConnection:
    """Test connecting multiple robots."""

    def test_connect_two_robots(self, registry):
        """Can connect two simulated robots."""
        robot1 = registry.connect_robot(
            robot_id="robot1",
            robot_type="simulated",
            set_primary=True,
        )
        robot2 = registry.connect_robot(
            robot_id="robot2",
            robot_type="simulated",
            set_primary=False,
        )

        assert robot1 is not None
        assert robot2 is not None
        assert registry.robot_count == 2
        assert registry.primary_id == "robot1"

    def test_get_robots_by_id(self, registry):
        """Can get individual robots by ID."""
        registry.connect_robot(robot_id="alpha", robot_type="simulated")
        registry.connect_robot(robot_id="beta", robot_type="simulated")

        alpha = registry.get_robot("alpha")
        beta = registry.get_robot("beta")

        assert alpha is not None
        assert beta is not None
        assert alpha.robot_id == "alpha"
        assert beta.robot_id == "beta"

    def test_robots_have_independent_state(self, registry):
        """Each robot maintains independent state."""
        registry.connect_robot(robot_id="r1", robot_type="simulated")
        registry.connect_robot(robot_id="r2", robot_type="simulated")

        r1 = registry.get_robot("r1")
        r2 = registry.get_robot("r2")

        # Wake only r1
        r1.wake_up()

        assert r1.state.is_awake
        assert not r2.state.is_awake

    def test_independent_motion(self, registry):
        """Robots can move independently."""
        registry.connect_robot(robot_id="left", robot_type="simulated")
        registry.connect_robot(robot_id="right", robot_type="simulated")

        left = registry.get_robot("left")
        right = registry.get_robot("right")

        left.wake_up()
        right.wake_up()

        # Move to different poses
        left.goto_target(MotionTarget(head_yaw=0.5))
        right.goto_target(MotionTarget(head_yaw=-0.5))

        left_pose = left.get_current_pose()
        right_pose = right.get_current_pose()

        assert left_pose["yaw"] == 0.5
        assert right_pose["yaw"] == -0.5


class TestCoordinatedOperations:
    """Test coordinated multi-robot operations."""

    def test_wake_all(self, registry):
        """wake_all wakes all robots."""
        registry.connect_robot(robot_id="a", robot_type="simulated")
        registry.connect_robot(robot_id="b", robot_type="simulated")
        registry.connect_robot(robot_id="c", robot_type="simulated")

        results = registry.wake_all()

        assert all(results.values())
        assert registry.get_robot("a").state.is_awake
        assert registry.get_robot("b").state.is_awake
        assert registry.get_robot("c").state.is_awake

    def test_sleep_all(self, registry):
        """sleep_all sleeps all robots."""
        registry.connect_robot(robot_id="a", robot_type="simulated")
        registry.connect_robot(robot_id="b", robot_type="simulated")
        registry.wake_all()

        results = registry.sleep_all()

        assert all(results.values())
        assert not registry.get_robot("a").state.is_awake
        assert not registry.get_robot("b").state.is_awake

    def test_move_all(self, registry):
        """move_all moves all robots to same pose."""
        registry.connect_robot(robot_id="a", robot_type="simulated")
        registry.connect_robot(robot_id="b", robot_type="simulated")
        registry.wake_all()

        target = MotionTarget(head_pitch=0.3, head_yaw=0.2)
        results = registry.move_all(target)

        assert all(results.values())

        pose_a = registry.get_robot("a").get_current_pose()
        pose_b = registry.get_robot("b").get_current_pose()

        assert pose_a["pitch"] == 0.3
        assert pose_a["yaw"] == 0.2
        assert pose_b["pitch"] == 0.3
        assert pose_b["yaw"] == 0.2

    def test_move_subset(self, registry):
        """move_all can target a subset of robots."""
        registry.connect_robot(robot_id="a", robot_type="simulated")
        registry.connect_robot(robot_id="b", robot_type="simulated")
        registry.connect_robot(robot_id="c", robot_type="simulated")
        registry.wake_all()

        # Only move a and c
        target = MotionTarget(head_pitch=0.5)
        results = registry.move_all(target, robot_ids=["a", "c"])

        assert "a" in results
        assert "b" not in results
        assert "c" in results

        assert registry.get_robot("a").get_current_pose()["pitch"] == 0.5
        assert registry.get_robot("b").get_current_pose()["pitch"] == 0.0  # Unchanged
        assert registry.get_robot("c").get_current_pose()["pitch"] == 0.5

    def test_center_all(self, registry):
        """center_all centers all robots."""
        registry.connect_robot(robot_id="a", robot_type="simulated")
        registry.connect_robot(robot_id="b", robot_type="simulated")
        registry.wake_all()

        # Move to non-center
        registry.get_robot("a").goto_target(MotionTarget(head_pitch=0.5, head_yaw=0.5))
        registry.get_robot("b").goto_target(MotionTarget(head_pitch=-0.5, head_yaw=-0.5))

        results = registry.center_all()

        assert all(results.values())

        pose_a = registry.get_robot("a").get_current_pose()
        pose_b = registry.get_robot("b").get_current_pose()

        assert pose_a["pitch"] == 0.0
        assert pose_a["yaw"] == 0.0
        assert pose_b["pitch"] == 0.0
        assert pose_b["yaw"] == 0.0

    def test_get_all_poses(self, registry):
        """get_all_poses returns poses from all robots."""
        registry.connect_robot(robot_id="a", robot_type="simulated")
        registry.connect_robot(robot_id="b", robot_type="simulated")
        registry.wake_all()

        registry.get_robot("a").goto_target(MotionTarget(head_pitch=0.1))
        registry.get_robot("b").goto_target(MotionTarget(head_pitch=0.2))

        poses = registry.get_all_poses()

        assert "a" in poses
        assert "b" in poses
        assert poses["a"]["pitch"] == 0.1
        assert poses["b"]["pitch"] == 0.2


class TestMultiRobotStreams:
    """Test data streams from multiple robots."""

    def test_each_robot_has_streams(self, registry):
        """Each robot has independent streams."""
        registry.connect_robot(robot_id="cam1", robot_type="simulated")
        registry.connect_robot(robot_id="cam2", robot_type="simulated")

        cam1 = registry.get_robot("cam1")
        cam2 = registry.get_robot("cam2")

        cam1.start_recording()
        cam2.start_recording()

        video1 = cam1.get_video_stream()
        video2 = cam2.get_video_stream()

        assert video1 is not None
        assert video2 is not None
        assert video1 is not video2  # Different stream objects

    def test_get_frames_from_multiple_robots(self, registry):
        """Can get frames from multiple robots."""
        registry.connect_robot(robot_id="a", robot_type="simulated")
        registry.connect_robot(robot_id="b", robot_type="simulated")

        a = registry.get_robot("a")
        b = registry.get_robot("b")

        a.start_recording()
        b.start_recording()

        frame_a = a.get_video_stream().get_frame_nonblocking()
        frame_b = b.get_video_stream().get_frame_nonblocking()

        assert frame_a is not None
        assert frame_b is not None
        assert frame_a.shape == (480, 640, 3)
        assert frame_b.shape == (480, 640, 3)


class TestRobotIdInTools:
    """Test robot_id parameter in tools."""

    def test_get_robot_from_registry_helper(self, registry):
        """_get_robot_from_registry helper works."""
        from maxim.tools.reachy import _get_robot_from_registry

        registry.connect_robot(robot_id="test", robot_type="simulated", set_primary=True)

        # Get by ID
        robot = _get_robot_from_registry("test")
        assert robot is not None
        assert robot.robot_id == "test"

        # Get primary when ID is None
        robot = _get_robot_from_registry(None)
        assert robot is not None
        assert robot.robot_id == "test"

    def test_get_robot_nonexistent(self, registry):
        """Returns None for nonexistent robot."""
        from maxim.tools.reachy import _get_robot_from_registry

        robot = _get_robot_from_registry("does_not_exist")
        assert robot is None
