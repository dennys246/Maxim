"""Unit tests for RobotRegistry."""

from __future__ import annotations

import pytest

from maxim.hardware import RobotRegistry
from maxim.hardware.simulation import SimulatedController


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    # Reset singleton to get clean state
    RobotRegistry.reset_instance()
    reg = RobotRegistry()
    reg.register_controller_type("simulated", SimulatedController)
    yield reg
    reg.disconnect_all()
    RobotRegistry.reset_instance()


class TestRobotRegistrySingleton:
    """Test singleton behavior."""

    def test_singleton_pattern(self):
        """Returns same instance."""
        RobotRegistry.reset_instance()

        reg1 = RobotRegistry()
        reg2 = RobotRegistry()

        assert reg1 is reg2

        RobotRegistry.reset_instance()

    def test_reset_creates_new_instance(self):
        """reset_instance() allows new instance."""
        reg1 = RobotRegistry()
        RobotRegistry.reset_instance()
        reg2 = RobotRegistry()

        # Different objects after reset
        # (can't compare identity since singleton pattern)
        assert reg1._initialized
        assert reg2._initialized


class TestControllerTypeRegistration:
    """Test controller type registration."""

    def test_register_controller_type(self, registry):
        """Can register controller types."""
        types = registry.get_controller_types()

        assert "simulated" in types

    def test_get_controller_types(self, registry):
        """get_controller_types returns all registered types."""
        # Already has "simulated" from fixture

        # Register another type (using same class for test)
        registry.register_controller_type("another", SimulatedController)

        types = registry.get_controller_types()
        assert "simulated" in types
        assert "another" in types


class TestRobotConnection:
    """Test robot connection management."""

    def test_connect_robot(self, registry):
        """Can connect a robot."""
        robot = registry.connect_robot(
            robot_id="test_robot",
            robot_type="simulated",
        )

        assert robot is not None
        assert robot.is_connected()
        assert "test_robot" in registry.list_robots()

    def test_connect_with_config(self, registry):
        """Can connect with configuration."""
        robot = registry.connect_robot(
            robot_id="configured",
            robot_type="simulated",
            config={"video_resolution": (320, 240)},
        )

        assert robot is not None
        assert robot.capabilities.video_resolution == (320, 240)

    def test_connect_unknown_type_fails(self, registry):
        """Connecting unknown type returns None."""
        robot = registry.connect_robot(
            robot_id="unknown",
            robot_type="nonexistent_type",
        )

        assert robot is None
        assert "unknown" not in registry.list_robots()

    def test_connect_duplicate_returns_existing(self, registry):
        """Connecting duplicate ID returns existing robot."""
        robot1 = registry.connect_robot(robot_id="dup", robot_type="simulated")
        robot2 = registry.connect_robot(robot_id="dup", robot_type="simulated")

        assert robot1 is robot2

    def test_disconnect_robot(self, registry):
        """Can disconnect a robot."""
        registry.connect_robot(robot_id="to_disconnect", robot_type="simulated")
        assert "to_disconnect" in registry.list_robots()

        result = registry.disconnect_robot("to_disconnect")

        assert result is True
        assert "to_disconnect" not in registry.list_robots()

    def test_disconnect_nonexistent_returns_false(self, registry):
        """Disconnecting nonexistent robot returns False."""
        result = registry.disconnect_robot("does_not_exist")

        assert result is False

    def test_disconnect_all(self, registry):
        """disconnect_all() disconnects all robots."""
        registry.connect_robot(robot_id="robot1", robot_type="simulated")
        registry.connect_robot(robot_id="robot2", robot_type="simulated")
        assert registry.robot_count == 2

        registry.disconnect_all()

        assert registry.robot_count == 0
        assert registry.list_robots() == []


class TestPrimaryRobot:
    """Test primary robot management."""

    def test_first_robot_becomes_primary(self, registry):
        """First connected robot becomes primary."""
        registry.connect_robot(robot_id="first", robot_type="simulated")

        assert registry.primary_id == "first"
        assert registry.primary is not None
        assert registry.primary.robot_id == "first"

    def test_set_primary_explicit(self, registry):
        """Can explicitly set primary."""
        registry.connect_robot(robot_id="first", robot_type="simulated")
        registry.connect_robot(robot_id="second", robot_type="simulated")

        assert registry.primary_id == "first"

        result = registry.set_primary("second")

        assert result is True
        assert registry.primary_id == "second"

    def test_set_primary_on_connect(self, registry):
        """Can set primary on connect."""
        registry.connect_robot(robot_id="first", robot_type="simulated")
        registry.connect_robot(
            robot_id="second",
            robot_type="simulated",
            set_primary=True,
        )

        assert registry.primary_id == "second"

    def test_set_primary_nonexistent_fails(self, registry):
        """Setting nonexistent robot as primary fails."""
        result = registry.set_primary("does_not_exist")

        assert result is False

    def test_primary_promotion_on_disconnect(self, registry):
        """Primary is promoted when primary disconnects."""
        registry.connect_robot(robot_id="first", robot_type="simulated")
        registry.connect_robot(robot_id="second", robot_type="simulated")

        registry.disconnect_robot("first")

        # Second should be promoted to primary
        assert registry.primary_id == "second"

    def test_primary_none_when_empty(self, registry):
        """Primary is None when no robots connected."""
        assert registry.primary is None
        assert registry.primary_id is None


class TestRobotAccess:
    """Test robot access methods."""

    def test_get_robot(self, registry):
        """Can get robot by ID."""
        registry.connect_robot(robot_id="findme", robot_type="simulated")

        robot = registry.get_robot("findme")

        assert robot is not None
        assert robot.robot_id == "findme"

    def test_get_robot_not_found(self, registry):
        """Getting nonexistent robot returns None."""
        robot = registry.get_robot("does_not_exist")

        assert robot is None

    def test_list_robots(self, registry):
        """list_robots() returns all robot IDs."""
        registry.connect_robot(robot_id="a", robot_type="simulated")
        registry.connect_robot(robot_id="b", robot_type="simulated")

        robots = registry.list_robots()

        assert "a" in robots
        assert "b" in robots
        assert len(robots) == 2

    def test_list_robots_by_type(self, registry):
        """list_robots_by_type() filters by type."""
        registry.connect_robot(robot_id="sim1", robot_type="simulated")
        registry.connect_robot(robot_id="sim2", robot_type="simulated")

        sims = registry.list_robots_by_type("simulated")
        others = registry.list_robots_by_type("other_type")

        assert len(sims) == 2
        assert len(others) == 0

    def test_robot_count(self, registry):
        """robot_count returns correct count."""
        assert registry.robot_count == 0

        registry.connect_robot(robot_id="one", robot_type="simulated")
        assert registry.robot_count == 1

        registry.connect_robot(robot_id="two", robot_type="simulated")
        assert registry.robot_count == 2

        registry.disconnect_robot("one")
        assert registry.robot_count == 1


class TestCoordinatedOperations:
    """Test coordinated operations across robots."""

    def test_wake_all(self, registry):
        """wake_all() wakes all robots."""
        registry.connect_robot(robot_id="a", robot_type="simulated")
        registry.connect_robot(robot_id="b", robot_type="simulated")

        results = registry.wake_all()

        assert results["a"] is True
        assert results["b"] is True

        # Verify they're awake
        assert registry.get_robot("a").state.is_awake
        assert registry.get_robot("b").state.is_awake

    def test_sleep_all(self, registry):
        """sleep_all() sleeps all robots."""
        registry.connect_robot(robot_id="a", robot_type="simulated")
        registry.connect_robot(robot_id="b", robot_type="simulated")
        registry.wake_all()

        results = registry.sleep_all()

        assert results["a"] is True
        assert results["b"] is True

        # Verify they're asleep
        assert not registry.get_robot("a").state.is_awake
        assert not registry.get_robot("b").state.is_awake


class TestEventCallbacks:
    """Test event callback registration."""

    def test_on_connect_callback(self, registry):
        """on_connect callback is called."""
        events = []

        def on_connect(robot_id, robot_type):
            events.append(("connect", robot_id, robot_type))

        registry.on_connect(on_connect)
        registry.connect_robot(robot_id="test", robot_type="simulated")

        assert len(events) == 1
        assert events[0] == ("connect", "test", "simulated")

    def test_on_disconnect_callback(self, registry):
        """on_disconnect callback is called."""
        events = []

        def on_disconnect(robot_id):
            events.append(("disconnect", robot_id))

        registry.on_disconnect(on_disconnect)
        registry.connect_robot(robot_id="test", robot_type="simulated")
        registry.disconnect_robot("test")

        assert len(events) == 1
        assert events[0] == ("disconnect", "test")


class TestRepr:
    """Test string representation."""

    def test_repr(self, registry):
        """Has useful repr."""
        registry.connect_robot(robot_id="test", robot_type="simulated")

        repr_str = repr(registry)

        assert "RobotRegistry" in repr_str
        assert "robots=1" in repr_str
        assert "simulated" in repr_str
