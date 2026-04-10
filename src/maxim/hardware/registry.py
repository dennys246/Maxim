"""Robot registry for managing multiple robot connections.

The RobotRegistry is a singleton that manages all robot connections,
allowing coordination between multiple robots.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from maxim.hardware.controller import RobotController

if TYPE_CHECKING:
    from maxim.hardware.controller import MotionTarget, PixelTarget

logger = logging.getLogger(__name__)


@dataclass
class RobotRegistration:
    """Registration info for a connected robot."""

    robot_id: str
    robot_type: str
    controller: RobotController
    config: dict[str, Any]
    is_primary: bool = False


class RobotRegistry:
    """Singleton registry for managing multiple robot connections.

    The registry allows:
    - Registering controller types (factories)
    - Connecting/disconnecting robots
    - Accessing robots by ID
    - Designating a primary robot
    - Coordinated operations across robots

    Usage:
        registry = RobotRegistry()

        # Register controller types
        registry.register_controller_type("reachy_mini", ReachyMiniController)
        registry.register_controller_type("simulated", SimulatedController)

        # Connect robots
        registry.connect_robot(
            robot_id="reachy_1",
            robot_type="reachy_mini",
            config={"robot_name": "reachy_mini"},
            set_primary=True,
        )

        # Access robots
        primary = registry.primary
        reachy_1 = registry.get_robot("reachy_1")

        # Iterate robots
        for robot_id in registry.list_robots():
            robot = registry.get_robot(robot_id)
            robot.wake_up()

        # Cleanup
        registry.disconnect_all()
    """

    _instance: RobotRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> RobotRegistry:
        """Singleton pattern - return existing instance or create new."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the registry (only once due to singleton)."""
        if getattr(self, "_initialized", False):
            return

        self._controller_types: dict[str, type[RobotController]] = {}
        self._robots: dict[str, RobotRegistration] = {}
        self._primary_id: str | None = None
        self._robots_lock = threading.RLock()

        # Event callbacks
        self._on_connect_callbacks: list[Callable[[str, str], None]] = []
        self._on_disconnect_callbacks: list[Callable[[str], None]] = []

        self._initialized = True

        # Auto-discover external robot plugins via the ``maxim.robots``
        # entry-point group. Lets third-party packages register controllers
        # without modifying core code:
        #
        #     [project.entry-points."maxim.robots"]
        #     atlas = "maxim_atlas.controller:AtlasController"
        #
        # Failures are silent at debug level — a missing optional plugin
        # shouldn't crash the registry.
        self._discover_entry_point_plugins()

    def _discover_entry_point_plugins(self) -> None:
        """Discover and register robot controllers from the ``maxim.robots`` entry-point group."""
        try:
            from importlib.metadata import entry_points
        except ImportError:
            return

        try:
            eps = entry_points(group="maxim.robots")
        except Exception as e:
            logger.debug("entry_points() lookup failed: %s", e)
            return

        for ep in eps:
            try:
                controller_class = ep.load()
                if not isinstance(controller_class, type) or not issubclass(controller_class, RobotController):
                    logger.warning(
                        "Entry-point %s did not yield a RobotController subclass; got %r",
                        ep.name,
                        controller_class,
                    )
                    continue
                self._controller_types[ep.name] = controller_class
                logger.info(
                    "Auto-registered robot plugin: %s -> %s",
                    ep.name,
                    controller_class.__name__,
                )
            except Exception as e:
                logger.debug("Failed to load robot plugin %s: %s", ep.name, e)

    # ─────────────────────────────────────────────────────────────────────────
    # Controller Type Registration
    # ─────────────────────────────────────────────────────────────────────────

    def register_controller_type(
        self,
        robot_type: str,
        controller_class: type[RobotController],
    ) -> None:
        """Register a controller class for a robot type.

        Args:
            robot_type: Type identifier (e.g., "reachy_mini", "simulated").
            controller_class: Controller class to instantiate for this type.
        """
        self._controller_types[robot_type] = controller_class
        logger.debug("Registered controller type: %s -> %s", robot_type, controller_class.__name__)

    def get_controller_types(self) -> list[str]:
        """Get list of registered controller types."""
        return list(self._controller_types.keys())

    # ─────────────────────────────────────────────────────────────────────────
    # Robot Connection
    # ─────────────────────────────────────────────────────────────────────────

    def connect_robot(
        self,
        robot_id: str,
        robot_type: str,
        config: dict[str, Any] | None = None,
        *,
        set_primary: bool = False,
        timeout: float = 30.0,
    ) -> RobotController | None:
        """Connect a robot.

        Args:
            robot_id: Unique identifier for this robot.
            robot_type: Type of robot (must be registered).
            config: Configuration dict passed to controller.
            set_primary: If True, set this robot as the primary.
            timeout: Connection timeout in seconds.

        Returns:
            Connected RobotController, or None if connection failed.
        """
        if robot_type not in self._controller_types:
            logger.error("Unknown robot type: %s (registered: %s)", robot_type, list(self._controller_types.keys()))
            return None

        with self._robots_lock:
            if robot_id in self._robots:
                logger.warning("Robot already connected: %s", robot_id)
                return self._robots[robot_id].controller

            config = config or {}
            controller_class = self._controller_types[robot_type]

            try:
                controller = controller_class(robot_id=robot_id, **config)
            except TypeError as e:
                # Try without robot_id if controller doesn't expect it
                logger.debug("Retrying controller init without robot_id: %s", e)
                controller = controller_class(**config)

            logger.info("Connecting robot: %s (type=%s)", robot_id, robot_type)

            if not controller.connect(timeout=timeout):
                logger.error("Failed to connect robot: %s", robot_id)
                return None

            registration = RobotRegistration(
                robot_id=robot_id,
                robot_type=robot_type,
                controller=controller,
                config=config,
                is_primary=set_primary,
            )

            self._robots[robot_id] = registration

            if set_primary or self._primary_id is None:
                self._primary_id = robot_id

            # Notify callbacks
            for callback in self._on_connect_callbacks:
                try:
                    callback(robot_id, robot_type)
                except Exception as e:
                    logger.error("Connect callback error: %s", e)

            logger.info("Robot connected: %s (primary=%s)", robot_id, set_primary or self._primary_id == robot_id)
            return controller

    def disconnect_robot(self, robot_id: str) -> bool:
        """Disconnect a robot.

        Args:
            robot_id: ID of robot to disconnect.

        Returns:
            True if robot was disconnected.
        """
        with self._robots_lock:
            if robot_id not in self._robots:
                logger.warning("Robot not found: %s", robot_id)
                return False

            registration = self._robots[robot_id]

            try:
                registration.controller.disconnect()
            except Exception as e:
                logger.error("Error disconnecting robot %s: %s", robot_id, e)

            del self._robots[robot_id]

            if self._primary_id == robot_id:
                # Promote next robot to primary, or None if empty
                self._primary_id = next(iter(self._robots.keys()), None)
                if self._primary_id:
                    self._robots[self._primary_id].is_primary = True

            # Notify callbacks
            for callback in self._on_disconnect_callbacks:
                try:
                    callback(robot_id)
                except Exception as e:
                    logger.error("Disconnect callback error: %s", e)

            logger.info("Robot disconnected: %s", robot_id)
            return True

    def disconnect_all(self) -> None:
        """Disconnect all robots."""
        with self._robots_lock:
            robot_ids = list(self._robots.keys())

        for robot_id in robot_ids:
            self.disconnect_robot(robot_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Robot Access
    # ─────────────────────────────────────────────────────────────────────────

    def get_robot(self, robot_id: str) -> RobotController | None:
        """Get a robot by ID.

        Args:
            robot_id: Robot identifier.

        Returns:
            RobotController, or None if not found.
        """
        with self._robots_lock:
            registration = self._robots.get(robot_id)
            return registration.controller if registration else None

    @property
    def primary(self) -> RobotController | None:
        """Get the primary robot controller."""
        with self._robots_lock:
            if self._primary_id is None:
                return None
            registration = self._robots.get(self._primary_id)
            return registration.controller if registration else None

    @property
    def primary_id(self) -> str | None:
        """Get the primary robot's ID."""
        return self._primary_id

    def set_primary(self, robot_id: str) -> bool:
        """Set a robot as the primary.

        Args:
            robot_id: ID of robot to set as primary.

        Returns:
            True if primary was set.
        """
        with self._robots_lock:
            if robot_id not in self._robots:
                logger.warning("Cannot set primary - robot not found: %s", robot_id)
                return False

            # Demote old primary
            if self._primary_id and self._primary_id in self._robots:
                self._robots[self._primary_id].is_primary = False

            self._primary_id = robot_id
            self._robots[robot_id].is_primary = True
            logger.info("Primary robot set: %s", robot_id)
            return True

    def list_robots(self) -> list[str]:
        """Get list of connected robot IDs."""
        with self._robots_lock:
            return list(self._robots.keys())

    def list_robots_by_type(self, robot_type: str) -> list[str]:
        """Get list of robot IDs of a specific type."""
        with self._robots_lock:
            return [reg.robot_id for reg in self._robots.values() if reg.robot_type == robot_type]

    @property
    def robot_count(self) -> int:
        """Get number of connected robots."""
        with self._robots_lock:
            return len(self._robots)

    # ─────────────────────────────────────────────────────────────────────────
    # Event Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def on_connect(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for robot connection events.

        Callback receives (robot_id, robot_type).
        """
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable[[str], None]) -> None:
        """Register callback for robot disconnection events.

        Callback receives (robot_id,).
        """
        self._on_disconnect_callbacks.append(callback)

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinated Operations
    # ─────────────────────────────────────────────────────────────────────────

    def wake_all(self) -> dict[str, bool]:
        """Wake up all connected robots.

        Returns:
            Dict mapping robot_id to success status.
        """
        results = {}
        with self._robots_lock:
            for robot_id, registration in self._robots.items():
                try:
                    results[robot_id] = registration.controller.wake_up()
                except Exception as e:
                    logger.error("Failed to wake robot %s: %s", robot_id, e)
                    results[robot_id] = False
        return results

    def sleep_all(self) -> dict[str, bool]:
        """Put all connected robots to sleep.

        Returns:
            Dict mapping robot_id to success status.
        """
        results = {}
        with self._robots_lock:
            for robot_id, registration in self._robots.items():
                try:
                    results[robot_id] = registration.controller.goto_sleep()
                except Exception as e:
                    logger.error("Failed to sleep robot %s: %s", robot_id, e)
                    results[robot_id] = False
        return results

    def move_all(
        self,
        target: "MotionTarget",
        robot_ids: list[str] | None = None,
    ) -> dict[str, bool]:
        """Move multiple robots to the same target pose.

        Args:
            target: Target pose for all robots.
            robot_ids: Optional list of robot IDs. If None, moves all robots.

        Returns:
            Dict mapping robot_id to success status.
        """

        results = {}
        with self._robots_lock:
            ids_to_move = robot_ids if robot_ids else list(self._robots.keys())
            for robot_id in ids_to_move:
                registration = self._robots.get(robot_id)
                if registration is None:
                    results[robot_id] = False
                    continue
                try:
                    results[robot_id] = registration.controller.goto_target(target)
                except Exception as e:
                    logger.error("Failed to move robot %s: %s", robot_id, e)
                    results[robot_id] = False
        return results

    def look_at_all(
        self,
        target: "PixelTarget",
        robot_ids: list[str] | None = None,
    ) -> dict[str, bool]:
        """Make multiple robots look at the same pixel coordinate.

        Args:
            target: Target pixel coordinates.
            robot_ids: Optional list of robot IDs. If None, all robots look.

        Returns:
            Dict mapping robot_id to success status.
        """

        results = {}
        with self._robots_lock:
            ids_to_look = robot_ids if robot_ids else list(self._robots.keys())
            for robot_id in ids_to_look:
                registration = self._robots.get(robot_id)
                if registration is None:
                    results[robot_id] = False
                    continue
                try:
                    results[robot_id] = registration.controller.look_at_pixel(target)
                except Exception as e:
                    logger.error("Failed look_at for robot %s: %s", robot_id, e)
                    results[robot_id] = False
        return results

    def center_all(
        self,
        duration: float = 0.5,
        robot_ids: list[str] | None = None,
    ) -> dict[str, bool]:
        """Center vision on all robots.

        Args:
            duration: Movement duration in seconds.
            robot_ids: Optional list of robot IDs. If None, centers all robots.

        Returns:
            Dict mapping robot_id to success status.
        """
        results = {}
        with self._robots_lock:
            ids_to_center = robot_ids if robot_ids else list(self._robots.keys())
            for robot_id in ids_to_center:
                registration = self._robots.get(robot_id)
                if registration is None:
                    results[robot_id] = False
                    continue
                try:
                    results[robot_id] = registration.controller.center_vision(duration)
                except Exception as e:
                    logger.error("Failed to center robot %s: %s", robot_id, e)
                    results[robot_id] = False
        return results

    def get_all_poses(self) -> dict[str, dict[str, float]]:
        """Get current poses from all robots.

        Returns:
            Dict mapping robot_id to pose dict.
        """
        results = {}
        with self._robots_lock:
            for robot_id, registration in self._robots.items():
                try:
                    results[robot_id] = registration.controller.get_current_pose()
                except Exception as e:
                    logger.error("Failed to get pose for robot %s: %s", robot_id, e)
                    results[robot_id] = {}
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Testing Support
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing only)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.disconnect_all()
                cls._instance = None

    def __repr__(self) -> str:
        return (
            f"RobotRegistry(robots={len(self._robots)}, "
            f"primary={self._primary_id!r}, "
            f"types={list(self._controller_types.keys())})"
        )
