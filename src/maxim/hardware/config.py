"""Robot configuration loader.

Loads robot configurations from YAML files and initializes
the RobotRegistry with configured robots.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from maxim.hardware.registry import RobotRegistry

logger = logging.getLogger(__name__)

# Default config paths (searched in order)
DEFAULT_CONFIG_PATHS = [
    "data/util/robots.yaml",
    "robots.yaml",
    "~/.maxim/robots.yaml",
]


@dataclass
class RobotConfig:
    """Configuration for a single robot."""

    robot_id: str
    robot_type: str
    primary: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, robot_id: str, data: dict[str, Any]) -> RobotConfig:
        """Create from configuration dict."""
        return cls(
            robot_id=robot_id,
            robot_type=str(data.get("type", "reachy_mini")),
            primary=bool(data.get("primary", False)),
            config=dict(data.get("config", {})),
        )


@dataclass
class RobotsConfig:
    """Configuration for all robots."""

    robots: list[RobotConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotsConfig:
        """Create from configuration dict."""
        robots_data = data.get("robots", {})
        robots = []

        for robot_id, robot_config in robots_data.items():
            if isinstance(robot_config, dict):
                robots.append(RobotConfig.from_dict(robot_id, robot_config))

        return cls(robots=robots)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotsConfig:
        """Load configuration from YAML file."""
        path = Path(path).expanduser()

        if not path.exists():
            logger.warning("Robot config not found: %s", path)
            return cls()

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)
        except Exception as e:
            logger.error("Failed to load robot config from %s: %s", path, e)
            return cls()

    def get_primary(self) -> RobotConfig | None:
        """Get the primary robot configuration."""
        for robot in self.robots:
            if robot.primary:
                return robot
        return self.robots[0] if self.robots else None


def find_config_file(search_paths: list[str] | None = None) -> Path | None:
    """Find the first existing config file in search paths."""
    paths = search_paths or DEFAULT_CONFIG_PATHS

    for path_str in paths:
        path = Path(path_str).expanduser()
        if path.exists():
            return path

    return None


def load_robots_config(
    config_path: str | Path | None = None,
    search_paths: list[str] | None = None,
) -> RobotsConfig:
    """Load robot configuration.

    Args:
        config_path: Explicit path to config file (overrides search).
        search_paths: Paths to search for config file.

    Returns:
        RobotsConfig instance (empty if no config found).
    """
    if config_path is not None:
        return RobotsConfig.from_yaml(config_path)

    found_path = find_config_file(search_paths)
    if found_path is not None:
        logger.debug("Loading robot config from: %s", found_path)
        return RobotsConfig.from_yaml(found_path)

    logger.debug("No robot config file found, using defaults")
    return RobotsConfig()


def connect_configured_robots(
    registry: RobotRegistry,
    config: RobotsConfig,
    *,
    timeout: float = 30.0,
    skip_on_failure: bool = True,
) -> dict[str, bool]:
    """Connect all robots from configuration.

    Args:
        registry: RobotRegistry to connect robots to.
        config: Robot configuration.
        timeout: Connection timeout per robot.
        skip_on_failure: If True, continue connecting other robots on failure.

    Returns:
        Dict mapping robot_id to connection success status.
    """
    results = {}

    for robot_config in config.robots:
        try:
            robot = registry.connect_robot(
                robot_id=robot_config.robot_id,
                robot_type=robot_config.robot_type,
                config=robot_config.config,
                timeout=timeout,
                set_primary=robot_config.primary,
            )
            results[robot_config.robot_id] = robot is not None

            if robot is not None:
                logger.info(
                    "Connected robot: %s (type=%s, primary=%s)",
                    robot_config.robot_id,
                    robot_config.robot_type,
                    robot_config.primary,
                )
            else:
                logger.warning("Failed to connect robot: %s", robot_config.robot_id)
                if not skip_on_failure:
                    break

        except Exception as e:
            logger.error("Error connecting robot %s: %s", robot_config.robot_id, e)
            results[robot_config.robot_id] = False
            if not skip_on_failure:
                break

    return results


def setup_robots_from_config(
    config_path: str | Path | None = None,
    *,
    timeout: float = 30.0,
) -> tuple[RobotRegistry, dict[str, bool]]:
    """Convenience function to load config and connect all robots.

    Args:
        config_path: Path to robots.yaml (uses search paths if None).
        timeout: Connection timeout per robot.

    Returns:
        Tuple of (registry, connection_results).
    """
    config = load_robots_config(config_path)
    registry = RobotRegistry()

    # Import and register controller types
    from maxim.hardware.reachy import ReachyMiniController
    from maxim.hardware.simulation import SimulatedController

    registry.register_controller_type("reachy_mini", ReachyMiniController)
    registry.register_controller_type("simulated", SimulatedController)

    results = connect_configured_robots(registry, config, timeout=timeout)

    return registry, results
