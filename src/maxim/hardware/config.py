"""Robot configuration loader.

Loads robot configurations from YAML files and initializes
the RobotRegistry with configured robots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from maxim.hardware.registry import RobotRegistry

logger = logging.getLogger(__name__)

# Default config paths (searched in order)
DEFAULT_CONFIG_PATHS = [
    "~/.maxim/robots.yaml",
    "robots.yaml",
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
        """Create from configuration dict.

        ``type`` defaults to ``"reachy_mini"`` for legacy reasons (Reachy Mini
        was the first supported robot). For other robots, set ``type`` explicitly
        in the YAML — e.g. ``type: atlas`` or ``type: spot``. The
        :class:`~maxim.hardware.registry.RobotRegistry` auto-discovers
        controllers via the ``maxim.robots`` entry-point group, so any installed
        plugin can be referenced by its registered name.
        """
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

    def get(self, robot_id: str | None) -> RobotConfig | None:
        """Get a robot config by id, or ``None`` if not found / id is falsy."""
        if not robot_id:
            return None
        for robot in self.robots:
            if robot.robot_id == robot_id:
                return robot
        return None


def resolve_body_ref(robot_config: RobotConfig | None) -> str | None:
    """Resolve the SEM body component ref for a robot from its config.

    The body is declared in ``robots.yaml`` via the free-form ``config`` dict
    — ``config: {body: bodies/reachy_mini}`` — NOT a typed ``RobotConfig`` field (per
    the CC10 / review NH-2 rule: ride the existing free-form dict rather than a
    schema change). This is the ``[declaration]`` seam the embodiment-runtime
    wiring (Track 1) and the audio reflex (Track 2) both build on.

    Returns the declared component ref, or ``None`` when no body is declared.
    ``None`` means "wire no embodiment" — the safe default: the runtime keeps
    its current bodiless behavior and the per-iteration drift tick is a no-op.
    Body-wiring is therefore **opt-in**; defaulting it on per robot_type is a
    separate, deliberate decision (bundled with body_state wiring + the Acting
    Coach drive-modulation fix), not a silent behavior change here.
    """
    if robot_config is None:
        return None
    declared = robot_config.config.get("body")
    return str(declared) if declared else None


def find_config_file(search_paths: list[str] | None = None) -> Path | None:
    """Find the first existing config file in search paths."""
    paths = search_paths or DEFAULT_CONFIG_PATHS

    for path_str in paths:
        path = Path(path_str).expanduser()
        if path.exists():
            return path

    return None


def resolve_connection_config(
    robots_config: RobotsConfig,
    robot_id: str | None,
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge the operator's declared robot config over runtime defaults.

    The single place the LIVE connect path resolves ``robots.yaml``'s
    free-form ``config:`` dict (host, connection_mode, tunnel, ...) for a
    robot. Matching mirrors ``_resolve_body_wiring``'s rule exactly: exact
    ``robot_id`` match first; else fall back to the primary ONLY when it is
    unambiguous (a single robot, or one explicitly marked ``primary``) —
    never guess the first of several unmarked robots.

    Declared keys WIN over ``defaults`` (operator intent beats runtime
    convenience). Returns just the defaults when nothing matches, so a
    machine without a robots.yaml behaves exactly as before.

    Pre-fix history (2026-07-31 live-smoke debugging): the selfy connect
    path built its controller config inline and silently ignored
    robots.yaml — while the connect-failure message told the operator to
    "set host: <ip> in robots.yaml". This helper is what makes that advice
    true.
    """
    merged = dict(defaults or {})
    match = robots_config.get(robot_id)
    if match is None:
        has_explicit_primary = any(r.primary for r in robots_config.robots)
        if len(robots_config.robots) == 1 or has_explicit_primary:
            match = robots_config.get_primary()
    if match is not None:
        merged.update(match.config)
    return merged


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

    # Built-in controllers — Reachy is optional (requires the [reachy] extra),
    # simulated always works. Third-party robots register themselves via the
    # ``maxim.robots`` entry-point group (auto-discovered in RobotRegistry.__init__).
    try:
        from maxim.hardware.reachy import ReachyMiniController

        registry.register_controller_type("reachy_mini", ReachyMiniController)
    except ImportError as e:
        # Reachy SDK not installed — fine if the user is on a peer or another robot.
        import logging

        logging.getLogger(__name__).debug("Reachy controller not available (optional): %s", e)

    from maxim.hardware.simulation import SimulatedController

    registry.register_controller_type("simulated", SimulatedController)

    results = connect_configured_robots(registry, config, timeout=timeout)

    return registry, results
