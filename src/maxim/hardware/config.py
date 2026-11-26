"""Robot configuration loader.

Loads robot configurations from YAML files and initializes
the RobotRegistry with configured robots.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from maxim.exceptions import InvalidConfigError, RobotsConfigError
from maxim.hardware.registry import RobotRegistry

logger = logging.getLogger(__name__)

# robots.yaml lives in exactly one place: ``<data_home>/robots.yaml``
# (``~/.maxim`` unless ``MAXIM_DATA_HOME`` overrides — the same resolution
# as every other persisted surface). There is no ``./robots.yaml`` cwd
# fallback (#1806 owner decision D11): a file planted in the working
# directory must never pick the robot a run connects to.
ROBOTS_YAML_NAME = "robots.yaml"

DEFAULT_ROBOT_NAME = "reachy_mini"


def resolve_daemon_name(typed: str | None) -> tuple[str, str]:
    """The robot's daemon name for a run, and which source chose it: ``(name, source)`` (#1806 PR 0).

    The ONE copy of the legacy chain: the typed ``--robot``/``--robot-name``, else ``$MAXIM_ROBOT_NAME``, else
    ``reachy_mini``. ``selfy.Maxim`` dials it and the CLI's startup log prints it, so the two can never disagree.
    An empty string, typed or in the environment, counts as unset (the repo's env convention), so a blank value can
    never become a ``.local`` dial. Since PR 0 ``MAXIM_ROBOT_NAME`` is read here, at resolution time, not as the
    argparse default.
    """
    import os

    if typed:
        return typed, "--robot-name"
    env_name = os.environ.get("MAXIM_ROBOT_NAME") or ""
    if env_name:
        return env_name, "$MAXIM_ROBOT_NAME"
    return DEFAULT_ROBOT_NAME, "default"


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
        if "config" in data and not isinstance(data["config"], dict):
            # ``config: null`` (an empty ``config:`` line) or a scalar would
            # otherwise TypeError in dict() or silently drop every key.
            raise RobotsConfigError(
                f"robots.yaml entry {robot_id!r}: 'config' must be a mapping, got "
                f"{type(data['config']).__name__} (an empty 'config:' line is null; remove it or give it keys)"
            )
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
    def from_dict(cls, data: Any) -> RobotsConfig:
        """Create from a parsed robots.yaml mapping.

        Raises :class:`~maxim.exceptions.RobotsConfigError` when the top
        level, ``robots:``, or any entry is not a mapping, or an entry's
        ``config:`` is not one (#1806 PR 0). Pre-fix a non-dict entry was
        skipped silently, so a typo'd entry quietly vanished from the
        registry and the run connected to whatever was left.
        """
        if not isinstance(data, dict):
            raise RobotsConfigError(f"robots.yaml must be a mapping with a 'robots:' key, got {type(data).__name__}")
        robots_data = data.get("robots")
        if robots_data is None:  # absent, or an empty ``robots:`` line: no entries
            robots_data = {}
        if not isinstance(robots_data, dict):
            raise RobotsConfigError(
                f"robots.yaml 'robots:' must be a mapping of robot id to entry, got {type(robots_data).__name__}"
            )
        robots = []
        for robot_id, robot_config in robots_data.items():
            if not isinstance(robot_config, dict):
                raise RobotsConfigError(
                    f"robots.yaml entry {robot_id!r} must be a mapping, got {type(robot_config).__name__}"
                )
            robots.append(RobotConfig.from_dict(str(robot_id), robot_config))
        return cls(robots=robots)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotsConfig:
        """Load configuration from a YAML file.

        A missing file is not an error: it means no entries. Anything else
        that stops the file being read as a registry (unreadable, a YAML
        parse error, a malformed shape) raises
        :class:`~maxim.exceptions.RobotsConfigError` naming the path. Pre-fix
        every such error was logged and an EMPTY config returned, so a robot
        run with a broken file connected with defaults (silent-wrong).
        """
        path = Path(path).expanduser()

        if not path.exists():
            logger.debug("Robot config not found: %s", path)
            return cls()

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            raise RobotsConfigError(f"cannot read robot config {path}: {e}") from e
        if data is None:  # an empty file: no entries
            return cls()
        try:
            return cls.from_dict(data)
        except RobotsConfigError as e:
            raise RobotsConfigError(f"{path}: {e}") from e

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


def robots_yaml_data_home() -> Path:
    """The data home robots.yaml resolves under, refusing what would mislead it.

    Raises :class:`~maxim.exceptions.RobotsConfigError` for a RELATIVE
    ``MAXIM_DATA_HOME`` (it would resolve against whatever cwd the run
    started in, so a harness child could read a planted or the operator's
    real file) and when the data home cannot be resolved at all. Scoped to
    robots.yaml resolution only: ``utils/paths.py::data_home()`` itself
    keeps accepting relative dirs, which benchmarks rely on.
    """
    import os

    from maxim.utils.paths import data_home

    raw = os.environ.get("MAXIM_DATA_HOME", "")
    if raw and not Path(raw).is_absolute():
        raise RobotsConfigError(
            f"MAXIM_DATA_HOME={raw!r} is relative; robots.yaml resolution needs an absolute path "
            "(a relative one resolves against the run's cwd)"
        )
    try:
        home = data_home()
    except (OSError, RuntimeError, KeyError) as e:
        # Do NOT fall back to a hardcoded "~/.maxim/robots.yaml" or to
        # "no file": inside an isolated harness run either would silently
        # pick the robot (the operator's real file, or defaults).
        raise RobotsConfigError(f"cannot resolve the data home for robots.yaml: {e}") from e
    if not home.is_absolute():
        raise RobotsConfigError(f"data home {str(home)!r} is relative; robots.yaml resolution needs an absolute path")
    return home


def find_config_file(search_paths: list[str] | None = None) -> Path | None:
    """Find the robots.yaml to load, or ``None`` when there is none.

    Default: ``<data_home>/robots.yaml`` only (``~/.maxim`` unless
    ``MAXIM_DATA_HOME`` overrides). The ``./robots.yaml`` cwd fallback is
    gone (#1806 D11). Raises :class:`~maxim.exceptions.RobotsConfigError`
    when the data home is relative or cannot be resolved. An explicit
    ``search_paths`` is searched in order instead.
    """
    if search_paths is not None:
        paths = [Path(p).expanduser() for p in search_paths]
    else:
        paths = [robots_yaml_data_home() / ROBOTS_YAML_NAME]

    for path in paths:
        if path.exists():
            return path

    return None


def resolve_robot_entry(
    robots_config: RobotsConfig,
    robot_id: str | None,
) -> RobotConfig | None:
    """Resolve the declared robot entry for a runtime id — THE match rule.

    Exact ``robot_id`` match first; else fall back to the primary ONLY
    when it is unambiguous (a single robot, or one explicitly marked
    ``primary``) — never guess the first of several unmarked robots.
    Shared by :func:`resolve_connection_config` and the selfy connect
    path's declared-``type`` resolution so the two can never disagree on
    which robot the operator meant.
    """
    match = robots_config.get(robot_id)
    if match is None:
        has_explicit_primary = any(r.primary for r in robots_config.robots)
        if len(robots_config.robots) == 1 or has_explicit_primary:
            match = robots_config.get_primary()
    return match


def resolve_connection_config(
    robots_config: RobotsConfig,
    robot_id: str | None,
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge the operator's declared robot config over runtime defaults.

    The single place the LIVE connect path resolves ``robots.yaml``'s
    free-form ``config:`` dict (host, connection_mode, tunnel, ...) for a
    robot. Matching is :func:`resolve_robot_entry`'s rule.

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
    match = resolve_robot_entry(robots_config, robot_id)
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
        RobotsConfig instance (empty if no config file exists).

    Raises:
        RobotsConfigError: the file exists but cannot be read as a
            registry, or the data home is relative / unresolvable. Callers
            let it propagate (a robot run must fail, not run headless);
            ``maxim doctor`` reports it as a warning.
    """
    if config_path is not None:
        return RobotsConfig.from_yaml(config_path)

    found_path = find_config_file(search_paths)
    if found_path is not None:
        logger.debug("Loading robot config from: %s", found_path)
        return RobotsConfig.from_yaml(found_path)

    logger.debug("No robot config file found, using defaults")
    return RobotsConfig()


def robot_connect_timeout(default: float) -> float:
    """The robot connect timeout: ``MAXIM_ROBOT_TIMEOUT`` when set (faster headless startup), else ``default``.

    A value that is not a finite positive number raises :class:`~maxim.exceptions.InvalidConfigError` naming the
    variable (#1806 PR 0); pre-fix the bare ``float()`` raised an unexplained ``ValueError`` out of ``Maxim.__init__``.
    """
    raw = os.environ.get("MAXIM_ROBOT_TIMEOUT")
    if raw is None or not raw.strip():
        return float(default)
    refusal = f"MAXIM_ROBOT_TIMEOUT={raw!r} is not a positive number of seconds; unset it or fix it"
    try:
        value = float(raw)
    except ValueError as e:
        raise InvalidConfigError(refusal) from e
    if not math.isfinite(value) or value <= 0:
        raise InvalidConfigError(refusal)
    return value


def resolve_robot_connection(
    robot_id: str, robot_name: str, media_backend: str, *, log: logging.Logger
) -> tuple[str, dict[str, Any], RobotConfig | None]:
    """A live (non-simulated) run's ``(robot_type, connect_config, entry)`` from robots.yaml, read ONCE (#1806 PR 0).

    ``entry`` is the matched robots.yaml entry or ``None``; ``selfy.Maxim._resolve_robot_connection`` hands it to the
    agentic runtime, so the connect and the body wiring read the same robot. A broken robots.yaml raises
    :class:`~maxim.exceptions.RobotsConfigError`: a robot run with a broken file fails, it does not connect with
    defaults or run headless.
    """
    robot_type = "reachy_mini"
    log.info("Connecting to Reachy Mini '%s'...", robot_id)
    # Operator-declared connection config (host, connection_mode, tunnel, ...) comes from <data_home>/robots.yaml.
    # Pre-fix (2026-07-31) this path built the config inline and silently ignored the file, so the connect error's
    # own advice ("set host: <ip> in robots.yaml") did nothing: the controller fell back to mDNS and failed on hosts
    # where .local resolution is blocked. Declared keys win over the inline defaults; wiring-layer keys (body,
    # audio_localization) are filtered to the constructor's signature by connect_robot.
    robots_cfg = load_robots_config()
    connect_config = resolve_connection_config(
        robots_cfg, robot_id, defaults={"robot_name": robot_name, "media_backend": media_backend}
    )
    entry = resolve_robot_entry(robots_cfg, robot_id)
    if entry is None and robots_cfg.robots:
        # F1: several robots, none matching and none marked primary. Never adopt a foreign robot's config or body.
        log.warning(
            "robots.yaml has %d robots, none matching id=%r and none marked primary; "
            "not guessing one. Connecting with defaults and running bodiless.",
            len(robots_cfg.robots),
            robot_id,
        )
    # Honor the operator's declared `type:` (Exp 49 seam, 2026-08-04): `type: simulated` runs the FULL live stack
    # against SimulatedController, declaratively. Same entry as the connection config, so type and config always
    # come from the same robot.
    if entry is not None and entry.robot_type:
        if entry.robot_type != robot_type:
            log.info(
                "robots.yaml declares type=%s for %r — overriding default %s", entry.robot_type, robot_id, robot_type
            )
        robot_type = entry.robot_type
    return robot_type, connect_config, entry


def connect_configured_robots(
    registry: RobotRegistry,
    config: RobotsConfig,
    *,
    timeout: float = 30.0,
    skip_on_failure: bool = True,
) -> dict[str, bool]:
    """Connect all robots from configuration.

    Dormant since 2026-11-25: no ``src/`` caller — the live robot runtime
    connects its one robot through ``selfy.Maxim`` and
    ``resolve_robot_entry``, and #1806 PR 1 routes every connect through
    the registry's identity pin, which this bulk path does not carry. Kept
    wired for the public ``maxim.hardware`` export; do not build on it.

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

    Dormant since 2026-11-25: no ``src/`` caller (see
    :func:`connect_configured_robots`); kept for the public
    ``maxim.hardware`` export. Do not build on it.

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
