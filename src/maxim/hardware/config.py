"""Robot configuration loader.

Loads robot configurations from YAML files and initializes
the RobotRegistry with configured robots.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from maxim.exceptions import ConfigurationError, InvalidConfigError, RobotPinUnverifiable, RobotsConfigError
from maxim.hardware.devices_types import HARDWARE_ID_PATTERN, UNPINNED_LITERAL, DevicePin
from maxim.hardware.registry import RobotRegistry
from maxim.utils.format_version import check_format_version

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


# The robots.yaml format this build reads and (from #1806 PR 2) writes. A file with a root ``_format_version`` is
# writer-managed: every entry must then carry ``hardware_id:`` (``unpinned`` when it has no pin).
ROBOTS_FORMAT_VERSION = "1.0"

# The entry-level keys, exactly (#1806 §4.1). Anything else, a typo such as ``hardware_idd:`` included, is a
# RobotsConfigError, so a typo can never yield a silently unpinned entry.
ROOT_KEYS = frozenset({"robots", "_format_version"})
ENTRY_KEYS = frozenset({"type", "primary", "config", "kind", "hardware_id", "hostname", "launch"})
_ENTRY_ONLY_KEYS = ("hardware_id", "hostname")  # refused under ``config:``, where nothing would read them
ENTRY_KINDS = frozenset({"robot"})  # reserved; one kind exists today (§1)
# ``launch:`` holds the run defaults ``maxim embody`` uses (#1806 PR 2b); never ``operational_mode`` (D8).
_LAUNCH_KEYS: dict[str, type] = {"mode": str, "llm": str, "interactive": bool}

# The Reachy Mini SDK's own vocabularies, matched on ``value.lower()`` exactly as the SDK normalises them
# (ReachyMini._normalize_connection_mode and _configure_mediamanager), so Maxim's classification and the SDK's
# can never disagree.
REACHY_CONNECTION_MODES = frozenset({"auto", "localhost_only", "network"})
# Every backend the SDK resolves to LOCAL: it reads the Maxim HOST's camera IPC socket and host audio, whatever the
# connection mode. On a networked robot that is always the wrong body.
REACHY_LOCAL_MEDIA = frozenset(
    {"local", "gstreamer", "gstreamer_no_video", "sounddevice_no_video", "sounddevice_opencv"}
)
REACHY_REMOTE_MEDIA = frozenset({"default", "auto", "webrtc", "no_media"})


_MERGE_TAG = "tag:yaml.org,2002:merge"


class _NoDuplicateKeysLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that refuses a duplicate mapping key (#1806 PR 1a review E1).

    PyYAML keeps the LAST duplicate silently, so a second ``hardware_id: unpinned`` line under an entry, or a second
    entry with the same id, would replace the first without a word. The check runs in the COMPOSER, where PyYAML
    builds each mapping node exactly once (an inline merge value included) before any merge flattening or
    construction: two constructor-level attempts each broke a legitimate merge, because ``flatten_mapping`` rewrites
    nodes in place (delta rounds 3 and 4). Merge keys are skipped, so ``<<: *base`` followed by an explicit key stays
    a standard override; a second merge key in one mapping is refused, since its precedence would depend on the
    spelling. Construction is SafeLoader's own, unchanged.
    """

    def construct_object(self, node: yaml.Node, deep: bool = False) -> Any:
        """SafeLoader's construction, with its raw scalar errors typed (round 6 J1).

        SafeLoader's scalar constructors raise bare ``KeyError`` (``!!bool maybe``), ``AttributeError``
        (``!!timestamp abc``), ``IndexError`` (``!!int ''``) and ``ValueError`` (``!!int abc``); each becomes a
        ``ConstructorError`` at the node, so :meth:`RobotsConfig.from_yaml` reports it as a ``RobotsConfigError``.
        """
        try:
            return super().construct_object(node, deep=deep)
        except (KeyError, AttributeError, IndexError, ValueError, TypeError) as e:
            raise yaml.constructor.ConstructorError(
                None, None, f"cannot construct {node.tag} value {getattr(node, 'value', '')!r}: {e}", node.start_mark
            ) from e

    def compose_mapping_node(self, anchor: Any) -> yaml.MappingNode:
        node = super().compose_mapping_node(anchor)
        seen: set[Any] = set()
        merge_seen = False
        for key_node, _value_node in node.value:
            if key_node.tag == _MERGE_TAG:
                if merge_seen:
                    raise yaml.composer.ComposerError(
                        "while composing a mapping",
                        node.start_mark,
                        "found a second merge key '<<'",
                        key_node.start_mark,
                    )
                merge_seen = True
                continue
            if not isinstance(key_node, yaml.ScalarNode):
                continue  # a collection key: SafeLoader refuses it as unhashable at construction
            key = self.construct_object(key_node)
            if not isinstance(key, Hashable):
                # A collection-tagged scalar key (``!!set q:``). Checked explicitly, as SafeLoader does: ``set() in
                # seen`` does not raise (sets are looked up as frozensets), but ``seen.add`` would, untyped.
                raise yaml.composer.ComposerError(
                    "while composing a mapping",
                    node.start_mark,
                    f"found an unhashable key {key!r}",
                    key_node.start_mark,
                )
            if key in seen:
                raise yaml.composer.ComposerError(
                    "while composing a mapping", node.start_mark, f"found duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return node


def _entry_error(robot_id: str, message: str) -> RobotsConfigError:
    return RobotsConfigError(f"robots.yaml entry {robot_id!r}: {message}")


def _parse_pin(robot_id: str, data: dict[str, Any], *, managed: bool) -> DevicePin:
    """The entry's ``hardware_id:`` as a :class:`DevicePin`. Absent means unpinned only in a hand-written file."""
    if "hardware_id" not in data:
        if managed:
            raise _entry_error(
                robot_id,
                "'hardware_id' is required in a writer-managed robots.yaml (one with _format_version); "
                f"write 'hardware_id: {UNPINNED_LITERAL}' for an entry with no pin",
            )
        return DevicePin.UNPINNED
    raw = data["hardware_id"]
    if not isinstance(raw, str):
        # An unquoted all-digit id loads as an int (as octal with a leading 0); never coerce it.
        raise _entry_error(
            robot_id, f"'hardware_id' must be a quoted string, got {type(raw).__name__} {raw!r} (quote it)"
        )
    if raw == UNPINNED_LITERAL:
        return DevicePin.UNPINNED
    if HARDWARE_ID_PATTERN.fullmatch(raw) is None:
        raise _entry_error(
            robot_id, f"'hardware_id' must be 16 lowercase hex characters or {UNPINNED_LITERAL!r}, got {raw!r}"
        )
    return DevicePin(raw)


def _parse_launch(robot_id: str, raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise _entry_error(robot_id, f"'launch' must be a mapping, got {type(raw).__name__}")
    if "operational_mode" in raw:
        raise _entry_error(
            robot_id,
            "'launch' never holds operational_mode (#1806 D8): the operational mode is chosen at launch, not stored",
        )
    unknown = sorted(str(k) for k in raw if k not in _LAUNCH_KEYS)
    if unknown:
        raise _entry_error(robot_id, f"unknown 'launch' key(s) {unknown}; allowed: {sorted(_LAUNCH_KEYS)}")
    for key, value in raw.items():
        if type(value) is not _LAUNCH_KEYS[key]:
            raise _entry_error(robot_id, f"'launch.{key}' must be a {_LAUNCH_KEYS[key].__name__}, got {value!r}")
    return dict(raw)


def check_reachy_media(
    config: dict[str, Any], *, where: str, error: type[ConfigurationError] = RobotsConfigError
) -> dict[str, Any]:
    """A copy of a Reachy connect config with ``connection_mode``/``media_backend`` canonical, or a typed refusal.

    Owner rule (#1806, 2026-11-25): a LOCAL media backend (``local`` or any alias the SDK resolves to LOCAL) reads
    the Maxim host's camera and audio, so it is allowed only when the lowercased ``connection_mode`` is exactly
    ``localhost_only`` and the entry is not ``tunnel: true``. Everything else counts as networked: an absent mode
    (``network``), ``auto`` (the SDK falls back to the network and opens the camera in the same call), and a
    tunnel. Unknown values and non-strings are refused rather than left to the SDK, which falls back to
    auto-detect on an unknown backend. ``tunnel`` must be a real bool: the controller branches on truthiness, so a
    quoted ``"true"`` would open the tunnel while a bool check called the entry "not a tunnel". ``where`` names
    the source in the error.

    The ONE copy of this rule: PR 1b's controller-side check (approach note §4.2, "in the controller") calls this
    function and never re-implements the sets. Two residuals remain until PR 1b's controller work: a
    ``tunnel: true`` entry with ``default``/``auto`` media is forced to ``localhost_only`` by the controller, so the
    SDK may still pick LOCAL; and a manual SSH tunnel written as ``localhost_only`` passes here (the onboard proof is
    1b's).
    """
    out = dict(config)
    tunnel = out.get("tunnel", False)
    if type(tunnel) is not bool:
        raise error(f"{where}: 'tunnel' must be true or false (unquoted), got {tunnel!r}")
    mode = out.get("connection_mode")
    if "connection_mode" in out:  # a present null is refused too: the SDK would call None.lower()
        if not isinstance(mode, str) or mode.lower() not in REACHY_CONNECTION_MODES:
            raise error(f"{where}: 'connection_mode' must be one of {sorted(REACHY_CONNECTION_MODES)}, got {mode!r}")
        out["connection_mode"] = mode.lower()
    media = out.get("media_backend")
    if "media_backend" in out:
        known = REACHY_LOCAL_MEDIA | REACHY_REMOTE_MEDIA
        if not isinstance(media, str) or media.lower() not in known:
            # YAML ``no`` parses as a bool, so ``media_backend: no`` lands here rather than meaning no_media.
            raise error(f"{where}: 'media_backend' must be one of {sorted(known)}, got {media!r}")
        out["media_backend"] = media.lower()
        if out["media_backend"] in REACHY_LOCAL_MEDIA and (out.get("connection_mode") != "localhost_only" or tunnel):
            raise error(
                f"{where}: media_backend {media!r} reads THIS machine's camera and audio, so it is allowed only "
                "for connection_mode: localhost_only without tunnel: true (Maxim running on the robot itself). "
                "For a networked robot use media_backend: default (WebRTC from the robot) or no_media"
            )
    return out


@dataclass
class RobotConfig:
    """Configuration for a single robot.

    ``hardware_id`` is the entry's identity pin (never ``None``; :attr:`DevicePin.UNPINNED` when it has none),
    ``hostname`` the name to re-resolve when ``config.host`` is absent, and ``launch`` the stored run defaults
    (#1806 §4.1). ``kind`` is reserved and is ``"robot"`` today.
    """

    robot_id: str
    robot_type: str
    primary: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    hardware_id: DevicePin = field(default_factory=lambda: DevicePin.UNPINNED)
    hostname: str | None = None
    launch: dict[str, Any] = field(default_factory=dict)
    kind: str = "robot"

    def __post_init__(self) -> None:
        """The typed-field invariants, on EVERY construction (#1806 PR 1a review W1).

        In the constructor rather than :meth:`from_dict`, so a ``RobotConfig`` built directly (the public
        ``maxim.hardware`` export, the bulk connect) cannot bypass them: a str ``hardware_id`` or a LOCAL media
        backend on a networked Reachy is refused here too. ``config`` is replaced by its canonical copy. The checks
        hold at construction only: the class stays mutable, and no ``src/`` caller mutates an entry after it.
        """
        if not isinstance(self.robot_id, str) or not self.robot_id:
            raise RobotsConfigError(f"robots.yaml: a robot id must be a non-empty name, got {self.robot_id!r}")
        rid = self.robot_id
        if not isinstance(self.robot_type, str) or not self.robot_type:
            raise _entry_error(rid, f"'type' must be a non-empty string, got {self.robot_type!r}")
        if type(self.primary) is not bool:
            # ``primary: 1`` or a quoted "false" (truthy) would otherwise pick the robot.
            raise _entry_error(rid, f"'primary' must be true or false (unquoted), got {self.primary!r}")
        if not isinstance(self.kind, str) or self.kind not in ENTRY_KINDS:
            raise _entry_error(rid, f"'kind' must be one of {sorted(ENTRY_KINDS)}, got {self.kind!r}")
        if not isinstance(self.hardware_id, DevicePin):
            raise _entry_error(rid, f"'hardware_id' must be a DevicePin, got {type(self.hardware_id).__name__}")
        hostname = self.hostname
        if hostname is not None and (not isinstance(hostname, str) or not hostname or hostname != hostname.strip()):
            raise _entry_error(rid, f"'hostname' must be a non-empty name, got {hostname!r}")
        if not isinstance(self.config, dict):
            raise _entry_error(rid, f"'config' must be a mapping, got {type(self.config).__name__}")
        if self.robot_type == "reachy_mini":
            self.config = check_reachy_media(self.config, where=f"robots.yaml entry {rid!r}")

    @classmethod
    def from_dict(cls, robot_id: str, data: dict[str, Any], *, managed: bool = False) -> RobotConfig:
        """Create from configuration dict.

        ``type`` defaults to ``"reachy_mini"`` for legacy reasons (Reachy Mini
        was the first supported robot). For other robots, set ``type`` explicitly
        in the YAML — e.g. ``type: atlas`` or ``type: spot``. The
        :class:`~maxim.hardware.registry.RobotRegistry` auto-discovers
        controllers via the ``maxim.robots`` entry-point group, so any installed
        plugin can be referenced by its registered name.

        Strict (#1806 PR 1a): the entry keys are exactly :data:`ENTRY_KEYS`, ``primary`` is a real bool,
        ``hardware_id`` is a quoted 16-hex id or ``unpinned`` (required when ``managed``, i.e. the file has a
        ``_format_version``), and a ``reachy_mini`` entry's ``config`` passes :func:`check_reachy_media`. Every
        violation is a :class:`~maxim.exceptions.RobotsConfigError`; nothing is coerced.
        """
        unknown = sorted(str(k) for k in data if k not in ENTRY_KEYS)
        if unknown:
            raise _entry_error(robot_id, f"unknown key(s) {unknown}; allowed: {sorted(ENTRY_KEYS)}")
        if "config" in data and not isinstance(data["config"], dict):
            # ``config: null`` (an empty ``config:`` line) or a scalar would
            # otherwise TypeError in dict() or silently drop every key.
            raise _entry_error(
                robot_id,
                f"'config' must be a mapping, got {type(data['config']).__name__} "
                "(an empty 'config:' line is null; remove it or give it keys)",
            )
        config = dict(data.get("config", {}))
        misplaced = [k for k in _ENTRY_ONLY_KEYS if k in config]
        if misplaced:
            raise _entry_error(robot_id, f"{misplaced} belong at the entry level, not under 'config:'")
        return cls(
            robot_id=robot_id,
            robot_type=data.get("type", "reachy_mini"),
            primary=data.get("primary", False),
            config=config,
            hardware_id=_parse_pin(robot_id, data, managed=managed),
            hostname=data.get("hostname"),
            launch=_parse_launch(robot_id, data.get("launch")),
            kind=data.get("kind", "robot"),
        )


def _is_writer_managed(data: dict[str, Any]) -> bool:
    """Whether the file is writer-managed (has a ``_format_version``); refuses a malformed or newer one.

    Absent means a hand-written legacy file, accepted without :func:`check_format_version`'s "re-save" warning,
    which it always emits for an absent key. A present value must be a quoted string (an unquoted ``2.0`` is a
    float, and must not read as legacy). A newer version is refused, modelled on
    ``config_writer._refuse_if_newer``: this build cannot know what the newer keys mean.
    """
    if "_format_version" not in data:
        return False
    raw = data["_format_version"]
    if not isinstance(raw, str) or not raw:
        raise RobotsConfigError(f"robots.yaml: _format_version must be a quoted string, got {raw!r} (quote it)")
    if re.fullmatch(r"[0-9]+\.[0-9]+", raw) is None:
        raise RobotsConfigError(f"robots.yaml: _format_version {raw!r} is not a major.minor version such as '1.0'")
    major, minor = (int(p) for p in raw.split("."))
    known = tuple(int(p) for p in ROBOTS_FORMAT_VERSION.split("."))
    if (major, minor) > known:
        raise RobotsConfigError(
            f"robots.yaml was written by a newer Maxim (_format_version {raw!r}; this build reads "
            f"{ROBOTS_FORMAT_VERSION}). Upgrade Maxim: this build cannot know what the newer keys mean"
        )
    if major < known[0]:
        raise RobotsConfigError(f"robots.yaml: _format_version {raw!r} is not a version any Maxim wrote")
    check_format_version(data, "robots_yaml", log=logger)
    return True


@dataclass
class RobotsConfig:
    """Configuration for all robots."""

    robots: list[RobotConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        """At most one primary and unique ids, on every construction (#1806 PR 1a review W1)."""
        ids = [r.robot_id for r in self.robots]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise RobotsConfigError(f"robots.yaml has more than one entry with the id {duplicates}")
        primaries = [r.robot_id for r in self.robots if r.primary]
        if len(primaries) > 1:
            # Pre-fix the first one silently won, so reordering the file switched robots.
            raise RobotsConfigError(f"robots.yaml marks more than one entry primary: {primaries}; keep one")

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
        unknown_root = sorted(str(k) for k in data if k not in ROOT_KEYS)
        if unknown_root:
            # A ``robotz:`` typo would otherwise mean no robots, and the run would connect with defaults.
            raise RobotsConfigError(
                f"robots.yaml: unknown top-level key(s) {unknown_root}; allowed: {sorted(ROOT_KEYS)}"
            )
        managed = _is_writer_managed(data)
        robots_data = data.get("robots")
        if robots_data is None:  # absent, or an empty ``robots:`` line: no entries
            robots_data = {}
        if not isinstance(robots_data, dict):
            raise RobotsConfigError(
                f"robots.yaml 'robots:' must be a mapping of robot id to entry, got {type(robots_data).__name__}"
            )
        robots = []
        for robot_id, robot_config in robots_data.items():
            if not isinstance(robot_id, str) or not robot_id:
                # YAML would make ``yes:`` the id 'True' and ``1:`` collide with ``'1':``; never coerce an id.
                raise RobotsConfigError(f"robots.yaml: a robot id must be a non-empty quoted name, got {robot_id!r}")
            if not isinstance(robot_config, dict):
                raise RobotsConfigError(
                    f"robots.yaml entry {robot_id!r} must be a mapping, got {type(robot_config).__name__}"
                )
            robots.append(RobotConfig.from_dict(robot_id, robot_config, managed=managed))
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
                data = yaml.load(f, Loader=_NoDuplicateKeysLoader)  # noqa: S506 - a SafeLoader subclass
        except (OSError, yaml.YAMLError, ValueError) as e:
            # ValueError: a file that is not UTF-8 (UnicodeDecodeError); scalar errors are typed by the loader.
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


def pinned_connect_refusal(entry: RobotConfig | None) -> RobotPinUnverifiable | None:
    """The refusal for connecting ``entry`` in this build, or ``None`` when it may connect (#1806 PR 1a, interim).

    PR 1a parses ``hardware_id:`` but does not yet verify it on connect; that check is PR 1b's. Until then a
    pinned entry is refused rather than connected unverified, so no build trusts a pin it has not checked. The
    live resolver, the bulk connect and ``maxim doctor`` all read this one message, so doctor never reports a
    robot ready that the run would refuse. Nothing writes ``hardware_id:`` before #1806 PR 2.
    """
    if entry is None or not entry.hardware_id.is_pinned:
        return None
    from maxim import __version__

    return RobotPinUnverifiable(
        f"robots.yaml entry {entry.robot_id!r} pins hardware_id {entry.hardware_id}, and this build (Maxim "
        f"{__version__}) cannot yet verify a pin on connect (#1806 PR 1b), so it refuses rather than connect "
        "unverified"
    )


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
    refusal = pinned_connect_refusal(entry)
    if refusal is not None:
        raise refusal
    if robot_type == "reachy_mini":
        # The file's own values passed at load; this covers the run's media_backend default when the entry does
        # not set one (or there is no entry), so a LOCAL backend never reaches a networked connect.
        connect_config = check_reachy_media(connect_config, where="the run's media_backend", error=InvalidConfigError)
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
        refusal = pinned_connect_refusal(robot_config)
        if refusal is not None:
            logger.error("Not connecting robot %s: %s (%s)", robot_config.robot_id, refusal, refusal.fix_hint)
            results[robot_config.robot_id] = False
            if not skip_on_failure:
                break
            continue
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
