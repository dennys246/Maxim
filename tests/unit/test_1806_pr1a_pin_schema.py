"""robots.yaml entries are strict and typed, and a pinned entry refuses to connect until PR 1b (#1806 PR 1a).

Pins, each against the pre-fix behaviour it replaces:

- ``DevicePin``: the pin is a typed value, never ``None``; an unpinned pin matches nothing, a null id included.
- Strict entry keys: a typo (``hardware_idd:``) or ``hardware_id:`` under ``config:`` is a ``RobotsConfigError``
  (pre-fix: unknown keys were dropped, so a typo yielded a silently unpinned entry).
- Typed scalars: ``primary`` is a real bool, one primary at most (pre-fix: ``bool()`` coerced a quoted "false" to
  True, and the first of several primaries silently won); ``hardware_id`` is a quoted 16-hex id or ``unpinned``.
- ``_format_version``: absent is hand-written legacy, present makes ``hardware_id:`` required, malformed or newer
  is refused.
- Reachy ``connection_mode``/``tunnel``/``media_backend``: a LOCAL backend on anything but a non-tunnel
  ``localhost_only`` entry is refused at load and for the run's own default (owner rule, 2026-11-25; pre-fix it
  read the Maxim host's camera for a networked robot), values are canonicalised, unknown values refused.
- The interim refusal: the live resolver, the bulk connect and doctor all refuse a pinned entry with one message.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

import maxim
from maxim.exceptions import InvalidConfigError, RobotPinUnverifiable, RobotsConfigError
from maxim.hardware.config import (
    RobotConfig,
    RobotsConfig,
    check_reachy_media,
    connect_configured_robots,
    pinned_connect_refusal,
    resolve_robot_connection,
)
from maxim.hardware.devices_types import DevicePin

PIN = "0123456789abcdef"


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    from maxim.utils.paths import _reset_caches

    home = tmp_path / "data_home"
    home.mkdir()
    monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
    _reset_caches()
    yield home
    _reset_caches()


def _load(text: str) -> RobotsConfig:
    return RobotsConfig.from_dict(yaml.safe_load(text))


def _entry(body: str, *, header: str = "") -> str:
    return f"{header}robots:\n  bot:\n" + "".join(f"    {line}\n" for line in body.strip().splitlines())


# ─────────────────────────────────────────────────────────────────────────────
# DevicePin
# ─────────────────────────────────────────────────────────────────────────────


def test_unpinned_is_a_sentinel_that_matches_nothing():
    assert DevicePin.UNPINNED is not None
    assert not DevicePin.UNPINNED.is_pinned
    assert not DevicePin.UNPINNED.matches(None)
    assert not DevicePin.UNPINNED.matches(PIN)
    assert str(DevicePin.UNPINNED) == "unpinned"


def test_a_pin_matches_only_its_own_id():
    pin = DevicePin(PIN)
    assert pin.is_pinned and pin.matches(PIN)
    assert not pin.matches(None) and not pin.matches(PIN.upper()) and not pin.matches("f" * 16)
    assert pin == DevicePin(PIN, extra={"seen": 1})  # CC3 path (a): extra is outside equality


def test_a_pins_extra_is_read_only_and_cannot_shadow_a_field():
    with pytest.raises(TypeError):
        DevicePin.UNPINNED.extra["leak"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="collide"):
        DevicePin(PIN, extra={"hardware_id": "x"})


@pytest.mark.parametrize("bad", ["", "0123", PIN.upper(), PIN + "0", "g" * 16, 123])
def test_a_malformed_pin_cannot_be_built(bad):
    with pytest.raises(ValueError):
        DevicePin(bad)


# ─────────────────────────────────────────────────────────────────────────────
# Strict, typed entries
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "body, needle",
    [
        (f"hardware_idd: '{PIN}'", "unknown key(s) ['hardware_idd']"),
        (f"config: {{hardware_id: '{PIN}'}}", "belong at the entry level"),
        ("config: {hostname: bot.lan}", "belong at the entry level"),
        ("primary: 1", "'primary' must be true or false"),
        ("primary: 'false'", "'primary' must be true or false"),
        ("hardware_id: 1234567890123456", "quote it"),
        ("hardware_id: 0123", "quote it"),
        ("hardware_id: ''", "16 lowercase hex"),
        (f"hardware_id: '{PIN.upper()}'", "16 lowercase hex"),
        ("hardware_id:", "quote it"),
        ("kind: phone", "'kind' must be one of"),
        ("type: 3", "'type' must be a non-empty string"),
        ("hostname: ''", "'hostname' must be a non-empty name"),
        ("hostname: ' bot.lan'", "'hostname' must be a non-empty name"),
        ("launch: {operational_mode: singularity}", "never holds operational_mode"),
        ("launch: {mood: calm}", "unknown 'launch' key(s)"),
        ("launch: {interactive: 'yes'}", "'launch.interactive' must be a bool"),
        ("launch: [mode]", "'launch' must be a mapping"),
        ("kind: [robot]", "'kind' must be one of"),
    ],
)
def test_a_bad_entry_is_refused(body, needle):
    with pytest.raises(RobotsConfigError) as info:
        _load(_entry(body))
    assert needle in str(info.value)
    assert "'bot'" in str(info.value)


@pytest.mark.parametrize(
    "text, needle",
    [
        (
            f"robots:\n  bot:\n    hardware_id: '{PIN}'\n    hardware_id: unpinned\n",
            "found duplicate key 'hardware_id'",
        ),
        ("robots:\n  bot: {type: simulated}\n  bot: {type: reachy_mini}\n", "found duplicate key 'bot'"),
    ],
    ids=["duplicate-pin-line", "duplicate-entry"],
)
def test_a_duplicate_key_is_refused_not_last_wins(tmp_path, text, needle):
    path = tmp_path / "robots.yaml"
    path.write_text(text)
    with pytest.raises(RobotsConfigError, match=needle):
        RobotsConfig.from_yaml(path)


@pytest.mark.parametrize("text", ["robots:\n  yes: {}\n", "robots:\n  1: {}\n", "robots:\n  ~: {}\n"])
def test_a_robot_id_is_never_coerced(text):
    with pytest.raises(RobotsConfigError, match="a robot id must be a non-empty quoted name"):
        _load(text)


def test_an_unknown_root_key_is_refused():
    with pytest.raises(RobotsConfigError, match=r"unknown top-level key\(s\) \['robotz'\]"):
        _load("robotz:\n  bot: {}\n")


def test_more_than_one_primary_is_refused():
    text = "robots:\n  a: {primary: true}\n  b: {primary: true}\n"
    with pytest.raises(RobotsConfigError, match="more than one entry primary"):
        _load(text)


def test_a_full_entry_loads_typed():
    cfg = _load(
        _entry(
            f"""
type: reachy_mini
kind: robot
primary: yes
hardware_id: '{PIN}'
hostname: rosie.lan
launch: {{mode: exploration, llm: qwen, interactive: false}}
config: {{host: 10.0.0.9}}
"""
        )
    )
    (bot,) = cfg.robots
    assert bot.primary is True  # YAML 1.1 `yes` is a real bool and is accepted
    assert bot.hardware_id == DevicePin(PIN)
    assert bot.hostname == "rosie.lan"
    assert bot.launch == {"mode": "exploration", "llm": "qwen", "interactive": False}
    assert bot.kind == "robot"


def test_a_legacy_entry_without_a_pin_is_unpinned():
    (bot,) = _load(_entry("type: reachy_mini")).robots
    assert bot.hardware_id is DevicePin.UNPINNED
    assert bot.hostname is None and bot.launch == {}
    assert RobotConfig("x", "reachy_mini").hardware_id is DevicePin.UNPINNED  # the dataclass default too


def test_the_unpinned_literal_is_unpinned():
    (bot,) = _load(_entry("hardware_id: unpinned")).robots
    assert bot.hardware_id is DevicePin.UNPINNED


# ─────────────────────────────────────────────────────────────────────────────
# _format_version
# ─────────────────────────────────────────────────────────────────────────────


def test_a_writer_managed_file_requires_the_pin_line():
    with pytest.raises(RobotsConfigError, match="'hardware_id' is required"):
        _load(_entry("type: reachy_mini", header="_format_version: '1.0'\n"))
    (bot,) = _load(_entry("hardware_id: unpinned", header="_format_version: '1.0'\n")).robots
    assert bot.hardware_id is DevicePin.UNPINNED


@pytest.mark.parametrize(
    "header, needle",
    [
        ("_format_version: 1.0\n", "must be a quoted string"),
        ("_format_version: ''\n", "must be a quoted string"),
        ("_format_version: 'one'\n", "not a major.minor"),
        ("_format_version: '1.1'\n", "newer Maxim"),
        ("_format_version: '2.0'\n", "newer Maxim"),
        ("_format_version: '0.9'\n", "not a version any Maxim wrote"),
        ("_format_version: ' 1.0'\n", "not a major.minor"),
        ("_format_version: '1.0.9'\n", "not a major.minor"),
        ("_format_version: '\u0661.0'\n", "not a major.minor"),  # int() accepts Arabic-Indic digits
    ],
)
def test_a_bad_or_newer_format_version_is_refused(header, needle):
    with pytest.raises(RobotsConfigError, match=needle):
        _load(_entry("hardware_id: unpinned", header=header))


def test_neither_a_legacy_nor_a_current_file_nags(caplog):
    from maxim.utils.format_version import reset_warned_types

    reset_warned_types()
    with caplog.at_level(logging.WARNING):
        _load(_entry("type: reachy_mini"))
        _load(_entry("hardware_id: unpinned", header="_format_version: '1.0'\n"))
    assert "_format_version" not in caplog.text
    reset_warned_types()


# ─────────────────────────────────────────────────────────────────────────────
# Reachy connection_mode / tunnel / media_backend
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "config",
    [
        "{connection_mode: network, media_backend: local}",
        "{connection_mode: auto, media_backend: local}",
        "{media_backend: gstreamer}",  # an absent mode is network
        "{connection_mode: Network, media_backend: local}",
        "{connection_mode: localhost_only, tunnel: true, host: 10.0.0.9, media_backend: local}",
        "{connection_mode: localhost_only, media_backend: sounddevice_opencv, tunnel: true, host: h}",
    ],
    ids=["network", "auto", "absent-gstreamer", "Network", "tunnel-local", "tunnel-alias"],
)
def test_a_local_backend_on_a_networked_entry_is_refused(config):
    with pytest.raises(RobotsConfigError, match="reads THIS machine's camera"):
        _load(_entry(f"config: {config}"))


@pytest.mark.parametrize(
    "config, needle",
    [
        ("{connection_mode: bogus}", "'connection_mode' must be one of"),
        ("{connection_mode: true}", "'connection_mode' must be one of"),
        ("{tunnel: 'true', connection_mode: localhost_only, media_backend: local}", "'tunnel' must be true or false"),
        ("{tunnel: 1}", "'tunnel' must be true or false"),
        ("{media_backend: no}", "'media_backend' must be one of"),  # YAML `no` is a bool, not no_media
        ("{media_backend: ' no_media'}", "'media_backend' must be one of"),
        ("{media_backend: opencv}", "'media_backend' must be one of"),
        ("{media_backend: null}", "'media_backend' must be one of"),  # the SDK would call None.lower()
        ("{connection_mode: null}", "'connection_mode' must be one of"),
    ],
)
def test_an_unknown_or_mistyped_reachy_value_is_refused(config, needle):
    with pytest.raises(RobotsConfigError, match=needle):
        _load(_entry(f"config: {config}"))


def test_local_media_is_allowed_onboard_and_values_are_canonical():
    (bot,) = _load(_entry("config: {connection_mode: LOCALHOST_ONLY, media_backend: Local}")).robots
    assert bot.config == {"connection_mode": "localhost_only", "media_backend": "local"}
    (bot,) = _load(_entry("config: {connection_mode: Network, media_backend: WebRTC}")).robots
    assert bot.config == {"connection_mode": "network", "media_backend": "webrtc"}


def test_a_non_reachy_entry_keeps_its_own_keys():
    (bot,) = _load(_entry("type: simulated\nconfig: {connection_mode: whatever, media_backend: local}")).robots
    assert bot.config == {"connection_mode": "whatever", "media_backend": "local"}


def test_the_check_returns_a_copy():
    raw = {"connection_mode": "Network"}
    assert check_reachy_media(raw, where="x") == {"connection_mode": "network"}
    assert raw == {"connection_mode": "Network"}


# ─────────────────────────────────────────────────────────────────────────────
# The live resolver: the run's media default, and the interim pinned refusal
# ─────────────────────────────────────────────────────────────────────────────

_LOG = logging.getLogger("test_1806_pr1a")


def test_the_runs_local_default_is_refused_with_no_robots_yaml(data_home):
    with pytest.raises(InvalidConfigError, match="the run's media_backend"):
        resolve_robot_connection("bot", "reachy_mini", "local", log=_LOG)


def test_the_runs_local_default_is_allowed_for_an_onboard_entry(data_home):
    (data_home / "robots.yaml").write_text(_entry("config: {connection_mode: localhost_only}"))
    _type, config, _entry_found = resolve_robot_connection("bot", "reachy_mini", "Local", log=_LOG)
    assert config["media_backend"] == "local"


def test_the_files_media_backend_wins_and_is_not_rechecked_as_the_default(data_home):
    (data_home / "robots.yaml").write_text(_entry("config: {media_backend: no_media}"))
    _type, config, _ = resolve_robot_connection("bot", "reachy_mini", "local", log=_LOG)
    assert config["media_backend"] == "no_media"


def test_a_pinned_entry_is_refused_by_the_live_resolver(data_home):
    (data_home / "robots.yaml").write_text(_entry(f"hardware_id: '{PIN}'\nconfig: {{host: 10.0.0.9}}"))
    with pytest.raises(RobotsConfigError) as info:
        resolve_robot_connection("bot", "reachy_mini", "default", log=_LOG)
    assert isinstance(info.value, RobotPinUnverifiable)  # a RobotsConfigError subclass: the CLI's exit-2 path
    assert str(info.value) == str(pinned_connect_refusal(load_one(data_home)))
    assert "cannot yet verify a pin" in str(info.value) and maxim.__version__ in str(info.value)
    assert "hardware_id: unpinned" in info.value.fix_hint
    assert "hardware_id: unpinned" not in str(info.value)  # the CLI prints the hint once, on its own line


def test_an_unpinned_entry_still_connects(data_home):
    (data_home / "robots.yaml").write_text(_entry("hardware_id: unpinned\nconfig: {host: 10.0.0.9}"))
    _type, config, entry = resolve_robot_connection("bot", "reachy_mini", "default", log=_LOG)
    assert config["host"] == "10.0.0.9" and entry is not None


def load_one(home: Path) -> RobotConfig:
    from maxim.hardware.config import load_robots_config

    (entry,) = load_robots_config().robots
    return entry


def test_the_bulk_connect_never_dials_a_pinned_entry():
    calls: list[str] = []

    class _Registry:
        def connect_robot(self, *, robot_id, **_kw):
            calls.append(robot_id)
            return object()

    cfg = _load(f"robots:\n  pinned: {{hardware_id: '{PIN}', type: simulated}}\n  free: {{type: simulated}}\n")
    results = connect_configured_robots(_Registry(), cfg)  # type: ignore[arg-type]
    assert results == {"pinned": False, "free": True}
    assert calls == ["free"]


def test_doctor_reports_a_pinned_entry_as_refused_without_probing_it(monkeypatch):
    import socket

    from maxim.doctor import checks

    def _no_probe(*_a, **_k):
        raise AssertionError("doctor probed an entry the run would refuse")

    monkeypatch.setattr(socket, "gethostbyname", _no_probe)
    monkeypatch.setattr(checks, "_check_one_robot", _no_probe)
    (bot,) = _load(_entry(f"hardware_id: '{PIN}'\nconfig: {{host: 10.0.0.9}}")).robots
    row = checks._check_one_robot_never_raises(None, bot)  # type: ignore[arg-type]
    assert row.status == "warn"
    refusal = pinned_connect_refusal(bot)
    assert refusal is not None
    assert (row.message, row.fix) == (str(refusal), refusal.fix_hint)


# ─────────────────────────────────────────────────────────────────────────────
# The invariants hold on direct construction too (review W1): the public export cannot bypass them
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"config": {"media_backend": "local", "connection_mode": "network"}}, "reads THIS machine's camera"),
        ({"hardware_id": PIN}, "must be a DevicePin"),
        ({"primary": "false"}, "'primary' must be true or false"),
        ({"kind": "phone"}, "'kind' must be one of"),
        ({"config": None}, "'config' must be a mapping"),
        ({"robot_id": ""}, "a robot id must be a non-empty name"),
        ({"robot_id": 1}, "a robot id must be a non-empty name"),
    ],
)
def test_a_directly_built_entry_is_checked(kwargs, needle):
    kwargs = {"robot_id": "y", **kwargs}
    with pytest.raises(RobotsConfigError, match=needle):
        RobotConfig(robot_type="reachy_mini", **kwargs)


def test_a_pinned_entry_copies_pickles_and_serialises():
    import copy
    import dataclasses
    import pickle

    entry = RobotConfig("y", "reachy_mini", hardware_id=DevicePin(PIN, extra={"seen": 1}))
    for clone in (copy.deepcopy(entry), pickle.loads(pickle.dumps(entry))):
        assert clone == entry and clone.hardware_id.extra == {"seen": 1}
        with pytest.raises(TypeError):
            clone.hardware_id.extra["leak"] = 2  # type: ignore[index]
    assert dataclasses.asdict(entry)["hardware_id"] == {"hardware_id": PIN, "extra": {"seen": 1}}
    assert copy.deepcopy(DevicePin.UNPINNED) == DevicePin.UNPINNED


@pytest.mark.parametrize(
    "text, needle",
    [
        ("robots:\n  bot: {<<: {type: simulated, type: reachy_mini}}\n", "found duplicate key 'type'"),
        ("robots:\n  bot: {<<: [{type: simulated, type: reachy_mini}]}\n", "found duplicate key 'type'"),
        ("a: &a {type: simulated}\nrobots:\n  bot: {<<: *a, <<: *a}\n", "found a second merge key"),
    ],
    ids=["inline-merge", "sequence-merge", "two-merge-keys"],
)
def test_a_duplicate_inside_a_merge_is_refused(tmp_path, text, needle):
    path = tmp_path / "robots.yaml"
    path.write_text(text)
    with pytest.raises(RobotsConfigError, match=needle):
        RobotsConfig.from_yaml(path)


@pytest.mark.parametrize(
    "text",
    [
        "b: &b {x: 1}\ns: &s {<<: *b, x: 2}\na: {<<: *s}\n",  # a chained merge (delta round 4 G1)
        "b: &b {x: 1}\ns: &s {<<: *b, x: 2}\na: {<<: [*s]}\n",
        "a: {<<: &s {<<: &b {x: 1}, x: 2}}\nc: {<<: *s}\n",
        "a: &a {<<: *a}\n",  # a self-referencing merge: no RecursionError (G2)
        "a: &a {k: *a}\n",  # a recursive structure still builds (G3)
        "b: &b {x: 1}\nc: &c {y: 2}\na: {<<: [*b, *c], x: 3}\n",
    ],
    ids=["chained", "chained-seq", "inline-anchors", "self-merge", "recursive", "seq-override"],
)
def test_legitimate_yaml_loads_exactly_as_safe_load_does(text):
    from maxim.hardware.config import _NoDuplicateKeysLoader

    ours = yaml.load(text, Loader=_NoDuplicateKeysLoader)  # noqa: S506 - a SafeLoader subclass
    theirs = yaml.safe_load(text)
    assert repr(ours) == repr(theirs)


@pytest.mark.parametrize(
    "text",
    [
        "robots: {!!set q: {}}\n",
        "robots:\n  ? [a]\n  : {}\n",
        "robots: {!!int abc: {}}\n",
        "robots: {bot: {primary: !!bool maybe}}\n",
        "robots: {bot: {launch: {llm: !!timestamp abc}}}\n",
        "robots: {bot: {config: {x: !!int ''}}}\n",
        "robots: {bot: {config: {x: !!float ''}}}\n",
    ],
    ids=[
        "collection-tagged-key",
        "collection-key",
        "bad-int-key",
        "bad-bool",
        "bad-timestamp",
        "empty-int",
        "empty-float",
    ],
)
def test_an_unusable_key_is_a_typed_refusal(tmp_path, text):
    path = tmp_path / "robots.yaml"
    path.write_text(text)
    with pytest.raises(RobotsConfigError, match="cannot read robot config"):
        RobotsConfig.from_yaml(path)


def test_a_chained_merge_in_robots_yaml_loads(tmp_path):
    path = tmp_path / "robots.yaml"
    path.write_text(
        "robots:\n  base: &b {type: simulated}\n  mid: &m {<<: *b, primary: true}\n  bot: {<<: *m, primary: false}\n"
    )
    cfg = RobotsConfig.from_yaml(path)
    assert [r.robot_id for r in cfg.robots if r.primary] == ["mid"]


def test_a_yaml_merge_override_is_not_a_duplicate(tmp_path):
    path = tmp_path / "robots.yaml"
    path.write_text(
        "robots:\n  base: &b {type: simulated, config: &c {host: a}}\n"
        "  bot: {<<: *b, primary: true, config: {<<: *c, host: b}}\n"
    )
    cfg = RobotsConfig.from_yaml(path)
    bot = cfg.get("bot")
    assert bot is not None and bot.primary is True and bot.config == {"host": "b"}


def test_a_directly_built_registry_is_checked():
    a, b = RobotConfig("a", "simulated", True), RobotConfig("b", "simulated", True)
    with pytest.raises(RobotsConfigError, match="more than one entry primary"):
        RobotsConfig(robots=[a, b])
    with pytest.raises(RobotsConfigError, match="more than one entry with the id"):
        RobotsConfig(robots=[a, RobotConfig("a", "simulated")])
