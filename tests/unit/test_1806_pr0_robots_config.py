"""robots.yaml loads loud, resolves only under the data home, and is handed to the runtime once (#1806 PR 0).

Pins, each against the pre-fix behaviour it replaces:

- ``RobotsConfig.from_yaml`` / ``from_dict`` raise ``RobotsConfigError`` on a parse error, a non-mapping root,
  ``robots:`` or entry, and ``config: null`` (pre-fix: an EMPTY config, so a robot run connected with defaults);
  a missing or empty file is still no entries.
- ``find_config_file`` has no ``./robots.yaml`` cwd fallback (D11) and raises for a relative ``MAXIM_DATA_HOME``
  or an unresolvable data home (pre-fix: searched cwd only, with a warning).
- ``selfy.Maxim._resolve_robot_connection`` lets ``RobotsConfigError`` propagate (pre-fix: caught at DEBUG) and
  stores THE entry the connect used as ``_resolved_robot_config``, which ``_resolve_body_wiring`` reads without
  reloading.
- ``MAXIM_ROBOT_TIMEOUT`` garbage is a typed ``InvalidConfigError`` naming the variable (pre-fix: a bare
  ``ValueError`` from ``float()``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from maxim.exceptions import ConfigurationError, InvalidConfigError, RobotsConfigError
from maxim.hardware.config import RobotsConfig, find_config_file, load_robots_config


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    from maxim.utils.paths import _reset_caches

    home = tmp_path / "data_home"
    home.mkdir()
    monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
    _reset_caches()
    yield home
    _reset_caches()


def _write(home: Path, text: str) -> Path:
    path = home / "robots.yaml"
    path.write_text(text)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Load errors are typed and loud
# ─────────────────────────────────────────────────────────────────────────────


def test_robots_config_error_is_a_stable_configuration_error():
    import maxim.exceptions as exc

    assert issubclass(RobotsConfigError, ConfigurationError)
    assert "RobotsConfigError" in exc.__all__


@pytest.mark.parametrize(
    "text, needle",
    [
        ("robots: {primary: [unclosed\n", "cannot read"),
        ("- a\n- b\n", "must be a mapping"),
        ("robots: [a, b]\n", "'robots:' must be a mapping"),
        ("robots:\n  primary: reachy_mini\n", "entry 'primary' must be a mapping"),
        ("robots:\n  primary:\n    type: reachy_mini\n    config:\n", "'config' must be a mapping"),
        ("robots:\n  primary:\n    type: reachy_mini\n    config: 3\n", "'config' must be a mapping"),
    ],
    ids=["parse-error", "top-level-list", "robots-list", "scalar-entry", "config-null", "config-scalar"],
)
def test_a_malformed_file_raises_naming_the_path(tmp_path, text, needle):
    path = _write(tmp_path, text)
    with pytest.raises(RobotsConfigError) as info:
        RobotsConfig.from_yaml(path)
    assert needle in str(info.value)
    assert str(path) in str(info.value)


def test_from_dict_refuses_a_non_mapping_root():
    with pytest.raises(RobotsConfigError):
        RobotsConfig.from_dict(["primary"])


def test_a_missing_or_empty_file_is_no_entries(tmp_path):
    assert RobotsConfig.from_yaml(tmp_path / "absent.yaml").robots == []
    assert RobotsConfig.from_yaml(_write(tmp_path, "")).robots == []
    assert RobotsConfig.from_yaml(_write(tmp_path, "robots:\n")).robots == []


def test_a_good_file_still_loads(tmp_path):
    cfg = RobotsConfig.from_yaml(
        _write(tmp_path, "robots:\n  primary:\n    type: simulated\n    primary: true\n    config: {host: 10.0.0.9}\n")
    )
    assert [(r.robot_id, r.robot_type, r.primary, r.config) for r in cfg.robots] == [
        ("primary", "simulated", True, {"host": "10.0.0.9"})
    ]


def test_the_shipped_template_loads_and_carries_no_dead_timeout_key():
    template = Path(__file__).resolve().parents[2] / "data" / "util" / "robots.yaml"
    text = template.read_text()
    assert "timeout" not in text  # the controller-constructor filter drops it: it never did anything
    cfg = RobotsConfig.from_dict(yaml.safe_load(text))
    assert cfg.get_primary() is not None


# ─────────────────────────────────────────────────────────────────────────────
# Resolution: the data home only, never cwd, never a relative or unresolvable home
# ─────────────────────────────────────────────────────────────────────────────


def test_a_robots_yaml_in_cwd_is_never_picked(tmp_path, monkeypatch, data_home):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "robots.yaml").write_text("robots:\n  planted:\n    type: reachy_mini\n    config: {host: 10.6.6.6}\n")
    monkeypatch.chdir(cwd)
    assert find_config_file() is None
    assert load_robots_config().robots == []


def test_the_data_home_file_is_found(data_home):
    path = _write(data_home, "robots: {}\n")
    assert find_config_file() == path


def test_a_relative_data_home_raises(tmp_path, monkeypatch):
    from maxim.utils.paths import _reset_caches

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MAXIM_DATA_HOME", "relative/home")
    _reset_caches()
    try:
        with pytest.raises(RobotsConfigError, match="MAXIM_DATA_HOME"):
            find_config_file()
        with pytest.raises(RobotsConfigError):
            load_robots_config()
    finally:
        _reset_caches()


def test_a_relative_data_home_is_still_fine_for_data_home_itself(tmp_path, monkeypatch):
    """Scoped to robots.yaml: benchmarks call ``data_home()`` with relative work dirs (wire N2)."""
    from maxim.utils.paths import _reset_caches, data_home

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MAXIM_DATA_HOME", "relative/home")
    _reset_caches()
    try:
        assert data_home() == Path("relative/home")
    finally:
        _reset_caches()


def test_an_unresolvable_data_home_raises(monkeypatch):
    import maxim.utils.paths as paths

    def _boom():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(paths, "data_home", _boom)
    with pytest.raises(RobotsConfigError, match="cannot resolve the data home"):
        find_config_file()


def test_explicit_search_paths_are_searched_in_order(tmp_path):
    second = _write(tmp_path, "robots: {}\n")
    assert find_config_file([str(tmp_path / "nope.yaml"), str(second)]) == second


# ─────────────────────────────────────────────────────────────────────────────
# The runtime handoff: selfy resolves once, loud; agentic_runtime receives
# ─────────────────────────────────────────────────────────────────────────────


def _selfy_stub(robot_id=None, name="reachy_mini"):
    from maxim.embodied_runtime.selfy import Maxim

    stub = Maxim.__new__(Maxim)
    stub._simulation = False
    stub._robot_id = robot_id
    stub.name = name
    stub.log = logging.getLogger("test.1806_pr0")
    return stub


def test_a_broken_file_fails_the_robot_connect(data_home):
    _write(data_home, "robots:\n  primary: [broken\n")
    with pytest.raises(RobotsConfigError):
        _selfy_stub()._resolve_robot_connection("default")


def test_the_connect_and_the_body_wiring_read_one_entry(data_home, monkeypatch):
    import maxim.hardware.config as cfg_mod

    _write(
        data_home,
        "robots:\n"
        "  desk:\n"
        "    type: simulated\n"
        "    primary: true\n"
        "    config: {host: 10.0.0.9, body: bodies/reachy_mini, motor_binding: false}\n",
    )
    stub = _selfy_stub()
    robot_id, robot_type, connect_config = stub._resolve_robot_connection("default")
    assert (robot_id, robot_type) == ("reachy_mini", "simulated")
    assert connect_config["host"] == "10.0.0.9" and connect_config["media_backend"] == "default"
    entry = stub._resolved_robot_config
    assert entry is not None and entry.robot_id == "desk"

    def _no_reload(*a, **k):
        raise AssertionError("the runtime must not reload robots.yaml")

    monkeypatch.setattr(cfg_mod, "load_robots_config", _no_reload)
    ref, registry = stub._resolve_body_wiring(object(), object())
    assert ref == "bodies/reachy_mini" and registry is not None
    assert stub._resolved_robot_config is entry  # the body wiring did not replace the handed-over entry


def test_no_file_connects_with_defaults_and_no_entry(data_home):
    stub = _selfy_stub()
    robot_id, robot_type, connect_config = stub._resolve_robot_connection("default")
    assert (robot_id, robot_type) == ("reachy_mini", "reachy_mini")
    assert connect_config == {"robot_name": "reachy_mini", "media_backend": "default"}
    assert stub._resolved_robot_config is None
    assert stub._resolve_body_wiring(object(), object()) == (None, None)


def test_an_ambiguous_file_adopts_no_foreign_robot(data_home, caplog):
    _write(
        data_home,
        "robots:\n"
        "  alpha: {type: simulated, config: {host: 10.0.0.1, body: bodies/reachy_mini}}\n"
        "  beta: {type: simulated, config: {host: 10.0.0.2, body: bodies/reachy_mini}}\n",
    )
    stub = _selfy_stub(robot_id="gamma")
    with caplog.at_level(logging.WARNING):
        _, robot_type, connect_config = stub._resolve_robot_connection("default")
    assert robot_type == "reachy_mini" and "host" not in connect_config
    assert stub._resolved_robot_config is None
    assert any("not guessing" in r.getMessage() for r in caplog.records)
    assert stub._resolve_body_wiring(object(), object()) == (None, None)


# ─────────────────────────────────────────────────────────────────────────────
# MAXIM_ROBOT_TIMEOUT
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", ["abc", "nan", "inf", "0", "-5"])
def test_a_garbage_robot_timeout_is_a_typed_error_naming_the_variable(monkeypatch, raw):
    from maxim.hardware.config import robot_connect_timeout

    monkeypatch.setenv("MAXIM_ROBOT_TIMEOUT", raw)
    with pytest.raises(InvalidConfigError, match="MAXIM_ROBOT_TIMEOUT"):
        robot_connect_timeout(30.0)


def test_the_robot_timeout_reads_the_env_or_the_default(monkeypatch):
    from maxim.hardware.config import robot_connect_timeout

    assert robot_connect_timeout(30.0) == 30.0
    monkeypatch.setenv("MAXIM_ROBOT_TIMEOUT", " 4.5 ")
    assert robot_connect_timeout(30.0) == 4.5
