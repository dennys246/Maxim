"""#1403: an unreadable config dir is ONE typed, actionable error, never a traceback.

With ``$XDG_CONFIG_HOME/maxim`` at mode 000, every ``maxim`` command died with a raw
``PermissionError`` traceback: ``models/language/profile_loader.py::load_user_profiles`` probed
``profiles.yml`` with ``Path.is_file()`` at import time, and ``cli.main``'s handler re-imported
``models.language`` and re-raised. ``Path.is_file()`` raises on Python <=3.13 and returns False on
3.14, so on 3.14 the same state silently drops the operator's profiles and config instead (#1312's
class). Every config-dir probe now goes through ``utils/paths.py::config_file_present``.

Every module under test is imported HERE, before a fixture locks the dir. ``models.language`` used to load the user
profiles at import; since #1315 step 0a it loads them on the first lookup, and conftest makes that lookup at
collection, under the isolated (readable) dir.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from maxim import cli
from maxim.exceptions import ConfigDirUnreadable, ConfigurationError
from maxim.hivemind.registry import HiveRegistry
from maxim.models.language.profile_loader import load_user_profiles
from maxim.peer.config import read_peer_config
from maxim.peer.mesh_config import read_mesh_config
from maxim.runtime import config_loader, role
from maxim.utils.paths import config_file_present

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX permission bits; root ignores them",
)

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def locked_config_dir(tmp_path, monkeypatch):
    """An XDG config home whose ``maxim`` dir holds a profiles.yml and config.json but is mode 000."""
    xdg = tmp_path / "xdg"
    cfg = xdg / "maxim"
    cfg.mkdir(parents=True)
    (cfg / "profiles.yml").write_text("profiles: {}\n")
    (cfg / "config.json").write_text('{"_format_version": "1.0"}\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.delenv("MAXIM_ROLE", raising=False)
    config_loader.invalidate_config_cache()
    os.chmod(cfg, 0)
    yield cfg
    os.chmod(cfg, 0o700)
    config_loader.invalidate_config_cache()


def test_config_file_present_tells_missing_from_unreadable(tmp_path, locked_config_dir):
    assert config_file_present(tmp_path / "nope.json") is False
    (tmp_path / "a_dir").mkdir()
    assert config_file_present(tmp_path / "a_dir") is False
    (tmp_path / "ok.json").write_text("{}")
    assert config_file_present(tmp_path / "ok.json") is True
    with pytest.raises(ConfigDirUnreadable) as exc:
        config_file_present(locked_config_dir / "config.json")
    assert isinstance(exc.value, ConfigurationError)
    assert f"chmod 700 {locked_config_dir}" in str(exc.value)


def test_the_fix_names_the_ancestor_that_blocks_not_the_file_dir(tmp_path):
    """With ``$XDG_CONFIG_HOME`` itself locked, ``chmod 700 <xdg>/maxim`` cannot work: name ``<xdg>``."""
    xdg = tmp_path / "xdg"
    (xdg / "maxim").mkdir(parents=True)
    os.chmod(xdg, 0)
    try:
        with pytest.raises(ConfigDirUnreadable) as exc:
            config_file_present(xdg / "maxim" / "config.json")
    finally:
        os.chmod(xdg, 0o700)
    assert f"chmod 700 {xdg}" in str(exc.value)
    assert f"chmod 700 {xdg / 'maxim'}" not in str(exc.value)


def test_a_non_permission_error_gets_no_chmod_advice(tmp_path):
    loop = tmp_path / "config.json"
    loop.symlink_to(loop)
    with pytest.raises(ConfigDirUnreadable) as exc:
        config_file_present(loop)
    assert "chmod" not in str(exc.value)


def test_profiles_loader_refuses_typed(locked_config_dir):
    with pytest.raises(ConfigDirUnreadable, match="chmod 700"):
        load_user_profiles(path=locked_config_dir / "profiles.yml")


def test_load_config_refuses_instead_of_defaulting(locked_config_dir, monkeypatch):
    """On 3.14 ``is_file`` returned False here, so load_config silently served defaults. The migration shim is
    marked done so ITS probe cannot mask load_config's own."""
    monkeypatch.setattr(config_loader, "_migration_attempted", True)
    with pytest.raises(ConfigDirUnreadable, match="chmod 700"):
        config_loader.load_config(locked_config_dir / "config.json")


def test_the_peer_yml_migration_probe_refuses_too(locked_config_dir, monkeypatch):
    monkeypatch.setattr(config_loader, "_migration_attempted", False)
    with pytest.raises(ConfigDirUnreadable):
        config_loader._maybe_migrate_from_peer_yml(locked_config_dir / "config.json")


def test_resolve_setting_does_not_fall_back_to_defaults(locked_config_dir):
    """``resolve_setting`` swallowed every load failure into defaults: an unreadable dir silently dropped the
    operator's ``tools.deny`` (the console builds its permissions from it), failing open."""
    with pytest.raises(ConfigDirUnreadable):
        config_loader.resolve_setting("tools.deny")


@pytest.mark.parametrize("probe", ["_peer_yml_exists", "_mesh_yml_exists", "_config_json_role"])
def test_role_probes_do_not_swallow_the_refusal(locked_config_dir, probe):
    """Each probe caught every exception and answered "absent": a peer whose config dir is unreadable silently
    detected as solo."""
    with pytest.raises(ConfigDirUnreadable):
        getattr(role, probe)()


@pytest.mark.parametrize("loader", ["peer.yml", "mesh.yml", "hive.json"])
def test_every_declarative_loader_refuses_rather_than_reading_absent(locked_config_dir, loader):
    """Each loader read "missing" for an unreadable dir on 3.14 (``is_file`` returned False): the registry
    as empty, the mesh and peer config as unset. All three go through ``config_file_present``."""
    calls = {
        "peer.yml": read_peer_config,
        "mesh.yml": lambda: read_mesh_config(locked_config_dir / "mesh.yml"),
        "hive.json": lambda: HiveRegistry(locked_config_dir / "hive.json")._load(),
    }
    with pytest.raises(ConfigDirUnreadable):
        calls[loader]()


def test_cli_main_renders_one_typed_error_in_process(locked_config_dir, capsys, caplog):
    with caplog.at_level(logging.WARNING):
        assert cli.main(["--help"]) == 2
    captured = capsys.readouterr()
    out = captured.out + captured.err + caplog.text
    assert "Traceback" not in out
    assert out.count(f"chmod 700 {locked_config_dir}") == 1, out
    assert "role detection failed" not in out


def test_role_detection_still_warns_past_other_failures(monkeypatch, caplog):
    """The carve-out is ConfigDirUnreadable only: role detection's other failures keep warning (Plan 2 R2a)."""

    def _boom(argv):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(role, "detect_and_apply_role", _boom)
    with caplog.at_level(logging.WARNING):
        cli._detect_role_or_warn([])
    assert "role detection failed: probe exploded" in caplog.text


def test_cli_main_re_raises_other_errors_unmasked(monkeypatch):
    """A failure that is not a refusal re-raises as itself, even when importing BackendError also fails."""
    import builtins

    real_import = builtins.__import__

    def _no_types(name, *a, **k):
        if name == "maxim.models.language.types":
            raise ImportError("types unavailable")
        return real_import(name, *a, **k)

    def _impl(argv):
        raise ValueError("a real bug")

    monkeypatch.setattr(cli, "_main_impl", _impl)
    monkeypatch.setattr(builtins, "__import__", _no_types)
    with pytest.raises(ValueError, match="a real bug"):
        cli.main([])


def test_cli_subprocess_prints_one_typed_error_not_a_traceback(locked_config_dir):
    env = {**os.environ, "PYTHONPATH": str(_REPO_SRC)}
    proc = subprocess.run(
        [sys.executable, "-m", "maxim", "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    out = proc.stdout + proc.stderr
    assert "Traceback" not in out, out[-3000:]
    assert proc.returncode == 2, out[-3000:]
    assert out.count(f"chmod 700 {locked_config_dir}") == 1, out[-3000:]
