"""#1403: an unreadable config dir is ONE typed, actionable error, never a traceback.

With ``$XDG_CONFIG_HOME/maxim`` at mode 000, every ``maxim`` command died with a raw
``PermissionError`` traceback: ``models/language/profile_loader.py::load_user_profiles`` probed
``profiles.yml`` with ``Path.is_file()`` at import time, and ``cli.main``'s handler re-imported
``models.language`` and re-raised. ``Path.is_file()`` raises on Python <=3.13 and returns False on
3.14, so on 3.14 the same state silently drops the operator's profiles and config instead (#1312's
class). Every config-dir probe now goes through ``utils/paths.py::config_file_present``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from maxim.exceptions import ConfigurationError

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
    os.chmod(cfg, 0)
    yield cfg
    os.chmod(cfg, 0o700)


@pytest.mark.xfail(strict=True, reason="#1403 red gate")
def test_config_file_present_tells_missing_from_unreadable(tmp_path, locked_config_dir):
    from maxim.utils.paths import config_file_present

    assert config_file_present(tmp_path / "nope.json") is False
    (tmp_path / "a_dir").mkdir()
    assert config_file_present(tmp_path / "a_dir") is False
    (tmp_path / "ok.json").write_text("{}")
    assert config_file_present(tmp_path / "ok.json") is True
    with pytest.raises(ConfigurationError) as exc:
        config_file_present(locked_config_dir / "config.json")
    assert str(locked_config_dir) in str(exc.value)
    assert f"chmod 700 {locked_config_dir}" in str(exc.value)


@pytest.mark.xfail(strict=True, reason="#1403 red gate")
def test_profiles_loader_refuses_typed(locked_config_dir):
    from maxim.models.language.profile_loader import load_user_profiles

    with pytest.raises(ConfigurationError, match="chmod 700"):
        load_user_profiles(path=locked_config_dir / "profiles.yml")


@pytest.mark.xfail(strict=True, reason="#1403 red gate")
def test_load_config_refuses_instead_of_defaulting(locked_config_dir):
    """On 3.14 ``is_file`` returned False here, so load_config silently served defaults."""
    from maxim.runtime.config_loader import load_config

    with pytest.raises(ConfigurationError, match="chmod 700"):
        load_config(locked_config_dir / "config.json")


@pytest.mark.xfail(strict=True, reason="#1403 red gate")
def test_role_probe_does_not_swallow_the_refusal(locked_config_dir):
    """``_peer_yml_exists`` caught every exception and said "no peer.yml": a peer whose config dir is
    unreadable silently detected as solo."""
    from maxim.runtime import role

    with pytest.raises(ConfigurationError):
        role._peer_yml_exists()


@pytest.mark.xfail(strict=True, reason="#1403 red gate")
def test_cli_prints_one_typed_error_not_a_traceback(locked_config_dir):
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
    assert f"chmod 700 {locked_config_dir}" in out
