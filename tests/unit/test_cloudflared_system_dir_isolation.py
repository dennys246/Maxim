"""The host's system cloudflared config never reaches a test (2026-11-05).

Role detection's rank 4 (``runtime/role.py::_cloudflared_config_exists``), the tunnel spawner
(``tunnel/daemon_spawner.py::resolve_config_path``) and the doctor's sync check
(``doctor/checks.py::check_tunnel_config_sync``) each read ``/etc/cloudflared/config.yml``. conftest isolates
``HOME`` and ``XDG_*`` but never ``/etc``, so on a leader dev box that has the real tunnel config (the owner's PC,
WSL) every role-dependent test resolved ``leader``: 25 failures and 3 errors in a full suite that CI, whose runners
have no such file, never saw.

The fix is one canonical reader, ``utils/paths.py::cloudflared_system_dir`` (default ``/etc/cloudflared``,
override ``MAXIM_CLOUDFLARED_SYSTEM_DIR``), and conftest exporting the override into its isolation root at import,
beside ``HOME``, so subprocesses inherit it the same way. These guards fail on ANY machine without the fix: they check
the mechanism, not whether this host happens to have the file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ENV = "MAXIM_CLOUDFLARED_SYSTEM_DIR"
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"


def test_the_suite_points_the_system_cloudflared_dir_into_its_isolation_root():
    from maxim.utils.paths import cloudflared_system_dir

    raw = os.environ.get(_ENV, "")
    assert raw, f"conftest must export {_ENV} beside HOME"
    isolation_root = Path(os.environ["HOME"]).parent
    assert Path(raw).parent == isolation_root
    assert cloudflared_system_dir() == Path(raw)
    assert cloudflared_system_dir() != Path("/etc/cloudflared")


def test_the_reader_defaults_to_etc_cloudflared_when_unset(monkeypatch):
    from maxim.utils.paths import cloudflared_system_dir

    monkeypatch.delenv(_ENV, raising=False)
    assert cloudflared_system_dir() == Path("/etc/cloudflared")


def test_a_relative_override_is_refused_not_silently_ignored(monkeypatch):
    from maxim.exceptions import ConfigurationError
    from maxim.utils.paths import cloudflared_system_dir

    monkeypatch.setenv(_ENV, "etc/cloudflared")
    with pytest.raises(ConfigurationError, match="must be an absolute path"):
        cloudflared_system_dir()


def test_the_strip_helpers_drop_overrides_but_keep_the_isolation(monkeypatch):
    """The in-process form (_orch_harness, test_tty_stdin) and the child form (without_maxim_overrides)."""
    from tests._child_env import drop_maxim_overrides, maxim_overrides, without_maxim_overrides

    environ = {"MAXIM_LLM_PROFILE": "operator", _ENV: "/iso/etc-cloudflared", "PATH": "/usr/bin"}
    drop_maxim_overrides(environ)
    assert environ == {_ENV: "/iso/etc-cloudflared", "PATH": "/usr/bin"}
    assert maxim_overrides({"MAXIM_X": "1", _ENV: "/iso"}) == {"MAXIM_X": "1"}

    monkeypatch.setenv("MAXIM_LLM_PROFILE", "operator")
    child = without_maxim_overrides()
    assert "MAXIM_LLM_PROFILE" not in child
    assert child[_ENV] == os.environ[_ENV]


@pytest.mark.parametrize("name", ["config.yml", "config.yaml"])
def test_role_detection_reads_the_system_config_through_the_reader(tmp_path, monkeypatch, name):
    import maxim.runtime.role as role

    monkeypatch.setenv(_ENV, str(tmp_path))
    assert role._cloudflared_config_exists() is None
    (tmp_path / name).write_text("tunnel: x\n")
    assert role._cloudflared_config_exists() == tmp_path / name


def test_the_tunnel_spawner_resolves_the_system_config_through_the_reader(tmp_path, monkeypatch):
    from maxim.tunnel import daemon_spawner

    monkeypatch.setattr(daemon_spawner, "USER_CONFIG_PATH", tmp_path / "no-user.yml")
    monkeypatch.setenv(_ENV, str(tmp_path / "etc"))
    assert daemon_spawner.resolve_config_path() is None
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "config.yml").write_text("tunnel: x\n")
    assert daemon_spawner.resolve_config_path() == tmp_path / "etc" / "config.yml"


def test_the_doctor_sync_check_compares_against_the_reader(tmp_path, monkeypatch):
    import platform

    from maxim.doctor.checks import check_tunnel_config_sync

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    home = tmp_path / "home"
    (home / ".cloudflared").mkdir(parents=True)
    (home / ".cloudflared" / "config.yml").write_text("tunnel: user\n")
    monkeypatch.setenv("HOME", str(home))
    system_dir = tmp_path / "etc"
    system_dir.mkdir()
    monkeypatch.setenv(_ENV, str(system_dir))

    assert check_tunnel_config_sync().message.startswith("no systemd config")
    (system_dir / "config.yml").write_text("tunnel: stale\n")
    result = check_tunnel_config_sync()
    assert result.status == "warn"
    assert str(system_dir / "config.yml") in result.message


def _child_role(env: dict[str, str]) -> str:
    code = "from maxim.runtime.role import detect_role; print(detect_role([])[0])"
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**env, "PYTHONPATH": str(_REPO_SRC)},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_a_child_process_inherits_the_isolation(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "MAXIM_ROLE"}
    assert env.get(_ENV), f"conftest must export {_ENV}, or a child process reads the host's /etc"
    assert _child_role(env) == "solo"
    env[_ENV] = str(tmp_path)
    (tmp_path / "config.yml").write_text("tunnel: x\n")
    assert _child_role(env) == "leader"
