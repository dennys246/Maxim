"""#1806 PR 0 (review A1): every config-dir file resolves through ``utils/paths.py::maxim_config_dir``.

The file tools' control-plane write deny (``tools/base.py::_control_plane_roots``) denies ONE config directory. That is
only sound while every reader and writer of a config-dir file agrees on where it is: the six resolvers below used to be
hand-copied APPDATA/XDG bodies, so one drifting copy (a new override in one of them) would silently move that file out
of the deny. This pins the agreement structurally, under every branch of the resolver, and pins the resolver's own
semantics (so the agreement cannot be met by all six drifting together).
"""

from __future__ import annotations

import platform
from pathlib import Path

import pytest


def _resolved_files() -> dict[str, Path]:
    from maxim.models.language.profile_loader import profiles_config_path
    from maxim.peer.config import peer_config_path
    from maxim.peer.mesh_config import mesh_config_path
    from maxim.runtime.config_loader import config_path
    from maxim.runtime.role import _mesh_yml_path
    from maxim.tunnel.keys import CONSOLE_KEY_NAME, MESH_KEY_NAME, key_file_path

    return {
        "peer.yml": peer_config_path(),
        "config.json": config_path(),
        "role mesh.yml": Path(_mesh_yml_path()),
        "profiles.yml": profiles_config_path(),
        "api_key": key_file_path(MESH_KEY_NAME),
        "console_token": key_file_path(CONSOLE_KEY_NAME),
        "mesh.yml": mesh_config_path(),
    }


def _assert_all_in(expected: Path) -> None:
    from maxim.tools.base import _control_plane_roots
    from maxim.utils.paths import maxim_config_dir

    assert maxim_config_dir() == expected
    for name, path in _resolved_files().items():
        assert path.parent == expected, f"{name} resolves outside maxim_config_dir(): {path}"
    assert str(expected) in _control_plane_roots()[0]


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    return home


@pytest.mark.parametrize("xdg", [None, ""])
def test_posix_without_xdg_is_home_dot_config(home: Path, monkeypatch, xdg: str | None) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    if xdg is not None:
        monkeypatch.setenv("XDG_CONFIG_HOME", xdg)  # empty counts as unset
    _assert_all_in(home / ".config" / "maxim")


def test_posix_honours_xdg_config_home(home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _assert_all_in(tmp_path / "xdg" / "maxim")


def test_windows_honours_appdata_and_ignores_xdg(home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _assert_all_in(tmp_path / "roaming" / "maxim")


@pytest.mark.parametrize("appdata", [None, ""])
def test_windows_without_appdata_is_home_appdata_roaming(home: Path, monkeypatch, appdata: str | None) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    if appdata is not None:
        monkeypatch.setenv("APPDATA", appdata)
    _assert_all_in(home / "AppData" / "Roaming" / "maxim")


def test_maxim_config_dir_never_creates_the_directory(home: Path, monkeypatch) -> None:
    from maxim.utils.paths import maxim_config_dir

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    maxim_config_dir()
    assert not (home / ".config").exists()


def test_data_home_if_resolved_has_no_side_effects(tmp_path: Path, monkeypatch) -> None:
    """None before :func:`data_home` runs (and nothing created); the cached path after."""
    from maxim.utils import paths

    target = tmp_path / "data"
    monkeypatch.setenv("MAXIM_DATA_HOME", str(target))
    paths._reset_caches()
    assert paths.data_home_if_resolved() is None
    assert not target.exists()
    assert paths.data_home() == target
    assert paths.data_home_if_resolved() == target
    paths._reset_caches()
