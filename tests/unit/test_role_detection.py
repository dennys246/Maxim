"""Tests for runtime/role.py — Plan 2 R2a."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from maxim.runtime.role import (
    detect_and_apply_role,
    detect_role,
    migrate_persisted_model_file,
)


@pytest.fixture(autouse=True)
def _clean_role_env(monkeypatch):
    monkeypatch.delenv("MAXIM_ROLE", raising=False)


@pytest.fixture
def _no_config(monkeypatch):
    monkeypatch.setattr("maxim.runtime.role._peer_yml_exists", lambda: False)
    monkeypatch.setattr("maxim.runtime.role._mesh_yml_exists", lambda: False)


def test_env_var_wins(monkeypatch, _no_config):
    monkeypatch.setenv("MAXIM_ROLE", "peer")
    assert detect_role([]) == ("peer", "env_var")


def test_env_var_invalid_falls_through(monkeypatch, _no_config):
    monkeypatch.setenv("MAXIM_ROLE", "bogus")
    assert detect_role([]) == ("leader", "default")


def test_mesh_yml_beats_peer_yml(monkeypatch):
    monkeypatch.setattr("maxim.runtime.role._mesh_yml_exists", lambda: True)
    monkeypatch.setattr("maxim.runtime.role._peer_yml_exists", lambda: True)
    assert detect_role([]) == ("peer", "mesh_yml")


def test_peer_yml_exists(monkeypatch):
    monkeypatch.setattr("maxim.runtime.role._mesh_yml_exists", lambda: False)
    monkeypatch.setattr("maxim.runtime.role._peer_yml_exists", lambda: True)
    assert detect_role([]) == ("peer", "peer_yml")


def test_cli_flag_solo(_no_config):
    assert detect_role(["--llm", "mistral-7b"]) == ("solo", "cli_flag")


def test_cli_flag_equals_form(_no_config):
    assert detect_role(["--llm=mistral-7b"]) == ("solo", "cli_flag")


def test_default_leader(_no_config):
    assert detect_role([]) == ("leader", "default")


def test_apply_role_exports_env(monkeypatch, _no_config):
    assert os.environ.get("MAXIM_ROLE") is None
    role, _ = detect_role([])
    from maxim.runtime.role import apply_role

    apply_role(role, "default")
    assert os.environ["MAXIM_ROLE"] == "leader"


# ── Migration: all four pre-existing states ─────────────────────────────


@pytest.fixture
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    from maxim.utils.paths import _reset_caches

    _reset_caches()
    util = tmp_path / "util"
    util.mkdir(parents=True, exist_ok=True)
    yield tmp_path
    _reset_caches()


def _touch_old(tmp_path: Path, content: str = "mistral-7b") -> Path:
    old = tmp_path / "util" / "active_llm_model.txt"
    old.write_text(content)
    return old


def test_migration_peer_deletes(_isolated_home):
    old = _touch_old(_isolated_home)
    migrate_persisted_model_file("peer")
    assert not old.exists()
    assert not (_isolated_home / "util" / "active_llm_model.peer.txt").exists()


def test_migration_solo_renames(_isolated_home):
    old = _touch_old(_isolated_home)
    migrate_persisted_model_file("solo")
    assert not old.exists()
    new = _isolated_home / "util" / "active_llm_model.solo.txt"
    assert new.is_file()
    assert new.read_text() == "mistral-7b"


def test_migration_leader_renames(_isolated_home):
    old = _touch_old(_isolated_home)
    migrate_persisted_model_file("leader")
    assert not old.exists()
    new = _isolated_home / "util" / "active_llm_model.leader.txt"
    assert new.is_file()


def test_migration_no_old_file_noop(_isolated_home):
    migrate_persisted_model_file("leader")
    assert not (_isolated_home / "util" / "active_llm_model.leader.txt").exists()


def test_detect_and_apply_integration(monkeypatch, _isolated_home, _no_config):
    role = detect_and_apply_role([])
    assert role == "leader"
    assert os.environ["MAXIM_ROLE"] == "leader"
