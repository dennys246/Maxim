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


# ── R2 review fixes ──────────────────────────────────────────────────────


def test_fix2_subcommand_suppresses_llm_flag_scan(_no_config):
    """Fix #2: `maxim tunnel --llm foo` must not flip role to solo."""
    assert detect_role(["tunnel", "--llm", "foo"]) == ("leader", "default")
    assert detect_role(["doctor", "--llm", "foo"]) == ("leader", "default")
    assert detect_role(["peer", "llm", "--llm", "foo"]) == ("leader", "default")


def test_fix2_bare_llm_still_detects_solo(_no_config):
    """Sanity: non-subcommand argv still routes --llm through to solo."""
    assert detect_role(["--llm", "mistral-7b"]) == ("solo", "cli_flag")
    assert detect_role(["--sim", "x", "--llm", "mistral-7b"]) == ("solo", "cli_flag")


def test_fix5_migration_destination_collision(_isolated_home):
    """Fix #5: if .leader.txt already exists, delete legacy, preserve target."""
    old = _touch_old(_isolated_home, content="legacy-model")
    new = _isolated_home / "util" / "active_llm_model.leader.txt"
    new.write_text("role-specific-model")
    migrate_persisted_model_file("leader")
    assert not old.exists()
    assert new.read_text() == "role-specific-model"  # preserved, not clobbered


def test_fix11_role_source_field_name(_no_config):
    """Fix #11: log event uses `role_source`, not `source`, to avoid
    colliding with StructuredFormatter's `s` top-level short-key."""
    from maxim.runtime.role import apply_role
    from maxim.utils.structured_logging import get_abstraction_buffer

    buf = get_abstraction_buffer()
    buf.clear()
    apply_role("leader", "default")
    recent = buf.get_by_event("role_detected", n=5)
    assert recent, "role_detected event not emitted"
    data = recent[-1].get("data", recent[-1])  # compact shape may nest data
    # Flatten — the compact dict can surface the data keys at the top.
    haystack = {**recent[-1], **(data if isinstance(data, dict) else {})}
    assert "role_source" in haystack or haystack.get("role_source") == "default"
    assert haystack.get("role_source") == "default"


def test_fix7_lazy_migration_in_model_state_file(_isolated_home):
    """Fix #7: library-import users who never go through cli.py::main
    still get the legacy file migrated on first _model_state_file() call."""
    import maxim.runtime.llm_server as llm_server_mod

    # Seed legacy file
    old = _isolated_home / "util" / "active_llm_model.txt"
    old.write_text("legacy-model")
    # Reset the idempotent guard so the test runs fresh
    llm_server_mod._LAZY_MIGRATION_DONE = False
    # Ensure role defaults to leader (MAXIM_ROLE unset via autouse fixture)
    path = llm_server_mod._model_state_file()
    assert path.name == "active_llm_model.leader.txt"
    # Legacy file should have been renamed by the lazy migration.
    assert not old.exists()
    assert path.exists()
    assert path.read_text() == "legacy-model"


def test_fix9_role_divergence_warns(monkeypatch, _isolated_home, _no_config, caplog):
    """Fix #9: explicit warning when leader_mode + role disagree."""
    import logging as _logging

    from maxim.runtime import leader_mode
    from maxim.runtime.leader_mode import RoleDecision

    monkeypatch.setattr(
        leader_mode,
        "detect_role",
        lambda: RoleDecision(role="client", bind_host="127.0.0.1", reason="forced client"),
    )
    with caplog.at_level(_logging.WARNING, logger="maxim.runtime.role"):
        detect_and_apply_role([])  # role.py says leader, leader_mode says client → warn
    assert any(
        "divergence" in rec.message.lower() or "role_divergence" in rec.message for rec in caplog.records
    ) or any("divergence" in str(rec) for rec in caplog.records)
