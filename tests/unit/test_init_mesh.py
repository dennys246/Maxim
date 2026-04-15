"""Tests for maxim.peer.init_mesh (Plan 4 Stage C3.1).

Decision tree coverage matrix from the module docstring:

============ ============ ======== =====================================  ====
peer.yml     mesh.yml     --force  Action                                 Exit
============ ============ ======== =====================================  ====
absent       absent       —        nothing to convert                       1
absent       present      —        already-converted no-op                  0
present      absent       —        synthesize mesh.yml                      0
present      present      no       refuse with --force hint                 2
present      present      yes      backup + synthesize                      0
============ ============ ======== =====================================  ====

Plus regression guards for the load-bearing invariants:
- peer.yml is NEVER touched (role detection still works post-init)
- Synthesized mesh.yml round-trips through parse_mesh_config
- Backup file content matches original byte-for-byte
- mesh.yml has 0o600 perms after first write (POSIX only)
- Drain immediately works post-init (end-to-end integration)
"""

from __future__ import annotations

import os
import platform

import pytest

from maxim.peer.init_mesh import run_init_mesh
from maxim.peer.mesh_config import parse_mesh_config


VALID_PEER_YAML = "url: https://leader.example.com/v1\napi_key: sk-cluster-abc\n"
EXISTING_MESH_YAML = (
    "cluster_key: sk-existing\n"
    "self: leader-desk\n"
    "protocol_version: 1\n"
    "nodes:\n"
    "  - name: leader-desk\n"
    "    url: http://192.168.1.10:8099/v1\n"
    "    role: leader\n"
)


@pytest.fixture
def isolated_xdg(tmp_path, monkeypatch):
    """Each test gets its own ~/.config/maxim/."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("MAXIM_ROLE", "leader")
    from maxim.utils import paths

    paths._reset_caches()
    return tmp_path


def _peer_path(tmp_path):
    p = tmp_path / "config" / "maxim" / "peer.yml"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _mesh_path(tmp_path):
    p = tmp_path / "config" / "maxim" / "mesh.yml"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ─── decision tree row 1: nothing to convert ───────────────────────────


class TestNothingToConvert:
    def test_no_peer_no_mesh_returns_exit_1(self, isolated_xdg, capsys):
        rc = run_init_mesh([])
        err = capsys.readouterr().err
        assert rc == 1
        assert "Nothing to convert" in err
        assert "maxim peer connect" in err


# ─── decision tree row 2: peer.yml absent, mesh.yml present ────────────


class TestMeshAlreadyExistsNoOp:
    def test_only_mesh_returns_exit_0(self, isolated_xdg, capsys):
        _mesh_path(isolated_xdg).write_text(EXISTING_MESH_YAML)
        rc = run_init_mesh([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "already exists" in out
        assert "Nothing to do" in out

    def test_does_not_touch_existing_mesh(self, isolated_xdg):
        mesh_p = _mesh_path(isolated_xdg)
        mesh_p.write_text(EXISTING_MESH_YAML)
        original = mesh_p.read_text()
        run_init_mesh([])
        assert mesh_p.read_text() == original


# ─── decision tree row 3: happy path ───────────────────────────────────


class TestHappyPath:
    def test_synthesizes_mesh_from_peer(self, isolated_xdg, capsys):
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        rc = run_init_mesh([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Synthesized" in out

        mesh_p = _mesh_path(isolated_xdg)
        assert mesh_p.is_file()
        cfg = parse_mesh_config(mesh_p.read_text())
        assert cfg.cluster_key == "sk-cluster-abc"
        assert cfg.self_name == "leader"
        assert len(cfg.nodes) == 1
        assert cfg.nodes[0].name == "leader"
        assert cfg.nodes[0].url == "https://leader.example.com/v1"
        assert cfg.nodes[0].role == "leader"
        assert cfg.protocol_version == 1

    def test_peer_yml_preserved_post_init(self, isolated_xdg):
        """Load-bearing: runtime/role.py reads peer.yml existence
        as part of role detection. init-mesh MUST NOT touch it."""
        peer_p = _peer_path(isolated_xdg)
        peer_p.write_text(VALID_PEER_YAML)
        original_content = peer_p.read_text()
        original_mtime = peer_p.stat().st_mtime
        run_init_mesh([])
        assert peer_p.is_file()
        assert peer_p.read_text() == original_content
        # mtime should also be untouched (we never opened the file
        # for writing). Tolerate filesystem mtime granularity.
        assert peer_p.stat().st_mtime == original_mtime

    def test_synthesized_mesh_roundtrips_through_parser(self, isolated_xdg):
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        run_init_mesh([])
        mesh_p = _mesh_path(isolated_xdg)
        # Parse what we wrote, re-serialize, parse again — should be stable
        cfg1 = parse_mesh_config(mesh_p.read_text())
        cfg2 = parse_mesh_config(cfg1.to_yaml())
        assert cfg1 == cfg2

    @pytest.mark.skipif(platform.system() == "Windows", reason="POSIX-only chmod")
    def test_mesh_yml_perms_are_0600_on_first_write(self, isolated_xdg):
        """write_mesh_config chmods to 0o600 on first write because
        cluster_key is a secret per the C2 invariant."""
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        run_init_mesh([])
        mesh_p = _mesh_path(isolated_xdg)
        mode = os.stat(mesh_p).st_mode & 0o777
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_success_output_lists_next_steps(self, isolated_xdg, capsys):
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        run_init_mesh([])
        out = capsys.readouterr().out
        assert "list-drained" in out
        assert "list-nodes" in out
        assert "drain" in out


# ─── decision tree row 4: mesh exists, no --force ──────────────────────


class TestRefuseWithoutForce:
    def test_existing_mesh_refused_without_force(self, isolated_xdg, capsys):
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        _mesh_path(isolated_xdg).write_text(EXISTING_MESH_YAML)
        rc = run_init_mesh([])
        err = capsys.readouterr().err
        assert rc == 2
        assert "already exists" in err
        assert "--force" in err
        assert "mesh.yml.bak" in err

    def test_existing_mesh_unchanged_after_refusal(self, isolated_xdg):
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        mesh_p = _mesh_path(isolated_xdg)
        mesh_p.write_text(EXISTING_MESH_YAML)
        run_init_mesh([])
        assert mesh_p.read_text() == EXISTING_MESH_YAML

    def test_no_backup_created_on_refusal(self, isolated_xdg):
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        mesh_p = _mesh_path(isolated_xdg)
        mesh_p.write_text(EXISTING_MESH_YAML)
        run_init_mesh([])
        backup_p = mesh_p.with_suffix(mesh_p.suffix + ".bak")
        assert not backup_p.is_file()


# ─── decision tree row 5: mesh exists, --force ─────────────────────────


class TestForceOverwrite:
    def test_force_creates_backup_then_writes(self, isolated_xdg, capsys):
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        mesh_p = _mesh_path(isolated_xdg)
        mesh_p.write_text(EXISTING_MESH_YAML)
        rc = run_init_mesh(["--force"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Synthesized" in out
        assert "backed up" in out

        # New mesh.yml has the synthesized content, not the old
        new_cfg = parse_mesh_config(mesh_p.read_text())
        assert new_cfg.cluster_key == "sk-cluster-abc"
        assert new_cfg.nodes[0].url == "https://leader.example.com/v1"

    def test_backup_byte_for_byte_match(self, isolated_xdg):
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        mesh_p = _mesh_path(isolated_xdg)
        mesh_p.write_text(EXISTING_MESH_YAML)
        run_init_mesh(["--force"])
        backup_p = mesh_p.with_suffix(mesh_p.suffix + ".bak")
        assert backup_p.is_file()
        assert backup_p.read_text() == EXISTING_MESH_YAML

    def test_force_on_first_init_no_backup(self, isolated_xdg):
        """--force is harmless when mesh.yml doesn't exist yet."""
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        rc = run_init_mesh(["--force"])
        assert rc == 0
        backup_p = _mesh_path(isolated_xdg).with_suffix(".yml.bak")
        assert not backup_p.is_file()


# ─── option parsing ────────────────────────────────────────────────────


class TestOptionParsing:
    def test_help_short(self, isolated_xdg, capsys):
        rc = run_init_mesh(["-h"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "init-mesh" in out
        assert "--force" in out

    def test_help_long(self, isolated_xdg, capsys):
        rc = run_init_mesh(["--help"])
        assert rc == 0
        assert "init-mesh" in capsys.readouterr().out

    def test_unknown_option_rejected(self, isolated_xdg, capsys):
        rc = run_init_mesh(["--bogus"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Unknown option" in err


# ─── error paths ────────────────────────────────────────────────────────


class TestErrorPaths:
    def test_malformed_peer_yml_fails_before_touching_mesh(self, isolated_xdg, capsys):
        peer_p = _peer_path(isolated_xdg)
        # Missing required api_key field — read_peer_config returns None
        peer_p.write_text("url: https://x.example/v1\n")
        rc = run_init_mesh([])
        err = capsys.readouterr().err
        assert rc == 1
        assert "could not be parsed" in err
        # mesh.yml NOT created
        assert not _mesh_path(isolated_xdg).is_file()

    def test_backup_failure_aborts_before_overwrite(self, isolated_xdg, capsys, monkeypatch):
        """If shutil.copy2 fails (read-only filesystem, perms), init-mesh
        must refuse to proceed — we never want to destroy the original
        without a backup in place."""
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        mesh_p = _mesh_path(isolated_xdg)
        mesh_p.write_text(EXISTING_MESH_YAML)

        def _broken_copy(*args, **kwargs):
            raise OSError("simulated backup failure")

        monkeypatch.setattr("shutil.copy2", _broken_copy)
        rc = run_init_mesh(["--force"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "Failed to back up" in err
        # Original mesh.yml is untouched
        assert mesh_p.read_text() == EXISTING_MESH_YAML


# ─── integration: drain works immediately post init-mesh ───────────────


class TestEndToEndDrainPostInit:
    def test_drain_after_init_mesh(self, isolated_xdg):
        """Regression guard: the entire C3.1 motivation is "drain
        works on peer.yml-only installs after init-mesh." This test
        runs the actual sequence."""
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)

        # Step 1: init-mesh
        assert run_init_mesh([]) == 0

        # Step 2: drain the synthesized leader node
        from maxim.peer.drain_state import drain_node, read_drained_nodes

        # Use force_self because the only node IS self in a one-node mesh
        from maxim.peer.mesh_config import read_mesh_config

        mesh = read_mesh_config()
        assert mesh is not None
        known = {n.name for n in mesh.nodes}
        result = drain_node("leader", known, self_name=mesh.self_name, force_self=True)
        assert "leader" in result

        # Step 3: read it back
        read_result = read_drained_nodes(known)
        assert "leader" in read_result.drained
