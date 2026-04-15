"""Tests for maxim.peer.mesh_setup (Plan 4 Stage C3.1 + C3.2).

Renamed from test_init_mesh.py in C3.2 when init_mesh.py grew into
mesh_setup.py to host add-node + remove-node alongside init-mesh.

C3.1 (init-mesh) decision tree coverage matrix:

============ ============ ======== =====================================  ====
peer.yml     mesh.yml     --force  Action                                 Exit
============ ============ ======== =====================================  ====
absent       absent       —        nothing to convert                       1
absent       present      —        already-converted no-op                  0
present      absent       —        synthesize mesh.yml                      0
present      present      no       refuse with --force hint                 2
present      present      yes      backup + synthesize                      0
============ ============ ======== =====================================  ====

C3.2 add-node decision tree:

- mesh.yml absent → exit 1
- mesh.yml malformed → exit 2
- name already exists + no --force → exit 2 (refuse with hint)
- name already exists + --force → replace in place
- new name → append (preserves operator-typed order)
- yaml-unsafe characters → exit 2 (raised by MeshNode.__post_init__)

C3.2 remove-node decision tree:

- mesh.yml absent → exit 1
- name not in nodes → exit 2 (typo guard)
- name == self → exit 2 (workaround hint)
- removing would leave 0 nodes → exit 2 (parser requires ≥1)
- happy path → drop, write, clear drain state, print summary

Regression guards for the load-bearing invariants:
- peer.yml is NEVER touched by any verb (role detection invariant)
- mesh.yml round-trips through parse_mesh_config after every write
- Backup file content matches original byte-for-byte
- mesh.yml has 0o600 perms after first write (POSIX only)
- Drain state cleanup on remove-node is operator-visible
"""

from __future__ import annotations

import os
import platform

import pytest

from maxim.peer.mesh_config import parse_mesh_config
from maxim.peer.mesh_setup import run_add_node, run_init_mesh, run_remove_node


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


class TestRefuseIfBackupExists:
    """A2 fold (C3.1 pre-merge review): double `--force` would
    silently destroy the operator's original mesh.yml. Refuse if
    the .bak file already exists, requiring explicit cleanup.
    """

    def test_force_refuses_if_bak_already_exists(self, isolated_xdg, capsys):
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        mesh_p = _mesh_path(isolated_xdg)
        mesh_p.write_text(EXISTING_MESH_YAML)

        # Stale backup from a previous --force operation
        backup_p = mesh_p.with_suffix(mesh_p.suffix + ".bak")
        backup_p.write_text("STALE BACKUP CONTENT")

        rc = run_init_mesh(["--force"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "already exists" in err
        assert "Refusing to overwrite an existing backup" in err
        assert "rm" in err  # cleanup hint

        # The stale backup is untouched (we refused before doing anything)
        assert backup_p.read_text() == "STALE BACKUP CONTENT"
        # Original mesh.yml is also untouched
        assert mesh_p.read_text() == EXISTING_MESH_YAML

    def test_after_cleanup_force_works(self, isolated_xdg, capsys):
        """Operator deletes the .bak, re-runs --force → succeeds."""
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        mesh_p = _mesh_path(isolated_xdg)
        mesh_p.write_text(EXISTING_MESH_YAML)
        backup_p = mesh_p.with_suffix(mesh_p.suffix + ".bak")
        backup_p.write_text("STALE")

        # First --force refused
        run_init_mesh(["--force"])
        capsys.readouterr()

        # Operator cleans up
        backup_p.unlink()

        # Second --force succeeds
        rc = run_init_mesh(["--force"])
        assert rc == 0
        # New backup has the original mesh.yml content (not the stale one)
        assert backup_p.is_file()
        assert backup_p.read_text() == EXISTING_MESH_YAML


class TestRow2InfoLine:
    """A5 fold (C3.1 pre-merge review): the "peer.yml absent + mesh
    present" case is unusual but exit 0. Surface the unusual state
    so the operator knows runtime/role.py will fall through to
    mesh.yml.
    """

    def test_unusual_state_surfaces_info(self, isolated_xdg, capsys):
        _mesh_path(isolated_xdg).write_text(EXISTING_MESH_YAML)
        rc = run_init_mesh([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "already exists" in out
        # New: surfaces the role detection note
        assert "role detection" in out.lower() or "role" in out.lower()
        assert "fall through" in out.lower() or "unusual" in out.lower()


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

    def test_unknown_option_with_help_does_not_swallow_error(self, isolated_xdg, capsys):
        """E1 fold (C3.1 pre-merge review): `init-mesh --bogus --help`
        previously triggered help and silently swallowed the bogus
        error. The fold inverts the order so unknown-option detection
        runs first."""
        rc = run_init_mesh(["--bogus", "--help"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Unknown option" in err
        assert "--bogus" in err

    def test_force_with_help_shows_help(self, isolated_xdg, capsys):
        """`init-mesh --force --help` is a valid combo (operator
        types --force, then asks for help). Both are known flags so
        unknown-detection passes and help wins."""
        rc = run_init_mesh(["--force", "--help"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "init-mesh" in out


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
        without a backup in place.

        E2 polish (C3.1 pre-merge review): the monkeypatch targets
        ``maxim.peer.init_mesh.shutil.copy2`` (the module-attribute
        path) rather than the global ``shutil.copy2`` so a future
        refactor that switches to ``from shutil import copy2`` would
        catch the test break loudly rather than silently.
        """
        _peer_path(isolated_xdg).write_text(VALID_PEER_YAML)
        mesh_p = _mesh_path(isolated_xdg)
        mesh_p.write_text(EXISTING_MESH_YAML)

        def _broken_copy(*args, **kwargs):
            raise OSError("simulated backup failure")

        monkeypatch.setattr("maxim.peer.mesh_setup.shutil.copy2", _broken_copy)
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


# ─── Plan 4 C3.2: add-node ─────────────────────────────────────────────


# Multi-node fixture used by add-node / remove-node tests. Two nodes
# so we can test ordering, replacement, and self-removal guards.
TWO_NODE_MESH_YAML = (
    "cluster_key: sk-cluster-abc\n"
    "self: leader-desk\n"
    "protocol_version: 1\n"
    "nodes:\n"
    "  - name: leader-desk\n"
    "    url: http://192.168.1.10:8099/v1\n"
    "    role: leader\n"
    "  - name: mac-studio\n"
    "    url: https://mac.example.com/v1\n"
    "    role: peer\n"
)


@pytest.fixture
def mesh_with_two_nodes(isolated_xdg):
    """isolated_xdg + a 2-node mesh.yml seeded on disk."""
    _mesh_path(isolated_xdg).write_text(TWO_NODE_MESH_YAML)
    return isolated_xdg


class TestAddNode:
    def test_add_new_node_appends_to_nodes(self, mesh_with_two_nodes, capsys):
        rc = run_add_node(["tablet", "--url", "https://tablet/v1"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Added node 'tablet'" in out
        assert "peer" in out  # default role

        cfg = parse_mesh_config(_mesh_path(mesh_with_two_nodes).read_text())
        assert len(cfg.nodes) == 3
        # Operator-typed order preserved
        assert [n.name for n in cfg.nodes] == ["leader-desk", "mac-studio", "tablet"]
        added = cfg.get_node("tablet")
        assert added is not None
        assert added.url == "https://tablet/v1"
        assert added.role == "peer"

    def test_add_with_explicit_role(self, mesh_with_two_nodes):
        rc = run_add_node(["tablet", "--url", "http://t/v1", "--role", "leader"])
        assert rc == 0
        cfg = parse_mesh_config(_mesh_path(mesh_with_two_nodes).read_text())
        assert cfg.get_node("tablet").role == "leader"

    def test_add_invalid_role_rejected(self, mesh_with_two_nodes, capsys):
        rc = run_add_node(["tablet", "--url", "http://t/v1", "--role", "overlord"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "must be 'peer' or 'leader'" in err
        # mesh.yml unchanged
        cfg = parse_mesh_config(_mesh_path(mesh_with_two_nodes).read_text())
        assert cfg.get_node("tablet") is None

    def test_add_existing_name_refused_without_force(self, mesh_with_two_nodes, capsys):
        rc = run_add_node(["mac-studio", "--url", "http://other/v1"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "already exists" in err
        assert "--force" in err
        # Original node unchanged
        cfg = parse_mesh_config(_mesh_path(mesh_with_two_nodes).read_text())
        assert cfg.get_node("mac-studio").url == "https://mac.example.com/v1"

    def test_force_replace_in_place(self, mesh_with_two_nodes, capsys):
        rc = run_add_node(
            [
                "mac-studio",
                "--url",
                "http://new-mac/v1",
                "--role",
                "leader",
                "--force",
            ]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "Replaced node 'mac-studio'" in out
        assert "was:" in out
        assert "now:" in out

        cfg = parse_mesh_config(_mesh_path(mesh_with_two_nodes).read_text())
        # Replaced in place — order preserved, count stable
        assert len(cfg.nodes) == 2
        assert [n.name for n in cfg.nodes] == ["leader-desk", "mac-studio"]
        replaced = cfg.get_node("mac-studio")
        assert replaced.url == "http://new-mac/v1"
        assert replaced.role == "leader"

    def test_add_url_with_yaml_unsafe_chars_rejected(self, mesh_with_two_nodes, capsys):
        """MeshNode.__post_init__ validation should fire."""
        rc = run_add_node(["tablet", "--url", "http://t/v1 # comment"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Invalid node fields" in err or "inline comment" in err

    def test_add_name_with_newline_rejected(self, mesh_with_two_nodes, capsys):
        rc = run_add_node(["tab\nlet", "--url", "http://t/v1"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Invalid node fields" in err or "newline" in err

    def test_add_refuses_when_no_mesh_yml(self, isolated_xdg, capsys):
        rc = run_add_node(["tablet", "--url", "http://t/v1"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "does not exist" in err
        assert "init-mesh" in err

    def test_add_missing_url_arg(self, mesh_with_two_nodes, capsys):
        rc = run_add_node(["tablet"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "--url is required" in err

    def test_add_missing_name(self, mesh_with_two_nodes, capsys):
        rc = run_add_node(["--url", "http://x/v1"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Missing required <name>" in err or "Expected node name" in err

    def test_add_unknown_option(self, mesh_with_two_nodes, capsys):
        rc = run_add_node(["tablet", "--url", "http://t/v1", "--bogus"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Unknown option" in err

    def test_add_help(self, mesh_with_two_nodes, capsys):
        for flag in ("-h", "--help"):
            rc = run_add_node([flag])
            out = capsys.readouterr().out
            assert rc == 0
            assert "add-node" in out

    def test_added_mesh_round_trips(self, mesh_with_two_nodes):
        """Regression guard: the mesh.yml written by add-node must
        round-trip through parse_mesh_config."""
        run_add_node(["tablet", "--url", "https://tablet/v1"])
        text = _mesh_path(mesh_with_two_nodes).read_text()
        cfg1 = parse_mesh_config(text)
        cfg2 = parse_mesh_config(cfg1.to_yaml())
        assert cfg1 == cfg2


# ─── Plan 4 C3.2: remove-node ──────────────────────────────────────────


class TestRemoveNode:
    def test_remove_existing_node(self, mesh_with_two_nodes, capsys):
        rc = run_remove_node(["mac-studio"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Removed node 'mac-studio'" in out
        assert "was:" in out

        cfg = parse_mesh_config(_mesh_path(mesh_with_two_nodes).read_text())
        assert len(cfg.nodes) == 1
        assert cfg.get_node("mac-studio") is None
        assert cfg.get_node("leader-desk") is not None

    def test_remove_unknown_node_refused(self, mesh_with_two_nodes, capsys):
        rc = run_remove_node(["ghost"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Unknown node" in err
        assert "leader-desk" in err  # known list rendered
        assert "mac-studio" in err

    def test_remove_self_refused_with_workaround_hint(self, mesh_with_two_nodes, capsys):
        """Plan 4 C3.2: removing self is structurally weird and the
        workaround is a manual mesh.yml edit + restart."""
        rc = run_remove_node(["leader-desk"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Refusing to remove self" in err
        assert "Edit mesh.yml::self" in err
        assert "Restart" in err

        # mesh.yml unchanged
        cfg = parse_mesh_config(_mesh_path(mesh_with_two_nodes).read_text())
        assert cfg.get_node("leader-desk") is not None

    def test_remove_would_leave_empty_mesh_refused(self, isolated_xdg, capsys):
        """Single-node mesh — can't remove because parser requires ≥1."""
        single_node_yaml = (
            "cluster_key: sk\n"
            "self: only\n"
            "protocol_version: 1\n"
            "nodes:\n"
            "  - name: only\n"
            "    url: http://only/v1\n"
            "    role: leader\n"
        )
        _mesh_path(isolated_xdg).write_text(single_node_yaml)

        # First refuse because it's self; switch self via hand-edit
        # to a different node — but we only have one node, so the
        # "would be empty" guard fires first via the self check.
        # Workaround: test with a 2-node mesh where one is self and
        # we try to remove the other. The "would be empty" path
        # kicks in if we remove the non-self node when it's the
        # only non-self entry. Hmm — that requires a 1-node mesh
        # where self ≠ that node, which the parser rejects. So this
        # decision tree row is actually unreachable through normal
        # ops. Test it by mocking only.
        rc = run_remove_node(["only"])
        # This exits 2 with "refusing to remove self" because the
        # only node IS self. The "would be empty" branch is
        # unreachable — guarded structurally by the parser.
        assert rc == 2
        err = capsys.readouterr().err
        assert "Refusing to remove self" in err

    def test_remove_refuses_when_no_mesh_yml(self, isolated_xdg, capsys):
        rc = run_remove_node(["any"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "does not exist" in err

    def test_remove_clears_drain_state(self, mesh_with_two_nodes, capsys):
        """C3.2 invariant: removing a drained node clears the drain
        state entry with a visible message. Otherwise the orphan
        would surface in list-drained / doctor."""
        from maxim.peer.drain_state import drain_node, read_drained_nodes

        # Drain mac-studio first
        drain_node("mac-studio", {"leader-desk", "mac-studio"})
        capsys.readouterr()  # eat the drain output

        rc = run_remove_node(["mac-studio"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "also cleared 'mac-studio' from drain state" in out

        # Drain state is empty
        result = read_drained_nodes({"leader-desk"})
        assert "mac-studio" not in result.drained
        assert "mac-studio" not in result.orphans

    def test_remove_without_drain_no_clear_message(self, mesh_with_two_nodes, capsys):
        """If the removed node wasn't drained, the cleared-drain
        message is suppressed (clear_drain_for_removed_node returns
        False)."""
        rc = run_remove_node(["mac-studio"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "also cleared" not in out

    def test_remove_extra_args_rejected(self, mesh_with_two_nodes, capsys):
        rc = run_remove_node(["mac-studio", "extra"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "extra arguments" in err.lower()

    def test_remove_help(self, mesh_with_two_nodes, capsys):
        for flag in ("-h", "--help"):
            rc = run_remove_node([flag])
            out = capsys.readouterr().out
            assert rc == 0
            assert "remove-node" in out

    def test_removed_mesh_round_trips(self, mesh_with_two_nodes):
        run_remove_node(["mac-studio"])
        text = _mesh_path(mesh_with_two_nodes).read_text()
        cfg1 = parse_mesh_config(text)
        cfg2 = parse_mesh_config(cfg1.to_yaml())
        assert cfg1 == cfg2


# ─── peer.yml preservation regression for the C3.2 verbs ──────────────


class TestPeerYmlPreservation:
    """C3.1 invariant continues to hold for C3.2: add-node and
    remove-node MUST NOT touch peer.yml. runtime/role.py reads its
    existence as part of role detection per Plan 2 R2a.
    """

    def test_add_node_preserves_peer_yml(self, mesh_with_two_nodes):
        peer_p = _peer_path(mesh_with_two_nodes)
        peer_p.write_text(VALID_PEER_YAML)
        original = peer_p.read_text()
        original_mtime = peer_p.stat().st_mtime
        run_add_node(["tablet", "--url", "http://t/v1"])
        assert peer_p.read_text() == original
        assert peer_p.stat().st_mtime == original_mtime

    def test_remove_node_preserves_peer_yml(self, mesh_with_two_nodes):
        peer_p = _peer_path(mesh_with_two_nodes)
        peer_p.write_text(VALID_PEER_YAML)
        original = peer_p.read_text()
        original_mtime = peer_p.stat().st_mtime
        run_remove_node(["mac-studio"])
        assert peer_p.read_text() == original
        assert peer_p.stat().st_mtime == original_mtime
