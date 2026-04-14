"""Tests for maxim.peer.mesh_cli (Plan 4 Stage C1)."""

from __future__ import annotations

import json

import pytest

from maxim.peer import mesh_cli


# Canonical mesh.yml used across the test file. Two nodes, leader + peer.
VALID_MESH_YAML = """\
cluster_key: sk-cluster-abc
self: leader-desk
protocol_version: 1
nodes:
  - name: leader-desk
    url: http://192.168.1.10:8099/v1
    role: leader
  - name: mac-studio
    url: https://mac.example.com/v1
    role: peer
"""


class _FakeProbeResult:
    def __init__(self, outcome: str, detail: str = "", latency_ms: float | None = None):
        self.outcome = outcome
        self.detail = detail
        self.latency_ms = latency_ms


def _make_fake_backend(result: _FakeProbeResult):
    """Build a fake _MaximPeerBackend class bound to a specific probe
    result. Each call to ``_install_fake_backend`` gets a fresh class
    so there's no shared mutable state between tests (pre-merge review
    F15 fix).
    """

    class _FakeBackend:
        def __init__(self, r):
            self._result = r

        @classmethod
        def for_url(cls, url: str, *, api_key: str | None = None, model: str | None = None):
            return cls(result)

        def health_check(self, *, enable_stage2: bool = True):
            return self._result

    return _FakeBackend


def _install_fake_backend(monkeypatch, outcome: str, detail: str = "ok", latency_ms: float = 10.0):
    fake = _make_fake_backend(_FakeProbeResult(outcome, detail, latency_ms))
    import maxim.models.language.maxim_peer_backend as mpb

    monkeypatch.setattr(mpb, "_MaximPeerBackend", fake)


@pytest.fixture
def mesh_home(tmp_path, monkeypatch):
    """Set up a working XDG dir + MAXIM_DATA_HOME + mesh.yml."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("MAXIM_ROLE", "leader")
    from maxim.utils import paths

    paths._reset_caches()
    mesh_path = tmp_path / "config" / "maxim" / "mesh.yml"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text(VALID_MESH_YAML)
    return tmp_path


@pytest.fixture
def peer_only_home(tmp_path, monkeypatch):
    """No mesh.yml, only peer.yml — for testing the fallback path (F16)."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("MAXIM_ROLE", "leader")
    from maxim.utils import paths

    paths._reset_caches()
    peer_path = tmp_path / "config" / "maxim" / "peer.yml"
    peer_path.parent.mkdir(parents=True)
    peer_path.write_text("url: https://leader.example.com/v1\napi_key: sk-peer-fallback\n")
    return tmp_path


class TestListNodes:
    def test_happy_path_table(self, mesh_home, monkeypatch, capsys):
        _install_fake_backend(monkeypatch, outcome="ok", detail="HTTP 200", latency_ms=42.0)
        rc = mesh_cli.run_list_nodes([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "leader-desk" in out
        assert "mac-studio" in out
        assert "(self)" in out  # leader-desk is marked as self
        assert "✓" in out
        assert "42" in out  # latency rendered

    def test_json_output_shape(self, mesh_home, monkeypatch, capsys):
        _install_fake_backend(monkeypatch, outcome="ok", detail="HTTP 200", latency_ms=12.0)
        rc = mesh_cli.run_list_nodes(["--json"])
        out = capsys.readouterr().out
        assert rc == 0
        doc = json.loads(out)
        assert doc["self"] == "leader-desk"
        assert doc["worst_status"] == "ok"
        names = [n["name"] for n in doc["nodes"]]
        assert names == ["leader-desk", "mac-studio"]
        assert all("status" in n and "url" in n and "role" in n for n in doc["nodes"])
        # Plan 4 C2: drained boolean field on every node report
        assert all("drained" in n and n["drained"] is False for n in doc["nodes"])
        assert doc["orphans"] == []

    def test_exit_code_nonzero_on_any_fail(self, mesh_home, monkeypatch, capsys):
        _install_fake_backend(monkeypatch, outcome="auth_rejected", detail="HTTP 401")
        rc = mesh_cli.run_list_nodes([])
        assert rc == 1

    def test_no_mesh_config_errors_out(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        from maxim.utils import paths

        paths._reset_caches()
        rc = mesh_cli.run_list_nodes([])
        err = capsys.readouterr().err
        assert rc == 1
        assert "No mesh.yml or peer.yml" in err

    def test_auth_rejected_surfaces_key_rotate_hint(self, mesh_home, monkeypatch, capsys):
        _install_fake_backend(monkeypatch, outcome="auth_rejected", detail="HTTP 401")
        mesh_cli.run_list_nodes([])
        out = capsys.readouterr().out
        assert "auth rejected" in out
        assert "tunnel key rotate" in out

    def test_fallback_from_peer_yml_end_to_end(self, peer_only_home, monkeypatch, capsys):
        """F16: peer.yml → synthesized one-node mesh. Zero breaking change."""
        _install_fake_backend(monkeypatch, outcome="ok", detail="HTTP 200", latency_ms=25.0)
        rc = mesh_cli.run_list_nodes([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "1 node(s)" in out
        assert "leader" in out  # synthesized name
        assert "https://leader.example.com/v1" in out


class TestNodeSubcommand:
    def test_status_dispatches_to_named_node(self, mesh_home, monkeypatch, capsys):
        _install_fake_backend(monkeypatch, outcome="ok", detail="HTTP 200", latency_ms=5.0)
        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "status"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "mac-studio" in out
        assert "leader-desk" not in out  # single-node output

    def test_health_is_alias_for_status(self, mesh_home, monkeypatch, capsys):
        _install_fake_backend(monkeypatch, outcome="ok")
        rc = mesh_cli.run_node_subcommand(["--node", "leader-desk", "health"])
        assert rc == 0
        assert "leader-desk" in capsys.readouterr().out

    def test_unknown_node_errors_with_known_list(self, mesh_home, capsys):
        rc = mesh_cli.run_node_subcommand(["--node", "ghost", "status"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Unknown node" in err
        assert "leader-desk" in err
        assert "mac-studio" in err

    def test_unknown_verb_errors(self, mesh_home, capsys):
        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "teleport"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Unknown --node verb" in err

    def test_inference_broken_has_chat_endpoint_hint(self, mesh_home, monkeypatch, capsys):
        _install_fake_backend(monkeypatch, outcome="inference_broken", detail="stage2: timeout")
        rc = mesh_cli.run_node_subcommand(["--node", "leader-desk", "status"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "chat endpoint broken" in out
        assert "maxim peer llm --status" in out

    def test_missing_args_distinguishes_missing_name_from_verb(self, mesh_home, capsys):
        """F14: the error must say what's missing."""
        rc = mesh_cli.run_node_subcommand(["--node"])
        assert rc == 2
        assert "Missing node name" in capsys.readouterr().err

        rc = mesh_cli.run_node_subcommand(["--node", "leader-desk"])
        assert rc == 2
        assert "Missing verb" in capsys.readouterr().err

    def test_drain_ships_in_c2(self, mesh_home, capsys):
        """Plan 4 C2: drain verb now works against a known node."""
        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Drained 'mac-studio'" in out
        # Verify persistence
        from maxim.peer.drain_state import read_drained_nodes

        result = read_drained_nodes({"leader-desk", "mac-studio"})
        assert "mac-studio" in result.drained

    def test_resume_ships_in_c2(self, mesh_home, capsys):
        """Plan 4 C2: resume verb removes the drain entry."""
        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        assert rc == 0
        capsys.readouterr()
        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "resume"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Resumed 'mac-studio'" in out
        from maxim.peer.drain_state import read_drained_nodes

        result = read_drained_nodes({"leader-desk", "mac-studio"})
        assert result.drained == frozenset()


class TestDrainResumeExitCodes:
    """Plan 4 C2: exit code contract from review finding E9."""

    def test_drain_unknown_node_exit_2(self, mesh_home, capsys):
        rc = mesh_cli.run_node_subcommand(["--node", "ghost", "drain"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Unknown node" in err

    def test_drain_idempotent_exit_0(self, mesh_home, capsys):
        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        assert rc == 0
        capsys.readouterr()
        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "already drained" in out

    def test_resume_idempotent_exit_0(self, mesh_home, capsys):
        """Resume on a not-drained node should succeed silently."""
        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "resume"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "was not drained" in out

    def test_drain_self_rejected_without_force(self, mesh_home, capsys):
        """Draining yourself strands in-flight requests. Require
        --force-self to opt in."""
        rc = mesh_cli.run_node_subcommand(["--node", "leader-desk", "drain"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Refusing to drain self" in err
        assert "--force-self" in err

    def test_drain_self_with_force_succeeds(self, mesh_home, capsys):
        rc = mesh_cli.run_node_subcommand(["--node", "leader-desk", "drain", "--force-self"])
        assert rc == 0
        assert "Drained" in capsys.readouterr().out


class TestListDrained:
    def test_empty_state(self, mesh_home, capsys):
        rc = mesh_cli.run_list_drained([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "No nodes drained" in out

    def test_with_drains(self, mesh_home, capsys):
        mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        capsys.readouterr()
        rc = mesh_cli.run_list_drained([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "mac-studio" in out

    def test_orphan_surfaces_with_hint(self, mesh_home, capsys, tmp_path):
        """Orphan drain entries (drained name no longer in mesh.yml)
        surface as a warning without blocking the command."""
        # Drain a node that exists
        mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        capsys.readouterr()

        # Rewrite mesh.yml to remove mac-studio
        mesh_path = tmp_path / "config" / "maxim" / "mesh.yml"
        mesh_path.write_text(
            "cluster_key: sk-cluster-abc\n"
            "self: leader-desk\n"
            "protocol_version: 1\n"
            "nodes:\n"
            "  - name: leader-desk\n"
            "    url: http://192.168.1.10:8099/v1\n"
            "    role: leader\n"
        )

        rc = mesh_cli.run_list_drained([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Orphan drain entries" in out
        assert "mac-studio" in out
        assert "resume" in out


class TestListNodesDrainDisplay:
    def test_drained_node_shown_with_symbol(self, mesh_home, monkeypatch, capsys):
        _install_fake_backend(monkeypatch, outcome="ok")
        mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        capsys.readouterr()
        rc = mesh_cli.run_list_nodes([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "⊝" in out  # drained symbol
        assert "drained (not probed)" in out
        assert "drained" in out  # header count marker

    def test_drained_node_not_probed(self, mesh_home, monkeypatch, capsys):
        """Regression guard: drained nodes MUST NOT make a network
        call. If the fake backend IS called for the drained node, the
        call counter fires."""
        call_count = {"n": 0}

        class _CountingBackend:
            @classmethod
            def for_url(cls, url, *, api_key=None, model=None):
                call_count["n"] += 1
                return cls()

            def health_check(self, *, enable_stage2=True):
                return _FakeProbeResult("ok", "HTTP 200", 5.0)

        import maxim.models.language.maxim_peer_backend as mpb

        monkeypatch.setattr(mpb, "_MaximPeerBackend", _CountingBackend)

        mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        capsys.readouterr()
        mesh_cli.run_list_nodes([])
        # Only leader-desk should have been probed; mac-studio is drained
        assert call_count["n"] == 1

    def test_drained_node_status_subcommand(self, mesh_home, monkeypatch, capsys):
        """`--node X status` for a drained node shows info, no probe."""
        mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        capsys.readouterr()
        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "status"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "drained" in out.lower()

    def test_json_includes_drained_flag(self, mesh_home, monkeypatch, capsys):
        import json as _json

        _install_fake_backend(monkeypatch, outcome="ok")
        mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        capsys.readouterr()
        mesh_cli.run_list_nodes(["--json"])
        out = capsys.readouterr().out
        doc = _json.loads(out)
        drained_node = next(n for n in doc["nodes"] if n["name"] == "mac-studio")
        assert drained_node["drained"] is True
        assert drained_node["status"] == "info"
        leader = next(n for n in doc["nodes"] if n["name"] == "leader-desk")
        assert leader["drained"] is False

    def test_json_orphans_field(self, mesh_home, monkeypatch, capsys, tmp_path):
        """Orphan drain entries appear in the top-level JSON 'orphans' field."""
        import json as _json

        _install_fake_backend(monkeypatch, outcome="ok")
        mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        capsys.readouterr()

        # Remove mac-studio from mesh.yml
        mesh_path = tmp_path / "config" / "maxim" / "mesh.yml"
        mesh_path.write_text(
            "cluster_key: sk-cluster-abc\n"
            "self: leader-desk\n"
            "protocol_version: 1\n"
            "nodes:\n"
            "  - name: leader-desk\n"
            "    url: http://192.168.1.10:8099/v1\n"
            "    role: leader\n"
        )

        mesh_cli.run_list_nodes(["--json"])
        out = capsys.readouterr().out
        doc = _json.loads(out)
        assert doc["orphans"] == ["mac-studio"]


class TestImportErrorFallback:
    """Round 2 A5R2: the ``ImportError`` defensive branch in ``_probe_node``
    needs regression coverage or it's a comment in code form.

    Simulates the ``llm-server`` extra not being installed.
    """

    def test_missing_backend_produces_warn_with_extra_hint(self, mesh_home, monkeypatch, capsys):
        import sys

        # Force the import to fail at call time. Using a class that raises
        # on __getattr__ is cleaner than setitem(None) because we need the
        # specific ``ImportError`` branch path.
        class _Broken:
            def __getattr__(self, name):
                raise ImportError("simulated: llm-server extra not installed")

        monkeypatch.setitem(sys.modules, "maxim.models.language.maxim_peer_backend", _Broken())

        rc = mesh_cli.run_list_nodes([])
        out = capsys.readouterr().out
        # Probe reports warn (not fail) so the exit code is 0 — import
        # failure is graceful degrade, not an operator error.
        assert rc == 0
        assert "peer backend import failed" in out
        assert "llm-server" in out
