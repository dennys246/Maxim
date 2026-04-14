"""Tests for maxim.peer.mesh_cli (Plan 4 Stage C1)."""

from __future__ import annotations

import json

import pytest

from maxim.peer import mesh_cli
from maxim.peer.mesh_config import MeshNode


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


class _FakeBackend:
    """Stub for ``_MaximPeerBackend.for_url(...).health_check()``."""

    def __init__(self, result: _FakeProbeResult):
        self._result = result

    @classmethod
    def for_url(cls, url: str, *, api_key: str | None = None, model: str | None = None):
        return cls(cls._next_result)

    def health_check(self, *, enable_stage2: bool = True):
        return self._result


def _install_fake_backend(monkeypatch, outcome: str, detail: str = "ok", latency_ms: float = 10.0):
    _FakeBackend._next_result = _FakeProbeResult(outcome, detail, latency_ms)

    # The import happens inside _probe_node, so we patch the real module.
    import maxim.models.language.maxim_peer_backend as mpb

    monkeypatch.setattr(mpb, "_MaximPeerBackend", _FakeBackend)


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

    def test_exit_code_nonzero_on_any_fail(self, mesh_home, monkeypatch, capsys):
        _install_fake_backend(monkeypatch, outcome="auth_rejected", detail="HTTP 401")
        rc = mesh_cli.run_list_nodes([])
        assert rc == 1

    def test_drained_node_shown_as_drained_not_probed(self, mesh_home, monkeypatch, capsys):
        # Drain mac-studio first.
        from maxim.peer.mesh_config import drain_node

        drain_node("mac-studio")

        # Set up a backend that would fail if it were called — but for
        # drained nodes it should never be invoked.
        call_counter = {"n": 0}

        class _CountingBackend(_FakeBackend):
            @classmethod
            def for_url(cls, url, *, api_key=None, model=None):
                call_counter["n"] += 1
                return cls(_FakeProbeResult("ok"))

        import maxim.models.language.maxim_peer_backend as mpb

        monkeypatch.setattr(mpb, "_MaximPeerBackend", _CountingBackend)

        rc = mesh_cli.run_list_nodes([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "drained" in out.lower()
        assert call_counter["n"] == 1  # only leader-desk probed; mac-studio skipped

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

    def test_drain_and_resume_round_trip(self, mesh_home, capsys):
        from maxim.peer.mesh_config import read_drained_nodes

        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        assert rc == 0
        assert "mac-studio" in read_drained_nodes()
        assert "✓ Drained mac-studio" in capsys.readouterr().out

        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "resume"])
        assert rc == 0
        assert "mac-studio" not in read_drained_nodes()
        assert "✓ Resumed mac-studio" in capsys.readouterr().out

    def test_inference_broken_has_chat_endpoint_hint(self, mesh_home, monkeypatch, capsys):
        _install_fake_backend(monkeypatch, outcome="inference_broken", detail="stage2: timeout")
        rc = mesh_cli.run_node_subcommand(["--node", "leader-desk", "status"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "chat endpoint broken" in out
        assert "maxim peer llm --status" in out

    def test_missing_args_errors(self, mesh_home, capsys):
        rc = mesh_cli.run_node_subcommand(["--node"])
        assert rc == 2


class TestClassifyProbeResult:
    """Direct coverage of the outcome → CheckResult mapping table."""

    def _mk_node(self):
        return MeshNode(name="x", url="http://a/v1", role="leader")

    def test_ok(self):
        node = self._mk_node()
        r = mesh_cli._classify_probe_result(node, _FakeProbeResult("ok", "HTTP 200", 5))
        assert r.status == "ok"

    def test_auth_rejected_specific_first(self):
        node = self._mk_node()
        r = mesh_cli._classify_probe_result(node, _FakeProbeResult("auth_rejected", "HTTP 401"))
        assert r.status == "fail"
        assert "tunnel key rotate" in (r.fix or "")

    def test_inference_broken_specific(self):
        node = self._mk_node()
        r = mesh_cli._classify_probe_result(node, _FakeProbeResult("inference_broken", "stage2: HTTP 500"))
        assert r.status == "fail"
        assert "chat" in (r.fix or "").lower()

    def test_dns_fail_bucketed_as_warn(self):
        node = self._mk_node()
        r = mesh_cli._classify_probe_result(node, _FakeProbeResult("dns_fail", "no such host"))
        assert r.status == "warn"

    def test_unknown_outcome_warn(self):
        node = self._mk_node()
        r = mesh_cli._classify_probe_result(node, _FakeProbeResult("plasma_storm", "?"))
        assert r.status == "warn"
