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


class TestRoleDetectionGuard:
    """J1 fold (C2 pre-merge review): cli.py::main already calls
    detect_and_apply_role once before dispatching to the peer
    subcommand. Calling it a second time would re-emit role_detected
    with role_source=env_var (confusing) and re-fire role_divergence
    WARNINGs. The guard in run_peer_connect_subcommand skips the
    second call when MAXIM_ROLE is already set.
    """

    def test_second_call_skipped_when_maxim_role_already_set(self, mesh_home, monkeypatch):
        """Simulate the normal flow: cli.py::main ran and set
        MAXIM_ROLE. Dispatch to run_peer_connect_subcommand should
        NOT invoke detect_and_apply_role a second time."""
        monkeypatch.setenv("MAXIM_ROLE", "leader")

        call_count = {"n": 0}

        def _spy(argv):
            call_count["n"] += 1

        import maxim.runtime.role as role_mod

        monkeypatch.setattr(role_mod, "detect_and_apply_role", _spy)

        from maxim.peer.cli import run_peer_connect_subcommand

        # `show` is a cheap verb that doesn't touch drain state; we
        # just care that the dispatcher runs without re-triggering
        # role detection.
        run_peer_connect_subcommand(["show"])

        assert call_count["n"] == 0, (
            "detect_and_apply_role must not be called a second time "
            "when MAXIM_ROLE is already set — cli.py::main ran it "
            "once upstream."
        )

    def test_detection_runs_when_maxim_role_unset(self, mesh_home, monkeypatch):
        """Defensive path: a caller that skips cli.py::main (direct
        import of run_peer_connect_subcommand from a test or script)
        and doesn't set MAXIM_ROLE should still get role detection."""
        monkeypatch.delenv("MAXIM_ROLE", raising=False)

        call_count = {"n": 0}

        def _spy(argv):
            call_count["n"] += 1
            import os

            os.environ["MAXIM_ROLE"] = "leader"

        import maxim.runtime.role as role_mod

        monkeypatch.setattr(role_mod, "detect_and_apply_role", _spy)

        from maxim.peer.cli import run_peer_connect_subcommand

        run_peer_connect_subcommand(["show"])

        assert call_count["n"] == 1, (
            "detect_and_apply_role must run once when MAXIM_ROLE is unset — covers the direct-import defensive path."
        )


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


# ─── C3.3: maxim peer --node <name> install <extras> ────────────────────


class TestParseNodeInstallTokens:
    """Unit tests for the argv-tail parser used by _run_node_install.

    The parser mirrors _cmd_install's shape (comma-split positional,
    -- flags ignored) but rejects positional URLs because the mesh-
    aware verb resolves the URL from mesh.yml::nodes.
    """

    def test_simple_single_token(self):
        from maxim.peer.mesh_cli import _parse_node_install_tokens

        assert _parse_node_install_tokens(["semantic"]) == ["semantic"]

    def test_comma_split(self):
        from maxim.peer.mesh_cli import _parse_node_install_tokens

        assert _parse_node_install_tokens(["semantic,llm-torch"]) == [
            "semantic",
            "llm-torch",
        ]

    def test_multiple_positional_tokens(self):
        from maxim.peer.mesh_cli import _parse_node_install_tokens

        assert _parse_node_install_tokens(["semantic", "llm-torch"]) == [
            "semantic",
            "llm-torch",
        ]

    def test_double_dash_flag_ignored(self):
        from maxim.peer.mesh_cli import _parse_node_install_tokens

        assert _parse_node_install_tokens(["--future-flag", "semantic"]) == ["semantic"]

    def test_positional_url_raises_value_error_with_fallback_hint(self):
        """Unlike _cmd_install, this verb rejects positional URLs and
        points the operator at the no-mesh.yml fallback verb."""
        from maxim.peer.mesh_cli import _parse_node_install_tokens

        with pytest.raises(ValueError, match="maxim peer install"):
            _parse_node_install_tokens(["semantic", "https://example.com/v1"])

    def test_empty_argv_tail(self):
        from maxim.peer.mesh_cli import _parse_node_install_tokens

        assert _parse_node_install_tokens([]) == []


class TestRunNodeInstall:
    """Plan 4 C3.3 — mesh-aware install verb composing
    drain → install_on_target → resume.

    Mocks ``maxim.peer.mesh_cli.install_on_target`` to avoid real
    HTTP traffic. The fold commit moved the shared install core from
    ``peer/cli.py`` to ``peer/install_core.py``; ``mesh_cli.py``
    imports ``install_on_target`` at module load, so the monkeypatch
    has to target the ``mesh_cli`` namespace (where
    ``_run_node_install`` looks the name up), not the
    ``install_core`` definition site. The regression test for the
    core's HTTP body shape lives in ``test_peer_install.py``
    (commit 1). This class exclusively covers the composition layer:
    drain/resume bookkeeping, self-guard, error propagation, and
    was-drained sticky semantics.
    """

    def _install_stub(self, monkeypatch, return_code: int):
        """Patch ``install_on_target`` on the mesh_cli module and
        record its call args. Returns a dict populated on first call
        with ``url``, ``key``, ``extras``, ``packages``.

        Patches on ``mesh_cli`` (the consumer), not on ``install_core``
        (the definition site), because ``mesh_cli`` binds the name at
        module load via ``from maxim.peer.install_core import
        install_on_target``. A patch on ``install_core`` would not
        propagate to the consumer's already-bound reference.
        """
        captured: dict = {}

        def fake_install(url, key, extras, packages):
            captured["url"] = url
            captured["key"] = key
            captured["extras"] = extras
            captured["packages"] = packages
            return return_code

        monkeypatch.setattr(mesh_cli, "install_on_target", fake_install)
        return captured

    # ─── self-guard ────────────────────────────────────────────────

    def test_install_on_self_refused_with_pip_hint(self, mesh_home, capsys):
        """Self-install is a round-trip through the operator's own
        admin endpoint. Point at `pip install` directly instead."""
        rc = mesh_cli.run_node_subcommand(
            ["--node", "leader-desk", "install", "semantic"],
        )
        err = capsys.readouterr().err
        assert rc == 2
        assert "Refusing to install on self" in err
        assert "pip install pymaxim" in err

    # ─── unknown-node guard (via dispatcher) ───────────────────────

    def test_install_unknown_node_exit_2(self, mesh_home, capsys):
        rc = mesh_cli.run_node_subcommand(
            ["--node", "ghost", "install", "semantic"],
        )
        err = capsys.readouterr().err
        assert rc == 2
        assert "Unknown node" in err

    # ─── empty-tokens guard ────────────────────────────────────────

    def test_install_no_tokens_shows_usage(self, mesh_home, capsys):
        rc = mesh_cli.run_node_subcommand(["--node", "mac-studio", "install"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Usage:" in err
        assert "mac-studio" in err
        assert "extras:" in err

    def test_install_only_whitespace_tokens_shows_no_valid_tokens(self, mesh_home, monkeypatch, capsys):
        """A single whitespace-only token (e.g. `install "  "`) passes
        the empty-tokens check but classifies to nothing — surface a
        distinct error rather than POSTing an empty body.
        """
        self._install_stub(monkeypatch, return_code=0)
        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", "   "],
        )
        err = capsys.readouterr().err
        assert rc == 2
        assert "No valid install tokens" in err

    def test_install_positional_url_rejected_with_fallback_hint(self, mesh_home, capsys):
        rc = mesh_cli.run_node_subcommand(
            [
                "--node",
                "mac-studio",
                "install",
                "semantic",
                "https://other.example.com/v1",
            ],
        )
        err = capsys.readouterr().err
        assert rc == 2
        assert "positional URL" in err
        assert "maxim peer install" in err

    # ─── happy path: drain → install → resume ──────────────────────

    def test_happy_path_drains_installs_resumes(self, mesh_home, monkeypatch, capsys):
        captured = self._install_stub(monkeypatch, return_code=0)

        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", "semantic"],
        )
        out = capsys.readouterr().out

        assert rc == 0
        # The core got the mesh.yml-resolved URL + cluster key.
        assert captured["url"] == "https://mac.example.com/v1"
        assert captured["key"] == "sk-cluster-abc"
        assert captured["extras"] == ["semantic"]
        assert captured["packages"] == []
        # Drain + resume bookkeeping both reported.
        assert "Drained 'mac-studio' for install" in out
        assert "Resumed 'mac-studio' after install" in out
        # Node is NOT in the drain set at the end.
        from maxim.peer.drain_state import read_drained_nodes

        assert "mac-studio" not in read_drained_nodes(set()).active

    def test_happy_path_classifies_mixed_tokens(self, mesh_home, monkeypatch, capsys):
        """Mixed comma-separated tokens: known extras go to `extras`,
        unknown go to `packages`. Verifies the classifier is shared
        with the positional-URL verb's wire-level shape.
        """
        captured = self._install_stub(monkeypatch, return_code=0)

        rc = mesh_cli.run_node_subcommand(
            [
                "--node",
                "mac-studio",
                "install",
                "semantic,llm-torch,sentence-transformers",
            ],
        )
        assert rc == 0
        assert captured["extras"] == ["semantic", "llm-torch"]
        assert captured["packages"] == ["sentence-transformers"]

    # ─── was-drained sticky semantics ──────────────────────────────

    def test_operator_prewalked_drain_is_sticky(self, mesh_home, monkeypatch, capsys):
        """If operator drained the node BEFORE running install, the
        verb must skip both drain AND auto-resume. Operator intent is
        sticky."""
        # Operator drains first.
        mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        capsys.readouterr()

        captured = self._install_stub(monkeypatch, return_code=0)
        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", "semantic"],
        )
        out = capsys.readouterr().out

        assert rc == 0
        assert captured["extras"] == ["semantic"]
        assert "already drained by operator" in out
        # We did NOT resume the node — operator's drain is still sticky.
        assert "Resumed" not in out
        from maxim.peer.drain_state import read_drained_nodes

        assert "mac-studio" in read_drained_nodes(set()).active

    # ─── install-failure branches ──────────────────────────────────

    def test_install_failure_leaves_node_drained_with_loud_message(self, mesh_home, monkeypatch, capsys):
        """If drain succeeds but install fails, the node stays
        drained with a loud 'STILL DRAINED → run resume' message.
        """
        self._install_stub(monkeypatch, return_code=1)

        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", "semantic"],
        )
        captured = capsys.readouterr()

        assert rc == 1
        assert "STILL DRAINED" in captured.err
        assert "maxim peer --node mac-studio resume" in captured.err
        # Node IS still in the drain set — the verb did not clean up.
        from maxim.peer.drain_state import read_drained_nodes

        assert "mac-studio" in read_drained_nodes(set()).active

    def test_install_failure_with_prewalked_drain_no_still_drained_shout(self, mesh_home, monkeypatch, capsys):
        """If operator pre-drained AND install fails, we should NOT
        yell 'STILL DRAINED' because WE didn't drain it — the pre-
        drained state is operator intent, not a side effect of this
        verb.
        """
        mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        capsys.readouterr()

        self._install_stub(monkeypatch, return_code=1)
        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", "semantic"],
        )
        captured = capsys.readouterr()

        assert rc == 1
        # No "STILL DRAINED" shout because we didn't change drain state.
        assert "STILL DRAINED" not in captured.err

    # ─── dispatcher wire-up ────────────────────────────────────────

    def test_dispatcher_routes_install_verb(self, mesh_home, monkeypatch, capsys):
        """Confirm `--node <name> install` routes through
        run_node_subcommand to _run_node_install, not some other
        verb branch.
        """
        captured = self._install_stub(monkeypatch, return_code=0)
        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", "semantic"],
        )
        assert rc == 0
        assert "url" in captured  # install_on_target was called

    def test_dispatcher_unknown_verb_includes_install_in_hint(self, mesh_home, capsys):
        """The unknown-verb error message should list install as one
        of the valid verbs.
        """
        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "bogus-verb"],
        )
        err = capsys.readouterr().err
        assert rc == 2
        assert "install" in err


class TestRunNodeInstallFoldGuards:
    """Regression guards for the C3.3 pre-merge review folds.

    Separate class from ``TestRunNodeInstall`` so the fold-specific
    assertions don't bloat the happy-path class. Each test maps to a
    numbered fold finding; the mapping is in the test docstring.
    """

    def _install_stub(self, monkeypatch, return_code: int):
        """Same shape as TestRunNodeInstall._install_stub — patches
        ``mesh_cli.install_on_target``, returns a captured-args dict.
        """
        captured: dict = {}

        def fake_install(url, key, extras, packages):
            captured["url"] = url
            captured["key"] = key
            captured["extras"] = extras
            captured["packages"] = packages
            return return_code

        monkeypatch.setattr(mesh_cli, "install_on_target", fake_install)
        return captured

    # ─── Fold I4: exit code 3 for resume-after-success failure ───

    def test_resume_failure_after_successful_install_returns_exit_code_3(self, mesh_home, monkeypatch, capsys):
        """Fold I4: install succeeded but post-install auto-resume
        failed. Operators tailing exit codes must be able to tell
        this case apart from exit code 1 (install itself failed).
        """
        self._install_stub(monkeypatch, return_code=0)

        # Patch _resume_node on the mesh_cli module to raise DrainError
        # (simulating a filelock timeout or similar).
        from maxim.peer.drain_state import DrainError

        def fake_resume(name, known):
            raise DrainError("simulated filelock timeout during resume")

        monkeypatch.setattr(mesh_cli, "_resume_node", fake_resume)

        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", "semantic"],
        )
        captured = capsys.readouterr()
        # Exit 3, NOT 1 — this is the whole point of fold I4.
        assert rc == 3
        assert "Install succeeded but resume failed" in captured.err
        assert "maxim peer --node mac-studio resume" in captured.err

    # ─── Fold I6: try/finally for stacked interruption ───────────

    def test_keyboard_interrupt_mid_install_leaves_drained_with_hint(self, mesh_home, monkeypatch, capsys):
        """Fold I6: any unexpected exception during install (including
        KeyboardInterrupt) should still print the 'still drained'
        hint to stderr before re-raising.
        """

        def fake_install_raises(url, key, extras, packages):
            raise KeyboardInterrupt()

        monkeypatch.setattr(mesh_cli, "install_on_target", fake_install_raises)

        with pytest.raises(KeyboardInterrupt):
            mesh_cli.run_node_subcommand(
                ["--node", "mac-studio", "install", "semantic"],
            )
        err = capsys.readouterr().err
        assert "Install interrupted" in err
        assert "STILL DRAINED" in err
        assert "maxim peer --node mac-studio resume" in err

    def test_unexpected_exception_mid_install_prints_hint_and_reraises(self, mesh_home, monkeypatch, capsys):
        """Even a plain ValueError from deep inside install_on_target
        should trigger the still-drained hint."""

        def fake_install_raises(url, key, extras, packages):
            raise ValueError("programming bug in the install core")

        monkeypatch.setattr(mesh_cli, "install_on_target", fake_install_raises)

        with pytest.raises(ValueError, match="programming bug"):
            mesh_cli.run_node_subcommand(
                ["--node", "mac-studio", "install", "semantic"],
            )
        err = capsys.readouterr().err
        assert "Install interrupted" in err
        assert "STILL DRAINED" in err

    def test_interrupt_when_operator_pre_drained_no_hint(self, mesh_home, monkeypatch, capsys):
        """If operator pre-drained AND install is interrupted, we
        should NOT print the STILL DRAINED banner because WE didn't
        change drain state. The interrupted state is the pre-existing
        operator intent, not something our verb created.
        """
        mesh_cli.run_node_subcommand(["--node", "mac-studio", "drain"])
        capsys.readouterr()

        def fake_install_raises(url, key, extras, packages):
            raise KeyboardInterrupt()

        monkeypatch.setattr(mesh_cli, "install_on_target", fake_install_raises)

        with pytest.raises(KeyboardInterrupt):
            mesh_cli.run_node_subcommand(
                ["--node", "mac-studio", "install", "semantic"],
            )
        err = capsys.readouterr().err
        # No "Install interrupted" banner since drained_here was False.
        assert "Install interrupted" not in err
        assert "STILL DRAINED" not in err

    # ─── Fold I7: STILL DRAINED banner is the LAST line on stderr ──

    def test_still_drained_banner_comes_after_install_error_output(self, mesh_home, monkeypatch, capsys):
        """Fold I7: the install core prints its own error message
        first (HTTP status, body snippet), then _run_node_install's
        'STILL DRAINED' banner. Operator should see the actionable
        recovery hint LAST so it's the visible line on their screen.
        """

        def fake_install_with_stderr(url, key, extras, packages):
            import sys as _sys

            print("INSTALL CORE: HTTP 500 upstream crash", file=_sys.stderr)
            return 1

        monkeypatch.setattr(mesh_cli, "install_on_target", fake_install_with_stderr)

        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", "semantic"],
        )
        captured = capsys.readouterr()
        assert rc == 1
        # STILL DRAINED appears AFTER the install core's own stderr.
        still_drained_pos = captured.err.rfind("STILL DRAINED")
        core_error_pos = captured.err.find("HTTP 500 upstream crash")
        assert still_drained_pos != -1
        assert core_error_pos != -1
        assert still_drained_pos > core_error_pos

    # ─── Fold CC2: drain TOCTOU under concurrent operator drain ───

    def test_concurrent_operator_drain_is_not_clobbered(self, mesh_home, monkeypatch, capsys):
        """Fold CC2 (cross-confirmed BLOCKING): in the pre-fold code,
        a concurrent operator drain between ``read_drained_nodes``
        and ``drain_node`` could be silently clobbered when the
        install verb auto-resumed on success. With
        ``drain_node_if_absent``, the atomic check returns
        ``we_added_it = False`` if anyone else (including an earlier
        operator invocation whose state is already on disk) drained
        the node, and the verb skips the auto-resume.

        We simulate the race by pre-seeding the drain state file
        BEFORE the verb runs (equivalent to the "another actor
        drained between our read and our write" scenario from the
        review finding).
        """
        # Operator drain already landed on disk before our verb runs.
        from maxim.peer.drain_state import drain_node

        drain_node("mac-studio", {"leader-desk", "mac-studio"})

        # Our verb fires. The atomic check should see the existing
        # drain and NOT auto-resume.
        self._install_stub(monkeypatch, return_code=0)
        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", "semantic"],
        )
        assert rc == 0

        # Verify: the drain state file STILL has mac-studio drained
        # after our verb completed, because was-drained-sticky skipped
        # the auto-resume.
        from maxim.peer.drain_state import read_drained_nodes

        assert "mac-studio" in read_drained_nodes(set()).active

    # ─── Fold I3: httpx / http-client are not URLs ────────────────

    def test_httpx_as_install_token_not_rejected_as_url(self, mesh_home, monkeypatch, capsys):
        """Fold I3: ``arg.startswith("http")`` false-positived on
        real PyPI packages. Fix uses ``"://"`` via
        :func:`_looks_like_url`.
        """
        captured = self._install_stub(monkeypatch, return_code=0)
        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", "httpx"],
        )
        assert rc == 0
        assert captured["packages"] == ["httpx"]

    def test_http_client_as_install_token_not_rejected_as_url(self, mesh_home, monkeypatch, capsys):
        captured = self._install_stub(monkeypatch, return_code=0)
        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", "http-client"],
        )
        assert rc == 0
        assert captured["packages"] == ["http-client"]

    def test_real_positional_url_with_scheme_still_rejected(self, mesh_home, capsys):
        """The I3 fix tightens URL detection but must NOT loosen the
        positional-URL rejection for actual URLs with ``"://"``.
        """
        rc = mesh_cli.run_node_subcommand(
            [
                "--node",
                "mac-studio",
                "install",
                "semantic",
                "https://evil.example.com/v1",
            ],
        )
        err = capsys.readouterr().err
        assert rc == 2
        assert "positional URL" in err
        # Fold M3: full URL should NOT appear verbatim in stderr,
        # only the scheme-redacted form.
        assert "https://evil.example.com/v1" not in err
        assert "https://..." in err

    # ─── Fold M3: URL redaction in error messages ─────────────────

    def test_positional_url_error_redacts_secrets(self, mesh_home, capsys):
        """Fold M3: URLs can contain secrets in the path. The error
        message should redact to scheme-only.
        """
        rc = mesh_cli.run_node_subcommand(
            [
                "--node",
                "mac-studio",
                "install",
                "semantic",
                "https://host.example.com/sk-abc123-secret-token",
            ],
        )
        err = capsys.readouterr().err
        assert rc == 2
        # The secret path must NOT appear.
        assert "sk-abc123" not in err
        # Redacted form IS present.
        assert "https://..." in err

    # ─── Fold M6: comma edge cases in mesh parser ─────────────────

    def test_leading_trailing_commas_no_empty_entries_mesh_parser(self, mesh_home, monkeypatch):
        """Fold M6: ``,semantic,`` via the mesh parser should drop
        the empty leading/trailing entries without polluting the
        classifier output.
        """
        captured = self._install_stub(monkeypatch, return_code=0)
        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", ",semantic,"],
        )
        assert rc == 0
        assert captured["extras"] == ["semantic"]
        assert captured["packages"] == []

    def test_triple_commas_produce_empty_tokens_error(self, mesh_home, monkeypatch, capsys):
        """``,,,`` classifies to nothing → empty-tokens error."""
        self._install_stub(monkeypatch, return_code=0)
        rc = mesh_cli.run_node_subcommand(
            ["--node", "mac-studio", "install", ",,,"],
        )
        err = capsys.readouterr().err
        assert rc == 2
        assert "No valid install tokens" in err


class TestDrainNodeIfAbsent:
    """Fold CC2: regression guards for the state-layer primitive that
    closes the drain TOCTOU window.
    """

    def test_returns_true_on_fresh_drain(self, mesh_home):
        from maxim.peer.drain_state import drain_node_if_absent

        drain_set, we_added = drain_node_if_absent("mac-studio", {"leader-desk", "mac-studio"})
        assert we_added is True
        assert "mac-studio" in drain_set

    def test_returns_false_on_already_drained(self, mesh_home):
        from maxim.peer.drain_state import drain_node, drain_node_if_absent

        # Pre-drain.
        drain_node("mac-studio", {"leader-desk", "mac-studio"})

        # Second call should report we_added=False, state unchanged.
        drain_set, we_added = drain_node_if_absent("mac-studio", {"leader-desk", "mac-studio"})
        assert we_added is False
        assert "mac-studio" in drain_set

    def test_rejects_unknown_node(self, mesh_home):
        from maxim.peer.drain_state import DrainError, drain_node_if_absent

        with pytest.raises(DrainError, match="unknown node"):
            drain_node_if_absent("ghost", {"leader-desk", "mac-studio"})

    def test_rejects_self_without_force(self, mesh_home):
        from maxim.peer.drain_state import DrainError, drain_node_if_absent

        with pytest.raises(DrainError, match="refusing to drain self"):
            drain_node_if_absent(
                "leader-desk",
                {"leader-desk", "mac-studio"},
                self_name="leader-desk",
                force_self=False,
            )

    def test_allows_self_with_force(self, mesh_home):
        from maxim.peer.drain_state import drain_node_if_absent

        drain_set, we_added = drain_node_if_absent(
            "leader-desk",
            {"leader-desk", "mac-studio"},
            self_name="leader-desk",
            force_self=True,
        )
        assert we_added is True
        assert "leader-desk" in drain_set
