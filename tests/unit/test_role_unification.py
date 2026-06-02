"""Regression guards for C3 of config_unification.md — role-detector unification.

The seven-rank decision order is implemented in
:func:`maxim.runtime.role.detect_role`. The eight-cell regression
matrix below pins every Mac Mini failure mode + ensures
:mod:`maxim.runtime.leader_mode` (the back-compat wrapper) returns
results consistent with the unified detector.

Mac Mini triggers tested:

- Trigger #2: ``~/.cloudflared/config.yaml`` (yaml extension) is
  now accepted for the cloudflared signal.
- Trigger #3: a stale ``peer.yml`` on a now-leader machine no longer
  forces ``role=peer`` because cloudflared rank 4 outranks peer.yml
  rank 5.

The eight test cells correspond to the named rows in the plan doc's
C3 section.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maxim.runtime import leader_mode, role


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect both XDG_CONFIG_HOME (for peer.yml + mesh.yml + config.json)
    and HOME (for ~/.cloudflared/) to ``tmp_path`` so each test starts
    with a clean filesystem state."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    # Also clear any cached config.json state
    from maxim.runtime.config_loader import reset_config_cache

    reset_config_cache()
    yield tmp_path
    reset_config_cache()


def _write_peer_yml(home: Path) -> None:
    """Create a minimal peer.yml so ``_peer_yml_exists`` returns True."""
    peer_dir = home / "maxim"
    peer_dir.mkdir(parents=True, exist_ok=True)
    (peer_dir / "peer.yml").write_text("url: http://leader.example.com\napi_key: sk-test\n")


def _write_mesh_yml(home: Path) -> None:
    mesh_dir = home / "maxim"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / "mesh.yml").write_text(
        "cluster_key: ck\nself_name: a\nnodes:\n  - name: a\n    url: http://a/v1\n    role: leader\n"
    )


def _write_cloudflared_yml(home: Path) -> None:
    cf_dir = home / ".cloudflared"
    cf_dir.mkdir(parents=True, exist_ok=True)
    (cf_dir / "config.yml").write_text("tunnel: abc\n")


def _write_cloudflared_yaml(home: Path) -> None:
    """The Mac Mini Trigger #2 case — .yaml not .yml."""
    cf_dir = home / ".cloudflared"
    cf_dir.mkdir(parents=True, exist_ok=True)
    (cf_dir / "config.yaml").write_text("tunnel: abc\n")


def _write_config_json(home: Path, role_value: str) -> None:
    cfg_dir = home / "maxim"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text('{"_format_version": "1.0", "role": "' + role_value + '"}')


# ─────────────────────────────────────────────────────────────────────────────
# Eight-cell regression matrix (CR2 + C-3 fold)
# ─────────────────────────────────────────────────────────────────────────────


class TestSevenRankPrecedence:
    """The eight-cell matrix that pins every Mac Mini failure mode."""

    def test_env_var_wins_over_config_json(self, fake_home, monkeypatch):
        """Rank 1: explicit env always trumps config."""
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        _write_config_json(fake_home, "peer")
        result, source = role.detect_role(argv=[])
        assert result == "leader"
        assert source == "env_var"

    def test_config_json_role_wins_over_peer_yml_legacy(self, fake_home):
        """Rank 2 over rank 5: explicit config.json beats legacy
        peer.yml-implies-peer signal. Closes Mac Mini Trigger #3 in
        the canonical form."""
        _write_config_json(fake_home, "leader")
        _write_peer_yml(fake_home)
        result, source = role.detect_role(argv=[])
        assert result == "leader"
        assert source == "config_json"

    def test_stale_peer_yml_plus_cloudflared_yml_resolves_to_leader_not_peer(self, fake_home):
        """Rank 4 over rank 5: the actual Mac Mini bug. Stale peer.yml
        + cloudflared (.yml) → leader, not peer."""
        _write_peer_yml(fake_home)
        _write_cloudflared_yml(fake_home)
        result, source = role.detect_role(argv=[])
        assert result == "leader"
        assert source == "cloudflared"

    def test_stale_peer_yml_plus_cloudflared_yaml_resolves_to_leader_not_peer(self, fake_home):
        """Mac Mini Trigger #2 fix: extension widening. The .yaml
        variant of cloudflared config also outranks peer.yml."""
        _write_peer_yml(fake_home)
        _write_cloudflared_yaml(fake_home)
        result, source = role.detect_role(argv=[])
        assert result == "leader"
        assert source == "cloudflared"

    def test_cli_flag_solo_vs_cloudflared_leader_precedence(self, fake_home):
        """Rank 4 over rank 6: --llm + cloudflared present → leader.
        Cloudflared is a system-level "this box is provisioned as a
        tunneled leader" signal that overrides the local-only --llm
        hint."""
        _write_cloudflared_yml(fake_home)
        result, source = role.detect_role(argv=["--llm", "mistral-7b"])
        assert result == "leader"
        assert source == "cloudflared"

    def test_cli_flag_solo_with_no_peer_signals(self, fake_home):
        """Rank 6: --llm alone → solo (no higher-rank signals)."""
        result, source = role.detect_role(argv=["--llm", "mistral-7b"])
        assert result == "solo"
        assert source == "cli_flag"

    def test_no_signals_at_all_defaults_to_leader(self, fake_home):
        """Rank 7: nothing matched → leader fallback."""
        result, source = role.detect_role(argv=[])
        assert result == "leader"
        assert source == "default"

    def test_leader_mode_detect_role_returns_same_role_as_role_py(self, fake_home):
        """Wrapper consistency: every cell in the 8-cell matrix must
        return the same role (modulo peer↔client translation) from
        both surfaces."""
        # Iterate the same cells via the wrapper
        scenarios = [
            ("env_var", lambda h, m: m.setenv("MAXIM_ROLE", "leader"), "leader"),
            (
                "config_json_over_peer_yml",
                lambda h, m: (_write_config_json(h, "leader"), _write_peer_yml(h)),
                "leader",
            ),
            (
                "stale_peer_yml_plus_cloudflared",
                lambda h, m: (_write_peer_yml(h), _write_cloudflared_yml(h)),
                "leader",
            ),
            (
                "stale_peer_yml_plus_cloudflared_yaml",
                lambda h, m: (_write_peer_yml(h), _write_cloudflared_yaml(h)),
                "leader",
            ),
            (
                "default_only",
                lambda h, m: None,
                "leader",
            ),
        ]
        for name, setup, expected_role in scenarios:
            # Need a fresh monkeypatch + fresh home per cell
            with pytest.MonkeyPatch.context() as mp:
                fresh_home = fake_home / f"cell_{name}"
                fresh_home.mkdir(exist_ok=True)
                mp.setenv("XDG_CONFIG_HOME", str(fresh_home))
                mp.setenv("HOME", str(fresh_home))
                mp.delenv("MAXIM_ROLE", raising=False)
                from maxim.runtime.config_loader import reset_config_cache

                reset_config_cache()
                setup(fresh_home, mp)

                role_value, _ = role.detect_role(argv=[])
                wrapper_decision = leader_mode.detect_role()

                # role.py returns peer; wrapper returns client for back-compat
                expected_wrapper_role = "client" if role_value == "peer" else role_value
                assert wrapper_decision.role == expected_wrapper_role, (
                    f"Cell {name}: role.py returned {role_value!r} but wrapper returned {wrapper_decision.role!r}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Additional precedence cases beyond the 8-cell matrix
# ─────────────────────────────────────────────────────────────────────────────


class TestAdditionalPrecedence:
    def test_mesh_yml_wins_over_cloudflared(self, fake_home):
        """Rank 3 over rank 4: mesh.yml is an explicit multi-node
        topology signal; it outranks cloudflared."""
        _write_mesh_yml(fake_home)
        _write_cloudflared_yml(fake_home)
        result, source = role.detect_role(argv=[])
        assert result == "peer"
        assert source == "mesh_yml"

    def test_env_role_invalid_falls_through(self, fake_home, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "boss")
        result, source = role.detect_role(argv=[])
        assert result == "leader"
        assert source == "default"

    def test_env_role_legacy_client_coerces_to_peer(self, fake_home, monkeypatch):
        """Legacy ``MAXIM_ROLE=client`` from pre-C3 leader_mode
        vocabulary coerces to peer silently at the env-var rank."""
        monkeypatch.setenv("MAXIM_ROLE", "client")
        result, source = role.detect_role(argv=[])
        assert result == "peer"
        assert source == "env_var"

    def test_config_json_invalid_role_falls_through(self, fake_home):
        """An invalid role in config.json should not crash detection —
        it should fall through to lower ranks. The config_loader
        validation raises at parse, which _config_json_role catches."""
        cfg_dir = fake_home / "maxim"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text('{"_format_version": "1.0", "role": "boss"}')
        # Fall through to default
        result, source = role.detect_role(argv=[])
        assert result == "leader"
        assert source == "default"

    def test_config_json_missing_falls_through_normally(self, fake_home):
        """No config.json file → falls through to mesh.yml / peer.yml /
        cloudflared / default. With nothing else set, defaults to leader."""
        result, source = role.detect_role(argv=[])
        assert result == "leader"
        assert source == "default"

    def test_peer_yml_alone_resolves_to_peer_via_auto_migration(self, fake_home):
        """The legitimate zero-config peer flow: only peer.yml present,
        no cloudflared, no config.json.

        Post-C4 IM5 fold: the auto-migration shim fires when
        config.json is read (via _config_json_role at rank 2), writing
        config.json from peer.yml. Detection then returns ``peer`` via
        ``config_json`` source — not ``peer_yml`` — because the
        migration ran first. peer.yml stays in place for back-compat.

        This is the intended IM5 behavior: existing peer.yml-only
        setups get a one-time migration on first startup and pick up
        the canonical config.json source thereafter."""
        _write_peer_yml(fake_home)
        result, source = role.detect_role(argv=[])
        assert result == "peer"
        # Post-IM5: migration shim ran during _config_json_role, so
        # config.json now exists and wins at rank 2
        assert source == "config_json"

    def test_peer_yml_alone_with_cloudflared_present_stays_peer_yml(self, fake_home):
        """Negative case: when cloudflared is present, the migration
        shim does NOT fire (preserves the leader case). peer.yml then
        contributes at rank 5. But cloudflared at rank 4 still wins —
        so this pins the rank-4-over-rank-5 ordering."""
        _write_peer_yml(fake_home)
        _write_cloudflared_yml(fake_home)
        result, source = role.detect_role(argv=[])
        assert result == "leader"
        assert source == "cloudflared"

    def test_cloudflared_yaml_extension_alone(self, fake_home):
        """Extension-widening unit test: .yaml alone (no .yml) is
        accepted as the cloudflared signal."""
        _write_cloudflared_yaml(fake_home)
        result, source = role.detect_role(argv=[])
        assert result == "leader"
        assert source == "cloudflared"

    def test_subcommand_argv_does_not_trigger_solo(self, fake_home):
        """maxim tunnel --llm foo → must NOT detect solo via --llm.
        The subcommand-name guard at rank 6 prevents it."""
        result, source = role.detect_role(argv=["tunnel", "--llm", "foo"])
        # No higher rank fired → falls to default leader
        assert result == "leader"
        assert source == "default"


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper translation: peer ↔ client back-compat
# ─────────────────────────────────────────────────────────────────────────────


class TestWrapperTranslation:
    def test_wrapper_returns_client_for_role_py_peer(self, fake_home):
        """Back-compat: existing consumers compare .role == "client"."""
        _write_peer_yml(fake_home)
        decision = leader_mode.detect_role()
        assert decision.role == "client"
        assert decision.bind_host == "127.0.0.1"

    def test_wrapper_returns_leader_unchanged(self, fake_home):
        _write_cloudflared_yml(fake_home)
        decision = leader_mode.detect_role()
        assert decision.role == "leader"
        assert decision.bind_host == "0.0.0.0"
        assert decision.is_leader is True

    def test_wrapper_returns_solo_unchanged(self, fake_home):
        decision = leader_mode.detect_role()  # default leader actually
        # default = leader, not solo, when no signals present
        assert decision.role == "leader"

    def test_wrapper_solo_when_cli_flag_only(self, fake_home, monkeypatch):
        """The wrapper doesn't see argv directly — argv-driven cells
        only fire through detect_role(argv=...). Verify the wrapper
        with MAXIM_ROLE=solo set."""
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        decision = leader_mode.detect_role()
        assert decision.role == "solo"
        assert decision.bind_host == "127.0.0.1"

    def test_wrapper_reason_includes_signal_name(self, fake_home):
        _write_cloudflared_yml(fake_home)
        decision = leader_mode.detect_role()
        assert "cloudflared" in decision.reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# role_divergence event back-compat (N-2 fold)
# ─────────────────────────────────────────────────────────────────────────────


class TestRoleDivergenceCompat:
    def test_divergence_event_fires_at_debug_with_deprecated_marker(self, fake_home, caplog):
        """N-2 fold: keep emitting role_divergence at DEBUG with
        deprecated:true so external dashboards continue to see it.
        Drops in 1.2."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="maxim.runtime.role"):
            role.detect_and_apply_role(argv=[])

        # The event was emitted at DEBUG and tagged via extra
        divergence_records = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG
            and getattr(r, "event", None) == "role_divergence"
        ]
        assert divergence_records, (
            "role_divergence event must continue firing at DEBUG for "
            "back-compat per N-2 fold"
        )
        data = getattr(divergence_records[-1], "data", {})
        assert data.get("deprecated") is True
        assert "single detector" in data.get("reason", "")


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry: role_detected carries config_json_present
# ─────────────────────────────────────────────────────────────────────────────


class TestRoleDetectedTelemetry:
    def test_role_detected_event_carries_config_json_present_field(self, fake_home, caplog, monkeypatch):
        """C3 telemetry addition: role_detected gains a
        config_json_present boolean so adoption can be tracked."""
        import logging

        _write_config_json(fake_home, "leader")
        # Need to clear MAXIM_ROLE so the autouse fixture doesn't
        # interfere
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        with caplog.at_level(logging.INFO, logger="maxim.runtime.role"):
            role.detect_and_apply_role(argv=[])

        # log_structured attaches data via extra={"event":..., "data":...}
        # (structured_logging.py:374), which surfaces as LogRecord attrs.
        info_records = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and getattr(r, "event", None) == "role_detected"
        ]
        assert info_records, "role_detected event must fire at INFO"
        data = getattr(info_records[-1], "data", {})
        assert "config_json_present" in data, (
            f"role_detected data must carry config_json_present per C3 "
            f"telemetry addition; got keys: {sorted(data.keys())}"
        )
        assert data["config_json_present"] is True
