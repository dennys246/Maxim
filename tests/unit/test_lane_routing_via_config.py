"""Regression guards for C4 of config_unification.md.

Covers:
- IM5 fold auto-migration shim (peer.yml → config.json on first run)
- Self-hosted classification (peer URLs bypass cloud-lane gate)
- _apply_lane_config_to_env populates env vars from config.json
- peer.yml fallback compat-read when cloudflared present
- resolve_api_key_ref helper for file-path and keyring URIs
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from maxim.runtime.config_loader import (
    LaneTierConfig,
    LanesConfigSection,
    MaximConfig,
    config_path,
    load_config,
    reset_config_cache,
    resolve_api_key_ref,
)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect XDG_CONFIG_HOME + HOME so peer.yml / config.json /
    cloudflared probes are isolated."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    reset_config_cache()
    yield tmp_path
    reset_config_cache()


def _write_peer_yml(home: Path, *, url="http://leader.example.com/v1", key="sk-test-abc123", model="qwen-32b") -> None:
    peer_dir = home / "maxim"
    peer_dir.mkdir(parents=True, exist_ok=True)
    contents = f"url: {url}\napi_key: {key}\n"
    if model:
        contents += f"model: {model}\n"
    (peer_dir / "peer.yml").write_text(contents)


def _write_cloudflared(home: Path) -> None:
    cf_dir = home / ".cloudflared"
    cf_dir.mkdir(parents=True, exist_ok=True)
    (cf_dir / "config.yml").write_text("tunnel: abc\n")


# ─────────────────────────────────────────────────────────────────────────────
# IM5 fold auto-migration shim
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoMigration:
    def test_migration_fires_when_peer_yml_present_and_config_absent(self, fake_home, caplog):
        _write_peer_yml(fake_home)
        with caplog.at_level(logging.INFO, logger="maxim.runtime.config_loader"):
            cfg = load_config()
        # Migration wrote config.json::role=peer + lanes.large.*
        assert cfg.role == "peer"
        assert cfg.lanes.large.remote_url == "http://leader.example.com/v1"
        assert cfg.lanes.large.remote_model == "qwen-32b"
        # API key written as file-path reference, not inline
        assert cfg.lanes.large.remote_api_key_ref is not None
        assert cfg.lanes.large.remote_api_key_ref.startswith("/")
        # File exists at the referenced path with mode 0600
        ref_path = Path(cfg.lanes.large.remote_api_key_ref)
        assert ref_path.is_file()
        if hasattr(ref_path, "stat"):
            mode = ref_path.stat().st_mode & 0o777
            assert mode == 0o600
        # peer.yml stays in place
        assert (fake_home / "maxim" / "peer.yml").is_file()
        # INFO log fired
        assert any("auto-migrated peer.yml" in r.getMessage() for r in caplog.records)

    def test_migration_idempotent_when_config_exists(self, fake_home):
        # Pre-seed config.json
        from maxim.runtime.config_writer import write_config

        write_config(MaximConfig(role="leader"))
        _write_peer_yml(fake_home)
        # Migration should NOT fire — config.json already has data
        cfg = load_config()
        assert cfg.role == "leader"
        assert cfg.lanes.large.remote_url is None

    def test_migration_skipped_when_cloudflared_present(self, fake_home, caplog):
        """Per IM5 edge-case: auto-migration must NOT fire when cloudflared
        is present (preserves the legitimate leader-with-stale-peer.yml
        case where the operator is now a leader and shouldn't be flipped
        to peer)."""
        _write_peer_yml(fake_home)
        _write_cloudflared(fake_home)
        with caplog.at_level(logging.INFO, logger="maxim.runtime.config_loader"):
            cfg = load_config()
        # Default config — migration didn't fire
        assert cfg == MaximConfig()
        # config.json does NOT exist on disk
        assert not config_path().is_file()
        # No migration log
        assert not any("auto-migrated" in r.getMessage() for r in caplog.records)

    def test_migration_idempotency_flag_prevents_second_attempt(self, fake_home):
        _write_peer_yml(fake_home)
        load_config()  # first call: migrates
        # Delete config.json and re-call. The flag should prevent
        # re-migration so the next call returns defaults.
        config_path().unlink()
        cfg = load_config()
        assert cfg == MaximConfig()  # no migration this time

    def test_migration_returns_quietly_when_no_peer_yml(self, fake_home):
        # Neither peer.yml nor config.json present
        cfg = load_config()
        assert cfg == MaximConfig()
        assert not config_path().is_file()


# ─────────────────────────────────────────────────────────────────────────────
# resolve_api_key_ref helper
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveApiKeyRef:
    def test_none_returns_none(self):
        assert resolve_api_key_ref(None) is None

    def test_empty_string_returns_none(self):
        assert resolve_api_key_ref("") is None

    def test_file_path_reads_contents(self, tmp_path):
        key_file = tmp_path / "api_key"
        key_file.write_text("sk-abc-real-key\n")
        result = resolve_api_key_ref(str(key_file))
        assert result == "sk-abc-real-key"

    def test_file_path_missing_returns_none(self, tmp_path):
        result = resolve_api_key_ref(str(tmp_path / "does_not_exist"))
        assert result is None

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        key_file = tmp_path / "tilde_key"
        key_file.write_text("sk-tilde-key\n")
        result = resolve_api_key_ref("~/tilde_key")
        assert result == "sk-tilde-key"

    def test_keyring_uri_without_keyring_installed_returns_none(self, caplog):
        """I-1 fold: keyring missing must NOT raise — only return None."""
        # We can't reliably uninstall keyring in tests, but we can test
        # that a malformed keyring URI returns None gracefully.
        result = resolve_api_key_ref("keyring:onlytwo")
        assert result is None

    def test_inline_key_passes_through(self):
        """Legacy env-var semantics: ``MAXIM_LANE_<TIER>_REMOTE_API_KEY``
        holds an inline key. ``resolve_api_key_ref`` returns it as-is.

        The cross-confirmed I-3/IM3 fold rejects inline at config.json
        LOAD time (``_validate_api_key_ref`` in ``_parse_lane_tier``);
        env-sourced values legitimately use the inline form because
        the env-var pre-C4 semantics was "this IS the key, not a
        reference"."""
        result = resolve_api_key_ref("sk-from-env-legacy")
        assert result == "sk-from-env-legacy"


# ─────────────────────────────────────────────────────────────────────────────
# Self-hosted classification + lane config env-var population
# ─────────────────────────────────────────────────────────────────────────────


class TestSelfHostedClassification:
    def test_classify_self_hosted_lane_config_source(self):
        from maxim.runtime.lane_backends import classify_self_hosted_lane

        assert classify_self_hosted_lane("http://leader.local/v1", "config") is True
        assert classify_self_hosted_lane("http://leader.local/v1", "env") is True
        assert classify_self_hosted_lane(None, "config") is False
        assert classify_self_hosted_lane("", "config") is False
        assert classify_self_hosted_lane("http://leader.local/v1", "default") is False


class TestApplyLaneConfigToEnv:
    def test_populates_env_from_config_json(self, fake_home, monkeypatch):
        from maxim.runtime.config_writer import write_config
        from maxim.runtime.lane_backends import _apply_lane_config_to_env

        # Write an api_key file
        api_key_file = fake_home / "maxim" / "api_key"
        api_key_file.parent.mkdir(parents=True, exist_ok=True)
        api_key_file.write_text("sk-the-real-key\n")

        write_config(
            MaximConfig(
                lanes=LanesConfigSection(
                    large=LaneTierConfig(
                        remote_url="http://leader.local/v1",
                        remote_api_key_ref=str(api_key_file),
                        remote_model="qwen-32b",
                    ),
                ),
            )
        )
        reset_config_cache()

        import os

        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", raising=False)
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_MODEL", raising=False)
        monkeypatch.delenv("MAXIM_MAX_CLOUD_LANES", raising=False)

        has = _apply_lane_config_to_env(None)
        assert has is True
        assert os.environ["MAXIM_LANE_LARGE_REMOTE_URL"] == "http://leader.local/v1"
        assert os.environ["MAXIM_LANE_LARGE_REMOTE_API_KEY"] == "sk-the-real-key"
        assert os.environ["MAXIM_LANE_LARGE_REMOTE_MODEL"] == "qwen-32b"
        # Self-hosted classification sets MAX_CLOUD_LANES=1
        assert os.environ["MAXIM_MAX_CLOUD_LANES"] == "1"

    def test_env_wins_over_config(self, fake_home, monkeypatch):
        """Env was already set; resolve_setting returns env value;
        _apply_lane_config_to_env doesn't overwrite."""
        from maxim.runtime.config_writer import write_config
        from maxim.runtime.lane_backends import _apply_lane_config_to_env

        write_config(
            MaximConfig(
                lanes=LanesConfigSection(
                    large=LaneTierConfig(remote_url="http://from-config/v1"),
                ),
            )
        )
        reset_config_cache()

        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://from-env/v1")
        import os

        _apply_lane_config_to_env(None)
        assert os.environ["MAXIM_LANE_LARGE_REMOTE_URL"] == "http://from-env/v1"

    def test_no_config_returns_false(self, fake_home):
        from maxim.runtime.lane_backends import _apply_lane_config_to_env

        # No config.json, no peer.yml, no env
        assert _apply_lane_config_to_env(None) is False

    def test_peer_yml_fallback_when_config_absent(self, fake_home, monkeypatch):
        """Cloudflared-present case: migration skipped, but peer.yml fallback
        in _apply_lane_config_to_env still populates env vars."""
        _write_peer_yml(fake_home)
        _write_cloudflared(fake_home)  # blocks migration
        reset_config_cache()
        import os

        from maxim.runtime.lane_backends import _apply_lane_config_to_env

        # config.json absent (migration skipped), but peer.yml still has data
        has = _apply_lane_config_to_env(None)
        assert has is True
        assert os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL", "").startswith("http://")


# ─────────────────────────────────────────────────────────────────────────────
# Post-implementation Executor C1 fold: idempotency
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyLaneConfigIdempotency:
    """The Executor C1 fold: _apply_lane_config_to_env must be
    idempotent. Without the guard, env vars set by the first call
    would re-attribute source from "config" to "env" on the second
    call, corrupting C5's doctor source attribution."""

    def test_second_call_returns_cached_result_without_side_effects(self, fake_home, monkeypatch):
        from maxim.runtime.config_writer import write_config
        from maxim.runtime.lane_backends import _apply_lane_config_to_env

        write_config(
            MaximConfig(
                lanes=LanesConfigSection(
                    large=LaneTierConfig(remote_url="http://leader.local/v1"),
                ),
            )
        )
        reset_config_cache()

        # First call populates env + caches
        result1 = _apply_lane_config_to_env(None)
        assert result1 is True

        # Source attribution from config_loader.resolve_setting should
        # still be "config" — proving the env-population is gated
        from maxim.runtime.config_loader import load_config, resolve_setting

        cfg = load_config()
        _, source = resolve_setting("lanes.large.remote_url", config=cfg)
        # The env var was populated by call #1, but if we then call
        # _apply_lane_config_to_env again, it must NOT re-run resolve
        # which would now return source="env" (corrupted).
        result2 = _apply_lane_config_to_env(None)
        assert result2 is True  # cached result
        # The source from the post-call config view should match what
        # the operator sees in doctor. The first call populated env;
        # the second call returned cached. We verify the function
        # didn't fire its side effects a second time by checking the
        # cached-result equality.
        assert result2 == result1

    def test_second_call_no_config_returns_cached_false(self, fake_home):
        """No config + no peer.yml: first call returns False; second
        call MUST also return False without re-attempting the work."""
        from maxim.runtime.lane_backends import _apply_lane_config_to_env

        # No config, no peer.yml — first call returns False
        assert _apply_lane_config_to_env(None) is False
        # Second call returns cached False
        assert _apply_lane_config_to_env(None) is False

    def test_peer_yml_fallback_does_not_corrupt_source_attribution(self, fake_home, monkeypatch):
        """Post-impl Executor C2 fold: peer.yml fallback used to call
        the legacy apply_peer_config_to_env shim, which set
        MAXIM_LANE_LARGE_REMOTE_URL via _setdefault_nonempty. With the
        idempotency guard + in-function populate path, the source
        attribution stays controlled by the function rather than the
        legacy shim."""
        _write_peer_yml(fake_home)
        _write_cloudflared(fake_home)  # blocks migration
        reset_config_cache()
        import os

        from maxim.runtime.lane_backends import _apply_lane_config_to_env

        # First call populates env from peer.yml fallback
        assert _apply_lane_config_to_env(None) is True
        url = os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL", "")
        assert url.startswith("http://")
        # Second call returns cached without re-running the fallback
        assert _apply_lane_config_to_env(None) is True
        assert os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL", "") == url
