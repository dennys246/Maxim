"""Regression guards for C5 of config_unification.md.

The "Resolved Config" doctor section surfaces every absorbed field
with its effective value + source marker. This is the single answer
to "what does this instance think it's configured as?" — collapsing
the pre-C5 cross-reference burden across 96 env vars, 4 config
files, and 2 role detectors.

N2 fold from the pre-implementation review: shadow, convergence,
missing api_key file, mode != 0600, inline-string pre-migration,
keyring not installed, peer.yml deprecation all surface as WARN
rows in this section.
"""

from __future__ import annotations


import pytest

from maxim.doctor.checks import (
    _check_lane_api_key_refs_health,
    _check_peer_yml_deprecation,
    check_resolved_config,
)
from maxim.runtime.config_loader import (
    LaneTierConfig,
    LanesConfigSection,
    LLMConfigSection,
    MaximConfig,
    _FIELD_TO_ENV,
    reset_config_cache,
)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    reset_config_cache()
    yield tmp_path
    reset_config_cache()


def _names(results):
    return [r.name for r in results]


def _by_name(results, name):
    return next(r for r in results if r.name == name)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage — every absorbed field surfaces
# ─────────────────────────────────────────────────────────────────────────────


class TestResolvedConfigCoverage:
    def test_section_shows_every_absorbed_field(self, fake_home):
        results = check_resolved_config()
        names = set(_names(results))
        for field_path in _FIELD_TO_ENV.keys():
            assert field_path in names, f"missing absorbed field {field_path!r}"

    def test_default_config_shows_all_fields_as_info(self, fake_home):
        """No config.json, no env vars set — every field renders at
        ``info`` source=default."""
        results = check_resolved_config()
        for field_path in _FIELD_TO_ENV.keys():
            r = _by_name(results, field_path)
            assert r.status == "info"
            assert "[source=default]" in r.message

    def test_section_starts_with_config_json_path(self, fake_home):
        results = check_resolved_config()
        assert results[0].name == "config.json path"
        # No file yet
        assert "absent" in results[0].message.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Shadow / convergence / divergence (CR3 fold)
# ─────────────────────────────────────────────────────────────────────────────


class TestShadowAndConvergence:
    def test_config_only_shows_source_config(self, fake_home):
        from maxim.runtime.config_writer import write_config

        write_config(MaximConfig(llm=LLMConfigSection(profile="from-config")))
        reset_config_cache()
        results = check_resolved_config()
        r = _by_name(results, "llm.profile")
        assert r.status == "ok"
        assert "[source=config.json]" in r.message
        assert "from-config" in r.message

    def test_env_only_shows_source_env(self, fake_home, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "from-env")
        results = check_resolved_config()
        r = _by_name(results, "llm.profile")
        assert r.status == "ok"
        assert "[source=env]" in r.message
        assert "from-env" in r.message

    def test_env_shadowing_config_warns_with_both_values(self, fake_home, monkeypatch):
        from maxim.runtime.config_writer import write_config

        write_config(MaximConfig(llm=LLMConfigSection(profile="from-config")))
        reset_config_cache()
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "from-env")
        results = check_resolved_config()
        r = _by_name(results, "llm.profile")
        assert r.status == "warn"
        assert "from-env" in r.message
        assert "from-config" in r.message
        assert "shadows" in r.message

    def test_convergence_logs_info_with_marker(self, fake_home, monkeypatch):
        """Pre-implementation CR3 fold: log on convergence even when
        env and config agree, so the operator sees the two-sources-of-
        truth confusion class."""
        from maxim.runtime.config_writer import write_config

        write_config(MaximConfig(llm=LLMConfigSection(profile="qwen-32b")))
        reset_config_cache()
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "qwen-32b")
        results = check_resolved_config()
        r = _by_name(results, "llm.profile")
        assert r.status == "info"
        assert "config.json also sets identically" in r.message

    def test_empty_string_env_var_flagged_per_c1_fold(self, fake_home, monkeypatch):
        """C-1 fold: ``export MAXIM_LLM_PROFILE=`` is treated as unset
        for precedence purposes; the doctor surfaces this as a WARN so
        the operator sees the leaked-empty-export Mac Mini case."""
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "")
        results = check_resolved_config()
        r = _by_name(results, "llm.profile")
        assert r.status == "warn"
        assert "empty string" in r.message
        assert r.fix is not None
        assert "unset MAXIM_LLM_PROFILE" in r.fix


# ─────────────────────────────────────────────────────────────────────────────
# API key ref health (N2 fold)
# ─────────────────────────────────────────────────────────────────────────────


class TestApiKeyRefHealth:
    def test_path_mode_with_missing_file_warns(self, fake_home, tmp_path):
        from maxim.runtime.config_writer import write_config

        missing = tmp_path / "does-not-exist"
        write_config(
            MaximConfig(
                lanes=LanesConfigSection(
                    large=LaneTierConfig(
                        remote_url="http://leader.local/v1",
                        remote_api_key_ref=str(missing),
                    ),
                ),
            )
        )
        reset_config_cache()
        from maxim.runtime.config_loader import load_config

        out = _check_lane_api_key_refs_health(load_config())
        names = _names(out)
        assert "lanes.large.remote_api_key_ref" in names
        r = _by_name(out, "lanes.large.remote_api_key_ref")
        assert r.status == "warn"
        assert "file missing" in r.message

    def test_path_mode_with_wrong_mode_warns(self, fake_home, tmp_path):
        from maxim.runtime.config_writer import write_config

        key_file = tmp_path / "permissive_key"
        key_file.write_text("sk-permissive\n")
        # Default mode is 0o644 — explicitly chmod to confirm
        key_file.chmod(0o644)
        write_config(
            MaximConfig(
                lanes=LanesConfigSection(
                    large=LaneTierConfig(
                        remote_url="http://leader.local/v1",
                        remote_api_key_ref=str(key_file),
                    ),
                ),
            )
        )
        reset_config_cache()
        from maxim.runtime.config_loader import load_config

        out = _check_lane_api_key_refs_health(load_config())
        r = _by_name(out, "lanes.large.remote_api_key_ref")
        assert r.status == "warn"
        assert "mode" in r.message
        assert "chmod 0600" in (r.fix or "")

    def test_path_mode_with_correct_mode_no_warning(self, fake_home, tmp_path):
        from maxim.runtime.config_writer import write_config

        key_file = tmp_path / "good_key"
        key_file.write_text("sk-good\n")
        key_file.chmod(0o600)
        write_config(
            MaximConfig(
                lanes=LanesConfigSection(
                    large=LaneTierConfig(
                        remote_url="http://leader.local/v1",
                        remote_api_key_ref=str(key_file),
                    ),
                ),
            )
        )
        reset_config_cache()
        from maxim.runtime.config_loader import load_config

        out = _check_lane_api_key_refs_health(load_config())
        names = _names(out)
        # No warning for this tier
        assert "lanes.large.remote_api_key_ref" not in names

    def test_inline_env_key_flagged_for_migration(self, fake_home, monkeypatch):
        """N2 fold: legacy env-var inline keys should surface as a
        WARN with a migration fix-hint."""
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", "sk-inline-legacy")
        from maxim.runtime.config_loader import load_config

        out = _check_lane_api_key_refs_health(load_config())
        names = _names(out)
        assert "lanes.large.remote_api_key_ref" in names
        r = _by_name(out, "lanes.large.remote_api_key_ref")
        assert r.status == "warn"
        assert "inline key from env" in r.message
        assert "chmod 0600" in (r.fix or "")


# ─────────────────────────────────────────────────────────────────────────────
# peer.yml deprecation (C4 IM5 fold)
# ─────────────────────────────────────────────────────────────────────────────


class TestPeerYmlDeprecation:
    def test_peer_yml_absent_returns_empty(self, fake_home):
        out = _check_peer_yml_deprecation()
        assert out == []

    def test_peer_yml_present_warns(self, fake_home):
        peer_dir = fake_home / "maxim"
        peer_dir.mkdir(parents=True, exist_ok=True)
        (peer_dir / "peer.yml").write_text("url: http://leader/v1\napi_key: sk\n")
        out = _check_peer_yml_deprecation()
        assert len(out) == 1
        assert out[0].status == "warn"
        assert "deprecated" in out[0].message.lower()
        assert "maxim peer forget" in (out[0].fix or "")


# ─────────────────────────────────────────────────────────────────────────────
# Invalid config.json surfaces as a fail row
# ─────────────────────────────────────────────────────────────────────────────


class TestInvalidConfigJson:
    def test_invalid_json_surfaces_as_fail(self, fake_home):
        cfg_dir = fake_home / "maxim"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text("{not valid json")
        reset_config_cache()
        out = check_resolved_config()
        # Single fail row mentioning the file
        assert any(r.status == "fail" and r.name == "config.json" for r in out)
        fail = next(r for r in out if r.name == "config.json")
        assert "invalid JSON" in fail.message or "load" in fail.message.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Integration: run_all_checks includes the section
# ─────────────────────────────────────────────────────────────────────────────


class TestSectionIntegration:
    def test_run_all_checks_includes_resolved_config_section(self, fake_home):
        from maxim.doctor.checks import run_all_checks
        from maxim.doctor.platform_detect import detect_platform

        info = detect_platform()
        # role=solo to skip the peer-mode branch (and avoid network probes)
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("MAXIM_ROLE", "solo")
            mp.setattr("maxim.runtime.role._cloudflared_config_exists", lambda: None)
            mp.setattr("maxim.runtime.role._peer_yml_exists", lambda: False)
            mp.setattr("maxim.runtime.role._mesh_yml_exists", lambda: False)
            mp.setattr("maxim.runtime.role._config_json_role", lambda: None)
            sections = run_all_checks(info, role="solo")
        section_names = [name for name, _ in sections]
        assert "Resolved Config" in section_names
