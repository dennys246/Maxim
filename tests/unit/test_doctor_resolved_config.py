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


# ─────────────────────────────────────────────────────────────────────────────
# VRAM context-fit projection (llm_timeout_scalability.md follow-up)
# ─────────────────────────────────────────────────────────────────────────────


class TestVramContextFitCheck:
    """Cross-platform VRAM projection row in Resolved Config.

    The row answers "will the configured (n_ctx, profile) fit comfortably
    in available VRAM?" — sibling to the proxy.context_admission row.
    """

    def test_row_omitted_when_n_ctx_unset(self, fake_home, monkeypatch):
        """If llm.n_ctx is unresolvable, the admission row already noted
        the absence — VRAM check returns no row to keep doctor output tight."""
        from maxim.doctor.checks import _check_vram_context_pressure

        monkeypatch.delenv("MAXIM_LLM_N_CTX", raising=False)
        monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
        results = _check_vram_context_pressure()
        # No row at all (we don't want two rows screaming about the same gap)
        assert results == []

    def test_no_vram_detected_renders_info_row(self, fake_home, monkeypatch):
        """CPU-only / no GPU systems get an info row, not a false-positive warn."""
        from maxim.doctor import checks as _checks

        monkeypatch.setenv("MAXIM_LLM_N_CTX", "8192")
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "qwen2.5-14b-instruct")

        # Stub detect_compute_resources to report no VRAM (e.g., a CPU-only box)
        monkeypatch.setattr(
            "maxim.runtime.capabilities.detect_compute_resources",
            lambda: (False, "cpu", 0.0, 16.0),
        )
        results = _checks._check_vram_context_pressure()
        assert len(results) == 1
        r = results[0]
        assert r.name == "proxy.vram_context_fit"
        assert r.status == "info"
        assert "no GPU" in r.message or "unified memory" in r.message

    def test_spillover_risk_renders_warn_with_recommendation(self, fake_home, monkeypatch):
        """The Mac Mini incident shape: n_ctx=16384 on a 36 GB system
        projects 34.8 GB — spillover risk. Expect a WARN row with a
        fix string naming both the recommended n_ctx AND the unset-env
        step (because env is the source)."""
        from dataclasses import dataclass

        from maxim.doctor import checks as _checks

        monkeypatch.setenv("MAXIM_LLM_N_CTX", "16384")
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "qwen2.5-32b-instruct")

        monkeypatch.setattr(
            "maxim.runtime.capabilities.detect_compute_resources",
            lambda: (True, "mps", 36.0, 36.0),
        )

        @dataclass(frozen=True)
        class FakeProjection:
            profile: str = "qwen2.5-32b-instruct"
            n_ctx: int = 16384
            weights_gb: float = 19.9
            kv_cache_gb: float = 4.0
            headroom_gb: float = 10.9
            projected_total_gb: float = 34.8
            physical_vram_gb: float = 36.0
            spillover_risk: bool = True
            recommended_n_ctx: int = 13312

        monkeypatch.setattr(
            "maxim.runtime.lane_models.project_vram_usage",
            lambda *a, **kw: FakeProjection(),
        )

        results = _checks._check_vram_context_pressure()
        assert len(results) == 1
        r = results[0]
        assert r.name == "proxy.vram_context_fit"
        assert r.status == "warn"
        assert "34.8 GB" in r.message
        assert "n_ctx=16384" in r.message
        assert "spill" in r.message.lower()
        # Fix string names the recommended value AND the unset-env step
        # (because n_ctx_src == "env")
        assert r.fix is not None
        assert "13312" in r.fix
        assert "unset MAXIM_LLM_N_CTX" in r.fix
        assert "maxim config set llm.n_ctx 13312" in r.fix

    def test_comfortable_fit_renders_ok_no_fix(self, fake_home, monkeypatch):
        from dataclasses import dataclass

        from maxim.doctor import checks as _checks

        monkeypatch.setenv("MAXIM_LLM_N_CTX", "4096")
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "mistral-7b-instruct-v0.2")

        monkeypatch.setattr(
            "maxim.runtime.capabilities.detect_compute_resources",
            lambda: (True, "cuda", 24.0, 64.0),
        )

        @dataclass(frozen=True)
        class FakeProjection:
            profile: str = "mistral-7b-instruct-v0.2"
            n_ctx: int = 4096
            weights_gb: float = 4.5
            kv_cache_gb: float = 1.0
            headroom_gb: float = 1.5
            projected_total_gb: float = 7.0
            physical_vram_gb: float = 24.0
            spillover_risk: bool = False
            recommended_n_ctx: int = 4096

        monkeypatch.setattr(
            "maxim.runtime.lane_models.project_vram_usage",
            lambda *a, **kw: FakeProjection(),
        )

        results = _checks._check_vram_context_pressure()
        assert len(results) == 1
        r = results[0]
        assert r.status == "ok"
        assert "fits comfortably" in r.message
        assert r.fix is None or r.fix == ""

    def test_unknown_profile_renders_info_row(self, fake_home, monkeypatch):
        """Profile not in _BUILTIN_PROFILES → projection returns None
        → info row (not warn, not silently dropped)."""
        from maxim.doctor import checks as _checks

        monkeypatch.setenv("MAXIM_LLM_N_CTX", "8192")
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "fictional-model-42b")

        monkeypatch.setattr(
            "maxim.runtime.capabilities.detect_compute_resources",
            lambda: (True, "cuda", 24.0, 64.0),
        )
        monkeypatch.setattr(
            "maxim.runtime.lane_models.project_vram_usage",
            lambda *a, **kw: None,
        )
        results = _checks._check_vram_context_pressure()
        assert len(results) == 1
        assert results[0].status == "info"
        assert "fictional-model-42b" in results[0].message

    def test_spillover_fix_omits_unset_when_source_is_config(self, fake_home, monkeypatch):
        """If the operator set n_ctx via config.json (not env), the
        fix string should NOT tell them to ``unset MAXIM_LLM_N_CTX``.
        The mesh.yml two-layer-split invariant prefers config.json as
        the canonical surface — when it's already in use, we just
        rewrite it."""
        from dataclasses import dataclass

        from maxim.doctor import checks as _checks
        from maxim.runtime import config_writer

        # Set via config.json (NOT env) so source resolves to "config"
        monkeypatch.delenv("MAXIM_LLM_N_CTX", raising=False)
        monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
        config_writer.set_field("llm.n_ctx", 16384)
        config_writer.set_field("llm.profile", "qwen2.5-32b-instruct")

        monkeypatch.setattr(
            "maxim.runtime.capabilities.detect_compute_resources",
            lambda: (True, "mps", 36.0, 36.0),
        )

        @dataclass(frozen=True)
        class FakeProjection:
            profile: str = "qwen2.5-32b-instruct"
            n_ctx: int = 16384
            weights_gb: float = 19.9
            kv_cache_gb: float = 4.0
            headroom_gb: float = 10.9
            projected_total_gb: float = 34.8
            physical_vram_gb: float = 36.0
            spillover_risk: bool = True
            recommended_n_ctx: int = 13312

        monkeypatch.setattr(
            "maxim.runtime.lane_models.project_vram_usage",
            lambda *a, **kw: FakeProjection(),
        )

        results = _checks._check_vram_context_pressure()
        assert len(results) == 1
        r = results[0]
        assert r.status == "warn"
        # Fix names the new value but NOT the unset step
        assert "13312" in r.fix
        assert "unset MAXIM_LLM_N_CTX" not in r.fix
