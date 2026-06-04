"""Regression guards for ``maxim.runtime.config_loader`` (C1 of
config_unification.md).

Test surface (matches the plan's "Regression guards summary" section):

* Schema + loader: minimum valid config, CC3 path markers, defaults
* Precedence chain: CLI > env > config > default, parametrized over
  every absorbed field
* C-1: empty string env vars treated as unset
* C-2: coercion table truthy/falsy/range cases
* CR3: convergence AND divergence logging
* I-3 / IM3: inline-string API-key refs rejected at load time
* I-4: deprecation INFO fires exactly once per env var per startup
* I-6 / IM4: unknown-key handling tied to ``_format_version``
* IM1: every section dataclass declares its CC3 path in its docstring
* N1: ``_format_version`` is the first field of :class:`MaximConfig`
* Version handling: same-version strict, future-minor tolerant,
  future-major rejected
* Path resolution + JSON parse errors

The autouse ``_isolate_config_json_env`` fixture in conftest.py scrubs
the absorbed env vars + clears the loader's singleton between tests.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import fields
from pathlib import Path

import pytest

from maxim.exceptions import ConfigurationError
from maxim.runtime.config_loader import (
    CONFIG_FORMAT_VERSION,
    AutoSpawnConfigSection,
    CloudConfigSection,
    DataConfigSection,
    LaneTierConfig,
    LanesConfigSection,
    LLMConfigSection,
    MaximConfig,
    ProxyConfigSection,
    _ABSORBED_ENV_VARS,
    _env_is_set,
    _FIELD_TO_ENV,
    _reset_warned_envs,
    config_path,
    get_config,
    load_config,
    reset_config_cache,
    resolve_setting,
)


# ─────────────────────────────────────────────────────────────────────────────
# Schema-shape tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMinimumValidConfig:
    """Empty file / missing file → all schema defaults."""

    def test_missing_file_returns_defaults_only(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.json")
        assert cfg == MaximConfig()
        assert cfg.role is None
        assert cfg.llm.enabled is True
        assert cfg.llm.n_ctx == 8192
        assert cfg.llm.backend == "llama_cpp"
        assert cfg.lanes.large.remote_url is None
        assert cfg.cloud.enabled is False
        assert cfg.proxy.max_concurrent == 4
        assert cfg.auto_spawn.llm_server is True
        assert cfg.data.home is None

    def test_empty_file_returns_defaults(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("")
        assert load_config(path) == MaximConfig()

    def test_whitespace_only_file_returns_defaults(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("   \n   \t\n")
        assert load_config(path) == MaximConfig()

    def test_empty_object_returns_defaults(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{}")
        # Empty object has no _format_version; the loader treats absence
        # as same-version since dict.get defaults to CONFIG_FORMAT_VERSION.
        assert load_config(path) == MaximConfig()


class TestFormatVersionFieldOrder:
    """N1 fold: ``_format_version`` declared first."""

    def test_format_version_is_first_dataclass_field(self):
        names = [f.name for f in fields(MaximConfig)]
        assert names[0] == "_format_version"

    def test_format_version_default_matches_module_constant(self):
        cfg = MaximConfig()
        assert cfg._format_version == CONFIG_FORMAT_VERSION == "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# CC3 path declarations (IM1 fold)
# ─────────────────────────────────────────────────────────────────────────────


class TestCC3PathDeclarations:
    """Every section dataclass declares its CC3 path in its docstring."""

    @pytest.mark.parametrize(
        "section_cls",
        [
            MaximConfig,
            LLMConfigSection,
            LanesConfigSection,
            CloudConfigSection,
            ProxyConfigSection,
            AutoSpawnConfigSection,
            DataConfigSection,
        ],
    )
    def test_shape_frozen_sections_declare_path_b(self, section_cls):
        doc = section_cls.__doc__ or ""
        assert "SHAPE-FROZEN at 1.0 (CC3)" in doc, (
            f"{section_cls.__name__} must declare path (b) shape-frozen "
            f"per CLAUDE.md CC3 invariant; missing 'SHAPE-FROZEN at 1.0 "
            f"(CC3)' marker in docstring"
        )

    def test_lane_tier_extra_collision_with_declared_field_raises_at_construction(self):
        """Post-implementation Architecture #4 fold: the collision
        guard moved from ``_serialize_for_json`` (writer-only) into
        ``LaneTierConfig.__post_init__`` so a malformed dataclass
        cannot reach the writer at all. Mirrors CLAUDE.md's "push
        silent-no-op invariants into types, not helpers" discipline.
        """
        from maxim.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="collide with declared fields"):
            LaneTierConfig(
                remote_url="http://leader/v1",
                extra={"remote_url": "http://different/v1"},
            )
        with pytest.raises(ConfigurationError, match="collide with declared fields"):
            LaneTierConfig(extra={"remote_api_key_ref": "~/foo"})
        # Genuinely additive forward-growth fields are fine
        ok = LaneTierConfig(
            remote_url="http://leader/v1",
            extra={"remote_health_path": "/health", "remote_timeout_s": 30},
        )
        assert ok.extra == {"remote_health_path": "/health", "remote_timeout_s": 30}

    def test_lane_tier_config_declares_path_a_escape_hatch(self):
        doc = LaneTierConfig.__doc__ or ""
        assert "ESCAPE-HATCH at 1.0 (CC3)" in doc, (
            "LaneTierConfig is the one section type expected to use CC3 path (a) — see config_unification.md IM1 fold"
        )
        # The extra field exists; we test its hash=False/compare=False
        # contract by verifying two LaneTierConfig instances with
        # different extras still compare equal AND hash equal.
        assert any(f.name == "extra" for f in fields(LaneTierConfig))
        a = LaneTierConfig(extra={"x": 1})
        b = LaneTierConfig(extra={"y": 2})
        assert a == b  # compare=False on extra means it doesn't affect equality
        assert hash(a) == hash(b)  # hash=False omits it from the hash too


class TestLaneTierTimeoutField:
    """llm_timeout_scalability.md Stage 2: per-tier timeout_s field."""

    def test_default_is_none(self):
        cfg = LaneTierConfig()
        assert cfg.timeout_s is None

    def test_positive_float_accepted(self):
        cfg = LaneTierConfig(timeout_s=600.0)
        assert cfg.timeout_s == 600.0

    def test_positive_int_accepted(self):
        # int → float at the dataclass level is fine; the loader-side
        # parser converts to float during construction from JSON.
        cfg = LaneTierConfig(timeout_s=120)
        assert cfg.timeout_s == 120

    def test_zero_rejected(self):
        from maxim.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="must be positive"):
            LaneTierConfig(timeout_s=0)

    def test_negative_rejected(self):
        from maxim.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="must be positive"):
            LaneTierConfig(timeout_s=-5.0)

    def test_non_numeric_rejected(self):
        from maxim.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="must be a number"):
            LaneTierConfig(timeout_s="600")  # type: ignore[arg-type]

    def test_resolve_setting_from_env(self, monkeypatch):
        from maxim.runtime.config_loader import resolve_setting

        monkeypatch.setenv("MAXIM_LANE_LARGE_TIMEOUT_S", "450")
        value, source = resolve_setting("lanes.large.timeout_s")
        assert value == 450.0
        assert source == "env"

    def test_resolve_setting_default_is_none(self, monkeypatch):
        from maxim.runtime.config_loader import resolve_setting

        monkeypatch.delenv("MAXIM_LANE_LARGE_TIMEOUT_S", raising=False)
        value, source = resolve_setting("lanes.large.timeout_s")
        assert value is None
        assert source == "default"

    def test_resolve_setting_rejects_malformed_env(self, monkeypatch):
        from maxim.exceptions import ConfigurationError
        from maxim.runtime.config_loader import resolve_setting

        monkeypatch.setenv("MAXIM_LANE_LARGE_TIMEOUT_S", "not-a-number")
        with pytest.raises(ConfigurationError, match="expected float"):
            resolve_setting("lanes.large.timeout_s")

    def test_resolve_setting_rejects_zero_env(self, monkeypatch):
        from maxim.exceptions import ConfigurationError
        from maxim.runtime.config_loader import resolve_setting

        monkeypatch.setenv("MAXIM_LANE_LARGE_TIMEOUT_S", "0")
        with pytest.raises(ConfigurationError, match="must be positive"):
            resolve_setting("lanes.large.timeout_s")

    def test_resolve_setting_rejects_negative_env(self, monkeypatch):
        from maxim.exceptions import ConfigurationError
        from maxim.runtime.config_loader import resolve_setting

        monkeypatch.setenv("MAXIM_LANE_LARGE_TIMEOUT_S", "-10")
        with pytest.raises(ConfigurationError, match="below minimum"):
            resolve_setting("lanes.large.timeout_s")

    def test_parse_lane_tier_from_json_dict(self):
        from maxim.runtime.config_loader import _parse_lane_tier

        section = {"timeout_s": 600}
        cfg = _parse_lane_tier(section, "lanes.large", tolerate_unknown=False)
        assert cfg.timeout_s == 600.0

    def test_parse_lane_tier_rejects_zero_in_json(self):
        from maxim.exceptions import ConfigurationError
        from maxim.runtime.config_loader import _parse_lane_tier

        with pytest.raises(ConfigurationError, match="must be positive"):
            _parse_lane_tier({"timeout_s": 0}, "lanes.large", tolerate_unknown=False)

    def test_parse_lane_tier_rejects_string_in_json(self):
        from maxim.exceptions import ConfigurationError
        from maxim.runtime.config_loader import _parse_lane_tier

        with pytest.raises(ConfigurationError, match="must be a number"):
            _parse_lane_tier({"timeout_s": "600"}, "lanes.large", tolerate_unknown=False)

    def test_parse_lane_tier_rejects_bool_in_json(self):
        """bool is technically a subclass of int in Python — explicit
        guard prevents ``"timeout_s": true`` parsing as 1.0."""
        from maxim.exceptions import ConfigurationError
        from maxim.runtime.config_loader import _parse_lane_tier

        with pytest.raises(ConfigurationError, match="must be a number"):
            _parse_lane_tier({"timeout_s": True}, "lanes.large", tolerate_unknown=False)

    def test_parse_lane_tier_null_keeps_none(self):
        from maxim.runtime.config_loader import _parse_lane_tier

        cfg = _parse_lane_tier({"timeout_s": None}, "lanes.large", tolerate_unknown=False)
        assert cfg.timeout_s is None


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigPath:
    """Path resolution mirrors ``peer_config_path``."""

    def test_posix_uses_xdg_config_home(self, tmp_path, monkeypatch):
        if platform.system() == "Windows":
            pytest.skip("POSIX-only path test")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert config_path() == tmp_path / "maxim" / "config.json"

    def test_posix_falls_back_to_dot_config(self, monkeypatch):
        if platform.system() == "Windows":
            pytest.skip("POSIX-only path test")
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        assert config_path() == Path.home() / ".config" / "maxim" / "config.json"


# ─────────────────────────────────────────────────────────────────────────────
# C-1 fold: empty-string env vars treated as unset
# ─────────────────────────────────────────────────────────────────────────────


class TestEnvIsSet:
    def test_unset_returns_false(self, monkeypatch):
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        assert _env_is_set("MAXIM_ROLE") is False

    def test_empty_string_returns_false(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "")
        assert _env_is_set("MAXIM_ROLE") is False

    def test_whitespace_only_returns_false(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "   \t\n")
        assert _env_is_set("MAXIM_ROLE") is False

    def test_real_value_returns_true(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        assert _env_is_set("MAXIM_ROLE") is True

    def test_value_with_surrounding_whitespace_returns_true(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "  leader  ")
        assert _env_is_set("MAXIM_ROLE") is True


class TestEmptyStringEnvIsTreatedAsUnset:
    """C-1 regression guard at the precedence-chain layer."""

    def test_empty_env_falls_through_to_config(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "")
        cfg = MaximConfig(llm=LLMConfigSection(profile="from-config"))
        value, source = resolve_setting("llm.profile", config=cfg)
        assert source == "config"
        assert value == "from-config"

    def test_empty_env_falls_through_to_default_when_no_config(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "")
        value, source = resolve_setting("llm.n_ctx")
        assert source == "default"
        assert value == 8192


# ─────────────────────────────────────────────────────────────────────────────
# Precedence chain
# ─────────────────────────────────────────────────────────────────────────────


class TestPrecedenceChain:
    """CLI > env > config > default for every absorbed field."""

    def test_cli_wins_over_env_and_config(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "from-env")
        cfg = MaximConfig(llm=LLMConfigSection(profile="from-config"))
        value, source = resolve_setting("llm.profile", cli_value="from-cli", config=cfg)
        assert source == "cli"
        assert value == "from-cli"

    def test_env_wins_over_config(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "from-env")
        cfg = MaximConfig(llm=LLMConfigSection(profile="from-config"))
        value, source = resolve_setting("llm.profile", config=cfg)
        assert source == "env"
        assert value == "from-env"

    def test_config_wins_over_default(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LLM_N_CTX", raising=False)
        cfg = MaximConfig(llm=LLMConfigSection(n_ctx=16384))
        value, source = resolve_setting("llm.n_ctx", config=cfg)
        assert source == "config"
        assert value == 16384

    def test_default_fires_when_nothing_else_set(self):
        value, source = resolve_setting("llm.n_ctx")
        assert source == "default"
        assert value == 8192

    @pytest.mark.parametrize("field_path,env_name", sorted(_FIELD_TO_ENV.items()))
    def test_every_absorbed_field_has_env_mapping(self, field_path, env_name):
        assert env_name.startswith("MAXIM_")
        assert env_name in _ABSORBED_ENV_VARS

    def test_unknown_field_path_raises(self):
        with pytest.raises(ConfigurationError, match="unknown field path"):
            resolve_setting("llm.nonexistent_field")

    def test_auto_loads_config_when_kwarg_omitted(self, tmp_path, monkeypatch):
        """Regression guard for the 2026-06-04 fold: callers that don't
        pass ``config=cfg`` previously got incorrect ``source=default``
        results because ``_read_from_config(None, ...)`` returned ``None``
        immediately. The fix auto-loads via ``get_config()`` (cached).

        Symptom: ``maxim doctor`` showed three contradictory rows about
        ``llm.n_ctx`` — one saying "using default (8192)" and another
        saying "13312 [source=config.json]" for the same field. The
        contradiction came from this exact gap."""
        from maxim.runtime.config_writer import write_config

        monkeypatch.delenv("MAXIM_LLM_N_CTX", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        reset_config_cache()
        write_config(MaximConfig(llm=LLMConfigSection(n_ctx=13312)))
        reset_config_cache()

        value, source = resolve_setting("llm.n_ctx")  # no config= kwarg
        assert source == "config"
        assert value == 13312

    def test_explicit_config_overrides_auto_load(self, tmp_path, monkeypatch):
        """Passing an explicit ``config=cfg`` overrides the on-disk
        config.json. Useful when callers want to resolve against a
        stub config in tests without writing to disk."""
        from maxim.runtime.config_writer import write_config

        monkeypatch.delenv("MAXIM_LLM_N_CTX", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        reset_config_cache()
        write_config(MaximConfig(llm=LLMConfigSection(n_ctx=13312)))
        reset_config_cache()

        explicit_cfg = MaximConfig(llm=LLMConfigSection(n_ctx=2048))
        value, source = resolve_setting("llm.n_ctx", config=explicit_cfg)
        assert source == "config"
        assert value == 2048  # explicit stub wins over on-disk config.json


class TestOverrideLogging:
    """CR3 fold: log on every layer interaction — shadow AND convergence."""

    def test_divergence_logs_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "from-env")
        cfg = MaximConfig(llm=LLMConfigSection(profile="from-config"))
        with caplog.at_level(logging.WARNING, logger="maxim.runtime.config_loader"):
            value, source = resolve_setting("llm.profile", config=cfg)
        assert source == "env"
        assert value == "from-env"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("shadows" in r.getMessage() for r in warnings)
        assert any("from-env" in r.getMessage() and "from-config" in r.getMessage() for r in warnings)

    def test_convergence_logs_info(self, monkeypatch, caplog):
        """The Mac Mini "two sources of truth converge by accident" case."""
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "qwen-32b")
        cfg = MaximConfig(llm=LLMConfigSection(profile="qwen-32b"))
        with caplog.at_level(logging.INFO, logger="maxim.runtime.config_loader"):
            value, source = resolve_setting("llm.profile", config=cfg)
        assert source == "env"
        assert value == "qwen-32b"
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("same value" in r.getMessage() for r in infos)
        # No WARNING should fire — convergence is INFO not WARNING
        warnings_with_shadow = [
            r for r in caplog.records if r.levelno == logging.WARNING and "shadows" in r.getMessage()
        ]
        assert not warnings_with_shadow

    def test_env_only_no_config_logs_no_override(self, monkeypatch, caplog):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "from-env")
        with caplog.at_level(logging.INFO, logger="maxim.runtime.config_loader"):
            value, source = resolve_setting("llm.profile")
        assert source == "env"
        # The deprecation INFO should fire (env var is absorbed), but
        # no "same value" or "shadows" message
        msgs = [r.getMessage() for r in caplog.records]
        assert not any("same value" in m for m in msgs)
        assert not any("shadows" in m for m in msgs)


class TestDeprecationInfo:
    """I-4 fold: once-per-startup INFO log for absorbed env vars."""

    def test_deprecation_info_fires_for_absorbed_env_var(self, monkeypatch, caplog):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "from-env")
        with caplog.at_level(logging.INFO, logger="maxim.runtime.config_loader"):
            resolve_setting("llm.profile")
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("deprecated" in r.getMessage() for r in infos), (
            "Expected once-per-startup deprecation INFO mentioning 'deprecated' to fire when an absorbed env var is set"
        )

    def test_deprecation_info_fires_once_across_100_resolve_calls(self, monkeypatch, caplog):
        """The load-bearing once-per-startup guarantee."""
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "from-env")
        with caplog.at_level(logging.INFO, logger="maxim.runtime.config_loader"):
            for _ in range(100):
                resolve_setting("llm.profile")
        infos = [r for r in caplog.records if r.levelno == logging.INFO and "deprecated" in r.getMessage()]
        assert len(infos) == 1, (
            f"Expected exactly one deprecation INFO across 100 resolve_setting calls; got {len(infos)}"
        )

    def test_reset_warned_envs_re_arms_the_log(self, monkeypatch, caplog):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "from-env")
        with caplog.at_level(logging.INFO, logger="maxim.runtime.config_loader"):
            resolve_setting("llm.profile")
            _reset_warned_envs()
            resolve_setting("llm.profile")
        infos = [r for r in caplog.records if r.levelno == logging.INFO and "deprecated" in r.getMessage()]
        assert len(infos) == 2


# ─────────────────────────────────────────────────────────────────────────────
# C-2 fold: coercion table
# ─────────────────────────────────────────────────────────────────────────────


class TestCoercionBool:
    @pytest.mark.parametrize("raw", ["1", "true", "True", "TRUE", "yes", "on", "  yes  "])
    def test_truthy_values(self, raw, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_ENABLED", raw)
        value, source = resolve_setting("llm.enabled")
        assert source == "env"
        assert value is True

    @pytest.mark.parametrize("raw", ["0", "false", "False", "no", "off", "  off  "])
    def test_falsy_values(self, raw, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_ENABLED", raw)
        value, source = resolve_setting("llm.enabled")
        assert source == "env"
        assert value is False

    @pytest.mark.parametrize("raw", ["maybe", "2", "y", "n", "tru", "fals"])
    def test_invalid_values_raise(self, raw, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_ENABLED", raw)
        with pytest.raises(ConfigurationError, match="bool-like"):
            resolve_setting("llm.enabled")


class TestCoercionInt:
    def test_valid_int(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "16384")
        value, _ = resolve_setting("llm.n_ctx")
        assert value == 16384

    def test_below_minimum_raises(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "100")
        with pytest.raises(ConfigurationError, match="below minimum 256"):
            resolve_setting("llm.n_ctx")

    def test_negative_int_raises(self, monkeypatch):
        monkeypatch.setenv("MAXIM_PROXY_MAX_CONCURRENT", "-1")
        with pytest.raises(ConfigurationError, match="below minimum 0"):
            resolve_setting("proxy.max_concurrent")

    def test_port_out_of_range_raises(self, monkeypatch):
        monkeypatch.setenv("MAXIM_AUTO_SPAWN_PORT", "70000")
        with pytest.raises(ConfigurationError, match="above maximum 65535"):
            resolve_setting("auto_spawn.port")

    def test_port_zero_rejected(self, monkeypatch):
        monkeypatch.setenv("MAXIM_AUTO_SPAWN_PORT", "0")
        with pytest.raises(ConfigurationError, match="below minimum 1"):
            resolve_setting("auto_spawn.port")

    def test_non_numeric_raises(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "huge")
        with pytest.raises(ConfigurationError, match="expected int"):
            resolve_setting("llm.n_ctx")


class TestCoercionFloat:
    def test_valid_float(self, monkeypatch):
        monkeypatch.setenv("MAXIM_CLOUD_SESSION_BUDGET", "12.50")
        value, _ = resolve_setting("cloud.session_budget_usd")
        assert value == 12.50

    def test_negative_float_raises(self, monkeypatch):
        monkeypatch.setenv("MAXIM_CLOUD_SESSION_BUDGET", "-1.5")
        with pytest.raises(ConfigurationError, match="below minimum 0"):
            resolve_setting("cloud.session_budget_usd")


class TestCoercionEnum:
    def test_valid_role(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        value, _ = resolve_setting("role")
        assert value == "leader"

    def test_legacy_client_coerces_to_peer(self, monkeypatch, caplog):
        monkeypatch.setenv("MAXIM_ROLE", "client")
        with caplog.at_level(logging.WARNING, logger="maxim.runtime.config_loader"):
            value, _ = resolve_setting("role")
        assert value == "peer"
        assert any("client" in r.getMessage() for r in caplog.records)

    def test_invalid_role_raises(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "boss")
        with pytest.raises(ConfigurationError, match="expected one of"):
            resolve_setting("role")

    def test_invalid_backend_raises(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_BACKEND", "tensorflow")
        with pytest.raises(ConfigurationError, match="expected one of"):
            resolve_setting("llm.backend")

    def test_invalid_redaction_policy_raises(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_REDACTION_POLICY", "paranoid")
        with pytest.raises(ConfigurationError, match="expected one of"):
            resolve_setting("cloud.redaction_policy")


# ─────────────────────────────────────────────────────────────────────────────
# I-3 / IM3 fold: inline-string API-key refs rejected at load time
# ─────────────────────────────────────────────────────────────────────────────


class TestAPIKeyRefValidation:
    """Cross-confirmed I-3 + IM3 fold: only file-path and keyring URI accepted."""

    def test_file_path_accepted(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "lanes": {"large": {"remote_api_key_ref": "/etc/maxim/api_key"}},
                }
            )
        )
        cfg = load_config(path)
        assert cfg.lanes.large.remote_api_key_ref == "/etc/maxim/api_key"

    def test_tilde_path_accepted(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "lanes": {"large": {"remote_api_key_ref": "~/.config/maxim/api_key"}},
                }
            )
        )
        cfg = load_config(path)
        assert cfg.lanes.large.remote_api_key_ref == "~/.config/maxim/api_key"

    def test_keyring_uri_accepted(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "lanes": {"large": {"remote_api_key_ref": "keyring:maxim:default"}},
                }
            )
        )
        cfg = load_config(path)
        assert cfg.lanes.large.remote_api_key_ref == "keyring:maxim:default"

    def test_inline_sk_key_rejected(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "lanes": {"large": {"remote_api_key_ref": "sk-abc123def456ghi789"}},
                }
            )
        )
        with pytest.raises(ConfigurationError, match="Inline plaintext API keys"):
            load_config(path)

    def test_inline_plain_string_rejected(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "lanes": {"large": {"remote_api_key_ref": "my-secret-token"}},
                }
            )
        )
        with pytest.raises(ConfigurationError, match="must be a file path"):
            load_config(path)

    def test_malformed_keyring_uri_rejected(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "lanes": {"large": {"remote_api_key_ref": "keyring:onlyservice"}},
                }
            )
        )
        with pytest.raises(ConfigurationError, match="keyring URI must be"):
            load_config(path)

    def test_inline_key_via_env_var_passes_through(self, monkeypatch):
        """The cross-confirmed I-3/IM3 fold rejects inline plaintext in
        ``config.json`` (load-time validation in ``_parse_lane_tier``).
        It does NOT apply to env vars: the legacy
        ``MAXIM_LANE_<TIER>_REMOTE_API_KEY`` env var legitimately holds
        the inline key directly — that's its pre-C4 semantics, and the
        C4 migration shim preserves it for back-compat. The validation
        only fires when WRITING to config.json (via set_field) or when
        LOADING config.json."""
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", "sk-abc123")
        value, source = resolve_setting("lanes.large.remote_api_key_ref")
        assert source == "env"
        assert value == "sk-abc123"


# ─────────────────────────────────────────────────────────────────────────────
# I-6 / IM4 fold: unknown-key handling tied to _format_version
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownKeyHandling:
    def test_same_version_unknown_top_level_rejected(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "embodiment": {"robot_name": "reachy"},
                }
            )
        )
        with pytest.raises(ConfigurationError, match="unknown top-level"):
            load_config(path)

    def test_same_version_unknown_nested_rejected(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "llm": {"profile": "qwen-32b", "typo_field": "value"},
                }
            )
        )
        with pytest.raises(ConfigurationError, match="unknown key"):
            load_config(path)

    def test_future_minor_unknown_top_level_tolerated(self, tmp_path, caplog):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.5",
                    "embodiment": {"robot_name": "reachy"},
                }
            )
        )
        with caplog.at_level(logging.WARNING, logger="maxim.runtime.config_loader"):
            cfg = load_config(path)
        assert cfg._format_version == "1.5"
        assert any("future minor" in r.getMessage() for r in caplog.records)

    def test_future_minor_unknown_nested_tolerated(self, tmp_path, caplog):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.5",
                    "llm": {"profile": "qwen-32b", "future_field": "value"},
                }
            )
        )
        with caplog.at_level(logging.WARNING, logger="maxim.runtime.config_loader"):
            cfg = load_config(path)
        assert cfg.llm.profile == "qwen-32b"
        # No future_field attribute — drop with WARNING
        assert any("future minor" in r.getMessage() for r in caplog.records)

    def test_future_major_rejected(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"_format_version": "2.0"}))
        with pytest.raises(ConfigurationError, match="future major"):
            load_config(path)

    def test_unknown_lane_tier_always_rejected_even_on_future_minor(self, tmp_path):
        """Tier names are FROZEN per the existing invariant — even
        future-minor writers cannot introduce a new tier."""
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.5",
                    "lanes": {"huge": {"remote_url": "http://example.com"}},
                }
            )
        )
        with pytest.raises(ConfigurationError, match="unknown tier name"):
            load_config(path)

    def test_lane_tier_extra_field_lands_in_extra_dict(self, tmp_path):
        """LaneTierConfig is the path (a) escape-hatch type."""
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "lanes": {
                        "large": {
                            "remote_url": "http://example.com",
                            "remote_health_path": "/health",  # future field
                            "remote_timeout_s": 30,  # future field
                        },
                    },
                }
            )
        )
        cfg = load_config(path)
        assert cfg.lanes.large.remote_url == "http://example.com"
        assert cfg.lanes.large.extra == {
            "remote_health_path": "/health",
            "remote_timeout_s": 30,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Schema errors
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaErrors:
    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{invalid json")
        with pytest.raises(ConfigurationError, match="invalid JSON"):
            load_config(path)

    def test_non_object_root_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("[]")
        with pytest.raises(ConfigurationError, match="root must be a JSON object"):
            load_config(path)

    def test_wrong_type_for_bool_field_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "llm": {"enabled": "yes"},  # JSON should have native bool
                }
            )
        )
        with pytest.raises(ConfigurationError, match="expected bool"):
            load_config(path)

    def test_wrong_type_for_int_field_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "llm": {"n_ctx": "8192"},  # string not int
                }
            )
        )
        with pytest.raises(ConfigurationError, match="expected int"):
            load_config(path)

    def test_int_field_below_minimum_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "llm": {"n_ctx": 100},
                }
            )
        )
        with pytest.raises(ConfigurationError, match="below minimum 256"):
            load_config(path)

    def test_invalid_enum_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "role": "boss",
                }
            )
        )
        with pytest.raises(ConfigurationError, match="expected one of"):
            load_config(path)

    def test_format_version_not_string_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"_format_version": 1.0}))
        with pytest.raises(ConfigurationError, match="non-empty string"):
            load_config(path)

    def test_format_version_malformed_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"_format_version": "garbage"}))
        with pytest.raises(ConfigurationError, match="not a valid major.minor"):
            load_config(path)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy role coercion
# ─────────────────────────────────────────────────────────────────────────────


class TestLegacyRoleCoercion:
    def test_legacy_client_in_config_coerces_to_peer(self, tmp_path, caplog):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "role": "client",
                }
            )
        )
        with caplog.at_level(logging.WARNING, logger="maxim.runtime.config_loader"):
            cfg = load_config(path)
        assert cfg.role == "peer"
        assert any("client" in r.getMessage() for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# Full-roundtrip integration
# ─────────────────────────────────────────────────────────────────────────────


class TestFullConfigRoundtrip:
    def test_full_config_loads_cleanly(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "role": "leader",
                    "llm": {
                        "enabled": True,
                        "profile": "qwen2.5-32b-instruct",
                        "n_ctx": 16384,
                        "backend": "llama_cpp",
                        "auto_download": True,
                    },
                    "lanes": {
                        "large": {
                            "remote_url": "https://leader.example.com/v1",
                            "remote_model": "qwen2.5-32b",
                            "remote_api_key_ref": "~/.config/maxim/api_key",
                        },
                    },
                    "cloud": {
                        "enabled": False,
                        "max_lanes": 0,
                        "session_budget_usd": 5.0,
                        "redaction_policy": "standard",
                    },
                    "proxy": {"max_concurrent": 4, "rate_limit_rpm": 0},
                    "auto_spawn": {
                        "llm_server": True,
                        "tunnel": True,
                        "port": 8100,
                        "timeout_s": 120,
                    },
                    "data": {"home": None, "budget_gb": 50.0},
                }
            )
        )
        cfg = load_config(path)
        assert cfg.role == "leader"
        assert cfg.llm.profile == "qwen2.5-32b-instruct"
        assert cfg.llm.n_ctx == 16384
        assert cfg.lanes.large.remote_url == "https://leader.example.com/v1"
        assert cfg.lanes.large.remote_api_key_ref == "~/.config/maxim/api_key"
        assert cfg.cloud.session_budget_usd == 5.0
        assert cfg.proxy.max_concurrent == 4
        assert cfg.auto_spawn.port == 8100
        assert cfg.data.budget_gb == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singleton cache
# ─────────────────────────────────────────────────────────────────────────────


class TestLazySingleton:
    def test_get_config_caches_across_calls(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"_format_version": "1.0", "role": "leader"}))
        a = get_config(path)
        b = get_config(path)
        assert a is b

    def test_get_config_reloads_when_path_changes(self, tmp_path):
        path_a = tmp_path / "a.json"
        path_a.write_text(json.dumps({"_format_version": "1.0", "role": "leader"}))
        path_b = tmp_path / "b.json"
        path_b.write_text(json.dumps({"_format_version": "1.0", "role": "peer"}))
        a = get_config(path_a)
        b = get_config(path_b)
        assert a is not b
        assert a.role == "leader"
        assert b.role == "peer"

    def test_reset_config_cache_clears_singleton(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"_format_version": "1.0"}))
        a = get_config(path)
        reset_config_cache()
        b = get_config(path)
        assert a is not b


# ─────────────────────────────────────────────────────────────────────────────
# Field-mapping completeness
# ─────────────────────────────────────────────────────────────────────────────


class TestFieldMappingCompleteness:
    def test_every_absorbed_env_var_maps_to_a_field_path(self):
        mapped_env_vars = set(_FIELD_TO_ENV.values())
        assert mapped_env_vars == _ABSORBED_ENV_VARS

    def test_every_field_path_resolves_to_a_dataclass_attribute(self):
        cfg = MaximConfig()
        for field_path in _FIELD_TO_ENV:
            # Walk the dot path — should not raise
            obj = cfg
            for part in field_path.split("."):
                obj = getattr(obj, part)

    def test_field_to_env_naming_convention(self):
        for env_name in _FIELD_TO_ENV.values():
            assert env_name.startswith("MAXIM_")
            assert env_name == env_name.upper()
