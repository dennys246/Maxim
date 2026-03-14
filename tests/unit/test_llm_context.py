"""Unit tests for maxim.agents.llm_context module."""

from __future__ import annotations

import json

import pytest

from maxim.agents.llm_context import (
    _CLOUD_PROVIDER_TYPES,
    _COST_BRIDGE_DEFAULTS,
    _is_cloud_provider_type,
    _load_cost_bridge_config,
)

# ---------------------------------------------------------------------------
# _load_cost_bridge_config
# ---------------------------------------------------------------------------


class TestLoadCostBridgeConfig:
    """Tests for _load_cost_bridge_config."""

    def test_missing_file_returns_defaults(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.json")
        result = _load_cost_bridge_config(missing)
        assert result == _COST_BRIDGE_DEFAULTS
        assert result["cost_energy_scale"] == 100.0

    def test_empty_path_returns_defaults(self):
        assert _load_cost_bridge_config("") == _COST_BRIDGE_DEFAULTS

    def test_none_path_returns_defaults(self):
        # path is falsy
        assert _load_cost_bridge_config(None) == _COST_BRIDGE_DEFAULTS  # type: ignore[arg-type]

    def test_invalid_json_returns_defaults(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all {{{", encoding="utf-8")
        result = _load_cost_bridge_config(str(bad))
        assert result == _COST_BRIDGE_DEFAULTS

    def test_non_dict_raw_returns_defaults(self, tmp_path):
        f = tmp_path / "list.json"
        f.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        result = _load_cost_bridge_config(str(f))
        assert result == _COST_BRIDGE_DEFAULTS

    def test_valid_json_without_cost_bridge_key_returns_defaults(self, tmp_path):
        f = tmp_path / "no_bridge.json"
        f.write_text(json.dumps({"other": 42}), encoding="utf-8")
        result = _load_cost_bridge_config(str(f))
        assert result == _COST_BRIDGE_DEFAULTS

    def test_cost_bridge_not_dict_returns_defaults(self, tmp_path):
        f = tmp_path / "bridge_str.json"
        f.write_text(json.dumps({"cost_bridge": "not a dict"}), encoding="utf-8")
        result = _load_cost_bridge_config(str(f))
        assert result == _COST_BRIDGE_DEFAULTS

    def test_valid_cost_bridge_overrides_scale(self, tmp_path):
        f = tmp_path / "energy.json"
        f.write_text(
            json.dumps({"cost_bridge": {"cost_energy_scale": 250.0}}),
            encoding="utf-8",
        )
        result = _load_cost_bridge_config(str(f))
        assert result["cost_energy_scale"] == 250.0

    def test_cost_bridge_missing_scale_keeps_default(self, tmp_path):
        f = tmp_path / "energy.json"
        f.write_text(
            json.dumps({"cost_bridge": {"other_key": 5}}),
            encoding="utf-8",
        )
        result = _load_cost_bridge_config(str(f))
        assert result["cost_energy_scale"] == 100.0

    def test_cost_bridge_scale_as_int_converted_to_float(self, tmp_path):
        f = tmp_path / "energy.json"
        f.write_text(
            json.dumps({"cost_bridge": {"cost_energy_scale": 50}}),
            encoding="utf-8",
        )
        result = _load_cost_bridge_config(str(f))
        assert result["cost_energy_scale"] == 50.0
        assert isinstance(result["cost_energy_scale"], float)

    def test_cost_bridge_scale_invalid_value_keeps_default(self, tmp_path):
        f = tmp_path / "energy.json"
        f.write_text(
            json.dumps({"cost_bridge": {"cost_energy_scale": "not_a_number"}}),
            encoding="utf-8",
        )
        result = _load_cost_bridge_config(str(f))
        # float("not_a_number") raises ValueError -> except pass -> keeps default
        assert result["cost_energy_scale"] == 100.0

    def test_returns_copy_not_original_defaults(self, tmp_path):
        missing = str(tmp_path / "nope.json")
        result = _load_cost_bridge_config(missing)
        result["cost_energy_scale"] = 999.0
        assert _COST_BRIDGE_DEFAULTS["cost_energy_scale"] == 100.0


# ---------------------------------------------------------------------------
# _is_cloud_provider_type
# ---------------------------------------------------------------------------


class TestIsCloudProviderType:
    """Tests for _is_cloud_provider_type."""

    @pytest.mark.parametrize("provider", sorted(_CLOUD_PROVIDER_TYPES))
    def test_known_providers_return_true(self, provider: str):
        assert _is_cloud_provider_type(provider) is True

    def test_case_insensitivity(self):
        assert _is_cloud_provider_type("Anthropic") is True
        assert _is_cloud_provider_type("OPENAI") is True
        assert _is_cloud_provider_type("Claude") is True

    def test_leading_trailing_whitespace(self):
        assert _is_cloud_provider_type("  anthropic  ") is True

    def test_hyphen_normalization(self):
        assert _is_cloud_provider_type("openai-compatible") is True
        assert _is_cloud_provider_type("openai-compat") is True
        assert _is_cloud_provider_type("OPENAI-Compatible") is True

    def test_unknown_provider_returns_false(self):
        assert _is_cloud_provider_type("google") is False
        assert _is_cloud_provider_type("local") is False
        assert _is_cloud_provider_type("ollama") is False

    def test_empty_string_returns_false(self):
        assert _is_cloud_provider_type("") is False

    def test_none_returns_false(self):
        assert _is_cloud_provider_type(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _load_foundational_context
# ---------------------------------------------------------------------------


class TestLoadFoundationalContext:
    """Tests for _load_foundational_context."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        """Reset the module-level cache before and after each test."""
        import maxim.agents.llm_context as mod

        mod._foundational_context_cache = None
        yield
        mod._foundational_context_cache = None

    def test_returns_string(self):
        from maxim.agents.llm_context import _load_foundational_context

        result = _load_foundational_context()
        assert isinstance(result, str)

    def test_caching_returns_same_object(self):
        from maxim.agents.llm_context import _load_foundational_context

        first = _load_foundational_context()
        second = _load_foundational_context()
        assert first is second

    def test_cache_reset_allows_reload(self):
        import maxim.agents.llm_context as mod
        from maxim.agents.llm_context import _load_foundational_context

        first = _load_foundational_context()
        mod._foundational_context_cache = None
        second = _load_foundational_context()
        # After reset, should still produce equal content
        assert first == second

    def test_returns_nonempty_when_files_exist(self):
        """If CONSTITUTION.md or AGENTS.md exist in parents of __file__,
        the returned string should be non-empty."""
        from pathlib import Path

        import maxim.agents.llm_context as mod
        from maxim.agents.llm_context import _load_foundational_context

        # Check if the repo actually has these files
        src_file = Path(mod.__file__).resolve()
        has_docs = False
        for parent in src_file.parents:
            if (parent / "CONSTITUTION.md").exists() or (parent / "AGENTS.md").exists():
                has_docs = True
                break

        result = _load_foundational_context()
        if has_docs:
            assert len(result) > 0
        else:
            assert result == ""

    def test_cache_set_after_first_call(self):
        import maxim.agents.llm_context as mod
        from maxim.agents.llm_context import _load_foundational_context

        assert mod._foundational_context_cache is None
        _load_foundational_context()
        assert mod._foundational_context_cache is not None
