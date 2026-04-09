"""Tests for models/language/cloud_dispatch.py — extracted from router.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from maxim.models.language.cloud_dispatch import (
    DEFAULT_PRICING,
    MODEL_DOWNGRADE_MAP,
    JSON_RULES,
    SYSTEM_TOOL_RESPONSE,
    SYSTEM_JSON_ONLY,
    SYSTEM_ROUTE,
    normalize_providers,
    load_routing_policy,
    load_pricing_table,
    load_cost_config,
)


class TestConstants:
    def test_default_pricing_has_known_models(self):
        assert "gpt-4o" in DEFAULT_PRICING
        assert "local" in DEFAULT_PRICING
        assert DEFAULT_PRICING["local"].input_price == 0.0

    def test_downgrade_map_has_entries(self):
        assert "gpt-4o" in MODEL_DOWNGRADE_MAP
        assert MODEL_DOWNGRADE_MAP["gpt-4o"] == "gpt-4o-mini"

    def test_json_rules_contains_critical(self):
        assert "CRITICAL JSON RULES" in JSON_RULES

    def test_system_prompts_embed_json_rules(self):
        assert JSON_RULES in SYSTEM_TOOL_RESPONSE
        assert JSON_RULES in SYSTEM_JSON_ONLY
        assert JSON_RULES in SYSTEM_ROUTE


class TestNormalizeProviders:
    def test_dict_providers(self):
        cfg = MagicMock()
        cfg.providers = {"local": {"type": "llama_cpp"}, "cloud": {"type": "anthropic"}}
        result = normalize_providers(cfg)
        assert "local" in result
        assert "cloud" in result

    def test_empty_providers_synthesizes_local(self):
        cfg = MagicMock()
        cfg.providers = {}
        cfg.backend = "llama_cpp"
        cfg.model = "test"
        cfg.model_base = ""
        cfg.model_path = "/tmp/test"
        cfg.n_ctx = 4096
        result = normalize_providers(cfg)
        assert "local" in result
        assert result["local"]["type"] == "llama_cpp"


class TestLoadRoutingPolicy:
    def test_defaults(self):
        policy = load_routing_policy({})
        assert policy.fallback_on_rate_limit is True
        assert policy.require_cloud_opt_in is True

    def test_overrides(self):
        policy = load_routing_policy({
            "fallback_on_rate_limit": False,
            "max_cost_per_request": 1.50,
        })
        assert policy.fallback_on_rate_limit is False
        assert policy.max_cost_per_request == 1.50


class TestLoadPricingTable:
    def test_includes_defaults(self):
        cfg = MagicMock()
        cfg.pricing = {}
        result = load_pricing_table(cfg)
        assert "gpt-4o" in result
        assert "local" in result

    def test_config_overrides(self):
        cfg = MagicMock()
        cfg.pricing = {
            "custom-model": {"input_price": 1.0, "output_price": 2.0}
        }
        result = load_pricing_table(cfg)
        assert "custom-model" in result
        assert result["custom-model"].input_price == 1.0


class TestLoadCostConfig:
    def test_defaults(self):
        cfg = MagicMock()
        cfg.routing = {}
        result = load_cost_config(cfg)
        assert result.persistence_interval_s == 10.0
        assert result.min_spend_samples == 5


class TestImportPaths:
    def test_constants_from_router(self):
        from maxim.models.language.router import _JSON_RULES, _DEFAULT_PRICING, _MODEL_DOWNGRADE_MAP
        assert "CRITICAL" in _JSON_RULES
        assert "gpt-4o" in _DEFAULT_PRICING
        assert "gpt-4o" in _MODEL_DOWNGRADE_MAP

    def test_system_prompts_from_router(self):
        from maxim.models.language.router import _SYSTEM_TOOL_RESPONSE, _SYSTEM_JSON_ONLY, _SYSTEM_ROUTE
        assert "Maxim" in _SYSTEM_TOOL_RESPONSE
        assert "JSON" in _SYSTEM_JSON_ONLY
        assert "JSON" in _SYSTEM_ROUTE
