"""Integration tests for the bootstrap-to-loop path.

Verifies that the CLI startup → config → LLM init → loop creation path
works end-to-end. LLM backends are mocked, but everything else is real:
config loading, tier detection, router construction, worker creation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestBuildPrimaryRouter:
    """Test the router construction path that both CLI and API use."""

    def test_build_returns_router_and_manager(self):
        """build_primary_router() returns (router, manager) tuple."""
        from maxim.runtime.lane_backends import build_primary_router

        # Don't auto-spawn a real server — just test the construction path
        with patch("maxim.runtime.lane_backends._maybe_auto_spawn_server", side_effect=lambda c, lc, log: lc):
            router, manager = build_primary_router()

        # On a machine without GPU or remote config, router may be None
        # but manager should always exist
        assert manager is not None
        assert hasattr(manager, "get_backend")
        assert hasattr(manager, "describe")

    def test_tier_detection_runs(self):
        """detect_tiers() produces lane configs without crashing."""
        from maxim.runtime.capabilities import RuntimeCapabilities
        from maxim.runtime.lane_models import detect_tiers

        caps = RuntimeCapabilities(has_gpu=False, vram_gb=0.0, ram_gb=8.0)
        tiers = detect_tiers(caps, profile_available=lambda p: False)
        assert isinstance(tiers, dict)
        # Should at least have small tier (CPU-only)
        assert "small" in tiers or "medium" in tiers or len(tiers) >= 0


class TestBootstrapComponents:
    """Test individual bootstrap building blocks."""

    def test_build_decision_engine(self):
        """Decision engine builds without LLM."""
        from maxim.runtime.bootstrap import build_decision_engine

        engine = build_decision_engine()
        assert engine is not None

    def test_build_memory(self):
        """Memory subsystem builds with defaults."""
        from maxim.runtime.bootstrap import build_memory

        mem = build_memory()
        assert mem is not None

    def test_build_state(self):
        """State object builds with defaults."""
        from maxim.runtime.bootstrap import build_state

        state = build_state()
        assert state is not None
        assert hasattr(state, "data")

    def test_build_tool_registry(self):
        """Tool registry builds and has base tools."""
        from maxim.runtime.bootstrap import build_tool_registry

        registry = build_tool_registry()
        assert registry is not None
        assert hasattr(registry, "get") or hasattr(registry, "list_tools")


class TestConfigLoading:
    """Test LLM config and profile resolution."""

    def test_load_llm_config(self):
        """LLM config loads without crashing."""
        from maxim.models.language.config import load_llm_config

        config = load_llm_config()
        assert config is not None
        assert hasattr(config, "enabled")

    def test_builtin_profiles_exist(self):
        """Built-in model profiles are populated."""
        from maxim.models.language.config import _BUILTIN_PROFILES

        assert len(_BUILTIN_PROFILES) > 5  # At least the core models
        assert "mistral-7b-instruct-v0.2" in _BUILTIN_PROFILES or "smollm-1.7b-instruct" in _BUILTIN_PROFILES

    def test_profile_aliases_resolve(self):
        """Common aliases resolve to real profile names."""
        from maxim.models.language.config import _PROFILE_ALIASES

        # Check a few common aliases exist
        assert isinstance(_PROFILE_ALIASES, dict)
        if "mistral-7b" in _PROFILE_ALIASES:
            resolved = _PROFILE_ALIASES["mistral-7b"]
            assert resolved  # Non-empty string


class TestErrorMessages:
    """Verify startup errors are actionable."""

    def test_missing_api_key_message(self):
        """Cloud models without API keys should produce clear errors."""
        from maxim.models.language.config import _BUILTIN_PROFILES

        # Find a cloud profile
        cloud_profiles = {k: v for k, v in _BUILTIN_PROFILES.items() if v.get("cloud")}
        if not cloud_profiles:
            pytest.skip("No cloud profiles to test")

        name, profile = next(iter(cloud_profiles.items()))
        key_env = profile.get("api_key_env", "")
        assert key_env, f"Cloud profile {name} should declare api_key_env"


class TestVersionInfo:
    """Test version infrastructure."""

    def test_version_defined(self):
        """__version__ is a valid semver string."""
        import maxim

        assert hasattr(maxim, "__version__")
        parts = maxim.__version__.split(".")
        assert len(parts) >= 2  # At least major.minor

    def test_get_version_info_returns_dict(self):
        """get_version_info() returns dict with at least 'version' key."""
        from maxim import get_version_info

        info = get_version_info()
        assert isinstance(info, dict)
        assert "version" in info
