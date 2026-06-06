"""Tests for optional dependency error handling.

Verifies that importing modules that require optional deps (cv2, anthropic,
twilio) gives clear, actionable error messages instead of raw ImportErrors.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def test_vision_rtm_engine_without_cv2():
    """RTMEngine._import_cv2() gives clear install instructions."""
    # Temporarily hide cv2
    saved = sys.modules.pop("cv2", None)
    try:
        with patch.dict(sys.modules, {"cv2": None}):
            from maxim.models.vision.rtm_engine import _import_cv2

            with pytest.raises(ImportError, match="pip install pymaxim\\[vision\\]"):
                _import_cv2()
    finally:
        if saved is not None:
            sys.modules["cv2"] = saved


def test_vision_ultralytics_without_cv2():
    """UltralyticsEngine._import_cv2() gives clear install instructions."""
    saved = sys.modules.pop("cv2", None)
    try:
        with patch.dict(sys.modules, {"cv2": None}):
            from maxim.models.vision.ultralytics_engine import _import_cv2

            with pytest.raises(ImportError, match="pip install pymaxim\\[vision\\]"):
                _import_cv2()
    finally:
        if saved is not None:
            sys.modules["cv2"] = saved


def test_validate_model_missing_anthropic_sdk():
    """_validate_model for Claude without anthropic SDK gives install hint."""
    import os

    from maxim.exceptions import ConfigurationError

    # Set the API key so we get past that check
    saved_key = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    saved_mod = sys.modules.pop("anthropic", None)
    try:
        with patch.dict(sys.modules, {"anthropic": None}):
            # Need to reimport to get fresh validation
            from maxim.api import _validate_model

            with pytest.raises(ConfigurationError, match="pip install pymaxim\\[llm-anthropic\\]"):
                _validate_model("claude-sonnet")
    finally:
        if saved_mod is not None:
            sys.modules["anthropic"] = saved_mod
        if saved_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved_key
        else:
            os.environ.pop("ANTHROPIC_API_KEY", None)


def test_validate_model_missing_openai_sdk():
    """_validate_model for GPT-4o without openai SDK gives install hint."""
    import os

    from maxim.exceptions import ConfigurationError

    saved_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-key"

    saved_mod = sys.modules.pop("openai", None)
    try:
        with patch.dict(sys.modules, {"openai": None}):
            from maxim.api import _validate_model

            with pytest.raises(ConfigurationError, match="pip install pymaxim\\[llm-openai\\]"):
                _validate_model("gpt-4o")
    finally:
        if saved_mod is not None:
            sys.modules["openai"] = saved_mod
        if saved_key is not None:
            os.environ["OPENAI_API_KEY"] = saved_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)


# ─────────────────────────────────────────────────────────────────────────────
# Central helper — maxim.utils.optional_deps
# (PR: optional-dependency-loud-failures, 2026-06-05)
# ─────────────────────────────────────────────────────────────────────────────


class TestRequireOptionalDependency:
    def test_returns_module_when_present(self):
        from maxim.utils.optional_deps import require_optional_dependency

        mod = require_optional_dependency("json", feature="x")
        assert mod.__name__ == "json"

    def test_raises_with_extra_and_feature(self):
        from maxim.utils.optional_deps import OptionalDependencyError, require_optional_dependency

        with pytest.raises(OptionalDependencyError) as exc:
            require_optional_dependency("definitely_not_installed_xyz", extra="vision", feature="My feature")
        e = exc.value
        assert e.import_name == "definitely_not_installed_xyz"
        assert e.extra == "vision"
        assert e.feature == "My feature"
        assert "pip install pymaxim[vision]" in str(e)
        assert "My feature" in str(e)
        assert e.fix_hint == "Fix: pip install pymaxim[vision]"

    def test_extra_defaults_from_map(self):
        from maxim.utils.optional_deps import OptionalDependencyError, require_optional_dependency

        # 'anthropic' is in EXTRA_FOR_IMPORT → extra inferred even when omitted.
        saved = sys.modules.pop("anthropic", None)
        try:
            with patch.dict(sys.modules, {"anthropic": None}):
                with pytest.raises(OptionalDependencyError) as exc:
                    require_optional_dependency("anthropic")
            assert exc.value.extra == "llm-anthropic"
            assert "pip install pymaxim[llm-anthropic]" in str(exc.value)
        finally:
            if saved is not None:
                sys.modules["anthropic"] = saved

    def test_unknown_import_falls_back_to_bare_pip(self):
        from maxim.utils.optional_deps import OptionalDependencyError, require_optional_dependency

        with pytest.raises(OptionalDependencyError) as exc:
            require_optional_dependency("some_unmapped_pkg_qqq")
        assert exc.value.fix_hint == "Fix: pip install some_unmapped_pkg_qqq"

    def test_is_importerror_subclass(self):
        from maxim.utils.optional_deps import OptionalDependencyError

        assert issubclass(OptionalDependencyError, ImportError)

    def test_is_not_a_backend_error(self):
        # Load-bearing: the router catches BackendError subclasses to FAIL OVER.
        # OptionalDependencyError must NOT be one, so the router re-raises it
        # (aborts) instead of silently trying the next provider.
        from maxim.models.language.types import BackendError
        from maxim.utils.optional_deps import OptionalDependencyError

        assert not issubclass(OptionalDependencyError, BackendError)


class TestOptionalDependencyAvailable:
    def test_true_for_present(self):
        from maxim.utils.optional_deps import optional_dependency_available

        assert optional_dependency_available("json") is True

    def test_false_for_missing(self):
        from maxim.utils.optional_deps import optional_dependency_available

        assert optional_dependency_available("nope_not_real_zzz") is False


class TestWarnOptionalFallback:
    def test_logs_once_with_hint(self, caplog):
        import logging

        from maxim.utils.optional_deps import warn_optional_fallback

        with caplog.at_level(logging.WARNING):
            warn_optional_fallback("sentence_transformers", fallback="using bag-of-words", feature="Encoder")
            warn_optional_fallback("sentence_transformers", fallback="using bag-of-words", feature="Encoder")
        msgs = [r.getMessage() for r in caplog.records]
        # Deduped: exactly one notice for the same (import, feature).
        hits = [m for m in msgs if "sentence_transformers" in m and "Encoder" in m]
        assert len(hits) == 1
        assert "pip install pymaxim[semantic]" in hits[0]
        assert "using bag-of-words" in hits[0]


# ─────────────────────────────────────────────────────────────────────────────
# LLM backends hard-raise on missing dependency (the 2026-06-05 incident class)
# ─────────────────────────────────────────────────────────────────────────────


class TestBackendsRaiseOnMissingDep:
    def test_anthropic_ensure_client_raises(self):
        from maxim.models.language.anthropic_backend import _AnthropicBackend
        from maxim.utils.optional_deps import OptionalDependencyError

        saved = sys.modules.pop("anthropic", None)
        try:
            with patch.dict(sys.modules, {"anthropic": None}):
                backend = _AnthropicBackend(cfg=_FakeCfg())
                with pytest.raises(OptionalDependencyError):
                    backend._ensure_client()
        finally:
            if saved is not None:
                sys.modules["anthropic"] = saved

    def test_openai_ensure_client_raises(self):
        from maxim.models.language.openai_backend import _OpenAIBackend
        from maxim.utils.optional_deps import OptionalDependencyError

        saved = sys.modules.pop("openai", None)
        try:
            with patch.dict(sys.modules, {"openai": None}):
                backend = _OpenAIBackend(cfg=_FakeCfg())
                with pytest.raises(OptionalDependencyError):
                    backend._ensure_client()
        finally:
            if saved is not None:
                sys.modules["openai"] = saved


# ─────────────────────────────────────────────────────────────────────────────
# CLI synchronous validation (cli_utils) — the reliable main-thread abort
# ─────────────────────────────────────────────────────────────────────────────


class TestCliMissingBackendDependency:
    def test_returns_tuple_when_missing(self):
        from maxim.cli_utils import _missing_backend_dependency

        saved = sys.modules.pop("anthropic", None)
        try:
            with patch.dict(sys.modules, {"anthropic": None}):
                assert _missing_backend_dependency("anthropic") == ("anthropic", "llm-anthropic")
        finally:
            if saved is not None:
                sys.modules["anthropic"] = saved

    def test_none_when_present(self):
        # Env-independent: force the availability probe to report the SDK as
        # importable (CI does not install the llm-anthropic extra, so we can't
        # rely on `anthropic` actually being present here).
        from maxim.cli_utils import _missing_backend_dependency

        with patch("maxim.utils.optional_deps.optional_dependency_available", return_value=True):
            assert _missing_backend_dependency("anthropic") is None

    def test_none_for_unknown_backend(self):
        from maxim.cli_utils import _missing_backend_dependency

        assert _missing_backend_dependency("some_local_backend") is None


# ─────────────────────────────────────────────────────────────────────────────
# Router PROPAGATES OptionalDependencyError instead of masking it as
# "no eligible providers" (the core fix).
# ─────────────────────────────────────────────────────────────────────────────


class TestRouterPropagatesDependencyError:
    def test_dispatch_reraises_not_swallow(self):
        from maxim.utils.optional_deps import OptionalDependencyError

        # A stub backend that raises OptionalDependencyError on invocation,
        # mimicking a requested cloud backend whose SDK is missing.
        class _RaisingBackend:
            requires_prompt_formatting = False
            supports_model_override = False

            def complete_with_usage(self, **kwargs):
                raise OptionalDependencyError("anthropic", extra="llm-anthropic", feature="Anthropic backend")

        from maxim.models.language.router import LLMRouter

        router = _build_minimal_router(LLMRouter)
        # Inject the raising backend for the single provider.
        provider_key = next(iter(router._providers))
        router._backends[provider_key] = _RaisingBackend()

        with pytest.raises(OptionalDependencyError):
            router._complete_text_locked(
                system="sys",
                user="hi",
                temperature=0.0,
                max_tokens=8,
                stream=False,
                request_context=None,
            )


class _FakeCfg:
    """Minimal stand-in for an LLMConfig provider cfg used by backend ctors."""

    backend = "anthropic"
    model = "claude-sonnet-4-6"
    model_base = "claude-sonnet-4-6"
    api_key_env = "ANTHROPIC_API_KEY"
    providers: dict = {}
    n_ctx = 4096

    def __getattr__(self, name):  # tolerate attribute probes
        return None


def _build_minimal_router(LLMRouter):
    """Construct an LLMRouter with a single non-cloud provider for dispatch."""
    from maxim.models.language.config import load_llm_config

    cfg = load_llm_config(profile_override="smollm-1.7b-instruct")
    router = LLMRouter(cfg)
    # Ensure there's at least one provider in the priority list.
    if not router._providers:
        router._providers = {"local": {"type": "llama_cpp", "model": "smollm-1.7b-instruct"}}
    return router
