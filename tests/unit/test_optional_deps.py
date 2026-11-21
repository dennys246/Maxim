"""Tests for optional dependency error handling.

Verifies that importing modules that require optional deps (cv2, anthropic,
twilio) gives clear, actionable error messages instead of raw ImportErrors.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_vision_rtm_engine_without_cv2():
    """RTMEngine._import_cv2() gives clear install instructions."""
    # Temporarily hide cv2
    saved = sys.modules.pop("cv2", None)
    try:
        with patch.dict(sys.modules, {"cv2": None}):
            from maxim.models.vision.rtm_engine import _import_cv2

            with pytest.raises(ImportError, match="pip install 'pymaxim\\[vision\\]'"):
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

            with pytest.raises(ImportError, match="pip install 'pymaxim\\[vision\\]'"):
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

            with pytest.raises(ConfigurationError, match="pip install 'pymaxim\\[llm-anthropic\\]'"):
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

            with pytest.raises(ConfigurationError, match="pip install 'pymaxim\\[llm-openai\\]'"):
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
        assert "pip install 'pymaxim[vision]'" in str(e)
        assert "My feature" in str(e)
        assert e.fix_hint == "Fix: pip install 'pymaxim[vision]'"

    def test_extra_defaults_from_map(self):
        from maxim.utils.optional_deps import OptionalDependencyError, require_optional_dependency

        # 'anthropic' is in EXTRA_FOR_IMPORT → extra inferred even when omitted.
        saved = sys.modules.pop("anthropic", None)
        try:
            with patch.dict(sys.modules, {"anthropic": None}):
                with pytest.raises(OptionalDependencyError) as exc:
                    require_optional_dependency("anthropic")
            assert exc.value.extra == "llm-anthropic"
            assert "pip install 'pymaxim[llm-anthropic]'" in str(exc.value)
        finally:
            if saved is not None:
                sys.modules["anthropic"] = saved

    def test_unknown_import_falls_back_to_bare_pip(self):
        from maxim.utils.optional_deps import OptionalDependencyError, require_optional_dependency

        with pytest.raises(OptionalDependencyError) as exc:
            require_optional_dependency("some_unmapped_pkg_qqq")
        # Bare package (no extra) needs no quoting — not a glob.
        assert exc.value.fix_hint == "Fix: pip install some_unmapped_pkg_qqq"

    def test_extra_hint_is_zsh_safe_quoted(self):
        """The pymaxim[extra] spec MUST be single-quoted so it is copy-paste
        safe in zsh (the macOS default), where unquoted brackets glob to
        'no matches found'. Regression for the 2026-06-09 install-hint bug."""
        from maxim.utils.optional_deps import OptionalDependencyError, require_optional_dependency

        with pytest.raises(OptionalDependencyError) as exc:
            require_optional_dependency("definitely_not_installed_xyz", extra="vision")
        hint = exc.value.fix_hint
        assert "'pymaxim[vision]'" in hint, hint

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
        assert "pip install 'pymaxim[semantic]'" in hits[0]
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


def test_the_spec_only_probe_never_imports_and_never_raises():
    import sys

    from maxim.utils.optional_deps import optional_dependency_installed

    assert optional_dependency_installed("json") is True
    assert optional_dependency_installed("zz_not_installed_1436") is False
    assert optional_dependency_installed("zz_not_installed_1436.sub") is False
    sys.modules.pop("colorsys", None)
    assert optional_dependency_installed("colorsys") is True
    assert "colorsys" not in sys.modules, "find_spec only: the probe must not import the package"


def _exception_from(module: str, base: type[BaseException] = Exception) -> BaseException:
    cls = type("Err", (base,), {})
    cls.__module__ = module
    return cls("x")


@pytest.mark.parametrize(
    ("exc", "import_name", "refused"),
    [
        (ImportError("x"), "pkg", True),
        (OSError("libportaudio not found"), "sounddevice", True),
        (_exception_from("sounddevice"), "sounddevice", True),  # PortAudioError from import (#1437)
        (_exception_from("pkg._internal"), "pkg", True),  # the package's own submodule
        (_exception_from("pkg"), "pkg.sub", True),  # a dotted import name
        (_exception_from("pkg", OSError), "other", True),
        (RuntimeError("a defect"), "pkg", False),
        (ValueError("a defect"), "pkg", False),
        (_exception_from("otherpkg"), "pkg", False),  # another package's error stays loud
        (_exception_from("pkgextra"), "pkg", False),  # a prefix is not the package
    ],
)
def test_import_refused_classifies_by_cause(exc, import_name, refused):
    from maxim.utils.optional_deps import import_refused

    assert import_refused(exc, import_name) is refused


# ── #1647: a module owned by an extra imports without it; only USING the extra's surface needs it ─────────────────
# The fresh-import sweep (tests/integration/test_fresh_import_sweep.py, nightly) imports every module first in a bare
# interpreter. ``motor_cortex`` raised a plain ImportError at import when keras was absent, so the slow lane (which
# does not install ``training``) was red from 2026-11-01. The keras-derived classes are now built on first access and
# a missing keras surfaces there, as the typed error, with the ``training`` hint.

_MOTOR = "maxim.models.movement.motor_cortex"
_SRC = str(Path(__file__).resolve().parents[2] / "src")


def _run_child(code: str, *, hide_keras: bool) -> subprocess.CompletedProcess[str]:
    prelude = "import sys\nsys.path.insert(0, sys.argv[1])\n"
    if hide_keras:
        prelude += "sys.modules['keras'] = None\nsys.modules['tensorflow'] = None\n"
    return subprocess.run(  # noqa: S603 -- our own interpreter, our own script
        [sys.executable, "-I", "-c", prelude + code, _SRC],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_motor_cortex_imports_without_keras_and_refuses_typed_on_use():
    proc = _run_child(
        "import importlib, json, os\n"
        f"m = importlib.import_module({_MOTOR!r})\n"
        "out = {'tf_env_set': 'TF_GPU_ALLOCATOR' in os.environ, 'resolve': m._resolve_input_shape(None)}\n"
        "from maxim.utils.optional_deps import OptionalDependencyError\n"
        "for name in ('LayerScale', 'MotorCortex', 'motor_cortex', 'load_motor_model'):\n"
        "    try:\n"
        f"        exec(f'from {_MOTOR} import {{name}}')\n"
        "        obj = getattr(m, name)\n"
        "        out[name] = 'ok' if name == 'load_motor_model' else 'no error'\n"
        "    except OptionalDependencyError as e:\n"
        "        out[name] = [e.import_name, e.extra]\n"
        "try:\n"
        "    m.no_such_name\n"
        "except AttributeError:\n"
        "    out['missing'] = 'AttributeError'\n"
        "print(json.dumps(out))\n",
        hide_keras=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["resolve"] == [None, None, 3]
    assert out["tf_env_set"] is False, "the TF env defaults belong to building the model, not to importing the module"
    for name in ("LayerScale", "MotorCortex", "motor_cortex"):
        assert out[name] == ["keras", "training"], (name, out[name])
    assert out["load_motor_model"] == "ok"
    assert out["missing"] == "AttributeError"


def test_motor_cortex_builds_its_keras_classes_once_with_their_own_names():
    """The with-keras half, against a stand-in keras (the real one costs minutes of TensorFlow start-up): the classes
    subclass keras's, are built once, keep the module-level names a checkpoint records, and the alias holds."""
    proc = _run_child(
        "import json, types\n"
        "keras = types.ModuleType('keras')\n"
        "keras.layers = types.SimpleNamespace(Layer=type('Layer', (), {}))\n"
        "keras.Model = type('Model', (), {})\n"
        "sys.modules['keras'] = keras\n"
        f"import {_MOTOR} as m\n"
        f"from {_MOTOR} import LayerScale, MotorCortex, motor_cortex\n"
        "print(json.dumps({\n"
        "  'layer': issubclass(LayerScale, keras.layers.Layer), 'model': issubclass(MotorCortex, keras.Model),\n"
        "  'alias': motor_cortex is MotorCortex, 'cached': m.MotorCortex is MotorCortex,\n"
        "  'names': [LayerScale.__qualname__, MotorCortex.__qualname__, LayerScale.__module__],\n"
        "}))\n",
        hide_keras=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {
        "layer": True,
        "model": True,
        "alias": True,
        "cached": True,
        "names": ["LayerScale", "MotorCortex", _MOTOR],
    }
