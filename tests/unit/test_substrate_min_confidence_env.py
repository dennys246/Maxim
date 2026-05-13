"""Unit tests for the ``MAXIM_NAC_MIN_CONFIDENCE`` env-var override.

Added in 0.9.1 (release_0_9_1.md Stage 0a). The resolver in
``agent_loop._resolve_min_confidence`` is the surface that lets Roy-2c
drop ``propose_via_substrate``'s gate to 0.0 without code edits, and
that Wire-A's ablation experiment relies on.
"""

from __future__ import annotations

from maxim.runtime.agent_loop import (
    _DEFAULT_SUBSTRATE_MIN_CONFIDENCE,
    _resolve_min_confidence,
)


class TestResolveMinConfidence:
    def test_default_when_no_env_and_no_explicit(self):
        # Autouse fixture in tests/conftest.py scrubs the env var pre-test.
        assert _resolve_min_confidence(None) == _DEFAULT_SUBSTRATE_MIN_CONFIDENCE

    def test_explicit_caller_arg_wins(self):
        assert _resolve_min_confidence(0.7) == 0.7

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("MAXIM_NAC_MIN_CONFIDENCE", "0.0")
        assert _resolve_min_confidence(None) == 0.0

    def test_env_var_negative_value_passes_through(self, monkeypatch):
        # Negative thresholds are nonsense in production but the resolver
        # is a pure float-parse contract; nonsense semantics are the
        # caller's problem, not the resolver's.
        monkeypatch.setenv("MAXIM_NAC_MIN_CONFIDENCE", "-0.5")
        assert _resolve_min_confidence(None) == -0.5

    def test_invalid_env_falls_back_to_default(self, monkeypatch, caplog):
        monkeypatch.setenv("MAXIM_NAC_MIN_CONFIDENCE", "not-a-float")
        with caplog.at_level("WARNING"):
            assert _resolve_min_confidence(None) == _DEFAULT_SUBSTRATE_MIN_CONFIDENCE
        # Warning should mention the invalid value so operators can debug.
        assert any("not-a-float" in r.message for r in caplog.records)

    def test_empty_env_string_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("MAXIM_NAC_MIN_CONFIDENCE", "")
        assert _resolve_min_confidence(None) == _DEFAULT_SUBSTRATE_MIN_CONFIDENCE

    def test_explicit_zero_is_honored(self):
        # Regression check: explicit 0.0 must NOT be treated as falsy and
        # fall through to the env-var path. The implementation uses
        # ``is None`` for the explicit check; this test pins that.
        assert _resolve_min_confidence(0.0) == 0.0

    def test_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("MAXIM_NAC_MIN_CONFIDENCE", "0.0")
        assert _resolve_min_confidence(0.9) == 0.9
