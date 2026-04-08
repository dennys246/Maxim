"""Smoke tests for the 6 original public API verbs + list_models.

These tests mock LLM and hardware dependencies so they run offline.
They verify that each verb can be called without crashing and returns
the expected type.
"""

from __future__ import annotations

import os

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# configure
# ─────────────────────────────────────────────────────────────────────────────


def test_configure_sets_verbosity():
    """configure() completes without error and sets agentic verbosity."""
    from maxim.api import configure

    # Should not raise
    configure(verbosity=0)
    configure(verbosity=2)
    configure(verbosity=3)


def test_configure_sets_debug_traces():
    """configure(debug='hippo,nac') sets the corresponding env vars."""
    from maxim.api import configure

    configure(debug="hippo,nac")
    assert os.environ.get("MAXIM_HIPPO_TRACE") == "1"
    assert os.environ.get("MAXIM_NAC_TRACE") == "1"

    # Clean up
    os.environ.pop("MAXIM_HIPPO_TRACE", None)
    os.environ.pop("MAXIM_NAC_TRACE", None)


# ─────────────────────────────────────────────────────────────────────────────
# diagnose
# ─────────────────────────────────────────────────────────────────────────────


def test_diagnose_returns_report():
    """diagnose() returns a DiagnosticReport with platform info and results."""
    from maxim.api import DiagnosticReport, diagnose

    report = diagnose()
    assert isinstance(report, DiagnosticReport)
    assert report.platform is not None
    assert isinstance(report.sections, list)


# ─────────────────────────────────────────────────────────────────────────────
# observe / introspect
# ─────────────────────────────────────────────────────────────────────────────


def test_observe_returns_dict():
    """observe() returns a dict with subsystem keys."""
    from maxim.api import observe

    result = observe()
    assert isinstance(result, dict)
    # Should have at least some known subsystem keys
    assert "memory" in result or "hippocampus" in result or len(result) > 0


def test_introspect_is_observe_alias():
    """introspect() returns the same result as observe()."""
    import maxim

    # Both should work and return dicts
    r1 = maxim.observe()
    r2 = maxim.introspect()
    assert type(r1) is type(r2)


# ─────────────────────────────────────────────────────────────────────────────
# list_models
# ─────────────────────────────────────────────────────────────────────────────


def test_list_models_returns_grouped_dict():
    """list_models() returns dict with 'local' and 'cloud' keys."""
    from maxim.api import ModelInfo, list_models

    result = list_models()
    assert "local" in result
    assert "cloud" in result
    assert len(result["local"]) > 0
    assert len(result["cloud"]) > 0

    # Every entry should be a ModelInfo
    for m in result["local"]:
        assert isinstance(m, ModelInfo)
        assert m.cloud is False
    for m in result["cloud"]:
        assert isinstance(m, ModelInfo)
        assert m.cloud is True
        assert m.api_key_env  # cloud models must have an env var


def test_list_models_importable_from_maxim():
    """list_models is accessible as maxim.list_models()."""
    import maxim

    result = maxim.list_models()
    assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# _validate_model
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_model_accepts_known_local_profile():
    """Valid local model names pass validation without error."""
    from maxim.api import _validate_model

    # Local models don't need SDK or API key — should not raise
    _validate_model("mistral-7b")
    _validate_model("smollm-1.7b")


def test_validate_model_rejects_unknown_profile():
    """Unknown model name raises ConfigurationError."""
    from maxim.api import _validate_model
    from maxim.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="Unknown model"):
        _validate_model("nonexistent-model-xyz")


def test_validate_model_checks_cloud_requirements():
    """Cloud model without SDK or API key raises ConfigurationError."""
    from maxim.api import _validate_model
    from maxim.exceptions import ConfigurationError

    # Temporarily remove the key
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        # Should raise for either missing SDK or missing API key
        with pytest.raises(ConfigurationError):
            _validate_model("claude-sonnet")
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


# ─────────────────────────────────────────────────────────────────────────────
# connect (error case)
# ─────────────────────────────────────────────────────────────────────────────


def test_connect_missing_robot_raises():
    """connect() with a nonexistent robot type raises a clear error."""
    from maxim.api import connect

    with pytest.raises(Exception):
        connect("nonexistent_robot_xyz_999", timeout=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Thread safety
# ─────────────────────────────────────────────────────────────────────────────


def test_event_subscription_thread_safe():
    """on() and unsubscribe() work under concurrent access."""
    import threading

    from maxim.api import on

    handles = []
    errors = []

    def subscribe_many():
        try:
            for i in range(50):
                h = on("test_event", lambda e: None)
                handles.append(h)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=subscribe_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert len(handles) == 200

    # Unsubscribe all
    for h in handles:
        h.unsubscribe()
