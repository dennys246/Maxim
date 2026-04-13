"""Tests for the BackendError hierarchy (Plan 2 R2b).

Mirror-tests against the HTTPError shape in ``maxim.utils.http``. Every
subclass must expose ``.status``, ``.response``, ``.fix_hint``, and
``.provider_key``; no subclass may override ``BackendError.__init__`` in a
way that breaks caller-supplied kwargs.
"""

from __future__ import annotations

import pytest

from maxim.models.language.types import (
    INFERENCE_BROKEN_BACKOFF_S,
    BackendAuthFailed,
    BackendDown,
    BackendError,
    BackendInferenceBroken,
    BackendModelMissing,
    BackendOverloaded,
    BackendTimeout,
)


ALL_SUBCLASSES = [
    BackendAuthFailed,
    BackendDown,
    BackendInferenceBroken,
    BackendModelMissing,
    BackendOverloaded,
    BackendTimeout,
]


@pytest.mark.parametrize("cls", ALL_SUBCLASSES)
def test_every_subclass_has_nonempty_fix_hint(cls):
    assert cls.fix_hint, f"{cls.__name__} missing class-level fix_hint"
    assert isinstance(cls.fix_hint, str)


@pytest.mark.parametrize("cls", ALL_SUBCLASSES)
def test_subclass_discoverable_via_subclasses(cls):
    assert cls in BackendError.__subclasses__()


@pytest.mark.parametrize("cls", ALL_SUBCLASSES)
def test_three_access_patterns(cls):
    e = cls("local-llama")
    assert hasattr(e, "status")
    assert hasattr(e, "response")
    assert hasattr(e, "fix_hint")
    assert hasattr(e, "provider_key")
    assert e.provider_key == "local-llama"
    assert e.status is None
    assert e.response is None
    assert e.fix_hint  # non-empty


def test_backend_overloaded_carries_retry_after():
    e = BackendOverloaded("local-llama", retry_after_s=2.5, queue_depth=12)
    assert e.retry_after_s == 2.5
    assert e.queue_depth == 12


def test_backend_timeout_carries_elapsed():
    e = BackendTimeout("local-llama", elapsed_s=61.0)
    assert e.elapsed_s == 61.0


def test_backend_model_missing_carries_requested_model():
    e = BackendModelMissing("local-llama", requested_model="mistral-7b")
    assert e.requested_model == "mistral-7b"


def test_inference_broken_backoff_constant_exported():
    assert INFERENCE_BROKEN_BACKOFF_S == 15.0
    assert isinstance(INFERENCE_BROKEN_BACKOFF_S, float)


def test_exception_str_includes_provider_and_hint():
    e = BackendAuthFailed("local-llama")
    s = str(e)
    assert "BackendAuthFailed" in s
    assert "local-llama" in s
    assert "Cluster key rejected" in s


def test_caller_supplied_fix_hint_overrides_class_default():
    e = BackendDown("peer-01", fix_hint="Custom hint")
    assert e.fix_hint == "Custom hint"


def test_caller_can_attach_status_on_raise():
    # Matches the HTTPError convention — raiser sets status + response.
    e = BackendAuthFailed("openai", status=401)
    assert e.status == 401


# ── _normalize_request_context canonical shim ────────────────────────────


def test_normalize_request_context_handles_none():
    from maxim.agents.llm_worker import _normalize_request_context

    ctx = _normalize_request_context(None)
    assert ctx.request_id  # auto-generated
    assert ctx.agent_id is None


def test_normalize_request_context_reads_legacy_agent_key():
    from maxim.agents.llm_worker import _normalize_request_context

    ctx = _normalize_request_context({"agent": "planner", "request_id": "req-1"})
    assert ctx.agent_id == "planner"
    assert ctx.request_id == "req-1"


def test_normalize_request_context_prefers_agent_id():
    from maxim.agents.llm_worker import _normalize_request_context

    ctx = _normalize_request_context({"agent": "legacy", "agent_id": "new"})
    assert ctx.agent_id == "new"


def test_normalize_request_context_preserves_lane_and_session():
    from maxim.agents.llm_worker import _normalize_request_context

    ctx = _normalize_request_context({"agent_id": "a", "lane": "large", "session_id": "s-1"})
    assert ctx.lane == "large"
    assert ctx.session_id == "s-1"
