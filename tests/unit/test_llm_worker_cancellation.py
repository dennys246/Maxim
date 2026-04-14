"""Reproducer + regression guards for the LLMWorker cancellation leak.

**The bug (Plan 3.5 R1):** when ``LLMWorker._call_llm_with_timeout``
abandons a future via ``future.cancel()``, the orphaned background
thread is still executing inside ``LLMRouter._inference_lock``.
``future.cancel()`` only sets a flag on a running future — it does not
stop the thread. The running thread holds ``_inference_lock`` until the
underlying backend call returns (observed in prod: Cloudflare 524 at
~125s), blocking every subsequent LLM call.

**What this file tests:**

1. When a slow backend holds ``_inference_lock`` and the agent-level
   timeout fires, the worker returns a ``{"_timeout": True}`` fallback.
   (This already works.)

2. **After the timeout fires, ``_inference_lock`` is released.** This
   currently FAILS — the orphan thread is still inside the ``with``
   block. Passes after R4 wires cancellation through the backend.

3. **After the timeout fires, ``_provider_states`` is not polluted**
   with a phantom failure from the eventual orphan-thread error. This
   currently FAILS. Passes after R4.

4. **A second call made after a cancellation does not block for the
   full duration of the first (orphaned) call.** This is the real-world
   symptom that the stress test exposed. Currently FAILS.

These tests use a real ``LLMRouter`` with a mocked ``_invoke_backend``
that blocks on a controllable :class:`threading.Event`. That gives us
full control over timing without depending on a real HTTP path.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from maxim.agents.llm_worker import (
    DEFAULT_LLM_CALL_TIMEOUT_S,
    LLMWorker,
    _read_llm_call_timeout_env,
)
from maxim.models.language.config import LLMConfig
from maxim.models.language.router import LLMRouter
from maxim.models.language.types import ProviderState


def _make_router() -> LLMRouter:
    """Real LLMRouter with one fake provider, no real backend init."""
    cfg = dataclasses.replace(
        LLMConfig(),
        enabled=True,
        providers={
            "fake-peer": {
                "type": "maxim_peer",
                "base_url": "http://127.0.0.1:9999/v1",
                "api_key_env": "FAKE_KEY",
                "model": "fake-model",
                "allow_local_endpoints": True,
                "pricing_required": False,
            }
        },
    )
    router = LLMRouter(cfg)
    router._provider_states.setdefault("fake-peer", ProviderState())
    return router


def _make_worker(router: LLMRouter, timeout_s: float) -> LLMWorker:
    """LLMWorker wired to the real router, with a tight agent-level timeout."""
    worker = LLMWorker(llm=router, llm_timeout_s=timeout_s)
    worker.start()
    return worker


@pytest.fixture
def blocking_backend():
    """A mock for ``router._invoke_backend`` that blocks on a released Event.

    Yields ``(release_event, call_started_event, call_count, fake, diag)``:
    - ``release_event``: set by the test to unblock the mock.
    - ``call_started_event``: set by the mock when it begins blocking.
    - ``call_count``: list whose length tracks how many times the mock
      was entered (use ``len(call_count)`` to count).
    - ``fake``: the side_effect callable to patch onto _invoke_backend.
    - ``diag``: a dict the mock populates with diagnostic state (the
      Event reference it observed via current_cancel_event, plus poll
      counts) so failing tests can prove whether ContextVar propagation
      worked or not.
    """
    release_event = threading.Event()
    call_started = threading.Event()
    call_count: list[int] = []
    diag: dict[str, Any] = {
        "observed_event": None,  # the Event the mock saw via current_cancel_event
        "polls": 0,  # how many times we checked is_cancelled
        "saw_cancelled": False,
    }

    def fake_invoke_backend(**kwargs):
        call_count.append(1)
        call_started.set()
        # Capture diagnostic state so tests can assert on what the
        # orphan thread actually saw via the ContextVar.
        from maxim.agents.cancellation import current_cancel_event
        from maxim.models.language.types import BackendDown

        diag["observed_event"] = current_cancel_event()

        while not release_event.is_set():
            diag["polls"] += 1
            ev = current_cancel_event()
            if ev is not None and ev.is_set():
                diag["saw_cancelled"] = True
                raise BackendDown(
                    "fake-peer",
                    fix_hint="cancelled by test",
                )
            time.sleep(0.01)
        return "fake response", {"prompt_tokens": 1, "completion_tokens": 1}

    yield release_event, call_started, call_count, fake_invoke_backend, diag


def test_timeout_returns_fallback(blocking_backend):
    """Baseline: the agent-level timeout fires and returns ``_timeout: True``.

    This already works in the pre-fix code and serves as a sanity check.
    """
    release, started, count, fake, diag = blocking_backend
    router = _make_router()
    worker = _make_worker(router, timeout_s=1.0)

    try:
        with patch.object(router, "_invoke_backend", side_effect=fake):
            result = worker._call_llm_with_timeout(
                prompt="test",
                temperature=0.0,
                max_tokens=10,
            )
        assert result is not None
        assert result.get("_timeout") is True
        assert started.is_set(), "mock backend was never called"
    finally:
        release.set()
        worker.stop()


def test_inference_lock_released_after_timeout(blocking_backend):
    """**Load-bearing regression guard for Plan 3.5.**

    When the agent-level timeout fires, ``_inference_lock`` must be
    released within a reasonable window. Pre-fix: the orphan thread
    holds the lock indefinitely until the backend call returns.
    Post-fix (R4): the orphan thread sees the cancellation event on
    its next check and raises, unwinding the ``with`` block cleanly.
    """
    release, started, count, fake, diag = blocking_backend
    router = _make_router()
    worker = _make_worker(router, timeout_s=1.0)

    try:
        with patch.object(router, "_invoke_backend", side_effect=fake):
            result = worker._call_llm_with_timeout(
                prompt="test",
                temperature=0.0,
                max_tokens=10,
            )
            assert result.get("_timeout") is True
            # Give the fix up to 5s to cascade through cancellation →
            # BackendDown → router unwinds _inference_lock. A correct
            # fix should release the lock within ~100ms of the timeout;
            # 5s is generous tolerance for slow CI hardware.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not router._inference_lock.locked():
                    break
                time.sleep(0.05)
            # Diagnostic snapshot used by the assertion message so a
            # CI failure tells us EXACTLY which step of the propagation
            # broke instead of just "lock still held".
            observed = diag["observed_event"]
            polls = diag["polls"]
            saw = diag["saw_cancelled"]
            assert not router._inference_lock.locked(), (
                "_inference_lock is still held after agent-level timeout. "
                f"diag: orphan_observed_cancel_event_in_context={observed is not None}, "
                f"orphan_polled_is_cancelled_n_times={polls}, "
                f"orphan_saw_cancelled_True={saw}. "
                "If observed=False → contextvars.copy_context().run is not "
                "propagating the binding. If polls>0 but saw=False → cancel_event.set() "
                "in the parent did not flip the orphan's view of the Event "
                "(reference identity issue). See Plan 3.5 R4."
            )
    finally:
        release.set()
        worker.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Plan 3.5 R4 — end-to-end cancellation propagation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def stuck_backend():
    """Backend mock that can ONLY be unblocked via the cancellation event,
    not via an external release signal. Exposes the worst case: an
    orphan thread that has no other exit path.
    """
    call_started = threading.Event()

    def fake_invoke_backend(**kwargs):
        call_started.set()
        from maxim.agents.cancellation import is_cancelled
        from maxim.models.language.types import BackendDown

        deadline = time.monotonic() + 10.0  # safety cap
        while time.monotonic() < deadline:
            if is_cancelled():
                raise BackendDown("fake-peer", fix_hint="cancelled by test")
            time.sleep(0.01)
        raise TimeoutError("stuck_backend hit 10s safety cap — cancellation never fired")

    return call_started, fake_invoke_backend


def test_cancellation_propagates_to_stuck_backend_under_500ms(stuck_backend):
    """**The R4 end-to-end proof.** With cancellation wiring in place,
    a stuck backend must observe the cancel event and unwind
    ``_inference_lock`` within ~500ms of the agent-level timeout firing.
    """
    started, fake = stuck_backend
    router = _make_router()
    worker = _make_worker(router, timeout_s=1.0)

    try:
        with patch.object(router, "_invoke_backend", side_effect=fake):
            t0 = time.monotonic()
            result = worker._call_llm_with_timeout(prompt="test", temperature=0.0, max_tokens=10)
            total_elapsed = time.monotonic() - t0
            assert result.get("_timeout") is True
            # Lock should release within 500ms of the timeout
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                if not router._inference_lock.locked():
                    break
                time.sleep(0.01)
            assert not router._inference_lock.locked(), (
                "Orphan did not observe cancellation Event within 500ms — "
                "check contextvars.copy_context().run wrapping in _call_llm_with_timeout"
            )
            assert total_elapsed < 3.0, (
                f"Total elapsed {total_elapsed:.2f}s — orphan was not cancelled (10s safety cap hit?)"
            )
    finally:
        worker.stop()


def test_second_call_not_blocked_by_cancelled_orphan(stuck_backend):
    """**Regression guard for the stacked-60s-timeout cascade from trace2.**

    After the first call's cancellation, the second call must acquire
    ``_inference_lock`` without waiting for the orphan's 10s safety cap.
    It may hit its own 1s timeout, but must NOT stack behind the first.
    """
    started, fake = stuck_backend
    router = _make_router()
    worker = _make_worker(router, timeout_s=1.0)

    try:
        with patch.object(router, "_invoke_backend", side_effect=fake):
            r1 = worker._call_llm_with_timeout(prompt="first", temperature=0.0, max_tokens=10)
            assert r1.get("_timeout") is True

            # Give the orphan a moment to observe cancellation and unwind
            time.sleep(0.2)
            assert not router._inference_lock.locked()

            # Second call should complete within ~1.5s (its own 1s timeout + overhead)
            t1 = time.monotonic()
            r2 = worker._call_llm_with_timeout(prompt="second", temperature=0.0, max_tokens=10)
            second_elapsed = time.monotonic() - t1
            assert second_elapsed < 2.0, f"second call took {second_elapsed:.2f}s — blocked behind orphan"
            assert r2.get("_timeout") is True
    finally:
        worker.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Plan 3.5 R2 — timeout alignment
# ─────────────────────────────────────────────────────────────────────────────


def test_default_llm_call_timeout_is_300s():
    """The default agent-level LLM call timeout is 300s (Plan 3.5 R2).

    Regression guard: if this is ever changed back to 60s or another
    value below the HTTP layer's _INFERENCE_PROXY_TIMEOUT_S, the
    stacked-timeout cascade from Plan 3.5's evidence section returns.
    """
    assert DEFAULT_LLM_CALL_TIMEOUT_S == 300.0


def test_env_var_unset_returns_default(monkeypatch):
    """Unset env var → fallback to the default."""
    monkeypatch.delenv("MAXIM_LLM_CALL_TIMEOUT_S", raising=False)
    assert _read_llm_call_timeout_env() == DEFAULT_LLM_CALL_TIMEOUT_S
    assert _read_llm_call_timeout_env(fallback=42.0) == 42.0


def test_env_var_parseable_value_wins(monkeypatch):
    """Valid numeric env var overrides the fallback."""
    monkeypatch.setenv("MAXIM_LLM_CALL_TIMEOUT_S", "120")
    assert _read_llm_call_timeout_env() == 120.0
    monkeypatch.setenv("MAXIM_LLM_CALL_TIMEOUT_S", "45.5")
    assert _read_llm_call_timeout_env() == 45.5


def test_env_var_clamped_to_min(monkeypatch):
    """Values below 10.0 are clamped up to 10.0 (can't disable the safety net)."""
    monkeypatch.setenv("MAXIM_LLM_CALL_TIMEOUT_S", "0.5")
    assert _read_llm_call_timeout_env() == 10.0


def test_env_var_clamped_to_max(monkeypatch):
    """Values above 1800.0 are clamped down to 1800.0 (30 minutes absolute max)."""
    monkeypatch.setenv("MAXIM_LLM_CALL_TIMEOUT_S", "3600")
    assert _read_llm_call_timeout_env() == 1800.0


def test_env_var_unparseable_returns_fallback(monkeypatch):
    """Garbage in env var → fallback, not a crash."""
    monkeypatch.setenv("MAXIM_LLM_CALL_TIMEOUT_S", "not-a-number")
    assert _read_llm_call_timeout_env() == DEFAULT_LLM_CALL_TIMEOUT_S


def test_llm_worker_uses_env_override(monkeypatch):
    """Constructing an LLMWorker with no explicit timeout reads the env var."""
    monkeypatch.setenv("MAXIM_LLM_CALL_TIMEOUT_S", "150")
    router = _make_router()
    worker = LLMWorker(llm=router)  # no explicit llm_timeout_s
    try:
        assert worker._llm_timeout == 150.0
    finally:
        worker.stop()


def test_llm_worker_explicit_arg_wins_over_env(monkeypatch):
    """An explicit ``llm_timeout_s`` parameter overrides the env var."""
    monkeypatch.setenv("MAXIM_LLM_CALL_TIMEOUT_S", "150")
    router = _make_router()
    worker = LLMWorker(llm=router, llm_timeout_s=45.0)
    try:
        assert worker._llm_timeout == 45.0
    finally:
        worker.stop()


def test_llm_worker_defaults_to_300s_when_neither_set(monkeypatch):
    """With no env var and no explicit arg, default is 300s."""
    monkeypatch.delenv("MAXIM_LLM_CALL_TIMEOUT_S", raising=False)
    router = _make_router()
    worker = LLMWorker(llm=router)
    try:
        assert worker._llm_timeout == DEFAULT_LLM_CALL_TIMEOUT_S
        assert worker._llm_timeout == 300.0
    finally:
        worker.stop()
