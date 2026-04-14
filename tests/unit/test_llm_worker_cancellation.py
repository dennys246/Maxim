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
from unittest.mock import patch

import pytest

from maxim.agents.llm_worker import LLMWorker
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

    Yields ``(release_event, call_started_event, call_count)``:
    - ``release_event``: set by the test to unblock the mock.
    - ``call_started_event``: set by the mock when it begins blocking.
    - ``call_count``: list whose length tracks how many times the mock
      was entered (use ``len(call_count)`` to count).
    """
    release_event = threading.Event()
    call_started = threading.Event()
    call_count: list[int] = []

    def fake_invoke_backend(**kwargs):
        call_count.append(1)
        call_started.set()
        # Block until released. In the buggy code path, the agent-level
        # timeout fires while we're blocked here — the test then asserts
        # the router's _inference_lock was released anyway. In the fixed
        # code path, a cancellation event check inside _MaximPeerBackend
        # will raise early and unwind the lock cleanly.
        #
        # Check the cancellation event periodically so the fix can short-circuit.
        from maxim.agents.cancellation import is_cancelled
        from maxim.models.language.types import BackendDown

        while not release_event.is_set():
            if is_cancelled():
                raise BackendDown(
                    "fake-peer",
                    fix_hint="cancelled by test",
                )
            time.sleep(0.01)
        return "fake response", {"prompt_tokens": 1, "completion_tokens": 1}

    yield release_event, call_started, call_count, fake_invoke_backend


def test_timeout_returns_fallback(blocking_backend):
    """Baseline: the agent-level timeout fires and returns ``_timeout: True``.

    This already works in the pre-fix code and serves as a sanity check.
    """
    release, started, count, fake = blocking_backend
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
    release, started, count, fake = blocking_backend
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
            # Give the fix up to 2s to cascade through cancellation →
            # BackendDown → router unwinds _inference_lock. A correct
            # fix should release the lock within ~100ms of the timeout.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if not router._inference_lock.locked():
                    break
                time.sleep(0.05)
            assert not router._inference_lock.locked(), (
                "_inference_lock is still held after agent-level timeout — "
                "orphan thread is blocking future LLM calls. See Plan 3.5 R4."
            )
    finally:
        release.set()
        worker.stop()


# Additional regression guards (second-call-not-blocked, provider-state-clean)
# will be added in R4 alongside the cancellation wiring. They require the
# fix to exist before their assertions can be made tight enough — the
# pre-fix code path can accidentally satisfy loose assertions, leading to
# false-positive passes that mask the bug.
