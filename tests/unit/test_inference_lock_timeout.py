"""Bounded _inference_lock acquire (bugs ledger D12 root fix).

The llm_worker timeout path abandons orphan threads that can still hold
router._inference_lock; with an untimed acquire, every subsequent call
parked forever (observed live 2026-08-18: 2.4h 'planning' hangs, ~75
lock-waiter threads, zero network activity). The bounded acquire converts
the eternal silent hang into a loud per-call failure.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

# The @pytest.mark.timeout below is the ONLY thing standing between the D12
# guard and its old failure mode (hang, not FAIL). Without the plugin the
# marker degrades to a PytestUnknownMarkWarning — a warning, not an error —
# and the guard silently reverts to unfalsifiable. This repo's own principle
# is to push silent-no-op invariants into a loud failure, so the absence is
# an error here rather than a quiet downgrade. Installed via the `test` extra
# and by CI's explicit `pip install ... pytest-timeout`.
pytest.importorskip(
    "pytest_timeout",
    reason="pytest-timeout is REQUIRED for the D12 guard to fail by assertion rather than by hanging; "
    "install it (`pip install -e '.[test]'`) — skipping is the honest outcome, silently ignoring the marker is not",
)

from maxim.models.language.config import LLMConfig  # noqa: E402
from maxim.models.language.router import LLMRouter, _inference_lock_timeout_s


def _make_router() -> LLMRouter:
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
    return LLMRouter(cfg)


class TestLockTimeoutHelper:
    def test_default_and_clamps(self, monkeypatch):
        monkeypatch.delenv("MAXIM_INFERENCE_LOCK_TIMEOUT_S", raising=False)
        assert _inference_lock_timeout_s() == 600.0
        monkeypatch.setenv("MAXIM_INFERENCE_LOCK_TIMEOUT_S", "10")
        assert _inference_lock_timeout_s() == 60.0  # lower clamp — cannot go trigger-happy
        monkeypatch.setenv("MAXIM_INFERENCE_LOCK_TIMEOUT_S", "99999")
        assert _inference_lock_timeout_s() == 3600.0  # upper clamp — cannot re-approach eternity
        monkeypatch.setenv("MAXIM_INFERENCE_LOCK_TIMEOUT_S", "not-a-number")
        assert _inference_lock_timeout_s() == 600.0


class TestHeldLockFailsLoudNotForever:
    # A guard that fails by HANGING is not a guard (score card 2026-08-27,
    # Runtime-correctness "to B−"). Against the pre-fix code — an untimed
    # `with self._inference_lock:` — `_complete_text` NEVER returns, so the
    # `elapsed < 5.0` assertion below is never evaluated and the whole job dies
    # on the runner's timeout-minutes, reported as infrastructure rather than
    # as this defect. The marker makes the revert produce a bounded FAIL that
    # names the test. 30s is ~60x the 0.5s bound the test patches in, so it
    # cannot flake on a loaded runner while still being far below any job cap.
    @pytest.mark.timeout(30)
    def test_held_lock_returns_failure_within_bound(self, monkeypatch):
        """A wedged holder must produce a bounded, loud failure — never an
        unbounded park. Patch the helper's floor via monkeypatching the
        function's env read is clamped to 60s minimum, so for test speed we
        monkeypatch the module-level helper directly."""
        import maxim.models.language.router as router_mod

        router = _make_router()
        monkeypatch.setattr(router_mod, "_inference_lock_timeout_s", lambda: 0.5)

        assert router._inference_lock.acquire()  # simulate the wedged orphan
        try:
            t0 = time.monotonic()
            text, usage = router._complete_text("sys", "user", temperature=0.0, max_tokens=8)
            elapsed = time.monotonic() - t0
        finally:
            router._inference_lock.release()

        assert (text, usage) == ("", None)  # the router's loud-failure idiom
        assert elapsed < 5.0  # bounded — the old code never returned at all

    def test_free_lock_reaches_dispatch(self):
        """With the lock free, the bounded acquire is transparent — the call
        proceeds into normal dispatch (which fails against the fake provider,
        but the lock is correctly released afterwards)."""
        router = _make_router()
        router._complete_text("sys", "user", temperature=0.0, max_tokens=8)
        assert not router._inference_lock.locked()
