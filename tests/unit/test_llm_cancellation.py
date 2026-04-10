"""Unit tests for the LLM cancellation primitive.

These tests verify the shared shutdown event used by anthropic/openai
backends to abort in-flight retry loops on user Ctrl+C. The primitive
is a module-level threading.Event so tests must reset it between cases
to avoid state leakage.
"""

from __future__ import annotations

import threading
import time

import pytest

from maxim.models.language.cancellation import (
    is_shutdown_requested,
    request_shutdown,
    reset_shutdown,
    shutdown_wait,
)


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Every test starts with a clean shutdown event."""
    reset_shutdown()
    yield
    reset_shutdown()


class TestShutdownFlag:
    def test_clean_start(self):
        assert is_shutdown_requested() is False

    def test_request_sets_flag(self):
        request_shutdown()
        assert is_shutdown_requested() is True

    def test_reset_clears_flag(self):
        request_shutdown()
        reset_shutdown()
        assert is_shutdown_requested() is False

    def test_idempotent_request(self):
        request_shutdown()
        request_shutdown()
        request_shutdown()
        assert is_shutdown_requested() is True


class TestShutdownWait:
    def test_wait_full_duration_when_not_signalled(self):
        start = time.monotonic()
        result = shutdown_wait(0.1)
        elapsed = time.monotonic() - start
        assert result is False
        assert elapsed >= 0.09  # small scheduling slack

    def test_wait_returns_immediately_when_already_signalled(self):
        request_shutdown()
        start = time.monotonic()
        result = shutdown_wait(5.0)
        elapsed = time.monotonic() - start
        assert result is True
        assert elapsed < 0.1  # should not actually wait 5s

    def test_wait_interrupted_mid_sleep(self):
        """The critical test: a wait in progress wakes up on request_shutdown()."""

        def _signal_after_delay():
            time.sleep(0.05)
            request_shutdown()

        signaller = threading.Thread(target=_signal_after_delay, daemon=True)
        signaller.start()

        start = time.monotonic()
        result = shutdown_wait(5.0)
        elapsed = time.monotonic() - start

        assert result is True
        # Should wake up shortly after the 0.05s signal, well before 5s
        assert elapsed < 1.0, f"wait did not wake on signal (took {elapsed:.2f}s)"

    def test_concurrent_waiters_all_wake(self):
        """Multiple threads waiting on the same event should all wake together."""
        wake_times: list[float] = []
        barrier = threading.Barrier(4)  # 3 waiters + main

        def _waiter():
            barrier.wait()
            shutdown_wait(5.0)
            wake_times.append(time.monotonic())

        threads = [threading.Thread(target=_waiter, daemon=True) for _ in range(3)]
        for t in threads:
            t.start()
        barrier.wait()
        time.sleep(0.02)  # let waiters enter the wait
        signal_time = time.monotonic()
        request_shutdown()

        for t in threads:
            t.join(timeout=1.0)

        assert len(wake_times) == 3
        # All waiters should wake within a short window of the signal
        for wake_time in wake_times:
            assert wake_time - signal_time < 0.5
