"""Tests for TickTypeTracker (Stage 5 — Observability).

Covers:
- Basic recording and counting
- Thinking stall detection
- Snapshot for cost breakdown
- Thread safety
"""

from __future__ import annotations

import threading
import time

from maxim.runtime.tick_tracker import TickType, TickTypeTracker


class TestTickTypeTrackerBasic:
    def test_record_and_snapshot(self):
        tracker = TickTypeTracker()
        tracker.record(TickType.THINKING, "deliberating")
        tracker.record(TickType.ACTION, "navigate")
        tracker.record(TickType.SPEECH, "hello")
        tracker.record(TickType.NOOP)

        snap = tracker.snapshot()
        assert snap[TickType.THINKING] == 1
        assert snap[TickType.ACTION] == 1
        assert snap[TickType.SPEECH] == 1
        assert snap[TickType.NOOP] == 1

    def test_recent(self):
        tracker = TickTypeTracker()
        for i in range(5):
            tracker.record(TickType.THINKING, f"tick-{i}")
        recent = tracker.recent(limit=3)
        assert len(recent) == 3
        assert recent[-1].detail == "tick-4"

    def test_multiple_same_type(self):
        tracker = TickTypeTracker()
        for _ in range(10):
            tracker.record(TickType.THINKING)
        snap = tracker.snapshot()
        assert snap[TickType.THINKING] == 10


class TestThinkingStallDetection:
    def test_no_stall_when_recent_action(self):
        tracker = TickTypeTracker(thinking_stall_s=1.0)
        tracker.record(TickType.ACTION)
        tracker.record(TickType.THINKING)
        assert tracker.check_thinking_stall() is False

    def test_stall_after_prolonged_thinking(self):
        tracker = TickTypeTracker(thinking_stall_s=0.1)
        # Record some thinking ticks with last action in the past
        tracker.record(TickType.THINKING)
        tracker._last_action_time = time.time() - 1.0  # 1s ago
        assert tracker.check_thinking_stall() is True

    def test_no_stall_when_idle(self):
        """If recent ticks are NOT thinking, it's idle — not a thinking stall."""
        tracker = TickTypeTracker(thinking_stall_s=0.1)
        tracker.record(TickType.NOOP)
        tracker._last_action_time = time.time() - 1.0
        assert tracker.check_thinking_stall() is False

    def test_stall_warn_rate_limited(self):
        """Warns at most once per stall window."""
        tracker = TickTypeTracker(thinking_stall_s=0.1)
        tracker.record(TickType.THINKING)
        tracker._last_action_time = time.time() - 1.0

        assert tracker.check_thinking_stall() is True
        # Second call within window: returns True but doesn't re-warn
        assert tracker.check_thinking_stall() is True

    def test_action_resets_stall(self):
        tracker = TickTypeTracker(thinking_stall_s=0.1)
        tracker.record(TickType.THINKING)
        tracker._last_action_time = time.time() - 1.0
        assert tracker.check_thinking_stall() is True
        # Now record an action
        tracker.record(TickType.ACTION)
        assert tracker.check_thinking_stall() is False


class TestTickTypeTrackerConcurrency:
    def test_concurrent_recording(self):
        tracker = TickTypeTracker()
        barrier = threading.Barrier(4)

        def worker(tick_type: TickType) -> None:
            barrier.wait(timeout=5)
            for _ in range(50):
                tracker.record(tick_type)

        threads = [
            threading.Thread(target=worker, args=(TickType.THINKING,)),
            threading.Thread(target=worker, args=(TickType.ACTION,)),
            threading.Thread(target=worker, args=(TickType.SPEECH,)),
            threading.Thread(target=worker, args=(TickType.NOOP,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        snap = tracker.snapshot()
        assert snap[TickType.THINKING] == 50
        assert snap[TickType.ACTION] == 50
        assert snap[TickType.SPEECH] == 50
        assert snap[TickType.NOOP] == 50


class TestTickTypeEnum:
    def test_all_types_exist(self):
        expected = {"thinking", "action", "speech", "noop"}
        actual = {t.value for t in TickType}
        assert actual == expected
