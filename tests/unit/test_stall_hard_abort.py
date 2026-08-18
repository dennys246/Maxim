"""should_hard_abort — the D12 hard-abort decision (pure function).

A nudge is an injected percept; against a HUNG LLM call it is useless — the
orchestrator thread is blocked inside the call and never reads it (observed
live 2026-08-18: 'planning first probe' at 8,624s and 3,286s with a healthy,
idle server). This function decides when nudging has provably failed and the
sim must terminate loudly. Conservative by design: a false positive kills a
campaign run.
"""

from __future__ import annotations

from maxim.runtime.stall_threshold import should_hard_abort


def _call(**overrides):
    kw = dict(
        stall_duration_s=0.0,
        threshold_s=60.0,
        nudge_count=0,
        byte_silence_s=None,
        byte_silence_threshold_s=120.0,
    )
    kw.update(overrides)
    return should_hard_abort(**kw)


class TestKnownWedgedRoute:
    def test_wedged_call_with_grace_elapsed_aborts(self):
        assert _call(byte_silence_s=130.0, stall_duration_s=200.0) is True

    def test_wedged_call_within_grace_holds(self):
        """Byte-silence over threshold but total stall hasn't outlasted the
        extra grace threshold — give it the benefit of the doubt."""
        assert _call(byte_silence_s=130.0, stall_duration_s=150.0) is False

    def test_healthy_bytes_never_abort_via_this_route(self):
        assert _call(byte_silence_s=30.0, stall_duration_s=10_000.0) is False


class TestPersistentStallRoute:
    def test_three_unconsumed_nudges_and_long_stall_aborts(self):
        assert _call(nudge_count=3, stall_duration_s=200.0) is True

    def test_tonights_hang_would_have_aborted(self):
        """The observed 8,624s first-probe hang (assuming nudges fired)."""
        assert _call(nudge_count=5, stall_duration_s=8624.0) is True

    def test_few_nudges_hold_even_on_long_stall(self):
        assert _call(nudge_count=2, stall_duration_s=10_000.0) is False

    def test_many_nudges_but_short_stall_holds(self):
        """A slow-but-recovering narrator gets its 3x-threshold grace."""
        assert _call(nudge_count=5, stall_duration_s=170.0) is False

    def test_floor_is_threshold_plus_120_for_small_thresholds(self):
        """With a 30s threshold, 3x = 90s would be trigger-happy; the
        +120s floor keeps the minimum abort age at 150s."""
        assert _call(threshold_s=30.0, nudge_count=3, stall_duration_s=140.0) is False
        assert _call(threshold_s=30.0, nudge_count=3, stall_duration_s=151.0) is True


class TestDegenerateInputs:
    def test_zero_duration_never_aborts(self):
        assert _call(nudge_count=99, stall_duration_s=0.0) is False

    def test_zero_threshold_never_aborts(self):
        assert _call(threshold_s=0.0, nudge_count=99, stall_duration_s=9999.0) is False
