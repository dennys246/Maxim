"""FocusOnSoundTool guards — the zero-numeric closed-loop orient action.

Designed off the 2026-08-03 mirror-turn post-mortem: no signed scalar
crosses the LLM interface (the failure mode that produced the mirror
robot), and the azimuth used is the live DoA-feed reading at EXECUTION
time, not the model's stale copy. Pins:

- hardware-verified signs: sound right (az > 0) -> negative (rightward)
  yaw; sound left -> positive; relative to the CURRENT head yaw;
- clamping to the +/-45 deg head-yaw envelope;
- fail-soft when no sound has been heard (silent room != error);
- dispatch goes through maxim.move (the real, head-matrix-shipping motor
  path);
- registration/allow-list coherence: autonomy ALWAYS_ALLOWED and every
  mode's CORE_TOOLS include it, so it can never hit the live-path
  confirmation deadlock or lose its prompt description to a mode filter.
"""

from __future__ import annotations

import time

import pytest

from maxim.tools.reachy import FocusOnSoundTool


class _FakeFeed:
    def __init__(self, latest):
        self.latest = latest


class _FakeMaxim:
    def __init__(self, latest=None, yaw=0.0):
        self._doa_feed = _FakeFeed(latest) if latest is not None else None
        self.yaw = yaw
        self.moves = []

    def move(self, **kwargs):
        self.moves.append(kwargs)


def _run(latest, yaw=0.0, **params):
    maxim = _FakeMaxim(latest=latest, yaw=yaw)
    result = FocusOnSoundTool(maxim).execute(**params)
    return result, maxim


class TestClosedLoopSigns:
    def test_sound_on_right_turns_right(self):
        # az +0.5 = 45 deg to the right; +yaw = LEFT, so target is NEGATIVE.
        result, maxim = _run(latest=(0.5, time.monotonic()))
        assert result.success
        assert maxim.moves[0]["yaw"] == pytest.approx(-45.0)
        assert result.output["sound_side"] == "right"

    def test_sound_on_left_turns_left(self):
        result, maxim = _run(latest=(-0.5, time.monotonic()))
        assert result.success
        assert maxim.moves[0]["yaw"] == pytest.approx(45.0)
        assert result.output["sound_side"] == "left"

    def test_turn_is_relative_to_current_yaw(self):
        # Head already at +20 deg (left); sound 45 deg right of the HEAD ->
        # absolute target 20 - 45 = -25.
        result, maxim = _run(latest=(0.5, time.monotonic()), yaw=20.0)
        assert maxim.moves[0]["yaw"] == pytest.approx(-25.0)

    def test_far_sound_clamps_to_head_envelope(self):
        # az +1.0 = 90 deg right -> raw target -90 -> clamped to -45.
        result, maxim = _run(latest=(1.0, time.monotonic()))
        assert maxim.moves[0]["yaw"] == pytest.approx(-45.0)
        assert result.output["clamped_to_head_limit"] is True

    def test_out_of_range_azimuth_is_clamped_first(self):
        result, maxim = _run(latest=(7.0, time.monotonic()))
        assert result.success
        assert result.output["azimuth"] == pytest.approx(1.0)

    def test_reading_age_reported(self):
        result, _ = _run(latest=(0.2, time.monotonic() - 3.0))
        assert result.output["reading_age_s"] == pytest.approx(3.0, abs=0.5)


class TestFailSoft:
    def test_no_feed_is_soft_failure(self):
        maxim = _FakeMaxim(latest=None)
        result = FocusOnSoundTool(maxim).execute()
        assert result.success is False
        assert "No sound" in result.error
        assert maxim.moves == []

    def test_feed_without_reading_is_soft_failure(self):
        maxim = _FakeMaxim()
        maxim._doa_feed = _FakeFeed(None)
        result = FocusOnSoundTool(maxim).execute()
        assert result.success is False
        assert maxim.moves == []

    def test_no_maxim_context(self):
        result = FocusOnSoundTool(None).execute()
        assert result.success is False


class TestWiringCoherence:
    def test_always_allowed_by_autonomy(self):
        from maxim.agents.autonomy import AutonomyController

        assert "focus_on_sound" in AutonomyController.ALWAYS_ALLOWED_TOOLS

    def test_every_mode_makes_it_available(self):
        # CORE_TOOLS membership means no mode filter ever strips its
        # description from the prompt (the 'bare token' failure the deep
        # dive found for move). Checked through get_available_tools — an
        # empty allowed_tools set means all-tools-allowed (mode 'active').
        from maxim.modes.definitions import CORE_TOOLS, get_mode

        assert "focus_on_sound" in CORE_TOOLS
        registry = {"focus_on_sound", "respond", "move"}
        for mode_name in ("passive", "observe", "active", "exploration", "live"):
            available = get_mode(mode_name).get_available_tools(registry)
            assert "focus_on_sound" in available, mode_name

    def test_description_promises_no_parameters(self):
        d = FocusOnSoundTool.description
        assert "No parameters" in d

    def test_duration_passthrough(self):
        result, maxim = _run(latest=(0.5, time.monotonic()), duration=0.5)
        assert maxim.moves[0]["duration"] == pytest.approx(0.5)
