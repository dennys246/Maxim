"""FocusOnSoundTool guards — the zero-numeric closed-loop orient action.

Designed off the 2026-08-03 mirror-turn post-mortem and hardened by a
two-lens pre-merge round. No signed scalar crosses the LLM interface,
the azimuth is the live DoA reading at execution time, and the fold
closed three review blockers:

- dispatch via the CONTROLLER's goto_target (one-shot minjerk; NOT
  maxim.move()'s 2°/call step clamp, which silently under-turned a 45°
  orient by 43° while reporting success);
- the turn is computed in the CAPTURE-time head frame (re-invocation on
  an unchanged reading is idempotent; computing against the current yaw
  marched the head to the limit stop);
- stale readings fail soft (during silence the cache holds the last
  value forever — a faded sound is not a stimulus);
- headless/sim sessions get a NoOp stub, so CORE_TOOLS never advertises
  a bare undispatchable token (the deep-dive 'move' lesson, found again
  by both lenses).
"""

from __future__ import annotations

import math
import time

import pytest

from maxim.tools.reachy import FocusOnSoundTool


class _FakeFeed:
    def __init__(self, latest):
        self.latest = latest


class _FakeRobot:
    def __init__(self, accept=True):
        self.targets = []
        self._accept = accept

    def goto_target(self, target):
        self.targets.append(target)
        return self._accept


class _FakeMaxim:
    def __init__(self, latest=None, yaw=0.0, accept=True, workspace_yaw=None):
        self._doa_feed = _FakeFeed(latest) if latest is not None else None
        self.yaw = yaw
        self._robot = _FakeRobot(accept=accept)
        if workspace_yaw is not None:
            self._get_workspace_limits = lambda: {"yaw": workspace_yaw}


def _now():
    return time.monotonic()


def _run(latest, yaw=0.0, **params):
    maxim = _FakeMaxim(latest=latest, yaw=yaw)
    result = FocusOnSoundTool(maxim).execute(**params)
    return result, maxim


def _yaw_deg(maxim, i=0):
    return math.degrees(maxim._robot.targets[i].head_yaw)


class TestClosedLoopSigns:
    def test_sound_on_right_turns_right(self):
        # az +0.5 = 45° right; +yaw = LEFT, so target is NEGATIVE.
        result, maxim = _run(latest=(0.5, _now(), 0.0))
        assert result.success
        assert _yaw_deg(maxim) == pytest.approx(-45.0)
        assert result.output["sound_side"] == "right"

    def test_sound_on_left_turns_left(self):
        result, maxim = _run(latest=(-0.5, _now(), 0.0))
        assert result.success
        assert _yaw_deg(maxim) == pytest.approx(45.0)
        assert result.output["sound_side"] == "left"

    def test_turn_uses_capture_frame_not_current(self):
        # Reading captured at head yaw +20; head has since moved to -10.
        # Target must be capture-frame: 20 - 45 = -25 (NOT -10 - 45 = -55).
        result, maxim = _run(latest=(0.5, _now(), 20.0), yaw=-10.0)
        assert _yaw_deg(maxim) == pytest.approx(-25.0)

    def test_reinvocation_on_same_reading_is_idempotent(self):
        """The review blocker: computing against CURRENT yaw re-subtracts
        the delta every call and marches the head to the limit stop."""
        latest = (0.2, _now(), 0.0)  # 18° right, captured at yaw 0
        maxim = _FakeMaxim(latest=latest, yaw=0.0)
        tool = FocusOnSoundTool(maxim)
        tool.execute()
        maxim.yaw = -18.0  # head arrived at the target
        tool.execute()  # same reading again
        t1 = math.degrees(maxim._robot.targets[0].head_yaw)
        t2 = math.degrees(maxim._robot.targets[1].head_yaw)
        assert t1 == pytest.approx(t2)  # stable fixed point AT the sound
        assert t1 == pytest.approx(-18.0)

    def test_far_sound_clamps_to_head_envelope(self):
        result, maxim = _run(latest=(1.0, _now(), 0.0))
        assert _yaw_deg(maxim) == pytest.approx(-45.0)
        assert result.output["clamped_to_head_limit"] is True

    def test_learned_workspace_bound_tightens_envelope(self):
        maxim = _FakeMaxim(latest=(1.0, _now(), 0.0), workspace_yaw=40.0)
        result = FocusOnSoundTool(maxim).execute()
        assert result.success
        assert _yaw_deg(maxim) == pytest.approx(-40.0)

    def test_learned_bound_never_widens(self):
        maxim = _FakeMaxim(latest=(1.0, _now(), 0.0), workspace_yaw=80.0)
        FocusOnSoundTool(maxim).execute()
        assert _yaw_deg(maxim) == pytest.approx(-45.0)

    def test_out_of_range_azimuth_is_clamped_first(self):
        result, _ = _run(latest=(7.0, _now(), 0.0))
        assert result.success
        assert result.output["azimuth"] == pytest.approx(1.0)

    def test_legacy_two_tuple_reading_falls_back_to_current_yaw(self):
        result, maxim = _run(latest=(0.5, _now()), yaw=10.0)
        assert result.success
        assert _yaw_deg(maxim) == pytest.approx(10.0 - 45.0)

    def test_duration_passthrough_and_default(self):
        _, maxim = _run(latest=(0.5, _now(), 0.0), duration=0.5)
        assert maxim._robot.targets[0].duration == pytest.approx(0.5)
        _, maxim2 = _run(latest=(0.5, _now(), 0.0))
        assert maxim2._robot.targets[0].duration == pytest.approx(1.0)


class TestFailSoft:
    def test_no_feed_is_soft_failure(self):
        maxim = _FakeMaxim(latest=None)
        result = FocusOnSoundTool(maxim).execute()
        assert result.success is False
        assert "No sound" in result.error
        assert maxim._robot.targets == []

    def test_stale_reading_fails_soft(self):
        """During silence the cache holds the final value forever — a
        faded sound is a memory, not a stimulus."""
        maxim = _FakeMaxim(latest=(0.5, _now() - 60.0, 0.0))
        result = FocusOnSoundTool(maxim).execute()
        assert result.success is False
        assert "faded" in result.error
        assert maxim._robot.targets == []

    def test_fresh_reading_within_window_proceeds(self):
        result, maxim = _run(latest=(0.5, _now() - 5.0, 0.0))
        assert result.success
        assert result.output["reading_age_s"] == pytest.approx(5.0, abs=0.5)

    def test_controller_rejection_is_reported(self):
        maxim = _FakeMaxim(latest=(0.5, _now(), 0.0), accept=False)
        result = FocusOnSoundTool(maxim).execute()
        assert result.success is False

    def test_no_maxim_context(self):
        assert FocusOnSoundTool(None).execute().success is False


class TestWiringCoherence:
    def test_always_allowed_by_autonomy(self):
        from maxim.agents.autonomy import AutonomyController

        assert "focus_on_sound" in AutonomyController.ALWAYS_ALLOWED_TOOLS

    def test_every_mode_makes_it_available(self):
        from maxim.modes.definitions import CORE_TOOLS, get_mode

        assert "focus_on_sound" in CORE_TOOLS
        registry = {"focus_on_sound", "respond", "move"}
        for mode_name in ("passive", "observe", "active", "exploration", "live"):
            available = get_mode(mode_name).get_available_tools(registry)
            assert "focus_on_sound" in available, mode_name

    def test_registered_on_the_live_path(self):
        """Registry-level pin (review finding): CORE_TOOLS/mode tests stay
        green even if the bootstrap registration is deleted — assert the
        registry itself."""
        from maxim.runtime import build_tool_registry

        registry = build_tool_registry(maxim=_FakeMaxim(latest=(0.0, _now(), 0.0)))
        assert "focus_on_sound" in registry.list()

    def test_headless_gets_a_noop_stub_not_a_bare_token(self):
        """Cross-confirmed review blocker: without a stub, CORE_TOOLS
        advertises an undispatchable name in every headless/sim session —
        the deep-dive 'move' bare-token failure, again."""
        from maxim.runtime import build_tool_registry

        registry = build_tool_registry(maxim=None)
        assert "focus_on_sound" in registry.list()
        tool = registry.get("focus_on_sound")
        result = tool.execute()
        assert result.success is False or result.output.get("faced_sound") is False
