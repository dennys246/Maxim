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
    def __init__(self, accept=True, pose=None, track_target=False):
        self.targets = []
        self._accept = accept
        self._pose = pose  # dict in RADIANS (controller contract) or None
        self._track_target = track_target

    def goto_target(self, target):
        self.targets.append(target)
        if self._track_target and self._accept:
            # Perfect actuation: achieved pose == commanded body-relative yaw.
            self._pose = {"yaw": float(target.head_yaw), "body_yaw": 0.0}
        return self._accept

    def get_current_pose(self):
        # None → empty dict: the tool treats a falsy/keyless pose as "no
        # readback available" (unknown, not failed).
        return dict(self._pose) if self._pose else {}


class _FakeMaxim:
    def __init__(self, latest=None, yaw=0.0, accept=True, workspace_yaw=None, robot=None):
        self._doa_feed = _FakeFeed(latest) if latest is not None else None
        self.yaw = yaw
        self._robot = robot if robot is not None else _FakeRobot(accept=accept)
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

    def test_followup_type_is_process_so_llm_sees_the_result(self):
        """Review fold 2026-08-04 #1: without a TOOL_DESCRIPTIONS entry
        the agent loop truncates the output to ~50 chars and the LLM sees
        only the first key — the honesty payload (faced_sound, note,
        achieved_yaw_deg) never arrives, and the LLM re-issues the same
        clamped call (eight times, 2026-08-03 live)."""
        from maxim.modes.definitions import TOOL_DESCRIPTIONS, get_tool_followup_type

        assert "focus_on_sound" in TOOL_DESCRIPTIONS
        assert get_tool_followup_type("focus_on_sound") == "process"

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


class TestHonestReadback:
    """Verify-actuation honesty (2026-08-04): the 2026-08-03 live session
    showed eight consecutive edge-of-envelope commands where the daemon
    accepted the goto, nothing moved, and the tool reported
    faced_sound=True — so the LLM re-issued the identical call. The tool
    now reads the pose back after the (blocking) goto and reports what
    actually happened."""

    def _run_with_robot(self, latest, robot, yaw=0.0, **params):
        maxim = _FakeMaxim(latest=latest, yaw=yaw, robot=robot)
        return FocusOnSoundTool(maxim).execute(**params), maxim

    def test_reached_unclamped_target_faces_sound(self):
        robot = _FakeRobot(track_target=True)  # perfect actuation
        result, _ = self._run_with_robot(latest=(0.5, _now(), 0.0), robot=robot)
        assert result.success
        assert result.output["faced_sound"] is True
        assert result.output["reached_target"] is True
        assert result.output["achieved_yaw_deg"] == pytest.approx(-45.0, abs=0.1)

    def test_confirmed_shortfall_reports_not_faced(self):
        """Commanded -45, head physically stopped at -12 (neck saturated):
        faced_sound must be False and the note must say the head fell
        short — success=True still (dispatch worked), honesty lives in
        the output the LLM reads."""
        robot = _FakeRobot(pose={"yaw": math.radians(-12.0), "body_yaw": 0.0})
        result, _ = self._run_with_robot(latest=(0.5, _now(), 0.0), robot=robot)
        assert result.success
        assert result.output["faced_sound"] is False
        assert result.output["reached_target"] is False
        assert result.output["achieved_yaw_deg"] == pytest.approx(-12.0, abs=0.1)
        assert "fell short" in (result.output["note"] or "")

    def test_clamped_target_never_claims_faced_even_when_reached(self):
        """THE live bug: az=-1.0 clamps to +45; even if the head reaches
        the clamp target exactly, the sound lies BEYOND it — faced_sound
        False, and the note points at a body turn."""
        robot = _FakeRobot(track_target=True)
        result, _ = self._run_with_robot(latest=(-1.0, _now(), 0.0), robot=robot)
        assert result.success
        assert result.output["clamped_to_head_limit"] is True
        assert result.output["reached_target"] is True  # reached the CLAMP
        assert result.output["faced_sound"] is False  # but not the SOUND
        assert "body" in (result.output["note"] or "")

    def test_achieved_is_body_relative(self):
        """Readback converts the controller's WORLD yaw to body-relative
        (world − body), the same frame as the commanded target."""
        # Commanded -25 (az 0.5 from capture 20); world -5 with body +20
        # → body-relative -25 → reached.
        robot = _FakeRobot(pose={"yaw": math.radians(-5.0), "body_yaw": math.radians(20.0)})
        result, _ = self._run_with_robot(latest=(0.5, _now(), 20.0), robot=robot)
        assert result.output["achieved_yaw_deg"] == pytest.approx(-25.0, abs=0.1)
        assert result.output["reached_target"] is True

    def test_no_readback_stays_optimistic_with_note(self):
        """A controller without a usable pose readback: unknown ≠ failed —
        faced_sound keeps the pre-readback semantics but the note says
        the motion is unverified."""
        robot = _FakeRobot(pose=None)
        result, _ = self._run_with_robot(latest=(0.5, _now(), 0.0), robot=robot)
        assert result.success
        assert result.output["faced_sound"] is True
        assert result.output["reached_target"] is None
        assert result.output["achieved_yaw_deg"] is None
        assert "not verified" in (result.output["note"] or "")


class TestBodyFrameCorrection:
    """sem_motor_binding.md Phase 1: once SEM turns really rotate the body,
    a reading captured BEFORE a body turn must be corrected by
    (capture_body - current_body) or the aim points at the wrong world
    direction by exactly the body rotation."""

    def test_body_rotation_between_capture_and_execute_corrects_frame(self):
        # Captured at head 0 / body +20 deg; body has since turned to 0.
        # az +0.5 -> capture-frame target -45; same world direction from
        # the new body pose = -45 + (20 - 0) = -25.
        robot = _FakeRobot(pose={"yaw": 0.0, "body_yaw": 0.0})
        maxim = _FakeMaxim(latest=(0.5, _now(), 0.0, 20.0), robot=robot)
        result = FocusOnSoundTool(maxim).execute()
        assert result.success
        assert math.degrees(maxim._robot.targets[0].head_yaw) == pytest.approx(-25.0)

    def test_three_tuple_reading_keeps_fixed_body_behavior(self):
        # Pre-Phase-1 stamp shape (no body element): fixed-body assumption,
        # exactly the previous behavior.
        robot = _FakeRobot(pose={"yaw": 0.0, "body_yaw": 0.0})
        maxim = _FakeMaxim(latest=(0.5, _now(), 0.0), robot=robot)
        result = FocusOnSoundTool(maxim).execute()
        assert result.success
        assert math.degrees(maxim._robot.targets[0].head_yaw) == pytest.approx(-45.0)

    def test_clamped_note_names_the_registered_turn_tool(self):
        # A wired body (via the DoA feed's embodiment ref) lets the note
        # name the literal registered tool instead of hallucination bait.
        class _Root:
            name = "reachy_mini"

        class _Emb:
            root = _Root()

        maxim = _FakeMaxim(latest=(-1.0, _now(), 0.0))
        maxim._doa_feed._embodiment = _Emb()
        result = FocusOnSoundTool(maxim).execute()
        assert result.output["clamped_to_head_limit"] is True
        assert "reachy_mini_turn_left_big" in (result.output["note"] or "")

    def test_fell_short_note_names_real_actions(self):
        """2026-08-04 live: the un-named fell-short advice made the LLM
        hallucinate `adjust_yaw`, parking the loop at an approval prompt.
        The note must name real registered actions."""

        class _Root:
            name = "reachy_mini"

        class _Emb:
            root = _Root()

        # Unclamped target, confirmed shortfall via pose readback.
        robot = _FakeRobot(pose={"yaw": math.radians(-6.6), "body_yaw": 0.0})
        maxim = _FakeMaxim(latest=(0.28, _now(), 4.9), robot=robot)
        maxim._doa_feed._embodiment = _Emb()
        result = FocusOnSoundTool(maxim).execute()
        note = result.output["note"] or ""
        assert result.output["reached_target"] is False
        assert "focus_on_sound" in note
        assert "reachy_mini_turn_" in note


class TestLearningTier:
    """D53: the LEARNING tier is separate from mechanical success.

    The tool's payload was always scrupulously honest — `faced_sound`,
    `clamped_to_head_limit`, an explanatory `note`. But it returned
    `success=True` with NO `side_effects`, and
    `tool_dispatch.record_outcome` computed
    `learn_success = success and not embodiment_failed`, so a clamp
    populated no channel at all and the substrate booked
    `Valence.POSITIVE` + `+1.0` cluster credit for a motion the tool
    itself reports did not happen. The honesty fix stopped at the prose
    layer; this carries it to the learner.

    The rule keys on the OUTCOME, never on clamp-occurrence — booking a
    non-positive for every clamp would invert the bug.
    """

    def _run(self, latest, robot, yaw=0.0):
        maxim = _FakeMaxim(latest=latest, yaw=yaw, robot=robot)
        return FocusOnSoundTool(maxim).execute()

    def _tier(self, result):
        return (result.side_effects or {}).get("outcome_valence")

    def test_head_already_at_limit_is_neutral(self):
        """THE D53 case: the head is already at the envelope, the command is
        refused, nothing moves — and the old code booked a full +1.

        The robot starts AT +45 so the pre-motion readback (not the
        ``maxim.yaw`` mirror) reports no movement."""
        robot = _FakeRobot(pose={"yaw": math.radians(45.0), "body_yaw": 0.0}, track_target=True)
        result = self._run(latest=(-1.0, _now(), math.radians(45.0)), robot=robot, yaw=45.0)
        assert result.success is True  # dispatch still succeeded
        assert result.output["clamped_to_head_limit"] is True
        assert self._tier(result) == "neutral"

    def test_no_motion_is_judged_from_the_controller_not_the_stale_mirror(self):
        """``maxim.yaw`` is written only by movement.py::move /
        sync_head_position; this path dispatches via goto_target and never
        syncs, so a second call inside the DN's 3 s window sees a stale
        mirror. Here the mirror says 0 while the head is really at +45 — the
        old target-vs-mirror test would have called that a real motion."""
        robot = _FakeRobot(pose={"yaw": math.radians(45.0), "body_yaw": 0.0}, track_target=True)
        result = self._run(latest=(-1.0, _now(), math.radians(45.0)), robot=robot, yaw=0.0)
        assert self._tier(result) == "neutral"

    def test_clamped_but_moved_stays_effective(self):
        """A clamped turn that STILL moved toward the sound is a real turn.
        Keying on clamp-occurrence rather than outcome would wrongly
        neutralise this one."""
        robot = _FakeRobot(track_target=True)
        result = self._run(latest=(-1.0, _now(), 0.0), robot=robot)
        assert result.output["clamped_to_head_limit"] is True
        assert result.output["reached_target"] is True
        assert self._tier(result) is None

    def test_unverifiable_readback_is_ineffective(self):
        """Unknown != achieved. With no pose readback the tool cannot
        establish that the motion happened, so it asserts nothing."""
        robot = _FakeRobot(pose=None)
        result = self._run(latest=(0.5, _now(), 0.0), robot=robot)
        assert result.output["reached_target"] is None
        assert self._tier(result) == "neutral"

    def test_confirmed_good_turn_is_effective(self):
        robot = _FakeRobot(track_target=True)
        result = self._run(latest=(0.5, _now(), 0.0), robot=robot)
        assert result.output["faced_sound"] is True
        assert self._tier(result) is None

    def test_confirmed_shortfall_books_negative(self):
        """A CONFIRMED shortfall is a real NEGATIVE outcome — and the reason
        the key had to be three-valued.

        The pre-merge round reproduced this booking a full POSITIVE under
        the first (boolean) shape: ``reached is False`` fitted neither
        "ineffective" nor "harm", and the tool returns ``success=True``, so
        nothing carried it. It is NOT harm — it must not be laundered
        through ``embodiment_failures`` — so it needs its own tier.
        ``reached_target`` reaches no learner on its own: it has exactly one
        consumer in ``src/``, the producer itself."""
        robot = _FakeRobot(pose={"yaw": math.radians(-12.0), "body_yaw": 0.0})
        result = self._run(latest=(0.5, _now(), 0.0), robot=robot)
        assert result.success is True
        assert result.output["reached_target"] is False
        assert self._tier(result) == "negative"
        assert (result.side_effects or {}).get("embodiment_failures") is None
