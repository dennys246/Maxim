"""MoveTool gaze-interface guards (2026-08-03 mirror-turn fix).

The live audio-orient debut turned the head the WRONG way: sensor truthful
(right ear -> angle ~pi), motor stack truthful (+30 deg yaw -> physically
LEFT, hardware-verified), but MoveTool's LLM interface (a) documented no
yaw sign, so the model's compass prior (+ = right) inverted every turn,
and (b) mapped target_x ("left to right") onto the head's x-TRANSLATION
axis — "look right" slid the head millimetres instead of rotating it
(and the robot_id branch dropped it entirely).

Pins: target_x/target_y are GAZE (yaw/pitch) with hardware-verified
signs — +target_x (right, the azimuth convention) -> NEGATIVE yaw
(+yaw = LEFT); +target_y (down) -> POSITIVE pitch; raw angles win over
normalized targets; both execution branches honor the mapping; the
description names the sign conventions so no model has to guess.
"""

from __future__ import annotations

import math

import pytest

from maxim.tools.reachy import MoveTool


class _RecordingMaxim:
    def __init__(self):
        self.calls = []

    def move(self, **kwargs):
        self.calls.append(kwargs)


def _run(**params):
    maxim = _RecordingMaxim()
    result = MoveTool(maxim).execute(**params)
    assert result.success, result.error
    assert len(maxim.calls) == 1
    return maxim.calls[0]


class TestGazeMapping:
    def test_look_right_is_negative_yaw(self):
        # azimuth-sign convention: +1 = full right; +yaw = LEFT on hardware.
        call = _run(target_x=1.0)
        assert call["yaw"] == pytest.approx(-45.0)
        assert call["x"] is None  # gaze must NOT translate the head

    def test_look_left_is_positive_yaw(self):
        call = _run(target_x=-1.0)
        assert call["yaw"] == pytest.approx(45.0)

    def test_azimuth_passthrough_faces_the_sound(self):
        # A sound at azimuth +0.78 (well right): target_x=0.78 must yield a
        # rightward (negative) yaw — the exact live failure this fix closes.
        call = _run(target_x=0.78)
        assert call["yaw"] < 0
        assert call["yaw"] == pytest.approx(-0.78 * 45.0)

    def test_target_y_down_is_positive_pitch(self):
        call = _run(target_y=1.0)
        assert call["pitch"] == pytest.approx(30.0)
        assert call["y"] is None

    def test_target_y_up_is_negative_pitch(self):
        call = _run(target_y=-1.0)
        assert call["pitch"] == pytest.approx(-30.0)

    def test_targets_clamp_to_unit_range(self):
        call = _run(target_x=5.0)
        assert call["yaw"] == pytest.approx(-45.0)

    def test_raw_yaw_wins_over_target_x(self):
        call = _run(target_x=1.0, yaw=10.0)
        assert call["yaw"] == pytest.approx(10.0)

    def test_raw_translation_params_still_pass_through(self):
        call = _run(x=0.5, y=-0.25)
        assert call["x"] == pytest.approx(0.5)
        assert call["y"] == pytest.approx(-0.25)
        assert call["yaw"] is None


class TestRobotIdBranchHonorsGaze:
    def test_target_x_reaches_motion_target_yaw(self, monkeypatch):
        """Pre-fix the robot_id branch silently DROPPED target_x."""
        recorded = {}

        class _FakeRobot:
            def goto_target(self, target):
                recorded["head_yaw"] = target.head_yaw
                return True

        import maxim.tools.reachy as reachy_mod

        monkeypatch.setattr(reachy_mod, "_get_robot_from_registry", lambda rid, m: _FakeRobot())
        result = MoveTool(_RecordingMaxim()).execute(target_x=1.0, robot_id="r1")
        assert result.success
        assert recorded["head_yaw"] == pytest.approx(math.radians(-45.0))


class TestDescriptionNamesTheSigns:
    def test_description_documents_yaw_sign_and_azimuth_tie(self):
        """The root cause was an undocumented sign — the model had to guess.
        The description must name the convention and the azimuth alignment."""
        d = MoveTool.description
        assert "LEFT" in d and "RIGHT" in d
        assert "azimuth" in d
