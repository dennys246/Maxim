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
from unittest.mock import MagicMock

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

    def test_controller_context_uses_selected_robot_without_robot_id(self):
        """The stable run(robot=) path uses its controller, not another primary."""
        from maxim.hardware.simulation import SimulatedController

        recorded = {}
        controller = SimulatedController(robot_id="selected")
        controller.goto_target = lambda target: recorded.setdefault("head_yaw", target.head_yaw) is not None
        result = MoveTool(controller).execute(target_x=-1.0)

        assert result.success
        assert recorded["head_yaw"] == pytest.approx(math.radians(45.0))
        assert result.output["robot_id"] == "selected"

    def test_controller_context_rejects_unsupported_translation(self):
        from maxim.hardware.simulation import SimulatedController

        controller = SimulatedController(robot_id="selected")
        controller.goto_target = lambda target: pytest.fail("unsupported translation must not dispatch")
        result = MoveTool(controller).execute(x=0.5)

        assert result.success is False
        assert "translation is unavailable" in result.error

    def test_controller_context_rejects_global_registry_retarget(self):
        """A run-owned tool cannot escape its lease through robot_id."""
        from maxim.hardware.registry import RobotRegistry
        from maxim.hardware.simulation import SimulatedController

        RobotRegistry.reset_instance()
        registry = RobotRegistry()
        registry.register_controller_type("simulated", SimulatedController)
        unrelated = registry.connect_robot(robot_id="unrelated", robot_type="simulated")
        selected = SimulatedController(robot_id="selected")
        selected.goto_target = lambda target: pytest.fail("rejected retarget must not dispatch")
        unrelated.goto_target = lambda target: pytest.fail("unleased robot must not dispatch")
        try:
            result = MoveTool(selected).execute(target_x=1.0, robot_id="unrelated")

            assert result.success is False
            assert "cannot retarget" in result.error
        finally:
            registry.disconnect_all()
            RobotRegistry.reset_instance()

    def test_missing_context_does_not_fall_back_to_global_primary(self, monkeypatch):
        """An absent context must never broaden into global robot actuation."""
        import maxim.tools.reachy as reachy_mod

        lookup = MagicMock()
        monkeypatch.setattr(reachy_mod, "_get_robot_from_registry", lookup)
        result = MoveTool(None).execute(target_x=1.0)

        assert result.success is False
        assert "No Maxim context" in result.error
        lookup.assert_not_called()


def test_controller_context_registers_only_usable_robot_tools(tmp_path, monkeypatch):
    """A bare RobotController must not advertise legacy capture tools in CI."""
    from maxim.hardware.simulation import SimulatedController
    from maxim.runtime.bootstrap import build_tool_registry

    monkeypatch.chdir(tmp_path)
    controller = SimulatedController(robot_id="selected")
    names = set(build_tool_registry(maxim=controller).list())
    move = build_tool_registry(maxim=controller).get("move")

    assert "move" in names
    assert "robot_id" not in move.input_schema
    assert "focus_interests" not in names
    assert "focus_on_sound" not in names
    assert "maxim_command" not in names
    assert "track_target" not in names
    assert "novelty_track" not in names


class TestDescriptionNamesTheSigns:
    def test_description_documents_yaw_sign_and_azimuth_tie(self):
        """The root cause was an undocumented sign — the model had to guess.
        The description must name the convention and the azimuth alignment."""
        d = MoveTool.description
        assert "LEFT" in d and "RIGHT" in d
        assert "azimuth" in d
