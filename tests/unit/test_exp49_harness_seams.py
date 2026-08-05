"""Exp 49 harness seams (docs/experiments/49_two_joint_centering.md).

Pins the src/ seams the two-joint-centering harness rides on:

1. ``SimulatedDoAScenario`` — honest physics: linear-array fold, exact
   inverse of the PRODUCTION ``doa_to_azimuth`` mapping, seeded noise,
   speech density, and the unfolded-theta truth channel.
2. ``SimulatedController.goto_target`` — body ride-along (a body-only
   command moves the head world yaw with the body, matching the real
   controller's composed head matrix) + the neck/body envelope clamps.
3. ``video_enabled=False`` → no mock video stream → capability truth
   (``derive_media_capabilities`` downgrades on the stream surface).
4. ``resolve_robot_entry`` match rule + data-home-aware robots.yaml
   search (``MAXIM_DATA_HOME`` isolation covers robots.yaml).
5. ``MovementMixin._sync_head_position_from_controller`` — the non-SDK
   frame-mirror sync.
6. ``config_flag_disabled`` — the shared robots.yaml opt-out parser
   (``motor_binding: false`` arm-A seam).
7. ``_emit_motor_credit_trace`` — the env-gated H3 credit-visibility
   event.
"""

from __future__ import annotations

import logging
import math
import os

import pytest

from maxim.embodiment.audio_localization import doa_to_azimuth
from maxim.hardware.controller import MotionTarget
from maxim.hardware.simulation.controller import (
    SimulatedController,
)


def _make_controller(**kwargs) -> SimulatedController:
    c = SimulatedController(**kwargs)
    assert c.connect()
    return c


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scenario physics
# ─────────────────────────────────────────────────────────────────────────────


class TestScenarioPhysics:
    def _az(self, controller: SimulatedController) -> float:
        reader = controller.get_doa_reader()
        assert reader is not None
        reading = reader()
        assert reading is not None
        doa, is_speech = reading
        assert is_speech
        # Route through the PRODUCTION mapping — the scenario emits raw
        # DoA radians precisely so this code path runs under test.
        return doa_to_azimuth(doa)

    def test_front_left_source_reads_negative_azimuth(self):
        # +bearing = LEFT (pose-yaw convention); azimuth -1 = left.
        c = _make_controller(doa_source_bearing_deg=45.0, doa_noise_sigma=0.0, doa_seed=1)
        assert self._az(c) == pytest.approx(-0.5, abs=1e-6)

    def test_front_right_source_reads_positive_azimuth(self):
        c = _make_controller(doa_source_bearing_deg=-45.0, doa_noise_sigma=0.0, doa_seed=1)
        assert self._az(c) == pytest.approx(0.5, abs=1e-6)

    def test_rear_source_folds_to_front_mirror_same_side(self):
        # 120° left is BEHIND the interaural axis: folds to 60° left.
        c = _make_controller(doa_source_bearing_deg=120.0, doa_noise_sigma=0.0, doa_seed=1)
        assert self._az(c) == pytest.approx(-60.0 / 90.0, abs=1e-6)

    def test_directly_behind_reads_as_centered_but_theta_is_honest(self):
        # The fold's worst case: facing exactly away reads az ~ 0 — the
        # reason the harness's centered criterion gates on UNFOLDED theta.
        c = _make_controller(doa_source_bearing_deg=180.0, doa_noise_sigma=0.0, doa_seed=1)
        assert abs(self._az(c)) < 1e-6
        scenario = c._doa_scenario
        assert scenario is not None
        pose = c.get_current_pose()
        theta = math.degrees(math.radians(180.0) - float(pose["yaw"]))
        assert abs(abs(theta) - 180.0) < 1e-6  # truth channel disagrees with the fold

    def test_reading_tracks_controller_pose(self):
        c = _make_controller(doa_source_bearing_deg=90.0, doa_noise_sigma=0.0, doa_seed=1)
        assert self._az(c) == pytest.approx(-1.0, abs=1e-6)
        # Turn the body 60° left (head rides along): offset now 30° left.
        c.goto_target(MotionTarget(body_yaw=math.radians(60.0)))
        assert self._az(c) == pytest.approx(-30.0 / 90.0, abs=1e-6)

    def test_noise_is_seed_deterministic(self):
        def seq(seed: int) -> list[float]:
            c = _make_controller(doa_source_bearing_deg=40.0, doa_noise_sigma=0.05, doa_seed=seed)
            r = c.get_doa_reader()
            return [r()[0] for _ in range(10)]

        assert seq(7) == seq(7)
        assert seq(7) != seq(8)

    def test_speech_density_zero_never_speech_one_always(self):
        c0 = _make_controller(doa_source_bearing_deg=40.0, doa_speech_density=0.0, doa_seed=3)
        r0 = c0.get_doa_reader()
        assert all(r0()[1] is False for _ in range(20))
        c1 = _make_controller(doa_source_bearing_deg=40.0, doa_speech_density=1.0, doa_seed=3)
        r1 = c1.get_doa_reader()
        assert all(r1()[1] is True for _ in range(20))

    def test_injected_reader_wins_over_scenario(self):
        marker = object()

        def injected():
            return (1.0, True)

        c = _make_controller(doa_reader=injected, doa_source_bearing_deg=40.0)
        assert c.get_doa_reader() is injected
        del marker

    def test_no_scenario_no_reader(self):
        c = _make_controller()
        assert c.get_doa_reader() is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ride-along + envelopes
# ─────────────────────────────────────────────────────────────────────────────


class TestBodyRideAlong:
    def test_body_only_move_carries_head_world_yaw(self):
        c = _make_controller()
        c.goto_target(MotionTarget(head_yaw=0.1))
        c.goto_target(MotionTarget(body_yaw=0.5))
        pose = c.get_current_pose()
        # Head-relative preserved (0.1); world = 0.1 + 0.5.
        assert pose["yaw"] == pytest.approx(0.6, abs=1e-9)
        assert pose["body_yaw"] == pytest.approx(0.5, abs=1e-9)

    def test_head_yaw_command_is_body_relative_world_stored(self):
        c = _make_controller()
        c.goto_target(MotionTarget(body_yaw=0.5))
        c.goto_target(MotionTarget(head_yaw=0.1))
        assert c.get_current_pose()["yaw"] == pytest.approx(0.6, abs=1e-9)

    def test_pitch_only_move_leaves_yaw_frame_untouched(self):
        c = _make_controller()
        c.goto_target(MotionTarget(head_yaw=0.2, body_yaw=0.3))
        c.goto_target(MotionTarget(head_pitch=0.1))
        pose = c.get_current_pose()
        assert pose["yaw"] == pytest.approx(0.5, abs=1e-9)
        assert pose["body_yaw"] == pytest.approx(0.3, abs=1e-9)
        assert pose["pitch"] == pytest.approx(0.1, abs=1e-9)

    def test_head_envelope_clamps_relative_yaw(self):
        c = _make_controller(head_yaw_limit_deg=22.0)
        c.goto_target(MotionTarget(head_yaw=math.radians(45.0)))
        assert math.degrees(c.get_current_pose()["yaw"]) == pytest.approx(22.0, abs=1e-6)
        # The clamp is on the BODY-RELATIVE yaw: after a body turn the
        # world yaw is body + 22, not 22.
        c.goto_target(MotionTarget(body_yaw=math.radians(40.0), head_yaw=math.radians(45.0)))
        pose = c.get_current_pose()
        assert math.degrees(pose["yaw"]) == pytest.approx(62.0, abs=1e-6)

    def test_body_envelope_clamps(self):
        c = _make_controller(body_yaw_limit_deg=160.0)
        c.goto_target(MotionTarget(body_yaw=math.radians(200.0)))
        assert math.degrees(c.get_current_pose()["body_yaw"]) == pytest.approx(160.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Capability truth for disabled sim streams
# ─────────────────────────────────────────────────────────────────────────────


class TestSimMediaCapabilityTruth:
    def test_video_disabled_downgrades_has_vision(self):
        from maxim.runtime.capabilities import derive_media_capabilities

        c = _make_controller(video_enabled=False)
        assert c.get_video_stream() is None
        has_vision, has_audio = derive_media_capabilities(c)
        assert has_vision is False
        assert has_audio is True

    def test_default_sim_keeps_both(self):
        from maxim.runtime.capabilities import derive_media_capabilities

        c = _make_controller()
        assert derive_media_capabilities(c) == (True, True)

    def test_controller_without_stream_getters_keeps_permissive_true(self):
        from maxim.runtime.capabilities import derive_media_capabilities

        class Bare:
            pass

        assert derive_media_capabilities(Bare()) == (True, True)

    def test_sdk_media_introspection_still_wins(self):
        from maxim.runtime.capabilities import derive_media_capabilities

        class Media:
            camera = None
            audio = object()

        class Mini:
            media = Media()

        class Robot:
            mini = Mini()

            # Stream getters would say True — the SDK handle must win.
            def get_video_stream(self):
                return object()

            def get_audio_stream(self):
                return object()

        assert derive_media_capabilities(Robot()) == (False, True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. robots.yaml resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestRobotsYamlResolution:
    def test_resolve_robot_entry_exact_match_then_primary(self):
        from maxim.hardware.config import RobotsConfig, resolve_robot_entry

        cfg = RobotsConfig.from_dict(
            {
                "robots": {
                    "a": {"type": "simulated"},
                    "b": {"type": "reachy_mini", "primary": True},
                }
            }
        )
        assert resolve_robot_entry(cfg, "a").robot_type == "simulated"
        # No exact match → explicit primary wins.
        assert resolve_robot_entry(cfg, "nope").robot_type == "reachy_mini"

    def test_resolve_robot_entry_ambiguous_returns_none(self):
        from maxim.hardware.config import RobotsConfig, resolve_robot_entry

        cfg = RobotsConfig.from_dict({"robots": {"a": {"type": "simulated"}, "b": {"type": "simulated"}}})
        assert resolve_robot_entry(cfg, "nope") is None

    def test_find_config_file_honors_data_home(self, tmp_path, monkeypatch):
        from maxim.hardware.config import find_config_file
        from maxim.utils.paths import _reset_caches

        home = tmp_path / "isolated_home"
        home.mkdir()
        (home / "robots.yaml").write_text("robots: {}\n")
        monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
        _reset_caches()
        try:
            found = find_config_file()
            assert found == home / "robots.yaml"
        finally:
            _reset_caches()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Non-SDK head sync
# ─────────────────────────────────────────────────────────────────────────────


class TestControllerHeadSync:
    def _host(self, controller):
        from maxim.embodied_runtime.movement import MovementMixin

        class Host(MovementMixin):
            def __init__(self):
                self._robot = controller
                self.mini = None
                self.log = logging.getLogger("test.sync")
                self.yaw = 0.0
                self.pitch = 0.0
                self.roll = 0.0
                self.body_yaw = 0.0

        return Host()

    def test_sync_reads_controller_pose_into_mirrors(self):
        c = _make_controller()
        c.goto_target(MotionTarget(head_yaw=math.radians(10.0), body_yaw=math.radians(30.0)))
        host = self._host(c)
        assert host.sync_head_position() is True
        assert host.body_yaw == pytest.approx(30.0, abs=1e-6)
        # Mirror yaw is BODY-RELATIVE degrees (world − body).
        assert host.yaw == pytest.approx(10.0, abs=1e-6)

    def test_sync_false_when_disconnected(self):
        c = SimulatedController()
        host = self._host(c)
        assert host.sync_head_position() is False


# ─────────────────────────────────────────────────────────────────────────────
# 6. Config opt-out parser (motor_binding arm-A seam)
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigFlagDisabled:
    def test_spellings(self):
        from maxim.embodied_runtime.agentic_runtime import config_flag_disabled

        for raw in (False, "false", "False", " no ", "0", "off", "OFF"):
            assert config_flag_disabled(raw) is True, raw
        for raw in (None, True, "true", "1", "yes", "", "banana"):
            assert config_flag_disabled(raw) is False, raw


# ─────────────────────────────────────────────────────────────────────────────
# 7. Motor-credit trace (H3 visibility)
# ─────────────────────────────────────────────────────────────────────────────


class TestMotorCreditTrace:
    def test_emits_structured_event_when_enabled(self, caplog):
        from maxim.embodiment.tool_bridge import _emit_motor_credit_trace

        os.environ["MAXIM_MOTOR_CREDIT_TRACE"] = "1"
        with caplog.at_level(logging.INFO, logger="maxim.motor_credit"):
            _emit_motor_credit_trace(
                tool_name="reachy_mini_turn_left",
                affordance="turn_left",
                transitions={"azimuth": (-0.5, -0.3)},
                potential_diff=0.2,
            )
        records = [r for r in caplog.records if getattr(r, "event", None) == "motor_credit.measured"]
        assert len(records) == 1
        data = records[0].data
        assert data["tool"] == "reachy_mini_turn_left"
        assert data["transitions"]["azimuth"] == [-0.5, -0.3]
        assert data["potential_diff"] == pytest.approx(0.2)

    def test_silent_when_disabled(self, caplog):
        from maxim.embodiment.tool_bridge import _emit_motor_credit_trace

        os.environ.pop("MAXIM_MOTOR_CREDIT_TRACE", None)
        with caplog.at_level(logging.DEBUG, logger="maxim.motor_credit"):
            _emit_motor_credit_trace(
                tool_name="t",
                affordance="a",
                transitions={"azimuth": (0.1, 0.0)},
                potential_diff=0.1,
            )
        assert not [r for r in caplog.records if getattr(r, "event", None) == "motor_credit.measured"]
