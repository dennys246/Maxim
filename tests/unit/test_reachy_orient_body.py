"""Regression guard for the Reachy Mini orient-to-center body wiring (Phase 0a).

Pins the production artifact that the audio (DoA) and visual (substrate gaze)
tracks both consume: the `bodies/reachy_mini` SEM entity must declare the
exteroceptive `azimuth` sensor with a world-coupled centeredness drive and the
two discrete `orient` affordances with `head_yaw` self_effect. See
docs/plans/archive/audiovisual_orienting.md (Phase 0a) and
docs/plans/perception_pipeline_placement.md.
"""

from __future__ import annotations

from maxim.embodiment.component_registry import ComponentRegistry


def _reachy():
    return ComponentRegistry().instantiate("bodies/reachy_mini")


def test_azimuth_sensor_present():
    assert "azimuth" in _reachy().vital_metrics


def test_azimuth_centeredness_drive():
    ds = _reachy().drive_specs["azimuth"]
    assert ds.drift_mode == "homeostatic" if hasattr(ds, "drift_mode") else True
    assert ds.set_point == 0.0
    # drift_rate MUST be 0: a world-set sensor must not auto-return to set_point,
    # or tick_vital_drift fabricates "centered" between re-measurements (the gotcha).
    assert ds.drift_rate == 0.0
    assert ds.comfort_band == 0.1


def test_orient_affordances_with_head_yaw_self_effect():
    orient = _reachy().modulators["orient"]
    assert set(orient.affordances) == {"turn_left", "turn_right"}
    # turn_left drives azimuth toward +, turn_right toward - (self-consistent sim
    # convention; the step-sign<->azimuth mapping is a Phase-1 hardware calibration).
    assert orient.affordances["turn_left"].self_effect == {"head_yaw": 0.3}
    assert orient.affordances["turn_right"].self_effect == {"head_yaw": -0.3}
