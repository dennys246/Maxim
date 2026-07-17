"""Regression guard for the Reachy Mini orient-to-center body wiring.

Pins the production artifact that the audio (DoA) and visual (substrate gaze)
tracks both consume: the `bodies/reachy_mini` SEM entity must declare the
exteroceptive `azimuth` sensor with a world-coupled centeredness drive and the
discrete `orient` affordances with `head_yaw` self_effect.

Updated 2026-07-16 (Exp 45b/45c): the orient set grew from 2 actions (direction
only) to a 2x2 MAGNITUDE set. The magnitudes are load-bearing, not cosmetic —
`potential_diff` credit has no cost for large moves, so magnitude is only
learnable because the big step OVERSHOOTS near centre. Changing these values
moves the derived decision boundary (`gain*(|d_big|+|d_normal|)/2`) and re-opens
[Exp 45c](../../docs/experiments/45c_flip_bins.md), which reached magnitude 1.00
with them. See docs/plans/substrate_native_orienting.md +
docs/embodiment/porting_orient_loop.md ("The design constants are DERIVED").
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
    # 2x2: {left, right} x {normal, big}. turn_left/turn_right keep their original
    # names AND values so a pre-magnitude policy (queen-mind v0.1) still loads —
    # it simply has not met the _big actions yet.
    assert set(orient.affordances) == {"turn_left", "turn_right", "turn_left_big", "turn_right_big"}
    # turn_* drives azimuth toward +, turn_right* toward - (self-consistent sim
    # convention; the step-sign<->azimuth mapping is a Phase-1 hardware calibration).
    assert orient.affordances["turn_left"].self_effect == {"head_yaw": 0.3}
    assert orient.affordances["turn_right"].self_effect == {"head_yaw": -0.3}
    assert orient.affordances["turn_left_big"].self_effect == {"head_yaw": 0.9}
    assert orient.affordances["turn_right_big"].self_effect == {"head_yaw": -0.9}


def test_orient_magnitudes_admit_a_learnable_decision_boundary():
    """The 3:1 magnitude ratio is what makes MAGNITUDE learnable — pin the property.

    potential_diff has no cost for large moves, so the big step wins everywhere
    UNLESS it overshoots near centre. That requires two distinct magnitudes whose
    midpoint-derived boundary (gain*(d_small+d_big)/2) lands INSIDE the sensor's
    usable range — otherwise one action dominates and there is nothing to learn.
    At the measured Reachy gain (~0.55 az/rad, Exp 45c) the boundary is ~0.33,
    comfortably inside the reliable |az| <= ~0.85.
    """
    orient = _reachy().modulators["orient"]
    mags = sorted({abs(a.self_effect["head_yaw"]) for a in orient.affordances.values()})
    assert len(mags) == 2, "magnitude learning needs exactly two distinct step sizes here"
    small, big = mags
    assert big / small >= 2.0, "magnitudes too close: big would never overshoot -> nothing to learn"
    measured_gain = 0.55  # Exp 45c, post-headfix; four measurements within 0.03
    boundary = measured_gain * (small + big) / 2.0
    assert 0.15 < boundary < 0.85, f"derived boundary {boundary:.2f} outside the usable azimuth range"
