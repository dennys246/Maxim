"""Unit tests for ``drive_pain_for_value`` — the single source of truth for the
drive-pain formula, shared by ``Embodiment.evaluate_failures`` (per-tick pain)
and the motor-credit ``potential_diff`` (orient reward).

The motor-credit contract that matters for the cradle/orient work: the
*reduction* in this value from before to after an action is the RELIEF the
action produced. For the centeredness (homeostatic, set_point 0) drive, turning
toward the sound (|azimuth| down) must yield a POSITIVE potential_diff and
turning away a NEGATIVE one — otherwise substrate-primary selection cannot learn
"turn toward the sound." These tests pin that sign behaviour.
"""

from __future__ import annotations

from maxim.embodiment.sem import (
    EntropicDriveSpec,
    HomeostaticDriveSpec,
    drive_comfort_progress,
    drive_pain_for_value,
)


def _centeredness() -> HomeostaticDriveSpec:
    # Mirrors base_humanoid.yaml azimuth centeredness drive.
    return HomeostaticDriveSpec(set_point=0.0, drift_rate=0.0, comfort_band=0.1, pain_scale=0.3)


# ── homeostatic ─────────────────────────────────────────────────────────────


def test_homeostatic_zero_inside_comfort_band():
    ds = _centeredness()
    assert drive_pain_for_value(ds, 0.0) == 0.0
    assert drive_pain_for_value(ds, 0.1) == 0.0  # at the band edge, excess == 0
    assert drive_pain_for_value(ds, -0.05) == 0.0


def test_homeostatic_pain_grows_with_deviation():
    ds = _centeredness()
    p_small = drive_pain_for_value(ds, 0.3)  # excess 0.2 * 0.3 = 0.06
    p_big = drive_pain_for_value(ds, 0.7)  # excess 0.6 * 0.3 = 0.18
    assert p_small == 0.0 or p_small > 0
    assert abs(p_small - 0.06) < 1e-9
    assert abs(p_big - 0.18) < 1e-9
    assert p_big > p_small


def test_homeostatic_is_symmetric_in_sign():
    """Pain depends on |deviation|, so left and right of centre hurt equally —
    the sign that matters for orienting comes from the *potential_diff*, not the
    absolute pain."""
    ds = _centeredness()
    assert drive_pain_for_value(ds, -0.7) == drive_pain_for_value(ds, 0.7)


def test_homeostatic_clamped_to_one():
    ds = HomeostaticDriveSpec(set_point=0.0, drift_rate=0.0, comfort_band=0.0, pain_scale=10.0)
    assert drive_pain_for_value(ds, 1.0) == 1.0  # 1.0 * 10 clamps to 1.0


# ── drive_comfort_progress (the motor-credit signal — value-based) ───────────


def test_comfort_progress_homeostatic_toward_vs_away():
    """The load-bearing motor-credit property: moving toward the set_point is
    positive, away is negative. Sound on the left (azimuth -0.7); turn_left moves
    it toward 0 (progress), turn_right further negative (regress)."""
    ds = _centeredness()
    # value-based: reduction in |value - set_point|, graded (not pain-stepped)
    assert drive_comfort_progress(ds, -0.7, -0.4) > 0  # toward center
    assert abs(drive_comfort_progress(ds, -0.7, -0.4) - 0.3) < 1e-9
    assert drive_comfort_progress(ds, -0.7, -1.0) < 0  # away from center
    # symmetric in sign of the deviation (left and right behave the same)
    assert drive_comfort_progress(ds, 0.7, 0.4) > 0


def test_comfort_progress_entropic_subthreshold_still_positive():
    """The #405 fix: an entropic reduction that does NOT cross the deprivation
    threshold is 0.0 under drive_pain_for_value (step) but POSITIVE under
    drive_comfort_progress (graded) — so feeding/warmth keeps its credit."""
    ds = _hunger()  # drift up, threshold 0.6
    # both 0.9 and 0.7 are ABOVE the threshold -> pain-diff is 0.0
    assert drive_pain_for_value(ds, 0.9) - drive_pain_for_value(ds, 0.7) == 0.0
    # value-progress credits the real reduction
    assert abs(drive_comfort_progress(ds, 0.9, 0.7) - 0.2) < 1e-9
    # increasing hunger (worse) is negative
    assert drive_comfort_progress(ds, 0.5, 0.8) < 0


def test_comfort_progress_entropic_down_direction():
    ds = EntropicDriveSpec(
        drift_direction="down",
        drift_rate=0.01,
        deprivation_threshold=0.3,
        deprivation_pain=0.4,
        satisfaction_threshold=0.8,
    )
    # low is bad -> increasing the value is progress
    assert drive_comfort_progress(ds, 0.2, 0.6) > 0
    assert drive_comfort_progress(ds, 0.6, 0.2) < 0


# ── entropic ────────────────────────────────────────────────────────────────


def _hunger() -> EntropicDriveSpec:
    return EntropicDriveSpec(
        drift_direction="up",
        drift_rate=0.01,
        deprivation_threshold=0.6,
        deprivation_pain=0.5,
        satisfaction_threshold=0.2,
    )


def test_entropic_fires_past_threshold_up():
    ds = _hunger()
    assert drive_pain_for_value(ds, 0.5) == 0.0  # below threshold
    assert drive_pain_for_value(ds, 0.6) == 0.5  # at threshold
    assert drive_pain_for_value(ds, 0.9) == 0.5  # past threshold


def test_entropic_feeding_relief_is_positive_potential_diff():
    """Feeding drops hunger below threshold → pain 0.5 → 0.0 → relief +0.5."""
    ds = _hunger()
    before = drive_pain_for_value(ds, 0.8)  # hungry, past threshold
    after = drive_pain_for_value(ds, 0.3)  # fed, below threshold
    assert before - after > 0


def test_entropic_down_direction():
    ds = EntropicDriveSpec(
        drift_direction="down",
        drift_rate=0.01,
        deprivation_threshold=0.3,
        deprivation_pain=0.4,
        satisfaction_threshold=0.8,
    )
    assert drive_pain_for_value(ds, 0.5) == 0.0
    assert drive_pain_for_value(ds, 0.3) == 0.4
    assert drive_pain_for_value(ds, 0.1) == 0.4
