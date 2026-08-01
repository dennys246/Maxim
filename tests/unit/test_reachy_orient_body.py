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

Updated 2026-07-31 (live_audio_orient_wiring.md Stages 0b + 0c):

- 0b: `azimuth.pain_scale` recalibrated 1.0 → 0.25 so a fully off-center
  sound cannot out-hurt genuine noxious failure modes the moment DoA feeds
  the sensor ("off to the side" < "motors overheating").
- 0c (Gate A): orient self_effects went DUAL-KEY — `head_yaw` (motor
  semantics) + `azimuth` (the drive-bearing key). head_yaw carries no drive
  spec, so the old head_yaw-only shape intersected drive_specs to {} and
  `drive_potential_diff` was NEVER emitted: orient relief credit was
  structurally dead regardless of feed quality. Azimuth deltas are the
  measured gain (0.57 az/rad post-headfix) times the head_yaw step, SAME
  sign — turning left rotates a stationary sound's head-relative bearing
  toward + under the -1=left/+1=right convention.
"""

from __future__ import annotations

import pytest

from maxim.embodiment.body import Embodiment
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.embodiment.tool_bridge import ModulatorAffordanceTool


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


# The four orient affordances and their expected (head_yaw, azimuth) deltas.
# azimuth = head_yaw * measured gain (0.57 az/rad), same sign (Stage 0c).
_ORIENT_EFFECTS = {
    "turn_left": (0.3, 0.17),
    "turn_right": (-0.3, -0.17),
    "turn_left_big": (0.9, 0.50),
    "turn_right_big": (-0.9, -0.50),
}


def test_orient_affordances_with_dual_key_self_effect():
    orient = _reachy().modulators["orient"]
    # 2x2: {left, right} x {normal, big}. turn_left/turn_right keep their original
    # names AND head_yaw values so a pre-magnitude policy (queen-mind v0.1) still
    # loads — it simply has not met the _big actions yet.
    assert set(orient.affordances) == set(_ORIENT_EFFECTS)
    for name, (head_yaw, azimuth) in _ORIENT_EFFECTS.items():
        eff = orient.affordances[name].self_effect
        assert eff == {"head_yaw": pytest.approx(head_yaw), "azimuth": pytest.approx(azimuth)}, (
            f"{name}.self_effect={eff} — Gate A requires BOTH the motor key "
            "(head_yaw) and the drive-bearing key (azimuth)"
        )


def test_orient_azimuth_sign_matches_head_yaw_sign():
    """A left turn (+head_yaw) must move a left-of-center (negative) azimuth
    toward the 0.0 set-point, i.e. a + delta — SAME sign as head_yaw. An
    opposed sign would credit turning AWAY from the sound."""
    orient = _reachy().modulators["orient"]
    for name in _ORIENT_EFFECTS:
        eff = orient.affordances[name].self_effect
        assert eff["azimuth"] * eff["head_yaw"] > 0, (
            f"{name}: azimuth delta {eff['azimuth']} opposes head_yaw "
            f"{eff['head_yaw']} — sign convention broken (see the DUAL-KEY "
            "comment in reachy_mini.yaml)"
        )


def test_azimuth_pain_scale_below_every_noxious_mode():
    """Stage 0b: worst-case azimuth pain ((1 - comfort_band) * pain_scale)
    must stay BELOW the weakest declared failure mode. At pain_scale 1.0 an
    off-center sound out-shouted thermal_throttling (0.4) and camera_lost
    (0.6) — the hierarchy inversion this pin exists to prevent."""
    ds = _reachy().drive_specs["azimuth"]
    assert ds.pain_scale <= 0.3, f"azimuth.pain_scale={ds.pain_scale} — Stage 0b caps it at 0.3"
    worst_case = (1.0 - ds.comfort_band) * ds.pain_scale
    weakest_noxious = 0.3  # low_battery, the weakest failure_mode pain on this body
    assert worst_case < weakest_noxious, (
        f"worst-case azimuth pain {worst_case} >= weakest failure mode "
        f"{weakest_noxious} — the Stage 0b hierarchy inversion is back"
    )


class TestGateAEndToEnd:
    """The actual Gate A fix: executing a turn on the real reachy body emits
    drive_potential_diff with the correct sign. Pre-0c this was silently
    absent — head_yaw ∩ drive_specs = {} — so the substrate could never
    learn "turn TOWARD the sound," only "turning succeeds"."""

    def _run(self, initial_azimuth: float, affordance: str):
        body = _reachy()
        body.vital_metrics["azimuth"] = initial_azimuth
        emb = Embodiment(body)
        mod = body.modulators["orient"]
        tool = ModulatorAffordanceTool(
            body, mod, affordance, mod.affordances[affordance], affordance, embodiment=emb
        )
        return tool.execute()

    def test_turn_toward_left_sound_emits_positive_relief(self):
        # Sound at -0.5 (left); turn_left → -0.5 + 0.17 = -0.33.
        result = self._run(-0.5, "turn_left")
        assert result.success
        assert result.side_effects is not None
        assert result.side_effects["drive_potential_diff"] == pytest.approx(0.17, abs=1e-6)

    def test_big_turn_centers_far_left_sound(self):
        # Sound at -0.5; turn_left_big → 0.0 exactly, relief +0.5.
        result = self._run(-0.5, "turn_left_big")
        assert result.success
        assert result.side_effects["drive_potential_diff"] == pytest.approx(0.50, abs=1e-6)

    def test_turn_away_emits_negative_credit(self):
        # Sound at -0.5; turn_right → -0.67, further off-center.
        result = self._run(-0.5, "turn_right")
        assert result.success
        assert result.side_effects["drive_potential_diff"] == pytest.approx(-0.17, abs=1e-6)


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
