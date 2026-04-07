"""Unit tests for MovementMixin pure-math methods."""

from __future__ import annotations

import logging
import math

import pytest

from maxim.conscience.movement import MovementMixin


# ---------------------------------------------------------------------------
# Testable stub
# ---------------------------------------------------------------------------


class _TestMixin(MovementMixin):
    """Minimal concrete wrapper so we can instantiate the mixin."""

    def __init__(self):
        self._default_network = None
        self.log = logging.getLogger("test_movement")
        self._enqueue_motor = lambda fn, *a, **kw: None  # no-op


@pytest.fixture
def mixin() -> _TestMixin:
    return _TestMixin()


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify class-level constants exist and have expected values."""

    def test_pixel_bounds(self):
        assert MovementMixin._LOOK_AT_PIXEL_BOUNDS == {
            "u_min": 64,
            "u_max": 576,
            "v_min": 48,
            "v_max": 432,
        }

    def test_translation_limits(self):
        assert MovementMixin._SAFE_X_LIMIT == 15.0
        assert MovementMixin._SAFE_Y_LIMIT == 18.0
        assert MovementMixin._SAFE_Z_LIMIT == 15.0

    def test_rotation_limits(self):
        assert MovementMixin._SAFE_ROLL_LIMIT == 35.0
        assert MovementMixin._SAFE_PITCH_LIMIT == 35.0
        assert MovementMixin._SAFE_YAW_LIMIT == 55.0

    def test_camera_parameters(self):
        assert MovementMixin._IMAGE_CENTER_U == 320.0
        assert MovementMixin._IMAGE_CENTER_V == 240.0
        assert MovementMixin._IMAGE_WIDTH == 640.0
        assert MovementMixin._IMAGE_HEIGHT == 480.0

    def test_movement_per_pixel(self):
        assert MovementMixin._MM_PER_PIXEL_H == 0.05
        assert MovementMixin._MM_PER_PIXEL_V == 0.04
        assert MovementMixin._DEG_PER_PIXEL_H == 0.10
        assert MovementMixin._DEG_PER_PIXEL_V == 0.08


# ═══════════════════════════════════════════════════════════════════════════
# _get_workspace_limits
# ═══════════════════════════════════════════════════════════════════════════


class TestGetWorkspaceLimits:
    def test_hardcoded_fallback(self, mixin: _TestMixin):
        """With _default_network=None, returns hardcoded limits."""
        limits = mixin._get_workspace_limits()
        assert limits == {
            "x": 15.0,
            "y": 18.0,
            "z": 15.0,
            "roll": 35.0,
            "pitch": 35.0,
            "yaw": 55.0,
        }

    def test_no_bounds_learner_on_network(self, mixin: _TestMixin):
        """Network exists but has no _bounds_learner attribute."""

        class _FakeNetwork:
            pass

        mixin._default_network = _FakeNetwork()
        limits = mixin._get_workspace_limits()
        assert limits["yaw"] == 55.0  # still hardcoded

    def test_learned_bounds(self, mixin: _TestMixin):
        """When bounds_learner is present, its values are used."""

        class _FakeBoundsLearner:
            def get_bound(self, axis: str) -> float:
                return {"x": 10.0, "y": 12.0, "z": 8.0, "roll": 20.0, "pitch": 25.0, "yaw": 40.0}[axis]

        class _FakeNetwork:
            _bounds_learner = _FakeBoundsLearner()

        mixin._default_network = _FakeNetwork()
        limits = mixin._get_workspace_limits()
        assert limits == {
            "x": 10.0,
            "y": 12.0,
            "z": 8.0,
            "roll": 20.0,
            "pitch": 25.0,
            "yaw": 40.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# _clamp_to_workspace_6d
# ═══════════════════════════════════════════════════════════════════════════


class TestClampToWorkspace6D:
    def test_within_limits_unchanged(self, mixin: _TestMixin):
        """Values well within limits should pass through untouched."""
        result = mixin._clamp_to_workspace_6d(1.0, 2.0, 3.0, 5.0, 5.0, 10.0)
        assert result == pytest.approx((1.0, 2.0, 3.0, 5.0, 5.0, 10.0))

    def test_zero_input(self, mixin: _TestMixin):
        result = mixin._clamp_to_workspace_6d(0, 0, 0, 0, 0, 0)
        assert result == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_at_rectangular_limits(self, mixin: _TestMixin):
        """Values exactly at limits should be unchanged (no ellipsoid issue
        because single-axis => norm=1 which is not >1)."""
        result = mixin._clamp_to_workspace_6d(15.0, 0, 0, 0, 0, 0)
        assert result[0] == pytest.approx(15.0)

    def test_beyond_rectangular_limits_clamped(self, mixin: _TestMixin):
        """Translation beyond a single axis limit gets clamped."""
        result = mixin._clamp_to_workspace_6d(100.0, 0, 0, 0, 0, 0)
        # Rectangular clamp brings x to 15.0; single-axis ellipsoid norm=1.0
        assert result[0] == pytest.approx(15.0)

    def test_negative_beyond_limits(self, mixin: _TestMixin):
        result = mixin._clamp_to_workspace_6d(0, -50, 0, 0, 0, 0)
        assert result[1] == pytest.approx(-18.0)

    def test_rotation_clamped(self, mixin: _TestMixin):
        result = mixin._clamp_to_workspace_6d(0, 0, 0, 0, 0, 100.0)
        assert result[5] == pytest.approx(55.0)

    def test_translation_ellipsoid_clamping(self, mixin: _TestMixin):
        """When all three translation axes are at their limits, the ellipsoid
        norm = 3.0 > 1.0 so values get scaled down."""
        x, y, z = 15.0, 18.0, 15.0
        result = mixin._clamp_to_workspace_6d(x, y, z, 0, 0, 0)
        # After rectangular clamp: all at limits
        # Ellipsoid norm = 1 + 1 + 1 = 3, scale = 1/sqrt(3)
        scale = 1.0 / math.sqrt(3.0)
        assert result[0] == pytest.approx(15.0 * scale)
        assert result[1] == pytest.approx(18.0 * scale)
        assert result[2] == pytest.approx(15.0 * scale)

    def test_rotation_ellipsoid_clamping(self, mixin: _TestMixin):
        """All three rotation axes at limits triggers ellipsoid scaling."""
        result = mixin._clamp_to_workspace_6d(0, 0, 0, 35.0, 35.0, 55.0)
        scale = 1.0 / math.sqrt(3.0)
        assert result[3] == pytest.approx(35.0 * scale)
        assert result[4] == pytest.approx(35.0 * scale)
        assert result[5] == pytest.approx(55.0 * scale)

    def test_combined_constraint(self, mixin: _TestMixin):
        """When both trans and rot usage are high, combined constraint kicks in."""
        # Use single-axis values at the limit so ellipsoid norm = 1.0 each
        # trans_usage = 1.0, rot_usage = 1.0 => combined = 1.0 > 0.85
        result = mixin._clamp_to_workspace_6d(15.0, 0, 0, 0, 0, 55.0)
        # After rectangular+ellipsoid: x=15.0, yaw=55.0
        # trans_usage = 1.0, rot_usage = 1.0 (checking only yaw_limit>0 gate)
        # combined = 0.5*1 + 0.5*1 = 1.0 > 0.85 => scale = 0.85
        expected_scale = 0.85 / 1.0
        assert result[0] == pytest.approx(15.0 * expected_scale)
        assert result[5] == pytest.approx(55.0 * expected_scale)

    def test_combined_constraint_not_triggered_when_low(self, mixin: _TestMixin):
        """Small values should not trigger the combined constraint."""
        result = mixin._clamp_to_workspace_6d(1.0, 0, 0, 0, 0, 5.0)
        assert result[0] == pytest.approx(1.0)
        assert result[5] == pytest.approx(5.0)

    def test_symmetry(self, mixin: _TestMixin):
        """Positive and negative should produce symmetric results."""
        pos = mixin._clamp_to_workspace_6d(10, 10, 10, 20, 20, 30)
        neg = mixin._clamp_to_workspace_6d(-10, -10, -10, -20, -20, -30)
        for p, n in zip(pos, neg):
            assert p == pytest.approx(-n)


# ═══════════════════════════════════════════════════════════════════════════
# _clamp_to_workspace (legacy 2D)
# ═══════════════════════════════════════════════════════════════════════════


class TestClampToWorkspace2D:
    def test_within_limits(self, mixin: _TestMixin):
        yaw_out, pitch_out = mixin._clamp_to_workspace(10.0, 5.0)
        assert yaw_out == pytest.approx(10.0)
        assert pitch_out == pytest.approx(5.0)

    def test_beyond_limits(self, mixin: _TestMixin):
        yaw_out, pitch_out = mixin._clamp_to_workspace(100.0, 100.0)
        # Single-axis: rectangular clamp only (ellipsoid=1 for each axis alone)
        # But combined: pitch=35, yaw=55 => rot_usage = sqrt((35/35)^2 + (55/55)^2) = sqrt(2)
        # Then rot ellipsoid kicks in: scale = 1/sqrt(2)
        # After rot ellipsoid: pitch=35/sqrt(2), yaw=55/sqrt(2)
        # trans_usage=0, rot_usage=1.0, combined=0.5 < 0.85 => no combined
        scale = 1.0 / math.sqrt(2.0)
        assert yaw_out == pytest.approx(55.0 * scale)
        assert pitch_out == pytest.approx(35.0 * scale)

    def test_returns_yaw_pitch_order(self, mixin: _TestMixin):
        """Verify the return order is (yaw, pitch) not (pitch, yaw)."""
        yaw_out, pitch_out = mixin._clamp_to_workspace(20.0, 10.0)
        assert yaw_out == pytest.approx(20.0)
        assert pitch_out == pytest.approx(10.0)


# ═══════════════════════════════════════════════════════════════════════════
# _calculate_movement_for_pixel
# ═══════════════════════════════════════════════════════════════════════════


class TestCalculateMovementForPixel:
    def test_center_pixel_zero(self, mixin: _TestMixin):
        """Zero offset should produce all-zero movement."""
        result = mixin._calculate_movement_for_pixel(0.0, 0.0)
        assert result == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_positive_du(self, mixin: _TestMixin):
        """Positive du (right of center) -> negative dy and negative dyaw."""
        dx, dy, dz, droll, dpitch, dyaw = mixin._calculate_movement_for_pixel(100.0, 0.0)
        assert dx == 0.0
        assert dy == pytest.approx(-100.0 * 0.05)
        assert dz == 0.0
        assert droll == 0.0
        assert dpitch == 0.0
        assert dyaw == pytest.approx(-100.0 * 0.10)

    def test_positive_dv(self, mixin: _TestMixin):
        """Positive dv (below center) -> negative dz and positive dpitch."""
        dx, dy, dz, droll, dpitch, dyaw = mixin._calculate_movement_for_pixel(0.0, 100.0)
        assert dx == 0.0
        assert dy == 0.0
        assert dz == pytest.approx(-100.0 * 0.04)
        assert droll == 0.0
        assert dpitch == pytest.approx(100.0 * 0.08)
        assert dyaw == 0.0

    def test_negative_du(self, mixin: _TestMixin):
        """Negative du (left of center) -> positive dy and positive dyaw."""
        dx, dy, dz, droll, dpitch, dyaw = mixin._calculate_movement_for_pixel(-50.0, 0.0)
        assert dy == pytest.approx(50.0 * 0.05)
        assert dyaw == pytest.approx(50.0 * 0.10)

    def test_negative_dv(self, mixin: _TestMixin):
        """Negative dv (above center) -> positive dz and negative dpitch."""
        dx, dy, dz, droll, dpitch, dyaw = mixin._calculate_movement_for_pixel(0.0, -80.0)
        assert dz == pytest.approx(80.0 * 0.04)
        assert dpitch == pytest.approx(-80.0 * 0.08)

    def test_combined_du_dv(self, mixin: _TestMixin):
        """Both offsets at once."""
        dx, dy, dz, droll, dpitch, dyaw = mixin._calculate_movement_for_pixel(30.0, 20.0)
        assert dy == pytest.approx(-30.0 * 0.05)
        assert dz == pytest.approx(-20.0 * 0.04)
        assert dyaw == pytest.approx(-30.0 * 0.10)
        assert dpitch == pytest.approx(20.0 * 0.08)
        assert dx == 0.0
        assert droll == 0.0

    def test_dx_and_droll_always_zero(self, mixin: _TestMixin):
        """dx and droll should always be zero regardless of input."""
        for du, dv in [(10, 20), (-5, 15), (0, -100), (200, 200)]:
            dx, _, _, droll, _, _ = mixin._calculate_movement_for_pixel(du, dv)
            assert dx == 0.0
            assert droll == 0.0
