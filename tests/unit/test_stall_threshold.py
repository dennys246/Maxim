"""Tests for runtime/stall_threshold.py.

Pin the load-bearing invariants:
- Floor returned when lane_timeout_s is None / 0 / negative
- Derived value max(floor, lane_timeout_s + margin) when lane_timeout_s > 0
- Floor + margin env overrides
- Clamps for malformed / out-of-range env values
- **future kwargs accepted without TypeError (Stage 3 forward-compat)
- max_byte_silence_threshold_s honors env override + clamps
"""

from __future__ import annotations

import pytest

from maxim.runtime.stall_threshold import (
    DEFAULT_MAX_BYTE_SILENCE_S,
    DEFAULT_STALL_FLOOR_S,
    DEFAULT_STALL_MARGIN_S,
    compute_stall_threshold,
    max_byte_silence_threshold_s,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in (
        "MAXIM_STALL_FLOOR_S",
        "MAXIM_STALL_MARGIN_S",
        "MAXIM_STALL_MAX_BYTE_SILENCE_S",
        "MAXIM_HEARTBEAT_STALL_S",
    ):
        monkeypatch.delenv(name, raising=False)


# ─── compute_stall_threshold — lane_timeout cases ────────────────────────


def test_floor_when_no_lane_timeout():
    assert compute_stall_threshold(lane_tier="large") == DEFAULT_STALL_FLOOR_S


def test_floor_when_lane_timeout_zero():
    assert compute_stall_threshold(lane_tier="large", lane_timeout_s=0) == DEFAULT_STALL_FLOOR_S


def test_floor_when_lane_timeout_negative():
    assert compute_stall_threshold(lane_tier="large", lane_timeout_s=-100) == DEFAULT_STALL_FLOOR_S


def test_derived_value_when_lane_timeout_set():
    # Default floor=30, margin=10, lane_timeout=600 → max(30, 600+10) = 610
    assert compute_stall_threshold(lane_tier="large", lane_timeout_s=600) == 610.0


def test_floor_wins_when_lane_timeout_plus_margin_below_floor():
    # lane_timeout=5, margin=10 → 15 < floor 30 → floor wins
    assert compute_stall_threshold(lane_tier="large", lane_timeout_s=5) == DEFAULT_STALL_FLOOR_S


# ─── Env overrides ───────────────────────────────────────────────────────


def test_floor_env_override(monkeypatch):
    monkeypatch.setenv("MAXIM_STALL_FLOOR_S", "120")
    assert compute_stall_threshold(lane_tier="large") == 120.0
    # With lane_timeout < new floor: floor wins
    assert compute_stall_threshold(lane_tier="large", lane_timeout_s=50) == 120.0
    # With lane_timeout > new floor: derived wins
    assert compute_stall_threshold(lane_tier="large", lane_timeout_s=600) == 610.0


def test_margin_env_override(monkeypatch):
    monkeypatch.setenv("MAXIM_STALL_MARGIN_S", "60")
    assert compute_stall_threshold(lane_tier="large", lane_timeout_s=600) == 660.0


def test_margin_zero(monkeypatch):
    monkeypatch.setenv("MAXIM_STALL_MARGIN_S", "0")
    assert compute_stall_threshold(lane_tier="large", lane_timeout_s=600) == 600.0


def test_per_consumer_floor_env_var(monkeypatch):
    """Heartbeat uses its own floor env var (legacy compat)."""
    monkeypatch.setenv("MAXIM_HEARTBEAT_STALL_S", "45")
    result = compute_stall_threshold(
        lane_tier="large",
        floor_env_var="MAXIM_HEARTBEAT_STALL_S",
    )
    assert result == 45.0


# ─── Clamps for malformed / out-of-range ─────────────────────────────────


def test_malformed_floor_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MAXIM_STALL_FLOOR_S", "not-a-number")
    assert compute_stall_threshold(lane_tier="large") == DEFAULT_STALL_FLOOR_S


def test_malformed_margin_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MAXIM_STALL_MARGIN_S", "garbage")
    assert compute_stall_threshold(lane_tier="large", lane_timeout_s=600) == 600 + DEFAULT_STALL_MARGIN_S


def test_floor_clamp_below_minimum(monkeypatch):
    """Floor clamped to >= 5s."""
    monkeypatch.setenv("MAXIM_STALL_FLOOR_S", "0")
    assert compute_stall_threshold(lane_tier="large") == 5.0


def test_floor_clamp_above_maximum(monkeypatch):
    monkeypatch.setenv("MAXIM_STALL_FLOOR_S", "99999")
    # Clamped to 3600
    assert compute_stall_threshold(lane_tier="large") == 3600.0


def test_margin_clamp_above_maximum(monkeypatch):
    monkeypatch.setenv("MAXIM_STALL_MARGIN_S", "9999")
    # Clamped to 120
    assert compute_stall_threshold(lane_tier="large", lane_timeout_s=600) == 720.0


def test_margin_negative_clamped_to_zero(monkeypatch):
    monkeypatch.setenv("MAXIM_STALL_MARGIN_S", "-50")
    assert compute_stall_threshold(lane_tier="large", lane_timeout_s=600) == 600.0


# ─── **future kwargs (Stage 3 forward-compat) ────────────────────────────


def test_compute_stall_threshold_accepts_future_kwargs():
    # Must not raise TypeError; model + prompt_tokens reserved for Stage 3
    assert (
        compute_stall_threshold(
            lane_tier="large",
            lane_timeout_s=600,
            model="qwen2.5-32b",
            prompt_tokens=4096,
            some_future_param="anything",
        )
        == 610.0
    )


# ─── max_byte_silence_threshold_s ────────────────────────────────────────


def test_max_byte_silence_default():
    assert max_byte_silence_threshold_s() == DEFAULT_MAX_BYTE_SILENCE_S


def test_max_byte_silence_env_override(monkeypatch):
    monkeypatch.setenv("MAXIM_STALL_MAX_BYTE_SILENCE_S", "150")
    assert max_byte_silence_threshold_s() == 150.0


def test_max_byte_silence_clamps(monkeypatch):
    monkeypatch.setenv("MAXIM_STALL_MAX_BYTE_SILENCE_S", "10")
    assert max_byte_silence_threshold_s() == 30.0  # clamped to lo=30
    monkeypatch.setenv("MAXIM_STALL_MAX_BYTE_SILENCE_S", "9999")
    assert max_byte_silence_threshold_s() == 600.0  # clamped to hi=600


def test_max_byte_silence_malformed_falls_back(monkeypatch):
    monkeypatch.setenv("MAXIM_STALL_MAX_BYTE_SILENCE_S", "bad")
    assert max_byte_silence_threshold_s() == DEFAULT_MAX_BYTE_SILENCE_S
