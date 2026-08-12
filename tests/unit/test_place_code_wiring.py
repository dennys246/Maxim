"""Guards for the exteroceptive place-code wiring (Exp 46 promotion).

The claim being wired: a raw azimuth scalar resolves into ~2-3 EC clusters at the
production sensor threshold, and a Gaussian population code over the same range
resolves into many more — Exp 46 measured 6/6 and got the behavioural result
(taught 0.19 → 0.82). These tests pin the RESOLUTION DELTA through the real
SensorEncoder + real EntorhinalCortex, not just the arithmetic of the code.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from maxim.runtime.agent_loop import (
    _read_exteroceptive_ranges,
    _read_exteroceptive_states,
    place_code_exteroception_enabled,
)
from maxim.similarity.ec import EntorhinalCortex
from maxim.similarity.encoder import SensorEncoder
from maxim.similarity.place_code import DEFAULT_CELL_CENTERS, place_code, place_code_ranges


def _executor(az: float | None = 0.0, *, with_sensor: bool = True):
    """Minimal duck-typed executor with an azimuth root sensor."""
    sensor = SimpleNamespace(reading_schema={"range": [-1.0, 1.0]})
    sensors = {"azimuth": sensor} if with_sensor else {}
    vm = {"azimuth": az} if (with_sensor and az is not None) else {}
    return SimpleNamespace(embodiment=SimpleNamespace(root=SimpleNamespace(sensors=sensors, vital_metrics=vm)))


class TestPureFunction:
    def test_peak_at_center_and_falloff(self):
        code = place_code(-0.9, prefix="d")
        assert code["d0"] == pytest.approx(1.0)  # centre -0.9
        assert code["d1"] < 0.2  # neighbour at -0.6, 2.5 widths away

    def test_distinct_directions_peak_on_distinct_cells(self):
        peaks = []
        for az in DEFAULT_CELL_CENTERS:
            code = place_code(az, prefix="d")
            peaks.append(max(code, key=lambda k: code[k]))
        assert len(set(peaks)) == len(DEFAULT_CELL_CENTERS)  # one cell per direction

    def test_left_and_right_never_collapse(self):
        """The anti-ring-code invariant: az=-1 and az=+1 are 180 deg apart
        (doa_to_azimuth: -1 = hard left, +1 = hard right). A circular encoding
        would identify them and destroy Exp 45/48's discrimination."""
        left, right = place_code(-1.0, prefix="d"), place_code(1.0, prefix="d")
        assert set(left) & set(right) == set()  # no shared active cell at all

    def test_ranges_declared_for_every_cell(self):
        rngs = place_code_ranges(prefix="d")
        assert len(rngs) == len(DEFAULT_CELL_CENTERS)
        assert set(rngs.values()) == {(0.0, 1.0)}

    def test_zero_width_rejected(self):
        with pytest.raises(ValueError):
            place_code(0.0, prefix="d", width=0.0)


class TestFlagWiring:
    def test_default_off_is_byte_identical(self):
        assert place_code_exteroception_enabled() is False
        assert _read_exteroceptive_states(_executor(0.5)) == {"azimuth": 0.5}
        assert _read_exteroceptive_ranges(_executor(0.5)) == {"azimuth": (-1.0, 1.0)}

    def test_enabled_replaces_scalar_with_population(self, monkeypatch):
        monkeypatch.setenv("MAXIM_PLACE_CODE_EXTEROCEPTION", "1")
        vals = _read_exteroceptive_states(_executor(-0.9))
        assert "azimuth" not in vals  # REPLACES, never emits both (dilution)
        assert all(k.startswith("azdir_azimuth_") for k in vals)
        assert max(vals.values()) == pytest.approx(1.0)

    def test_value_and_range_walks_stay_in_lockstep(self, monkeypatch):
        """The _read_drive_ranges lesson applied here: a value with no declared
        range silently re-folds through the legacy range-blind map."""
        monkeypatch.setenv("MAXIM_PLACE_CODE_EXTEROCEPTION", "1")
        ex = _executor(0.3)
        vals, rngs = _read_exteroceptive_states(ex), _read_exteroceptive_ranges(ex)
        assert set(vals).issubset(set(rngs)), "every emitted value needs a declared range"

    def test_no_azimuth_sensor_emits_nothing(self, monkeypatch):
        """A body without the sensor must not acquire a phantom audio channel."""
        monkeypatch.setenv("MAXIM_PLACE_CODE_EXTEROCEPTION", "1")
        assert _read_exteroceptive_states(_executor(with_sensor=False)) == {}
        assert _read_exteroceptive_ranges(_executor(with_sensor=False)) == {}


class TestResolutionDelta:
    """The load-bearing test: measured through the REAL encoder + REAL EC at the
    production sensor threshold and the production (frozen) modality tag."""

    def _nodes(self, coded: bool) -> int:
        enc = SensorEncoder(ec=EntorhinalCortex())
        seen = set()
        for i in range(21):
            az = -1.0 + i * 0.1
            sensors = place_code(az, prefix="d") if coded else {"azimuth": az}
            ranges = place_code_ranges(prefix="d") if coded else {"azimuth": (-1.0, 1.0)}
            node = enc.encode_sensors(agent_id=f"res_{coded}", sensors=sensors, modality="audio", ranges=ranges)
            if node:
                seen.add(node)
        return len(seen)

    def test_place_code_resolves_more_than_raw_scalar(self):
        raw, coded = self._nodes(False), self._nodes(True)
        assert raw <= 3, f"raw azimuth should be coarse (Exp 46: ~2), got {raw}"
        assert coded >= 6, f"place code should resolve per-direction (Exp 46: 6/6), got {coded}"
        assert coded > raw

    def test_provenance_records_the_shape_change(self):
        """encoder_provenance stamps sensor_names, so a CALLER-side place code is
        at least auditable in a persisted substrate (nothing compares it at load
        yet — see the plan's follow-ups)."""
        ec = EntorhinalCortex()
        enc = SensorEncoder(ec=ec)
        enc.encode_sensors(
            agent_id="prov",
            sensors=place_code(0.0, prefix="d"),
            modality="audio",
            ranges=place_code_ranges(prefix="d"),
        )
        prov = getattr(ec, "encoder_provenance", None) or {}
        flat = repr(prov)
        assert "d0" in flat or "sensor_names" in flat


def test_env_var_has_autouse_scrub():
    """Pins the CLAUDE.md rule: opt-in env vars on hot paths ship with a scrub."""
    assert os.environ.get("MAXIM_PLACE_CODE_EXTEROCEPTION") is None
