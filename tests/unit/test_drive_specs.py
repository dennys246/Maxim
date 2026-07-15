"""Unit tests for the drive protocol — HomeostaticDriveSpec + EntropicDriveSpec.

Tests cover:
- Dataclass construction and field validation
- YAML parsing via _parse_drive_spec
- Entity.drive_specs population from entity-level and modulator-level sensors
- Homeostatic drift toward set_point in tick_vital_drift
- Entropic drift in drift_direction in tick_vital_drift
- Wall-clock dt (actual elapsed time, not planned interval)
- Homeostatic pain evaluation (proportional to deviation beyond comfort_band)
- Entropic deprivation pain evaluation (threshold crossing)
- Drive pain published to PainBus
- Serialization/deserialization roundtrip (to_dict/from_dict)
"""

from __future__ import annotations

import pytest

from maxim.embodiment.sem import (
    CouplingSpec,
    Entity,
    EntropicDriveSpec,
    HomeostaticDriveSpec,
    ModulationSpec,
)


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestDriveSpecConstruction:
    def test_homeostatic_defaults(self):
        ds = HomeostaticDriveSpec(set_point=0.0, drift_rate=0.001)
        assert ds.comfort_band == 0.0
        assert ds.pain_scale == 0.5
        assert ds.pain_model == "linear"
        assert ds.modulated_by is None

    def test_homeostatic_with_modulation(self):
        mod = ModulationSpec(source="scn", target_field="set_point", modulation_range=(-0.1, 0.1))
        ds = HomeostaticDriveSpec(
            set_point=0.0,
            drift_rate=0.001,
            modulated_by=(mod,),
        )
        assert ds.modulated_by is not None
        assert len(ds.modulated_by) == 1
        assert ds.modulated_by[0].source == "scn"

    def test_entropic_required_fields(self):
        ds = EntropicDriveSpec(
            drift_direction="up",
            drift_rate=0.002,
            deprivation_threshold=0.7,
            deprivation_pain=0.3,
            satisfaction_threshold=0.3,
        )
        assert ds.drift_direction == "up"
        assert ds.coupled_to is None

    def test_entropic_with_coupling(self):
        coupling = CouplingSpec(sensor="stamina", below=0.4, multiplier=2.0)
        ds = EntropicDriveSpec(
            drift_direction="up",
            drift_rate=0.002,
            deprivation_threshold=0.7,
            deprivation_pain=0.3,
            satisfaction_threshold=0.3,
            coupled_to=(coupling,),
        )
        assert ds.coupled_to is not None
        assert ds.coupled_to[0].multiplier == 2.0

    def test_drive_spec_union_type(self):
        h = HomeostaticDriveSpec(set_point=0.0, drift_rate=0.001)
        e = EntropicDriveSpec(
            drift_direction="up",
            drift_rate=0.002,
            deprivation_threshold=0.7,
            deprivation_pain=0.3,
            satisfaction_threshold=0.3,
        )
        # Both should be valid DriveSpec
        assert isinstance(h, (HomeostaticDriveSpec, EntropicDriveSpec))
        assert isinstance(e, (HomeostaticDriveSpec, EntropicDriveSpec))

    def test_frozen(self):
        ds = HomeostaticDriveSpec(set_point=0.0, drift_rate=0.001)
        with pytest.raises(AttributeError):
            ds.set_point = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


class TestDriveParsing:
    def test_parse_homeostatic(self):
        from maxim.embodiment.spec import _parse_drive_spec

        data = {
            "drift_mode": "homeostatic",
            "set_point": 0.0,
            "drift_rate": 0.001,
            "comfort_band": 0.4,
            "pain_scale": 0.5,
        }
        ds = _parse_drive_spec(data)
        assert isinstance(ds, HomeostaticDriveSpec)
        assert ds.set_point == 0.0
        assert ds.comfort_band == 0.4

    def test_parse_entropic(self):
        from maxim.embodiment.spec import _parse_drive_spec

        data = {
            "drift_mode": "entropic",
            "drift_direction": "up",
            "drift_rate": 0.002,
            "deprivation_threshold": 0.7,
            "deprivation_pain": 0.3,
            "satisfaction_threshold": 0.3,
        }
        ds = _parse_drive_spec(data)
        assert isinstance(ds, EntropicDriveSpec)
        assert ds.drift_direction == "up"

    def test_parse_with_coupling(self):
        from maxim.embodiment.spec import _parse_drive_spec

        data = {
            "drift_mode": "entropic",
            "drift_direction": "up",
            "drift_rate": 0.002,
            "deprivation_threshold": 0.7,
            "deprivation_pain": 0.3,
            "satisfaction_threshold": 0.3,
            "coupled_to": [{"sensor": "stamina", "below": 0.4, "multiplier": 2.0}],
        }
        ds = _parse_drive_spec(data)
        assert isinstance(ds, EntropicDriveSpec)
        assert ds.coupled_to is not None
        assert ds.coupled_to[0].sensor == "stamina"

    def test_parse_invalid_mode(self):
        from maxim.embodiment.spec import _parse_drive_spec

        with pytest.raises(ValueError, match="Unknown drive drift_mode"):
            _parse_drive_spec({"drift_mode": "unknown"})

    def test_entity_level_drive_from_yaml(self):
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "hunger": {
                    "unit": "ratio",
                    "range": [0, 1],
                    "initial": 0.0,
                    "drive": {
                        "drift_mode": "entropic",
                        "drift_direction": "up",
                        "drift_rate": 0.002,
                        "deprivation_threshold": 0.7,
                        "deprivation_pain": 0.3,
                        "satisfaction_threshold": 0.3,
                    },
                },
            },
        }
        entity = _parse_entity(data)
        assert "hunger" in entity.drive_specs
        assert isinstance(entity.drive_specs["hunger"], EntropicDriveSpec)

    def test_modulator_level_drive_from_yaml(self):
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "test_body",
            "entity_type": "body",
            "modulators": {
                "arms": {
                    "sensors": {
                        "thermal": {
                            "unit": "celsius_norm",
                            "range": [-1, 1],
                            "initial": 0.0,
                            "weight": 0.0,
                            "drive": {
                                "drift_mode": "homeostatic",
                                "set_point": 0.0,
                                "drift_rate": 0.0008,
                                "comfort_band": 0.5,
                                "pain_scale": 0.4,
                            },
                        },
                    },
                },
            },
        }
        entity = _parse_entity(data)
        assert "arms.thermal" in entity.drive_specs
        assert isinstance(entity.drive_specs["arms.thermal"], HomeostaticDriveSpec)


# ---------------------------------------------------------------------------
# Drift evaluation
# ---------------------------------------------------------------------------


class TestDriveDrift:
    def _make_entity_with_entropic_drive(self, initial: float = 0.0) -> Entity:
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "hunger": {
                    "unit": "ratio",
                    "range": [0, 1],
                    "initial": initial,
                    "drive": {
                        "drift_mode": "entropic",
                        "drift_direction": "up",
                        "drift_rate": 0.1,  # fast for testing
                        "deprivation_threshold": 0.7,
                        "deprivation_pain": 0.3,
                        "satisfaction_threshold": 0.3,
                    },
                },
            },
        }
        return _parse_entity(data)

    def _make_entity_with_homeostatic_drive(self, initial: float = 0.0) -> Entity:
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "temperature": {
                    "unit": "celsius_norm",
                    "range": [-1, 1],
                    "initial": initial,
                    "drive": {
                        "drift_mode": "homeostatic",
                        "set_point": 0.0,
                        "drift_rate": 0.1,  # fast for testing
                        "comfort_band": 0.3,
                        "pain_scale": 0.5,
                    },
                },
            },
        }
        return _parse_entity(data)

    def test_entropic_drift_up(self):
        from maxim.embodiment.body import Embodiment

        entity = self._make_entity_with_entropic_drive(initial=0.0)
        emb = Embodiment(entity)
        emb.tick_vital_drift(dt=1.0)
        assert entity.vital_metrics["hunger"] == pytest.approx(0.1, abs=1e-6)

    def test_entropic_drift_respects_dt(self):
        from maxim.embodiment.body import Embodiment

        entity = self._make_entity_with_entropic_drive(initial=0.0)
        emb = Embodiment(entity)
        emb.tick_vital_drift(dt=5.0)
        assert entity.vital_metrics["hunger"] == pytest.approx(0.5, abs=1e-6)

    def test_entropic_drift_clamps_at_1(self):
        from maxim.embodiment.body import Embodiment

        entity = self._make_entity_with_entropic_drive(initial=0.95)
        emb = Embodiment(entity)
        emb.tick_vital_drift(dt=1.0)
        assert entity.vital_metrics["hunger"] == 1.0

    def test_homeostatic_drift_toward_setpoint(self):
        from maxim.embodiment.body import Embodiment

        entity = self._make_entity_with_homeostatic_drive(initial=0.5)
        emb = Embodiment(entity)
        emb.tick_vital_drift(dt=1.0)
        # Should drift toward 0.0 by 0.1
        assert entity.vital_metrics["temperature"] == pytest.approx(0.4, abs=1e-6)

    def test_homeostatic_drift_negative_toward_setpoint(self):
        from maxim.embodiment.body import Embodiment

        entity = self._make_entity_with_homeostatic_drive(initial=-0.5)
        emb = Embodiment(entity)
        emb.tick_vital_drift(dt=1.0)
        # Should drift toward 0.0 by 0.1
        assert entity.vital_metrics["temperature"] == pytest.approx(-0.4, abs=1e-6)

    def test_homeostatic_drift_stops_at_setpoint(self):
        from maxim.embodiment.body import Embodiment

        entity = self._make_entity_with_homeostatic_drive(initial=0.05)
        emb = Embodiment(entity)
        emb.tick_vital_drift(dt=1.0)
        # drift_rate*dt = 0.1, but distance to set_point is only 0.05
        # min(0.05, 0.1) = 0.05 → should land at exactly 0.0
        assert entity.vital_metrics["temperature"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Pain evaluation
# ---------------------------------------------------------------------------


class TestDrivePain:
    def test_homeostatic_pain_outside_comfort_band(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "temperature": {
                    "unit": "celsius_norm",
                    "range": [-1, 1],
                    "initial": 0.8,  # way outside comfort_band of 0.3
                    "drive": {
                        "drift_mode": "homeostatic",
                        "set_point": 0.0,
                        "drift_rate": 0.001,
                        "comfort_band": 0.3,
                        "pain_scale": 0.5,
                    },
                },
            },
        }
        entity = _parse_entity(data)
        emb = Embodiment(entity)
        events = emb.evaluate_failures()

        drive_events = [e for e in events if e.failure_name.startswith("drive:")]
        assert len(drive_events) == 1
        assert drive_events[0].failure_name == "drive:temperature:discomfort"
        # excess = |0.8 - 0.0| - 0.3 = 0.5, pain = 0.5 * 0.5 = 0.25
        assert drive_events[0].pain_intensity == pytest.approx(0.25, abs=0.01)

    def test_homeostatic_no_pain_inside_comfort_band(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "temperature": {
                    "unit": "celsius_norm",
                    "range": [-1, 1],
                    "initial": 0.2,  # inside comfort_band of 0.3
                    "drive": {
                        "drift_mode": "homeostatic",
                        "set_point": 0.0,
                        "drift_rate": 0.001,
                        "comfort_band": 0.3,
                        "pain_scale": 0.5,
                    },
                },
            },
        }
        entity = _parse_entity(data)
        emb = Embodiment(entity)
        events = emb.evaluate_failures()

        drive_events = [e for e in events if e.failure_name.startswith("drive:")]
        assert len(drive_events) == 0

    def test_entropic_pain_above_threshold(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "hunger": {
                    "unit": "ratio",
                    "range": [0, 1],
                    "initial": 0.8,  # above deprivation_threshold 0.7
                    "drive": {
                        "drift_mode": "entropic",
                        "drift_direction": "up",
                        "drift_rate": 0.002,
                        "deprivation_threshold": 0.7,
                        "deprivation_pain": 0.3,
                        "satisfaction_threshold": 0.3,
                    },
                },
            },
        }
        entity = _parse_entity(data)
        emb = Embodiment(entity)
        events = emb.evaluate_failures()

        drive_events = [e for e in events if e.failure_name.startswith("drive:")]
        assert len(drive_events) == 1
        assert drive_events[0].failure_name == "drive:hunger:deprived"
        assert drive_events[0].pain_intensity == pytest.approx(0.3, abs=0.01)

    def test_entropic_no_pain_below_threshold(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "hunger": {
                    "unit": "ratio",
                    "range": [0, 1],
                    "initial": 0.5,  # below deprivation_threshold 0.7
                    "drive": {
                        "drift_mode": "entropic",
                        "drift_direction": "up",
                        "drift_rate": 0.002,
                        "deprivation_threshold": 0.7,
                        "deprivation_pain": 0.3,
                        "satisfaction_threshold": 0.3,
                    },
                },
            },
        }
        entity = _parse_entity(data)
        emb = Embodiment(entity)
        events = emb.evaluate_failures()

        drive_events = [e for e in events if e.failure_name.startswith("drive:")]
        assert len(drive_events) == 0

    def test_drive_pain_publishes_to_painbus(self):
        from unittest.mock import MagicMock

        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "hunger": {
                    "unit": "ratio",
                    "range": [0, 1],
                    "initial": 0.8,
                    "drive": {
                        "drift_mode": "entropic",
                        "drift_direction": "up",
                        "drift_rate": 0.002,
                        "deprivation_threshold": 0.7,
                        "deprivation_pain": 0.3,
                        "satisfaction_threshold": 0.3,
                    },
                },
            },
        }
        entity = _parse_entity(data)
        mock_bus = MagicMock()
        emb = Embodiment(entity, pain_bus=mock_bus)
        emb.evaluate_failures()

        assert mock_bus.publish.called
        signal = mock_bus.publish.call_args[0][0]
        assert signal.context["source"] == "drive:hunger"
        assert signal.intensity == pytest.approx(0.3, abs=0.01)


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


class TestDriveSerialization:
    def test_entity_roundtrip_with_drives(self):
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "hunger": {
                    "unit": "ratio",
                    "range": [0, 1],
                    "initial": 0.0,
                    "drive": {
                        "drift_mode": "entropic",
                        "drift_direction": "up",
                        "drift_rate": 0.002,
                        "deprivation_threshold": 0.7,
                        "deprivation_pain": 0.3,
                        "satisfaction_threshold": 0.3,
                    },
                },
                "temperature": {
                    "unit": "celsius_norm",
                    "range": [-1, 1],
                    "initial": 0.0,
                    "drive": {
                        "drift_mode": "homeostatic",
                        "set_point": 0.0,
                        "drift_rate": 0.001,
                        "comfort_band": 0.4,
                        "pain_scale": 0.5,
                    },
                },
            },
        }
        entity = _parse_entity(data)
        assert len(entity.drive_specs) == 2

        # Roundtrip
        d = entity.to_dict()
        restored = Entity.from_dict(d)
        assert len(restored.drive_specs) == 2
        assert isinstance(restored.drive_specs["hunger"], EntropicDriveSpec)
        assert isinstance(restored.drive_specs["temperature"], HomeostaticDriveSpec)
        assert restored.drive_specs["hunger"].drift_rate == 0.002
        assert restored.drive_specs["temperature"].comfort_band == 0.4


# ---------------------------------------------------------------------------
# Live tick cycle — auto-drift inside evaluate_failures (ed8b187f)
# ---------------------------------------------------------------------------


class _FakeTime:
    """Minimal stand-in for the ``time`` module binding in body.py."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def time(self) -> float:
        return self.now


class TestEvaluateFailuresAutoDrift:
    """Regression guard for the LIVE embodiment tick cycle.

    Commit ed8b187f (2026-04-26) moved drive drift INTO
    ``Body.evaluate_failures`` (wall-clock dt) because nothing constructed
    ``EmbodimentPerceptSource`` and drives were frozen. It shipped without
    tests; these pin the auto-drift semantics the production paths
    (tool_bridge, sim tools, substrate-primary tick) rely on. See the
    CLAUDE.md embodiment-tick invariant.
    """

    def _make_embodiment(self, initial: float = 0.0):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "hunger": {
                    "unit": "ratio",
                    "range": [0, 1],
                    "initial": initial,
                    "drive": {
                        "drift_mode": "entropic",
                        "drift_direction": "up",
                        "drift_rate": 0.1,  # fast for testing
                        "deprivation_threshold": 0.7,
                        "deprivation_pain": 0.3,
                        "satisfaction_threshold": 0.3,
                    },
                },
            },
        }
        entity = _parse_entity(data)
        return Embodiment(entity), entity

    def _patch_time(self, monkeypatch, start: float = 1000.0) -> _FakeTime:
        import maxim.embodiment.body as body_mod

        fake = _FakeTime(start)
        monkeypatch.setattr(body_mod, "time", fake)
        return fake

    def test_first_call_sets_baseline_without_drift(self, monkeypatch):
        emb, entity = self._make_embodiment(initial=0.0)
        self._patch_time(monkeypatch)
        emb.evaluate_failures()
        assert entity.vital_metrics["hunger"] == pytest.approx(0.0, abs=1e-9)

    def test_wall_clock_drift_applied_on_next_call(self, monkeypatch):
        emb, entity = self._make_embodiment(initial=0.0)
        fake = self._patch_time(monkeypatch)
        emb.evaluate_failures()  # baseline
        fake.now += 5.0
        emb.evaluate_failures()
        # drift_rate 0.1 × 5s elapsed wall time
        assert entity.vital_metrics["hunger"] == pytest.approx(0.5, abs=1e-6)

    def test_no_double_drift_at_same_instant(self, monkeypatch):
        emb, entity = self._make_embodiment(initial=0.0)
        fake = self._patch_time(monkeypatch)
        emb.evaluate_failures()  # baseline
        fake.now += 5.0
        emb.evaluate_failures()
        emb.evaluate_failures()  # same instant — dt == 0, no extra drift
        assert entity.vital_metrics["hunger"] == pytest.approx(0.5, abs=1e-6)

    def test_elapsed_time_applied_lazily_across_gaps(self, monkeypatch):
        """Time passing with NO call (e.g. LLM latency) is applied in full
        at the next call — deferred, never lost."""
        emb, entity = self._make_embodiment(initial=0.0)
        fake = self._patch_time(monkeypatch)
        emb.evaluate_failures()  # baseline
        fake.now += 2.0
        emb.evaluate_failures()
        assert entity.vital_metrics["hunger"] == pytest.approx(0.2, abs=1e-6)
        fake.now += 4.0  # long silent gap
        emb.evaluate_failures()
        assert entity.vital_metrics["hunger"] == pytest.approx(0.6, abs=1e-6)
