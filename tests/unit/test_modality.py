"""Tests for sensory modality types and SensoryGate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from maxim.agents.modality import SensoryModality, SensoryTag
from maxim.agents.sensory_gate import ACUITY_FLOOR, MODALITY_SENSOR_MAP, SensoryGate


# ── Mock classes for testing ──────────────────────────────────────────────────


@dataclass
class MockSensorReading:
    sensor_name: str
    entity_name: str
    value: Any
    unit: str = "ratio"


class MockSensor:
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self._value = value
        self.reading_schema = {"type": "float", "range": [0, 1]}

    def read(self) -> MockSensorReading:
        return MockSensorReading(
            sensor_name=self.name,
            entity_name="test_entity",
            value=self._value,
        )


@dataclass
class MockEntity:
    name: str = "perception"
    entity_type: str = "sensory_bundle"
    full_path: str = "derek.perception"
    sensors: dict[str, MockSensor] = field(default_factory=dict)


@dataclass
class MockPercept:
    timestamp: float = 0.0
    source: str = "dm"
    content: str = "test"
    salience: float = 0.7
    sensory: SensoryTag | None = None


# ── SensoryModality tests ────────────────────────────────────────────────────


class TestSensoryModality:
    def test_all_modalities_have_values(self) -> None:
        assert len(SensoryModality) == 7
        values = {m.value for m in SensoryModality}
        assert "sight" in values
        assert "sound" in values
        assert "touch" in values
        assert "smell" in values
        assert "intero" in values
        assert "narrative" in values
        assert "abstract" in values

    def test_modality_from_value(self) -> None:
        assert SensoryModality("sight") == SensoryModality.SIGHT
        assert SensoryModality("sound") == SensoryModality.SOUND


# ── SensoryTag tests ─────────────────────────────────────────────────────────


class TestSensoryTag:
    def test_creation_defaults(self) -> None:
        tag = SensoryTag(modality=SensoryModality.SIGHT)
        assert tag.modality == SensoryModality.SIGHT
        assert tag.submodality == ""
        assert tag.spatial_source == ""
        assert tag.intensity == 0.5
        assert tag.entity_source == ""
        assert tag.perceived_intensity is None
        assert tag.modulated_by == ""

    def test_creation_full(self) -> None:
        tag = SensoryTag(
            modality=SensoryModality.SOUND,
            submodality="speech",
            spatial_source="behind",
            intensity=0.8,
            entity_source="guard_captain",
        )
        assert tag.modality == SensoryModality.SOUND
        assert tag.submodality == "speech"
        assert tag.spatial_source == "behind"
        assert tag.intensity == 0.8
        assert tag.entity_source == "guard_captain"

    def test_frozen(self) -> None:
        tag = SensoryTag(modality=SensoryModality.SIGHT)
        with pytest.raises(AttributeError):
            tag.intensity = 0.9  # type: ignore[misc]

    def test_round_trip_serialization(self) -> None:
        tag = SensoryTag(
            modality=SensoryModality.TOUCH,
            submodality="pain",
            spatial_source="left_arm",
            intensity=0.6,
            entity_source="longsword",
            perceived_intensity=0.42,
            modulated_by="derek.perception.pain_sensitivity",
        )
        d = tag.to_dict()
        restored = SensoryTag.from_dict(d)
        assert restored == tag

    def test_from_dict_defaults(self) -> None:
        tag = SensoryTag.from_dict({"modality": "smell"})
        assert tag.modality == SensoryModality.SMELL
        assert tag.intensity == 0.5
        assert tag.perceived_intensity is None


# ── SensoryGate tests ─────────────────────────────────────────────────────────


class TestSensoryGate:
    def _make_entity(self, **acuities: float) -> MockEntity:
        sensors = {}
        for name, value in acuities.items():
            sensors[name] = MockSensor(name, value)
        return MockEntity(sensors=sensors)

    def test_legacy_percept_passes_through(self) -> None:
        entity = self._make_entity(sight_acuity=0.8)
        gate = SensoryGate(entity)
        percept = MockPercept(sensory=None)
        result = gate.modulate(percept)
        assert result is percept  # Same object, unmodified

    def test_narrative_passes_through(self) -> None:
        entity = self._make_entity(sight_acuity=0.0)  # blind
        gate = SensoryGate(entity)
        tag = SensoryTag(modality=SensoryModality.NARRATIVE, intensity=0.9)
        percept = MockPercept(sensory=tag)
        result = gate.modulate(percept)
        assert result is percept  # Narrative is never filtered

    def test_abstract_passes_through(self) -> None:
        entity = self._make_entity()
        gate = SensoryGate(entity)
        tag = SensoryTag(modality=SensoryModality.ABSTRACT)
        percept = MockPercept(sensory=tag)
        result = gate.modulate(percept)
        assert result is percept

    def test_sight_modulated_by_acuity(self) -> None:
        entity = self._make_entity(sight_acuity=0.6)
        gate = SensoryGate(entity)
        tag = SensoryTag(modality=SensoryModality.SIGHT, intensity=0.8)
        percept = MockPercept(sensory=tag, salience=0.7)
        result = gate.modulate(percept)
        assert result is not None
        assert result.sensory.perceived_intensity == pytest.approx(0.48)  # 0.8 * 0.6
        assert result.salience == pytest.approx(0.42)  # 0.7 * 0.6
        assert "sight_acuity" in result.sensory.modulated_by

    def test_blind_drops_sight_percept(self) -> None:
        entity = self._make_entity(sight_acuity=0.03)  # Below ACUITY_FLOOR
        gate = SensoryGate(entity)
        tag = SensoryTag(modality=SensoryModality.SIGHT, intensity=0.9)
        percept = MockPercept(sensory=tag)
        result = gate.modulate(percept)
        assert result is None  # Dropped — entity is blind

    def test_deaf_drops_sound_percept(self) -> None:
        entity = self._make_entity(hearing_acuity=0.0)
        gate = SensoryGate(entity)
        tag = SensoryTag(modality=SensoryModality.SOUND, intensity=0.7)
        percept = MockPercept(sensory=tag)
        result = gate.modulate(percept)
        assert result is None

    def test_no_sensor_passes_through(self) -> None:
        """If entity has no sensor for this modality, percept passes unmodulated."""
        entity = self._make_entity()  # No sensors at all
        gate = SensoryGate(entity)
        tag = SensoryTag(modality=SensoryModality.SIGHT, intensity=0.8)
        percept = MockPercept(sensory=tag, salience=0.7)
        result = gate.modulate(percept)
        assert result is percept  # Unmodulated

    def test_high_acuity_minimal_reduction(self) -> None:
        entity = self._make_entity(hearing_acuity=0.95)
        gate = SensoryGate(entity)
        tag = SensoryTag(modality=SensoryModality.SOUND, intensity=0.6)
        percept = MockPercept(sensory=tag, salience=0.8)
        result = gate.modulate(percept)
        assert result is not None
        assert result.sensory.perceived_intensity == pytest.approx(0.57)  # 0.6 * 0.95
        assert result.salience == pytest.approx(0.76)  # 0.8 * 0.95

    def test_smell_modulation(self) -> None:
        entity = self._make_entity(smell_acuity=0.3)
        gate = SensoryGate(entity)
        tag = SensoryTag(modality=SensoryModality.SMELL, intensity=0.5, entity_source="forge")
        percept = MockPercept(sensory=tag)
        result = gate.modulate(percept)
        assert result is not None
        assert result.sensory.perceived_intensity == pytest.approx(0.15)
        assert result.sensory.entity_source == "forge"  # Preserved

    def test_get_acuity(self) -> None:
        entity = self._make_entity(sight_acuity=0.7, hearing_acuity=0.4)
        gate = SensoryGate(entity)
        assert gate.get_acuity(SensoryModality.SIGHT) == pytest.approx(0.7)
        assert gate.get_acuity(SensoryModality.SOUND) == pytest.approx(0.4)
        assert gate.get_acuity(SensoryModality.SMELL) == 1.0  # No sensor → unmodulated

    def test_is_blind_and_deaf(self) -> None:
        entity = self._make_entity(sight_acuity=0.02, hearing_acuity=0.8)
        gate = SensoryGate(entity)
        assert gate.is_blind() is True
        assert gate.is_deaf() is False

    def test_modality_sensor_map_covers_physical_senses(self) -> None:
        """All physical senses should have a sensor mapping."""
        physical = {
            SensoryModality.SIGHT,
            SensoryModality.SOUND,
            SensoryModality.TOUCH,
            SensoryModality.SMELL,
            SensoryModality.INTEROCEPTION,
        }
        assert physical.issubset(set(MODALITY_SENSOR_MAP.keys()))

    def test_acuity_floor_is_sensible(self) -> None:
        assert 0.0 < ACUITY_FLOOR < 0.1
