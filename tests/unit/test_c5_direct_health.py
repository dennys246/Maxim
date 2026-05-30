"""C5: hard ConfigurationError for direct ``health`` sensor when an entity
also has modulators with sensors (component-damage model).

The 1.0 contract: one source of truth for entity health. When an entity
has modulators with sub-sensors, ``Body.evaluate_failures`` derives
``vital_metrics["health"]`` from modulator integrities. A direct
``health`` sensor duplicates that state and goes stale.

0.9.x emitted ``DeprecationWarning`` at parse time (PR #220, 2026-04-30).
1.0 flipped to ``ConfigurationError`` per v1_refinement.md §C5 (PR #299).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maxim.embodiment.spec import _parse_entity, load_spec
from maxim.exceptions import ConfigurationError


def _make_entity_dict(
    *,
    name: str = "test_entity",
    sensors: dict | None = None,
    modulators: dict | None = None,
    health_marker: object = None,
) -> dict:
    """Build a minimal entity-spec dict for ``_parse_entity``."""
    data: dict = {"name": name, "entity_type": "test"}
    if sensors is not None:
        data["sensors"] = sensors
    if modulators is not None:
        data["modulators"] = modulators
    if health_marker is not None:
        data["health"] = health_marker
    return data


# ---------------------------------------------------------------------------
# (a) entity with modulators-with-sensors AND direct health → RAISE
# ---------------------------------------------------------------------------


class TestOffenderRaises:
    def test_modulators_with_sensors_plus_direct_health_raises(self):
        data = _make_entity_dict(
            name="dragon",
            sensors={
                "health": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
            },
            modulators={
                "wing": {
                    "sensors": {
                        "integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
                    },
                    "affordances": {},
                },
            },
        )
        with pytest.raises(ConfigurationError) as exc:
            _parse_entity(data)

        msg = str(exc.value)
        assert "dragon" in msg
        assert "(C5)" in msg
        # Both opt-in paths surfaced in the message.
        assert "health: derived" in msg
        assert "remove the direct 'health' sensor" in msg


# ---------------------------------------------------------------------------
# (b) entity with modulators-with-sensors AND derived health → no raise
# ---------------------------------------------------------------------------


class TestDerivedOptIn:
    def test_health_derived_string_marker_suppresses_error(self):
        data = _make_entity_dict(
            name="dragon_v2",
            sensors={
                "health": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
            },
            modulators={
                "wing": {
                    "sensors": {
                        "integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
                    },
                    "affordances": {},
                },
            },
            health_marker="derived",
        )
        # Must not raise.
        _parse_entity(data)

    def test_health_derived_boolean_true_does_NOT_suppress_error(self):
        # Single canonical form for 1.0: only the string "derived" suppresses.
        # Boolean True is rejected so we ship one valid spelling — prevents
        # "two valid forms" drift.
        data = _make_entity_dict(
            name="dragon_v3",
            sensors={
                "health": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
            },
            modulators={
                "wing": {
                    "sensors": {
                        "integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
                    },
                    "affordances": {},
                },
            },
            health_marker=True,
        )
        with pytest.raises(ConfigurationError):
            _parse_entity(data)


# ---------------------------------------------------------------------------
# (c) entity with only modulators-with-sensors (no direct health) → no raise
# ---------------------------------------------------------------------------


class TestModulatorOnlyNoRaise:
    def test_modulators_with_sensors_no_health_sensor_no_raise(self):
        data = _make_entity_dict(
            name="hardened_drone",
            sensors={
                "battery": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
            },
            modulators={
                "rotor": {
                    "sensors": {
                        "integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
                    },
                    "affordances": {},
                },
            },
        )
        _parse_entity(data)


# ---------------------------------------------------------------------------
# (d) primitive entity with only direct health (no modulators) → no raise
# ---------------------------------------------------------------------------


class TestPrimitiveNoRaise:
    def test_direct_health_alone_no_raise(self):
        data = _make_entity_dict(
            name="apple",
            sensors={
                "health": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
            },
        )
        _parse_entity(data)

    def test_direct_health_with_capability_only_modulator_no_raise(self):
        # Capability-only modulators (no sensors) don't trigger component
        # damage modeling, so direct health is still a valid single source
        # of truth. Mirrors npcs/ferryman.yaml shape.
        data = _make_entity_dict(
            name="ferryman",
            sensors={
                "trust": {"unit": "ratio", "range": [0, 1], "initial": 0.3},
                "health": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
            },
            modulators={
                "social": {
                    "abstract": True,
                    "affordances": {
                        "speak": {"params": {"message": "str"}, "description": "say"},
                    },
                },
            },
        )
        _parse_entity(data)


# ---------------------------------------------------------------------------
# Bundled component audit — every shipped YAML must load cleanly post-flip.
# ---------------------------------------------------------------------------


class TestBundledComponentsClean:
    def test_no_bundled_component_violates_c5(self):
        bundled = Path(__file__).resolve().parents[2] / "src" / "maxim" / "_data" / "components"
        assert bundled.is_dir(), f"bundled components dir not found: {bundled}"

        offenders: list[tuple[str, str]] = []
        for yaml_path in sorted(bundled.rglob("*.yaml")):
            try:
                load_spec(yaml_path)
            except ConfigurationError as exc:
                if "(C5)" in str(exc):
                    offenders.append((str(yaml_path), str(exc)))
            except Exception:
                # Other load errors (missing registry, etc.) are out of C5 scope.
                continue

        assert offenders == [], (
            "Bundled components violate C5 — migrate to "
            "'health: derived' or remove the direct health sensor:\n" + "\n".join(f"  {p}: {m}" for p, m in offenders)
        )


# ---------------------------------------------------------------------------
# Recursion: the check fires on the first offender during recursive parse.
# ---------------------------------------------------------------------------


class TestRecursionRaisesOnOffender:
    def test_offending_child_in_clean_root_raises(self):
        data = {
            "name": "carrier",
            "entity_type": "vehicle",
            "sensors": {"speed": {"unit": "ratio", "range": [0, 1], "initial": 0.0}},
            "modulators": {
                "engine": {
                    "sensors": {"rpm": {"unit": "ratio", "range": [0, 1], "initial": 0.5}},
                    "affordances": {},
                },
            },
            "health": "derived",
            "children": [
                {
                    "name": "passenger_drone",
                    "entity_type": "drone",
                    "sensors": {
                        "health": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
                    },
                    "modulators": {
                        "rotor": {
                            "sensors": {"integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0}},
                            "affordances": {},
                        },
                    },
                },
            ],
        }
        with pytest.raises(ConfigurationError) as exc:
            _parse_entity(data)
        assert "passenger_drone" in str(exc.value)
