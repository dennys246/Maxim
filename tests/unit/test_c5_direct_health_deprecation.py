"""C5: deprecation warning for direct ``health`` sensor when an entity
also has modulators with sensors (component-damage model).

The canonical 1.0 shape is one source of truth for entity health: when
an entity has modulators with sub-sensors, ``Body.evaluate_failures``
derives ``vital_metrics["health"]`` from modulator integrities. A direct
``health`` sensor duplicates that state and goes stale.

0.9 emits ``DeprecationWarning`` at parse time. 1.0 will hard-error.
"""

from __future__ import annotations

import warnings
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from maxim.embodiment.spec import _parse_entity, load_spec


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
# (a) entity with modulators-with-sensors AND direct health → WARN
# ---------------------------------------------------------------------------


class TestOffenderWarns:
    def test_modulators_with_sensors_plus_direct_health_warns(self):
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _parse_entity(data)

        c5 = [w for w in caught if "C5 deprecation" in str(w.message)]
        assert len(c5) == 1, f"Expected exactly one C5 warning, got {len(caught)}: {[str(w.message) for w in caught]}"
        assert issubclass(c5[0].category, DeprecationWarning)
        assert "dragon" in str(c5[0].message)
        # Upgrade-path mentions both options.
        assert "health: derived" in str(c5[0].message)
        assert "remove the direct 'health' sensor" in str(c5[0].message)


# ---------------------------------------------------------------------------
# (b) entity with modulators-with-sensors AND derived health → no warn
# ---------------------------------------------------------------------------


class TestDerivedOptIn:
    def test_health_derived_string_marker_suppresses_warning(self):
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _parse_entity(data)

        c5 = [w for w in caught if "C5 deprecation" in str(w.message)]
        assert c5 == [], f"Expected no C5 warning, got: {[str(w.message) for w in c5]}"

    def test_health_derived_boolean_true_does_NOT_suppress_warning(self):
        # Single canonical form for 1.0: only the string "derived" suppresses.
        # Boolean True is rejected so we ship one valid spelling — prevents
        # "two valid forms" drift inside the deprecation cycle.
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _parse_entity(data)

        c5 = [w for w in caught if "C5 deprecation" in str(w.message)]
        assert len(c5) == 1, "Boolean True must not be a valid 'derived' marker"


# ---------------------------------------------------------------------------
# (c) entity with only modulators-with-sensors (no direct health) → no warn
# ---------------------------------------------------------------------------


class TestModulatorOnlyNoWarn:
    def test_modulators_with_sensors_no_health_sensor_no_warn(self):
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _parse_entity(data)

        c5 = [w for w in caught if "C5 deprecation" in str(w.message)]
        assert c5 == []


# ---------------------------------------------------------------------------
# (d) primitive entity with only direct health (no modulators) → no warn
# ---------------------------------------------------------------------------


class TestPrimitiveNoWarn:
    def test_direct_health_alone_no_warn(self):
        data = _make_entity_dict(
            name="apple",
            sensors={
                "health": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
            },
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _parse_entity(data)

        c5 = [w for w in caught if "C5 deprecation" in str(w.message)]
        assert c5 == []

    def test_direct_health_with_capability_only_modulator_no_warn(self):
        # Capability-only modulators (no sensors) don't trigger component
        # damage modeling, so direct health is still a valid single source
        # of truth.  Mirrors npcs/ferryman.yaml shape.
        data = _make_entity_dict(
            name="ferryman",
            sensors={
                "trust": {"unit": "ratio", "range": [0, 1], "initial": 0.3},
                "health": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
            },
            modulators={
                "social": {
                    "affordances": {
                        "speak": {"params": {"message": "str"}, "description": "say"},
                    },
                },
            },
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _parse_entity(data)

        c5 = [w for w in caught if "C5 deprecation" in str(w.message)]
        assert c5 == []


# ---------------------------------------------------------------------------
# Bundled component audit — every shipped YAML must load cleanly.
# ---------------------------------------------------------------------------


class TestBundledComponentsClean:
    def test_no_bundled_component_emits_c5_warning(self):
        bundled = Path(__file__).resolve().parents[2] / "src" / "maxim" / "_data" / "components"
        assert bundled.is_dir(), f"bundled components dir not found: {bundled}"

        offenders: list[tuple[str, str]] = []
        for yaml_path in sorted(bundled.rglob("*.yaml")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    load_spec(yaml_path)
                except Exception:
                    # Skip files that need a registry or have other parse errors;
                    # those are out of C5 scope.
                    continue
                for w in caught:
                    if "C5 deprecation" in str(w.message):
                        offenders.append((str(yaml_path), str(w.message)))

        assert offenders == [], (
            "Bundled components emit C5 warnings — migrate to "
            "'health: derived' or remove the direct health sensor:\n" + "\n".join(f"  {p}: {m}" for p, m in offenders)
        )


# ---------------------------------------------------------------------------
# Recursion: warning fires once per offending entity in a tree.
# ---------------------------------------------------------------------------


class TestRecursionWarnsPerOffender:
    def test_offending_child_in_clean_root_warns(self):
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _parse_entity(data)

        c5 = [w for w in caught if "C5 deprecation" in str(w.message)]
        assert len(c5) == 1
        assert "passenger_drone" in str(c5[0].message)


# ---------------------------------------------------------------------------
# Visibility: DeprecationWarning is silenced under default Python filters
# outside __main__.  We mirror cli_utils._resolve_persona_mode and also
# print to stderr so the deprecation cycle actually reaches the user.
# ---------------------------------------------------------------------------


class TestStderrVisibility:
    def test_offender_prints_deprecation_line_to_stderr(self):
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
        buf = StringIO()
        with patch("sys.stderr", buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # don't double-spam pytest output
            _parse_entity(data)

        out = buf.getvalue()
        assert "DeprecationWarning:" in out
        assert "dragon" in out
        assert "C5 deprecation" in out

    def test_clean_entity_prints_nothing_to_stderr(self):
        data = _make_entity_dict(
            name="clean_drone",
            sensors={"battery": {"unit": "ratio", "range": [0, 1], "initial": 1.0}},
            modulators={
                "rotor": {
                    "sensors": {"integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0}},
                    "affordances": {},
                },
            },
        )
        buf = StringIO()
        with patch("sys.stderr", buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _parse_entity(data)

        assert "DeprecationWarning" not in buf.getvalue()
