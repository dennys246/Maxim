"""Unit tests for C4 — modulator-without-sensor hard error.

Per [docs/plans/archive/v1_refinement.md](../../docs/plans/archive/v1_refinement.md) §C4:
- A modulator that declares no sensors is silent dead weight for the
  component-damage model. Shipped as a ``DeprecationWarning`` in 0.9
  (PR #146, 2026-04-30); flipped to ``ConfigurationError`` in 1.0.
- Capability-only modulators (no sensors by design — ``weapon.combat``,
  ``npc.social``) opt out by declaring ``abstract: true`` in their YAML.
- A modulator with at least one sensor is fine and loads cleanly.
"""

from __future__ import annotations

import pytest

from maxim.embodiment.spec import _parse_entity
from maxim.exceptions import ConfigurationError


def _entity_with_modulator(mod_spec: dict) -> dict:
    return {
        "name": "test_entity",
        "entity_type": "test",
        "modulators": {
            "test_mod": mod_spec,
        },
    }


class TestC4ModulatorAbstract:
    def test_bare_modulator_raises_configuration_error(self):
        """No sensors + no abstract → ConfigurationError."""
        spec = _entity_with_modulator(
            {
                "affordances": {
                    "do_thing": {"params": {}, "description": "noop"},
                },
            }
        )
        with pytest.raises(ConfigurationError, match=r"declares no sensors"):
            _parse_entity(spec)

    def test_error_message_names_entity_and_modulator(self):
        """The error message must identify the offender so authors can fix it."""
        spec = _entity_with_modulator({"affordances": {}})
        with pytest.raises(ConfigurationError) as excinfo:
            _parse_entity(spec)
        msg = str(excinfo.value)
        assert "test_entity" in msg
        assert "test_mod" in msg
        # And the upgrade path
        assert "abstract: true" in msg

    def test_modulator_with_sensor_loads_cleanly(self):
        """A modulator with at least one sensor is honest about its state."""
        spec = _entity_with_modulator(
            {
                "sensors": {
                    "integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
                },
                "affordances": {
                    "do_thing": {"params": {}, "description": "noop"},
                },
            }
        )
        entity = _parse_entity(spec)
        assert "test_mod" in entity.modulators
        assert entity.modulators["test_mod"].sensors == {
            "integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
        }

    def test_abstract_modulator_loads_cleanly(self):
        """`abstract: true` is the explicit opt-out for capability-only modulators."""
        spec = _entity_with_modulator(
            {
                "abstract": True,
                "affordances": {
                    "slash": {"params": {"target": "str"}, "description": "swing"},
                },
            }
        )
        entity = _parse_entity(spec)
        mod = entity.modulators["test_mod"]
        assert mod.abstract is True
        assert mod.sensors == {}
        assert "slash" in mod.affordances

    def test_abstract_field_default_is_false(self):
        """Modulators with sensors don't need to set abstract — default is False."""
        spec = _entity_with_modulator(
            {
                "sensors": {
                    "integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
                },
                "affordances": {},
            }
        )
        entity = _parse_entity(spec)
        assert entity.modulators["test_mod"].abstract is False

    def test_bundled_components_load_without_configuration_error(self):
        """The bundled `_data/components/**/*.yaml` set must parse cleanly.

        Otherwise every user who imports a bundled component crashes —
        exactly the unprofessional state the audit was meant to prevent.
        Regression guard: if a new bundled component lands without sensors
        and without `abstract: true`, this test fails.
        """
        from pathlib import Path

        import yaml

        import maxim

        root = Path(maxim.__file__).parent / "_data" / "components"
        assert root.exists(), f"Bundled components dir missing: {root}"

        offenders: list[str] = []
        for yaml_path in sorted(root.rglob("*.yaml")):
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            ent = data.get("entity", data) if isinstance(data, dict) else None
            if not isinstance(ent, dict) or "name" not in ent:
                continue
            try:
                _parse_entity(ent)
            except ConfigurationError as exc:
                offenders.append(f"{yaml_path.name}: {exc}")
        assert not offenders, (
            "Bundled components raise C4 ConfigurationError. "
            "Either add sensors or set `abstract: true`. Offenders:\n  " + "\n  ".join(offenders)
        )

    def test_abstract_marker_survives_to_dict_from_dict_roundtrip(self):
        """Persisted entities must reload with the same abstract flag.

        BodySnapshot serialization goes through Entity.to_dict / from_dict.
        If `abstract` were dropped on the way out, every reload would
        silently re-introduce the C4 hard-error surface — exactly the silent
        regression `_format_version` discipline exists to prevent.
        """
        from maxim.embodiment.sem import Entity

        spec = _entity_with_modulator(
            {
                "abstract": True,
                "affordances": {
                    "slash": {"params": {"target": "str"}, "description": "swing"},
                },
            }
        )
        entity = _parse_entity(spec)
        assert entity.modulators["test_mod"].abstract is True

        dumped = entity.to_dict()
        assert dumped["modulators"]["test_mod"]["abstract"] is True

        reloaded = Entity.from_dict(dumped)
        assert reloaded.modulators["test_mod"].abstract is True

    def test_non_abstract_modulator_emits_abstract_false_in_to_dict(self):
        """Symmetric serialization: every modulator dict carries an
        explicit `abstract` boolean so a reloaded entity is indistinguishable
        from the source. Asymmetric emission silently loses the False signal
        once the C4 invariant is a hard error."""
        spec = _entity_with_modulator(
            {
                "sensors": {
                    "integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
                },
                "affordances": {},
            }
        )
        entity = _parse_entity(spec)
        dumped = entity.to_dict()
        assert dumped["modulators"]["test_mod"]["abstract"] is False

    def test_error_fires_on_raw_specmodulator_construction(self):
        """The structural enforcement lives at the constructor, not the
        parser — so raw `SpecModulator(...)` calls (tests, foundry-generated
        specs that bypass `_parse_entity`, future programmatic builders)
        still surface the hard error. Mirrors the `build_executor(pain_bus=...)`
        precedent: forgetting the kwarg is loud, not silent.
        """
        from maxim.embodiment.spec import SpecModulator

        with pytest.raises(ConfigurationError, match=r"declares no sensors"):
            SpecModulator(
                _name="bare_mod",
                _entity_name="entity",
                _affordances={},
            )

    def test_raw_specmodulator_with_abstract_true_loads_cleanly(self):
        """`_abstract=True` opt-out works for raw construction too."""
        from maxim.embodiment.spec import SpecModulator

        mod = SpecModulator(
            _name="abstract_mod",
            _entity_name="entity",
            _affordances={},
            _abstract=True,
        )
        assert mod.abstract is True

    def test_sensors_null_in_yaml_raises(self):
        """`sensors: null` is legal YAML but semantically empty — must raise,
        not crash on `.items()`. Pre-fix the parser called `.items()` on `None`.
        """
        spec = _entity_with_modulator(
            {
                "sensors": None,
                "affordances": {"do_thing": {"params": {}, "description": "noop"}},
            }
        )
        with pytest.raises(ConfigurationError, match=r"declares no sensors"):
            _parse_entity(spec)

    def test_sensors_empty_list_in_yaml_raises(self):
        """`sensors: []` (list, not dict) — same normalisation."""
        spec = _entity_with_modulator(
            {
                "sensors": [],
                "affordances": {"do_thing": {"params": {}, "description": "noop"}},
            }
        )
        with pytest.raises(ConfigurationError, match=r"declares no sensors"):
            _parse_entity(spec)

    def test_sensor_bearing_modulator_round_trips_with_sensors_preserved(self):
        """Pre-C4 the to_dict/from_dict cycle dropped per-modulator sensors,
        so reloaded modulators became no-sensor stubs. Post-flip that gap
        now crashes on every load. The fix preserves sensor metadata so
        sensor-bearing modulators reload as themselves and stay live.
        """
        from maxim.embodiment.sem import Entity

        spec = _entity_with_modulator(
            {
                "sensors": {
                    "integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
                },
                "affordances": {
                    "do_thing": {"params": {}, "description": "noop"},
                },
            }
        )
        entity = _parse_entity(spec)
        dumped = entity.to_dict()

        # Sensor metadata survives the dump.
        assert "integrity" in dumped["modulators"]["test_mod"]["sensors"]

        # And the reload doesn't trip the C4 error.
        reloaded = Entity.from_dict(dumped)
        mod = reloaded.modulators["test_mod"]
        assert mod.sensors == {"integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0}}
        assert mod.abstract is False

    def test_legacy_pre_c4_snapshot_loads_without_error(self):
        """Snapshots written before C4 had neither `sensors` nor `abstract`
        on the modulator dict. Treat them as abstract=True on load so
        existing user state doesn't crash — there's no information
        in the dict to do better than the safe default.
        """
        from maxim.embodiment.sem import Entity

        legacy_dict = {
            "name": "legacy_entity",
            "entity_type": "test",
            "modulators": {
                "combat": {
                    "name": "combat",
                    "affordances": {
                        "slash": {"description": "swing", "timeout": 30.0},
                    },
                    # NB: no `sensors`, no `abstract` — legacy shape.
                },
            },
        }
        reloaded = Entity.from_dict(legacy_dict)
        assert reloaded.modulators["combat"].abstract is True

    def test_error_message_links_to_migration_doc(self):
        """An error a user will hit in 2027 needs an entry point to the
        design rationale. Link to the v1_refinement plan §C4."""
        spec = _entity_with_modulator({"affordances": {}})
        with pytest.raises(ConfigurationError) as excinfo:
            _parse_entity(spec)
        msg = str(excinfo.value)
        assert "v1_refinement" in msg or "§C4" in msg
