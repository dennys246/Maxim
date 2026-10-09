"""#1124: a dotted top-level key shadows a modulator's real sub-sensor, and a reload freezes component state.

``Embodiment.evaluate_failures`` read an entity's top-level ``vital_metrics`` before its modulators' sub-sensors,
and wrote each derived ``<mod>.integrity`` there on every call. ``Entity.to_dict`` saved those keys but not the
modulators' own values, and ``from_dict`` rebuilt every modulator empty (no integrity function, no damage
affinities). A reloaded body's integrity and derived health stayed at their saved values forever. A trigger
whose field had no reading evaluated it as 0.0, so a ``<`` trigger fired on a missing key.

Owner decisions 2026-10-07: derived integrity is computed from the modulators and never stored on
``vital_metrics``; save/load carries modulator state (Entity JSON format 1.1); a dotted top-level key is dropped
on load with a WARNING and loses to a real sub-sensor at read; a trigger with no reading does not fire and warns
once. Tests run on the shipped bodies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from maxim.embodiment.body import Embodiment
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.embodiment.sem import Entity, FailureMode, FailureTrigger


FORMAT_1_0_DRAGON = Path(__file__).resolve().parents[1] / "fixtures" / "entity_format_1_0" / "dragon.json"


def _body(name: str) -> Embodiment:
    return Embodiment(ComponentRegistry().instantiate(name))


def _owner(root: Entity, mod_name: str) -> Entity:
    return next(ent for ent in root.walk() if mod_name in ent.modulators)


def _fired(events) -> set[str]:
    return {e.failure_name for e in events}


# -- the shadow ------------------------------------------------------------------------------------


def test_a_real_sub_sensor_beats_a_dotted_top_level_key():
    body = _body("bodies/infant_humanoid")
    ent = _owner(body.root, "arms")
    ent.modulators["arms"].vital_metrics["thermal"] = 0.9  # past the 0.5 comfort band
    ent.vital_metrics["arms.thermal"] = 0.0  # a pre-#874 orphan at neutral
    assert "drive:arms.thermal:discomfort" in _fired(body.evaluate_failures())


def test_a_shadowed_dotted_key_is_reported_once(caplog):
    body = _body("bodies/infant_humanoid")
    _owner(body.root, "arms").vital_metrics["arms.thermal"] = 0.0
    with caplog.at_level(logging.WARNING, logger="maxim.embodiment.body"):
        body.evaluate_failures()
        body.evaluate_failures()
    assert len([r for r in caplog.records if "arms.thermal" in r.getMessage()]) == 1


def test_evaluating_writes_no_derived_integrity_onto_vital_metrics():
    body = _body("bodies/base_humanoid")
    body.evaluate_failures()
    assert not [k for ent in body.root.walk() for k in ent.vital_metrics if "." in k]


def test_integrity_triggers_still_fire_from_the_modulators():
    body = _body("bodies/base_humanoid")
    head = _owner(body.root, "head").modulators["head"]
    head.vital_metrics.update(awareness=0.1, skull_integrity=0.1)  # integrity 0.1 < 0.2
    assert "concussion" in _fired(body.evaluate_failures())


# -- save and load ---------------------------------------------------------------------------------


def test_a_reload_keeps_the_modulators_state_and_spec():
    body = _body("bodies/base_humanoid")
    head = _owner(body.root, "head").modulators["head"]
    head.apply_damage(0.5, "slash")
    before = dict(head.vital_metrics)
    reloaded = Entity.from_dict(json.loads(json.dumps(body.root.to_dict())))
    again = _owner(reloaded, "head").modulators["head"]
    assert again.vital_metrics == pytest.approx(before)
    assert again.integrity_fn == head.integrity_fn
    assert again.damage_affinities == head.damage_affinities
    assert again.compute_integrity() == pytest.approx(head.compute_integrity())


def test_a_reload_keeps_a_non_default_integrity_function_and_its_damage_routing():
    """The dragon's torso aggregates by ``min`` and routes ``fire`` to its armor alone."""
    reloaded = Entity.from_dict(json.loads(json.dumps(_body("creatures/dragon").root.to_dict())))
    owner = _owner(reloaded, "torso")
    torso = owner.modulators["torso"]
    assert torso.integrity_fn == "min"
    torso.apply_damage(0.5, "fire")
    assert torso.vital_metrics == pytest.approx({"armor_integrity": 0.45, "organ_health": 1.0})
    assert torso.compute_integrity() == pytest.approx(0.45)
    assert torso._entity_ref is owner  # D59's back-reference (its ``requires`` consumer does not round-trip: #1159)


def test_a_reloaded_body_still_takes_damage_into_its_triggers():
    saved = _body("bodies/base_humanoid")
    saved.evaluate_failures()  # as a running body is, before anything saves it
    body = Embodiment(Entity.from_dict(saved.root.to_dict()))
    assert "concussion" not in _fired(body.evaluate_failures())
    _owner(body.root, "head").modulators["head"].apply_damage(2.0, "blunt")
    assert "concussion" in _fired(body.evaluate_failures())


def test_an_undamaged_reloaded_body_fires_no_integrity_trigger():
    """Saved before its first evaluation, the file holds no ``head.integrity``; the reloaded modulators are
    empty, so nothing re-derives it and the missing key read as 0.0: concussion and crippled, unhurt."""
    body = Embodiment(Entity.from_dict(_body("bodies/base_humanoid").root.to_dict()))
    assert not _fired(body.evaluate_failures()) & {"concussion", "crippled"}


def test_a_format_1_0_file_keeps_its_saved_damage(caplog):
    """``tests/fixtures/entity_format_1_0/dragon.json`` was written by the 1.0 writer (origin/main at e2c34130):
    head and torso damaged (torso integrity ``min`` = 0.45), health 0.696, no modulator values, integrity function
    or affinities, and a dotted orphan key. Its integrity migrates onto the sub-sensors (the torso, whose ``min`` was
    never saved, loads as ``weighted_mean`` and still reproduces 0.45); the orphan is dropped."""
    data = json.loads(FORMAT_1_0_DRAGON.read_text())
    assert data["_format_version"] == "1.0" and "values" not in data["modulators"]["torso"]
    with caplog.at_level(logging.WARNING, logger="maxim.embodiment.sem"):
        body = Embodiment(Entity.from_dict(data))
    assert not [k for ent in body.root.walk() for k in ent.vital_metrics if "." in k]
    integrities = _owner(body.root, "head").component_integrities()
    assert integrities["head"] == pytest.approx(0.664)
    assert integrities["torso"] == pytest.approx(0.45)
    body.evaluate_failures()
    assert body.root.vital_metrics["health"] == pytest.approx(data["vital_metrics"]["health"])
    [warning] = [r.getMessage() for r in caplog.records if "#1124" in r.getMessage()]
    assert "head.integrity" in warning and "torso.integrity" in warning and "head.awareness_orphan" in warning


def _format_1_0_infant(arms_integrity: float) -> dict:
    data = _body("bodies/infant_humanoid").root.to_dict()
    for mod in data["modulators"].values():
        for key in ("values", "integrity", "damage_affinities"):
            mod.pop(key, None)
    data["_format_version"] = "1.0"
    data["vital_metrics"]["arms.integrity"] = arms_integrity
    return data


def test_a_format_1_0_migration_leaves_a_drive_sub_sensor_alone():
    """``arms.thermal`` is a drive: setting it to a saved integrity would be a burn. Given weight 1 here, so only
    the drive exclusion keeps it."""
    data = _format_1_0_infant(0.3)
    data["modulators"]["arms"]["sensors"]["thermal"]["weight"] = 1.0
    arms = _owner(Entity.from_dict(data), "arms").modulators["arms"]
    assert arms.vital_metrics["thermal"] == 0.0


def test_a_format_1_0_migration_leaves_an_unweighted_sub_sensor_alone():
    """``arms.texture`` (weight 0, no drive) does not count toward integrity, so it is not damage either."""
    data = _format_1_0_infant(0.3)
    start = _owner(Entity.from_dict(_format_1_0_infant(1.0)), "arms").modulators["arms"].vital_metrics["texture"]
    arms = _owner(Entity.from_dict(data), "arms").modulators["arms"]
    assert arms.vital_metrics["texture"] == start
    assert arms.compute_integrity() == pytest.approx(0.3)


@pytest.mark.parametrize("bad", ["bad", None, float("nan"), float("inf")])
def test_a_format_1_0_integrity_that_is_not_a_finite_number_is_dropped_and_reported(bad, caplog):
    start = _owner(Entity.from_dict(_format_1_0_infant(1.0)), "arms").modulators["arms"].vital_metrics
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="maxim.embodiment.sem"):
        arms = _owner(Entity.from_dict(_format_1_0_infant(bad)), "arms").modulators["arms"]
    assert arms.vital_metrics == start
    [warning] = [r.getMessage() for r in caplog.records if "#1124" in r.getMessage()]
    assert "dropped ['arms.integrity']" in warning and "not a finite number" in warning


def test_a_save_drops_a_dotted_vital_metrics_key_with_a_warning(caplog):
    body = _body("bodies/base_humanoid")
    body.root.vital_metrics["gps.lat"] = 1.0
    with caplog.at_level(logging.WARNING, logger="maxim.embodiment.sem"):
        saved = body.root.to_dict()
    assert "gps.lat" not in saved["vital_metrics"]
    assert any("gps.lat" in r.getMessage() for r in caplog.records)


def test_parsing_warns_about_a_trigger_nothing_produces(tmp_path, caplog):
    from maxim.embodiment.spec import load_spec

    spec = tmp_path / "body.yaml"
    spec.write_text(
        "body:\n"
        "  name: lamp\n"
        "  entity_type: object\n"
        "  sensors:\n"
        "    heat: {unit: ratio, range: [0, 1], initial: 0.2}\n"
        "  failure_modes:\n"
        "    - name: fine\n"
        "      trigger: {field: heat, op: '>', value: 0.9, pain: 0.2}\n"
        "    - name: broken\n"
        "      trigger: {field: bulb.integrity, op: '<', value: 0.3, pain: 0.2}\n"
    )
    with caplog.at_level(logging.WARNING, logger="maxim.embodiment.spec"):
        load_spec(spec)
    messages = [r.getMessage() for r in caplog.records if "NEVER fire" in r.getMessage()]
    assert len(messages) == 1 and "bulb.integrity" in messages[0] and "lamp.broken" in messages[0]


def test_a_saved_entity_carries_format_1_1(tmp_path):
    body = _body("bodies/base_humanoid")
    path = tmp_path / "entity.json"
    body.root.save(str(path))
    assert json.loads(path.read_text())["_format_version"] == "1.1"
    assert Entity.load(str(path)).name == body.root.name


# -- a trigger with no reading ---------------------------------------------------------------------


def test_a_trigger_with_no_reading_does_not_fire():
    fm = FailureMode(name="phantom", triggers=[FailureTrigger(field="nowhere", op="<", value=0.5)])
    assert fm.evaluate({}) is False


def test_a_missing_recovery_field_does_not_clear_an_active_failure():
    fm = FailureMode(
        name="held",
        triggers=[FailureTrigger(field="x", op=">", value=0.5)],
        persistent=True,
        recovery_condition=FailureTrigger(field="x", op="<", value=0.2),
        active=True,
    )
    assert fm.evaluate({}) is True


def test_a_missing_trigger_field_warns_once(caplog):
    fm = FailureMode(name="phantom", triggers=[FailureTrigger(field="nowhere", op="<", value=0.5)])
    with caplog.at_level(logging.WARNING, logger="maxim.embodiment.sem"):
        fm.evaluate({})
        fm.evaluate({})
    assert len([r for r in caplog.records if "nowhere" in r.getMessage()]) == 1


# -- the readers that saw <mod>.integrity on vital_metrics keep seeing it --------------------------


def test_drive_telemetry_still_reports_component_integrity():
    from types import SimpleNamespace

    from maxim.simulation.substrate_telemetry import _drive_snapshot

    body = _body("bodies/base_humanoid")
    body.evaluate_failures()
    snap = _drive_snapshot(SimpleNamespace(embodiment=body))
    assert f"{body.root.name}.head.integrity" in snap["sensors"]


def test_drive_telemetry_reports_the_derived_integrity_over_a_stray_dotted_key():
    from types import SimpleNamespace

    from maxim.simulation.substrate_telemetry import _drive_snapshot

    body = _body("creatures/dragon")
    owner = _owner(body.root, "wing")
    owner.modulators["wing"].apply_damage(0.9)
    owner.vital_metrics["wing.integrity"] = 1.0
    owner.vital_metrics["wing.feathers"] = 0.2  # an orphan: no such sub-sensor
    snap = _drive_snapshot(SimpleNamespace(embodiment=body))
    assert snap["sensors"][f"{owner.name}.wing.integrity"] == pytest.approx(owner.component_integrities()["wing"])
    assert f"{owner.name}.wing.feathers" not in snap["sensors"]


def test_visible_sensors_still_report_component_integrity_at_every_depth():
    from maxim.embodiment.resolution import get_visible_sensors

    body = _body("bodies/base_humanoid")
    body.evaluate_failures()
    ent = _owner(body.root, "head")
    for depth in (2, 3):
        assert "head.integrity" in get_visible_sensors(ent, depth=depth)
