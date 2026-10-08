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

import pytest

from maxim.embodiment.body import Embodiment
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.embodiment.sem import Entity, FailureMode, FailureTrigger


def _body(name: str) -> Embodiment:
    return Embodiment(ComponentRegistry().instantiate(name))


def _owner(root: Entity, mod_name: str) -> Entity:
    return next(ent for ent in root.walk() if mod_name in ent.modulators)


def _fired(events) -> set[str]:
    return {e.failure_name for e in events}


# -- the shadow ------------------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="#1124: a dotted top-level key shadows the modulator's real value")
def test_a_real_sub_sensor_beats_a_dotted_top_level_key():
    body = _body("bodies/infant_humanoid")
    ent = _owner(body.root, "arms")
    ent.modulators["arms"].vital_metrics["thermal"] = 0.9  # past the 0.5 comfort band
    ent.vital_metrics["arms.thermal"] = 0.0  # a pre-#874 orphan at neutral
    assert "drive:arms.thermal:discomfort" in _fired(body.evaluate_failures())


@pytest.mark.xfail(strict=True, reason="#1124: evaluate_failures writes <mod>.integrity onto vital_metrics")
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


@pytest.mark.xfail(strict=True, reason="#1124: to_dict/from_dict drop the modulators' values and integrity spec")
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


@pytest.mark.xfail(strict=True, reason="#1124: a reloaded body's integrity is frozen at its saved value")
def test_a_reloaded_body_still_takes_damage_into_its_triggers():
    saved = _body("bodies/base_humanoid")
    saved.evaluate_failures()  # as a running body is, before anything saves it
    body = Embodiment(Entity.from_dict(saved.root.to_dict()))
    assert "concussion" not in _fired(body.evaluate_failures())
    _owner(body.root, "head").modulators["head"].apply_damage(2.0, "blunt")
    assert "concussion" in _fired(body.evaluate_failures())


@pytest.mark.xfail(strict=True, reason="#1124: a reloaded body reads a never-written <mod>.integrity as 0.0")
def test_an_undamaged_reloaded_body_fires_no_integrity_trigger():
    """Saved before its first evaluation, the file holds no ``head.integrity``; the reloaded modulators are
    empty, so nothing re-derives it and the missing key read as 0.0: concussion and crippled, unhurt."""
    body = Embodiment(Entity.from_dict(_body("bodies/base_humanoid").root.to_dict()))
    assert not _fired(body.evaluate_failures()) & {"concussion", "crippled"}


@pytest.mark.xfail(strict=True, reason="#1124: a legacy file's dotted top-level keys load verbatim")
def test_a_legacy_file_drops_dotted_top_level_keys_with_a_warning(caplog):
    data = _body("bodies/base_humanoid").root.to_dict()
    data.setdefault("vital_metrics", {}).update({"head.integrity": 0.05, "arms.thermal": 0.0})
    for mod in data.get("modulators", {}).values():
        mod.pop("values", None)  # a file written before 1.1 carries no modulator values
    with caplog.at_level(logging.WARNING, logger="maxim.embodiment.sem"):
        ent = Entity.from_dict(data)
    assert not [k for k in ent.vital_metrics if "." in k]
    assert any("head.integrity" in r.getMessage() for r in caplog.records)
    # The modulator starts from its spec, as a fresh body does.
    assert _owner(ent, "head").modulators["head"].vital_metrics == {"awareness": 0.9, "skull_integrity": 1.0}


@pytest.mark.xfail(strict=True, reason="#1124: the Entity file format does not change yet")
def test_a_saved_entity_carries_format_1_1(tmp_path):
    body = _body("bodies/base_humanoid")
    path = tmp_path / "entity.json"
    body.root.save(str(path))
    assert json.loads(path.read_text())["_format_version"] == "1.1"
    assert Entity.load(str(path)).name == body.root.name


# -- a trigger with no reading ---------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="#1124: a missing trigger field reads as 0.0")
def test_a_trigger_with_no_reading_does_not_fire():
    fm = FailureMode(name="phantom", triggers=[FailureTrigger(field="nowhere", op="<", value=0.5)])
    assert fm.evaluate({}) is False


@pytest.mark.xfail(strict=True, reason="#1124: a missing recovery field reads as 0.0 and clears")
def test_a_missing_recovery_field_does_not_clear_an_active_failure():
    fm = FailureMode(
        name="held",
        triggers=[FailureTrigger(field="x", op=">", value=0.5)],
        persistent=True,
        recovery_condition=FailureTrigger(field="x", op="<", value=0.2),
        active=True,
    )
    assert fm.evaluate({}) is True


@pytest.mark.xfail(strict=True, reason="#1124: a missing trigger field is silent")
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


def test_visible_sensors_still_report_component_integrity_at_every_depth():
    from maxim.embodiment.resolution import get_visible_sensors

    body = _body("bodies/base_humanoid")
    body.evaluate_failures()
    ent = _owner(body.root, "head")
    for depth in (2, 3):
        assert "head.integrity" in get_visible_sensors(ent, depth=depth)
