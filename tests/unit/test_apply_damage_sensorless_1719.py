"""#1719: damage aimed at a modulator with no sub-sensors is refused, never a silent ``1.0``.

``SpecModulator.apply_damage`` used to return ``1.0`` (full integrity) on a modulator with no sub-sensors,
with no log, no error and no state change. ``DamageComponentTool`` refuses such parts before calling
(#873), so the no-op was dead on that path, but a host that calls ``apply_damage`` directly (a flat body
whose modulators carry no sub-sensors) saw the damage vanish. How a partless body should take damage stays
open in ``docs/plans/deferred/reflex_layering.md`` §3; until then the call refuses loudly (owner, 2026-11-21).
"""

from __future__ import annotations

import copy

import pytest

from maxim.embodiment.component_registry import ComponentRegistry
from maxim.exceptions import DamageRefused, MaximError


def _wolf_part(name: str):
    root = ComponentRegistry().instantiate("creatures/wolf")
    mod = root.modulators[name]
    assert not mod.vital_metrics, f"fixture premise: wolf {name!r} has no sub-sensors"
    return root, mod


@pytest.mark.parametrize("damage_type", [None, "fire"])
def test_damage_to_a_sensorless_modulator_is_refused(damage_type):
    root, combat = _wolf_part("combat")
    before = copy.deepcopy(root.vital_metrics)
    with pytest.raises(DamageRefused) as info:
        combat.apply_damage(0.4, damage_type)
    err = info.value
    assert err.modulator == "combat"
    assert "combat" in str(err)
    # Nothing changed: no sub-sensor appeared, the root is untouched, integrity reads as before.
    assert combat.vital_metrics == {}
    assert root.vital_metrics == before
    assert combat.compute_integrity() == 1.0


def test_the_refusal_is_a_maxim_error_and_a_value_error():
    """Bad input, like ``MotionRefused``: a host's ``except ValueError`` sees it, and so does ``except MaximError``."""
    _, movement = _wolf_part("movement")
    with pytest.raises(ValueError):
        movement.apply_damage(0.1)
    with pytest.raises(MaximError):
        movement.apply_damage(0.1)


def test_the_refusal_survives_pickle_and_copy():
    """Its init is keyword-only, so ``__reduce__`` must rebuild it (a worker process or ``copy`` keeps ``modulator``)."""
    import pickle

    err = DamageRefused("no sub-sensors", modulator="combat")
    for clone in (pickle.loads(pickle.dumps(err)), copy.copy(err), copy.deepcopy(err)):
        assert isinstance(clone, DamageRefused)
        assert clone.modulator == "combat"
        assert str(clone) == "no sub-sensors"


def test_a_part_with_sub_sensors_still_takes_damage():
    root = ComponentRegistry().instantiate("bodies/base_humanoid")
    torso = root.modulators["torso"]
    assert torso.vital_metrics
    assert torso.apply_damage(0.3) < 1.0


def test_capability_only_integrity_is_unchanged():
    """``compute_integrity`` keeps its backward-compatible 1.0 for capability-only parts; only damage refuses."""
    _, combat = _wolf_part("combat")
    assert combat.compute_integrity() == 1.0
