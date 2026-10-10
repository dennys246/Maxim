"""``tick_vital_drift`` rebuilt on ``sem.drift_step`` writes exactly what it wrote before (grounding GL2a, G14).

GL2a factors the drive-spec drift arithmetic out of ``Embodiment.tick_vital_drift`` into the pure
``embodiment/sem.py::drift_step``, so the body can report the drift it APPLIED (clamp included) to an
outcome window. That must not move a single drive value: this pins the rebuilt method against a
verbatim copy of the pre-GL2a method body over a value x dt grid on EVERY shipped component with a
drive spec, by exact float equality.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any

import pytest

from maxim.embodiment.body import Embodiment
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.embodiment.sem import EntropicDriveSpec, HomeostaticDriveSpec

DTS = (0.0, 1e-4, 0.5, 1.0, 7.3, 20.0, 600.0)
FRACTIONS = (0.0, 0.1, 0.37, 0.5, 0.9, 1.0)


def _legacy_tick_vital_drift(emb: Embodiment, dt: float) -> None:
    """``Embodiment.tick_vital_drift`` as it was on main at d9e89f5a, verbatim (the reference)."""
    rate = emb.config.vital_drift_rate
    for ent in emb.root.walk():
        for ds_name, ds in ent.drive_specs.items():
            if "." in ds_name:
                mod_name, sensor_name = ds_name.split(".", 1)
                mod = ent.modulators.get(mod_name)
                if mod is None or not hasattr(mod, "vital_metrics"):
                    continue
                current = mod.vital_metrics.get(sensor_name)
                if current is None:
                    continue
                if isinstance(ds, HomeostaticDriveSpec):
                    delta = ds.set_point - current
                    step = min(abs(delta), ds.drift_rate * dt)
                    mod.vital_metrics[sensor_name] = current + (step if delta > 0 else -step)
                elif isinstance(ds, EntropicDriveSpec):
                    if ds.drift_direction == "up":
                        mod.vital_metrics[sensor_name] = min(1.0, current + ds.drift_rate * dt)
                    else:
                        mod.vital_metrics[sensor_name] = max(0.0, current - ds.drift_rate * dt)
            else:
                current = ent.vital_metrics.get(ds_name)
                if current is None:
                    continue
                if isinstance(ds, HomeostaticDriveSpec):
                    delta = ds.set_point - current
                    step = min(abs(delta), ds.drift_rate * dt)
                    ent.vital_metrics[ds_name] = current + (step if delta > 0 else -step)
                elif isinstance(ds, EntropicDriveSpec):
                    if ds.drift_direction == "up":
                        ent.vital_metrics[ds_name] = min(1.0, current + ds.drift_rate * dt)
                    else:
                        ent.vital_metrics[ds_name] = max(0.0, current - ds.drift_rate * dt)
        driven_sensors = set(ent.drive_specs.keys())
        for vname in list(ent.vital_metrics.keys()):
            if vname in driven_sensors:
                continue
            if vname in ("fatigue", "strain", "exhaustion"):
                ent.vital_metrics[vname] = min(1.0, ent.vital_metrics[vname] + rate * dt)
            elif vname in ("durability", "sharpness"):
                ent.vital_metrics[vname] = max(0.0, ent.vital_metrics[vname] - rate * dt)


def _state(emb: Embodiment) -> list[Any]:
    out = []
    for ent in emb.root.walk():
        out.append((ent.full_path, dict(ent.vital_metrics)))
        for name, mod in sorted(ent.modulators.items()):
            metrics = getattr(mod, "vital_metrics", None)
            if metrics is not None:
                out.append((f"{ent.full_path}.{name}", dict(metrics)))
    return out


def _set_all_drives(root: Any, fraction: float) -> None:
    """Every drive value to ``lo + fraction * (hi - lo)`` of its declared range (else [0, 1])."""
    from maxim.embodiment.sem import _resolve_sensor_slot

    for ent in root.walk():
        for name in ent.drive_specs:
            slot = _resolve_sensor_slot(ent, name)
            if slot is not None:
                metrics, key, lo, hi = slot
                metrics[key] = lo + fraction * (hi - lo)


def _driven_refs() -> list[str]:
    registry = ComponentRegistry()
    refs = []
    for ref in registry.list_refs():
        try:
            entity = registry.instantiate(ref)
        except Exception:  # noqa: BLE001 - a component that cannot instantiate has no drift to pin
            continue
        if any(ent.drive_specs for ent in entity.walk()):
            refs.append(ref)
    return sorted(refs)


DRIVEN = _driven_refs()


def test_the_grid_covers_the_shipped_drive_bodies() -> None:
    """Known answer: the parametrization is not vacuous."""
    assert {"bodies/infant_humanoid", "bodies/infant_humanoid_chilled", "bodies/minecraft_player"} <= set(DRIVEN)


@pytest.mark.parametrize("ref", DRIVEN)
def test_tick_vital_drift_is_byte_identical_to_the_pre_gl2a_body(ref: str) -> None:
    registry = ComponentRegistry()
    for fraction, dt in itertools.product(FRACTIONS, DTS):
        root = registry.instantiate(ref)
        _set_all_drives(root, fraction)
        new, old = Embodiment(root), Embodiment(copy.deepcopy(root))
        assert _state(new) == _state(old)
        applied = new.tick_vital_drift(dt)
        _legacy_tick_vital_drift(old, dt)
        assert _state(new) == _state(old), (ref, fraction, dt)
        # And the drift it reports is exactly what it wrote.
        root2 = registry.instantiate(ref)
        _set_all_drives(root2, fraction)
        before = Embodiment(root2)
        snapshot = dict(_state(before))
        for path, drives in applied.items():
            for name, value in drives.items():
                where, key = (f"{path}.{name.split('.', 1)[0]}", name.split(".", 1)[1]) if "." in name else (path, name)
                assert dict(_state(new))[where][key] - snapshot[where][key] == value
