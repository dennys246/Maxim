#!/usr/bin/env python3
"""Body-YAML rest-at-neutral lint — the L11 range principle, mechanized (H1 of the 2026-09-16
set-point-aware-neutral design review, `docs/experiments/rationale/setpoint-neutral/`).

The A4-gained world channel weights each sensor by its distance from the range MIDPOINT: a sensor
resting at the midpoint is silent, one resting at an extreme is a full-weight CONSTANT in every
situation and dilutes whatever moved. So every gained sensor's range must be declared so the value
the world actually RESTS at sits at the midpoint. That rule lived in YAML comments
(``# rest (20 hp, full) = midpoint``) and was broken once without anyone noticing until a live run:
Exp 60 gate (ii) FAILED at cos 0.8502 because ``saturation`` declared ``[0, 10]`` (rest 5) while the
game rests at 10 (#725 → #726). A declared rest the world never visits is a constant, not a neutral.

What this lint makes checkable (DECLARATIONS only — whether the declared rest is where the world
actually rests is a MEASUREMENT, the apparatus check's job, e.g. ``exp60_water_check.py``; this lint
makes the author's claim explicit so that check has something to compare against):

1. Every GAINED sensor (``modality:`` in ``GAINED_MODALITIES``) with a ``range:`` declares its rest —
   an explicit ``rest: <value>``, or, for a homeostatic drive, ``drive.set_point`` (the drive's own
   declared rest; ABSENT defaults to 0.0 exactly as ``spec.py::_parse_drive_spec`` does, so an
   undeclared set-point is linted as the rest-at-0 it really is). A gained ranged sensor with neither
   FAILS: the author has not said where rest is.
2. The declared rest equals the range midpoint (relative tolerance ``REL_TOL`` of the span). A
   homeostatic ``set_point`` off-midpoint FAILS. ONE rest source per sensor: a homeostatic sensor that
   ALSO carries ``rest:`` FAILS (two declarations of one number, reconciled by a lint, is the
   redundancy the set-point review's bio-faithful lens warned against — SF-4).
3. ``rest: null`` is the explicit "this sensor has NO rest value" declaration (cyclic ``time_of_day``,
   place-dependent ``light_level``): accepted, PRINTED every run as the roster of known constant-mass
   contributors, and the SHIPPED roster is pinned by the unit test (``test_shipped_components_are_clean``)
   so a new ``null`` is a visible test edit, not a free pass. Not allowed on a homeostatic sensor
   (rule 2).
4. A gained sensor WITHOUT a range FAILS: the range principle cannot be applied, and the range-blind
   map's neutral is raw 0.5 by accident.

Scope: every ``*.yaml`` under ``src/maxim/_data/components/`` (all categories — the encoder gains by
MODALITY, not by component category), with ``component.extends:`` chains RESOLVED using the
registry's ``deep_merge`` semantics (dicts merge recursively, everything else replaces) so a child
that overrides only ``range:`` on an inherited gained sensor is linted against the rest it inherits —
the Exp 60 shape verbatim. Archetype-synthesized sensors are the loader's business, not a
declaration. ``initial:`` is deliberately NOT read as a rest: it is the boot value the vital-metrics
seed uses (``spec.py``), and it was exactly the field that was wrong for ``saturation``. An empty or
missing tree, or a tree with ZERO gained sensors, FAILS — an absent check must not read as a pass.

Ungained modalities (interoception drives, audio) are out of scope — their contribution is not
gain-weighted, so their neutral is not load-bearing.

Naming: if a RUNTIME set-point declaration is ever built (the deferred primitive's revival path in
``docs/plans/setpoint_aware_neutral.md``), it MUST read this same ``rest:`` key — not a new key — so
the lint's declaration and the encoder's neutral cannot be two concepts.

Exits: 0 clean; 1 violations (stderr). Positive control: ``tests/unit/test_lint_body_rest_neutral.py``
(which also pins ``GAINED_MODALITIES`` to ``SensorEncoderConfig().gain_modalities`` so the two cannot
drift).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPONENTS_DIR = REPO_ROOT / "src" / "maxim" / "_data" / "components"

# Mirrors `maxim.similarity.encoder.SensorEncoderConfig.gain_modalities` (A4). Kept literal so the
# lint has no runtime import (the CI lint job installs only ruff + pyyaml); the unit test asserts the
# two sets are equal.
GAINED_MODALITIES: frozenset[str] = frozenset({"world"})
REL_TOL = 1e-9
# `spec.py::_parse_drive_spec`: `set_point=float(drive_data.get("set_point", 0.0))`.
HOMEOSTATIC_DEFAULT_SET_POINT = 0.0


def _num(x: object) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return float(x)


def _deep_merge(base: dict, override: dict) -> dict:
    """`component_registry.deep_merge` semantics, duplicated so the lint needs no runtime import."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def resolved_sensors(path: Path, components_dir: Path, _seen: tuple[Path, ...] = ()) -> dict:
    """The file's ``entity.sensors`` with its ``component.extends:`` chain folded in (parent first)."""
    if path in _seen:
        raise ValueError(f"extends cycle: {' -> '.join(str(p) for p in (*_seen, path))}")
    data = _load(path)
    own = ((data.get("entity") or {}).get("sensors")) or {}
    own = own if isinstance(own, dict) else {}
    parent_ref = (data.get("component") or {}).get("extends")
    if not parent_ref:
        return own
    parent_path = components_dir / f"{parent_ref}.yaml"
    if not parent_path.is_file():
        raise FileNotFoundError(f"{path}: extends {parent_ref!r} but {parent_path} does not exist")
    return _deep_merge(resolved_sensors(parent_path, components_dir, (*_seen, path)), own)


def lint_component(path: Path, components_dir: Path) -> tuple[list[str], list[str], int]:
    """Return ``(failures, no_rest_roster, gained_sensor_count)`` for one component YAML."""
    fails: list[str] = []
    no_rest: list[str] = []
    gained = 0
    where = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    try:
        sensors = resolved_sensors(path, components_dir)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        return [f"{where}: {exc}"], no_rest, 0
    for name, spec in sensors.items():
        if not isinstance(spec, dict) or spec.get("modality") not in GAINED_MODALITIES:
            continue
        gained += 1
        tag = f"{where}::{name}"
        rng = spec.get("range")
        if rng is None:
            fails.append(f"{tag}: gained sensor declares no `range:` — the range principle cannot be applied")
            continue
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            fails.append(f"{tag}: `range:` must be `[lo, hi]`, got {rng!r}")
            continue
        lo, hi = _num(rng[0]), _num(rng[1])
        if lo is None or hi is None or hi <= lo:
            fails.append(f"{tag}: degenerate range {rng!r} (need finite numbers with lo < hi)")
            continue
        span = hi - lo
        mid = (lo + hi) / 2.0
        tol = REL_TOL * span
        drive = spec.get("drive") or {}
        set_point: float | None = None
        if isinstance(drive, dict) and drive.get("drift_mode") == "homeostatic":
            set_point = _num(drive.get("set_point", HOMEOSTATIC_DEFAULT_SET_POINT))
            if set_point is None:
                fails.append(f"{tag}: homeostatic set_point must be a finite number, got {drive.get('set_point')!r}")
                continue
        has_rest = "rest" in spec
        rest = spec.get("rest")
        if set_point is not None and has_rest:
            fails.append(
                f"{tag}: one rest source per sensor — a homeostatic drive declares its rest as drive.set_point "
                f"({set_point:g}); drop `rest:` (two declarations of one number is the redundancy the "
                f"set-point review warned against, bio-faithful SF-4)"
            )
            continue
        if set_point is not None and abs(set_point - mid) > tol:
            fails.append(
                f"{tag}: drive.set_point {set_point:g} != range midpoint {mid:g} (range {lo:g}..{hi:g}) — a homeostatic "
                f"rest off the midpoint is a constant-mass contributor under the A4 gain; re-declare the range so the "
                f"set-point is the midpoint (the `saturation` lesson, Exp 60 #726)"
            )
        if not has_rest and set_point is None:
            fails.append(
                f"{tag}: declares no rest — add `rest: {mid:g}` (the range midpoint) or `rest: null` "
                f"(a documented no-rest sensor: constant mass wherever it sits)"
            )
            continue
        if has_rest and rest is None:
            no_rest.append(tag)
            continue
        if has_rest:
            rest_v = _num(rest)
            if rest_v is None:
                fails.append(f"{tag}: `rest:` must be a finite number or null, got {rest!r}")
                continue
            if abs(rest_v - mid) > tol:
                fails.append(
                    f"{tag}: rest {rest_v:g} != range midpoint {mid:g} (range {lo:g}..{hi:g}) — a resting sensor "
                    f"off the midpoint is a constant-mass contributor under the A4 gain; re-declare the range so "
                    f"the measured rest is the midpoint (the `saturation` lesson, Exp 60 #726)"
                )
    return fails, no_rest, gained


def lint_components(components_dir: Path = COMPONENTS_DIR) -> tuple[list[str], list[str], int]:
    fails: list[str] = []
    no_rest: list[str] = []
    gained = 0
    if not components_dir.is_dir():
        return [f"components dir not found: {components_dir} — an absent tree must not read as a pass"], no_rest, 0
    for path in sorted(components_dir.rglob("*.yaml")):
        f, n, g = lint_component(path, components_dir)
        fails.extend(f)
        no_rest.extend(n)
        gained += g
    if gained == 0:
        fails.append(f"no gained sensor found under {components_dir} — an absent check must not read as a pass")
    return fails, no_rest, gained


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    components_dir = Path(args[0]) if args else COMPONENTS_DIR
    fails, no_rest, gained = lint_components(components_dir)
    print(
        f"body rest-at-neutral lint: {gained} gained sensor(s) read; {len(no_rest)} declared no-rest "
        f"(constant mass wherever they sit):"
    )
    for tag in no_rest:
        print(f"  {tag}")
    if fails:
        print(
            f"FAIL: {len(fails)} problem(s) — declared rest missing, off the range midpoint, or tree unreadable:",
            file=sys.stderr,
        )
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("PASS: every gained ranged sensor declares a rest at its range midpoint (or an explicit no-rest).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
