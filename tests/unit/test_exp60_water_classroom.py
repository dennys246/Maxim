"""Exp 60 water classroom — structural guards on the apparatus geometry + check evaluators.

The builder (``scripts/survival_world/setup_world.py::water_classroom_geometry``) is pure,
so the properties the design rests on are asserted offline: the dive target's head is IN
the water, the shore is DRY and above stone, the water column is WALLED on every side and
its only open face is the top (the reachable air escape), the Exp 58 clearance guard keeps
the persistent clustermob beyond the hostile horizon, and the live check's evaluators
classify synthetic dives/escapes the way the prereg gates say.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from survival_world import exp60_water_check as chk  # noqa: E402
from survival_world.setup_world import (  # noqa: E402
    WATER_DEPTH_DEFAULT,
    WATER_MAX_DIST_FROM_SPAWN,
    WATER_MIN_DIST_FROM_EXP58,
    exp58_clearance,
    spawn_clearance,
    water_anchor_record,
    water_classroom_commands,
    water_classroom_geometry,
    water_classroom_verifications,
)


def _inside(p: tuple[int, int, int], c: tuple[int, int, int, int, int, int]) -> bool:
    x, y, z = p
    x0, y0, z0, x1, y1, z1 = c
    return x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1


def _cells(c):
    x0, y0, z0, x1, y1, z1 = c
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                yield (x, y, z)


def _block_at(geom, p) -> str:
    """The block the ordered fills leave at p (later fills overwrite earlier ones)."""
    if _inside(p, geom["pool"]):
        return "water"
    if _inside(p, geom["chamber"]):
        return "air"
    if _inside(p, geom["shell"]):
        return "stone"
    return "world"


GEOM = water_classroom_geometry(100, -40)


class TestGeometry:
    def test_dive_target_head_is_in_water_and_feet_on_stone(self):
        bx, by, bz = GEOM["submerged"]
        assert _block_at(GEOM, (bx, by, bz)) == "water"
        assert _block_at(GEOM, GEOM["submerged_head"]) == "water"
        assert GEOM["submerged_head"] == (bx, by + 1, bz)
        assert _block_at(GEOM, GEOM["pool_floor"]) == "stone"
        assert GEOM["pool_floor"] == (bx, by - 1, bz)

    def test_shore_is_dry_air_over_stone(self):
        sx, sy, sz = GEOM["shore"]
        assert _block_at(GEOM, (sx, sy, sz)) == "air"
        assert _block_at(GEOM, (sx, sy + 1, sz)) == "air"
        assert _block_at(GEOM, GEOM["shore_floor"]) == "stone"
        assert _block_at(GEOM, GEOM["lip"]) == "stone"

    def test_water_column_is_walled_with_only_the_top_open(self):
        x0, y0, z0, x1, y1, z1 = GEOM["pool"]
        for p in _cells(GEOM["pool"]):
            x, y, z = p
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, 1, 0)):
                n = (x + dx, y + dy, z + dz)
                if _inside(n, GEOM["pool"]):
                    continue
                if dy == 1:
                    assert _block_at(GEOM, n) == "air", f"top of {p} must open to air, got {n}"
                else:
                    assert _block_at(GEOM, n) == "stone", f"side/floor of {p} must be stone at {n}"

    def test_surface_cell_is_air_directly_above_the_dive_target(self):
        ux, uy, uz = GEOM["surface"]
        bx, by, bz = GEOM["submerged"]
        assert (ux, uz) == (bx, bz)
        assert _block_at(GEOM, (ux, uy, uz)) == "air"
        assert _block_at(GEOM, (ux, uy - 1, uz)) == "water"
        # the head must rise depth-1 blocks: from submerged_head to the surface cell
        assert uy - GEOM["submerged_head"][1] == GEOM["depth"] - 1

    def test_shore_and_dive_altitudes_differ_by_more_than_the_probe_settle_tolerance(self):
        # l11_geometry_probe settles on y within 3 blocks; the two situations must be
        # distinguishable to it at the default depth.
        assert GEOM["shore"][1] - GEOM["submerged"][1] == WATER_DEPTH_DEFAULT > 3

    def test_forceload_covers_the_shell(self):
        fx0, fz0, fx1, fz1 = GEOM["forceload"]
        x0, _, z0, x1, _, z1 = GEOM["shell"]
        assert (fx0, fz0, fx1, fz1) == (x0, z0, x1, z1)

    @pytest.mark.parametrize("depth", [2, 13])
    def test_depth_bounds_refuse(self, depth):
        with pytest.raises(ValueError):
            water_classroom_geometry(0, 0, depth=depth)

    def test_depth_parameter_moves_the_floor_not_the_shore(self):
        g = water_classroom_geometry(0, 0, depth=8)
        assert g["shore"] == (-1, GEOM["shore"][1], 0)
        assert g["submerged"][1] == g["shore"][1] - 8


class TestCommandsAndVerifications:
    def test_forceload_precedes_every_fill_and_fills_are_ordered(self):
        cmds = water_classroom_commands(GEOM, "maxim")
        assert cmds[0].startswith("forceload add")
        fills = [c for c in cmds if c.startswith("fill")]
        assert [c.split()[-1] for c in fills] == ["minecraft:stone", "minecraft:air", "minecraft:water"]
        assert "gamerule doMobSpawning false" in cmds
        assert any(c.startswith("spawnpoint maxim ") for c in cmds)
        assert "scoreboard objectives add exp60_deaths deathCount" in cmds

    def test_verifications_are_block_tests_that_cover_the_design_claims(self):
        ver = water_classroom_verifications(GEOM)
        assert all(cmd.startswith("execute if block ") for cmd, _ in ver)
        proves = " | ".join(p for _, p in ver)
        for claim in ("head cell is water", "surface cell", "shore feet", "shore floor", "source"):
            assert claim in proves
        assert sum("level=0" in cmd for cmd, _ in ver) >= 3  # sources, not flowing water


class TestExp58Clearance:
    def _exp58(self, dx: float) -> dict:
        cx, _cy, cz = GEOM["submerged"]
        return {"anchor": [cx + dx, 40, cz], "dark": [cx + dx + 21, 28, cz]}

    def test_refuses_inside_the_horizon_and_accepts_at_the_margin(self):
        ok, d = exp58_clearance(GEOM, self._exp58(WATER_MIN_DIST_FROM_EXP58 - 1))
        assert not ok and d == pytest.approx(WATER_MIN_DIST_FROM_EXP58 - 1)
        ok, d = exp58_clearance(GEOM, self._exp58(WATER_MIN_DIST_FROM_EXP58))
        assert ok and d == pytest.approx(WATER_MIN_DIST_FROM_EXP58)

    def test_nearest_of_anchor_and_pit_counts(self):
        # pit (dark) closer than the anchor on the other side
        cx, _cy, cz = GEOM["submerged"]
        ok, d = exp58_clearance(GEOM, {"anchor": [cx - 200, 40, cz], "dark": [cx + 10, 28, cz]})
        assert not ok and d == pytest.approx(10)

    def test_no_exp58_record_means_nothing_to_clear(self):
        assert exp58_clearance(GEOM, None) == (True, float("inf"))


class TestSpawnClearance:
    def test_unknown_spawn_is_not_checked_not_cleared(self):
        assert spawn_clearance(GEOM, None) == (True, None)

    def test_three_d_distance_bound(self):
        bx, by, bz = GEOM["submerged"]
        ok, d = spawn_clearance(GEOM, (bx + WATER_MAX_DIST_FROM_SPAWN, by, bz))
        assert ok and d == pytest.approx(WATER_MAX_DIST_FROM_SPAWN)
        # the sensor is a 3D distance: a vertical offset counts
        ok, d = spawn_clearance(GEOM, (bx + WATER_MAX_DIST_FROM_SPAWN, by + 30, bz))
        assert not ok and d > WATER_MAX_DIST_FROM_SPAWN


class TestAnchorRecord:
    def test_clearances_default_to_not_checked(self):
        rec = water_anchor_record(GEOM)
        assert rec["exp58_clearance_blocks"] is None and rec["spawn_clearance_blocks"] is None
        rec = water_anchor_record(GEOM, exp58_clearance_blocks=80.0, spawn_clearance_blocks=42.5)
        assert rec["exp58_clearance_blocks"] == 80.0 and rec["spawn_clearance_blocks"] == 42.5

    def test_record_carries_what_the_check_probe_and_harness_read(self):
        rec = water_anchor_record(GEOM)
        assert rec["_format_version"] == "1.0"
        assert rec["shore"] == list(GEOM["shore"]) and rec["submerged"] == list(GEOM["submerged"])
        assert rec["surface_y"] == GEOM["shore_y"] and rec["depth"] == GEOM["depth"]
        assert list(rec["probe_situations"]) == ["shore", "submerged"]  # baseline FIRST
        assert rec["probe_settle"] == {"shore": {"is_in_water": 0, "on_ground": 1}, "submerged": {"is_in_water": 1}}
        assert rec["probe_rescue"] == {"submerged": "shore"}
        assert rec["deaths_objective"] == "exp60_deaths"


def _dive(t_max=16.5, *, floor=35.0, in_water_from=0.25, float_at=None, damage_at=15.5, oxygen_rate=20 / 15):
    out = []
    t = 0.0
    while t <= t_max:
        wet = t >= in_water_from and (float_at is None or t < float_at)
        oxy = max(0.0, 20.0 - oxygen_rate * max(0.0, t - in_water_from)) if wet else 20.0
        out.append(
            {
                "t": round(t, 3),
                "oxygen": round(oxy),
                "health": 18.0 if t >= damage_at else 20.0,
                "y": floor if wet else floor + 5,
                "in_water": wet,
            }
        )
        if t >= damage_at:
            break
        t += 0.25
    return out


class TestEvaluateDive:
    def test_nominal_dive_passes_and_measures_the_edges(self):
        r = chk.evaluate_dive(_dive(), floor_y=35.0)
        assert r["pass"], r["reasons"]
        assert r["t_in_water"] == 0.25
        assert r["sink_hold"] and r["oxygen_monotone"]
        assert r["t_pain_edge"] is not None and r["t_pain_edge"] < r["t_damage_onset"]
        assert 12.0 <= r["t_damage_onset"] <= 20.0
        assert r["health_at_rescue"] == 18.0

    def test_auto_float_fails_the_sink_hold(self):
        r = chk.evaluate_dive(_dive(float_at=2.0, damage_at=99, t_max=8.0), floor_y=35.0)
        assert not r["pass"]
        assert any("stay submerged" in x for x in r["reasons"])

    def test_damage_outside_the_window_fails(self):
        r = chk.evaluate_dive(_dive(damage_at=8.0), floor_y=35.0)
        assert not r["pass"] and any("first drowning damage" in x for x in r["reasons"])

    def test_oxygen_rising_underwater_fails(self):
        s = _dive()
        s[20]["oxygen"] = 20.0  # an air pocket mid-dive
        r = chk.evaluate_dive(s, floor_y=35.0)
        assert not r["pass"] and any("rose" in x for x in r["reasons"])

    def test_slow_rescue_fails(self):
        s = _dive()
        s[-1]["health"] = 14.0
        r = chk.evaluate_dive(s, floor_y=35.0)
        assert not r["pass"] and any("rescue too slow" in x for x in r["reasons"])


class TestEvaluateSurface:
    def test_surface_then_sinkback(self):
        s = [{"t": t / 4, "in_water": not (2.0 <= t / 4 < 4.0), "oxygen": 18} for t in range(0, 24)]
        r = chk.evaluate_surface(s)
        assert r["pass"] and r["t_surface"] == 2.0 and r["t_sinkback"] == 4.0

    def test_never_surfacing_fails(self):
        s = [{"t": t / 4, "in_water": True, "oxygen": 10} for t in range(0, 40)]
        r = chk.evaluate_surface(s)
        assert not r["pass"] and r["t_surface"] is None and r["t_sinkback"] is None

    def test_late_surface_fails(self):
        s = [{"t": t / 4, "in_water": t / 4 < 7.0, "oxygen": 10} for t in range(0, 40)]
        r = chk.evaluate_surface(s)
        assert not r["pass"] and r["t_surface"] == 7.0


class TestCheckContract:
    def test_frozen_gamerules_cover_prepare_and_builder_conditions(self):
        assert chk.FROZEN_GAMERULES == {
            "doMobSpawning": "false",
            "doDaylightCycle": "false",
            "doWeatherCycle": "false",
            "doImmediateRespawn": "true",
            "keepInventory": "true",
        }

    def test_spawn_bound_matches_the_builder(self):
        assert chk.SPAWN_DIST_MAX == WATER_MAX_DIST_FROM_SPAWN

    def test_measured_edges_from_a_pass_report(self):
        report = {
            "ts": 1.0,
            "cycles": [
                {
                    "w1_shore": {"distance_from_spawn": 40.0},
                    "w2_dive": {"t_damage_onset": 16.25},
                    "w4_escape": {"t_surface": 2.5, "t_sinkback": 3.75},
                },
                {
                    "w1_shore": {"distance_from_spawn": 41.0},
                    "w2_dive": {"t_damage_onset": 16.5},
                    "w4_escape": {"t_surface": 3.0, "t_sinkback": None},
                },
            ],
        }
        m = chk.measured_edges(report)
        assert m["t_damage_onset_min_s"] == 16.25 and m["t_damage_onset_max_s"] == 16.5
        assert m["t_surface_max_s"] == 3.0 and m["t_sinkback_min_s"] == 3.75
        assert m["distance_from_spawn"] == 41.0
