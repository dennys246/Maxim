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


@pytest.fixture(autouse=True)
def _never_touch_the_real_records(tmp_path, monkeypatch):
    """No test may read or write ``~/.maxim/exp60_water_classroom.json``.

    A review of this very file caught it writing there: the builder/check default paths are module
    constants, so a test that forgets to patch them edits the LIVE apparatus record — on the rig
    that means injecting a synthetic `measured` block into the pool Exp 60 and R3 run on, with no
    git copy to restore from. Autouse, so forgetting is not possible.
    """
    from survival_world import setup_world as SW

    monkeypatch.setattr(SW, "WATER_ANCHOR_FILE", tmp_path / "guard_exp60_water_classroom.json")
    monkeypatch.setattr(SW, "EXP58_ANCHOR_FILE", tmp_path / "guard_exp58_classroom.json")
    monkeypatch.setattr(chk, "ANCHOR_FILE", tmp_path / "guard_check_record.json")


from survival_world.setup_world import (  # noqa: E402
    WATER_DEPTH_DEFAULT,
    WATER_SHORE_Y,
    WATER_MAX_DIST_FROM_SPAWN,
    WATER_MIN_DIST_FROM_EXP58,
    exp58_clearance,
    pools_disjoint,
    record_shell,
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
                    "w2_dive": {"t_damage_onset": 16.25, "t_pain_edge": 5.19},
                    "w4_escape": {"t_surface": 2.5, "t_sinkback": 3.75},
                },
                {
                    "w1_shore": {"distance_from_spawn": 41.0},
                    "w2_dive": {"t_damage_onset": 16.5, "t_pain_edge": 5.09},
                    "w4_escape": {"t_surface": 3.0, "t_sinkback": None},
                },
            ],
        }
        m = chk.measured_edges(report)
        assert m["t_damage_onset_min_s"] == 16.25 and m["t_damage_onset_max_s"] == 16.5
        assert m["t_surface_max_s"] == 3.0 and m["t_sinkback_min_s"] == 3.75
        assert m["distance_from_spawn"] == 41.0
        assert m["t_pain_edge_min_s"] == 5.09


class TestPartialCyclePreservation:
    def test_incomplete_cycle_never_counts_as_pass(self):
        good = {"cycle": 0, "pass": True}
        assert chk.all_cycles_pass([good, {"cycle": 1, "pass": True}], 2)
        assert not chk.all_cycles_pass([good], 2)  # fewer cycles than expected
        assert not chk.all_cycles_pass([good, {"cycle": 1, "incomplete": True, "failed_at": "w4_escape"}], 2)
        assert not chk.all_cycles_pass([], 0) and not chk.all_cycles_pass([], 3)

    def test_partial_record_is_preserved_with_its_stage(self):
        report = {"cycles": [{"cycle": 0, "pass": True}]}
        rec = {"cycle": 1, "w1_shore": {"pass": True}, "w2_dive": {"pass": False}}
        chk._preserve_partial(report, rec, "w4_escape")
        assert report["failed_at"] == "w4_escape"
        assert report["cycles"][-1] is rec and rec["incomplete"] is True and rec["failed_at"] == "w4_escape"
        # nothing in progress (error before the loop): report still names the stage, cycles untouched
        report2 = {"cycles": []}
        chk._preserve_partial(report2, None, "preflight")
        assert report2["failed_at"] == "preflight" and report2["cycles"] == []


class TestBridgeRosterGate:
    def test_gate_reads_the_raw_bridge_roster_not_the_body(self):
        # a pre-#719 bridge emits oxygen but not is_in_water: the body would still CARRY
        # is_in_water (declared, initial 0) — the gate must name it missing
        old_bridge = {
            "oxygen": 20,
            "health": 20,
            "y_altitude": 35,
            "on_ground": 1,
            "nearest_hostile_dist": 64,
            "distance_from_spawn": 70,
        }
        assert chk.missing_bridge_sensors(old_bridge, chk.REQUIRED_BRIDGE_SENSORS) == {"is_in_water"}
        assert chk.missing_bridge_sensors({**old_bridge, "is_in_water": 0}, chk.REQUIRED_BRIDGE_SENSORS) == set()
        assert chk.missing_bridge_sensors({}, chk.REQUIRED_BRIDGE_SENSORS) == set(chk.REQUIRED_BRIDGE_SENSORS)

    def test_required_roster_covers_every_gated_sensor(self):
        assert {
            "is_in_water",
            "oxygen",
            "health",
            "y_altitude",
            "nearest_hostile_dist",
            "distance_from_spawn",
        } <= chk.REQUIRED_BRIDGE_SENSORS


class TestTwoPoolPlumbing:
    """Exp 62 plumbing: a SECOND pool at another height, recorded in its OWN file.

    The failure this guards against is concrete: one fixed anchor path meant building or checking
    pool 2 overwrote pool 1's record — and the `measured` block in it is what every harness refuses
    to run without (it happened twice during the R3 campaign, recovered by `git checkout --`).
    """

    def test_shore_y_moves_the_whole_pool_and_keeps_its_shape(self):
        low, high = water_classroom_geometry(0, 0), water_classroom_geometry(0, 0, shore_y=95)
        assert low["shore_y"] == WATER_SHORE_Y and high["shore_y"] == 95
        dy = 95 - WATER_SHORE_Y
        for key in ("shore", "submerged", "surface", "pool_floor", "submerged_head", "lip", "shore_floor"):
            assert [high[key][0], high[key][1] - dy, high[key][2]] == list(low[key]), key
        assert high["depth"] == low["depth"]  # the column is as deep, just higher up
        # the forceload is an x/z footprint: stacking must not change it (same chunks, no load window)
        assert high["forceload"] == low["forceload"]

    def test_stacked_pools_are_disjoint_and_an_overlap_refuses(self):
        pool1 = water_classroom_geometry(0, 0)
        rec1 = water_anchor_record(pool1, pool_id="pool1")
        ok, why = pools_disjoint(water_classroom_geometry(0, 0, shore_y=95), rec1)
        assert ok and "vertical gap" in why
        # one block of overlap is a shared wall — refuse it
        touching = water_classroom_geometry(0, 0, shore_y=WATER_SHORE_Y + (pool1["shell"][4] - pool1["shell"][1]))
        ok2, why2 = pools_disjoint(touching, rec1)
        assert not ok2 and "intersect" in why2

    def test_a_legacy_record_is_DERIVED_not_waved_through(self):
        """Pool 1's record on the rig predates `shell`; deriving it is what arms the guard. The
        alternative — rebuilding pool 1 for the field — drops `measured` and makes exp60_run and
        r3_run refuse every row."""
        pool1 = water_classroom_geometry(0, 0)
        legacy = {k: v for k, v in water_anchor_record(pool1).items() if k != "shell"}
        assert record_shell(legacy) == tuple(pool1["shell"])
        ok, why = pools_disjoint(water_classroom_geometry(0, 0, shore_y=95), legacy)
        assert ok and "vertical gap" in why
        ok2, why2 = pools_disjoint(water_classroom_geometry(0, 0, shore_y=41), legacy)
        assert not ok2 and "intersect" in why2

    def test_guard_says_NOT_CHECKED_only_when_it_truly_cannot(self):
        pool2 = water_classroom_geometry(0, 0, shore_y=95)
        assert pools_disjoint(pool2, None) == (True, "no other pool given — NOT CHECKED")
        ok, why = pools_disjoint(pool2, {"pool_id": "mystery"})
        assert ok and "NOT CHECKED" in why
        with pytest.raises(ValueError):
            record_shell({"shell": [1, 2, 3]})

    def test_reported_gap_is_the_blocks_between_the_shells(self):
        pool1 = water_classroom_geometry(0, 0)
        rec1 = water_anchor_record(pool1)
        height = pool1["shell"][4] - pool1["shell"][1]
        one_gap = water_classroom_geometry(0, 0, shore_y=WATER_SHORE_Y + height + 2)
        assert pools_disjoint(one_gap, rec1) == (True, "vertical gap 1 block(s) between shells")
        assert pools_disjoint(water_classroom_geometry(400, 400), rec1) == (True, "shells are separated horizontally")

    def test_record_carries_what_a_second_pool_needs(self):
        geom = water_classroom_geometry(0, 0, shore_y=95)
        rec = water_anchor_record(geom, pool_id="pool2", world_spawn=(1.0, 64.0, 2.0))
        assert rec["pool_id"] == "pool2"
        assert rec["shell"] == list(geom["shell"])  # the disjointness guard reads this
        assert (rec["flee_x"], rec["flee_z"]) == (geom["shore"][0], geom["shore"][2])  # bridge --flee_x/_z
        assert rec["world_spawn"] == [1.0, 64.0, 2.0]
        assert rec["surface_y"] == 95 and rec["depth"] == geom["depth"]
        # not given is None, never "clear"
        assert water_anchor_record(geom)["world_spawn"] is None

    def test_check_stamps_measured_into_the_pool_it_was_given(self, tmp_path):
        pool1, pool2 = tmp_path / "exp60_water_classroom.json", tmp_path / "exp62_pool2_water_classroom.json"
        import json

        pool1.write_text(json.dumps({"pool_id": "pool1", "measured": {"t_damage_onset_min_s": 16.0}}))
        pool2.write_text(json.dumps({"pool_id": "pool2"}))
        report = {
            "ts": 1.0,
            "cycles": [
                {
                    "w1_shore": {"distance_from_spawn": 12.0},
                    "w2_dive": {"t_damage_onset": 16.2, "t_pain_edge": 5.2},
                    "w4_escape": {"t_surface": 2.1, "t_sinkback": 2.4},
                }
            ],
        }
        chk._stamp_measured(report, tmp_path / "out.json", anchor_file=pool2)
        assert "measured" in json.loads(pool2.read_text())  # the pool we asked for
        assert json.loads(pool1.read_text())["measured"] == {"t_damage_onset_min_s": 16.0}  # untouched


class TestTwoPoolBuildOffline:
    """The builder RUN end to end, twice, against a fake server — not just its pure helpers.

    The standing rule is that a rig-bound script runs offline before its PR; the pure geometry
    tests above would not have caught a flag that never reaches `water_classroom_geometry`, a
    record written to the wrong path, or a guard that refuses nothing because it reads the file
    it is about to write.
    """

    class FakeRcon:
        """Answers exactly the verbs the builder issues, and remembers them."""

        def __init__(self, **_kw):
            self.sent: list[str] = []

        def command(self, cmd: str) -> str:
            self.sent.append(cmd)
            head = cmd.split()[0]
            if head == "fill":
                return "Successfully filled 9 block(s)"
            if head == "execute":  # every post-build block assertion
                return "Test passed"
            if head == "forceload":
                return "Marked chunks to be force loaded"
            if head in {"spawnpoint", "setworldspawn"}:
                return "Set the spawn point"
            if head == "gamerule":
                return f"Gamerule {cmd.split()[1]} is now set to: false"
            return "(ok)"

        def close(self) -> None:
            pass

    def _build(self, monkeypatch, tmp_path, *, shore_y, anchor_file, pool_id, extra=()):
        import time as _time

        from survival_world import setup_world as SW

        fake = self.FakeRcon()
        monkeypatch.setattr(SW, "RconControl", lambda *a, **k: fake)
        monkeypatch.setattr(SW, "EXP58_ANCHOR_FILE", tmp_path / "no_exp58.json")
        monkeypatch.setattr(_time, "sleep", lambda *_a: None)
        argv = [
            "water_classroom",
            "--rcon-password",
            "x",
            "--anchor-x",
            "0",
            "--anchor-z",
            "0",
            *(() if shore_y is None else ("--shore-y", str(shore_y))),
            "--anchor-file",
            str(anchor_file),
            "--pool-id",
            pool_id,
            *extra,
        ]
        return SW.main(argv), fake

    def test_two_pools_build_into_their_own_records_and_the_first_is_untouched(self, monkeypatch, tmp_path):
        import json

        pool1 = tmp_path / "exp60_water_classroom.json"
        pool2 = tmp_path / "exp62_pool2_water_classroom.json"

        rc, fake1 = self._build(monkeypatch, tmp_path, shore_y=40, anchor_file=pool1, pool_id="pool1")
        assert rc == 0, "pool 1 build refused"
        rec1 = json.loads(pool1.read_text())
        assert rec1["pool_id"] == "pool1" and rec1["surface_y"] == 40
        assert any(c.startswith("fill") for c in fake1.sent) and any(c.startswith("forceload") for c in fake1.sent)

        # the check's stamp is what a rebuild of the OTHER pool must not destroy
        rec1["measured"] = {"t_damage_onset_min_s": 16.07}
        pool1.write_text(json.dumps(rec1))

        rc2, _ = self._build(monkeypatch, tmp_path, shore_y=95, anchor_file=pool2, pool_id="pool2")
        assert rc2 == 0, "pool 2 build refused"
        rec2 = json.loads(pool2.read_text())
        assert rec2["pool_id"] == "pool2" and rec2["surface_y"] == 95
        assert rec2["shore"][1] == 95 and rec2["submerged"][1] == 95 - rec2["depth"]
        assert json.loads(pool1.read_text())["measured"] == {"t_damage_onset_min_s": 16.07}  # the whole point

    def test_a_colliding_second_pool_is_refused_before_any_fill(self, monkeypatch, tmp_path):
        pool1 = tmp_path / "exp60_water_classroom.json"
        pool2 = tmp_path / "exp62_pool2_water_classroom.json"
        assert self._build(monkeypatch, tmp_path, shore_y=40, anchor_file=pool1, pool_id="pool1")[0] == 0
        rc, fake = self._build(
            monkeypatch, tmp_path, shore_y=41, anchor_file=pool2, pool_id="pool2", extra=("--stack-on", str(pool1))
        )
        assert rc == 4, "an overlapping shell must refuse"
        assert not pool2.exists(), "a refused build must not leave a record"
        assert not any(c.startswith("fill") for c in fake.sent), "it must refuse BEFORE touching the world"

    def test_without_stack_on_the_guard_says_so_and_records_null(self, monkeypatch, tmp_path, capsys):
        """The build must not be a function of ambient directory contents — but silence is not ok."""
        import json

        pool2 = tmp_path / "exp62_pool2_water_classroom.json"
        assert self._build(monkeypatch, tmp_path, shore_y=95, anchor_file=pool2, pool_id="pool2")[0] == 0
        assert "NOT CHECKED" in capsys.readouterr().out
        assert json.loads(pool2.read_text())["pool_clearance"] is None

    def test_stack_on_records_the_clearance_it_measured(self, monkeypatch, tmp_path):
        import json

        pool1, pool2 = tmp_path / "p1.json", tmp_path / "p2.json"
        self._build(monkeypatch, tmp_path, shore_y=40, anchor_file=pool1, pool_id="pool1")
        rc, _ = self._build(
            monkeypatch, tmp_path, shore_y=95, anchor_file=pool2, pool_id="pool2", extra=("--stack-on", str(pool1))
        )
        assert rc == 0
        clearance = json.loads(pool2.read_text())["pool_clearance"]
        assert clearance["other_record"] == str(pool1) and "vertical gap" in clearance["result"]

    def test_a_rebuild_inherits_the_recorded_height_and_a_mismatch_refuses(self, monkeypatch, tmp_path):
        """A hard default of 40 would silently rebuild pool 2 down inside pool 1's band."""
        import json

        pool2 = tmp_path / "p2.json"
        self._build(monkeypatch, tmp_path, shore_y=95, anchor_file=pool2, pool_id="pool2")
        rc, _ = self._build(monkeypatch, tmp_path, shore_y=None, anchor_file=pool2, pool_id="pool2")
        assert rc == 0 and json.loads(pool2.read_text())["surface_y"] == 95
        rc2, fake2 = self._build(monkeypatch, tmp_path, shore_y=40, anchor_file=pool2, pool_id="pool2")
        assert rc2 == 2, "rebuilding one record at another height is a different pool — refuse"
        assert not any(c.startswith("fill") for c in fake2.sent)
        assert json.loads(pool2.read_text())["surface_y"] == 95

    def test_backfill_adds_the_new_fields_and_keeps_measured(self, monkeypatch, tmp_path):
        import json

        from survival_world import setup_world as SW

        pool1 = tmp_path / "p1.json"
        legacy = {k: v for k, v in water_anchor_record(water_classroom_geometry(0, 0)).items() if k != "shell"}
        legacy["measured"] = {"t_damage_onset_min_s": 16.07}
        pool1.write_text(json.dumps(legacy))
        assert SW.main(["water_classroom", "--rcon-password", "x", "--anchor-file", str(pool1), "--backfill"]) == 0
        rec = json.loads(pool1.read_text())
        assert rec["measured"] == {"t_damage_onset_min_s": 16.07}, "backfill must never touch measured"
        assert rec["shell"] == list(water_classroom_geometry(0, 0)["shell"])
        assert (rec["flee_x"], rec["flee_z"]) == (legacy["shore"][0], legacy["shore"][2])

    def test_the_printed_next_step_names_this_pool(self, monkeypatch, tmp_path, capsys):
        pool2 = tmp_path / "exp62_pool2_water_classroom.json"
        self._build(monkeypatch, tmp_path, shore_y=95, anchor_file=pool2, pool_id="pool2")
        out = capsys.readouterr().out
        # The builder must not hard-code a docs/experiments/data path (the provenance lint reads
        # that as "this script writes gated records"); it names the FLAGS, and the check itself
        # refuses a default --out while naming the path to use.
        assert "--anchor-file" in out and str(pool2) in out
        assert "--out" in out and "docs/experiments/data" not in out

    def test_a_failed_fill_or_verification_refuses_the_build(self, monkeypatch, tmp_path):
        """The fake must not only speak success — the builder's refusal branches are load-bearing."""
        import time as _time

        from survival_world import setup_world as SW

        class Broken(TestTwoPoolBuildOffline.FakeRcon):
            def __init__(self, mode):
                super().__init__()
                self.mode = mode

            def command(self, cmd: str) -> str:
                self.sent.append(cmd)
                if self.mode == "fill" and cmd.startswith("fill"):
                    return "No blocks were filled"
                if self.mode == "verify" and cmd.startswith("execute"):
                    return "Test failed"
                return super().command(cmd)

        for mode in ("fill", "verify"):
            rec = tmp_path / f"{mode}.json"
            monkeypatch.setattr(SW, "RconControl", lambda *a, _m=mode, **k: Broken(_m))
            monkeypatch.setattr(_time, "sleep", lambda *_a: None)
            rc = SW.main(
                [
                    "water_classroom",
                    "--rcon-password",
                    "x",
                    "--anchor-x",
                    "0",
                    "--anchor-z",
                    "0",
                    "--shore-y",
                    "40",
                    "--anchor-file",
                    str(rec),
                    "--pool-id",
                    "pool1",
                ]
            )
            assert rc == 4, f"a {mode} failure must refuse the build"
            assert not rec.exists(), "a refused build must not record geometry that is not there"
