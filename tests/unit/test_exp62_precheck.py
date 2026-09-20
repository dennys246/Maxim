"""Exp 62 live pre-check — the pure rows offline, and the whole thing against a two-pool bridge.

The standing rule is that a rig-bound script runs end to end offline before its PR. The pure rows
(spawn derivation, the context comparison, the gate-(ii) citation) are asserted here directly; the
`carry` row — one agent trained in pool 1 and read in pool 2 with the loop off — runs against
`ScriptedWaterBridge` with BOTH pools, which is the case the bridge gained multi-pool support for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from survival_world import exp62_precheck as P  # noqa: E402
from survival_world.scripted_water import ScriptedWaterBridge, ScriptedWaterControl  # noqa: E402
from survival_world.setup_world import water_anchor_record, water_classroom_geometry  # noqa: E402


def _xyz(triple) -> dict[str, float]:
    return {"x": float(triple[0]), "y": float(triple[1]), "z": float(triple[2])}


def _record(shore_y: int, pool_id: str) -> dict:
    rec = water_anchor_record(water_classroom_geometry(10, 20, depth=5, shore_y=shore_y), pool_id=pool_id)
    rec["measured"] = {"t_damage_onset_min_s": 16.0, "t_damage_onset_max_s": 16.2, "t_pain_edge_min_s": 5.1}
    return rec


class TestSpawnDerivation:
    """World spawn from the snapshot, so the bridge protocol never has to change for it."""

    def test_two_readings_at_different_heights_fix_x_z_and_y(self):
        spawn = {"x": 100.0, "y": 64.0, "z": -50.0}
        positions = [{"x": 130.0, "y": 40.0, "z": -10.0}, {"x": 130.0, "y": 95.0, "z": -10.0}]
        snaps = []
        for p in positions:
            dx, dy, dz = p["x"] - spawn["x"], p["y"] - spawn["y"], p["z"] - spawn["z"]
            snaps.append({"offset_x": dx, "offset_z": dz, "distance_from_spawn": (dx * dx + dy * dy + dz * dz) ** 0.5})
        out = P.derive_world_spawn(snaps, positions)
        assert out["agrees"]
        assert out["x"] == pytest.approx(spawn["x"]) and out["z"] == pytest.approx(spawn["z"])
        assert out["y"] == pytest.approx(spawn["y"], abs=0.5)

    def test_disagreeing_readings_report_None_rather_than_a_confident_number(self):
        positions = [{"x": 130.0, "y": 40.0, "z": -10.0}, {"x": 130.0, "y": 95.0, "z": -10.0}]
        snaps = [
            {"offset_x": 30.0, "offset_z": 40.0, "distance_from_spawn": 55.0},
            {"offset_x": 31.0, "offset_z": 40.0, "distance_from_spawn": 55.0},  # x disagrees
        ]
        out = P.derive_world_spawn(snaps, positions)
        assert out["x"] is None and not out["agrees"]

    def test_it_refuses_a_single_reading(self):
        with pytest.raises(ValueError):
            P.derive_world_spawn([{"offset_x": 0, "offset_z": 0, "distance_from_spawn": 0}], [{"x": 0, "y": 0, "z": 0}])


class TestContextComparison:
    """Light and time are the full-weight constants the cross-pool cosine rests on."""

    def test_matching_pools_pass(self):
        ctx = {
            "shore": {"light_level": 0.0, "time_of_day": 0.0417},
            "floor": {"light_level": 0.0, "time_of_day": 0.0417},
        }
        out = P.compare_context({**ctx, "pool": "pool1"}, {**ctx, "pool": "pool2"})
        assert out["match"] and out["mismatches"] == []

    def test_a_lit_second_pool_is_named_not_waved_through(self):
        dark = {
            "shore": {"light_level": 0.0, "time_of_day": 0.0417},
            "floor": {"light_level": 0.0, "time_of_day": 0.0417},
        }
        lit = {
            "shore": {"light_level": 15.0, "time_of_day": 0.0417},
            "floor": {"light_level": 15.0, "time_of_day": 0.0417},
        }
        out = P.compare_context({**dark, "pool": "pool1"}, {**lit, "pool": "pool2"})
        assert not out["match"] and len(out["mismatches"]) == 2
        assert "light_level" in out["mismatches"][0]


class TestGateCitation:
    def test_it_reads_the_committed_records_and_refuses_an_empty_citation(self, tmp_path):
        rec = {
            "cosine": {"a4_gained": 0.7875, "threshold": 0.85},
            "run_gate": {"pass": True},
            "verdict": "separable_here",
        }
        p = tmp_path / "gate.json"
        p.write_text(json.dumps(rec))
        out = P.gate_ii_rows([p])
        assert out["all_pass"] and out["rows"][0]["cos_a4"] == 0.7875
        assert P.gate_ii_rows([])["all_pass"] is False  # citing nothing is not a pass

    def test_a_failed_gate_propagates(self, tmp_path):
        p = tmp_path / "gate.json"
        p.write_text(json.dumps({"cosine": {"a4_gained": 0.91, "threshold": 0.85}, "run_gate": {"pass": False}}))
        assert P.gate_ii_rows([p])["all_pass"] is False


@pytest.mark.slow
def test_all_four_rows_against_a_two_pool_scripted_bridge(tmp_path, monkeypatch):
    """The whole pre-check offline: one agent, one instrument attach, two pools.

    This is the case `ScriptedWaterBridge` gained multi-pool support for — with a single submerged
    point, pool 2's floor would read as dry and the carry row would be meaningless.
    """
    from survival_world import exp61_run as E
    from survival_world import water_trial as WT

    pool1, pool2 = _record(64, "pool1"), _record(120, "pool2")
    for name, rec in (("pool1.json", pool1), ("pool2.json", pool2)):
        (tmp_path / name).write_text(json.dumps(rec))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"cosine": {"a4_gained": 0.78, "threshold": 0.85}, "run_gate": {"pass": True}}))

    srv = ScriptedWaterBridge(
        shore=_xyz(pool1["shore"]),
        submerged=[_xyz(pool1["submerged"]), _xyz(pool2["submerged"])],
        damage_onset_s=16.0,
        damage_per_s=2.0,
    )
    rcon = ScriptedWaterControl(srv, gamerules={r: v for r, v in WT.R3_GAMERULES})
    monkeypatch.setattr(WT.WaterTrial, "check_fingerprint", lambda self, m: {"offline": True})
    monkeypatch.setitem(E.FROZEN["exp60"], "K_usable_episodes", 1)
    monkeypatch.setitem(E.FROZEN["exp60"], "loop_liveness_s", 2.0)
    monkeypatch.setitem(E.FROZEN["exp60"], "loop_liveness_min_ticks", 3)
    monkeypatch.setitem(E.FROZEN["exp60"], "loop_warm_s", 0.5)
    monkeypatch.setattr(P, "build_trial", P.build_trial)  # explicit: the real assembly runs

    import argparse

    args = argparse.Namespace(
        rcon_host="h",
        rcon_port=1,
        rcon_password="p",
        username="maxim",
        bridge_host="127.0.0.1",
        bridge_port=srv.port,
        agent_id="exp62_precheck_offline",
        workdir=str(tmp_path / "home"),
    )
    monkeypatch.setattr("exp56.common.RconControl", lambda *a, **k: rcon)

    trial, aut, pump, _rcon = P.build_trial(args, pool1, Path(args.workdir))
    try:
        record = P.run(
            trial,
            pool2,
            gate_records=[gate],
            positions=[_xyz(pool1["shore"]), _xyz(pool2["shore"])],
        )
    finally:
        trial.detach_instruments()
        E.close_and_stage(aut, pump, None)
        srv.close()

    assert set(record["rows_ok"]) == {"spawn", "context", "gate_ii", "carry"}
    assert record["context"]["match"], record["context"]["mismatches"]
    assert record["gate_ii"]["all_pass"]
    carry = record["carry"]
    assert carry["trained_at_pool1"]["usable_episodes"] >= 1
    assert carry["read_at_pool2"]["water_cluster_pool2"], "pool 2's floor must read as a water situation"
    assert "reading" in carry["read_at_pool2"]
    assert record["authorizes"].startswith("nothing")
