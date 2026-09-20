#!/usr/bin/env python3
"""Exp 62 live pre-check — four rows, run BEFORE the harness exists.

``docs/experiments/exp62_pressure_interoception_prereg.md`` §Rung A: *"Live pre-check before the
harness (four rows, ≈ 20 min): world spawn coordinates; light and time at pool 2's shore and floor
after build and after a client reconnect; pool 2's own gate (ii); ONE shipped agent trained at pool 1
and read loop-OFF at pool 2. That last row decides the rung before a harness exists — if the reading
does not resolve to the trained node, the replay was wrong about something live, and that is the
finding."*

This is a DIAGNOSTIC. It authorizes nothing, freezes nothing and gates nothing; its record is
evidence for the harness PR's design, and a row that fails is the finding, not a failure to retry.

Rows
----
``spawn``    world spawn, DERIVED from two snapshots rather than read from a new bridge field. The
             bridge already sends signed spawn-relative ``offset_x``/``offset_z`` and the 3D
             ``distance_from_spawn``: x/z come straight from the offsets, and y falls out of the
             distance at a KNOWN position (Exp 62 has two pools at different heights, so the sign
             is resolved by the second reading). Adding ``spawn_*`` to the payload would be a bridge
             protocol change, which fires the re-run trigger on four EARNED ledger rows for a number
             that is already recoverable.
``context``  ``light_level`` and ``time_of_day`` at both pools' shore and floor. These are the
             full-weight constants the cross-pool cosine rests on: if they differ between pools, the
             contrast is a LIGHT contrast (the committed replay measures a lit pool at 0.588 against
             the cave pool) and rung A would measure the context wall, not the body.
``gate_ii``  the committed gate-(ii) records for both pools, cited not re-run (they are their own
             script, ``l11_geometry_probe``); this row asserts both PASS and reports both cosines.
``carry``    the deciding row. ONE agent, trained at pool 1 by Exp 60's propose-only protocol, then
             read at pool 2 with the loop OFF: does pool 2's submerged reading resolve to the node
             the fear was booked on, and does the production read (``anticipatory_threat_need``)
             clear the loop's strict floor there?

Offline first (the standing rule): ``tests/unit/test_exp62_precheck.py`` runs every row against
``scripted_water.ScriptedWaterBridge`` with two pools before this touches the rig.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from survival_world.common import InstrumentError, sync_snapshot  # noqa: E402
from survival_world.water_trial import WaterTrial  # noqa: E402

LIVE_NEED_FLOOR = 0.5  # the loop's activation floor is STRICT (>), not >=


def derive_world_spawn(snapshots: list[dict[str, Any]], positions: list[dict[str, float]]) -> dict[str, Any]:
    """World spawn from two (snapshot, known position) pairs — no bridge change needed.

    ``offset_x``/``offset_z`` are signed position-minus-spawn, so x/z are exact from one reading.
    ``distance_from_spawn`` is 3D, so ``dy² = d² − dx² − dz²`` gives |dy| per reading; the two
    pools sit at different heights, so the pair fixes the sign. Returns the derivation and its
    residual — never a bare number, because a disagreeing pair means one of the two inputs is wrong.
    """
    if len(snapshots) != 2 or len(positions) != 2:
        raise ValueError("derive_world_spawn needs exactly two (snapshot, position) pairs")
    out: dict[str, Any] = {"readings": []}
    candidates: list[float] = []
    for snap, pos in zip(snapshots, positions, strict=True):
        dx, dz = float(snap["offset_x"]), float(snap["offset_z"])
        d = float(snap["distance_from_spawn"])
        spawn_x, spawn_z = pos["x"] - dx, pos["z"] - dz
        dy_sq = d * d - dx * dx - dz * dz
        dy = math.sqrt(dy_sq) if dy_sq > 0 else 0.0
        out["readings"].append(
            {
                "pos": pos,
                "offset_x": dx,
                "offset_z": dz,
                "distance": d,
                "spawn_x": spawn_x,
                "spawn_z": spawn_z,
                "abs_dy": round(dy, 3),
            }
        )
        candidates.extend([pos["y"] - dy, pos["y"] + dy])
    xs = {round(r["spawn_x"], 3) for r in out["readings"]}
    zs = {round(r["spawn_z"], 3) for r in out["readings"]}
    out["x"] = out["readings"][0]["spawn_x"] if len(xs) == 1 else None
    out["z"] = out["readings"][0]["spawn_z"] if len(zs) == 1 else None
    # the y the two readings agree on (within a block); None when they do not
    best, best_err = None, None
    for a in candidates[:2]:
        for b in candidates[2:]:
            err = abs(a - b)
            if best_err is None or err < best_err:
                best, best_err = (a + b) / 2.0, err
    out["y"] = round(best, 3) if best is not None and best_err is not None and best_err <= 1.0 else None
    out["y_residual"] = round(best_err, 3) if best_err is not None else None
    out["agrees"] = out["x"] is not None and out["z"] is not None and out["y"] is not None
    return out


def context_constants(trial: WaterTrial, label: str) -> dict[str, Any]:
    """``light_level`` and ``time_of_day`` at this pool's shore and floor, from the live snapshot."""
    pick = ("light_level", "time_of_day")
    trial.rescue(f"{label}-context-shore")
    shore = sync_snapshot(trial.aut) or {}
    trial.submerge(f"{label}-context-floor")
    floor = sync_snapshot(trial.aut) or {}
    trial.rescue(f"{label}-context-back")
    return {
        "pool": label,
        "shore": {k: shore.get(k) for k in pick},
        "floor": {k: floor.get(k) for k in pick},
    }


def compare_context(pool1: dict[str, Any], pool2: dict[str, Any]) -> dict[str, Any]:
    """The pools' full-weight constants must MATCH, or the cross-pool contrast is a light contrast."""
    mismatches = [
        f"{where}.{key}: pool1={pool1[where][key]} pool2={pool2[where][key]}"
        for where in ("shore", "floor")
        for key in ("light_level", "time_of_day")
        if pool1[where][key] != pool2[where][key]
    ]
    return {"pool1": pool1, "pool2": pool2, "match": not mismatches, "mismatches": mismatches}


def gate_ii_rows(paths: list[Path]) -> dict[str, Any]:
    """Cite the committed gate-(ii) records; this script never re-runs the probe."""
    rows = []
    for p in paths:
        rec = json.loads(Path(p).read_text())
        rows.append(
            {
                "record": str(p),
                "cos_a4": (rec.get("cosine") or {}).get("a4_gained"),
                "threshold": (rec.get("cosine") or {}).get("threshold"),
                "pass": bool((rec.get("run_gate") or {}).get("pass")),
                "verdict": rec.get("verdict"),
            }
        )
    return {"rows": rows, "all_pass": all(r["pass"] for r in rows) if rows else False}


def carry_row(trial1: WaterTrial, geom2: dict[str, Any]) -> dict[str, Any]:
    """Train at pool 1 (Exp 60's propose-only protocol), then read at pool 2 with the loop OFF.

    ONE agent, ONE instrument attach: the SAME trial object is used for both pools by swapping the
    geometry it teleports against, because a second WaterTrial on the same agent double-wraps the
    executor spy (the trial refuses that, and the prereg names it).
    """
    training, episode_clusters = trial1.train()
    trial1.rescue("carry-boundary")
    pool1_cluster = trial1.encode_world_cluster()
    nac = trial1.aut.bio.nac
    trained = {
        "usable_episodes": training.get("usable_episodes"),
        "episode_clusters": sorted(set(episode_clusters)),
        "majority_cluster": max(set(episode_clusters), key=episode_clusters.count) if episode_clusters else None,
        "shore_cluster_pool1": pool1_cluster,
    }
    trial1.submerge("carry-pool1-read")
    water1 = trial1.encode_world_cluster()
    trial1.rescue("carry-pool1-back")
    trained["water_cluster_pool1"] = water1
    trained["fear_pool1"] = round(nac.cluster_fear(trial1.agent_id, water1), 4) if water1 else None
    trained["need_pool1"] = nac.anticipatory_threat_need(trial1.agent_id, {"world": water1}) if water1 else None

    # ---- pool 2, same agent, loop still OFF: only the geometry changes ----
    geom1 = trial1.geom
    trial1.use_geometry(geom2)
    trial1.submerge("carry-pool2-read")
    water2 = trial1.encode_world_cluster()
    trial1.rescue("carry-pool2-back")
    shore2 = trial1.encode_world_cluster()
    trial1.use_geometry(geom1)
    read = {
        "water_cluster_pool2": water2,
        "shore_cluster_pool2": shore2,
        "fear_pool2": round(nac.cluster_fear(trial1.agent_id, water2), 4) if water2 else None,
        "need_pool2": nac.anticipatory_threat_need(trial1.agent_id, {"world": water2}) if water2 else None,
        "same_node_as_pool1_water": bool(water2) and water2 == trained["water_cluster_pool1"],
        "same_node_as_training_majority": bool(water2) and water2 == trained["majority_cluster"],
    }
    read["clears_live_floor_at_pool2"] = (read["need_pool2"] or 0.0) > LIVE_NEED_FLOOR
    # The prereg's own reading of this row: a hit is the predicted result, a miss is the FINDING.
    read["reading"] = (
        "pool 2's water reading resolves to the trained node and the production read clears the "
        "loop's floor — the replay's prediction holds live"
        if read["same_node_as_pool1_water"] and read["clears_live_floor_at_pool2"]
        else "pool 2's water reading does NOT carry the trained fear live — the finding the pre-check exists for"
    )
    return {"trained_at_pool1": trained, "read_at_pool2": read}


def run(
    trial1: WaterTrial, geom2: dict[str, Any], *, gate_records: list[Path], positions: list[dict[str, float]]
) -> dict[str, Any]:
    """All four rows against an ALREADY-BUILT trial (one agent, instruments attached once)."""
    record: dict[str, Any] = {
        "_format_version": "1.0",
        "kind": "exp62_precheck",
        "authorizes": "nothing — diagnostic evidence for the harness design",
        "ts": time.time(),
    }
    # The spawn row reads the RAW bridge state, not the sensed snapshot: `offset_x`/`offset_z` are
    # bridge fields the body does not declare as sensors (its roster is 17 world sensors, offsets not
    # among them), so `sync_snapshot` — which returns what the BODY senses — cannot see them.
    trial1.rescue("precheck-spawn-1")
    snap1 = dict(trial1.aut.client.latest_state() or {})
    geom1 = trial1.geom
    trial1.use_geometry(geom2)
    trial1.rescue("precheck-spawn-2")
    snap2 = dict(trial1.aut.client.latest_state() or {})
    pool2_context = context_constants(trial1, "pool2")
    trial1.use_geometry(geom1)
    pool1_context = context_constants(trial1, "pool1")

    record["spawn"] = derive_world_spawn([snap1, snap2], positions)
    record["context"] = compare_context(pool1_context, pool2_context)
    record["gate_ii"] = gate_ii_rows(gate_records)
    record["carry"] = carry_row(trial1, geom2)
    record["rows_ok"] = {
        "spawn": bool(record["spawn"]["agrees"]),
        "context": bool(record["context"]["match"]),
        "gate_ii": bool(record["gate_ii"]["all_pass"]),
        "carry": bool(record["carry"]["read_at_pool2"]["same_node_as_pool1_water"]),
    }
    return record


def build_trial(args: argparse.Namespace, geom1: dict[str, Any], home: Path) -> tuple[WaterTrial, Any, Any, Any]:
    """The canonical Exp 60/61 assembly — one agent, the production encoder, ONE instrument attach.

    The caps come from POOL 1's `measured` block because that is where the training happens: the
    probe window must end before this pool's pain edge, and a training episode before its damage
    onset, both with the frozen margins.
    """
    from exp56.common import RconControl  # the proven RCON client both harnesses use
    from survival_world.exp61_run import FROZEN, build_aut

    rcon = RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    aut, encoder, pump = build_aut(args, agent_id=args.agent_id, home=home)
    measured = geom1["measured"]
    trial = WaterTrial(
        aut=aut,
        rcon=rcon,
        username=args.username,
        geom=geom1,
        frozen=FROZEN["exp60"],
        probe_cap_s=float(measured["t_pain_edge_min_s"]) - FROZEN["exp60"]["probe_cap_margin_s"],
        train_cap_s=float(measured["t_damage_onset_min_s"]) - FROZEN["exp60"]["train_cap_margin_s"],
        persistence_dir=home,
        agent_id=args.agent_id,
        encoder=encoder,
        settle_guard=FROZEN["settle_guard"],
    )
    trial.attach_instruments()
    return trial, aut, pump, rcon


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool1-anchor", default=str(Path.home() / ".maxim" / "exp60_water_classroom.json"))
    ap.add_argument("--pool2-anchor", default=str(Path.home() / ".maxim" / "exp62_pool2_water_classroom.json"))
    ap.add_argument(
        "--gate-record", action="append", default=[], help="a committed gate-(ii) record (give one per pool)"
    )
    ap.add_argument("--out", required=True, help="where to write the diagnostic record")
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=25567)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=25575)
    ap.add_argument("--rcon-password", required=True)
    ap.add_argument("--username", default="maxim")
    ap.add_argument("--agent-id", default="exp62_precheck")
    ap.add_argument("--workdir", default=None, help="durable home for the agent (default: a tmpdir)")
    args = ap.parse_args(argv)

    geom1 = json.loads(Path(args.pool1_anchor).expanduser().read_text())
    geom2 = json.loads(Path(args.pool2_anchor).expanduser().read_text())
    for label, geom in (("pool 1", geom1), ("pool 2", geom2)):
        if "measured" not in geom:
            print(f"INSTRUMENT ERROR: {label}'s record carries no `measured` block — run exp60_water_check on it first")
            return 4

    import tempfile

    from survival_world.exp61_run import close_and_stage

    home = Path(args.workdir).expanduser() if args.workdir else Path(tempfile.mkdtemp(prefix="exp62_precheck_"))
    trial = aut = pump = rcon = None
    try:
        trial, aut, pump, rcon = build_trial(args, geom1, home)
        record = run(
            trial,
            geom2,
            gate_records=[Path(p) for p in args.gate_record],
            positions=[
                {"x": float(geom1["shore"][0]), "y": float(geom1["shore"][1]), "z": float(geom1["shore"][2])},
                {"x": float(geom2["shore"][0]), "y": float(geom2["shore"][1]), "z": float(geom2["shore"][2])},
            ],
        )
    except InstrumentError as exc:
        print(f"INSTRUMENT ERROR: {exc}")
        return 4
    finally:
        if trial is not None:
            try:
                trial.detach_instruments()
                trial.final_rescue()
            except Exception as exc:  # noqa: BLE001 — teardown must not hide the rows
                print(f"WARNING: trial teardown raised: {exc!r}")
        if aut is not None:
            try:
                close_and_stage(aut, pump, None)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: closing the agent raised: {exc!r}")
        if rcon is not None:
            rcon.close()

    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).expanduser().write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps(record["rows_ok"], indent=2))
    print(record["carry"]["read_at_pool2"]["reading"])
    print(f"\nprecheck record -> {args.out}")
    return 0 if all(record["rows_ok"].values()) else 1  # 1 = a row did not hold; READ it, do not retry


if __name__ == "__main__":
    raise SystemExit(main())
