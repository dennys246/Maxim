#!/usr/bin/env python3
"""Exp 62 rung A LIVE campaign — does the SHIPPED body carry the drowning-fear from pool 1 to pool 2?

Build step 4 of ``docs/experiments/exp62_pressure_interoception_prereg.md`` (v2, D1–D4 taken). The
claim is bounded: *invariant across altitude and spawn distance inside the sealed-shell, frozen-day
apparatus class* — never "anywhere". It closes the line Exp 60 and Exp 61 both left under §Not
claimed, and it is a property of the shipped ``minecraft_player`` body: no mechanism change, no new
sensor, no ingest. Everything here composes ``water_trial.WaterTrial`` (Exp 60's preflights,
training, live G2 read, placement) with Exp 61's DV helpers; nothing is hand-composed.

Arms — one FRESH agent per row, the fear learned at pool 1 by Exp 60's propose-only protocol:

=================  ========  ===================  ==  ============================================
arm                trained   read + contact        n  predicted
=================  ========  ===================  ==  ============================================
``cross``          pool 1    pool 2               12  fires; representation gate on the trained node
``same``           pool 1    pool 1               12  fires (the ceiling; Exp 60's own result)
``cross_ablated``  pool 1*   pool 2                3  censored, zero calls (* subscriber detached)
=================  ========  ===================  ==  ============================================

Per row, in this order and never the other way round:

1. preflights at pool 1 (fingerprint, bridge, liveness, gamerules, LIVE cluster-distinct, tools);
2. ``train()`` — K yoked propose-only episodes at pool 1; ZERO positive escape links either side;
3. ``use_geometry(read pool)`` — the SAME agent and its ONE instrument attach move pools;
4. the **NODE gate**, loop OFF: the read pool's submerged reading must resolve to the node the
   training booked fear on, with water fear at the cap, shore fear 0, and the production read
   (``anticipatory_threat_need``) above the loop's STRICT floor. **This row is the result.**
5. the **first contact**, loop LIVE: one US-free placement at the read pool — Exp 60's DV unit,
   scored by Exp 61's ``first_contact_outcome`` (decision DV ∧ behavioural DV ∧ drive-decisive).

Two design points a reviewer should check first, because both are places this harness could have
lied quietly:

**A failed read at the read pool is a RESULT, not a refusal.** ``live_g2`` raises on a failed read,
which is right for Exp 60/61 — there, the fear not reading is a broken instrument. Here it is the
finding the pre-registration names ("if the reading does not resolve to the trained node, the replay
was wrong about something live, and that is the finding"). So the raise is caught and CLASSIFIED
(:func:`classify_g2`): the needs on the TRAINING episode clusters say whether the fear was booked
and readable at all. Booked but not readable at the read pool → a clean row with NODE false, the
finding. Not booked → a refusal, because a training failure must never be reported as a carry
failure. That distinction is the whole confound.

**One probe cap for both pools.** The cap is ``min(pool 1, pool 2 pain edge) − margin``, so the
window is identical in every arm and sits below BOTH pools' pain edges. A per-pool cap would make
the arms' latencies incomparable and put the cap difference inside the contrast.

Run ON the bridge box (server + bridge from current main at ONE code hash — no ``git pull`` between
the first and last row; bridge at ``--state_interval_ms=100``; no second player)::

    export PYTHONPATH="$PWD/src"
    python scripts/survival_world/exp62_run.py replay --campaign-id <id> --write-experiment-results
    python scripts/survival_world/exp62_run.py run --campaign-id <id> --rcon-password '<pw>' \\
        --workdir ~/exp62_work \\
        --gate-record docs/experiments/data/exp60_geometry_2026-09-15b.json \\
        --gate-record docs/experiments/data/exp62_pool2_geometry.json --write-experiment-results
    python scripts/survival_world/exp62_run.py verdict --data docs/experiments/data/exp62_rows.jsonl \\
        --campaign-id <id> --json docs/experiments/data/exp62_verdict.json --write-experiment-results
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _provenance import (  # noqa: E402
    DirtyTreeError,
    ProvenanceError,
    evidence_out_paths_or_exit,
    in_process_code_provenance,
)
from exp56 import common as C  # noqa: E402
from survival_world.common import InstrumentError  # noqa: E402
from survival_world.exp61_run import (  # noqa: E402
    ESCAPE_SUFFIX,
    build_aut,
    close_and_stage,
    exp60_frozen_matches,
    first_contact_outcome,
    fisher_one_sided_p,
    wilson_interval,
)
from survival_world.exp61_run import FROZEN as FROZEN61  # noqa: E402
from survival_world.water_trial import Refusal, WaterTrial, _detach_fear_subscriber  # noqa: E402

REPO_ROOT = C.REPO_ROOT
REPLAY_SCRIPT = REPO_ROOT / "docs" / "experiments" / "data" / "exp62_cross_pool_replay.py"

# Frozen with the prereg. The Exp 60/61 numbers are LITERAL re-uses of Exp 61's FROZEN blocks and
# `frozen_matches()` fails if either drifts — nothing is inherited silently.
FROZEN: dict[str, Any] = {
    "arms": {"cross": 12, "same": 12, "cross_ablated": 3},
    # The ONE thing that differs between arm 1 and arm 2. Body, training, caps and guards are
    # identical, which is what makes the contrast the POOL and not anything else.
    "read_pool": {"cross": "pool2", "same": "pool1", "cross_ablated": "pool2"},
    "train_pool": "pool1",
    "g2_arm": {"cross": "fear", "same": "fear", "cross_ablated": "ablated"},
    "seeds": {
        "cross": list(range(600, 612)),
        "same": list(range(620, 632)),
        "cross_ablated": [640, 641, 642],
    },
    "fear_value_cap": -1.0,  # the trained node's fear must sit at the cap
    "read_floor": 0.5,  # the consumer's STRICT activation floor (>, not >=)
    "cosine_threshold": 0.85,  # the encoder's pattern threshold, for the replay row
    "drift_max_s": 0.5,  # same-arm first-contact latency, last quartile − first quartile
    "gates": {
        "node_min": 11 / 12,
        "cross_min": 0.70,
        "same_min": 0.70,
        "alpha": 0.05,
    },
    # DEEP COPIES, not aliases: `frozen_matches()` compares these against Exp 61's live blocks, and
    # a comparison of an object with itself can never fail. A later edit on either side shows up.
    "settle_guard": copy.deepcopy(FROZEN61["settle_guard"]),
    "exp60": copy.deepcopy(FROZEN61["exp60"]),
}
ARM_ORDER = ("cross", "same", "cross_ablated")


def frozen_matches() -> list[str]:
    """Every borrowed constant, checked against its source — Exp 61's, and through it Exp 60's."""
    drift = list(exp60_frozen_matches())
    if FROZEN["exp60"] != FROZEN61["exp60"]:
        drift.append("exp60 block differs from exp61's")
    if FROZEN["settle_guard"] != FROZEN61["settle_guard"]:
        drift.append("settle_guard differs from exp61's")
    for k in ("fear_value_cap", "read_floor"):
        if FROZEN[k] != FROZEN61[k]:
            drift.append(f"{k} differs from exp61's")
    return drift


# ─────────────────────────── pure: the read classification ───────────────────────────


def classify_g2(fields: dict[str, Any], *, arm: str, floor: float) -> dict[str, Any]:
    """Was the fear BOOKED at pool 1, and does it READ at the read pool? (pure)

    The whole confound of this experiment lives here. ``live_g2`` reports the production read on two
    things at once: the TRAINING episode clusters (pool 1's — "was the fear booked and is it still
    readable at all") and the PROBE cluster (the read pool's — "does it read *here*"). A harness that
    treated every failed read as an instrument refusal would silently convert a real carry failure
    into a dropped row; one that treated every failed read as a result would report a broken training
    as evidence against the claim. So:

    * needs on the episode clusters above the floor → the fear is booked, and whatever the probe
      cluster says is a MEASUREMENT: the row is clean either way, with ``pass`` the finding;
    * otherwise → the training left no readable fear, and the caller refuses the row.

    The ablated arm inverts it: there, anything readable means the detach LEAKED, which is an
    instrument failure and refuses.
    """
    g2 = fields.get("live_g2") or {}
    needs = g2.get("need_episode_clusters") or {}
    trained = g2.get("training_majority_cluster")
    probe = g2.get("probe_water_cluster")
    shore = g2.get("probe_shore_cluster")
    need_probe = g2.get("need_probe_cluster")
    water_fear = fields.get("water_fear")
    shore_fear = fields.get("shore_fear")
    booked = bool(needs) and all(float(n) > floor for n in needs.values())
    out: dict[str, Any] = {
        "arm": arm,
        "trained_node": trained,
        "read_node": probe,
        "read_shore_node": shore,
        "same_node": bool(probe) and probe == trained,
        "need_at_read_node": need_probe,
        "need_at_training_nodes": needs,
        "water_fear": water_fear,
        "shore_fear": shore_fear,
        "booked_at_training_nodes": booked,
        "why": None,
    }
    if arm == "cross_ablated":
        # anti-vacuity: NOTHING may be readable. A leak is an instrument failure, not a result.
        clean = water_fear == 0.0 and shore_fear == 0.0 and (need_probe or 0.0) == 0.0 and not booked
        out["refusal"] = (
            None
            if clean
            else (
                f"the ablated arm carries readable fear (water={water_fear} shore={shore_fear} "
                f"need={need_probe} training-node needs={needs}) — the subscriber detach did not hold; "
                "an instrument failure, never a null"
            )
        )
        out["pass"] = out["refusal"] is None
        return out
    if not booked:
        out["refusal"] = (
            f"training left NO readable fear on its own episode clusters (needs {needs}, strict floor {floor}) — "
            "a training failure must not be reported as a carry failure"
        )
        out["pass"] = False
        return out
    if probe and shore and probe == shore:
        out["refusal"] = (
            f"the read pool's shore and water encode to ONE live cluster ({probe}) — it cannot measure a carry"
        )
        out["pass"] = False
        return out
    out["refusal"] = None
    # The prereg's mechanism read, all four clauses: same node, fear at the cap, shore clean, and
    # the PRODUCTION read above the loop's strict floor.
    out["pass"] = bool(
        out["same_node"]
        and water_fear == FROZEN["fear_value_cap"]
        and shore_fear == 0.0
        and (need_probe is not None and float(need_probe) > floor)
    )
    if not out["pass"]:
        out["why"] = _node_gate_reason(out, floor)
    return out


def _node_gate_reason(out: dict[str, Any], floor: float) -> str:
    bits = []
    if not out["same_node"]:
        bits.append(f"read node {str(out['read_node'])[:8]} != trained node {str(out['trained_node'])[:8]}")
    if out["water_fear"] != FROZEN["fear_value_cap"]:
        bits.append(f"water fear {out['water_fear']} is not at the cap {FROZEN['fear_value_cap']}")
    if out["shore_fear"] != 0.0:
        bits.append(f"shore fear {out['shore_fear']} != 0 (specificity)")
    if out["need_at_read_node"] is None or float(out["need_at_read_node"]) <= floor:
        bits.append(f"production read {out['need_at_read_node']} <= the strict floor {floor}")
    return "; ".join(bits)


def quartile_medians(xs: list[float]) -> tuple[float | None, float | None]:
    if len(xs) < 4:
        return None, None
    q = max(1, len(xs) // 4)

    def med(v: list[float]) -> float:
        v = sorted(v)
        m = len(v) // 2
        return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2

    return med(xs[:q]), med(xs[-q:])


def campaign_drift(rows: list[dict[str, Any]], *, max_s: float) -> list[str]:
    """The SAME arm is the within-pool ceiling: its first-contact latency drifting across the
    campaign is the apparatus drifting, and it would move the cross arm the same way."""
    ordered = sorted((r for r in rows if r.get("refusal") is None and r.get("kind") == "row"), key=lambda r: r["ts"])
    lat = [
        float(r["first_contact"]["t_first_air"])
        for r in ordered
        if r.get("arm") == "same" and (r.get("first_contact") or {}).get("t_first_air") is not None
    ]
    a, b = quartile_medians(lat)
    if a is not None and b is not None and b - a > max_s:
        return [f"same-arm first-contact latency drifted {a:.2f}s → {b:.2f}s (> {max_s}s)"]
    return []


# ─────────────────────────── pure: the verdict ───────────────────────────


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def compute_verdict(rows: list[dict[str, Any]], *, campaign_id: str | None) -> dict[str, Any]:
    """The prereg's five gates over the committed rows. Pure; nothing here graduates anything."""
    in_campaign = [r for r in rows if campaign_id is None or r.get("campaign_id") == campaign_id]
    refused: list[str] = []
    # A later CLEAN row supersedes an earlier REFUSED row for the same (arm, seed) — what --resume
    # writes. The refusal is still NAMED; refusals are never dropped, and never counted as zeros.
    clean_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for r in sorted((r for r in in_campaign if r.get("kind") == "row"), key=lambda r: r["ts"]):
        key = (str(r["arm"]), int(r["seed"]))
        if r.get("refusal") is not None:
            refused.append(f"{r['arm']} seed {r['seed']}: {r['refusal']}")
            continue
        if key in clean_by_key:
            refused.append(f"duplicate clean (arm, seed) row {key} — pass --campaign-id to select one campaign")
            continue
        clean_by_key[key] = r
    clean: dict[str, list[dict[str, Any]]] = {a: [] for a in FROZEN["arms"]}
    for (arm, _seed), r in clean_by_key.items():
        clean.setdefault(arm, []).append(r)

    hashes = {(r.get("provenance") or {}).get("executed_git_hash") for r in in_campaign if "provenance" in r}
    if len(hashes) > 1:
        refused.append(f"rows span {len(hashes)} code hashes {sorted(map(str, hashes))} — one code hash per campaign")

    rates: dict[str, dict[str, Any]] = {}
    binaries: dict[str, list[float]] = {}
    for arm in FROZEN["arms"]:
        rs = clean.get(arm, [])
        fcs = [r.get("first_contact") or {} for r in rs]
        succ = [1.0 if fc.get("success") else 0.0 for fc in fcs]
        binaries[arm] = succ
        k, n = int(sum(succ)), len(succ)
        node = [1.0 if (r.get("node_gate") or {}).get("pass") else 0.0 for r in rs]
        kn = int(sum(node))
        lat = [float(fc["t_first_air"]) for fc in fcs if fc.get("t_first_air") is not None]
        rates[arm] = {
            "n": n,
            "successes": k,
            "rate": (k / n) if n else None,
            "wilson95": list(wilson_interval(k, n)) if n else None,
            "node_gate": {
                "hits": kn,
                "n": len(node),
                "rate": (kn / len(node)) if node else None,
                "wilson95": list(wilson_interval(kn, len(node))) if node else None,
            },
            "decision_dv_rate": (sum(1 for fc in fcs if fc.get("decision_dv")) / n) if n else None,
            "behavioural_dv_rate": (sum(1 for fc in fcs if fc.get("behavioural_dv")) / n) if n else None,
            "not_decisive": sum(1 for fc in fcs if fc.get("behavioural_dv") and not fc.get("decisive")),
            "executor_calls": sum(int(fc.get("escape_calls") or 0) + int(fc.get("flee_calls") or 0) for fc in fcs),
            "t_first_air_s": sorted(lat),
            "t_first_air_median": _median(lat),
        }

    g = FROZEN["gates"]
    incomplete = [
        f"{a}: {rates[a]['n']} clean rows < {FROZEN['arms'][a]}"
        for a in FROZEN["arms"]
        if rates[a]["n"] < FROZEN["arms"][a]
    ]
    incomplete.extend(campaign_drift(in_campaign, max_s=FROZEN["drift_max_s"]))
    replay_rows = [r for r in in_campaign if r.get("kind") == "replay" and r.get("refusal") is None]
    apparatus_rows = [r for r in in_campaign if r.get("kind") == "apparatus" and r.get("refusal") is None]
    if not replay_rows:
        incomplete.append("no replay row recorded for this campaign")
    if not apparatus_rows:
        incomplete.append("no apparatus citation row recorded for this campaign")

    cross, same, abl = rates["cross"], rates["same"], rates["cross_ablated"]
    replay = replay_consistency(replay_rows, cross["node_gate"]["rate"])
    checks = {
        # THE mechanism read, and the one gate that can fail for a reason the replay could not see.
        "NODE": cross["node_gate"]["rate"] is not None and cross["node_gate"]["rate"] >= g["node_min"],
        "CROSS": cross["rate"] is not None and cross["rate"] >= g["cross_min"],
        "SAME": same["rate"] is not None and same["rate"] >= g["same_min"],
        "ANTI_VACUITY": abl["n"] > 0 and abl["rate"] == 0.0 and abl["executor_calls"] == 0,
        "REPLAY": bool(replay["pass"]),
    }
    fisher = (
        fisher_one_sided_p(
            int(sum(binaries["cross"])),
            len(binaries["cross"]),
            int(sum(binaries["cross_ablated"])),
            len(binaries["cross_ablated"]),
        )
        if binaries["cross"] and binaries["cross_ablated"]
        else None
    )

    verdict, cause = "INCOMPLETE", None
    if not incomplete and not any(m.startswith("rows span") for m in refused):
        if all(checks.values()):
            verdict = "EARNED"
        elif (
            not checks["CROSS"]
            and cross["decision_dv_rate"] is not None
            and cross["decision_dv_rate"] >= g["cross_min"]
            and cross["behavioural_dv_rate"] is not None
            and cross["behavioural_dv_rate"] < g["cross_min"]
        ):
            verdict, cause = "INCOMPLETE", "actuation timing: the decision DV passes while the behavioural DV fails"
        elif not checks["REPLAY"]:
            verdict, cause = "INCOMPLETE", replay.get("note") or "the replay and the live NODE outcome disagree"
        else:
            verdict = "NULL"
            if not checks["SAME"]:
                cause = (
                    "the SAME arm did not reach its own ceiling — the apparatus, not the carry, is what this measured"
                )
            elif cross["not_decisive"]:
                cause = (
                    f"{cross['not_decisive']} cross surface(s) were won by a component other than the drive "
                    "(counted against)"
                )
            elif not checks["NODE"]:
                cause = "the pool-2 reading does not resolve to the trained node — the fear does not carry across pools here"
    if incomplete:
        cause = "; ".join(incomplete)
    return {
        "_format_version": "1.0",
        "kind": "exp62_verdict",
        "campaign_id": campaign_id,
        "what_this_is": "rung A: does the shipped body carry the fear across pools — bounded to this apparatus class",
        "refused": refused,
        "n_clean": {a: rates[a]["n"] for a in FROZEN["arms"]},
        "rates": rates,
        "replay": replay,
        "fisher_cross_vs_ablated": fisher,
        "checks": checks,
        "gates": g,
        "verdict": verdict,
        "incomplete_cause": cause,
    }


def replay_consistency(replay_rows: list[dict[str, Any]], live_node_rate: float | None) -> dict[str, Any]:
    """The committed replay predicted a cross-pool HIT on the BUILT geometry; the live NODE gate
    must agree. Offline-hit / live-miss is INCOMPLETE with the disagreement NAMED — the replay was
    wrong about something live, and that finding is worth more than the rung. A predicted hit that
    hits is the RESULT."""
    if not replay_rows:
        return {"rule": "a replay row must be present", "measured": None, "pass": False, "note": None}
    last = replay_rows[-1]
    predicted = bool(last.get("predicted_cross_hit"))
    live = live_node_rate is not None and live_node_rate >= FROZEN["gates"]["node_min"]
    agree = predicted == live
    return {
        "rule": "the committed replay's prediction and the live NODE outcome agree",
        "measured": {
            "predicted_cross_hit": predicted,
            "cross_pool_cosine": last.get("cross_pool_cosine"),
            "live_node_hit": live,
            "live_node_rate": live_node_rate,
        },
        "pass": agree,
        "note": None
        if agree
        else (
            "offline predicted a HIT and the live read MISSED — the replay was wrong about something live "
            "(light, time, the EC's live thresholds); name it in §Outcome"
            if predicted
            else "offline predicted a MISS and the live read HIT — the replay understated the body; name it in §Outcome"
        ),
    }


# ─────────────────────────── the replay row ───────────────────────────


def _dist3(a: Any, b: Any) -> float:
    ax, ay, az = (a["x"], a["y"], a["z"]) if isinstance(a, dict) else (a[0], a[1], a[2])
    bx, by, bz = (b["x"], b["y"], b["z"]) if isinstance(b, dict) else (b[0], b[1], b[2])
    return math.sqrt((float(ax) - float(bx)) ** 2 + (float(ay) - float(by)) ** 2 + (float(az) - float(bz)) ** 2)


def replay_prediction(rec1: dict[str, Any], rec2: dict[str, Any], *, script: Path = REPLAY_SCRIPT) -> dict[str, Any]:
    """Re-run the committed offline replay on the geometry that was ACTUALLY BUILT.

    The prereg's 0.999 is for the *recommended* floor y 95; the pools on the rig are where the
    operator's clearance guard put them, so a copied constant would be a prediction about a pool
    that does not exist. This loads the committed replay module by path (it is the artifact under
    citation, not a library) and overrides only pool 2's two place absolutes — exactly its own
    "stacked pool 2" row.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("exp62_cross_pool_replay", script)
    if spec is None or spec.loader is None:
        raise Refusal(f"cannot load the committed replay at {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    base = mod.cos(mod.embed(mod.SHORE), mod.embed(mod.SUB))
    if abs(base - mod.LIVE) > 0.002:
        raise Refusal(
            f"the replay no longer reproduces its own live record ({base:.4f} v {mod.LIVE:.4f}) — "
            "the sensor→basis mapping moved; the prediction is not citable"
        )
    spawn1, spawn2 = rec1.get("world_spawn"), rec2.get("world_spawn")
    if spawn1 is None or spawn2 is None or json.dumps(spawn1, sort_keys=True) != json.dumps(spawn2, sort_keys=True):
        raise Refusal(
            "the two anchor records do not carry ONE stamped world_spawn "
            f"({spawn1!r} v {spawn2!r}) — distance_from_spawn is not derivable"
        )
    y1, y2 = float(rec1["submerged"][1]), float(rec2["submerged"][1])
    d1, d2 = _dist3(spawn1, rec1["submerged"]), _dist3(spawn1, rec2["submerged"])
    s1 = mod.embed(mod.pool2(mod.SUB, y=y1, d=d1))
    s2 = mod.embed(mod.pool2(mod.SUB, y=y2, d=d2))
    shore2 = mod.embed(mod.pool2(mod.SHORE, y=float(rec2["shore"][1]), d=_dist3(spawn1, rec2["shore"])))
    cross = mod.cos(s1, s2)
    own2 = mod.cos(shore2, s2)
    th = float(FROZEN["cosine_threshold"])
    return {
        "replay_script": str(Path(script).resolve()),
        "reproduces_live_record": round(base, 4),
        "threshold": th,
        "pool1": {"submerged_y": y1, "distance_from_spawn": round(d1, 2)},
        "pool2": {"submerged_y": y2, "distance_from_spawn": round(d2, 2)},
        "cross_pool_cosine": round(cross, 4),
        "pool2_own_shore_water_cosine": round(own2, 4),
        # a HIT = pool 2's submerged vector completes into pool 1's trained water cluster
        "predicted_cross_hit": bool(cross >= th),
        # …and pool 2 must still separate its OWN shore from its OWN water, or the gate is vacuous
        "pool2_separates_internally": bool(own2 < th),
    }


# ─────────────────────────── the live campaign ───────────────────────────


class Exp62Campaign:
    """One row = one fresh agent: trained at pool 1, read and contacted at the arm's pool."""

    def __init__(
        self,
        args: argparse.Namespace,
        geoms: dict[str, dict[str, Any]],
        *,
        provenance: dict[str, Any],
        out_path: Path,
        campaign_id: str,
    ) -> None:
        self.args = args
        self.geoms = geoms
        self.provenance = provenance
        self.out_path = out_path
        self.campaign_id = campaign_id
        self.rcon = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
        fz = FROZEN["exp60"]
        # ONE cap for both pools: the window must sit below BOTH pain edges, or the arms' latencies
        # are not comparable and the cap difference lands inside the contrast.
        edges = {p: float(geoms[p]["measured"]["t_pain_edge_min_s"]) for p in ("pool1", "pool2")}
        self.pain_edges = edges
        self.probe_cap_s = min(edges.values()) - fz["probe_cap_margin_s"]
        self.train_cap_s = float(geoms["pool1"]["measured"]["t_damage_onset_min_s"]) - fz["train_cap_margin_s"]
        self.workdir = Path(args.workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

    def base_row(self, kind: str, arm: str, seed: int | str) -> dict[str, Any]:
        return {
            "_format_version": "1.0",
            "kind": kind,
            "ts": time.time(),
            "campaign_id": self.campaign_id,
            "arm": arm,
            "seed": seed,
            "provenance": self.provenance,
            "refusal": None,
        }

    def write(self, row: dict[str, Any]) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def apparatus_citation(self) -> dict[str, Any]:
        """Both pools' committed gate-(ii) records, cited not re-run (they are their own script).
        A missing or failed gate stops the campaign before a single agent is built."""
        row = self.base_row("apparatus", "-", "-")
        try:
            drift = frozen_matches()
            if drift:
                raise Refusal(f"frozen constants drifted from Exp 60/61 on {drift}")
            cited = []
            for label, path in zip(("pool1", "pool2"), self.args.gate_record, strict=True):
                rec = json.loads(Path(path).expanduser().read_text())
                entry = {
                    "pool": label,
                    "record": str(path),
                    "cos_a4": (rec.get("cosine") or {}).get("a4_gained"),
                    "threshold": (rec.get("cosine") or {}).get("threshold"),
                    "run_gate_pass": bool((rec.get("run_gate") or {}).get("pass")),
                    "verdict": rec.get("verdict"),
                }
                if not entry["run_gate_pass"]:
                    raise Refusal(f"{label}'s committed gate (ii) does not PASS ({path})")
                cited.append(entry)
            row["gate_ii"] = cited
            row["caps"] = {
                "probe_cap_s": round(self.probe_cap_s, 3),
                "train_cap_s": round(self.train_cap_s, 3),
                "pain_edges_s": self.pain_edges,
                "note": "one cap for both pools = min(pain edge) - margin",
            }
            row["measured"] = {p: self.geoms[p]["measured"] for p in ("pool1", "pool2")}
            row["pool_ids"] = {p: self.geoms[p].get("pool_id") for p in ("pool1", "pool2")}
        except (Refusal, OSError, ValueError) as exc:
            row["refusal"] = str(exc)
            print(f"REFUSED apparatus citation: {exc}")
        self.write(row)
        return row

    def replay_row(self) -> dict[str, Any]:
        row = self.base_row("replay", "-", "-")
        try:
            row.update(replay_prediction(self.geoms["pool1"], self.geoms["pool2"]))
            print(
                f"replay on the BUILT geometry: cross-pool cos {row['cross_pool_cosine']} "
                f"(threshold {row['threshold']}) -> predicted {'HIT' if row['predicted_cross_hit'] else 'MISS'}; "
                f"pool 2 separates internally: {row['pool2_separates_internally']}"
            )
        except (Refusal, OSError, ValueError, KeyError, AttributeError) as exc:
            row["refusal"] = str(exc)
            print(f"REFUSED replay row: {exc}")
        self.write(row)
        return row

    def row(self, arm: str, seed: int) -> dict[str, Any]:
        read_pool = FROZEN["read_pool"][arm]
        row = self.base_row("row", arm, seed)
        row.update({"train_pool": FROZEN["train_pool"], "read_pool": read_pool})
        home = self.workdir / f"{arm}_{seed}"
        agent_id = f"exp62_{arm}_{seed}"
        aut = pump = trial = None
        try:
            aut, encoder, pump = build_aut(self.args, agent_id=agent_id, home=home)
            trial = WaterTrial(
                aut=aut,
                rcon=self.rcon,
                username=self.args.username,
                geom=self.geoms[FROZEN["train_pool"]],
                frozen=FROZEN["exp60"],
                probe_cap_s=self.probe_cap_s,
                train_cap_s=self.train_cap_s,
                persistence_dir=home,
                agent_id=agent_id,
                encoder=encoder,
                settle_guard=FROZEN["settle_guard"],
            )
            trial.attach_instruments()  # ONE attach; use_geometry moves the pool, not the spy
            if arm == "cross_ablated":
                detached = _detach_fear_subscriber(aut)
                row["detached_subscribers"] = detached
                if detached != 1:
                    raise Refusal(f"the ablation detached {detached} subscriber(s), expected exactly 1")

            # 1. preflights, at the TRAINING pool (Exp 60's set, unchanged)
            row["fingerprint"] = trial.check_fingerprint(FROZEN["exp60"]["usable_oxygen_max"])
            row["bridge_state_interval_s"] = trial.check_bridge()
            row["loop_liveness_ticks"] = trial.check_liveness()
            trial.check_gamerules()
            shore_pre, water_pre = trial.check_clusters_distinct()
            row["preflight_clusters"] = {"shore": shore_pre, "water": water_pre}
            trial.resolve_tools()
            trial.rescue("ready")
            trial.deaths0 = trial.deaths()
            trial.check_no_positive_escape_link()

            # 2. train at pool 1 — propose-only; nothing executes, so no causal link is booked
            row["training"], episode_clusters = trial.train()
            trial.check_no_positive_escape_link()
            row["positive_escape_links_after_training"] = trial.positive_escape_links()

            # 3. the SAME agent moves pools
            previous = trial.use_geometry(self.geoms[read_pool])
            row["previous_geometry_pool_id"] = (previous or {}).get("pool_id")

            # 4. the NODE gate at the read pool, loop OFF — THE RESULT
            try:
                fields = trial.live_g2(FROZEN["g2_arm"][arm], episode_clusters, water_pre)
                g2_raised = None
            except Refusal as exc:
                # A failed read here is a MEASUREMENT, not automatically an instrument failure —
                # classify_g2 decides which, from the needs on the TRAINING clusters.
                fields = getattr(exc, "partial", None) or {}
                g2_raised = str(exc)
                if not fields:
                    raise
            row.update(fields)
            node = classify_g2(fields, arm=arm, floor=FROZEN["read_floor"])
            node["g2_raised"] = g2_raised
            row["node_gate"] = node
            if node["refusal"]:
                raise Refusal(node["refusal"], partial={"node_gate": node})

            # 5. FIRST CONTACT at the read pool, loop LIVE, US-free — the behavioural consequence
            trial.deaths0 = trial.deaths()
            with C.RecommendCapture() as cap:
                n0 = len(cap.events)
                pl = trial.placement("first-contact", window_label=f"{arm}-{seed}")
                events = [{**dict(e.get("data", {})), "t": e.get("t")} for e in cap.events[n0:]]
            # the EXECUTED proposal: the first escape-best event that PASSED the gate
            escape_events = [
                e
                for e in events
                if str(e.get("best_tool", "")).endswith(ESCAPE_SUFFIX) and e.get("passed_gate") is True
            ]
            fc = first_contact_outcome(
                pl, cap_s=self.probe_cap_s, decision_event=escape_events[0] if escape_events else None
            )
            fc["placement"] = pl
            fc["decision_events"] = events[:8]
            row["first_contact"] = fc
            if fc["refusal"]:
                raise Refusal(fc["refusal"])
            row["positive_escape_links_after_contact"] = trial.positive_escape_links()
            trial.check_death_cap()
            print(
                f"{arm} {seed}: NODE={node['pass']} (read {str(node['read_node'])[:8]} v trained "
                f"{str(node['trained_node'])[:8]}, need {node['need_at_read_node']}) | contact success={fc['success']} "
                f"(decision={fc['decision_dv']}, air={fc['behavioural_dv']}, decisive={fc['decisive']}) "
                f"t_air={fc['t_first_air']}"
            )
        except (Refusal, InstrumentError) as exc:
            row.update(getattr(exc, "partial", None) or {})
            row["refusal"] = str(exc)
            print(f"REFUSED {arm} seed {seed}: {exc}")
        finally:
            if trial is not None:
                try:
                    trial.detach_instruments()
                    trial.final_rescue()
                except Exception as exc:  # noqa: BLE001 — teardown must never hide the row
                    print(f"WARNING: teardown raised for {arm} {seed}: {exc!r}")
            if aut is not None and pump is not None:
                try:
                    close_and_stage(aut, pump, None)
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: closing {arm} {seed} raised: {exc!r}")
        self.write(row)
        return row


# ─────────────────────────── wiring ───────────────────────────


def load_geoms(pool1: str, pool2: str) -> dict[str, dict[str, Any]]:
    geoms: dict[str, dict[str, Any]] = {}
    for label, path in (("pool1", pool1), ("pool2", pool2)):
        rec = json.loads(Path(path).expanduser().read_text())
        if "measured" not in rec:
            raise Refusal(f"{label}'s record ({path}) carries no `measured` block — run exp60_water_check on it first")
        geoms[label] = rec
    if geoms["pool1"].get("pool_id") == geoms["pool2"].get("pool_id"):
        raise Refusal(
            f"both anchor records carry pool_id {geoms['pool1'].get('pool_id')!r} — "
            "one of them was overwritten by a build; rebuild with --anchor-file and --pool-id"
        )
    return geoms


def existing_clean(out_path: Path, campaign_id: str) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not out_path.is_file():
        return done
    for ln in out_path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("kind") == "row" and r.get("campaign_id") == campaign_id and r.get("refusal") is None:
            done.add((str(r["arm"]), int(r["seed"])))
    return done


def interleaved_plan(arms: list[str], *, rows: int | None) -> list[tuple[str, int]]:
    """Seed-by-seed across the arms, so campaign drift hits them EQUALLY — the cross/same contrast
    is the whole result, and a block design would put time inside it. (pure)"""
    seeds = {a: FROZEN["seeds"][a][:rows] if rows else FROZEN["seeds"][a] for a in arms}
    plan: list[tuple[str, int]] = []
    for i in range(max((len(s) for s in seeds.values()), default=0)):
        for arm in arms:
            if i < len(seeds[arm]):
                plan.append((arm, seeds[arm][i]))
    return plan


def _out_path(arg: str, args: argparse.Namespace) -> Path:
    p = Path(arg)
    return evidence_out_paths_or_exit(
        REPO_ROOT,
        [str(p if p.is_absolute() else REPO_ROOT / p)],
        write_experiment_results=args.write_experiment_results,
        allow_dirty=args.allow_dirty,
    )[0]


def _provenance_or_none(out_path: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    import maxim

    try:
        return in_process_code_provenance(REPO_ROOT, maxim.__file__, out_path=out_path, allow_dirty=args.allow_dirty)
    except (DirtyTreeError, ProvenanceError) as exc:
        print(f"[FAIL] provenance: {exc}")
        return None


def cmd_replay(args: argparse.Namespace) -> int:
    """The replay row, written BEFORE the campaign so its prediction cannot be tuned to the result."""
    out_path = _out_path(args.out, args)
    provenance = _provenance_or_none(out_path, args)
    if provenance is None:
        return 3
    try:
        geoms = load_geoms(args.pool1_anchor, args.pool2_anchor)
    except (Refusal, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 4
    camp = Exp62Campaign(args, geoms, provenance=provenance, out_path=out_path, campaign_id=args.campaign_id)
    return 0 if camp.replay_row().get("refusal") is None else 4


def cmd_run(args: argparse.Namespace) -> int:
    out_path = _out_path(args.out, args)
    provenance = _provenance_or_none(out_path, args)
    if provenance is None:
        return 3
    if args.resume and not args.campaign_id:
        print("[FAIL] --resume requires --campaign-id")
        return 2
    if len(args.gate_record) != 2:
        print("[FAIL] pass --gate-record twice: pool 1's committed gate-(ii) record, then pool 2's")
        return 2
    try:
        geoms = load_geoms(args.pool1_anchor, args.pool2_anchor)
    except (Refusal, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 4
    campaign_id = args.campaign_id or uuid.uuid4().hex[:12]
    camp = Exp62Campaign(args, geoms, provenance=provenance, out_path=out_path, campaign_id=campaign_id)
    if camp.apparatus_citation().get("refusal") is not None:
        return 4  # nothing measured after this would be trustworthy
    done = existing_clean(out_path, campaign_id) if args.resume else set()
    arms = [a for a in ARM_ORDER if not args.only or a in args.only]
    print(
        f"campaign {campaign_id}: workdir {camp.workdir}; probe cap {camp.probe_cap_s:.2f}s "
        f"(min of {camp.pain_edges}); train cap {camp.train_cap_s:.2f}s; {len(done)} clean rows already"
    )
    exit_code = 0
    prev_cwd = os.getcwd()
    try:
        for arm, seed in interleaved_plan(arms, rows=args.rows):
            if (arm, seed) in done:
                print(f"skip {arm} {seed} (clean row present)")
                continue
            row_dir = camp.workdir / f"{arm}_{seed}"
            row_dir.mkdir(parents=True, exist_ok=True)
            # The loop writes cwd-relative (`init_prefetcher(base_path=os.getcwd())`, `FileSystemEnv`).
            os.chdir(row_dir)
            print(f"\n=== {arm} seed {seed} (train {FROZEN['train_pool']} -> read {FROZEN['read_pool'][arm]}) ===")
            if camp.row(arm, seed).get("refusal") is not None:
                exit_code = 4
    finally:
        os.chdir(prev_cwd)
        camp.rcon.close()
    print(f"\ncampaign {campaign_id} -> {out_path}")
    return exit_code


def cmd_verdict(args: argparse.Namespace) -> int:
    rows = [json.loads(ln) for ln in Path(args.data).expanduser().read_text().splitlines() if ln.strip()]
    v = compute_verdict(rows, campaign_id=args.campaign_id)
    v["data"] = str(args.data)
    print(json.dumps({k: v[k] for k in v if k != "gates"}, indent=2, default=str))
    print(f"\nVERDICT: {v['verdict']}" + (f" ({v['incomplete_cause']})" if v.get("incomplete_cause") else ""))
    if args.json:
        out = _out_path(args.json, args)
        provenance = _provenance_or_none(out, args)
        if provenance is None:
            return 3
        v["provenance"] = provenance
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(v, indent=2, default=str) + "\n")
        print(f"[verdict] wrote -> {out}")
    return 0 if v["verdict"] == "EARNED" else 1


def _evidence_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--write-experiment-results", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")


def _anchor_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pool1-anchor", default=str(Path.home() / ".maxim" / "exp60_water_classroom.json"))
    p.add_argument("--pool2-anchor", default=str(Path.home() / ".maxim" / "exp62_pool2_water_classroom.json"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("replay", help="write the replay row on the BUILT geometry (before the campaign)")
    rp.add_argument("--campaign-id", required=True)
    rp.add_argument("--out", default="docs/experiments/data/exp62_rows.jsonl")
    _anchor_args(rp)
    _evidence_args(rp)
    rp.set_defaults(func=cmd_replay, workdir=".", gate_record=[], rcon_host="", rcon_port=0, rcon_password="")

    r = sub.add_parser("run", help="the live campaign (arms interleaved seed by seed)")
    r.add_argument("--campaign-id", default=None)
    r.add_argument("--out", default="docs/experiments/data/exp62_rows.jsonl")
    _anchor_args(r)
    r.add_argument(
        "--gate-record", action="append", default=[], help="committed gate-(ii) record; pass TWICE, pool 1 first"
    )
    r.add_argument("--workdir", required=True)
    r.add_argument("--only", action="append", default=[], help="run only these arms")
    r.add_argument("--rows", type=int, default=None, help="cap the seeds per arm (a dry run uses 1)")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--bridge-host", default="127.0.0.1")
    r.add_argument("--bridge-port", type=int, default=25567)
    r.add_argument("--rcon-host", default="127.0.0.1")
    r.add_argument("--rcon-port", type=int, default=25575)
    r.add_argument("--rcon-password", required=True)
    r.add_argument("--username", default="maxim")
    _evidence_args(r)
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("verdict", help="the five frozen gates, pure over committed rows")
    v.add_argument("--data", required=True)
    v.add_argument("--campaign-id", default=None)
    v.add_argument("--json", default=None)
    _evidence_args(v)
    v.set_defaults(func=cmd_verdict)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
