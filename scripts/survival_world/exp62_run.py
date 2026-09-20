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

**Cite pool 2's RECONNECT gate record.** Its first probe read `light_level` 1.0 and the reconnect
read 0.0 — the prereg's named "stale-light read at a freshly filled box". Both records carry
`run_gate.pass: true`, so a gate-pass check alone waves the stale one through, and the real
cross-pool cosine would be 0.5878 (a MISS) while the synthetic replay prediction still said 0.9995
(a HIT), because light is unrepresentable in that construction. The harness now refuses the pair
outright (`cite_gate_records`), and computes the cosine the records' own vectors imply beside the
prediction — but the recipe above names the right record.

**One probe cap for both pools.** The cap is ``min(pool 1, pool 2 pain edge) − margin``, so the
window is identical in every arm and sits below BOTH pools' pain edges. A per-pool cap would make
the arms' latencies incomparable and put the cap difference inside the contrast.

Run ON the bridge box (server + bridge from current main at ONE code hash — no ``git pull`` between
the first and last row; bridge at ``--state_interval_ms=100``; no second player)::

    export PYTHONPATH="$PWD/src"
    python scripts/survival_world/exp62_run.py replay --campaign-id <id> --write-experiment-results \\
        --gate-record docs/experiments/data/exp60_geometry_2026-09-15b.json \\
        --gate-record docs/experiments/data/exp62_pool2_geometry_reconnect.json
    python scripts/survival_world/exp62_run.py run --campaign-id <id> --rcon-password '<pw>' \\
        --workdir ~/exp62_work \\
        --gate-record docs/experiments/data/exp60_geometry_2026-09-15b.json \\
        --gate-record docs/experiments/data/exp62_pool2_geometry_reconnect.json --write-experiment-results
    python scripts/survival_world/exp62_run.py verdict --data docs/experiments/data/exp62_rows.jsonl \\
        --campaign-id <id> --json docs/experiments/data/exp62_verdict.json --write-experiment-results
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    quartile_medians,
    wilson_interval,
)
from survival_world.exp61_run import FROZEN as FROZEN61  # noqa: E402
from survival_world.r3_run import bootstrap_median_ci  # noqa: E402
from survival_world.water_trial import Refusal, WaterTrial, _detach_fear_subscriber, _median  # noqa: E402

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
    },
    # Reported, never gated: the prereg's statistic is the INTERVAL, not p ("both fear arms are
    # predicted at the ceiling"), so Fisher is recorded where a contrast exists and decides nothing.
    "reported_not_gated": {"alpha": 0.05},
    # LITERAL copies, exactly as Exp 61 copies Exp 60's (its FROZEN comment says why): a later edit
    # to Exp 60's or Exp 61's block must FAIL `frozen_matches()`, never be inherited silently. A
    # deep copy taken at import would take that edit with it and the equality could never fail.
    "settle_guard": {"is_raining": 0.0, "nearest_player_dist": 64.0},
    "exp60": {
        "K_usable_episodes": 10,
        "placements_per_probe": 6,
        "shore_roam_s": 10.0,
        "probe_cap_margin_s": 0.75,
        "train_cap_margin_s": 1.0,
        "loop_warm_s": 1.0,
        "bridge_state_interval_max_s": 0.15,
        "loop_liveness_min_ticks": 4,
        "loop_liveness_s": 3.0,
        "usable_oxygen_max": 12.0,
        "usable_pain_intensity_min": 1.0,
        "loop_hz": 4.0,
        "death_cap": 2,
        "specificity_ratio": 0.2,
        "fingerprint": {
            "cluster_fear_alpha": 0.5,
            "max_cluster_fear": 1.0,
            "cluster_fear_threshold": 0.5,
            "cluster_fear_failure_modes": ["drive:health", "drive:oxygen"],
            "encoder_pattern_threshold": 0.85,
            "substrate_explore_bonus_weight": 0.0,
            "oxygen_drive": {"set_point": 20.0, "comfort_band": 6.0},
            "sensor_ranges": {"is_in_water": [-1.0, 1.0], "oxygen": [0.0, 40.0], "saturation": [0.0, 20.0]},
        },
    },
}
ARM_ORDER = ("cross", "same", "cross_ablated")


def frozen_matches() -> list[str]:
    """Every borrowed constant, checked against its source — Exp 61's, and through it Exp 60's.

    The blocks above are LITERALS, so each comparison here is a real one: a one-sided edit in any
    of the three harnesses shows up as drift instead of propagating silently into a live campaign.
    """
    drift = list(exp60_frozen_matches())
    if FROZEN["exp60"] != FROZEN61["exp60"]:
        drift.append("exp60 block differs from exp61's")
    if FROZEN["settle_guard"] != FROZEN61["settle_guard"]:
        drift.append("settle_guard differs from exp61's")
    for k in ("fear_value_cap", "read_floor", "drift_max_s"):
        if FROZEN[k] != FROZEN61.get(k, FROZEN61["gates"].get(k)):
            drift.append(f"{k} differs from exp61's")
    # `cosine_threshold` has TWO live sources it could drift from — the encoder's pattern threshold
    # (enforced live by `check_fingerprint`) and the replay module's own TH. Both are checked.
    if FROZEN["cosine_threshold"] != FROZEN["exp60"]["fingerprint"]["encoder_pattern_threshold"]:
        drift.append("cosine_threshold differs from the encoder pattern threshold the fingerprint pins")
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
    booked = bool(needs) and all(n is not None and float(n) > floor for n in needs.values())
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
        # Anti-vacuity: NOTHING may be readable, so this arm HAS no node gate — it has an ablation
        # check. `pass` stays None: a True here would be published as "the ablated arm's reading
        # resolved to the trained node in 3/3", which is the opposite of what it means.
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
        out["ablation_held"] = out["refusal"] is None
        out["pass"] = None
        return out
    if not booked:
        out["refusal"] = (
            f"training left NO readable fear on its own episode clusters (needs {needs}, strict floor {floor}) — "
            "a training failure must not be reported as a carry failure"
        )
        out["pass"] = False
        return out
    # `encode_world_cluster` returns None on an encode failure or a stale snapshot (it never raises
    # into the loop). Scoring that as a node MISS would publish an instrument failure as the
    # finding — and the one-cluster clause below is skipped precisely when `probe` is None. Exp 60's
    # `check_clusters_distinct` refuses on a missing cluster; this must not invert it.
    if not probe:
        out["refusal"] = "the read pool produced no live world cluster — an instrument failure, not a carry miss"
        out["pass"] = False
        return out
    if not shore:
        out["refusal"] = (
            "the read pool produced no live SHORE cluster — specificity cannot be measured, and a "
            "shore fear of 0 would credit the clause vacuously"
        )
        out["pass"] = False
        return out
    if probe == shore:
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
        and shore is not None  # specificity was MEASURED, not inherited from a missing cluster
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


def compute_verdict(rows: list[dict[str, Any]], *, campaign_id: str | None) -> dict[str, Any]:
    """The prereg's five gates over the committed rows. Pure; nothing here graduates anything."""
    in_campaign = [r for r in rows if campaign_id is None or r.get("campaign_id") == campaign_id]
    refused: list[str] = []
    duplicates: list[str] = []
    # A later CLEAN row supersedes an earlier REFUSED row for the same (arm, seed) — what --resume
    # writes. The refusal is still NAMED; refusals are never dropped, and never counted as zeros.
    clean_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for r in sorted((r for r in in_campaign if r.get("kind") == "row"), key=lambda r: r["ts"]):
        key = (str(r["arm"]), int(r["seed"]))
        if r.get("refusal") is not None:
            refused.append(f"{r['arm']} seed {r['seed']}: {r['refusal']}")
            continue
        if key in clean_by_key:
            # NOT a footnote: the first row silently supplies every number while the second is
            # invisible to all of them. A campaign restarted without --resume under one id lands
            # 24 clean cross rows and reports run 1's rate as if it were the campaign's.
            duplicates.append(f"duplicate clean (arm, seed) row {key} — pass --campaign-id to select one campaign")
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
        # an arm whose `pass` is None has NO node gate (the ablated arm) — it is not a zero
        node = [
            1.0 if (r.get("node_gate") or {}).get("pass") else 0.0
            for r in rs
            if (r.get("node_gate") or {}).get("pass") is not None
        ]
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
                "not_applicable": len(node) == 0 and bool(rs),
            },
            "ablation_held": sum(1 for r in rs if (r.get("node_gate") or {}).get("ablation_held")),
            "decision_dv_rate": (sum(1 for fc in fcs if fc.get("decision_dv")) / n) if n else None,
            "behavioural_dv_rate": (sum(1 for fc in fcs if fc.get("behavioural_dv")) / n) if n else None,
            "not_decisive": sum(1 for fc in fcs if fc.get("behavioural_dv") and not fc.get("decisive")),
            # EVERY executor call the spy recorded — "the apparatus does not surface an agent by
            # itself" is a claim about the executor, not about two affordances of it
            "executor_calls": sum(len((fc.get("placement") or {}).get("calls") or []) for fc in fcs),
            "escape_calls": sum(int(fc.get("escape_calls") or 0) for fc in fcs),
            "t_first_air_s": sorted(lat),
            "t_first_air_median": _median(lat) if lat else None,
            # the prereg's INFORMATIVE number: "both fear arms are predicted at the ceiling, so the
            # statistic is the interval, not p". Fisher below is recorded, and gates nothing.
            "t_first_air_median_ci95": list(bootstrap_median_ci(lat)) if len(lat) >= 2 else None,
        }

    g = FROZEN["gates"]
    incomplete = [
        f"{a}: {rates[a]['n']} clean rows < {FROZEN['arms'][a]}"
        for a in FROZEN["arms"]
        if rates[a]["n"] < FROZEN["arms"][a]
    ]
    incomplete.extend(campaign_drift(list(clean_by_key.values()), max_s=FROZEN["drift_max_s"]))
    replay_rows = [r for r in in_campaign if r.get("kind") == "replay" and r.get("refusal") is None]
    apparatus_rows = [r for r in in_campaign if r.get("kind") == "apparatus" and r.get("refusal") is None]
    if not replay_rows:
        incomplete.append("no replay row recorded for this campaign")
    if not apparatus_rows:
        incomplete.append("no apparatus citation row recorded for this campaign")
    incomplete.extend(duplicates)
    refused.extend(duplicates)
    unknown = {str(r.get("arm")) for r in in_campaign if r.get("kind") == "row"} - set(FROZEN["arms"])
    if unknown:
        incomplete.append(f"rows carry unknown arm(s) {sorted(unknown)} — they are in no rate and no gate")

    cross, same, abl = rates["cross"], rates["same"], rates["cross_ablated"]
    replay = replay_consistency(replay_rows, cross["node_gate"]["rate"])
    checks = {
        # THE mechanism read, and the one gate that can fail for a reason the replay could not see.
        "NODE": cross["node_gate"]["rate"] is not None and cross["node_gate"]["rate"] >= g["node_min"],
        "CROSS": cross["rate"] is not None and cross["rate"] >= g["cross_min"],
        "SAME": same["rate"] is not None and same["rate"] >= g["same_min"],
        "ANTI_VACUITY": abl["n"] > 0
        and abl["rate"] == 0.0
        and abl["executor_calls"] == 0
        and abl["ablation_held"] == abl["n"],
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
        elif not checks["ANTI_VACUITY"]:
            # An apparatus that surfaces an agent by itself, or an ablation that leaked, is a
            # statement about the INSTRUMENT. Calling it NULL would assert a mechanism result on an
            # apparatus that cannot support one — the same shape as a failed SAME arm.
            verdict, cause = (
                "INCOMPLETE",
                f"anti-vacuity failed: the ablated arm surfaced {abl['successes']}/{abl['n']} with "
                f"{abl['executor_calls']} executor call(s), ablation held in {abl['ablation_held']}/{abl['n']} "
                "row(s) — the apparatus, not the carry, is what this measured",
            )
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
            else:
                cause = (
                    f"the cross arm's first-contact rate {cross['rate']} is below the frozen {g['cross_min']} "
                    "while its node gate holds — the fear reads at pool 2 but does not drive the escape there"
                )
    if incomplete:
        cause = "; ".join(incomplete)
    elif verdict != "EARNED" and not cause:
        cause = "; ".join(refused) or "no cause recorded"
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


# The constants the cross-pool cosine actually rests on. Both carry `rest: null` by design and ride
# at (or near) full gain, so a difference in either ROTATES the vector far more than the place
# absolutes do — which is the whole contrast. The prereg gates them equal for that reason.
CONTEXT_SENSORS = ("light_level", "time_of_day")


def context_constants_match(rec1: dict[str, Any], rec2: dict[str, Any]) -> dict[str, Any]:
    """Do the two pools' committed gate-(ii) records agree on light and time? (pure)

    The prereg's §Apparatus requires it in as many words — "Light and time GATED equal at pool 2's
    shore and floor (the full-weight constants; a stale-light read at a freshly filled box is the
    known failure)". That failure is not hypothetical: pool 2's FIRST probe read `light_level` 1.0
    and its reconnect read 0.0, and both records carry `run_gate.pass: true`, so a gate-pass check
    alone waves the stale one through. Citing it would put the real cross-pool cosine at 0.588 (a
    MISS) while the synthetic replay prediction still said 0.999 (a HIT) — the harness would have
    measured the LIGHT contrast and called it the pool.
    """
    v1 = {s["sensor"]: s for s in rec1.get("per_sensor") or []}
    v2 = {s["sensor"]: s for s in rec2.get("per_sensor") or []}
    rows, mismatches = [], []
    for sensor in CONTEXT_SENSORS:
        a, b = v1.get(sensor), v2.get(sensor)
        if a is None or b is None:
            mismatches.append(f"{sensor} absent from {'pool 1' if a is None else 'pool 2'}'s record")
            continue
        row = {
            "sensor": sensor,
            "pool1": {"shore": a.get("v_safe"), "submerged": a.get("v_dark")},
            "pool2": {"shore": b.get("v_safe"), "submerged": b.get("v_dark")},
        }
        rows.append(row)
        for where, key in (("shore", "v_safe"), ("submerged", "v_dark")):
            if a.get(key) != b.get(key):
                mismatches.append(f"{sensor} at the {where}: pool 1 {a.get(key)} v pool 2 {b.get(key)}")
    return {"rows": rows, "mismatches": mismatches, "match": not mismatches and len(rows) == len(CONTEXT_SENSORS)}


def gate_record_matches_pool(rec: dict[str, Any], anchor: dict[str, Any]) -> str | None:
    """Is this gate-(ii) record the one for THIS pool? (pure)

    The records carry no `pool_id`, so binding them by CLI argument order alone makes a swapped pair
    undetectable — and a swapped pair passes every other check in this harness. The probe's own
    `y_altitude` is the discriminator: pool 1 and pool 2 sit at different heights by construction.
    """
    row = next((s for s in rec.get("per_sensor") or [] if s["sensor"] == "y_altitude"), None)
    if row is None or row.get("v_dark") is None:
        return "the gate record carries no y_altitude reading — it cannot be bound to a pool"
    sensed = float(row["v_dark"]) * 128.0  # the encoder's y normalization, inverted
    built = float(anchor["submerged"][1])
    if abs(sensed - built) > 1.5:
        return f"the gate record was probed at y≈{sensed:.1f} but this pool's floor is y {built:.1f} — records swapped?"
    return None


def sensed_place(rec: dict[str, Any], where: str) -> tuple[float, float] | None:
    """(y_altitude, distance_from_spawn) as the BODY sensed them, from a gate-(ii) record.

    The replay's `pool2()` takes raw y and d and re-applies the encoder's normalizations
    (`y/128`, `(d+128)/256`), so inverting the record's stored normalized values hands it exactly
    what it wants. This is what the probe actually measured, which beats recomputing the pair from
    anchor coordinates — and it needs no `world_spawn`, which is stamped only when a pool is BUILT
    with `--spawn-x/y/z` (world spawn is not readable over RCON, so both live pools carry null).
    """
    key = "v_dark" if where == "submerged" else "v_safe"
    by = {r["sensor"]: r for r in rec.get("per_sensor") or []}
    y, d = by.get("y_altitude"), by.get("distance_from_spawn")
    if y is None or d is None or y.get(key) is None or d.get(key) is None:
        return None
    return float(y[key]) * 128.0, float(d[key]) * 256.0 - 128.0


def _dist3(a: Any, b: Any) -> float:
    ax, ay, az = (a["x"], a["y"], a["z"]) if isinstance(a, dict) else (a[0], a[1], a[2])
    bx, by, bz = (b["x"], b["y"], b["z"]) if isinstance(b, dict) else (b[0], b[1], b[2])
    return math.sqrt((float(ax) - float(bx)) ** 2 + (float(ay) - float(by)) ** 2 + (float(az) - float(bz)) ** 2)


def replay_prediction(
    rec1: dict[str, Any],
    rec2: dict[str, Any],
    *,
    script: Path = REPLAY_SCRIPT,
    gate1: dict[str, Any] | None = None,
    gate2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-run the committed offline replay on the geometry that ACTUALLY EXISTS — twice.

    The prereg's 0.999 is for the *recommended* floor y 95; the pools on the rig are where the
    operator's clearance guard put them, so a copied constant would be a prediction about a pool
    that does not exist. This loads the committed replay module by path (it is the artifact under
    citation, not a library) and overrides pool 2's two place ABSOLUTES on pool 1's sensor map —
    the replay's own "stacked pool 2" row, and the citable prediction. The absolutes come from the
    gate records, i.e. what the BODY sensed at each pool (see :func:`sensed_place`).

    That synthetic construction cannot see a light or time difference: pool 2's other sensors are
    pool 1's by definition. So when both pools' gate-(ii) records are supplied, the cosine their
    ACTUAL `v_dark` vectors imply is computed beside it, and the two disagreeing across the
    threshold is a refusal. The synthetic number is the prediction; the record-derived one is what
    catches a pool that was not built to spec. (Without it, the stale-light pool-2 record predicts
    HIT at 0.9995 and the real vectors say 0.5878.)
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
    # WHERE the place absolutes come from. Preferred: the gate records, because that is what the
    # body sensed at each pool — and it needs no `world_spawn`, which is stamped only when a pool is
    # built with `--spawn-x/y/z` and is null on both live pools. Fallback: the anchor coordinates
    # plus a stamped world spawn, for a what-if over geometry that has not been probed yet.
    place1 = sensed_place(gate1, "submerged") if gate1 is not None else None
    place2 = sensed_place(gate2, "submerged") if gate2 is not None else None
    shore2_place = sensed_place(gate2, "shore") if gate2 is not None else None
    place_source = "gate records (sensed)"
    if place1 is None or place2 is None or shore2_place is None:
        place_source = "anchor coordinates + stamped world_spawn"
        spawn1, spawn2 = rec1.get("world_spawn"), rec2.get("world_spawn")
        # compared NUMERICALLY: the builder writes a list and the pre-check a dict, and [0, 64, 0]
        # must not refuse against [0.0, 64.0, 0.0]
        if spawn1 is None or spawn2 is None or _dist3(spawn1, spawn2) > 1e-6:
            raise Refusal(
                "the place absolutes are not derivable: the gate records carry no y_altitude/"
                "distance_from_spawn pair, and the anchors do not carry ONE stamped world_spawn "
                f"({spawn1!r} v {spawn2!r}) — pass both gate records, or rebuild with --spawn-x/y/z"
            )
        place1 = (float(rec1["submerged"][1]), _dist3(spawn1, rec1["submerged"]))
        place2 = (float(rec2["submerged"][1]), _dist3(spawn1, rec2["submerged"]))
        shore2_place = (float(rec2["shore"][1]), _dist3(spawn1, rec2["shore"]))
    (y1, d1), (y2, d2) = place1, place2
    s1 = mod.embed(mod.pool2(mod.SUB, y=y1, d=d1))
    s2 = mod.embed(mod.pool2(mod.SUB, y=y2, d=d2))
    shore2 = mod.embed(mod.pool2(mod.SHORE, y=shore2_place[0], d=shore2_place[1]))
    cross = mod.cos(s1, s2)
    own2 = mod.cos(shore2, s2)
    th = float(FROZEN["cosine_threshold"])
    if abs(th - float(mod.TH)) > 1e-9:
        raise Refusal(
            f"the replay's threshold ({mod.TH}) is not this harness's ({th}) — the prediction is not comparable"
        )
    # The guard above validates the ENCODER MAPPING (`embed(SUB)` still reproduces the record's
    # cosine). It says nothing about whether the anchors describe the pools the records describe —
    # and the prediction is built from `pool2(SUB, y, d)`, i.e. the anchors. Those two bindings are
    # what `gate1` closes: anchor ↔ cited record ↔ the record the replay predicts from.
    from_records: dict[str, Any] | None = None
    if gate1 is not None:
        # The replay's basis is a HARDCODED record (`mod.REC`). `reproduces_live_record` proves the
        # encoder mapping has not moved — not that those vectors describe the pool 1 being run.
        # This binds them. The record-to-anchor half of the chain is `gate_record_matches_pool`,
        # asserted in `cite_gate_records` before any of this runs.
        cited_sub = {r["sensor"]: r["v_dark"] for r in gate1.get("per_sensor") or []}
        if cited_sub != mod.SUB:
            raise Refusal(
                f"the cited pool-1 gate record is not the one the replay predicts from "
                f"({Path(mod.REC).name}) — re-point the replay or cite that record; a prediction "
                "from a different pool 1 is not citable"
            )
    if gate1 is not None and gate2 is not None:
        r1 = mod.embed({s["sensor"]: s["v_dark"] for s in gate1.get("per_sensor") or []})
        r2 = mod.embed({s["sensor"]: s["v_dark"] for s in gate2.get("per_sensor") or []})
        rec_cross = mod.cos(r1, r2)
        from_records = {"cross_pool_cosine": round(rec_cross, 4), "predicted_cross_hit": bool(rec_cross >= th)}
        if (rec_cross >= th) != (cross >= th):
            raise Refusal(
                f"the synthetic prediction ({cross:.4f}) and the pools' OWN probe vectors ({rec_cross:.4f}) "
                f"fall on opposite sides of {th} — the pools differ in something the place-absolutes-only "
                "construction cannot represent (light and time are the candidates); this is an apparatus "
                "refusal, not a prediction"
            )
    return {
        "replay_script": str(Path(script).resolve()),
        "reproduces_live_record": round(base, 4),
        "place_absolutes_from": place_source,
        "from_gate_records": from_records,
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
        self._rcon: Any | None = None  # LAZY: `RconControl.__init__` connects eagerly, and the
        # replay row needs no world control. Constructing it in __init__ made the documented
        # `replay` invocation crash before writing anything — and the replay row is one of the five
        # frozen gates, so every verdict would have read INCOMPLETE.
        fz = FROZEN["exp60"]
        # ONE cap for both pools: the window must sit below BOTH pain edges, or the arms' latencies
        # are not comparable and the cap difference lands inside the contrast.
        edges = {p: float(geoms[p]["measured"]["t_pain_edge_min_s"]) for p in ("pool1", "pool2")}
        self.pain_edges = edges
        self.probe_cap_s = min(edges.values()) - fz["probe_cap_margin_s"]
        self.train_cap_s = float(geoms["pool1"]["measured"]["t_damage_onset_min_s"]) - fz["train_cap_margin_s"]
        self.workdir = Path(args.workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

    @property
    def rcon(self) -> Any:
        if self._rcon is None:
            self._rcon = C.RconControl(self.args.rcon_host, self.args.rcon_port, self.args.rcon_password)
        return self._rcon

    def close(self) -> None:
        if self._rcon is not None:
            self._rcon.close()
            self._rcon = None

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
            cited, records = cite_gate_records(self.args.gate_record, self.geoms)
            row["gate_ii"] = cited
            row["context_constants"] = context_constants_match(records["pool1"], records["pool2"])
            self.gate_records = records
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
            g = getattr(self, "gate_records", None) or {}
            row.update(
                replay_prediction(self.geoms["pool1"], self.geoms["pool2"], gate1=g.get("pool1"), gate2=g.get("pool2"))
            )
            if not row["pool2_separates_internally"]:
                # the "must" this number exists for, given a caller: without it the cross-pool gate
                # is vacuous — a pool that cannot tell its own shore from its own water can match
                # anything, and matching is what the NODE gate reads.
                raise Refusal(
                    f"pool 2 does not separate its OWN shore from its OWN water "
                    f"({row['pool2_own_shore_water_cosine']} >= {row['threshold']}) — the cross-pool "
                    "gate would be vacuous there"
                )
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
        # the agent's home sits BESIDE the row's cwd, never on it: the row chdirs into
        # `workdir/<arm>_<seed>` (the loop writes cwd-relative) and the home is cleared below
        home = self.workdir / f"{arm}_{seed}" / "agent"
        agent_id = f"exp62_{arm}_{seed}"
        aut = pump = trial = None
        try:
            # "One FRESH agent per row" is the design, and `build_bio_stack` defaults to
            # `load_persisted=True` — so a home left by an earlier ATTEMPT at this (arm, seed) would
            # be restored, fear and all. `--resume` exists to re-run refused rows, which makes that
            # the common case, not an edge one: the rebuilt agent would carry the previous attempt's
            # booked fear (possibly booked at the previous attempt's READ pool) into a row published
            # as a clean cross-pool carry, and would turn the ablated arm into a false leak refusal.
            # `check_no_positive_escape_link` cannot see it — it reads causal links, not fear.
            # Exp 61 clears the home before every build; so does this.
            shutil.rmtree(home, ignore_errors=True)
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
            row["train_pool_state"] = _context_snapshot(trial)
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

            # 4. the NODE gate at the read pool, loop OFF — THE RESULT.
            # The live snapshot is recorded on BOTH sides: when the gate misses, §Outcome has to
            # name WHICH live thing differed, and light/time is the named suspect (A1). Without it
            # the report can only say "the replay was wrong about something live".
            row["live_state"] = {"read_pool": _context_snapshot(trial)}
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


def cite_gate_records(
    paths: list[str], geoms: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load, bind and GATE the two committed gate-(ii) records. Refuses; never scores.

    Both subcommands go through this, because the replay row is written first and is exactly where
    a stale record would do its damage (its synthetic prediction cannot see light or time at all).
    """
    if len(paths) != 2:
        raise Refusal("pass --gate-record twice: pool 1's committed gate-(ii) record, then pool 2's")
    cited, records = [], {}
    for label, path in zip(("pool1", "pool2"), paths, strict=True):
        rec = json.loads(Path(path).expanduser().read_text())
        records[label] = rec
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
        mismatch = gate_record_matches_pool(rec, geoms[label])
        if mismatch:
            raise Refusal(f"{label}: {mismatch} ({path})")
        cited.append(entry)
    # The prereg's light/time gate. A run_gate PASS is not it: pool 2's stale-light record passes
    # its OWN gate and still puts the cross-pool contrast on the light axis.
    ctx = context_constants_match(records["pool1"], records["pool2"])
    if not ctx["match"]:
        raise Refusal(
            "the two pools' gate-(ii) records disagree on the full-weight constants — "
            + "; ".join(ctx["mismatches"])
            + ". The contrast would be a LIGHT/TIME contrast, not a pool contrast (prereg §Apparatus: "
            "light and time GATED equal; a stale-light read at a freshly filled box is the known "
            "failure — re-probe after a client reconnect and cite THAT record)"
        )
    return cited, records


def _context_snapshot(trial: WaterTrial) -> dict[str, Any]:
    """The full-weight constants as the BODY currently senses them, straight off the bridge."""
    state = trial.aut.client.latest_state() or {}
    return {k: state.get(k) for k in (*CONTEXT_SENSORS, "y_altitude", "distance_from_spawn", "is_in_water")}


def load_geoms(pool1: str, pool2: str) -> dict[str, dict[str, Any]]:
    geoms: dict[str, dict[str, Any]] = {}
    for label, path in (("pool1", pool1), ("pool2", pool2)):
        rec = json.loads(Path(path).expanduser().read_text())
        measured = rec.get("measured")
        if not isinstance(measured, dict):
            raise Refusal(f"{label}'s record ({path}) carries no `measured` block — run exp60_water_check on it first")
        # a PARTIAL block is its own failure: the caps are derived from these two keys, and a bare
        # KeyError three frames later is not the documented refusal
        for key in ("t_pain_edge_min_s", "t_damage_onset_min_s"):
            if measured.get(key) is None:
                raise Refusal(f"{label}'s `measured` block ({path}) has no {key} — re-run exp60_water_check on it")
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
    try:
        _cited, camp.gate_records = cite_gate_records(args.gate_record, geoms)
    except (Refusal, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 4
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
    unknown_arms = set(args.only) - set(ARM_ORDER)
    if unknown_arms:
        print(f"[FAIL] --only names no such arm: {sorted(unknown_arms)} (arms are {list(ARM_ORDER)})")
        return 2
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
        camp.close()
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
    rp.add_argument(
        "--gate-record", action="append", default=[], help="committed gate-(ii) record; pass TWICE, pool 1 first"
    )
    _evidence_args(rp)
    rp.set_defaults(func=cmd_replay, workdir=".", rcon_host="", rcon_port=0, rcon_password="")

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
