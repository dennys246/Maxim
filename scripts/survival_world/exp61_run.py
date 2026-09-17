#!/usr/bin/env python3
"""Exp 61 LIVE campaign harness — shared survival fear (a learned drowning-fear transfers between
independent agents and drives the receiver's first loop-live submersion).

Build step 3 of ``docs/experiments/exp61_shared_fear_prereg.md`` (v2, four-lens review folded;
D1 = 0.75 social discount at the ingest bound, D2 = arm sizes 24/12/12/24). Composes the Exp 60
seed context (``water_trial.WaterTrial`` — the SAME preflights, rescue/submerge, loop window,
placement, propose-only training and live G2 read) with Exp 56's REAL-CLI export/ingest helpers
(``exp56.common``: ``export_bundle`` incl. the nac-only ``dangling`` re-compose, ``ingest_bundle_into``
reading the ingestion JOURNAL). Nothing is hand-composed: the donor's fear reaches the receiver
only through ``maxim substrate export`` → ``maxim substrate ingest``.

Per PAIR (seed-paired donor/receiver, arms interleaved within the pair so drift affects them
equally). The wet APPARATUS checks (bridge roster + cadence, gamerules, rain/spectator, LIVE
cluster-distinct, escape actuation through the bridge) run ONCE PER PAIR on a THROWAWAY agent
that is then discarded — no donor or receiver is submerged by them.

  DONOR (arms 2/3; arm 4 re-composes a fear donor's staged nac-only — pairs 1–12 their own arm-2
  donor, pairs 13–24 the arm-2 donor of pair k−12: donor identity is immaterial in arm 4 because
  the fear is DROPPED by construction, so the no-reuse rule is scoped to arms 2/3 — recorded in the
  row as ``donor_pair``; the freeze PR carries the prereg wording):
    build (hub session opened) → fingerprint, bridge, liveness, gamerules, LIVE cluster-distinct →
    K yoked propose-only training episodes → the live G2 read → FULL close + stage
    (``aut_nac.json``/``aut_ec.json``). The loop's own session pair CLOSES the hub session the
    harness opened (environment lens S1, one layer up): ``WaterTrial`` re-opens it after every loop
    run and the staging close PROVES it persisted (an already-closed hub returns {} and saves
    nothing → Refusal). DONOR SANITY ON THE STAGED FILES: export-before-probe as a file fact
    (``links == {}``, welford/bias maps empty; fear keys ≥ 1, all ``drive:oxygen``, all at −1.0, all on
    training-noted world nodes, none on the shore; one geometry tag; the ablated donor carries no
    fear) → REAL CLI export with the ``body:``-rooted spec beside this file. NO donor probe, ever.

  RECEIVER (every arm), numbered as the prereg's §Receiver lifecycle:
    1. B pre-ingest: build (hub session opened) → no loop, no water → full close + stage; the
       stage must hold ZERO world nodes and no fear;
    2. ingest through the REAL CLI with ``--receiver-agent-id`` B; the journal entry is gated per
       arm on ``fear_rekeyed``/``fear_dropped``/``fear_below_floor``/``fear_discount``; every fear key
       carries B's id; the folded value equals the post-discount cap; donor node ids ⊆ B's
       post-ingest ids with the count unchanged; both ``saved_at`` recorded;
    3. B reboots from its home; shore only: liveness (no executor call during it), no positive
       ``escape_water``/``flee`` link, B's live shore tag == the donor's world-node tag;
    4. the REPRESENTATION + READABILITY gate: ONE loop-OFF submersion (US-free; oxygen at the
       gate and zero pain publishes recorded) — B's reading must complete into the transferred
       node in arms 2/3 (a fresh id in arm 1), the need on it > 0.5 in arm 2 and 0 elsewhere,
       shore fear 0, ids distinct — each miss its own named refusal, before the one-shot
       placement is spent;
    5. FIRST CONTACT: the first teleport into water B ever receives with the loop live. Success
       requires BOTH the decision DV (``escape_water`` executed inside the window) AND the
       behavioural DV (head in air by bridge truth before the cap); a surface with zero executor
       calls is a REFUSAL; an executed escape with no captured proposal is a REFUSAL (instrument);
       the executed proposal's decision provenance (the first escape-best event that PASSED the
       gate) must be DRIVE-decisive (causal 0, learned bias 0) or the placement counts AGAINST;
    6. one further placement recorded (fear + own link), never gated; teardown; homes are
       DURABLE per pair (``--workdir``), ``--resume`` skips (kind, arm, pair) rows already clean.

  ANTI-VACUITY: after the first clean transfer pair the kit runs over that pair's STAGED files
  (the real aligned ``substrate_merge`` must make the receiver read the fear; the no-op variants —
  receiver unchanged, empty state — must read 0) and is written as a campaign row; the verdict
  REQUIRES it (absent → INCOMPLETE).

Every rescue settle re-checks ``is_raining == 0`` and ``nearest_player_dist == 64`` (an absent key
refuses, never defaults); a per-pair RSS line is stamped on every row.

``verdict`` (pure, unit-tested): per-arm first-contact rates with 95 % Wilson intervals; the six
frozen gates; Fisher's exact one-sided test on the receiver binaries for arm 2 > arm 1 and
arm 2 > arm 3 (the exact permutation test on two binary samples, in closed form — 12 v 24
relabellings cannot be enumerated); INCOMPLETE on n, on rows (donor or receiver) spanning two
code hashes, on campaign drift, on a missing kit row, and INCOMPLETE-with-cause (actuation
timing) when the decision DV passes while the behavioural DV fails; a campaign whose surfaces
were won by another component is NULL with the count named. A later CLEAN row supersedes an
earlier REFUSED row for the same (arm, pair) — what ``--resume`` writes; the refusal is still
named. Refusals are never dropped.

Run ON the bridge box (server + bridge from current main at ONE code hash — no ``git pull``
between the first and last row; bridge at ``--state_interval_ms=100``; no second player):
    export PYTHONPATH="$PWD/src"
    python scripts/survival_world/exp61_run.py run --rcon-password '<pw>' --username maxim \\
        --workdir ~/exp61_work --write-experiment-results
    python scripts/survival_world/exp61_run.py verdict --data docs/experiments/data/exp61_pairs.jsonl \\
        --json docs/experiments/data/exp61_verdict.json --campaign-id <id> --write-experiment-results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
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
from survival_world.exp60_run import ANCHOR_FILE, APPARATUS_RECORD, GATE_RECORD  # noqa: E402
from survival_world.exp60_run import FROZEN as FROZEN60  # noqa: E402
from survival_world.water_trial import (  # noqa: E402
    Refusal,
    WaterTrial,
    _detach_fear_subscriber,
    min_pain_edge_s,
)

BODY_REF = "minecraft_player"
COMPONENT_YAML = C.REPO_ROOT / "src" / "maxim" / "_data" / "components" / "bodies" / "minecraft_player.yaml"
# The export CLI's --body-yaml reads a `body:`-rooted spec, never a component file (executor lens
# B2); this mirror is pinned equal to the component's affordance roster by the unit tests.
BODY_YAML = Path(__file__).resolve().parent / "body_spec_minecraft_player.yaml"
FEAR_MODE = "drive:oxygen"
ESCAPE_SUFFIX = "_escape_water"
FLEE_SUFFIX = "_flee"
ARMS_WITH_DONOR = {"transferred": "fear", "cluster_not_fear": "ablated", "dangling": "fear"}
ARM_ORDER = ("isolated", "transferred", "cluster_not_fear", "dangling")

# Frozen with the prereg (v2 → FROZEN in the freeze PR; the analyzer refuses drift). The Exp 60
# numbers are LITERAL copies (architecture lens S7): a later Exp 60 edit must fail the equality
# test below, never be inherited silently.
FROZEN: dict[str, Any] = {
    "arms": {"isolated": 24, "transferred": 12, "cluster_not_fear": 12, "dangling": 24},
    "pair_seeds": list(range(200, 224)),  # 24 pairs; arms 2/3 run on the first 12
    "dangling_donor_offset": 12,  # pairs 13–24 re-compose the arm-2 donor of pair k−12 (fear dropped by construction)
    "foreign_fear_discount": 0.75,  # prereg D1 — asserted against the ingest constant
    "fear_value_cap": -1.0,  # a donor's fear must sit at the cap
    "read_floor": 0.5,  # the consumer's STRICT activation floor
    "actuation_max_s": 2.5,  # per-pair refusal bound on the throwaway's escape actuation
    "drift_max_s": 0.5,  # campaign-level: last-quartile − first-quartile median > this → INCOMPLETE
    "settle_guard": {"is_raining": 0.0, "nearest_player_dist": 64.0},  # at every rescue settle; absent key refuses
    "gates": {
        "transferred_min": 0.70,
        "above_floor_min": 0.20,
        "cluster_not_fear_min": 0.20,
        "both_halves_max": 0.10,
        "alpha": 0.05,
    },
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


def exp60_frozen_matches() -> list[str]:
    """Every Exp 60 number this harness depends on, checked against Exp 60's live FROZEN."""
    return [k for k, v in FROZEN["exp60"].items() if FROZEN60.get(k) != v]


# ─────────────────────────── pure helpers (unit-tested) ───────────────────────────


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson score interval for k successes in n (house style: reported, not gated)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_one_sided_p(k_a: int, n_a: int, k_b: int, n_b: int) -> dict[str, Any]:
    """Exact one-sided test that arm A's success rate exceeds arm B's, on receiver binaries: the
    exact permutation test on two binary samples IS Fisher's exact test, computed in closed form
    from the hypergeometric distribution. p = P(X >= k_a | row/column totals fixed), with X the
    number of successes among the n_a receivers drawn from n = n_a + n_b holding k = k_a + k_b."""
    n = n_a + n_b
    k = k_a + k_b
    if n_a == 0 or n_b == 0:
        return {"p_one_sided": None, "observed_diff": None, "k": [k_a, k_b], "n": [n_a, n_b]}
    hi = min(k, n_a)
    total = math.comb(n, n_a)  # draws of n_a from n (the numerator counts x successes among the n_a drawn)
    p = sum(math.comb(k, x) * math.comb(n - k, n_a - x) for x in range(k_a, hi + 1)) / total
    return {
        "p_one_sided": p,
        "observed_diff": k_a / n_a - k_b / n_b,
        "k": [k_a, k_b],
        "n": [n_a, n_b],
        "test": "fisher-exact-one-sided (hypergeometric; = exact permutation on binaries)",
    }


def decision_decisive(event_data: dict[str, Any] | None, tool_suffix: str = ESCAPE_SUFFIX) -> tuple[bool, str]:
    """Exp 56's decision-provenance clause, for fear: the EXECUTED escape proposal must have been
    won by the DRIVE component with causal == 0 and learned bias == 0. A surface won by any other
    component counts AGAINST the claim; a missing event is not decisive."""
    if not event_data:
        return False, "no NAc_RECOMMEND event captured for the window"
    best = str(event_data.get("best_tool") or "")
    if not best.endswith(tool_suffix):
        return False, f"winning tool {best!r} is not the escape affordance"
    comps = event_data.get("score_components") or {}
    drive = float(comps.get("drive", 0.0) or 0.0)
    causal = float(comps.get("causal", 0.0) or 0.0)
    learned = float(comps.get("learned_bias", 0.0) or 0.0)
    if drive <= 0.0:
        return False, f"drive component {drive} is not positive"
    if causal != 0.0 or learned != 0.0:
        return False, f"won with causal={causal} learned_bias={learned} (must both be 0)"
    return True, "drive-decisive"


def first_contact_outcome(
    placement: dict[str, Any], *, cap_s: float, decision_event: dict[str, Any] | None
) -> dict[str, Any]:
    """The co-primary DVs and the censoring class for ONE first-contact placement.

    success = decision DV (escape_water EXECUTED inside the window) AND behavioural DV (head in air
    by bridge truth before the cap and not DIRTY) AND decisive provenance. Two REFUSAL shapes
    (raised by the caller on ``refusal``): a surface with ZERO executor calls (a held control or
    teleport artefact), and an executed escape with NO captured proposal (the sink did not deliver —
    an instrument inconsistency, never a mechanism null)."""
    calls = placement.get("calls") or []
    escape_calls = [c for c in calls if str(c.get("tool", "")).endswith(ESCAPE_SUFFIX)]
    flee_calls = [c for c in calls if str(c.get("tool", "")).endswith(FLEE_SUFFIX)]
    t_escape = min((float(c["t"]) for c in escape_calls), default=None)
    t_flee = min((float(c["t"]) for c in flee_calls), default=None)
    surfaced = bool(placement.get("surfaced")) and not placement.get("dirty")
    decision_dv = t_escape is not None and t_escape <= cap_s
    decisive, why = decision_decisive(decision_event)
    refusal = None
    if surfaced and not calls:
        refusal = "head in air with ZERO executor calls (a held control or teleport artefact must not score)"
    elif decision_dv and decision_event is None:
        refusal = "escape_water executed but no NAc_RECOMMEND proposal was captured (instrument: sink not delivering)"
    censoring = None
    if placement.get("dirty"):
        censoring = "US/damage inside the window (dirty) — the cap is not below the pain edge on this apparatus"
    elif decision_dv and not surfaced:
        censoring = "escape called before the cap, head not in air by the cap"
    elif not decision_dv and not surfaced:
        censoring = "no escape call inside the window"
    return {
        "success": bool(surfaced and decision_dv and decisive and refusal is None),
        "behavioural_dv": surfaced,
        "decision_dv": decision_dv,
        "decisive": decisive,
        "decisive_reason": why,
        "t_flee_call": t_flee,
        "t_escape_call": t_escape,
        "t_first_air": placement.get("latency_s") if surfaced else None,
        "escape_calls": len(escape_calls),
        "flee_calls": len(flee_calls),
        "censoring": censoring,
        "refusal": refusal,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def donor_sanity_staged(
    stage_dir: Path,
    *,
    donor_kind: str,
    episode_clusters: list[str],
    shore_node: str | None,
) -> dict[str, Any]:
    """Asserted on the STAGED files the export reads (never the object). Pure over the files."""
    nac = _load_json(stage_dir / "aut_nac.json")
    ec = _load_json(stage_dir / "aut_ec.json")
    nodes = (ec.get("substrate_nodes") or {}) if isinstance(ec, dict) else {}
    world_nodes = {nid for nid, nd in nodes.items() if isinstance(nd, dict) and nd.get("modality") == "world"}
    tags = {nd.get("geometry") for nid, nd in nodes.items() if nid in world_nodes}
    fear = nac.get("cluster_fear") or {}
    reasons: list[str] = []
    for field in ("links", "event_outcome_welford", "cluster_reward_bias"):
        if nac.get(field):
            reasons.append(f"{field} is not empty ({len(nac[field])}) — a probe or an execution happened before export")
    # Node-level `reward_bias` must be EMPTY on a fresh donor, and the two ways it is not are told
    # apart. Dry run 1 (2026-09-17, pair 200) refused both donors on three ZERO-valued keys: the
    # pain credit (`temporal_credit.distribute` → `NAc.credit_node`, a negative share clamped at
    # 0.0) used to STORE the key. That was the NAc's wart, fixed the same day (`credit_node` now
    # removes a bias that clamps to zero — the meaning the decay prune already gave it), and the
    # offline smoke pins that propose-only training stages `reward_bias == {}`. Donors are always
    # trained fresh, in THIS process, so a zero key here can only mean the running `maxim` is not
    # the repo's (a provenance fault) or a zero-writer regressed into the NAc — both refuse. A
    # NON-ZERO bias can only come from a positive reaction (relief / success) being credited: a
    # probe or an execution before export. The zero count stays on the row for the record.
    rb = nac.get("reward_bias") or {}
    nonzero_rb = {k: v for k, v in rb.items() if float(v) != 0.0}
    reward_bias_zero_nodes = len(rb) - len(nonzero_rb)
    if nonzero_rb:
        reasons.append(
            f"reward_bias carries {len(nonzero_rb)} non-zero node bias(es) — a positive reaction was credited: "
            "a probe or an execution happened before export"
        )
    if reward_bias_zero_nodes:
        reasons.append(
            f"reward_bias carries {reward_bias_zero_nodes} ZERO-valued node key(s) — the running NAc still stores "
            "the pain credit's clamp (fixed 2026-09-17): a stale `maxim` install or a regressed writer"
        )
    pv = nac.get("percept_valences") or {}
    if not any(FEAR_MODE in str(k) for k in pv):
        reasons.append("percept_valences carries no drive:oxygen entry — the pain never published")
    if donor_kind == "fear":
        if not fear:
            reasons.append("no cluster_fear key on the fear donor")
        for k, v in fear.items():
            parts = str(k).split("\x1f")
            if len(parts) != 3:
                reasons.append(f"malformed fear key {k!r}")
                continue
            if parts[2] != FEAR_MODE:
                reasons.append(f"fear key under {parts[2]!r} (only {FEAR_MODE} may ship in this arm)")
            if parts[1] not in world_nodes:
                reasons.append(f"fear on a non-world node {parts[1][:8]}")
            if parts[1] not in set(episode_clusters):
                reasons.append(f"fear on node {parts[1][:8]} that training never noted")
            if shore_node and parts[1] == shore_node:
                reasons.append("fear on the SHORE node")
            if float(v) != FROZEN["fear_value_cap"]:
                reasons.append(f"fear value {v} != cap {FROZEN['fear_value_cap']}")
    else:
        if fear:
            reasons.append(f"ablated donor carries {len(fear)} cluster_fear key(s)")
    if not world_nodes:
        reasons.append("the staged EC carries no world node — the training situation was never persisted")
    if len(tags - {None}) != 1 or None in tags:
        reasons.append(
            f"world nodes carry {len(tags)} geometry tag(s) (need exactly one, all stamped): {sorted(map(str, tags))}"
        )
    return {
        "pass": not reasons,
        "reasons": reasons,
        "fear_shipped": len(fear),
        "fear_keys": sorted(fear),
        "reward_bias_zero_nodes": reward_bias_zero_nodes,
        "world_nodes": len(world_nodes),
        "geometry_tag": next(iter(tags)) if len(tags) == 1 else None,
        "saved_at": nac.get("saved_at"),
        "nac_sha256": hashlib.sha256((stage_dir / "aut_nac.json").read_bytes()).hexdigest(),
    }


def ingest_gate(arm: str, entry: dict[str, Any], *, shipped: int) -> str | None:
    """The per-arm gate on the ingestion JOURNAL entry (the durable surface). None = pass; a
    MISSING key reads -1 and fails, never passes silently."""
    rk, dr, bf = (int(entry.get(k, -1)) for k in ("fear_rekeyed", "fear_dropped", "fear_below_floor"))
    disc = entry.get("fear_discount")
    if arm == "transferred":
        if shipped < 1:
            return "transfer arm shipped no fear"
        if rk != shipped or dr != 0 or bf != 0:
            return f"transfer ingest rekeyed={rk} dropped={dr} below_floor={bf} vs shipped={shipped}"
        if disc != FROZEN["foreign_fear_discount"]:
            return f"ingest discount {disc!r} != frozen {FROZEN['foreign_fear_discount']}"
    elif arm == "cluster_not_fear":
        if shipped != 0 or rk != 0 or dr != 0:
            return f"cluster-not-fear ingest carried fear: shipped={shipped} rekeyed={rk} dropped={dr}"
    elif arm == "dangling":
        if shipped < 1 or rk != 0 or dr != shipped:
            return f"dangling ingest rekeyed={rk} dropped={dr} vs shipped={shipped} (the representation half must drop LOUDLY)"
    return None


def anti_vacuity_kit(donor_stage: Path, receiver_pre_stage: Path, *, receiver_agent_id: str) -> dict[str, Any]:
    """Exp 56's kit, for fear: the REAL aligned fold of one transfer pair's staged files must make the
    receiver read the donor's fear, and the no-op variants (receiver unchanged; empty state) must
    read 0. A gate that cannot fail is not a gate (D62) — recorded as a campaign row that the
    verdict REQUIRES. Pure over the staged files; `substrate_merge` is the trusted-local fold, so
    the fear lands verbatim here (the discount is the ingest bound's)."""
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.hivemind.merge import substrate_merge

    donor_nac = _load_json(donor_stage / "aut_nac.json")
    donor_ec = _load_json(donor_stage / "aut_ec.json").get("substrate_nodes") or {}
    recv_nac = _load_json(receiver_pre_stage / "aut_nac.json")
    recv_ec = _load_json(receiver_pre_stage / "aut_ec.json").get("substrate_nodes") or {}

    def read(nac_state: dict[str, Any], ec_nodes: dict[str, Any]) -> float:
        nac = NAc(NACConfig())
        nac.load_state(nac_state)
        world = [nid for nid, nd in ec_nodes.items() if isinstance(nd, dict) and nd.get("modality") == "world"]
        return max((nac.anticipatory_threat_need(receiver_agent_id, {"world": cid}) for cid in world), default=0.0)

    real = substrate_merge(
        receiver_nac=recv_nac,
        receiver_ec=recv_ec,
        donor_nac=donor_nac,
        donor_ec=donor_ec,
        receiver_source="receiver",
        donor_source="donor",
        receiver_agent_id=receiver_agent_id,
    )
    real_read = read(real.nac, real.ec_nodes)
    variants = {"receiver_unchanged": read(recv_nac, recv_ec), "empty_state": read({}, {})}
    return {
        "real_read": real_read,
        "real_fear_rekeyed": real.fear_rekeyed,
        "variants": variants,
        "pass": real_read > FROZEN["read_floor"] and all(v == 0.0 for v in variants.values()),
    }


def _quartile_medians(xs: list[float]) -> tuple[float | None, float | None]:
    if len(xs) < 4:
        return None, None
    q = max(1, len(xs) // 4)

    def med(v: list[float]) -> float:
        v = sorted(v)
        m = len(v) // 2
        return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2

    return med(xs[:q]), med(xs[-q:])


def campaign_drift(rows: list[dict[str, Any]], *, max_s: float) -> list[str]:
    """Monotone-ish drift across the campaign: last-quartile median − first-quartile median of the
    per-pair actuation t_surface (apparatus rows, in order) and of arm-2's first-contact latency."""
    problems: list[str] = []
    ordered = sorted((r for r in rows if r.get("refusal") is None), key=lambda r: r["ts"])
    act = [
        float(r["apparatus"]["actuation"]["t_surface"])
        for r in ordered
        if r.get("kind") == "apparatus" and (r.get("apparatus") or {}).get("actuation", {}).get("t_surface") is not None
    ]
    lat = [
        float(r["first_contact"]["t_first_air"])
        for r in ordered
        if r.get("kind") == "receiver"
        and r.get("arm") == "transferred"
        and (r.get("first_contact") or {}).get("t_first_air") is not None
    ]
    for name, xs in (("actuation t_surface", act), ("transferred first-contact latency", lat)):
        a, b = _quartile_medians(xs)
        if a is not None and b is not None and b - a > max_s:
            problems.append(f"{name} drifted {a:.2f}s → {b:.2f}s (> {max_s}s)")
    return problems


def compute_verdict(rows: list[dict[str, Any]], *, campaign_id: str | None) -> dict[str, Any]:
    """The prereg's gate decision over the per-(pair, arm) receiver rows. Pure."""
    in_campaign = [r for r in rows if campaign_id is None or r.get("campaign_id") == campaign_id]
    sel = [r for r in in_campaign if r.get("kind") == "receiver"]
    anti = [r for r in in_campaign if r.get("kind") == "anti_vacuity"]
    refused: list[str] = []
    # A later CLEAN row supersedes an earlier REFUSED row for the same (arm, pair) — that is what
    # `--resume` writes; the refusal is still named. Two CLEAN rows for one key is a duplicate.
    clean_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for r in sorted(sel, key=lambda r: r["ts"]):
        key = (r["arm"], int(r["pair_seed"]))
        if r.get("refusal") is not None:
            refused.append(f"{r['arm']} pair {r['pair_seed']}: {r['refusal']}")
            continue
        if key in clean_by_key:
            refused.append(f"duplicate clean (arm, pair) row {key} — pass --campaign-id to select one campaign")
            continue
        clean_by_key[key] = r
    clean: dict[str, list[dict[str, Any]]] = {a: [] for a in FROZEN["arms"]}
    for (arm, _seed), r in clean_by_key.items():
        clean.setdefault(arm, []).append(r)
    # ONE code hash per campaign — donor, apparatus and receiver rows alike ("between the first and last ROW")
    hashes = {(r.get("provenance") or {}).get("executed_git_hash") for r in in_campaign if "provenance" in r}
    if len(hashes) > 1:
        refused.append(f"rows span {len(hashes)} code hashes {sorted(map(str, hashes))} — one code hash per campaign")
    n_clean = {a: len(v) for a, v in clean.items()}
    rates: dict[str, dict[str, Any]] = {}
    binaries: dict[str, list[float]] = {}
    for arm, rs in clean.items():
        fcs = [r.get("first_contact") or {} for r in rs]
        succ = [1.0 if fc.get("success") else 0.0 for fc in fcs]
        binaries[arm] = succ
        k, n = int(sum(succ)), len(succ)
        rates[arm] = {
            "n": n,
            "successes": k,
            "rate": (k / n) if n else None,
            "wilson95": list(wilson_interval(k, n)) if n else None,
            "decision_dv_rate": (sum(1 for fc in fcs if fc.get("decision_dv")) / n) if n else None,
            "behavioural_dv_rate": (sum(1 for fc in fcs if fc.get("behavioural_dv")) / n) if n else None,
            "not_decisive": sum(1 for fc in fcs if fc.get("behavioural_dv") and not fc.get("decisive")),
        }
    g = FROZEN["gates"]
    incomplete = [
        f"{a}: {n_clean[a]} clean pairs < {FROZEN['arms'][a]}" for a in FROZEN["arms"] if n_clean[a] < FROZEN["arms"][a]
    ]
    incomplete.extend(campaign_drift(in_campaign, max_s=FROZEN["drift_max_s"]))
    if not anti:
        incomplete.append("anti-vacuity kit not recorded for this campaign")

    def _fisher(a: str, b: str) -> dict[str, Any] | None:
        if not binaries[a] or not binaries[b]:
            return None
        return fisher_one_sided_p(int(sum(binaries[a])), len(binaries[a]), int(sum(binaries[b])), len(binaries[b]))

    perm21 = _fisher("transferred", "isolated")
    perm23 = _fisher("transferred", "cluster_not_fear")
    r2, r1, r3, r4 = (rates[a]["rate"] for a in ("transferred", "isolated", "cluster_not_fear", "dangling"))
    dangling_accounting = all(
        (r.get("ingest") or {}).get("fear_rekeyed") == 0
        and (r.get("ingest") or {}).get("fear_dropped") == (r.get("donor") or {}).get("fear_shipped")
        for r in clean["dangling"]
    )
    checks = {
        "transferred": r2 is not None and r2 >= g["transferred_min"],
        "above_floor": (
            r2 is not None
            and r1 is not None
            and (r2 - r1) >= g["above_floor_min"]
            and perm21 is not None
            and perm21["p_one_sided"] < g["alpha"]
        ),
        "cluster_not_fear": (
            r2 is not None
            and r3 is not None
            and (r2 - r3) >= g["cluster_not_fear_min"]
            and perm23 is not None
            and perm23["p_one_sided"] < g["alpha"]
        ),
        "both_halves": r4 is not None and r1 is not None and (r4 - r1) < g["both_halves_max"] and dangling_accounting,
        # honest as a count of refusals: a CLEAN transferred row passed the representation gate by construction
        "specificity": bool(clean["transferred"])
        and all((r.get("representation_gate") or {}).get("pass") for r in clean["transferred"]),
        "anti_vacuity": bool(anti) and all(bool((r.get("kit") or {}).get("pass")) for r in anti),
    }
    verdict = "INCOMPLETE"
    cause: str | None = None
    if not incomplete and not any(m.startswith("rows span") for m in refused):
        t = rates["transferred"]
        if all(checks.values()):
            verdict = "EARNED"
        elif (
            not checks["transferred"]
            and t["decision_dv_rate"] is not None
            and t["decision_dv_rate"] >= g["transferred_min"]
            and t["behavioural_dv_rate"] is not None
            and t["behavioural_dv_rate"] < g["transferred_min"]
        ):
            verdict, cause = "INCOMPLETE", "actuation timing: the decision DV passes while the behavioural DV fails"
        else:
            verdict = "NULL"
            if t["not_decisive"]:
                cause = f"{t['not_decisive']} transferred surface(s) were won by a component other than the drive (counted against)"
    if incomplete:
        cause = "; ".join(incomplete)
    return {
        "_format_version": "1.0",
        "kind": "exp61_verdict",
        "campaign_id": campaign_id,
        "refused": refused,
        "n_clean": n_clean,
        "rates": rates,
        "permutation": {"transferred_vs_isolated": perm21, "transferred_vs_cluster_not_fear": perm23},
        "checks": checks,
        "gates": g,
        "verdict": verdict,
        "incomplete_cause": cause,
    }


def pair_plan(idx: int, pair_seed: int, *, n_full: int, offset: int) -> tuple[list[str], dict[str, int]]:
    """Which arms a pair runs and which pair's donor each donor-bearing arm uses (pure)."""
    if idx < n_full:
        return list(ARM_ORDER), {"transferred": pair_seed, "cluster_not_fear": pair_seed, "dangling": pair_seed}
    return ["isolated", "dangling"], {"dangling": pair_seed - offset}


# ─────────────────────────────── the live campaign ───────────────────────────────


def _rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024, 1)


def _build(args: argparse.Namespace, *, agent_id: str, home: Path) -> tuple[Any, Any, Any]:
    """Canonical assembly on the player body with the production encoder; hub session OPENED."""
    from maxim.simulation.minecraft import MinecraftClient
    from maxim.simulation.minecraft_harness import MinecraftSyncPump, build_minecraft_aut
    from survival_world.common import make_fresh_encoder

    home.mkdir(parents=True, exist_ok=True)
    client = MinecraftClient(args.bridge_host, args.bridge_port)
    client.connect(confirm_timeout_s=4.0, retries=8, backoff_s=0.5)
    aut = build_minecraft_aut(
        agent_id=agent_id,
        bridge_port=args.bridge_port,
        bridge_host=args.bridge_host,
        persistence_dir=str(home),
        entity_ref=f"bodies/{BODY_REF}",
        client=client,
    )
    aut.bio.memory_hub.on_session_start()  # the object that closes it must open it (D41/D42)
    encoder = make_fresh_encoder(aut)
    pump = MinecraftSyncPump(aut, interval_s=0.25)
    pump.start()
    return aut, encoder, pump


def close_and_stage(aut: Any, pump: Any, stage_dir: Path | None) -> None:
    """FULL close (hub first, then bio), PROVEN to persist, then stage the pair the CLI reads.

    The loop's own session pair closes the hub session the harness opened (environment lens S1,
    one layer up); the hub is re-opened (idempotent, no reload from disk) so THIS close is the one
    that persists, and an already-closed hub — which returns {} and saves nothing — is a Refusal,
    never a stale file copied as if fresh."""
    try:
        try:
            pump.stop()
        except Exception as exc:
            print(f"WARNING: pump stop raised: {exc!r}")
        aut.bio.memory_hub.on_session_start()
        stats = aut.bio.memory_hub.on_session_end()
        if not stats:
            raise Refusal("hub session was not active at the staging close — nothing persisted (S1)")
        aut.bio.on_session_end()
        home = Path(aut.persistence_dir)
        if stage_dir is not None:
            stage_dir.mkdir(parents=True, exist_ok=True)
            for src, dst in (("nac.json", "aut_nac.json"), ("ec.json", "aut_ec.json")):
                if not (home / src).is_file():
                    raise Refusal(f"full close did not persist {src} in {home}")
                shutil.copyfile(home / src, stage_dir / dst)
    finally:
        try:
            aut.client.close()
        except (OSError, ConnectionError) as exc:
            print(f"WARNING: client close raised: {exc!r}")


def _world_ids(ec_path: Path) -> set[str]:
    nodes = _load_json(ec_path).get("substrate_nodes") or {}
    return {nid for nid, nd in nodes.items() if isinstance(nd, dict) and nd.get("modality") == "world"}


class _Campaign:
    """One live campaign: the shared apparatus context every pair uses."""

    def __init__(
        self, args: argparse.Namespace, *, provenance: dict[str, Any], out_path: Path, campaign_id: str
    ) -> None:
        self.args = args
        self.provenance = provenance
        self.out_path = out_path
        self.campaign_id = campaign_id
        apparatus = _load_json(C.REPO_ROOT / APPARATUS_RECORD)
        gate = _load_json(C.REPO_ROOT / args.gate_record)
        if not apparatus.get("all_pass") or not (gate.get("run_gate") or {}).get("pass"):
            raise SystemExit("[FAIL] the Exp 60 apparatus check / geometry gate records must carry PASS on main")
        self.geom = _load_json(ANCHOR_FILE)
        pain_edge_min = min_pain_edge_s(apparatus)
        if pain_edge_min is None:
            raise SystemExit("[FAIL] apparatus record carries no measured t_pain_edge")
        self.pain_edge_min = pain_edge_min
        self.probe_cap_s = pain_edge_min - FROZEN["exp60"]["probe_cap_margin_s"]
        self.train_cap_s = float(self.geom["measured"]["t_damage_onset_min_s"]) - FROZEN["exp60"]["train_cap_margin_s"]
        self.apparatus_ts = apparatus.get("ts")
        self.gate_hash = gate.get("code_hash")
        drift = exp60_frozen_matches()
        if drift:
            raise SystemExit(
                f"[FAIL] Exp 61's frozen copy of the Exp 60 constants differs from Exp 60's FROZEN on {drift}"
            )
        from maxim.hivemind.ingest import FOREIGN_FEAR_DISCOUNT

        if FOREIGN_FEAR_DISCOUNT != FROZEN["foreign_fear_discount"]:
            raise SystemExit(
                f"[FAIL] ingest FOREIGN_FEAR_DISCOUNT {FOREIGN_FEAR_DISCOUNT} != frozen {FROZEN['foreign_fear_discount']}"
            )
        self.rcon = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
        self.workdir = Path(args.workdir).expanduser().resolve()  # the campaign chdirs per pair; never relative
        self.workdir.mkdir(parents=True, exist_ok=True)

    def trial(self, aut: Any, encoder: Any, *, agent_id: str, home: Path) -> WaterTrial:
        return WaterTrial(
            aut=aut,
            rcon=self.rcon,
            username=self.args.username,
            geom=self.geom,
            frozen=FROZEN["exp60"],
            probe_cap_s=self.probe_cap_s,
            train_cap_s=self.train_cap_s,
            persistence_dir=home,
            agent_id=agent_id,
            encoder=encoder,
            settle_guard=FROZEN["settle_guard"],
        )

    def base_row(self, kind: str, arm: str, pair_seed: int) -> dict[str, Any]:
        return {
            "ts": time.time(),
            "kind": kind,
            "campaign_id": self.campaign_id,
            "arm": arm,
            "pair_seed": pair_seed,
            "frozen": FROZEN,
            "probe_cap_s": self.probe_cap_s,
            "train_cap_s": self.train_cap_s,
            "pain_edge_min_s": self.pain_edge_min,
            "provenance": self.provenance,
            "apparatus_record_ts": self.apparatus_ts,
            "gate_record_code_hash": self.gate_hash,
            "rss_mb": _rss_mb(),
            "refusal": None,
        }

    def write(self, row: dict[str, Any]) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    # ── the wet apparatus checks, once per pair, on a throwaway agent ──

    def apparatus(self, pair_seed: int, pair_dir: Path) -> dict[str, Any]:
        row = self.base_row("apparatus", "-", pair_seed)
        home = pair_dir / "throwaway"
        shutil.rmtree(home, ignore_errors=True)
        out: dict[str, Any] = {}
        try:
            aut, encoder, pump = _build(self.args, agent_id=f"exp61_throwaway_{pair_seed}", home=home)
            trial = self.trial(aut, encoder, agent_id=f"exp61_throwaway_{pair_seed}", home=home)
            trial.attach_instruments()
            try:
                out["fingerprint_live"] = trial.check_fingerprint(FROZEN["exp60"]["usable_oxygen_max"])
                out["bridge_state_interval_s"] = trial.check_bridge()
                trial.check_gamerules()
                shore_c, water_c = trial.check_clusters_distinct()
                out["clusters_distinct"] = [shore_c, water_c]
                trial.resolve_tools()
                out["actuation"] = trial.check_escape_actuation()
                if out["actuation"]["t_surface"] > FROZEN["actuation_max_s"]:
                    raise Refusal(f"escape actuation {out['actuation']['t_surface']}s > {FROZEN['actuation_max_s']}s")
                if trial.calls:
                    raise Refusal("the actuation check reached the executor — it must go through the bridge only")
            finally:
                trial.detach_instruments()
                trial.final_rescue()
                try:
                    close_and_stage(aut, pump, None)
                except Refusal as exc:
                    print(f"WARNING: throwaway close: {exc}")
                shutil.rmtree(home, ignore_errors=True)
        except (Refusal, InstrumentError) as exc:
            out.update(getattr(exc, "partial", None) or {})
            row["refusal"] = str(exc)
            print(f"REFUSED apparatus pair {pair_seed}: {exc}")
        row["apparatus"] = out
        self.write(row)
        return row

    # ── the donor ──

    def donor(self, pair_seed: int, pair_dir: Path, kind: str) -> dict[str, Any]:
        """Train one donor (fear / ablated) and export it; returns the donor row (also written)."""
        row = self.base_row("donor", kind, pair_seed)
        agent_id = f"exp61_donor_{kind}_{pair_seed}"
        home = pair_dir / f"donor_{kind}"
        stage = pair_dir / f"donor_{kind}_stage"
        shutil.rmtree(home, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
        aut, encoder, pump = _build(self.args, agent_id=agent_id, home=home)
        trial = self.trial(aut, encoder, agent_id=agent_id, home=home)
        trial.attach_instruments()
        staged = False
        try:
            row["fingerprint_live"] = trial.check_fingerprint(FROZEN["exp60"]["usable_oxygen_max"])
            if kind == "ablated":
                detached = _detach_fear_subscriber(aut)
                row["detached_subscribers"] = detached
                if detached != 1:
                    raise Refusal(f"expected exactly 1 fear subscriber, detached {detached}")
            row["bridge_state_interval_s"] = trial.check_bridge()
            row["loop_liveness_ticks"] = trial.check_liveness()
            trial.check_gamerules()
            _shore_pre, water_pre = trial.check_clusters_distinct()
            trial.resolve_tools()
            trial.rescue("donor-ready")
            trial.deaths0 = trial.deaths()
            row["training"], episode_clusters = trial.train()
            row.update(trial.live_g2("fear" if kind == "fear" else "ablated", episode_clusters, water_pre))
            row["training_end_ts"] = time.time()
            # NO donor probe. Stage NOW (export-before-probe as a file fact).
            trial.detach_instruments()
            trial.final_rescue()
            close_and_stage(aut, pump, stage)
            staged = True
            try:
                sanity = donor_sanity_staged(
                    stage,
                    donor_kind=kind,
                    episode_clusters=episode_clusters,
                    shore_node=row["live_g2"]["probe_shore_cluster"],
                )
            except (OSError, ValueError) as exc:
                raise Refusal(f"staged donor files unreadable: {exc}") from exc
            row["donor_sanity"] = sanity
            if not sanity["pass"]:
                raise Refusal("donor sanity on the STAGED files: " + "; ".join(sanity["reasons"]))
            cid = f"exp61-donor-{kind}-{pair_seed}"
            (stage / "donor_meta.json").write_text(
                json.dumps(
                    {
                        "fear_shipped": sanity["fear_shipped"],
                        "fear_keys": sanity["fear_keys"],
                        "nac_sha256": sanity["nac_sha256"],
                        "geometry_tag": sanity["geometry_tag"],
                        "water_node": row["live_g2"]["probe_water_cluster"],
                        "shore_node": row["live_g2"]["probe_shore_cluster"],
                        "saved_at": sanity["saved_at"],
                        "contributor_id": cid,
                    }
                )
            )
            try:
                C.export_bundle(
                    stage, pair_dir / f"{kind}.zip", contributor_id=cid, body_ref=BODY_REF, body_spec_yaml=BODY_YAML
                )
                if kind == "fear":
                    C.export_bundle(
                        stage,
                        pair_dir / "dangling.zip",
                        contributor_id=cid,
                        dangling=True,
                        body_ref=BODY_REF,
                        body_spec_yaml=BODY_YAML,
                    )
            except (RuntimeError, OSError) as exc:  # the CLI's rc != 0 surfaces as RuntimeError — a named refusal
                raise Refusal(f"substrate export failed: {exc}") from exc
            row["contributor_id"] = cid
            row["bundle_sha256"] = hashlib.sha256((pair_dir / f"{kind}.zip").read_bytes()).hexdigest()
        except (Refusal, InstrumentError) as exc:
            row.update(getattr(exc, "partial", None) or {})
            row["refusal"] = str(exc)
            print(f"REFUSED donor {kind} pair {pair_seed}: {exc}")
        finally:
            if not staged:
                trial.detach_instruments()
                trial.final_rescue()
                try:
                    close_and_stage(aut, pump, None)
                except Exception as exc:
                    print(f"WARNING: donor teardown raised: {exc!r}")
        self.write(row)
        return row

    # ── the receiver ──

    def receiver(self, pair_seed: int, pair_dir: Path, arm: str, donor_dir: Path | None) -> dict[str, Any]:
        row = self.base_row("receiver", arm, pair_seed)
        recv_id = f"exp61_recv_{arm}_{pair_seed}"
        home = pair_dir / f"recv_{arm}"
        pre_stage = pair_dir / f"recv_{arm}_pre"
        shutil.rmtree(home, ignore_errors=True)
        shutil.rmtree(pre_stage, ignore_errors=True)
        try:
            donor_meta: dict[str, Any] | None = None
            stage: Path | None = None
            if arm in ARMS_WITH_DONOR:
                if donor_dir is None:
                    raise Refusal("no clean donor for this arm")
                kind = ARMS_WITH_DONOR[arm]
                stage = donor_dir / f"donor_{kind}_stage"
                try:
                    donor_meta = _load_json(stage / "donor_meta.json")
                except (OSError, ValueError) as exc:
                    raise Refusal(f"donor stage unreadable: {exc}") from exc
                row["donor_pair"] = int(donor_dir.name.split("_")[-1])
            # 1. B pre-ingest: no loop, no water, full close + stage
            aut, encoder, pump = _build(self.args, agent_id=recv_id, home=home)
            close_and_stage(aut, pump, pre_stage)
            if _world_ids(pre_stage / "aut_ec.json"):
                raise Refusal("fresh receiver holds world nodes before ingest")
            if _load_json(pre_stage / "aut_nac.json").get("cluster_fear"):
                raise Refusal("fresh receiver holds fear before ingest")
            row["receiver_saved_at_pre"] = _load_json(pre_stage / "aut_nac.json").get("saved_at")
            # 2. ingest (arms with a donor)
            if donor_meta is not None and stage is not None and donor_dir is not None:
                bundle = donor_dir / ("dangling.zip" if arm == "dangling" else f"{ARMS_WITH_DONOR[arm]}.zip")
                row["bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
                row["donor"] = {
                    "contributor_id": donor_meta["contributor_id"],
                    "nac_sha256": donor_meta["nac_sha256"],
                    "fear_shipped": donor_meta["fear_shipped"],
                    "saved_at": donor_meta["saved_at"],
                }
                try:
                    entry = C.ingest_bundle_into(
                        home,
                        bundle,
                        contributor_id=donor_meta["contributor_id"],
                        receiver_agent_id=recv_id,
                        receiver_body=BODY_REF,
                    )
                except (RuntimeError, OSError, ValueError) as exc:  # IngestRefused / gate 7 / journal missing → rc != 0
                    raise Refusal(f"substrate ingest refused: {exc}") from exc
                row["ingest"] = {
                    k: entry.get(k)
                    for k in (
                        "fear_rekeyed",
                        "fear_dropped",
                        "fear_below_floor",
                        "fear_discount",
                        "biases_rekeyed",
                        "biases_dropped",
                        "donor_nodes",
                    )
                }
                bad = ingest_gate(arm, entry, shipped=int(donor_meta["fear_shipped"]))
                if bad:
                    raise Refusal(f"ingest gate: {bad}")
                post_nac = _load_json(home / "nac.json")
                expected_value = round(FROZEN["fear_value_cap"] * FROZEN["foreign_fear_discount"], 6)
                for key, v in (post_nac.get("cluster_fear") or {}).items():
                    aid = str(key).split("\x1f")[0]
                    if aid != recv_id:
                        raise Refusal(f"post-ingest fear key carries agent id {aid!r}, not the receiver's")
                    if arm == "transferred" and round(float(v), 6) != expected_value:
                        raise Refusal(f"folded fear value {v} != {expected_value} (post-discount cap)")
                if arm != "dangling":
                    donor_ids = _world_ids(stage / "aut_ec.json")
                    post_ids = _world_ids(home / "ec.json")
                    if not donor_ids <= post_ids or len(post_ids) != len(donor_ids):
                        raise Refusal(
                            f"id_map is not the identity on a fresh receiver: donor {len(donor_ids)} world nodes, "
                            f"receiver {len(post_ids)} after ingest"
                        )
                row["receiver_saved_at_post"] = post_nac.get("saved_at")
            # 3. B reboots; shore only
            aut, encoder, pump = _build(self.args, agent_id=recv_id, home=home)
            trial = self.trial(aut, encoder, agent_id=recv_id, home=home)
            trial.attach_instruments()
            try:
                trial.resolve_tools()
                if trial.positive_escape_links() or (
                    trial.flee_tool and aut.bio.nac.get_positive_outcomes(f"tool:{trial.flee_tool}")
                ):
                    raise Refusal("receiver carries a positive escape/flee link before first contact")
                row["loop_liveness_ticks"] = trial.check_liveness()
                if trial.calls:
                    raise Refusal(
                        f"executor call(s) during the shore liveness window: {[c['tool'] for c in trial.calls]}"
                    )
                trial.rescue("shore-tag")
                shore_node = trial.encode_world_cluster()
                live_tag = aut.bio.ec._substrate_node_geometries.get(shore_node)
                if donor_meta is not None and arm != "dangling" and live_tag != donor_meta["geometry_tag"]:
                    raise Refusal(
                        f"geometry tag mismatch: receiver live {live_tag} vs donor {donor_meta['geometry_tag']}"
                    )
                row["geometry_tag"] = live_tag
                # 4. representation + readability gate (ONE loop-OFF submersion, US-free)
                n_sig = len(trial.signals)
                trial.submerge("representation-gate")
                water_node = trial.encode_world_cluster()
                vm = trial.aut.client.latest_state() or {}
                oxygen_at_gate = float(vm.get("oxygen", 20.0))  # one read after the encode (not a dwell minimum)
                trial.rescue("representation-gate")
                pain = [s for s in trial.signals[n_sig:] if s["failure_mode"] in (FEAR_MODE, "drive:health")]
                need = aut.bio.nac.anticipatory_threat_need(recv_id, {"world": water_node})
                shore_fear = aut.bio.nac.cluster_fear(recv_id, shore_node)
                water_fear = aut.bio.nac.cluster_fear(recv_id, water_node)
                transferred_node = None if donor_meta is None else donor_meta["water_node"]
                gate = {
                    "water_node": water_node,
                    "shore_node": shore_node,
                    "transferred_node": transferred_node,
                    "need": need,
                    "water_fear": water_fear,
                    "shore_fear": shore_fear,
                    "oxygen_at_gate": oxygen_at_gate,
                    "pain_publishes": len(pain),
                    "pass": False,
                }
                row["representation_gate"] = gate
                if pain:
                    raise Refusal("the representation gate published pain — not US-free")
                if water_node == shore_node:
                    raise Refusal("shore and water encode to one live cluster on the receiver")
                if arm in ("transferred", "cluster_not_fear") and water_node != transferred_node:
                    raise Refusal(
                        f"receiver's submerged reading completed into {str(water_node)[:8]}, not the transferred node "
                        f"{str(transferred_node)[:8]}"
                    )
                if arm == "transferred" and not need > FROZEN["read_floor"]:
                    raise Refusal(f"transferred fear not readable: need {need} <= {FROZEN['read_floor']}")
                if arm != "transferred" and need != 0.0:
                    raise Refusal(f"fear readable in arm {arm}: need {need}")
                if shore_fear != 0.0:
                    raise Refusal(f"fear on the shore node ({shore_fear}) — specificity")
                gate["pass"] = True
                trial.deaths0 = trial.deaths()
                # 5. FIRST CONTACT — the DV, one placement
                with C.RecommendCapture() as cap:
                    n0 = len(cap.events)
                    pl = trial.placement("first-contact")
                    events = [{**dict(e.get("data", {})), "t": e.get("t")} for e in cap.events[n0:]]
                # the EXECUTED proposal: the first escape-best event that PASSED the gate (a sub-threshold
                # escape-best event is never executed); its sim-log `t` is kept beside t_escape_call
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
                row["positive_escape_links_after"] = trial.positive_escape_links()
                # 6. one further placement, recorded, never gated
                row["second_placement"] = trial.placement("second")
                trial.check_death_cap()
                print(
                    f"pair {pair_seed} {arm}: first contact success={fc['success']} (decision={fc['decision_dv']}, "
                    f"air={fc['behavioural_dv']}, decisive={fc['decisive']}) t_escape={fc['t_escape_call']} t_air={fc['t_first_air']}"
                )
            finally:
                trial.detach_instruments()
                trial.final_rescue()
                try:
                    close_and_stage(aut, pump, pair_dir / f"recv_{arm}_post")
                except Exception as exc:
                    print(f"WARNING: receiver teardown raised: {exc!r}")
        except (Refusal, InstrumentError) as exc:
            row.update(getattr(exc, "partial", None) or {})
            row["refusal"] = str(exc)
            print(f"REFUSED receiver {arm} pair {pair_seed}: {exc}")
        self.write(row)
        return row


def _existing_clean(out_path: Path, campaign_id: str) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not out_path.is_file():
        return done
    for ln in out_path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("campaign_id") == campaign_id and r.get("refusal") is None:
            done.add((r["kind"], r["arm"], int(r["pair_seed"])))
    return done


def _run(args: argparse.Namespace) -> int:
    out_arg = Path(args.out)
    out_abs = out_arg if out_arg.is_absolute() else (C.REPO_ROOT / out_arg)
    out_path = evidence_out_paths_or_exit(
        C.REPO_ROOT,
        [str(out_abs)],
        write_experiment_results=args.write_experiment_results,
        allow_dirty=args.allow_dirty,
    )[0]
    import maxim

    try:
        provenance = in_process_code_provenance(
            C.REPO_ROOT, maxim.__file__, out_path=out_path, allow_dirty=args.allow_dirty
        )
    except (DirtyTreeError, ProvenanceError) as exc:
        print(f"[FAIL] provenance: {exc}")
        return 3
    campaign_id = args.campaign_id or uuid.uuid4().hex[:12]
    if args.resume and not args.campaign_id:
        print("[FAIL] --resume requires --campaign-id")
        return 2
    camp = _Campaign(args, provenance=provenance, out_path=out_path, campaign_id=campaign_id)
    done = _existing_clean(out_path, campaign_id) if args.resume else set()
    kit_done = any(k[0] == "anti_vacuity" for k in done)
    print(
        f"campaign {campaign_id}: workdir {camp.workdir} probe cap {camp.probe_cap_s:.2f}s train cap {camp.train_cap_s:.2f}s; "
        f"{len(done)} clean rows already"
    )
    exit_code = 0
    prev_cwd = os.getcwd()
    try:
        n_full = FROZEN["arms"]["transferred"]
        for idx, pair_seed in enumerate(FROZEN["pair_seeds"][: args.pairs]):
            pair_dir = camp.workdir / f"pair_{pair_seed}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            # The loop writes cwd-relative (`init_prefetcher(base_path=os.getcwd())`, `FileSystemEnv`):
            # one pair's throwaway, donors and receivers share this cwd, as Exp 60's seeds shared theirs.
            os.chdir(pair_dir)
            arms, donor_of = pair_plan(idx, pair_seed, n_full=n_full, offset=FROZEN["dangling_donor_offset"])
            # apparatus checks ONCE per pair
            if ("apparatus", "-", pair_seed) not in done:
                print(f"\n=== pair {pair_seed}: apparatus ===")
                if camp.apparatus(pair_seed, pair_dir).get("refusal") is not None:
                    exit_code = 4
                    continue  # nothing this pair measures is trustworthy; --resume retries the pair
            # donors this pair trains itself
            for kind, arm in (("fear", "transferred"), ("ablated", "cluster_not_fear")):
                if arm not in arms or donor_of.get(arm) != pair_seed:
                    continue
                if ("donor", kind, pair_seed) in done and (
                    pair_dir / f"donor_{kind}_stage" / "donor_meta.json"
                ).is_file():
                    continue
                print(f"\n=== pair {pair_seed}: donor {kind} ===")
                if camp.donor(pair_seed, pair_dir, kind).get("refusal") is not None:
                    exit_code = 4
            for arm in arms:
                if ("receiver", arm, pair_seed) in done:
                    continue
                donor_dir = None
                if arm in ARMS_WITH_DONOR:
                    cand = camp.workdir / f"pair_{donor_of[arm]}"
                    if (cand / f"donor_{ARMS_WITH_DONOR[arm]}_stage" / "donor_meta.json").is_file():
                        donor_dir = cand
                print(f"\n=== pair {pair_seed}: receiver {arm} ===")
                r = camp.receiver(pair_seed, pair_dir, arm, donor_dir)
                if r.get("refusal") is not None:
                    exit_code = 4
                elif arm == "transferred" and not kit_done:
                    kit_row = camp.base_row("anti_vacuity", arm, pair_seed)
                    try:
                        kit_row["kit"] = anti_vacuity_kit(
                            pair_dir / "donor_fear_stage",
                            pair_dir / f"recv_{arm}_pre",
                            receiver_agent_id=f"exp61_recv_{arm}_{pair_seed}",
                        )
                    except (OSError, ValueError, RuntimeError) as exc:
                        kit_row["refusal"] = f"anti-vacuity kit failed: {exc}"
                    camp.write(kit_row)
                    kit_done = kit_row.get("refusal") is None and bool(kit_row["kit"]["pass"])
    finally:
        os.chdir(prev_cwd)
        camp.rcon.close()
    print(f"\ncampaign {campaign_id} -> {out_path}")
    return exit_code


def _verdict(args: argparse.Namespace) -> int:
    rows = [json.loads(ln) for ln in Path(args.data).expanduser().read_text().splitlines() if ln.strip()]
    v = compute_verdict(rows, campaign_id=args.campaign_id)
    v["data"] = str(args.data)
    print(json.dumps({k: v[k] for k in v if k not in ("gates",)}, indent=2))
    print(f"VERDICT: {v['verdict']}" + (f" ({v['incomplete_cause']})" if v.get("incomplete_cause") else ""))
    if args.json:
        out_arg = Path(args.json)
        out_abs = out_arg if out_arg.is_absolute() else (C.REPO_ROOT / out_arg)
        out = evidence_out_paths_or_exit(
            C.REPO_ROOT,
            [str(out_abs)],
            write_experiment_results=args.write_experiment_results,
            allow_dirty=args.allow_dirty,
        )[0]
        import maxim

        v["provenance"] = in_process_code_provenance(
            C.REPO_ROOT, maxim.__file__, out_path=out, allow_dirty=args.allow_dirty
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(v, indent=2) + "\n")
        print(f"[verdict] wrote -> {out}")
    return 0 if v["verdict"] == "EARNED" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the live campaign (all arms, pairs interleaved)")
    r.add_argument("--rcon-host", default="127.0.0.1")
    r.add_argument("--rcon-port", type=int, default=25575)
    r.add_argument("--rcon-password", required=True)
    r.add_argument("--username", default="maxim")
    r.add_argument("--bridge-host", default="127.0.0.1")
    r.add_argument("--bridge-port", type=int, default=25567)
    r.add_argument("--workdir", required=True, help="DURABLE per-pair homes, stages and bundles (needed by --resume)")
    r.add_argument("--pairs", type=int, default=len(FROZEN["pair_seeds"]))
    r.add_argument("--campaign-id", default=None)
    r.add_argument("--resume", action="store_true", help="skip (kind, arm, pair) rows already clean for --campaign-id")
    r.add_argument("--gate-record", default=GATE_RECORD)
    r.add_argument("--out", default="docs/experiments/data/exp61_pairs.jsonl")
    r.add_argument("--write-experiment-results", action="store_true")
    r.add_argument("--allow-dirty", action="store_true")
    r.set_defaults(func=_run)
    v = sub.add_parser("verdict", help="pure verdict over the per-(pair, arm) rows")
    v.add_argument("--data", required=True)
    v.add_argument("--json", default=None)
    v.add_argument("--campaign-id", default=None)
    v.add_argument("--write-experiment-results", action="store_true")
    v.add_argument("--allow-dirty", action="store_true")
    v.set_defaults(func=_verdict)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
