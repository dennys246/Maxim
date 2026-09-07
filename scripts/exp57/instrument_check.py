#!/usr/bin/env python3
"""Exp 57 Phase 0 — the instrument checks + apparatus-constant calibration.

Pre-registration §Phase 0 (all six checks must pass or the campaign does not
start; the apparatus constants K_max/C/W are PROPOSED here for the operator to
freeze by a pre-campaign amendment). Writes ``57_phase0.json`` through the
gated-evidence path. Exit 4 on any failing check (the S3 refusal convention).

Checks (prereg §Phase 0):

1.  G = 4 discriminability, PAIRWISE — the four slots separate from REST AND
    from each other (distinct clusters).
2.  Merge-alignment sanity + the quantitative shared-vs-union split — two
    contributors that learned the SAME contingency fold to a SHARED key; two
    that learned DIFFERENT ones fold to a UNION.
2b. Contributor divergence — the coverage-widening tripwire: independent-seed
    contributors cover DIFFERENT contingencies first (Jaccard < 1, union grows).
3.  L12 zero-prior — ``score_components["drive"] == 0.0`` on every probe
    (asserted inside the coverage read; reported here).
4.  Calibration — the single-contributor coverage-vs-trial curve; PROPOSES
    K_max/C/W (target C = 3/4, W = 3, K_max so N = 1 sits ~1/2). NOT hardcoded
    — printed for the operator to freeze by amendment.
5.  Pilot ladder — a few cohorts at N in {1, 8} confirming tau(8) < tau(1) is
    OBTAINABLE and giving a tau-spread sanity check (the >= 20 count is fixed by
    the prereg, never chosen from the pilot).

LIVE mode is the confirmatory Phase 0; ``--mock`` exercises the same checks
against the deterministic ScriptedBridgeServer for harness development.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from _provenance import (  # noqa: E402
    assert_repo_interpreter,
    evidence_out_paths,
    executed_code_provenance,
    preflight_gated_record_or_exit,
)
from exp56 import common as C  # noqa: E402
from exp57 import common57 as X  # noqa: E402

DISCRIMINABILITY_BAR = 0.70  # prereg check 1 (separation + stability)
PILOT_K_MAX = 64  # calibration scan depth (proposal only; NOT the frozen K_max)
PILOT_REPS = 1
PILOT_COHORTS = 5  # check 5: >= 5 cohorts (prereg)


def _train(session, world, *, seed, slot_to_target, bot, k_max, settle, teach=True):
    return X.train_contributor_with_snapshots(
        session,
        world=world,
        contributor_seed=seed,
        slot_to_target=slot_to_target,
        bot_name=bot,
        k_max=k_max,
        reps_per_cell=PILOT_REPS,
        settle_s=settle,
        teach=teach,
    )


def check_discriminability(bridge_port, world, bot, work, *, settle) -> dict:
    """Check 1: each slot separates from REST (>= 0.70) AND the four situation
    clusters are pairwise distinct."""
    session = C.build_bench_session(
        agent_id="phase0_disc", bridge_port=bridge_port, home=work / "disc_home", pair_seed=1, body_ref=X.BODY_REF57
    )
    situation_ids: dict[int, str] = {}
    transitions = separated = repeats = stable = 0
    for g, slot in enumerate(X.CONTINGENCY_SLOTS):
        seq: list[tuple[bool, str | None]] = []
        for _cycle in range(5):
            for situation in (False, True, True):  # rest, onset, repeat
                world.teleport(bot, slot if situation else C.FROZEN["rest_anchor"])
                time.sleep(settle)
                session.sync_world()
                clusters = session.encode_clusters()
                seq.append((situation, clusters.get("world")))
        for i in range(1, len(seq)):
            (psit, pid), (sit, cid) = seq[i - 1], seq[i]
            if pid is None or cid is None:
                continue
            if not psit and sit:
                transitions += 1
                separated += int(cid != pid)
            elif psit and sit:
                repeats += 1
                stable += int(cid == pid)
            if sit and cid is not None:
                situation_ids[g] = cid
    C.close_and_stage_session(session, stage_dir=work / "disc_stage")
    separation = separated / transitions if transitions else 0.0
    stability = stable / repeats if repeats else 0.0
    distinct = len(set(situation_ids.values())) == len(situation_ids) and len(situation_ids) == X.G
    return {
        "separation": round(separation, 4),
        "stability": round(stability, 4),
        "pairwise_distinct_situation_clusters": distinct,
        "situation_cluster_count": len(set(situation_ids.values())),
        "bar": DISCRIMINABILITY_BAR,
        "pass": separation >= DISCRIMINABILITY_BAR and stability >= DISCRIMINABILITY_BAR and distinct,
    }


def _covered_set(snaps_last, *, receiver_agent_id, slot_to_target, work) -> set[int]:
    """The set of contingencies a single contributor covers at its budget."""
    merged = X.fold_snapshots([snaps_last], receiver_agent_id, workdir=work, contributor_ids=["c0"])
    ec_nodes = json.loads((work / "recv" / "ec.json").read_text()).get("substrate_nodes", {})
    clusters = X.contingency_clusters_from_ec_nodes(ec_nodes, receiver_agent_id=receiver_agent_id)
    _cov, detail = X.coverage(merged, clusters, slot_to_target, receiver_agent_id=receiver_agent_id, return_detail=True)
    return {g for g, v in detail.items() if v}


def check_drive_zero(bridge_port, world, bot, work, *, settle) -> dict:
    """Check 3 — L12 zero-prior, made real (not the old hardcoded pass:True).

    Trains ONE contributor a short budget, folds 1->1, and runs the coverage
    probe — which asserts ``score_components['drive'] == 0`` per decision
    (common57.coverage). A nonzero drive component at d1=0 would raise
    AssertionError, caught here and reported as a FAILING check rather than
    crashing the run. A can't-fail check is not a check (the D62 lesson the
    code lens flagged).
    """
    cohort_seed = 30303
    slot_to_target = X.cohort_slot_to_target(cohort_seed)
    seed = X.contributor_seeds(cohort_seed, 1, 1, salt=9)[0]
    k = 2 * len(C.AFFORDANCES) * PILOT_REPS * 2  # ~2 contingency blocks
    home = work / "drivezero_home"
    session = C.build_bench_session(
        agent_id="dz-c0", bridge_port=bridge_port, home=home, pair_seed=seed, body_ref=X.BODY_REF57
    )
    snaps = _train(session, world, seed=seed, slot_to_target=slot_to_target, bot=bot, k_max=k, settle=settle)
    C.close_and_stage_session(session, stage_dir=work / "dz_stage")
    try:
        _covered_set(
            snaps[-1],
            receiver_agent_id="recv-dz",
            slot_to_target=slot_to_target,
            work=work / "dz_fold",
        )
        return {"pass": True, "note": "score_components['drive']==0 held on every coverage probe decision"}
    except AssertionError as exc:
        return {"pass": False, "error": str(exc)}


def check_alignment_and_divergence(bridge_port, world, bot, work, *, settle) -> dict:
    """Checks 2 + 2b: alignment shared-vs-union split, and contributor
    divergence (Jaccard < 1, union grows N=1->pilot)."""
    cohort_seed = 9001
    slot_to_target = X.cohort_slot_to_target(cohort_seed)
    pilot_n = 3
    seeds = X.contributor_seeds(cohort_seed, pilot_n, pilot_n, salt=99)
    covered: list[set[int]] = []
    finals: list = []
    for i, s in enumerate(seeds):
        session = C.build_bench_session(
            agent_id=f"phase0_div_{i}",
            bridge_port=bridge_port,
            home=work / f"div_{i}_home",
            pair_seed=s,
            body_ref=X.BODY_REF57,
        )
        snaps = _train(session, world, seed=s, slot_to_target=slot_to_target, bot=bot, k_max=PILOT_K_MAX, settle=settle)
        C.close_and_stage_session(session, stage_dir=work / f"div_{i}_close")
        finals.append(snaps[-1])
        covered.append(
            _covered_set(
                snaps[-1],
                receiver_agent_id=f"recv-div-{i}",
                slot_to_target=slot_to_target,
                work=work / f"div_{i}_fold",
            )
        )

    # 2b: mean pairwise Jaccard + union growth.
    jaccards = []
    for a, b in itertools.combinations(range(pilot_n), 2):
        inter = len(covered[a] & covered[b])
        uni = len(covered[a] | covered[b]) or 1
        jaccards.append(inter / uni)
    mean_jaccard = statistics.mean(jaccards) if jaccards else 1.0
    union_1 = len(covered[0])
    union_all = len(set().union(*covered)) if covered else 0

    # 2: fold two SAME-contingency contributors vs a same-vs-different check by
    # inspecting the merged bias-key count against the per-contributor counts.
    merged_two = X.fold_snapshots(
        [finals[0], finals[1]], "recv-align", workdir=work / "align_fold", contributor_ids=["c0", "c1"]
    )
    keys0 = set((finals[0][0].get("cluster_reward_bias") or {}).keys())
    keys1 = set((finals[1][0].get("cluster_reward_bias") or {}).keys())
    merged_keys = set((merged_two.get("cluster_reward_bias") or {}).keys())
    # After re-key, shared contingencies collapse to one key (convex-combined);
    # distinct ones union. So merged_keys <= |keys0| + |keys1| and shared > 0
    # iff any contingency was covered by both.
    shared_estimate = max(0, len(keys0) + len(keys1) - len(merged_keys))
    # 2b bar strengthened (methodology-lens finding 7): `jaccard < 1.0` only
    # trips on BYTE-identical contributors — 95%-identical ones (which still
    # force a near-flat curve) would pass. Require the mean pairwise Jaccard
    # BELOW a real ceiling so contributors "genuinely differ" (the prereg's
    # word), not merely differ by one element.
    JACCARD_MAX = 0.8
    # 2 gated (not just reported): the merge must actually SHARE at least one key
    # (over-aligning collapses to all-shared; under-aligning never averages, so
    # shared==0 with distinct contingencies is the under-alignment failure). With
    # G=4 slots and 3 contributors covering overlapping-but-distinct sets, some
    # sharing AND some union is the healthy signal.
    aligns = shared_estimate > 0
    return {
        "mean_pairwise_jaccard": round(mean_jaccard, 4),
        "jaccard_max": JACCARD_MAX,
        "union_N1": union_1,
        "union_Npilot": union_all,
        "merged_bias_keys": len(merged_keys),
        "shared_key_estimate": shared_estimate,
        "union_grows": union_all > union_1,
        "merge_shares_a_key": aligns,
        # Healthy apparatus: contributors genuinely differ (Jaccard below the
        # ceiling), the union widens N=1->pilot (coverage CAN widen), AND the
        # merge shares at least one aligned key (not under-aligning). Over-
        # alignment (Jaccard high, union flat) and under-alignment (no shared
        # key) are both apparatus failures caught here, not at campaign price.
        "pass": (mean_jaccard < JACCARD_MAX) and (union_all > union_1) and aligns,
    }


def check_calibration(bridge_port, world, bot, work, *, settle) -> dict:
    """Check 4: single-contributor coverage-vs-trial curve; PROPOSE K_max/C/W.

    The proposal targets C = 3/4, W = 3, and K_max where a single contributor
    sits ~1/2 (2 of 4). Printed for the operator to freeze by amendment; the
    harness does NOT hardcode the final K_max.
    """
    cohort_seed = 4242
    slot_to_target = X.cohort_slot_to_target(cohort_seed)
    seed = X.contributor_seeds(cohort_seed, 1, 1, salt=4)[0]
    session = C.build_bench_session(
        agent_id="phase0_cal", bridge_port=bridge_port, home=work / "cal_home", pair_seed=seed, body_ref=X.BODY_REF57
    )
    snaps = _train(session, world, seed=seed, slot_to_target=slot_to_target, bot=bot, k_max=PILOT_K_MAX, settle=settle)
    C.close_and_stage_session(session, stage_dir=work / "cal_close")

    curve: list[float] = []
    for t in range(len(snaps)):
        merged = X.fold_snapshots([snaps[t]], "recv-cal", workdir=work / f"cal_fold_{t}", contributor_ids=["c0"])
        ec_nodes = json.loads((work / f"cal_fold_{t}" / "recv" / "ec.json").read_text()).get("substrate_nodes", {})
        clusters = X.contingency_clusters_from_ec_nodes(ec_nodes, receiver_agent_id="recv-cal")
        curve.append(X.coverage(merged, clusters, slot_to_target, receiver_agent_id="recv-cal"))

    c_target = X.CRITERION_TARGET
    w_target = X.WINDOW_TARGET
    # Propose K_max: the first t where single-contributor coverage reaches ~1/2
    # (partway, below the ceiling). If it never reaches 1/2, the body/schedule
    # is too coarse — an apparatus failure surfaced here (a proceed would be
    # gaming the curve).
    half = 0.5
    proposed_k_max: int | None = None
    for t, cov in enumerate(curve, start=1):
        if cov >= half:
            proposed_k_max = t
            break
    endpoint = curve[-1] if curve else 0.0
    return {
        "coverage_curve": [round(c, 4) for c in curve],
        "endpoint_coverage": round(endpoint, 4),
        "proposed_K_max": proposed_k_max,
        "proposed_C": c_target,
        "proposed_W": w_target,
        "note": (
            "PROPOSAL ONLY — freeze K_max/C/W by a pre-campaign amendment. "
            "Target: single contributor ~1/2 at K_max; C = 3/4; W = 3."
        ),
        # The window must exist: a single contributor must sit partway (reach
        # 1/2 but stay below C at the proposed budget).
        "pass": proposed_k_max is not None and curve[proposed_k_max - 1] < c_target,
    }


def check_pilot_ladder(bridge_port, world, bot, work, *, settle) -> dict:
    """Check 5: a few cohorts at N in {1, 8} — tau(8) < tau(1) OBTAINABLE."""
    tau_by_rung: dict[int, list[int]] = {1: [], 8: []}
    for cohort in range(PILOT_COHORTS):
        cohort_seed = 7000 + cohort
        slot_to_target = X.cohort_slot_to_target(cohort_seed)
        for rung in (1, 8):
            seeds = X.contributor_seeds(cohort_seed, rung, rung, salt=5)
            per_contrib = []
            for i, s in enumerate(seeds):
                session = C.build_bench_session(
                    agent_id=f"pilot_{cohort}_{rung}_{i}",
                    bridge_port=bridge_port,
                    home=work / f"pilot_{cohort}_{rung}_{i}_home",
                    pair_seed=s,
                    body_ref=X.BODY_REF57,
                )
                per_contrib.append(
                    _train(
                        session, world, seed=s, slot_to_target=slot_to_target, bot=bot, k_max=PILOT_K_MAX, settle=settle
                    )
                )
                C.close_and_stage_session(session, stage_dir=work / f"pilot_{cohort}_{rung}_{i}_close")
            cov_series = []
            for t in range(PILOT_K_MAX):
                snaps_t = [pc[t] for pc in per_contrib]
                fd = work / f"pilot_{cohort}_{rung}_fold_{t}"
                merged = X.fold_snapshots(
                    snaps_t, f"recv-pilot-{cohort}-{rung}", workdir=fd, contributor_ids=[f"c{i}" for i in range(rung)]
                )
                ec_nodes = json.loads((fd / "recv" / "ec.json").read_text()).get("substrate_nodes", {})
                clusters = X.contingency_clusters_from_ec_nodes(
                    ec_nodes, receiver_agent_id=f"recv-pilot-{cohort}-{rung}"
                )
                cov_series.append(
                    X.coverage(merged, clusters, slot_to_target, receiver_agent_id=f"recv-pilot-{cohort}-{rung}")
                )
            tau_by_rung[rung].append(X.tau(cov_series, X.CRITERION_TARGET, X.WINDOW_TARGET, PILOT_K_MAX))
    med1 = statistics.median(tau_by_rung[1]) if tau_by_rung[1] else None
    med8 = statistics.median(tau_by_rung[8]) if tau_by_rung[8] else None
    obtainable = med1 is not None and med8 is not None and med8 < med1
    return {
        "tau_rung1": tau_by_rung[1],
        "tau_rung8": tau_by_rung[8],
        "median_tau_1": med1,
        "median_tau_8": med8,
        "tau8_lt_tau1_obtainable": obtainable,
        "cohorts": PILOT_COHORTS,
        # A pilot that does not obtain the effect is an early falsifier SIGNAL,
        # recorded — the campaign may still run to quantify it (prereg outcome
        # tree), so this check reports but does not by itself block. The gate is
        # plumbing integrity: every rung produced a tau.
        "pass": all(len(v) == PILOT_COHORTS for v in tau_by_rung.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/experiments/data/57_phase0.json")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=25580)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=25575)
    ap.add_argument("--rcon-password", default=os.environ.get("EXP57_RCON_PASSWORD", ""))
    ap.add_argument("--bot-name", default=os.environ.get("EXP57_BOT_NAME", "maxim_bench"))
    ap.add_argument("--settle-s", type=float, default=0.6)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    if os.environ.get("MAXIM_OPERANT_ONLY_CREDIT") != "1":
        print("error: MAXIM_OPERANT_ONLY_CREDIT=1 is required (frozen apparatus). Export it and re-run.")
        return 3

    assert_repo_interpreter(C.REPO_ROOT, "maxim", exempt=args.mock)
    out_path = evidence_out_paths(
        C.REPO_ROOT, [args.out], write_experiment_results=args.write_experiment_results, allow_dirty=args.allow_dirty
    )[0]
    preflight = preflight_gated_record_or_exit(C.REPO_ROOT, out_path, allow_dirty=args.allow_dirty)
    provenance = executed_code_provenance(C.REPO_ROOT, "maxim", out_path=out_path, allow_dirty=args.allow_dirty)

    if args.mock:
        server = C.ScriptedBridgeServer(seed=1)
        world = C.ScriptedWorldControl(server, settle_s=0.08)
        bridge_port = server.port
        settle = 0.02
    else:
        server = None
        world = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
        bridge_port = args.bridge_port
        settle = args.settle_s

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="exp57_phase0_"))
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {"ts": time.time(), "mock": bool(args.mock), "G": X.G, "slots": X.CONTINGENCY_SLOTS}
    report.update(preflight)
    report["provenance"] = provenance
    try:
        report["check1_discriminability"] = check_discriminability(
            bridge_port, world, args.bot_name, work, settle=settle
        )
        report["check2_2b_alignment_divergence"] = check_alignment_and_divergence(
            bridge_port, world, args.bot_name, work, settle=settle
        )
        report["check3_drive_zero"] = check_drive_zero(bridge_port, world, args.bot_name, work, settle=settle)
        report["check4_calibration"] = check_calibration(bridge_port, world, args.bot_name, work, settle=settle)
        report["check5_pilot_ladder"] = check_pilot_ladder(bridge_port, world, args.bot_name, work, settle=settle)
    finally:
        world.close()
        if server is not None:
            server.close()
        if args.workdir is None:
            shutil.rmtree(work, ignore_errors=True)

    check_keys = [k for k in report if k.startswith("check")]
    all_pass = all(report[k]["pass"] for k in check_keys)
    report["all_pass"] = all_pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k].get("pass") for k in check_keys}, indent=2))
    print(f"phase0: {'PASS' if all_pass else 'FAIL'} -> {out_path}")
    if "check4_calibration" in report:
        cal = report["check4_calibration"]
        print(f"calibration PROPOSAL: K_max={cal['proposed_K_max']} C={cal['proposed_C']} W={cal['proposed_W']}")
    return 0 if all_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
