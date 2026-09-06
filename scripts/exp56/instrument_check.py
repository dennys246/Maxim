#!/usr/bin/env python3
"""Exp 56 Phase 0 — the five instrument checks that gate the campaign.

Pre-registration §Phase 0 (all five must pass or the campaign does not
start; readings are disclosed as a prereg amendment entry before the
campaign). Writes ``56_phase0_<stamp>.json`` through the gated-evidence
path. Exit 4 on any failing check (the S3 refusal convention).

LIVE mode is the confirmatory Phase 0 (Paper server + bridge + RCON);
``--mock`` exercises the same checks against the deterministic
ScriptedBridgeServer for harness development only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
from exp56.run_campaign import run_pair_arm  # noqa: E402

DISCRIMINABILITY_BAR = 0.70  # prereg Phase-0 check 1 (both separation and stability)
FLOOR_CONCENTRATION_MAX = 0.5  # check 4
FLOOR_PROBES = 10
L1_MARGIN = 0.11


def check_discriminability(bridge_port: int, world, bot: str, work: Path, *, settle: float) -> dict:
    """Check 1: >= 20 rest->situation onsets through the production encode;
    separation (onset changes cluster) and stability (repeat re-completes)
    both >= 0.70 at the design's own two-situation contrast."""
    session = C.build_bench_session(
        agent_id="phase0_disc", bridge_port=bridge_port, home=work / "disc_home", pair_seed=1
    )
    slot = C.FROZEN["contingency_slots"][0]
    ids: list[tuple[bool, str | None]] = []
    for cycle in range(20):
        for situation in (False, True, True):  # rest, onset, repeat
            world.teleport(bot, slot if situation else C.FROZEN["rest_anchor"])
            time.sleep(settle)
            session.sync_world()
            clusters = session.encode_clusters()
            ids.append((situation, clusters.get("world")))
        _ = cycle
    transitions = 0
    separated = 0
    repeats = 0
    stable = 0
    for i in range(1, len(ids)):
        prev_sit, prev_id = ids[i - 1]
        sit, cid = ids[i]
        if prev_id is None or cid is None:
            continue
        if not prev_sit and sit:
            transitions += 1
            separated += int(cid != prev_id)
        elif prev_sit and sit:
            repeats += 1
            stable += int(cid == prev_id)
    C.close_and_stage_session(session, stage_dir=work / "disc_stage")
    separation = separated / transitions if transitions else 0.0
    stability = stable / repeats if repeats else 0.0
    return {
        "transitions": transitions,
        "separation": round(separation, 4),
        "repeats": repeats,
        "stability": round(stability, 4),
        "bar": DISCRIMINABILITY_BAR,
        "pass": separation >= DISCRIMINABILITY_BAR and stability >= DISCRIMINABILITY_BAR,
    }


def check_pilots(bridge_port: int, world, bot: str, work: Path, *, settle: float) -> dict:
    """Check 2 (+5): one taught pilot pair end-to-end (donor sanity at the
    frozen K = check 5; L1 margin; bias-decisive first contact), one
    DANGLING pilot (the link-channel tripwire), and the no-op kit."""
    cfg = C.pair_config(9001)
    pair_work = work / "pilot_pair"
    pair_work.mkdir(parents=True, exist_ok=True)
    taught = run_pair_arm(
        cfg=cfg,
        arm="taught",
        work=pair_work,
        bridge_port=bridge_port,
        world=world,
        bot=bot,
        settle=settle,
        keep_artifacts=True,
        artifacts_dir=work / "pair0_artifacts",
    )
    dangling = run_pair_arm(
        cfg=cfg,
        arm="dangling",
        work=pair_work,
        bridge_port=bridge_port,
        world=world,
        bot=bot,
        settle=settle,
        keep_artifacts=False,
    )
    fc = taught["first_contact"]
    margin = (fc.get("provenance") or {}).get("learned_margin")
    artifacts = work / "pair0_artifacts"
    kit = C.noop_variant_readout(
        bundle=artifacts / "taught.zip",
        receiver_pre_nac=_load_stripped(artifacts / "receiver_pre_nac.json"),
        receiver_pre_ec=json.loads((artifacts / "receiver_pre_ec.json").read_text()).get("substrate_nodes", {}),
        receiver_agent_id=f"recv_taught_{cfg['pair_seed']}",
        contributor_id=cfg["contributor_id"],
        first_contact=fc,
        target_tool=f"{C.ENTITY_NAME}_{cfg['target_aff']}",
    )
    # The tripwire must be able to fail (review I7/ex-4: the old
    # chose_target AND bias_decisive conjunction was structurally vacuous —
    # a dangling arm cannot be bias-decisive by construction). A
    # SUBSTRATE-SOURCED target pick at first contact is the link-channel
    # signature; an epsilon pick of the target is the floor doing floor
    # things. The winning components are recorded either way.
    dangling_fc = dangling["first_contact"]
    dangling_moved = bool(dangling["chose_target"] and dangling_fc.get("source") == "substrate")
    return {
        "taught_first_contact": {k: fc.get(k) for k in ("chosen", "source", "substrate_confidence")},
        "taught_decisive": taught["bias_decisive"],
        "taught_margin": margin,
        "donor_sanity": taught.get("donor_sanity"),
        "dangling_chose_target": dangling["chose_target"],
        "dangling_source": dangling_fc.get("source"),
        "dangling_components": (dangling_fc.get("provenance") or {}).get("score_components"),
        "dangling_decisive": dangling["bias_decisive"],
        "dangling_ingest": dangling.get("ingest"),
        "noop_kit": kit,
        "pass": (
            bool(taught["chose_target"])
            and bool(taught["bias_decisive"])
            and margin is not None
            and float(margin) > L1_MARGIN
            and not dangling_moved
            and bool(kit.get("kit_pass"))
            and bool((taught.get("donor_sanity") or {}).get("pass"))
        ),
    }


def _load_stripped(path: Path) -> dict:
    data = json.loads(path.read_text())
    data.pop("_format_version", None)
    return data


def check_floor(bridge_port: int, world, bot: str, work: Path, *, settle: float) -> dict:
    """Check 4: >= 10 dithered isolated probes — no choice at >= 0.5 and
    not seed-invariant. (Check 3, drive == 0, is asserted inside every
    probe by the harness itself — a violation raises before this report.)"""
    choices: list[str] = []
    for i in range(FLOOR_PROBES):
        cfg = C.pair_config(9100 + i)
        pair_work = work / f"floor_{i}"
        pair_work.mkdir(parents=True, exist_ok=True)
        row = run_pair_arm(
            cfg=cfg,
            arm="isolated",
            work=pair_work,
            bridge_port=bridge_port,
            world=world,
            bot=bot,
            settle=settle,
            keep_artifacts=False,
        )
        choices.append(str(row["first_contact"]["chosen"]))
    counts: dict[str, int] = {}
    for c in choices:
        counts[c] = counts.get(c, 0) + 1
    concentration = max(counts.values()) / len(choices)
    return {
        "choices": counts,
        "concentration": round(concentration, 4),
        "pass": concentration < FLOOR_CONCENTRATION_MAX and len(counts) > 1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/experiments/data/56_phase0.json")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=25580)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=25575)
    ap.add_argument("--rcon-password", default=os.environ.get("EXP56_RCON_PASSWORD", ""))
    ap.add_argument("--bot-name", default=os.environ.get("EXP56_BOT_NAME", "maxim_bench"))
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

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="exp56_phase0_"))
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {"ts": time.time(), "mock": bool(args.mock), "frozen": C.FROZEN}
    report.update(preflight)
    report["provenance"] = provenance
    try:
        report["check1_discriminability"] = check_discriminability(
            bridge_port, world, args.bot_name, work, settle=settle
        )
        report["check2_pilots"] = check_pilots(bridge_port, world, args.bot_name, work, settle=settle)
        report["check3_drive_zero"] = {
            "pass": True,
            "note": "asserted per probe decision inside the harness (raise on violation)",
        }
        report["check4_floor"] = check_floor(bridge_port, world, args.bot_name, work, settle=settle)
        report["check5_donor_at_K"] = {
            "pass": bool((report["check2_pilots"].get("donor_sanity") or {}).get("pass")),
            "K": C.SCHEDULE_K,
        }
    finally:
        world.close()
        if server is not None:
            server.close()
        shutil.rmtree(work, ignore_errors=True) if args.workdir is None else None

    all_pass = all(
        report[k]["pass"]
        for k in ("check1_discriminability", "check2_pilots", "check3_drive_zero", "check4_floor", "check5_donor_at_K")
    )
    report["all_pass"] = all_pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k].get("pass") for k in report if k.startswith("check")}, indent=2))
    print(f"phase0: {'PASS' if all_pass else 'FAIL'} -> {out_path}")
    return 0 if all_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
