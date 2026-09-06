#!/usr/bin/env python3
"""Exp 56 confirmatory campaign runner (four arms × N pairs).

Pre-registration: docs/experiments/protocols/exp56_four_arm_sharing_preregistration.md
(frozen; this runner implements it — the analyzer's constants are the
verdict authority). One JSONL row per (pair, arm) with full provenance.

Modes
-----
* LIVE (default): real Paper server + Mineflayer bridge (`--bridge-port`)
  + RCON world control (`--rcon-port`/`--rcon-password`). Confirmatory.
* ``--mock``: the deterministic ScriptedBridgeServer — wiring smoke ONLY,
  never a confirmatory record (stamped ``mock: true`` on every row; the
  analyzer refuses a verdict on mock data).

``--resume`` skips (pair, arm) rows already present in the output file.
Committed-evidence writes route through ``evidence_out_paths``
(``--write-experiment-results`` required to touch ``docs/experiments/``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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

REPO_ROOT = C.REPO_ROOT
ARMS = ("isolated", "taught", "satiated", "dangling")
BOT_NAME_ENV = "EXP56_BOT_NAME"


def _existing_rows(out_path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "pair_seed" in rec and "arm" in rec:
                done.add((int(rec["pair_seed"]), str(rec["arm"])))
    return done


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _train_donor(work: Path, cfg: dict, *, body_ref: str, arm: str, bridge_port: int, world, bot: str, settle: float):
    """Train one donor, sanity-check it, stage its session pair, and write
    the ``donor_meta.json`` sidecar (bias count + every attempt's sanity
    readings — review folds ex-8/I4: the failing attempt's readings are
    kept, and the shipped-count is durably beside the stage for arm 4's
    dropped==shipped assertion).

    A donor failing sanity is an APPARATUS failure: re-run once on a
    shifted schedule seed (target/slot/translation stay the pair's — the
    retry is a fresh sample of the SAME design point), recorded; a second
    failure raises (S3 refusal)."""
    attempts: list[dict] = []
    for attempt, seed_shift in enumerate((0, 100_003)):
        seed = cfg["pair_seed"] + seed_shift
        home = work / f"{arm}_donor_home"
        shutil.rmtree(home, ignore_errors=True)
        session = C.build_bench_session(
            agent_id=f"donor_{arm}_{cfg['pair_seed']}",
            bridge_port=bridge_port,
            home=home,
            pair_seed=cfg["pair_seed"],  # the TRANSLATION stays the pair's
            body_ref=body_ref,
        )
        telemetry = C.run_donor_training(
            session,
            world=world,
            pair_seed=seed,  # the schedule shuffle may shift on retry
            target_aff=cfg["target_aff"],
            arm=arm,
            slot=cfg["slot"],
            bot_name=bot,
            settle_s=settle,
        )
        sanity = C.donor_sanity(session, arm=arm)
        sanity["attempt"] = attempt
        attempts.append(dict(sanity))
        stage = C.close_and_stage_session(session, stage_dir=work / f"{arm}_donor_stage")
        if sanity["pass"]:
            meta = {
                "pair_seed": cfg["pair_seed"],
                "arm": arm,
                "bias_entries": sanity["total_cluster_bias_entries"],
                "sanity_attempts": attempts,
                "nac_sha256": _sha256(stage / "aut_nac.json"),
            }
            (stage / "donor_meta.json").write_text(json.dumps(meta, indent=2))
            return stage, sanity, telemetry
    raise RuntimeError(f"exp56 pair {cfg['pair_seed']} {arm} donor failed sanity twice: {attempts}")


def run_pair_arm(
    *,
    cfg: dict,
    arm: str,
    work: Path,
    bridge_port: int,
    world,
    bot: str,
    settle: float,
    keep_artifacts: bool,
    artifacts_dir: "Path | None" = None,
) -> dict:
    """One (pair, arm) measurement. Returns the JSONL row."""
    pair_seed = cfg["pair_seed"]
    row: dict = {"pair_seed": pair_seed, "arm": arm, "target_aff": cfg["target_aff"], "slot": cfg["slot"]}
    if keep_artifacts and artifacts_dir is not None:
        row["_artifacts_dir"] = artifacts_dir

    bundle: Path | None = None
    if arm in ("taught", "dangling"):
        stage = work / "taught_donor_stage"
        if arm == "dangling" and not (stage / "donor_meta.json").is_file():
            # The frozen arm-4 invariant: "the SAME taught donors as arm 2".
            # A dangling run whose pair workdir lacks the taught stage
            # (fresh tmpdir under --resume, or --arms dangling alone) would
            # silently train a NEW donor — refuse instead (review C1/ex-1).
            raise RuntimeError(
                f"exp56 pair {pair_seed}: dangling requires this pair's taught donor stage "
                f"({stage}) — run the taught arm in the same durable --workdir first"
            )
        if not (stage / "aut_nac.json").is_file():
            stage, sanity, telemetry = _train_donor(
                work,
                cfg,
                body_ref=C.BODY_REF,
                arm="taught",
                bridge_port=bridge_port,
                world=world,
                bot=bot,
                settle=settle,
            )
            row["donor_sanity"] = sanity
            row["teacher_feeds"] = sum(1 for t in telemetry if t["fed"])
            row["teacher_credits"] = sum(1 for t in telemetry if t["credited"])
            row["teacher_fed_trials"] = [t for t in telemetry if t["fed"]]  # the Exp 52 audit surface (N5)
        bundle = work / ("dangling.zip" if arm == "dangling" else "taught.zip")
        C.export_bundle(stage, bundle, contributor_id=cfg["contributor_id"], dangling=(arm == "dangling"))
    elif arm == "satiated":
        stage, sanity, telemetry = _train_donor(
            work,
            cfg,
            body_ref=C.BODY_REF_SATIATED,
            arm="satiated",
            bridge_port=bridge_port,
            world=world,
            bot=bot,
            settle=settle,
        )
        row["donor_sanity"] = sanity
        row["teacher_feeds"] = sum(1 for t in telemetry if t["fed"])
        row["teacher_credits"] = sum(1 for t in telemetry if t["credited"])
        row["teacher_fed_trials"] = [t for t in telemetry if t["fed"]]  # the Exp 52 audit surface (N5)
        bundle = work / "satiated.zip"
        C.export_bundle(stage, bundle, contributor_id=cfg["contributor_id"])

    # Receiver: fresh, closed at rest, (maybe) ingested, reopened, probed.
    recv_home = work / f"{arm}_recv_home"
    shutil.rmtree(recv_home, ignore_errors=True)
    recv_id = f"recv_{arm}_{pair_seed}"
    recv = C.build_bench_session(agent_id=recv_id, bridge_port=bridge_port, home=recv_home, pair_seed=pair_seed)
    C.close_and_stage_session(recv, stage_dir=work / f"{arm}_recv_stage")
    if bundle is not None:
        # Auditability (review ex-1): every merged arm's row carries the
        # exact bundle + donor-nac identity, so a donor-identity violation
        # is visible in the record, not just refused in the flow.
        donor_meta = json.loads((stage / "donor_meta.json").read_text())
        row["bundle_sha256"] = _sha256(bundle)
        row["donor_nac_sha256"] = donor_meta["nac_sha256"]
        row["donor_meta"] = {k: donor_meta[k] for k in ("bias_entries", "sanity_attempts")}
        # S3 independence, pre-ingest (review I5): the receiver's EC must
        # hold zero donor cluster ids (D44's disjoint-by-construction).
        donor_ec_ids = (
            set(json.loads((stage / "aut_ec.json").read_text()).get("substrate_nodes", {}).keys())
            if (stage / "aut_ec.json").is_file()
            else set()
        )
        recv_ec_ids = set(
            json.loads((work / f"{arm}_recv_stage" / "aut_ec.json").read_text()).get("substrate_nodes", {}).keys()
        )
        if donor_ec_ids & recv_ec_ids:
            raise RuntimeError(
                f"exp56 pair {pair_seed} {arm}: receiver pre-ingest EC shares "
                f"{len(donor_ec_ids & recv_ec_ids)} cluster ids with the donor — independence violated (S3)"
            )
        entry = C.ingest_bundle_into(recv_home, bundle, contributor_id=cfg["contributor_id"], receiver_agent_id=recv_id)
        row["ingest"] = {
            k: entry.get(k) for k in ("biases_rekeyed", "biases_dropped", "biases_tightened", "inherent_keys_admitted")
        }
        if arm == "dangling":
            # The honest indicator, BOTH halves (review I4/ex-5): all
            # shipped biases dropped, none landed. A missing key is a
            # violation, never a silent pass.
            if int(entry.get("biases_rekeyed", -1)) != 0 or int(entry.get("biases_dropped", -1)) != int(
                donor_meta["bias_entries"]
            ):
                raise RuntimeError(
                    f"exp56 pair {pair_seed}: dangling ingest rekeyed={entry.get('biases_rekeyed')} "
                    f"dropped={entry.get('biases_dropped')} vs shipped={donor_meta['bias_entries']} — "
                    "the apparatus is not testing what it claims (S3)"
                )
        elif arm == "taught":
            if int(entry.get("biases_rekeyed", 0)) < 1:
                raise RuntimeError(
                    f"exp56 pair {pair_seed}: taught ingest landed no biases — the D43 silent-zero "
                    "shape; apparatus failure, not a floor-rate data point (S3)"
                )
            # S3 independence, post-ingest: every landed cluster-bias key
            # carries the RECEIVER's agent id (the boundary normalization
            # under test).
            post = json.loads((recv_home / "nac.json").read_text())
            for key in post.get("cluster_reward_bias") or {}:
                aid = str(key).split("\x1f")[0]
                if aid != recv_id:
                    raise RuntimeError(
                        f"exp56 pair {pair_seed}: post-ingest bias key carries agent id {aid!r}, "
                        f"not the receiver's {recv_id!r} — boundary normalization failed (S3)"
                    )
    recv2 = C.build_bench_session(agent_id=recv_id, bridge_port=bridge_port, home=recv_home, pair_seed=pair_seed)
    probe = C.probe_receiver(recv2, world=world, pair_seed=pair_seed, slot=cfg["slot"], bot_name=bot, settle_s=settle)
    C.close_and_stage_session(recv2, stage_dir=work / f"{arm}_recv_post")

    fc = probe["first_contact"]
    target_tool = f"{C.ENTITY_NAME}_{cfg['target_aff']}"
    chose_target = fc["chosen"] == target_tool
    latency = None
    for d in probe["decisions"]:
        if d["situation_active"] and d["chosen"] == target_tool:
            latency = d["decision"]  # decisions from probe start (frozen secondary)
            break
    # The per-pair opaque-name permutation, recorded in provenance (N4).
    row["action_map"] = {aff: [name, params] for aff, (name, params) in recv2.aut.client.action_map.items()}
    row.update(
        {
            "first_contact": fc,
            "contact_at": probe["contact_at"],
            "chose_target": chose_target,
            "bias_decisive": C.bias_decisive(fc, fc["chosen"]),
            "approach_latency": latency,
            "n_decisions": len(probe["decisions"]),
            "ts": time.time(),
        }
    )
    if keep_artifacts and arm == "taught" and "_artifacts_dir" in row:
        # The anti-vacuity kit's inputs live NEXT TO the campaign output
        # (review I6/ex-6: a tmpdir workdir made the runbook's analyze
        # invocation a FileNotFoundError), stamped with their pair so the
        # analyzer re-runs the MATCHING row.
        keep = row.pop("_artifacts_dir")
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle, keep / "taught.zip")
        for name in ("nac.json", "ec.json"):
            src = work / f"{arm}_recv_stage" / f"aut_{name}"
            if src.is_file():
                shutil.copyfile(src, keep / f"receiver_pre_{name}")
        (keep / "meta.json").write_text(json.dumps({"pair_seed": pair_seed, "receiver_agent_id": recv_id}))
    else:
        row.pop("_artifacts_dir", None)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=None, help="Scratch root (default: a tmpdir; use durable storage live)")
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=25580)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=25575)
    ap.add_argument("--rcon-password", default=os.environ.get("EXP56_RCON_PASSWORD", ""))
    ap.add_argument("--bot-name", default=os.environ.get(BOT_NAME_ENV, "maxim_bench"))
    ap.add_argument("--settle-s", type=float, default=0.6)
    ap.add_argument("--mock", action="store_true", help="ScriptedBridgeServer wiring smoke — NEVER confirmatory")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="Disallowed for the confirmatory campaign (prereg)")
    args = ap.parse_args()

    if os.environ.get("MAXIM_OPERANT_ONLY_CREDIT") != "1":
        # The flag is part of the frozen apparatus in EVERY arm and phase;
        # setting it here (rather than requiring the operator to) would hide
        # an ambient disagreement — refuse instead (the Exp 52 exit-3 shape).
        print("error: MAXIM_OPERANT_ONLY_CREDIT=1 is required (frozen apparatus). Export it and re-run.")
        return 3

    assert_repo_interpreter(REPO_ROOT, "maxim", exempt=args.mock)
    out_path = evidence_out_paths(
        REPO_ROOT, [args.out], write_experiment_results=args.write_experiment_results, allow_dirty=args.allow_dirty
    )[0]
    preflight = preflight_gated_record_or_exit(REPO_ROOT, out_path, allow_dirty=args.allow_dirty)
    provenance = executed_code_provenance(REPO_ROOT, "maxim", out_path=out_path, allow_dirty=args.allow_dirty)

    if args.mock:
        server = C.ScriptedBridgeServer(seed=args.seed_base)
        world = C.ScriptedWorldControl(server, settle_s=min(args.settle_s, 0.08))
        bridge_port = server.port
        settle = 0.02
    else:
        server = None
        world = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
        bridge_port = args.bridge_port
        settle = args.settle_s

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        print(f"error: unknown arms {sorted(unknown)}")
        return 2
    if args.resume and not args.workdir:
        # A fresh tmpdir under --resume silently orphans the taught donor
        # stages arm 4 must reuse (review ex-1) — refuse.
        print("error: --resume requires a durable --workdir (the pair stages must survive)")
        return 2
    if args.resume and str(out_path) != str(Path(args.out).resolve()) and not args.write_experiment_results:
        # evidence_out_paths redirected --out (committed path without the
        # flag): a resume against the redirect sees zero prior rows and
        # silently restarts (review ex-10). Refuse the ambiguity.
        print(
            f"error: --resume with a redirected output ({out_path}); pass --write-experiment-results or a non-committed --out"
        )
        return 2
    done = _existing_rows(out_path) if args.resume else set()

    import tempfile

    work_root = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="exp56_"))
    work_root.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    try:
        with out_path.open("a", encoding="utf-8") as fh:
            for i in range(args.pairs):
                pair_seed = args.seed_base + i
                cfg = C.pair_config(pair_seed)
                pair_work = work_root / f"pair_{pair_seed}"
                pair_work.mkdir(parents=True, exist_ok=True)
                for arm in arms:
                    if (pair_seed, arm) in done:
                        continue
                    row = run_pair_arm(
                        cfg=cfg,
                        arm=arm,
                        work=pair_work,
                        bridge_port=bridge_port,
                        world=world,
                        bot=args.bot_name,
                        settle=settle,
                        keep_artifacts=(i == 0),
                        artifacts_dir=out_path.parent / "pair0_artifacts",
                    )
                    row["mock"] = bool(args.mock)
                    row.update(preflight)
                    row["provenance"] = provenance
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    wrote += 1
                    print(
                        f"pair {pair_seed} {arm}: chose={row['first_contact']['chosen']} "
                        f"target={row['target_aff']} decisive={row['bias_decisive']}"
                    )
    finally:
        world.close()
        if server is not None:
            server.close()
    print(f"wrote {wrote} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
