#!/usr/bin/env python3
"""Exp 57 dose-response ladder runner (N in {1,2,4,8} x >=20 cohorts).

Pre-registration:
``docs/experiments/protocols/exp57_dose_response_ladder_preregistration.md``
(frozen; this runner implements it — ``scripts/analyze_exp57.py``'s constants
are the verdict authority). One JSONL row per (cohort, rung, condition,
checkpoint t) with full provenance.

For each cohort x rung:

* A-phase — trains the contributors against the (live or ``--mock``) bridge
  with per-trial ``(NAc, EC)`` snapshots: creche(N) taught contributors with
  INDEPENDENT seeds+orders; the single_matched(N) contributor trained to
  N*K_max; creche_none(N) contributors with the teacher WITHHELD.
* B-phase (offline) — at each per-agent checkpoint t folds the contributors'
  t-th snapshots into a fresh receiver through the REAL 1.2 ingest path
  (:func:`common57.fold_snapshots`, left-associative), derives the merged
  contingency clusters, reads DETERMINISTIC bias-decisive coverage from
  ``NAc_RECOMMEND`` provenance, and computes tau.

Modes
-----
* LIVE (default): Paper server + Mineflayer bridge + RCON world control.
  Confirmatory.
* ``--mock``: the deterministic ScriptedBridgeServer — wiring smoke ONLY,
  stamped ``mock: true`` (the analyzer refuses a verdict on mock rows).

``--seed-base`` drives EVERYTHING (cohort seeds, per-contributor seeds and
orders) — this covers R0's multi-seed requirement. ``--resume`` skips
(cohort, rung, condition) triples already present in the output.
"""

from __future__ import annotations

import argparse
import json
import os
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
from exp57 import common57 as X  # noqa: E402

REPO_ROOT = C.REPO_ROOT
CONDITIONS = ("creche", "single_matched", "creche_none")
BOT_NAME_ENV = "EXP57_BOT_NAME"


def _existing_rows(out_path: Path) -> set[tuple[int, int, str]]:
    done: set[tuple[int, int, str]] = set()
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if {"cohort", "rung", "condition"} <= rec.keys():
                done.add((int(rec["cohort"]), int(rec["rung"]), str(rec["condition"])))
    return done


def _close(session: "C.BenchSession", work: Path, tag: str) -> None:
    """Close a contributor session (frees the one-client bridge slot)."""
    C.close_and_stage_session(session, stage_dir=work / f"_close_{tag}")


def _train(
    *,
    seed: int,
    slot_to_target: dict[int, str],
    bridge_port: int,
    world,
    bot: str,
    home: Path,
    close_dir: Path,
    k_max: int,
    reps_per_cell: int,
    settle: float,
    teach: bool,
    tag: str,
) -> list:
    """Build one contributor, train with per-trial snapshots, close it."""
    import shutil

    shutil.rmtree(home, ignore_errors=True)
    session = C.build_bench_session(
        agent_id=f"aut_{tag}", bridge_port=bridge_port, home=home, pair_seed=seed, body_ref=X.BODY_REF57
    )
    snaps = X.train_contributor_with_snapshots(
        session,
        world=world,
        contributor_seed=seed,
        slot_to_target=slot_to_target,
        bot_name=bot,
        k_max=k_max,
        reps_per_cell=reps_per_cell,
        settle_s=settle,
        teach=teach,
    )
    _close(session, close_dir, tag)
    return snaps


def _checkpoint_rows(
    *,
    condition: str,
    cohort: int,
    rung: int,
    per_contributor_snaps: list[list],
    single: bool,
    slot_to_target: dict[int, str],
    k_max_index: int,
    criterion: float,
    window: int,
    workdir: Path,
) -> list[dict]:
    """B-phase: fold at each checkpoint, compute coverage + tau, emit rows."""
    recv_id = f"recv-{cohort}-{rung}-{condition}"
    contributor_ids = [f"c-{cohort}-{rung}-{condition}-{i}" for i in range(len(per_contributor_snaps))]

    cov_series: list[float] = []
    detail_series: list[dict[int, bool]] = []
    fold_reports: list[dict] = []
    for t in range(k_max_index):
        if single:
            snaps_t = [per_contributor_snaps[0][t]]
            cids = contributor_ids[:1]
        else:
            snaps_t = [snaps[t] for snaps in per_contributor_snaps]
            cids = contributor_ids
        fold_dir = workdir / f"fold_{condition}_{t}"
        merged = X.fold_snapshots(snaps_t, recv_id, workdir=fold_dir, contributor_ids=cids)
        merged_ec = json.loads((fold_dir / "recv" / "ec.json").read_text()).get("substrate_nodes", {})
        clusters = X.contingency_clusters_from_ec_nodes(merged_ec, receiver_agent_id=recv_id)
        cov, detail = X.coverage(merged, clusters, slot_to_target, receiver_agent_id=recv_id, return_detail=True)
        cov_series.append(cov)
        detail_series.append(detail)
        fold_reports.append(
            {
                "n_contributors": len(snaps_t),
                "merged_bias_keys": len(merged.get("cluster_reward_bias", {}) or {}),
                "covered_clusters": sum(1 for v in clusters.values() if v),
            }
        )

    tau_value = X.tau(cov_series, criterion, window, k_max_index)
    censored = X.is_censored(tau_value, k_max_index)
    rows: list[dict] = []
    for t in range(k_max_index):
        rows.append(
            {
                "cohort": cohort,
                "rung": rung,
                "condition": condition,
                "t": t + 1,
                "coverage": cov_series[t],
                "tau": tau_value,
                "tau_censored": censored,
                "criterion": criterion,
                "window": window,
                "k_max": k_max_index,
                "per_contingency": {str(g): bool(v) for g, v in detail_series[t].items()},
                "fold_report": fold_reports[t],
                "slot_to_target": {str(g): a for g, a in slot_to_target.items()},
            }
        )
    return rows


def run_cohort_rung(
    *,
    cohort: int,
    rung: int,
    seed_base: int,
    bridge_port: int,
    world,
    bot: str,
    work: Path,
    k_max: int,
    reps_per_cell: int,
    criterion: float,
    window: int,
    settle: float,
    conditions: list[str],
    artifacts_dir: "Path | None" = None,
) -> list[dict]:
    cohort_seed = seed_base + cohort
    slot_to_target = X.cohort_slot_to_target(cohort_seed)
    rows: list[dict] = []

    if "creche" in conditions:
        seeds = X.contributor_seeds(cohort_seed, rung, rung, salt=1)
        snaps = [
            _train(
                seed=s,
                slot_to_target=slot_to_target,
                bridge_port=bridge_port,
                world=world,
                bot=bot,
                home=work / f"creche_{i}_home",
                close_dir=work,
                k_max=k_max,
                reps_per_cell=reps_per_cell,
                settle=settle,
                teach=True,
                tag=f"creche_{cohort}_{rung}_{i}",
            )
            for i, s in enumerate(seeds)
        ]
        # Anti-vacuity artifacts: keep cohort-0's final-checkpoint creche
        # snapshots NEXT TO the campaign output, so `analyze_exp57.py
        # --assert-noop-fails` can re-run the D62 no-op kit on the MATCHING
        # fold (mirrors run_campaign's pair0_artifacts).
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            final = [list(s[-1]) for s in snaps]  # [(nac, ec), ...] at t = k_max
            (artifacts_dir / "snapshots.json").write_text(json.dumps(final))
            (artifacts_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "cohort": cohort,
                        "rung": rung,
                        "condition": "creche",
                        "receiver_agent_id": f"recv-{cohort}-{rung}-creche",
                        "contributor_ids": [f"c-{cohort}-{rung}-creche-{i}" for i in range(len(snaps))],
                        "slot_to_target": {str(g): a for g, a in slot_to_target.items()},
                    }
                )
            )
        rows += _checkpoint_rows(
            condition="creche",
            cohort=cohort,
            rung=rung,
            per_contributor_snaps=snaps,
            single=False,
            slot_to_target=slot_to_target,
            k_max_index=k_max,
            criterion=criterion,
            window=window,
            workdir=work / "creche_b",
        )

    if "single_matched" in conditions:
        # Prereg §Conditions: "creche(1) ≡ single_matched(1) ≡ one taught
        # contributor at its own budget." At rung 1 reuse creche's salt-1 seed so
        # the two arms ARE the identical agent (methodology-lens finding 9); for
        # N>=2 single_matched draws its own salt-2 stream (no collision).
        if rung == 1:
            sm_seed = X.contributor_seeds(cohort_seed, 1, 1, salt=1)[0]
        else:
            sm_seed = X.contributor_seeds(cohort_seed, rung, 1, salt=2)[0]
        sm_index = rung * k_max  # ONE agent given all N*K_max trials
        sm_snaps = _train(
            seed=sm_seed,
            slot_to_target=slot_to_target,
            bridge_port=bridge_port,
            world=world,
            bot=bot,
            home=work / "single_home",
            close_dir=work,
            k_max=sm_index,
            reps_per_cell=reps_per_cell,
            settle=settle,
            teach=True,
            tag=f"single_{cohort}_{rung}",
        )
        rows += _checkpoint_rows(
            condition="single_matched",
            cohort=cohort,
            rung=rung,
            per_contributor_snaps=[sm_snaps],
            single=True,
            slot_to_target=slot_to_target,
            k_max_index=sm_index,
            criterion=criterion,
            window=window,
            workdir=work / "single_b",
        )

    if "creche_none" in conditions:
        seeds = X.contributor_seeds(cohort_seed, rung, rung, salt=3)
        none_snaps = [
            _train(
                seed=s,
                slot_to_target=slot_to_target,
                bridge_port=bridge_port,
                world=world,
                bot=bot,
                home=work / f"none_{i}_home",
                close_dir=work,
                k_max=k_max,
                reps_per_cell=reps_per_cell,
                settle=settle,
                teach=False,  # teacher WITHHELD — the noise floor
                tag=f"none_{cohort}_{rung}_{i}",
            )
            for i, s in enumerate(seeds)
        ]
        rows += _checkpoint_rows(
            condition="creche_none",
            cohort=cohort,
            rung=rung,
            per_contributor_snaps=none_snaps,
            single=False,
            slot_to_target=slot_to_target,
            k_max_index=k_max,
            criterion=criterion,
            window=window,
            workdir=work / "none_b",
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rungs", default="1,2,4,8", help="comma-separated N values")
    ap.add_argument("--cohorts", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=None, help="Scratch root (default: a tmpdir; durable storage live)")
    # Apparatus constants set by the Phase-0 amendment (design targets below).
    ap.add_argument("--k-max", type=int, default=6, help="per-agent budget (Phase-0 sets the real value)")
    ap.add_argument("--reps-per-cell", type=int, default=1)
    ap.add_argument("--criterion", type=float, default=X.CRITERION_TARGET, help="C (target 3/4)")
    ap.add_argument("--window", type=int, default=X.WINDOW_TARGET, help="W (target 3)")
    # World transport.
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=25580)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=25575)
    ap.add_argument("--rcon-password", default=os.environ.get("EXP57_RCON_PASSWORD", ""))
    ap.add_argument("--bot-name", default=os.environ.get(BOT_NAME_ENV, "maxim_bench"))
    ap.add_argument("--settle-s", type=float, default=0.6)
    ap.add_argument("--mock", action="store_true", help="ScriptedBridgeServer wiring smoke — NEVER confirmatory")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="Disallowed for the confirmatory campaign (prereg)")
    args = ap.parse_args()

    if os.environ.get("MAXIM_OPERANT_ONLY_CREDIT") != "1":
        # Part of the frozen apparatus in EVERY condition; setting it here
        # would hide an ambient disagreement — refuse (the Exp 52/56 exit-3
        # shape).
        print("error: MAXIM_OPERANT_ONLY_CREDIT=1 is required (frozen apparatus). Export it and re-run.")
        return 3

    try:
        rungs = [int(r.strip()) for r in args.rungs.split(",") if r.strip()]
    except ValueError:
        print(f"error: --rungs must be comma-separated integers, got {args.rungs!r}")
        return 2
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        print(f"error: unknown conditions {sorted(unknown)}")
        return 2

    assert_repo_interpreter(REPO_ROOT, "maxim", exempt=args.mock)
    out_path = evidence_out_paths(
        REPO_ROOT, [args.out], write_experiment_results=args.write_experiment_results, allow_dirty=args.allow_dirty
    )[0]
    preflight = preflight_gated_record_or_exit(REPO_ROOT, out_path, allow_dirty=args.allow_dirty)
    provenance = executed_code_provenance(REPO_ROOT, "maxim", out_path=out_path, allow_dirty=args.allow_dirty)

    if args.resume and not args.workdir:
        print("error: --resume requires a durable --workdir (contributor homes must survive)")
        return 2
    if args.resume and str(out_path) != str(Path(args.out).resolve()) and not args.write_experiment_results:
        print(
            f"error: --resume with a redirected output ({out_path}); pass --write-experiment-results "
            "or a non-committed --out"
        )
        return 2
    done = _existing_rows(out_path) if args.resume else set()

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

    import shutil
    import tempfile

    work_root = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="exp57_"))
    work_root.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    try:
        with out_path.open("a", encoding="utf-8") as fh:
            for cohort in range(args.cohorts):
                for rung in rungs:
                    pending = [c for c in conditions if (cohort, rung, c) not in done]
                    if not pending:
                        continue
                    work = work_root / f"cohort_{cohort}_rung_{rung}"
                    work.mkdir(parents=True, exist_ok=True)
                    # Keep anti-vacuity artifacts from cohort 0's LARGEST rung
                    # (most contributors -> the fold's averaging is exercised).
                    artifacts_dir = (
                        out_path.parent / "cohort0_artifacts"
                        if cohort == 0 and rung == max(rungs) and "creche" in pending
                        else None
                    )
                    rows = run_cohort_rung(
                        cohort=cohort,
                        rung=rung,
                        seed_base=args.seed_base,
                        bridge_port=bridge_port,
                        world=world,
                        bot=args.bot_name,
                        work=work,
                        k_max=args.k_max,
                        reps_per_cell=args.reps_per_cell,
                        criterion=args.criterion,
                        window=args.window,
                        settle=settle,
                        conditions=pending,
                        artifacts_dir=artifacts_dir,
                    )
                    for row in rows:
                        row["mock"] = bool(args.mock)
                        row["seed_base"] = args.seed_base
                        row["ts"] = time.time()
                        row.update(preflight)
                        row["provenance"] = provenance
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        wrote += 1
                    # A per-checkpoint condition summary line (tau is constant
                    # across a condition's rows).
                    for c in pending:
                        crows = [r for r in rows if r["condition"] == c]
                        if crows:
                            print(
                                f"cohort {cohort} rung {rung} {c}: tau={crows[0]['tau']} "
                                f"censored={crows[0]['tau_censored']} endpoint_cov={crows[-1]['coverage']:.3f}"
                            )
                    shutil.rmtree(work, ignore_errors=True) if args.workdir is None else None
    finally:
        world.close()
        if server is not None:
            server.close()
    print(f"wrote {wrote} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
