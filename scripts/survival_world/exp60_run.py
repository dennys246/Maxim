#!/usr/bin/env python3
"""Exp 60 LIVE trial harness — learned drowning-avoidance (FEAR vs ABLATED arms).

Chunk (iii) of the Exp 60 build (docs/experiments/exp60_drowning_avoidance_prereg.md
§Design (iii)). Runs the prereg's design against the live water classroom (built by
``setup_world.py water_classroom``, MEASURED by ``exp60_water_check.py``, separability
gated by ``l11_geometry_probe.py`` run_gate). Modelled on ``exp58_run.py`` (which ran live)
with the changes the four design lenses required. Per seed × arm:

  1. PREFLIGHTS (all before any measurement; refusal = exit 3/4, no behavioural data):
     - both gated records on main PASS (apparatus check ``all_pass``; geometry gate
       ``run_gate.pass``) and the anchor carries the check's stamped ``measured`` onset;
     - frozen-apparatus fingerprint asserted + stamped — fear α/cap/θ/allowlist (WITH
       ``drive:oxygen``), encoder threshold, explore weight, the OXYGEN drive spec, and the
       declared ranges of ``is_in_water``/``oxygen``/``saturation`` (they decide separability);
     - the RAW bridge roster emits every gated sensor; frozen gamerules verified, not toggled;
     - the LIVE cluster-distinct preflight on the live agent's EC: shore and submerged encode
       to different world clusters (gate (ii) is necessary-not-sufficient; this is its partner);
     - ESCAPE ACTUATION CHECK: one real executor ``escape_water`` from the pool floor puts the
       head in air within 6 s by BRIDGE TRUTH (``is_in_water`` 0), or the seed refuses.
  2. PRE-PROBE: P placements on the pool FLOOR with the FULL loop live — WARMED on the shore
     before each teleport (loop boot is not in the window) and stopped after it (the bot cannot
     wander into the pool between placements). DV per placement = latency from the TELEPORT to
     the first ``is_in_water`` 0 read; CENSORED at the cap = **measured air-hunger pain edge
     (min) − margin ≈ 4.3 s**, rescue teleport FIRST then loop stop. A probe is a TEST TRIAL and
     must be US-FREE: the unconditioned stimulus is the ``drive:oxygen`` pain (~5 s in, measured),
     not drowning damage (~16 s) — a longer window would CONDITION the FEAR arm during the
     pre-probe and un-yoke exposure (both review lenses). Belt: a bus subscriber runs for the
     whole seed; any pain publish inside a probe window marks the placement DIRTY and is counted
     as unyoked exposure, and the pre-probe must leave ZERO fear on the water cluster (both arms)
     or the seed refuses. A health drop marks DIRTY too. Then a shore free-roam window (activity
     control; water entry recorded). The read path's FIRST fear-driven pick is ``flee`` (name
     tie-break), which the bridge now fails FAST when submerged — ``flee`` calls per placement
     are recorded and disclosed.
  3. TRAINING: K yoked, harness-scheduled, PROPOSE-ONLY episodes on the pool floor (no execution:
     a full loop would book arm-asymmetric negative credit on the escape read path). Usable
     episode = a ``drive:oxygen`` PainSignal PUBLISHED (recorded by a bus subscriber) while the
     underwater world cluster is noted, the head is in water, sensed oxygen ≤ 12 and the
     signal's intensity is 1.0 (the saturating publish — a 0.5-intensity write converges fear to
     exactly −θ, which the loop's strict activation floor treats as DEAD; architecture lens) —
     ~6 s in, BEFORE tissue damage; cap = measured damage onset − 1 s; ANY health drop on
     arrival at the shore refuses the seed. Latch clears on OBSERVED recovery.
  4. LIVE G2 (readability): the PRODUCTION read ``anticipatory_threat_need`` clears the loop's
     activation floor (> 0.5) on the probe-activated underwater cluster AND on every distinct
     training-episode cluster (a jitter-split minority id must not be dead at recall); the
     ABLATED arm must carry ZERO fear (ablation verified). Specificity (|shore fear| <
     0.2·|water fear|) is a RECORDED mechanism DV gated in the verdict (a Wire-4 property, not an
     instrument fault). ``escape_water`` / ``flee`` negative-link counts recorded.
  5. POST-PROBE identical to (2). Per-seed record appended (run_id-stamped) through the gated
     evidence path; a refused seed is stamped REFUSED with no behavioural DVs.

``verdict`` (pure, offline, unit-tested) turns the per-seed JSONL into the prereg's gate
decision: seed = unit; primary DV = post-training P(surface before the US) in the US-free
window; exact two-sample permutation test FEAR vs ABLATED on post P(surface) + the per-seed
gates (incl. specificity). Duplicate (arm, seed) rows REFUSE unless ``--run-id`` selects a run.
Proposal cadence inside the loop is 2 Hz (``llm_submit_interval`` 0.5 s), stated so latencies
are read against it.

Run ON the bridge box (server + bridge from current main, classroom built, records merged).
The bridge MUST run at a 100 ms state cadence for sensor FRESHNESS (the DV clock reads is_in_water
from the latest snapshot at 4 Hz, so the snapshot interval must not exceed the 0.25 s sampling
period); the preflight measures the cadence and refuses a slow bridge, and a LOOP LIVENESS
preflight refuses a loop that does not reach its substrate branch >= 4 times in 3 s on the shore:

    (cd scripts/minecraft_bridge && node index.js --mc_host=127.0.0.1 --mc_port=25565 \\
        --bridge_port=25567 --username=maxim --state_interval_ms=100)
    export PYTHONPATH="$PWD/src"
    python scripts/survival_world/exp60_run.py run --arm fear --rcon-password '<pw>' \\
        --username maxim --write-experiment-results
    python scripts/survival_world/exp60_run.py run --arm ablated ...   # same, other arm
    python scripts/survival_world/exp60_run.py verdict --data docs/experiments/data/exp60_trials.jsonl \\
        --json docs/experiments/data/exp60_verdict.json --write-experiment-results
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import tempfile
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
from survival_world.water_trial import (  # noqa: E402  (re-exported: the Exp 60 unit tests pin these names)
    Refusal,
    WaterTrial,
    _detach_fear_subscriber,
    _f,
    _median,
    _telemetry_ticks,
    classify_placement,
    fingerprint_drift,
    median_interval_s,
    min_pain_edge_s,
    p_surface,
)

__all__ = [
    "Refusal",
    "_detach_fear_subscriber",
    "_f",
    "_median",
    "_telemetry_ticks",
    "classify_placement",
    "fingerprint_drift",
    "median_interval_s",
    "min_pain_edge_s",
    "p_surface",
]

ANCHOR_FILE = Path.home() / ".maxim" / "exp60_water_classroom.json"
APPARATUS_RECORD = "docs/experiments/data/exp60_water_apparatus.json"
GATE_RECORD = "docs/experiments/data/exp60_geometry_2026-09-15b.json"

FROZEN: dict[str, Any] = {
    "K_usable_episodes": 10,
    "placements_per_probe": 6,
    "shore_roam_s": 10.0,  # activity-control window on the shore, full loop live
    "probe_cap_margin_s": 0.75,  # probe cap = measured air-hunger pain edge (min) − this (US-FREE window)
    "train_cap_margin_s": 1.0,  # training cap = measured damage onset (min) − this (pre-damage)
    "loop_warm_s": 1.0,  # the full loop runs on the shore this long before a placement teleport
    # Sensor FRESHNESS for a 4 Hz-sampled 4.3 s window: the DV clock reads is_in_water from the
    # latest snapshot, so a 500 ms bridge adds up to 0.5 s to every latency. The bridge runs at
    # --state_interval_ms=100 and the harness MEASURES it at preflight (refuses a slow bridge).
    # (The "loop ticks ~5 snapshots apart" reading that first motivated this was the FAKE
    # bridge's every-5th-state event waking the loop; the live loop was idling for want of any
    # event at all — fixed in the loop itself, agent_loop `_substrate_tick_due`, 2026-09-16.)
    "bridge_state_interval_max_s": 0.15,
    # Loop LIVENESS preflight: the full loop on the shore must reach its substrate branch this many
    # times in this window (2 Hz proposal cadence → ~6 expected) or the seed refuses BEFORE any
    # placement — run 1 read one tick per window (agent_loop idle-gate defect, Amendment 5/6).
    "loop_liveness_min_ticks": 4,
    "loop_liveness_s": 3.0,
    "usable_oxygen_max": 12.0,  # the SATURATING publish (intensity 1.0); must sit BELOW set_point − comfort_band
    "usable_pain_intensity_min": 1.0,
    "seeds": (11, 12, 13, 14, 15),
    "loop_hz": 4.0,
    "death_cap": 2,
    "specificity_ratio": 0.2,  # |shore fear| < ratio · |water fear|
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
}
# Verdict gates (the prereg's, frozen with it):
GATES: dict[str, Any] = {
    "seeds_per_arm": 5,
    "fear_post_p_surface_min": 0.5,  # per-seed MEDIAN across FEAR seeds
    "ablated_post_p_surface_max": 0.2,  # per-seed MEDIAN across ABLATED seeds
    "fear_post_minus_pre_min": 0.0,  # FEAR post must exceed its own pre (strictly)
    "permutation_alpha": 0.05,  # exact two-sample, one-sided FEAR > ABLATED on post P(surface)
    "specificity_ratio": 0.2,  # mechanism DV: |shore fear| < ratio · |water fear| on every FEAR seed
}


# ─────────────────────────── pure helpers (unit-tested) ───────────────────────────


def exact_permutation_p(a: list[float], b: list[float]) -> dict[str, Any]:
    """Exact two-sample permutation test on the difference of means, one-sided a > b (pure).

    Enumerates every relabelling (C(n, |a|) — 252 at 5 v 5); p = fraction of relabellings
    whose mean difference is ≥ the observed one (the observed labelling counts, so the
    minimum attainable p at 5 v 5 is 1/252).
    """
    pooled = list(a) + list(b)
    n_a = len(a)
    obs = (sum(a) / n_a - sum(b) / len(b)) if a and b else 0.0
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), n_a):
        ga = [pooled[i] for i in idx]
        gb = [pooled[i] for i in range(len(pooled)) if i not in idx]
        diff = sum(ga) / len(ga) - sum(gb) / len(gb)
        total += 1
        if diff >= obs - 1e-12:
            count += 1
    return {"observed_diff": round(obs, 4), "p_one_sided": count / total if total else None, "relabellings": total}


def select_run(
    records: list[dict[str, Any]], run_id: "list[str] | str | None"
) -> tuple[list[dict[str, Any]], list[str]]:
    """Filter to one run and name duplicate (arm, seed) rows (pure).

    A re-run appended to the same JSONL must never double n: without ``run_id`` any
    duplicate (arm, seed) is a refusal reason; with it, only that run's rows count.
    """
    # Each `run` invocation mints its own run_id, so a two-arm trial is TWO ids — one per arm.
    wanted = set([run_id] if isinstance(run_id, str) else (run_id or []))
    rows = [r for r in records if not wanted or r.get("run_id") in wanted]
    seen: dict[tuple[str, Any], int] = {}
    for r in rows:
        seen[(r.get("arm"), r.get("seed"))] = seen.get((r.get("arm"), r.get("seed")), 0) + 1
    dups = sorted(f"{arm}/seed{seed}×{n}" for (arm, seed), n in seen.items() if n > 1)
    return rows, dups


def compute_verdict(
    records: list[dict[str, Any]], *, gates: dict[str, Any] = GATES, run_id: "list[str] | str | None" = None
) -> dict[str, Any]:
    """The prereg's gate decision from per-seed records (pure).

    ``verdict`` ∈ {"EARNED", "NULL", "INCOMPLETE"}: INCOMPLETE when either arm has fewer
    clean seeds than the prereg requires or duplicate (arm, seed) rows are present
    (refusals are named, never silently dropped); NULL when the arms have the seeds but a
    gate fails (a null ships as a null).
    """
    records, dups = select_run(records, run_id)
    if dups:
        return {
            "_format_version": "1.0",
            "kind": "exp60_verdict",
            "verdict": "INCOMPLETE",
            "reason": f"duplicate (arm, seed) rows: {dups} — pass --run-id ONCE PER ARM (each arm invocation mints its own id)",
            "duplicates": dups,
        }
    clean = [r for r in records if r.get("refusal") is None and "post" in r and "pre" in r]
    refused = [
        {"arm": r.get("arm"), "seed": r.get("seed"), "refusal": r.get("refusal")} for r in records if r.get("refusal")
    ]
    by_arm: dict[str, list[dict[str, Any]]] = {"fear": [], "ablated": []}
    for r in clean:
        by_arm.setdefault(r["arm"], []).append(r)
    per_seed = {
        arm: [
            {
                "seed": r["seed"],
                "pre_p_surface": p_surface(r["pre"]),
                "post_p_surface": p_surface(r["post"]),
                "water_fear": r.get("water_fear"),
                "shore_fear": r.get("shore_fear"),
            }
            for r in rows
        ]
        for arm, rows in by_arm.items()
    }
    need = gates["seeds_per_arm"]
    out: dict[str, Any] = {
        "_format_version": "1.0",
        "kind": "exp60_verdict",
        "gates": gates,
        "per_seed": per_seed,
        "refused": refused,
        "n_clean": {arm: len(rows) for arm, rows in by_arm.items()},
    }
    if any(len(by_arm.get(arm, [])) != need for arm in ("fear", "ablated")):
        out["verdict"] = "INCOMPLETE"
        out["reason"] = f"expected exactly {need} clean seeds per arm: {out['n_clean']} (refusals: {len(refused)})"
        return out
    fear_post = [s["post_p_surface"] for s in per_seed["fear"] if s["post_p_surface"] is not None]
    fear_pre = [s["pre_p_surface"] for s in per_seed["fear"] if s["pre_p_surface"] is not None]
    abl_post = [s["post_p_surface"] for s in per_seed["ablated"] if s["post_p_surface"] is not None]
    if len(fear_post) < need or len(abl_post) < need or len(fear_pre) < need:
        out["verdict"] = "INCOMPLETE"
        out["reason"] = "a seed has no clean placements in a probe (all dirty/never-submerged)"
        return out
    perm = exact_permutation_p(fear_post, abl_post)
    checks = {
        "fear_post_median_ge_min": _median(fear_post) >= gates["fear_post_p_surface_min"],
        "ablated_post_median_le_max": _median(abl_post) <= gates["ablated_post_p_surface_max"],
        "fear_post_gt_pre_every_seed": all(
            (s["post_p_surface"] or 0.0) - (s["pre_p_surface"] or 0.0) > gates["fear_post_minus_pre_min"]
            for s in per_seed["fear"]
        ),
        "permutation_p_lt_alpha": perm["p_one_sided"] is not None and perm["p_one_sided"] < gates["permutation_alpha"],
        # mechanism DV (confounding SF-2): fear must sit on the WATER cluster, not bleed to the shore
        "fear_specificity_every_seed": all(
            s["water_fear"] is not None
            and s["shore_fear"] is not None
            and abs(s["shore_fear"]) < gates["specificity_ratio"] * abs(s["water_fear"])
            for s in per_seed["fear"]
        ),
    }
    out.update(
        {
            "fear_post_median": _median(fear_post),
            "fear_pre_median": _median(fear_pre),
            "ablated_post_median": _median(abl_post),
            "permutation": perm,
            "checks": checks,
            "verdict": "EARNED" if all(checks.values()) else "NULL",
        }
    )
    return out


# ─────────────────────────────────── live ───────────────────────────────────


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

    # ── Preflight: BOTH gated records PASS (prereg instrument-gate stop rule) ──
    def _load(rel: str) -> dict[str, Any]:
        p = C.REPO_ROOT / rel
        try:
            return json.loads(p.read_text())
        except OSError:
            print(f"[FAIL] gated record missing: {p} — it must be on main before live trials")
            raise SystemExit(3)

    apparatus = _load(APPARATUS_RECORD)
    gate = _load(args.gate_record)
    if not apparatus.get("all_pass"):
        print(f"[FAIL] apparatus record does not carry all_pass=true: {APPARATUS_RECORD}")
        return 3
    if not (gate.get("run_gate") or {}).get("pass"):
        print(f"[FAIL] geometry gate record does not carry run_gate.pass=true: {args.gate_record}")
        return 3
    pain_edge_min = min_pain_edge_s(apparatus)
    try:
        geom = json.loads(ANCHOR_FILE.read_text())
        onset_min = float(geom["measured"]["t_damage_onset_min_s"])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(
            f"[FAIL] anchor {ANCHOR_FILE} lacks the check's stamped measured onset ({exc}) — run exp60_water_check first"
        )
        return 3
    if pain_edge_min is None:
        print(
            f"[FAIL] apparatus record carries no measured t_pain_edge — the probe cap cannot be set: {APPARATUS_RECORD}"
        )
        return 3
    probe_cap_s = pain_edge_min - FROZEN["probe_cap_margin_s"]  # US-FREE window
    train_cap_s = onset_min - FROZEN["train_cap_margin_s"]  # pre-damage conditioning

    from maxim.simulation.minecraft import MinecraftClient
    from maxim.simulation.minecraft_harness import MinecraftSyncPump, build_minecraft_aut
    from survival_world.common import make_fresh_encoder

    run_id = uuid.uuid4().hex[:12]
    rcon = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    print(
        f"run {run_id}: arm={args.arm} shore={geom['shore']} submerged={geom['submerged']} probe cap={probe_cap_s:.2f}s "
        f"(pain edge min {pain_edge_min:.2f}) train cap={train_cap_s:.2f}s (onset min {onset_min:.2f})"
    )

    records: list[dict[str, Any]] = []
    exit_code = 0
    try:
        for seed in FROZEN["seeds"][: args.seeds]:
            print(f"\n=== arm={args.arm} seed={seed} ===")
            persistence_dir = tempfile.mkdtemp(prefix=f"exp60_{args.arm}_{seed}_")
            agent_id = f"exp60_{args.arm}_s{seed}"
            client = MinecraftClient(args.bridge_host, args.bridge_port)
            client.connect(confirm_timeout_s=4.0, retries=8, backoff_s=0.5)
            aut = build_minecraft_aut(
                agent_id=agent_id,
                bridge_port=args.bridge_port,
                bridge_host=args.bridge_host,
                persistence_dir=persistence_dir,
                entity_ref="bodies/minecraft_player",
                client=client,
            )
            encoder = make_fresh_encoder(aut)
            pump = MinecraftSyncPump(aut, interval_s=0.25)
            pump.start()
            prev_cwd = os.getcwd()
            os.chdir(persistence_dir)
            record: dict[str, Any] = {
                "ts": time.time(),
                "run_id": run_id,
                "arm": args.arm,
                "seed": seed,
                "frozen": FROZEN,
                "probe_cap_s": probe_cap_s,
                "train_cap_s": train_cap_s,
                "pain_edge_min_s": pain_edge_min,
                "provenance": provenance,
                "apparatus_record_ts": apparatus.get("ts"),
                "gate_record_code_hash": gate.get("code_hash"),
                "refusal": None,
            }
            trial = WaterTrial(
                aut=aut,
                rcon=rcon,
                username=args.username,
                geom=geom,
                frozen=FROZEN,
                probe_cap_s=probe_cap_s,
                train_cap_s=train_cap_s,
                persistence_dir=persistence_dir,
                agent_id=agent_id,
                encoder=encoder,
            )
            trial.attach_instruments()
            try:
                # ── Fingerprint (frozen-apparatus stop rule) ──
                record["fingerprint_live"] = trial.check_fingerprint(FROZEN["usable_oxygen_max"])
                if args.arm == "ablated":
                    detached = _detach_fear_subscriber(aut)
                    record["detached_subscribers"] = detached
                    if detached != 1:
                        raise Refusal(f"expected exactly 1 fear subscriber, detached {detached}")
                record["bridge_state_interval_s"] = trial.check_bridge()
                # ── Loop LIVENESS: the full loop, on the shore, must tick at its cadence ──
                record["loop_liveness_ticks"] = trial.check_liveness()
                trial.check_gamerules()
                # ── LIVE cluster-distinct preflight (the live agent's EC) ──
                shore_cluster_pre, water_cluster_pre = trial.check_clusters_distinct()
                # ── Escape actuation check through the BACKEND, never the executor ──
                trial.resolve_tools()
                record["actuation_preflight"] = trial.check_escape_actuation()
                trial.check_no_positive_escape_link()
                print(
                    f"  preflight: live clusters distinct ({shore_cluster_pre[:8]} vs {water_cluster_pre[:8]}); "
                    f"escape actuation OK ({record['actuation_preflight']['t_surface']}s); no positive escape link"
                )
                trial.deaths0 = trial.deaths()

                pre = trial.probe("pre")
                # The pre-probe must have been US-free: ZERO fear on the water cluster, both arms.
                trial.submerge("post-pre-check")
                water_cluster_after_pre = trial.encode_world_cluster()
                trial.rescue("post-pre-check")
                water_fear_pre = round(aut.bio.nac.cluster_fear(agent_id, water_cluster_after_pre), 4)
                record["water_fear_pre"] = water_fear_pre
                if water_fear_pre != 0.0 or pre["unyoked_us_events"]:
                    raise Refusal(
                        f"pre-probe was NOT US-free: water fear {water_fear_pre}, unyoked US events {pre['unyoked_us_events']} — "
                        "the probe cap is not below the pain edge on this apparatus"
                    )

                # ── Training: yoked, propose-only conditioning at the pool floor ──
                record["training"], episode_clusters = trial.train()

                # ── LIVE G2: readability through the PRODUCTION read (+ ablation verified) ──
                record.update(trial.live_g2(args.arm, episode_clusters, water_cluster_pre))

                post = trial.probe("post")
                record.update(trial.negative_links())
                record["pre"] = pre
                record["post"] = post
                print(
                    f"seed {seed}: pre P(surface)={pre['p_surface']} post P(surface)={post['p_surface']} "
                    f"fear={record['water_fear']}/{record['shore_fear']} escape_neg_links={record['escape_negative_links']} "
                    f"flee_neg_links={record['flee_negative_links']}"
                )
            except (Refusal, InstrumentError) as exc:
                record.update(getattr(exc, "partial", None) or {})  # the refused step's own measurements
                record["refusal"] = str(exc)
                print(f"REFUSED seed {seed}: {exc}")
                exit_code = 4
            finally:
                trial.detach_instruments()
                trial.final_rescue()
                try:
                    pump.stop()
                except Exception as exc:
                    print(f"WARNING: pump stop raised: {exc!r}")
                try:
                    aut.bio.on_session_end()
                except Exception as exc:
                    print(f"WARNING: bio teardown raised: {exc!r}")
                try:
                    aut.client.close()
                except (OSError, ConnectionError) as exc:
                    print(f"WARNING: client close raised: {exc!r}")
                os.chdir(prev_cwd)
                shutil.rmtree(persistence_dir, ignore_errors=True)
            records.append(record)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
    finally:
        rcon.close()

    ok = sum(1 for r in records if r.get("refusal") is None)
    print(f"\narm={args.arm} run={run_id}: {ok}/{len(records)} seeds clean -> {out_path}")
    return exit_code


def _verdict(args: argparse.Namespace) -> int:
    recs = [json.loads(ln) for ln in Path(args.data).expanduser().read_text().splitlines() if ln.strip()]
    v = compute_verdict(recs, run_id=args.run_id)
    v["data"] = str(args.data)
    v["run_ids"] = args.run_id
    print(json.dumps({k: v[k] for k in v if k not in ("per_seed", "gates")}, indent=2))
    print(f"VERDICT: {v['verdict']}")
    if args.json:
        # The verdict is the HEADLINE record: the same gated-evidence path as `run`
        # (dirty-tree refusal + the --write-experiment-results redirect), not the
        # diagnostic-only stamp the L11 probe uses.
        out_arg = Path(args.json)
        out_abs = out_arg if out_arg.is_absolute() else (C.REPO_ROOT / out_arg)
        out = evidence_out_paths_or_exit(
            C.REPO_ROOT,
            [str(out_abs)],
            write_experiment_results=args.write_experiment_results,
            allow_dirty=args.allow_dirty,
        )[0]
        import maxim  # noqa: PLC0415

        v["provenance"] = in_process_code_provenance(
            C.REPO_ROOT, maxim.__file__, out_path=out, allow_dirty=args.allow_dirty
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(v, indent=2) + "\n")
        print(f"[verdict] wrote -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="LIVE: one arm, all seeds")
    r.add_argument("--arm", choices=("fear", "ablated"), required=True)
    r.add_argument("--seeds", type=int, default=len(FROZEN["seeds"]))
    r.add_argument("--out", default="docs/experiments/data/exp60_trials.jsonl")
    r.add_argument("--gate-record", default=GATE_RECORD)
    r.add_argument("--bridge-host", default="127.0.0.1")
    r.add_argument("--bridge-port", type=int, default=25567)
    r.add_argument("--rcon-host", default="127.0.0.1")
    r.add_argument("--rcon-port", type=int, default=25575)
    r.add_argument("--rcon-password", required=True)
    r.add_argument("--username", default="maxim")
    r.add_argument("--write-experiment-results", action="store_true")
    r.add_argument("--allow-dirty", action="store_true")
    r.set_defaults(func=_run)
    v = sub.add_parser("verdict", help="OFFLINE: gate decision from the per-seed JSONL")
    v.add_argument("--data", required=True)
    v.add_argument("--json", default=None)
    v.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="select these runs' rows when the JSONL holds re-runs — repeat once per ARM (each `run` invocation mints its own id)",
    )
    v.add_argument("--write-experiment-results", action="store_true")
    v.add_argument("--allow-dirty", action="store_true")
    v.set_defaults(func=_verdict)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
