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

Run ON the bridge box (server + bridge from current main, classroom built, records merged):

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
import threading
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
from survival_world.common import InstrumentError, settle_until, sync_snapshot  # noqa: E402
from survival_world.exp60_water_check import (  # noqa: E402
    IN_WATER_WITHIN_S,
    RECOVER_OXYGEN_MIN,
    REQUIRED_BRIDGE_SENSORS,
    STALE_MAX_CONSECUTIVE,
    STALE_STATE_S,
    SURFACE_WITHIN_S,
    missing_bridge_sensors,
)
from survival_world.l11_geometry_probe import SATURATION_REST  # noqa: E402

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


class Refusal(RuntimeError):
    """A prereg stop rule fired — the seed must not produce a verdict row."""


# ─────────────────────────── pure helpers (unit-tested) ───────────────────────────


def classify_placement(samples: list[dict[str, Any]], *, cap_s: float) -> dict[str, Any]:
    """One probe placement from ``(t, in_water, health)`` samples, t from teleport (pure).

    ``surfaced`` = the first head-in-air read AFTER the head was seen in water and BEFORE
    any health drop or the cap. A health drop inside the window is a DIRTY placement
    (damage occurred: the window was not damage-free) and censors it.
    """
    seen_water = False
    health0 = None
    for s in samples:
        if s["t"] > cap_s:
            break
        if health0 is None and s.get("health") is not None:
            health0 = s["health"]
        if s.get("health") is not None and health0 is not None and s["health"] < health0:
            return {"surfaced": False, "latency_s": None, "censored": True, "dirty": True, "t_end": s["t"]}
        if s["in_water"]:
            seen_water = True
        elif seen_water:
            return {"surfaced": True, "latency_s": round(s["t"], 3), "censored": False, "dirty": False, "t_end": s["t"]}
    return {
        "surfaced": False,
        "latency_s": None,
        "censored": True,
        "dirty": False,
        "t_end": samples[-1]["t"] if samples else 0.0,
        "never_submerged": not seen_water,
    }


def p_surface(probe: dict[str, Any]) -> float | None:
    """Per-seed P(surface before first damage tick) over the probe's CLEAN placements."""
    ps = [p for p in probe.get("placements", []) if not p.get("dirty") and not p.get("never_submerged")]
    if not ps:
        return None
    return sum(1 for p in ps if p["surfaced"]) / len(ps)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


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


def select_run(records: list[dict[str, Any]], run_id: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Filter to one run and name duplicate (arm, seed) rows (pure).

    A re-run appended to the same JSONL must never double n: without ``run_id`` any
    duplicate (arm, seed) is a refusal reason; with it, only that run's rows count.
    """
    rows = [r for r in records if run_id is None or r.get("run_id") == run_id]
    seen: dict[tuple[str, Any], int] = {}
    for r in rows:
        seen[(r.get("arm"), r.get("seed"))] = seen.get((r.get("arm"), r.get("seed")), 0) + 1
    dups = sorted(f"{arm}/seed{seed}×{n}" for (arm, seed), n in seen.items() if n > 1)
    return rows, dups


def compute_verdict(
    records: list[dict[str, Any]], *, gates: dict[str, Any] = GATES, run_id: str | None = None
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
            "reason": f"duplicate (arm, seed) rows: {dups} — pass --run-id",
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


def fingerprint_drift(live: dict[str, Any], frozen: dict[str, Any]) -> list[str]:
    """Keys whose live value differs from the frozen apparatus (pure; lists compared sorted)."""

    def norm(v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return (
                sorted(norm(x) for x in v) if all(not isinstance(x, (list, dict)) for x in v) else [norm(x) for x in v]
            )
        if isinstance(v, dict):
            return {k: norm(x) for k, x in sorted(v.items())}
        if isinstance(v, float):
            return round(v, 6)
        return v

    return sorted(k for k in set(live) | set(frozen) if norm(live.get(k)) != norm(frozen.get(k)))


def min_pain_edge_s(apparatus: dict[str, Any]) -> float | None:
    """The earliest measured air-hunger pain edge across the apparatus check's cycles (pure)."""
    edges = [c.get("w2_dive", {}).get("t_pain_edge") for c in apparatus.get("cycles", [])]
    edges = [float(e) for e in edges if e is not None]
    return min(edges) if edges else None


# ─────────────────────────────────── live ───────────────────────────────────


def _detach_fear_subscriber(aut: Any) -> int:
    bus = aut.bio.pain_bus
    targets = [cb for cb in list(bus._pain_signal_subs) if "cluster_fear" in getattr(cb, "__qualname__", "")]
    for cb in targets:
        bus.unsubscribe(cb)
    return len(targets)


def _f(vm: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(vm.get(key, default))
    except (TypeError, ValueError):
        return default


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
    shore = {"x": float(geom["shore"][0]), "y": float(geom["shore"][1]), "z": float(geom["shore"][2])}
    sub = {"x": float(geom["submerged"][0]), "y": float(geom["submerged"][1]), "z": float(geom["submerged"][2])}

    from maxim.runtime.agent_loop import _encode_current_clusters, _read_world_ranges, propose_via_substrate
    from maxim.similarity.encoder import SensorEncoderConfig
    from maxim.simulation.minecraft import MinecraftClient
    from maxim.simulation.minecraft_harness import MinecraftSyncPump, build_minecraft_aut, run_minecraft_aut
    from survival_world.common import make_fresh_encoder

    run_id = uuid.uuid4().hex[:12]
    rcon = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    print(
        f"run {run_id}: arm={args.arm} shore={shore} submerged={sub} probe cap={probe_cap_s:.2f}s "
        f"(pain edge min {pain_edge_min:.2f}) train cap={train_cap_s:.2f}s (onset min {onset_min:.2f})"
    )

    def _heal() -> None:
        rcon.command(f"effect give {args.username} minecraft:instant_health 1 10 true")
        rcon.command(f"effect give {args.username} minecraft:saturation 1 10 true")

    def _deaths() -> int:
        resp = rcon.command(f"scoreboard players get {args.username} {geom.get('deaths_objective', 'exp60_deaths')}")
        try:
            return int(resp.split(" has ")[1].split()[0])
        except (IndexError, ValueError):
            return 0

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
            # The bus subscriber runs for the WHOLE seed: every pain publish is timestamped so a
            # probe window can prove it was US-free (or be marked DIRTY + unyoked exposure).
            signals: list[dict[str, Any]] = []

            def _record_pain(sig: Any) -> None:
                ctx = getattr(sig, "context", None) or {}
                signals.append(
                    {
                        "t": time.monotonic(),
                        "failure_mode": ctx.get("failure_mode"),
                        "intensity": float(getattr(sig, "intensity", 0.0)),
                    }
                )

            aut.bio.pain_bus.subscribe(_record_pain)

            def _pain_between(t_a: float, t_b: float) -> list[dict[str, Any]]:
                return [
                    s for s in signals if t_a <= s["t"] <= t_b and s["failure_mode"] in ("drive:oxygen", "drive:health")
                ]

            def _rescue(label: str) -> float:
                """Shore + OBSERVED recovery + satiation. Returns the SENSED health on arrival
                (before the heal) so callers can see whether damage happened."""
                rcon.teleport(args.username, shore)
                vm = settle_until(
                    aut,
                    lambda vm: _f(vm, "is_in_water", 1) < 0.5 and _f(vm, "oxygen", 0) >= RECOVER_OXYGEN_MIN,
                    timeout_s=15.0,
                )
                if vm is None:
                    raise Refusal(f"{label}: rescue did not restore air on the shore")
                arrival_health = _f(vm, "health", 20.0)
                _heal()
                if (
                    settle_until(
                        aut,
                        lambda vm: _f(vm, "health", 0) >= 20.0
                        and _f(vm, "food", 0) >= 16.0
                        and _f(vm, "saturation", 0) >= SATURATION_REST,
                        timeout_s=10.0,
                    )
                    is None
                ):
                    raise Refusal(
                        f"{label}: heal/satiate never settled (health 20, food >= 16, saturation >= {SATURATION_REST})"
                    )
                return arrival_health

            def _stop_motion() -> None:
                try:
                    aut.client.call_action("stop", {})
                except Exception as exc:
                    print(f"WARNING: stop action raised: {exc!r}")

            def _sample(t0: float) -> dict[str, Any] | None:
                if aut.client.state_age_s() > STALE_STATE_S:
                    return None
                vm = sync_snapshot(aut)
                if vm is None or "is_in_water" not in vm:
                    return None  # never let an absent key read as a surface
                return {
                    "t": round(time.monotonic() - t0, 3),
                    "in_water": _f(vm, "is_in_water", 0) >= 0.5,
                    "health": _f(vm, "health", 20.0),
                    "oxygen": _f(vm, "oxygen", 20.0),
                }

            def _submerge(label: str) -> float:
                """Teleport to the pool floor; returns the TELEPORT time (the window's clock)."""
                _stop_motion()
                t_tp = time.monotonic()
                rcon.teleport(args.username, sub)
                if settle_until(aut, lambda vm: _f(vm, "is_in_water", 0) >= 0.5, timeout_s=IN_WATER_WITHIN_S) is None:
                    _rescue(label)
                    raise Refusal(f"{label}: is_in_water did not reflect the submerged teleport")
                return t_tp

            try:
                # ── Fingerprint (frozen-apparatus stop rule) ──
                cfg = aut.bio.nac.config
                oxy = aut.executor.embodiment.root.drive_specs.get("oxygen")
                ranges = _read_world_ranges(aut.executor)
                live_fp = {
                    "cluster_fear_alpha": cfg.cluster_fear_alpha,
                    "max_cluster_fear": cfg.max_cluster_fear,
                    "cluster_fear_threshold": cfg.cluster_fear_threshold,
                    "cluster_fear_failure_modes": sorted(cfg.cluster_fear_failure_modes),
                    "encoder_pattern_threshold": float(SensorEncoderConfig().pattern_threshold),
                    "substrate_explore_bonus_weight": float(getattr(cfg, "substrate_explore_bonus_weight", 0.0)),
                    "oxygen_drive": None
                    if oxy is None
                    else {"set_point": float(oxy.set_point), "comfort_band": float(oxy.comfort_band)},
                    "sensor_ranges": {
                        k: [float(v) for v in ranges[k]] for k in ("is_in_water", "oxygen", "saturation") if k in ranges
                    },
                }
                record["fingerprint_live"] = live_fp
                drift = fingerprint_drift(live_fp, FROZEN["fingerprint"])
                if drift:
                    raise Refusal(f"config fingerprint drift on {drift}: live={live_fp}")
                if oxy is None or FROZEN["usable_oxygen_max"] >= oxy.set_point - oxy.comfort_band:
                    raise Refusal("usable_oxygen_max does not sit below the oxygen comfort band (band-edge trap)")
                if "drive:oxygen" not in cfg.cluster_fear_failure_modes:
                    raise Refusal("drive:oxygen not in the fear allowlist")

                if args.arm == "ablated":
                    detached = _detach_fear_subscriber(aut)
                    record["detached_subscribers"] = detached
                    if detached != 1:
                        raise Refusal(f"expected exactly 1 fear subscriber, detached {detached}")

                if settle_until(aut, lambda vm: "is_in_water" in vm and "oxygen" in vm, timeout_s=10.0) is None:
                    raise Refusal("bridge never delivered state")
                missing = missing_bridge_sensors(aut.client.latest_state(), REQUIRED_BRIDGE_SENSORS)
                if missing:
                    raise Refusal(f"the running bridge does not emit {sorted(missing)} — restart it from current main")
                for rule, want in (
                    ("doMobSpawning", "false"),
                    ("doDaylightCycle", "false"),
                    ("doWeatherCycle", "false"),
                    ("doImmediateRespawn", "true"),
                    ("keepInventory", "true"),
                ):
                    resp = rcon.command(f"gamerule {rule}").strip().lower()
                    if want not in resp:
                        raise Refusal(f"gamerule {rule} is not {want} ({resp!r})")

                # ── LIVE cluster-distinct preflight (the live agent's EC) ──
                _rescue("preflight")
                shore_cluster_pre = _encode_current_clusters(encoder, agent_id, aut.executor).get("world")
                _submerge("preflight")
                water_cluster_pre = _encode_current_clusters(encoder, agent_id, aut.executor).get("world")
                if not water_cluster_pre or water_cluster_pre == shore_cluster_pre:
                    _rescue("preflight")
                    raise Refusal(
                        f"shore and submerged encode to the same LIVE world cluster ({shore_cluster_pre}) — "
                        "the offline gate passed but the live EC does not separate; not a behavioural null"
                    )
                # ── Escape actuation check through the BACKEND, never the executor: an executor
                #    success would book a POSITIVE causal link that makes escape_water selectable
                #    with ZERO fear in both arms (executor lens, CRITICAL). The bridge action is
                #    called directly; bridge truth (is_in_water 0) decides. ──
                escape_tool = next((t for t in aut.executor.registry.list() if t.endswith("_escape_water")), None)
                flee_tool = next((t for t in aut.executor.registry.list() if t.endswith("_flee")), None)
                if escape_tool is None:
                    raise Refusal("no *_escape_water tool registered")
                t0 = time.monotonic()
                outcome: dict[str, Any] = {}

                def _bridge_escape() -> None:
                    try:
                        outcome.update(aut.client.call_action("escape_water", {}))
                    except Exception as exc:
                        outcome["ok"] = False
                        outcome["detail"] = repr(exc)

                th = threading.Thread(target=_bridge_escape, daemon=True)
                th.start()
                surfaced_at = None
                while time.monotonic() - t0 < SURFACE_WITHIN_S + 2.0:
                    s = _sample(t0)
                    if s is not None and not s["in_water"]:
                        surfaced_at = s["t"]
                        break
                    time.sleep(0.25)
                th.join(timeout=10.0)
                record["actuation_preflight"] = {
                    "t_surface": surfaced_at,
                    "bridge": {k: outcome.get(k) for k in ("ok", "detail")},
                }
                if surfaced_at is None or surfaced_at > SURFACE_WITHIN_S:
                    _rescue("preflight")
                    raise Refusal(
                        f"escape actuation check FAILED ({outcome}) — head not in air within {SURFACE_WITHIN_S}s"
                    )
                _rescue("preflight")
                pos_links = len(aut.bio.nac.get_positive_outcomes(f"tool:{escape_tool}"))
                if pos_links:
                    raise Refusal(
                        f"preflight seeded {pos_links} positive causal link(s) on escape_water — the probe would surface without fear"
                    )
                print(
                    f"  preflight: live clusters distinct ({shore_cluster_pre[:8]} vs {water_cluster_pre[:8]}); "
                    f"escape actuation OK ({surfaced_at}s); no positive escape link"
                )
                deaths0 = _deaths()

                def _loop_window(seconds: float, *, enter: Any, on_sample: Any, label: str) -> dict[str, Any]:
                    """WARM the full loop on the shore, `enter()` the situation (returns the
                    window's t0), sample at 4 Hz until `on_sample` says stop or the cap; then
                    RESCUE FIRST (teleport to the shore) and only then stop/join the loop."""
                    stop = threading.Event()
                    actions0 = len(getattr(aut.executor, "_tools_succeeded", []) or [])
                    loop = threading.Thread(
                        target=run_minecraft_aut,
                        args=(aut,),
                        kwargs={"max_steps": 100_000, "target_hz": FROZEN["loop_hz"], "stop_event": stop},
                        daemon=True,
                    )
                    loop.start()
                    time.sleep(FROZEN["loop_warm_s"])  # loop boot is NOT inside the window
                    samples: list[dict[str, Any]] = []
                    stale = 0
                    t0 = enter()
                    t_rescue = None
                    stuck = False
                    try:
                        while time.monotonic() - t0 < seconds:
                            s = _sample(t0)
                            if s is None:
                                stale += 1
                                if stale >= STALE_MAX_CONSECUTIVE:
                                    raise InstrumentError(
                                        f"{label}: bridge stopped delivering fresh state inside a loop window"
                                    )
                            else:
                                stale = 0
                                samples.append(s)
                                if on_sample(s):
                                    break
                            time.sleep(0.25)
                    finally:
                        t_rescue = time.monotonic()
                        try:
                            rcon.teleport(args.username, shore)  # rescue BEFORE the loop drains
                        except Exception as exc:
                            print(f"WARNING: rescue teleport raised: {exc!r}")
                        stop.set()
                        loop.join(timeout=20.0)
                        stuck = loop.is_alive()
                        _stop_motion()
                    if stuck:
                        raise Refusal(f"{label}: loop thread did not stop")
                    succeeded = list(getattr(aut.executor, "_tools_succeeded", []) or [])[actions0:]
                    return {
                        "samples": samples,
                        "actions": succeeded,
                        "t0": t0,
                        "t_rescue": t_rescue,
                        "us_events": _pain_between(t0, t_rescue),
                    }

                def _probe(label: str) -> dict[str, Any]:
                    placements = []
                    for i in range(FROZEN["placements_per_probe"]):
                        _rescue(f"{label}-placement-{i}")
                        seen = {"water": False, "h0": None}

                        def _until(s: dict[str, Any]) -> bool:
                            if seen["h0"] is None:
                                seen["h0"] = s["health"]
                            if s["health"] < seen["h0"]:
                                return True  # damage: rescue NOW (dirty placement)
                            if s["in_water"]:
                                seen["water"] = True
                                return False
                            return seen["water"]  # first head-in-air read after being submerged

                        win = _loop_window(
                            probe_cap_s,
                            enter=lambda: _submerge(f"{label}-placement-{i}"),
                            on_sample=_until,
                            label=label,
                        )
                        arrival_health = _rescue(f"{label}-placement-{i}-after")
                        cls = classify_placement(win["samples"], cap_s=probe_cap_s)
                        cls["actions"] = win["actions"]
                        cls["escape_water_calls"] = sum(1 for a in win["actions"] if a.endswith("_escape_water"))
                        cls["flee_calls"] = sum(1 for a in win["actions"] if a.endswith("_flee"))
                        cls["us_events"] = win["us_events"]
                        cls["arrival_health"] = arrival_health
                        if win["us_events"] or arrival_health < 20.0:
                            # the window was NOT US-free / damage-free: exclude and count as unyoked exposure
                            cls["dirty"] = True
                            cls["surfaced"] = False
                            cls["censored"] = True
                        placements.append(cls)
                        print(
                            f"  {label} placement {i}: {'SURFACED %.2fs' % cls['latency_s'] if cls['surfaced'] else 'censored'}"
                            f"{' [DIRTY]' if cls['dirty'] else ''} actions={len(win['actions'])} flee={cls['flee_calls']}"
                        )
                        if _deaths() - deaths0 > FROZEN["death_cap"]:
                            raise Refusal(f"death cap exceeded ({_deaths() - deaths0})")
                    # Shore free-roam: activity control + P(enter water) secondary (structurally near
                    # 0 v 0 — no drive fires on the shore; it re-tests specificity, recorded not gated)
                    _rescue(f"{label}-roam")
                    roam = _loop_window(
                        FROZEN["shore_roam_s"],
                        enter=time.monotonic,
                        on_sample=lambda s: s["in_water"],
                        label=f"{label}-roam",
                    )
                    _rescue(f"{label}-roam-end")
                    return {
                        "placements": placements,
                        "p_surface": p_surface({"placements": placements}),
                        "cap_s": probe_cap_s,
                        "unyoked_us_events": sum(len(p["us_events"]) for p in placements),
                        "positive_escape_links": len(aut.bio.nac.get_positive_outcomes(f"tool:{escape_tool}")),
                        "shore_roam": {
                            "actions": roam["actions"],
                            "entered_water": any(s["in_water"] for s in roam["samples"]),
                            "window_s": FROZEN["shore_roam_s"],
                        },
                    }

                pre = _probe("pre")
                # The pre-probe must have been US-free: ZERO fear on the water cluster, both arms.
                _submerge("post-pre-check")
                water_cluster_after_pre = _encode_current_clusters(encoder, agent_id, aut.executor).get("world")
                _rescue("post-pre-check")
                water_fear_pre = round(aut.bio.nac.cluster_fear(agent_id, water_cluster_after_pre), 4)
                record["water_fear_pre"] = water_fear_pre
                if water_fear_pre != 0.0 or pre["unyoked_us_events"]:
                    raise Refusal(
                        f"pre-probe was NOT US-free: water fear {water_fear_pre}, unyoked US events {pre['unyoked_us_events']} — "
                        "the probe cap is not below the pain edge on this apparatus"
                    )

                # ── Training: yoked, propose-only conditioning at the pool floor ──
                usable = 0
                attempts = 0
                episode_clusters: list[str] = []
                deadline = time.monotonic() + FROZEN["K_usable_episodes"] * (train_cap_s + 20.0) * 1.5
                while usable < FROZEN["K_usable_episodes"] and time.monotonic() < deadline:
                    attempts += 1
                    _rescue(f"train-{attempts}")
                    t0 = _submerge(f"train-{attempts}")
                    while time.monotonic() - t0 < train_cap_s:
                        n_before = len(signals)
                        propose_via_substrate(
                            nac=aut.bio.nac, agent_id=agent_id, executor=aut.executor, sensor_encoder=encoder
                        )
                        new = [
                            s
                            for s in signals[n_before:]
                            if s["failure_mode"] == "drive:oxygen"
                            and s["intensity"] >= FROZEN["usable_pain_intensity_min"]
                        ]
                        vm = sync_snapshot(aut) or {}
                        noted = aut.bio.nac.active_clusters(agent_id).get("world")
                        if (
                            new
                            and noted
                            and _f(vm, "is_in_water", 0) >= 0.5
                            and _f(vm, "oxygen", 99) <= FROZEN["usable_oxygen_max"]
                        ):
                            episode_clusters.append(noted)
                            usable += 1
                            print(
                                f"  training: usable episode {usable}/{FROZEN['K_usable_episodes']} (oxygen {vm.get('oxygen')})"
                            )
                            break
                        time.sleep(1.0 / FROZEN["loop_hz"])
                    arrival_health = _rescue(f"train-{attempts}-end")
                    if arrival_health < 20.0:
                        raise Refusal(
                            f"drowning DAMAGE during propose-only training (health {arrival_health}) — the cap did not keep conditioning pre-damage"
                        )
                    for _ in range(4):  # healthy ticks: the latch observes recovery
                        propose_via_substrate(
                            nac=aut.bio.nac, agent_id=agent_id, executor=aut.executor, sensor_encoder=encoder
                        )
                        time.sleep(0.25)
                    if _deaths() - deaths0 > FROZEN["death_cap"]:
                        raise Refusal(f"death cap exceeded ({_deaths() - deaths0})")
                record["training"] = {
                    "usable_episodes": usable,
                    "attempts": attempts,
                    "oxygen_pain_signals": sum(1 for s in signals if s["failure_mode"] == "drive:oxygen"),
                    "health_pain_signals": sum(1 for s in signals if s["failure_mode"] == "drive:health"),
                    "episode_clusters": episode_clusters,
                    "deaths": _deaths() - deaths0,
                }
                if usable < FROZEN["K_usable_episodes"]:
                    raise Refusal(f"only {usable}/{FROZEN['K_usable_episodes']} usable episodes")
                if record["training"]["health_pain_signals"]:
                    raise Refusal("drowning DAMAGE pain fired during propose-only training")

                # ── LIVE G2: readability through the PRODUCTION read (+ ablation verified) ──
                _submerge("g2")
                water_cluster = _encode_current_clusters(encoder, agent_id, aut.executor).get("world")
                _rescue("g2")
                shore_cluster = _encode_current_clusters(encoder, agent_id, aut.executor).get("world")
                theta = float(cfg.cluster_fear_threshold)
                water_fear = round(aut.bio.nac.cluster_fear(agent_id, water_cluster), 4)
                shore_fear = round(aut.bio.nac.cluster_fear(agent_id, shore_cluster), 4) if shore_cluster else 0.0
                need_probe = aut.bio.nac.anticipatory_threat_need(agent_id, {"world": water_cluster})
                need_episodes = {
                    cid: aut.bio.nac.anticipatory_threat_need(agent_id, {"world": cid}) for cid in set(episode_clusters)
                }
                record.update({"water_fear": water_fear, "shore_fear": shore_fear})
                lock = getattr(aut.bio.nac, "_lock", None)
                with lock if lock is not None else _NullCtx():
                    record["cluster_fear_dump"] = {
                        f"{cid}|{fm}": v
                        for (aid, cid, fm), v in getattr(aut.bio.nac, "_cluster_fear", {}).items()
                        if aid == agent_id
                    }
                # the loop's activation floor is STRICT (> 0.5): a need of exactly θ is dead at recall
                live_floor = 0.5
                if args.arm == "fear":
                    g2_pass = need_probe > live_floor and all(n > live_floor for n in need_episodes.values())
                else:
                    g2_pass = water_fear == 0.0 and shore_fear == 0.0 and need_probe == 0.0
                record["live_g2"] = {
                    "pre_water_cluster": water_cluster_pre,
                    "training_majority_cluster": max(set(episode_clusters), key=episode_clusters.count)
                    if episode_clusters
                    else None,
                    "probe_water_cluster": water_cluster,
                    "probe_shore_cluster": shore_cluster,
                    "distinct_episode_clusters": len(set(episode_clusters)),
                    "need_probe_cluster": need_probe,
                    "need_episode_clusters": need_episodes,
                    "specificity_ok": abs(shore_fear) < FROZEN["specificity_ratio"] * abs(water_fear)
                    if water_fear
                    else None,
                    "pass": g2_pass,
                }
                if not g2_pass:
                    raise Refusal(
                        f"LIVE G2 FAILED ({args.arm}): need on probe cluster={need_probe}, on episode clusters={need_episodes}, "
                        f"water fear={water_fear} shore fear={shore_fear} (θ={theta}, floor {live_floor}) — fear not readable at "
                        "recall through the production read; must not ship as a behavioural null"
                    )

                post = _probe("post")
                record["escape_negative_links"] = len(aut.bio.nac.get_negative_outcomes(f"tool:{escape_tool}"))
                record["flee_negative_links"] = (
                    len(aut.bio.nac.get_negative_outcomes(f"tool:{flee_tool}")) if flee_tool else None
                )
                record["pre"] = pre
                record["post"] = post
                print(
                    f"seed {seed}: pre P(surface)={pre['p_surface']} post P(surface)={post['p_surface']} "
                    f"fear={water_fear}/{shore_fear} escape_neg_links={record['escape_negative_links']} flee_neg_links={record['flee_negative_links']}"
                )
            except (Refusal, InstrumentError) as exc:
                record["refusal"] = str(exc)
                print(f"REFUSED seed {seed}: {exc}")
                exit_code = 4
            finally:
                try:
                    aut.bio.pain_bus.unsubscribe(_record_pain)
                except Exception as exc:
                    print(f"WARNING: pain unsubscribe raised: {exc!r}")
                try:
                    rcon.teleport(args.username, shore)
                    _heal()
                except Exception as exc:
                    print(f"WARNING: final rescue raised: {exc!r}")
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


class _NullCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def _verdict(args: argparse.Namespace) -> int:
    recs = [json.loads(ln) for ln in Path(args.data).expanduser().read_text().splitlines() if ln.strip()]
    v = compute_verdict(recs, run_id=args.run_id)
    v["data"] = str(args.data)
    v["run_id"] = args.run_id
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
    v.add_argument("--run-id", default=None, help="select one run's rows when the JSONL holds re-runs")
    v.add_argument("--write-experiment-results", action="store_true")
    v.add_argument("--allow-dirty", action="store_true")
    v.set_defaults(func=_verdict)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
