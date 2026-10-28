#!/usr/bin/env python3
"""Exp 60 LIVE trial harness — learned drowning-avoidance (FEAR vs ABLATED arms).

Chunk (iii) of the Exp 60 build (docs/experiments/exp60_drowning_avoidance_prereg.md
§Design (iii)). Runs the prereg's design against the live water classroom (built by
``setup_world.py water_classroom``, MEASURED by ``exp60_water_check.py``, separability
gated by ``l11_geometry_probe.py`` run_gate). Modelled on ``exp58_run.py`` (which ran live)
with the changes the four design lenses required. Per seed × arm:

  1. PREFLIGHTS (all before any measurement; refusal = exit 3/4, no behavioural data):
     - both gated records on main PASS (the apparatus check through
       ``_provenance.instrument_check_authorizes``, M1b PR 5a-2, and — for a gated ``--out`` —
       committed on main byte for byte, ``committed_record_problem``; geometry gate
       ``run_gate.pass``, whose on-main state is NOT checked) and the anchor carries the check's
       stamped ``measured`` onset from the SAME check (``anchor_record_mismatch``);
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
    python scripts/survival_world/exp60_run.py run --arm fear \\
        --apparatus-record docs/experiments/data/rerun_exp60_<date>/water_apparatus.json --rcon-password '<pw>' \\
        --username maxim --out docs/experiments/data/rerun_exp60_<date>/trials.jsonl --write-experiment-results
    python scripts/survival_world/exp60_run.py run --arm ablated ...   # same --out + record, other arm, same tree
    python scripts/survival_world/exp60_run.py verdict --data docs/experiments/data/rerun_exp60_<date>/trials.jsonl \\
        --json docs/experiments/data/rerun_exp60_<date>/verdict.json --write-experiment-results

``--apparatus-record`` names the water check that authorizes the run (default ``APPARATUS_RECORD``, the
committed pre-M1b record, which ``instrument_check_authorizes`` refuses: a re-run passes the fresh check's
``--out``). A named record goes through the same rule as the default and must be the check that last stamped the
anchor's ``measured`` block (same ``ts``); a missing or refused record ends the run (exit 3), never the default in
its place. ENFORCED for a gated ``--out`` (under ``docs/experiments/data/``): the record is tracked, its bytes equal
its blob at HEAD, and HEAD is an ancestor of ``origin/main`` (``committed_record_problem``), else exit 3 before any
RCON or bridge connection; ``--allow-dirty`` passes it, stamped as ``apparatus_record_check`` in every row. Every row stamps ``apparatus_record`` (the path read), ``apparatus_record_sha256`` and
``apparatus_record_ts``, and the verdict lists them as ``apparatus_records``. One verdict is one apparatus record:
rows naming more than one (by path + sha256; a legacy row with neither never matches a named one) make the verdict
INCOMPLETE with ``incomplete_cause: "mixed_apparatus_records"``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
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
    append_refusal,
    instrument_check_authorizes,
    is_ancestor_of_main,
    apparatus_check_stamp,
    is_gated_path,
    stamp_harness_row,
    stamp_verdict,
    any_not_stamped_real,
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
        # Which RETENTION MODEL the run used (memory-strength plan §Guardrails). A new key
        # with a stated default, so a recorded row from before 2c-3 reads as "access_based" --
        # which it was, since no other name changed retention behaviour then. ~/.maxim is
        # shared across worktrees, so a stray `maxim config set memory.strategy strength`
        # would otherwise silently reconfigure a campaign with nothing in the record saying so.
        "memory_strategy": "access_based",
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
    duplicate (arm, seed) is a refusal reason; with it, only that run's rows count. Selecting a run here is for
    reading a legacy file: the ledger evidence gate (M1b 5b-2) refuses a verdict over a file holding a second
    complete run of an arm, so a confirmatory re-run goes to a NEW ``--out``, never appended and picked.
    """
    # Each `run` invocation mints its own run_id, so a two-arm trial is TWO ids — one per arm.
    wanted = set([run_id] if isinstance(run_id, str) else (run_id or []))
    rows = [r for r in records if not wanted or r.get("run_id") in wanted]
    seen: dict[tuple[str, Any], int] = {}
    for r in rows:
        seen[(r.get("arm"), r.get("seed"))] = seen.get((r.get("arm"), r.get("seed")), 0) + 1
    dups = sorted(f"{arm}/seed{seed}×{n}" for (arm, seed), n in seen.items() if n > 1)
    return rows, dups


def apparatus_records_used(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every distinct apparatus record the rows say authorized them, as ``{path, sha256, ts}`` (pure).

    A verdict names what its rows were run under: the path and bytes digest since ``--apparatus-record``
    (rows from before it carry only ``apparatus_record_ts``, so their path and digest read ``None``)."""
    seen = {
        (r.get("apparatus_record"), r.get("apparatus_record_sha256"), r.get("apparatus_record_ts")): None
        for r in records
    }
    return [{"path": p, "sha256": s, "ts": t} for p, s, t in sorted(seen, key=repr)]


def apparatus_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    """Which apparatus record a row ran under, for the one-record-per-verdict rule (pure).

    Named by ``(path, sha256)``. A row missing either (a legacy row from before ``--apparatus-record``) cannot say
    which bytes it read, so it is keyed by what it does carry, ``("unnamed", path, sha256, ts)``: it never matches a
    named record (a mix of legacy and named rows is refused), and legacy rows match each other only when they agree
    on every field, the ``ts`` included."""
    path, sha = row.get("apparatus_record"), row.get("apparatus_record_sha256")
    if path is not None and sha is not None:
        return ("named", path, sha)
    return ("unnamed", path, sha, row.get("apparatus_record_ts"))


def anchor_record_mismatch(measured: Any, apparatus: dict[str, Any]) -> str | None:
    """Why the anchor's ``measured`` block is NOT the given apparatus check's, or None when it is (pure).

    ``exp60_water_check`` stamps its measured edges into the anchor on a PASS, carrying the report's own ``ts``. The
    run takes its probe cap from the record and its training cap from the anchor, so the two must be one check: the
    same ``ts`` and the same earliest pain edge."""
    if not isinstance(measured, dict):
        return "the anchor carries no measured block — run exp60_water_check first"
    ts = apparatus.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return f"the apparatus record carries no epoch ts ({ts!r})"
    if measured.get("ts") != ts:
        return (
            f"the anchor's measured edges are from the check at ts {measured.get('ts')!r}, the record is the check at "
            f"ts {ts!r} — pass the record of the check that last stamped the anchor, or re-run the check"
        )
    edge = min_pain_edge_s(apparatus)
    if measured.get("t_pain_edge_min_s") != edge:
        return f"the anchor's pain edge {measured.get('t_pain_edge_min_s')!r} is not the record's {edge!r}"
    return None


def compute_verdict(
    records: list[dict[str, Any]], *, gates: dict[str, Any] = GATES, run_id: "list[str] | str | None" = None
) -> dict[str, Any]:
    """The prereg's gate decision from per-seed records (pure).

    ``verdict`` ∈ {"EARNED", "NULL", "INCOMPLETE"}: INCOMPLETE when either arm has fewer
    clean seeds than the prereg requires, duplicate (arm, seed) rows are present, or the rows
    name more than one apparatus record (judged over the whole file, before ``run_id``; see
    :func:`apparatus_identity`); each INCOMPLETE names its ``incomplete_cause``: ``mixed_apparatus_records``,
    ``duplicate_arm_seed_rows``, ``too_few_clean_seeds`` or ``no_clean_placements`` (refusals are named, never silently dropped); NULL when the arms have the seeds but a
    gate fails (a null ships as a null).
    """
    # One apparatus record per verdict (owner decision 2026-10-10): the probe cap is the record's, so arms (or seeds)
    # run under two records are not one experiment. Judged over the WHOLE file, before `--run-id` selects rows, so a
    # killed run under one record beside complete runs under another still refuses; refused rows count too.
    identities = {apparatus_identity(r) for r in records}
    if len(identities) > 1:
        return {
            "_format_version": "1.0",
            "record_kind": "verdict",
            "kind": "exp60_verdict",
            "verdict": "INCOMPLETE",
            "incomplete_cause": "mixed_apparatus_records",
            "reason": f"the file's rows name {len(identities)} apparatus records (by path + sha256; a legacy row with "
            "no path or digest never matches a named one; judged before --run-id) — one verdict is one apparatus "
            "record: re-run into a fresh --out under one --apparatus-record",
            "apparatus_records": apparatus_records_used(records),
        }
    records, dups = select_run(records, run_id)
    if dups:
        return {
            "_format_version": "1.0",
            "record_kind": "verdict",  # M1b (#1003)
            "kind": "exp60_verdict",
            "verdict": "INCOMPLETE",
            "incomplete_cause": "duplicate_arm_seed_rows",
            "reason": f"duplicate (arm, seed) rows: {dups} — pass --run-id ONCE PER ARM (each arm invocation mints its own id)",
            "duplicates": dups,
            "apparatus_records": apparatus_records_used(records),
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
        "record_kind": "verdict",  # M1b (#1003)
        "kind": "exp60_verdict",
        "gates": gates,
        "per_seed": per_seed,
        "refused": refused,
        "n_clean": {arm: len(rows) for arm, rows in by_arm.items()},
        "apparatus_records": apparatus_records_used(records),
    }
    if any(len(by_arm.get(arm, [])) != need for arm in ("fear", "ablated")):
        out["verdict"] = "INCOMPLETE"
        out["incomplete_cause"] = "too_few_clean_seeds"
        out["reason"] = f"expected exactly {need} clean seeds per arm: {out['n_clean']} (refusals: {len(refused)})"
        return out
    fear_post = [s["post_p_surface"] for s in per_seed["fear"] if s["post_p_surface"] is not None]
    fear_pre = [s["pre_p_surface"] for s in per_seed["fear"] if s["pre_p_surface"] is not None]
    abl_post = [s["post_p_surface"] for s in per_seed["ablated"] if s["post_p_surface"] is not None]
    if len(fear_post) < need or len(abl_post) < need or len(fear_pre) < need:
        out["verdict"] = "INCOMPLETE"
        out["incomplete_cause"] = "no_clean_placements"
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


class ApparatusRecordError(Exception):
    """A named apparatus record that cannot be read as one: the run refuses (exit 3), never falls back."""


def load_apparatus_record(arg: "str | Path") -> dict[str, Any]:
    """Read the apparatus record ``arg`` names (relative = under the repo root) ONCE: its parsed ``record``, the
    ``path`` to stamp (repo-relative when inside the repo, else absolute) and the ``sha256`` of the bytes parsed.
    Raises :class:`ApparatusRecordError` when it is missing, unreadable, not JSON or not an object."""
    raw = Path(arg).expanduser()
    p = raw if raw.is_absolute() else (C.REPO_ROOT / raw)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise ApparatusRecordError(
            f"apparatus record {p} cannot be read ({type(exc).__name__}) — it must exist (on main) before live trials"
        ) from exc
    try:
        record = json.loads(data)
    except ValueError as exc:
        raise ApparatusRecordError(f"apparatus record {p} is not JSON ({exc})") from exc
    if not isinstance(record, dict):
        raise ApparatusRecordError(f"apparatus record {p} is not a JSON object")
    resolved = p.resolve()
    try:
        shown = resolved.relative_to(Path(C.REPO_ROOT).resolve()).as_posix()
    except ValueError:
        shown = str(resolved)
    return {
        "record": record,
        "path": shown,
        "abs_path": resolved,
        "bytes": data,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def committed_record_problem(repo_root: "str | Path", path: "str | Path", data: bytes) -> str | None:
    """Why the apparatus record is NOT the committed, on-main record a gated run needs, or None when it is.

    The bytes the run read and hashed (``data``) must be: (a) a path inside the repo that git tracks, (b) equal to
    that path's blob at HEAD, and (c) from a HEAD that is an ancestor of ``origin/main``
    (``_provenance.is_ancestor_of_main``, shared with R3). Any git failure is a problem: unknown is never committed.
    """
    root = Path(repo_root).resolve()
    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(root).as_posix()
    except ValueError:
        return f"{resolved} is outside the repo ({root}) — a gated run reads a record committed on main"
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel], cwd=root, capture_output=True, timeout=30
        )
        if tracked.returncode != 0:
            return f"{rel} is not tracked by git — commit it (and merge it to main) first"
        blob = subprocess.run(["git", "cat-file", "blob", f"HEAD:{rel}"], cwd=root, capture_output=True, timeout=30)
        if blob.returncode != 0:
            return f"{rel} is not in HEAD — commit it (and merge it to main) first"
        if blob.stdout != data:
            return f"{rel} on disk differs from its blob at HEAD — the run must read the committed bytes"
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=30)
        if head.returncode != 0:
            return f"HEAD cannot be read ({head.stderr.strip()})"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git could not establish the record ({type(exc).__name__}: {exc})"
    sha = head.stdout.strip()
    if not is_ancestor_of_main(root, sha):
        try:
            ref = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", "origin/main^{commit}"],
                cwd=root,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"origin/main cannot be read ({type(exc).__name__}: {exc})"
        if ref.returncode != 0:
            return "origin/main is not a valid ref (fetch it) — the record cannot be shown to be on main"
        return f"HEAD {sha[:12]} is not an ancestor of origin/main — the record is not on main"
    return None


def _write_row(out_path: Path, record: dict) -> None:
    """Append one seed's row, stamped as the evidence gate reads it (M1b PR 5a): record_kind, status
    (a refusal → failed) and mock."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as fh:
        fh.write(json.dumps(stamp_harness_row(record, mock=False)) + "\n")


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
    refusal = append_refusal(out_path, provenance)
    if refusal is not None:
        # One code tree per file, never onto a pre-M1b file (M1b PR 5a-3, #1022): the operator names a fresh file.
        print(f"[FAIL] refusing to append: {refusal}")
        return 2

    # ── Preflight: BOTH gated records PASS (prereg instrument-gate stop rule) ──
    def _load(rel: str) -> dict[str, Any]:
        p = C.REPO_ROOT / rel
        try:
            return json.loads(p.read_text())
        except OSError:
            print(f"[FAIL] gated record missing: {p} — it must be on main before live trials")
            raise SystemExit(3)

    # The record THIS run is authorized by: `--apparatus-record`, default the module's APPARATUS_RECORD. No fallback:
    # a named record that is missing, unreadable or refused ends the run here, never the default in its place.
    try:
        apparatus_used = load_apparatus_record(args.apparatus_record)
    except ApparatusRecordError as exc:
        print(f"[FAIL] {exc}")
        return 3
    apparatus, apparatus_path = apparatus_used["record"], apparatus_used["path"]
    # A GATED run (rows into docs/experiments/data/) reads the record COMMITTED on main, byte for byte: enforced
    # here, before any RCON or bridge connection. --allow-dirty passes it through, stamped into every row.
    gated = is_gated_path(C.REPO_ROOT, out_path)
    record_problem = (
        committed_record_problem(C.REPO_ROOT, apparatus_used["abs_path"], apparatus_used["bytes"]) if gated else None
    )
    if record_problem is not None and not args.allow_dirty:
        print(
            f"[FAIL] apparatus record {apparatus_path} is not the committed record on main a gated run needs: "
            f"{record_problem} (or pass --allow-dirty, stamped into every row)"
        )
        return 3
    if record_problem is not None:
        print(f"WARNING: --allow-dirty: apparatus record {apparatus_path} is not committed on main: {record_problem}")
    # The waiver the ledger gate reads (`_evidence_records.judge_provenance`: allowed-dirty, never sole support), even
    # on a clean tree, and `apparatus_record_check`, which keeps the detail: both stamps from one builder (#1220).
    apparatus_check, provenance = apparatus_check_stamp(gated, record_problem, provenance)
    gate = _load(args.gate_record)
    refusal = instrument_check_authorizes(apparatus)
    if refusal is not None:
        print(f"[FAIL] apparatus record {apparatus_path} does not authorize a run: {refusal}")
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
            f"[FAIL] apparatus record carries no measured t_pain_edge — the probe cap cannot be set: {apparatus_path}"
        )
        return 3
    # The probe cap comes from the record, the training cap from the anchor's `measured` block: both must be the
    # SAME check's, or the run would mix two apparatus measurements (a record chosen by flag makes that easy).
    mismatch = anchor_record_mismatch(geom.get("measured"), apparatus)
    if mismatch is not None:
        print(f"[FAIL] anchor {ANCHOR_FILE} and apparatus record {apparatus_path} disagree: {mismatch}")
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
                # The apparatus record this run read (M1b: what authorized it, not only when that check ran):
                # its path (repo-relative when inside the repo), the sha256 of the bytes read, and its `ts`.
                "apparatus_record": apparatus_path,
                "apparatus_record_sha256": apparatus_used["sha256"],
                "apparatus_record_ts": apparatus.get("ts"),
                # gated: was the committed-on-main preflight applied; problem: why it failed (only with --allow-dirty)
                "apparatus_record_check": apparatus_check,
                "anchor_measured": geom.get("measured"),
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
            _write_row(out_path, record)
    finally:
        rcon.close()

    ok = sum(1 for r in records if r.get("refusal") is None)
    print(f"\narm={args.arm} run={run_id}: {ok}/{len(records)} seeds clean -> {out_path}")
    return exit_code


def _verdict(args: argparse.Namespace) -> int:
    data_bytes = Path(args.data).expanduser().read_bytes()  # read once: the verdict and its hash see the same bytes
    recs = [json.loads(ln) for ln in data_bytes.decode().splitlines() if ln.strip()]
    v = compute_verdict(recs, run_id=args.run_id)
    v["run_ids"] = args.run_id
    # The rows file as a repo-relative path + its sha256, and the scope selecting its rows (M1b PR 5a).
    stamp_verdict(
        v,
        repo_root=C.REPO_ROOT,
        kind="exp60_verdict",
        data=Path(args.data).expanduser(),
        data_bytes=data_bytes,
        mock=any_not_stamped_real(recs),  # a verdict over a smoke, or over unstamped rows, is a smoke
        scope={"run_ids": args.run_id} if args.run_id else {"all_rows": True},
    )
    print(json.dumps({k: v[k] for k in v if k not in ("per_seed", "gates")}, indent=2))
    print(f"VERDICT: {v['verdict']}" + (f" ({v['incomplete_cause']})" if v.get("incomplete_cause") else ""))
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
    r.add_argument("--out", required=True, help="a fresh rows file; both arms go into the same one for `verdict`")
    r.add_argument(
        "--apparatus-record",
        default=APPARATUS_RECORD,
        help="the water apparatus check that authorizes this run (repo-relative or absolute; default %(default)s). "
        "It passes the same rule as the default (a stamped, real, passing instrument check with a measured pain "
        "edge, from the same check as the anchor's `measured` block); a missing or refused record ends the run, "
        "never falls back to the default. With a gated --out (under docs/experiments/data/) it must also be "
        "tracked, byte-equal to its blob at HEAD, and HEAD an ancestor of origin/main (--allow-dirty overrides, "
        "stamped). Its path (repo-relative; absolute when outside the repo, which a gated run refuses) and sha256 "
        "are stamped into every row and the verdict",
    )
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
        help="select these runs' rows when the JSONL holds re-runs — repeat once per ARM (each `run` invocation mints "
        "its own id); evidence re-runs go to a NEW --out: the ledger gate refuses a file with two complete runs of an arm",
    )
    v.add_argument("--write-experiment-results", action="store_true")
    v.add_argument("--allow-dirty", action="store_true")
    v.set_defaults(func=_verdict)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
