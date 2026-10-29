#!/usr/bin/env python3
"""Baseline floors B1 — the Exp 60 floor: FEAR vs RANDOM vs tabular Q on the live water classroom.

Runs docs/experiments/b1_exp60_floor_prereg.md (build plan: docs/plans/baseline_floors_b1_build.md PR-B). One
multi-arm driver: per seed (11–18) the five arms run in the frozen Williams order, each arm-seed on a fresh agent
with Exp 60's apparatus, preflights and probes. Only the POLICY that picks the substrate proposal differs:

- FEAR — ``recommend_action`` (no policy), Exp 60's propose-only training and live G2, unchanged.
- RANDOM — :class:`floor_policies.RandomPolicy` over the stamped ``available_tools``; no training.
- Q_EXPLORE / Q_EXPLORE_PRIOR / Q_EXPLORE_RAW — :class:`floor_policies.TabularQPolicy` (full repertoire / the two
  threat-affinity tools / raw-sensor keys), trained over K = 10 EXECUTED episodes through real loop windows.

A floor policy reaches the loop through the ``SubstratePolicy`` hook (``WaterTrial.substrate_policy``), wrapped in
:class:`floor_policies.ArmedPolicy`: armed just before each submerging teleport and for the shore roam, idle
otherwise (loop liveness and the shore warm-up run disarmed). Every placement and episode ends with ``end_episode``
in the frozen order (disarm → ``end_episode`` → RCON rescue → loop stop → bridge ``stop``).

Data: one CAMPAIGN per file (``campaign_id``, the session), a ``session`` row first, then per arm-seed attempt an
``attempt_start`` header followed by one ``row``, each with its arm's own ``run_id``. A start left without a row (the
process died mid-attempt) is closed on ``--resume`` by an ``unrecorded`` row, never rerun. An instrument-cause
refusal reruns that arm-seed once (rule 6), so a key can carry a refused row and a clean one; the verdict refuses
anything else. Data names are ``b1_*`` only.

    export PYTHONPATH="$PWD/src"
    python scripts/survival_world/b1_floor_run.py run --apparatus-record docs/experiments/data/<check>.json \\
        --rcon-password '<pw>' --write-experiment-results
    python scripts/survival_world/b1_floor_run.py run --diagnostic --arms random,q_explore ...   # seed 99, never pooled
    python scripts/survival_world/b1_floor_run.py verdict --data docs/experiments/data/b1_exp60_floor_<id>.jsonl \\
        --json docs/experiments/data/b1_exp60_floor_<id>_verdict.json --write-experiment-results

The pure parts (the order, the verdict, the end-of-episode rule, the rig check, resume) are unit-tested in
tests/unit/test_b1_floor_run.py; the trial against the scripted water bridge in the same file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _provenance import (  # noqa: E402
    DirtyTreeError,
    ProvenanceError,
    any_not_stamped_real,
    append_refusal,
    campaign_out_path,
    evidence_out_paths_or_exit,
    in_process_code_provenance,
    instrument_check_authorizes,
    apparatus_check_stamp,
    is_gated_path,
    stamp_harness_header,
    stamp_harness_row,
    stamp_verdict,
)
from exp56 import common as C  # noqa: E402
from survival_world import floor_policies as FP  # noqa: E402
from survival_world.common import InstrumentError  # noqa: E402
from survival_world.exp60_run import (  # noqa: E402  (pure helpers + the frozen Exp 60 apparatus only)
    ANCHOR_FILE,
    APPARATUS_RECORD,
    GATE_RECORD,
    ApparatusRecordError,
    anchor_record_mismatch,
    apparatus_identity,
    apparatus_records_used,
    committed_record_problem,
    exact_permutation_p,
    load_apparatus_record,
)
from survival_world.exp60_run import FROZEN as EXP60_FROZEN  # noqa: E402
from survival_world.water_trial import Refusal, WaterTrial, min_pain_edge_s, p_surface  # noqa: E402

DATA_STEM = "b1_exp60_floor"
ARMS = ("fear", "random", "q_explore", "q_explore_prior", "q_explore_raw")
ARM_LETTER = {"fear": "F", "random": "R", "q_explore": "Q", "q_explore_prior": "P", "q_explore_raw": "W"}
GATED_ARMS = ("fear", "random", "q_explore")
Q_ARMS = ("q_explore", "q_explore_prior", "q_explore_raw")
SEEDS = (11, 12, 13, 14, 15, 16, 17, 18)
DIAGNOSTIC_SEED = 99
PAIN_MODES = ("drive:oxygen", "drive:health")


def williams_rows() -> list[tuple[int, ...]]:
    """The Williams design for five treatments (prereg rule 8): first sequence 0 1 4 2 3, square A = its five cyclic
    shifts, square B = their reversals (pure)."""
    first = (0, 1, 4, 2, 3)
    square_a = [tuple((t + k) % 5 for t in first) for k in range(5)]
    return square_a + [tuple(reversed(r)) for r in square_a]


# Treatment index → arm: 0 F, 1 R, 2 Q, 3 P, 4 W. Seeds 11–15 take square A, 16–18 the first three rows of square B.
ORDER: dict[int, tuple[str, ...]] = {
    seed: tuple(ARMS[i] for i in row) for seed, row in zip(SEEDS, williams_rows()[: len(SEEDS)])
}
ORDER[DIAGNOSTIC_SEED] = ARMS

# Exp 60's frozen apparatus (fingerprint, probe/loop constants), plus B1's own frozen values (the prereg's).
FROZEN: dict[str, Any] = {
    **EXP60_FROZEN,
    "seeds": SEEDS,
    "diagnostic_seed": DIAGNOSTIC_SEED,
    "arms": ARMS,
    "order": {str(s): "".join(ARM_LETTER[a] for a in ORDER[s]) for s in SEEDS},
    "K_executed_episodes": 10,  # Q arms: executed training episodes, whatever ends them (rule 3)
    "q_train_cap_margin_s": 3.0,  # Q episode cap = measured damage onset (min) − this
    "max_attempts": 2,  # one mechanical rerun for an instrument cause (rule 6)
    "rig_tps_min": 19.5,  # rule 11
    "rig_idle_s": 120.0,  # rule 11: no leader-proxy request in this window
    "q": {
        "alpha": FP.ALPHA,
        "gamma": FP.GAMMA,
        "q_init": FP.Q_INIT,
        "epsilon_train": FP.EPSILON_TRAIN,
        "epsilon_probe": FP.EPSILON_PROBE,
        "oxygen_rest": FP.OXYGEN_REST,
        "cost_tick_s": FP.COST_TICK_S,
        "pain_terminal": FP.PAIN_TERMINAL,
        "raw_oxygen_bins": FP.RAW_OXYGEN_BINS,
        "prior_suffixes": FP.PRIOR_SUFFIXES,
    },
}
GATES: dict[str, Any] = {
    "seeds_per_gated_arm": len(SEEDS),
    "fear_first_post_min": 6,  # FEAR sanity gate: first-placement successes out of 8 (else INCOMPLETE)
    "family_alpha": 0.05,  # H1 + H2, Holm-corrected
}
VERDICTS = ("FLOOR_MATCHED_CHANCE", "FLOOR_MATCHED_Q", "FLOORS_BEATEN", "INCOMPLETE")
# A row's ``refusal_cause`` (None = clean). Only "instrument" is ever rerun (rule 6); "unrecorded" closes an attempt
# that started but wrote no row (the process ended mid-attempt; owner decision D2: consumed, never rerun). Pinned to
# the evidence gate's ``b1_campaign`` rule.
REFUSAL_CAUSES = (None, "instrument", "behavioural", "unrecorded")
UNRECORDED_REFUSAL = "unrecorded: the attempt started but wrote no row (the process ended mid-attempt)"


class BehaviouralRefusal(Refusal):
    """A refusal that is never rerun (rule 6): damage, a policy fault, > 2 deaths, FEAR's training or G2."""


# ─────────────────────────── pure helpers (unit-tested) ───────────────────────────


def policy_seed(arm: str, seed: int) -> int:
    """The policy RNG's seed, per seed × arm, process-independent."""
    from maxim.utils.seeding import stable_hash_32

    return stable_hash_32(f"b1|{arm}|{seed}")


def episode_end_reason(samples: list[dict[str, Any]], *, pains: list[dict[str, Any]]) -> str:
    """How a probe placement ended, for ``end_episode`` (rule 4; pure): ``dirty`` (a health drop or any pain in the
    window), ``never_submerged``, ``surfaced`` (head in air after being in water) or ``cap``."""
    health0 = next((s["health"] for s in samples if s.get("health") is not None), None)
    if pains or any(s.get("health") is not None and health0 is not None and s["health"] < health0 for s in samples):
        return "dirty"
    if not any(s["in_water"] for s in samples):
        return "never_submerged"
    if not samples[-1]["in_water"]:
        return "surfaced"
    return "cap"


def training_terminal(
    s: dict[str, Any], state: dict[str, Any], *, oxygen_max: float, saturating_pain: bool
) -> str | None:
    """A Q training episode's terminal from one 4 Hz bridge sample (rule 3; pure). ``state`` carries ``h0`` and
    ``seen_water`` across samples. Returns ``damage`` / ``pain`` / ``surfaced`` or None (keep going)."""
    if state.get("h0") is None:
        state["h0"] = s["health"]
    if s["health"] < state["h0"]:
        return "damage"
    if s["in_water"]:
        state["seen_water"] = True
        if s["oxygen"] <= oxygen_max or saturating_pain:
            return "pain"
        return None
    return "surfaced" if state.get("seen_water") else None


def parse_tick_query(resp: str) -> float | None:
    """The server's ticks per second from ``tick query`` (pure): the target rate, capped by 1000 / the average
    milliseconds per tick. None when either number cannot be read (the rig check then refuses)."""
    import re

    target = re.search(r"target tick rate:\s*([0-9.]+)", resp, re.IGNORECASE)
    avg = re.search(r"average time per tick:\s*([0-9.]+)\s*ms", resp, re.IGNORECASE)
    if not target or not avg:
        return None
    rate, mspt = float(target.group(1)), float(avg.group(1))
    return rate if mspt <= 0 else min(rate, 1000.0 / mspt)


class RigProblem(NamedTuple):
    """One reason the rig is not quiet. ``transient``: waiting can clear it (a job in flight, a recent request, a
    TPS dip); otherwise it cannot, and the attempt refuses at once instead of polling (rig, 2026-10-13: a leader
    proxy with no lane metrics sat out every attempt's full wait and was refused anyway)."""

    message: str
    transient: bool


def rig_problems(
    status: Mapping[str, Any] | None, last_requests: Mapping[str, Any] | None, tps: float | None, *, now: float
) -> list[RigProblem]:
    """Why the rig is not quiet (rule 11; pure). ``status``/``last_requests`` None = no leader proxy on this box
    (passes; recorded by the caller). A present proxy that reports no lane metrics REFUSES (idle cannot be shown):
    run B1 with the leader down, recorded ``absent``."""
    problems = []
    if status is not None:
        lane = status.get("infer_lane")
        if not isinstance(lane, Mapping) or lane.get("in_flight") is None:
            problems.append(
                RigProblem("the leader proxy reports no infer_lane.in_flight (idle cannot be shown)", False)
            )
        elif lane["in_flight"] != 0:
            problems.append(RigProblem(f"an inference job is in flight on this box ({lane['in_flight']})", True))
        recent = (last_requests or {}).get("recent")
        if not isinstance(recent, list):
            problems.append(RigProblem("the leader proxy's last-requests ring cannot be read", False))
        else:
            stamps = [r.get("timestamp") if isinstance(r, Mapping) else None for r in recent]
            if any(isinstance(t, bool) or not isinstance(t, (int, float)) for t in stamps):
                problems.append(RigProblem("a leader-proxy request entry carries no numeric `timestamp`", False))
            late = [t for t in stamps if isinstance(t, (int, float)) and t > now - FROZEN["rig_idle_s"]]
            if late:
                problems.append(
                    RigProblem(f"{len(late)} leader-proxy request(s) in the last {FROZEN['rig_idle_s']:.0f} s", True)
                )
    if tps is None:
        problems.append(RigProblem("server TPS could not be read from `tick query`", False))
    elif tps < FROZEN["rig_tps_min"]:
        problems.append(RigProblem(f"server TPS {tps:.2f} < {FROZEN['rig_tps_min']}", True))
    return problems


class RigNotQuiet(InstrumentError):
    """The rig check refused. ``transient`` is True only when every problem can clear by waiting."""

    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


def key_done(row: Mapping[str, Any], attempt: int) -> bool:
    """Whether a key is finished after ``row``, its attempt number ``attempt`` (pure; the one rule ``attempts_done``,
    ``_after_attempt`` and ``compute_verdict`` share): a clean row, any cause other than "instrument" ("behavioural"
    and "unrecorded" are final), or ``max_attempts`` attempts."""
    return row.get("refusal") is None or row.get("refusal_cause") != "instrument" or attempt >= FROZEN["max_attempts"]


def apparatus_day_problem(apparatus_ts: Any, now: float) -> str | None:
    """Why the apparatus record is not from the campaign's UTC start day, or None (prereg §Run runbook step 1, "same
    day", mechanized; pure). Applied at a fresh gated campaign's start only: a ``--resume`` keeps its campaign's one
    record (the verdict refuses a file that names two), and a diagnostic is never gated."""
    import datetime as dt

    if isinstance(apparatus_ts, bool) or not isinstance(apparatus_ts, (int, float)):
        return f"the apparatus record carries no epoch ts ({apparatus_ts!r})"
    day = dt.datetime.fromtimestamp(float(apparatus_ts), tz=dt.timezone.utc).date()
    today = dt.datetime.fromtimestamp(now, tz=dt.timezone.utc).date()
    if day != today:
        return f"the apparatus check ran on {day} (UTC), not today ({today}): run a fresh check (runbook step 1)"
    return None


def attempts_done(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """Per (arm, seed): the attempts recorded and whether another is owed (resume; pure). Rows are read in attempt
    order (never file order); a key is DONE once any of its rows satisfies :func:`key_done`."""
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("kind") == "row":
            by_key.setdefault((r["arm"], r["seed"]), []).append(r)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, attempts in by_key.items():
        attempts = sorted(attempts, key=lambda r: r.get("attempt", 1))
        out[key] = {
            "attempts": len(attempts),
            "done": any(key_done(r, i) for i, r in enumerate(attempts, 1)),
        }
    return out


def _attempt_key(r: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (r.get("arm"), r.get("seed"), r.get("attempt"))


def attempt_start_record(
    campaign_id: str, run_id: str, arm: str, seed: int, attempt: int, *, provenance: dict[str, Any]
) -> dict[str, Any]:
    """The header line written before every arm-seed attempt (rule item 6 of the ``b1_campaign`` gate rule). The gate
    judges every header's provenance, so ``provenance`` is a REQUIRED keyword: a start without it would make the whole
    campaign file unestablishable (the PR-C review's blocker), and forgetting it is now a TypeError."""
    return {
        "kind": "attempt_start",
        "ts": time.time(),
        "campaign_id": campaign_id,
        "run_id": run_id,
        "arm": arm,
        "seed": seed,
        "attempt": attempt,
        "provenance": provenance,
    }


def orphan_starts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every ``attempt_start`` with no ``row`` of the same (arm, seed, attempt), in file order, once each (pure)."""
    recorded = {_attempt_key(r) for r in rows if r.get("kind") == "row"}
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for r in rows:
        k = _attempt_key(r)
        if r.get("kind") == "attempt_start" and k not in recorded and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def closing_rows(rows: list[dict[str, Any]], base: Any) -> list[dict[str, Any]]:
    """One ``unrecorded`` row per orphan start (resume; D2: the attempt is consumed, never rerun). ``base(arm, seed,
    attempt)`` gives the fields a normal row carries (provenance, apparatus); the start's own campaign and run ids
    are kept."""
    out = []
    for start in orphan_starts(rows):
        row = base(start["arm"], start["seed"], start["attempt"])
        row.update(
            {
                "ts": time.time(),
                "campaign_id": start.get("campaign_id"),
                "run_id": start.get("run_id"),
                "refusal": UNRECORDED_REFUSAL,
                "refusal_cause": "unrecorded",
                "refusal_step": "unrecorded",
            }
        )
        out.append(row)
    return out


def first_post(row: dict[str, Any]) -> dict[str, Any] | None:
    placements = (row.get("post") or {}).get("placements") or []
    return placements[0] if placements else None


def holm(pvalues: dict[str, float | None], alpha: float) -> dict[str, bool]:
    """Holm's step-down over a family (pure): a None p is never significant and stops the step-down."""
    order = sorted(pvalues, key=lambda k: (pvalues[k] is None, pvalues[k] or 0.0))
    m, out, still = len(order), {}, True
    for i, name in enumerate(order):
        p = pvalues[name]
        still = still and p is not None and p < alpha / (m - i)
        out[name] = still
    return out


def compute_verdict(records: list[dict[str, Any]], *, gates: dict[str, Any] = GATES) -> dict[str, Any]:
    """The prereg's verdict from one campaign's rows (pure). INCOMPLETE names its ``incomplete_cause``."""
    base: dict[str, Any] = {"_format_version": "1.0", "record_kind": "verdict", "kind": "b1_verdict", "gates": gates}

    def incomplete(cause: str, reason: str, **extra: Any) -> dict[str, Any]:
        return {**base, "verdict": "INCOMPLETE", "incomplete_cause": cause, "reason": reason, **extra}

    rows = [r for r in records if r.get("kind") == "row"]
    starts = [r for r in records if r.get("kind") == "attempt_start"]  # read only for the campaign and pairing checks
    campaigns = {r.get("campaign_id") for r in records}
    if len(campaigns) != 1:
        return incomplete("mixed_campaigns", f"the rows name {len(campaigns)} campaigns (one per file)")
    identities = {apparatus_identity(r) for r in rows}
    if len(identities) > 1:
        return incomplete(
            "mixed_apparatus_records",
            f"the rows name {len(identities)} apparatus records — one verdict is one apparatus record",
            apparatus_records=apparatus_records_used(rows),
        )
    foreign = sorted({r.get("seed") for r in rows if r.get("seed") not in SEEDS}, key=repr)
    if foreign:
        return incomplete("foreign_seed_rows", f"rows for seeds outside the frozen set (never pooled): {foreign}")
    unknown = sorted({str(r.get("arm")) for r in rows if r.get("arm") not in ARMS})
    if unknown:
        return incomplete("unknown_arm_rows", f"rows for unknown arms: {unknown}")
    run_arms: dict[str, set[str]] = {}
    for r in rows:
        run_arms.setdefault(str(r.get("run_id")), set()).add(r["arm"])
    arm_runs: dict[str, set[str]] = {}
    for rid, arms in run_arms.items():
        for a in arms:
            arm_runs.setdefault(a, set()).add(rid)
    if any(len(a) != 1 for a in run_arms.values()) or any(len(r) != 1 for r in arm_runs.values()):
        return incomplete("run_ids_not_one_per_arm", "each arm has exactly one run_id and each run_id one arm")
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        by_key.setdefault((r["arm"], r["seed"]), []).append(r)
    final: dict[tuple[str, int], dict[str, Any]] = {}
    for key, attempts in by_key.items():
        attempts.sort(key=lambda r: r.get("attempt", 1))
        if [r.get("attempt", 1) for r in attempts] != list(range(1, len(attempts) + 1)):
            return incomplete("attempt_numbering", f"{key}: attempts are not numbered 1..n")
        if len(attempts) > gates.get("max_attempts", FROZEN["max_attempts"]):
            return incomplete("too_many_attempts", f"{key}: {len(attempts)} attempts")
        for earlier in attempts[:-1]:
            if earlier.get("refusal") is None:
                return incomplete("rerun_after_clean_row", f"{key}: a clean row was rerun")
            cause = earlier.get("refusal_cause")
            if cause != "instrument":  # only an instrument cause is rerun-eligible; "unrecorded" never is
                if cause == "unrecorded":
                    return incomplete("rerun_after_unrecorded_attempt", f"{key}: an unrecorded attempt was rerun")
                return incomplete("rerun_after_behavioural_refusal", f"{key}: a {cause!r} refusal was rerun")
        final[key] = attempts[-1]
    # Every row has exactly one attempt_start and every attempt_start exactly one row (D2): a deleted row leaves its
    # start behind, and an orphan start is never a present key.
    n_starts: dict[tuple[Any, Any, Any], int] = {}
    n_rows: dict[tuple[Any, Any, Any], int] = {}
    for r in starts:
        n_starts[_attempt_key(r)] = n_starts.get(_attempt_key(r), 0) + 1
    for r in rows:
        n_rows[_attempt_key(r)] = n_rows.get(_attempt_key(r), 0) + 1
    unpaired = sorted(
        (k for k in set(n_starts) | set(n_rows) if n_starts.get(k, 0) != 1 or n_rows.get(k, 0) != 1), key=repr
    )
    if unpaired:
        return incomplete(
            "attempt_start_mismatch",
            f"(arm, seed, attempt) without exactly one attempt_start and one row: {unpaired}",
        )
    clean = {k: r for k, r in final.items() if r.get("refusal") is None}
    refused = [
        {"arm": a, "seed": s, "attempt": r.get("attempt"), "cause": r.get("refusal_cause"), "refusal": r["refusal"]}
        for (a, s), r in sorted(final.items())
        if r.get("refusal") is not None
    ]
    per_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in ARMS}
    for (arm, seed), r in sorted(clean.items()):
        fp = first_post(r)
        if fp is None or fp.get("dirty") or fp.get("never_submerged"):
            return incomplete(
                "unclean_first_post_in_clean_row", f"{arm}/seed{seed}: the first post placement is unclean"
            )
        per_arm[arm].append(
            {
                "seed": seed,
                "first_post_surfaced": bool(fp["surfaced"]),
                "first_post_latency_s": fp.get("latency_s"),
                "post_p_surface": p_surface(r["post"]),
                "pre_p_surface": p_surface(r["pre"]),
                "shore_roam_calls": len(((r["post"].get("shore_roam") or {}).get("calls")) or []),
                "training": (r.get("training_summary") or None),
                "key_coverage": r.get("key_coverage"),
            }
        )
    n_clean = {a: len(v) for a, v in per_arm.items()}
    rates = {a: (sum(s["first_post_surfaced"] for s in v) / len(v) if v else None) for a, v in per_arm.items()}
    out = {
        **base,
        "per_arm": per_arm,
        "n_clean": n_clean,
        "first_post_rate": rates,
        "refused": refused,
        "attempts": {f"{a}/{s}": len(v) for (a, s), v in sorted(by_key.items())},
        "apparatus_records": apparatus_records_used(rows),
    }
    need = gates["seeds_per_gated_arm"]
    if any(n_clean[a] != need for a in GATED_ARMS):
        return {
            **out,
            "verdict": "INCOMPLETE",
            "incomplete_cause": "too_few_clean_seeds",
            "reason": f"FEAR, RANDOM and Q_EXPLORE need exactly {need} clean seeds each: {n_clean}",
        }
    fear_hits = sum(s["first_post_surfaced"] for s in per_arm["fear"])
    if fear_hits < gates["fear_first_post_min"]:
        return {
            **out,
            "verdict": "INCOMPLETE",
            "incomplete_cause": "fear_sanity_gate",
            "reason": f"FEAR surfaced on {fear_hits}/{need} first post placements (< {gates['fear_first_post_min']}): "
            "a collapsed FEAR is not a floor match",
        }

    def dv(arm: str) -> list[float]:
        return [1.0 if s["first_post_surfaced"] else 0.0 for s in per_arm[arm]]

    tests = {
        "H1": exact_permutation_p(dv("fear"), dv("random")),
        "H2": exact_permutation_p(dv("fear"), dv("q_explore")),
    }
    shown = holm({h: t["p_one_sided"] for h, t in tests.items()}, gates["family_alpha"])
    verdict = (
        "FLOORS_BEATEN"
        if shown["H1"] and shown["H2"]
        else ("FLOOR_MATCHED_Q" if shown["H1"] else "FLOOR_MATCHED_CHANCE")
    )
    # Floor competence (prereg H2): a beaten floor that never tried escape in the key it read at the first post
    # placement is "floor under-explored", never "floor: beaten" (the verdict value itself stays frozen).
    under = sorted(
        s["seed"] for s in per_arm["q_explore"] if not (s.get("key_coverage") or {}).get("first_select_key_tried")
    )
    annotation = None
    if verdict == "FLOORS_BEATEN":
        annotation = "floor under-explored" if under else "floor: beaten"
    return {
        **out,
        "tests": tests,
        "holm_shown": shown,
        "verdict": verdict,
        "q_explore_under_explored_seeds": under,
        "annotation": annotation,
    }


def training_summary(row: dict[str, Any]) -> dict[str, Any] | None:
    """The reported training descriptives of one Q row (pure): the episode of first surfacing, the greedy water
    action after episodes 1, 2, 5 and 10, and executed actions to criterion (pre-probe + episodes through the first
    surfaced one; rule 4)."""
    training = row.get("training")
    if not isinstance(training, dict) or "episodes" not in training:
        return None
    episodes = training["episodes"]
    first = next((e["index"] for e in episodes if e["reason"] == "surfaced"), None)
    pre_calls = sum(len(p.get("calls") or []) for p in (row.get("pre") or {}).get("placements", []))
    pre_calls += len(((row.get("pre") or {}).get("shore_roam") or {}).get("calls") or [])
    upto = [e for e in episodes if first is None or e["index"] <= first]
    return {
        "first_surfaced_episode": first,
        "greedy_in_water_after": {
            str(n): episodes[n - 1].get("greedy_in_water") for n in (1, 2, 5, 10) if len(episodes) >= n
        },
        "executed_actions_to_criterion": pre_calls + sum(len(e["calls"]) for e in upto),
        "criterion_reached": first is not None,
        "reasons": [e["reason"] for e in episodes],
    }


def key_coverage(policy_record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Rule 13's diagnostic from a Q policy's picks (pure): the share of first-post-placement selects whose key had
    ``escape_water`` tried before the post probe, whether the first select's key had, and the distinct water keys."""
    if not policy_record or "picks" not in policy_record:
        return None
    picks = policy_record["picks"]
    if any("key" not in p for p in picks):
        return None  # RANDOM keeps no state keys
    tried = {  # before the post probe: the pre-probe placements and the training episodes (the learning happens there)
        json.dumps(p["key"])
        for p in picks
        if str(p.get("phase", "")).startswith(("pre-placement", "qtrain"))
        and str(p["action"]).endswith("_escape_water")
    }
    first = [p for p in picks if p.get("phase") == "post-placement-0"]
    return {
        "first_post_selects": len(first),
        "share_visited_with_escape_tried": (
            sum(1 for p in first if json.dumps(p["key"]) in tried) / len(first) if first else None
        ),
        "first_select_key_tried": (json.dumps(first[0]["key"]) in tried) if first else None,
        "distinct_water_keys": len({json.dumps(p["key"]) for p in picks if p.get("in_water")}),
    }


# ─────────────────────────────────── the trial ───────────────────────────────────


class PainCursor:
    """``pain_since()`` for the Q learner: the summed intensity of the allow-listed pain recorded since the previous
    call, read from the trial's own pain record (no second bus subscriber; each pain counted once; rule 2)."""

    def __init__(self, signals: list[dict[str, Any]]) -> None:
        self.signals = signals
        self.i = len(signals)

    def __call__(self) -> float:
        n = len(self.signals)
        new, self.i = self.signals[self.i : n], n
        return sum(s["intensity"] for s in new if s["failure_mode"] in PAIN_MODES)


class FloorWaterTrial(WaterTrial):
    """Exp 60's seed context with an armed floor policy (``policy`` None = FEAR, the loop's own NAc)."""

    def __init__(self, *, arm: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.arm = arm
        self.armed: FP.ArmedPolicy | None = None
        self.started: list[dict[str, Any]] = []  # every executor call at its START (credit for an in-flight call)
        self.ends: list[dict[str, Any]] = []  # every end_episode: label, reason, t
        self._arm_next: str | None = None  # the label to arm under at the next submerging teleport

    # ── instruments ──
    def attach_instruments(self) -> None:
        super().attach_instruments()
        spy, started = self.aut.executor.execute, self.started

        def _start_marker(action: dict[str, Any]) -> Any:
            started.append({"t": time.monotonic(), "tool": (action or {}).get("tool_name")})
            return spy(action)

        _start_marker._water_trial_spy = True  # type: ignore[attr-defined]  (the base's per-agent sentinel)
        self.aut.executor.execute = _start_marker

    def was_executed(self, action: str, since: float) -> bool:
        """Did the loop START executing ``action`` at or after ``since`` (rule 5)? A pick the same-tool cap
        suppressed, or one pending at stop, never started."""
        for s in reversed(self.started):
            if s["t"] < since:
                return False
            if s["tool"] == action:
                return True
        return False

    def set_policy(self, inner: Any) -> FP.ArmedPolicy:
        self.armed = FP.ArmedPolicy(inner)
        self.substrate_policy = self.armed
        return self.armed

    # ── arming and episode ends ──
    def before_teleport(self, label: str) -> None:
        if self._arm_next is not None and self.armed is not None:
            self.armed.arm(self._arm_next, water_only=True)

    def _end(self, label: str, reason: str) -> None:
        """Disarm, then ``end_episode`` (before the rescue teleport; the frozen order). The loop executes a pass's
        proposal on its NEXT pass, so at a pain terminal the newest pick may not have started yet; the harness waits
        two loop periods (still submerged, before the rescue) so it starts and is credited with the terminal it was
        chosen into, rather than dropped as never executed with the −1 booked to nothing."""
        if self.armed is not None:
            self.armed.disarm()
            if reason == "pain":
                time.sleep(2.0 / float(self.frozen["loop_hz"]))
            end = getattr(self.armed.inner, "end_episode", None)
            if end is not None:
                end(reason)
        self.ends.append({"label": label, "reason": reason, "t": time.monotonic()})

    def check_window(self, label: str, win: dict[str, Any] | None) -> None:
        """Rule 7: a policy fault or a loop thread that ended before `stop` refuses the arm-seed loudly."""
        if self.armed is not None and self.armed.all_violations():
            raise BehaviouralRefusal(f"{label}: policy fault: {self.armed.all_violations()}")
        if win is not None and win.get("loop_exited_early"):
            # policy faults are recorded as violations (above), so an early exit without one is the instrument's
            raise Refusal(f"{label}: the loop thread exited before it was told to stop")

    def check_death_cap(self) -> None:
        if self.deaths() - self.deaths0 > self.frozen["death_cap"]:
            raise BehaviouralRefusal(f"death cap exceeded ({self.deaths() - self.deaths0})")

    def placement(self, label: str, *, window_label: str | None = None, before_rescue: Any = None) -> dict[str, Any]:
        def _end_placement(samples: list[dict[str, Any]], t0: float) -> None:
            self._end(label, episode_end_reason(samples, pains=self.pain_between(t0, time.monotonic())))
            if before_rescue is not None:  # a caller's own callback still runs, after the episode end
                before_rescue(samples, t0)

        self._arm_next = label
        try:
            cls = super().placement(label, window_label=window_label, before_rescue=_end_placement)
        finally:
            self._arm_next = None
            if self.armed is not None:
                self.armed.disarm()
        self.check_window(label, self.last_window)
        cls["end_reason"] = self.ends[-1]["reason"] if self.ends and self.ends[-1]["label"] == label else None
        cls["t_first_water"] = next(
            (s["t"] for s in (self.last_window or {}).get("samples", []) if s["in_water"]), None
        )
        if self.armed is not None:
            cls["idle_dry"] = {"before_water": self.armed.idle_dry_before, "after_water": self.armed.idle_dry_after}
        return cls

    def probe(
        self, label: str, *, before_rescue: Any = None, roam_enter: Any = None, roam_before_rescue: Any = None
    ) -> dict[str, Any]:
        """Exp 60's probe, greedy (ε = 0, still updating online), with the shore roam ARMED."""
        if self.armed is not None and hasattr(self.armed.inner, "set_probe_mode"):
            self.armed.inner.set_probe_mode(True)

        def _arm_roam() -> float:
            assert self.armed is not None
            self.armed.arm(f"{label}-roam")
            return time.monotonic()

        armed = self.armed is not None
        out = super().probe(
            label,
            roam_enter=_arm_roam if armed else None,
            roam_before_rescue=(lambda samples, t0: self._end(f"{label}-roam", "cap")) if armed else None,
        )
        self.check_window(f"{label}-roam", self.last_roam)
        out["shore_roam"]["calls"] = (self.last_roam or {}).get("calls", [])
        return out

    def train_executed(self) -> dict[str, Any]:
        """Q arms: K executed episodes through real loop windows (rule 3). Terminals by bridge truth on the harness
        clock; cap = damage onset − 3 s (truncated); damage or any `drive:health` pain refuses (BehaviouralRefusal)."""
        assert self.armed is not None
        inner = self.armed.inner
        inner.set_probe_mode(False)
        fz = self.frozen
        episodes: list[dict[str, Any]] = []
        for i in range(1, fz["K_executed_episodes"] + 1):
            label = f"qtrain-{i}"
            self.rescue(label)
            state: dict[str, Any] = {"t0": None, "reason": None}

            def _enter(label: str = label, state: dict[str, Any] = state) -> float:
                state["t0"] = self.submerge(label)
                return state["t0"]

            def _on_sample(s: dict[str, Any], state: dict[str, Any] = state) -> bool:
                t0 = state["t0"]
                saturating = any(
                    sig["t"] >= t0
                    and sig["failure_mode"] == "drive:oxygen"
                    and sig["intensity"] >= fz["usable_pain_intensity_min"]
                    for sig in self.signals
                )
                state["reason"] = training_terminal(
                    s, state, oxygen_max=fz["usable_oxygen_max"], saturating_pain=saturating
                )
                return state["reason"] is not None

            def _before_rescue(samples: list[dict[str, Any]], t0: float, label: str = label, state=state) -> None:
                reason = state["reason"]
                if reason is None:
                    reason = "cap" if any(s["in_water"] for s in samples) else "never_submerged"
                self._end(label, "dirty" if reason == "damage" else reason)

            n_updates = len(getattr(inner, "updates", []))
            self._arm_next = label
            try:
                win = self.loop_window(
                    self.train_cap_s, enter=_enter, on_sample=_on_sample, label=label, before_rescue=_before_rescue
                )
            finally:
                self._arm_next = None
                self.armed.disarm()
            self.check_window(label, win)
            arrival_health = self.rescue(f"{label}-end")
            episode = {
                "index": i,
                "reason": self.ends[-1]["reason"],
                "terminal_sample": win["samples"][-1] if win["samples"] else None,
                "duration_s": round(win["t_rescue"] - win["t0"], 3),
                "calls": win["calls"],
                "pain_events": [{**p, "t": round(p["t"] - win["t0"], 3)} for p in win["us_events"]],
                "updates": len(getattr(inner, "updates", [])) - n_updates,
                "arrival_health": arrival_health,
                "greedy_in_water": greedy_in_water(inner),
            }
            episodes.append(episode)
            print(f"  q-train {i}: {episode['reason']} in {episode['duration_s']}s calls={len(win['calls'])}")
            if (
                arrival_health < 20.0
                or state["reason"] == "damage"
                or any(p["failure_mode"] == "drive:health" for p in win["us_events"])
            ):
                raise BehaviouralRefusal(
                    f"{label}: drowning DAMAGE during executed training (health {arrival_health})",
                    partial={"training": {"episodes": episodes}},
                )
            self.check_death_cap()
        return {"episodes": episodes, "deaths": self.deaths() - self.deaths0}


def greedy_in_water(inner: Any) -> dict[str, Any] | None:
    """The greedy action(s) in the most-visited water key so far (the learning-curve read)."""
    picks = [p for p in getattr(inner, "picks", []) if p.get("in_water") and "key" in p]
    if not picks:
        return None
    counts: dict[Any, int] = {}
    for p in picks:
        counts[p["key"]] = counts.get(p["key"], 0) + 1
    top = max(sorted(counts, key=repr), key=lambda k: counts[k])  # ties: the first in a stable order
    row = inner.q.get(top) or {}
    best = max(row.values()) if row else None
    return {
        "key": json.loads(json.dumps(top)),
        "best": sorted(a for a, v in row.items() if v == best) if row else None,
        "q": dict(row),
    }


def make_policy(arm: str, seed: int, trial: FloorWaterTrial) -> Any:
    """The arm's inner policy (None for FEAR), seeded per seed × arm."""
    if arm == "fear":
        return None
    rng_seed = policy_seed(arm, seed)
    if arm == "random":
        return FP.RandomPolicy(rng_seed)
    return FP.TabularQPolicy(
        seed=rng_seed,
        key_mode="raw" if arm == "q_explore_raw" else "clusters",
        prior_only=arm == "q_explore_prior",
        was_executed=trial.was_executed,
        pain_since=PainCursor(trial.signals),
    )


def policy_record(armed: FP.ArmedPolicy | None, arm: str, seed: int, t_ref: float) -> dict[str, Any] | None:
    """Everything the policy did, JSON-shaped, times relative to ``t_ref`` (the arm-seed's start)."""
    if armed is None:
        return None
    inner = armed.inner

    def rel(t: float) -> float:
        return round(t - t_ref, 3)

    rec: dict[str, Any] = {
        "class": type(inner).__name__,
        "rng_seed": policy_seed(arm, seed),
        "tool_sets": [list(t) for t in armed.tool_sets],
        "violations": armed.all_violations(),
        "picks": [
            {
                **{k: v for k, v in p.items() if k != "q"},
                "t": rel(p["t"]),
                **({"key": list(p["key"])} if "key" in p else {}),
            }
            for p in inner.picks
        ],
    }
    if isinstance(inner, FP.TabularQPolicy):
        rec.update(
            {
                "key_mode": inner.key_mode,
                "prior_only": inner.prior_only,
                "alpha": inner.alpha,
                "gamma": inner.gamma,
                "q_init": FP.Q_INIT,
                "actions": list(inner.actions or ()),
                "q_table": [{"key": list(k), "values": dict(v)} for k, v in inner.q.items()],
                "updates": [{**u, "t": rel(u["t"]), "key": list(u["key"])} for u in inner.updates],
                "drops": [{**d, "t": rel(d["t"]), "key": list(d["key"])} for d in inner.drops],
            }
        )
    return rec


def apparatus_geometry(anchor: Mapping[str, Any]) -> dict[str, Any]:
    """The classroom geometry rebuilt from the anchor's stamped build params, asserted equal to what it records."""
    from survival_world.setup_world import water_classroom_geometry

    sx, _sy, sz = anchor["shore"]
    geom = water_classroom_geometry(int(sx) + 1, int(sz), depth=int(anchor["depth"]), shore_y=int(anchor["surface_y"]))
    for k in ("shore", "submerged", "pool"):
        if [float(v) for v in geom[k]] != [float(v) for v in anchor[k]]:
            raise InstrumentError(f"the anchor's {k} {anchor[k]} is not the geometry its build params give ({geom[k]})")
    return geom


def reverify_apparatus(rcon: Any, geom: Mapping[str, Any], trial: WaterTrial) -> int:
    """The 10 `execute if block` checks of the built pool + the surface cell (before and after every arm-seed)."""
    from survival_world.setup_world import water_classroom_verifications

    checks = water_classroom_verifications(dict(geom))
    for cmd, proves in checks:
        resp = rcon.command(cmd)
        if "passed" not in str(resp).lower():
            raise InstrumentError(f"apparatus re-verify failed: {proves} — `{cmd}` -> {resp!r}")
    trial.check_surface_cell_air()
    return len(checks)


def food_state(rcon: Any, username: str, trial: WaterTrial) -> dict[str, Any]:
    """Food 20 (true value, over RCON) after the rescue's satiation, and the bread count (stamped)."""
    food = trial.read_food_state()
    if food["foodLevel"] < 20.0:
        trial.heal()
        food = trial.read_food_state()
        if food["foodLevel"] < 20.0:
            raise InstrumentError(f"food did not restore to 20 ({food})")
    bread = str(rcon.command(f"clear {username} minecraft:bread 0")).strip()
    return {**food, "bread": bread}


def rig_check(rcon: Any, proxy_url: str | None) -> dict[str, Any]:
    """Rule 11, recorded with no URL or auth in the data. Raises :class:`RigNotQuiet` when the rig is not quiet, saying
    whether waiting can clear it."""
    from maxim.utils import http

    status = last = None
    proxy = "not_checked" if not proxy_url else "absent"
    if proxy_url:
        from maxim.tunnel.keys import read_key
        from maxim.utils.net import is_loopback_url

        policy = http.TimeoutPolicy(connect_s=1.0, read_s=2.0, total_s=3.0)
        # The proxy's debug routes need the key, loopback included (2026-10-14); send this box's key only to a
        # loopback proxy, never to a remote URL.
        key = read_key() if is_loopback_url(proxy_url) else None
        auth = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            status = json.loads(
                http.fetch_url(proxy_url.rstrip("/") + "/v1/debug/status", headers=auth, timeout=policy).content
            )
            last = json.loads(
                http.fetch_url(proxy_url.rstrip("/") + "/v1/debug/last-requests", headers=auth, timeout=policy).content
            )
            proxy = "present"
        except http.HTTPConnectionError:
            status = last = None  # no proxy on this box: passes, recorded
        except (http.HTTPTimeout, http.HTTPServerError, http.HTTPRateLimited) as exc:
            # a busy leader: its debug read timed out or failed server-side, which waiting can clear
            raise RigNotQuiet(f"rig check: the leader proxy is busy ({type(exc).__name__})", transient=True) from exc
        except (http.HTTPError, ValueError) as exc:
            raise RigNotQuiet(
                f"rig check: the leader proxy answered but idle cannot be shown ({type(exc).__name__})",
                transient=False,
            ) from exc
    tps_reply = str(rcon.command("tick query"))
    tps = parse_tick_query(tps_reply)
    problems = rig_problems(status, last, tps, now=time.time())
    record = {
        "proxy": proxy,
        "in_flight": ((status or {}).get("infer_lane") or {}).get("in_flight") if status else None,
        "recent_requests": len((last or {}).get("recent") or []) if last else None,
        "tps": tps,
        "problems": [p.message for p in problems],
    }
    if problems:
        raise RigNotQuiet(f"rig check: {[p.message for p in problems]}", transient=all(p.transient for p in problems))
    return record


# ─────────────────────────────────── live ───────────────────────────────────


def _write_row(out_path: Path, record: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as fh:
        fh.write(json.dumps(stamp_harness_row(record, mock=False)) + "\n")


def _write_header(out_path: Path, record: dict[str, Any]) -> None:
    """A header line (the campaign's session row, or an attempt_start): never a counted run
    (``stamp_harness_header``)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as fh:
        fh.write(json.dumps(stamp_harness_header(record, mock=False)) + "\n")


class Session:
    """What every arm-seed of one campaign shares (resolved once, before any RCON or bridge connection)."""

    def __init__(self, args: argparse.Namespace, **kw: Any) -> None:
        self.args = args
        self.__dict__.update(kw)


def row_base(ses: Session, arm: str, seed: int, attempt: int) -> dict[str, Any]:
    """The fields every row of an arm-seed attempt carries (provenance, apparatus), before any outcome."""
    return {
        "kind": "row",
        "ts": time.time(),
        "campaign_id": ses.campaign_id,
        "run_id": ses.run_ids[arm],
        "arm": arm,
        "seed": seed,
        "attempt": attempt,
        "probe_cap_s": ses.probe_cap_s,
        "train_cap_s": ses.train_caps[arm],
        "provenance": ses.provenance,
        "apparatus_record": ses.apparatus_path,
        "apparatus_record_sha256": ses.apparatus_sha256,
        "apparatus_record_ts": ses.apparatus.get("ts"),
        "apparatus_record_check": ses.apparatus_check,
        "refusal": None,
        "refusal_cause": None,
    }


def run_arm_seed(ses: Session, arm: str, seed: int, attempt: int) -> dict[str, Any]:
    """One arm-seed attempt on a fresh agent. Returns its row (a refusal carries ``refusal`` + ``refusal_cause``)."""
    from maxim.simulation.minecraft import MinecraftClient
    from maxim.simulation.minecraft_harness import MinecraftSyncPump, build_minecraft_aut
    from survival_world.common import make_fresh_encoder

    args = ses.args
    t_ref = time.monotonic()
    persistence_dir = tempfile.mkdtemp(prefix=f"b1_{arm}_{seed}_a{attempt}_")
    agent_id = f"b1_{arm}_s{seed}_a{attempt}"
    record = row_base(ses, arm, seed, attempt)
    step = "rig"
    aut: Any = None
    pump: Any = None
    trial: FloorWaterTrial | None = None
    prev_cwd = os.getcwd()
    try:
        # wait (bounded) for a quiet rig BEFORE an agent exists: a recent proxy request would otherwise refuse this
        # attempt and its immediate rerun inside the same 2-minute window (rule 11)
        record["rig"] = wait_for_quiet_rig(ses.rcon, args.proxy_url)
        step = "connect"
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
        os.chdir(persistence_dir)
        trial = FloorWaterTrial(
            arm=arm,
            aut=aut,
            rcon=ses.rcon,
            username=args.username,
            geom=ses.geom,
            frozen=FROZEN,
            probe_cap_s=ses.probe_cap_s,
            train_cap_s=ses.train_caps[arm],
            persistence_dir=persistence_dir,
            agent_id=agent_id,
            encoder=encoder,
        )
        trial.attach_instruments()
        inner = make_policy(arm, seed, trial)
        if inner is not None:
            trial.set_policy(inner)
        step = "apparatus"
        record["apparatus_checks_before"] = reverify_apparatus(ses.rcon, ses.build_geom, trial)
        step = "preflight"
        # The Exp 60 per-seed preflight sequence, copied in its order (exp60_run._run; a test pins the order).
        record["fingerprint_live"] = trial.check_fingerprint(FROZEN["usable_oxygen_max"])
        record["bridge_state_interval_s"] = trial.check_bridge()
        record["loop_liveness_ticks"] = trial.check_liveness()
        trial.check_gamerules()
        shore_cluster_pre, water_cluster_pre = trial.check_clusters_distinct()
        trial.resolve_tools()
        record["actuation_preflight"] = trial.check_escape_actuation()
        trial.check_no_positive_escape_link()
        trial.deaths0 = trial.deaths()
        record["food"] = food_state(ses.rcon, args.username, trial)

        step = "pre_probe"
        pre = trial.probe("pre")
        record["pre"] = pre
        step = "us_free_check"
        trial.submerge("post-pre-check")
        water_cluster_after_pre = trial.encode_world_cluster()
        trial.rescue("post-pre-check")
        record["water_fear_pre"] = round(aut.bio.nac.cluster_fear(agent_id, water_cluster_after_pre), 4)
        if record["water_fear_pre"] != 0.0 or pre["unyoked_us_events"]:
            raise Refusal(
                f"pre-probe was NOT US-free: water fear {record['water_fear_pre']}, unyoked US events "
                f"{pre['unyoked_us_events']}"
            )

        step = "training"
        if arm == "fear":
            try:
                record["training"], episode_clusters = trial.train()
                record.update(trial.live_g2("fear", episode_clusters, water_cluster_pre))
            except Refusal as exc:  # FEAR's training and G2 are never rerun (rule 6, strict)
                raise BehaviouralRefusal(str(exc), partial=getattr(exc, "partial", None)) from exc
        elif arm in Q_ARMS:
            record["training"] = trial.train_executed()
        else:
            record["training"] = None

        step = "post_probe"
        post = trial.probe("post")
        record["post"] = post
        first = post["placements"][0]
        if (
            first.get("surfaced")
            and first.get("t_first_water") is not None
            and any(
                str(c["tool"]).endswith("_escape_water") and c["t"] < first["t_first_water"] for c in first["calls"]
            )
        ):
            # rule 17's FEAR half, checked: the first post placement must be won from the water, not launched on the
            # shore before the body read water (the floors cannot, by their gate)
            raise BehaviouralRefusal(
                "the first post placement's escape started before the first in-water read — a shore-launched act"
            )
        if first.get("dirty") or first.get("never_submerged"):
            # damage (a health drop, no pain needed) is behavioural; a pain inside the US-free window or a placement
            # that never submerged is the apparatus's (the probe cap or the teleport), rerun once
            arrival = first.get("arrival_health")
            damaged = (arrival is not None and arrival < 20.0) or (
                first.get("end_reason") == "dirty" and not first["us_events"]
            )
            msg = f"the first post placement is not clean ({first.get('end_reason')}) — refused, never substituted"
            raise BehaviouralRefusal(msg) if damaged else Refusal(msg)
        record["positive_escape_links"] = trial.positive_escape_links()
        record.update(trial.negative_links())
        record["shore_cluster_pre"], record["water_cluster_pre"] = shore_cluster_pre, water_cluster_pre
        step = "apparatus_after"
        record["apparatus_checks_after"] = reverify_apparatus(ses.rcon, ses.build_geom, trial)
        print(
            f"{arm} seed {seed}: first post {'SURFACED' if first['surfaced'] else 'censored'}; "
            f"post P(surface)={post['p_surface']}"
        )
    except BehaviouralRefusal as exc:
        record.update(getattr(exc, "partial", None) or {})
        record["refusal"], record["refusal_cause"], record["refusal_step"] = str(exc), "behavioural", step
        print(f"REFUSED (behavioural) {arm} seed {seed}: {exc}")
    except (Refusal, InstrumentError) as exc:
        record.update(getattr(exc, "partial", None) or {})
        record["refusal"], record["refusal_cause"], record["refusal_step"] = str(exc), "instrument", step
        if isinstance(exc, RigNotQuiet):
            record["rig_refusal_transient"] = exc.transient
        print(f"REFUSED (instrument) {arm} seed {seed}: {exc}")
    except OSError as exc:  # a transport fault (RCON socket, bridge ConnectionError, a timeout) on a 3-hour run: an
        # instrument refusal, recorded. A programming error is NOT caught: it crashes the campaign (--resume exists).
        record["refusal"], record["refusal_cause"], record["refusal_step"] = (
            f"{type(exc).__name__}: {exc}",
            "instrument",
            step,
        )
        record["refusal_exception"] = type(exc).__name__
        print(f"REFUSED (instrument, {type(exc).__name__}) {arm} seed {seed}: {exc}")
    finally:
        try:
            if trial is not None:
                record["policy"] = policy_record(trial.armed, arm, seed, t_ref)
                record["episode_ends"] = [{**e, "t": round(e["t"] - t_ref, 3)} for e in trial.ends]
                record["calls_all"] = [
                    {**c, "t": round(c["t"] - t_ref, 3), "t_return": round(c["t_return"] - t_ref, 3)}
                    for c in trial.calls
                ]
            record["training_summary"] = training_summary(record)
            record["key_coverage"] = key_coverage(record.get("policy"))
        finally:  # the row's derived fields never skip the teardown
            _teardown(trial, pump, aut)
            os.chdir(prev_cwd)
            shutil.rmtree(persistence_dir, ignore_errors=True)
    return record


def _teardown(trial: FloorWaterTrial | None, pump: Any, aut: Any) -> None:
    if trial is not None:
        trial.detach_instruments()
        trial.final_rescue()
    if pump is not None:
        try:
            pump.stop()
        except Exception as exc:
            print(f"WARNING: pump stop raised: {exc!r}")
    if aut is not None:
        try:
            aut.bio.on_session_end()
        except Exception as exc:
            print(f"WARNING: bio teardown raised: {exc!r}")
        try:
            aut.client.close()
        except (OSError, ConnectionError) as exc:
            print(f"WARNING: client close raised: {exc!r}")


def wait_for_quiet_rig(
    rcon: Any, proxy_url: str | None, *, max_wait_s: float = 600.0, poll_s: float = 15.0
) -> dict[str, Any]:
    """``rig_check`` until it passes or ``max_wait_s`` elapses (then its refusal stands). Records the wait. A refusal
    that waiting cannot clear (``RigNotQuiet.transient`` False) stands at once."""
    t0 = time.monotonic()
    while True:
        try:
            rec = rig_check(rcon, proxy_url)
            return {**rec, "waited_s": round(time.monotonic() - t0, 1)}
        except RigNotQuiet as exc:
            if not exc.transient or time.monotonic() - t0 + poll_s > max_wait_s:
                raise
            time.sleep(poll_s)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _run(args: argparse.Namespace) -> int:
    if os.environ.get("MAXIM_SUBSTRATE_TOOL_WHITELIST"):
        print("[FAIL] MAXIM_SUBSTRATE_TOOL_WHITELIST is set — every arm must see the same unfiltered repertoire")
        return 3
    arms = tuple(a.strip() for a in args.arms.split(",")) if args.arms else ARMS
    if set(arms) - set(ARMS):
        print(f"[FAIL] unknown arm(s): {sorted(set(arms) - set(ARMS))}")
        return 2
    seeds: tuple[int, ...] = (DIAGNOSTIC_SEED,) if args.diagnostic else SEEDS
    prior_rows: list[dict[str, Any]] = []
    if args.resume:
        resume_path = Path(args.resume)
        prior_rows = _read_rows(resume_path if resume_path.is_absolute() else C.REPO_ROOT / resume_path)
        session_rows = [r for r in prior_rows if r.get("kind") == "session"]
        if len(session_rows) != 1:
            print(f"[FAIL] --resume: {len(session_rows)} session rows in the file (need 1)")
            return 3
        campaign_id, run_ids = session_rows[0]["campaign_id"], dict(session_rows[0]["run_ids"])
        if bool(session_rows[0].get("diagnostic")) != bool(args.diagnostic):
            print("[FAIL] --resume: --diagnostic must match the campaign's session row (never pool seed 99)")
            return 3
        out_arg = resume_path
    else:
        campaign_id = ("diag-" if args.diagnostic else "") + uuid.uuid4().hex[:12]
        run_ids = {a: uuid.uuid4().hex[:12] for a in ARMS}
        out_arg = Path(args.out or campaign_out_path(DATA_STEM, campaign_id))
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
        print(f"[FAIL] refusing to append: {refusal}")
        return 2

    # ── the gated records (Exp 60's rules, its pure helpers) ──
    try:
        apparatus_used = load_apparatus_record(args.apparatus_record)
    except ApparatusRecordError as exc:
        print(f"[FAIL] {exc}")
        return 3
    apparatus = apparatus_used["record"]
    gated = is_gated_path(C.REPO_ROOT, out_path)
    record_problem = (
        committed_record_problem(C.REPO_ROOT, apparatus_used["abs_path"], apparatus_used["bytes"]) if gated else None
    )
    if record_problem is not None and not args.allow_dirty:
        print(f"[FAIL] apparatus record {apparatus_used['path']} is not committed on main: {record_problem}")
        return 3
    apparatus_check, provenance = apparatus_check_stamp(gated, record_problem, provenance)  # both stamps, one builder
    refusal = instrument_check_authorizes(apparatus)
    if refusal is not None:
        print(f"[FAIL] apparatus record {apparatus_used['path']} does not authorize a run: {refusal}")
        return 3
    try:
        gate = json.loads((C.REPO_ROOT / args.gate_record).read_text())
    except (OSError, ValueError) as exc:
        print(f"[FAIL] geometry gate record unreadable: {args.gate_record} ({exc})")
        return 3
    if not (gate.get("run_gate") or {}).get("pass"):
        print(f"[FAIL] geometry gate record does not carry run_gate.pass=true: {args.gate_record}")
        return 3
    if gated and not args.diagnostic and not args.resume:
        day_problem = apparatus_day_problem(apparatus.get("ts"), time.time())
        if day_problem is not None:
            print(f"[FAIL] {day_problem}")
            return 3
    pain_edge_min = min_pain_edge_s(apparatus)
    try:
        geom = json.loads(ANCHOR_FILE.read_text())
        onset_min = float(geom["measured"]["t_damage_onset_min_s"])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"[FAIL] anchor {ANCHOR_FILE} lacks the check's stamped measured onset ({exc})")
        return 3
    if pain_edge_min is None:
        print("[FAIL] apparatus record carries no measured t_pain_edge — the probe cap cannot be set")
        return 3
    mismatch = anchor_record_mismatch(geom.get("measured"), apparatus)
    if mismatch is not None:
        print(f"[FAIL] anchor and apparatus record disagree: {mismatch}")
        return 3
    try:
        build_geom = apparatus_geometry(geom)
    except InstrumentError as exc:
        print(f"[FAIL] {exc}")
        return 3
    probe_cap_s = pain_edge_min - FROZEN["probe_cap_margin_s"]
    fear_cap = onset_min - FROZEN["train_cap_margin_s"]
    q_cap = onset_min - FROZEN["q_train_cap_margin_s"]
    train_caps = {a: (fear_cap if a == "fear" else q_cap if a in Q_ARMS else 0.0) for a in ARMS}

    rcon = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    ses = Session(
        args,
        campaign_id=campaign_id,
        run_ids=run_ids,
        provenance=provenance,
        apparatus=apparatus,
        apparatus_path=apparatus_used["path"],
        apparatus_sha256=apparatus_used["sha256"],
        apparatus_check=apparatus_check,
        geom=geom,
        build_geom=build_geom,
        probe_cap_s=probe_cap_s,
        train_caps=train_caps,
        rcon=rcon,
    )
    print(
        f"campaign {campaign_id}: arms={arms} seeds={seeds} probe cap={probe_cap_s:.2f}s caps={train_caps} -> {out_path}"
    )
    exit_code = 0
    try:
        # A rig problem waiting cannot clear is a property of the CAMPAIGN, not of one key: refuse it here, before any
        # attempt is spent (code review: otherwise every key's attempt and its rerun refuse in seconds and the
        # campaign ends unresumable).
        try:
            wait_for_quiet_rig(rcon, args.proxy_url)
        except RigNotQuiet as exc:
            print(f"[FAIL] {exc} -- fix the rig (rule 11: run B1 with the leader down), then start the campaign")
            return 3
        except OSError as exc:  # RCON unreachable before anything was written: a clean refusal, not a traceback
            print(f"[FAIL] rig check: RCON failed ({type(exc).__name__}: {exc})")
            return 3
        if not args.resume:
            _write_header(
                out_path,
                {
                    "kind": "session",
                    "ts": time.time(),
                    "campaign_id": campaign_id,
                    "run_ids": run_ids,
                    "diagnostic": bool(args.diagnostic),
                    "frozen": FROZEN,
                    "gates": GATES,
                    "provenance": provenance,
                    "apparatus_record": apparatus_used["path"],
                    "apparatus_record_sha256": apparatus_used["sha256"],
                    "apparatus_record_ts": apparatus.get("ts"),
                    "anchor_measured": geom.get("measured"),
                    "gate_record_code_hash": gate.get("code_hash"),
                    "probe_cap_s": probe_cap_s,
                    "train_caps": train_caps,
                },
            )
        # Resume: close every orphan start (an attempt the process died inside) with an "unrecorded" row, AFTER the
        # provenance stamp and append_refusal passed (a changed tree is refused first, leaving its orphan to fail the
        # file), and BEFORE attempts_done reads the rows — so that key is consumed, never rerun (D2).
        for closing in closing_rows(prior_rows, lambda a, s, n: row_base(ses, a, s, n)):
            _write_row(out_path, closing)
            prior_rows.append(closing)
            exit_code = 4
            print(
                f"[resume] closed an unrecorded attempt: {closing['arm']} seed {closing['seed']} #{closing['attempt']}"
            )
        done = attempts_done(prior_rows)
        for seed in seeds:
            for arm in ORDER[seed]:
                if arm not in arms:
                    continue
                entry = done.get((arm, seed), {"attempts": 0, "done": False})
                attempt = entry["attempts"]
                while not entry["done"]:
                    attempt += 1
                    print(f"\n=== seed={seed} arm={arm} attempt={attempt} ===")
                    _write_header(
                        out_path,
                        attempt_start_record(campaign_id, run_ids[arm], arm, seed, attempt, provenance=provenance),
                    )
                    rec = run_arm_seed(ses, arm, seed, attempt)
                    _write_row(out_path, rec)
                    if rec.get("rig_refusal_transient") is False:
                        # the rig went bad mid-campaign in a way waiting cannot clear: stop, never spend the rerun;
                        # `--resume` gets it back once the rig is fixed (the refusal is an instrument cause)
                        print(
                            f"[FAIL] the rig check refused mid-campaign ({rec['refusal']}); stopping. Resume once fixed."
                        )
                        return 3
                    if rec["refusal"] is not None:
                        exit_code = 4
                    entry = _after_attempt(rec, attempt)
    finally:
        rcon.close()
    print(f"\ncampaign {campaign_id} -> {out_path}")
    return exit_code


def _after_attempt(rec: dict[str, Any], attempt: int) -> dict[str, Any]:
    """Whether this key is done after ``rec`` (the ``attempts_done`` rule for one key)."""
    return {"attempts": attempt, "done": key_done(rec, attempt)}


def _verdict(args: argparse.Namespace) -> int:
    data = Path(args.data).expanduser()
    data_bytes = data.read_bytes()
    recs = [json.loads(ln) for ln in data_bytes.decode().splitlines() if ln.strip()]
    v = compute_verdict(recs)
    stamp_verdict(
        v,
        repo_root=C.REPO_ROOT,
        kind="b1_verdict",
        data=data,
        data_bytes=data_bytes,
        mock=any_not_stamped_real(recs),
        scope={"all_rows": True},
    )
    print(json.dumps({k: v[k] for k in v if k not in ("per_arm", "gates")}, indent=2, default=str))
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
    r = sub.add_parser("run", help="LIVE: every arm × seed in the frozen order (one campaign)")
    r.add_argument("--arms", default=None, help="comma-separated subset (diagnostics); default all five")
    r.add_argument("--diagnostic", action="store_true", help=f"seed {DIAGNOSTIC_SEED} only (never pooled)")
    # A static help string: campaign_out_path validates its id, so a placeholder like "<campaign_id>" raised here and
    # crashed every invocation before any argument was read (rig, 2026-10-12).
    r.add_argument(
        "--out",
        default=None,
        help=f"default docs/experiments/data/{DATA_STEM}_<campaign_id>.jsonl (one campaign per file)",
    )
    r.add_argument("--resume", default=None, help="continue the campaign in this rows file (same code tree)")
    r.add_argument("--apparatus-record", default=APPARATUS_RECORD)
    r.add_argument("--gate-record", default=GATE_RECORD)
    r.add_argument("--proxy-url", default="http://127.0.0.1:8099", help="the leader proxy on this box ('' = skip)")
    r.add_argument("--bridge-host", default="127.0.0.1")
    r.add_argument("--bridge-port", type=int, default=25567)
    r.add_argument("--rcon-host", default="127.0.0.1")
    r.add_argument("--rcon-port", type=int, default=25575)
    r.add_argument("--rcon-password", required=True)
    r.add_argument("--username", default="maxim")
    r.add_argument("--write-experiment-results", action="store_true")
    r.add_argument("--allow-dirty", action="store_true")
    r.set_defaults(func=_run)
    v = sub.add_parser("verdict", help="OFFLINE: the prereg's verdict from one campaign's rows")
    v.add_argument("--data", required=True)
    v.add_argument("--json", default=None)
    v.add_argument("--write-experiment-results", action="store_true")
    v.add_argument("--allow-dirty", action="store_true")
    v.set_defaults(func=_verdict)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
