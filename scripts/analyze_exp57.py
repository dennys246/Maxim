#!/usr/bin/env python3
"""Exp 57 verdict analyzer — the frozen gates, one place.

Pre-registration:
``docs/experiments/protocols/exp57_dose_response_ladder_preregistration.md``.
The GATE constants are FROZEN here at the harness-merge commit and are
extended, never retuned (house convention). The APPARATUS constants (K_max, C,
W) are read from the rows (set by the Phase-0 amendment). The analyzer:

* refuses a verdict on mock rows or dirty-tree rows (apparatus, not data);
* MONOTONICITY — the per-cohort tau(creche(N)) values fall across N by a
  hand-rolled Jonckheere-Terpstra decreasing-trend statistic with a PERMUTATION
  null (scipy has no jonckheere), PLUS two censoring-artifact guards: (i) the
  trend survives dropping fully-censored rungs, and (ii) uncensored endpoint
  coverage d(N, K_max) rises with N (positive Spearman);
* NOT-JUST-MORE-DATA — per rung N>=2, N*tau(creche) <= tau(single_matched) +
  delta_eff (delta_eff = 0);
* NOISE-FLOOR — creche_none never reaches C at any rung;
* ``--assert-noop-fails`` — re-runs the kept cohort-0 artifacts through the
  no-op merge variants (the D62 kit generalized to coverage) and refuses a
  verdict if the two must-collapse variants do not collapse.

Exit 0 PASS / 4 NO-VERDICT / 1 FAIL.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

# ── FROZEN gate constants (gate v1) ──────────────────────────────────────
GATES_V1 = {
    "delta_eff": 0,  # NOT-JUST-MORE-DATA margin — frozen HERE, never fit to pilot data
    "p_threshold": 0.05,  # one-sided JT permutation p
    "rungs": (1, 2, 4, 8),
    "cohorts_min": 20,
    "permutations": 10_000,
    "endpoint_spearman_min": 0.0,  # guard (ii): endpoint coverage must RISE with N (rho > 0)
    # L2 gate (prereg §check 4 / §Gates / outcome tree): the isolated N=1
    # coverage must NOT be seed-invariant across cohorts, or effective-n has
    # collapsed (the varying contributor seeds are not picking different covered
    # sets). Concentration = fraction of cohorts sharing the modal N=1 endpoint
    # coverage value; above this bound the arm is seed-invariant -> NO-VERDICT.
    # Mirrors exp56's 0.90 choice-concentration L2 gate.
    "l2_concentration_max": 0.90,
}


# ── Jonckheere-Terpstra (decreasing alternative), hand-rolled ────────────


def _jt_decreasing_statistic(groups: list[list[float]]) -> float:
    """Jonckheere-Terpstra statistic for the DECREASING ordered alternative.

    Groups are given in ASCENDING order of the ordering variable (rung N).
    For each ordered pair of groups (i < j) we count, over all (x in g_i,
    y in g_j), how often y < x (with ties at 0.5) — so a LARGE J means later
    (higher-N) groups hold SMALLER values, i.e. tau decreases as N grows.
    scipy has no jonckheere, so this is the hand-rolled form the prereg pins.
    """
    j = 0.0
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            for x in groups[a]:
                for y in groups[b]:
                    if y < x:
                        j += 1.0
                    elif y == x:
                        j += 0.5
    return j


def jt_permutation_test(groups: list[list[float]], *, permutations: int, seed: int = 20260907) -> dict:
    """One-sided permutation p for the JT decreasing statistic.

    The null shuffles the rung labels (pools all values, re-partitions into the
    same group sizes) ``permutations`` times — the frozen form, because an
    entire rung is expected tied at the censoring sentinel and the normal
    approximation degrades there (prereg §Gates).
    """
    sizes = [len(g) for g in groups]
    pooled = np.array([v for g in groups for v in g], dtype=float)
    obs = _jt_decreasing_statistic(groups)
    if len(pooled) == 0 or any(s == 0 for s in sizes):
        return {"statistic": obs, "p_value": 1.0, "permutations": 0}
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(permutations):
        shuffled = pooled.copy()
        rng.shuffle(shuffled)
        perm_groups: list[list[float]] = []
        idx = 0
        for s in sizes:
            perm_groups.append(list(shuffled[idx : idx + s]))
            idx += s
        if _jt_decreasing_statistic(perm_groups) >= obs:
            ge += 1
    # +1 in numerator and denominator: the observed arrangement is itself a
    # valid permutation (never report p = 0).
    p = (ge + 1) / (permutations + 1)
    return {"statistic": obs, "p_value": p, "permutations": permutations}


def _spearman(xs: list[float], ys: list[float]) -> dict:
    """Spearman rho + two-sided p (scipy).

    A FLAT endpoint series (a legitimate gate case — guard (ii) failing) makes
    the correlation undefined; scipy warns and returns NaN, which we map to
    rho 0.0 (guard (ii) then correctly fails). The warning is expected on that
    path, so it is silenced rather than surfaced as noise.
    """
    import warnings

    from scipy.stats import spearmanr

    if len(xs) < 2:
        return {"rho": 0.0, "p_value": 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = spearmanr(xs, ys)
    rho = 0.0 if rho != rho else float(rho)  # NaN guard (constant input)
    p = 1.0 if p != p else float(p)
    return {"rho": rho, "p_value": p}


# ── row aggregation ──────────────────────────────────────────────────────


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Collapse per-checkpoint rows into per-(rung, condition, cohort) tau and
    per-(rung, cohort) creche endpoint coverage."""
    # tau is constant across a (cohort, rung, condition)'s rows.
    tau: dict[tuple[int, str], dict[int, int]] = {}
    endpoint_creche: dict[int, dict[int, float]] = {}
    creche_none_max_cov = 0.0
    creche_none_by_rung: dict[int, float] = {}  # per-rung max coverage (NOISE-FLOOR, esp N=8)
    for r in rows:
        rung = int(r["rung"])
        cond = str(r["condition"])
        cohort = int(r["cohort"])
        tau.setdefault((rung, cond), {})[cohort] = int(r["tau"])
        if cond == "creche" and int(r["t"]) == int(r["k_max"]):
            endpoint_creche.setdefault(rung, {})[cohort] = float(r["coverage"])
        if cond == "creche_none":
            cov = float(r["coverage"])
            creche_none_max_cov = max(creche_none_max_cov, cov)
            creche_none_by_rung[rung] = max(creche_none_by_rung.get(rung, 0.0), cov)
    return {
        "tau": tau,
        "endpoint_creche": endpoint_creche,
        "creche_none_max_cov": creche_none_max_cov,
        "creche_none_by_rung": creche_none_by_rung,
    }


# ── the gates ────────────────────────────────────────────────────────────


def analyze(rows: list[dict], *, cohorts_min: int, permutations: int) -> dict:
    problems: list[str] = []
    for r in rows:
        if r.get("mock"):
            problems.append("mock rows present — the ScriptedBridge smoke is never a confirmatory record")
            break
    for r in rows:
        if r.get("working_tree_dirty_src_scripts") and not r.get("allow_dirty"):
            problems.append("dirty-tree rows without allow_dirty — gated-record contract violated")
            break

    if not rows:
        return {"verdict": "NO-VERDICT", "problems": ["no rows"], "gates": {}, "constants": GATES_V1}

    criterion = float(rows[0].get("criterion", 0.75))
    rungs = list(GATES_V1["rungs"])
    agg = aggregate(rows)
    tau = agg["tau"]

    # Per-cohort counts (power) — EVERY condition a gate consumes must have
    # >= cohorts_min, not creche alone (methodology-lens finding 5): otherwise
    # NOT-JUST-MORE-DATA / NOISE-FLOOR could rest on a handful of cohorts while
    # creche has 20. single_matched only exists for N>=2; creche_none at all rungs.
    stats: dict[str, dict] = {}
    for rung in rungs:
        creche = tau.get((rung, "creche"), {})
        single = tau.get((rung, "single_matched"), {})
        none_ = tau.get((rung, "creche_none"), {})
        n = len(creche)
        stats[f"rung_{rung}"] = {
            "n_cohorts": n,
            "n_single_matched": len(single),
            "n_creche_none": len(none_),
            "creche_tau_median": statistics.median(creche.values()) if creche else None,
            "single_matched_tau_median": statistics.median(single.values()) if single else None,
        }
        if n < cohorts_min:
            problems.append(f"rung {rung}: creche n_cohorts={n} < {cohorts_min} (frozen power)")
        if rung >= 2 and len(single) < cohorts_min:
            problems.append(f"rung {rung}: single_matched n_cohorts={len(single)} < {cohorts_min} (frozen power)")
        if len(none_) < cohorts_min:
            problems.append(f"rung {rung}: creche_none n_cohorts={len(none_)} < {cohorts_min} (frozen power)")

    gates: dict[str, bool | None] = {}
    details: dict[str, dict] = {}

    # L2 gate (prereg §check 4 / §Gates / outcome tree; methodology-lens
    # finding 2) — the isolated N=1 endpoint coverage must NOT be seed-invariant
    # across cohorts, or effective-n has collapsed. Applied at N=1 (the isolated
    # arm): higher-N saturation to the same coverage is the EFFECT, not a
    # seed-invariance failure, so the gate targets the single-contributor rung
    # the prereg names. Missing N=1 endpoint data -> cannot clear the gate.
    n1_endpoint = agg["endpoint_creche"].get(1, {})
    if not n1_endpoint:
        gates["L2_SEED_VARIANCE"] = None
        problems.append("L2 gate: no N=1 endpoint coverage rows — cannot verify seed-variance")
        details["L2_SEED_VARIANCE"] = {"error": "no N=1 endpoint rows"}
    else:
        vals = list(n1_endpoint.values())
        modal_count = max(vals.count(v) for v in set(vals))
        concentration = modal_count / len(vals)
        l2_ok = concentration <= GATES_V1["l2_concentration_max"]
        gates["L2_SEED_VARIANCE"] = bool(l2_ok)
        details["L2_SEED_VARIANCE"] = {
            "n1_concentration": round(concentration, 4),
            "max": GATES_V1["l2_concentration_max"],
            "distinct_values": sorted(set(vals)),
            "n_cohorts": len(vals),
        }
        if not l2_ok:
            problems.append(
                f"L2 gate: N=1 endpoint coverage is seed-invariant (concentration "
                f"{concentration:.2f} > {GATES_V1['l2_concentration_max']}) — effective-n collapsed"
            )

    # MONOTONICITY — JT decreasing + two censoring-artifact guards.
    creche_groups = [list(tau.get((rung, "creche"), {}).values()) for rung in rungs]
    have_all = all(len(g) > 0 for g in creche_groups)
    if have_all:
        jt = jt_permutation_test([[float(v) for v in g] for g in creche_groups], permutations=permutations)
        pass_jt = jt["p_value"] < GATES_V1["p_threshold"]

        # Guard (i): drop fully-censored rungs (all tau at the sentinel), re-test.
        k_max_by_rung = {
            rung: int(next(r["k_max"] for r in rows if int(r["rung"]) == rung and r["condition"] == "creche"))
            for rung in rungs
        }
        kept = [
            (rung, g) for rung, g in zip(rungs, creche_groups) if not all(int(v) >= k_max_by_rung[rung] + 1 for v in g)
        ]
        if len(kept) >= 2:
            jt_dropped = jt_permutation_test([[float(v) for v in g] for _rung, g in kept], permutations=permutations)
            guard_i = jt_dropped["p_value"] < GATES_V1["p_threshold"]
        else:
            jt_dropped = {"p_value": 1.0, "note": "fewer than 2 uncensored rungs"}
            guard_i = False

        # Guard (ii): endpoint coverage rises with N (positive Spearman).
        endpoint = agg["endpoint_creche"]
        endpoint_medians = [
            statistics.median(endpoint.get(rung, {}).values()) if endpoint.get(rung) else 0.0 for rung in rungs
        ]
        sp = _spearman([float(r) for r in rungs], endpoint_medians)
        guard_ii = sp["rho"] > GATES_V1["endpoint_spearman_min"]

        gates["MONOTONICITY"] = bool(pass_jt and guard_i and guard_ii)
        details["MONOTONICITY"] = {
            "jt": jt,
            "jt_drop_censored": jt_dropped,
            "guard_i_drop_censored": guard_i,
            "endpoint_spearman": sp,
            "endpoint_medians": endpoint_medians,
            "guard_ii_endpoint_rises": guard_ii,
            "pass_jt": pass_jt,
        }
    else:
        problems.append("missing creche rows for one or more rungs — MONOTONICITY undefined")
        gates["MONOTONICITY"] = None

    # NOT-JUST-MORE-DATA — per rung N>=2: N*tau(creche) <= tau(single) + delta.
    njmd_pass = True
    njmd_detail: dict[str, dict] = {}
    for rung in rungs:
        if rung < 2:
            continue
        creche = tau.get((rung, "creche"), {})
        single = tau.get((rung, "single_matched"), {})
        if not creche or not single:
            # A missing condition for a rung is an apparatus/data-completeness
            # failure, not a NOT-JUST-MORE-DATA verdict — refuse the verdict
            # (NO-VERDICT) rather than silently reporting PARTIAL (code-lens
            # review, finding 3).
            problems.append(f"rung {rung}: missing creche or single_matched rows — cannot evaluate NOT-JUST-MORE-DATA")
            njmd_pass = False
            njmd_detail[f"rung_{rung}"] = {"error": "missing creche or single_matched"}
            continue
        creche_med = statistics.median(creche.values())
        single_med = statistics.median(single.values())
        ok = rung * creche_med <= single_med + GATES_V1["delta_eff"]
        njmd_detail[f"rung_{rung}"] = {
            "N_times_creche_tau": rung * creche_med,
            "single_matched_tau": single_med,
            "ok": ok,
        }
        njmd_pass = njmd_pass and ok
    gates["NOT_JUST_MORE_DATA"] = njmd_pass if any(r >= 2 for r in rungs) else None
    details["NOT_JUST_MORE_DATA"] = njmd_detail

    # NOISE-FLOOR — creche_none never reaches C at any rung. The arm must be
    # PRESENT at every rung, especially the TOP rung N=8 where residual link
    # contribution is largest (methodology-lens finding 3): an ABSENT arm leaves
    # creche_none_max_cov at 0.0 and would vacuously pass 0.0 < C. Absence ->
    # refuse the verdict rather than read a can't-fail 0.
    none_by_rung = agg["creche_none_by_rung"]
    missing_none = [rung for rung in rungs if rung not in none_by_rung]
    if missing_none:
        problems.append(
            f"NOISE-FLOOR: creche_none arm absent at rung(s) {missing_none} — cannot verify the noise floor"
        )
        gates["NOISE_FLOOR"] = None
    else:
        gates["NOISE_FLOOR"] = all(none_by_rung[rung] < criterion for rung in rungs)
    details["NOISE_FLOOR"] = {
        "creche_none_max_coverage": agg["creche_none_max_cov"],
        "creche_none_by_rung": none_by_rung,
        "criterion": criterion,
    }

    # Verdict composition.
    if problems:
        verdict = "NO-VERDICT"
    elif gates.get("MONOTONICITY") and gates.get("NOT_JUST_MORE_DATA") and gates.get("NOISE_FLOOR"):
        verdict = "PASS"
    elif gates.get("MONOTONICITY") and gates.get("NOT_JUST_MORE_DATA") is False:
        verdict = "PARTIAL"  # pooling scales per-agent trials but costs total experience
    else:
        verdict = "FAIL"

    return {
        "stats": stats,
        "gates": gates,
        "gate_details": details,
        "problems": problems,
        "verdict": verdict,
        "criterion": criterion,
        "constants": GATES_V1,
    }


def run_noop_kit(artifacts_dir: Path) -> dict:
    """Re-run the kept cohort-0 fold under no-op merge variants (the D62 kit)."""
    from exp57 import common57 as X

    meta_path = artifacts_dir / "meta.json"
    snaps_path = artifacts_dir / "snapshots.json"
    if not meta_path.is_file() or not snaps_path.is_file():
        return {"kit_pass": False, "error": f"no artifacts (meta.json/snapshots.json) in {artifacts_dir}"}
    meta = json.loads(meta_path.read_text())
    raw = json.loads(snaps_path.read_text())
    snapshots = [(s[0], s[1]) for s in raw]
    slot_to_target = {int(k): v for k, v in meta["slot_to_target"].items()}
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        return X.noop_coverage_kit(
            snapshots=snapshots,
            contributor_ids=list(meta["contributor_ids"]),
            receiver_agent_id=str(meta["receiver_agent_id"]),
            slot_to_target=slot_to_target,
            workdir=Path(td),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--gate", default="v1", choices=["v1"])
    ap.add_argument("--assert-noop-fails", action="store_true")
    ap.add_argument("--artifacts", default=None, help="cohort0_artifacts dir (default: <in>.parent/cohort0_artifacts)")
    ap.add_argument("--cohorts-min", type=int, default=GATES_V1["cohorts_min"])
    ap.add_argument("--permutations", type=int, default=GATES_V1["permutations"])
    args = ap.parse_args()

    path = Path(args.inp)
    rows = load_rows(path)
    report = analyze(rows, cohorts_min=args.cohorts_min, permutations=args.permutations)

    if args.assert_noop_fails:
        artifacts = Path(args.artifacts) if args.artifacts else path.parent / "cohort0_artifacts"
        kit = run_noop_kit(artifacts)
        report["noop_kit"] = kit
        if not kit.get("kit_pass"):
            report["problems"].append("ANTI-VACUITY: a must-collapse no-op variant did not collapse — no verdict")
            report["verdict"] = "NO-VERDICT"
    elif report["verdict"] in ("PASS", "PARTIAL"):
        # ANTI-VACUITY is "apparatus; no verdict without it" (prereg §Gates /
        # outcome tree; methodology-lens finding 4). A scored positive verdict
        # MUST NOT be rendered without the no-op kit having run — the verdict
        # authority refuses rather than trusting the operator to remember the
        # flag. (A FAIL/NO-VERDICT is not a false claim, so it is left as-is.)
        report["problems"].append(
            "ANTI-VACUITY: the no-op kit did not run (pass --assert-noop-fails) — no verdict without it"
        )
        report["verdict"] = "NO-VERDICT"

    print(json.dumps(report, indent=2, default=str))
    # PARTIAL is not a silent PASS (owner call at the release checkpoint) — it
    # exits non-zero like FAIL; the printed verdict distinguishes them.
    return {"PASS": 0, "NO-VERDICT": 4, "FAIL": 1, "PARTIAL": 1}.get(report["verdict"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
