#!/usr/bin/env python3
"""Exp 56 verdict analyzer — the frozen gates, one place.

Pre-registration: docs/experiments/protocols/exp56_four_arm_sharing_preregistration.md.
Verdict constants are FROZEN here at the harness-merge commit and are
extended, never retuned (house convention). The analyzer:

* refuses a verdict on mock rows or dirty-tree rows (apparatus, not data);
* computes per-arm first-contact rates with 95% Wilson intervals —
  the TAUGHT arm's gate-counted rate admits only BIAS-DECISIVE successes
  (the prereg's mechanism assertion; raw rates reported beside it);
* evaluates the four frozen gates + the L2 seed-invariance apparatus gate;
* ``--assert-noop-fails`` re-runs the kept pair-0 artifacts through the
  no-op merge variants (the D62 kit) and refuses a verdict if the two
  must-collapse variants do not collapse.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

# ── frozen verdict constants (gate v1) ───────────────────────────────────
GATES_V1 = {
    "transferred_min": 0.70,
    "above_floor_min": 0.20,
    "want_not_file_min": 0.20,
    "both_halves_band": 0.10,  # one-sided: dangling - isolated < band
    "l2_concentration": 0.90,  # same first-contact choice in >= 90% of an arm = seed-invariant
    "min_pairs": 50,
}


def wilson(successes: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def analyze(rows: list[dict], *, min_pairs: int) -> dict:
    problems: list[str] = []
    for r in rows:
        if r.get("mock"):
            problems.append("mock rows present — the ScriptedBridge smoke is never a confirmatory record")
            break
    for r in rows:
        if r.get("working_tree_dirty_src_scripts") and not r.get("allow_dirty"):
            problems.append("dirty-tree rows without allow_dirty — gated-record contract violated")
            break

    arms: dict[str, list[dict]] = {}
    for r in rows:
        arms.setdefault(str(r.get("arm")), []).append(r)

    stats: dict[str, dict] = {}
    for arm, arm_rows in arms.items():
        n = len(arm_rows)
        raw = sum(1 for r in arm_rows if r.get("chose_target"))
        decisive = sum(1 for r in arm_rows if r.get("chose_target") and r.get("bias_decisive"))
        p_raw, lo_raw, hi_raw = wilson(raw, n)
        p_dec, lo_dec, hi_dec = wilson(decisive, n)
        # L2 seed-invariance: concentration of the literal first-contact
        # CHOICE (not target-relative) within the arm.
        choices: dict[str, int] = {}
        for r in arm_rows:
            c = str((r.get("first_contact") or {}).get("chosen"))
            choices[c] = choices.get(c, 0) + 1
        concentration = max(choices.values()) / n if n else 0.0
        stats[arm] = {
            "n": n,
            "rate_raw": round(p_raw, 4),
            "raw_ci": [round(lo_raw, 4), round(hi_raw, 4)],
            "rate_decisive": round(p_dec, 4),
            "decisive_ci": [round(lo_dec, 4), round(hi_dec, 4)],
            "choice_concentration": round(concentration, 4),
        }
        if n < min_pairs:
            problems.append(f"arm {arm}: n={n} < {min_pairs} (frozen power)")
        if concentration >= GATES_V1["l2_concentration"]:
            problems.append(
                f"arm {arm}: first-contact choice concentration {concentration:.2f} >= "
                f"{GATES_V1['l2_concentration']} — L2 seed-invariance apparatus gate fires (no verdict)"
            )

    gates: dict[str, bool | None] = {}
    taught = stats.get("taught")
    isolated = stats.get("isolated")
    satiated = stats.get("satiated")
    dangling = stats.get("dangling")
    if taught and isolated and satiated and dangling:
        # TRANSFERRED counts only bias-decisive successes (mechanism
        # assertion); the comparison gates use the taught DECISIVE rate too
        # (conservative: decisive <= raw) against the controls' raw rates.
        gates["TRANSFERRED"] = taught["rate_decisive"] >= GATES_V1["transferred_min"]
        gates["ABOVE_FLOOR"] = taught["rate_decisive"] - isolated["rate_raw"] >= GATES_V1["above_floor_min"]
        gates["WANT_NOT_FILE"] = taught["rate_decisive"] - satiated["rate_raw"] >= GATES_V1["want_not_file_min"]
        gates["BOTH_HALVES"] = dangling["rate_raw"] - isolated["rate_raw"] < GATES_V1["both_halves_band"]
    else:
        problems.append("missing arm(s) — all four are required for a verdict")

    verdict = "NO-VERDICT" if problems else ("PASS" if all(gates.values()) else "FAIL")
    return {"stats": stats, "gates": gates, "problems": problems, "verdict": verdict, "constants": GATES_V1}


def run_noop_kit(artifacts_dir: Path, rows: list[dict]) -> dict:
    from exp56 import common as C

    taught_rows = [r for r in rows if r.get("arm") == "taught"]
    if not taught_rows:
        return {"kit_pass": False, "error": "no taught rows to re-run"}
    row = taught_rows[0]
    bundle = artifacts_dir / "taught.zip"
    pre_nac = json.loads((artifacts_dir / "receiver_pre_nac.json").read_text())
    pre_nac.pop("_format_version", None)
    pre_ec = json.loads((artifacts_dir / "receiver_pre_ec.json").read_text()).get("substrate_nodes", {})
    cfg = C.pair_config(int(row["pair_seed"]))
    return C.noop_variant_readout(
        bundle=bundle,
        receiver_pre_nac=pre_nac,
        receiver_pre_ec=pre_ec,
        receiver_agent_id=f"recv_taught_{row['pair_seed']}",
        contributor_id=cfg["contributor_id"],
        first_contact=row.get("first_contact") or {},
        target_tool=f"{C.ENTITY_NAME}_{row['target_aff']}",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--gate", default="v1", choices=["v1"])
    ap.add_argument("--assert-noop-fails", action="store_true")
    ap.add_argument("--artifacts", default=None, help="pair0_artifacts dir (default: <in>.parent/pair0_artifacts)")
    ap.add_argument("--min-pairs", type=int, default=GATES_V1["min_pairs"])
    args = ap.parse_args()

    path = Path(args.inp)
    rows = load_rows(path)
    report = analyze(rows, min_pairs=args.min_pairs)

    if args.assert_noop_fails:
        artifacts = Path(args.artifacts) if args.artifacts else path.parent / "pair0_artifacts"
        kit = run_noop_kit(artifacts, rows)
        report["noop_kit"] = kit
        if not kit.get("kit_pass"):
            report["problems"].append("ANTI-VACUITY: a must-collapse no-op variant did not collapse — no verdict")
            report["verdict"] = "NO-VERDICT"

    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else (4 if report["verdict"] == "NO-VERDICT" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
