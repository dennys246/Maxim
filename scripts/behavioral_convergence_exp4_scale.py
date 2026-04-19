#!/usr/bin/env python3
"""Behavioral Convergence Experiment 4 — Scale Validation (20+ seeds).

Wraps the Tier 3 organic learning experiment and runs it N times to prove
the learning effect is statistically robust, not a fluke of LLM sampling.

Each seed runs the full 3-session training pipeline + 1 fresh control
with an isolated persist directory.  Per-seed metrics are collected and
aggregated into mean ± std with statistical significance tests.

Primary metric: Session 3 teal rate > Session 1 teal rate
Statistical test: Wilcoxon signed-rank (paired, non-parametric)
Pass gate: p < 0.05 AND mean Session 3 teal rate >= 0.70

Usage
-----
    # 20 seeds (default), ~60-100 min on single GPU:
    PYTHONPATH=src python scripts/behavioral_convergence_exp4_scale.py

    # Custom seed count:
    PYTHONPATH=src python scripts/behavioral_convergence_exp4_scale.py --seeds 25

    # Resume from a partial run:
    PYTHONPATH=src python scripts/behavioral_convergence_exp4_scale.py --resume-dir /tmp/maxim_scale_xyz

    # JSON output for downstream analysis:
    PYTHONPATH=src python scripts/behavioral_convergence_exp4_scale.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("MAXIM_SUBSTRATE_PATH", "1")

# Import the single-trial experiment building blocks
from behavioral_convergence_exp4_tier3 import (
    NAc,
    make_hippocampus,
    run_single,
    wire_pipeline,
)


def run_single_trial(trial_id: int, base_dir: Path) -> dict:
    """Run one complete 3-session + control trial with isolated state."""
    trial_dir = base_dir / f"seed_{trial_id:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    hippo_path = str(trial_dir / "hippocampus.json")
    nac_path = str(trial_dir / "nac.json")

    t0 = time.time()
    runs = []

    # Session 1: Fresh agent explores
    h1 = make_hippocampus(hippo_path)
    nac1 = NAc()
    bus1 = wire_pipeline(h1, nac1)
    r1 = run_single(h1, nac1, bus1, 1, f"[seed {trial_id}] Fresh explore")
    h1.save()
    nac1.save(nac_path)
    runs.append(r1)

    # Session 2: Loaded state
    h2 = make_hippocampus(hippo_path)
    h2.load()
    nac2 = NAc()
    nac2.load(nac_path)
    bus2 = wire_pipeline(h2, nac2)
    r2 = run_single(h2, nac2, bus2, 2, f"[seed {trial_id}] Loaded improve")
    h2.save()
    nac2.save(nac_path)
    runs.append(r2)

    # Session 3: Convergence
    h3 = make_hippocampus(hippo_path)
    h3.load()
    nac3 = NAc()
    nac3.load(nac_path)
    bus3 = wire_pipeline(h3, nac3)
    r3 = run_single(h3, nac3, bus3, 3, f"[seed {trial_id}] Convergence")
    runs.append(r3)

    # Fresh control
    h4 = make_hippocampus()
    nac4 = NAc()
    bus4 = wire_pipeline(h4, nac4)
    r4 = run_single(h4, nac4, bus4, 4, f"[seed {trial_id}] Fresh control")
    runs.append(r4)

    elapsed = time.time() - t0

    # Extract per-session teal rates
    teal_rates = []
    for run in runs[:3]:
        total_choices = len(run["choices"]) or 1
        teal = run["choice_counts"].get("teal_cylindrical_ceramic_vial", 0)
        teal_rates.append(teal / total_choices)

    control_escaped = runs[3]["escaped"]
    control_teal = (
        runs[3]["choice_counts"].get("teal_cylindrical_ceramic_vial", 0)
        / max(1, len(runs[3]["choices"]))
    )

    return {
        "trial_id": trial_id,
        "persist_dir": str(trial_dir),
        "elapsed_s": round(elapsed, 1),
        "teal_rate_s1": teal_rates[0],
        "teal_rate_s2": teal_rates[1],
        "teal_rate_s3": teal_rates[2],
        "s1_escaped": runs[0]["escaped"],
        "s2_escaped": runs[1]["escaped"],
        "s3_escaped": runs[2]["escaped"],
        "s3_turns": runs[2]["turns"],
        "control_escaped": control_escaped,
        "control_teal_rate": control_teal,
        "control_died": runs[3]["died"],
        "runs": runs,
    }


def load_completed_trials(base_dir: Path) -> list[dict]:
    """Load completed trial results from a previous run."""
    results_file = base_dir / "partial_results.jsonl"
    if not results_file.exists():
        return []
    trials = []
    for line in results_file.read_text().strip().split("\n"):
        if line.strip():
            trials.append(json.loads(line))
    return trials


def save_trial_result(base_dir: Path, trial: dict) -> None:
    """Append a trial result to the partial results file."""
    results_file = base_dir / "partial_results.jsonl"
    # Write a lightweight version (no full runs data) for JSONL
    lightweight = {k: v for k, v in trial.items() if k != "runs"}
    with open(results_file, "a") as f:
        f.write(json.dumps(lightweight, default=str) + "\n")


def compute_statistics(trials: list[dict]) -> dict:
    """Compute aggregate statistics across all completed trials."""
    import numpy as np

    n = len(trials)
    s1_rates = np.array([t["teal_rate_s1"] for t in trials])
    s2_rates = np.array([t["teal_rate_s2"] for t in trials])
    s3_rates = np.array([t["teal_rate_s3"] for t in trials])
    control_rates = np.array([t["control_teal_rate"] for t in trials])

    s3_escaped = sum(1 for t in trials if t["s3_escaped"])
    control_died = sum(1 for t in trials if t["control_died"])
    control_escaped = sum(1 for t in trials if t["control_escaped"])

    # Session 3 vs Session 1 improvement
    improvement = s3_rates - s1_rates

    # Wilcoxon signed-rank test: is Session 3 teal rate > Session 1?
    p_value = None
    try:
        from scipy.stats import wilcoxon

        # One-sided: alternative="greater" — Session 3 > Session 1
        stat, p_value = wilcoxon(
            s3_rates, s1_rates, alternative="greater", zero_method="wilcox"
        )
        p_value = float(p_value)
    except Exception as e:
        p_value = None
        print(f"  WARNING: Wilcoxon test failed: {e}")

    # Session 3 vs control
    p_value_vs_control = None
    try:
        from scipy.stats import mannwhitneyu

        stat2, p_value_vs_control = mannwhitneyu(
            s3_rates, control_rates, alternative="greater"
        )
        p_value_vs_control = float(p_value_vs_control)
    except Exception:
        pass

    stats = {
        "n_seeds": n,
        "teal_rate": {
            "session_1": {
                "mean": float(np.mean(s1_rates)),
                "std": float(np.std(s1_rates, ddof=1)) if n > 1 else 0.0,
                "min": float(np.min(s1_rates)),
                "max": float(np.max(s1_rates)),
            },
            "session_2": {
                "mean": float(np.mean(s2_rates)),
                "std": float(np.std(s2_rates, ddof=1)) if n > 1 else 0.0,
                "min": float(np.min(s2_rates)),
                "max": float(np.max(s2_rates)),
            },
            "session_3": {
                "mean": float(np.mean(s3_rates)),
                "std": float(np.std(s3_rates, ddof=1)) if n > 1 else 0.0,
                "min": float(np.min(s3_rates)),
                "max": float(np.max(s3_rates)),
            },
            "control": {
                "mean": float(np.mean(control_rates)),
                "std": float(np.std(control_rates, ddof=1)) if n > 1 else 0.0,
            },
        },
        "improvement_s3_vs_s1": {
            "mean": float(np.mean(improvement)),
            "std": float(np.std(improvement, ddof=1)) if n > 1 else 0.0,
        },
        "wilcoxon_p_value": p_value,
        "mannwhitney_s3_vs_control_p": p_value_vs_control,
        "s3_escape_rate": s3_escaped / n,
        "control_death_rate": control_died / n,
        "control_escape_rate": control_escaped / n,
    }
    return stats


def print_report(trials: list[dict], stats: dict) -> bool:
    """Print a human-readable report and return True if all gates pass."""
    print("\n" + "=" * 72)
    print("TIER 3 SCALE VALIDATION — ORGANIC LLM LEARNING")
    print(f"Seeds: {stats['n_seeds']}")
    print("=" * 72)

    # Per-seed summary table
    print(f"\n{'Seed':>6}  {'S1 Teal':>8}  {'S2 Teal':>8}  {'S3 Teal':>8}  {'S3 Esc':>6}  {'Ctrl':>8}  {'Time':>6}")
    print("-" * 72)
    for t in trials:
        ctrl = "DIED" if t["control_died"] else f"{t['control_teal_rate']:.0%}"
        s3_esc = "YES" if t["s3_escaped"] else "NO"
        print(
            f"  {t['trial_id']:>4}  {t['teal_rate_s1']:>7.0%}  "
            f"{t['teal_rate_s2']:>7.0%}  {t['teal_rate_s3']:>7.0%}  "
            f"{s3_esc:>6}  {ctrl:>8}  {t['elapsed_s']:>5.0f}s"
        )

    # Aggregate statistics
    tr = stats["teal_rate"]
    print(f"\n{'AGGREGATE':>10}")
    print("-" * 72)
    print(
        f"  Session 1 teal rate:  {tr['session_1']['mean']:.1%} ± {tr['session_1']['std']:.1%}  "
        f"[{tr['session_1']['min']:.0%}, {tr['session_1']['max']:.0%}]"
    )
    print(
        f"  Session 2 teal rate:  {tr['session_2']['mean']:.1%} ± {tr['session_2']['std']:.1%}  "
        f"[{tr['session_2']['min']:.0%}, {tr['session_2']['max']:.0%}]"
    )
    print(
        f"  Session 3 teal rate:  {tr['session_3']['mean']:.1%} ± {tr['session_3']['std']:.1%}  "
        f"[{tr['session_3']['min']:.0%}, {tr['session_3']['max']:.0%}]"
    )
    print(f"  Control teal rate:   {tr['control']['mean']:.1%} ± {tr['control']['std']:.1%}")
    imp = stats["improvement_s3_vs_s1"]
    print(f"\n  S3-S1 improvement:   {imp['mean']:+.1%} ± {imp['std']:.1%}")
    print(f"  S3 escape rate:      {stats['s3_escape_rate']:.0%}")
    print(f"  Control death rate:  {stats['control_death_rate']:.0%}")

    # Statistical tests
    print(f"\n{'STATISTICAL TESTS':>20}")
    print("-" * 72)
    p = stats["wilcoxon_p_value"]
    if p is not None:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  Wilcoxon signed-rank (S3 > S1):  p = {p:.6f}  {sig}")
    else:
        print("  Wilcoxon signed-rank:  FAILED (not enough data?)")

    p2 = stats["mannwhitney_s3_vs_control_p"]
    if p2 is not None:
        sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "n.s."
        print(f"  Mann-Whitney (S3 > control):     p = {p2:.6f}  {sig2}")

    # Gates
    print(f"\n{'PASS GATES':>12}")
    print("-" * 72)
    gates_passed = 0
    gates_total = 0

    def gate(desc: str, cond: bool) -> None:
        nonlocal gates_passed, gates_total
        gates_total += 1
        if cond:
            gates_passed += 1
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {desc}")

    gate(
        f"Mean S3 teal rate >= 70% ({tr['session_3']['mean']:.0%})",
        tr["session_3"]["mean"] >= 0.70,
    )
    gate(
        f"Mean S3-S1 improvement > 0 ({imp['mean']:+.1%})",
        imp["mean"] > 0,
    )
    gate(
        f"Wilcoxon p < 0.05 (p={p})",
        p is not None and p < 0.05,
    )
    gate(
        f"S3 escape rate >= 80% ({stats['s3_escape_rate']:.0%})",
        stats["s3_escape_rate"] >= 0.80,
    )
    gate(
        f"Control death rate >= 60% ({stats['control_death_rate']:.0%})",
        stats["control_death_rate"] >= 0.60,
    )
    gate(
        f"Mean S3 teal > mean control teal ({tr['session_3']['mean']:.0%} vs {tr['control']['mean']:.0%})",
        tr["session_3"]["mean"] > tr["control"]["mean"],
    )

    all_pass = gates_passed == gates_total
    print(f"\n  Gates: {gates_passed}/{gates_total}")

    if all_pass:
        print("\n" + "=" * 72)
        print("ALL GATES PASS — LEARNING IS ROBUST, NOT A FLUKE")
        print("=" * 72)
    else:
        print("\n" + "=" * 72)
        print(f"  {gates_total - gates_passed} gate(s) FAILED")
        print("=" * 72)

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Tier 3 scale validation — 20+ seed organic learning"
    )
    parser.add_argument(
        "--seeds", type=int, default=20,
        help="Number of independent seeds to run (default: 20)"
    )
    parser.add_argument(
        "--resume-dir", type=str, default=None,
        help="Resume from a partial run directory"
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model override (default: peer config model)"
    )
    args = parser.parse_args()

    # Set up base directory
    if args.resume_dir:
        base_dir = Path(args.resume_dir)
        if not base_dir.exists():
            print(f"ERROR: resume directory not found: {base_dir}")
            sys.exit(1)
    else:
        base_dir = Path(tempfile.mkdtemp(prefix="maxim_scale_"))

    print(f"Scale validation: {args.seeds} seeds")
    print(f"Output directory: {base_dir}")

    # Load any completed trials from a partial run
    completed = load_completed_trials(base_dir)
    completed_ids = {t["trial_id"] for t in completed}
    if completed:
        print(f"Resuming: {len(completed)} seeds already complete")

    # Run remaining trials
    all_trials = list(completed)
    total_t0 = time.time()

    for seed_id in range(1, args.seeds + 1):
        if seed_id in completed_ids:
            continue

        print(f"\n{'#' * 60}")
        print(f"  SEED {seed_id}/{args.seeds}")
        print(f"{'#' * 60}")

        trial = run_single_trial(seed_id, base_dir)
        all_trials.append(trial)
        save_trial_result(base_dir, trial)

        # Progress update
        done = len(all_trials)
        elapsed = time.time() - total_t0
        if done > len(completed):
            new_done = done - len(completed)
            per_trial = elapsed / new_done
            remaining = (args.seeds - done) * per_trial
            print(
                f"\n  Progress: {done}/{args.seeds} seeds, "
                f"~{remaining / 60:.0f} min remaining"
            )

    # Sort by trial_id for consistent output
    all_trials.sort(key=lambda t: t["trial_id"])

    # Compute statistics
    stats = compute_statistics(all_trials)

    total_elapsed = time.time() - total_t0
    stats["total_elapsed_s"] = round(total_elapsed, 1)

    # Save full results
    full_results = {
        "experiment": "tier3_scale_validation",
        "n_seeds": args.seeds,
        "base_dir": str(base_dir),
        "statistics": stats,
        "trials": [{k: v for k, v in t.items() if k != "runs"} for t in all_trials],
    }
    results_path = base_dir / "scale_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    if args.json:
        print(json.dumps(full_results, indent=2, default=str))
        # Check gates even in JSON mode for proper exit code
        all_pass = (
            stats["teal_rate"]["session_3"]["mean"] >= 0.70
            and stats["improvement_s3_vs_s1"]["mean"] > 0
            and stats.get("wilcoxon_p_value") is not None
            and stats["wilcoxon_p_value"] < 0.05
        )
        sys.exit(0 if all_pass else 1)
    else:
        success = print_report(all_trials, stats)
        print(f"\n  Full results: {results_path}")
        print(f"  Total time: {total_elapsed / 60:.1f} min")
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
