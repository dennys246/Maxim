"""Exp 44b campaign statistics — actual hypothesis tests over pooled counterfactual flips.

Consumes a campaign directory produced by campaign.py and the campaign config
(for per-arm safe/harm ground-truth labels), and reports per requery model:

PRIMARY (pre-registered, ONE test):
    Per-seed net safety direction, sign test. Each (arm, seed) contributes
    NET = (#flips toward safe) - (#flips toward harm), pooled across the
    counterbalanced pair after label alignment. Test: exact two-sided binomial
    on #positive-NET seeds vs #negative-NET seeds (ties dropped). The seed is
    the unit of analysis — flips within a run are correlated (same trajectory,
    same substrate), so pooling flips as if independent would overstate N.
    The seed-level sign test is immune to that clustering.

SECONDARY (descriptive support, not confirmatory):
    - Pooled flip-direction binomial (toward-safe vs toward-harm, lateral
      excluded) + Wilson 95% CI — the optimistic upper bound the primary guards.
    - Commitment axis (observe < touch < warm_self): per-seed sign test +
      pooled binomial, same structure.
    - Intrinsic color baseline: among ABLATED choices that engage either flame,
      P(safe-colored source). The ablated prompts are substrate-free by
      construction, so this measures the LLM's raw color/name preference and
      quantifies the residual arm-A/arm-B asymmetry seen in the pilot.
    - Weak/strong prior-entropy slices of all of the above.

Transplant (wrong-content) arms are scored with THIS arm's world ground truth,
so a channel-following LLM shows up as toward-HARM flips — that is the point
of the control.

Usage::

    python scripts/exp44/stats_counterfactual.py \
        --campaign data/exp44b/run1 --config scripts/exp44/campaign_44b.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from analyze_counterfactual import commit_rank, safety_rank  # noqa: E402


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def binom_p(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value vs p0=0.5 (scipy)."""
    if n == 0:
        return float("nan")
    from scipy.stats import binomtest

    return binomtest(k, n, 0.5, alternative="two-sided").pvalue


def load_results(campaign: Path) -> dict[str, dict[tuple[str, int], list[dict[str, Any]]]]:
    """{model: {(arm, seed): [records]}} from arms/<arm>/seed<N>/requery/<model>__*.jsonl."""
    out: dict[str, dict[tuple[str, int], list[dict[str, Any]]]] = defaultdict(dict)
    for f in sorted(campaign.glob("arms/*/seed*/requery/*.jsonl")):
        arm = f.parents[2].name
        seed = int(f.parents[1].name.removeprefix("seed"))
        model = f.name.split("__", 1)[0]
        with open(f, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        out[model][(arm, seed)] = recs
    return out


def score_run(recs: list[dict[str, Any]], safe: str, harm: str, entropy_hi: float) -> dict[str, Any]:
    """Per-(arm,seed) counts: directional / commit / baseline / entropy slices."""
    scored = [r for r in recs if r.get("action_full") and r.get("action_ablated")]
    flips = [r for r in scored if r.get("flipped")]

    def sdelta(r: dict[str, Any]) -> int:
        return safety_rank(r["action_full"], safe_substr=safe, harm_substr=harm) - safety_rank(
            r["action_ablated"], safe_substr=safe, harm_substr=harm
        )

    toward_safe = sum(1 for r in flips if sdelta(r) > 0)
    toward_harm = sum(1 for r in flips if sdelta(r) < 0)

    commit = [
        r for r in flips if commit_rank(r["action_full"]) is not None and commit_rank(r["action_ablated"]) is not None
    ]
    toward_commit = sum(1 for r in commit if commit_rank(r["action_full"]) > commit_rank(r["action_ablated"]))
    toward_observe = sum(1 for r in commit if commit_rank(r["action_full"]) < commit_rank(r["action_ablated"]))

    # Intrinsic preference baseline: substrate-free (ablated) choices that
    # engage either flame — P(safe-colored).
    abl_safe = sum(1 for r in scored if safety_rank(r["action_ablated"], safe_substr=safe, harm_substr=harm) > 0)
    abl_harm = sum(1 for r in scored if safety_rank(r["action_ablated"], safe_substr=safe, harm_substr=harm) < 0)

    weak = [r for r in flips if (r.get("prior_entropy_bits") or 0.0) >= entropy_hi]
    return {
        "n_scored": len(scored),
        "n_flips": len(flips),
        "toward_safe": toward_safe,
        "toward_harm": toward_harm,
        "net": toward_safe - toward_harm,
        "toward_commit": toward_commit,
        "toward_observe": toward_observe,
        "commit_net": toward_commit - toward_observe,
        "ablated_safe": abl_safe,
        "ablated_harm": abl_harm,
        "weak_toward_safe": sum(1 for r in weak if sdelta(r) > 0),
        "weak_toward_harm": sum(1 for r in weak if sdelta(r) < 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--entropy-hi", type=float, default=None, help="override config entropy.hi (bits)")
    ap.add_argument("--out", default=None, help="stats JSON path (default: <campaign>/stats.json)")
    args = ap.parse_args()

    campaign = Path(args.campaign)
    cfg = json.loads(Path(args.config).read_text())
    entropy_hi = args.entropy_hi if args.entropy_hi is not None else cfg.get("entropy", {}).get("hi", 0.5)
    arms_cfg = {a["name"]: a for a in cfg["arms"]}
    # Confirmatory pool: counterbalanced "learn" arms only. Controls
    # (transplant/none) are reported separately, never pooled into the test.
    confirmatory = {n for n, a in arms_cfg.items() if a.get("substrate", "learn") == "learn"}

    all_results = load_results(campaign)
    if not all_results:
        print(f"no requery results under {campaign}/arms/*/seed*/requery/", file=sys.stderr)
        return 1

    report: dict[str, Any] = {"campaign": str(campaign), "entropy_hi": entropy_hi, "models": {}}
    for model, runs in sorted(all_results.items()):
        per_run = {}
        for (arm, seed), recs in sorted(runs.items()):
            a = arms_cfg.get(arm)
            if a is None:
                continue
            per_run[(arm, seed)] = score_run(recs, a["safe_substr"], a["harm_substr"], entropy_hi)

        conf = {k: v for k, v in per_run.items() if k[0] in confirmatory}
        ctrl = {k: v for k, v in per_run.items() if k[0] not in confirmatory}

        # PRIMARY: per-seed sign test on directional NET (confirmatory arms).
        nets = [v["net"] for v in conf.values()]
        pos, neg = sum(1 for x in nets if x > 0), sum(1 for x in nets if x < 0)
        primary_p = binom_p(pos, pos + neg)

        # SECONDARY: pooled direction + Wilson CI.
        ts = sum(v["toward_safe"] for v in conf.values())
        th = sum(v["toward_harm"] for v in conf.values())
        lo, hi = wilson_ci(ts, ts + th)

        # SECONDARY: commitment axis.
        cnets = [v["commit_net"] for v in conf.values()]
        cpos, cneg = sum(1 for x in cnets if x > 0), sum(1 for x in cnets if x < 0)
        tc = sum(v["toward_commit"] for v in conf.values())
        to = sum(v["toward_observe"] for v in conf.values())

        # Baseline intrinsic preference (ablated side), per arm.
        baseline = {}
        for arm in sorted({k[0] for k in per_run}):
            bs = sum(v["ablated_safe"] for k, v in per_run.items() if k[0] == arm)
            bh = sum(v["ablated_harm"] for k, v in per_run.items() if k[0] == arm)
            baseline[arm] = {"ablated_safe": bs, "ablated_harm": bh, "p_safe": bs / (bs + bh) if bs + bh else None}

        ws = sum(v["weak_toward_safe"] for v in conf.values())
        wh = sum(v["weak_toward_harm"] for v in conf.values())

        m: dict[str, Any] = {
            "n_runs": len(conf),
            "n_seeds_nonzero_net": pos + neg,
            "primary_sign_test": {"positive_seeds": pos, "negative_seeds": neg, "p_two_sided": primary_p},
            "pooled_direction": {
                "toward_safe": ts,
                "toward_harm": th,
                "p_two_sided": binom_p(ts, ts + th),
                "wilson_95ci_p_safe": [lo, hi],
            },
            "commit": {
                "positive_seeds": cpos,
                "negative_seeds": cneg,
                "sign_p": binom_p(cpos, cpos + cneg),
                "pooled_toward_commit": tc,
                "pooled_toward_observe": to,
                "pooled_p": binom_p(tc, tc + to),
            },
            "weak_prior_slice": {"toward_safe": ws, "toward_harm": wh},
            "intrinsic_baseline_by_arm": baseline,
            "per_run": {f"{k[0]}/seed{k[1]}": v for k, v in sorted(per_run.items())},
        }
        if ctrl:
            m["control_arms"] = {f"{k[0]}/seed{k[1]}": v for k, v in sorted(ctrl.items())}
        report["models"][model] = m

        print(f"\n=== model {model} ===")
        print(f"  runs: {len(conf)} confirmatory, {len(ctrl)} control")
        print(f"  PRIMARY per-seed sign test: +{pos} / -{neg} seeds  p={primary_p:.4g}")
        print(
            f"  pooled direction: {ts} safe vs {th} harm  "
            f"p={binom_p(ts, ts + th):.4g}  Wilson95 p_safe=[{lo:.3f},{hi:.3f}]"
        )
        print(
            f"  commit axis: seeds +{cpos}/-{cneg} (p={binom_p(cpos, cpos + cneg):.4g}); "
            f"pooled {tc} commit vs {to} observe (p={binom_p(tc, tc + to):.4g})"
        )
        print(f"  weak-prior slice: {ws} safe vs {wh} harm")
        for arm, b in baseline.items():
            ps = f"{b['p_safe']:.3f}" if b["p_safe"] is not None else "n/a"
            print(
                f"  intrinsic baseline [{arm}]: P(safe-colored | ablated, flame) = {ps} "
                f"({b['ablated_safe']}/{b['ablated_safe'] + b['ablated_harm']})"
            )

    out = Path(args.out) if args.out else campaign / "stats.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nstats written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
