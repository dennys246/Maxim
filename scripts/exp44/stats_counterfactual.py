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
      P(safe-colored source). CAVEAT (two-lens fold): the ablated prompt lacks
      the substrate ANNOTATION but is NOT experience-free — the captured
      trajectory was generated under substrate-steered actions, and --resume-sim
      restores hippocampal episodes into both variants. This measures the LLM's
      within-context color preference, not a raw prior; the flip metric itself
      stays valid because both variants share that context symmetrically.
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
    """{model: {(arm, seed): [records]}} from arms/<arm>/seed<N>/requery/<model>__*.jsonl.

    FAILS LOUD on duplicate (model, arm, seed) files: campaign.py prunes the
    requery dir on re-capture, so a duplicate means stale results from a
    superseded capture coexist with fresh ones — silently keeping either would
    make the confirmatory analysis nondeterministic (cross-confirmed two-lens
    finding). Delete the stale file rather than letting sort order pick.
    """
    out: dict[str, dict[tuple[str, int], list[dict[str, Any]]]] = defaultdict(dict)
    for f in sorted(campaign.glob("arms/*/seed*/requery/*.jsonl")):
        arm = f.parents[2].name
        seed = int(f.parents[1].name.removeprefix("seed"))
        model = f.name.split("__", 1)[0]
        if (arm, seed) in out[model]:
            raise SystemExit(
                f"DUPLICATE requery results for model={model} arm={arm} seed={seed} "
                f"(stale capture-hash files coexist under {f.parent}) — delete the stale one and re-run"
            )
        with open(f, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        out[model][(arm, seed)] = recs
    return out


def is_void(campaign: Path, arm: str, seed: int) -> bool:
    """True when campaign.py marked this control cell VOID (transplanted
    substrate never surfaced in the capture prompts — see the pre-registration's
    transplant validity gate)."""
    return (campaign / "arms" / arm / f"seed{seed}" / "control_void.json").exists()


# All warmth entities across every 44b/44c arc, LONGEST FIRST so the most
# specific name wins the substring match ("green_flame_b" before "green_flame";
# bare "hearth" LAST — it is a substring of every hearth twin AND names the
# 44c collision arm's false hearth).
_FLAME_ENTITIES = (
    "purple_hearth_b",
    "purple_flame_b",
    "green_hearth_b",
    "green_flame_b",
    "purple_hearth",
    "purple_flame",
    "green_hearth",
    "green_flame",
    "hearth",
)


def referenced_flame(action: str | None) -> str | None:
    """Which warmth entity an action name references (longest-match), or None."""
    if not action:
        return None
    for ent in _FLAME_ENTITIES:
        if ent in action:
            return ent
    return None


def is_phantom(action: str | None, world: tuple[str, ...] | None) -> bool:
    """True when the action references a flame that does NOT exist in this
    arm's world — a cross-arc twin surfaced by registry-wide discovery
    enrichment (pilot finding, 2026-08-10). Phantom picks are unexecutable in
    the arm's scene, and the naive substring safety-rank would mis-score them
    (``"green_flame" in "green_flame_b_warm_self"`` → counted safe in arm A).
    Excluded from scoring, COUNTED and reported (never silently dropped)."""
    if world is None:
        return False
    ent = referenced_flame(action)
    return ent is not None and ent not in world


def score_run(
    recs: list[dict[str, Any]],
    safe: str,
    harm: str,
    entropy_hi: float,
    world: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Per-(arm,seed) counts: directional / commit / baseline / entropy slices."""
    usable = [r for r in recs if r.get("action_full") is not None and r.get("action_ablated") is not None]
    phantom = [
        r for r in usable if is_phantom(r.get("action_full"), world) or is_phantom(r.get("action_ablated"), world)
    ]
    scored = [r for r in usable if r not in phantom]
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
        "n_phantom_excluded": len(phantom),
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

    # Pools (frozen in the prereg): "confirmatory" = the counterbalanced pair,
    # the ONLY arms entering the primary test; "companion" = pre-registered
    # 44c arms (collision / hearth twins) reported separately with their own
    # frozen predictions; controls (transplant/none) also never pooled.
    # Default: learn arms are confirmatory unless they declare pool=companion.
    def _pool(a: dict[str, Any]) -> str:
        explicit = a.get("pool")
        if explicit:
            return explicit
        return "confirmatory" if a.get("substrate", "learn") == "learn" else "control"

    confirmatory = {n for n, a in arms_cfg.items() if _pool(a) == "confirmatory"}
    companion = {n for n, a in arms_cfg.items() if _pool(a) == "companion"}

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
            world = tuple(a["world_entities"]) if a.get("world_entities") else None
            per_run[(arm, seed)] = score_run(recs, a["safe_substr"], a["harm_substr"], entropy_hi, world=world)

        conf = {k: v for k, v in per_run.items() if k[0] in confirmatory}
        comp = {k: v for k, v in per_run.items() if k[0] in companion}
        ctrl = {k: v for k, v in per_run.items() if k[0] not in confirmatory and k[0] not in companion}

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
            m["control_arms"] = {
                f"{k[0]}/seed{k[1]}": {**v, **({"VOID": True} if is_void(campaign, k[0], k[1]) else {})}
                for k, v in sorted(ctrl.items())
            }
            n_void = sum(1 for k in ctrl if is_void(campaign, k[0], k[1]))
            if n_void:
                print(
                    f"  WARNING: {n_void}/{len(ctrl)} control cell(s) VOID — transplanted "
                    f"substrate never surfaced (see control_void.json markers); "
                    f"the wrong-content control is uninterpretable for those cells"
                )
        if comp:
            # Companion arms (44c): per-arm aggregates, NEVER pooled into the
            # primary. The paper's dose-response comparison reads flip-direction
            # and commit rates across pools from these blocks.
            m["companion_arms"] = {}
            for arm_name in sorted({k[0] for k in comp}):
                runs_c = {k: v for k, v in comp.items() if k[0] == arm_name}
                ts_c = sum(v["toward_safe"] for v in runs_c.values())
                th_c = sum(v["toward_harm"] for v in runs_c.values())
                tc_c = sum(v["toward_commit"] for v in runs_c.values())
                to_c = sum(v["toward_observe"] for v in runs_c.values())
                m["companion_arms"][arm_name] = {
                    "n_runs": len(runs_c),
                    "toward_safe": ts_c,
                    "toward_harm": th_c,
                    "direction_wilson_95ci": list(wilson_ci(ts_c, ts_c + th_c)),
                    "toward_commit": tc_c,
                    "toward_observe": to_c,
                    "per_run": {f"seed{k[1]}": v for k, v in sorted(runs_c.items())},
                }
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
        n_phantom = sum(v["n_phantom_excluded"] for v in per_run.values())
        if n_phantom:
            print(
                f"  phantom picks excluded: {n_phantom} (action referenced a flame "
                f"not in the arm's world — cross-arc discovery leakage; see prereg)"
            )
        for arm, b in baseline.items():
            ps = f"{b['p_safe']:.3f}" if b["p_safe"] is not None else "n/a"
            print(
                f"  intrinsic baseline [{arm}]: P(safe-colored | ablated, flame) = {ps} "
                f"({b['ablated_safe']}/{b['ablated_safe'] + b['ablated_harm']})"
            )
        for arm_name, c in m.get("companion_arms", {}).items():
            lo_c, hi_c = c["direction_wilson_95ci"]
            print(
                f"  companion [{arm_name}]: {c['toward_safe']} safe vs {c['toward_harm']} harm "
                f"(Wilson95 [{lo_c:.2f},{hi_c:.2f}]); commit {c['toward_commit']}/{c['toward_observe']} "
                f"(never pooled into the primary)"
            )
        # H3 is a HEADLINE, not a footnote: the wrong-content control is what
        # separates "the substrate steers by learned CONTENT" from "naming any
        # tool in the prompt raises its selection". Pre-fix these numbers went
        # to stats.json and NEVER to the console (only a VOID warning did), so
        # the pilot's most decisive result was invisible in its own report.
        for arm_name in sorted({k[0] for k in ctrl}):
            rows = [v for k, v in ctrl.items() if k[0] == arm_name]
            voids = sum(1 for k in ctrl if k[0] == arm_name and is_void(campaign, k[0], k[1]))
            ts_x = sum(v["toward_safe"] for v in rows)
            th_x = sum(v["toward_harm"] for v in rows)
            lo_x, hi_x = wilson_ci(th_x, ts_x + th_x)  # CI on the FOLLOWS-ANNOTATION side
            tag = f" [{voids}/{len(rows)} VOID]" if voids else ""
            print(
                f"  CONTROL [{arm_name}]{tag}: {th_x} follow-annotation (toward-harm here) vs "
                f"{ts_x} toward-safe; Wilson95 p_follow=[{lo_x:.2f},{hi_x:.2f}]; "
                f"commit {sum(v['toward_commit'] for v in rows)}/{sum(v['toward_observe'] for v in rows)}"
            )
            print(
                "    ^ H3: follow-annotation DOMINANT => channel is generic, learned content "
                "is what makes 44b's steering safe. toward-safe DOMINANT => the LLM is "
                "correcting the wrong substrate from context (report honestly; weakens H3)."
            )

    out = Path(args.out) if args.out else campaign / "stats.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nstats written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
