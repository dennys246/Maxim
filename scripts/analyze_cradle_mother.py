#!/usr/bin/env python
"""Analyze the cradle-mother fade-curve results (benchmark_cradle_mother.py).

The claim: as the mother's guide fades (Act 1 full → Act 3/4 none), the TAUGHT
infant keeps getting fed because it learned to orient toward the voice ITSELF,
while the SCAFFOLD_ONLY infant (guided+fed but bio-learning disabled) drops off,
and DRIVE_ONLY (no caregiver) is the built-in-drive floor.

Metric: per act, ``self_orient_rate`` = fraction of turns the infant was oriented
by its OWN action (fed on a turn the mother did not guide). The AUTONOMOUS acts
(act3/act4, no guide) are the payoff — a high self_orient_rate there means the
scaffold successfully faded into learned behavior.

Verdict:
  LEARNED         : taught autonomous self_orient ≥ 0.5 AND ≥ scaffold_only + 0.30
                    (the fade is LEARNED, not just guided-and-fed).
  TAUGHT>DRIVE     : taught autonomous ≥ drive_only + 0.20 (the mother's feed-
                    scaffold beats the built-in drive alone).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AUTONOMOUS_ACTS = ("act3_autonomous", "act4_autonomous_voice")
ACT_ORDER = ("act1_fully_guided", "act2_co_active", "act3_autonomous", "act4_autonomous_voice")
LEARNED_MIN = 0.5
LEARNED_MARGIN = 0.30
DRIVE_MARGIN = 0.20


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--trials", type=int, default=None, help="expected seeds/arm (warns if short)")
    args = p.parse_args()

    rows = [json.loads(x) for x in Path(args.inp).read_text().splitlines() if x.strip()]
    if not rows:
        print("no rows", file=sys.stderr)
        return 2

    # per arm: per-act list of self_orient_rate, and the autonomous-act pooled value
    arms: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        arm = r["arm"]
        acts = arms.setdefault(arm, {})
        for act, m in (r.get("fade") or {}).items():
            acts.setdefault(act, []).append(float(m.get("self_orient_rate", 0.0)))

    def auto(arm: str) -> float:
        vals: list[float] = []
        for a in AUTONOMOUS_ACTS:
            vals += arms.get(arm, {}).get(a, [])
        return _mean(vals)

    print("## Cradle-mother fade curve\n")
    print(f"{'arm':14} " + " ".join(f"{a.split('_')[0]:>6}" for a in ACT_ORDER) + "   seeds")
    print("-" * 56)
    for arm in ("taught", "drive_only", "scaffold_only"):
        if arm not in arms:
            continue
        cells = []
        for a in ACT_ORDER:
            xs = arms[arm].get(a, [])
            cells.append(f"{_mean(xs):6.2f}" if xs else "   -- ")
        n = max((len(v) for v in arms[arm].values()), default=0)
        flag = ""
        if args.trials and n < args.trials:
            flag = f"  ⚠ {n}/{args.trials}"
        print(f"{arm:14} " + " ".join(cells) + f"   {n}{flag}")

    t_auto = auto("taught")
    c_auto = auto("scaffold_only")
    d_auto = auto("drive_only")

    print("\n## Verdict (autonomous acts self_orient_rate)")
    learned = t_auto >= LEARNED_MIN and (t_auto - c_auto) >= LEARNED_MARGIN
    beats_drive = (t_auto - d_auto) >= DRIVE_MARGIN
    print(f"  taught={t_auto:.3f}  scaffold_only={c_auto:.3f}  drive_only={d_auto:.3f}")
    print(f"  LEARNED (taught ≥ {LEARNED_MIN} and ≥ scaffold_only+{LEARNED_MARGIN}): {'PASS' if learned else 'FAIL'}")
    print(f"  TAUGHT>DRIVE (taught ≥ drive_only+{DRIVE_MARGIN}): {'PASS' if beats_drive else 'FAIL'}")

    if learned and beats_drive:
        print("\n**GRADUATE — the infant learned to orient toward the voice as the scaffold faded.**")
        return 0
    if not any(arms.get("taught", {}).get(a) for a in AUTONOMOUS_ACTS):
        print("\n**VOID — no taught autonomous-act data.**")
        return 4
    print("\n**NOT GRADUATED — see the per-arm curve above.**")
    return 1


if __name__ == "__main__":
    sys.exit(main())
