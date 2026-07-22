#!/usr/bin/env python
"""Analyze the cradle-mother OPERANT orient results (benchmark_cradle_mother.py).

The claim: a hungry infant with NO intrinsic orient drive learns to orient toward
a sound PURELY because a mother feeds it (operant credit) when its own turn moved
toward the sound. As the session proceeds, the TAUGHT infant's DIRECTEDNESS (turns
that moved toward the sound) RISES to "learned", while the NO_FEED control (mother
places the sound but never feeds/credits) stays at chance — it has no teacher.

Metric: per time-bin ("act"), ``directedness`` = fraction of turns the infant
turned TOWARD the sound (progress > 0). Logged in both arms.

Verdict:
  LEARNED       : taught LATE-bin directedness ≥ 0.65 AND rose from the EARLY bin
                  by ≥ 0.15 (it learned over the session, not innate — with the
                  drive removed, early should sit near chance).
  MOTHER-TAUGHT : taught late ≥ no_feed late + 0.20 (the mother is WHY — remove
                  her and the infant stays at chance).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ACT_ORDER = ("act1_early", "act2_warming", "act3_consolidating", "act4_autonomous")
EARLY_ACTS = ("act1_early", "act2_warming")
LATE_ACTS = ("act3_consolidating", "act4_autonomous")
LEARNED_MIN = 0.65
RISE_MARGIN = 0.15
MOTHER_MARGIN = 0.20


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

    # per arm: per-act list of directedness
    arms: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        arm = r["arm"]
        acts = arms.setdefault(arm, {})
        for act, m in (r.get("fade") or {}).items():
            acts.setdefault(act, []).append(float(m.get("directedness", 0.0)))

    def pooled(arm: str, act_names: tuple[str, ...]) -> float:
        vals: list[float] = []
        for a in act_names:
            vals += arms.get(arm, {}).get(a, [])
        return _mean(vals)

    print("## Cradle-mother operant learning curve (directedness)\n")
    print(f"{'arm':10} " + " ".join(f"{a.split('_')[1][:6]:>7}" for a in ACT_ORDER) + "    seeds")
    print("-" * 56)
    for arm in ("taught", "no_feed"):
        if arm not in arms:
            continue
        cells = []
        for a in ACT_ORDER:
            xs = arms[arm].get(a, [])
            cells.append(f"{_mean(xs):7.2f}" if xs else "    -- ")
        n = max((len(v) for v in arms[arm].values()), default=0)
        flag = f"  ⚠ {n}/{args.trials}" if (args.trials and n < args.trials) else ""
        print(f"{arm:10} " + " ".join(cells) + f"    {n}{flag}")

    t_early = pooled("taught", EARLY_ACTS)
    t_late = pooled("taught", LATE_ACTS)
    n_late = pooled("no_feed", LATE_ACTS)

    print("\n## Verdict")
    print(f"  taught: early={t_early:.3f} late={t_late:.3f}   no_feed late={n_late:.3f}")
    learned = t_late >= LEARNED_MIN and (t_late - t_early) >= RISE_MARGIN
    mother = (t_late - n_late) >= MOTHER_MARGIN
    print(f"  LEARNED (taught late ≥ {LEARNED_MIN} and rose ≥ {RISE_MARGIN}): {'PASS' if learned else 'FAIL'}")
    print(f"  MOTHER-TAUGHT (taught late ≥ no_feed late + {MOTHER_MARGIN}): {'PASS' if mother else 'FAIL'}")

    if learned and mother:
        print("\n**GRADUATE — the infant learned to orient toward the mother's voice, taught by her feeding alone.**")
        return 0
    if not any(arms.get("taught", {}).get(a) for a in LATE_ACTS):
        print("\n**VOID — no taught late-act data.**")
        return 4
    print("\n**NOT GRADUATED — see the curve above.**")
    return 1


if __name__ == "__main__":
    sys.exit(main())
