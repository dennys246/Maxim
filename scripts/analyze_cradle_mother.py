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
# Gate v2 (re-pre-registered 2026-08-14, frozen pre-data — 48_cradle_mother_seam.md
# §Gate v2): EARLY is act1 ONLY. The v1 mean(act1, act2) folded already-learned
# act2 behavior into the baseline (heartbeat bin-alignment finding).
EARLY_ACTS = ("act1_early",)
LATE_ACTS = ("act3_consolidating", "act4_autonomous")
LEARNED_MIN = 0.65
RISE_MARGIN = 0.15
MOTHER_MARGIN = 0.20
# S7 ceiling clause: early at/above LEARNED_MIN makes the rise criterion
# structurally unattainable — report LEARNED-AT-CEILING (require late-level +
# non-degradation), never a silent pass or fail.
CEILING_MIN = 0.65
CEILING_DEGRADE_TOL = 0.05
# S5 exposure contract: flag a >20% mean-turns mismatch between arms.
EXPOSURE_TOL = 0.20


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

    # per arm: per-act list of directedness (+ per-row total turns for the
    # S5 exposure contract)
    arms: dict[str, dict[str, list[float]]] = {}
    arm_turns: dict[str, list[float]] = {}
    for r in rows:
        arm = r["arm"]
        acts = arms.setdefault(arm, {})
        row_turns = 0.0
        for act, m in (r.get("fade") or {}).items():
            acts.setdefault(act, []).append(float(m.get("directedness", 0.0)))
            row_turns += float(m.get("turns", 0) or 0)
        arm_turns.setdefault(arm, []).append(row_turns)

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
    # A missing control arm must be a VOID, never a PASS: _mean([]) is 0.0,
    # so a single-arm run used to report MOTHER-TAUGHT: PASS against a
    # control that never ran (found during the Exp 48 heartbeat
    # investigation — the analyzer passed vacuously on absent data).
    has_nofeed_late = any(arms.get("no_feed", {}).get(a) for a in LATE_ACTS)
    n_late = pooled("no_feed", LATE_ACTS) if has_nofeed_late else None

    print("\n## Verdict (gate v2 — 48_cradle_mother_seam.md, frozen 2026-08-14)")
    n_late_str = f"{n_late:.3f}" if n_late is not None else "-- (arm absent)"
    print(f"  taught: early={t_early:.3f} late={t_late:.3f}   no_feed late={n_late_str}")

    # S5 exposure contract: report mean recorded turns per arm; flag >20% skew.
    exposure_flagged = False
    t_turns = _mean(arm_turns.get("taught", []))
    n_turns = _mean(arm_turns.get("no_feed", []))
    if t_turns and n_turns:
        skew = abs(t_turns - n_turns) / max(t_turns, n_turns)
        flag = f"  ⚠ EXPOSURE-FLAG (skew {skew:.0%} > {EXPOSURE_TOL:.0%})" if skew > EXPOSURE_TOL else ""
        exposure_flagged = bool(flag)
        print(f"  exposure: taught {t_turns:.0f} turns/seed, no_feed {n_turns:.0f} turns/seed{flag}")

    at_ceiling = t_early >= CEILING_MIN
    if at_ceiling:
        learned = t_late >= LEARNED_MIN and t_late >= (t_early - CEILING_DEGRADE_TOL)
        print(
            f"  LEARNED-AT-CEILING (early {t_early:.3f} ≥ {CEILING_MIN} — rise unattainable; "
            f"late ≥ {LEARNED_MIN} and non-degrading): {'PASS' if learned else 'FAIL'} "
            "— teaching claim rests on MOTHER-TAUGHT"
        )
    else:
        learned = t_late >= LEARNED_MIN and (t_late - t_early) >= RISE_MARGIN
        print(
            f"  LEARNED (taught late ≥ {LEARNED_MIN} and rose ≥ {RISE_MARGIN} from act1): {'PASS' if learned else 'FAIL'}"
        )
    if has_nofeed_late:
        mother = (t_late - n_late) >= MOTHER_MARGIN
        print(f"  MOTHER-TAUGHT (taught late ≥ no_feed late + {MOTHER_MARGIN}): {'PASS' if mother else 'FAIL'}")
    else:
        mother = False
        print(
            f"  MOTHER-TAUGHT (taught late ≥ no_feed late + {MOTHER_MARGIN}): "
            "VOID — no no_feed rows in the input; a single-arm run cannot pass this gate"
        )

    if learned and mother and exposure_flagged:
        print(
            "\n**EXPOSURE-FLAGGED — both gates pass but the arms are exposure-skewed"
            " (>20% turns mismatch); resolve the skew before recording a verdict (S5).**"
        )
        return 7
    if learned and mother:
        print("\n**GRADUATE — the infant learned to orient toward the mother's voice, taught by her feeding alone.**")
        return 0
    if not any(arms.get("taught", {}).get(a) for a in LATE_ACTS):
        print("\n**VOID — no taught late-act data.**")
        return 4
    if learned and not has_nofeed_late:
        print(
            "\n**INCOMPLETE — the taught arm cleared LEARNED but the no_feed"
            " control never ran; run both arms before recording a verdict.**"
        )
        return 5
    print("\n**NOT GRADUATED — see the curve above.**")
    return 1


if __name__ == "__main__":
    sys.exit(main())
