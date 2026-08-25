#!/usr/bin/env python
"""Analyze the cradle-mother OPERANT orient results (benchmark_cradle_mother.py).

The claim: a hungry infant with NO intrinsic orient drive learns to orient toward
a sound PURELY because a mother feeds it (operant credit) when its own turn moved
toward the sound. As the session proceeds, the TAUGHT infant's DIRECTEDNESS (turns
that moved toward the sound) RISES to "learned", while the NO_FEED control (mother
places the sound but never feeds/credits) stays at chance — it has no teacher.

Metric: per time-bin ("act"), ``directedness`` = fraction of turns the infant
turned TOWARD the sound (progress > 0). Logged in both arms.

Verdict (GATE V2 — re-pre-registered 2026-08-14, frozen pre-data; see
48_cradle_mother_seam.md §Gate v2. Do NOT retune post-hoc):
  LEARNED       : taught LATE-bin (act3+act4) directedness ≥ 0.65 AND rose
                  ≥ 0.15 from the EARLY bin (act1 ONLY — v1's mean(act1,act2)
                  folded learned act2 behavior into the baseline).
  CEILING (S7)  : early ≥ 0.65 makes the rise unattainable → LEARNED-AT-CEILING
                  (late ≥ 0.65 + non-degradation), reported explicitly; the
                  teaching claim then rests on MOTHER-TAUGHT.
  MOTHER-TAUGHT : taught late ≥ no_feed late + 0.20 (the mother is WHY — remove
                  her and the infant stays at chance). VOID if the control
                  never ran; exposure skew >20% turns → EXPOSURE-FLAG (exit 7).

Gate V3 (``--gate v3``; Exp 52 Nurture, pre-registered 2026-08-25 — see
exp52_nurture_preregistration.md §Phase B; v2 constants carried, NOT retuned):
  HUNGER-NECESSARY : taught late ≥ satiated late + 0.20 AND the satiated arm's
                     rise (late − act1) < 0.15 AND satiated late ≤ no_feed late + 0.20
                     (amendment 2: a cap, mirroring Phase A's). The satiated arm is fed on the same
                     contingency but is never hungry → relief-sourced credit mints
                     nothing; if it still learns, the credit is not coming from
                     relief. VOID if the arm never ran.
  APPARATUS (L2)   : every arm's per-seed late-bin directedness must SPREAD across
                     seeds (population SD > 0 with ≥ 3 seeds). Seed-invariant exact
                     fractions are the v2 phase-lock signature; if the shuffle did
                     not break it — or an arm has < 3 seeds so the check cannot
                     run — no science verdict is issued (exit 8).
  APPARATUS (S3)   : the pre-registration's in-sim assertions, from the per-turn
                     mother telemetry: satiated credited_rate == 0; no negative
                     reward; no credit without relief. Any violation → exit 8.
  Outcomes: satiated arm absent → INCOMPLETE (exit 5, like a missing no_feed);
  LEARNED + MOTHER-TAUGHT pass but HUNGER-NECESSARY fails → HUNGER-LEAK
  (exit 9): the credit is not coming from relief — apparatus, not a result.
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
# Gate v3 (Exp 52, frozen 2026-08-25): the satiated-arm gate + the L2 apparatus check.
HUNGER_MARGIN = 0.20
SATIATED_RISE_MAX = 0.15
# Amendment 2 (pre-data, structural): the satiated arm must also be indistinguishable
# from the teacherless control — a cap, as Phase A has (satiated ≤ 0.60).
SATIATED_CAP_MARGIN = 0.20
SEED_SPREAD_MIN_N = 3


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pstdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _per_seed_late(rows: list[dict], arm: str) -> list[float]:
    """Late-bin (act3+act4) directedness per ROW (= per seed) for one arm."""
    out: list[float] = []
    for r in rows:
        if r.get("arm") != arm:
            continue
        fade = r.get("fade") or {}
        vals = [float(fade[a].get("directedness", 0.0)) for a in LATE_ACTS if a in fade]
        if vals:
            out.append(_mean(vals))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--trials", type=int, default=None, help="expected seeds/arm (warns if short)")
    p.add_argument(
        "--gate",
        default="v2",
        choices=["v2", "v3"],
        help="v2 (Exp 48) or v3 (Exp 52: + HUNGER-NECESSARY + L2 apparatus)",
    )
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
    for arm in ("taught", "satiated", "no_feed"):
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
    for other in ("no_feed", "satiated"):
        o_turns = _mean(arm_turns.get(other, []))
        if t_turns and o_turns:
            skew = abs(t_turns - o_turns) / max(t_turns, o_turns)
            flag = f"  ⚠ EXPOSURE-FLAG (skew {skew:.0%} > {EXPOSURE_TOL:.0%})" if skew > EXPOSURE_TOL else ""
            exposure_flagged = exposure_flagged or bool(flag)
            print(f"  exposure: taught {t_turns:.0f} turns/seed, {other} {o_turns:.0f} turns/seed{flag}")

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

    v3_ok = True
    if args.gate == "v3":
        print("\n## Gate v3 additions (Exp 52 — exp52_nurture_preregistration.md, frozen 2026-08-25)")
        # APPARATUS (L2): per-seed late-bin spread per arm.
        for arm in ("taught", "satiated", "no_feed"):
            if arm not in arms:
                continue
            per_seed = _per_seed_late(rows, arm)
            if len(per_seed) >= SEED_SPREAD_MIN_N:
                spread = _pstdev(per_seed)
                distinct = len({round(v, 6) for v in per_seed})
                flag = "" if spread > 0.0 else "  ✗ SEED-INVARIANT (L2 phase-lock signature)"
                print(
                    f"  APPARATUS {arm:9} per-seed late SD = {spread:.3f} over {len(per_seed)} seeds "
                    f"({distinct} distinct value{'s' if distinct != 1 else ''}){flag}"
                )
                if spread <= 0.0:
                    v3_ok = False
            else:
                print(
                    f"  APPARATUS {arm:9} SKIPPED — {len(per_seed)} seed(s) < {SEED_SPREAD_MIN_N}: "
                    "the L2 check cannot run, so no science verdict can be issued"
                )
                v3_ok = False
        # S3 in-sim assertions, from the per-turn mother telemetry the harness
        # aggregates (satiated never credited; no negative reward; no credit
        # without relief). Missing keys (old rows / mock) count as 0.
        s3_viol: list[str] = []
        for r in rows:
            for act, m in (r.get("fade") or {}).items():
                if r.get("arm") == "satiated" and float(m.get("credited_rate", 0.0) or 0.0) > 0.0:
                    s3_viol.append(f"satiated seed {r.get('seed')} {act}: credited_rate {m['credited_rate']:.2f} > 0")
                if int(m.get("neg_reward", 0) or 0) > 0:
                    s3_viol.append(f"{r.get('arm')} seed {r.get('seed')} {act}: {m['neg_reward']} negative reward(s)")
                if int(m.get("credited_no_relief", 0) or 0) > 0:
                    s3_viol.append(
                        f"{r.get('arm')} seed {r.get('seed')} {act}: {m['credited_no_relief']} credit(s) without relief"
                    )
        if s3_viol:
            v3_ok = False
            print("  APPARATUS S3 assertions VIOLATED:")
            for v in s3_viol[:10]:
                print(f"    - {v}")
        else:
            print(
                "  APPARATUS S3 assertions: OK (satiated never credited; no negative reward; no credit without relief)"
            )
        has_sat_late = any(arms.get("satiated", {}).get(a) for a in LATE_ACTS)
        has_sat_early = any(arms.get("satiated", {}).get(a) for a in EARLY_ACTS)
        if has_sat_late and has_sat_early and has_nofeed_late:
            s_late = pooled("satiated", LATE_ACTS)
            s_early = pooled("satiated", EARLY_ACTS)
            capped = s_late <= (n_late + SATIATED_CAP_MARGIN)
            hunger = (t_late - s_late) >= HUNGER_MARGIN and (s_late - s_early) < SATIATED_RISE_MAX and capped
            print(
                f"  HUNGER-NECESSARY (taught late {t_late:.3f} ≥ satiated late {s_late:.3f} + {HUNGER_MARGIN}; "
                f"satiated rise {s_late - s_early:+.3f} < {SATIATED_RISE_MAX}; "
                f"satiated late ≤ no_feed late {n_late:.3f} + {SATIATED_CAP_MARGIN}): {'PASS' if hunger else 'FAIL'}"
            )
        elif has_sat_late and not has_sat_early:
            hunger = False
            has_sat_late = False  # act1 missing → the rise term is undefined: treat as absent (VOID)
            print("  HUNGER-NECESSARY: VOID — satiated rows lack act1 (the rise term is undefined)")
        else:
            hunger = False
            print(f"  HUNGER-NECESSARY (taught late ≥ satiated late + {HUNGER_MARGIN}): VOID — no satiated rows")
        if not v3_ok:
            print(
                "\n**APPARATUS — the L2 seed-spread check failed or could not run, or an S3 in-sim "
                "assertion was violated. No science verdict is issued (exit 8).**"
            )
            return 8
        if not has_sat_late:
            print(
                "\n**INCOMPLETE — the satiated control never ran; gate v3 needs taught, no_feed AND "
                "satiated before recording a verdict.**"
            )
            return 5
        if learned and mother and not hunger:
            print(
                "\n**HUNGER-LEAK — LEARNED and MOTHER-TAUGHT pass but the never-hungry infant learned "
                "too: the credit is not coming from relief. Apparatus, not a result (exit 9) — find the "
                "non-relief credit source before any re-run (pre-registration outcome tree).**"
            )
            return 9
        mother = mother and hunger

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
