#!/usr/bin/env python3
"""Apparatus diagnostic for cradle_mother runs — the no-move / thrashing rate.

Usage:  python nomove_rate.py <workdir> [<workdir> ...]

Decomposes the Exp 48 directedness denominator into toward / away / NO-MOVE,
and counts turn_left vs turn_right executions per mother-turn (the thrashing
signature).

WHY THIS EXISTS: the 1.1 heartbeat re-run of Exp 48 failed its LEARNED gate
(rise +0.079 vs +0.15 required). The cause is not the teaching signal —
`fed_rate` still tracks `directedness` and the MOTHER-TAUGHT gate passes — but
a ~20% `progress == 0` floor that caps directedness at ~0.80, produced by the
infant firing ~31 turn_left AND ~31 turn_right per mother-turn (net ≈ 0).

Reference measurement at f05c63aa (12 taught seeds, 2026-08-10):
    act1 56.4% toward / 18.2% away / 25.4% no-move
    act2 73.1 / 7.8 / 19.1
    act3 73.6 / 7.2 / 19.2
    act4 69.8 / 9.9 / 20.3
    67.9 turn-actions per mother-turn, left/right imbalance 0.6%
    OVERALL NO-MOVE 21.0%
(The raw logs behind those numbers were lost to a /tmp wipe on reboot —
which is exactly standard S4 in docs/plans/simulation_apparatus_standards.md.
Keep run workdirs OFF /tmp.)

Candidate for promotion to the S2 canary.
"""

import collections
import glob
import json
import os
import re
import sys

ACTS = ["act1_early", "act2_warming", "act3_consolidating", "act4_autonomous"]


def analyze_run(run_dir: str) -> tuple[dict, collections.Counter]:
    per_act = {a: collections.Counter() for a in ACTS}
    actions: collections.Counter = collections.Counter()
    log = os.path.join(run_dir, "mother_log.jsonl")
    if not os.path.exists(log):
        return per_act, actions
    for line in open(log, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        e = d.get("e")
        if e == "sim_mother":
            f = dict(re.findall(r"(\w+)=(\S+)", d.get("message", "")))
            act, prog = f.get("act"), f.get("progress")
            if act not in per_act or prog in (None, "None"):
                continue
            v = float(prog)
            per_act[act]["n"] += 1
            if v > 1e-9:
                per_act[act]["toward"] += 1
            elif v < -1e-9:
                per_act[act]["away"] += 1
            else:
                per_act[act]["no_move"] += 1
        elif e == "sim_exec":
            m = d.get("message", "")
            if "Completed:" in m and "turn_left" in m:
                actions["turn_left"] += 1
            elif "Completed:" in m and "turn_right" in m:
                actions["turn_right"] += 1
    return per_act, actions


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for target in argv:
        runs = sorted(glob.glob(os.path.join(target, "taught_seed*"))) or [target]
        tot = {a: collections.Counter() for a in ACTS}
        acts_all: collections.Counter = collections.Counter()
        n_runs = 0
        for r in runs:
            pa, ac = analyze_run(r)
            if not any(pa[a]["n"] for a in ACTS):
                continue
            n_runs += 1
            for a in ACTS:
                tot[a].update(pa[a])
            acts_all.update(ac)
        print(f"\n=== {target}   ({n_runs} taught run(s))")
        if not n_runs:
            print("    no parsable mother records found")
            continue
        print("    act                    n   toward%   away%   NO-MOVE%")
        for a in ACTS:
            c = tot[a]
            n = c["n"] or 1
            print(
                f"    {a:20} {c['n']:4d}   {100 * c['toward'] / n:5.1f}   "
                f"{100 * c['away'] / n:5.1f}    {100 * c['no_move'] / n:5.1f}"
            )
        turns = sum(tot[a]["n"] for a in ACTS) or 1
        L, R = acts_all["turn_left"], acts_all["turn_right"]
        print(
            f"    turn actions: left={L} right={R}  "
            f"({(L + R) / turns:.1f} per mother-turn, imbalance {abs(L - R) / max(1, L + R):.1%})"
        )
        overall = sum(tot[a]["no_move"] for a in ACTS) / turns
        print(f"    OVERALL NO-MOVE RATE: {100 * overall:.1f}%   (f05c63aa reference: 21.0%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
