#!/usr/bin/env python
"""Exp 54 — magnitude choice by |stimulus| bin (REPORTED, NOT GATED).

The pre-registration's Phase A promises "the fraction of big vs normal turns by
|stimulus| bin" (Exp 45c/d predict big-at-far, normal-at-near if the credit resolves
it; a coarse 3-bin representation may not). Neither benchmark_cradle_mother.py nor
analyze_cradle_mother.py reads tool names, so this is the producer (amendment 1
item 8): post hoc, from each run's archived ``mother_log.jsonl`` in the Phase A
workdir — the ``act=… az_stimulus=…`` mother record placed THIS turn, paired with
the ``Executing: <tool> by …`` lines that follow before the next mother record.

    python scripts/analyze_exp54_magnitude.py --workdir ~/exp54/phaseA [--out 54_magnitude.json]

Reports per arm × |stimulus| (the arc's six values) and per 3-bin roll-up
(near 0.4–0.5 / mid 0.6–0.7 / far 0.8–0.9), all acts and late acts (act3+act4)
separately, plus the leftward/rightward direction split as a sanity check. A
body with no ``_big`` affordances (Exp 52's) reports big = 0 everywhere — the
harness check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LATE_ACTS = ("act3_consolidating", "act4_autonomous")
BINS = (("near", 0.4, 0.5), ("mid", 0.6, 0.7), ("far", 0.8, 0.9))
_EXEC = re.compile(r"^Executing: (\S+) by ")


def _mother(msg: str) -> dict | None:
    if not msg.startswith("act="):
        return None
    out: dict = {}
    for tok in msg.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out if "act" in out else None


def pair_turns(log_path: Path) -> list[dict]:
    """[{act, stim, tool, big, left}] — one row per infant action, keyed to the stimulus
    the mother placed on the turn the action answered."""
    rows: list[dict] = []
    current: dict | None = None
    for line in log_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = str(rec.get("message", ""))
        m = _mother(msg)
        if m is not None:
            stim = m.get("az_stimulus")
            try:
                current = {"act": m["act"], "stim": float(stim)} if stim not in (None, "None", "") else None
            except ValueError:
                current = None
            continue
        ex = _EXEC.match(msg)
        if ex is None or current is None:
            continue
        tool = ex.group(1)
        if "_turn_" not in tool:
            continue
        rows.append(
            {
                "act": current["act"],
                "stim": current["stim"],
                "tool": tool,
                "big": tool.endswith("_big"),
                "left": "_turn_left" in tool,
            }
        )
    return rows


def _bin(abs_stim: float) -> str | None:
    for name, lo, hi in BINS:
        if lo - 1e-9 <= abs_stim <= hi + 1e-9:
            return name
    return None


def summarize(rows: list[dict]) -> dict:
    def table(sel: list[dict]) -> dict:
        by_stim: dict[str, dict] = {}
        by_bin: dict[str, dict] = {}
        for r in sel:
            a = round(abs(r["stim"]), 2)
            for key, store in ((f"{a:.1f}", by_stim), (_bin(a), by_bin)):
                if key is None:
                    continue
                e = store.setdefault(key, {"n": 0, "big": 0, "toward": 0})
                e["n"] += 1
                e["big"] += int(r["big"])
                e["toward"] += int(r["left"] == (r["stim"] < 0))
        for store in (by_stim, by_bin):
            for e in store.values():
                e["big_frac"] = round(e["big"] / e["n"], 3) if e["n"] else None
                e["toward_frac"] = round(e["toward"] / e["n"], 3) if e["n"] else None
        return {"by_abs_stimulus": dict(sorted(by_stim.items())), "by_bin": by_bin, "n": len(sel)}

    return {"all_acts": table(rows), "late_acts": table([r for r in rows if r["act"] in LATE_ACTS])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", required=True, help="Phase A workdir: <arm>_seed<n>_ew<w>/mother_log.jsonl per run")
    ap.add_argument("--out", default=None, help="write the JSON report here (default: print only)")
    args = ap.parse_args()
    workdir = Path(args.workdir).expanduser()
    per_arm: dict[str, list[dict]] = {}
    runs = 0
    for log in sorted(workdir.glob("*_seed*_ew*/mother_log.jsonl")):
        arm = log.parent.name.split("_seed", 1)[0]
        rows = pair_turns(log)
        runs += 1
        per_arm.setdefault(arm, []).extend(rows)
    if not runs:
        print(f"no */mother_log.jsonl under {workdir}", file=sys.stderr)
        return 2
    report = {"_format_version": "1.0", "workdir": str(workdir), "runs": runs, "arms": {}}
    for arm, rows in sorted(per_arm.items()):
        report["arms"][arm] = summarize(rows)
        late = report["arms"][arm]["late_acts"]["by_bin"]
        print(f"## {arm}: {len(rows)} turn actions over {runs} runs")
        print(
            "  late acts, big fraction by |stimulus| bin: "
            + "  ".join(f"{b}={late[b]['big_frac']} (n={late[b]['n']})" for b in ("near", "mid", "far") if b in late)
        )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
