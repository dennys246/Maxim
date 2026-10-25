#!/usr/bin/env python3
"""The BF2 obligation lint: a floor-matched B1 verdict obliges the Exp 60 row to sit at `PARTIAL (narrow: …)` or below.

Owner rule BF2 (docs/plans/baseline_floors.md): a claim that does not beat its floor takes `PARTIAL (narrow: floor
matched)`. The evidence gate (scripts/lint_evidence_gate.py) gates MOVES; nothing there forces the demotion, and an
ordinary Exp 60 re-run could re-earn EARNED over a floor-matched B1 verdict. This lint closes that for T1-13 (owner
decision D1, 2026-10-11; a partial mechanization of backlog row M46). The obligation is pass-table data (`obliges`
on the `b1_verdict` entry), read from the MERGE-BASE like the gate's own rule data.
Spec: docs/plans/archive/baseline_floors_b1_evidence.md §3.

Rules, judged at HEAD against the merge-base:
- B1 verdicts are append-only by identity (data_sha256, verdict value): every pair on the merge-base is still present
  under the data root at HEAD (a move keeps it; no B1 retirement route exists, backlog M50).
- Counted verdicts: FLOOR_MATCHED_* whatever their status (fail closed); FLOORS_BEATEN only when the gate's own judge
  calls it ESTABLISHED with prereg PASS; INCOMPLETE dropped. Grouped by data_sha256: differing counted values over one
  data file fail. Each group's data time is the latest `ts` over the harness rows of its bound bytes (sha-checked by
  the gate's shared helpers); unreadable fails. Newest by that float; an exact tie between data files fails.
- If the newest counted verdict obliges, the row's HEAD status is rank < 2, or PARTIAL with qualifier head `narrow`
  and a date on or after the verdict's data time (UTC day).

    python scripts/lint_bf2_obligation.py          # CI lint job
    python scripts/lint_bf2_obligation.py --base <sha>

Exits: 0 clean; 1 violations; 2 the base could not be read where a skip is not allowed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import _ledger as L  # noqa: E402
import _lint_git  # noqa: E402
from _evidence_records import (  # noqa: E402
    DATA_ROOT,
    ESTABLISHED,
    MALFORMED,
    GateError,
    Repo,
    decompressed,
    json_lines,
    kind_of,
    verdict_data_bytes,
    verdict_data_path_problem,
)
from lint_evidence_gate import build_ctx, judge_entry  # noqa: E402

REPO_ROOT = SCRIPTS.parent


def obligations(table: dict) -> dict[str, tuple[str, frozenset, frozenset]]:
    """verdict kind -> (the row it obliges, the obliging values, the clearing values), from the pass table's `obliges`
    blocks. Called with the MERGE-BASE table (like the gate's own rule data), so a PR cannot loosen its own
    obligation; a malformed block is the evidence gate's to fail (`obliges_problem`)."""
    out = {}
    for kind, entry in (table or {}).items():
        ob = entry.get("obliges") if isinstance(entry, dict) else None
        if isinstance(ob, dict) and isinstance(ob.get("row"), str):
            out[kind] = (ob["row"], frozenset(ob.get("values") or ()), frozenset(ob.get("clears") or ()))
    return out


def verdicts_at(repo: Repo, ref: str, kind: str) -> dict[str, dict]:
    """Every tracked verdict record of ``kind`` under the data root at ``ref``: path -> record."""
    out = {}
    for path, (mode, _sha) in repo.tree(ref).items():
        if not path.startswith(DATA_ROOT + "/") or not path.endswith(".json") or mode == "120000":
            continue
        try:
            rec = json.loads(repo.blob(ref, path) or b"")
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("record_kind") == "verdict" and rec.get("kind") == kind:
            out[path] = rec
    return out


def data_time(rec: dict, repo: Repo, ref: str, retired: dict) -> tuple[float | None, str | None]:
    """The latest harness-row `ts` in the verdict's bound bytes (the gate's own path + sha binding), or why not."""
    problem = verdict_data_path_problem(rec, retired)
    if problem is not None:
        return None, problem
    data, problem = verdict_data_bytes(rec, repo, ref)
    if data is None:
        return None, problem
    try:
        lines = json_lines(decompressed(rec["data"], data))
    except (GateError, *MALFORMED) as exc:  # a truncated or corrupt committed file: unreadable, reported, never a crash
        return None, f"verdict data {rec['data']} cannot be parsed ({type(exc).__name__}: {exc})"
    stamps = [
        r["ts"]
        for r in lines
        if kind_of(r) == "harness_row" and isinstance(r.get("ts"), (int, float)) and not isinstance(r["ts"], bool)
    ]
    if not stamps:
        return None, f"verdict data {rec['data']} holds no harness row with a numeric ts"
    return float(max(stamps)), None


def obligation_problems(repo: Repo, base: str, *, ctx=None) -> list[str]:
    """Every BF2 obligation violation at HEAD against ``base``."""
    if ctx is None:
        ctx, _ = build_ctx(repo, base)
    failures: list[str] = []
    rows = {r.id: r for r in L.parse((repo.blob("HEAD", L.LEDGER_PATH) or b"").decode("utf-8"))[0]}
    for kind, (row_id, obliging, clearing) in obligations(ctx.table).items():
        head = verdicts_at(repo, "HEAD", kind)
        base_pairs = {(r.get("data_sha256"), r.get("verdict")) for r in verdicts_at(repo, base, kind).values()}
        head_pairs = {(r.get("data_sha256"), r.get("verdict")) for r in head.values()}
        for sha, value in sorted(base_pairs - head_pairs, key=repr):
            failures.append(
                f"{kind}: the verdict {value!r} over data {str(sha)[:12]} on main is gone at HEAD (B1 verdicts are "
                "append-only by (data_sha256, value); no retirement route exists, backlog M50)"
            )
        groups: dict[str, set] = {}
        timed: dict[str, float] = {}
        for path, rec in sorted(head.items()):
            value = rec.get("verdict")
            if value in clearing:
                j = judge_entry(path, ctx)
                if j.status != ESTABLISHED or j.prereg != "PASS" or j.data_prereg != "PASS":
                    continue  # a clearing verdict is support: it counts only when judged established
            elif value not in obliging:
                continue  # INCOMPLETE and anything else neither oblige nor clear
            sha = str(rec.get("data_sha256"))
            groups.setdefault(sha, set()).add(value)
            when, problem = data_time(rec, repo, "HEAD", ctx.retired)
            if when is None:
                failures.append(f"{kind}: {path}: its data time cannot be read ({problem})")
                continue
            timed[sha] = when
        for sha, values in groups.items():
            if len(values) > 1:
                failures.append(f"{kind}: data {sha[:12]} carries differing verdicts {sorted(values)}")
        if failures or not timed:
            continue
        newest = max(timed.values())
        top = [sha for sha, t in timed.items() if t == newest]
        if len(top) > 1:
            failures.append(f"{kind}: {len(top)} data files tie for the newest data time ({newest})")
            continue
        (value,) = groups[top[0]]
        if value not in obliging:
            continue  # the newest counted verdict clears
        row = rows.get(row_id)
        if row is None:
            failures.append(f"{kind}: obliges {row_id}, which is not in the ledger")
            continue
        due = dt.datetime.fromtimestamp(newest, tz=dt.timezone.utc).date().isoformat()
        low = L.RANK.get(row.token or "", 0) < 2
        narrow = (
            row.token == "PARTIAL"
            and not row.qualifier_unparsed
            and row.qualifier is not None
            and L.qualifier_head(row.qualifier) == "narrow"
            and row.date is not None
            and row.date >= due
        )
        if not (low or narrow):
            failures.append(
                f"{row_id} is {row.token} ({row.qualifier or 'unqualified'}) dated {row.date}, but the newest {kind} "
                f"({value}, data {top[0][:12]}) obliges it to sit below PARTIAL, or at PARTIAL (narrow: …) dated on "
                f"or after {due} (owner rule BF2)"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="the merge-base to compare against (default: as the evidence gate)")
    args = ap.parse_args(argv)
    repo = Repo(REPO_ROOT)
    try:
        base = args.base or _lint_git.base_ref(repo.root)
        failures = obligation_problems(repo, base)
    except _lint_git.GitUnavailable as exc:
        if _lint_git.must_not_skip(f"BF2 obligation: {exc}"):
            return 2
        print(f"SKIP BF2 obligation (no merge-base: {exc})", file=sys.stderr)
        return 0
    except GateError as exc:
        print(f"BF2 obligation FAILED: {exc}", file=sys.stderr)
        return 1
    if failures:
        print(f"BF2 obligation FAILED ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("BF2 obligation: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
