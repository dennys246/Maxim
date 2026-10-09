#!/usr/bin/env python3
"""One source of truth for claims (roadmap 1.3.2 item 8, mechanization backlog M2).

The ledger (``docs/plans/behavioral_graduation_candidates.md``, parsed by ``scripts/_ledger.py``) is the single
source of a claim's status (owner decision 2026-10-04). Public surfaces cite a ledger row with a marker,
``<!-- claim: T<n>-<m> -->`` (an HTML comment: GitHub hides it, and PyPI's readme_renderer + nh3 drop it; verified by
rendering the README, 2026-10-04 review, not by a test), and this
lint checks that each cited surface says what the ledger says. Linted, not generated: the prose stays curated.

The motivating errors (2026-09-27 score cards): the README's Exp 10 row said "re-run pending" after the re-run,
and an index cell can say "EARNED" while the ledger says "RE-VALIDATED". Every run, not diff-scoped:

**README** (the table under the exact header ``| Result | What was measured |``, found exactly once, with at
least one row):
- every row carries exactly one marker naming a ledger row (R1);
- it displays that row's ``<TOKEN> <YYYY-MM-DD>`` verbatim (R2), and no OTHER uppercase ledger token (a row
  cannot say "MAINTAINED … pending" or "EARNED" next to "RE-VALIDATED"; lowercase prose is free);
- when the ledger row has a scope qualifier, its head (text before the first ``:``/``;``/``,``) follows the date
  as a whole word (R3);
- a ``SUPERSEDED`` row names its successor's ID; ``DROPPED``, ``DORMANT``, ``TIER-2``, ``SETUP`` and
  ``BORDERLINE`` are barred from the table "what it has shown" (R4). ``STALE``/``BROKEN`` may stand, displayed
  (owner decision 2026-10-04: honest, and both already block the next release).

**Experiments index** (``docs/experiments/README.md``): a marker may sit only in a table whose header has a
``Status`` column, and that row's Status cell obeys R2–R3 and the successor rule (R5). Every Tier 1 row except
``DROPPED``/``TIER-2`` (and the reasoned ``NO_INDEX_ENTRY`` list) is cited by at least one index row (R6). Unmarked index rows are unconstrained.

**Both:** a malformed marker, two markers in one row, a marker outside its allowed tables (R7), or an unknown ID
fails. Fenced code blocks are skipped; a table runs until a blank line, as in GFM, so a row without a
leading ``|`` is still a row. A ledger parse problem fails the run.

Stated limits: the lint checks status, date and scope words, never the truth of the prose (the 2026-09-27
"accumulate" error was claim content, and this cannot see it); a displayed token elsewhere in the row passes
(catches forgetting, not evasion). Not yet covered (mechanization backlog): CHANGELOG release-claim lines,
CLAUDE.md "Active initiatives" / docs/index.md prose, Tier 3 completeness.

Regression guard: tests/unit/test_lint_claims_sync.py. Exits: 0 clean; 1 violations (stderr).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ledger as L  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
README = "README.md"
INDEX = "docs/experiments/README.md"
README_HEADER = ("Result", "What was measured")
BARRED_FROM_README = frozenset({"DROPPED", "DORMANT", "TIER-2", "SETUP", "BORDERLINE"})
EXEMPT_FROM_INDEX = frozenset({"DROPPED", "TIER-2"})
#: Tier 1 rows with no experiment entry to cite (R6), pinned to the ledger status they were granted at: when the row
#: moves, or an index row starts citing it, the exemption is stale and fails.
#: Empty since 2026-10-08: T1-5, the only exemption, was DROPPED (GL0 of the grounding line), and DROPPED rows are
#: exempt from R6 by rule.
NO_INDEX_ENTRY: dict[str, tuple[str, str, str]] = {}

_ANY_MARKER = re.compile(r"<!--\s*claim\b[^>]*-->")
_MARKER = re.compile(r"<!-- claim: (T\d-\d+) -->")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")


@dataclass
class Table:
    header: tuple[str, ...]
    rows: list[tuple[int, str, list[str]]] = field(default_factory=list)  # (1-based line, raw line, cells)


def scan(text: str) -> tuple[list[Table], list[tuple[int, str]]]:
    """GFM tables (outside fenced code) and every marker-shaped comment outside fenced code, with line numbers."""
    tables: list[Table] = []
    markers: list[tuple[int, str]] = []
    lines = text.split("\n")
    in_fence = False
    current: Table | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            in_fence = not in_fence
            current = None
            i += 1
            continue
        if in_fence:
            i += 1
            continue
        markers.extend((i + 1, m.group(0)) for m in _ANY_MARKER.finditer(line))
        stripped = line.strip()
        if current is not None and stripped:
            # GFM keeps a table open until a blank line: a row needs no leading `|` (executor review: a pipe-less row
            # rendered as a results row and escaped every rule).
            current.rows.append((i + 1, stripped, L.split_cells(stripped)))
            i += 1
            continue
        current = None
        if stripped.startswith("|") and i + 1 < len(lines) and _SEPARATOR.match(lines[i + 1].strip()):
            current = Table(tuple(L.split_cells(stripped)))
            tables.append(current)
            i += 2
            continue
        i += 1
    return tables, markers


def _norm(text: str) -> str:
    return " ".join(text.split())


def _token_re(token: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Z0-9-]){re.escape(token)}(?![A-Z0-9-])")


def display_problems(where: str, text: str, row: L.Row) -> list[str]:
    """R2, R3 and the successor rule for one cited cell or row."""
    out: list[str] = []
    norm = _norm(text)
    shown = f"{row.token} {row.date}"
    if shown not in norm:
        out.append(f"{where}: cites {row.id} but does not display its ledger status `{shown}`")
    for token in L.RANK:
        if token != row.token and _token_re(token).search(norm):
            out.append(f"{where}: shows `{token}`, but {row.id}'s ledger status is `{shown}`")
    if row.qualifier:
        head = L.qualifier_head(row.qualifier)
        pattern = rf"{re.escape(shown)}\W{{0,6}}{re.escape(head)}(?![\w-])"  # whole word: `narrow-ish` is not `narrow`
        if head and not re.search(pattern, norm, re.IGNORECASE):
            out.append(f"{where}: {row.id} is scoped `({head} …)` in the ledger; show `{head}` right after `{shown}`")
    if row.token == "SUPERSEDED" and row.superseded_by and not re.search(rf"\b{row.superseded_by}\b", norm):
        out.append(f"{where}: {row.id} is SUPERSEDED by {row.superseded_by}; name the successor")
    return out


def _markers_in(where: str, line: str) -> tuple[str | None, list[str]]:
    found = _ANY_MARKER.findall(line)
    ids = [m.group(1) for m in _MARKER.finditer(line)]
    problems = []
    if len(found) != len(ids):
        problems.append(f"{where}: malformed claim marker (the form is `<!-- claim: T1-13 -->`)")
    if len(ids) > 1:
        problems.append(f"{where}: {len(ids)} claim markers in one row; a row cites one ledger row")
    return (ids[0] if len(ids) == 1 else None), problems


def lint(readme: str, index: str, ledger: str) -> list[str]:
    rows, ledger_problems = L.parse(ledger)
    out = [f"{L.LEDGER_PATH}: {p}" for p in ledger_problems]
    by_id = {r.id: r for r in rows}

    def cited(where: str, rid: str) -> L.Row | None:
        row = by_id.get(rid)
        if row is None:
            out.append(f"{where}: cites {rid}, which the ledger does not have")
        elif row.token is None:
            out.append(f"{where}: cites {rid}, whose ledger status line does not parse")
            return None
        return row

    # ── README ───────────────────────────────────────────────────────────────────────────────────────────
    tables, markers = scan(readme)
    results = [t for t in tables if t.header == README_HEADER]
    allowed: set[int] = set()
    if len(results) != 1:
        out.append(f"{README}: expected exactly one `| Result | What was measured |` table, found {len(results)}")
    elif not results[0].rows:
        out.append(f"{README}: the results table has no rows")
    else:
        for ln, line, _cells in results[0].rows:
            where = f"{README}:{ln}"
            allowed.add(ln)
            rid, problems = _markers_in(where, line)
            out.extend(problems)
            if rid is None:
                if not problems:
                    out.append(f"{where}: a results row must cite its ledger row (`<!-- claim: T1-13 -->`)")
                continue
            row = cited(where, rid)
            if row is None:
                continue
            if row.token in BARRED_FROM_README:
                out.append(f"{where}: {rid} is `{row.token}` in the ledger; it does not belong under what it has shown")
            out.extend(display_problems(where, line, row))
    out.extend(f"{README}:{ln}: claim marker outside the results table" for ln, _m in markers if ln not in allowed)

    # ── experiments index ────────────────────────────────────────────────────────────────────────────────
    tables, markers = scan(index)
    allowed = set()
    cited_ids: set[str] = set()
    for t in tables:
        if "Status" not in t.header:
            continue
        col = t.header.index("Status")
        for ln, line, cells in t.rows:
            if not _ANY_MARKER.search(line):
                continue
            where = f"{INDEX}:{ln}"
            allowed.add(ln)
            rid, problems = _markers_in(where, line)
            out.extend(problems)
            if rid is None:
                continue
            cited_ids.add(rid)
            row = cited(where, rid)
            if row is not None:
                out.extend(display_problems(where + " (Status cell)", cells[col] if col < len(cells) else "", row))
    out.extend(
        f"{INDEX}:{ln}: claim marker outside a table with a Status column" for ln, _m in markers if ln not in allowed
    )
    for rid, (token, date, _reason) in NO_INDEX_ENTRY.items():
        row = by_id.get(rid)
        if row is None or (row.token, row.date) != (token, date) or rid in cited_ids:
            out.append(
                f"{INDEX}: the NO_INDEX_ENTRY exemption for {rid} ({token} {date}) is stale; remove or re-grant it"
            )
    for r in rows:
        if (
            r.id.startswith("T1-")
            and r.token not in EXEMPT_FROM_INDEX
            and r.id not in NO_INDEX_ENTRY
            and r.id not in cited_ids
        ):
            out.append(f"{INDEX}: Tier 1 row {r.id} ({r.token}) is cited by no index entry")
    return out


def main() -> int:
    failures = lint(
        (REPO_ROOT / README).read_text(encoding="utf-8"),
        (REPO_ROOT / INDEX).read_text(encoding="utf-8"),
        (REPO_ROOT / L.LEDGER_PATH).read_text(encoding="utf-8"),
    )
    if failures:
        print(f"claims sync FAILED ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("claims sync: README results table and experiments index agree with the ledger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
