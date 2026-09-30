"""The behavioural-graduation ledger, parsed (M1b PR 3; design: docs/plans/m1b_ledger_evidence_gate.md).

One parser for the ledger's two status tables, shared by ``scripts/lint_ledger_format.py`` (the format lint)
and M1b PR 5's evidence gate, so both read a row the same way. Stdlib-only, and it imports nothing from
``maxim``.

A row's Status cell opens with a machine-readable status line, then an optional Evidence field, then prose::

    **Status: <TOKEN> <YYYY-MM-DD>** (<qualifier>). **Evidence:** <entry>, <entry>; <entry>. <prose ...>

- TOKEN comes from :data:`RANK` (a closed vocabulary), and the date is when that status was SET. A later
  trigger walk's annotation is prose, never the token.
- The qualifier is at most one parenthesised group, and carries claim scope ("narrow").
- An Evidence entry is a markdown link (its TARGET is the entry, resolved relative to ``docs/plans/``) or a
  code span (a repo-relative path). Entries are separated by ``,`` or ``;``, and the list ends with ``.``,
  ``**`` or the end of the cell. Anything else is a grammar error, never a silently truncated list.
"""

from __future__ import annotations

import datetime as _dt
import posixpath
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

LEDGER_PATH = "docs/plans/behavioral_graduation_candidates.md"
DATA_ROOT = "docs/experiments/data"

# Status rank. A RAISE is a move to a higher rank. Rank 3 is the positive set, which the evidence gate treats
# as run-evidence-backed. RE-VALIDATED-BY-TESTS and LEGACY sit below it: named as unchecked by the gate.
RANK: dict[str, int] = {
    "DROPPED": 0,
    "DORMANT": 0,
    "BROKEN": 0,
    "STALE": 0,
    "SUPERSEDED": 0,
    "TIER-2": 0,
    "SETUP": 1,
    "BORDERLINE": 1,
    "PARTIAL": 2,
    "RE-VALIDATED-BY-TESTS": 2,
    "LEGACY": 2,
    "EARNED": 3,
    "MAINTAINED": 3,
    "RE-VALIDATED": 3,
}
POSITIVE = frozenset(t for t, r in RANK.items() if r == 3)
BY_TESTS = "RE-VALIDATED-BY-TESTS"
# Tokens whose row must carry a `Regression guard:` field (what re-validates it when a trigger fires).
NEEDS_GUARD = POSITIVE | {BY_TESTS, "LEGACY"}

# The two status tables, found by their exact header cells. `claim` names the cells whose change is a claim
# change (a PR 5 trigger).
TABLES: dict[str, dict[str, tuple[str, ...]]] = {
    "T1": {"header": ("ID", "Claim", "Bio-mechanism", "Status"), "claim": ("Claim",)},
    "T3": {
        "header": ("ID", "CLAUDE.md ref", "Mechanism", "Bio-claim", "Graduation predicate", "Status"),
        "claim": ("Bio-claim", "Graduation predicate"),
    },
}

ID_RE = re.compile(r"^T(\d)-(\d+)$")
ROW_ID_RE = re.compile(r"^[ >]*\|\s*(T\d-\d+)\s*\|", re.M)  # a row's first cell, header-independent (base scans)
STATUS_RE = re.compile(r"^\*\*Status: ([A-Z0-9-]+) (\d{4}-\d{2}-\d{2})\*\*(?: \(([^()]*)\))?")
EVIDENCE_MARK = "**Evidence:**"
SUPERSEDED_BY = re.compile(r"^\s*by (T\d-\d+)\b")  # right after the status line: `**Status: SUPERSEDED d** by T1-9`
GUARD = re.compile(r"regression\s+guards?\s*:", re.IGNORECASE)
# Never evidence: scripts, markdown, notebooks; names (any path component, `-`/`_` ignored) marking a capture
# that is not a gated run. A superset of lint_prereg_precedes_data's NON_RECORD_SUFFIXES / NON_GATED_MARKERS,
# which stay as they are (widening them would change which files THAT lint judges).
REFUSED_SUFFIXES = (".py", ".md", ".sh", ".ipynb")
REFUSED_MARKERS = ("dryrun", "nonfrozen", "aborted", "invalid")
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)\)")
_CODE = re.compile(r"`([^`]+)`")


@dataclass
class Entry:
    """One Evidence entry: ``kind`` is ``link`` or ``code``; ``path`` is repo-relative, normalised; ``raw``
    is the target as written; ``text`` is a link's text (empty for a code span)."""

    kind: str
    raw: str
    path: str
    text: str = ""


@dataclass
class Row:
    id: str
    table: str
    line: int
    cells: dict[str, str]
    token: str | None = None
    date: str | None = None
    qualifier: str | None = None
    evidence: list[Entry] = field(default_factory=list)
    has_evidence_field: bool = False
    superseded_by: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def status_cell(self) -> str:
        return self.cells.get("Status", "")

    @property
    def claim(self) -> str:
        return " | ".join(self.cells.get(c, "") for c in TABLES[self.table]["claim"])


def split_cells(line: str) -> list[str]:
    """A GFM table row's cells: split on UNESCAPED ``|`` (inside a code span too, as GitHub does)."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    cells, cur, i = [], [], 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body) and body[i + 1] == "|":
            cur.append("\\|")
            i += 2
            continue
        if ch == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def _resolve(kind: str, raw: str) -> str:
    """Repo-relative, normalised path of an entry: a link relative to the ledger's directory, a code span
    relative to the repo root. No fallback between the two."""
    base = posixpath.dirname(LEDGER_PATH) if kind == "link" else ""
    return posixpath.normpath(posixpath.join(base, raw)) if raw else ""


def _parse_evidence(text: str) -> tuple[list[Entry], str | None]:
    """Parse the entry list right after ``**Evidence:**``. Returns (entries, error)."""
    entries: list[Entry] = []
    i = 0
    while True:
        while i < len(text) and text[i] == " ":
            i += 1
        m_link, m_code = _LINK.match(text, i), _CODE.match(text, i)
        if m_link:
            label, raw = m_link.group(1), m_link.group(2)
            entries.append(Entry("link", raw, _resolve("link", raw), label))
            i = m_link.end()
        elif m_code:
            raw = m_code.group(1)
            entries.append(Entry("code", raw, _resolve("code", raw)))
            i = m_code.end()
        else:
            return entries, f"Evidence entry {len(entries) + 1} is not a link or a code span: {text[i : i + 40]!r}"
        while i < len(text) and text[i] == " ":
            i += 1
        if i < len(text) and text[i] in ",;":
            i += 1
            continue
        if i >= len(text) or text.startswith(".", i) or text.startswith("**", i):
            return entries, None
        return entries, f"Evidence list must end with '.', '**' or the cell's end, not {text[i : i + 30]!r}"


def _parse_status(row: Row) -> None:
    cell = row.status_cell
    m = STATUS_RE.match(cell)
    if not m:
        row.problems.append("Status cell does not open with **Status: <TOKEN> <YYYY-MM-DD>**")
        return
    row.token, row.date, row.qualifier = m.group(1), m.group(2), m.group(3)
    rest = cell[m.end() :]
    sup = SUPERSEDED_BY.match(rest)
    row.superseded_by = sup.group(1) if sup else None
    n_marks = cell.count(EVIDENCE_MARK)
    if n_marks > 1:
        row.problems.append("more than one **Evidence:** field")
    if not n_marks:
        return
    after = rest.lstrip(". ")
    if not after.startswith(EVIDENCE_MARK):
        row.problems.append("the **Evidence:** field must come straight after the Status line (and qualifier)")
        return
    row.has_evidence_field = True
    row.evidence, error = _parse_evidence(after[len(EVIDENCE_MARK) :])
    if error:
        row.problems.append(error)


def parse(text: str) -> tuple[list[Row], list[str]]:
    """Every status-table row, and the problems that are not a single row's (missing tables, stray rows)."""
    raw_lines = text.split("\n")
    # GitHub continues a table through a row indented up to 3 spaces, and renders a `>`-quoted table: a status
    # row must not hide from the lint behind either, so every line is read without them.
    lines = [ln.lstrip(" >") for ln in raw_lines]
    rows: list[Row] = []
    problems: list[str] = []
    table_lines: set[int] = set()
    for prefix, spec in TABLES.items():
        header = spec["header"]
        starts = [i for i, ln in enumerate(lines) if ln.startswith("|") and tuple(split_cells(ln)) == header]
        if len(starts) != 1:
            problems.append(f"{prefix} status table: expected one header {header}, found {len(starts)}")
            continue
        start = starts[0]
        table_lines.update((start, start + 1))
        i = start + 2
        while i < len(lines) and lines[i].startswith("|"):
            table_lines.add(i)
            cells = split_cells(lines[i])
            row = Row(id=cells[0] if cells else "", table=prefix, line=i + 1, cells={})
            if len(cells) != len(header):
                row.problems.append(f"{len(cells)} cells, header has {len(header)} (escape a '|' in code as '\\|')")
            else:
                row.cells = dict(zip(header, cells))
                m = ID_RE.match(row.id)
                if not m or f"T{m.group(1)}" != prefix:
                    row.problems.append(f"first cell {row.id!r} is not a {prefix}-<n> ID")
                _parse_status(row)
            rows.append(row)
            i += 1
        if i < len(lines) and lines[i].strip() == "" and i + 1 < len(lines) and lines[i + 1].startswith("|"):
            problems.append(f"{prefix} status table is split by a blank line at line {i + 1}")
    for i, ln in enumerate(lines):
        if i in table_lines or not ln.startswith("|"):
            continue
        cells = split_cells(ln)
        if (cells and ID_RE.match(cells[0])) or any(c.startswith("**Status: ") for c in cells):
            problems.append(f"line {i + 1}: a status-shaped row outside the two status tables")
    seen: dict[str, int] = {}
    for row in rows:
        if row.id in seen:
            problems.append(f"duplicate ID {row.id} (lines {seen[row.id]} and {row.line})")
        seen.setdefault(row.id, row.line)
    return rows, problems


def is_raise(old_token: str | None, new_token: str) -> bool:
    """A move to a higher rank. ``old_token`` None (a new row) counts as rank 0: a new row arriving at rank
    >= 1 is a raise from nothing."""
    return RANK.get(new_token, 0) > (RANK.get(old_token, 0) if old_token else 0)


def needs_new_date(old_token: str, new_token: str) -> bool:
    """A status change that asserts something new happened: a raise, or a move between positive tokens (a
    re-validation claimed without a new date would fire no trigger)."""
    return is_raise(old_token, new_token) or (old_token != new_token and {old_token, new_token} <= POSITIVE)


def tracked_files(repo_root: Path | str) -> dict[str, str]:
    """``path -> mode`` for every tracked file under the data root and ``tests/`` (mode 120000 = symlink)."""
    out = subprocess.run(
        ["git", "ls-files", "-s", "--", DATA_ROOT, "tests"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tracked: dict[str, str] = {}
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        tracked[path] = meta.split()[0]
    return tracked


def classify_entry(entry: Entry, tracked: dict[str, str], *, tests: bool = False) -> tuple[str | None, list[str]]:
    """``(kind, problems)`` for one Evidence entry: kind ``file`` / ``session`` (a directory holding a
    ``report.json``) / ``test``, or None when it is not a record. The one classification the format lint and
    the evidence gate share."""
    path, out = entry.path, []
    if entry.kind == "link" and _looks_like_path(entry.text):
        label = entry.text.strip("` ")
        if label not in (posixpath.basename(path), path, entry.raw) and not path.endswith("/" + label):
            out.append(f"link text {entry.text!r} names a different file than its target {entry.raw!r}")
    if "#" in entry.raw or "://" in entry.raw:
        return None, [*out, f"{entry.raw!r}: cite a whole tracked file (no anchor, no URL)"]
    if tests:
        name = posixpath.basename(path)
        if not (path.startswith("tests/") and name.startswith("test_") and name.endswith(".py")):
            return None, [*out, f"{entry.raw!r}: RE-VALIDATED-BY-TESTS cites tests/**/test_*.py files"]
        if tracked.get(path) in (None, "120000"):
            return None, [*out, f"{path}: not a tracked test file"]
        return ("test" if not out else None), out
    if not path.startswith(DATA_ROOT + "/"):
        return None, [*out, f"{entry.raw!r} resolves to {path!r}, outside {DATA_ROOT}/"]
    parts = [p.lower().replace("-", "").replace("_", "") for p in path.split("/")[3:]]
    if any(m in p for p in parts for m in REFUSED_MARKERS):
        out.append(f"{path}: an aborted, invalid or non-gated capture is not evidence")
    if path.lower().endswith(REFUSED_SUFFIXES) or posixpath.basename(path).lower().startswith("readme"):
        out.append(f"{path}: a script, README or markdown file is context, not evidence")
    kind: str | None = None
    if path in tracked:
        if tracked[path] == "120000":
            out.append(f"{path}: a symlink is not evidence")
        else:
            kind = "file"
    elif any(p.startswith(path + "/") for p in tracked):
        if tracked.get(path + "/report.json") is None:
            out.append(f"{path}: a directory is evidence only as a session directory holding a report.json")
        else:
            kind = "session"
    else:
        out.append(f"{path}: not a tracked file or directory")
    return (kind if not out else None), out


def _looks_like_path(text: str) -> bool:
    t = text.strip("` ")
    return "/" in t or bool(posixpath.splitext(t)[1])


def valid_date(date: str) -> bool:
    try:
        _dt.date.fromisoformat(date)
    except ValueError:
        return False
    return True
