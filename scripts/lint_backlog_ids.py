#!/usr/bin/env python3
"""Mechanization-backlog IDs are unique, and every backlog citation names one that exists (#1175).

``docs/plans/outstanding.md`` §Mechanization backlog is the register: CLAUDE.md's "Enforced, or on the backlog"
principle cites its rows (``Regression guard: process invariant — mechanization backlog M<n>``), as do briefs, plans and
decision records. On 2026-10-08/09 three parallel PRs each took the next free number while main took the same three;
only a textual conflict exposed it, and a row added elsewhere would have merged a duplicate silently.

Two checks, over the TREE (stateless, not diff-scoped: two PRs that each pass alone but collide after merge fail on the
merged commit, and branch protection's up-to-date rule makes the second PR re-run against the first):

1. DEFINITIONS, only in ``docs/plans/outstanding.md``: an open-table row ``| M<n> |`` or a Closed entry ``- **M<n> — ``
   (line-anchored, so an ID inside a row's prose is a citation, not a definition). Each ``<n>`` is defined exactly once
   across both; a first cell shaped like ``M<n><suffix>`` is malformed.
2. CITATIONS, in every tracked ``*.md *.py *.yml *.toml`` file (the register included: a definition's own ID
   resolves trivially; this lint and its tests, which quote example citations, are skipped): within a paragraph that carries
   backlog context (``mechanization backlog``, ``backlog row(s)``, ``outstanding.md`` or a link to it, ``§Mechanization``;
   case-insensitive), every ``M<n>`` token (``M4/M23`` is two) and every range ``M<a>–M<b>`` (``M`` on both ends) must
   name a defined ID. A letter-suffixed token (``M1b``, the M1b PR series) is not a backlog ID and is skipped; a
   backwards range is malformed.

Stated limits (this catches forgetting, not evasion):
- A citation outside a backlog-context paragraph is not checked (a markdown list with blank lines between its items is
  several paragraphs, so only the item carrying the context word is checked), and an unrelated ``M<n>`` inside one (``Mac M2``) is
  read as a citation (it resolves harmlessly unless that number is undefined; list it in ``EXEMPT`` then), as is a path
  or URL segment like ``docs/M3/``. A range written ``M10..M14`` is read as ``M10`` alone.
- RESOLVABLE IS NOT CORRECT: after a collision is fixed by renumbering, an old citation still resolves, to the wrong row.
  Review checks that each citation names the row it means.
- Dated records are never relinked (outstanding.md §Editing rule); ``EXEMPT`` is the explicit way out for one whose
  citation cannot be edited. It is empty today.

Exits: 0 clean; 1 violations; 2 the register cannot be read or defines nothing.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTER = "docs/plans/outstanding.md"
SUFFIXES = ("*.md", "*.py", "*.yml", "*.toml")
#: Files that quote example citations on purpose (this lint's own source and tests).
SELF = frozenset({"scripts/lint_backlog_ids.py", "tests/unit/test_lint_backlog_ids.py"})

#: (path, token) citations exempted by owner decision; each entry names its reason. Empty today.
EXEMPT: dict[tuple[str, str], str] = {}

_TABLE_DEF = re.compile(r"^\|\s*M(\d\w*?)\s*\|")
_CLOSED_DEF = re.compile(r"^-\s*\*\*M(\d\w*?)\s*[—-]")
_CONTEXT = re.compile(r"mechanization backlog|backlog rows?\b|outstanding\.md|§\s*mechanization", re.IGNORECASE)
_TOKEN = re.compile(r"(?<![A-Za-z0-9_.-])M(\d+)(?:\s*[–—-]\s*M(\d+))?(?![A-Za-z0-9])")
#: ``M10–14``: a range whose second end lacks its ``M`` would be read as ``M10`` alone, so it is refused (#1175 review).
_BARE_RANGE = re.compile(r"(?<![A-Za-z0-9_.-])M\d+[–—]\d+(?![A-Za-z0-9])")
_PARAGRAPH = re.compile(r"\n\s*\n")


def definitions(text: str) -> tuple[dict[int, list[int]], list[str]]:
    """``{n: [line numbers]}`` of every defined ID, and the malformed definition lines."""
    defs: dict[int, list[int]] = {}
    problems: list[str] = []
    for i, line in enumerate(text.split("\n"), 1):
        m = _TABLE_DEF.match(line) or _CLOSED_DEF.match(line)
        if not m:
            continue
        if not m.group(1).isdigit():
            problems.append(f"{REGISTER}:{i}: malformed backlog ID 'M{m.group(1)}' (an ID is M<number>)")
            continue
        defs.setdefault(int(m.group(1)), []).append(i)
    return defs, problems


def citations(text: str) -> list[tuple[int, str, list[int] | None]]:
    """``(line, token, ids)`` for every ``M<n>`` / range in a backlog-context paragraph; ids is None when malformed."""
    out: list[tuple[int, str, list[int] | None]] = []
    offset = 0
    for para in _PARAGRAPH.split(text):
        start = text.find(para, offset)
        offset = start + len(para)
        if not _CONTEXT.search(para):
            continue
        for m in _BARE_RANGE.finditer(para):
            out.append((text.count("\n", 0, start + m.start()) + 1, m.group(0), None))
        for m in _TOKEN.finditer(para):
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            line = text.count("\n", 0, start + m.start()) + 1
            out.append((line, m.group(0), list(range(a, b + 1)) if b >= a else None))
    return out


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(["git", "ls-files", *SUFFIXES], cwd=root, capture_output=True, text=True, check=True)
    return [f for f in result.stdout.split("\n") if f and f not in SELF]


def lint(root: Path = REPO_ROOT) -> tuple[list[str], int]:
    """(violations, number of citations checked). Raises OSError / ValueError when the register is unusable."""
    defs, problems = definitions((root / REGISTER).read_text(encoding="utf-8"))
    if not defs:
        raise ValueError(f"{REGISTER} defines no backlog IDs")
    for n, lines in sorted(defs.items()):
        if len(lines) > 1:
            problems.append(f"{REGISTER}: M{n} is defined {len(lines)} times (lines {lines}); an ID is defined once")
    checked = 0
    for path in tracked_files(root):
        try:
            text = (root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: nothing to cite
        for line, token, ids in citations(text):
            if (path, token) in EXEMPT:
                continue
            checked += 1
            if ids is None:
                problems.append(f"{path}:{line}: malformed backlog range {token!r} (write M<a>–M<b>, low to high)")
                continue
            missing = [n for n in ids if n not in defs]
            if missing:
                names = ", ".join(f"M{n}" for n in missing)
                problems.append(f"{path}:{line}: {token!r} cites {names}, which {REGISTER} does not define")
    return problems, checked


def main() -> int:
    try:
        problems, checked = lint(REPO_ROOT)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"backlog-ID lint could not run: {exc}", file=sys.stderr)
        return 2
    if problems:
        print(f"backlog-ID lint FAILED ({len(problems)}):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"backlog-ID lint: clean ({checked} citation(s) resolve; every ID in {REGISTER} defined once)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
