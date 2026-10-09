#!/usr/bin/env python3
"""The behavioural-graduation ledger's format lint (M1b PR 3; design: docs/plans/m1b_ledger_evidence_gate.md).

Every run, over the ledger's two status tables (parsed by ``scripts/_ledger.py``, the parser PR 5's evidence
gate shares):

- **Structure:** both tables exist and are contiguous. Every row has the header's cell count and a unique ID
  (``T1-<n>`` / ``T3-<n>``). No status-shaped row sits outside the tables.
- **Status line:** ``**Status: <TOKEN> <YYYY-MM-DD>**``. The token comes from the closed vocabulary, and the
  date is a real calendar date no later than today (UTC).
- **Evidence:**
  - A positive status (EARNED / MAINTAINED / RE-VALIDATED / REPRODUCED) cites a record under
    ``docs/experiments/data/``.
  - Each record is tracked by git and is not a symlink. It is a file, or a session directory holding a
    ``report.json``. It is not a script, a README or markdown, the data root, or anything named
    aborted/invalid/dry-run/non-frozen.
  - RE-VALIDATED-BY-TESTS cites tracked ``tests/**/test_*.py`` files.
  - Link text that looks like a path must name the link's own target.
- **Regression guard:** positive, RE-VALIDATED-BY-TESTS and LEGACY rows carry a ``Regression guard:`` field that
  cites a complete link or a code span before its sentence or field ends (#1012).
- **SUPERSEDED** names a different, existing row that is not itself superseded (``by T<n>-<m>``).

Against the merge-base (diff-scoped, like the other ``_lint_git`` lints):

- No ID vanishes. A row's date never decreases.
- A raise (a higher rank), or a move between positive tokens, needs a later date. A lowering may keep it.
- LEGACY can be kept but never entered.
- A row ENTERING SUPERSEDED (from any other token, or new), or changing its ``by`` target, names a successor that
  REACHES a positive status (EARNED / MAINTAINED / RE-VALIDATED / REPRODUCED) IN THE SAME DIFF (positive at HEAD, not at the
  base, or new) and whose row names the superseded row's id. SUPERSEDED is rank 0 like STALE and BROKEN but does
  not block a release, so retiring a row behind an unearned successor would clear the block with no evidence, and
  pointing at an already-earned row would let an old verdict retire a new claim (owner decision 2026-10-03, D3).
  Any other case needs a ``superseded`` clause already on main in ``docs/experiments/evidence_exceptions.json``
  (the evidence gate's file and semantics: append-only, only clauses on the base act) naming exactly this
  transition: ``{"id", "kind": "superseded", "row", "from": <base token or null>, "by", "to_date", "owner",
  "reason", "date"}``. Checked at the transition only: a successor that later goes STALE is its own row's problem.
- A new row's date is no earlier than the branch point's date (on a PR merge ref, the fork point of the PR head).
- A ledger absent at the base fails.
- A base from before this format (no ``**Status: `` line anywhere) skips only these checks.

This lint catches forgetting and makes the ledger machine-readable. Whether a cited record is RIGHT for its
claim is PR 5's gate (``record_kind``, provenance, ``finish_reason``) and review.

Exits: 0 clean; 1 violations; 2 the base could not be read on a pull request (``_lint_git.must_not_skip``).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ledger as L  # noqa: E402
import _lint_git  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


#: Qualifiers that predate the scope-vocabulary rule (#1105), pinned to the exact status they were granted at: the
#: entry goes stale, and fails, once the row's token, date or qualifier changes (re-judge the row and fix it then).
GRANDFATHERED_QUALIFIERS = {"T1-5": ("PARTIAL", "2026-06-15", "reframed")}


def qualifier_problems(row: L.Row) -> list[str]:
    """A qualifier opens with a scope word (``_ledger.SCOPE_HEAD``); a reason goes in the prose (#1105)."""
    pin = GRANDFATHERED_QUALIFIERS.get(row.id)
    if pin is not None:
        if (row.token, row.date, row.qualifier) != pin:
            return [f"the GRANDFATHERED_QUALIFIERS entry for {row.id} {pin} is stale; remove it and fix the qualifier"]
        return []
    if row.qualifier is not None and not L.SCOPE_HEAD.match(L.qualifier_head(row.qualifier)):  # `()` included
        return [
            f"qualifier ({row.qualifier}) must open with a scope word (`narrow`, `rung <X>`); a reason goes in the "
            "prose (the format spec: the qualifier carries SCOPE only)"
        ]
    return []


_NEXT_FIELD = re.compile(r"\*\*[^*]+:\*\*")
#: What a guard citation looks like: a path or a test node (``/`` or ``::``, or a ``.py``/``.md`` name).
_GUARD_TARGET = re.compile(r"/|::|\.(?:py|md)\b")


def cited_guard(cell: str) -> bool:
    """A ``Regression guard:`` phrase followed, before its sentence or field ends, by a link whose target is a path
    (not empty, not only an ``#anchor``) or a code span shaped like a path or test node (#1012: presence alone passed
    prose such as "there is no regression guard: TODO", and a backticked `TODO` is still prose). The
    span ends at ``. `` outside links and code spans, or at the next ``**Field:**`` label (a following Evidence code
    span is not the guard's citation)."""
    for m in L.GUARD.finditer(cell):
        rest = cell[m.end() :]
        masked = rest
        for rx in (L._LINK, L._CODE):  # a ". " inside link text or a code span does not end the sentence
            masked = rx.sub(lambda t: "\0" * len(t.group(0)), masked)
        ends = [x.start() for x in (re.search(r"\.\s", masked), _NEXT_FIELD.search(masked)) if x]
        span = rest[: min(ends)] if ends else rest
        links = (t.group(2) for t in L._LINK.finditer(span))
        codes = (t.group(1) for t in L._CODE.finditer(span))
        if any(x and not x.startswith("#") for x in links) or any(_GUARD_TARGET.search(c) for c in codes):
            return True
    return False


def row_problems(row: L.Row, rows_by_id: dict[str, L.Row], tracked: dict[str, str], today: str) -> list[str]:
    out = list(row.problems)
    if row.token is None:
        return out
    out += qualifier_problems(row)
    if row.token not in L.RANK:
        out.append(f"status token {row.token!r} is not in the vocabulary ({', '.join(L.RANK)})")
    if not L.valid_date(row.date or ""):
        out.append(f"status date {row.date!r} is not a calendar date")
    elif (row.date or "") > today:
        out.append(f"status date {row.date} is after today ({today}, UTC)")
    if (row.token in L.POSITIVE or row.token == L.BY_TESTS) and not row.evidence:
        out.append(f"{row.token} needs an **Evidence:** field citing at least one record")
    for entry in row.evidence:
        out.extend(L.classify_entry(entry, tracked, tests=row.token == L.BY_TESTS)[1])
    if row.token in L.NEEDS_GUARD and not cited_guard(row.status_cell):
        out.append(
            f"{row.token} row has no 'Regression guard:' field citing a link or `code` span in its sentence "
            "(the phrase alone, or prose like 'no regression guard: TODO', is not a guard; #1012)"
        )
    if row.token == "SUPERSEDED":
        target = rows_by_id.get(row.superseded_by or "")
        if target is None or target.id == row.id or target.token == "SUPERSEDED":
            out.append("SUPERSEDED must name a different, existing, not-superseded row right after it: 'by T<n>-<m>'")
    return out


EXCEPTIONS = "docs/experiments/evidence_exceptions.json"
SUPERSEDED_CLAUSE_FIELDS = ("id", "kind", "row", "by", "to_date", "owner", "reason", "date")


def superseded_clause(base_exceptions: list, row: L.Row, old: L.Row | None) -> dict | None:
    """The base's ``superseded`` exception clause naming exactly this transition, or None."""
    for e in base_exceptions:
        if not isinstance(e, dict) or e.get("kind") != "superseded" or "from" not in e:
            continue
        if any(not e.get(f) for f in SUPERSEDED_CLAUSE_FIELDS):
            continue  # malformed: never acts (the evidence gate reports it)
        if (e["row"], e["from"], e["by"], e["to_date"]) == (row.id, old.token if old else None, row.superseded_by,
                                                            row.date):  # fmt: skip
            return e
    return None


def superseded_entry_problems(
    row: L.Row, now: dict[str, L.Row], base_rows: dict[str, L.Row], old: L.Row | None, base_exceptions: list
) -> list[str]:
    """A row entering SUPERSEDED, or re-pointing it: its successor reaches a positive status in this diff, or a clause
    on main excepts exactly this transition (D3)."""
    target = now.get(row.superseded_by or "")
    before = base_rows.get(row.superseded_by or "")
    names_row = target is not None and re.search(
        rf"(?<![\w-]){re.escape(row.id)}(?![\w-])", " ".join(target.cells.values())
    )
    reached = target is not None and target.token in L.POSITIVE and (before is None or before.token not in L.POSITIVE)
    if reached and names_row:
        return []
    if superseded_clause(base_exceptions, row, old) is not None:
        return []
    have = None if target is None else target.token
    was = old.token if old else "a new row"
    return [
        f"{row.id}: {was} -> SUPERSEDED by {row.superseded_by} needs that successor to REACH a positive status "
        f"({', '.join(sorted(L.POSITIVE))}) in this same diff (it is {have}"
        f"{'' if before is None else f', was {before.token}'}) and its row naming {row.id} "
        f"({'it does' if names_row else 'it does not'}), or a `superseded` clause on main in {EXCEPTIONS}"
    ]


def base_problems(
    rows: list[L.Row], base_text: str | None, base_date: str, base_exceptions: list | None = None
) -> list[str]:
    """The diff-scoped rules. ``base_text`` None = the ledger did not exist at the base."""
    if base_text is None:
        return [f"{L.LEDGER_PATH} does not exist at the merge-base (renamed or moved?)"]
    # Both the raw scan (header-independent) and the parser's rows: an ID the parser reads is an ID.
    base_ids = set(L.ROW_ID_RE.findall(base_text)) | {r.id for r in L.parse(base_text)[0] if L.ID_RE.match(r.id)}
    now = {r.id: r for r in rows}
    out = [
        f"{i}: a row ID in the base is gone (IDs are never removed; retire a row as DROPPED)"
        for i in sorted(base_ids - set(now))
    ]
    if "**Status: " not in base_text:
        return out  # the base predates this format: nothing to compare dates or tokens against
    base_rows = {r.id: r for r in L.parse(base_text)[0] if r.token and r.date}
    exceptions = base_exceptions or []
    for rid, row in now.items():
        if not (row.token and row.date):
            continue
        old = base_rows.get(rid)
        if old is None:
            if row.date < base_date:
                out.append(f"{rid}: a new row's date {row.date} is before the branch point ({base_date})")
            if row.token == "LEGACY":
                out.append(f"{rid}: LEGACY can be kept, never entered")
            if row.token == "SUPERSEDED":
                out += superseded_entry_problems(row, now, base_rows, None, exceptions)
            continue
        if row.date < old.date:
            out.append(f"{rid}: date moved back from {old.date} to {row.date}")
        if row.token == "SUPERSEDED" and (old.token != "SUPERSEDED" or old.superseded_by != row.superseded_by):
            out += superseded_entry_problems(row, now, base_rows, old, exceptions)
        if row.token != old.token:
            if row.token == "LEGACY":
                out.append(f"{rid}: LEGACY can be kept, never entered (was {old.token})")
            if L.needs_new_date(old.token, row.token) and row.date <= old.date:
                out.append(f"{rid}: {old.token} -> {row.token} asserts something new and needs a date after {old.date}")
    return out


def _branch_point(repo_root: Path, base: str) -> str:
    """Where the change forked from main. On a pull request CI checks out a merge commit whose first parent is
    main's tip (the merge-base): the branch point is then the merge-base of main and the PR's own head, so a
    new row's date is judged against when the branch started, not against how far main has moved since."""
    try:
        first = _lint_git.git(repo_root, "rev-parse", "HEAD^1").strip()
        second = _lint_git.git(repo_root, "rev-parse", "--verify", "--quiet", "HEAD^2").strip()
    except _lint_git.GitUnavailable:
        return base
    if second and first == _lint_git.git(repo_root, "rev-parse", base).strip():
        return _lint_git.git(repo_root, "merge-base", base, second).strip() or base
    return base


def _branch_epoch(repo_root: Path, base: str) -> int:
    """When the change being judged forked from main. On a push (#1089) the range from the last green push can hold
    several merged PRs, and a squash or rebase merge keeps no fork point in git: each unit's own fork comes from
    ``_lint_git.push_units`` (its PR's first commit when git cannot say), and the EARLIEST is used. That is lenient
    for a row in a later unit of a multi-PR range (a stated residual; the row was judged on its PR), and exact for a
    single-unit push, which is the direct push this half of the gate exists to catch."""
    if os.environ.get("GITHUB_EVENT_NAME") == "push":
        units = _lint_git.push_units(repo_root, base)
        if units:
            return min(u.fork_epoch for u in units)
    return int(_lint_git.git(repo_root, "show", "-s", "--format=%ct", _branch_point(repo_root, base)).strip())


def lint(repo_root: Path = REPO_ROOT, *, base: str | None = None, today: str | None = None) -> tuple[list[str], int]:
    """(violations, exit_code_if_the_base_was_unreadable_or_0)."""
    today = today or _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    ledger = repo_root / L.LEDGER_PATH
    if not ledger.exists():
        return [f"{L.LEDGER_PATH} is missing: the ledger format cannot be checked"], 2
    text = ledger.read_text(encoding="utf-8")
    rows, problems = L.parse(text)
    tracked = L.tracked_files(repo_root)
    by_id = {r.id: r for r in rows}
    failures = [f"{L.LEDGER_PATH}: {p}" for p in problems]
    for row in rows:
        failures.extend(f"{L.LEDGER_PATH}:{row.line} {row.id}: {p}" for p in row_problems(row, by_id, tracked, today))
    failures.extend(
        f"{L.LEDGER_PATH}: the GRANDFATHERED_QUALIFIERS entry for {rid} names no ledger row; remove it"
        for rid in GRANDFATHERED_QUALIFIERS
        if rid not in by_id
    )
    try:
        base = base or _lint_git.base_ref(repo_root)
        base_text = _lint_git.show(repo_root, base, L.LEDGER_PATH) or None
        base_exc_text = _lint_git.show(repo_root, base, EXCEPTIONS)
        epoch = _branch_epoch(repo_root, base)
        base_date = _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc).date().isoformat()
    except _lint_git.GitUnavailable as exc:
        if _lint_git.must_not_skip(f"no merge-base for the ledger's diff-scoped checks: {exc}"):
            return failures, 2
        print(f"SKIP ledger base checks (no merge-base: {exc})", file=sys.stderr)
        return failures, 0
    try:
        base_exceptions = json.loads(base_exc_text) if base_exc_text else []
    except ValueError:
        base_exceptions = None
    if not isinstance(base_exceptions, list):
        failures.append(f"{EXCEPTIONS} at the base is not a JSON list: no exception clause can act")
        base_exceptions = []
    failures.extend(f"{L.LEDGER_PATH}: {p}" for p in base_problems(rows, base_text, base_date, base_exceptions))
    return failures, 0


def main(argv: list[str] | None = None) -> int:
    failures, code = lint()
    if failures:
        print(f"ledger format lint FAILED ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return code or 1
    if code:
        return code
    print("ledger format lint: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
