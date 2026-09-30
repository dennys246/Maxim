#!/usr/bin/env python3
"""The behavioural-graduation ledger's format lint (M1b PR 3; design: docs/plans/m1b_ledger_evidence_gate.md).

Every run, over the ledger's two status tables (parsed by ``scripts/_ledger.py``, the parser PR 5's evidence
gate shares):

- **Structure:** both tables exist and are contiguous. Every row has the header's cell count and a unique ID
  (``T1-<n>`` / ``T3-<n>``). No status-shaped row sits outside the tables.
- **Status line:** ``**Status: <TOKEN> <YYYY-MM-DD>**``. The token comes from the closed vocabulary, and the
  date is a real calendar date no later than today (UTC).
- **Evidence:**
  - A positive status (EARNED / MAINTAINED / RE-VALIDATED) cites at least one record under
    ``docs/experiments/data/``.
  - Each record is tracked by git and is not a symlink. It is a file, or a session directory holding a
    ``report.json``. It is not a script, a README or markdown, the data root, or anything named
    aborted/invalid/dry-run/non-frozen.
  - RE-VALIDATED-BY-TESTS cites tracked ``tests/**/test_*.py`` files.
  - Link text that looks like a path must name the link's own target.
- **Regression guard:** positive, RE-VALIDATED-BY-TESTS and LEGACY rows carry a ``Regression guard:`` field.
- **SUPERSEDED** names a different, existing row that is not itself superseded (``by T<n>-<m>``).

Against the merge-base (diff-scoped, like the other ``_lint_git`` lints):

- No ID vanishes. A row's date never decreases.
- A raise (a higher rank), or a move between positive tokens, needs a later date. A lowering may keep it.
- LEGACY can be kept but never entered.
- A new row's date is no earlier than the branch point's date (on a PR merge ref, the fork point of the PR head).
- A ledger absent at the base fails.
- A base from before this format (no ``**Status: `` line anywhere) skips only these checks.

This lint catches forgetting and makes the ledger machine-readable. Whether a cited record is RIGHT for its
claim is PR 5's gate (``record_kind``, provenance, ``finish_reason``) and review.

Exits: 0 clean; 1 violations; 2 the base could not be read on a pull request (``_lint_git.must_not_skip``).
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ledger as L  # noqa: E402
import _lint_git  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def row_problems(row: L.Row, rows_by_id: dict[str, L.Row], tracked: dict[str, str], today: str) -> list[str]:
    out = list(row.problems)
    if row.token is None:
        return out
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
    if row.token in L.NEEDS_GUARD and not L.GUARD.search(row.status_cell):
        out.append(f"{row.token} row has no 'Regression guard:' field in its Status cell")
    if row.token == "SUPERSEDED":
        target = rows_by_id.get(row.superseded_by or "")
        if target is None or target.id == row.id or target.token == "SUPERSEDED":
            out.append("SUPERSEDED must name a different, existing, not-superseded row right after it: 'by T<n>-<m>'")
    return out


def base_problems(rows: list[L.Row], base_text: str | None, base_date: str) -> list[str]:
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
    for rid, row in now.items():
        if not (row.token and row.date):
            continue
        old = base_rows.get(rid)
        if old is None:
            if row.date < base_date:
                out.append(f"{rid}: a new row's date {row.date} is before the branch point ({base_date})")
            if row.token == "LEGACY":
                out.append(f"{rid}: LEGACY can be kept, never entered")
            continue
        if row.date < old.date:
            out.append(f"{rid}: date moved back from {old.date} to {row.date}")
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
    try:
        base = base or _lint_git.base_ref(repo_root)
        base_text = _lint_git.show(repo_root, base, L.LEDGER_PATH) or None
        epoch = int(_lint_git.git(repo_root, "show", "-s", "--format=%ct", _branch_point(repo_root, base)).strip())
        base_date = _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc).date().isoformat()
    except _lint_git.GitUnavailable as exc:
        if _lint_git.must_not_skip(f"no merge-base for the ledger's diff-scoped checks: {exc}"):
            return failures, 2
        print(f"SKIP ledger base checks (no merge-base: {exc})", file=sys.stderr)
        return failures, 0
    failures.extend(f"{L.LEDGER_PATH}: {p}" for p in base_problems(rows, base_text, base_date))
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
