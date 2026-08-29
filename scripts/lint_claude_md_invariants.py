#!/usr/bin/env python3
"""Lint the canonical agent-guidance corpus and its provider-neutral adapter.

Five checks (the first is the original Principle 5 lint; the next two were added by
docs/plans/archive/claude_md_diet.md, 2026-08-13; the fourth closes the AGENTS.md drift seam;
the fifth is roadmap 1.1.x item 16.9 from the Exp 53/53b release-day incident):

1. **Guard citations.** For each `[engineering]` invariant, the body must contain a
   `Regression guard:` reference; for each `[behavioral]` invariant, a `Roy experiment:`
   reference. In CLAUDE.md the audit covers the "Lessons learned" and "Architectural
   invariants" sections (as always); in `docs/agents/*.md` (the per-subsystem working
   briefs) EVERY tagged opener anywhere in the file is audited — briefs carry relocated
   invariant stubs and get no section exemption.
   Missing references are visible coverage gaps by design and fail the lint.
   Plural-tolerant matching ("Regression guards" / "Roy experiments" both count).

2. **Token ceiling.** CLAUDE.md must stay under ~12K estimated tokens (`len(text) // 4`,
   the same chars-per-token estimate family the proxy admission gate uses — deliberately
   dependency-free). The diet target is ≤10K; the ceiling has headroom so the ledger
   cannot silently regrow to its pre-2026-08 ~63K-token size.

3. **Link existence.** Every markdown link in CLAUDE.md and the briefs whose target is a
   repo-relative path (notably the `docs/lessons/<slug>.md` archive links the compressed
   stubs point at) must resolve to an existing file. External URLs and paths escaping the
   repo root are skipped.

4. **Pointer-only AGENTS.md.** AGENTS.md must match the frozen provider-neutral adapter
   exactly. It points auto-loading tools at CLAUDE.md but duplicates no checks, routing
   entries, or project rules that could drift independently.

5. **EARNED ledger rows cite their data.** In `docs/plans/behavioral_graduation_candidates.md`
   every table row with a cell starting `**EARNED` (case-insensitive — Tier 3 writes `**Earned <date>**`) must carry a `Regression guard:`
   field AND either a markdown link into `docs/experiments/data/` that resolves, or a dated
   data-lost annotation (`data lost … YYYY-MM-DD` / `Data lost (YYYY-MM-DD)`). A row went
   EARNED on 2026-08-26 from records whose code-under-test could not be established; the
   lesson (docs/lessons/experiment-prereg-precedes-data.md) is that the ledger must point at
   committed data, not at a claim. Verified to fail 3/3 on the pre-fix ledger (rows L185
   EC pattern completion, L186 SEM pain → NAc, L188 substrate-primary — the third had a
   guard citing pre-registrations but no data link although Exp 42's data is committed).

Exits:
  0 — all checks pass.
  1 — one or more violations; details printed to stderr.
  2 — CLAUDE.md missing or unreadable.

Designed to run in CI from the repo root; not parametrized intentionally so the workflow
step stays a single-line invocation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ENG_FIELD_PATTERN = re.compile(r"regression\s+guards?\s*:", re.IGNORECASE)
BEH_FIELD_PATTERN = re.compile(r"roy\s+experiments?\s*:", re.IGNORECASE)

# Sections to audit. Section ends at the next "## " header.
TARGET_SECTIONS = ("## Lessons learned", "## Architectural invariants")

# claude_md_diet: CLAUDE.md must not regrow. Estimate is chars/4 (dependency-free).
TOKEN_CEILING = 12_000

EXPECTED_AGENTS_ADAPTER = """# AGENTS.md — provider-neutral entrypoint

> The canonical instruction corpus for this repository is [CLAUDE.md](CLAUDE.md).
> This file exists only for tools that auto-load `AGENTS.md`.

Before doing any work in this repository, read `CLAUDE.md` in full and follow its
required checks, safety rules, routing table, and subsystem-reading instructions.

Do not add project rules, commands, routing entries, or subsystem knowledge here.
Put cross-cutting guidance in `CLAUDE.md` and scoped guidance in `docs/agents/`.
CI enforces this pointer-only adapter byte-for-byte to prevent instruction drift.
"""

MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

# Check 5 — the behavioral-graduation ledger. A row is EARNED when any cell starts with
# **EARNED (covers "EARNED (de facto)", "EARNED post-1.0", "EARNED 2026-08-25 via …").
LEDGER_PATH = Path("docs/plans/behavioral_graduation_candidates.md")
EARNED_CELL = re.compile(r"^\*\*EARNED", re.IGNORECASE)  # the Tier-3 form is `**Earned <date>**`
NON_GATED_MARKERS = ("dry_run", "dryrun", "nonfrozen")  # kept in step with lint_prereg_precedes_data.py
DATA_DIR_MARK = "docs/experiments/data/"
DATA_LOST = re.compile(r"data[- ]lost\b[^|]{0,120}?\d{4}-\d{2}-\d{2}", re.IGNORECASE)

# Invariant opener: line starts with optional bullet prefix, then **[engineering] or **[behavioral].
# Lessons learned uses no bullet; Architectural invariants uses "- ".
OPENER_PATTERN = re.compile(r"^[-\s]*\*\*\[(engineering|behavioral)\]")

# Extract a title snippet for nicer error reporting (best-effort, falls back to the raw line).
TITLE_AFTER_TAG = re.compile(r"\*\*\[(?:engineering|behavioral)\][^*]*?\*\*")


def _title_snippet(line: str, max_chars: int = 80) -> str:
    """Pull a human-readable title out of an opener line for error messages."""
    stripped = line.lstrip("- ").strip()
    # Drop the closing ** of the bold-title block if we can find it; what follows usually starts
    # with ":" or "(" or a regular sentence.
    m = TITLE_AFTER_TAG.search(stripped)
    if m:
        # Use the content INSIDE the bold-title block (after the tag).
        bold_block = m.group(0)
        # Strip leading **[tag] and trailing **
        inner = re.sub(r"^\*\*\[[^\]]+\]\s*", "", bold_block).rstrip("*").strip()
        if inner:
            return (inner[: max_chars - 1] + "…") if len(inner) > max_chars else inner
    return (stripped[: max_chars - 1] + "…") if len(stripped) > max_chars else stripped


def _collect_guard_violations(lines: list[str], *, audit_whole_file: bool) -> list[tuple[int, str, str, str]]:
    """Return (line_num, tag, title, missing_field) violations for one document.

    audit_whole_file=False keeps the original CLAUDE.md behavior (only the
    TARGET_SECTIONS are audited); True audits every tagged opener (briefs).
    """
    violations: list[tuple[int, str, str, str]] = []

    in_target = audit_whole_file
    opener_line: int | None = None
    current_tag: str | None = None
    current_title: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_tag, current_title, current_body, opener_line
        if current_tag is None or opener_line is None:
            return
        body_text = " ".join(current_body)
        if current_tag == "engineering" and not ENG_FIELD_PATTERN.search(body_text):
            violations.append((opener_line, "engineering", current_title or "", "Regression guard:"))
        elif current_tag == "behavioral" and not BEH_FIELD_PATTERN.search(body_text):
            violations.append((opener_line, "behavioral", current_title or "", "Roy experiment:"))
        current_tag = None
        current_title = None
        current_body = []
        opener_line = None

    for i, line in enumerate(lines, start=1):
        # Section header — flush + flip in/out of target.
        if line.startswith("## "):
            flush()
            if not audit_whole_file:
                in_target = any(line.startswith(h) for h in TARGET_SECTIONS)
            continue

        if not in_target:
            continue

        # New invariant opener?
        if OPENER_PATTERN.match(line):
            flush()
            tag_match = OPENER_PATTERN.match(line)
            assert tag_match is not None  # for type checker
            current_tag = tag_match.group(1)
            current_title = _title_snippet(line)
            current_body = [line]
            opener_line = i
            continue

        # In CLAUDE.md a blank line ends the body (invariants are single-paragraph).
        # In briefs (audit_whole_file) merged stubs are legitimately multi-paragraph —
        # the body extends to the next tagged opener or header instead.
        if current_tag is not None and line.strip() == "" and not audit_whole_file:
            flush()
            continue

        # Otherwise accumulate body.
        if current_tag is not None:
            current_body.append(line)

    flush()
    return violations


def _broken_repo_links(doc_path: Path, repo_root: Path) -> list[tuple[int, str]]:
    """(line_num, target) for markdown links to repo-relative paths that don't exist."""
    broken: list[tuple[int, str]] = []
    for i, line in enumerate(doc_path.read_text().split("\n"), start=1):
        for m in MARKDOWN_LINK.finditer(line):
            target = m.group(1).split("#")[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            # Guard lines are copied byte-exact from CLAUDE.md (repo-root-relative paths),
            # so accept a link if it resolves relative to EITHER the containing file's
            # directory (standard markdown) or the repo root (the CLAUDE.md convention).
            if target.startswith("/"):
                candidates = [Path(target)]
            else:
                candidates = [(doc_path.parent / target).resolve(), (repo_root / target).resolve()]
            inside = [p for p in candidates if p.is_relative_to(repo_root)]
            if not inside:
                continue  # e.g. links into ~/.claude memory — out of lint scope
            if not any(p.exists() for p in inside):
                broken.append((i, m.group(1)))
    return broken


def _ledger_earned_violations(ledger_path: Path, repo_root: Path) -> list[tuple[int, str, str]]:
    """(line_num, row_title, problem) for EARNED ledger rows without a data citation."""
    violations: list[tuple[int, str, str]] = []
    for i, line in enumerate(ledger_path.read_text().split("\n"), start=1):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not any(EARNED_CELL.match(c) for c in cells):
            continue
        title = (cells[0] if not cells[0].isdigit() else cells[1])[:70]
        if not ENG_FIELD_PATTERN.search(line):
            violations.append((i, title, "EARNED row has no 'Regression guard:' field"))
            continue
        data_links = [t for t in MARKDOWN_LINK.findall(line) if DATA_DIR_MARK in t or "experiments/data/" in t]
        if data_links:
            for target in data_links:
                target = target.split("#")[0]
                if any(mk in target for mk in NON_GATED_MARKERS):
                    violations.append((i, title, f"data link is a non-gated shakedown capture, not evidence: {target}"))
                    continue
                candidates = [(ledger_path.parent / target).resolve(), (repo_root / target).resolve()]
                if not any(c.exists() for c in candidates):
                    violations.append((i, title, f"data link does not resolve: {target}"))
            continue
        if DATA_LOST.search(line):
            continue
        violations.append(
            (
                i,
                title,
                "EARNED row cites no docs/experiments/data/ path and carries no dated data-lost annotation "
                "(item 16.9: an EARNED status points at committed data or says, with a date, that it is lost)",
            )
        )
    return violations


def lint(claude_md_path: Path) -> int:
    if not claude_md_path.exists():
        print(f"ERROR: {claude_md_path} does not exist", file=sys.stderr)
        return 2

    try:
        text = claude_md_path.read_text()
    except OSError as exc:
        print(f"ERROR: failed to read {claude_md_path}: {exc}", file=sys.stderr)
        return 2

    repo_root = claude_md_path.parent
    failed = False

    # Check 0 — AGENTS.md is a frozen pointer, not a second instruction corpus.
    agents_md_path = repo_root / "AGENTS.md"
    try:
        agents_text = agents_md_path.read_text()
    except OSError as exc:
        print(f"ERROR: failed to read {agents_md_path}: {exc}", file=sys.stderr)
        return 2
    if agents_text != EXPECTED_AGENTS_ADAPTER:
        failed = True
        print(
            "FAIL: AGENTS.md must remain the exact pointer-only adapter defined in "
            "scripts/lint_claude_md_invariants.py; put substantive guidance in CLAUDE.md "
            "or docs/agents/.",
            file=sys.stderr,
        )

    # Check 1 — guard citations, CLAUDE.md (target sections) + docs/agents briefs (whole file).
    docs: list[tuple[Path, bool]] = [(claude_md_path, False)]
    docs += [(p, True) for p in sorted((repo_root / "docs" / "agents").glob("*.md"))]
    n_docs = 0
    for doc, whole in docs:
        n_docs += 1
        violations = _collect_guard_violations(doc.read_text().split("\n"), audit_whole_file=whole)
        if violations:
            failed = True
            rel = doc.relative_to(repo_root)
            print(
                f"FAIL: {len(violations)} invariant(s) in {rel} missing required field per CLAUDE.md Principle 5.",
                file=sys.stderr,
            )
            for line_num, tag, title, missing in violations:
                print(f"  {rel}:{line_num}: [{tag}] missing '{missing}' field", file=sys.stderr)
                print(f"    title: {title}", file=sys.stderr)

    # Check 2 — token ceiling on CLAUDE.md (chars/4 estimate; see module docstring).
    est_tokens = len(text) // 4
    if est_tokens > TOKEN_CEILING:
        failed = True
        print(
            f"FAIL: CLAUDE.md is ~{est_tokens} estimated tokens (> ceiling {TOKEN_CEILING}). "
            "Per docs/plans/archive/claude_md_diet.md the core must not regrow: move the new entry's "
            "full prose to docs/lessons/<slug>.md and its stub to the owning docs/agents/ brief.",
            file=sys.stderr,
        )

    # Check 3 — repo-relative markdown links must resolve (CLAUDE.md + briefs + adapter).
    link_docs = [doc for doc, _ in docs] + [agents_md_path]
    for doc in link_docs:
        broken = _broken_repo_links(doc, repo_root)
        if broken:
            failed = True
            rel = doc.relative_to(repo_root)
            print(f"FAIL: {len(broken)} broken repo-relative link(s) in {rel}:", file=sys.stderr)
            for line_num, target in broken:
                print(f"  {rel}:{line_num}: {target}", file=sys.stderr)

    # Check 5 — EARNED rows in the behavioral-graduation ledger cite committed data.
    ledger_path = repo_root / LEDGER_PATH
    if not ledger_path.exists():
        print(f"ERROR: {LEDGER_PATH} missing — the EARNED-row check cannot run", file=sys.stderr)
        return 2
    ledger_violations = _ledger_earned_violations(ledger_path, repo_root)
    if ledger_violations:
        failed = True
        print(
            f"FAIL: {len(ledger_violations)} EARNED row(s) in {LEDGER_PATH} without a resolving "
            "docs/experiments/data/ link or a dated data-lost annotation (roadmap 1.1.x item 16.9):",
            file=sys.stderr,
        )
        for line_num, title, problem in ledger_violations:
            print(f"  {LEDGER_PATH}:{line_num}: {problem}", file=sys.stderr)
            print(f"    row: {title}", file=sys.stderr)

    if failed:
        print(
            "\nSee CLAUDE.md 'Working principles for new mechanisms' Principle 5 for the format "
            "convention.\nEach [engineering] invariant declares 'Regression guard: <path>' at the "
            "end of its body;\neach [behavioral] invariant declares 'Roy experiment: <path>'.",
            file=sys.stderr,
        )
        return 1

    print(
        f"PASS: {n_docs} doc(s) audited — all tagged invariants carry the required field, "
        f"CLAUDE.md ~{est_tokens} est. tokens (ceiling {TOKEN_CEILING}), AGENTS.md adapter "
        "matches, all repo links resolve, every EARNED ledger row cites its data."
    )
    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    return lint(repo_root / "CLAUDE.md")


if __name__ == "__main__":
    sys.exit(main())
