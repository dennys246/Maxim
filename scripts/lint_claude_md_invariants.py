#!/usr/bin/env python3
"""Lint CLAUDE.md + docs/agents/ briefs to enforce Principle 5 and the claude_md_diet contract.

Three checks (the first is the original Principle 5 lint; the other two were added by
docs/plans/claude_md_diet.md, 2026-08-13):

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

MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

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
            "Per docs/plans/claude_md_diet.md the core must not regrow: move the new entry's "
            "full prose to docs/lessons/<slug>.md and its stub to the owning docs/agents/ brief.",
            file=sys.stderr,
        )

    # Check 3 — repo-relative markdown links must resolve (CLAUDE.md + briefs).
    for doc, _ in docs:
        broken = _broken_repo_links(doc, repo_root)
        if broken:
            failed = True
            rel = doc.relative_to(repo_root)
            print(f"FAIL: {len(broken)} broken repo-relative link(s) in {rel}:", file=sys.stderr)
            for line_num, target in broken:
                print(f"  {rel}:{line_num}: {target}", file=sys.stderr)

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
        f"CLAUDE.md ~{est_tokens} est. tokens (ceiling {TOKEN_CEILING}), all repo links resolve."
    )
    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    return lint(repo_root / "CLAUDE.md")


if __name__ == "__main__":
    sys.exit(main())
