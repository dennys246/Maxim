#!/usr/bin/env python3
"""Lint CLAUDE.md to enforce Principle 5 (regression-guard / experiment citation per invariant).

For each `[engineering]` invariant in the "Lessons learned" and "Architectural invariants"
sections, the body must contain a `Regression guard:` reference. For each `[behavioral]`
invariant, the body must contain a `Roy experiment:` reference.

Missing references are visible coverage gaps by design (per CLAUDE.md Principle 5) and fail
the lint. Plural-tolerant matching ("Regression guards" / "Roy experiments" both count).

Exits:
  0 — all tagged invariants in target sections carry the required field.
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


def lint(claude_md_path: Path) -> int:
    if not claude_md_path.exists():
        print(f"ERROR: {claude_md_path} does not exist", file=sys.stderr)
        return 2

    try:
        lines = claude_md_path.read_text().split("\n")
    except OSError as exc:
        print(f"ERROR: failed to read {claude_md_path}: {exc}", file=sys.stderr)
        return 2

    violations: list[tuple[int, str, str, str]] = []  # (line_num, tag, title, missing_field)

    in_target = False
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

        # Blank line ends the body (most invariants are single-paragraph).
        if current_tag is not None and line.strip() == "":
            flush()
            continue

        # Otherwise accumulate body.
        if current_tag is not None:
            current_body.append(line)

    flush()

    if not violations:
        print(f"PASS: all tagged invariants in {claude_md_path.name} carry the required field.")
        return 0

    print(f"FAIL: {len(violations)} invariant(s) missing required field per CLAUDE.md Principle 5.", file=sys.stderr)
    print(file=sys.stderr)
    for line_num, tag, title, missing in violations:
        print(f"  CLAUDE.md:{line_num}: [{tag}] missing '{missing}' field", file=sys.stderr)
        print(f"    title: {title}", file=sys.stderr)
        print(file=sys.stderr)
    print(
        "See CLAUDE.md 'Working principles for new mechanisms' Principle 5 for the format convention.\n"
        "Each [engineering] invariant declares 'Regression guard: <path>' at the end of its body;\n"
        "each [behavioral] invariant declares 'Roy experiment: <path>'.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    return lint(repo_root / "CLAUDE.md")


if __name__ == "__main__":
    sys.exit(main())
