#!/usr/bin/env python3
"""`os.replace` outside utils/atomic_io.py — counted, printed, and ratcheted (roadmap 1.1.x item 16.3).

CLAUDE.md's atomic-persistence invariant: writers use ``maxim.utils.atomic_io.atomic_write_json``
(fsync + tmp cleanup) and do not hand-roll ``open().write()`` + ``os.replace()``. The
KNOWN-GAP note admitted a count of hand-rolled sites that was "detection-only, not enforced"
— and the number in the note drifted from the code (the note said 17 from a text grep that
also counted comments; the AST call-site count is what this lint prints). A stale quantified
confession was the score card's specific Documentation-honesty deduction, so:

1. **Print the truth every run.** The per-file and total counts of ``os.replace(`` CALL
   sites (AST, so comments and docstrings do not count) in ``src/maxim/`` outside
   ``utils/atomic_io.py``. CLAUDE.md cites this output, not a number.
2. **Ratchet, diff-scoped.** Every file is grandfathered at its ``origin/main`` count;
   a branch may not raise any file's count (the lint_no_silent_swallows.py shape). New
   files start at zero. Burn-down (count goes DOWN) is free and is the point.

Catches FORGETTING, not evasion: ``from os import replace`` / ``shutil.move`` / ``Path.replace``
escape the AST match (module aliases ``import os as X`` are resolved); extend
``replace_call_lines`` if one shows up in review.

Exits: 0 clean; 1 a file's count rose (stderr); INFO + 0 for the ratchet when no base ref.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = Path("src/maxim")
CANONICAL = Path("src/maxim/utils/atomic_io.py")


def replace_call_lines(text: str) -> list[int]:
    """1-indexed lines of ``os.replace(...)`` call sites, through any ``import os as X`` alias.

    (The first draft matched the bare name only and missed ``models/download.py``'s
    ``_os.replace`` — the count would have been 6, not 7.)
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    os_names = {"os"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    os_names.add(alias.asname or "os")
    hits: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "replace"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in os_names
        ):
            hits.append(node.lineno)
    return sorted(hits)


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def _base(cwd: Path) -> str | None:
    for ref in ("origin/main", "main"):
        try:
            return _git("merge-base", ref, "HEAD", cwd=cwd).strip()
        except RuntimeError:
            continue
    return None


def counts(repo_root: Path = REPO_ROOT) -> dict[str, list[int]]:
    """{repo-relative path: call-site lines} for every offending file, sorted by path."""
    out: dict[str, list[int]] = {}
    for path in sorted((repo_root / SRC).rglob("*.py")):
        rel = path.relative_to(repo_root)
        if rel == CANONICAL:
            continue
        lines = replace_call_lines(path.read_text(errors="replace"))
        if lines:
            out[rel.as_posix()] = lines
    return out


def ratchet_violations(repo_root: Path, base: str, current: dict[str, list[int]]) -> list[str]:
    fails: list[str] = []
    changed = [
        f
        for f in _git("diff", "--name-only", base, "HEAD", "--", SRC.as_posix(), cwd=repo_root).split()
        if f.endswith(".py") and f != CANONICAL.as_posix()
    ]
    for rel in changed:
        new_count = len(current.get(rel, []))
        try:
            old_text = _git("show", f"{base}:{rel}", cwd=repo_root)
        except RuntimeError:
            old_text = ""  # new file — grandfathered at zero
        old_count = len(replace_call_lines(old_text))
        if new_count > old_count:
            fails.append(
                f"{rel}: os.replace() call sites rose {old_count} → {new_count} on this branch "
                f"(lines {current.get(rel)}) — use maxim.utils.atomic_io.atomic_write_json "
                "(CLAUDE.md atomic-persistence invariant); the count only ratchets down"
            )
    return fails


def main() -> int:
    current = counts()
    total = sum(len(v) for v in current.values())
    print(f"atomic_io ratchet: {total} os.replace() call site(s) outside {CANONICAL} in {len(current)} file(s):")
    for rel, lines in current.items():
        print(f"  {len(lines):2d}  {rel}  (lines {', '.join(map(str, lines))})")
    base = _base(REPO_ROOT)
    if base is None:
        print("INFO: no base ref (origin/main) available; skipping the diff-scoped ratchet")
        return 0
    fails = ratchet_violations(REPO_ROOT, base, current)
    if fails:
        print("atomic_io ratchet FAILED:", file=sys.stderr)
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("atomic_io ratchet: no file's count rose on this branch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
