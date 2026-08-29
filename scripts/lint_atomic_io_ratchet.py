#!/usr/bin/env python3
"""Hand-rolled atomic renames outside utils/atomic_io.py — counted, printed, ratcheted (roadmap 1.1.x item 16.3).

CLAUDE.md's atomic-persistence invariant: writers go through
``maxim.utils.atomic_io`` (fsync + tmp cleanup) instead of hand-rolling
``open().write()`` + a rename. The KNOWN-GAP note admitted a count that was
"detection-only, not enforced" — and the number in the note had drifted from the
code (it counted comments and docstrings, and it saw only ONE spelling). A stale
quantified confession was the 2026-08-27 score card's specific Documentation-honesty
deduction, so:

1. **Print the truth every run.** Per-file and total AST CALL-SITE counts of the four
   hand-rolled-rename spellings — ``os.replace``/``os.rename`` (through any
   ``import os as X`` alias) and ``Path.replace``/``Path.rename`` (an attribute call
   with exactly one positional argument; ``dataclasses.replace`` is resolved and
   excluded, and ``str.replace`` takes two arguments so it never matches) — in
   ``src/maxim/`` outside the canonical writer. CLAUDE.md cites THIS output.
2. **Ratchet, diff-scoped.** Every file is grandfathered at its ``origin/main``
   count; a branch may not raise any file's count (shared ratchet from
   ``scripts/_lint_git.py``). New files start at zero. Burn-down is free.

**What the number does and does not mean** (the review-round correction, 2026-08-29):
it counts hand-rolled atomic renames, NOT "JSON written without ``atomic_write_json``"
— as of 2026-08-29 not one counted site writes JSON. Five duplicate
``atomic_write_text``; ``hivemind/bundle.py`` (zip) and ``models/download.py`` (GGUF)
write BYTES, for which ``atomic_io`` exposes no writer at all, so burning those down
needs an ``atomic_write_bytes`` first. The first draft of this lint matched only the
bare ``os.replace`` spelling and reported 6 where the truth was 12 — it missed the
``import os as _os`` alias in ``models/download.py``, ``Path.replace`` on the
provenance decision log (``runtime/decision_log.py``), and ``os.rename`` in
``inference/transcribe_audio.py``.

Catches FORGETTING, not evasion: ``shutil.move``, ``Path.replace`` reached through a
variable holding the bound method, and renames in C extensions escape the AST match.

Exits: 0 clean; 1 a file's count rose (stderr); 2 unexpected error.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_git import GitUnavailable, base_ref, count_ratchet, must_not_skip  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = Path("src/maxim")
CANONICAL = Path("src/maxim/utils/atomic_io.py")
_RENAME_ATTRS = ("replace", "rename")


def _module_aliases(tree: ast.AST, module: str) -> set[str]:
    names = {module}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    names.add(alias.asname or module)
    return names


def rename_call_sites(text: str) -> list[tuple[int, str]]:
    """(line, spelling) for every hand-rolled atomic-rename call site."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    os_names = _module_aliases(tree, "os")
    dc_names = _module_aliases(tree, "dataclasses")
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        if attr not in _RENAME_ATTRS:
            continue
        recv = node.func.value
        if isinstance(recv, ast.Name) and recv.id in os_names:
            hits.append((node.lineno, f"os.{attr}"))
        elif isinstance(recv, ast.Name) and recv.id in dc_names:
            continue  # dataclasses.replace(obj) — same shape, unrelated
        elif len(node.args) == 1 and not node.keywords:
            hits.append((node.lineno, f"Path.{attr}"))  # str.replace takes two args
    return sorted(hits)


def counts(repo_root: Path = REPO_ROOT) -> dict[str, list[tuple[int, str]]]:
    """{repo-relative path: call sites} for every offending file, sorted by path."""
    out: dict[str, list[tuple[int, str]]] = {}
    for path in sorted((repo_root / SRC).rglob("*.py")):
        rel = path.relative_to(repo_root)
        if rel == CANONICAL:
            continue
        sites = rename_call_sites(path.read_text(errors="replace"))
        if sites:
            out[rel.as_posix()] = sites
    return out


def main() -> int:
    try:
        current = counts()
    except OSError as exc:
        print(f"ERROR: cannot scan {SRC}: {exc}", file=sys.stderr)
        return 2
    total = sum(len(v) for v in current.values())
    print(f"atomic_io ratchet: {total} hand-rolled rename call site(s) outside {CANONICAL} in {len(current)} file(s):")
    for rel, sites in current.items():
        spellings = ", ".join(f"{s} L{ln}" for ln, s in sites)
        print(f"  {len(sites):2d}  {rel}  ({spellings})")
    try:
        base = base_ref(REPO_ROOT)
    except GitUnavailable as exc:
        if must_not_skip(str(exc)):
            return 2
        print(f"INFO: no base ref available; skipping the diff-scoped ratchet ({exc})")
        return 0
    try:
        fails = count_ratchet(
            REPO_ROOT,
            base,
            SRC.as_posix(),
            rename_call_sites,
            exclude=frozenset({CANONICAL.as_posix()}),
            what="hand-rolled rename call sites",
            advice=(
                "persist through maxim.utils.atomic_io (atomic_write_text/json/secret); a BYTES payload has no "
                "canonical writer yet, so adding one is the prerequisite for that burn-down, not a reason to "
                "hand-roll (CLAUDE.md atomic-persistence invariant). The count only ratchets down."
            ),
        )
    except GitUnavailable as exc:
        print(f"INFO: diff-scoped ratchet skipped mid-run ({exc})")
        return 0
    if fails:
        print("atomic_io ratchet FAILED:", file=sys.stderr)
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("atomic_io ratchet: no file's count rose on this branch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
