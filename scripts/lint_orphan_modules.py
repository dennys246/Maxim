#!/usr/bin/env python3
"""Ratchet lint: no NEW orphan modules ship in the wheel.

CLAUDE.md's dead-code lesson: "15 dead modules (~8,500 LOC) shipping in the wheel", found
once by a manual grep, with "no automated test enforces". This is that enforcement — as a
ratchet, not a purge: it lists every `src/maxim/**` module that nothing in the repo
references, grandfathers the current set, and fails only when a NEW orphan appears. Removing
a module from the grandfather list (by wiring or deleting it) is the ratchet-down.

Deliberately NOT a bulk-delete: the repo is intimately wired and CLAUDE.md prefers
dormancy over whim-deletion (secondary-breakage history), so the grandfathered set is
tracked here for triage, not auto-removed.

"Referenced" is judged PERMISSIVELY (import parts/names, any textual occurrence across
src/tests/scripts, or a mention in pyproject.toml) — the safe direction is to under-report
a dead module, never to falsely flag a live one.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CANDIDATE_ROOT = _REPO / "src" / "maxim"
_REFERENCE_ROOTS = [_REPO / "src", _REPO / "tests", _REPO / "scripts"]

# Modules with no textual reference anywhere but which are reached dynamically
# (entry points, lazy importlib by string). Never "orphans" even if unreferenced.
_DYNAMIC_ALLOW = {
    "__main__",  # `python -m maxim` entry point
}

# Grandfathered orphans as of this lint's introduction (2026-09-10): NONE. Every
# `src/maxim/**` module is referenced somewhere in src/tests/scripts (the 8 that looked
# orphaned in a src-only scan each turned out to have a dedicated test). Ratchet-DOWN
# only: never ADD to this set to make a new orphan pass — wire it, delete it, or (if
# reached purely dynamically) add it to _DYNAMIC_ALLOW with a reason.
_GRANDFATHERED: set[str] = set()


def _candidate_modules() -> dict[str, Path]:
    mods: dict[str, Path] = {}
    for p in _CANDIDATE_ROOT.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        mods[p.stem] = p
    return mods


def _collect_references() -> tuple[set[str], str]:
    """Return (imported-name parts, concatenated text of all reference files)."""
    imported: set[str] = set()
    texts: list[str] = []
    for root in _REFERENCE_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if p.resolve() == Path(__file__).resolve():
                continue  # don't let this lint's own text (e.g. _DYNAMIC_ALLOW) count as a reference
            text = p.read_text(encoding="utf-8", errors="ignore")
            texts.append(text)
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imported.update(a.name.split("."))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported.update(node.module.split("."))
                    for a in node.names:
                        imported.add(a.name)
    pyproject = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    return imported, "\n".join(texts) + "\n" + pyproject


def _orphans() -> list[str]:
    mods = _candidate_modules()
    imported, blob = _collect_references()
    orphans = []
    for name, path in mods.items():
        if name in _DYNAMIC_ALLOW or name in imported:
            continue
        own = path.read_text(encoding="utf-8", errors="ignore")
        # Textual reference anywhere OUTSIDE the module's own text counts as "referenced".
        occurrences = len(re.findall(re.escape(name), blob))
        own_occurrences = len(re.findall(re.escape(name), own))
        if occurrences - own_occurrences > 0:
            continue
        orphans.append(name)
    return sorted(orphans)


def check() -> int:
    orphans = set(_orphans())
    new = sorted(orphans - _GRANDFATHERED)
    healed = sorted(_GRANDFATHERED - orphans)

    print(f"orphan-module ratchet: {len(orphans)} orphan(s); {len(_GRANDFATHERED)} grandfathered")
    for o in sorted(orphans):
        tag = "NEW" if o in new else "grandfathered"
        print(f"  [{tag}] {o}")

    if healed:
        print("\nNote: these are no longer orphaned — remove from _GRANDFATHERED to ratchet down: " + ", ".join(healed))

    if new:
        print("\norphan-module ratchet: FAIL")
        print("  New orphan module(s) with no reference anywhere (import, text, or pyproject): " + ", ".join(new))
        print("  Wire it in, delete it, or (if reached dynamically) add to _DYNAMIC_ALLOW.")
        return 1
    print("orphan-module ratchet: clean")
    return 0


if __name__ == "__main__":
    sys.exit(check())
