#!/usr/bin/env python3
"""Lint: every `maxim` import in src/ and scripts/ names a module and a name that exist (#1436).

`db274547` deleted a re-export shim, and a lazy import of it inside `MediaLoopMixin.live` turned
the robot's voice input off for six months behind one warning. The orphan-module lint cannot see
that (it asks whether a module is REACHED, not whether an import's TARGET exists), and a test only
catches it if the branch runs. This lint resolves every `import maxim...` / `from maxim... import
...` (and every relative import inside src/maxim) statically, at ANY depth: module top level,
`if TYPE_CHECKING:` blocks, class bodies and function bodies. Each import must name:

- a module that exists as `src/maxim/<path>.py` or `src/maxim/<path>/__init__.py`, and
- for `from X import Y`, a submodule `X.Y` or a name bound at `X`'s runtime top level (def, class,
  assignment, import, including inside top-level `if`/`try`/`with` blocks, but NOT inside an
  `if TYPE_CHECKING:` block, which binds nothing at runtime). A module defining `__getattr__`
  accepts its closed set of string literals.

Known blind spots (fail-open; none hides a dead import today): `importlib.import_module("maxim...")`
string literals and dotted-path registries are not resolved, and a target that re-exports with
`from x import *` accepts any name. Both are recorded in the design note's follow-ups.

Zero tolerance, no allowlist: maxim-internal code is never optional, so a dead target is a defect
wherever it sits. Why not the mypy ratchet: it only stops a file's error count rising (a new
`import-not-found` can be offset elsewhere in the file, old ones are grandfathered) and does not
cover scripts/. Design and adversarial pass: docs/plans/lazy_import_lint_1436.md.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _module_file(src: Path, dotted: str) -> Path | None:
    parts = dotted.split(".")
    if not parts or parts[0] != "maxim" or not all(parts):
        return None
    base = src.joinpath(*parts)
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _bind_targets(node: ast.AST, names: set[str]) -> None:
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)


def _runtime_bindings(stmts: list[ast.stmt], names: set[str]) -> None:
    """Names a module binds at runtime top level (not inside functions/classes, not under TYPE_CHECKING)."""
    for node in stmts:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                _bind_targets(t, names)
        elif isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                _runtime_bindings(node.body, names)
            _runtime_bindings(node.orelse, names)
        elif isinstance(node, ast.Try):
            _runtime_bindings(node.body, names)
            for h in node.handlers:
                _runtime_bindings(h.body, names)
            _runtime_bindings(node.orelse, names)
            _runtime_bindings(node.finalbody, names)
        elif isinstance(node, (ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While)):
            if isinstance(node, (ast.For, ast.AsyncFor)):
                _bind_targets(node.target, names)
            _runtime_bindings(node.body, names)
            _runtime_bindings(getattr(node, "orelse", []), names)


class _Index:
    """Parses each target module once and answers 'does module M bind name N at runtime?'."""

    def __init__(self, src: Path) -> None:
        self.src = src
        self._names: dict[Path, tuple[set[str], set[str] | None]] = {}

    def binds(self, path: Path, name: str) -> bool:
        if path not in self._names:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            names: set[str] = set()
            _runtime_bindings(tree.body, names)
            lazy: set[str] | None = None
            if "__getattr__" in names:  # a lazy module: the strings in its top-level literal tables
                lazy = set()
                for stmt in tree.body:
                    value = getattr(stmt, "value", None)
                    if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(
                        value, (ast.Dict, ast.Set, ast.Tuple, ast.List)
                    ):
                        for n in ast.walk(value):
                            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                                lazy.add(n.value)
            self._names[path] = (names, lazy)
        names, lazy = self._names[path]
        # a `from x import *` re-export can bind any name: accept rather than guess (fail-open, documented)
        return name in names or "*" in names or (lazy is not None and name in lazy)


def _package_of(src: Path, path: Path) -> list[str] | None:
    try:
        rel = path.relative_to(src)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    else:
        parts = parts[:-1]
    return parts


def _absolute(src: Path, path: Path, node: ast.ImportFrom) -> str | None:
    """The absolute dotted module of a `from ... import`, or None when it is not a maxim import."""
    if node.level == 0:
        mod = node.module or ""
        return mod if mod == "maxim" or mod.startswith("maxim.") else None
    pkg = _package_of(src, path)
    if pkg is None or node.level - 1 > len(pkg):
        return None
    base = pkg[: len(pkg) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def check(repo: Path = _REPO) -> list[str]:
    """Every dead maxim import in src/maxim and scripts/, as `path:line: message`."""
    src = repo / "src"
    index = _Index(src)
    problems: list[str] = []
    for root in (src / "maxim", repo / "scripts"):
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            rel = path.relative_to(repo).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if (a.name == "maxim" or a.name.startswith("maxim.")) and _module_file(src, a.name) is None:
                            problems.append(f"{rel}:{node.lineno}: module {a.name} does not exist")
                elif isinstance(node, ast.ImportFrom):
                    mod = _absolute(src, path, node)
                    if mod is None:
                        continue
                    target = _module_file(src, mod)
                    if target is None:
                        problems.append(f"{rel}:{node.lineno}: module {mod} does not exist")
                        continue
                    for a in node.names:
                        if a.name == "*" or _module_file(src, f"{mod}.{a.name}") is not None:
                            continue
                        if not index.binds(target, a.name):
                            problems.append(f"{rel}:{node.lineno}: {mod} has no {a.name}")
    return problems


def main() -> int:
    problems = check()
    if problems:
        print(f"dead-import lint: FAIL ({len(problems)})")
        for p in problems:
            print(f"  {p}")
        print("  A maxim import names a module or a name that does not exist: point it at the live code or delete it.")
        return 1
    print("dead-import lint: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
