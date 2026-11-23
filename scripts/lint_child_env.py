#!/usr/bin/env python3
"""Process spawns whose environment does not come from ``child_env``: counted, printed, ratcheted (#1593).

``maxim.utils.child_env.child_env`` is the one place a child's environment is built. It removes the
``PYTHONPATH`` entries that a ``.pth`` hook (the GStreamer wheels') points into this interpreter's own
site-packages. A spawn that inherits ``os.environ``, or copies it by hand, hands those entries to the child:
- a child ``maxim`` imports a stray ``site-packages`` copy instead of the checkout;
- a child of another interpreter imports this venv's packages.

**What counts as a spawn:** a call to
- ``subprocess.run`` / ``Popen`` / ``call`` / ``check_call`` / ``check_output``;
- ``asyncio.create_subprocess_exec`` / ``create_subprocess_shell``;
- an ``os.exec*`` / ``os.spawn*`` / ``os.posix_spawn*`` function.

**A spawn is clean when** its environment argument (``env=``, or the ``env`` positional of
``os.execve``/``os.execvpe``/``os.spawnve``/``os.posix_spawn``) is one of these:
- a call to a sanctioned builder;
- a name assigned, in the same function, from an expression that calls one;
- an expression that calls one.

The sanctioned builders are ``child_env`` itself, the builders listed in :data:`SANCTIONED` (each checked to
call ``child_env``; see ``tests/unit/test_lint_child_env.py``), and any function in the same file whose body
calls a sanctioned builder. An ``os.exec*`` without an ``e`` suffix always counts, because it inherits
``os.environ``.

**Ratchet:** every file is grandfathered at its ``origin/main`` count, and a branch may not raise any file's
count. New files start at zero, and burning a count down is free. This catches FORGETTING, not evasion: an
environment built through a variable from another module, or a spawn reached through an alias, escapes it.

Exits: 0 clean; 1 a file's count rose (stderr); 2 unexpected error.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_git import GitUnavailable, base_ref, count_ratchet, must_not_skip  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCOPES = ("src/maxim", "scripts")
CANONICAL = frozenset({"src/maxim/utils/child_env.py", "scripts/_child_env.py"})
#: Builders in other modules that return a ``child_env``-derived environment.
SANCTIONED = frozenset({"child_env", "host_tool_env", "git_env"})
_SUBPROCESS = frozenset({"run", "Popen", "call", "check_call", "check_output"})
_ASYNCIO = frozenset({"create_subprocess_exec", "create_subprocess_shell"})
#: os functions that take an environment, and the positional index of it.
_OS_ENV_POS = {
    "execve": 2,
    "execvpe": 2,
    "execle": -1,
    "execlpe": -1,
    "spawnve": 3,
    "spawnvpe": 3,
    "spawnle": -1,
    "spawnlpe": -1,
    "posix_spawn": 3,
    "posix_spawnp": 3,
}
_OS_NO_ENV = frozenset({"execv", "execvp", "execl", "execlp", "spawnv", "spawnvp", "spawnl", "spawnlp"})


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def _local_builders(tree: ast.AST) -> set[str]:
    """Functions in this file whose bodies call a sanctioned builder, resolved to a fixed point."""
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    builders = set(SANCTIONED)
    changed = True
    while changed:
        changed = False
        for name, fn in funcs.items():
            if name not in builders and _called_names(fn) & builders:
                builders.add(name)
                changed = True
    return builders


def _spawn_kind(call: ast.Call) -> tuple[str, int | None] | None:
    """(spelling, env positional index or None) for a spawn call; None when it is not one."""
    f = call.func
    if not isinstance(f, ast.Attribute) or not isinstance(f.value, ast.Name):
        return None
    mod, attr = f.value.id, f.attr
    if mod == "subprocess" and attr in _SUBPROCESS:
        return f"subprocess.{attr}", None
    if mod == "asyncio" and attr in _ASYNCIO:
        return f"asyncio.{attr}", None
    if mod == "os" and attr in _OS_ENV_POS:
        return f"os.{attr}", _OS_ENV_POS[attr]
    if mod == "os" and attr in _OS_NO_ENV:
        return f"os.{attr}", None
    return None


def _env_arg(call: ast.Call, pos: int | None) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "env":
            return kw.value
    if pos is not None and call.args:
        try:
            arg = call.args[pos]
        except IndexError:
            return None
        return None if isinstance(arg, ast.Starred) else arg
    return None


def _assignments(fn: ast.AST) -> dict[str, list[ast.expr]]:
    out: dict[str, list[ast.expr]] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            out.setdefault(node.target.id, []).append(node.value)
    return out


def _clean(env: ast.expr | None, scope: ast.AST, builders: set[str]) -> bool:
    if env is None:
        return False
    if _called_names(env) & builders:
        return True
    if isinstance(env, ast.Name):
        values = _assignments(scope).get(env.id, [])
        return bool(values) and all(_called_names(v) & builders for v in values)
    return False


def unclean_spawns(text: str) -> list[tuple[int, str]]:
    """(line, spelling) for every spawn whose environment is not built by ``child_env``."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    builders = _local_builders(tree)
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def enclosing(node: ast.AST) -> ast.AST:
        while node in parent:
            node = parent[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        return tree

    hits: list[tuple[int, str]] = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        kind = _spawn_kind(call)
        if kind is None:
            continue
        spelling, pos = kind
        if spelling[3:] in _OS_NO_ENV and spelling.startswith("os."):
            hits.append((call.lineno, spelling))  # inherits os.environ by construction
            continue
        if pos == -1:  # execle-style: the environment is the last positional
            last = call.args[-1] if call.args else None
            env = None if last is None or isinstance(last, ast.Starred) else last
        else:
            env = _env_arg(call, pos)
        if not _clean(env, enclosing(call), builders):
            hits.append((call.lineno, spelling))
    return sorted(hits)


def counts(repo_root: Path = REPO_ROOT) -> dict[str, list[tuple[int, str]]]:
    """{repo-relative path: unclean spawns} for every offending file, sorted by path."""
    out: dict[str, list[tuple[int, str]]] = {}
    for scope in SCOPES:
        for path in sorted((repo_root / scope).rglob("*.py")):
            rel = path.relative_to(repo_root).as_posix()
            if rel in CANONICAL:
                continue
            sites = unclean_spawns(path.read_text(errors="replace"))
            if sites:
                out[rel] = sites
    return out


def main() -> int:
    try:
        current = counts()
    except OSError as exc:
        print(f"ERROR: cannot scan {SCOPES}: {exc}", file=sys.stderr)
        return 2
    total = sum(len(v) for v in current.values())
    print(f"child_env ratchet: {total} spawn(s) without a child_env environment in {len(current)} file(s):")
    for rel, sites in current.items():
        print(f"  {len(sites):3d}  {rel}")
    try:
        base = base_ref(REPO_ROOT)
    except GitUnavailable as exc:
        if must_not_skip(str(exc)):
            return 2
        print(f"INFO: no base ref available; skipping the diff-scoped ratchet ({exc})")
        return 0
    fails: list[str] = []
    try:
        for scope in SCOPES:
            fails += count_ratchet(
                REPO_ROOT,
                base,
                scope,
                unclean_spawns,
                exclude=CANONICAL,
                what="spawns without a child_env environment",
                advice=(
                    "pass env=child_env(...) (maxim.utils.child_env; scripts: from _child_env import child_env), "
                    "or os.execve(..., child_env()) for a re-exec (#1593). The count only ratchets down."
                ),
            )
    except GitUnavailable as exc:
        if must_not_skip(f"git failed mid-run: {exc}"):
            return 2
        print(f"INFO: diff-scoped ratchet skipped mid-run ({exc})")
        return 0
    if fails:
        print("child_env ratchet FAILED:", file=sys.stderr)
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("child_env ratchet: no file's count rose on this branch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
