#!/usr/bin/env python3
"""Process spawns whose environment does not come from ``child_env``: counted, printed, ratcheted (#1593).

``maxim.utils.child_env.child_env`` is the one place a child's environment is built. It removes the
``PYTHONPATH`` entries that a ``.pth`` hook (the GStreamer wheels') points into this interpreter's own
site-packages. A spawn that inherits ``os.environ``, or copies it by hand, hands those entries to the child:
- a child ``maxim`` imports a stray ``site-packages`` copy instead of the checkout;
- a child of another interpreter imports this venv's packages.

**What counts as a spawn:** a call to
- ``subprocess.run`` / ``Popen`` / ``call`` / ``check_call`` / ``check_output``;
- ``subprocess.getoutput`` / ``getstatusoutput``, which take no environment and so always count;
- ``asyncio.create_subprocess_exec`` / ``create_subprocess_shell``;
- an ``os.exec*`` / ``os.spawn*`` / ``os.posix_spawn*`` function. One without an ``e`` suffix always counts,
  because it inherits ``os.environ``.

**A spawn is clean when** its environment argument (``env=``, or the ``env`` positional of
``os.execve``/``os.execvpe``/``os.spawnve``/``os.posix_spawn``) evaluates to a ``child_env``-built
environment:
- a call to ``child_env`` or another builder in :data:`SANCTIONED`, by name or as ``module.name`` (each is
  checked to reach ``child_env`` in ``tests/unit/test_lint_child_env.py``);
- a call to a same-file function, as ``f()`` or ``self.f()``, every one of whose ``return`` values is clean.
  A method name alone never counts, so a local ``def copy()`` does not clean ``os.environ.copy()``;
- ``{**clean, "K": v}``, where every ``**`` unpack is clean, ``a if c else b``, where both branches are, or
  ``{k: v for k, v in clean.items() if ...}``, a filter of a clean environment;
- a name every assignment to which, in the spawn's own function scope, is clean, and which nothing
  ``.update()``s with ``os.environ``.

**Ratchet:** every file is grandfathered at its ``origin/main`` count, and a branch may not raise any file's
count. New files start at zero, and burning a count down is free. This catches FORGETTING, not evasion:
- an environment built in another module, or a spawn reached through an alias (``import subprocess as sp``,
  ``from subprocess import run``), escapes it;
- an environment passed in as a function PARAMETER, or unpacked from a tuple, always counts as unclean. It
  neither credits a correct caller nor catches a regressed one; pass ``env=child_env(env)`` (idempotent) to
  make such a site count as clean. An item assignment or ``|=`` after ``env = child_env()``
  (``env["PYTHONPATH"] = os.environ[...]``, ``env |= os.environ``) is not seen either; only ``.update`` is. The
  harnesses' sub-sim spawns take their environment as a parameter, which
  is why ``scripts/lint_harness_provenance.py::_env_failures`` checks where those environments are BUILT.

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
#: subprocess helpers that take no ``env``: they always inherit ``os.environ``.
_SUBPROCESS_NO_ENV = frozenset({"getoutput", "getstatusoutput"})
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
    "posix_spawn": 2,
    "posix_spawnp": 2,
}
_OS_NO_ENV = frozenset({"execv", "execvp", "execl", "execlp", "spawnv", "spawnvp", "spawnl", "spawnlp"})


def _own_nodes(fn: ast.AST):
    """Every node in ``fn``'s own scope: nested function and class bodies excluded."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _is_builder_call(node: ast.AST, local: set[str]) -> bool:
    """``child_env(...)`` / a SANCTIONED builder (by name or as ``module.name``), or a local builder (``f()``,
    ``self.f()``). A bare method name never counts on its own: ``os.environ.copy()`` is not ``copy()``."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Name):
        return f.id in SANCTIONED or f.id in local
    if isinstance(f, ast.Attribute):
        if f.attr in SANCTIONED:
            return True
        return isinstance(f.value, ast.Name) and f.value.id in ("self", "cls") and f.attr in local
    return False


def _mentions_environ(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Attribute) and n.attr == "environ" and isinstance(n.value, ast.Name) and n.value.id == "os"
        for n in ast.walk(node)
    )


def _clean_expr(expr: ast.expr | None, scope: ast.AST, local: set[str], seen: frozenset[str] = frozenset()) -> bool:
    """True when ``expr`` evaluates to a ``child_env``-built environment.

    ``seen`` holds the names being resolved: ``env = {**env, "A": "1"}`` refers to ``env`` itself, and that
    self-reference is decided by ``env``'s other assignments, not by recursing forever."""
    if expr is None:
        return False
    if _is_builder_call(expr, local):
        return True
    if isinstance(expr, ast.IfExp):
        return _clean_expr(expr.body, scope, local, seen) and _clean_expr(expr.orelse, scope, local, seen)
    if isinstance(expr, ast.Dict):
        unpacks = [v for k, v in zip(expr.keys, expr.values) if k is None]
        return bool(unpacks) and all(_clean_expr(v, scope, local, seen) for v in unpacks)
    if isinstance(expr, ast.DictComp) and len(expr.generators) == 1:
        it = expr.generators[0].iter  # {k: v for k, v in <clean>.items() if ...}: a filter of a clean environment
        return (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Attribute)
            and it.func.attr == "items"
            and _clean_expr(it.func.value, scope, local, seen)
        )
    if isinstance(expr, ast.Name):
        if expr.id in seen:
            return True  # a self-reference: the name's other assignments decide
        values: list[ast.expr] = []
        for node in _own_nodes(scope):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == expr.id for t in node.targets):
                values.append(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == expr.id:
                if node.value is not None:
                    values.append(node.value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == expr.id
                and any(_mentions_environ(arg) for arg in [*node.args, *(k.value for k in node.keywords)])
            ):
                return False  # env = child_env(); env.update(os.environ) puts the hook's value back
        inner = seen | {expr.id}
        return bool(values) and all(_clean_expr(v, scope, local, inner) for v in values)
    return False


def _local_builders(tree: ast.AST) -> set[str]:
    """Same-file functions every one of whose ``return`` values is a clean environment, to a fixed point."""
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    local: set[str] = set()
    changed = True
    while changed:
        changed = False
        for fn in funcs:
            if fn.name in local:
                continue
            returns = [n for n in _own_nodes(fn) if isinstance(n, ast.Return)]
            if returns and all(_clean_expr(r.value, fn, local) for r in returns):
                local.add(fn.name)
                changed = True
    return local


def _spawn_kind(call: ast.Call) -> tuple[str, int | None] | None:
    """(spelling, env positional index or None) for a spawn call; None when it is not one."""
    f = call.func
    if not isinstance(f, ast.Attribute) or not isinstance(f.value, ast.Name):
        return None
    mod, attr = f.value.id, f.attr
    if mod == "subprocess" and attr in _SUBPROCESS:
        return f"subprocess.{attr}", None
    if mod == "subprocess" and attr in _SUBPROCESS_NO_ENV:
        return f"subprocess.{attr}", -2
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
        if pos == -2 or (spelling.startswith("os.") and spelling[3:] in _OS_NO_ENV):
            hits.append((call.lineno, spelling))  # inherits os.environ by construction
            continue
        if pos == -1:  # execle-style: the environment is the last positional
            last = call.args[-1] if call.args else None
            env = None if last is None or isinstance(last, ast.Starred) else last
        else:
            env = _env_arg(call, pos)
        if not _clean_expr(env, enclosing(call), builders):
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
                    "pass env=child_env(...) (maxim.utils.child_env; scripts: from _child_env import child_env; idempotent, so "
                    "child_env(env) also works for an env passed in), "
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
