#!/usr/bin/env python3
"""ONE reader of the operational mode (#963): no raw mode read in the agent loop.

**The contract it guards.** What the model is shown (the prompt roster, the follow-up type and mode, the Default
Network) and what dispatch enforces read ONE value, ``loop_state.operational_mode(executor, state)``, which
delegates to ``Executor.effective_operational_mode()`` (the operator's grant, else the loop's mode source,
normalised; owner decisions on #963, 2026-10-09). The loop state's raw RUN mode, ``loop_state.run_mode(state)``,
serves the two lifecycle reads only (the ``shutdown`` sentinel and the ``maxim_runtime["mode"]`` copy). Before
#963 the follow-up type read ``state.data.get("mode", "live")`` and ignored a passive grant.

Whole-tree, not diff-scoped: no git, no env, no suppression comments. Every run prints the scan set and the
allowlist with each entry's measured count. Exit 1 on a finding, 2 on a broken scope.

**Scope (rules A and E).** The seeds ``runtime/agent_loop.py``, ``runtime/tool_dispatch.py`` and
``runtime/loop_*.py``, plus ``runtime/executor.py`` and the executor wrapper modules (``WRAPPERS``), plus the
transitive closure of their ``maxim.runtime.*`` imports: ``import``/``from`` (absolute, relative, ``from
maxim.runtime import <submodule> [as x]``), lazy imports inside functions, and ``import_module("maxim.runtime.x")``
with a constant string. A later slice's new helper module is in scope as soon as the loop imports it. A missing
required seed, or a closure smaller than ``MIN_SCOPE``, exits 2: a scope that silently shrank is not a pass.

**Rule A (in scope).** Each of these is a finding:

1. an ``ast.Constant == "mode"``, except a dict-display key or a docstring (writes count, not only reads);
2. ``**X``, ``dict(X)``, ``X.copy()``, ``X.items()`` or ``X.values()`` where ``X`` is a ``*.data`` receiver;
3. ``.update(mode=...)`` or ``.update({"mode": ...})`` on any receiver, and ``X |= ...`` on a ``*.data`` receiver;
4. a non-constant key (subscript, ``.get``, ``.pop``, ``.setdefault``) on a ``*.data`` receiver, on a one-step
   alias of one in the same function (``d = <expr ending .data | getattr(..., "data") | ...snapshot()>``), or on a
   subscript of either (``snap["data"][k]``);
5. ``vars(...)``, ``.__dict__``, or ``getattr(<x>, "data")``.

**Rule B (repo-wide, ``src/maxim``).** A load of ``operational_override``, ``_operational_override`` or
``_mode_source`` (attribute, or a ``getattr``/``hasattr`` string) outside ``Executor.__init__``,
``Executor.set_mode_source``, ``Executor.set_operational_override`` and ``Executor.effective_operational_mode``;
and any ``def effective_operational_mode`` / ``def operational_override`` outside ``Executor``; and a ``getattr``/
``hasattr`` string naming a rule-E function sink (``getattr(ctrl, "configure_dn_for_mode")(...)``). No allowlist.

**Rule C (repo-wide).** Any reference to ``run_mode`` (a Name, an attribute such as ``loop_state.run_mode``, a
``getattr`` string, an ``import ... as`` alias) outside ``loop_state.py``'s ``run_mode``, ``shutdown_requested``,
``seed_runtime_run_mode`` and ``operational_mode``, and the loop's mode source: a ``lambda`` passed to
``set_mode_source`` inside ``loop_setup.py::_prepare_executor`` (and ``loop_setup.py``'s plain import of it).
No allowlist.

**Rule D (repo-wide, outside the rule-A scope).** A ``"mode"``-keyed read (subscript, ``.get``/``.pop``/
``.setdefault``, ``in``) on a ``*.data``, ``maxim_runtime`` or ``*snapshot*`` receiver. Pinned at today's three
#1193 sites through the allowlist (the rule-A scope already flags every ``"mode"`` constant).

**Rule E (sinks, in scope).** The mode argument of ``get_mode(·)``, ``ModeInfo(name=·)``,
``get_tool_followup_type(_, ·)``, ``configure_dn_for_mode(·)``, the Default Network controller's
``configure_for_mode(·)`` / ``inhibit_for_tool(·)``, ``ActionFollowup(mode=·)``, ``StructuredContext(mode=·)``,
``log_action(mode=·)`` and any ``get_available_tools(mode=·)`` must be one of: an ``operational_mode(<executor>,
...)`` call whose executor is the name ``executor`` or an attribute chain ending in ``.executor`` (never ``None``,
a computed expression or another object, which would drop or forge the grant); ``self.`` or that executor's
``.effective_operational_mode()``; a local name whose every assignment in its function is one of those; the
never-rebound MODE-SLOT parameter of a function that is itself a sink (its
callers are checked at their call sites, so "my callers are checked" is verified, not asserted:
``LoopController.configure_dn_for_mode`` -> ``configure_for_mode`` -> ``get_mode``); or the literal ``"passive"``
(the fail-closed fallback). A missing mode argument is a finding too. This also catches a hard-coded capability
mode. A function sink may not be aliased: ``from ... import get_mode as g`` and any non-call reference to one
(``f = ctrl.configure_dn_for_mode``, passing ``get_mode`` as a callback) are findings.

**Allowlist (rules A, D and E).** ``ALLOWLIST`` below: ``{file, qualname, count, axis, ref, reason}``, with
``axis`` one of ``lifecycle`` (the run-mode axis's two reads and its one raw reader), ``unrelated`` (a ``"mode"``
that is not the loop's mode), ``r-site`` (a mode-keyed reader outside the loop, #1193) or ``grant-validation``
(``Executor.set_operational_override`` resolving the grant it is about to store, so an unknown name is refused at
launch; it decides nothing). The count is every
rule-A/D/E finding in that function, compared by STRICT EQUALITY: a function that gains or loses one fails, and
an entry naming a function with no finding (an orphan) fails. Entries are SELF-APPROVED: they make a site
visible and reviewable, they do not prove anyone approved it.

**Residual blind spots, stated:** arbitrary string building on an alias (``k = "mo" + "de"; d[k]`` is caught on a
``*.data`` receiver or its one-step alias, not through a longer alias chain or a helper's parameter); an imported
constant used through an alias; a mode-keyed reader outside ``maxim.runtime`` handed the state by the loop (rule
D sees the ``*.data``/``maxim_runtime``/``*snapshot*`` receivers only); and masking within one function (an
allowlisted function's count is a count, so swapping one finding for another passes, as in the swallow lint).
The sink list is CLOSED: a capability lookup it does not name is not checked -- ``OPERATIONAL_MODES[...]``,
``executes_code``, ``raises_capability`` and ``_LEGACY_NAME_MAP`` are such lookups (none is in scope today); the
class sinks (``ModeInfo``, ``ActionFollowup``, ``StructuredContext``) may be aliased. And CACHING: a value read
through the accessor once and kept in a function-local hoisted above the tick passes every rule; the Default
Network and the follow-up are pinned against it by grant-flip-between-ticks tests (in ``test_loop_gates_characterization.py`` and
``test_one_mode_accessor_963.py``), the roster only by its source pin
(``test_operational_mode_launch.py::test_what_the_model_is_shown_follows_the_grant``) until §6 is extracted.

Regression guard: tests/unit/test_lint_loop_mode_reads.py drives ``main()`` on a fixture tree with one failing
fixture per evasion shape, rules B-E and the allowlist checks, the seed-missing exit 2, the deletion probe (the
pre-#963 ``state.data.get("mode", "live")`` restored in ``tool_dispatch``) and this checkout clean.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG = "src/maxim"

SEEDS = ("runtime/agent_loop.py", "runtime/tool_dispatch.py", "runtime/loop_state.py", "runtime/loop_setup.py")
SEED_GLOB = "runtime/loop_*.py"
EXECUTOR = "runtime/executor.py"
# The executor wrappers: each forwards ``effective_operational_mode`` through ``__getattr__`` (none defines its own).
WRAPPERS = ("runtime/fear_gate.py", "runtime/pain_interceptor.py", "simulation/instrumented_executor.py")
# The scope (seeds + executor + wrappers + closure) measured 2026-10-09 was 34 files; a resolver break that
# collapses it must not pass as clean. Lower this only with the module removal that justifies it.
MIN_SCOPE = 34

AXES = ("lifecycle", "unrelated", "r-site", "grant-validation")
_REF_RE = re.compile(r"#\d+")

ALLOWLIST: list[dict[str, object]] = [
    {
        "file": "runtime/loop_state.py",
        "qualname": "run_mode",
        "count": 1,
        "axis": "lifecycle",
        "ref": "#963",
        "reason": "THE raw read of the loop state's run mode; rule C confines its callers to the lifecycle helpers",
    },
    {
        "file": "runtime/loop_state.py",
        "qualname": "seed_runtime_run_mode",
        "count": 2,
        "axis": "lifecycle",
        "ref": "#963",
        "reason": "writes the run mode into maxim_runtime['mode'] (MemoryAgent/ExecAgent's run-mode axis, Q5)",
    },
    {
        "file": "runtime/bootstrap.py",
        "qualname": "build_tool_registry.<locals>.get_mode",
        "count": 2,
        "axis": "r-site",
        "ref": "#1193",
        "reason": "ModeSwitchTool's current-mode callback reads the Maxim object's mode, not the loop state",
    },
    {
        "file": "runtime/bootstrap.py",
        "qualname": "build_tool_registry",
        "count": 1,
        "axis": "r-site",
        "ref": "#1193",
        "reason": "hands that get_mode callback to ModeSwitchTool (a non-call reference; not definitions.get_mode)",
    },
    {
        "file": "runtime/skill_matcher.py",
        "qualname": "SkillMatcher._load_skill",
        "count": 1,
        "axis": "unrelated",
        "ref": "#963",
        "reason": "a skill file's front-matter 'mode' key (SkillPrompt.requires_mode), not the loop's mode",
    },
    {
        "file": "runtime/executor.py",
        "qualname": "Executor.set_operational_override",
        "count": 1,
        "axis": "grant-validation",
        "ref": "#963",
        "reason": "get_mode(mode) validates the grant being STORED (unknown names refused at launch); decides nothing",
    },
    {
        "file": "planning/adaptive_planner.py",
        "qualname": "_extract_context",
        "count": 1,
        "axis": "r-site",
        "ref": "#1193",
        "reason": "keys NAc/EC context on state.data['mode']; run or operational mode is #1193's decision",
    },
    {
        "file": "agents/memory_agent.py",
        "qualname": "MemoryAgent._on_percept",
        "count": 1,
        "axis": "r-site",
        "ref": "#1193",
        "reason": "reads the maxim_runtime run-mode copy (ExecAgent's run-mode branches, #1193 item 2)",
    },
    {
        "file": "memory/hippocampus.py",
        "qualname": "Hippocampus.capture_from_loop",
        "count": 1,
        "axis": "r-site",
        "ref": "#1193",
        "reason": "reads a state snapshot's top-level 'mode' (always absent: #1193 item 1)",
    },
]

# Rule B: the Executor's own members, the only places the grant and the mode source are read.
_B_NAMES = frozenset({"operational_override", "_operational_override", "_mode_source"})
_B_DEFS = frozenset({"effective_operational_mode", "operational_override"})
_B_SITES = frozenset(
    {
        "Executor.__init__",
        "Executor.set_mode_source",
        "Executor.set_operational_override",
        "Executor.effective_operational_mode",
    }
)
# Rule C: loop_state's accessor functions (the loop_setup lambda is checked structurally).
_C_SITES = frozenset({"run_mode", "shutdown_requested", "seed_runtime_run_mode", "operational_mode"})

_E_ACCEPTED_CALLS = frozenset({"operational_mode", "effective_operational_mode"})
_E_FALLBACK = "passive"


@dataclass(frozen=True)
class Finding:
    rule: str
    file: str
    qualname: str
    line: int
    what: str

    def __str__(self) -> str:
        return f"{PKG}/{self.file}:{self.line} [{self.rule}] {self.qualname}: {self.what}"


class ScopeError(RuntimeError):
    """The scan set cannot be established -- exit 2, never a pass."""


# ── parsing helpers ─────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Module:
    rel: str  # relative to src/maxim
    tree: ast.Module
    parents: dict[ast.AST, ast.AST]
    quals: dict[ast.AST, str]  # every node -> its enclosing function's qualname ("<module>" at top level)
    funcs: dict[ast.AST, ast.AST | None]  # every node -> its innermost enclosing def/lambda (None at module level)


def _parse(root: Path, rel: str) -> Module:
    path = root / PKG / rel
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as e:
        raise ScopeError(f"{PKG}/{rel}: cannot parse ({e})") from e
    parents: dict[ast.AST, ast.AST] = {}
    quals: dict[ast.AST, str] = {}
    funcs: dict[ast.AST, ast.AST | None] = {}

    def visit(node: ast.AST, parent: ast.AST, qual: str, func: ast.AST | None, in_def: bool) -> None:
        parents[node] = parent
        quals[node] = qual or "<module>"
        funcs[node] = func
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = f"{qual}.<locals>.{node.name}" if in_def else _join(qual, node.name)
            outer = [*node.decorator_list, *node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
            if node.returns is not None:
                outer.append(node.returns)
            for child in ast.iter_child_nodes(node):
                if any(child is o for o in outer):
                    visit(child, node, qual, func, in_def)
                else:
                    visit(child, node, inner, node, True)
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if child in node.body:
                    visit(child, node, _join(qual, node.name), func, False)
                else:
                    visit(child, node, qual, func, in_def)
        elif isinstance(node, ast.Lambda):
            for child in ast.iter_child_nodes(node):
                visit(child, node, qual, node, in_def)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, node, qual, func, in_def)

    for top in ast.iter_child_nodes(tree):
        visit(top, tree, "", None, False)
    return Module(rel, tree, parents, quals, funcs)


def _join(qual: str, name: str) -> str:
    return f"{qual}.{name}" if qual else name


def _qual(m: Module, node: ast.AST) -> str:
    return m.quals.get(node, "<module>")


def _is_docstring(m: Module, node: ast.Constant) -> bool:
    expr = m.parents.get(node)
    if not isinstance(expr, ast.Expr):
        return False
    owner = m.parents.get(expr)
    body = getattr(owner, "body", None)
    return (
        isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and isinstance(body, list)
        and bool(body)
        and body[0] is expr
    )


def _is_const(node: ast.AST | None, value: object) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _call_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _is_data_receiver(node: ast.AST) -> bool:
    """``<x>.data`` (or ``getattr(<x>, "data")``)."""
    if isinstance(node, ast.Attribute) and node.attr == "data":
        return True
    return (
        isinstance(node, ast.Call)
        and _call_name(node) == "getattr"
        and len(node.args) >= 2
        and (_is_const(node.args[1], "data"))
    )


def _is_alias_source(node: ast.AST) -> bool:
    """The right-hand side of a one-step alias: an expression ending ``.data``, ``getattr(..., "data")`` or a call
    to ``...snapshot()``."""
    if _is_data_receiver(node):
        return True
    return isinstance(node, ast.Call) and (_call_name(node) or "").endswith("snapshot")


def _aliases(func: ast.AST | None, tree: ast.Module) -> set[str]:
    """Names bound in ``func`` (or at module level) by ``name = <alias source>``."""
    scope = func if func is not None else tree
    out: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and _is_alias_source(node.value):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            if _is_alias_source(node.value) and isinstance(node.target, ast.Name):
                out.add(node.target.id)
    return out


# ── the scope ───────────────────────────────────────────────────────────────────────────────────────────


def _module_file(pkg_root: Path, dotted: str) -> str | None:
    """``maxim.runtime.x`` -> ``runtime/x.py`` (or the package ``__init__.py``), relative to src/maxim."""
    parts = dotted.split(".")
    if parts[:2] != ["maxim", "runtime"]:
        return None
    base = pkg_root.joinpath(*parts[1:])
    if (base / "__init__.py").is_file():
        return (base / "__init__.py").relative_to(pkg_root).as_posix()
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py").relative_to(pkg_root).as_posix()
    return None


def _package_of(rel: str) -> list[str]:
    parts = ["maxim", *Path(rel).with_suffix("").parts]
    return parts[:-1]  # a module's package; for an __init__.py, its own package (the "__init__" part drops)


def runtime_imports(m: Module) -> set[str]:
    """Every ``maxim.runtime...`` dotted name the module imports, anywhere in it (lazy imports included)."""
    out: set[str] = set()
    for node in ast.walk(m.tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = _package_of(m.rel)
                pkg = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                base = ".".join(pkg + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Call) and _call_name(node) in ("import_module", "__import__"):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                out.add(node.args[0].value)
    return {name for name in out if name == "maxim.runtime" or name.startswith("maxim.runtime.")}


def scan_set(root: Path) -> dict[str, Module]:
    pkg_root = root / PKG
    required = [*SEEDS, EXECUTOR, *WRAPPERS]
    missing = [r for r in required if not (pkg_root / r).is_file()]
    if missing:
        raise ScopeError(f"required seed(s) missing: {', '.join(missing)} -- the scope cannot be established")
    seeds = sorted({*required, *(p.relative_to(pkg_root).as_posix() for p in pkg_root.glob(SEED_GLOB))})
    mods: dict[str, Module] = {}
    todo = list(seeds)
    while todo:
        rel = todo.pop()
        if rel in mods:
            continue
        mods[rel] = _parse(root, rel)
        for dotted in runtime_imports(mods[rel]):
            target = _module_file(pkg_root, dotted)
            if target is not None and target not in mods:
                todo.append(target)
    if len(mods) < MIN_SCOPE:
        raise ScopeError(f"the scope resolved to {len(mods)} files, below MIN_SCOPE={MIN_SCOPE}")
    return dict(sorted(mods.items()))


# ── rules ───────────────────────────────────────────────────────────────────────────────────────────────


def rule_a(m: Module) -> list[Finding]:
    out: list[Finding] = []

    def hit(node: ast.AST, what: str) -> None:
        out.append(Finding("A", m.rel, _qual(m, node), getattr(node, "lineno", 0), what))

    alias_cache: dict[ast.AST | None, set[str]] = {}

    def is_data_or_alias(node: ast.AST) -> bool:
        if _is_data_receiver(node):
            return True
        if isinstance(node, ast.Name):
            func = m.funcs.get(node)
            if func not in alias_cache:
                alias_cache[func] = _aliases(func, m.tree)
            return node.id in alias_cache[func]
        if isinstance(node, ast.Subscript):  # ``d["data"][k]``, ``state.data["x"][k]``: one level into it
            return is_data_or_alias(node.value)
        return False

    for node in ast.walk(m.tree):
        if isinstance(node, ast.Constant) and node.value == "mode":
            parent = m.parents.get(node)
            if isinstance(parent, ast.Dict) and any(k is node for k in parent.keys):
                continue
            if _is_docstring(m, node):
                continue
            hit(node, 'the constant "mode"')
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is None and _is_data_receiver(v):
                    hit(node, "**<x>.data")
        elif isinstance(node, ast.keyword):
            if node.arg is None and _is_data_receiver(node.value):
                hit(node.value, "**<x>.data")
        elif isinstance(node, ast.Attribute) and node.attr == "__dict__":
            hit(node, ".__dict__")
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.BitOr) and is_data_or_alias(node.target):
            hit(node, "|= on <x>.data")
        elif isinstance(node, ast.Subscript) and is_data_or_alias(node.value):
            if not isinstance(node.slice, ast.Constant):
                hit(node, "a non-constant key on <x>.data")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            func = node.func
            if name == "vars" and isinstance(func, ast.Name):
                hit(node, "vars(...)")
            elif name == "getattr" and isinstance(func, ast.Name) and len(node.args) >= 2:
                if _is_const(node.args[1], "data"):
                    hit(node, 'getattr(<x>, "data")')
            elif name == "dict" and isinstance(func, ast.Name) and node.args and is_data_or_alias(node.args[0]):
                hit(node, "dict(<x>.data)")
            elif isinstance(func, ast.Attribute):
                if name in ("copy", "items", "values") and is_data_or_alias(func.value):
                    hit(node, f"<x>.data.{name}()")
                elif name == "update":
                    mode_kw = any(k.arg == "mode" for k in node.keywords)
                    mode_dict = any(
                        isinstance(a, ast.Dict) and any(_is_const(k, "mode") for k in a.keys) for a in node.args
                    )
                    if mode_kw or mode_dict:
                        hit(node, '.update() writing "mode"')
                elif name in ("get", "pop", "setdefault") and is_data_or_alias(func.value):
                    if node.args and not isinstance(node.args[0], ast.Constant):
                        hit(node, f"a non-constant .{name}() key on <x>.data")
    return out


def _b_site(m: Module, node: ast.AST) -> bool:
    return m.rel == EXECUTOR and _qual(m, node) in _B_SITES


def rule_b(m: Module) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(m.tree):
        if isinstance(node, ast.Attribute) and node.attr in _B_NAMES and isinstance(node.ctx, ast.Load):
            if not _b_site(m, node):
                out.append(Finding("B", m.rel, _qual(m, node), node.lineno, f"reads .{node.attr}"))
        elif isinstance(node, ast.Call) and _call_name(node) in ("getattr", "hasattr") and len(node.args) >= 2:
            arg = node.args[1]
            if isinstance(arg, ast.Constant) and arg.value in _B_NAMES and not _b_site(m, node):
                out.append(Finding("B", m.rel, _qual(m, node), node.lineno, f"{_call_name(node)}(.., {arg.value!r})"))
            elif isinstance(arg, ast.Constant) and arg.value in _E_SINK_FUNCS:
                # A rule-E sink reached by a string evades its call check (#963 delta review S2).
                out.append(Finding("B", m.rel, _qual(m, node), node.lineno, f"{_call_name(node)}(.., {arg.value!r})"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _B_DEFS:
            owner = m.parents.get(node)
            if not (m.rel == EXECUTOR and isinstance(owner, ast.ClassDef) and owner.name == "Executor"):
                out.append(Finding("B", m.rel, _qual(m, node), node.lineno, f"defines {node.name} outside Executor"))
            elif node.name == "operational_override":
                out.append(Finding("B", m.rel, _qual(m, node), node.lineno, "Executor.operational_override is gone"))
    return out


def _c_allowed(m: Module, node: ast.AST) -> bool:
    if m.rel == "runtime/loop_state.py":
        return _qual(m, node) in _C_SITES
    if m.rel != "runtime/loop_setup.py":
        return False
    if isinstance(node, ast.alias):
        return node.asname is None
    lam = m.funcs.get(node)
    if not isinstance(lam, ast.Lambda) or _qual(m, node) != "_prepare_executor":
        return False
    call = m.parents.get(lam)
    return isinstance(call, ast.Call) and _call_name(call) == "set_mode_source" and call.args == [lam]


def rule_c(m: Module) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(m.tree):
        what = None
        if isinstance(node, ast.Name) and node.id == "run_mode":
            what = "references run_mode"
        elif isinstance(node, ast.Attribute) and node.attr == "run_mode":
            what = "references .run_mode"
        elif isinstance(node, ast.Call) and _call_name(node) in ("getattr", "hasattr") and len(node.args) >= 2:
            if _is_const(node.args[1], "run_mode"):
                what = 'getattr(.., "run_mode")'
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "run_mode":
                    m.quals[a] = _qual(m, node)
                    if not _c_allowed(m, a):
                        out.append(Finding("C", m.rel, _qual(m, node), node.lineno, "imports run_mode"))
            continue
        if what is not None and not _c_allowed(m, node):
            out.append(Finding("C", m.rel, _qual(m, node), getattr(node, "lineno", 0), what))
    return out


def _is_d_receiver(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr in ("data", "maxim_runtime") or "snapshot" in node.attr
    if isinstance(node, ast.Name):
        return node.id == "maxim_runtime" or "snapshot" in node.id
    if isinstance(node, ast.Call):
        return "snapshot" in (_call_name(node) or "")
    if isinstance(node, ast.Subscript):
        return _is_d_receiver(node.value)
    return False


def rule_d(m: Module) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(m.tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            hit = _is_const(node.slice, "mode") and _is_d_receiver(node.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            hit = (
                node.func.attr in ("get", "pop", "setdefault")
                and bool(node.args)
                and _is_const(node.args[0], "mode")
                and _is_d_receiver(node.func.value)
            )
        elif isinstance(node, ast.Compare):
            hit = (
                _is_const(node.left, "mode")
                and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
                and any(_is_d_receiver(c) for c in node.comparators)
            )
        else:
            continue
        if hit:
            out.append(Finding("D", m.rel, _qual(m, node), node.lineno, '"mode"-keyed read on a state receiver'))
    return out


# sink -> (positional index or None, keyword name)
_E_SINKS: dict[str, tuple[int | None, str | None]] = {
    "get_mode": (0, None),
    "ModeInfo": (None, "name"),
    "get_tool_followup_type": (1, "mode_name"),
    "configure_dn_for_mode": (0, "mode_name"),
    "ActionFollowup": (None, "mode"),
    "StructuredContext": (None, "mode"),
    "log_action": (None, "mode"),
    # The Default Network controller's own lookups (``LoopController.configure_dn_for_mode`` forwards to them).
    "configure_for_mode": (0, "mode_name"),
    "inhibit_for_tool": (0, "mode_name"),
}
# The sinks that are FUNCTIONS (the classes, ``ModeInfo`` / ``ActionFollowup`` / ``StructuredContext``, are named in
# annotations and ``isinstance`` too, so only their calls are checked). A function sink may not be aliased.
_E_SINK_FUNCS = frozenset(set(_E_SINKS) - {"ModeInfo", "ActionFollowup", "StructuredContext"} | {"get_available_tools"})


def _executor_ref(node: ast.AST) -> bool:
    """The run's executor, structurally: the name ``executor`` or an attribute chain on a name that ends in
    ``.executor`` (``self.executor``). Never a constant (``None`` drops the grant), a computed expression or another
    object (``ctrl.dn_ctrl``)."""
    if isinstance(node, ast.Name):
        return node.id == "executor"
    if not (isinstance(node, ast.Attribute) and node.attr == "executor"):
        return False
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _accepted_call(node: ast.AST) -> bool:
    """``operational_mode(<executor>, ...)``, or ``<executor>.effective_operational_mode()`` (inside the executor)."""
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node)
    if name == "operational_mode":
        return bool(node.args) and _executor_ref(node.args[0])
    if name == "effective_operational_mode":
        # The executor's own read (``self.``) or the run's executor. Arguments are not checked: the method takes
        # none, so a call with any raises ``TypeError`` at runtime.
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        return receiver is not None and (
            (isinstance(receiver, ast.Name) and receiver.id == "self") or _executor_ref(receiver)
        )
    return False


def _rebound(func: ast.AST, name: str) -> bool:
    """Whether ``name`` is bound anywhere in ``func``'s own scope (an assignment, a loop or ``with`` target, ...)."""
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load):
            return True
        if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return True
        if isinstance(node, ast.ExceptHandler) and node.name == name:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (a.asname or a.name.split(".")[0]) == name for a in node.names
        ):
            return True
        stack.extend(ast.iter_child_nodes(node))
    return False


def _mode_slot(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The parameter(s) in a sink's own MODE slot: its ``_E_SINKS`` keyword, and the parameter at its positional
    mode index (``self``/``cls`` skipped). Every caller of the sink is checked on exactly that argument."""
    if func.name == "get_available_tools":
        return {"mode", "mode_name"}
    pos, kw = _E_SINKS[func.name]
    out = {kw} if kw else set()
    positional = [p.arg for p in (*func.args.posonlyargs, *func.args.args)]
    if positional and positional[0] in ("self", "cls"):
        positional = positional[1:]
    if pos is not None and len(positional) > pos:
        out.add(positional[pos])
    return out


def _forwarded_param(m: Module, value: ast.Name) -> bool:
    """The never-rebound MODE-slot parameter of a function that is ITSELF a sink: its callers are checked on that
    argument at their call sites (``LoopController.configure_dn_for_mode`` -> ``configure_for_mode`` ->
    ``get_mode``), so this is verified, not asserted. Any other parameter (``fallback="singularity"``) is not."""
    func = m.funcs.get(value)
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or func.name not in _E_SINK_FUNCS:
        return False
    return value.id in _mode_slot(func) and not _rebound(func, value.id)


def _assignments(func: ast.AST | None, tree: ast.Module, name: str) -> list[ast.AST] | None:
    """The values bound to ``name`` in ``func``'s own scope; None when it is bound some other way (a parameter, a
    loop target, ``with ... as``, an import, ``global``/``nonlocal``), which never counts as accepted."""
    scope: ast.AST = func if func is not None else tree
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = scope.args
        params = [
            *a.posonlyargs,
            *a.args,
            *a.kwonlyargs,
            *([a.vararg] if a.vararg else []),
            *([a.kwarg] if a.kwarg else []),
        ]
        if any(p.arg == name for p in params):
            return None
    values: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue  # a nested scope's bindings are its own
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    values.append(node.value)
                elif any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(t)):
                    return None  # tuple unpacking
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                if node.value is None:
                    if isinstance(node, ast.AnnAssign):
                        stack.extend(ast.iter_child_nodes(node))
                        continue
                    return None
                values.append(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return None
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node.target)):
                return None
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node.optional_vars)):
                return None
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((a.asname or a.name.split(".")[0]) == name for a in node.names):
                return None
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return None
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            return None
        stack.extend(ast.iter_child_nodes(node))
    return values or None


def _e_value_ok(m: Module, value: ast.AST) -> bool:
    if _is_const(value, _E_FALLBACK) or _accepted_call(value):
        return True
    if isinstance(value, ast.Name):
        if _forwarded_param(m, value):
            return True
        bound = _assignments(m.funcs.get(value), m.tree, value.id)
        return bound is not None and all(_accepted_call(v) for v in bound)
    return False


def rule_e(m: Module) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(m.tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in _E_SINK_FUNCS and a.asname not in (None, a.name):
                    out.append(
                        Finding("E", m.rel, _qual(m, node), node.lineno, f"imports the sink {a.name} as {a.asname}")
                    )
            continue
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            ident = node.id if isinstance(node, ast.Name) else node.attr
            parent = m.parents.get(node)
            if ident in _E_SINK_FUNCS and not (isinstance(parent, ast.Call) and parent.func is node):
                out.append(
                    Finding("E", m.rel, _qual(m, node), node.lineno, f"a non-call reference to the sink {ident}")
                )
            continue
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "get_available_tools":
            value = next((k.value for k in node.keywords if k.arg in ("mode", "mode_name")), None)
            if value is not None and not _e_value_ok(m, value):
                out.append(
                    Finding("E", m.rel, _qual(m, node), node.lineno, "get_available_tools(mode=<not the accessor>)")
                )
            continue
        if name not in _E_SINKS:
            continue
        if name == "get_mode" and not node.args and not node.keywords:
            continue  # a zero-argument callback named get_mode (bootstrap's), not the definitions lookup
        pos, kw = _E_SINKS[name]
        value = next((k.value for k in node.keywords if kw is not None and k.arg == kw), None)
        if value is None and pos is not None and len(node.args) > pos:
            value = node.args[pos]
        if value is None:
            out.append(Finding("E", m.rel, _qual(m, node), node.lineno, f"{name}(...) without a mode argument"))
        elif not _e_value_ok(m, value):
            out.append(Finding("E", m.rel, _qual(m, node), node.lineno, f"{name}(...)'s mode is not the accessor"))
    return out


# ── the allowlist ───────────────────────────────────────────────────────────────────────────────────────


def allowlist_problems(allowlist: list[dict[str, object]]) -> list[str]:
    problems: list[str] = []
    keys = {"file", "qualname", "count", "axis", "ref", "reason"}
    seen: set[tuple[object, object]] = set()
    for i, e in enumerate(allowlist):
        if set(e) != keys:
            problems.append(f"allowlist[{i}]: keys must be exactly {sorted(keys)}")
            continue
        if e["axis"] not in AXES:
            problems.append(f"allowlist[{i}]: axis {e['axis']!r} is not one of {AXES}")
        if not (isinstance(e["ref"], str) and _REF_RE.fullmatch(e["ref"])):
            problems.append(f"allowlist[{i}]: ref {e['ref']!r} is not #NNN")
        if not (isinstance(e["reason"], str) and e["reason"].strip()):
            problems.append(f"allowlist[{i}]: reason is empty")
        if not (isinstance(e["count"], int) and not isinstance(e["count"], bool) and e["count"] > 0):
            problems.append(f"allowlist[{i}]: count must be a positive int")
        key = (e["file"], e["qualname"])
        if key in seen:
            problems.append(f"allowlist[{i}]: duplicate entry for {e['file']}::{e['qualname']}")
        seen.add(key)
    return problems


def check(root: Path, allowlist: list[dict[str, object]] | None = None) -> tuple[list[str], list[str]]:
    """(failures, report lines). Raises ScopeError for exit 2."""
    allowlist = ALLOWLIST if allowlist is None else allowlist
    report: list[str] = []
    failures = allowlist_problems(allowlist)
    scope = scan_set(root)
    report.append(f"scan set (rules A, E): {len(scope)} files (MIN_SCOPE {MIN_SCOPE})")
    report.extend(f"  {PKG}/{rel}" for rel in scope)

    allowable: list[Finding] = []
    for m in scope.values():
        allowable += rule_a(m) + rule_e(m)
    hard: list[Finding] = []
    for path in sorted((root / PKG).rglob("*.py")):
        rel = path.relative_to(root / PKG).as_posix()
        m = scope.get(rel) or _parse(root, rel)
        hard += rule_b(m) + rule_c(m)
        if rel not in scope:
            allowable += rule_d(m)

    measured = Counter((f.file, f.qualname) for f in allowable)
    listed = {(str(e["file"]), str(e["qualname"])): e for e in allowlist}
    report.append(f"allowlist: {len(allowlist)} entries (self-approved visibility, not approval)")
    for (file, qual), e in listed.items():
        n = measured.get((file, qual), 0)
        report.append(f"  {PKG}/{file}::{qual} = {n} (pin {e['count']}, {e['axis']}, {e['ref']}): {e['reason']}")
        if n == 0:
            failures.append(f"{PKG}/{file}::{qual}: allowlisted but has no finding (orphan: remove the entry)")
        elif n != e["count"]:
            failures.append(f"{PKG}/{file}::{qual}: {n} finding(s), allowlist pins {e['count']} (strict equality)")
    for f in allowable:
        if (f.file, f.qualname) not in listed:
            failures.append(str(f))
    failures.extend(str(f) for f in hard)
    return failures, report


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        failures, report = check(REPO_ROOT)
    except ScopeError as e:
        print(f"lint_loop_mode_reads: ERROR: {e}", file=sys.stderr)
        return 2
    print("\n".join(report))
    if failures:
        print(
            f"\nlint_loop_mode_reads: {len(failures)} finding(s) -- read the mode through "
            "loop_state.operational_mode(executor, state) (see this script's docstring):",
            file=sys.stderr,
        )
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("lint_loop_mode_reads: OK -- one operational-mode reader")
    return 0


if __name__ == "__main__":
    sys.exit(main())
