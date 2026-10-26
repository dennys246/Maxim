"""Atonia gate (#1315) strict red gates for item 2 (``Tool.effect`` + the registry ceiling) and item 3 (the required
``ToolPolicy``), plus advisor_mode.md's O1, O2 and O5 rows.

WHAT THIS FILE IS. Tests only, written from ``docs/plans/atonia_gate_approach_notes.md`` (§1 V1-V10, §1A, §2, §3;
§1A wins where they disagree) before any ``src/`` change. Each defect is pinned as a strict red gate
(``xfail(strict=True)``), so its fix is visible the day it lands (the gate XPASSes, and strict turns that into a
failure until the decorator is removed), and none is vacuous: each fails today, with an ``AssertionError``, for the
reason its docstring names.

HOW A GATE IS SHAPED. Where an API does not exist yet, the gate's first step is an explicit assertion that it exists
(``item 2 not built: tools/base.py has no Effect``). The new symbols are imported lazily after that assertion, never
at module top, and the gate goes on to exercise the behaviour exactly as the notes pin it, so once the symbol lands
the gate flips only when the BEHAVIOUR is right. "Refuses with X" is asserted with :func:`_expect_raises`, which
raises ``AssertionError`` (``pytest.raises`` raises ``Failed``, which an ``xfail(raises=AssertionError)`` would
report as a hard failure, not as the expected one).

THE GATES, AND WHAT FLIPS EACH
  item 2  ``test_item2_every_tool_declares_own_effect`` -- every concrete Tool subclass (AST closure over
          ``src/maxim``) declares a non-zero ``effect`` in its own class body; the two factories too.
  item 2  ``test_item2_process_and_code_runners_declare_all`` -- a Tool whose methods reach a process/code sink
          (§2.1 WRITE_LOCAL row) declares ``Effect.ALL`` or is on ``tests/fixtures/effect_exemptions.json``.
  item 2  ``test_item2_effect_record_semantics`` -- ``Effect(0)``/non-Effect refused at class creation, inherited
          effect does not count, reassignment refuses admission, ``exceeds`` is the subset check.
  item 2  ``test_item2_registration_sweep_under_read_ceiling`` -- every built-in tool refused by a READ ceiling
          unless it is READ.
  item 2  ``test_item2_same_name_replacement_below_all_refuses`` -- V10's replacement rule.
  item 2  ``test_item2_sim_registry_unknown_name_raises_below_all``.
  item 3  ``test_item3_policy_is_required_and_permissions_removed`` -- V3 + §1A.5.
  item 3  ``test_item3_tool_policy_is_sealed``.
  item 3  ``test_item3_dispatch_checks_the_object_it_runs`` -- the find/get race probe.
  item 3  ``test_item3_composite_spawn_builds_narrowed_child`` and
          ``test_item3_composite_spawn_refuses_without_dispatch_policy`` -- the ``current_policy`` composite probe.
  item 3  ``test_item3_auto_sense_checks_both_direct_calls`` -- V6.
  O1      ``test_o1_maxim_run_executor_carries_v10_policy`` -- both ``maxim.run`` branches.
  O2/O5   ``test_o2_o5_ceiling_refuses_acting_tools_the_mode_gate_allows``.
Plus one passing regression guard, ``test_v10_registered_tools_dispatch_under_maxim_run``.

PRECONDITIONS ARE NOT THE GATE. A fixture tool that cannot be built, a registry builder that raises, a seam that was
never reached: those raise ``ScenarioBroken`` (not an ``AssertionError``), and every gate is
``xfail(strict=True, raises=AssertionError)``, so a broken scenario FAILS the run instead of hiding as the expected
failure.
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import os
import pickle
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PKG = SRC / "maxim"
EXEMPTIONS = REPO / "tests" / "fixtures" / "effect_exemptions.json"


class ScenarioBroken(Exception):
    """The scenario did not happen as designed: NOT the defect a gate pins (module docstring)."""


def require(cond: Any, msg: str) -> None:
    if not cond:
        raise ScenarioBroken(msg)


def _expect_raises(exc: type[BaseException] | tuple[type[BaseException], ...], fn: Callable[[], Any], msg: str) -> Any:
    """Call *fn*; it must raise *exc*. Anything else (no raise, or another type) is an ``AssertionError``."""
    try:
        fn()
    except exc as e:  # noqa: PERF203 -- the expected path
        return e
    except Exception as e:  # the wrong refusal is the defect too, reported as an assertion
        raise AssertionError(f"{msg} (raised {type(e).__name__}: {e} instead)") from e
    raise AssertionError(msg)


# ── lazy access to the not-yet-built API ─────────────────────────────────


def _require_effect() -> Any:
    """``maxim.tools.base`` with item 2's surface (§2.2, §1A.3), or an AssertionError naming what is missing."""
    import maxim.tools.base as base

    assert hasattr(base, "Effect"), "item 2 not built: tools/base.py has no Effect"
    for sym in ("effect_of", "exceeds"):
        assert hasattr(base, sym), f"item 2 not built: tools/base.py has no {sym}"
    return base


def _tool_effect_refused() -> type[Exception]:
    """``ToolEffectRefused``. The notes name it without a module; §1A.3 puts the effect machinery in
    ``tools/base.py`` and admission in ``tools/registry.py``, so either home is accepted (choice, stated)."""
    import maxim.tools.base as base
    import maxim.tools.registry as registry

    exc = getattr(registry, "ToolEffectRefused", None) or getattr(base, "ToolEffectRefused", None)
    assert exc is not None, "item 2 not built: no ToolEffectRefused in tools/registry.py or tools/base.py"
    return exc


def _require_policy() -> Any:
    """``maxim.runtime.tool_policy`` (§1A.3), or an AssertionError."""
    spec = importlib.util.find_spec("maxim.runtime.tool_policy")
    assert spec is not None, "item 3 not built: maxim/runtime/tool_policy.py does not exist"
    mod = importlib.import_module("maxim.runtime.tool_policy")
    assert hasattr(mod, "ToolPolicy"), "item 3 not built: runtime/tool_policy.py has no ToolPolicy"
    return mod


def _require_gate_kinds() -> Any:
    """``ToolErrorKind`` with §3's typed refusal members and ``GATE_REFUSAL_KINDS`` (§1A.2)."""
    import maxim.tools.base as base

    for member in ("NOT_REGISTERED", "POLICY_CEILING", "POLICY_PERMISSIONS", "POLICY_PIN", "MODE", "OFFLINE"):
        assert hasattr(base.ToolErrorKind, member), f"item 3 not built: ToolErrorKind has no {member}"
    assert hasattr(base, "GATE_REFUSAL_KINDS"), "item 3 not built: tools/base.py has no GATE_REFUSAL_KINDS"
    return base


# ── the AST closure of Tool subclasses over src/maxim ────────────────────

_SINK_PREFIXES = (
    "subprocess.",
    "multiprocessing.",
    "runpy.",
    "ctypes.",
    "asyncio.create_subprocess_",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.fork",
)
_SINK_EXACT = frozenset({"os.system", "os.popen", "pty.spawn"})
_NAME_ARG_SINKS = frozenset({"importlib.import_module"})  # a sink only with a non-literal name (§2.1 sink rules)
# §2.1: pinned literal-name helpers; the call sites are checked elsewhere, so reach does not descend into them.
_PINNED_HELPERS = frozenset(
    {
        ("maxim.utils.optional_deps", "require_optional_dependency"),
        ("maxim.utils.optional_deps", "optional_dependency_available"),
        ("maxim.runtime.lane_backends", "resolve_backend_class"),
    }
)
_TOOL_ROOT = ("maxim.tools.base", "Tool")


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _Module:
    def __init__(self, name: str, path: Path, tree: ast.Module) -> None:
        self.name = name
        self.path = path
        self.tree = tree
        self.is_pkg = path.name == "__init__.py"
        self.imports: dict[str, tuple[str, str | None]] = {}  # local name -> (module, attr-or-None)
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.classes: dict[str, list[ast.ClassDef]] = {}

    def resolve_relative(self, level: int, module: str | None) -> str:
        pkg_parts = self.name.split(".") if self.is_pkg else self.name.split(".")[:-1]
        if level > 1:
            pkg_parts = pkg_parts[: len(pkg_parts) - (level - 1)]
        base = ".".join(pkg_parts)
        return f"{base}.{module}" if module else base


@lru_cache(maxsize=1)
def _index() -> dict[str, _Module]:
    mods: dict[str, _Module] = {}
    for path in sorted(PKG.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as e:  # pragma: no cover - the suite would not import either
            raise ScenarioBroken(f"cannot parse {path}: {e}") from e
        mods[_module_name(path)] = _Module(_module_name(path), path, tree)
    for m in mods.values():
        for node in ast.walk(m.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        m.imports[a.asname] = (a.name, None)
                    else:
                        top = a.name.split(".")[0]
                        m.imports.setdefault(top, (top, None))
            elif isinstance(node, ast.ImportFrom):
                src = m.resolve_relative(node.level, node.module) if node.level else (node.module or "")
                for a in node.names:
                    local = a.asname or a.name
                    if f"{src}.{a.name}" in mods:
                        m.imports[local] = (f"{src}.{a.name}", None)
                    else:
                        m.imports[local] = (src, a.name)
            elif isinstance(node, ast.ClassDef):
                m.classes.setdefault(node.name, []).append(node)
        for node in m.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                m.functions[node.name] = node
    return mods


def _dotted(expr: ast.expr) -> list[str] | None:
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return parts[::-1]
    return None


def _resolve_class_ref(mod: _Module, expr: ast.expr) -> tuple[str, str] | None:
    """A base-class expression -> (module, class name), when it names a class in src/maxim."""
    mods = _index()
    parts = _dotted(expr)
    if not parts:
        return None
    if len(parts) == 1:
        name = parts[0]
        if name in mod.classes:
            return (mod.name, name)
        if name in mod.imports:
            src, attr = mod.imports[name]
            if attr is not None and src in mods:
                return (src, attr)
        return None
    head, *rest = parts
    if head in mod.imports and mod.imports[head][1] is None:
        dotted = ".".join([mod.imports[head][0], *rest[:-1]])
        if dotted in mods:
            return (dotted, rest[-1])
    return None


@lru_cache(maxsize=1)
def _tool_closure() -> dict[tuple[str, str], tuple[_Module, ast.ClassDef]]:
    """Every ClassDef in src/maxim (nested ones included) that descends from ``maxim.tools.base.Tool``."""
    mods = _index()
    closure: dict[tuple[str, str], tuple[_Module, ast.ClassDef]] = {}
    known = {_TOOL_ROOT}
    changed = True
    while changed:
        changed = False
        for m in mods.values():
            for cname, defs in m.classes.items():
                key = (m.name, cname)
                if key == _TOOL_ROOT or key in closure:
                    continue
                for cdef in defs:
                    if any(_resolve_class_ref(m, b) in known for b in cdef.bases):
                        closure[key] = (m, cdef)
                        known.add(key)
                        changed = True
                        break
    require(len(closure) >= 60, f"the Tool closure found only {len(closure)} classes; the 2+3 pass counted 83")
    return closure


def _bases_in_closure(key: tuple[str, str]) -> list[tuple[str, str]]:
    m, cdef = _tool_closure()[key]
    out = []
    for b in cdef.bases:
        ref = _resolve_class_ref(m, b)
        if ref is not None and ref in _tool_closure():
            out.append(ref)
    return out


def _defines(cdef: ast.ClassDef, name: str) -> bool:
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in cdef.body)


def _is_concrete(key: tuple[str, str], seen: frozenset[tuple[str, str]] = frozenset()) -> bool:
    """Concrete = ``execute`` is defined on the class or on an in-closure ancestor (``Tool.execute`` is abstract)."""
    if key in seen:
        return False
    _m, cdef = _tool_closure()[key]
    if _defines(cdef, "execute"):
        return True
    return any(_is_concrete(b, seen | {key}) for b in _bases_in_closure(key))


def _own_effect_value(cdef: ast.ClassDef) -> ast.expr | None:
    """The ``effect`` assigned in the class's OWN body (§2.2: own ``__dict__``), or None."""
    for n in cdef.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "effect" for t in n.targets):
            return n.value
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "effect":
            return n.value
    return None


def _is_zero_effect(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Call)
        and len(expr.args) == 1
        and isinstance(expr.args[0], ast.Constant)
        and expr.args[0].value == 0
    ) or (isinstance(expr, ast.Constant) and expr.value in (0, None))


def _is_effect_all(expr: ast.expr | None) -> bool:
    return isinstance(expr, ast.Attribute) and expr.attr == "ALL"


# ── reach: the call graph across src/maxim, function-level (§2.1) ───────


def _sink_name(full: str) -> bool:
    return full in _SINK_EXACT or any(full.startswith(p) for p in _SINK_PREFIXES)


def _literal_first_arg(call: ast.Call) -> bool:
    return bool(call.args) and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)


class _Reach:
    """Does a function reach a sink? Memoised; cycles count as not-reaching on the back edge."""

    def __init__(self) -> None:
        self._memo: dict[int, str | None] = {}
        self._active: set[int] = set()

    def function(self, mod: _Module, fn: ast.AST, cls: tuple[str, str] | None) -> str | None:
        key = id(fn)
        if key in self._memo:
            return self._memo[key]
        if key in self._active:
            return None
        self._active.add(key)
        found: str | None = None
        for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
            found = self._call(mod, call, cls)
            if found:
                break
        self._active.discard(key)
        self._memo[key] = found
        return found

    def _target(self, src: str, attr: str) -> str | None:
        mods = _index()
        if (src, attr) in _PINNED_HELPERS or src not in mods:
            return None
        tm = mods[src]
        if attr in tm.functions:
            return self.function(tm, tm.functions[attr], None)
        if attr in tm.classes:
            init = next((n for n in tm.classes[attr][0].body if getattr(n, "name", None) == "__init__"), None)
            return self.function(tm, init, (src, attr)) if init is not None else None
        return None

    def _call(self, mod: _Module, call: ast.Call, cls: tuple[str, str] | None) -> str | None:
        f = call.func
        if isinstance(f, ast.Name):
            n = f.id
            if n in ("exec", "eval") and n not in mod.imports:
                return n
            if n in ("compile", "__import__") and n not in mod.imports and not _literal_first_arg(call):
                return n
            if n in mod.imports:
                src, attr = mod.imports[n]
                if attr is None:
                    return None
                full = f"{src}.{attr}"
                if full in _NAME_ARG_SINKS:
                    return None if _literal_first_arg(call) else full
                if _sink_name(full):
                    return full
                return self._target(src, attr)
            if n in mod.functions:
                return self.function(mod, mod.functions[n], None)
            if n in mod.classes:
                return self._target(mod.name, n)
            return None
        if isinstance(f, ast.Attribute):
            parts = _dotted(f)
            if not parts:
                return None
            head, *rest = parts
            if head == "self" and len(rest) == 1 and cls is not None:
                return self._self_method(cls, rest[0])
            if head in mod.imports:
                src, attr = mod.imports[head]
                full = ".".join([src if attr is None else f"{src}.{attr}", *rest])
                if full in _NAME_ARG_SINKS:
                    return None if _literal_first_arg(call) else full
                if _sink_name(full):
                    return full
                owner, _, fname = full.rpartition(".")
                if owner in _index():
                    return self._target(owner, fname)
        return None

    def _self_method(self, cls: tuple[str, str], name: str) -> str | None:
        """``self.name()`` on the class, or on an in-closure ancestor in the SAME module (§2.1 blind spot)."""
        mods = _index()
        todo, seen = [cls], set()
        while todo:
            key = todo.pop(0)
            if key in seen or key[0] not in mods or key[0] != cls[0]:
                continue
            seen.add(key)
            m = mods[key[0]]
            for cdef in m.classes.get(key[1], []):
                for n in cdef.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                        return self.function(m, n, key)
                for b in cdef.bases:
                    ref = _resolve_class_ref(m, b)
                    if ref is not None:
                        todo.append(ref)
        return None

    def tool_class(self, key: tuple[str, str]) -> str | None:
        m, cdef = _tool_closure()[key]
        for n in cdef.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                hit = self.function(m, n, key)
                if hit:
                    return f"{n.name} -> {hit}"
        for base in _bases_in_closure(key):  # inherited methods count only inside the same module (§2.1)
            if base[0] == key[0] and base != _TOOL_ROOT:
                hit = self.tool_class(base)
                if hit:
                    return hit
        return None


# ── fixture tools (built lazily: Effect does not exist at import time today) ──


def _probe_tool(name: str, effect: Any, *, ran: list[str] | None = None, auto_fire: bool = False) -> Any:
    """An instance of a fresh Tool subclass declaring *effect* in its own body; records each run in *ran*."""
    from maxim.tools.base import Tool, ToolOutput

    log = ran if ran is not None else []

    try:
        cls = type(
            f"Probe_{name}",
            (Tool,),
            {
                "name": name,
                "description": f"probe {name}",
                "input_schema": {},
                "effect": effect,
                "auto_fire": auto_fire,
                "execute": lambda self, **kw: (log.append(name), ToolOutput(success=True, output=name))[1],
            },
        )
        return cls()
    except Exception as e:
        raise ScenarioBroken(f"fixture tool {name!r} could not be built: {type(e).__name__}: {e}") from e


# ══ item 2 ══════════════════════════════════════════════════════════════


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="#1315 item 2: every concrete Tool declares its own non-zero effect"
)
def test_item2_every_tool_declares_own_effect() -> None:
    """Every concrete Tool subclass reachable from src/maxim (AST closure, nested classes included) assigns a
    non-zero ``effect`` in its OWN class body (§2.2: an inherited value does not count), and the two factories
    (``create_stub_tool`` and ``@maxim.tool``) put it in the class body they build. AST, not
    ``Tool.__subclasses__()``, because the latter sees only modules some import already reached and factory
    classes only once called."""
    concrete = [k for k in _tool_closure() if _is_concrete(k)]
    require(any(k[1] == "BashTool" for k in concrete), "the Tool closure missed BashTool")
    base = _require_effect()

    missing = []
    for key in sorted(concrete):
        _m, cdef = _tool_closure()[key]
        val = _own_effect_value(cdef)
        if val is None or _is_zero_effect(val):
            missing.append(f"{key[0]}::{key[1]}")
    assert not missing, f"{len(missing)} concrete Tool classes declare no own non-zero effect: {missing[:10]}"

    # The factories' output, at runtime.
    import maxim.api as api
    import maxim.tools.reachy_stubs as stubs

    for name, obj in vars(stubs).items():
        if isinstance(obj, type) and issubclass(obj, base.Tool) and obj is not base.Tool:
            eff = obj.__dict__.get("effect")
            assert isinstance(eff, base.Effect) and eff, f"create_stub_tool output {name} has no own effect"

    def _atonia_probe_fn(x: str) -> str:
        return x

    api.tool(_atonia_probe_fn)
    try:
        made = [t for t in api._registered_tools if getattr(t, "name", None) == "_atonia_probe_fn"]
        require(made, "@maxim.tool recorded no registration")
        eff = type(made[-1]).__dict__.get("effect")
        assert isinstance(eff, base.Effect) and eff, "@maxim.tool's FunctionTool has no own effect"
    finally:
        api.unregister_tool("_atonia_probe_fn")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1315 item 2: process/code runners declare Effect.ALL or sit on tests/fixtures/effect_exemptions.json",
)
def test_item2_process_and_code_runners_declare_all() -> None:
    """§2.1 WRITE_LOCAL row: a Tool class whose methods reach a sink (subprocess.*, os.exec*/system/popen/spawn*/
    posix_spawn*/fork*, pty.spawn, asyncio.create_subprocess_*, multiprocessing.*, exec, eval, compile/
    importlib.import_module/__import__ with a non-literal name, runpy.*, ctypes.*), through the call graph across
    src/maxim modules (resolved imports, ``self.`` methods, module functions), declares ``Effect.ALL`` or is on the
    exemption list with a reason. The exemption file's shape is not pinned by the notes; chosen here:
    ``{"exemptions": [{"class": "<module>::<Class>", "reason": "..."}]}``."""
    reach = _Reach()
    reaching = {}
    for key in _tool_closure():
        hit = reach.tool_class(key)
        if hit:
            reaching[f"{key[0]}::{key[1]}"] = hit
    # The analyser works: bash is the canonical process runner (§2.1 examples).
    require(any(k.endswith("::BashTool") for k in reaching), f"the reach analysis missed BashTool: {sorted(reaching)}")

    assert EXEMPTIONS.exists(), f"item 2 not built: {EXEMPTIONS.relative_to(REPO)} does not exist"
    data = json.loads(EXEMPTIONS.read_text(encoding="utf-8"))
    entries = data.get("exemptions", []) if isinstance(data, dict) else []
    exempt = {}
    for e in entries:
        assert isinstance(e, dict) and e.get("class") and str(e.get("reason", "")).strip(), f"bad exemption: {e!r}"
        exempt[e["class"]] = e["reason"]

    violations = []
    for qual, hit in sorted(reaching.items()):
        mod, cname = qual.split("::")
        _m, cdef = _tool_closure()[(mod, cname)]
        if not _is_effect_all(_own_effect_value(cdef)) and qual not in exempt:
            violations.append(f"{qual} ({hit})")
    assert not violations, f"{len(violations)} sink-reaching tools neither declare ALL nor are exempt: {violations}"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 2: the effect record (§2.2)")
def test_item2_effect_record_semantics() -> None:
    """§2.2: ``Effect(0)`` and a non-Effect value are refused at class creation (TypeError); a subclass of a READ
    tool with no own ``effect`` records ALL; reassigning ``cls.effect`` after creation makes admission refuse
    (``ToolEffectRefused``, loudly); ``exceeds`` is the subset check, so WRITE_LOCAL exceeds READ|NETWORK."""
    base = _require_effect()
    from maxim.tools.base import Effect, Tool, ToolOutput, effect_of, exceeds
    from maxim.tools.registry import ToolRegistry

    refused = _tool_effect_refused()

    def _make(effect: Any) -> type:
        return type("ZeroProbe", (Tool,), {"name": "zero", "effect": effect, "execute": lambda self, **kw: None})

    _expect_raises(TypeError, lambda: _make(Effect(0)), "Effect(0) was accepted at class creation")
    _expect_raises(TypeError, lambda: _make("READ"), "a non-Effect effect was accepted at class creation")
    _expect_raises(TypeError, lambda: _make(1), "an int effect was accepted at class creation")

    read_tool = _probe_tool("read_probe", Effect.READ)
    assert effect_of(read_tool) == Effect.READ, "a declared READ tool does not record READ"

    sub = type(type(read_tool).__name__ + "Sub", (type(read_tool),), {"execute": lambda self, **kw: ToolOutput(True)})
    assert "effect" not in sub.__dict__, "fixture: the subclass must not declare its own effect"
    assert effect_of(sub()) == Effect.ALL, "an inherited READ counted: a subclass with no own effect must record ALL"
    assert effect_of(SimpleNamespace(name="duck")) == Effect.ALL, "a duck-typed tool with no record is not ALL"

    mutated_cls = type(_probe_tool("mut_probe", Effect.READ))
    mutated = mutated_cls()
    mutated_cls.effect = Effect.ALL  # after creation: the record says READ
    _expect_raises(refused, lambda: ToolRegistry().register(mutated), "a reassigned cls.effect was admitted")

    assert exceeds(Effect.WRITE_LOCAL, Effect.READ | Effect.NETWORK), "WRITE_LOCAL does not exceed READ|NETWORK"
    assert not exceeds(Effect.READ, Effect.READ | Effect.NETWORK), "READ exceeds READ|NETWORK"
    assert exceeds(Effect.ALL, Effect.READ | Effect.SELF_SCOPE | Effect.WRITE_LOCAL), "ALL fits a partial ceiling"
    names = {"READ", "SELF_SCOPE", "WRITE_LOCAL", "NETWORK", "COMMS", "ACTUATE", "GOVERN"}
    assert {m.name for m in base.Effect} == names, f"Effect flags are not the seven of §2.1: {list(base.Effect)}"


def _full_default_registry(tmp_path: Path) -> Any:
    """``build_tool_registry`` with every optional block that needs no network and no robot: the internet tools
    (construction only), mode/autonomy, sleep, response and comms tools (fake gateway), contained to *tmp_path*."""
    from maxim.agents.autonomy import AutonomyController
    from maxim.runtime.bootstrap import build_tool_registry
    from maxim.utils.response_output import ResponseOutput

    try:
        return build_tool_registry(
            internet_launch_enabled=True,
            autonomy_controller=AutonomyController(),
            response_output=ResponseOutput(sandbox_path=str(tmp_path)),
            gateway=MagicMock(name="gateway"),
            state_manager=MagicMock(name="state_manager"),
            allowed_dirs_override=[str(tmp_path)],
        )
    except Exception as e:
        raise ScenarioBroken(f"build_tool_registry raised: {type(e).__name__}: {e}") from e


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="#1315 item 2: a READ ceiling refuses every non-READ built-in tool"
)
def test_item2_registration_sweep_under_read_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Registration sweep (a): every built-in tool of a fully built default registry, registered into a
    ``ToolRegistry(ceiling=Effect.READ)``, raises ``ToolEffectRefused`` unless ``effect_of`` is READ."""
    monkeypatch.chdir(tmp_path)
    default = _full_default_registry(tmp_path)
    tools = [default.get(n) for n in default.list_all()]
    require(len(tools) >= 20, f"the default registry built only {len(tools)} tools")
    require({"bash", "write_file", "send_message", "http_fetch"} <= {t.name for t in tools}, "sweep tools missing")

    _require_effect()
    from maxim.tools.base import Effect, effect_of
    from maxim.tools.registry import ToolRegistry

    refused = _tool_effect_refused()
    wrong = []
    for t in tools:
        reg = ToolRegistry(ceiling=Effect.READ)
        try:
            reg.register(t)
            admitted = True
        except refused:
            admitted = False
        if admitted != (effect_of(t) == Effect.READ):
            wrong.append(f"{t.name} (effect {effect_of(t)!r}, admitted={admitted})")
    assert not wrong, f"the READ ceiling admitted/refused wrongly: {wrong}"


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="#1315 item 2: same-name replacement refused below ALL (V10)"
)
def test_item2_same_name_replacement_below_all_refuses() -> None:
    """§2.2/V10: under any ceiling below ALL a same-name replacement refuses; under ALL it is allowed (and the
    replacement wins), as ``register_tool`` needs. The refusal type is not pinned; ``ToolEffectRefused`` is chosen,
    since the replacement is refused by the ceiling layer."""
    _require_effect()
    from maxim.tools.base import Effect
    from maxim.tools.registry import ToolRegistry

    refused = _tool_effect_refused()
    first, second = _probe_tool("dup", Effect.READ), _probe_tool("dup", Effect.READ)
    narrow = ToolRegistry(ceiling=Effect.READ | Effect.SELF_SCOPE)
    narrow.register(first)
    _expect_raises(refused, lambda: narrow.register(second), "a same-name replacement was allowed below ALL")
    assert narrow.get("dup") is first, "the refused replacement still replaced the original"

    wide = ToolRegistry(ceiling=Effect.ALL)
    wide.register(first)
    wide.register(second)
    assert wide.get("dup") is second, "under ALL the replacement must be allowed (register_tool, V10)"


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="#1315 item 2: SimToolRegistry.get raises KeyError below ALL"
)
def test_item2_sim_registry_unknown_name_raises_below_all() -> None:
    """§2.2: ``SimToolRegistry.get`` of an unknown name under a ceiling below ALL raises ``KeyError`` instead of
    fabricating a ``_FallbackRedirectTool``; under ALL the orchestrator's fallback stays."""
    from maxim.simulation.tools import SimToolRegistry, _FallbackRedirectTool

    require(isinstance(SimToolRegistry().get("nope"), _FallbackRedirectTool), "today's fallback is gone")
    _require_effect()
    from maxim.tools.base import Effect

    narrow = SimToolRegistry(ceiling=Effect.READ)
    _expect_raises(KeyError, lambda: narrow.get("no_such_tool"), "an unknown name below ALL fabricated a tool")
    assert isinstance(SimToolRegistry(ceiling=Effect.ALL).get("nope"), _FallbackRedirectTool), "ALL lost the fallback"


# ══ item 3 ══════════════════════════════════════════════════════════════


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="#1315 item 3: policy= required everywhere; permissions= removed"
)
def test_item3_policy_is_required_and_permissions_removed() -> None:
    """V3 + §1A.5: ``Executor(registry)``, ``build_executor(registry, pain_bus=None)`` and ``AgentConfig(...)``
    without a policy each raise TypeError; a duck-typed policy is a TypeError; everything after the registry is
    keyword-only; ``permissions=`` is no longer accepted anywhere."""
    from maxim.runtime.agent_factory import AgentConfig
    from maxim.runtime.bootstrap import build_executor
    from maxim.runtime.executor import Executor
    from maxim.tools.registry import ToolRegistry

    reg = ToolRegistry()
    _expect_raises(TypeError, lambda: Executor(reg), "Executor(registry) with no policy was accepted")
    _expect_raises(TypeError, lambda: build_executor(reg, pain_bus=None), "build_executor with no policy accepted")
    _expect_raises(TypeError, lambda: AgentConfig(agent_id="a"), "AgentConfig(...) with no policy was accepted")

    tp = _require_policy()
    policy = tp.ToolPolicy.unrestricted(reason="red-gate probe")
    try:
        Executor(reg, policy=policy)
    except Exception as e:
        raise ScenarioBroken(f"Executor(reg, policy=unrestricted) raised: {e}") from e
    duck = SimpleNamespace(ceiling=None, permissions=None, pinned=None, reason="duck")
    _expect_raises(TypeError, lambda: Executor(reg, policy=duck), "a duck-typed policy was accepted")
    _expect_raises(TypeError, lambda: Executor(reg, None), "a positional second argument was accepted")
    _expect_raises(
        TypeError, lambda: Executor(reg, policy=policy, permissions=None), "Executor still accepts permissions="
    )
    _expect_raises(
        TypeError,
        lambda: build_executor(reg, pain_bus=None, policy=policy, permissions=None),
        "build_executor still accepts permissions=",
    )
    _expect_raises(
        TypeError, lambda: AgentConfig(agent_id="a", policy=policy, permissions=None), "AgentConfig has permissions="
    )
    try:
        AgentConfig(agent_id="a", policy=None)  # §3: None means "no executor is built" (create_npc_agent)
    except Exception as e:
        raise AssertionError(f"AgentConfig(policy=None) must be accepted for no-executor configs: {e}") from e


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 3: ToolPolicy is sealed and unforgeable")
def test_item3_tool_policy_is_sealed() -> None:
    """§3: no public constructor; ``__new__`` without the key raises; assignment and ``del`` raise; subclassing
    raises; ``copy.copy``/``deepcopy`` return the same object; pickling raises; ``pinned`` is snapshotted. The
    exception types are not pinned (any exception counts as a refusal)."""
    tp = _require_policy()
    ToolPolicy = tp.ToolPolicy
    _require_effect()
    from maxim.agents.permissions import AgentPermissions
    from maxim.tools.base import Effect

    _expect_raises(Exception, lambda: ToolPolicy(), "ToolPolicy() constructed without the key")
    _expect_raises(
        Exception,
        lambda: ToolPolicy(permissions=AgentPermissions(), ceiling=Effect.ALL, pinned=None, reason="forged"),
        "ToolPolicy(...) constructed directly without the key",
    )
    _expect_raises(Exception, lambda: ToolPolicy.__new__(ToolPolicy), "ToolPolicy.__new__ without the key")

    p = ToolPolicy.only(["read_probe"], ceiling=Effect.READ, reason="red-gate probe")

    def _assign() -> None:
        p.ceiling = Effect.ALL

    def _delete() -> None:
        del p.ceiling

    def _new_attr() -> None:
        p.extra = 1

    _expect_raises(Exception, _assign, "a ToolPolicy field was reassigned")
    _expect_raises(Exception, _delete, "a ToolPolicy field was deleted")
    _expect_raises(Exception, _new_attr, "a new attribute was set on a ToolPolicy")
    assert p.ceiling == Effect.READ, "the ceiling changed"
    _expect_raises(Exception, lambda: type("Wider", (ToolPolicy,), {}), "ToolPolicy was subclassed")
    assert copy.copy(p) is p, "copy.copy did not return the same ToolPolicy"
    assert copy.deepcopy(p) is p, "copy.deepcopy did not return the same ToolPolicy"
    _expect_raises(Exception, lambda: pickle.dumps(p), "a ToolPolicy was pickled")

    a, b = _probe_tool("pin_a", Effect.READ), _probe_tool("pin_b", Effect.READ)
    source = {"pin_a": a}
    pinned = ToolPolicy.pinned(source, ceiling=Effect.READ, reason="red-gate probe")
    source["pin_b"] = b
    assert "pin_b" not in pinned.pinned, "mutating the source dict after build added a pin"
    assert pinned.pinned["pin_a"] is a, "the pin is not the instance given"

    def _pin_write() -> None:
        pinned.pinned["pin_b"] = b

    _expect_raises(TypeError, _pin_write, "the pinned mapping is writable")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 3: dispatch checks the object it runs")
def test_item3_dispatch_checks_the_object_it_runs() -> None:
    """§3 "Dispatch checks the object it runs": register READ tool A; the registry then answers A on the first
    lookup and an above-ceiling tool B (same name) on every later one, so a check-then-get executor checks A and
    runs B. Nothing above the ceiling may run, and the executor looks the tool up once (``get`` + ``find`` calls
    counted together, per "Get the tool once"; alias resolution must reuse that lookup)."""
    tp = _require_policy()
    _require_effect()
    from maxim.runtime.executor import Executor
    from maxim.tools.base import Effect
    from maxim.tools.registry import ToolRegistry

    ran: list[str] = []
    a = _probe_tool("race", Effect.READ, ran=ran)
    b_ran: list[str] = []
    b = _probe_tool("race", Effect.ALL, ran=b_ran)
    reg = ToolRegistry()
    reg.register(a)
    policy = tp.ToolPolicy.only(["race"], ceiling=Effect.READ, reason="red-gate probe")
    ex = Executor(reg, policy=policy)

    lookups: list[str] = []
    real_get, real_find = reg.get, reg.find

    def _swap(real: Callable[[str], Any], label: str) -> Callable[[str], Any]:
        def lookup(name: str) -> Any:
            lookups.append(label)
            hit = real(name)
            return hit if len(lookups) == 1 or hit is None else b

        return lookup

    reg.get = _swap(real_get, "get")  # type: ignore[method-assign]
    reg.find = _swap(real_find, "find")  # type: ignore[method-assign]
    ex.execute({"tool_name": "race", "params": {}})
    assert not b_ran, f"an above-ceiling tool swapped in after the check RAN (lookups: {lookups})"
    assert len(lookups) == 1, f"dispatch looked the tool up {len(lookups)} times ({lookups}), not once"
    assert ran == ["race"], "the checked object did not run"


def _sub_sim_rig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Stub the sub-sim's heavy pieces; capture the child executor the loop would be handed."""
    import maxim.simulation.bridge as bridge_mod

    loop_calls: list[tuple[Any, ...]] = []
    registry_calls: list[dict[str, Any]] = []
    import maxim.runtime.bootstrap as bootstrap

    real_build = bootstrap.build_tool_registry

    def build_tool_registry(**kw: Any) -> Any:
        registry_calls.append(kw)
        return real_build(**kw)

    monkeypatch.setattr(bootstrap, "build_tool_registry", build_tool_registry)
    monkeypatch.setattr("maxim.runtime.agent_loop.run_agentic_loop", lambda *a, **kw: loop_calls.append(a))
    monkeypatch.setattr("maxim.agents.maxim_agent.MaximAgent", lambda *a, **kw: MagicMock(name="sub_agent"))
    monkeypatch.setattr(
        bridge_mod.SimulationBridge,
        "send_and_wait",
        lambda self, goal: {"turn": 0, "response": None, "actions": [], "blocked": [], "timed_out": False},
    )
    from maxim.simulation.tools import SpawnSubSimulationTool

    tool = SpawnSubSimulationTool(llm_router=None, sim_tmpdir=str(tmp_path), sandbox_dirs=[str(tmp_path)])
    return SimpleNamespace(tool=tool, loop_calls=loop_calls, registry_calls=registry_calls)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1315 item 3: a spawn under a narrow policy builds a child refusing bash",
)
def test_item3_composite_spawn_builds_narrowed_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§3 composite: ``SpawnSubSimulationTool`` dispatched through an executor with a narrow policy reads
    ``current_policy`` and builds its child with ``current.narrowed(...)``, so the child refuses ``bash``. The loop
    run is mocked; the child executor handed to it is inspected (``permits`` and ``check`` on its ``bash``)."""
    monkeypatch.chdir(tmp_path)
    rig = _sub_sim_rig(monkeypatch, tmp_path)
    tp = _require_policy()
    _require_effect()
    import maxim.tools.base as base
    from maxim.runtime.executor import Executor
    from maxim.tools.base import Effect
    from maxim.tools.registry import ToolRegistry

    assert hasattr(base, "current_policy"), "item 3 not built: tools/base.py has no current_policy()"
    reg = ToolRegistry([rig.tool])
    # Narrow by allow-list, not by ceiling: the spawn tool is itself ALL (§2.1 examples), so a ceiling below ALL
    # would refuse the spawn before it could build anything. ``narrowed`` intersects allow-lists, so the child
    # inherits ["spawn_sub_simulation"] and must refuse bash.
    policy = tp.ToolPolicy.only(["spawn_sub_simulation"], ceiling=Effect.ALL, reason="red-gate probe")
    out = Executor(reg, policy=policy).execute({"tool_name": "spawn_sub_simulation", "params": {"goal": "hello"}})
    if rig.tool._sub_thread is not None:
        rig.tool._sub_thread.join(timeout=10)
    assert rig.loop_calls, f"the spawn built no child under a narrow dispatching policy (result: {out})"
    child = rig.loop_calls[-1][5]  # run_agentic_loop(agent, env, state, memory, engine, executor, ...)
    assert not child.permits("bash"), "the spawned child's policy permits bash"
    bash = child.registry.find("bash")
    if bash is not None:
        assert child.check(bash) is not None, "the child's check() lets bash through"


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="#1315 item 3: a spawn with no current_policy refuses (fail-closed)"
)
def test_item3_composite_spawn_refuses_without_dispatch_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§3: a composite reads ``current_policy`` and refuses if it is absent. ``.execute`` called directly (no
    executor dispatch, so no contextvar) returns a failed ToolOutput and builds no child. Today it builds one."""
    monkeypatch.chdir(tmp_path)
    rig = _sub_sim_rig(monkeypatch, tmp_path)
    out = rig.tool.execute(goal="hello")
    if rig.tool._sub_thread is not None:
        rig.tool._sub_thread.join(timeout=10)
    require(out is not None, "execute returned nothing")
    assert not rig.registry_calls and not rig.loop_calls, "a spawn with no dispatching policy built a child"
    assert not getattr(out, "success", True), f"a spawn with no dispatching policy succeeded: {out}"


@pytest.mark.parametrize("which", ["both_refused", "sense_refused"])
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="#1315 item 3 / V6: auto_sense calls executor.check() before both calls"
)
def test_item3_auto_sense_checks_both_direct_calls(which: str) -> None:
    """V6 / §1A.10: ``runtime/loop_perception.py::auto_sense`` calls ``executor.check(tool)`` before BOTH of its
    direct calls -- the auto-fire loop and the interoception ``sense`` call. ``both_refused``: neither runs.
    ``sense_refused``: the auto-fire tool is allowed (so interoception is reached) and ``sense`` must not run."""
    tp = _require_policy()
    _require_effect()
    from maxim.runtime.executor import Executor
    from maxim.runtime.loop_perception import auto_sense
    from maxim.tools.base import Effect
    from maxim.tools.registry import ToolRegistry

    assert hasattr(Executor, "check"), "item 3 not built: Executor has no check()"
    ran: list[str] = []
    af_effect = Effect.ALL if which == "both_refused" else Effect.READ
    presence = _probe_tool("probe_presence", af_effect, ran=ran, auto_fire=True)
    object.__setattr__(
        presence, "_entity_map", SimpleNamespace(list_self_entities=lambda: [SimpleNamespace(name="me")])
    )
    sense = _probe_tool("sense", Effect.ALL, ran=ran)
    reg = ToolRegistry([presence, sense])
    require(reg.get_auto_fire_tools() == [presence], "the auto-fire fixture is not auto-fire")
    allowed = ["probe_presence"] if which == "sense_refused" else []
    policy = tp.ToolPolicy.only(allowed, ceiling=Effect.READ, reason="red-gate probe")
    ex = Executor(reg, policy=policy)
    auto_sense(step_num=1, observation={"transcript": "hello"}, executor=ex)
    expected = ["probe_presence"] if which == "sense_refused" else []
    assert ran == expected, f"auto_sense ran {ran} under a policy refusing them (expected {expected})"


# ══ O1 / V10: the maxim.run seam ════════════════════════════════════════


def _drive_maxim_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, learning: bool) -> SimpleNamespace:
    """Run the REAL ``maxim.run`` with its real ``build_tool_registry`` and executor build (``build_executor``, or
    ``AgentFactory.create_full_agent`` when ``learning``), the LLM, agent and loop mocked; return the executor the
    loop was handed. The seam: ``api.run`` is driven end to end, so the policy is asserted on the object that
    would really dispatch (the choice the task allows; no src seam is needed)."""
    import maxim.api as api

    loop_calls: list[dict[str, Any]] = []
    router = SimpleNamespace(n_ctx=8192, get_token_counter=lambda: None)
    worker = MagicMock(name="llm_worker")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api, "_validate_model", lambda model: None)
    monkeypatch.setattr(api, "_resolve_model", lambda model: model)
    monkeypatch.setattr(api, "configure", lambda **kwargs: None)
    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", lambda: (router, None))
    monkeypatch.setattr("maxim.agents.llm_worker.LLMWorker", lambda **kw: worker)
    monkeypatch.setattr(
        "maxim.agents.maxim_agent.MaximAgent",
        lambda **kw: MagicMock(name="agent", spec=["wire_robot", "wire_memory_hub"]),
    )
    monkeypatch.setattr("maxim.runtime.agent_loop.run_agentic_loop", lambda **kw: loop_calls.append(kw))
    home = tmp_path / "home"
    try:
        api.run(model="test-profile", home_dir=str(home), learning=learning)
    except Exception as e:
        raise ScenarioBroken(f"maxim.run(learning={learning}) raised: {type(e).__name__}: {e}") from e
    require(len(loop_calls) == 1, "maxim.run never reached the loop")
    executor = loop_calls[0]["executor"]
    require(executor is not None and hasattr(executor, "execute"), "the loop got no executor")
    return SimpleNamespace(executor=executor, home=os.path.realpath(api._api_home(str(home))))


@pytest.mark.parametrize("learning", [False, True])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1315 item 3 / O1: maxim.run's executor carries ToolPolicy.unrestricted (V10) and allowed_dirs == [home]",
)
def test_o1_maxim_run_executor_carries_v10_policy(
    learning: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O1 flip criterion (§1A.4): in both ``maxim.run`` branches the executor carries
    ``ToolPolicy.unrestricted(reason="maxim.run: stable verb, V10")`` -- a real ``ToolPolicy`` with ceiling ALL and
    a reason containing "V10", which a translated ``None`` cannot produce -- and the filesystem tools' allowed dirs
    are exactly the run's home (#1342). The executor's policy attribute is not pinned by the notes; ``.policy`` is
    chosen. "The run's home" is read as ``_api_home(home_dir)``."""
    run = _drive_maxim_run(monkeypatch, tmp_path, learning=learning)
    tp = _require_policy()
    _require_effect()
    from maxim.tools.base import Effect

    policy = getattr(run.executor, "policy", None)
    assert type(policy) is tp.ToolPolicy, f"maxim.run's executor carries no ToolPolicy (got {policy!r})"
    assert policy.ceiling == Effect.ALL, f"maxim.run's ceiling is {policy.ceiling!r}, not ALL (V10)"
    assert "V10" in policy.reason, f"maxim.run's policy reason does not cite V10: {policy.reason!r}"
    fs = {
        n: t._allowed_dirs
        for n in run.executor.registry.list_all()
        if hasattr(t := run.executor.registry.get(n), "_allowed_dirs")
    }
    require(fs, "no filesystem tool in maxim.run's registry")
    wrong = {n: d for n, d in fs.items() if d != [run.home]}
    assert not wrong, f"filesystem tools are not confined to the run's home {run.home}: {wrong}"


@pytest.mark.parametrize("learning", [False, True])
def test_v10_registered_tools_dispatch_under_maxim_run(
    learning: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard (V10, §1A.4): a tool registered through ``maxim.register_tool`` and one through
    ``@maxim.tool``, neither declaring an effect, are present in, and dispatch through, the registry and executor
    ``maxim.run`` builds -- under the passive mode source the loop installs (``loop_setup``). Passes today; it must
    keep passing when the ceiling lands (``maxim.run`` keeps ALL)."""
    import maxim.api as api
    from maxim.tools.base import Tool, ToolOutput

    calls: list[str] = []

    class _RegisteredProbe(Tool):
        name = "v10_registered_probe"
        description = "probe"
        input_schema: dict[str, Any] = {}

        def execute(self, **kwargs: Any) -> ToolOutput:
            calls.append(self.name)
            return ToolOutput(success=True, output="ok")

    def v10_decorated_probe(text: str = "x") -> str:
        """probe"""
        calls.append("v10_decorated_probe")
        return text

    api.register_tool(_RegisteredProbe())
    api.tool(v10_decorated_probe)
    try:
        run = _drive_maxim_run(monkeypatch, tmp_path, learning=learning)
    finally:
        api.unregister_tool("v10_registered_probe")
        api.unregister_tool("v10_decorated_probe")
    ex = run.executor
    ex.set_mode_source(lambda: "passive")
    for name in ("v10_registered_probe", "v10_decorated_probe"):
        assert ex.registry.find(name) is not None, f"{name} is missing from maxim.run's registry"
        out = ex.execute({"tool_name": name, "params": {}})
        assert out.success, f"{name} did not dispatch under maxim.run: {out.error}"
    assert calls == ["v10_registered_probe", "v10_decorated_probe"], calls


# ══ O2 / O5: the mode gate allows acting tools a ceiling must refuse ════


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1315 item 2+3 / O2,O5: a READ|SELF_SCOPE ceiling refuses write_file/send_message/call_user",
)
def test_o2_o5_ceiling_refuses_acting_tools_the_mode_gate_allows(tmp_path: Path) -> None:
    """advisor_mode.md O2/O5: the mode gate is a name list, and ``write_file``, ``send_message`` and ``call_user``
    pass it in passive mode. Under a ceiling of READ|SELF_SCOPE (allow-listed by name, so only the ceiling can
    refuse) each is refused at dispatch with ``ToolErrorKind.POLICY_CEILING`` and its body never runs."""
    from maxim.modes.definitions import get_mode
    from maxim.tools.comms import CallUserTool, SendMessageTool
    from maxim.tools.filesystem import WriteFileTool
    from maxim.tools.registry import ToolRegistry

    passive = get_mode("passive")
    names = ("write_file", "send_message", "call_user")
    require(passive is not None and all(passive.dispatch_refusal(n) is None for n in names), "O2 no longer holds")

    gateway = MagicMock(name="gateway")
    tools = [WriteFileTool(allowed_dirs=[str(tmp_path)]), SendMessageTool(gateway), CallUserTool(gateway)]
    ran: list[str] = []
    for t in tools:
        real = t.execute
        object.__setattr__(t, "execute", lambda _real=real, _n=t.name, **kw: (ran.append(_n), _real(**kw))[1])

    tp = _require_policy()
    _require_effect()
    base = _require_gate_kinds()
    from maxim.runtime.executor import Executor
    from maxim.tools.base import Effect

    policy = tp.ToolPolicy.only(list(names), ceiling=Effect.READ | Effect.SELF_SCOPE, reason="red-gate probe")
    ex = Executor(ToolRegistry(tools), policy=policy)
    ex.set_mode_source(lambda: "passive")
    params = {
        "write_file": {"path": str(tmp_path / "x.txt"), "content": "x"},
        "send_message": {"recipient": "someone", "message": "hi"},
        "call_user": {"recipient": "someone", "message": "hi"},
    }
    kinds = {n: ex.execute({"tool_name": n, "params": params[n]}).error_kind for n in names}
    assert not ran, f"a ceiling of READ|SELF_SCOPE let these run: {ran}"
    assert all(k is base.ToolErrorKind.POLICY_CEILING for k in kinds.values()), f"refusal kinds: {kinds}"
