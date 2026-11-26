#!/usr/bin/env python3
"""Cuts of exception text that are not URL-reduced first: a RATCHET over the forms in ``src/maxim`` today (#1801 F2).

A boundary that reduces URLs (the record factory, ``StructuredFormatter``, ``sim_log``, the wire choke points,
``ToolOutput.error``) reduces a CUT URL when a site has already cut it: ``https://ghp_TOKEN@github.com`` cut to
``https://ghp_TOK`` survives as a "host", and a tail cut (``stderr[-500:]``) can drop the scheme so no URL is seen at
all. So a site reduces before it slices (``redact_urls(str(e))[:50]``). This lint keeps the known forms from coming
back; it is NOT the boundary (docs/plans/log_url_redaction_1801_approach.md v28, §Guard).

**Taint sources:** an ``except ... as`` name; a parameter named ``e``/``exc``/``err``/``error`` or annotated as an
exception (a name ending in ``Exception``/``Error``); any ``.stderr``/``.stdout`` attribute; ``traceback.format_exc()``
(and ``format_exception``/``format_exc`` by bare name).

**Propagation**, flow-insensitive within one scope: through ``str()``/``repr()``/any call with a tainted argument
(``json.dumps``, ``" ".join``, ``dict(...)``), method calls on a tainted receiver, f-strings, ``%``/``+``, ``.attr``,
subscripts, containers, ``or``/``if``-``else``, and assignments (plain, augmented, annotated, tuple-unpack, walrus,
``for`` and ``with`` targets). A name is tainted when ANY of its assignments is (a branched assignment counts). A
nested function or lambda sees the taint of a name it reads from its enclosing scope (a closure). A call to
``len``/``bool``/``int``/``float``/``type``/``isinstance``/``id``/``hash``/``callable``/``hasattr`` returns no text
and so is never tainted; ``type(e).__name__`` is clean.

**The sanitizers:** a ``redact_urls(...)``/``url_scrubbed(...)`` call (by bare or attribute name) is clean, and a
name every assignment of which is one is clean.

**A cut** of a tainted value is:
- a slice (``x[:50]``, ``x[-500:]``) or a ``slice(...)`` index;
- a format precision: an f-string ``{x:.50}``/``{x!s:.50}``, ``"%.50s" % x`` or ``"%.*s" % (n, x)``,
  ``"{:.50}".format(x)`` and ``format(x, ".50")`` (a numeric presentation type, ``.1f``, is not a text cut);
- ``textwrap.shorten(x, ...)``;
- a ``split``/``rsplit``/``splitlines``/``partition``/``rpartition`` result (the method, ``str.partition(e, "@")`` or
  ``re.split(p, x)``), when it is subscripted, ``.pop()``-ed, tuple-unpacked, or bound to a name that is.

A cut passes only when its value is itself a sanitizer call or a clean name: ``redact_urls(str(e)[:50])`` FAILS (the
slice runs first).

**Allowlist:** a reviewed cut in ``src/`` today that does not reach a guarded sink with a cut URL, keyed by
``file::innermost_function`` (the dotted qualname of the innermost enclosing def, ``<lambda>``, or ``<module>``) and the
``ast.unparse`` text of the cut, with an EXACT count and a reason. A hit beyond an entry's count fails (P2), and an
entry with fewer hits than its count fails as stale (Q2), so the allowlist can only shrink by being edited.

**Blind spots, stated and not chased** (rounds 7-24 showed a static taint lint can always be evaded once more; the
boundary is the gate, and its deletion probes prove each sink): indirect callees and labels (``getattr``,
``operator.attrgetter``/``methodcaller``, ``functools.partial``, aliases, class-body, ``setattr`` or ``globals()``
labels, other modules); cuts inside helpers (a function that slices its parameter, unless the parameter is named
as a source); exotic cutting callables (``textwrap.wrap``/``fill``, ``itertools.islice``, ``deque(maxlen=)``,
``reprlib``, ``__mod__``/``operator.mod``, ``format_map``/``vformat``, dunder calls); a precision given by a
variable format string.

Exits: 0 clean; 1 a new or over-count cut, or a stale allowlist entry (stderr); 2 unexpected error.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCOPE = "src/maxim"

SOURCE_PARAMS = frozenset({"e", "exc", "err", "error"})
SOURCE_ATTRS = frozenset({"stderr", "stdout"})
SOURCE_CALLS = frozenset({"format_exc", "format_exception"})
SANITIZERS = frozenset({"redact_urls", "url_scrubbed"})
NON_TEXT_CALLS = frozenset(
    {"len", "bool", "int", "float", "type", "isinstance", "issubclass", "id", "hash", "callable", "hasattr"}
)
SPLITTERS = frozenset({"split", "rsplit", "splitlines", "partition", "rpartition"})
_PRECISION_PCT = re.compile(r"%[-+ #0]*\d*\.(?:\d+|\*)[sra]?")
_PRECISION_BRACE = re.compile(r"\{[^{}]*:[^{}]*\.\d+[sra]?\}")
_NUMERIC_TYPES = frozenset("bcdoxXneEfFgG%")


@dataclass(frozen=True)
class Hit:
    """One cut of tainted text. SHAPE-FROZEN at 1.0 (CC3): lint-internal, never persisted or sent."""

    file: str
    scope: str
    text: str
    line: int


# file::innermost_function -> {ast.unparse text of the cut: (exact count, reason)}
ALLOWLIST: dict[str, dict[str, tuple[int, str]]] = {
    "src/maxim/doctor/checks.py::_child_maxim_dir": {
        "lines[-1]": (1, "a child `python -c` stderr's last line, shown by `maxim doctor` only (print, out of scope)"),
        "lines[-1][:200]": (1, "the same doctor-only line"),
    },
    "src/maxim/doctor/platform_detect.py::detect_lan_ip": {
        "result.stdout.strip().split()[0]": (
            1,
            "`hostname -I` output parsed for an IP: subprocess output parsed for a value, never logged or sent cut",
        ),
    },
    "src/maxim/doctor/platform_detect.py::detect_wsl_ip": {
        "parts[0]": (
            1,
            "`ip route` output parsed for an IP: subprocess output parsed for a value, never logged or sent cut",
        ),
    },
    "src/maxim/models/language/router.py::LLMRouter._note_provider_down": {
        "f'down_{http_status}'[:200]": (1, "an int HTTP status read off the exception, no text"),
    },
    "src/maxim/models/language/router.py::LLMRouter._try_provider": {
        "f'model_missing:{requested}'[:200]": (1, "the typed error's requested model name, not its message"),
        "f'timeout_{elapsed:.1f}s'[:200]": (1, "a float elapsed time read off the exception, no text"),
    },
    "src/maxim/peer/install_core.py::install_on_target": {
        "e.response.text[:500]": (1, "the leader's error body, which only reaches print on the peer's terminal (AF8)"),
    },
    "src/maxim/planning/plan_manager.py::PlanManager._build_replan_context": {
        "getattr(m, 'content', str(m))[:100]": (
            1,
            "a recalled memory's content (tainted only because the recall query holds the error); memory text, not exception text",
        ),
    },
    "src/maxim/runtime/leader_proxy.py::_query_nvidia_smi": {
        "result.stdout.strip().split('\\n')[0]": (
            1,
            "`nvidia-smi` output parsed for VRAM: subprocess output parsed for a value, never logged or sent cut",
        ),
    },
    "src/maxim/runtime/llm_spawn_record.py::established_connections": {
        "line[1:]": (1, "`lsof -Fn` field lines (addresses, no URLs) for a diagnostic"),
        "line[1:].split('->', 1)[-1]": (1, "the same lsof address field"),
    },
    "src/maxim/runtime/orphan_reaper.py::_find_orphans_ps": {
        "pid_str, etime_str, cmdline = parts": (
            1,
            "`ps` output parsed for pids: subprocess output parsed for a value, never logged or sent cut",
        ),
    },
    "src/maxim/runtime/prefetch.py::discover_files_with_find": {
        "found_files[:max_files]": (1, "a LIST of `find` paths capped by count, not a text cut"),
    },
    "src/maxim/runtime/system_metrics.py::_memory_macos": {
        "parts[0]": (
            1,
            "`vm_stat` output parsed for numbers: subprocess output parsed for a value, never logged or sent cut",
        ),
        "parts[1]": (1, "the same vm_stat field"),
        "result.stdout.strip().split('\\n')[1:]": (1, "the same vm_stat lines"),
    },
    "src/maxim/runtime/system_metrics.py::_wifi_linux": {
        "line.split('Signal level=')[1]": (
            1,
            "`iwconfig` output parsed for a number: subprocess output parsed for a value, never logged or sent cut",
        ),
        "line.split('Signal level=')[1].split()[0]": (1, "the same iwconfig field"),
    },
    "src/maxim/runtime/system_metrics.py::_wifi_macos": {
        "key, _, val = line.partition(':')": (
            1,
            "`airport -I` key/value lines: subprocess output parsed for a value, never logged or sent cut",
        ),
    },
    "src/maxim/runtime/system_metrics.py::collect_gpu": {
        "result.stdout.strip().split('\\n')[0]": (
            1,
            "`nvidia-smi` output parsed for numbers: subprocess output parsed for a value, never logged or sent cut",
        ),
    },
    "src/maxim/simulation/container_runner.py::LocalDockerRunner.launch": {
        "container_id[:12]": (1, "a docker container id from `docker run` stdout, a hex id"),
    },
    "src/maxim/tunnel/cloudflared.py::cloudflared_version": {
        "result.stdout.strip().splitlines()[0]": (
            1,
            "`cloudflared --version` output: subprocess output parsed for a value, never logged or sent cut",
        ),
    },
    "src/maxim/utils/reachy_diagnostics.py::test_ping": {
        "parts[1]": (
            1,
            "`ping` output parsed for loss and latency: subprocess output parsed for a value, never logged or sent cut",
        ),
        "stats[1]": (1, "the same ping summary"),
    },
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_sanitizer(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_name(node.func) in SANITIZERS


def _annotated_exception(ann: ast.AST | None) -> bool:
    if ann is None:
        return False
    for n in ast.walk(ann):
        name = _call_name(n) if isinstance(n, (ast.Name, ast.Attribute)) else ""
        if name.endswith(("Exception", "Error")):
            return True
    return False


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for elt in target.elts for n in _target_names(elt)]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of ``scope`` that are not inside a nested scope (the nested scope's own node is included)."""
    out: list[ast.AST] = []
    stack: list[ast.AST] = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        out.append(node)
        if isinstance(node, _SCOPES):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return out


class _Scope:
    def __init__(self, node: ast.AST, inherited: frozenset[str], qualname: str) -> None:
        self.node = node
        self.qualname = qualname
        self.nodes = _own_nodes(node)
        self.assigned: dict[str, list[ast.AST | None]] = {}  # None = a source binding
        self.split_names: set[str] = set()
        self._collect_bindings()
        # A free name keeps its enclosing scope's taint (a closure, P3).
        self.tainted: set[str] = {n for n in inherited if n not in self.assigned}
        self._solve()

    def _bind(self, name: str, value: ast.AST | None) -> None:
        self.assigned.setdefault(name, []).append(value)

    def _collect_bindings(self) -> None:
        node = self.node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arg in [
                *a.posonlyargs,
                *a.args,
                *a.kwonlyargs,
                *([a.vararg] if a.vararg else []),
                *([a.kwarg] if a.kwarg else []),
            ]:
                self._bind(
                    arg.arg,
                    None if (arg.arg in SOURCE_PARAMS or _annotated_exception(arg.annotation)) else ast.Constant(None),
                )
        for n in self.nodes:
            if isinstance(n, ast.ExceptHandler) and n.name:
                self._bind(n.name, None)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    for name in _target_names(t):
                        self._bind(name, n.value)
            elif (
                isinstance(n, (ast.AugAssign, ast.AnnAssign)) and isinstance(n.target, ast.Name) and n.value is not None
            ):
                self._bind(n.target.id, n.value)
                if isinstance(n, ast.AugAssign):
                    self._bind(n.target.id, ast.Name(id=n.target.id))  # ``s += x`` keeps s's own taint
            elif isinstance(n, ast.NamedExpr):
                self._bind(n.target.id, n.value)
            elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
                for name in _target_names(n.target):
                    self._bind(name, n.iter)
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for item in n.items:
                    if item.optional_vars is not None:
                        for name in _target_names(item.optional_vars):
                            self._bind(name, item.context_expr)

    def _solve(self) -> None:
        changed = True
        while changed:
            changed = False
            for name, values in self.assigned.items():
                if name in self.tainted:
                    continue
                if any(v is None or self.taint(v) for v in values):
                    self.tainted.add(name)
                    changed = True
        for name, values in self.assigned.items():
            if any(v is not None and self.split_call(v) for v in values):
                self.split_names.add(name)

    def taint(self, node: ast.AST | None) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Name):
            return node.id in self.tainted
        if isinstance(node, ast.Attribute):
            return node.attr in SOURCE_ATTRS or self.taint(node.value)
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in SANITIZERS or name in NON_TEXT_CALLS:
                return False
            if name in SOURCE_CALLS:
                return True
            args = [*node.args, *(k.value for k in node.keywords)]
            if any(self.taint(a) for a in args):
                return True
            return isinstance(node.func, ast.Attribute) and self.taint(node.func.value)
        if isinstance(node, (ast.JoinedStr, ast.BoolOp)):
            return any(self.taint(v) for v in node.values)
        if isinstance(node, ast.FormattedValue):
            return self.taint(node.value)
        if isinstance(node, ast.BinOp):
            return self.taint(node.left) or self.taint(node.right)
        if isinstance(node, ast.IfExp):
            return self.taint(node.body) or self.taint(node.orelse)
        if isinstance(node, (ast.Subscript, ast.Starred, ast.Await, ast.NamedExpr)):
            return self.taint(node.value)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return any(self.taint(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            return any(self.taint(v) for v in [*node.keys, *node.values] if v is not None)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return self.taint(node.elt) or any(self.taint(g.iter) for g in node.generators)
        if isinstance(node, ast.DictComp):
            return self.taint(node.key) or self.taint(node.value) or any(self.taint(g.iter) for g in node.generators)
        return False

    def split_call(self, node: ast.AST) -> bool:
        """A ``split``/``partition`` family call over tainted text (method, ``str.partition(e, ...)``, ``re.split``)."""
        if not (isinstance(node, ast.Call) and _call_name(node.func) in SPLITTERS):
            return False
        if isinstance(node.func, ast.Attribute) and self.taint(node.func.value):
            return True
        return any(self.taint(a) for a in node.args)

    def _split_result(self, node: ast.AST) -> bool:
        return self.split_call(node) or (isinstance(node, ast.Name) and node.id in self.split_names)

    def cuts(self) -> list[ast.AST]:
        found: list[ast.AST] = []
        for n in self.nodes:
            if isinstance(n, ast.Subscript):
                if isinstance(n.slice, ast.Slice) or (
                    isinstance(n.slice, ast.Call) and _call_name(n.slice.func) == "slice"
                ):
                    if self.taint(n.value) and not _is_sanitizer(n.value):
                        found.append(n)
                elif self._split_result(n.value):
                    found.append(n)
            elif isinstance(n, ast.FormattedValue) and n.format_spec is not None:
                spec = "".join(v.value for v in n.format_spec.values if isinstance(v, ast.Constant))  # type: ignore[attr-defined]
                if "." in spec and spec[-1:] not in _NUMERIC_TYPES and self.taint(n.value):
                    found.append(n)
            elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod):
                if (
                    isinstance(n.left, ast.Constant)
                    and isinstance(n.left.value, str)
                    and _PRECISION_PCT.search(n.left.value)
                    and self.taint(n.right)
                ):
                    found.append(n)
            elif isinstance(n, ast.Call):
                found.extend(self._call_cut(n))
            elif isinstance(n, ast.Assign) and isinstance(n.targets[0], (ast.Tuple, ast.List)):
                if self._split_result(n.value):
                    found.append(n)
        return found

    def _call_cut(self, n: ast.Call) -> list[ast.AST]:
        name = _call_name(n.func)
        args = [*n.args, *(k.value for k in n.keywords)]
        if name == "shorten" and n.args and self.taint(n.args[0]):
            return [n]
        if name == "pop" and isinstance(n.func, ast.Attribute) and self._split_result(n.func.value):
            return [n]
        if name == "format" and isinstance(n.func, ast.Attribute):
            fmt = n.func.value
            if isinstance(fmt, ast.Constant) and isinstance(fmt.value, str) and _PRECISION_BRACE.search(fmt.value):
                if any(self.taint(a) for a in args):
                    return [n]
        if name == "format" and isinstance(n.func, ast.Name) and len(n.args) >= 2:
            spec = n.args[1]
            if isinstance(spec, ast.Constant) and isinstance(spec.value, str) and "." in spec.value:
                if spec.value[-1:] not in _NUMERIC_TYPES and self.taint(n.args[0]):
                    return [n]
        return []


def _scopes(tree: ast.Module) -> list[_Scope]:
    out: list[_Scope] = []

    def visit(node: ast.AST, inherited: frozenset[str], qual: str) -> None:
        scope = _Scope(node, inherited, qual)
        out.append(scope)
        for child in scope.nodes:
            if isinstance(child, _SCOPES):
                name = "<lambda>" if isinstance(child, ast.Lambda) else child.name  # type: ignore[union-attr]
                child_qual = name if qual == "<module>" else f"{qual}.{name}"
                # a class body's names are not visible to its methods (Python scoping)
                passed = frozenset() if isinstance(node, ast.ClassDef) else frozenset(scope.tainted)
                visit(child, passed, child_qual)

    visit(tree, frozenset(), "<module>")
    return out


def hits_in_source(source: str, rel: str) -> list[Hit]:
    tree = ast.parse(source)
    found: list[Hit] = []
    for scope in _scopes(tree):
        for cut in scope.cuts():
            found.append(Hit(rel, scope.qualname, ast.unparse(cut), getattr(cut, "lineno", 0)))
    return found


def scan(root: Path = REPO_ROOT) -> list[Hit]:
    found: list[Hit] = []
    for path in sorted((root / SCOPE).rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        found.extend(hits_in_source(path.read_text(encoding="utf-8"), rel))
    return found


def judge(found: list[Hit], allowlist: dict[str, dict[str, tuple[int, str]]] = ALLOWLIST) -> list[str]:
    """Failures: a cut no entry covers, a cut beyond its entry's count, an entry with fewer hits than its count."""
    counts = Counter((f"{h.file}::{h.scope}", h.text) for h in found)
    lines = {(f"{h.file}::{h.scope}", h.text): h.line for h in found}
    errors: list[str] = []
    for (key, text), n in sorted(counts.items()):
        allowed = allowlist.get(key, {}).get(text)
        if allowed is None:
            errors.append(
                f"{key} (line {lines[(key, text)]}): `{text}` cuts exception text before reducing it: "
                "reduce first, `redact_urls(...)[:N]` (#1801)"
            )
        elif n > allowed[0]:
            errors.append(f"{key}: `{text}` has {n} hits, more than its allowlist count {allowed[0]}")
    for key, entries in sorted(allowlist.items()):
        for text, (count, _reason) in sorted(entries.items()):
            n = counts.get((key, text), 0)
            if n < count:
                errors.append(f"{key}: allowlist entry `{text}` is stale ({n} hits, count {count}): lower or remove it")
    return errors


def lint(root: Path = REPO_ROOT) -> int:
    found = scan(root)
    errors = judge(found)
    allowed = sum(c for entries in ALLOWLIST.values() for c, _ in entries.values())
    print(f"cut-before-redact: {len(found)} cut(s) of exception text in {SCOPE} ({allowed} allowlisted with reasons)")
    if errors:
        print("cut-before-redact FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("cut-before-redact: clean")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(lint())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — an unexpected failure is exit 2, never a silent pass
        print(f"cut-before-redact: unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
