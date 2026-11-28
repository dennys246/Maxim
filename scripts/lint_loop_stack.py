#!/usr/bin/env python3
"""Every agent-loop entry point goes through ``bootstrap.run_loop_stack`` (docs/plans/loop_stack_builder.md).

``run_loop_stack`` takes every stack component as a REQUIRED keyword, so a site that forgets one gets a TypeError
instead of a silently less capable loop. This lint makes it the only way in. Over every ``.py`` under ``src/maxim/``
and ``scripts/`` (``tests/`` and ``docs/`` excluded on purpose: tests drive the loop with fakes, docs are examples):

**Counted sites.** Per innermost enclosing def, with ``lint_loop_mode_reads``' visitor and qualname convention
(``.<locals>.``; a lambda counts toward its enclosing def; decorators and defaults in the outer scope), a site is any
LOAD of a guarded name (``NAMES``): a Name (including an alias bound by ``import … as``) or an Attribute, plus any
non-docstring string constant equal to one (``getattr``, ``importlib``, ``_LAZY`` keys, ``__all__``). A load counts,
not only a call, so ``Thread(target=…)``, ``partial(…)`` and a stored reference count. Exempt: the sanctioned caller
(``SANCTIONED_QUALNAME``), ``run_agentic_loop``'s own call to ``_run_agentic_loop_body`` in ``agent_loop.py``, and in
this file only the string constants inside the module-level ``NAMES`` / ``SANCTIONED_QUALNAME`` / ``BUILDER_NAME``
assignments. Every counted site needs an entry in ``scripts/loop_stack_ledger.json``.

**The builder is called directly, without a splat.** Any load of ``run_loop_stack`` (``BUILDER_NAME``, same
definition of a load) is a finding unless it is the callee of a Call with no ``*``/``**`` argument. A splat-free
direct call through an alias or an attribute is allowed. So every site writes out each STACK keyword, and a reviewer
sees every ``None`` at the call.

**The ledger** (``{file, qualname, count, ref, reason, permanent, moved_from?}``, keyed by ``file::qualname``):

1. measured equality: each entry's count equals its measured count; a measured 0 is an orphan and fails;
2. unlisted fails: a counted site with no entry fails;
3. shrink only: HEAD's entries are a subset of the merge-base's, except valid moves (4); no count rises above its
   base pin, and ``permanent`` never changes;
4. moves: an entry not at base needs ``moved_from: {file, qualname}`` naming a base entry absent at HEAD, with a count
   no larger than the source's and the source's ``permanent``; each source is used once (no splits or merges: those
   need a reviewed change to this lint);
5. bootstrap: a ledger absent at base passes rules 3-4 only when this branch adds it (diff status ``A``);
6. a readable HEAD ledger: missing, unparsable or malformed exits 2.

Rules 1, 2 and 6 read HEAD and always run. Rules 3-5 need the base; with no base ref they skip on a local run and
fail on a pull request or push (``_lint_git.must_not_skip``). A broken scope exits 2.

**Accepted blind spots:** a computed string (``"run_" + x``), ``exec``/``eval``, and a new caller of a wrapper that is
already allow-listed (its entry's ``reason`` names it). **Residual:** the same PR can widen this file's constants and
their pins in ``tests/unit/test_lint_loop_stack.py`` together; review is the guard.

Positive control: ``tests/unit/test_lint_loop_stack.py``.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_allowance import ref_ok  # noqa: E402
from _lint_git import GitUnavailable, base_ref, git, must_not_skip, show  # noqa: E402
from lint_loop_mode_reads import Module, ScopeError, _is_docstring, _parse  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOTS = ("src/maxim", "scripts")
LEDGER = "scripts/loop_stack_ledger.json"
AGENT_LOOP = "src/maxim/runtime/agent_loop.py"
BOOTSTRAP = "src/maxim/runtime/bootstrap.py"
SELF = "scripts/lint_loop_stack.py"

# The lint's own name constants: the only strings in this file it does not count (module-level assignments only).
NAMES = ("run_agentic_loop", "_run_agentic_loop_body", "run_agent_loop")
BUILDER_NAME = "run_loop_stack"
SANCTIONED_QUALNAME = "src/maxim/runtime/bootstrap.py::run_loop_stack"
_SELF_CONSTANTS = frozenset({"NAMES", "BUILDER_NAME", "SANCTIONED_QUALNAME"})

_ENTRY_KEYS = frozenset({"file", "qualname", "count", "ref", "reason", "permanent"})


@dataclass(frozen=True)
class Site:
    file: str  # repo-relative
    qualname: str
    line: int
    what: str

    @property
    def key(self) -> str:
        return f"{self.file}::{self.qualname}"


class LedgerError(RuntimeError):
    """The HEAD ledger cannot be read or is malformed -- exit 2, never a pass."""


# ── scanning ─────────────────────────────────────────────────────────────────────────────────────────────


def scan_files(root: Path) -> list[str]:
    files: list[str] = []
    for top in ROOTS:
        base = root / top
        if not base.is_dir():
            raise ScopeError(f"{top}/ is missing")
        files += sorted(p.relative_to(root).as_posix() for p in base.rglob("*.py"))
    if not files:
        raise ScopeError("no .py files in scope")
    for required in (AGENT_LOOP, BOOTSTRAP):
        if required not in files:
            raise ScopeError(f"{required} is missing from the scan")
    return files


def _parse_rel(root: Path, rel: str) -> Module:
    top, _, inner = rel.partition("/")
    if top == "src":
        return _parse(root, rel.removeprefix("src/maxim/"), "src/maxim")
    return _parse(root, inner, top)


def _aliases(tree: ast.Module, names: tuple[str, ...]) -> dict[str, str]:
    """``import … as`` bindings of a guarded name, module-wide: alias -> the name it binds."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                leaf = a.name.rsplit(".", 1)[-1]
                if leaf in names and a.asname:
                    out[a.asname] = leaf
    return out


def _self_constant_nodes(tree: ast.Module) -> set[int]:
    """ids of the string constants inside this lint's module-level name-constant assignments."""
    out: set[int] = set()
    for stmt in tree.body:
        targets: list[ast.expr] = []
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id in _SELF_CONSTANTS for t in targets):
            out |= {id(n) for n in ast.walk(value) if isinstance(n, ast.Constant)}
    return out


def _loaded_name(node: ast.AST, aliases: dict[str, str], names: tuple[str, ...]) -> str | None:
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        if node.id in names:
            return node.id
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in names:
        return node.attr
    return None


def _splat_free_callee(m: Module, node: ast.AST) -> bool:
    call = m.parents.get(node)
    return (
        isinstance(call, ast.Call)
        and call.func is node
        and not any(k.arg is None for k in call.keywords)
        and not any(isinstance(a, ast.Starred) for a in call.args)
    )


def scan_module(rel: str, m: Module) -> tuple[list[Site], list[Site]]:
    """(counted sites of a guarded name, builder findings) in one module."""
    loop_aliases = _aliases(m.tree, NAMES)
    builder_aliases = _aliases(m.tree, (BUILDER_NAME,))
    exempt_consts = _self_constant_nodes(m.tree) if rel == SELF else set()
    sites: list[Site] = []
    builder: list[Site] = []
    for node in ast.walk(m.tree):
        qual = m.quals.get(node, "<module>")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in exempt_consts or _is_docstring(m, node):
                continue
            if node.value in NAMES:
                sites.append(Site(rel, qual, node.lineno, f"the string {node.value!r}"))
            elif node.value == BUILDER_NAME:
                builder.append(Site(rel, qual, node.lineno, f"the string {BUILDER_NAME!r} (a dynamic lookup)"))
            continue
        name = _loaded_name(node, loop_aliases, NAMES)
        if name is not None:
            key = f"{rel}::{qual}"
            if key == SANCTIONED_QUALNAME:
                continue
            if rel == AGENT_LOOP and qual == NAMES[0] and name == NAMES[1]:
                continue  # run_agentic_loop's own call to its body
            sites.append(Site(rel, qual, getattr(node, "lineno", 0), f"a load of {name}"))
            continue
        if _loaded_name(node, builder_aliases, (BUILDER_NAME,)) is not None and not _splat_free_callee(m, node):
            builder.append(
                Site(rel, qual, getattr(node, "lineno", 0), f"{BUILDER_NAME} used other than as a splat-free callee")
            )
    return sites, builder


def measure(root: Path) -> tuple[list[Site], list[Site]]:
    sites: list[Site] = []
    builder: list[Site] = []
    files = scan_files(root)
    for rel in files:
        s, b = scan_module(rel, _parse_rel(root, rel))
        sites += s
        builder += b
    bootstrap = _parse_rel(root, BOOTSTRAP)
    if not any(
        isinstance(n, ast.FunctionDef) and n.name == BUILDER_NAME for n in bootstrap.tree.body
    ):  # the sanctioned caller must exist, or every migrated site is calling nothing
        raise ScopeError(f"{SANCTIONED_QUALNAME} is missing")
    return sites, builder


# ── the ledger ───────────────────────────────────────────────────────────────────────────────────────────


def parse_ledger(text: str, *, where: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(text)
    except ValueError as e:
        raise LedgerError(f"{where}: not JSON ({e})") from e
    if not isinstance(data, list):
        raise LedgerError(f"{where}: the ledger is a JSON list of entries")
    seen: set[str] = set()
    for i, e in enumerate(data):
        if not isinstance(e, dict):
            raise LedgerError(f"{where}[{i}]: an entry is an object")
        keys = set(e)
        if not _ENTRY_KEYS <= keys or not keys <= _ENTRY_KEYS | {"moved_from"}:
            raise LedgerError(f"{where}[{i}]: keys must be {sorted(_ENTRY_KEYS)} plus an optional moved_from")
        if not (isinstance(e["file"], str) and isinstance(e["qualname"], str) and e["file"] and e["qualname"]):
            raise LedgerError(f"{where}[{i}]: file and qualname are non-empty strings")
        if not (isinstance(e["count"], int) and not isinstance(e["count"], bool) and e["count"] > 0):
            raise LedgerError(f"{where}[{i}]: count must be a positive int")
        if not isinstance(e["permanent"], bool):
            raise LedgerError(f"{where}[{i}]: permanent must be true or false")
        if not ref_ok(e["ref"]):
            raise LedgerError(f"{where}[{i}]: ref {e['ref']!r} is not #NNN or a github.com PR/issue URL")
        if not (isinstance(e["reason"], str) and e["reason"].strip()):
            raise LedgerError(f"{where}[{i}]: reason is empty")
        mf = e.get("moved_from")
        if mf is not None and not (
            isinstance(mf, dict)
            and set(mf) == {"file", "qualname"}
            and all(isinstance(v, str) and v for v in mf.values())
        ):
            raise LedgerError(f"{where}[{i}]: moved_from must be {{file, qualname}}")
        key = f"{e['file']}::{e['qualname']}"
        if key in seen:
            raise LedgerError(f"{where}[{i}]: duplicate entry for {key}")
        seen.add(key)
    return data


def _key(e: dict[str, Any]) -> str:
    return f"{e['file']}::{e['qualname']}"


def head_rules(ledger: list[dict[str, Any]], sites: list[Site]) -> tuple[list[str], list[str]]:
    """Rules 1 and 2: (failures, report lines)."""
    failures: list[str] = []
    report: list[str] = []
    measured = Counter(s.key for s in sites)
    listed = {_key(e): e for e in ledger}
    report.append(f"ledger: {len(ledger)} entries")
    for key, e in listed.items():
        n = measured.get(key, 0)
        tag = "permanent, non-shrinking" if e["permanent"] else "shrinks when the site migrates"
        report.append(f"  {key} = {n} (pin {e['count']}, {tag}, {e['ref']}): {e['reason']}")
        if n == 0:
            failures.append(f"{key}: listed but measures 0 (orphan: a migrated or moved site deletes its entry)")
        elif n != e["count"]:
            failures.append(f"{key}: measures {n}, the ledger pins {e['count']} (strict equality)")
    for s in sites:
        if s.key not in listed:
            failures.append(
                f"{s.file}:{s.line}: {s.what} in {s.qualname}: an unlisted loop entry point -- call "
                f"bootstrap.{BUILDER_NAME}(...) instead"
            )
    return failures, report


def base_rules(base: list[dict[str, Any]], head: list[dict[str, Any]]) -> list[str]:
    """Rules 3 and 4 against the merge-base ledger."""
    failures: list[str] = []
    base_by = {_key(e): e for e in base}
    head_keys = {_key(e) for e in head}
    sources: Counter[str] = Counter()
    for e in head:
        key = _key(e)
        b = base_by.get(key)
        if b is not None:
            if e["count"] > b["count"]:
                failures.append(f"{key}: count {e['count']} rises above its base pin {b['count']} (shrink only)")
            if e["permanent"] != b["permanent"]:
                failures.append(f"{key}: permanent changed from {b['permanent']} (it never changes)")
            continue
        mf = e.get("moved_from")
        if mf is None:
            failures.append(f"{key}: a new entry with no moved_from (shrink only: migrate the site instead)")
            continue
        src = f"{mf['file']}::{mf['qualname']}"
        sources[src] += 1
        s = base_by.get(src)
        if s is None:
            failures.append(f"{key}: moved_from {src} is not a base entry")
            continue
        if src in head_keys:
            failures.append(f"{key}: moved_from {src} is still present (a move removes its source)")
        if e["count"] > s["count"]:
            failures.append(f"{key}: moved count {e['count']} is above its source's {s['count']}")
        if e["permanent"] != s["permanent"]:
            failures.append(f"{key}: a move keeps its source's permanent ({s['permanent']})")
    for src, n in sources.items():
        if n > 1:
            failures.append(f"{src}: named as moved_from by {n} entries (no splits: change this lint, reviewed)")
    return failures


def _added_on_branch(root: Path, base: str, rel: str) -> bool:
    out = git(root, "diff", "--name-status", base, "HEAD", "--", rel)
    return any(line.split("\t")[0] == "A" for line in out.splitlines())


def check(root: Path) -> tuple[list[str], list[str]]:
    """(failures, report). Raises ScopeError / LedgerError for exit 2."""
    sites, builder = measure(root)
    path = root / LEDGER
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise LedgerError(f"{LEDGER}: cannot read ({e})") from e
    head = parse_ledger(text, where=LEDGER)
    failures, report = head_rules(head, sites)
    failures += [f"{b.file}:{b.line}: {b.what} in {b.qualname}" for b in builder]
    try:
        base = base_ref(root)
        base_text = show(root, base, LEDGER)
    except GitUnavailable as e:
        if must_not_skip(f"no base ref: {e}"):
            failures.append(f"rules 3-5 cannot run: no base ref ({e})")
        else:
            report.append(f"rules 3-5 skipped locally: no base ref ({e})")
        return failures, report
    if not base_text:
        if _added_on_branch(root, base, LEDGER):
            report.append(f"{LEDGER} is added by this branch: rules 3-4 have no base to compare (rule 5)")
        else:
            failures.append(f"{LEDGER} is absent at the merge-base but not added by this branch (rule 5)")
        return failures, report
    failures += base_rules(parse_ledger(base_text, where=f"{LEDGER}@base"), head)
    return failures, report


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        failures, report = check(REPO_ROOT)
    except (ScopeError, LedgerError) as e:
        print(f"lint_loop_stack: ERROR: {e}", file=sys.stderr)
        return 2
    print("\n".join(report))
    if failures:
        print(f"\nlint_loop_stack: {len(failures)} finding(s) (see this script's docstring):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("lint_loop_stack: OK -- every loop entry point is run_loop_stack or a ledgered site")
    return 0


if __name__ == "__main__":
    sys.exit(main())
