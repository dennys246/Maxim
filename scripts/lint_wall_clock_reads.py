#!/usr/bin/env python3
"""Lint (M72, extended): no wall-clock or calendar read in a learning or prompt path outside the allowlist.

World clock A-pre, D-WC4 (``docs/plans/world_clock_a_pre.md``; design and adversarial pass:
``docs/plans/world_clock_a_pre_guard_note.md``). An embedder (Overdrive) steps the agent in its own world time, so a
``time.time()`` that reaches learning, or a ``datetime.now()`` that reaches a prompt, leaks the host's wall into the
agent. This lint refuses every such reference in M72's two scopes (``scripts/_clock_refs.py``: machinery and prompt)
unless ``scripts/wall_clock_allowlist.json`` places it.

**Absolute from birth** (owner, PR 1b, 2026-11-26). Every in-scope reference has an entry, so the guard note's
migration ratchet would have run with every file already at zero; there is none. A reference without an entry fails.

**What counts.** References, not only calls (``field(default_factory=time.time)`` binds at class definition), resolved
through local import aliases by the shared resolver (``scripts/_clock_refs.py``, also the census's), so the two read a
reference the same way. The lint never reads the census's verdicts. Counted kinds: ``wall`` (``time.time``,
``time_ns``), ``monotonic`` (``monotonic``, ``perf_counter`` and their ``_ns``) and ``calendar`` (``localtime``,
``gmtime``, ``strftime``, ``ctime``, ``datetime.now``/``utcnow``/``today``/``fromtimestamp``, ``date.today``). ``uuid1``
and ``uuid4`` are printed as warnings, never failures (D-WC1 defers ids). Draws from a global random generator are
OD-A7's and are not this lint's.

**The allowlist.** One entry per ``file::function`` (the census's qualified name: ``Class.method``, a class name for a
class-body field default, ``<module>`` at module level), with:
- ``category``: ``process``, ``scheduling``, ``motor-io``, ``real-world-io``, ``cost`` or ``pending:<where>`` (a site
  the plan moves; ``<where>`` names the PR or line that moves it, and the entry leaves in that PR). The definitions
  live in ``docs/plans/experience_clock_decay.md`` §(g);
- ``reads``: the exact number of counted references in that function. More fails (a read was added and nobody
  placed it, the blind spot a key alone leaves open); fewer fails too, so an entry never outlives its reads (a
  function with none left is a stale entry);
- ``reason``: why the read stays on the wall, or where it goes.

Fails closed on: a name bound to two clock targets in one module, or a star import from a clock module (the
resolver's ``conflicts``); an entry for a file outside the scopes or not tracked; a duplicate entry; an unknown
category; a missing reason.

Stated blind spots (each also in the guard note): dynamic access (``getattr(time, "time")``, ``importlib``); a
callable passed in; a transitive read through a helper outside the scopes; a local variable shadowing an imported
name; a module outside both scope lists (M99 derives the prompt scope). G-long, G-render, G-tz and G-scn cover what
their reach allows.

Usage: ``python scripts/lint_wall_clock_reads.py [--root DIR]``. Scans files tracked by git under
``<root>/src/maxim``. Exits 0 clean, 1 violations, 2 a file failed to parse, git could not list the tree, or the
allowlist is unreadable (never a silent pass).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _clock_refs import SRC, Bindings, kind_of, module_bindings, rel_to_src, resolve, scope_of  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_REL = "scripts/wall_clock_allowlist.json"
COUNTED = frozenset({"wall", "monotonic", "calendar"})
CATEGORIES = frozenset({"process", "scheduling", "motor-io", "real-world-io", "cost"})
_PENDING = re.compile(r"^pending:\S.*$")
#: The collaborators' own module reads the wall by design (the host defaults).
EXEMPT = frozenset({"time/agent_time.py"})


class LintError(Exception):
    """The lint could not judge the tree (exit 2), as opposed to finding violations (exit 1)."""


@dataclass(frozen=True)
class Ref:
    """One counted reference. Lint output only: never persisted, so CC3's forward-compat rule does not apply."""

    rel: str
    line: int
    function: str
    name: str
    kind: str


class _Scanner(ast.NodeVisitor):
    """Each banned reference with its enclosing qualified name.

    A chain counts once without marking: no counted name is a prefix of another (``datetime.datetime`` is not banned,
    ``datetime.datetime.now`` is), so only one link of ``datetime.datetime.now()`` resolves to a counted kind.
    ``tests/unit/test_lint_wall_clock_reads.py::test_no_counted_name_nests_in_another`` holds that premise.
    """

    def __init__(self, rel: str, bindings: Bindings) -> None:
        self.rel = rel
        self.bindings = bindings
        self.stack: list[str] = []
        self.refs: list[Ref] = []

    def _scoped(self, node: ast.AST, name: str) -> None:
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node, node.name)

    def _ref(self, node: ast.Name | ast.Attribute) -> None:
        name = resolve(node, self.bindings)
        kind = kind_of(name) if name else None
        if name is None or kind is None:
            return
        function = ".".join(self.stack) if self.stack else "<module>"
        self.refs.append(Ref(self.rel, node.lineno, function, name, kind))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._ref(node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._ref(node)


def scan_text(text: str, rel: str) -> tuple[list[Ref], list[str]]:
    """(references, conflicts) for one module's source. Raises :class:`LintError` on a syntax error."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise LintError(f"{rel}: does not parse ({exc.msg}, line {exc.lineno})") from exc
    bindings = module_bindings(tree)
    scanner = _Scanner(rel, bindings)
    scanner.visit(tree)
    return scanner.refs, [f"{rel} {c}" for c in bindings.conflicts]


def tracked_in_scope(root: Path) -> list[str]:
    """Tracked ``src/maxim`` files (relative to ``src/maxim``) inside either M72 scope."""
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", SRC],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LintError(f"git ls-files failed: {exc}") from exc
    if r.returncode != 0:
        raise LintError(f"git ls-files failed: {r.stderr.strip()}")
    paths = [p for p in r.stdout.split("\0") if p.endswith(".py")]
    if not paths:
        raise LintError(f"git lists no tracked .py file under {root / SRC}: refusing to pass an empty scan")
    return sorted(rel_to_src(p) for p in paths if scope_of(p) != "-" and rel_to_src(p) not in EXEMPT)


@dataclass(frozen=True)
class Entry:
    """One allowlist row. Read from the committed JSON only; never written by code."""

    site: str
    category: str
    reads: int
    reason: str


def load_allowlist(path: Path) -> tuple[dict[str, Entry], list[str]]:
    """(entries by site, shape violations). Raises :class:`LintError` when the file is missing or not JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LintError(f"{path}: unreadable allowlist ({exc})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise LintError(f"{path}: expected an object with an 'entries' list")
    entries: dict[str, Entry] = {}
    problems: list[str] = []
    for i, raw in enumerate(data["entries"]):
        if not isinstance(raw, dict) or set(raw) != {"site", "category", "reads", "reason"}:
            problems.append(f"entry {i}: needs exactly the keys site, category, reads, reason ({raw!r})")
            continue
        site, category, reads, reason = raw["site"], raw["category"], raw["reads"], raw["reason"]
        if not isinstance(site, str) or "::" not in site:
            problems.append(f"entry {i}: site must be 'file::function', got {site!r}")
            continue
        if site in entries:
            problems.append(f"{site}: listed twice")
            continue
        if not isinstance(category, str) or not (category in CATEGORIES or _PENDING.match(category)):
            problems.append(
                f"{site}: unknown category {category!r} (§(g): {', '.join(sorted(CATEGORIES))}, pending:<where>)"
            )
        if not isinstance(reads, int) or isinstance(reads, bool) or reads < 1:
            problems.append(f"{site}: reads must be a positive integer, got {reads!r}")
        if not isinstance(reason, str) or not reason.strip():
            problems.append(f"{site}: no reason")
        entries[site] = Entry(site, str(category), reads if isinstance(reads, int) else 0, str(reason))
    return entries, problems


@dataclass
class Result:
    """What one run found. Lint output only."""

    refs: list[Ref]
    violations: list[str]
    warnings: list[str]
    entries: dict[str, Entry]


def check(root: Path, allowlist: Path | None = None) -> Result:
    files = tracked_in_scope(root)
    entries, violations = load_allowlist(allowlist or root / ALLOWLIST_REL)
    counted: list[Ref] = []
    warnings: list[str] = []
    for rel in files:
        path = root / SRC / rel
        refs, conflicts = scan_text(path.read_text(encoding="utf-8"), rel)
        violations.extend(f"{c}: resolve it to one target (the lint fails closed, not guessing)" for c in conflicts)
        for ref in refs:
            if ref.kind in COUNTED:
                counted.append(ref)
            elif ref.kind == "id":
                warnings.append(f"{rel}:{ref.line} {ref.function}: {ref.name} (ids are deferred, D-WC1; not a failure)")
    by_site: dict[str, list[Ref]] = {}
    for ref in counted:
        by_site.setdefault(f"{ref.rel}::{ref.function}", []).append(ref)
    in_scope = set(files)
    for site, refs in sorted(by_site.items()):
        entry = entries.get(site)
        lines = ", ".join(f"L{r.line} {r.name}" for r in refs)
        if entry is None:
            violations.append(
                f"{site}: {len(refs)} wall-clock read(s) with no allowlist entry ({lines}). Read the collaborator "
                f"instead, or add an entry to {ALLOWLIST_REL} with its §(g) category"
            )
        elif entry.reads != len(refs):
            verb = "rose" if len(refs) > entry.reads else "fell"
            violations.append(
                f"{site}: reads {verb} {entry.reads} → {len(refs)} ({lines}); place the new read or update the "
                f"entry's 'reads'"
            )
    for site in sorted(set(entries) - set(by_site)):
        rel = site.split("::", 1)[0]
        where = "a file outside M72's scopes or not tracked" if rel not in in_scope else "a function with no read"
        violations.append(f"{site}: stale entry ({where}); delete it")
    return Result(counted, violations, warnings, entries)


def render(result: Result) -> str:
    out: list[str] = []
    for w in result.warnings:
        out.append(f"WARNING: {w}")
    placed: Counter[str] = Counter()
    for e in result.entries.values():
        placed["pending" if e.category.startswith("pending:") else e.category] += e.reads
    summary = ", ".join(f"{k} {v}" for k, v in sorted(placed.items()))
    out.append(f"wall-clock reads in M72's scopes: {len(result.refs)} (allowlisted by category: {summary or 'none'})")
    for v in result.violations:
        out.append(f"ERROR: {v}")
    out.append("lint_wall_clock_reads: " + (f"{len(result.violations)} violation(s)" if result.violations else "OK"))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        result = check(args.root)
    except LintError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(render(result))
    return 1 if result.violations else 0


if __name__ == "__main__":
    sys.exit(main())
