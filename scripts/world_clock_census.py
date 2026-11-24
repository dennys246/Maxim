#!/usr/bin/env python3
"""World clock A-pre census: every time, calendar, id and randomness site in ``src/maxim/`` (PR 0).

``docs/plans/world_clock_a_pre.md`` says the census, not the plan's illustrative lists, is authoritative for
where each site goes. This script prints, from the AST alone (it imports nothing from ``maxim``):

1. **Sites.** Every reference (not only calls) to a banned clock, calendar, id or randomness name, resolved
   through local import aliases, with its ``file::function``, kind, M72 scope and placement.
2. **Stamp–reader pairs (I4).** The writers of the four stamp classes' ``timestamp`` and every function that
   subtracts something from a ``.timestamp``, with the clock the other operand reads. Readers are untyped:
   AST cannot tell an ``EpisodicMemory`` from any other object with a ``timestamp``, so every subtraction is
   listed and a reviewer reads the list.
3. **TemporalSignature and SCN callers,** with the clock their arguments read (input) and what the result
   feeds (output use).

The output is for review. Neither the M72 lint nor G-long reads its verdicts; they rescan. A site inside an M72
scope that no rule places prints as ``UNPLACED``, so a reviewer sees it rather than a guessed placement; PR 1
gives each one an allowlist entry or a move. A site outside both scopes prints as ``outside M72's scopes``.

Usage: ``python scripts/world_clock_census.py [--json] [--root DIR]``. The scan covers files tracked by git
under ``<root>/src/maxim`` (``git ls-files``), so untracked scratch files and fixtures never count.

Exits: 0 printed; 2 a file failed to parse, or git could not list the tree (never a silent pass).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = "src/maxim"

WALL = frozenset({"time.time", "time.time_ns"})
MONOTONIC = frozenset({"time.monotonic", "time.monotonic_ns", "time.perf_counter", "time.perf_counter_ns"})
CALENDAR = frozenset(
    {
        "time.localtime",
        "time.gmtime",
        "time.strftime",
        "time.ctime",
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
        "datetime.datetime.today",
        "datetime.datetime.fromtimestamp",
        "datetime.datetime.utcfromtimestamp",
        "datetime.date.today",
        "datetime.date.fromtimestamp",
    }
)
IDS = frozenset({"uuid.uuid1", "uuid.uuid4"})
_RANDOM_PREFIXES = ("random.", "numpy.random.")
# Names in the random modules that are types or constants, not draws from the process-global generator.
_RANDOM_NON_DRAWS = frozenset({"random.Random", "random.SystemRandom", "numpy.random.Generator"})

STAMP_CLASSES = ("EpisodicMemory", "Percept", "SensorReading", "PainSignal")
STAMP_ATTR = "timestamp"

# M72's two scopes (world_clock_a_pre_guard_note.md §1). A path ending in "/" is a directory prefix.
MACHINERY_SCOPE = (
    "decisions/",
    "memory/",
    "embodiment/body.py",
    "integration/memory_hub.py",
    "agents/memory_agent.py",
    "agents/context_pool.py",
    "attention/",
    "salience/",
    "default_network/network.py",
    "bridges/salience_bridge.py",
    "bridges/planning_bridge.py",
    "bridges/spatial_bridge.py",
    "similarity/signature.py",
    "embodiment/spec.py",
    "proprioception/pain.py",
    "proprioception/pain_bus.py",
    "reactions/bus.py",
    "models/language/cost_tracker.py",
)
PROMPT_SCOPE = (
    "agents/prompt_builder.py",
    "agents/context_pool.py",
    "agents/llm_fallback.py",
    "agents/exec_agent.py",
    "agents/exec_prompts.py",
    "agents/statistician_agent.py",
    "retrieval/temporal_signal.py",
    "tools/http_fetch.py",
    "tools/introspection.py",
    "simulation/introspection.py",
)

# Placement rules from world_clock_a_pre.md (§Site families, §Placements, §What stays on the wall), first match
# wins: (kinds or None for any, path prefix, function regex or None for any, placement). Kind rules come first
# because ids and randomness are OD-A7's wherever they sit (D-WC1).
_ANY = None
PLACEMENT_RULES: tuple[tuple[frozenset[str] | None, str, str | None, str], ...] = (
    (frozenset({"id", "random"}), "", _ANY, "OD-A7, after the 1.4.0 cut (D-WC1)"),
    (_ANY, "mesh/", _ANY, "out of scope: off the embedded path"),
    (_ANY, "utils/web_cache.py", _ANY, "out of scope: off the embedded path"),
    (_ANY, "utils/output_watcher.py", _ANY, "out of scope: off the embedded path"),
    (_ANY, "runtime/worker_pool.py", _ANY, "out of scope: off the embedded path"),
    (_ANY, "runtime/agent_loop.py", _ANY, "A-post: fenced file (OD-A8)"),
    (_ANY, "simulation/orchestrator.py", _ANY, "A-post: fenced file (OD-A8)"),
    (_ANY, "runtime/tool_dispatch.py", r"^execute_and_learn$", "A-post with B2: OD-A4's tool duration (fenced)"),
    (_ANY, "runtime/tool_dispatch.py", _ANY, "A-post: fenced file (OD-A8)"),
    (_ANY, "agents/exec_agent.py", r"_build_llm_context$", "allow: real-world-io (workspace mtimes)"),
    (_ANY, "agents/exec_agent.py", _ANY, "A-post: hot file (D-WC7)"),
    (_ANY, "models/language/router.py", _ANY, "A-post: hot file (D-WC7)"),
    (_ANY, "cli.py", _ANY, "A-post: hot file (D-WC7)"),
    (_ANY, "runtime/loop_setup.py", _ANY, "A-post: hot file (D-WC7)"),
    (_ANY, "agents/prompt_builder.py", r"(^|\.)(build_prompt|_build_engage_prompt)$", "PR 2: the port (OD-A3)"),
    (_ANY, "agents/context_pool.py", r"(^|\.)(_entries_to_text|get_context_text)$", "PR 2: perceive (OD-A3)"),
    (_ANY, "tools/introspection.py", r"^_format_episodic_memory$", "PR 2: omitted under NO_CALENDAR"),
    (_ANY, "tools/introspection.py", r"^SceneSummaryTool\.", "PR 2: omitted under NO_CALENDAR"),
    (_ANY, "tools/introspection.py", r"^TemporalPatternsTool\.", "PR 3: SCN-derived render (D-WC2)"),
    (
        _ANY,
        "agents/statistician_agent.py",
        r"(_assess_with_temporal_context|summarize_for_context)$",
        "PR 3: [TEMPORAL] (D-WC2)",
    ),
    (_ANY, "simulation/introspection.py", _ANY, "PR 3: current_phase (D-WC2)"),
    (_ANY, "retrieval/temporal_signal.py", _ANY, "PR 3: relative dates (D-WC2)"),
    (_ANY, "memory/hippocampus.py", r"(^|\.)store_observation$", "PR 4: the dedup window (closed pair)"),
    (
        _ANY,
        "memory/hippocampus.py",
        r"(^|\.)(_build_capture_record|get|_touch_internal|consolidate[a-z_]*)$",
        "A-post: OD-A4 stamp half (D-WC6)",
    ),
    (_ANY, "proprioception/pain_bus.py", r"(^|\.)publish$", "PR 5 (1.3.6): refractory (D-WC3)"),
    (_ANY, "reactions/bus.py", r"(^|\.)publish$", "PR 5 (1.3.6): refractory (D-WC3)"),
    (_ANY, "proprioception/pain.py", r"(^|\.)(_emit_pain|record_tool_error)$", "PR 5 (1.3.6): PainDetector pair"),
    (_ANY, "proprioception/pain.py", r"(^|\.)_can_emit_pain$", "allow: motor-io (motion pain cooldown)"),
    (_ANY, "models/language/cost_tracker.py", _ANY, "allow: cost (D-WC5; #1769 in PR 1)"),
    (_ANY, "tools/http_fetch.py", r"^HttpFetchTool\.execute$", "allow: real-world-io (accessed_at)"),
    (_ANY, "embodiment/body.py", _ANY, "A-post (OD-A8) with #1361 S4"),
    (_ANY, "embodiment/spec.py", r"^SpecSensor\.read$", "A-post: OD-A4 stamp half (SensorReading)"),
    (_ANY, "similarity/signature.py", _ANY, "with the stamp half (SituationSignature binning)"),
    (_ANY, "integration/memory_hub.py", r"_register_in_scn$", "with the stamp half (SCN registration)"),
    (_ANY, "memory/concept_extractor.py", _ANY, "with the stamp half (SCN registration)"),
    (_ANY, "time/scn.py", _ANY, "with the stamp half (SCN registration)"),
    (_ANY, "salience/", _ANY, "A-post with OD-A9"),
    (_ANY, "attention/gaze_history.py", _ANY, "A-post with OD-A9"),
    (_ANY, "attention/salience_map.py", _ANY, "A-post with OD-A9"),
    (_ANY, "default_network/network.py", _ANY, "A-post with OD-A9"),
    (_ANY, "decisions/nac.py", _ANY, "A-post, after #1361 S3b (NAc stamps and anchors)"),
    (_ANY, "motion/", _ANY, "allow: motor-io"),
    (_ANY, "hardware/", _ANY, "allow: motor-io"),
)


@dataclass(frozen=True)
class Site:
    """One banned reference. SHAPE-FROZEN at 1.0 (CC3): census output only, never persisted or sent."""

    path: str
    line: int
    function: str
    name: str
    kind: str
    call: bool
    scope: str
    placement: str


@dataclass(frozen=True)
class StampSubtraction:
    """A ``.timestamp`` on one side of a subtraction. SHAPE-FROZEN at 1.0 (CC3): census output only."""

    path: str
    line: int
    function: str
    other_clock: str
    placement: str


@dataclass(frozen=True)
class StampWriter:
    """A stamp class built with an explicit ``timestamp=``. SHAPE-FROZEN at 1.0 (CC3): census output only."""

    path: str
    line: int
    function: str
    stamp_class: str
    input_clock: str


@dataclass(frozen=True)
class TemporalCaller:
    """A TemporalSignature or SCN call. SHAPE-FROZEN at 1.0 (CC3): census output only."""

    path: str
    line: int
    function: str
    callee: str
    input_clock: str
    output_use: str
    placement: str


@dataclass
class FileCensus:
    sites: list[Site] = field(default_factory=list)
    subtractions: list[StampSubtraction] = field(default_factory=list)
    writers: list[StampWriter] = field(default_factory=list)
    temporal: list[TemporalCaller] = field(default_factory=list)


# ── alias resolution ─────────────────────────────────────────────────────────


def module_bindings(tree: ast.AST) -> dict[str, str]:
    """Local name → canonical dotted name, for every import in *tree* (any nesting depth).

    Handles ``import time``, ``import time as _t``, ``import os.path``, ``from time import time as now``,
    ``from datetime import datetime``, ``import datetime as dt`` and ``from datetime import datetime as DT``.
    A relative import (``from . import x``) binds nothing here: it cannot name a stdlib clock. Bindings are
    module-wide: a local variable that shadows an imported name is not tracked (stated blind spot).
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bindings[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    bindings[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bindings


def resolve(node: ast.AST, bindings: dict[str, str]) -> str | None:
    """The canonical dotted name of a ``Name``/``Attribute`` chain rooted at an imported name, else None."""
    attrs: list[str] = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in bindings:
        return None
    return ".".join([bindings[node.id], *reversed(attrs)])


def kind_of(name: str) -> str | None:
    if name in WALL:
        return "wall"
    if name in MONOTONIC:
        return "monotonic"
    if name in CALENDAR:
        return "calendar"
    if name in IDS:
        return "id"
    if name.startswith(_RANDOM_PREFIXES) and name not in _RANDOM_NON_DRAWS:
        # ``import random`` itself resolves to "random", which has no dot after the prefix: not a reference.
        return "random"
    return None


# ── scope and placement ──────────────────────────────────────────────────────


def _rel_to_src(path: str) -> str:
    return path[len(SRC) + 1 :] if path.startswith(SRC + "/") else path


def _in(rel: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel.startswith(p) if p.endswith("/") else rel == p for p in prefixes)


def scope_of(path: str) -> str:
    rel = _rel_to_src(path)
    scopes = [
        name for name, prefixes in (("machinery", MACHINERY_SCOPE), ("prompt", PROMPT_SCOPE)) if _in(rel, prefixes)
    ]
    return "+".join(scopes) if scopes else "-"


def placement_of(path: str, function: str, kind: str | None) -> str:
    rel = _rel_to_src(path)
    for kinds, prefix, func_re, placement in PLACEMENT_RULES:
        if kinds is not None and kind not in kinds:
            continue
        if prefix and not (rel.startswith(prefix) if prefix.endswith("/") else rel == prefix):
            continue
        if func_re is not None and not re.search(func_re, function):
            continue
        return placement
    # A site in neither M72 scope is not A-pre's to place; one inside a scope that no rule names is a finding.
    return "UNPLACED" if scope_of(path) != "-" else "outside M72's scopes"


# ── the scan ─────────────────────────────────────────────────────────────────


class _Scanner(ast.NodeVisitor):
    def __init__(self, path: str, bindings: dict[str, str]) -> None:
        self.path = path
        self.bindings = bindings
        self.stack: list[str] = []
        self.out = FileCensus()
        self._value_of_banned: set[int] = set()

    @property
    def function(self) -> str:
        return ".".join(self.stack) if self.stack else "<module>"

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

    def _clock_kinds_in(self, nodes: list[ast.AST]) -> str:
        kinds: set[str] = set()
        for top in nodes:
            for sub in ast.walk(top):
                if isinstance(sub, (ast.Name, ast.Attribute)):
                    name = resolve(sub, self.bindings)
                    kind = kind_of(name) if name else None
                    if kind in ("wall", "monotonic", "calendar"):
                        kinds.add(kind)
        return "+".join(sorted(kinds)) if kinds else "caller-supplied"

    def _banned_ref(self, node: ast.Name | ast.Attribute) -> None:
        name = resolve(node, self.bindings)
        kind = kind_of(name) if name else None
        if kind is None or name is None:
            return
        # The innermost banned prefix is the site: for ``np.random.rand`` the ``np.random`` value under it is
        # marked so it is not counted a second time.
        if isinstance(node, ast.Attribute):
            self._value_of_banned.add(id(node.value))
        if id(node) in self._value_of_banned:
            return
        parent_call = getattr(node, "_census_parent", None)
        is_call = isinstance(parent_call, ast.Call) and parent_call.func is node
        self.out.sites.append(
            Site(
                path=self.path,
                line=node.lineno,
                function=self.function,
                name=name,
                kind=kind,
                call=is_call,
                scope=scope_of(self.path),
                placement=placement_of(self.path, self.function, kind),
            )
        )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._banned_ref(node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._banned_ref(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Sub):
            sides = (node.left, node.right)
            stamped = [s for s in sides if isinstance(s, ast.Attribute) and s.attr == STAMP_ATTR]
            if stamped:
                others = [s for s in sides if s is not stamped[0]]
                self.out.subtractions.append(
                    StampSubtraction(
                        path=self.path,
                        line=node.lineno,
                        function=self.function,
                        other_clock=self._clock_kinds_in(list(others)),
                        placement=placement_of(self.path, self.function, None),
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _callee_text(node.func)
        if callee:
            last = callee.split(".")[-1]
            if last in STAMP_CLASSES:
                stamp_kw = [k for k in node.keywords if k.arg == STAMP_ATTR]
                if stamp_kw:
                    self.out.writers.append(
                        StampWriter(
                            path=self.path,
                            line=node.lineno,
                            function=self.function,
                            stamp_class=last,
                            input_clock=self._clock_kinds_in([stamp_kw[0].value]),
                        )
                    )
            if _is_temporal_callee(callee):
                args: list[ast.AST] = [*node.args, *(k.value for k in node.keywords)]
                self.out.temporal.append(
                    TemporalCaller(
                        path=self.path,
                        line=node.lineno,
                        function=self.function,
                        callee=callee,
                        input_clock=_temporal_input_clock(callee, self._clock_kinds_in(args)),
                        output_use=_output_use(node),
                        placement=placement_of(self.path, self.function, None),
                    )
                )
        self.generic_visit(node)


def _callee_text(func: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    elif isinstance(func, ast.Call):
        inner = _callee_text(func.func)
        parts.append(f"{inner}()" if inner else "?()")
    else:
        return None
    return ".".join(reversed(parts))


_SCN_SEGMENT = re.compile(r"^_?scn$", re.IGNORECASE)


def _is_temporal_callee(callee: str) -> bool:
    if "()" in callee:
        return False  # a method on a call's result (``X.now().to_dict()``): the inner call is the one recorded
    segments = callee.split(".")
    if any(seg.startswith("TemporalSignature") for seg in segments):
        return True
    # A method called on an SCN object (``self.scn.x``, ``self._scn.x``, ``scn.x``) or the SCN constructor.
    return any(_SCN_SEGMENT.match(seg) for seg in segments[:-1]) or segments[-1] == "SCN"


def _temporal_input_clock(callee: str, from_args: str) -> str:
    """``TemporalSignature.now()`` takes no clock: it reads ``time.time()`` and bins it (time/temporal_signature.py)."""
    if callee.endswith("TemporalSignature.now") and from_args == "caller-supplied":
        return "wall (inside TemporalSignature.now)"
    return from_args


def _output_use(call: ast.Call) -> str:
    """What the call's result feeds: climbs through a method chained on the result (``x().y()``) first."""
    node: ast.AST = call
    parent = getattr(node, "_census_parent", None)
    while (isinstance(parent, ast.Attribute) and parent.value is node) or (
        isinstance(parent, ast.Call) and parent.func is node
    ):
        node, parent = parent, getattr(parent, "_census_parent", None)
    if isinstance(parent, (ast.FormattedValue, ast.JoinedStr)):
        return "rendered (f-string)"
    if isinstance(parent, ast.Return):
        return "returned"
    if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        return "assigned"
    if isinstance(parent, ast.Expr):
        return "statement"
    if isinstance(parent, (ast.Call, ast.keyword)):
        return "argument"
    if isinstance(parent, (ast.Compare, ast.BoolOp, ast.If, ast.IfExp)):
        return "tested"
    return type(parent).__name__ if parent is not None else "?"


def scan_text(text: str, path: str) -> FileCensus:
    """Census of one file's source. Raises ``SyntaxError`` for a file that does not parse."""
    tree = ast.parse(text, filename=path)
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._census_parent = parent  # type: ignore[attr-defined]
    scanner = _Scanner(path, module_bindings(tree))
    scanner.visit(tree)
    return scanner.out


class CensusError(Exception):
    """The tree could not be listed or a file could not be parsed: no census, exit 2."""


def tracked_files(root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--", f"{SRC}/*.py", f"{SRC}/**/*.py"],
            cwd=root,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CensusError(f"git ls-files failed under {root}: {exc}") from exc
    files = sorted({p for p in proc.stdout.decode().split("\0") if p})
    if not files:
        raise CensusError(f"git lists no tracked .py files under {root / SRC}")
    return files


def census(root: Path) -> FileCensus:
    total = FileCensus()
    for rel in tracked_files(root):
        try:
            text = (root / rel).read_text(encoding="utf-8")
            one = scan_text(text, rel)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            raise CensusError(f"{rel}: {exc}") from exc
        total.sites.extend(one.sites)
        total.subtractions.extend(one.subtractions)
        total.writers.extend(one.writers)
        total.temporal.extend(one.temporal)
    return total


# ── output ───────────────────────────────────────────────────────────────────


def render(result: FileCensus) -> str:
    lines: list[str] = []
    by_kind: dict[str, int] = {}
    by_placement: dict[str, int] = {}
    for s in result.sites:
        by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
        by_placement[s.placement] = by_placement.get(s.placement, 0) + 1
    lines.append(f"world clock census: {len(result.sites)} site(s)")
    lines.append("  by kind: " + ", ".join(f"{k} {n}" for k, n in sorted(by_kind.items())))
    lines.append("  by placement:")
    for placement, n in sorted(by_placement.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"    {n:5d}  {placement}")
    lines.append("")
    lines.append("== sites (path:line  function  name  kind  ref/call  scope  placement) ==")
    for s in result.sites:
        ref = "call" if s.call else "ref"
        lines.append(f"{s.path}:{s.line}  {s.function}  {s.name}  {s.kind}  {ref}  {s.scope}  {s.placement}")
    lines.append("")
    lines.append(f"== stamp writers ({', '.join(STAMP_CLASSES)} built with timestamp=) ==")
    for w in result.writers:
        lines.append(f"{w.path}:{w.line}  {w.function}  {w.stamp_class}  input={w.input_clock}")
    lines.append("")
    lines.append("== stamp readers: every subtraction with a .timestamp operand (untyped; I4 pairs) ==")
    for r in result.subtractions:
        lines.append(f"{r.path}:{r.line}  {r.function}  other={r.other_clock}  {r.placement}")
    lines.append("")
    lines.append("== TemporalSignature and SCN callers (input clock -> output use) ==")
    for t in result.temporal:
        lines.append(f"{t.path}:{t.line}  {t.function}  {t.callee}  {t.input_clock} -> {t.output_use}  {t.placement}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="print the census as JSON")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root (default: this checkout)")
    args = parser.parse_args(argv)
    try:
        result = census(args.root)
    except CensusError as exc:
        print(f"ERROR: world clock census: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(asdict(result), indent=1))
    else:
        print(render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
