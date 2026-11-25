#!/usr/bin/env python3
"""World clock A-pre census: every time, calendar, id and randomness site in ``src/maxim/`` (PR 0).

``docs/plans/world_clock_a_pre.md`` says the census, not the plan's illustrative lists, is authoritative for
where each site goes. This script prints, from the AST alone (it imports nothing from ``maxim``):

1. **Sites.** Every reference (not only calls) to a banned clock, calendar, id or randomness name, resolved
   through local import aliases (``scripts/_clock_refs.py``, shared with M72's lint), with its ``file::function``,
   kind, M72 scope and placement.
2. **Stamps (I4).** Writers of the stamp attributes (``timestamp``, ``created_at``, ``accessed_at``,
   ``consolidated_at``): a stamp class built with one as a keyword (including ``cls(...)`` inside a stamp class),
   every assignment to one (plain, augmented, annotated or inside a tuple target), a class-body field default, and
   every dict literal with one as a key. Readers: every subtraction with a stamp operand, in attribute, subscript
   (``x["timestamp"]``), ``getattr(x, "timestamp", ...)`` or ``x.get("timestamp", ...)`` form, with the clock the
   other operand reads. Both are untyped: AST cannot tell an ``EpisodicMemory`` from any other object with a
   ``timestamp``, so every one is listed and a reviewer reads the list.
3. **TemporalSignature and SCN callers,** with the clock their arguments read (input) and what the result
   feeds (output use).
4. **Resolution conflicts:** a name bound to two clock targets in one module, or a star import from a clock
   module. The census does not guess; M72's lint fails closed on them.

The output is for review. Neither the M72 lint nor G-long reads its verdicts; they rescan. A site inside an M72
scope that no rule places prints as ``UNPLACED``, so a reviewer sees it rather than a guessed placement; PR 1
gives each one an allowlist entry, a ``pending:`` pin or a move before M72 goes live. A site outside both scopes
prints as ``outside M72's scopes``. ``tests/unit/test_world_clock_census.py`` fails when a placement rule matches
nothing on the tree (a rule naming a function that moved), unless the rule says why it is empty today.

Stated blind spots: dynamic access (``getattr(time, "time")``), a callable passed in, a local variable shadowing an
imported name, render sites that format a stamp without reading a clock (``_format_episodic_memory``), a subscript
assignment (``d["timestamp"] = ...``; none on today's tree), and an UNSEEDED owned generator
(``random.Random()``, ``np.random.default_rng()``), which draws from OS entropy but is excluded with the seeded ones.

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
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _clock_refs import SRC, Bindings, kind_of, module_bindings, rel_to_src, resolve, scope_of, under  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

STAMP_CLASSES = frozenset({"EpisodicMemory", "MemoryRecord", "Percept", "SensorReading", "PainSignal"})
STAMP_ATTRS = frozenset({"timestamp", "created_at", "accessed_at", "consolidated_at"})


class Rule(NamedTuple):
    """One placement rule. ``kinds`` None matches any kind; ``func`` None any function. ``empty_today`` is the reason
    a rule matches no site on today's tree (it fences a file or family that has none yet); a rule without it must
    match at least one item, and a rule with it must match none."""

    kinds: frozenset[str] | None
    prefix: str
    func: str | None
    placement: str
    empty_today: str | None = None


_ANY = None
_FENCED = "A-post: fenced file (OD-A8)"
_HOT = "A-post: hot file (D-WC7)"
_OFF_PATH = "out of scope: off the embedded path"
_STAMP_HALF = "A-post: OD-A4 stamp half (D-WC6)"
_SCN_REG = "with the stamp half (SCN registration)"
_OD_A9 = "A-post with OD-A9"

# From world_clock_a_pre.md (§Site families, §Placements, §What stays on the wall). First match wins. Kind rules
# come first because ids and randomness are OD-A7's wherever they sit (D-WC1).
PLACEMENT_RULES: tuple[Rule, ...] = (
    Rule(frozenset({"id", "random"}), "", _ANY, "OD-A7, after the 1.4.0 cut (D-WC1)"),
    Rule(_ANY, "mesh/", _ANY, _OFF_PATH),
    Rule(_ANY, "utils/web_cache.py", _ANY, _OFF_PATH),
    Rule(_ANY, "utils/output_watcher.py", _ANY, _OFF_PATH),
    Rule(_ANY, "runtime/worker_pool.py", _ANY, _OFF_PATH),
    Rule(_ANY, "runtime/agent_loop.py", _ANY, _FENCED),
    Rule(_ANY, "simulation/orchestrator.py", _ANY, _FENCED),
    Rule(_ANY, "runtime/tool_dispatch.py", r"^execute_and_learn$", "A-post with B2: OD-A4's tool duration (#1788)"),
    Rule(_ANY, "runtime/tool_dispatch.py", _ANY, _FENCED),
    Rule(_ANY, "agents/exec_agent.py", r"_build_llm_context$", "allow: real-world-io (workspace mtimes)"),
    Rule(_ANY, "agents/exec_agent.py", _ANY, _HOT),
    Rule(_ANY, "models/language/router.py", _ANY, _HOT),
    Rule(_ANY, "cli.py", _ANY, _HOT),
    Rule(_ANY, "runtime/loop_setup.py", _ANY, _HOT, empty_today="no clock site today; kept to fence the hot file"),
    Rule(_ANY, "agents/prompt_builder.py", r"(^|\.)(build_prompt|_build_engage_prompt)$", "PR 2: the port (OD-A3)"),
    Rule(_ANY, "agents/context_pool.py", r"(^|\.)(_entries_to_text|get_context_text)$", "PR 2: perceive (OD-A3)"),
    Rule(_ANY, "tools/introspection.py", r"^SceneSummaryTool\.", "PR 2: omitted under NO_CALENDAR"),
    Rule(_ANY, "tools/introspection.py", r"^TemporalPatternsTool\.", "PR 3: SCN-derived render (D-WC2)"),
    Rule(
        _ANY,
        "agents/statistician_agent.py",
        r"(_assess_with_temporal_context|summarize_for_context)$",
        "PR 3: [TEMPORAL] (D-WC2)",
    ),
    Rule(_ANY, "simulation/introspection.py", _ANY, "PR 3: current_phase (D-WC2)"),
    Rule(_ANY, "retrieval/temporal_signal.py", _ANY, "PR 3: relative dates (D-WC2)"),
    Rule(_ANY, "memory/hippocampus.py", r"(^|\.)store_observation$", "PR 4: the dedup window (closed pair)"),
    Rule(
        _ANY,
        "memory/hippocampus.py",
        r"(^|\.)(_build_capture_record|get|_touch_internal|_evict_one|consolidate[a-z_]*)$",
        _STAMP_HALF,
    ),
    # The forgetting readers that move with the stamps (§Placements: ConsolidationMixin.sleep, _evict_one).
    Rule(_ANY, "memory/hippocampus_consolidation.py", _ANY, _STAMP_HALF),
    Rule(_ANY, "proprioception/pain_bus.py", r"(^|\.)publish$", "PR 5 (1.3.6): refractory (D-WC3)"),
    Rule(_ANY, "reactions/bus.py", r"(^|\.)publish$", "PR 5 (1.3.6): refractory (D-WC3)"),
    Rule(_ANY, "proprioception/pain.py", r"(^|\.)(_emit_pain|record_tool_error)$", "PR 5 (1.3.6): PainDetector pair"),
    # Motion pain's cooldown: _can_emit_pain takes ``now`` from these callers, which read the wall.
    Rule(
        _ANY,
        "proprioception/pain.py",
        r"(^|\.)(set_movement_target|_check_movement_failure|_check_for_pain)$",
        "allow: motor-io (motion pain cooldown)",
    ),
    Rule(_ANY, "models/language/cost_tracker.py", _ANY, "allow: cost (D-WC5; #1769 in PR 1)"),
    Rule(_ANY, "tools/http_fetch.py", r"^HttpFetchTool\.execute$", "allow: real-world-io (accessed_at)"),
    Rule(_ANY, "embodiment/body.py", _ANY, "A-post (OD-A8) with #1361 S4"),
    Rule(_ANY, "embodiment/spec.py", r"^SpecSensor\.read$", "A-post: OD-A4 stamp half (SensorReading)"),
    Rule(_ANY, "similarity/signature.py", _ANY, "with the stamp half (SituationSignature binning)"),
    Rule(_ANY, "integration/memory_hub.py", r"_on_memory_captured_scn$", _SCN_REG),
    Rule(_ANY, "memory/concept_extractor.py", _ANY, _SCN_REG),
    Rule(_ANY, "time/scn.py", _ANY, _SCN_REG),
    Rule(_ANY, "salience/", _ANY, _OD_A9),
    Rule(_ANY, "attention/gaze_history.py", _ANY, _OD_A9),
    Rule(_ANY, "attention/salience_map.py", _ANY, _OD_A9),
    Rule(_ANY, "default_network/network.py", _ANY, _OD_A9),
    Rule(_ANY, "decisions/nac.py", _ANY, "A-post, after #1361 S3b (NAc stamps and anchors)"),
    Rule(_ANY, "motion/", _ANY, "allow: motor-io", empty_today="no clock site today; kept so motion stays motor-io"),
    Rule(_ANY, "hardware/", _ANY, "allow: motor-io"),
)


def match_rule(path: str, function: str, kind: str | None) -> int | None:
    """The index of the first rule placing (path, function, kind), or None."""
    rel = rel_to_src(path)
    for i, rule in enumerate(PLACEMENT_RULES):
        if rule.kinds is not None and kind not in rule.kinds:
            continue
        if rule.prefix and not under(rel, rule.prefix):
            continue
        if rule.func is not None and not re.search(rule.func, function):
            continue
        return i
    return None


def placement_of(path: str, function: str, kind: str | None) -> str:
    i = match_rule(path, function, kind)
    if i is not None:
        return PLACEMENT_RULES[i].placement
    # A site in neither M72 scope is not A-pre's to place; one inside a scope that no rule names is a finding.
    return "UNPLACED" if scope_of(path) != "-" else "outside M72's scopes"


# Census output only: never persisted or sent over a wire, so CC3's forward-compat rule does not apply.
@dataclass(frozen=True)
class Site:
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
    path: str
    line: int
    function: str
    attr: str
    form: str
    other_clock: str
    placement: str


@dataclass(frozen=True)
class StampWriter:
    path: str
    line: int
    function: str
    target: str
    attrs: str
    input_clock: str


@dataclass(frozen=True)
class TemporalCaller:
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
    conflicts: list[str] = field(default_factory=list)


def _stamp_operand(node: ast.AST) -> tuple[str, str] | None:
    """(attr, form) when *node* reads a stamp attribute, else None."""
    if isinstance(node, ast.Attribute) and node.attr in STAMP_ATTRS:
        return node.attr, "attribute"
    if isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and key.value in STAMP_ATTRS:
            return str(key.value), "subscript"
    if isinstance(node, ast.Call) and node.args:
        key = node.args[1] if len(node.args) > 1 else None
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" and isinstance(key, ast.Constant):
            if key.value in STAMP_ATTRS:
                return str(key.value), "getattr"
        first = node.args[0]
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(first, ast.Constant)
            and first.value in STAMP_ATTRS
        ):
            return str(first.value), ".get"
    return None


class _Scanner(ast.NodeVisitor):
    def __init__(self, path: str, bindings: Bindings) -> None:
        self.path = path
        self.bindings = bindings
        self.stack: list[str] = []
        self.classes: list[str] = []
        self.out = FileCensus(conflicts=[f"{path} {c}" for c in bindings.conflicts])
        self._value_of_banned: set[int] = set()

    @property
    def function(self) -> str:
        return ".".join(self.stack) if self.stack else "<module>"

    def _scoped(self, node: ast.AST, name: str) -> None:
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self._scoped(node, node.name)
        self.classes.pop()

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
        # Visiting is outermost first, so the outermost banned name in a chain is the site: for
        # ``np.random.mtrand.rand`` the ``np.random.mtrand`` under it is marked and not counted again.
        if isinstance(node, ast.Attribute):
            self._value_of_banned.add(id(node.value))
        if id(node) in self._value_of_banned:
            return
        parent = getattr(node, "_census_parent", None)
        self.out.sites.append(
            Site(
                path=self.path,
                line=node.lineno,
                function=self.function,
                name=name,
                kind=kind,
                call=isinstance(parent, ast.Call) and parent.func is node,
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
            # Each stamp operand is its own reader: ``rec.created_at - rec.consolidated_at`` reads two stamps.
            for side, other in ((node.left, node.right), (node.right, node.left)):
                stamp = _stamp_operand(side)
                if stamp is not None:
                    self.out.subtractions.append(
                        StampSubtraction(
                            path=self.path,
                            line=node.lineno,
                            function=self.function,
                            attr=stamp[0],
                            form=stamp[1],
                            other_clock=self._clock_kinds_in([other]),
                            placement=placement_of(self.path, self.function, None),
                        )
                    )
        self.generic_visit(node)

    def _writer(self, line: int, target: str, attr: str, value: ast.AST) -> None:
        self.out.writers.append(
            StampWriter(
                path=self.path,
                line=line,
                function=self.function,
                target=target,
                attrs=attr,
                input_clock=self._clock_kinds_in([value]),
            )
        )

    def _assigned(self, targets: list[ast.expr], value: ast.AST, line: int, how: str = "assignment") -> None:
        for target in targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                self._assigned(list(target.elts), value, line, how)
            elif isinstance(target, ast.Attribute) and target.attr in STAMP_ATTRS:
                self._writer(line, f"{how} .{target.attr}", target.attr, value)
            elif (
                isinstance(target, ast.Name)
                and target.id in STAMP_ATTRS
                and self.classes
                and self.stack
                and self.stack[-1] == self.classes[-1]
            ):
                # A class-body field (``timestamp: float = field(default_factory=time.time)``) stamps every instance.
                self._writer(line, f"field default {self.classes[-1]}.{target.id}", target.id, value)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._assigned(list(node.targets), node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._assigned([node.target], node.value, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._assigned([node.target], node.value, node.lineno, how="augmented assignment")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value in STAMP_ATTRS:
                self._writer(node.lineno, f'dict literal "{key.value}"', str(key.value), value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _callee_text(node.func)
        if callee:
            last = callee.split(".")[-1]
            in_stamp_class = last == "cls" and bool(self.classes) and self.classes[-1] in STAMP_CLASSES
            stamp_kw = [k for k in node.keywords if k.arg in STAMP_ATTRS]
            if stamp_kw and (last in STAMP_CLASSES or in_stamp_class):
                self.out.writers.append(
                    StampWriter(
                        path=self.path,
                        line=node.lineno,
                        function=self.function,
                        target=f"{self.classes[-1]} (cls)" if in_stamp_class else last,
                        attrs=",".join(sorted(str(k.arg) for k in stamp_kw)),
                        input_clock=self._clock_kinds_in([k.value for k in stamp_kw]),
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
        # A git pathspec ``*`` matches across ``/``, so this lists nested files too.
        proc = subprocess.run(["git", "ls-files", "-z", "--", f"{SRC}/*.py"], cwd=root, capture_output=True, check=True)
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
            one = scan_text((root / rel).read_text(encoding="utf-8"), rel)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            raise CensusError(f"{rel}: {exc}") from exc
        total.sites.extend(one.sites)
        total.subtractions.extend(one.subtractions)
        total.writers.extend(one.writers)
        total.temporal.extend(one.temporal)
        total.conflicts.extend(one.conflicts)
    return total


def rule_hits(result: FileCensus) -> list[int]:
    """How many sites, stamp readers and temporal callers each placement rule places, by rule index."""
    hits = [0] * len(PLACEMENT_RULES)
    items: list[tuple[str, str, str | None]] = [(s.path, s.function, s.kind) for s in result.sites]
    items += [(r.path, r.function, None) for r in result.subtractions]
    items += [(t.path, t.function, None) for t in result.temporal]
    for path, function, kind in items:
        i = match_rule(path, function, kind)
        if i is not None:
            hits[i] += 1
    return hits


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
    lines.append(f"== resolution conflicts ({len(result.conflicts)}) ==")
    lines.extend(result.conflicts)
    lines.append("")
    lines.append("== sites (path:line  function  name  kind  ref/call  scope  placement) ==")
    for s in result.sites:
        ref = "call" if s.call else "ref"
        lines.append(f"{s.path}:{s.line}  {s.function}  {s.name}  {s.kind}  {ref}  {s.scope}  {s.placement}")
    lines.append("")
    lines.append(f"== stamp writers ({', '.join(sorted(STAMP_ATTRS))}: constructors, assignments, fields, dicts) ==")
    for w in result.writers:
        lines.append(f"{w.path}:{w.line}  {w.function}  {w.target}  {w.attrs}  input={w.input_clock}")
    lines.append("")
    lines.append("== stamp readers: every subtraction with a stamp operand (untyped; I4 pairs) ==")
    for r in result.subtractions:
        lines.append(f"{r.path}:{r.line}  {r.function}  {r.attr} ({r.form})  other={r.other_clock}  {r.placement}")
    lines.append("")
    lines.append("== TemporalSignature and SCN callers (input clock -> output use) ==")
    for t in result.temporal:
        lines.append(f"{t.path}:{t.line}  {t.function}  {t.callee}  {t.input_clock} -> {t.output_use}  {t.placement}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="print the census as JSON")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root (default: this checkout)")
    args = parser.parse_args(argv)
    try:
        result = census(args.root)
    except CensusError as exc:
        print(f"ERROR: world clock census: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(result), indent=1) if args.json else render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
