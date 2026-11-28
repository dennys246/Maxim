"""Shared clock-reference vocabulary for the world-clock census and M72's lint (world_clock_a_pre_guard_note.md §1).

One place for the banned names, the two M72 scopes, and the alias resolver, so ``scripts/world_clock_census.py``
(PR 0) and ``scripts/lint_wall_clock_reads.py`` (PR 1) cannot drift apart: the census and the lint each scan and
judge independently, but they read references the same way. Imports nothing from ``maxim``.

Resolution is per module. A local name bound to two different targets anywhere in one module (a function-local
``from datetime import datetime`` beside a module-level ``import datetime``), or a star import from a clock module,
is reported in ``Bindings.conflicts`` rather than resolved one way: the census prints it, and the lint fails closed
on it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

SRC = "src/maxim"

# clock_gettime reads whichever clock its argument names; it counts as wall, the conservative reading.
WALL = frozenset({"time.time", "time.time_ns", "time.clock_gettime", "time.clock_gettime_ns"})
MONOTONIC = frozenset({"time.monotonic", "time.monotonic_ns", "time.perf_counter", "time.perf_counter_ns"})
CALENDAR = frozenset(
    {
        "time.localtime",
        "time.gmtime",
        "time.strftime",
        "time.ctime",
        "time.asctime",
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
        "datetime.datetime.today",
        "datetime.datetime.fromtimestamp",
        "datetime.datetime.utcfromtimestamp",
        "datetime.date.today",
        "datetime.date.fromtimestamp",
        # The host calendar through maxim's own helper (SCN bins; a transitive read, counted by name, PR 1b).
        "maxim.time.temporal_signature.TemporalSignature.now",
    }
)
IDS = frozenset({"uuid.uuid1", "uuid.uuid4"})
RANDOM_PREFIXES = ("random.", "numpy.random.")
# Types and constructors of OWNED generators: not draws from a process-global one.
RANDOM_NON_DRAWS = frozenset(
    {
        "random.Random",
        "random.SystemRandom",
        "numpy.random.Generator",
        "numpy.random.RandomState",
        "numpy.random.default_rng",
        "numpy.random.SeedSequence",
    }
)
CLOCK_MODULES = frozenset({"time", "datetime", "uuid", "random", "numpy.random", "maxim.time.temporal_signature"})

# M72's two scopes. A path ending in "/" is a directory prefix; paths are relative to src/maxim.
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
    # M72's original row names focus_learner's _apply_decay; the whole file is in scope (owner, PR 1b, 2026-11-26).
    "proprioception/focus_learner.py",
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


def rel_to_src(path: str) -> str:
    return path[len(SRC) + 1 :] if path.startswith(SRC + "/") else path


def under(rel: str, prefix: str) -> bool:
    return rel.startswith(prefix) if prefix.endswith("/") else rel == prefix


def scope_of(path: str) -> str:
    """``machinery``, ``prompt``, ``machinery+prompt`` or ``-`` for a repo-relative path."""
    rel = rel_to_src(path)
    names = [
        name
        for name, prefixes in (("machinery", MACHINERY_SCOPE), ("prompt", PROMPT_SCOPE))
        if any(under(rel, p) for p in prefixes)
    ]
    return "+".join(names) if names else "-"


@dataclass
class Bindings:
    """A module's import bindings: local name → canonical dotted name, plus what could not be resolved one way."""

    names: dict[str, str] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)


def _is_clock(target: str) -> bool:
    return any(target == m or target.startswith(m + ".") for m in CLOCK_MODULES)


def _bind(out: Bindings, local: str, target: str, line: int) -> None:
    previous = out.names.get(local)
    if previous is not None and previous != target and (_is_clock(previous) or _is_clock(target)):
        out.conflicts.append(f"L{line}: {local!r} bound to both {previous} and {target}")
    out.names[local] = target


def module_bindings(tree: ast.AST) -> Bindings:
    """Every import in *tree*, at any nesting depth.

    Handles ``import time``, ``import time as _t``, ``import os.path``, ``from time import time as now``,
    ``from datetime import datetime``, ``import datetime as dt`` and ``from datetime import datetime as DT``. A
    relative import binds nothing (it cannot name a stdlib clock). A star import from a clock module, and a name
    bound to two targets, go to ``conflicts``. Stated blind spot: a local *variable* (not an import) that shadows an
    imported name is not tracked.
    """
    out = Bindings()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    _bind(out, alias.asname, alias.name, node.lineno)
                else:
                    root = alias.name.split(".")[0]
                    _bind(out, root, root, node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                if alias.name == "*":
                    if node.module in CLOCK_MODULES:
                        out.conflicts.append(f"L{node.lineno}: from {node.module} import * hides which names it binds")
                    continue
                _bind(out, alias.asname or alias.name, f"{node.module}.{alias.name}", node.lineno)
    return out


def resolve(node: ast.AST, bindings: Bindings) -> str | None:
    """The canonical dotted name of a ``Name``/``Attribute`` chain rooted at an imported name, else None."""
    attrs: list[str] = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in bindings.names:
        return None
    return ".".join([bindings.names[node.id], *reversed(attrs)])


def kind_of(name: str) -> str | None:
    """``wall``, ``monotonic``, ``calendar``, ``id``, ``random`` or None for a canonical dotted name."""
    if name in WALL:
        return "wall"
    if name in MONOTONIC:
        return "monotonic"
    if name in CALENDAR:
        return "calendar"
    if name in IDS:
        return "id"
    if name.startswith(RANDOM_PREFIXES) and name not in RANDOM_NON_DRAWS:
        return "random"
    return None
