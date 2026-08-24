"""Architecture audit: verify layer ownership and one-way dependency rules.

Two halves:

1. ``audit_architecture()`` scans ``src/maxim/`` for imports that cross a
   forbidden layer boundary (``LAYER_RULES``) and reports each as an
   :class:`AuditViolation`, tagged with the *scope* of the import — module
   level, function-local, or under ``if TYPE_CHECKING:``.
2. ``compare_to_baseline()`` diffs those findings against the committed,
   reviewed accepted-debt baseline (``architecture_baseline.json``, next to
   this file). The baseline is what makes the audit a regression gate instead
   of a permanently-red report (bugs ledger D19): CI fails on any finding that
   is not in the baseline, on any baseline entry that no longer matches the
   code (stale — the baseline must shrink with the debt), and on any entry
   whose disposition is still ``UNREVIEWED``.

Baseline entries are keyed by ``file::imported_module::scope`` with a count
AND the accepted symbol set, never by line number — line numbers drift on
every unrelated edit. The same import escalating from ``type-checking`` to
``module`` scope therefore reads as a NEW finding, and so does widening an
accepted ``from X import a`` to ``from X import a, b``: both are added debt.
A symbol no longer imported makes the entry stale, like a dropped occurrence.

``from maxim import tools`` and relative forms (``from .. import tools``) are
resolved to the canonical ``maxim.<layer>...`` module and checked per imported
name, so an accepted edge has one key whether written absolute or relative.
Dynamic imports (``importlib.import_module``, ``__import__``) are out of scope
by design. A bare ``import maxim.agents.bus`` records the module path as its
only symbol, which accepts every attribute — accepted debt should be written
``from X import name`` so the symbol check has teeth.

The baseline is a checked-in review artifact shipped as package data, not
runtime persistence: it is versioned by ``baseline_format_version`` (like
``console/contract_surface.json``), not by the ``_format_version`` stamp that
``utils/format_version.py`` puts on files Maxim writes at runtime.

Scope detection is conservative in the runtime direction: ``if TYPE_CHECKING
and X:`` (a BoolOp test) and the ``else`` branch of ``if not TYPE_CHECKING:``
are both reported as ``module`` scope. Over-reporting a typing edge as runtime
is the safe failure; none of those shapes exist in ``src/maxim/`` today.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASELINE_PATH = Path(__file__).with_name("architecture_baseline.json")
BASELINE_FORMAT_VERSION = 1

SCOPE_MODULE = "module"
SCOPE_FUNCTION_LOCAL = "function-local"
SCOPE_TYPE_CHECKING = "type-checking"
SCOPES = (SCOPE_MODULE, SCOPE_FUNCTION_LOCAL, SCOPE_TYPE_CHECKING)

DISPOSITION_ACCEPTED = "accepted"
DISPOSITION_TYPING_ONLY = "typing-only"
DISPOSITION_UNREVIEWED = "UNREVIEWED"
REVIEWED_DISPOSITIONS = (DISPOSITION_ACCEPTED, DISPOSITION_TYPING_ONLY)


@dataclass(frozen=True)
class AuditViolation:
    """A single architecture rule violation.

    Runtime-ephemeral (never persisted or sent over a wire): the committed
    baseline is its own JSON shape, rendered by :func:`render_baseline`.
    ``scope`` defaults to ``"module"`` so pre-existing constructors keep working.
    """

    file: str
    line: int
    layer: str
    imported_module: str
    rule: str  # e.g., "agents must_not_import tools"
    scope: str = SCOPE_MODULE
    names: tuple[str, ...] = ()  # imported symbols (``from X import a, b`` → ("a", "b"))

    @property
    def baseline_key(self) -> str:
        return baseline_key(self.file, self.imported_module, self.scope)


@dataclass(frozen=True)
class BaselineEntry:
    """One reviewed accepted-debt finding from ``architecture_baseline.json``.

    In-memory form of a checked-in record. Forward-compat lives in
    :func:`parse_baseline`, not here: unknown keys are ignored, ``rationale``
    defaults, and the document is gated on ``baseline_format_version``.
    """

    file: str
    imported_module: str
    rule: str
    scope: str
    count: int
    disposition: str
    symbols: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def key(self) -> str:
        return baseline_key(self.file, self.imported_module, self.scope)

    @property
    def is_reviewed(self) -> bool:
        return self.disposition in REVIEWED_DISPOSITIONS and bool(self.rationale.strip())


@dataclass(frozen=True)
class StaleEntry:
    """A baseline entry the code no longer justifies (runtime-ephemeral)."""

    entry: BaselineEntry
    live_count: int
    missing_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaselineDiff:
    """Result of comparing live findings against the baseline (runtime-ephemeral).

    ``added``: findings with no baseline entry, in excess of an entry's count,
    or importing a symbol the entry does not accept. ``stale``: baseline
    entries with fewer live findings than their count or listing symbols no
    longer imported (the debt shrank — tighten the baseline). ``unreviewed``:
    entries still carrying the ``UNREVIEWED`` disposition or no rationale.
    """

    added: tuple[AuditViolation, ...] = ()
    stale: tuple[StaleEntry, ...] = ()
    unreviewed: tuple[BaselineEntry, ...] = ()
    accepted_count: int = 0
    baseline_entry_count: int = 0

    @property
    def ok(self) -> bool:
        return not (self.added or self.stale or self.unreviewed)


def baseline_key(file: str, imported_module: str, scope: str) -> str:
    return f"{file}::{imported_module}::{scope}"


LAYER_RULES: dict[str, dict[str, list[str]]] = {
    "agents": {
        "must_not_import": ["tools", "environment", "hardware", "conscience", "runtime"],
    },
    "planning": {
        "must_not_import": ["tools", "environment", "hardware", "runtime"],
    },
    "environment": {
        "must_not_import": ["tools", "hardware"],
    },
    "tools": {
        "must_not_import": ["agents", "planning", "environment", "runtime", "conscience"],
    },
    "memory": {
        "must_not_import": ["agents", "planning", "tools", "environment", "runtime"],
    },
    "runtime": {
        "must_not_import": ["conscience"],
    },
    "skills": {
        "must_not_import": ["agents", "planning", "runtime", "conscience"],
    },
    "bridges": {
        "must_not_import": ["agents", "planning", "tools", "runtime", "conscience"],
    },
}


def audit_architecture(src_root: str | Path | None = None) -> list[AuditViolation]:
    """Scan src/maxim/ for layer dependency violations.

    Returns list of violations found. Empty list means all rules pass.
    """
    if src_root is None:
        # Find src/maxim/ relative to this file
        src_root = Path(__file__).resolve().parent.parent  # src/maxim/
    src_root = Path(src_root)

    violations: list[AuditViolation] = []

    for layer, rules in LAYER_RULES.items():
        layer_dir = src_root / layer
        if not layer_dir.is_dir():
            continue

        forbidden = rules.get("must_not_import", [])
        if not forbidden:
            continue

        for py_file in sorted(layer_dir.rglob("*.py")):
            file_violations = _check_file(py_file, layer, forbidden, src_root)
            violations.extend(file_violations)

    return violations


def _is_type_checking_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _iter_imports_with_scope(tree: ast.AST):
    """Yield ``(import_node, scope)`` for every import in the tree.

    Scope is determined by the innermost enclosing construct: an
    ``if TYPE_CHECKING:`` body wins over a function body, so a typing-only
    import inside a function is still ``type-checking``.
    """
    stack: list[tuple[ast.AST, str]] = [(tree, SCOPE_MODULE)]
    while stack:
        node, scope = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                yield child, scope
                continue
            child_scope = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and scope != SCOPE_TYPE_CHECKING:
                child_scope = SCOPE_FUNCTION_LOCAL
            elif isinstance(child, ast.If) and _is_type_checking_test(child.test):
                # Only the ``if`` body is typing-only; the ``else`` branch is real.
                stack.append((_BodyOnly(child.orelse), scope))
                stack.append((_BodyOnly(child.body), SCOPE_TYPE_CHECKING))
                continue
            stack.append((child, child_scope))


class _BodyOnly(ast.AST):
    """Wrapper so ``ast.iter_child_nodes`` walks just one branch of an ``if``."""

    _fields = ("body",)

    def __init__(self, body: list[ast.stmt]):
        super().__init__()
        self.body = body


def _check_file(
    file_path: Path,
    layer: str,
    forbidden_layers: list[str],
    src_root: Path,
) -> list[AuditViolation]:
    """Check a single Python file for forbidden imports."""
    violations = []
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    rel_path = str(file_path.relative_to(src_root.parent))  # Relative to src/
    # Dotted package of this file relative to the maxim package root, for
    # resolving relative imports: memory/foo.py → ["memory"]; memory/__init__.py → ["memory"].
    # (dropping the last part covers both cases: the module name, or ``__init__``).
    package_parts = list(file_path.relative_to(src_root).with_suffix("").parts)[:-1]

    for node, scope in _iter_imports_with_scope(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violation = _check_import_name(
                    alias.name, layer, forbidden_layers, rel_path, node.lineno, scope, names=(alias.name,)
                )
                if violation:
                    violations.append(violation)

        elif isinstance(node, ast.ImportFrom):
            module = _resolve_from_module(node, package_parts)
            names = tuple(alias.name for alias in node.names)
            violation = (
                _check_import_name(module, layer, forbidden_layers, rel_path, node.lineno, scope, names=names)
                if module
                else None
            )
            if violation:
                violations.append(violation)
                continue
            # ``from maxim import tools`` / ``from .. import tools``: the layer is
            # the imported NAME, not the module. One violation per offending name.
            for alias in node.names:
                candidate = f"{module}.{alias.name}" if module else alias.name
                violation = _check_import_name(
                    candidate, layer, forbidden_layers, rel_path, node.lineno, scope, names=(alias.name,)
                )
                if violation:
                    violations.append(violation)

    violations.sort(key=lambda v: v.line)
    return violations


def _resolve_from_module(node: ast.ImportFrom, package_parts: list[str]) -> str:
    """Absolute dotted module for an ``ImportFrom``, relative to the maxim package root.

    Relative imports resolve against ``package_parts`` (the file's package):
    ``from . import x`` in ``memory/foo.py`` → ``"memory"``; ``from .. import
    tools`` → ``""`` (package root, so the names are checked directly).
    Absolute imports are returned unchanged. A relative import deeper than the
    package root (an ``ImportError`` at runtime) clamps to the root and is
    still checked — over-reporting dead code is the safe direction.
    """
    if node.level == 0:
        return node.module or ""
    base = package_parts[: max(0, len(package_parts) - (node.level - 1))]
    if node.module:
        base = base + node.module.split(".")
    # Canonical ``maxim.``-prefixed form so absolute and relative spellings of
    # the same edge share one baseline key.
    return ".".join(["maxim", *base]) if base else "maxim"


def target_layer(import_name: str) -> str | None:
    """The layer an import name points at: ``maxim.tools.base`` / ``tools.base`` → ``"tools"``."""
    parts = import_name.split(".")
    if not parts:
        return None
    if parts[0] == "maxim":
        return parts[1] if len(parts) > 1 else None
    return parts[0] if parts[0] in LAYER_RULES or parts[0] in _ALL_TARGETS else None


_ALL_TARGETS: frozenset[str] = frozenset(t for rules in LAYER_RULES.values() for t in rules["must_not_import"])


def _check_import_name(
    import_name: str,
    layer: str,
    forbidden_layers: list[str],
    file_path: str,
    line: int,
    scope: str = SCOPE_MODULE,
    names: tuple[str, ...] = (),
) -> AuditViolation | None:
    """Check if an import name violates layer rules."""
    target = target_layer(import_name)
    if target is not None and target in forbidden_layers:
        return AuditViolation(
            file=file_path,
            line=line,
            layer=layer,
            imported_module=import_name,
            rule=f"{layer} must_not_import {target}",
            scope=scope,
            names=names,
        )
    return None


# ── Baseline ──────────────────────────────────────────────────────────────


class BaselineFormatError(ValueError):
    """The baseline file is malformed (wrong version, duplicate key, bad scope)."""


def load_baseline(path: str | Path = BASELINE_PATH) -> dict[str, BaselineEntry]:
    """Load ``architecture_baseline.json`` → ``{key: BaselineEntry}``.

    Raises :class:`FileNotFoundError` if absent and :class:`BaselineFormatError`
    on a malformed file — never returns a partial baseline.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineFormatError(f"architecture baseline: invalid JSON at {path}: {exc}") from exc
    return parse_baseline(data)


def parse_baseline(data: Any) -> dict[str, BaselineEntry]:
    if not isinstance(data, dict) or data.get("baseline_format_version") != BASELINE_FORMAT_VERSION:
        raise BaselineFormatError(
            f"architecture baseline: expected baseline_format_version={BASELINE_FORMAT_VERSION}, "
            f"got {data.get('baseline_format_version') if isinstance(data, dict) else type(data).__name__}"
        )
    entries: dict[str, BaselineEntry] = {}
    raw_entries = data.get("entries", [])
    if not isinstance(raw_entries, list):
        raise BaselineFormatError(f"architecture baseline: 'entries' must be a list, got {type(raw_entries).__name__}")
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise BaselineFormatError(f"architecture baseline: bad entry {raw!r}: not an object")
        if not isinstance(raw.get("symbols"), list) or not raw["symbols"]:
            raise BaselineFormatError(
                f"architecture baseline: 'symbols' must be a non-empty list in entry {raw.get('file')!r} → {raw.get('imported_module')!r}"
            )
        try:
            entry = BaselineEntry(
                file=str(raw["file"]),
                imported_module=str(raw["imported_module"]),
                rule=str(raw["rule"]),
                scope=str(raw["scope"]),
                count=int(raw["count"]),
                disposition=str(raw["disposition"]),
                symbols=tuple(sorted(str(n) for n in raw["symbols"])),
                rationale=str(raw.get("rationale", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BaselineFormatError(f"architecture baseline: bad entry {raw!r}: {exc}") from exc
        if entry.scope not in SCOPES:
            raise BaselineFormatError(f"architecture baseline: unknown scope {entry.scope!r} in {entry.key}")
        if entry.count < 1:
            raise BaselineFormatError(f"architecture baseline: count must be >= 1 in {entry.key}")
        if (entry.disposition == DISPOSITION_TYPING_ONLY) != (entry.scope == SCOPE_TYPE_CHECKING):
            raise BaselineFormatError(
                f"architecture baseline: disposition {entry.disposition!r} does not match scope "
                f"{entry.scope!r} in {entry.key} (typing-only <=> type-checking scope)"
            )
        if entry.key in entries:
            raise BaselineFormatError(f"architecture baseline: duplicate key {entry.key}")
        entries[entry.key] = entry
    return entries


def compare_to_baseline(
    violations: list[AuditViolation],
    baseline: dict[str, BaselineEntry],
) -> BaselineDiff:
    """Diff live findings against the reviewed baseline (see :class:`BaselineDiff`)."""
    by_key: dict[str, list[AuditViolation]] = {}
    for v in violations:
        by_key.setdefault(v.baseline_key, []).append(v)

    added: list[AuditViolation] = []
    accepted = 0
    for key, live in by_key.items():
        entry = baseline.get(key)
        allowed = entry.count if entry else 0
        accepted_symbols = set(entry.symbols) if entry else set()
        # Occurrences whose symbols the review accepted fill the allowance first,
        # lowest line first; whatever is left over is the surplus. Every
        # occurrence is listed by key so a reviewer can find them all.
        live = sorted(live, key=lambda v: (not set(v.names) <= accepted_symbols, v.line))
        kept, surplus = live[:allowed], live[allowed:]
        added.extend(sorted(surplus, key=lambda v: v.line))
        # An accepted edge that now pulls in symbols the review never saw is
        # new debt even though the key and count are unchanged.
        widened = [v for v in kept if not set(v.names) <= accepted_symbols]
        added.extend(widened)
        accepted += len(kept) - len(widened)

    stale: list[StaleEntry] = []
    unreviewed: list[BaselineEntry] = []
    for key, entry in baseline.items():
        live = by_key.get(key, [])
        live_symbols = {n for v in live for n in v.names}
        missing = tuple(sorted(set(entry.symbols) - live_symbols))
        if len(live) < entry.count or missing:
            stale.append(StaleEntry(entry, len(live), missing))
        if not entry.is_reviewed:
            unreviewed.append(entry)

    return BaselineDiff(
        added=tuple(added),
        stale=tuple(stale),
        unreviewed=tuple(unreviewed),
        accepted_count=accepted,
        baseline_entry_count=len(baseline),
    )


def render_baseline(
    violations: list[AuditViolation],
    existing: dict[str, BaselineEntry] | None = None,
    *,
    reviewed: str = "",
    comment: str = "",
) -> dict[str, Any]:
    """Build a baseline document from live findings (there is no CLI writer — this
    is for tests and for maintainers regenerating the file by hand).

    Known keys keep their existing disposition/rationale; new keys enter as
    ``UNREVIEWED`` so the gate stays red until a human classifies them. The
    symbol set is always taken from the live code.
    """
    existing = existing or {}
    grouped: dict[str, list[AuditViolation]] = {}
    for v in violations:
        grouped.setdefault(v.baseline_key, []).append(v)
    entries = []
    for key in sorted(grouped):
        live = grouped[key]
        prior = existing.get(key)
        entries.append(
            {
                "file": live[0].file,
                "imported_module": live[0].imported_module,
                "rule": live[0].rule,
                "scope": live[0].scope,
                "count": len(live),
                "symbols": sorted({n for v in live for n in v.names}),
                "disposition": prior.disposition if prior else DISPOSITION_UNREVIEWED,
                "rationale": prior.rationale if prior else "",
            }
        )
    doc: dict[str, Any] = {}
    if comment:
        doc["_comment"] = comment
    doc.update(
        {
            "baseline_format_version": BASELINE_FORMAT_VERSION,
            "reviewed": reviewed,
            "entries": entries,
        }
    )
    return doc


def format_diff(diff: BaselineDiff, baseline_path: str | Path = BASELINE_PATH) -> str:
    """Human-readable report for the CLI and test failure messages."""
    lines = [
        f"Architecture audit: {diff.accepted_count} accepted-debt finding(s) across "
        f"{diff.baseline_entry_count} baseline entr{'y' if diff.baseline_entry_count == 1 else 'ies'} "
        f"({baseline_path})."
    ]
    if diff.added:
        lines.append(f"NEW (not in baseline) — {len(diff.added)}:")
        for v in diff.added:
            lines.append(
                f"  {v.file}:{v.line} — {v.rule} (from {v.imported_module} import {', '.join(v.names) or '*'}; scope={v.scope})"
            )
        lines.append(
            "  Fix the import, or — if it is deliberate accepted debt — add/extend the entry in the "
            "baseline (symbols, count) with a disposition and rationale."
        )
    if diff.stale:
        lines.append(f"STALE (baseline overstates the debt) — {len(diff.stale)}:")
        moved = {(v.imported_module, v.scope) for v in diff.added}
        for st in diff.stale:
            entry = st.entry
            detail = f"baseline count {entry.count}, live {st.live_count}"
            if st.missing_symbols:
                detail += f"; symbols no longer imported: {', '.join(st.missing_symbols)}"
            hint = (
                " (a NEW finding shares module+scope — a move? carry the rationale over)"
                if (entry.imported_module, entry.scope) in moved
                else ""
            )
            lines.append(f"  {entry.key}: {detail} — shrink or remove the entry{hint}")
    if diff.unreviewed:
        lines.append(f"UNREVIEWED — {len(diff.unreviewed)}:")
        for entry in diff.unreviewed:
            lines.append(
                f"  {entry.key}: disposition={entry.disposition!r}, rationale={'set' if entry.rationale.strip() else 'EMPTY'}"
            )
    if diff.ok:
        lines.append("No new, stale, or unreviewed findings.")
    return "\n".join(lines)
