"""Architecture audit: verify layer ownership and one-way dependency rules."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditViolation:
    """A single architecture rule violation."""

    file: str
    line: int
    layer: str
    imported_module: str
    rule: str  # e.g., "agents must_not_import tools"


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

        for py_file in layer_dir.rglob("*.py"):
            file_violations = _check_file(py_file, layer, forbidden, src_root)
            violations.extend(file_violations)

    return violations


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

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violation = _check_import_name(alias.name, layer, forbidden_layers, rel_path, node.lineno)
                if violation:
                    violations.append(violation)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                violation = _check_import_name(node.module, layer, forbidden_layers, rel_path, node.lineno)
                if violation:
                    violations.append(violation)

    return violations


def _check_import_name(
    import_name: str,
    layer: str,
    forbidden_layers: list[str],
    file_path: str,
    line: int,
) -> AuditViolation | None:
    """Check if an import name violates layer rules."""
    # Normalize: handle both "maxim.tools.base" and "tools.base"
    parts = import_name.split(".")
    # Find the layer part of the import
    for i, part in enumerate(parts):
        if part == "maxim" and i + 1 < len(parts):
            target_layer = parts[i + 1]
        elif part in LAYER_RULES and i == 0:
            target_layer = part
        else:
            continue

        if target_layer in forbidden_layers:
            return AuditViolation(
                file=file_path,
                line=line,
                layer=layer,
                imported_module=import_name,
                rule=f"{layer} must_not_import {target_layer}",
            )
    return None
