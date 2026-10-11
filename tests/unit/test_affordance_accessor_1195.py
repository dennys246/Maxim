"""#1195: an affordance tool's declaration and cause are read through a PUBLIC accessor, never its privates.

GL2a (#1194) read ``ModulatorAffordanceTool._affordance_schema`` / ``_entity`` / ``_affordance_name`` from the
executor by ``getattr``, as ``tools/discovery.py`` and ``simulation/foundry.py`` did: a rename of any of them
would leave an empty drive block, an empty cause or a skipped affordance, silently. The tool now exposes
read-only ``affordance_schema`` / ``acting_entity`` / ``affordance_name`` and the readers use them.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
_PRIVATE = {"_affordance_name", "_affordance_schema"}
_PRIVATE_READS = {"_entity", "_modulator"}  # common names: flagged only as non-``self`` READS (tests assign them)
_PRIVATE_BY_NAME = _PRIVATE | {"_entity"}  # ``getattr(tool, "_entity")``: ``_entity`` is common, so by string only


def _private_reads() -> list[str]:
    """Reads of an affordance tool's private attributes outside ``embodiment/tool_bridge.py`` (its owner), in
    ``src/``, ``tests/`` and ``scripts/``:
    ``x._affordance_*`` on anything but ``self``, and ``getattr``/``hasattr`` with one of the names."""
    found: list[str] = []
    paths = [
        *(ROOT / "src" / "maxim").rglob("*.py"),
        *(ROOT / "tests").rglob("*.py"),
        *(ROOT / "scripts").rglob("*.py"),
    ]
    for path in sorted(paths):
        rel = str(path.relative_to(ROOT))
        # The owner, and this file (its name lists): a test double must mock the accessor, not the private.
        if rel in ("src/maxim/embodiment/tool_bridge.py", "tests/unit/test_affordance_accessor_1195.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in _PRIVATE
                and not (isinstance(node.value, ast.Name) and node.value.id == "self")
            ):
                found.append(f"{rel}:{node.lineno} .{node.attr}")
            if (
                isinstance(node, ast.Attribute)
                and node.attr in _PRIVATE_READS
                and isinstance(node.ctx, ast.Load)
                and not (isinstance(node.value, ast.Name) and node.value.id == "self")
            ):
                found.append(f"{rel}:{node.lineno} .{node.attr}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr")
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in _PRIVATE_BY_NAME
            ):
                found.append(f"{rel}:{node.lineno} {node.func.id}(..., {node.args[1].value!r})")
    return found


def test_no_module_reads_an_affordance_tools_privates() -> None:
    assert _private_reads() == []


def test_the_tool_exposes_read_only_accessors() -> None:
    from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

    for name in ("affordance_name", "affordance_schema", "acting_entity", "modulator"):
        attr = getattr(ModulatorAffordanceTool, name, None)
        assert isinstance(attr, property) and attr.fset is None, name
