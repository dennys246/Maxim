"""B2's prerequisite 1 (roadmap_1_3_x.md §1.3.6): the ``_llm_unavailable`` fallback leaves the loop's body for a
module with no bio imports.

Slice 6 moved it, with the rest of §2, into ``runtime/loop_intake.py``. Its OWN imports name no bio package (pinned
below). Importing it still loads bio modules, only through ``loop_controller``'s import of ``safe_agent_name`` and
``tool_to_run`` from ``tool_dispatch`` (whose own bio imports are #1150, B2's prerequisite 2). So the closure check is a
strict red gate that flips when that import is cut: by the PR right after slice 6 that moves those two helpers to a
bio-free leaf (owner decision 2026-11-21), or with #1150.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LEAF = REPO / "src" / "maxim" / "runtime" / "loop_intake.py"
BIO = ("maxim.memory", "maxim.decisions", "maxim.integration", "maxim.similarity", "maxim.hivemind")


def _imported_modules(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_the_intake_leaf_imports_no_bio_package_itself() -> None:
    bio = sorted(m for m in _imported_modules(LEAF) if m.startswith(BIO))
    assert bio == [], bio


@pytest.mark.xfail(
    strict=True,
    reason=(
        "loop_controller imports safe_agent_name/tool_to_run from tool_dispatch, which imports bio modules (#1150); "
        "cut by the PR right after slice 6 (owner decision 2026-11-21)"
    ),
)
def test_importing_the_intake_leaf_loads_no_bio_module() -> None:
    """A fresh interpreter (this process has the whole loop loaded), on this checkout's ``src``."""
    code = (
        "import json, sys; import maxim.runtime.loop_intake; "
        f"print(json.dumps(sorted(m for m in sys.modules if m.startswith({BIO!r}))))"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []
