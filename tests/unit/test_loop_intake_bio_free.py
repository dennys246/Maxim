"""B2's prerequisite 1 (roadmap_1_3_x.md §1.3.6): the ``_llm_unavailable`` fallback leaves the loop's body for a
module with no bio imports.

Slice 6 moved it, with the rest of §2, into ``runtime/loop_intake.py``. Its OWN imports name no bio package (pinned
below). Importing it still loads bio modules through ``loop_controller`` -> ``tool_dispatch``, whose bio imports
are #1150 (B2's prerequisite 2), so the closure check is a strict red gate that flips when #1150 lands.
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
    reason="#1150: loop_controller imports tool_dispatch, which imports decisions.causal_link and bio_integration",
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
