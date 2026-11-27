"""B2's prerequisite 1 (roadmap_1_3_x.md §1.3.6): the ``_llm_unavailable`` fallback leaves the loop's body for a
module with no bio imports.

Slice 6 moved it, with the rest of §2, into ``runtime/loop_intake.py``. Its OWN imports name no bio package, and since
the PR right after slice 6 (owner decision 2026-11-21) neither does its import closure: ``loop_controller`` imported
``safe_agent_name`` and ``tool_to_run`` from ``tool_dispatch`` (whose bio imports are #1150, B2's prerequisite 2), and
both helpers moved verbatim to the bio-free ``loop_state``. Both checks below are now green; the closure check was a
strict red gate until that move. What this gives B2: the roadmap's prerequisite as worded. B2 itself takes no
``runtime.*`` import (baseline_floors.md §B2), so its first PR still pulls the shared form of the fallback out of this
runtime leaf.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LEAF = REPO / "src" / "maxim" / "runtime" / "loop_intake.py"
# B2's bio denylist: baseline_floors.md §Guards' PLANNED ``utils/audit.py::LAYER_RULES["baselines"]`` (not in the code
# yet) minus ``runtime`` and ``agents``, which B2 itself may not import but this runtime leaf necessarily does; plus
# ``similarity`` and ``hivemind``. When B2 adds that rule, derive this tuple from it so the two lists cannot drift.
BIO = (
    "maxim.memory",
    "maxim.decisions",
    "maxim.embodiment",
    "maxim.harm",
    "maxim.default_network",
    "maxim.salience",
    "maxim.imagination",
    "maxim.integration",
    "maxim.reactions",
    "maxim.proprioception",
    "maxim.similarity",
    "maxim.hivemind",
)


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


def test_importing_the_intake_leaf_loads_no_bio_module() -> None:
    """A fresh interpreter (this process has the whole loop loaded), on this checkout's ``src``."""
    code = (
        "import json, sys; import maxim.runtime.loop_intake; "
        f"print(json.dumps(sorted(m for m in sys.modules if m.startswith({BIO!r}))))"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []
