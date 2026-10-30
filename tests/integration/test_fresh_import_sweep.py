"""Every ``maxim`` module imports on its own, in a fresh interpreter (#1315 step 0a, V9; atonia notes §1A.1).

The package inits are lazy (PEP 562), so importing one module no longer imports its whole package first. A cycle
that the old eager inits hid surfaces only when a module is the FIRST thing a process imports (step 0a found one:
``agents/base.py``'s bottom-of-module ``GoalAgent`` re-export). One child per module, so import order in a shared
process cannot mask a cycle. ``slow``: one interpreter per module.
"""

from __future__ import annotations

import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

# ``maxim.__main__`` runs the CLI on import: it is the entry point, not a library module.
SKIP = frozenset({"maxim.__main__"})

_CHILD = (
    "import importlib, json, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "fails = {}\n"
    "for name in json.loads(sys.argv[2]):\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "    except Exception as exc:\n"
    "        fails[name] = f'{type(exc).__name__}: {exc}'[:300]\n"
    "print(json.dumps(fails))\n"
)


def _modules() -> list[str]:
    sys.path.insert(0, str(SRC))
    try:
        import maxim

        return sorted(m.name for m in pkgutil.walk_packages(maxim.__path__, "maxim.") if m.name not in SKIP)
    finally:
        sys.path.remove(str(SRC))


@pytest.mark.slow
def test_every_module_imports_first_in_a_fresh_process(tmp_path: Path) -> None:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path)}
    fails: dict[str, str] = {}
    for name in _modules():
        proc = subprocess.run(  # noqa: S603 -- our own interpreter, our own script
            [sys.executable, "-I", "-c", _CHILD, str(SRC), json.dumps([name])],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        lines = proc.stdout.strip().splitlines()
        if proc.returncode != 0 or not lines:
            fails[name] = f"child exited {proc.returncode}: {proc.stderr[-300:]}"
            continue
        fails.update(json.loads(lines[-1]))
    assert not fails, "modules that fail when imported first:\n  " + "\n  ".join(f"{k}: {v}" for k, v in fails.items())
