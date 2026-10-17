"""No test may ``importlib.reload`` a module (#1350).

``reload`` re-runs a module's body inside the SAME module object: every class and function
it defines is replaced in place, so a name another test imported earlier (``from
maxim.runtime.lane_backends import BackendGateError``) now refers to a stale object. The
next ``pytest.raises(BackendGateError)`` misses the freshly defined class the reloaded code
raises, and the outcome depends on which files ran earlier on that worker. Serial file order
hid it; pytest-xdist exposed it (#1339). ``tests/script_module_guard.py`` cannot see this
class (identity is unchanged), so it is banned outright: reset module state with the module's
own helper or ``monkeypatch.setattr`` instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]


def _reload_calls(tree: ast.AST) -> list[int]:
    """Line numbers of reload calls: ``importlib.reload(...)`` (or an ``import importlib as X`` alias) and a
    bare ``reload(...)`` / alias bound by ``from importlib import reload [as Y]``."""
    module_names = {"importlib"}
    func_names: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            module_names |= {a.asname for a in n.names if a.name == "importlib" and a.asname}
        elif isinstance(n, ast.ImportFrom) and n.module == "importlib":
            func_names |= {a.asname or a.name for a in n.names if a.name == "reload"}
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "reload" and isinstance(fn.value, ast.Name):
            if fn.value.id in module_names:
                lines.append(node.lineno)
        elif isinstance(fn, ast.Name) and fn.id in func_names:
            lines.append(node.lineno)
    return sorted(lines)


def test_reload_calls_detects_both_spellings() -> None:
    src = (
        "import importlib\nfrom importlib import reload\nimportlib.reload(m)\nreload(m)\nobj.reload()\n"
        "import importlib as il\nfrom importlib import reload as rl\nil.reload(m)\nrl(m)\n'reload(m)'\n"
    )
    assert _reload_calls(ast.parse(src)) == [3, 4, 8, 9]


def test_no_test_reloads_a_module() -> None:
    offenders = []
    for path in sorted(TESTS.rglob("*.py")):
        if path == Path(__file__).resolve():
            continue
        lines = _reload_calls(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        offenders += [f"{path.relative_to(TESTS.parent)}:{ln}" for ln in lines]
    assert not offenders, (
        "importlib.reload replaces a module's classes/functions in place, so names other tests imported go "
        "stale and outcomes depend on file order (#1350). Reset state with the module's own helper or "
        "monkeypatch.setattr instead:\n  " + "\n  ".join(offenders)
    )
