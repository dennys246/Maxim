"""tests/conftest.py redirects HOME before any `maxim` import, or the suite reads the developer's real config.

Found 2026-10-27 calibrating the big-mac-mini suite runner: `from maxim.memory.encoding import EncodingSignals` sat
above the isolation block, and importing it reads `~/.config/maxim/profiles.yml` into `_BUILTIN_PROFILES` with the
REAL home. The rig's own `r1-distill-qwen-32b` profile then failed four `test_profile_metadata` cases, recorded since
2026-09-22 as "environment" failures. A machine whose real profiles happen to be clean (CI, most laptops) hides it.
"""

from __future__ import annotations

import ast
from pathlib import Path

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


def _home_redirect_line(tree: ast.Module) -> int:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and ast.unparse(target.value) == "os.environ"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "HOME"
                ):
                    return node.lineno
    raise AssertionError("tests/conftest.py no longer redirects HOME at module level")


def _maxim_imports(tree: ast.Module) -> list[int]:
    lines = []
    for node in tree.body:  # module level: the imports that run while conftest loads
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "maxim":
            lines.append(node.lineno)
        elif isinstance(node, ast.Import) and any(a.name.split(".")[0] == "maxim" for a in node.names):
            lines.append(node.lineno)
    return lines


def test_every_maxim_import_in_conftest_comes_after_the_home_redirect() -> None:
    tree = ast.parse(CONFTEST.read_text())
    home = _home_redirect_line(tree)
    early = [ln for ln in _maxim_imports(tree) if ln < home]
    assert not early, f"tests/conftest.py imports maxim at line(s) {early}, before HOME is redirected at line {home}"
