"""No-growth ratchet for the god functions (roadmap 1.1.x item 16.4).

`src/maxim/utils/function_length_baseline.json` pins `run_agentic_loop`,
`start_simulation_mode` and `_main_impl` at their v1.1.0 AST spans (3,546 / 3,342 / 1,752).
Growth fails the fast suite; shrinkage ALSO fails until the ceiling is tightened in the same
commit, so the file never overstates the debt. Verified to fail on revert: with the baseline
edited to v1.1.0 - 1 for any entry the growth assertion fires; with a 3-line stub appended
to `_main_impl` it fires on the real file.
"""

from __future__ import annotations

import ast
import json
from importlib import resources
from pathlib import Path

import pytest

import maxim

REPO_SRC = Path(maxim.__file__).resolve().parent.parent
BASELINE = Path(maxim.__file__).resolve().parent / "utils" / "function_length_baseline.json"


def function_span(path: Path, name: str) -> int:
    """AST span (end_lineno - lineno + 1; decorators excluded) of the top-level function ``name``."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node.end_lineno - node.lineno + 1
    raise LookupError(f"{name} not found at module level in {path}")


def load_baseline() -> dict:
    data = json.loads(BASELINE.read_text())
    assert data["baseline_format_version"] == 1
    assert isinstance(data["entries"], list) and data["entries"], "baseline must list at least one function"
    for e in data["entries"]:
        assert set(e) == {"file", "function", "lines"}, e
        assert isinstance(e["lines"], int) and e["lines"] > 0
    return data


def test_baseline_ships_with_the_package() -> None:
    assert resources.files("maxim.utils").joinpath("function_length_baseline.json").is_file()


@pytest.mark.parametrize("entry", load_baseline()["entries"], ids=lambda e: e["function"])
def test_god_function_does_not_grow_and_baseline_does_not_overstate(entry: dict) -> None:
    actual = function_span(REPO_SRC / entry["file"], entry["function"])
    ceiling = entry["lines"]
    assert actual <= ceiling, (
        f"{entry['function']} grew to {actual} lines (ceiling {ceiling}, v1.1.0 baseline in {BASELINE.name}). "
        "The god functions do not grow (roadmap 1.1.x item 16.4): move the new logic into a module-level "
        "function with its own test, or — if the growth is a deliberate, reviewed exception — raise the "
        "ceiling in the same commit with a rationale in the PR."
    )
    assert actual == ceiling, (
        f"{entry['function']} is {actual} lines but the ceiling is {ceiling} — it shrank. Tighten "
        f"{BASELINE.name} to {actual} in the same commit so the baseline never overstates the debt."
    )


def test_function_span_counts_the_ast_span(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("@dec\ndef f():\n    a = 1\n\n    return a\n\n\ndef g():\n    pass\n")
    assert function_span(p, "f") == 4  # def line through `return`, decorator excluded
    assert function_span(p, "g") == 2
    with pytest.raises(LookupError):
        function_span(p, "h")
