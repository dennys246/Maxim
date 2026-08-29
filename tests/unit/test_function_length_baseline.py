"""No-growth ratchet for the god functions (roadmap 1.1.x item 16.4).

`src/maxim/utils/function_length_baseline.json` (co-located with the code it measures;
deliberately NOT package data — nothing reads it at runtime) pins `run_agentic_loop`,
`start_simulation_mode` and `_main_impl` at their v1.1.0 AST spans (3,546 / 3,342 / 1,752).
Growth fails the fast suite; shrinkage ALSO fails until the ceiling is tightened in the same
commit, so the file never overstates the debt. Verified to fail on revert: with the baseline
edited to v1.1.0 - 1 for any entry the growth assertion fires; with a 3-line stub appended
to `_main_impl` it fires on the real file.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

# The CHECKOUT this test file lives in — never `Path(maxim.__file__)`. A worktree
# session without `export PYTHONPATH="$PWD/src"` imports the MAIN checkout's maxim
# (CLAUDE.md's worktree rule + the Exp 42b "assert the code under test is YOUR repo"
# lesson), and this guard would then silently measure another tree's functions.
REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SRC = REPO_ROOT / "src"
BASELINE = REPO_SRC / "maxim" / "utils" / "function_length_baseline.json"


def function_span(path: Path, name: str) -> int:
    """AST span (end_lineno - lineno + 1; decorators excluded) of the top-level function ``name``.

    A conditional redefinition would make "the first one" the wrong answer, so an
    ambiguous name is an error rather than a silently-measured guess.
    """
    tree = ast.parse(path.read_text())
    found = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if not found:
        raise LookupError(f"{name} not found at module level in {path}")
    if len(found) > 1:
        raise LookupError(f"{name} is defined {len(found)}x at module level in {path} — ambiguous span")
    return found[0].end_lineno - found[0].lineno + 1


def load_baseline() -> dict:
    data = json.loads(BASELINE.read_text())
    assert data["baseline_format_version"] == 1
    assert isinstance(data["entries"], list) and data["entries"], "baseline must list at least one function"
    for e in data["entries"]:
        assert set(e) == {"file", "function", "lines"}, e
        assert isinstance(e["lines"], int) and e["lines"] > 0
    return data


def test_baseline_declares_its_review_metadata() -> None:
    """The D19 precedent fails on UNREVIEWED entries; the analogue here is that the
    file says WHEN it was measured and reviewed, so a stale ceiling is visible."""
    data = load_baseline()
    assert data["measured_at"] == "v1.1.0"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["reviewed"]), data["reviewed"]
    assert "no writer" in data["_comment"]


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
    p.write_text("def f():\n    pass\n\n\ndef f():\n    pass\n")
    with pytest.raises(LookupError, match="ambiguous span"):
        function_span(p, "f")
