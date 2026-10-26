"""Regression tests for the canonical-guidance and adapter lint."""

from pathlib import Path

import pytest

from scripts.lint_claude_md_invariants import EXPECTED_AGENTS_ADAPTER, lint


def _write_minimal_guidance_tree(root: Path, *, agents_text: str, ledger_text: str = "# Ledger\n") -> Path:
    """Create the smallest repo-shaped tree accepted by the guidance lint."""
    claude_path = root / "CLAUDE.md"
    claude_path.write_text("# CLAUDE.md\n")
    (root / "AGENTS.md").write_text(agents_text)
    (root / "docs" / "agents").mkdir(parents=True)
    (root / "docs" / "plans").mkdir(parents=True)
    (root / "docs" / "plans" / "behavioral_graduation_candidates.md").write_text(ledger_text)
    return claude_path


def test_pointer_only_agents_adapter_passes(tmp_path: Path) -> None:
    claude_path = _write_minimal_guidance_tree(
        tmp_path,
        agents_text=EXPECTED_AGENTS_ADAPTER,
    )

    assert lint(claude_path) == 0


def test_agents_adapter_drift_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    claude_path = _write_minimal_guidance_tree(
        tmp_path,
        agents_text=EXPECTED_AGENTS_ADAPTER + "\n## Duplicated rules\n",
    )

    assert lint(claude_path) == 1
    captured = capsys.readouterr()
    assert "AGENTS.md must remain the exact pointer-only adapter" in captured.err


# Check 5 (the ledger's EARNED rows) moved to scripts/lint_ledger_format.py on 2026-09-30 (M1b PR 3);
# its tests are in tests/unit/test_lint_ledger_format.py.


def _brief_citing(root: Path, citation: str) -> Path:
    claude = _write_minimal_guidance_tree(root, agents_text=EXPECTED_AGENTS_ADAPTER)
    (root / "src" / "maxim" / "runtime").mkdir(parents=True)
    (root / "src" / "maxim" / "runtime" / "loop.py").write_text(
        "from maxim.x import SHARED as SHARED\n\nclass Loop:\n    def tick(self):\n        pass\n\ndef build():\n    pass\n"
    )
    (root / "docs" / "agents" / "runtime.md").write_text(f"# Runtime\n\nThe seam is `{citation}`.\n")
    return claude


def test_a_citation_to_a_moved_symbol_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#1121: a brief citing ``path.py::symbol`` after the symbol moved (review caught one by hand) fails the lint."""
    assert lint(_brief_citing(tmp_path, "runtime/loop.py::_resolve_situation_cue")) == 1
    assert "runtime/loop.py::_resolve_situation_cue" in capsys.readouterr().err


@pytest.mark.parametrize(
    "citation",
    [
        "src/maxim/runtime/loop.py::build",  # repo-relative
        "runtime/loop.py::Loop.tick",  # relative to src/maxim, dotted
        "loop.py::SHARED",  # a bare file name, and a re-export counts as defined
    ],
)
def test_a_citation_that_resolves_passes(tmp_path: Path, citation: str) -> None:
    assert lint(_brief_citing(tmp_path, citation)) == 0


@pytest.mark.parametrize(
    "citation",
    [
        "runtime/loop.py::local_only",  # a local variable is not a definition
        "runtime/loop.py::Loop.build",  # build is module-level, not inside Loop
        "runtime/loop.py::os",  # a stdlib import inside a function body is not a module binding
        "runtime/loop.py::x",  # one-character symbols are checked too
    ],
)
def test_the_binding_rule_is_strict(tmp_path: Path, citation: str) -> None:
    claude = _brief_citing(tmp_path, citation)
    (tmp_path / "src" / "maxim" / "runtime" / "loop.py").write_text(
        "from maxim.x import SHARED as SHARED\n\nclass Loop:\n    def tick(self):\n        pass\n\n"
        "def build():\n    import os\n    local_only = 1\n    return local_only\n"
    )
    assert lint(claude) == 1


def test_an_exact_path_is_judged_alone(tmp_path: Path) -> None:
    """``report.py::f`` means ``src/maxim/report.py`` when that file exists, even if another ``report.py`` defines f."""
    claude = _brief_citing(tmp_path, "report.py::only_in_the_other")
    (tmp_path / "src" / "maxim" / "report.py").write_text("def unrelated():\n    pass\n")
    (tmp_path / "src" / "maxim" / "simulation").mkdir()
    (tmp_path / "src" / "maxim" / "simulation" / "report.py").write_text("def only_in_the_other():\n    pass\n")
    assert lint(claude) == 1
