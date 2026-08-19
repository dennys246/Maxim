"""Regression tests for the canonical-guidance and adapter lint."""

from pathlib import Path

import pytest

from scripts.lint_claude_md_invariants import EXPECTED_AGENTS_ADAPTER, lint


def _write_minimal_guidance_tree(root: Path, *, agents_text: str) -> Path:
    """Create the smallest repo-shaped tree accepted by the guidance lint."""
    claude_path = root / "CLAUDE.md"
    claude_path.write_text("# CLAUDE.md\n")
    (root / "AGENTS.md").write_text(agents_text)
    (root / "docs" / "agents").mkdir(parents=True)
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
