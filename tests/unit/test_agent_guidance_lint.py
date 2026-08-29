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


LEDGER_HEADER = "# Ledger\n\n| Claim | Mechanism | Status |\n|---|---|---|\n"


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


# ── check 5: EARNED ledger rows cite their data (roadmap 1.1.x item 16.9) ──────
# Verified to fail 3/3 rows on the pre-fix ledger (L185, L186, L188).


def _ledger_lint(tmp_path: Path, row: str, *, data_file: str | None = None) -> int:
    claude_path = _write_minimal_guidance_tree(
        tmp_path, agents_text=EXPECTED_AGENTS_ADAPTER, ledger_text=LEDGER_HEADER + row + "\n"
    )
    if data_file:
        p = tmp_path / data_file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}\n")
    return lint(claude_path)


def test_earned_row_without_guard_fails(tmp_path: Path, capsys) -> None:
    assert _ledger_lint(tmp_path, "| X | mech | **EARNED** — Roy-2c |") == 1
    assert "no 'Regression guard:' field" in capsys.readouterr().err


def test_earned_row_with_guard_but_no_data_fails(tmp_path: Path, capsys) -> None:
    row = "| X | mech | **EARNED post-1.0** via Exp 9. **Regression guard:** [tests/unit/t.py](../../tests/unit/t.py) |"
    assert _ledger_lint(tmp_path, row) == 1
    assert "cites no docs/experiments/data/ path" in capsys.readouterr().err


def test_earned_row_with_resolving_data_link_passes(tmp_path: Path) -> None:
    row = "| X | mech | **EARNED** via Exp 9. **Regression guard:** [data](../experiments/data/9_results.jsonl) |"
    assert _ledger_lint(tmp_path, row, data_file="docs/experiments/data/9_results.jsonl") == 0


def test_earned_row_with_broken_data_link_fails(tmp_path: Path, capsys) -> None:
    row = "| X | mech | **EARNED** via Exp 9. **Regression guard:** [data](../experiments/data/gone.jsonl) |"
    assert _ledger_lint(tmp_path, row) == 1
    assert "does not resolve" in capsys.readouterr().err


def test_earned_row_with_dated_data_lost_annotation_passes(tmp_path: Path) -> None:
    row = "| X | mech | **EARNED (de facto)** — in-suite run. **Data lost (2026-08-29 annotation):** never archived. **Regression guard:** [t](../../tests/unit/t.py) |"
    assert _ledger_lint(tmp_path, row) == 0


def test_non_earned_rows_are_not_judged(tmp_path: Path) -> None:
    row = "| X | mech | **PARTIAL — reframed**; was EARNED, pulled 2026-08-01 |"
    assert _ledger_lint(tmp_path, row) == 0


def test_missing_ledger_is_exit_2(tmp_path: Path) -> None:
    claude_path = _write_minimal_guidance_tree(tmp_path, agents_text=EXPECTED_AGENTS_ADAPTER)
    (tmp_path / "docs/plans/behavioral_graduation_candidates.md").unlink()
    assert lint(claude_path) == 2
