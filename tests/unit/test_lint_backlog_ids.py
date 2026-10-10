"""#1175: mechanization-backlog IDs are unique, and every backlog citation resolves (scripts/lint_backlog_ids.py)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import lint_backlog_ids as B

REGISTER = """# Outstanding

## Mechanization backlog

| # | Rule | Check | Axis |
|---|---|---|---|
| M1 | one | x | y |
| M2 | two | x | y |

### Closed

- **M3 — three**, closed 2026-10-08.
"""


def _repo(tmp_path: Path, register: str = REGISTER, files: dict[str, str] | None = None) -> Path:
    (tmp_path / "docs/plans").mkdir(parents=True)
    (tmp_path / B.REGISTER).write_text(register)
    for rel, text in (files or {}).items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def test_the_real_repo_is_clean() -> None:
    problems, checked = B.lint()
    assert problems == [] and checked > 50, (problems, checked)


def test_a_clean_register_and_resolving_citations_pass(tmp_path: Path) -> None:
    root = _repo(
        tmp_path, files={"CLAUDE.md": "Regression guard: process invariant — mechanization backlog M1 + M3.\n"}
    )
    assert B.lint(root) == ([], 2)


@pytest.mark.parametrize(
    ("register", "why"),
    [
        (REGISTER.replace("| M2 | two", "| M1 | two"), "M1 is defined 2 times"),  # duplicate in the table
        (REGISTER.replace("- **M3 — three**", "- **M2 — three**"), "M2 is defined 2 times"),  # table + Closed
        (REGISTER.replace("| M2 | two", "| M2b | two"), "malformed backlog ID 'M2b'"),
    ],
)
def test_a_register_defect_fails(tmp_path: Path, register: str, why: str) -> None:
    problems, _ = B.lint(_repo(tmp_path, register))
    assert any(why in p for p in problems), problems


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("See mechanization backlog M9.\n", "cites M9"),
        ("Backlog row M2 and M7.\n", "cites M7"),  # case, and a continuation in the same paragraph
        ("Rows in `docs/plans/outstanding.md`: M1, M2 and\nM8.\n", "cites M8"),  # wrapped list, context first
        ("**M7** ([outstanding](../plans/outstanding.md)).\n", "cites M7"),  # ID before its context
        ("mechanization backlog M1–M4\n", "cites M4"),  # a range member
        ("mechanization backlog M3–M1\n", "malformed backlog range"),
    ],
)
def test_an_unresolved_citation_fails(tmp_path: Path, text: str, why: str) -> None:
    problems, _ = B.lint(_repo(tmp_path, files={"docs/x.md": text}))
    assert any(why in p for p in problems), problems


@pytest.mark.parametrize(
    "text",
    [
        "The Stages M0–M4 above are superseded.\n",  # no backlog context: not a citation
        "M1b PR 5b ships; outstanding.md row M1 holds.\n",  # M1b is the PR series, not an ID
        "The M9b PR series, see outstanding.md.\n",  # a suffixed token never reads as M9 (undefined here)
        "mechanization backlog M2\n\nMac M9 Pro is a laptop.\n",  # context does not leak across paragraphs
    ],
)
def test_text_that_is_not_a_backlog_citation_passes(tmp_path: Path, text: str) -> None:
    problems, _ = B.lint(_repo(tmp_path, files={"docs/x.md": text}))
    assert problems == [], problems


def test_an_exempt_citation_is_skipped(tmp_path: Path, monkeypatch) -> None:
    root = _repo(tmp_path, files={"docs/old.md": "mechanization backlog M9\n"})
    monkeypatch.setattr(B, "EXEMPT", {("docs/old.md", "M9"): "a dated record, never relinked"})
    assert B.lint(root)[0] == []


@pytest.mark.parametrize("register", [None, "# nothing defined\n"])
def test_an_unusable_register_is_exit_2(tmp_path: Path, monkeypatch, register) -> None:
    root = _repo(tmp_path, register or REGISTER)
    if register is None:
        (root / B.REGISTER).unlink()
    monkeypatch.setattr(B, "REPO_ROOT", root)
    assert B.main() == 2


# ── review folds (2026-10-09) ─────────────────────────────────────────────────────────────────────────────


def test_the_registers_own_prose_citations_are_checked(tmp_path: Path) -> None:
    register = REGISTER.replace("| M2 | two |", "| M2 | two, with mechanization backlog M8 |")
    problems, _ = B.lint(_repo(tmp_path, register))
    assert any("cites M8" in p for p in problems), problems


def test_an_id_after_a_slash_is_read(tmp_path: Path) -> None:
    problems, _ = B.lint(_repo(tmp_path, files={"docs/x.md": "mechanization backlog M1/M9\n"}))
    assert any("cites M9" in p for p in problems), problems


@pytest.mark.parametrize("text", ["mechanization backlog M2 - 2026-10-08\n", "backlog row M1 – 3 checks\n"])
def test_a_hyphen_before_a_bare_number_is_not_a_range(tmp_path: Path, text: str) -> None:
    assert B.lint(_repo(tmp_path, files={"docs/x.md": text}))[0] == []


def test_a_non_numeric_first_cell_in_the_register_is_not_a_definition(tmp_path: Path) -> None:
    register = REGISTER + "\n| MAXIM_FOO | an env var table |\n- **Mechanization — a heading bullet**\n"
    assert B.lint(_repo(tmp_path, register))[0] == []


def test_this_lint_and_its_tests_are_skipped(tmp_path: Path) -> None:
    root = _repo(tmp_path, files={"tests/unit/test_lint_backlog_ids.py": "mechanization backlog M9\n"})
    assert B.lint(root)[0] == []


@pytest.mark.parametrize("text", ["mechanization backlog M1–9\n", "mechanization backlog M1—9\n"])
def test_a_range_missing_its_second_m_is_malformed(tmp_path: Path, text: str) -> None:
    problems, _ = B.lint(_repo(tmp_path, files={"docs/x.md": text}))
    assert any("malformed backlog range" in p for p in problems), problems


def test_an_em_dash_range_is_a_range(tmp_path: Path) -> None:
    problems, _ = B.lint(_repo(tmp_path, files={"docs/x.md": "mechanization backlog M1—M9\n"}))
    assert any("cites M4" in p for p in problems), problems
