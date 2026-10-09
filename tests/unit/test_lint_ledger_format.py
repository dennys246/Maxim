"""M1b PR 3 -- the ledger format lint (scripts/lint_ledger_format.py, parser scripts/_ledger.py).

Every rule is pinned against a throwaway git repo, so tracked-file, symlink and merge-base checks run for
real. Absorbs the former ``lint_claude_md_invariants`` check 5 tests (EARNED rows cite their data).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts import lint_ledger_format as F

L = F.L  # the parser module object the lint itself imports: `scripts._ledger` would be a second copy (#1012)

REPO = Path(__file__).resolve().parents[2]
T1_HEADER = "| ID | Claim | Bio-mechanism | Status |\n| --- | --- | --- | --- |\n"
T3_HEADER = (
    "| ID | CLAUDE.md ref | Mechanism | Bio-claim | Graduation predicate | Status |\n"
    "| --- | --- | --- | --- | --- | --- |\n"
)
GUARD = " **Regression guard:** `tests/unit/test_x.py`."
DATA = "../experiments/data/"
OK_T1 = f"| T1-1 | claim | mech | **Status: EARNED 2026-09-01**. **Evidence:** [r.jsonl]({DATA}r.jsonl).{GUARD} |"
OK_T3 = "| T3-1 | L1 | mech | bio | pred | **Status: DROPPED 2026-08-30**. history |"
TODAY = "2026-09-30"


@pytest.fixture(autouse=True)
def _no_real_pins_on_synthetic_ledgers(request, monkeypatch):
    """GRANDFATHERED_QUALIFIERS pins rows of the REAL ledger; a synthetic one has none of them, so the pin-orphan check
    would fire in every fixture test. Tests reading the real ledger keep the real pins."""
    if "real_ledger" not in request.node.name:
        monkeypatch.setattr(F, "GRANDFATHERED_QUALIFIERS", {})


def _git(repo: Path, *args: str, date: str = "2026-09-29T12:00:00Z") -> str:
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True, env=env).stdout


def _ledger(t1: list[str], t3: list[str] | None = None) -> str:
    t3 = [OK_T3] if t3 is None else t3
    return "# Ledger\n\n" + T1_HEADER + "\n".join(t1) + "\n\ntext\n\n" + T3_HEADER + "\n".join(t3) + "\n"


def _repo(tmp_path: Path, ledger: str, files: dict[str, str] | None = None) -> Path:
    """A repo whose base commit holds ``ledger``; returns the root (the base is HEAD)."""
    repo = tmp_path / "repo"
    (repo / "docs/plans").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    files = {"docs/experiments/data/r.jsonl": "{}\n", "tests/unit/test_x.py": "x = 1\n", **(files or {})}
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    (repo / L.LEDGER_PATH).write_text(ledger)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _lint(repo: Path, ledger: str | None = None) -> list[str]:
    if ledger is not None:
        (repo / L.LEDGER_PATH).write_text(ledger)
    failures, _ = F.lint(repo, base=_git(repo, "rev-parse", "HEAD").strip(), today=TODAY)
    return failures


def _one(failures: list[str], needle: str) -> None:
    assert failures and any(needle in f for f in failures), failures


# ── the real ledger ──────────────────────────────────────────────────────


def test_the_real_ledger_is_clean(monkeypatch) -> None:
    """The format rules only (base = HEAD): the unit-test job's shallow checkout has no merge-base, and the
    lint job (full history) runs the diff-scoped rules on every pull request."""
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    failures, code = F.lint(REPO, base="HEAD")
    assert failures == [] and code == 0


def test_the_real_ledger_parses_every_row() -> None:
    rows, problems = L.parse((REPO / L.LEDGER_PATH).read_text())
    assert problems == [] and all(not r.problems and r.token in L.RANK for r in rows)
    assert {r.id for r in rows} >= {"T1-1", "T1-15", "T3-1", "T3-20"}


def test_a_clean_minimal_ledger_passes(tmp_path: Path) -> None:
    assert _lint(_repo(tmp_path, _ledger([OK_T1]))) == []


# ── structure ────────────────────────────────────────────────────────────


def test_a_missing_table_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _ledger([OK_T1]))
    _one(_lint(repo, "# Ledger\n\n" + T1_HEADER + OK_T1 + "\n"), "T3 status table")


def test_a_blank_line_inside_a_table_fails(tmp_path: Path) -> None:
    row2 = OK_T1.replace("T1-1", "T1-2")
    repo = _repo(tmp_path, _ledger([OK_T1]))
    failures = _lint(repo, _ledger([OK_T1 + "\n", row2]))
    _one(failures, "split by a blank line")
    _one(failures, "status-shaped row outside")


def test_an_unescaped_pipe_in_code_breaks_the_row(tmp_path: Path) -> None:
    bad = OK_T1.replace("claim", "claim `|x|`")
    _one(_lint(_repo(tmp_path, _ledger([OK_T1])), _ledger([bad])), "cells, header has 4")
    escaped = OK_T1.replace("claim", "claim `\\|x\\|`")
    assert _lint(_repo(tmp_path / "b", _ledger([OK_T1])), _ledger([escaped])) == []


def test_ids_are_unique_and_prefixed(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _ledger([OK_T1]))
    _one(_lint(repo, _ledger([OK_T1, OK_T1])), "duplicate ID T1-1")
    _one(_lint(repo, _ledger([OK_T1.replace("T1-1", "T3-9")])), "is not a T1-<n> ID")


# ── the status line ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "status, needle",
    [
        ("**EARNED** via Exp 9.", "does not open with **Status:"),
        ("**Status: GREAT 2026-09-01**.", "not in the vocabulary"),
        ("**Status: DROPPED 2026-02-30**.", "not a calendar date"),
        ("**Status: DROPPED 2026-10-01**.", "after today"),
    ],
)
def test_the_status_line(tmp_path: Path, status: str, needle: str) -> None:
    row = f"| T1-1 | claim | mech | {status} |"
    _one(_lint(_repo(tmp_path, _ledger([OK_T1])), _ledger([row])), needle)


# ── evidence ─────────────────────────────────────────────────────────────


def _ev(evidence: str, token: str = "EARNED") -> str:
    return f"| T1-1 | claim | mech | **Status: {token} 2026-09-01**. **Evidence:** {evidence}.{GUARD} |"


@pytest.mark.parametrize(
    "evidence, needle",
    [
        ("", "not a link or a code span"),
        (f"[gone.jsonl]({DATA}gone.jsonl)", "not a tracked file or directory"),
        (f"[README.md]({DATA}README.md)", "context, not evidence"),
        (f"[s.py]({DATA}s.py)", "context, not evidence"),
        (f"[x.jsonl]({DATA}run_aborted/x.jsonl)", "not evidence"),
        (f"[d.jsonl]({DATA}9_Dry_Run.jsonl)", "not evidence"),
        (f"[data]({DATA})", "outside docs/experiments/data/"),
        (f"[sess]({DATA}sessions)", "session directory holding a report.json"),
        (f"[r.jsonl]({DATA}r.jsonl#L3)", "no anchor"),
        ("`docs/plans/other.md`", "outside docs/experiments/data/"),
        (f"[other.jsonl]({DATA}r.jsonl)", "names a different file than its target"),
        (f"[a]({DATA}r.jsonl) / [b]({DATA}r.jsonl)", "must end with"),
    ],
)
def test_evidence_entries_are_records(tmp_path: Path, evidence: str, needle: str) -> None:
    files = {
        "docs/experiments/data/README.md": "x",
        "docs/experiments/data/s.py": "x",
        "docs/experiments/data/run_aborted/x.jsonl": "{}",
        "docs/experiments/data/9_Dry_Run.jsonl": "{}",
        "docs/experiments/data/sessions/log.txt": "x",
    }
    _one(_lint(_repo(tmp_path, _ledger([OK_T1]), files), _ledger([_ev(evidence)])), needle)


def test_a_session_directory_with_a_report_is_evidence(tmp_path: Path) -> None:
    files = {"docs/experiments/data/s1/report.json": "{}", "docs/experiments/data/s1/actions.jsonl": "{}"}
    row = _ev(f"[s1]({DATA}s1); `docs/experiments/data/r.jsonl`")
    assert _lint(_repo(tmp_path, _ledger([OK_T1]), files), _ledger([row])) == []


def test_a_symlink_is_not_evidence(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _ledger([OK_T1]))
    (repo / "docs/experiments/data/link.jsonl").symlink_to("r.jsonl")
    _git(repo, "add", "-A")
    _one(_lint(repo, _ledger([_ev(f"[link.jsonl]({DATA}link.jsonl)")])), "a symlink is not evidence")


def test_an_untracked_file_is_not_evidence(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _ledger([OK_T1]))
    (repo / "docs/experiments/data/new.jsonl").write_text("{}")
    _one(_lint(repo, _ledger([_ev(f"[new.jsonl]({DATA}new.jsonl)")])), "not a tracked file")


def test_a_positive_status_needs_evidence(tmp_path: Path) -> None:
    row = f"| T1-1 | claim | mech | **Status: MAINTAINED 2026-09-01** (narrow).{GUARD} |"
    _one(_lint(_repo(tmp_path, _ledger([OK_T1])), _ledger([row])), "needs an **Evidence:** field")


def test_evidence_must_follow_the_status_line_once(tmp_path: Path) -> None:
    late = (
        f"| T1-1 | claim | mech | **Status: EARNED 2026-09-01**. prose **Evidence:** [r.jsonl]({DATA}r.jsonl).{GUARD} |"
    )
    twice = OK_T1.replace(GUARD, f" **Evidence:** [r.jsonl]({DATA}r.jsonl).{GUARD}")
    repo = _repo(tmp_path, _ledger([OK_T1]))
    _one(_lint(repo, _ledger([late])), "straight after the Status line")
    _one(_lint(repo, _ledger([twice])), "more than one **Evidence:** field")


def test_by_tests_cites_test_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _ledger([OK_T1]))
    assert _lint(repo, _ledger([_ev("`tests/unit/test_x.py`", "RE-VALIDATED-BY-TESTS")])) == []
    _one(_lint(repo, _ledger([_ev(f"[r.jsonl]({DATA}r.jsonl)", "RE-VALIDATED-BY-TESTS")])), "tests/**/test_*.py")


# ── the guard field, SUPERSEDED ──────────────────────────────────────────


@pytest.mark.parametrize("token", ["EARNED", "REPRODUCED", "LEGACY"])
def test_claims_need_a_regression_guard(tmp_path: Path, token: str) -> None:
    ev = f" **Evidence:** [r.jsonl]({DATA}r.jsonl)." if token != "LEGACY" else ""
    row = f"| T1-1 | claim | mech | **Status: {token} 2026-09-01**.{ev} history |"
    _one(_lint(_repo(tmp_path, _ledger([OK_T1])), _ledger([row])), "no 'Regression guard:' field")


@pytest.mark.parametrize(
    "pointer, ok",
    [("by T1-1", True), ("by T1-2", False), ("by T1-9", False), ("replaced", False), ("by T1-3", False)],
)
def test_superseded_names_a_live_row(tmp_path: Path, pointer: str, ok: bool) -> None:
    sup = f"| T1-2 | old | mech | **Status: SUPERSEDED 2026-08-25** {pointer}. |"
    sup3 = "| T1-3 | older | mech | **Status: SUPERSEDED 2026-08-25** by T1-1. |"
    failures = _lint(_repo(tmp_path, _ledger([OK_T1, sup, sup3])), _ledger([OK_T1, sup, sup3]))
    assert (failures == []) is ok, failures


# ── against the merge-base ───────────────────────────────────────────────


def _base_then(tmp_path: Path, base_rows: list[str], new_rows: list[str]) -> list[str]:
    return _lint(_repo(tmp_path, _ledger(base_rows)), _ledger(new_rows))


def test_an_id_never_vanishes(tmp_path: Path) -> None:
    row2 = OK_T1.replace("T1-1", "T1-2")
    _one(_base_then(tmp_path, [OK_T1, row2], [OK_T1]), "T1-2: a row ID in the base is gone")


def test_a_date_never_moves_back(tmp_path: Path) -> None:
    old = OK_T1.replace("2026-09-01", "2020-01-01")
    _one(_base_then(tmp_path, [OK_T1], [old]), "date moved back")


def test_a_raise_needs_a_later_date_and_a_lowering_may_keep_it(tmp_path: Path) -> None:
    partial = "| T1-1 | claim | mech | **Status: PARTIAL 2026-09-01**. |"
    _one(_base_then(tmp_path, [partial], [OK_T1]), "asserts something new and needs a date after 2026-09-01")
    stale = "| T1-1 | claim | mech | **Status: STALE 2026-09-01**. |"
    assert _base_then(tmp_path / "b", [OK_T1], [stale]) == []


def test_legacy_is_kept_never_entered(tmp_path: Path) -> None:
    legacy = f"| T1-1 | claim | mech | **Status: LEGACY 2026-09-01**.{GUARD} |"
    partial = "| T1-1 | claim | mech | **Status: PARTIAL 2026-09-01**. |"
    _one(_base_then(tmp_path, [partial], [legacy]), "LEGACY can be kept, never entered")
    assert _base_then(tmp_path / "b", [legacy], [legacy]) == []
    _one(_base_then(tmp_path / "c", [OK_T1], [OK_T1, legacy.replace("T1-1", "T1-2")]), "never entered")


STALE_1 = "| T1-1 | claim | mech | **Status: STALE 2026-09-01**. |"
SETUP_2 = "| T1-2 | new claim | mech | **Status: SETUP 2026-09-01**. |"
EARNED_2 = (
    f"| T1-2 | new claim (replaces T1-1 and T1-3) | mech | **Status: EARNED 2026-09-29**. "
    f"**Evidence:** [r.jsonl]({DATA}r.jsonl).{GUARD} |"
)
RETIRED_1 = "| T1-1 | claim | mech | **Status: SUPERSEDED 2026-09-29** by T1-2. |"


def _exceptions_then(tmp_path: Path, clauses: list, base_rows: list[str], new_rows: list[str]) -> list[str]:
    """``clauses`` are committed on the base (only clauses on main act)."""
    import json

    files = {"docs/experiments/evidence_exceptions.json": json.dumps(clauses)}
    return _lint(_repo(tmp_path, _ledger(base_rows), files), _ledger(new_rows))


@pytest.mark.parametrize("was", ["STALE", "BROKEN", "PARTIAL"])
def test_a_successor_earned_in_the_same_diff_retires_a_row(tmp_path: Path, was: str) -> None:
    """D3: the verdict PR raises the new row and retires the old one together (Exp 63's PASS row)."""
    old = STALE_1.replace("STALE", was)
    assert _base_then(tmp_path, [old, SETUP_2], [RETIRED_1, EARNED_2]) == []


@pytest.mark.parametrize(
    "successor_base, successor_head",
    [
        ("SETUP", "SETUP"),  # never earned
        ("SETUP", "STALE"),  # not positive
        ("EARNED", "EARNED"),  # positive, but already at the base: an old verdict cannot retire a new claim
    ],
)
def test_superseded_needs_its_successor_to_reach_positive_in_the_same_diff(
    tmp_path: Path, successor_base: str, successor_head: str
) -> None:
    def row(token: str) -> str:
        return EARNED_2 if token == "EARNED" else SETUP_2.replace("SETUP 2026-09-01", f"{token} 2026-09-01")

    failures = _base_then(tmp_path, [STALE_1, row(successor_base)], [RETIRED_1, row(successor_head)])
    _one(failures, "T1-1: STALE -> SUPERSEDED by T1-2 needs that successor to REACH a positive status")


def test_the_successor_must_name_the_row_it_supersedes(tmp_path: Path) -> None:
    """Arch NIT: an earned successor that does not name the retired row cannot retire it (T1-16 names T1-1); a
    longer id (T1-16) is not a mention of T1-1."""
    silent = EARNED_2.replace("(replaces T1-1 and T1-3)", "(replaces T1-16)")
    _one(_base_then(tmp_path, [STALE_1, SETUP_2], [RETIRED_1, silent]), "its row naming T1-1 (it does not)")


def test_a_new_row_entering_superseded_needs_a_successor_earned_in_the_diff(tmp_path: Path) -> None:
    new = "| T1-3 | claim | mech | **Status: SUPERSEDED 2026-09-29** by T1-2. |"
    _one(_base_then(tmp_path, [OK_T1, SETUP_2], [OK_T1, SETUP_2, new]), "a new row -> SUPERSEDED by T1-2")
    assert _base_then(tmp_path / "b", [OK_T1, SETUP_2], [OK_T1, EARNED_2, new]) == []


def test_re_pointing_a_superseded_row_is_checked_too(tmp_path: Path) -> None:
    sup = "| T1-3 | old | mech | **Status: SUPERSEDED 2026-08-25** by T1-1. |"
    moved = "| T1-3 | old | mech | **Status: SUPERSEDED 2026-09-29** by T1-2. |"
    _one(_base_then(tmp_path, [OK_T1, SETUP_2, sup], [OK_T1, SETUP_2, moved]), "SUPERSEDED -> SUPERSEDED by T1-2")
    assert _base_then(tmp_path / "b", [OK_T1, SETUP_2, sup], [OK_T1, EARNED_2, moved]) == []


def test_a_kept_superseded_row_is_not_rejudged(tmp_path: Path) -> None:
    """Legitimate history (T1-8 -> T1-9): unchanged, it passes; a successor that later goes STALE is its own row's
    problem."""
    sup = "| T1-2 | old | mech | **Status: SUPERSEDED 2026-08-25** by T1-1. |"
    assert _base_then(tmp_path, [OK_T1, sup], [OK_T1, sup]) == []
    assert _base_then(tmp_path / "b", [OK_T1, sup], [STALE_1.replace("2026-09-01", "2026-09-29"), sup]) == []


def _clause(**change) -> dict:
    clause = {"id": "s1", "kind": "superseded", "row": "T1-1", "from": "STALE", "by": "T1-2",
              "to_date": "2026-09-29", "owner": "o", "reason": "r", "date": "2026-09-28"}  # fmt: skip
    return {**clause, **change}


def test_a_superseded_clause_on_main_excepts_exactly_its_transition(tmp_path: Path) -> None:
    assert _exceptions_then(tmp_path, [_clause()], [STALE_1, SETUP_2], [RETIRED_1, SETUP_2]) == []
    for i, change in enumerate(
        ({"by": "T1-3"}, {"from": "BROKEN"}, {"to_date": "2026-09-28"}, {"row": "T1-4"}, {"reason": ""})
    ):
        failures = _exceptions_then(tmp_path / str(i), [_clause(**change)], [STALE_1, SETUP_2], [RETIRED_1, SETUP_2])
        _one(failures, "needs that successor to REACH a positive status")


def test_a_superseded_clause_added_in_the_same_diff_does_not_act(tmp_path: Path) -> None:
    import json

    repo = _repo(tmp_path, _ledger([STALE_1, SETUP_2]))
    (repo / "docs/experiments/evidence_exceptions.json").write_text(json.dumps([_clause()]))
    _one(_lint(repo, _ledger([RETIRED_1, SETUP_2])), "needs that successor to REACH a positive status")


def test_the_clause_fields_are_the_evidence_gates() -> None:
    from scripts import lint_evidence_gate as G

    assert F.SUPERSEDED_CLAUSE_FIELDS == G.SUPERSEDED_FIELDS and F.EXCEPTIONS == G.EXCEPTIONS
    assert G.exceptions_problems([], [_clause()]) == []
    assert G.exceptions_problems([], [_clause(by="")]) and G.exceptions_problems([], [_clause(kind="other")])


def test_a_new_row_is_not_backdated(tmp_path: Path) -> None:
    new = OK_T1.replace("T1-1", "T1-2").replace("2026-09-01", "2026-09-28")
    _one(_base_then(tmp_path, [OK_T1], [OK_T1, new]), "before the branch point (2026-09-29)")
    fresh = new.replace("2026-09-28", "2026-09-29")
    assert _base_then(tmp_path / "b", [OK_T1], [OK_T1, fresh]) == []


def test_a_pre_format_base_skips_only_the_date_rules(tmp_path: Path) -> None:
    legacy_row = "| T1-1 | claim | mech | **EARNED** via Exp 9 |"
    legacy_t3 = "| 1 | L1 | mech | bio | pred | **Dropped 2026-08-30** — history |"
    repo = _repo(tmp_path, _ledger([legacy_row], [legacy_t3]))
    assert _lint(repo, _ledger([OK_T1])) == []  # the new row's date predates the base: not judged yet


def test_a_ledger_missing_at_the_base_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _ledger([OK_T1]))
    _git(repo, "rm", "-q", L.LEDGER_PATH)
    _git(repo, "commit", "-q", "-m", "moved")
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / L.LEDGER_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / L.LEDGER_PATH).write_text(_ledger([OK_T1]))
    failures, _ = F.lint(repo, base=base, today=TODAY)
    _one(failures, "does not exist at the merge-base")


def test_a_missing_ledger_is_exit_2(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _ledger([OK_T1]))
    (repo / L.LEDGER_PATH).unlink()
    assert F.lint(repo, base="HEAD", today=TODAY)[1] == 2


def test_an_unreadable_base_is_an_error_on_a_pull_request(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path, _ledger([OK_T1]))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    failures, code = F.lint(repo, base="0" * 40, today=TODAY)
    assert code == 2


# ── review-round folds ───────────────────────────────────────────────────


def test_a_move_between_positive_tokens_needs_a_new_date(tmp_path: Path) -> None:
    maintained = OK_T1.replace("EARNED", "MAINTAINED")
    _one(_base_then(tmp_path, [maintained], [OK_T1]), "asserts something new and needs a date after")
    later = OK_T1.replace("2026-09-01", "2026-09-02")
    assert _base_then(tmp_path / "b", [maintained], [later]) == []


@pytest.mark.parametrize("prefix", ["   ", "> ", " > "])
def test_an_indented_or_quoted_status_row_cannot_hide(tmp_path: Path, prefix: str) -> None:
    hidden = prefix + "| T1-2 | c | m | **Status: EARNED 2026-09-01**. no evidence |"
    failures = _lint(_repo(tmp_path, _ledger([OK_T1])), _ledger([OK_T1, hidden]))
    _one(failures, "needs an **Evidence:** field")


def test_a_status_cell_outside_the_tables_is_caught(tmp_path: Path) -> None:
    stray = "\n| x | **Status: EARNED 2026-09-01** |\n"
    _one(_lint(_repo(tmp_path, _ledger([OK_T1])), _ledger([OK_T1]) + stray), "status-shaped row outside")


@pytest.mark.parametrize("name", ["exp_dry-run.jsonl", "a_non-frozen.jsonl", "x_INVALID.jsonl", "nonfrozen/r.jsonl"])
def test_every_spelling_of_a_non_gated_capture_is_refused(tmp_path: Path, name: str) -> None:
    files = {f"docs/experiments/data/{name}": "{}"}
    row = _ev(f"[{name.split('/')[-1]}]({DATA}{name})")
    _one(_lint(_repo(tmp_path, _ledger([OK_T1]), files), _ledger([row])), "not evidence")


def test_the_refused_markers_cover_the_prereg_lints() -> None:
    from scripts.lint_prereg_precedes_data import NON_GATED_MARKERS

    assert {m.replace("_", "") for m in NON_GATED_MARKERS} <= set(L.REFUSED_MARKERS)


def test_the_guard_counts_only_in_the_status_cell(tmp_path: Path) -> None:
    row = f"| T1-1 | claim Regression guard: none | mech | **Status: EARNED 2026-09-01**. **Evidence:** [r.jsonl]({DATA}r.jsonl). |"
    _one(_lint(_repo(tmp_path, _ledger([OK_T1])), _ledger([row])), "no 'Regression guard:' field")


def test_superseded_names_its_successor_right_after_the_status_line(tmp_path: Path) -> None:
    buried = "| T1-2 | old | mech | **Status: SUPERSEDED 2026-08-25**. History: replaced by T1-1. |"
    _one(_lint(_repo(tmp_path, _ledger([OK_T1, buried])), _ledger([OK_T1, buried])), "SUPERSEDED must name")


def test_link_text_may_be_the_path_under_the_data_root(tmp_path: Path) -> None:
    files = {"docs/experiments/data/s/r.jsonl": "{}"}
    row = _ev(f"[s/r.jsonl]({DATA}s/r.jsonl)")
    assert _lint(_repo(tmp_path, _ledger([OK_T1]), files), _ledger([row])) == []


def test_a_new_row_is_dated_against_the_branch_point_not_mains_tip(tmp_path: Path) -> None:
    """On a PR, CI lints a merge commit whose first parent is main's tip. A row the branch added on 09-21 must
    not fail because main moved on 09-29 afterwards."""
    repo = tmp_path / "repo"
    (repo / "docs/plans").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    for rel, text in {"docs/experiments/data/r.jsonl": "{}", "tests/unit/test_x.py": "x = 1"}.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    (repo / L.LEDGER_PATH).write_text(_ledger([OK_T1]))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork point", date="2026-09-20T12:00:00Z")
    _git(repo, "checkout", "-q", "-b", "pr")
    new = OK_T1.replace("T1-1", "T1-2").replace("2026-09-01", "2026-09-21")
    (repo / L.LEDGER_PATH).write_text(_ledger([OK_T1, new]))
    _git(repo, "commit", "-q", "-am", "add a row", date="2026-09-21T12:00:00Z")
    _git(repo, "checkout", "-q", "main")
    (repo / "docs/experiments/data/other.jsonl").write_text("{}")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main moves", date="2026-09-29T12:00:00Z")
    tip = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "merge", "-q", "--no-ff", "pr", "-m", "merge ref", date="2026-09-29T13:00:00Z")
    failures, _ = F.lint(repo, base=tip, today=TODAY)
    assert failures == [], failures


@pytest.mark.parametrize("prefix", ["  ", "> "])
def test_an_indented_or_quoted_base_row_cannot_vanish(tmp_path: Path, prefix: str) -> None:
    row2 = prefix + OK_T1.replace("T1-1", "T1-2")
    _one(_base_then(tmp_path, [OK_T1, row2], [OK_T1]), "T1-2: a row ID in the base is gone")


def test_reproduced_is_a_positive_token_and_the_ledger_documents_every_token() -> None:
    """#1059: REPRODUCED (a successor O19 campaign's label) is rank 3, needs a guard, and the ledger's vocabulary
    table names every token the parser accepts."""
    assert L.RANK["REPRODUCED"] == 3 and "REPRODUCED" in L.POSITIVE and "REPRODUCED" in L.NEEDS_GUARD
    text = (REPO / L.LEDGER_PATH).read_text()
    table = text[text.index("| Token | Rank | Meaning |") :].split("\n\n", 1)[0]
    assert all(f"`{token}`" in table for token in L.RANK), [t for t in L.RANK if f"`{t}`" not in table]


# ── #1105: pipe-less rows and the scope vocabulary ─────────────────────────────────────────────────────


def test_a_row_without_a_leading_pipe_is_still_parsed(tmp_path: Path) -> None:
    """GFM keeps a table open until a blank line: a pipe-less row renders as a ledger row and must be linted."""
    bad = "T1-2 | claim | mech | **Status: EARNED 2099-01-01**. **Evidence:** [r.jsonl](../experiments/data/r.jsonl). |"
    rows, problems = L.parse(_ledger([OK_T1, bad]))
    assert [r.id for r in rows if r.table == "T1"] == ["T1-1", "T1-2"], problems
    repo = _repo(tmp_path, _ledger([OK_T1]))
    assert any("T1-2" in f and "after today" in f for f in _lint(repo, _ledger([OK_T1, bad])))


@pytest.mark.parametrize(
    ("qualifier", "ok"),
    [
        ("(narrow)", True),
        ("(narrow: H3 not measured)", True),
        ("(rung A)", True),
        ("(rung AB)", False),
        ("(narrow-ish)", False),
        ("(reframed)", False),
        ("(because the run was short)", False),
        ("()", False),  # #1141: an empty qualifier is no scope word
    ],
)
def test_a_qualifier_opens_with_a_scope_word(tmp_path: Path, qualifier: str, ok: bool) -> None:
    row = OK_T1.replace("**Status: EARNED 2026-09-01**.", f"**Status: EARNED 2026-09-01** {qualifier}.")
    repo = _repo(tmp_path, _ledger([row]))
    hit = any("must open with a scope word" in f for f in _lint(repo))
    assert hit is not ok


def test_the_grandfathered_qualifier_is_pinned(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(F, "GRANDFATHERED_QUALIFIERS", {"T1-1": ("EARNED", "2026-09-01", "reframed")})
    row = OK_T1.replace("**Status: EARNED 2026-09-01**.", "**Status: EARNED 2026-09-01** (reframed).")
    repo = _repo(tmp_path, _ledger([row]))
    assert not any("scope word" in f or "stale" in f for f in _lint(repo))
    moved = row.replace("(reframed)", "(narrow)")
    assert any("GRANDFATHERED_QUALIFIERS entry for T1-1" in f and "stale" in f for f in _lint(repo, _ledger([moved])))


def test_a_grandfather_pin_without_its_row_is_stale(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(F, "GRANDFATHERED_QUALIFIERS", {"T1-9": ("EARNED", "2026-09-01", "reframed")})
    repo = _repo(tmp_path, _ledger([OK_T1]))
    assert any("GRANDFATHERED_QUALIFIERS entry for T1-9 names no ledger row" in f for f in _lint(repo))


def test_a_nested_paren_qualifier_is_a_row_problem(tmp_path: Path) -> None:
    """#1141 review: STATUS_RE does not capture ``(rung B (x))``; the row would read as unqualified, the full claim."""
    row = OK_T1.replace("**Status: EARNED 2026-09-01**.", "**Status: EARNED 2026-09-01** (rung B (see x)).")
    rows, _ = L.parse(_ledger([row]))
    t1 = next(r for r in rows if r.id == "T1-1")
    assert t1.qualifier is None and t1.qualifier_unparsed
    repo = _repo(tmp_path, _ledger([row]))
    assert any("did not parse (nested parentheses?)" in f for f in _lint(repo))


def test_the_scope_word_has_no_trailing_newline() -> None:
    assert L.SCOPE_HEAD.match("rung A") and not L.SCOPE_HEAD.match("rung A\n")


# ── #1012: the follow-up edges ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("guard", "ok"),
    [
        (" **Regression guard:** `tests/unit/test_x.py`.", True),
        (" **Regression guard:** [t](../../tests/unit/test_x.py).", True),
        (" There is no regression guard: TODO.", False),
        (" Regression guard: none yet. See `tests/unit/test_x.py`.", False),  # the citation is past its sentence
        (" **Regression guard:** none yet **Evidence:** `tests/unit/test_x.py`.", False),  # another field's citation
        (" **Regression guard:** TODO [](", False),  # an unfinished link is no citation
        (" **Regression guard:** TODO [x]().", False),  # a link with an empty target is no citation
        (" There is no regression guard: `TODO`.", False),  # a backticked word is still prose
        (" **Regression guard:** [x](#top).", False),  # an anchor is no path
        (" **Regression guard:** `tests/unit/test_x.py::test_y`.", True),
        (" **Regression guard:** none **corrected 2026-09-25:** `tests/unit/test_x.py`.", False),  # a lowercase label
        (" **Regression guard:** [Exp 1. run](../../tests/unit/test_x.py).", True),  # ". " inside link text
    ],
)
def test_a_guard_cites_something_in_its_sentence(tmp_path: Path, guard: str, ok: bool) -> None:
    row = OK_T1.replace(GUARD, guard)
    hits = [f for f in _lint(_repo(tmp_path, _ledger([OK_T1])), _ledger([row])) if "Regression guard:" in f]
    assert (hits == []) is ok, hits


@pytest.mark.parametrize(
    ("line", "cells"),
    [
        ("| a | b |", ["a", "b"]),
        ("| a \\| b |", ["a \\| b"]),
        ("| a \\\\| b |", ["a \\\\| b"]),  # GitHub: a backslash-pipe never splits, whatever precedes it
        ("| a | b \\|", ["a", "b \\|"]),  # an escaped closing pipe stays in the cell
        ("| a | |", ["a", ""]),
    ],
)
def test_split_cells_matches_githubs_renderer(line: str, cells: list[str]) -> None:
    """Pinned against GitHub's renderer (`gh api markdown -f mode=gfm`, #1012 review): #1012 item 5's premise that a
    double backslash makes the pipe split was wrong, so the parser keeps GitHub's reading."""
    assert L.split_cells(line) == cells


def test_a_link_target_may_hold_balanced_parentheses() -> None:
    entries, error = L._parse_evidence(" [r](../experiments/data/r_(1).jsonl).")
    assert error is None and entries[0].path == "docs/experiments/data/r_(1).jsonl", (entries, error)


@pytest.mark.parametrize(
    ("line", "ok"),
    [("|---|---|", True), ("| :-- | --: |", True), ("---", False), ("|---|", False), ("|---|x|", False)],
)
def test_the_delimiter_row_is_githubs(line: str, ok: bool) -> None:
    """A bare `---` is a setext heading, and a cell count unlike the header's renders no table (#1012 review)."""
    assert L.is_delimiter_row(line, 2) is ok
    assert not L.is_delimiter_row("---", 1)  # a setext underline even under a one-cell header


def test_a_header_without_its_delimiter_line_is_a_problem() -> None:
    text = _ledger([OK_T1])
    lines = text.split("\n")
    delim = next(i for i, ln in enumerate(lines) if ln.startswith("|---") or ln.startswith("| ---"))
    rows, problems = L.parse("\n".join(lines[:delim] + lines[delim + 1 :]))
    assert any("no delimiter line" in p for p in problems), problems
    assert "T1-1" in [r.id for r in rows]  # the first data row is read, not swallowed as a delimiter
