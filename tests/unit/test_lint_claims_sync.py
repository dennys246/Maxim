"""scripts/lint_claims_sync.py — public claims agree with the ledger (roadmap 1.3.2 item 8, backlog M2).

Every test mutates the REAL README, experiments index or ledger text and asserts the one rule it breaks, so the
positive controls exercise the shipped surfaces, not a toy fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import lint_claims_sync as C

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / C.README).read_text(encoding="utf-8")
INDEX = (ROOT / C.INDEX).read_text(encoding="utf-8")
LEDGER = (ROOT / C.L.LEDGER_PATH).read_text(encoding="utf-8")


def _swap(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"fixture drifted: {old!r} occurs {text.count(old)} times"
    return text.replace(old, new)


def _fails(readme: str = README, index: str = INDEX, ledger: str = LEDGER) -> list[str]:
    return C.lint(readme, index, ledger)


def test_the_real_surfaces_agree_with_the_ledger():
    assert _fails() == []


# ── README ────────────────────────────────────────────────────────────────────────────────────────────


def test_a_results_row_without_a_marker_FAILS():
    out = _fails(readme=_swap(README, "<!-- claim: T1-13 --> ", ""))
    assert any("must cite its ledger row" in f for f in out), out


def test_a_row_that_does_not_display_the_ledger_status_FAILS():
    """The 2026-09-27 class: a public row saying something other than the ledger's status."""
    out = _fails(readme=_swap(README, "**EARNED 2026-09-16**", "**EARNED 2026-09-15**"))
    assert any("does not display its ledger status `EARNED 2026-09-16`" in f for f in out), out


def test_a_row_showing_a_second_status_token_FAILS():
    """'MAINTAINED … pending' and 'EARNED' beside 'RE-VALIDATED' contradict the one status shown."""
    out = _fails(readme=_swap(README, "on Paper 1.20.4)", "on Paper 1.20.4; EARNED in 1.2)"))
    assert any("shows `EARNED`, but T1-11's ledger status is `RE-VALIDATED 2026-09-19`" in f for f in out), out


def test_lowercase_prose_is_not_a_status_token():
    assert _fails(readme=_swap(README, "on Paper 1.20.4)", "on Paper 1.20.4; first earned in 1.2)")) == []


def test_a_scoped_claim_must_show_its_scope_word_after_the_status():
    out = _fails(readme=_swap(README, "**EARNED 2026-10-04** (narrow:", "**EARNED 2026-10-04**: narrow-ish ("))
    assert any("scoped `(narrow …)`" in f for f in out), out


def test_a_barred_status_cannot_sit_under_what_it_has_shown():
    ledger = _swap(LEDGER, "**Status: EARNED 2026-09-16**", "**Status: DORMANT 2026-09-16**")
    readme = _swap(README, "**EARNED 2026-09-16**", "**DORMANT 2026-09-16**")
    index = _swap(INDEX, "**EARNED 2026-09-16** — complete", "**DORMANT 2026-09-16** — complete")
    out = _fails(readme=readme, index=index, ledger=ledger)
    assert any("T1-13 is `DORMANT`" in f and "does not belong" in f for f in out), out


def test_a_marker_outside_the_results_table_FAILS():
    readme = README.replace("## What it has shown", "<!-- claim: T1-13 -->\n\n## What it has shown", 1)
    assert any("claim marker outside the results table" in f for f in _fails(readme=readme))


def test_a_marker_inside_fenced_code_is_ignored():
    readme = README.replace("## What it has shown", "```\n| <!-- claim: T9-9 --> | x |\n```\n\n## What it has shown", 1)
    assert _fails(readme=readme) == []


@pytest.mark.parametrize(
    ("marker", "message"),
    [("<!-- claim:T1-13 -->", "malformed claim marker"), ("<!-- claim: T1-99 -->", "which the ledger does not have")],
)
def test_malformed_and_unknown_markers_FAIL(marker, message):
    out = _fails(readme=_swap(README, "<!-- claim: T1-13 -->", marker))
    assert any(message in f for f in out), out


def test_two_markers_in_one_row_FAIL():
    out = _fails(readme=_swap(README, "<!-- claim: T1-13 -->", "<!-- claim: T1-13 --> <!-- claim: T1-14 -->"))
    assert any("2 claim markers in one row" in f for f in out), out


def test_the_results_table_must_be_found():
    out = _fails(readme=_swap(README, "| Result | What was measured |", "| Result | Measured |"))
    assert any("expected exactly one `| Result | What was measured |` table, found 0" in f for f in out), out


# ── experiments index ─────────────────────────────────────────────────────────────────────────────────


def test_an_index_status_cell_must_match_the_ledger():
    out = _fails(index=_swap(INDEX, "**RE-VALIDATED 2026-09-19** — complete", "**EARNED 2026-09-06** — complete"))
    assert any("T1-11 but does not display its ledger status" in f for f in out), out


def test_a_scope_word_matches_whole_words_only():
    """`rung A` must not be satisfied by `rung avoidance` (adversarial pass)."""
    out = _fails(index=_swap(INDEX, "**EARNED 2026-09-20 (rung A)**", "**EARNED 2026-09-20 (rung avoidance)**"))
    assert any("scoped `(rung A …)`" in f for f in out), out


def test_a_superseded_row_must_name_its_successor():
    out = _fails(
        index=_swap(INDEX, "**SUPERSEDED 2026-10-04** by T1-16 (Exp 63)", "**SUPERSEDED 2026-10-04** (Exp 63)")
    )
    assert any("SUPERSEDED by T1-16; name the successor" in f for f in out), out


def test_every_tier1_row_is_cited_by_the_index():
    out = _fails(index=_swap(INDEX, "<!-- claim: T1-13 --> ", ""))
    assert any("Tier 1 row T1-13 (EARNED) is cited by no index entry" in f for f in out), out


def test_an_index_marker_outside_a_status_table_FAILS():
    index = INDEX.replace("## Index", "| A | B |\n|---|---|\n| <!-- claim: T1-13 --> x | y |\n\n## Index", 1)
    assert any("claim marker outside a table with a Status column" in f for f in _fails(index=index))


def test_no_tier1_row_is_exempt_from_the_index():
    """T1-5 was the only exemption; DROPPED 2026-10-08, it is exempt by rule. A new exemption is a deliberate edit here."""
    assert C.NO_INDEX_ENTRY == {}


# ── the ledger ────────────────────────────────────────────────────────────────────────────────────────


def test_a_ledger_parse_problem_fails_the_run():
    ledger = _swap(LEDGER, "| ID | Claim | Bio-mechanism | Status |", "| ID | Claim | Mechanism | Status |")
    assert any(f.startswith(C.L.LEDGER_PATH) for f in _fails(ledger=ledger))


def test_main_is_clean_on_this_checkout(capsys):
    assert C.main() == 0
    assert "agree with the ledger" in capsys.readouterr().out


def test_a_results_row_without_a_leading_pipe_is_still_a_row():
    """GFM keeps the table open until a blank line; a pipe-less fake claim rendered as a results row (executor)."""
    anchor = "| <!-- claim: T1-14 -->"
    i = README.index(anchor)
    j = README.index("\n", i)
    readme = README[: j + 1] + "**Fake claim** (Exp 99; **EARNED 2026-01-01**) | made up\n" + README[j + 1 :]
    assert any("must cite its ledger row" in f for f in _fails(readme=readme))


def test_a_stale_exemption_FAILS(monkeypatch):
    """An exemption pinned to a status the row no longer has (T1-5 moved PARTIAL -> DROPPED) fails."""
    monkeypatch.setattr(C, "NO_INDEX_ENTRY", {"T1-5": ("PARTIAL", "2026-06-15", "a PoC with no experiment doc")})
    assert any("NO_INDEX_ENTRY exemption for T1-5" in f for f in _fails())
