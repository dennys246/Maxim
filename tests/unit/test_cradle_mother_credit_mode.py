"""``generative_runner.resolve_mother_credit_mode`` — the Exp 52 credit-value toggle is
validated ONCE before the turn loop and fails loud on a typo (the per-turn mother tick
sits inside a debug-swallowing try, so an invalid value would otherwise silently kill
the mother for a whole campaign)."""

from __future__ import annotations

import pytest

from maxim.simulation.generative_runner import resolve_mother_credit_mode


def test_default_is_relief(monkeypatch):
    monkeypatch.delenv("MAXIM_CRADLE_MOTHER_CREDIT", raising=False)
    assert resolve_mother_credit_mode() == "relief"


@pytest.mark.parametrize("raw", ["constant", "CONSTANT", " relief "])
def test_accepts_both_modes_case_and_space_insensitive(monkeypatch, raw):
    monkeypatch.setenv("MAXIM_CRADLE_MOTHER_CREDIT", raw)
    assert resolve_mother_credit_mode() == raw.strip().lower()


def test_typo_fails_loud_before_any_turn(monkeypatch):
    monkeypatch.setenv("MAXIM_CRADLE_MOTHER_CREDIT", "relif")
    with pytest.raises(ValueError, match="MAXIM_CRADLE_MOTHER_CREDIT"):
        resolve_mother_credit_mode()
