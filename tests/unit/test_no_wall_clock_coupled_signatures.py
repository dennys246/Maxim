"""Temporal-signature tests must not depend on what time the suite runs.

`NAc.distribute_reward`'s temporal fallback scores an anchor by
`anchor.similarity(TemporalSignature.now())`. A test that hardcodes absolute
phases is therefore measuring the clock: the old `test_nac._make_sig` defaults
(circadian 0.5 = midday) scored sim≈0.22 in US afternoons and sim≈0.09 in the
UTC small hours, straddling the 0.01 credit threshold at
temporal_credit_weight=0.1.

Result: the suite passed all day in US time zones and failed on UTC CI runners
overnight (2026-08-22 00:33 UTC), reproducible with `TZ=UTC pytest`. It went
unnoticed because CI ran only `tests/unit/` under a config that happened not to
surface it, and nobody develops in UTC.

This is the repo's own "decay is tick-anchored, not wall-clock" lesson showing
up in a test rather than in a mechanism.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent.parent

# Phase kwargs whose value is compared against `now()` downstream.
_PHASE_KWARGS = {"circadian_phase", "weekly_phase", "monthly_phase", "annual_phase"}

# Files allowed to pin absolute phases because the DISTANCE is the thing under
# test (they assert on similarity//phase arithmetic directly rather than on a
# threshold crossing). Add here only with a comment saying why.
_ALLOWED = {
    "test_scn.py",  # exercises phase arithmetic itself
    "test_scn_oscillator_feedback.py",  # sweeps phase explicitly as the variable
    "test_no_wall_clock_coupled_signatures.py",
}


def _literal_phase_sites(path: Path) -> list[tuple[int, str]]:
    """(lineno, kwarg) for every literal phase value in a test file.

    Covers BOTH shapes, because the first version of this guard only checked
    direct keywords on a `TemporalSignature(...)` call and therefore missed the
    very bug it was written for — `test_nac._make_sig` built a dict and splatted
    it (`TemporalSignature(**defaults)`), so the phases were never keywords on
    the call. A guard that cannot detect its own failure case is not a guard.
    """
    tree = ast.parse(path.read_text())
    hits: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # shape 1: TemporalSignature(circadian_phase=0.5, ...)
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "TemporalSignature":
                for kw in node.keywords:
                    if kw.arg in _PHASE_KWARGS and isinstance(kw.value, ast.Constant):
                        hits.append((kw.lineno, kw.arg))

        # shape 2: {"circadian_phase": 0.5, ...} — splatted, assigned, or nested
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in _PHASE_KWARGS and isinstance(value, ast.Constant):
                    hits.append((key.lineno, str(key.value)))

    return hits


def test_no_test_pins_absolute_temporal_phases() -> None:
    offenders: dict[str, list[tuple[int, str]]] = {}
    for path in TESTS.rglob("test_*.py"):
        if path.name in _ALLOWED:
            continue
        hits = _literal_phase_sites(path)
        if hits:
            offenders[str(path.relative_to(TESTS))] = hits

    assert not offenders, (
        "Temporal phases pinned to literals — these compare against "
        "TemporalSignature.now() and so depend on when the suite runs "
        f"(fails on UTC CI overnight): {offenders}. Anchor to now() instead, "
        "or add the file to _ALLOWED with a reason if the distance IS the test."
    )
