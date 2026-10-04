"""#840: FearCircuitBridge is Dormant (2026-10-04). It stays constructed and wired, but its broken NAc calls
report through log_swallowed_exception instead of passing silently."""

from __future__ import annotations

import time

import maxim.bridges.fear_bridge as fb
from maxim.decisions.nac import NAc, NACConfig


def _bridge(monkeypatch):
    reported: list[bool] = []
    monkeypatch.setattr(fb, "log_swallowed_exception", lambda *a, **k: reported.append(True))
    return fb.FearCircuitBridge(hippocampus=None, nac=NAc(NACConfig()), ec=None), reported


def test_the_broken_nac_write_reports(monkeypatch) -> None:
    bridge, reported = _bridge(monkeypatch)
    event = fb.RiskEvent("abcdef0123", "code_execution", "high", "code_review", True, True, time.time())
    bridge._report_to_nac(event)  # record_event has no such signature (#840)
    assert reported == [True]


def test_the_broken_nac_read_reports_and_stays_neutral(monkeypatch) -> None:
    bridge, reported = _bridge(monkeypatch)
    assert bridge._get_nac_risk_factor("code_execution", "subprocess") == 1.0  # NAc has no predict_outcome
    assert reported == [True]


def test_the_module_says_it_is_dormant() -> None:
    assert "Dormant since 2026-10-04" in (fb.__doc__ or "")
