"""#1239: the Observer's ``temporal_patterns`` reads a REAL SCN.

It called ``hub.scn.current_phase()``, a method ``SCN`` never had; the blanket ``except`` turned the
``AttributeError`` into ``{"error": "SCN query failed"}`` on every call, so the Observer (``session.observe("temporal")``
and the sim's ``temporal`` query) never reported a temporal phase. Its unit test passed only because its fake SCN
defined ``current_phase``.
"""

from __future__ import annotations

from types import SimpleNamespace


from maxim.simulation.introspection import Observer
from maxim.time.scn import SCN
from maxim.time.temporal_signature import TemporalSignature


def test_temporal_patterns_reports_the_real_scns_phase() -> None:
    scn = SCN()
    scn.register("m1", TemporalSignature.now())
    out = Observer(memory_hub=SimpleNamespace(scn=scn)).temporal_patterns()
    assert out["available"] is True and "error" not in out, out
    hour, day, week, month = TemporalSignature.now().to_bins()
    assert out["current_phase"]["hour_bin"] == hour and out["current_phase"]["day_bin"] == day
    assert out["signatures"] == 1
