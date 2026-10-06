"""Post-extraction tests for ``runtime/loop_gates.py`` (1.3.2 decomposition, slice 2).

The behaviour of the pre-tick gate is characterized through the public loop in
``test_loop_gates_characterization.py``; this file pins what only the extracted seam has.
"""

from __future__ import annotations

import pytest

from maxim.runtime import agent_loop as AL
from tests.unit.test_loop_setup_characterization import _run


def test_an_unhandled_gate_outcome_fails_closed(monkeypatch, tmp_path):
    """``run_agentic_loop`` dispatches BREAK / IDLE / EXHAUSTED and treats only RUN as fall-through: a new
    ``GateOutcome`` member that nobody taught the loop must raise, never silently act as RUN."""
    sentinel = object()
    monkeypatch.setattr(AL, "pre_tick_gate", lambda **_kw: sentinel)
    with pytest.raises(AssertionError, match="unhandled GateOutcome"):
        _run(monkeypatch, tmp_path, stop=False, max_steps=1)
