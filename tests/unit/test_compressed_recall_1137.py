"""#1137's red gate: a fact the user stated still reaches the prompt after its memory is COMPRESSED.

#1465 made "a fact said in session N is answerable in session N+1" true, but only until compression. A
``CompressedMemory`` kept ``had_user_input`` as a bool and nothing it heard, and ``goal`` intent-first only, so
once a record aged past its compression threshold the words were gone. Sleep consolidation compresses a record
holding user words after 3 x ``compression_age``, about 3 days under ``access_based``. The owner decided
(2026-10-25) that #1137 ships in 1.3.4, as a release condition for the recall claim.

Same shipped composition as ``test_cross_session_recall_1444.py``: a real ``MaximHandle``, its talk loop, the real
``LLMWorker`` and ``PromptBuilder``, and a recording fake model. Session N states the fact. Its records are aged past
the compression threshold, the only test-side manipulation, standing in for days of wall time. ``stop()`` runs the
full consolidation, which compresses them, and saves. A new handle on the same home asks. The fact's record must be a
``CompressedMemory`` on reload (so the gate cannot pass on an uncompressed store), and "green" must reach the prompt
inside the untrusted-content frame.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from tests.unit.test_cross_session_recall_1444 import (
    _FACT,
    _QUESTION,
    _RecordingModel,
    _ask,
    _framed_bodies,
    _handle,
)

_DAYS = 5 * 24 * 3600  # past 3 x compression_age (1 day): user-input records compress at 3 days (access_based)


def _age_every_record(handle: Any, seconds: float) -> None:
    """Days of wall time, compressed: every record of session N was made ``seconds`` ago."""
    hippocampus = handle.instance.hippocampus
    if hasattr(hippocampus, "flush"):
        hippocampus.flush()
    for record in list(hippocampus._memories.values()):
        record.created_at -= seconds
        record.timestamp -= seconds
        record.accessed_at -= seconds


@pytest.fixture(scope="module")
def compressed_session(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from maxim.simulation.sim_logger import disable_sim_logging

    home = tmp_path_factory.mktemp("compressed_gate") / "home"
    model = _RecordingModel()
    with pytest.MonkeyPatch.context() as mp:
        import maxim.runtime.lane_backends as lane_backends
        from maxim.simulation.bridge import SimulationBridge

        mp.setattr(lane_backends, "build_primary_router", lambda *a, **k: (model, None))
        mp.setenv("MAXIM_SKIP_REMOTE_PROBE", "1")
        real_init = SimulationBridge.__init__

        def _quick_settle(self: Any, *a: Any, **k: Any) -> None:  # timing only: Talk settles 2 s after each reply
            real_init(self, *a, **{**k, "settle_s": 0.2})

        mp.setattr(SimulationBridge, "__init__", _quick_settle)

        session_n = _handle(home)
        try:
            _ask(session_n, model, _FACT)
            time.sleep(0.5)
            _age_every_record(session_n, _DAYS)
        finally:
            session_n.stop()

        stored = json.loads((home / "hippocampus.json").read_text())
        session_n1 = _handle(home)
        try:
            prompts = _ask(session_n1, model, _QUESTION)
        finally:
            session_n1.stop()
            disable_sim_logging()
    return {"stored": stored, "prompts": prompts}


def test_the_facts_memory_was_compressed(compressed_session):
    """Non-vacuity: the session-end consolidation did compress the record holding the user's words."""
    assert any(m.get("_compressed") and m.get("had_user_input") for m in compressed_session["stored"]["memories"]), [
        (m.get("_compressed"), m.get("had_user_input")) for m in compressed_session["stored"]["memories"]
    ]


@pytest.mark.xfail(strict=True, reason="#1137: a compressed record keeps no words, so the fact is lost")
def test_a_compressed_fact_still_reaches_the_prompt_framed(compressed_session):
    prompts = compressed_session["prompts"]
    assert any("green" in body.lower() for p in prompts for body in _framed_bodies(p)), prompts[-1]
