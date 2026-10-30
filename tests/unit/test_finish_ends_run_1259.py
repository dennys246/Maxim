"""#1259: the narrator's ``finish_simulation`` ends the run.

Before the fix the tool recorded ``finish_context`` and signalled the AUT bridge and the narrator's percept source,
but set nothing the narrator loop stops on, so the run waited out the source's 180 s exhaustion grace while the stall
detector nudged the finished narrator (each nudge a fresh narrator LLM call). The evidence drives the REAL
``start_simulation_mode`` offline (``tests/unit/_orch_harness.py``, the narrator arm), whose finish watchdog sends
the operator ``/cancel`` that used to follow the finish only if the run has not stopped by itself
``FINISH_WATCHDOG_S`` after it, and then records it as a watchdog line. Red gate: commit b2f0f384 (xfail strict).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.unit._orch_harness import WATCHDOG_CANCEL, run_arm  # noqa: E402


@pytest.mark.timeout(240)
def test_the_narrators_finish_ends_the_run_without_an_operator_stop_1259(tmp_path: Path) -> None:
    trace = run_arm("narrator", tmp_path)

    watchdog = [line for line in trace["stdin_lines"] if line.get("after") == WATCHDOG_CANCEL]
    assert watchdog == [], f"the run did not stop on the narrator's finish; the harness cancelled it: {watchdog}"
    answers = [((call.get("answer") or {}).get("tool_name")) for call in trace["narrator_calls"]]
    assert "finish_simulation" in answers, answers
    first_finish = answers.index("finish_simulation")
    assert answers[first_finish + 1 :] == [], f"the narrator was asked again after it finished: {answers}"
    assert trace["result"]["finish_reason"] == "completed", trace["result"]["finish_reason"]
