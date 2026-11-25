"""#1743 wire fold (#997's pattern): the planning worker's exception sim-log line reduces every URL in the error to
its origin. An LLM transport error can carry ``user:pw@`` or a private path, and the sim log reaches
``MAXIM_LOG_FILE`` and the memory agent's context; the proposal's own ``error`` stays raw for the loop's intake,
which reduces it at each of its sinks (``tests/unit/test_loop_intake_characterization.py``, the ``*_url_*`` tests).
"""

from __future__ import annotations

import time
from typing import Any

import pytest

ERR = "connect failed: https://user:hunter2@llm.internal:8100/v1/chat?key=SECRET"


class _UnusedLLM:
    def generate_json(self, *_a: Any, **_k: Any) -> dict[str, Any] | None:
        raise AssertionError("the planning call is not reached: the prompt build raises first")


def _request() -> Any:
    from maxim.agents.autonomy import AutonomyLevel
    from maxim.agents.bus import StructuredContext
    from maxim.agents.llm_types import LLMRequest, ModeInfo

    return LLMRequest(
        request_id="r1743",
        context=StructuredContext(timestamp=time.time()),
        mode=ModeInfo(name="active", goal="assist", context_prompt=""),
        autonomy_level=AutonomyLevel.SUPERVISED,
        internet_access=False,
        internet_policy_summary="",
        available_tools={"respond"},
        tool_descriptions={"respond": "reply"},
        use_tool_prompting=True,
        triggering_input="look left",
        deliberation_available=False,
    )


def test_the_exception_sim_log_line_keeps_only_the_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.simulation.sim_logger as sim_logger
    from maxim.agents.llm_worker import LLMWorker

    lines: list[str] = []
    monkeypatch.setattr(sim_logger, "sim_log", lambda cat, msg, data=None: lines.append(msg))
    worker = LLMWorker(llm=_UnusedLLM(), n_ctx=8192)

    def _raise(_request: Any) -> str:  # an error the model call's own fallback does not absorb
        raise ConnectionError(ERR)

    monkeypatch.setattr(worker, "_build_prompt", _raise)
    proposal = worker._process_request(_request())
    assert proposal is not None and proposal.error  # the error proposal the loop's intake reads
    logged = [m for m in lines if m.startswith("LLMProposal EXCEPTION")]
    assert logged and all("https://llm.internal:8100" in m for m in logged)
    assert not any(s in m for m in logged for s in ("hunter2", "user:", "/v1/chat", "SECRET"))
