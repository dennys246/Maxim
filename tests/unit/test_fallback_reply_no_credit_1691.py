"""#1691: a fallback reply runs, but it teaches NAc nothing.

Outside a sim (``NullSimulationAdapter``), the worker's fallback proposal for a missing or unparseable model reply becomes
the turn's action. It is spoken, so the user hears something, but no decision produced it: booking it credited
``respond`` under the synthetic goal ``"llm_fallback"`` on every failed model turn. Owner decision 2026-11-19: the
proposal's own label (``strategy_used="fallback"``) keys it; ``execute_and_learn`` keeps the outcome window, the
carryover and the context pool, and books no causal link, goal credit, cluster reward, energy cost or plan outcome,
like a gate refusal (#1315 item 3). Behaviour tier: engineering degraded-mode output.

The fallback here is the real one: a real ``LLMWorker`` whose model returns nothing builds it, and it runs through
the real ``execute_and_learn`` on a real executor, a real ``NAc`` and the real ``record_outcome``. Before the fix,
``test_a_parse_failure_fallback_books_no_learning`` fails: NAc's links and goal bias change and the plan bridge is
called.
"""

from __future__ import annotations

import functools
import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.agents.autonomy import AutonomyController, AutonomyLevel
from maxim.decisions.nac import NAc, NACConfig
from maxim.proprioception.pain_bus import build_pain_bus
from maxim.runtime import tool_dispatch
from maxim.runtime.bootstrap import build_executor
from maxim.runtime.sim_adapter import NullSimulationAdapter
from maxim.runtime.tool_policy import ToolPolicy
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry

AGENT_ID = "fallback_1691"
GOAL = "help the user"

# NAc's learned state, snapshot before and after (the #1315 red gate's list).
_NAC_STATE = (
    "_links",
    "_outcome_index",
    "_priors",
    "_reward_bias",
    "_visit_count",
    "_eligibility",
    "_cluster_reward_bias",
    "_cluster_reward_source",
    "_pending_operant_action",
    "_goal_reward_bias",
    "_percept_valences",
    "_cluster_fear",
    "_event_outcome_welford",
)


class _Respond(Tool):
    name = "respond"
    description = "reply to the user"
    input_schema: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.said: list[str] = []

    def execute(self, **kwargs: Any) -> Any:
        self.said.append(str(kwargs.get("message", "")))
        return ToolOutput(success=True, output="said")


class _SilentLLM:
    """A model whose reply never parses: ``generate_json`` returns None."""

    def generate_json(self, prompt: str, *_a: Any, **_k: Any) -> None:
        return None


def _fallback_proposal() -> Any:
    from maxim.agents.bus import StructuredContext
    from maxim.agents.llm_types import LLMRequest, ModeInfo
    from maxim.agents.llm_worker import LLMWorker

    request = LLMRequest(
        request_id="r1691",
        context=StructuredContext(timestamp=time.time()),
        mode=ModeInfo(name="active", goal="assist", context_prompt=""),
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        internet_access=False,
        internet_policy_summary="",
        available_tools={"respond"},
        tool_descriptions={"respond": "reply"},
        use_tool_prompting=True,
        triggering_input="can you look to your left",
        deliberation_available=False,
    )
    worker = LLMWorker(llm=_SilentLLM(), n_ctx=8192)
    worker.start()
    try:
        proposal = worker._process_request(request)
    finally:
        worker.stop()
    assert proposal is not None and proposal.strategy_used == "fallback", proposal
    assert proposal.reasoning == "llm_fallback" and proposal.action["tool_name"] == "respond"
    # Outside a sim nothing drops it: it is executed as the turn's action.
    assert NullSimulationAdapter().should_skip_fallback_proposal(proposal) is False
    return proposal


def _nac_state(nac: NAc) -> dict[str, str]:
    missing = [name for name in _NAC_STATE if not hasattr(nac, name)]
    assert not missing, f"NAc no longer has {missing}: update _NAC_STATE"
    return {name: repr(getattr(nac, name)) for name in _NAC_STATE}


def _run(proposal: Any, *, run_id: str = "run_1691") -> dict[str, Any]:
    nac = NAc(NACConfig())
    registry = ToolRegistry()
    respond = _Respond()
    registry.register(respond)
    executor = build_executor(
        registry,
        pain_bus=build_pain_bus(hippocampus=None, nac=nac),
        policy=ToolPolicy.unrestricted(reason="#1691 test"),
        nac=nac,
    )
    llm_worker, memory_hub, recent = MagicMock(), MagicMock(), []
    sim = MagicMock()
    sim.is_sim_mode = False
    before = _nac_state(nac)
    outcome = tool_dispatch.execute_and_learn(
        agent=MagicMock(_execution_tracker=None),
        agent_name="aut",
        agent_id=AGENT_ID,
        executor=executor,
        sim=sim,
        state=MagicMock(data={"active_goal": GOAL}),
        environment=MagicMock(),
        memory=MagicMock(),
        hippocampus=None,
        memory_hub=memory_hub,
        result_cache=MagicMock(),
        autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
        rec_outcome=functools.partial(tool_dispatch.record_outcome, drive_relief_only=False),
        recent_outcomes=recent,
        max_recent=10,
        llm_worker=llm_worker,
        context_pool=MagicMock(),
        nac=nac,
        run_id=run_id,
        overlay={},
        action=proposal.action,
        confidence=proposal.confidence,
        proposal=proposal,
        observation={},
        human_involved=False,
    )
    assert not outcome.raised, outcome.error
    after = _nac_state(nac)
    return {
        "said": respond.said,
        "changed": sorted(name for name in _NAC_STATE if before[name] != after[name]),
        "plan_outcomes": memory_hub.record_plan_outcome.call_count,
        "carryover": llm_worker.record_outcome.call_count,
        "recent": len(recent),
    }


def test_a_parse_failure_fallback_books_no_learning() -> None:
    got = _run(_fallback_proposal())
    assert len(got["said"]) == 1, "the fallback reply was not spoken"
    assert got["changed"] == [], f"a fallback reply taught NAc: {got['changed']}"
    assert got["plan_outcomes"] == 0, "a fallback reply booked a plan outcome"
    assert got["carryover"] == 1 and got["recent"] == 1, "the outcome window or the carryover lost the turn"


def test_the_same_action_from_a_decision_still_books() -> None:
    """The control: the scenario books when the label is not ``fallback``, so the test above is not vacuous."""
    fallback = _fallback_proposal()
    decided = SimpleNamespace(
        action=fallback.action,
        confidence=0.9,
        reasoning=GOAL,
        strategy_used="assist",
        citations=[],
        cluster_id=None,
        clusters=None,
        triggering_input=fallback.triggering_input,
    )
    got = _run(decided)
    assert "_links" in got["changed"] and "_goal_reward_bias" in got["changed"], got["changed"]
    assert got["plan_outcomes"] == 1


@pytest.mark.parametrize("reasoning", ["llm_fallback", "llm_timeout", "LLM unavailable, using fallback behavior"])
def test_every_fallback_producer_is_recognised(reasoning: str) -> None:
    assert tool_dispatch.is_fallback_proposal(SimpleNamespace(strategy_used="fallback", reasoning=reasoning))
    assert not tool_dispatch.is_fallback_proposal(SimpleNamespace(strategy_used=None, reasoning=reasoning))
    assert not tool_dispatch.is_fallback_proposal(None)


def test_the_unavailable_backend_fallback_carries_the_label() -> None:
    from maxim.agents.llm_fallback import FallbackBehavior

    assert tool_dispatch.is_fallback_proposal(FallbackBehavior.get_fallback_proposal("active", None))


def test_the_skip_is_logged_once_per_run(caplog: pytest.LogCaptureFixture) -> None:
    proposal = _fallback_proposal()
    run_id = f"run_log_{time.time_ns()}"
    with caplog.at_level(logging.INFO, logger="maxim.runtime.agent_loop"):
        _run(proposal, run_id=run_id)
        _run(proposal, run_id=run_id)
    skips = [r for r in caplog.records if "without NAc credit" in r.getMessage()]
    assert len(skips) == 1, [r.getMessage() for r in skips]
