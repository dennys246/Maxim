"""#1083 — an APPROVED planning proposal is credited to the situation it was proposed in.

``run_agentic_loop`` step 4 (``AutonomyLevel.PLANNING``) queues the proposal as an autonomy
``Proposal``; step 5 executes approved ones and records their outcome with
the proposal's situation. Before the fix it read ``proposal.cluster_id`` / ``proposal.clusters``,
which ``Proposal`` never had, so the outcome record raised ``AttributeError`` AFTER the tool ran: a
false "Approved action failed" ERROR, then the ``except`` handler's own record raised again and
ended the run. Owner decision (2026-10-04): the situation is keyed to PROPOSAL time, so the queued
``Proposal`` references its ``LLMProposal`` (``source``) and the approved path credits the same
situation key as the autonomous path. Deleting the ``source=`` assignment fails test (a).

Driven through the real loop: substrate-primary (so no LLM worker is needed) with the substrate
proposer replaced by one that returns a proposal carrying clusters, and a PLANNING controller
whose queue is approved the moment the loop submits to it — standing in for the human (or
embedder) who calls ``proposal_queue.approve``. The loop's own non-interactive auto-approve cannot
serve: ``plan_approval`` is a critical context, so ``should_prompt`` is True in every mode.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

_CLUSTERS = {"interoception": "c_intero_7", "world": "c_world_3"}
_TOOL = "probe_1083"


def _run_loop_once(monkeypatch: pytest.MonkeyPatch, tmp_path) -> tuple[list[dict[str, Any]], list[str]]:
    """Run the loop until the approved proposal's outcome is recorded; return (outcome kwargs, executed)."""
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.agents.llm_types import LLMProposal
    from maxim.agents.maxim_agent import MaximAgent
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.state import RuntimeState
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry

    executed: list[str] = []

    class _Probe(Tool):
        name = _TOOL
        description = "stub"
        input_schema: dict = {}

        def execute(self, **kwargs):
            executed.append(_TOOL)
            return ToolOutput(success=True, output="done")

    registry = ToolRegistry()
    registry.register(_Probe())
    executor = build_executor(registry, pain_bus=None, permissions=None)

    proposed: list[int] = []

    def _propose(**_kw):
        if proposed:
            return None
        proposed.append(1)
        return LLMProposal(
            request_id="p-1083",
            action={"tool_name": _TOOL, "params": {}},
            reasoning="probe",
            strategy_used="substrate-primary",
            confidence=0.9,
            mode_goal_achieved=False,
            triggering_input="",
            cluster_id=_CLUSTERS["interoception"],
            clusters=dict(_CLUSTERS),
        )

    outcomes: list[dict[str, Any]] = []
    real_record = AL._record_outcome

    def _spy_record(**kw):
        if kw.get("tool_name") == _TOOL:
            outcomes.append(kw)
        return real_record(**kw)

    monkeypatch.setattr(AL, "propose_via_substrate", _propose)
    monkeypatch.setattr(AL, "_record_outcome", _spy_record)

    controller = AutonomyController()
    assert controller.current_level == AutonomyLevel.PLANNING
    queue = controller.proposal_queue
    real_submit = queue.submit

    def _submit_and_approve(proposal):
        real_submit(proposal)
        assert queue.approve(proposal.id, approved_by="test:operator")

    monkeypatch.setattr(queue, "submit", _submit_and_approve)
    state = RuntimeState()
    state.data["mode"] = "active"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    AL.run_agentic_loop(
        MaximAgent(),
        FileSystemEnv(str(workspace)),
        state,
        build_memory(),
        build_decision_engine(),
        executor,
        autonomy_controller=controller,
        aut_mode="substrate-primary",
        max_steps=6,
        target_hz=200.0,
        idle_sleep_s=0.0,
    )
    return outcomes, executed


@pytest.mark.xfail(strict=True, reason="#1083")
@pytest.mark.timeout(60)
def test_approved_proposal_is_credited_to_its_proposal_time_situation(monkeypatch, tmp_path):
    """(a) The approved outcome reaches the recorder with the proposal's clusters, and the run ends normally."""
    outcomes, executed = _run_loop_once(monkeypatch, tmp_path)
    assert executed == [_TOOL], "the approved proposal never reached its tool"
    assert len(outcomes) == 1, outcomes
    assert outcomes[0]["success"] is True
    assert outcomes[0]["cluster_id"] == _CLUSTERS["interoception"]
    assert outcomes[0]["clusters"] == _CLUSTERS


@pytest.mark.xfail(strict=True, reason="#1083")
@pytest.mark.timeout(60)
def test_a_successful_approved_action_logs_no_failure(monkeypatch, tmp_path, caplog):
    """(b) A tool that succeeded is not reported as "Approved action failed"."""
    with caplog.at_level(logging.ERROR, logger="maxim.runtime.agent_loop"):
        _, executed = _run_loop_once(monkeypatch, tmp_path)
    assert executed == [_TOOL]
    assert not [r for r in caplog.records if "Approved action failed" in r.getMessage()]
