"""The agent loop's §2, the LLM proposal intake (1.3.6 loop phase 2, slice 6).

``take_llm_proposal`` is what one pass of ``agent_loop.run_agentic_loop`` does with what the worker hands back,
moved verbatim from the loop's body (its sim fallback-drop branch in ``_drop_skipped_fallback``, so each function
stays at 200 lines or fewer, owner decision 2026-11-21): it polls ``llm_worker.get_latest_proposal()`` once, stamps the D13 clock on
ANY proposal, drops a stale one (``_drop_stale_proposal``), in sim mode drops a parse-failure fallback or a
proposal naming an unregistered tool (recording the bad name in the executor's hallucinated-tool list, clearing
the pending user input and surfacing ``_llm_unavailable`` to the action sink), installs an executable proposal on
``ctrl`` (queuing its ``next_actions``), drops an error proposal and, under planning liveness, a proposal with no
action; under planning liveness every drop reschedules the turn through ``loop_controller._handle_planning_failure``.
After the loop's exhaustion check, ``take_queued_next_action`` installs the next queued action as a ``multi_step``
continuation when nothing is pending.

**What it returns.** Two values the body carries, in a frozen ``IntakeOutcome``: ``exhausted`` (the planning
retry budget is spent: the loop ``break``s through its normal teardown, then raises ``PlanningLivenessExhausted``)
and ``consecutive_llm_fallbacks`` (the stall counter, carried from pass to pass: the body passes its own count in
and rebinds it from the outcome). ``exhausted`` starts False here: the loop's flag is False whenever §2 runs, since
every write of True is followed by a ``break``, and the call site ORs the result into it.

**Arguments.** Explicit keywords, never the loop's ``LoopRun`` (rule (d)): exactly the names the block reads, each
the loop's own local, named as the body has always named it (``_loop_name``, ``_planning_liveness_on``,
``_consecutive_llm_fallbacks``), so the block is byte for byte the inline one, re-indented. ``_LLM_STALL_THRESHOLD``
was a constant local of the body read only here, so it is a module constant. The block LOGS on the
``maxim.runtime.agent_loop`` logger (``logger`` below is that same object), so its records keep the loop's name.

**Not bio-free yet.** B2's prerequisite ("the ``_llm_unavailable`` fallback leaves the body for a module with no bio
imports") is met by this module's own imports, but ``loop_controller`` imports ``tool_dispatch``, whose bio imports
are #1150; the strict red gate in ``tests/unit/test_loop_intake_characterization.py`` flips when #1150 lands.

Characterization: ``tests/unit/test_loop_intake_characterization.py`` (written before the move, through the public
``run_agentic_loop``, and kept green by it).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from maxim.agents.bus import StreamEvent
from maxim.agents.llm_types import LLMProposal
from maxim.runtime.loop_controller import _handle_planning_failure
from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import log_agentic

# The SAME logger object as ``agent_loop.logger`` (logging returns one logger per name), so records keep
# the ``maxim.runtime.agent_loop`` name.
logger = logging.getLogger("maxim.runtime.agent_loop")

_LLM_STALL_THRESHOLD = 1  # Surface immediately — only one fallback per LLM request


@dataclass(frozen=True)
class IntakeOutcome:
    """What one pass's §2 hands the rest of the pass (module docstring).

    Runtime-ephemeral: built and consumed inside one pass of ``run_agentic_loop``, never persisted and never
    sent anywhere, so the CC3 forward-compat rule for frozen dataclasses does not apply.
    """

    exhausted: bool  # the planning retry budget is spent -> the loop breaks
    consecutive_llm_fallbacks: int  # the stall counter, carried to the next pass


# A completed proposal that waited longer than this is dropped (#1048's cutoff).
_STALE_PROPOSAL_AGE_S = 35.0


def _drop_stale_proposal(proposal: Any, *, ctrl: Any, llm_worker: Any, sim: Any, liveness_on: bool) -> tuple[Any, bool]:
    """Discard a proposal that sat in the result queue too long. The worker's ``_stale_threshold`` governs
    request freshness; this guards completed proposals. Returns ``(proposal or None, liveness_exhausted)``.

    D13: a dropped stale proposal is a consumed planning turn with nothing executed, so it is rescheduled, never
    idled. With one planning request in flight per narrator this is close to unreachable (#1057 tracks re-planning
    from the current context)."""
    age = time.time() - proposal.timestamp
    if age <= _STALE_PROPOSAL_AGE_S:
        return proposal, False
    logger.warning("Skipping stale LLM proposal (age=%.1fs, request_id=%s)", age, proposal.request_id)
    sim.log("EXEC", f"DROPPED: stale proposal (age={age:.1f}s)")
    exhausted = liveness_on and _handle_planning_failure(
        ctrl,
        llm_worker,
        sim,
        reason="stale_proposal_dropped",
        original_request=proposal.original_request,
        exhausted_status="planning_failed",
    )
    return None, bool(exhausted)


def _proposal_without_action_reason(proposal: Any) -> str | None:
    """Classify a parsed proposal that supplied no executable action."""
    if getattr(proposal, "action", None) or getattr(proposal, "error", None):
        return None
    if not getattr(proposal, "ready_to_act", True):
        return "proposal_not_ready_to_act"
    if getattr(proposal, "mode_goal_achieved", False):
        return "proposal_completed_without_action"
    return "proposal_without_action"


def _drop_skipped_fallback(
    *,
    new_proposal: Any,
    sim: Any,
    ctrl: Any,
    state: Any,
    executor: Any,
    llm_worker: Any,
    action_sink: Any,
    _planning_liveness_on: bool,
    _consecutive_llm_fallbacks: int,
) -> tuple[int, bool]:
    """§2, sim mode: drop a proposal the adapter skips (a parse-failure fallback or an unregistered tool).

    The inline branch's body, moved verbatim: it counts the fallback, records an unregistered name in the executor's
    hallucinated-tool list, reschedules the turn under planning liveness, clears the pending user input and surfaces
    ``_llm_unavailable`` to the action sink. Returns ``(consecutive_llm_fallbacks, liveness_exhausted)``; the caller
    sets its own ``new_proposal`` to None (the branch's assignment of it here rebinds only this function's local).
    """
    _planning_liveness_exhausted = False
    _consecutive_llm_fallbacks += 1
    logger.info(
        "Sim mode: skipping fallback proposal (%d consecutive)",
        _consecutive_llm_fallbacks,
    )
    sim.log("EXEC", f"DROPPED: fallback proposal (sim mode, #{_consecutive_llm_fallbacks})")
    _fallback_original_request = new_proposal.original_request
    # ``should_skip_fallback_proposal`` rejects for TWO different
    # reasons and they are not the same defect (review round,
    # executor lens F1). ``llm_fallback`` means the LLM never
    # produced a usable answer — a lost planning turn. A proposal
    # naming an unregistered tool is a RESPONSIVE model making a
    # bad choice; calling that "wedged" would be a false
    # diagnosis, and requeueing it byte-identically just
    # reproduces the same bad name. Record the name through the
    # existing hallucinated-tool channel so the retry can differ.
    _is_parse_failure = getattr(new_proposal, "reasoning", "") == "llm_fallback"
    if not _is_parse_failure:
        _bad_action = new_proposal.action if isinstance(new_proposal.action, dict) else {}
        _bad_tool = str(_bad_action.get("tool_name", "") or "")
        if _bad_tool:
            try:
                _hallucinated = getattr(executor, "_tools_hallucinated", None)
                if _hallucinated is not None and _bad_tool not in _hallucinated:
                    _hallucinated.append(_bad_tool)
            except Exception as e:
                log_swallowed_exception(e, operation="record_hallucinated_tool")
    new_proposal = None
    # D13: pre-fix this drop was terminal — pending_action_followup
    # was already cleared and the triggering input deduped when the
    # failed request was BUILT, so nothing ever re-armed the loop
    # (status 200 → parse failure → idle forever). Reschedule the
    # turn, bounded. Both drop kinds must reschedule (either one
    # left the loop with nothing to wake it), but they carry
    # different reasons so the abort report names what actually
    # happened.
    if _planning_liveness_on and _handle_planning_failure(
        ctrl,
        llm_worker,
        sim,
        reason=("fallback_proposal_dropped" if _is_parse_failure else "unregistered_tool_proposed"),
        original_request=_fallback_original_request,
        failed_tool=None if _is_parse_failure else _bad_tool,
        exhausted_status="llm_wedged" if _is_parse_failure else "planning_failed",
    ):
        _planning_liveness_exhausted = True
    # Clear pending input so the loop doesn't block on a
    # response that will never come.  Without this, the
    # has_pending_llm_input guard (30s timeout) prevents
    # propose_intent from running AND prevents the bridge
    # send_and_wait from timing out, causing a 120s stall.
    state.data.pop("pending_user_input", None)
    state.data.pop("pending_user_input_time", None)
    state.data.pop("pending_user_input_source", None)

    # Stall detection: after N consecutive LLM failures, record a
    # synthetic action so the bridge's settle loop detects activity
    # and exits instead of spinning for 120s.
    if _consecutive_llm_fallbacks >= _LLM_STALL_THRESHOLD and action_sink is not None:
        from maxim.simulation.sinks import ActionRecord

        action_sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name="_llm_unavailable",
                tool_args={},
                result_success=False,
                result_output=None,
                result_error=f"LLM unavailable after {_consecutive_llm_fallbacks} consecutive failures",
            )
        )
        sim.log(
            "EXEC",
            f"LLM stall detected ({_consecutive_llm_fallbacks} failures) — surfacing to bridge",
        )
        logger.warning(
            "LLM stall: %d consecutive fallbacks dropped, surfacing _llm_unavailable to bridge",
            _consecutive_llm_fallbacks,
        )
        _consecutive_llm_fallbacks = 0  # Reset so we don't flood
    return _consecutive_llm_fallbacks, _planning_liveness_exhausted


def take_llm_proposal(
    *,
    llm_worker: Any,
    ctrl: Any,
    sim: Any,
    state: Any,
    executor: Any,
    step_num: int,
    action_sink: Any,
    on_event: Any,
    last_surfaced_tools: list[str],
    pending_next_actions: list[dict[str, Any]],
    _loop_name: str,
    _planning_liveness_on: bool,
    _consecutive_llm_fallbacks: int,
) -> IntakeOutcome:
    """§2: poll the worker once and act on what it returned (module docstring)."""
    _planning_liveness_exhausted = False
    if llm_worker:
        new_proposal = llm_worker.get_latest_proposal()
        if new_proposal is not None:
            # D13 planning liveness: ANY proposal (even one dropped below
            # as stale/fallback) proves the worker answered this submit —
            # the await-expiry backstop keys on last_proposal_time <
            # last_llm_submit_time.
            ctrl.last_proposal_time = time.time()
            ctrl.reset_planning_transport_failures()
        # Sim-mode periodic traces
        if sim.is_sim_mode and step_num % 20 == 0:
            sim.log("PIPELINE", f"Loop step {step_num}, proposal={'YES' if new_proposal else 'none'}")
        # Fire a trace the FIRST time a proposal is pulled so we
        # can tie the llm_worker's "LLMProposal built" log to the
        # agent loop actually consuming it.
        if sim.is_sim_mode and new_proposal is not None:
            _tool_name = new_proposal.action.get("tool_name") if isinstance(new_proposal.action, dict) else None
            sim.log(
                "EXEC",
                f"Proposal consumed by {_loop_name}: tool={_tool_name} age={time.time() - new_proposal.timestamp:.2f}s",
            )
        if new_proposal:
            sim.log(
                "EXEC",
                f"Proposal received: tool={new_proposal.action.get('tool_name') if isinstance(new_proposal.action, dict) else None}",
            )

            new_proposal, _stale_exhausted = _drop_stale_proposal(
                new_proposal, ctrl=ctrl, llm_worker=llm_worker, sim=sim, liveness_on=_planning_liveness_on
            )
            _planning_liveness_exhausted = _planning_liveness_exhausted or _stale_exhausted
        # In simulation mode, skip fallback proposals — wait for real LLM
        if new_proposal and sim.should_skip_fallback_proposal(new_proposal):
            _consecutive_llm_fallbacks, _dropped_exhausted = _drop_skipped_fallback(
                new_proposal=new_proposal,
                sim=sim,
                ctrl=ctrl,
                state=state,
                executor=executor,
                llm_worker=llm_worker,
                action_sink=action_sink,
                _planning_liveness_on=_planning_liveness_on,
                _consecutive_llm_fallbacks=_consecutive_llm_fallbacks,
            )
            new_proposal = None  # dropped: the helper's own ``new_proposal = None`` is its local
            _planning_liveness_exhausted = _planning_liveness_exhausted or _dropped_exhausted
        if new_proposal:
            _consecutive_llm_fallbacks = 0  # Real proposal arrived — LLM is back
            if callable(on_event):
                try:
                    on_event(StreamEvent("inference_end", {"has_proposal": True}))
                except Exception as e:
                    log_swallowed_exception(e, operation="on_event:inference_end")
            if new_proposal.action:
                tool_name = new_proposal.action.get("tool_name", "unknown")
                logger.info("LLM proposal received: tool=%s, confidence=%.2f", tool_name, new_proposal.confidence)
                sim.log("EXEC", f"LLM proposes: {tool_name} (confidence={new_proposal.confidence:.2f})")
                # Log to agentic stream
                log_agentic(
                    "agent_loop",
                    "llm_response",
                    {
                        "tool": tool_name,
                        "confidence": round(new_proposal.confidence, 2),
                        "reasoning": new_proposal.reasoning[:80] if new_proposal.reasoning else None,
                        "has_plan": bool(getattr(new_proposal, "plan_text", None)),
                        "requires_approval": getattr(new_proposal, "requires_approval", False),
                    },
                )
                # D13: installing an executable proposal resets the
                # planning-failure streak — enforced by the property
                # setter on LoopController, so every install site
                # (including the multi-step and deliberation paths)
                # recovers, not just this one.
                ctrl.pending_proposal = new_proposal
                # Record surfaced-but-unused signal for learned tool index
                _bridge = getattr(executor, "_tool_pain_bridge", None)
                _tidx = getattr(_bridge, "_tool_index", None) if _bridge else None
                if _tidx and last_surfaced_tools:
                    _goal = state.data.get("active_goal", "") or ""
                    if _goal:
                        _tidx.record_surfaced_but_unused(_goal, last_surfaced_tools, tool_name)
                # Clear pending user input now that LLM has responded
                state.data.pop("pending_user_input", None)
                state.data.pop("pending_user_input_time", None)
                state.data.pop("pending_user_input_source", None)
                # Queue any next_actions for sequential execution
                if new_proposal.next_actions:
                    pending_next_actions.extend(new_proposal.next_actions)
                    log_agentic(
                        "agent_loop",
                        "multi_step_queued",
                        {"count": len(new_proposal.next_actions)},
                    )
            elif new_proposal.error:
                # Clear pending user input on error too
                state.data.pop("pending_user_input", None)
                state.data.pop("pending_user_input_time", None)
                state.data.pop("pending_user_input_source", None)
                logger.warning("LLM proposal error: %s", new_proposal.error)
                sim.log("EXEC", f"DROPPED: proposal error — {new_proposal.error}")
                log_agentic(
                    "agent_loop",
                    "error",
                    {"context": "llm_proposal", "error": str(new_proposal.error)[:100]},
                    level="WARNING",
                )
                # D13: an error proposal is also a consumed planning turn
                # with nothing executed. Shutdown is deliberate, not a
                # liveness failure.
                if (
                    _planning_liveness_on
                    and new_proposal.error != "shutdown"
                    and _handle_planning_failure(
                        ctrl,
                        llm_worker,
                        sim,
                        reason=f"proposal_error:{str(new_proposal.error)[:60]}",
                        original_request=new_proposal.original_request,
                    )
                ):
                    _planning_liveness_exhausted = True
            elif _planning_liveness_on:
                # A syntactically valid response can still contain no
                # action and no error. Pre-fix this was stamped as
                # received and then ignored forever, permanently
                # disabling the idle backstop. Treat it as an explicit
                # non-executable planning outcome and retry boundedly.
                _no_action_reason = _proposal_without_action_reason(new_proposal)
                if _no_action_reason is not None:
                    state.data.pop("pending_user_input", None)
                    state.data.pop("pending_user_input_time", None)
                    state.data.pop("pending_user_input_source", None)
                    logger.warning("LLM proposal had no executable action: %s", _no_action_reason)
                    sim.log("EXEC", f"DROPPED: {_no_action_reason}")
                    if _handle_planning_failure(
                        ctrl,
                        llm_worker,
                        sim,
                        reason=_no_action_reason,
                        original_request=new_proposal.original_request,
                        exhausted_status="planning_failed",
                    ):
                        _planning_liveness_exhausted = True
    return IntakeOutcome(exhausted=_planning_liveness_exhausted, consecutive_llm_fallbacks=_consecutive_llm_fallbacks)


def take_queued_next_action(*, ctrl: Any, pending_next_actions: list[dict[str, Any]]) -> None:
    """After §2's exhaustion check: install the next queued action as a ``multi_step`` continuation."""
    # Check for queued next_actions if no pending proposal
    if ctrl.pending_proposal is None and pending_next_actions:
        next_action = pending_next_actions.pop(0)
        ctrl.pending_proposal = LLMProposal(
            request_id=f"next-{time.time_ns()}",
            action=next_action,
            reasoning="multi_step_continuation",
            strategy_used="multi_step",
            confidence=0.8,
            mode_goal_achieved=False,
        )
