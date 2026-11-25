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
imports") is met by this module's own imports. Its import closure reaches bio modules only through ``loop_controller``'s
import of ``safe_agent_name`` and ``tool_to_run`` from ``tool_dispatch`` (whose own bio imports are #1150). The strict red
gate ``tests/unit/test_loop_intake_bio_free.py::test_importing_the_intake_leaf_loads_no_bio_module`` flips when that
import is cut: by the PR right after slice 6 that moves those two helpers (owner decision 2026-11-21), or with #1150.

Characterization: ``tests/unit/test_loop_intake_characterization.py`` (written before the move, through the public
``run_agentic_loop``, and kept green by it).

**Not verbatim since #1185.** #1185 (approval intent, merged onto this module after the move) changed three lines
of the moved block, so a re-diff against slice 6's base shows them: ``_drop_skipped_fallback`` clears the pending
input through ``ctrl.clear_pending_user_input()`` (the same three keys), the stall marker is
``sinks.LLM_UNAVAILABLE`` (the same string), and ``take_llm_proposal`` counts each turn with ``note_llm_turn``
instead of resetting the stall counter. ``note_llm_turn`` lives here (moved verbatim from ``tool_dispatch``, its
only caller being the intake): a timeout or parse-failure turn counts as failed, and any other turn resets the
counter and pops ``state.data["machine_timeout_retries"]``, a state write the inline block never made.

**One intake since #1743.** Everything after the poll is ``accept_llm_proposal``, which §6's PFC deliberation
(``loop_request._deliberate``) calls too: it used to install its answers on ``ctrl`` itself, so an action-less one
(installed past every check here) blocked every later submit, and its ``next_actions`` were dropped. The split
changed no line of the moved text except the one ``declined_is_failure`` gate (``accept_llm_proposal``'s docstring)
and the D13 comment naming the install sites, and added one statement at its end: a turn that ends with no action and
no retry records the sim sink's no-op marker (``_record_declined_turn``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from maxim.agents.bus import StreamEvent
from maxim.agents.llm_types import LLMProposal
from maxim.runtime.loop_controller import _handle_planning_failure
from maxim.runtime.loop_perception import is_tool_followup
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


def note_llm_turn(proposal: Any, state: Any, consecutive_failures: int) -> int:
    """One LLM turn's effect on the loop's consecutive-LLM-failure count (the stall detector's liveness signal) and
    on the machine's timeout-retry count (#1185, owner decision 2026-11-21). A timeout proposal (``llm_timeout``) is
    a failed turn: it counts, instead of resetting the count as an answered turn does, and so is a parse failure
    (``llm_fallback``: no usable answer; outside a sim it reaches here, where the null adapter keeps it). Any other
    turn is the LLM answering: both counts reset."""
    if getattr(proposal, "reasoning", "") in ("llm_timeout", "llm_fallback"):
        return consecutive_failures + 1
    state.data.pop("machine_timeout_retries", None)
    return 0


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
) -> IntakeOutcome:
    """§2, sim mode: drop a proposal the adapter skips (a parse-failure fallback or an unregistered tool).

    The inline branch's body, moved verbatim except for its ``return`` (and #1185's two lines, module docstring): it counts the fallback, records an unregistered name in the executor's
    hallucinated-tool list, reschedules the turn under planning liveness, clears the pending user input and surfaces
    ``_llm_unavailable`` to the action sink. Returns the same ``IntakeOutcome`` as ``take_llm_proposal``; the caller
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
    ctrl.clear_pending_user_input()

    # Stall detection: after N consecutive LLM failures, record a
    # synthetic action so the bridge's settle loop detects activity
    # and exits instead of spinning for 120s.
    if _consecutive_llm_fallbacks >= _LLM_STALL_THRESHOLD and action_sink is not None:
        from maxim.simulation.sinks import LLM_UNAVAILABLE, ActionRecord

        action_sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name=LLM_UNAVAILABLE,
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
    return IntakeOutcome(exhausted=_planning_liveness_exhausted, consecutive_llm_fallbacks=_consecutive_llm_fallbacks)


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
        return accept_llm_proposal(
            new_proposal,
            llm_worker=llm_worker,
            ctrl=ctrl,
            sim=sim,
            state=state,
            executor=executor,
            action_sink=action_sink,
            on_event=on_event,
            last_surfaced_tools=last_surfaced_tools,
            pending_next_actions=pending_next_actions,
            _planning_liveness_on=_planning_liveness_on,
            _consecutive_llm_fallbacks=_consecutive_llm_fallbacks,
            declined_is_failure=True,  # §2's rule: an action-less answer is a planning failure, retried
            ends_turn=answers_a_fresh_input(new_proposal),
        )
    return IntakeOutcome(exhausted=_planning_liveness_exhausted, consecutive_llm_fallbacks=_consecutive_llm_fallbacks)


def accept_llm_proposal(
    new_proposal: Any,
    *,
    llm_worker: Any,
    ctrl: Any,
    sim: Any,
    state: Any,
    executor: Any,
    action_sink: Any,
    on_event: Any,
    last_surfaced_tools: list[str],
    pending_next_actions: list[dict[str, Any]],
    _planning_liveness_on: bool,
    _consecutive_llm_fallbacks: int,
    declined_is_failure: bool,
    ends_turn: bool,
) -> IntakeOutcome:
    """The ONE intake of a worker's answer (#1743): §2 calls it with what it polled, ``loop_request._deliberate``
    with each answer the PFC deliberation consumed outside §2's poll. A stale answer and a sim-skipped fallback are
    dropped, the turn is counted (#1185), and only an answer with an action is installed (its ``next_actions``
    queued); an error is a planning failure. ``new_proposal`` may be None (nothing polled): nothing happens.

    ``declined_is_failure``: with planning liveness on, whether an answer with no action and no error is a planning
    failure, retried boundedly (§2's rule, the default), or a turn deliberately declined, dropped with no retry (the
    deliberation path, owner decision 2026-11-24: actions per turn is a measured quantity, executor F3 on
    ``agent_loop._wait_for_proposal``). Either way it is never installed: an installed action-less proposal blocks
    §6's ``pending_proposal is None`` guard and nothing ever clears it. It is required, so a caller states its policy.
    No shipped loop runs planning liveness with a deliberation today (the narrator has no enrichment pipeline; the
    AUT and the CLI have liveness off), so the False branch guards a future one; with liveness off both callers drop
    the answer alike, and leave the pending user input for ``has_pending_llm_input``'s own 30 s expiry.

    ``ends_turn``: whether this answer ends the sim bridge's current turn, so that, when the turn ends with nothing
    done and nothing retried, ``_record_declined_turn`` tells the bridge (owner decision 2026-11-24). The
    deliberation passes True (its answers answer this pass's submit, synchronously); §2 passes
    ``answers_a_fresh_input``, so a late answer to a tool follow-up or to a deliberation's continuation, which can
    land after the bridge has moved on, never ends the next turn early.

    For the deliberation the stale rule bites in one case: when cycles 2+ stop early (a wait timed out), the answer
    they return is an earlier cycle's, possibly older than ``_STALE_PROPOSAL_AGE_S``, and it is dropped, not run.
    """
    _planning_liveness_exhausted = False
    if new_proposal:
        new_proposal, _stale_exhausted = _drop_stale_proposal(
            new_proposal, ctrl=ctrl, llm_worker=llm_worker, sim=sim, liveness_on=_planning_liveness_on
        )
        _planning_liveness_exhausted = _planning_liveness_exhausted or _stale_exhausted
        if new_proposal is None and ends_turn and not _planning_liveness_on:  # dropped, not retried: the turn is over
            _record_declined_turn(action_sink, "stale_proposal_dropped")
    # In simulation mode, skip fallback proposals — wait for real LLM
    if new_proposal and sim.should_skip_fallback_proposal(new_proposal):
        _dropped = _drop_skipped_fallback(
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
        _consecutive_llm_fallbacks = _dropped.consecutive_llm_fallbacks
        _planning_liveness_exhausted = _planning_liveness_exhausted or _dropped.exhausted
    if new_proposal:
        _consecutive_llm_fallbacks = note_llm_turn(new_proposal, state, _consecutive_llm_fallbacks)  # #1185
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
            # (including the multi-step path; the deliberation
            # installs through here since #1743) recovers.
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
                if declined_is_failure and _handle_planning_failure(
                    ctrl,
                    llm_worker,
                    sim,
                    reason=_no_action_reason,
                    original_request=new_proposal.original_request,
                    exhausted_status="planning_failed",
                ):
                    _planning_liveness_exhausted = True
        if ends_turn and _turn_is_over(new_proposal, liveness_on=_planning_liveness_on, retries=declined_is_failure):
            _record_declined_turn(action_sink, _proposal_without_action_reason(new_proposal) or "proposal_error")
    return IntakeOutcome(exhausted=_planning_liveness_exhausted, consecutive_llm_fallbacks=_consecutive_llm_fallbacks)


def answers_a_fresh_input(proposal: Any) -> bool:
    """Whether §2's answer answers a fresh input (a person's words or a plan modification), the thing the sim bridge's
    current turn waits on: its request had a triggering input and no tool follow-up among its inputs. A deliberation
    continuation (no triggering input) and a tool follow-up's answer do not: they can arrive after the bridge settled
    on an earlier record and moved on (#1743 delta round)."""
    request = getattr(proposal, "original_request", None)
    if request is None or not getattr(request, "triggering_input", ""):
        return False
    inputs = getattr(getattr(request, "context", None), "cli_inputs", None) or []
    return not any(is_tool_followup(text) for text in inputs)


def _turn_is_over(proposal: Any, *, liveness_on: bool, retries: bool) -> bool:
    """An answer with no action that nothing retries: an error (but shutdown, which ends the run) is retried only under
    planning liveness; an action-less answer only under liveness with ``declined_is_failure``."""
    if proposal.action:
        return False
    if proposal.error:
        return proposal.error != "shutdown" and not liveness_on
    return not (liveness_on and retries)


def _record_declined_turn(action_sink: Any, reason: str) -> None:
    """A turn that ended with nothing done and nothing retried records the sink's no-op marker
    (``sinks.DELIBERATION_NOOP``): the deliberation's declined turns and §2's (an action-less answer, an error or a
    stale drop, all with planning liveness off). The sim bridge then sees the AUT answered and settles instead of
    waiting out its response timeout; it counts the marker as ``noop_count``, never as an action, so ``timed_out``
    and the action cap are unchanged. Built in 0.8 (Stage 4, F2), first recorded by #1743 (owner decision
    2026-11-24), through the ``ActionSink`` protocol's ``record`` as ``_drop_skipped_fallback``'s marker is."""
    if action_sink is None:
        return
    from maxim.simulation.sinks import DELIBERATION_NOOP, ActionRecord

    action_sink.record(
        ActionRecord(timestamp=time.time(), tool_name=DELIBERATION_NOOP, result_success=True, result_output=reason)
    )


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
