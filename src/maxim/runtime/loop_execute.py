"""The agent loop's §4, the pending-action execution: what one pass does with the proposal §2 installed (1.3.6 loop
phase 2, slice 8).

``execute_pending_tick`` is §4 as one pass of ``agent_loop.run_agentic_loop`` runs it, moved verbatim from the loop's
body:

- ``loop_controller.set_aside_run_end``: the #1451 re-plan rule (a run-ending answer that inputs were held behind is
  set aside once), before anything runs.
- With an action pending: the proposal is re-stamped with the live situation (``substrate_proposal.
  _attach_live_situation``, through the module reference, the one binding a patch must hit); then
  ``_same_tool_cap``, the consecutive same-tool cap (refinement plan 1.5e: the ``_MAX_CONSECUTIVE_SAME_TOOL + 1``-th
  call in a row of one tool with identical params is dropped, and a SYSTEM outcome names the other tools); then
  ``parallel_actions`` run together as one ``batched_exploration`` follow-up; then the plan gate
  (``LoopController.gate_plan_approval``); then the typed autonomy check, which runs the action
  (``execute_and_learn``, its follow-up installed), queues it for approval at PLANNING, parks a confirmation, or books
  a hard rejection.

By rule (a) the two log helpers called only from this code moved here too: ``_log_goal_proposed`` and
``_log_autonomy_check`` (each moved out of the body earlier, rule T for #1775 and #1185). ``_MAX_CONSECUTIVE_SAME_TOOL``
was a constant local of the body read only here, so it is a module constant.

**What it returns.** ``ExecuteOutcome``: whether the rest of the pass is skipped (the body's ``continue`` after the
cap, the parallel batch and the plan gate; the body continues on it, so §5 to §9 do not run and the pass is not
counted in ``state.steps_taken``), and the cap's three counters, body locals a LATER pass's §4 reads. The block's other
locals (``action``, ``confidence``, ``can_execute``, ``reason``, ...) are read nowhere else: §3 assigns its own before
reading them.

**Arguments.** Explicit keywords, never the loop's ``LoopRun`` (rule (d)), each the loop's own local named as the body
names it (``_loop_nac``, ``_get_all_tools``, ...), so the moved text is the inline text, re-indented. The code LOGS on
the ``maxim.runtime.agent_loop`` logger (``logger`` below is that same object).

Characterization: ``tests/unit/test_loop_execute_characterization.py`` (written before the move, through the public
``run_agentic_loop``, and kept green by it).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from maxim.agents.autonomy import AutonomyLevel
from maxim.runtime import loop_state as _lstate
from maxim.runtime.loop_controller import note_system_outcome, set_aside_run_end
from maxim.runtime.loop_state import operational_mode
from maxim.runtime.loop_types import ActionFollowup
from maxim.runtime.tool_dispatch import execute_parallel_actions as _execute_parallel
from maxim.utils.cloud_redaction import text_shape
from maxim.utils.content_safety import safe_tool_name
from maxim.utils.structured_logging import log_agentic

# Module reference, not a name import: a test that patches the leaf also reaches §4's call.
from maxim.runtime import substrate_proposal as _sp

# The SAME logger object as ``agent_loop.logger`` (logging returns one logger per name), so records keep
# the ``maxim.runtime.agent_loop`` name.
logger = logging.getLogger("maxim.runtime.agent_loop")

# Consecutive same-tool cap — prevents respond loops (refinement plan 1.5e)
_MAX_CONSECUTIVE_SAME_TOOL: int = 5


@dataclass(frozen=True)
class ExecuteOutcome:
    """What one §4 hands back to the loop's body (module docstring, "What it returns"). Runtime-ephemeral (built and
    read in one pass, never persisted, never crossing a wire), so CC3 does not apply."""

    skip_rest_of_pass: bool  # the body's ``continue``: §5 to §9 of this pass do not run
    consecutive_same_tool: str  # the cap's counters, which the next pass's §4 reads
    consecutive_same_tool_count: int
    consecutive_same_content_hash: int


def _log_goal_proposed(*, tool: Any, confidence: Any, reasoning: str | None) -> None:
    """§4's ``goal_proposed`` event (moved out of ``_run_agentic_loop_body``, rule T for #1775). The model's reasoning
    is logged by its shape only: the stream reaches stdout and can quote a PRIVATE tool's output."""
    log_agentic(
        "agent_loop",
        "goal_proposed",
        {
            "tool": tool,
            "confidence": confidence,
            "reasoning": text_shape(reasoning) if reasoning else None,
            "source": "llm_worker",
        },
    )


def _log_autonomy_check(action: dict[str, Any], can_execute: bool, reason: Any, confidence: Any) -> None:
    """§4's autonomy-check log line and event (moved verbatim out of ``_run_agentic_loop_body``, rule T for #1185)."""
    logger.info("Autonomy check: can_execute=%s, reason=%s", can_execute, reason)

    # Log autonomy check
    log_agentic(
        "agent_loop",
        "autonomy_check",
        {
            "tool": action.get("tool_name"),
            "can_execute": can_execute,
            "reason": reason if not can_execute else None,
            "confidence": confidence,
        },
    )


def _same_tool_cap(
    *,
    action: Any,
    ctrl: Any,
    executor: Any,
    autonomy_controller: Any,
    sim: Any,
    recent_outcomes: Any,
    max_recent_outcomes: int,
    _get_all_tools: Callable[[], set[str]],
    _consecutive_same_tool: str,
    _consecutive_same_tool_count: int,
    _consecutive_same_content_hash: int,
) -> tuple[bool, str, int, int]:
    """§4's consecutive same-tool cap (refinement plan 1.5e), moved verbatim: whether this action is the
    ``_MAX_CONSECUTIVE_SAME_TOOL + 1``-th identical call in a row (then the proposal is dropped, a SYSTEM outcome
    tells the model, and the counters reset), with the three counters as they now stand. The body's
    ``continue`` after the reset is the ``True`` return; the caller ends the pass on it."""
    # ── Consecutive same-tool cap (respond loop prevention) ──────
    # Content-aware: only counts consecutive calls with the SAME
    # tool AND same params hash.  Different params (e.g. varied
    # send_message text) reset the counter — the cap targets
    # hallucination loops, not legitimate varied usage.
    _this_tool = action.get("tool_name", "") if isinstance(action, dict) else ""
    _this_params = action.get("params", {}) if isinstance(action, dict) else {}
    try:
        _this_content_hash = hash(
            str(sorted(_this_params.items())) if isinstance(_this_params, dict) else str(_this_params)
        )
    except Exception:
        _this_content_hash = 0
    if _this_tool == _consecutive_same_tool and _this_content_hash == _consecutive_same_content_hash:
        _consecutive_same_tool_count += 1
    elif _this_tool == _consecutive_same_tool:
        # Same tool, different content — reset count (varied usage)
        _consecutive_same_tool_count = 1
        _consecutive_same_content_hash = _this_content_hash
    else:
        _consecutive_same_tool = _this_tool
        _consecutive_same_tool_count = 1
        _consecutive_same_content_hash = _this_content_hash

    if _consecutive_same_tool_count > _MAX_CONSECUTIVE_SAME_TOOL:
        logger.warning(
            "Consecutive same-tool cap hit: %s called %d times with identical params — breaking chain",
            _this_tool,
            _consecutive_same_tool_count,
        )
        sim.log(
            "EXEC",
            f"CAPPED: {_this_tool} repeated {_consecutive_same_tool_count}x (identical params) — breaking respond loop",
        )
        # Fix 2: Tell the LLM the cap fired so it doesn't blindly
        # re-propose the same call.  Inject into recent_outcomes
        # which the prompt builder surfaces in the LLM's context.
        # NOTE: we deliberately do NOT write to pending_cli_input
        # because that's a single-slot user-input channel and
        # using it for system feedback risks silent overwrite of
        # real input (cross-confirmed by both review lenses).
        _other_tools = sorted(
            _lstate._advertisable_tools(_get_all_tools() - {_this_tool}, executor, autonomy_controller)
        )
        _cap_msg = (
            f"SYSTEM: '{safe_tool_name(_this_tool)}' was called {_consecutive_same_tool_count} times "
            f"with identical parameters — blocked to prevent a loop. "
            f"Try a DIFFERENT tool or different parameters. "
            f"Available tools: {', '.join([t for t in _other_tools if safe_tool_name(t) == t][:8]) if _other_tools else 'none'}"
        )
        note_system_outcome(recent_outcomes, max_recent_outcomes, _this_tool, _cap_msg, "consecutive_tool_cap")

        ctrl.pending_proposal = None
        _consecutive_same_tool_count = 0
        _consecutive_same_tool = ""
        _consecutive_same_content_hash = 0
        return True, _consecutive_same_tool, _consecutive_same_tool_count, _consecutive_same_content_hash
    return False, _consecutive_same_tool, _consecutive_same_tool_count, _consecutive_same_content_hash


def execute_pending_tick(
    *,
    ctrl: Any,
    memory: Any,
    executor: Any,
    sim: Any,
    state: Any,
    observation: Any,
    llm_worker: Any,
    context_pool: Any,
    autonomy_controller: Any,
    recent_outcomes: Any,
    max_recent_outcomes: int,
    aut_mode: str,
    _loop_sensor_encoder: Any,
    _loop_agent_id: str,
    _loop_nac: Any,
    _drive_relief_only: Any,
    _planning_liveness_on: bool,
    _execute_and_learn: Callable[..., Any],
    _book_machine_refusal: Callable[..., Any],
    _book_hard_rejection: Callable[..., Any],
    _get_all_tools: Callable[[], set[str]],
    _consecutive_same_tool: str,
    _consecutive_same_tool_count: int,
    _consecutive_same_content_hash: int,
) -> ExecuteOutcome:
    """§4 of one pass: the #1451 re-plan rule, then the pending proposal's action (module docstring)."""
    set_aside_run_end(ctrl, memory, executor, sim, _planning_liveness_on)  # re-plan once (#1451)
    if ctrl.pending_proposal and ctrl.pending_proposal.action:
        action = ctrl.pending_proposal.action
        confidence = ctrl.pending_proposal.confidence

        ctrl.pending_proposal = _sp._attach_live_situation(
            ctrl.pending_proposal,
            aut_mode=aut_mode,
            sensor_encoder=_loop_sensor_encoder,
            agent_id=_loop_agent_id,
            executor=executor,
        )

        _capped, _consecutive_same_tool, _consecutive_same_tool_count, _consecutive_same_content_hash = _same_tool_cap(
            action=action,
            ctrl=ctrl,
            executor=executor,
            autonomy_controller=autonomy_controller,
            sim=sim,
            recent_outcomes=recent_outcomes,
            max_recent_outcomes=max_recent_outcomes,
            _get_all_tools=_get_all_tools,
            _consecutive_same_tool=_consecutive_same_tool,
            _consecutive_same_tool_count=_consecutive_same_tool_count,
            _consecutive_same_content_hash=_consecutive_same_content_hash,
        )
        if _capped:
            return ExecuteOutcome(
                skip_rest_of_pass=True,
                consecutive_same_tool=_consecutive_same_tool,
                consecutive_same_tool_count=_consecutive_same_tool_count,
                consecutive_same_content_hash=_consecutive_same_content_hash,
            )

        # ───────────────────────────────────────────────────────────────
        # PARALLEL ACTIONS: Execute all together for efficient batching
        # ───────────────────────────────────────────────────────────────
        parallel_actions = getattr(ctrl.pending_proposal, "parallel_actions", [])
        if parallel_actions:
            all_parallel_actions = ctrl.pending_proposal.get_parallel_actions()
            _par_results, combined_results = _execute_parallel(
                agent_id=_loop_agent_id,
                actions=all_parallel_actions,
                executor=executor,
                autonomy_controller=autonomy_controller,
                confidence=confidence,
                reasoning=getattr(ctrl.pending_proposal, "reasoning", "") if ctrl.pending_proposal else "",
                recent_outcomes=recent_outcomes,
                max_recent=max_recent_outcomes,
                llm_worker=llm_worker,
                context_pool=context_pool,
                nac=_loop_nac,
                active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
                cluster_id=getattr(ctrl.pending_proposal, "cluster_id", None),
                clusters=getattr(ctrl.pending_proposal, "clusters", None),
                drive_relief_only=_drive_relief_only,
                proposal=ctrl.pending_proposal,
            )

            # Queue as a followup for the next LLM call
            ctrl.pending_action_followup = ActionFollowup(
                tool="batched_exploration",
                result=combined_results,
                original_query=ctrl.pending_proposal.triggering_input,
                followup_type="process",
                mode=operational_mode(executor, state),
                timestamp=time.time(),
            )
            logger.info("Batched exploration complete, queuing followup for LLM")

            # Clear proposal - will be handled via followup
            ctrl.pending_proposal = None
            return ExecuteOutcome(
                skip_rest_of_pass=True,
                consecutive_same_tool=_consecutive_same_tool,
                consecutive_same_tool_count=_consecutive_same_tool_count,
                consecutive_same_content_hash=_consecutive_same_content_hash,
            )  # Skip normal execution flow

        # ───────────────────────────────────────────────────────────────
        # PLANNING MODE: Check if this proposal requires user approval
        # ───────────────────────────────────────────────────────────────
        if ctrl.gate_plan_approval(  # a plan: run once if approved, parked for the person, or refused
            execute_and_learn=_execute_and_learn,
            book_machine_refusal=_book_machine_refusal,
            observation=observation,
            sim=sim,
        ):
            return ExecuteOutcome(
                skip_rest_of_pass=True,
                consecutive_same_tool=_consecutive_same_tool,
                consecutive_same_tool_count=_consecutive_same_tool_count,
                consecutive_same_content_hash=_consecutive_same_content_hash,
            )

        logger.info("Executing LLM proposal: tool=%s, confidence=%.2f", action.get("tool_name"), confidence)

        _log_goal_proposed(
            tool=action.get("tool_name"), confidence=confidence, reasoning=ctrl.pending_proposal.reasoning
        )

        _verdict = autonomy_controller.action_verdict(action, confidence=confidence)  # typed (#1185)
        can_execute, reason = _verdict.allowed, _verdict.reason

        _log_autonomy_check(action, can_execute, reason, confidence)  # moved out (rule T, #1185)

        if can_execute:
            # Execute it and book what it teaches: ONE function for every path that runs an action
            # the agent chose (#1133; moved verbatim from here into tool_dispatch.execute_and_learn).
            _outcome = _execute_and_learn(
                action=action,
                confidence=confidence,
                proposal=ctrl.pending_proposal,
                observation=observation,
                human_involved=False,
            )
            if _outcome.followup is not None:
                ctrl.pending_action_followup = _outcome.followup

            ctrl.pending_proposal = None

        elif autonomy_controller.current_level == AutonomyLevel.PLANNING:
            ctrl.queue_for_approval(action, confidence=confidence, sim=sim)  # Queue for human approval

        else:
            # A confirmation request (a reason a person's approval satisfies), not a hard rejection (#1185: typed)
            if _verdict.needs_confirmation:
                ctrl.park_confirmation(action, sim=sim)
            else:  # Hard rejection: booked, then fed back to the model or the turn ends (#1413, tool_dispatch)
                _rejected = _book_hard_rejection(
                    proposal=ctrl.pending_proposal,
                    action=action,
                    reason=reason,
                    confidence=confidence,
                    active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
                )
                ctrl.pending_action_followup = _rejected or ctrl.pending_action_followup
            ctrl.pending_proposal = None

    return ExecuteOutcome(
        skip_rest_of_pass=False,
        consecutive_same_tool=_consecutive_same_tool,
        consecutive_same_tool_count=_consecutive_same_tool_count,
        consecutive_same_content_hash=_consecutive_same_content_hash,
    )
