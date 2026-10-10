"""The agent loop's PLANNING approved path, section 5 (1.3.2 decomposition, slice 4; #1085 PR-a).

At ``AutonomyLevel.PLANNING`` the loop queues every proposal in ``AutonomyController.proposal_queue`` (section 4).
``drain_approved`` executes the entries an approver marked approved. Today that approver is an EMBEDDER calling
``proposal_queue.approve``; this module opens no other approval route (the approval surface is #1185).

Owner decisions (2026-10-08, and 2026-10-09 after the review):

- **One dispatch.** Each approved action runs through the run's bound ``tool_dispatch.execute_and_learn`` with
  ``human_involved=True``, the function the autonomous and the SUPERVISED-confirmed paths call (#1133), so an
  approved action learns exactly what it would have learned autonomously. The credit and the capture are keyed to
  the situation the action was PROPOSED in (``Proposal.source``, #1083). The function never writes the controller:
  it returns the follow-up and the caller assigns it.
- **Approval lifts only the PLANNING level.** ``AutonomyController.approved_action_blocker`` is checked before each
  execution: the critical safety constraints and the pause (``approval_blocker``, the non-level half of
  ``can_execute_action``) and the supervision policy's hard denials (``SupervisionPolicy.hard_deny``: forbidden
  prefixes, categories and tools; owner decision S7). A blocked entry is refused (``blocked``) and not executed.
  The policy's "needs approval" checks (the ``allowed_tools`` list, confidence, ``requires_confirmation``, the
  sandbox/CWD rules) are what the approval satisfies (owner decision 2026-10-09). **The pause check is narrow:**
  while the controller is paused the loop's gate idles and this drain does not run at all, so the check catches
  only a pause landing after this tick's gate (later in the tick, or during the drain). Approved entries do not
  expire, and a later ``resume()`` would run them; that is #1185's.
- **One entry at a time.** ``ProposalQueue.pop_approved`` takes the oldest approved entry, which then executes
  before the next is taken. If one step raises (any ``BaseException``), every entry still approved is refused
  (``drain_aborted``) and the original exception propagates, as it does on the main path.
- **The DRAIN's refusals book no NAc.** ``blocked`` and ``drain_aborted`` are machine refusals, booked through
  ``LoopRun.book_machine_refusal``: the outcome window and the LLM's carryover, no NAc, no situation, no goal
  credit (``no_action`` is reported, not booked). (Elsewhere a refusal still books NAc: §4's hard rejection and
  ``handle_confirmation``'s "no", sim AUTO_REJECT included; #1185.)

A refusal is ordered: the entry is rejected (``status``/``rejected_reason``), then the typed ``plan_refused`` event
and the audit entry, then the booking (#1185 S5), so a raise in the report or the booking cannot leave the entry
approved; such a raise aborts the drain like any other.

**Arguments.** Explicit keywords, never the loop's ``LoopRun`` (rule (d) of the roadmap's import-direction
paragraph). Logs on the ``maxim.runtime.agent_loop`` logger. Characterization:
``tests/unit/test_approved_path_characterization.py`` (written before the move); gates:
``tests/unit/test_approved_path_learns_1085.py``.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from maxim.runtime.loop_state import operational_mode
from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import log_agentic

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController, Proposal
    from maxim.runtime.loop_types import ActionFollowup
    from maxim.runtime.tool_dispatch import ExecutionOutcome

# The SAME logger object as ``agent_loop.logger``, so records keep the ``maxim.runtime.agent_loop`` name.
logger = logging.getLogger("maxim.runtime.agent_loop")

# Typed refusal reasons (the ``plan_refused`` event's ``reason`` and the entry's ``rejected_reason``).
BLOCKED = "blocked"
DRAIN_ABORTED = "drain_aborted"
NO_ACTION = "no_action"


def drain_approved(
    *,
    autonomy_controller: AutonomyController,
    execute_and_learn: Callable[..., ExecutionOutcome],
    book_machine_refusal: Callable[..., None],
    executor: Any,
    observation: Any,
    state: Any,
    sim: Any,
) -> ActionFollowup | None:
    """Execute every approved PLANNING proposal, oldest first; return the last follow-up any of them queued.

    ``execute_and_learn`` / ``book_machine_refusal`` are the run's bound callables (``LoopRun``); ``executor`` is the
    run's, whose operational mode a refusal's audit entry records (#963); ``observation`` is this tick's, the
    observation the capture stores (the answer-time observation, as on the confirmed path).

    Exception-safe: the WHOLE per-entry step (the blocker, the execution, that entry's own refusal) runs inside
    one ``try``. On a raise (any ``BaseException``, ``KeyboardInterrupt`` included), every entry still approved
    is first popped and rejected (``drain_aborted``) with no I/O, then each is reported and booked under its own
    guard, and the ORIGINAL exception is re-raised: an ``Exception`` raised in that sweep is logged, never re-raised
    in its place. A second interrupt (a ``BaseException``) raised during the sweep does escape, but by then every
    entry is already rejected, because the reject pass runs before any I/O. So no approved entry survives an
    aborted drain to run later.
    """
    queue = autonomy_controller.proposal_queue
    refuser = _Refuser(
        autonomy_controller=autonomy_controller,
        book_machine_refusal=book_machine_refusal,
        executor=executor,
        state=state,
        sim=sim,
    )
    followup: ActionFollowup | None = None
    while (proposal := queue.pop_approved()) is not None:
        try:
            outcome = _drain_one(
                proposal,
                autonomy_controller=autonomy_controller,
                execute_and_learn=execute_and_learn,
                observation=observation,
                refuser=refuser,
            )
        # BaseException: a KeyboardInterrupt must not leave entries approved either. Re-raised, never swallowed.
        except BaseException as exc:
            _abort_drain(queue, exc, refuser=refuser)
            raise
        if outcome is not None and outcome.followup is not None:
            followup = outcome.followup
    return followup


def _drain_one(
    proposal: Proposal,
    *,
    autonomy_controller: AutonomyController,
    execute_and_learn: Callable[..., ExecutionOutcome],
    observation: Any,
    refuser: _Refuser,
) -> ExecutionOutcome | None:
    """One drained entry: refuse it (typed) or execute it. ``None`` when it was refused."""
    action = proposal.action
    if not action:
        refuser.refuse(proposal, reason=NO_ACTION, detail="the approved proposal carries no action")
        return None
    blocker = autonomy_controller.approved_action_blocker(action)
    if blocker is not None:
        refuser.refuse(proposal, reason=BLOCKED, detail=blocker)
        return None
    return execute_and_learn(
        action=action,
        confidence=proposal.confidence,
        # The situation it was PROPOSED in (#1083). A Proposal built outside the loop has no source: it is then
        # its own record (reasoning, citations) and credits no situation.
        proposal=proposal.source if proposal.source is not None else proposal,
        observation=observation,
        human_involved=True,
    )


def _abort_drain(queue: Any, exc: BaseException, *, refuser: _Refuser) -> None:
    """Refuse every entry still approved after ``exc``: reject them all first (no I/O), then report and book each
    under its own guard, so a failing report cannot leave an entry approved or replace ``exc``."""
    rest: list[Proposal] = []
    while (entry := queue.pop_approved()) is not None:
        _reject(entry, DRAIN_ABORTED)
        rest.append(entry)
    detail = f"an earlier approved action raised {type(exc).__name__}: {exc}"[:300]
    for entry in rest:
        try:
            refuser.report_and_book(entry, reason=DRAIN_ABORTED, detail=detail)
        except Exception:
            log_swallowed_exception(site="loop_planning.py:_abort_drain:drain_aborted")


def _reject(proposal: Proposal, reason: str) -> None:
    proposal.status = "rejected"
    proposal.rejected_reason = reason


@dataclasses.dataclass(frozen=True)
class _Refuser:
    """The run's handles a refusal needs. Runtime-ephemeral (one drain), so CC3 does not apply."""

    autonomy_controller: AutonomyController
    book_machine_refusal: Callable[..., None]
    executor: Any
    state: Any
    sim: Any

    def refuse(self, proposal: Proposal, *, reason: str, detail: str) -> None:
        """Refuse one drained entry, typed: reject it, then report and book it (#1185 S5's order)."""
        _reject(proposal, reason)
        self.report_and_book(proposal, reason=reason, detail=detail)

    def report_and_book(self, proposal: Proposal, *, reason: str, detail: str) -> None:
        """The ``plan_refused`` event, the log and sim lines, the audit entry, then the booking for the LLM (no NAc;
        an entry with no action is not booked)."""
        tool_name = str((proposal.action or {}).get("tool_name") or "unknown")
        log_agentic(
            "agent_loop",
            "plan_refused",
            {"reason": reason, "detail": detail, "tool": tool_name, "proposal_id": proposal.id},
            level="WARNING",
        )
        logger.warning("Approved proposal %s refused (%s): %s -- %s", proposal.id, reason, tool_name, detail)
        self.sim.log("PIPELINE", f"Approved proposal refused ({reason}): {tool_name} -- {detail}")
        self.autonomy_controller.log_action(
            action_type="rejected",
            action=proposal.action,
            reasoning=f"Refused ({reason}): {detail}",
            mode=operational_mode(self.executor, self.state),
            confidence=proposal.confidence,
            human_involved=False,
        )
        if proposal.action:
            self.book_machine_refusal(
                tool_name=tool_name,
                error=f"Refused ({reason}): {detail}",
                reasoning=proposal.reasoning or "",
            )
