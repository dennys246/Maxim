"""LoopController: Extracted phase methods for the agentic loop.

Phase 2 of loop modularization — turns the monolithic ``run_agentic_loop``
function into a class with discrete, testable phase methods.

The controller manages all transient loop state as instance attributes
instead of local variables scattered across a 2,300-line function.
"""

from __future__ import annotations

import collections
import logging
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from maxim.agents.llm_worker import LLMProposal
from maxim.runtime.tool_dispatch import safe_agent_name as _safe_agent_name, tool_to_run
from maxim.runtime.dn_controller import DefaultNetworkController
from maxim.runtime.loop_state import no_person_to_ask, operational_mode
from maxim.runtime.loop_types import (
    ActionFollowup,
    PendingConfirmation,
    TimeoutRetry,
)
from maxim.utils.input_source import InputSource
from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import log_agentic

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController
    from maxim.agents.llm_worker import LLMWorker
    from maxim.evaluation.base import Evaluator
    from maxim.runtime.tool_dispatch import ExecutionOutcome

logger = logging.getLogger(__name__)


class PlanningLivenessExhausted(RuntimeError):
    """A sim's planning turns failed repeatedly and the bounded retry budget
    is spent (bugs ledger D13).

    Raised by ``run_agentic_loop`` AFTER its normal teardown (state persist +
    bio session end) so the sim aborts loudly instead of idling forever on a
    silently-dropped planning turn. Homed here, beside the
    ``record_planning_failure`` state machine that decides it — ``runtime/
    stall_threshold.py`` declares itself the single source of truth for
    stall-THRESHOLD derivation, so a loop-control exception did not belong
    there (pre-merge review, architecture lens S3).

    Only raised by a loop that opted in via ``planning_liveness=True`` — in
    practice the sim orchestrator, the one loop with no other wake source
    between turns.
    """

    def __init__(self, message: str, *, finish_status: str = "llm_wedged") -> None:
        super().__init__(message)
        self.finish_status = finish_status


class LoopController:
    """Manages agentic loop state and provides phase methods.

    Each numbered section of the original ``run_agentic_loop`` becomes a
    method here.  The ``run()`` method orchestrates them in the canonical
    order: observe → parse input → check proposals → execute → submit → persist.
    """

    # ── Construction ─────────────────────────────────────────────────────

    def __init__(
        self,
        agent: Any,
        environment: Any,
        state: Any,
        memory: Any,
        decision_engine: Any,
        executor: Any,
        *,
        autonomy_controller: AutonomyController,
        llm_worker: LLMWorker | None = None,
        default_network: Any | None = None,
        hippocampus: Any | None = None,
        memory_hub: Any | None = None,
        evaluators: list[Evaluator] | None = None,
        max_steps: int = 0,
        run_id: str | None = None,
        state_path: str | None = None,
        stop_event: Any | None = None,
        on_step: Any | None = None,
        on_event: Any | None = None,
        idle_sleep_s: float = 0.05,
        persist_every_n_steps: int = 10,
        target_hz: float = 30.0,
        use_tool_prompting: bool = True,
        protocol_registry: Any | None = None,
        percept_source: Any | None = None,
        action_sink: Any | None = None,
        pain_bus: Any | None = None,
    ) -> None:
        # ── Core dependencies ────────────────────────────────────────────
        self.agent = agent
        self.environment = environment
        self.state = state
        self.memory = memory
        self.decision_engine = decision_engine
        self.executor = executor
        self.autonomy_controller = autonomy_controller
        self.llm_worker = llm_worker
        self.default_network = default_network
        self.hippocampus = hippocampus
        self.memory_hub = memory_hub
        self.evaluators = evaluators or []
        self.stop_event = stop_event
        self.on_step = on_step
        self.on_event = on_event
        self.protocol_registry = protocol_registry
        self.percept_source = percept_source
        self.action_sink = action_sink
        self.pain_bus = pain_bus
        self.use_tool_prompting = use_tool_prompting

        # ── Loop configuration ───────────────────────────────────────────
        self.max_steps = int(max_steps or 0)
        self.run_id = run_id or time.strftime("%Y-%m-%d_%H%M%S")
        self.idle_sleep_s = idle_sleep_s
        self.persist_every_n_steps = persist_every_n_steps
        self.target_period = 1.0 / target_hz

        # ── Agent identity & persistence ─────────────────────────────────
        self.agent_name = _safe_agent_name(agent)
        # The run's claimed state file (``loop_setup.build_loop_run`` passes it, #1264). A controller built on its own
        # has none: the controller never writes it, so it computes no path it would not use.
        self.state_path: str | None = state_path

        # ── Transient loop state (typed — replaces local variables) ──────
        # Backing field for the pending_proposal property below; planning-
        # liveness fields are initialized further down, so the raw field is
        # set here rather than going through the setter.
        self._pending_proposal: LLMProposal | None = None
        self.pending_next_actions: list[dict[str, Any]] = []
        self.pending_plan_proposal: LLMProposal | None = None
        self.pending_action_followup: ActionFollowup | None = None
        # #1413: the rejection follow-ups the current user turn has had (``tool_dispatch.book_hard_rejection``) and
        # that turn's input. A turn starts when a fresh user input arrives (``note_user_input``), never by comparing
        # proposal text: a repeated utterance is a new turn, and deliberated proposals carry no input.
        self.rejection_turn: str | None = None
        self.rejection_followups: int = 0
        # A proposal parked for the user's yes/no (#1133: typed, on the controller, off ``state.data``).
        self.pending_confirmation: PendingConfirmation | None = None
        self.pending_prefetch: Any | None = None
        self.last_llm_submit_time: float = 0.0
        self.llm_submit_interval: float = 0.5
        # ── Planning liveness (bugs ledger D13) ──────────────────────────
        # last_proposal_time: when get_latest_proposal() last returned ANY
        # proposal (even one that was subsequently dropped as stale/fallback).
        # last_proposal_time < last_llm_submit_time at idle-gate time means a
        # planning submit went out and NOTHING ever came back — a lost turn.
        self.last_proposal_time: float = 0.0
        self.planning_failure_streak: int = 0
        self.planning_retry_limit: int = 3
        self.planning_transport_failure_streak: int = 0
        self.planning_transport_retry_limit: int = 3
        self.planning_exhausted: bool = False
        self.planning_exhausted_status: str = "llm_wedged"
        self.planning_exhausted_reason: str = ""
        self.processed_cli_inputs: collections.deque[str] = collections.deque(maxlen=20)
        # #1451: a run-ending answer was set aside for the inputs held behind it, and the re-plan's answer has not
        # run yet. "Once": while it is set, a run-ending answer runs (``set_aside_run_end``).
        self.run_end_set_aside: bool = False
        self.recent_outcomes: list[dict[str, Any]] = []
        self.max_recent_outcomes: int = 10
        self.agent_states: list[dict[str, Any]] = []
        self.last_surfaced_tools: list[str] = []
        # Assigned by ``run_agentic_loop`` right after construction (it builds both from the
        # loop's own config); declared here so the attributes exist on the type.
        self.context_pool: Any = None
        self.prefetcher: Any = None

        # ── Default Network ──────────────────────────────────────────────
        self.dn_ctrl = DefaultNetworkController(default_network)
        self.dn_enabled = self.dn_ctrl.enabled

    def note_user_input(self, text: str) -> None:
        """A fresh user input arrived, by any channel (the observation, the interactive CLI, voice, the API seed):
        a new turn, so its rejection count starts again (#1413). Called where the loop settles the tick's input."""
        self.rejection_turn, self.rejection_followups = text, 0

    # ── Pending proposal (planning-liveness recovery point) ──────────────

    @property
    def pending_proposal(self) -> LLMProposal | None:
        return self._pending_proposal

    @pending_proposal.setter
    def pending_proposal(self, proposal: LLMProposal | None) -> None:
        """Installing an executable proposal IS the recovery signal.

        The reset used to live at one call site in ``run_agentic_loop``'s
        section 2, which three other install sites bypassed — the multi-step
        `next_actions` continuation and both PFC-deliberation sites — so a sim
        recovering through those carried a stale failure streak forever and a
        single later failure could abort a run that had been executing fine
        for many turns (pre-merge review, executor lens F2). Making the reset
        a consequence of the ASSIGNMENT means a future install site cannot
        forget it: the invariant is in the type, not in a call site.
        """
        self._pending_proposal = proposal
        if proposal is not None:
            self.reset_planning_failures()

    # ── Typed state helpers ──────────────────────────────────────────────

    def get_timeout_retry(self) -> TimeoutRetry | None:
        raw = self.state.data.get("pending_timeout_retry")
        if raw is None:
            return None
        # Plan 3.5 R2: fall back to the current agent-level LLM timeout
        # default if the persisted state is missing timeout_s. Was
        # hardcoded 60.0 pre-plan (mesh-era value).
        from maxim.agents.llm_worker import DEFAULT_LLM_CALL_TIMEOUT_S

        return TimeoutRetry(
            original_request=raw.get("original_request"),
            timeout_s=raw.get("timeout_s", DEFAULT_LLM_CALL_TIMEOUT_S),
        )

    def set_timeout_retry(self, tr: TimeoutRetry | None) -> None:
        if tr is None:
            self.state.data.pop("pending_timeout_retry", None)
        else:
            self.state.data["pending_timeout_retry"] = {
                "original_request": tr.original_request,
                "timeout_s": tr.timeout_s,
            }

    def clear_pending_user_input(self) -> None:
        """Remove all pending user input keys from state."""
        self.state.data.pop("pending_user_input", None)
        self.state.data.pop("pending_user_input_time", None)
        self.state.data.pop("pending_user_input_source", None)

    # ── Planning liveness (bugs ledger D13) ──────────────────────────────

    def record_planning_failure(self, *, reason: str, exhausted_status: str = "llm_wedged") -> str:
        """Bounded-retry state machine for failed planning turns.

        A planning submit that ends in parse failure, an invalid response,
        a dropped proposal, or a silently-expired await window must LOUDLY
        reschedule or abort — never fall through to idle (the D13 livelock).
        This method only decides; the caller does the logging and requeueing.

        Returns:
            ``"retry"`` — within budget; caller requeues the request and
            re-stamps ``last_llm_submit_time``.
            ``"exhausted"`` — this failure spent the budget; caller emits the
            loud abort. ``planning_exhausted`` latches True.
            ``"already_exhausted"`` — budget was already spent; caller must
            not retry or re-log.
        """
        if self.planning_exhausted:
            return "already_exhausted"
        self.planning_failure_streak += 1
        if self.planning_failure_streak > self.planning_retry_limit:
            self.planning_exhausted = True
            self.planning_exhausted_status = exhausted_status
            self.planning_exhausted_reason = reason
            return "exhausted"
        return "retry"

    def record_planning_transport_failure(self, *, reason: str) -> str:
        """Bound retries for a worker job that could not run or publish.

        Transport failures do not spend the model-planning budget, but they
        cannot retry forever either. Exhaustion is reported separately as
        ``worker_unavailable`` so a healthy model is never labeled wedged.
        """
        if self.planning_exhausted:
            return "already_exhausted"
        self.planning_transport_failure_streak += 1
        if self.planning_transport_failure_streak > self.planning_transport_retry_limit:
            self.planning_exhausted = True
            self.planning_exhausted_status = "worker_unavailable"
            self.planning_exhausted_reason = reason
            return "exhausted"
        return "retry"

    def reset_planning_transport_failures(self) -> None:
        """A worker job completed and published; transport is healthy."""
        self.planning_transport_failure_streak = 0

    def reset_planning_failures(self) -> None:
        """An executable proposal arrived — planning is alive again."""
        self.planning_failure_streak = 0
        self.reset_planning_transport_failures()
        self.planning_exhausted = False
        self.planning_exhausted_status = "llm_wedged"
        self.planning_exhausted_reason = ""

    # ── Tool registry cache ──────────────────────────────────────────────

    def get_all_tools(self) -> set[str]:
        """The tools the prompt roster may offer: the registry's ``advertised()`` (#1042: a decoy stays dispatchable but
        is never offered). Every registry the executor takes defines it, so a registry without one fails loudly here
        rather than silently advertising its decoys."""
        registry = getattr(self.executor, "registry", None)
        if registry is None:
            return set()
        return set(registry.advertised())

    @staticmethod
    def _clear_confirmation_prompt() -> None:
        """Reset the input panel after a confirmation is resolved."""
        try:
            from maxim.simulation.sim_logger import get_active_display

            display = get_active_display()
            if display is not None:
                display.set_prompt("> ")
                display.set_urgent(False)
        except Exception:
            log_swallowed_exception()  # a display that cannot reset its prompt must not stop the loop (#1134)

    @staticmethod
    def _display_confirmed_outcome(pc: PendingConfirmation, outcome: ExecutionOutcome) -> None:
        """Show what a confirmed action did. Display only: the outcome is already booked, so a display
        failure is reported (Stage 1) and never turns a success into "Action failed"."""
        from maxim.simulation.sim_logger import display_action, display_status

        try:
            if outcome.success:
                display_action(pc.tool_name, pc.params)
                if outcome.output:
                    display_status(f"Result: {str(outcome.output)}")
            else:
                display_status(f"Action failed: {outcome.error or 'unknown error'}")
        except Exception:
            log_swallowed_exception()

    # ── Phase: Answer a pending gating prompt ────────────────────────────

    def answer_pending_prompt(
        self,
        cli_text: str,
        *,
        execute_and_learn: Callable[..., ExecutionOutcome],
        book_refusal: Callable[..., None],
        book_machine_refusal: Callable[..., None],
        observation: Any,
        sim: Any,
        answer_source: InputSource,
        answers: str,
    ) -> bool:
        """The tick's input as the answer to whichever gating prompt is pending (#1185): the action confirmation,
        the timeout retry, then the plan approval. True when the input was the answer (consumed: the caller drops
        it, as the next turn it is not). A tool's ``[ACTION_FOLLOWUP`` result is never an answer to any prompt.
        The callables are the run's (``LoopRun``), ``observation`` this tick's, ``sim`` the run's adapter, and
        ``answer_source`` who this tick's input came from (``UserInput.source``; required: these handlers run before
        ``store_user_input``, so ``state.data["pending_user_input_source"]`` still names the previous input's), and
        ``answers`` the prompt the loop's own reply answers (``UserInput.answers``; required, "" for any other input:
        a reply routed by the old order could answer the wrong prompt)."""
        from maxim.runtime.loop_perception import is_tool_followup

        if is_tool_followup(cli_text):
            return False
        # The loop's own reply names the prompt it answers (#1185), whoever made it (the sim's policy or the loop): a
        # timeout retry's "yes" must never reach a confirmation pending at the same time (it would run the parked
        # action, or be refused there, and strand the retry), and vice versa.
        if answers == "timeout_retry":
            return self.handle_timeout_retry(cli_text, sim=sim, answer_source=answer_source)
        confirmed = self.handle_confirmation(
            cli_text,
            execute_and_learn=execute_and_learn,
            book_refusal=book_refusal,
            book_machine_refusal=book_machine_refusal,
            observation=observation,
            sim=sim,
            answer_source=answer_source,
        )
        if confirmed or answers == "confirmation":  # a confirmation's own reply answers nothing else
            return confirmed
        if self.handle_timeout_retry(cli_text, sim=sim, answer_source=answer_source):
            return True
        return self.handle_plan_approval(
            cli_text,
            book_refusal=book_refusal,
            execute_and_learn=execute_and_learn,
            book_machine_refusal=book_machine_refusal,
            observation=observation,
            sim=sim,
            answer_source=answer_source,
        )

    # ── Phase: Handle confirmation input ─────────────────────────────────

    def handle_confirmation(
        self,
        cli_text: str,
        *,
        execute_and_learn: Callable[..., ExecutionOutcome],
        book_refusal: Callable[..., None],
        book_machine_refusal: Callable[..., None],
        observation: Any,
        sim: Any,
        answer_source: InputSource,
    ) -> bool:
        """Process user input when a confirmation is pending.

        Returns True if the input was consumed (caller should skip further processing).

        A "yes" is an approval, so it takes the one check every approval takes (#1185, corrected owner decision A):
        ``AutonomyController.approved_action_denial(approver=answer_source)``. A pause or a forbid that landed while
        the prompt waited refuses it, and so does a "yes" from the loop itself (the non-interactive auto-yes, an
        adapter's default: ``InputSource.LOOP`` satisfies no reason that needs a person), typed, as a machine refusal
        (no NAc), shown to the person; the plan prompt and this one can never disagree.

        A "yes" runs the parked action through the run's ``execute_and_learn`` (#1133), the SAME function
        the autonomous path calls, so a confirmed action learns exactly what it would have learned
        unconfirmed: its side effects, the hub's ``agent_id``, the situation it was PROPOSED in
        (``pc.source``), the capture, the plan outcome, ``environment.step`` and the main path's follow-up
        rule. The function books before it returns; the display runs after it, inside its own guard, so a
        display error cannot undo a recorded outcome or show "Action failed"; the bookkeeping runs in
        ``finally``: a raise from ``execute_and_learn`` (its pre-execution snapshot, or its own ``except``
        branch) propagates as it does on the main path, with the confirmation already cleared. A "no" is
        booked through ``book_refusal``. Both callables are the run's own (``LoopRun.execute_and_learn`` /
        ``LoopRun.book_refusal``), passed as required keywords: the controller holds no run state of its own
        for them.
        """
        # Import at FUNCTION scope, before any branch (2026-08-04 live
        # crash): the import previously lived inside the approve branch's
        # try-block, which makes ``display_status`` a LOCAL name for the
        # whole function — the cancel/modification branches then raised
        # UnboundLocalError and killed the agentic loop the moment a user
        # typed anything while a confirmation was pending.
        from maxim.simulation.sim_logger import display_status

        pc = self.pending_confirmation
        if pc is None:
            return False

        from maxim.runtime.approval import detect_approval_intent

        # One vocabulary for every gating prompt (#1185): only a plain yes/no answers; an ambiguous reply re-asks.
        intent, _ = detect_approval_intent(cli_text)

        if intent == "approve":
            logger.info("User confirmed action: %s", pc.tool_name)
            log_agentic("agent_loop", "user_confirmed", {"tool": pc.tool_name, "approved": True})
            denial = self.autonomy_controller.approved_action_denial(
                pc.action, confidence=pc.confidence, approver=answer_source
            )
            if denial is not None:
                self._refuse_confirmed(pc, denial, book_machine_refusal=book_machine_refusal, sim=sim)
                return True
            try:
                outcome = execute_and_learn(
                    action=pc.action,
                    confidence=pc.confidence,
                    proposal=pc.source,
                    observation=observation,
                    human_involved=True,
                )
                if outcome.followup is not None:
                    self.pending_action_followup = outcome.followup
                    logger.info(
                        "Confirmed action %s queued follow-up (type=%s)", pc.tool_name, outcome.followup.followup_type
                    )
                self._display_confirmed_outcome(pc, outcome)
            finally:
                self.pending_confirmation = None
                # The tick's input slots were read and popped by loop_perception.take_user_input; anything there now arrived
                # after that read and is the NEXT tick's (a person's input that waited behind a loop reply; Session R)
                self.clear_pending_user_input()
                self.pending_proposal = None
                self._clear_confirmation_prompt()
            return True

        elif intent == "reject":
            logger.info("User rejected action: %s", pc.tool_name)
            log_agentic("agent_loop", "user_confirmed", {"tool": pc.tool_name, "approved": False})
            self.autonomy_controller.log_action(
                action_type="rejected",
                action=pc.action,
                reasoning="User rejected confirmation",
                mode=operational_mode(self.executor, self.state),
                confidence=pc.confidence,
                human_involved=True,
            )
            display_status("Action cancelled by user")

            book_refusal(
                source=pc.source,
                tool_name=pc.tool_name,
                error="User rejected this action",
                reasoning=pc.reasoning,
            )

            self.pending_confirmation = None
            self.clear_pending_user_input()
            self.pending_proposal = None
            self._clear_confirmation_prompt()
            return True

        elif intent == "unknown":
            self._reask(
                "confirmation_reask",
                pc.tool_name,
                f"Confirm {pc.tool_name}? Type 'yes' or 'no', or describe changes.",
                sim=sim,
            )
            return True

        else:
            # Treat as modification request
            logger.info("User requested modification for action: %s", pc.tool_name)
            log_agentic(
                "agent_loop",
                "user_modification_request",
                {"tool": pc.tool_name, "modification": cli_text[:100]},
            )
            self.state.data["pending_modification"] = {
                "original_action": pc.action,
                "original_reasoning": pc.reasoning,
                "original_tool_name": pc.tool_name,
                "user_modification": cli_text,
                "timestamp": time.time(),
            }
            self.pending_confirmation = None
            display_status(f'Modification requested — revising action based on: "{cli_text}"')
            self.clear_pending_user_input()
            self._clear_confirmation_prompt()
            return True

    def _refuse_confirmed(
        self,
        pc: PendingConfirmation,
        denial: tuple[str, str],
        *,
        book_machine_refusal: Callable[..., None],
        sim: Any,
    ) -> None:
        """A confirmed action ``approved_action_denial`` still refuses: refused typed through the approved path's
        own refusal (``loop_planning.refuse_plan``: ``plan_refused``, the audit entry, the person told, a machine
        refusal with no NAc), and the confirmation cleared."""
        from maxim.runtime.loop_planning import refuse_plan

        if pc.source is not None:
            refuse_plan(
                pc.source,
                reason=denial[0],
                detail=denial[1],
                autonomy_controller=self.autonomy_controller,
                book_machine_refusal=book_machine_refusal,
                executor=self.executor,
                state=self.state,
                sim=sim,
            )
        else:  # a confirmation built outside the loop: no proposal to book, so it is only reported
            logger.warning("Confirmed action %s refused (%s): %s", pc.tool_name, denial[0], denial[1])
        self.pending_confirmation = None
        self.clear_pending_user_input()
        self.pending_proposal = None
        self._clear_confirmation_prompt()

    def park_confirmation(self, action: dict[str, Any], *, sim: Any) -> None:
        """Park ``pending_proposal`` behind the SUPERVISED confirmation prompt (moved out of
        ``_run_agentic_loop_body``'s §4, rule T for #1185): the sim response policy answers it, or the prompt is
        shown, or (non-interactive) the policy's auto-yes answers it. The answer arrives as the next tick's input."""
        # The typed record keeps the proposal as its ``source``, so the confirmed action is
        # credited to the situation it was proposed in (#1133; the proposal itself is cleared
        # below, as for every non-executed branch).
        _pending = PendingConfirmation.from_proposal(self.pending_proposal)
        tool_name = _pending.tool_name
        params = action.get("params", {})

        # In sim mode, auto-resolve via response policy instead of blocking
        sim_response = sim.resolve_confirmation(_pending.policy_view())
        if sim_response is not None:
            # Inject the response as if the user typed it
            sim.log("PIPELINE", f"Auto-resolved confirmation for {tool_name}: {sim_response}")
            self.pending_confirmation = _pending
            self.state.data["pending_loop_reply"] = sim_response
            self.state.data["pending_loop_reply_source"] = sim.answer_source().value  # the sim's, or the loop's
            self.state.data["pending_loop_reply_for"] = "confirmation"
        else:
            # Production mode: display prompt if interactive, auto-resolve if not
            from maxim.simulation.sim_logger import should_prompt, display_scene

            if should_prompt("confirmation"):
                display_scene(f"\n  Action requires confirmation: {tool_name}")
                param_lines = []
                for key, value in params.items():
                    param_lines.append(f"    {key}: {str(value)}")
                if param_lines:
                    display_scene("\n".join(param_lines))
                if self.pending_proposal.reasoning:
                    display_scene(f"  Reasoning: {self.pending_proposal.reasoning}")
                # Show confirmation prompt in the input panel so typed
                # characters are visible with question context.
                try:
                    from maxim.simulation.sim_logger import get_active_display

                    _conf_display = get_active_display()
                    if _conf_display is not None:
                        _conf_display.set_prompt(f"Confirm {tool_name}? Type 'yes' or 'no'\n\n> ")
                        _conf_display.set_urgent(True)
                except Exception:
                    log_swallowed_exception()  # the input panel is display only; the scene line still prompts
                    display_scene("  Type 'yes' or 'no' to confirm/reject:")
            else:
                # Non-interactive: the loop answers itself, a MACHINE "yes" (#1185): it satisfies no reason that
                # needs a person, so handle_confirmation refuses it typed (needs_person) instead of running it
                self.state.data["pending_loop_reply"] = "yes"
                self.state.data["pending_loop_reply_source"] = InputSource.LOOP.value
                self.state.data["pending_loop_reply_for"] = "confirmation"
                sim.log("PIPELINE", f"Auto-approved (non-interactive): {tool_name}")

            self.pending_confirmation = _pending
        self.pending_proposal = None  # parked: the confirmation holds it now, as queue_for_approval's queue does

    def queue_for_approval(self, action: dict[str, Any], *, confidence: float, sim: Any) -> None:
        """Queue ``pending_proposal`` for approval at PLANNING (moved out of ``_run_agentic_loop_body``'s §4, rule T
        for #1185). An embedder's ``proposal_queue.approve`` approves it. In OFF mode (``interactive="off"``,
        ``no_person_to_ask``) the approval is the MACHINE's (owner decisions 2026-11-21): it lifts the PLANNING gate
        only, so the drain refuses what needs a person (``needs_person``). An embedder that wants to approve plans
        itself runs with ``interactive`` on (a ``maxim.api`` queue-approval surface is #1734)."""
        from maxim.agents.autonomy import Proposal

        proposal = Proposal(
            id=self.pending_proposal.request_id,
            action=action,
            reasoning=self.pending_proposal.reasoning,
            confidence=confidence,
            strategy_used=self.pending_proposal.strategy_used,
            citations=self.pending_proposal.citations,
            source=self.pending_proposal,  # its situation is credited at approval (#1083)
        )
        self.autonomy_controller.proposal_queue.submit(proposal)
        self.autonomy_controller.log_action(
            action_type="proposed",
            action=action,
            reasoning=self.pending_proposal.reasoning,
            mode=operational_mode(self.executor, self.state),
            confidence=confidence,
        )
        if no_person_to_ask():  # OFF mode: the machine approves (an embedder that approves runs with interactive on)
            self.autonomy_controller.proposal_queue.approve(
                proposal.id, approved_by="auto:non-interactive", source=InputSource.LOOP
            )
            sim.log("PIPELINE", f"Auto-approved (non-interactive PLANNING, machine): {action.get('tool_name')}")
        self.pending_proposal = None

    # ── Phase: Handle timeout retry ──────────────────────────────────────

    def handle_timeout_retry(self, cli_text: str, *, sim: Any, answer_source: InputSource) -> bool:
        """Process user input when a timeout retry is pending.

        Returns True if the input was consumed. The answers are the gating prompts' yes/no (#1185, owner
        decision C): "yes" doubles the time limit, a whole number of minutes 1-10 sets it, "no" declines; anything
        else keeps the retry pending and asks again (``_reask``), as the confirmation and plan prompts do.
        ``answer_source`` is who answered: the loop's own "no" (the capped retries ran out) already told the person
        it gave up, so it is not reported as a person's decline.
        """
        tr = self.get_timeout_retry()
        if tr is None:
            return False

        from maxim.runtime.approval import classify_yes_no

        response = cli_text.lower().strip()
        answer = classify_yes_no(cli_text)  # the same yes/no vocabulary as every gating prompt (#1185)
        self.set_timeout_retry(None)

        if answer == "no":
            from maxim.simulation.sim_logger import display_status

            if answer_source is InputSource.LOOP:
                logger.info("Loop declined timeout retry (machine retries exhausted)")
            else:
                logger.info("%s declined timeout retry", answer_source.value)
                display_status("Understood, skipping.")
            # The tick's input slots were read and popped by loop_perception.take_user_input; anything there now arrived
            # after that read and is the NEXT tick's (a person's input that waited behind a loop reply; Session R)
            self.clear_pending_user_input()
            return True

        new_timeout_s = None
        if answer == "yes":
            new_timeout_s = tr.timeout_s * 2
        else:
            try:
                minutes = int(response)
                if 1 <= minutes <= 10:
                    new_timeout_s = minutes * 60
            except ValueError:
                pass

        if new_timeout_s is not None and self.llm_worker is not None:
            if tr.original_request is not None:
                logger.info("Retrying LLM with timeout=%.0fs", new_timeout_s)
                from maxim.simulation.sim_logger import display_status

                display_status(f"Retrying with {int(new_timeout_s)}s time limit...")
                self.llm_worker.retry_with_timeout(tr.original_request, new_timeout_s)
                # The tick's input slots were read and popped by loop_perception.take_user_input; anything there now arrived
                # after that read and is the NEXT tick's (a person's input that waited behind a loop reply; Session R)
                self.clear_pending_user_input()
                return True

        if new_timeout_s is None:
            self.set_timeout_retry(tr)  # not an answer: the retry stays pending and the question is asked again
            self._reask(
                "timeout_retry_reask",
                "llm_timeout",
                "Try again with more time? Type 'yes' to double the time limit, a number of minutes (1-10), or 'no'.",
                sim=sim,
            )
            return True
        # An answer, but nothing to retry with (no worker, or no request): consumed, the retry dropped
        logger.debug("Timeout retry answered but nothing to retry: %s", response)
        self.clear_pending_user_input()
        return True

    # ── Phase: Handle plan approval ──────────────────────────────────────

    def gate_plan_approval(
        self,
        *,
        execute_and_learn: Callable[..., ExecutionOutcome],
        book_machine_refusal: Callable[..., None],
        observation: Any,
        sim: Any,
    ) -> bool:
        """The loop's plan gate (moved out of ``_run_agentic_loop_body``, rule T for #1185). A proposal that
        ``requires_approval`` with a ``plan_text`` leaves ``pending_proposal`` here, and the caller skips execution
        (True); any other proposal is the caller's to execute (False).

        - The sim's response policy answers first: its "yes" (``classify_yes_no``, the one reader) runs the plan
          through the SAME approved path as a person's (``_run_approved_plan``, owner decision B); anything else
          drops it.
        - Otherwise the plan is parked for the person in ``pending_plan_proposal`` and its real tool and params are
          shown. While a plan is already waiting, a new one is refused typed (``awaiting_human``, a machine refusal,
          no NAc) and the waiting one keeps the slot (S3), so the person's answer can only ever answer the plan
          they were shown.
        """
        proposal = self.pending_proposal
        if not (getattr(proposal, "requires_approval", False) and proposal.plan_text):
            return False
        self.pending_proposal = None  # from here the plan runs once, waits for the person, or is refused
        run = {"execute_and_learn": execute_and_learn, "book_machine_refusal": book_machine_refusal, "sim": sim}
        sim_plan_response = sim.resolve_plan_approval(proposal.plan_text)
        if sim_plan_response is not None:
            from maxim.runtime.approval import classify_yes_no

            sim.log("PIPELINE", f"Auto-resolved plan approval: {sim_plan_response}")
            if classify_yes_no(sim_plan_response) == "yes":
                logger.info("Sim mode: plan auto-approved, executing")
                self._run_approved_plan(
                    proposal,
                    approved_by="sim:response_policy",
                    approver=sim.answer_source(),  # the one place both prompts read the sim's approver
                    observation=observation,
                    **run,
                )
            else:
                logger.info("Sim mode: plan auto-rejected")
            return True
        if no_person_to_ask():  # nobody would answer a parked plan: the machine approval, PLANNING gate only
            logger.info("Non-interactive: plan machine-approved")
            self._run_approved_plan(
                proposal, approved_by="auto:non-interactive", approver=InputSource.LOOP, observation=observation, **run
            )
            return True
        if self.pending_plan_proposal is not None:
            from maxim.runtime.loop_planning import AWAITING_HUMAN, refuse_plan

            waiting = (self.pending_plan_proposal.action or {}).get("tool_name") or "unknown"
            refuse_plan(
                proposal,
                reason=AWAITING_HUMAN,
                detail=f"a plan for {waiting} is waiting for the person's answer",
                autonomy_controller=self.autonomy_controller,
                book_machine_refusal=book_machine_refusal,
                executor=self.executor,
                state=self.state,
                sim=sim,
            )
            return True
        logger.info("Proposal requires approval, showing plan to user")
        action = proposal.action or {}
        log_agentic(
            "agent_loop",
            "plan_awaiting_approval",
            {
                "tool": action.get("tool_name"),
                "params": _params_text(action)[:200],  # bounded: a write_file body must not flood the stream
                "plan_preview": proposal.plan_text[:100] if proposal.plan_text else None,
            },
        )
        self.pending_plan_proposal = proposal
        self._show_prompt(self._plan_prompt(proposal), sim=sim)
        return True

    @staticmethod
    def _plan_prompt(plan: LLMProposal) -> str:
        """What the person approves: the tool and params that will run (B2's half: the model's prose is shown as
        its description, never as what runs)."""
        action = plan.action or {}
        return (
            f"[Awaiting approval] Run {action.get('tool_name') or 'unknown'} with {_params_text(action)}\n"
            f"  The model's description: {plan.plan_text}\n"
            "Type 'yes' to approve, 'no' to cancel, or describe changes."
        )

    def _run_approved_plan(
        self,
        plan: LLMProposal,
        *,
        approved_by: str,
        approver: InputSource,
        execute_and_learn: Callable[..., ExecutionOutcome],
        book_machine_refusal: Callable[..., None],
        observation: Any,
        sim: Any,
    ) -> None:
        """The one approved-plan path, for a person's "yes", the sim policy's (owner decision B) and the
        non-interactive machine approval (``approver=InputSource.LOOP``): one ``loop_planning.execute_approved_plan``; its
        follow-up, if any, is the loop's next."""
        from maxim.runtime.loop_planning import execute_approved_plan

        outcome = execute_approved_plan(
            plan,
            autonomy_controller=self.autonomy_controller,
            execute_and_learn=execute_and_learn,
            book_machine_refusal=book_machine_refusal,
            executor=self.executor,
            observation=observation,
            state=self.state,
            sim=sim,
            approved_by=approved_by,
            approver=approver,
        )
        if outcome is not None and outcome.followup is not None:
            self.pending_action_followup = outcome.followup

    def handle_plan_approval(
        self,
        cli_text: str,
        *,
        book_refusal: Callable[..., None],
        execute_and_learn: Callable[..., ExecutionOutcome],
        book_machine_refusal: Callable[..., None],
        observation: Any,
        sim: Any,
        answer_source: InputSource,
    ) -> bool:
        """Process user input when a plan proposal is awaiting approval.

        Returns True when the input answered the plan (the caller drops it, as the next turn it is not, #1185).
        A modification is consumed too, as on the confirmation path: the plan is dropped and the text reaches the
        LLM once, as ``state.data["pending_modification"]`` (the shape ``handle_confirmation`` writes), which
        ``settle_pending_modification`` wakes the loop for and the §6 submit reads (or, in a mode with no LLM
        submit, reports and clears). A rejection is booked through ``book_refusal``
        (``LoopRun.book_refusal``, the run's recorder), so it is required.

        An approval executes the plan exactly once, here (#1185 Surface, owner decision 2026-11-21), through
        ``_run_approved_plan``. The slot holds one plan and ``gate_plan_approval`` refuses any other while it is
        occupied, so the plan answered is the plan whose tool and params were shown; the slot is emptied BEFORE it
        runs, so a second "yes" finds nothing to approve; and the plan never re-enters ``pending_proposal``, so
        the gate never parks it again. ``execute_and_learn`` / ``book_machine_refusal`` are the run's
        (``LoopRun``); ``observation`` is this tick's, ``sim`` the run's adapter.
        """
        from maxim.runtime.approval import detect_approval_intent

        if self.pending_plan_proposal is None:
            return False

        intent, modification = detect_approval_intent(cli_text)
        log_agentic(
            "agent_loop",
            "plan_approval_check",
            {"input": cli_text[:50], "intent": intent, "has_pending_plan": True},
        )

        if intent == "approve":
            logger.info("Plan approved by user, executing stored action")
            log_agentic(
                "agent_loop",
                "plan_approved",
                {
                    "tool": self.pending_plan_proposal.action.get("tool_name")
                    if self.pending_plan_proposal.action
                    else None,
                },
            )
            plan = self.pending_plan_proposal  # the one plan in the slot, the one whose tool and params were shown
            self.pending_plan_proposal = None  # spent before it runs: a repeated "yes" approves nothing
            self.clear_pending_user_input()
            self._run_approved_plan(
                plan,
                approved_by="human:plan_prompt",
                approver=answer_source,
                execute_and_learn=execute_and_learn,
                book_machine_refusal=book_machine_refusal,
                observation=observation,
                sim=sim,
            )
            return True

        elif intent == "reject":
            # A model-chosen action may name no tool (or null): "unknown", as a missing action always was,
            # never ``None`` (#1133: NAc keyed it ``tool:None``).
            rejected_tool = (self.pending_plan_proposal.action or {}).get("tool_name") or "unknown"
            logger.info("Plan rejected by user, cancelling")
            log_agentic("agent_loop", "plan_rejected", {"tool": rejected_tool})

            book_refusal(
                source=self.pending_plan_proposal,
                tool_name=rejected_tool,
                error="User rejected the proposed plan",
                reasoning=self.pending_plan_proposal.reasoning or "",
            )

            self.pending_plan_proposal = None
            self.clear_pending_user_input()
            return True

        elif intent == "modify":
            logger.info("Plan modification requested: %s", modification[:50] if modification else "")
            log_agentic(
                "agent_loop",
                "plan_modify_requested",
                {
                    "modification": modification[:100] if modification else None,
                },
            )
            plan = self.pending_plan_proposal
            self.state.data["pending_modification"] = {
                "original_action": plan.action or {},
                "original_reasoning": plan.reasoning or plan.plan_text or "",
                "original_tool_name": (plan.action or {}).get("tool_name") or "unknown",
                "user_modification": modification,
                "timestamp": time.time(),
            }
            self.pending_plan_proposal = None
            self.clear_pending_user_input()
            return True  # consumed: the LLM reads it once, as ``pending_modification`` (settle_pending_modification)

        else:  # unknown: the plan stays pending, the reply is consumed, and the question is asked again
            tool = (self.pending_plan_proposal.action or {}).get("tool_name") or "unknown"
            self._reask("plan_awaiting_approval", tool, self._plan_prompt(self.pending_plan_proposal), sim=sim)
            return True

    def _reask(self, event: str, tool_name: str, prompt: str, *, sim: Any) -> None:
        """Ask a gating question again after an answer that was neither a plain yes nor a plain no (#1185).

        The reply is consumed (the pending input slot is cleared, the caller drops it), the pending item stays
        parked, and the question goes back on the surfaces the original prompt used: the abstraction stream
        (``event``, flagged ``reask``), then ``_show_prompt``'s."""
        logger.info("Could not read a yes or no for %s, asking again", tool_name)
        log_agentic("agent_loop", event, {"tool": tool_name, "reask": True})
        self.clear_pending_user_input()
        self._show_prompt(prompt, sim=sim)

    @staticmethod
    def _show_prompt(prompt: str, *, sim: Any) -> None:
        """Show a gating question: the status line, the input panel's prompt, and the sim's channel
        (``sim.emit_prompt``: a simulated person reads the AUT's words from its action sink, never the terminal)."""
        from maxim.simulation.sim_logger import display_status

        display_status(prompt)
        sim.emit_prompt(prompt)
        try:
            from maxim.simulation.sim_logger import get_active_display

            display = get_active_display()
            if display is not None:
                display.set_prompt("Type 'yes' or 'no'\n\n> ")
                display.set_urgent(True)
        except Exception:
            log_swallowed_exception()  # a display that cannot show the prompt must not stop the loop (#1134)

    def settle_pending_modification(self, *, llm_reads_it: bool, sim: Any) -> bool:
        """A change the person asked for at a confirmation or plan prompt (``state.data["pending_modification"]``,
        #1185): True when it should wake the loop.

        Only the §6 LLM submit reads it (once: it pops it into the request, beside the original action, the shape
        ``prompt_builder.build_modification_section`` renders). Where that submit can read it (``llm_reads_it``: an
        LLM worker, not substrate-primary, not asleep; ``loop_gates`` decides) it wakes the loop until then.
        Anywhere else nothing would ever read it, so it must not keep the loop awake: the person is told plainly,
        on the prompts' surfaces (``_show_prompt``), that this mode cannot apply a change, and it is cleared."""
        mod = self.state.data.get("pending_modification")
        if not mod:
            return False
        if llm_reads_it:
            return True
        self.state.data.pop("pending_modification", None)
        tool = str(mod.get("original_tool_name") or "unknown")
        log_agentic(
            "agent_loop",
            "modification_unsupported",
            {"tool": tool, "modification": str(mod.get("user_modification") or "")[:100]},
            level="WARNING",
        )
        self._show_prompt(f"This mode can't apply a change to the proposed action ({tool}); it was not run.", sim=sim)
        return False

    # ── Phase: Store user input in state and memory ──────────────────────

    def store_user_input(self, cli_text: str, source_type: str) -> None:
        """Store CLI/voice input in state and memory for LLM processing."""
        self.state.data["pending_user_input"] = cli_text
        self.state.data["pending_user_input_time"] = time.time()
        self.state.data["pending_user_input_source"] = source_type
        logger.debug("Agent loop received %s input: %s", source_type, cli_text[:100])
        log_agentic(
            "agent_loop",
            "user_input_received",
            {"text": cli_text[:100], "source": source_type},
        )
        if hasattr(self.memory, "record_command"):
            try:
                self.memory.record_command(cli_text)
                logger.debug("Recorded %s input to memory: %s", source_type, cli_text[:50])
            except Exception:
                log_swallowed_exception(context={"source_type": source_type})  # memory is best-effort (#1134)

    # ── Phase: Speculative pre-fetch ─────────────────────────────────────

    def run_prefetch(self, cli_text: str) -> None:
        """Pre-gather file context if user mentions files."""
        try:
            self.pending_prefetch = self.prefetcher.prefetch_for_input(cli_text, cwd=os.getcwd())
            if self.pending_prefetch.discovery_plan:
                plan = self.pending_prefetch.discovery_plan
                log_agentic(
                    "agent_loop",
                    "topic_discovery",
                    {
                        "topics": plan.topic_extraction.explicit_topics[:5],
                        "dirs": plan.topic_extraction.directory_hints[:5],
                        "candidates": len(plan.candidates),
                        "full_reads": len(plan.full_content_files),
                        "summaries": len(plan.summary_files),
                        "complexity": plan.topic_extraction.complexity,
                    },
                )
                logger.info(
                    "Topic discovery: %d topics → %d candidates (%d full, %d summary)",
                    len(plan.topic_extraction.explicit_topics),
                    len(plan.candidates),
                    len(plan.full_content_files),
                    len(plan.summary_files),
                )
            if self.pending_prefetch.file_references:
                log_agentic(
                    "agent_loop",
                    "prefetch_complete",
                    {
                        "files": [r.pattern for r in self.pending_prefetch.file_references[:3]],
                        "intent": self.pending_prefetch.intent,
                        "skip_exploration": self.pending_prefetch.skip_exploration,
                        "prefetched_files": len(self.pending_prefetch.file_contents),
                    },
                )
                if self.pending_prefetch.skip_exploration:
                    if self.pending_prefetch.intent == "create" and not self.pending_prefetch.file_contents:
                        logger.info("Pre-fetch: New file creation detected, skipping exploration")
                    else:
                        logger.info(
                            "Pre-fetch: Systematic discovery complete (%d files), LLM can write directly",
                            len(self.pending_prefetch.file_contents),
                        )
                elif self.pending_prefetch.file_contents:
                    logger.info("Pre-fetch: Gathered %d files for context", len(self.pending_prefetch.file_contents))
        except Exception:
            log_swallowed_exception()  # pre-fetch is non-critical, but its failure is reported (#1134)
            self.pending_prefetch = None

    # ── Phase: Configure Default Network for mode ────────────────────────

    def configure_dn_for_mode(self, mode_name: str) -> None:
        """Delegate to DefaultNetworkController."""
        self.dn_ctrl.configure_for_mode(mode_name)


# ── The D13 planning-liveness handlers (1.3.2 decomposition, slice 2) ──
# Module functions beside the LoopController counters they drive (record_planning_failure /
# record_planning_transport_failure), moved from agent_loop.py with their bodies unchanged except that
# they log through ``_loop_logger`` (this module's own ``logger`` is ``maxim.runtime.loop_controller``).
# Callers: loop_gates.pre_tick_gate (the idle-gate backstop: both handlers) and loop_intake (section 2's
# failure sites and _drop_stale_proposal, moved from agent_loop in 1.3.6 slice 6: _handle_planning_failure).
#
# Kept for log continuity; a later slice may adopt the module's own logger. The handlers' records keep
# the ``maxim.runtime.agent_loop`` name they have always had (the SAME object as agent_loop.logger).
_loop_logger = logging.getLogger("maxim.runtime.agent_loop")


def note_system_outcome(recent_outcomes: list[Any], max_recent: int, tool: str, message: str, error: str) -> None:
    """Tell the agent, through the recent outcomes its next prompt shows, why its call did not run: the
    consecutive-same-tool cap (moved out of the loop's body) and a run-ending answer set aside (#1451). A failed outcome
    of ``tool`` carrying a ``SYSTEM:`` message; the list is trimmed to its newest ``max_recent``. Never
    ``pending_cli_input``: that single-slot user-input channel would overwrite real input."""
    recent_outcomes.append(
        {"tool": tool, "tool_name": tool, "success": False, "result": message, "error": error, "timestamp": time.time()}
    )
    if len(recent_outcomes) > max_recent:
        recent_outcomes.pop(0)


# Memory and executor types already warned about lacking what the re-plan reads (one WARNING per type, #1451).
_REPLAN_OFF_WARNED: set[str] = set()


def _warn_replan_off_once(obj: Any, lacks: str) -> None:
    kind = type(obj).__qualname__
    if kind not in _REPLAN_OFF_WARNED:
        _REPLAN_OFF_WARNED.add(kind)
        _loop_logger.warning("%s has no %s: held inputs cannot be re-planned (#1451)", kind, lacks)


def is_unread_input(line: Any, processed_cli_inputs: Any) -> bool:
    """THE test of an input the loop has heard but not yet submitted, shared by the follow-up's fold of held inputs
    (``agent_loop._take_deferred_inputs``, #1048) and the re-plan (#1451). A synthetic follow-up line is not an input.
    The triggering-input scan's own rules (the ``SKIP_LLM_COMMANDS`` list, a voice input without its wake word) are
    not applied: they mark such inputs processed as the scan meets them, and a planning-liveness loop's inputs are
    the narrator's CLI lines, which those rules never skip (its operator text arrives prefixed ``USER GUIDANCE:``)."""
    from maxim.runtime.loop_perception import is_tool_followup

    return bool(line) and line not in processed_cli_inputs and not is_tool_followup(str(line))


def _unread_inputs(memory: Any, processed_cli_inputs: Any) -> list[str]:
    """The inputs ``memory`` heard that the loop has not yet submitted (``is_unread_input``), oldest first. A memory
    that cannot list what it heard (``recent_cli_inputs``: ``InMemoryMemory``, the narrator's, and ``MemoryAgent``)
    reads as none, with one WARNING per type: the re-plan is then off for that loop, said aloud."""
    recent = getattr(memory, "recent_cli_inputs", None)
    if not callable(recent):
        _warn_replan_off_once(memory, "recent_cli_inputs()")
        return []
    return [c for c in recent() if is_unread_input(c, processed_cli_inputs)]


def set_aside_run_end(ctrl: Any, memory: Any, executor: Any, sim: Any, liveness_on: bool) -> None:
    """Re-plan once (#1451; owner decisions 2026-10-30 and 2026-11-01). Called before the pending proposal runs.
    Behaviour tier: invariant (a fixed rule of the loop, not an innate prior or learned).

    When the proposal's tool ends the run (``Tool.ends_run``) and inputs arrived while it was being planned, the
    agent decided to end the run without reading them. The FIRST such answer is set aside: the proposal is dropped,
    a recent outcome tells the agent why, and the held inputs, still unread, are all planned on the next submit
    (``take_replan_inputs``). The re-plan's answer then runs whatever it is: a run-ending answer with inputs held
    again runs, and those inputs get one WARNING naming them ("unread at finish"). Every run-ending answer let through
    re-arms the rule (a finish the tool refuses, below the run's turn floor, leaves the run going), and so does any
    answer that does not end the run. Only a loop with planning liveness (the sim narrator) holds inputs, so only it
    is affected. Scope: the primary action only. A run-ending tool among ``parallel_actions``, or one approved
    through the SUPERVISED confirmation or the PLANNING queue, is not set aside (the narrator runs AUTONOMOUS and
    plans one action; deliberation cycles 2+ submit without the held inputs, as for a follow-up).

    ``ctrl.run_end_set_aside`` does two jobs: it marks the answer already set aside once, and it tells
    ``take_replan_inputs`` the next submit is the re-plan. A re-plan that comes back without an action (a planning
    failure, then its retry) leaves it set, so the retry carries the still-unread inputs again and the next
    run-ending answer runs ("once")."""
    proposal = ctrl.pending_proposal
    action = getattr(proposal, "action", None)
    if not liveness_on or not isinstance(action, dict):
        return
    tool_name = str(action.get("tool_name", "") or "")
    tool = tool_to_run(executor, tool_name)
    if tool is None and not callable(getattr(getattr(executor, "registry", None), "find", None)):  # no lookup
        _warn_replan_off_once(executor, "registry.find()")
    if getattr(tool, "ends_run", False) is not True:  # ``is True``: a mock's attribute is not a declaration
        ctrl.run_end_set_aside = False
        return
    unread = _unread_inputs(memory, ctrl.processed_cli_inputs)
    if ctrl.run_end_set_aside or not unread:
        if unread:
            _loop_logger.warning(
                "%s runs with %d input(s) it was not shown, held behind its re-plan; unread at finish: %s (#1451)",
                tool_name,
                len(unread),
                [c[:80] for c in unread],
            )
        ctrl.run_end_set_aside = False  # let through: the next run-ending answer may be set aside again
        return
    ctrl.run_end_set_aside = True
    ctrl.pending_proposal = None
    # The prompt shows an outcome's first 50 characters (``PromptBuilder._add_perception_sections``): the
    # instruction leads.
    note = (
        f"SYSTEM: {len(unread)} new input(s) to read before finishing. Your {tool_name} was set aside because they "
        "arrived while it was being planned: act on them, or finish again."
    )
    note_system_outcome(ctrl.recent_outcomes, ctrl.max_recent_outcomes, tool_name, note, "set_aside_for_held_inputs")
    _loop_logger.info("%s set aside: %d input(s) arrived while it was being planned (#1451)", tool_name, len(unread))
    sim.log("EXEC", f"SET ASIDE: {tool_name} ({len(unread)} input(s) held behind it; re-planning once, #1451)")


def take_replan_inputs(ctrl: Any, memory: Any, liveness_on: bool) -> list[str]:
    """The re-plan's submit carries EVERY input held behind the set-aside answer (#1451): the scan made the oldest
    unread one this submit's request and marked it processed; the rest ride as the request's held inputs, the way a
    follow-up carries them (#1048), and are marked processed (a refused submit releases them:
    ``loop_external._release_unsent_deferred_inputs``). Empty unless a run-ending answer was just set aside."""
    if not liveness_on or not ctrl.run_end_set_aside:
        return []
    taken = _unread_inputs(memory, ctrl.processed_cli_inputs)
    ctrl.processed_cli_inputs.extend(taken)
    return taken


def _report_planning_exhaustion(ctrl: Any, sim: Any, *, reason: str, kind: str) -> bool:
    """Emit one terminal planning/transport failure and tell the loop to stop."""
    status = ctrl.planning_exhausted_status
    msg = (
        f"planning {kind} exhausted: planning_failures={ctrl.planning_failure_streak}, "
        f"transport_failures={ctrl.planning_transport_failure_streak} "
        f"(last: {reason}, status: {status}) — aborting sim (bugs ledger D13)"
    )
    _loop_logger.error(msg)
    sim.log("EXEC", f"🛑 {msg}")
    log_agentic(
        "agent_loop",
        "planning_liveness_exhausted",
        {
            "planning_streak": ctrl.planning_failure_streak,
            "transport_streak": ctrl.planning_transport_failure_streak,
            "reason": reason,
            "status": status,
        },
        level="ERROR",
    )
    return True


def _handle_planning_failure(
    ctrl: Any,
    llm_worker: Any,
    sim: Any,
    *,
    reason: str,
    original_request: Any | None,
    failed_tool: str | None = None,
    exhausted_status: str = "llm_wedged",
) -> bool:
    """Planning-liveness handler (bugs ledger D13): a planning submit ended
    without an executable proposal — parse failure, invalid response, a
    dropped proposal, or an await window that expired with nothing back.

    LOUDLY reschedules the turn with bounded retries (byte-identical requeue
    of the original request — no fabricated percepts) or, when the budget is
    spent, tells the caller to abort. Never silent, never a fall-through to
    idle.

    Returns True when planning liveness is exhausted (caller breaks the loop
    and raises ``PlanningLivenessExhausted`` after normal teardown).
    """
    verdict = ctrl.record_planning_failure(reason=reason, exhausted_status=exhausted_status)
    if verdict == "already_exhausted":
        return True
    if verdict == "exhausted":
        return _report_planning_exhaustion(ctrl, sim, reason=reason, kind="retry budget")

    # verdict == "retry"
    # Every retry says why the last answer failed when the model erred (narrator reliability, 2026-10-02).
    if original_request is not None:
        requeued = bool(llm_worker.requeue_request(original_request, failed_tool=failed_tool, reason=reason))
    else:
        requeued = bool(llm_worker.requeue_last_request(failed_tool=failed_tool, reason=reason))
    attempt = ctrl.planning_failure_streak
    limit = ctrl.planning_retry_limit
    note = (
        f"rescheduled (attempt {attempt}/{limit})"
        if requeued
        else "REQUEUE REJECTED — bounded worker-transport recovery will retry"
    )
    _loop_logger.warning("planning turn failed (%s) — %s", reason, note)
    sim.log("EXEC", f"⚠ planning turn failed ({reason}) — {note}")
    log_agentic(
        "agent_loop",
        "planning_retry",
        {"reason": reason, "attempt": attempt, "limit": limit, "requeued": requeued},
        level="WARNING",
    )
    # Retained for diagnostics and for non-liveness callers' legacy await
    # window. The liveness-enabled orchestrator uses exact WorkerPool state.
    ctrl.last_llm_submit_time = time.time()
    return False


def _handle_planning_transport_failure(
    ctrl: Any,
    llm_worker: Any,
    sim: Any,
    *,
    reason: str,
) -> bool:
    """Bound retries for a worker attempt that failed before publication."""
    verdict = ctrl.record_planning_transport_failure(reason=reason)
    if verdict == "already_exhausted":
        return True
    if verdict == "exhausted":
        return _report_planning_exhaustion(ctrl, sim, reason=reason, kind="transport budget")

    requeued = bool(llm_worker.requeue_last_request())
    attempt = ctrl.planning_transport_failure_streak
    limit = ctrl.planning_transport_retry_limit
    note = (
        f"worker retry accepted (attempt {attempt}/{limit})"
        if requeued
        else f"worker retry rejected (attempt {attempt}/{limit})"
    )
    _loop_logger.warning("planning transport failed (%s) — %s", reason, note)
    sim.log("EXEC", f"⚠ planning transport failed ({reason}) — {note}")
    log_agentic(
        "agent_loop",
        "planning_transport_retry",
        {"reason": reason, "attempt": attempt, "limit": limit, "requeued": requeued},
        level="WARNING",
    )
    ctrl.last_llm_submit_time = time.time()
    return False


def _params_text(action: dict[str, Any]) -> str:
    """An action's params as one line of JSON (repr when not serialisable), for the plan prompt and its event."""
    import json

    try:
        return json.dumps(action.get("params", {}), sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(action.get("params", {}))
