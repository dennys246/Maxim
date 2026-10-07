"""The agent loop's pre-tick gate, sections 0-0.6 (1.3.2 decomposition, slice 2).

``pre_tick_gate`` is the head of every pass of ``agent_loop.run_agentic_loop``, moved verbatim: the stop
checks (``stop_event``, a ``"shutdown"`` mode), the Default Network's mode (the operator's grant wins,
#829), the pause sleep, the live tick (body drift + experience clock) and the display auto-revert, the
percept-source exhaustion check, and the idle gate with its D13 planning-liveness backstop. It returns a
``GateOutcome``; the loop maps each value to the control flow the inline block had (the enum's docstring).
The gate's sleeps (``time.sleep(idle_sleep_s)`` on a paused or idle pass) happen HERE, before it returns,
exactly where the inline block slept before its ``continue``.

**Arguments.** Explicit keywords, never the loop's ``LoopRun`` (rule (d) of the roadmap's import-direction
paragraph). Each is the loop's own local, passed as-is, so the move stays pure: ``autonomy_controller`` the
run's controller, ``experience_driver`` the run's ``ExperienceClockDriver`` (``LoopRun.xclock``) and
``planning_liveness_on`` the run's one D13 gate (``LoopRun.planning_liveness_on``). Two of the loop's locals
are read through ``ctrl`` instead: ``pending_next_actions`` and ``llm_submit_interval`` were aliases of
``ctrl.pending_next_actions`` and ``ctrl.llm_submit_interval``, and nothing in ``src/`` reassigns either field
(the list is only mutated in place), so the gate reads the same object and the same value. In the body,
``_loop_xclock`` became ``experience_driver``, ``_planning_liveness_on`` became ``planning_liveness_on`` and
those two aliases became their ``ctrl.`` fields; nothing else changed but its exits.

**What the gate does NOT own.** The ``_planning_liveness_exhausted`` flag stays a local of the loop: on
``EXHAUSTED`` the loop sets it and breaks, and raises ``PlanningLivenessExhausted`` after its teardown, as
before. The loop-alive heartbeat and the ``loop_iteration`` event before section 0 stay in the loop (they
never exit a pass).

**Helpers.** The four that only this block called moved here with it, bodies verbatim (rule (a)):
``tick_embodiment_drift``, ``_loop_live_tick``, ``_maybe_auto_revert_display``, ``_loop_is_idle``. The ones the
loop's body also calls live in leaves both import: ``loop_state`` (``_effective_mode`` and the wake predicates
``_substrate_tick_due`` / ``_planning_attempt_is_active``) and ``loop_controller`` (the D13 handlers, beside the
counters they drive). No patch seam is read through ``agent_loop``: no test patched any of these names there
except ``test_experience_clock.py``'s unit test of ``_loop_live_tick``, which now patches
``tick_embodiment_drift`` here, beside the function it tests. The gate LOGS on the ``maxim.runtime.agent_loop``
logger (``logger`` below is that same object), so its records keep the agent loop's logger name.

Characterization: ``tests/unit/test_loop_gates_characterization.py`` (written before the move, through the
public ``run_agentic_loop``, and kept green unchanged by it).
"""

from __future__ import annotations

import enum
import logging
import time
from typing import TYPE_CHECKING, Any

from maxim.agents.llm_worker import LLMAttemptState
from maxim.runtime.loop_controller import _handle_planning_failure, _handle_planning_transport_failure
from maxim.runtime.loop_state import _effective_mode, _planning_attempt_is_active, _substrate_tick_due
from maxim.utils.structured_logging import log_agentic

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController
    from maxim.runtime.loop_controller import LoopController

# The SAME logger object as ``agent_loop.logger`` (logging returns one logger per name), so records keep
# the ``maxim.runtime.agent_loop`` name.
logger = logging.getLogger("maxim.runtime.agent_loop")


class GateOutcome(enum.Enum):
    """What the loop does with this pass. Each value is one exit of the former inline block:

    - ``RUN``: fall through to section 1 (perception).
    - ``IDLE``: ``continue`` -- a paused pass, or an idle one; the gate already slept ``idle_sleep_s``.
    - ``BREAK``: ``break`` -- the stop event is set, the mode is ``"shutdown"``, or the percept source is
      exhausted. Normal teardown follows.
    - ``EXHAUSTED``: ``break`` after setting ``_planning_liveness_exhausted`` -- the D13 retry budget is
      spent; the loop raises ``PlanningLivenessExhausted`` after its teardown.
    """

    RUN = "run"
    IDLE = "idle"
    BREAK = "break"
    EXHAUSTED = "exhausted"


def pre_tick_gate(
    *,
    step_num: int,
    stop_event: Any | None,
    state: Any,
    executor: Any,
    ctrl: LoopController,
    autonomy_controller: AutonomyController,
    sim: Any,
    percept_source: Any | None,
    llm_worker: Any | None,
    planning_liveness_on: bool,
    aut_mode: str,
    experience_driver: Any,
    idle_sleep_s: float,
) -> GateOutcome:
    """Sections 0-0.6 of one pass of ``run_agentic_loop`` (module docstring)."""

    # ─────────────────────────────────────────────────────────────────
    # 0. CHECK STOP CONDITIONS
    # ─────────────────────────────────────────────────────────────────
    try:
        if stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set():
            log_agentic("agent_loop", "shutdown", {"reason": "stop_event"})
            return GateOutcome.BREAK
    except (AttributeError, RuntimeError):
        pass

    # Check for shutdown mode - break immediately to stop LLM worker promptly
    current_mode = state.data.get("mode", "")
    if current_mode == "shutdown":
        log_agentic("agent_loop", "shutdown", {"reason": "shutdown_mode"})
        return GateOutcome.BREAK

    # Configure Default Network for current mode (the operator's grant wins, #829)
    if _dn_mode := _effective_mode(executor, state, current_mode):
        ctrl.configure_dn_for_mode(_dn_mode)

    # Check if autonomy is paused
    if autonomy_controller.is_paused:
        time.sleep(idle_sleep_s)
        return GateOutcome.IDLE

    # ─────────────────────────────────────────────────────────────────
    # 0.45 EMBODIMENT DRIFT TICK (llm-primary)
    # ─────────────────────────────────────────────────────────────────
    # Advance the body's wall-clock drive drift every live iteration so a
    # Reachy body does not freeze through pure-thinking turns / idle gates
    # / LLM latency. Placed BEFORE the 0.6 idle gate (which ``continue``s
    # on no stimulus) so a *sitting* robot still gets cold/hungry, and
    # AFTER the pause check so an operator-paused agent stays frozen.
    # No-op on substrate-primary (it ticks itself) and when unembodied.
    _loop_live_tick(executor, aut_mode, experience_driver)

    # Expire a temporary agent display escalation back to the user's
    # floor (DisplayModeTool's documented auto-revert). Also the
    # production producer of the EVENT seam's display/revert event.
    _maybe_auto_revert_display()

    # 0.5 CHECK PERCEPT SOURCE EXHAUSTION (simulation mode)
    if sim.check_exhaustion(ctrl.pending_proposal):
        return GateOutcome.BREAK

    # ─────────────────────────────────────────────────────────────────
    # 0.6 IDLE GATE — skip full cycle when there's nothing to react to
    # ─────────────────────────────────────────────────────────────────
    # The agent loop spins at target_hz for responsiveness, but should
    # NOT burn LLM cycles when idle.  We check for any pending stimulus
    # BEFORE running perception/pipeline agents.  If nothing is pending,
    # sleep briefly and loop back.  This keeps the loop responsive to
    # new input (sub-second latency) without wasting GPU on empty cycles.
    #
    # "Stimulus" means:
    #   - User input (CLI or voice) waiting in state.data
    #   - Simulation percept available from percept_source
    #   - Pending proposal from LLM (needs execution)
    #   - Pending action followup (tool result needs LLM processing)
    #   - Pending next_actions chain (multi-step plan in progress)
    #   - First iteration (startup — run initial cycle once)
    #   - Carried live percept / awaited LLM job / substrate tick due (see _loop_is_idle)
    _has_pending_input = bool(state.data.get("pending_cli_input") or state.data.get("pending_voice_input"))
    _has_pending_work = bool(
        ctrl.pending_proposal or ctrl.pending_action_followup or ctrl.pending_next_actions or ctrl.pending_plan_proposal
    )
    _has_sim_percept = (
        sim.is_sim_mode and percept_source is not None and getattr(percept_source, "has_pending", lambda: True)()
    )
    # A live producer's carried percept (the DoA feed → NullSimulation-
    # Adapter mailbox, Stage 3 of live_audio_orient_wiring.md) must WAKE
    # the loop — 2026-08-01 live-smoke fix. This gate's percept check
    # was gated on is_sim_mode: the same proxy the Stage-3 §1.16 re-gate
    # removed, one layer up. Without this term a live audio percept sat
    # undelivered forever on an idle robot (the loop slept BEFORE
    # next_observation surfaced it), so audio escalation only ever fired
    # when typed input happened to wake the loop in the same window.
    _has_carried_percept = bool(getattr(sim, "has_carried_percept", lambda: False)())
    _is_first_step = step_num == 0
    # If we submitted to the LLM, keep polling until the exact WorkerPool
    # job reaches a terminal state. ``COMPLETED`` remains active until
    # get_latest_proposal() consumes the result, which closes the old
    # provider-return/result-publication race without a timing guess.
    #
    # Non-liveness callers retain their legacy 120s window. The exact
    # state machine is deliberately scoped to the orchestrator opt-in;
    # it must not alter unrelated agent-loop lifecycle policy.
    _planning_attempt_state = LLMAttemptState.NONE
    if planning_liveness_on and llm_worker is not None:  # implied by planning_liveness_on
        try:
            _planning_attempt_state = llm_worker.latest_attempt_state()
        except Exception as e:
            logger.warning("planning worker state unavailable: %s", e)
            _planning_attempt_state = LLMAttemptState.MISSING
        if _planning_attempt_state is LLMAttemptState.COMPLETED:
            ctrl.reset_planning_transport_failures()

    _submitted_recently = bool(
        not planning_liveness_on
        and llm_worker is not None
        and ctrl.pending_proposal is None
        and (time.time() - ctrl.last_llm_submit_time) < 120.0
    )
    _awaiting_llm = (
        _planning_attempt_is_active(_planning_attempt_state) if planning_liveness_on else _submitted_recently
    )

    _wake = _has_pending_input or _has_pending_work or _has_sim_percept or _has_carried_percept
    if _loop_is_idle(
        _wake, _is_first_step, _awaiting_llm, _substrate_tick_due(aut_mode, ctrl, ctrl.llm_submit_interval)
    ):
        # D13 planning-liveness backstop: the loop is about to idle, but
        # the exact job for the last planning submit is terminal and no
        # executable proposal was installed. Worker execution failures
        # use a separate bounded transport budget; completed-but-empty
        # results are planning failures. Neither can silently fall
        # through to idle or retry forever.
        if (
            planning_liveness_on
            and not ctrl.planning_exhausted
            and ctrl.last_llm_submit_time > 0
            and ctrl.last_proposal_time < ctrl.last_llm_submit_time
        ):
            if _planning_attempt_state in {
                LLMAttemptState.FAILED,
                LLMAttemptState.CANCELLED,
                LLMAttemptState.MISSING,
            }:
                if _handle_planning_transport_failure(
                    ctrl,
                    llm_worker,
                    sim,
                    reason=f"worker_job_{_planning_attempt_state.value}",
                ):
                    return GateOutcome.EXHAUSTED
            elif _planning_attempt_state in {
                LLMAttemptState.NONE,
                LLMAttemptState.CONSUMED,
            } and _handle_planning_failure(
                ctrl,
                llm_worker,
                sim,
                reason="planning_job_completed_without_proposal",
                original_request=None,
                exhausted_status="planning_failed",
            ):
                return GateOutcome.EXHAUSTED
        time.sleep(idle_sleep_s)
        return GateOutcome.IDLE
    return GateOutcome.RUN


# ── helpers moved from agent_loop.py (rule (a): only this block calls them) ──


def tick_embodiment_drift(executor: Any, aut_mode: str) -> None:
    """Advance the body's wall-clock drive drift on the llm-primary path.

    On ``substrate-primary``, :func:`propose_via_substrate` already ticks the
    body every proposal. On ``llm-primary`` the body tick is otherwise
    *event-driven* — it only fires when a tool executes (``tool_bridge`` /
    sim tools calling ``evaluate_failures()``). So a body sitting through
    pure-thinking turns, idle gates, or LLM latency would never drift: its
    drives freeze (the Track A "frozen Reachy body" finding). Calling
    ``evaluate_failures()`` once per live loop iteration advances wall-clock
    drift so the llm-primary body has the same clock as substrate-primary.

    Idempotent w.r.t. elapsed time: ``evaluate_failures`` applies
    ``dt = now - _last_poll`` via ``tick_vital_drift`` lazily, so calling it
    here AND on a later tool execution in the same iteration cannot
    double-drift (the second call sees ~0 elapsed dt). No-op on
    substrate-primary (that path ticks itself — calling here too would double
    the tick) and when no embodiment is wired. This calls the public
    ``evaluate_failures()`` tick, not ``tick_vital_drift`` directly, per the
    CLAUDE.md embodiment-tick invariant (single ``tick_vital_drift`` call site
    in body.py).

    CADENCE CAVEAT (three-lens review, 2026-07-17): ``evaluate_failures`` does
    not only drift — it re-publishes drive-pain for any *standing* breach on
    every call, so this per-iteration cadence makes drive-pain state-based
    rather than onset/transition-based. This is exactly the change
    ``docs/plans/deferred/transition_based_drive_pain.md`` names as its revival
    trigger ("before any change to evaluate_failures cadence"). It is dampened
    to *valence noise, not false causal links* by three existing guards — the
    drift tick DISCARDS the returned FailureEvents (pain flows only via
    PainBus), the PainBus ``(entity, failure_mode)`` refractory caps the rate
    to ~2 Hz, and the ``_context_similarity`` denominator mismatch keeps these
    events from linking to tool actions — so it is a should-fix, not a blocker.
    Two consequences to keep in mind: (1) it is latent for the shipped reachy
    body (its only drive, azimuth, is world-set with ``drift_rate: 0`` and
    sits centered until DoA is fed in Track 2); (2) it DOES change the drive-
    pain cadence for embodied llm-primary sims (Exp 44, ``--embodiment``), so
    prior Exp 44 numbers need re-validation before being relied on.
    """
    if aut_mode == "substrate-primary":
        return
    embodiment = getattr(executor, "embodiment", None)
    if embodiment is None:
        return
    try:
        embodiment.evaluate_failures()
    except Exception:
        logger.debug("llm-primary embodiment tick: evaluate_failures raised", exc_info=True)


def _maybe_auto_revert_display() -> None:
    """Expire a temporary agent display escalation back to the user's floor.

    ``DisplayModeTool`` documents escalations as auto-reverting; before this
    tick nothing ever reverted one (``revert_display_to_floor`` had zero
    production callers), so an escalation stuck for the rest of the session
    and the EVENT seam's ``display``/revert wire event had no producer.
    Cheap on the common path: one float compare, no escalation → immediate
    return.
    """
    from maxim.simulation.sim_logger import maybe_auto_revert_display

    try:
        maybe_auto_revert_display()
    except Exception:
        # Mirrors tick_embodiment_drift's containment: a display-tier bookkeeping
        # failure must never take down the main loop.
        logger.debug("display auto-revert tick raised", exc_info=True)


def _loop_live_tick(executor: Any, aut_mode: str, experience_driver: Any) -> None:
    """Per LIVE pass (after the pause check, before the idle gate): the world's time advances.

    The body's drive drift and the agent's experience clock (memory-strength Phase 2 decision 1)
    both run here, so an idle agent still lives through the world's time and a paused one does not.
    """
    tick_embodiment_drift(executor, aut_mode)
    experience_driver.on_live_pass()


def _loop_is_idle(*wake_sources: object) -> bool:
    """True when NO wake source holds — the loop sleeps ``idle_sleep_s`` and continues.

    Extracted with the substrate wake term (function-length ratchet: grow the god
    function by extracting, never inline). Order of the sources is documented at the
    call site: pending input, pending work, sim percept, carried percept, first step,
    awaited LLM, substrate tick due. Truthiness semantics are the old ``not (a or b …)``:
    a third-party ``has_pending`` may return a count, so the sources are ``object``.
    """
    return not any(wake_sources)
