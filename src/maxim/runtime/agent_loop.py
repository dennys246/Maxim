from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, Literal

from maxim.evaluation.base import Evaluator
from maxim.utils.logging import log_swallowed_exception, warn
from maxim.utils.structured_logging import log_agentic

# Extracted to tool_dispatch.py
from maxim.runtime.tool_dispatch import (
    safe_agent_name as _safe_agent_name,
    record_outcome as _record_outcome,
    execute_parallel_actions as _execute_parallel,
)

# Extracted to bio_integration.py
from maxim.runtime.bio_integration import (
    capture_episodic_memory as _capture_episodic,
    record_plan_outcome as _record_plan_outcome,
    start_bio_session as _start_bio_session,
    end_bio_session as _end_bio_session,
)
import maxim.runtime.bio_integration as _bio_integration

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController
    from maxim.agents.llm_worker import LLMWorker

# Import LLMProposal for runtime use (multi-step action creation)
from maxim.agents.llm_worker import LLMProposal
from maxim.agents.bus import StreamEvent
from maxim.embodiment.sensory_streams import AUDIO_TAG, INTEROCEPTION_TAG, ModalityChannel

# Import Hippocampus and MemoryHub for episodic memory (optional)
try:
    from maxim.memory.hippocampus import Hippocampus
except ImportError:
    Hippocampus = None  # type: ignore

try:
    from maxim.integration.memory_hub import MemoryHub
except ImportError:
    MemoryHub = None  # type: ignore

logger = logging.getLogger(__name__)


# Wire 3 (release_0_9_1.md Stage 1): regex matching the felt-sensation
# annotations Embodiment.integrity_to_felt_phrase produces. The agent_loop
# hook uses this to strip a stale annotation before re-applying the
# current-tick one — guards against multi-tick integrity drift accumulating
# multiple suffixes (e.g., integrity 0.55 → ``(feels strained)``, then 0.40
# → ``(feels weakened, prone to failing)``; without the strip, both would
# coexist in the description). The phrases are pinned in tests so a future
# additional band must update both this regex AND
# ``Embodiment.integrity_to_felt_phrase`` together.
_WIRE3_PHRASE_RE = re.compile(r" \((?:feels strained|feels weakened, prone to failing)\)$")

# Wire 1 (release_0_9_1.md Stage 4): regex matching the experience-voice
# annotations the variance-annotation hook produces. Same shape rationale as
# the Wire 3 regex above — strip stale annotation before re-applying the
# current-tick one so the description doesn't accumulate suffixes across
# observations.
#
# **Register choice (bio-fidelity fold from pre-merge review):** Wire 3's
# phrases ("feels strained" / "feels weakened, prone to failing") carry a
# SOMATIC voice — they describe proprioceptive body-state. Wire 1's signal
# is METACOGNITIVE — variance over outcome reliability is "what experience
# has taught me about this action", not "what my body senses right now".
# Reusing the "feels X" stem for both collapsed two distinct bio-system
# registers (somatic vs experience-acquired) into one indistinguishable
# surface, so the LLM could not separate "I will fail because the body
# is broken" from "I will fail because the outcome is stochastic". The
# experience-voice phrasing "(unpredictable from prior experience)" /
# "(reliable from prior experience)" aligns with Wire-A's
# "[... from prior experience]" prompt-section register (the only other
# substrate surface that exposes learned variability to the LLM), keeping
# the experience-acquired signals coherent across wires while Wire 3 owns
# the somatic surface alone.
#
# Two phrases match the two bands (high variance / reliable); the middle
# band emits no annotation. Wire 1 annotation appears AFTER any Wire 3
# annotation in the description, so the regex is anchored at end-of-
# string and Wire 3's strip runs first (the two suffixes can co-occur on
# one tool — physical-damage + outcome-variance signals are orthogonal).
_WIRE1_PHRASE_RE = re.compile(r" \((?:unpredictable|reliable) from prior experience\)$")

# Wire 1 thresholds. Bernoulli variance on binary {0, 1} reward maxes at 0.25
# (p = 0.5). The bands are pre-registered in release_0_9_1.md Stage 4:
#   variance >= 0.15 → "(unpredictable from prior experience)"  — ~30/70 or worse
#   variance <  0.05 → "(reliable from prior experience)"       — ~94/6 or better
#   otherwise        → no annotation (middle band)
# Min observation count guards against single-sample noise — Welford variance
# stabilises around n ~= 5 for binary signals.
_WIRE1_HIGH_VARIANCE_THRESHOLD = 0.15
_WIRE1_LOW_VARIANCE_THRESHOLD = 0.05
_WIRE1_MIN_OBSERVATIONS = 5

# Wire 1 ablation gate. Reuses Wire-A's canonical
# ``annotation_disabled_via_env`` parser so the truthy-value set is a single
# source of truth across 0.9.1's two annotation gates. Default OFF
# (annotation ON in 0.9.1 by design). Roy-3 may set this for variance-
# annotation-off arms; the conftest scrub clears it between tests.
_WIRE1_DISABLE_ENV = "MAXIM_DISABLE_VARIANCE_ANNOTATION"
_WIRE1_HIGH_PHRASE = "unpredictable from prior experience"
_WIRE1_LOW_PHRASE = "reliable from prior experience"


from maxim.runtime.loop_state import (
    _persist_state_json,
    _get_failure_strategy,
    _get_plan_depth,
    _build_replan_context,
)


def _reset_deliberation(executor: Any) -> None:
    """Reset ThinkTool deliberation state when a non-think action fires (L2).

    Single call site for both dispatch paths — prevents drift.
    """
    try:
        _registry = getattr(executor, "registry", None)
        if _registry is not None:
            _think_tool = _registry.get("think")
            if hasattr(_think_tool, "reset_deliberation"):
                _think_tool.reset_deliberation()
    except (KeyError, Exception):
        pass  # think tool not registered or not a ThinkTool


def _idle_sleep(idle_sleep_s: float) -> None:
    """Sleep for ``idle_sleep_s`` seconds, logging any failure at warning.

    Extracted because the loop body has 4+ identical sleep+swallow blocks.
    A ``time.sleep`` failure is essentially impossible in normal operation
    (it would mean the interpreter is in a degraded state), but if it does
    happen we want to know — log at warning so it surfaces.
    """
    try:
        time.sleep(float(idle_sleep_s))
    except Exception as e:
        logger.warning("idle_sleep failed (loop will continue without backoff): %s", e, exc_info=True)


def run_agent_loop(
    agent: Any,
    environment: Any,
    state: Any,
    memory: Any,
    decision_engine: Any,
    executor: Any,
    *,
    evaluators: list[Evaluator] | None = None,
    max_steps: int = 100,
    run_id: str | None = None,
    stop_event: Any | None = None,
    on_step: Any | None = None,
    on_event: Any | None = None,  # Fine-grained streaming callback
    break_on_no_intent: bool = False,
    idle_sleep_s: float = 0.25,
    persist_every_n_steps: int = 10,
) -> None:
    """
    Canonical agentic loop:
    observe → agent proposes intent → planner proposes plans → policy constrains → decision engine selects action → executor runs tool

    Args:
        persist_every_n_steps: How often to persist state to disk. Default is every 10 steps.
                              Set to 1 for per-step persistence, 0 to only save at end.
    """
    if evaluators is None:
        evaluators = []

    if not run_id:
        run_id = time.strftime("%Y-%m-%d_%H%M%S")
    agent_name = _safe_agent_name(agent)
    state_path = os.path.join("data", "agents", agent_name, "runtime", f"state_{run_id}.json")
    _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})

    max_steps_i = int(max_steps or 0)
    step_iter = itertools.count() if max_steps_i <= 0 else range(max_steps_i)
    for _ in step_iter:
        try:
            if stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set():
                break
        except Exception as e:
            # A stop_event check failure during the loop is serious — it means
            # we can't honor shutdown requests cleanly. Surface at warning so
            # users can spot wedged loops in their logs.
            logger.warning("stop_event.is_set() check failed: %s", e, exc_info=True)

        # ── Consume pending replan candidate from previous failure ──
        # If the previous iteration triggered ADaPT decomposition, the replan
        # candidate bypasses propose_intent/decide and executes directly.
        _replan = None
        if hasattr(state, "data") and isinstance(state.data, dict):
            _replan = state.data.pop("_replan_candidate", None)

        if _replan is not None and hasattr(_replan, "actions") and _replan.actions:
            decision = {"action": _replan.actions[0], "plan": _replan, "score": 1.0}
            goal = state.data.get("_replan_goal", "replan")
            state.data.pop("_replan_goal", None)
            intent = {"goal": goal, "source": "replan"}
        else:
            # ── Normal path: observe → propose intent → decide ──
            observation = environment.observe()
            state.update(observation)

            # Extract and record CLI input if present
            cli_input = None
            if hasattr(observation, "get"):
                cli_input = observation.get("cli_input")
            elif hasattr(observation, "cli_input"):
                cli_input = getattr(observation, "cli_input", None)

            if cli_input:
                cli_text = str(cli_input).strip()
                state.data["pending_user_input"] = cli_text
                state.data["pending_user_input_time"] = time.time()
                # Record in memory so it appears in context
                if hasattr(memory, "record_command"):
                    try:
                        memory.record_command(cli_text)
                    except Exception as e:
                        log_swallowed_exception(e, operation="record_command", context={"text_len": len(cli_text)})

            intent = None
            try:
                if hasattr(agent, "propose_intent"):
                    intent = agent.propose_intent(state, memory)
                elif hasattr(agent, "decide"):
                    # Legacy fallback: treat `decide()` as a goal provider.
                    out = agent.decide(state, memory)
                    if isinstance(out, dict):
                        intent = out
                    elif isinstance(out, str) and out:
                        intent = {"goal": out, "confidence": 1.0}
            except Exception as e:
                warn("Agent propose_intent/decide failed: %s", e)
                intent = None

            if not isinstance(intent, dict) or not intent:
                if break_on_no_intent:
                    break
                _idle_sleep(idle_sleep_s)
                continue

            goal = intent.get("goal") or intent.get("intent")
            if goal is None:
                if break_on_no_intent:
                    break
                _idle_sleep(idle_sleep_s)
                continue

            decision = decision_engine.decide(goal, state, memory)
            if not isinstance(decision, dict) or not decision.get("action"):
                if break_on_no_intent:
                    break
                _idle_sleep(idle_sleep_s)
                continue

        action = decision["action"]
        if not isinstance(action, dict):
            warn("Invalid action selected: %r", action)
            if break_on_no_intent:
                break
            _idle_sleep(idle_sleep_s)
            continue

        ctx = {
            "intent": intent,
            "plan": decision.get("plan"),
            "registered_tools": getattr(getattr(executor, "registry", None), "list", lambda: [])(),
        }
        result = executor.execute(action)
        ctx["tool_result"] = result
        eval_results = []
        for evaluator in evaluators:
            try:
                eval_results.append(evaluator.evaluate(ctx))
            except Exception as e:
                logger.warning("Evaluator %s failed: %s", type(evaluator).__name__, e)
                continue
        try:
            if callable(on_step):
                on_step(
                    {
                        "intent": intent,
                        "goal": goal,
                        "decision": decision,
                        "action": action,
                        "tool_result": result,
                        "evaluations": eval_results,
                        "state": state,
                        "memory": memory,
                    }
                )
        except Exception as e:
            log_swallowed_exception(e, operation="on_step_callback")

        try:
            followup = environment.step(result)
            if followup:
                state.update(followup)
        except Exception as e:
            log_swallowed_exception(e, operation="environment_step")

        try:
            memory.store_raw(
                content={
                    "state": state.snapshot(),
                    "intent": intent,
                    "decision": decision,
                    "tool_result": result,
                    "evaluations": eval_results,
                },
                metadata={"type": "episode"},
            )
        except Exception as e:
            warn("Memory store failed: %s", e)

        if getattr(result, "success", True) is False:
            try:
                state.mark_failure(getattr(result, "error", None))
            except Exception as e:
                log_swallowed_exception(e, operation="mark_failure")

            # ── ADaPT replan: decompose failed goal at depth+1 ──
            failure_strategy = _get_failure_strategy(intent, action)
            if failure_strategy == "replan" and hasattr(decision_engine, "planner"):
                planner = decision_engine.planner
                if hasattr(planner, "decompose"):
                    current_depth = _get_plan_depth(decision)
                    # Pass hippocampus for B4 prior-attempt retrieval if available
                    _hippo = memory if hasattr(memory, "recall") else None
                    replan_ctx = _build_replan_context(
                        intent,
                        action,
                        result,
                        state,
                        hippocampus=_hippo,
                    )
                    try:
                        redecomposed = planner.decompose(
                            failed_goal={"description": str(goal), "tool_name": action.get("tool_name")},
                            replan_ctx=replan_ctx,
                            depth=current_depth,
                        )
                        if redecomposed is not None:
                            state.data["_replan_candidate"] = redecomposed
                            state.data["_replan_goal"] = str(goal)
                            if callable(on_event):
                                on_event(
                                    {
                                        "type": "replan",
                                        "depth": current_depth + 1,
                                        "sub_actions": len(redecomposed.actions),
                                    }
                                )
                    except Exception as e:
                        log_swallowed_exception(e, operation="adaptive_replan")

        try:
            state.steps_taken += 1
        except AttributeError:
            pass

        # Persist state periodically based on persist_every_n_steps setting
        if persist_every_n_steps > 0 and state.steps_taken % persist_every_n_steps == 0:
            _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})

        if state.is_done():
            break

    # Always persist final state
    _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})


# ─────────────────────────────────────────────────────────────────────────
# PFC Deliberation Cycle
# ─────────────────────────────────────────────────────────────────────────


def _wait_for_proposal(
    llm_worker: Any,
    stop_event: Any,
    timeout: float = 300.0,
    ctrl: Any | None = None,
) -> Any:
    """Block until LLM responds, checking stop_event every 100ms.

    Returns LLMProposal or None on timeout/cancellation.

    ``ctrl`` (optional) is stamped with ``last_proposal_time`` when a proposal
    arrives. This path consumes proposals OUTSIDE section 2's poll, so without
    the stamp the planning-liveness backstop would see "submitted, nothing
    came back" after every deliberation tick and requeue a turn the
    deliberation had deliberately declined to act on — an extra, unsolicited
    action in a loop whose actions/turn is a measured quantity (pre-merge
    review, executor lens F3).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            return None
        proposal = llm_worker.get_latest_proposal()
        if proposal is not None:
            if ctrl is not None:
                ctrl.last_proposal_time = time.time()
            return proposal
        time.sleep(0.1)
    logger.warning("_wait_for_proposal timed out after %.0fs", timeout)
    return None


def _planning_liveness_enabled_via_env() -> bool:
    """Operator opt-OUT for the D13 planning-liveness abort.

    The abort terminates a campaign, and it lives inside the measurement
    instrument — apparatus standard S5/S6 say such a control must be
    experiment-visible and disableable (pre-merge review, architecture lens
    S5; mirrors ``MAXIM_SIM_HARD_ABORT`` for the D12 abort). Default ON;
    set to 0/false/no/off to fall back to pre-fix behavior (a dropped
    planning turn idles, which is the bug — use only to reproduce it).
    """
    # Deliberately the MAXIM_SIM_HARD_ABORT idiom, not the canonical
    # ``annotation_disabled_via_env``: that parser is for MAXIM_DISABLE_*
    # style vars where a TRUTHY value means "disable". This is an
    # enable-with-opt-out control, so it mirrors its sibling abort toggle
    # exactly — same file family, same falsy-set, same default-ON meaning.
    return os.environ.get("MAXIM_SIM_PLANNING_LIVENESS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _planning_call_in_flight() -> bool:
    """True when the large lane has a live LLM call per the process-global
    call registry (bugs ledger D13).

    The await window's 120s literal is a WALL-CLOCK guess, and a big model
    routinely exceeds it — qwen2.5-32b measured ~140s/turn in the Exp 37
    heartbeat. Two things therefore key on this observed signal instead of
    the literal: the idle gate stays awake while a call is genuinely in
    flight (otherwise a slow call's proposal is never polled — a second
    flavor of the same livelock), and the liveness backstop only declares a
    turn LOST when nothing is actually running.

    Attribution is per-lane, not per-loop: in a sim the orchestrator and AUT
    loops share the large lane, so a busy AUT reads as "in flight" for the
    orchestrator too. That bias is deliberate — it can only DELAY a liveness
    abort, never manufacture one, and a false abort kills a campaign run
    (same conservatism as ``stall_threshold.should_hard_abort``). A call
    that is in flight but WEDGED stays the D12 hard-abort's job (byte
    silence); a turn lost with nothing running is this fix's job.

    Never raises: registry trouble must not wedge the loop.
    """
    try:
        from maxim.runtime.llm_call_registry import any_call_in_flight

        return bool(any_call_in_flight(tier="large"))
    except Exception as e:
        log_swallowed_exception(e, operation="planning_liveness:any_call_in_flight")
        return False


def _handle_planning_failure(
    ctrl: Any,
    llm_worker: Any,
    sim: Any,
    *,
    reason: str,
    original_request: Any | None,
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
    verdict = ctrl.record_planning_failure(reason=reason)
    if verdict == "already_exhausted":
        return True
    if verdict == "exhausted":
        msg = (
            f"planning liveness exhausted: {ctrl.planning_failure_streak} consecutive "
            f"planning-turn failures (last: {reason}) — aborting sim (bugs ledger D13)"
        )
        logger.error(msg)
        sim.log("EXEC", f"🛑 {msg}")
        log_agentic(
            "agent_loop",
            "planning_liveness_exhausted",
            {"streak": ctrl.planning_failure_streak, "reason": reason},
            level="ERROR",
        )
        return True

    # verdict == "retry"
    if original_request is not None:
        requeued = bool(llm_worker.requeue_request(original_request))
    else:
        requeued = bool(llm_worker.requeue_last_request())
    if not requeued:
        # A rejected requeue is a TRANSPORT failure (queue full under lane
        # contention), not evidence the model cannot plan — spending the
        # retry budget on it would abort a healthy campaign on contention
        # alone. Refund the strike; the window still re-opens below so the
        # expiry backstop tries again. (Pre-merge review, executor lens S2.)
        ctrl.refund_planning_failure()
    attempt = ctrl.planning_failure_streak
    limit = ctrl.planning_retry_limit
    note = (
        f"rescheduled (attempt {attempt}/{limit})"
        if requeued
        else "REQUEUE FAILED (worker queue full) — retry budget NOT spent; the expiry backstop will retry"
    )
    logger.warning("planning turn failed (%s) — %s", reason, note)
    sim.log("EXEC", f"⚠ planning turn failed ({reason}) — {note}")
    log_agentic(
        "agent_loop",
        "planning_retry",
        {"reason": reason, "attempt": attempt, "limit": limit, "requeued": requeued},
        level="WARNING",
    )
    # Re-open the await window in BOTH outcomes: on a successful requeue it
    # tracks the new in-flight turn; on a failed one it paces the expiry
    # backstop to one attempt per window instead of one per idle tick.
    ctrl.last_llm_submit_time = time.time()
    return False


def _jaccard_similarity(keywords_a: set[str], keywords_b: set[str]) -> float:
    """Jaccard similarity between two keyword sets.  Returns 0.0-1.0."""
    if not keywords_a or not keywords_b:
        return 0.0
    union = len(keywords_a | keywords_b)
    if union == 0:
        return 0.0
    return len(keywords_a & keywords_b) / union


def _jaccard_convergence(keywords_a: set[str], keywords_b: set[str], threshold: float = 0.8) -> bool:
    """Check if two keyword sets have converged (Jaccard >= threshold)."""
    if len(keywords_a) < 3 or len(keywords_b) < 3:
        return False
    return _jaccard_similarity(keywords_a, keywords_b) >= threshold


def _compute_thought_salience(
    n_sections: int,
    n_memories: int,
    jaccard_with_previous: float,
) -> float:
    """Compute salience for a THOUGHT WMS entry.

    Range: [0.0, 1.0].  Components weighted to favor novelty
    (a thought that says something new) over mere activation
    (a thought that triggers many systems but says the same thing).

    Args:
        n_sections: Number of bio-system sections that fired (0-5).
        n_memories: Number of memories recalled by enrichment.
        jaccard_with_previous: Jaccard similarity with the previous
            cycle's reasoning keywords (0.0 = fully novel, 1.0 = identical).
    """
    section_score = min(n_sections / 5.0, 1.0)
    recall_score = min(n_memories / 4.0, 1.0)
    novelty_score = 1.0 - jaccard_with_previous
    return 0.3 * section_score + 0.3 * recall_score + 0.4 * novelty_score


def _run_deliberation_cycles(
    *,
    first_proposal: Any,
    bio_enrichment: Any,
    working_memory: Any | None,
    context: Any,
    submit_fn: Any,
    llm_worker: Any,
    stop_event: Any | None,
    thought_gate: Any | None,
    active_goal: str | None,
    step_num: int,
    max_cycles: int = 3,
    ctrl: Any | None = None,
) -> Any | None:
    """PFC deliberation cycles 2+ — recurrence after first proposal.

    Called from the LLM submission point (section 6) when cycle 1 returned
    a proposal with ``ready_to_act == False``.  Cycle 1 enrichment + gate
    check are already done in section 1.2; this function handles the
    recurrence: feed the LLM's reasoning back through bio-enrichment,
    re-submit, wait, repeat.

    Args:
        first_proposal: Cycle 1 result (``ready_to_act == False``).
        bio_enrichment: BioEnrichmentPipeline instance.
        working_memory: WorkingMemorySet (or None).
        context: StructuredContext — mutated in place (bio_enrichment_context,
            working_memory_thoughts).
        submit_fn: Callable(context) -> bool.  Closure that captures all
            LLMWorker.submit_context params (mode, tools, etc.).
        llm_worker: For polling via ``_wait_for_proposal``.
        stop_event: Threading stop event.
        thought_gate: For refractory reset after cycle completes.
        active_goal: Current goal text for enrichment context.
        step_num: Loop iteration counter (for refractory reset).
        max_cycles: Hard cap including cycle 1 (default 3).

    Returns:
        LLMProposal or None.
    """
    from maxim.simulation.sim_logger import (
        sim_log,
        sim_pre_deliberation,
        sim_contemplation,
        sim_deliberation_update,
        sim_deliberation_end,
    )

    recent_keywords: list[set[str]] = []
    last_proposal = first_proposal

    # Collect keywords from cycle 1 reasoning for convergence check
    reasoning_1 = first_proposal.reasoning or ""
    recent_keywords.append(set(reasoning_1.lower().split()))

    enrich_text = reasoning_1

    # Push cycle 1 reasoning to thinking panel
    sim_deliberation_update(reasoning_1, cycle=1, max_cycles=max_cycles)

    # Build accumulating transcript (Stage 1: thoughts build on each other)
    transcript: list[str] = []

    # Cycles 2..max_cycles
    for cycle in range(2, max_cycles + 1):
        # 1. Enrich the LLM's reasoning through bio-systems
        try:
            from maxim.integration.bio_enrichment import EnrichmentContext

            enrich_ctx = EnrichmentContext(active_goal=active_goal)
            enrich_result = bio_enrichment.enrich(enrich_text, context=enrich_ctx, bypass_gate=True)
        except Exception as e:
            log_swallowed_exception(e, operation="deliberation_enrich", context={"cycle": cycle})
            break

        if enrich_result is None:
            logger.debug("Deliberation cycle %d: enrichment returned None, stopping", cycle)
            break

        formatted = bio_enrichment.format_thought_response(enrich_result)
        n_sections = sum(
            1
            for f in (
                enrich_result.memories,
                enrich_result.predictions,
                enrich_result.concepts,
                enrich_result.affordances,
                enrich_result.recent_context,
            )
            if f
        )
        n_memories = len(enrich_result.memories) if enrich_result.memories else 0

        # Compute Jaccard with previous cycle for salience novelty signal
        current_keywords = set(enrich_text.lower().split())
        jaccard_prev = _jaccard_similarity(current_keywords, recent_keywords[-1]) if recent_keywords else 0.0
        salience = _compute_thought_salience(n_sections, n_memories, jaccard_prev)

        # Derive enrichment tag names for thinking panel
        _enrich_tags: list[str] = []
        if enrich_result.memories:
            _enrich_tags.append("hippocampus")
        if enrich_result.predictions:
            _enrich_tags.append("nac")
        if enrich_result.concepts:
            _enrich_tags.append("ec")
        if enrich_result.affordances:
            _enrich_tags.append("cerebellum")
        if enrich_result.recent_context:
            _enrich_tags.append("scn")

        sim_pre_deliberation(
            gate_passed=True,
            score=0.0,
            threshold=0.0,
            enrichment_sections=n_sections,
        )
        sim_log(
            "DELIBERATION",
            f"cycle {cycle}: {n_sections} enrichment section(s), "
            f"salience={salience:.2f}, {n_memories} memories, "
            f"novelty={1.0 - jaccard_prev:.2f}",
        )
        sim_deliberation_update(
            enrich_text,
            cycle=cycle,
            max_cycles=max_cycles,
            enrichment_tags=_enrich_tags,
            salience=salience,
        )

        # 2. Add THOUGHT to working memory with computed salience
        if working_memory is not None and formatted:
            import uuid

            from maxim.agents.working_memory import WorkingMemoryKind

            working_memory.add(
                WorkingMemoryKind.THOUGHT,
                ref=f"thought:{uuid.uuid4().hex[:8]}",
                content={"source": "pfc_deliberation", "cycle": cycle, "enrichment": formatted[:500]},
                salience=salience,
                goal_tag=active_goal,
            )

        # 3. Build transcript entry (reasoning + bio-system response)
        if formatted:
            entry = f"You thought: {enrich_text[:600]}\nYour experience responded:\n{formatted[:800]}"
            transcript.append(entry)

        # 4. Update StructuredContext
        if formatted:
            context.bio_enrichment_context = formatted

        # Set transcript on context (replaces working_memory_thoughts for multi-cycle)
        if transcript:
            context.deliberation_transcript = list(transcript)

        # Populate working_memory_thoughts for backward compat (prompt builder
        # uses transcript when present, falls back to this)
        if working_memory is not None:
            from maxim.agents.working_memory import WorkingMemoryKind

            thought_entries = working_memory.by_kind({WorkingMemoryKind.THOUGHT}, limit=6)
            context.working_memory_thoughts = [
                str(e.content.get("enrichment", ""))[:200] if isinstance(e.content, dict) else str(e.content)[:200]
                for e in thought_entries
            ]

        # 5. Re-submit to LLM and wait
        if not submit_fn(context):
            logger.warning("LLM worker queue full during deliberation cycle %d", cycle)
            break

        proposal = _wait_for_proposal(llm_worker, stop_event, ctrl=ctrl)
        if proposal is None:
            break

        last_proposal = proposal

        # 6. Check ready_to_act
        if proposal.ready_to_act:
            sim_contemplation(
                gate_passed=True,
                refined=True,
                score=0.0,
            )
            sim_log("DELIBERATION", f"deliberation converged: ready_to_act after {cycle} cycles")
            sim_deliberation_end(cycle=cycle, max_cycles=max_cycles, summary=f"Ready to act after {cycle} cycles")
            if thought_gate is not None:
                thought_gate.reset_refractory(step_num)
            return proposal

        # 7. Not ready — check convergence before looping
        reasoning = proposal.reasoning or ""
        keywords = set(reasoning.lower().split())
        recent_keywords.append(keywords)

        if len(recent_keywords) >= 2 and _jaccard_convergence(recent_keywords[-1], recent_keywords[-2]):
            logger.info("Deliberation converged after %d cycles (Jaccard >= 0.8)", cycle)
            sim_contemplation(
                gate_passed=True,
                refined=True,
                score=0.0,
            )
            sim_log("DELIBERATION", f"deliberation converged (Jaccard) after {cycle} cycles")
            sim_deliberation_end(
                cycle=cycle, max_cycles=max_cycles, summary=f"Converged (Jaccard) after {cycle} cycles"
            )
            if thought_gate is not None:
                thought_gate.reset_refractory(step_num)
            return proposal if proposal.action else None

        # 8. Feed reasoning back for next enrichment
        enrich_text = reasoning
        logger.debug("Deliberation cycle %d: reasoning fed back for enrichment (%d chars)", cycle, len(reasoning))

    # Max cycles reached without ready_to_act
    sim_contemplation(
        gate_passed=True,
        refined=True,
        score=0.0,
    )
    sim_log("DELIBERATION", f"max cycles ({max_cycles}) reached, forcing action")
    sim_deliberation_end(
        cycle=max_cycles, max_cycles=max_cycles, summary=f"Max cycles ({max_cycles}) reached, forcing action"
    )
    if thought_gate is not None:
        thought_gate.reset_refractory(step_num)

    # Return last proposal if it has an action, otherwise None (IDLE)
    if last_proposal is not None and last_proposal.action:
        return last_proposal
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Substrate-primary action generation (Phase -1 of grounded_language_acquisition.md)
# ─────────────────────────────────────────────────────────────────────────────


def _read_drive_states(executor: Any) -> dict[str, float]:
    """Extract current drive values from the executor's embodiment.

    Returns ``{drive_name: value in [0, 1]}`` for every drive declared on
    every entity walked from the embodiment root. Empty dict when no
    embodiment is wired.

    Used by substrate-primary AUT mode to feed ``NAc.recommend_action``
    without going through the LLM. Reads the current values directly from
    ``Entity.vital_metrics`` (entity-level drives) and modulator
    ``vital_metrics`` (sub-sensor drives like ``arms.thermal``).
    """
    embodiment = getattr(executor, "embodiment", None)
    if embodiment is None or getattr(embodiment, "root", None) is None:
        return {}

    drives: dict[str, float] = {}
    # Derived corrective "cold" need (see below). Accumulated as the max
    # breach across all homeostatic thermal drives, emitted once at the end.
    cold_need = 0.0
    for ent in embodiment.root.walk():
        specs = getattr(ent, "drive_specs", {})
        for ds_name, spec in specs.items():
            if "." in ds_name:
                mod_name, sensor_name = ds_name.split(".", 1)
                mod = ent.modulators.get(mod_name)
                if mod is None or not hasattr(mod, "vital_metrics"):
                    continue
                value = mod.vital_metrics.get(sensor_name)
            else:
                value = ent.vital_metrics.get(ds_name)
            if value is None:
                continue
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            drives[ds_name] = fval

            # Derive a positive corrective "cold" need from homeostatic thermal
            # DEFICITS. The drive-affinity heuristic in NAc.recommend_action only
            # fires on positive [0,1] need intensities (entropic drives like
            # hunger that climb up), so a homeostatic deficit — cold = a
            # temperature drive sitting below its set_point — is otherwise
            # invisible to substrate-primary action selection, and warmth-seeking
            # affordances never become salient. This completes the homeostatic
            # drive protocol for the LLM-free path (this function is only called
            # by propose_via_substrate; LLM-AUT reads body_state directly, so
            # Exp 37/38 are unaffected). The derived need maps to the existing
            # "cold" affinity → ("warm","fire","blanket","huddle"). Above-set_point
            # (hot) has no corrective affinity entry today, so only the cold
            # direction is derived.
            # NOTE: thermal drives are detected by NAME convention ("temp"/
            # "thermal" in the drive name) — the affinity table is keyed on the
            # semantic "cold" need, and we have no structured drive→need-name map
            # yet. If a thermal drive is named otherwise, add it here (or give
            # DriveSpec a declared corrective-need name).
            set_point = getattr(spec, "set_point", None)
            if set_point is not None and ("temp" in ds_name.lower() or "thermal" in ds_name.lower()):
                comfort = float(getattr(spec, "comfort_band", 0.0) or 0.0)
                deviation = fval - float(set_point)
                if deviation < -comfort:  # below set_point, past the comfort band
                    cold_need = max(cold_need, min(1.0, abs(deviation)))

    if cold_need > 0.0:
        # setdefault: never clobber a real drive literally named "cold".
        drives.setdefault("cold", cold_need)
    return drives


def _read_drive_ranges(executor: Any) -> "dict[str, tuple[float, float]]":
    """Per-sensor ``(lo, hi)`` range for the drive sensors ``_read_drive_states``
    reads, so ``SensorEncoder.encode_sensors`` normalizes SIGNED sensors
    (azimuth / thermal on ``[-1, 1]``) MONOTONICALLY instead of folding (P1 —
    the range-blind map aliases center with hard-left and collides opposite-sign
    values near center, so a left sound and a right sound could share one EC
    cluster and the orient policy couldn't condition on direction).

    Mirrors ``_read_drive_states``' walk (they iterate the same ``drive_specs``);
    the ``test_read_drive_ranges_covers_every_signed_drive`` guard pins that they
    agree so a future signed drive can't silently re-fold. A drive sensor with no
    declared range is omitted → the encoder falls back to the legacy ``[0, 1]``-ish
    map, which is correct for ``[0, 1]`` drives (hunger/thirst/energy) and the
    derived ``"cold"`` need (also ``[0, 1]``). Only signed sensors need a range.

    UNITS INVARIANT: ``reading_schema["range"]`` MUST be in the same units as the
    values ``_read_drive_states`` reads from ``vital_metrics`` (which ``spec.py``
    initializes from the declared range, so this holds for every YAML drive). A
    drive that declared a raw-unit range (e.g. ``[0, 360]``) but wrote normalized
    values would map every value near 0 — worse than the fold. Do not mix units.
    """
    embodiment = getattr(executor, "embodiment", None)
    if embodiment is None or getattr(embodiment, "root", None) is None:
        return {}
    ranges: dict[str, tuple[float, float]] = {}
    for ent in embodiment.root.walk():
        for ds_name in getattr(ent, "drive_specs", {}):
            rng = None
            if "." in ds_name:
                mod_name, sub_name = ds_name.split(".", 1)
                mod = ent.modulators.get(mod_name)
                if mod is not None and hasattr(mod, "_sensors"):
                    sub = mod._sensors.get(sub_name, {})
                    if isinstance(sub, dict):
                        rng = sub.get("range")
            else:
                sensor = ent.sensors.get(ds_name)
                if sensor is not None:
                    rng = sensor.reading_schema.get("range")
            # Per-sensor guard: a malformed range (non-iterable scalar, wrong
            # length, non-numeric bounds) must NOT bubble — this function is
            # evaluated as an argument inside the encode_sensors try/except, so a
            # raise here would silently disable ALL substrate encoding for the
            # agent, every tick. Skip just the bad sensor → it falls back to the
            # legacy [0,1] map instead.
            try:
                if rng is not None and len(rng) == 2:
                    ranges[ds_name] = (float(rng[0]), float(rng[1]))
            except (TypeError, ValueError):
                logger.debug("drive %r has a malformed range %r; skipping (legacy map)", ds_name, rng)
    return ranges


# Exteroceptive world-set sensors the substrate encodes for PERCEPTION (not as
# drives/needs). ``azimuth`` = head-relative sound direction (base_humanoid's
# capability-driven orient sensor). Kept a named set so a future exteroceptive
# sensor (e.g. a light-direction) is one entry, not a code change at the read site.
_EXTEROCEPTIVE_ROOT_SENSORS: tuple[str, ...] = ("azimuth",)


# Place-code opt-in (modality_resolution_and_alignment.md; Exp 46 validated).
# Default OFF: turning it on changes EC cluster identity for the audio channel,
# which is a re-validation trigger for Exp 48 (and Exp 46's own numbers). Same
# default-OFF-pending-ablation shape as MAXIM_ENABLE_BODY_STATE_PROMPT.
_PLACE_CODE_ENV = "MAXIM_PLACE_CODE_EXTEROCEPTION"
_PLACE_CODE_PREFIX = "azdir"


def place_code_exteroception_enabled() -> bool:
    """True when the exteroceptive channel should emit a population code.

    Read per call (not cached): the autouse conftest scrub flips it between
    tests, and a cached read would leak one test's arm into the next.
    """
    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

    return annotation_disabled_via_env(os.environ.get(_PLACE_CODE_ENV))


def _read_exteroceptive_states(executor: Any) -> dict[str, float]:
    """Read world-set EXTEROCEPTIVE root sensors (``azimuth``) — the value
    source for the ``"audio"`` ModalityChannel, encoded in its OWN
    ``encode_sensors(modality="audio")`` call so an agent can condition its
    action on WHERE a stimulus is, even when it carries no drive/need about it.

    Distinct from ``_read_drive_states``: those are interoceptive needs that
    also drive the affinity heuristic; these are pure perception and NEVER
    enter ``current_drives`` — nor the interoception encode (the pre-seam
    ``{**drives, **extero}`` merge diluted direction among the drives and
    collapsed left/right onto one cluster; see
    docs/plans/exteroception_interoception_seam.md). Load-bearing for
    ``bodies/infant_operant`` (cradle_mother operant experiment), whose azimuth
    sensor has ``drive: null``. A body whose azimuth ALSO carries a drive gets
    the value in BOTH encodes. The plan's intent is two representations of two
    DIFFERENT things — location (audio cluster, this read) vs discomfort
    (interoception cluster, the drive read) — but that split is only
    STRUCTURALLY REACHABLE today, not enforced: ``_read_drive_states`` reads
    the raw signed value (not a comfort-distance fold), so for such a body the
    interoception encode carries the same signed azimuth as the audio encode,
    and drive-relief credit (interoception) plus operant credit (audio) can
    reinforce the same directional contingency on two stacking clusters. No
    shipped body has an azimuth drive + this sensor; folding drive-bearing
    signed sensors to discomfort magnitude in the intero read is a named
    deferred item in docs/plans/exteroception_interoception_seam.md (trigger:
    the first body that gives azimuth a drive).
    """
    embodiment = getattr(executor, "embodiment", None)
    root = getattr(embodiment, "root", None)
    if root is None:
        return {}
    sensors = getattr(root, "sensors", {}) or {}
    vm = getattr(root, "vital_metrics", {}) or {}
    out: dict[str, float] = {}
    for name in _EXTEROCEPTIVE_ROOT_SENSORS:
        if name in sensors and name in vm:
            try:
                out[name] = float(vm[name])
            except (TypeError, ValueError):
                continue
    if out and place_code_exteroception_enabled():
        # Population code REPLACES the raw scalar — emitting both would hand the
        # encoder a redundant basis pair whose constant-ish contribution dilutes
        # the very dimension the code exists to resolve (the extero/intero
        # dilution failure, one level down).
        from maxim.similarity.place_code import place_code

        coded: dict[str, float] = {}
        for name, value in out.items():
            coded.update(place_code(value, prefix=f"{_PLACE_CODE_PREFIX}_{name}_"))
        return coded
    return out


def _read_exteroceptive_ranges(executor: Any) -> "dict[str, tuple[float, float]]":
    """Declared ``(lo, hi)`` for the exteroceptive sensors ``_read_exteroceptive_
    states`` reads, so signed sensors (azimuth on ``[-1, 1]``) fold MONOTONICALLY
    (P1) — a left sound and a right sound must not collapse into one cluster."""
    embodiment = getattr(executor, "embodiment", None)
    root = getattr(embodiment, "root", None)
    if root is None:
        return {}
    sensors = getattr(root, "sensors", {}) or {}
    # LOCKSTEP INVARIANT (same class as _read_drive_ranges): this walk and
    # _read_exteroceptive_states must emit the same sensor SET. A value with no
    # declared range silently re-folds through the legacy range-blind map (P1),
    # so a place-coded value walk with a raw range walk would encode seven
    # activations under the wrong normalisation. Guarded by
    # test_place_code_wiring.py::test_value_and_range_walks_stay_in_lockstep.
    if place_code_exteroception_enabled():
        from maxim.similarity.place_code import place_code_ranges

        coded_ranges: dict[str, tuple[float, float]] = {}
        for name in _EXTEROCEPTIVE_ROOT_SENSORS:
            if sensors.get(name) is None:
                continue
            coded_ranges.update(place_code_ranges(prefix=f"{_PLACE_CODE_PREFIX}_{name}_"))
        return coded_ranges

    ranges: dict[str, tuple[float, float]] = {}
    for name in _EXTEROCEPTIVE_ROOT_SENSORS:
        sensor = sensors.get(name)
        if sensor is None:
            continue
        rng = sensor.reading_schema.get("range")
        try:
            if rng is not None and len(rng) == 2:
                ranges[name] = (float(rng[0]), float(rng[1]))
        except (TypeError, ValueError):
            logger.debug("exteroceptive %r has a malformed range %r; skipping (legacy map)", name, rng)
    return ranges


# ── Substrate modality channels (extero/intero seam) ─────────────────────
#
# Declarative registry: one entry per sensory stream, one
# ``encode_sensors(modality=tag)`` call per non-empty channel — NEVER merged
# into a single encode (docs/plans/exteroception_interoception_seam.md: the
# pre-seam ``{**drives, **extero}`` merge diluted exteroceptive direction
# among the interoceptive drives in one text-embed cluster, collapsing
# left/right onto the same EC node → the embodied orient sim at chance).
# Adding a future modality (vision, touch) is one tuple entry here.
# EC scans within-modality only and "audio" is already frozen-centroid, so
# each channel gets its own cluster space with the right centroid policy.
# NOTE (selection dynamics): ``max_cluster_reward_bias`` caps PER cluster, so
# the summed cluster term in ``recommend_action`` scales with the number of
# active channels (±N for N modalities) — adding a channel here is a
# selection-dynamics change; re-check gate calibration (min_confidence)
# when you add one.
_SUBSTRATE_CHANNELS: "tuple[ModalityChannel, ...]" = (
    ModalityChannel(INTEROCEPTION_TAG, _read_drive_states, _read_drive_ranges),
    ModalityChannel(AUDIO_TAG, _read_exteroceptive_states, _read_exteroceptive_ranges),
)


def _encode_current_clusters(sensor_encoder: Any, agent_id: str, executor: Any) -> dict[str, str]:
    """Encode the CURRENT sensor state into ``{modality: cluster_id}``.

    The same per-channel encode ``propose_via_substrate`` does, but callable at
    outcome time so an llm-primary / real-hardware action (where the LLM, not the
    substrate, chose the action) can still key its real drive-relief outcome onto
    the interoception (and audio) cluster — closing the substrate WRITE path in
    those modes (Phase 1, substrate_learns_from_experience.md). Returns ``{}`` on
    no encoder / no sensors / encode failure (never raises into the loop).
    """
    clusters: dict[str, str] = {}
    if sensor_encoder is None:
        return clusters
    for ch in _SUBSTRATE_CHANNELS:
        try:
            vals = ch.read_values(executor)
            if not vals:
                continue
            node_id = sensor_encoder.encode_sensors(
                agent_id=agent_id,
                sensors=vals,
                modality=ch.tag,
                ranges=ch.read_ranges(executor) or None,
            )
        except Exception:
            # Same policy as propose_via_substrate: a channel that fails to encode
            # is "sensors but no cluster" — surface it, don't crash.
            logger.warning(
                "substrate channel %r encoding raised at outcome time — cluster absent",
                ch.tag,
                exc_info=True,
            )
            continue
        if node_id:
            clusters[ch.tag] = node_id
    return clusters


_DEFAULT_SUBSTRATE_MIN_CONFIDENCE = 0.3


def _resolve_min_confidence(explicit: float | None) -> float:
    """Resolve ``min_confidence`` for ``propose_via_substrate``.

    Precedence: explicit caller argument > ``MAXIM_NAC_MIN_CONFIDENCE`` env
    var > ``_DEFAULT_SUBSTRATE_MIN_CONFIDENCE`` (0.3). The env var exists for
    Roy-2c (H1 vs H2 disambiguator) and the Wire-A ablation surface in
    [docs/plans/archive/release_0_9_1.md](../../docs/plans/archive/release_0_9_1.md). Invalid
    env values fall back to the default with a warning, not a crash.
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get("MAXIM_NAC_MIN_CONFIDENCE")
    if raw is None or raw == "":
        return _DEFAULT_SUBSTRATE_MIN_CONFIDENCE
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "MAXIM_NAC_MIN_CONFIDENCE=%r is not a float; using default %.2f",
            raw,
            _DEFAULT_SUBSTRATE_MIN_CONFIDENCE,
        )
        return _DEFAULT_SUBSTRATE_MIN_CONFIDENCE


def propose_via_substrate(
    *,
    nac: Any,
    agent_id: str,
    executor: Any,
    min_confidence: float | None = None,
    sensor_encoder: Any | None = None,
) -> LLMProposal | None:
    """Build an ``LLMProposal`` from ``NAc.recommend_action`` — no LLM call.

    Called from the agent loop when ``aut_mode == "substrate-primary"`` in
    place of ``llm_worker.submit_context``. Returns ``None`` when the
    substrate has no opinion (no learned bias, no active drive) — the loop
    treats this as IDLE for that tick rather than proposing randomly.

    The returned proposal carries ``strategy_used="substrate-primary"`` so
    downstream tracing can distinguish substrate-proposed actions from
    LLM-proposed ones.

    Args:
        sensor_encoder: Optional :class:`SensorEncoder` (Phase 0 of
            grounded_language_acquisition.md). When wired, the current
            drive snapshot is hashed into the substrate via
            ``encode_sensors`` once per tick *before* reading drives for
            ``recommend_action``. This lets EC accumulate sensor-pattern
            nodes during substrate-primary runs — without it the
            text-only ``LinguisticEncoder`` path is the substrate's only
            front door, so substrate-primary mode never produces EC nodes.
    """
    if nac is None or executor is None:
        return None

    registry = getattr(executor, "registry", None)
    if registry is None or not hasattr(registry, "list"):
        return None

    available_tools = list(registry.list())
    if not available_tools:
        return None

    # Substrate-primary is an EMBODIED action test: exclude read-only cognitive
    # introspection tools (memory_recall, temporal_patterns, system_stats, …).
    # They always succeed, so their causal confidence snowballs toward the cap
    # and dominates recommend_action — starving the embodied affordances the
    # mode exists to measure (the meta-tool fixation that VOID'd the Exp 42
    # triage: the agent fidgeted with temporal_patterns/system_stats instead of
    # warming). LLM-AUT is unaffected — it never calls this path. Set lives in
    # tools/introspection.py so it can't drift from the registered tool names.
    from maxim.tools.introspection import INTROSPECTION_TOOL_NAMES

    available_tools = [t for t in available_tools if t not in INTROSPECTION_TOOL_NAMES]
    if not available_tools:
        return None

    # Optional experiment-scoped whitelist: restrict substrate-primary action
    # selection to a MINIMAL affordance repertoire. The introspection filter above
    # only removes read-only cognitive tools; non-introspection tools that also
    # "always succeed" (sense_presence, sense, examine, say, …) still snowball
    # causal confidence and out-compete the affordances under test — the cradle
    # orient infant kept choosing sense_presence (causal_pos 0.99) over turn_left/
    # turn_right. A newborn's motor repertoire is small; the 22 generic tools are
    # the artificial part. Substring match (tools are body-prefixed, e.g.
    # infant_operant_turn_left). Experiment/harness toggle (env, not config).
    # Autouse scrub: tests/conftest.py.
    #
    # BAND-AID (tracked): this masks the ROOT cause rather than fixing it — a tool
    # that merely EXECUTES accrues causal credit as if it made goal/drive progress,
    # so mechanically-successful tools drown a specific operant/drive signal. The
    # real fix (credit-on-progress-not-execution) is
    # docs/plans/deferred/credit_on_progress_not_execution.md; this whitelist is a
    # scoped work-around for the dormant cradle_mother demo until that lands.
    _tool_whitelist = os.environ.get("MAXIM_SUBSTRATE_TOOL_WHITELIST", "").strip()
    if _tool_whitelist:
        _wl_terms = [w.strip() for w in _tool_whitelist.split(",") if w.strip()]
        if _wl_terms:
            available_tools = [t for t in available_tools if any(term in t for term in _wl_terms)]
            if not available_tools:
                return None

    # Substrate-primary mode owns its own clock — without an LLM submit
    # path there's no other code that calls into the embodiment, so
    # drive drift would never advance. Ticking evaluate_failures() here
    # mirrors the llm-primary path, where the tick is event-driven via
    # tool execution (tool_bridge / sim tools calling evaluate_failures):
    # applies wall-clock drift via tick_vital_drift, then evaluates
    # failures (which publish pain signals to NAc). See the CLAUDE.md
    # embodiment-tick invariant.
    embodiment = getattr(executor, "embodiment", None)
    if embodiment is not None:
        try:
            embodiment.evaluate_failures()
        except Exception:
            logger.debug("substrate-primary tick: evaluate_failures raised", exc_info=True)

    # Per-modality channel reads (extero/intero seam). Each channel is read
    # once; interoception feeds ``current_drives`` (the drive-affinity
    # heuristic — perception is not a need, so exteroceptive channels stay
    # out of it) and EVERY non-empty channel gets its OWN encode below.
    channel_values: dict[str, dict[str, float]] = {ch.tag: ch.read_values(executor) for ch in _SUBSTRATE_CHANNELS}
    drives = channel_values.get(INTEROCEPTION_TAG, {})

    # Phase 0 sensor encoding — feed the current sensor snapshot to EC so
    # substrate-primary mode produces nodes the way the LLM-primary
    # text-percept path does. ONE ``encode_sensors(modality=tag)`` call per
    # non-empty channel — NEVER merged: the pre-seam ``{**drives, **extero}``
    # merge encoded exteroceptive direction as one term in a text-embed sum
    # dominated by the drives, so left/right collapsed onto one EC cluster
    # and the agent was blind to direction (the dilution root cause,
    # docs/plans/exteroception_interoception_seam.md). Fail-soft per channel:
    # an encoding error must not block the action proposal or the other
    # channels. The resulting ``{modality: cluster_id}`` set flows into
    # recommend_action (additive cluster_reward_bias sum) and onto the
    # proposal for the outcome path's credit routing.
    clusters: dict[str, str] = {}
    if sensor_encoder is not None:
        for ch in _SUBSTRATE_CHANNELS:
            vals = channel_values.get(ch.tag)
            if not vals:
                continue
            try:
                node_id = sensor_encoder.encode_sensors(
                    agent_id=agent_id,
                    sensors=vals,
                    modality=ch.tag,
                    ranges=ch.read_ranges(executor) or None,
                )
            except Exception:
                # WARNING, not debug: a channel with sensors that failed to
                # encode IS "sensors but no cluster" — downstream, the credit
                # router silently falls back (operant pending keys on
                # interoception when audio is missing), so a quiet failure
                # here becomes invisible mis-routed credit (pre-merge review,
                # both lenses).
                logger.warning(
                    "substrate channel %r encoding raised — its cluster is absent this tick",
                    ch.tag,
                    exc_info=True,
                )
                continue
            if node_id:
                clusters[ch.tag] = node_id
            else:
                # A channel with sensors that yields no cluster is the
                # dilution failure mode's silent sibling — surface it.
                logger.warning(
                    "substrate channel %r has %d sensor(s) but yielded no cluster",
                    ch.tag,
                    len(vals),
                )
    cluster_id = clusters.get(INTEROCEPTION_TAG)

    resolved_min_confidence = _resolve_min_confidence(min_confidence)
    recommendation = nac.recommend_action(
        agent_id=agent_id,
        available_tools=available_tools,
        current_drives=drives or None,
        current_cluster_id=cluster_id,
        current_clusters=clusters or None,
        min_confidence=resolved_min_confidence,
    )
    if recommendation is None:
        return None

    action = {
        "tool_name": recommendation["tool_name"],
        "params": recommendation.get("params", {}) or {},
    }
    return LLMProposal(
        request_id=f"substrate-{int(time.time() * 1000)}",
        action=action,
        reasoning=recommendation.get("reasoning", ""),
        strategy_used="substrate-primary",
        confidence=float(recommendation.get("confidence", resolved_min_confidence)),
        mode_goal_achieved=False,
        triggering_input="",
        # G4 + seam: stash the active EC cluster set on the proposal so the
        # outcome path can route credit per modality into
        # ``NAc._cluster_reward_bias[(agent, cluster, tool)]`` — see
        # record_outcome in tool_dispatch.py. ``cluster_id`` is the legacy
        # interoception alias; ``clusters`` is the full per-modality set.
        # Both ``None``/empty when no sensor encoder was wired or no channel
        # produced a cluster.
        cluster_id=cluster_id,
        clusters=clusters or None,
    )


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


def run_agentic_loop(
    agent: Any,
    environment: Any,
    state: Any,
    memory: Any,
    decision_engine: Any,
    executor: Any,
    *,
    autonomy_controller: AutonomyController | None = None,
    llm_worker: LLMWorker | None = None,
    default_network: Any | None = None,  # DefaultNetwork for reactive behaviors
    hippocampus: Any | None = None,  # Hippocampus for episodic memory
    memory_hub: Any | None = None,  # MemoryHub for cross-system memory integration
    evaluators: list[Evaluator] | None = None,
    max_steps: int = 0,  # 0 = unlimited
    run_id: str | None = None,
    stop_event: Any | None = None,
    on_step: Any | None = None,
    on_event: Any | None = None,  # Fine-grained streaming callback
    idle_sleep_s: float = 0.05,  # Fast loop for responsiveness
    persist_every_n_steps: int = 10,
    target_hz: float = 30.0,  # Target loop frequency
    context_pool_config: dict[str, Any] | None = None,  # Context pool configuration
    use_tool_prompting: bool = True,  # Enable tool-aware LLM prompts
    protocol_registry: Any | None = None,  # ProtocolRegistry for dynamic skills
    percept_source: Any | None = None,  # PerceptSource for simulation
    action_sink: Any | None = None,  # ActionSink for recording tool outputs
    pain_bus: Any | None = None,  # PainBus for simulation pain routing
    imagination_trigger: Any | None = None,  # ImaginationTrigger for real-time entity design
    bio_enrichment_pipeline: Any | None = None,  # BioEnrichmentPipeline for percept enrichment (L1)
    thought_gate: Any | None = None,  # ThoughtGate for PFC deliberation gating
    aut_mode: str = "llm-primary",  # "llm-primary" | "substrate-primary" — Phase -1 of grounded_language_acquisition.md
    substrate_telemetry: Any
    | None = None,  # SubstrateTelemetry writer (Phase 0). Called after each substrate-primary tick when set.
    substrate_action_gate: Any | None = None,  # Callable[[], bool] — turn-scoped action budget for the
    # substrate-primary branch (the Exp 48 thrashing fix). When set and returning False, the branch
    # skips proposing this cadence tick (telemetry still fires with proposal=None). The orchestrator
    # wires SimulationBridge.substrate_action_allowed here; None = unbounded (pre-fix behavior).
    consolidation: Literal["full", "lightweight"] | None = None,  # HANDLE seam (b): explicit session-end flavor
    planning_liveness: bool = False,  # Bugs ledger D13 — OPT-IN, and deliberately so.
    # Enables bounded reschedule-or-abort for planning turns that end without an executable
    # proposal. Correct ONLY for a loop with no other wake source between turns: the sim
    # ORCHESTRATOR, which has no action_sink and no external percept producer, so a dropped
    # planning turn idles it forever. Every other caller already has a re-arm path and must
    # NOT opt in — the AUT recovers via the `_llm_unavailable` synthetic action breaking the
    # bridge's settle loop (so the next probe re-arms it), HANDLE/interactive have a human,
    # and the live-robot + headless paths have real percept producers. Enabling it wholesale
    # would convert those existing RECOVERIES into aborts (pre-merge review, architecture
    # lens B3) and, for a threaded AUT, into a silently dead thread whose campaign keeps
    # emitting empty turns (B1).
    sim_adapter: Any | None = None,  # Pre-built NullSimulationAdapter (Stage 3, live_audio_orient_wiring.md):
    # lets a live producer (the DoA feed) hold the adapter and carry_percept() into the loop's
    # modality-preserving side-channel WITHOUT a percept_source (which would flip is_sim_mode).
    # Only honored when percept_source is None; ignored (with the sim adapter built as before) otherwise.
) -> None:
    """
    Non-blocking agentic loop with LLM worker integration.

    Key differences from run_agent_loop:
    - Never blocks on LLM inference
    - Checks for LLM proposals asynchronously
    - Applies autonomy level gating before execution
    - Maintains target loop frequency for real-time responsiveness
    - Hard stops work instantly regardless of LLM state
    - Integrates with DefaultNetwork for reactive behaviors

    Args:
        autonomy_controller: Controls what actions can be executed
        llm_worker: Background LLM worker for async inference
        default_network: DefaultNetwork for reactive behaviors (optional)
        target_hz: Target loop frequency (default 30Hz)
    """
    from maxim.agents.autonomy import (
        AutonomyLevel,
        AutonomyController,
        Proposal,
        check_hard_stop,
    )
    from maxim.agents.context_pool import ContextPool, ContextPoolConfig
    from maxim.agents.llm_worker import ModeInfo
    from maxim.modes.definitions import get_mode, TOOL_DESCRIPTIONS
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.loop_types import ActionFollowup
    from maxim.runtime.prefetch import (
        init_prefetcher,
        get_result_cache,
    )

    if evaluators is None:
        evaluators = []

    # Wrap executor with instrumentation if action_sink is provided
    if action_sink is not None:
        from maxim.simulation.instrumented_executor import InstrumentedExecutor

        executor = InstrumentedExecutor(executor, action_sink)

    # Create simulation adapter (Phase 4: isolate sim concerns)
    from maxim.runtime.sim_adapter import SimulationAdapter, NullSimulationAdapter

    if percept_source is not None:
        sim = SimulationAdapter(percept_source, action_sink, pain_bus)
        # Wire tool registry for deregistered-tool filtering in should_skip_fallback_proposal
        if executor is not None and hasattr(executor, "registry"):
            sim._tool_registry = executor.registry
    else:
        # Stage 3 (live_audio_orient_wiring.md): a caller-held adapter lets a
        # live producer carry_percept() into the side-channel; is_sim_mode
        # stays False either way. A sim-mode adapter smuggled through this
        # kwarg would flip the 12 is_sim_mode consumer sites without a
        # percept_source — fail loud instead (pre-merge review fold).
        if sim_adapter is not None and getattr(sim_adapter, "is_sim_mode", True) is not False:
            raise ValueError(
                "sim_adapter= must be a non-sim adapter (is_sim_mode False); "
                "sim mode is entered via percept_source=, never this kwarg"
            )
        sim = sim_adapter if sim_adapter is not None else NullSimulationAdapter()

    if not run_id:
        run_id = time.strftime("%Y-%m-%d_%H%M%S")
    agent_name = _safe_agent_name(agent)
    state_path = os.path.join("data", "agents", agent_name, "runtime", f"state_{run_id}.json")
    _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})

    # Initialize autonomy controller if not provided
    if autonomy_controller is None:
        autonomy_controller = AutonomyController()

    # Initialize context pool for accumulated observations
    pool_config = ContextPoolConfig()
    if context_pool_config:
        pool_config = ContextPoolConfig(
            max_tokens=context_pool_config.get("max_tokens", 2000),
            summary_target_tokens=context_pool_config.get("summary_target_tokens", 500),
            max_entries=context_pool_config.get("max_entries", 50),
            keep_recent=context_pool_config.get("keep_recent", 5),
            include_agent_states=context_pool_config.get("include_agent_states", True),
            include_outcomes=context_pool_config.get("include_outcomes", True),
            include_abstractions=context_pool_config.get("include_abstractions", True),
            persistence_path=context_pool_config.get("persistence_path"),
        )
    context_pool = ContextPool(config=pool_config)

    # Initialize speculative pre-fetcher for efficient context gathering
    prefetcher = init_prefetcher(executor=executor, base_path=os.getcwd())
    result_cache = get_result_cache()

    # ── LoopController holds all transient state (Phase 1+2) ─────────────
    ctrl = LoopController(
        agent=agent,
        environment=environment,
        state=state,
        memory=memory,
        decision_engine=decision_engine,
        executor=executor,
        autonomy_controller=autonomy_controller,
        llm_worker=llm_worker,
        default_network=default_network,
        hippocampus=hippocampus,
        memory_hub=memory_hub,
        evaluators=evaluators,
        max_steps=max_steps,
        run_id=run_id,
        stop_event=stop_event,
        on_step=on_step,
        on_event=on_event,
        idle_sleep_s=idle_sleep_s,
        persist_every_n_steps=persist_every_n_steps,
        target_hz=target_hz,
        use_tool_prompting=use_tool_prompting,
        protocol_registry=protocol_registry,
        percept_source=percept_source,
        action_sink=action_sink,
        pain_bus=pain_bus,
    )
    ctrl.context_pool = context_pool
    ctrl.prefetcher = prefetcher
    ctrl.result_cache = result_cache

    # Thought novelty tracker: deque of recent thought word-sets for
    # cross-turn novelty gating.  Thoughts with >= 75% word overlap with
    # any recent entry are suppressed from the display (they're redundant).
    from collections import deque as _deque

    _recent_thought_words: _deque[set[str]] = _deque(maxlen=8)

    def _is_novel_thought(text: str, min_novelty: float = 0.40) -> bool:
        """Check if a thought is sufficiently novel vs recent thoughts.

        Returns True if the thought should be shown (novel enough).
        Side effect: appends the thought's words to the tracker if novel.
        """
        words = set(text.lower().split())
        if not words:
            return False
        for recent in _recent_thought_words:
            union = len(words | recent)
            if union and len(words & recent) / union >= (1.0 - min_novelty):
                return False  # Too similar to a recent thought
        _recent_thought_words.append(words)
        return True

    # Mutable-container aliases — safe because in-place mutation is shared.
    # State variables (pending_proposal, pending_action_followup, etc.) use
    # ctrl.X directly to prevent local/controller divergence.
    pending_next_actions = ctrl.pending_next_actions
    processed_cli_inputs = ctrl.processed_cli_inputs
    recent_outcomes = ctrl.recent_outcomes
    max_recent_outcomes = ctrl.max_recent_outcomes
    agent_states = ctrl.agent_states
    last_surfaced_tools = ctrl.last_surfaced_tools
    pending_prefetch = ctrl.pending_prefetch
    llm_submit_interval = ctrl.llm_submit_interval

    # Consecutive same-tool cap — prevents respond loops (refinement plan 1.5e)
    # Content-aware: tracks (tool_name, content_hash) so same tool with
    # different params (e.g. send_message with varied text) is NOT capped.
    _consecutive_same_tool: str = ""
    _consecutive_same_tool_count: int = 0
    _consecutive_same_content_hash: int = 0  # hash of params for content-aware cap
    _MAX_CONSECUTIVE_SAME_TOOL: int = 5

    def _get_all_tools() -> set[str]:
        return ctrl.get_all_tools()

    # Loop timing
    target_period = 1.0 / target_hz
    max_steps_i = int(max_steps or 0)
    step_iter = itertools.count() if max_steps_i <= 0 else range(max_steps_i)

    # Default Network lifecycle — managed by controller
    dn_enabled = ctrl.dn_enabled
    if dn_enabled:
        if not ctrl.dn_ctrl.start():
            dn_enabled = False
            ctrl.dn_enabled = False

    # Extract NAc reference for causal learning (passed to _record_outcome)
    _loop_nac = getattr(memory_hub, "nac", None) if memory_hub is not None else None

    # P4 multi-agent attribution: per-agent stash key.  Producer
    # (MemoryHub.on_percept_received) writes substrate nodes keyed by
    # the hub's owning agent_id; the consumer here must use the same
    # key or the stash leaks (consumer never finds the producer's
    # write).  Prefer memory_hub.agent_id (canonical per-agent
    # identifier from AgentFactory.create_agent) and fall back to
    # the loop's filesystem-safe agent_name for raw-loop callers
    # that don't construct a MemoryHub.
    _loop_agent_id: str = (getattr(memory_hub, "agent_id", None) if memory_hub is not None else None) or agent_name

    # Phase 1 (substrate_learns_from_experience.md): outside substrate-primary the
    # LLM issues a broad always-succeed action stream, so the tool-success floor in
    # record_outcome would flood the interoception cluster with "this tool ran".
    # Credit the cluster surface from the body's real drive signal ONLY.
    _drive_relief_only: bool = aut_mode != "substrate-primary"
    # Bind the flag once so every outcome site inherits it (no per-call threading).
    _rec_outcome = functools.partial(_record_outcome, drive_relief_only=_drive_relief_only)

    # Phase 0 sensor encoder — built once per loop when substrate-primary
    # is active and EC is reachable through memory_hub. Without this,
    # substrate-primary bypasses the LinguisticEncoder text path and EC
    # node_count stays at zero forever (which is what blocked the Phase 0
    # smoke run from being a measurement). See
    # docs/plans/grounded_language_acquisition.md Phase 0 + the
    # SensorEncoder docstring in similarity/encoder.py.
    # Built in ALL modes (Phase 1, substrate_learns_from_experience.md), not just
    # substrate-primary: llm-primary / real-hardware actions also encode the
    # current interoception cluster at outcome time (section 4) so their real
    # drive-relief outcomes reinforce the cluster-reward substrate. Harmless when
    # unused (an unembodied chat agent never calls encode); cheap to construct.
    _loop_sensor_encoder: Any | None = None
    if memory_hub is not None:
        _ec = getattr(memory_hub, "ec", None)
        if _ec is not None:
            try:
                from maxim.similarity.encoder import SensorEncoder

                _loop_sensor_encoder = SensorEncoder(
                    ec=_ec,
                    atl=getattr(memory_hub, "atl", None),
                    nac=_loop_nac,
                )
            except Exception:
                logger.debug("substrate-primary: SensorEncoder init failed", exc_info=True)

    # Initialize bio-system session (MemoryHub + hippocampus capture worker)
    memory_hub_enabled = _start_bio_session(memory_hub=memory_hub, hippocampus=hippocampus)

    # Diagnostic heartbeat: log once per agent on first iteration + every
    # ~10s thereafter so we can see if a loop is alive but stuck. Silent
    # unless sim mode is active.
    _last_heartbeat_time = [0.0]
    _loop_name = _safe_agent_name(agent)
    _consecutive_llm_fallbacks = 0  # Track consecutive LLM failures for stall detection
    _LLM_STALL_THRESHOLD = 1  # Surface immediately — only one fallback per LLM request
    # D13 planning liveness: set when the bounded retry budget is spent; the
    # loop breaks through NORMAL teardown (state persist + bio session end)
    # and raises PlanningLivenessExhausted afterwards, so the sim aborts
    # loudly instead of idling forever on a dropped planning turn.
    _planning_liveness_exhausted = False
    # Consecutive idle ticks observing "submitted, nothing in flight, nothing
    # came back" — two are required before declaring a turn lost (see the
    # backstop's comment for the call-end/proposal-enqueue race).
    _lost_turn_observations = 0
    # ONE gate for all five failure sites, computed once so no site can drift
    # (pre-merge review, architecture lens S7: the substrate-primary exclusion
    # was previously only incidental — an aut_llm_worker IS constructed in
    # substrate-primary runs, so `if llm_worker:` does run there).
    # Substrate-primary proposals never flow through get_latest_proposal.
    _planning_liveness_on = (
        bool(planning_liveness)
        and aut_mode != "substrate-primary"
        and llm_worker is not None
        and _planning_liveness_enabled_via_env()
    )
    if planning_liveness and not _planning_liveness_on:
        logger.info(
            "planning liveness requested but inactive (aut_mode=%s, llm_worker=%s, env_opt_out=%s)",
            aut_mode,
            "yes" if llm_worker is not None else "no",
            "yes" if not _planning_liveness_enabled_via_env() else "no",
        )

    for step_num in step_iter:
        loop_start = time.time()

        # Loop-alive heartbeat in sim mode — shows which loops are
        # iterating and their current state. Fires at most every 10s.
        if sim.is_sim_mode:
            if step_num == 0:
                sim.log(
                    "PIPELINE",
                    f"Agent loop started: {_loop_name} target_hz={target_hz} "
                    f"llm_worker={'YES' if llm_worker else 'no'}",
                )
                _last_heartbeat_time[0] = loop_start
            elif loop_start - _last_heartbeat_time[0] >= 10.0:
                _hb_state = (
                    f"ctrl.pending_proposal={'yes' if ctrl.pending_proposal else 'no'} "
                    f"pending_plan={'yes' if ctrl.pending_plan_proposal else 'no'} "
                    f"autonomy={autonomy_controller.current_level.value} "
                    f"paused={autonomy_controller.is_paused}"
                )
                sim.log("PIPELINE", f"Heartbeat step={step_num} {_hb_state}")
                _last_heartbeat_time[0] = loop_start

        # Log loop iteration (DEBUG level - only shown at verbosity 3)
        log_agentic(
            "agent_loop",
            "loop_iteration",
            {"step": step_num, "autonomy": autonomy_controller.current_level.value},
            level="DEBUG",
        )

        # ─────────────────────────────────────────────────────────────────
        # 0. CHECK STOP CONDITIONS
        # ─────────────────────────────────────────────────────────────────
        try:
            if stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set():
                log_agentic("agent_loop", "shutdown", {"reason": "stop_event"})
                break
        except (AttributeError, RuntimeError):
            pass

        # Check for shutdown mode - break immediately to stop LLM worker promptly
        current_mode = state.data.get("mode", "")
        if current_mode == "shutdown":
            log_agentic("agent_loop", "shutdown", {"reason": "shutdown_mode"})
            break

        # Configure Default Network for current mode
        if current_mode:
            ctrl.configure_dn_for_mode(current_mode)

        # Check if autonomy is paused
        if autonomy_controller.is_paused:
            time.sleep(idle_sleep_s)
            continue

        # ─────────────────────────────────────────────────────────────────
        # 0.45 EMBODIMENT DRIFT TICK (llm-primary)
        # ─────────────────────────────────────────────────────────────────
        # Advance the body's wall-clock drive drift every live iteration so a
        # Reachy body does not freeze through pure-thinking turns / idle gates
        # / LLM latency. Placed BEFORE the 0.6 idle gate (which ``continue``s
        # on no stimulus) so a *sitting* robot still gets cold/hungry, and
        # AFTER the pause check so an operator-paused agent stays frozen.
        # No-op on substrate-primary (it ticks itself) and when unembodied.
        tick_embodiment_drift(executor, aut_mode)

        # Expire a temporary agent display escalation back to the user's
        # floor (DisplayModeTool's documented auto-revert). Also the
        # production producer of the EVENT seam's display/revert event.
        _maybe_auto_revert_display()

        # 0.5 CHECK PERCEPT SOURCE EXHAUSTION (simulation mode)
        if sim.check_exhaustion(ctrl.pending_proposal):
            break

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
        _has_pending_input = bool(state.data.get("pending_cli_input") or state.data.get("pending_voice_input"))
        _has_pending_work = bool(
            ctrl.pending_proposal or ctrl.pending_action_followup or pending_next_actions or ctrl.pending_plan_proposal
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
        # If we submitted to the LLM recently, we're awaiting a proposal —
        # don't idle-gate or we'll never pick up the result.
        #
        # D13: the 120s literal is a wall-clock GUESS that a big model
        # routinely exceeds (qwen2.5-32b ~140s/turn in the Exp 37
        # heartbeat). Pre-fix, the window lapsing mid-call made the loop
        # idle straight past section 2's proposal poll, so the answer to a
        # slow call was never consumed — a livelock indistinguishable from
        # the lost-turn one. An observed in-flight call now holds the gate
        # open no matter what the clock says.
        _submitted_recently = (
            llm_worker is not None
            and ctrl.pending_proposal is None
            and (time.time() - ctrl.last_llm_submit_time) < 120.0
        )
        _call_in_flight = (
            llm_worker is not None
            and ctrl.pending_proposal is None
            and not _submitted_recently
            and _planning_call_in_flight()
        )
        _awaiting_llm = _submitted_recently or _call_in_flight

        if not (
            _has_pending_input
            or _has_pending_work
            or _has_sim_percept
            or _has_carried_percept
            or _is_first_step
            or _awaiting_llm
        ):
            # D13 planning-liveness backstop: the loop is about to idle, but
            # the last planning submit never produced ANY proposal — the
            # await window expired silently (lost turn: worker died, stale
            # drop, queue race). Reaching here means no call is in flight:
            # _awaiting_llm above stays True for as long as the registry
            # observes a live call, so a merely SLOW model is never mistaken
            # for a lost turn, and a call in flight but wedged remains the
            # D12 hard-abort's job (byte silence).
            #
            # Two consecutive observations are required because "call ended"
            # and "proposal reached the completed queue" are not simultaneous:
            # the router deregisters the call in its `finally`, then the
            # worker parses the response and builds the LLMProposal. Firing
            # inside that gap would requeue a turn whose good answer arrives
            # milliseconds later — executing the SAME planning turn twice,
            # and only on calls that already exceeded 120s, i.e. exactly the
            # big-model case this fix protects (pre-merge review, executor
            # lens F5). One extra idle tick costs 50ms and closes it.
            if (
                _planning_liveness_on
                and not ctrl.planning_exhausted
                and ctrl.last_llm_submit_time > 0
                and ctrl.last_proposal_time < ctrl.last_llm_submit_time
            ):
                _lost_turn_observations += 1
                if _lost_turn_observations >= 2:
                    _lost_turn_observations = 0
                    if _handle_planning_failure(
                        ctrl,
                        llm_worker,
                        sim,
                        reason="await_window_expired",
                        original_request=None,
                    ):
                        _planning_liveness_exhausted = True
                        break
            else:
                _lost_turn_observations = 0
            time.sleep(idle_sleep_s)
            continue

        # ─────────────────────────────────────────────────────────────────
        # 1. PERCEPTION — via SimulationAdapter or environment
        # ─────────────────────────────────────────────────────────────────
        observation = sim.next_observation(environment, default_network)
        state.update(observation)

        # ─────────────────────────────────────────────────────────────────
        # 1.1 IMAGINATION — extract novel entities from percept text
        # ─────────────────────────────────────────────────────────────────
        # Post-state.update hook: scan percept text for novel entity
        # phrases, check ComponentIndex for existing matches, and if truly
        # novel, dispatch to ImaginationDesigner for real-time SEM entity
        # generation. Gates: DN arousal + energy budget (checked inside trigger).
        _imagination_results: list = []  # ImaginationResult list for enrichment context
        if imagination_trigger is not None:
            try:
                percept_text = ""
                if hasattr(observation, "get"):
                    percept_text = str(
                        observation.get("transcript")
                        or observation.get("raw_transcript_text")
                        or observation.get("cli_input")
                        or ""
                    )
                elif hasattr(observation, "transcript"):
                    percept_text = str(
                        getattr(observation, "transcript", "") or getattr(observation, "cli_input", "") or ""
                    )
                if percept_text:
                    scene_id = state.data.get("current_scene_id") if hasattr(state, "data") else None
                    scene_ctx = state.data.get("scene_context") if hasattr(state, "data") else None
                    _imagination_results = imagination_trigger.process_percept(
                        percept_text, scene_context=scene_ctx, scene_id=scene_id
                    )
                else:
                    try:
                        from maxim.simulation.sim_logger import sim_log

                        _obs_keys = (
                            list(observation.keys()) if hasattr(observation, "keys") else type(observation).__name__
                        )
                        sim_log(
                            "SEM_TRACE",
                            f"Imagination skipped: no percept_text (obs keys: {_obs_keys})",
                            _force_debug=True,
                        )
                    except Exception:
                        log_swallowed_exception()
            except Exception as e:
                log_swallowed_exception(e, operation="imagination_trigger", context={"step": step_num})

        _auto_sense_text = ""  # populated by section 1.15, set on context at submission
        _audio_escalate_this_tick = False  # §1.16: a salient audio percept forces a submission (B1)

        # ─────────────────────────────────────────────────────────────────
        # 1.15 AUTO-SENSE — passive perception (exteroception + interoception)
        # ─────────────────────────────────────────────────────────────────
        # On each new percept (not empty ticks), auto-run sense_presence
        # (what's around me?) and sense on self-entity (how do I feel?).
        # Results are injected into StructuredContext so the LLM sees them
        # alongside the narrative — the agent passively perceives its
        # surroundings and body state without choosing to call tools.
        # Check if there's a new percept this tick (reuse observation parsing)
        _has_new_percept = False
        if hasattr(observation, "get"):
            _has_new_percept = bool(
                observation.get("transcript") or observation.get("raw_transcript_text") or observation.get("cli_input")
            )
        elif hasattr(observation, "transcript"):
            _has_new_percept = bool(getattr(observation, "transcript", ""))
        if _has_new_percept and executor is not None:
            try:
                _tool_reg = getattr(executor, "_registry", None) or getattr(executor, "registry", None)
                if _tool_reg is not None:
                    _auto_sense_parts = []

                    # Exteroception: dispatch every auto_fire tool with no
                    # arguments. Pre-W1 this hardcoded ``sense_presence``;
                    # the declarative ``auto_fire=True`` metadata on
                    # ``SensePresenceTool`` (and any future auto-discovery
                    # tool) drives this loop now. The
                    # ``get_auto_fire_tools()`` helper preserves the
                    # bypass invariant: results are injected into the
                    # next prompt as passive perception, never logged to
                    # ``actions.jsonl``. See
                    # [docs/plans/deferred/sense_tool_registry.md] § "Phase 2".
                    _presence_tool = None  # canonical sense_presence instance, for entity_map handoff
                    _auto_fire_tools = []
                    try:
                        _auto_fire_tools = _tool_reg.get_auto_fire_tools()
                    except AttributeError:
                        # Older registries without the helper — fall back
                        # to the legacy by-name lookup so out-of-tree
                        # ToolRegistry subclasses keep working.
                        try:
                            _auto_fire_tools = [_tool_reg.get("sense_presence")]
                        except KeyError:
                            _auto_fire_tools = []
                    for _af_tool in _auto_fire_tools:
                        if _af_tool is None:
                            continue
                        try:
                            _af_result = _af_tool.execute()
                            if _af_result.success and _af_result.output:
                                _auto_sense_parts.append(str(_af_result.output))
                        except Exception as _exc:
                            log_swallowed_exception(
                                _exc,
                                operation=f"auto_fire:{_af_tool.name}",
                                context={"step": step_num},
                            )
                        # Capture an entity-map source so the
                        # interoception block below can reuse the same
                        # entity tree the auto-fire scan saw. Pre-fold,
                        # this keyed on the literal name "sense_presence"
                        # — re-introducing the implicit-by-name coupling
                        # Phase 2 set out to retire. Capturing by
                        # attribute presence (any auto-fire tool that
                        # exposes ``_entity_map``) keeps the loop name-
                        # agnostic: a future auto-discovery tool that
                        # carries an entity map participates without
                        # needing a hardcoded branch here.
                        if _presence_tool is None and hasattr(_af_tool, "_entity_map"):
                            _presence_tool = _af_tool

                    # Interoception: sense self-entity (health, stamina,
                    # hunger). This is NOT auto_fire today — it's a
                    # special dispatch path that needs to be called
                    # once per self-entity with the entity name. The
                    # underlying ``sense`` tool stays LLM-callable
                    # (``kind="core-universal"``, ``auto_fire=False``)
                    # so the agent can also invoke it explicitly.
                    # Deferring the metadata-fication of this loop to
                    # 1.1+ (multi-arg auto_fire is out of MVP scope).
                    _sense = None
                    try:
                        _sense = _tool_reg.get("sense")
                    except KeyError:
                        pass
                    if _sense is not None and _presence_tool is not None:
                        try:
                            _emap = getattr(_presence_tool, "_entity_map", None)
                            if _emap is not None:
                                _self_ents = _emap.list_self_entities()
                                for _se in _self_ents:
                                    _sense_result = _sense.execute(entity_name=_se.name)
                                    if _sense_result.success and _sense_result.output:
                                        _auto_sense_parts.append(f"Body state ({_se.name}): {_sense_result.output}")
                        except Exception as _exc:
                            log_swallowed_exception(
                                _exc,
                                operation="auto_sense_self",
                                context={"step": step_num},
                            )
                    elif _sense is not None and _presence_tool is None and _auto_fire_tools:
                        # Observable signal that interoception was
                        # skipped despite ``sense`` being registered.
                        # Avoids the silent-no-op that the architecture
                        # review flagged if a future auto-fire tool
                        # roster doesn't expose an entity_map.
                        logger.debug(
                            "auto-sense interoception skipped: no auto-fire tool exposes _entity_map "
                            "(sense tool present but cannot be dispatched per-entity)",
                        )

                    if _auto_sense_parts:
                        _auto_sense_text = "\n".join(_auto_sense_parts)

                        try:
                            from maxim.simulation.sim_logger import sim_log

                            _n_entities = _auto_sense_text.count("[SCENE]") + _auto_sense_text.count("[YOU]")
                            sim_log("PERCEPTION", f"auto-sense: {_n_entities} entities, body state updated")
                        except Exception:
                            log_swallowed_exception()
            except Exception as _ase:
                log_swallowed_exception(_ase, operation="auto_sense", context={"step": step_num})

        # ─────────────────────────────────────────────────────────────────
        # 1.16 AUDIO ORIENTATION — exteroceptive sound-direction (thalamic relay)
        # ─────────────────────────────────────────────────────────────────
        # First consumer of the modality-preserving side-channel
        # (``sim.current_percept``): when this tick's percept is an audio/DoA
        # percept, fold a passive azimuth observation into the auto-sense
        # channel. A SALIENT audio percept ESCALATES to a submission — the
        # thalamic gate: a sub-threshold sound is perceived-but-ignored, an
        # above-threshold one reaches the LLM. Escalation sets
        # ``_audio_escalate_this_tick`` so the has_meaningful_input gate below
        # does NOT discard the audio-only tick (B1 fix — without this the line
        # was folded, logged, and thrown away before the model ever saw it).
        # GATE (re-gated in Stage 3 of live_audio_orient_wiring.md): the real
        # condition was always "a modality-preserving percept is present this
        # tick" — the old ``sim.is_sim_mode`` check was its proxy, and kept the
        # live path dark. Both adapters now carry ``current_percept``
        # (SimulationAdapter from its percept_source; NullSimulationAdapter
        # from a producer's ``carry_percept`` — the Stage-2 DoA feed), so the
        # gate reads the side-channel directly. Production ticks with no
        # carried percept cost one property read (None → skip, N1 preserved).
        # substrate-primary stays excluded: the drive/EC path reads the sensor
        # directly and §1.16 would double-write (S1).
        if getattr(sim, "current_percept", None) is not None and aut_mode != "substrate-primary":
            try:
                from maxim.embodiment.audio_localization import (
                    audio_attention_profile,
                    format_audio_orientation,
                    is_audio_escalation,
                    is_orienting_reflex,
                    reflex_oriented_azimuth,
                    resolve_orienting_profile,
                    should_emit_orientation,
                    world_set_azimuth,
                )

                _ap = getattr(sim, "current_percept", None)
                _az = None
                if _ap is not None:
                    _ameta = getattr(_ap, "metadata", None) or {}
                    _az = _ameta.get("azimuth")
                if _az is not None:
                    _sal = getattr(_ap, "salience", 0.0)
                    _nov = getattr(_ap, "novelty", 0.0)
                    _emb = getattr(executor, "embodiment", None)
                    # Per-entity reactivity + orient limits (data-driven; default
                    # profile when the body declares no `orienting:` config).
                    _oprofile = resolve_orienting_profile(_emb)
                    # World-set the body's azimuth sensor on ANY audio percept
                    # (before the tier gate) so `listen` can read the current
                    # sound direction — the agent can attend even to a
                    # sub-threshold sound it chose to notice. Capability-gated +
                    # fail-soft: bodies without an `azimuth` sensor are
                    # unaffected. Sim mirror of live DoA → azimuth (Track 2 L2).
                    # Skipped when a LIVE measurement stream owns the sensor
                    # (#508 review fold): on live, the DoA feed already wrote a
                    # fresher value than this percept echo, and the anonymous
                    # write would be refused by world_set_axis's ownership
                    # guard anyway — skipping here keeps routine live audio
                    # from opening every session with the refusal WARNING.
                    if _emb is not None and "azimuth" not in (getattr(_emb, "live_world_set_sensors", None) or ()):
                        world_set_azimuth(_emb, _az)

                    _trace = audio_attention_profile(_sal, _nov)
                    # Reflex tier is SIM-ONLY (pre-merge review fold): its
                    # world_set models a turn the body then "has made" — on
                    # the live path no motor was dispatched, so the modeled
                    # oriented azimuth would be a fabricated measurement (the
                    # head-frame lesson's failure class). Live reflex-speed
                    # orienting is Stage 5's DN behavior, with real motion.
                    _reflex = sim.is_sim_mode and _emb is not None and is_orienting_reflex(_sal, _nov, _oprofile)
                    _escalates = is_audio_escalation(_sal, _oprofile)

                    if _reflex:
                        # REFLEX tier: loud AND sudden → AUTOMATIC orient toward
                        # the sound (superior-colliculus startle), bypassing LLM
                        # deliberation. Model the turn by moving the azimuth
                        # toward center, clamped to the body's physical reach
                        # (max_orient_azimuth). The agent becomes aware AFTER — a
                        # delivered post-reflex notice.
                        _oriented = reflex_oriented_azimuth(_az, _oprofile)
                        world_set_azimuth(_emb, _oriented)
                        state.data["_last_audio_orient_az"] = _oriented
                        _reflex_line = (
                            f"A loud, sudden sound made you orient toward it (it was at azimuth {float(_az):+.2f})."
                        )
                        _auto_sense_text = f"{_auto_sense_text}\n{_reflex_line}" if _auto_sense_text else _reflex_line
                        _audio_escalate_this_tick = True
                        _trace["reflex"] = True
                        _trace["escalated"] = True
                        try:
                            from maxim.simulation.sim_logger import sim_log

                            sim_log(
                                "REACTION",
                                f"orienting reflex: turned toward a loud, sudden sound "
                                f"(was {float(_az):+.2f}, now {_oriented:+.2f})",
                                data=_trace,
                            )
                        except Exception:
                            log_swallowed_exception()
                    elif should_emit_orientation(state.data.get("_last_audio_orient_az"), _az):
                        # DELIBERATIVE tier: the agent CHOOSES to attend. Change-
                        # gate skips an unchanged direction (prompt noise — the
                        # first live run re-announced the same direction ~every 2s).
                        _audio_line = format_audio_orientation(_ap)
                        if _audio_line:
                            _auto_sense_text = f"{_auto_sense_text}\n{_audio_line}" if _auto_sense_text else _audio_line
                            # Advance the change-gate on DELIVERY — the line
                            # was folded into auto_sense, so an unchanged
                            # direction must not re-announce next tick. Store
                            # the CLAMPED value (N2) so an out-of-range
                            # reading can't spoof the delta gate. (Pre-fold
                            # this advance was nested under _escalates, so
                            # sub-threshold percepts — the DEFAULT 0.5/0.3
                            # weights, i.e. every live DoA percept — re-folded
                            # the identical direction every fresh reading:
                            # exactly the prompt noise this gate exists to
                            # prevent.)
                            state.data["_last_audio_orient_az"] = max(-1.0, min(1.0, float(_az)))
                            if _escalates:
                                _audio_escalate_this_tick = True
                            _trace["reflex"] = False
                            _trace["escalated"] = _escalates
                            try:
                                from maxim.simulation.sim_logger import sim_log

                                sim_log("PERCEPTION", f"audio-orient: {_audio_line}", data=_trace)
                            except Exception:
                                log_swallowed_exception()
            except Exception as _aoe:
                log_swallowed_exception(_aoe, operation="audio_orientation", context={"step": step_num})

        # ─────────────────────────────────────────────────────────────────
        # 1.2 BIO-ENRICHMENT — PFC deliberation (gate + enrich)
        # ─────────────────────────────────────────────────────────────────
        # Phase 1 of the PFC cycle: extract percept text and run
        # ThoughtGate + BioEnrichment.  The multi-cycle deliberation
        # (LLM calls) happens later at the submission point (section 2)
        # where StructuredContext is available.
        _percept_enrichment_text = ""
        _percept_text_for_cycle = ""
        _pfc_gate_passed = False
        _wms = None
        if bio_enrichment_pipeline is not None or thought_gate is not None:
            try:
                if hasattr(observation, "get"):
                    _percept_text_for_cycle = str(
                        observation.get("transcript")
                        or observation.get("raw_transcript_text")
                        or observation.get("cli_input")
                        or ""
                    )
                elif hasattr(observation, "transcript"):
                    _percept_text_for_cycle = str(
                        getattr(observation, "transcript", "") or getattr(observation, "cli_input", "") or ""
                    )
                # Defaults so the no-gate path (and an exception) still log
                # coherent numbers rather than tripping a NameError.
                _gate_score = 0.0
                _gate_threshold = 0.0
                _gate_reason = ""
                if _percept_text_for_cycle and thought_gate is not None:
                    _wms = None
                    _exec = getattr(agent, "exec_agent", None)
                    if _exec is not None:
                        _wms = getattr(_exec, "working_memory", None)
                    # Use step_num (loop iteration counter) for refractory,
                    # not WMS current_tick (memory entry counter — stays at 0
                    # until entries are added, causing perpetual refractory).
                    try:
                        _goal_bias = 0.0
                        if _loop_nac is not None:
                            _active_goal = state.data.get("active_goal") if hasattr(state, "data") else None
                            _goal_bias = _loop_nac.get_goal_reward_bias(_active_goal)
                        _gate_decision = thought_gate.should_think(
                            working_memory=_wms,
                            current_tick=step_num,
                            goal_reward_bias=_goal_bias,
                        )
                        _pfc_gate_passed = _gate_decision.passed
                        # Keep the REAL numbers + reason: the rejection log
                        # below used to hardcode 0.0/0.0, which printed
                        # "score=0.00 < 0.00" for EVERY rejection — including
                        # refractory, energy-exhausted and empty-working-memory,
                        # none of which are threshold comparisons at all. That
                        # made a live console session conclude the gate was
                        # scoring zero when the number had never been measured.
                        _gate_score = float(getattr(getattr(_gate_decision, "score", None), "combined", 0.0) or 0.0)
                        _gate_threshold = float(getattr(_gate_decision, "threshold_used", 0.0) or 0.0)
                        _gate_reason = str(getattr(_gate_decision, "reason", "") or "")
                    except Exception as _ge:
                        log_swallowed_exception(_ge, operation="thought_gate", context={"step": step_num})
                        # Otherwise the rejection log falls back to the
                        # fabricated "0.00 < 0.00" this commit exists to kill.
                        _gate_reason = f"gate error: {type(_ge).__name__}"
                elif _percept_text_for_cycle:
                    _pfc_gate_passed = bio_enrichment_pipeline is not None
                    # Resolve WMS for the no-gate path (needed by cycles 2+)
                    _exec = getattr(agent, "exec_agent", None)
                    if _exec is not None:
                        _wms = getattr(_exec, "working_memory", None)

                # Gate-fired minimum: always enrich cycle 1 when gate passes
                if _pfc_gate_passed and bio_enrichment_pipeline is not None and _percept_text_for_cycle:
                    from maxim.integration.bio_enrichment import EnrichmentContext

                    # Collect entity refs resolved by ImaginationTrigger this tick
                    _resolved_entity_refs = (
                        tuple(r.ref for r in _imagination_results if r.ref) if _imagination_results else ()
                    )
                    _enrich_ctx = EnrichmentContext(
                        active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
                        resolved_entities=_resolved_entity_refs,
                    )
                    _enrich_result = bio_enrichment_pipeline.enrich(
                        _percept_text_for_cycle, context=_enrich_ctx, bypass_gate=True
                    )
                    if _enrich_result is not None:
                        _percept_enrichment_text = bio_enrichment_pipeline.format_thought_response(_enrich_result)
                        _n_sections = sum(
                            1
                            for _f in (
                                _enrich_result.memories,
                                _enrich_result.predictions,
                                _enrich_result.concepts,
                                _enrich_result.affordances,
                                _enrich_result.recent_context,
                            )
                            if _f
                        )
                        from maxim.simulation.sim_logger import sim_log, sim_pre_deliberation, sim_deliberation_update

                        # Real numbers here too — a PASS logged as 0.00/0.00 is
                        # just as uninformative as a rejection was.
                        sim_pre_deliberation(
                            gate_passed=True,
                            score=_gate_score,
                            threshold=_gate_threshold,
                            enrichment_sections=_n_sections,
                        )
                        # Push cycle 1 enrichment to thinking panel
                        _c1_tags: list[str] = []
                        if _enrich_result.memories:
                            _c1_tags.append("hippocampus")
                        if _enrich_result.predictions:
                            _c1_tags.append("nac")
                        if _enrich_result.concepts:
                            _c1_tags.append("ec")
                        if _enrich_result.affordances:
                            _c1_tags.append("cerebellum")
                        if _enrich_result.recent_context:
                            _c1_tags.append("scn")
                        # Compute salience for cycle 1 THOUGHT
                        _n_memories_c1 = len(_enrich_result.memories) if _enrich_result.memories else 0
                        _salience_c1 = _compute_thought_salience(_n_sections, _n_memories_c1, 0.0)
                        _max_cyc_for_display = 3 if getattr(state, "data", {}).get("percept_source") else 2
                        # NOTE: don't push percept text to thinking panel here —
                        # the percept is an INPUT, not a thought. The AUT's actual
                        # reasoning will be pushed after the LLM responds (section 6).
                        sim_log(
                            "THOUGHT",
                            f"cycle 1: salience={_salience_c1:.2f}, sections={_n_sections}, memories={_n_memories_c1}",
                        )
                        # Add THOUGHT to working memory with computed salience
                        if _wms is not None and _percept_enrichment_text:
                            import uuid

                            from maxim.agents.working_memory import WorkingMemoryKind

                            _c1_goal = state.data.get("active_goal") if hasattr(state, "data") else None
                            _wms.add(
                                WorkingMemoryKind.THOUGHT,
                                ref=f"thought:{uuid.uuid4().hex[:8]}",
                                content={
                                    "source": "pfc_deliberation",
                                    "cycle": 1,
                                    "enrichment": _percept_enrichment_text[:500],
                                },
                                salience=_salience_c1,
                                goal_tag=_c1_goal,
                            )
                    else:
                        from maxim.simulation.sim_logger import sim_pre_deliberation

                        # The gate PASSED here — enrichment just produced no
                        # sections. Reusing _gate_reason printed "gate rejected
                        # (deliberation approved)", swapping one fabricated
                        # line for a self-contradictory one.
                        sim_pre_deliberation(
                            gate_passed=False,
                            score=_gate_score,
                            threshold=_gate_threshold,
                            enrichment_sections=0,
                            reason="enrichment produced no sections",
                        )
                        _pfc_gate_passed = False
                elif not _pfc_gate_passed and _percept_text_for_cycle:
                    # Only log gate rejection when there was actual percept text to evaluate
                    from maxim.simulation.sim_logger import sim_pre_deliberation

                    sim_pre_deliberation(
                        gate_passed=False,
                        score=_gate_score,
                        threshold=_gate_threshold,
                        enrichment_sections=0,
                        reason=_gate_reason,
                    )
            except Exception as e:
                log_swallowed_exception(e, operation="pfc_enrichment", context={"step": step_num})

        # Log cycle 1 enrichment outcome + reset refractory.
        # If multi-cycle deliberation runs at section 6, it logs the
        # final outcome separately and resets refractory again (idempotent).
        if _pfc_gate_passed and _percept_enrichment_text:
            from maxim.simulation.sim_logger import sim_contemplation

            # Real measured score — _gate_score is in scope here. Same class as
            # the pre_deliberation fabrication above: a logged 0.00 that was
            # never measured reads as a real reading to whoever debugs next.
            sim_contemplation(gate_passed=True, refined=False, score=_gate_score)
            if thought_gate is not None:
                thought_gate.reset_refractory(step_num)

        # Ensure maxim_runtime contains mode from state.data for MemoryAgent
        # This propagates mode set in CLI to PerceptionAgent -> MemoryAgent -> ExecAgent
        if "maxim_runtime" not in state.data:
            state.data["maxim_runtime"] = {}
        if isinstance(state.data.get("maxim_runtime"), dict):
            # Preserve existing mode if set, otherwise use state.data["mode"]
            if "mode" not in state.data["maxim_runtime"] and "mode" in state.data:
                state.data["maxim_runtime"]["mode"] = state.data["mode"]

        # Check for hard stops in observation
        transcript = None
        hard_override = None
        cli_input = None
        if hasattr(observation, "get"):
            transcript = observation.get("transcript") or observation.get("raw_transcript_text")
            hard_override = observation.get("hard_override")
            cli_input = observation.get("cli_input")
        elif hasattr(observation, "transcript"):
            transcript = getattr(observation, "transcript", None)
            hard_override = getattr(observation, "hard_override", None)
            cli_input = getattr(observation, "cli_input", None)

        # Also check state.data for CLI input injected from the interactive CLI thread
        # IMPORTANT: Always pop pending_cli_input to prevent duplicate processing
        # If observation already has cli_input, just discard the pending one
        pending_cli = state.data.pop("pending_cli_input", None)
        if not cli_input:
            cli_input = pending_cli

        # Check for voice input injected from the transcription thread
        # Voice input only gets forwarded when agentic mode is enabled (wake word was said)
        voice_input = state.data.pop("pending_voice_input", None)
        state.data.pop("pending_voice_transcript", None)
        is_agentic_voice_input = False

        # Use voice input if no CLI input (CLI takes priority)
        if not cli_input and voice_input:
            cli_input = voice_input
            is_agentic_voice_input = True  # Came from agentic voice session (wake word already said)
            logger.info("Using voice transcript as user input: %s", voice_input[:50] if voice_input else "")

        # Store CLI input in state and memory for LLM processing
        if cli_input:
            cli_text = str(cli_input).strip()
            source_type = "voice" if is_agentic_voice_input else "CLI"

            # ───────────────────────────────────────────────────────────────
            # PREEMPTION: New user/sim input interrupts followup chains
            # ───────────────────────────────────────────────────────────────
            # In simulation mode, the AUT can get stuck in bash→followup→bash
            # loops (e.g., "ls /tmp" → process output → "ls /tmp" again).
            # When a new percept arrives from the orchestrator during this
            # loop, we must cancel the pending followup so the new input
            # gets submitted to the LLM instead.
            if not cli_text.startswith("[ACTION_FOLLOWUP"):
                if ctrl.pending_action_followup:
                    _preempted_tool = ctrl.pending_action_followup.tool
                    ctrl.pending_action_followup = None
                    logger.info(
                        "Preempted followup chain (%s) for new input: %s",
                        _preempted_tool,
                        cli_text[:60],
                    )
                    sim.log("PIPELINE", f"Preempted {_preempted_tool} followup for new CLI input")
                if ctrl.pending_proposal and getattr(ctrl.pending_proposal, "strategy_used", None) in (
                    "multi_step",
                    "fallback",
                ):
                    _preempted_tool = (
                        ctrl.pending_proposal.action.get("tool_name", "?")
                        if isinstance(ctrl.pending_proposal.action, dict)
                        else "?"
                    )
                    logger.info(
                        "Preempted pending %s proposal for new input: %s",
                        _preempted_tool,
                        cli_text[:60],
                    )
                    sim.log("PIPELINE", f"Preempted pending {_preempted_tool} proposal for new CLI input")
                    ctrl.pending_proposal = None

            # ───────────────────────────────────────────────────────────────
            # CONFIRMATION / TIMEOUT / PLAN APPROVAL — delegated to controller
            # ───────────────────────────────────────────────────────────────
            if ctrl.handle_confirmation(cli_text):
                cli_input = None
                continue

            if ctrl.handle_timeout_retry(cli_text):
                cli_input = None
                continue

            # Store in state and memory (only reached if NOT a confirmation/timeout)
            ctrl.store_user_input(cli_text, source_type)

            # Speculative pre-fetching
            ctrl.run_prefetch(cli_text)
            pending_prefetch = ctrl.pending_prefetch

            # Planning mode: check if this input is approval/rejection/modify
            if ctrl.handle_plan_approval(cli_text):
                if ctrl.pending_proposal is None and ctrl.pending_plan_proposal is None:
                    # Plan was rejected — don't send to LLM
                    cli_input = None

        hard_stop_reason = check_hard_stop(transcript, hard_override)
        if hard_stop_reason:
            log_agentic(
                "agent_loop",
                "hard_stop",
                {"reason": hard_stop_reason},
                level="WARNING",
            )
            logger.warning(f"Hard stop triggered: {hard_stop_reason}")
            autonomy_controller.emergency_halt(hard_stop_reason)
            continue

        # ─────────────────────────────���───────────────────────────────────
        # 1.5 ADD PERCEPT TO CONTEXT POOL
        # ─────────────────────────────────────────────────────────────────
        if observation:
            # Build a percept-like object from observation for context pool
            try:
                from maxim.agents.bus import Percept

                percept = None
                if hasattr(observation, "get"):
                    # Mark as having maxim keyword if:
                    # 1. CLI input (always treated as for LLM - maxim prefix presumed), OR
                    # 2. Voice input contains "maxim"/"reachy", OR
                    # 3. This is agentic voice input (wake word was already said)
                    if cli_input and not is_agentic_voice_input:
                        # CLI input - always has keyword (presumed)
                        has_keyword = True
                    elif is_agentic_voice_input:
                        # Voice with wake word already said
                        has_keyword = True
                    else:
                        # Voice input - check for wake word
                        has_keyword = bool(
                            cli_input and ("maxim" in str(cli_input).lower() or "reachy" in str(cli_input).lower())
                        )
                    from maxim.agents.modality import SensoryModality, SensoryTag
                    from maxim.agents.percept_factory import make_text_percept

                    _obs_text = cli_input or transcript or ""
                    percept = make_text_percept(
                        str(_obs_text),
                        source=observation.get("source", "observation"),
                        sensory=SensoryTag(modality=SensoryModality.ABSTRACT, submodality="observation"),
                    )
                    percept.transcript_chunk = transcript
                    percept.cli_input = cli_input
                    percept.has_maxim_keyword = has_keyword
                elif isinstance(observation, Percept):
                    percept = observation

                if percept:
                    context_pool.add_percept(percept)
            except Exception as e:
                logger.debug(f"Failed to add percept to context pool: {e}")

        # ─────────────────────────────────────────────────────────────────
        # 2. CHECK FOR LLM PROPOSALS (non-blocking)
        # ─────────────────────────────────────────────────────────────────
        if llm_worker:
            new_proposal = llm_worker.get_latest_proposal()
            if new_proposal is not None:
                # D13 planning liveness: ANY proposal (even one dropped below
                # as stale/fallback) proves the worker answered this submit —
                # the await-expiry backstop keys on last_proposal_time <
                # last_llm_submit_time.
                ctrl.last_proposal_time = time.time()
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
                    f"Proposal consumed by {_loop_name}: tool={_tool_name} "
                    f"age={time.time() - new_proposal.timestamp:.2f}s",
                )
            if new_proposal:
                sim.log(
                    "EXEC",
                    f"Proposal received: tool={new_proposal.action.get('tool_name') if isinstance(new_proposal.action, dict) else None}",
                )

                # Staleness guard: discard proposals that sat in the result
                # queue too long. The worker's _stale_threshold (default 5s)
                # governs request freshness; this guards completed proposals.
                _STALE_PROPOSAL_AGE_S = 35.0
                proposal_age = time.time() - new_proposal.timestamp
                if proposal_age > _STALE_PROPOSAL_AGE_S:
                    logger.warning(
                        "Skipping stale LLM proposal (age=%.1fs, request_id=%s)",
                        proposal_age,
                        new_proposal.request_id,
                    )
                    sim.log("EXEC", f"DROPPED: stale proposal (age={proposal_age:.1f}s)")
                    _stale_original_request = new_proposal.original_request
                    new_proposal = None
                    # D13: a dropped stale proposal is a consumed planning
                    # turn with nothing executed — reschedule, don't idle.
                    if _planning_liveness_on and _handle_planning_failure(
                        ctrl,
                        llm_worker,
                        sim,
                        reason="stale_proposal_dropped",
                        original_request=_stale_original_request,
                    ):
                        _planning_liveness_exhausted = True
            # In simulation mode, skip fallback proposals — wait for real LLM
            if new_proposal and sim.should_skip_fallback_proposal(new_proposal):
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

        # D13: retry budget spent inside section 2 — leave through the normal
        # teardown path (state persist + bio session end), then raise.
        if _planning_liveness_exhausted:
            break

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

        # ─────────────────────────────────────────────────────────────────
        # 3. AGENT PROPOSE_INTENT FALLBACK (when no LLM worker or no pending proposal)
        # ─────────────────────────────────────────────────────────────────
        # If there's no pending LLM proposal, let the agent propose directly
        # This enables reactive behaviors like default tracking
        #
        # IMPORTANT: Skip agent fallback when there's pending user input for LLM
        # This prevents the novelty tracker from running while waiting for LLM response
        pending_user_input = state.data.get("pending_user_input", "")
        pending_input_time = state.data.get("pending_user_input_time", 0)
        pending_input_source = state.data.get("pending_user_input_source", "CLI")
        llm_response_timeout = 30.0  # Max seconds to wait for LLM before allowing fallback

        # CLI input always goes to LLM (maxim prefix is presumed)
        # Voice input requires "maxim" wake word to prevent accidental triggers
        if pending_input_source == "CLI":
            input_is_for_llm = True  # CLI input always processed by LLM
        else:
            # Voice input requires wake word
            input_is_for_llm = "maxim" in pending_user_input.lower() or "reachy" in pending_user_input.lower()

        has_pending_llm_input = bool(
            pending_user_input
            and input_is_for_llm
            and llm_worker is not None
            and (time.time() - pending_input_time) < llm_response_timeout  # Timeout after 30s
        )

        # Preemption hold check — skip goal proposal while holding
        if hasattr(agent, "goal") and hasattr(agent.goal, "check_hold"):
            if agent.goal.check_hold():
                continue  # Still in tonic hold — skip this cycle

        if ctrl.pending_proposal is None and not has_pending_llm_input:
            try:
                if hasattr(agent, "propose_intent"):
                    intent = agent.propose_intent(state, memory)
                    if isinstance(intent, dict) and intent:
                        # Log the intent proposal
                        log_agentic(
                            "agent_loop",
                            "intent_proposed",
                            {
                                "source": intent.get("source", "agent"),
                                "confidence": intent.get("confidence"),
                                "has_goal": bool(intent.get("goal") or intent.get("intent")),
                            },
                        )

                        goal = intent.get("goal") or intent.get("intent")
                        if isinstance(goal, dict) and goal.get("tool_name"):
                            # Convert goal dict to action format
                            action = {
                                "tool_name": goal["tool_name"],
                                "params": goal.get("params", {}),
                            }
                            confidence = float(intent.get("confidence", 0.5))

                            # Check if autonomy allows this action
                            can_execute, reason = autonomy_controller.can_execute_action(action, confidence=confidence)

                            # Log autonomy check
                            log_agentic(
                                "agent_loop",
                                "autonomy_check",
                                {
                                    "tool": action["tool_name"],
                                    "can_execute": can_execute,
                                    "reason": reason if not can_execute else None,
                                    "confidence": confidence,
                                },
                            )

                            if can_execute:
                                try:
                                    if callable(on_event):
                                        try:
                                            on_event(StreamEvent("tool_start", {"tool_name": action["tool_name"]}))
                                        except Exception as e:
                                            log_swallowed_exception(e, operation="on_event:tool_start")

                                    result = executor.execute(action)

                                    # Log tool execution
                                    success = getattr(result, "success", True)

                                    if callable(on_event):
                                        try:
                                            on_event(
                                                StreamEvent(
                                                    "tool_end", {"tool_name": action["tool_name"], "success": success}
                                                )
                                            )
                                        except Exception as e:
                                            log_swallowed_exception(e, operation="on_event:tool_end")
                                    log_agentic(
                                        "agent_loop",
                                        "tool_called",
                                        {
                                            "tool": action["tool_name"],
                                            "success": success,
                                            "source": intent.get("source", "agent_fallback"),
                                        },
                                    )

                                    # Log tool result details at verbose level
                                    output = getattr(result, "output", None)
                                    if output:
                                        log_agentic(
                                            "agent_loop",
                                            "tool_result",
                                            {
                                                "tool": action["tool_name"],
                                                "output": output if isinstance(output, dict) else str(output)[:100],
                                            },
                                        )

                                    autonomy_controller.log_action(
                                        action_type="executed",
                                        action=action,
                                        reasoning=intent.get("source", "agent_fallback"),
                                        mode=state.data.get("mode", "unknown"),
                                        confidence=confidence,
                                        outcome="success" if success else "failure",
                                        error=getattr(result, "error", None),
                                    )

                                    # Process result
                                    try:
                                        followup = environment.step(result)
                                        if followup:
                                            state.update(followup)
                                    except Exception as e:
                                        log_swallowed_exception(e, operation="environment.step_followup")

                                    # Store in memory
                                    try:
                                        memory.store_raw(
                                            content={
                                                "action": action,
                                                "reasoning": intent.get("source", "agent_fallback"),
                                                "result": getattr(result, "output", None),
                                                "success": getattr(result, "success", True),
                                            },
                                            metadata={"type": "agent_action"},
                                        )
                                    except Exception as e:
                                        log_swallowed_exception(e, operation="memory.store_raw")

                                    # Capture episodic memory to Hippocampus (async)
                                    if hippocampus is not None:
                                        _capture_episodic(
                                            hippocampus=hippocampus,
                                            executor=executor,
                                            observation=observation,
                                            state=state,
                                            intent=intent,
                                            action={
                                                "tool_name": action["tool_name"],
                                                "params": action.get("params", {}),
                                                "confidence": confidence,
                                            },
                                            result=result,
                                            run_id=run_id or "",
                                        )
                                        _bio_integration.observe_episode(
                                            hippocampus=hippocampus,
                                            agent_id=_loop_agent_id,
                                            channel="text",
                                            activated_nodes=(),
                                            after_tool_execution=True,
                                            salience_spike=_bio_integration.consume_pain_intensity(
                                                agent_id=_loop_agent_id
                                            ),
                                        )

                                except Exception as e:
                                    log_agentic(
                                        "agent_loop",
                                        "error",
                                        {"context": "agent_fallback_action", "error": str(e)},
                                        level="ERROR",
                                    )
                                    logger.debug(f"Agent fallback action failed: {e}")

                                    # Track exception in recent_outcomes for LLM learning
                                    _rec_outcome(
                                        agent_id=_loop_agent_id,
                                        tool_name=action["tool_name"],
                                        success=False,
                                        result_summary=None,
                                        error=str(e),
                                        reasoning=getattr(ctrl.pending_proposal, "reasoning", "")
                                        if ctrl.pending_proposal
                                        else "",
                                        recent_outcomes=recent_outcomes,
                                        max_recent=max_recent_outcomes,
                                        llm_worker=llm_worker,
                                        context_pool=context_pool,
                                        nac=_loop_nac,
                                        active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
                                        cluster_id=getattr(ctrl.pending_proposal, "cluster_id", None),
                                        clusters=getattr(ctrl.pending_proposal, "clusters", None),
                                    )
                            else:
                                # Log rejected action
                                log_agentic(
                                    "agent_loop",
                                    "action_rejected",
                                    {
                                        "tool": action["tool_name"],
                                        "reason": reason,
                                        "confidence": confidence,
                                    },
                                )
                    else:
                        # Log idle state at debug level
                        log_agentic("agent_loop", "idle", {"step": step_num}, level="DEBUG")
            except Exception as e:
                log_agentic(
                    "agent_loop",
                    "error",
                    {"context": "propose_intent", "error": str(e)},
                    level="ERROR",
                )
                logger.debug("Agent propose_intent failed: %s", e)

        # ─────────────────────────────────────────────────────────────────
        # 4. EXECUTE PENDING LLM ACTION (if autonomy allows)
        # ─────────────────────────────────────────────────────────────────
        if ctrl.pending_proposal and ctrl.pending_proposal.action:
            action = ctrl.pending_proposal.action
            confidence = ctrl.pending_proposal.confidence

            # Phase 1 (substrate_learns_from_experience.md): in llm-primary the LLM
            # chose the action, so propose_via_substrate never ran and no substrate
            # cluster was captured. Encode the current interoception (+audio) state
            # HERE — from the PRE-action drive state, the correct credit key — so the
            # real drive-relief outcome reinforces the cluster-reward substrate via
            # record_outcome (drive_relief_only → no tool-success floor). No-op in
            # substrate-primary (clusters already captured) and when unembodied.
            if (
                aut_mode != "substrate-primary"
                and _loop_sensor_encoder is not None
                and getattr(ctrl.pending_proposal, "clusters", None) is None
                and getattr(executor, "embodiment", None) is not None
            ):
                _live_clusters = _encode_current_clusters(_loop_sensor_encoder, _loop_agent_id, executor)
                if _live_clusters:
                    ctrl.pending_proposal = dataclasses.replace(
                        ctrl.pending_proposal,
                        cluster_id=_live_clusters.get(INTEROCEPTION_TAG),
                        clusters=_live_clusters,
                    )

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
                _other_tools = sorted(_get_all_tools() - {_this_tool})
                _cap_msg = (
                    f"SYSTEM: '{_this_tool}' was called {_consecutive_same_tool_count} times "
                    f"with identical parameters — blocked to prevent a loop. "
                    f"Try a DIFFERENT tool or different parameters. "
                    f"Available tools: {', '.join(_other_tools[:8]) if _other_tools else 'none'}"
                )
                recent_outcomes.append(
                    {
                        "tool": _this_tool,
                        "tool_name": _this_tool,
                        "success": False,
                        "result": _cap_msg,
                        "error": "consecutive_tool_cap",
                        "timestamp": time.time(),
                    }
                )
                if len(recent_outcomes) > max_recent_outcomes:
                    recent_outcomes.pop(0)

                ctrl.pending_proposal = None
                _consecutive_same_tool_count = 0
                _consecutive_same_tool = ""
                _consecutive_same_content_hash = 0
                continue

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
                )

                # Queue as a followup for the next LLM call
                ctrl.pending_action_followup = ActionFollowup(
                    tool="batched_exploration",
                    result=combined_results,
                    original_query=ctrl.pending_proposal.triggering_input,
                    followup_type="process",
                    mode=state.data.get("mode", "exploration"),
                    timestamp=time.time(),
                )
                logger.info("Batched exploration complete, queuing followup for LLM")

                # Clear proposal - will be handled via followup
                ctrl.pending_proposal = None
                continue  # Skip normal execution flow

            # ───────────────────────────────────────────────────────────────
            # PLANNING MODE: Check if this proposal requires user approval
            # ───────────────────────────────────────────────────────────────
            if getattr(ctrl.pending_proposal, "requires_approval", False) and ctrl.pending_proposal.plan_text:
                # In sim mode, auto-resolve plan approval via response policy
                sim_plan_response = sim.resolve_plan_approval(ctrl.pending_proposal.plan_text)
                if sim_plan_response is not None:
                    sim.log("PIPELINE", f"Auto-resolved plan approval: {sim_plan_response}")
                    if sim_plan_response.lower() in ("yes", "y"):
                        # Auto-approved — skip the approval flow, proceed to execution
                        logger.info("Sim mode: plan auto-approved, executing")
                    else:
                        # Auto-rejected
                        logger.info("Sim mode: plan auto-rejected")
                        ctrl.pending_proposal = None
                        continue
                else:
                    # Production mode: store and wait for real user
                    logger.info("Proposal requires approval, showing plan to user")
                    log_agentic(
                        "agent_loop",
                        "plan_awaiting_approval",
                        {
                            "tool": action.get("tool_name"),
                            "plan_preview": ctrl.pending_proposal.plan_text[:100]
                            if ctrl.pending_proposal.plan_text
                            else None,
                        },
                    )

                    state.data["pending_plan_text"] = ctrl.pending_proposal.plan_text
                    state.data["pending_plan_tool"] = action.get("tool_name")

                    ctrl.pending_plan_proposal = ctrl.pending_proposal
                    ctrl.pending_proposal = None
                    continue

            logger.info("Executing LLM proposal: tool=%s, confidence=%.2f", action.get("tool_name"), confidence)

            # Log LLM proposal received
            log_agentic(
                "agent_loop",
                "goal_proposed",
                {
                    "tool": action.get("tool_name"),
                    "confidence": confidence,
                    "reasoning": ctrl.pending_proposal.reasoning[:100] if ctrl.pending_proposal.reasoning else None,
                    "source": "llm_worker",
                },
            )

            can_execute, reason = autonomy_controller.can_execute_action(action, confidence=confidence)

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

            if can_execute:
                # Capture pre-execution snapshot for preemption reversal
                if hasattr(agent, "_execution_tracker") and agent._execution_tracker:
                    goal_desc = ctrl.pending_proposal.reasoning or ""
                    robot_handle = getattr(agent, "goal", None)
                    robot_handle = getattr(robot_handle, "robot", None) if robot_handle else None
                    agent._execution_tracker.capture_before(
                        goal_description=goal_desc[:200],
                        tool_name=action.get("tool_name", ""),
                        tool_params=action.get("params", {}),
                        robot=robot_handle,
                    )

                # Execute the action
                try:
                    exec_start = time.time()
                    logger.info("Starting tool execution: %s", action.get("tool_name"))
                    if sim.is_sim_mode:
                        sim.log(
                            "EXEC",
                            f"Executing: {action.get('tool_name')} "
                            f"by {_loop_name} params={list((action.get('params') or {}).keys())}",
                        )
                    result = executor.execute(action)
                    exec_elapsed = time.time() - exec_start
                    success = getattr(result, "success", True)
                    # An action that mechanically succeeded but harmed the body
                    # (self_effect breached a sensor's comfort band → SEM
                    # embodiment_failures) is a NEGATIVE learning outcome. Pass
                    # this through so record_outcome doesn't book a spurious
                    # positive that masks the aversion (substrate_primary_cradle_
                    # readiness.md B5). The ToolPainBridge owns the primary
                    # negative attribution; this prevents the competing positive.
                    _side = getattr(result, "side_effects", None)
                    _embodiment_failed = bool(_side and _side.get("embodiment_failures"))
                    # Motor-credit (GAP 1): the drive relief this action produced,
                    # if it touched a drive sensor (orient→azimuth, eat→hunger).
                    # record_outcome prefers this as the cluster-reward magnitude
                    # over the ±1 tool-success — the state-conditioned signal that
                    # lets substrate-primary selection learn "turn toward the
                    # sound." None/absent → ±1 fallback. See tool_side_effects.md.
                    _drive_potential_diff = _side.get("drive_potential_diff") if _side else None
                    # sem_motor_binding.md Phase 1: drive-touched-but-
                    # unmeasured (motor-bound live turn, credit deferred to
                    # the Phase 2 measured slice) — suppress the flat +1
                    # tool-success cluster floor for THIS action.
                    _drive_credit_withheld = bool(_side.get("drive_credit_withheld")) if _side else False
                    # Phase 2 (sem_motor_binding.md): measured exteroceptive
                    # relief routes to the direction-bearing cluster.
                    _drive_relief_channel = _side.get("drive_relief_channel") if _side else None
                    logger.info(
                        "Tool execution completed in %.2fs: %s, success=%s",
                        exec_elapsed,
                        action.get("tool_name"),
                        success,
                    )
                    if sim.is_sim_mode:
                        sim.log(
                            "EXEC",
                            f"Completed: {action.get('tool_name')} success={success} elapsed={exec_elapsed:.2f}s",
                        )

                    # Auto-recover: write_file failed because file exists → retry with overwrite
                    if (
                        not success
                        and action.get("tool_name") == "write_file"
                        and "already exists" in str(getattr(result, "error", "")).lower()
                    ):
                        raw_params = action.get("params")
                        safe_params = raw_params if isinstance(raw_params, dict) else {}
                        logger.info(
                            "Auto-recovery: retrying write_file with overwrite=True for %s",
                            safe_params.get("path", "?"),
                        )
                        retry_action = dict(action)
                        retry_params = dict(safe_params)
                        retry_params["overwrite"] = True
                        retry_action["params"] = retry_params
                        result = executor.execute(retry_action)
                        success = getattr(result, "success", True)
                        if success:
                            logger.info("Auto-recovery succeeded for write_file")
                        else:
                            logger.warning(
                                "Auto-recovery failed for write_file: %s",
                                getattr(result, "error", "unknown"),
                            )

                    # If this was a timeout retry prompt, store state for user response
                    if action.get("_timeout_retry") and success:
                        # Plan 3.5 R2: fall back to the current agent-level LLM
                        # timeout default if the action didn't include _timeout_s.
                        # Was hardcoded 60.0 pre-plan (mesh-era value).
                        from maxim.agents.llm_worker import DEFAULT_LLM_CALL_TIMEOUT_S

                        timeout_s = action.get("_timeout_s", DEFAULT_LLM_CALL_TIMEOUT_S)
                        # In sim mode, auto-resolve instead of blocking
                        sim_timeout_response = sim.resolve_timeout_retry(timeout_s)
                        if sim_timeout_response is not None:
                            sim.log("PIPELINE", f"Auto-resolved timeout retry: {sim_timeout_response}")
                            state.data["pending_timeout_retry"] = {
                                "original_request": action.get("_original_request"),
                                "timeout_s": timeout_s,
                            }
                            state.data["pending_cli_input"] = sim_timeout_response
                        else:
                            state.data["pending_timeout_retry"] = {
                                "original_request": action.get("_original_request"),
                                "timeout_s": timeout_s,
                            }

                    # Invalidate cache for write operations to ensure fresh reads
                    tool_name = action.get("tool_name", "")
                    if tool_name == "write_file" and success:
                        written_path = action.get("params", {}).get("path")
                        if written_path:
                            invalidated = result_cache.invalidate(path=written_path)
                            if invalidated > 0:
                                logger.debug("Invalidated %d cache entries for: %s", invalidated, written_path)

                    # Log tool execution
                    log_agentic(
                        "agent_loop",
                        "tool_called",
                        {
                            "tool": action.get("tool_name"),
                            "success": success,
                            "source": "llm_worker",
                        },
                    )

                    # Log tool result details
                    output = getattr(result, "output", None)
                    if output:
                        log_agentic(
                            "agent_loop",
                            "tool_result",
                            {
                                "tool": action.get("tool_name"),
                                "output": output if isinstance(output, dict) else str(output)[:100],
                            },
                        )

                    # Log to autonomy controller
                    autonomy_controller.log_action(
                        action_type="executed",
                        action=action,
                        reasoning=ctrl.pending_proposal.reasoning,
                        mode=state.data.get("mode", "unknown"),
                        confidence=confidence,
                        citations=ctrl.pending_proposal.citations,
                        outcome="success" if success else "failure",
                        error=getattr(result, "error", None),
                    )

                    # Track outcome for context pool and learning
                    # Get followup type to determine result storage and follow-up behavior
                    tool_name = action.get("tool_name", "")
                    current_mode = state.data.get("mode", "live")

                    # L2: Reset deliberation state when a non-think action fires.
                    if tool_name != "think":
                        _reset_deliberation(executor)

                    from maxim.modes.definitions import get_tool_followup_type

                    followup_type = get_tool_followup_type(tool_name, current_mode)

                    # Store more result for tools that need processing (up to 3000 chars)
                    needs_processing = followup_type in ("process", "respond", "engage")
                    result_limit = 3000 if needs_processing else 100
                    # Handle empty lists/dicts as valid output (use 'is not None' check)
                    if output is not None:
                        # Format search results in a more LLM-friendly way
                        if tool_name == "internet_search" and isinstance(output, list):
                            formatted_parts = []
                            for i, item in enumerate(output[:10], 1):  # Limit to 10 results
                                if isinstance(item, dict):
                                    title = item.get("title", "")
                                    url = item.get("url", "")
                                    snippet = item.get("snippet", "")
                                    formatted_parts.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}")
                            result_str = "\n\n".join(formatted_parts)[:result_limit]
                        else:
                            result_str = str(output)[:result_limit]
                        # For empty results, include metadata message if available
                        if not output and hasattr(result, "metadata"):
                            msg = result.metadata.get("message", "")
                            if msg:
                                result_str = f"[No results: {msg}]"
                    else:
                        # When output is None but tool returned an error,
                        # include the error text so followup re-thinks can
                        # see WHY the tool failed (e.g. "use send_message
                        # instead of respond").
                        error_msg = getattr(result, "error", None) if result else None
                        if error_msg:
                            result_str = f"[ERROR: {str(error_msg)[:result_limit]}]"
                        else:
                            result_str = None

                    _rec_outcome(
                        agent_id=_loop_agent_id,
                        tool_name=tool_name or "unknown",
                        success=success,
                        result_summary=result_str,
                        error=getattr(result, "error", None),
                        reasoning=getattr(ctrl.pending_proposal, "reasoning", "") if ctrl.pending_proposal else "",
                        recent_outcomes=recent_outcomes,
                        max_recent=max_recent_outcomes,
                        llm_worker=llm_worker,
                        context_pool=context_pool,
                        nac=_loop_nac,
                        active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
                        tool_params=action.get("params"),
                        cluster_id=getattr(ctrl.pending_proposal, "cluster_id", None),
                        clusters=getattr(ctrl.pending_proposal, "clusters", None),
                        embodiment_failed=_embodiment_failed,
                        drive_potential_diff=_drive_potential_diff,
                        drive_credit_withheld=_drive_credit_withheld,
                        drive_relief_channel=_drive_relief_channel,
                    )

                    # Record plan outcome in MemoryHub for learning. A plan that
                    # led to bodily harm is a NEGATIVE plan outcome even if the
                    # tool mechanically succeeded (B5) — otherwise the plan path
                    # books a positive CausalLink that competes with the tool's
                    # learned aversion (the PlanHistoryBridge records under the
                    # same tool event signature).
                    if memory_hub_enabled and memory_hub is not None:
                        _record_plan_outcome(
                            memory_hub=memory_hub,
                            goal=ctrl.pending_proposal.reasoning or "",
                            tool_name=tool_name,
                            success=success and not _embodiment_failed,
                        )

                    # If this tool has a followup_type, trigger a follow-up LLM cycle.
                    # The followup_type determines how the LLM should handle the results:
                    #   "process" - LLM processes results for next action (coding agent)
                    #   "respond" - LLM synthesizes results into user response
                    #   "engage"  - LLM responds AND offers proactive follow-ups
                    # "process" followups fire even on failure so the LLM can
                    # learn from the error and retry with a different tool
                    # (e.g. sim orchestrator's catch-all 'respond' rejects →
                    # LLM should immediately re-think, not stall for 60s).
                    # Note: Use 'is not None' to handle empty lists [] which are falsy but still valid output
                    should_followup = followup_type and ((success and output is not None) or followup_type == "process")
                    if should_followup:
                        triggering_input = getattr(ctrl.pending_proposal, "triggering_input", "")
                        ctrl.pending_action_followup = ActionFollowup(
                            tool=tool_name,
                            result=result_str,
                            original_query=triggering_input,
                            followup_type=followup_type,
                            mode=current_mode,
                            timestamp=time.time(),
                        )
                        logger.info(
                            "Tool %s completed with followup_type=%s, queuing follow-up", tool_name, followup_type
                        )

                    # Track conversation history for response/speak actions
                    tool_name = action.get("tool_name", "")
                    if tool_name in ("respond", "speak") and success:
                        raw_params = action.get("params")
                        params = raw_params if isinstance(raw_params, dict) else {}
                        response_message = params.get("message") or params.get("text", "")
                        triggering_input = getattr(ctrl.pending_proposal, "triggering_input", "")
                        if response_message and triggering_input:
                            context_pool.add_conversation_turn(
                                user_input=triggering_input,
                                assistant_response=response_message,
                                tool_used=tool_name,
                            )

                    # Process result
                    try:
                        followup = environment.step(result)
                        if followup:
                            state.update(followup)
                    except Exception as e:
                        log_swallowed_exception(e, operation="environment.step_followup")

                    # Store in memory
                    try:
                        memory.store_raw(
                            content={
                                "action": action,
                                "reasoning": ctrl.pending_proposal.reasoning,
                                "result": getattr(result, "output", None),
                                "success": getattr(result, "success", True),
                            },
                            metadata={"type": "action_execution"},
                        )
                    except Exception as e:
                        log_swallowed_exception(e, operation="memory.store_raw")

                    # Capture episodic memory to Hippocampus (async)
                    if hippocampus is not None:
                        _capture_episodic(
                            hippocampus=hippocampus,
                            executor=executor,
                            observation=observation,
                            state=state,
                            intent={"goal": ctrl.pending_proposal.reasoning, "source": "llm_worker"},
                            action={
                                "tool_name": action.get("tool_name"),
                                "params": action.get("params", {}),
                                "confidence": confidence,
                            },
                            result=result,
                            run_id=run_id or "",
                        )
                        _bio_integration.observe_episode(
                            hippocampus=hippocampus,
                            agent_id=_loop_agent_id,
                            channel="text",
                            activated_nodes=(),
                            after_tool_execution=True,
                            salience_spike=_bio_integration.consume_pain_intensity(agent_id=_loop_agent_id),
                        )

                    # Handle failure
                    if success is False:
                        log_agentic(
                            "agent_loop",
                            "goal_failed",
                            {
                                "tool": action.get("tool_name"),
                                "error": getattr(result, "error", None),
                            },
                            level="WARNING",
                        )
                        try:
                            state.mark_failure(getattr(result, "error", None))
                        except Exception as e:
                            log_swallowed_exception(e, operation="state.mark_failure")

                except Exception as e:
                    logger.error(f"Action execution failed: {e}")
                    autonomy_controller.log_action(
                        action_type="executed",
                        action=action,
                        reasoning=ctrl.pending_proposal.reasoning,
                        mode=state.data.get("mode", "unknown"),
                        confidence=confidence,
                        outcome="error",
                        error=str(e),
                    )

                    # Track exception in recent_outcomes for LLM learning
                    _rec_outcome(
                        agent_id=_loop_agent_id,
                        tool_name=action.get("tool_name", "unknown"),
                        success=False,
                        result_summary=None,
                        error=str(e),
                        reasoning=getattr(ctrl.pending_proposal, "reasoning", "") if ctrl.pending_proposal else "",
                        recent_outcomes=recent_outcomes,
                        max_recent=max_recent_outcomes,
                        llm_worker=llm_worker,
                        context_pool=context_pool,
                        nac=_loop_nac,
                        active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
                        cluster_id=getattr(ctrl.pending_proposal, "cluster_id", None),
                        clusters=getattr(ctrl.pending_proposal, "clusters", None),
                    )

                    # Mark failure in state
                    try:
                        state.mark_failure(str(e))
                    except Exception as mf_err:
                        log_swallowed_exception(mf_err, operation="state.mark_failure_exc")

                ctrl.pending_proposal = None

            elif autonomy_controller.current_level == AutonomyLevel.PLANNING:
                # Queue for human approval
                proposal = Proposal(
                    id=ctrl.pending_proposal.request_id,
                    action=action,
                    reasoning=ctrl.pending_proposal.reasoning,
                    confidence=confidence,
                    strategy_used=ctrl.pending_proposal.strategy_used,
                    citations=ctrl.pending_proposal.citations,
                )
                autonomy_controller.proposal_queue.submit(proposal)
                autonomy_controller.log_action(
                    action_type="proposed",
                    action=action,
                    reasoning=ctrl.pending_proposal.reasoning,
                    mode=state.data.get("mode", "unknown"),
                    confidence=confidence,
                )

                # Non-interactive: auto-approve if safety constraints pass.
                # Prevents the proposal queue from silently expiring with
                # no human to approve. The agent acts on its own judgment.
                from maxim.simulation.sim_logger import should_prompt

                if not should_prompt("plan_approval"):
                    autonomy_controller.proposal_queue.approve(proposal.id, approved_by="auto:non-interactive")
                    sim.log("PIPELINE", f"Auto-approved (non-interactive PLANNING): {action.get('tool_name')}")

                ctrl.pending_proposal = None

            else:
                # Check if this is a confirmation request (not a hard rejection)
                # Autonomy controller may say "requires approval" or "requires confirmation"
                if reason and ("requires confirmation" in reason.lower() or "requires approval" in reason.lower()):
                    tool_name = action.get("tool_name", "unknown")
                    params = action.get("params", {})

                    confirmation_data = {
                        "action": action,
                        "reasoning": ctrl.pending_proposal.reasoning,
                        "confidence": confidence,
                        "tool_name": tool_name,
                    }

                    # In sim mode, auto-resolve via response policy instead of blocking
                    sim_response = sim.resolve_confirmation(confirmation_data)
                    if sim_response is not None:
                        # Inject the response as if the user typed it
                        sim.log("PIPELINE", f"Auto-resolved confirmation for {tool_name}: {sim_response}")
                        state.data["pending_confirmation"] = confirmation_data
                        state.data["pending_cli_input"] = sim_response
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
                            if ctrl.pending_proposal.reasoning:
                                display_scene(f"  Reasoning: {ctrl.pending_proposal.reasoning}")
                            # Show confirmation prompt in the input panel so typed
                            # characters are visible with question context.
                            try:
                                from maxim.simulation.sim_logger import get_active_display

                                _conf_display = get_active_display()
                                if _conf_display is not None:
                                    _conf_display.set_prompt(f"Confirm {tool_name}? Type 'yes' or 'no'\n\n> ")
                                    _conf_display.set_urgent(True)
                            except Exception:
                                display_scene("  Type 'yes' or 'no' to confirm/reject:")
                        else:
                            # Non-interactive: auto-approve via supervision policy
                            state.data["pending_cli_input"] = "yes"
                            sim.log("PIPELINE", f"Auto-approved (non-interactive): {tool_name}")

                        state.data["pending_confirmation"] = confirmation_data
                    # Don't clear ctrl.pending_proposal yet - we need to wait for response
                else:
                    # Hard rejection - tool not allowed
                    autonomy_controller.log_action(
                        action_type="rejected",
                        action=action,
                        reasoning=f"Rejected: {reason}",
                        mode=state.data.get("mode", "unknown"),
                        confidence=confidence,
                    )
                    # Record rejection so LLM knows not to re-propose
                    rejection_msg = f"Rejected by autonomy: {reason}"
                    _rec_outcome(
                        agent_id=_loop_agent_id,
                        tool_name=action.get("tool_name", "unknown"),
                        success=False,
                        result_summary=None,
                        error=rejection_msg,
                        reasoning=ctrl.pending_proposal.reasoning or "",
                        recent_outcomes=recent_outcomes,
                        max_recent=max_recent_outcomes,
                        llm_worker=llm_worker,
                        context_pool=context_pool,
                        nac=_loop_nac,
                        active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
                        cluster_id=ctrl.pending_proposal.cluster_id,
                        clusters=ctrl.pending_proposal.clusters,
                    )
                    logger.info("Hard rejection recorded for LLM: %s", rejection_msg)
                ctrl.pending_proposal = None

        # ─────────────────────────────────────────────────────────────────
        # 5. CHECK FOR APPROVED PROPOSALS (PLANNING mode)
        # ─────────────────────────────────────────────────────────────────
        if autonomy_controller.current_level == AutonomyLevel.PLANNING:
            approved = autonomy_controller.proposal_queue.get_approved()
            for proposal in approved:
                if proposal.action:
                    tool_name = proposal.action.get("tool_name", "unknown")
                    try:
                        result = executor.execute(proposal.action)
                        success = getattr(result, "success", True)
                        output = getattr(result, "output", None)
                        error_msg = getattr(result, "error", None)
                        autonomy_controller.log_action(
                            action_type="executed",
                            action=proposal.action,
                            reasoning=proposal.reasoning,
                            mode=state.data.get("mode", "unknown"),
                            confidence=proposal.confidence,
                            human_involved=True,
                            outcome="success" if success else "failure",
                        )

                        # Record outcome so LLM sees the result
                        result_str = str(output)[:3000] if output is not None else None
                        _rec_outcome(
                            agent_id=_loop_agent_id,
                            tool_name=tool_name,
                            success=success,
                            result_summary=result_str,
                            error=error_msg,
                            reasoning=proposal.reasoning or "",
                            recent_outcomes=recent_outcomes,
                            max_recent=max_recent_outcomes,
                            llm_worker=llm_worker,
                            context_pool=context_pool,
                            nac=_loop_nac,
                            active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
                            cluster_id=proposal.cluster_id,
                            clusters=proposal.clusters,
                        )

                        # L2: Reset deliberation state on non-think tool execution
                        if tool_name != "think":
                            _reset_deliberation(executor)

                        # Queue follow-up so LLM can continue
                        from maxim.modes.definitions import get_tool_followup_type

                        current_mode = state.data.get("mode", "live")
                        followup_type = get_tool_followup_type(tool_name, current_mode)
                        should_followup = followup_type and (
                            (success and output is not None) or followup_type == "process"
                        )
                        if should_followup:
                            ctrl.pending_action_followup = ActionFollowup(
                                tool=tool_name,
                                result=result_str,
                                original_query="",
                                followup_type=followup_type,
                                mode=current_mode,
                                timestamp=time.time(),
                            )

                    except Exception as e:
                        logger.error(f"Approved action failed: {e}")
                        # Record failure so LLM knows (also fixes missing llm_worker call)
                        _rec_outcome(
                            agent_id=_loop_agent_id,
                            tool_name=tool_name,
                            success=False,
                            result_summary=None,
                            error=str(e),
                            reasoning=proposal.reasoning or "",
                            recent_outcomes=recent_outcomes,
                            max_recent=max_recent_outcomes,
                            llm_worker=llm_worker,
                            context_pool=context_pool,
                            nac=_loop_nac,
                            active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
                            cluster_id=proposal.cluster_id,
                            clusters=proposal.clusters,
                        )

        # ─────────────────────────────────────────────────────────────────
        # 6. SUBMIT NEW CONTEXT TO LLM (non-blocking, event-driven)
        # Only trigger LLM when there's something meaningful to respond to
        # ─────────────────────────────────────────────────────────────────
        # Diagnostic: trace why LLM submission is skipped
        if llm_worker and ctrl.pending_proposal is not None and cli_input and sim.is_sim_mode:
            _pp_tool = (
                ctrl.pending_proposal.action.get("tool_name", "?")
                if isinstance(ctrl.pending_proposal.action, dict)
                else "?"
            )
            sim.log(
                "PIPELINE", f"LLM gate BLOCKED: ctrl.pending_proposal={_pp_tool}, new cli_input={str(cli_input)[:40]}"
            )

        # ─────────────────────────────────────────────────────────────────
        # Substrate-primary action generation (Phase -1 of grounded
        # language acquisition). Skips the LLM entirely; calls
        # NAc.recommend_action() with current drives + tool registry. If
        # the substrate has no opinion (no learned bias, no active drive),
        # the tick is IDLE — no random fallback. Mutually exclusive with
        # the LLM submit branch below.
        if aut_mode == "substrate-primary" and ctrl.pending_proposal is None:
            now = time.time()
            if now - ctrl.last_llm_submit_time > llm_submit_interval:
                # Turn-scoped action budget (apparatus standard S6; the Exp 48
                # thrashing fix). A denied tick skips the proposal — the AUT
                # idles until the orchestrator opens the next turn window —
                # but still advances last_llm_submit_time and fires telemetry
                # (proposal=None, gated=True) so the cadence stays observable
                # AND gate-idle is distinguishable from substrate-no-opinion
                # IDLE in the telemetry artifact itself (review fold — the
                # once-per-window sim_log line alone marks the window, not
                # the rows). Drive drift is unaffected: it is wall-clock-lazy
                # and the next propose_via_substrate applies the accumulated dt.
                _substrate_gate_denied = substrate_action_gate is not None and not substrate_action_gate()
                substrate_proposal = None
                if not _substrate_gate_denied:
                    substrate_proposal = propose_via_substrate(
                        nac=_loop_nac,
                        agent_id=_loop_agent_id,
                        executor=executor,
                        sensor_encoder=_loop_sensor_encoder,
                    )
                ctrl.last_llm_submit_time = now
                if substrate_proposal is not None:
                    ctrl.pending_proposal = substrate_proposal
                    if sim.is_sim_mode:
                        sim.log(
                            "EXEC",
                            f"substrate-primary proposal: tool="
                            f"{substrate_proposal.action.get('tool_name') if substrate_proposal.action else None} "
                            f"confidence={substrate_proposal.confidence:.2f} "
                            f"reasoning={substrate_proposal.reasoning[:80]}",
                        )

                # Phase 0 telemetry — fires every tick (proposal or
                # IDLE). Fail-soft: telemetry exceptions never crash
                # the loop. See simulation/substrate_telemetry.py.
                if substrate_telemetry is not None:
                    try:
                        _ec_ref = getattr(memory_hub, "ec", None) if memory_hub is not None else None
                        substrate_telemetry.snapshot(
                            step=step_num,
                            nac=_loop_nac,
                            ec=_ec_ref,
                            executor=executor,
                            proposal=substrate_proposal,
                            gated=_substrate_gate_denied,
                        )
                    except Exception:
                        logger.debug("substrate telemetry callback raised", exc_info=True)

        if aut_mode != "substrate-primary" and llm_worker and ctrl.pending_proposal is None:
            now = time.time()
            if now - ctrl.last_llm_submit_time > llm_submit_interval:
                # Cache tool registry snapshot for this submission (avoids 3 redundant traversals)
                _all_tools = _get_all_tools()

                # Build context for LLM
                try:
                    context = None
                    if hasattr(memory, "build_context"):
                        context = memory.build_context()

                    # CRITICAL: If there's a pending action followup but no context,
                    # create a minimal context to ensure the followup gets processed.
                    # This fixes the bug where search results were returned but not
                    # processed because memory.build_context() returned None.
                    if ctrl.pending_action_followup and context is None:
                        from maxim.agents.bus import StructuredContext

                        context = StructuredContext(
                            timestamp=time.time(),
                            mode=state.data.get("mode", "active"),
                            autonomy_level=state.data.get("autonomy_level", "supervised"),
                            internet_access=state.data.get("internet_access", True),
                        )
                        logger.info("Created minimal context for pending action followup")

                    # §1.16 B1: a salient audio percept escalated this tick but
                    # carries no text, so memory may have built no context — mint
                    # a minimal one to carry the auto_sense_context (the azimuth
                    # line) into the submission.
                    if _audio_escalate_this_tick and context is None:
                        from maxim.agents.bus import StructuredContext

                        context = StructuredContext(
                            timestamp=time.time(),
                            mode=state.data.get("mode", "active"),
                            autonomy_level=state.data.get("autonomy_level", "supervised"),
                            internet_access=state.data.get("internet_access", True),
                        )
                        logger.debug("Created minimal context for audio-orient escalation")

                    # Inject bio-enrichment context if percept passed novelty gate
                    if context is not None and _percept_enrichment_text:
                        context.bio_enrichment_context = _percept_enrichment_text

                    # Auto-sense context (section 1.15) — passive perception
                    if context is not None and _auto_sense_text:
                        context.auto_sense_context = _auto_sense_text

                    # Wire-A (release_0_9_1.md Stage 2): cluster-bias annotation.
                    # Surface NAc agent-wide tool-bias to the LLM proposer so it
                    # can read substrate-acquired signal across percept regimes
                    # the substrate didn't directly drill (Roy-2c finding).
                    #
                    # The env var MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION=1 disables
                    # the read for the Roy-3 ablation; absence → on by default.
                    # Truthy-value parsing is shared with the conftest scrub +
                    # test suite via ``annotation_disabled_via_env`` so a future
                    # divergence here trips the test layer.
                    #
                    # ValueError narrowing per pre-merge review (architecture
                    # lens): a misconfigured per-agent stash (empty agent_id)
                    # MUST surface loudly via the WARNING log so the Roy-3
                    # measurement arm doesn't silently degrade to
                    # annotation-off behavior. Other exceptions propagate —
                    # the producer is on the LLM-submission hot path; a real
                    # bug here is correctness-critical (Roy-3 evidence
                    # integrity), not a recoverable nuisance.
                    #
                    # NOTE: the deliberation cycle re-submits the same context
                    # object without re-running this hook, so biases captured
                    # here are the snapshot at first submission. Bias change
                    # within a 1-tick deliberation cycle is bounded by
                    # ``reward_bias_alpha`` per call (~0.15); the stale-by-one-
                    # tick read is acceptable, and avoiding per-cycle re-reads
                    # keeps NAc lock contention bounded.
                    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

                    if (
                        context is not None
                        and _loop_nac is not None
                        and not annotation_disabled_via_env(os.environ.get("MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION"))
                    ):
                        try:
                            context.cluster_bias_annotations = _loop_nac.get_agent_tool_biases(
                                agent_id=_loop_agent_id,
                                top_n=5,
                            )
                            # S1 credit-source provenance: same agent-wide
                            # aggregation, keyed by RAW tool signature so the
                            # composer can join before the prefix strip.
                            # Pre-S1 persisted state has no sources — the
                            # empty dict renders the pre-S1 format.
                            context.cluster_bias_sources = _loop_nac.get_cluster_reward_sources(
                                agent_id=_loop_agent_id,
                            )
                        except ValueError as e:
                            logger.warning(
                                "Wire-A annotation skipped due to invalid agent_id (%s); "
                                "Roy-3 measurement integrity may be affected.",
                                e,
                            )
                            context.cluster_bias_annotations = None
                            context.cluster_bias_sources = None

                    # W1 sense_tool_registry MVP (grayscale visibility).
                    # Surfaces SEM-derived tools the substrate has a
                    # non-zero reward bias for but that are NOT in the
                    # active scene roster — the Roy-3a-retry gap. The
                    # full filter (strip ``tool:`` prefix, exclude
                    # active, require SEM kind, cap at top-N) lives in
                    # ``prompts/grayscale_tools_annotation.py`` so it
                    # can be unit-tested against realistic NAc-shaped
                    # input — the pre-merge bio-fidelity review caught
                    # a silent prefix mismatch the in-line producer
                    # would have hidden in production.
                    #
                    # Reuses ``cluster_bias_annotations`` populated by
                    # the Wire-A block above so the producer makes ONE
                    # NAc call per submission, not two.
                    #
                    # Shares Wire-A's env-var kill switch
                    # (MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION=1). The two
                    # sections are different surfaces of the same
                    # substrate signal; disabling one without the other
                    # leaks the signal under a different header and
                    # pollutes Roy ablation evidence.
                    if (
                        context is not None
                        and _loop_nac is not None
                        and executor is not None
                        and not annotation_disabled_via_env(os.environ.get("MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION"))
                    ):
                        try:
                            _gs_registry = getattr(executor, "_registry", None) or getattr(executor, "registry", None)
                            # Reuse Wire-A's biases when present; the
                            # env-var gate above means the only way
                            # ``cluster_bias_annotations`` is None here
                            # is the Wire-A ValueError branch (invalid
                            # agent_id), in which case grayscale would
                            # hit the same error — skip rather than
                            # double-log.
                            _gs_biases = context.cluster_bias_annotations
                            if _gs_biases:
                                from maxim.prompts.grayscale_tools_annotation import (
                                    build_grayscale_annotations,
                                )

                                _gs_annotations = build_grayscale_annotations(
                                    _gs_biases,
                                    _gs_registry,
                                    top_n=5,
                                )
                                if _gs_annotations:
                                    context.grayscale_tool_annotations = _gs_annotations
                        except (AttributeError, ValueError) as e:
                            # Narrowed per architecture-lens review:
                            # AttributeError catches broken executor /
                            # registry shape (out-of-tree subclass
                            # without the new helpers); ValueError
                            # mirrors Wire-A's discipline. Other
                            # exceptions propagate — the producer is on
                            # the LLM-submission hot path and a real
                            # bug here is correctness-critical, not a
                            # recoverable nuisance.
                            logger.warning("Grayscale tools annotation skipped: %s", e)
                            context.grayscale_tool_annotations = None

                    # Check if there's something meaningful to react to
                    # Only submit to LLM if we have:
                    # 1. New CLI input with "maxim" keyword - not already processed
                    # 2. New speech detected (transcription with "maxim" keyword)
                    # 3. Direct address via maxim keyword in recent percepts
                    # 4. Hard override commands
                    # 5. Exploration mode with high novelty detection (periodic checks)
                    #
                    # SLEEP STATE: Skip LLM processing unless wake keyword detected
                    is_sleeping = state.data.get("processing_state", "awake") == "sleep"
                    has_meaningful_input = False
                    new_cli_input = None

                    # If a fresh percept arrived THIS iteration (cli_input from
                    # observation), it must be processed even if the same text was
                    # seen before.  The context.cli_inputs dedup below catches
                    # stale entries from the MemoryAgent deque, but a fresh percept
                    # is a new turn — e.g., the orchestrator sending the same probe
                    # twice is intentional and each must reach the LLM.
                    if cli_input and not is_sleeping:
                        new_cli_input = str(cli_input)
                        has_meaningful_input = True
                    # §1.16 B1: a salient audio percept escalated this tick — it
                    # must reach the LLM even with no text input, or the folded
                    # azimuth line is discarded at the `context = None` gate below.
                    if _audio_escalate_this_tick and not is_sleeping:
                        has_meaningful_input = True
                    # Track original query from followups for conversation history
                    # This ensures followup responses are saved with the original user question
                    followup_original_query = ""
                    if context:
                        # Known commands that should NOT be sent to LLM
                        # These are handled by Selfy's phrase response system
                        SKIP_LLM_COMMANDS = frozenset(
                            {
                                # System commands
                                "maxim shutdown",
                                "shutdown maxim",
                                "maxim stop",
                                "stop maxim",
                                "maxim pause",
                                "pause maxim",
                                "maxim resume",
                                "resume maxim",
                                # Sleep/wake (processing state)
                                "maxim sleep",
                                "sleep maxim",
                                "maxim nap",
                                "maxim rest",
                                "maxim wake",
                                "wake maxim",
                                "maxim wake up",
                                "wake up maxim",
                                # Strategy switching
                                "maxim observe",
                                "observe maxim",
                                "maxim watch",
                                "maxim explore",
                                "explore maxim",
                                "maxim research",
                                "research maxim",
                                "maxim assist",
                                "maxim help",
                                "maxim reflect",
                                "maxim reflection",
                                "maxim learn",
                                "maxim train",
                                # Mode switching
                                "maxim passive",
                                "maxim active",
                                "maxim singularity",
                            }
                        )

                        # Check for NEW CLI input (not already processed)
                        # CLI input always goes to LLM (maxim prefix presumed)
                        # Voice input still requires "maxim" wake word
                        input_source = state.data.get("pending_user_input_source", "CLI")
                        if context.cli_inputs:
                            for cli_input in context.cli_inputs:
                                if cli_input and cli_input not in processed_cli_inputs:
                                    cli_lower = cli_input.lower().strip()
                                    # CLI source: always process (prefix presumed)
                                    # Voice source: require "maxim" or "reachy" wake word
                                    should_process = (
                                        (input_source == "CLI") or ("maxim" in cli_lower) or ("reachy" in cli_lower)
                                    )
                                    if should_process:
                                        # Skip LLM for known commands (handled by Selfy)
                                        if cli_lower in SKIP_LLM_COMMANDS:
                                            logger.info("Skipping LLM for command: %s", cli_input)
                                            processed_cli_inputs.append(cli_input)
                                            continue
                                        new_cli_input = cli_input
                                        has_meaningful_input = True
                                        break
                                    else:
                                        # Mark voice inputs without wake word as processed
                                        processed_cli_inputs.append(cli_input)

                        # Check for speech with maxim keyword
                        if context.detected_speech:
                            for speech in context.detected_speech:
                                if speech and "maxim" in speech.lower():
                                    has_meaningful_input = True
                                    break

                        # Check for maxim keyword in current percept
                        if context.current_percept and context.current_percept.has_maxim_keyword:
                            has_meaningful_input = True

                        # Check for hard override commands
                        if context.current_percept and context.current_percept.hard_override:
                            has_meaningful_input = True

                        # Check for pending action followup (tools that need LLM processing)
                        if ctrl.pending_action_followup:
                            has_meaningful_input = True
                            # Inject the action result into CLI inputs so LLM can process
                            followup_query = ctrl.pending_action_followup.original_query
                            followup_result = ctrl.pending_action_followup.result or ""
                            followup_tool = ctrl.pending_action_followup.tool
                            followup_type = ctrl.pending_action_followup.followup_type
                            followup_mode = ctrl.pending_action_followup.mode

                            # Preserve original query for conversation history tracking
                            # This ensures followup responses are saved with the original user question
                            followup_original_query = followup_query

                            # Create a synthetic input with followup metadata
                            # Format: [ACTION_FOLLOWUP type=X tool=Y query='Z']: result
                            synthetic_input = (
                                f"[ACTION_FOLLOWUP type={followup_type} tool={followup_tool} "
                                f"mode={followup_mode} query='{followup_query}']: {followup_result}"
                            )
                            if context.cli_inputs:
                                context.cli_inputs.append(synthetic_input)
                            else:
                                context.cli_inputs = [synthetic_input]
                            logger.info(
                                "Injected action followup into context: type=%s, tool=%s, result_len=%d",
                                followup_type,
                                followup_tool,
                                len(followup_result),
                            )
                            # Clear the followup after processing
                            ctrl.pending_action_followup = None

                    # Skip LLM if nothing to react to
                    if not has_meaningful_input:
                        if sim.is_sim_mode and context and context.cli_inputs:
                            _unprocessed = [c for c in context.cli_inputs if c not in processed_cli_inputs]
                            if _unprocessed:
                                sim.log(
                                    "PIPELINE",
                                    f"LLM skip: has_meaningful=False but {len(_unprocessed)} unprocessed cli_inputs (all already processed)",
                                )
                        context = None  # Skip this submission

                    # Sleep state: Skip LLM unless wake keyword detected
                    # This allows background processing while monitoring for activation
                    if is_sleeping and context:
                        from maxim.modes.definitions import is_wake_keyword

                        has_wake_keyword = False
                        if context.cli_inputs:
                            for cli in context.cli_inputs:
                                if cli and is_wake_keyword(cli):
                                    has_wake_keyword = True
                                    break
                        if not has_wake_keyword and context.detected_speech:
                            for speech in context.detected_speech:
                                if speech and is_wake_keyword(speech):
                                    has_wake_keyword = True
                                    break
                        if not has_wake_keyword:
                            logger.debug("Sleep state: skipping LLM (no wake keyword)")
                            context = None

                    # Mark new CLI input as processed to prevent duplicate submissions
                    # deque(maxlen=20) auto-evicts oldest on overflow — no manual check needed
                    if new_cli_input:
                        processed_cli_inputs.append(new_cli_input)

                    if context:
                        # Get mode info
                        mode_name = state.data.get("mode", "observe")

                        # Get mode definition for tool access
                        mode_def = get_mode(mode_name)
                        available_tools_for_mode = set()
                        if mode_def and _all_tools:
                            available_tools_for_mode = mode_def.get_available_tools(_all_tools)

                        mode_info = ModeInfo(
                            name=mode_name,
                            goal=mode_def.goal if mode_def else "Respond to user requests",
                            context_prompt=mode_def.context_prompt if mode_def else "",
                            allowed_tools=available_tools_for_mode,
                            forbidden_tools=mode_def.forbidden_tools if mode_def else set(),
                            can_access_filesystem=mode_def.can_access_filesystem if mode_def else True,
                            can_access_network=mode_def.can_access_network if mode_def else True,
                            uses_tool_relevance_filter=(mode_def.uses_tool_relevance_filter if mode_def else False),
                        )

                        # Get internet access status
                        internet_access = state.data.get("internet_access", False)
                        internet_policy_summary = state.data.get("internet_policy_summary", "")

                        # Filter CLI inputs to only include new ones (not already processed)
                        # This prevents the LLM from seeing the same input multiple times
                        if context.cli_inputs:
                            new_inputs = [
                                inp
                                for inp in context.cli_inputs
                                if inp not in processed_cli_inputs or inp == new_cli_input
                            ]
                            # Only keep the most recent new input
                            if new_cli_input:
                                context.cli_inputs = [new_cli_input]
                            else:
                                context.cli_inputs = new_inputs[-1:] if new_inputs else []

                        # Get available tools for this mode
                        available_tools = mode_info.get_available_tools(_all_tools)

                        # Body ownership is not a mode privilege
                        # (sem_motor_binding.md Phase 1): SEM affordance
                        # tools generated from a wired body join the
                        # described tool list past the mode filter —
                        # otherwise a live mode's non-empty allowed_tools
                        # hides reachy_mini_turn_* and they surface only
                        # as bare tokens via the relevance filter (the
                        # 'move' bare-token lesson).
                        if executor is not None and getattr(executor, "embodiment", None) is not None:
                            try:
                                from maxim.embodiment.tool_bridge import always_active_sem_tools

                                _exec_registry = executor.registry
                                for _sem_tool in always_active_sem_tools(_exec_registry):
                                    if _exec_registry.is_tool_active(_sem_tool.name):
                                        available_tools.add(_sem_tool.name)
                            except Exception:
                                # Silent loss here re-creates the bare-token
                                # bug this union exists to fix — log it.
                                logger.debug("SEM prompt union failed", exc_info=True)

                        # Wire 3 (release_0_9_1.md Stage 1): filter tools
                        # routed through critically-damaged components and
                        # collect degraded-affordance annotations. Pulls
                        # from Embodiment.get_disabled_affordances() /
                        # .get_degraded_affordances() which read each
                        # modulator's compute_integrity(). Default
                        # thresholds: integrity < 0.3 → disabled
                        # (filtered out); 0.3 <= integrity < 0.6 →
                        # annotated with a felt-sensation phrase
                        # ("feels strained" / "feels weakened, prone to
                        # failing") so the LLM proposer reads the
                        # affordance's cost in proprioceptive voice
                        # rather than as a system advisor (bio-fidelity
                        # fold). Fail-open at the narrowed exception
                        # surface (no embodiment, missing methods, broken
                        # modulator shape) — the filter is a no-op but
                        # the WARNING surfaces operator-visible signal.
                        #
                        # NOTE on `last_surfaced_tools`: post-filter is
                        # intentional. The learned tool-relevance index
                        # at line ~1700 calls `record_surfaced_but_unused`
                        # — disabled tools weren't surfaced, so they
                        # don't decay. Future: if a disabled tool
                        # recovers, the relevance index resumes decay
                        # the next tick the tool surfaces again.
                        _wire3_embodiment = getattr(executor, "embodiment", None)
                        _wire3_degraded: dict[str, float] = {}
                        _wire3_disabled: set[str] = set()
                        if _wire3_embodiment is not None:
                            try:
                                _wire3_disabled = _wire3_embodiment.get_disabled_affordances()
                                if _wire3_disabled:
                                    available_tools = [t for t in available_tools if t not in _wire3_disabled]
                                _wire3_degraded = _wire3_embodiment.get_degraded_affordances()
                            except (AttributeError, TypeError) as e:
                                # Narrowed from broad Exception per
                                # arch-lens A4: the inner body.py guard
                                # already swallows compute_integrity
                                # raises with a WARNING. Outer surface
                                # failures here are method-shape
                                # mismatches (non-Embodiment object
                                # plugged into executor.embodiment) —
                                # WARN so operator review catches it.
                                logger.warning(
                                    "Wire 3: get_disabled/degraded_affordances shape mismatch — filter no-op: %s",
                                    e,
                                )
                                _wire3_degraded = {}
                                _wire3_disabled = set()
                            # Emit Roy-3 disambiguator (bio-fidelity B2):
                            # without this, "Wire 3 hid the tool" and
                            # "substrate learned avoidance" are
                            # indistinguishable post-hoc. The event
                            # lists which affordances were filtered /
                            # annotated each LLM submission so Roy-3
                            # can quantify Wire 3's effect surface.
                            if _wire3_disabled or _wire3_degraded:
                                try:
                                    from maxim.simulation import sim_logger as _sl_w3

                                    _w3_tick = int(time.time() - _sl_w3._sim_start) if _sl_w3._sim_start > 0.0 else 0
                                    _sl_w3.sim_log(
                                        "WIRE_3_FILTER",
                                        f"wire_3: disabled={len(_wire3_disabled)} degraded={len(_wire3_degraded)}",
                                        {
                                            "tick": _w3_tick,
                                            "disabled_tools": sorted(_wire3_disabled),
                                            # Pass integrity floats only here — the LLM
                                            # sees the felt phrases above.
                                            "degraded_integrities": {
                                                name: round(integrity, 4) for name, integrity in _wire3_degraded.items()
                                            },
                                        },
                                    )
                                except ImportError:
                                    # Non-sim runtime — observability
                                    # is optional, never load-bearing.
                                    pass
                        last_surfaced_tools = list(available_tools)

                        # Get full tool info for prompt (description, params, example).
                        # Route through Tool.to_json_schema() so dynamic tools authored
                        # in either input_schema format render correctly. Pre-CC9 this
                        # iterated input_schema.items() directly and silently produced
                        # empty params for JSONSchema-authored tools (the @maxim.tool
                        # decorator path) via the swallow-everything except below.
                        tool_descriptions = {}
                        for name in available_tools:
                            if name in TOOL_DESCRIPTIONS:
                                tool_descriptions[name] = TOOL_DESCRIPTIONS[name]
                            else:
                                # Dynamic tool (from skill/protocol) — build from Tool instance
                                try:
                                    tool = executor.registry.get(name)
                                    json_schema = (
                                        tool.to_json_schema()
                                        if hasattr(tool, "to_json_schema")
                                        else {"properties": {}, "required": []}
                                    )
                                    properties = json_schema.get("properties", {}) or {}
                                    required = set(json_schema.get("required", []) or [])
                                    params: dict[str, str] = {}
                                    for param_name, prop in properties.items():
                                        prop_type = prop.get("type", "string") if isinstance(prop, dict) else "string"
                                        if param_name in required:
                                            params[param_name] = str(prop_type)
                                        else:
                                            default = prop.get("default") if isinstance(prop, dict) else None
                                            params[param_name] = f"({prop_type}, default={default!r})"
                                    tool_descriptions[name] = {
                                        "description": tool.description,
                                        "params": params,
                                        "example": None,
                                        "followup_type": None,
                                    }
                                except (KeyError, Exception):
                                    pass

                        # Wire 3: annotate degraded tools' descriptions in
                        # place with a felt-sensation phrase (bio-fidelity
                        # fold). The annotation lives at the end of the
                        # description string so the LLM reads the body's
                        # state in proprioceptive voice without losing
                        # the tool's normal capability blurb. Uses the
                        # per-tool entry's structure (dict for dynamic
                        # tools, TOOL_DESCRIPTIONS dict for builtin);
                        # skips any tool whose description shape we don't
                        # recognise, fail-open.
                        #
                        # Idempotency under integrity drift (arch A1):
                        # if integrity ticks 0.5 → 0.4 → 0.5 across a
                        # session, the felt phrase changes per band.
                        # The regex strip removes any existing
                        # ``(feels …)`` Wire 3 annotation before
                        # appending the current one so phrases don't
                        # accumulate. Healthy / disabled affordances
                        # never enter this loop, so the only way to
                        # have an annotation present is via a prior
                        # Wire 3 pass.
                        if _wire3_degraded:
                            for name, integrity in _wire3_degraded.items():
                                entry = tool_descriptions.get(name)
                                if not isinstance(entry, dict):
                                    continue
                                base_desc = entry.get("description", "")
                                if not isinstance(base_desc, str):
                                    continue
                                phrase = _wire3_embodiment.integrity_to_felt_phrase(integrity)
                                if not phrase:
                                    continue
                                annotation = f" ({phrase})"
                                # Strip any prior felt annotation pinned
                                # by Embodiment.integrity_to_felt_phrase
                                # — the two bands give two distinct
                                # suffixes which could otherwise stack.
                                stripped = _WIRE3_PHRASE_RE.sub("", base_desc)
                                # Copy-on-write — TOOL_DESCRIPTIONS is
                                # a shared module-level dict; mutating
                                # it would poison the description for
                                # future calls (and other agents).
                                tool_descriptions[name] = {**entry, "description": stripped + annotation}

                        # Wire 1 (release_0_9_1.md Stage 4): annotate
                        # tools with outcome-variance experience phrasing.
                        # Runs AFTER Wire 3 so an integrity-degraded tool
                        # that is ALSO outcome-variable gets both
                        # annotations (orthogonal signals: somatic body
                        # damage vs. experience-acquired unpredictability).
                        # The hybrid bio-system + LLM caveat is documented
                        # in docs/plans/deferred/bio_emergent_persona_foundations.md
                        # § Wire 1 — variance reaches the LLM through
                        # description text, not a pre-filter ranker.
                        # A cleaner post-1.0 design would pre-rank tools
                        # before the LLM sees them.
                        #
                        # Idempotency: _WIRE1_PHRASE_RE strips the prior
                        # annotation before the current one is appended
                        # so the description does not accumulate suffixes
                        # across observations. Wire 3 (somatic
                        # "feels strained" / "feels weakened, prone to
                        # failing") and Wire 1 (experience-voice
                        # "unpredictable from prior experience" /
                        # "reliable from prior experience") use distinct
                        # phrases so the two regexes do not conflict.
                        _wire1_annotated_high: list[str] = []
                        _wire1_annotated_low: list[str] = []
                        _wire1_middle_band: list[tuple[str, float]] = []
                        _wire1_felt_phrases: dict[str, str] = {}
                        # Gate on bio-system + tool list FIRST so cold-start
                        # agents skip the env-var read entirely (one fewer
                        # os.environ lookup per submission). The ablation
                        # parser is Wire-A's canonical helper — single
                        # source of truth across 0.9.1's two gates.
                        _risk_profile: dict[str, float] = {}
                        if _loop_nac is not None and available_tools:
                            from maxim.prompts.cluster_bias_annotation import (
                                annotation_disabled_via_env,
                            )

                            _wire1_disabled_via_env = annotation_disabled_via_env(os.environ.get(_WIRE1_DISABLE_ENV))
                            if not _wire1_disabled_via_env:
                                try:
                                    _risk_profile = _loop_nac.get_action_risk_profile(
                                        agent_id=_loop_agent_id,
                                        min_observations=_WIRE1_MIN_OBSERVATIONS,
                                    )
                                except (ValueError, AttributeError) as e:
                                    # ValueError: empty agent_id (defensive —
                                    # _loop_agent_id is non-empty by construction
                                    # at line ~1025, but a future refactor could
                                    # introduce a regression here). AttributeError:
                                    # _loop_nac is something other than an NAc
                                    # (test stubs / forward-compat). Either case:
                                    # WARN so operators notice + no-op the wire.
                                    logger.warning("Wire 1: risk profile fetch failed — annotation no-op: %s", e)
                                    _risk_profile = {}
                            # _risk_profile keys are "tool:<name>" (or
                            # "tool:use:<action>" for the generic use
                            # tool); available_tools entries are bare
                            # names. Strip the "tool:" prefix to match.
                            for event_sig, variance in _risk_profile.items():
                                if event_sig.startswith("tool:use:"):
                                    # The compound generic-use signature
                                    # doesn't map back to a bare tool
                                    # name in available_tools — the
                                    # available_tools entry is "use".
                                    # Skip; Wire 1 doesn't annotate at
                                    # the action-arg level (no separate
                                    # description per use:<action>).
                                    continue
                                tool_name = event_sig[len("tool:") :] if event_sig.startswith("tool:") else event_sig
                                if tool_name not in available_tools:
                                    continue
                                if variance >= _WIRE1_HIGH_VARIANCE_THRESHOLD:
                                    phrase = _WIRE1_HIGH_PHRASE
                                    _wire1_annotated_high.append(tool_name)
                                    _wire1_felt_phrases[tool_name] = phrase
                                elif variance < _WIRE1_LOW_VARIANCE_THRESHOLD:
                                    phrase = _WIRE1_LOW_PHRASE
                                    _wire1_annotated_low.append(tool_name)
                                    _wire1_felt_phrases[tool_name] = phrase
                                else:
                                    # Middle band: no annotation. Strip
                                    # any prior annotation so an LLM
                                    # cannot read stale signal after the
                                    # tool's variance has decayed back
                                    # to the neutral band.
                                    _wire1_middle_band.append((tool_name, variance))
                                    entry = tool_descriptions.get(tool_name)
                                    if isinstance(entry, dict):
                                        base_desc = entry.get("description", "")
                                        if isinstance(base_desc, str) and _WIRE1_PHRASE_RE.search(base_desc):
                                            tool_descriptions[tool_name] = {
                                                **entry,
                                                "description": _WIRE1_PHRASE_RE.sub("", base_desc),
                                            }
                                    continue
                                entry = tool_descriptions.get(tool_name)
                                if not isinstance(entry, dict):
                                    continue
                                base_desc = entry.get("description", "")
                                if not isinstance(base_desc, str):
                                    continue
                                stripped = _WIRE1_PHRASE_RE.sub("", base_desc)
                                tool_descriptions[tool_name] = {
                                    **entry,
                                    "description": f"{stripped} ({phrase})",
                                }
                            # Emit Roy-3 disambiguator (bio-fidelity
                            # measurability): without this event,
                            # "Wire 1 annotated the tool" and "LLM
                            # ignored the annotation" are
                            # indistinguishable post-hoc. Mirrors the
                            # Wire 3 WIRE_3_FILTER shape so Roy-3 can
                            # count annotation effects uniformly.
                            # Payload carries:
                            #   - agent_id          (multi-agent attribution)
                            #   - high_variance_tools, reliable_tools
                            #   - felt_phrases      (exact strings the LLM saw)
                            #   - annotated_variances (numeric float per annotated tool)
                            #   - middle_band       (tool, variance) tuples for
                            #                       counterfactual Roy-3 analysis
                            if _wire1_annotated_high or _wire1_annotated_low or _wire1_middle_band:
                                try:
                                    from maxim.simulation import sim_logger as _sl_w1

                                    _w1_tick = int(time.time() - _sl_w1._sim_start) if _sl_w1._sim_start > 0.0 else 0
                                    _sl_w1.sim_log(
                                        "WIRE_1_ANNOTATION",
                                        f"wire_1: high_variance={len(_wire1_annotated_high)} "
                                        f"reliable={len(_wire1_annotated_low)} "
                                        f"middle={len(_wire1_middle_band)}",
                                        {
                                            "tick": _w1_tick,
                                            "agent_id": _loop_agent_id,
                                            "high_variance_tools": sorted(_wire1_annotated_high),
                                            "reliable_tools": sorted(_wire1_annotated_low),
                                            # felt_phrases is the LLM-visible
                                            # text — Roy-3 reads this to
                                            # decide what the prompt actually
                                            # contained (vs reconstructing
                                            # from thresholds + variances).
                                            "felt_phrases": dict(_wire1_felt_phrases),
                                            # Numeric variance for each
                                            # annotated tool — LLM does NOT
                                            # see these floats, only the
                                            # felt phrase above.
                                            "annotated_variances": {
                                                (
                                                    event_sig[len("tool:") :]
                                                    if event_sig.startswith("tool:")
                                                    else event_sig
                                                ): round(variance, 4)
                                                for event_sig, variance in _risk_profile.items()
                                                if not event_sig.startswith("tool:use:")
                                                and (
                                                    event_sig[len("tool:") :]
                                                    if event_sig.startswith("tool:")
                                                    else event_sig
                                                )
                                                in (_wire1_annotated_high + _wire1_annotated_low)
                                            },
                                            # Middle-band tools: present
                                            # in the profile but not
                                            # annotated. The counterfactual
                                            # for Roy-3 ablation analysis
                                            # ("substrate produced variance
                                            # in this band but no
                                            # annotation reached the LLM").
                                            "middle_band_variances": {
                                                name: round(var, 4) for name, var in _wire1_middle_band
                                            },
                                        },
                                    )
                                except ImportError:
                                    pass

                        # Get context pool text
                        context_pool_text = context_pool.get_context_text(
                            max_tokens=mode_info.context_window_tokens // 2
                        )

                        # Get conversation history for context
                        conversation_history_text = context_pool.get_conversation_text(max_turns=5)

                        # Check for pending modification request
                        pending_modification = state.data.pop("pending_modification", None)

                        # Format prefetch context if available
                        prefetch_context_text = ""
                        skip_exploration = False
                        if pending_prefetch is not None:
                            prefetch_context_text = prefetcher.format_prefetch_context(pending_prefetch)
                            skip_exploration = pending_prefetch.skip_exploration
                            # Clear after use
                            pending_prefetch = None

                        # Get protocol context (re-injected fresh each submission)
                        _protocol_context = ""
                        if protocol_registry is not None:
                            _protocol_context = protocol_registry.get_context_for_llm()

                        if callable(on_event):
                            try:
                                on_event(StreamEvent("inference_start", {}))
                            except Exception as e:
                                log_swallowed_exception(e, operation="on_event:inference_start")

                        # EXPERIMENTAL — hallucination-hint feature.
                        # MAXIM_TOOL_FAILURE_HINTS=0 disables (default on).
                        # Disable for grounded-language acquisition Phase 0/1.
                        # See docs/plans/grounded_language_acquisition.md.
                        _failed_tools: list[str] = []
                        if os.environ.get("MAXIM_TOOL_FAILURE_HINTS", "0") != "0":
                            # __getattr__ on wrappers (InstrumentedExecutor,
                            # PainInterceptorExecutor, FearGatedExecutor) walks
                            # down to the base Executor where the list lives.
                            _failed_tools = list(getattr(executor, "_tools_hallucinated", []))

                        submitted = llm_worker.submit_context(
                            context=context,
                            mode=mode_info,
                            autonomy_level=autonomy_controller.current_level,
                            internet_access=internet_access,
                            internet_policy_summary=internet_policy_summary,
                            available_tools=available_tools,
                            tool_descriptions=tool_descriptions,
                            failed_tools=_failed_tools,
                            context_pool_text=context_pool_text,
                            agent_states=agent_states,
                            recent_outcomes=recent_outcomes,
                            use_tool_prompting=use_tool_prompting and bool(available_tools),
                            # Use modification text as triggering_input if no new input but pending modification
                            triggering_input=new_cli_input
                            or followup_original_query
                            or (pending_modification.get("user_modification", "") if pending_modification else ""),
                            conversation_history_text=conversation_history_text,
                            pending_modification=pending_modification,
                            prefetch_context=prefetch_context_text,
                            skip_exploration=skip_exploration,
                            is_sleeping=is_sleeping,
                            protocol_context=_protocol_context,
                        )
                        ctrl.last_llm_submit_time = now
                        # Log submission for both user input and followups
                        if submitted:
                            # Check if this is a followup submission
                            is_followup = context.cli_inputs and any(
                                inp.startswith("[ACTION_FOLLOWUP") for inp in context.cli_inputs
                            )
                            if is_followup:
                                logger.info("Submitted followup to LLM for processing")
                            elif new_cli_input:
                                logger.info(
                                    "Submitted to LLM: %s",
                                    new_cli_input[:50] if len(new_cli_input) > 50 else new_cli_input,
                                )
                            log_agentic(
                                "agent_loop",
                                "llm_submit",
                                {
                                    "input": new_cli_input[:50]
                                    if new_cli_input
                                    else "followup"
                                    if is_followup
                                    else None,
                                    "mode": mode_info.name,
                                    "autonomy": autonomy_controller.current_level.value,
                                    "tools_available": len(available_tools),
                                },
                            )
                            sim.log("EXEC", f"LLM submit: {new_cli_input if new_cli_input else 'followup'}")

                        # ── PFC multi-cycle deliberation ──────────────────
                        # When the gate passed in section 1.2 and the first
                        # proposal says ready_to_act=False, run cycles 2+
                        # right here where all submission params are in scope.
                        if submitted and _pfc_gate_passed and bio_enrichment_pipeline is not None:
                            _first = _wait_for_proposal(llm_worker, stop_event, ctrl=ctrl)
                            if _first is not None:
                                _max_cyc = 3 if percept_source is not None else 2
                                if not _first.ready_to_act:
                                    # Build a submit closure that captures all params
                                    _submit_kwargs = dict(
                                        mode=mode_info,
                                        autonomy_level=autonomy_controller.current_level,
                                        internet_access=internet_access,
                                        internet_policy_summary=internet_policy_summary,
                                        available_tools=available_tools,
                                        tool_descriptions=tool_descriptions,
                                        context_pool_text=context_pool_text,
                                        agent_states=agent_states,
                                        recent_outcomes=recent_outcomes,
                                        use_tool_prompting=use_tool_prompting and bool(available_tools),
                                        triggering_input="",
                                        conversation_history_text=conversation_history_text,
                                        pending_modification=None,
                                        prefetch_context="",
                                        skip_exploration=False,
                                        is_sleeping=is_sleeping,
                                        protocol_context=_protocol_context,
                                    )

                                    def _submit_fn(_ctx: Any, _kw: dict = _submit_kwargs) -> bool:
                                        return llm_worker.submit_context(context=_ctx, **_kw)

                                    _active_goal = state.data.get("active_goal") if hasattr(state, "data") else None
                                    _delib = _run_deliberation_cycles(
                                        first_proposal=_first,
                                        bio_enrichment=bio_enrichment_pipeline,
                                        working_memory=_wms,
                                        context=context,
                                        submit_fn=_submit_fn,
                                        llm_worker=llm_worker,
                                        stop_event=stop_event,
                                        thought_gate=thought_gate,
                                        active_goal=_active_goal,
                                        step_num=step_num,
                                        max_cycles=_max_cyc,
                                        ctrl=ctrl,
                                    )
                                    if _delib is not None:
                                        ctrl.pending_proposal = _delib
                                        sim.log(
                                            "DELIBERATION",
                                            f"multi-cycle deliberation yielded proposal: "
                                            f"tool={_delib.action.get('tool_name') if isinstance(_delib.action, dict) else None}",
                                        )
                                    else:
                                        sim.log("DELIBERATION", "multi-cycle deliberation returned None (IDLE)")
                                else:
                                    # ready_to_act == True on cycle 1 — use directly
                                    ctrl.pending_proposal = _first
                                    from maxim.simulation.sim_logger import (
                                        sim_contemplation,
                                        sim_deliberation_end,
                                        sim_deliberation_update,
                                    )

                                    # Update thinking panel with the AUT's actual reasoning
                                    # Only if the thought is novel enough vs recent thoughts
                                    _first_reasoning = _first.reasoning or ""
                                    if _first_reasoning and _is_novel_thought(_first_reasoning):
                                        sim_deliberation_update(
                                            _first_reasoning,
                                            cycle=1,
                                            max_cycles=_max_cyc,
                                            salience=locals().get("_salience_c1"),
                                        )
                                    # NOTE: score=0.0 is a placeholder here —
                                    # the gate decision is not in this scope
                                    # (different function). Left explicit
                                    # rather than silently fabricated; wire a
                                    # real value if this log is ever relied on.
                                    sim_contemplation(gate_passed=True, refined=False, score=0.0)
                                    sim_deliberation_end(cycle=1, max_cycles=_max_cyc, summary="Ready to act (cycle 1)")
                                    if thought_gate is not None:
                                        thought_gate.reset_refractory(step_num)

                except Exception as e:
                    import traceback

                    logger.warning(f"Failed to submit context to LLM: {type(e).__name__}: {e}")
                    logger.debug(f"Full traceback:\n{traceback.format_exc()}")

        # ─────────────────────────────────────────────────────────────────
        # 7. CALL STEP CALLBACK
        # ─────────────────────────────────────────────────────────────────
        try:
            if callable(on_step):
                on_step(
                    {
                        "step": step_num,
                        "state": state,
                        "memory": memory,
                        "autonomy_level": autonomy_controller.current_level.value,
                        "ctrl.pending_proposal": ctrl.pending_proposal is not None,
                    }
                )
        except Exception:
            logger.debug("on_step callback failed", exc_info=True)

        # ─────────────────────────────────────────────────────────────────
        # 8. INCREMENT STEP COUNTER AND PERSIST
        # ─────────────────────────────────────────────────────────────────
        try:
            state.steps_taken += 1
        except Exception as e:
            log_swallowed_exception(e, operation="state.steps_taken_increment")

        if persist_every_n_steps > 0 and state.steps_taken % persist_every_n_steps == 0:
            _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})

        if state.is_done():
            break

        # ─────────────────────────────────────────────────────────────────
        # 8.5 BIO-SYSTEM PER-TICK MAINTENANCE
        # ─────────────────────────────────────────────────────────────────
        # NAc eligibility traces and reward biases decay each tick.
        # Without this, traces persist indefinitely and distribute_reward
        # credits ALL nodes ever activated in the session at original
        # strength — incorrect for causal credit assignment.
        if _loop_nac is not None:
            try:
                _loop_nac.decay_eligibility()
                _loop_nac.decay_reward_biases()
                _loop_nac.decay_goal_reward_biases()
                _loop_nac.decay_cluster_reward_biases()
                # Wire 2 (release_0_9_1.md Stage 3): Pavlovian percept
                # aversion. Without per-tick decay, ``_percept_valences``
                # ages into permanent fossils — ``TextSalienceScorer``
                # would silently treat "burned-by-dragon six sessions
                # ago" as equally salient to "burned-by-dragon last tick."
                _loop_nac.decay_percept_valences()
                # Substrate exploration policy (substrate_exploration_policy.md
                # Phase 2): decay per-(agent, tool) visit counts so a tool the
                # agent stopped selecting regains novelty over time. No-op when
                # exploration is off (empty map → early return).
                _loop_nac.decay_exploration_visits()
            except Exception as e:
                log_swallowed_exception(e, operation="nac_per_tick_decay")

        # ─────────────────────────────────────────────────────────────────
        # 9. MAINTAIN LOOP FREQUENCY
        # ─────────────────────────��───────────────────────────────────────
        elapsed = time.time() - loop_start
        sleep_time = target_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Always persist final state and context pool
    _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})
    try:
        context_pool.save()
    except Exception as e:
        logger.debug(f"Failed to save context pool: {e}")

    # End bio-system session (hippocampus flush/save + MemoryHub session_end).
    # Consolidation flavor is explicit (HANDLE seam b): a caller-supplied
    # `consolidation` override wins (a persistent HANDLE forces "full" even when
    # driven through the sim loop); absent an override we derive it from the sim
    # flag at THIS call site — the choice is explicit at end_bio_session, not
    # inferred from a proxy flag inside it. Default when neither: "full".
    _resolved_consolidation = (
        consolidation if consolidation is not None else ("lightweight" if sim.is_sim_mode else "full")
    )
    _end_bio_session(
        memory_hub=memory_hub,
        memory_hub_enabled=memory_hub_enabled,
        hippocampus=hippocampus,
        consolidation=_resolved_consolidation,
    )

    # Stop Default Network if running (skip in sim — no DN)
    if dn_enabled and not sim.is_sim_mode:
        ctrl.dn_ctrl.stop()

    # D13 planning liveness: raise AFTER teardown so state persistence and
    # the bio session end ran, but the sim still aborts loudly (the
    # orchestrator converts this into a llm_wedged finish report) instead of
    # having idled forever on a silently-dropped planning turn.
    if _planning_liveness_exhausted:
        from maxim.runtime.loop_controller import PlanningLivenessExhausted

        raise PlanningLivenessExhausted(
            f"{ctrl.planning_failure_streak} consecutive planning-turn failures "
            f"with the retry budget spent (limit {ctrl.planning_retry_limit}) — "
            f"see bugs ledger D13"
        )
