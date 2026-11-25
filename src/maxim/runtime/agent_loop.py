from __future__ import annotations

import itertools
import logging
import os
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from maxim.evaluation.base import Evaluator
from maxim.memory.text import Provenance
from maxim.utils.content_safety import safe_tool_name
from maxim.utils.cancellation import stop_event_registered, yield_requested
from maxim.utils.logging import log_swallowed_exception, warn
from maxim.utils.structured_logging import log_agentic

# Extracted to tool_dispatch.py
from maxim.runtime.tool_dispatch import (
    book_agent_fallback_failure,
    safe_agent_name as _safe_agent_name,
    execute_parallel_actions as _execute_parallel,
    log_tool_result as _log_tool_result,
)

# Extracted to bio_integration.py
from maxim.runtime.bio_integration import (
    BioSession,
    bio_session,
    capture_loop_action,
)
from maxim.runtime.loop_gates import GateOutcome, pre_tick_gate
from maxim.runtime.loop_intake import take_llm_proposal, take_queued_next_action

# Module reference: §4 reaches the roster filter through its home, the one binding a patch must hit.
from maxim.runtime import loop_state as _lstate
from maxim.runtime.loop_request import log_submit_blocked, submit_tick
from maxim.runtime.loop_setup import RESPONSE_POLICY_UNSET, build_loop_run
from maxim.runtime.loop_external import (
    ExternalInbox,
    inbound_comms_subscription,
)
from maxim.runtime.loop_controller import (
    note_system_outcome,
    set_aside_run_end,
)
from maxim.runtime.loop_substrate import substrate_tick
from maxim.runtime.loop_planning import drain_approved
from maxim.runtime.loop_perception import (
    is_tool_followup,
    perceive,
    percept_enrichment_context,
    percept_text,
    take_observation,
    take_user_input,
)

# Module reference, not a name import: a test that patches the leaf also reaches the loop's call.
from maxim.runtime import substrate_proposal as _sp

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController
    from maxim.agents.llm_worker import LLMWorker
    from maxim.utils.deadline import EndBound

# The streaming event type (§4 tool_start/tool_end, §6 inference_start)
from maxim.agents.bus import StreamEvent

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


from maxim.runtime.loop_state import (
    loop_liveness_scope,
    operational_mode,
    seed_runtime_run_mode,
    start_run_state,
    _planning_attempt_is_active,
    _substrate_tick_due,
    _persist_state_json,
    _get_failure_strategy,
    _get_plan_depth,
    _build_replan_context,
)


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

    agent_name = _safe_agent_name(agent)
    run_id, state_path = start_run_state(state, agent_name, run_id)

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

            proposed_intent: dict[str, Any] | None = None
            try:
                if hasattr(agent, "propose_intent"):
                    proposed_intent = agent.propose_intent(state, memory)
                elif hasattr(agent, "decide"):
                    # Legacy fallback: treat `decide()` as a goal provider.
                    out = agent.decide(state, memory)
                    if isinstance(out, dict):
                        proposed_intent = out
                    elif isinstance(out, str) and out:
                        proposed_intent = {"goal": out, "confidence": 1.0}
            except Exception as e:
                warn("Agent propose_intent/decide failed: %s", e)
                proposed_intent = None

            if not isinstance(proposed_intent, dict) or not proposed_intent:
                if break_on_no_intent:
                    break
                _idle_sleep(idle_sleep_s)
                continue
            intent = proposed_intent

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

        ctx: dict[str, Any] = {
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
            if ctrl is not None:  # as §2's poll does on ANY proposal (#1743: one intake, both halves)
                ctrl.last_proposal_time = time.time()
                ctrl.reset_planning_transport_failures()
            return proposal
        time.sleep(0.1)
    logger.warning("_wait_for_proposal timed out after %.0fs", timeout)
    return None


def _planning_submit_in_flight(llm_worker: Any, liveness_on: bool) -> bool:
    """Whether this loop's previous planning job can still publish a proposal (one planning request in flight per
    narrator, #1048). Only loops with planning liveness (the sim narrator) wait; the AUT is unaffected. Inputs that
    arrive meanwhile ride the next submit (``loop_submit._take_deferred_inputs``)."""
    attempt_state = getattr(llm_worker, "latest_attempt_state", None)
    held = bool(liveness_on and callable(attempt_state) and _planning_attempt_is_active(attempt_state()))
    if callable(attempt_state) and held != _PLANNING_HELD.get(id(llm_worker), False):  # on a change: asked every tick
        _PLANNING_HELD[id(llm_worker)] = held
        # With "received CLI input" and "Submitted", tells an input that waited its turn from a stranded one (#1488).
        logger.debug("Planning job %s (#1048)", "in flight: new inputs wait" if held else "done: inputs are free")
    return held


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


def _submit_held_or_stopped(llm_worker: Any, liveness_on: bool, stop_event: Any) -> bool:
    """Whether this tick must not submit: a planning job is still in flight (#1048), or the run has stopped. A tick
    whose action ended the run (the narrator's finish sets the stop, #1259) would otherwise still submit the inputs
    held behind it: an LLM call nobody reads (#1451)."""
    try:
        stopped = stop_event is not None and callable(getattr(stop_event, "is_set", None)) and stop_event.is_set()
    except (AttributeError, RuntimeError) as exc:  # the pre-tick gate (loop_gates §0) reads a broken event as clear
        logger.debug("stop_event.is_set() failed at the submit gate (read as not stopped): %s", exc)
        stopped = False
    # The process stop request too (#1438 PR 1b), as at the pre-tick gate: a loop given no stop event yields here.
    return bool(stopped) or yield_requested() or _planning_submit_in_flight(llm_worker, liveness_on)


# Each worker's last planning-held state, for the change-only log above. Keyed by id(): a run builds a handful of
# workers, and an id reused after one is collected costs at most one missing or extra DEBUG line.
_PLANNING_HELD: dict[int, bool] = {}


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


def _add_deliberation_thought(
    working_memory: Any, *, cycle: int, enrichment: str, salience: float, goal: str | None
) -> None:
    """One deliberation cycle's enrichment as a THOUGHT in working memory: the agent's own composed text
    (``Provenance.INTERNAL``), whose outside words are already framed inside it (#1519)."""
    import uuid

    from maxim.agents.working_memory import WorkingMemoryKind

    working_memory.add(
        WorkingMemoryKind.THOUGHT,
        ref=f"thought:{uuid.uuid4().hex[:8]}",
        content={"source": "pfc_deliberation", "cycle": cycle, "enrichment": enrichment[:500]},
        salience=salience,
        goal_tag=goal,
        provenance=Provenance.INTERNAL,
    )


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
            _add_deliberation_thought(
                working_memory, cycle=cycle, enrichment=formatted, salience=salience, goal=active_goal
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
            context.working_memory_thoughts = [e.prompt_text(limit=200) for e in thought_entries]  # #1519

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
            return proposal  # one with no action is a declined turn: the intake drops it (#1743)

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

    return last_proposal  # as above: the intake (loop_intake.accept_llm_proposal) drops one with no action (#1743)


# ─────────────────────────────────────────────────────────────────────────────
# Extracted numbered sections of ``run_agentic_loop``
# (docs/plans/archive/god_function_decomposition.md — mechanical moves only)
#
# Each ``_loop_<slug>`` below is one numbered section banner lifted verbatim out
# of the loop body and given a name. The context object is the ``LoopController``
# the loop already builds (``ctrl``); genuinely loop-local values are passed
# keyword-only.
#
# The alias question the plan flags ("no local aliases for mutable state") was
# checked by AST before the move, not assumed: within ``run_agentic_loop``,
# ``state``, ``memory``, ``on_step`` and ``persist_every_n_steps`` are NEVER
# rebound, and ``autonomy_controller`` is bound once BEFORE ``ctrl`` is
# constructed — so ``ctrl.<field> is <local>`` holds for the whole run and
# reading through ``ctrl`` is the same object, not a snapshot.
#
# Section ORDER is load-bearing and unchanged: 7 still runs before 8, and 8.5
# still runs after 8's persist-and-break.
# ─────────────────────────────────────────────────────────────────────────────


def _loop_step_callback(ctrl: Any, *, step_num: int) -> None:
    """Section 7 — CALL STEP CALLBACK.

    Hands the host one per-step snapshot. Deliberately total: a caller's
    callback raising must not take down the loop, so the handler stays broad
    and logs at DEBUG (unchanged from the inline form).
    """
    try:
        if callable(ctrl.on_step):
            ctrl.on_step(
                {
                    "step": step_num,
                    "state": ctrl.state,
                    "memory": ctrl.memory,
                    "autonomy_level": ctrl.autonomy_controller.current_level.value,
                    "ctrl.pending_proposal": ctrl.pending_proposal is not None,
                }
            )
    except Exception:
        logger.debug("on_step callback failed", exc_info=True)


#: Section 8.5's per-tick NAc decays, in call order. Each runs on its own (one
#: raising never skips the rest). ``decay_percept_valences`` — Wire 2
#: (release_0_9_1.md Stage 3): without it ``_percept_valences`` ages into
#: permanent fossils, and ``TextSalienceScorer`` would treat "burned-by-dragon
#: six sessions ago" as salient as "last tick". ``decay_exploration_visits`` —
#: substrate_exploration_policy.md Phase 2: a tool the agent stopped selecting
#: regains novelty; a no-op when exploration is off.
_NAC_PER_TICK_DECAYS: tuple[str, ...] = (
    "decay_eligibility",
    "decay_reward_biases",
    "decay_goal_reward_biases",
    "decay_cluster_reward_biases",
    "decay_percept_valences",
    "decay_exploration_visits",
)


def _loop_bio_tick_maintenance(nac: Any) -> None:
    """Section 8.5 — BIO-SYSTEM PER-TICK MAINTENANCE.

    NAc eligibility traces and reward biases decay each tick. Without this,
    traces persist indefinitely and ``distribute_reward`` credits ALL nodes
    ever activated in the session at original strength — incorrect for causal
    credit assignment.

    Each decay in ``_NAC_PER_TICK_DECAYS`` is contained separately and
    reported in the Stage-1 form under its own site, so one raising no longer
    skips every decay after it, every tick, behind a DEBUG line.

    Takes the NAc directly rather than re-deriving it from
    ``ctrl.memory_hub``: the loop binds ``_loop_nac`` once, before the loop,
    and re-deriving per tick would be a semantic change, not a move.
    """
    if nac is None:
        return
    for name in _NAC_PER_TICK_DECAYS:
        try:
            getattr(nac, name)()
        except Exception:
            log_swallowed_exception(site=f"agent_loop.py:_loop_bio_tick_maintenance:{name}")


_MISSING_CLOSE_HOOK_WARNED: set[tuple[type, str]] = set()


def _fallback_close_hook(agent: Any, intent: Mapping[str, Any], hook: str) -> tuple[str, Any] | None:
    """The agent's ``hook`` and the intent's call id, when §3 has a goal call to close; else None.

    An intent with no ``tool_call_id`` (not from the goal agent) has nothing to close. An intent WITH one, on an agent
    that lacks the hook, breaks the contract: its call would be proposed again on every pass (#1414), so it logs one
    WARNING per agent type and hook instead of closing nothing in silence.
    """
    tool_call_id = intent.get("tool_call_id")
    if not tool_call_id:
        return None
    fn = getattr(agent, hook, None)
    if callable(fn):
        return str(tool_call_id), fn
    key = (type(agent), hook)
    if key not in _MISSING_CLOSE_HOOK_WARNED:
        _MISSING_CLOSE_HOOK_WARNED.add(key)
        logger.warning(
            "agent %s proposed a goal tool call (%s) but has no %s(): the call is never closed, so it is "
            "proposed again on every pass (#1414)",
            type(agent).__name__,
            tool_call_id,
            hook,
        )
    return None


def _close_fallback_call(
    agent: Any, intent: Mapping[str, Any], action: Mapping[str, Any], tool_output: Any = None, *, error: Any = None
) -> None:
    """Close the goal agent's tool call that the agent-fallback path (§3) ran or that raised (#1414).

    ``AgenticGoalAgent.propose_intent`` returns its pending tool call until it learns that call's fate. Before #1414
    nothing told it, so a call §3 ran was proposed again on the next pass, and a call autonomy refused was refused
    again on every pass (pulse saw one refusal 87 times in 44 s). A REFUSED call is closed by
    :func:`_fallback_rejected` instead. The close is a direct call (``agent.close_tool_call``), not a ``ToolResult``
    on the bus: nothing publishes one yet, and doing it from §3 alone would wake ``ToolResult``'s dormant
    subscribers for a biased slice of calls (owner decision 2026-11-21; #1727).

    §3 calls this right after ``executor.execute`` returns, so a ``tool_output`` means the call is already closed: a
    later raise in §3's try reaches the ``except`` with one, and nothing is closed twice. With no ``tool_output`` the
    execute itself raised, and the call closes as failed with ``error``.
    """
    if error is not None and tool_output is not None:
        return
    found = _fallback_close_hook(agent, intent, "close_tool_call")
    if found is None:
        return
    tool_call_id, close = found
    failure = error if error is not None else getattr(tool_output, "error", None)
    close(
        tool_call_id,
        error is None and bool(getattr(tool_output, "success", True)),
        getattr(tool_output, "output", None),
        None if failure is None else str(failure),
    )


def _fallback_rejected(
    agent: Any, intent: Mapping[str, Any], action: Mapping[str, Any], reason: Any, confidence: float
) -> None:
    """§3's autonomy refusal of the agent-fallback action: logged, and the goal agent's call is closed as REFUSED
    (``refuse_tool_call``), so it is not proposed again on the next pass (#1414). The goal ends as a failed
    ``GoalCompleted`` with ``refused=True`` unless its sub-goal is SKIP: memory, the goal metric and the plan's
    replanning see a failed goal, and ExecAgent's contemplation learning skips it."""
    log_agentic(
        "agent_loop",
        "action_rejected",
        {"tool": action["tool_name"], "reason": reason, "confidence": confidence},
    )
    found = _fallback_close_hook(agent, intent, "refuse_tool_call")
    if found is not None:
        tool_call_id, refuse = found
        refuse(tool_call_id, f"Rejected by autonomy: {reason}")


class _LoopOwner:
    """One agent-loop invocation's identity as a memory-session owner (an object, compared with ``is``)."""

    def __init__(self, run_id: str | None) -> None:
        self.run_id = run_id

    def __repr__(self) -> str:
        return f"<agent loop {self.run_id or 'unnamed'}>"


@loop_liveness_scope  # one liveness entry per run for the heartbeat's stall rule, cleared in a finally (#1268)
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
    response_policy: Any = RESPONSE_POLICY_UNSET,  # a sim's ResponsePolicy, or None: required with percept_source
    action_sink: Any | None = None,  # ActionSink for recording tool outputs
    pain_bus: Any | None = None,  # PainBus for simulation pain routing
    imagination_trigger: Any | None = None,  # ImaginationTrigger for real-time entity design
    bio_enrichment_pipeline: Any | None = None,  # BioEnrichmentPipeline for percept enrichment (L1)
    thought_gate: Any | None = None,  # ThoughtGate for PFC deliberation gating
    aut_mode: str = "llm-primary",  # "llm-primary" | "substrate-primary" — Phase -1 of grounded_language_acquisition.md
    substrate_telemetry: Any
    | None = None,  # SubstrateTelemetry writer (Phase 0). Called after each substrate-primary tick when set.
    substrate_policy: _sp.SubstratePolicy | None = None,  # in place of recommend_action (B1); a raise ends the loop
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
    end_bound: EndBound | None = None,  # #1536, wrapper-only: the bound a per-handle stop hands this run's session end
    state_home: str | os.PathLike[str] | None = None,  # #1264: the state file's root (data-home-shaped)
) -> None:
    """The agent loop: a thin wrapper over ``_run_agentic_loop_body``, which holds the loop and its full docstring.

    Split verbatim (docs/plans/bounded_stop.md §3 step 0) so session ownership can wrap the body in one seam
    (``bio_session``, PR 0). Every argument is forwarded unchanged and the signature is the body's, except
    ``end_bound`` (#1536), which only this wrapper's ``bio_session`` reads. The run's inbound comms are held for
    exactly its lifetime (#1481)."""
    # ONE seam owns the run's memory session (#1438 PR 0): claimed before the body's setup starts anything, ended on
    # every exit path. The owner is a fresh token per invocation, never a label. The loop's stop event is registered
    # with the process stop request for the run, so a stop ends the loop at its next tick (#1438 PR 1b).
    with (
        stop_event_registered(stop_event),
        bio_session(
            memory_hub=memory_hub,
            hippocampus=hippocampus,
            owner=_LoopOwner(run_id),
            consolidation=consolidation,
            end_bound=end_bound,
        ) as session,
        inbound_comms_subscription(agent) as inbound_inbox,
    ):
        deferred = _run_agentic_loop_body(
            agent,
            environment,
            state,
            memory,
            decision_engine,
            executor,
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
            context_pool_config=context_pool_config,
            use_tool_prompting=use_tool_prompting,
            protocol_registry=protocol_registry,
            percept_source=percept_source,
            response_policy=response_policy,
            action_sink=action_sink,
            pain_bus=pain_bus,
            imagination_trigger=imagination_trigger,
            bio_enrichment_pipeline=bio_enrichment_pipeline,
            thought_gate=thought_gate,
            aut_mode=aut_mode,
            substrate_telemetry=substrate_telemetry,
            substrate_policy=substrate_policy,
            substrate_action_gate=substrate_action_gate,
            consolidation=consolidation,
            planning_liveness=planning_liveness,
            sim_adapter=sim_adapter,
            state_home=state_home,
            session=session,
            inbound_inbox=inbound_inbox,
        )
    # D13 (bugs ledger): raised only AFTER the session ended, so an aborted sim still persisted and consolidated.
    if deferred is not None:
        raise deferred


def _loop_timing(target_hz: float, max_steps: int | None) -> tuple[float, Any]:
    """The loop's target period and its step iterator: unbounded when ``max_steps`` is 0 or None (moved verbatim out
    of ``_run_agentic_loop_body``, #1264 rule T)."""
    target_period = 1.0 / target_hz
    max_steps_i = int(max_steps or 0)
    step_iter = itertools.count() if max_steps_i <= 0 else range(max_steps_i)
    return target_period, step_iter


def _persist_final_state(state: Any, state_path: str, context_pool: Any, *, run_id: str, agent_name: str) -> None:
    """The loop's teardown persist: the run's state file and its context pool (moved verbatim out of
    ``_run_agentic_loop_body``, #1264 rule T)."""
    _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})
    try:
        context_pool.save()
    except Exception as e:
        logger.debug(f"Failed to save context pool: {e}")


def _run_agentic_loop_body(
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
    response_policy: Any = RESPONSE_POLICY_UNSET,  # a sim's ResponsePolicy, or None: required with percept_source
    action_sink: Any | None = None,  # ActionSink for recording tool outputs
    pain_bus: Any | None = None,  # PainBus for simulation pain routing
    imagination_trigger: Any | None = None,  # ImaginationTrigger for real-time entity design
    bio_enrichment_pipeline: Any | None = None,  # BioEnrichmentPipeline for percept enrichment (L1)
    thought_gate: Any | None = None,  # ThoughtGate for PFC deliberation gating
    aut_mode: str = "llm-primary",  # "llm-primary" | "substrate-primary" — Phase -1 of grounded_language_acquisition.md
    substrate_telemetry: Any
    | None = None,  # SubstrateTelemetry writer (Phase 0). Called after each substrate-primary tick when set.
    substrate_policy: _sp.SubstratePolicy | None = None,  # in place of recommend_action (B1); a raise ends the loop
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
    state_home: str | os.PathLike[str] | None,  # #1264: the state file's root (data-home-shaped)
    session: BioSession,  # the run's memory session, claimed by run_agentic_loop's bio_session (#1438 PR 0)
    inbound_inbox: ExternalInbox | None = None,  # #1481: the run's held inbound comms (wrapper-supplied)
) -> Exception | None:
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
        check_hard_stop,
    )
    from maxim.runtime.loop_types import ActionFollowup

    # The setup block (1.3.2 decomposition, slice 1): built once, unpacked into the local names the
    # body below has always read. ``ctrl`` stays the only mutable carrier (runtime/loop_setup.py).
    run = build_loop_run(
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
        context_pool_config=context_pool_config,
        use_tool_prompting=use_tool_prompting,
        protocol_registry=protocol_registry,
        percept_source=percept_source,
        response_policy=response_policy,
        action_sink=action_sink,
        pain_bus=pain_bus,
        aut_mode=aut_mode,
        planning_liveness=planning_liveness,
        sim_adapter=sim_adapter,
        session=session,
        state_home=state_home,
    )
    executor, sim, ctrl, autonomy_controller = run.executor, run.sim, run.ctrl, run.autonomy_controller
    run_id, agent_name, state_path = run.run_id, run.agent_name, run.state_path
    context_pool, prefetcher, result_cache = run.context_pool, run.prefetcher, run.result_cache
    _is_novel_thought = run.is_novel_thought
    _max_response_tokens_override, _max_cycles_override = run.max_response_tokens_override, run.max_cycles_override
    _loop_nac, _loop_xclock, _loop_agent_id = run.nac, run.xclock, run.agent_id
    _drive_relief_only, _rec_outcome = run.drive_relief_only, run.rec_outcome
    _loop_sensor_encoder, _loop_situation_cue = run.sensor_encoder, run.situation_cue
    _planning_liveness_on = run.planning_liveness_on
    _execute_and_learn, _book_refusal = run.execute_and_learn, run.book_refusal
    _book_machine_refusal, _tool_overlay = run.book_machine_refusal, run.tool_overlay
    _book_hard_rejection = run.book_hard_rejection

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
    target_period, step_iter = _loop_timing(target_hz, max_steps)

    # Diagnostic heartbeat: log once per agent on first iteration + every
    # ~10s thereafter so we can see if a loop is alive but stuck. Silent
    # unless sim mode is active.
    _last_heartbeat_time = [0.0]
    _loop_name = _safe_agent_name(agent)
    _consecutive_llm_fallbacks = 0  # Track consecutive LLM failures for stall detection
    # D13 planning liveness: set when the bounded retry budget is spent; the
    # loop breaks through NORMAL teardown (state persist + bio session end)
    # and raises PlanningLivenessExhausted afterwards, so the sim aborts
    # loudly instead of idling forever on a dropped planning turn.
    _planning_liveness_exhausted = False

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
        # 0-0.6 PRE-TICK GATE: stop checks, DN mode, pause, live tick, display
        # revert, exhaustion, idle gate + D13 backstop (runtime/loop_gates.py)
        # ─────────────────────────────────────────────────────────────────
        _gate = pre_tick_gate(
            step_num=step_num,
            stop_event=stop_event,
            state=state,
            executor=executor,
            ctrl=ctrl,
            autonomy_controller=autonomy_controller,
            sim=sim,
            percept_source=percept_source,
            llm_worker=llm_worker,
            planning_liveness_on=_planning_liveness_on,
            aut_mode=aut_mode,
            experience_driver=_loop_xclock,
            idle_sleep_s=idle_sleep_s,
        )
        if _gate is GateOutcome.EXHAUSTED:
            _planning_liveness_exhausted = True
            break
        if _gate is GateOutcome.BREAK:
            break
        if _gate is GateOutcome.IDLE:
            continue
        if _gate is not GateOutcome.RUN:  # a new GateOutcome member must not silently act as RUN
            raise AssertionError(f"unhandled GateOutcome: {_gate!r}")

        # ─────────────────────────────────────────────────────────────────
        # 1. PERCEPTION — via SimulationAdapter or environment
        # ─────────────────────────────────────────────────────────────────
        observation = take_observation(sim=sim, ctrl=ctrl)

        # ─────────────────────────────────────────────────────────────────
        # 1.1-1.16 IMAGINATION, AUTO-SENSE, AUDIO ORIENTATION (runtime/loop_perception.py)
        # ─────────────────────────────────────────────────────────────────
        _perception = perceive(
            step_num=step_num,
            observation=observation,
            state=state,
            sim=sim,
            executor=executor,
            aut_mode=aut_mode,
            imagination_trigger=imagination_trigger,
        )
        _imagination_results = _perception.imagination_results  # §1.2's EnrichmentContext.resolved_entities
        _auto_sense_text = _perception.auto_sense_text  # §6's context.auto_sense_context
        _audio_escalate_this_tick = _perception.audio_escalate  # §6's B1 minimal context + has_meaningful_input

        # ─────────────────────────────────────────────────────────────────
        # 1.2 BIO-ENRICHMENT — PFC deliberation (gate + enrich)
        # ─────────────────────────────────────────────────────────────────
        # Phase 1 of the PFC cycle: extract percept text and run
        # ThoughtGate + BioEnrichment.  The multi-cycle deliberation
        # (LLM calls) happens later at the submission point (section 2)
        # where StructuredContext is available.
        # Ensure maxim_runtime carries the RUN mode for MemoryAgent (PerceptionAgent -> MemoryAgent -> ExecAgent):
        # before the input's arrival percept, which carries it
        if "maxim_runtime" not in state.data:
            state.data["maxim_runtime"] = {}
        seed_runtime_run_mode(state)  # keeps a mode already there (#963 Q5: the run mode, not the operational one)

        # The tick's input, read ONCE from every channel and memorized as it arrives (Session R): §1.2 enriches it
        _user_input = take_user_input(observation, state, agent, sim_mode=sim.is_sim_mode)
        transcript, hard_override, cli_input, is_agentic_voice_input, _addressed = _user_input[:5]
        _percept_enrichment_text = ""
        _percept_text_for_cycle = ""
        _pfc_gate_passed = False
        _wms = None
        if bio_enrichment_pipeline is not None or thought_gate is not None:
            try:
                _percept_text_for_cycle = percept_text(observation)  # what the gate (deliberation) reads, as before
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
                _enrich_text = _percept_text_for_cycle or _addressed  # a mailbox input (voice, CLI, API) too (D5)
                if (_pfc_gate_passed or _addressed) and bio_enrichment_pipeline is not None and _enrich_text:
                    _enrich_ctx = percept_enrichment_context(state, _imagination_results, _user_input)
                    _enrich_result = bio_enrichment_pipeline.enrich(  # recall-only when only addressed (D5)
                        _enrich_text,
                        context=_enrich_ctx,
                        bypass_gate=True,
                        evaluate_reflexes=_pfc_gate_passed,
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
                        from maxim.simulation.sim_logger import sim_log, sim_pre_deliberation

                        # Real numbers here too — a PASS logged as 0.00/0.00 is
                        # just as uninformative as a rejection was.
                        sim_pre_deliberation(  # the gate's real decision: an addressed input is enriched either way
                            gate_passed=_pfc_gate_passed,
                            score=_gate_score,
                            threshold=_gate_threshold,
                            enrichment_sections=_n_sections,
                            reason="" if _pfc_gate_passed else f"{_gate_reason or 'gate closed'}; recall only",
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
                        _max_cyc_for_display = (
                            _max_cycles_override
                            if _max_cycles_override is not None
                            else (3 if state.data.get("percept_source") else 2)
                        )
                        # NOTE: don't push percept text to thinking panel here —
                        # the percept is an INPUT, not a thought. The AUT's actual
                        # reasoning will be pushed after the LLM responds (section 6).
                        sim_log(
                            "THOUGHT",
                            f"cycle 1: salience={_salience_c1:.2f}, sections={_n_sections}, memories={_n_memories_c1}",
                        )
                        # Add THOUGHT to working memory with computed salience
                        if _wms is not None and _percept_enrichment_text:
                            _c1_goal = state.data.get("active_goal") if hasattr(state, "data") else None
                            _add_deliberation_thought(
                                _wms, cycle=1, enrichment=_percept_enrichment_text, salience=_salience_c1, goal=_c1_goal
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
            if not is_tool_followup(cli_text):
                ctrl.note_user_input(cli_text)  # a new user turn, whatever its channel (#1413)
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
            if ctrl.answer_pending_prompt(
                cli_text,
                execute_and_learn=_execute_and_learn,
                book_refusal=_book_refusal,
                book_machine_refusal=_book_machine_refusal,
                observation=observation,
                sim=sim,
                answer_source=_user_input.source,
                answers=_user_input.answers,
            ):
                cli_input = None  # the answer to a pending prompt, never a new turn (#1185)
                continue

            # Store in state and memory (only reached if NOT the answer to a pending prompt)
            ctrl.store_user_input(cli_text, source_type)

            # Speculative pre-fetching
            ctrl.run_prefetch(cli_text)
            pending_prefetch = ctrl.pending_prefetch

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
        _intake = take_llm_proposal(
            llm_worker=llm_worker,
            ctrl=ctrl,
            sim=sim,
            state=state,
            executor=executor,
            step_num=step_num,
            action_sink=action_sink,
            on_event=on_event,
            last_surfaced_tools=last_surfaced_tools,
            pending_next_actions=pending_next_actions,
            _loop_name=_loop_name,
            _planning_liveness_on=_planning_liveness_on,
            _consecutive_llm_fallbacks=_consecutive_llm_fallbacks,
        )  # §2, runtime/loop_intake.py (slice 6)
        _planning_liveness_exhausted = _planning_liveness_exhausted or _intake.exhausted
        _consecutive_llm_fallbacks = _intake.consecutive_llm_fallbacks

        # D13: retry budget spent inside section 2 — leave through the normal
        # teardown path (state persist + bio session end), then raise.
        if _planning_liveness_exhausted:
            break

        take_queued_next_action(ctrl=ctrl, pending_next_actions=pending_next_actions)

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
                                result = None  # per action: a raised execute must not leave an EARLIER stamp (#1200)
                                try:
                                    if callable(on_event):
                                        try:
                                            on_event(StreamEvent("tool_start", {"tool_name": action["tool_name"]}))
                                        except Exception as e:
                                            log_swallowed_exception(e, operation="on_event:tool_start")

                                    result = executor.execute(action)
                                    _close_fallback_call(agent, intent, action, result)

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

                                    _log_tool_result(action["tool_name"], getattr(result, "output", None))

                                    autonomy_controller.log_action(
                                        action_type="executed",
                                        action=action,
                                        reasoning=intent.get("source", "agent_fallback"),
                                        mode=operational_mode(executor, state),
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

                                    capture_loop_action(
                                        hippocampus,
                                        executor,
                                        observation,
                                        state,
                                        intent,
                                        action,
                                        confidence,
                                        result,
                                        run_id,
                                        _loop_agent_id,
                                        proposal=None,  # the agent-fallback path has none
                                    )

                                except Exception as e:
                                    _close_fallback_call(agent, intent, action, result, error=e)
                                    book_agent_fallback_failure(
                                        e,
                                        rec_outcome=_rec_outcome,
                                        agent_id=_loop_agent_id,
                                        action=action,
                                        result=result,
                                        proposal=ctrl.pending_proposal,
                                        recent_outcomes=recent_outcomes,
                                        max_recent=max_recent_outcomes,
                                        llm_worker=llm_worker,
                                        context_pool=context_pool,
                                        nac=_loop_nac,
                                        state=state,
                                    )
                            else:
                                _fallback_rejected(agent, intent, action, reason, confidence)
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
                continue  # Skip normal execution flow

            # ───────────────────────────────────────────────────────────────
            # PLANNING MODE: Check if this proposal requires user approval
            # ───────────────────────────────────────────────────────────────
            if ctrl.gate_plan_approval(  # a plan: run once if approved, parked for the person, or refused
                execute_and_learn=_execute_and_learn,
                book_machine_refusal=_book_machine_refusal,
                observation=observation,
                sim=sim,
            ):
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

        # ─────────────────────────────────────────────────────────────────
        # 5. CHECK FOR APPROVED PROPOSALS (PLANNING mode) -- runtime/loop_planning.py (#1085)
        # ─────────────────────────────────────────────────────────────────
        if autonomy_controller.current_level == AutonomyLevel.PLANNING:
            _approved_followup = drain_approved(
                autonomy_controller=autonomy_controller,
                execute_and_learn=_execute_and_learn,
                book_machine_refusal=_book_machine_refusal,
                executor=executor,
                observation=observation,
                state=state,
                sim=sim,
            )
            if _approved_followup is not None:
                ctrl.pending_action_followup = _approved_followup

        # ─────────────────────────────────────────────────────────────────
        # 6. SUBMIT NEW CONTEXT TO LLM (non-blocking, event-driven)
        # Only trigger LLM when there's something meaningful to respond to
        # ─────────────────────────────────────────────────────────────────
        log_submit_blocked(llm_worker=llm_worker, ctrl=ctrl, cli_input=cli_input, sim=sim)  # runtime/loop_request.py

        # ─────────────────────────────────────────────────────────────────
        # Substrate-primary action generation (Phase -1 of grounded
        # language acquisition). Skips the LLM entirely; calls
        # NAc.recommend_action() with current drives + tool registry. If
        # the substrate has no opinion (no learned bias, no active drive),
        # the tick is IDLE — no random fallback. Mutually exclusive with
        # the LLM submit branch below.
        if _substrate_tick_due(aut_mode, ctrl, ctrl.llm_submit_interval):  # ONE predicate, shared with the idle gate
            # 6b. the substrate tick (runtime/loop_substrate.py)
            substrate_tick(
                step_num=step_num,
                ctrl=ctrl,
                executor=executor,
                sim=sim,
                memory_hub=memory_hub,
                substrate_action_gate=substrate_action_gate,
                substrate_telemetry=substrate_telemetry,
                substrate_policy=substrate_policy,
                _loop_nac=_loop_nac,
                _loop_agent_id=_loop_agent_id,
                _loop_situation_cue=_loop_situation_cue,
                _loop_sensor_encoder=_loop_sensor_encoder,
            )

        _submit_held = _submit_held_or_stopped(llm_worker, _planning_liveness_on, stop_event)  # #1048, #1451
        if aut_mode != "substrate-primary" and llm_worker and ctrl.pending_proposal is None and not _submit_held:
            now = time.time()
            if now - ctrl.last_llm_submit_time > ctrl.llm_submit_interval:
                # Cache tool registry snapshot for this submission (avoids 3 redundant traversals)
                _all_tools = _get_all_tools()

                # Build context for LLM, submit it, deliberate (§6, runtime/loop_request.py; its gate: loop_submit, 7a)
                _carry = submit_tick(
                    memory=memory,
                    ctrl=ctrl,
                    executor=executor,
                    state=state,
                    sim=sim,
                    cli_input=cli_input,
                    processed_cli_inputs=processed_cli_inputs,
                    _audio_escalate_this_tick=_audio_escalate_this_tick,
                    _percept_enrichment_text=_percept_enrichment_text,
                    _auto_sense_text=_auto_sense_text,
                    _loop_nac=_loop_nac,
                    _loop_agent_id=_loop_agent_id,
                    _planning_liveness_on=_planning_liveness_on,
                    _all_tools=_all_tools,
                    _max_response_tokens_override=_max_response_tokens_override,
                    autonomy_controller=autonomy_controller,
                    _tool_overlay=_tool_overlay,
                    context_pool=context_pool,
                    prefetcher=prefetcher,
                    llm_worker=llm_worker,
                    agent_states=agent_states,
                    recent_outcomes=recent_outcomes,
                    use_tool_prompting=use_tool_prompting,
                    protocol_registry=protocol_registry,
                    on_event=on_event,
                    bio_enrichment_pipeline=bio_enrichment_pipeline,
                    inbound_inbox=inbound_inbox,
                    now=now,
                    _pfc_gate_passed=_pfc_gate_passed,
                    stop_event=stop_event,
                    _max_cycles_override=_max_cycles_override,
                    percept_source=percept_source,
                    _wms=_wms,
                    thought_gate=thought_gate,
                    step_num=step_num,
                    _is_novel_thought=_is_novel_thought,
                    _salience_c1=locals().get("_salience_c1"),
                    _run_deliberation_cycles=_run_deliberation_cycles,
                    _wait_for_proposal=_wait_for_proposal,
                    last_surfaced_tools=last_surfaced_tools,
                    pending_prefetch=pending_prefetch,
                    action_sink=action_sink,
                    _consecutive_llm_fallbacks=_consecutive_llm_fallbacks,
                )
                last_surfaced_tools, pending_prefetch = _carry.last_surfaced_tools, _carry.pending_prefetch
                _consecutive_llm_fallbacks = _carry.consecutive_llm_fallbacks  # #1743; its exhaustion: the gate's latch

        # ─────────────────────────────────────────────────────────────────
        # 7. CALL STEP CALLBACK
        # ─────────────────────────────────────────────────────────────────
        _loop_step_callback(ctrl, step_num=step_num)

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
        _loop_bio_tick_maintenance(_loop_nac)

        # ─────────────────────────────────────────────────────────────────
        # 9. MAINTAIN LOOP FREQUENCY
        # ─────────────────────────��───────────────────────────────────────
        elapsed = time.time() - loop_start
        sleep_time = target_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Always persist final state and context pool
    _persist_final_state(state, state_path, context_pool, run_id=run_id, agent_name=agent_name)

    # D13 planning liveness: RETURNED, and raised by run_agentic_loop AFTER its bio_session ends (#1438 PR 0), so
    # state persistence and the bio session end ran, but the sim still aborts loudly (the orchestrator converts this
    # into a typed finish report) instead of having idled forever on a silently-dropped planning turn.
    return _planning_liveness_abort(ctrl) if _planning_liveness_exhausted or ctrl.planning_exhausted else None


def _planning_liveness_abort(ctrl: Any) -> Exception:
    """The D13 abort the loop body returns once planning liveness is exhausted (see bugs ledger D13)."""
    from maxim.runtime.loop_controller import PlanningLivenessExhausted

    return PlanningLivenessExhausted(
        f"planning failures={ctrl.planning_failure_streak} "
        f"(limit {ctrl.planning_retry_limit}), transport failures="
        f"{ctrl.planning_transport_failure_streak} "
        f"(limit {ctrl.planning_transport_retry_limit}); "
        f"last={ctrl.planning_exhausted_reason or 'unknown'} — see bugs ledger D13",
        finish_status=ctrl.planning_exhausted_status,
    )
