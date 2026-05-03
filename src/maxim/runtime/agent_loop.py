from __future__ import annotations

import logging
import os
import time
import itertools
from typing import TYPE_CHECKING, Any

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
) -> Any:
    """Block until LLM responds, checking stop_event every 100ms.

    Returns LLMProposal or None on timeout/cancellation.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            return None
        proposal = llm_worker.get_latest_proposal()
        if proposal is not None:
            return proposal
        time.sleep(0.1)
    logger.warning("_wait_for_proposal timed out after %.0fs", timeout)
    return None


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

        proposal = _wait_for_proposal(llm_worker, stop_event)
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
        sim = NullSimulationAdapter()

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

    # Initialize bio-system session (MemoryHub + hippocampus capture worker)
    memory_hub_enabled = _start_bio_session(memory_hub=memory_hub, hippocampus=hippocampus)

    # Diagnostic heartbeat: log once per agent on first iteration + every
    # ~10s thereafter so we can see if a loop is alive but stuck. Silent
    # unless sim mode is active.
    _last_heartbeat_time = [0.0]
    _loop_name = _safe_agent_name(agent)
    _consecutive_llm_fallbacks = 0  # Track consecutive LLM failures for stall detection
    _LLM_STALL_THRESHOLD = 1  # Surface immediately — only one fallback per LLM request

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
        _is_first_step = step_num == 0
        # If we submitted to the LLM recently, we're awaiting a proposal —
        # don't idle-gate or we'll never pick up the result.
        _awaiting_llm = (
            llm_worker is not None
            and ctrl.pending_proposal is None
            and (time.time() - ctrl.last_llm_submit_time) < 120.0
        )

        if not (_has_pending_input or _has_pending_work or _has_sim_percept or _is_first_step or _awaiting_llm):
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
                        pass
            except Exception as e:
                log_swallowed_exception(e, operation="imagination_trigger", context={"step": step_num})

        _auto_sense_text = ""  # populated by section 1.15, set on context at submission

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

                    # Exteroception: sense_presence (what entities are around me?)
                    # KeyError = tool not registered (agent has no exteroception
                    # by config); other exceptions are real failures and get
                    # logged so silent blindness can't hide a bug.
                    _presence = None
                    try:
                        _presence = _tool_reg.get("sense_presence")
                    except KeyError:
                        pass
                    if _presence is not None:
                        try:
                            _presence_result = _presence.execute()
                            if _presence_result.success and _presence_result.output:
                                _auto_sense_parts.append(str(_presence_result.output))
                        except Exception as _exc:
                            log_swallowed_exception(
                                _exc,
                                operation="auto_sense_presence",
                                context={"step": step_num},
                            )

                    # Interoception: sense self-entity (health, stamina, hunger)
                    _sense = None
                    try:
                        _sense = _tool_reg.get("sense")
                    except KeyError:
                        pass
                    if _sense is not None and _presence is not None:
                        try:
                            _emap = getattr(_presence, "_entity_map", None)
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

                    if _auto_sense_parts:
                        _auto_sense_text = "\n".join(_auto_sense_parts)

                        try:
                            from maxim.simulation.sim_logger import sim_log

                            _n_entities = _auto_sense_text.count("[SCENE]") + _auto_sense_text.count("[YOU]")
                            sim_log("PERCEPTION", f"auto-sense: {_n_entities} entities, body state updated")
                        except Exception:
                            pass
            except Exception as _ase:
                log_swallowed_exception(_ase, operation="auto_sense", context={"step": step_num})

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
                    except Exception as _ge:
                        log_swallowed_exception(_ge, operation="thought_gate", context={"step": step_num})
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

                        sim_pre_deliberation(
                            gate_passed=True, score=0.0, threshold=0.0, enrichment_sections=_n_sections
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

                        sim_pre_deliberation(gate_passed=False, score=0.0, threshold=0.0, enrichment_sections=0)
                        _pfc_gate_passed = False
                elif not _pfc_gate_passed and _percept_text_for_cycle:
                    # Only log gate rejection when there was actual percept text to evaluate
                    from maxim.simulation.sim_logger import sim_pre_deliberation

                    sim_pre_deliberation(gate_passed=False, score=0.0, threshold=0.0, enrichment_sections=0)
            except Exception as e:
                log_swallowed_exception(e, operation="pfc_enrichment", context={"step": step_num})

        # Log cycle 1 enrichment outcome + reset refractory.
        # If multi-cycle deliberation runs at section 6, it logs the
        # final outcome separately and resets refractory again (idempotent).
        if _pfc_gate_passed and _percept_enrichment_text:
            from maxim.simulation.sim_logger import sim_contemplation

            sim_contemplation(gate_passed=True, refined=False, score=0.0)
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
                    new_proposal = None
            # In simulation mode, skip fallback proposals — wait for real LLM
            if new_proposal and sim.should_skip_fallback_proposal(new_proposal):
                _consecutive_llm_fallbacks += 1
                logger.info(
                    "Sim mode: skipping fallback proposal (%d consecutive)",
                    _consecutive_llm_fallbacks,
                )
                sim.log("EXEC", f"DROPPED: fallback proposal (sim mode, #{_consecutive_llm_fallbacks})")
                new_proposal = None
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
                                    _record_outcome(
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

                    _record_outcome(
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
                    )

                    # Record plan outcome in MemoryHub for learning
                    if memory_hub_enabled and memory_hub is not None:
                        _record_plan_outcome(
                            memory_hub=memory_hub,
                            goal=ctrl.pending_proposal.reasoning or "",
                            tool_name=tool_name,
                            success=success,
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
                    _record_outcome(
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
                    _record_outcome(
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
                        _record_outcome(
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
                        _record_outcome(
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
        if llm_worker and ctrl.pending_proposal is None:
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

                    # Inject bio-enrichment context if percept passed novelty gate
                    if context is not None and _percept_enrichment_text:
                        context.bio_enrichment_context = _percept_enrichment_text

                    # Auto-sense context (section 1.15) — passive perception
                    if context is not None and _auto_sense_text:
                        context.auto_sense_context = _auto_sense_text

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

                        submitted = llm_worker.submit_context(
                            context=context,
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
                            _first = _wait_for_proposal(llm_worker, stop_event)
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

    # End bio-system session (hippocampus flush/save + MemoryHub session_end)
    _end_bio_session(
        memory_hub=memory_hub,
        memory_hub_enabled=memory_hub_enabled,
        hippocampus=hippocampus,
        is_sim_mode=sim.is_sim_mode,
    )

    # Stop Default Network if running (skip in sim — no DN)
    if dn_enabled and not sim.is_sim_mode:
        ctrl.dn_ctrl.stop()
