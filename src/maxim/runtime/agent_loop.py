from __future__ import annotations

import collections
import logging
import os
import re
import time
import itertools
import json
from typing import TYPE_CHECKING, Any

from maxim.evaluation.base import Evaluator
from maxim.utils.logging import log_swallowed_exception, warn
from maxim.utils.structured_logging import log_agentic

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


from maxim.runtime.approval import detect_approval_intent, _APPROVAL_YES, _APPROVAL_NO


def _safe_agent_name(agent: Any) -> str:
    raw = None
    try:
        raw = getattr(agent, "state_name", None) or getattr(agent, "agent_name", None) or getattr(agent, "name", None)
    except (AttributeError, TypeError) as e:
        log_swallowed_exception(e, operation="get_agent_name")
        raw = None
    if not raw:
        raw = type(agent).__name__
    name = str(raw).strip() or "agent"
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name.strip("._-") or "agent"


from maxim.runtime.loop_state import (
    _persist_state_json,
    _get_failure_strategy,
    _get_plan_depth,
    _build_replan_context,
)


def _record_outcome(
    *,
    tool_name: str,
    success: bool,
    result_summary: str | None,
    error: str | None,
    reasoning: str,
    recent_outcomes: list[dict[str, Any]],
    max_recent: int,
    llm_worker: Any | None,
    context_pool: Any,
) -> None:
    """Record a tool outcome to all three sinks (Phase 0.1 consolidation).

    Appends to recent_outcomes, records reasoning carryover on llm_worker,
    and adds to context_pool.  Previously copy-pasted in ~10 locations.
    """
    recent_outcomes.append(
        {
            "tool": tool_name,
            "success": success,
            "result": result_summary,
            "error": error,
            "timestamp": time.time(),
        }
    )
    if len(recent_outcomes) > max_recent:
        recent_outcomes.pop(0)

    if llm_worker is not None:
        llm_worker.record_outcome(
            tool_name=tool_name,
            reasoning=reasoning,
            success=success,
            result_summary=(result_summary or "")[:200],
        )

    context_pool.add_outcome(
        tool_name=tool_name,
        success=success,
        result_summary=result_summary,
        error=error,
    )


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
        except Exception:
            logger.debug("stop_event check failed", exc_info=True)

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
                try:
                    time.sleep(float(idle_sleep_s))
                except Exception:
                    logger.debug("idle_sleep failed", exc_info=True)
                continue

            goal = intent.get("goal") or intent.get("intent")
            if goal is None:
                if break_on_no_intent:
                    break
                try:
                    time.sleep(float(idle_sleep_s))
                except Exception:
                    logger.debug("idle_sleep failed", exc_info=True)
                continue

            decision = decision_engine.decide(goal, state, memory)
            if not isinstance(decision, dict) or not decision.get("action"):
                if break_on_no_intent:
                    break
                try:
                    time.sleep(float(idle_sleep_s))
                except Exception:
                    logger.debug("idle_sleep failed", exc_info=True)
                continue

        action = decision["action"]
        if not isinstance(action, dict):
            warn("Invalid action selected: %r", action)
            if break_on_no_intent:
                break
            try:
                time.sleep(float(idle_sleep_s))
            except Exception:
                logger.debug("idle_sleep failed", exc_info=True)
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
            except Exception:
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
                    replan_ctx = _build_replan_context(intent, action, result, state)
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
    from maxim.agents.llm_worker import ModeInfo, StrategyInfo
    from maxim.modes.definitions import get_mode, TOOL_DESCRIPTIONS
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.loop_types import ActionFollowup
    from maxim.runtime.prefetch import (
        init_prefetcher,
        get_result_cache,
        PrefetchResult,
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

    # Aliases for backward compat — loop body still references these directly.
    # As more sections migrate into controller methods, these will shrink.
    pending_proposal = ctrl.pending_proposal
    pending_next_actions = ctrl.pending_next_actions
    pending_action_followup = ctrl.pending_action_followup
    pending_plan_proposal = ctrl.pending_plan_proposal
    processed_cli_inputs = ctrl.processed_cli_inputs
    recent_outcomes = ctrl.recent_outcomes
    max_recent_outcomes = ctrl.max_recent_outcomes
    agent_states = ctrl.agent_states
    last_surfaced_tools = ctrl.last_surfaced_tools
    pending_prefetch = ctrl.pending_prefetch
    last_llm_submit_time = ctrl.last_llm_submit_time
    llm_submit_interval = ctrl.llm_submit_interval

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

    # Initialize MemoryHub session (restores priors from episodic memory)
    memory_hub_enabled = memory_hub is not None
    if memory_hub_enabled:
        try:
            startup_stats = memory_hub.on_session_start()
            log_agentic(
                "memory_hub",
                "session_start",
                startup_stats,
                level="INFO",
            )
        except Exception as e:
            logger.warning("Failed to start MemoryHub session: %s", e)
            memory_hub_enabled = False

    # Start hippocampus async capture worker (after session_start, which reads synchronously)
    if hippocampus is not None:
        try:
            hippocampus.start_capture_worker()
        except Exception as e:
            logger.debug("Failed to start hippocampus capture worker: %s", e)

    # Diagnostic heartbeat: log once per agent on first iteration + every
    # ~10s thereafter so we can see if a loop is alive but stuck. Silent
    # unless sim mode is active.
    _last_heartbeat_time = [0.0]
    _loop_name = _safe_agent_name(agent)

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
                    f"pending_proposal={'yes' if pending_proposal else 'no'} "
                    f"pending_plan={'yes' if pending_plan_proposal else 'no'} "
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
        if sim.check_exhaustion(pending_proposal):
            break

        # ─────────────────────────────────────────────────────────────────
        # 1. PERCEPTION — via SimulationAdapter or environment
        # ─────────────────────────────────────────────────────────────────
        observation = sim.next_observation(environment, default_network)
        state.update(observation)

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
                if pending_action_followup:
                    _preempted_tool = pending_action_followup.tool
                    pending_action_followup = None
                    logger.info(
                        "Preempted followup chain (%s) for new input: %s",
                        _preempted_tool,
                        cli_text[:60],
                    )
                    sim.log("PIPELINE", f"Preempted {_preempted_tool} followup for new CLI input")
                if pending_proposal and getattr(pending_proposal, "strategy_used", None) in (
                    "multi_step",
                    "fallback",
                ):
                    _preempted_tool = (
                        pending_proposal.action.get("tool_name", "?")
                        if isinstance(pending_proposal.action, dict)
                        else "?"
                    )
                    logger.info(
                        "Preempted pending %s proposal for new input: %s",
                        _preempted_tool,
                        cli_text[:60],
                    )
                    sim.log("PIPELINE", f"Preempted pending {_preempted_tool} proposal for new CLI input")
                    pending_proposal = None

            # ───────────────────────────────────────────────────────────────
            # CONFIRMATION / TIMEOUT / PLAN APPROVAL — delegated to controller
            # ───────────────────────────────────────────────────────────────
            if ctrl.handle_confirmation(cli_text):
                pending_proposal = ctrl.pending_proposal
                pending_action_followup = ctrl.pending_action_followup
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
                pending_proposal = ctrl.pending_proposal
                pending_plan_proposal = ctrl.pending_plan_proposal
                if pending_proposal is None and pending_plan_proposal is None:
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
                    percept = Percept(
                        timestamp=time.time(),
                        source=observation.get("source", "observation"),
                        transcript_chunk=transcript,
                        cli_input=cli_input,
                        has_maxim_keyword=has_keyword,
                    )
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

                # Staleness guard: discard proposals older than LLM timeout + margin
                proposal_age = time.time() - new_proposal.timestamp
                if proposal_age > 35.0:
                    logger.warning(
                        "Skipping stale LLM proposal (age=%.1fs, request_id=%s)",
                        proposal_age,
                        new_proposal.request_id,
                    )
                    sim.log("EXEC", f"DROPPED: stale proposal (age={proposal_age:.1f}s)")
                    new_proposal = None
            # In simulation mode, skip fallback proposals — wait for real LLM
            if new_proposal and sim.should_skip_fallback_proposal(new_proposal):
                logger.info("Sim mode: skipping fallback proposal, waiting for real LLM")
                sim.log("EXEC", "DROPPED: fallback proposal (sim mode)")
                new_proposal = None
            if new_proposal:
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
                    pending_proposal = new_proposal
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
        if pending_proposal is None and pending_next_actions:
            next_action = pending_next_actions.pop(0)
            pending_proposal = LLMProposal(
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

        if pending_proposal is None and not has_pending_llm_input:
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

                                    # Capture episodic memory to Hippocampus (async — fire-and-forget)
                                    if hippocampus is not None:
                                        # Boost salience for surprising outcomes (high RPE)
                                        if hasattr(executor, "get_last_rpe"):
                                            rpe = executor.get_last_rpe()
                                            if rpe > 0.0 and isinstance(observation, dict):
                                                current_salience = observation.get("salience", 0.5)
                                                observation["salience"] = min(1.0, current_salience + rpe * 0.5)
                                        try:
                                            hippocampus.capture_from_loop_async(
                                                observation=observation if isinstance(observation, dict) else {},
                                                state=state,
                                                intent=intent,
                                                decision={"action": action, "confidence": confidence},
                                                action={
                                                    "tool": action["tool_name"],
                                                    "params": action.get("params", {}),
                                                },
                                                result=result,
                                                run_id=run_id or "",
                                            )
                                        except Exception as e:
                                            logger.debug("Hippocampus capture failed: %s", e)

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
                                        tool_name=action["tool_name"],
                                        success=False,
                                        result_summary=None,
                                        error=str(e),
                                        reasoning=getattr(pending_proposal, "reasoning", "")
                                        if pending_proposal
                                        else "",
                                        recent_outcomes=recent_outcomes,
                                        max_recent=max_recent_outcomes,
                                        llm_worker=llm_worker,
                                        context_pool=context_pool,
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
        if pending_proposal and pending_proposal.action:
            action = pending_proposal.action
            confidence = pending_proposal.confidence

            # ───────────────────────────────────────────────────────────────
            # PARALLEL ACTIONS: Execute all together for efficient batching
            # ───────────────────────────────────────────────────────────────
            parallel_actions = getattr(pending_proposal, "parallel_actions", [])
            if parallel_actions:
                # Collect all actions to execute (primary + parallel)
                all_parallel_actions = pending_proposal.get_parallel_actions()
                parallel_results: list[dict[str, Any]] = []
                all_succeeded = True

                logger.info("Executing %d parallel actions for batched exploration", len(all_parallel_actions))
                log_agentic(
                    "agent_loop",
                    "parallel_batch_start",
                    {"count": len(all_parallel_actions), "tools": [a.get("tool_name") for a in all_parallel_actions]},
                )

                for idx, parallel_action in enumerate(all_parallel_actions):
                    tool_name = parallel_action.get("tool_name", "unknown")
                    try:
                        # Check autonomy for each action
                        can_exec, reason = autonomy_controller.can_execute_action(
                            parallel_action, confidence=confidence
                        )
                        if not can_exec:
                            logger.warning("Parallel action %s rejected: %s", tool_name, reason)
                            parallel_results.append(
                                {
                                    "tool": tool_name,
                                    "success": False,
                                    "error": f"Rejected: {reason}",
                                    "result": None,
                                }
                            )
                            continue

                        # Execute the action
                        result = executor.execute(parallel_action)
                        success = getattr(result, "success", True)
                        output = getattr(result, "output", None)
                        error = getattr(result, "error", None)

                        parallel_results.append(
                            {
                                "tool": tool_name,
                                "params": parallel_action.get("params", {}),
                                "success": success,
                                "result": str(output)[:2000] if output else None,
                                "error": error,
                            }
                        )

                        if not success:
                            all_succeeded = False

                        log_agentic(
                            "agent_loop",
                            "parallel_action_complete",
                            {"tool": tool_name, "index": idx, "success": success},
                        )

                    except Exception as e:
                        logger.error("Parallel action %s failed: %s", tool_name, e)
                        parallel_results.append(
                            {
                                "tool": tool_name,
                                "success": False,
                                "error": str(e),
                                "result": None,
                            }
                        )
                        all_succeeded = False

                # Record individual outcomes so LLM has structured history
                for pr in parallel_results:
                    _record_outcome(
                        tool_name=pr["tool"],
                        success=pr["success"],
                        result_summary=pr.get("result"),
                        error=pr.get("error"),
                        reasoning=getattr(pending_proposal, "reasoning", "") if pending_proposal else "",
                        recent_outcomes=recent_outcomes,
                        max_recent=max_recent_outcomes,
                        llm_worker=llm_worker,
                        context_pool=context_pool,
                    )

                # Combine results into a followup for the next LLM call
                log_agentic(
                    "agent_loop",
                    "parallel_batch_complete",
                    {"count": len(parallel_results), "all_succeeded": all_succeeded},
                )

                # Build combined result text for LLM context
                combined_parts = ["=== BATCHED EXPLORATION RESULTS ==="]
                for pr in parallel_results:
                    tool = pr["tool"]
                    if pr["success"]:
                        result_text = pr["result"] or "[no output]"
                        combined_parts.append(f"\n[{tool}] SUCCESS:\n{result_text}")
                    else:
                        combined_parts.append(f"\n[{tool}] FAILED: {pr.get('error', 'unknown error')}")
                combined_parts.append("\n=== END BATCHED RESULTS ===")
                combined_results = "\n".join(combined_parts)

                # Queue this as a followup for the next LLM call
                pending_action_followup = ActionFollowup(
                    tool="batched_exploration",
                    result=combined_results,
                    original_query=pending_proposal.triggering_input,
                    followup_type="process",
                    mode=state.data.get("mode", "exploration"),
                    timestamp=time.time(),
                )
                logger.info("Batched exploration complete, queuing followup for LLM")

                # Clear proposal - will be handled via followup
                pending_proposal = None
                continue  # Skip normal execution flow

            # ───────────────────────────────────────────────────────────────
            # PLANNING MODE: Check if this proposal requires user approval
            # ───────────────────────────────────────────────────────────────
            if getattr(pending_proposal, "requires_approval", False) and pending_proposal.plan_text:
                # In sim mode, auto-resolve plan approval via response policy
                sim_plan_response = sim.resolve_plan_approval(pending_proposal.plan_text)
                if sim_plan_response is not None:
                    sim.log("PIPELINE", f"Auto-resolved plan approval: {sim_plan_response}")
                    if sim_plan_response.lower() in ("yes", "y"):
                        # Auto-approved — skip the approval flow, proceed to execution
                        logger.info("Sim mode: plan auto-approved, executing")
                    else:
                        # Auto-rejected
                        logger.info("Sim mode: plan auto-rejected")
                        pending_proposal = None
                        continue
                else:
                    # Production mode: store and wait for real user
                    logger.info("Proposal requires approval, showing plan to user")
                    log_agentic(
                        "agent_loop",
                        "plan_awaiting_approval",
                        {
                            "tool": action.get("tool_name"),
                            "plan_preview": pending_proposal.plan_text[:100] if pending_proposal.plan_text else None,
                        },
                    )

                    state.data["pending_plan_text"] = pending_proposal.plan_text
                    state.data["pending_plan_tool"] = action.get("tool_name")

                    pending_plan_proposal = pending_proposal
                    pending_proposal = None
                    continue

            logger.info("Executing LLM proposal: tool=%s, confidence=%.2f", action.get("tool_name"), confidence)

            # Log LLM proposal received
            log_agentic(
                "agent_loop",
                "goal_proposed",
                {
                    "tool": action.get("tool_name"),
                    "confidence": confidence,
                    "reasoning": pending_proposal.reasoning[:100] if pending_proposal.reasoning else None,
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
                    goal_desc = pending_proposal.reasoning or ""
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
                        timeout_s = action.get("_timeout_s", 60.0)
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
                        reasoning=pending_proposal.reasoning,
                        mode=state.data.get("mode", "unknown"),
                        confidence=confidence,
                        citations=pending_proposal.citations,
                        outcome="success" if success else "failure",
                        error=getattr(result, "error", None),
                    )

                    # Track outcome for context pool and learning
                    # Get followup type to determine result storage and follow-up behavior
                    tool_name = action.get("tool_name", "")
                    current_mode = state.data.get("mode", "live")

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
                        tool_name=tool_name or "unknown",
                        success=success,
                        result_summary=result_str,
                        error=getattr(result, "error", None),
                        reasoning=getattr(pending_proposal, "reasoning", "") if pending_proposal else "",
                        recent_outcomes=recent_outcomes,
                        max_recent=max_recent_outcomes,
                        llm_worker=llm_worker,
                        context_pool=context_pool,
                    )

                    # Record plan outcome in MemoryHub for learning
                    if memory_hub_enabled and memory_hub is not None:
                        try:
                            goal = pending_proposal.reasoning or ""
                            memory_hub.record_plan_outcome(
                                goal=goal[:200],  # Limit goal length
                                tool_sequence=[tool_name],
                                success=success,
                            )
                        except Exception as e:
                            logger.debug(f"Failed to record plan outcome: {e}")

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
                        triggering_input = getattr(pending_proposal, "triggering_input", "")
                        pending_action_followup = ActionFollowup(
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
                        triggering_input = getattr(pending_proposal, "triggering_input", "")
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
                                "reasoning": pending_proposal.reasoning,
                                "result": getattr(result, "output", None),
                                "success": getattr(result, "success", True),
                            },
                            metadata={"type": "action_execution"},
                        )
                    except Exception as e:
                        log_swallowed_exception(e, operation="memory.store_raw")

                    # Capture episodic memory to Hippocampus (async — fire-and-forget)
                    if hippocampus is not None:
                        # Boost salience for surprising outcomes (high RPE)
                        if hasattr(executor, "get_last_rpe"):
                            rpe = executor.get_last_rpe()
                            if rpe > 0.0 and isinstance(observation, dict):
                                current_salience = observation.get("salience", 0.5)
                                observation["salience"] = min(1.0, current_salience + rpe * 0.5)
                        try:
                            hippocampus.capture_from_loop_async(
                                observation=observation if isinstance(observation, dict) else {},
                                state=state,
                                intent={"goal": pending_proposal.reasoning, "source": "llm_worker"},
                                decision={"action": action, "confidence": confidence},
                                action={"tool": action.get("tool_name"), "params": action.get("params", {})},
                                result=result,
                                run_id=run_id or "",
                            )
                        except Exception as e:
                            logger.debug("Hippocampus capture failed: %s", e)

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
                        reasoning=pending_proposal.reasoning,
                        mode=state.data.get("mode", "unknown"),
                        confidence=confidence,
                        outcome="error",
                        error=str(e),
                    )

                    # Track exception in recent_outcomes for LLM learning
                    _record_outcome(
                        tool_name=action.get("tool_name", "unknown"),
                        success=False,
                        result_summary=None,
                        error=str(e),
                        reasoning=getattr(pending_proposal, "reasoning", "") if pending_proposal else "",
                        recent_outcomes=recent_outcomes,
                        max_recent=max_recent_outcomes,
                        llm_worker=llm_worker,
                        context_pool=context_pool,
                    )

                    # Mark failure in state
                    try:
                        state.mark_failure(str(e))
                    except Exception as mf_err:
                        log_swallowed_exception(mf_err, operation="state.mark_failure_exc")

                pending_proposal = None

            elif autonomy_controller.current_level == AutonomyLevel.PLANNING:
                # Queue for human approval
                proposal = Proposal(
                    id=pending_proposal.request_id,
                    action=action,
                    reasoning=pending_proposal.reasoning,
                    confidence=confidence,
                    strategy_used=pending_proposal.strategy_used,
                    citations=pending_proposal.citations,
                )
                autonomy_controller.proposal_queue.submit(proposal)
                autonomy_controller.log_action(
                    action_type="proposed",
                    action=action,
                    reasoning=pending_proposal.reasoning,
                    mode=state.data.get("mode", "unknown"),
                    confidence=confidence,
                )
                pending_proposal = None

            else:
                # Check if this is a confirmation request (not a hard rejection)
                # Autonomy controller may say "requires approval" or "requires confirmation"
                if reason and ("requires confirmation" in reason.lower() or "requires approval" in reason.lower()):
                    tool_name = action.get("tool_name", "unknown")
                    params = action.get("params", {})

                    confirmation_data = {
                        "action": action,
                        "reasoning": pending_proposal.reasoning,
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
                        # Production mode: print prompt and wait for real user
                        print("\n" + "=" * 60)
                        print("⚠️  ACTION REQUIRES CONFIRMATION")
                        print("=" * 60)
                        print(f"Tool: {tool_name}")
                        print("Parameters:")
                        for key, value in params.items():
                            display_value = str(value)
                            if len(display_value) > 200:
                                display_value = display_value[:200] + "..."
                            print(f"  {key}: {display_value}")
                        print(f"Reasoning: {pending_proposal.reasoning}")
                        print("=" * 60)
                        print("Type 'yes' or 'no' to confirm/reject:")

                        state.data["pending_confirmation"] = confirmation_data
                    # Don't clear pending_proposal yet - we need to wait for response
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
                        tool_name=action.get("tool_name", "unknown"),
                        success=False,
                        result_summary=None,
                        error=rejection_msg,
                        reasoning=pending_proposal.reasoning or "",
                        recent_outcomes=recent_outcomes,
                        max_recent=max_recent_outcomes,
                        llm_worker=llm_worker,
                        context_pool=context_pool,
                    )
                    logger.info("Hard rejection recorded for LLM: %s", rejection_msg)
                pending_proposal = None

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
                            tool_name=tool_name,
                            success=success,
                            result_summary=result_str,
                            error=error_msg,
                            reasoning=proposal.reasoning or "",
                            recent_outcomes=recent_outcomes,
                            max_recent=max_recent_outcomes,
                            llm_worker=llm_worker,
                            context_pool=context_pool,
                        )

                        # Queue follow-up so LLM can continue
                        from maxim.modes.definitions import get_tool_followup_type

                        current_mode = state.data.get("mode", "live")
                        followup_type = get_tool_followup_type(tool_name, current_mode)
                        should_followup = followup_type and (
                            (success and output is not None) or followup_type == "process"
                        )
                        if should_followup:
                            pending_action_followup = ActionFollowup(
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
                            tool_name=tool_name,
                            success=False,
                            result_summary=None,
                            error=str(e),
                            reasoning=proposal.reasoning or "",
                            recent_outcomes=recent_outcomes,
                            max_recent=max_recent_outcomes,
                            llm_worker=llm_worker,
                            context_pool=context_pool,
                        )

        # ─────────────────────────────────────────────────────────────────
        # 6. SUBMIT NEW CONTEXT TO LLM (non-blocking, event-driven)
        # Only trigger LLM when there's something meaningful to respond to
        # ─────────────────────────────────────────────────────────────────
        # Diagnostic: trace why LLM submission is skipped
        if llm_worker and pending_proposal is not None and cli_input and sim.is_sim_mode:
            _pp_tool = (
                pending_proposal.action.get("tool_name", "?") if isinstance(pending_proposal.action, dict) else "?"
            )
            sim.log("PIPELINE", f"LLM gate BLOCKED: pending_proposal={_pp_tool}, new cli_input={str(cli_input)[:40]}")
        if llm_worker and pending_proposal is None:
            now = time.time()
            if now - last_llm_submit_time > llm_submit_interval:
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
                    if pending_action_followup and context is None:
                        from maxim.agents.bus import StructuredContext

                        context = StructuredContext(
                            timestamp=time.time(),
                            mode=state.data.get("mode", "observe"),
                            autonomy_level=state.data.get("autonomy_level", "supervised"),
                            internet_access=state.data.get("internet_access", True),
                            exploration_mode=state.data.get("exploration_mode", False),
                            exploration_focus=state.data.get("exploration_focus", ""),
                        )
                        logger.info("Created minimal context for pending action followup")

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

                        # In exploration mode, check for high novelty (something interesting)
                        is_exploration = state.data.get("exploration_mode", False)
                        if is_exploration and context.current_percept:
                            # Only trigger LLM in exploration if novelty is high
                            if context.current_percept.novelty > 0.7:
                                has_meaningful_input = True
                            # Or if there's a new explore command
                            if context.current_percept.explore_command:
                                has_meaningful_input = True

                        # Check for pending action followup (tools that need LLM processing)
                        if pending_action_followup:
                            has_meaningful_input = True
                            # Inject the action result into CLI inputs so LLM can process
                            followup_query = pending_action_followup.original_query
                            followup_result = pending_action_followup.result or ""
                            followup_tool = pending_action_followup.tool
                            followup_type = pending_action_followup.followup_type
                            followup_mode = pending_action_followup.mode

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
                            pending_action_followup = None

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

                        # Check if exploration mode
                        is_exploration = state.data.get("exploration_mode", False)
                        exploration_focus = state.data.get("exploration_focus", "")

                        if is_exploration:
                            from maxim.modes.definitions import get_exploration_mode_with_policy
                            from maxim.modes.exploration import ExplorationPolicy
                            from maxim.modes.strategies import get_strategy_library

                            # Get exploration policy from state
                            policy_dict = state.data.get("exploration_policy", {})
                            policy = ExplorationPolicy.from_dict(policy_dict) if policy_dict else ExplorationPolicy()

                            # Get the exploration mode definition
                            exploration_mode_def = get_exploration_mode_with_policy(policy)

                            # Build context prompt with focus
                            context_prompt = exploration_mode_def.context_prompt.format(
                                focus=exploration_focus or "general exploration"
                            )

                            # Get available tools for exploration
                            exploration_tools = (
                                exploration_mode_def.get_available_tools(_all_tools) if _all_tools else set()
                            )

                            mode_info = ModeInfo(
                                name="exploration",
                                goal=exploration_mode_def.goal,
                                context_prompt=context_prompt,
                                allowed_tools=exploration_tools,
                                forbidden_tools=exploration_mode_def.forbidden_tools,
                                can_access_filesystem=exploration_mode_def.can_access_filesystem,
                                can_access_network=exploration_mode_def.can_access_network,
                            )

                            # Select exploration strategies
                            strategy_library = get_strategy_library()
                            selected_strategies = []
                            if strategy_library:
                                selected = strategy_library.select_strategies(
                                    exploration_mode_def, context, max_strategies=4
                                )
                                selected_strategies = [
                                    StrategyInfo(
                                        name=s.name,
                                        description=s.description,
                                        approach_prompt=s.approach_prompt,
                                    )
                                    for s in selected
                                ]

                            # Update context with exploration fields
                            if hasattr(context, "exploration_mode"):
                                context.exploration_mode = True
                                context.exploration_focus = exploration_focus
                                context.exploration_session_id = state.data.get("exploration_session_id", "")
                                context.exploration_policy = policy_dict

                            # Log exploration context submission
                            log_agentic(
                                "agent_loop",
                                "exploration_context",
                                {
                                    "focus": exploration_focus,
                                    "session": state.data.get("exploration_session_id", ""),
                                    "strategies": [s.name for s in selected_strategies],
                                },
                            )
                        else:
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
                            )
                            selected_strategies = []

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

                        # Get full tool info for prompt (description, params, example)
                        tool_descriptions = {}
                        for name in available_tools:
                            if name in TOOL_DESCRIPTIONS:
                                tool_descriptions[name] = TOOL_DESCRIPTIONS[name]
                            else:
                                # Dynamic tool (from skill/protocol) — build from Tool instance
                                try:
                                    tool = executor.registry.get(name)
                                    tool_descriptions[name] = {
                                        "description": tool.description,
                                        "params": {
                                            k: f"({v[0].__name__}, default={v[1]!r})"
                                            if isinstance(v, tuple)
                                            else v.__name__
                                            for k, v in getattr(tool, "input_schema", {}).items()
                                        },
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
                            strategies=selected_strategies,
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
                            current_strategy=state.data.get("current_strategy", ""),
                            is_sleeping=is_sleeping,
                            protocol_context=_protocol_context,
                        )
                        last_llm_submit_time = now
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
                            sim.log("EXEC", f"LLM submit: {new_cli_input[:60] if new_cli_input else 'followup'}")

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
                        "pending_proposal": pending_proposal is not None,
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

    # Drain async capture queue and save hippocampus
    if hippocampus is not None:
        try:
            hippocampus.flush(timeout=5.0)
            hippocampus.stop_capture_worker()
        except Exception as e:
            logger.debug(f"Failed to flush hippocampus: {e}")
        try:
            if hasattr(hippocampus, "config") and hippocampus.config.persistence_path:
                hippocampus.save()
                log_agentic("hippocampus", "saved", {"memories": len(hippocampus)}, level="INFO")
        except Exception as e:
            logger.debug(f"Failed to save hippocampus: {e}")

    # End MemoryHub session (runs sleep consolidation and bridge cleanup)
    # Skip session_end in simulation mode — it runs consolidation which
    # can block for a long time and we'll start a new turn immediately
    if memory_hub_enabled and memory_hub is not None and not sim.is_sim_mode:
        try:
            session_stats = memory_hub.on_session_end()
            log_agentic(
                "memory_hub",
                "session_end",
                session_stats,
                level="INFO",
            )
        except Exception as e:
            logger.debug(f"Failed to end MemoryHub session: {e}")

    # Stop Default Network if running (skip in sim — no DN)
    if dn_enabled and not sim.is_sim_mode:
        ctrl.dn_ctrl.stop()
