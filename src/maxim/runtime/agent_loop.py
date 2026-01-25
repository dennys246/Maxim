from __future__ import annotations

import logging
import os
import re
import time
import itertools
import json
from typing import TYPE_CHECKING, Any

from maxim.evaluation.base import Evaluator
from maxim.runtime.state import RuntimeState
from maxim.utils.logging import warn
from maxim.utils.structured_logging import log_agentic

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController
    from maxim.agents.llm_worker import LLMWorker

# Import LLMProposal for runtime use (multi-step action creation)
from maxim.agents.llm_worker import LLMProposal

logger = logging.getLogger(__name__)


def _safe_agent_name(agent: Any) -> str:
    raw = None
    try:
        raw = (
            getattr(agent, "state_name", None)
            or getattr(agent, "agent_name", None)
            or getattr(agent, "name", None)
        )
    except Exception:
        raw = None
    if not raw:
        raw = type(agent).__name__
    name = str(raw).strip() or "agent"
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name.strip("._-") or "agent"


def _persist_state_json(state: Any, path: str, *, meta: dict[str, Any]) -> None:
    try:
        if hasattr(state, "save_json") and callable(getattr(state, "save_json")):
            try:
                state.save_json(path, meta=meta)
            except TypeError:
                state.save_json(path)
            return
        if hasattr(state, "snapshot") and callable(getattr(state, "snapshot")):
            snap = state.snapshot()
        else:
            snap = {"state": repr(state)}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump({"saved_at": time.time(), **meta, **snap}, fp, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        warn("Failed to persist runtime state: %s", e)


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
            pass

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
            # Record in memory so it appears in context
            if hasattr(memory, "record_command"):
                try:
                    memory.record_command(cli_text)
                except Exception:
                    pass

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
                pass
            continue

        goal = intent.get("goal") or intent.get("intent")
        if goal is None:
            if break_on_no_intent:
                break
            try:
                time.sleep(float(idle_sleep_s))
            except Exception:
                pass
            continue

        decision = decision_engine.decide(goal, state, memory)
        if not isinstance(decision, dict) or not decision.get("action"):
            if break_on_no_intent:
                break
            try:
                time.sleep(float(idle_sleep_s))
            except Exception:
                pass
            continue

        action = decision["action"]
        if not isinstance(action, dict):
            warn("Invalid action selected: %r", action)
            if break_on_no_intent:
                break
            try:
                time.sleep(float(idle_sleep_s))
            except Exception:
                pass
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
        except Exception:
            pass

        try:
            followup = environment.step(result)
            if followup:
                state.update(followup)
        except Exception:
            pass

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
            except Exception:
                pass

        try:
            state.steps_taken += 1
        except Exception:
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
    evaluators: list[Evaluator] | None = None,
    max_steps: int = 0,  # 0 = unlimited
    run_id: str | None = None,
    stop_event: Any | None = None,
    on_step: Any | None = None,
    idle_sleep_s: float = 0.05,  # Fast loop for responsiveness
    persist_every_n_steps: int = 10,
    target_hz: float = 30.0,  # Target loop frequency
    context_pool_config: dict[str, Any] | None = None,  # Context pool configuration
    use_tool_prompting: bool = True,  # Enable tool-aware LLM prompts
) -> None:
    """
    Non-blocking agentic loop with LLM worker integration.

    Key differences from run_agent_loop:
    - Never blocks on LLM inference
    - Checks for LLM proposals asynchronously
    - Applies autonomy level gating before execution
    - Maintains target loop frequency for real-time responsiveness
    - Hard stops work instantly regardless of LLM state

    Args:
        autonomy_controller: Controls what actions can be executed
        llm_worker: Background LLM worker for async inference
        target_hz: Target loop frequency (default 30Hz)
    """
    from maxim.agents.autonomy import (
        AutonomyLevel,
        AutonomyController,
        Proposal,
        check_hard_stop,
    )
    from maxim.agents.context_pool import ContextPool, ContextPoolConfig, AgentStateEntry
    from maxim.agents.llm_worker import FallbackBehavior, ModeInfo, StrategyInfo
    from maxim.modes.definitions import get_mode, TOOL_DESCRIPTIONS

    if evaluators is None:
        evaluators = []

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

    # Track pending proposal from LLM
    pending_proposal: LLMProposal | None = None
    pending_next_actions: list[dict[str, Any]] = []  # Multi-step action queue
    last_llm_submit_time = 0.0
    llm_submit_interval = 0.5  # Don't flood LLM with requests

    # Track processed CLI inputs to avoid duplicate submissions
    processed_cli_inputs: set[str] = set()

    # Track recent outcomes for learning
    recent_outcomes: list[dict[str, Any]] = []
    max_recent_outcomes = 10

    # Track agent states
    agent_states: list[dict[str, Any]] = []

    # Get available tools from executor registry
    all_tools: set[str] = set()
    if hasattr(executor, "registry") and hasattr(executor.registry, "list"):
        try:
            all_tools = set(executor.registry.list())
        except Exception:
            pass

    # Loop timing
    target_period = 1.0 / target_hz
    max_steps_i = int(max_steps or 0)
    step_iter = itertools.count() if max_steps_i <= 0 else range(max_steps_i)

    for step_num in step_iter:
        loop_start = time.time()

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
        except Exception:
            pass

        # Check for shutdown mode - break immediately to stop LLM worker promptly
        current_mode = state.data.get("mode", "")
        if current_mode == "shutdown":
            log_agentic("agent_loop", "shutdown", {"reason": "shutdown_mode"})
            break

        # Check if autonomy is paused
        if autonomy_controller.is_paused:
            time.sleep(idle_sleep_s)
            continue

        # ─────────────────────────────────────────────────────────────────
        # 1. PERCEPTION (fast, always runs)
        # ─────────────────────────────────────────────────────────────────
        observation = environment.observe()
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
        if not cli_input:
            cli_input = state.data.pop("pending_cli_input", None)

        # Check for voice input injected from the transcription thread
        # Voice input only gets forwarded when agentic mode is enabled (wake word was said)
        voice_input = state.data.pop("pending_voice_input", None)
        voice_transcript = state.data.pop("pending_voice_transcript", None)
        is_agentic_voice_input = False

        # Use voice input if no CLI input (CLI takes priority)
        if not cli_input and voice_input:
            cli_input = voice_input
            is_agentic_voice_input = True  # Came from agentic voice session (wake word already said)
            logger.info("Using voice transcript as user input: %s", voice_input[:50] if voice_input else "")

        # Store CLI input in state and memory for LLM processing
        if cli_input:
            cli_text = str(cli_input).strip()
            state.data["pending_user_input"] = cli_text
            source_type = "voice" if is_agentic_voice_input else "CLI"
            logger.warning("Agent loop received %s input: %s", source_type, cli_text[:100])
            log_agentic(
                "agent_loop",
                "user_input_received",
                {"text": cli_text[:100], "source": source_type},
            )
            # Record in memory so it appears in context.cli_inputs
            # For agentic voice input, prepend "maxim: " so LLM worker recognizes it as a command
            if hasattr(memory, "record_command"):
                try:
                    memory_text = cli_text
                    if is_agentic_voice_input and "maxim" not in cli_text.lower():
                        memory_text = f"maxim: {cli_text}"
                    memory.record_command(memory_text)
                    logger.warning("Recorded %s input to memory: %s", source_type, memory_text[:50])
                except Exception as e:
                    logger.warning("Failed to record %s input: %s", source_type, e)

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

        # ─────────────────────────────────────────────────────────────────
        # 1.5 ADD PERCEPT TO CONTEXT POOL
        # ─────────────────────────────────────────────────────────────────
        if observation:
            # Build a percept-like object from observation for context pool
            try:
                from maxim.agents.bus import Percept
                percept = None
                if hasattr(observation, "get"):
                    # Mark as having maxim keyword if:
                    # 1. Text contains "maxim", OR
                    # 2. This is agentic voice input (wake word was already said)
                    has_keyword = bool(cli_input and "maxim" in str(cli_input).lower()) or is_agentic_voice_input
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
            if new_proposal:
                if new_proposal.action:
                    logger.warning("LLM proposal received: tool=%s, confidence=%.2f",
                                   new_proposal.action.get("tool_name"), new_proposal.confidence)
                    pending_proposal = new_proposal
                    # Queue any next_actions for sequential execution
                    if new_proposal.next_actions:
                        pending_next_actions.extend(new_proposal.next_actions)
                        log_agentic(
                            "agent_loop",
                            "multi_step_queued",
                            {"count": len(new_proposal.next_actions)},
                        )
                elif new_proposal.error:
                    logger.warning("LLM proposal error: %s", new_proposal.error)

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
        if pending_proposal is None:
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
                            can_execute, reason = autonomy_controller.can_execute_action(
                                action, confidence=confidence
                            )

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
                                    result = executor.execute(action)

                                    # Log tool execution
                                    success = getattr(result, "success", True)
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
                                    except Exception:
                                        pass

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
                                    except Exception:
                                        pass

                                except Exception as e:
                                    log_agentic(
                                        "agent_loop",
                                        "error",
                                        {"context": "agent_fallback_action", "error": str(e)},
                                        level="ERROR",
                                    )
                                    logger.debug(f"Agent fallback action failed: {e}")
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
                logger.debug(f"Agent propose_intent failed: {e}")

        # ─────────────────────────────────────────────────────────────────
        # 4. EXECUTE PENDING LLM ACTION (if autonomy allows)
        # ─────────────────────────────────────────────────────────────────
        if pending_proposal and pending_proposal.action:
            action = pending_proposal.action
            confidence = pending_proposal.confidence

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

            can_execute, reason = autonomy_controller.can_execute_action(
                action, confidence=confidence
            )

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
                # Execute the action
                try:
                    result = executor.execute(action)
                    success = getattr(result, "success", True)

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
                    outcome = {
                        "tool": action.get("tool_name"),
                        "success": success,
                        "result": str(output)[:100] if output else None,
                        "error": getattr(result, "error", None),
                        "timestamp": time.time(),
                    }
                    recent_outcomes.append(outcome)
                    if len(recent_outcomes) > max_recent_outcomes:
                        recent_outcomes.pop(0)

                    # Add to context pool
                    context_pool.add_outcome(
                        tool_name=action.get("tool_name", "unknown"),
                        success=success,
                        result_summary=str(output)[:100] if output else None,
                        error=getattr(result, "error", None),
                    )

                    # Process result
                    try:
                        followup = environment.step(result)
                        if followup:
                            state.update(followup)
                    except Exception:
                        pass

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
                    except Exception:
                        pass

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
                        except Exception:
                            pass

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
                # Can't execute and not PLANNING - log rejection
                autonomy_controller.log_action(
                    action_type="rejected",
                    action=action,
                    reasoning=f"Rejected: {reason}",
                    mode=state.data.get("mode", "unknown"),
                    confidence=confidence,
                )
                pending_proposal = None

        # ─────────────────────────────────────────────────────────────────
        # 5. CHECK FOR APPROVED PROPOSALS (PLANNING mode)
        # ─────────────────────────────────────────────────────────────────
        if autonomy_controller.current_level == AutonomyLevel.PLANNING:
            approved = autonomy_controller.proposal_queue.get_approved()
            for proposal in approved:
                if proposal.action:
                    try:
                        result = executor.execute(proposal.action)
                        autonomy_controller.log_action(
                            action_type="executed",
                            action=proposal.action,
                            reasoning=proposal.reasoning,
                            mode=state.data.get("mode", "unknown"),
                            confidence=proposal.confidence,
                            human_involved=True,
                            outcome="success" if getattr(result, "success", True) else "failure",
                        )
                    except Exception as e:
                        logger.error(f"Approved action failed: {e}")

        # ─────────────────────────────────────────────────────────────────
        # 6. SUBMIT NEW CONTEXT TO LLM (non-blocking, event-driven)
        # Only trigger LLM when there's something meaningful to respond to
        # ─────────────────────────────────────────────────────────────────
        if llm_worker and pending_proposal is None:
            now = time.time()
            if now - last_llm_submit_time > llm_submit_interval:
                # Build context for LLM
                try:
                    context = None
                    if hasattr(memory, "build_context"):
                        context = memory.build_context()

                    # Check if there's something meaningful to react to
                    # Only submit to LLM if we have:
                    # 1. New CLI input with "maxim" keyword - not already processed
                    # 2. New speech detected (transcription with "maxim" keyword)
                    # 3. Direct address via maxim keyword in recent percepts
                    # 4. Hard override commands
                    # 5. Exploration mode with high novelty detection (periodic checks)
                    has_meaningful_input = False
                    new_cli_input = None
                    if context:
                        # Known commands that should NOT be sent to LLM
                        # These are handled by Selfy's phrase response system
                        SKIP_LLM_COMMANDS = frozenset({
                            "maxim shutdown", "shutdown maxim",
                            "maxim sleep", "sleep maxim",
                            "maxim nap", "maxim rest",
                            "maxim observe", "observe maxim",
                            "maxim watch", "maxim wake", "maxim wake up", "wake up maxim",
                            "maxim explore", "explore maxim",
                            "maxim stop", "stop maxim",
                            "maxim pause", "pause maxim",
                            "maxim resume", "resume maxim",
                        })

                        # Check for NEW CLI input with "maxim" keyword (not already processed)
                        # Only process inputs that address Maxim directly
                        if context.cli_inputs:
                            for cli_input in context.cli_inputs:
                                if cli_input and cli_input not in processed_cli_inputs:
                                    cli_lower = cli_input.lower().strip()
                                    if "maxim" in cli_lower:
                                        # Skip LLM for known commands (handled by Selfy)
                                        if cli_lower in SKIP_LLM_COMMANDS:
                                            logger.info("Skipping LLM for command: %s", cli_input)
                                            processed_cli_inputs.add(cli_input)
                                            continue
                                        new_cli_input = cli_input
                                        has_meaningful_input = True
                                        break
                                    else:
                                        # Mark non-maxim inputs as processed so we don't re-check
                                        processed_cli_inputs.add(cli_input)

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

                    # Skip LLM if nothing to react to
                    if not has_meaningful_input:
                        context = None  # Skip this submission

                    # Mark new CLI input as processed to prevent duplicate submissions
                    if new_cli_input:
                        processed_cli_inputs.add(new_cli_input)
                        # Keep only last 20 processed inputs to prevent memory growth
                        if len(processed_cli_inputs) > 20:
                            processed_cli_inputs.pop()

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
                            exploration_tools = exploration_mode_def.get_available_tools(all_tools) if all_tools else set()

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
                            if mode_def and all_tools:
                                available_tools_for_mode = mode_def.get_available_tools(all_tools)

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
                            new_inputs = [inp for inp in context.cli_inputs if inp not in processed_cli_inputs or inp == new_cli_input]
                            # Only keep the most recent new input
                            if new_cli_input:
                                context.cli_inputs = [new_cli_input]
                            else:
                                context.cli_inputs = new_inputs[-1:] if new_inputs else []

                        # Get available tools for this mode
                        available_tools = mode_info.get_available_tools(all_tools)

                        # Get tool descriptions for prompt
                        tool_descriptions = {
                            name: TOOL_DESCRIPTIONS.get(name, {}).get("description", "")
                            for name in available_tools
                        }

                        # Get context pool text
                        context_pool_text = context_pool.get_context_text(
                            max_tokens=mode_info.context_window_tokens // 2
                        )

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
                        )
                        last_llm_submit_time = now
                        if submitted and new_cli_input:
                            logger.info("Submitted to LLM: %s", new_cli_input[:50] if len(new_cli_input) > 50 else new_cli_input)

                except Exception as e:
                    logger.warning(f"Failed to submit context to LLM: {e}")

        # ─────────────────────────────────────────────────────────────────
        # 7. CALL STEP CALLBACK
        # ─────────────────────────────────────────────────────────────────
        try:
            if callable(on_step):
                on_step({
                    "step": step_num,
                    "state": state,
                    "memory": memory,
                    "autonomy_level": autonomy_controller.current_level.value,
                    "pending_proposal": pending_proposal is not None,
                })
        except Exception:
            pass

        # ─────────────────────────────────────────────────────────────────
        # 8. INCREMENT STEP COUNTER AND PERSIST
        # ─────────────────────────────────────────────────────────────────
        try:
            state.steps_taken += 1
        except Exception:
            pass

        if persist_every_n_steps > 0 and state.steps_taken % persist_every_n_steps == 0:
            _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})

        if state.is_done():
            break

        # ─────────────────────────────────────────────────────────────────
        # 9. MAINTAIN LOOP FREQUENCY
        # ─────────────────────────────────────────────────────────────────
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
