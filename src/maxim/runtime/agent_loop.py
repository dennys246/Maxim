from __future__ import annotations

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


# ─────────────────────────────────────────────────────────────────────────────
# PLANNING MODE APPROVAL DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# Keywords for approval detection (case-insensitive)
_APPROVAL_YES = frozenset({
    "yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "approve",
    "approved", "go", "go ahead", "do it", "proceed", "execute", "run",
    "confirm", "confirmed", "accept", "accepted", "sounds good", "looks good",
    "that works", "perfect", "great", "good", "fine", "correct", "right",
})

_APPROVAL_NO = frozenset({
    "no", "n", "nope", "nah", "stop", "cancel", "abort", "reject",
    "rejected", "deny", "denied", "don't", "dont", "do not", "never",
    "negative", "wrong", "incorrect", "bad", "not that",
})


def detect_approval_intent(text: str) -> tuple[str, str | None]:
    """
    Detect user intent from text: approval, rejection, or modification.

    Returns:
        Tuple of (intent, modification_text):
        - ("approve", None) - user approved the plan
        - ("reject", None) - user rejected the plan
        - ("modify", "new instructions") - user wants to modify the plan
        - ("unknown", None) - could not determine intent
    """
    if not text:
        return ("unknown", None)

    text_lower = text.lower().strip()
    text_words = set(text_lower.split())

    # Check for exact match or word-level match for approval
    if text_lower in _APPROVAL_YES or text_words & _APPROVAL_YES:
        # But make sure it's not a modification (has other content)
        # Short responses like "yes" are approval, but "yes but change X" is modify
        if len(text_lower) < 20 or text_lower in _APPROVAL_YES:
            return ("approve", None)

    # Check for rejection
    if text_lower in _APPROVAL_NO or text_words & _APPROVAL_NO:
        if len(text_lower) < 20 or text_lower in _APPROVAL_NO:
            return ("reject", None)

    # Check for modification indicators
    modify_indicators = [
        "but", "instead", "change", "modify", "update", "different",
        "actually", "rather", "how about", "what if", "can you",
        "could you", "would you", "please", "also", "add", "remove",
    ]
    for indicator in modify_indicators:
        if indicator in text_lower:
            return ("modify", text)

    # If text is short and starts with approval/rejection word
    first_word = text_words.pop() if text_words else ""
    if first_word in _APPROVAL_YES:
        return ("approve", None)
    if first_word in _APPROVAL_NO:
        return ("reject", None)

    # Default: treat longer unknown text as modification request
    if len(text_lower) > 10:
        return ("modify", text)

    return ("unknown", None)


def _safe_agent_name(agent: Any) -> str:
    raw = None
    try:
        raw = (
            getattr(agent, "state_name", None)
            or getattr(agent, "agent_name", None)
            or getattr(agent, "name", None)
        )
    except (AttributeError, TypeError) as e:
        log_swallowed_exception(e, operation="get_agent_name")
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
    default_network: Any | None = None,  # DefaultNetwork for reactive behaviors
    hippocampus: Any | None = None,  # Hippocampus for episodic memory
    memory_hub: Any | None = None,  # MemoryHub for cross-system memory integration
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
    protocol_registry: Any | None = None,  # ProtocolRegistry for dynamic skills
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
    from maxim.runtime.prefetch import (
        init_prefetcher,
        get_result_cache,
        PrefetchResult,
    )

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

    # Initialize speculative pre-fetcher for efficient context gathering
    prefetcher = init_prefetcher(executor=executor, base_path=os.getcwd())
    result_cache = get_result_cache()
    pending_prefetch: PrefetchResult | None = None

    # Track pending proposal from LLM
    pending_proposal: LLMProposal | None = None
    pending_next_actions: list[dict[str, Any]] = []  # Multi-step action queue
    last_llm_submit_time = 0.0
    llm_submit_interval = 0.5  # Don't flood LLM with requests

    # Track when tools with followup_type complete and need follow-up
    # This triggers another LLM cycle based on the followup type:
    #   "process" - LLM processes results for next action
    #   "respond" - LLM synthesizes results into user response
    #   "engage"  - LLM responds AND offers proactive follow-ups
    pending_action_followup: dict[str, Any] | None = None

    # Planning mode: track proposal awaiting user approval
    # When requires_approval=True, we store the proposal here and wait for user response
    pending_plan_proposal: LLMProposal | None = None

    # Track processed CLI inputs to avoid duplicate submissions
    processed_cli_inputs: set[str] = set()

    # Track recent outcomes for learning
    recent_outcomes: list[dict[str, Any]] = []
    max_recent_outcomes = 10

    # Track agent states
    agent_states: list[dict[str, Any]] = []

    # Live read from tool registry — picks up dynamically registered tools
    def _get_all_tools() -> set[str]:
        if hasattr(executor, "registry") and hasattr(executor.registry, "list"):
            try:
                return set(executor.registry.list())
            except Exception:
                pass
        return set()

    # Loop timing
    target_period = 1.0 / target_hz
    max_steps_i = int(max_steps or 0)
    step_iter = itertools.count() if max_steps_i <= 0 else range(max_steps_i)

    # Default Network lifecycle management
    dn_enabled = default_network is not None
    dn_last_mode: str | None = None

    def configure_dn_for_mode(mode_name: str) -> None:
        """Configure DN based on current mode settings."""
        nonlocal dn_last_mode
        if not dn_enabled or default_network is None:
            return
        if mode_name == dn_last_mode:
            return  # No change needed

        mode_def = get_mode(mode_name)
        if mode_def is None:
            return

        dn_config = mode_def.default_network

        # Enable/disable DN based on mode
        if not dn_config.enabled:
            if default_network.is_running:
                default_network.stop()
                log_agentic("default_network", "dn_inhibited", {"reason": "mode_disabled", "mode": mode_name})
        else:
            if not default_network.is_running:
                default_network.start()
                log_agentic("default_network", "dn_released", {"reason": "mode_enabled", "mode": mode_name})

            # Apply behavior priority modifiers
            default_network.clear_behavior_overrides()
            for behavior_name, modifier in dn_config.behavior_priority_modifiers.items():
                default_network.boost_behavior(behavior_name, modifier)

            # Update gate escalation threshold
            if hasattr(default_network, 'gate') and hasattr(default_network.gate, '_adaptive'):
                if default_network.gate._adaptive:
                    default_network.gate._adaptive._novelty_threshold = dn_config.escalation_threshold
                    default_network.gate._adaptive._salience_threshold = dn_config.escalation_threshold

        dn_last_mode = mode_name

    def inhibit_dn_for_tool(mode_name: str) -> bool:
        """Check if DN should be inhibited during tool execution."""
        if not dn_enabled or default_network is None:
            return False
        mode_def = get_mode(mode_name)
        if mode_def and mode_def.default_network.inhibit_during_tool_execution:
            return True
        return False

    # Start DN if enabled (will be configured on first mode check)
    if dn_enabled and default_network is not None:
        try:
            default_network.start()
            log_agentic("default_network", "startup", {"status": "started"}, level="INFO")
        except Exception as e:
            logger.warning("Failed to start DefaultNetwork: %s", e)
            dn_enabled = False

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

        # Configure Default Network for current mode
        if current_mode:
            configure_dn_for_mode(current_mode)

        # Check if autonomy is paused
        if autonomy_controller.is_paused:
            time.sleep(idle_sleep_s)
            continue

        # ─────────────────────────────────────────────────────────────────
        # 1. PERCEPTION (fast, always runs)
        # ─────��───────────────────────────────────────────────────────────
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
            # CONFIRMATION MODE: Check FIRST if user is confirming a tool execution
            # Must check before storing in memory to prevent "yes"/"no" being sent to LLM
            # ───────────────────────────────────────────────────────────────
            pending_confirmation = state.data.get("pending_confirmation")
            if pending_confirmation:
                response = cli_text.lower().strip()
                if response in ("yes", "y", "ok", "sure", "proceed", "confirm"):
                    # User approved - execute the action
                    action = pending_confirmation["action"]
                    tool_name = pending_confirmation["tool_name"]
                    reasoning = pending_confirmation["reasoning"]
                    confidence = pending_confirmation["confidence"]

                    logger.info("User confirmed action: %s", tool_name)
                    log_agentic(
                        "agent_loop",
                        "user_confirmed",
                        {"tool": tool_name, "approved": True},
                    )

                    confirmed_success = False
                    confirmed_result_str = None
                    try:
                        result = executor.execute(action)
                        success = getattr(result, "success", True)
                        error_msg = getattr(result, "error", None)
                        autonomy_controller.log_action(
                            action_type="executed",
                            action=action,
                            reasoning=reasoning,
                            mode=state.data.get("mode", "unknown"),
                            confidence=confidence,
                            human_involved=True,
                            outcome="success" if success else "failure",
                        )
                        output = getattr(result, "output", None)
                        if success:
                            confirmed_success = True
                            print("✅ Action executed successfully")
                            if output:
                                if isinstance(output, dict):
                                    print(f"   Result: {output}")
                                else:
                                    print(f"   Result: {str(output)[:200]}")
                        else:
                            print(f"❌ Action failed: {error_msg or 'unknown error'}")

                        # Record outcome so LLM sees the result and can follow up
                        confirmed_result_str = str(output)[:3000] if output is not None else None
                        recent_outcomes.append({
                            "tool": tool_name,
                            "success": success,
                            "result": confirmed_result_str,
                            "error": error_msg,
                            "timestamp": time.time(),
                        })
                        if len(recent_outcomes) > max_recent_outcomes:
                            recent_outcomes.pop(0)

                        # Record reasoning carryover for the LLM
                        if llm_worker is not None:
                            llm_worker.record_outcome(
                                tool_name=tool_name,
                                reasoning=reasoning or "",
                                success=success,
                                result_summary=(confirmed_result_str or "")[:200],
                            )

                        # Add to context pool so conversation history includes it
                        context_pool.add_outcome(
                            tool_name=tool_name,
                            success=success,
                            result_summary=confirmed_result_str,
                            error=error_msg,
                        )

                    except Exception as e:
                        logger.error(f"Confirmed action failed: {e}")
                        print(f"❌ Action failed: {e}")

                    # Queue a follow-up LLM cycle so it can continue
                    # the conversation (e.g., propose next action)
                    from maxim.modes.definitions import get_tool_followup_type
                    current_mode = state.data.get("mode", "live")
                    followup_type = get_tool_followup_type(tool_name, current_mode)
                    if followup_type and confirmed_success and confirmed_result_str is not None:
                        pending_action_followup = {
                            "tool": tool_name,
                            "result": confirmed_result_str,
                            "original_query": getattr(pending_proposal, "triggering_input", "") if pending_proposal else "",
                            "followup_type": followup_type,
                            "mode": current_mode,
                            "timestamp": time.time(),
                        }
                        logger.info(
                            "Confirmed action %s queued follow-up (type=%s)",
                            tool_name, followup_type,
                        )

                    # Clear ALL input sources to prevent "yes" from being processed again
                    state.data.pop("pending_confirmation", None)
                    state.data.pop("pending_cli_input", None)  # Clear duplicate source
                    state.data.pop("pending_user_input", None)  # Clear stored input
                    pending_proposal = None  # Clear so Section 6 can submit to LLM
                    cli_input = None  # Skip further processing
                    continue  # Skip rest of this iteration

                elif response in ("no", "n", "cancel", "reject", "abort"):
                    # User rejected
                    action = pending_confirmation["action"]
                    tool_name = pending_confirmation["tool_name"]
                    confidence = pending_confirmation["confidence"]
                    reasoning = pending_confirmation.get("reasoning", "")

                    logger.info("User rejected action: %s", tool_name)
                    log_agentic(
                        "agent_loop",
                        "user_confirmed",
                        {"tool": tool_name, "approved": False},
                    )
                    autonomy_controller.log_action(
                        action_type="rejected",
                        action=action,
                        reasoning="User rejected confirmation",
                        mode=state.data.get("mode", "unknown"),
                        confidence=confidence,
                        human_involved=True,
                    )
                    print("❌ Action cancelled by user")

                    # Record rejection so LLM knows and doesn't re-propose
                    recent_outcomes.append({
                        "tool": tool_name,
                        "success": False,
                        "result": None,
                        "error": "User rejected this action",
                        "timestamp": time.time(),
                    })
                    if len(recent_outcomes) > max_recent_outcomes:
                        recent_outcomes.pop(0)
                    if llm_worker is not None:
                        llm_worker.record_outcome(
                            tool_name=tool_name,
                            reasoning=reasoning,
                            success=False,
                            result_summary="User rejected this action",
                        )
                    context_pool.add_outcome(
                        tool_name=tool_name,
                        success=False,
                        result_summary=None,
                        error="User rejected this action",
                    )

                    # Clear ALL input sources to prevent "no" from being processed again
                    state.data.pop("pending_confirmation", None)
                    state.data.pop("pending_user_input", None)  # Clear stored input
                    pending_proposal = None  # Clear so LLM can re-engage
                    cli_input = None  # Skip further processing
                    continue  # Skip rest of this iteration

                # If input doesn't match yes/no, treat it as a modification request
                # Store the original action and user's modification for LLM to revise
                action = pending_confirmation["action"]
                tool_name = pending_confirmation["tool_name"]
                reasoning = pending_confirmation["reasoning"]

                logger.info("User requested modification for action: %s", tool_name)
                log_agentic(
                    "agent_loop",
                    "user_modification_request",
                    {"tool": tool_name, "modification": cli_text[:100]},
                )

                # Store pending modification for LLM to process
                state.data["pending_modification"] = {
                    "original_action": action,
                    "original_reasoning": reasoning,
                    "original_tool_name": tool_name,
                    "user_modification": cli_text,
                    "timestamp": time.time(),
                }

                # Clear the confirmation - LLM will propose revised action
                state.data.pop("pending_confirmation", None)
                print(f"📝 Modification requested - revising action based on: \"{cli_text[:80]}{'...' if len(cli_text) > 80 else ''}\"")

                # Clear input sources to prevent double processing
                state.data.pop("pending_user_input", None)
                cli_input = None
                continue  # Skip rest of iteration - let LLM process modification

            # ───────────────────────────────────────────────────────────────
            # TIMEOUT RETRY: Check if user is responding to a timeout prompt
            # ───────────────────────────────────────────────────────────────
            pending_timeout = state.data.get("pending_timeout_retry")
            if pending_timeout:
                response = cli_text.lower().strip()
                state.data.pop("pending_timeout_retry", None)

                if response in ("no", "n", "cancel", "skip"):
                    logger.info("User declined timeout retry")
                    print("Understood, skipping.")
                    state.data.pop("pending_cli_input", None)
                    state.data.pop("pending_user_input", None)
                    cli_input = None
                    continue

                # Parse timeout: "yes"/"y" → double, integer → minutes
                new_timeout_s = None
                if response in ("yes", "y", "ok", "sure"):
                    new_timeout_s = pending_timeout["timeout_s"] * 2
                else:
                    try:
                        minutes = int(response)
                        if 1 <= minutes <= 10:
                            new_timeout_s = minutes * 60
                    except ValueError:
                        pass

                if new_timeout_s is not None and llm_worker is not None:
                    original_request = pending_timeout.get("original_request")
                    if original_request is not None:
                        logger.info("Retrying LLM with timeout=%.0fs", new_timeout_s)
                        print(f"Retrying with {int(new_timeout_s)}s time limit...")
                        llm_worker.retry_with_timeout(original_request, new_timeout_s)
                        state.data.pop("pending_cli_input", None)
                        state.data.pop("pending_user_input", None)
                        cli_input = None
                        continue

                # If we couldn't parse the response, fall through to normal processing
                logger.debug("Could not parse timeout retry response: %s", response)

            # Now store in state and memory (only reached if NOT a confirmation response)
            state.data["pending_user_input"] = cli_text
            state.data["pending_user_input_time"] = time.time()  # Track when input was received
            state.data["pending_user_input_source"] = source_type  # Track source for LLM routing
            logger.warning("Agent loop received %s input: %s", source_type, cli_text[:100])
            log_agentic(
                "agent_loop",
                "user_input_received",
                {"text": cli_text[:100], "source": source_type},
            )
            # Record in memory so it appears in context.cli_inputs
            # Voice transcripts are only forwarded if they contain wake word (maxim/reachy)
            if hasattr(memory, "record_command"):
                try:
                    memory.record_command(cli_text)
                    logger.warning("Recorded %s input to memory: %s", source_type, cli_text[:50])
                except Exception as e:
                    logger.warning("Failed to record %s input: %s", source_type, e)

            # ───────────────────────────────────────────────────────────────
            # SPECULATIVE PRE-FETCHING: Pre-gather file context if user
            # mentions files (reduces LLM calls from 2 to 1 for file ops)
            # ───────────────────────────────────────────────────────────────
            try:
                pending_prefetch = prefetcher.prefetch_for_input(cli_text, cwd=os.getcwd())
                if pending_prefetch.discovery_plan:
                    plan = pending_prefetch.discovery_plan
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
                if pending_prefetch.file_references:
                    log_agentic(
                        "agent_loop",
                        "prefetch_complete",
                        {
                            "files": [r.pattern for r in pending_prefetch.file_references[:3]],
                            "intent": pending_prefetch.intent,
                            "skip_exploration": pending_prefetch.skip_exploration,
                            "prefetched_files": len(pending_prefetch.file_contents),
                        },
                    )
                    if pending_prefetch.skip_exploration:
                        if pending_prefetch.intent == "create" and not pending_prefetch.file_contents:
                            logger.info("Pre-fetch: New file creation detected, skipping exploration")
                        else:
                            logger.info("Pre-fetch: Systematic discovery complete (%d files), LLM can write directly", len(pending_prefetch.file_contents))
                    elif pending_prefetch.file_contents:
                        logger.info("Pre-fetch: Gathered %d files for context", len(pending_prefetch.file_contents))
            except Exception as e:
                logger.debug("Pre-fetch failed (non-critical): %s", e)
                pending_prefetch = None

            # ───────────────────────────────────────────────────────────────
            # PLANNING MODE: Check if this input is approval/rejection/modify
            # ───────────────────────────────────────────────────────────────
            if pending_plan_proposal is not None:
                intent, modification = detect_approval_intent(cli_text)
                log_agentic(
                    "agent_loop",
                    "plan_approval_check",
                    {"input": cli_text[:50], "intent": intent, "has_pending_plan": True},
                )

                if intent == "approve":
                    # User approved - execute the stored action
                    logger.info("Plan approved by user, executing stored action")
                    log_agentic("agent_loop", "plan_approved", {"tool": pending_plan_proposal.action.get("tool_name") if pending_plan_proposal.action else None})
                    # Move plan proposal to pending_proposal for execution
                    pending_proposal = pending_plan_proposal
                    pending_plan_proposal = None
                    # Clear from state so it doesn't show again
                    state.data.pop("pending_plan_text", None)

                elif intent == "reject":
                    # User rejected - cancel the pending plan
                    rejected_tool = pending_plan_proposal.action.get("tool_name") if pending_plan_proposal.action else "unknown"
                    logger.info("Plan rejected by user, cancelling")
                    log_agentic("agent_loop", "plan_rejected", {"tool": rejected_tool})

                    # Record rejection so LLM knows its plan was rejected
                    recent_outcomes.append({
                        "tool": rejected_tool,
                        "success": False,
                        "result": None,
                        "error": "User rejected the proposed plan",
                        "timestamp": time.time(),
                    })
                    if len(recent_outcomes) > max_recent_outcomes:
                        recent_outcomes.pop(0)
                    if llm_worker is not None:
                        llm_worker.record_outcome(
                            tool_name=rejected_tool,
                            reasoning=pending_plan_proposal.reasoning or "",
                            success=False,
                            result_summary="User rejected the proposed plan",
                        )
                    context_pool.add_outcome(
                        tool_name=rejected_tool,
                        success=False,
                        result_summary=None,
                        error="User rejected the proposed plan",
                    )

                    pending_plan_proposal = None
                    state.data.pop("pending_plan_text", None)
                    # Don't send to LLM, just clear
                    cli_input = None

                elif intent == "modify":
                    # User wants to modify - send modification to LLM with context
                    logger.info("Plan modification requested: %s", modification[:50] if modification else "")
                    log_agentic("agent_loop", "plan_modify_requested", {"modification": modification[:100] if modification else None})
                    # Store modification context for LLM
                    state.data["plan_modification_context"] = {
                        "original_plan": pending_plan_proposal.plan_text,
                        "original_action": pending_plan_proposal.action,
                        "user_modification": modification,
                    }
                    # Clear pending plan so LLM can generate new one
                    pending_plan_proposal = None
                    state.data.pop("pending_plan_text", None)
                    # cli_input stays set so it gets sent to LLM

                # If unknown intent, ask for clarification
                elif intent == "unknown":
                    logger.info("Could not determine approval intent, asking for clarification")
                    # Keep the pending_plan_proposal and don't process the input
                    state.data["pending_plan_text"] = f"[Awaiting approval] {pending_plan_proposal.plan_text}\n\nPlease respond with 'yes' to approve, 'no' to cancel, or describe changes."
                    cli_input = None  # Don't send unknown input to LLM

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
                        has_keyword = bool(cli_input and ("maxim" in str(cli_input).lower() or "reachy" in str(cli_input).lower()))
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
                # Staleness guard: discard proposals older than LLM timeout + margin
                proposal_age = time.time() - new_proposal.timestamp
                if proposal_age > 35.0:
                    logger.warning(
                        "Skipping stale LLM proposal (age=%.1fs, request_id=%s)",
                        proposal_age, new_proposal.request_id,
                    )
                    new_proposal = None
            if new_proposal:
                if new_proposal.action:
                    tool_name = new_proposal.action.get("tool_name", "unknown")
                    logger.info("LLM proposal received: tool=%s, confidence=%.2f",
                                tool_name, new_proposal.confidence)
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
            pending_user_input and
            input_is_for_llm and
            llm_worker is not None and
            (time.time() - pending_input_time) < llm_response_timeout  # Timeout after 30s
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

                                    # Capture episodic memory to Hippocampus (async — fire-and-forget)
                                    if hippocampus is not None:
                                        try:
                                            hippocampus.capture_from_loop_async(
                                                observation=observation if isinstance(observation, dict) else {},
                                                state=state,
                                                intent=intent,
                                                decision={"action": action, "confidence": confidence},
                                                action={"tool": action["tool_name"], "params": action.get("params", {})},
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
                                    outcome = {
                                        "tool": action["tool_name"],
                                        "success": False,
                                        "result": None,
                                        "error": str(e),
                                        "timestamp": time.time(),
                                    }
                                    recent_outcomes.append(outcome)
                                    if len(recent_outcomes) > max_recent_outcomes:
                                        recent_outcomes.pop(0)

                                    # Record reasoning carryover
                                    if llm_worker is not None and pending_proposal is not None:
                                        llm_worker.record_outcome(
                                            tool_name=action.get("tool_name", "unknown"),
                                            reasoning=getattr(pending_proposal, "reasoning", ""),
                                            success=False,
                                            result_summary=str(e)[:200],
                                        )

                                    # Add to context pool
                                    context_pool.add_outcome(
                                        tool_name=action["tool_name"],
                                        success=False,
                                        result_summary=None,
                                        error=str(e),
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
                            parallel_results.append({
                                "tool": tool_name,
                                "success": False,
                                "error": f"Rejected: {reason}",
                                "result": None,
                            })
                            continue

                        # Execute the action
                        result = executor.execute(parallel_action)
                        success = getattr(result, "success", True)
                        output = getattr(result, "output", None)
                        error = getattr(result, "error", None)

                        parallel_results.append({
                            "tool": tool_name,
                            "params": parallel_action.get("params", {}),
                            "success": success,
                            "result": str(output)[:2000] if output else None,
                            "error": error,
                        })

                        if not success:
                            all_succeeded = False

                        log_agentic(
                            "agent_loop",
                            "parallel_action_complete",
                            {"tool": tool_name, "index": idx, "success": success},
                        )

                    except Exception as e:
                        logger.error("Parallel action %s failed: %s", tool_name, e)
                        parallel_results.append({
                            "tool": tool_name,
                            "success": False,
                            "error": str(e),
                            "result": None,
                        })
                        all_succeeded = False

                # Record individual outcomes so LLM has structured history
                for pr in parallel_results:
                    recent_outcomes.append({
                        "tool": pr["tool"],
                        "success": pr["success"],
                        "result": pr.get("result"),
                        "error": pr.get("error"),
                        "timestamp": time.time(),
                    })
                    if len(recent_outcomes) > max_recent_outcomes:
                        recent_outcomes.pop(0)
                    context_pool.add_outcome(
                        tool_name=pr["tool"],
                        success=pr["success"],
                        result_summary=pr.get("result"),
                        error=pr.get("error"),
                    )
                if llm_worker is not None and pending_proposal is not None:
                    llm_worker.record_outcome(
                        tool_name="batched_exploration",
                        reasoning=getattr(pending_proposal, "reasoning", ""),
                        success=all_succeeded,
                        result_summary=f"{len(parallel_results)} actions, {sum(1 for p in parallel_results if p['success'])} succeeded",
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
                pending_action_followup = {
                    "tool": "batched_exploration",
                    "result": combined_results,
                    "original_query": pending_proposal.triggering_input,
                    "followup_type": "process",  # LLM decides next action
                    "mode": state.data.get("mode", "exploration"),
                    "timestamp": time.time(),
                }
                logger.info("Batched exploration complete, queuing followup for LLM")

                # Clear proposal - will be handled via followup
                pending_proposal = None
                continue  # Skip normal execution flow

            # ───────────────────────────────────────────────────────────────
            # PLANNING MODE: Check if this proposal requires user approval
            # ───────────────────────────────────────────────────────────────
            if getattr(pending_proposal, "requires_approval", False) and pending_proposal.plan_text:
                # Store proposal for approval and show plan to user
                logger.info("Proposal requires approval, showing plan to user")
                log_agentic(
                    "agent_loop",
                    "plan_awaiting_approval",
                    {
                        "tool": action.get("tool_name"),
                        "plan_preview": pending_proposal.plan_text[:100] if pending_proposal.plan_text else None,
                    },
                )

                # Store plan text in state so it can be displayed to user
                # The CLI/UI can read this and display it
                state.data["pending_plan_text"] = pending_proposal.plan_text
                state.data["pending_plan_tool"] = action.get("tool_name")

                # Store in pending_plan_proposal for approval flow
                pending_plan_proposal = pending_proposal
                pending_proposal = None

                # Don't execute - wait for approval
                continue

            logger.info("Executing LLM proposal: tool=%s, confidence=%.2f",
                        action.get("tool_name"), confidence)

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
                    result = executor.execute(action)
                    exec_elapsed = time.time() - exec_start
                    success = getattr(result, "success", True)
                    logger.info("Tool execution completed in %.2fs: %s, success=%s",
                                exec_elapsed, action.get("tool_name"), success)

                    # Auto-recover: write_file failed because file exists → retry with overwrite
                    if (
                        not success
                        and action.get("tool_name") == "write_file"
                        and "already exists" in str(getattr(result, "error", "")).lower()
                    ):
                        logger.info(
                            "Auto-recovery: retrying write_file with overwrite=True for %s",
                            action.get("params", {}).get("path", "?"),
                        )
                        retry_action = dict(action)
                        retry_params = dict(retry_action.get("params", {}))
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
                        state.data["pending_timeout_retry"] = {
                            "original_request": action.get("_original_request"),
                            "timeout_s": action.get("_timeout_s", 60.0),
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
                                    formatted_parts.append(
                                        f"[{i}] {title}\n    URL: {url}\n    {snippet}"
                                    )
                            result_str = "\n\n".join(formatted_parts)[:result_limit]
                        else:
                            result_str = str(output)[:result_limit]
                        # For empty results, include metadata message if available
                        if not output and hasattr(result, "metadata"):
                            msg = result.metadata.get("message", "")
                            if msg:
                                result_str = f"[No results: {msg}]"
                    else:
                        result_str = None

                    outcome = {
                        "tool": tool_name,
                        "success": success,
                        "result": result_str,
                        "error": getattr(result, "error", None),
                        "timestamp": time.time(),
                    }
                    recent_outcomes.append(outcome)
                    if len(recent_outcomes) > max_recent_outcomes:
                        recent_outcomes.pop(0)

                    # Record reasoning carryover
                    if llm_worker is not None and pending_proposal is not None:
                        llm_worker.record_outcome(
                            tool_name=tool_name or "unknown",
                            reasoning=getattr(pending_proposal, "reasoning", ""),
                            success=success,
                            result_summary=(result_str or "")[:200],
                        )

                    # Add to context pool
                    context_pool.add_outcome(
                        tool_name=tool_name or "unknown",
                        success=success,
                        result_summary=result_str,
                        error=getattr(result, "error", None),
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

                    # If this tool has a followup_type and succeeded, trigger a follow-up LLM cycle
                    # The followup_type determines how the LLM should handle the results:
                    #   "process" - LLM processes results for next action (coding agent)
                    #   "respond" - LLM synthesizes results into user response
                    #   "engage"  - LLM responds AND offers proactive follow-ups
                    # Note: Use 'is not None' to handle empty lists [] which are falsy but still valid output
                    if followup_type and success and output is not None:
                        triggering_input = getattr(pending_proposal, "triggering_input", "")
                        pending_action_followup = {
                            "tool": tool_name,
                            "result": result_str,
                            "original_query": triggering_input,
                            "followup_type": followup_type,
                            "mode": current_mode,
                            "timestamp": time.time(),
                        }
                        logger.info("Tool %s completed with followup_type=%s, queuing follow-up", tool_name, followup_type)

                    # Track conversation history for response/speak actions
                    tool_name = action.get("tool_name", "")
                    if tool_name in ("respond", "speak") and success:
                        params = action.get("params", {})
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

                    # Capture episodic memory to Hippocampus (async — fire-and-forget)
                    if hippocampus is not None:
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

                    # Track exception in recent_outcomes for LLM learning
                    outcome = {
                        "tool": action.get("tool_name"),
                        "success": False,
                        "result": None,
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                    recent_outcomes.append(outcome)
                    if len(recent_outcomes) > max_recent_outcomes:
                        recent_outcomes.pop(0)

                    # Record reasoning carryover
                    if llm_worker is not None and pending_proposal is not None:
                        llm_worker.record_outcome(
                            tool_name=action.get("tool_name", "unknown"),
                            reasoning=getattr(pending_proposal, "reasoning", ""),
                            success=False,
                            result_summary=str(e)[:200],
                        )

                    # Add to context pool so LLM can learn from failures
                    context_pool.add_outcome(
                        tool_name=action.get("tool_name", "unknown"),
                        success=False,
                        result_summary=None,
                        error=str(e),
                    )

                    # Mark failure in state
                    try:
                        state.mark_failure(str(e))
                    except Exception:
                        pass

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
                    # Store pending confirmation in state for CLI handler to detect
                    tool_name = action.get("tool_name", "unknown")
                    params = action.get("params", {})

                    # Format the action for display
                    print("\n" + "=" * 60)
                    print("⚠️  ACTION REQUIRES CONFIRMATION")
                    print("=" * 60)
                    print(f"Tool: {tool_name}")
                    print("Parameters:")
                    for key, value in params.items():
                        # Truncate long values for display
                        display_value = str(value)
                        if len(display_value) > 200:
                            display_value = display_value[:200] + "..."
                        print(f"  {key}: {display_value}")
                    print(f"Reasoning: {pending_proposal.reasoning}")
                    print("=" * 60)
                    print("Type 'yes' or 'no' to confirm/reject:")

                    # Store pending confirmation for CLI input handler
                    state.data["pending_confirmation"] = {
                        "action": action,
                        "reasoning": pending_proposal.reasoning,
                        "confidence": confidence,
                        "tool_name": tool_name,
                    }
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
                    recent_outcomes.append({
                        "tool": action.get("tool_name", "unknown"),
                        "success": False,
                        "result": None,
                        "error": rejection_msg,
                        "timestamp": time.time(),
                    })
                    if len(recent_outcomes) > max_recent_outcomes:
                        recent_outcomes.pop(0)
                    if llm_worker is not None:
                        llm_worker.record_outcome(
                            tool_name=action.get("tool_name", "unknown"),
                            reasoning=pending_proposal.reasoning or "",
                            success=False,
                            result_summary=rejection_msg[:200],
                        )
                    context_pool.add_outcome(
                        tool_name=action.get("tool_name", "unknown"),
                        success=False,
                        result_summary=None,
                        error=rejection_msg,
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
                        recent_outcomes.append({
                            "tool": tool_name,
                            "success": success,
                            "result": result_str,
                            "error": error_msg,
                            "timestamp": time.time(),
                        })
                        if len(recent_outcomes) > max_recent_outcomes:
                            recent_outcomes.pop(0)
                        if llm_worker is not None:
                            llm_worker.record_outcome(
                                tool_name=tool_name,
                                reasoning=proposal.reasoning or "",
                                success=success,
                                result_summary=(result_str or "")[:200],
                            )
                        context_pool.add_outcome(
                            tool_name=tool_name,
                            success=success,
                            result_summary=result_str,
                            error=error_msg,
                        )

                        # Queue follow-up so LLM can continue
                        from maxim.modes.definitions import get_tool_followup_type
                        current_mode = state.data.get("mode", "live")
                        followup_type = get_tool_followup_type(tool_name, current_mode)
                        if followup_type and success and output is not None:
                            pending_action_followup = {
                                "tool": tool_name,
                                "result": result_str,
                                "original_query": "",
                                "followup_type": followup_type,
                                "mode": current_mode,
                                "timestamp": time.time(),
                            }

                    except Exception as e:
                        logger.error(f"Approved action failed: {e}")
                        # Record failure so LLM knows
                        recent_outcomes.append({
                            "tool": tool_name,
                            "success": False,
                            "result": None,
                            "error": str(e),
                            "timestamp": time.time(),
                        })
                        if len(recent_outcomes) > max_recent_outcomes:
                            recent_outcomes.pop(0)
                        context_pool.add_outcome(
                            tool_name=tool_name,
                            success=False,
                            result_summary=None,
                            error=str(e),
                        )

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
                    # Track original query from followups for conversation history
                    # This ensures followup responses are saved with the original user question
                    followup_original_query = ""
                    if context:
                        # Known commands that should NOT be sent to LLM
                        # These are handled by Selfy's phrase response system
                        SKIP_LLM_COMMANDS = frozenset({
                            # System commands
                            "maxim shutdown", "shutdown maxim",
                            "maxim stop", "stop maxim",
                            "maxim pause", "pause maxim",
                            "maxim resume", "resume maxim",
                            # Sleep/wake (processing state)
                            "maxim sleep", "sleep maxim",
                            "maxim nap", "maxim rest",
                            "maxim wake", "wake maxim",
                            "maxim wake up", "wake up maxim",
                            # Strategy switching
                            "maxim observe", "observe maxim", "maxim watch",
                            "maxim explore", "explore maxim",
                            "maxim research", "research maxim",
                            "maxim assist", "maxim help",
                            "maxim reflect", "maxim reflection",
                            "maxim learn", "maxim train",
                            # Mode switching
                            "maxim passive",
                            "maxim active",
                            "maxim singularity",
                        })

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
                                    should_process = (input_source == "CLI") or ("maxim" in cli_lower) or ("reachy" in cli_lower)
                                    if should_process:
                                        # Skip LLM for known commands (handled by Selfy)
                                        if cli_lower in SKIP_LLM_COMMANDS:
                                            logger.info("Skipping LLM for command: %s", cli_input)
                                            processed_cli_inputs.add(cli_input)
                                            continue
                                        new_cli_input = cli_input
                                        has_meaningful_input = True
                                        break
                                    else:
                                        # Mark voice inputs without wake word as processed
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

                        # Check for pending action followup (tools that need LLM processing)
                        if pending_action_followup:
                            has_meaningful_input = True
                            # Inject the action result into CLI inputs so LLM can process
                            followup_query = pending_action_followup.get("original_query", "")
                            followup_result = pending_action_followup.get("result", "")
                            followup_tool = pending_action_followup.get("tool", "unknown")
                            followup_type = pending_action_followup.get("followup_type", "process")
                            followup_mode = pending_action_followup.get("mode", "live")

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
                            logger.info("Injected action followup into context: type=%s, tool=%s, result_len=%d", followup_type, followup_tool, len(followup_result))
                            # Clear the followup after processing
                            pending_action_followup = None

                    # Skip LLM if nothing to react to
                    if not has_meaningful_input:
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
                            _tools = _get_all_tools()
                            exploration_tools = exploration_mode_def.get_available_tools(_tools) if _tools else set()

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
                            _tools = _get_all_tools()
                            if mode_def and _tools:
                                available_tools_for_mode = mode_def.get_available_tools(_tools)

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
                        available_tools = mode_info.get_available_tools(_get_all_tools())

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
                                            k: f"({v[0].__name__}, default={v[1]!r})" if isinstance(v, tuple) else v.__name__
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
                        conversation_history_text = context_pool.get_conversation_text(
                            max_turns=5
                        )

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
                            triggering_input=new_cli_input or followup_original_query or (pending_modification.get("user_modification", "") if pending_modification else ""),
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
                                logger.info("Submitted to LLM: %s", new_cli_input[:50] if len(new_cli_input) > 50 else new_cli_input)
                            log_agentic(
                                "agent_loop",
                                "llm_submit",
                                {
                                    "input": new_cli_input[:50] if new_cli_input else "followup" if is_followup else None,
                                    "mode": mode_info.name,
                                    "autonomy": autonomy_controller.current_level.value,
                                    "tools_available": len(available_tools),
                                },
                            )

                except Exception as e:
                    import traceback
                    logger.warning(f"Failed to submit context to LLM: {type(e).__name__}: {e}")
                    logger.debug(f"Full traceback:\n{traceback.format_exc()}")

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
    if memory_hub_enabled and memory_hub is not None:
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

    # Stop Default Network if running
    if dn_enabled and default_network is not None:
        try:
            default_network.stop()
            log_agentic("default_network", "shutdown", {"status": "stopped"}, level="INFO")
        except Exception as e:
            logger.debug(f"Failed to stop DefaultNetwork: {e}")
