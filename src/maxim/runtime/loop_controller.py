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
from typing import TYPE_CHECKING, Any

from maxim.agents.llm_worker import LLMProposal
from maxim.runtime.tool_dispatch import record_outcome as _record_outcome, safe_agent_name as _safe_agent_name
from maxim.runtime.dn_controller import DefaultNetworkController
from maxim.runtime.loop_types import (
    ActionFollowup,
    PendingConfirmation,
    TimeoutRetry,
)
from maxim.utils.structured_logging import log_agentic

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController
    from maxim.agents.llm_worker import LLMWorker
    from maxim.evaluation.base import Evaluator

logger = logging.getLogger(__name__)


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
        self.state_path = os.path.join(
            "data",
            "agents",
            self.agent_name,
            "runtime",
            f"state_{self.run_id}.json",
        )

        # ── Transient loop state (typed — replaces local variables) ──────
        self.pending_proposal: LLMProposal | None = None
        self.pending_next_actions: list[dict[str, Any]] = []
        self.pending_plan_proposal: LLMProposal | None = None
        self.pending_action_followup: ActionFollowup | None = None
        self.pending_prefetch: Any | None = None
        self.last_llm_submit_time: float = 0.0
        self.llm_submit_interval: float = 0.5
        self.processed_cli_inputs: collections.deque[str] = collections.deque(maxlen=20)
        self.recent_outcomes: list[dict[str, Any]] = []
        self.max_recent_outcomes: int = 10
        self.agent_states: list[dict[str, Any]] = []
        self.last_surfaced_tools: list[str] = []

        # ── Memory hub / hippocampus ─────────────────────────────────────
        self.memory_hub_enabled = memory_hub is not None

        # ── Default Network ──────────────────────────────────────────────
        self.dn_ctrl = DefaultNetworkController(default_network)
        self.dn_enabled = self.dn_ctrl.enabled

    # ── Typed state helpers ──────────────────────────────────────────────

    def get_pending_confirmation(self) -> PendingConfirmation | None:
        raw = self.state.data.get("pending_confirmation")
        if raw is None:
            return None
        return PendingConfirmation(**raw)

    def set_pending_confirmation(self, pc: PendingConfirmation | None) -> None:
        if pc is None:
            self.state.data.pop("pending_confirmation", None)
        else:
            self.state.data["pending_confirmation"] = {
                "action": pc.action,
                "reasoning": pc.reasoning,
                "confidence": pc.confidence,
                "tool_name": pc.tool_name,
            }

    def get_timeout_retry(self) -> TimeoutRetry | None:
        raw = self.state.data.get("pending_timeout_retry")
        if raw is None:
            return None
        return TimeoutRetry(
            original_request=raw.get("original_request"),
            timeout_s=raw.get("timeout_s", 60.0),
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

    # ── Outcome recording (delegates to module-level helper) ─────────────

    def record_outcome(
        self,
        *,
        tool_name: str,
        success: bool,
        result_summary: str | None,
        error: str | None,
        reasoning: str,
    ) -> None:
        _record_outcome(
            tool_name=tool_name,
            success=success,
            result_summary=result_summary,
            error=error,
            reasoning=reasoning,
            recent_outcomes=self.recent_outcomes,
            max_recent=self.max_recent_outcomes,
            llm_worker=self.llm_worker,
            context_pool=self.context_pool,
        )

    # ── Tool registry cache ──────────────────────────────────────────────

    def get_all_tools(self) -> set[str]:
        if hasattr(self.executor, "registry") and hasattr(self.executor.registry, "list"):
            try:
                return set(self.executor.registry.list())
            except (KeyError, AttributeError):
                pass
        return set()

    # ── Phase: Handle confirmation input ─────────────────────────────────

    def handle_confirmation(self, cli_text: str) -> bool:
        """Process user input when a confirmation is pending.

        Returns True if the input was consumed (caller should skip further processing).
        """
        from maxim.modes.definitions import get_tool_followup_type

        pc = self.get_pending_confirmation()
        if pc is None:
            return False

        response = cli_text.lower().strip()

        if response in ("yes", "y", "ok", "sure", "proceed", "confirm"):
            logger.info("User confirmed action: %s", pc.tool_name)
            log_agentic("agent_loop", "user_confirmed", {"tool": pc.tool_name, "approved": True})

            confirmed_success = False
            confirmed_result_str = None
            try:
                result = self.executor.execute(pc.action)
                success = getattr(result, "success", True)
                error_msg = getattr(result, "error", None)
                self.autonomy_controller.log_action(
                    action_type="executed",
                    action=pc.action,
                    reasoning=pc.reasoning,
                    mode=self.state.data.get("mode", "unknown"),
                    confidence=pc.confidence,
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

                confirmed_result_str = str(output)[:3000] if output is not None else None
                self.record_outcome(
                    tool_name=pc.tool_name,
                    success=success,
                    result_summary=confirmed_result_str,
                    error=error_msg,
                    reasoning=pc.reasoning or "",
                )

            except Exception as e:
                logger.error(f"Confirmed action failed: {e}")
                print(f"❌ Action failed: {e}")

            # Queue a follow-up LLM cycle so it can continue
            current_mode = self.state.data.get("mode", "live")
            followup_type = get_tool_followup_type(pc.tool_name, current_mode)
            if followup_type and confirmed_success and confirmed_result_str is not None:
                self.pending_action_followup = ActionFollowup(
                    tool=pc.tool_name,
                    result=confirmed_result_str,
                    original_query=getattr(self.pending_proposal, "triggering_input", "")
                    if self.pending_proposal
                    else "",
                    followup_type=followup_type,
                    mode=current_mode,
                    timestamp=time.time(),
                )
                logger.info("Confirmed action %s queued follow-up (type=%s)", pc.tool_name, followup_type)

            self.set_pending_confirmation(None)
            self.state.data.pop("pending_cli_input", None)
            self.clear_pending_user_input()
            self.pending_proposal = None
            return True

        elif response in ("no", "n", "cancel", "reject", "abort"):
            logger.info("User rejected action: %s", pc.tool_name)
            log_agentic("agent_loop", "user_confirmed", {"tool": pc.tool_name, "approved": False})
            self.autonomy_controller.log_action(
                action_type="rejected",
                action=pc.action,
                reasoning="User rejected confirmation",
                mode=self.state.data.get("mode", "unknown"),
                confidence=pc.confidence,
                human_involved=True,
            )
            print("❌ Action cancelled by user")

            self.record_outcome(
                tool_name=pc.tool_name,
                success=False,
                result_summary=None,
                error="User rejected this action",
                reasoning=pc.reasoning,
            )

            self.set_pending_confirmation(None)
            self.clear_pending_user_input()
            self.pending_proposal = None
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
            self.set_pending_confirmation(None)
            print(
                f'📝 Modification requested - revising action based on: "{cli_text[:80]}{"..." if len(cli_text) > 80 else ""}"'
            )
            self.clear_pending_user_input()
            return True

    # ── Phase: Handle timeout retry ──────────────────────────────────────

    def handle_timeout_retry(self, cli_text: str) -> bool:
        """Process user input when a timeout retry is pending.

        Returns True if the input was consumed.
        """
        tr = self.get_timeout_retry()
        if tr is None:
            return False

        response = cli_text.lower().strip()
        self.set_timeout_retry(None)

        if response in ("no", "n", "cancel", "skip"):
            logger.info("User declined timeout retry")
            print("Understood, skipping.")
            self.state.data.pop("pending_cli_input", None)
            self.clear_pending_user_input()
            return True

        new_timeout_s = None
        if response in ("yes", "y", "ok", "sure"):
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
                print(f"Retrying with {int(new_timeout_s)}s time limit...")
                self.llm_worker.retry_with_timeout(tr.original_request, new_timeout_s)
                self.state.data.pop("pending_cli_input", None)
                self.clear_pending_user_input()
                return True

        # Could not parse — fall through to normal processing
        logger.debug("Could not parse timeout retry response: %s", response)
        return False

    # ── Phase: Handle plan approval ──────────────────────────────────────

    def handle_plan_approval(self, cli_text: str) -> bool:
        """Process user input when a plan proposal is awaiting approval.

        Returns True if the input was consumed.
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
            self.pending_proposal = self.pending_plan_proposal
            self.pending_plan_proposal = None
            self.state.data.pop("pending_plan_text", None)
            return True

        elif intent == "reject":
            rejected_tool = (
                self.pending_plan_proposal.action.get("tool_name") if self.pending_plan_proposal.action else "unknown"
            )
            logger.info("Plan rejected by user, cancelling")
            log_agentic("agent_loop", "plan_rejected", {"tool": rejected_tool})

            self.record_outcome(
                tool_name=rejected_tool,
                success=False,
                result_summary=None,
                error="User rejected the proposed plan",
                reasoning=self.pending_plan_proposal.reasoning or "",
            )

            self.pending_plan_proposal = None
            self.state.data.pop("pending_plan_text", None)
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
            self.state.data["plan_modification_context"] = {
                "original_plan": self.pending_plan_proposal.plan_text,
                "original_action": self.pending_plan_proposal.action,
                "user_modification": modification,
            }
            self.pending_plan_proposal = None
            self.state.data.pop("pending_plan_text", None)
            # cli_input stays set so it gets sent to LLM
            return False  # Don't consume — let LLM see the modification text

        else:  # unknown
            logger.info("Could not determine approval intent, asking for clarification")
            self.state.data["pending_plan_text"] = (
                f"[Awaiting approval] {self.pending_plan_proposal.plan_text}\n\n"
                "Please respond with 'yes' to approve, 'no' to cancel, or describe changes."
            )
            return True  # Consume unknown input

    # ── Phase: Store user input in state and memory ──────────────────────

    def store_user_input(self, cli_text: str, source_type: str) -> None:
        """Store CLI/voice input in state and memory for LLM processing."""
        self.state.data["pending_user_input"] = cli_text
        self.state.data["pending_user_input_time"] = time.time()
        self.state.data["pending_user_input_source"] = source_type
        logger.warning("Agent loop received %s input: %s", source_type, cli_text[:100])
        log_agentic(
            "agent_loop",
            "user_input_received",
            {"text": cli_text[:100], "source": source_type},
        )
        if hasattr(self.memory, "record_command"):
            try:
                self.memory.record_command(cli_text)
                logger.warning("Recorded %s input to memory: %s", source_type, cli_text[:50])
            except Exception as e:
                logger.warning("Failed to record %s input: %s", source_type, e)

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
        except Exception as e:
            logger.debug("Pre-fetch failed (non-critical): %s", e)
            self.pending_prefetch = None

    # ── Phase: Configure Default Network for mode ────────────────────────

    def configure_dn_for_mode(self, mode_name: str) -> None:
        """Delegate to DefaultNetworkController."""
        self.dn_ctrl.configure_for_mode(mode_name)
