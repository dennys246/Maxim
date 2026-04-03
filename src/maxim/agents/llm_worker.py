"""Dedicated LLM worker thread for non-blocking inference.

The LLMWorker processes LLM requests asynchronously, ensuring the main control
loop never blocks on LLM latency. Includes fallback behaviors for when the
LLM is slow or unavailable.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

from maxim.agents.autonomy import AutonomyLevel
from maxim.models.language.router import CharEstimateCounter

# ─────────────────────────────────────────────────────────────────────────────
# Re-exports for backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

from maxim.agents.llm_types import (  # noqa: F401
    LLMBackend,
    LLMProposal,
    LLMRequest,
    ModeInfo,
    StrategyInfo,
)
from maxim.agents.llm_context import (  # noqa: F401
    _COST_BRIDGE_DEFAULTS,
    _CLOUD_PROVIDER_TYPES,
    _is_cloud_provider_type,
    _load_cost_bridge_config,
    _load_foundational_context,
)
from maxim.agents.prompt_budgeter import (  # noqa: F401
    PromptBudgeter,
    PromptSection,
    SectionPriority,
    _truncate_context_pool,
    _truncate_conversation,
    _truncate_manifest,
    _truncate_reasoning_carryover,
    _truncate_tool_guidance,
)
from maxim.agents.llm_fallback import (  # noqa: F401
    FallbackBehavior,
    ReasoningCarryover,
    ReasoningEntry,
    _ARITHMETIC_PATTERN,
    _SIMPLE_OPS,
    _TRAILING_OP_PATTERN,
    _UNARY_MATH_PATTERNS,
    _compile_phrase_pattern,
    evaluate_simple_arithmetic,
    evaluate_unary_math,
    generate_llm_fallback,
    generate_simple_answer,
    matches_phrase,
    normalize_phrases,
)
from maxim.agents.prompt_builder import (  # noqa: F401
    PromptBuilder,
    build_datetime_section,
    build_identity_section,
    build_instructions_section,
    build_modification_section,
    build_observation_section,
    build_planning_banner,
    build_tool_guidance_core,
    build_tool_guidance_extended,
    build_tools_section,
    build_workspace_manifest,
    compute_budget_tier,
    cost_energy_divergence,
    estimate_hours_until_limit,
    format_spend_rate,
    format_usd,
    is_realtime_request,
    scan_workspace_entries,
)

# Optional energy tracking import
try:
    from maxim.energy.llm_tracker import LLMEnergyTracker

    _HAS_ENERGY_TRACKING = True
except ImportError:
    _HAS_ENERGY_TRACKING = False
    LLMEnergyTracker = None  # type: ignore

if TYPE_CHECKING:
    from maxim.agents.bus import StructuredContext
    from maxim.runtime.worker_pool import WorkerPool


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Worker
# ─────────────────────────────────────────────────────────────────────────────


class LLMWorker:
    """
    Dedicated thread for LLM inference.

    Design principles:
    - Main loop NEVER waits on LLM
    - Stale contexts are dropped (only latest matters)
    - Results are non-blocking to consume
    - Graceful degradation if LLM is slow/unavailable
    """

    def __init__(
        self,
        llm: LLMBackend,
        max_queue_size: int = 5,
        stale_threshold_s: float = 2.0,
        llm_timeout_s: float = 60.0,
        energy_tracker: "LLMEnergyTracker | None" = None,
        n_ctx: int = 4096,
        token_counter: Any | None = None,
        pool: "WorkerPool | None" = None,
        tool_index: Any = None,
    ):
        self._llm = llm
        self._stale_threshold = stale_threshold_s
        self._llm_timeout = llm_timeout_s
        self._energy_tracker = energy_tracker
        self._n_ctx = n_ctx
        self._tool_index = tool_index
        self._token_counter = token_counter or CharEstimateCounter()
        self._reasoning_carryover = ReasoningCarryover(max_entries=5)
        self._cost_energy_scale = _load_cost_bridge_config().get("cost_energy_scale", 100.0)
        self._provider_semaphores: dict[str, threading.Semaphore] = {}

        if self._has_cloud_providers():
            providers = self._llm.get_provider_configs()
            try:
                max_ctx = max(
                    int(cfg.get("n_ctx", self._n_ctx) or self._n_ctx)
                    for cfg in providers.values()
                )
                self._n_ctx = max(self._n_ctx, max_ctx)
            except Exception:
                pass

        # WorkerPool integration
        self._pool: WorkerPool | None = pool
        self._owns_pool = pool is None  # create internal pool in start()

        self._stop_event = threading.Event()

        # Thread pool for timeout-wrapped LLM calls
        self._llm_executor: concurrent.futures.ThreadPoolExecutor | None = None

        # Metrics
        self._requests_processed = 0
        self._requests_dropped = 0
        self._avg_latency_ms = 0.0

        # PromptBuilder for prompt construction
        self._prompt_builder = PromptBuilder(
            llm=self._llm,
            reasoning_carryover=self._reasoning_carryover,
            n_ctx=self._n_ctx,
            token_counter=self._token_counter,
            tool_index=self._tool_index,
        )

    def _has_cloud_providers(self) -> bool:
        """Check if the LLM router has cloud providers configured and allowed."""
        return (
            hasattr(self._llm, "cloud_allowed")
            and self._llm.cloud_allowed()
            and hasattr(self._llm, "get_provider_configs")
        )

    def _get_provider_hint(
        self, system: str, user: str, temperature: float, max_tokens: int,
    ) -> tuple[str | None, bool]:
        """Preview which provider would handle a request.

        Returns (provider_name, is_cloud) or (None, False) on failure.
        """
        if not hasattr(self._llm, "preview_provider"):
            return None, False
        try:
            preview = self._llm.preview_provider(
                system=system, user=user,
                temperature=temperature, max_tokens=max_tokens,
            )
            if isinstance(preview, dict):
                return preview.get("provider"), bool(preview.get("is_cloud"))
        except Exception:
            pass
        return None, False

    def record_outcome(
        self,
        tool_name: str,
        reasoning: str,
        success: bool,
        result_summary: str,
    ) -> None:
        """Record a decision+outcome into the reasoning carryover buffer."""
        self._reasoning_carryover.record(tool_name, reasoning, success, result_summary)

    def _init_provider_semaphores(self) -> int:
        """Initialize per-provider concurrency semaphores for cloud backends."""
        self._provider_semaphores.clear()
        max_workers = 1
        if hasattr(self._llm, "get_provider_configs"):
            providers = self._llm.get_provider_configs()
            for key, cfg in providers.items():
                provider_type = cfg.get("type", "")
                if not _is_cloud_provider_type(provider_type):
                    continue
                try:
                    limit = int(cfg.get("max_concurrent_requests", 1) or 1)
                except Exception:
                    limit = 1
                limit = max(1, limit)
                self._provider_semaphores[key] = threading.Semaphore(limit)
                max_workers = max(max_workers, limit)
        return max_workers

    def retry_with_timeout(self, request: LLMRequest, timeout_s: float) -> bool:
        """Resubmit a request with a per-request timeout override.

        Used when the user asks for more processing time after a timeout.
        The override applies only to this request and does not affect
        the worker's default timeout for other requests.

        Returns True if queued, False if queue full.
        """
        old_timeout = self._llm_timeout
        request.timeout_override = timeout_s
        # Refresh timestamp so staleness checks don't drop it
        request.timestamp = time.time()
        request.sort_index = (-request.priority, request.timestamp)
        logger.info("Retrying LLM request with timeout=%.0fs (was %.0fs)", timeout_s, old_timeout)

        if self._pool is not None:
            def _retry_fn(prefetched=None):
                if self._stop_event.is_set():
                    return None
                proposal = self._process_request(request)
                self._requests_processed += 1
                return proposal

            try:
                self._pool.submit(
                    lane="infer",
                    job_id=f"{request.request_id}-retry",
                    fn=_retry_fn,
                    priority=-request.priority,
                )
                return True
            except queue.Full:
                self._requests_dropped += 1
                return False

    def start(self) -> None:
        """Start the LLM worker via WorkerPool."""
        self._stop_event.clear()

        # Initialize provider semaphores for concurrency control
        self._init_provider_semaphores()

        # Always create the LLM executor (needed for _call_llm_with_timeout)
        if self._llm_executor is None:
            self._llm_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="LLMCall"
            )

        # Create internal WorkerPool if none was provided
        if self._owns_pool:
            from maxim.runtime.worker_pool import LaneConfig, WorkerPool

            net_workers = self._init_provider_semaphores()
            self._pool = WorkerPool(lane_configs={
                "infer": LaneConfig(name="infer", max_workers=1, requires_gpu=True),
                "infer_net": LaneConfig(name="infer_net", max_workers=net_workers, requires_gpu=False),
            })

        if self._pool is not None:
            self._pool.start()
        logger.info("LLM worker started")

    def stop(self) -> None:
        """Stop the LLM worker."""
        self._stop_event.set()

        if self._pool is not None and self._owns_pool:
            self._pool.stop()
        logger.info("LLM worker stopped")

        # Shutdown the LLM executor
        if self._llm_executor is not None:
            try:
                self._llm_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 doesn't have cancel_futures
                self._llm_executor.shutdown(wait=False)
            self._llm_executor = None

    def _call_llm_with_timeout(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        *,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
        system_override: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking: dict[str, Any] | None = None,
        stream: bool = False,
        timeout_override: float | None = None,
    ) -> dict[str, Any] | None:
        """Call LLM with timeout to allow graceful shutdown.

        Returns:
            LLM response dict or None if timeout/error/shutdown.
        """
        if self._stop_event.is_set():
            return None

        executor = self._llm_executor
        if executor is None:
            return None

        try:
            future = executor.submit(
                self._llm.generate_json,
                prompt,
                temperature,
                max_tokens,
                provider_hint=provider_hint,
                request_context=request_context,
                system_override=system_override,
                tools=tools,
                thinking=thinking,
                stream=stream,
            )
            # Wait with timeout, checking stop_event periodically
            timeout_remaining = timeout_override or self._llm_timeout
            poll_interval = 0.5
            while timeout_remaining > 0:
                if self._stop_event.is_set():
                    future.cancel()
                    return None
                try:
                    result = future.result(timeout=min(poll_interval, timeout_remaining))
                    return result
                except concurrent.futures.TimeoutError:
                    timeout_remaining -= poll_interval
                    continue
            # Final timeout exceeded
            logger.warning("LLM call timed out after %.1fs", self._llm_timeout)
            future.cancel()
            # Replace the executor so the orphaned thread doesn't block
            # future LLM calls (max_workers=1 means next submit() queues
            # behind the still-running orphan, causing cascading timeouts).
            try:
                old_executor = self._llm_executor
                self._llm_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="LLMCall"
                )
                if old_executor is not None:
                    old_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.error("Failed to replace LLM executor after timeout: %s", e)
            return {"_timeout": True, "_timeout_s": self._llm_timeout}
        except concurrent.futures.CancelledError:
            return None
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

    def _record_usage(
        self,
        *,
        prompt: str,
        response: dict[str, Any] | None,
        latency_ms: float,
        request_id: str,
        lane: str,
        mode_name: str = "unknown",
    ) -> None:
        if not response or not isinstance(response, dict):
            return

        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        if usage is None:
            usage = response.get("_usage") if isinstance(response.get("_usage"), dict) else {}
        if not isinstance(usage, dict):
            usage = {}

        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))

        if self._energy_tracker is not None:
            # Estimate tokens if missing from usage payload
            if input_tokens == 0:
                input_tokens = len(prompt) // 4
            if output_tokens == 0:
                try:
                    response_str = json.dumps(response)
                    output_tokens = len(response_str) // 4
                except Exception:
                    output_tokens = 50

            try:
                self._energy_tracker.record(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=getattr(self._llm, "model_name", "unknown"),
                    latency_ms=latency_ms,
                    context={
                        "request_id": request_id,
                        "mode": mode_name or "unknown",
                        "lane": lane,
                    },
                )
            except Exception as e:
                logger.debug("Failed to record LLM energy: %s", e)

        cost_usd = usage.get("cost_usd", 0) if isinstance(usage, dict) else 0
        if cost_usd:
            try:
                from maxim.energy.signal import EnergySignal, EnergyType
                from maxim.energy.registry import get_global_registry
                registry = get_global_registry()
                signal = EnergySignal(
                    energy_type=EnergyType.LLM_COST,
                    amount=float(cost_usd) * float(self._cost_energy_scale or 100.0),
                    source="llm_cost",
                    duration_ms=latency_ms,
                    context={
                        "usd": float(cost_usd),
                        "model": usage.get("model") if isinstance(usage, dict) else "",
                        "provider": usage.get("provider") if isinstance(usage, dict) else "",
                        "lane": lane,
                    },
                )
                registry.record_signal(signal)
            except Exception as e:
                logger.debug("Failed to record LLM cost energy: %s", e)

    def generate_json_direct(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        request_id: str,
        agent_name: str = "exec_agent",
        lane: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | None:
        """Direct JSON generation path for specialized prompts (ExecAgent)."""
        if self._stop_event.is_set():
            return None

        start_time = time.time()
        provider_hint = None
        provider_semaphore = None
        is_cloud = False

        provider_hint, is_cloud = self._get_provider_hint(
            system, user, temperature, max_tokens,
        )

        lane_name = lane or ("infer_net" if is_cloud else "infer")
        if provider_hint and provider_hint in self._provider_semaphores:
            provider_semaphore = self._provider_semaphores[provider_hint]

        request_context = {
            "request_id": request_id,
            "agent": agent_name,
            "lane": lane_name,
            "provider_hint": provider_hint or "",
        }

        call_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "provider_hint": provider_hint,
            "request_context": request_context,
            "system_override": system,
        }
        if tools:
            call_kwargs["tools"] = tools
        if thinking:
            call_kwargs["thinking"] = thinking
        if stream:
            call_kwargs["stream"] = True

        if provider_semaphore:
            with provider_semaphore:
                response = self._call_llm_with_timeout(user, **call_kwargs)
        else:
            response = self._call_llm_with_timeout(user, **call_kwargs)

        latency_ms = (time.time() - start_time) * 1000
        self._update_avg_latency(latency_ms)
        self._record_usage(
            prompt=f"{system}\n\n{user}".strip(),
            response=response if isinstance(response, dict) else None,
            latency_ms=latency_ms,
            request_id=request_id,
            lane=lane_name,
            mode_name=agent_name,
        )
        self._requests_processed += 1
        return response

    def get_budget_context(self) -> str:
        """Expose budget context for external prompt builders."""
        try:
            return self._build_budget_context()
        except Exception as e:
            logger.debug("Failed to build budget context: %s", e)
            return ""

    def submit_context(
        self,
        context: StructuredContext,
        mode: ModeInfo,
        autonomy_level: AutonomyLevel,
        strategies: list[StrategyInfo],
        internet_access: bool,
        internet_policy_summary: str,
        priority: int = 0,
        *,
        available_tools: set[str] | None = None,
        tool_descriptions: dict[str, str] | None = None,
        context_pool_text: str = "",
        agent_states: list[dict[str, Any]] | None = None,
        recent_outcomes: list[dict[str, Any]] | None = None,
        use_tool_prompting: bool = False,
        triggering_input: str = "",
        conversation_history_text: str = "",
        pending_modification: dict[str, Any] | None = None,
        prefetch_context: str = "",
        skip_exploration: bool = False,
        current_strategy: str = "",
        is_sleeping: bool = False,
        protocol_context: str = "",
    ) -> bool:
        """
        Submit context for LLM processing (non-blocking).

        Returns True if queued, False if dropped (queue full).
        Main loop should call this frequently; stale requests are pruned.

        Args:
            context: Structured context from memory agent
            mode: Current mode information
            autonomy_level: Current autonomy level
            strategies: Available strategies
            internet_access: Whether internet is available
            internet_policy_summary: Summary of internet policy
            priority: Request priority (higher = more urgent)
            available_tools: Set of tool names available in current mode
            tool_descriptions: Dict of tool name -> description for prompts
            context_pool_text: Accumulated context/observations summary
            agent_states: List of agent state snapshots
            recent_outcomes: List of recent action outcomes for learning
            use_tool_prompting: Whether to use full tool-aware prompts
            triggering_input: The user input that triggered this request
            conversation_history_text: Formatted conversation history for context
        """
        request = LLMRequest(
            request_id=f"req-{time.time_ns()}",
            context=context,
            mode=mode,
            autonomy_level=autonomy_level,
            strategies=strategies,
            internet_access=internet_access,
            internet_policy_summary=internet_policy_summary,
            priority=priority,
            available_tools=available_tools or set(),
            tool_descriptions=tool_descriptions or {},
            context_pool_text=context_pool_text,
            agent_states=agent_states or [],
            recent_outcomes=recent_outcomes or [],
            use_tool_prompting=use_tool_prompting,
            triggering_input=triggering_input,
            conversation_history_text=conversation_history_text,
            pending_modification=pending_modification,
            prefetch_context=prefetch_context,
            skip_exploration=skip_exploration,
            current_strategy=current_strategy,
            is_sleeping=is_sleeping,
            protocol_context=protocol_context,
        )

        if self._pool is not None:
            lane = "infer"
            if self._has_cloud_providers():
                for cfg in self._llm.get_provider_configs().values():
                    if _is_cloud_provider_type(cfg.get("type", "")):
                        lane = "infer_net"
                        break
            request.lane = lane
            # WorkerPool mode: wrap _process_request in a job
            def _infer_job(prefetched=None):
                # Staleness guard inside the job
                age = time.time() - request.timestamp
                if age > self._stale_threshold:
                    self._requests_dropped += 1
                    logger.debug("Dropped stale LLM request (age=%.2fs)", age)
                    return None
                if self._stop_event.is_set():
                    return None
                proposal = self._process_request(request)
                self._requests_processed += 1
                return proposal

            try:
                # Negate priority: higher caller priority -> lower queue value
                self._pool.submit(
                    lane=lane,
                    job_id=request.request_id,
                    fn=_infer_job,
                    priority=-priority,
                )
                return True
            except queue.Full:
                self._requests_dropped += 1
                logger.warning("LLM request dropped (queue full): %s (total dropped: %d)",
                               request.request_id if hasattr(request, 'request_id') else 'unknown',
                               self._requests_dropped)
                return False

    def get_latest_proposal(self) -> LLMProposal | None:
        """
        Get the most recent proposal (non-blocking).

        Main loop calls this each iteration to check for LLM output.
        Returns None if no proposal available.
        """
        if self._pool is None:
            return None
        completed = self._pool.get_completed("infer")
        if completed is not None and completed.result is not None:
            return completed.result
        return None

    def get_all_proposals(self) -> list[LLMProposal]:
        """Get all pending proposals (non-blocking)."""
        if self._pool is None:
            return []
        proposals = []
        while True:
            completed = self._pool.get_completed("infer")
            if completed is None or completed.result is None:
                break
            proposals.append(completed.result)
        return proposals

    def _process_request(self, request: LLMRequest) -> LLMProposal:
        """Process a single LLM request."""
        start_time = time.time()

        try:
            prompt = self._build_prompt(request)

            # Skip LLM call if no meaningful prompt (idle observation)
            if not prompt or not prompt.strip():
                return LLMProposal(
                    request_id=request.request_id,
                    action=None,
                    reasoning="No user input to respond to",
                    strategy_used=None,
                    confidence=0.0,
                    mode_goal_achieved=False,
                    citations=[],
                    latency_ms=0.0,
                    triggering_input=request.triggering_input,
                )

            # Check if this is a pre-built JSON response (simple answer)
            # These don't need LLM - just parse and return
            if prompt.startswith('{"action":'):
                latency_ms = (time.time() - start_time) * 1000
                try:
                    response = json.loads(prompt)
                    return LLMProposal(
                        request_id=request.request_id,
                        action=response.get("action"),
                        reasoning=response.get("reasoning", "direct_answer"),
                        strategy_used="direct_answer",
                        confidence=response.get("confidence", 0.95),
                        mode_goal_achieved=response.get("mode_goal_achieved", False),
                        citations=[],
                        latency_ms=latency_ms,
                        triggering_input=request.triggering_input,
                    )
                except Exception:
                    pass  # Fall through to LLM if parse fails

            # Check for shutdown before LLM call
            if self._stop_event.is_set():
                return LLMProposal(
                    request_id=request.request_id,
                    action=None,
                    reasoning="Shutdown requested",
                    strategy_used=None,
                    confidence=0.0,
                    mode_goal_achieved=False,
                    citations=[],
                    latency_ms=(time.time() - start_time) * 1000,
                    error="shutdown",
                    triggering_input=request.triggering_input,
                )

            # Use mode-specific max tokens for dynamic response length
            max_tokens = request.mode.max_response_tokens
            provider_hint, _ = self._get_provider_hint(
                "", prompt, 0.3, max_tokens,
            )
            provider_semaphore = None
            if provider_hint and provider_hint in self._provider_semaphores:
                provider_semaphore = self._provider_semaphores[provider_hint]

            request_context = {
                "request_id": request.request_id,
                "agent": "llm_worker",
                "lane": request.lane or "infer",
                "provider_hint": provider_hint or "",
            }

            if provider_semaphore:
                with provider_semaphore:
                    response = self._call_llm_with_timeout(
                        prompt,
                        temperature=0.3,
                        max_tokens=max_tokens,
                        provider_hint=provider_hint,
                        request_context=request_context,
                        timeout_override=request.timeout_override,
                    )
            else:
                response = self._call_llm_with_timeout(
                    prompt,
                    temperature=0.3,
                    max_tokens=max_tokens,
                    provider_hint=provider_hint,
                    request_context=request_context,
                    timeout_override=request.timeout_override,
                )

            # Check for timeout -- ask user if they want to wait longer
            if isinstance(response, dict) and response.get("_timeout"):
                timeout_s = response.get("_timeout_s", self._llm_timeout)
                question = (request.triggering_input or "your request")[:50]
                timeout_msg = (
                    f"I ran out of time processing '{question}' "
                    f"(took longer than {int(timeout_s)}s). "
                    f"Would you like me to try again with more time? "
                    f"Say 'yes' to double my time limit, a number of minutes "
                    f"(e.g. '2'), or 'no' to skip."
                )
                return LLMProposal(
                    request_id=request.request_id,
                    action={
                        "tool_name": "respond",
                        "params": {"message": timeout_msg},
                        "_timeout_retry": True,
                        "_original_request": request,
                        "_timeout_s": timeout_s,
                    },
                    reasoning="llm_timeout",
                    strategy_used="fallback",
                    confidence=0.7,
                    mode_goal_achieved=False,
                    citations=[],
                    latency_ms=(time.time() - start_time) * 1000,
                    triggering_input=request.triggering_input,
                )

            # Check for shutdown after LLM call
            if self._stop_event.is_set():
                return LLMProposal(
                    request_id=request.request_id,
                    action=None,
                    reasoning="Shutdown requested",
                    strategy_used=None,
                    confidence=0.0,
                    mode_goal_achieved=False,
                    citations=[],
                    latency_ms=(time.time() - start_time) * 1000,
                    error="shutdown",
                    triggering_input=request.triggering_input,
                )

            latency_ms = (time.time() - start_time) * 1000
            self._update_avg_latency(latency_ms)

            self._record_usage(
                prompt=prompt,
                response=response if isinstance(response, dict) else None,
                latency_ms=latency_ms,
                request_id=request.request_id,
                lane=request.lane or "infer",
                mode_name=request.mode.name if request.mode else "unknown",
            )

            # Trace for simulation debugging
            try:
                from maxim.simulation.sim_logger import sim_log
                _action = response.get("action") if isinstance(response, dict) else None
                _tool = _action.get("tool_name") if isinstance(_action, dict) else None
                sim_log("EXEC", f"LLM raw response parsed: tool={_tool}, type={type(response).__name__}")
            except Exception:
                pass

            if not response or not isinstance(response, dict):
                # LLM failed - generate a fallback response for the user
                fallback = self._generate_llm_fallback(request)
                if fallback:
                    return LLMProposal(
                        request_id=request.request_id,
                        action=fallback,
                        reasoning="llm_fallback",
                        strategy_used="fallback",
                        confidence=0.7,
                        mode_goal_achieved=False,
                        citations=[],
                        latency_ms=latency_ms,
                        triggering_input=request.triggering_input,
                    )

                return LLMProposal(
                    request_id=request.request_id,
                    action=None,
                    reasoning="LLM returned invalid response",
                    strategy_used=None,
                    confidence=0.0,
                    mode_goal_achieved=False,
                    citations=[],
                    latency_ms=latency_ms,
                    error="Invalid LLM response",
                    triggering_input=request.triggering_input,
                )

            # Extract next_actions if present (sequential execution)
            next_actions = response.get("next_actions", [])
            if not isinstance(next_actions, list):
                next_actions = []

            # Extract parallel_actions if present (batched execution)
            # These execute together before the next LLM call
            parallel_actions = response.get("parallel_actions", [])
            if not isinstance(parallel_actions, list):
                parallel_actions = []

            # Extract planning mode fields (prefixed with _)
            plan_text = response.pop("_plan_text", None)
            requires_approval = response.pop("_requires_approval", False)

            return LLMProposal(
                request_id=request.request_id,
                action=response.get("action"),
                reasoning=response.get("reasoning", ""),
                strategy_used=response.get("strategy"),
                confidence=response.get("confidence", 0.5),
                mode_goal_achieved=response.get("mode_goal_achieved", False),
                citations=response.get("citations", []),
                latency_ms=latency_ms,
                next_actions=next_actions,
                parallel_actions=parallel_actions,
                triggering_input=request.triggering_input,
                plan_text=plan_text,
                requires_approval=requires_approval,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LLMProposal(
                request_id=request.request_id,
                action=None,
                reasoning="",
                strategy_used=None,
                confidence=0.0,
                mode_goal_achieved=False,
                latency_ms=latency_ms,
                error=str(e),
                triggering_input=request.triggering_input,
            )

    # ── Delegation to extracted modules ──────────────────────────────────

    def _generate_llm_fallback(self, request: LLMRequest) -> dict[str, Any] | None:
        """Delegate to module-level generate_llm_fallback()."""
        return generate_llm_fallback(request)

    # ── Delegation to PromptBuilder ──────────────────────────────────────

    def _build_prompt(self, request: LLMRequest) -> str:
        return self._prompt_builder.build_prompt(request)

    def _build_budget_context(self) -> str:
        return self._prompt_builder.build_budget_context()

    def _build_followup_prompt(self, followup_input: str) -> str:
        return self._prompt_builder._build_followup_prompt(followup_input)

    def _build_process_prompt(self, tool_name: str, query: str, result: str) -> str:
        return self._prompt_builder._build_process_prompt(tool_name, query, result)

    def _build_respond_prompt(self, tool_name: str, query: str, result: str) -> str:
        return self._prompt_builder._build_respond_prompt(tool_name, query, result)

    def _build_engage_prompt(self, tool_name: str, query: str, result: str, mode_name: str) -> str:
        return self._prompt_builder._build_engage_prompt(tool_name, query, result, mode_name)

    def _build_tool_aware_prompt(
        self, request: LLMRequest, question_text: str, date_str: str, time_str: str,
    ) -> str:
        return self._prompt_builder._build_tool_aware_prompt(request, question_text, date_str, time_str)

    # ── Metrics ──────────────────────────────────────────────────────────

    def _update_avg_latency(self, latency_ms: float) -> None:
        """Update rolling average latency."""
        alpha = 0.1  # Smoothing factor
        self._avg_latency_ms = alpha * latency_ms + (1 - alpha) * self._avg_latency_ms

    @property
    def stats(self) -> dict[str, Any]:
        """Get worker statistics."""
        return {
            "requests_processed": self._requests_processed,
            "requests_dropped": self._requests_dropped,
            "avg_latency_ms": self._avg_latency_ms,
            "pool_running": self._pool is not None and self._pool.is_alive() if hasattr(self._pool, "is_alive") else self._pool is not None,
        }
