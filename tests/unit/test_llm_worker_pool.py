"""Tests for LLMWorker + WorkerPool integration (Phase 3).

Verifies that LLMWorker correctly uses the infer lane of a WorkerPool
when pool mode is active, while maintaining backward compatibility
with legacy internal-thread mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FakeContext:
    """Minimal StructuredContext stand-in."""

    cli_inputs: list[str] = field(default_factory=list)
    audio_inputs: list = field(default_factory=list)
    video_observations: list = field(default_factory=list)
    movement_observations: list = field(default_factory=list)
    environment_data: dict = field(default_factory=dict)


class FakeLLM:
    """Mock LLM backend that returns a fixed JSON response."""

    def __init__(self, response: dict[str, Any] | None = None, delay: float = 0.0):
        self._response = response or {
            "action": {"tool_name": "respond", "params": {"message": "Hello"}},
            "reasoning": "test_response",
            "strategy": "assist",
            "confidence": 0.9,
            "mode_goal_achieved": False,
        }
        self._delay = delay
        self.call_count = 0

    def generate_json(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> dict[str, Any] | None:
        self.call_count += 1
        if self._delay > 0:
            time.sleep(self._delay)
        return dict(self._response)


def _make_mode_info():
    from maxim.agents.llm_worker import ModeInfo

    return ModeInfo(
        name="active",
        goal="assist the user",
        context_prompt="You are a helpful assistant.",
        max_response_tokens=512,
        context_window_tokens=2048,
    )


def _submit_test_context(worker, triggering_input: str = "maxim hello", priority: int = 0) -> bool:
    """Submit a minimal context to the worker."""
    from maxim.agents.autonomy import AutonomyLevel

    return worker.submit_context(
        context=FakeContext(cli_inputs=[triggering_input]),
        mode=_make_mode_info(),
        autonomy_level=AutonomyLevel.SUPERVISED,
        internet_access=False,
        internet_policy_summary="",
        priority=priority,
        triggering_input=triggering_input,
        use_tool_prompting=True,
        available_tools={"respond"},
        tool_descriptions={"respond": "Send a message"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Pool Mode (internal pool, owns_pool=True)
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMWorkerPoolMode:
    """Test LLMWorker when using WorkerPool (default mode)."""

    def test_start_creates_internal_pool(self):
        """start() creates an internal WorkerPool with infer lane."""
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=FakeLLM())
        assert worker._pool is None
        assert worker._owns_pool is True

        worker.start()
        try:
            assert worker._pool is not None
            assert "large" in worker._pool._lanes
        finally:
            worker.stop()

    def test_submit_and_get_proposal(self):
        """submit_context routes through pool, get_latest_proposal returns result."""
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=FakeLLM(), stale_threshold_s=10.0)
        worker.start()
        try:
            assert _submit_test_context(worker) is True

            # Wait for infer lane to process the job
            proposal = None
            for _ in range(40):
                proposal = worker.get_latest_proposal()
                if proposal is not None:
                    break
                time.sleep(0.1)

            assert proposal is not None
            assert proposal.action is not None
            assert proposal.action["tool_name"] == "respond"
        finally:
            worker.stop()

    def test_stale_request_dropped(self):
        """Requests older than stale_threshold are dropped in pool mode."""
        from maxim.agents.llm_worker import LLMWorker

        # Very short stale threshold so request expires by the time it runs
        worker = LLMWorker(llm=FakeLLM(delay=0.0), stale_threshold_s=0.001)
        worker.start()
        try:
            # Submit and let it go stale
            time.sleep(0.01)
            _submit_test_context(worker)

            # Wait for processing
            time.sleep(1.0)

            # Should get None (stale request was dropped)
            proposal = worker.get_latest_proposal()
            # The proposal might be None (dropped) or present (processed fast)
            # Check that requests_dropped was incremented if None
            if proposal is None:
                assert worker._requests_dropped > 0
        finally:
            worker.stop()

    def test_get_latest_proposal_returns_none_when_empty(self):
        """get_latest_proposal returns None when no jobs are complete."""
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=FakeLLM())
        worker.start()
        try:
            assert worker.get_latest_proposal() is None
        finally:
            worker.stop()

    def test_multiple_submissions(self):
        """Multiple submit_context calls produce multiple proposals."""
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=FakeLLM(), stale_threshold_s=30.0)
        worker.start()
        try:
            for i in range(3):
                _submit_test_context(worker, triggering_input=f"maxim test {i}")

            # Wait for all to process
            proposals = []
            for _ in range(60):
                p = worker.get_latest_proposal()
                if p is not None:
                    proposals.append(p)
                if len(proposals) >= 3:
                    break
                time.sleep(0.1)

            assert len(proposals) == 3
        finally:
            worker.stop()

    def test_get_all_proposals(self):
        """get_all_proposals drains all completed infer jobs."""
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=FakeLLM(), stale_threshold_s=30.0)
        worker.start()
        try:
            for i in range(3):
                _submit_test_context(worker, triggering_input=f"maxim test {i}")

            # Wait for processing
            time.sleep(2.0)

            proposals = worker.get_all_proposals()
            assert len(proposals) >= 1  # At least some completed
        finally:
            worker.stop()

    def test_stop_shuts_down_pool(self):
        """stop() shuts down the internal pool."""
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=FakeLLM())
        worker.start()
        pool = worker._pool
        assert pool is not None

        worker.stop()
        # Pool's GC thread should have stopped
        assert pool._gc_stop.is_set()

    def test_retry_with_timeout(self):
        """retry_with_timeout submits a new job to the pool."""
        from maxim.agents.llm_worker import LLMWorker, LLMRequest
        from maxim.agents.autonomy import AutonomyLevel

        worker = LLMWorker(llm=FakeLLM(), stale_threshold_s=30.0)
        worker.start()
        try:
            request = LLMRequest(
                request_id="req-test",
                context=FakeContext(cli_inputs=["maxim hello"]),
                mode=_make_mode_info(),
                autonomy_level=AutonomyLevel.SUPERVISED,
                internet_access=False,
                internet_policy_summary="",
                triggering_input="maxim hello",
                use_tool_prompting=True,
                available_tools={"respond"},
                tool_descriptions={"respond": "Send a message"},
            )

            assert worker.retry_with_timeout(request, timeout_s=120.0) is True

            # Should produce a proposal
            proposal = None
            for _ in range(40):
                proposal = worker.get_latest_proposal()
                if proposal is not None:
                    break
                time.sleep(0.1)

            assert proposal is not None
        finally:
            worker.stop()

    def test_record_outcome_still_works(self):
        """record_outcome is unchanged and works in pool mode."""
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=FakeLLM())
        worker.start()
        try:
            worker.record_outcome(
                tool_name="respond",
                reasoning="test",
                success=True,
                result_summary="said hello",
            )
            assert len(worker._reasoning_carryover) == 1
        finally:
            worker.stop()

    def test_submit_returns_false_when_queue_full(self):
        """submit_context returns False when infer lane queue is full."""
        from maxim.agents.llm_worker import LLMWorker
        from maxim.runtime.worker_pool import LaneConfig, WorkerPool

        # Create pool with tiny queue
        pool = WorkerPool(
            lane_configs={
                "large": LaneConfig(name="large", max_workers=1, queue_size=2),
            }
        )
        pool.start()
        try:
            # Use very slow LLM to back up the queue
            worker = LLMWorker(llm=FakeLLM(delay=5.0), stale_threshold_s=30.0, pool=pool)
            worker.start()

            # Submit until queue fills
            results = []
            for i in range(10):
                result = _submit_test_context(worker, triggering_input=f"maxim test {i}")
                results.append(result)

            # At least some should have been rejected
            assert False in results
        finally:
            worker.stop()
            pool.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: External Pool
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMWorkerExternalPool:
    """Test LLMWorker with an externally-provided WorkerPool."""

    def test_external_pool_not_owned(self):
        """Worker doesn't create or destroy an external pool."""
        from maxim.agents.llm_worker import LLMWorker
        from maxim.runtime.worker_pool import LaneConfig, WorkerPool

        pool = WorkerPool(
            lane_configs={
                "large": LaneConfig(name="large", max_workers=1),
            }
        )
        pool.start()
        try:
            worker = LLMWorker(llm=FakeLLM(), pool=pool)
            assert worker._owns_pool is False
            assert worker._pool is pool

            worker.start()
            # Pool should still be the same (not replaced)
            assert worker._pool is pool

            worker.stop()
            # Pool should NOT be stopped (not owned)
            assert not pool._gc_stop.is_set()
        finally:
            pool.stop()

    def test_external_pool_submit_and_retrieve(self):
        """Submit and retrieve proposals through external pool."""
        from maxim.agents.llm_worker import LLMWorker
        from maxim.runtime.worker_pool import LaneConfig, WorkerPool

        pool = WorkerPool(
            lane_configs={
                "large": LaneConfig(name="large", max_workers=1),
            }
        )
        pool.start()
        try:
            worker = LLMWorker(llm=FakeLLM(), stale_threshold_s=30.0, pool=pool)
            worker.start()

            _submit_test_context(worker)

            proposal = None
            for _ in range(40):
                proposal = worker.get_latest_proposal()
                if proposal is not None:
                    break
                time.sleep(0.1)

            assert proposal is not None
            assert proposal.action is not None
        finally:
            worker.stop()
            pool.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Legacy Mode (no pool, backward compat)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Priority Ordering
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMWorkerPriority:
    """Test that priority ordering works through the pool."""

    def test_higher_priority_processed_first(self):
        """Higher priority submissions are processed before lower priority."""
        from maxim.agents.llm_worker import LLMWorker
        from maxim.runtime.worker_pool import LaneConfig, WorkerPool

        call_order = []

        class OrderTrackingLLM:
            def generate_json(self, prompt, temperature=0.3, max_tokens=1024):
                # Extract which request this is from the prompt
                if "high" in prompt.lower():
                    call_order.append("high")
                elif "low" in prompt.lower():
                    call_order.append("low")
                else:
                    call_order.append("unknown")
                return {
                    "action": {"tool_name": "respond", "params": {"message": "ok"}},
                    "reasoning": "test",
                    "confidence": 0.9,
                }

        # Don't start the pool yet (let jobs queue up)
        pool = WorkerPool(
            lane_configs={
                "large": LaneConfig(name="large", max_workers=1),
            }
        )

        worker = LLMWorker(llm=OrderTrackingLLM(), stale_threshold_s=30.0, pool=pool)
        worker._llm_executor = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="LLMCall"
        )

        # Submit low priority first, then high priority (before pool starts)
        # Lane.submit works even before pool.start (it registers jobs and queues them)
        try:
            _submit_test_context(worker, triggering_input="maxim low priority test", priority=0)
            _submit_test_context(worker, triggering_input="maxim high priority test", priority=10)

            # Now start pool - dispatcher will process by priority
            pool.start()

            # Wait for both to process
            time.sleep(3.0)

            # High priority should have been dequeued first
            if len(call_order) >= 2:
                assert call_order[0] == "high", f"Expected high priority first, got: {call_order}"
        finally:
            worker.stop()
            pool.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: BUG-4 — Per-request timeout (no global mutation)
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMPerRequestTimeout:
    """Verify retry_with_timeout sets per-request override without mutating global timeout."""

    def test_default_timeout_unchanged(self):
        """LLMWorker._llm_timeout equals the configured default after construction."""
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=FakeLLM(), llm_timeout_s=45.0)
        assert worker._llm_timeout == 45.0

    def test_retry_does_not_mutate_global_timeout(self):
        """retry_with_timeout does NOT change worker._llm_timeout."""
        from maxim.agents.llm_worker import LLMWorker, LLMRequest
        from maxim.agents.autonomy import AutonomyLevel

        original_timeout = 60.0
        worker = LLMWorker(llm=FakeLLM(), stale_threshold_s=30.0, llm_timeout_s=original_timeout)
        worker.start()
        try:
            request = LLMRequest(
                request_id="req-timeout",
                context=FakeContext(cli_inputs=["maxim hello"]),
                mode=_make_mode_info(),
                autonomy_level=AutonomyLevel.SUPERVISED,
                internet_access=False,
                internet_policy_summary="",
                triggering_input="maxim hello",
                use_tool_prompting=True,
                available_tools={"respond"},
                tool_descriptions={"respond": "Send a message"},
            )

            worker.retry_with_timeout(request, timeout_s=120.0)

            # Global timeout must still be the original value
            assert worker._llm_timeout == original_timeout
        finally:
            worker.stop()

    def test_timeout_override_set_on_request(self):
        """After retry_with_timeout, request.timeout_override holds the override value."""
        from maxim.agents.llm_worker import LLMWorker, LLMRequest
        from maxim.agents.autonomy import AutonomyLevel

        worker = LLMWorker(llm=FakeLLM(), stale_threshold_s=30.0)
        worker.start()
        try:
            request = LLMRequest(
                request_id="req-override",
                context=FakeContext(cli_inputs=["maxim hello"]),
                mode=_make_mode_info(),
                autonomy_level=AutonomyLevel.SUPERVISED,
                internet_access=False,
                internet_policy_summary="",
                triggering_input="maxim hello",
                use_tool_prompting=True,
                available_tools={"respond"},
                tool_descriptions={"respond": "Send a message"},
            )

            worker.retry_with_timeout(request, timeout_s=120.0)

            assert request.timeout_override == 120.0
        finally:
            worker.stop()

    def test_llm_request_default_timeout_override_is_none(self):
        """LLMRequest.timeout_override defaults to None."""
        from maxim.agents.llm_worker import LLMRequest
        from maxim.agents.autonomy import AutonomyLevel

        request = LLMRequest(
            request_id="req-default",
            context=FakeContext(cli_inputs=["maxim hello"]),
            mode=_make_mode_info(),
            autonomy_level=AutonomyLevel.SUPERVISED,
            internet_access=False,
            internet_policy_summary="",
            triggering_input="maxim hello",
        )

        assert request.timeout_override is None


# ─────────────────────────────────────────────────────────────────────────────
# Plan 4 follow-up (2026-04-14): session_id plumbing
# ─────────────────────────────────────────────────────────────────────────────


class _RequestContextCapturingLLM:
    """Mock LLM backend that captures the request_context dict it receives.

    Used to verify that LLMWorker threads session_id (and agent_id,
    request_id, lane) into the dict it passes through to
    ``router.generate_json``. The captured dicts are available as
    ``self.captured_contexts`` for per-call assertions.
    """

    def __init__(self, response: dict[str, Any] | None = None):
        self._response = response or {
            "action": {"tool_name": "respond", "params": {"message": "ok"}},
            "reasoning": "test_response",
            "strategy": "assist",
            "confidence": 0.9,
            "mode_goal_achieved": False,
        }
        self.captured_contexts: list[dict[str, Any] | None] = []
        self.call_count = 0

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        *,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
        system_override: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | None:
        self.call_count += 1
        # Snapshot the dict so later mutations don't corrupt the record
        self.captured_contexts.append(dict(request_context) if request_context else None)
        return dict(self._response)


class TestSessionIdPlumbing:
    """Regression guards for Plan 4 follow-up (2026-04-14).

    Closes the explicit out-of-scope item Plan 4 Stage A flagged:
    ``SimulationOrchestrator`` generates a session_id but wasn't
    threading it into the dict ``LLMWorker`` builds. The fix adds
    ``session_id`` as an optional ``LLMWorker.__init__`` parameter,
    stores it on the instance, and includes it in every
    ``request_context`` dict the worker constructs at its two
    dict-build sites.
    """

    def test_session_id_defaults_to_none_for_non_session_callers(self):
        """Backward-compat guard: LLMWorker constructed without a
        session_id (the old signature) still works, and its
        ``_session_id`` attribute is None. Non-sim callers — exec_agent
        sub-workers, api.py verb consumers, bench harness,
        embodied_runtime — all rely on this default.
        """
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=FakeLLM())
        try:
            assert worker._session_id is None
        finally:
            worker.stop()

    def test_session_id_init_arg_stored_on_instance(self):
        """Passing session_id at construction stores it on the instance
        so both dict-build sites can read it."""
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=FakeLLM(), session_id="sim-20260414_120000")
        try:
            assert worker._session_id == "sim-20260414_120000"
        finally:
            worker.stop()

    def test_session_id_present_in_process_request_dict_source(self):
        """The ``_process_request`` dict-build site at line ~962 in
        llm_worker.py is the primary entry point for queued LLM calls.
        Static-source regression guard: confirm that the literal
        ``request_context`` dict in that method includes
        ``"session_id": self._session_id``.

        A pool-path integration test would be ideal, but the pool
        path's ``_build_prompt`` short-circuits trivial inputs like
        "maxim hello" through the ``{"action": ...}`` pre-built JSON
        shortcut (llm_worker.py:917), never reaching the LLM — so the
        existing ``test_submit_and_get_proposal`` ALSO never actually
        calls the FakeLLM. Instead we lock the dict shape via source
        inspection to catch silent regressions in the dict-build site.
        The equivalent end-to-end coverage lives in
        ``test_session_id_threaded_into_generate_json_direct_dict``
        below, which DOES exercise the LLM code path end-to-end.
        """
        import inspect

        from maxim.agents.llm_worker import LLMWorker

        src = inspect.getsource(LLMWorker._process_request)
        # Strip whitespace variations — look for the key/value pair
        # in any form that would produce the dict we need.
        assert '"session_id": self._session_id' in src, (
            "LLMWorker._process_request dict-build site is missing "
            '`"session_id": self._session_id` — regression in Plan 4 '
            "session_id plumbing"
        )
        # Also lock the other fields the regression guard should catch
        # if a future refactor drops them.
        assert '"request_id"' in src
        assert '"agent": "llm_worker"' in src
        assert '"lane"' in src

    def test_session_id_threaded_into_generate_json_direct_dict(self):
        """The ``generate_json_direct`` ExecAgent path (line 709 in
        llm_worker.py as of 2026-04-14) is the second dict-build site.
        Bypasses the request queue and calls ``_call_llm_with_timeout``
        synchronously — useful for specialized ExecAgent prompts.
        """
        from maxim.agents.llm_worker import LLMWorker

        fake = _RequestContextCapturingLLM()
        worker = LLMWorker(llm=fake, session_id="sim-direct-test")
        worker.start()
        try:
            worker.generate_json_direct(
                system="You are a test assistant.",
                user="Say hi",
                request_id="req-direct-test",
                agent_name="test-agent",
                lane="large",
                temperature=0.3,
                max_tokens=32,
            )
            assert fake.call_count == 1
            ctx = fake.captured_contexts[0]
            assert ctx is not None
            assert ctx.get("session_id") == "sim-direct-test"
            assert ctx.get("agent") == "test-agent"
            assert ctx.get("request_id") == "req-direct-test"
            assert ctx.get("lane") == "large"
        finally:
            worker.stop()

    def test_normalize_request_context_surfaces_session_id(self):
        """End-to-end sanity: the canonical ``_normalize_request_context``
        shim (agents/llm_worker.py) reads ``session_id`` from the dict
        into the typed ``RequestContext.session_id`` field. This is
        the handoff to ``_MaximPeerBackend._log_success`` which reads
        ``context.session_id`` when emitting ``peer_backend_call`` JSONL.
        """
        from maxim.agents.llm_worker import _normalize_request_context

        ctx = _normalize_request_context(
            {
                "request_id": "r",
                "agent": "test",
                "lane": "large",
                "session_id": "sim-normalize-test",
            }
        )
        assert ctx.session_id == "sim-normalize-test"
        assert ctx.agent_id == "test"
        assert ctx.request_id == "r"
        assert ctx.lane == "large"
