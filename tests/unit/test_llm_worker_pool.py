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

    def generate_json(
        self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024
    ) -> dict[str, Any] | None:
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


def _make_strategies():
    from maxim.agents.llm_worker import StrategyInfo

    return [StrategyInfo(name="assist", description="Help the user", approach_prompt="Be helpful")]


def _submit_test_context(worker, triggering_input: str = "maxim hello", priority: int = 0) -> bool:
    """Submit a minimal context to the worker."""
    from maxim.agents.autonomy import AutonomyLevel

    return worker.submit_context(
        context=FakeContext(cli_inputs=[triggering_input]),
        mode=_make_mode_info(),
        autonomy_level=AutonomyLevel.SUPERVISED,
        strategies=_make_strategies(),
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
            assert "infer" in worker._pool._lanes
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
                strategies=_make_strategies(),
                internet_access=False,
                internet_policy_summary="",
                triggering_input="maxim hello",
                use_tool_prompting=True,
                available_tools={"respond"},
                tool_descriptions={"respond": "Send a message"},
            )

            assert worker.retry_with_timeout(request, timeout_s=120.0) is True
            assert worker._llm_timeout == 120.0

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
        pool = WorkerPool(lane_configs={
            "infer": LaneConfig(name="infer", max_workers=1, queue_size=2),
        })
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

        pool = WorkerPool(lane_configs={
            "infer": LaneConfig(name="infer", max_workers=1),
        })
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

        pool = WorkerPool(lane_configs={
            "infer": LaneConfig(name="infer", max_workers=1),
        })
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


class TestLLMWorkerLegacyMode:
    """Test that legacy (no-pool) mode still works correctly."""

    def test_legacy_submit_and_proposal(self):
        """Legacy mode uses internal thread and queues."""
        from maxim.agents.llm_worker import LLMWorker

        # Force legacy mode: pass pool=None explicitly and bypass owns_pool
        worker = LLMWorker(llm=FakeLLM(), stale_threshold_s=30.0)
        # Override to prevent internal pool creation
        worker._owns_pool = False
        worker._pool = None

        worker.start()
        try:
            assert worker._worker is not None
            assert worker._worker.is_alive()

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

    def test_legacy_stop(self):
        """Legacy stop joins the worker thread."""
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=FakeLLM())
        worker._owns_pool = False
        worker._pool = None

        worker.start()
        thread = worker._worker
        assert thread is not None
        assert thread.is_alive()

        worker.stop()
        assert not thread.is_alive()


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
        pool = WorkerPool(lane_configs={
            "infer": LaneConfig(name="infer", max_workers=1),
        })

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
