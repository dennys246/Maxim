"""Tests for runtime/bio_integration.py — extracted from agent_loop.py."""

from __future__ import annotations

from unittest.mock import MagicMock
from maxim.runtime.bio_integration import (
    capture_episodic_memory,
    record_plan_outcome,
    start_bio_session,
    end_bio_session,
)


class TestCaptureEpisodicMemory:
    """Test hippocampus capture with RPE salience boosting."""

    def test_basic_capture(self):
        hippo = MagicMock()
        executor = MagicMock(spec=[])  # No get_last_rpe
        obs = {"source": "test", "salience": 0.5}
        capture_episodic_memory(
            hippocampus=hippo,
            executor=executor,
            observation=obs,
            state=MagicMock(),
            intent={"goal": "test"},
            action={"tool_name": "look", "params": {}},
            result=MagicMock(),
            run_id="run-1",
        )
        hippo.capture_from_loop_async.assert_called_once()

    def test_rpe_salience_boost(self):
        hippo = MagicMock()
        executor = MagicMock()
        executor.get_last_rpe.return_value = 0.6
        obs = {"source": "test", "salience": 0.5}
        capture_episodic_memory(
            hippocampus=hippo,
            executor=executor,
            observation=obs,
            state=MagicMock(),
            intent={"goal": "test"},
            action={"tool_name": "look", "params": {}},
            result=MagicMock(),
            run_id="run-1",
        )
        # Salience should be boosted: 0.5 + 0.6*0.5 = 0.8
        assert obs["salience"] == 0.8

    def test_rpe_salience_capped_at_1(self):
        hippo = MagicMock()
        executor = MagicMock()
        executor.get_last_rpe.return_value = 1.5
        obs = {"source": "test", "salience": 0.9}
        capture_episodic_memory(
            hippocampus=hippo,
            executor=executor,
            observation=obs,
            state=MagicMock(),
            intent={"goal": "test"},
            action={"tool_name": "look", "params": {}},
            result=MagicMock(),
            run_id="run-1",
        )
        assert obs["salience"] <= 1.0

    def test_no_rpe_no_boost(self):
        hippo = MagicMock()
        executor = MagicMock()
        executor.get_last_rpe.return_value = 0.0
        obs = {"source": "test", "salience": 0.5}
        capture_episodic_memory(
            hippocampus=hippo,
            executor=executor,
            observation=obs,
            state=MagicMock(),
            intent={"goal": "test"},
            action={"tool_name": "look", "params": {}},
            result=MagicMock(),
            run_id="run-1",
        )
        assert obs["salience"] == 0.5

    def test_capture_failure_does_not_propagate(self):
        hippo = MagicMock()
        hippo.capture_from_loop_async.side_effect = RuntimeError("boom")
        executor = MagicMock(spec=[])
        # Should not raise
        capture_episodic_memory(
            hippocampus=hippo,
            executor=executor,
            observation={},
            state=MagicMock(),
            intent={"goal": "test"},
            action={"tool_name": "look", "params": {}},
            result=MagicMock(),
            run_id="run-1",
        )

    def test_non_dict_observation_handled(self):
        hippo = MagicMock()
        executor = MagicMock(spec=[])
        # Non-dict observation — should pass {} to capture
        capture_episodic_memory(
            hippocampus=hippo,
            executor=executor,
            observation="not a dict",
            state=MagicMock(),
            intent={"goal": "test"},
            action={"tool_name": "look", "params": {}},
            result=MagicMock(),
            run_id="run-1",
        )
        call_kwargs = hippo.capture_from_loop_async.call_args[1]
        assert call_kwargs["observation"] == {}


class TestRecordPlanOutcome:
    """Test MemoryHub plan outcome recording."""

    def test_records_success(self):
        hub = MagicMock()
        record_plan_outcome(memory_hub=hub, goal="explore the room", tool_name="look", success=True)
        hub.record_plan_outcome.assert_called_once_with(goal="explore the room", tool_sequence=["look"], success=True)

    def test_truncates_long_goal(self):
        hub = MagicMock()
        long_goal = "x" * 300
        record_plan_outcome(memory_hub=hub, goal=long_goal, tool_name="look", success=True)
        call_kwargs = hub.record_plan_outcome.call_args[1]
        assert len(call_kwargs["goal"]) == 200

    def test_failure_does_not_propagate(self):
        hub = MagicMock()
        hub.record_plan_outcome.side_effect = RuntimeError("db error")
        # Should not raise
        record_plan_outcome(memory_hub=hub, goal="test", tool_name="look", success=True)


class TestStartBioSession:
    """Test bio-system session initialization."""

    def test_starts_both(self):
        hub = MagicMock()
        hub.on_session_start.return_value = {"status": "ok"}
        hippo = MagicMock()
        result = start_bio_session(memory_hub=hub, hippocampus=hippo)
        assert result is True
        hub.on_session_start.assert_called_once()
        hippo.start_capture_worker.assert_called_once()

    def test_hub_failure_returns_false(self):
        hub = MagicMock()
        hub.on_session_start.side_effect = RuntimeError("init failed")
        hippo = MagicMock()
        result = start_bio_session(memory_hub=hub, hippocampus=hippo)
        assert result is False
        # Hippocampus should still start
        hippo.start_capture_worker.assert_called_once()

    def test_none_hub_returns_false(self):
        result = start_bio_session(memory_hub=None, hippocampus=MagicMock())
        assert result is False

    def test_none_hippocampus_ok(self):
        hub = MagicMock()
        hub.on_session_start.return_value = {}
        result = start_bio_session(memory_hub=hub, hippocampus=None)
        assert result is True

    def test_hippo_worker_failure_nonfatal(self):
        hippo = MagicMock()
        hippo.start_capture_worker.side_effect = RuntimeError("thread error")
        result = start_bio_session(memory_hub=None, hippocampus=hippo)
        assert result is False  # No hub


class TestEndBioSession:
    """Test bio-system session teardown."""

    def test_flushes_and_saves_hippocampus(self):
        hippo = MagicMock()
        hippo.config = MagicMock()
        hippo.config.persistence_path = "/tmp/test.json"
        hippo.__len__ = MagicMock(return_value=10)
        end_bio_session(
            memory_hub=None,
            memory_hub_enabled=False,
            hippocampus=hippo,
            is_sim_mode=False,
        )
        hippo.flush.assert_called_once_with(timeout=5.0)
        hippo.stop_capture_worker.assert_called_once()
        hippo.save.assert_called_once()

    def test_ends_memory_hub_session(self):
        hub = MagicMock()
        hub.on_session_end.return_value = {"consolidated": 5}
        end_bio_session(
            memory_hub=hub,
            memory_hub_enabled=True,
            hippocampus=None,
            is_sim_mode=False,
        )
        hub.on_session_end.assert_called_once()

    def test_skips_hub_in_sim_mode(self):
        hub = MagicMock()
        end_bio_session(
            memory_hub=hub,
            memory_hub_enabled=True,
            hippocampus=None,
            is_sim_mode=True,
        )
        hub.on_session_end.assert_not_called()

    def test_skips_hub_when_not_enabled(self):
        hub = MagicMock()
        end_bio_session(
            memory_hub=hub,
            memory_hub_enabled=False,
            hippocampus=None,
            is_sim_mode=False,
        )
        hub.on_session_end.assert_not_called()

    def test_all_failures_nonfatal(self):
        hippo = MagicMock()
        hippo.flush.side_effect = RuntimeError("flush error")
        hippo.config = MagicMock()
        hippo.config.persistence_path = "/tmp/test.json"
        hippo.save.side_effect = RuntimeError("save error")
        hub = MagicMock()
        hub.on_session_end.side_effect = RuntimeError("end error")
        # Should not raise
        end_bio_session(
            memory_hub=hub,
            memory_hub_enabled=True,
            hippocampus=hippo,
            is_sim_mode=False,
        )
