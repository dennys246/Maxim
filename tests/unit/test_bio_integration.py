"""Tests for runtime/bio_integration.py — extracted from agent_loop.py."""

from __future__ import annotations

import pytest

from unittest.mock import MagicMock
from maxim.runtime.bio_integration import (
    capture_episodic_memory,
    record_plan_outcome,
    start_bio_session,
    end_bio_session,
    record_substrate_nodes,
    consume_substrate_nodes,
    record_pain_intensity,
    consume_pain_intensity,
    observe_episode,
    reset_agent_stash,
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

    def test_lightweight_consolidation_in_sim_mode(self):
        """Sim mode (deprecated is_sim_mode fallback) → lightweight consolidation."""
        hub = MagicMock()
        end_bio_session(
            memory_hub=hub,
            memory_hub_enabled=True,
            hippocampus=None,
            is_sim_mode=True,
        )
        hub.on_session_end.assert_not_called()
        hub.on_session_end_lightweight.assert_called_once()

    def test_default_consolidation_is_full(self):
        """HANDLE seam (b): with neither param set, default is FULL consolidation
        (the count-silent-failures default — wrongly-lightweight loses data)."""
        hub = MagicMock()
        end_bio_session(memory_hub=hub, memory_hub_enabled=True, hippocampus=None)
        hub.on_session_end.assert_called_once()
        hub.on_session_end_lightweight.assert_not_called()

    def test_explicit_consolidation_full_overrides_sim_flag(self):
        """A persistent HANDLE forces "full" even when driven through the sim loop:
        explicit consolidation wins over the deprecated is_sim_mode proxy."""
        hub = MagicMock()
        end_bio_session(
            memory_hub=hub,
            memory_hub_enabled=True,
            hippocampus=None,
            consolidation="full",
            is_sim_mode=True,  # would say lightweight — explicit param wins
        )
        hub.on_session_end.assert_called_once()
        hub.on_session_end_lightweight.assert_not_called()

    def test_explicit_consolidation_lightweight(self):
        """CC8 preserved: an explicit "lightweight" still gets the lightweight path."""
        hub = MagicMock()
        end_bio_session(
            memory_hub=hub,
            memory_hub_enabled=True,
            hippocampus=None,
            consolidation="lightweight",
        )
        hub.on_session_end_lightweight.assert_called_once()
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


@pytest.fixture(autouse=True)
def _reset_per_agent_stash():
    """Make sure module-level per-agent dicts don't leak between tests."""
    for agent in ("alpha", "bravo", "charlie", "agent"):
        reset_agent_stash(agent)
    yield
    for agent in ("alpha", "bravo", "charlie", "agent"):
        reset_agent_stash(agent)


class TestPerAgentSubstrateStash:
    """Per-agent substrate-node stash isolation (P4)."""

    def test_record_consume_roundtrip(self):
        record_substrate_nodes(("n1", "n2"), agent_id="alpha")
        assert consume_substrate_nodes(agent_id="alpha") == ("n1", "n2")
        # Subsequent consume returns empty (popped on first consume)
        assert consume_substrate_nodes(agent_id="alpha") == ()

    def test_two_agents_isolated(self):
        record_substrate_nodes(("alpha-1",), agent_id="alpha")
        record_substrate_nodes(("bravo-1", "bravo-2"), agent_id="bravo")
        assert consume_substrate_nodes(agent_id="alpha") == ("alpha-1",)
        assert consume_substrate_nodes(agent_id="bravo") == ("bravo-1", "bravo-2")

    def test_consume_unknown_agent_returns_empty(self):
        assert consume_substrate_nodes(agent_id="nobody") == ()

    def test_record_overwrites_per_agent(self):
        record_substrate_nodes(("first",), agent_id="alpha")
        record_substrate_nodes(("second",), agent_id="alpha")
        assert consume_substrate_nodes(agent_id="alpha") == ("second",)


class TestPerAgentPainIntensity:
    """Per-agent pain-intensity stash isolation (P4)."""

    def test_record_consume_roundtrip(self):
        record_pain_intensity(0.7, agent_id="alpha")
        assert consume_pain_intensity(agent_id="alpha") == 0.7
        assert consume_pain_intensity(agent_id="alpha") is None

    def test_two_agents_isolated(self):
        record_pain_intensity(0.4, agent_id="alpha")
        record_pain_intensity(0.9, agent_id="bravo")
        assert consume_pain_intensity(agent_id="alpha") == 0.4
        assert consume_pain_intensity(agent_id="bravo") == 0.9

    def test_max_merge_per_agent(self):
        record_pain_intensity(0.3, agent_id="alpha")
        record_pain_intensity(0.7, agent_id="alpha")
        record_pain_intensity(0.5, agent_id="alpha")  # lower, ignored
        assert consume_pain_intensity(agent_id="alpha") == 0.7

    def test_zero_returns_none(self):
        # Never recorded — pop default 0.0 → None
        assert consume_pain_intensity(agent_id="alpha") is None


class TestObserveEpisode:
    """observe_episode threads per-agent ticks and stash."""

    def test_per_agent_tick_counter(self):
        hippo_a = MagicMock()
        hippo_b = MagicMock()
        observe_episode(hippocampus=hippo_a, agent_id="alpha")
        observe_episode(hippocampus=hippo_a, agent_id="alpha")
        observe_episode(hippocampus=hippo_b, agent_id="bravo")
        # Alpha got ticks 1 and 2; bravo got tick 1.
        a_calls = hippo_a.observe_episode_event.call_args_list
        b_calls = hippo_b.observe_episode_event.call_args_list
        assert [c.args[0].tick for c in a_calls] == [1, 2]
        assert [c.args[0].tick for c in b_calls] == [1]

    def test_stash_consumed_only_for_owning_agent(self):
        hippo_a = MagicMock()
        hippo_b = MagicMock()
        record_substrate_nodes(("alpha-node",), agent_id="alpha")
        record_substrate_nodes(("bravo-node",), agent_id="bravo")
        observe_episode(hippocampus=hippo_a, agent_id="alpha")
        observe_episode(hippocampus=hippo_b, agent_id="bravo")
        evt_a = hippo_a.observe_episode_event.call_args.args[0]
        evt_b = hippo_b.observe_episode_event.call_args.args[0]
        assert evt_a.activated_nodes == ("alpha-node",)
        assert evt_b.activated_nodes == ("bravo-node",)

    def test_caller_provided_nodes_skip_stash_consume(self):
        # When the caller passes activated_nodes, the stash is NOT consumed.
        hippo = MagicMock()
        record_substrate_nodes(("stashed",), agent_id="alpha")
        observe_episode(
            hippocampus=hippo,
            agent_id="alpha",
            activated_nodes=("explicit",),
        )
        evt = hippo.observe_episode_event.call_args.args[0]
        assert evt.activated_nodes == ("explicit",)
        # Stash is still there for the next call
        assert consume_substrate_nodes(agent_id="alpha") == ("stashed",)
