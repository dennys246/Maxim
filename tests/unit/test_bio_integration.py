"""Tests for runtime/bio_integration.py — extracted from agent_loop.py."""

from __future__ import annotations

import pytest

from types import SimpleNamespace
from unittest.mock import MagicMock

from maxim.tools.base import ToolOutput
from maxim.runtime.bio_integration import (
    capture_episodic_memory,
    record_plan_outcome,
    bio_session,
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
        executor = MagicMock(spec=[])
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
            novelty=None,
            situation=None,
        )
        hippo.capture_from_loop_async.assert_called_once()

    def test_rpe_salience_boost(self):
        hippo = MagicMock()
        executor = MagicMock()
        rpe_result = ToolOutput(success=True, rpe=0.6)
        obs = {"source": "test", "salience": 0.5}
        capture_episodic_memory(
            hippocampus=hippo,
            executor=executor,
            observation=obs,
            state=MagicMock(),
            intent={"goal": "test"},
            action={"tool_name": "look", "params": {}},
            result=rpe_result,
            run_id="run-1",
            novelty=None,
            situation=None,
        )
        # Salience should be boosted: 0.5 + 0.6*0.5 = 0.8
        assert obs["salience"] == 0.8

    def test_rpe_salience_capped_at_1(self):
        hippo = MagicMock()
        executor = MagicMock()
        rpe_result = ToolOutput(success=True, rpe=1.0)  # |R - V| <= 1: V is bounded at every producer
        obs = {"source": "test", "salience": 0.9}
        capture_episodic_memory(
            hippocampus=hippo,
            executor=executor,
            observation=obs,
            state=MagicMock(),
            intent={"goal": "test"},
            action={"tool_name": "look", "params": {}},
            result=rpe_result,
            run_id="run-1",
            novelty=None,
            situation=None,
        )
        assert obs["salience"] <= 1.0

    def test_no_rpe_no_boost(self):
        hippo = MagicMock()
        executor = MagicMock()
        rpe_result = ToolOutput(success=True, rpe=0.0)
        obs = {"source": "test", "salience": 0.5}
        capture_episodic_memory(
            hippocampus=hippo,
            executor=executor,
            observation=obs,
            state=MagicMock(),
            intent={"goal": "test"},
            action={"tool_name": "look", "params": {}},
            result=rpe_result,
            run_id="run-1",
            novelty=None,
            situation=None,
        )
        assert obs["salience"] == 0.5

    def test_a_result_that_is_not_a_tool_output_gets_no_boost(self):
        obs = {"source": "test", "salience": 0.5}
        capture_episodic_memory(
            hippocampus=MagicMock(),
            executor=MagicMock(),
            observation=obs,
            state=MagicMock(),
            intent={"goal": "test"},
            action={"tool_name": "look", "params": {}},
            result=MagicMock(),  # no stamped surprise -> none inferred
            run_id="run-1",
            novelty=None,
            situation=None,
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
            novelty=None,
            situation=None,
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
            novelty=None,
            situation=None,
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


class TestBioSession:
    """The ONE seam that opens and ends an agent loop's memory session (#1438 PR 0)."""

    def _hub(self):
        hub = MagicMock()
        hub.on_session_start.return_value = SimpleNamespace(generation=1, start_stats={"restored": 1})
        return hub

    def test_claims_as_a_live_loop_and_starts_its_own_capture_worker(self):
        hub, hippo, owner = self._hub(), MagicMock(), object()
        with bio_session(memory_hub=hub, hippocampus=hippo, owner=owner) as s:
            hub.on_session_start.assert_called_once_with(owner=owner, live=True)
            assert s.ticket is hub.on_session_start.return_value and s.memory_hub_enabled
            assert s.capture is hippo.start_capture_worker.return_value
        hub.on_session_end.assert_called_once_with(s.ticket)  # a clean, non-sim exit: full
        hippo.stop_capture_worker.assert_called_once_with(s.capture)  # only ITS worker
        hub.release_session_claim.assert_called_once_with(s.ticket)

    def test_a_sim_run_ends_lightweight_and_an_override_wins(self):
        hub = self._hub()
        with bio_session(memory_hub=hub, hippocampus=None, owner=object()) as s:
            s.is_sim_mode = True
        hub.on_session_end_lightweight.assert_called_once()
        hub = self._hub()
        with bio_session(memory_hub=hub, hippocampus=None, owner=object(), consolidation="full") as s:
            s.is_sim_mode = True  # a persistent HANDLE forces "full" even through the sim loop
        hub.on_session_end.assert_called_once()

    def test_an_abnormal_exit_persists_lightweight_and_re_raises(self):
        hub, stopped = self._hub(), []
        with pytest.raises(KeyboardInterrupt):
            with bio_session(memory_hub=hub, hippocampus=None, owner=object()) as s:
                s.stop_default_network = lambda: stopped.append("dn")
                raise KeyboardInterrupt
        assert stopped == ["dn"]  # the DN stops first, on every path
        hub.on_session_end_lightweight.assert_called_once()
        hub.release_session_claim.assert_called_once()

    def test_its_own_errors_never_replace_an_in_flight_exception(self):
        hub = self._hub()
        hub.on_session_end_lightweight.side_effect = RuntimeError("end error")
        with pytest.raises(ValueError, match="the real one"):
            with bio_session(memory_hub=hub, hippocampus=None, owner=object()):
                raise ValueError("the real one")

    def test_a_claim_refusal_aborts_and_starts_nothing(self):
        from maxim.exceptions import SessionHeld

        hub, hippo = self._hub(), MagicMock()
        hub.on_session_start.side_effect = SessionHeld("held by a live loop")
        with pytest.raises(SessionHeld):
            with bio_session(memory_hub=hub, hippocampus=hippo, owner=object()):
                pass  # pragma: no cover - never entered
        hippo.start_capture_worker.assert_not_called()

    def test_no_hub_is_the_one_soft_skip(self):
        hippo = MagicMock()
        with bio_session(memory_hub=None, hippocampus=hippo, owner=object()) as s:
            assert s.ticket is None and not s.memory_hub_enabled
        hippo.stop_capture_worker.assert_called_once_with(s.capture)  # its worker still started AND stopped

    def test_a_capture_worker_failure_is_non_fatal(self):
        hippo = MagicMock()
        hippo.start_capture_worker.side_effect = RuntimeError("thread error")
        with bio_session(memory_hub=None, hippocampus=hippo, owner=object()) as s:
            assert s.capture is None

    def test_an_expired_stop_bound_skips_the_abnormal_persist(self):
        from maxim.utils.cancellation import request_stop
        from maxim.utils.deadline import Deadline

        hub = self._hub()
        request_stop("test", Deadline.after(0))
        hippo = MagicMock()
        with pytest.raises(RuntimeError):
            with bio_session(memory_hub=hub, hippocampus=hippo, owner=object()) as s:
                raise RuntimeError("boom")
        hub.on_session_end_lightweight.assert_not_called()
        hub.abandon_session.assert_called_once_with(s.ticket)  # closed without a persist: never left OPEN
        s.capture.stop.set.assert_called_once()  # the capture worker still stops (no flush: the bound is spent)
        hub.release_session_claim.assert_called_once()  # the claim is still released


class TestEndBioSession:
    """Test bio-system session teardown (the owner's end; #1438 PR 0 tickets)."""

    def test_flushes_and_saves_hippocampus(self):
        hippo = MagicMock()
        hippo.config = MagicMock()
        hippo.config.persistence_path = "/tmp/test.json"
        hippo.__len__ = MagicMock(return_value=10)
        worker = object()
        end_bio_session(memory_hub=None, ticket=None, hippocampus=hippo, capture=worker)
        hippo.flush.assert_called_once_with(timeout=5.0)
        hippo.stop_capture_worker.assert_called_once_with(worker)
        hippo.save.assert_called_once()

    def test_ends_memory_hub_session_with_its_ticket(self):
        hub, ticket = MagicMock(), object()
        hub.on_session_end.return_value = {"consolidated": 5}
        end_bio_session(memory_hub=hub, ticket=ticket, hippocampus=None)
        hub.on_session_end.assert_called_once_with(ticket)

    def test_default_consolidation_is_full(self):
        """HANDLE seam (b): the default is FULL consolidation (wrongly-lightweight loses data)."""
        hub = MagicMock()
        end_bio_session(memory_hub=hub, ticket=object(), hippocampus=None)
        hub.on_session_end.assert_called_once()
        hub.on_session_end_lightweight.assert_not_called()

    def test_explicit_consolidation_lightweight(self):
        """CC8 preserved: an explicit "lightweight" gets the lightweight path."""
        hub = MagicMock()
        end_bio_session(memory_hub=hub, ticket=object(), hippocampus=None, consolidation="lightweight")
        hub.on_session_end_lightweight.assert_called_once()
        hub.on_session_end.assert_not_called()

    def test_skips_hub_without_a_ticket(self):
        hub = MagicMock()
        end_bio_session(memory_hub=hub, ticket=None, hippocampus=None)
        hub.on_session_end.assert_not_called()

    def test_all_failures_nonfatal(self):
        hippo = MagicMock()
        hippo.flush.side_effect = RuntimeError("flush error")
        hippo.config = MagicMock()
        hippo.config.persistence_path = "/tmp/test.json"
        hippo.save.side_effect = RuntimeError("save error")
        hub = MagicMock()
        hub.on_session_end.side_effect = RuntimeError("end error")
        end_bio_session(memory_hub=hub, ticket=object(), hippocampus=hippo)  # should not raise


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
