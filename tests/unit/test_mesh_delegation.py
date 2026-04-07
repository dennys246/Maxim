"""Tests for mesh task delegation (Phase 5) and distributed planning (Phase 6).

Covers:
- TaskDelegator: find_best_peer, delegate (sync blocking), handle_response, loop detection
- TaskReceiver: evaluate_proposal (depth, cycle, queue, tool, NAc checks)
- AdaptivePlanner._tag_delegatable_subgoals: peer selection, gated peer skip
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from maxim.mesh.admission import MeshAdmissionControl
from maxim.mesh.peer_info import PeerInfo
from maxim.mesh.peer_registry import PeerRegistry
from maxim.mesh.task_delegation import TaskDelegator, TaskReceiver


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_peer_with_identity(
    peer_id: str,
    tools: list[str],
    top_tools: list[dict] | None = None,
) -> PeerInfo:
    """Create a PeerInfo with a mock AgentIdentity."""
    peer = PeerInfo(peer_id=peer_id, base_url=f"http://{peer_id}:8080/v1")
    peer.is_alive = True
    identity = MagicMock()
    identity.available_tools = tools
    identity.top_tools = top_tools or []
    peer.identity = identity
    return peer


def _make_registry(*peers: PeerInfo) -> PeerRegistry:
    reg = PeerRegistry()
    for p in peers:
        info = reg.register(p.peer_id, p.base_url)
        info.identity = p.identity
        info.is_alive = p.is_alive
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# TaskDelegator
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskDelegatorFindBestPeer:
    def test_finds_peer_with_required_tool(self):
        peer = _make_peer_with_identity("robot-A", ["grasp", "look"])
        reg = _make_registry(peer)
        delegator = TaskDelegator(reg, MagicMock(), "local")

        best = delegator.find_best_peer({"tool_name": "grasp"})
        assert best is not None
        assert best.peer_id == "robot-A"

    def test_returns_none_when_no_peer_has_tool(self):
        peer = _make_peer_with_identity("robot-A", ["look"])
        reg = _make_registry(peer)
        delegator = TaskDelegator(reg, MagicMock(), "local")

        assert delegator.find_best_peer({"tool_name": "fly"}) is None

    def test_returns_none_without_tool_name(self):
        reg = PeerRegistry()
        delegator = TaskDelegator(reg, MagicMock(), "local")
        assert delegator.find_best_peer({"description": "do something"}) is None

    def test_skips_dead_peers(self):
        peer = _make_peer_with_identity("robot-A", ["grasp"])
        peer.is_alive = False
        reg = _make_registry(peer)
        delegator = TaskDelegator(reg, MagicMock(), "local")

        assert delegator.find_best_peer({"tool_name": "grasp"}) is None

    def test_skips_gated_peers(self):
        peer = _make_peer_with_identity("robot-A", ["grasp"])
        reg = _make_registry(peer)
        admission = MeshAdmissionControl()
        admission.gate_peer("robot-A", 60.0, "test gate")
        delegator = TaskDelegator(reg, MagicMock(), "local", admission=admission)

        assert delegator.find_best_peer({"tool_name": "grasp"}) is None

    def test_skips_peers_without_identity(self):
        peer = PeerInfo(peer_id="robot-A", base_url="http://a")
        peer.is_alive = True
        peer.identity = None
        reg = _make_registry(peer)
        # Re-set identity to None (register helper may have set it)
        reg.get_peer("robot-A").identity = None
        delegator = TaskDelegator(reg, MagicMock(), "local")

        assert delegator.find_best_peer({"tool_name": "grasp"}) is None

    def test_picks_peer_with_highest_success_rate(self):
        peer_a = _make_peer_with_identity(
            "robot-A",
            ["grasp"],
            top_tools=[{"name": "grasp", "success_rate": 0.7}],
        )
        peer_b = _make_peer_with_identity(
            "robot-B",
            ["grasp"],
            top_tools=[{"name": "grasp", "success_rate": 0.95}],
        )
        reg = _make_registry(peer_a, peer_b)
        delegator = TaskDelegator(reg, MagicMock(), "local")

        best = delegator.find_best_peer({"tool_name": "grasp"})
        assert best is not None
        assert best.peer_id == "robot-B"


class TestTaskDelegatorDelegate:
    def test_delegate_sends_and_blocks(self):
        """delegate() sends a GOAL_PROPOSAL and blocks until response."""
        reg = PeerRegistry()
        reg.register("peer1", "http://peer1")
        channel = MagicMock()
        channel.send_mesh.return_value = True

        delegator = TaskDelegator(reg, channel, "local_node")

        # Simulate response arriving from another thread
        def respond():
            time.sleep(0.05)
            # Find the pending correlation_id
            with delegator._lock:
                cid = next(iter(delegator._pending))
            delegator.handle_response(cid, {"status": "completed", "result": "ok"})

        thread = threading.Thread(target=respond)
        thread.start()

        result = delegator.delegate(
            {"tool_name": "look", "description": "look around"},
            reg.get_peer("peer1"),
            timeout_s=5.0,
        )

        thread.join()
        assert result is not None
        assert result["status"] == "completed"

    def test_delegate_returns_none_on_timeout(self):
        reg = PeerRegistry()
        reg.register("peer1", "http://peer1")
        channel = MagicMock()
        channel.send_mesh.return_value = True

        delegator = TaskDelegator(reg, channel, "local_node")
        result = delegator.delegate(
            {"tool_name": "look"},
            reg.get_peer("peer1"),
            timeout_s=0.05,
        )
        assert result is None

    def test_delegate_returns_none_on_send_failure(self):
        reg = PeerRegistry()
        reg.register("peer1", "http://peer1")
        channel = MagicMock()
        channel.send_mesh.return_value = False  # Send failed

        delegator = TaskDelegator(reg, channel, "local_node")
        result = delegator.delegate(
            {"tool_name": "look"},
            reg.get_peer("peer1"),
        )
        assert result is None

    def test_delegate_injects_delegation_tracking(self):
        reg = PeerRegistry()
        reg.register("peer1", "http://peer1")
        channel = MagicMock()
        channel.send_mesh.return_value = True

        delegator = TaskDelegator(reg, channel, "local_node")

        # Start delegation in background (it will time out)
        def do_delegate():
            delegator.delegate({"tool_name": "look"}, reg.get_peer("peer1"), timeout_s=0.05)

        t = threading.Thread(target=do_delegate)
        t.start()
        t.join()

        # Check the payload sent to channel
        call_kwargs = channel.send_mesh.call_args
        payload = call_kwargs.kwargs.get("payload") or call_kwargs[1].get("payload")
        assert payload["delegation_depth"] == 1
        assert "local_node" in payload["delegation_chain"]

    def test_delegate_does_not_mutate_original_goal(self):
        reg = PeerRegistry()
        reg.register("peer1", "http://peer1")
        channel = MagicMock()
        channel.send_mesh.return_value = True

        delegator = TaskDelegator(reg, channel, "local_node")
        goal = {"tool_name": "look"}

        def do_delegate():
            delegator.delegate(goal, reg.get_peer("peer1"), timeout_s=0.05)

        t = threading.Thread(target=do_delegate)
        t.start()
        t.join()

        # Original dict untouched
        assert "delegation_depth" not in goal

    def test_pending_count(self):
        reg = PeerRegistry()
        delegator = TaskDelegator(reg, MagicMock(), "local")
        assert delegator.pending_count == 0


class TestTaskDelegatorHandleResponse:
    def test_handle_response_unknown_correlation(self):
        """Responses for unknown correlation IDs are silently dropped."""
        delegator = TaskDelegator(PeerRegistry(), MagicMock(), "local")
        delegator.handle_response("unknown_cid", {"result": "data"})
        # No crash


# ─────────────────────────────────────────────────────────────────────────────
# TaskReceiver
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskReceiverEvaluateProposal:
    def test_accepts_simple_goal(self):
        executor = MagicMock()
        executor.registry.get.return_value = MagicMock()  # Tool exists
        receiver = TaskReceiver("local", executor=executor)

        accepted, reason = receiver.evaluate_proposal(
            {"tool_name": "look", "delegation_depth": 0, "delegation_chain": []},
            "peer-A",
        )
        assert accepted is True
        assert reason == "accepted"

    def test_rejects_deep_delegation(self):
        receiver = TaskReceiver("local")
        accepted, reason = receiver.evaluate_proposal(
            {"tool_name": "look", "delegation_depth": 3},
            "peer-A",
        )
        assert accepted is False
        assert "too deep" in reason

    def test_rejects_delegation_cycle(self):
        receiver = TaskReceiver("local")
        accepted, reason = receiver.evaluate_proposal(
            {"tool_name": "look", "delegation_depth": 1, "delegation_chain": ["local"]},
            "peer-A",
        )
        assert accepted is False
        assert "cycle" in reason

    def test_rejects_when_overloaded(self):
        receiver = TaskReceiver("local")
        # Simulate full queue
        for _ in range(TaskReceiver.MAX_DELEGATION_QUEUE):
            receiver.begin_delegation()

        accepted, reason = receiver.evaluate_proposal(
            {"tool_name": "look", "delegation_depth": 0, "delegation_chain": []},
            "peer-A",
        )
        assert accepted is False
        assert "overloaded" in reason

    def test_rejects_missing_tool(self):
        executor = MagicMock()
        executor.registry.get.side_effect = KeyError("not found")
        receiver = TaskReceiver("local", executor=executor)

        accepted, reason = receiver.evaluate_proposal(
            {"tool_name": "fly", "delegation_depth": 0, "delegation_chain": []},
            "peer-A",
        )
        assert accepted is False
        assert "not available" in reason

    def test_rejects_no_tool_name(self):
        receiver = TaskReceiver("local")
        accepted, reason = receiver.evaluate_proposal(
            {"description": "do something", "delegation_depth": 0, "delegation_chain": []},
            "peer-A",
        )
        assert accepted is False
        assert "no tool" in reason

    def test_rejects_nac_negative_prediction(self):
        nac = MagicMock()
        prediction = MagicMock()
        prediction.predicted_valence.value = "negative"
        prediction.confidence = 0.9
        nac.predict.return_value = prediction

        executor = MagicMock()
        executor.registry.get.return_value = MagicMock()

        receiver = TaskReceiver("local", executor=executor, nac=nac)
        accepted, reason = receiver.evaluate_proposal(
            {"tool_name": "grasp", "delegation_depth": 0, "delegation_chain": []},
            "peer-A",
        )
        assert accepted is False
        assert "NAc predicts failure" in reason

    def test_accepts_nac_positive_prediction(self):
        nac = MagicMock()
        prediction = MagicMock()
        prediction.predicted_valence.value = "positive"
        prediction.confidence = 0.8
        nac.predict.return_value = prediction

        executor = MagicMock()
        executor.registry.get.return_value = MagicMock()

        receiver = TaskReceiver("local", executor=executor, nac=nac)
        accepted, _ = receiver.evaluate_proposal(
            {"tool_name": "grasp", "delegation_depth": 0, "delegation_chain": []},
            "peer-A",
        )
        assert accepted is True

    def test_nac_error_is_non_fatal(self):
        nac = MagicMock()
        nac.predict.side_effect = RuntimeError("nac broken")

        executor = MagicMock()
        executor.registry.get.return_value = MagicMock()

        receiver = TaskReceiver("local", executor=executor, nac=nac)
        accepted, _ = receiver.evaluate_proposal(
            {"tool_name": "look", "delegation_depth": 0, "delegation_chain": []},
            "peer-A",
        )
        assert accepted is True  # NAc failure doesn't block acceptance

    def test_no_executor_skips_tool_check(self):
        receiver = TaskReceiver("local", executor=None)
        accepted, _ = receiver.evaluate_proposal(
            {"tool_name": "anything", "delegation_depth": 0, "delegation_chain": []},
            "peer-A",
        )
        assert accepted is True

    def test_begin_end_delegation(self):
        receiver = TaskReceiver("local")
        assert receiver.active_delegations == 0
        receiver.begin_delegation()
        assert receiver.active_delegations == 1
        receiver.end_delegation()
        assert receiver.active_delegations == 0

    def test_end_delegation_clamps_to_zero(self):
        receiver = TaskReceiver("local")
        receiver.end_delegation()  # Shouldn't go negative
        assert receiver.active_delegations == 0


# ─────────────────────────────────────────────────────────────────────────────
# AdaptivePlanner._tag_delegatable_subgoals (Phase 6)
# ─────────────────────────────────────────────────────────────────────────────


class TestTagDelegatableSubgoals:
    def _make_planner(self, nac=None):
        from maxim.planning.adaptive_planner import AdaptivePlanner

        return AdaptivePlanner(nac=nac)

    def test_no_registry_returns_unchanged(self):
        planner = self._make_planner()
        actions = [{"tool_name": "look"}]
        result = planner._tag_delegatable_subgoals(actions)
        assert "_preferred_node" not in result[0]

    def test_tags_tool_not_locally_available(self):
        """Peer has tool, we don't → delegate."""
        planner = self._make_planner()
        peer = _make_peer_with_identity("robot-A", ["fly"])
        reg = _make_registry(peer)
        planner.set_mesh_context(peer_registry=reg, local_tools=["look", "grasp"])

        actions = [{"tool_name": "fly"}]
        result = planner._tag_delegatable_subgoals(actions)
        assert result[0].get("_preferred_node") == "robot-A"

    def test_does_not_tag_local_tool(self):
        """Tool available locally → don't delegate (even if peer has it too)."""
        planner = self._make_planner()
        peer = _make_peer_with_identity("robot-A", ["look"])
        reg = _make_registry(peer)
        planner.set_mesh_context(peer_registry=reg, local_tools=["look"])

        actions = [{"tool_name": "look"}]
        result = planner._tag_delegatable_subgoals(actions)
        assert "_preferred_node" not in result[0]

    def test_tags_peer_with_better_success_rate(self):
        """Peer has >0.8 success rate and local NAc predicts poorly → delegate."""
        nac = MagicMock()
        prediction = MagicMock()
        prediction.predicted_value = 0.3  # Low local prediction
        nac.predict.return_value = prediction

        planner = self._make_planner(nac=nac)
        peer = _make_peer_with_identity(
            "robot-A",
            ["grasp"],
            top_tools=[{"name": "grasp", "success_rate": 0.9}],
        )
        reg = _make_registry(peer)
        planner.set_mesh_context(peer_registry=reg, local_tools=["grasp"])

        actions = [{"tool_name": "grasp"}]
        result = planner._tag_delegatable_subgoals(actions)
        assert result[0].get("_preferred_node") == "robot-A"

    def test_skips_gated_peer(self):
        planner = self._make_planner()
        peer = _make_peer_with_identity("robot-A", ["fly"])
        reg = _make_registry(peer)
        admission = MeshAdmissionControl()
        admission.gate_peer("robot-A", 60.0, "misbehaving")
        planner.set_mesh_context(peer_registry=reg, admission=admission, local_tools=[])

        actions = [{"tool_name": "fly"}]
        result = planner._tag_delegatable_subgoals(actions)
        assert "_preferred_node" not in result[0]

    def test_skips_dead_peer(self):
        planner = self._make_planner()
        peer = _make_peer_with_identity("robot-A", ["fly"])
        peer.is_alive = False
        reg = _make_registry(peer)
        reg.get_peer("robot-A").is_alive = False
        planner.set_mesh_context(peer_registry=reg, local_tools=[])

        actions = [{"tool_name": "fly"}]
        result = planner._tag_delegatable_subgoals(actions)
        assert "_preferred_node" not in result[0]

    def test_actions_without_tool_name_skipped(self):
        planner = self._make_planner()
        planner.set_mesh_context(peer_registry=PeerRegistry(), local_tools=[])

        actions = [{"description": "think hard"}]
        result = planner._tag_delegatable_subgoals(actions)
        assert "_preferred_node" not in result[0]

    def test_multiple_actions_tagged_independently(self):
        planner = self._make_planner()
        peer = _make_peer_with_identity("robot-A", ["fly", "swim"])
        reg = _make_registry(peer)
        planner.set_mesh_context(peer_registry=reg, local_tools=["look"])

        actions = [
            {"tool_name": "look"},  # Local → no tag
            {"tool_name": "fly"},  # Remote only → tag
            {"tool_name": "swim"},  # Remote only → tag
        ]
        result = planner._tag_delegatable_subgoals(actions)
        assert "_preferred_node" not in result[0]
        assert result[1]["_preferred_node"] == "robot-A"
        assert result[2]["_preferred_node"] == "robot-A"
