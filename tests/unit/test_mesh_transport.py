"""Tests for mesh transport: PeerInfo, PeerRegistry, PeerChannel, MeshAdmissionControl.

Covers Phase 3 + 3b of the agent mesh plan. Focuses on:
- PeerInfo liveness tracking
- PeerRegistry thread safety, bootstrap from peer config, lookup
- PeerChannel queueing, Channel interface, receive → percept
- MeshAdmissionControl rate limiting, burst detection, gating, ungating
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch


from maxim.mesh.admission import MeshAdmissionControl
from maxim.mesh.message import MeshMessage, MeshMessageType
from maxim.mesh.peer_channel import PeerChannel
from maxim.mesh.peer_info import PeerInfo
from maxim.mesh.peer_registry import PeerRegistry, _url_to_peer_id


# ─────────────────────────────────────────────────────────────────────────────
# PeerInfo
# ─────────────────────────────────────────────────────────────────────────────


class TestPeerInfo:
    def test_creation_defaults(self):
        p = PeerInfo(peer_id="node1", base_url="https://example.com/v1")
        assert p.peer_id == "node1"
        assert p.is_alive is True
        assert p.trust_level == "unknown"

    def test_touch_updates_last_seen(self):
        p = PeerInfo(peer_id="node1", base_url="http://localhost")
        old_seen = p.last_seen
        time.sleep(0.01)
        p.touch()
        assert p.last_seen > old_seen
        assert p.is_alive is True

    def test_check_alive_within_timeout(self):
        p = PeerInfo(peer_id="node1", base_url="http://localhost")
        p.HEARTBEAT_TIMEOUT_S = 60.0
        p.last_seen = time.time()
        assert p.check_alive() is True

    def test_check_alive_expired(self):
        p = PeerInfo(peer_id="node1", base_url="http://localhost")
        p.HEARTBEAT_TIMEOUT_S = 0.01
        p.last_seen = time.time() - 1.0  # Well past timeout
        assert p.check_alive() is False
        assert p.is_alive is False

    def test_to_dict(self):
        p = PeerInfo(
            peer_id="node1",
            base_url="https://example.com/v1",
            trust_level="verified",
            models=["mistral-7b"],
            device="gpu",
            vram_gb=16.0,
        )
        d = p.to_dict()
        assert d["peer_id"] == "node1"
        assert d["trust_level"] == "verified"
        assert d["models"] == ["mistral-7b"]
        assert d["vram_gb"] == 16.0
        # api_key should NOT appear in serialized form
        assert "api_key" not in d


# ─────────────────────────────────────────────────────────────────────────────
# PeerRegistry
# ─────────────────────────────────────────────────────────────────────────────


class TestPeerRegistry:
    def test_register_and_lookup(self):
        reg = PeerRegistry()
        info = reg.register("node1", "https://example.com/v1", trust_level="verified")
        assert info.peer_id == "node1"

        found = reg.get_peer("node1")
        assert found is not None
        assert found.base_url == "https://example.com/v1"

    def test_register_updates_existing(self):
        reg = PeerRegistry()
        reg.register("node1", "https://old.com/v1", trust_level="unknown")
        reg.register("node1", "https://new.com/v1", trust_level="verified")

        found = reg.get_peer("node1")
        assert found is not None
        assert found.base_url == "https://new.com/v1"
        assert found.trust_level == "verified"

    def test_unregister(self):
        reg = PeerRegistry()
        reg.register("node1", "https://example.com/v1")
        assert reg.unregister("node1") is True
        assert reg.get_peer("node1") is None
        assert reg.unregister("node1") is False  # Already gone

    def test_all_peers(self):
        reg = PeerRegistry()
        reg.register("a", "http://a")
        reg.register("b", "http://b")
        assert len(reg.all_peers()) == 2

    def test_alive_peers_filters_dead(self):
        reg = PeerRegistry()
        alive = reg.register("alive", "http://alive")
        alive.last_seen = time.time()

        dead = reg.register("dead", "http://dead")
        dead.HEARTBEAT_TIMEOUT_S = 0.001
        dead.last_seen = time.time() - 10.0

        result = reg.alive_peers()
        assert len(result) == 1
        assert result[0].peer_id == "alive"

    def test_peer_count(self):
        reg = PeerRegistry()
        assert reg.peer_count() == 0
        reg.register("a", "http://a")
        assert reg.peer_count() == 1

    def test_thread_safety(self):
        """Concurrent register/lookup from multiple threads."""
        reg = PeerRegistry()
        errors = []

        def register_batch(start: int) -> None:
            try:
                for i in range(50):
                    reg.register(f"node_{start + i}", f"http://node{start + i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_batch, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert reg.peer_count() == 200

    def test_to_dict(self):
        reg = PeerRegistry()
        reg.register("a", "http://a", trust_level="verified")
        reg.register("b", "http://b", trust_level="remote")
        result = reg.to_dict()
        assert len(result) == 2
        ids = {d["peer_id"] for d in result}
        assert ids == {"a", "b"}


class TestUrlToPeerId:
    def test_strips_protocol(self):
        assert _url_to_peer_id("https://maxim.example.com/v1") == "maxim.example.com"

    def test_strips_http(self):
        assert _url_to_peer_id("http://192.168.1.50:8080/v1") == "192.168.1.50:8080"

    def test_strips_common_ports(self):
        assert _url_to_peer_id("https://maxim.example.com:443/v1") == "maxim.example.com"

    def test_preserves_custom_port(self):
        assert _url_to_peer_id("http://192.168.1.50:9090/v1") == "192.168.1.50:9090"

    def test_no_protocol(self):
        assert _url_to_peer_id("maxim.example.com/v1") == "maxim.example.com"


class TestPeerRegistryBootstrap:
    def test_from_peer_config_no_config(self):
        with patch("maxim.peer.config.read_peer_config", return_value=None):
            reg = PeerRegistry.from_peer_config()
            assert reg.peer_count() == 0

    def test_from_peer_config_with_config(self):
        from maxim.peer.config import PeerConfig

        cfg = PeerConfig(url="https://maxim.example.com/v1", api_key="sk-test123")
        with patch("maxim.peer.config.read_peer_config", return_value=cfg):
            reg = PeerRegistry.from_peer_config()
            assert reg.peer_count() == 1
            peer = reg.all_peers()[0]
            assert peer.api_key == "sk-test123"
            assert peer.trust_level == "verified"
            assert "/v1" in peer.base_url

    def test_from_peer_config_cloud_url(self):
        from maxim.peer.config import PeerConfig

        cfg = PeerConfig(url="https://api.openai.com/v1", api_key="sk-cloud", is_cloud=True)
        with patch("maxim.peer.config.read_peer_config", return_value=cfg):
            reg = PeerRegistry.from_peer_config()
            peer = reg.all_peers()[0]
            assert peer.trust_level == "remote"


# ─────────────────────────────────────────────────────────────────────────────
# PeerChannel
# ─────────────────────────────────────────────────────────────────────────────


class TestPeerChannel:
    def _make_channel(self, registry=None, bus=None):
        if registry is None:
            registry = PeerRegistry()
        ch = PeerChannel(registry, local_agent_id="local_node", bus=bus)
        return ch

    def test_send_to_unknown_peer_returns_false(self):
        ch = self._make_channel()
        try:
            assert ch.send("nonexistent", '{"test": true}') is False
        finally:
            ch.stop()

    def test_send_mesh_enqueues(self):
        reg = PeerRegistry()
        reg.register("peer1", "http://peer1:8080/v1")
        ch = self._make_channel(registry=reg)
        try:
            result = ch.send_mesh(
                "peer1",
                {"tool_name": "grasp"},
                msg_type=MeshMessageType.GOAL_PROPOSAL,
            )
            assert result is True
            assert ch._send_queue.qsize() >= 1
        finally:
            ch.stop()

    def test_send_mesh_returns_false_when_stopped(self):
        reg = PeerRegistry()
        reg.register("peer1", "http://peer1")
        ch = self._make_channel(registry=reg)
        ch.stop()
        assert ch.send_mesh("peer1", {}) is False

    def test_channel_interface_compat(self):
        """PeerChannel satisfies the Channel ABC."""
        from maxim.comms.channels.base import Channel

        ch = self._make_channel()
        try:
            assert isinstance(ch, Channel)
        finally:
            ch.stop()

    def test_receive_publishes_percept(self):
        bus = MagicMock()
        ch = self._make_channel(bus=bus)
        try:
            msg = MeshMessage(
                sender="remote_peer",
                recipient="local_node",
                msg_type=MeshMessageType.GOAL_PROPOSAL,
                payload={"tool_name": "look_around"},
                correlation_id="corr1",
            )
            ch.receive(msg)

            bus.publish.assert_called_once()
            percept = bus.publish.call_args[0][0]
            assert percept.source == "mesh:remote_peer"
            assert percept.salience == 0.7
            assert percept.metadata["msg_type"] == "GOAL_PROPOSAL"
            assert percept.metadata["correlation_id"] == "corr1"
            assert percept.metadata["external"] is True
            assert json.loads(percept.content) == {"tool_name": "look_around"}
        finally:
            ch.stop()

    def test_receive_rejects_future_protocol(self):
        bus = MagicMock()
        ch = self._make_channel(bus=bus)
        try:
            msg = MeshMessage(
                sender="remote_peer",
                recipient="local_node",
                msg_type=MeshMessageType.HEARTBEAT,
                protocol_version=99,
            )
            ch.receive(msg)
            bus.publish.assert_not_called()
        finally:
            ch.stop()

    def test_receive_no_bus_doesnt_crash(self):
        ch = self._make_channel(bus=None)
        try:
            msg = MeshMessage(
                sender="remote_peer",
                recipient="local_node",
                msg_type=MeshMessageType.HEARTBEAT,
            )
            ch.receive(msg)  # Should not raise
        finally:
            ch.stop()

    def test_receive_touches_peer_liveness(self):
        reg = PeerRegistry()
        peer = reg.register("remote_peer", "http://remote")
        old_seen = peer.last_seen

        bus = MagicMock()
        ch = self._make_channel(registry=reg, bus=bus)
        try:
            time.sleep(0.01)
            msg = MeshMessage(
                sender="remote_peer",
                recipient="local_node",
                msg_type=MeshMessageType.HEARTBEAT,
            )
            ch.receive(msg)
            assert peer.last_seen > old_seen
        finally:
            ch.stop()

    def test_stats(self):
        ch = self._make_channel()
        try:
            s = ch.stats()
            assert s["messages_sent"] == 0
            assert s["messages_failed"] == 0
            assert s["messages_dropped"] == 0
            assert s["stopped"] is False
        finally:
            ch.stop()
            assert ch.stats()["stopped"] is True

    def test_drain_thread_retries_on_failure(self):
        """Verify the drain thread retries failed sends."""
        reg = PeerRegistry()
        reg.register("peer1", "http://peer1:8080/v1")
        ch = self._make_channel(registry=reg)
        try:
            calls = []
            done = threading.Event()

            def mock_post(peer, msg):
                calls.append(1)
                if len(calls) < 3:
                    return False  # Simulate HTTP non-200
                done.set()
                return True

            ch._post_message = mock_post  # type: ignore

            ch.send_mesh("peer1", {"test": True}, msg_type=MeshMessageType.HEARTBEAT)

            # Wait for the drain thread to process all retries
            done.wait(timeout=15.0)
            assert len(calls) == 3
            assert ch._messages_sent == 1
        finally:
            ch.stop()


# ─────────────────────────────────────────────────────────────────────────────
# MeshAdmissionControl
# ─────────────────────────────────────────────────────────────────────────────


class TestMeshAdmissionControl:
    def test_first_message_admitted(self):
        ac = MeshAdmissionControl()
        admitted, reason = ac.check("peer1", "verified")
        assert admitted is True
        assert reason == "ok"

    def test_rate_limit_triggers_gate(self):
        ac = MeshAdmissionControl()
        # Disable burst detection so we isolate rate limiting
        ac.BURST_THRESHOLD = 999

        # "unknown" trust has limit of 20
        for i in range(20):
            admitted, _ = ac.check("peer1", "unknown")
            assert admitted is True

        # 21st message should be gated
        admitted, reason = ac.check("peer1", "unknown")
        assert admitted is False
        assert "rate limit" in reason

    def test_verified_has_higher_limit(self):
        ac = MeshAdmissionControl()
        # Disable burst detection so we isolate rate limiting
        ac.BURST_THRESHOLD = 999

        # "verified" trust has limit of 120
        for i in range(120):
            admitted, _ = ac.check("peer1", "verified")
            assert admitted is True

        admitted, reason = ac.check("peer1", "verified")
        assert admitted is False
        assert "rate limit" in reason

    def test_burst_detection(self):
        ac = MeshAdmissionControl()
        ac.BURST_THRESHOLD = 5  # Lower for testing
        ac.BURST_WINDOW = 10.0

        for i in range(5):
            admitted, _ = ac.check("peer1", "verified")
            assert admitted is True

        # 6th message in same burst window triggers gate
        admitted, reason = ac.check("peer1", "verified")
        assert admitted is False
        assert "burst" in reason

    def test_escalating_gate_durations(self):
        ac = MeshAdmissionControl()
        ac.BURST_THRESHOLD = 2  # Very low for testing

        # First violation: 30s gate
        ac.check("peer1", "verified")
        ac.check("peer1", "verified")
        ac.check("peer1", "verified")  # Triggers burst gate

        state = ac._peers["peer1"]
        assert state.violations == 1
        first_duration = state.gated_until - time.monotonic()
        assert 25 < first_duration < 35  # ~30s

    def test_gated_peer_rejected(self):
        ac = MeshAdmissionControl()
        ac.gate_peer("peer1", 60.0, "test gate")

        admitted, reason = ac.check("peer1")
        assert admitted is False
        assert "test gate" in reason

    def test_ungate_peer(self):
        ac = MeshAdmissionControl()
        ac.gate_peer("peer1", 60.0, "test gate")
        assert ac.is_peer_gated("peer1") is True

        ac.ungate_peer("peer1")
        assert ac.is_peer_gated("peer1") is False

        # Should be admitted again
        admitted, _ = ac.check("peer1")
        assert admitted is True

    def test_is_peer_gated_unknown_peer(self):
        ac = MeshAdmissionControl()
        assert ac.is_peer_gated("nonexistent") is False

    def test_manual_gate_increments_violations(self):
        ac = MeshAdmissionControl()
        ac.gate_peer("peer1", 10.0, "manual")
        state = ac._peers["peer1"]
        assert state.violations == 1

    def test_get_status(self):
        ac = MeshAdmissionControl()
        ac.check("peer1", "verified")
        ac.check("peer2", "unknown")
        status = ac.get_status()
        assert len(status) == 2
        ids = {s["peer_id"] for s in status}
        assert ids == {"peer1", "peer2"}
        for s in status:
            assert "trust_level" in s
            assert "messages_received" in s
            assert "violations" in s
            assert "is_gated" in s

    def test_window_resets_after_expiry(self):
        ac = MeshAdmissionControl()
        ac.WINDOW_SECONDS = 0.01  # Very short window for testing
        ac.BURST_THRESHOLD = 999  # Disable burst so we test window reset

        # Fill up the window (unknown = 20 limit)
        for _ in range(20):
            ac.check("peer1", "unknown")

        # Wait for window to expire
        time.sleep(0.02)

        # Should be admitted again (window reset)
        admitted, _ = ac.check("peer1", "unknown")
        assert admitted is True

    def test_thread_safety(self):
        """Concurrent admission checks from multiple threads don't crash."""
        ac = MeshAdmissionControl()
        ac.BURST_THRESHOLD = 999  # Disable burst — we're testing lock safety
        errors = []

        def check_batch(peer_id: str) -> None:
            try:
                for _ in range(50):
                    ac.check(peer_id, "verified")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_batch, args=(f"peer_{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Each peer got 50 messages, all under 120 verified limit
        status = ac.get_status()
        assert len(status) == 4
        for s in status:
            assert s["messages_received"] == 50

    def test_reset(self):
        ac = MeshAdmissionControl()
        ac.check("peer1", "verified")
        ac.gate_peer("peer2", 60.0, "test")
        ac.reset()
        assert ac.get_status() == []
        assert ac.is_peer_gated("peer2") is False
