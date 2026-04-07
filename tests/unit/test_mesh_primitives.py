"""Tests for mesh primitives — AgentProfile, AgentIdentity, UMR, MeshMessage, LocalMessageBus."""

from __future__ import annotations

import pytest

from maxim.mesh.identity import AgentProfile
from maxim.mesh.agent_identity import AgentIdentity
from maxim.mesh.naming import UMR, parse_umr
from maxim.mesh.message import MESH_PROTOCOL_VERSION, MeshMessage, MeshMessageType
from maxim.mesh.bus import LocalMessageBus


# ── AgentProfile ─────────────────────────────────────────────────────────────


class TestAgentProfile:
    def test_creation_defaults(self):
        p = AgentProfile(nickname="researcher", role="researcher")
        assert p.nickname == "researcher"
        assert p.role == "researcher"
        assert p.capabilities == []
        assert len(p.agent_id) == 12
        assert p.started_at > 0

    def test_display_name_short(self):
        p = AgentProfile(nickname="writer", role="writer")
        assert p.display_name == "writer"

    def test_display_name_truncation(self):
        p = AgentProfile(nickname="a" * 25, role="test")
        assert len(p.display_name) == 20
        assert p.display_name.endswith("...")

    def test_display_name_exact_20(self):
        p = AgentProfile(nickname="a" * 20, role="test")
        assert p.display_name == "a" * 20

    def test_roundtrip_dict(self):
        p = AgentProfile(
            nickname="reviewer",
            role="reviewer",
            capabilities=["read_section", "submit_review"],
            personality="critical academic",
        )
        d = p.to_dict()
        p2 = AgentProfile.from_dict(d)
        assert p2.nickname == p.nickname
        assert p2.role == p.role
        assert p2.capabilities == p.capabilities
        assert p2.personality == p.personality
        assert p2.agent_id == p.agent_id

    def test_from_dict_defaults(self):
        p = AgentProfile.from_dict({"nickname": "x", "role": "y"})
        assert p.capabilities == []
        assert p.personality == ""


# ── UMR ──────────────────────────────────────────────────────────────────────


class TestUMR:
    def test_build_and_str(self):
        u = UMR.build("researcher", "hippo", "exp_001")
        assert str(u) == "researcher.hippo.exp_001"

    def test_parse_valid(self):
        u = parse_umr("researcher.hippo.exp_001")
        assert u.nickname == "researcher"
        assert u.region == "hippo"
        assert u.ref_id == "exp_001"

    def test_parse_with_dots_in_id(self):
        u = parse_umr("writer.paper.methods.v2")
        assert u.nickname == "writer"
        assert u.region == "paper"
        assert u.ref_id == "methods.v2"

    def test_parse_invalid_too_few(self):
        with pytest.raises(ValueError, match="3 dot-separated"):
            parse_umr("researcher.hippo")

    def test_parse_invalid_single(self):
        with pytest.raises(ValueError, match="3 dot-separated"):
            parse_umr("researcher")

    def test_frozen(self):
        u = UMR.build("a", "b", "c")
        with pytest.raises(AttributeError):
            u.nickname = "d"  # type: ignore[misc]

    def test_roundtrip(self):
        original = "reviewer.review.round_1"
        assert str(parse_umr(original)) == original


# ── MeshMessage ──────────────────────────────────────────────────────────────


class TestMeshMessage:
    def test_creation(self):
        m = MeshMessage(
            sender="researcher",
            recipient="writer",
            msg_type=MeshMessageType.EXPERIMENT_DATA,
            payload={"experiments": [1, 2, 3]},
        )
        assert m.sender == "researcher"
        assert m.recipient == "writer"
        assert m.msg_type == MeshMessageType.EXPERIMENT_DATA
        assert len(m.msg_id) == 12

    def test_roundtrip_dict(self):
        m = MeshMessage(
            sender="reviewer",
            recipient="writer",
            msg_type=MeshMessageType.REVIEW_RESULT,
            payload={"verdict": "revise", "issues": ["missing data"]},
            in_reply_to="abc123",
        )
        d = m.to_dict()
        m2 = MeshMessage.from_dict(d)
        assert m2.sender == m.sender
        assert m2.recipient == m.recipient
        assert m2.msg_type == m.msg_type
        assert m2.payload == m.payload
        assert m2.in_reply_to == "abc123"

    def test_broadcast_recipient(self):
        m = MeshMessage(
            sender="orchestrator",
            recipient="*",
            msg_type=MeshMessageType.TASK_COMPLETE,
        )
        assert m.recipient == "*"


# ── LocalMessageBus ──────────────────────────────────────────────────────────


class TestLocalMessageBus:
    def test_send_and_receive(self):
        bus = LocalMessageBus()
        received = []
        bus.register("writer", lambda m: received.append(m))

        msg = MeshMessage(
            sender="researcher",
            recipient="writer",
            msg_type=MeshMessageType.EXPERIMENT_DATA,
            payload={"data": "test"},
        )
        bus.send(msg)

        assert len(received) == 1
        assert received[0].payload == {"data": "test"}

    def test_broadcast_excludes_sender(self):
        bus = LocalMessageBus()
        r_received = []
        w_received = []
        bus.register("researcher", lambda m: r_received.append(m))
        bus.register("writer", lambda m: w_received.append(m))

        msg = MeshMessage(
            sender="researcher",
            recipient="*",
            msg_type=MeshMessageType.TASK_COMPLETE,
        )
        bus.send(msg)

        assert len(r_received) == 0  # Sender excluded
        assert len(w_received) == 1

    def test_unregister(self):
        bus = LocalMessageBus()
        received = []
        bus.register("writer", lambda m: received.append(m))
        bus.unregister("writer")

        bus.send(
            MeshMessage(
                sender="researcher",
                recipient="writer",
                msg_type=MeshMessageType.EXPERIMENT_DATA,
            )
        )

        assert len(received) == 0

    def test_history_filtering(self):
        bus = LocalMessageBus()
        bus.register("writer", lambda m: None)
        bus.register("reviewer", lambda m: None)

        bus.send(MeshMessage(sender="researcher", recipient="writer", msg_type=MeshMessageType.EXPERIMENT_DATA))
        bus.send(MeshMessage(sender="writer", recipient="reviewer", msg_type=MeshMessageType.PAPER_DRAFT))
        bus.send(MeshMessage(sender="reviewer", recipient="writer", msg_type=MeshMessageType.REVIEW_RESULT))

        assert bus.message_count == 3
        assert len(bus.get_history(sender="researcher")) == 1
        assert len(bus.get_history(recipient="writer")) == 2
        assert len(bus.get_history(msg_type=MeshMessageType.PAPER_DRAFT)) == 1

    def test_handler_error_doesnt_crash_bus(self):
        bus = LocalMessageBus()

        def bad_handler(m: MeshMessage) -> None:
            raise RuntimeError("handler broke")

        received = []
        bus.register("writer", bad_handler)
        bus.register("writer", lambda m: received.append(m))

        bus.send(
            MeshMessage(
                sender="researcher",
                recipient="writer",
                msg_type=MeshMessageType.EXPERIMENT_DATA,
            )
        )

        # Second handler still ran despite first crashing
        assert len(received) == 1

    def test_clear(self):
        bus = LocalMessageBus()
        bus.send(MeshMessage(sender="a", recipient="b", msg_type=MeshMessageType.REQUEST))
        assert bus.message_count == 1
        bus.clear()
        assert bus.message_count == 0

    def test_multiple_handlers_same_nickname(self):
        bus = LocalMessageBus()
        counts = [0, 0]

        bus.register("writer", lambda m: counts.__setitem__(0, counts[0] + 1))
        bus.register("writer", lambda m: counts.__setitem__(1, counts[1] + 1))

        bus.send(MeshMessage(sender="r", recipient="writer", msg_type=MeshMessageType.REQUEST))
        assert counts == [1, 1]


# ── AgentIdentity (Phase 1) ────────────────────────────────────────────────


class TestAgentIdentity:
    def test_creation_minimal(self):
        profile = AgentProfile(nickname="desktop", role="general")
        identity = AgentIdentity(
            profile=profile,
            node_id="abcd1234",
            agent_name="desktop-coder",
        )
        assert identity.node_id == "abcd1234"
        assert identity.agent_name == "desktop-coder"
        assert identity.profile.nickname == "desktop"
        assert identity.episodic_memory_count == 0
        assert identity.inference_available is False

    def test_roundtrip_dict(self):
        profile = AgentProfile(
            nickname="reachy",
            role="embodied",
            capabilities=["grasp", "look_around"],
        )
        identity = AgentIdentity(
            profile=profile,
            node_id="node123",
            agent_name="reachy-kitchen",
            capabilities={"has_gpu": True, "vram_gb": 16.0},
            available_tools=["grasp", "look_around", "navigate"],
            available_skills=["exploration"],
            episodic_memory_count=500,
            causal_link_count=42,
            concept_count=15,
            top_tools=[{"name": "grasp", "uses": 100, "success_rate": 0.85}],
            top_causal_patterns=[{"event": "tool:grasp", "outcome": "success", "confidence": 0.9}],
            recent_domains=["object_manipulation"],
            inference_models=[{"profile": "mistral-7b", "device": "cuda", "vram_gb": 8.0}],
            inference_available=True,
        )
        d = identity.to_dict()
        restored = AgentIdentity.from_dict(d)
        assert restored.node_id == "node123"
        assert restored.agent_name == "reachy-kitchen"
        assert restored.profile.nickname == "reachy"
        assert restored.profile.capabilities == ["grasp", "look_around"]
        assert restored.episodic_memory_count == 500
        assert restored.causal_link_count == 42
        assert restored.inference_available is True
        assert len(restored.top_tools) == 1
        assert restored.inference_models[0]["profile"] == "mistral-7b"

    def test_from_dict_missing_optional_fields(self):
        d = {
            "profile": {"nickname": "test", "role": "test"},
            "node_id": "n1",
            "agent_name": "test-agent",
        }
        identity = AgentIdentity.from_dict(d)
        assert identity.episodic_memory_count == 0
        assert identity.available_tools == []
        assert identity.inference_available is False

    def test_build_from_subsystems_no_subsystems(self):
        profile = AgentProfile(nickname="bare", role="test")
        identity = AgentIdentity.build_from_subsystems(
            profile=profile,
            node_id="bare_node",
            agent_name="bare-agent",
        )
        assert identity.episodic_memory_count == 0
        assert identity.causal_link_count == 0
        assert identity.concept_count == 0
        assert identity.top_causal_patterns == []

    def test_build_from_subsystems_with_nac(self, nac, valence_positive):
        # Create some links for stats
        for i in range(5):
            nac.observe(
                event_type="tool",
                event_signature=f"tool:test_{i}",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        profile = AgentProfile(nickname="smart", role="test")
        identity = AgentIdentity.build_from_subsystems(
            profile=profile,
            node_id="smart_node",
            agent_name="smart-agent",
            nac=nac,
        )
        assert identity.causal_link_count == 5

    def test_node_id_distinct_from_profile_agent_id(self):
        profile = AgentProfile(nickname="test", role="test")
        identity = AgentIdentity(
            profile=profile,
            node_id="persistent_node_id",
            agent_name="test",
        )
        # node_id is persistent, profile.agent_id is session-scoped
        assert identity.node_id != identity.profile.agent_id
        assert identity.node_id == "persistent_node_id"


# ── MeshMessage Phase 2 additions ──────────────────────────────────────────


class TestMeshMessagePhase2:
    def test_new_message_types_exist(self):
        # Verify all Phase 2 types are defined
        assert MeshMessageType.HEARTBEAT
        assert MeshMessageType.IDENTITY_REQUEST
        assert MeshMessageType.IDENTITY_RESPONSE
        assert MeshMessageType.GOAL_PROPOSAL
        assert MeshMessageType.GOAL_ACCEPTED
        assert MeshMessageType.GOAL_REJECTED
        assert MeshMessageType.GOAL_RESULT
        assert MeshMessageType.EXPERIENCE_OFFER
        assert MeshMessageType.EXPERIENCE_REQUEST
        assert MeshMessageType.EXPERIENCE_RESPONSE
        assert MeshMessageType.CAUSAL_LINK_SHARE
        assert MeshMessageType.REFLECTION_SHARE
        assert MeshMessageType.PREDICT_REQUEST
        assert MeshMessageType.PREDICT_RESPONSE
        assert MeshMessageType.INFERENCE_REQUEST
        assert MeshMessageType.INFERENCE_RESPONSE

    def test_protocol_version_default(self):
        m = MeshMessage(sender="a", recipient="b", msg_type=MeshMessageType.HEARTBEAT)
        assert m.protocol_version == MESH_PROTOCOL_VERSION
        assert m.protocol_version == 1

    def test_correlation_id_default(self):
        m = MeshMessage(sender="a", recipient="b", msg_type=MeshMessageType.REQUEST)
        assert m.correlation_id == ""

    def test_roundtrip_with_new_fields(self):
        m = MeshMessage(
            sender="peer-A",
            recipient="peer-B",
            msg_type=MeshMessageType.GOAL_PROPOSAL,
            payload={"tool_name": "grasp", "delegation_depth": 1},
            correlation_id="corr_abc",
            protocol_version=1,
        )
        d = m.to_dict()
        assert d["protocol_version"] == 1
        assert d["correlation_id"] == "corr_abc"
        assert d["msg_type"] == "GOAL_PROPOSAL"

        restored = MeshMessage.from_dict(d)
        assert restored.protocol_version == 1
        assert restored.correlation_id == "corr_abc"
        assert restored.msg_type == MeshMessageType.GOAL_PROPOSAL

    def test_from_dict_rejects_future_version(self):
        d = {
            "sender": "a",
            "recipient": "b",
            "msg_type": "HEARTBEAT",
            "protocol_version": 99,
        }
        with pytest.raises(ValueError, match="Unsupported mesh protocol version 99"):
            MeshMessage.from_dict(d)

    def test_from_dict_accepts_lower_version(self):
        d = {
            "sender": "a",
            "recipient": "b",
            "msg_type": "HEARTBEAT",
            "protocol_version": 1,
        }
        m = MeshMessage.from_dict(d)
        assert m.protocol_version == 1

    def test_from_dict_backward_compat_no_version(self):
        """Messages from pre-Phase-2 peers have no protocol_version."""
        d = {
            "sender": "a",
            "recipient": "b",
            "msg_type": "EXPERIMENT_DATA",
            "payload": {"data": "old"},
        }
        m = MeshMessage.from_dict(d)
        assert m.protocol_version == 1
        assert m.correlation_id == ""

    def test_existing_research_types_still_work(self):
        """Existing research protocol messages are unaffected."""
        m = MeshMessage(
            sender="researcher",
            recipient="writer",
            msg_type=MeshMessageType.EXPERIMENT_DATA,
            payload={"experiments": [1]},
            in_reply_to="prev_msg",
        )
        d = m.to_dict()
        restored = MeshMessage.from_dict(d)
        assert restored.msg_type == MeshMessageType.EXPERIMENT_DATA
        assert restored.in_reply_to == "prev_msg"
