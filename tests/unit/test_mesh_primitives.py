"""Tests for mesh primitives — AgentProfile, UMR, MeshMessage, LocalMessageBus."""

from __future__ import annotations

import pytest

from maxim.mesh.identity import AgentProfile
from maxim.mesh.naming import UMR, parse_umr
from maxim.mesh.message import MeshMessage, MeshMessageType
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
