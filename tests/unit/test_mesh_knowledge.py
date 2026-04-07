"""Tests for mesh knowledge sharing: ExperienceBroker, KnowledgeProvider/Receiver protocol.

Covers Phase 4 of the agent mesh plan. Tests:
- Protocol conformance (runtime_checkable)
- ExperienceBroker registration, routing, get_shareable, import_knowledge
- CausalLinkProvider: filtering by confidence/observations, tool_name filter
- CausalLinkReceiver: transfer discount, deduplication, provenance tagging
- ReflectionProvider: filtering for reflection text
- ReflectionReceiver: salience caps per trust level
- Custom provider/receiver registration (extensibility)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

from maxim.mesh.knowledge import (
    CausalLinkProvider,
    CausalLinkReceiver,
    ExperienceBroker,
    KnowledgeProvider,
    KnowledgeReceiver,
    MotorProgramProvider,
    MotorProgramReceiver,
    ReflectionProvider,
    ReflectionReceiver,
)


# ─────────────────────────────────────────────────────────────────────────────
# Protocol conformance
# ─────────────────────────────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_causal_link_provider_is_knowledge_provider(self):
        nac = MagicMock()
        p = CausalLinkProvider(nac)
        assert isinstance(p, KnowledgeProvider)

    def test_causal_link_receiver_is_knowledge_receiver(self):
        nac = MagicMock()
        r = CausalLinkReceiver(nac)
        assert isinstance(r, KnowledgeReceiver)

    def test_reflection_provider_is_knowledge_provider(self):
        hippo = MagicMock()
        p = ReflectionProvider(hippo)
        assert isinstance(p, KnowledgeProvider)

    def test_reflection_receiver_is_knowledge_receiver(self):
        hippo = MagicMock()
        r = ReflectionReceiver(hippo)
        assert isinstance(r, KnowledgeReceiver)

    def test_custom_provider_conforms(self):
        """A plain class with the right methods satisfies the protocol."""

        class MyProvider:
            def knowledge_type(self) -> str:
                return "custom"

            def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict]:
                return [{"value": 42}]

        assert isinstance(MyProvider(), KnowledgeProvider)


# ─────────────────────────────────────────────────────────────────────────────
# ExperienceBroker
# ─────────────────────────────────────────────────────────────────────────────


class TestExperienceBroker:
    def test_empty_broker(self):
        broker = ExperienceBroker()
        assert broker.registered_types == []
        assert broker.get_shareable() == []
        assert broker.import_knowledge("causal_link", {}, "peer", "verified") is False

    def test_register_provider(self):
        broker = ExperienceBroker()

        class FakeProvider:
            def knowledge_type(self) -> str:
                return "test_type"

            def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict]:
                return [{"data": "hello"}]

        broker.register_provider(FakeProvider())
        assert "test_type" in broker.provider_types
        assert "test_type" in broker.registered_types

    def test_register_receiver(self):
        broker = ExperienceBroker()

        class FakeReceiver:
            def knowledge_type(self) -> str:
                return "test_type"

            def import_knowledge(self, data: dict, source: str, trust: str) -> bool:
                return True

        broker.register_receiver(FakeReceiver())
        assert "test_type" in broker.receiver_types

    def test_get_shareable_by_type(self):
        broker = ExperienceBroker()

        class Provider:
            def knowledge_type(self) -> str:
                return "widget"

            def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict]:
                return [{"id": 1}, {"id": 2}]

        broker.register_provider(Provider())
        result = broker.get_shareable(knowledge_type="widget")
        assert len(result) == 2
        assert result[0]["knowledge_type"] == "widget"
        assert result[0]["id"] == 1

    def test_get_shareable_all_types(self):
        broker = ExperienceBroker()

        class ProviderA:
            def knowledge_type(self) -> str:
                return "alpha"

            def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict]:
                return [{"src": "a"}]

        class ProviderB:
            def knowledge_type(self) -> str:
                return "beta"

            def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict]:
                return [{"src": "b"}]

        broker.register_provider(ProviderA())
        broker.register_provider(ProviderB())
        result = broker.get_shareable()
        types = {r["knowledge_type"] for r in result}
        assert types == {"alpha", "beta"}

    def test_get_shareable_respects_limit(self):
        broker = ExperienceBroker()

        class BigProvider:
            def knowledge_type(self) -> str:
                return "big"

            def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict]:
                return [{"i": i} for i in range(limit)]

        broker.register_provider(BigProvider())
        result = broker.get_shareable(limit=3)
        assert len(result) == 3

    def test_get_shareable_unknown_type_returns_empty(self):
        broker = ExperienceBroker()
        assert broker.get_shareable(knowledge_type="nonexistent") == []

    def test_import_routes_to_receiver(self):
        broker = ExperienceBroker()
        imported = []

        class Receiver:
            def knowledge_type(self) -> str:
                return "gadget"

            def import_knowledge(self, data: dict, source: str, trust: str) -> bool:
                imported.append((data, source, trust))
                return True

        broker.register_receiver(Receiver())
        result = broker.import_knowledge("gadget", {"val": 1}, "peer-A", "verified")
        assert result is True
        assert len(imported) == 1
        assert imported[0] == ({"val": 1}, "peer-A", "verified")

    def test_import_unknown_type_returns_false(self):
        broker = ExperienceBroker()
        assert broker.import_knowledge("unknown", {}, "peer", "unknown") is False

    def test_filters_passed_to_provider(self):
        broker = ExperienceBroker()
        received_filters: dict[str, Any] = {}

        class FilterProvider:
            def knowledge_type(self) -> str:
                return "filtered"

            def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict]:
                received_filters.update(filters)
                return []

        broker.register_provider(FilterProvider())
        broker.get_shareable(knowledge_type="filtered", tool_name="grasp")
        assert received_filters.get("tool_name") == "grasp"


# ─────────────────────────────────────────────────────────────────────────────
# CausalLinkProvider
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalLinkProvider:
    def test_knowledge_type(self):
        p = CausalLinkProvider(MagicMock())
        assert p.knowledge_type() == "causal_link"

    def test_filters_by_confidence_and_observations(self, nac, valence_positive):
        """Only links with confidence >= 0.6 and observations >= 3 are shared."""
        # Create a link with enough observations to build confidence
        for _ in range(5):
            nac.observe(
                event_type="tool",
                event_signature="tool:grasp",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        p = CausalLinkProvider(nac)
        shareable = p.get_shareable()

        # Check all returned links meet thresholds
        for item in shareable:
            assert item.get("confidence", 0) >= p.SHARE_MIN_CONFIDENCE
            assert item.get("observation_count", 0) >= p.SHARE_MIN_OBSERVATIONS

    def test_filters_by_tool_name(self, nac, valence_positive):
        for _ in range(5):
            nac.observe(
                event_type="tool",
                event_signature="tool:look",
                outcome_type="result",
                outcome_signature="found",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )
        for _ in range(5):
            nac.observe(
                event_type="tool",
                event_signature="tool:grasp",
                outcome_type="result",
                outcome_signature="grabbed",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        p = CausalLinkProvider(nac)
        look_only = p.get_shareable(tool_name="look")
        for item in look_only:
            assert "look" in item.get("event_signature", "")

    def test_empty_nac_returns_empty(self, nac):
        p = CausalLinkProvider(nac)
        assert p.get_shareable() == []

    def test_respects_limit(self, nac, valence_positive):
        for i in range(10):
            for _ in range(5):
                nac.observe(
                    event_type="tool",
                    event_signature=f"tool:t{i}",
                    outcome_type="result",
                    outcome_signature=f"ok{i}",
                    outcome_valence=valence_positive,
                    delta_seconds=1.0,
                )
        p = CausalLinkProvider(nac)
        assert len(p.get_shareable(limit=3)) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# CausalLinkReceiver
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalLinkReceiver:
    def _make_link_data(self, event_sig: str = "tool:grasp", outcome_sig: str = "success") -> dict:
        from maxim.decisions.causal_link import CausalLink, TemporalDelta, Valence

        link = CausalLink(
            id=f"link_{event_sig}_{outcome_sig}",
            event_type="tool",
            event_signature=event_sig,
            event_context={},
            outcome_type="result",
            outcome_signature=outcome_sig,
            outcome_valence=Valence.POSITIVE,
            predicted_value=0.8,
            temporal_delta=TemporalDelta(observed_deltas=(1.0,)),
            observation_count=5,
            confidence=0.9,
            last_observed=time.time(),
        )
        return link.to_dict()

    def test_knowledge_type(self):
        r = CausalLinkReceiver(MagicMock())
        assert r.knowledge_type() == "causal_link"

    def test_import_applies_transfer_discount(self, nac):
        r = CausalLinkReceiver(nac)
        data = self._make_link_data()
        result = r.import_knowledge(data, "peer-A", "verified")
        assert result is True

        links = nac.get_links_for_event("tool:grasp")
        assert len(links) == 1
        imported = links[0]
        # Verified discount = 0.5, original confidence = 0.9
        assert abs(imported.confidence - 0.45) < 0.01
        assert imported.predicted_value == 0.5  # Reset
        assert imported.observation_count == 1

    def test_import_tags_provenance(self, nac):
        r = CausalLinkReceiver(nac)
        data = self._make_link_data()
        r.import_knowledge(data, "peer-B", "remote")

        links = nac.get_links_for_event("tool:grasp")
        ctx = links[0].event_context
        assert ctx["_imported_from"] == "peer-B"
        assert ctx["_import_trust"] == "remote"
        assert "_import_timestamp" in ctx

    def test_import_deduplicates(self, nac):
        r = CausalLinkReceiver(nac)
        data = self._make_link_data()
        assert r.import_knowledge(data, "peer-A", "verified") is True
        assert r.import_knowledge(data, "peer-A", "verified") is False  # Duplicate

        links = nac.get_links_for_event("tool:grasp")
        assert len(links) == 1

    def test_unknown_trust_gets_lowest_discount(self, nac):
        r = CausalLinkReceiver(nac)
        data = self._make_link_data(event_sig="tool:unknown_test")
        r.import_knowledge(data, "sketchy-peer", "unknown")

        links = nac.get_links_for_event("tool:unknown_test")
        # Unknown discount = 0.1, original confidence = 0.9
        assert abs(links[0].confidence - 0.09) < 0.01

    def test_different_trust_levels(self, nac):
        """Each trust level applies its own discount."""
        r = CausalLinkReceiver(nac)
        tests = [
            ("tool:v", "verified", 0.5),
            ("tool:d", "discovered", 0.3),
            ("tool:r", "remote", 0.3),
            ("tool:u", "unknown", 0.1),
        ]
        for sig, trust, expected_discount in tests:
            data = self._make_link_data(event_sig=sig)
            r.import_knowledge(data, "peer", trust)
            links = nac.get_links_for_event(sig)
            assert abs(links[0].confidence - 0.9 * expected_discount) < 0.01, (
                f"trust={trust}: expected {0.9 * expected_discount}, got {links[0].confidence}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ReflectionProvider
# ─────────────────────────────────────────────────────────────────────────────


class TestReflectionProvider:
    def test_knowledge_type(self):
        p = ReflectionProvider(MagicMock())
        assert p.knowledge_type() == "reflection"

    def test_filters_for_reflection_text(self):
        """Only memories with outcome.result.reflection are returned."""
        hippo = MagicMock()
        # Memory with reflection
        m1 = MagicMock()
        m1.outcome.result = {"reflection": "I learned something"}
        m1.to_dict.return_value = {"id": "m1", "has_reflection": True}
        # Memory without reflection
        m2 = MagicMock()
        m2.outcome.result = {"output": "just data"}
        m2.to_dict.return_value = {"id": "m2"}
        # Memory with no outcome
        m3 = MagicMock(spec=[])  # No outcome attr

        hippo.recall.return_value = [m1, m2, m3]
        p = ReflectionProvider(hippo)
        result = p.get_shareable()
        assert len(result) == 1
        assert result[0]["id"] == "m1"

    def test_passes_tool_filter(self):
        hippo = MagicMock()
        hippo.recall.return_value = []
        p = ReflectionProvider(hippo)
        p.get_shareable(tool_name="grasp")
        hippo.recall.assert_called_once_with(limit=10, tool="grasp")


# ─────────────────────────────────────────────────────────────────────────────
# ReflectionReceiver
# ─────────────────────────────────────────────────────────────────────────────


class TestReflectionReceiver:
    def test_knowledge_type(self):
        r = ReflectionReceiver(MagicMock())
        assert r.knowledge_type() == "reflection"

    def test_salience_caps_per_trust(self):
        """Each trust level caps salience differently."""
        tests = [
            ("verified", 0.5),
            ("discovered", 0.35),
            ("remote", 0.35),
            ("unknown", 0.15),
        ]
        for trust, expected_cap in tests:
            hippo = MagicMock()
            hippo.capture.return_value = "mem_id"

            # Patch EpisodicMemory.from_dict to return a mock memory
            from unittest.mock import patch

            mock_mem = MagicMock()
            mock_mem.perception.salience = 0.9  # High original salience
            mock_mem.perception.observations = {}

            with patch("maxim.memory.types.EpisodicMemory") as mock_cls:
                mock_cls.from_dict.return_value = mock_mem
                r = ReflectionReceiver(hippo)
                result = r.import_knowledge({"data": "test"}, "peer", trust)

            assert result is True
            assert mock_mem.perception.salience <= expected_cap + 0.001, (
                f"trust={trust}: salience {mock_mem.perception.salience} > cap {expected_cap}"
            )

    def test_tags_provenance(self):
        hippo = MagicMock()
        hippo.capture.return_value = "mem_id"

        from unittest.mock import patch

        mock_mem = MagicMock()
        mock_mem.perception.salience = 0.9
        mock_mem.perception.observations = {}

        with patch("maxim.memory.types.EpisodicMemory") as mock_cls:
            mock_cls.from_dict.return_value = mock_mem
            r = ReflectionReceiver(hippo)
            r.import_knowledge({"data": "test"}, "peer-C", "remote")

        assert mock_mem.perception.observations["_imported_from"] == "peer-C"
        assert mock_mem.perception.observations["_import_trust"] == "remote"

    def test_returns_false_on_capture_failure(self):
        hippo = MagicMock()
        hippo.capture.return_value = None  # Capture failed

        from unittest.mock import patch

        mock_mem = MagicMock()
        mock_mem.perception.salience = 0.5
        mock_mem.perception.observations = {}

        with patch("maxim.memory.types.EpisodicMemory") as mock_cls:
            mock_cls.from_dict.return_value = mock_mem
            r = ReflectionReceiver(hippo)
            assert r.import_knowledge({}, "peer", "unknown") is False


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: broker with real adapters
# ─────────────────────────────────────────────────────────────────────────────


class TestBrokerWithAdapters:
    def test_full_causal_link_flow(self, nac, valence_positive):
        """Provider exports → broker routes → receiver imports."""
        # Build up some local knowledge
        for _ in range(5):
            nac.observe(
                event_type="tool",
                event_signature="tool:navigate",
                outcome_type="result",
                outcome_signature="arrived",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        broker = ExperienceBroker()
        broker.register_provider(CausalLinkProvider(nac))

        # Create a separate NAc for the "receiving" agent
        from maxim.decisions.nac import NAc, NACConfig

        receiver_nac = NAc(NACConfig(max_links=100))
        broker.register_receiver(CausalLinkReceiver(receiver_nac))

        # Export from provider
        shareable = broker.get_shareable(knowledge_type="causal_link")

        # Import into receiver
        for item in shareable:
            kt = item.pop("knowledge_type")
            broker.import_knowledge(kt, item, "peer-sender", "verified")

        # Verify receiver got the links
        links = receiver_nac.get_links_for_event("tool:navigate")
        assert len(links) >= 1
        imported = links[0]
        assert imported.event_context["_imported_from"] == "peer-sender"
        assert imported.confidence < 0.9  # Transfer discount applied

    def test_multiple_types_registered(self, nac, valence_positive):
        """Broker with both causal_link and reflection providers."""
        broker = ExperienceBroker()
        broker.register_provider(CausalLinkProvider(nac))

        hippo = MagicMock()
        hippo.recall.return_value = []
        broker.register_provider(ReflectionProvider(hippo))

        assert sorted(broker.provider_types) == ["causal_link", "reflection"]
        assert sorted(broker.registered_types) == ["causal_link", "reflection"]

    def test_extensibility_custom_type(self):
        """A custom knowledge type works without modifying the broker."""
        broker = ExperienceBroker()

        class ThresholdProvider:
            def knowledge_type(self) -> str:
                return "adaptive_threshold"

            def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict]:
                return [{"startle_threshold": 0.7, "orienting_threshold": 0.4}]

        class ThresholdReceiver:
            def __init__(self) -> None:
                self.received: list[dict] = []

            def knowledge_type(self) -> str:
                return "adaptive_threshold"

            def import_knowledge(self, data: dict, source: str, trust: str) -> bool:
                if trust == "unknown":
                    return False  # Require at least discovered trust
                self.received.append(data)
                return True

        receiver = ThresholdReceiver()
        broker.register_provider(ThresholdProvider())
        broker.register_receiver(receiver)

        # Export
        shareable = broker.get_shareable(knowledge_type="adaptive_threshold")
        assert len(shareable) == 1
        assert shareable[0]["knowledge_type"] == "adaptive_threshold"

        # Import (verified = accepted)
        item = shareable[0].copy()
        kt = item.pop("knowledge_type")
        assert broker.import_knowledge(kt, item, "peer", "verified") is True
        assert len(receiver.received) == 1

        # Import (unknown = rejected by receiver logic)
        assert broker.import_knowledge(kt, item, "peer", "unknown") is False


# ─────────────────────────────────────────────────────────────────────────────
# MotorProgramProvider
# ─────────────────────────────────────────────────────────────────────────────


def _make_registry_with_programs():
    """Create a ProgramRegistry with test programs."""
    from maxim.embodiment.motor import MotorProgram, MotorStep, ProgramRegistry

    reg = ProgramRegistry()
    # Good program — high success, enough executions
    good = MotorProgram(
        name="reach_and_grasp",
        goal_signature="pick_up_object",
        steps=[
            MotorStep(entity_path="arm.hand", modulator="gripper", affordance="grasp", params={"force": 0.5}),
        ],
        confidence=0.8,
        total_executions=10,
        total_successes=9,
        source="discovered",
    )
    # Bad program — low success rate
    bad = MotorProgram(
        name="flail_wildly",
        goal_signature="clear_table",
        steps=[
            MotorStep(entity_path="arm.elbow", modulator="joint", affordance="rotate", params={"deg": 360}),
        ],
        confidence=0.3,
        total_executions=10,
        total_successes=2,
        source="discovered",
    )
    # New program — not enough executions
    new = MotorProgram(
        name="poke_carefully",
        goal_signature="test_object",
        steps=[
            MotorStep(entity_path="arm.finger", modulator="tip", affordance="press", params={"force": 0.1}),
        ],
        confidence=0.5,
        total_executions=1,
        total_successes=1,
        source="discovered",
    )
    reg.register(good)
    reg.register(bad)
    reg.register(new)
    return reg


class TestMotorProgramProvider:
    def test_knowledge_type(self):
        reg = MagicMock()
        reg.export_state.return_value = {"programs": {}}
        p = MotorProgramProvider(reg)
        assert p.knowledge_type() == "motor_program"

    def test_filters_by_execution_and_success(self):
        reg = _make_registry_with_programs()
        p = MotorProgramProvider(reg)
        shareable = p.get_shareable()

        names = {item["name"] for item in shareable}
        assert "reach_and_grasp" in names  # Good: 10 execs, 90% success
        assert "flail_wildly" not in names  # Bad: 20% success rate
        assert "poke_carefully" not in names  # New: only 1 execution

    def test_filters_by_goal(self):
        reg = _make_registry_with_programs()
        p = MotorProgramProvider(reg)
        shareable = p.get_shareable(goal="pick_up_object")
        assert len(shareable) == 1
        assert shareable[0]["goal_signature"] == "pick_up_object"

    def test_respects_limit(self):
        from maxim.embodiment.motor import MotorProgram, MotorStep, ProgramRegistry

        reg = ProgramRegistry()
        for i in range(10):
            prog = MotorProgram(
                name=f"prog_{i}",
                goal_signature=f"goal_{i}",
                steps=[MotorStep(entity_path="arm", modulator="m", affordance="a")],
                total_executions=5,
                total_successes=5,
            )
            reg.register(prog)
        p = MotorProgramProvider(reg)
        assert len(p.get_shareable(limit=3)) == 3

    def test_empty_registry(self):
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        p = MotorProgramProvider(reg)
        assert p.get_shareable() == []

    def test_protocol_conformance(self):
        reg = MagicMock()
        reg.export_state.return_value = {"programs": {}}
        assert isinstance(MotorProgramProvider(reg), KnowledgeProvider)


# ─────────────────────────────────────────────────────────────────────────────
# MotorProgramReceiver
# ─────────────────────────────────────────────────────────────────────────────


def _make_program_data(name="imported_reach", entity_path="arm.hand"):
    """Create a serialized MotorProgram dict."""
    from maxim.embodiment.motor import MotorProgram, MotorStep

    prog = MotorProgram(
        name=name,
        goal_signature="pick_up",
        steps=[
            MotorStep(entity_path=entity_path, modulator="gripper", affordance="grasp", params={"force": 0.3}),
        ],
        confidence=0.9,
        total_executions=20,
        total_successes=18,
        source="discovered",
    )
    return prog.to_dict()


class TestMotorProgramReceiver:
    def test_knowledge_type(self):
        from maxim.embodiment.motor import ProgramRegistry

        r = MotorProgramReceiver(ProgramRegistry())
        assert r.knowledge_type() == "motor_program"

    def test_rejects_unknown_trust(self):
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        r = MotorProgramReceiver(reg)
        data = _make_program_data()
        assert r.import_knowledge(data, "sketchy-peer", "unknown") is False
        assert reg.get("imported_reach") is None

    def test_rejects_remote_trust(self):
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        r = MotorProgramReceiver(reg)
        data = _make_program_data()
        assert r.import_knowledge(data, "remote-peer", "remote") is False

    def test_accepts_verified_trust(self):
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        r = MotorProgramReceiver(reg)  # No embodiment = skip entity validation
        data = _make_program_data()
        assert r.import_knowledge(data, "trusted-peer", "verified") is True
        assert reg.get("imported_reach") is not None

    def test_accepts_discovered_trust(self):
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        r = MotorProgramReceiver(reg)
        data = _make_program_data()
        assert r.import_knowledge(data, "lan-peer", "discovered") is True

    def test_applies_confidence_discount(self):
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        r = MotorProgramReceiver(reg)
        data = _make_program_data()  # confidence=0.9
        r.import_knowledge(data, "peer", "verified")
        prog = reg.get("imported_reach")
        assert prog is not None
        assert abs(prog.confidence - 0.45) < 0.01  # 0.9 * 0.5

    def test_resets_execution_stats(self):
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        r = MotorProgramReceiver(reg)
        data = _make_program_data()  # 20 executions, 18 successes
        r.import_knowledge(data, "peer", "verified")
        prog = reg.get("imported_reach")
        assert prog is not None
        assert prog.total_executions == 0
        assert prog.total_successes == 0

    def test_tags_source(self):
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        r = MotorProgramReceiver(reg)
        data = _make_program_data()
        r.import_knowledge(data, "reachy-kitchen", "verified")
        prog = reg.get("imported_reach")
        assert prog is not None
        assert prog.source == "peer:reachy-kitchen"

    def test_deduplicates(self):
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        r = MotorProgramReceiver(reg)
        data = _make_program_data()
        assert r.import_knowledge(data, "peer", "verified") is True
        assert r.import_knowledge(data, "peer", "verified") is False  # Already exists

    def test_rejects_unknown_entity_path(self):
        """Programs referencing hardware the local robot doesn't have are rejected."""
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        # Mock embodiment that only has "arm.hand"
        embodiment = MagicMock()
        embodiment.find.return_value = None  # Entity not found

        r = MotorProgramReceiver(reg, embodiment=embodiment)
        data = _make_program_data(entity_path="wing.feather")  # Not in local body
        assert r.import_knowledge(data, "peer", "verified") is False

    def test_accepts_matching_entity_path(self):
        """Programs where all entity paths exist locally are accepted."""
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        embodiment = MagicMock()
        embodiment.find.return_value = MagicMock()  # Entity found

        r = MotorProgramReceiver(reg, embodiment=embodiment)
        data = _make_program_data(entity_path="arm.hand")
        assert r.import_knowledge(data, "peer", "verified") is True
        embodiment.find.assert_called_with("arm.hand")

    def test_no_embodiment_skips_entity_check(self):
        """Without embodiment, entity validation is skipped (non-embodied agent)."""
        from maxim.embodiment.motor import ProgramRegistry

        reg = ProgramRegistry()
        r = MotorProgramReceiver(reg, embodiment=None)
        data = _make_program_data(entity_path="anything.goes")
        assert r.import_knowledge(data, "peer", "verified") is True

    def test_protocol_conformance(self):
        from maxim.embodiment.motor import ProgramRegistry

        assert isinstance(MotorProgramReceiver(ProgramRegistry()), KnowledgeReceiver)

    def test_broker_integration(self):
        """Full flow through broker with motor programs."""
        source_reg = _make_registry_with_programs()
        broker = ExperienceBroker()
        broker.register_provider(MotorProgramProvider(source_reg))

        from maxim.embodiment.motor import ProgramRegistry

        target_reg = ProgramRegistry()
        broker.register_receiver(MotorProgramReceiver(target_reg))

        shareable = broker.get_shareable(knowledge_type="motor_program")
        assert len(shareable) >= 1  # At least reach_and_grasp

        for item in shareable:
            kt = item.pop("knowledge_type")
            broker.import_knowledge(kt, item, "peer-robot", "verified")

        assert target_reg.get("reach_and_grasp") is not None
        prog = target_reg.get("reach_and_grasp")
        assert prog.source == "peer:peer-robot"
        assert prog.total_executions == 0  # Reset
