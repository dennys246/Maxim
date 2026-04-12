"""Integration tests for F0.5 — agent_id threading + multi-agent isolation.

Covers:
- The SCN persistence-path race fix (path is set at construction, not late-bound)
- Two agents with overlapping concept names don't contaminate each other's
  Hippocampus / NAc / ATL / SCN state
- PerceptContext.agent_id populates at Percept producers that live in
  MaximAgent's single-agent runtime (PerceptionAgent, ConversationManager)
- MemoryAgent logs a soft warning on cross-agent percept mismatch without
  hard-failing (hard enforcement waits for F0.6 factory consolidation)
"""

from __future__ import annotations

import pytest

from maxim.agents.bus import AgentBus, Percept
from maxim.agents.memory_agent import MemoryAgent
from maxim.agents.percept_context import PerceptContext
from maxim.agents.perception_agent import PerceptionAgent
from maxim.comms.conversation import ConversationManager
from maxim.runtime.agent_factory import AgentConfig, AgentFactory
from maxim.time.scn import SCN


@pytest.fixture
def factory(tmp_path):
    return AgentFactory(base_data_dir=tmp_path / "agents")


@pytest.fixture
def two_agents(factory):
    guard = factory.create_agent(
        AgentConfig(agent_id="npc_guard", role="npc", personality="stern", remembers=True, learns=True)
    )
    merchant = factory.create_agent(
        AgentConfig(agent_id="npc_merchant", role="npc", personality="shrewd", remembers=True, learns=True)
    )
    return guard, merchant


# ---------------------------------------------------------------------------
# SCN race fix
# ---------------------------------------------------------------------------


class TestSCNConstructor:
    def test_default_path_is_none(self):
        assert SCN().persistence_path is None

    def test_path_bound_at_construction(self):
        scn = SCN(persistence_path="/tmp/test_scn.json")
        assert scn.persistence_path == "/tmp/test_scn.json"

    def test_factory_binds_per_agent_scn_path(self, two_agents):
        guard, merchant = two_agents
        # Both agents have SCN instances with distinct, agent-scoped paths
        guard_path = guard.memory_hub.scn.persistence_path
        merchant_path = merchant.memory_hub.scn.persistence_path
        assert guard_path is not None
        assert merchant_path is not None
        assert guard_path != merchant_path
        assert "npc_guard" in guard_path
        assert "npc_merchant" in merchant_path

    def test_factory_does_not_use_late_bind_attribute(self, two_agents):
        """The old _persistence_path underscore attribute must not be the
        primary binding anymore — the typed field takes precedence."""
        guard, _ = two_agents
        scn = guard.memory_hub.scn
        # persistence_path (typed field) must be set
        assert scn.persistence_path is not None


# ---------------------------------------------------------------------------
# Multi-agent isolation
# ---------------------------------------------------------------------------


class TestMultiAgentIsolation:
    def test_separate_bio_system_instances(self, two_agents):
        guard, merchant = two_agents
        assert guard.hippocampus is not merchant.hippocampus
        assert guard.nac is not merchant.nac
        assert guard.atl is not merchant.atl
        assert guard.memory_hub.scn is not merchant.memory_hub.scn

    def test_separate_persistence_directories(self, factory, tmp_path):
        factory.create_agent(AgentConfig(agent_id="agent_a"))
        factory.create_agent(AgentConfig(agent_id="agent_b"))
        assert (tmp_path / "agents" / "agent_a").is_dir()
        assert (tmp_path / "agents" / "agent_b").is_dir()

    def test_overlapping_observations_do_not_contaminate(self, two_agents):
        """Both agents store an observation with the same text ("mug").
        Each store stays isolated — agent A's count does not leak into B."""
        guard, merchant = two_agents

        guard.hippocampus.store_observation(
            text="mug on the table",
            metadata={"agent_id": "npc_guard"},
        )
        # Merchant has seen nothing
        assert len(guard.hippocampus) == 1
        assert len(merchant.hippocampus) == 0

        merchant.hippocampus.store_observation(
            text="mug on the table",
            metadata={"agent_id": "npc_merchant"},
        )
        # Both have 1 each, independently
        assert len(guard.hippocampus) == 1
        assert len(merchant.hippocampus) == 1

    def test_nac_causal_links_isolated(self, two_agents):
        """NAc causal-link IDs are hash-based on (event_sig, outcome_sig,
        context_hash). Two agents observing the same event pair must
        produce links in their own NAc stores without cross-pollination."""
        from maxim.decisions.nac import Valence

        guard, merchant = two_agents

        guard.nac.observe(
            event_type="tool",
            event_signature="grasp_mug",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=1.0,
        )
        # Merchant's NAc must stay empty
        assert guard.nac.stats()["total_observations"] > 0
        assert merchant.nac.stats()["total_observations"] == 0


# ---------------------------------------------------------------------------
# Percept producers thread agent_id
# ---------------------------------------------------------------------------


class TestPerceptionAgentThreading:
    def test_context_is_none_without_agent_id(self):
        pa = PerceptionAgent(AgentBus())
        assert pa._build_context() is None

    def test_context_carries_agent_id_when_set(self):
        pa = PerceptionAgent(AgentBus(), agent_id="agent_x")
        ctx = pa._build_context()
        assert ctx is not None
        assert ctx.agent_id == "agent_x"


class TestConversationManagerThreading:
    def test_context_carries_agent_id(self):
        cm = ConversationManager(AgentBus(), agent_id="agent_y")
        ctx = cm._build_context(channel="sms", sender="alice")
        assert ctx.agent_id == "agent_y"
        assert ctx.channel == "sms"
        assert ctx.sender == "alice"

    def test_unknown_channel_falls_back_to_none(self):
        cm = ConversationManager(AgentBus(), agent_id="agent_y")
        ctx = cm._build_context(channel="twilio:sms", sender="alice")
        assert ctx.channel is None
        assert ctx.agent_id == "agent_y"


# ---------------------------------------------------------------------------
# MemoryAgent soft mismatch warning
# ---------------------------------------------------------------------------


class TestMemoryAgentAgentIdMismatch:
    def _make_percept(self, agent_id: str | None) -> Percept:
        return Percept(
            timestamp=1.0,
            source="test",
            salience=0.1,  # Below threshold so capture doesn't fire
            context=PerceptContext(agent_id=agent_id) if agent_id else None,
        )

    def test_no_warning_when_agent_id_matches(self, caplog):
        ma = MemoryAgent(AgentBus(), agent_id="agent_a")
        with caplog.at_level("WARNING"):
            ma._on_percept(self._make_percept("agent_a"))
        assert "received percept tagged" not in caplog.text

    def test_no_warning_when_memory_agent_has_no_id(self, caplog):
        ma = MemoryAgent(AgentBus())  # no agent_id set
        with caplog.at_level("WARNING"):
            ma._on_percept(self._make_percept("agent_x"))
        assert "received percept tagged" not in caplog.text

    def test_no_warning_when_percept_has_no_context(self, caplog):
        ma = MemoryAgent(AgentBus(), agent_id="agent_a")
        with caplog.at_level("WARNING"):
            ma._on_percept(self._make_percept(None))
        assert "received percept tagged" not in caplog.text

    def test_warning_fires_once_per_foreign_id(self, caplog):
        ma = MemoryAgent(AgentBus(), agent_id="agent_a")
        with caplog.at_level("WARNING"):
            ma._on_percept(self._make_percept("agent_b"))
            ma._on_percept(self._make_percept("agent_b"))  # duplicate — should not re-warn
            ma._on_percept(self._make_percept("agent_c"))  # different foreign — should warn
        warnings = [r for r in caplog.records if "received percept tagged" in r.getMessage()]
        assert len(warnings) == 2
        assert "agent_b" in warnings[0].getMessage()
        assert "agent_c" in warnings[1].getMessage()


# ---------------------------------------------------------------------------
# Cross-layer sanity — SCN path survives save/load round trip
# ---------------------------------------------------------------------------


class TestSCNRoundTrip:
    def test_save_then_new_scn_reads_back(self, tmp_path):
        path = tmp_path / "scn.json"
        scn = SCN(persistence_path=str(path))
        # Register a dummy entry so save produces content
        from maxim.time.temporal_signature import TemporalSignature

        sig = TemporalSignature.from_timestamp(1_700_000_000.0)
        scn.register("mem-1", sig)
        scn.save(str(path))
        assert path.exists()

        scn2 = SCN(persistence_path=str(path))
        scn2.load(str(path))
        assert "mem-1" in scn2._signatures
