"""Tests for Phase 4 — Percept factories + AgentPool runtime unification."""

from __future__ import annotations

from maxim.agents.bus import Percept
from maxim.agents.percept_factory import (
    make_intero_percept,
    make_scene_percept,
    make_text_percept,
)


class TestMakeTextPercept:
    def test_basic(self):
        p = make_text_percept("hello world")
        assert isinstance(p, Percept)
        assert p.content == "hello world"
        assert p.modality == "text"
        assert p.source == "text"
        assert p.context is not None

    def test_agent_id_propagates(self):
        p = make_text_percept("hi", agent_id="agent_a")
        assert p.context.agent_id == "agent_a"

    def test_channel_and_sender(self):
        p = make_text_percept("msg", channel="sms", sender="alice")
        assert p.context.channel == "sms"
        assert p.context.sender == "alice"

    def test_metadata_passthrough(self):
        p = make_text_percept("x", metadata={"key": "val"})
        assert p.metadata == {"key": "val"}

    def test_custom_source(self):
        p = make_text_percept("x", source="npc_turn")
        assert p.source == "npc_turn"

    def test_round_trip(self):
        p = make_text_percept("hello", agent_id="a1", channel="sms", sender="bob")
        d = p.to_dict()
        r = Percept.from_dict(d)
        assert r.content == "hello"
        assert r.modality == "text"
        assert r.context.agent_id == "a1"
        assert r.context.channel == "sms"


class TestMakeScenePercept:
    def test_basic(self):
        p = make_scene_percept("You enter the tavern.")
        assert p.modality == "text"
        assert p.source == "narrative"
        assert p.context.channel == "narrative"
        assert p.content == "You enter the tavern."

    def test_agent_id(self):
        p = make_scene_percept("scene", agent_id="npc_guard")
        assert p.context.agent_id == "npc_guard"


class TestMakeInteroPercept:
    def test_basic(self):
        p = make_intero_percept("battery: 3.7V")
        assert p.modality == "intero"
        assert p.source == "embodiment"
        assert p.context.channel == "internal"

    def test_agent_id(self):
        p = make_intero_percept("temp: 42C", agent_id="bot_1")
        assert p.context.agent_id == "bot_1"


class TestAgentPoolUnification:
    def test_run_turn_produces_typed_percept(self):
        from maxim.runtime.agent_factory import AgentConfig, AgentFactory
        from maxim.runtime.agent_pool import AgentPool

        factory = AgentFactory(base_data_dir="/tmp/test_phase4_agents")
        pool = AgentPool()
        instance = factory.create_agent(AgentConfig(agent_id="npc_test", personality="friendly"))
        pool.add(instance)

        result = pool.run_turn("npc_test", "You see a merchant.")
        assert result.percept is not None
        assert isinstance(result.percept, Percept)
        assert result.percept.content == "You see a merchant."
        assert result.percept.modality == "text"
        assert result.percept.context.agent_id == "npc_test"
        assert result.percept.context.channel == "narrative"

        pool.shutdown()

    def test_run_turn_still_stores_observation(self):
        from maxim.runtime.agent_factory import AgentConfig, AgentFactory
        from maxim.runtime.agent_pool import AgentPool

        factory = AgentFactory(base_data_dir="/tmp/test_phase4_agents_obs")
        pool = AgentPool()
        instance = factory.create_agent(AgentConfig(agent_id="npc_obs", personality="stern", remembers=True))
        pool.add(instance)

        pool.run_turn("npc_obs", "A dragon attacks.")
        assert len(instance.hippocampus) >= 1

        pool.shutdown()
