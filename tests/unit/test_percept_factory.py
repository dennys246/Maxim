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


class TestSingleLoggingLayer:
    """percept_factory is the SINGLE percept-logging layer.

    ConversationalSource used to log again after building through the factory,
    so every conversational percept was emitted TWICE — in the terminal, the
    JSONL trail, and the console /ws stream (where it showed as duplicate
    `percept` events).
    """

    def test_conversational_source_does_not_relog(self):
        # Structural: the source builds via the factory, so a sim_percept call
        # here is by definition a duplicate.
        import inspect

        from maxim.simulation import conversational_source

        src = inspect.getsource(conversational_source)
        assert "make_text_percept" in src, "source no longer builds via the factory — revisit this guard"
        # Ignore comments/docstring prose — assert on actual CALL sites.
        code_lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
        calls = [ln for ln in code_lines if "sim_percept(" in ln]
        assert not calls, f"ConversationalSource must not log percepts (the factory already does): {calls}"

    def test_factory_logs_each_percept_exactly_once(self, monkeypatch):
        from maxim.agents import percept_factory as pf

        seen: list[tuple] = []
        monkeypatch.setattr(pf, "_log_percept", lambda *a, **k: seen.append(a))
        pf.make_text_percept("hello", source="cli")
        assert len(seen) == 1

    def test_enqueue_through_conversational_source_logs_once(self, monkeypatch):
        # Behavioural counterpart: drive the real path and count emissions.
        from maxim.agents import percept_factory as pf
        from maxim.simulation.conversational_source import ConversationalSource

        seen: list[tuple] = []
        monkeypatch.setattr(pf, "_log_percept", lambda *a, **k: seen.append(a))
        ConversationalSource().inject_cli("hello there")
        assert len(seen) == 1, f"expected exactly one percept log, got {len(seen)}"
