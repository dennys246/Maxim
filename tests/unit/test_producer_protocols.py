"""Tests for Phase 3 — PerceptProducer/ReactionProducer protocols + SEM integration."""

from __future__ import annotations

from unittest.mock import MagicMock

from maxim.reactions.protocols import PerceptProducer


class TestPerceptProducerProtocol:
    def test_embodiment_percept_source_satisfies_protocol(self):
        from maxim.embodiment.percepts import EmbodimentPerceptSource

        assert isinstance(EmbodimentPerceptSource, type)
        mock_embodiment = MagicMock()
        mock_embodiment.root.name = "test_entity"
        source = EmbodimentPerceptSource(mock_embodiment, agent_id="agent_a")
        assert isinstance(source, PerceptProducer)
        assert source.name == "embodiment:test_entity"

    def test_conversational_source_satisfies_protocol(self):
        from maxim.simulation.conversational_source import ConversationalSource

        source = ConversationalSource()
        assert hasattr(source, "next_percept")
        assert hasattr(source, "name")

    def test_scenario_source_satisfies_protocol(self):
        from maxim.simulation.scenario_source import ScenarioSource

        assert hasattr(ScenarioSource, "next_percept")


class TestEmbodimentPerceptSourceContext:
    def test_percept_carries_agent_id_when_set(self):
        from maxim.embodiment.percepts import EmbodimentPerceptSource

        mock_embodiment = MagicMock()
        mock_embodiment.root.name = "test_bot"
        mock_embodiment.evaluate_failures.return_value = []
        mock_embodiment.format_body_state_for_prompt.return_value = "battery: 3.7V"
        mock_embodiment.tick_vital_drift = MagicMock()

        source = EmbodimentPerceptSource(mock_embodiment, poll_hz=1000.0, agent_id="bot_1")
        percept = source.next_percept()

        assert percept is not None
        assert percept.modality == "intero"
        assert percept.context is not None
        assert percept.context.agent_id == "bot_1"

    def test_percept_context_is_none_without_agent_id(self):
        from maxim.embodiment.percepts import EmbodimentPerceptSource

        mock_embodiment = MagicMock()
        mock_embodiment.root.name = "test_bot"
        mock_embodiment.evaluate_failures.return_value = []
        mock_embodiment.format_body_state_for_prompt.return_value = "battery: 3.7V"
        mock_embodiment.tick_vital_drift = MagicMock()

        source = EmbodimentPerceptSource(mock_embodiment, poll_hz=1000.0)
        percept = source.next_percept()

        assert percept is not None
        assert percept.context is None


class TestCerebellumModulatorReactionEmission:
    def _make_modulator(self, reaction_bus=None):
        from maxim.embodiment.backends.cerebellum_modulator import CerebellumModulator

        entity = MagicMock()
        entity.name = "shock_baton"
        entity.sensors = {}
        cerebellum = MagicMock()
        cerebellum.predict.return_value = None
        cerebellum.record_llm_fallback = MagicMock()

        return CerebellumModulator(
            entity=entity,
            modulator_name="striker",
            affordances={"shock_strike": MagicMock()},
            cerebellum=cerebellum,
            reaction_bus=reaction_bus,
        )

    def test_emits_pain_on_unknown_affordance(self):
        from maxim.reactions.bus import ReactionBus

        bus = ReactionBus()
        received = []
        bus.subscribe("pain", received.append)

        mod = self._make_modulator(reaction_bus=bus)
        result = mod.execute("nonexistent", {})

        assert result.success is False
        assert len(received) == 1
        assert received[0].kind == "pain"
        assert "cerebellum:shock_baton.striker.nonexistent" == received[0].source

    def test_no_emission_without_bus(self):
        mod = self._make_modulator(reaction_bus=None)
        result = mod.execute("nonexistent", {})
        assert result.success is False

    def test_emits_pain_on_fallback_failure(self):
        from maxim.reactions.bus import ReactionBus

        bus = ReactionBus()
        received = []
        bus.subscribe("pain", received.append)

        mod = self._make_modulator(reaction_bus=bus)
        mock_fallback = MagicMock()
        from maxim.embodiment.sem import ModulatorResult

        mock_fallback.execute.return_value = ModulatorResult(
            success=False,
            modulator_name="striker",
            entity_name="shock_baton",
            affordance="shock_strike",
            params={},
            error="overheated",
        )
        mod._fallback = mock_fallback

        result = mod.execute("shock_strike", {})
        assert result.success is False
        assert len(received) >= 1

    def test_no_emission_on_success(self):
        from maxim.reactions.bus import ReactionBus

        bus = ReactionBus()
        received = []
        bus.subscribe("pain", received.append)

        mod = self._make_modulator(reaction_bus=bus)
        mock_fallback = MagicMock()
        from maxim.embodiment.sem import ModulatorResult

        mock_fallback.execute.return_value = ModulatorResult(
            success=True,
            modulator_name="striker",
            entity_name="shock_baton",
            affordance="shock_strike",
            params={},
        )
        mod._fallback = mock_fallback

        result = mod.execute("shock_strike", {})
        assert result.success is True
        assert len(received) == 0
