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

    def test_percept_context_has_no_agent_id_when_unset(self):
        from maxim.embodiment.percepts import EmbodimentPerceptSource

        mock_embodiment = MagicMock()
        mock_embodiment.root.name = "test_bot"
        mock_embodiment.evaluate_failures.return_value = []
        mock_embodiment.format_body_state_for_prompt.return_value = "battery: 3.7V"
        mock_embodiment.tick_vital_drift = MagicMock()

        source = EmbodimentPerceptSource(mock_embodiment, poll_hz=1000.0)
        percept = source.next_percept()

        assert percept is not None
        # F0.6 factory migration: make_intero_percept always creates a
        # context (channel="internal"), but agent_id stays None.
        assert percept.context is not None
        assert percept.context.agent_id is None


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


class TestCerebellumModulatorFactoryWiring:
    """Regression guard for reaction_bus_unification.md Gap A.

    Pre-audit: ``cerebellum_modulator_factory`` did NOT accept
    ``reaction_bus=``. Every CerebellumModulator created by the factory
    had ``self._reaction_bus = None`` and silently dropped all modulator
    failure reactions via ``_emit_failure_reaction``'s early return.

    Post-fix: the factory accepts ``reaction_bus=`` and forwards it to
    each CerebellumModulator it creates. When wired, failure reactions
    flow to the bus.
    """

    def test_factory_forwards_reaction_bus(self):
        """Factory-created modulators emit failure reactions when bus is wired."""
        from maxim.embodiment.backends.cerebellum_modulator import cerebellum_modulator_factory
        from maxim.reactions.bus import ReactionBus

        bus = ReactionBus()
        received = []
        bus.subscribe("pain", received.append)

        cerebellum = MagicMock()
        cerebellum.predict.return_value = None
        cerebellum.record_llm_fallback = MagicMock()

        factory = cerebellum_modulator_factory(
            cerebellum=cerebellum,
            reaction_bus=bus,
        )

        entity = MagicMock()
        entity.name = "test_weapon"
        entity.sensors = {}
        spec_mod = MagicMock()
        spec_mod.affordances = {"strike": MagicMock()}

        mod = factory(entity, "striker", spec_mod)

        # Execute an unknown affordance — should emit a failure reaction.
        result = mod.execute("nonexistent", {})
        assert result.success is False
        assert len(received) == 1
        assert received[0].kind == "pain"
        assert "cerebellum:test_weapon.striker.nonexistent" == received[0].source

    def test_factory_without_reaction_bus_silently_drops(self):
        """Factory without reaction_bus= still works but drops reactions."""
        from maxim.embodiment.backends.cerebellum_modulator import cerebellum_modulator_factory

        cerebellum = MagicMock()
        cerebellum.predict.return_value = None

        factory = cerebellum_modulator_factory(cerebellum=cerebellum)

        entity = MagicMock()
        entity.name = "test_weapon"
        entity.sensors = {}
        spec_mod = MagicMock()
        spec_mod.affordances = {"strike": MagicMock()}

        mod = factory(entity, "striker", spec_mod)
        # Should not raise, just silently drop.
        result = mod.execute("nonexistent", {})
        assert result.success is False
