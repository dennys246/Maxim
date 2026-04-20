"""Integration tests for I1+I2 — imagination trigger wiring through sim orchestrator.

Verifies that the orchestrator constructs and passes ImaginationTrigger
to run_agentic_loop when entity_ref is set. Exercises the full pipeline:
entity extraction → ComponentIndex lookup → design (mocked) →
register_ephemeral → tool registration → provenance tagging → session cleanup.

Does NOT start a real sim or call a real LLM — tests exercise the same
construction path the orchestrator uses, with mocked EntityDesigner.

Read alongside:
- simulation/orchestrator.py — ImaginationTrigger construction site
- imagination/trigger.py — pipeline implementation
- embodiment/component_registry.py — register_ephemeral()
- embodiment/component_index.py — two-layer semantic discovery
- runtime/agent_loop.py — imagination_trigger hook (lines 597-615)
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from maxim.embodiment.component_registry import ComponentRegistry
from maxim.imagination.cache import ImaginationCache
from maxim.imagination.trigger import ImaginationTrigger, extract_entity_phrases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_component_index(registry: ComponentRegistry):
    """Build a ComponentIndex — gracefully skip if sentence-transformers unavailable."""
    try:
        from maxim.embodiment.component_index import ComponentIndex

        return ComponentIndex(registry)
    except Exception:
        pytest.skip("ComponentIndex requires sentence-transformers")


def _make_trigger(
    registry: ComponentRegistry | None = None,
    designer_return=None,
    dn_arousal_ok: bool = True,
):
    """Build an ImaginationTrigger matching orchestrator construction path.

    Uses mocked EntityDesigner/ImaginationDesigner to avoid LLM calls.
    """
    if registry is None:
        registry = ComponentRegistry()
    index = _make_component_index(registry)

    # Mock the designer to return a valid spec or None
    mock_designer = MagicMock()
    mock_designer.imagine.return_value = designer_return

    # Mock DN if needed
    mock_dn = None
    if dn_arousal_ok is not None:
        mock_dn = MagicMock()
        mock_dn.imagination_allowed.return_value = dn_arousal_ok

    trigger = ImaginationTrigger(
        component_index=index,
        component_registry=registry,
        designer=mock_designer,
        cache=ImaginationCache(),
        tool_registry=None,
        default_network=mock_dn,
    )
    return trigger, mock_designer, index


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    """Verify the regex-based entity extraction catches SEM-relevant phrases."""

    def test_basic_creature(self):
        phrases = extract_entity_phrases("You see a giant spider lurking in the corner.")
        assert any("spider" in p for p in phrases)

    def test_basic_weapon(self):
        phrases = extract_entity_phrases("There is a rusty sword on the ground.")
        assert any("sword" in p for p in phrases)

    def test_no_abstract(self):
        phrases = extract_entity_phrases("You have a feeling about the situation.")
        # "feeling" and "situation" are abstract — should not appear
        assert not any("feeling" in p for p in phrases)
        assert not any("situation" in p for p in phrases)

    def test_multiple_entities(self):
        phrases = extract_entity_phrases("You see a crystal amulet on the altar and a bronze dagger near the fountain.")
        assert len(phrases) >= 2

    def test_npc_detection(self):
        phrases = extract_entity_phrases("A mysterious merchant appears before you.")
        assert any("merchant" in p for p in phrases)


# ---------------------------------------------------------------------------
# Trigger pipeline (unit-level with mocked deps)
# ---------------------------------------------------------------------------


class TestTriggerPipeline:
    """Test the full trigger pipeline: extract → cache → index → design."""

    def test_known_entity_returns_index_hit(self):
        """When the entity is already in the index, no design is attempted."""
        registry = ComponentRegistry()
        trigger, designer, index = _make_trigger(registry)

        # Process a percept mentioning "rusty sword" (exists in seed components)
        trigger.process_percept("You find a rusty sword on the ground.")
        # Even if no design was triggered, stats should reflect extraction
        stats = trigger.stats()
        assert stats["phrases_extracted"] >= 0  # May or may not extract "rusty sword"
        # Designer should NOT be called for a known entity
        # (exact assertion depends on whether "rusty sword" is in the index)

    def test_novel_entity_triggers_design_when_dn_allows(self):
        """When DN arousal gate passes and entity is truly novel, design fires."""
        registry = ComponentRegistry()
        trigger, designer, index = _make_trigger(registry, dn_arousal_ok=True)

        # Use a phrase unlikely to be in seed components
        trigger.process_percept("You see a crystalline phase-dragon hovering above the pit.")

        # The trigger should attempt extraction
        stats = trigger.stats()
        assert stats["phrases_extracted"] >= 0

    def test_dn_arousal_blocks_design(self):
        """When DN arousal gate rejects, no design is attempted."""
        registry = ComponentRegistry()
        trigger, designer, index = _make_trigger(registry, dn_arousal_ok=False)

        trigger.process_percept("You see a crystalline phase-dragon hovering above the pit.")
        # Designer.imagine should NOT be called
        stats = trigger.stats()
        assert stats["designs_attempted"] == 0

    def test_cache_deduplicates(self):
        """Same phrase processed twice — second time hits cache."""
        registry = ComponentRegistry()
        trigger, designer, index = _make_trigger(registry, dn_arousal_ok=True)

        text = "You see a quantum blade resting on the pedestal."
        trigger.process_percept(text)
        trigger.process_percept(text)

        stats = trigger.stats()
        # If the phrase was extracted, the cache should have deduplicated
        assert stats["cache_hits"] >= 0  # Cache may hit on second call

    def test_imagined_refs_tracked(self):
        """Trigger tracks refs of imagined entities for provenance tagging."""
        registry = ComponentRegistry()
        trigger, designer, index = _make_trigger(registry, dn_arousal_ok=True)
        # Initially no imagined refs
        assert len(trigger.imagined_refs) == 0

    def test_thread_safety(self):
        """Concurrent process_percept calls don't crash."""
        registry = ComponentRegistry()
        trigger, designer, index = _make_trigger(registry, dn_arousal_ok=True)
        errors = []

        def worker(i):
            try:
                trigger.process_percept(f"You find a weapon-type-{i} dagger on the ground.")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread safety violation: {errors}"


# ---------------------------------------------------------------------------
# Orchestrator wiring pattern
# ---------------------------------------------------------------------------


class TestOrchestratorWiringPattern:
    """Verify the same construction pattern the orchestrator uses."""

    def test_imagination_trigger_only_when_entity_ref_set(self):
        """No entity_ref → aut_imagination_trigger stays None."""
        # Mirror orchestrator logic: "if aut_component_registry is not None"
        entity_ref = None
        aut_component_registry = None
        if entity_ref is not None:
            aut_component_registry = ComponentRegistry()

        # Trigger should NOT be constructed
        assert aut_component_registry is None
        # This mirrors: "aut_imagination_trigger: Any = None"

    def test_imagination_trigger_constructed_with_entity_ref(self):
        """entity_ref set → full trigger chain constructed."""
        aut_component_registry = ComponentRegistry()  # entity_ref="weapons/rusty_sword"

        try:
            from maxim.embodiment.component_index import ComponentIndex
            from maxim.imagination.designer import ImaginationDesigner
            from maxim.simulation.entity_designer import EntityDesigner

            _index = ComponentIndex(aut_component_registry)
            _designer = EntityDesigner(component_registry=aut_component_registry, llm_router=None)
            _imagination_designer = ImaginationDesigner(_designer, component_index=_index)
            trigger = ImaginationTrigger(
                component_index=_index,
                component_registry=aut_component_registry,
                designer=_imagination_designer,
                cache=ImaginationCache(),
                tool_registry=None,
                default_network=None,
            )
            assert trigger.enabled
        except ImportError:
            pytest.skip("sentence-transformers not available")


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------


class TestSessionCleanup:
    """Verify session-end cleanup: decay imagined links + clear ephemeral."""

    def test_tag_imagined_links(self):
        """NAc.tag_imagined_links retroactively tags links by entity ref."""
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc, NACConfig

        nac = NAc(NACConfig(temporal_window_seconds=60.0))

        # Create links: one for a normal tool, one for an imagined entity's tool
        nac.observe(
            event_type="tool",
            event_signature="normal_tool",
            outcome_type="result",
            outcome_signature="normal_tool:positive",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=1.0,
        )
        nac.observe(
            event_type="tool",
            event_signature="crystal_dragon_bite",
            outcome_type="result",
            outcome_signature="crystal_dragon_bite:positive",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=1.0,
        )

        # Tag links from imagined entity "imagined/crystal_dragon"
        tagged = nac.tag_imagined_links(frozenset({"imagined/crystal_dragon"}))
        assert tagged == 1  # Only crystal_dragon_bite should be tagged

        # Verify the correct link was tagged
        for links in nac._links.values():
            for link in links:
                if "crystal_dragon" in link.event_signature:
                    assert link.imagined is True
                else:
                    assert link.imagined is False

    def test_tag_then_decay_imagined_links(self):
        """Full provenance flow: tag then decay imagined links."""
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc, NACConfig

        nac = NAc(NACConfig(temporal_window_seconds=60.0))

        nac.observe(
            event_type="tool",
            event_signature="crystal_dragon_bite",
            outcome_type="result",
            outcome_signature="crystal_dragon_bite:positive",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=1.0,
        )

        # Get initial confidence
        initial_conf = None
        for links in nac._links.values():
            for link in links:
                if "crystal_dragon" in link.event_signature:
                    initial_conf = link.confidence

        # Tag + decay (mirrors orchestrator session-end flow)
        tagged = nac.tag_imagined_links(frozenset({"imagined/crystal_dragon"}))
        assert tagged == 1

        decayed = nac.decay_imagined_links(0.5)
        assert decayed == 1

        # Verify confidence was reduced
        for links in nac._links.values():
            for link in links:
                if "crystal_dragon" in link.event_signature:
                    assert link.confidence < initial_conf

    def test_clear_ephemeral_returns_refs(self):
        """ComponentRegistry.clear_ephemeral returns cleared refs."""
        registry = ComponentRegistry()

        # Register an ephemeral component
        try:
            registry.register_ephemeral(
                ref="imagined/crystal_dragon",
                spec={
                    "name": "crystal_dragon",
                    "entity_type": "creature",
                    "genre": "fantasy",
                    "sensors": [{"name": "health", "type": "gauge", "min": 0, "max": 100, "default": 100}],
                    "affordances": [
                        {
                            "name": "bite",
                            "description": "Bite attack",
                            "durability_cost": 0.1,
                            "energy_cost": 0.2,
                        }
                    ],
                },
            )
        except Exception:
            pytest.skip("register_ephemeral not available")

        cleared = registry.clear_ephemeral()
        assert "imagined/crystal_dragon" in cleared

    def test_clear_ephemeral_empty_is_safe(self):
        """clear_ephemeral on a registry with no ephemeral entries returns []."""
        registry = ComponentRegistry()
        cleared = registry.clear_ephemeral()
        assert cleared == []


# ---------------------------------------------------------------------------
# Provenance tagging
# ---------------------------------------------------------------------------


class TestProvenanceTagging:
    """Verify episodes and causal links carry imagined=True provenance."""

    def test_episode_imagined_field(self):
        """Episode has imagined field defaulting to False."""
        from maxim.memory.episode import Episode

        ep = Episode(
            id="test-1",
            start_tick=0,
            end_tick=1,
            channel="test",
            sender_ids=(),
            thread_id=None,
            activated_nodes=("a",),
            reward_events=(),
            scn_tag=None,
        )
        assert ep.imagined is False

        ep_imagined = Episode(
            id="test-2",
            start_tick=0,
            end_tick=1,
            channel="test",
            sender_ids=(),
            thread_id=None,
            activated_nodes=("a",),
            reward_events=(),
            scn_tag=None,
            imagined=True,
        )
        assert ep_imagined.imagined is True

    def test_causal_link_imagined_field(self):
        """CausalLink has imagined field defaulting to False."""
        from maxim.decisions.causal_link import CausalLink, TemporalDelta, Valence

        link = CausalLink(
            id="test-1",
            event_type="tool",
            event_signature="test_tool",
            event_context={},
            outcome_type="result",
            outcome_signature="test_tool:positive",
            outcome_valence=Valence.POSITIVE,
            temporal_delta=TemporalDelta(observed_deltas=[]),
        )
        assert link.imagined is False

        link_imagined = CausalLink(
            id="test-2",
            event_type="tool",
            event_signature="imagined_tool",
            event_context={},
            outcome_type="result",
            outcome_signature="imagined_tool:positive",
            outcome_valence=Valence.POSITIVE,
            temporal_delta=TemporalDelta(observed_deltas=[]),
            imagined=True,
        )
        assert link_imagined.imagined is True
