"""Unit tests for the imagination system (I1 + I2).

Tests entity extraction, ImaginationCache, ImaginationTrigger,
ImaginationDesigner, ComponentRegistry.register_ephemeral,
Episode/CausalLink imagined provenance, and DN arousal gate.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch


from maxim.imagination.cache import ImaginationCache, ImaginationResult
from maxim.imagination.trigger import ImaginationTrigger, extract_entity_phrases
from maxim.imagination.designer import ImaginationDesigner, DesignResult


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    """Tests for extract_entity_phrases."""

    def test_basic_extraction(self):
        text = "You see a rusty gate and a large wolf."
        phrases = extract_entity_phrases(text)
        assert any("gate" in p for p in phrases)
        assert any("wolf" in p for p in phrases)

    def test_intro_patterns(self):
        text = "There is a healing potion on the table."
        phrases = extract_entity_phrases(text)
        assert any("potion" in p for p in phrases)

    def test_notice_pattern(self):
        text = "You notice a silver dagger lying on the ground."
        phrases = extract_entity_phrases(text)
        assert any("dagger" in p for p in phrases)

    def test_appears_pattern(self):
        text = "A giant spider appears from the shadows."
        phrases = extract_entity_phrases(text)
        assert "giant spider" in phrases

    def test_filters_abstract_concepts(self):
        text = "The world is a place of things and ideas."
        phrases = extract_entity_phrases(text)
        assert len(phrases) == 0

    def test_filters_body_parts(self):
        text = "He raises his hand and clenches his fingers."
        phrases = extract_entity_phrases(text)
        assert len(phrases) == 0

    def test_filters_clothing(self):
        text = "She wears a dark cloak and tall boots."
        phrases = extract_entity_phrases(text)
        assert len(phrases) == 0

    def test_deduplication(self):
        """Same head noun mentioned multiple times → deduplicated."""
        text = "You see a wolf. The wolf growls. A wolf howls."
        phrases = extract_entity_phrases(text)
        wolf_phrases = [p for p in phrases if "wolf" in p]
        assert len(wolf_phrases) == 1

    def test_empty_text(self):
        assert extract_entity_phrases("") == []
        assert extract_entity_phrases("   ") == []
        assert extract_entity_phrases(None) == []  # type: ignore[arg-type]

    def test_multi_word_entity(self):
        text = "A rusty iron gate blocks the passage."
        phrases = extract_entity_phrases(text)
        assert any("gate" in p for p in phrases)

    def test_creature_detection(self):
        text = "A skeleton warrior stands before you."
        phrases = extract_entity_phrases(text)
        assert any("skeleton" in p for p in phrases)

    def test_weapon_detection(self):
        text = "You find a sharpened battle axe."
        phrases = extract_entity_phrases(text)
        assert any("axe" in p for p in phrases)

    def test_item_detection(self):
        text = "There sits a glowing crystal on the pedestal."
        phrases = extract_entity_phrases(text)
        assert any("crystal" in p for p in phrases)

    def test_environment_detection(self):
        text = "You discover a hidden portal behind the altar."
        phrases = extract_entity_phrases(text)
        assert any("portal" in p or "altar" in p for p in phrases)


# ---------------------------------------------------------------------------
# ImaginationCache
# ---------------------------------------------------------------------------


class TestImaginationCache:
    """Tests for ImaginationCache."""

    def test_put_and_get(self):
        cache = ImaginationCache()
        result = ImaginationResult(phrase="rusty gate", ref="environments/rusty_gate", imagined=False, score=0.85)
        cache.put(result)
        retrieved = cache.get("rusty gate")
        assert retrieved is not None
        assert retrieved.ref == "environments/rusty_gate"

    def test_normalization(self):
        cache = ImaginationCache()
        result = ImaginationResult(phrase="rusty gate", ref="environments/rusty_gate", imagined=False)
        cache.put(result)
        # Different casing and whitespace should still hit
        assert cache.get("Rusty  Gate") is not None
        assert cache.get("RUSTY GATE") is not None

    def test_cache_miss(self):
        cache = ImaginationCache()
        assert cache.get("nonexistent") is None

    def test_has(self):
        cache = ImaginationCache()
        assert not cache.has("wolf")
        cache.put(ImaginationResult(phrase="wolf", ref="creatures/wolf", imagined=False))
        assert cache.has("wolf")

    def test_mention_count(self):
        cache = ImaginationCache()
        assert cache.mention_count("wolf") == 0
        assert cache.record_mention("wolf") == 1
        assert cache.record_mention("wolf") == 2
        assert cache.mention_count("wolf") == 2

    def test_clear(self):
        cache = ImaginationCache()
        cache.put(ImaginationResult(phrase="wolf", ref="creatures/wolf", imagined=True))
        cache.record_mention("wolf")
        cache.clear()
        assert cache.size == 0
        assert not cache.has("wolf")
        assert cache.mention_count("wolf") == 0

    def test_imagined_refs(self):
        cache = ImaginationCache()
        cache.put(ImaginationResult(phrase="wolf", ref="creatures/wolf", imagined=False))
        cache.put(ImaginationResult(phrase="shadow spider", ref="creatures/shadow_spider", imagined=True))
        cache.put(ImaginationResult(phrase="fire sword", ref="weapons/fire_sword", imagined=True))
        refs = cache.imagined_refs()
        assert "creatures/shadow_spider" in refs
        assert "weapons/fire_sword" in refs
        assert "creatures/wolf" not in refs

    def test_thread_safety(self):
        cache = ImaginationCache()
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    cache.put(ImaginationResult(phrase=f"item_{n}_{i}", ref=f"items/item_{n}_{i}", imagined=True))
                    cache.record_mention(f"item_{n}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert cache.size == 800


# ---------------------------------------------------------------------------
# ComponentRegistry.register_ephemeral
# ---------------------------------------------------------------------------


class TestRegisterEphemeral:
    """Tests for ComponentRegistry ephemeral registration."""

    def _make_registry(self):
        """Create a minimal ComponentRegistry without scanning default paths."""
        from maxim.embodiment.component_registry import ComponentRegistry

        return ComponentRegistry(search_paths=[], include_defaults=False)

    def test_register_and_retrieve(self):
        reg = self._make_registry()
        spec = {
            "component": {"name": "shadow_spider", "category": "creatures"},
            "entity": {"name": "shadow_spider", "sensors": {}},
        }
        reg.register_ephemeral("creatures/shadow_spider", spec, provenance="imagined")
        assert reg.has("creatures/shadow_spider")
        retrieved = reg.get("creatures/shadow_spider")
        assert retrieved["entity"]["name"] == "shadow_spider"

    def test_get_info_checks_ephemeral(self):
        reg = self._make_registry()
        spec = {
            "component": {"name": "test_item", "category": "items"},
            "entity": {"name": "test_item"},
        }
        reg.register_ephemeral("items/test_item", spec)
        info = reg.get_info("items/test_item")
        assert info is not None
        assert info.ref == "items/test_item"
        assert info.source_path == "<imagined>"

    def test_clear_ephemeral(self):
        reg = self._make_registry()
        spec = {
            "component": {"name": "test", "category": "items"},
            "entity": {"name": "test"},
        }
        reg.register_ephemeral("items/test", spec)
        assert reg.has("items/test")
        cleared = reg.clear_ephemeral()
        assert "items/test" in cleared
        assert not reg.has("items/test")

    def test_is_ephemeral(self):
        reg = self._make_registry()
        spec = {
            "component": {"name": "test", "category": "items"},
            "entity": {"name": "test"},
        }
        reg.register_ephemeral("items/test", spec)
        assert reg.is_ephemeral("items/test")
        assert not reg.is_ephemeral("items/nonexistent")

    def test_ephemeral_wins_over_persistent(self):
        """Ephemeral overlay takes priority over persistent index."""
        reg = self._make_registry()
        persistent_spec = {
            "component": {"name": "gate", "category": "environments"},
            "entity": {"name": "gate", "description": "persistent"},
        }
        reg.register("environments/gate", persistent_spec)

        ephemeral_spec = {
            "component": {"name": "gate", "category": "environments"},
            "entity": {"name": "gate", "description": "ephemeral"},
        }
        reg.register_ephemeral("environments/gate", ephemeral_spec)

        retrieved = reg.get("environments/gate")
        assert retrieved["entity"]["description"] == "ephemeral"


# ---------------------------------------------------------------------------
# Episode + CausalLink imagined provenance
# ---------------------------------------------------------------------------


class TestImaginedProvenance:
    """Tests for imagined=True provenance on Episode and CausalLink."""

    def test_episode_imagined_default_false(self):
        from maxim.memory.episode import Episode

        ep = Episode(
            id="test",
            start_tick=0,
            end_tick=10,
            channel="sim",
            sender_ids=("aut",),
            thread_id=None,
            activated_nodes=("n1",),
            reward_events=(),
            scn_tag=None,
        )
        assert ep.imagined is False

    def test_episode_imagined_true(self):
        from maxim.memory.episode import Episode

        ep = Episode(
            id="test",
            start_tick=0,
            end_tick=10,
            channel="sim",
            sender_ids=("aut",),
            thread_id=None,
            activated_nodes=("n1",),
            reward_events=(),
            scn_tag=None,
            imagined=True,
        )
        assert ep.imagined is True

    def test_episode_to_dict_imagined_true(self):
        from maxim.memory.episode import Episode

        ep = Episode(
            id="test",
            start_tick=0,
            end_tick=10,
            channel="sim",
            sender_ids=(),
            thread_id=None,
            activated_nodes=(),
            reward_events=(),
            scn_tag=None,
            imagined=True,
        )
        d = ep.to_dict()
        assert d["imagined"] is True

    def test_episode_to_dict_imagined_false_omitted(self):
        from maxim.memory.episode import Episode

        ep = Episode(
            id="test",
            start_tick=0,
            end_tick=10,
            channel="sim",
            sender_ids=(),
            thread_id=None,
            activated_nodes=(),
            reward_events=(),
            scn_tag=None,
        )
        d = ep.to_dict()
        assert "imagined" not in d  # Omitted when False (backward compat)

    def test_episode_from_dict_backward_compat(self):
        from maxim.memory.episode import Episode

        # Old data without imagined field
        d = {
            "id": "old",
            "start_tick": 0,
            "end_tick": 5,
            "channel": "sim",
            "sender_ids": [],
            "thread_id": None,
            "activated_nodes": [],
            "reward_events": [],
            "scn_tag": None,
        }
        ep = Episode.from_dict(d)
        assert ep.imagined is False

    def test_episode_from_dict_imagined(self):
        from maxim.memory.episode import Episode

        d = {
            "id": "new",
            "start_tick": 0,
            "end_tick": 5,
            "channel": "sim",
            "sender_ids": [],
            "thread_id": None,
            "activated_nodes": [],
            "reward_events": [],
            "scn_tag": None,
            "imagined": True,
        }
        ep = Episode.from_dict(d)
        assert ep.imagined is True

    def test_pending_episode_imagined_finalize(self):
        from maxim.memory.episode import PendingEpisodeState

        pending = PendingEpisodeState(id="p1", start_tick=0, last_tick=5, channel="sim")
        pending.imagined = True
        ep = pending.finalize()
        assert ep.imagined is True

    def test_causal_link_imagined_default_false(self):
        from maxim.decisions.causal_link import CausalLink, Valence, TemporalDelta

        link = CausalLink(
            id="test",
            event_type="tool",
            event_signature="use_sword",
            event_context={},
            outcome_type="tool_result",
            outcome_signature="damage",
            outcome_valence=Valence.POSITIVE,
            temporal_delta=TemporalDelta(observed_deltas=(1.0,)),
        )
        assert link.imagined is False

    def test_causal_link_imagined_serialization(self):
        from maxim.decisions.causal_link import CausalLink, Valence, TemporalDelta

        link = CausalLink(
            id="test",
            event_type="tool",
            event_signature="use_sword",
            event_context={},
            outcome_type="tool_result",
            outcome_signature="damage",
            outcome_valence=Valence.POSITIVE,
            temporal_delta=TemporalDelta(observed_deltas=(1.0,)),
            imagined=True,
        )
        d = link.to_dict()
        assert d["imagined"] is True
        restored = CausalLink.from_dict(d)
        assert restored.imagined is True

    def test_causal_link_from_dict_backward_compat(self):
        from maxim.decisions.causal_link import CausalLink

        d = {
            "id": "old",
            "event_type": "tool",
            "event_signature": "test",
            "event_context": {},
            "outcome_type": "tool_result",
            "outcome_signature": "result",
            "outcome_valence": "positive",
            "temporal_delta": {"observed_deltas": [1.0]},
        }
        link = CausalLink.from_dict(d)
        assert link.imagined is False


# ---------------------------------------------------------------------------
# NAc decay_imagined_links
# ---------------------------------------------------------------------------


class TestNacDecayImaginedLinks:
    """Tests for NAc.decay_imagined_links."""

    def test_decay_imagined_links(self):
        from maxim.decisions.nac import NAc
        from maxim.decisions.causal_link import CausalLink, Valence, TemporalDelta

        nac = NAc()
        # Create links directly for testing
        link_normal = CausalLink(
            id="normal",
            event_type="tool",
            event_signature="normal_tool",
            event_context={},
            outcome_type="tool_result",
            outcome_signature="result",
            outcome_valence=Valence.POSITIVE,
            temporal_delta=TemporalDelta(observed_deltas=(1.0,)),
            confidence=0.8,
        )
        link_imagined = CausalLink(
            id="imagined",
            event_type="tool",
            event_signature="imagined_tool",
            event_context={},
            outcome_type="tool_result",
            outcome_signature="result",
            outcome_valence=Valence.POSITIVE,
            temporal_delta=TemporalDelta(observed_deltas=(1.0,)),
            confidence=0.8,
            imagined=True,
        )
        nac._links["normal_tool"] = [link_normal]
        nac._links["imagined_tool"] = [link_imagined]

        count = nac.decay_imagined_links(factor=0.5)
        assert count == 1
        assert abs(link_imagined.confidence - 0.4) < 0.01
        assert abs(link_normal.confidence - 0.8) < 0.01


# ---------------------------------------------------------------------------
# DN arousal gate
# ---------------------------------------------------------------------------


class TestDNArousalGate:
    """Tests for DefaultNetwork.imagination_allowed."""

    def _make_dn(self, running=True, inhibited=False, idle_seconds=10.0):
        """Create a mock DN with controllable arousal state."""
        dn = MagicMock()
        dn._running = running
        dn._inhibited = inhibited
        dn._last_interesting_time = time.time() - idle_seconds
        dn._config = MagicMock()
        dn._config.idle_exploration_min_seconds = 3.0

        # Use the real method
        from maxim.default_network.network import DefaultNetwork

        dn.imagination_allowed = DefaultNetwork.imagination_allowed.__get__(dn)
        return dn

    def test_allowed_when_idle(self):
        dn = self._make_dn(running=True, inhibited=False, idle_seconds=10.0)
        assert dn.imagination_allowed() is True

    def test_blocked_when_inhibited(self):
        dn = self._make_dn(running=True, inhibited=True, idle_seconds=10.0)
        assert dn.imagination_allowed() is False

    def test_blocked_when_aroused(self):
        """Recent interesting event blocks imagination."""
        dn = self._make_dn(running=True, inhibited=False, idle_seconds=0.5)
        assert dn.imagination_allowed() is False

    def test_allowed_when_not_running(self):
        """DN not running (sim mode) → always allowed."""
        dn = self._make_dn(running=False, inhibited=False, idle_seconds=0.0)
        assert dn.imagination_allowed() is True


# ---------------------------------------------------------------------------
# ImaginationDesigner
# ---------------------------------------------------------------------------


class TestImaginationDesigner:
    """Tests for ImaginationDesigner."""

    def test_infer_entity_type_creature(self):
        assert ImaginationDesigner._infer_entity_type("giant wolf") == "creatures"

    def test_infer_entity_type_weapon(self):
        assert ImaginationDesigner._infer_entity_type("rusty sword") == "weapons"

    def test_infer_entity_type_item(self):
        assert ImaginationDesigner._infer_entity_type("healing potion") == "items"

    def test_infer_entity_type_environment(self):
        assert ImaginationDesigner._infer_entity_type("iron gate") == "environments"

    def test_infer_entity_type_npc(self):
        assert ImaginationDesigner._infer_entity_type("old merchant") == "npcs"

    def test_infer_entity_type_vehicle(self):
        assert ImaginationDesigner._infer_entity_type("hover bike") == "vehicles"

    def test_infer_entity_type_unknown(self):
        assert ImaginationDesigner._infer_entity_type("mysterious glowing orb") is None

    def test_slugify(self):
        assert ImaginationDesigner._slugify("Rusty Iron Gate") == "rusty_iron_gate"
        assert ImaginationDesigner._slugify("a b c") == "a_b_c"
        assert ImaginationDesigner._slugify("") == "unknown_entity"

    def test_imagine_with_mock_designer(self):
        mock_designer = MagicMock()
        mock_designer.design.return_value = {
            "entity": {
                "name": "shadow_spider",
                "sensors": {
                    "hp": {"unit": "health", "range": [0, 100], "initial": 50},
                },
                "modulators": {
                    "body": {
                        "affordances": {
                            "bite": {"description": "venomous bite"},
                        }
                    }
                },
                "synonyms": ["dark spider", "shadow arachnid"],
            }
        }

        designer = ImaginationDesigner(entity_designer=mock_designer)
        result = designer.imagine("shadow spider", {"genre": "fantasy"})

        assert result is not None
        assert result.ref == "creatures/shadow_spider"
        assert "dark spider" in result.synonyms
        assert "shadow spider" in result.synonyms

    def test_imagine_validation_failure(self):
        mock_designer = MagicMock()
        mock_designer.design.return_value = {
            "entity": {
                # Missing name → validation should fail
            }
        }

        designer = ImaginationDesigner(entity_designer=mock_designer)
        result = designer.imagine("broken entity")
        assert result is None

    def test_imagine_design_exception(self):
        mock_designer = MagicMock()
        mock_designer.design.side_effect = RuntimeError("LLM unavailable")

        designer = ImaginationDesigner(entity_designer=mock_designer)
        result = designer.imagine("impossible entity")
        assert result is None

    def test_sensor_sanity_warnings(self):
        warnings = ImaginationDesigner._sensor_sanity(
            {
                "sensors": {
                    "hp": {"range": [0, 0], "initial": 0},
                    "energy": {"range": [0, 100000], "initial": 50},
                    "mana": {"range": [0, 100], "initial": 200},
                }
            }
        )
        assert any("zero-width" in w for w in warnings)
        assert any("very large" in w for w in warnings)
        assert any("outside range" in w for w in warnings)


# ---------------------------------------------------------------------------
# ImaginationTrigger
# ---------------------------------------------------------------------------


class TestImaginationTrigger:
    """Tests for ImaginationTrigger end-to-end."""

    def _make_trigger(self, dn=None, designer=None, threshold=2):
        mock_index = MagicMock()
        mock_index.find.return_value = None  # Default: nothing found
        mock_registry = MagicMock()
        cache = ImaginationCache()
        trigger = ImaginationTrigger(
            component_index=mock_index,
            component_registry=mock_registry,
            designer=designer,
            cache=cache,
            default_network=dn,
            imagination_threshold=threshold,
        )
        return trigger, mock_index, mock_registry

    def test_disabled_returns_empty(self):
        trigger, _, _ = self._make_trigger()
        trigger.enabled = False
        results = trigger.process_percept("You see a wolf.")
        assert results == []

    def test_index_hit_caches_result(self):
        trigger, mock_index, _ = self._make_trigger(threshold=1)
        from maxim.embodiment.component_index import ComponentMatch

        mock_index.find.return_value = ComponentMatch(ref="creatures/wolf", name="wolf", score=0.9, layer="alias")

        results = trigger.process_percept("You see a wolf.")
        assert len(results) == 1
        assert results[0].ref == "creatures/wolf"
        assert results[0].imagined is False

    def test_cache_hit_on_second_call(self):
        trigger, mock_index, _ = self._make_trigger(threshold=1)
        from maxim.embodiment.component_index import ComponentMatch

        mock_index.find.return_value = ComponentMatch(ref="creatures/wolf", name="wolf", score=0.9, layer="alias")

        trigger.process_percept("You see a wolf.")
        trigger.process_percept("The wolf growls.")

        stats = trigger.stats()
        assert stats["cache_hits"] >= 1

    def test_threshold_gates_design(self):
        """Novel phrase below threshold → no design."""
        mock_designer = MagicMock()
        trigger, _, _ = self._make_trigger(designer=mock_designer, threshold=3)

        # First two mentions — below threshold
        trigger.process_percept("You see a strange sword.")
        trigger.process_percept("The sword glows.")

        mock_designer.imagine.assert_not_called()

    def test_design_on_threshold_reached(self):
        """Novel phrase at threshold → design fires."""
        mock_designer = MagicMock()
        mock_designer.imagine.return_value = DesignResult(
            ref="weapons/glowing_sword",
            spec={"entity": {"name": "glowing_sword"}},
            synonyms=["shining blade"],
            validation_warnings=(),
        )
        trigger, _, mock_registry = self._make_trigger(designer=mock_designer, threshold=2)

        trigger.process_percept("You see a glowing sword.")
        trigger.process_percept("The glowing sword hums.")

        mock_designer.imagine.assert_called_once()
        mock_registry.register_ephemeral.assert_called_once()

    def test_energy_gate_blocks_design(self):
        """Critical energy → design blocked."""
        mock_designer = MagicMock()
        trigger, _, _ = self._make_trigger(designer=mock_designer, threshold=1)

        with patch("maxim.imagination.trigger.ImaginationTrigger._is_energy_available", return_value=False):
            trigger.process_percept("You see a cursed dagger.")
            mock_designer.imagine.assert_not_called()

    def test_arousal_gate_blocks_design(self):
        """High arousal → design blocked."""
        mock_dn = MagicMock()
        mock_dn.imagination_allowed.return_value = False
        mock_designer = MagicMock()
        trigger, _, _ = self._make_trigger(dn=mock_dn, designer=mock_designer, threshold=1)

        trigger.process_percept("You see a fire dragon.")
        mock_designer.imagine.assert_not_called()

    def test_clear_session(self):
        trigger, _, _ = self._make_trigger()
        trigger._cache.put(ImaginationResult(phrase="wolf", ref="creatures/wolf", imagined=True))
        trigger.clear_session()
        assert trigger.cache.size == 0
        stats = trigger.stats()
        assert stats["phrases_extracted"] == 0

    def test_no_designer_skips_gracefully(self):
        trigger, _, _ = self._make_trigger(designer=None, threshold=1)
        results = trigger.process_percept("You see a mystical sword.")
        # No designer → no design, but no crash
        assert all(not r.imagined for r in results)


# ---------------------------------------------------------------------------
# Affordance concept encoding (Stage 1)
# ---------------------------------------------------------------------------


class TestAffordanceConceptEncoding:
    """Tests for affordance name encoding through the substrate path.

    Verifies that when entities are registered, their affordance names
    are decomposed and encoded through EC → ATL → NAc eligibility.
    """

    def _make_entity(self, name="dragon", affordances=None):
        """Create a minimal Entity with affordances for testing."""
        from maxim.embodiment.sem import AffordanceSchema, Entity
        from maxim.embodiment.spec import SpecModulator

        if affordances is None:
            affordances = {
                "fire_breath": AffordanceSchema(description="breathe fire"),
                "tail_sweep": AffordanceSchema(description="sweep with tail"),
            }
        mod = SpecModulator(_name="combat", _entity_name=name, _affordances=affordances)
        ent = Entity(name=name, entity_type="creature")
        ent.modulators["combat"] = mod
        return ent

    def _make_encoder(self):
        """Create a LinguisticEncoder with EC + ATL + NAc for testing."""
        from maxim.decisions.nac import NAc
        from maxim.memory.atl import ATL
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import LinguisticEncoder

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.50))
        atl = ATL()
        nac = NAc()
        encoder = LinguisticEncoder(ec=ec, atl=atl, nac=nac)
        return encoder, ec, atl, nac

    def test_encode_entity_affordances_creates_nodes(self):
        """Encoding affordances creates substrate nodes in ATL."""
        from maxim.imagination.trigger import encode_entity_affordances

        encoder, ec, atl, nac = self._make_encoder()
        entity = self._make_entity()

        node_ids = encode_entity_affordances(entity, encoder, agent_id="aut-1")

        # fire_breath → ["fire breath", "fire", "breath"] = 3 nodes
        # tail_sweep → ["tail sweep", "tail", "sweep"] = 3 nodes
        # (some may share if EC completes to existing nodes)
        assert len(node_ids) >= 4  # At least 4 unique nodes

    def test_encode_creates_eligibility_traces(self):
        """Encoded affordance concepts have NAc eligibility traces."""
        from maxim.imagination.trigger import encode_entity_affordances

        encoder, ec, atl, nac = self._make_encoder()
        entity = self._make_entity()

        node_ids = encode_entity_affordances(entity, encoder, agent_id="aut-1")

        # Each node should have an eligibility trace under the agent_id
        for nid in node_ids:
            assert ("aut-1", nid) in nac._eligibility
            assert nac._eligibility[("aut-1", nid)] > 0

    def test_encode_creates_atl_concepts(self):
        """Encoded affordances produce ATL substrate concepts."""
        from maxim.imagination.trigger import encode_entity_affordances

        encoder, ec, atl, nac = self._make_encoder()
        entity = self._make_entity()

        encode_entity_affordances(entity, encoder, agent_id="aut-1")

        # ATL should have substrate concepts. Short words may complete to
        # the same node as their compound (EC groups by similarity), so
        # check category rather than exact name.
        substrate_concepts = atl.recall(category="substrate")
        assert len(substrate_concepts) > 0

    def test_shared_component_creates_single_node(self):
        """Two entities with overlapping affordance components share EC nodes."""
        from maxim.embodiment.sem import AffordanceSchema
        from maxim.imagination.trigger import encode_entity_affordances

        encoder, ec, atl, nac = self._make_encoder()

        dragon = self._make_entity("dragon", {"fire_breath": AffordanceSchema(description="fire")})
        mage = self._make_entity("mage", {"fire_bolt": AffordanceSchema(description="fire")})

        dragon_nodes = encode_entity_affordances(dragon, encoder, agent_id="aut-1")
        mage_nodes = encode_entity_affordances(mage, encoder, agent_id="aut-1")

        # "fire" from both should complete to the same EC node
        # (cosine("fire", "fire") = 1.0 > 0.50 threshold)
        fire_nodes_dragon = [n for n in dragon_nodes]
        fire_nodes_mage = [n for n in mage_nodes]
        shared = set(fire_nodes_dragon) & set(fire_nodes_mage)
        assert len(shared) >= 1, "Dragon and mage should share at least the 'fire' node"

    def test_trigger_encodes_on_entity_registration(self):
        """ImaginationTrigger with encoder encodes affordances on entity live."""
        encoder, ec, atl, nac = self._make_encoder()

        mock_index = MagicMock()
        mock_registry = MagicMock()

        trigger = ImaginationTrigger(
            component_index=mock_index,
            component_registry=mock_registry,
            encoder=encoder,
            agent_id="aut-1",
        )

        # Wire entity_map so _ensure_entity_live can register
        mock_entity_map = MagicMock()
        mock_entity_map.resolve.return_value = None  # Not yet live
        trigger._entity_map = mock_entity_map

        # Mock registry.get to return a parseable spec
        mock_registry.get.return_value = {
            "entity": {
                "name": "dragon",
                "entity_type": "creature",
                "modulators": {
                    "combat": {
                        "name": "combat",
                        "affordances": {
                            "fire_breath": {"description": "breathe fire"},
                        },
                    },
                },
            },
        }

        trigger._ensure_entity_live("creatures/dragon", "dragon", None, None)

        # Verify ATL got substrate concepts for the affordance
        substrate_concepts = atl.recall(category="substrate")
        assert len(substrate_concepts) > 0

    def test_no_encoder_skips_gracefully(self):
        """Trigger without encoder doesn't crash on entity registration."""
        mock_index = MagicMock()
        mock_registry = MagicMock()

        trigger = ImaginationTrigger(
            component_index=mock_index,
            component_registry=mock_registry,
            # No encoder
            agent_id="aut-1",
        )
        assert trigger._aff_encoder is None
        # _encode_entity_affordances should return empty
        entity = self._make_entity()
        result = trigger._encode_entity_affordances(entity)
        assert result == []

    def test_single_word_affordance_creates_one_node(self):
        """Single-word affordance (no underscore) creates exactly one node."""
        from maxim.embodiment.sem import AffordanceSchema
        from maxim.imagination.trigger import encode_entity_affordances

        encoder, ec, atl, nac = self._make_encoder()
        entity = self._make_entity("beast", {"circle": AffordanceSchema(description="fly in circles")})

        node_ids = encode_entity_affordances(entity, encoder, agent_id="aut-1")

        assert len(node_ids) == 1  # "circle" → one chunk → one node


# ---------------------------------------------------------------------------
# Phase 1: process_manifest + generate_scene_manifest
# ---------------------------------------------------------------------------


class TestProcessManifest:
    """Tests for ImaginationTrigger.process_manifest()."""

    def _make_trigger(self, designer=None, threshold=2):
        mock_index = MagicMock()
        mock_index.find.return_value = None
        mock_registry = MagicMock()
        cache = ImaginationCache()
        trigger = ImaginationTrigger(
            component_index=mock_index,
            component_registry=mock_registry,
            designer=designer,
            cache=cache,
            imagination_threshold=threshold,
        )
        return trigger, mock_index, mock_registry

    def test_manifest_bypasses_threshold(self):
        """Entities in manifest resolve on first mention — no threshold."""
        trigger, mock_index, _ = self._make_trigger(threshold=99)
        from maxim.embodiment.component_index import ComponentMatch

        mock_index.find.return_value = ComponentMatch(ref="creatures/dragon", name="dragon", score=1.0, layer="alias")

        results = trigger.process_manifest("a fire-breathing dragon in the cave")
        assert len(results) == 1
        assert results[0].ref == "creatures/dragon"

    def test_manifest_bypasses_arousal_gate(self):
        """process_manifest does NOT check arousal — pre-trigger is deliberate."""
        mock_dn = MagicMock()
        mock_dn.imagination_allowed.return_value = False  # Would block process_percept

        mock_index = MagicMock()
        from maxim.embodiment.component_index import ComponentMatch

        mock_index.find.return_value = ComponentMatch(ref="creatures/wolf", name="wolf", score=1.0, layer="alias")
        trigger = ImaginationTrigger(
            component_index=mock_index,
            component_registry=MagicMock(),
            default_network=mock_dn,
            imagination_threshold=2,
        )

        results = trigger.process_manifest("a large wolf appears")
        assert len(results) == 1
        # arousal gate was NOT checked
        mock_dn.imagination_allowed.assert_not_called()

    def test_manifest_caps_entities(self):
        """Hard cap on max_entities limits total resolved."""
        trigger, mock_index, _ = self._make_trigger()
        from maxim.embodiment.component_index import ComponentMatch

        # Return a match for every query
        mock_index.find.side_effect = lambda q: ComponentMatch(
            ref=f"items/{q.split()[-1]}", name=q.split()[-1], score=1.0, layer="alias"
        )

        manifest = "a rusty sword\na magic staff\na healing potion\na silver dagger\na wooden bow\n"
        results = trigger.process_manifest(manifest, max_entities=3)
        assert len(results) <= 3

    def test_manifest_caps_designs(self):
        """max_designs limits LLM design calls for novel entities."""
        mock_designer = MagicMock()
        call_count = 0

        def _imagine(phrase, ctx):
            nonlocal call_count
            call_count += 1
            return DesignResult(
                ref=f"items/{phrase.replace(' ', '_')}",
                spec={"entity": {"name": phrase.replace(" ", "_")}},
                synonyms=[],
                validation_warnings=(),
            )

        mock_designer.imagine.side_effect = _imagine
        trigger, mock_index, _ = self._make_trigger(designer=mock_designer)

        manifest = "a crystal orb\na silver amulet\na magic scroll\na healing potion\na rusty dagger\n"
        trigger.process_manifest(manifest, max_designs=2)
        assert call_count <= 2

    def test_manifest_skips_self_entity(self):
        """Self-entity is not resolved via manifest to avoid ownership collision."""
        trigger, mock_index, _ = self._make_trigger()
        from maxim.embodiment.component_index import ComponentMatch

        mock_index.find.return_value = ComponentMatch(
            ref="bodies/base_humanoid", name="base_humanoid", score=1.0, layer="alias"
        )

        # Simulate EntityMap with a self-entity named "humanoid"
        mock_entity = MagicMock()
        mock_entity.name = "humanoid"
        mock_entity_map = MagicMock()
        mock_entity_map.list_self_entities.return_value = [mock_entity]
        mock_entity_map.resolve.return_value = None  # Not already live
        trigger._entity_map = mock_entity_map

        # "humanoid" head noun matches self-entity → skipped
        # "guard" is also an indicator but "humanoid" is the self-entity
        trigger.process_manifest("a humanoid figure stands guard")
        # Test directly with head noun = "humanoid"
        results2 = trigger.process_manifest("the humanoid")
        # Head noun is "humanoid" which matches self-entity
        assert len(results2) == 0

    def test_manifest_skips_already_live(self):
        """Already-live entities are skipped."""
        trigger, mock_index, _ = self._make_trigger()

        mock_entity_map = MagicMock()
        mock_entity_map.list_self_entities.return_value = []
        # dragon is already live
        mock_entity_map.resolve.side_effect = lambda n: "exists" if n == "dragon" else None
        trigger._entity_map = mock_entity_map

        results = trigger.process_manifest("a fire-breathing dragon")
        assert len(results) == 0

    def test_manifest_disabled_returns_empty(self):
        trigger, _, _ = self._make_trigger()
        trigger.enabled = False
        results = trigger.process_manifest("a dragon and a wolf")
        assert results == []

    def test_manifest_empty_text(self):
        trigger, _, _ = self._make_trigger()
        results = trigger.process_manifest("")
        assert results == []

    def test_manifest_index_hits_before_designs(self):
        """ComponentIndex hits are processed before any LLM designs."""
        call_order = []
        trigger, mock_index, _ = self._make_trigger()

        from maxim.embodiment.component_index import ComponentMatch

        def _find(q):
            if "dragon" in q:
                call_order.append("index_hit")
                return ComponentMatch(ref="creatures/dragon", name="dragon", score=1.0, layer="alias")
            call_order.append("index_miss")
            return None

        mock_index.find.side_effect = _find

        mock_designer = MagicMock()

        def _imagine(phrase, ctx):
            call_order.append("design")
            return DesignResult(
                ref=f"items/{phrase.replace(' ', '_')}",
                spec={"entity": {"name": phrase.replace(" ", "_")}},
                synonyms=[],
                validation_warnings=(),
            )

        mock_designer.imagine.side_effect = _imagine
        trigger._designer = mock_designer

        trigger.process_manifest("a dragon guards a magic scroll")
        # Index hits come before designs
        assert call_order.index("index_hit") < call_order.index("design")


class TestGenerateSceneManifest:
    """Tests for generate_scene_manifest()."""

    def test_returns_llm_text(self):
        from maxim.simulation.narrator import generate_scene_manifest

        mock_llm = MagicMock()
        mock_llm.generate_json.return_value = {"entities": ["a rusty sword", "a large dragon", "a healing potion"]}

        result = generate_scene_manifest(mock_llm, "explore a dungeon")
        assert "dragon" in result
        assert "sword" in result

    def test_returns_empty_on_failure(self):
        from maxim.simulation.narrator import generate_scene_manifest

        mock_llm = MagicMock()
        mock_llm.generate_json.side_effect = RuntimeError("LLM down")

        result = generate_scene_manifest(mock_llm, "explore a dungeon")
        assert result == ""

    def test_returns_empty_on_none(self):
        from maxim.simulation.narrator import generate_scene_manifest

        mock_llm = MagicMock()
        mock_llm.generate_json.return_value = None

        result = generate_scene_manifest(mock_llm, "test memory")
        assert result == ""

    def test_returns_empty_on_whitespace(self):
        from maxim.simulation.narrator import generate_scene_manifest

        mock_llm = MagicMock()
        mock_llm.generate_json.return_value = {"entities": []}

        result = generate_scene_manifest(mock_llm, "test")
        assert result == ""


class TestGenerateSceneManifestSubstrateAware:
    """W2 Hookup 1 — substrate-aware manifest generation.

    Pins the contract for ``generate_scene_manifest(..., nac_top_biases=...)``:

    - ``None`` (default) preserves the pre-W2 byte-identical prompt
      so cradle's substrate-blind path is unchanged.
    - non-None + non-empty injects Wire-A's shared section header
      (``=== Substrate associations from prior experience ===``) plus
      a manifest-specific directive line.
    - **Substrate-voice consistency** (bio-fidelity fold C1): rendering
      delegates to ``compose_cluster_bias_annotation_section`` so the
      manifest LLM and AUT proposer LLM see the same band labels, same
      "from prior experience" framing, and NO raw-float leakage.
    - All-neutral lists produce no section (matches Wire-A's filter).

    See docs/plans/imagination_substrate_signals.md Hookup 1.
    """

    def _capture_prompt(self, mock_llm) -> str:
        """Pull the prompt sent to ``generate_json`` from the mock."""
        assert mock_llm.generate_json.called
        args, kwargs = mock_llm.generate_json.call_args
        # generate_json(prompt, max_tokens=...) — prompt is positional.
        return args[0] if args else kwargs.get("prompt", "")

    def test_none_biases_preserves_pre_w2_prompt(self):
        """Default ``nac_top_biases=None`` must produce the pre-W2 prompt."""
        from maxim.simulation.narrator import generate_scene_manifest

        mock_llm_default = MagicMock()
        mock_llm_default.generate_json.return_value = {"entities": ["a sword"]}
        generate_scene_manifest(mock_llm_default, "explore a dungeon")
        prompt_default = self._capture_prompt(mock_llm_default)

        mock_llm_none = MagicMock()
        mock_llm_none.generate_json.return_value = {"entities": ["a sword"]}
        generate_scene_manifest(mock_llm_none, "explore a dungeon", nac_top_biases=None)
        prompt_none = self._capture_prompt(mock_llm_none)

        assert prompt_default == prompt_none
        assert "Substrate associations" not in prompt_default

    def test_empty_biases_preserves_pre_w2_prompt(self):
        """Empty list must behave the same as ``None`` (no section emitted)."""
        from maxim.simulation.narrator import generate_scene_manifest

        mock_llm_default = MagicMock()
        mock_llm_default.generate_json.return_value = {"entities": ["a sword"]}
        generate_scene_manifest(mock_llm_default, "explore a dungeon")
        prompt_default = self._capture_prompt(mock_llm_default)

        mock_llm_empty = MagicMock()
        mock_llm_empty.generate_json.return_value = {"entities": ["a sword"]}
        generate_scene_manifest(mock_llm_empty, "explore a dungeon", nac_top_biases=[])
        prompt_empty = self._capture_prompt(mock_llm_empty)

        assert prompt_default == prompt_empty
        assert "Substrate associations" not in prompt_empty

    def test_all_neutral_biases_emit_no_section(self):
        """All-neutral biases route through the shared filter and emit
        nothing — matches Wire-A's all-neutral-skip behavior so the
        manifest LLM doesn't get token-noise without signal."""
        from maxim.simulation.narrator import generate_scene_manifest

        mock_llm_default = MagicMock()
        mock_llm_default.generate_json.return_value = {"entities": ["a sword"]}
        generate_scene_manifest(mock_llm_default, "explore a dungeon")
        prompt_default = self._capture_prompt(mock_llm_default)

        mock_llm_neutral = MagicMock()
        mock_llm_neutral.generate_json.return_value = {"entities": ["a sword"]}
        # All entries in (-0.1, 0.1) → neutral / mixed band.
        generate_scene_manifest(
            mock_llm_neutral,
            "explore a dungeon",
            nac_top_biases=[("tool:a", 0.05), ("tool:b", -0.08)],
        )
        prompt_neutral = self._capture_prompt(mock_llm_neutral)

        assert prompt_default == prompt_neutral
        assert "Substrate associations" not in prompt_neutral

    def test_non_empty_biases_injects_substrate_section(self):
        """Non-empty biases inject the shared Wire-A renderer output
        plus W2's manifest-specific directive."""
        from maxim.simulation.narrator import generate_scene_manifest

        mock_llm = MagicMock()
        mock_llm.generate_json.return_value = {"entities": ["a fruit"]}

        biases = [
            ("tool:sense_food_source", 0.85),
            ("tool:sense_water", -0.6),
        ]
        generate_scene_manifest(mock_llm, "find food", nac_top_biases=biases)
        prompt = self._capture_prompt(mock_llm)

        # Shared header from Wire-A's renderer.
        assert "=== Substrate associations from prior experience ===" in prompt
        # Bare tool names (prefix stripped) appear in the prompt.
        assert "sense_food_source" in prompt
        assert "sense_water" in prompt
        # The "tool:" prefix must NOT appear in the rendered tool-name lines
        # (Wire-A's strip_tool_prefix discipline).
        assert "tool:sense_food_source" not in prompt
        assert "tool:sense_water" not in prompt
        # Bands route through bias_to_band — 0.85 → strongly rewarding.
        assert "strongly rewarding" in prompt
        # -0.6 → strongly aversive (|-0.6| >= 0.5 lower band).
        assert "strongly aversive" in prompt
        # The W2-specific directive line is present so the manifest LLM
        # reads the section as direction, not just data.
        assert "prefer entities" in prompt

    def test_no_raw_float_leakage(self):
        """Bio-fidelity fold C1 — raw float values MUST NOT appear in
        the prompt. The substrate-voice argument (cluster_bias_annotation.py
        docstring lines 30-53) is that bands are display-layer translation,
        not substrate state. Leaking the numeric anchors the LLM on
        threshold boundaries across surfaces."""
        from maxim.simulation.narrator import generate_scene_manifest

        mock_llm = MagicMock()
        mock_llm.generate_json.return_value = {"entities": ["x"]}
        # Deliberately distinct values so any leak is detectable.
        biases = [
            ("tool:a", 0.857),
            ("tool:b", -0.731),
        ]
        generate_scene_manifest(mock_llm, "g", nac_top_biases=biases)
        prompt = self._capture_prompt(mock_llm)

        # None of the source values may appear in any rendered form
        # (raw decimal, signed, or three-digit truncation).
        for needle in ("0.857", "+0.857", "0.731", "-0.731", "0.86", "0.73"):
            assert needle not in prompt, f"raw float leak: {needle!r} found in prompt"

    def test_from_prior_experience_framing_present(self):
        """Bio-fidelity fold C1 — substrate-voice framing must mark the
        signal as episodic recall, not standing fact. Wire-A appends
        ``from prior experience`` to non-neutral bands; W2 inherits this
        via the shared renderer."""
        from maxim.simulation.narrator import generate_scene_manifest

        mock_llm = MagicMock()
        mock_llm.generate_json.return_value = {"entities": ["x"]}
        biases = [("tool:sense_food_source", 0.85)]
        generate_scene_manifest(mock_llm, "g", nac_top_biases=biases)
        prompt = self._capture_prompt(mock_llm)

        assert "from prior experience" in prompt

    def test_substrate_section_appears_before_goal(self):
        """Substrate context precedes the goal so the LLM reads
        preferences as standing context, not goal-overriding noise."""
        from maxim.simulation.narrator import generate_scene_manifest

        mock_llm = MagicMock()
        mock_llm.generate_json.return_value = {"entities": ["x"]}
        biases = [("tool:sense_food_source", 0.85)]
        generate_scene_manifest(mock_llm, "explore", nac_top_biases=biases)
        prompt = self._capture_prompt(mock_llm)

        substrate_idx = prompt.index("=== Substrate associations from prior experience ===")
        goal_idx = prompt.index("Simulation goal:")
        assert substrate_idx < goal_idx

    def test_band_rendering_routes_through_bias_to_band(self):
        """All four non-neutral bands (per ``bias_to_band``) render via
        the shared section. Neutral entries appear when mixed with non-
        neutrals (per Wire-A's renderer)."""
        from maxim.simulation.narrator import generate_scene_manifest

        # bias_to_band thresholds: 0.5 / 0.1 / -0.1 / -0.5
        biases = [
            ("tool:a", 0.9),  # strongly rewarding
            ("tool:b", 0.3),  # mildly rewarding
            ("tool:c", 0.0),  # neutral / mixed (kept by Wire-A when mixed)
            ("tool:d", -0.3),  # mildly aversive
            ("tool:e", -0.9),  # strongly aversive
        ]
        mock_llm = MagicMock()
        mock_llm.generate_json.return_value = {"entities": ["x"]}
        generate_scene_manifest(mock_llm, "g", nac_top_biases=biases)
        prompt = self._capture_prompt(mock_llm)

        assert "strongly rewarding" in prompt
        assert "mildly rewarding" in prompt
        assert "neutral / mixed" in prompt
        assert "mildly aversive" in prompt
        assert "strongly aversive" in prompt

    def test_uses_shared_renderer_not_duplicated_logic(self):
        """Cross-confirmation guard — the W2 manifest section MUST be a
        superset of Wire-A's renderer output. If a future refactor
        replaces the shared call with a private composer, this test
        catches the substrate-voice drift."""
        from maxim.prompts.cluster_bias_annotation import (
            compose_cluster_bias_annotation_section,
        )
        from maxim.simulation.narrator import generate_scene_manifest

        biases = [
            ("tool:sense_food_source", 0.85),
            ("tool:sense_water", -0.6),
        ]
        mock_llm = MagicMock()
        mock_llm.generate_json.return_value = {"entities": ["x"]}
        generate_scene_manifest(mock_llm, "g", nac_top_biases=biases)
        prompt = self._capture_prompt(mock_llm)

        shared_section = compose_cluster_bias_annotation_section(biases)
        assert shared_section  # guard against test-fixture regression
        assert shared_section in prompt


class TestFindAliasOnly:
    """Tests for ComponentIndex.find_alias_only()."""

    def test_exact_alias_match(self):
        from maxim.embodiment.component_index import ComponentIndex

        index = ComponentIndex()
        index._aliases = {"dragon": "creatures/dragon"}

        match = index.find_alias_only("dragon")
        assert match is not None
        assert match.ref == "creatures/dragon"
        assert match.score == 1.0

    def test_head_noun_fallback(self):
        from maxim.embodiment.component_index import ComponentIndex

        index = ComponentIndex()
        index._aliases = {"dragon": "creatures/dragon"}

        match = index.find_alias_only("fire-breathing dragon")
        assert match is not None
        assert match.ref == "creatures/dragon"
        assert match.score == 0.95

    def test_no_embedding_computation(self):
        """find_alias_only must NOT trigger embedding computation."""
        from maxim.embodiment.component_index import ComponentIndex

        index = ComponentIndex()
        index._aliases = {}
        index._embed = MagicMock(side_effect=AssertionError("should not be called"))

        result = index.find_alias_only("something unknown")
        assert result is None

    def test_empty_query(self):
        from maxim.embodiment.component_index import ComponentIndex

        index = ComponentIndex()
        assert index.find_alias_only("") is None
        assert index.find_alias_only("   ") is None
