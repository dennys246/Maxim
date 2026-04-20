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
