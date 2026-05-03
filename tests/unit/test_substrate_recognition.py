"""Tests for substrate recognition — B1 + P1 components.

Covers:
- SubstrateModality derivation
- EC pattern_complete_or_separate
- ATL modality tags + filtered queries
- LinguisticEncoder pipeline
- Dual-path flag wiring
- P1 metric extractor
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from maxim.agents.bus import Percept
from maxim.agents.modality import (
    SensoryModality,
    SensoryTag,
    _SUBSTRATE_MAP,
    substrate_modality,
)
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.semantic_types import Concept, SemanticMemory
from maxim.similarity.ec import ECConfig, EntorhinalCortex, _cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# SubstrateModality tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSubstrateModality:
    """SubstrateModality derivation from SensoryModality."""

    def test_sight_maps_to_vision(self):
        percept = Percept(
            timestamp=0.0,
            source="test",
            sensory=SensoryTag(modality=SensoryModality.SIGHT),
        )
        assert substrate_modality(percept) == "vision"

    def test_sound_maps_to_text(self):
        percept = Percept(
            timestamp=0.0,
            source="test",
            sensory=SensoryTag(modality=SensoryModality.SOUND),
        )
        assert substrate_modality(percept) == "text"

    def test_narrative_maps_to_text(self):
        percept = Percept(
            timestamp=0.0,
            source="test",
            sensory=SensoryTag(modality=SensoryModality.NARRATIVE),
        )
        assert substrate_modality(percept) == "text"

    def test_abstract_maps_to_text(self):
        percept = Percept(
            timestamp=0.0,
            source="test",
            sensory=SensoryTag(modality=SensoryModality.ABSTRACT),
        )
        assert substrate_modality(percept) == "text"

    def test_interoception_maps_to_text(self):
        percept = Percept(
            timestamp=0.0,
            source="test",
            sensory=SensoryTag(modality=SensoryModality.INTEROCEPTION),
        )
        assert substrate_modality(percept) == "text"

    def test_no_sensory_falls_back_to_modality_literal(self):
        percept = Percept(timestamp=0.0, source="test", modality="vision")
        assert substrate_modality(percept) == "vision"

    def test_no_sensory_no_modality_defaults_to_text(self):
        percept = Percept(timestamp=0.0, source="test")
        assert substrate_modality(percept) == "text"

    def test_all_modalities_have_mapping(self):
        for mod in SensoryModality:
            assert mod in _SUBSTRATE_MAP, f"{mod} missing from _SUBSTRATE_MAP"


# ─────────────────────────────────────────────────────────────────────────────
# Percept embedding field tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPerceptEmbedding:
    """Percept.embedding and substrate_node_id fields."""

    def test_embedding_default_none(self):
        p = Percept(timestamp=0.0, source="test")
        assert p.embedding is None

    def test_substrate_node_id_default_none(self):
        p = Percept(timestamp=0.0, source="test")
        assert p.substrate_node_id is None

    def test_embedding_settable(self):
        p = Percept(timestamp=0.0, source="test")
        p.embedding = [0.1, 0.2, 0.3]
        assert p.embedding == [0.1, 0.2, 0.3]

    def test_to_dict_includes_substrate_node_id(self):
        p = Percept(timestamp=0.0, source="test", substrate_node_id="abc123")
        d = p.to_dict()
        assert d["substrate_node_id"] == "abc123"

    def test_from_dict_round_trip(self):
        p = Percept(timestamp=1.0, source="test", substrate_node_id="node42")
        d = p.to_dict()
        p2 = Percept.from_dict(d)
        assert p2.substrate_node_id == "node42"


# ─────────────────────────────────────────────────────────────────────────────
# EC pattern_complete_or_separate tests
# ─────────────────────────────────────────────────────────────────────────────


class TestECPatternCompletion:
    """EntorhinalCortex.pattern_complete_or_separate."""

    def test_first_embedding_creates_new_node(self):
        ec = EntorhinalCortex()
        emb = [1.0, 0.0, 0.0]
        result = ec.pattern_complete_or_separate(emb, "text")
        assert result.is_new
        assert result.similarity == 0.0
        assert result.node_id

    def test_identical_embedding_completes(self):
        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.90))
        emb = [1.0, 0.0, 0.0]
        r1 = ec.pattern_complete_or_separate(emb, "text")
        ec.register_substrate_node(r1.node_id, emb, "text")
        r2 = ec.pattern_complete_or_separate(emb, "text")
        assert not r2.is_new
        assert r2.node_id == r1.node_id
        assert r2.similarity == pytest.approx(1.0)

    def test_orthogonal_embedding_separates(self):
        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.50))
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        r1 = ec.pattern_complete_or_separate(emb1, "text")
        ec.register_substrate_node(r1.node_id, emb1, "text")
        r2 = ec.pattern_complete_or_separate(emb2, "text")
        assert r2.is_new
        assert r2.node_id != r1.node_id

    def test_modality_isolation(self):
        """Text and vision nodes don't compete."""
        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.50))
        emb = [1.0, 0.0, 0.0]
        r1 = ec.pattern_complete_or_separate(emb, "text")
        ec.register_substrate_node(r1.node_id, emb, "text")
        # Same embedding but different modality — should NOT match
        r2 = ec.pattern_complete_or_separate(emb, "vision")
        assert r2.is_new
        assert r2.node_id != r1.node_id

    def test_threshold_override_widens_radius(self):
        """Per-node threshold override from P2 reward modulation."""
        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.99))
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.6, 0.8, 0.0]  # ~0.6 cosine sim — below 0.99 but above 0.50
        r1 = ec.pattern_complete_or_separate(emb1, "text")
        ec.register_substrate_node(r1.node_id, emb1, "text")

        # Without override, high threshold means separation
        r2 = ec.pattern_complete_or_separate(emb2, "text")
        assert r2.is_new

        # With override, lower threshold allows completion
        r3 = ec.pattern_complete_or_separate(emb2, "text", threshold_override={r1.node_id: 0.50})
        assert not r3.is_new
        assert r3.node_id == r1.node_id

    def test_substrate_node_count(self):
        ec = EntorhinalCortex()
        assert ec.substrate_node_count == 0
        r = ec.pattern_complete_or_separate([1.0, 0.0], "text")
        ec.register_substrate_node(r.node_id, [1.0, 0.0], "text")
        assert ec.substrate_node_count == 1

    def test_remove_substrate_node(self):
        ec = EntorhinalCortex()
        r = ec.pattern_complete_or_separate([1.0, 0.0], "text")
        ec.register_substrate_node(r.node_id, [1.0, 0.0], "text")
        ec.remove_substrate_node(r.node_id)
        assert ec.substrate_node_count == 0

    def test_persistence_round_trip(self, tmp_path):
        """Substrate nodes survive save/load."""
        ec = EntorhinalCortex()
        emb = [0.5, 0.5, 0.5]
        r = ec.pattern_complete_or_separate(emb, "text")
        ec.register_substrate_node(r.node_id, emb, "text")

        path = str(tmp_path / "ec.json")
        ec.save(path)

        ec2 = EntorhinalCortex()
        ec2.load(path)
        assert ec2.substrate_node_count == 1
        # Should pattern-complete to the loaded node
        r2 = ec2.pattern_complete_or_separate(emb, "text")
        assert not r2.is_new
        assert r2.node_id == r.node_id


class TestCosineSimlarity:
    """_cosine_similarity helper."""

    def test_identical(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ATL modality tag tests
# ─────────────────────────────────────────────────────────────────────────────


class TestATLModalityTags:
    """ATL modality-tagged concepts and filtered queries."""

    def test_store_with_substrate_modality(self):
        atl = ATL()
        mem = SemanticMemory(
            id="test-1",
            timestamp=time.time(),
            name="hello",
            category="substrate",
            substrate_modality="text",
        )
        atl.store(mem)
        assert "test-1" in atl._modality_index["text"]

    def test_recall_filtered_by_modality(self):
        atl = ATL()
        mem_text = SemanticMemory(
            id="t1",
            timestamp=time.time(),
            name="word",
            category="substrate",
            substrate_modality="text",
        )
        mem_vision = SemanticMemory(
            id="v1",
            timestamp=time.time(),
            name="object",
            category="substrate",
            substrate_modality="vision",
        )
        atl.store(mem_text)
        atl.store(mem_vision)

        text_only = atl.recall(limit=10, substrate_modality="text")
        assert len(text_only) == 1
        assert text_only[0].id == "t1"

        vision_only = atl.recall(limit=10, substrate_modality="vision")
        assert len(vision_only) == 1
        assert vision_only[0].id == "v1"

    def test_get_by_modality(self):
        atl = ATL()
        for i in range(5):
            atl.store(
                SemanticMemory(
                    id=f"text-{i}",
                    timestamp=time.time(),
                    name=f"word-{i}",
                    category="substrate",
                    substrate_modality="text",
                )
            )
        atl.store(
            SemanticMemory(
                id="vis-0",
                timestamp=time.time(),
                name="obj-0",
                category="substrate",
                substrate_modality="vision",
            )
        )

        text_concepts = atl.get_by_modality("text")
        assert len(text_concepts) == 5
        vision_concepts = atl.get_by_modality("vision")
        assert len(vision_concepts) == 1

    def test_activate_substrate_node_creates_concept(self):
        atl = ATL()
        node_id = atl.activate_substrate_node(
            node_id="new-node",
            text="hello world",
            substrate_modality="text",
        )
        assert node_id == "new-node"
        concept = atl.get("new-node")
        assert concept is not None
        assert concept.substrate_modality == "text"
        assert concept.category == "substrate"

    def test_activate_substrate_node_reinforces_existing(self):
        atl = ATL()
        atl.activate_substrate_node("n1", "hello", "text")
        concept_before = atl.get("n1")
        count_before = concept_before.reinforcement_count

        atl.activate_substrate_node("n1", "hello", "text")
        concept_after = atl.get("n1")
        assert concept_after.reinforcement_count > count_before

    def test_remove_cleans_modality_index(self):
        atl = ATL()
        atl.store(
            SemanticMemory(
                id="rm1",
                timestamp=time.time(),
                name="bye",
                category="substrate",
                substrate_modality="text",
            )
        )
        assert "rm1" in atl._modality_index["text"]
        atl.remove("rm1")
        assert "rm1" not in atl._modality_index["text"]

    def test_persistence_preserves_modality(self, tmp_path):
        """substrate_modality survives save/load."""
        atl = ATL(ATLConfig(persistence_path=str(tmp_path / "atl.json")))
        atl.activate_substrate_node("n1", "hello", "text")
        atl.save()

        atl2 = ATL(ATLConfig(persistence_path=str(tmp_path / "atl.json")))
        atl2.load()
        concept = atl2.get("n1")
        assert concept is not None
        assert concept.substrate_modality == "text"
        assert "n1" in atl2._modality_index["text"]


# ─────────────────────────────────────────────────────────────────────────────
# LinguisticEncoder tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLinguisticEncoder:
    """LinguisticEncoder pipeline (uses fallback embedder in tests)."""

    def _make_encoder(self):
        from maxim.similarity.encoder import LinguisticEncoder

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.50))
        atl = ATL()
        return LinguisticEncoder(ec=ec, atl=atl), ec, atl

    def test_encode_creates_node(self):
        encoder, ec, atl = self._make_encoder()
        percept = Percept(
            timestamp=0.0,
            source="test",
            transcript_chunk="Hello, how are you?",
        )
        node_id = encoder.encode(percept)
        assert node_id is not None
        assert percept.embedding is not None
        assert percept.substrate_node_id == node_id
        assert ec.substrate_node_count == 1
        assert len(atl) == 1

    def test_encode_no_text_returns_none(self):
        encoder, _, _ = self._make_encoder()
        percept = Percept(timestamp=0.0, source="test")
        assert encoder.encode(percept) is None

    def test_encode_uses_content_if_no_transcript(self):
        encoder, _, _ = self._make_encoder()
        percept = Percept(timestamp=0.0, source="test", content="some content")
        node_id = encoder.encode(percept)
        assert node_id is not None

    def test_identical_text_collapses(self):
        """Same text should map to same node (deterministic fallback)."""
        encoder, ec, atl = self._make_encoder()
        p1 = Percept(timestamp=0.0, source="test", transcript_chunk="Hello world")
        p2 = Percept(timestamp=0.0, source="test", transcript_chunk="Hello world")
        n1 = encoder.encode(p1)
        n2 = encoder.encode(p2)
        assert n1 == n2
        assert ec.substrate_node_count == 1

    def test_different_text_separates(self):
        """Very different text should create different nodes."""
        encoder, ec, _ = self._make_encoder()
        p1 = Percept(timestamp=0.0, source="test", transcript_chunk="hello world")
        p2 = Percept(timestamp=0.0, source="test", transcript_chunk="quantum physics equations")
        encoder.encode(p1)
        encoder.encode(p2)
        assert ec.substrate_node_count == 2

    def test_using_fallback_flag(self):
        from maxim.similarity.encoder import LinguisticEncoder

        ec = EntorhinalCortex()
        atl = ATL()
        encoder = LinguisticEncoder(ec=ec, atl=atl)
        # In test env, sentence-transformers may not be installed
        # so this should be True (fallback) or False (real model)
        assert isinstance(encoder.using_fallback, bool)

    def test_stats(self):
        encoder, _, _ = self._make_encoder()
        stats = encoder.stats()
        assert "model_name" in stats
        assert "using_fallback" in stats


class TestEncoderNAcEligibility:
    """P2: encoder must seed eligibility on BOTH new-node and completion paths.

    Regression guard for the `not is_new` asymmetry that blocked the
    reward-widens-recognition bootstrap — a just-created target node
    must be creditable by a reward arriving in the same tick.
    """

    def _make(self):
        from maxim.decisions.nac import NAc
        from maxim.similarity.encoder import LinguisticEncoder

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.50))
        atl = ATL()
        nac = NAc()
        encoder = LinguisticEncoder(ec=ec, atl=atl, nac=nac)
        return encoder, ec, atl, nac

    def test_new_node_creation_seeds_eligibility(self):
        encoder, _, _, nac = self._make()
        percept = Percept(timestamp=0.0, source="test", transcript_chunk="first encounter")
        node_id = encoder.encode(percept)
        assert node_id is not None
        # Eligibility trace for the brand-new node should be non-zero
        # so a reward arriving next tick can credit it.
        assert ("", node_id) in nac._eligibility
        assert nac._eligibility[("", node_id)] == pytest.approx(1.0)

    def test_completion_updates_eligibility_with_similarity(self):
        encoder, _, _, nac = self._make()
        p1 = Percept(timestamp=0.0, source="test", transcript_chunk="hello world")
        p2 = Percept(timestamp=0.0, source="test", transcript_chunk="hello world")
        n1 = encoder.encode(p1)
        # First encode seeded eligibility at 1.0; second (identical) encode
        # completes and overwrites with similarity (≈1.0 for identical text).
        n2 = encoder.encode(p2)
        assert n1 == n2
        assert nac._eligibility[("", n1)] > 0.0

    def test_reward_credits_just_created_node(self):
        """End-to-end: encode → distribute_reward → reward_bias > 0."""
        encoder, _, _, nac = self._make()
        percept = Percept(timestamp=0.0, source="test", transcript_chunk="target concept")
        node_id = encoder.encode(percept)
        credited = nac.distribute_reward(agent_id="", reward=1.0)
        # The brand-new node must appear in the credited list — this is the
        # bootstrap case the old guard broke.
        credited_ids = [nid for nid, _ in credited]
        assert node_id in credited_ids
        assert nac.reward_bias("", node_id) > 0.0

    def test_no_nac_no_eligibility_updates(self):
        """Encoder without NAc wiring must not crash."""
        from maxim.similarity.encoder import LinguisticEncoder

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.50))
        atl = ATL()
        encoder = LinguisticEncoder(ec=ec, atl=atl)  # no nac
        percept = Percept(timestamp=0.0, source="test", transcript_chunk="hello")
        assert encoder.encode(percept) is not None

    def test_reward_overrides_fire_on_empty_nac(self):
        """Regression: NAc.__len__ returns 0 for fresh NAc, which made
        `if self._nac` falsy and silently suppressed reward-bias threshold
        overrides. Must use `is not None` to check wiring.
        """
        from maxim.decisions.nac import NAc
        from maxim.similarity.encoder import LinguisticEncoder

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.50))
        atl = ATL()
        nac = NAc()
        assert len(nac) == 0
        assert bool(nac) is False  # ← the footgun
        encoder = LinguisticEncoder(ec=ec, atl=atl, nac=nac)

        # Encode one percept and credit its node.
        p1 = Percept(timestamp=0.0, source="test", transcript_chunk="anchor")
        node_id = encoder.encode(p1)
        nac.credit_node(agent_id="", node_id=node_id, reward=2.0)
        assert nac.get_threshold_overrides("") == {node_id: pytest.approx(0.20)}

        # Spy on EC to capture the threshold_override dict the encoder passes.
        captured: list[dict[str, float] | None] = []
        orig = ec.pattern_complete_or_separate

        def spy(embedding, modality, threshold=None, threshold_override=None):
            captured.append(threshold_override)
            return orig(embedding, modality, threshold, threshold_override)

        ec.pattern_complete_or_separate = spy  # type: ignore[method-assign]

        p2 = Percept(timestamp=0.0, source="test", transcript_chunk="second")
        encoder.encode(p2)

        # The encoder must have passed a non-empty override dict through
        # to EC. Before the fix, `if self._nac` was False (empty __len__)
        # and the override was silently None.
        assert captured, "EC.pattern_complete_or_separate was not called"
        assert captured[-1] is not None, (
            "encoder suppressed reward overrides because `if self._nac` treated an empty NAc as falsy — regression!"
        )
        assert node_id in captured[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Dual-path flag wiring tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDualPathWiring:
    """MAXIM_SUBSTRATE_PATH env var wiring."""

    def test_memory_hub_substrate_disabled_by_default(self):
        """Without env var, substrate path is off."""
        from maxim.integration.memory_hub import MemoryHub

        hub = MemoryHub(
            hippocampus=Mock(),
            scn=Mock(register=Mock(), unregister=Mock()),
            nac=Mock(remove_memory=Mock()),
            ec=Mock(semantic_enabled=False, remove_signature=Mock()),
        )
        assert not hub._substrate_enabled
        assert hub._encoder is None

    def test_memory_hub_substrate_enabled_with_flag(self):
        """With env var and ATL, substrate path activates."""
        import os

        from maxim.integration.memory_hub import MemoryHub

        os.environ["MAXIM_SUBSTRATE_PATH"] = "1"
        try:
            atl = ATL()
            ec = EntorhinalCortex()
            hub = MemoryHub(
                hippocampus=Mock(),
                scn=Mock(register=Mock(), unregister=Mock()),
                nac=Mock(remove_memory=Mock()),
                ec=ec,
                atl=atl,
            )
            assert hub._substrate_enabled
            assert hub._encoder is not None
        finally:
            os.environ.pop("MAXIM_SUBSTRATE_PATH", None)

    def test_on_percept_received_no_op_when_disabled(self):
        """Calling on_percept_received when disabled is a safe no-op."""
        from maxim.integration.memory_hub import MemoryHub

        hub = MemoryHub(
            hippocampus=Mock(),
            scn=Mock(register=Mock(), unregister=Mock()),
            nac=Mock(remove_memory=Mock()),
            ec=Mock(semantic_enabled=False, remove_signature=Mock()),
        )
        percept = Percept(timestamp=0.0, source="test", transcript_chunk="hello")
        # Should not raise
        hub.on_percept_received(percept)


# ─────────────────────────────────────────────────────────────────────────────
# SemanticMemory substrate_modality field tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSemanticMemorySubstrateModality:
    """substrate_modality field on SemanticMemory and Concept."""

    def test_default_none(self):
        mem = SemanticMemory(id="t1", timestamp=0.0)
        assert mem.substrate_modality is None

    def test_serialization_round_trip(self):
        mem = SemanticMemory(
            id="t1",
            timestamp=1.0,
            name="test",
            substrate_modality="text",
        )
        d = mem.to_dict()
        assert d["substrate_modality"] == "text"
        mem2 = SemanticMemory.from_dict(d)
        assert mem2.substrate_modality == "text"

    def test_concept_inherits_substrate_modality(self):
        concept = Concept(
            id="c1",
            timestamp=1.0,
            name="test",
            substrate_modality="vision",
        )
        d = concept.to_dict()
        assert d["substrate_modality"] == "vision"
        c2 = Concept.from_dict(d)
        assert c2.substrate_modality == "vision"

    def test_legacy_data_without_field(self):
        """Old data without substrate_modality deserializes as None."""
        d = {"id": "old", "timestamp": 1.0, "name": "legacy"}
        mem = SemanticMemory.from_dict(d)
        assert mem.substrate_modality is None
