"""Unit tests for concept decomposition (similarity/decomposer.py).

Tests the Protocol-based decomposer, built-in strategies, and encoder
integration. Uses spaCy ``en_core_web_sm`` when available; tests that
require spaCy are marked with ``pytest.importorskip``.
"""

from __future__ import annotations

import pytest

from maxim.similarity.decomposer import (
    ConceptChunk,
    ConceptDecomposer,
    DecomposerConfig,
    DecompositionStrategy,
    IdentityStrategy,
)


# ─────────────────────────────────────────────────────────────────────────
# IdentityStrategy
# ─────────────────────────────────────────────────────────────────────────


class TestIdentityStrategy:
    def test_returns_full_text_as_single_chunk(self) -> None:
        s = IdentityStrategy()
        chunks = s.extract("I see a blue mug on the table")
        assert len(chunks) == 1
        assert chunks[0].text == "I see a blue mug on the table"
        assert chunks[0].span == (0, 29)

    def test_single_word(self) -> None:
        s = IdentityStrategy()
        chunks = s.extract("mug")
        assert len(chunks) == 1
        assert chunks[0].text == "mug"

    def test_empty_string(self) -> None:
        s = IdentityStrategy()
        chunks = s.extract("")
        assert len(chunks) == 1
        assert chunks[0].text == ""


# ─────────────────────────────────────────────────────────────────────────
# Custom strategy (Protocol compliance)
# ─────────────────────────────────────────────────────────────────────────


class SplitOnCommaStrategy:
    """Test strategy that splits on commas."""

    def extract(self, text: str) -> list[ConceptChunk]:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return [ConceptChunk(text=p) for p in parts] or [ConceptChunk(text=text)]


class TestCustomStrategy:
    def test_protocol_compliance(self) -> None:
        assert isinstance(SplitOnCommaStrategy(), DecompositionStrategy)

    def test_custom_strategy_in_decomposer(self) -> None:
        d = ConceptDecomposer(strategy=SplitOnCommaStrategy())
        chunks = d.extract("cat, dog, fish")
        assert len(chunks) == 3
        assert [c.text for c in chunks] == ["cat", "dog", "fish"]

    def test_strategy_name(self) -> None:
        d = ConceptDecomposer(strategy=SplitOnCommaStrategy())
        assert d.strategy_name == "SplitOnCommaStrategy"


# ─────────────────────────────────────────────────────────────────────────
# ConceptDecomposer coordinator
# ─────────────────────────────────────────────────────────────────────────


class TestConceptDecomposer:
    def test_disabled_uses_identity(self) -> None:
        d = ConceptDecomposer(config=DecomposerConfig(enabled=False))
        assert d.strategy_name == "IdentityStrategy"
        chunks = d.extract("I see a blue mug")
        assert len(chunks) == 1
        assert chunks[0].text == "I see a blue mug"

    def test_empty_input(self) -> None:
        d = ConceptDecomposer(config=DecomposerConfig(enabled=False))
        chunks = d.extract("")
        assert len(chunks) == 1

    def test_whitespace_input(self) -> None:
        d = ConceptDecomposer(config=DecomposerConfig(enabled=False))
        chunks = d.extract("   ")
        assert len(chunks) == 1

    def test_explicit_strategy_overrides_config(self) -> None:
        d = ConceptDecomposer(
            strategy=SplitOnCommaStrategy(),
            config=DecomposerConfig(enabled=False),
        )
        # Explicit strategy takes precedence over enabled=False
        assert d.strategy_name == "SplitOnCommaStrategy"


# ─────────────────────────────────────────────────────────────────────────
# SpaCy strategy (requires en_core_web_sm)
# ─────────────────────────────────────────────────────────────────────────


class TestSpaCyNounChunkStrategy:
    @pytest.fixture(autouse=True)
    def _require_spacy(self) -> None:
        pytest.importorskip("spacy")
        # Also check the model is installed
        import spacy

        try:
            spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model en_core_web_sm not installed")

    def _make_strategy(self) -> "DecompositionStrategy":
        from maxim.similarity.decomposer import SpaCyNounChunkStrategy

        return SpaCyNounChunkStrategy()

    def test_basic_sentence(self) -> None:
        s = self._make_strategy()
        chunks = s.extract("I see a blue mug on the table")
        texts = [c.text for c in chunks]
        assert "blue mug" in texts
        assert "table" in texts

    def test_multiple_noun_phrases(self) -> None:
        s = self._make_strategy()
        chunks = s.extract("The red plate is next to the green cup")
        texts = [c.text for c in chunks]
        assert "red plate" in texts
        assert "green cup" in texts

    def test_bare_class_name_identity(self) -> None:
        """Bare class names (like P4 fixture uses) should pass through
        as single chunks — decomposition is a no-op."""
        s = self._make_strategy()
        for name in ["lotus", "azalea", "balloon flower", "water lily"]:
            chunks = s.extract(name)
            assert len(chunks) == 1, f"'{name}' decomposed into {len(chunks)} chunks: {[c.text for c in chunks]}"

    def test_strips_determiners(self) -> None:
        s = self._make_strategy()
        chunks = s.extract("the big red ball")
        texts = [c.text for c in chunks]
        assert any("big red ball" in t for t in texts)
        assert not any(t.startswith("the ") for t in texts)

    def test_filters_pronouns(self) -> None:
        s = self._make_strategy()
        chunks = s.extract("He picked it up from the shelf")
        texts = [c.text for c in chunks]
        # "He" and "it" should not appear as standalone chunks
        assert "He" not in texts
        assert "it" not in texts

    def test_span_offsets(self) -> None:
        s = self._make_strategy()
        text = "I see a blue mug"
        chunks = s.extract(text)
        for chunk in chunks:
            if chunk.span is not None:
                start, end = chunk.span
                assert text[start:end].strip() == chunk.text or chunk.text in text[start:end]

    def test_short_fragment_fallback(self) -> None:
        """Very short inputs with no noun phrases should fall back to full text."""
        s = self._make_strategy()
        chunks = s.extract("go")
        assert len(chunks) == 1
        assert chunks[0].text == "go"

    def test_sem_affordance_string(self) -> None:
        """SEM-style affordance descriptions should produce reasonable chunks."""
        s = self._make_strategy()
        chunks = s.extract("the rusty sword feels heavy")
        texts = [c.text for c in chunks]
        assert any("rusty sword" in t for t in texts)

    def test_complex_scene_description(self) -> None:
        """Multi-object scene should produce multiple concept chunks."""
        s = self._make_strategy()
        chunks = s.extract("I see a blue mug on the table next to the red plate")
        texts = [c.text for c in chunks]
        assert len(texts) >= 2
        assert any("mug" in t for t in texts)

    def test_minimum_chunk_length(self) -> None:
        from maxim.similarity.decomposer import SpaCyNounChunkStrategy

        s = SpaCyNounChunkStrategy(min_chunk_len=5)
        chunks = s.extract("a cat on the mat")
        texts = [c.text for c in chunks]
        # "cat" (3 chars) should be filtered; "mat" (3 chars) should be filtered
        # Falls back to full text
        for t in texts:
            assert len(t) >= 5 or t == "a cat on the mat"

    def test_thread_safety(self) -> None:
        """Multiple threads can call extract concurrently without crash."""
        import threading

        s = self._make_strategy()
        errors: list[Exception] = []

        def _worker() -> None:
            try:
                for _ in range(10):
                    chunks = s.extract("The quick brown fox jumps over the lazy dog")
                    assert len(chunks) >= 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread errors: {errors}"

    # ── Stage 2: relation extraction ──────────────────────────────────

    def test_spatial_relation(self) -> None:
        """Spatial prepositions (on, in, at) should produce spatial relations."""
        s = self._make_strategy()
        chunks = s.extract("I see a blue mug on the table")
        relations = [c.relation for c in chunks if c.relation is not None]
        # "table" is the pobj of "on" → spatial
        assert "spatial" in relations

    def test_action_relation(self) -> None:
        """Subject/object of a verb should produce action relations."""
        s = self._make_strategy()
        chunks = s.extract("The cat chased the mouse")
        relations = [c.relation for c in chunks if c.relation is not None]
        assert "action" in relations

    def test_possessive_relation(self) -> None:
        """Possessive modifiers produce possessive relations."""
        s = self._make_strategy()
        chunks = s.extract("The color of the sky is beautiful")
        relations = [c.relation for c in chunks if c.relation is not None]
        assert "possessive" in relations

    def test_bare_class_name_no_relation(self) -> None:
        """Bare class names should have no relation (single chunk)."""
        s = self._make_strategy()
        chunks = s.extract("mug")
        assert len(chunks) == 1
        assert chunks[0].relation is None

    def test_relation_preserved_in_chunk(self) -> None:
        """Relation is stored on the ConceptChunk itself."""
        s = self._make_strategy()
        chunks = s.extract("A sword on the stone")
        for c in chunks:
            if "stone" in c.text.lower():
                assert c.relation == "spatial", f"Expected 'spatial', got {c.relation!r}"


# ─────────────────────────────────────────────────────────────────────────
# Encoder integration (encode_decomposed)
# ─────────────────────────────────────────────────────────────────────────


class TestEncodeDecomposed:
    """Test LinguisticEncoder.encode_decomposed with a mock decomposer."""

    def _make_encoder_with_decomposer(
        self,
        strategy: DecompositionStrategy | None = None,
        pattern_complete_threshold: float = 1.0,
    ):
        """Build a LinguisticEncoder with EC + ATL mocks and a decomposer."""
        from unittest.mock import MagicMock

        from maxim.similarity.ec import ECConfig, EntorhinalCortex

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=pattern_complete_threshold))
        atl = MagicMock()
        atl.activate_substrate_node = MagicMock()

        decomposer = (
            ConceptDecomposer(strategy=strategy)
            if strategy
            else ConceptDecomposer(config=DecomposerConfig(enabled=False))
        )

        from maxim.similarity.encoder import LinguisticEncoder

        encoder = LinguisticEncoder(ec=ec, atl=atl, decomposer=decomposer)
        return encoder, ec, atl

    def test_identity_returns_single_node(self) -> None:
        encoder, ec, atl = self._make_encoder_with_decomposer()
        node_ids = encoder.encode_decomposed("lotus", "text")
        assert len(node_ids) == 1

    def test_custom_strategy_returns_multiple_nodes(self) -> None:
        encoder, ec, atl = self._make_encoder_with_decomposer(SplitOnCommaStrategy())
        node_ids = encoder.encode_decomposed("cat, dog, fish", "text")
        assert len(node_ids) == 3
        # Each should be a unique node ID
        assert len(set(node_ids)) == 3

    def test_vision_modality_bypasses_decomposition(self) -> None:
        encoder, ec, atl = self._make_encoder_with_decomposer(SplitOnCommaStrategy())
        # Even with comma strategy, vision modality should not decompose
        node_ids = encoder.encode_decomposed("cat, dog, fish", "vision")
        assert len(node_ids) == 1

    def test_empty_text_returns_empty(self) -> None:
        encoder, ec, atl = self._make_encoder_with_decomposer()
        node_ids = encoder.encode_decomposed("", "text")
        assert len(node_ids) == 0

    def test_nodes_registered_in_ec(self) -> None:
        encoder, ec, atl = self._make_encoder_with_decomposer(SplitOnCommaStrategy())
        node_ids = encoder.encode_decomposed("cat, dog", "text")
        # Each node should be in EC's substrate nodes
        for nid in node_ids:
            assert nid in ec._substrate_nodes

    def test_atl_activated_for_each_chunk(self) -> None:
        encoder, ec, atl = self._make_encoder_with_decomposer(SplitOnCommaStrategy())
        encoder.encode_decomposed("cat, dog, fish", "text")
        assert atl.activate_substrate_node.call_count == 3

    def test_pattern_completion_merges_identical_chunks(self) -> None:
        """Two identical concept texts should pattern-complete to the same node."""
        encoder, ec, atl = self._make_encoder_with_decomposer(SplitOnCommaStrategy(), pattern_complete_threshold=0.40)
        ids1 = encoder.encode_decomposed("cat, cat", "text")
        # The second "cat" should pattern-complete to the first
        assert len(ids1) == 2
        assert ids1[0] == ids1[1]
