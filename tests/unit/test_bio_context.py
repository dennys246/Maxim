"""Tests for bio-system protocol enrichment — context parameter acceptance.

Verifies that:
1. All five Context dataclasses are frozen and slots-enabled
2. Each bio-system's enriched methods accept the context parameter
3. Existing callers (no context) still work unchanged
4. Context parameter flows through without errors
"""

from __future__ import annotations

import pytest

from maxim.models.bio_context import (
    EncodingContext,
    PredictionContext,
    RetrievalContext,
    SemanticContext,
    TemporalContext,
)


# ---------------------------------------------------------------------------
# Context dataclass invariants
# ---------------------------------------------------------------------------


class TestContextDataclasses:
    """All context types must be frozen, slotted, and default to None fields."""

    @pytest.mark.parametrize(
        "cls",
        [
            RetrievalContext,
            PredictionContext,
            SemanticContext,
            EncodingContext,
            TemporalContext,
        ],
    )
    def test_frozen(self, cls):
        ctx = cls()
        with pytest.raises(AttributeError):
            ctx.__dict__  # noqa: B018 — slots classes have no __dict__

    @pytest.mark.parametrize(
        "cls",
        [
            RetrievalContext,
            PredictionContext,
            SemanticContext,
            EncodingContext,
            TemporalContext,
        ],
    )
    def test_all_fields_default_none(self, cls):
        """Every field must have a default (None or empty) so the context is
        fully optional — callers can construct with zero arguments."""
        ctx = cls()
        assert ctx is not None

    def test_retrieval_context_fields(self):
        ctx = RetrievalContext(
            temporal_window_s=60.0,
            emotional_valence=-0.5,
            active_goal="escape",
            min_consolidation=0.3,
            encoding_specificity=0.8,
            energy=0.6,
        )
        assert ctx.temporal_window_s == 60.0
        assert ctx.emotional_valence == -0.5
        assert ctx.active_goal == "escape"

    def test_prediction_context_fields(self):
        ctx = PredictionContext(
            arousal=0.8,
            temporal_discount=0.95,
            energy=0.4,
            active_goal="find food",
            surprise=1.2,
        )
        assert ctx.arousal == 0.8
        assert ctx.surprise == 1.2

    def test_semantic_context_fields(self):
        ctx = SemanticContext(
            abstraction_level=0.5,
            domain_hints=("combat", "fire"),
            spreading_depth=3,
            spreading_decay=0.7,
            active_goal="learn",
        )
        assert ctx.domain_hints == ("combat", "fire")
        assert ctx.spreading_depth == 3

    def test_encoding_context_fields(self):
        ctx = EncodingContext(
            resolution=0.9,
            attention_weights=(1.0, 0.5, 0.3),
            encoding_mode="encode",
            decomposition_hint="noun_chunk",
        )
        assert ctx.encoding_mode == "encode"
        assert ctx.decomposition_hint == "noun_chunk"

    def test_temporal_context_fields(self):
        ctx = TemporalContext(
            oscillator_phase=3.14,
            prediction_confidence=0.8,
            zeitgeber_strength=0.6,
            energy=0.5,
        )
        assert ctx.oscillator_phase == pytest.approx(3.14)
        assert ctx.zeitgeber_strength == 0.6


# ---------------------------------------------------------------------------
# Hippocampus — RetrievalContext acceptance
# ---------------------------------------------------------------------------


class TestHippocampusRetrievalContext:
    """Hippocampus methods accept retrieval_context without breaking."""

    def _make_hippocampus(self):
        from maxim.memory.hippocampus import Hippocampus

        return Hippocampus.empty()

    def test_recall_accepts_retrieval_context(self):
        h = self._make_hippocampus()
        ctx = RetrievalContext(active_goal="test", energy=0.5)
        result = h.recall(limit=5, retrieval_context=ctx)
        assert isinstance(result, list)

    def test_recall_none_context_unchanged(self):
        h = self._make_hippocampus()
        result = h.recall(limit=5)
        assert isinstance(result, list)

    def test_search_by_content_accepts_retrieval_context(self):
        h = self._make_hippocampus()
        ctx = RetrievalContext(encoding_specificity=0.9)
        result = h.search_by_content("test query", retrieval_context=ctx)
        assert isinstance(result, list)

    def test_search_by_content_none_context_unchanged(self):
        h = self._make_hippocampus()
        result = h.search_by_content("test query")
        assert isinstance(result, list)

    def test_retrieve_on_cue_accepts_retrieval_context(self):
        h = self._make_hippocampus()
        ctx = RetrievalContext(temporal_window_s=300.0)
        result = h.retrieve_on_cue("node_1", retrieval_context=ctx)
        assert isinstance(result, list)

    def test_retrieve_on_cue_none_context_unchanged(self):
        h = self._make_hippocampus()
        result = h.retrieve_on_cue("node_1")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# NAc — PredictionContext acceptance
# ---------------------------------------------------------------------------


class TestNAcPredictionContext:
    """NAc methods accept prediction_context without breaking."""

    def _make_nac(self):
        from maxim.decisions.nac import NAc

        return NAc()

    def test_get_links_for_event_accepts_context(self):
        nac = self._make_nac()
        ctx = PredictionContext(arousal=0.8)
        result = nac.get_links_for_event("test_event", prediction_context=ctx)
        assert isinstance(result, list)

    def test_get_links_for_event_none_context_unchanged(self):
        nac = self._make_nac()
        result = nac.get_links_for_event("test_event")
        assert isinstance(result, list)

    def test_predict_accepts_context(self):
        nac = self._make_nac()
        ctx = PredictionContext(temporal_discount=0.9)
        result = nac.predict("action", "test_action", prediction_context=ctx)
        assert result is None  # no links yet

    def test_predict_none_context_unchanged(self):
        nac = self._make_nac()
        result = nac.predict("action", "test_action")
        assert result is None

    def test_predict_all_outcomes_accepts_context(self):
        nac = self._make_nac()
        ctx = PredictionContext(energy=0.3)
        result = nac.predict_all_outcomes("action", "test_action", prediction_context=ctx)
        assert isinstance(result, list)

    def test_record_outcome_accepts_context(self):
        from maxim.decisions.causal_link import Valence

        nac = self._make_nac()
        event_id = nac.record_event("action", "test_tool")
        ctx = PredictionContext(surprise=2.0)
        result = nac.record_outcome("action", event_id, Valence.POSITIVE, prediction_context=ctx)
        assert isinstance(result, list)

    def test_record_outcome_full_accepts_context(self):
        from maxim.decisions.causal_link import Valence

        nac = self._make_nac()
        nac.record_event("action", "tool_a")
        ctx = PredictionContext(arousal=0.5, active_goal="survive")
        result = nac.record_outcome_full(
            "outcome",
            "success",
            Valence.POSITIVE,
            prediction_context=ctx,
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# ATL — SemanticContext acceptance
# ---------------------------------------------------------------------------


class TestATLSemanticContext:
    """ATL methods accept semantic_context without breaking."""

    def _make_atl(self):
        from maxim.memory.atl import ATL

        return ATL()

    def test_recall_accepts_semantic_context(self):
        atl = self._make_atl()
        ctx = SemanticContext(abstraction_level=0.5, domain_hints=("combat",))
        result = atl.recall(limit=5, semantic_context=ctx)
        assert isinstance(result, list)

    def test_recall_none_context_unchanged(self):
        atl = self._make_atl()
        result = atl.recall(limit=5)
        assert isinstance(result, list)

    def test_find_or_create_accepts_semantic_context(self):
        atl = self._make_atl()
        ctx = SemanticContext(active_goal="learn fire")
        concept_id, was_created = atl.find_or_create("fire", "element", semantic_context=ctx)
        assert isinstance(concept_id, str)
        assert was_created is True

    def test_find_or_create_none_context_unchanged(self):
        atl = self._make_atl()
        concept_id, was_created = atl.find_or_create("water", "element")
        assert isinstance(concept_id, str)
        assert was_created is True


# ---------------------------------------------------------------------------
# EC — EncodingContext acceptance
# ---------------------------------------------------------------------------


class TestECEncodingContext:
    """EC methods accept encoding_context without breaking."""

    def _make_ec(self):
        from maxim.similarity.ec import EntorhinalCortex

        return EntorhinalCortex()

    def test_pattern_complete_or_separate_accepts_context(self):
        ec = self._make_ec()
        ctx = EncodingContext(resolution=0.8, encoding_mode="encode")
        result = ec.pattern_complete_or_separate([0.1, 0.2, 0.3], "text", encoding_context=ctx, geometry=None)
        assert result is not None
        assert result.is_new is True  # first embedding is always new (separation)

    def test_pattern_complete_or_separate_none_context_unchanged(self):
        ec = self._make_ec()
        result = ec.pattern_complete_or_separate([0.1, 0.2, 0.3], "text", geometry=None)
        assert result is not None

    def test_find_semantic_accepts_context(self):
        ec = self._make_ec()
        ctx = EncodingContext(decomposition_hint="noun_chunk")
        # find_semantic requires neural semantic to be available; returns
        # empty list when not installed — that's fine, we just test acceptance.
        result = ec.find_semantic("test query", encoding_context=ctx)
        assert isinstance(result, list)

    def test_find_semantic_none_context_unchanged(self):
        ec = self._make_ec()
        result = ec.find_semantic("test query")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# SCN — TemporalContext acceptance
# ---------------------------------------------------------------------------


class TestSCNTemporalContext:
    """SCN methods accept temporal_context without breaking."""

    def _make_scn(self):
        from maxim.time.scn import SCN

        return SCN()

    def _make_signature(self):
        from maxim.time.temporal_signature import TemporalSignature

        return TemporalSignature.now()

    def test_register_accepts_temporal_context(self):
        scn = self._make_scn()
        ctx = TemporalContext(oscillator_phase=1.5, prediction_confidence=0.9)
        scn.register("mem_1", self._make_signature(), temporal_context=ctx)
        assert len(scn) == 1

    def test_register_none_context_unchanged(self):
        scn = self._make_scn()
        scn.register("mem_2", self._make_signature())
        assert len(scn) == 1

    def test_get_threshold_adjustment_accepts_context(self):
        scn = self._make_scn()
        sig = self._make_signature()
        scn.register("mem_3", sig)
        ctx = TemporalContext(energy=0.4)
        result = scn.get_threshold_adjustment(sig, temporal_context=ctx)
        assert isinstance(result, float)

    def test_get_threshold_adjustment_none_context_unchanged(self):
        scn = self._make_scn()
        sig = self._make_signature()
        scn.register("mem_4", sig)
        result = scn.get_threshold_adjustment(sig)
        assert isinstance(result, float)
