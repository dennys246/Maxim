"""Unit tests for Nucleus Accumbens (NAc) - Causal inference and TD learning.

Tests the core functionality of causal link creation, observation,
prediction, and persistence.
"""

from __future__ import annotations

import pytest


class TestNAcObservation:
    """Test causal link creation and updates."""

    def test_observe_creates_new_link(self, nac, valence_positive):
        """First observation creates a new causal link."""
        link = nac.observe(
            event_type="tool",
            event_signature="internet_search",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=2.0,
        )

        assert link is not None
        assert link.observation_count == 1
        assert link.confidence == 0.5  # Initial confidence
        assert link.event_signature == "internet_search"

    def test_observe_updates_existing_link(self, nac, valence_positive):
        """Repeated observations update the existing link."""
        # First observation
        nac.observe(
            event_type="tool",
            event_signature="look_around",
            outcome_type="result",
            outcome_signature="found_object",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
            context={"mode": "exploration"},
        )

        # Same event-outcome pair
        link = nac.observe(
            event_type="tool",
            event_signature="look_around",
            outcome_type="result",
            outcome_signature="found_object",
            outcome_valence=valence_positive,
            delta_seconds=1.5,
            context={"mode": "exploration"},
        )

        assert link.observation_count == 2
        assert link.confidence > 0.5  # Should increase with observations

    def test_different_contexts_create_separate_links(self, nac, valence_positive):
        """Different contexts create separate causal links."""
        link1 = nac.observe(
            event_type="tool",
            event_signature="grasp",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
            context={"object": "mug"},
        )

        link2 = nac.observe(
            event_type="tool",
            event_signature="grasp",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
            context={"object": "plate"},  # Different context
        )

        # Should be different links
        assert link1.id != link2.id

    def test_observe_records_temporal_delta(self, nac, valence_positive):
        """Observations record temporal delays."""
        nac.observe(
            event_type="tool",
            event_signature="wait",
            outcome_type="result",
            outcome_signature="done",
            outcome_valence=valence_positive,
            delta_seconds=5.0,
        )
        nac.observe(
            event_type="tool",
            event_signature="wait",
            outcome_type="result",
            outcome_signature="done",
            outcome_valence=valence_positive,
            delta_seconds=3.0,
        )

        links = nac.get_links_for_event("wait")
        assert len(links) >= 1
        link = links[0]
        assert len(link.temporal_delta.observed_deltas) == 2
        assert link.temporal_delta.mean == 4.0


class TestNAcPrediction:
    """Test outcome prediction."""

    def test_predict_returns_none_for_unknown_event(self, nac):
        """Unknown events return None prediction."""
        prediction = nac.predict("tool", "never_seen_before")
        assert prediction is None

    def test_predict_returns_prediction_for_known_event(self, nac, valence_positive):
        """Known events return predictions."""
        # Create some observations
        for _ in range(3):
            nac.observe(
                event_type="tool",
                event_signature="reliable_tool",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        prediction = nac.predict("tool", "reliable_tool")

        assert prediction is not None
        assert prediction.event_signature == "reliable_tool"
        assert prediction.confidence > 0.3

    def test_predict_uses_highest_confidence_link(self, nac, valence_positive, valence_negative):
        """Prediction uses the highest-confidence link."""
        # Create many positive observations
        for _ in range(10):
            nac.observe(
                event_type="tool",
                event_signature="mixed_tool",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        # Create one negative observation
        nac.observe(
            event_type="tool",
            event_signature="mixed_tool",
            outcome_type="result",
            outcome_signature="failure",
            outcome_valence=valence_negative,
            delta_seconds=1.0,
        )

        prediction = nac.predict("tool", "mixed_tool")

        # Should predict based on more confident link (positive)
        assert prediction is not None
        assert prediction.predicted_value > 0.5

    def test_predict_respects_min_confidence_threshold(self, nac, valence_positive):
        """Low-confidence links are excluded from predictions."""
        from maxim.decisions.nac import NAc, NACConfig

        # Create NAc with high threshold
        strict_nac = NAc(config=NACConfig(min_confidence_threshold=0.9))

        # Single observation = low confidence
        strict_nac.observe(
            event_type="tool",
            event_signature="new_tool",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )

        prediction = strict_nac.predict("tool", "new_tool")
        assert prediction is None  # Below confidence threshold

    def test_cold_start_priors(self, nac):
        """Priors provide predictions when no links exist."""
        nac.set_prior("tool", "untested_tool", predicted_value=0.7, confidence=0.3)

        prediction = nac.predict("tool", "untested_tool")

        assert prediction is not None
        assert prediction.predicted_value == 0.7
        assert prediction.confidence == 0.3

    def test_predict_all_outcomes(self, nac, valence_positive, valence_negative):
        """predict_all_outcomes returns multiple possible outcomes."""
        # Same event, different outcomes
        nac.observe(
            event_type="tool",
            event_signature="risky_action",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )
        nac.observe(
            event_type="tool",
            event_signature="risky_action",
            outcome_type="result",
            outcome_signature="partial_success",
            outcome_valence=valence_positive,
            delta_seconds=2.0,
        )

        # Observe enough for confidence
        for _ in range(5):
            nac.observe(
                event_type="tool",
                event_signature="risky_action",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        predictions = nac.predict_all_outcomes("tool", "risky_action")

        assert len(predictions) >= 1


class TestNAcQueries:
    """Test link query methods."""

    def test_get_links_for_event(self, nac, valence_positive):
        """Query links by event signature."""
        nac.observe(
            event_type="tool",
            event_signature="specific_tool",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )

        links = nac.get_links_for_event("specific_tool")
        assert len(links) == 1
        assert links[0].event_signature == "specific_tool"

    def test_get_positive_outcomes(self, nac, valence_positive, valence_negative):
        """Query only positive outcome links."""
        nac.observe(
            event_type="tool",
            event_signature="tool_a",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )
        nac.observe(
            event_type="tool",
            event_signature="tool_a",
            outcome_type="result",
            outcome_signature="failure",
            outcome_valence=valence_negative,
            delta_seconds=1.0,
            context={"different": True},  # Different context for separate link
        )

        positive = nac.get_positive_outcomes("tool_a")
        negative = nac.get_negative_outcomes("tool_a")

        assert len(positive) >= 1
        assert len(negative) >= 1

    def test_get_links_for_memory(self, nac, valence_positive):
        """Query links by memory ID."""
        nac.observe(
            event_type="tool",
            event_signature="memory_linked",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
            memory_id="mem_123",
        )

        links = nac.get_links_for("mem_123")
        assert len(links) == 1


class TestNAcMaintenance:
    """Test maintenance operations."""

    def test_enforce_limits_removes_low_confidence_links(self, valence_positive):
        """Enforces max_links by removing lowest confidence."""
        from maxim.decisions.nac import NAc, NACConfig

        small_nac = NAc(config=NACConfig(max_links=5))

        # Create more links than allowed
        for i in range(10):
            small_nac.observe(
                event_type="tool",
                event_signature=f"tool_{i}",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        # Should have enforced limit
        assert len(small_nac) <= 5

    def test_decay_reduces_confidence(self, nac, valence_positive):
        """Decay reduces link confidence."""
        link = nac.observe(
            event_type="tool",
            event_signature="decaying",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )

        initial_confidence = link.confidence
        nac.decay_all(factor=0.5)

        assert link.confidence == initial_confidence * 0.5


class TestNAcPersistence:
    """Test save/load roundtrip."""

    def test_save_load_preserves_links(self, nac, valence_positive, tmp_path):
        """Links survive save/load cycle."""
        # Create some links
        nac.observe(
            event_type="tool",
            event_signature="persistent_tool",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=2.5,
        )

        path = tmp_path / "nac.json"
        nac.save(str(path))

        # Load into fresh instance
        from maxim.decisions.nac import NAc

        nac2 = NAc()
        nac2.load(str(path))

        assert len(nac2) == len(nac)
        links = nac2.get_links_for_event("persistent_tool")
        assert len(links) == 1
        assert links[0].temporal_delta.mean == 2.5

    def test_save_load_preserves_stats(self, nac, valence_positive, tmp_path):
        """Statistics survive save/load cycle."""
        for i in range(5):
            nac.observe(
                event_type="tool",
                event_signature=f"tool_{i}",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        path = tmp_path / "nac.json"
        nac.save(str(path))

        from maxim.decisions.nac import NAc

        nac2 = NAc()
        nac2.load(str(path))

        assert nac2.stats()["total_observations"] == nac.stats()["total_observations"]


class TestNAcStats:
    """Test statistics collection."""

    def test_stats_returns_expected_keys(self, nac, valence_positive):
        """Stats returns all expected keys."""
        nac.observe(
            event_type="tool",
            event_signature="test",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )

        stats = nac.stats()

        assert "total_links" in stats
        assert "event_signatures" in stats
        assert "total_observations" in stats
        assert "pending_events" in stats

    def test_len_returns_total_links(self, nac, valence_positive):
        """__len__ returns total link count."""
        assert len(nac) == 0

        nac.observe(
            event_type="tool",
            event_signature="test1",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=valence_positive,
            delta_seconds=1.0,
        )

        assert len(nac) == 1