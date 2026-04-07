"""Tests for Benchmark Phases 6-9 — baseline, write-paper, transcriber, Tier 3."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock


from maxim.simulation.benchmark import BenchmarkReport, BenchmarkRunner, ModelResult
from maxim.simulation.narrative_transcriber import NarrativeTranscriber, _get_class_id


# ── Phase 7: write_paper smoke test ──────────────────────────────────


class TestWritePaper:
    def test_write_paper_returns_none_without_llm(self):
        """write_paper returns None gracefully when no LLM is available."""
        yaml_content = "name: test\npercepts: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            runner = BenchmarkRunner(models=["test"], suite_path=f.name)

        report = BenchmarkReport(
            timestamp="test",
            suite="test",
            models=["test"],
            runs_per_model=1,
            results={
                "test": ModelResult(model="test", metrics={"memory_formation_rate": 2.0}, score=0.6),
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should return None (no LLM available in test env)
            result = runner.write_paper(report, Path(tmpdir))
            # Either None (no LLM) or a Path (if LLM happens to be available)
            assert result is None or isinstance(result, Path)


# ── Phase 8: Narrative Transcriber ───────────────────────────────────


class TestNarrativeTranscriber:
    def test_regex_extracts_proper_nouns(self):
        transcriber = NarrativeTranscriber()
        detections = transcriber.transcribe("Elara grabs your arm. Kael is at his forge.")
        labels = {d["label"] for d in detections}
        assert "elara" in labels
        assert "kael" in labels

    def test_regex_extracts_quoted_speech(self):
        transcriber = NarrativeTranscriber()
        detections = transcriber.transcribe('She whispers: "Say the name Verath at the door."')
        # Should have at least the proper noun + speech
        assert len(detections) >= 1

    def test_stable_track_ids(self):
        """Same entity gets same track_id across calls."""
        transcriber = NarrativeTranscriber()
        d1 = transcriber.transcribe("Elara is here.")
        d2 = transcriber.transcribe("Elara returns.")
        elara_ids = [d["track_id"] for ds in [d1, d2] for d in ds if d["label"] == "elara"]
        assert len(elara_ids) == 2
        assert elara_ids[0] == elara_ids[1]

    def test_detections_have_required_fields(self):
        """Each detection has track_id, class_id, label, conf, bbox_xyxy."""
        transcriber = NarrativeTranscriber()
        detections = transcriber.transcribe("A silver elm stands in the forest. Kael waves.")
        for d in detections:
            assert "track_id" in d
            assert "class_id" in d
            assert "label" in d
            assert "conf" in d
            assert "bbox_xyxy" in d
            assert isinstance(d["class_id"], int)
            assert d["class_id"] >= 900  # Narrative classes start at 900

    def test_empty_text_returns_empty(self):
        transcriber = NarrativeTranscriber()
        assert transcriber.transcribe("") == []
        assert transcriber.transcribe("   ") == []

    def test_reset_clears_entity_ids(self):
        transcriber = NarrativeTranscriber()
        transcriber.transcribe("Elara is here.")
        assert len(transcriber._entity_ids) > 0
        transcriber.reset()
        assert len(transcriber._entity_ids) == 0

    def test_class_ids_are_stable(self):
        """Same label always gets same class_id."""
        id1 = _get_class_id("silver_elm")
        id2 = _get_class_id("silver_elm")
        assert id1 == id2
        # Different label gets different ID
        id3 = _get_class_id("stone_door")
        assert id3 != id1

    def test_filters_common_words(self):
        """Doesn't extract 'The', 'You', 'What' as entities."""
        transcriber = NarrativeTranscriber()
        detections = transcriber.transcribe("The door. You see. What happened.")
        labels = {d["label"] for d in detections}
        assert "the" not in labels
        assert "you" not in labels
        assert "what" not in labels

    def test_deduplicates_entities(self):
        """Same entity mentioned twice in one text appears once."""
        transcriber = NarrativeTranscriber()
        detections = transcriber.transcribe("Elara speaks. Then Elara leaves.")
        elara_count = sum(1 for d in detections if d["label"] == "elara")
        assert elara_count == 1

    def test_llm_fallback_on_error(self):
        """Falls back to regex when LLM router raises."""
        mock_router = MagicMock()
        mock_router.generate_json.side_effect = RuntimeError("no model")
        transcriber = NarrativeTranscriber(router=mock_router)
        # Should still work via regex fallback
        detections = transcriber.transcribe("Kael the blacksmith hammers a blade.")
        assert len(detections) >= 1
        labels = {d["label"] for d in detections}
        assert "kael" in labels


# ── Phase 9: Tier 3 embodiment hooks ─────────────────────────────────


class TestTier3Hooks:
    def test_collect_tier3_no_introspector(self):
        yaml_content = "name: test\npercepts: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            runner = BenchmarkRunner(models=["test"], suite_path=f.name)

        result = MagicMock()
        result.introspector = None
        assert runner._collect_tier3_metrics(result) == {}

    def test_collect_tier3_no_embodiment_stats(self):
        yaml_content = "name: test\npercepts: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            runner = BenchmarkRunner(models=["test"], suite_path=f.name)

        result = MagicMock()
        result.introspector = MagicMock(spec=[])  # No embodiment_stats method
        assert runner._collect_tier3_metrics(result) == {}

    def test_collect_tier3_with_embodiment(self):
        yaml_content = "name: test\npercepts: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            runner = BenchmarkRunner(models=["test"], suite_path=f.name)

        result = MagicMock()
        result.introspector = MagicMock()
        result.introspector.embodiment_stats.return_value = {
            "pain_sigma": 0.15,
            "forward_model_mae": 2.3,
        }
        metrics = runner._collect_tier3_metrics(result)
        assert metrics["pain_sigma"] == 0.15
        assert metrics["forward_model_mae"] == 2.3


# ── Bug fix: load_campaign_turns format ──────────────────────────────


class TestLoadCampaignTurnsFormat:
    def test_turns_have_text_key(self):
        """Turns must have 'text' key matching orchestrator's expectation."""
        from maxim.simulation.experiment import load_campaign_turns

        yaml_content = """
percepts:
  - at: 0
    cli_input: "Hello world"
    salience: 0.9
    novelty: 0.8
    metadata:
      phase: "intro"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            turns = load_campaign_turns(f.name)

        assert len(turns) == 1
        assert turns[0]["text"] == "Hello world"
        assert turns[0]["cli_input"] == "Hello world"  # backward compat
        assert turns[0]["phase"] == "intro"
        assert turns[0]["salience"] == 0.9
        assert turns[0]["novelty"] == 0.8
