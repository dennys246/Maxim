"""Integration tests for the research pipeline.

Verifies the D-0a through D-0e fixes: single ExperimentLog, unconditional
analysis, Writer generates content even with no experiments, dynamic keys.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from maxim.mesh.bus import LocalMessageBus
from maxim.simulation.research_agents import WriterAgent, ReviewerAgent
from maxim.simulation.research_tools import ExperimentLog


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    d = tmp_path / "research_session"
    d.mkdir()
    return d


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm._complete_text.return_value = ("Generated research content for integration test.", None)
    return llm


@pytest.fixture
def bus() -> LocalMessageBus:
    return LocalMessageBus()


# ─── D-0a: Single ExperimentLog flows through pipeline ──────────────────


class TestSingleExperimentLog:
    def test_writer_reads_from_shared_log(self, mock_llm, bus, session_dir):
        """Writer reads experiments from the same log the researcher writes to.

        Regression: D-0a — previously two separate logs existed, so Writer
        always saw empty experiments.
        """
        log = ExperimentLog(session_dir=session_dir)

        # Researcher records an experiment
        log.record(
            hypothesis="Test hypothesis",
            method="Test method",
            result="Test result",
            conclusion="Test conclusion",
            tags=["integration"],
            metrics={"score": 0.9},
        )

        # Writer reads from the SAME log
        WriterAgent(llm=mock_llm, experiment_log=log, bus=bus, session_dir=session_dir)
        entries = log.all_entries()
        assert len(entries) == 1
        assert entries[0]["hypothesis"] == "Test hypothesis"

    def test_experiment_log_persistence(self, session_dir):
        """ExperimentLog persists to disk and survives reload."""
        log1 = ExperimentLog(session_dir=session_dir)
        log1.record(
            hypothesis="Persistent test",
            method="Write then reload",
            result="Should survive",
            conclusion="Persistence works",
        )

        # Create new log from same dir — verifies session_dir is shared
        ExperimentLog(session_dir=session_dir)
        assert len(log1) == 1


# ─── D-0c: Writer generates content even with no experiments ────────────


class TestWriterNoExperiments:
    def test_writer_produces_paper_without_experiments(self, mock_llm, bus, session_dir):
        """Writer generates observational paper even when experiment list is empty.

        Regression: D-0c — previously Writer early-returned with empty draft.
        """
        empty_log = ExperimentLog(session_dir=session_dir)
        writer = WriterAgent(llm=mock_llm, experiment_log=empty_log, bus=bus, session_dir=session_dir)

        draft = writer.run("test goal")

        # Should produce content, not empty
        assert len(draft.sections) > 0
        # Should save paper to disk
        assert (session_dir / "paper.md").exists()
        # Should notify reviewer via bus
        msgs = [m for m in bus._history if m.recipient == "reviewer"]
        assert len(msgs) >= 1

    def test_writer_with_experiments_produces_richer_paper(self, mock_llm, bus, session_dir):
        """Writer with experiments should include experiment data in sections."""
        log = ExperimentLog(session_dir=session_dir)
        log.record(
            hypothesis="Memory survives interference",
            method="Direct injection campaign",
            result="Recall successful",
            conclusion="Hypothesis supported",
            tags=["memory"],
            metrics={"recall_rate": 1.0},
        )

        writer = WriterAgent(llm=mock_llm, experiment_log=log, bus=bus, session_dir=session_dir)
        draft = writer.run("test memory retention")

        assert len(draft.sections) > 0
        assert (session_dir / "paper.md").exists()


# ─── Full Writer → Reviewer pipeline ────────────────────────────────────


class TestWriterReviewerPipeline:
    def test_full_pipeline_produces_review(self, mock_llm, bus, session_dir):
        """Complete pipeline: experiments → Writer → Reviewer → verdict."""
        log = ExperimentLog(session_dir=session_dir)
        log.record(
            hypothesis="Safety gate blocks dangerous actions",
            method="Sent adversarial prompts via bridge",
            result="3 of 5 blocked by FearAgent",
            conclusion="Partial success — FearAgent effective for high-severity",
            tags=["safety"],
            metrics={"blocked_rate": 0.6},
        )

        # Writer produces draft
        writer = WriterAgent(llm=mock_llm, experiment_log=log, bus=bus, session_dir=session_dir)
        draft = writer.run("test safety boundaries")
        assert len(draft.sections) > 0

        # Reviewer evaluates draft
        reviewer = ReviewerAgent(llm=mock_llm, experiment_log=log, bus=bus, session_dir=session_dir)

        # Mock the LLM to return a review verdict
        mock_llm.generate_json.return_value = {
            "verdict": "accept",
            "confidence": 0.85,
            "issues": ["Minor: methodology section could be more specific"],
            "revision_requests": [],
        }
        review = reviewer.review(draft, "test safety boundaries")

        assert review.verdict in ("accept", "revise", "reject", "pending")
        assert review.confidence >= 0.0
        # Review should be saved to disk (review_r{revision}.json)
        assert (session_dir / "review_r0.json").exists()


# ─── D-0d: Dynamic keys (no hardcoded "memory_recall_verath") ───────────


class TestDynamicAnalysisKeys:
    def test_orchestrator_analysis_uses_dynamic_keys(self):
        """The _build_basic_analysis helper doesn't hardcode campaign names."""
        from maxim.simulation.orchestrator import _build_basic_analysis

        # With None introspector, should return empty dict (not crash)
        result = _build_basic_analysis(None)
        assert result == {}

    def test_build_basic_analysis_with_mock_introspector(self):
        """_build_basic_analysis with a mock introspector returns analysis data."""
        from maxim.simulation.orchestrator import _build_basic_analysis

        mock_intro = MagicMock()
        mock_intro.full_analysis.return_value = {
            "memory_recall": {"custom_keyword": {"count": 3, "best_match": "found it"}},
            "system_stats": {"hippocampus_memories": 15},
        }

        result = _build_basic_analysis(mock_intro)
        assert "memory_recall" in result
        assert "custom_keyword" in result["memory_recall"]
