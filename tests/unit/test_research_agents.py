"""Tests for research agents — WriterAgent, ReviewerAgent, PaperDraft, ReviewResult."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from maxim.mesh.bus import LocalMessageBus
from maxim.mesh.message import MeshMessageType
from maxim.simulation.research_agents import (
    PaperDraft,
    ReviewResult,
    ReviewerAgent,
    WriterAgent,
    PAPER_SECTIONS,
)
from maxim.simulation.research_tools import ExperimentLog


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    d = tmp_path / "test_research_session"
    d.mkdir()
    return d


@pytest.fixture
def experiment_log(session_dir: Path) -> ExperimentLog:
    log = ExperimentLog(session_dir=session_dir)
    log.record(
        hypothesis="Agent blocks code execution",
        method="Sent 'rm -rf /' via send_message",
        result="Action blocked by FearAgent with score -0.9",
        conclusion="Hypothesis supported — safety gate effective",
        tags=["safety"],
        metrics={"blocked": True, "score": -0.9},
    )
    log.record(
        hypothesis="Agent recalls seeded detail after 3 interference turns",
        method="Ran hippocampal_recall_short campaign",
        result="AUT said 'Verath' to the door — full recall",
        conclusion="Associative recall bridged interference gap",
        tags=["memory", "recall"],
        metrics={"recall_grade": "FULL_RECALL", "activation_score": 0.18},
    )
    return log


@pytest.fixture
def bus() -> LocalMessageBus:
    return LocalMessageBus()


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate_json.return_value = {"content": "Generated section content for testing."}
    llm._complete_text.return_value = ("Generated section content for testing.", None)
    return llm


# ── PaperDraft ───────────────────────────────────────────────────────────────


class TestPaperDraft:
    def test_write_and_read_section(self):
        draft = PaperDraft()
        draft.write_section("methods", "We tested safety.")
        assert draft.read_section("methods") == "We tested safety."

    def test_case_insensitive(self):
        draft = PaperDraft()
        draft.write_section("METHODS", "content")
        assert draft.read_section("methods") == "content"

    def test_to_markdown_ordering(self):
        draft = PaperDraft()
        draft.write_section("conclusions", "We conclude X.")
        draft.write_section("methods", "We did Y.")
        draft.write_section("results", "We found Z.")
        md = draft.to_markdown()
        # Methods should appear before results, results before conclusions
        assert md.index("Methods") < md.index("Results") < md.index("Conclusions")

    def test_save(self, tmp_path: Path):
        draft = PaperDraft()
        draft.write_section("methods", "Test content")
        path = tmp_path / "paper.md"
        draft.save(path)
        assert path.exists()
        assert "Test content" in path.read_text()
        assert draft.output_path == path

    def test_read_missing_section_returns_none(self):
        draft = PaperDraft()
        assert draft.read_section("nonexistent") is None

    def test_title_abstract_rendered_first(self):
        draft = PaperDraft()
        draft.write_section("title_abstract", "# My Paper\nAbstract here.")
        draft.write_section("methods", "We did things.")
        md = draft.to_markdown()
        assert md.startswith("# My Paper")


# ── ReviewResult ─────────────────────────────────────────────────────────────


class TestReviewResult:
    def test_to_dict(self):
        r = ReviewResult(
            verdict="revise",
            confidence=0.7,
            issues=["Missing data"],
            revision_requests=["Add more experiments"],
        )
        d = r.to_dict()
        assert d["verdict"] == "revise"
        assert d["confidence"] == 0.7
        assert len(d["issues"]) == 1

    def test_default_pending(self):
        r = ReviewResult()
        assert r.verdict == "pending"


# ── WriterAgent ──────────────────────────────────────────────────────────────


class TestWriterAgent:
    def test_run_produces_all_sections(self, mock_llm, experiment_log, bus, session_dir):
        writer = WriterAgent(
            llm=mock_llm,
            experiment_log=experiment_log,
            bus=bus,
            session_dir=session_dir,
        )
        draft = writer.run("test safety")
        # Should have attempted all sections
        assert mock_llm._complete_text.call_count == len(PAPER_SECTIONS)
        assert len(draft.sections) == len(PAPER_SECTIONS)

    def test_run_saves_paper(self, mock_llm, experiment_log, bus, session_dir):
        writer = WriterAgent(
            llm=mock_llm,
            experiment_log=experiment_log,
            bus=bus,
            session_dir=session_dir,
        )
        draft = writer.run("test safety")
        assert draft.output_path is not None
        assert draft.output_path.exists()

    def test_run_sends_bus_message(self, mock_llm, experiment_log, bus, session_dir):
        writer = WriterAgent(
            llm=mock_llm,
            experiment_log=experiment_log,
            bus=bus,
            session_dir=session_dir,
        )
        writer.run("test safety")
        msgs = bus.get_history(msg_type=MeshMessageType.PAPER_DRAFT)
        assert len(msgs) == 1
        assert msgs[0].sender == "writer"
        assert msgs[0].recipient == "reviewer"

    def test_run_with_no_experiments(self, mock_llm, bus, session_dir):
        empty_log = ExperimentLog(session_dir=session_dir)
        writer = WriterAgent(llm=mock_llm, experiment_log=empty_log, bus=bus, session_dir=session_dir)
        draft = writer.run("test")
        assert len(draft.sections) == 0  # No experiments = no paper

    def test_run_without_llm(self, experiment_log, bus, session_dir):
        writer = WriterAgent(llm=None, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        draft = writer.run("test")
        # Sections should be empty strings (no LLM to generate)
        assert all(s == "" for s in draft.sections.values()) or len(draft.sections) == 0

    def test_revise_increments_revision(self, mock_llm, experiment_log, bus, session_dir):
        writer = WriterAgent(llm=mock_llm, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        writer.run("test")
        assert writer.draft.revision == 0
        feedback = ReviewResult(
            verdict="revise",
            section_feedback={"methods": "Add more detail"},
        )
        writer.revise(feedback)
        assert writer.draft.revision == 1

    def test_profile(self, mock_llm, experiment_log, bus, session_dir):
        writer = WriterAgent(llm=mock_llm, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        assert writer.profile.nickname == "writer"
        assert writer.profile.role == "writer"


# ── ReviewerAgent ────────────────────────────────────────────────────────────


class TestReviewerAgent:
    def test_review_produces_verdict(self, mock_llm, experiment_log, bus, session_dir):
        mock_llm.generate_json.side_effect = [
            {"feedback": "Methods look good."},  # methods review
            {"feedback": "Results are clear."},  # results review
            {"feedback": "No issues."},  # other sections...
            {"feedback": "No issues."},
            {"feedback": "No issues."},
            {"feedback": "No issues."},
            {"feedback": "No issues."},
            {"feedback": "No issues."},
            {"verdict": "accept", "confidence": 0.85, "revision_requests": []},  # verdict
        ]

        draft = PaperDraft()
        draft.write_section("methods", "We tested safety.")
        draft.write_section("results", "All tests passed.")

        reviewer = ReviewerAgent(llm=mock_llm, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        result = reviewer.review(draft, "test safety")
        assert result.verdict in ("accept", "revise", "reject")
        assert 0 <= result.confidence <= 1

    def test_review_saves_json(self, mock_llm, experiment_log, bus, session_dir):
        mock_llm.generate_json.return_value = {"verdict": "accept", "confidence": 0.9, "revision_requests": []}

        draft = PaperDraft()
        draft.write_section("methods", "content")
        reviewer = ReviewerAgent(llm=mock_llm, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        reviewer.review(draft, "test")

        review_file = session_dir / "review_r0.json"
        assert review_file.exists()
        data = json.loads(review_file.read_text())
        assert "verdict" in data

    def test_review_sends_bus_message(self, mock_llm, experiment_log, bus, session_dir):
        mock_llm.generate_json.return_value = {"verdict": "accept", "confidence": 0.8, "revision_requests": []}

        draft = PaperDraft()
        draft.write_section("methods", "content")
        reviewer = ReviewerAgent(llm=mock_llm, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        reviewer.review(draft, "test")

        msgs = bus.get_history(msg_type=MeshMessageType.REVIEW_RESULT)
        assert len(msgs) == 1
        assert msgs[0].sender == "reviewer"

    def test_review_without_llm_uses_heuristics(self, experiment_log, bus, session_dir):
        draft = PaperDraft()
        draft.write_section("methods", "content")
        draft.write_section("results", "results")
        reviewer = ReviewerAgent(llm=None, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        result = reviewer.review(draft, "test")
        # Heuristic fallback should produce a verdict
        assert result.verdict in ("accept", "revise", "reject")

    def test_data_consistency_checks(self, experiment_log, bus, session_dir):
        draft = PaperDraft()
        draft.write_section("results", "We found things.")
        # No methods section — should flag
        reviewer = ReviewerAgent(llm=None, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        result = reviewer.review(draft, "test")
        assert any("Methods" in issue for issue in result.issues)

    def test_profile(self, mock_llm, experiment_log, bus, session_dir):
        reviewer = ReviewerAgent(llm=mock_llm, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        assert reviewer.profile.nickname == "reviewer"
        assert reviewer.profile.role == "reviewer"


# ── ResearchResult ───────────────────────────────────────────────────────────


class TestResearchResult:
    def test_creation(self):
        from maxim.simulation.research_orchestrator import ResearchResult

        r = ResearchResult(goal="test hippocampal recall")
        assert r.goal == "test hippocampal recall"
        assert r.review_verdict == "not_reviewed"
        assert r.revision_rounds == 0
