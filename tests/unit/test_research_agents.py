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
        # D-0c fix: Writer now generates observational paper even with no experiments
        assert len(draft.sections) > 0  # Should still produce content
        assert (session_dir / "paper.md").exists()  # Paper saved to disk

    def test_run_without_llm(self, experiment_log, bus, session_dir):
        writer = WriterAgent(llm=None, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        draft = writer.run("test")
        # No LLM, no section: no paper saved, announced or sent to the reviewer (#1680)
        assert draft.sections == {} and draft.output_path is None
        assert not (session_dir / "paper.md").exists()
        assert bus.get_history(msg_type=MeshMessageType.PAPER_DRAFT) == []

    def test_no_text_from_any_section_is_no_paper(self, experiment_log, bus, session_dir):
        """The real router returns "" on a failed dispatch; a draft with no section must not read as the paper."""
        llm = MagicMock()
        llm._complete_text.return_value = ("", None)
        writer = WriterAgent(llm=llm, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        draft = writer.run("test")
        assert draft.output_path is None and not (session_dir / "paper.md").exists()
        assert bus.get_history(msg_type=MeshMessageType.PAPER_DRAFT) == []
        writer.revise(ReviewResult(verdict="revise", section_feedback={"methods": "more"}))
        assert writer.draft.output_path is None and not (session_dir / "paper.md").exists()

    def test_a_partial_draft_is_saved_and_names_its_missing_sections(self, experiment_log, bus, session_dir, caplog):
        llm = MagicMock()
        llm._complete_text.side_effect = [("Methods text.", None)] + [("", None)] * (len(PAPER_SECTIONS) - 1)
        writer = WriterAgent(llm=llm, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        with caplog.at_level("WARNING", logger="maxim.simulation.research_agents"):
            draft = writer.run("test")
        assert draft.output_path == session_dir / "paper.md" and draft.output_path.exists()
        assert "sections missing" in caplog.text and "'results'" in caplog.text

    def test_a_backend_error_is_no_section_but_a_code_defect_raises(self, experiment_log, bus, session_dir):
        from maxim.models.language.types import BackendError

        llm = MagicMock()
        llm._complete_text.side_effect = BackendError("server 500")
        writer = WriterAgent(llm=llm, experiment_log=experiment_log, bus=bus, session_dir=session_dir)
        assert writer.run("test").output_path is None
        llm._complete_text.side_effect = TypeError("unexpected keyword argument")
        with pytest.raises(TypeError):
            writer.run("test")

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
        assert r.finish_reason == ""

    def test_runtime_abort_skips_writer_and_persists_terminal_status(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from maxim.simulation.research_orchestrator import start_research_mode

        from maxim.utils.paths import _reset_caches

        monkeypatch.chdir(tmp_path)
        # The session directory lives under the data home, not ./data/ (#932).
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "home"))
        _reset_caches()
        monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", lambda **kwargs: (None, None))
        monkeypatch.setattr(
            "maxim.models.language.router.load_llm_config",
            lambda: SimpleNamespace(enabled=False),
        )
        monkeypatch.setattr(
            "maxim.simulation.orchestrator.start_simulation_mode",
            lambda **kwargs: SimpleNamespace(
                finish_reason="planning_failed",
                campaign_analysis={},
                duration_s=0.1,
                turns=1,
                total_actions=0,
            ),
        )
        monkeypatch.setattr(
            "maxim.simulation.research_agents.WriterAgent.run",
            lambda *args, **kwargs: pytest.fail("writer must not run on unusable experiment data"),
        )

        result = start_research_mode(goal="abort contract")

        assert result.finish_reason == "planning_failed"
        assert result.review_verdict == "aborted"
        payload = json.loads(
            (tmp_path / "home" / "sim_reports" / f"research_{result.session_id}" / "research_result.json").read_text()
        )
        assert payload["finish_reason"] == "planning_failed"
        assert payload["review_verdict"] == "aborted"


class TestResearchProtocolWithNoPaper:
    """``start_research_mode`` end to end (only the sim and the LLM stubbed): when the Writer saves no paper, nothing
    is reviewed, and the run reports ``no_paper`` with its cause, never an accept beside ``Paper: None`` (#1680)."""

    def _run(self, tmp_path, monkeypatch, llm):
        from types import SimpleNamespace

        from maxim.simulation.research_orchestrator import start_research_mode
        from maxim.utils.paths import _reset_caches

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "home"))
        _reset_caches()
        monkeypatch.setattr("maxim.simulation.research_orchestrator._build_research_llm", lambda *a, **k: llm)
        monkeypatch.setattr(
            "maxim.simulation.orchestrator.start_simulation_mode",
            lambda **kwargs: SimpleNamespace(
                finish_reason="completed", campaign_analysis={}, duration_s=0.1, turns=1, total_actions=1
            ),
        )
        result = start_research_mode(goal="no paper contract")
        session = tmp_path / "home" / "sim_reports" / f"research_{result.session_id}"
        return result, session

    @pytest.mark.parametrize("has_llm", [False, True], ids=["no-llm", "llm-returns-no-text"])
    def test_no_paper_is_not_reviewed_and_says_why(self, tmp_path, monkeypatch, has_llm):
        from types import SimpleNamespace

        from maxim.cli import _research_result_exit_code

        llm = None
        if has_llm:
            llm = MagicMock(session_cost=0.0)
            llm._complete_text.return_value = ("", None)  # what the real router returns on a failed call
        result, session = self._run(tmp_path, monkeypatch, llm)

        assert result.review_verdict == "no_paper" and result.paper_path == "" and result.revision_rounds == 0
        assert not (session / "paper.md").exists() and not list(session.glob("review_r*.json"))
        payload = json.loads((session / "research_result.json").read_text())
        assert payload["review_verdict"] == "no_paper"
        assert payload["review_issues"] == [
            "the LLM returned no text for any paper section"
            if has_llm
            else "no research LLM, so the Writer wrote no section"
        ]
        if llm is not None:
            llm.generate_json.assert_not_called()  # the Reviewer's call
        assert _research_result_exit_code(SimpleNamespace(finish_reason="completed", review_verdict="no_paper")) == 1


class TestReviewerFailures:
    """The Reviewer reads a failed call (``BackendError``) as no feedback; anything else propagates (#1680)."""

    def _reviewer(self, experiment_log, bus, session_dir, error):
        llm = MagicMock()
        llm.generate_json.side_effect = error
        return ReviewerAgent(llm=llm, experiment_log=experiment_log, bus=bus, session_dir=session_dir)

    def test_a_backend_error_is_no_feedback(self, experiment_log, bus, session_dir):
        from maxim.models.language.types import BackendError

        reviewer = self._reviewer(experiment_log, bus, session_dir, BackendError("server 500"))
        draft = PaperDraft(sections={"methods": "m"})
        review = reviewer.review(draft, "g")
        assert review.section_feedback == {} and review.verdict == "revise"

    @pytest.mark.parametrize("site", ["section", "verdict"])
    def test_a_code_defect_or_a_leader_refusal_raises(self, experiment_log, bus, session_dir, site):
        from maxim.exceptions import LeaderRefusal

        reviewer = self._reviewer(experiment_log, bus, session_dir, None)
        if site == "section":
            reviewer.llm.generate_json.side_effect = TypeError("unexpected keyword argument")
            with pytest.raises(TypeError):
                reviewer._review_section("methods", "m", "g", [])
        else:
            reviewer.llm.generate_json.side_effect = LeaderRefusal("no key")
            with pytest.raises(LeaderRefusal):
                reviewer._determine_verdict(ReviewResult(), "g", [])

    def test_a_confidence_that_is_not_a_number_is_logged_not_raised(self, experiment_log, bus, session_dir):
        reviewer = self._reviewer(experiment_log, bus, session_dir, None)
        reviewer.llm.generate_json.side_effect = None
        reviewer.llm.generate_json.return_value = {"verdict": "accept", "confidence": "high"}
        assert reviewer._determine_verdict(ReviewResult(), "g", []) == {
            "verdict": "accept",
            "confidence": 0.5,
            "revision_requests": [],
        }
