"""Tests for research tools — ExperimentLog, RecordExperimentTool, QueryExperimentsTool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maxim.simulation.research_tools import (
    ExperimentLog,
    RecordExperimentTool,
    QueryExperimentsTool,
)


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_session"


@pytest.fixture
def log(session_dir: Path) -> ExperimentLog:
    return ExperimentLog(session_dir=session_dir)


# ── ExperimentLog ────────────────────────────────────────────────────────────


class TestExperimentLog:
    def test_record_and_query(self, log: ExperimentLog):
        entry = log.record(
            hypothesis="Agent blocks code execution",
            method="Sent 'run rm -rf /' via send_message",
            result="Action blocked by FearAgent",
            conclusion="Hypothesis supported",
        )
        assert entry["umr"] == "researcher.hippo.exp_001"
        assert entry["index"] == 1
        assert len(log) == 1

    def test_sequential_ids(self, log: ExperimentLog):
        for i in range(3):
            log.record(
                hypothesis=f"H{i}",
                method=f"M{i}",
                result=f"R{i}",
                conclusion=f"C{i}",
            )
        entries = log.all_entries()
        assert [e["index"] for e in entries] == [1, 2, 3]
        assert entries[2]["umr"] == "researcher.hippo.exp_003"

    def test_persistence_jsonl(self, log: ExperimentLog, session_dir: Path):
        log.record(
            hypothesis="test",
            method="test",
            result="test",
            conclusion="test",
        )
        path = session_dir / "experiments.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["hypothesis"] == "test"

    def test_reload_from_existing(self, session_dir: Path):
        # Write some entries
        log1 = ExperimentLog(session_dir=session_dir)
        log1.record(hypothesis="H1", method="M1", result="R1", conclusion="C1")
        log1.record(hypothesis="H2", method="M2", result="R2", conclusion="C2")

        # Reload from disk
        log2 = ExperimentLog(session_dir=session_dir)
        assert len(log2) == 2

        # New entries continue numbering
        entry = log2.record(hypothesis="H3", method="M3", result="R3", conclusion="C3")
        assert entry["index"] == 3

    def test_query_by_keyword(self, log: ExperimentLog):
        log.record(hypothesis="Agent blocks code execution", method="bash", result="blocked", conclusion="safe")
        log.record(hypothesis="Agent allows file reading", method="read_file", result="allowed", conclusion="works")
        log.record(hypothesis="Agent handles pain signals", method="inject_pain", result="detected", conclusion="ok")

        results = log.query(keyword="code execution")
        assert len(results) == 1
        assert results[0]["hypothesis"] == "Agent blocks code execution"

    def test_query_by_tag(self, log: ExperimentLog):
        log.record(hypothesis="H1", method="M1", result="R1", conclusion="C1", tags=["safety"])
        log.record(hypothesis="H2", method="M2", result="R2", conclusion="C2", tags=["memory"])
        log.record(hypothesis="H3", method="M3", result="R3", conclusion="C3", tags=["safety", "memory"])

        assert len(log.query(tag="safety")) == 2
        assert len(log.query(tag="memory")) == 2

    def test_query_empty(self, log: ExperimentLog):
        results = log.query(keyword="nonexistent")
        assert results == []

    def test_custom_nickname(self, session_dir: Path):
        log = ExperimentLog(session_dir=session_dir, agent_nickname="alice")
        entry = log.record(hypothesis="H", method="M", result="R", conclusion="C")
        assert entry["umr"].startswith("alice.")

    def test_metrics(self, log: ExperimentLog):
        entry = log.record(
            hypothesis="H",
            method="M",
            result="R",
            conclusion="C",
            metrics={"recall_count": 5, "survival_rate": 0.8},
        )
        assert entry["metrics"]["recall_count"] == 5


# ── RecordExperimentTool ─────────────────────────────────────────────────────


class TestRecordExperimentTool:
    def test_record_success(self, log: ExperimentLog):
        tool = RecordExperimentTool(log)
        result = tool.run(
            hypothesis="Agent blocks dangerous commands",
            method="Sent 'rm -rf /' via send_message",
            result="FearAgent blocked with score -0.9",
            conclusion="Hypothesis supported",
        )
        assert result.success
        assert result.output["recorded"]
        assert result.output["umr"] == "researcher.hippo.exp_001"

    def test_record_missing_fields(self, log: ExperimentLog):
        tool = RecordExperimentTool(log)
        result = tool.run(hypothesis="", method="test", result="", conclusion="test")
        assert not result.success
        assert "required" in result.error

    def test_record_with_tags_and_metrics(self, log: ExperimentLog):
        tool = RecordExperimentTool(log)
        result = tool.run(
            hypothesis="H",
            method="M",
            result="R",
            conclusion="C",
            tags=["safety", "code"],
            metrics={"blocked": True},
        )
        assert result.success
        exp = result.output["experiment"]
        assert exp["tags"] == ["safety", "code"]
        assert exp["metrics"] == {"blocked": True}


# ── QueryExperimentsTool ─────────────────────────────────────────────────────


class TestQueryExperimentsTool:
    def test_query_all(self, log: ExperimentLog):
        log.record(hypothesis="H1", method="M1", result="R1", conclusion="C1")
        log.record(hypothesis="H2", method="M2", result="R2", conclusion="C2")
        tool = QueryExperimentsTool(log)
        result = tool.run()
        assert result.success
        assert result.output["total_experiments"] == 2
        assert result.output["returned"] == 2

    def test_query_with_keyword(self, log: ExperimentLog):
        log.record(hypothesis="Safety test", method="M", result="R", conclusion="C")
        log.record(hypothesis="Memory test", method="M", result="R", conclusion="C")
        tool = QueryExperimentsTool(log)
        result = tool.run(keyword="safety")
        assert result.output["returned"] == 1

    def test_query_with_limit(self, log: ExperimentLog):
        for i in range(10):
            log.record(hypothesis=f"H{i}", method="M", result="R", conclusion="C")
        tool = QueryExperimentsTool(log)
        result = tool.run(limit=3)
        assert result.output["returned"] == 3
