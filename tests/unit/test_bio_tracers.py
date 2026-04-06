"""Tests for bio-system tracers — NacTracer, ATLTracer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from maxim.decisions.nac_tracer import NacTracer, _short
from maxim.memory.atl_tracer import ATLTracer


# ── NacTracer ────────────────────────────────────────────────────────────────


class TestNacTracerHelpers:
    def test_short_truncates(self):
        assert _short("a" * 50, 20) == "a" * 17 + "..."

    def test_short_preserves(self):
        assert _short("hello", 20) == "hello"


class TestNacTracer:
    @pytest.fixture
    def mock_nac(self):
        nac = MagicMock()
        nac.record_event.return_value = "evt_001"
        nac.record_outcome.return_value = None
        nac.observe.return_value = MagicMock(confidence=0.8, observation_count=5)
        nac.predict.return_value = MagicMock(predicted_value=0.7, confidence=0.6)
        nac.stats.return_value = {"total_links": 3}
        return nac

    def test_wraps_record_event(self, mock_nac, capsys):
        NacTracer(mock_nac)
        mock_nac.record_event("tool", "bash:rm")
        captured = capsys.readouterr()
        assert "NAc EVENT" in captured.err
        assert "bash:rm" in captured.err

    def test_wraps_observe(self, mock_nac, capsys):
        NacTracer(mock_nac)
        mock_nac.observe("tool", "bash", "outcome", "success", "POSITIVE", 1.0)
        captured = capsys.readouterr()
        assert "NAc LEARN" in captured.err
        assert "bash" in captured.err

    def test_wraps_predict(self, mock_nac, capsys):
        NacTracer(mock_nac)
        mock_nac.predict("tool", "bash:ls")
        captured = capsys.readouterr()
        assert "NAc PRED" in captured.err

    def test_wraps_record_outcome(self, mock_nac, capsys):
        NacTracer(mock_nac)
        mock_nac.record_outcome("evt_001", "success", "POSITIVE")
        captured = capsys.readouterr()
        assert "NAc OUTCOM" in captured.err

    def test_summary(self, mock_nac):
        tracer = NacTracer(mock_nac)
        s = tracer.summary()
        assert "0 events" in s
        assert "0 observations" in s
        assert "3 causal links" in s

    def test_counts_increment(self, mock_nac):
        tracer = NacTracer(mock_nac)
        mock_nac.record_event("tool", "bash")
        mock_nac.record_event("tool", "read_file")
        assert tracer._event_count == 2


# ── ATLTracer ────────────────────────────────────────────────────────────────


class TestATLTracer:
    @pytest.fixture
    def mock_atl(self):
        atl = MagicMock()
        atl._on_concept_captured = []
        atl._on_concept_deleted = []
        atl.recall = MagicMock(return_value=[])
        atl.store = MagicMock(return_value="concept_001")

        def reg_capture(cb):
            atl._on_concept_captured.append(cb)

        def reg_delete(cb):
            atl._on_concept_deleted.append(cb)

        atl.register_capture_callback = reg_capture
        atl.register_deletion_callback = reg_delete
        atl.__len__ = MagicMock(return_value=10)
        return atl

    def test_registers_callbacks(self, mock_atl):
        ATLTracer(mock_atl)
        assert len(mock_atl._on_concept_captured) == 1
        assert len(mock_atl._on_concept_deleted) == 1

    def test_store_callback_prints(self, mock_atl, capsys):
        ATLTracer(mock_atl)
        concept = MagicMock()
        concept.name = "safety"
        concept.category = "skill_execution"
        for cb in mock_atl._on_concept_captured:
            cb("c_001", concept)
        captured = capsys.readouterr()
        assert "ATL STORE" in captured.err
        assert "safety" in captured.err

    def test_delete_callback_prints(self, mock_atl, capsys):
        ATLTracer(mock_atl)
        for cb in mock_atl._on_concept_deleted:
            cb("c_002_abcdef")
        captured = capsys.readouterr()
        assert "ATL DELETE" in captured.err

    def test_recall_wraps_and_traces(self, mock_atl, capsys):
        ATLTracer(mock_atl)
        mock_atl.recall(concept_name="safety")
        captured = capsys.readouterr()
        assert "ATL RECALL" in captured.err

    def test_summary(self, mock_atl):
        tracer = ATLTracer(mock_atl)
        s = tracer.summary()
        assert "0 stored" in s
        assert "10 concepts alive" in s

    def test_store_count_increments(self, mock_atl):
        tracer = ATLTracer(mock_atl)
        concept = MagicMock()
        concept.name = "test"
        concept.category = ""
        for cb in mock_atl._on_concept_captured:
            cb("c_1", concept)
            cb("c_2", concept)
        assert tracer._store_count == 2
