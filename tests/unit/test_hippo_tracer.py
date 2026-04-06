"""Tests for HippocampusTracer — real-time tracing of memory operations."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from maxim.memory.hippo_tracer import HippocampusTracer, _short_id, _excerpt


class TestHelpers:
    def test_short_id(self):
        assert _short_id("abcdef1234567890") == "abcdef1234"
        assert _short_id("short") == "short"
        assert _short_id("") == "?"

    def test_excerpt_truncation(self):
        assert _excerpt("hello world") == "hello world"
        assert len(_excerpt("a" * 100, max_len=20)) == 20
        assert _excerpt("a" * 100, max_len=20).endswith("...")

    def test_excerpt_newlines(self):
        assert _excerpt("line1\nline2") == "line1 line2"

    def test_excerpt_none(self):
        assert _excerpt(None) == ""


class TestHippocampusTracer:
    @pytest.fixture
    def mock_hippo(self):
        """Create a mock Hippocampus with the callback registration methods."""
        hippo = MagicMock()
        hippo._on_memory_deleted = []
        hippo._on_memory_captured = []
        hippo._on_memory_compressed = []
        hippo._graph = MagicMock()
        hippo._graph.get_outgoing.return_value = []
        hippo._graph._outgoing = {}
        hippo.recall = MagicMock(return_value=[])
        hippo.recall_associated = MagicMock(return_value=[])

        def reg_capture(cb):
            hippo._on_memory_captured.append(cb)

        def reg_compress(cb):
            hippo._on_memory_compressed.append(cb)

        hippo.register_capture_callback = reg_capture
        hippo.register_compression_callback = reg_compress
        hippo.__len__ = MagicMock(return_value=5)
        return hippo

    def test_tracer_registers_callbacks(self, mock_hippo):
        _tracer = HippocampusTracer(mock_hippo)  # noqa: F841
        assert len(mock_hippo._on_memory_captured) == 1
        assert len(mock_hippo._on_memory_compressed) == 1
        assert len(mock_hippo._on_memory_deleted) == 1

    def test_tracer_wraps_recall(self, mock_hippo):
        HippocampusTracer(mock_hippo)
        # The original recall was replaced
        result = mock_hippo.recall(goal="test")
        assert isinstance(result, list)

    def test_capture_callback_prints(self, mock_hippo, capsys):
        HippocampusTracer(mock_hippo)

        # Build a mock EpisodicMemory
        memory = MagicMock()
        memory.context.active_goal = "find the key"
        memory.action.tool_name = "respond"
        memory.outcome.success = True
        memory.perception.salience = 0.9
        memory.perception.novelty = 0.8
        memory.perception.cli_input = "test input"

        # Trigger the callback
        for cb in mock_hippo._on_memory_captured:
            cb("mem_001_abcdef", memory)

        captured = capsys.readouterr()
        assert "CAPTURE" in captured.err
        assert "mem_001_ab" in captured.err
        assert "respond" in captured.err

    def test_compression_callback_prints(self, mock_hippo, capsys):
        HippocampusTracer(mock_hippo)
        for cb in mock_hippo._on_memory_compressed:
            cb("mem_002_abcdef")

        captured = capsys.readouterr()
        assert "COMPRESS" in captured.err
        assert "mem_002_ab" in captured.err

    def test_deletion_callback_prints(self, mock_hippo, capsys):
        HippocampusTracer(mock_hippo)
        for cb in mock_hippo._on_memory_deleted:
            cb("mem_003_abcdef")

        captured = capsys.readouterr()
        assert "EVICT" in captured.err
        assert "mem_003_ab" in captured.err

    def test_summary(self, mock_hippo):
        tracer = HippocampusTracer(mock_hippo)
        s = tracer.summary()
        assert "0 captures" in s
        assert "0 recalls" in s

    def test_capture_increments_count(self, mock_hippo):
        tracer = HippocampusTracer(mock_hippo)
        memory = MagicMock()
        memory.context.active_goal = "test"
        memory.action.tool_name = "respond"
        memory.outcome.success = True
        memory.perception.salience = 0.5
        memory.perception.novelty = 0.5
        memory.perception.cli_input = ""

        for cb in mock_hippo._on_memory_captured:
            cb("mem_1", memory)
            cb("mem_2", memory)

        assert tracer._capture_count == 2
