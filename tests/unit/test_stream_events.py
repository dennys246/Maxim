"""Unit tests for StreamEvent from bus.py."""

from __future__ import annotations

import dataclasses
import time

import pytest

from maxim.agents.bus import StreamEvent


class TestStreamEvent:
    def test_creation(self):
        evt = StreamEvent(kind="tool_start", payload={"tool": "read_file"})
        assert evt.kind == "tool_start"
        assert evt.payload == {"tool": "read_file"}

    def test_frozen(self):
        evt = StreamEvent(kind="tool_start", payload={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.kind = "tool_end"

    def test_auto_timestamp(self):
        before = time.time()
        evt = StreamEvent(kind="inference_start")
        after = time.time()
        assert before <= evt.timestamp <= after

    def test_explicit_timestamp(self):
        evt = StreamEvent(kind="tool_end", timestamp=123.0)
        assert evt.timestamp == 123.0

    def test_various_kinds(self):
        for kind in ("inference_start", "inference_end", "tool_start", "tool_end"):
            evt = StreamEvent(kind=kind)
            assert evt.kind == kind
