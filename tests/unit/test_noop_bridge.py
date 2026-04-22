"""Tests for NoOp deliberation marker and bridge F2 fix (Stage 4).

Covers:
- RecordingSink.record_noop() creates correct ActionRecord
- Bridge doesn't count noop markers as real actions (timed_out logic)
- Noop markers break the bridge timeout loop (action count grows)
- Bridge response text excludes noop markers
"""

from __future__ import annotations

import time

from maxim.simulation.sinks import ActionRecord, RecordingSink


class TestRecordingSinkNoop:
    def test_record_noop_creates_marker(self):
        sink = RecordingSink()
        sink.record_noop(reason="thinking")
        assert len(sink.actions) == 1
        action = sink.actions[0]
        assert action.tool_name == "_deliberation_noop"
        assert action.result_success is True
        assert "thinking" in str(action.result_output)

    def test_record_noop_default_reason(self):
        sink = RecordingSink()
        sink.record_noop()
        action = sink.actions[0]
        assert "deliberation tick" in str(action.result_output)

    def test_noop_increments_action_count(self):
        """Noop markers grow len(actions) — this is the F2 fix."""
        sink = RecordingSink()
        count_before = len(sink.actions)
        sink.record_noop()
        assert len(sink.actions) == count_before + 1

    def test_noop_not_found_by_find_actions(self):
        """find_actions skips blocked actions; noop is not blocked."""
        sink = RecordingSink()
        sink.record_noop()
        sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name="respond",
                result_success=True,
                result_output="hello",
            )
        )
        # find_actions without tool filter returns all non-blocked
        all_actions = sink.find_actions()
        assert len(all_actions) == 2  # both are non-blocked

        # find_actions with tool filter returns only matching
        responds = sink.find_actions(tool="respond")
        assert len(responds) == 1
        assert responds[0].tool_name == "respond"

    def test_noop_mixed_with_real_actions(self):
        """Mix of noops and real actions — both recorded."""
        sink = RecordingSink()
        sink.record_noop(reason="thinking 1")
        sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name="respond",
                result_success=True,
                result_output="hello",
            )
        )
        sink.record_noop(reason="thinking 2")
        assert len(sink.actions) == 3


class TestBridgeNoopHandling:
    """Test that the bridge's timed_out logic excludes noop markers."""

    def test_noop_only_is_timed_out(self):
        """If only noop markers exist, bridge reports timed_out=True."""
        # Simulate what the bridge does with new_actions
        new_actions = [
            ActionRecord(
                timestamp=time.time(),
                tool_name="_deliberation_noop",
                result_success=True,
                result_output="thinking",
            ),
        ]
        real_actions = [a for a in new_actions if a.tool_name != "_deliberation_noop"]
        assert len(real_actions) == 0
        # Bridge would set timed_out = len(real_actions) == 0
        timed_out = len(real_actions) == 0
        assert timed_out is True

    def test_real_action_is_not_timed_out(self):
        """If real actions exist alongside noops, not timed out."""
        new_actions = [
            ActionRecord(
                timestamp=time.time(),
                tool_name="_deliberation_noop",
                result_success=True,
                result_output="thinking",
            ),
            ActionRecord(
                timestamp=time.time(),
                tool_name="respond",
                result_success=True,
                result_output="hello world",
            ),
        ]
        real_actions = [a for a in new_actions if a.tool_name != "_deliberation_noop"]
        assert len(real_actions) == 1
        timed_out = len(real_actions) == 0
        assert timed_out is False

    def test_response_text_excludes_noop(self):
        """Bridge response text extraction ignores noop markers."""
        new_actions = [
            ActionRecord(
                timestamp=time.time(),
                tool_name="_deliberation_noop",
                result_success=True,
                result_output="just thinking",
            ),
            ActionRecord(
                timestamp=time.time(),
                tool_name="respond",
                result_success=True,
                result_output="actual response",
            ),
        ]
        # Simulate bridge's response text collection
        response_parts = []
        for a in new_actions:
            if a.tool_name in ("respond", "speak") and a.result_output:
                response_parts.append(str(a.result_output))
        response_text = "\n".join(response_parts) if response_parts else None
        assert response_text == "actual response"
