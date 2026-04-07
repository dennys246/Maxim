"""Tests for ask_user tool (Stage C) and benchmark CLI (Stage D).

Covers:
- AskUserTool non-interactive mode
- AskUserTool replay mode
- Timeout escalation tracking
- JSONL audit writing and replay reading
- UserInteraction serialization
"""

from __future__ import annotations

import json

from maxim.simulation.tools_user import (
    AskUserTool,
    InteractionAuditWriter,
    InteractionReplayReader,
    TimeoutEscalation,
    UserInteraction,
)


# ---------------------------------------------------------------------------
# UserInteraction
# ---------------------------------------------------------------------------


class TestUserInteraction:
    def test_serialization(self):
        interaction = UserInteraction(
            turn=3,
            question="Left or right?",
            options=["left", "right"],
            response="left",
            timed_out=False,
            elapsed_s=4.2,
        )
        d = interaction.to_dict()
        assert d["turn"] == 3
        assert d["question"] == "Left or right?"
        assert d["options"] == ["left", "right"]
        assert d["response"] == "left"
        assert d["timed_out"] is False

    def test_timeout_interaction(self):
        interaction = UserInteraction(
            turn=7,
            question="Roll for perception",
            response=None,
            timed_out=True,
            elapsed_s=11.3,
        )
        d = interaction.to_dict()
        assert d["timed_out"] is True
        assert d["response"] is None


# ---------------------------------------------------------------------------
# Timeout escalation
# ---------------------------------------------------------------------------


class TestTimeoutEscalation:
    def test_initial_state(self):
        esc = TimeoutEscalation()
        assert esc.consecutive == 0
        assert esc.total == 0
        assert esc.escalation_level == "none"

    def test_first_timeout(self):
        esc = TimeoutEscalation()
        esc.record_timeout()
        assert esc.consecutive == 1
        assert esc.escalation_level == "gentle_nudge"

    def test_multiple_timeouts(self):
        esc = TimeoutEscalation()
        esc.record_timeout()
        esc.record_timeout()
        assert esc.consecutive == 2
        assert esc.escalation_level == "world_reacts"

    def test_heavy_timeout(self):
        esc = TimeoutEscalation()
        for _ in range(5):
            esc.record_timeout()
        assert esc.escalation_level == "passive_protagonist"

    def test_response_resets_consecutive(self):
        esc = TimeoutEscalation()
        esc.record_timeout()
        esc.record_timeout()
        esc.record_response()
        assert esc.consecutive == 0
        assert esc.total == 2  # total stays
        assert esc.escalation_level == "none"

    def test_narrator_hint_none(self):
        esc = TimeoutEscalation()
        assert esc.to_narrator_hint() == ""

    def test_narrator_hint_gentle(self):
        esc = TimeoutEscalation()
        esc.record_timeout()
        hint = esc.to_narrator_hint()
        assert "gentle nudge" in hint.lower()

    def test_narrator_hint_world_reacts(self):
        esc = TimeoutEscalation()
        esc.record_timeout()
        esc.record_timeout()
        hint = esc.to_narrator_hint()
        assert "world reacts" in hint.lower()

    def test_narrator_hint_passive(self):
        esc = TimeoutEscalation()
        for _ in range(5):
            esc.record_timeout()
        hint = esc.to_narrator_hint()
        assert "passive protagonist" in hint.lower()


# ---------------------------------------------------------------------------
# Audit writer + replay reader
# ---------------------------------------------------------------------------


class TestAuditAndReplay:
    def test_write_and_read(self, tmp_path):
        writer = InteractionAuditWriter(tmp_path)
        writer.record(
            UserInteraction(
                turn=1,
                question="Go left?",
                options=["yes", "no"],
                response="yes",
                timed_out=False,
                elapsed_s=2.0,
            )
        )
        writer.record(
            UserInteraction(
                turn=2,
                question="Roll dice",
                response=None,
                timed_out=True,
                elapsed_s=10.5,
            )
        )

        reader = InteractionReplayReader(tmp_path)
        assert reader.remaining == 2

        r1 = reader.next_response()
        assert r1 is not None
        assert r1["response"] == "yes"
        assert r1["timed_out"] is False

        r2 = reader.next_response()
        assert r2 is not None
        assert r2["timed_out"] is True

        r3 = reader.next_response()
        assert r3 is None  # exhausted

    def test_empty_session(self, tmp_path):
        reader = InteractionReplayReader(tmp_path)
        assert reader.remaining == 0
        assert reader.next_response() is None


# ---------------------------------------------------------------------------
# AskUserTool — non-interactive mode
# ---------------------------------------------------------------------------


class TestAskUserToolNonInteractive:
    def test_returns_default(self):
        tool = AskUserTool(non_interactive=True)
        result = tool.execute(
            question="Choose wisely",
            options=["a", "b"],
            default="a",
        )
        assert result["response"] == "a"
        assert result["was_default"] is True
        assert result["timed_out"] is False

    def test_escalation_not_triggered(self):
        tool = AskUserTool(non_interactive=True)
        tool.execute(question="Q1", default="x")
        tool.execute(question="Q2", default="y")
        assert tool.escalation.consecutive == 0

    def test_audit_recorded(self, tmp_path):
        tool = AskUserTool(non_interactive=True, session_dir=tmp_path)
        tool.execute(question="Test?", default="yes")

        # Check JSONL was written
        jsonl_path = tmp_path / "user_interactions.jsonl"
        assert jsonl_path.exists()
        with open(jsonl_path) as f:
            record = json.loads(f.readline())
        assert record["question"] == "Test?"
        assert record["response"] == "yes"


# ---------------------------------------------------------------------------
# AskUserTool — replay mode
# ---------------------------------------------------------------------------


class TestAskUserToolReplay:
    def test_replays_recorded_responses(self, tmp_path):
        # Write some interactions
        writer = InteractionAuditWriter(tmp_path)
        writer.record(
            UserInteraction(
                turn=1,
                question="Q1",
                response="answer1",
                timed_out=False,
            )
        )
        writer.record(
            UserInteraction(
                turn=2,
                question="Q2",
                response=None,
                timed_out=True,
            )
        )

        # Replay
        tool = AskUserTool(replay_from=tmp_path, session_dir=tmp_path / "replay")
        r1 = tool.execute(question="Q1", default="default1")
        assert r1["response"] == "answer1"
        assert r1["timed_out"] is False

        r2 = tool.execute(question="Q2", default="default2")
        assert r2["response"] == "default2"  # timed_out → uses default
        assert r2["timed_out"] is True

    def test_replay_exhausted_returns_default(self, tmp_path):
        writer = InteractionAuditWriter(tmp_path)
        writer.record(
            UserInteraction(
                turn=1,
                question="Q1",
                response="a",
                timed_out=False,
            )
        )

        tool = AskUserTool(replay_from=tmp_path)
        tool.execute(question="Q1", default="x")  # consumes the one record
        r = tool.execute(question="Q2", default="fallback")
        assert r["response"] == "fallback"
        assert r["timed_out"] is True

    def test_replay_tracks_escalation(self, tmp_path):
        writer = InteractionAuditWriter(tmp_path)
        writer.record(UserInteraction(turn=1, question="Q", timed_out=True))
        writer.record(UserInteraction(turn=2, question="Q", timed_out=True))
        writer.record(UserInteraction(turn=3, question="Q", response="ok", timed_out=False))

        tool = AskUserTool(replay_from=tmp_path)
        tool.execute(question="Q", default="d")
        assert tool.escalation.consecutive == 1
        tool.execute(question="Q", default="d")
        assert tool.escalation.consecutive == 2
        tool.execute(question="Q", default="d")
        assert tool.escalation.consecutive == 0  # reset on success
