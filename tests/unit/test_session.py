"""Tests for session persistence (Phase 3 #8)."""

from __future__ import annotations

import time
from pathlib import Path


from maxim.agents.bus import Percept, StopReason
from maxim.runtime.session import AgentSession


# ---------------------------------------------------------------------------
# 1. Percept to_dict roundtrip
# ---------------------------------------------------------------------------


class TestPerceptRoundtrip:
    """Percept.to_dict -> Percept.from_dict should preserve key fields."""

    def test_roundtrip_preserves_fields(self) -> None:
        original = Percept(
            timestamp=1234567890.0,
            source="transcript",
            salience=0.7,
            novelty=0.5,
            has_voice_command=True,
            has_maxim_keyword=False,
            cli_input="hello maxim",
            content="test content",
        )
        d = original.to_dict()
        restored = Percept.from_dict(d)

        assert restored.timestamp == original.timestamp
        assert restored.source == original.source
        assert restored.salience == original.salience
        assert restored.novelty == original.novelty
        assert restored.has_voice_command is True
        assert restored.cli_input == "hello maxim"
        assert restored.content == "test content"


# ---------------------------------------------------------------------------
# 2. Percept to_dict omits large fields
# ---------------------------------------------------------------------------


class TestPerceptOmitsLargeFields:
    """to_dict should NOT include raw_transcript_text or maxim_runtime."""

    def test_large_fields_omitted(self) -> None:
        percept = Percept(
            timestamp=time.time(),
            source="vision",
            raw_transcript_text="a very long transcript...",
            maxim_runtime={"heavy": "data"},
        )
        d = percept.to_dict()
        assert "raw_transcript_text" not in d
        assert "maxim_runtime" not in d


# ---------------------------------------------------------------------------
# 3. AgentSession save/load roundtrip
# ---------------------------------------------------------------------------


class TestAgentSessionRoundtrip:
    """Save then load should reproduce all fields."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        session = AgentSession(session_id="sess-001")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")
        session.log_event("goal", "set_goal", "Build a widget")
        session.total_steps = 5
        session.plan_id = "plan-42"

        session.save(tmp_path)
        loaded = AgentSession.from_saved(tmp_path / "sess-001.json")

        assert loaded.session_id == "sess-001"
        assert len(loaded.messages) == 2
        assert loaded.messages[0]["role"] == "user"
        assert loaded.messages[0]["content"] == "Hello"
        assert loaded.messages[1]["content"] == "Hi there"
        assert len(loaded.events) == 1
        assert loaded.total_steps == 5
        assert loaded.plan_id == "plan-42"


# ---------------------------------------------------------------------------
# 4. AgentSession atomic write
# ---------------------------------------------------------------------------


class TestAgentSessionAtomicWrite:
    """After save(), no .tmp file should remain."""

    def test_no_tmp_file_remains(self, tmp_path: Path) -> None:
        session = AgentSession(session_id="atomic-test")
        session.save(tmp_path)

        tmp_file = tmp_path / "atomic-test.tmp"
        assert not tmp_file.exists()

        json_file = tmp_path / "atomic-test.json"
        assert json_file.exists()


# ---------------------------------------------------------------------------
# 5. AgentSession log_event
# ---------------------------------------------------------------------------


class TestAgentSessionLogEvent:
    """log_event should accumulate entries with timestamps."""

    def test_three_events(self) -> None:
        session = AgentSession(session_id="evt-test")
        session.log_event("goal", "start", "detail1")
        session.log_event("plan", "transition", "detail2")
        session.log_event("error", "tool_fail", "detail3")

        assert len(session.events) == 3
        for event in session.events:
            assert "ts" in event
            assert isinstance(event["ts"], float)
        assert session.events[0]["cat"] == "goal"
        assert session.events[1]["title"] == "transition"
        assert session.events[2]["detail"] == "detail3"


# ---------------------------------------------------------------------------
# 6. AgentSession with StopReason
# ---------------------------------------------------------------------------


class TestAgentSessionStopReason:
    """StopReason should survive save/load."""

    def test_stop_reason_preserved(self, tmp_path: Path) -> None:
        session = AgentSession(session_id="stop-test")
        session.stop_reason = StopReason.COMPLETED
        session.save(tmp_path)

        loaded = AgentSession.from_saved(tmp_path / "stop-test.json")
        assert loaded.stop_reason == StopReason.COMPLETED

    def test_none_stop_reason_preserved(self, tmp_path: Path) -> None:
        session = AgentSession(session_id="no-stop")
        session.save(tmp_path)

        loaded = AgentSession.from_saved(tmp_path / "no-stop.json")
        assert loaded.stop_reason is None
