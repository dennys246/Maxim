"""Unit tests for maxim.conscience.input_handlers.InputHandlerMixin."""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from maxim.conscience.input_handlers import InputHandlerMixin


class _TestHandler(InputHandlerMixin):
    def __init__(self):
        self.log = logging.getLogger("test")
        self._cli_logger = None
        self._training_logger = None
        self._observation_lock = threading.Lock()
        self._voice_agentic_enabled = False
        self._last_transcript_event = None
        self._outcome_code = 0
        self._last_action_event_id = None
        self._last_motor_sample = None
        self._phrase_last_trigger_ts = {}
        self._training_paused = threading.Event()
        self.phrase_responses = {}
        self.key_responses = {}
        self.interactive = True
        self.audio = False
        self.run_id = "test-run"
        self.mode = "active"
        self.current_epoch = 0
        self.video_path = None
        self.audio_path = None
        self.transcript_path = None
        self.train = False
        self._agentic_state = None


@pytest.fixture
def handler():
    return _TestHandler()


# ---------- label_outcome ----------


class TestLabelOutcome:
    def test_sets_outcome_code(self, handler):
        handler.label_outcome(5)
        assert handler._outcome_code == 5

    def test_clamps_negative_to_zero(self, handler):
        handler.label_outcome(-3)
        assert handler._outcome_code == 0

    def test_clamps_above_nine(self, handler):
        handler.label_outcome(15)
        assert handler._outcome_code == 9

    def test_clamps_exactly_zero(self, handler):
        handler._outcome_code = 7
        handler.label_outcome(0)
        assert handler._outcome_code == 0

    def test_clamps_exactly_nine(self, handler):
        handler.label_outcome(9)
        assert handler._outcome_code == 9

    def test_logs_event_with_flush(self, handler):
        mock_logger = MagicMock()
        handler._training_logger = mock_logger
        handler.label_outcome(3, source="cli", trigger="3", note="test note")
        mock_logger.log_event.assert_called_once()
        call_args = mock_logger.log_event.call_args
        record = call_args[0][0]
        assert record["kind"] == "outcome_label"
        assert record["code"] == 3
        assert record["source"] == "cli"
        assert record["trigger"] == "3"
        assert record["note"] == "test note"
        assert call_args[1]["flush"] is True

    def test_no_log_when_no_training_logger(self, handler):
        # Should not raise even with no logger
        handler.label_outcome(5)
        assert handler._outcome_code == 5

    def test_non_numeric_code_ignored(self, handler):
        handler._outcome_code = 4
        handler.label_outcome("not_a_number")
        # int("not_a_number") raises, so code stays unchanged
        assert handler._outcome_code == 4

    def test_includes_last_motor_sample_id(self, handler):
        mock_logger = MagicMock()
        handler._training_logger = mock_logger
        handler._last_motor_sample = {"sample_id": "motor-123"}
        handler.label_outcome(2)
        record = mock_logger.log_event.call_args[0][0]
        assert record["last_motor_sample_id"] == "motor-123"

    def test_includes_transcript_ref(self, handler):
        mock_logger = MagicMock()
        handler._training_logger = mock_logger
        handler._last_transcript_event = {
            "chunk_index": 7,
            "start_s": 1.0,
            "end_s": 2.0,
            "text": "hello maxim",
        }
        handler.label_outcome(1)
        record = mock_logger.log_event.call_args[0][0]
        assert record["transcript"]["text"] == "hello maxim"
        assert record["transcript"]["chunk_index"] == 7


# ---------- _handle_keypress ----------


class TestHandleKeypress:
    def test_dispatches_known_key(self, handler):
        handler.key_responses = {
            "q": {"call": "label_outcome", "args": [0]},
        }
        handler.label_outcome = MagicMock()
        handler._handle_keypress("q")
        handler.label_outcome.assert_called_once()

    def test_ignores_unknown_key(self, handler):
        handler.key_responses = {}
        # Should not raise
        handler._handle_keypress("z")

    def test_ignores_non_dict_spec(self, handler):
        handler.key_responses = {"x": "not-a-dict"}
        handler._handle_keypress("x")


# ---------- _log_cli_input ----------


class TestLogCliInput:
    def test_noop_when_no_logger(self, handler):
        handler._cli_logger = None
        # Should not raise
        handler._log_cli_input("hello")

    def test_calls_log_input(self, handler):
        mock_logger = MagicMock()
        handler._cli_logger = mock_logger
        handler._log_cli_input("test input")
        mock_logger.log_input.assert_called_once()
        record = mock_logger.log_input.call_args[0][0]
        assert record["kind"] == "cli_input"
        assert record["text"] == "test input"
        assert record["run_id"] == "test-run"
        assert record["mode"] == "active"


# ---------- _log_event ----------


class TestLogEvent:
    def test_noop_when_no_logger(self, handler):
        handler._training_logger = None
        handler._log_event({"kind": "test"})

    def test_calls_log_event(self, handler):
        mock_logger = MagicMock()
        handler._training_logger = mock_logger
        record = {"kind": "test_event"}
        handler._log_event(record)
        mock_logger.log_event.assert_called_once_with(record, flush=False)

    def test_calls_log_event_with_flush(self, handler):
        mock_logger = MagicMock()
        handler._training_logger = mock_logger
        record = {"kind": "test_event"}
        handler._log_event(record, flush=True)
        mock_logger.log_event.assert_called_once_with(record, flush=True)

    def test_swallows_logger_exception(self, handler):
        mock_logger = MagicMock()
        mock_logger.log_event.side_effect = RuntimeError("boom")
        handler._training_logger = mock_logger
        # Should not raise
        handler._log_event({"kind": "test"})


# ---------- _handle_cli_text ----------


class TestHandleCliText:
    def test_empty_text_ignored(self, handler):
        handler._handle_cli_text("")
        handler._handle_cli_text("   ")
        handler._handle_cli_text(None)

    def test_single_char_dispatches_as_key(self, handler):
        handler.key_responses = {
            "q": {"call": "label_outcome", "args": [0]},
        }
        handler.label_outcome = MagicMock()
        handler._handle_cli_text("q")
        handler.label_outcome.assert_called_once()

    def test_single_char_falls_through_if_not_in_key_responses(self, handler):
        handler.key_responses = {}
        handler.phrase_responses = {
            "x": {"call": "label_outcome", "args": [1], "_normalized": "x"},
        }
        handler.label_outcome = MagicMock()
        handler._handle_cli_text("x")
        handler.label_outcome.assert_called_once()

    def test_forwards_to_agentic_state(self, handler):
        agentic = MagicMock()
        agentic.data = {}
        handler._agentic_state = agentic
        handler._handle_cli_text("hello world")
        assert agentic.data["pending_cli_input"] == "hello world"

    def test_multi_char_goes_to_phrase_text(self, handler):
        handler.phrase_responses = {
            "hello": {"call": "label_outcome", "args": [5], "_normalized": "hello"},
        }
        handler.label_outcome = MagicMock()
        handler._handle_cli_text("hello")
        handler.label_outcome.assert_called_once()

    def test_logs_cli_input(self, handler):
        mock_logger = MagicMock()
        handler._cli_logger = mock_logger
        handler._handle_cli_text("test")
        mock_logger.log_input.assert_called_once()


# ---------- _handle_transcript_record ----------


class TestHandleTranscriptRecord:
    def test_empty_text_ignored(self, handler):
        handler._handle_transcript_record({"text": ""})
        assert handler._last_transcript_event is None

    def test_missing_text_ignored(self, handler):
        handler._handle_transcript_record({})
        assert handler._last_transcript_event is None

    def test_stores_last_transcript_event(self, handler):
        record = {"text": "something"}
        handler._handle_transcript_record(record)
        assert handler._last_transcript_event is record

    def test_wake_word_maxim_forwards_to_agentic(self, handler):
        agentic = MagicMock()
        agentic.data = {}
        handler._agentic_state = agentic
        record = {"text": "hey maxim do something"}
        handler._handle_transcript_record(record)
        assert agentic.data["pending_voice_input"] == "hey maxim do something"
        assert agentic.data["pending_voice_transcript"] is record

    def test_wake_word_reachy_forwards_to_agentic(self, handler):
        agentic = MagicMock()
        agentic.data = {}
        handler._agentic_state = agentic
        record = {"text": "reachy wave hello"}
        handler._handle_transcript_record(record)
        assert agentic.data["pending_voice_input"] == "reachy wave hello"

    def test_maximum_also_triggers_wake(self, handler):
        """'maximum' contains 'maxim' so it should trigger wake word forwarding."""
        agentic = MagicMock()
        agentic.data = {}
        handler._agentic_state = agentic
        record = {"text": "maximum effort"}
        handler._handle_transcript_record(record)
        assert agentic.data["pending_voice_input"] == "maximum effort"

    def test_no_wake_word_no_forward(self, handler):
        agentic = MagicMock()
        agentic.data = {}
        handler._agentic_state = agentic
        record = {"text": "hello world"}
        handler._handle_transcript_record(record)
        assert "pending_voice_input" not in agentic.data

    def test_no_agentic_state_logs_warning(self, handler):
        handler._agentic_state = None
        record = {"text": "hey maxim"}
        with patch.object(handler.log, "warning") as mock_warn:
            handler._handle_transcript_record(record)
            mock_warn.assert_called()


# ---------- _handle_phrase_text ----------


class TestHandlePhraseText:
    def test_no_match_returns_without_action(self, handler):
        handler.phrase_responses = {
            "goodbye": {"call": "label_outcome", "args": [1], "_normalized": "goodbye"},
        }
        handler.label_outcome = MagicMock()
        handler._handle_phrase_text("hello", source="cli")
        handler.label_outcome.assert_not_called()

    def test_exact_match_triggers_action(self, handler):
        handler.phrase_responses = {
            "hello": {"call": "label_outcome", "args": [5], "_normalized": "hello"},
        }
        handler.label_outcome = MagicMock()
        handler._handle_phrase_text("hello", source="cli")
        handler.label_outcome.assert_called_once()

    def test_empty_text_returns(self, handler):
        handler.phrase_responses = {
            "hello": {"call": "label_outcome", "args": [5], "_normalized": "hello"},
        }
        handler.label_outcome = MagicMock()
        handler._handle_phrase_text("", source="cli")
        handler.label_outcome.assert_not_called()

    def test_case_insensitive_matching(self, handler):
        handler.phrase_responses = {
            "hello": {"call": "label_outcome", "args": [5], "_normalized": "hello"},
        }
        handler.label_outcome = MagicMock()
        handler._handle_phrase_text("HELLO", source="cli")
        handler.label_outcome.assert_called_once()

    def test_substring_match(self, handler):
        handler.phrase_responses = {
            "stop": {"call": "label_outcome", "args": [0], "_normalized": "stop"},
        }
        handler.label_outcome = MagicMock()
        handler._handle_phrase_text("please stop now", source="cli")
        handler.label_outcome.assert_called_once()

    def test_cooldown_blocks_voice_source(self, handler):
        handler.phrase_responses = {
            "stop": {
                "call": "label_outcome",
                "args": [0],
                "_normalized": "stop",
                "cooldown_s": 9999,
            },
        }
        handler.label_outcome = MagicMock()
        # First trigger sets the timestamp
        handler._handle_phrase_text("stop", source="voice")
        handler.label_outcome.assert_called_once()
        handler.label_outcome.reset_mock()
        # Second trigger within cooldown should be blocked
        handler._handle_phrase_text("stop", source="voice")
        handler.label_outcome.assert_not_called()

    def test_cooldown_ignored_for_cli_source(self, handler):
        handler.phrase_responses = {
            "stop": {
                "call": "label_outcome",
                "args": [0],
                "_normalized": "stop",
                "cooldown_s": 9999,
            },
        }
        handler.label_outcome = MagicMock()
        handler._handle_phrase_text("stop", source="cli")
        handler.label_outcome.assert_called_once()
        handler.label_outcome.reset_mock()
        # CLI source ignores cooldown
        handler._handle_phrase_text("stop", source="cli")
        handler.label_outcome.assert_called_once()

    def test_requires_agentic_blocks_voice_when_disabled(self, handler):
        handler._voice_agentic_enabled = False
        handler.phrase_responses = {
            "do thing": {
                "call": "label_outcome",
                "args": [1],
                "_normalized": "do thing",
                "requires_agentic": True,
            },
        }
        handler.label_outcome = MagicMock()
        handler._handle_phrase_text("do thing", source="voice")
        handler.label_outcome.assert_not_called()

    def test_requires_agentic_allowed_for_cli(self, handler):
        handler._voice_agentic_enabled = False
        handler.phrase_responses = {
            "do thing": {
                "call": "label_outcome",
                "args": [1],
                "_normalized": "do thing",
                "requires_agentic": True,
            },
        }
        handler.label_outcome = MagicMock()
        handler._handle_phrase_text("do thing", source="cli")
        handler.label_outcome.assert_called_once()
