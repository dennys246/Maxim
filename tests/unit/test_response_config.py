"""Unit tests for response configuration loading."""

from __future__ import annotations

import json



class TestNormalizeTriggerText:
    """Test text normalization for phrase matching."""

    def test_lowercase(self):
        """Converts to lowercase."""
        from maxim.utils.response_config import normalize_trigger_text

        assert normalize_trigger_text("HELLO WORLD") == "hello world"

    def test_removes_punctuation(self):
        """Removes punctuation."""
        from maxim.utils.response_config import normalize_trigger_text

        assert normalize_trigger_text("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        """Collapses multiple spaces."""
        from maxim.utils.response_config import normalize_trigger_text

        assert normalize_trigger_text("hello   world") == "hello world"

    def test_empty_string(self):
        """Returns empty for empty input."""
        from maxim.utils.response_config import normalize_trigger_text

        assert normalize_trigger_text("") == ""
        assert normalize_trigger_text("   ") == ""

    def test_preserves_unicode_words(self):
        """Preserves unicode word characters."""
        from maxim.utils.response_config import normalize_trigger_text

        assert normalize_trigger_text("cafe") == "cafe"


class TestNormalizeTranscriptText:
    """Test transcript normalization with alias expansion."""

    def test_removes_standalone_s(self):
        """Removes standalone 's' tokens."""
        from maxim.utils.response_config import normalize_transcript_text

        assert normalize_transcript_text("hello s world") == "hello world"

    def test_expands_maximum_alias(self):
        """Expands 'maximum' to 'maxim'."""
        from maxim.utils.response_config import normalize_transcript_text

        assert normalize_transcript_text("maximum sleep") == "maxim sleep"

    def test_expands_maximums_alias(self):
        """Expands 'maximums' to 'maxim'."""
        from maxim.utils.response_config import normalize_transcript_text

        assert normalize_transcript_text("maximums wake") == "maxim wake"

    def test_combined_normalization(self):
        """Combines all normalizations."""
        from maxim.utils.response_config import normalize_transcript_text

        assert normalize_transcript_text("MAXIMUM s SHUTDOWN!") == "maxim shutdown"


class TestLoadKeyResponses:
    """Test key response loading."""

    def test_returns_default_when_no_file(self):
        """Returns defaults when no config file exists."""
        from maxim.utils.response_config import load_key_responses

        responses = load_key_responses()

        # Check default key responses
        assert "c" in responses
        assert responses["c"]["call"] == "center_vision"

    def test_numeric_keys_map_to_label_outcome(self):
        """Numeric keys 0-9 map to label_outcome."""
        from maxim.utils.response_config import load_key_responses

        responses = load_key_responses()

        for i in range(10):
            assert str(i) in responses
            assert responses[str(i)]["call"] == "label_outcome"
            assert responses[str(i)]["args"] == [i]

    def test_loads_from_json_file(self, tmp_path):
        """Loads custom responses from JSON file."""
        import os
        from maxim.utils.response_config import load_key_responses

        config_path = tmp_path / "key_responses.json"
        config_path.write_text(json.dumps({
            "x": {"call": "custom_action", "args": [1, 2]},
        }))

        old_env = os.environ.get("MAXIM_KEY_RESPONSES")
        try:
            os.environ["MAXIM_KEY_RESPONSES"] = str(config_path)
            responses = load_key_responses()
        finally:
            if old_env is None:
                os.environ.pop("MAXIM_KEY_RESPONSES", None)
            else:
                os.environ["MAXIM_KEY_RESPONSES"] = old_env

        assert "x" in responses
        assert responses["x"]["call"] == "custom_action"
        assert responses["x"]["args"] == [1, 2]


class TestLoadPhraseResponses:
    """Test phrase response loading."""

    def test_returns_default_when_no_file(self):
        """Returns defaults when no config file exists."""
        from maxim.utils.response_config import load_phrase_responses

        responses = load_phrase_responses()

        # Check some default phrases
        assert "maxim shutdown" in responses
        assert responses["maxim shutdown"]["call"] == "request_shutdown"

    def test_wake_words_have_correct_flags(self):
        """Wake words have wake_word=True, requires_agentic=False."""
        from maxim.utils.response_config import load_phrase_responses

        responses = load_phrase_responses()

        assert "maxim" in responses
        assert responses["maxim"]["wake_word"] is True

    def test_patterns_are_compiled(self):
        """Phrase patterns are compiled regex objects."""
        import re
        from maxim.utils.response_config import load_phrase_responses

        responses = load_phrase_responses()

        for phrase, spec in responses.items():
            if isinstance(spec, dict):
                pattern = spec.get("_pattern")
                assert pattern is None or isinstance(pattern, re.Pattern)

    def test_normalized_text_is_set(self):
        """Normalized text is set for all phrases."""
        from maxim.utils.response_config import load_phrase_responses

        responses = load_phrase_responses()

        for phrase, spec in responses.items():
            if isinstance(spec, dict):
                assert "_normalized" in spec
                assert isinstance(spec["_normalized"], str)

    def test_cooldown_defaults_to_2(self):
        """Cooldown defaults to 2.0 seconds."""
        from maxim.utils.response_config import load_phrase_responses

        responses = load_phrase_responses()

        for phrase, spec in responses.items():
            if isinstance(spec, dict):
                assert spec.get("cooldown_s", 2.0) == 2.0


class TestKeyResponseDataclass:
    """Test KeyResponse dataclass."""

    def test_creation(self):
        """Can create KeyResponse with defaults."""
        from maxim.utils.response_config import KeyResponse

        kr = KeyResponse(call="test_method")

        assert kr.call == "test_method"
        assert kr.args == []
        assert kr.kwargs == {}
        assert kr.pause_training is False


class TestPhraseResponseDataclass:
    """Test PhraseResponse dataclass."""

    def test_creation(self):
        """Can create PhraseResponse with defaults."""
        from maxim.utils.response_config import PhraseResponse

        pr = PhraseResponse(call="test_method")

        assert pr.call == "test_method"
        assert pr.args == []
        assert pr.kwargs == {}
        assert pr.pause_training is False
        assert pr.wake_word is False
        assert pr.requires_agentic is True
        assert pr.cooldown_s == 2.0
        assert pr.pattern is None
        assert pr.normalized == ""
