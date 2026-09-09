"""Tests for the pairing-announcer factory (A9.1 spoken-code pairing).

`make_pairing_announcer` composes a TTS engine + an audio sink into the
`(code: str) -> None` callable `build_app(pairing_announcer=...)` expects. The
contract that matters: it speaks the code at the right sample rate, spells digits for
room clarity, and — the load-bearing security property — NEVER logs the code (A7),
even when the failure path fires.
"""

from __future__ import annotations

import logging

from maxim.console import make_pairing_announcer


class _FakeTTS:
    def __init__(self, sample_rate: int | None = 22050) -> None:
        if sample_rate is not None:
            self.sample_rate = sample_rate
        self.synth_calls: list[str] = []

    def synthesize(self, text: str):
        self.synth_calls.append(text)
        return [0, 1, 2]  # stand-in samples


class _FakeSpeak:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, samples, sample_rate: int = 16000):
        self.calls.append((samples, sample_rate))
        return True


def test_composes_synthesize_then_speak_at_tts_sample_rate():
    tts = _FakeTTS(sample_rate=22050)
    speak = _FakeSpeak()
    announce = make_pairing_announcer(tts, speak)

    announce("482913")

    assert tts.synth_calls == ["Your pairing code is 4, 8, 2, 9, 1, 3."]
    assert speak.calls == [([0, 1, 2], 22050)]


def test_default_phrase_spells_digits():
    tts = _FakeTTS()
    announce = make_pairing_announcer(tts, _FakeSpeak())
    announce("007")
    assert tts.synth_calls == ["Your pairing code is 0, 0, 7."]


def test_custom_phrase_is_used():
    tts = _FakeTTS()
    speak = _FakeSpeak()
    announce = make_pairing_announcer(tts, speak, phrase=lambda c: f"code {c} now")
    announce("13")
    assert tts.synth_calls == ["code 13 now"]


def test_sample_rate_override_wins():
    tts = _FakeTTS(sample_rate=22050)
    speak = _FakeSpeak()
    announce = make_pairing_announcer(tts, speak, sample_rate=48000)
    announce("111111")
    assert speak.calls[0][1] == 48000


def test_sample_rate_falls_back_to_16000_without_tts_attr():
    tts = _FakeTTS(sample_rate=None)  # no sample_rate attribute
    assert not hasattr(tts, "sample_rate")
    speak = _FakeSpeak()
    announce = make_pairing_announcer(tts, speak)
    announce("222222")
    assert speak.calls[0][1] == 16000


def test_returns_none_and_matches_the_contract():
    announce = make_pairing_announcer(_FakeTTS(), _FakeSpeak())
    assert announce("424242") is None  # (code: str) -> None


def test_synth_failure_does_not_propagate_and_never_logs_the_code(caplog):
    code = "482913"

    class _BoomTTS:
        sample_rate = 22050

        def synthesize(self, text: str):
            # Worst case for A7: the failure ECHOES the synth text (which contains
            # the code). The announcer must still not leak it into the log.
            raise RuntimeError(f"synth blew up on: {text}")

    announce = make_pairing_announcer(_BoomTTS(), _FakeSpeak())

    with caplog.at_level(logging.WARNING):
        assert announce(code) is None  # never propagates (daemon-thread callable)

    assert caplog.records, "a failure should be logged, not silently swallowed"
    for rec in caplog.records:
        assert code not in rec.getMessage(), "A7 violation: pairing code leaked into a log"
    # It logs the failure TYPE, which cannot contain the code.
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)


def test_speak_failure_does_not_propagate_and_never_logs_the_code(caplog):
    code = "999000"

    def _boom_speak(samples, sample_rate: int = 16000):
        raise OSError(f"speaker died mid-announce of {samples}")

    announce = make_pairing_announcer(_FakeTTS(), _boom_speak)
    with caplog.at_level(logging.WARNING):
        assert announce(code) is None
    for rec in caplog.records:
        assert code not in rec.getMessage()
    assert any("OSError" in r.getMessage() for r in caplog.records)
