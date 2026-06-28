"""Tests for the audio sound-localization front-end (commit 4).

Covers the hardware-agnostic core — normalization + the pull-per-tick
``AzimuthDoASource`` with an injected fake DoA reader — plus the
optional-dependency guard on the live Reachy adapter. The live adapter
itself is verified on-device (the ``reachy`` extra is not installed in CI).

Regression guard for perception_pipeline_placement.md (commit 4).
"""

from __future__ import annotations

import math

import pytest

from maxim.agents.percept_factory import make_audio_percept
from maxim.embodiment.audio_localization import (
    AzimuthDoASource,
    doa_to_azimuth,
    make_reachy_doa_reader,
)
from maxim.simulation.sources import PerceptSource


class TestDoAToAzimuth:
    def test_left_maps_to_minus_one(self):
        assert doa_to_azimuth(0.0) == -1.0

    def test_front_maps_to_zero(self):
        assert doa_to_azimuth(math.pi / 2) == 0.0

    def test_right_maps_to_plus_one(self):
        assert doa_to_azimuth(math.pi) == 1.0

    def test_midleft_is_negative(self):
        # π/4 → halfway between left(-1) and front(0) → -0.5
        assert doa_to_azimuth(math.pi / 4) == pytest.approx(-0.5)

    def test_clamps_out_of_range(self):
        assert doa_to_azimuth(-1.0) == -1.0  # below 0 → clamp low
        assert doa_to_azimuth(10.0) == 1.0  # above π → clamp high


class _FakeReader:
    """Returns queued readings, then None forever."""

    def __init__(self, readings):
        self._readings = list(readings)

    def __call__(self):
        if self._readings:
            return self._readings.pop(0)
        return None


class TestAzimuthDoASource:
    def test_conforms_to_percept_source_protocol(self):
        src = AzimuthDoASource(_FakeReader([]))
        assert isinstance(src, PerceptSource)

    def test_name_and_capabilities(self):
        src = AzimuthDoASource(_FakeReader([]), name="reachy:audio-doa")
        assert src.name == "reachy:audio-doa"
        assert src.capabilities == {"audio"}
        assert src.is_exhausted() is False

    def test_speech_reading_emits_audio_percept_with_azimuth(self):
        # DoA at π (right), speech present → azimuth +1 audio percept.
        src = AzimuthDoASource(_FakeReader([(math.pi, True)]), agent_id="reachy")
        p = src.next_percept()
        assert p is not None
        assert p.modality == "audio"
        assert p.metadata["azimuth"] == pytest.approx(1.0)
        assert p.context.agent_id == "reachy"

    def test_no_speech_returns_none(self):
        # A reading with is_speech_detected=False must NOT fabricate a direction.
        src = AzimuthDoASource(_FakeReader([(math.pi / 2, False)]))
        assert src.next_percept() is None

    def test_no_reading_returns_none(self):
        # Reader yields None (stream not ready) → non-blocking None.
        src = AzimuthDoASource(_FakeReader([]))
        assert src.next_percept() is None

    def test_reader_exception_is_swallowed_to_none(self):
        def _boom():
            raise RuntimeError("usb hiccup")

        src = AzimuthDoASource(_boom)
        assert src.next_percept() is None  # degrades, doesn't crash the loop

    def test_sequence_of_readings(self):
        src = AzimuthDoASource(_FakeReader([(0.0, True), (math.pi / 2, False), (math.pi, True)]))
        first = src.next_percept()
        second = src.next_percept()
        third = src.next_percept()
        assert first is not None and first.metadata["azimuth"] == pytest.approx(-1.0)
        assert second is None  # no speech
        assert third is not None and third.metadata["azimuth"] == pytest.approx(1.0)


class TestMakeAudioPercept:
    def test_modality_and_sensory_and_metadata(self):
        from maxim.agents.modality import SensoryModality

        p = make_audio_percept(-0.5, agent_id="a")
        assert p.modality == "audio"
        assert p.sensory.modality is SensoryModality.SOUND
        assert p.metadata["azimuth"] == pytest.approx(-0.5)
        assert p.context.channel == "internal"
        assert "azimuth" in p.content

    def test_extra_metadata_merges(self):
        p = make_audio_percept(0.0, metadata={"confidence": 0.9})
        assert p.metadata["azimuth"] == 0.0
        assert p.metadata["confidence"] == 0.9


class _FakeMedia:
    def __init__(self, doa_result):
        self._doa = doa_result
        self.started = False

    def start_recording(self):
        self.started = True

    def get_DoA(self):
        return self._doa


class _FakeMini:
    def __init__(self, media):
        self.media = media


class TestReachyAdapter:
    """Adapter glue, exercised with an injected fake ``mini`` (no hardware
    connection). The real ``get_DoA()`` lifecycle is verified on-device."""

    def test_adapter_starts_recording_and_wraps_get_doa(self):
        media = _FakeMedia((math.pi, True))
        reader = make_reachy_doa_reader(_FakeMini(media))
        assert media.started is True  # start_recording() called at setup
        assert reader() == (math.pi, True)

    def test_adapter_coerces_types(self):
        # get_DoA may return ints / numpy-ish values; reader normalizes to
        # (float, bool).
        media = _FakeMedia((3, 1))
        reader = make_reachy_doa_reader(_FakeMini(media))
        doa, speech = reader()
        assert isinstance(doa, float) and doa == 3.0
        assert speech is True

    def test_adapter_passes_through_none(self):
        media = _FakeMedia(None)
        reader = make_reachy_doa_reader(_FakeMini(media))
        assert reader() is None
