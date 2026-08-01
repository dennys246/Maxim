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
    build_reachy_audio_orienting_source,
    make_reachy_doa_reader,
    make_reachy_rest_doa_reader,
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


class TestRestDoAReader:
    """The OFF-ROBOT reader — GET /api/state/doa. All exercised WITHOUT a robot or a
    network, via the injected `fetch` seam (the whole point: this must be provable
    with the hardware down)."""

    def test_returns_reader_that_calls_the_fetch_seam(self):
        seen = {}

        def fake_fetch(url, timeout):
            seen["url"] = url
            seen["timeout"] = timeout
            return (math.pi / 2, True)

        reader = make_reachy_rest_doa_reader("10.6.0.63", fetch=fake_fetch)
        assert reader() == (math.pi / 2, True)
        assert seen["url"] == "http://10.6.0.63:8000/api/state/doa"
        assert seen["timeout"] == 2.0

    def test_honors_port_and_timeout(self):
        seen = {}

        def fake_fetch(url, timeout):
            seen["url"], seen["timeout"] = url, timeout
            return None

        make_reachy_rest_doa_reader("host", port=9999, timeout=0.5, fetch=fake_fetch)()
        assert seen["url"] == "http://host:9999/api/state/doa"
        assert seen["timeout"] == 0.5

    def test_none_reading_passes_through(self):
        reader = make_reachy_rest_doa_reader("h", fetch=lambda u, t: None)
        assert reader() is None

    def test_feeds_azimuth_source_end_to_end(self):
        reader = make_reachy_rest_doa_reader("h", fetch=lambda u, t: (math.pi, True))
        src = AzimuthDoASource(reader)
        percept = src.next_percept()
        assert percept is not None
        assert percept.metadata["azimuth"] == doa_to_azimuth(math.pi)

    def test_no_speech_suppresses_the_percept(self):
        reader = make_reachy_rest_doa_reader("h", fetch=lambda u, t: (math.pi, False))
        assert AzimuthDoASource(reader).next_percept() is None

    def test_default_path_routes_through_maxim_http_not_raw_urllib(self):
        import maxim.utils.http as maxim_http

        calls = {}

        class _Resp:
            def json(self):
                return {"angle": math.pi, "speech_detected": True}

        orig = maxim_http.fetch_url
        maxim_http.fetch_url = lambda url, timeout=None: (calls.setdefault("url", url), _Resp())[1]
        try:
            reader = make_reachy_rest_doa_reader("10.0.0.5")
            assert reader() == (math.pi, True)
            assert calls["url"] == "http://10.0.0.5:8000/api/state/doa"
        finally:
            maxim_http.fetch_url = orig

    def test_default_path_swallows_fetch_errors_to_none(self):
        import maxim.utils.http as maxim_http

        orig = maxim_http.fetch_url

        def _boom(url, timeout=None):
            raise RuntimeError("network down")

        maxim_http.fetch_url = _boom
        try:
            assert make_reachy_rest_doa_reader("h")() is None
        finally:
            maxim_http.fetch_url = orig


class TestBuildReachyAudioOrientingSource:
    """The builder that chooses the DoA transport. All offline — no robot, no network."""

    def test_network_mode_with_host_builds_a_source(self):
        src = build_reachy_audio_orienting_source(connection_mode="network", host="10.6.0.63")
        assert isinstance(src, AzimuthDoASource)
        assert "audio" in src.capabilities

    def test_no_host_no_mini_returns_none(self):
        # Absent transport => absent source (capability-driven), NOT an error.
        assert build_reachy_audio_orienting_source(connection_mode="network", host=None) is None

    def test_localhost_mode_without_mini_returns_none(self):
        # localhost_only + no injected SDK object => no reader to build.
        assert build_reachy_audio_orienting_source(connection_mode="localhost_only", host="x") is None

    def test_injected_mini_uses_onboard_reader(self):
        # A fake SDK object with a media.get_DoA() => onboard path, no network.
        media = _FakeMedia((math.pi / 2, True))
        src = build_reachy_audio_orienting_source(mini=_FakeMini(media))
        assert isinstance(src, AzimuthDoASource)
        percept = src.next_percept()
        assert percept is not None
        assert percept.metadata["azimuth"] == doa_to_azimuth(math.pi / 2)

    def test_agent_id_and_name_thread_through(self):
        # name is observable directly; agent_id is observable on the emitted percept.
        src = build_reachy_audio_orienting_source(
            connection_mode="network", host="h", agent_id="reachy-01", name="my-audio"
        )
        assert src is not None
        assert src.name == "my-audio"
        src._reader = lambda: (math.pi, True)  # force a reading without a network
        percept = src.next_percept()
        assert percept is not None
        assert percept.context.agent_id == "reachy-01"  # agent_id lives on the PerceptContext

    def test_network_source_pulls_through_the_rest_reader(self):
        # End to end: builder -> AzimuthDoASource -> REST reader, with the fetch
        # seam faked so no network happens.
        import maxim.utils.http as maxim_http

        class _Resp:
            def json(self):
                return {"angle": math.pi, "speech_detected": True}

        orig = maxim_http.fetch_url
        maxim_http.fetch_url = lambda url, timeout=None: _Resp()
        try:
            src = build_reachy_audio_orienting_source(connection_mode="network", host="1.2.3.4")
            percept = src.next_percept()
            assert percept is not None
            assert percept.metadata["azimuth"] == doa_to_azimuth(math.pi)
        finally:
            maxim_http.fetch_url = orig


# ─── gated_azimuth (promoted from scripts/orient_backbone, Stage 1d) ───────


class TestGatedAzimuth:
    """Speech-gated median-of-k sampling — the library home of the script's
    signal hygiene (live_audio_orient_wiring.md Stage 1d). poll_s=0 keeps
    these tests instant."""

    def _seq_reader(self, readings):
        it = iter(readings)
        return lambda: next(it, None)

    def test_median_of_k_speech_samples(self):
        from maxim.embodiment.audio_localization import gated_azimuth

        # Three speech readings around front (pi/2): median wins.
        reader = self._seq_reader(
            [(math.pi / 2 + 0.3, True), (math.pi / 2, True), (math.pi / 2 - 0.3, True)]
        )
        az = gated_azimuth(reader, k=3, timeout_s=1.0, poll_s=0.0)
        assert az == pytest.approx(doa_to_azimuth(math.pi / 2))

    def test_non_speech_readings_are_gated_out(self):
        from maxim.embodiment.audio_localization import gated_azimuth

        # Loud non-speech transients must never fabricate a direction.
        reader = self._seq_reader(
            [(0.0, False), (math.pi, False), (math.pi / 2, True), (math.pi / 2, True), (math.pi / 2, True)]
        )
        az = gated_azimuth(reader, k=3, timeout_s=1.0, poll_s=0.0)
        assert az == pytest.approx(0.0)

    def test_silence_times_out_to_none(self):
        from maxim.embodiment.audio_localization import gated_azimuth

        az = gated_azimuth(lambda: None, k=3, timeout_s=0.05, poll_s=0.0)
        assert az is None

    def test_reader_exception_is_swallowed(self):
        from maxim.embodiment.audio_localization import gated_azimuth

        def _boom():
            raise RuntimeError("transport hiccup")

        az = gated_azimuth(_boom, k=1, timeout_s=0.05, poll_s=0.0)
        assert az is None

    def test_partial_samples_still_yield_a_median(self):
        from maxim.embodiment.audio_localization import gated_azimuth

        # Only one speech sample arrives before timeout — use it rather
        # than discarding a real reading.
        reader = self._seq_reader([(math.pi, True)])
        az = gated_azimuth(reader, k=3, timeout_s=0.05, poll_s=0.0)
        assert az == pytest.approx(1.0)


# ─── world_set_axis (Stage 2 multi-axis infra) ─────────────────────────────


class _AxisRoot:
    def __init__(self, sensors, vital):
        self.sensors = sensors
        self.vital_metrics = vital


class _AxisEmb:
    def __init__(self, root):
        self.root = root


class _RangedSensor:
    def __init__(self, lo, hi):
        self.reading_schema = {"range": [lo, hi]}


class TestWorldSetAxis:
    """The axis-generic helper behind world_set_azimuth. Elevation-ready by
    DATA: a new axis is one body-YAML sensor + one call — no AxisSpec type
    (the perception_pipeline_placement.md guardrail)."""

    def test_writes_declared_sensor(self):
        from maxim.embodiment.audio_localization import world_set_axis

        emb = _AxisEmb(_AxisRoot({"elevation": object()}, {"elevation": 0.0}))
        assert world_set_axis(emb, "elevation", 0.4) is True
        assert emb.root.vital_metrics["elevation"] == 0.4

    def test_clamps_to_declared_range(self):
        from maxim.embodiment.audio_localization import world_set_axis

        emb = _AxisEmb(_AxisRoot({"elevation": _RangedSensor(-0.5, 0.5)}, {"elevation": 0.0}))
        world_set_axis(emb, "elevation", 2.0)
        assert emb.root.vital_metrics["elevation"] == 0.5

    def test_default_range_used_when_sensor_has_none(self):
        from maxim.embodiment.audio_localization import world_set_axis

        emb = _AxisEmb(_AxisRoot({"elevation": object()}, {"elevation": 0.0}))
        world_set_axis(emb, "elevation", 2.0, default_range=(-1.0, 1.0))
        assert emb.root.vital_metrics["elevation"] == 1.0

    def test_fail_soft_without_sensor(self):
        from maxim.embodiment.audio_localization import world_set_axis

        emb = _AxisEmb(_AxisRoot({"azimuth": object()}, {"azimuth": 0.0}))
        assert world_set_axis(emb, "elevation", 0.4) is False
        assert "elevation" not in emb.root.vital_metrics

    def test_world_set_azimuth_delegates_and_keeps_unit_clamp(self):
        """The legacy entry point stays byte-compatible: clamps to [-1, 1]
        even for a sensor with no declared range (the pre-Stage-2 contract)."""
        from maxim.embodiment.audio_localization import world_set_azimuth

        emb = _AxisEmb(_AxisRoot({"azimuth": object()}, {"azimuth": 0.0}))
        assert world_set_azimuth(emb, 5.0) is True
        assert emb.root.vital_metrics["azimuth"] == 1.0
