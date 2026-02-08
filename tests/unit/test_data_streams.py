"""Unit tests for data stream abstractions."""

from __future__ import annotations

import numpy as np
import pytest

from maxim.hardware.simulation.streams import MockAudioStream, MockVideoStream


class TestMockVideoStream:
    """Test MockVideoStream implementation."""

    def test_creation(self):
        """Can create MockVideoStream."""
        stream = MockVideoStream()

        assert stream.resolution == (640, 480)
        assert not stream.is_active

    def test_custom_resolution(self):
        """Can create with custom resolution."""
        stream = MockVideoStream(resolution=(320, 240))

        assert stream.resolution == (320, 240)

    def test_start_stop(self):
        """Can start and stop stream."""
        stream = MockVideoStream()

        assert not stream.is_active

        stream.start()
        assert stream.is_active

        stream.stop()
        assert not stream.is_active

    def test_get_frame_when_inactive(self):
        """Returns None when stream is inactive."""
        stream = MockVideoStream()

        frame = stream.get_frame_nonblocking()
        assert frame is None

    def test_get_frame_nonblocking(self):
        """Can get frame without blocking."""
        stream = MockVideoStream(resolution=(640, 480))
        stream.start()

        frame = stream.get_frame_nonblocking()

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

        stream.stop()

    def test_get_frame_blocking(self):
        """Can get frame with blocking."""
        stream = MockVideoStream(resolution=(320, 240))
        stream.start()

        frame = stream.get_frame(timeout=1.0)

        assert frame is not None
        assert frame.shape == (240, 320, 3)

        stream.stop()

    def test_custom_color(self):
        """Can specify base color."""
        stream = MockVideoStream(color=(255, 0, 0), add_noise=False)
        stream.start()

        frame = stream.get_frame_nonblocking()

        # Should be mostly blue (BGR format)
        assert frame[0, 0, 0] == 255  # Blue
        assert frame[0, 0, 1] == 0  # Green
        assert frame[0, 0, 2] == 0  # Red

        stream.stop()

    def test_noise_adds_variation(self):
        """Noise adds variation to frames."""
        stream = MockVideoStream(color=(128, 128, 128), add_noise=True)
        stream.start()

        frame1 = stream.get_frame_nonblocking()
        frame2 = stream.get_frame_nonblocking()

        # Frames should be slightly different due to noise
        assert not np.array_equal(frame1, frame2)

        stream.stop()


class TestMockAudioStream:
    """Test MockAudioStream implementation."""

    def test_creation(self):
        """Can create MockAudioStream."""
        stream = MockAudioStream()

        assert stream.input_sample_rate == 16000
        assert stream.output_sample_rate == 16000
        assert not stream.is_active

    def test_custom_sample_rate(self):
        """Can create with custom sample rate."""
        stream = MockAudioStream(
            input_sample_rate=48000,
            output_sample_rate=44100,
        )

        assert stream.input_sample_rate == 48000
        assert stream.output_sample_rate == 44100

    def test_start_stop(self):
        """Can start and stop stream."""
        stream = MockAudioStream()

        assert not stream.is_active

        stream.start()
        assert stream.is_active

        stream.stop()
        assert not stream.is_active

    def test_get_sample_when_inactive(self):
        """Returns None when stream is inactive."""
        stream = MockAudioStream()

        sample = stream.get_sample(timeout=0.01)
        assert sample is None

    def test_get_sample(self):
        """Can get audio sample."""
        stream = MockAudioStream(chunk_size=512)
        stream.start()

        sample = stream.get_sample(timeout=0.5)

        assert sample is not None
        assert isinstance(sample, np.ndarray)
        assert sample.dtype == np.float32
        assert len(sample) == 512

        stream.stop()

    def test_push_sample_when_inactive(self):
        """Returns False when stream is inactive."""
        stream = MockAudioStream()

        samples = np.zeros(100, dtype=np.float32)
        result = stream.push_sample(samples)

        assert result is False

    def test_push_sample(self):
        """Can push audio sample."""
        stream = MockAudioStream()
        stream.start()

        samples = np.zeros(100, dtype=np.float32)
        result = stream.push_sample(samples)

        assert result is True

        stream.stop()

    def test_samples_are_nearly_silent(self):
        """Generated samples are nearly silent."""
        stream = MockAudioStream(chunk_size=1024)
        stream.start()

        sample = stream.get_sample(timeout=0.5)

        # Should be very low amplitude (noise floor)
        assert np.max(np.abs(sample)) < 0.01

        stream.stop()


class TestStreamAbstraction:
    """Test that streams follow the abstract interface."""

    def test_video_stream_interface(self):
        """MockVideoStream implements VideoStream interface."""
        from maxim.hardware.streams import VideoStream

        stream = MockVideoStream()

        assert isinstance(stream, VideoStream)
        assert hasattr(stream, "get_frame")
        assert hasattr(stream, "get_frame_nonblocking")
        assert hasattr(stream, "resolution")
        assert hasattr(stream, "is_active")
        assert hasattr(stream, "start")
        assert hasattr(stream, "stop")

    def test_audio_stream_interface(self):
        """MockAudioStream implements AudioStream interface."""
        from maxim.hardware.streams import AudioStream

        stream = MockAudioStream()

        assert isinstance(stream, AudioStream)
        assert hasattr(stream, "get_sample")
        assert hasattr(stream, "push_sample")
        assert hasattr(stream, "input_sample_rate")
        assert hasattr(stream, "output_sample_rate")
        assert hasattr(stream, "is_active")
        assert hasattr(stream, "start")
        assert hasattr(stream, "stop")
