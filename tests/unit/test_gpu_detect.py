"""Tests for maxim.utils.gpu_detect — lazy Blackwell GPU detection."""

from __future__ import annotations

import os
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _reset_detection_state():
    """Reset module-level detection state between tests."""
    from maxim.utils import gpu_detect

    gpu_detect._detected = None
    yield
    gpu_detect._detected = None


class TestEnsureBlackwellGuards:
    """Tests for GPU detection and guard application."""

    def test_cached_yes_applies_guards(self):
        from maxim.utils.gpu_detect import ensure_blackwell_guards

        with mock.patch.dict(os.environ, {"_MAXIM_BLACKWELL_CHECKED": "yes"}):
            result = ensure_blackwell_guards()
            assert result is True
            assert os.environ.get("GST_CUDA_NO_CUDA") == "1"

    def test_cached_no_skips_detection(self):
        from maxim.utils.gpu_detect import ensure_blackwell_guards

        with mock.patch.dict(os.environ, {"_MAXIM_BLACKWELL_CHECKED": "no"}, clear=False):
            result = ensure_blackwell_guards()
            assert result is False

    @mock.patch("maxim.utils.gpu_detect.subprocess.run")
    def test_detects_rtx_5080(self, mock_run):
        from maxim.utils.gpu_detect import ensure_blackwell_guards

        mock_run.return_value = mock.Mock(returncode=0, stdout="NVIDIA GeForce RTX 5080\n")

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("_MAXIM_BLACKWELL_CHECKED", None)
            result = ensure_blackwell_guards()
            assert result is True
            assert os.environ.get("GST_CUDA_NO_CUDA") == "1"

    @mock.patch("maxim.utils.gpu_detect.subprocess.run")
    def test_non_blackwell_gpu(self, mock_run):
        from maxim.utils.gpu_detect import ensure_blackwell_guards

        mock_run.return_value = mock.Mock(returncode=0, stdout="NVIDIA GeForce RTX 4090\n")

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("_MAXIM_BLACKWELL_CHECKED", None)
            result = ensure_blackwell_guards()
            assert result is False

    @mock.patch("maxim.utils.gpu_detect.subprocess.run")
    def test_nvidia_smi_not_found(self, mock_run):
        from maxim.utils.gpu_detect import ensure_blackwell_guards

        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("_MAXIM_BLACKWELL_CHECKED", None)
            result = ensure_blackwell_guards()
            assert result is False

    @mock.patch("maxim.utils.gpu_detect.subprocess.run")
    def test_nvidia_smi_timeout(self, mock_run):
        import subprocess
        from maxim.utils.gpu_detect import ensure_blackwell_guards

        mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 0.5)

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("_MAXIM_BLACKWELL_CHECKED", None)
            result = ensure_blackwell_guards()
            assert result is False

    def test_idempotent(self):
        from maxim.utils.gpu_detect import ensure_blackwell_guards

        with mock.patch.dict(os.environ, {"_MAXIM_BLACKWELL_CHECKED": "no"}, clear=False):
            result1 = ensure_blackwell_guards()
            result2 = ensure_blackwell_guards()
            assert result1 is False
            assert result2 is False

    def test_hide_cuda_sets_visible_devices(self):
        from maxim.utils.gpu_detect import ensure_blackwell_guards

        with mock.patch.dict(
            os.environ,
            {
                "_MAXIM_BLACKWELL_CHECKED": "yes",
                "MAXIM_BLACKWELL_HIDE_CUDA": "1",
            },
        ):
            ensure_blackwell_guards()
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


class TestIsBlackwellDetected:
    """Tests for the query function."""

    def test_returns_false_before_detection(self):
        from maxim.utils.gpu_detect import is_blackwell_detected

        with mock.patch.dict(os.environ, {"_MAXIM_BLACKWELL_CHECKED": "no"}, clear=False):
            assert is_blackwell_detected() is False

    def test_returns_true_after_detection(self):
        from maxim.utils.gpu_detect import is_blackwell_detected

        with mock.patch.dict(os.environ, {"_MAXIM_BLACKWELL_CHECKED": "yes"}, clear=False):
            assert is_blackwell_detected() is True
