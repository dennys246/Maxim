"""Tests for MPS VRAM detection on Apple Silicon (P2).

The pure helper `_mps_effective_vram_gb` can be tested directly with
simple inputs. The integration path through `detect_compute_resources`
requires mocking `torch.backends.mps`, `torch.cuda`, `platform.machine`,
and `psutil.virtual_memory` — covered in a second test class with
explicit patches.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from maxim.runtime.capabilities import (
    _is_apple_silicon,
    _mps_effective_vram_gb,
    detect_compute_resources,
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Each test starts without the headroom env var set."""
    saved = os.environ.pop("MAXIM_MPS_VRAM_HEADROOM", None)
    yield
    if saved is not None:
        os.environ["MAXIM_MPS_VRAM_HEADROOM"] = saved
    else:
        os.environ.pop("MAXIM_MPS_VRAM_HEADROOM", None)


class TestMpsEffectiveVramGb:
    def test_24gb_mac_default_headroom(self):
        """24GB Mac at the default 0.75 headroom factor should report
        18GB effective VRAM."""
        assert _mps_effective_vram_gb(24.0) == 18.0

    def test_8gb_mac_default_headroom(self):
        assert _mps_effective_vram_gb(8.0) == 6.0

    def test_16gb_mac_default_headroom(self):
        assert _mps_effective_vram_gb(16.0) == 12.0

    def test_ceiling_caps_huge_ram(self):
        """A 192GB M2 Ultra at 0.75 would report 144GB, but the
        default 64GB ceiling should clamp it down. This keeps the
        number meaningful — no currently supported quantized GGUF
        needs more than ~48GB."""
        result = _mps_effective_vram_gb(192.0)
        assert result == 64.0

    def test_explicit_headroom_override(self):
        """Callers can pass headroom_factor directly to bypass env."""
        assert _mps_effective_vram_gb(24.0, headroom_factor=0.5) == 12.0

    def test_explicit_ceiling_override(self):
        """Smaller ceiling for testing / tight environments."""
        assert _mps_effective_vram_gb(192.0, headroom_factor=0.75, ceiling_gb=32.0) == 32.0

    def test_env_var_override(self):
        """MAXIM_MPS_VRAM_HEADROOM overrides the default 0.75."""
        os.environ["MAXIM_MPS_VRAM_HEADROOM"] = "0.5"
        assert _mps_effective_vram_gb(24.0) == 12.0

    def test_env_var_invalid_falls_back_to_default(self):
        """Malformed env var values should fall back silently, not raise."""
        os.environ["MAXIM_MPS_VRAM_HEADROOM"] = "not-a-float"
        assert _mps_effective_vram_gb(24.0) == 18.0

    def test_clamps_below_minimum(self):
        """Reserving less than 25% of RAM for the OS risks crashes.
        The function clamps to 0.25 minimum regardless of env var."""
        os.environ["MAXIM_MPS_VRAM_HEADROOM"] = "0.95"
        # Should clamp DOWN to 0.85 (the max headroom allowed)
        result = _mps_effective_vram_gb(24.0)
        assert result == 24.0 * 0.85

    def test_clamps_above_maximum(self):
        """headroom < 0.25 is so generous it starves the OS; clamp up."""
        os.environ["MAXIM_MPS_VRAM_HEADROOM"] = "0.1"
        # Should clamp UP to 0.25 minimum
        result = _mps_effective_vram_gb(24.0)
        assert result == 24.0 * 0.25

    def test_zero_ram_returns_zero(self):
        assert _mps_effective_vram_gb(0.0) == 0.0

    def test_negative_ram_returns_zero(self):
        """Defensive: negative input is nonsense but must not crash."""
        assert _mps_effective_vram_gb(-4.0) == 0.0


class TestDetectComputeResourcesMps:
    """Integration tests through `detect_compute_resources` with mocked
    torch and platform detection. These verify the full path from
    `has_gpu=True, gpu_type="mps"` through to a populated `vram_gb`."""

    def test_apple_silicon_24gb_populates_vram(self):
        """The critical regression case: 24GB Mac should report
        (True, "mps", 18.0, 24.0) not (True, "mps", 0.0, 24.0)."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.cuda.device_count.return_value = 0
        fake_torch.backends.mps.is_available.return_value = True

        fake_psutil = MagicMock()
        fake_memory = MagicMock()
        fake_memory.total = 24 * (1024**3)
        fake_psutil.virtual_memory.return_value = fake_memory

        with (
            patch.dict("sys.modules", {"torch": fake_torch, "psutil": fake_psutil}),
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()

        assert has_gpu is True
        assert gpu_type == "mps"
        assert vram_gb == 18.0
        assert ram_gb == 24.0

    def test_intel_mac_mps_leaves_vram_zero(self):
        """Intel Macs with PyTorch MPS (deprecated, discrete AMD GPUs)
        get `has_gpu=True, gpu_type="mps"` but `vram_gb=0.0` because
        the unified-memory assumption doesn't hold there. User sees
        medium tier fallback, not large tier."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.cuda.device_count.return_value = 0
        fake_torch.backends.mps.is_available.return_value = True

        fake_psutil = MagicMock()
        fake_memory = MagicMock()
        fake_memory.total = 32 * (1024**3)
        fake_psutil.virtual_memory.return_value = fake_memory

        with (
            patch.dict("sys.modules", {"torch": fake_torch, "psutil": fake_psutil}),
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
        ):
            has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()

        assert has_gpu is True
        assert gpu_type == "mps"
        # Intel Mac: vram stays 0 even though MPS is available
        assert vram_gb == 0.0
        assert ram_gb == 32.0

    def test_cuda_still_works_unchanged(self):
        """CUDA path is unchanged — uses torch.cuda.get_device_properties
        regardless of platform."""
        fake_props = MagicMock()
        fake_props.name = "NVIDIA RTX 5080"
        fake_props.total_memory = 16 * (1024**3)

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 1
        fake_torch.cuda.get_device_properties.return_value = fake_props

        fake_psutil = MagicMock()
        fake_memory = MagicMock()
        fake_memory.total = 64 * (1024**3)
        fake_psutil.virtual_memory.return_value = fake_memory

        with (
            patch.dict("sys.modules", {"torch": fake_torch, "psutil": fake_psutil}),
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
        ):
            has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()

        assert has_gpu is True
        assert gpu_type == "NVIDIA RTX 5080"
        assert vram_gb == 16.0
        assert ram_gb == 64.0

    def test_no_gpu_linux_returns_cpu_only(self):
        """Headless Linux with no GPU reports has_gpu=False."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.cuda.device_count.return_value = 0
        fake_torch.backends.mps.is_available.return_value = False

        fake_psutil = MagicMock()
        fake_memory = MagicMock()
        fake_memory.total = 64 * (1024**3)
        fake_psutil.virtual_memory.return_value = fake_memory

        with (
            patch.dict("sys.modules", {"torch": fake_torch, "psutil": fake_psutil}),
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
        ):
            has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()

        assert has_gpu is False
        assert gpu_type is None
        assert vram_gb == 0.0
        assert ram_gb == 64.0


class TestIsAppleSilicon:
    def test_arm64_darwin(self):
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            assert _is_apple_silicon() is True

    def test_x86_64_darwin(self):
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
        ):
            assert _is_apple_silicon() is False

    def test_arm64_linux(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="arm64"),
        ):
            assert _is_apple_silicon() is False
