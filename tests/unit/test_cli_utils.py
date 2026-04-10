"""Tests for cli_utils.py — extracted from cli.py."""

from __future__ import annotations

import argparse

from maxim.cli_utils import (
    normalize_epoch_value,
    normalize_args,
    gpu_available,
    clear_python_cache,
    clear_memory,
    MEMORY_PATHS,
)


class TestNormalizeEpochValue:
    def test_positive_int(self):
        assert normalize_epoch_value(5) == 5

    def test_zero(self):
        assert normalize_epoch_value(0) == 0

    def test_negative(self):
        assert normalize_epoch_value(-3) == 0

    def test_string_int(self):
        assert normalize_epoch_value("10") == 10

    def test_invalid_string(self):
        assert normalize_epoch_value("abc") == 0

    def test_none(self):
        assert normalize_epoch_value(None) == 0


class TestNormalizeArgs:
    def _make_args(self, **kwargs):
        defaults = {
            "audio": "true",
            "interactive": "true",
            "mode": "active",
            "epochs": 0,
            "language_model": None,
            "cloud_fallback": None,
            "cloud_lane": None,
            "cloud_budget": None,
            "segmentation_model": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_audio_true_variants(self):
        for val in ("true", "1", "yes", "on", "True", "YES"):
            args = self._make_args(audio=val)
            normalize_args(args)
            assert args.audio is True

    def test_audio_false_variants(self):
        for val in ("false", "0", "no", "off", "False"):
            args = self._make_args(audio=val)
            normalize_args(args)
            assert args.audio is False

    def test_audio_invalid_exits(self):
        args = self._make_args(audio="maybe")
        try:
            normalize_args(args)
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass

    def test_sleep_mode_forces_audio(self):
        args = self._make_args(mode="sleep", audio="false")
        normalize_args(args)
        assert args.audio is True

    def test_epochs_normalized(self):
        args = self._make_args(epochs=-5)
        normalize_args(args)
        assert args.epochs == 0


class TestGpuAvailable:
    def test_returns_bool(self):
        result = gpu_available()
        assert isinstance(result, bool)


class TestClearPythonCache:
    def test_returns_int(self, tmp_path):
        # Create a fake __pycache__
        cache = tmp_path / "pkg" / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "mod.cpython-312.pyc").write_bytes(b"fake")
        result = clear_python_cache(str(tmp_path))
        assert isinstance(result, int)
        assert result >= 1
        assert not cache.exists()


class TestClearMemory:
    def test_unknown_type(self):
        results = clear_memory("nonexistent_type", home_dir="/tmp/fake")
        assert results["nonexistent_type"] is False

    def test_all_keyword(self, tmp_path):
        # Create one of the known memory files
        nac_path = tmp_path / "util" / "nac_state.json"
        nac_path.parent.mkdir(parents=True)
        nac_path.write_text("{}")
        results = clear_memory("nac", home_dir=str(tmp_path))
        assert results["nac"] is True
        assert not nac_path.exists()

    def test_not_found(self, tmp_path):
        results = clear_memory("hippo", home_dir=str(tmp_path))
        assert results["hippo"] is False


class TestMemoryPaths:
    def test_has_expected_keys(self):
        assert "hippo" in MEMORY_PATHS
        assert "nac" in MEMORY_PATHS
        assert "atl" in MEMORY_PATHS
        assert "fear" in MEMORY_PATHS


class TestImportPaths:
    def test_import_from_cli(self):
        from maxim.cli import _normalize_epoch_value, _normalize_args

        assert callable(_normalize_epoch_value)
        assert callable(_normalize_args)

    def test_import_gpu_from_cli(self):
        from maxim.cli import _gpu_available, _check_gpu_status

        assert callable(_gpu_available)
        assert callable(_check_gpu_status)

    def test_import_clear_from_cli(self):
        from maxim.cli import _clear_python_cache, _clear_memory

        assert callable(_clear_python_cache)
        assert callable(_clear_memory)

    def test_import_memory_paths_from_cli(self):
        from maxim.cli import MEMORY_PATHS as mp

        assert isinstance(mp, dict)
