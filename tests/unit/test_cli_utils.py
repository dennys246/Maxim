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


class TestPersonaModeResolution:
    """Stage 1 of docs/plans/persona_cleanup_and_mode_transition.md.

    --mode is the new flag, --persona is deprecated in 0.9 and removed
    in 1.1. After normalize_args, args.sim_persona is always set (the
    rest of the CLI treats it as the canonical dispatch slot).
    """

    def _bare_args(self, **kwargs):
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

    def test_neither_flag_falls_back_to_default(self):
        args = self._bare_args()
        normalize_args(args)
        assert args.sim_persona == "adversarial"

    def test_mode_only_sets_sim_persona(self):
        args = self._bare_args(sim_mode="researcher")
        normalize_args(args)
        assert args.sim_persona == "researcher"

    def test_mode_only_does_not_warn(self, capsys, recwarn):
        args = self._bare_args(sim_mode="cooperative")
        normalize_args(args)
        captured = capsys.readouterr()
        assert "DeprecationWarning" not in captured.err
        assert not any(issubclass(w.category, DeprecationWarning) for w in recwarn.list)

    def test_persona_only_emits_deprecation(self, capsys, recwarn):
        args = self._bare_args(sim_persona="researcher")
        normalize_args(args)
        # sim_persona preserved so dispatch keeps working
        assert args.sim_persona == "researcher"
        # Stderr line for human visibility
        captured = capsys.readouterr()
        assert "DeprecationWarning" in captured.err
        assert "--sim-mode" in captured.err
        # DeprecationWarning emitted for programmatic catch
        dep_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) == 1
        assert "1.1" in str(dep_warnings[0].message)

    def test_both_flags_consistent_prefer_mode_silently(self, capsys, recwarn):
        args = self._bare_args(sim_persona="adversarial", sim_mode="adversarial")
        normalize_args(args)
        assert args.sim_persona == "adversarial"
        # No conflict warning because values match; no deprecation either
        # because --sim-mode is the canonical path.
        captured = capsys.readouterr()
        assert "WARNING" not in captured.err
        assert "DeprecationWarning" not in captured.err
        assert not any(issubclass(w.category, DeprecationWarning) for w in recwarn.list)

    def test_both_flags_conflict_warns_and_prefers_mode(self, capsys):
        args = self._bare_args(sim_persona="researcher", sim_mode="adversarial")
        normalize_args(args)
        assert args.sim_persona == "adversarial"
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "--sim-mode" in captured.err and "--persona" in captured.err


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
