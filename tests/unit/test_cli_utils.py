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


class TestSimModeResolution:
    """1.1 persona hard-remove (persona_cleanup_and_mode_transition.md
    Stages 3-5): --persona/--sim-persona are gone; normalize_args just
    guarantees args.sim_mode is set (default "generative")."""

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

    def test_no_flag_falls_back_to_generative(self):
        args = self._bare_args()
        normalize_args(args)
        assert args.sim_mode == "generative"

    def test_explicit_mode_preserved(self):
        args = self._bare_args(sim_mode="research")
        normalize_args(args)
        assert args.sim_mode == "research"

    def test_mode_does_not_warn(self, capsys, recwarn):
        args = self._bare_args(sim_mode="benchmark")
        normalize_args(args)
        captured = capsys.readouterr()
        assert "DeprecationWarning" not in captured.err
        assert not any(issubclass(w.category, DeprecationWarning) for w in recwarn.list)

    def test_empty_mode_falls_back_to_generative(self):
        args = self._bare_args(sim_mode="")
        normalize_args(args)
        assert args.sim_mode == "generative"


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

    def test_bio_system_globs_across_agent_dirs(self, tmp_path):
        # Bio-systems now persist per-agent under agents/<id>/; clearing a
        # bio-system must remove it for EVERY agent (the bug: it only ever
        # looked at the stale legacy util/ path).
        a1 = tmp_path / "agents" / "a1" / "hippocampus.json"
        a2 = tmp_path / "agents" / "a2" / "hippocampus.json"
        for p in (a1, a2):
            p.parent.mkdir(parents=True)
            p.write_text("{}")
        results = clear_memory("hippo", home_dir=str(tmp_path))
        assert results["hippo"] is True
        assert not a1.exists()
        assert not a2.exists()

    def test_bio_system_clears_legacy_path_too(self, tmp_path):
        # Back-compat: a pre-0.9 flat util/ file is still cleaned up.
        legacy = tmp_path / "util" / "nac_state.json"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("{}")
        results = clear_memory("nac", home_dir=str(tmp_path))
        assert results["nac"] is True
        assert not legacy.exists()

    def test_angular_gyrus_clears_per_agent(self, tmp_path):
        ag = tmp_path / "agents" / "cli_agent" / "angular_gyrus.json"
        ag.parent.mkdir(parents=True)
        ag.write_text("{}")
        results = clear_memory("angular_gyrus", home_dir=str(tmp_path))
        assert results["angular_gyrus"] is True
        assert not ag.exists()

    def test_planning_directory_is_removed(self, tmp_path):
        # planning maps to a directory — must rmtree, not unlink.
        plan_dir = tmp_path / "planning"
        plan_dir.mkdir()
        (plan_dir / "plan.json").write_text("{}")
        results = clear_memory("planning", home_dir=str(tmp_path))
        assert results["planning"] is True
        assert not plan_dir.exists()


class TestMemoryPaths:
    def test_has_expected_keys(self):
        assert "hippo" in MEMORY_PATHS
        assert "nac" in MEMORY_PATHS
        assert "atl" in MEMORY_PATHS
        assert "fear" in MEMORY_PATHS

    def test_bio_systems_target_agent_dirs(self):
        # Regression guard: bio-system entries must glob across agents/, not
        # point only at the stale legacy util/*_state.json layout.
        for key in ("hippo", "nac", "scn", "atl", "angular_gyrus"):
            assert key in MEMORY_PATHS
            assert any(p.startswith("agents/*/") for p in MEMORY_PATHS[key]), key


class TestLlmBackendAvailable:
    """D15: the bare `maxim` menu warns when no LLM backend is reachable."""

    _ENV = (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "MISTRAL_API_KEY",
        "DEEPSEEK_API_KEY",
        "MAXIM_LLM_ENABLED",
        "MAXIM_LANE_LARGE_REMOTE_URL",
    )

    def _clear(self, monkeypatch):
        for k in self._ENV:
            monkeypatch.delenv(k, raising=False)

    def test_false_when_nothing_available(self, monkeypatch):
        from maxim import cli

        self._clear(monkeypatch)
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        assert cli._llm_backend_available() is False

    def test_true_with_cloud_key(self, monkeypatch):
        from maxim import cli

        self._clear(monkeypatch)
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert cli._llm_backend_available() is True

    def test_true_with_llm_enabled(self, monkeypatch):
        from maxim import cli

        self._clear(monkeypatch)
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        monkeypatch.setenv("MAXIM_LLM_ENABLED", "1")
        assert cli._llm_backend_available() is True

    def test_true_with_local_backend(self, monkeypatch):
        from maxim import cli

        self._clear(monkeypatch)
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda name: object() if name == "llama_cpp" else None,
        )
        assert cli._llm_backend_available() is True


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
