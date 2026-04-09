"""Tests for maxim.utils.paths — centralized path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _reset_path_caches():
    """Reset path caches between tests so env var mocks take effect."""
    from maxim.utils.paths import _reset_caches

    _reset_caches()
    yield
    _reset_caches()


class TestBundledData:
    """Tests for bundled (read-only) data paths."""

    def test_bundled_data_returns_path(self):
        from maxim.utils.paths import bundled_data

        result = bundled_data()
        assert isinstance(result, Path)
        assert result.name == "_data"

    def test_bundled_data_is_inside_package(self):
        from maxim.utils.paths import bundled_data

        result = bundled_data()
        # Should be under src/maxim/_data or installed package
        assert "maxim" in str(result)

    def test_bundled_templates_returns_templates_subdir(self):
        from maxim.utils.paths import bundled_templates

        result = bundled_templates()
        assert result.name == "templates"
        assert result.parent.name == "_data"

    def test_bundled_templates_dir_exists(self):
        from maxim.utils.paths import bundled_templates

        result = bundled_templates()
        assert result.is_dir(), f"Expected {result} to exist"

    def test_bundled_templates_contains_llm_json(self):
        from maxim.utils.paths import bundled_templates

        llm_json = bundled_templates() / "llm.json"
        assert llm_json.is_file(), f"Expected {llm_json} to exist"


class TestDataHome:
    """Tests for user data directory (~/.maxim/)."""

    def test_data_home_returns_path(self, tmp_path):
        from maxim.utils.paths import data_home

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path / "test_maxim")}):
            result = data_home()
            assert isinstance(result, Path)
            assert result.is_dir()

    def test_data_home_creates_directory(self, tmp_path):
        from maxim.utils.paths import data_home

        target = tmp_path / "new_dir"
        assert not target.exists()

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(target)}):
            result = data_home()
            assert result == target
            assert target.is_dir()

    def test_data_home_respects_env_var(self, tmp_path):
        from maxim.utils.paths import data_home

        custom_path = tmp_path / "custom_maxim_home"
        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(custom_path)}):
            result = data_home()
            assert result == custom_path

    def test_data_home_defaults_to_home_dir(self):
        from maxim.utils.paths import data_home

        with mock.patch.dict(os.environ, {}, clear=False):
            # Remove MAXIM_DATA_HOME if set
            os.environ.pop("MAXIM_DATA_HOME", None)
            result = data_home()
            assert result == Path.home() / ".maxim"


class TestSubdirectories:
    """Tests for specific subdirectory helpers."""

    def test_user_memory_creates_subdir(self, tmp_path):
        from maxim.utils.paths import user_memory

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            result = user_memory()
            assert result == tmp_path / "memory"
            assert result.is_dir()

    def test_sim_reports_creates_subdir(self, tmp_path):
        from maxim.utils.paths import sim_reports

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            result = sim_reports()
            assert result == tmp_path / "sim_reports"
            assert result.is_dir()

    def test_model_dir_creates_subdir(self, tmp_path):
        from maxim.utils.paths import model_dir

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            result = model_dir()
            assert result == tmp_path / "models"
            assert result.is_dir()

    def test_agent_data_creates_per_agent_subdir(self, tmp_path):
        from maxim.utils.paths import agent_data

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            result = agent_data("npc_guard")
            assert result == tmp_path / "agents" / "npc_guard"
            assert result.is_dir()

    def test_benchmarks_dir(self, tmp_path):
        from maxim.utils.paths import benchmarks_dir

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            result = benchmarks_dir()
            assert result == tmp_path / "benchmarks"
            assert result.is_dir()

    def test_provenance_dir(self, tmp_path):
        from maxim.utils.paths import provenance_dir

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            result = provenance_dir()
            assert result == tmp_path / "provenance"
            assert result.is_dir()

    def test_sessions_dir(self, tmp_path):
        from maxim.utils.paths import sessions_dir

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            result = sessions_dir()
            assert result == tmp_path / "sessions"
            assert result.is_dir()

    def test_planning_dir(self, tmp_path):
        from maxim.utils.paths import planning_dir

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            result = planning_dir()
            assert result == tmp_path / "planning"
            assert result.is_dir()


class TestResolveConfig:
    """Tests for config file resolution with bundled fallback."""

    def test_resolve_config_finds_bundled(self):
        from maxim.utils.paths import resolve_config

        # llm.json exists in bundled templates
        result = resolve_config("llm.json")
        assert result.is_file()
        assert result.name == "llm.json"

    def test_resolve_config_prefers_user_override(self, tmp_path):
        from maxim.utils.paths import resolve_config

        # Create a user override
        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            config_dir = tmp_path / "config"
            config_dir.mkdir()
            user_file = config_dir / "llm.json"
            user_file.write_text('{"custom": true}')

            result = resolve_config("llm.json")
            assert result == user_file

    def test_resolve_config_raises_for_missing(self, tmp_path):
        from maxim.utils.paths import resolve_config

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            with pytest.raises(FileNotFoundError, match="nonexistent_file.json"):
                resolve_config("nonexistent_file.json")


class TestResolveUserState:
    """Tests for user state file resolution."""

    def test_resolve_user_state_creates_parent(self, tmp_path):
        from maxim.utils.paths import resolve_user_state

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            result = resolve_user_state("util/nac_state.json")
            assert result == tmp_path / "util" / "nac_state.json"
            assert result.parent.is_dir()

    def test_resolve_user_state_nested(self, tmp_path):
        from maxim.utils.paths import resolve_user_state

        with mock.patch.dict(os.environ, {"MAXIM_DATA_HOME": str(tmp_path)}):
            result = resolve_user_state("memory/hippocampus.json")
            assert result == tmp_path / "memory" / "hippocampus.json"
            assert result.parent.is_dir()
