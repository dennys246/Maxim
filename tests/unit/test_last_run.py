"""Tests for maxim.utils.last_run — CLI invocation save/restore."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from maxim.utils.last_run import (
    should_save,
    save_last_run,
    load_last_run,
    load_all_runs,
    clear_last_run,
    format_last_run,
    format_all_runs,
)


@pytest.fixture(autouse=True)
def tmp_last_runs(tmp_path, monkeypatch):
    """Redirect last_runs.json to a temp file for every test."""
    path = tmp_path / ".maxim" / "last_runs.json"
    monkeypatch.setattr("maxim.utils.last_run._LAST_RUNS_PATH", path)
    return path


# ── should_save ─────────────────────────────────────────────────────

class TestShouldSave:
    def test_sim_is_saveable(self):
        assert should_save(["--sim", "agent", "--goal", "test"]) is True

    def test_mode_is_saveable(self):
        assert should_save(["--mode", "agentic"]) is True

    def test_generate_simulation_is_saveable(self):
        assert should_save(["--generate-simulation", "a scene"]) is True

    def test_help_not_saveable(self):
        assert should_save(["--help"]) is False

    def test_last_not_saveable(self):
        assert should_save(["--last"]) is False

    def test_show_last_not_saveable(self):
        assert should_save(["--show-last"]) is False

    def test_clear_memory_not_saveable(self):
        assert should_save(["--clear-memory"]) is False

    def test_bare_args_not_saveable(self):
        assert should_save(["--verbosity", "2"]) is False

    def test_skip_overrides_save(self):
        """--last in argv prevents saving even if --sim is present."""
        assert should_save(["--sim", "agent", "--last"]) is False


# ── save / load ─────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load(self):
        save_last_run(["--sim", "agent", "--goal", "test safety"])
        run = load_last_run(1)
        assert run is not None
        assert run["args"] == ["--sim", "agent", "--goal", "test safety"]
        assert "test safety" in run["command"]

    def test_load_empty_returns_none(self):
        assert load_last_run(1) is None

    def test_most_recent_first(self):
        save_last_run(["--sim", "agent", "--goal", "first"])
        save_last_run(["--sim", "agent", "--goal", "second"])
        runs = load_all_runs()
        assert len(runs) == 2
        assert "second" in runs[0]["command"]
        assert "first" in runs[1]["command"]

    def test_deduplication(self):
        save_last_run(["--sim", "agent", "--goal", "same"])
        save_last_run(["--sim", "agent", "--goal", "same"])
        runs = load_all_runs()
        assert len(runs) == 1

    def test_max_saved_runs(self):
        for i in range(7):
            save_last_run(["--sim", "agent", "--goal", f"run {i}"])
        runs = load_all_runs()
        assert len(runs) == 5  # _MAX_SAVED_RUNS
        # Most recent kept
        assert "run 6" in runs[0]["command"]

    def test_load_by_index(self):
        save_last_run(["--mode", "agentic"])
        save_last_run(["--sim", "agent", "--goal", "test"])
        assert "test" in load_last_run(1)["command"]
        assert "agentic" in load_last_run(2)["command"]

    def test_load_out_of_range(self):
        save_last_run(["--sim", "agent"])
        assert load_last_run(5) is None

    def test_args_with_spaces_quoted_in_command(self):
        save_last_run(["--sim", "agent", "--goal", "sweep boundaries"])
        run = load_last_run(1)
        assert '"sweep boundaries"' in run["command"]


# ── clear ───────────────────────────────────────────────────────────

class TestClear:
    def test_clear_returns_false_when_empty(self):
        assert clear_last_run() is False

    def test_clear_removes_file(self, tmp_last_runs):
        save_last_run(["--sim", "agent"])
        assert tmp_last_runs.is_file()
        assert clear_last_run() is True
        assert not tmp_last_runs.is_file()

    def test_load_after_clear(self):
        save_last_run(["--sim", "agent"])
        clear_last_run()
        assert load_last_run(1) is None


# ── format ──────────────────────────────────────────────────────────

class TestFormat:
    def test_format_last_run(self):
        entry = {"command": "maxim --sim agent", "timestamp": "2026-04-03 12:00:00"}
        out = format_last_run(entry, index=1)
        assert "[1]" in out
        assert "maxim --sim agent" in out

    def test_format_all_empty(self):
        assert "No saved runs" in format_all_runs()

    def test_format_all_with_runs(self):
        save_last_run(["--sim", "agent", "--goal", "test"])
        save_last_run(["--mode", "agentic"])
        out = format_all_runs()
        assert "[1]" in out
        assert "[2]" in out


# ── old format migration ───────────────────────────────────────────

class TestMigration:
    def test_migrates_single_dict(self, tmp_last_runs):
        """Old format was a single dict, not a list."""
        tmp_last_runs.parent.mkdir(parents=True, exist_ok=True)
        old = {"timestamp": "2026-01-01 00:00:00", "args": ["--sim", "agent"], "command": "maxim --sim agent"}
        tmp_last_runs.write_text(json.dumps(old))
        runs = load_all_runs()
        assert len(runs) == 1
        assert runs[0]["args"] == ["--sim", "agent"]
