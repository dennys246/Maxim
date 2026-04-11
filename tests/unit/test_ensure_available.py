"""Tests for the auto-download preflight introduced in P5.

`maxim.models.download.ensure_available` is the single entry point used
by `build_primary_router` to guarantee a profile's GGUF is on disk
before auto-spawn tries to load it. These tests cover the decision tree:

* file already present → no-op
* missing profile not in registry → graceful refusal
* disk full / soft budget exceeded → refusal with report
* tty + auto flag off → prompt path (yes / no / timeout)
* non-tty + auto flag off → hard fail with actionable error
* auto flag on → skip prompt + download
* concurrent lock contention → polite refusal
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from maxim.models import download as dl_mod


# ─── helpers ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch):
    monkeypatch.delenv("MAXIM_AUTO_DOWNLOAD_MODELS", raising=False)
    monkeypatch.delenv("MAXIM_DATA_BUDGET_GB", raising=False)


@pytest.fixture
def fake_data_home(tmp_path, monkeypatch):
    """Point ~/.maxim at a tmp dir so the file lock + storage report
    don't touch the developer machine's real state."""
    home = tmp_path / "maxim_home"
    home.mkdir()
    (home / "util").mkdir()
    monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
    # Reset storage cache so the new home is observed.
    from maxim.utils import storage as storage_mod

    storage_mod._reset_cache_for_tests()
    return home


# ─── ensure_available decision tree ──────────────────────────────────────────


class TestEnsureAvailable:
    def test_file_already_present_returns_true_without_download(self, fake_data_home):
        with (
            patch("maxim.runtime.llm_server.profile_has_local_file", return_value=True),
            patch.object(dl_mod, "download_llm") as mock_dl,
        ):
            assert dl_mod.ensure_available("mistral-7b-instruct-v0.2") is True
            mock_dl.assert_not_called()

    def test_unknown_profile_returns_false(self, fake_data_home, capsys):
        with patch("maxim.runtime.llm_server.profile_has_local_file", return_value=False):
            assert dl_mod.ensure_available("not-a-real-profile") is False
        captured = capsys.readouterr()
        assert "not in the LLM_MODELS registry" in captured.out

    def test_disk_full_refuses_with_report(self, fake_data_home, capsys):
        with (
            patch("maxim.runtime.llm_server.profile_has_local_file", return_value=False),
            patch(
                "maxim.utils.storage.can_download",
                return_value=(False, "insufficient disk space: 0.1 GB free"),
            ),
            patch.object(dl_mod, "download_llm") as mock_dl,
        ):
            assert dl_mod.ensure_available("mistral-7b-instruct-v0.2") is False
            mock_dl.assert_not_called()
        captured = capsys.readouterr()
        assert "insufficient disk space" in captured.out
        assert "Filesystem" in captured.out  # format_report header

    def test_auto_flag_skips_prompt_and_downloads(self, fake_data_home):
        with (
            patch("maxim.runtime.llm_server.profile_has_local_file", side_effect=[False, False]),
            patch("maxim.utils.storage.can_download", return_value=(True, "ok")),
            patch.object(dl_mod, "download_llm", return_value=True) as mock_dl,
            patch.object(dl_mod, "_prompt_yes_no_with_timeout") as mock_prompt,
        ):
            ok = dl_mod.ensure_available("mistral-7b-instruct-v0.2", auto=True)
            assert ok is True
            mock_dl.assert_called_once()
            mock_prompt.assert_not_called()

    def test_env_auto_download_skips_prompt(self, fake_data_home, monkeypatch):
        monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", "1")
        with (
            patch("maxim.runtime.llm_server.profile_has_local_file", side_effect=[False, False]),
            patch("maxim.utils.storage.can_download", return_value=(True, "ok")),
            patch.object(dl_mod, "download_llm", return_value=True),
            patch.object(dl_mod, "_prompt_yes_no_with_timeout") as mock_prompt,
        ):
            assert dl_mod.ensure_available("mistral-7b-instruct-v0.2") is True
            mock_prompt.assert_not_called()

    def test_interactive_yes_proceeds(self, fake_data_home):
        with (
            patch("maxim.runtime.llm_server.profile_has_local_file", side_effect=[False, False]),
            patch("maxim.utils.storage.can_download", return_value=(True, "ok")),
            patch.object(dl_mod, "_prompt_yes_no_with_timeout", return_value=True),
            patch.object(dl_mod, "download_llm", return_value=True) as mock_dl,
        ):
            ok = dl_mod.ensure_available("mistral-7b-instruct-v0.2", interactive=True)
            assert ok is True
            mock_dl.assert_called_once()

    def test_interactive_no_aborts(self, fake_data_home, capsys):
        with (
            patch("maxim.runtime.llm_server.profile_has_local_file", return_value=False),
            patch("maxim.utils.storage.can_download", return_value=(True, "ok")),
            patch.object(dl_mod, "_prompt_yes_no_with_timeout", return_value=False),
            patch.object(dl_mod, "download_llm") as mock_dl,
        ):
            ok = dl_mod.ensure_available("mistral-7b-instruct-v0.2", interactive=True)
            assert ok is False
            mock_dl.assert_not_called()
        assert "declined" in capsys.readouterr().out

    def test_interactive_timeout_aborts(self, fake_data_home, capsys):
        with (
            patch("maxim.runtime.llm_server.profile_has_local_file", return_value=False),
            patch("maxim.utils.storage.can_download", return_value=(True, "ok")),
            patch.object(dl_mod, "_prompt_yes_no_with_timeout", return_value=None),
            patch.object(dl_mod, "download_llm") as mock_dl,
        ):
            ok = dl_mod.ensure_available("mistral-7b-instruct-v0.2", interactive=True)
            assert ok is False
            mock_dl.assert_not_called()
        assert "timeout" in capsys.readouterr().out.lower()

    def test_non_tty_no_flag_hard_fails_with_actionable_error(self, fake_data_home, capsys):
        with (
            patch("maxim.runtime.llm_server.profile_has_local_file", return_value=False),
            patch("maxim.utils.storage.can_download", return_value=(True, "ok")),
            patch.object(dl_mod, "download_llm") as mock_dl,
        ):
            ok = dl_mod.ensure_available("mistral-7b-instruct-v0.2", interactive=False)
            assert ok is False
            mock_dl.assert_not_called()
        out = capsys.readouterr().out
        assert "--auto-download" in out
        assert "python -m maxim.models.download" in out

    def test_lock_contention_returns_polite_refusal(self, fake_data_home, capsys):
        """A LockContended escaping ensure_available's non-blocking acquire
        must surface as the user-facing 'another process downloading'
        message — not a stack trace, not a crash. We simulate the
        contention by patching ``file_lock`` directly because BSD flock
        on macOS lets the same process re-acquire its own advisory lock,
        which would otherwise make a real-thread test silently pass.
        """
        from contextlib import contextmanager

        from maxim.utils.filelock import LockContended

        @contextmanager
        def _always_contended(*_args, **_kwargs):
            raise LockContended("simulated")
            yield  # pragma: no cover — make it a generator for @contextmanager

        with (
            patch("maxim.runtime.llm_server.profile_has_local_file", return_value=False),
            patch("maxim.utils.storage.can_download", return_value=(True, "ok")),
            patch.object(dl_mod, "download_llm") as mock_dl,
            patch("maxim.utils.filelock.file_lock", _always_contended),
        ):
            ok = dl_mod.ensure_available("mistral-7b-instruct-v0.2", auto=True)
            assert ok is False
            mock_dl.assert_not_called()

        assert "Another maxim process is downloading" in capsys.readouterr().out

    def test_double_check_inside_lock_skips_redundant_download(self, fake_data_home):
        """Two processes pass the initial profile_has_local_file=False
        check, both contend for the lock; the loser sees the file appear
        between losing-the-prompt and acquiring the lock and skips the
        re-download."""
        # Sequence: outer-check (False) → re-check inside lock (True) → no download.
        side_effects = iter([False, True])
        with (
            patch(
                "maxim.runtime.llm_server.profile_has_local_file",
                side_effect=lambda *_: next(side_effects),
            ),
            patch("maxim.utils.storage.can_download", return_value=(True, "ok")),
            patch.object(dl_mod, "download_llm") as mock_dl,
        ):
            ok = dl_mod.ensure_available("mistral-7b-instruct-v0.2", auto=True)
            assert ok is True
            mock_dl.assert_not_called()


# ─── soft budget plumbing ────────────────────────────────────────────────────


class TestSoftBudget:
    def test_soft_budget_unset_returns_none(self):
        assert dl_mod._soft_budget_gb() is None

    def test_soft_budget_valid(self, monkeypatch):
        monkeypatch.setenv("MAXIM_DATA_BUDGET_GB", "20")
        assert dl_mod._soft_budget_gb() == 20.0

    def test_soft_budget_zero_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("MAXIM_DATA_BUDGET_GB", "0")
        assert dl_mod._soft_budget_gb() is None

    def test_soft_budget_invalid_returns_none(self, monkeypatch):
        monkeypatch.setenv("MAXIM_DATA_BUDGET_GB", "twenty gigs")
        assert dl_mod._soft_budget_gb() is None


# ─── --auto-download CLI plumbing ────────────────────────────────────────────


class TestAutoDownloadFlag:
    def test_normalize_args_sets_env_when_flag_set(self, monkeypatch):
        from argparse import Namespace

        from maxim.cli_utils import normalize_args

        monkeypatch.delenv("MAXIM_AUTO_DOWNLOAD_MODELS", raising=False)
        args = Namespace(
            audio="false",
            interactive=None,
            mode="exploration",
            epochs=0,
            language_model=None,
            llm_n_ctx=None,
            auto_download=True,
            cloud_fallback=None,
            cloud_lane=None,
            cloud_budget=None,
        )
        normalize_args(args)
        assert dl_mod._auto_download_enabled() is True

    def test_normalize_args_omits_env_when_flag_unset(self, monkeypatch):
        from argparse import Namespace

        from maxim.cli_utils import normalize_args

        monkeypatch.delenv("MAXIM_AUTO_DOWNLOAD_MODELS", raising=False)
        args = Namespace(
            audio="false",
            interactive=None,
            mode="exploration",
            epochs=0,
            language_model=None,
            llm_n_ctx=None,
            auto_download=False,
            cloud_fallback=None,
            cloud_lane=None,
            cloud_budget=None,
        )
        normalize_args(args)
        assert dl_mod._auto_download_enabled() is False
