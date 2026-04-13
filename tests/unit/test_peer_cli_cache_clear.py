"""Tests that `maxim peer ...` commands clear the P6 probe cache.

The probe cache (`~/.maxim/util/last_probe_status.json`) is meant as a
short-TTL hint, not source of truth. Hot operations like
`maxim peer connect/forget/restart/update/llm` should drop the entry
they're invalidating so the next `maxim` startup re-probes the
freshly-changed leader instead of trusting a stale `outcome="ok"`.

Each test patches `_clear_probe_cache` (the helper in `peer/cli.py`)
and asserts the relevant command path calls it. We patch the helper
itself rather than the underlying `probe_cache` module so we can
distinguish "all-cache wipe" from "single-URL eviction" by inspecting
the call args.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.fixture
def fake_data_home(tmp_path, monkeypatch):
    home = tmp_path / "maxim_home"
    (home / "util").mkdir(parents=True)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
    from maxim.utils import paths

    paths._reset_caches()
    return home


@pytest.fixture
def fake_peer_cfg(tmp_path, monkeypatch):
    """Provide a stub peer config so commands that auto-detect the URL
    don't fail with 'No peer config' before reaching the cache-clear path."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    monkeypatch.setattr(
        "maxim.peer.config.peer_config_path",
        lambda: cfg_dir / "peer.yml",
    )
    from maxim.peer.config import PeerConfig, write_peer_config

    cfg = PeerConfig(url="https://leader.example.com/v1", api_key="k")
    write_peer_config(cfg)
    return cfg


# ─── _cmd_connect clears the cache (full wipe) ──────────────────────────────


class TestCmdConnectClearsCache:
    def test_connect_clears_full_cache_after_save(self, fake_data_home, monkeypatch):
        """A new connect URL means everything we know about other URLs is
        also potentially stale (the user is changing leader topology).
        Helper is called with no URL = full clear."""
        from maxim.peer import cli as peer_cli

        # Stub the connectivity test so we don't actually probe.
        monkeypatch.setattr(peer_cli, "_run_peer_test", lambda url, key: 0)

        with (
            patch.object(peer_cli, "_clear_probe_cache") as mock_clear,
            patch.object(
                peer_cli,
                "write_peer_config",
                return_value="/fake/peer.yml",
            ),
        ):
            rc = peer_cli._cmd_connect(["https://leader.example.com/v1", "--key", "k"])
        assert rc == 0
        mock_clear.assert_called_once()
        # Full clear -> called with no URL argument
        assert mock_clear.call_args.args == () or mock_clear.call_args.args == (None,)


# ─── _cmd_forget clears the cache (full wipe) ───────────────────────────────


class TestCmdForgetClearsCache:
    def test_forget_clears_cache_when_config_existed(self, fake_data_home):
        from maxim.peer import cli as peer_cli

        with (
            patch.object(peer_cli, "delete_peer_config", return_value=True),
            patch.object(peer_cli, "_clear_probe_cache") as mock_clear,
        ):
            rc = peer_cli._cmd_forget()
        assert rc == 0
        mock_clear.assert_called_once()

    def test_forget_skips_clear_when_no_config(self, fake_data_home):
        from maxim.peer import cli as peer_cli

        with (
            patch.object(peer_cli, "delete_peer_config", return_value=False),
            patch.object(peer_cli, "_clear_probe_cache") as mock_clear,
        ):
            peer_cli._cmd_forget()
        # No config = nothing was changed, no cache to invalidate
        mock_clear.assert_not_called()


# ─── _cmd_restart clears the cache for the leader URL ──────────────────────


class TestCmdRestartClearsCache:
    def test_restart_clears_cache_for_url(self, fake_data_home, fake_peer_cfg):
        from maxim.peer import cli as peer_cli

        # Make the HTTP call return a no-op restart response.
        from maxim.utils import http as _http

        fake_resp = _http.Response(
            status=200,
            headers={},
            content=json.dumps({"status": "noop"}).encode(),
            elapsed_ms=1.0,
            endpoint=_http._EXTERNAL_ENDPOINT,
            request_id="r",
        )
        with (
            patch("maxim.utils.http.fetch_url", return_value=fake_resp),
            patch.object(peer_cli, "_clear_probe_cache") as mock_clear,
        ):
            peer_cli._cmd_restart([])
        # Single-URL eviction (the leader's URL), not a full wipe
        mock_clear.assert_called_once()
        called_args = mock_clear.call_args.args
        assert called_args and "leader.example.com" in called_args[0]


# ─── _cmd_update clears the cache for the leader URL ───────────────────────


class TestCmdUpdateClearsCache:
    def test_update_clears_cache_for_url(self, fake_data_home, fake_peer_cfg):
        from maxim.peer import cli as peer_cli

        from maxim.utils import http as _http

        fake_resp = _http.Response(
            status=200,
            headers={},
            content=json.dumps({"status": "up_to_date"}).encode(),
            elapsed_ms=1.0,
            endpoint=_http._EXTERNAL_ENDPOINT,
            request_id="r",
        )
        with (
            patch("maxim.utils.http.fetch_url", return_value=fake_resp),
            patch.object(peer_cli, "_clear_probe_cache") as mock_clear,
        ):
            peer_cli._cmd_update([])
        mock_clear.assert_called_once()
        called_args = mock_clear.call_args.args
        assert called_args and "leader.example.com" in called_args[0]

    def test_update_dry_run_skips_clear(self, fake_data_home, fake_peer_cfg):
        """A --dry-run preview doesn't actually update the leader, so
        the cache shouldn't be invalidated."""
        from maxim.peer import cli as peer_cli

        from maxim.utils import http as _http

        fake_resp = _http.Response(
            status=200,
            headers={},
            content=json.dumps({"status": "up_to_date"}).encode(),
            elapsed_ms=1.0,
            endpoint=_http._EXTERNAL_ENDPOINT,
            request_id="r",
        )
        with (
            patch("maxim.utils.http.fetch_url", return_value=fake_resp),
            patch.object(peer_cli, "_clear_probe_cache") as mock_clear,
        ):
            peer_cli._cmd_update(["--dry-run"])
        mock_clear.assert_not_called()


# ─── _cmd_llm clears the cache for the leader URL ──────────────────────────


class TestCmdLlmClearsCache:
    def test_llm_swap_clears_cache_for_url(self, fake_data_home, fake_peer_cfg):
        from maxim.peer import cli as peer_cli

        from maxim.utils import http as _http

        fake_resp = _http.Response(
            status=200,
            headers={},
            content=json.dumps({"status": "swapped", "model": "qwen2.5-14b-instruct"}).encode(),
            elapsed_ms=1.0,
            endpoint=_http._EXTERNAL_ENDPOINT,
            request_id="r",
        )
        with (
            patch("maxim.utils.http.fetch_url", return_value=fake_resp),
            patch.object(peer_cli, "_clear_probe_cache") as mock_clear,
        ):
            peer_cli._cmd_llm(["qwen2.5-14b-instruct"])
        mock_clear.assert_called_once()
        called_args = mock_clear.call_args.args
        assert called_args and "leader.example.com" in called_args[0]


# ─── _clear_probe_cache helper itself ──────────────────────────────────────


class TestClearProbeCacheHelper:
    def test_helper_full_wipe_calls_clear_cache(self):
        from maxim.peer import cli as peer_cli

        with patch("maxim.runtime.probe_cache.clear_cache") as mock_full:
            peer_cli._clear_probe_cache()
        mock_full.assert_called_once()

    def test_helper_url_eviction_calls_clear_cache_for_url(self):
        from maxim.peer import cli as peer_cli

        with patch("maxim.runtime.probe_cache.clear_cache_for_url") as mock_one:
            peer_cli._clear_probe_cache("https://leader.example.com/v1")
        mock_one.assert_called_once_with("https://leader.example.com/v1")

    def test_helper_swallows_import_errors(self, monkeypatch):
        """Telemetry/cache failures must never break a peer command."""
        from maxim.peer import cli as peer_cli

        def _bad_import(*_args, **_kwargs):
            raise ImportError("simulated")

        monkeypatch.setattr("maxim.runtime.probe_cache.clear_cache", _bad_import)
        # Must not raise
        peer_cli._clear_probe_cache()
