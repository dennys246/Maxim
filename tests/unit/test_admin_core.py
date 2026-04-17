"""Regression tests for ``maxim.peer.admin_core`` (Plan 4 C3.5 + C3.6).

Tests the wire-level shape of ``update_on_target``, ``restart_on_target``,
and ``llm_swap_on_target``. Same pattern as ``test_peer_install.py``
(Plan 4 C3.3): patches ``maxim.utils.http.fetch_url`` and asserts the
endpoint URL, headers, JSON body, and exit code contract.

Each function is the single source of truth for its admin endpoint:

- ``/v1/admin/update`` — ``update_on_target``
- ``/v1/admin/restart`` — ``restart_on_target``
- ``/v1/admin/llm-swap`` — ``llm_swap_on_target``

CI grep in ``.github/workflows/test.yml`` enforces these endpoint strings
only appear in ``admin_core.py``, this test file, and ``leader_proxy.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _FakeResp:
    """Minimal Response stub."""

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeHTTPError(Exception):
    """Minimal HTTPError stub."""

    def __init__(self, status: int, fix_hint: str = "", response=None):
        self.status = status
        self.fix_hint = fix_hint
        self.response = response
        super().__init__(f"HTTP {status}")


@pytest.fixture
def no_probe_clear(monkeypatch):
    """Prevent probe cache from touching disk."""
    mock = MagicMock()
    monkeypatch.setattr("maxim.peer.cli._clear_probe_cache", mock)
    return mock


@pytest.fixture
def no_proxy_ping(monkeypatch):
    """Prevent proxy ping from making real HTTP calls."""
    mock = MagicMock(return_value=None)
    monkeypatch.setattr("maxim.peer.cli._check_proxy_ping", mock)
    return mock


# ─── update_on_target ─────────────────────────────────────────────────


class TestUpdateOnTarget:
    def test_updated_returns_0(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import update_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp({"status": "updated", "commits_applied": ["abc123 fix bug"]})
            rc = update_on_target("https://leader.example.com/v1", "sk-test")

        assert rc == 0
        out = capsys.readouterr().out
        assert "Updated!" in out
        assert "abc123 fix bug" in out
        # Verify endpoint shape
        call_args = mock.call_args
        assert "/v1/admin/update" in call_args[0][0]
        assert call_args[1]["json"]["branch"] == "main"
        assert call_args[1]["json"]["dry_run"] is False

    def test_already_up_to_date_returns_0(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import update_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp({"status": "up_to_date"})
            rc = update_on_target("https://leader.example.com/v1", "sk-test")

        assert rc == 0
        assert "up to date" in capsys.readouterr().out

    def test_preview_returns_0(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import update_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp({"status": "preview", "pending_commits": ["abc pending"]})
            rc = update_on_target("https://leader.example.com/v1", "sk-test", dry_run=True)

        assert rc == 0
        out = capsys.readouterr().out
        assert "pending" in out
        # dry_run should NOT clear probe cache
        no_probe_clear.assert_not_called()

    def test_custom_branch_and_force(self, no_probe_clear):
        from maxim.peer.admin_core import update_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp({"status": "up_to_date"})
            update_on_target(
                "https://leader.example.com/v1",
                "sk-test",
                branch="develop",
                force=True,
            )

        call_args = mock.call_args
        assert call_args[1]["json"]["branch"] == "develop"
        assert call_args[1]["json"]["force"] is True

    def test_auth_failure_returns_1(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import update_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            from maxim.utils import http as _http

            mock.side_effect = _http.HTTPError("test", status=401, fix_hint="check key")
            rc = update_on_target("https://leader.example.com/v1", "sk-test")

        assert rc == 1
        assert "Authentication failed" in capsys.readouterr().err

    def test_forbidden_returns_1(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import update_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            from maxim.utils import http as _http

            mock.side_effect = _http.HTTPError("test", status=403, fix_hint="disabled")
            rc = update_on_target("https://leader.example.com/v1", "sk-test")

        assert rc == 1
        assert "MAXIM_ALLOW_REMOTE_UPDATE" in capsys.readouterr().err

    def test_dirty_tree_409_returns_1(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import update_on_target

        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "dirty_files": ["foo.py"],
            "hint": "stash or commit",
        }

        with patch("maxim.utils.http.fetch_url") as mock:
            from maxim.utils import http as _http

            err = _http.HTTPError("test", status=409, fix_hint="dirty")
            err.response = fake_resp
            mock.side_effect = err
            rc = update_on_target("https://leader.example.com/v1", "sk-test")

        assert rc == 1
        err = capsys.readouterr().err
        assert "dirty working tree" in err
        assert "foo.py" in err

    def test_connection_failure_returns_1(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import update_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.side_effect = ConnectionError("refused")
            rc = update_on_target("https://leader.example.com/v1", "sk-test")

        assert rc == 1
        assert "Connection failed" in capsys.readouterr().err

    def test_url_v1_stripped_for_endpoint(self, no_probe_clear):
        """URL with /v1 suffix should strip it before composing endpoint."""
        from maxim.peer.admin_core import update_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp({"status": "up_to_date"})
            update_on_target("https://leader.example.com/v1", None)

        endpoint = mock.call_args[0][0]
        assert endpoint == "https://leader.example.com/v1/admin/update"
        # No double /v1/v1
        assert "/v1/v1" not in endpoint

    def test_no_auth_header_when_key_is_none(self, no_probe_clear):
        from maxim.peer.admin_core import update_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp({"status": "up_to_date"})
            update_on_target("https://leader.example.com/v1", None)

        headers = mock.call_args[1]["headers"]
        assert "Authorization" not in headers


# ─── restart_on_target ─────────────────────────────────────────────────


class TestRestartOnTarget:
    def test_restarting_returns_0(self, no_probe_clear, no_proxy_ping, capsys):
        from maxim.peer.admin_core import restart_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp({"status": "restarting", "uptime_s": 3600})
            rc = restart_on_target("https://leader.example.com/v1", "sk-test")

        assert rc == 0
        out = capsys.readouterr().out
        assert "restarting" in out.lower()
        # Verify endpoint
        call_args = mock.call_args
        assert "/v1/admin/restart" in call_args[0][0]
        assert call_args[1]["json"]["delay_s"] == 1.5

    def test_auth_failure_returns_1(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import restart_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            from maxim.utils import http as _http

            mock.side_effect = _http.HTTPError("test", status=401, fix_hint="check key")
            rc = restart_on_target("https://leader.example.com/v1", "sk-test")

        assert rc == 1
        assert "Authentication failed" in capsys.readouterr().err

    def test_forbidden_returns_1(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import restart_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            from maxim.utils import http as _http

            mock.side_effect = _http.HTTPError("test", status=403, fix_hint="disabled")
            rc = restart_on_target("https://leader.example.com/v1", "sk-test")

        assert rc == 1
        assert "MAXIM_ALLOW_REMOTE_UPDATE" in capsys.readouterr().err

    def test_clears_probe_cache(self, no_probe_clear, no_proxy_ping):
        from maxim.peer.admin_core import restart_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp({"status": "restarting"})
            restart_on_target("https://leader.example.com/v1", "sk-test")

        no_probe_clear.assert_called_once()

    def test_connection_failure_returns_1(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import restart_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.side_effect = ConnectionError("refused")
            rc = restart_on_target("https://leader.example.com/v1", "sk-test")

        assert rc == 1
        assert "Connection failed" in capsys.readouterr().err


# ─── llm_swap_on_target ───────────────────────────────────────────────


class TestLlmSwapOnTarget:
    def test_swapped_returns_0(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import llm_swap_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp(
                {
                    "status": "swapped",
                    "previous_model": "mistral-7b",
                    "model": "qwen2.5-14b",
                    "startup_s": 45.2,
                }
            )
            rc = llm_swap_on_target("https://leader.example.com/v1", "sk-test", "qwen2.5-14b")

        assert rc == 0
        out = capsys.readouterr().out
        assert "mistral-7b" in out
        assert "qwen2.5-14b" in out
        # Verify endpoint shape
        call_args = mock.call_args
        assert "/v1/admin/llm-swap" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "qwen2.5-14b"

    def test_already_running_returns_0(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import llm_swap_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp({"status": "already_running", "model": "qwen2.5-14b"})
            rc = llm_swap_on_target("https://leader.example.com/v1", "sk-test", "qwen2.5-14b")

        assert rc == 0
        assert "Already running" in capsys.readouterr().out

    def test_http_error_returns_1(self, no_probe_clear, capsys):
        from maxim.peer.admin_core import llm_swap_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            from maxim.utils import http as _http

            mock.side_effect = _http.HTTPError("test", status=500, fix_hint="server error")
            rc = llm_swap_on_target("https://leader.example.com/v1", "sk-test", "qwen2.5-14b")

        assert rc == 1
        assert "Error" in capsys.readouterr().err

    def test_clears_probe_cache(self, no_probe_clear):
        from maxim.peer.admin_core import llm_swap_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp({"status": "swapped", "model": "m"})
            llm_swap_on_target("https://leader.example.com/v1", "sk-test", "m")

        no_probe_clear.assert_called_once()

    def test_url_v1_stripped(self, no_probe_clear):
        from maxim.peer.admin_core import llm_swap_on_target

        with patch("maxim.utils.http.fetch_url") as mock:
            mock.return_value = _FakeResp({"status": "swapped", "model": "m"})
            llm_swap_on_target("https://leader.example.com/v1", None, "m")

        endpoint = mock.call_args[0][0]
        assert endpoint == "https://leader.example.com/v1/admin/llm-swap"
        assert "/v1/v1" not in endpoint
