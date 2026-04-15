"""Regression tests for ``maxim peer install <extras> [url]``.

Plan 4 C3.3 commit 1 — locks the behavior of the existing
``_cmd_install`` verb BEFORE the C3.3 refactor lifts its body into a
shared ``_install_on_target(url, key, extras, packages)`` core. The C2
review caught the analog of this when the drain refactor almost broke
single-leader fallback; the regression guard goes in commit 1, the
refactor lift in commit 2, the new ``--node <name> install`` verb in
commit 3. Bisect surface stays clean.

Each test patches ``maxim.peer.cli._http.fetch_url`` (or the helpers it
delegates to) and asserts the wire-level shape:

- POST to ``<base>/v1/admin/install`` with ``/v1`` stripped from the
  URL when present, ``Content-Type: application/json``, and
  ``Authorization: Bearer <key>`` when a key is available.
- JSON body ``{"extras": [...], "packages": [...]}`` after the
  ``KNOWN_EXTRAS`` classifier splits the user-provided tokens.
- Sync ``status: ok`` returns 0 and clears the probe cache for the
  target URL.
- Async ``status: started`` polls ``/v1/debug/install-status`` until
  ``ok`` / ``error`` / 10-min deadline.
- Failure paths (no packages, 401, 403, 404, generic HTTP error,
  network exception) return their established exit codes and stderr
  shapes.

The lift in commit 2 must keep ALL of these tests green without
modification. Any test that needs to change to accommodate the lift is
a sign that the refactor is no longer behavior-preserving — STOP and
re-design instead of relaxing the test.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class _FakeResp:
    """Minimal ``maxim.utils.http.Response``-shaped stub.

    ``_cmd_install`` only touches ``.json()`` on the success path. The
    error path goes through ``HTTPError.response.text`` which we
    construct explicitly in the failure tests.
    """

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def stub_peer_config(monkeypatch):
    """Stub ``read_peer_config`` so URL/key auto-detection has a value
    to return without touching the filesystem.

    Returned object has ``.url`` and ``.api_key`` attributes matching
    the real ``PeerConfig`` shape used by ``_cmd_install``.
    """
    from maxim.peer import cli as peer_cli

    class _Cfg:
        url = "https://leader.example.com/v1"
        api_key = "test-key-abc"

    monkeypatch.setattr(peer_cli, "read_peer_config", lambda: _Cfg())
    return _Cfg


@pytest.fixture
def no_probe_clear(monkeypatch):
    """Prevent the probe cache clear from touching ``~/.maxim/util/``.
    Returns the mock so tests can assert call args.
    """
    from unittest.mock import MagicMock

    from maxim.peer import cli as peer_cli

    mock = MagicMock()
    monkeypatch.setattr(peer_cli, "_clear_probe_cache", mock)
    return mock


@pytest.fixture
def fast_polling(monkeypatch):
    """Make the async polling loop's ``time.sleep`` calls instant so
    async tests don't spend 5 seconds per iteration.

    ``_cmd_install`` does ``import time`` inline in the polling block,
    so we patch the module's attribute after the inline import has
    bound it. The simplest path is to patch ``time.sleep`` globally
    for the duration of the test, which is safe because no other code
    on this hot path relies on real sleep semantics.
    """
    import time

    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)


# ─── argv parsing + classification ──────────────────────────────────────


class TestInstallArgvParsing:
    """Lock the argv parser shape: ``http*`` → URL, ``--*`` → ignored,
    everything else → comma-split into packages, then KNOWN_EXTRAS
    classification.
    """

    def test_no_packages_returns_2_with_usage(self, capsys, no_probe_clear):
        from maxim.peer import cli as peer_cli

        rc = peer_cli._cmd_install([])
        assert rc == 2
        err = capsys.readouterr().err
        assert "Usage:" in err
        assert "extras:" in err

    def test_known_extras_classified_into_extras_list(
        self, stub_peer_config, no_probe_clear
    ):
        """``semantic`` is in KNOWN_EXTRAS → goes into ``extras``."""
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        captured: dict = {}

        def fake_fetch(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            return _FakeResp({"status": "ok"})

        with patch.object(_http, "fetch_url", side_effect=fake_fetch):
            rc = peer_cli._cmd_install(["semantic"])
        assert rc == 0
        assert captured["json"] == {"extras": ["semantic"], "packages": []}

    def test_unknown_package_classified_into_packages_list(
        self, stub_peer_config, no_probe_clear
    ):
        """``sentence-transformers`` is not in KNOWN_EXTRAS → packages."""
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        captured: dict = {}

        def fake_fetch(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _FakeResp({"status": "ok"})

        with patch.object(_http, "fetch_url", side_effect=fake_fetch):
            rc = peer_cli._cmd_install(["sentence-transformers"])
        assert rc == 0
        assert captured["json"] == {
            "extras": [],
            "packages": ["sentence-transformers"],
        }

    def test_comma_separated_mixed_extras_and_packages(
        self, stub_peer_config, no_probe_clear
    ):
        """``semantic,llm-torch,my-pkg`` → 2 extras + 1 package."""
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        captured: dict = {}

        def fake_fetch(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _FakeResp({"status": "ok"})

        with patch.object(_http, "fetch_url", side_effect=fake_fetch):
            rc = peer_cli._cmd_install(["semantic,llm-torch,my-pkg"])
        assert rc == 0
        assert captured["json"]["extras"] == ["semantic", "llm-torch"]
        assert captured["json"]["packages"] == ["my-pkg"]

    def test_unknown_double_dash_flag_ignored(
        self, stub_peer_config, no_probe_clear
    ):
        """``--future-flag`` → ignored, doesn't become a package name."""
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        captured: dict = {}

        def fake_fetch(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _FakeResp({"status": "ok"})

        with patch.object(_http, "fetch_url", side_effect=fake_fetch):
            rc = peer_cli._cmd_install(["--future-flag", "semantic"])
        assert rc == 0
        # --future-flag must NOT appear in the body.
        assert "--future-flag" not in captured["json"]["packages"]
        assert captured["json"]["extras"] == ["semantic"]

    def test_empty_string_in_comma_split_skipped(
        self, stub_peer_config, no_probe_clear
    ):
        """``semantic,,llm-torch`` (trailing/double comma) → no empty entries."""
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        captured: dict = {}

        def fake_fetch(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _FakeResp({"status": "ok"})

        with patch.object(_http, "fetch_url", side_effect=fake_fetch):
            rc = peer_cli._cmd_install(["semantic,,llm-torch"])
        assert rc == 0
        assert "" not in captured["json"]["extras"]
        assert "" not in captured["json"]["packages"]
        assert captured["json"]["extras"] == ["semantic", "llm-torch"]


# ─── URL + auth resolution ──────────────────────────────────────────────


class TestInstallUrlResolution:
    """Lock the URL/key resolution: positional ``http*`` wins, else
    ``read_peer_config()``. The base URL has its trailing ``/v1``
    stripped before composing ``/v1/admin/install``.
    """

    def test_positional_url_overrides_peer_config(
        self, stub_peer_config, no_probe_clear
    ):
        """An ``http*`` arg in argv beats the peer.yml-stored URL."""
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        captured: dict = {}

        def fake_fetch(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            return _FakeResp({"status": "ok"})

        with patch.object(_http, "fetch_url", side_effect=fake_fetch):
            rc = peer_cli._cmd_install(
                ["semantic", "https://other-leader.example.com/v1"],
            )
        assert rc == 0
        assert (
            captured["url"]
            == "https://other-leader.example.com/v1/admin/install"
        )
        # Positional URL path provides no key — no Authorization header.
        assert "Authorization" not in captured["headers"]

    def test_peer_config_fallback_includes_bearer(
        self, stub_peer_config, no_probe_clear
    ):
        """When no positional URL is given, read_peer_config() supplies
        the URL AND key; the request gets ``Authorization: Bearer``.
        """
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        captured: dict = {}

        def fake_fetch(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            return _FakeResp({"status": "ok"})

        with patch.object(_http, "fetch_url", side_effect=fake_fetch):
            rc = peer_cli._cmd_install(["semantic"])
        assert rc == 0
        assert captured["url"] == "https://leader.example.com/v1/admin/install"
        assert captured["headers"]["Authorization"] == "Bearer test-key-abc"
        assert captured["headers"]["Content-Type"] == "application/json"

    def test_no_peer_config_no_url_returns_1(self, capsys, monkeypatch):
        """No positional URL AND no peer.yml → exit 1 with 'No peer config'."""
        from maxim.peer import cli as peer_cli

        monkeypatch.setattr(peer_cli, "read_peer_config", lambda: None)
        rc = peer_cli._cmd_install(["semantic"])
        assert rc == 1
        assert "No peer config" in capsys.readouterr().err

    def test_url_without_v1_suffix_works(
        self, stub_peer_config, no_probe_clear
    ):
        """A bare ``https://host`` URL (no /v1) gets ``/v1/admin/install``
        appended directly — no double /v1.
        """
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        captured: dict = {}

        def fake_fetch(url, **kwargs):
            captured["url"] = url
            return _FakeResp({"status": "ok"})

        with patch.object(_http, "fetch_url", side_effect=fake_fetch):
            rc = peer_cli._cmd_install(["semantic", "https://bare.example.com"])
        assert rc == 0
        assert captured["url"] == "https://bare.example.com/v1/admin/install"

    def test_url_with_trailing_slash_is_normalized(
        self, stub_peer_config, no_probe_clear
    ):
        """Trailing slash is stripped before /v1 detection."""
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        captured: dict = {}

        def fake_fetch(url, **kwargs):
            captured["url"] = url
            return _FakeResp({"status": "ok"})

        with patch.object(_http, "fetch_url", side_effect=fake_fetch):
            rc = peer_cli._cmd_install(["semantic", "https://t.example.com/v1/"])
        assert rc == 0
        assert captured["url"] == "https://t.example.com/v1/admin/install"


# ─── sync success path ──────────────────────────────────────────────────


class TestInstallSyncSuccess:
    """``status: ok`` → exit 0 + clears probe cache for the URL."""

    def test_sync_ok_returns_0_and_clears_probe_cache(
        self, stub_peer_config, no_probe_clear, capsys
    ):
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        with patch.object(
            _http, "fetch_url", return_value=_FakeResp({"status": "ok"})
        ):
            rc = peer_cli._cmd_install(["semantic"])
        assert rc == 0
        assert "Installed successfully" in capsys.readouterr().out
        # Single-URL clear, not a full wipe.
        no_probe_clear.assert_called_once()
        call_args = no_probe_clear.call_args
        # Either positional or kwarg is fine — both shapes are valid
        # given the helper's signature ``(url=None)``.
        url_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        assert url_arg == "https://leader.example.com/v1"


# ─── async polling path ─────────────────────────────────────────────────


class TestInstallAsyncPolling:
    """``status: started`` → poll ``/v1/debug/install-status`` until
    ``ok`` / ``error`` / 10-min deadline.
    """

    def test_async_started_polls_until_ok(
        self, stub_peer_config, no_probe_clear, fast_polling, capsys
    ):
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        # First call (POST install) returns "started"; subsequent calls
        # (GET install-status) return "ok" with the installed list.
        responses = [
            _FakeResp({"status": "started"}),
            _FakeResp({"status": "ok", "installed": ["pymaxim[semantic]"]}),
        ]
        call_log: list[str] = []

        def fake_fetch(url, **kwargs):
            call_log.append(url)
            return responses.pop(0)

        with patch.object(_http, "fetch_url", side_effect=fake_fetch):
            rc = peer_cli._cmd_install(["semantic"])
        assert rc == 0
        assert call_log[0].endswith("/v1/admin/install")
        assert call_log[1].endswith("/v1/debug/install-status")
        out = capsys.readouterr().out
        assert "Install started on leader" in out
        assert "Installed successfully" in out

    def test_async_started_polls_until_error(
        self, stub_peer_config, no_probe_clear, fast_polling, capsys
    ):
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        responses = [
            _FakeResp({"status": "started"}),
            _FakeResp({"status": "error", "error": "pip resolution failed"}),
        ]

        def fake_fetch(url, **kwargs):
            return responses.pop(0)

        with patch.object(_http, "fetch_url", side_effect=fake_fetch):
            rc = peer_cli._cmd_install(["semantic"])
        assert rc == 1
        assert "pip resolution failed" in capsys.readouterr().err


# ─── HTTP failure paths ─────────────────────────────────────────────────


class TestInstallHttpFailures:
    """Lock the exit codes + stderr shapes for each ``HTTPError`` status
    the verb branches on.
    """

    def test_http_403_returns_1_with_remote_update_hint(
        self, stub_peer_config, no_probe_clear, capsys
    ):
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        err = _http.HTTPClientError(
            "peer", status=403, fix_hint="forbidden"
        )
        with patch.object(_http, "fetch_url", side_effect=err):
            rc = peer_cli._cmd_install(["semantic"])
        assert rc == 1
        msg = capsys.readouterr().err
        assert "Install disabled" in msg
        assert "MAXIM_ALLOW_REMOTE_UPDATE" in msg

    def test_http_401_returns_1_with_auth_hint(
        self, stub_peer_config, no_probe_clear, capsys
    ):
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        err = _http.HTTPAuthError("peer", status=401, fix_hint="auth failed")
        with patch.object(_http, "fetch_url", side_effect=err):
            rc = peer_cli._cmd_install(["semantic"])
        assert rc == 1
        assert "Authentication failed" in capsys.readouterr().err

    def test_http_404_returns_1_with_unsupported_hint(
        self, stub_peer_config, no_probe_clear, capsys
    ):
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        err = _http.HTTPClientError("peer", status=404, fix_hint="not found")
        with patch.object(_http, "fetch_url", side_effect=err):
            rc = peer_cli._cmd_install(["semantic"])
        assert rc == 1
        assert "does not support remote install" in capsys.readouterr().err

    def test_http_500_returns_1_with_generic_message(
        self, stub_peer_config, no_probe_clear, capsys
    ):
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        err = _http.HTTPServerError(
            "peer", status=500, fix_hint="upstream crashed"
        )
        with patch.object(_http, "fetch_url", side_effect=err):
            rc = peer_cli._cmd_install(["semantic"])
        assert rc == 1
        assert "Install failed (HTTP 500)" in capsys.readouterr().err

    def test_generic_exception_returns_1_with_message(
        self, stub_peer_config, no_probe_clear, capsys
    ):
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        with patch.object(
            _http, "fetch_url", side_effect=RuntimeError("network down")
        ):
            rc = peer_cli._cmd_install(["semantic"])
        assert rc == 1
        assert "Install failed: network down" in capsys.readouterr().err

    def test_install_failed_status_in_response_returns_1(
        self, stub_peer_config, no_probe_clear, capsys
    ):
        """200 OK but ``status`` is neither ``started`` nor ``ok`` → exit 1."""
        from maxim.peer import cli as peer_cli
        from maxim.utils import http as _http

        with patch.object(
            _http,
            "fetch_url",
            return_value=_FakeResp(
                {"status": "broken", "error": "leader confused"},
            ),
        ):
            rc = peer_cli._cmd_install(["semantic"])
        assert rc == 1
        assert "Install failed: leader confused" in capsys.readouterr().err
