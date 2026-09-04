"""Device handoff (hardening decision A9) — the Reachy on-device console seam.

Two pieces, tested with accepting + refusing counterparts (the PR 1/2 suites'
non-vacuity rule): ``device_console_handoff`` (mint-or-reuse the standard
console token, return the ``/#token=`` fragment URL for Pollen's
``custom_app_url`` with a redacting repr — A7) and
``build_app(extra_trusted_origins=…)`` (the embedder's explicit admission of
its own LAN origin to the browser-relay trust guard; bearer auth unaffected).
The run_serve banner-flush test lives here too (same shipment): the ``#token=``
handoff must reach a REDIRECTED stdout before the server loop starts, proven
by a child that hard-exits (``os._exit``) so unflushed buffers are dropped,
not rescued at exit. Skips cleanly without the console extra.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from maxim.console.server import build_app, device_console_handoff  # noqa: E402
from maxim.runtime.config_loader import ConfigurationError  # noqa: E402
from maxim.tunnel import keys  # noqa: E402

_TOKEN = "mxc_" + "t" * 43
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}
_LAN = "http://10.6.0.63:8765"
_LAN_HOST = {"Host": "10.6.0.63:8765"}


@pytest.fixture()
def _isolated(monkeypatch, tmp_path):
    monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
    monkeypatch.delenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path


# ── the handoff object ───────────────────────────────────────────────────────


class TestDeviceConsoleHandoff:
    def test_mints_and_reuses_the_standard_token_file(self, _isolated):
        h = device_console_handoff("10.6.0.63", 8765)
        # The SAME file maxim serve uses — --show-token / --rotate-token and
        # the per-request disk re-read keep working on-device.
        path = keys.key_file_path(keys.CONSOLE_KEY_NAME)
        assert path.is_file() and (path.stat().st_mode & 0o777) == 0o600
        token = keys.read_console_token()
        assert h.origin == _LAN
        assert h.url == f"{_LAN}/#token={token}"
        assert device_console_handoff("10.6.0.63", 8765).url == h.url  # reuse, not re-mint

    def test_token_rides_the_fragment_never_path_or_query(self, _isolated):
        h = device_console_handoff("10.6.0.63", 8765)
        token = keys.read_console_token()
        before_fragment = h.url.split("#", 1)[0]
        assert token not in before_fragment  # a fragment never reaches access logs (A5)
        assert h.url.split("#", 1)[1] == f"token={token}"

    def test_repr_and_str_redact_the_token(self, _isolated):
        h = device_console_handoff("10.6.0.63", 8765)
        token = keys.read_console_token()
        for rendered in (repr(h), str(h), f"{h}"):
            assert token not in rendered
            assert "<redacted>" in rendered
        assert token in h.url  # …while the field itself still carries it (the point)

    def test_ipv6_host_brackets(self, _isolated):
        assert device_console_handoff("fe80::1", 8765).origin == "http://[fe80::1]:8765"

    def test_default_port_normalizes_like_a_browser_origin(self, _isolated):
        assert device_console_handoff("10.6.0.63", 80).origin == "http://10.6.0.63"

    @pytest.mark.parametrize("host", ["", "evil/path", "a#b"])
    def test_junk_host_raises_loudly(self, _isolated, host):
        with pytest.raises(ConfigurationError):
            device_console_handoff(host, 8765)

    def test_nothing_reaches_a_logger(self, _isolated, caplog):
        with caplog.at_level(logging.DEBUG):
            device_console_handoff("10.6.0.63", 8765)
        assert "mxc_" not in caplog.text  # A7: the URL is a return value, never a log line


# ── extra_trusted_origins on build_app ───────────────────────────────────────


class TestExtraTrustedOrigins:
    def test_lan_host_refused_without_the_param(self, _isolated):
        # The refusing counterpart: the guard's default posture is unchanged.
        c = TestClient(build_app(None, auth_token=_TOKEN))
        assert c.get("/api/identity", headers={**_AUTH, **_LAN_HOST}).status_code == 400

    def test_lan_host_served_with_the_param(self, _isolated):
        c = TestClient(build_app(None, auth_token=_TOKEN, extra_trusted_origins=[_LAN]))
        assert c.get("/api/identity", headers={**_AUTH, **_LAN_HOST}).status_code == 200

    def test_bearer_auth_is_not_widened(self, _isolated):
        # Admitting the origin widens the browser-relay guard ONLY: a
        # tokenless request from the admitted host still 401s.
        c = TestClient(build_app(None, auth_token=_TOKEN, extra_trusted_origins=[_LAN]))
        assert c.get("/api/identity", headers=_LAN_HOST).status_code == 401

    def test_post_origin_allowed_with_param_refused_without(self, _isolated):
        body = {"mode": "sim", "input": "x"}  # 501 from dispatch = got past the guard
        headers = {**_AUTH, **_LAN_HOST, "Origin": _LAN, "Content-Type": "application/json"}
        refused = TestClient(build_app(None, auth_token=_TOKEN))
        assert refused.post("/api/run", json=body, headers=headers).status_code == 400  # Host guard first
        allowed = TestClient(build_app(None, auth_token=_TOKEN, extra_trusted_origins=[_LAN]))
        assert allowed.post("/api/run", json=body, headers=headers).status_code == 501

    def test_ws_origin_allowed_with_param_refused_without(self, _isolated):
        headers = {**_AUTH, **_LAN_HOST, "Origin": _LAN}
        with pytest.raises(WebSocketDisconnect):
            with TestClient(build_app(None, auth_token=_TOKEN)).websocket_connect("/ws", headers=headers):
                pass
        with TestClient(build_app(None, auth_token=_TOKEN, extra_trusted_origins=[_LAN])).websocket_connect(
            "/ws", headers=headers
        ) as ws:
            assert ws.receive_json()["kind"] == "identity"

    def test_config_list_and_param_union(self, _isolated, monkeypatch):
        monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", "https://tunnel.example")
        c = TestClient(build_app(None, auth_token=_TOKEN, extra_trusted_origins=[_LAN]))
        for host in ({"Host": "tunnel.example"}, _LAN_HOST):
            assert c.get("/api/identity", headers={**_AUTH, **host}).status_code == 200

    def test_junk_entry_raises_at_build_time(self, _isolated):
        # Same loudness as a junk config entry: a bare hostname can never
        # match a browser Origin, and half-applying is the worst state.
        with pytest.raises(ConfigurationError):
            build_app(None, auth_token=_TOKEN, extra_trusted_origins=["10.6.0.63"])


# ── run_serve banner flush ───────────────────────────────────────────────────


class TestBannerFlush:
    def test_token_handoff_reaches_redirected_stdout_before_the_loop(self, tmp_path):
        """Stub uvicorn.run with os._exit: a hard exit drops unflushed stdio,
        so anything the parent captures was flushed BEFORE the server loop —
        exactly what a log file / launchd / the Reachy app runner sees. This
        fails against unflushed print() (verified while writing the fix).
        """
        code = textwrap.dedent(
            """
            import os
            import uvicorn
            uvicorn.run = lambda *a, **k: os._exit(7)
            from maxim.console.server import run_serve
            run_serve([])
            """
        )
        env = {k: v for k, v in os.environ.items() if k not in ("MAXIM_CONSOLE_SANDBOX", "PYTEST_CURRENT_TEST")}
        env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
        env["MAXIM_DATA_HOME"] = str(tmp_path / "data")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
        assert proc.returncode == 7, proc.stderr
        # Source checkout → no UI bundle → the curl-form handoff line. Either
        # handoff shape counts; what matters is a token line pre-loop.
        assert "Authorization: Bearer" in proc.stdout or "#token=" in proc.stdout
        assert "127.0.0.1" in proc.stdout
