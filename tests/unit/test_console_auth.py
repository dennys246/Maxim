"""Console bearer auth — always on, fail-closed; sandbox is the one exception.

Acceptance suite for hardening PR 2 (docs/plans/console_tunnel_hardening.md,
decisions A1–A8). Every refusal is paired with its accepting counterpart so a
guard that refuses everything cannot pass; the fail-closed test is the
anti-`leader_proxy._check_auth` trap (that check returns True on an EMPTY
key — the console must do the opposite). Skips cleanly without the console
extra.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from maxim.console import server as srv  # noqa: E402
from maxim.console.server import build_app  # noqa: E402
from maxim.tunnel import keys  # noqa: E402

_TOKEN = "mxc_" + "t" * 43
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


@pytest.fixture()
def authed_app(monkeypatch, tmp_path):
    monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
    monkeypatch.delenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return build_app(None, auth_token=_TOKEN)


@pytest.fixture()
def sandbox_app(monkeypatch, tmp_path):
    monkeypatch.setenv("MAXIM_CONSOLE_SANDBOX", "1")
    monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", "https://sandbox.example")
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return build_app(None, auth_token=_TOKEN)


# ── fail-closed (the anti-leader_proxy trap) ─────────────────────────────────


class TestFailClosed:
    def test_injected_none_refuses_even_a_bearer_header(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        app = build_app(None, auth_token=None)
        assert TestClient(app).get("/api/identity", headers=_AUTH).status_code == 401

    def test_absent_disk_token_refuses(self, monkeypatch, tmp_path):
        # Default token source is the on-disk file; no file = fail closed,
        # never an open console (design A2).
        monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        app = build_app(None)
        assert TestClient(app).get("/api/identity").status_code == 401
        assert TestClient(app).get("/api/identity", headers=_AUTH).status_code == 401

    def test_empty_bearer_credential_refused(self, authed_app):
        r = TestClient(authed_app).get("/api/identity", headers={"Authorization": "Bearer "})
        assert r.status_code == 401

    def test_unknown_scheme_is_a_clean_401(self, authed_app):
        # CC13 pattern: scheme parsed before credential — a future Signature
        # scheme is refused cleanly, never mistaken for a malformed Bearer.
        r = TestClient(authed_app).get("/api/identity", headers={"Authorization": f"Signature {_TOKEN}"})
        assert r.status_code == 401


# ── every covered surface: 401 without, serves with ──────────────────────────


class TestCoverage:
    @pytest.mark.parametrize("path", ["/api/identity", "/api/campaigns", "/docs", "/openapi.json"])
    def test_reads_401_without_token(self, authed_app, path):
        r = TestClient(authed_app).get(path)
        assert r.status_code == 401
        assert r.headers.get("www-authenticate") == "Bearer"

    @pytest.mark.parametrize("path", ["/api/identity", "/api/campaigns", "/docs", "/openapi.json"])
    def test_reads_serve_with_token(self, authed_app, path):
        assert TestClient(authed_app).get(path, headers=_AUTH).status_code == 200

    def test_mutation_401_without_token(self, authed_app):
        r = TestClient(authed_app).post("/api/run", json={"mode": "sim", "input": "x"})
        assert r.status_code == 401
        assert r.json()["detail"] == srv._AUTH_REFUSAL

    def test_mutation_serves_with_token(self, authed_app):
        r = TestClient(authed_app).post("/api/run", json={"mode": "sim", "input": "x"}, headers=_AUTH)
        assert r.status_code == 501  # reached mode dispatch

    def test_auth_runs_before_the_csrf_belts(self, authed_app):
        # An unauthenticated caller learns nothing about origin/content-type
        # policy: 401 fires first (middleware order — design A3).
        r = TestClient(authed_app).post(
            "/api/run", json={"mode": "sim", "input": "x"}, headers={"origin": "https://evil.example"}
        )
        assert r.status_code == 401

    def test_wrong_token_refused(self, authed_app):
        r = TestClient(authed_app).get("/api/identity", headers={"Authorization": "Bearer mxc_wrong"})
        assert r.status_code == 401


# ── the tokenless surfaces ───────────────────────────────────────────────────


class TestTokenless:
    def test_hello_reports_bearer(self, authed_app):
        r = TestClient(authed_app).get("/api/hello")
        assert r.status_code == 200
        body = r.json()
        assert body["auth"] == "bearer"
        assert body["contract_version"] == srv.CONSOLE_CONTRACT_VERSION

    def test_hello_reports_none_under_sandbox(self, sandbox_app):
        assert TestClient(sandbox_app).get("/api/hello").json()["auth"] == "none"

    def test_static_root_is_tokenless(self, authed_app):
        assert TestClient(authed_app).get("/").status_code == 200  # the no-UI page


# ── /ws: both transports, refused before accept ──────────────────────────────


class TestWs:
    def test_no_credentials_refused_before_accept(self, authed_app):
        with TestClient(authed_app) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws"):
                    pass

    def test_authorization_header_accepted(self, authed_app):
        with TestClient(authed_app) as client:
            with client.websocket_connect("/ws", headers=_AUTH) as ws:
                assert ws.receive_json()["kind"] == "identity"

    def test_bearer_subprotocol_accepted_and_echoed(self, authed_app):
        # The browser transport: no upgrade headers, so the token rides
        # Sec-WebSocket-Protocol beside the app subprotocol (design A4).
        with TestClient(authed_app) as client:
            with client.websocket_connect(
                "/ws", subprotocols=[srv._WS_APP_SUBPROTOCOL, f"{srv._WS_BEARER_PREFIX}{_TOKEN}"]
            ) as ws:
                assert ws.accepted_subprotocol == srv._WS_APP_SUBPROTOCOL
                assert ws.receive_json()["kind"] == "identity"

    def test_wrong_subprotocol_token_refused(self, authed_app):
        with TestClient(authed_app) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(
                    "/ws", subprotocols=[srv._WS_APP_SUBPROTOCOL, f"{srv._WS_BEARER_PREFIX}mxc_wrong"]
                ):
                    pass


# ── sandbox negative control ─────────────────────────────────────────────────


class TestSandbox:
    def test_no_token_demanded_under_sandbox(self, sandbox_app):
        # The proxy owns that edge; the engine deliberately demands nothing.
        assert TestClient(sandbox_app).get("/api/identity").status_code == 200

    def test_sandbox_ws_still_requires_listed_origin_not_token(self, sandbox_app):
        with TestClient(sandbox_app) as client:
            with client.websocket_connect("/ws", headers={"origin": "https://sandbox.example"}) as ws:
                assert ws.receive_json()["kind"] == "identity"


# ── the token itself: file, shape, rotation ──────────────────────────────────


class TestToken:
    def test_console_token_is_a_separate_prefixed_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        token = keys.ensure_console_token()
        assert token.startswith(keys.CONSOLE_TOKEN_PREFIX)
        assert keys.key_file_path(keys.CONSOLE_KEY_NAME).is_file()
        assert not keys.key_file_path().is_file()  # the mesh key was NOT created
        assert keys.ensure_console_token() == token  # stable across calls
        mode = keys.key_file_path(keys.CONSOLE_KEY_NAME).stat().st_mode & 0o777
        assert mode == 0o600

    def test_mesh_key_path_unchanged(self, monkeypatch, tmp_path):
        # The named-key generalization must leave the mesh key byte-identical.
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        assert keys.key_file_path().name == "api_key"
        assert not keys.ensure_key().startswith(keys.CONSOLE_TOKEN_PREFIX)

    def test_rotation_invalidates_on_the_next_request(self, monkeypatch, tmp_path):
        # Disk-sourced tokens are re-read per request (design A7): rotating
        # from a second terminal logs devices out without a server restart.
        monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        old = keys.ensure_console_token()
        app = build_app(None)  # default: disk-sourced
        client = TestClient(app)
        old_auth = {"Authorization": f"Bearer {old}"}
        assert client.get("/api/identity", headers=old_auth).status_code == 200
        new = keys.rotate_console_token()
        assert client.get("/api/identity", headers=old_auth).status_code == 401
        assert client.get("/api/identity", headers={"Authorization": f"Bearer {new}"}).status_code == 200


# ── the token never reaches a log line ───────────────────────────────────────


class TestLogHygiene:
    def test_token_absent_from_logs_on_success_and_refusal(self, authed_app, caplog):
        with caplog.at_level(logging.DEBUG):
            c = TestClient(authed_app)
            assert c.get("/api/identity", headers=_AUTH).status_code == 200
            assert c.get("/api/identity").status_code == 401
            assert c.get("/api/identity", headers={"Authorization": "Bearer mxc_wrong"}).status_code == 401
        assert _TOKEN not in caplog.text
        assert "mxc_" not in caplog.text
