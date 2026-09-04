"""Console trust guard — Host (DNS rebinding) + Origin (CSRF) browser-relay protection.

The console binds 127.0.0.1 and is unauthenticated; the trust guard closes the
two classes an attacker can relay THROUGH the operator's browser: cross-origin
"simple" POSTs (Starlette parses JSON regardless of Content-Type, so no
preflight protects the mutating routes) and DNS-rebinding reads (a public name
re-resolving to 127.0.0.1, arriving with a non-local Host). It is always on —
sandbox mode layers its stricter /ws origin-required rule on top. Every
refusal here is paired with the accepting counterpart (loopback / listed /
header-absent) so a guard that refuses everything cannot pass. Skips cleanly
without the console extra.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from maxim.console import server as srv  # noqa: E402
from maxim.console.server import build_app  # noqa: E402

_LISTED = "https://app.example"


@pytest.fixture()
def plain_app(monkeypatch, tmp_path):
    monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
    monkeypatch.delenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return build_app(None)


@pytest.fixture()
def listed_app(monkeypatch, tmp_path):
    # allowed_origins WITHOUT sandbox: the "non-local origins I trust" list
    # (the shape a deliberate tunnel exposure will use).
    monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
    monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", f"{_LISTED}/, https://second.example")
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return build_app(None)


# `mode=sim` answers 501 from dispatch without building a handle or touching
# an LLM — reaching it proves the request got PAST the guard.
_PROBE_BODY = {"mode": "sim", "input": "x"}


# ── Origin rule on state-changing requests ───────────────────────────────────


class TestHttpOrigin:
    def test_untrusted_origin_post_refused(self, plain_app):
        r = TestClient(plain_app).post("/api/run", json=_PROBE_BODY, headers={"origin": "https://evil.example"})
        assert r.status_code == 403
        assert r.json()["detail"] == srv._ORIGIN_REFUSAL

    def test_null_origin_post_refused(self, plain_app):
        # Sandboxed iframes / file:// pages send the literal "null".
        r = TestClient(plain_app).post("/api/run", json=_PROBE_BODY, headers={"origin": "null"})
        assert r.status_code == 403

    @pytest.mark.parametrize(
        "origin",
        ["http://127.0.0.1:8765", "http://localhost:8765", "http://localhost:5173", "http://[::1]:8765"],
    )
    def test_loopback_origin_post_accepted_any_port(self, plain_app, origin):
        r = TestClient(plain_app).post("/api/run", json=_PROBE_BODY, headers={"origin": origin})
        assert r.status_code == 501  # reached mode dispatch

    def test_missing_origin_post_accepted(self, plain_app):
        # curl / native clients / the CLI send no Origin — browser-relay
        # protection, not authentication.
        assert TestClient(plain_app).post("/api/run", json=_PROBE_BODY).status_code == 501

    def test_listed_origin_post_accepted(self, listed_app):
        r = TestClient(listed_app).post("/api/run", json=_PROBE_BODY, headers={"origin": _LISTED.upper() + "/"})
        assert r.status_code == 501

    def test_unlisted_origin_still_refused_when_list_set(self, listed_app):
        r = TestClient(listed_app).post("/api/run", json=_PROBE_BODY, headers={"origin": "https://evil.example"})
        assert r.status_code == 403

    def test_guard_fires_before_the_body_touches_config_writers(self, plain_app, monkeypatch):
        wrote: list[str] = []
        monkeypatch.setattr(
            "maxim.runtime.config_writer.apply_mesh_setup", lambda *a, **k: wrote.append("mesh") or ("p", "w")
        )
        r = TestClient(plain_app).post(
            "/api/setup/mesh",
            json={"leader_url": "https://attacker.example", "api_key": "k"},
            headers={"origin": "https://evil.example"},
        )
        assert r.status_code == 403 and wrote == []

    def test_reads_are_not_origin_gated(self, plain_app):
        # Without CORS headers a cross-origin page cannot READ a response, so
        # GETs pass the Origin rule (the Host rule still covers rebinding).
        r = TestClient(plain_app).get("/api/identity", headers={"origin": "https://evil.example"})
        assert r.status_code == 200


# ── Host rule on every request ───────────────────────────────────────────────


class TestHttpHost:
    @pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "attacker.localhost.example"])
    def test_unrecognized_host_refused(self, plain_app, host):
        r = TestClient(plain_app).get("/api/identity", headers={"host": host})
        assert r.status_code == 400
        assert r.json()["detail"] == srv._HOST_REFUSAL

    @pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765", "localhost", "[::1]:8765", "testserver"])
    def test_local_hosts_accepted(self, plain_app, host):
        # "testserver" is TestClient's default and single-label (not publicly
        # resolvable) — the documented allowance that keeps this suite honest.
        assert TestClient(plain_app).get("/api/identity", headers={"host": host}).status_code == 200

    def test_listed_origin_host_accepted(self, listed_app):
        assert TestClient(listed_app).get("/api/identity", headers={"host": "app.example"}).status_code == 200

    def test_host_rule_covers_mutating_routes_too(self, plain_app):
        r = TestClient(plain_app).post("/api/run", json=_PROBE_BODY, headers={"host": "evil.example"})
        assert r.status_code == 400

    def test_host_rule_covers_the_static_root(self, plain_app):
        assert TestClient(plain_app).get("/", headers={"host": "evil.example"}).status_code == 400


# ── /ws (middleware does not cover websockets — the guard is hand-applied) ──


class TestWs:
    def test_untrusted_origin_refused_before_accept(self, plain_app):
        with TestClient(plain_app) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}):
                    pass

    def test_loopback_origin_accepted(self, plain_app):
        with TestClient(plain_app) as client:
            with client.websocket_connect("/ws", headers={"origin": "http://localhost:5173"}) as ws:
                assert ws.receive_json()["kind"] == "identity"

    def test_missing_origin_accepted(self, plain_app):
        with TestClient(plain_app) as client:
            with client.websocket_connect("/ws") as ws:
                assert ws.receive_json()["kind"] == "identity"

    def test_listed_origin_accepted(self, listed_app):
        with TestClient(listed_app) as client:
            with client.websocket_connect("/ws", headers={"origin": _LISTED}) as ws:
                assert ws.receive_json()["kind"] == "identity"

    def test_unrecognized_host_refused(self, plain_app):
        with TestClient(plain_app) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws", headers={"host": "evil.example"}):
                    pass


# ── policy resolution ────────────────────────────────────────────────────────


class TestPolicy:
    def test_default_is_loopback_only(self, plain_app):
        trust = plain_app.state.trust
        assert trust.allowed_origins == frozenset() and trust.allowed_hosts == frozenset()

    def test_origins_normalize_and_hosts_derive(self, listed_app):
        trust = listed_app.state.trust
        assert trust.allowed_origins == frozenset({_LISTED, "https://second.example"})
        assert trust.allowed_hosts == frozenset({"app.example", "second.example"})

    def test_policy_is_per_app_not_global(self, listed_app, monkeypatch, tmp_path):
        monkeypatch.delenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", raising=False)
        other = build_app(None)
        assert other.state.trust.allowed_origins == frozenset()
        assert listed_app.state.trust.allowed_origins  # unchanged

    @pytest.mark.parametrize(
        ("header", "host"),
        [("127.0.0.1:8765", "127.0.0.1"), ("[::1]:8765", "::1"), ("Localhost", "localhost"), ("", "")],
    )
    def test_request_host_parsing(self, header, host):
        assert srv._request_host(header) == host
