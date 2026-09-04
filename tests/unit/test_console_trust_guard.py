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

_TOKEN = "mxc_" + "t" * 43
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


@pytest.fixture()
def plain_app(monkeypatch, tmp_path):
    monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
    monkeypatch.delenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return build_app(None, auth_token=_TOKEN)


@pytest.fixture()
def listed_app(monkeypatch, tmp_path):
    # allowed_origins WITHOUT sandbox: the "non-local origins I trust" list
    # (the shape a deliberate tunnel exposure will use).
    monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
    monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", f"{_LISTED}/, https://second.example")
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return build_app(None, auth_token=_TOKEN)


# `mode=sim` answers 501 from dispatch without building a handle or touching
# an LLM — reaching it proves the request got PAST the guard.
_PROBE_BODY = {"mode": "sim", "input": "x"}


# ── Origin rule on state-changing requests ───────────────────────────────────


class TestHttpOrigin:
    def test_untrusted_origin_post_refused(self, plain_app):
        r = TestClient(plain_app).post(
            "/api/run", json=_PROBE_BODY, headers={**_AUTH, "origin": "https://evil.example"}
        )
        assert r.status_code == 403
        assert r.json()["detail"] == srv._ORIGIN_REFUSAL

    def test_null_origin_post_refused(self, plain_app):
        # Sandboxed iframes / file:// pages send the literal "null".
        r = TestClient(plain_app).post("/api/run", json=_PROBE_BODY, headers={**_AUTH, "origin": "null"})
        assert r.status_code == 403

    @pytest.mark.parametrize(
        "origin",
        ["http://127.0.0.1:8765", "http://localhost:8765", "http://localhost:5173", "http://[::1]:8765"],
    )
    def test_loopback_origin_post_accepted_any_port(self, plain_app, origin):
        r = TestClient(plain_app).post("/api/run", json=_PROBE_BODY, headers={**_AUTH, "origin": origin})
        assert r.status_code == 501  # reached mode dispatch

    def test_missing_origin_post_accepted(self, plain_app):
        # curl / native clients / the CLI send no Origin — browser-relay
        # protection, not authentication.
        assert TestClient(plain_app).post("/api/run", json=_PROBE_BODY, headers=_AUTH).status_code == 501

    def test_listed_origin_post_accepted(self, listed_app):
        r = TestClient(listed_app).post(
            "/api/run", json=_PROBE_BODY, headers={**_AUTH, "origin": _LISTED.upper() + "/"}
        )
        assert r.status_code == 501

    def test_unlisted_origin_still_refused_when_list_set(self, listed_app):
        r = TestClient(listed_app).post(
            "/api/run", json=_PROBE_BODY, headers={**_AUTH, "origin": "https://evil.example"}
        )
        assert r.status_code == 403

    def test_guard_fires_before_the_body_touches_config_writers(self, plain_app, monkeypatch):
        wrote: list[str] = []
        monkeypatch.setattr(
            "maxim.runtime.config_writer.apply_mesh_setup", lambda *a, **k: wrote.append("mesh") or ("p", "w")
        )
        r = TestClient(plain_app).post(
            "/api/setup/mesh",
            json={"leader_url": "https://attacker.example", "api_key": "k"},
            headers={**_AUTH, "origin": "https://evil.example"},
        )
        assert r.status_code == 403 and wrote == []

    def test_reads_are_not_origin_gated(self, plain_app):
        # Without CORS headers a cross-origin page cannot READ a response, so
        # GETs pass the Origin rule (the Host rule still covers rebinding).
        r = TestClient(plain_app).get("/api/identity", headers={**_AUTH, "origin": "https://evil.example"})
        assert r.status_code == 200

    def test_default_port_origin_matches_listed_bare(self, listed_app):
        # Browsers omit default ports from Origin; a client that sends one
        # must still match the bare listed form (canonicalization, both ways).
        r = TestClient(listed_app).post("/api/run", json=_PROBE_BODY, headers={**_AUTH, "origin": f"{_LISTED}:443"})
        assert r.status_code == 501

    def test_listed_default_port_matches_bare_origin(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
        monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", "https://app.example:443")
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        app = build_app(None, auth_token=_TOKEN)
        r = TestClient(app).post("/api/run", json=_PROBE_BODY, headers={**_AUTH, "origin": "https://app.example"})
        assert r.status_code == 501


class TestContentType:
    # The Origin-less belt: HTML forms cannot send application/json, and a
    # fetch() that sets it triggers a preflight this server never answers.

    def test_non_json_content_type_refused_on_mutations(self, plain_app):
        import json as _json

        r = TestClient(plain_app).post(
            "/api/run", content=_json.dumps(_PROBE_BODY), headers={**_AUTH, "content-type": "text/plain"}
        )
        assert r.status_code == 415
        assert r.json()["detail"] == srv._CONTENT_TYPE_REFUSAL

    def test_missing_content_type_refused_on_mutations(self, plain_app):
        import json as _json

        r = TestClient(plain_app).post("/api/run", content=_json.dumps(_PROBE_BODY), headers=_AUTH)
        assert r.status_code == 415

    def test_json_with_charset_suffix_accepted(self, plain_app):
        import json as _json

        r = TestClient(plain_app).post(
            "/api/run",
            content=_json.dumps(_PROBE_BODY),
            headers={**_AUTH, "content-type": "application/json; charset=utf-8"},
        )
        assert r.status_code == 501

    def test_reads_are_not_content_type_gated(self, plain_app):
        assert TestClient(plain_app).get("/api/identity", headers=_AUTH).status_code == 200


# ── Host rule on every request ───────────────────────────────────────────────


class TestHttpHost:
    @pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "attacker.localhost.example"])
    def test_unrecognized_host_refused(self, plain_app, host):
        r = TestClient(plain_app).get("/api/identity", headers={**_AUTH, "host": host})
        assert r.status_code == 400
        assert r.json()["detail"] == srv._HOST_REFUSAL

    @pytest.mark.parametrize(
        "host", ["127.0.0.1:8765", "localhost:8765", "localhost", "localhost.", "[::1]:8765", "testserver"]
    )
    def test_local_hosts_accepted(self, plain_app, host):
        # "localhost." (trailing FQDN dot) resolves to loopback; "testserver"
        # is TestClient's default, allowed only under pytest — see below.
        assert TestClient(plain_app).get("/api/identity", headers={**_AUTH, "host": host}).status_code == 200

    def test_testserver_allowance_is_pytest_scoped(self, monkeypatch):
        # In production (no PYTEST_CURRENT_TEST) a hostile LAN resolver could
        # rebind a single-label name, so the allowance must not ship — the
        # gate mirrors Django's setup_test_environment scoping (review fold,
        # cross-confirmed by both lenses).
        assert srv._host_allowed("testserver", srv._EMPTY_TRUST) is True  # under pytest
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        assert srv._host_allowed("testserver", srv._EMPTY_TRUST) is False
        assert srv._host_allowed("localhost", srv._EMPTY_TRUST) is True  # loopback unaffected

    def test_trailing_dot_does_not_relax_refusals(self, plain_app):
        assert TestClient(plain_app).get("/api/identity", headers={**_AUTH, "host": "evil.example."}).status_code == 400

    def test_listed_origin_host_accepted(self, listed_app):
        assert TestClient(listed_app).get("/api/identity", headers={**_AUTH, "host": "app.example"}).status_code == 200

    def test_host_rule_covers_mutating_routes_too(self, plain_app):
        r = TestClient(plain_app).post("/api/run", json=_PROBE_BODY, headers={**_AUTH, "host": "evil.example"})
        assert r.status_code == 400

    def test_host_rule_covers_the_static_root(self, plain_app):
        assert TestClient(plain_app).get("/", headers={**_AUTH, "host": "evil.example"}).status_code == 400


# ── /ws (middleware does not cover websockets — the guard is hand-applied) ──


class TestWs:
    def test_untrusted_origin_refused_before_accept(self, plain_app):
        with TestClient(plain_app) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws", headers={**_AUTH, "origin": "https://evil.example"}):
                    pass

    def test_loopback_origin_accepted(self, plain_app):
        with TestClient(plain_app) as client:
            with client.websocket_connect("/ws", headers={**_AUTH, "origin": "http://localhost:5173"}) as ws:
                assert ws.receive_json()["kind"] == "identity"

    def test_missing_origin_accepted(self, plain_app):
        with TestClient(plain_app) as client:
            with client.websocket_connect("/ws", headers=_AUTH) as ws:
                assert ws.receive_json()["kind"] == "identity"

    def test_listed_origin_accepted(self, listed_app):
        with TestClient(listed_app) as client:
            with client.websocket_connect("/ws", headers={**_AUTH, "origin": _LISTED}) as ws:
                assert ws.receive_json()["kind"] == "identity"

    def test_unrecognized_host_refused(self, plain_app):
        with TestClient(plain_app) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws", headers={**_AUTH, "host": "evil.example"}):
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
        other = build_app(None, auth_token=_TOKEN)
        assert other.state.trust.allowed_origins == frozenset()
        assert listed_app.state.trust.allowed_origins  # unchanged

    @pytest.mark.parametrize(
        ("header", "host"),
        [
            ("127.0.0.1:8765", "127.0.0.1"),
            ("[::1]:8765", "::1"),
            ("Localhost", "localhost"),
            ("localhost.:8765", "localhost"),
            ("", ""),
        ],
    )
    def test_request_host_parsing(self, header, host):
        assert srv._request_host(header) == host

    @pytest.mark.parametrize("bad", ["app.example", "https://b.example/path", "https://x:not-a-port", "https://"])
    def test_malformed_origin_entries_fail_loud_at_build(self, monkeypatch, tmp_path, bad):
        # A bare hostname / path-carrying / bad-port entry can never match a
        # browser Origin — silently half-applying it (Host relaxed, Origin
        # still refusing the operator's UI) is the hardest state to debug.
        # Same loud-at-build philosophy as the sandbox knobs (review fold,
        # cross-confirmed by both lenses).
        from maxim.runtime.config_loader import ConfigurationError

        monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
        monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", f"https://good.example, {bad}")
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        with pytest.raises(ConfigurationError, match="allowed_origins"):
            build_app(None, auth_token=_TOKEN)

    def test_default_ports_canonicalize_in_policy(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
        monkeypatch.setenv(
            "MAXIM_CONSOLE_ALLOWED_ORIGINS", "https://a.example:443, http://b.example:80, https://c.example:8443"
        )
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        trust = build_app(None, auth_token=_TOKEN).state.trust
        assert trust.allowed_origins == frozenset({"https://a.example", "http://b.example", "https://c.example:8443"})
        assert trust.allowed_hosts == frozenset({"a.example", "b.example", "c.example"})

    def test_non_http_scheme_origin_can_be_listed(self, monkeypatch, tmp_path):
        # A packaged native shell (e.g. Capacitor) sends a custom-scheme
        # Origin; the canonicalizer must not refuse it at build or at match.
        monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
        monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", "capacitor://app.example")
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        app = build_app(None, auth_token=_TOKEN)
        assert srv._origin_allowed("capacitor://app.example", app.state.trust) is True
