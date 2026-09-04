"""Console sandbox mode — the host-acting surfaces close, nothing else changes.

``maxim serve`` stays localhost-only and unauthenticated; sandbox mode
(``MAXIM_CONSOLE_SANDBOX=1``, read once at ``build_app``) is the engine-side
half of running it for an anonymous visitor behind an authenticating proxy:
it refuses the three surfaces that act on the HOST (probe-by-URL, mesh setup,
diagnose), refuses ``/ws`` upgrades from unlisted origins, and caps run input.
Every guard here is paired with its negative control — the same call with the
mode off — so a vacuous 403 cannot pass. Skips cleanly without the console extra.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from maxim.console import server as srv  # noqa: E402
from maxim.console.server import build_app  # noqa: E402
from maxim.runtime.config_loader import ConfigurationError  # noqa: E402

_ORIGIN = "https://sandbox.example"


@pytest.fixture()
def sandbox_app(monkeypatch, tmp_path):
    monkeypatch.setenv("MAXIM_CONSOLE_SANDBOX", "1")
    monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", f"{_ORIGIN}/, https://second.example")
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    app = build_app(None)
    assert app.state.sandbox is not None
    return app


@pytest.fixture()
def plain_app(monkeypatch, tmp_path):
    monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    app = build_app(None)
    assert app.state.sandbox is None
    return app


# ── the switch itself ─────────────────────────────────────────────────────────


class TestSwitch:
    def test_default_is_off(self, plain_app):
        assert plain_app.state.sandbox is None

    @pytest.mark.parametrize("raw", ["1", "true", "YES", "on"])
    def test_truthy_spellings(self, monkeypatch, raw):
        # The switch is `console.sandbox`, coerced by the config loader —
        # the env form shares the loader's truthy spellings.
        monkeypatch.setenv("MAXIM_CONSOLE_SANDBOX", raw)
        monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", _ORIGIN)
        assert srv._sandbox_policy() is not None

    def test_falsy_spelling_is_off(self, monkeypatch):
        monkeypatch.setenv("MAXIM_CONSOLE_SANDBOX", "0")
        assert srv._sandbox_policy() is None

    def test_origins_are_normalized(self, monkeypatch):
        monkeypatch.setenv("MAXIM_CONSOLE_SANDBOX", "1")
        monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", " HTTPS://Sandbox.Example/ ,, https://b.example")
        pol = srv._sandbox_policy()
        assert pol is not None
        assert pol.allowed_origins == frozenset({"https://sandbox.example", "https://b.example"})
        assert pol.max_input_chars == srv._SANDBOX_DEFAULT_MAX_INPUT_CHARS

    @pytest.mark.parametrize("raw", ["abc", "0", "-5"])
    def test_bad_input_cap_is_loud(self, monkeypatch, raw):
        # A sandbox silently running with the wrong cap is a vacuous guard;
        # the loader's own coercion refuses it.
        monkeypatch.setenv("MAXIM_CONSOLE_SANDBOX", "1")
        monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", _ORIGIN)
        monkeypatch.setenv("MAXIM_CONSOLE_MAX_INPUT_CHARS", raw)
        with pytest.raises(ConfigurationError):
            srv._sandbox_policy()

    @pytest.mark.parametrize("raw", [None, "", " , "])
    def test_sandbox_without_origins_is_loud(self, monkeypatch, raw):
        # The inverted vacuous guard: a sandbox whose UI can never open /ws.
        monkeypatch.setenv("MAXIM_CONSOLE_SANDBOX", "1")
        if raw is None:
            monkeypatch.delenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", raising=False)
        else:
            monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", raw)
        with pytest.raises(ConfigurationError, match="allowed_origins"):
            srv._sandbox_policy()

    def test_policy_is_per_app_not_global(self, sandbox_app, plain_app):
        # Two apps in one process carry their own policy (app.state), so a
        # test or embedder building several apps cannot cross-contaminate.
        assert sandbox_app.state.sandbox is not None and plain_app.state.sandbox is None


# ── closed surfaces (+ negative controls) ────────────────────────────────────


class TestClosedSurfaces:
    def test_probe_by_url_is_refused(self, sandbox_app):
        r = TestClient(sandbox_app).post("/api/probe", json={"url": "http://169.254.169.254/latest/"})
        assert r.status_code == 403
        assert r.json()["detail"] == srv._SANDBOX_REFUSAL

    def test_probe_by_provider_stays_open(self, sandbox_app):
        # The cloud wizard's pre-save key check is local and must keep working.
        r = TestClient(sandbox_app).post("/api/probe", json={"provider": "anthropic", "api_key": "sk-x"})
        assert r.status_code == 200
        assert r.json()["status"] == "warn"

    def test_setup_mesh_is_refused(self, sandbox_app):
        r = TestClient(sandbox_app).post("/api/setup/mesh", json={"leader_url": "https://x.example", "api_key": "k"})
        assert r.status_code == 403

    def test_diagnose_is_refused(self, sandbox_app):
        assert TestClient(sandbox_app).get("/api/diagnose").status_code == 403

    def test_setup_cloud_stays_open(self, sandbox_app):
        # BYO-key inside a session is a supported route; the broker (not the
        # engine) closes it when the operator's own key is in play.
        r = TestClient(sandbox_app).post(
            "/api/setup/cloud", json={"provider": "anthropic", "profile": "claude-sonnet", "api_key": "sk-cl"}
        )
        assert r.status_code != 403

    def test_negative_control_mesh_not_refused_when_off(self, plain_app):
        # Same body, mode off: validation runs (not the 403 gate). 422 comes
        # from the body shape — the point is only that the gate did not fire.
        r = TestClient(plain_app).post("/api/setup/mesh", json={"leader_url": "not a url"})
        assert r.status_code != 403

    def test_negative_control_probe_url_not_refused_when_off(self, plain_app, monkeypatch):
        calls: list[str] = []

        class _Backend:
            def __init__(self, url):
                self.url = url

            @classmethod
            def for_url(cls, url, api_key=None, model=None):
                calls.append(url)
                return cls(url)

            def health_check(self):
                from maxim.runtime.llm_server import ProbeResult

                return ProbeResult(url=self.url, outcome="ok", detail="", latency_ms=1.0)

        monkeypatch.setattr("maxim.models.language.maxim_peer_backend._MaximPeerBackend", _Backend)
        r = TestClient(plain_app).post("/api/probe", json={"url": "https://leader.example"})
        assert r.status_code == 200 and calls == ["https://leader.example"]

    def test_setup_detail_never_names_the_secret_path(self, plain_app):
        r = TestClient(plain_app).post(
            "/api/setup/cloud", json={"provider": "anthropic", "profile": "claude-sonnet", "api_key": "sk-cl"}
        )
        assert r.status_code == 200
        detail = r.json()["detail"]
        assert "0600 ref" in detail and "_api_key" not in detail and "sk-cl" not in detail


# ── identity tells the truth about what is closed ───────────────────────────


class TestIdentity:
    def test_closed_seams_reported_not_live(self, sandbox_app):
        seams = {s["name"]: s for s in TestClient(sandbox_app).get("/api/identity").json()["seams"]}
        assert seams["probe"]["live"] is False and "sandbox" in seams["probe"]["detail"]
        assert seams["setup"]["live"] is False and "sandbox" in seams["setup"]["detail"]
        assert seams["talk"]["live"] is True and seams["adventure"]["live"] is True

    def test_seams_live_when_off(self, plain_app):
        seams = {s["name"]: s for s in TestClient(plain_app).get("/api/identity").json()["seams"]}
        assert seams["probe"]["live"] is True and seams["setup"]["live"] is True
        assert seams["sim"]["live"] is False  # unchanged: sim is a CLI surface


# ── /ws origin gate ─────────────────────────────────────────────────────────


class TestWsOrigin:
    def test_unlisted_origin_refused_before_accept(self, sandbox_app):
        with TestClient(sandbox_app) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}):
                    pass

    def test_missing_origin_refused(self, sandbox_app):
        with TestClient(sandbox_app) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws"):
                    pass

    def test_listed_origin_accepted_and_gets_identity_first(self, sandbox_app):
        with TestClient(sandbox_app) as client:
            with client.websocket_connect("/ws", headers={"origin": _ORIGIN.upper() + "/"}) as ws:
                first = ws.receive_json()
                assert first["kind"] == "identity"

    def test_negative_control_missing_origin_accepted_when_off(self, plain_app):
        # Sandbox-off keeps its per-mode contract (origin NOT required — the
        # CLI and native clients send none), but an untrusted browser origin
        # is now refused by the always-on trust guard, not accepted: the old
        # "any origin when off" contract was the /ws half of the CSRF surface
        # (see test_console_trust_guard.py for the trust guard's own suite).
        with TestClient(plain_app) as client:
            with client.websocket_connect("/ws") as ws:
                assert ws.receive_json()["kind"] == "identity"

    def test_untrusted_origin_refused_even_when_off(self, plain_app):
        with TestClient(plain_app) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}):
                    pass


# ── input cap ────────────────────────────────────────────────────────────────


class TestInputCap:
    def test_over_cap_is_413_before_any_handle_is_built(self, sandbox_app, monkeypatch):
        sandbox_app.state.sandbox = srv._SandboxPolicy(allowed_origins=frozenset({_ORIGIN}), max_input_chars=10)
        built: list[str] = []
        monkeypatch.setattr(srv, "_get_handle", lambda: built.append("handle") or None)
        r = TestClient(sandbox_app).post("/api/run", json={"mode": "talk", "input": "x" * 11})
        assert r.status_code == 413 and "10 characters" in r.json()["detail"]
        assert built == []

    def test_under_cap_reaches_the_mode_dispatch(self, sandbox_app):
        sandbox_app.state.sandbox = srv._SandboxPolicy(allowed_origins=frozenset({_ORIGIN}), max_input_chars=10)
        # `sim` is the one mode that answers without a handle: its 501 proves
        # the request got past the cap and into dispatch.
        r = TestClient(sandbox_app).post("/api/run", json={"mode": "sim", "input": "x" * 10})
        assert r.status_code == 501

    def test_no_cap_when_off(self, plain_app):
        r = TestClient(plain_app).post("/api/run", json={"mode": "sim", "input": "x" * 100_000})
        assert r.status_code == 501


# ── shutdown contract ────────────────────────────────────────────────────────


class _FakeHandle:
    def __init__(self):
        self.stops: list[dict] = []

    def stop(self, **kw):
        self.stops.append(kw)


class TestShutdown:
    def test_drain_and_stop_forwards_the_waits(self, monkeypatch):
        fake = _FakeHandle()
        monkeypatch.setattr(srv, "_handle", fake)
        monkeypatch.setattr(srv, "_active_run", {"session_id": None, "thread": None})
        srv._drain_and_stop_handle(campaign_wait_s=1.5, talk_join_s=2.5)
        assert fake.stops == [{"campaign_wait_s": 1.5, "talk_join_s": 2.5}]

    def test_lifespan_stops_the_handle_on_exit(self, plain_app, monkeypatch):
        fake = _FakeHandle()
        monkeypatch.setattr(srv, "_handle", fake)
        with TestClient(plain_app):
            assert fake.stops == []
        assert len(fake.stops) == 1

    def test_lifespan_releases_the_sink_even_if_stop_raises(self, plain_app, monkeypatch):
        # The stop runs INSIDE the finally, ahead of the sink release, and a
        # raising stop must not strand the sink registration.
        released: list[bool] = []
        import maxim.simulation.sim_logger as sim_logger

        real_unregister = sim_logger.unregister_sim_sink

        def _spy(sink):
            released.append(True)
            return real_unregister(sink)

        monkeypatch.setattr(sim_logger, "unregister_sim_sink", _spy)

        def _boom(**kw):
            raise RuntimeError("stop exploded")

        monkeypatch.setattr(srv, "_drain_and_stop_handle", _boom)
        with pytest.raises(RuntimeError, match="stop exploded"):
            with TestClient(plain_app):
                pass
        assert released == [True]
