"""Spoken-code pairing (hardening A9.1) — the device sign-in path.

The robot announces a short-lived one-use code; the paste screen exchanges it
for the console token. Every refusal here is paired with its accepting
counterpart (the PR 1/2 non-vacuity rule), and A7 applies to the CODE exactly
as to the token: announced, never logged, never returned by /request. The
surface is embedder-gated — with no announcer registered (every plain
``maxim serve``) no code can ever exist. Skips cleanly without the console
extra.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from fastapi.testclient import TestClient  # noqa: E402

from maxim.console import server as srv  # noqa: E402
from maxim.console.server import build_app  # noqa: E402
from maxim.runtime.config_loader import ConfigurationError  # noqa: E402
from maxim.tunnel import keys  # noqa: E402

_TOKEN = "mxc_" + "t" * 43


class _Announcer:
    def __init__(self) -> None:
        self.codes: list[str] = []

    def __call__(self, code: str) -> None:
        self.codes.append(code)


@pytest.fixture()
def _isolated(monkeypatch, tmp_path):
    monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
    monkeypatch.delenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    # Rate limit off by default in tests; the rate-limit test sets it back.
    monkeypatch.setattr(srv, "_PAIR_MIN_REQUEST_INTERVAL_S", 0.0)


def _spoken(announcer: _Announcer) -> str:
    # The announcer runs on a daemon thread — join by polling briefly.
    import time

    for _ in range(200):
        if announcer.codes:
            return announcer.codes[-1]
        time.sleep(0.01)
    raise AssertionError("announcer never called")


@pytest.fixture()
def paired_app(_isolated):
    announcer = _Announcer()
    return build_app(None, auth_token=_TOKEN, pairing_announcer=announcer), announcer


class TestEmbedderGate:
    def test_plain_serve_refuses_both_endpoints(self, _isolated):
        # The refusing counterpart that keeps the default posture closed: no
        # announcer, no surface — a plain `maxim serve` gains no new path.
        c = TestClient(build_app(None, auth_token=_TOKEN))
        assert c.post("/api/pair/request", json={}).status_code == 409
        assert c.post("/api/pair/claim", json={"code": "000000"}).status_code == 409

    def test_hello_advertises_pairing(self, _isolated, paired_app):
        app, _ = paired_app
        assert TestClient(app).get("/api/hello").json()["pairing"] == "available"
        plain = build_app(None, auth_token=_TOKEN)
        assert TestClient(plain).get("/api/hello").json()["pairing"] == "none"

    def test_sandbox_combination_refused_at_build_time(self, _isolated, monkeypatch):
        monkeypatch.setenv("MAXIM_CONSOLE_SANDBOX", "1")
        monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", "https://sandbox.example")
        with pytest.raises(ConfigurationError, match="pairing_announcer"):
            build_app(None, auth_token=_TOKEN, pairing_announcer=_Announcer())


class TestHappyPath:
    def test_request_announces_and_claim_returns_the_token(self, paired_app):
        app, announcer = paired_app
        c = TestClient(app)
        r = c.post("/api/pair/request", json={})
        assert r.status_code == 202
        code = _spoken(announcer)
        assert code not in r.text  # the code is ANNOUNCED, never returned
        claim = c.post("/api/pair/claim", json={"code": code})
        assert claim.status_code == 200
        token = claim.json()["token"]
        # The claim hands out the credential THIS app accepts (static here),
        # never a mismatched disk token — the sign-in below is the proof.
        assert token == _TOKEN
        assert c.get("/api/identity", headers={"Authorization": f"Bearer {token}"}).status_code == 200

    def test_disk_sourced_app_hands_out_the_disk_token(self, _isolated):
        announcer = _Announcer()
        app = build_app(None, pairing_announcer=announcer)  # disk-sourced default: the device shape
        c = TestClient(app)
        assert c.post("/api/pair/request", json={}).status_code == 202
        code = _spoken(announcer)
        token = c.post("/api/pair/claim", json={"code": code}).json()["token"]
        assert token == keys.read_console_token()  # same file as maxim serve / --show-token
        assert c.get("/api/identity", headers={"Authorization": f"Bearer {token}"}).status_code == 200

    def test_fail_closed_inject_plus_announcer_refused_at_build(self, _isolated):
        with pytest.raises(ConfigurationError, match="auth_token=None"):
            build_app(None, auth_token=None, pairing_announcer=_Announcer())

    def test_whitespace_around_the_typed_code_is_forgiven(self, paired_app):
        app, announcer = paired_app
        c = TestClient(app)
        assert c.post("/api/pair/request", json={}).status_code == 202
        code = _spoken(announcer)
        assert c.post("/api/pair/claim", json={"code": f"  {code} "}).status_code == 200


class TestCodeLifecycle:
    def test_single_use(self, paired_app):
        app, announcer = paired_app
        c = TestClient(app)
        c.post("/api/pair/request", json={})
        code = _spoken(announcer)
        assert c.post("/api/pair/claim", json={"code": code}).status_code == 200
        assert c.post("/api/pair/claim", json={"code": code}).status_code == 410  # consumed

    def test_expiry(self, paired_app, monkeypatch):
        app, announcer = paired_app
        c = TestClient(app)
        monkeypatch.setattr(srv, "_PAIR_TTL_S", 0.0)
        c.post("/api/pair/request", json={})
        code = _spoken(announcer)
        assert c.post("/api/pair/claim", json={"code": code}).status_code == 410

    def test_five_wrong_attempts_burn_the_code(self, paired_app):
        app, announcer = paired_app
        c = TestClient(app)
        c.post("/api/pair/request", json={})
        code = _spoken(announcer)
        wrong = "000000" if code != "000000" else "000001"
        for i in range(4):
            assert c.post("/api/pair/claim", json={"code": wrong}).status_code == 403
        assert c.post("/api/pair/claim", json={"code": wrong}).status_code == 403  # 5th burns
        # Even the RIGHT code is now void — brute force gets 5 tries per announcement.
        assert c.post("/api/pair/claim", json={"code": code}).status_code == 410

    def test_new_request_replaces_the_previous_code(self, paired_app):
        app, announcer = paired_app
        c = TestClient(app)
        c.post("/api/pair/request", json={})
        first = _spoken(announcer)
        announcer.codes.clear()
        c.post("/api/pair/request", json={})
        second = _spoken(announcer)
        if first != second:  # 1-in-a-million collision guard for the assertion itself
            assert c.post("/api/pair/claim", json={"code": first}).status_code == 403
        assert c.post("/api/pair/claim", json={"code": second}).status_code == 200

    def test_rate_limit_stops_robot_babble(self, paired_app, monkeypatch):
        app, _ = paired_app
        c = TestClient(app)
        monkeypatch.setattr(srv, "_PAIR_MIN_REQUEST_INTERVAL_S", 60.0)
        assert c.post("/api/pair/request", json={}).status_code == 202
        assert c.post("/api/pair/request", json={}).status_code == 429


class TestHygiene:
    def test_code_and_token_never_reach_a_log_line(self, paired_app, caplog):
        app, announcer = paired_app
        c = TestClient(app)
        with caplog.at_level(logging.DEBUG):
            c.post("/api/pair/request", json={})
            code = _spoken(announcer)
            c.post("/api/pair/claim", json={"code": "999999" if code != "999999" else "999998"})
            c.post("/api/pair/claim", json={"code": code})
        assert code not in caplog.text
        assert "mxc_" not in caplog.text

    def test_raising_announcer_does_not_500_and_leaks_nothing(self, _isolated, caplog):
        def bad(code: str) -> None:
            raise RuntimeError(code)  # adversarial: the code IS the message

        app = build_app(None, auth_token=_TOKEN, pairing_announcer=bad)
        c = TestClient(app)
        with caplog.at_level(logging.DEBUG):
            r = c.post("/api/pair/request", json={})
            assert r.status_code == 202
            import time

            time.sleep(0.1)  # let the announcer thread raise and be logged
        # Type-only logging: the exception message (carrying the code) must not land.
        for record in caplog.records:
            assert not any(
                ch.isdigit() and len(tok) == 6 and tok.isdigit() for tok in record.getMessage().split() for ch in tok
            )
        assert "RuntimeError" in caplog.text

    def test_guard_order_origin_belt_still_applies(self, paired_app):
        # Pairing is auth-EXEMPT, not guard-exempt: a cross-site page cannot
        # drive the robot to announce codes.
        app, _ = paired_app
        c = TestClient(app)
        r = c.post("/api/pair/request", json={}, headers={"Origin": "https://evil.example"})
        assert r.status_code == 403
