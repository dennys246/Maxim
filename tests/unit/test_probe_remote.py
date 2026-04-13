"""Tests for the structured remote probe + cache (P6 of peer_leader_flexibility).

Three layers:

1. **probe_llm_server / _probe_once**: each network/HTTP failure mode is
   classified into a stable :class:`ProbeOutcome`. Tests mock
   ``maxim.utils.http.fetch_url`` (post Plan 1 R1) and assert the outcome
   string — pre-R1 these tests mocked ``urllib.request.urlopen`` directly.
2. **probe_cache**: load/save round-trip, freshness window, single-URL
   eviction, full clear, missing-file tolerance.
3. **_validate_remote_urls**: lanes whose probe says "unreachable" get
   ``remote_url`` cleared; ``ok`` and ``auth_rejected`` keep them wired;
   the cache short-circuits the probe when fresh; ``MAXIM_SKIP_REMOTE_PROBE``
   bypasses the whole stage.
"""

from __future__ import annotations

import socket
import ssl
import time
from unittest.mock import MagicMock, patch

import pytest

from maxim.runtime import probe_cache
from maxim.runtime.lane_backends import _validate_remote_urls
from maxim.runtime.llm_server import ProbeResult, _probe_once, probe_llm_server
from maxim.runtime.worker_pool import LaneConfig
from maxim.utils import http as _http


# ─── shared fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fake_data_home(tmp_path, monkeypatch):
    home = tmp_path / "maxim_home"
    home.mkdir()
    (home / "util").mkdir()
    monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
    # data_home() memoizes its first lookup at module scope. Reset that
    # cache so each test sees its own MAXIM_DATA_HOME.
    from maxim.utils import paths

    paths._reset_caches()
    return home


@pytest.fixture(autouse=True)
def _scrub_probe_env(monkeypatch):
    for k in (
        "MAXIM_SKIP_REMOTE_PROBE",
        "MAXIM_REMOTE_PROBE_CACHE_TTL_S",
        "MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S",
        "MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S",
    ):
        monkeypatch.delenv(k, raising=False)


def _ok_response(status: int = 200) -> _http.Response:
    return _http.Response(
        status=status,
        headers={},
        content=b"{}",
        elapsed_ms=1.0,
        endpoint=_http._EXTERNAL_ENDPOINT,
        request_id="r",
    )


# ─── _probe_once outcome classification ─────────────────────────────────────


class TestProbeOnce:
    URL = "http://leader.local:8100/v1"

    def test_ok_200(self):
        with patch("maxim.utils.http.fetch_url", return_value=_ok_response(200)):
            result = _probe_once(self.URL, None, 1.0)
        assert result.outcome == "ok"
        assert result.latency_ms is not None and result.latency_ms >= 0

    def test_auth_rejected_401(self):
        err = _http.HTTPAuthError(_http._EXTERNAL_ENDPOINT, status=401, fix_hint="Auth rejected (401)")
        with patch("maxim.utils.http.fetch_url", side_effect=err):
            result = _probe_once(self.URL, "wrong-key", 1.0)
        assert result.outcome == "auth_rejected"
        assert result.is_reachable is True

    def test_http_5xx(self):
        err = _http.HTTPServerError(_http._EXTERNAL_ENDPOINT, status=502, fix_hint="Server error (502)")
        with patch("maxim.utils.http.fetch_url", side_effect=err):
            result = _probe_once(self.URL, None, 1.0)
        assert result.outcome == "http_5xx"

    def test_http_4xx_other(self):
        err = _http.HTTPClientError(_http._EXTERNAL_ENDPOINT, status=404, fix_hint="Client error (404)")
        with patch("maxim.utils.http.fetch_url", side_effect=err):
            result = _probe_once(self.URL, None, 1.0)
        assert result.outcome == "other"
        assert "404" in result.detail

    def test_dns_fail(self):
        root = socket.gaierror("name or service not known")
        err = _http.HTTPConnectionError(_http._EXTERNAL_ENDPOINT, fix_hint="dns")
        err.__cause__ = root
        with patch("maxim.utils.http.fetch_url", side_effect=err):
            result = _probe_once(self.URL, None, 1.0)
        assert result.outcome == "dns_fail"

    def test_tls_error(self):
        root = ssl.SSLError("certificate verify failed")
        err = _http.HTTPConnectionError(_http._EXTERNAL_ENDPOINT, fix_hint="tls")
        err.__cause__ = root
        with patch("maxim.utils.http.fetch_url", side_effect=err):
            result = _probe_once(self.URL, None, 1.0)
        assert result.outcome == "tls_error"

    def test_connection_refused(self):
        root = ConnectionRefusedError("nope")
        err = _http.HTTPConnectionError(_http._EXTERNAL_ENDPOINT, fix_hint="refused")
        err.__cause__ = root
        with patch("maxim.utils.http.fetch_url", side_effect=err):
            result = _probe_once(self.URL, None, 1.0)
        assert result.outcome == "connection_refused"

    def test_timeout(self):
        with patch(
            "maxim.utils.http.fetch_url",
            side_effect=_http.HTTPTimeout(_http._EXTERNAL_ENDPOINT, fix_hint="slow"),
        ):
            result = _probe_once(self.URL, None, 1.0)
        assert result.outcome == "timeout"

    def test_other_unexpected(self):
        with patch("maxim.utils.http.fetch_url", side_effect=RuntimeError("weird")):
            result = _probe_once(self.URL, None, 1.0)
        assert result.outcome == "other"

    def test_api_key_added_to_request(self):
        captured: dict = {}

        def fake_fetch(url, *, method="GET", headers=None, timeout=None, **_):
            captured["headers"] = dict(headers or {})
            return _ok_response(200)

        with patch("maxim.utils.http.fetch_url", side_effect=fake_fetch):
            _probe_once(self.URL, "secret-token", 1.0)
        assert captured["headers"].get("Authorization") == "Bearer secret-token"

    def test_user_agent_set_for_cloudflare(self):
        """Post Plan 1 R1 the User-Agent is set once at _external endpoint
        registration (http.DEFAULT_USER_AGENT = "maxim-peer/1.0"). The
        Cloudflare incident is now structurally impossible — there is no
        code path that builds an outbound HTTP request without inheriting
        the endpoint's default headers.
        """
        from maxim.utils import http as _h

        _h._ensure_external_endpoint()
        ep = _h.get_endpoint(_h._EXTERNAL_ENDPOINT)
        assert ep.default_headers.get("User-Agent") == "maxim-peer/1.0"


# ─── probe_llm_server two-attempt retry ─────────────────────────────────────


class TestProbeLlmServerRetry:
    URL = "http://leader.local:8100/v1"

    def test_ok_short_circuits_no_retry(self):
        with patch(
            "maxim.runtime.llm_server._probe_once",
            return_value=ProbeResult(self.URL, "ok", "HTTP 200", 12.3),
        ) as mock:
            probe_llm_server(self.URL)
        assert mock.call_count == 1

    def test_auth_rejected_short_circuits_no_retry(self):
        with patch(
            "maxim.runtime.llm_server._probe_once",
            return_value=ProbeResult(self.URL, "auth_rejected", "401", 5.0),
        ) as mock:
            probe_llm_server(self.URL)
        assert mock.call_count == 1

    def test_timeout_retries_once(self):
        results = [
            ProbeResult(self.URL, "timeout", "0.8s", None),
            ProbeResult(self.URL, "ok", "HTTP 200", 1500.0),
        ]
        with patch("maxim.runtime.llm_server._probe_once", side_effect=results) as mock:
            final = probe_llm_server(self.URL)
        assert mock.call_count == 2
        assert final.outcome == "ok"


# ─── probe_cache ────────────────────────────────────────────────────────────


class TestProbeCache:
    def test_load_missing_file_returns_empty(self, fake_data_home):
        assert probe_cache.load_cache() == {}

    def test_save_load_round_trip(self, fake_data_home):
        entry = {
            "outcome": "ok",
            "detail": "HTTP 200",
            "probed_at": time.time(),
            "latency_ms": 42.0,
        }
        probe_cache.save_cache({"https://leader/v1": entry})
        loaded = probe_cache.load_cache()
        assert loaded["https://leader/v1"]["outcome"] == "ok"

    def test_is_fresh_within_window(self):
        entry = {"probed_at": time.time() - 10}
        assert probe_cache.is_fresh(entry, 30) is True

    def test_is_fresh_expired(self):
        entry = {"probed_at": time.time() - 120}
        assert probe_cache.is_fresh(entry, 30) is False

    def test_is_fresh_missing_field(self):
        assert probe_cache.is_fresh({}, 30) is False

    def test_clear_cache_removes_file(self, fake_data_home):
        probe_cache.save_cache({"u": {"outcome": "ok", "probed_at": time.time()}})
        probe_cache.clear_cache()
        assert probe_cache.load_cache() == {}

    def test_clear_cache_for_url_removes_one_entry(self, fake_data_home):
        now = time.time()
        probe_cache.save_cache(
            {
                "https://a/v1": {"outcome": "ok", "probed_at": now},
                "https://b/v1": {"outcome": "ok", "probed_at": now},
            }
        )
        probe_cache.clear_cache_for_url("https://a/v1")
        loaded = probe_cache.load_cache()
        assert "https://a/v1" not in loaded
        assert "https://b/v1" in loaded

    def test_load_handles_corrupt_json(self, fake_data_home):
        path = fake_data_home / "util" / "last_probe_status.json"
        path.write_text("{not valid json")
        assert probe_cache.load_cache() == {}


# ─── _validate_remote_urls integration ──────────────────────────────────────


def _ok(url):
    return ProbeResult(url, "ok", "HTTP 200", 10.0)


def _down(url, outcome="connection_refused"):
    return ProbeResult(url, outcome, "boom", None)


def _auth(url):
    return ProbeResult(url, "auth_rejected", "401", 5.0)


class TestValidateRemoteUrls:
    def _lane(self, url, name="large"):
        return LaneConfig(name=name, max_workers=1, remote_url=url, remote_api_key="k")

    def test_ok_keeps_lane_wired(self, fake_data_home):
        cfg = self._lane("https://leader.example.com/v1")
        with patch(
            "maxim.runtime.llm_server.probe_llm_server",
            return_value=_ok(cfg.remote_url),
        ):
            out = _validate_remote_urls({"large": cfg}, logger=None)
        assert out["large"].remote_url == cfg.remote_url

    def test_auth_rejected_keeps_lane_wired(self, fake_data_home):
        """Auth-gated leader is still alive — drop fallback would be the
        wrong move; the user should fix their key, not run locally."""
        cfg = self._lane("https://leader.example.com/v1")
        log = MagicMock()
        with patch(
            "maxim.runtime.llm_server.probe_llm_server",
            return_value=_auth(cfg.remote_url),
        ):
            out = _validate_remote_urls({"large": cfg}, logger=log)
        assert out["large"].remote_url == cfg.remote_url
        assert log.warning.called
        # Hint mentions key rotation
        msg = log.warning.call_args[0][0] % log.warning.call_args[0][1:]
        assert "maxim peer key" in msg

    def test_unreachable_drops_lane(self, fake_data_home):
        cfg = self._lane("https://dead.example.com/v1")
        with patch(
            "maxim.runtime.llm_server.probe_llm_server",
            return_value=_down(cfg.remote_url, "dns_fail"),
        ):
            out = _validate_remote_urls({"large": cfg}, logger=MagicMock())
        assert out["large"].remote_url is None
        assert out["large"].remote_api_key is None

    def test_fresh_cache_skips_probe(self, fake_data_home):
        url = "https://leader.example.com/v1"
        probe_cache.save_cache(
            {
                url: {
                    "outcome": "ok",
                    "detail": "HTTP 200",
                    "probed_at": time.time(),
                    "latency_ms": 10.0,
                }
            }
        )
        cfg = self._lane(url)
        with patch("maxim.runtime.llm_server.probe_llm_server") as mock_probe:
            out = _validate_remote_urls({"large": cfg}, logger=None)
        mock_probe.assert_not_called()
        assert out["large"].remote_url == url

    def test_stale_cache_reprobes(self, fake_data_home, monkeypatch):
        url = "https://leader.example.com/v1"
        # Force a 1-second TTL so the entry is stale immediately.
        monkeypatch.setenv("MAXIM_REMOTE_PROBE_CACHE_TTL_S", "1")
        probe_cache.save_cache(
            {
                url: {
                    "outcome": "ok",
                    "detail": "HTTP 200",
                    "probed_at": time.time() - 60,
                    "latency_ms": 10.0,
                }
            }
        )
        cfg = self._lane(url)
        with patch(
            "maxim.runtime.llm_server.probe_llm_server",
            return_value=_ok(url),
        ) as mock_probe:
            _validate_remote_urls({"large": cfg}, logger=None)
        mock_probe.assert_called_once()

    def test_skip_env_bypasses_probe(self, fake_data_home, monkeypatch):
        monkeypatch.setenv("MAXIM_SKIP_REMOTE_PROBE", "1")
        cfg = self._lane("https://anything.example.com/v1")
        with patch("maxim.runtime.llm_server.probe_llm_server") as mock_probe:
            out = _validate_remote_urls({"large": cfg}, logger=None)
        mock_probe.assert_not_called()
        assert out["large"].remote_url == cfg.remote_url

    def test_empty_remote_url_skipped(self, fake_data_home):
        cfg = LaneConfig(name="large", max_workers=1, remote_url=None)
        with patch("maxim.runtime.llm_server.probe_llm_server") as mock_probe:
            _validate_remote_urls({"large": cfg}, logger=None)
        mock_probe.assert_not_called()

    def test_probe_writes_cache_after_run(self, fake_data_home):
        url = "https://leader.example.com/v1"
        cfg = self._lane(url)
        with patch(
            "maxim.runtime.llm_server.probe_llm_server",
            return_value=_ok(url),
        ):
            _validate_remote_urls({"large": cfg}, logger=None)
        loaded = probe_cache.load_cache()
        assert url in loaded
        assert loaded[url]["outcome"] == "ok"


# ─── env-knob clamping ──────────────────────────────────────────────────────


class TestEnvKnobs:
    def test_first_timeout_clamped_low(self, fake_data_home, monkeypatch):
        monkeypatch.setenv("MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S", "0.01")
        from maxim.runtime.lane_backends import _safe_float_env

        assert _safe_float_env("MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S", 0.8, min_value=0.2, max_value=5.0) == 0.2

    def test_retry_timeout_clamped_high(self, fake_data_home, monkeypatch):
        monkeypatch.setenv("MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S", "999")
        from maxim.runtime.lane_backends import _safe_float_env

        assert _safe_float_env("MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S", 2.5, min_value=0.5, max_value=10.0) == 10.0

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("MAXIM_REMOTE_PROBE_CACHE_TTL_S", "not-a-number")
        from maxim.runtime.lane_backends import _safe_float_env

        assert _safe_float_env("MAXIM_REMOTE_PROBE_CACHE_TTL_S", 60.0) == 60.0
