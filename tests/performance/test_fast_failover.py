"""Fast-failover performance gate (Plan 3 R2.5).

The whole plan's motivation is killing the 52-second slow-fail incident
from 2026-04-12. This test file is the programmatic gate that locks in
the win: every failure mode must surface within 5 seconds against a
mocked-dead peer.

The 63s baseline (measured 2026-04-12 on RTX 5080 leader + Mac peer,
see README.md) is the pre-Plan-3 number. Post-Plan-3 target: each
failure path completes in well under 5 seconds — p99 against mocked
fixtures should be bounded by the TimeoutPolicy's read_s, which we set
to 2s for these tests.

Why mocking instead of a real server: real servers add port-allocation
flakiness, race conditions on startup, and 100ms-1s of spin-up overhead.
The claim we're testing is "no internal retry loop" — fully testable at
the HTTP client boundary via ``patch.object(_http, "post", ...)``.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from maxim.models.language.config import LLMConfig
from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
from maxim.models.language.types import (
    BackendAuthFailed,
    BackendDown,
    BackendTimeout,
)
from maxim.utils import http as _http
from maxim.utils.http import (
    HTTPAuthError,
    HTTPConnectionError,
    HTTPServerError,
    HTTPTimeout,
)

# Hard ceiling for every failure mode: well under the spec's < 5s gate.
# On CI this test should consistently finish in under 1s; the 5s cap is
# a loose safety margin against coincident load.
FAILOVER_CEILING_S = 5.0


@pytest.fixture
def fast_failover_backend() -> _MaximPeerBackend:
    """Construct a backend whose endpoint has an aggressively short
    :class:`TimeoutPolicy`. The base_url is never actually hit —
    ``_http.post`` is mocked — but registration still needs a valid URL
    to pass SSRF validation."""
    import os

    os.environ["PERF_TEST_KEY"] = "test-key"
    cfg = dataclasses.replace(
        LLMConfig(),
        providers={
            "perf-peer": {
                "type": "maxim_peer",
                "base_url": "http://127.0.0.1:9999/v1",
                "api_key_env": "PERF_TEST_KEY",
                "model": "perf-model",
                "allow_local_endpoints": True,
                "pricing_required": False,
                "timeout_s": 2.0,
            }
        },
    )
    return _MaximPeerBackend(cfg, provider_key="perf-peer")


class TestFastFailoverPerformance:
    """The programmatic gate on Plan 3's headline win.

    Each test asserts that a single failure mode completes in well under
    the former 52-second retry loop. Running the suite is fast (<1s
    total) because the HTTP layer is mocked — the test is actually
    verifying "no internal retry loop", not network speed.
    """

    def test_connection_refused_fails_under_ceiling(self, fast_failover_backend):
        """httpx.ConnectError → BackendDown. No repeat loop."""
        backend = fast_failover_backend
        exc = HTTPConnectionError("peer-perf-peer", fix_hint="connection refused")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_http, "post", lambda *a, **kw: (_ for _ in ()).throw(exc))
            start = time.monotonic()
            with pytest.raises(BackendDown):
                backend.complete_with_usage(system="", user="hi", max_tokens=10, temperature=0.0)
            elapsed = time.monotonic() - start
        assert elapsed < FAILOVER_CEILING_S, f"Fast failover broken: {elapsed:.2f}s (target < {FAILOVER_CEILING_S}s)"

    def test_502_bad_gateway_fails_under_ceiling(self, fast_failover_backend):
        """HTTP 502/503/504 chain → BackendDown immediately. Pre-Plan-3
        the ``_OpenAIBackend`` would spin up to ~50s through its gateway
        repeat loop; post-Plan-3 we exit on the first response."""
        backend = fast_failover_backend
        exc = HTTPServerError("peer-perf-peer", status=502, fix_hint="upstream bad")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_http, "post", lambda *a, **kw: (_ for _ in ()).throw(exc))
            start = time.monotonic()
            with pytest.raises(BackendDown):
                backend.complete_with_usage(system="", user="hi", max_tokens=10, temperature=0.0)
            elapsed = time.monotonic() - start
        assert elapsed < FAILOVER_CEILING_S, f"Gateway fail loop returned: {elapsed:.2f}s"

    def test_read_timeout_fails_under_ceiling(self, fast_failover_backend):
        """httpx.ReadTimeout → BackendTimeout bounded by the
        TimeoutPolicy's read_s (2s in the fixture)."""
        backend = fast_failover_backend
        exc = HTTPTimeout("peer-perf-peer", fix_hint="read timeout")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_http, "post", lambda *a, **kw: (_ for _ in ()).throw(exc))
            start = time.monotonic()
            with pytest.raises(BackendTimeout):
                backend.complete_with_usage(system="", user="hi", max_tokens=10, temperature=0.0)
            elapsed = time.monotonic() - start
        assert elapsed < FAILOVER_CEILING_S, f"Read timeout exceeded: {elapsed:.2f}s (target < {FAILOVER_CEILING_S}s)"

    def test_auth_failure_fails_under_ceiling(self, fast_failover_backend):
        """Auth rejection → BackendAuthFailed on first call. The key
        regression guard: the 2026-04-12 stage-2 probe bug made auth
        look like ``inference_broken`` with 15s extra cooldown on top —
        this test locks in "auth fails immediately, typed correctly"."""
        backend = fast_failover_backend
        exc = HTTPAuthError("peer-perf-peer", status=401, fix_hint="bad key")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_http, "post", lambda *a, **kw: (_ for _ in ()).throw(exc))
            start = time.monotonic()
            with pytest.raises(BackendAuthFailed):
                backend.complete_with_usage(system="", user="hi", max_tokens=10, temperature=0.0)
            elapsed = time.monotonic() - start
        assert elapsed < FAILOVER_CEILING_S, f"Auth failure path too slow: {elapsed:.2f}s"

    def test_multiple_consecutive_failures_do_not_accumulate(self, fast_failover_backend):
        """Ten consecutive failures should each clock in at the same
        latency — if internal state were accumulating cooldown or repeat
        counters, the 10th would be slower than the 1st. This test
        catches a whole class of "leaky backoff" bugs."""
        backend = fast_failover_backend
        exc = HTTPConnectionError("peer-perf-peer", fix_hint="refused")
        elapsed_per_call = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_http, "post", lambda *a, **kw: (_ for _ in ()).throw(exc))
            for _ in range(10):
                start = time.monotonic()
                with pytest.raises(BackendDown):
                    backend.complete_with_usage(system="", user="hi", max_tokens=10, temperature=0.0)
                elapsed_per_call.append(time.monotonic() - start)
        total = sum(elapsed_per_call)
        assert total < FAILOVER_CEILING_S * 2, (
            f"10 consecutive failures took {total:.2f}s — leaky cooldown or repeat counter suspected"
        )
        # Last call should be as fast as the first (within 3x noise)
        assert elapsed_per_call[-1] < max(elapsed_per_call[0] * 3, 0.1), (
            f"Per-call latency ramped: first={elapsed_per_call[0]:.3f}s last={elapsed_per_call[-1]:.3f}s"
        )
