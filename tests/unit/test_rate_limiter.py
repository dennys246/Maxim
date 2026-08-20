"""Unit tests for Phase 7b: TokenBucket + PeerRateLimiter."""

from __future__ import annotations

from io import BytesIO
import threading
import time
from unittest.mock import patch

import pytest

from maxim.runtime.rate_limiter import PeerRateLimiter, TokenBucket


class TestTokenBucket:
    """TokenBucket core functionality."""

    def test_allows_burst(self) -> None:
        b = TokenBucket(rate_per_minute=60, burst=5)
        for _ in range(5):
            assert b.try_acquire()
        # 6th should fail (burst exhausted)
        assert not b.try_acquire()

    def test_refills_over_time(self) -> None:
        b = TokenBucket(rate_per_minute=600, burst=1)  # 10/sec
        assert b.try_acquire()
        assert not b.try_acquire()
        time.sleep(0.15)  # wait for ~1.5 tokens to refill
        assert b.try_acquire()

    def test_burst_defaults_to_rate(self) -> None:
        b = TokenBucket(rate_per_minute=30)
        # burst defaults to rate_per_minute = 30
        for _ in range(30):
            assert b.try_acquire()
        assert not b.try_acquire()

    def test_tokens_available(self) -> None:
        b = TokenBucket(rate_per_minute=60, burst=10)
        assert b.tokens_available == pytest.approx(10.0, abs=0.5)
        b.try_acquire()
        assert b.tokens_available == pytest.approx(9.0, abs=0.5)

    def test_thread_safety(self) -> None:
        """Concurrent acquire shouldn't over-count or crash."""
        b = TokenBucket(rate_per_minute=6000, burst=100)
        acquired = []
        lock = threading.Lock()

        def worker() -> None:
            count = 0
            for _ in range(50):
                if b.try_acquire():
                    count += 1
            with lock:
                acquired.append(count)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = sum(acquired)
        # Should not exceed burst (100) + whatever refilled during test
        assert total <= 200  # generous bound


class TestPeerRateLimiter:
    """PeerRateLimiter registry."""

    def test_disabled_by_default(self) -> None:
        r = PeerRateLimiter(rate_per_minute=0)
        assert not r.enabled
        assert r.try_acquire("peer-1")

    def test_enabled_with_rate(self) -> None:
        r = PeerRateLimiter(rate_per_minute=60)
        assert r.enabled

    def test_per_peer_isolation(self) -> None:
        r = PeerRateLimiter(rate_per_minute=2)
        # peer-1 exhausts its 2-token burst
        assert r.try_acquire("peer-1")
        assert r.try_acquire("peer-1")
        assert not r.try_acquire("peer-1")
        # peer-2 still has full burst
        assert r.try_acquire("peer-2")
        assert r.try_acquire("peer-2")
        assert not r.try_acquire("peer-2")

    def test_peer_count(self) -> None:
        r = PeerRateLimiter(rate_per_minute=60)
        r.try_acquire("a")
        r.try_acquire("b")
        r.try_acquire("c")
        assert r.peer_count() == 3

    def test_unlimited_always_allows(self) -> None:
        r = PeerRateLimiter(rate_per_minute=0)
        for _ in range(1000):
            assert r.try_acquire("peer-1")


class TestLeaderProxyAdmission:
    """LeaderProxy admission control without opening a real listener."""

    def test_concurrency_semaphore_rejects(self) -> None:
        """When concurrency cap is hit, proxy should reject."""
        from maxim.runtime.leader_proxy import _ProxyHandler

        semaphore = threading.Semaphore(1)
        assert semaphore.acquire(blocking=False)

        handler = object.__new__(_ProxyHandler)
        handler.concurrency_semaphore = semaphore
        handler.rate_limiter = None
        handler.lane_metrics = None
        handler.client_address = ("127.0.0.1", 12345)
        handler._sent = []
        handler._send_json = lambda code, body: handler._sent.append((code, body))

        assert handler._check_admission() is False
        assert handler._sent == [(429, {"error": "Too many concurrent requests", "queue_depth": 0})]

        semaphore.release()
        assert handler._check_admission() is True
        handler._release_concurrency()

    def test_queue_depth_header_present(self) -> None:
        """Proxied responses should include X-Maxim-Queue-Depth."""
        from maxim.runtime.leader_proxy import _ProxyHandler

        handler = object.__new__(_ProxyHandler)
        handler.headers = {}
        handler.client_address = ("127.0.0.1", 12345)
        handler.path = "/v1/chat/completions"
        handler.upstream_url = "http://127.0.0.1:19998"
        handler.rfile = BytesIO()
        handler.wfile = BytesIO()
        handler.lane_metrics = None
        handler.request_log = None
        handler._status = []
        handler._headers = []
        handler.send_response = lambda status: handler._status.append(status)
        handler.send_header = lambda key, value: handler._headers.append((key, value))
        handler.end_headers = lambda: None

        with (
            patch(
                "maxim.utils.http.raw_proxy_forward_streaming",
                side_effect=ConnectionError("offline test upstream"),
            ),
            patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=None),
        ):
            handler._proxy_request("POST")

        assert handler._status == [502]
        assert ("X-Maxim-Queue-Depth", "0") in handler._headers
