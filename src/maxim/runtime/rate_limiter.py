"""Token-bucket rate limiter for LeaderProxy per-peer admission (Phase 7b).

Stdlib-only, thread-safe. Each peer (keyed by API key or source IP) gets
its own bucket. Configurable via MAXIM_PROXY_RATE_LIMIT_RPM (requests per
minute per peer, default unlimited).

Design: classic token bucket with 1-second granularity refill. Tokens
accumulate up to the burst cap (= rate limit). Each request consumes one
token. If the bucket is empty, the request is rejected with 429.
"""

from __future__ import annotations

import os
import threading
import time


class TokenBucket:
    """Single token bucket for one peer."""

    def __init__(self, rate_per_minute: float, burst: int | None = None) -> None:
        self._rate = rate_per_minute / 60.0  # tokens per second
        self._burst = burst if burst is not None else max(1, int(rate_per_minute))
        self._tokens = float(self._burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Try to consume one token. Returns True if allowed, False if rate-limited."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    @property
    def tokens_available(self) -> float:
        """Current token count (approximate, for diagnostics)."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            return min(self._burst, self._tokens + elapsed * self._rate)


class PeerRateLimiter:
    """Per-peer rate limiter registry.

    Creates a TokenBucket for each peer on first access. Thread-safe.
    Rate limit is read from MAXIM_PROXY_RATE_LIMIT_RPM (default 0 = unlimited).
    """

    def __init__(self, rate_per_minute: float | None = None) -> None:
        if rate_per_minute is None:
            try:
                rate_per_minute = float(os.environ.get("MAXIM_PROXY_RATE_LIMIT_RPM", "0"))
            except (ValueError, TypeError):
                rate_per_minute = 0.0
        self._rate = rate_per_minute
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._rate > 0

    def try_acquire(self, peer_key: str) -> bool:
        """Check if peer_key is allowed to make a request.

        Returns True if allowed (or rate limiting is disabled).
        Returns False if rate-limited (caller should respond 429).
        """
        if not self.enabled:
            return True
        bucket = self._get_bucket(peer_key)
        return bucket.try_acquire()

    def _get_bucket(self, peer_key: str) -> TokenBucket:
        with self._lock:
            if peer_key not in self._buckets:
                self._buckets[peer_key] = TokenBucket(self._rate)
            return self._buckets[peer_key]

    def peer_count(self) -> int:
        with self._lock:
            return len(self._buckets)


__all__ = ["TokenBucket", "PeerRateLimiter"]
