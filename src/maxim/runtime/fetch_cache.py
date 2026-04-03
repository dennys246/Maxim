"""Result caching with TTL and file mtime validation.

Provides a thread-safe cache for tool results (file reads, glob searches,
etc.) keyed by tool name + parameter hash. Entries expire after a
configurable TTL and file-read entries are additionally invalidated when
the underlying file's mtime changes.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """A cached result with TTL and validation."""

    result: Any
    timestamp: float
    file_mtime: float | None = None  # For file reads, track mtime
    ttl_seconds: float = 30.0


class ResultCache:
    """Cache for tool results with TTL and file mtime validation."""

    def __init__(self, default_ttl: float = 30.0, max_entries: int = 100):
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._max_entries = max_entries

    def _make_key(self, tool_name: str, params: dict[str, Any]) -> str:
        """Create a cache key from tool name and params."""
        # Sort params for consistent keys
        sorted_params = sorted(params.items())
        param_str = str(sorted_params)
        return f"{tool_name}:{hashlib.md5(param_str.encode()).hexdigest()}"

    def get(self, tool_name: str, params: dict[str, Any]) -> Any | None:
        """Get a cached result if valid.

        Returns None if not cached or expired.
        """
        key = self._make_key(tool_name, params)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            # Check TTL
            age = time.time() - entry.timestamp
            if age > entry.ttl_seconds:
                del self._cache[key]
                return None

            # For file reads, check mtime
            if entry.file_mtime is not None:
                path = params.get("path")
                if path:
                    try:
                        current_mtime = os.path.getmtime(path)
                        if current_mtime != entry.file_mtime:
                            del self._cache[key]
                            return None
                    except OSError:
                        del self._cache[key]
                        return None

            return entry.result

    def put(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: Any,
        ttl: float | None = None,
    ) -> None:
        """Cache a result."""
        key = self._make_key(tool_name, params)

        # Get file mtime if this is a file read
        file_mtime = None
        if tool_name == "read_file":
            path = params.get("path")
            if path:
                try:
                    file_mtime = os.path.getmtime(path)
                except OSError:
                    pass

        with self._lock:
            # Evict old entries if at capacity
            if len(self._cache) >= self._max_entries:
                # Remove oldest entries
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: x[1].timestamp,
                )
                for old_key, _ in sorted_entries[:self._max_entries // 4]:
                    del self._cache[old_key]

            self._cache[key] = CacheEntry(
                result=result,
                timestamp=time.time(),
                file_mtime=file_mtime,
                ttl_seconds=ttl or self._default_ttl,
            )

    def invalidate(self, tool_name: str | None = None, path: str | None = None) -> int:
        """Invalidate cache entries.

        Args:
            tool_name: If provided, invalidate entries for this tool
            path: If provided, invalidate entries containing this path

        Returns: Number of entries invalidated
        """
        with self._lock:
            to_remove = []
            for key, entry in self._cache.items():
                if tool_name and key.startswith(f"{tool_name}:"):
                    to_remove.append(key)
                elif path:
                    # Check if result contains this path
                    result_str = str(entry.result)
                    if path in result_str:
                        to_remove.append(key)

            for key in to_remove:
                del self._cache[key]

            return len(to_remove)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "default_ttl": self._default_ttl,
            }


# Global cache instance
_result_cache = ResultCache()


def get_result_cache() -> ResultCache:
    """Get the global result cache."""
    return _result_cache
