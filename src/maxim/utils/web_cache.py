"""Web content cache with TTL and salient extraction.

Provides temporary storage for raw web content with automatic expiration,
and persistence for only salient summaries that improve goal outcomes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cache Entry Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """A cached web content entry."""

    url: str
    content_hash: str
    title: str
    fetched_at: float
    expires_at: float
    content_type: str = "text/html"
    content_length: int = 0

    # Raw content (expires with TTL)
    raw_content: str | None = None

    # Salient extract (may be persisted)
    salient_summary: str | None = None
    keywords: list[str] = field(default_factory=list)

    # Usage tracking
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    contributed_to_goal: bool = False

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at

    def access(self) -> None:
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (without raw content)."""
        return {
            "url": self.url,
            "content_hash": self.content_hash,
            "title": self.title,
            "fetched_at": self.fetched_at,
            "expires_at": self.expires_at,
            "content_type": self.content_type,
            "content_length": self.content_length,
            "salient_summary": self.salient_summary,
            "keywords": self.keywords,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "contributed_to_goal": self.contributed_to_goal,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """Deserialize from dictionary."""
        return cls(
            url=str(data.get("url", "")),
            content_hash=str(data.get("content_hash", "")),
            title=str(data.get("title", "")),
            fetched_at=float(data.get("fetched_at", 0)),
            expires_at=float(data.get("expires_at", 0)),
            content_type=str(data.get("content_type", "text/html")),
            content_length=int(data.get("content_length", 0)),
            salient_summary=data.get("salient_summary"),
            keywords=list(data.get("keywords", [])),
            access_count=int(data.get("access_count", 0)),
            last_accessed=float(data.get("last_accessed", 0)),
            contributed_to_goal=bool(data.get("contributed_to_goal", False)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Web Cache
# ─────────────────────────────────────────────────────────────────────────────


class WebCache:
    """In-memory cache for web content with TTL expiration.

    Design:
    - Raw content is cached temporarily (default 15 minutes)
    - After expiration, only hashes + salient extracts remain
    - Salient extracts are persisted only when they contributed to goals
    """

    def __init__(
        self,
        default_ttl_seconds: int = 900,  # 15 minutes
        max_entries: int = 100,
        persistence_path: Path | str | None = None,
    ):
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl_seconds
        self._max_entries = max_entries
        self._persistence_path = Path(persistence_path) if persistence_path else None

        # Load persisted salient extracts
        if self._persistence_path:
            self._load_persisted()

    def get(self, url: str) -> CacheEntry | None:
        """Get cached entry for URL."""
        with self._lock:
            key = self._url_key(url)
            entry = self._entries.get(key)

            if entry is None:
                return None

            # Check expiration for raw content
            if entry.is_expired():
                # Clear raw content but keep metadata
                entry.raw_content = None

            entry.access()
            return entry

    def get_raw_content(self, url: str) -> str | None:
        """Get raw content if still cached and not expired."""
        entry = self.get(url)
        if entry and not entry.is_expired():
            return entry.raw_content
        return None

    def store(
        self,
        url: str,
        content: str,
        title: str = "",
        content_type: str = "text/html",
        ttl_seconds: int | None = None,
    ) -> CacheEntry:
        """Store content in cache."""
        with self._lock:
            # Enforce max entries
            if len(self._entries) >= self._max_entries:
                self._evict_oldest()

            key = self._url_key(url)
            now = time.time()
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl

            entry = CacheEntry(
                url=url,
                content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
                title=title,
                fetched_at=now,
                expires_at=now + ttl,
                content_type=content_type,
                content_length=len(content),
                raw_content=content,
            )

            self._entries[key] = entry
            return entry

    def set_salient_summary(
        self,
        url: str,
        summary: str,
        keywords: list[str] | None = None,
        contributed_to_goal: bool = False,
    ) -> bool:
        """Set salient summary for a cached entry."""
        with self._lock:
            key = self._url_key(url)
            entry = self._entries.get(key)

            if entry is None:
                return False

            entry.salient_summary = summary
            if keywords:
                entry.keywords = keywords
            entry.contributed_to_goal = contributed_to_goal

            # Persist if it contributed to a goal
            if contributed_to_goal and self._persistence_path:
                self._persist_entry(entry)

            return True

    def mark_goal_contribution(self, url: str) -> bool:
        """Mark that cached content contributed to goal achievement."""
        with self._lock:
            key = self._url_key(url)
            entry = self._entries.get(key)

            if entry is None:
                return False

            entry.contributed_to_goal = True

            # Persist the contribution
            if self._persistence_path and entry.salient_summary:
                self._persist_entry(entry)

            return True

    def cleanup_expired(self) -> int:
        """Remove expired raw content, keeping salient extracts.

        Returns number of entries cleaned.
        """
        with self._lock:
            cleaned = 0
            for entry in self._entries.values():
                if entry.is_expired() and entry.raw_content is not None:
                    entry.raw_content = None
                    cleaned += 1
            return cleaned

    def get_all_salient(self) -> list[dict[str, Any]]:
        """Get all salient extracts (for memory integration)."""
        with self._lock:
            return [
                {
                    "url": entry.url,
                    "title": entry.title,
                    "summary": entry.salient_summary,
                    "keywords": entry.keywords,
                    "content_hash": entry.content_hash,
                    "contributed_to_goal": entry.contributed_to_goal,
                }
                for entry in self._entries.values()
                if entry.salient_summary
            ]

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = len(self._entries)
            with_raw = sum(1 for e in self._entries.values() if e.raw_content)
            with_salient = sum(1 for e in self._entries.values() if e.salient_summary)
            contributed = sum(
                1 for e in self._entries.values() if e.contributed_to_goal
            )

            return {
                "total_entries": total,
                "with_raw_content": with_raw,
                "with_salient_summary": with_salient,
                "contributed_to_goals": contributed,
                "max_entries": self._max_entries,
                "default_ttl_seconds": self._default_ttl,
            }

    def _url_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:32]

    def _evict_oldest(self) -> None:
        """Evict oldest entry to make room (must hold lock)."""
        if not self._entries:
            return

        # Find entry with oldest last_accessed that didn't contribute to goals
        candidates = [
            (key, entry)
            for key, entry in self._entries.items()
            if not entry.contributed_to_goal
        ]

        if not candidates:
            # All contributed - evict oldest anyway
            candidates = list(self._entries.items())

        oldest_key = min(candidates, key=lambda x: x[1].last_accessed)[0]
        del self._entries[oldest_key]

    def _persist_entry(self, entry: CacheEntry) -> None:
        """Persist salient entry to disk (must hold lock)."""
        if not self._persistence_path:
            return

        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing
            entries = []
            if self._persistence_path.exists():
                with open(self._persistence_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)

            # Check if entry already exists
            key = self._url_key(entry.url)
            entries = [e for e in entries if e.get("content_hash") != entry.content_hash]

            # Add new entry
            entries.append(entry.to_dict())

            # Save
            with open(self._persistence_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist cache entry: {e}")

    def _load_persisted(self) -> None:
        """Load persisted salient extracts (must hold lock)."""
        if not self._persistence_path or not self._persistence_path.exists():
            return

        try:
            with open(self._persistence_path, "r", encoding="utf-8") as f:
                entries = json.load(f)

            for entry_data in entries:
                entry = CacheEntry.from_dict(entry_data)
                # Mark as expired (no raw content)
                entry.expires_at = 0
                entry.raw_content = None
                key = self._url_key(entry.url)
                self._entries[key] = entry

            logger.info(f"Loaded {len(entries)} persisted web cache entries")

        except Exception as e:
            logger.warning(f"Failed to load persisted web cache: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Global Cache Instance
# ─────────────────────────────────────────────────────────────────────────────


_global_cache: WebCache | None = None


def get_web_cache(
    persistence_path: Path | str | None = None,
    create_if_missing: bool = True,
) -> WebCache | None:
    """Get the global web cache instance."""
    global _global_cache

    if _global_cache is None and create_if_missing:
        default_path = Path("data/internet/web_cache.json")
        _global_cache = WebCache(
            persistence_path=persistence_path or default_path,
        )

    return _global_cache


def set_web_cache(cache: WebCache) -> None:
    """Set the global web cache instance."""
    global _global_cache
    _global_cache = cache
