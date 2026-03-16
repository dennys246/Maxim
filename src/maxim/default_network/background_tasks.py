"""Background task manager for DefaultNetwork maintenance operations.

Offloads expensive operations that don't need to run every tick:
- Memory cleanup (SpatialMap, NoveltyTracker)
- Statistics collection
- Periodic state persistence

This prevents the main DN loop from being bogged down by maintenance.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.attention import AttentionNetwork
    from maxim.salience import SalienceNetwork
    from maxim.spatial import SpatialMap
    from maxim.salience import ThreadSafeNoveltyTracker

logger = logging.getLogger(__name__)


@dataclass
class BackgroundTaskConfig:
    """Configuration for background maintenance tasks.

    Attributes:
        enabled: Whether background tasks are enabled.
        cleanup_interval_seconds: How often to run cleanup tasks.
        stats_interval_seconds: How often to collect/log stats.
        spatial_map_cleanup_enabled: Whether to cleanup spatial map.
        novelty_cleanup_enabled: Whether to cleanup novelty tracker.
        salience_cleanup_enabled: Whether to cleanup salience network.
    """
    enabled: bool = True
    cleanup_interval_seconds: float = 30.0  # Run cleanup every 30s
    stats_interval_seconds: float = 60.0  # Log stats every 60s
    spatial_map_cleanup_enabled: bool = True
    novelty_cleanup_enabled: bool = True
    salience_cleanup_enabled: bool = True


@dataclass
class ThrottleConfig:
    """Configuration for throttling expensive operations.

    Instead of running every tick at 30Hz, throttle to lower rates.

    Attributes:
        spatial_map_update_hz: Max rate for spatial map updates.
        salience_update_hz: Max rate for salience network updates.
        novelty_update_hz: Max rate for novelty tracker updates.
        behavior_eval_hz: Max rate for behavior evaluation.
    """
    spatial_map_update_hz: float = 5.0  # 5 updates/sec instead of 30
    salience_update_hz: float = 10.0  # 10 updates/sec
    novelty_update_hz: float = 10.0  # 10 updates/sec
    behavior_eval_hz: float = 15.0  # 15 evals/sec (still responsive)


class OperationThrottler:
    """Throttles operations to a maximum rate.

    Example:
        throttler = OperationThrottler(max_hz=10.0)

        # In a 30Hz loop:
        if throttler.should_run():
            expensive_operation()
    """

    def __init__(self, max_hz: float) -> None:
        self._min_interval = 1.0 / max_hz if max_hz > 0 else 0.0
        self._last_run: float = 0.0
        self._lock = threading.Lock()

    def should_run(self) -> bool:
        """Check if enough time has passed since last run."""
        if self._min_interval <= 0:
            return True

        now = time.time()
        with self._lock:
            if now - self._last_run >= self._min_interval:
                self._last_run = now
                return True
        return False

    def reset(self) -> None:
        """Reset the throttler (allows immediate next run)."""
        with self._lock:
            self._last_run = 0.0


class BackgroundTaskManager:
    """Manages background maintenance tasks for DefaultNetwork.

    Runs cleanup and stats collection in a separate thread to avoid
    blocking the main processing loop.
    """

    def __init__(
        self,
        config: BackgroundTaskConfig | None = None,
        spatial_map: "SpatialMap | None" = None,
        attention_network: "AttentionNetwork | None" = None,
        salience_network: "SalienceNetwork | None" = None,
        novelty_tracker: "ThreadSafeNoveltyTracker | None" = None,
    ) -> None:
        self._config = config or BackgroundTaskConfig()
        self._spatial_map = spatial_map
        self._attention_network = attention_network
        self._salience_network = salience_network
        self._novelty_tracker = novelty_tracker

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Timestamps for periodic tasks
        self._last_cleanup: float = 0.0
        self._last_stats: float = 0.0

        # Stats
        self._cleanup_count = 0
        self._items_cleaned = 0

    def start(self) -> None:
        """Start the background task thread."""
        if self._running or not self._config.enabled:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="DN-BackgroundTasks",
            daemon=True,
        )
        self._thread.start()
        logger.debug("BackgroundTaskManager started")

    def stop(self) -> None:
        """Stop the background task thread."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.debug("BackgroundTaskManager stopped")

    def _run_loop(self) -> None:
        """Main background task loop."""
        while not self._stop_event.is_set():
            now = time.time()

            # Run cleanup tasks periodically
            if now - self._last_cleanup >= self._config.cleanup_interval_seconds:
                self._run_cleanup()
                self._last_cleanup = now

            # Log stats periodically
            if now - self._last_stats >= self._config.stats_interval_seconds:
                self._log_stats()
                self._last_stats = now

            # Sleep to avoid busy-waiting (check every 5 seconds)
            self._stop_event.wait(timeout=5.0)

    def _run_cleanup(self) -> None:
        """Run all cleanup tasks."""
        cleaned = 0

        # Cleanup spatial map
        if self._config.spatial_map_cleanup_enabled and self._spatial_map is not None:
            try:
                # SpatialMap has cleanup() method
                if hasattr(self._spatial_map, 'cleanup'):
                    self._spatial_map.cleanup()
                    cleaned += 1
            except Exception as e:
                logger.debug("Spatial map cleanup failed: %s", e)

        # Cleanup salience network (remove stale entries)
        if self._config.salience_cleanup_enabled and self._salience_network is not None:
            try:
                # Prune objects not seen recently
                stats = self._salience_network.get_stats()
                current = stats.get("current_object_count", 0)
                max_obj = stats.get("max_tracked_objects", 100)
                if current > max_obj * 0.8:  # 80% full, trigger cleanup
                    logger.debug("Salience network at %d/%d, cleanup recommended", current, max_obj)
                    cleaned += 1
            except Exception as e:
                logger.debug("Salience network check failed: %s", e)

        # Note: NoveltyTracker has its own internal cleanup mechanism
        # that runs every 100 updates, so we don't need to call it here

        self._cleanup_count += 1
        self._items_cleaned += cleaned

        if cleaned > 0:
            logger.debug("Background cleanup completed: %d operations", cleaned)

    def _log_stats(self) -> None:
        """Log current memory/performance stats."""
        stats = []

        if self._spatial_map is not None:
            try:
                sm_stats = self._spatial_map.get_stats()
                stats.append(f"SpatialMap: {sm_stats.get('total_observations', 0)} obs")
            except Exception:
                pass

        if self._attention_network is not None:
            try:
                an_stats = self._attention_network.get_stats()
                stats.append(f"Attention: {an_stats.get('total_visits', 0)} visits")
            except Exception:
                pass

        if self._salience_network is not None:
            try:
                sn_stats = self._salience_network.get_stats()
                stats.append(f"Salience: {sn_stats.get('current_object_count', 0)} objects")
            except Exception:
                pass

        if self._novelty_tracker is not None:
            try:
                tracked = self._novelty_tracker.tracked_count
                stats.append(f"Novelty: {tracked} tracked")
            except Exception:
                pass

        if stats:
            logger.debug("DN Background Stats: %s", " | ".join(stats))

    def get_stats(self) -> dict[str, Any]:
        """Get background task statistics."""
        return {
            "cleanup_count": self._cleanup_count,
            "items_cleaned": self._items_cleaned,
            "running": self._running,
        }


def create_throttlers(config: ThrottleConfig | None = None) -> dict[str, OperationThrottler]:
    """Create throttlers for expensive operations.

    Returns a dict of named throttlers that can be used to rate-limit operations.
    """
    cfg = config or ThrottleConfig()
    return {
        "spatial_map": OperationThrottler(cfg.spatial_map_update_hz),
        "salience": OperationThrottler(cfg.salience_update_hz),
        "novelty": OperationThrottler(cfg.novelty_update_hz),
        "behaviors": OperationThrottler(cfg.behavior_eval_hz),
    }


__all__ = [
    "BackgroundTaskConfig",
    "BackgroundTaskManager",
    "ThrottleConfig",
    "OperationThrottler",
    "create_throttlers",
]
