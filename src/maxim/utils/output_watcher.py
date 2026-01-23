"""File watcher for cross-instance output discovery.

Monitors the shared outputs directory for new files from other Maxim instances
and notifies via the memory system.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from collections.abc import Iterator

logger = logging.getLogger(__name__)


@dataclass
class OutputFile:
    """Represents a file in the shared outputs directory."""

    path: str
    instance_id: str
    filename: str
    size: int
    modified_time: float
    created_time: float
    file_type: str  # Inferred from extension

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "instance_id": self.instance_id,
            "filename": self.filename,
            "size": self.size,
            "modified_time": self.modified_time,
            "created_time": self.created_time,
            "file_type": self.file_type,
        }

    @classmethod
    def from_path(cls, path: str, instance_id: str) -> OutputFile:
        """Create from file path."""
        stat = os.stat(path)
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()

        # Map extensions to types
        type_map = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".txt": "text",
            ".md": "markdown",
            ".csv": "csv",
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
            ".gif": "image",
            ".mp4": "video",
            ".wav": "audio",
            ".mp3": "audio",
            ".py": "python",
            ".sh": "shell",
            ".log": "log",
        }

        return cls(
            path=path,
            instance_id=instance_id,
            filename=filename,
            size=stat.st_size,
            modified_time=stat.st_mtime,
            created_time=stat.st_ctime,
            file_type=type_map.get(ext, "unknown"),
        )


@dataclass
class OutputEvent:
    """Event when a new output file is detected."""

    event_type: str  # "created", "modified", "deleted"
    file: OutputFile | None
    path: str
    instance_id: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "file": self.file.to_dict() if self.file else None,
            "path": self.path,
            "instance_id": self.instance_id,
            "timestamp": self.timestamp,
        }


class OutputWatcher:
    """Watches shared outputs directory for changes from other instances.

    Uses polling (portable) rather than inotify/fsevents (platform-specific).
    """

    def __init__(
        self,
        outputs_dir: str,
        own_instance_id: str,
        poll_interval: float = 2.0,
        on_new_file: Callable[[OutputEvent], None] | None = None,
        on_modified_file: Callable[[OutputEvent], None] | None = None,
        on_deleted_file: Callable[[OutputEvent], None] | None = None,
    ):
        """Initialize watcher.

        Args:
            outputs_dir: Path to shared outputs directory (data/shared/outputs)
            own_instance_id: This instance's ID (to filter out own files)
            poll_interval: Seconds between polls
            on_new_file: Callback for new files
            on_modified_file: Callback for modified files
            on_deleted_file: Callback for deleted files
        """
        self.outputs_dir = os.path.abspath(outputs_dir)
        self.own_instance_id = own_instance_id
        self.poll_interval = poll_interval

        self.on_new_file = on_new_file
        self.on_modified_file = on_modified_file
        self.on_deleted_file = on_deleted_file

        # Track file state
        self._file_state: dict[str, tuple[float, int]] = {}  # path -> (mtime, size)
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Event history for querying
        self._events: list[OutputEvent] = []
        self._events_lock = threading.Lock()
        self._max_events = 1000

    def start(self) -> None:
        """Start watching in background thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        # Initial scan
        self._scan_directory()

        # Start watcher thread
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"OutputWatcher started for {self.outputs_dir}")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop watching."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._thread = None
        logger.info("OutputWatcher stopped")

    def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running and not self._stop_event.is_set():
            try:
                self._check_for_changes()
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")

            self._stop_event.wait(timeout=self.poll_interval)

    def _scan_directory(self) -> None:
        """Initial scan to populate file state."""
        if not os.path.exists(self.outputs_dir):
            return

        for path in self._iter_output_files():
            try:
                stat = os.stat(path)
                self._file_state[path] = (stat.st_mtime, stat.st_size)
            except OSError:
                pass

    def _iter_output_files(self) -> Iterator[str]:
        """Iterate over all output files from other instances."""
        if not os.path.exists(self.outputs_dir):
            return

        for instance_dir in os.listdir(self.outputs_dir):
            instance_path = os.path.join(self.outputs_dir, instance_dir)
            if not os.path.isdir(instance_path):
                continue

            # Skip our own instance
            if instance_dir == self.own_instance_id:
                continue

            # Walk instance directory
            for root, _, files in os.walk(instance_path):
                for f in files:
                    yield os.path.join(root, f)

    def _get_instance_from_path(self, path: str) -> str | None:
        """Extract instance ID from file path."""
        try:
            rel_path = os.path.relpath(path, self.outputs_dir)
            parts = rel_path.split(os.sep)
            if parts:
                return parts[0]
        except ValueError:
            pass
        return None

    def _check_for_changes(self) -> None:
        """Check for new, modified, or deleted files."""
        current_files: set[str] = set()

        for path in self._iter_output_files():
            current_files.add(path)
            instance_id = self._get_instance_from_path(path)
            if instance_id is None:
                continue

            try:
                stat = os.stat(path)
                current_state = (stat.st_mtime, stat.st_size)

                if path not in self._file_state:
                    # New file
                    self._file_state[path] = current_state
                    self._emit_event("created", path, instance_id)

                elif self._file_state[path] != current_state:
                    # Modified file
                    self._file_state[path] = current_state
                    self._emit_event("modified", path, instance_id)

            except OSError:
                pass

        # Check for deleted files
        deleted = set(self._file_state.keys()) - current_files
        for path in deleted:
            instance_id = self._get_instance_from_path(path)
            if instance_id and instance_id != self.own_instance_id:
                del self._file_state[path]
                self._emit_event("deleted", path, instance_id)

    def _emit_event(self, event_type: str, path: str, instance_id: str) -> None:
        """Emit an event and call callbacks."""
        try:
            if event_type == "deleted":
                output_file = None
            else:
                output_file = OutputFile.from_path(path, instance_id)

            event = OutputEvent(
                event_type=event_type,
                file=output_file,
                path=path,
                instance_id=instance_id,
            )

            # Store event
            with self._events_lock:
                self._events.append(event)
                if len(self._events) > self._max_events:
                    self._events = self._events[-self._max_events:]

            # Call callback
            if event_type == "created" and self.on_new_file:
                try:
                    self.on_new_file(event)
                except Exception as e:
                    logger.error(f"Error in on_new_file callback: {e}")

            elif event_type == "modified" and self.on_modified_file:
                try:
                    self.on_modified_file(event)
                except Exception as e:
                    logger.error(f"Error in on_modified_file callback: {e}")

            elif event_type == "deleted" and self.on_deleted_file:
                try:
                    self.on_deleted_file(event)
                except Exception as e:
                    logger.error(f"Error in on_deleted_file callback: {e}")

            logger.debug(f"Output event: {event_type} {path} from {instance_id}")

        except Exception as e:
            logger.error(f"Error emitting event: {e}")

    def get_recent_events(
        self,
        count: int = 50,
        event_types: list[str] | None = None,
        from_instance: str | None = None,
    ) -> list[OutputEvent]:
        """Get recent events with optional filtering.

        Args:
            count: Maximum events to return
            event_types: Filter by event types (e.g., ["created"])
            from_instance: Filter by source instance

        Returns:
            List of matching events (most recent first)
        """
        with self._events_lock:
            events = list(reversed(self._events))

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        if from_instance:
            events = [e for e in events if e.instance_id == from_instance]

        return events[:count]

    def list_other_outputs(self, from_instance: str | None = None) -> list[OutputFile]:
        """List all output files from other instances.

        Args:
            from_instance: Filter to specific instance (None = all others)

        Returns:
            List of OutputFile objects
        """
        outputs: list[OutputFile] = []

        for path in self._iter_output_files():
            instance_id = self._get_instance_from_path(path)
            if instance_id is None:
                continue

            if from_instance and instance_id != from_instance:
                continue

            try:
                outputs.append(OutputFile.from_path(path, instance_id))
            except OSError:
                pass

        # Sort by modified time (newest first)
        outputs.sort(key=lambda f: f.modified_time, reverse=True)
        return outputs

    def get_other_instances(self) -> list[str]:
        """Get list of other instance IDs that have outputs."""
        if not os.path.exists(self.outputs_dir):
            return []

        instances = []
        for name in os.listdir(self.outputs_dir):
            path = os.path.join(self.outputs_dir, name)
            if os.path.isdir(path) and name != self.own_instance_id:
                instances.append(name)

        return sorted(instances)


# ─────────────────────────────────────────────────────────────────────────────
# Memory Integration
# ─────────────────────────────────────────────────────────────────────────────


def create_memory_callback(memory: Any) -> Callable[[OutputEvent], None]:
    """Create a callback that stores output events in memory.

    Args:
        memory: Memory object with store_raw method

    Returns:
        Callback function for OutputWatcher
    """

    def callback(event: OutputEvent) -> None:
        try:
            memory.store_raw(
                content={
                    "type": "cross_instance_output",
                    "event": event.to_dict(),
                    "summary": f"Instance '{event.instance_id}' {event.event_type} file: {event.path}",
                },
                metadata={
                    "source": "output_watcher",
                    "instance_id": event.instance_id,
                    "event_type": event.event_type,
                },
            )
        except Exception as e:
            logger.error(f"Failed to store output event in memory: {e}")

    return callback


# ─────────────────────────────────────────────────────────────────────────────
# Global Watcher Instance
# ─────────────────────────────────────────────────────────────────────────────


_global_watcher: OutputWatcher | None = None


def get_output_watcher() -> OutputWatcher | None:
    """Get the global output watcher."""
    return _global_watcher


def set_output_watcher(watcher: OutputWatcher) -> None:
    """Set the global output watcher."""
    global _global_watcher
    _global_watcher = watcher


def init_output_watcher(
    outputs_dir: str,
    instance_id: str,
    memory: Any | None = None,
    poll_interval: float = 2.0,
) -> OutputWatcher:
    """Initialize the global output watcher.

    Args:
        outputs_dir: Path to shared outputs directory
        instance_id: This instance's ID
        memory: Optional memory object for storing events
        poll_interval: Seconds between polls

    Returns:
        The initialized watcher
    """
    on_new_file = create_memory_callback(memory) if memory else None

    watcher = OutputWatcher(
        outputs_dir=outputs_dir,
        own_instance_id=instance_id,
        poll_interval=poll_interval,
        on_new_file=on_new_file,
    )
    set_output_watcher(watcher)
    return watcher
