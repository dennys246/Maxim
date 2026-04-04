"""Hippocampus persistence mixin — save, load, backup, and recovery."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from maxim.agents.bus import DependencyGraph, EdgeType
from maxim.memory.types import (
    CompressedMemory,
    EpisodicMemory,
)
from maxim.utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


class PersistenceMixin:
    """Persistence methods for Hippocampus.

    Provides save(), load(), load_with_recovery(), save_with_backup(),
    and graph restoration. Mixed into the Hippocampus class.
    """

    def save(self, path: str | None = None) -> None:
        """Save hippocampus to JSON file.

        Args:
            path: File path. Uses config.persistence_path if None.
        """
        path = path or self.config.persistence_path
        if not path:
            raise ValueError("No persistence path specified")

        with self._rwlock.read():
            # Serialize memories (handles both EpisodicMemory and CompressedMemory)
            memories_data = [m.to_dict() for m in self._memories.values()]

            # Serialize index (convert sets to lists)
            index_data = {k: list(v) for k, v in self._context_index.items()}

            # Serialize associative graph edges
            graph_data = self._graph.to_dict()

            payload = {
                "version": "3.0",  # Updated for associative graph support
                "saved_at": time.time(),
                "memories": memories_data,
                "context_index": index_data,
                "stats": dict(self._stats),
                "compressed_count": self._compressed_count,
                "associative_graph": graph_data,
            }

        atomic_write_json(path, payload)

        logger.info("Saved hippocampus to %s (%d memories)", path, len(memories_data))

    def load(self, path: str | None = None) -> None:
        """Load hippocampus from JSON file.

        Args:
            path: File path. Uses config.persistence_path if None.
        """
        path = path or self.config.persistence_path
        if not path:
            raise ValueError("No persistence path specified")

        if not os.path.exists(path):
            logger.warning("Hippocampus file not found: %s", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Version check - support 1.0, 2.0, and 3.0
        version = payload.get("version", "0.0")
        if version not in ("1.0", "2.0", "3.0"):
            raise ValueError(f"Unsupported hippocampus version: {version}")

        # Parse OUTSIDE write lock — if from_dict() raises, existing data is preserved
        temp_memories: dict[str, EpisodicMemory | CompressedMemory] = {}
        temp_compressed_count = 0
        for m_data in payload.get("memories", []):
            if m_data.get("_compressed", False):
                memory = CompressedMemory.from_dict(m_data)
                temp_memories[memory.id] = memory
                temp_compressed_count += 1
            else:
                memory = EpisodicMemory.from_dict(m_data)
                temp_memories[memory.id] = memory

        # Build context index and reverse index from payload
        temp_context_index: defaultdict[str, set[str]] = defaultdict(
            set,
            {k: set(v) for k, v in payload.get("context_index", {}).items()},
        )
        temp_memory_contexts: defaultdict[str, set[str]] = defaultdict(set)
        for index_key, memory_ids in temp_context_index.items():
            for memory_id in memory_ids:
                temp_memory_contexts[memory_id].add(index_key)

        temp_stats = payload.get("stats", {})

        # Restore compressed count from payload if available
        if "compressed_count" in payload:
            temp_compressed_count = payload["compressed_count"]

        # Build associative graph
        temp_graph = DependencyGraph()
        graph_data = payload.get("associative_graph")

        # All parsing succeeded — atomically swap inside the write lock
        with self._rwlock.write():
            self._memories.clear()
            self._memories.update(temp_memories)
            self._compressed_count = temp_compressed_count
            self._context_index = temp_context_index
            self._memory_contexts = temp_memory_contexts
            self._stats = temp_stats
            self._graph = temp_graph
            if graph_data:
                self._restore_graph(graph_data)

        edge_count = sum(
            len(self._graph.get_associated(mid))
            for mid in self._memories
            if self._graph.get_node(mid) is not None
        )

        logger.info(
            "Loaded hippocampus from %s (%d memories, %d compressed, %d edges)",
            path,
            len(self._memories),
            self._compressed_count,
            edge_count,
        )

    def load_with_recovery(
        self,
        path: str | None = None,
        on_error: str = "warn_and_continue",
    ) -> tuple[bool, str | None]:
        """Load with automatic recovery on failure.

        Args:
            path: File path. Uses config.persistence_path if None.
            on_error: "warn_and_continue" | "restore_backup" | "raise"

        Returns:
            (success, error_message) - error_message is None on success.
        """
        path = path or self.config.persistence_path
        if not path:
            return False, "No persistence path specified"

        if not os.path.exists(path):
            logger.info("No existing hippocampus file, starting fresh")
            return True, None

        try:
            self.load(path)
            return True, None

        except json.JSONDecodeError as e:
            error_msg = f"Corrupt hippocampus file: {e}"
            logger.warning(error_msg)

            if on_error == "restore_backup":
                backup_path = f"{path}.backup"
                if os.path.exists(backup_path):
                    try:
                        self.load(backup_path)
                        logger.info("Restored from backup: %s", backup_path)
                        return True, "Restored from backup (original corrupt)"
                    except Exception as be:
                        logger.error("Backup also corrupt: %s", be)

            if on_error == "raise":
                raise

            # warn_and_continue: start fresh
            logger.warning("Starting with empty hippocampus due to corrupt file")
            return True, error_msg

        except Exception as e:
            error_msg = f"Failed to load hippocampus: {e}"
            logger.error(error_msg)

            if on_error == "raise":
                raise

            return True, error_msg

    def save_with_backup(self, path: str | None = None) -> None:
        """Save with automatic backup of previous version.

        Args:
            path: File path. Uses config.persistence_path if None.
        """
        import shutil

        path = path or self.config.persistence_path
        if not path:
            raise ValueError("No persistence path specified")

        # Create backup of existing file
        if os.path.exists(path):
            backup_path = f"{path}.backup"
            try:
                shutil.copy2(path, backup_path)
            except Exception as e:
                logger.warning("Failed to create backup: %s", e)

        # Save normally
        self.save(path)

    def _restore_graph(self, graph_data: dict[str, Any]) -> None:
        """Restore the associative graph from serialized data (lock must be held).

        Reconstructs nodes and edges from the graph's to_dict() output.

        Args:
            graph_data: Serialized graph from DependencyGraph.to_dict().
        """
        # Restore nodes - only for memories that still exist
        for node_id in graph_data.get("nodes", []):
            if node_id in self._memories:
                self._graph.add_node(node_id, node_id)

        # Restore edges (only ASSOCIATES edges, skip duplicates from bidirectional)
        seen_pairs: set[tuple[str, str]] = set()
        for edge_data in graph_data.get("edges", []):
            source = edge_data.get("source", "")
            target = edge_data.get("target", "")
            weight = edge_data.get("weight", 1.0)
            edge_type_name = edge_data.get("type", "ASSOCIATES")

            # Only restore association edges
            if edge_type_name != "ASSOCIATES":
                continue

            # Skip if either memory no longer exists
            if source not in self._memories or target not in self._memories:
                continue

            # Avoid duplicating bidirectional edges
            pair = (min(source, target), max(source, target))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Ensure nodes exist
            if self._graph.get_node(source) is None:
                self._graph.add_node(source, source)
            if self._graph.get_node(target) is None:
                self._graph.add_node(target, target)

            self._graph.add_bidirectional(source, target, EdgeType.ASSOCIATES, weight)
