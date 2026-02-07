"""Hippocampus - Associative memory graph for agentic loops.

Phase 1 Implementation:
- Core capture and retrieval
- Hash index for O(1) lookups
- JSON persistence
- RWLock for concurrent access
- StateStore for state snapshot caching

See hippocampus_plan.md for full design and future phases.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Union
from uuid import uuid4

if TYPE_CHECKING:
    from maxim.memory.strategies import MemoryStrategy
    from maxim.time.scn import SCN

from maxim.memory.rwlock import RWLock
from maxim.memory.state_store import StateStore
from maxim.memory.types import (
    Action,
    CompressedMemory,
    Context,
    Decision,
    EpisodicMemory,
    Outcome,
    Perception,
)

logger = logging.getLogger(__name__)


@dataclass
class HippocampusConfig:
    """Configuration for the Hippocampus."""

    # Maximum nodes before warning (not enforced in Phase 1)
    max_nodes: int = 10_000

    # State store capacity
    state_store_max_entries: int = 1000

    # Persistence path (None = no auto-persistence)
    persistence_path: str | None = None

    # Index keys that should be tracked
    indexed_keys: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"goal", "tool", "object", "person", "success", "mode"}
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Sleep Consolidation Settings
    # ─────────────────────────────────────────────────────────────────────────

    # Enable sleep consolidation
    enable_sleep_consolidation: bool = True

    # Default time without access before memory is candidate for removal (1 week)
    max_age_without_access: float = 7 * 24 * 3600  # 604800 seconds

    # Time after which full memories are compressed to CompressedMemory (1 day)
    compression_age: float = 24 * 3600  # 86400 seconds

    # Retention threshold: memories scoring below this are removed
    retention_threshold: float = 0.3

    # Compression threshold: memories scoring below this get compressed
    compression_threshold: float = 0.6

    # Memory strategy to use: "access_based", "importance_based", "composite"
    memory_strategy: str = "access_based"

    # ─────────────────────────────────────────────────────────────────────────
    # Long-Term Memory Consolidation
    # ─────────────────────────────────────────────────────────────────────────

    # Run consolidation during sleep()
    consolidate_during_sleep: bool = True

    # Maximum candidates to track for consolidation (bounded queue)
    max_consolidation_candidates: int = 500

    # Immediate promotion thresholds (at capture time)
    immediate_promotion_salience: float = 0.95
    immediate_promotion_novelty: float = 0.95

    # Retention score boost for long-term memories (multiplicative)
    long_term_retention_boost: float = 2.0

    # Auto-save after sleep consolidation
    auto_save_after_sleep: bool = True


class Hippocampus:
    """Hybrid hash-table graph for associative memory of agentic loops.

    Each node stores a complete EpisodicMemory (observe -> decide -> act -> outcome).
    Combines O(1) context lookup with graph-based relational queries.
    Thread-safe for concurrent access via RWLock.

    Phase 1 Features:
    - capture(): Store a complete agentic loop
    - get(): Retrieve by memory_id
    - recall(): Query with filters
    - recall_similar(): Find memories similar to current perception
    - save()/load(): JSON persistence

    Example:
        hippocampus = Hippocampus()

        # Capture a memory
        memory_id = hippocampus.capture(
            perception=Perception(detected_objects=["cup"]),
            context=Context(active_goal="find cup"),
            decision=Decision(intent={"goal": "pick up cup"}),
            action=Action(tool_name="grasp"),
            outcome=Outcome(success=True),
        )

        # Query memories
        memories = hippocampus.recall(goal="find cup", success=True)
    """

    def __init__(self, config: HippocampusConfig | None = None) -> None:
        self.config = config or HippocampusConfig()

        # Primary storage: memory_id -> EpisodicMemory or CompressedMemory
        self._memories: dict[str, Union[EpisodicMemory, CompressedMemory]] = {}

        # Hash index: context_key -> set of memory_ids
        # e.g., "goal:find cup" -> {"mem_123", "mem_456"}
        self._context_index: dict[str, set[str]] = defaultdict(set)

        # Reverse index for cleanup: memory_id -> set of context_keys
        self._memory_contexts: dict[str, set[str]] = defaultdict(set)

        # State store for full snapshots (EpisodicMemory only stores hash reference)
        self._state_store = StateStore(max_entries=self.config.state_store_max_entries)

        # Thread safety - RWLock for concurrent reads, exclusive writes
        self._rwlock = RWLock()

        # Statistics
        self._stats = {
            "memories_captured": 0,
            "queries": 0,
        }

        # Deletion callbacks for subsystem cleanup
        self._on_memory_deleted: list[Callable[[str], None]] = []

        # Capture callbacks for subsystem notification (e.g., semantic embedding)
        self._on_memory_captured: list[Callable[[str, "EpisodicMemory"], None]] = []

        # Consolidation candidates queue (memories to evaluate for long-term promotion)
        self._consolidation_candidates: list[str] = []

        # Track compressed vs full memory counts
        self._compressed_count: int = 0

        # Optional SCN reference for temporal-aware strategies
        self._scn: "SCN | None" = None

    # ─────────────────────────────────────────────────────────────────────────
    # SCN Integration
    # ─────────────────────────────────────────────────────────────────────────

    def connect_scn(self, scn: "SCN") -> None:
        """Connect SCN for temporal-aware sleep consolidation.

        When connected, the Hippocampus will:
        - Use TemporalAwareStrategy for smarter memory retention
        - Protect sole representatives of time slots
        - Preserve memories in rhythmic patterns
        - Enable temporal clustering for batch operations

        Args:
            scn: SCN instance to use for temporal queries
        """
        self._scn = scn
        # Register for memory deletion notifications
        self.register_deletion_callback(scn.remove_memory)
        logger.info("Connected SCN to Hippocampus (temporal-aware consolidation enabled)")

    @property
    def scn(self) -> "SCN | None":
        """Get the connected SCN, if any."""
        return self._scn

    # ─────────────────────────────────────────────────────────────────────────
    # Core Operations
    # ─────────────────────────────────────────────────────────────────────────

    def capture(
        self,
        perception: Perception,
        context: Context,
        decision: Decision,
        action: Action,
        outcome: Outcome,
        run_id: str = "",
        *,
        state_snapshot: dict[str, Any] | None = None,
    ) -> str:
        """Capture a complete agentic loop as an episodic memory.

        Args:
            perception: What was observed.
            context: State context at decision time.
            decision: What was decided and why.
            action: What action was taken.
            outcome: What happened as a result.
            run_id: Optional run identifier.
            state_snapshot: Optional full state to store in StateStore.

        Returns:
            The memory_id of the captured memory.
        """
        memory_id = str(uuid4())
        now = time.time()

        # Store state snapshot and get reference
        if state_snapshot is not None:
            context.state_ref = self._state_store.store(state_snapshot)

        # Create the memory
        memory = EpisodicMemory(
            id=memory_id,
            timestamp=now,
            run_id=run_id,
            created_at=now,
            accessed_at=now,
            access_count=1,
            perception=perception,
            context=context,
            decision=decision,
            action=action,
            outcome=outcome,
        )

        with self._rwlock.write():
            # Store the memory
            self._memories[memory_id] = memory

            # Build index entries
            self._index_memory(memory_id, memory)

            # Update stats
            self._stats["memories_captured"] += 1

            if outcome.success:
                self._stats["successful"] = self._stats.get("successful", 0) + 1
            else:
                self._stats["failed"] = self._stats.get("failed", 0) + 1

            # Check for immediate promotion to long-term (very high importance)
            if (
                perception.salience >= self.config.immediate_promotion_salience
                or perception.novelty >= self.config.immediate_promotion_novelty
            ):
                memory.long_term = True
                memory.consolidated_at = now
                self._stats["long_term_count"] = self._stats.get("long_term_count", 0) + 1
                logger.debug("Immediately promoted memory %s to long-term", memory_id[:8])

            # Add to consolidation candidates if potentially important
            elif (
                perception.salience > 0.7
                or perception.novelty > 0.7
                or perception.cli_input
                or perception.transcript
            ):
                self._add_consolidation_candidate(memory_id)

        logger.debug(
            "Captured memory %s: goal=%s, tool=%s, success=%s",
            memory_id[:8],
            context.active_goal,
            action.tool_name,
            outcome.success,
        )

        # Notify capture callbacks (e.g., for semantic embedding)
        for callback in self._on_memory_captured:
            try:
                callback(memory_id, memory)
            except Exception as e:
                logger.warning("Memory capture callback failed: %s", e)

        return memory_id

    def capture_from_loop(
        self,
        observation: dict[str, Any],
        state: Any,
        intent: dict[str, Any],
        decision: dict[str, Any],
        action: dict[str, Any],
        result: Any,
        evaluations: list[dict[str, Any]] | None = None,
        run_id: str = "",
    ) -> str:
        """Convenience method to capture from agent_loop outputs.

        Converts the raw loop data into structured EpisodicMemory components.
        """
        # Extract perception
        perception = Perception(
            observations=observation if isinstance(observation, dict) else {},
            cli_input=observation.get("cli_input") if isinstance(observation, dict) else None,
            transcript=observation.get("transcript") if isinstance(observation, dict) else None,
            detected_objects=observation.get("detected_objects", []) if isinstance(observation, dict) else [],
            detected_people=observation.get("detected_people", []) if isinstance(observation, dict) else [],
            salience=observation.get("salience", 0.5) if isinstance(observation, dict) else 0.5,
            novelty=observation.get("novelty", 0.5) if isinstance(observation, dict) else 0.5,
        )

        # Extract state snapshot
        state_snapshot = None
        if hasattr(state, "snapshot") and callable(state.snapshot):
            try:
                state_snapshot = state.snapshot()
            except Exception:
                pass
        elif hasattr(state, "data"):
            state_snapshot = dict(state.data) if hasattr(state.data, "items") else {}

        # Build context
        ctx = Context(
            active_goal=intent.get("goal") if intent else None,
            active_mode=state_snapshot.get("mode", "observe") if state_snapshot else "observe",
        )

        # Build decision
        dec = Decision(
            intent=intent or {},
            reasoning=intent.get("reasoning") if intent else None,
            confidence=intent.get("confidence", 1.0) if intent else 1.0,
        )

        # Build action
        act = Action(
            tool_name=action.get("tool", "") if action else "",
            tool_params=action.get("params", {}) if action else {},
        )

        # Build outcome
        success = True
        error = None
        if hasattr(result, "success"):
            success = bool(result.success)
        if hasattr(result, "error"):
            error = str(result.error) if result.error else None

        out = Outcome(
            success=success,
            result=result if not hasattr(result, "to_dict") else result.to_dict() if hasattr(result, "to_dict") else str(result),
            error=error,
            evaluations=evaluations or [],
        )

        return self.capture(
            perception=perception,
            context=ctx,
            decision=dec,
            action=act,
            outcome=out,
            run_id=run_id,
            state_snapshot=state_snapshot,
        )

    def get(self, memory_id: str) -> Union[EpisodicMemory, CompressedMemory] | None:
        """Get a memory by its ID.

        Updates the access timestamp and count.

        Args:
            memory_id: The memory ID to retrieve.

        Returns:
            The memory (EpisodicMemory or CompressedMemory), or None if not found.
        """
        with self._rwlock.write():  # Write lock because we update access tracking
            memory = self._memories.get(memory_id)
            if memory is not None:
                memory.accessed_at = time.time()
                memory.access_count += 1
            return memory

    def recall(
        self,
        limit: int = 10,
        *,
        goal: str | None = None,
        tool: str | None = None,
        success: bool | None = None,
        object_detected: str | None = None,
        person_detected: str | None = None,
        mode: str | None = None,
        time_after: float | None = None,
        time_before: float | None = None,
        include_compressed: bool = True,
    ) -> list[Union[EpisodicMemory, CompressedMemory]]:
        """Find memories matching filters (hash lookup + filtering).

        All filters are AND-ed together. Use None to skip a filter.

        Args:
            limit: Maximum number of memories to return.
            goal: Match memories with this goal text.
            tool: Match memories that used this tool.
            success: Match memories with this success status.
            object_detected: Match memories that detected this object.
            person_detected: Match memories that detected this person.
            mode: Match memories in this mode.
            time_after: Only memories after this timestamp.
            time_before: Only memories before this timestamp.
            include_compressed: Include CompressedMemory results (default True).

        Returns:
            List of matching memories (EpisodicMemory or CompressedMemory), most recent first.
        """
        with self._rwlock.write():  # Write for stats update
            self._stats["queries"] = self._stats.get("queries", 0) + 1

            # Build index filters
            index_filters: dict[str, Any] = {}
            if goal is not None:
                index_filters["goal"] = goal
            if tool is not None:
                index_filters["tool"] = tool
            if success is not None:
                index_filters["success"] = success
            if object_detected is not None:
                index_filters["object"] = object_detected
            if person_detected is not None:
                index_filters["person"] = person_detected
            if mode is not None:
                index_filters["mode"] = mode

            # Find candidates via index intersection
            candidates: set[str] | None = None

            if index_filters:
                for key, value in index_filters.items():
                    index_key = self._make_index_key(key, value)
                    matching = self._context_index.get(index_key, set())
                    if candidates is None:
                        candidates = matching.copy()
                    else:
                        candidates &= matching

                    # Early exit if no matches
                    if not candidates:
                        return []
            else:
                # No filters - all memories are candidates
                candidates = set(self._memories.keys())

            # Filter by time range and collect results
            results: list[Union[EpisodicMemory, CompressedMemory]] = []
            for memory_id in candidates:
                memory = self._memories.get(memory_id)
                if memory is None:
                    continue

                # Skip compressed if not wanted
                if not include_compressed and isinstance(memory, CompressedMemory):
                    continue

                # Time filters
                if time_after is not None and memory.timestamp < time_after:
                    continue
                if time_before is not None and memory.timestamp > time_before:
                    continue

                results.append(memory)

            # Sort by timestamp (most recent first) and limit
            results.sort(key=lambda m: m.timestamp, reverse=True)
            return results[:limit]

    def recall_similar(
        self,
        perception: Perception,
        limit: int = 5,
        include_compressed: bool = True,
    ) -> list[Union[EpisodicMemory, CompressedMemory]]:
        """Find memories similar to the given perception.

        Scores memories by overlapping detected objects and people.
        Phase 4 will add semantic similarity via embeddings.

        Args:
            perception: Current perception to match against.
            limit: Maximum number of memories to return.
            include_compressed: Include CompressedMemory results.

        Returns:
            List of similar memories, most similar first.
        """
        with self._rwlock.write():  # Write for stats
            self._stats["queries"] = self._stats.get("queries", 0) + 1

            scores: dict[str, float] = defaultdict(float)

            # Score by overlapping detected objects
            for obj in perception.detected_objects:
                index_key = f"object:{obj}"
                for memory_id in self._context_index.get(index_key, set()):
                    scores[memory_id] += 1.0

            # Score by overlapping detected people
            for person in perception.detected_people:
                index_key = f"person:{person}"
                for memory_id in self._context_index.get(index_key, set()):
                    scores[memory_id] += 1.0

            # Sort by score, then by timestamp
            def get_timestamp(mid: str) -> float:
                mem = self._memories.get(mid)
                return mem.timestamp if mem else 0.0

            scored_ids = sorted(
                scores.keys(),
                key=lambda mid: (scores[mid], get_timestamp(mid)),
                reverse=True,
            )

            # Collect results
            results: list[Union[EpisodicMemory, CompressedMemory]] = []
            for memory_id in scored_ids[:limit]:
                memory = self._memories.get(memory_id)
                if memory is not None:
                    if not include_compressed and isinstance(memory, CompressedMemory):
                        continue
                    results.append(memory)

            return results

    def get_state(self, state_ref: str) -> dict[str, Any] | None:
        """Retrieve a full state snapshot from the StateStore.

        Args:
            state_ref: The state reference from a memory's context.state_ref.

        Returns:
            The full state dictionary, or None if evicted.
        """
        return self._state_store.get(state_ref)

    # ─────────────────────────────────────────────────────────────────────────
    # Index Views (for subsystem access)
    # ─────────────────────────────────────────────────────────────────────────

    def get_memories_by_index(self, index_key: str) -> list[EpisodicMemory]:
        """Get all memories matching an index key.

        Args:
            index_key: Full index key like "goal:find cup" or "tool:speak".

        Returns:
            List of matching memories.
        """
        with self._rwlock.read():
            memory_ids = self._context_index.get(index_key, set())
            return [
                self._memories[mid]
                for mid in memory_ids
                if mid in self._memories
            ]

    def index_keys(self) -> list[str]:
        """Return all index keys."""
        with self._rwlock.read():
            return list(self._context_index.keys())

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

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

            payload = {
                "version": "2.0",  # Updated for Phase 3 (CompressedMemory support)
                "saved_at": time.time(),
                "memories": memories_data,
                "context_index": index_data,
                "stats": dict(self._stats),
                "compressed_count": self._compressed_count,
            }

        # Write atomically
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, path)

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

        # Version check - support 1.0 and 2.0
        version = payload.get("version", "0.0")
        if version not in ("1.0", "2.0"):
            raise ValueError(f"Unsupported hippocampus version: {version}")

        with self._rwlock.write():
            # Clear existing data
            self._memories.clear()
            self._context_index.clear()
            self._memory_contexts.clear()
            self._compressed_count = 0

            # Load memories (detect compressed vs full)
            for m_data in payload.get("memories", []):
                if m_data.get("_compressed", False):
                    # CompressedMemory
                    memory = CompressedMemory.from_dict(m_data)
                    self._memories[memory.id] = memory
                    self._compressed_count += 1
                else:
                    # Full EpisodicMemory
                    memory = EpisodicMemory.from_dict(m_data)
                    self._memories[memory.id] = memory

            # Load index
            self._context_index = defaultdict(
                set,
                {k: set(v) for k, v in payload.get("context_index", {}).items()},
            )

            # Rebuild reverse index
            for index_key, memory_ids in self._context_index.items():
                for memory_id in memory_ids:
                    self._memory_contexts[memory_id].add(index_key)

            # Load stats
            self._stats = payload.get("stats", {})

            # Restore compressed count from payload if available
            if "compressed_count" in payload:
                self._compressed_count = payload["compressed_count"]

        logger.info(
            "Loaded hippocampus from %s (%d memories, %d compressed)",
            path,
            len(self._memories),
            self._compressed_count,
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
                        return True, f"Restored from backup (original corrupt)"
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

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics and Debugging
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return hippocampus statistics."""
        with self._rwlock.read():
            # Count long-term memories
            long_term_count = sum(
                1 for m in self._memories.values() if getattr(m, "long_term", False)
            )

            return {
                **self._stats,
                "total_memories": len(self._memories),
                "compressed_memories": self._compressed_count,
                "full_memories": len(self._memories) - self._compressed_count,
                "long_term_memories": long_term_count,
                "index_keys": len(self._context_index),
                "consolidation_candidates": len(self._consolidation_candidates),
                "state_store": self._state_store.stats(),
            }

    def check_consistency(self) -> list[str]:
        """Check for consistency issues between memories and indices.

        Returns:
            List of issue descriptions (empty if consistent).
        """
        issues = []

        with self._rwlock.read():
            # Check: All indexed memory IDs exist
            for index_key, memory_ids in self._context_index.items():
                for memory_id in memory_ids:
                    if memory_id not in self._memories:
                        issues.append(f"Orphan index entry: {index_key} -> {memory_id[:8]}")

            # Check: All memories are indexed
            for memory_id in self._memories:
                if memory_id not in self._memory_contexts:
                    issues.append(f"Unindexed memory: {memory_id[:8]}")

            # Check: Reverse index consistency
            for memory_id, index_keys in self._memory_contexts.items():
                for index_key in index_keys:
                    if memory_id not in self._context_index.get(index_key, set()):
                        issues.append(f"Reverse index mismatch: {memory_id[:8]} claims {index_key}")

        return issues

    def repair_consistency(self) -> int:
        """Repair consistency issues.

        Returns:
            Number of issues repaired.
        """
        with self._rwlock.write():
            repaired = 0

            # Remove orphan index entries
            for index_key in list(self._context_index.keys()):
                original_count = len(self._context_index[index_key])
                valid_ids = {
                    mid for mid in self._context_index[index_key]
                    if mid in self._memories
                }
                removed = original_count - len(valid_ids)
                if removed > 0:
                    self._context_index[index_key] = valid_ids
                    repaired += removed

                # Remove empty index entries
                if not self._context_index[index_key]:
                    del self._context_index[index_key]

            # Rebuild reverse index
            self._memory_contexts.clear()
            for index_key, memory_ids in self._context_index.items():
                for memory_id in memory_ids:
                    self._memory_contexts[memory_id].add(index_key)

            # Re-index any unindexed memories
            for memory_id, memory in self._memories.items():
                if memory_id not in self._memory_contexts or not self._memory_contexts[memory_id]:
                    self._index_memory(memory_id, memory)
                    repaired += 1

            if repaired > 0:
                logger.info("Repaired %d consistency issues", repaired)

            return repaired

    def __len__(self) -> int:
        """Return number of memories."""
        with self._rwlock.read():
            return len(self._memories)

    def __iter__(self) -> Iterator[Union[EpisodicMemory, CompressedMemory]]:
        """Iterate over all memories (both full and compressed)."""
        with self._rwlock.read():
            yield from self._memories.values()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _index_memory(self, memory_id: str, memory: EpisodicMemory) -> None:
        """Build hash index entries for a memory (lock must be held)."""
        index_entries: list[str] = []

        # Index by goal
        if memory.context.active_goal:
            index_entries.append(f"goal:{memory.context.active_goal}")

        # Index by tool
        if memory.action.tool_name:
            index_entries.append(f"tool:{memory.action.tool_name}")

        # Index by success/failure
        index_entries.append(f"success:{memory.outcome.success}")

        # Index by mode
        if memory.context.active_mode:
            index_entries.append(f"mode:{memory.context.active_mode}")

        # Index by detected objects
        for obj in memory.perception.detected_objects:
            index_entries.append(f"object:{obj}")

        # Index by detected people
        for person in memory.perception.detected_people:
            index_entries.append(f"person:{person}")

        # Add all entries to indices
        for index_key in index_entries:
            self._context_index[index_key].add(memory_id)
            self._memory_contexts[memory_id].add(index_key)

    def _make_index_key(self, key: str, value: Any) -> str:
        """Create an index key from a key-value pair."""
        return f"{key}:{value}"

    def _remove_memory(self, memory_id: str) -> None:
        """Remove a memory and clean up all references (lock must be held).

        This method handles:
        1. Context index cleanup
        2. Subsystem notification via callbacks
        """
        # Clean up context index
        for index_key in self._memory_contexts.get(memory_id, set()):
            self._context_index[index_key].discard(memory_id)
            if not self._context_index[index_key]:
                del self._context_index[index_key]

        self._memory_contexts.pop(memory_id, None)

        # Notify subsystems of deletion
        for callback in self._on_memory_deleted:
            try:
                callback(memory_id)
            except Exception as e:
                logger.warning("Memory deletion callback failed: %s", e)

        self._memories.pop(memory_id, None)

    def register_deletion_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback to be notified when memories are deleted.

        Subsystems like SCN, NAc, and EC can register to clean up their
        own indices when memories are removed.

        Args:
            callback: Function that takes memory_id and handles cleanup.
        """
        self._on_memory_deleted.append(callback)

    def register_capture_callback(
        self,
        callback: Callable[[str, "EpisodicMemory"], None],
    ) -> None:
        """Register a callback to be notified when memories are captured.

        Subsystems like EC can register to schedule async semantic embedding
        when new memories are captured.

        Args:
            callback: Function that takes (memory_id, memory) and handles processing.
        """
        self._on_memory_captured.append(callback)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Sleep Consolidation
    # ─────────────────────────────────────────────────────────────────────────

    def sleep(
        self,
        strategy: "MemoryStrategy | None" = None,
    ) -> dict[str, int]:
        """Perform sleep-like memory consolidation.

        This is the primary memory management process, inspired by biological
        sleep consolidation. It:
        1. Promotes important memories to long-term storage
        2. Compresses old, infrequently-accessed full memories
        3. Removes stale memories that haven't been accessed in too long

        Should be called periodically (e.g., during idle time, end of session).

        Args:
            strategy: Memory strategy to use. Defaults to config.memory_strategy.

        Returns:
            Dict with counts: {"compressed": N, "removed": M, "preserved": P, "promoted": Q}
        """
        from maxim.memory.strategies import MemoryStrategy

        with self._rwlock.write():
            return self._sleep(strategy)

    def _sleep(
        self,
        strategy: "MemoryStrategy | None" = None,
    ) -> dict[str, int]:
        """Internal sleep implementation (lock must be held)."""
        if not self.config.enable_sleep_consolidation:
            return {
                "compressed": 0,
                "removed": 0,
                "preserved": len(self._memories),
                "promoted": 0,
            }

        # Step 1: Consolidate important memories to long-term (before any removal)
        consolidation_results = {"promoted": 0}
        if self.config.consolidate_during_sleep:
            consolidation_results = self._consolidate()

        # Get or create strategy
        if strategy is None:
            strategy = self._get_memory_strategy()

        now = time.time()
        results = {
            "compressed": 0,
            "removed": 0,
            "preserved": 0,
            "promoted": consolidation_results.get("promoted", 0),
        }

        # Process each memory
        memories_to_remove: list[str] = []
        memories_to_compress: list[str] = []

        for memory_id, record in list(self._memories.items()):
            # Skip if already compressed
            if isinstance(record, CompressedMemory):
                # Still evaluate for removal
                score = strategy.score_for_retention(record, now, 0)
                # Apply long-term boost
                if record.long_term:
                    score *= self.config.long_term_retention_boost
                    score = min(1.0, score)

                if score < self.config.retention_threshold:
                    memories_to_remove.append(memory_id)
                else:
                    results["preserved"] += 1
                continue

            # Full EpisodicMemory - evaluate
            score = strategy.score_for_retention(record, now, 0)
            # Apply long-term boost
            if record.long_term:
                score *= self.config.long_term_retention_boost
                score = min(1.0, score)

            if score < self.config.retention_threshold:
                # Remove entirely
                memories_to_remove.append(memory_id)
            elif score < self.config.compression_threshold:
                # Candidate for compression
                if strategy.should_compress(record, now, 0):
                    memories_to_compress.append(memory_id)
                else:
                    results["preserved"] += 1
            else:
                # Keep as full record
                results["preserved"] += 1

        # Perform compression
        for memory_id in memories_to_compress:
            self._compress_memory(memory_id)
            results["compressed"] += 1

        # Perform removal
        for memory_id in memories_to_remove:
            self._remove_memory(memory_id)
            results["removed"] += 1

        # Update stats
        self._stats["consolidations"] = self._stats.get("consolidations", 0) + 1
        self._stats["compressions"] = self._stats.get("compressions", 0) + results["compressed"]
        self._stats["removals"] = self._stats.get("removals", 0) + results["removed"]

        # Auto-save if configured
        if self.config.auto_save_after_sleep and self.config.persistence_path:
            self.save_with_backup(self.config.persistence_path)

        logger.info(
            "Sleep consolidation complete: compressed=%d, removed=%d, preserved=%d, promoted=%d",
            results["compressed"],
            results["removed"],
            results["preserved"],
            results["promoted"],
        )

        return results

    def _compress_memory(self, memory_id: str) -> None:
        """Compress a full EpisodicMemory to CompressedMemory (lock must be held)."""
        if memory_id not in self._memories:
            return

        record = self._memories[memory_id]
        if not isinstance(record, EpisodicMemory):
            return  # Already compressed or invalid

        # Create compressed version
        compressed = CompressedMemory.from_episodic(record, edge_count=0)

        # Replace in storage
        self._memories[memory_id] = compressed  # type: ignore[assignment]
        self._compressed_count += 1

        logger.debug("Compressed memory %s", memory_id[:8])

    # ─────────────────────────────────────────────────────────────────────────
    # Long-Term Memory Consolidation
    # ─────────────────────────────────────────────────────────────────────────

    def consolidate(
        self,
        *,
        force_ids: list[str] | None = None,
        max_promotions: int = 100,
    ) -> dict[str, int]:
        """Consolidate important short-term memories into long-term storage.

        Long-term memories are resistant (but not immune) to removal during
        sleep consolidation. This mimics biological memory consolidation
        where important experiences get "burned in" over time.

        Can be called:
        - Automatically during sleep() (default behavior)
        - Manually to force promotion of specific memories
        - Periodically during idle time

        Args:
            force_ids: Specific memory IDs to promote (bypasses threshold check).
            max_promotions: Maximum memories to promote in one call.

        Returns:
            Dict with counts: {"promoted": N, "already_long_term": M, "skipped": P}
        """
        with self._rwlock.write():
            return self._consolidate(force_ids=force_ids, max_promotions=max_promotions)

    def _consolidate(
        self,
        *,
        force_ids: list[str] | None = None,
        max_promotions: int = 100,
    ) -> dict[str, int]:
        """Internal consolidation (lock must be held)."""
        now = time.time()
        results = {"promoted": 0, "already_long_term": 0, "skipped": 0}

        # Process forced promotions first
        if force_ids:
            for memory_id in force_ids[:max_promotions]:
                promoted = self._promote_to_long_term(memory_id, now)
                if promoted:
                    results["promoted"] += 1
                else:
                    results["already_long_term"] += 1
            return results

        # Evaluate candidates from the queue
        candidates_to_evaluate = list(self._consolidation_candidates)
        self._consolidation_candidates.clear()

        for memory_id in candidates_to_evaluate:
            if results["promoted"] >= max_promotions:
                # Re-queue remaining candidates for next consolidation
                self._consolidation_candidates.append(memory_id)
                continue

            if memory_id not in self._memories:
                continue

            record = self._memories[memory_id]
            if getattr(record, "long_term", False):
                results["already_long_term"] += 1
                continue

            # Check promotion criteria
            if self._should_promote(record, now):
                if self._promote_to_long_term(memory_id, now):
                    results["promoted"] += 1
                else:
                    results["already_long_term"] += 1
            else:
                results["skipped"] += 1

        return results

    def _should_promote(
        self,
        record: EpisodicMemory | CompressedMemory,
        now: float,
    ) -> bool:
        """Decide if a memory should be promoted to long-term.

        Promotion criteria (any of these triggers promotion):
        1. Very high salience (> 0.9)
        2. Very high novelty (> 0.9)
        3. User interaction with successful outcome
        4. High access count (> 5)
        """
        # Extract fields (works for both types)
        if isinstance(record, CompressedMemory):
            novelty = record.novelty
            salience = record.salience
            had_user_input = record.had_user_input
            success = record.success
            access_count = record.access_count
        else:
            novelty = record.perception.novelty
            salience = record.perception.salience
            had_user_input = bool(record.perception.cli_input or record.perception.transcript)
            success = record.outcome.success
            access_count = record.access_count

        # Criterion 1: Very high salience
        if salience > 0.9:
            return True

        # Criterion 2: Very high novelty
        if novelty > 0.9:
            return True

        # Criterion 3: Successful user interaction
        if had_user_input and success:
            return True

        # Criterion 4: Frequently accessed
        if access_count >= 5:
            return True

        return False

    def _promote_to_long_term(self, memory_id: str, now: float) -> bool:
        """Promote a memory to long-term status (lock must be held)."""
        if memory_id not in self._memories:
            return False

        record = self._memories[memory_id]
        if record is None:
            return False

        if getattr(record, "long_term", False):
            return False  # Already long-term

        # Update the record
        record.long_term = True
        record.consolidated_at = now

        self._stats["long_term_count"] = self._stats.get("long_term_count", 0) + 1
        logger.debug("Promoted memory %s to long-term", memory_id[:8])
        return True

    def _add_consolidation_candidate(self, memory_id: str) -> None:
        """Add a memory to the consolidation candidate queue (lock must be held).

        Called during capture() for memories that might be promotable but
        don't meet immediate promotion criteria.
        """
        # Bounded queue to prevent unbounded growth
        if len(self._consolidation_candidates) < self.config.max_consolidation_candidates:
            self._consolidation_candidates.append(memory_id)

    def _get_memory_strategy(self) -> "MemoryStrategy":
        """Get the configured memory strategy.

        If SCN is connected, wraps the base strategy with TemporalAwareStrategy
        for temporal-context-aware memory consolidation.
        """
        from maxim.memory.strategies import (
            AccessBasedStrategy,
            CompositeStrategy,
            ImportanceBasedStrategy,
            TemporalAwareStrategy,
        )

        strategy_name = self.config.memory_strategy

        # Build base strategy
        if strategy_name == "access_based":
            base_strategy = AccessBasedStrategy(
                max_age_without_access=self.config.max_age_without_access,
                compression_age=self.config.compression_age,
            )
        elif strategy_name == "importance_based":
            base_strategy = ImportanceBasedStrategy(
                compression_age=self.config.compression_age,
            )
        elif strategy_name == "composite":
            base_strategy = CompositeStrategy(
                [
                    (
                        AccessBasedStrategy(
                            max_age_without_access=self.config.max_age_without_access,
                            compression_age=self.config.compression_age,
                        ),
                        0.6,
                    ),
                    (
                        ImportanceBasedStrategy(
                            compression_age=self.config.compression_age,
                        ),
                        0.4,
                    ),
                ]
            )
        else:
            # Default to access-based
            base_strategy = AccessBasedStrategy(
                max_age_without_access=self.config.max_age_without_access,
                compression_age=self.config.compression_age,
            )

        # Wrap with TemporalAwareStrategy if SCN is connected
        if self._scn is not None:
            strategy = TemporalAwareStrategy(
                scn=self._scn,
                base_strategy=base_strategy,
            )
            # Pre-compute bin populations for efficient lookups
            strategy.prepare()
            return strategy

        return base_strategy

    def touch(self, memory_id: str) -> None:
        """Update the accessed_at timestamp for a memory.

        Called automatically when a memory is retrieved via get() or recall().
        Can also be called manually to mark a memory as "used".
        """
        with self._rwlock.write():
            self._touch_internal(memory_id)

    def _touch_internal(self, memory_id: str) -> None:
        """Internal touch (lock must be held)."""
        if memory_id in self._memories:
            record = self._memories[memory_id]
            record.accessed_at = time.time()
            record.access_count += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal Clustering (SCN-integrated consolidation)
    # ─────────────────────────────────────────────────────────────────────────

    def sleep_with_clustering(
        self,
        max_per_cluster: int = 3,
    ) -> dict[str, int]:
        """Perform temporal cluster-based sleep consolidation.

        This is the most efficient consolidation method when SCN is connected.
        Instead of evaluating each memory individually, it groups memories by
        temporal cluster (hour, day) and keeps only the best representatives.

        This can reduce 10K memories to ~500 while maintaining full temporal
        coverage.

        Args:
            max_per_cluster: Maximum memories to keep per (hour, day) cluster.
                Default is 3 (best, most accessed, most recent).

        Returns:
            Dict with counts including "clusters_processed".

        Raises:
            RuntimeError: If SCN is not connected.
        """
        if self._scn is None:
            raise RuntimeError("SCN must be connected for cluster-based consolidation")

        with self._rwlock.write():
            return self._sleep_with_clustering(max_per_cluster)

    def _sleep_with_clustering(
        self,
        max_per_cluster: int = 3,
    ) -> dict[str, int]:
        """Internal clustering implementation (lock must be held)."""
        from maxim.memory.strategies import TemporalAwareStrategy

        if self._scn is None:
            return self._sleep()

        now = time.time()
        results = {
            "compressed": 0,
            "removed": 0,
            "preserved": 0,
            "promoted": 0,
            "clusters_processed": 0,
        }

        # Step 1: Consolidate important memories first
        if self.config.consolidate_during_sleep:
            consolidation_results = self._consolidate()
            results["promoted"] = consolidation_results.get("promoted", 0)

        # Step 2: Get all temporal clusters from SCN
        clusters = self._scn.get_all_clusters()
        results["clusters_processed"] = len(clusters)

        # Step 3: Build strategy for selecting representatives
        base_strategy = self._get_memory_strategy()
        if isinstance(base_strategy, TemporalAwareStrategy):
            strategy = base_strategy
        else:
            strategy = TemporalAwareStrategy(
                scn=self._scn,
                base_strategy=base_strategy,
            )
            strategy.prepare()

        memories_to_remove: list[str] = []
        memories_to_compress: list[str] = []

        # Step 4: Process each cluster
        for (hour, day), cluster_ids in clusters.items():
            if len(cluster_ids) <= max_per_cluster:
                # Small cluster - keep all, evaluate individually
                for memory_id in cluster_ids:
                    record = self._memories.get(memory_id)
                    if record is None:
                        continue

                    score = strategy.score_for_retention(record, now, 0)
                    if record.long_term:
                        score *= self.config.long_term_retention_boost
                        score = min(1.0, score)

                    if score < self.config.retention_threshold:
                        memories_to_remove.append(memory_id)
                    elif score < self.config.compression_threshold:
                        if isinstance(record, EpisodicMemory):
                            if strategy.should_compress(record, now, 0):
                                memories_to_compress.append(memory_id)
                            else:
                                results["preserved"] += 1
                        else:
                            results["preserved"] += 1
                    else:
                        results["preserved"] += 1
                continue

            # Large cluster - select best representatives
            # Score all memories in cluster
            scored: list[tuple[str, float]] = []
            for memory_id in cluster_ids:
                record = self._memories.get(memory_id)
                if record is None:
                    continue

                score = strategy.score_for_retention(record, now, 0)
                if record.long_term:
                    score *= self.config.long_term_retention_boost
                    score = min(1.0, score)

                scored.append((memory_id, score))

            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Keep top max_per_cluster, compress/remove the rest
            for i, (memory_id, score) in enumerate(scored):
                record = self._memories.get(memory_id)
                if record is None:
                    continue

                if i < max_per_cluster:
                    # Keep as representative
                    if score < self.config.compression_threshold:
                        if isinstance(record, EpisodicMemory):
                            memories_to_compress.append(memory_id)
                        else:
                            results["preserved"] += 1
                    else:
                        results["preserved"] += 1
                else:
                    # Not a top representative - remove if low score, else compress
                    if score < self.config.retention_threshold:
                        memories_to_remove.append(memory_id)
                    elif isinstance(record, EpisodicMemory):
                        memories_to_compress.append(memory_id)
                    else:
                        # Already compressed, remove
                        memories_to_remove.append(memory_id)

        # Step 5: Handle memories not in any cluster (no temporal signature)
        clustered_ids = set()
        for cluster_ids in clusters.values():
            clustered_ids.update(cluster_ids)

        for memory_id, record in self._memories.items():
            if memory_id in clustered_ids:
                continue

            # No temporal data - use standard evaluation
            score = strategy.score_for_retention(record, now, 0)
            if record.long_term:
                score *= self.config.long_term_retention_boost
                score = min(1.0, score)

            if score < self.config.retention_threshold:
                memories_to_remove.append(memory_id)
            elif score < self.config.compression_threshold:
                if isinstance(record, EpisodicMemory):
                    if strategy.should_compress(record, now, 0):
                        memories_to_compress.append(memory_id)
                    else:
                        results["preserved"] += 1
                else:
                    results["preserved"] += 1
            else:
                results["preserved"] += 1

        # Step 6: Execute compression and removal
        for memory_id in memories_to_compress:
            self._compress_memory(memory_id)
            results["compressed"] += 1

        for memory_id in memories_to_remove:
            self._remove_memory(memory_id)
            results["removed"] += 1

        # Update stats
        self._stats["consolidations"] = self._stats.get("consolidations", 0) + 1
        self._stats["compressions"] = self._stats.get("compressions", 0) + results["compressed"]
        self._stats["removals"] = self._stats.get("removals", 0) + results["removed"]

        # Auto-save if configured
        if self.config.auto_save_after_sleep and self.config.persistence_path:
            self.save_with_backup(self.config.persistence_path)

        logger.info(
            "Cluster consolidation complete: clusters=%d, compressed=%d, removed=%d, preserved=%d",
            results["clusters_processed"],
            results["compressed"],
            results["removed"],
            results["preserved"],
        )

        return results


__all__ = ["Hippocampus", "HippocampusConfig"]
