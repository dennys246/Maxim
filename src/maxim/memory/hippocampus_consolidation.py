"""Hippocampus consolidation mixin — sleep, compression, and promotion."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.memory.strategies import MemoryStrategy
    from maxim.memory.types import CompressedMemory, EpisodicMemory

logger = logging.getLogger(__name__)


class ConsolidationMixin:
    """Consolidation methods for Hippocampus.

    Provides sleep(), consolidate(), memory compression, and
    long-term promotion. Mixed into the Hippocampus class.
    """

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

        with self._rwlock.write():
            results = self._sleep(strategy)

        # Log consolidation activity OUTSIDE rwlock (P3f — Tier 2)
        if hasattr(self, "_collector") and self._collector and \
           self._collector.verbosity >= 1:
            from maxim.provenance.types import PipelineStage
            self._collector.log_activity(
                PipelineStage.CONSOLIDATION, "hippocampus",
                f"Promoted {results['promoted']}, compressed {results['compressed']}, "
                f"removed {results['removed']}",
            )

        return results

    def _sleep(
        self,
        strategy: "MemoryStrategy | None" = None,
    ) -> dict[str, int]:
        """Internal sleep implementation (lock must be held)."""
        from maxim.memory.types import CompressedMemory

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
                # Still evaluate for removal (use stored edge_count)
                score = strategy.score_for_retention(record, now, record.edge_count)
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
            edge_count = len(self._graph.get_associated(memory_id))
            score = strategy.score_for_retention(record, now, edge_count)
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
        from maxim.memory.types import CompressedMemory, EpisodicMemory

        if memory_id not in self._memories:
            return

        record = self._memories[memory_id]
        if not isinstance(record, EpisodicMemory):
            return  # Already compressed or invalid

        # Create compressed version with actual edge count from the graph
        actual_edge_count = len(self._graph.get_associated(memory_id))
        compressed = CompressedMemory.from_episodic(record, edge_count=actual_edge_count)

        # Replace in storage
        self._memories[memory_id] = compressed  # type: ignore[assignment]
        self._compressed_count += 1

        logger.debug("Compressed memory %s", memory_id[:8])

        # Notify subsystems of compression (e.g., ConceptExtractor ref cleanup)
        for callback in self._on_memory_compressed:
            try:
                callback(memory_id)
            except Exception as e:
                logger.warning("Memory compression callback failed: %s", e)

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
        from maxim.memory.types import CompressedMemory

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
        # Deque has maxlen — append always succeeds and oldest is rotated out.
        self._consolidation_candidates.append(memory_id)

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
        from maxim.memory.types import EpisodicMemory

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
