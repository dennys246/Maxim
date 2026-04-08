"""Hippocampus retrieval mixin — recall, similarity, and association methods."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from maxim.memory.types import CompressedMemory, EpisodicMemory

if TYPE_CHECKING:
    from maxim.memory.types import Perception

logger = logging.getLogger(__name__)


def _rank_by_relevance(
    memories: list[EpisodicMemory | CompressedMemory],
    query: str,
    limit: int,
) -> list[EpisodicMemory | CompressedMemory]:
    """Rank memories by keyword overlap with a free-text query.

    Scores each memory by how many query tokens appear in its goal,
    tool name, detected objects/people, and observation text.
    Results are sorted by score descending, with recency as tiebreaker.
    """
    query_tokens = set(query.lower().split())
    if not query_tokens:
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        return memories[:limit]

    scored: list[tuple[float, float, EpisodicMemory | CompressedMemory]] = []
    for mem in memories:
        tokens: set[str] = set()

        # Extract searchable text from memory
        if isinstance(mem, CompressedMemory):
            tokens.update((mem.goal or "").lower().split())
            tokens.add((mem.tool_name or "").lower())
        else:
            # Full EpisodicMemory
            if mem.context and mem.context.active_goal:
                tokens.update(mem.context.active_goal.lower().split())
            if mem.action:
                tokens.add((mem.action.tool_name or "").lower())
            if mem.perception:
                for obj in (mem.perception.detected_objects or []):
                    tokens.update(obj.lower().split())
                for person in (mem.perception.detected_people or []):
                    tokens.update(person.lower().split())
                # Also check observation text
                obs_text = mem.perception.observations.get("text", "")
                if isinstance(obs_text, str):
                    tokens.update(obs_text.lower().split()[:50])  # Cap to avoid huge scans
                # Check decision_rationale
                rationale = getattr(mem.perception, "decision_rationale", "")
                if rationale:
                    tokens.update(rationale.lower().split())

        # Score = fraction of query tokens found
        overlap = query_tokens & tokens
        score = len(overlap) / len(query_tokens) if query_tokens else 0.0
        scored.append((score, mem.timestamp, mem))

    # Sort by score descending, then recency as tiebreaker
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [mem for _, _, mem in scored[:limit]]


class RetrievalMixin:
    """Retrieval methods for Hippocampus.

    Provides recall(), recall_similar(), recall_associated(), and
    association graph formation. Mixed into the Hippocampus class.
    """

    def recall(
        self,
        limit: int = 10,
        *,
        query: str | None = None,
        goal: str | None = None,
        tool: str | None = None,
        success: bool | None = None,
        object_detected: str | None = None,
        person_detected: str | None = None,
        mode: str | None = None,
        time_after: float | None = None,
        time_before: float | None = None,
        include_compressed: bool = True,
    ) -> list[EpisodicMemory | CompressedMemory]:
        """Find memories matching filters (hash lookup + filtering).

        All filters are AND-ed together. Use None to skip a filter.

        When *query* is provided, results are ranked by keyword relevance
        instead of recency.  The query is tokenized and scored against each
        memory's goal text, tool name, detected objects/people, and
        observation text.

        Args:
            limit: Maximum number of memories to return.
            query: Free-text query for relevance ranking (keyword overlap).
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
            List of matching memories, ranked by relevance if *query* is
            provided, or most recent first otherwise.
        """
        # Phase 1: Read lock for scanning (allows concurrent readers)
        with self._rwlock.read():
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
                        early_exit = True
                        break
                else:
                    early_exit = False
            else:
                early_exit = False
                # No filters - all memories are candidates
                candidates = set(self._memories.keys())

            if early_exit:
                results: list[EpisodicMemory | CompressedMemory] = []
            else:
                # Filter by time range and collect results
                results = []
                for memory_id in candidates:  # type: ignore[union-attr]
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

                # Rank by relevance if query provided, else by recency
                if query and results:
                    results = _rank_by_relevance(results, query, limit)
                else:
                    results.sort(key=lambda m: m.timestamp, reverse=True)
                    results = results[:limit]

        # Phase 2: Brief write lock for stats update only
        with self._rwlock.write():
            self._stats["queries"] = self._stats.get("queries", 0) + 1

        # Simulation verbosity
        if results:
            try:
                from maxim.simulation.sim_logger import sim_memory

                sim_memory(f"Recalled {len(results)} memories", query_filters=len(index_filters))
            except Exception:
                pass

        return results

    def recall_similar(
        self,
        perception: Perception,
        limit: int = 5,
        include_compressed: bool = True,
    ) -> list[EpisodicMemory | CompressedMemory]:
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
        # Phase 1: Read lock for scanning (allows concurrent readers)
        with self._rwlock.read():
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
            results: list[EpisodicMemory | CompressedMemory] = []
            for memory_id in scored_ids[:limit]:
                memory = self._memories.get(memory_id)
                if memory is not None:
                    if not include_compressed and isinstance(memory, CompressedMemory):
                        continue
                    results.append(memory)

        # Phase 2: Brief write lock for stats update only
        with self._rwlock.write():
            self._stats["queries"] = self._stats.get("queries", 0) + 1

        return results

    def recall_associated(
        self,
        seed_ids: list[str],
        limit: int = 10,
        *,
        decay: float | None = None,
        max_depth: int | None = None,
        threshold: float | None = None,
        include_compressed: bool = True,
    ) -> list[tuple[EpisodicMemory | CompressedMemory, float]]:
        """Retrieve memories via spreading activation through the associative graph.

        Starting from one or more seed memories, follows associative edges
        formed during capture() to find related memories that may not share
        direct perceptual features but are linked through chains of recall.

        This enables context-bridging recall: a memory about finding a cup
        (linked when the coffee-making memory was formed) becomes reachable
        from a "make coffee" query, even though "cup" and "coffee" share
        no direct index keys.

        Args:
            seed_ids: Memory IDs to start activation from.
            limit: Maximum memories to return.
            decay: Activation decay per hop (default: config value).
            max_depth: Maximum hops from seed (default: config value).
            threshold: Minimum activation to include (default: config value).
            include_compressed: Include CompressedMemory results.

        Returns:
            List of (memory, activation_score) tuples, highest activation first.
            Seed memories are excluded from results.
        """
        decay = decay if decay is not None else self.config.spreading_activation_decay
        max_depth = max_depth if max_depth is not None else self.config.spreading_activation_max_depth
        threshold = threshold if threshold is not None else self.config.spreading_activation_threshold

        with self._rwlock.read():
            # Run spreading activation on the graph
            activations = self._graph.spreading_activation(
                seed_ids,
                initial_activation=1.0,
                decay=decay,
                threshold=threshold,
                max_depth=max_depth,
            )

            # Collect results, excluding seeds
            seed_set = set(seed_ids)
            results: list[tuple[EpisodicMemory | CompressedMemory, float]] = []

            for memory_id, activation in activations.items():
                if memory_id in seed_set:
                    continue

                memory = self._memories.get(memory_id)
                if memory is None:
                    continue

                if not include_compressed and isinstance(memory, CompressedMemory):
                    continue

                results.append((memory, activation))

            # Sort by activation score descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]

    def get_associated_ids(
        self,
        memory_id: str,
    ) -> list[tuple[str, float]]:
        """Get directly associated memory IDs and weights for a memory.

        Returns one-hop neighbors only (no spreading activation).

        Args:
            memory_id: The memory to query.

        Returns:
            List of (neighbor_memory_id, edge_weight) tuples.
        """
        with self._rwlock.read():
            return self._graph.get_associated(memory_id)

    def get_edge_count(self, memory_id: str) -> int:
        """Get the number of associative edges for a memory.

        Args:
            memory_id: The memory to query.

        Returns:
            Number of outgoing association edges.
        """
        with self._rwlock.read():
            return len(self._graph.get_associated(memory_id))

    def _form_associations(
        self,
        memory_id: str,
        memory: EpisodicMemory,
    ) -> None:
        """Form bidirectional edges between a new memory and recalled similar memories.

        Called during capture() while write lock is held. This is the core
        mechanism for building the associative graph: when a new memory is
        stored, we find similar existing memories and link them, mimicking
        how biological hippocampal replay strengthens synaptic connections.

        Args:
            memory_id: ID of the newly captured memory.
            memory: The EpisodicMemory being captured.
        """
        from maxim.agents.bus import EdgeType

        # Recall similar memories (lock already held)
        similar = self._recall_similar_unlocked(
            memory.perception,
            limit=self.config.association_limit,
            exclude_id=memory_id,
        )

        if not similar:
            return

        # Register the new memory as a node in the graph
        self._graph.add_node(memory_id, memory_id)

        edges_formed = 0
        for recalled_mem, score in similar:
            if score < self.config.association_threshold:
                continue

            # Compute edge weight
            weight = self._compute_association_weight(memory, recalled_mem, score)

            # Ensure the recalled memory is a node in the graph
            if self._graph.get_node(recalled_mem.id) is None:
                self._graph.add_node(recalled_mem.id, recalled_mem.id)

            # Form bidirectional edge (new ↔ recalled)
            self._graph.add_bidirectional(memory_id, recalled_mem.id, EdgeType.ASSOCIATES, weight)

            # Touch the recalled memory (strengthens its retention)
            self._touch_internal(recalled_mem.id)

            edges_formed += 1

        if edges_formed > 0:
            self._stats["edges_formed"] = self._stats.get("edges_formed", 0) + edges_formed
            logger.debug(
                "Formed %d associative edges for memory %s",
                edges_formed,
                memory_id[:8],
            )

    def _recall_similar_unlocked(
        self,
        perception: Perception,
        limit: int = 5,
        exclude_id: str | None = None,
    ) -> list[tuple[EpisodicMemory | CompressedMemory, float]]:
        """Find similar memories with scores (lock must already be held).

        Returns (memory, score) tuples sorted by score descending.
        Used internally by capture() to form associative edges without
        re-acquiring the write lock.

        Args:
            perception: Current perception to match against.
            limit: Maximum memories to return.
            exclude_id: Memory ID to exclude (the one being captured).

        Returns:
            List of (memory, similarity_score) tuples.
        """
        scores: dict[str, float] = defaultdict(float)

        # Score by overlapping detected objects
        for obj in perception.detected_objects:
            index_key = f"object:{obj}"
            for memory_id in self._context_index.get(index_key, set()):
                if memory_id != exclude_id:
                    scores[memory_id] += 1.0

        # Score by overlapping detected people
        for person in perception.detected_people:
            index_key = f"person:{person}"
            for memory_id in self._context_index.get(index_key, set()):
                if memory_id != exclude_id:
                    scores[memory_id] += 1.0

        if not scores:
            return []

        # Normalize scores to 0-1 range
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            normalized = {mid: s / max_score for mid, s in scores.items()}
        else:
            normalized = scores

        # Sort by score, then by timestamp for tiebreaking
        def sort_key(mid: str) -> tuple[float, float]:
            mem = self._memories.get(mid)
            ts = mem.timestamp if mem else 0.0
            return (normalized[mid], ts)

        sorted_ids = sorted(normalized.keys(), key=sort_key, reverse=True)

        results: list[tuple[EpisodicMemory | CompressedMemory, float]] = []
        for memory_id in sorted_ids[:limit]:
            memory = self._memories.get(memory_id)
            if memory is not None:
                results.append((memory, normalized[memory_id]))

        return results

    def _compute_association_weight(
        self,
        new_memory: EpisodicMemory,
        recalled_memory: EpisodicMemory | CompressedMemory,
        similarity_score: float,
    ) -> float:
        """Compute edge weight for an associative connection.

        Combines perceptual similarity with goal and temporal proximity.

        Args:
            new_memory: The memory being captured.
            recalled_memory: A memory recalled during capture.
            similarity_score: Raw similarity from _recall_similar_unlocked.

        Returns:
            Edge weight (0-1).
        """
        weight = similarity_score * 0.6  # Base: perceptual overlap

        # Boost for shared goal
        new_goal = new_memory.context.active_goal or ""
        if isinstance(recalled_memory, CompressedMemory):
            old_goal = recalled_memory.goal or ""
        else:
            old_goal = recalled_memory.context.active_goal or ""

        if new_goal and old_goal and new_goal == old_goal:
            weight += 0.25
        elif new_goal and old_goal:
            # Partial goal overlap via word intersection
            w1 = set(new_goal.lower().split())
            w2 = set(old_goal.lower().split())
            common = {"the", "a", "an", "to", "and", "or", "in", "on", "at"}
            w1 -= common
            w2 -= common
            if w1 and w2:
                overlap = len(w1 & w2) / len(w1 | w2)
                weight += 0.25 * overlap

        # Temporal proximity bonus (closer in time = stronger link)
        time_diff = abs(new_memory.timestamp - recalled_memory.timestamp)
        temporal_bonus = 0.15 / (1.0 + time_diff / 3600)  # Decays over hours
        weight += temporal_bonus

        return min(1.0, weight)
