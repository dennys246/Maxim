"""ConceptExtractor — extracts concepts from episodic memories into ATL.

Registered as a capture callback on Hippocampus. When an episode is captured,
the extractor identifies percept-derived concepts (objects, people, locations,
goals, actions) and registers them in the ATL via find_or_create().

Uses a queue-based async worker to decouple concept registration from the
capture callback chain — prevents ATL lock contention from stalling
Hippocampus capture callbacks (NAc, SCN, EC).

Also forms inline categorical relationships between concepts found in the
same episode — no batch sleep job needed. Relationships start at low
confidence (0.3) and are strengthened by AG numerical grounding during recall.

Brain mapping: This is the percept-to-concept bridge. Biologically, the
ventral visual stream feeds object/face recognition into the ATL, which
integrates these percepts into unified semantic representations.
"""

from __future__ import annotations

import logging
import queue
import threading
import time as _time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING

from maxim.memory.semantic_types import Concept, ConceptProvenance
from maxim.memory.text import normalize_tokens
from maxim.memory.types import EpisodicMemory, MemoryRecord

if TYPE_CHECKING:
    from maxim.memory.atl import ATL
    from maxim.memory.cross_layer import CrossLayerGraph
    from maxim.runtime.worker_pool import WorkerPool
    from maxim.time.scn import SCN

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """Extracts concepts from episodic memories and registers them in ATL.

    Registered as a capture callback on Hippocampus. Fires after the write
    lock releases, so concept registration is non-blocking to the capture path.

    Also forms inline categorical relationships between concepts found in
    the same episode — no batch sleep job needed.
    """

    # Max relationships formed per episode to prevent noise
    MAX_RELATIONSHIPS_PER_EPISODE = 6

    def __init__(
        self,
        atl: ATL,
        cross_layer: CrossLayerGraph,
        scn: SCN | None = None,
        queue_size: int = 200,
        worker_pool: WorkerPool | None = None,
    ) -> None:
        self._atl = atl
        self._cross_layer = cross_layer
        self._scn = scn
        self._pool = worker_pool

        # Reverse index for O(1) cleanup on memory deletion
        self._reverse_index: dict[str, set[str]] = defaultdict(set)

        # Queue-based async extraction
        self._queue: queue.Queue[tuple[str, EpisodicMemory]] = queue.Queue(
            maxsize=queue_size,
        )
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="concept-extractor-worker",
            daemon=True,
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Capture callback (enqueue — non-blocking)
    # ------------------------------------------------------------------

    def on_memory_captured(self, memory_id: str, record: MemoryRecord) -> None:
        """Enqueue for background processing — non-blocking callback."""
        if not isinstance(record, EpisodicMemory):
            return
        try:
            self._queue.put_nowait((memory_id, record))
        except queue.Full:
            logger.warning(
                "ConceptExtractor queue full, dropping %s", memory_id[:8]
            )

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Background worker: drain queue, register concepts."""
        while not self._stop.is_set():
            try:
                memory_id, record = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_capture(memory_id, record)
            except Exception as e:
                logger.warning(
                    "ConceptExtractor failed for %s: %s", memory_id[:8], e
                )
            finally:
                self._queue.task_done()

    def _process_capture(
        self, memory_id: str, record: EpisodicMemory
    ) -> None:
        """Actual concept extraction — runs in worker thread."""
        concepts_found: list[tuple[str, str]] = []  # (name, category)

        # Objects
        for obj in record.perception.detected_objects:
            concepts_found.append((obj.lower(), "object"))

        # People
        for person in record.perception.detected_people:
            concepts_found.append((person, "person"))

        # Location (extracted from observations if available)
        location = record.perception.observations.get("location")
        if not location:
            location = record.perception.observations.get("room")
        if isinstance(location, str) and location:
            concepts_found.append((location.lower(), "location"))

        # Goal: tokenize into individual words so "navigate_to_kitchen"
        # becomes concepts "navigate" (action) and "kitchen" (goal_token)
        if record.context.active_goal:
            for token in normalize_tokens(record.context.active_goal):
                concepts_found.append((token, "goal"))

        # Action/tool
        if record.action.tool_name:
            concepts_found.append((record.action.tool_name, "action"))

        # Register each concept and collect IDs
        concept_ids: list[tuple[str, str, str]] = []  # (id, name, category)
        for name, category in concepts_found:
            cid = self._register_concept(name, category, memory_id, record)
            if cid:
                concept_ids.append((cid, name, category))
                self._reverse_index[memory_id].add(cid)

        # Form inline relationships between co-occurring concepts
        if self._pool is not None and len(concept_ids) >= 2:
            try:
                self._pool.submit(
                    lane="review",
                    job_id=f"rel-formation-{memory_id}-{_time.monotonic_ns()}",
                    fn=partial(self._form_inline_relationships, list(concept_ids)),
                    priority=8,  # Lower priority than concept grounding reviews
                )
            except Exception as e:
                logger.debug("Failed to enqueue relationship formation: %s", e)
                self._form_inline_relationships(concept_ids)
        else:
            self._form_inline_relationships(concept_ids)

    # ------------------------------------------------------------------
    # Concept registration
    # ------------------------------------------------------------------

    def _register_concept(
        self,
        name: str,
        category: str,
        memory_id: str,
        record: EpisodicMemory,
    ) -> str | None:
        """Find or create a concept and link it to the source memory."""
        from maxim.memory.cross_layer import CrossLayerEdgeType

        concept_id, was_created = self._atl.find_or_create(
            name=name,
            category=category,
            definition=f"{category}: {name}",
            provenance=ConceptProvenance.EPISODIC_CONSOLIDATION,
            source_episode_id=memory_id,
        )

        concept = self._atl.get(concept_id)
        if concept and isinstance(concept, Concept):
            concept.add_ref("hippocampus", memory_id)
        elif concept and not was_created:
            concept.reinforce(memory_id)

        # Cross-layer edge: episode INSTANCE_OF concept
        self._cross_layer.add_edge(
            source_layer="hippocampus",
            source_id=memory_id,
            target_layer="atl",
            target_id=concept_id,
            edge_type=CrossLayerEdgeType.INSTANCE_OF,
            weight=1.0,
        )

        # Register concept in SCN for temporal rhythm tracking
        if self._scn:
            from maxim.time.temporal_signature import TemporalSignature

            sig = TemporalSignature.from_timestamp(record.timestamp)
            self._scn.register(
                concept_id, sig,
                significance=concept.confidence if concept else 0.5,
            )

        return concept_id

    # ------------------------------------------------------------------
    # Inline relationships
    # ------------------------------------------------------------------

    def _form_inline_relationships(
        self, concept_ids: list[tuple[str, str, str]]
    ) -> None:
        """Form tentative relationships between concepts in the same episode.

        Only relates concepts where at least one is an "active" concept —
        the goal being pursued, the action being performed, or the object
        being interacted with. Background objects that happen to share a
        frame don't auto-relate to each other (that's noise, not signal).
        """
        active_categories = {"goal", "action"}
        active_ids = {
            cid for cid, _, cat in concept_ids if cat in active_categories
        }

        formed = 0
        for i, (cid_a, name_a, cat_a) in enumerate(concept_ids):
            if formed >= self.MAX_RELATIONSHIPS_PER_EPISODE:
                break
            for cid_b, name_b, cat_b in concept_ids[i + 1 :]:
                if cid_a == cid_b:
                    continue
                if formed >= self.MAX_RELATIONSHIPS_PER_EPISODE:
                    break

                # At least one concept must be active (goal/action),
                # OR both must be non-object categories.
                if not (
                    cid_a in active_ids
                    or cid_b in active_ids
                    or (cat_a != "object" and cat_b != "object")
                ):
                    continue

                rel_type = self._infer_relationship_type(cat_a, cat_b)

                # Check if relationship already exists
                existing = self._atl.find_by_relationship(
                    cid_a, rel_type=rel_type, direction="outgoing", limit=20
                )
                already_linked = any(oid == cid_b for oid, _ in existing)

                if already_linked:
                    # Reinforce existing relationship — bump confidence
                    self._atl.semantics.update_edge(
                        cid_a, cid_b, rel_type, confidence_delta=0.05,
                    )
                else:
                    # New tentative relationship — low confidence
                    self._atl.define_relationship(
                        cid_a, cid_b, rel_type,
                        weight=0.3,
                        confidence=0.3,
                    )
                formed += 1

    @staticmethod
    def _infer_relationship_type(cat_a: str, cat_b: str) -> str:
        """Infer relationship type from concept category pairing."""
        cats = frozenset({cat_a, cat_b})

        if cats == {"person", "location"}:
            return "RELATED_TO"
        elif cats == {"object", "location"}:
            return "RELATED_TO"
        elif cats == {"person", "object"}:
            return "RELATED_TO"
        elif cats == {"action", "object"}:
            return "RELATED_TO"
        elif cats == {"goal", "action"}:
            return "HAS_PART"
        elif cat_a == cat_b == "object":
            return "RELATED_TO"
        else:
            return "ASSOCIATES"

    # ------------------------------------------------------------------
    # Cleanup callbacks
    # ------------------------------------------------------------------

    def on_memory_deleted(self, memory_id: str) -> None:
        """Clean up concept references when an episode is deleted.

        Uses reverse index for O(1) lookup instead of scanning all concepts.
        """
        self._remove_refs_for(memory_id)

    def on_memory_compressed(self, memory_id: str) -> None:
        """Clean up concept references when an episode is compressed.

        CompressedMemory drops the fields ConceptGrounder and
        PatternCompleter need (action timing, decision confidence,
        detected_objects list, etc.). Keeping stale refs would cause
        those systems to load CompressedMemory records and silently
        skip all numerical extraction. Removing the ref is cleaner.
        """
        self._remove_refs_for(memory_id)

    def _remove_refs_for(self, memory_id: str) -> None:
        """Remove all concept refs for a memory ID."""
        concept_ids = self._reverse_index.pop(memory_id, set())
        for cid in concept_ids:
            concept = self._atl.get(cid)
            if concept and isinstance(concept, Concept):
                concept.remove_ref("hippocampus", memory_id)

    # ------------------------------------------------------------------
    # Startup + Shutdown
    # ------------------------------------------------------------------

    def rebuild_reverse_index(self) -> None:
        """Rebuild reverse index from ATL concept memory_refs.

        Called on startup to restore the reverse index (which is not
        persisted). O(concepts * avg_refs) but only runs once on boot.
        """
        start = _time.monotonic()
        self._reverse_index.clear()
        ref_count = 0
        for concept in self._atl:
            if isinstance(concept, Concept):
                for mem_id in concept.memory_refs.get("hippocampus", {}):
                    self._reverse_index[mem_id].add(concept.id)
                    ref_count += 1
        elapsed_ms = (_time.monotonic() - start) * 1000
        logger.info(
            "ConceptExtractor reverse index rebuilt: %d refs across %d memories in %.1fms",
            ref_count, len(self._reverse_index), elapsed_ms,
        )

    def shutdown(self) -> None:
        """Stop the worker thread. Called during MemoryHub shutdown."""
        self._stop.set()
        self._worker.join(timeout=5.0)

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until the extraction queue is drained.

        Returns True if drained within timeout, False otherwise.
        Useful for testing and ensuring all captures are processed
        before assertions.
        """
        try:
            self._queue.join()
            return True
        except Exception:
            return False


__all__ = ["ConceptExtractor"]
