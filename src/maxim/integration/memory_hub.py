"""MemoryHub - Central coordinator for all memory-subsystem bridges.

The MemoryHub serves as the integration point between the Hippocampus
episodic memory system and external perception/decision/action systems.

Architecture:
                      ┌─────────────────────────────────────────┐
                      │            HIPPOCAMPUS                  │
                      │  ┌───────┐ ┌───────┐ ┌───────┐         │
                      │  │  SCN  │ │  NAc  │ │  EC   │         │
                      │  └───┬───┘ └───┬───┘ └───┬───┘         │
                      └──────┼─────────┼─────────┼──────────────┘
                             │         │         │
     ┌──────────────┬────────┴─────────┼─────────┴────────┬──────────────┐
     ▼              ▼                  ▼                  ▼              ▼
  Attention     Salience          Spatial            Fear           Planning
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from maxim.agents.fear_agent import FearAgent
    from maxim.attention.attention_network import AttentionNetwork
    from maxim.bridges.escalation_bridge import EscalationLearningBridge
    from maxim.bridges.fear_bridge import FearCircuitBridge
    from maxim.bridges.planning_bridge import PlanHistoryBridge
    from maxim.bridges.salience_bridge import SalienceMemoryBridge
    from maxim.bridges.spatial_bridge import SpatialMemoryBridge
    from maxim.decisions.nac import NAc
    from maxim.math.angular_gyrus import AngularGyrus
    from maxim.memory.atl import ATL
    from maxim.memory.cross_layer import CrossLayerGraph
    from maxim.memory.hippocampus import Hippocampus
    from maxim.memory.semantic_promoter import PromotionSource, SemanticPromoter
    from maxim.runtime.worker_pool import WorkerPool
    from maxim.salience.salience_network import SalienceNetwork
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.spatial.spatial_map import SpatialMap
    from maxim.time.scn import SCN

logger = logging.getLogger(__name__)


@dataclass
class MemoryHub:
    """Central hub coordinating all memory-subsystem bridges.

    The MemoryHub manages lifecycle, connectivity, and fault tolerance
    for all bridge modules. It provides a unified interface for:
    - Session lifecycle management
    - Cross-system queries
    - Consolidated sleep/consolidation
    - Health monitoring

    Example:
        # Create core memory systems
        hippocampus = Hippocampus()
        scn = SCN()
        nac = NAc()
        ec = EntorhinalCortex()

        # Create hub
        hub = MemoryHub(
            hippocampus=hippocampus,
            scn=scn,
            nac=nac,
            ec=ec,
        )

        # Connect external systems
        hub.connect(
            spatial=spatial_map,
            attention=attention_network,
            salience=salience_network,
        )

        # Start session (restores priors from memory)
        hub.on_session_start()

        # During operation...
        boosts = hub.get_spatial_boosts("find mug")
        enriched = hub.enrich_salience(detections, goal="find mug")

        # End session (consolidates learning)
        hub.on_session_end()
    """

    # Core memory systems (required)
    hippocampus: "Hippocampus"
    scn: "SCN"
    nac: "NAc"
    ec: "EntorhinalCortex"

    # Bridges (lazy-initialized via connect())
    _spatial_bridge: "SpatialMemoryBridge | None" = None
    _salience_bridge: "SalienceMemoryBridge | None" = None
    _plan_bridge: "PlanHistoryBridge | None" = None
    _escalation_bridge: "EscalationLearningBridge | None" = None
    _fear_bridge: "FearCircuitBridge | None" = None

    # External system references
    _spatial_map: "SpatialMap | None" = None
    _attention: "AttentionNetwork | None" = None
    _fear_agent: "FearAgent | None" = None
    _salience_network: "SalienceNetwork | None" = None

    # Session state
    _session_active: bool = False
    _session_start_time: float = 0.0

    # Disabled bridges (for fault isolation)
    _disabled_bridges: set[str] = field(default_factory=set)

    # Callbacks for memory deletion
    _deletion_callbacks: list[Callable[[str], None]] = field(default_factory=list)

    # Semantic embedding settings (Phase 4)
    embedding_persist_path: str = "data/util/semantic_embeddings.npz"

    # Multi-layer memory (optional)
    atl: "ATL | None" = None
    angular_gyrus: "AngularGyrus | None" = None
    _cross_layer: "CrossLayerGraph | None" = None
    _promoter: "SemanticPromoter | None" = None

    # Concept extraction, grounding, context, and pattern completion (wired in _wire_multi_layer)
    _concept_extractor: Any = None
    _concept_grounder: Any = None
    _concept_context_builder: Any = None
    _pattern_completer: Any = None

    # WorkerPool for background concept memory processing (optional)
    worker_pool: "WorkerPool | None" = None

    # Long-horizon planning (optional)
    _plan_manager: Any = None

    def __post_init__(self) -> None:
        """Initialize and wire core systems."""
        # Connect SCN to Hippocampus for temporal-aware consolidation
        self.hippocampus.connect_scn(self.scn)

        # Register EC for memory deletion cleanup
        self.hippocampus.register_deletion_callback(self.ec.remove_signature)

        # Register NAc for memory deletion cleanup
        self.hippocampus.register_deletion_callback(self.nac.remove_memory)

        # Phase 4: Register for semantic embedding on capture
        if self.ec.semantic_enabled:
            self.hippocampus.register_capture_callback(self._on_memory_captured)
            # Also register embedding store for deletion cleanup
            if self.ec._embedding_store is not None:
                self.hippocampus.register_deletion_callback(
                    self.ec._embedding_store.remove
                )
            logger.info("Semantic embedding enabled")

        # Wire multi-layer memory (ATL, CrossLayerGraph, SemanticPromoter)
        if self.atl is not None:
            self._wire_multi_layer()

        logger.info("MemoryHub initialized with core systems")

    def _wire_multi_layer(self) -> None:
        """Wire ATL, CrossLayerGraph, and SemanticPromoter when ATL is available."""
        from maxim.memory.cross_layer import CrossLayerGraph
        from maxim.memory.semantic_promoter import SemanticPromoter

        # Build layers dict for CrossLayerGraph
        layers: dict[str, Any] = {"hippocampus": self.hippocampus}
        if self.atl is not None:
            layers["atl"] = self.atl
        if self.angular_gyrus is not None:
            layers["angular_gyrus"] = self.angular_gyrus

        # Create CrossLayerGraph if not provided externally
        if self._cross_layer is None:
            self._cross_layer = CrossLayerGraph(layers=layers)

        # Register cross-layer deletion callbacks
        if self._cross_layer is not None:
            self.hippocampus.register_deletion_callback(
                lambda rid: self._cross_layer.remove_record("hippocampus", rid)
            )
            if self.atl is not None:
                self.atl.register_deletion_callback(
                    lambda rid: self._cross_layer.remove_record("atl", rid)
                )

        # Build promotion sources
        sources: list[Any] = [self.nac]  # NAc always available

        # Create SemanticPromoter
        if self.atl is not None:
            self._promoter = SemanticPromoter(
                hippocampus=self.hippocampus,
                atl=self.atl,
                sources=sources,
                cross_layer=self._cross_layer,
            )

        # Wire ConceptExtractor for percept-to-concept extraction
        if self.atl is not None and self._cross_layer is not None:
            from maxim.memory.concept_extractor import ConceptExtractor

            self._concept_extractor = ConceptExtractor(
                atl=self.atl,
                cross_layer=self._cross_layer,
                scn=self.scn,
                worker_pool=self.worker_pool,
            )
            # Register as capture callback on Hippocampus
            self.hippocampus.register_capture_callback(
                self._concept_extractor.on_memory_captured
            )
            # Register for deletion cleanup
            self.hippocampus.register_deletion_callback(
                self._concept_extractor.on_memory_deleted
            )
            # Register for compression cleanup
            self.hippocampus.register_compression_callback(
                self._concept_extractor.on_memory_compressed
            )
            # Rebuild reverse index from persisted ATL state
            self._concept_extractor.rebuild_reverse_index()

        # Wire ConceptGrounder for AG numerical grounding of concepts
        if (
            self.atl is not None
            and self.angular_gyrus is not None
            and self._cross_layer is not None
        ):
            from maxim.math.ips import IPS
            from maxim.memory.concept_grounder import ConceptGrounder

            self._concept_grounder = ConceptGrounder(
                atl=self.atl,
                angular_gyrus=self.angular_gyrus,
                ips=IPS(),
                cross_layer=self._cross_layer,
                worker_pool=self.worker_pool,
            )

        # Wire ConceptContextBuilder for concept-aware recall
        if self.atl is not None:
            from maxim.memory.concept_context import ConceptContextBuilder

            self._concept_context_builder = ConceptContextBuilder(
                atl=self.atl,
                layers=layers,
                concept_grounder=self._concept_grounder,
            )

        # Wire PatternCompleter for graph-chaining pattern completion
        if self.atl is not None:
            from maxim.memory.pattern_completer import PatternCompleter

            self._pattern_completer = PatternCompleter(
                atl=self.atl,
                layers=layers,
            )

        logger.info(
            "Multi-layer memory wired: ATL=%s, AG=%s, cross_layer=%s",
            self.atl is not None,
            self.angular_gyrus is not None,
            self._cross_layer is not None,
        )

    def register_promotion_source(self, source: "PromotionSource") -> None:
        """Register an additional promotion source (e.g., StatisticianAgent)."""
        if self._promoter is not None:
            self._promoter._sources.append(source)

    # ─────────────────────────────────────────────────────────────────────────
    # Bridge Connection
    # ─────────────────────────────────────────────────────────────────────────

    def connect(
        self,
        spatial: "SpatialMap | None" = None,
        attention: "AttentionNetwork | None" = None,
        salience: "SalienceNetwork | None" = None,
        fear_agent: "FearAgent | None" = None,
        novelty_tracker: Any = None,
    ) -> None:
        """Wire up bridges to external systems.

        Call this after creating the hub with external systems you want
        to integrate with memory.

        Args:
            spatial: SpatialMap for spatial memory bridge
            attention: AttentionNetwork (used with spatial)
            salience: SalienceNetwork for salience memory bridge
            fear_agent: FearAgent for fear circuit bridge
            novelty_tracker: NoveltyTracker for sensitization wiring
        """
        from maxim.bridges.escalation_bridge import EscalationLearningBridge
        from maxim.bridges.fear_bridge import FearCircuitBridge
        from maxim.bridges.planning_bridge import PlanHistoryBridge
        from maxim.bridges.salience_bridge import SalienceMemoryBridge
        from maxim.bridges.spatial_bridge import SpatialMemoryBridge

        # Store references
        self._spatial_map = spatial
        self._attention = attention
        self._salience_network = salience
        self._fear_agent = fear_agent

        # Create spatial bridge
        if spatial:
            self._spatial_bridge = SpatialMemoryBridge(
                hippocampus=self.hippocampus,
                spatial_map=spatial,
                ec=self.ec,
                attention=attention,
            )
            logger.info("Connected SpatialMemoryBridge")

        # Create salience bridge
        if salience:
            self._salience_bridge = SalienceMemoryBridge(
                hippocampus=self.hippocampus,
                ec=self.ec,
                salience_network=salience,
            )
            logger.info("Connected SalienceMemoryBridge")

        # Always create planning bridge (uses NAc)
        self._plan_bridge = PlanHistoryBridge(
            hippocampus=self.hippocampus,
            nac=self.nac,
            ec=self.ec,
        )
        logger.info("Connected PlanHistoryBridge")

        # Always create escalation bridge (uses SCN + NAc)
        self._escalation_bridge = EscalationLearningBridge(
            hippocampus=self.hippocampus,
            scn=self.scn,
            nac=self.nac,
        )
        logger.info("Connected EscalationLearningBridge")

        # Always create fear circuit bridge (uses NAc for risk learning)
        self._fear_bridge = FearCircuitBridge(
            hippocampus=self.hippocampus,
            nac=self.nac,
            ec=self.ec,
        )
        logger.info("Connected FearCircuitBridge")

        # Wire sensitization modulation if both salience bridge and tracker available
        if self._salience_bridge and novelty_tracker is not None:
            self._wire_sensitization(novelty_tracker)

    def _wire_sensitization(self, novelty_tracker: Any) -> None:
        """Wire SalienceMemoryBridge interaction history to NoveltyTracker sensitization.

        Creates a callback that computes extremity-based modulation:
        - Classes with strongly positive OR strongly negative interaction history get
          sensitized (modulation > 1.0), resisting habituation.
        - Classes with no history or neutral outcomes get modulation = 1.0 (no effect).

        This mirrors VTA dopaminergic modulation: significant outcomes (both reward
        and aversion) slow sensory cortex habituation for those stimulus categories.
        """
        from maxim.tools.reachy import COCO_CLASSES

        bridge = self._salience_bridge
        min_interactions = 5
        sensitization_scale = 0.5

        def modulation_lookup(class_id: int) -> float:
            class_name = COCO_CLASSES.get(class_id)
            if class_name is None:
                return 1.0
            record = bridge.get_interaction_history(class_name)
            if record is None or record.total_interactions == 0:
                return 1.0
            success_rate = record.success_count / record.total_interactions
            extremity = abs(success_rate - 0.5) * 2  # 0..1
            confidence = min(1.0, record.total_interactions / min_interactions)
            return 1.0 + extremity * confidence * sensitization_scale

        if hasattr(novelty_tracker, "set_modulation_lookup"):
            novelty_tracker.set_modulation_lookup(modulation_lookup)
            logger.info("Wired sensitization modulation to NoveltyTracker")

    # ─────────────────────────────────────────────────────────────────────────
    # Session Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def on_session_start(self) -> dict[str, int]:
        """Initialize all bridges for a new session.

        Restores priors and learned patterns from Hippocampus.

        Returns:
            Dict with counts of restored items per bridge
        """
        self._session_active = True
        self._session_start_time = time.time()

        results = {}

        # Phase 4: Load semantic embeddings
        if self.ec.semantic_enabled and self.ec._embedding_store is not None:
            try:
                import os
                if os.path.exists(self.embedding_persist_path):
                    loaded = self.ec._embedding_store.load(self.embedding_persist_path)
                    results["semantic_embeddings_loaded"] = loaded
            except Exception as e:
                logger.warning("Failed to load semantic embeddings: %s", e)

        # Initialize each bridge
        if self._spatial_bridge and "spatial" not in self._disabled_bridges:
            try:
                results["spatial_priors"] = self._spatial_bridge.on_session_start()
            except Exception as e:
                logger.error("Spatial bridge startup failed: %s", e)
                self._disabled_bridges.add("spatial")

        if self._salience_bridge and "salience" not in self._disabled_bridges:
            try:
                results["salience_history"] = self._salience_bridge.on_session_start()
            except Exception as e:
                logger.error("Salience bridge startup failed: %s", e)
                self._disabled_bridges.add("salience")

        if self._escalation_bridge and "escalation" not in self._disabled_bridges:
            try:
                results["escalation_thresholds"] = (
                    self._escalation_bridge.on_session_start()
                )
            except Exception as e:
                logger.error("Escalation bridge startup failed: %s", e)
                self._disabled_bridges.add("escalation")

        # Load ATL state
        if self.atl is not None:
            try:
                self.atl.load()
                results["atl_concepts"] = len(self.atl)
            except Exception as e:
                logger.warning("Failed to load ATL state: %s", e)

        # Load Angular Gyrus state
        if self.angular_gyrus is not None:
            try:
                self.angular_gyrus.load()
                results["ag_records"] = len(self.angular_gyrus)
            except Exception as e:
                logger.warning("Failed to load Angular Gyrus state: %s", e)

        # Load cross-layer graph
        if self._cross_layer is not None:
            try:
                self._cross_layer.load()
                results["cross_layer_edges"] = self._cross_layer.stats()["total_edges"]
            except Exception as e:
                logger.warning("Failed to load cross-layer graph: %s", e)

        # Long-horizon planning session restore
        if self._plan_manager is not None:
            try:
                plan_result = self._plan_manager.on_session_start()
                if plan_result.get("plan_restored"):
                    results["plan_restored"] = plan_result["plan_restored"]
            except Exception as e:
                logger.warning("PlanManager session start failed: %s", e)

        logger.info("Session started: %s", results)
        return results

    def on_session_end(self) -> dict[str, int]:
        """End session and consolidate learning.

        Runs sleep consolidation on Hippocampus and any bridge-specific
        consolidation.

        Returns:
            Dict with consolidation statistics
        """
        if not self._session_active:
            return {}

        results = {}

        # Run sleep consolidation
        try:
            sleep_results = self.hippocampus.sleep()
            results.update(sleep_results)
        except Exception as e:
            logger.error("Sleep consolidation failed: %s", e)
            results["sleep_error"] = str(e)

        # Session cleanup
        if self._spatial_bridge and "spatial" not in self._disabled_bridges:
            try:
                self._spatial_bridge.on_session_end()
            except Exception as e:
                logger.warning("Spatial bridge cleanup failed: %s", e)

        # Phase 4: Save semantic embeddings
        if self.ec.semantic_enabled and self.ec._embedding_store is not None:
            try:
                import os
                os.makedirs(os.path.dirname(self.embedding_persist_path), exist_ok=True)
                self.ec._embedding_store.save(self.embedding_persist_path)
                results["semantic_embeddings_saved"] = len(self.ec._embedding_store)
            except Exception as e:
                logger.warning("Failed to save semantic embeddings: %s", e)

        # Run semantic promotion (before consolidation — new concepts get consolidated too)
        if self._promoter is not None:
            try:
                promoted = self._promoter.scan_for_promotions()
                results["concepts_promoted"] = len(promoted)
            except Exception as e:
                logger.warning("Semantic promotion failed: %s", e)

        # Consolidate ATL
        if self.atl is not None:
            try:
                atl_results = self.atl.consolidate()
                results["atl_removed"] = atl_results.get("removed", 0)
                results["atl_compressed"] = atl_results.get("compressed", 0)
            except Exception as e:
                logger.warning("ATL consolidation failed: %s", e)

        # Consolidate Angular Gyrus
        if self.angular_gyrus is not None:
            try:
                ag_results = self.angular_gyrus.consolidate()
                results.update({f"ag_{k}": v for k, v in ag_results.items()})
            except Exception as e:
                logger.warning("AG consolidation failed: %s", e)

        # Flush and shutdown ConceptExtractor before saving
        if self._concept_extractor is not None:
            try:
                self._concept_extractor.flush(timeout=5.0)
                self._concept_extractor.shutdown()
            except Exception as e:
                logger.warning("ConceptExtractor shutdown failed: %s", e)

        # Save ATL state
        if self.atl is not None:
            try:
                self.atl.save()
            except Exception as e:
                logger.warning("Failed to save ATL state: %s", e)

        # Save Angular Gyrus state
        if self.angular_gyrus is not None:
            try:
                self.angular_gyrus.save()
            except Exception as e:
                logger.warning("Failed to save AG state: %s", e)

        # Save cross-layer graph
        if self._cross_layer is not None:
            try:
                self._cross_layer.save()
            except Exception as e:
                logger.warning("Failed to save cross-layer graph: %s", e)

        # Long-horizon planning session save
        if self._plan_manager is not None:
            try:
                plan_result = self._plan_manager.on_session_end()
                if plan_result.get("plan_saved"):
                    results["plan_saved"] = plan_result["plan_saved"]
            except Exception as e:
                logger.warning("PlanManager session end failed: %s", e)

        # Flush provenance data and write session summary
        if hasattr(self, "_collector") and self._collector:
            try:
                self._collector.on_session_end()
            except Exception as e:
                logger.warning("Provenance session end failed: %s", e)

        self._session_active = False
        session_duration = time.time() - self._session_start_time
        results["session_duration_seconds"] = session_duration

        logger.info("Session ended after %.1fs: %s", session_duration, results)
        return results

    def sleep(self) -> dict[str, int]:
        """Consolidate learning across all bridges.

        Alias for on_session_end() for API compatibility.
        """
        return self.on_session_end()

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Layer Knowledge Queries
    # ─────────────────────────────────────────────────────────────────────────

    def recall_concepts(
        self, limit: int = 10, **filters: Any
    ) -> list[Any]:
        """Query ATL semantic concepts. Delegates to ATL.recall()."""
        if self.atl is None:
            return []
        return self.atl.recall(limit=limit, **filters)

    def recall_with_knowledge(
        self,
        seed_ids: list[str],
        start_layer: str = "hippocampus",
        limit: int = 10,
    ) -> dict[str, list[tuple[str, float]]]:
        """Cross-layer spreading activation across all three memory layers.

        Returns: {layer_name: [(record_id, activation_score), ...]}
        """
        if self._cross_layer is None:
            return {}
        return self._cross_layer.cross_layer_activation(
            start_layer=start_layer,
            seed_ids=seed_ids,
            max_depth=2,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4: Semantic Embedding
    # ─────────────────────────────────────────────────────────────────────────

    def _on_memory_captured(self, memory_id: str, memory: Any) -> None:
        """Handle memory capture for semantic embedding.

        Called by Hippocampus when a new memory is captured.
        Schedules async embedding if semantic is enabled.
        """
        if not self.ec.semantic_enabled:
            return

        if self.ec._neural_embedder is None or self.ec._embedding_store is None:
            return

        # Extract text for embedding from memory
        text_parts = []
        if hasattr(memory, "decision"):
            if hasattr(memory.decision, "intent"):
                intent = memory.decision.intent
                if isinstance(intent, dict):
                    text_parts.append(str(intent.get("goal", "")))
                else:
                    text_parts.append(str(intent))
            if hasattr(memory.decision, "reasoning"):
                text_parts.append(str(memory.decision.reasoning or ""))

        if hasattr(memory, "context"):
            if hasattr(memory.context, "active_goal"):
                text_parts.append(str(memory.context.active_goal or ""))

        text = " ".join(text_parts).strip()
        if not text:
            return

        # Schedule async embedding
        def embedding_callback(
            mid: str, embedding: Any, hash_bits: tuple[int, ...]
        ) -> None:
            if self.ec._embedding_store is not None:
                self.ec._embedding_store.set(mid, embedding, hash_bits)

        scheduled = self.ec._neural_embedder.schedule_embedding(
            memory_id, text, callback=embedding_callback
        )

        if not scheduled:
            # Fallback to sync embedding if queue full
            try:
                embedding = self.ec._neural_embedder.embed(text)
                hash_bits = self.ec._neural_embedder.hash(text)
                self.ec._embedding_store.set(memory_id, embedding, hash_bits)
            except Exception as e:
                logger.warning("Sync embedding fallback failed: %s", e)

    def find_semantic(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Find memories semantically similar to query text.

        Phase 4 feature: Uses neural embeddings for deep semantic similarity.
        "find mug" will match memories about "cup", "greet" matches "say hello".

        Falls back to structural similarity if semantic not enabled.

        Args:
            query: Natural language query
            k: Maximum results
            threshold: Minimum similarity (0-1)

        Returns:
            List of (memory_id, similarity) tuples
        """
        return self.ec.find_semantic(query, k=k, threshold=threshold)

    @property
    def semantic_enabled(self) -> bool:
        """Check if neural semantic similarity is available."""
        return self.ec.semantic_enabled

    # ─────────────────────────────────────────────────────────────────────────
    # Associative Recall API
    # ─────────────────────────────────────────────────────────────────────────

    def recall_associated(
        self,
        seed_ids: list[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[Any, float]]:
        """Retrieve memories via spreading activation through the associative graph.

        Follows recall-triggered edges formed during memory capture to find
        related memories that may not share direct features but are linked
        through chains of association.

        This is the primary API for context-bridging recall:
        "make coffee" -> recalls cup memory -> which is linked to kitchen memory.

        Args:
            seed_ids: Memory IDs to start activation from.
            limit: Maximum memories to return.
            **kwargs: Passed to hippocampus.recall_associated (decay, max_depth, threshold).

        Returns:
            List of (memory, activation_score) tuples, highest activation first.
        """
        try:
            return self.hippocampus.recall_associated(seed_ids, limit, **kwargs)
        except Exception as e:
            logger.warning("Associative recall failed: %s", e)
            return []

    def recall_with_associations(
        self,
        limit: int = 10,
        association_limit: int = 5,
        *,
        goal: str | None = None,
        tool: str | None = None,
        success: bool | None = None,
        object_detected: str | None = None,
        **recall_kwargs: Any,
    ) -> list[Any]:
        """Recall memories and expand results with associative neighbors.

        First performs a standard recall(), then follows associative edges
        from the results to find additional related memories. This enriches
        recall results with context that wouldn't be found by filter-based
        queries alone.

        Args:
            limit: Maximum direct recall results.
            association_limit: Maximum associated memories to add.
            goal: Goal filter for initial recall.
            tool: Tool filter for initial recall.
            success: Success filter for initial recall.
            object_detected: Object filter for initial recall.
            **recall_kwargs: Additional args for hippocampus.recall.

        Returns:
            Combined list of direct and associated memories (deduplicated).
        """
        try:
            # Step 1: Standard recall
            direct = self.hippocampus.recall(
                limit=limit,
                goal=goal,
                tool=tool,
                success=success,
                object_detected=object_detected,
                **recall_kwargs,
            )

            if not direct:
                return []

            # Step 2: Follow associations from direct results
            seed_ids = [m.id for m in direct]
            associated = self.hippocampus.recall_associated(
                seed_ids, limit=association_limit
            )

            # Step 3: Merge and deduplicate
            seen_ids = set(seed_ids)
            combined = list(direct)
            for mem, _score in associated:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    combined.append(mem)

            return combined

        except Exception as e:
            logger.warning("Recall with associations failed: %s", e)
            # Fall back to standard recall
            return list(
                self.hippocampus.recall(
                    limit=limit,
                    goal=goal,
                    tool=tool,
                    success=success,
                    object_detected=object_detected,
                )
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Spatial Bridge API
    # ─────────────────────────────────────────────────────────────────────────

    def get_spatial_boosts(
        self,
        goal: str,
        boost_factor: float = 1.5,
    ) -> list[tuple[tuple[int, int], float]]:
        """Get spatial attention boosts for a goal.

        Args:
            goal: Current goal text
            boost_factor: Multiplier for attention

        Returns:
            List of ((grid_u, grid_v), boost) tuples
        """
        if "spatial" in self._disabled_bridges:
            return []

        if self._spatial_bridge and self._spatial_bridge.is_healthy:
            try:
                return self._spatial_bridge.boost_attention_for_goal(goal, boost_factor)
            except Exception as e:
                logger.warning("Spatial boost failed: %s", e)
                return []

        return []

    def get_likely_positions(
        self,
        object_class: str,
        top_k: int = 5,
    ) -> list[tuple[int, int, float]]:
        """Get likely positions for an object class.

        Args:
            object_class: Object to find
            top_k: Maximum positions

        Returns:
            List of (grid_u, grid_v, probability) tuples
        """
        if "spatial" in self._disabled_bridges:
            return [(5, 5, 0.1)]

        if self._spatial_bridge and self._spatial_bridge.is_healthy:
            try:
                return self._spatial_bridge.get_likely_positions(object_class, top_k)
            except Exception as e:
                logger.warning("Position lookup failed: %s", e)

        return [(5, 5, 0.1)]

    def record_spatial_success(
        self,
        object_class: str,
        position: tuple[float, float],
        goal: str | None = None,
    ) -> None:
        """Record successful object find at position."""
        if "spatial" in self._disabled_bridges:
            return

        if self._spatial_bridge and self._spatial_bridge.is_healthy:
            try:
                self._spatial_bridge.record_success(object_class, position, goal)
            except Exception as e:
                logger.warning("Spatial recording failed: %s", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Salience Bridge API
    # ─────────────────────────────────────────────────────────────────────────

    def enrich_salience(
        self,
        detections: list[dict[str, Any]],
        goal: str | None = None,
    ) -> list[dict[str, Any]]:
        """Enrich detection salience with interaction history.

        Args:
            detections: Detection dicts with "label" and "salience"
            goal: Current goal for goal-aware boosting

        Returns:
            Enriched detections with updated "salience" values
        """
        if "salience" in self._disabled_bridges:
            return detections

        if self._salience_bridge and self._salience_bridge.is_healthy:
            try:
                return self._salience_bridge.enrich_salience(detections, goal)
            except Exception as e:
                logger.warning("Salience enrichment failed: %s", e)

        return detections

    def record_interaction(
        self,
        object_class: str,
        success: bool,
        goal: str | None = None,
    ) -> None:
        """Record an interaction with an object."""
        if "salience" in self._disabled_bridges:
            return

        if self._salience_bridge and self._salience_bridge.is_healthy:
            try:
                self._salience_bridge.record_interaction(object_class, success, goal)
            except Exception as e:
                logger.warning("Interaction recording failed: %s", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Planning Bridge API
    # ─────────────────────────────────────────────────────────────────────────

    def get_plan_templates(
        self,
        goal: str,
        limit: int = 5,
    ) -> list[Any]:
        """Get successful plan templates for a goal.

        Args:
            goal: Goal to find templates for
            limit: Maximum templates

        Returns:
            List of PlanTemplate objects
        """
        if "planning" in self._disabled_bridges:
            return []

        if self._plan_bridge and self._plan_bridge.is_healthy:
            try:
                return self._plan_bridge.get_plan_templates(goal, limit)
            except Exception as e:
                logger.warning("Plan template lookup failed: %s", e)

        return []

    def get_predicted_success(
        self,
        goal: str,
        tool_sequence: list[str],
    ) -> float:
        """Predict success for a plan.

        Args:
            goal: Goal to achieve
            tool_sequence: Tools in the plan

        Returns:
            Predicted success probability (0-1)
        """
        if "planning" in self._disabled_bridges:
            return 0.5

        if self._plan_bridge and self._plan_bridge.is_healthy:
            try:
                return self._plan_bridge.get_predicted_success(goal, tool_sequence)
            except Exception as e:
                logger.warning("Success prediction failed: %s", e)

        return 0.5

    def record_plan_outcome(
        self,
        goal: str,
        tool_sequence: list[str],
        success: bool,
        memory_id: str | None = None,
    ) -> None:
        """Record outcome of a plan execution."""
        if "planning" in self._disabled_bridges:
            return

        if self._plan_bridge and self._plan_bridge.is_healthy:
            try:
                self._plan_bridge.record_plan_outcome(
                    goal, tool_sequence, success, memory_id=memory_id
                )
            except Exception as e:
                logger.warning("Plan outcome recording failed: %s", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Escalation Bridge API
    # ─────────────────────────────────────────────────────────────────────────

    def get_escalation_threshold(
        self,
        goal: str | None = None,
        novelty: float = 0.5,
        salience: float = 0.5,
    ) -> float:
        """Get learned escalation threshold.

        Args:
            goal: Current goal
            novelty: Current novelty level
            salience: Current salience level

        Returns:
            Threshold value (0-1)
        """
        if "escalation" in self._disabled_bridges:
            return 0.65  # Default

        if self._escalation_bridge and self._escalation_bridge.is_healthy:
            try:
                return self._escalation_bridge.get_threshold(goal, novelty, salience)
            except Exception as e:
                logger.warning("Threshold lookup failed: %s", e)

        return 0.65

    def should_escalate(
        self,
        goal: str | None = None,
        novelty: float = 0.5,
        salience: float = 0.5,
    ) -> tuple[bool, str]:
        """Determine if we should escalate to human.

        Args:
            goal: Current goal
            novelty: Current novelty level
            salience: Current salience level

        Returns:
            (should_escalate, reason) tuple
        """
        if "escalation" in self._disabled_bridges:
            return False, "bridge_disabled"

        if self._escalation_bridge and self._escalation_bridge.is_healthy:
            try:
                return self._escalation_bridge.should_escalate(goal, novelty, salience)
            except Exception as e:
                logger.warning("Escalation check failed: %s", e)
                return True, f"error: {e}"  # Safe default

        return False, "no_bridge"

    def record_escalation_outcome(
        self,
        goal: str | None,
        escalated: bool,
        success: bool,
        novelty: float = 0.5,
        salience: float = 0.5,
    ) -> None:
        """Record the outcome of an escalation decision."""
        if "escalation" in self._disabled_bridges:
            return

        if self._escalation_bridge and self._escalation_bridge.is_healthy:
            try:
                self._escalation_bridge.record_outcome(
                    goal, escalated, success, novelty, salience
                )
            except Exception as e:
                logger.warning("Escalation outcome recording failed: %s", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Fear Bridge API
    # ─────────────────────────────────────────────────────────────────────────

    def get_risk_adjustment(
        self,
        category: str,
        pattern: str,
        context: str = "",
    ) -> float:
        """Get learned risk adjustment for a pattern.

        Returns a value between -0.3 and +0.3:
        - Negative: Lower risk than default (many false positives)
        - Zero: Use default risk assessment
        - Positive: Higher risk than default (many true positives)

        Args:
            category: DangerCategory value
            pattern: Specific pattern (e.g., "subprocess", "eval")
            context: Additional context

        Returns:
            Risk adjustment (-0.3 to +0.3)
        """
        if "fear" in self._disabled_bridges:
            return 0.0

        if self._fear_bridge and self._fear_bridge.is_healthy:
            try:
                return self._fear_bridge.get_risk_adjustment(category, pattern, context)
            except Exception as e:
                logger.warning("Risk adjustment lookup failed: %s", e)

        return 0.0

    def should_block_action(
        self,
        category: str,
        severity: str,
        pattern: str = "",
        context: str = "",
    ) -> tuple[bool, str]:
        """Determine if action should be blocked with memory-informed adjustment.

        Args:
            category: DangerCategory value
            severity: RiskLevel value
            pattern: Specific pattern detected
            context: Additional context

        Returns:
            (should_block, reason) tuple
        """
        if "fear" in self._disabled_bridges:
            # Default to static severity check
            return severity.lower() in ("high", "critical"), "bridge_disabled"

        if self._fear_bridge and self._fear_bridge.is_healthy:
            try:
                return self._fear_bridge.should_block(category, severity, pattern, context)
            except Exception as e:
                logger.warning("Block check failed: %s", e)
                return True, f"error: {e}"

        return severity.lower() in ("high", "critical"), "no_bridge"

    def record_risk_outcome(
        self,
        category: str,
        pattern: str,
        was_blocked: bool,
        actual_harm: bool | None = None,
        severity: str = "medium",
        context: str = "",
    ) -> None:
        """Record the outcome of a risk assessment for learning.

        Args:
            category: DangerCategory value
            pattern: Specific pattern detected
            was_blocked: Whether action was blocked
            actual_harm: Whether actual harm occurred (None if unknown)
            severity: RiskLevel value
            context: Additional context
        """
        if "fear" in self._disabled_bridges:
            return

        if self._fear_bridge and self._fear_bridge.is_healthy:
            try:
                self._fear_bridge.record_outcome(
                    category=category,
                    pattern=pattern,
                    was_blocked=was_blocked,
                    actual_harm=actual_harm,
                    severity=severity,
                    context=context,
                )
            except Exception as e:
                logger.warning("Risk outcome recording failed: %s", e)

    def get_false_positive_rate(self, category: str | None = None) -> float:
        """Get false positive rate for risk assessments.

        Args:
            category: Specific category, or None for overall

        Returns:
            False positive rate (0-1)
        """
        if "fear" in self._disabled_bridges:
            return 0.0

        if self._fear_bridge and self._fear_bridge.is_healthy:
            try:
                return self._fear_bridge.get_false_positive_rate(category)
            except Exception as e:
                logger.warning("FP rate lookup failed: %s", e)

        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Concept Context
    # ─────────────────────────────────────────────────────────────────────────

    def build_concept_context(
        self,
        detected_objects: list[str] | None = None,
        detected_people: list[str] | None = None,
        active_goal: str | None = None,
        limit: int = 5,
        budget_ms: float | None = None,
    ) -> list[dict]:
        """Build concept context entries for the current percept.

        Delegates to ConceptContextBuilder if available. Returns empty list
        if ATL or ConceptContextBuilder is not wired.
        """
        if self._concept_context_builder is None:
            return []
        try:
            return self._concept_context_builder.build(
                detected_objects=detected_objects,
                detected_people=detected_people,
                active_goal=active_goal,
                limit=limit,
                budget_ms=budget_ms,
            )
        except Exception as e:
            logger.warning("Concept context build failed: %s", e)
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Bridge Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def spatial(self) -> "SpatialMemoryBridge | None":
        """Get spatial memory bridge."""
        return self._spatial_bridge

    @property
    def salience(self) -> "SalienceMemoryBridge | None":
        """Get salience memory bridge."""
        return self._salience_bridge

    @property
    def planning(self) -> "PlanHistoryBridge | None":
        """Get plan history bridge."""
        return self._plan_bridge

    @property
    def escalation(self) -> "EscalationLearningBridge | None":
        """Get escalation learning bridge."""
        return self._escalation_bridge

    @property
    def fear(self) -> "FearCircuitBridge | None":
        """Get fear circuit bridge."""
        return self._fear_bridge

    # ─────────────────────────────────────────────────────────────────────────
    # Memory Deletion Handling
    # ─────────────────────────────────────────────────────────────────────────

    def _on_memory_deleted(self, memory_id: str) -> None:
        """Handle memory deletion across all subsystems.

        Called by Hippocampus when a memory is removed during sleep.
        Propagates to all subsystems that might reference this memory.
        """
        # SCN and EC are already registered directly with Hippocampus
        # NAc is also registered

        # Notify any additional callbacks
        for callback in self._deletion_callbacks:
            try:
                callback(memory_id)
            except Exception as e:
                logger.warning("Deletion callback failed: %s", e)

    def register_deletion_callback(
        self,
        callback: Callable[[str], None],
    ) -> None:
        """Register a callback for memory deletion notifications."""
        self._deletion_callbacks.append(callback)

    # ─────────────────────────────────────────────────────────────────────────
    # Health and Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def health_check(self) -> dict[str, bool]:
        """Check health of all bridges.

        Returns:
            Dict mapping bridge name to health status
        """
        return {
            "spatial": (
                self._spatial_bridge.is_healthy
                if self._spatial_bridge
                else False
            ),
            "salience": (
                self._salience_bridge.is_healthy
                if self._salience_bridge
                else False
            ),
            "planning": (
                self._plan_bridge.is_healthy
                if self._plan_bridge
                else False
            ),
            "escalation": (
                self._escalation_bridge.is_healthy
                if self._escalation_bridge
                else False
            ),
            "fear": (
                self._fear_bridge.is_healthy
                if self._fear_bridge
                else False
            ),
            "disabled_bridges": list(self._disabled_bridges),
        }

    def stats(self) -> dict[str, Any]:
        """Return comprehensive statistics from all subsystems."""
        stats = {
            "session_active": self._session_active,
            "disabled_bridges": list(self._disabled_bridges),
            "hippocampus": self.hippocampus.stats(),
            "scn": self.scn.stats(),
            "nac": self.nac.stats(),
            "ec": self.ec.stats(),
        }

        if self._spatial_bridge:
            stats["spatial_bridge"] = self._spatial_bridge.stats()

        if self._salience_bridge:
            stats["salience_bridge"] = self._salience_bridge.stats()

        if self._plan_bridge:
            stats["plan_bridge"] = self._plan_bridge.stats()

        if self._escalation_bridge:
            stats["escalation_bridge"] = self._escalation_bridge.stats()

        if self._fear_bridge:
            stats["fear_bridge"] = self._fear_bridge.stats()

        return stats

    def enable_bridge(self, bridge_name: str) -> bool:
        """Re-enable a disabled bridge.

        Args:
            bridge_name: "spatial", "salience", "planning", "escalation", or "fear"

        Returns:
            True if bridge was enabled, False if not found
        """
        if bridge_name in self._disabled_bridges:
            self._disabled_bridges.remove(bridge_name)
            logger.info("Re-enabled bridge: %s", bridge_name)
            return True
        return False


__all__ = ["MemoryHub"]