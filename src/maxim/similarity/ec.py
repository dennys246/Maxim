"""Entorhinal Cortex (EC) - Multi-modal similarity engine.

Provides efficient similarity queries across multiple dimensions.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from maxim.models.bio_context import EncodingContext

from maxim.similarity.indices import InvertedIndices
from maxim.similarity.lsh import LSHIndex, SemanticLSH
from maxim.similarity.signature import SituationSignature

# Phase 4: Neural semantic embeddings (optional)
try:
    from maxim.similarity.semantic import (
        NeuralSemanticLSH,
        SemanticEmbedderConfig,
        EmbeddingStore,
    )

    _NEURAL_SEMANTIC_AVAILABLE = True
except ImportError:
    NeuralSemanticLSH = None  # type: ignore
    SemanticEmbedderConfig = None  # type: ignore
    EmbeddingStore = None  # type: ignore
    _NEURAL_SEMANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors.

    Returns 0.0 for zero-norm vectors instead of NaN.

    DIMENSION MISMATCH IS NOT SIMILARITY (2026-08-06). ``zip`` silently
    truncates to the shorter vector, so vectors from DIFFERENT encoder
    spaces returned a plausible-but-wrong score over their shared prefix.
    Reachable within one agent across a LOAD boundary: an ``ec.json``
    written while ``sentence-transformers`` was installed holds 768-dim
    ``LinguisticEncoder`` nodes, but the same encoder silently falls back
    to a 384-dim bag-of-words hash when the ``semantic`` extra is absent
    — so the next session compares 384 against 768 and pattern-completes
    on garbage. Returning 0.0 puts the pair below every threshold, so it
    pattern-SEPARATES (a new node) instead of completing onto an
    incomparable one. Mirrors the same guard in ``hivemind/merge.py``.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────────────────────
# Roy-4 EC-activation instrumentation (Stage 0d of release_0_9_1.md)
#
# Per-tick `sim_ec_activation` JSONL events from every
# `pattern_complete_or_separate` call. Gated by
# `MAXIM_EC_TRACE_ACTIVATIONS=1`. Used by the post-hoc co-activation
# analyzer (scripts/analyze_roy_4_coactivation.py) to validate the
# proposed Hebbian binding rule of cross_modal_substrate_binding.md
# BEFORE the 1.1 implementation lands.
#
# Emission is intentionally opt-in — pair the env var with the autouse
# scrub fixture `_isolate_maxim_ec_trace_env` in tests/conftest.py per
# CLAUDE.md "opt-in env vars in hot startup paths need autouse scrubs".
# ─────────────────────────────────────────────────────────────────────────────

# Map the EC modality string to the Roy-4 modality_tag category
# (sensor / linguistic / drive). The plan spec names these three
# categories explicitly; anything else falls into ``sensor`` as the
# catch-all for non-linguistic substrate inputs.
_EC_TRACE_MODALITY_TAG_MAP: dict[str, str] = {
    "text": "linguistic",
    "interoception": "drive",
    "vision": "sensor",
}


def _ec_trace_enabled() -> bool:
    """Read ``MAXIM_EC_TRACE_ACTIVATIONS`` each call.

    Cheap (a single ``os.environ.get`` lookup). Read-per-call (not
    cached at module load) so Roy-4 runner environments that set the
    var before invoking ``maxim roy run`` pick it up without process
    restart, and so the conftest scrub fixture can deterministically
    enable/disable per-test.
    """
    raw = os.environ.get("MAXIM_EC_TRACE_ACTIVATIONS")
    if raw is None:
        return False
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def _emit_ec_activation(
    *,
    node_id: str,
    similarity: float,
    is_new: bool,
    modality: str,
) -> None:
    """Emit a ``sim_ec_activation`` event when EC instrumentation is on.

    Bound exactly to the two return paths in
    ``EntorhinalCortex.pattern_complete_or_separate``. Caller passes:

    - ``node_id``: the active EC node ID for this call (existing on
      pattern completion, freshly allocated on pattern separation).
    - ``similarity``: cosine score against the matched node; 0.0 on
      separation paths.
    - ``is_new``: True on separation (the node is allocated but not yet
      registered — the encoder registers it via
      ``register_substrate_node`` after this call returns).
    - ``modality``: the EC modality string passed in by the caller.

    The function is a no-op when ``MAXIM_EC_TRACE_ACTIVATIONS`` is unset
    OR ``sim_log`` is not active. It MUST emit events even on cold-start
    when ``active_node_id`` is freshly allocated — the analyzer needs
    pattern-separation events to compute co-activation when both members
    of a pair are new nodes.
    """
    if not _ec_trace_enabled():
        return
    try:
        # Lazy import — keeps EC importable in environments where
        # sim_logger / its transitive deps are not loaded (raw library
        # use). The Roy runner always enables sim logging before calling
        # into the agent loop so this import is satisfied in practice.
        from maxim.simulation import sim_logger as _sl

        if not getattr(_sl, "_sim_active", False):
            return
        import time as _time

        elapsed_s = _time.time() - _sl._sim_start
        # ``tick`` is a coarse 1-second integer bucket — sufficient for
        # the analyzer's "did these two nodes co-fire in the same tick
        # window" question. The continuous ``elapsed_s`` is preserved by
        # sim_log itself as the top-level ``t`` field so the analyzer
        # can rebucket at finer or coarser resolution if needed.
        tick = int(elapsed_s)
        activation_strength = 1.0 if is_new else float(similarity)
        agent_id = _sl._current_agent_id.get(None)
        modality_tag = _EC_TRACE_MODALITY_TAG_MAP.get(modality, "sensor")
        _sl.sim_log(
            "EC_TRACE",
            f"node={node_id[:8]} mod={modality} sim={similarity:.3f}{' NEW' if is_new else ''}",
            {
                "tick": tick,
                "active_node_id": node_id,
                "activation_strength": activation_strength,
                "modality_tag": modality_tag,
                "modality": modality,
                "is_new": is_new,
            },
            agent_id=agent_id,
        )
    except Exception:
        # Instrumentation must never crash the substrate path.
        logger.debug("EC trace emission raised", exc_info=True)


@dataclass
class ECConfig:
    """Configuration for Entorhinal Cortex."""

    # LSH settings
    num_lsh_tables: int = 4
    bits_per_table: int = 8

    # Query settings
    default_k: int = 10
    min_similarity: float = 0.3

    # Phase 4: Neural semantic embeddings
    enable_semantic: bool = False  # Enable neural semantic similarity
    semantic_model: str = "all-MiniLM-L6-v2"  # SentenceTransformer model
    async_embedding: bool = True  # Embed in background thread
    require_gpu: bool = False  # Require GPU for embeddings

    # Phase 4: Semantic embedding hash bits
    semantic_hash_bits: int = 16  # Number of bits for semantic LSH hash

    # Substrate (P1): pattern completion threshold for cosine similarity.
    # Originally tuned at 0.40 via P1 sweep — paraphrase-mpnet-base-v2 @ 0.40
    # → 91.7% collapse / 3.1% cross-cluster on tests/substrate/paraphrase_clusters.yaml.
    # Refined to 0.44 by docs/experiments/26_ec_drift_phase_2_regression.md after
    # the paraphrase-collapse diagnostic (docs/experiments/24+25_*.md) surfaced
    # sequential text-modality centroid drift at 0.40 — successive low-but-above-
    # threshold matches (cosine 0.42-0.45) pulled the running-mean centroid toward
    # a generic "second-person body sensation" prototype that admitted everything,
    # collapsing 19 of 20 unique strings into one mega-node on the Roy fixture.
    # The 0.01 fine sweep (scripts/fine_sweep_phase_2.py) named 0.44 as the sweet
    # spot: P1 collapse 92.3% (improved from 91.7%, only threshold with 10-of-10
    # seeds passing the strict P1 gate, tightest variance), Roy at the ceiling
    # (100% pair / 0% distractor / 6 distinct EC nodes). NAc's get_threshold_overrides
    # has a coupled hardcoded copy at src/maxim/decisions/nac.py — change in lockstep.
    pattern_complete_threshold: float = 0.44

    # Modalities for which pattern_complete_or_separate skips the
    # running-mean centroid update. Frozen-prototype semantics: the
    # first embedding to reach a node fixes its centroid, subsequent
    # matches don't shift it. Required for the "interoception"
    # modality (Phase 0 of grounded_language_acquisition.md) — without
    # it the running-mean centroid tracks smooth drive drift through
    # the trajectory and collapses every snapshot into one cluster
    # (see docs/experiments/13_phase0_harness_smoke.md "smooth drive
    # drift collapses to one cluster"). Declared at the EC config
    # layer, not at the call site, so any future encoder routing
    # through "interoception" automatically inherits the policy.
    # "audio" (exteroceptive sound-localization, perception_pipeline_placement.md
    # Q5): a densely-streamed continuous azimuth/elevation signal would walk a
    # running-mean centroid into the same collapse interoception suffers, so
    # exteroceptive localization nodes are frozen-prototype too — stable
    # per-direction clusters for NAc to attach reward-bias to.
    frozen_centroid_modalities: frozenset[str] = frozenset({"interoception", "audio"})

    # Cross-session persistence (nac_cross_session_persistence.md): path
    # for save()/load(), set by build_bio_stack (agent-home ``ec.json``).
    # The sim path keeps passing explicit paths (aut_ec.json) and ignores
    # this field. NOT serialized into the save payload — load() preserves
    # the live value (see the dataclasses.replace note there).
    persistence_path: str | None = None


@dataclass
class PatternResult:
    """Result of EC pattern_complete_or_separate.

    Attributes:
        node_id: ATL node this percept mapped to (existing or new).
        similarity: Cosine similarity to the matched node (0.0 if new).
        is_new: True if a new node was created (separation).
    """

    node_id: str
    similarity: float
    is_new: bool


class EntorhinalCortex:
    """Entorhinal Cortex - Multi-modal similarity engine.

    Provides efficient multi-modal similarity queries across all memory
    components. Just as the biological EC serves as the gateway between
    the hippocampus and neocortex, this subsystem enables:

    - Fast similarity queries - O(1) approximate nearest neighbor via LSH
    - Multi-modal matching - Combine semantic, structural, temporal signals
    - Composite signatures - Compress features into hashable representations

    KNOWN LIMITATION (Phase 2):
    The semantic_hash dimension requires Phase 4's semantic embedding model.
    Until Phase 4 is implemented, all memories have identical semantic hashes,
    meaning similarity queries rely only on:
    - structural_hash (tool, outcome type)
    - temporal_hash (SCN bins)
    - context_hash (mode, detected objects)

    Impact: Queries like "find mug" won't match memories with "find cup"
    because semantic similarity isn't computed.

    Workaround: Use explicit filters (tool=X, goal=Y) instead of semantic search.

    Example:
        ec = EntorhinalCortex()

        # Register a memory
        signature = SituationSignature.from_memory(memory)
        ec.register(memory.id, signature)

        # Find similar situations
        query_sig = SituationSignature.from_memory(current_situation)
        similar = ec.find_similar(query_sig, k=5)
        for memory_id, score in similar:
            print(f"{memory_id}: {score:.2f}")

        # Query by structural features
        tool_memories = ec.query(tool="internet_search")
    """

    def __init__(self, config: ECConfig | None = None):
        self.config = config or ECConfig()

        # LSH index for approximate nearest neighbor
        self._lsh = LSHIndex(
            num_tables=self.config.num_lsh_tables,
            bits_per_table=self.config.bits_per_table,
        )

        # Inverted indices for structural queries
        self._inverted = InvertedIndices()

        # All signatures
        self._signatures: dict[str, SituationSignature] = {}

        # Optional semantic hasher (Phase 4)
        self._semantic_hasher: SemanticLSH | None = None
        self._neural_embedder: Any | None = None
        self._embedding_store: Any | None = None

        if self.config.enable_semantic:
            # Try neural semantic first (Phase 4), fallback to simple LSH
            if _NEURAL_SEMANTIC_AVAILABLE and NeuralSemanticLSH is not None:
                try:
                    embedder_config = SemanticEmbedderConfig(
                        model_name=self.config.semantic_model,
                        async_embedding=self.config.async_embedding,
                        require_gpu=self.config.require_gpu,
                    )
                    self._neural_embedder = NeuralSemanticLSH(config=embedder_config)
                    self._embedding_store = EmbeddingStore(embedder=self._neural_embedder)
                    logger.info(
                        "EC using neural semantic embeddings: %s",
                        self.config.semantic_model,
                    )
                except Exception as e:
                    logger.warning("Neural semantic init failed, using fallback: %s", e)
                    self._semantic_hasher = SemanticLSH()
            else:
                # Fallback to simple word-based LSH
                self._semantic_hasher = SemanticLSH()

        # Substrate (P1): dense embedding store for pattern completion.
        # Keyed by node_id, stores (centroid_embedding, modality) pairs.
        # Centroid is the running mean of all embeddings that completed to this node.
        self._substrate_nodes: dict[str, tuple[list[float], str]] = {}
        # Member count per node — used for running mean update.
        self._substrate_node_counts: dict[str, int] = {}
        # Hivemind shareability (v1_refinement.md §B5): parallel dicts holding
        # per-node provenance + substrate domain. Stored alongside the
        # (embedding, modality) tuple rather than extending it so all
        # existing tuple-unpacking call sites stay stable. ``source`` is
        # ``"local"`` for nodes learned on this Maxim and an opaque
        # contributor ID for nodes merged in from a substrate bundle.
        # ``domain`` is ``None`` for undomained / generic nodes and a tag
        # string (``"combat"``, ``"cooking"``, ...) for domain-scoped nodes.
        self._substrate_node_sources: dict[str, str] = {}
        self._substrate_node_domains: dict[str, str | None] = {}
        # Artifact stamping (1.1 item 7, pulled forward from the fabric
        # plan's Stage 4): encoders RECORD their realized state here at
        # ENCODE time — the only moment the truth is knowable (a 384-dim
        # embedding could be the bag-of-words fallback OR a real 384-dim
        # model; post-hoc inspection of the arrays cannot distinguish
        # them). Persisted in save()/load() and carried into substrate
        # bundles so a calibration difference (fallback vs real encoder,
        # range-aware vs range-blind sensor normalization) is visible in
        # every circulating artifact instead of silently baking in.
        # Keyed by recorder ("linguistic", "sensor:<modality>").
        self._encoder_provenance: dict[str, dict[str, Any]] = {}

    def record_encoder_provenance(self, key: str, info: dict[str, Any]) -> None:
        """Merge an encoder's realized-state stamp under ``key``.

        Merge rules: ``sensor_names`` accumulates as a sorted union
        (bodies can grow sensors mid-session); ``normalization`` values
        accumulate into ``normalization_modes`` (a session that mixed
        range-aware and range-blind calls must say so — "mixed" is a
        finding, not an error); every other field is last-write-wins.
        Values must be JSON-serializable (they ride ``save()``).
        """
        entry = self._encoder_provenance.setdefault(key, {})
        for k, v in info.items():
            if k == "sensor_names":
                prev = set(entry.get("sensor_names", []))
                entry["sensor_names"] = sorted(prev | set(v))
            elif k == "normalization":
                modes = set(entry.get("normalization_modes", []))
                modes.add(v)
                entry["normalization_modes"] = sorted(modes)
            else:
                entry[k] = v

    @property
    def encoder_provenance(self) -> dict[str, dict[str, Any]]:
        """Read-only view of the recorded encoder stamps (for bundle export)."""
        return {k: dict(v) for k, v in self._encoder_provenance.items()}

    # ─────────────────────────────────────────────────────────────────────────
    # Substrate Pattern Completion (P1)
    # ─────────────────────────────────────────────────────────────────────────

    def pattern_complete_or_separate(
        self,
        embedding: list[float],
        modality: str,
        threshold: float | None = None,
        threshold_override: dict[str, float] | None = None,
        encoding_context: "EncodingContext | None" = None,
    ) -> PatternResult:
        """Route an embedding to an existing ATL node or create a new one.

        Cosine similarity against all stored embeddings of the same
        modality. If the best match exceeds ``threshold``, returns that
        node (pattern completion). Otherwise creates a new node
        (pattern separation).

        For modalities listed in ``config.frozen_centroid_modalities``
        (e.g. ``"interoception"``) the matched node's stored embedding
        is left untouched on completion — the first embedding to reach
        a node is the prototype. For all other modalities the centroid
        is updated as a running mean.

        Args:
            embedding: Dense vector from LinguisticEncoder.
            modality: Substrate modality ("text" or "vision").
            threshold: Override the default pattern_complete_threshold.
            threshold_override: Per-node threshold overrides keyed by
                node_id. Used by P2 reward modulation to widen the
                recognition radius for rewarded nodes.

        Returns:
            PatternResult with the node_id, similarity score, and
            whether a new node was created.
        """
        base_threshold = threshold if threshold is not None else self.config.pattern_complete_threshold
        overrides = threshold_override or {}

        best_node: str | None = None
        best_sim = -1.0

        for node_id, (stored_emb, stored_mod) in self._substrate_nodes.items():
            if stored_mod != modality:
                continue
            sim = _cosine_similarity(embedding, stored_emb)
            # Use per-node override if available, else base threshold
            node_thresh = overrides.get(node_id, base_threshold)
            if sim >= node_thresh and sim > best_sim:
                best_sim = sim
                best_node = node_id

        if best_node is not None:
            # Frozen-prototype modalities skip the centroid update —
            # the first embedding to reach a node fixes the prototype.
            # See ECConfig.frozen_centroid_modalities for rationale.
            if modality in self.config.frozen_centroid_modalities:
                self._substrate_node_counts[best_node] = self._substrate_node_counts.get(best_node, 1) + 1
                _emit_ec_activation(
                    node_id=best_node,
                    similarity=best_sim,
                    is_new=False,
                    modality=modality,
                )
                return PatternResult(node_id=best_node, similarity=best_sim, is_new=False)

            # Update centroid: running mean of all embeddings that completed here.
            # new_centroid = (old_centroid * n + new_embedding) / (n + 1)
            stored_emb, stored_mod = self._substrate_nodes[best_node]
            n = self._substrate_node_counts.get(best_node, 1)
            updated = [(s * n + e) / (n + 1) for s, e in zip(stored_emb, embedding)]
            self._substrate_nodes[best_node] = (updated, stored_mod)
            self._substrate_node_counts[best_node] = n + 1
            _emit_ec_activation(
                node_id=best_node,
                similarity=best_sim,
                is_new=False,
                modality=modality,
            )
            return PatternResult(node_id=best_node, similarity=best_sim, is_new=False)

        # Separation — allocate a new node ID but don't register yet.
        # The caller (LinguisticEncoder) registers via register_substrate_node
        # after ATL activation succeeds. This keeps EC stateless for the
        # separation path and allows the test harness to inspect without
        # side effects.
        new_id = str(uuid4())
        _emit_ec_activation(
            node_id=new_id,
            similarity=0.0,
            is_new=True,
            modality=modality,
        )
        return PatternResult(node_id=new_id, similarity=0.0, is_new=True)

    def register_substrate_node(
        self,
        node_id: str,
        embedding: list[float],
        modality: str,
        *,
        source: str = "local",
        domain: str | None = None,
    ) -> None:
        """Register or update a substrate node's embedding.

        ``source`` defaults to ``"local"`` (this Maxim learned it). Pass an
        opaque contributor ID (e.g. ``"oasis-abc123"``, ``"consensus"``)
        when registering a node imported from a substrate bundle. ``domain``
        is the optional Hivemind substrate-domain tag (``"combat"``,
        ``"cooking"``, ...); ``None`` for undomained / generic nodes.
        Both fields are keyword-only and additive — existing production
        callers in ``maxim.similarity.encoder`` continue to work unchanged
        and inherit ``source="local"`` / ``domain=None``.
        """
        self._substrate_nodes[node_id] = (embedding, modality)
        self._substrate_node_counts[node_id] = 1
        self._substrate_node_sources[node_id] = source
        self._substrate_node_domains[node_id] = domain

    def remove_substrate_node(self, node_id: str) -> None:
        """Remove a substrate node."""
        self._substrate_nodes.pop(node_id, None)
        self._substrate_node_counts.pop(node_id, None)
        self._substrate_node_sources.pop(node_id, None)
        self._substrate_node_domains.pop(node_id, None)

    def substrate_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        """Return ``{node_id, embedding, modality, member_count, source, domain}``.

        Used by the Hivemind merge functions (v1_refinement.md §B5 PR B)
        and the bundle composer (PR D) to inspect node provenance and
        domain without depending on the private parallel-dict layout.
        Returns ``None`` when the node is not registered.

        The ``node_id`` field is included so callers iterating a list of
        metadatas don't have to thread the key separately (the common
        bundle-composition pattern). ``member_count`` is the running
        count of embeddings absorbed into this node's centroid — named
        explicitly to distinguish from any future "observation count"
        statistic at the EC layer.
        """
        node = self._substrate_nodes.get(node_id)
        if node is None:
            return None
        emb, mod = node
        return {
            "node_id": node_id,
            "embedding": emb,
            "modality": mod,
            "member_count": self._substrate_node_counts.get(node_id, 1),
            "source": self._substrate_node_sources.get(node_id, "local"),
            "domain": self._substrate_node_domains.get(node_id),
        }

    @property
    def substrate_node_count(self) -> int:
        """Number of substrate nodes registered."""
        return len(self._substrate_nodes)

    def register(
        self,
        memory_id: str,
        signature: SituationSignature | None = None,
        memory: Any = None,
    ) -> SituationSignature:
        """Register a memory with the EC.

        Args:
            memory_id: Unique memory identifier
            signature: Pre-computed signature (optional)
            memory: EpisodicMemory to create signature from (if signature not provided)

        Returns:
            The registered signature
        """
        if signature is None:
            if memory is None:
                raise ValueError("Either signature or memory must be provided")
            signature = SituationSignature.from_memory(memory, semantic_hasher=self._semantic_hasher)

        self._signatures[memory_id] = signature
        self._lsh.add(memory_id, signature)
        self._inverted.add(memory_id, signature)

        return signature

    def unregister(self, memory_id: str) -> None:
        """Remove a memory from the EC.

        Args:
            memory_id: Memory to remove
        """
        signature = self._signatures.pop(memory_id, None)
        if signature:
            self._lsh.remove(memory_id)
            self._inverted.remove(memory_id, signature)

    def remove_signature(self, memory_id: str) -> None:
        """Alias for unregister for deletion callback compatibility."""
        self.unregister(memory_id)

    def get_signature(self, memory_id: str) -> SituationSignature | None:
        """Get the signature for a memory."""
        return self._signatures.get(memory_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Similarity Queries
    # ─────────────────────────────────────────────────────────────────────────

    def find_similar(
        self,
        signature: SituationSignature,
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[tuple[str, float]]:
        """Find k most similar memories.

        Args:
            signature: Query signature
            k: Number of results (default from config)
            min_similarity: Minimum similarity threshold (default from config)

        Returns:
            List of (memory_id, similarity_score) tuples
        """
        k = k or self.config.default_k
        min_sim = min_similarity or self.config.min_similarity

        # Query LSH index
        results = self._lsh.query(signature, k=k * 2, probe_radius=1)

        # Filter by minimum similarity
        filtered = [(mid, score) for mid, score in results if score >= min_sim]

        return filtered[:k]

    def find_similar_by_memory(
        self,
        memory_id: str,
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[tuple[str, float]]:
        """Find memories similar to an existing memory.

        Args:
            memory_id: Reference memory ID
            k: Number of results
            min_similarity: Minimum threshold

        Returns:
            List of (memory_id, similarity_score) tuples (excluding reference)
        """
        signature = self._signatures.get(memory_id)
        if not signature:
            return []

        results = self.find_similar(signature, k=k, min_similarity=min_similarity)

        # Exclude the reference memory
        return [(mid, score) for mid, score in results if mid != memory_id]

    def find_similar_situations_for_action(
        self,
        tool_name: str,
        context: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[tuple[str, SituationSignature]]:
        """Find similar situations where an action was taken.

        Useful for predicting outcomes of potential actions.

        Args:
            tool_name: Tool to find similar uses of
            context: Current context for filtering
            k: Maximum results

        Returns:
            List of (memory_id, signature) tuples
        """
        # Get all memories using this tool
        tool_memories = self._inverted.query_tool(tool_name)

        if not tool_memories:
            return []

        # Score by context similarity if provided
        if context:
            # Build a pseudo-signature for context matching
            mode = context.get("mode", "")
            scored = []
            for mid in tool_memories:
                sig = self._signatures.get(mid)
                if sig:
                    # Simple context match scoring
                    score = 0.5
                    if sig.mode == mode:
                        score += 0.3
                    if sig.outcome_type == "success":
                        score += 0.2
                    scored.append((mid, sig, score))

            scored.sort(key=lambda x: x[2], reverse=True)
            return [(mid, sig) for mid, sig, _ in scored[:k]]

        # No context, return most recent
        results = []
        for mid in list(tool_memories)[:k]:
            sig = self._signatures.get(mid)
            if sig:
                results.append((mid, sig))
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Semantic Search (Phase 4)
    # ─────────────────────────────────────────────────────────────────────────

    def find_semantic(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.5,
        encoding_context: "EncodingContext | None" = None,
    ) -> list[tuple[str, float]]:
        """Find memories semantically similar to query text.

        Phase 4 feature: Uses neural embeddings for deep semantic similarity.
        "find mug" will match memories about "cup", "find greeting" matches
        "say hello".

        Falls back to structural similarity if semantic not enabled.

        Args:
            query: Natural language query
            k: Maximum results
            threshold: Minimum similarity (0-1)

        Returns:
            List of (memory_id, similarity) tuples
        """
        if self._embedding_store is not None:
            # Use neural semantic search
            return self._embedding_store.find_similar(query, k=k, threshold=threshold)

        # Fallback: use LSH-based similarity on existing signatures
        # This provides structural but not semantic similarity
        if self._semantic_hasher:
            query_hash = self._semantic_hasher.hash(query)
            results = []
            for mid, sig in self._signatures.items():
                similarity = self._semantic_hasher.estimated_similarity(query_hash, sig.semantic_hash)
                if similarity >= threshold:
                    results.append((mid, similarity))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

        return []

    @property
    def semantic_enabled(self) -> bool:
        """Check if neural semantic similarity is available."""
        return self._neural_embedder is not None and self._neural_embedder.is_healthy

    # ─────────────────────────────────────────────────────────────────────────
    # Structural Queries
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        tool: str | None = None,
        outcome: str | None = None,
        mode: str | None = None,
        hour: int | None = None,
        day: int | None = None,
    ) -> set[str]:
        """Query by structural features.

        All filters are AND-ed together.

        Args:
            tool: Tool name filter
            outcome: Outcome type filter
            mode: Mode filter
            hour: Hour bin filter (0-23)
            day: Day bin filter (0-6)

        Returns:
            Set of matching memory IDs
        """
        return self._inverted.query_intersection(tool=tool, outcome=outcome, mode=mode, hour=hour, day=day)

    def find_by_temporal(
        self,
        hour_bin: int | None = None,
        day_bin: int | None = None,
    ) -> set[str]:
        """Find memories by temporal context.

        Args:
            hour_bin: Hour of day (0-23)
            day_bin: Day of week (0-6)

        Returns:
            Set of matching memory IDs
        """
        return self._inverted.query_intersection(hour=hour_bin, day=day_bin)

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return EC statistics."""
        stats = {
            "total_signatures": len(self._signatures),
            "lsh_entries": len(self._lsh),
            "inverted_indices": self._inverted.stats(),
            "semantic_enabled": self.config.enable_semantic,
            "neural_semantic_available": self._neural_embedder is not None,
            "substrate_nodes": len(self._substrate_nodes),
        }

        # Add neural embedder stats if available
        if self._neural_embedder is not None:
            stats["neural_embedder"] = self._neural_embedder.stats()

        if self._embedding_store is not None:
            stats["embeddings_stored"] = len(self._embedding_store)

        return stats

    def __len__(self) -> int:
        """Number of registered signatures."""
        return len(self._signatures)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save EC state to JSON file."""
        data = {
            "version": "1.0",
            # Marks that persisted hash-derived values (signature
            # structural/context/semantic hashes, LSH tables) were computed
            # with the process-stable sha256 scheme. Files WITHOUT this key
            # predate the stable-hash fix and their hashes can never match
            # recomputed values — load() warns so the failure mode is
            # visible instead of reading as "recall is just noisy".
            "hash_scheme": "stable-sha256-v1",
            "config": {
                "num_lsh_tables": self.config.num_lsh_tables,
                "bits_per_table": self.config.bits_per_table,
                "default_k": self.config.default_k,
                "min_similarity": self.config.min_similarity,
                "enable_semantic": self.config.enable_semantic,
            },
            "lsh": self._lsh.serialize(),
            "inverted": self._inverted.to_dict(),
            "signatures": {k: v.to_dict() for k, v in self._signatures.items()},
            "substrate_nodes": {
                nid: {
                    "embedding": emb,
                    "modality": mod,
                    "count": self._substrate_node_counts.get(nid, 1),
                    "source": self._substrate_node_sources.get(nid, "local"),
                    "domain": self._substrate_node_domains.get(nid),
                }
                for nid, (emb, mod) in self._substrate_nodes.items()
            },
            # Realized encoder state recorded at encode time (artifact
            # stamping, 1.1 item 7) — see record_encoder_provenance.
            "encoder_provenance": self._encoder_provenance,
        }

        from maxim.utils.atomic_io import atomic_write_json
        from maxim.utils.format_version import with_format_version

        atomic_write_json(path, with_format_version(data))

        logger.info(
            "Saved EC to %s (%d signatures, %d substrate nodes)",
            path,
            len(self._signatures),
            len(self._substrate_nodes),
        )

    def load(self, path: str) -> None:
        """Load EC state from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        from maxim.utils.format_version import check_format_version

        check_format_version(data, "ec", log=logger)

        # Inner payload version is the legacy "version" string; tombstoned
        # for the same reason the BioSystemSnapshot payload versions are
        # tombstoned. The root-level _format_version is the authoritative
        # contract going forward.
        version = data.get("version", "0.0")
        if version != "1.0":
            raise ValueError(f"Unsupported EC version: {version}")

        if "hash_scheme" not in data:
            logger.warning(
                "EC file %s predates stable hashing — its persisted signature "
                "hashes were computed with Python's randomized hash() and "
                "will not match values recomputed in this process. Matching "
                "against these signatures will fail until they are re-learned.",
                path,
            )

        # Load config. dataclasses.replace on the LIVE config, not a
        # fresh ECConfig: only five fields are serialized, and rebuilding
        # from scratch silently reset every runtime-configured field the
        # payload doesn't carry (persistence_path, pattern_complete_threshold,
        # frozen_centroid_modalities, ...) back to defaults.
        from dataclasses import replace as _dc_replace

        cfg_data = data.get("config", {})
        self.config = _dc_replace(
            self.config,
            num_lsh_tables=cfg_data.get("num_lsh_tables", 4),
            bits_per_table=cfg_data.get("bits_per_table", 8),
            default_k=cfg_data.get("default_k", 10),
            min_similarity=cfg_data.get("min_similarity", 0.3),
            enable_semantic=cfg_data.get("enable_semantic", False),
        )

        # Load LSH
        self._lsh = LSHIndex(
            num_tables=self.config.num_lsh_tables,
            bits_per_table=self.config.bits_per_table,
        )
        self._lsh.deserialize(data.get("lsh", {}))

        # Load inverted indices
        self._inverted = InvertedIndices.from_dict(data.get("inverted", {}))

        # Load signatures
        self._signatures = {k: SituationSignature.from_dict(v) for k, v in data.get("signatures", {}).items()}

        # Load substrate nodes (P1). Pre-B5 dumps lack the ``source`` and
        # ``domain`` fields — both default to ``"local"`` and ``None``.
        self._substrate_nodes = {}
        self._substrate_node_counts = {}
        self._substrate_node_sources = {}
        self._substrate_node_domains = {}
        for nid, ndata in data.get("substrate_nodes", {}).items():
            self._substrate_nodes[nid] = (ndata["embedding"], ndata["modality"])
            self._substrate_node_counts[nid] = ndata.get("count", 1)
            self._substrate_node_sources[nid] = ndata.get("source", "local")
            self._substrate_node_domains[nid] = ndata.get("domain")

        # Encoder stamps (artifact stamping, 1.1 item 7). Pre-stamping
        # files lack the key — empty dict, and the bundle will honestly
        # carry recorded=None for them.
        self._encoder_provenance = dict(data.get("encoder_provenance", {}))

        logger.info(
            "Loaded EC from %s (%d signatures, %d substrate nodes)",
            path,
            len(self._signatures),
            len(self._substrate_nodes),
        )

    def get_version(self) -> str:
        """Return data format version."""
        return "1.0"


__all__ = ["EntorhinalCortex", "ECConfig", "PatternResult", "_cosine_similarity"]
