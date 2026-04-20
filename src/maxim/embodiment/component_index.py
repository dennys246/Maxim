"""ComponentIndex — two-layer semantic discovery over ComponentRegistry.

Layer 1: Alias hash table — O(1) lookup from synonyms to component refs.
Layer 2: Semantic embedding index — cosine similarity against component
         signature embeddings.

The index bridges natural language queries to exact component refs:
``"old iron door"`` resolves to ``environments/rusty_gate`` via embedding
similarity without requiring exact naming.

Construction: built from a :class:`ComponentRegistry` at startup.
Incremental: :meth:`add` called on ``register`` / ``register_ephemeral``.
Thread-safe: RLock guards alias table + embedding list mutation.

Embedding persistence: cached to ``~/.maxim/component_index.npz``.
Rebuilt on registry content change (mtime-checked against YAML files).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from maxim.embodiment.component_registry import ComponentRegistry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentMatch:
    """Result of a component discovery lookup."""

    ref: str
    name: str
    score: float  # 1.0 for alias match, cosine similarity for embedding
    layer: str  # "alias" or "embedding"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_semantic_signature(spec: dict) -> str:
    """Build a text signature from a component spec for embedding.

    Composes name, description, sensor names, affordance names, and
    failure mode names into a single text string that captures the
    component's identity for similarity matching.
    """
    entity = spec.get("entity", spec)
    name = entity.get("name", "")
    entity_type = entity.get("entity_type", "")

    # Collect sensor names
    sensors = list(entity.get("sensors", {}).keys())

    # Collect affordance names + descriptions
    affordances = []
    for mod in entity.get("modulators", {}).values():
        for aff_name, aff_spec in mod.get("affordances", {}).items():
            desc = ""
            if isinstance(aff_spec, dict):
                desc = aff_spec.get("description", "")
            affordances.append(f"{aff_name}: {desc}" if desc else aff_name)

    # Collect failure mode names
    failures = []
    for fm in entity.get("failure_modes", []):
        if isinstance(fm, dict):
            failures.append(fm.get("name", ""))

    parts = [f"{name} ({entity_type})" if entity_type else name]
    if sensors:
        parts.append(f"sensors: {', '.join(sensors)}")
    if affordances:
        parts.append(f"affordances: {', '.join(affordances)}")
    if failures:
        parts.append(f"failures: {', '.join(f for f in failures if f)}")

    return ". ".join(parts)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _top_k_matches(
    query_emb: np.ndarray,
    embeddings: np.ndarray,
    refs: list[str],
    k: int,
    threshold: float = -1.0,
) -> list[tuple[str, float]]:
    """Find top-k matches by cosine similarity.

    Shared implementation for find_similar, dedup_check, and
    _find_by_embedding.  Returns (ref, score) pairs sorted descending.
    Only includes results above *threshold*.
    """
    scores = []
    for i, ref in enumerate(refs):
        sim = _cosine_similarity(query_emb, embeddings[i])
        if sim >= threshold:
            scores.append((ref, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


# ---------------------------------------------------------------------------
# ComponentIndex
# ---------------------------------------------------------------------------


class ComponentIndex:
    """Semantic discovery layer over ComponentRegistry.

    Two-layer lookup: alias hash table (fast, exact) + embedding index
    (slower, fuzzy). Constructed from a :class:`ComponentRegistry` at
    startup, updated incrementally when ``register``/``register_ephemeral``
    adds entities.

    Reuses the existing ``similarity.encoder._get_encoder`` singleton
    for the embedding model — no duplicate model in memory.

    Example::

        index = ComponentIndex(registry)
        match = index.find("old iron door")
        # ComponentMatch(ref='environments/rusty_gate', score=0.82, layer='embedding')

        dupes = index.dedup_check(candidate_spec, threshold=0.80)
        # ComponentMatch if near-duplicate exists, None otherwise
    """

    def __init__(
        self,
        registry: ComponentRegistry | None = None,
        *,
        encoder_model: str = "all-mpnet-base-v2",
        similarity_threshold: float = 0.65,
        cache_path: Path | str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._encoder_lock = threading.Lock()

        # Layer 1: alias → ref
        self._aliases: dict[str, str] = {}

        # Layer 2: embedding index
        self._refs: list[str] = []  # parallel to _embeddings rows
        self._signatures: list[str] = []  # text signatures for debugging
        self._embeddings: np.ndarray | None = None  # (N, dim) matrix

        self._encoder_model_name = encoder_model
        self._similarity_threshold = similarity_threshold
        self._encoder: Any = None  # lazy-loaded via _get_encoder singleton
        self._encoder_loaded = False
        self._using_fallback = False

        # Persistence
        if cache_path is not None:
            self._cache_path = Path(cache_path)
        else:
            self._cache_path = None  # resolved lazily from data_home

        # Build from registry if provided
        if registry is not None:
            self._build_from_registry(registry)

    # -- public API ---------------------------------------------------------

    def find(self, query: str) -> ComponentMatch | None:
        """Two-layer lookup: alias -> embedding -> None.

        Layer 1 checks for an exact alias match (O(1)).
        Layer 2 computes cosine similarity against all component embeddings.
        Returns the best match above ``similarity_threshold``, or None.
        """
        if not query or not query.strip():
            return None

        normalized = query.strip().lower()

        # Layer 1: alias table
        with self._lock:
            ref = self._aliases.get(normalized)
        if ref is not None:
            name = ref.rsplit("/", 1)[-1] if "/" in ref else ref
            return ComponentMatch(ref=ref, name=name, score=1.0, layer="alias")

        # Layer 2: embedding similarity
        return self._find_by_embedding(normalized)

    def find_similar(self, query: str, k: int = 5) -> list[ComponentMatch]:
        """Return top-k similar components by embedding cosine similarity.

        Always uses Layer 2 (embedding). Useful for browsing/exploration.
        Returns results sorted by score descending. May return fewer than
        k results if the index is smaller.
        """
        if not query or not query.strip():
            return []

        query_emb = self._embed(query.strip().lower())

        with self._lock:
            if self._embeddings is None or len(self._refs) == 0:
                return []
            embeddings = self._embeddings.copy()
            refs = list(self._refs)

        matches = _top_k_matches(query_emb, embeddings, refs, k)
        return [
            ComponentMatch(
                ref=ref,
                name=ref.rsplit("/", 1)[-1] if "/" in ref else ref,
                score=sim,
                layer="embedding",
            )
            for ref, sim in matches
        ]

    def add(
        self,
        ref: str,
        spec: dict,
        synonyms: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Index a new component (called on register + register_ephemeral).

        Adds aliases from *synonyms* to Layer 1 and computes + stores the
        semantic embedding for Layer 2. The component name itself is also
        added as an alias.

        Embedding computation happens OUTSIDE the lock (~5ms per call).
        Lock is only held for the dict/list mutations.
        """
        # Build semantic signature and embed outside the lock
        signature = _build_semantic_signature(spec)
        embedding = self._embed(signature)

        name = ref.rsplit("/", 1)[-1] if "/" in ref else ref

        with self._lock:
            # Layer 1: add aliases
            self._aliases[name.lower()] = ref
            self._aliases[name.lower().replace("_", " ")] = ref
            if synonyms:
                for syn in synonyms:
                    if syn and syn.strip():
                        self._aliases[syn.strip().lower()] = ref

            # Layer 2: add embedding
            if ref in self._refs:
                idx = self._refs.index(ref)
                self._signatures[idx] = signature
                if self._embeddings is not None:
                    self._embeddings[idx] = embedding
            else:
                self._refs.append(ref)
                self._signatures.append(signature)
                if self._embeddings is None:
                    self._embeddings = embedding.reshape(1, -1)
                else:
                    self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)])

    def dedup_check(
        self,
        spec: dict,
        threshold: float = 0.80,
    ) -> ComponentMatch | None:
        """Check if a candidate spec is a near-duplicate of an existing component.

        Computes the semantic signature of *spec* and checks cosine
        similarity against all indexed components. Returns the closest
        match if above *threshold*, else None.

        Used by the auto-curation pipeline to prevent promoting
        near-duplicate components to the persistent library.
        """
        signature = _build_semantic_signature(spec)
        query_emb = self._embed(signature)

        with self._lock:
            if self._embeddings is None or len(self._refs) == 0:
                return None
            embeddings = self._embeddings.copy()
            refs = list(self._refs)

        matches = _top_k_matches(query_emb, embeddings, refs, k=1, threshold=threshold)
        if matches:
            ref, score = matches[0]
            name = ref.rsplit("/", 1)[-1] if "/" in ref else ref
            return ComponentMatch(ref=ref, name=name, score=score, layer="embedding")
        return None

    @property
    def alias_count(self) -> int:
        """Number of aliases in Layer 1."""
        with self._lock:
            return len(self._aliases)

    @property
    def component_count(self) -> int:
        """Number of components indexed in Layer 2."""
        with self._lock:
            return len(self._refs)

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        with self._lock:
            return {
                "alias_count": len(self._aliases),
                "component_count": len(self._refs),
                "embedding_dim": self._embeddings.shape[1] if self._embeddings is not None else 0,
                "using_fallback": self._using_fallback,
                "encoder_model": self._encoder_model_name,
                "similarity_threshold": self._similarity_threshold,
            }

    # -- persistence --------------------------------------------------------

    def save(self, path: Path | str | None = None) -> None:
        """Save embeddings to disk for fast reload.

        Saves the embedding matrix as .npz and refs/signatures as a JSON
        sidecar.  Aliases are NOT persisted — they are rebuilt from the
        registry on load (source of truth is the YAML synonyms field).

        Uses atomic write to avoid corruption on crash.
        """
        save_path = Path(path) if path is not None else self._resolve_cache_path()
        if save_path is None:
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self._embeddings is None or len(self._refs) == 0:
                return
            embeddings = self._embeddings.copy()
            refs = list(self._refs)
            signatures = list(self._signatures)

        # Save embeddings (pure float array, no pickle needed)
        np.save(str(save_path), embeddings)

        # Save refs + signatures as JSON sidecar (no pickle)
        meta_path = save_path.with_suffix(".json")
        meta = {"refs": refs, "signatures": signatures}
        try:
            from maxim.utils.atomic_io import atomic_write_text

            atomic_write_text(meta_path, json.dumps(meta))
        except ImportError:
            # Fallback if atomic_io not available (e.g., minimal install)
            with open(meta_path, "w") as f:
                json.dump(meta, f)

        log.debug("ComponentIndex saved %d embeddings to %s", len(refs), save_path)

    def load(self, path: Path | str | None = None) -> bool:
        """Load cached embeddings from disk.

        Returns True if cache was loaded successfully, False otherwise.
        Aliases are NOT loaded — they must be rebuilt from the registry.
        """
        load_path = Path(path) if path is not None else self._resolve_cache_path()
        if load_path is None or not load_path.exists():
            return False

        meta_path = load_path.with_suffix(".json")
        if not meta_path.exists():
            return False

        try:
            # Load embeddings (pure float array, no pickle)
            embeddings = np.load(str(load_path))
            with open(meta_path) as f:
                meta = json.load(f)

            with self._lock:
                self._embeddings = embeddings
                self._refs = meta["refs"]
                self._signatures = meta["signatures"]
            log.debug("ComponentIndex loaded %d embeddings from %s", len(self._refs), load_path)
            return True
        except Exception as e:
            log.warning("Failed to load ComponentIndex cache: %s", e)
            return False

    # -- private ------------------------------------------------------------

    def _resolve_cache_path(self) -> Path | None:
        """Resolve the default cache path."""
        if self._cache_path is not None:
            return self._cache_path
        try:
            from maxim.utils.paths import data_home

            return data_home() / "component_index.npy"
        except Exception:
            return None

    def _build_from_registry(self, registry: ComponentRegistry) -> None:
        """Populate the index from all components in the registry.

        Collects all embeddings in a list first, then vstacks once at
        the end to avoid O(n^2) array copies during construction.
        """
        refs = registry.list_refs()
        log.debug("ComponentIndex building from %d registry entries", len(refs))

        # Collect embeddings in a list, vstack once at the end
        pending_embeddings: list[np.ndarray] = []
        pending_refs: list[str] = []
        pending_sigs: list[str] = []

        for ref in refs:
            try:
                spec = registry.get(ref)
            except Exception:
                continue

            # Use public get_info() to avoid private field access
            info = registry.get_info(ref)
            synonyms = info.synonyms if info is not None else ()

            # Build signature and embed
            signature = _build_semantic_signature(spec)
            embedding = self._embed(signature)

            # Populate alias table
            name = ref.rsplit("/", 1)[-1] if "/" in ref else ref
            self._aliases[name.lower()] = ref
            self._aliases[name.lower().replace("_", " ")] = ref
            if synonyms:
                for syn in synonyms:
                    if syn and syn.strip():
                        self._aliases[syn.strip().lower()] = ref

            pending_refs.append(ref)
            pending_sigs.append(signature)
            pending_embeddings.append(embedding)

        # Bulk vstack — single allocation
        with self._lock:
            self._refs = pending_refs
            self._signatures = pending_sigs
            if pending_embeddings:
                self._embeddings = np.vstack([e.reshape(1, -1) for e in pending_embeddings])
            else:
                self._embeddings = None

        log.info(
            "ComponentIndex built: %d components, %d aliases",
            self.component_count,
            self.alias_count,
        )

    def _ensure_encoder(self) -> None:
        """Lazy-load the embedding encoder (thread-safe).

        Reuses the ``similarity.encoder._get_encoder`` singleton so we
        don't load a duplicate model into memory (~500MB).  Falls back
        to bag-of-words hash if sentence-transformers is not installed.
        """
        if self._encoder_loaded:
            return

        with self._encoder_lock:
            # Double-checked locking
            if self._encoder_loaded:
                return

            try:
                from maxim.similarity.encoder import _get_encoder

                self._encoder = _get_encoder(self._encoder_model_name)
                self._using_fallback = self._encoder is None
                if self._encoder is not None:
                    log.info("ComponentIndex using shared encoder: %s", self._encoder_model_name)
                else:
                    log.debug("sentence-transformers not installed; ComponentIndex using fallback")
            except Exception as e:
                self._encoder = None
                self._using_fallback = True
                log.warning("Failed to load encoder %s: %s", self._encoder_model_name, e)

            self._encoder_loaded = True

    def _embed(self, text: str) -> np.ndarray:
        """Embed a text string. Uses sentence-transformers or fallback."""
        self._ensure_encoder()

        if self._encoder is not None:
            vec = self._encoder.encode(text, convert_to_numpy=True)
            return np.asarray(vec, dtype=np.float32)

        return self._fallback_embed(text)

    @staticmethod
    def _fallback_embed(text: str, dim: int = 384) -> np.ndarray:
        """Deterministic bag-of-words fallback embedding.

        Same approach as LinguisticEncoder's fallback — SHA-256 hash of
        sorted unique words. NOT semantically meaningful. Alias lookups
        (Layer 1) still work; embedding similarity (Layer 2) will only
        match exact word overlap.
        """
        import hashlib

        words = sorted(set(text.lower().split()))
        digest = hashlib.sha256(" ".join(words).encode()).digest()
        vec = np.zeros(dim, dtype=np.float32)
        for i in range(dim):
            vec[i] = (digest[i % len(digest)] / 127.5) - 1.0
        return vec

    def _find_by_embedding(self, query: str) -> ComponentMatch | None:
        """Layer 2: find best match by embedding cosine similarity."""
        query_emb = self._embed(query)

        with self._lock:
            if self._embeddings is None or len(self._refs) == 0:
                return None
            embeddings = self._embeddings.copy()
            refs = list(self._refs)

        matches = _top_k_matches(query_emb, embeddings, refs, k=1, threshold=self._similarity_threshold)
        if matches:
            ref, score = matches[0]
            name = ref.rsplit("/", 1)[-1] if "/" in ref else ref
            return ComponentMatch(ref=ref, name=name, score=score, layer="embedding")
        return None
