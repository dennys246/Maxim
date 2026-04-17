"""Linguistic Encoder — text percept → embedding → EC → ATL substrate path.

P1 implementation. Takes text from a Percept, encodes it into a dense
embedding, routes through EC pattern_complete_or_separate, and activates
or creates the corresponding ATL node.

The encoder is the substrate's "front door" for language input. It runs
alongside (not instead of) the legacy transcript_chunk → prompt path
during the dual-write migration phase.

Embedding model is sentence-transformers (optional dependency via the
``semantic`` extra). Falls back to a bag-of-words hash if
sentence-transformers is not installed — this gives deterministic
behaviour in test environments but won't pass the P1 paraphrase
collapse criterion.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded sentence-transformers model (shared singleton).
_encoder_model: Any = None
_encoder_model_name: str = ""


def _get_encoder(model_name: str = "all-mpnet-base-v2") -> Any | None:
    """Lazy-load the sentence-transformers model.

    Returns None if sentence-transformers is not installed.
    Thread-safe: worst case two threads load simultaneously, second
    overwrites first with identical model.
    """
    global _encoder_model, _encoder_model_name
    if _encoder_model is not None and _encoder_model_name == model_name:
        return _encoder_model

    try:
        from sentence_transformers import SentenceTransformer

        _encoder_model = SentenceTransformer(model_name)
        _encoder_model_name = model_name
        logger.info("LinguisticEncoder loaded model: %s", model_name)
        return _encoder_model
    except ImportError:
        logger.debug("sentence-transformers not installed; LinguisticEncoder will use fallback bag-of-words hash")
        return None
    except Exception as e:
        logger.warning("Failed to load encoder model %s: %s", model_name, e)
        return None


def _fallback_embed(text: str, dim: int = 384) -> list[float]:
    """Deterministic bag-of-words fallback embedding.

    Produces a fixed-dimension vector from a SHA-256 hash of sorted
    unique words. Not semantically meaningful — paraphrase collapse
    will NOT work with this. Exists so the substrate pipeline can
    run end-to-end in test environments without sentence-transformers.
    """
    words = sorted(set(text.lower().split()))
    digest = hashlib.sha256(" ".join(words).encode()).digest()
    # Expand digest bytes into floats in [-1, 1]
    vec = []
    for i in range(dim):
        byte_val = digest[i % len(digest)]
        vec.append((byte_val / 127.5) - 1.0)
    return vec


@dataclass
class EncoderConfig:
    """Configuration for the LinguisticEncoder."""

    model_name: str = "paraphrase-mpnet-base-v2"
    fallback_dim: int = 384


class LinguisticEncoder:
    """Encodes text percepts into the substrate.

    Coordinates: text → embedding → EC.pattern_complete_or_separate →
    ATL.activate_substrate_node. Populates ``percept.embedding`` and
    ``percept.substrate_node_id``.

    Usage::

        encoder = LinguisticEncoder(ec=ec, atl=atl)
        encoder.encode(percept)
        # percept.embedding and percept.substrate_node_id are now set
    """

    def __init__(
        self,
        ec: Any,
        atl: Any,
        config: EncoderConfig | None = None,
        nac: Any | None = None,
        decomposer: Any | None = None,
    ) -> None:
        self.ec = ec
        self.atl = atl
        self.config = config or EncoderConfig()
        self._nac = nac  # P2: for reward-bias threshold overrides
        self._decomposer = decomposer  # Concept decomposition (optional)
        self._model: Any | None = None
        self._model_loaded = False
        self._using_fallback = False

    def _ensure_model(self) -> None:
        """Ensure the embedding model is loaded (lazy)."""
        if self._model_loaded:
            return
        self._model = _get_encoder(self.config.model_name)
        self._using_fallback = self._model is None
        self._model_loaded = True

    def embed(self, text: str) -> list[float]:
        """Produce a dense embedding for a text string."""
        self._ensure_model()
        if self._model is not None:
            vec = self._model.encode(text, convert_to_numpy=True)
            return vec.tolist()
        return _fallback_embed(text, dim=self.config.fallback_dim)

    def encode(self, percept: Any) -> str | None:
        """Run the full substrate encoding pipeline on a percept.

        Extracts text from ``percept.transcript_chunk`` or
        ``percept.content``, embeds it, routes through EC, and
        activates the ATL node.

        When a decomposer is wired and the modality is ``"text"``,
        breaks the input into concept-level chunks and encodes each
        independently via ``encode_decomposed``. The first node ID
        goes to ``percept.substrate_node_id``; embedding is from the
        first chunk. Non-text modalities bypass decomposition.

        Mutates the percept in-place: sets ``percept.embedding`` and
        ``percept.substrate_node_id``.

        Returns:
            The ATL node ID if encoding succeeded, None if no text
            was available.
        """
        text = percept.transcript_chunk or percept.content
        if not text:
            return None

        from maxim.agents.modality import substrate_modality

        modality = substrate_modality(percept)

        # Concept decomposition path: text modality + decomposer wired
        if self._decomposer is not None and modality == "text":
            agent_id = ""
            if percept.context is not None and hasattr(percept.context, "agent_id"):
                agent_id = percept.context.agent_id or ""
            node_ids = self.encode_decomposed(text, modality, agent_id)
            if node_ids:
                percept.substrate_node_id = node_ids[0]
                # Embedding from the first chunk's text (aligned with substrate_node_id)
                chunks = self._decomposer.extract(text)
                percept.embedding = self.embed(chunks[0].text)
            return node_ids[0] if node_ids else None

        # Standard single-node path (non-text or no decomposer)
        embedding = self.embed(text)
        percept.embedding = embedding

        # EC pattern complete or separate.
        # Note: `is not None` — NAc defines __len__ over causal links, so
        # `if self._nac` is falsy for a fresh NAc with zero links even
        # though it's wired. P2 reward overrides must fire regardless of
        # whether any causal links have been recorded yet.
        threshold_override = self._get_reward_overrides(percept) if self._nac is not None else None
        result = self.ec.pattern_complete_or_separate(
            embedding=embedding,
            modality=modality,
            threshold_override=threshold_override,
        )

        if result.is_new:
            self.ec.register_substrate_node(result.node_id, embedding, modality)

        self.atl.activate_substrate_node(
            node_id=result.node_id,
            text=text,
            substrate_modality=modality,
            embedding_text=text,
        )

        percept.substrate_node_id = result.node_id

        # P2: Update eligibility trace
        if self._nac is not None:
            agent_id = ""
            if percept.context is not None and hasattr(percept.context, "agent_id"):
                agent_id = percept.context.agent_id or ""
            activation = 1.0 if result.is_new else result.similarity
            self._nac.update_eligibility(agent_id, result.node_id, activation)

        logger.debug(
            "Encoded percept → node %s (sim=%.3f, new=%s, mod=%s)",
            result.node_id[:8],
            result.similarity,
            result.is_new,
            modality,
        )

        return result.node_id

    def encode_decomposed(self, text: str, modality: str, agent_id: str = "") -> list[str]:
        """Concept-decomposed encoding: text → chunks → embed each → EC → node IDs.

        If a decomposer is wired and the modality is ``"text"``, breaks
        the input into concept-level chunks (e.g., noun phrases) and
        encodes each independently. Non-text modalities bypass
        decomposition (vision, proprioceptive, SEM inputs should not
        be noun-chunked).

        All returned node IDs should land in the same
        ``CaptureEvent.activated_nodes`` so they co-activate in one
        episode and get Hebbian-bound together.

        Returns:
            List of substrate node IDs (at least one).
        """
        if not text or not text.strip():
            return []

        # Modality gate: decompose text only
        if self._decomposer is not None and modality == "text":
            chunks = self._decomposer.extract(text)
        else:
            from maxim.similarity.decomposer import ConceptChunk

            chunks = [ConceptChunk(text=text, span=(0, len(text)))]

        threshold_override = None
        if self._nac is not None:
            overrides = self._nac.get_threshold_overrides(agent_id)
            threshold_override = overrides if overrides else None

        node_ids: list[str] = []
        for chunk in chunks:
            embedding = self.embed(chunk.text)

            result = self.ec.pattern_complete_or_separate(
                embedding=embedding,
                modality=modality,
                threshold_override=threshold_override,
            )

            if result.is_new:
                self.ec.register_substrate_node(result.node_id, embedding, modality)

            self.atl.activate_substrate_node(
                node_id=result.node_id,
                text=chunk.text,
                substrate_modality=modality,
                embedding_text=chunk.text,
            )

            if self._nac is not None:
                activation = 1.0 if result.is_new else result.similarity
                self._nac.update_eligibility(agent_id, result.node_id, activation)

            node_ids.append(result.node_id)

            logger.debug(
                "Decomposed chunk '%s' → node %s (sim=%.3f, new=%s)",
                chunk.text[:30],
                result.node_id[:8],
                result.similarity,
                result.is_new,
            )

        return node_ids

    def _get_reward_overrides(self, percept: Any) -> dict[str, float] | None:
        """P2: Get per-node threshold overrides from NAc reward bias.

        Returns None if NAc has no reward biases. Otherwise returns
        a dict mapping node_id → adjusted threshold for nodes whose
        reward bias should widen their recognition radius.
        """
        if self._nac is None:
            return None

        # Determine agent_id from percept context
        agent_id = ""
        if percept.context is not None and hasattr(percept.context, "agent_id"):
            agent_id = percept.context.agent_id or ""

        overrides = self._nac.get_threshold_overrides(agent_id)
        return overrides if overrides else None

    @property
    def using_fallback(self) -> bool:
        """True if using the bag-of-words fallback instead of sentence-transformers."""
        self._ensure_model()
        return self._using_fallback

    def stats(self) -> dict[str, Any]:
        """Return encoder statistics."""
        return {
            "model_name": self.config.model_name,
            "using_fallback": self._using_fallback,
            "model_loaded": self._model_loaded,
        }
