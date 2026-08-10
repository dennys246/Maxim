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
from maxim.utils.logging import log_swallowed_exception

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
        from maxim.utils.optional_deps import warn_optional_fallback

        warn_optional_fallback(
            "sentence_transformers",
            fallback="LinguisticEncoder is using deterministic bag-of-words hash embeddings (lower semantic quality)",
            feature="LinguisticEncoder",
            log=logger,
        )
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
            embedding = vec.tolist()
        else:
            embedding = _fallback_embed(text, dim=self.config.fallback_dim)
        # Artifact stamping (1.1 item 7): record the REALIZED state — the
        # dim is measured on the actual vector, and using_fallback is only
        # knowable here (a 384-dim array could be the fallback OR a real
        # 384-dim model; post-hoc array inspection cannot distinguish).
        # Direct call, not a getattr probe (review fold): every other
        # encoder→EC call in this file is direct and typed, and a silent
        # no-op on an EC-shaped substitute missing the method would ship
        # stamp-less state — the exact leak this feature exists to prevent.
        self.ec.record_encoder_provenance(
            "linguistic",
            {
                "model_name": self.config.model_name,
                "using_fallback": self._using_fallback,
                "embedding_dim": len(embedding),
            },
        )
        return embedding

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

        # P2: Update eligibility trace (with temporal anchor for phase-
        # similarity credit — same pattern as encode_decomposed).
        if self._nac is not None:
            agent_id = ""
            if percept.context is not None and hasattr(percept.context, "agent_id"):
                agent_id = percept.context.agent_id or ""
            activation = 1.0 if result.is_new else result.similarity
            temporal_sig = None
            try:
                from maxim.time.temporal_signature import TemporalSignature

                temporal_sig = TemporalSignature.now()
            except Exception:
                log_swallowed_exception()
            self._nac.update_eligibility(agent_id, result.node_id, activation, temporal_sig=temporal_sig)

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

        After this call, ``self.last_node_relations`` contains a mapping
        from ``frozenset({node_a, node_b})`` → relation type for any
        pairs whose chunks carried relation metadata (Stage 2). Callers
        can pass this to ``CaptureEvent.node_relations``.

        Returns:
            List of substrate node IDs (at least one).
        """
        self.last_node_relations: dict[frozenset[str], str] = {}

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
            overrides = self._nac.get_threshold_overrides(
                agent_id,
                base_threshold=self.ec.config.pattern_complete_threshold,
            )
            threshold_override = overrides if overrides else None

        node_ids: list[str] = []
        chunk_node_map: list[tuple[Any, str]] = []  # (chunk, node_id) pairs
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
                # Pass TemporalSignature for SCN-coupled eligibility credit.
                # If the fast-decay trace expires before a reward arrives,
                # distribute_reward can still credit this node via temporal
                # similarity (affordance concept transfer).
                temporal_sig = None
                try:
                    from maxim.time.temporal_signature import TemporalSignature

                    temporal_sig = TemporalSignature.now()
                except Exception:
                    log_swallowed_exception()
                self._nac.update_eligibility(agent_id, result.node_id, activation, temporal_sig=temporal_sig)

            node_ids.append(result.node_id)
            chunk_node_map.append((chunk, result.node_id))

            logger.debug(
                "Decomposed chunk '%s' → node %s (sim=%.3f, new=%s, rel=%s)",
                chunk.text[:30],
                result.node_id[:8],
                result.similarity,
                result.is_new,
                chunk.relation,
            )

        # Stage 2: build relation mapping between node pairs.
        # For each pair of chunks that both have relations, the relation
        # on the "dependent" chunk (the one whose relation describes its
        # connection to the head) annotates the edge.
        if len(chunk_node_map) >= 2:
            import itertools

            for (c1, nid1), (c2, nid2) in itertools.combinations(chunk_node_map, 2):
                if nid1 == nid2:
                    continue
                # Use the relation from whichever chunk has one.
                # First non-None wins (c1 preferred by iteration order).
                rel = c1.relation or c2.relation
                if rel is not None:
                    self.last_node_relations[frozenset({nid1, nid2})] = rel

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

        overrides = self._nac.get_threshold_overrides(
            agent_id,
            base_threshold=self.ec.config.pattern_complete_threshold,
        )
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


# ---------------------------------------------------------------------------
# Sensor encoding — Phase 0 of grounded_language_acquisition.md
# ---------------------------------------------------------------------------


def _stable_basis(name: str, dim: int, salt: str = "") -> list[float]:
    """Deterministic basis vector for a sensor name (+ optional salt).

    Same (name, salt) pair → same basis across processes/machines
    (SHA-256 over ``salt:name``, expanded to ``dim`` floats in [-1, 1]).
    Salt is used by ``_sensor_embed`` to derive two uncorrelated bases
    per sensor — ``low`` and ``high`` — so value changes rotate the
    embedding direction rather than just scaling magnitude.
    """
    keyed = f"{salt}:{name}".encode("utf-8") if salt else name.encode("utf-8")
    seed = hashlib.sha256(keyed).digest()
    out: list[float] = []
    counter = 0
    while len(out) < dim:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for b in block:
            out.append((b / 127.5) - 1.0)
            if len(out) >= dim:
                break
        counter += 1
    return out


def _normalize_value(value: float, value_range: "tuple[float, float] | None" = None) -> float:
    """Map a sensor value to [0, 1] for low/high basis interpolation.

    **Range-aware path (P1):** when ``value_range = (lo, hi)`` is supplied, maps
    linearly ``(v - lo) / (hi - lo)`` and clamps to ``[0, 1]``. A ``[0, 1]``
    range is the identity (unchanged from legacy). A ``[-1, 1]`` range becomes
    the monotonic ``(v + 1) / 2`` across the WHOLE range — center ``0.0 -> 0.5``,
    left ``< 0.5``, right ``> 0.5`` — so a signed sensor's left / center / right
    stay distinct. **This is what a signed azimuth needs**: the legacy
    range-blind map below folds it (see below).

    **Legacy range-blind fallback:** with no ``value_range`` (existing callers),
    reproduces the pre-P1 bimodal map EXACTLY — ``[-1, 0] -> [0, 0.5]`` via
    ``(v + 1) / 2``, ``[0, 1]`` identity — so those callers are byte-identical.
    But it CANNOT tell a ``[-1, 1]`` sensor's positive half from a ``[0, 1]``
    sensor: ``az = +0.5`` and ``hunger = 0.5`` both map to ``0.5``, and the fold
    at ``0`` collides opposite-signed values (``az = -0.5 -> 0.25`` equals
    ``az = +0.25 -> 0.25``). Supply a range for any ``[-1, 1]`` sensor.

    Values outside the range clip rather than raise — telemetry should never
    crash on a weird sensor reading.
    """
    v = float(value)
    if value_range is not None:
        lo, hi = value_range
        if hi > lo:
            return max(0.0, min(1.0, (v - lo) / (hi - lo)))
        return max(0.0, min(1.0, v))  # degenerate range -> clamp
    # --- legacy range-blind fallback (byte-identical to pre-P1) ---
    if v < -1.0:
        v = -1.0
    elif v > 1.0:
        v = 1.0
    if v < 0.0:
        # [-1, 0] half — remap to [0, 0.5]
        return (v + 1.0) * 0.5
    # [0, 1] half stays as-is for sensors already in that range
    return v if v <= 1.0 else 1.0


def _sensor_embed(
    sensors: dict[str, float],
    ranges: "dict[str, tuple[float, float]] | None" = None,
    dim: int = 384,
) -> list[float]:
    """Hash a ``{sensor_name: value}`` dict into a fixed-dim embedding.

    Each sensor contributes ``(1 - v) * basis_low(name) + v * basis_high(name)``
    to the sum, where ``v = _normalize_value(value, ranges.get(name))`` lives in
    [0, 1] and ``basis_low``/``basis_high`` are two uncorrelated SHA-derived
    bases for the same sensor name. This makes value changes *rotate* the
    embedding direction (rather than just scaling magnitude, which cosine
    similarity ignores) so EC pattern_complete_or_separate can actually tell
    "hunger=0" from "hunger=1".

    ``ranges`` (P1) maps a sensor to its ``(lo, hi)`` range so signed sensors
    (azimuth, thermal on ``[-1, 1]``) normalize monotonically instead of folding.
    A sensor absent from ``ranges`` (or ``ranges=None``) uses the legacy
    range-blind map — byte-identical to pre-P1.

    Geometry sanity check — single-sensor sweep, all other sensors
    fixed at neutral, ``dim=384``:

      ``cos(emb(v=0.0), emb(v=0.7)) ≈ 0.4`` (across the EC threshold)
      ``cos(emb(v=0.0), emb(v=1.0)) ≈ 0.0`` (orthogonal — different concept)

    Cross-sensor (hunger up vs thirst up) gives the orthogonal pattern
    that Phase 0 wants to distinguish.

    Empty dict short-circuits to the zero vector (caller should
    avoid encoding empty inputs; ``SensorEncoder.encode_sensors``
    skips this case).
    """
    vec = [0.0] * dim
    if not sensors:
        return vec
    for name in sorted(sensors):  # deterministic ordering
        vr = ranges.get(name) if ranges else None
        v = _normalize_value(sensors[name], vr)
        basis_low = _stable_basis(name, dim, salt="low")
        basis_high = _stable_basis(name, dim, salt="high")
        for i in range(dim):
            vec[i] += (1.0 - v) * basis_low[i] + v * basis_high[i]
    return vec


@dataclass
class SensorEncoderConfig:
    """Configuration for SensorEncoder."""

    embedding_dim: int = 384
    # Skip re-encoding when no sensor has moved by at least this much
    # since the last firing. Phase 0 perf optimization — without it,
    # EC.pattern_complete_or_separate scans every node every tick. The
    # default is conservative (5% of a [0,1] drive range).
    min_delta: float = 0.05
    # Cosine-similarity threshold for the interoception modality.
    # Higher than EC.pattern_complete_threshold (0.44, tuned for
    # paraphrase-mpnet text embeddings) because the SHA-derived
    # low/high bases produce embeddings where a single-sensor swing
    # from 0→1 still leaves cos≈0.83 with the baseline (the other
    # sensors share bases). 0.85 lets multi-sensor pattern shifts
    # form new clusters while drift within a single sensor pattern-
    # completes onto an existing node. Phase 0+ work will retune
    # this once we have cluster-purity data from a Roy run.
    pattern_threshold: float = 0.85


class SensorEncoder:
    """Hashes ``{sensor_name: value}`` dicts into the substrate.

    The substrate's "front door" for non-linguistic input. Mirrors the
    contract of :class:`LinguisticEncoder` (route through EC pattern
    completion, optionally update NAc eligibility), but takes a sensor
    dict rather than a percept and tags embeddings with the
    ``"interoception"`` modality so they don't collide with text/vision
    in EC's per-modality similarity scan.

    Modality choice — ``"interoception"``: the cradle drives this is
    built for (hunger, thirst, core_temperature, arms.thermal,
    arms.pressure, head.thermal) are all interoceptive. The biological
    term matches :class:`SensoryModality.INTEROCEPTION` in
    ``maxim.agents.modality``. A future ``"sensor"`` umbrella tag for
    exteroceptive surfaces (vision-as-pattern, audio-as-pattern) is a
    separate concern; keeping interoception distinct prevents the two
    cluster spaces from polluting each other when both encoders run
    side by side.

    Pattern of use (Phase 0 — substrate-primary AUT):

        encoder = SensorEncoder(ec=ec, atl=atl, nac=nac)
        encoder.encode_sensors(
            agent_id="cradle_infant",
            sensors={"hunger": 0.65, "core_temperature": -0.1, ...},
        )

    Returns the EC node ID the sensor pattern resolved to (existing or
    new). When ``min_delta`` gates the call, returns the previously
    resolved node ID without touching EC.
    """

    def __init__(
        self,
        *,
        ec: Any,
        atl: Any | None = None,
        nac: Any | None = None,
        config: SensorEncoderConfig | None = None,
    ) -> None:
        self.ec = ec
        self.atl = atl
        self._nac = nac
        self.config = config or SensorEncoderConfig()
        self._last_sensors: dict[tuple[str, str], dict[str, float]] = {}  # per (agent_id, modality)
        self._last_node_id: dict[tuple[str, str], str] = {}  # per (agent_id, modality)
        # Ranges identity per stash key (executor-lens review, artifact
        # stamping): the delta gate keyed on VALUES only, so a caller
        # switching ranges (range-blind → range-aware — exactly the P1
        # calibration event) while sensor values sat still was gated out:
        # the STALE node id came back even though the embedding function
        # had changed, and the provenance stamp never saw the mode flip.
        self._last_ranges: dict[tuple[str, str], "dict[str, tuple[float, float]] | None"] = {}

    def encode_sensors(
        self,
        *,
        agent_id: str,
        sensors: dict[str, float],
        modality: str = "interoception",
        ranges: "dict[str, tuple[float, float]] | None" = None,
    ) -> str | None:
        """Encode the current sensor reading; return EC node ID.

        Args:
            agent_id: Per-agent stash key for delta tracking and NAc
                eligibility updates. Empty string is permitted but the
                delta gate becomes process-global instead of per-agent.
            sensors: ``{sensor_name: float_value}`` snapshot. Empty
                dict short-circuits and returns None.
            modality: EC substrate modality tag for the encoded node.
                Defaults to ``"interoception"`` (the cradle drive path —
                existing callers are byte-identical). Exteroceptive
                sensors pass a distinct tag (e.g. ``"audio"`` for sound
                localization) so their nodes form a separate within-modality
                cluster space and inherit the right centroid policy (audio
                is frozen-centroid by default, like interoception — see
                ``ECConfig.frozen_centroid_modalities`` and
                ``docs/plans/perception_pipeline_placement.md`` Q5).
            ranges: (P1) ``{sensor_name: (lo, hi)}`` per-sensor value range so
                signed sensors (azimuth / thermal on ``[-1, 1]``) normalize
                MONOTONICALLY instead of folding (left / center / right stay
                distinct EC clusters). A sensor absent from ``ranges`` (or
                ``ranges=None``) uses the legacy range-blind map — byte-identical
                to pre-P1, so callers that don't pass ranges are unchanged. A
                ``[0, 1]`` range is the identity, also unchanged. Supply ranges
                for any ``[-1, 1]`` sensor whose SIGN must be preserved.

        Returns:
            The EC node ID the pattern resolved to, or ``None`` when
            the input is empty.
        """
        if not sensors:
            return None

        # Delta gate — skip EC scan when nothing has moved enough.
        # Keyed per (agent_id, modality) so an agent encoding multiple
        # modalities through one SensorEncoder (e.g. interoception drives +
        # audio localization) cannot conflate delta stashes across
        # modalities — the per-agent-stash footgun from CLAUDE.md.
        stash_key = (agent_id, modality)
        prev = self._last_sensors.get(stash_key)
        # The gate bypasses only when BOTH values and ranges identity are
        # unchanged: a ranges flip changes the embedding function itself,
        # so returning the cached node would hand back a node computed
        # under a different normalization (and hide the flip from the
        # provenance stamp).
        if (
            prev is not None
            and self._max_delta(prev, sensors) < self.config.min_delta
            and self._last_ranges.get(stash_key) == ranges
        ):
            return self._last_node_id.get(stash_key)

        embedding = _sensor_embed(sensors, ranges=ranges, dim=self.config.embedding_dim)

        # Artifact stamping (1.1 item 7): the sensor-NAME SET and the
        # normalization mode are part of the embedding's identity —
        # `_sensor_embed` sums a SHA basis per name, and range-aware vs
        # range-blind `_normalize_value` are different functions. A bundle
        # missing this stamp lets a range-blind-folded azimuth cluster
        # circulate as if comparable to a range-aware one. Per-sensor
        # granularity: a call may range only SOME sensors, so the mode is
        # recorded as mixed when the ranges dict doesn't cover the set.
        # SCOPE (stated deferral, fabric-plan Stage 4): the RANGE VALUES /
        # units bullet ("a raw-unit range with normalized values is worse
        # than the fold") and body-YAML-derived declarative fields are NOT
        # covered by this pull-forward — they land with the Stage-4
        # projection artifact in 1.3. This stamp records WHETHER ranges
        # were applied, not whether they were the right ranges.
        if not ranges:
            mode = "range-blind"
        elif all(name in ranges for name in sensors):
            mode = "range-aware"
        else:
            mode = "range-partial"
        self.ec.record_encoder_provenance(
            f"sensor:{modality}",
            {
                "embedding_dim": len(embedding),
                "sensor_names": sorted(sensors.keys()),
                "normalization": mode,
            },
        )

        result = self.ec.pattern_complete_or_separate(
            embedding=embedding,
            modality=modality,
            threshold=self.config.pattern_threshold,
        )
        if result.is_new:
            self.ec.register_substrate_node(result.node_id, embedding, modality)

        # ATL activation gives the node a human-readable label so
        # operator tooling that pages through EC nodes sees the
        # sensor snapshot rather than an opaque UUID. The label is
        # not used for substrate matching — the embedding is.
        if self.atl is not None:
            label = self._format_label(sensors)
            try:
                self.atl.activate_substrate_node(
                    node_id=result.node_id,
                    text=label,
                    substrate_modality=modality,
                    embedding_text=label,
                )
            except Exception:
                logger.debug("SensorEncoder: ATL activation raised", exc_info=True)

        # NAc eligibility update — same shape as LinguisticEncoder so
        # downstream temporal-credit machinery treats sensor-pattern
        # nodes uniformly with text-pattern nodes once Phase 0+ wires
        # reward distribution into them.
        if self._nac is not None:
            try:
                from maxim.time.temporal_signature import TemporalSignature

                temporal_sig = TemporalSignature.now()
            except Exception:
                temporal_sig = None
            try:
                activation = 1.0 if result.is_new else result.similarity
                self._nac.update_eligibility(
                    agent_id,
                    result.node_id,
                    activation,
                    temporal_sig=temporal_sig,
                )
            except Exception:
                logger.debug("SensorEncoder: NAc.update_eligibility raised", exc_info=True)

        self._last_sensors[stash_key] = dict(sensors)
        self._last_node_id[stash_key] = result.node_id
        self._last_ranges[stash_key] = dict(ranges) if ranges else None

        logger.debug(
            "Encoded sensor pattern → node %s (sim=%.3f, new=%s, sensors=%d)",
            result.node_id[:8],
            result.similarity,
            result.is_new,
            len(sensors),
        )

        return result.node_id

    @staticmethod
    def _max_delta(a: dict[str, float], b: dict[str, float]) -> float:
        """Largest absolute change across the union of keys."""
        keys = set(a) | set(b)
        return max((abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys), default=0.0)

    @staticmethod
    def _format_label(sensors: dict[str, float]) -> str:
        """Human-readable label for ATL display only; not used for matching."""
        parts = [f"{name}={value:.2f}" for name, value in sorted(sensors.items())]
        return "sensors:" + ",".join(parts)
