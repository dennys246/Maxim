"""P4 Stage 2 v2 — parameterized fixture builder.

Shared helper used by:

- ``scripts/p4_mug_test_sweep.py`` (one-shot mug test report)
- ``scripts/p4_mug_test_sweep_v2.py`` (12-combination calibration sweep)
- ``tests/substrate/test_p4_mug_test_roundtrip.py`` (subprocess round-trip)

Stage 2 v1 kept this logic inside the sweep script and the round-trip
test imported it via ``from scripts.p4_mug_test_sweep import _build_and_bind``.
Round 2 review flagged two problems with that arrangement:

1. **Layering smell** (Exec #4, Arch #9): tests importing from ``scripts/``
   is not a documented package structure and depends on ``sys.path``
   manipulation that doesn't propagate to subprocess children.
2. **Encoder-confound discipline** (Arch #3, #5): the v1 helper hard-
   coded paraphrase-mpnet + two module-level EC thresholds, blocking
   Stage 3 Arm B (CLIP-text + hippocampus) from reusing the same
   orchestrator without forking.

The v2 helper lives in ``tests/substrate/`` (sanctioned import origin
for test + script consumers via the repo-root sys.path that conftest
sets up), takes encoder factories + per-modality thresholds + an
optional ``FixtureNoiseConfig`` + an optional ``FixtureBridgeConfig``
as parameters, and is the single source of truth for "how Phase 2D
wires CLIP + hippocampus."

Design contract
===============

``build_and_bind(descriptor, images, config) -> BuildResult``

``BuildConfig`` carries:

- ``text_encoder: TextEncoder`` — callable ``(str) -> list[float]`` +
  a human-readable name for logging
- ``vision_encoder: VisionEncoder`` — callable ``(PIL.Image) ->
  list[float]`` + a name
- ``text_ec_threshold: float`` — per-call threshold for the text side
- ``vision_ec_threshold: float`` — per-call threshold for the vision
  side
- ``noise: FixtureNoiseConfig | None`` — cross-class contamination
  parameters (default None = no noise)
- ``bridges: FixtureBridgeConfig | None`` — text-text bridge topology
  (default None = no bridges)
- ``seed: int`` — deterministic noise/bridge sampling

``BuildResult`` carries:

- ``ec``, ``hippocampus`` — wired bio-system instances
- ``class_results: list[ClassResult]`` — per-class text + vision node
  ids + placeholder recall fields
- ``noise_edges: list[tuple[str, str, int]]`` — the cross-class
  (text_node, vision_node, reinforcement_count) contamination edges
  that got written, for post-hoc accounting
- ``bridge_nodes: list[str]`` — the text bridge node ids (if any)
- ``stats: dict`` — build-time counts for the report (total episodes
  written, total edges created, etc.)

The v2 helper intentionally does NOT run retrieval or compute metrics.
It returns a wired hippocampus + node ids; the caller (sweep script,
round-trip test, future Stage 3 Arm B/C runners) decides what to
measure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TextEncoder:
    """A text encoder adapter. ``encode(text)`` returns a list[float]
    in whatever dimension the underlying model uses."""

    name: str
    encode: Callable[[str], list[float]]


@dataclass(frozen=True)
class VisionEncoder:
    """A vision encoder adapter. ``encode(image)`` returns a list[float]."""

    name: str
    encode: Callable[["Image"], list[float]]


@dataclass(frozen=True)
class FixtureNoiseConfig:
    """Cross-class contamination parameters.

    Each class's `text_X` gains 1 noise pair (text_X bound to a
    vision node from a DIFFERENT class) reinforced ``noise_reps``
    times. This creates a weak Hebbian edge at weight
    ``hebbian_init + (noise_reps - 1) * hebbian_delta``, clipped at
    ``hebbian_max``. Setting ``noise_reps=0`` disables noise (the
    config still exists but no noise edges are written).

    The cross-class partner for each class's noise is picked
    deterministically from a seeded shuffle — class `i` gets the
    vision_0 of class `(i+1) % n_classes`. This rotating assignment
    ensures every class is both a source and target of noise exactly
    once, and the assignment is reproducible from the seed alone.
    """

    noise_reps: int = 0

    def __post_init__(self) -> None:
        if self.noise_reps < 0:
            raise ValueError(f"noise_reps must be >= 0, got {self.noise_reps}")


@dataclass(frozen=True)
class FixtureBridgeConfig:
    """Text-text bridge topology parameters.

    The only topology supported in Stage 2 v2 is ``shared_superclass``:
    a single shared text node (default ``text_flower``) is bound to
    one vision from each class AND bound to each class's text_X in
    separate episodes. This creates a dense bridge graph where every
    pair of distinct classes is connected via the shared superclass
    text node at path length 3 (text_X → text_flower → vision_Y_0).

    Topology B (pairwise bridges) is documented in the P4 plan as a
    known alternative but intentionally NOT implemented here to keep
    the v2 sweep focused. If Stage 3's head-to-head needs sensitivity
    analysis, revive Topology B by adding ``topology`` as a Literal
    field on this config.
    """

    shared_superclass_name: str = "text_flower"
    enabled: bool = False


@dataclass(frozen=True)
class BuildConfig:
    """Full parameterization for a P4 Stage 2 fixture build."""

    text_encoder: TextEncoder
    vision_encoder: VisionEncoder
    text_ec_threshold: float
    vision_ec_threshold: float
    noise: FixtureNoiseConfig = field(default_factory=FixtureNoiseConfig)
    bridges: FixtureBridgeConfig = field(default_factory=FixtureBridgeConfig)
    seed: int = 0


@dataclass
class ClassResult:
    """Per-class substrate node ids populated by the build step.
    Recall/topology fields are filled in by downstream measurement
    (sweep runner, round-trip test, Stage 3 arm runners). The build
    helper sets text_node_id + vision_node_ids and leaves the rest
    at their dataclass defaults.
    """

    class_name: str
    text_node_id: str
    vision_node_ids: tuple[str, ...]
    forward_top5_recall: float = 0.0
    forward_hits_in_topk: list[int] = field(default_factory=list)
    reverse_text_match_rate: float = 0.0
    forward_path_hops: list[int] = field(default_factory=list)


@dataclass
class BuildStats:
    """Build-time counters surfaced to the sweep report."""

    n_classes: int = 0
    n_class_episodes: int = 0
    n_noise_episodes: int = 0
    n_bridge_episodes: int = 0
    n_total_episodes: int = 0
    n_text_nodes: int = 0
    n_vision_nodes: int = 0
    n_bridge_text_nodes: int = 0


@dataclass
class BuildResult:
    """Everything a caller needs after a build pass."""

    ec: Any
    hippocampus: Any
    class_results: list[ClassResult]
    noise_edges: list[tuple[str, str, int]]
    bridge_nodes: list[str]
    stats: BuildStats


# ─────────────────────────────────────────────────────────────────────────
# Bio-system factories
# ─────────────────────────────────────────────────────────────────────────


def build_hippocampus() -> Any:
    """Build a fresh Hippocampus with default P3a/P4 config."""
    from maxim.memory.hippocampus import (
        EpisodeConfig,
        HebbianConfig,
        Hippocampus,
        HippocampusConfig,
        RetrievalConfig,
    )

    return Hippocampus(
        HippocampusConfig(
            episode=EpisodeConfig(
                boundary_tick_gap=50,
                hebbian=HebbianConfig(init=0.3, delta=0.1, max_weight=1.0),
                retrieval=RetrievalConfig(decay=0.7, threshold=0.001, max_depth=5),
            )
        )
    )


def build_ec() -> Any:
    """Build a fresh EntorhinalCortex at default pattern-complete threshold."""
    from maxim.similarity.ec import ECConfig, EntorhinalCortex

    return EntorhinalCortex(ECConfig())


# ─────────────────────────────────────────────────────────────────────────
# Encoder adapters — factory functions that return a TextEncoder/VisionEncoder
# ─────────────────────────────────────────────────────────────────────────


def paraphrase_mpnet_text_encoder() -> TextEncoder:
    """The substrate's default text encoder for Arm A. Lazy-loaded at
    first encode call via the cached sentence-transformers singleton
    in ``maxim.similarity.encoder``."""
    from sentence_transformers import SentenceTransformer

    logger.info("loading paraphrase-mpnet-base-v2 (Arm A substrate text encoder)")
    model = SentenceTransformer("paraphrase-mpnet-base-v2")

    def _encode(text: str) -> list[float]:
        vec = model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    return TextEncoder(name="paraphrase-mpnet-base-v2", encode=_encode)


def clip_text_encoder() -> TextEncoder:
    """CLIP-text side for Stage 3 Arm B / Arm C. Uses the same
    VisionEncoder singleton to avoid double-loading the model — a
    single ``SentenceTransformer("clip-ViT-B-32")`` serves both text
    and image sides in CLIP's shared 512-d space.
    """
    from maxim.models.vision.clip_encoder import VisionEncoder as MaximVisionEncoder

    encoder = MaximVisionEncoder()

    def _encode(text: str) -> list[float]:
        return encoder.encode_text(text).tolist()

    return TextEncoder(name="clip-ViT-B-32", encode=_encode)


def clip_vision_encoder() -> VisionEncoder:
    """CLIP ViT-B-32 vision side for Arms A/B/C."""
    from maxim.models.vision.clip_encoder import VisionEncoder as MaximVisionEncoder

    encoder = MaximVisionEncoder()

    def _encode(image: "Image") -> list[float]:
        return encoder.encode_image(image).tolist()

    return VisionEncoder(name="clip-ViT-B-32", encode=_encode)


# ─────────────────────────────────────────────────────────────────────────
# Episode binding helpers
# ─────────────────────────────────────────────────────────────────────────


def _make_pair_episode(hippocampus: Any, text_node_id: str, vision_node_id: str, tick: int) -> None:
    """Two events at consecutive ticks on the same cross_modal channel;
    one text, one vision; finalize_pending_episode closes and applies
    Hebbian + drains the sidecar."""
    from maxim.memory.episode import CaptureEvent

    hippocampus.observe_episode_event(
        CaptureEvent(
            tick=tick,
            channel="cross_modal",
            activated_nodes=(text_node_id,),
            modality="text",
        )
    )
    hippocampus.observe_episode_event(
        CaptureEvent(
            tick=tick + 1,
            channel="cross_modal",
            activated_nodes=(vision_node_id,),
            modality="vision",
        )
    )
    hippocampus.finalize_pending_episode()


def _make_text_pair_episode(hippocampus: Any, text_a_id: str, text_b_id: str, tick: int) -> None:
    """Two text events in the same episode — creates a text-text
    Hebbian edge. Used by the bridge topology to build text_X → text_flower
    links that Option 2's split filter would traverse."""
    from maxim.memory.episode import CaptureEvent

    hippocampus.observe_episode_event(
        CaptureEvent(
            tick=tick,
            channel="cross_modal",
            activated_nodes=(text_a_id,),
            modality="text",
        )
    )
    hippocampus.observe_episode_event(
        CaptureEvent(
            tick=tick + 1,
            channel="cross_modal",
            activated_nodes=(text_b_id,),
            modality="text",
        )
    )
    hippocampus.finalize_pending_episode()


# ─────────────────────────────────────────────────────────────────────────
# The main build entry point
# ─────────────────────────────────────────────────────────────────────────


def build_and_bind(
    fixture_descriptor: Any,
    fixture_images: dict[tuple[str, int], Any],
    config: BuildConfig,
) -> BuildResult:
    """Wire EC + Hippocampus, encode all fixture pairs, optionally add
    noise + bridges, bind everything in episodes.

    Returns a ``BuildResult`` with the wired bio-systems + per-class
    node ids + build accounting. Recall/topology numbers are NOT
    computed here — callers run their own measurement pass against
    the returned hippocampus.

    **Determinism:** noise partners are picked via a seeded numpy
    RNG; bridge assignments are deterministic given the fixture
    descriptor's class order. Same ``config.seed`` + same
    ``fixture_descriptor`` produces byte-identical build output.
    """
    ec = build_ec()
    hippocampus = build_hippocampus()

    class_results: list[ClassResult] = []
    text_node_by_class: dict[str, str] = {}
    vision_nodes_by_class: dict[str, list[str]] = {}

    tick = 0
    stats = BuildStats(n_classes=len(fixture_descriptor.classes))

    # ---------- Pass 1: encode + bind class-correct pairs ----------
    for cls in fixture_descriptor.classes:
        text_prompt = cls.name  # bare class name; see plan's text-side calibration notes
        text_emb = config.text_encoder.encode(text_prompt)
        text_result = ec.pattern_complete_or_separate(
            text_emb, modality="text", threshold=config.text_ec_threshold, geometry=None
        )
        text_node_id = text_result.node_id
        ec.register_substrate_node(text_node_id, text_emb, "text")
        text_node_by_class[cls.name] = text_node_id
        stats.n_text_nodes += 1

        vision_node_ids: list[str] = []
        for sample_idx in cls.sample_indices:
            image = fixture_images[(cls.name, sample_idx)]
            img_emb = config.vision_encoder.encode(image)
            vis_result = ec.pattern_complete_or_separate(
                img_emb, modality="vision", threshold=config.vision_ec_threshold, geometry=None
            )
            ec.register_substrate_node(vis_result.node_id, img_emb, "vision")
            vision_node_ids.append(vis_result.node_id)
            stats.n_vision_nodes += 1

            _make_pair_episode(hippocampus, text_node_id, vis_result.node_id, tick=tick)
            tick += 10
            stats.n_class_episodes += 1

        vision_nodes_by_class[cls.name] = vision_node_ids
        class_results.append(
            ClassResult(
                class_name=cls.name,
                text_node_id=text_node_id,
                vision_node_ids=tuple(vision_node_ids),
            )
        )

    # ---------- Pass 2: noise ----------
    # Rotating assignment — class i's noise partner is class (i+1) mod n.
    # Deterministic AND ensures every class is both source and target
    # of noise exactly once. No RNG needed; class order from the
    # fixture descriptor is the canonical sort.
    noise_edges: list[tuple[str, str, int]] = []
    if config.noise.noise_reps > 0:
        class_names = [c.class_name for c in class_results]
        n = len(class_names)
        for i, src_name in enumerate(class_names):
            dst_name = class_names[(i + 1) % n]
            src_text = text_node_by_class[src_name]
            # Pick the first vision node from the destination class as the noise target.
            dst_vision = vision_nodes_by_class[dst_name][0]

            for _rep in range(config.noise.noise_reps):
                _make_pair_episode(hippocampus, src_text, dst_vision, tick=tick)
                tick += 10
                stats.n_noise_episodes += 1
            noise_edges.append((src_text, dst_vision, config.noise.noise_reps))

    # ---------- Pass 3: bridges (Topology A — shared superclass) ----------
    bridge_nodes: list[str] = []
    if config.bridges.enabled:
        # The bridge text node is a real substrate text node, EC-registered
        # so retrieve_cross_modal snapshots it into the text modality bucket.
        bridge_text = config.bridges.shared_superclass_name
        bridge_text_emb = config.text_encoder.encode(bridge_text)
        bridge_result = ec.pattern_complete_or_separate(
            bridge_text_emb, modality="text", threshold=config.text_ec_threshold, geometry=None
        )
        bridge_node_id = bridge_result.node_id
        ec.register_substrate_node(bridge_node_id, bridge_text_emb, "text")
        bridge_nodes.append(bridge_node_id)
        stats.n_bridge_text_nodes += 1

        # For each class: bind (text_X, text_flower) in one episode → creates
        # text-text edge text_X ↔ text_flower. Under Option 1's single-hop
        # filter, this edge is UNREACHABLE by retrieve_cross_modal because
        # text_flower is not in the vision modality bucket.
        for cls in class_results:
            _make_text_pair_episode(hippocampus, cls.text_node_id, bridge_node_id, tick=tick)
            tick += 10
            stats.n_bridge_episodes += 1

        # For each class: bind (text_flower, vision_X_0) in one episode →
        # creates text_flower ↔ vision_X_0 edge. Combined with the
        # text-text edges above, this gives text_X → text_flower → vision_Y_0
        # at 2 hops (blocked by Option 1, reachable under Option 2).
        for cls in class_results:
            first_vision = cls.vision_node_ids[0]
            _make_pair_episode(hippocampus, bridge_node_id, first_vision, tick=tick)
            tick += 10
            stats.n_bridge_episodes += 1

    stats.n_total_episodes = stats.n_class_episodes + stats.n_noise_episodes + stats.n_bridge_episodes

    logger.info(
        "build_and_bind done: %d classes, %d class episodes, %d noise episodes, %d bridge episodes",
        stats.n_classes,
        stats.n_class_episodes,
        stats.n_noise_episodes,
        stats.n_bridge_episodes,
    )

    return BuildResult(
        ec=ec,
        hippocampus=hippocampus,
        class_results=class_results,
        noise_edges=noise_edges,
        bridge_nodes=bridge_nodes,
        stats=stats,
    )
