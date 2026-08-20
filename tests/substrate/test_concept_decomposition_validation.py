"""Concept decomposition validation — proves decomposition improves cross-modal retrieval.

Two test classes:

1. ``TestDecompositionMechanism`` — fast suite, synthetic embeddings only.
   Demonstrates that decomposed naturalistic sentences create multiple
   substrate nodes, that those nodes pattern-complete to bare concepts,
   and that cross-modal binding through decomposed chunks enables
   concept-level retrieval that whole-sentence encoding cannot.

2. ``TestDecompositionValidation`` — slow suite, requires sentence-transformers
   + spaCy.  Runs 5 naturalistic scenes through with-/without-decomposition
   arms and measures concept-level cross-modal recall.  Saves results to
   ``docs/experiments/results/concept_decomposition_validation.json``.

The primary metric is **concept-level cross-modal recall rate**: for each
concept extracted from a scene, query ``retrieve_cross_modal(concept_node,
target_modality="vision")`` and check whether the paired vision node
appears in the top-5 results.

Pass criteria (validation suite):
- decomposed_recall > baseline_recall  (strict improvement)
- decomposed_recall >= 0.60            (minimum bar)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from maxim.memory.episode import CaptureEvent
from maxim.memory.hippocampus import (
    EpisodeConfig,
    HebbianConfig,
    Hippocampus,
    HippocampusConfig,
    RetrievalConfig,
)
from maxim.similarity.decomposer import ConceptChunk, ConceptDecomposer
from maxim.similarity.ec import ECConfig, EntorhinalCortex

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Fixture scenes
# ─────────────────────────────────────────────────────────────────────────

SCENES = [
    {
        "description": "I see a blue mug on the wooden table",
        "concepts": ["blue mug", "wooden table"],
        "vision_labels": ["mug", "table"],
    },
    {
        "description": "The rusty sword lies next to the leather shield",
        "concepts": ["rusty sword", "leather shield"],
        "vision_labels": ["sword", "shield"],
    },
    {
        "description": "A red ball sits in the cardboard box near the glass window",
        "concepts": ["red ball", "cardboard box", "glass window"],
        "vision_labels": ["ball", "box", "window"],
    },
    {
        "description": "The old book rests on the marble shelf",
        "concepts": ["old book", "marble shelf"],
        "vision_labels": ["book", "shelf"],
    },
    {
        "description": "A golden key hangs from the iron chain",
        "concepts": ["golden key", "iron chain"],
        "vision_labels": ["key", "chain"],
    },
]


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _fresh_hippocampus() -> Hippocampus:
    return Hippocampus(
        HippocampusConfig(
            episode=EpisodeConfig(
                boundary_tick_gap=50,
                hebbian=HebbianConfig(init=0.3, delta=0.1, max_weight=1.0),
                retrieval=RetrievalConfig(decay=0.7, threshold=0.001, max_depth=5),
            )
        )
    )


def _make_pair_episode(
    hippocampus: Hippocampus,
    text_node_id: str,
    vision_node_id: str,
    tick: int,
) -> None:
    """Bind a text node and a vision node in one episode."""
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


def _make_multi_text_vision_episode(
    hippocampus: Hippocampus,
    text_node_ids: list[str],
    vision_node_ids: list[str],
    tick: int,
) -> None:
    """Bind multiple text nodes and multiple vision nodes in one episode.

    Each text node gets its own CaptureEvent (modality="text"), each
    vision node gets its own CaptureEvent (modality="vision").  All
    events share the same tick range so they land in one episode.
    """
    t = tick
    for tn in text_node_ids:
        hippocampus.observe_episode_event(
            CaptureEvent(tick=t, channel="cross_modal", activated_nodes=(tn,), modality="text")
        )
        t += 1
    for vn in vision_node_ids:
        hippocampus.observe_episode_event(
            CaptureEvent(tick=t, channel="cross_modal", activated_nodes=(vn,), modality="vision")
        )
        t += 1
    hippocampus.finalize_pending_episode()


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _has_spacy() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except (ImportError, OSError):
        return False


# ─────────────────────────────────────────────────────────────────────────
# Mock decomposer for mechanism tests (no spaCy dependency)
# ─────────────────────────────────────────────────────────────────────────


class FixedDecomposer:
    """Returns pre-defined chunks for known test sentences.

    Falls back to identity (full text as one chunk) for unknown inputs.
    """

    _DECOMPOSITIONS: dict[str, list[str]] = {
        "I see a blue mug on the wooden table": ["blue mug", "wooden table"],
        "The rusty sword lies next to the leather shield": ["rusty sword", "leather shield"],
        "A red ball sits in the cardboard box near the glass window": [
            "red ball",
            "cardboard box",
            "glass window",
        ],
        "The old book rests on the marble shelf": ["old book", "marble shelf"],
        "A golden key hangs from the iron chain": ["golden key", "iron chain"],
    }

    def extract(self, text: str) -> list[ConceptChunk]:
        chunks = self._DECOMPOSITIONS.get(text)
        if chunks is not None:
            return [ConceptChunk(text=c) for c in chunks]
        return [ConceptChunk(text=text, span=(0, len(text)))]


# ─────────────────────────────────────────────────────────────────────────
# Mechanism tests — fast, synthetic embeddings
# ─────────────────────────────────────────────────────────────────────────


class TestDecompositionMechanism:
    """Fast mechanism tests with synthetic embeddings.

    Proves the decomposition pipeline creates the correct substrate
    structure and that cross-modal retrieval benefits from it.
    No sentence-transformers or spaCy required.
    """

    def _make_encoder(self, *, with_decomposer: bool = True):
        """Build a LinguisticEncoder backed by fallback (hash) embeddings.

        Forces the fallback by monkeypatching ``_get_encoder`` to return
        None, so tests use the deterministic bag-of-words hash regardless
        of whether sentence-transformers is installed.
        """
        from unittest.mock import patch

        from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.70))
        atl = MagicMock()
        atl.activate_substrate_node = MagicMock(side_effect=lambda node_id, **kw: node_id)

        decomposer = ConceptDecomposer(strategy=FixedDecomposer()) if with_decomposer else None

        # Force fallback embeddings so different text -> different hash vectors
        with patch("maxim.similarity.encoder._get_encoder", return_value=None):
            encoder = LinguisticEncoder(
                ec=ec,
                atl=atl,
                config=EncoderConfig(model_name="paraphrase-mpnet-base-v2"),
                decomposer=decomposer,
            )
            # Trigger model loading while patch is active
            encoder._ensure_model()

        return encoder, ec, atl

    # ── Test 1: decomposed sentences create multiple nodes ──

    def test_decomposed_sentence_creates_multiple_nodes(self) -> None:
        """A naturalistic sentence with decomposition should produce
        one substrate node per noun-phrase chunk, not one blob node.
        """
        encoder, ec, _atl = self._make_encoder(with_decomposer=True)

        node_ids = encoder.encode_decomposed("I see a blue mug on the wooden table", "text")

        assert len(node_ids) == 2, f"Expected 2 decomposed nodes (blue mug, wooden table), got {len(node_ids)}"
        assert len(set(node_ids)) == 2, "Decomposed nodes should be distinct"

        # Both nodes registered in EC
        for nid in node_ids:
            assert nid in ec._substrate_nodes, f"Node {nid} not registered in EC"

    def test_without_decomposition_creates_single_node(self) -> None:
        """Without decomposition, the same sentence produces one blob node."""
        encoder, ec, _atl = self._make_encoder(with_decomposer=False)

        node_ids = encoder.encode_decomposed("I see a blue mug on the wooden table", "text")

        assert len(node_ids) == 1, f"Expected 1 node without decomposition, got {len(node_ids)}"

    # ── Test 2: decomposed concept nodes pattern-complete to bare concepts ──

    def test_decomposed_chunk_pattern_completes_to_same_concept(self) -> None:
        """If 'blue mug' is already encoded, a second encoding of
        'blue mug' (from decomposition of a sentence) should
        pattern-complete to the same node.
        """
        encoder, ec, _atl = self._make_encoder(with_decomposer=True)

        # First: encode bare "blue mug"
        bare_ids = encoder.encode_decomposed("blue mug", "text")
        assert len(bare_ids) == 1
        bare_node = bare_ids[0]

        # Second: encode a sentence containing "blue mug" — decomposer
        # extracts "blue mug" and "wooden table"
        sentence_ids = encoder.encode_decomposed("I see a blue mug on the wooden table", "text")
        assert len(sentence_ids) == 2

        # The "blue mug" chunk from the sentence should have pattern-
        # completed to the same node as the bare "blue mug" (because
        # both use the same fallback-hash embedding for identical text).
        assert bare_node in sentence_ids, (
            f"Bare 'blue mug' node {bare_node[:8]} not found in "
            f"decomposed sentence nodes {[n[:8] for n in sentence_ids]}"
        )

    # ── Test 3: cross-modal retrieval through decomposition ──

    def test_cross_modal_binding_through_decomposition(self) -> None:
        """Decomposed text nodes bound to vision nodes enable concept-level
        cross-modal retrieval.

        Setup:
        1. Encode "I see a blue mug on the wooden table" WITH decomposition
           -> text nodes: [node_mug, node_table]
        2. Create vision nodes for mug and table
        3. Bind (node_mug, vision_mug) and (node_table, vision_table)
        4. Query: retrieve_cross_modal(node_mug, target="vision") -> vision_mug
        """
        encoder, ec, _atl = self._make_encoder(with_decomposer=True)
        hippocampus = _fresh_hippocampus()

        # Encode sentence with decomposition
        text_node_ids = encoder.encode_decomposed("I see a blue mug on the wooden table", "text")
        assert len(text_node_ids) == 2
        node_mug, node_table = text_node_ids

        # Create synthetic vision nodes
        vision_mug_emb = [0.1] * 384
        vision_mug_emb[0] = 0.9
        vision_table_emb = [0.1] * 384
        vision_table_emb[1] = 0.9

        mug_vis = ec.pattern_complete_or_separate(vision_mug_emb, "vision")
        ec.register_substrate_node(mug_vis.node_id, vision_mug_emb, "vision")

        table_vis = ec.pattern_complete_or_separate(vision_table_emb, "vision")
        ec.register_substrate_node(table_vis.node_id, vision_table_emb, "vision")

        # Bind decomposed text nodes to corresponding vision nodes
        _make_pair_episode(hippocampus, node_mug, mug_vis.node_id, tick=0)
        _make_pair_episode(hippocampus, node_table, table_vis.node_id, tick=100)

        # Cross-modal retrieval: text concept -> vision
        mug_results = hippocampus.retrieve_cross_modal(node_mug, target_modality="vision", limit=5)
        mug_partner_ids = {nid for nid, _ in mug_results}
        assert mug_vis.node_id in mug_partner_ids, "vision_mug not retrieved from decomposed text concept node_mug"

        table_results = hippocampus.retrieve_cross_modal(node_table, target_modality="vision", limit=5)
        table_partner_ids = {nid for nid, _ in table_results}
        assert table_vis.node_id in table_partner_ids, (
            "vision_table not retrieved from decomposed text concept node_table"
        )

    def test_without_decomposition_bare_concept_misses_vision(self) -> None:
        """WITHOUT decomposition, encoding the full sentence as one blob
        node binds that blob to vision. But a BARE concept node ("blue mug"
        encoded separately) has NO binding to vision — cross-modal
        retrieval fails for concept-level queries.

        This is the gap decomposition closes.
        """
        encoder, ec, _atl = self._make_encoder(with_decomposer=False)
        hippocampus = _fresh_hippocampus()

        # Encode full sentence as one node (no decomposition)
        sentence_ids = encoder.encode_decomposed("I see a blue mug on the wooden table", "text")
        assert len(sentence_ids) == 1
        sentence_node = sentence_ids[0]

        # Encode bare concept separately
        bare_mug_ids = encoder.encode_decomposed("blue mug", "text")
        bare_mug_node = bare_mug_ids[0]

        # The sentence node and bare concept node should be DIFFERENT
        # (different hash embeddings because different text)
        assert sentence_node != bare_mug_node, "Sentence node and bare concept node should differ without decomposition"

        # Create and bind vision node to sentence node
        vision_mug_emb = [0.1] * 384
        vision_mug_emb[0] = 0.9
        mug_vis = ec.pattern_complete_or_separate(vision_mug_emb, "vision")
        ec.register_substrate_node(mug_vis.node_id, vision_mug_emb, "vision")

        _make_pair_episode(hippocampus, sentence_node, mug_vis.node_id, tick=0)

        # The sentence node retrieves vision_mug (it was directly bound)
        sentence_results = hippocampus.retrieve_cross_modal(sentence_node, target_modality="vision", limit=5)
        sentence_partner_ids = {nid for nid, _ in sentence_results}
        assert mug_vis.node_id in sentence_partner_ids

        # BUT the bare concept node has NO binding to vision_mug
        bare_results = hippocampus.retrieve_cross_modal(bare_mug_node, target_modality="vision", limit=5)
        bare_partner_ids = {nid for nid, _ in bare_results}
        assert mug_vis.node_id not in bare_partner_ids, (
            "Without decomposition, bare concept 'blue mug' should NOT "
            "retrieve vision_mug — it was never co-activated. "
            "This is the gap that decomposition closes."
        )

    def test_multi_scene_mechanism_sweep(self) -> None:
        """Run all 5 fixture scenes through decomposed encoding + binding
        and verify cross-modal retrieval works for each concept.
        """
        encoder, ec, _atl = self._make_encoder(with_decomposer=True)
        hippocampus = _fresh_hippocampus()
        tick = 0

        for scene in SCENES:
            text_node_ids = encoder.encode_decomposed(scene["description"], "text")
            assert len(text_node_ids) == len(scene["concepts"]), (
                f"Scene '{scene['description'][:40]}...' expected "
                f"{len(scene['concepts'])} decomposed nodes, got {len(text_node_ids)}"
            )

            # Create vision nodes for each label
            vision_nodes = []
            for i, label in enumerate(scene["vision_labels"]):
                emb = [0.01] * 384
                emb[(tick + i) % 384] = 0.99  # unique dimension
                vis = ec.pattern_complete_or_separate(emb, "vision")
                ec.register_substrate_node(vis.node_id, emb, "vision")
                vision_nodes.append(vis.node_id)

            # Bind each decomposed text node to its corresponding vision node
            for tn, vn in zip(text_node_ids, vision_nodes):
                _make_pair_episode(hippocampus, tn, vn, tick=tick)
                tick += 100

            # Verify cross-modal retrieval per concept
            for tn, vn, concept in zip(text_node_ids, vision_nodes, scene["concepts"]):
                results = hippocampus.retrieve_cross_modal(tn, target_modality="vision", limit=5)
                partner_ids = {nid for nid, _ in results}
                assert vn in partner_ids, f"Concept '{concept}' did not retrieve its paired vision node"

        print(f"\n  [mechanism] All {len(SCENES)} scenes passed cross-modal retrieval")


# ─────────────────────────────────────────────────────────────────────────
# Validation tests — slow, requires sentence-transformers + spaCy
# ─────────────────────────────────────────────────────────────────────────


# Needs the real encoder WEIGHTS, not just the packages: its autouse fixture
# checks that sentence-transformers and spaCy are INSTALLED, which the D20
# hermeticity work established is not evidence the weights are available. Left
# unmarked, an offline run degrades the encoder and reports the degradation as
# a decomposition result.
@pytest.mark.requires_model_cache
@pytest.mark.slow
class TestDecompositionValidation:
    """Real-embedding validation that decomposition improves cross-modal recall.

    Requires sentence-transformers and spaCy with en_core_web_sm.
    Compares baseline (whole-sentence encoding) vs decomposed encoding
    on concept-level cross-modal recall across 5 naturalistic scenes.
    """

    @pytest.fixture(autouse=True)
    def _require_deps(self) -> None:
        if not _has_sentence_transformers():
            pytest.skip("sentence-transformers not installed")
        if not _has_spacy():
            pytest.skip("spaCy or en_core_web_sm not installed")

    def _make_encoder(self, *, with_decomposer: bool):
        """Build a LinguisticEncoder with real sentence-transformers."""
        from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.70))
        atl = MagicMock()
        atl.activate_substrate_node = MagicMock(side_effect=lambda node_id, **kw: node_id)

        if with_decomposer:
            decomposer = ConceptDecomposer()  # auto-detects spaCy
        else:
            decomposer = None

        encoder = LinguisticEncoder(
            ec=ec,
            atl=atl,
            config=EncoderConfig(model_name="paraphrase-mpnet-base-v2"),
            decomposer=decomposer,
        )
        return encoder, ec

    def _run_arm(self, *, with_decomposition: bool) -> dict:
        """Run one arm of the validation experiment.

        Returns a dict with per-scene recall data and aggregate metrics.
        """
        encoder, ec = self._make_encoder(with_decomposer=with_decomposition)
        hippocampus = _fresh_hippocampus()
        tick = 0

        scene_results = []

        for scene in SCENES:
            # -- Encode text --
            # The encoder handles decomposition internally: when wired
            # with a decomposer it returns multiple nodes, otherwise one.
            text_node_ids = encoder.encode_decomposed(scene["description"], "text")

            # -- Also encode each bare concept (for concept-level queries) --
            concept_node_ids = []
            for concept in scene["concepts"]:
                cids = encoder.encode_decomposed(concept, "text")
                concept_node_ids.append(cids[0])

            # -- Create vision nodes from vision labels --
            # Use real sentence-transformers to embed the label text as a
            # proxy for vision embeddings (same embedding space).
            vision_node_ids = []
            for label in scene["vision_labels"]:
                vis_emb = encoder.embed(label)
                vis_result = ec.pattern_complete_or_separate(vis_emb, "vision")
                if vis_result.is_new:
                    ec.register_substrate_node(vis_result.node_id, vis_emb, "vision")
                vision_node_ids.append(vis_result.node_id)

            # -- Bind text nodes to vision nodes in an episode --
            # Bind ALL text nodes (from sentence encoding) with ALL
            # vision nodes in a shared episode.
            _make_multi_text_vision_episode(hippocampus, text_node_ids, vision_node_ids, tick=tick)
            tick += 100

            # -- Measure concept-level cross-modal recall --
            hits = 0
            total = 0
            per_concept = []

            for concept, concept_nid, expected_vis_nid in zip(scene["concepts"], concept_node_ids, vision_node_ids):
                results = hippocampus.retrieve_cross_modal(concept_nid, target_modality="vision", limit=5)
                retrieved_ids = {nid for nid, _ in results}
                hit = expected_vis_nid in retrieved_ids
                hits += int(hit)
                total += 1
                per_concept.append(
                    {
                        "concept": concept,
                        "concept_node": concept_nid[:12],
                        "expected_vision_node": expected_vis_nid[:12],
                        "hit": hit,
                        "n_results": len(results),
                    }
                )

            scene_recall = hits / total if total > 0 else 0.0
            scene_results.append(
                {
                    "description": scene["description"],
                    "recall": scene_recall,
                    "hits": hits,
                    "total": total,
                    "text_nodes_created": len(text_node_ids),
                    "per_concept": per_concept,
                }
            )

        # Aggregate
        total_hits = sum(s["hits"] for s in scene_results)
        total_queries = sum(s["total"] for s in scene_results)
        aggregate_recall = total_hits / total_queries if total_queries > 0 else 0.0

        return {
            "with_decomposition": with_decomposition,
            "aggregate_recall": aggregate_recall,
            "total_hits": total_hits,
            "total_queries": total_queries,
            "scenes": scene_results,
            "ec_substrate_nodes": ec.substrate_node_count,
        }

    def test_decomposition_improves_cross_modal_recall(self) -> None:
        """The primary validation test.

        Runs both arms (baseline vs decomposed) and asserts:
        1. decomposed_recall > baseline_recall
        2. decomposed_recall >= 0.60
        """
        print("\n")
        print("=" * 70)
        print("Concept Decomposition Validation")
        print("=" * 70)

        # -- Baseline arm (no decomposition) --
        print("\n--- Baseline (no decomposition) ---")
        baseline = self._run_arm(with_decomposition=False)
        print(f"  Aggregate recall: {baseline['aggregate_recall']:.3f}")
        print(f"  Hits: {baseline['total_hits']}/{baseline['total_queries']}")
        print(f"  EC nodes: {baseline['ec_substrate_nodes']}")
        for sr in baseline["scenes"]:
            print(
                f"    {sr['description'][:50]}... recall={sr['recall']:.2f} "
                f"({sr['hits']}/{sr['total']}) "
                f"text_nodes={sr['text_nodes_created']}"
            )

        # -- Decomposed arm --
        print("\n--- Decomposed ---")
        decomposed = self._run_arm(with_decomposition=True)
        print(f"  Aggregate recall: {decomposed['aggregate_recall']:.3f}")
        print(f"  Hits: {decomposed['total_hits']}/{decomposed['total_queries']}")
        print(f"  EC nodes: {decomposed['ec_substrate_nodes']}")
        for sr in decomposed["scenes"]:
            print(
                f"    {sr['description'][:50]}... recall={sr['recall']:.2f} "
                f"({sr['hits']}/{sr['total']}) "
                f"text_nodes={sr['text_nodes_created']}"
            )

        # -- Comparison --
        delta = decomposed["aggregate_recall"] - baseline["aggregate_recall"]
        print("\n--- Result ---")
        print(f"  Baseline recall:    {baseline['aggregate_recall']:.3f}")
        print(f"  Decomposed recall:  {decomposed['aggregate_recall']:.3f}")
        print(f"  Delta:              {delta:+.3f}")

        # -- Save results --
        results_path = Path(__file__).resolve().parents[2] / "docs" / "experiments" / "results"
        results_path.mkdir(parents=True, exist_ok=True)
        output_file = results_path / "concept_decomposition_validation.json"

        report = {
            "test": "concept_decomposition_validation",
            "baseline": baseline,
            "decomposed": decomposed,
            "delta": delta,
            "pass_criteria": {
                "strict_improvement": decomposed["aggregate_recall"] > baseline["aggregate_recall"],
                "minimum_bar": decomposed["aggregate_recall"] >= 0.60,
            },
        }
        output_file.write_text(json.dumps(report, indent=2))
        print(f"  Results saved to: {output_file}")

        # -- Assertions --
        assert decomposed["aggregate_recall"] > baseline["aggregate_recall"], (
            f"Decomposition did not improve recall: "
            f"baseline={baseline['aggregate_recall']:.3f}, "
            f"decomposed={decomposed['aggregate_recall']:.3f}"
        )
        assert decomposed["aggregate_recall"] >= 0.60, (
            f"Decomposed recall {decomposed['aggregate_recall']:.3f} below minimum bar 0.60"
        )

        print("  PASS: decomposition improves cross-modal recall")
