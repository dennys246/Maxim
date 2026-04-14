"""P3a Stage 2 — fixture-based validation test.

Runs the Hebbian binding mechanism head-to-head against the TF-IDF
bag-of-concepts baseline across ≥10 seeds on the hub+chain synthetic
episodes fixture. Asserts:

1. **Hebbian multi-hop clears the plan's precision/recall > 0.70 gate.**
2. **Hebbian multi-hop beats TF-IDF by baseline_mean + 2×baseline_std.**
   This is the plan's primary Stage 2 pass gate — the Hebbian
   mechanism's structural advantage over bag-of-words retrieval must
   manifest as a measurable F1 margin on a fixture with transitive
   structure.
3. **Persistence round-trip preserves retrieval F1 within ε=0.01.**
   A Hippocampus whose episodes survive a ``dump()``/``load_state()``
   cycle must produce numerically identical retrieval results on the
   fixture (via the P3.5 Stage 1 rebuild-from-episodes path).
4. **Ranking is robust to episode ingestion order.** Regression guard
   for the tie-fragility the Stage 2 fixture was explicitly
   reinforced to eliminate — chain targets must strictly outrank
   peripherals by edge weight, not by dict-iteration tiebreaks.
5. **Shuffle guard per feedback_shuffle_fixture_ordering.md** — the
   aggregate metrics must be stable under shuffled episode order.
6. **One-hop Hebbian retrieval is equivalent to TF-IDF on this
   fixture** (both cap at F1 ≈ 0.70). Documents the architectural
   finding that the Hebbian mechanism's value over bag-of-words
   baselines manifests in multi-hop / transitive retrieval, not in
   direct co-occurrence.
"""

from __future__ import annotations

import random

import pytest

from maxim.memory.episode import CaptureEvent
from maxim.memory.hippocampus import EpisodeConfig, Hippocampus, HippocampusConfig
from tests.substrate.p3a_fixture_gen import build_fixture
from tests.substrate.p3a_metrics import (
    aggregate_seeds,
    compare_to_baseline,
    run_probes,
)
from tests.substrate.tfidf_baseline import TfidfBaseline

SEEDS = list(range(10))
BOUNDARY_TICK_STEP = 1000  # > default EpisodeConfig.boundary_tick_gap=50


def _ingest(h: Hippocampus, episodes: list[dict]) -> None:
    """Feed fixture episodes into a Hippocampus, forcing boundary
    closure between each episode via a large tick gap."""
    tick = 0
    for ep in episodes:
        h.observe_episode_event(
            CaptureEvent(
                tick=tick,
                channel=ep["channel"],
                activated_nodes=tuple(ep["activated_nodes"]),
            )
        )
        tick += BOUNDARY_TICK_STEP
    h.finalize_pending_episode()


def _hebbian_multihop(h: Hippocampus):
    return lambda cue, k: h.retrieve_on_cue(cue, limit=k, multi_hop=True)


def _hebbian_onehop(h: Hippocampus):
    return lambda cue, k: h.retrieve_on_cue(cue, limit=k, multi_hop=False)


# ─────────────────────────────────────────────────────────────────────────
# Head-to-head validation — the Stage 2 pass gate
# ─────────────────────────────────────────────────────────────────────────


class TestStage2PassGate:
    """The plan's Stage 2 pass gate: Hebbian multi-hop must clear 0.70
    precision + recall AND beat TF-IDF by ``baseline + 2σ``."""

    def test_multi_hop_clears_precision_and_recall_gate(self):
        per_seed = []
        for seed in SEEDS:
            fixture = build_fixture(seed=seed)
            h = Hippocampus(HippocampusConfig())
            _ingest(h, fixture["episodes"])
            per_seed.append(run_probes(_hebbian_multihop(h), fixture["probes"], seed=seed))

        agg = aggregate_seeds(per_seed)
        assert agg.mean_precision > 0.70, (
            f"Hebbian multi-hop mean precision {agg.mean_precision:.4f} must exceed 0.70 "
            f"(plan Stage 2 gate). Per-seed: {[r.mean_precision for r in per_seed]}"
        )
        assert agg.mean_recall > 0.70, f"Hebbian multi-hop mean recall {agg.mean_recall:.4f} must exceed 0.70"

    def test_multi_hop_beats_tfidf_by_two_sigma(self):
        hebbian_results, tfidf_results = [], []
        for seed in SEEDS:
            fixture = build_fixture(seed=seed)
            h = Hippocampus(HippocampusConfig())
            _ingest(h, fixture["episodes"])
            tfidf = TfidfBaseline.from_episodes(fixture["episodes"])

            hebbian_results.append(run_probes(_hebbian_multihop(h), fixture["probes"], seed=seed))
            tfidf_results.append(run_probes(tfidf.retrieve, fixture["probes"], seed=seed))

        hebbian_agg = aggregate_seeds(hebbian_results)
        tfidf_agg = aggregate_seeds(tfidf_results)
        cmp = compare_to_baseline(hebbian_agg, tfidf_agg)

        assert cmp.beats_baseline, (
            f"Hebbian multi-hop F1 {cmp.mechanism_mean_f1:.4f} must beat "
            f"TF-IDF F1 {cmp.baseline_mean_f1:.4f} + 2×{cmp.baseline_std_f1:.4f} "
            f"= {cmp.baseline_mean_f1 + 2 * cmp.baseline_std_f1:.4f}. "
            f"Margin: {cmp.margin:.4f}"
        )
        # Stronger: require at least a 0.10 absolute margin for headroom
        # under future fixture or retrieval-param changes.
        assert cmp.margin >= 0.10, (
            f"Hebbian multi-hop margin over TF-IDF must be ≥ 0.10 absolute F1 for headroom; got {cmp.margin:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────
# One-hop parity — architectural finding
# ─────────────────────────────────────────────────────────────────────────


class TestOneHopArchitecturalFinding:
    """Stage 2 documents the finding that one-hop Hebbian retrieval is
    roughly equivalent to TF-IDF on bag-of-words tasks, and the
    mechanism's real value manifests only in multi-hop / transitive
    retrieval. These tests lock in that finding so future refactors
    don't silently invalidate the architectural claim."""

    def test_one_hop_does_not_beat_tfidf(self):
        onehop_results, tfidf_results = [], []
        for seed in SEEDS:
            fixture = build_fixture(seed=seed)
            h = Hippocampus(HippocampusConfig())
            _ingest(h, fixture["episodes"])
            tfidf = TfidfBaseline.from_episodes(fixture["episodes"])

            onehop_results.append(run_probes(_hebbian_onehop(h), fixture["probes"], seed=seed))
            tfidf_results.append(run_probes(tfidf.retrieve, fixture["probes"], seed=seed))

        onehop_agg = aggregate_seeds(onehop_results)
        tfidf_agg = aggregate_seeds(tfidf_results)

        # Architectural finding: one-hop Hebbian and TF-IDF are near-
        # equivalent on bag-of-words. Both sit at ~0.70 F1 on the
        # hub+chain fixture (one-hop reaches hub neighbors but not
        # chain interior; TF-IDF reaches direct co-occurrences).
        # Allow ±0.05 tolerance — anything wider would be a silent
        # regression that deserves investigation.
        diff = onehop_agg.mean_f1 - tfidf_agg.mean_f1
        assert abs(diff) < 0.05, (
            f"One-hop Hebbian F1 {onehop_agg.mean_f1:.4f} vs TF-IDF F1 "
            f"{tfidf_agg.mean_f1:.4f} — difference {diff:.4f} exceeds "
            f"the ±0.05 parity tolerance. On the Stage 2 fixture these "
            f"should be architecturally equivalent; a wider gap "
            f"indicates the fixture or retrieval shape has drifted."
        )


# ─────────────────────────────────────────────────────────────────────────
# Ranking-stability regression guards
# ─────────────────────────────────────────────────────────────────────────


class TestRankingStability:
    """Stage 2 fixture was explicitly reinforced (core episodes ×2) so
    chain targets strictly outrank peripherals by edge weight. These
    tests guard against a future refactor that could re-introduce
    weight ties and silently degrade retrieval."""

    def test_chain_targets_strictly_outrank_peripherals(self):
        """For every probe, every chain-target node must have a
        strictly higher multi-hop activation than every peripheral
        node retrieved. No weight ties allowed at the chain/peripheral
        boundary."""
        fixture = build_fixture(seed=0)
        h = Hippocampus(HippocampusConfig())
        _ingest(h, fixture["episodes"])

        # Collect every peripheral node across the fixture. Peripheral
        # episodes in the Stage 2 fixture always have shape {hub, peri},
        # with peri = activated_nodes[1] per the generator contract.
        peripheral_ids: set[str] = set()
        for ep in fixture["episodes"]:
            if ep["kind"] == "peripheral":
                peripheral_ids.add(ep["activated_nodes"][1])

        for probe in fixture["probes"]:
            retrieved = h.retrieve_on_cue(probe["cue"], limit=20, multi_hop=True)
            targets = set(probe["targets"])
            target_weights = [w for node, w in retrieved if node in targets]
            peri_weights = [w for node, w in retrieved if node in peripheral_ids]
            if not peri_weights:
                # No peripherals reached — trivially strict
                continue
            min_target_weight = min(target_weights) if target_weights else 0.0
            max_peri_weight = max(peri_weights)
            assert min_target_weight > max_peri_weight, (
                f"cue={probe['cue']}: min target weight {min_target_weight:.4f} "
                f"must be strictly greater than max peripheral weight "
                f"{max_peri_weight:.4f}. Ranking is fragile; check fixture "
                f"reinforcement is still doubling core episodes."
            )

    def test_ranking_robust_to_shuffled_ingestion_order(self):
        """Shuffle episode ingestion order and verify F1 stays at the
        same level. If a tie-break heuristic were quietly doing the
        work, shuffled ingestion would flip the result."""
        fixture = build_fixture(seed=0)

        f1s = []
        for shuffle_seed in range(5):
            shuffled = list(fixture["episodes"])
            random.Random(shuffle_seed).shuffle(shuffled)
            h = Hippocampus(HippocampusConfig())
            _ingest(h, shuffled)
            result = run_probes(_hebbian_multihop(h), fixture["probes"], seed=0)
            f1s.append(result.mean_f1)

        # All shuffles must produce identical F1 — weight stratification
        # makes the ranking fully order-independent.
        assert max(f1s) - min(f1s) < 1e-9, (
            f"F1 under shuffled ingestion: {f1s}. Stratification should "
            f"produce zero variance; non-zero variance indicates the "
            f"ranking is still depending on tiebreaks."
        )
        # And the stable value is the expected 1.0
        assert f1s[0] == pytest.approx(1.0, abs=1e-9), f"Shuffled-ingestion F1 = {f1s[0]:.4f}, expected 1.0."


# ─────────────────────────────────────────────────────────────────────────
# Persistence round-trip — depends on P3.5 Stage 1 rebuild-from-episodes
# ─────────────────────────────────────────────────────────────────────────


class TestFixturePersistenceRoundTrip:
    """The P3.5 Stage 1 ``Hippocampus.load_state`` rebuild-from-episodes
    path must preserve retrieval F1 within ε=0.01 across a dump →
    load cycle on the full fixture."""

    def test_retrieval_f1_preserved_within_epsilon(self):
        fixture = build_fixture(seed=0)

        h1 = Hippocampus(HippocampusConfig())
        _ingest(h1, fixture["episodes"])
        pre_dump = run_probes(_hebbian_multihop(h1), fixture["probes"], seed=0)

        dumped = h1.dump()
        h2 = Hippocampus(HippocampusConfig())
        h2.load_state(dumped)
        post_load = run_probes(_hebbian_multihop(h2), fixture["probes"], seed=0)

        assert abs(post_load.mean_f1 - pre_dump.mean_f1) < 0.01, (
            f"Persistence round-trip F1 drift: pre-dump {pre_dump.mean_f1:.4f} "
            f"vs post-load {post_load.mean_f1:.4f}. Plan ε=0.01."
        )
        # Stronger: exact preservation, since the rebuild-from-episodes
        # path is fully deterministic under the Stage 2 tuning params.
        assert post_load.mean_f1 == pytest.approx(pre_dump.mean_f1, abs=1e-9)

    def test_binding_graph_rebuilt_identically(self):
        """Edge-for-edge identity under rebuild."""
        fixture = build_fixture(seed=0)

        h1 = Hippocampus(HippocampusConfig())
        _ingest(h1, fixture["episodes"])
        dumped = h1.dump()

        h2 = Hippocampus(HippocampusConfig())
        h2.load_state(dumped)

        # Verify every probe's retrieval is identical
        for probe in fixture["probes"]:
            cue = probe["cue"]
            pre = h1.retrieve_on_cue(cue, limit=20, multi_hop=True)
            post = h2.retrieve_on_cue(cue, limit=20, multi_hop=True)
            assert pre == post, f"Rebuild drift at cue={cue}: pre={pre} post={post}"


# ─────────────────────────────────────────────────────────────────────────
# Fixture sanity checks
# ─────────────────────────────────────────────────────────────────────────


class TestFixtureShape:
    """Guard against accidental changes to the fixture generator that
    would invalidate the pass-gate assumptions."""

    def test_fixture_has_expected_shape(self):
        fixture = build_fixture(seed=0)
        assert fixture["n_topics"] == 10
        assert fixture["episodes_per_topic"] == 17  # 8 hub + 6 chain + 3 peri
        assert len(fixture["episodes"]) == 170
        assert len(fixture["probes"]) == 50  # 5 cores × 10 topics

    def test_every_probe_has_four_targets(self):
        fixture = build_fixture(seed=0)
        for probe in fixture["probes"]:
            assert len(probe["targets"]) == 4, f"cue={probe['cue']} has {len(probe['targets'])} targets, expected 4"

    def test_core_edge_weights_strictly_above_peripheral_edge_weights(self):
        """Verify the reinforcement-based weight stratification at the
        binding-graph level: hub↔chain and chain-adjacency edges must
        have higher weight than hub↔peripheral edges."""
        from maxim.agents.bus import EdgeType

        fixture = build_fixture(seed=0)
        h = Hippocampus(HippocampusConfig())
        _ingest(h, fixture["episodes"])

        # cooking.stove (hub) ↔ cooking.prep (chain): reinforced twice → 0.4
        core_edge = h._binding_graph.find_edge("cooking.stove", "cooking.prep", EdgeType.ASSOCIATES)
        assert core_edge is not None
        assert core_edge.weight == pytest.approx(0.4), (
            f"Core edge weight {core_edge.weight} should be 0.4 (0.3 init + 0.1 delta)"
        )

        # cooking.stove (hub) ↔ cooking.tomato (peripheral): single-shot → 0.3
        peri_edge = h._binding_graph.find_edge("cooking.stove", "cooking.tomato", EdgeType.ASSOCIATES)
        assert peri_edge is not None
        assert peri_edge.weight == pytest.approx(0.3), (
            f"Peripheral edge weight {peri_edge.weight} should be 0.3 (hebbian_init only)"
        )


class TestEpisodeConfigRetrievalDefaults:
    """The Stage 2 retrieval tuning params on EpisodeConfig must have
    sensible defaults that produce the expected pass-gate results."""

    def test_retrieval_defaults_produce_stage2_pass(self):
        cfg = HippocampusConfig()
        assert cfg.episode.retrieval_decay == pytest.approx(0.7)
        assert cfg.episode.retrieval_threshold == pytest.approx(0.001)
        assert cfg.episode.retrieval_max_depth == 5

    def test_retrieval_max_depth_override_propagates(self):
        """Test the override path that Stage 2 sweeps will use.

        Forcing max_depth=1 restricts spreading_activation to a single
        hop, which must degrade retrieval to match one-hop Hebbian
        (F1 ≈ 0.70) on chain-head probes. If the config override is
        ignored, F1 would stay at 1.0 because the default max_depth=5
        would still allow deep traversal.
        """
        default_h = Hippocampus(HippocampusConfig())
        overridden_h = Hippocampus(HippocampusConfig(episode=EpisodeConfig(retrieval_max_depth=1)))
        fixture = build_fixture(seed=0)
        _ingest(default_h, fixture["episodes"])
        _ingest(overridden_h, fixture["episodes"])

        default_result = run_probes(_hebbian_multihop(default_h), fixture["probes"], seed=0)
        override_result = run_probes(_hebbian_multihop(overridden_h), fixture["probes"], seed=0)

        # Default must still clear the gate (sanity)
        assert default_result.mean_f1 == pytest.approx(1.0, abs=1e-9), (
            f"Default retrieval_max_depth=5 should give F1=1.0, got {default_result.mean_f1:.4f}"
        )
        # Override must strictly degrade F1 — if override is silently
        # ignored, both would equal 1.0.
        assert override_result.mean_f1 < default_result.mean_f1, (
            f"Override retrieval_max_depth=1 should reduce F1 below default "
            f"{default_result.mean_f1:.4f}, got {override_result.mean_f1:.4f}. "
            f"Config plumbing through retrieve_on_cue may be broken."
        )
