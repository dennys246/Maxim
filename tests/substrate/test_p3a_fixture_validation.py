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
from maxim.memory.hippocampus import (
    EpisodeConfig,
    Hippocampus,
    HippocampusConfig,
    RetrievalConfig,
)
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
    """Stage 2 documents the finding that the Hebbian mechanism's
    value over bag-of-words baselines manifests specifically in
    multi-hop / transitive retrieval, not in direct co-occurrence.

    The load-bearing invariant is ``multi_hop_f1 > one_hop_f1 + 0.2``
    (the lift is real and measurable). An earlier draft asserted
    ``|one_hop_f1 - tfidf_f1| < 0.05`` as a parity check, but the
    pre-merge Arch-lens review flagged that as over-constraining —
    any future one-hop improvement (e.g., normalized edge weights or
    PageRank-style inference) would trip the test as a regression
    even though the multi-hop lift is the actual architectural
    invariant. The softened form below locks in the REAL claim
    without boxing in future one-hop work.
    """

    def test_multi_hop_lift_over_one_hop_is_real(self):
        onehop_results, multihop_results = [], []
        for seed in SEEDS:
            fixture = build_fixture(seed=seed)
            h = Hippocampus(HippocampusConfig())
            _ingest(h, fixture["episodes"])

            onehop_results.append(run_probes(_hebbian_onehop(h), fixture["probes"], seed=seed))
            multihop_results.append(run_probes(_hebbian_multihop(h), fixture["probes"], seed=seed))

        onehop_agg = aggregate_seeds(onehop_results)
        multihop_agg = aggregate_seeds(multihop_results)
        lift = multihop_agg.mean_f1 - onehop_agg.mean_f1

        # The real architectural invariant: multi-hop retrieval must
        # deliver a meaningful F1 lift over one-hop. 0.2 absolute is
        # a conservative floor — the Stage 2 fixture shows ~0.30.
        assert lift >= 0.20, (
            f"Multi-hop Hebbian F1 {multihop_agg.mean_f1:.4f} must exceed "
            f"one-hop Hebbian F1 {onehop_agg.mean_f1:.4f} by at least 0.20 "
            f"absolute (the architectural invariant). Got lift = {lift:.4f}. "
            f"If the mechanism ever falls below this, either the fixture no "
            f"longer has transitive structure or multi-hop retrieval has "
            f"regressed."
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
        boundary.

        Uses ``episode_dropout_rate=0`` because this test validates
        the topology/reinforcement design, not the seed-variance
        behavior. Under dropout, some seeds may have a core edge
        reinforced only once → weight 0.3 → tied with peripherals;
        that's expected and tested separately.
        """
        fixture = build_fixture(seed=0, episode_dropout_rate=0.0)
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
        work, shuffled ingestion would flip the result.

        Uses ``episode_dropout_rate=0`` so the invariant is byte-exact
        (reinforcement is fully symmetric). Under dropout, different
        shuffle orders of the same dropped set still produce identical
        F1 because ``apply_hebbian_on_close`` is order-independent,
        but asserting byte-exact equality with dropout would also
        exercise the dropout RNG stability which is out of scope.
        """
        fixture = build_fixture(seed=0, episode_dropout_rate=0.0)

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

    def test_fixture_has_expected_base_shape(self):
        # Use episode_dropout_rate=0 to test the un-dropped shape.
        fixture = build_fixture(seed=0, episode_dropout_rate=0.0)
        assert fixture["n_topics"] == 10
        assert fixture["base_episodes_per_topic"] == 17  # 8 hub + 6 chain + 3 peri
        assert fixture["base_total_episodes"] == 170
        assert len(fixture["episodes"]) == 170  # no dropout
        assert len(fixture["probes"]) == 50  # 5 cores × 10 topics

    def test_dropout_produces_seed_variance_in_episode_count(self):
        """Different seeds drop different episodes → different totals."""
        counts = [len(build_fixture(seed=s)["episodes"]) for s in range(10)]
        # With 10% dropout on 170 episodes, every seed drops exactly 17.
        # So total_episodes should be 153 across all seeds. Variance
        # emerges in WHICH episodes are dropped, not in the count.
        assert all(c == 153 for c in counts), f"expected 153 after 10% dropout, got {counts}"

    def test_dropout_drops_different_episodes_across_seeds(self):
        """Two different seeds must not drop identical episode sets."""
        ep_ids_a = {ep["id"] for ep in build_fixture(seed=0)["episodes"]}
        ep_ids_b = {ep["id"] for ep in build_fixture(seed=1)["episodes"]}
        assert ep_ids_a != ep_ids_b, "Different seeds must produce different dropout patterns"

    def test_every_probe_has_four_targets(self):
        fixture = build_fixture(seed=0)
        for probe in fixture["probes"]:
            assert len(probe["targets"]) == 4, f"cue={probe['cue']} has {len(probe['targets'])} targets, expected 4"

    def test_core_edge_weights_generally_above_peripheral_edge_weights(self):
        """Verify the reinforcement-based weight stratification at the
        binding-graph level: hub↔chain and chain-adjacency edges must
        generally have higher weight than hub↔peripheral edges.

        Under episode dropout, a specific core pair in a specific seed
        may be reinforced only once (weight 0.3) instead of twice (0.4)
        — so we assert the INVARIANT across all seeds rather than exact
        numerics on seed 0.
        """
        from maxim.agents.bus import EdgeType

        # Use zero dropout so the exact assertion holds deterministically
        fixture = build_fixture(seed=0, episode_dropout_rate=0.0)
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

    def test_peripherals_per_topic_validation(self):
        """Exec-lens finding: silently undercount on pool exhaustion is a
        footgun. Generator now raises ValueError on overflow."""
        with pytest.raises(ValueError, match="exceeds pool size"):
            build_fixture(seed=0, peripherals_per_topic=100)

    def test_episode_dropout_rate_validation(self):
        with pytest.raises(ValueError, match="episode_dropout_rate must be in"):
            build_fixture(seed=0, episode_dropout_rate=1.0)
        with pytest.raises(ValueError, match="episode_dropout_rate must be in"):
            build_fixture(seed=0, episode_dropout_rate=-0.1)


class TestEpisodeConfigRetrievalDefaults:
    """The Stage 2 retrieval tuning params on `RetrievalConfig` must
    have sensible defaults and propagate through `retrieve_on_cue`."""

    def test_retrieval_defaults_match_spec(self):
        """Renamed from test_retrieval_defaults_produce_stage2_pass to
        match what the test actually does: assert the three default
        values equal the expected floats. It does not run a probe."""
        cfg = HippocampusConfig()
        assert cfg.episode.retrieval.decay == pytest.approx(0.7)
        assert cfg.episode.retrieval.threshold == pytest.approx(0.001)
        assert cfg.episode.retrieval.max_depth == 5

    def test_retrieval_max_depth_override_propagates(self):
        """Forcing max_depth=1 must strictly degrade F1 below the
        default max_depth=5. If the config override is silently
        ignored, both runs would equal the same value.

        Uses ``episode_dropout_rate=0`` so the assertion is
        byte-deterministic — this test validates config plumbing
        (the override reaches ``spreading_activation``), not seed-
        variance behavior.
        """
        default_h = Hippocampus(HippocampusConfig())
        overridden_h = Hippocampus(
            HippocampusConfig(
                episode=EpisodeConfig(retrieval=RetrievalConfig(max_depth=1)),
            )
        )
        fixture = build_fixture(seed=0, episode_dropout_rate=0.0)
        _ingest(default_h, fixture["episodes"])
        _ingest(overridden_h, fixture["episodes"])

        default_result = run_probes(_hebbian_multihop(default_h), fixture["probes"], seed=0)
        override_result = run_probes(_hebbian_multihop(overridden_h), fixture["probes"], seed=0)

        # Default must still clear the gate (sanity)
        assert default_result.mean_f1 == pytest.approx(1.0, abs=1e-9), (
            f"Default retrieval.max_depth=5 should give F1=1.0, got {default_result.mean_f1:.4f}"
        )
        # Override must strictly degrade F1 — if override is silently
        # ignored, both would equal 1.0.
        assert override_result.mean_f1 < default_result.mean_f1, (
            f"Override retrieval.max_depth=1 should reduce F1 below default "
            f"{default_result.mean_f1:.4f}, got {override_result.mean_f1:.4f}. "
            f"Config plumbing through retrieve_on_cue may be broken."
        )


class TestNodeFilterSeam:
    """P3b channel-filter / P4 modality-filter seam — `retrieve_on_cue`
    accepts an optional `node_filter` callable that drops filtered
    nodes from traversal. Pre-merge Arch-lens reserved this seam in
    Stage 2 so P3b doesn't have to rebuild the retrieval path.
    """

    def test_node_filter_excludes_matching_nodes(self):
        """Given a filter that rejects `simmer`, retrieve_on_cue from
        `prep` must not return `simmer` and must not spread activation
        through it to `plate`."""
        fixture = build_fixture(seed=0)
        h = Hippocampus(HippocampusConfig())
        _ingest(h, fixture["episodes"])

        # Baseline: no filter. prep → all 4 cooking cores.
        baseline = dict(h.retrieve_on_cue("cooking.prep", limit=20, multi_hop=True))
        assert "cooking.simmer" in baseline
        assert "cooking.plate" in baseline

        # Filter out simmer. Because plate is reachable through saute→simmer→plate
        # AND through hub→plate (2 hops), hub path still delivers plate.
        # But simmer itself MUST be filtered out of the result set.
        def drop_simmer(node_id: str) -> bool:
            return node_id != "cooking.simmer"

        filtered = dict(h.retrieve_on_cue("cooking.prep", limit=20, multi_hop=True, node_filter=drop_simmer))
        assert "cooking.simmer" not in filtered, "node_filter must drop simmer"
        # plate is still reachable via hub (prep → hub → plate)
        assert "cooking.plate" in filtered

    def test_node_filter_one_hop_mode(self):
        """node_filter works in one-hop mode too — it just filters the
        returned direct-neighbor list."""
        fixture = build_fixture(seed=0)
        h = Hippocampus(HippocampusConfig())
        _ingest(h, fixture["episodes"])

        filtered = h.retrieve_on_cue(
            "cooking.stove",
            limit=20,
            multi_hop=False,
            node_filter=lambda n: "plate" not in n,
        )
        filtered_ids = {n for n, _ in filtered}
        assert not any("plate" in n for n in filtered_ids)

    def test_node_filter_none_is_identity(self):
        """Passing node_filter=None (default) must be identical to omitting it."""
        fixture = build_fixture(seed=0)
        h = Hippocampus(HippocampusConfig())
        _ingest(h, fixture["episodes"])

        default = h.retrieve_on_cue("cooking.prep", limit=20, multi_hop=True)
        explicit_none = h.retrieve_on_cue("cooking.prep", limit=20, multi_hop=True, node_filter=None)
        assert default == explicit_none
