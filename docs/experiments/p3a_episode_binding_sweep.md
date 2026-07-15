# Substrate P3a Stage 2 — Episode binding fixture sweep results

**Plan:** [substrate_p3a_episode_binding.md](../plans/archive/substrate_p3a_episode_binding.md) Stage 2
**Results JSON:** [results/p3a_episode_binding_sweep.json](results/p3a_episode_binding_sweep.json)
**Reproduction runbook:** [protocols/p3a_episode_binding_reproduction.md](protocols/p3a_episode_binding_reproduction.md)
**Date:** 2026-04-14

## TL;DR

On a 10-topic × 17-base-episodes hub+chain synthetic fixture with 10% per-seed episode dropout as a variance source, the Hebbian binding mechanism's **multi-hop retrieval (`spreading_activation`) clears the Stage 2 pass gate with F1 ≈ 0.9955 ± 0.0055 across 10 seeds**, beating the TF-IDF bag-of-concepts baseline by **0.324 absolute F1**. One-hop Hebbian ties TF-IDF within ~0.03 F1 (both near 0.70). The architectural finding: **the mechanism's value over bag-of-words retrieval manifests specifically in transitive / multi-hop graph traversal — a capability TF-IDF structurally cannot replicate** because bag-of-words has no edges to walk.

| Retriever | mean F1 | std F1 | Beats TF-IDF + 2σ? |
|---|---|---|---|
| **Hebbian multi-hop** | **0.9955** | **0.0055** | **✅ YES (margin 0.324)** |
| Hebbian one-hop | 0.6910 | 0.0074 | ≈ parity with TF-IDF |
| TF-IDF baseline | 0.6600 | 0.0058 | — |

**Multi-hop lift over one-hop: 0.3045** (architectural invariant ≥ 0.20 cleared).

## Pass gate (from substrate_p3a_episode_binding.md Stage 2)

1. ✅ `mean_precision > 0.70` — multi-hop gets 1.0
2. ✅ `mean_recall > 0.70` — multi-hop gets 1.0
3. ✅ Hebbian beats TF-IDF by `baseline_mean + 2 × baseline_std` — margin = 0.30 absolute
4. ✅ Persistence round-trip preserves retrieval F1 within ε = 0.01 — exact preservation via P3.5 Stage 1 rebuild-from-episodes

## Fixture shape

Each of 10 topics has 5 **core nodes** — 1 hub + 4 chain nodes arranged linearly:

```
    hub
     ↕ ↕ ↕ ↕
    c1 ↔ c2 ↔ c3 ↔ c4
```

The hub has direct edges to every chain node. Chain adjacency edges connect `c1 ↔ c2 ↔ c3 ↔ c4`. **Each topic has 17 episodes** (170 total across 10 topics):

- **8 hub episodes:** each `{hub, c_i}` pair presented twice. Reinforcement drives edge weight from `hebbian_init = 0.3` to `0.3 + hebbian_delta = 0.4`.
- **6 chain episodes:** each `{c_i, c_{i+1}}` adjacency presented twice. Same reinforcement to 0.4.
- **3 peripheral episodes:** `{hub, peripheral_j}` single-shot. Weight stays at `hebbian_init = 0.3`.

The reinforcement is **architecturally load-bearing, not a tuning knob.** Under single-shot core episodes, multi-hop chain targets (e.g., `plate` reached from cue `prep` at 2 hops) tied exactly with peripherals reached through the hub at the same 2-hop distance, leaving ranking dependent on dict-iteration order. Doubling reinforcement makes core-edge weights (0.4) **strictly** higher than peripheral-edge weights (0.3), so `spreading_activation` ranking is weight-ordered and **order-independent under shuffled ingestion** (regression-guarded).

**Retrieval task.** For each of the 5 core nodes in each topic (50 total probes), retrieve the other 4 core nodes of that topic. Ground truth = 4 targets per probe. Metrics: precision, recall, F1 at `k = len(targets) = 4`.

## Why TF-IDF loses

TF-IDF bag-of-concepts is a strong baseline on co-occurrence tasks but **structurally cannot walk the graph**. The fixture's chain structure puts chain-interior targets 2–3 hops away from chain-head cues, with no direct co-occurrence.

Concrete example: `cue = cooking.prep` (chain head). Ground truth targets = `{stove, saute, simmer, plate}`.

- **TF-IDF** sees `prep`'s episodes: `{hub, prep}×2`, `{prep, saute}×2`. It can score direct co-occurrences (`hub` and `saute`) but has no representation for the transitive chain `prep → saute → simmer → plate`. Returns `[saute, stove]` → recall 2/4.
- **Hebbian multi-hop** walks the binding graph:
  - Hop 1: `stove` (w=0.28), `saute` (w=0.28)
  - Hop 2: `simmer` via `saute` OR hub (max = 0.0784), `plate` via hub (0.0784)
  - Ranking: all 4 chain targets strictly above the 3 peripherals (0.0588).
  - Recall 4/4.

TF-IDF also loses on **hub probes** due to the inverse-document-frequency weighting. When `cue = hub`, chain nodes have moderate IDF (document frequency ~2, IDF ~3.9) while peripherals have maximal IDF (df = 1, IDF = 4.6). Peripherals outrank chain nodes in TF-IDF's top-4 → recall drops to 0.25 on hub probes. See the `breakdown_by_cue_kind_seed0` block in the results JSON for exact numbers.

## Seed-0 cue-kind breakdown

From the results JSON:

| Cue kind | Count | Hebbian multi-hop F1 | Hebbian one-hop F1 | TF-IDF F1 |
|---|---|---|---|---|
| Hub cues | 10 | **1.000** | 1.000 | 0.250 |
| Chain cues | 40 | **1.000** | 0.625 | 0.8125 |

- **Hub probes:** one-hop Hebbian wins via direct hub↔chain edges; TF-IDF is drowned by high-IDF peripherals.
- **Chain probes:** TF-IDF does reasonably well (0.8125) because chain cues see more varied co-occurrence; one-hop Hebbian is limited to 0.625 (can reach hub + 1 adjacent chain node, not the far end); multi-hop gets the full chain via 2–3 hop traversal.

## The one-hop parity finding (architectural)

**One-hop Hebbian (F1 = 0.70) is equivalent to TF-IDF (F1 = 0.70)** on this fixture. This is not a bug — it's the load-bearing finding that shaped Stage 2:

> On a bag-of-words co-occurrence task with no transitive structure, Hebbian one-hop retrieval and TF-IDF bag-of-concepts are algorithmically near-equivalent. Both measure "what appeared with the cue." The Hebbian mechanism's superiority emerges only when retrieval has to traverse structure — multi-hop via `spreading_activation` for P3a, cross-modal for P4, sleep replay for P8. The binding graph is the substrate those extensions depend on, but the binding mechanism itself does not beat bag-of-words at bag-of-words retrieval.

This finding came out of the Stage 2 investigation directly: the first draft of the fixture used a **clique-per-topic** topology (5 core nodes co-occurring in every episode). Both one-hop Hebbian and TF-IDF scored F1 ≈ 1.0 because cliques have no transitive structure — one-hop already reaches everything. The pivot to hub+chain topology was necessary to expose where Hebbian genuinely beats bag-of-words. See the "Design history" section of [tests/substrate/p3a_fixture_gen.py](../../tests/substrate/p3a_fixture_gen.py) for the full rationale.

This framing is now codified as a load-bearing invariant in [substrate_p3a_episode_binding.md](../plans/archive/substrate_p3a_episode_binding.md) so future P4 / P6 / P8 work inherits the correct mental model.

## Deterministic-across-seeds caveat

All three retrievers produce `std_f1 = 0.000` across the 10 seeds. This is **a feature of the fixture, not a missing variance source**. The seed argument controls which peripheral ids are drawn from each topic's peripheral pool, but peripherals only appear in the 3 `{hub, peripheral}` episodes and never interact with the 50 probes (which cue core nodes only). The core topology — hub + chain + adjacency edges — is fully deterministic.

Under the strict reading of `baseline_mean + 2 × std`, zero std means the gate collapses to `baseline_mean`. The Hebbian mechanism still clears it by 0.30 absolute F1, so the pass gate is not weakened by the determinism. A future Stage 3 sweep with real-text fixtures (P3b or later) will introduce genuine seed variance through text → node encoding noise.

## Persistence round-trip

[`Hippocampus._to_dict()`](../../src/maxim/memory/hippocampus_persistence.py) dumps episodes into the reserved `"episodes"` key (P3.5 Stage 1). [`load_state()`](../../src/maxim/memory/hippocampus_persistence.py) restores them and **rebuilds the binding graph from loaded episodes** via `apply_hebbian_on_close`. The rebuild is byte-exact because:

- The episode list round-trips through `EpisodeStore.to_dict` / `load_from_dict` without loss.
- `apply_hebbian_on_close` is deterministic given a fixed episode set + fixed `EpisodeConfig`.
- Pair enumeration uses `itertools.combinations` (unordered), so the replay order doesn't affect final edge weights.

The test asserts retrieval F1 is **byte-exact** post-load (not just within ε = 0.01). See `TestFixturePersistenceRoundTrip::test_binding_graph_rebuilt_identically` in [test_p3a_fixture_validation.py](../../tests/substrate/test_p3a_fixture_validation.py).

## Regression guards shipped with Stage 2

The fixture's tie-free stratification and the multi-hop ranking properties are all guarded:

- `TestStage2PassGate::test_multi_hop_clears_precision_and_recall_gate` — the plan's 0.70 gate.
- `TestStage2PassGate::test_multi_hop_beats_tfidf_by_two_sigma` — the head-to-head.
- `TestOneHopArchitecturalFinding::test_one_hop_does_not_beat_tfidf` — locks in the parity finding with ±0.05 tolerance.
- `TestRankingStability::test_chain_targets_strictly_outrank_peripherals` — no weight ties at the chain/peripheral boundary.
- `TestRankingStability::test_ranking_robust_to_shuffled_ingestion_order` — F1 is byte-exact under shuffled episode ingestion.
- `TestFixturePersistenceRoundTrip::test_retrieval_f1_preserved_within_epsilon` — Plan ε = 0.01.
- `TestFixturePersistenceRoundTrip::test_binding_graph_rebuilt_identically` — edge-for-edge identity under rebuild.
- `TestFixtureShape` — generator sanity.
- `TestEpisodeConfigRetrievalDefaults::test_retrieval_max_depth_override_propagates` — config plumbing test using `max_depth=1` to force degradation.

## What this does NOT prove

- **The mechanism beats TF-IDF on real text.** Stage 2 is a synthetic mechanism test. Real-text fixtures (encoded via `LinguisticEncoder`) are P3b territory and will re-run the head-to-head under encoder noise.
- **The multi-hop retrieval tuning is optimal.** Default `retrieval_decay = 0.7, retrieval_threshold = 0.001, retrieval_max_depth = 5` is calibrated to the hub+chain fixture. Deeper chains or richer topologies may need re-tuning.
- **Cross-modal retrieval works.** P4 is the 1.0-gating cross-modal mug test. P3a validates same-modality binding only.
- **The 1.0-floor under persistence holds at 10,000+ nodes.** P5 stress is the scaling test.

## Next

- Stage 3 of P3a: review round (Executor + Architecture lenses) + PR merge.
- P3b: channel integration on real-text fixtures, reuses the hub+chain structural insight and adds per-channel boundary rules.
- P4: cross-modal mug test, 1.0-gating.
