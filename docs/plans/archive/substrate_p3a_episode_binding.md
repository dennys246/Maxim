# Substrate P3a — Episode binding produces retrieval on partial cue

**Status:** ✅ COMPLETE — Stages 1 + 2 SHIPPED (PRs #109, #112; 2026-04-14). Stage 3 (real-data 10-seed sweep) was merged into Stage 2's head-to-head deliverable; there is no separate Stage 3 ship. **Final results: Hebbian multi-hop F1 = 0.9955 ± 0.0055 vs TF-IDF 0.6600 ± 0.0058, margin 0.324, lift over one-hop 0.3045** ([../experiments/p3a_episode_binding_sweep.md](../experiments/p3a_episode_binding_sweep.md)). The `retrieve_on_cue(multi_hop=True, node_filter=...)` seam reserved by Stage 2 is consumed by P3b Stage 1 and will be consumed by P4 Stage 1.
**Scope:** ~400 LOC + ~100 metric extractor across 3 stages
**Target version:** 0.3-target
**Gates:** First of the four plans (P3a + P3b + P3.5 + P4) that together close 0.3-target.
**Depends on:** substrate_recognition ✅, P3.5 Stage 1 shell (for the `Hippocampus._to_dict()` extraction that the round-trip test uses — P3.5 Stage 1 MUST land in the same branch BEFORE P3a Stage 1)
**Blocks:** P3b (channel integration reuses episode boundary machinery), P4 cross-modal binding (depends on episode-scoped binding working for same-modality first), B4 replanning (needs episode retrieval of prior attempts)
**Parent:** [substrate_binding_persistence.md](substrate_binding_persistence.md)
**Related:** [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md), [substrate_p3_5_persistence_snapshot.md](substrate_p3_5_persistence_snapshot.md)

## Goal

Ship the Hebbian episode-binding mechanism. An episode is a multi-event time window during which co-activated substrate nodes form durable associative links; presenting a partial cue from the episode retrieves the others with a margin greater than a TF-IDF bag-of-concepts baseline. This is the first substrate mechanism that produces **retrieval of things that were never directly queried**, which is the substrate's load-bearing claim for cross-modal binding in P4.

## Hypothesis (falsifiable)

Nodes co-occurring in the same Hippocampus episode form durable links through Hebbian updates on a binding graph owned by Hippocampus. Presenting a single node from a prior episode as a cue retrieves the other nodes from that episode by a margin greater than a TF-IDF bag-of-concepts baseline computed on the same episode fixtures. The margin is stable across ≥10 seeds at `precision > 0.70` and `recall > 0.70`.

## Dependencies — scaffolding audit

The [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md) audit established that P3a reuses the existing `DependencyGraph` utility class. **Round 1 pre-merge review surfaced a critical Architecture-lens finding that forced an architectural pivot from "Hebbian on ATL.graph" to "Hebbian on a separate binding graph owned by Hippocampus"** — see "Binding graph ownership" below. The `DependencyGraph` utility reuse is preserved; what changes is the instance that owns the Hebbian edges.

**Existing surfaces (verified 2026-04-14):**

| Surface | File:line | Purpose in P3a |
|---|---|---|
| `DependencyGraph.add_bidirectional(a, b, edge_type=ASSOCIATES, weight=1.0)` | [agents/bus.py](../../src/maxim/agents/bus.py) | Create new symmetric Hebbian edge pair. |
| `DependencyGraph.update_edge(src, tgt, ASSOCIATES, weight=new)` | agents/bus.py | Update one direction of an existing edge. One call per direction needed. |
| `DependencyGraph.find_edge(src, tgt, ASSOCIATES) -> Edge \| None` | agents/bus.py | Read current weight for Hebbian delta. |
| `DependencyGraph.get_associated(node_id, edge_types={ASSOCIATES}) -> list[(str, float)]` | agents/bus.py | One-hop retrieval — candidates + weights. |
| `DependencyGraph.spreading_activation(source_ids, ...)` | agents/bus.py | Multi-hop retrieval (Stage 2 fallback; decision rule below). |
| `EdgeType.ASSOCIATES` | agents/bus.py | No new edge type needed. |
| `DependencyGraph.add_edge` (underlying) | agents/bus.py | **⚠ no dedupe** — appends a new Edge to `_outgoing[src]` unconditionally. Double-calling `add_edge` silently creates two parallel edges. P3a's logic must `find_edge` FIRST; regression guard test below. |
| `Hippocampus.capture(memory)` / `capture_from_loop(...)` | [memory/hippocampus.py](../../src/maxim/memory/hippocampus.py) | Single-event capture. P3a wraps this in a boundary detector. |
| `Hippocampus._to_dict()` / `_load_from_dict(data)` (P3.5 Stage 1) | [memory/hippocampus_persistence.py:32](../../src/maxim/memory/hippocampus_persistence.py#L32) | Round-trip surface for the episode store. P3.5 Stage 1 reserves the `"episodes"` key; P3a populates it. |
| `BioSystemSnapshot` Protocol (P3.5 Stage 1) | [memory/snapshot.py](../../src/maxim/memory/snapshot.py) | In-place `load(state: dict) -> None` semantics. `EpisodeStore` piggybacks via Hippocampus. |

**New surfaces (what P3a actually builds):**

| Surface | Scope | Stage |
|---|---|---|
| `Episode` dataclass (`memory/episode.py`) | ~50 LOC | 1 |
| `EpisodeStore` class (`memory/episode.py`, **standalone class**, held on `Hippocampus` as `self._episode_store`) | ~100 LOC | 1 |
| `BoundaryRule` type + three default rule implementations (`memory/episode.py` — rule-list shape) | ~80 LOC | 1 |
| `Hippocampus._binding_graph: DependencyGraph` (new field) + Hebbian update rule on episode close | ~70 LOC | 1 |
| Partial-cue retrieval path (`Hippocampus.retrieve_on_cue`) | ~60 LOC | 1 |
| `EpisodeConfig` dataclass on `HippocampusConfig` (`memory/hippocampus.py` config section) | ~25 LOC | 1 |
| TF-IDF gate baseline (`tests/substrate/tfidf_baseline.py`) | ~100 LOC | 2 |
| Metric extractor (`tests/substrate/p3a_metrics.py`) | ~100 LOC | 1 shell + 2 full |
| Synthetic fixture (`scenarios/substrate/synthetic_episodes.yaml`) | YAML + loader | 2 |

### Binding graph ownership — Round 1 Arch-lens critical finding #6

**Decision:** Hebbian edges live on `Hippocampus._binding_graph: DependencyGraph` (a new field), NOT on `ATL.graph`.

**Why the pivot.** The original plan put Hebbian edges on `ATL.graph` with `EdgeType.ASSOCIATES`. Round 1 Architecture lens flagged that ATL runs concept eviction + compression in the background (`CompressedSemantic` replaces individual `SemanticMemory` records when staleness rules fire), which would silently destroy Hebbian edges whose endpoints got compressed. The Arch lens offered three resolutions:

1. Block ATL compression on Hebbian-edge endpoints (couples ATL eviction to P3a; intrusive).
2. Migrate Hebbian edges onto compressed nodes via a new ATL compression hook (new coupling surface).
3. **Put Hebbian edges on a separate graph.** Decouples from ATL lifecycle entirely.

Option 3 wins because:
- It still reuses `DependencyGraph` + `EdgeType.ASSOCIATES` — the split-proposal audit's intent ("reuse existing infrastructure, no new edge type") is preserved. What changes is the *instance* of DependencyGraph holding the edges, not the class.
- It decouples the substrate binding layer from ATL's concept-relationship semantics. `ATL.graph` stays the concept topology (REQUIRES/ENABLES/CAUSES/etc. domain edges); `Hippocampus._binding_graph` holds co-activation history. Two distinct kinds of graph = two distinct architectural layers. This is arguably the *right* shape even without the compression concern.
- It avoids the CLAUDE.md "no band-aid fixes" rule — deferring the compression question to Stage 3 review (as the original plan did) is exactly the band-aid class the rule forbids.

**Consequence for ATL wiring.** The `Hippocampus._atl` optional reference is no longer load-bearing for Stage 1 Hebbian mechanics (because Hebbian writes go to `self._binding_graph`, not `atl.graph`). The `is not None` vs truthy regression guard is preserved but repointed at a DIFFERENT check in Stage 1 — validating that `Hippocampus.finalize_pending_episode()` runs cleanly when `self._atl is None` AND when `self._atl is not None but len(self._atl) == 0`. The regression target is the general "never use truthy checks on bio-systems with `__len__`" discipline, not an ATL-specific Hebbian dependency. P3a Stage 2+ may optionally consult `atl.graph` as a secondary retrieval source (e.g., for spreading_activation across concept-topology edges after the binding-graph primary hop); that's a Stage 2 decision.

**What happens on ATL compression with the pivot.** Nothing — the binding graph's edges are keyed by stable substrate node IDs (encoded by `LinguisticEncoder` + stored in `PerceptTraceBuffer`). ATL compression replaces concept records but IDs survive (per P1+P2 invariants). Binding edges remain valid. Any P4+ experiment that needs to cross-reference ATL-compressed concepts goes through a lookup, not through graph state.

**Naming clarification.** There is an existing `EpisodicMemory` type at [memory/types.py:472](../../src/maxim/memory/types.py#L472), but it represents a single loop cycle (perception → decide → act → outcome), not a multi-event time window. These are orthogonal concepts. P3a's `Episode` is a new type, not an extension of `EpisodicMemory`. Simulation "episodes" (campaign runs) in `simulation/` are a third orthogonal concept in a different domain. **Alternative location considered + rejected:** `memory/types.py` already aggregates many memory types; CLAUDE.md's "many small files" convention + the standalone-`EpisodeStore` class fit better in a new `memory/episode.py` file co-locating all P3a-owned types. A reviewer re-opening this in Round 2 should close it by referencing this note.

## Stages

### Stage 1 — mechanism tests on synthetic geometry

**What's built:**

1. **`src/maxim/memory/episode.py`** (new, ~50 LOC):
   ```python
   @dataclass(frozen=True)
   class Episode:
       id: str
       start_tick: int
       end_tick: int
       channel: str
       sender_ids: tuple[str, ...]
       thread_id: str | None
       activated_nodes: tuple[str, ...]
       reward_events: tuple[tuple[int, float], ...]  # (tick, delta)
       scn_tag: str | None
   ```
   Plus `Episode.to_dict()` / `Episode.from_dict()` for P3.5 round-trip. `frozen=True` because episodes are immutable once closed; mutation would race with retrieval.

2. **`EpisodeStore` (standalone class in `memory/episode.py`, ~100 LOC)** — Round 1 Arch important finding #1:
   ```python
   class EpisodeStore:
       """Owns episodes and the node→episode inverted index.

       Lives as a field on Hippocampus (hippocampus._episode_store) rather
       than inlined so P3b channel rules and P5 bounded-storage can extend
       this class without touching Hippocampus itself.
       """

       def __init__(self) -> None:
           self._episodes: dict[str, Episode] = {}
           self._by_node: dict[str, set[str]] = {}
           self._lock = threading.RLock()

       def add(self, episode: Episode) -> None: ...
       def get(self, id: str) -> Episode | None: ...
       def episodes_containing(self, node_id: str) -> list[Episode]: ...
       def to_dict(self) -> dict[str, Any]: ...
       def load_from_dict(self, data: dict[str, Any]) -> None: ...  # in-place
   ```
   Held on `Hippocampus.__init__` as `self._episode_store = EpisodeStore()`. Hippocampus's `_to_dict()` (from P3.5 Stage 1) delegates the `episodes` key to `self._episode_store.to_dict()["episodes"]`; symmetric on load.

3. **Boundary detector as rule list** (~80 LOC in `memory/episode.py`) — Round 1 Arch important finding #2:
   ```python
   BoundaryRule = Callable[[PendingEpisodeState, CaptureEvent], bool]

   def tick_gap_rule(max_gap: int) -> BoundaryRule: ...
   def scn_tag_change_rule() -> BoundaryRule: ...
   def channel_change_rule() -> BoundaryRule: ...

   class EpisodeBoundaryDetector:
       def __init__(self, rules: list[BoundaryRule]) -> None: ...
       def should_close(self, pending, event) -> bool:
           return any(rule(pending, event) for rule in self._rules)
   ```
   Stage 1 ships three default rules (tick gap, scn_tag change, channel change) constructed from `EpisodeConfig`. P3b will append additional per-channel rules without touching Stage 1 code — same LOC count, cleaner extension seam.

4. **`EpisodeConfig` on `HippocampusConfig`** (~25 LOC in `memory/hippocampus.py` config block) — Round 1 cross-confirmed finding #3:
   ```python
   @dataclass
   class EpisodeConfig:
       boundary_tick_gap: int = 50
       hebbian_init: float = 0.3
       hebbian_delta: float = 0.1
       hebbian_max: float = 1.0  # clamp ceiling
   ```
   Added as `HippocampusConfig.episode: EpisodeConfig = field(default_factory=EpisodeConfig)`. Tests override via `HippocampusConfig(episode=EpisodeConfig(hebbian_delta=0.2, ...))`. No module-level constants, no monkeypatching required.

5. **Hebbian update rule on episode close** (~70 LOC, new private method `Hippocampus._apply_hebbian_on_close(episode)`, called from `finalize_pending_episode`) — folds Round 1 **cross-confirmed + Exec criticals**:
   ```python
   def _apply_hebbian_on_close(self, episode: Episode) -> None:
       cfg = self._config.episode
       graph = self._binding_graph  # owned by Hippocampus, not ATL
       nodes = episode.activated_nodes
       for a, b in itertools.combinations(nodes, 2):  # UNORDERED pairs — Round 1 Exec critical #1
           existing = graph.find_edge(a, b, EdgeType.ASSOCIATES)
           if existing is None:
               # add_bidirectional creates BOTH directions at cfg.hebbian_init
               graph.add_bidirectional(a, b, EdgeType.ASSOCIATES, weight=cfg.hebbian_init)
           else:
               # update_edge is directional — must update both (a,b) and (b,a)
               new_w = min(cfg.hebbian_max, existing.weight + cfg.hebbian_delta)
               graph.update_edge(a, b, EdgeType.ASSOCIATES, weight=new_w)
               graph.update_edge(b, a, EdgeType.ASSOCIATES, weight=new_w)
   ```
   **Pair enumeration uses `itertools.combinations`** (unordered): for nodes `[a, b, c]` pairs are `(a,b), (a,c), (b,c)` — three unique pairs, three Hebbian operations. The pre-Round-1 ordered-pair formulation would have visited each pair twice under `add_bidirectional`, double-applying `HEBBIAN_DELTA`. Regression guard test below verifies `len(graph._outgoing[a])` after N episode closes = expected pair count, not 2N or 4N.

6. **Wire discipline — `is not None` over truthy.** The binding graph is now held by Hippocampus directly (not via the optional ATL wire), so the truthy-trap regression case is slightly different. **Two regression guards required:**
   - `test_hebbian_update_fires_when_atl_is_none`: construct a Hippocampus with NO atl wire; close an episode; assert binding edges ARE created (because binding graph is Hippocampus-owned, ATL is irrelevant to Stage 1).
   - `test_hebbian_update_fires_when_atl_is_empty`: construct a Hippocampus wired to a freshly-constructed empty ATL (`len(atl) == 0`, evaluates falsy); close an episode; assert binding edges ARE created AND any ancillary ATL access code path uses `is not None`, never truthy.
   - **Grep-before-commit:** the P3a diff MUST contain zero occurrences of `if self\._(atl|nac|hippocampus|scn|ec|angular_gyrus)` truthy checks. Test runs `git grep` on the diff and asserts zero matches. (This replaces the earlier github-anchor link to [feedback_is_not_none_over_truthy.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_is_not_none_over_truthy.md) — dead anchors fixed per Round 1 Exec minor finding.)

7. **Partial-cue retrieval path** (~60 LOC, new method `Hippocampus.retrieve_on_cue(cue_node_id, limit=10) -> list[tuple[str, float]]`):
   - Query `self._binding_graph.get_associated(cue, edge_types={EdgeType.ASSOCIATES})` — returns `list[(neighbor, weight)]` in one call. No per-candidate `find_edge` loop; the graph already returns weights directly.
   - Sort by descending weight, take top-`limit`.
   - **One-hop only in Stage 1.** Multi-hop via `spreading_activation` is Stage 2 fallback. **Decision rule:** if Stage 2 fixture validation returns `mean recall < 0.70` under one-hop retrieval, switch to `spreading_activation([cue], decay=0.5, threshold=0.1, max_depth=3)` and retain the weights-based sort. If one-hop recall ≥ 0.70, multi-hop is deferred. (Round 1 Arch minor #3.)
   - **Perf note:** `get_associated` is O(|outgoing[cue]|) under a graph lock. On a popular cue with many binding edges, that's cheap. Stage 1 does NOT iterate episodes-containing-cue and union activated_nodes — that was the earlier quadratic formulation; get_associated is strictly faster.

8. **`_capture_thread_lock_ordering` guard** — Round 1 Exec important #5: document the acquire order invariant (`Hippocampus._episode_store._lock` THEN `Hippocampus._binding_graph._lock` — both RLocks, never the reverse) in `memory/episode.py` docstring. Stage 1 ships a test that exercises `finalize_pending_episode()` from inside a capture-thread-style calling context and asserts no deadlock within a 2-second budget. The capture thread is real (`Hippocampus._capture_worker`), so this test uses a controlled background thread plus an event to force the overlap.

9. **Metric extractor shell** (`tests/substrate/p3a_metrics.py`, ~50 LOC in Stage 1):
   - `precision_at_k(retrieved, ground_truth, k) -> float`
   - `recall_at_k(retrieved, ground_truth, k) -> float`
   - TODO marker for Stage 2 full metric + TF-IDF baseline.

10. **Synthetic mechanism tests** — `tests/substrate/test_p3a_episode_binding.py::TestP3aMechanism` (~350 LOC test file):
    - Use a `StubEncoder`-style approach — hand-crafted deterministic node IDs, no real text, no real embeddings. `add_edge` accepts unknown node IDs (verified), so fake IDs work end-to-end.
    - **Mechanism tests:**
      - `test_episode_close_creates_hebbian_edges`: episode `(a,b,c)`, finalize, assert three unordered pairs exist in binding graph at weight `hebbian_init`.
      - `test_episode_close_strengthens_existing_edges`: pre-seed `a↔b` at `hebbian_init`, close episode `(a,b)`, assert new weight = `hebbian_init + hebbian_delta` (no double-count — regression guard for Round 1 Exec critical #1).
      - `test_repeated_closes_no_edge_duplication`: close the same episode signature 5 times, assert `len(binding_graph._outgoing[a])` equals the expected pair count (one edge per unordered pair per direction), not 5× that. Regression guard for Round 1 Exec critical #2 (`add_edge` no-dedupe).
      - `test_strengthen_saturates_at_max`: close episode containing (a,b) N times where `N * delta >> max`, assert weight never exceeds `hebbian_max`.
    - **Retrieval tests:**
      - `test_partial_cue_retrieves_co_activated_nodes`: episode `(a,b,c,d)`, `retrieve_on_cue("a")` returns `{b, c, d}` with non-zero scores.
      - `test_partial_cue_baseline_non_member_returns_nothing`: episode `(a,b,c)`, `retrieve_on_cue("z")` returns `[]`.
      - `test_multiple_episodes_with_shared_node_merge_weights`: episode1 `(a,b)`, episode2 `(a,c)`; retrieve on `a`, assert both `b` and `c` present and `a↔b` weight reflects one reinforcement.
    - **Boundary detector tests:**
      - `test_boundary_tick_gap_closes_pending`: captures at ticks 0/10/20/100 produce two episodes.
      - `test_boundary_channel_change_closes_pending`: text@0 then vision@1 produce two episodes.
      - `test_boundary_scn_tag_change_closes_pending`: same-channel same-tick-band captures with different scn_tag produce two episodes.
      - `test_boundary_rule_list_extension`: construct `EpisodeBoundaryDetector` with a custom extra rule, verify it's consulted alongside the three defaults (extension-point validation for P3b).
    - **Wire discipline tests:**
      - `test_hebbian_update_fires_when_atl_is_none` (binding graph is Hippocampus-owned, so ATL is irrelevant).
      - `test_hebbian_update_fires_when_atl_is_empty` (len(atl) == 0, truthy-falsy trap, binding edges created anyway).
      - `test_no_truthy_bio_system_checks_in_diff` — subprocess `git grep` sentinel.
    - **Lock ordering test:**
      - `test_finalize_pending_episode_no_deadlock_with_capture_thread` — spawns a controlled background "capture" thread that holds `_episode_store._lock` while the main thread calls `finalize_pending_episode`; asserts completion within a 2-second budget.
    - **Persistence round-trip test:**
      - `test_episode_persistence_round_trip_via_hippocampus_dump`: close an episode, call `hippocampus._to_dict()`, construct a fresh hippocampus, call `hippocampus._load_from_dict(dumped)` (in-place, per P3.5 fold), assert episode round-trips + binding graph round-trips + `retrieve_on_cue` still works. **⚠ Depends on P3.5 Stage 1 committing FIRST in the same branch — Round 1 Arch minor #1.**

**Pass gate (Stage 1):**
- All 15+ synthetic mechanism tests in `TestP3aMechanism` pass.
- `ruff check` + `ruff format` clean on all touched files.
- Zero `if self\._(atl|nac|hippocampus|scn|ec|angular_gyrus)` truthy checks in the diff (verified by the in-test `git grep` sentinel).
- Zero edge duplication under repeated close (Round 1 Exec critical #2 regression guard).
- Zero double-delta strengthening (Round 1 Exec critical #1 regression guard).
- No capture-thread deadlock within 2s budget (Round 1 Exec important #5 test).
- Fast suite clean (standing exclusions per CLAUDE.md).
- Substrate subset clean: `PYTHONPATH=src python -m pytest tests/substrate/ tests/unit/test_pain_bus.py tests/unit/test_nac.py tests/unit/test_substrate_recognition.py tests/unit/test_bio_system_snapshot.py -q`.

**Tests (Stage 1):** See above test list. Metric extractor shell loads but only its two basic helpers are exercised; full baseline comparison is Stage 2.

### Stage 2 — fixture-based validation + TF-IDF baseline (**SHIPPED 2026-04-14**)

**What's built:**

- `tests/substrate/p3a_fixture_gen.py` — deterministic generator for `scenarios/substrate/synthetic_episodes.yaml`. 10 topics × 17 episodes each (170 total) using a **hub + chain** topology: each topic has a hub node connected to 4 chain nodes, with core episodes reinforced twice to drive edge weights to 0.4 (vs 0.3 for peripherals). See `p3a_fixture_gen.py` module docstring for the full design history.
- `tests/substrate/tfidf_baseline.py` — TF-IDF bag-of-concepts retriever matching the `retrieve_on_cue` callable shape for head-to-head comparison.
- `tests/substrate/p3a_metrics.py` — precision/recall/F1@k, per-probe / per-seed / aggregate stats, baseline-comparison helper.
- `tests/substrate/test_p3a_fixture_validation.py` — 12 tests across `TestStage2PassGate`, `TestOneHopArchitecturalFinding`, `TestRankingStability`, `TestFixturePersistenceRoundTrip`, `TestFixtureShape`, `TestEpisodeConfigRetrievalDefaults`.
- **`Hippocampus.retrieve_on_cue` now accepts `multi_hop: bool = False`.** Default stays one-hop for Stage 1 backward compatibility. Stage 2 validation calls with `multi_hop=True`, routing through `DependencyGraph.spreading_activation` on the binding graph.
- **`EpisodeConfig` extended with retrieval tuning** — `retrieval_decay=0.7`, `retrieval_threshold=0.001`, `retrieval_max_depth=5`. These drive the multi-hop path and are fully overridable via `HippocampusConfig(episode=EpisodeConfig(...))`.
- **Shuffle guard:** `TestRankingStability::test_ranking_robust_to_shuffled_ingestion_order` shuffles episode ingestion 5 ways and asserts byte-identical F1. Regression-guards the tie-fragility the reinforced fixture was specifically built to eliminate.
- **Persistence round-trip** via the P3.5 Stage 1 rebuild-from-episodes path. `TestFixturePersistenceRoundTrip` asserts byte-exact preservation (not just within ε=0.01).

**Stage 2 results:**

| Retriever | mean F1 | std F1 | Beats TF-IDF + 2σ? |
|---|---|---|---|
| **Hebbian multi-hop** | **1.0000** | **0.0000** | **✅ YES (margin 0.30)** |
| Hebbian one-hop | 0.7000 | 0.0000 | ❌ (parity) |
| TF-IDF baseline | 0.7000 | 0.0000 | — |

Full writeup: [../experiments/p3a_episode_binding_sweep.md](../experiments/p3a_episode_binding_sweep.md)
Results JSON: [../experiments/results/p3a_episode_binding_sweep.json](../experiments/results/p3a_episode_binding_sweep.json)
Reproduction runbook: [../experiments/protocols/p3a_episode_binding_reproduction.md](../experiments/protocols/p3a_episode_binding_reproduction.md)

**Stage 2 pass gate (all cleared):**
- ✅ Aggregate precision > 0.70, recall > 0.70 across 10 seeds.
- ✅ Hebbian multi-hop beats TF-IDF by `baseline_mean + 2 × baseline_std` (margin 0.30 absolute, std=0 makes the gate collapse to baseline_mean=0.70; multi-hop at 1.0 clears cleanly).
- ✅ Persistence round-trip preserves retrieval F1 within ε=0.01 (actually byte-exact).
- ✅ Fast suite + substrate subset + `ruff check` all green.

**Stage 2 architectural finding — load-bearing for P4/P6/P8:**

> **On a bag-of-words co-occurrence task, Hebbian one-hop retrieval and TF-IDF are algorithmically near-equivalent. The Hebbian mechanism's value over bag-of-words baselines manifests specifically in multi-hop / transitive retrieval via `spreading_activation` — a capability TF-IDF structurally cannot replicate because bag-of-words has no graph edges to walk.**

The first Stage 2 draft used a **clique-per-topic** fixture (5 core nodes all co-occurring in every episode). Both mechanisms scored F1 ≈ 1.0 because cliques have no transitive structure — one-hop already reaches everything, so TF-IDF matches. The pivot to hub+chain was the metric pivot the plan's "budget 2-3 pivots" guidance anticipated, and it exposed the real capability: chain-interior targets (e.g., `plate` from cue `prep`) are reachable via 2-3 hop graph walks but not via any bag-of-words overlap. `TestOneHopArchitecturalFinding::test_one_hop_does_not_beat_tfidf` locks this parity finding in with ±0.05 tolerance so future refactors don't silently invalidate the architectural claim.

**Tie-fragility fix (caught before ship):** Under single-shot core episodes, multi-hop chain targets tied EXACTLY with peripherals reached through the hub at the same 2-hop distance, leaving ranking dependent on dict-iteration order. Doubling core episode reinforcement (hub↔chain and chain adjacency) pushes core-edge weights to 0.4 vs peripheral 0.3, making chain targets **strictly** higher in `spreading_activation` scores. Regression-guarded by `TestRankingStability::test_chain_targets_strictly_outrank_peripherals`.

### Stage 3 — real-data sweep + pre-merge review

**What's built:**

- End-to-end sweep on ≥10 seeds × fixture with full numerical report (precision, recall, F1, baseline deltas).
- `docs/experiments/p3a_episode_binding_sweep.md` + `docs/experiments/results/p3a_episode_binding_sweep.json`.
- Reproduction runbook: `docs/experiments/protocols/p3a_episode_binding_reproduction.md`.
- **Pre-merge review round** (Executor + Architecture lenses in parallel, independent). Fold critical + important findings into the same branch before PR opens.

**Pass gate (Stage 3):**
- All 10 seeds pass individually (or document which seeds fall short and why, with aggregate still clearing).
- Review round completed with zero outstanding critical findings.
- Substrate subset + fast suite + `ruff check` all green.

**Tests (Stage 3):** Existing Stage 1+2 tests + any regression guards from review-round findings.

## Pass criteria (maps to version gate)

Stage 1 + Stage 2 + Stage 3 together constitute P3a's contribution to 0.3-target. 0.3-target closes when P3a + P3b + P3.5 + P4 are all `Status: COMPLETE`.

## Load-bearing invariants (post-Round-2 fold)

These are the invariants that both Round 1 and Round 2 pre-merge reviews established or hardened. Future changes to the episode binding surface must preserve all of them:

- **Hebbian edges live on `Hippocampus._binding_graph`, NOT `ATL.graph`.** Decouples from ATL compression. The `_binding_graph` field is Hippocampus-owned and orthogonal to both `ATL.graph` (concept topology) and `Hippocampus._graph` (associative edges between memory records). Node-id namespaces never overlap. `retrieve_on_cue()` queries ONLY the binding graph.
- **`itertools.combinations` (unordered) for pair enumeration.** Ordered-pair formulation visits each unordered pair twice under `add_bidirectional`, double-applying `hebbian_delta`. Regression guard: `test_episode_close_strengthens_existing_edges_by_exactly_delta`.
- **`DependencyGraph.add_edge` has no dedupe — always `find_edge` first** on any Hebbian write path, followed by either `add_bidirectional` (new edge) or a pair of directional `update_edge` calls. Regression guard: `test_repeated_closes_no_edge_duplication`.
- **`apply_hebbian_on_close` explicitly calls `add_node(id, id)` before any edge write** so the binding graph has a real node list for `to_dict()`. Regression guards: `test_binding_graph_nodes_populated_after_hebbian` + `test_binding_graph_to_dict_includes_nodes`.
- **`is not None` for bio-system wire checks, never truthy.** Regression guard: in-tree `inspect.getsource` grep over `memory/episode.py` + Hippocampus P3a methods (`test_p3a_source_has_no_truthy_biosystem_checks`).
- **`HippocampusConfig.episode: EpisodeConfig` is the config knob.** `EpisodeConfig` is defined ABOVE `HippocampusConfig` so `get_type_hints()` and dict-shaped YAML construction resolve eagerly (Round 2 Arch critical #1). No module-level constants, no monkeypatching.
- **`EpisodeStore` is a standalone class held as `Hippocampus._episode_store`.** Extension seam for P3b per-channel state + P5 bounded-storage eviction.
- **`EpisodeStore.load_from_dict` raises on duplicate episode ids** rather than silently overwriting. Guards against corrupt-file load corrupting the `_by_node` inverted index.
- **Boundary detector is a rule list**, not an if-chain. P3b appends via `Hippocampus.add_boundary_rule(rule)` — a public seam on Hippocampus, not reaching into `_episode_detector` directly.
- **Lock acquire order:** `Hippocampus._episode_lock` → `EpisodeStore._lock` → `binding_graph._lock`, never the reverse. Regression-guarded by the adversarial-thread deadlock test.
- **`_next_episode_ordinal` is persisted in `Hippocampus.dump()` and restored on `load_state()`.** Dump+reload+observe pre-fold crashed with `duplicate episode id: ep_1` because the ordinal was re-initialized to 0. Load has a fallback that derives the max ordinal from loaded episode ids for corrupt-file recovery.
- **Binding graph is rebuilt from loaded episodes on `load_state()`.** The binding graph itself is NOT persisted; its state is derived from the episodes. This eliminates the persistence asymmetry (Round 2 Arch important #4) — a restored Hippocampus produces the same `retrieve_on_cue` results as the original.
- **Multi-hop retrieval via `spreading_activation` is the primary Stage 2 path** and is the `retrieve_on_cue` **default**. `Hippocampus.retrieve_on_cue(cue, multi_hop=True)` is the default for all callers; `multi_hop=False` is an explicit opt-in reserved for Stage 1 mechanism tests that exercise one-hop weight semantics. Post-Stage-2 Arch lens flipped the default from `False` to `True` because the old default silently degraded every caller that forgot the kwarg while the "backward compat" hedge protected exactly one test suite.
- **`node_filter: Callable[[str], bool] | None` is the P3b/P4 retrieval-filter seam.** `retrieve_on_cue(cue, node_filter=...)` passes through to `spreading_activation(node_filter=...)`, which drops filtered-out nodes from traversal (both as sources and as hop targets). P3b channel integration will use this for per-channel retrieval; P4 cross-modal for modality filtering. Adding the seam in Stage 2 reserves the extension point so P3b/P4 don't rebuild `retrieve_on_cue` from scratch.
- **Multi-hop lift over one-hop is the architectural invariant**, NOT one-hop parity with TF-IDF. The test is `test_multi_hop_lift_over_one_hop_is_real` asserting `multi_hop_f1 > one_hop_f1 + 0.20` absolute. An earlier draft asserted `|one_hop_f1 - tfidf_f1| < 0.05` as a parity check, but the post-Stage-2 Arch lens flagged that as over-constraining — any future one-hop improvement (normalized edge weights, PageRank-style inference) would trip the test as a regression even though the multi-hop lift is the actual architectural claim. Current sweep: lift = 0.3045.
- **`EpisodeConfig` composes nested `HebbianConfig` + `RetrievalConfig`.** Field paths are `cfg.episode.hebbian.{init,delta,max_weight}` and `cfg.episode.retrieval.{decay,threshold,max_depth}`. The post-Stage-2 Arch lens flagged the earlier flat layout as a kitchen-sink risk (P3b/P4/P6 would each add their own knobs). The split is cheap now and expensive after P6 lands. `max_weight` is named with the suffix to avoid shadowing the builtin `max`.
- **Stage 2 fixture reinforces core episodes (hub↔chain and chain adjacency) twice** so core-edge weights (0.4) are strictly higher than peripheral-edge weights (0.3). Single-shot core episodes produced weight ties between chain targets and peripherals at 2-hop distance, making ranking fragile to dict-iteration order. Regression guard: `TestRankingStability::test_chain_targets_strictly_outrank_peripherals` (runs with `episode_dropout_rate=0` for byte-deterministic assertion).
- **Stage 2 fixture uses 10% per-seed episode dropout as a variance source.** Without dropout, every seed produced byte-identical metrics (std_f1 = 0.000) and the `baseline_mean + 2×std` gate collapsed to `baseline_mean` — ceremonial. With 10% dropout (17 of 170 base episodes dropped per seed via an independent seeded RNG), the gate does real statistical work: std_f1 ≈ 0.006 on all three retrievers, margin over baseline = 0.324 absolute. Probes are topology-only (do NOT depend on dropped episodes) so every seed runs the same 50 retrieval tasks; dropout affects what edges the retriever can build, not what it's asked to retrieve.

## Deferred concerns flagged by post-Stage-2 review (not in Stage 2 code — documented for downstream plans)

- **`spreading_activation` uses `max`-path aggregation.** Each node's final activation is the highest score from any path, not the sum. This means the current reinforcement-doubling fix (core episodes ×2, pushing core weight to 0.4 vs peripheral 0.3) creates a **fragile equilibrium with P6 extinction**: if extinction decays core edges back toward 0.3, the tie with peripherals re-emerges and multi-hop ranking can collapse silently. **P6 must decide:** either (a) add `sum` aggregation as an optional kwarg to `spreading_activation`, (b) use distinct `EdgeType` values (e.g., `HEBBIAN_BIND` vs `ASSOCIATES`) so extinction can decay without colliding with peripheral ranking, or (c) hold core edge weights above a strict floor during extinction. **Do not design P6 without picking one.**
- **P8 replay + `Hippocampus.load_state` rebuild both call `apply_hebbian_on_close` on the same episodes.** Stage 2 rebuilds the binding graph from persisted episodes on load; P8 sleep-replay will also re-run `apply_hebbian_on_close` on replayed episodes. If load fires on startup and P8 replays the same episode later in the session, edge weights double-apply (currently clamped at `hebbian_max=1.0` — so the bug is silently absorbed until someone raises `hebbian_max` or drops it). **P8 entry condition:** add an `(episode_id, edge_key)` idempotency marker or a `replayed_at_hebbian` flag so replay is by construction non-double-counting.
- **`RetrievalConfig` defaults are calibrated to the hub+chain fixture's exact weight arithmetic.** With `decay=0.7`, core weight 0.4, threshold 0.001, the effective reach is ~5 hops (before `max_depth=5` cap). On a fixture with different weight stratification (e.g., P3b real-text episodes averaging ~0.2 edge weight, or P5 10k+ node stress with sparser reinforcement), the threshold will prune earlier and multi-hop will not reach deep targets. **Re-tune when `retrieve_on_cue` recall drops on real-text fixtures.** The `RetrievalConfig` docstring (in `memory/hippocampus.py`) documents the calibration derivation.
- **`TfidfBaseline` currently lives in `tests/substrate/tfidf_baseline.py`.** If P3b / P4 want to reuse it for real-text or cross-modal baselines, they import from `tests.substrate.tfidf_baseline` (test-to-test imports work). When a 3rd consumer emerges, move to `src/maxim/memory/baselines/tfidf.py` (or similar). Rule of three — extract when 3 consumers exist, not 2.
- **`p3a_metrics.py` shares aggregation helpers with `p2_metrics.py` (P2 Stage 3).** Extract common `aggregate_seeds` / `compare_to_baseline` into `tests/substrate/metrics_common.py` when P3b adds a 3rd consumer. Split rule: anything that doesn't mention the phase's concepts (episode, channel, modality) goes to common.

## Review questions (Stage 3 reviewers — templates for Round 2 code review)

**Executor lens:**
- Does `_apply_hebbian_on_close` correctly handle N² pair enumeration when N is large (50+ nodes)? `itertools.combinations(N, 2)` is O(N²); any need to cap episode length?
- Does the episode boundary detector lose events during rapid channel switching? What happens if two rules fire simultaneously?
- Does `retrieve_on_cue` handle cases where a cue node is in a loaded episode but the binding graph was not round-tripped (partial-state scenario)?
- Are there re-entrancy concerns with `finalize_pending_episode` being called from within a capture thread? Lock ordering documented, but does the test actually exercise the adversarial path?
- Does the add-edge-no-dedupe regression guard catch the failure mode it claims to catch? Add a deliberate-bug test (call `add_edge` twice intentionally, verify the guard fires).

**Architecture lens:**
- Does `EpisodeStore` as a standalone class cleanly compose with P3b's per-channel rule additions and P5's bounded-storage eviction? Confirm the extension seam holds.
- Does the boundary detector rule-list shape force P3b to add rules in a specific order, or are rules commutative?
- If Stage 2 switches to multi-hop `spreading_activation`, does the `_binding_graph` retrieval path need any restructuring, or is it a one-line swap?
- When P4 ships cross-modal binding, should cross-modal binding edges live in `Hippocampus._binding_graph` alongside same-modality edges, or in a separate `_cross_modal_binding_graph`? Flag this before P4 opens.

## Deferred follow-ups

1. **Multi-hop retrieval via `spreading_activation`.** Stage 1 uses one-hop `get_associated`. Multi-hop switch is conditional on Stage 2 recall numbers; if one-hop clears, multi-hop is deferred.
2. **Episode compression** — merging similar episodes into a compressed representation. Deferred to P8 (sleep replay).
3. **Episode decay** — Hebbian edge weights decaying without reinforcement. Deferred to P6 (extinction).
4. **Reward-modulated Hebbian delta** — scaling `HEBBIAN_DELTA` by `sum(reward_events)` in the episode. Interesting but complicates the Stage 1 mechanism test. Deferred to Stage 2.
5. **Episode thread_id handling** — reserved in the dataclass but unused in Stage 1. P3b channel integration will wire it up.
6. **`retrieve_on_cue` perf under P5 stress** — one-hop `get_associated` is fast for modest graph sizes. Under 10k+ nodes with popular-cue hot spots, may need index optimizations. Deferred to P5.
7. **Binding graph ↔ ATL compression interaction (non-issue in Stage 1).** With the architectural pivot, ATL compression no longer destroys Hebbian edges because they're not on `ATL.graph`. If P4+ adds a cross-reference from binding graph nodes to ATL concepts, that cross-reference needs its own compression-safety check. Flagged here so P4 doesn't miss it.

## Production integration path (Round 2 Arch important #2)

`Hippocampus.observe_episode_event(event)` is the single Stage 1 entry point into the episode binding pipeline. In Stage 1 it is called only from `tests/substrate/test_p3a_episode_binding.py`; production integration is deferred to the session that wires behavioral experiments.

**When production integration lands** (post-Stage-1 session, not in this PR):

1. Call site will be inside `runtime/agent_loop.py` or `integration/memory_hub.py` alongside the existing `hippocampus.capture(...)` call — not replacing it. `capture()` persists the single-loop-cycle `EpisodicMemory` record; `observe_episode_event()` feeds the time-window episode binding detector. Both are load-bearing and orthogonal.

2. The `CaptureEvent` passed to `observe_episode_event` will be constructed from the existing percept/tick context the agent loop already has: `tick` from the loop clock, `channel` from the percept's `SensoryTag`, `sender_id` from the `PerceptContext` sender, `thread_id` from any conversation thread state, `scn_tag` from the wired `SCN`, and `activated_nodes` from the substrate encoder's output for the current percept.

3. **Do NOT invoke `observe_episode_event` from the `_capture_worker` background thread.** The Stage 1 lock-ordering test guarantees safety under controlled adversarial calls, but co-locating the call with `capture()` (which already runs on the main loop thread with clear lock semantics) is simpler.

4. The production integration session will ship: (a) a small glue function in `integration/memory_hub.py` that builds a `CaptureEvent` from percept + loop context, (b) one call site in the capture path, and (c) an end-to-end test that observes a real agent turn and verifies an episode lands in `hippocampus._episode_store`.

5. `Hippocampus.add_boundary_rule(rule)` is the public seam for P3b's per-channel rule additions. P3b will call this at agent construction time via the config surface, NOT by reaching into `_episode_detector` directly.

## Not in this plan

- Anything requiring P3b channel rules, P4 cross-modal, P5 stress, P6 extinction, P8 sleep replay.
- Integration with the production agent loop (captures via `runtime/agent_loop.py`). P3a wires to `Hippocampus` directly; runtime wiring lands when behavioral experiments need it.
- Real fixture YAML with natural-language text. Stage 1 is all synthetic. Real text fixtures land in Stage 2.
- Any touch to `similarity/encoder.py`, `decisions/nac.py`, `proprioception/pain_bus.py`, or other P2-shipped surfaces. The P2 load-bearing invariants in CLAUDE.md remain in force.
- Changes to `ATL.graph`. With the pivot, Hebbian edges never touch it.
