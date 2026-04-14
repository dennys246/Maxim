# Substrate P3a — Episode binding produces retrieval on partial cue

**Status:** Stage 1 in progress (2026-04-14, post-Round-1-review fold)
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

### Stage 2 — fixture-based validation + TF-IDF baseline

**What's built:**

- `scenarios/substrate/synthetic_episodes.yaml` — 100 synthetic episodes with labeled co-occurrence ground truth. Each episode names its "cue" node + "target" node set. ~1-2 days authoring (per parent plan scope estimate).
- TF-IDF bag-of-concepts baseline in `tests/substrate/tfidf_baseline.py`.
- Full metric extractor in `p3a_metrics.py`: per-seed precision/recall/F1, aggregate mean+std across seeds, baseline comparison (Hebbian vs TF-IDF by `baseline_mean + 2×baseline_std`).
- Fixture-driven validation test `tests/substrate/test_p3a_fixture_validation.py::TestP3aFixture`.
- **Shuffle guard:** test MUST run with shuffled fixture ordering (per [feedback_shuffle_fixture_ordering.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_shuffle_fixture_ordering.md) — link fixed per Round 1 Exec minor).
- Persistence round-trip on the fixture: dump after fixture run, load in subprocess, assert retrieval scores round-trip. Uses the P3.5 Stage 1/2 harness.
- **Multi-hop switch decision**: if one-hop `get_associated` returns mean recall `< 0.70` on the fixture, switch to `spreading_activation`. Document the decision (and the pre-switch baseline) in the Stage 2 results writeup.

**Pass gate (Stage 2):**
- Aggregate precision > 0.70, recall > 0.70 across ≥10 seeds on the 100-episode fixture.
- Hebbian mechanism beats TF-IDF baseline by `baseline_mean + 2×baseline_std`.
- Persistence round-trip preserves retrieval F1 within ε=0.01.
- Fast suite + substrate subset + `ruff check` all green.

**Tests (Stage 2):** See list above.

**Budget 2-3 metric pivots.** Per [feedback_three_iteration_metric_pivot.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_three_iteration_metric_pivot.md) and the P2 Stage 3 retrospective — the first fixture-based run WILL return numbers that look wrong. The response is NOT to widen the gate; the response is to figure out what the metric is actually measuring and rebuild it. A monolithic "write metric once and run it" approach is explicitly forbidden by the P2 retrospective.

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
- **One-hop retrieval via `get_associated`**; multi-hop via `spreading_activation` is a Stage 2 fallback, conditional on Stage 2 recall < 0.70.

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
