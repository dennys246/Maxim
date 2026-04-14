# Substrate P3a — Episode binding produces retrieval on partial cue

**Status:** Stage 1 in progress (2026-04-14)
**Scope:** ~400 LOC + ~100 metric extractor across 3 stages
**Target version:** 0.3-target
**Gates:** First of the four plans (P3a + P3b + P3.5 + P4) that together close 0.3-target.
**Depends on:** substrate_recognition ✅, P3.5 Stage 1 shell (for round-trip tests)
**Blocks:** P3b (channel integration reuses episode boundary machinery), P4 cross-modal binding (depends on episode-scoped binding working for same-modality first), B4 replanning (needs episode retrieval of prior attempts)
**Parent:** [substrate_binding_persistence.md](substrate_binding_persistence.md)
**Related:** [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md), [substrate_p3_5_persistence_snapshot.md](substrate_p3_5_persistence_snapshot.md)

## Goal

Ship the Hebbian episode-binding mechanism. An episode is a multi-event time window during which co-activated substrate nodes form durable associative links; presenting a partial cue from the episode retrieves the others with a margin greater than a TF-IDF bag-of-concepts baseline. This is the first substrate mechanism that produces **retrieval of things that were never directly queried**, which is the substrate's load-bearing claim for cross-modal binding in P4.

## Hypothesis (falsifiable)

Nodes co-occurring in the same Hippocampus episode form durable links through Hebbian updates on ATL's within-layer `DependencyGraph` edges. Presenting a single node from a prior episode as a cue retrieves the other nodes from that episode by a margin greater than a TF-IDF bag-of-concepts baseline computed on the same episode fixtures. The margin is stable across ≥10 seeds at `precision > 0.70` and `recall > 0.70`.

## Dependencies — scaffolding audit

The [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md) audit already established that P3a needs **less new infrastructure than the parent plan suggests**, because the Hebbian within-ATL edge mechanism has an existing home. This section pins the exact call sites.

**Existing surfaces (verified 2026-04-14):**

| Surface | File:line | Purpose in P3a |
|---|---|---|
| `DependencyGraph.add_bidirectional(a, b, edge_type=ASSOCIATES, weight=1.0)` | [agents/bus.py](../../src/maxim/agents/bus.py) | Create new Hebbian edge on first co-occurrence. |
| `DependencyGraph.update_edge(src, tgt, ASSOCIATES, weight=new)` | agents/bus.py | Update existing Hebbian edge weight. |
| `DependencyGraph.find_edge(src, tgt, ASSOCIATES) -> Edge \| None` | agents/bus.py | Read current weight for Hebbian delta. |
| `DependencyGraph.get_associated(node_id, edge_types={ASSOCIATES}) -> list[(str, float)]` | agents/bus.py | One-hop partial-cue retrieval. |
| `DependencyGraph.spreading_activation(source_ids, ...)` | agents/bus.py | Multi-hop retrieval (deferred to Stage 2; one-hop is enough for Stage 1 mechanism test). |
| `EdgeType.ASSOCIATES` | agents/bus.py | No new edge type needed. |
| `ATL.graph` | [memory/atl.py](../../src/maxim/memory/atl.py) | ATL's internal `DependencyGraph`. Hebbian edges live here. |
| `Hippocampus.capture(memory)` / `capture_from_loop(...)` | [memory/hippocampus.py](../../src/maxim/memory/hippocampus.py) | Single-event capture. P3a wraps this in a boundary detector that groups multiple events into an `Episode`. |
| `BioSystemSnapshot` Protocol + `Hippocampus.dump()` with `episodes` key | [memory/snapshot.py](../../src/maxim/memory/snapshot.py) (P3.5 Stage 1) | Persistence round-trip for episodes lands here. |
| `tests/substrate/persistence_harness.py` (S3) | existing | Subprocess round-trip harness. Used in P3a Stage 2+ for round-trip validation. |

**New surfaces (what P3a actually builds):**

| Surface | Scope | Stage |
|---|---|---|
| `Episode` dataclass | ~50 LOC | 1 |
| `EpisodeStore` on `Hippocampus` | ~80 LOC | 1 |
| Episode boundary detector (tick gap + scene signal) | ~80 LOC | 1 |
| Hebbian update rule on episode close | ~50 LOC | 1 |
| Partial-cue retrieval path | ~60 LOC | 1 |
| TF-IDF gate baseline | ~100 LOC | 2 |
| Metric extractor (`tests/substrate/p3a_metrics.py`) | ~100 LOC | 1 shell + 2 full |
| Synthetic fixture (`scenarios/substrate/synthetic_episodes.yaml`) | YAML + loader | 2 |

**Naming clarification.** There is an existing `EpisodicMemory` type at [memory/types.py:472](../../src/maxim/memory/types.py#L472), but it represents a single loop cycle (perception → decide → act → outcome), not a multi-event time window. These are orthogonal concepts. P3a's `Episode` is a new type, not an extension of `EpisodicMemory`. Simulation "episodes" (campaign runs) in `simulation/` are a third orthogonal concept in a different domain. The naming is unambiguous in context.

## Stages

### Stage 1 — mechanism tests on synthetic geometry

**What's built:**

1. **`src/maxim/memory/episode.py`** (new, ~50 LOC):
   ```python
   @dataclass
   class Episode:
       id: str
       start_tick: int
       end_tick: int
       channel: str
       sender_ids: list[str]
       thread_id: str | None
       activated_nodes: list[str]
       reward_events: list[tuple[int, float]]  # (tick, delta)
       scn_tag: str | None
   ```
   Plus `Episode.to_dict()` / `Episode.from_dict()` for P3.5 round-trip.

2. **`EpisodeStore` embedded in `Hippocampus`** (~80 LOC in hippocampus.py):
   - `self._episodes: dict[str, Episode]` (id → episode)
   - `self._episodes_by_node: dict[str, set[str]]` (node_id → set of episode ids — inverted index)
   - `add_episode(episode)` / `get_episode(id)` / `episodes_containing(node_id)`
   - Persistence: the P3.5 Stage 1 `Hippocampus._to_dict()` reserved key `"episodes"` is now populated by `[ep.to_dict() for ep in self._episodes.values()]`. Load symmetric.

3. **Episode boundary detector** (~80 LOC, new method on `Hippocampus`):
   - Rule 1 (tick gap): if current tick > previous capture tick + `boundary_tick_gap` (config, default 50), close the pending episode and open a new one.
   - Rule 2 (scene signal): if the incoming capture has `scn_tag != pending_episode.scn_tag`, close and open.
   - Rule 3 (channel change): same — channel switch closes the episode.
   - Pending episode state lives on the instance; episode close triggers the Hebbian update rule below.
   - A new `Hippocampus.finalize_pending_episode()` explicit method is also exposed for test control (do not force tests to wait for gap rules).

4. **Hebbian update rule on episode close** (~50 LOC, new private method `_apply_hebbian_on_close(episode)` on `Hippocampus`, called from `finalize_pending_episode` if an ATL reference is wired):
   - For every ordered pair `(a, b)` in `episode.activated_nodes`:
     - If `atl.graph.find_edge(a, b, ASSOCIATES)` is `None` → `atl.graph.add_bidirectional(a, b, EdgeType.ASSOCIATES, weight=HEBBIAN_INIT)`
     - Else → `update_edge(a, b, ASSOCIATES, weight=min(1.0, existing.weight + HEBBIAN_DELTA))` + symmetric update on `(b, a)`
   - `HEBBIAN_INIT = 0.3`, `HEBBIAN_DELTA = 0.1` (config-overridable).
   - **Load-bearing wire check:** the ATL reference is an **optional** wire on `Hippocampus` (passed via `set_atl(atl)` or constructor). The check MUST be `if self._atl is not None` and NEVER `if self._atl` — because ATL has `__len__` (concept count) and evaluates falsy when empty, which would silently skip Hebbian updates during a fresh-start sim. This is the exact bug class that bit P2 Stage 1 on NAc (see [feedback_is_not_none_over_truthy.md](https://github.com/dennys246/Maxim#is_not_none) and the P2 retrospective).
   - Grep-before-commit discipline: the P3a diff MUST NOT contain any occurrence of `if self\._(atl|nac|hippocampus|scn|ec|angular_gyrus)` — always `is not None`.

5. **Partial-cue retrieval path** (~60 LOC, new method `Hippocampus.retrieve_on_cue(cue_node_id, limit=10) -> list[tuple[str, float]]`):
   - Look up episodes containing `cue_node_id` via `_episodes_by_node[cue]`.
   - Union all `activated_nodes` across those episodes, excluding the cue itself.
   - For each candidate, compute a score = max edge weight on `ASSOCIATES` edge between cue and candidate (via `find_edge`).
   - Return top-`limit` sorted by descending score.
   - **Simplest possible retrieval** — multi-hop via `spreading_activation` is Stage 2+.

6. **Metric extractor shell** (`tests/substrate/p3a_metrics.py`, ~50 LOC in Stage 1, full ~100 LOC in Stage 2):
   - `precision_at_k(retrieved: list[str], ground_truth: set[str], k: int) -> float`
   - `recall_at_k(...)` — same
   - TODO marker for Stage 2: full metric including TF-IDF gate baseline comparison + F1 aggregation over a fixture.

7. **Synthetic mechanism tests** — `tests/substrate/test_p3a_episode_binding.py::TestP3aMechanism` (~250 LOC test file):
   - Use a `StubEncoder`-style approach — hand-crafted deterministic node IDs, no real text, no real embeddings.
   - `test_episode_close_creates_hebbian_edges`: build an episode with nodes `["a", "b", "c"]`, finalize, assert edges `a↔b`, `a↔c`, `b↔c` all exist in `atl.graph` with weight ≥ `HEBBIAN_INIT`.
   - `test_episode_close_strengthens_existing_edges`: pre-seed an `a↔b` edge at weight 0.3, close an episode containing both, assert new weight ≈ 0.4.
   - `test_partial_cue_retrieves_co_activated_nodes`: close an episode `["a", "b", "c", "d"]`, call `retrieve_on_cue("a")`, assert `{"b", "c", "d"}` returned with non-zero scores.
   - `test_partial_cue_baseline_non_member_returns_nothing`: close episode `["a", "b", "c"]`, call `retrieve_on_cue("z")` (never in any episode), assert empty.
   - `test_multiple_episodes_with_shared_node_merge_weights`: episode 1 `[a, b]`, episode 2 `[a, c]`. Close both. Retrieve on `a`. Assert both `b` and `c` returned; assert the `a↔b` weight reflects one reinforcement, not two (unless episode 1 closes, then re-opens and closes again — then yes two reinforcements).
   - `test_episode_boundary_tick_gap_closes_pending`: capture at ticks 0, 10, 20, then capture at tick 100 (> 50 gap). Assert the first three form one episode, the fourth starts a new pending episode.
   - `test_episode_boundary_channel_change_closes_pending`: capture on channel "text" at tick 0, channel "vision" at tick 1. Assert two separate episodes.
   - `test_hebbian_update_skipped_when_atl_is_none`: construct a `Hippocampus` without wiring ATL. Close an episode. Assert no crash, no edges created. Sanity + regression for the `is not None` guard.
   - `test_hebbian_update_fires_when_atl_is_empty`: construct a `Hippocampus` wired to a **freshly-constructed empty ATL** (0 concepts, `len(atl) == 0`, evaluates falsy under truthy check). Close an episode. Assert Hebbian edges ARE created. This is the regression guard for the `is not None` vs truthy bug class.
   - `test_episode_persistence_round_trip_via_hippocampus_dump`: close an episode, call `hippocampus._to_dict()`, construct a fresh hippocampus, call `_from_dict(dumped)`, assert the episode round-trips and `retrieve_on_cue` still works. **Depends on P3.5 Stage 1.**

**Pass gate (Stage 1):**
- All 9 synthetic mechanism tests in `TestP3aMechanism` pass.
- `ruff check` + `ruff format` clean on all touched files.
- No `if self\._(atl|nac|hippocampus|scn|ec|angular_gyrus)` truthy checks in the P3a diff — `git diff | grep` verifies zero hits.
- Fast suite clean (standing exclusions per CLAUDE.md).
- Substrate subset clean: `PYTHONPATH=src python -m pytest tests/substrate/ tests/unit/test_pain_bus.py tests/unit/test_nac.py tests/unit/test_substrate_recognition.py tests/unit/test_bio_system_snapshot.py -q`.

**Tests (Stage 1):** See above test list. Metric extractor shell loads but only its two basic helpers are exercised; full baseline comparison is Stage 2.

### Stage 2 — fixture-based validation + TF-IDF baseline

**What's built:**

- `scenarios/substrate/synthetic_episodes.yaml` — 100 synthetic episodes with labeled co-occurrence ground truth. Each episode names its "cue" node + "target" node set. ~1-2 days authoring (per parent plan scope estimate).
- TF-IDF bag-of-concepts baseline: computes a bag of concept IDs per episode, ranks candidate retrievals by IDF-weighted overlap. Ships as `tests/substrate/tfidf_baseline.py`.
- Full metric extractor in `p3a_metrics.py`: per-seed precision/recall/F1, aggregate mean+std across seeds, baseline comparison (Hebbian vs TF-IDF by `baseline_mean + 2×baseline_std`).
- Fixture-driven validation test `tests/substrate/test_p3a_fixture_validation.py::TestP3aFixture` — runs the mechanism on the 100-episode fixture, asserts aggregate precision > 0.70, recall > 0.70, beats TF-IDF baseline.
- Shuffle guard: test MUST run with shuffled fixture ordering (per [feedback_shuffle_fixture_ordering.md](https://github.com/dennys246/Maxim) — sequential ordering produced node-growth artifacts in P2).
- Persistence round-trip on the fixture: dump after fixture run, load in subprocess, assert retrieval scores round-trip. Uses the P3.5 Stage 1 harness.

**Pass gate (Stage 2):**
- Aggregate precision > 0.70, recall > 0.70 across ≥10 seeds on the 100-episode fixture.
- Hebbian mechanism beats TF-IDF baseline by `baseline_mean + 2×baseline_std`.
- Persistence round-trip preserves retrieval F1 within ε=0.01.
- Fast suite + substrate subset + `ruff check` all green.

**Tests (Stage 2):** See list above.

**Budget 2-3 metric pivots.** Per [feedback_three_iteration_metric_pivot.md](https://github.com/dennys246/Maxim) and the P2 Stage 3 retrospective — the first fixture-based run WILL return numbers that look wrong. The response is NOT to widen the gate; the response is to figure out what the metric is actually measuring and rebuild it. A monolithic "write metric once and run it" approach is explicitly forbidden by the P2 retrospective.

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

## Load-bearing invariants (filled in AFTER shipping Stage 1)

TODO. Populate after Stage 1 review round with the actual gotchas encountered. Expected candidates based on the audit + P2 retrospective:
- `is not None` for ATL wire check (prevented-bug; regression-guarded).
- HEBBIAN_INIT / HEBBIAN_DELTA tuning bounds (empirical; TBD).
- Episode boundary detector interaction with interleaved threads / multi-sender channels (TBD in Stage 2).
- Partial-cue retrieval score normalization — max-weight vs sum-weight vs spreading-activation (TBD if Stage 1 retrieval is too noisy).

## Review questions (Stage 3 reviewers — templates for later use)

**Executor lens:**
- Does `_apply_hebbian_on_close` correctly handle N² pair enumeration when N is large (episodes with 50+ nodes)? Any quadratic explosion?
- Does the episode boundary detector lose events during rapid channel switching? What happens to the "pending" episode buffer if a channel flip races with a tick-gap close?
- Does `retrieve_on_cue` handle the case where a cue node is in an episode that has been persisted-and-reloaded but the ATL graph has NOT been? (Cross-system partial state.)
- Is the `HEBBIAN_INIT` + `HEBBIAN_DELTA` value pair actually symmetric under `add_bidirectional`? Verify the `(b, a)` direction sees the same weight as `(a, b)`.
- Any re-entrancy concerns with `finalize_pending_episode` being called from within a capture thread?

**Architecture lens:**
- Should `Episode` live in `memory/episode.py` (current plan) or be merged into `memory/types.py` with `EpisodicMemory`? Confirm the orthogonality argument holds.
- Is `EpisodeStore` correctly embedded in `Hippocampus` (current plan) or should it be a separate class co-located in a new `memory/episode_store.py`?
- The `is not None` ATL wire check — does this shape generalize, and should it be a helper utility `_require_wired(system, name)`?
- When P3b ships channel integration, does the boundary detector API generalize or does P3b have to rewrite the detector? (Pre-plan the extension point now.)
- Does the Hebbian update on `ATL.graph` interact correctly with ATL's existing concept eviction / compression? What happens to Hebbian edges when ATL compresses a concept node into a `CompressedSemantic`?

## Deferred follow-ups

1. **Multi-hop retrieval via `spreading_activation`.** Stage 1 uses one-hop `get_associated`. Multi-hop is a Stage 2 experiment; if one-hop is good enough, multi-hop becomes a deferred follow-up.
2. **Episode compression** — merging similar episodes into a compressed representation. Deferred to P8 (sleep replay).
3. **Episode decay** — Hebbian edge weights decaying without reinforcement. Deferred to P6 (extinction).
4. **Reward-modulated Hebbian delta** — scaling `HEBBIAN_DELTA` by the reward events in the episode. Interesting but complicates the Stage 1 mechanism test. Deferred to Stage 2.
5. **Episode thread_id handling** — currently reserved in the dataclass but unused. P3b channel integration will wire it up.

## Not in this plan

- Anything requiring P3b channel rules, P4 cross-modal, P5 stress, P6 extinction, P8 sleep replay.
- Integration with the production agent loop (captures via `runtime/agent_loop.py`). P3a wires to `Hippocampus` directly; runtime wiring is an integration step that lands when behavioral experiments need it.
- Real fixture YAML with natural-language text. Stage 1 is all synthetic. Real text fixtures land in Stage 2.
- Any touch to `similarity/encoder.py`, `decisions/nac.py`, `proprioception/pain_bus.py`, or other P2-shipped surfaces. The P2 load-bearing invariants in CLAUDE.md remain in force.
