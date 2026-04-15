# Substrate P3b — Channel integration: boundary rules + filtered retrieval

**Status:** Draft (2026-04-14)
**Scope:** ~250 LOC + ~100 metric extractor across 3 stages
**Target version:** 0.3-target
**Gates:** Second of the four plans (P3a ✅ + P3b + P3.5 + P4) that together close 0.3-target. P3b opens now that P3a Stage 2 has shipped (PR #112 merged).
**Depends on:** P3a Stage 1+2 ✅ (Episode / EpisodeStore / BoundaryDetector / `Hippocampus.add_boundary_rule` / `retrieve_on_cue(node_filter=...)`), P3.5 Stage 1 ✅ (persistence round-trip surface)
**Blocks:** P4 cross-modal mug test (needs per-channel retrieval as the same-modality baseline it compares against)
**Parent:** [substrate_binding_persistence.md](substrate_binding_persistence.md)
**Related:** [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md), [substrate_p3a_episode_binding.md](substrate_p3a_episode_binding.md), [substrate_p3_5_persistence_snapshot.md](substrate_p3_5_persistence_snapshot.md)

## Goal

Ship channel-aware episode binding. Different communication channels have different natural episode-boundary semantics — an SMS conversation closes when the contact changes or the gap stretches past a "new conversation" threshold; a narrative passage closes on scene change or narrator handoff. P3b adds per-channel boundary rules on top of the P3a boundary detector's rule-list shape, and a channel-filtered retrieval path on top of the `node_filter` seam `retrieve_on_cue` already exposes. The architectural claim: **channel-filtered retrieval via the P3b seam beats a metadata-grep baseline by a real margin on channel-segregated episodes**, proving that the Hebbian binding graph captures structure the metadata-only grep cannot.

## Hypothesis (falsifiable)

Per-channel episode boundary rules produce episodes whose channel-filtered retrieval — via `Hippocampus.retrieve_on_cue(cue, node_filter=<channel/sender membership>)` — beats a metadata-grep baseline (which just returns nodes that appeared in any episode matching the channel/sender filter, without walking the binding graph) by `baseline_mean + 2 × baseline_std` on mean F1 across ≥10 seeds at `precision > 0.70, recall > 0.70`. The lift comes from multi-hop traversal through channel-filtered subgraphs — transitive structure that the grep baseline cannot represent.

## Dependencies — scaffolding audit

**What P3a Stage 1+2 already shipped that P3b directly reuses:**

| Surface | Location | How P3b uses it |
|---|---|---|
| `EpisodeBoundaryDetector` with rule-list shape (`add_rule`) | [memory/episode.py](../../src/maxim/memory/episode.py) | P3b appends channel-aware rules without touching the detector core. |
| `Hippocampus.add_boundary_rule(rule)` — public seam | [memory/hippocampus.py](../../src/maxim/memory/hippocampus.py) | P3b call sites attach per-channel rules at construction time. |
| `tick_gap_rule / channel_change_rule / scn_tag_change_rule` factories | memory/episode.py | Stage 1 defaults; P3b adds `sms_*` and `narrative_*` rule factories alongside. |
| `CaptureEvent.{channel, sender_id, thread_id, scn_tag}` | memory/episode.py | Per-channel rules read these fields. Already populated. |
| `EpisodeStore` with `_by_node` inverted index + `episodes_containing(node_id)` | memory/episode.py | P3b's channel filter queries the index then checks each episode's `channel` / `sender_ids` metadata. |
| `Episode.{channel, sender_ids, thread_id}` frozen dataclass | memory/episode.py | Already carries the fields the filter needs. |
| `Hippocampus.retrieve_on_cue(multi_hop=True, node_filter=...)` | memory/hippocampus.py | P3b's channel-filtered retrieval passes a channel-membership predicate as `node_filter`. No retrieval-path rewrite. |
| `DependencyGraph.spreading_activation(..., node_filter=...)` | agents/bus.py | Handles the filter at the traversal level — drops filtered nodes from source selection and hop targets. |
| P3.5 `load_state` rebuild-from-episodes + `next_episode_ordinal` round-trip | memory/hippocampus_persistence.py | P3b tests inherit P3a's byte-exact persistence round-trip. |

**Conclusion:** the boundary-rule and retrieval-filter infrastructure is already in place. P3b adds rule factories, a channel-membership node_filter builder, and tests. **Scope is ~250 LOC because the plumbing is 99% already there** — this is what the Stage 2 fold's "reserve the seam now" decisions bought us.

**New surfaces P3b must ship:**

| Surface | Scope | Stage |
|---|---|---|
| `sms_gap_rule(max_gap)` | ~15 LOC | 1 |
| `sms_sender_change_rule()` | ~20 LOC (new `sender_id` comparison) | 1 |
| `narrative_scene_rule()` | ~15 LOC (alias/refinement of `scn_tag_change_rule`) | 1 |
| `channel_specific_rule(channel, inner_rule)` wrapper | ~20 LOC | 1 |
| `Hippocampus.channel_membership_filter(channel, sender=None)` helper | ~40 LOC | 1 |
| Synthetic mechanism tests | ~250 LOC test file | 1 |
| `scenarios/substrate/channel_episodes.yaml` + generator | ~200 LOC generator + YAML | 2 |
| Metadata-grep baseline | ~80 LOC | 2 |
| Full metric extractor (reuses `p3a_metrics`, may promote shared helpers to `metrics_common.py`) | ~50 LOC delta | 2 |
| Fixture validation head-to-head test | ~200 LOC | 2 |
| Results writeup + reproduction protocol | ~300 lines docs | 3 |

## Stages

### Stage 1 — mechanism tests on synthetic geometry

**What's built:**

1. **New rule factories in `memory/episode.py`** (or a new `memory/channel_rules.py` if the growth bothers a reviewer):
   - `sms_gap_rule(max_gap_ticks=500)` — closes the pending episode when the next event is more than `max_gap_ticks` ticks after the previous capture. SMS-specific because conversations can have minute-to-hour gaps that aren't episode boundaries; narrative gaps are tighter. Default 500 is a tuning placeholder; Stage 2 sweep will refine.
   - `sms_sender_change_rule()` — closes when the incoming `event.sender_id` differs from ALL senders in the pending episode's `sender_ids` set. "A new person texted" = new episode. Guards against sender=None (no-op).
   - `narrative_scene_rule()` — thin alias over the existing `scn_tag_change_rule` factory, named for clarity. Stage 2 may refine with narrator-change detection.
   - `channel_specific_rule(channel: str, rule: BoundaryRule) -> BoundaryRule` — wraps a rule so it only fires when `pending.channel == channel`. Composes with the existing rule-list shape — no detector-core changes. The "per-channel rule set" architecture is a wrapping pattern, not a data-structure change.

2. **`Hippocampus.channel_membership_filter(channel, sender=None)` helper** — builds a `node_filter` callable that returns True for nodes appearing in ≥1 episode matching the channel (and optionally sender) criteria. Implementation: query `self._episode_store.episodes_containing(node_id)` and check each episode's `channel` + `sender_ids`. Reuses the existing inverted index. Callers pass the result directly to `retrieve_on_cue(cue, node_filter=...)` — no new retrieval method.

3. **Synthetic mechanism tests** — `tests/substrate/test_p3b_channel_integration.py::TestP3bMechanism` (~250 LOC test file, following the P3a Stage 1 pattern):
   - `test_sms_gap_rule_closes_on_long_gap`: captures at ticks 0, 10, 200 (within 500 gap) vs 0, 10, 600 (> 500 gap). First forms one episode, second closes after tick 10.
   - `test_sms_sender_change_rule_closes_on_new_contact`: capture from alice then alice then bob on SMS; rule fires on bob, closing the alice episode.
   - `test_sms_sender_change_rule_no_op_when_sender_is_none`: capture with `sender_id=None` on both events; rule does not fire.
   - `test_narrative_scene_rule_closes_on_scn_tag_change`: alias-level behavior check.
   - `test_channel_specific_rule_fires_only_in_matching_channel`: wrap an always-fire rule in `channel_specific_rule("sms", ...)`, confirm it closes SMS episodes but not narrative episodes.
   - `test_channel_specific_rule_composes_with_default_rules`: default rules (tick_gap, channel_change) still fire alongside the wrapped SMS rule.
   - **Channel-filtered retrieval tests:**
     - `test_channel_filter_returns_only_channel_members`: build a fixture with cue=X present in both SMS and narrative episodes. `retrieve_on_cue(X, node_filter=channel_membership_filter("sms"))` returns only nodes from SMS episodes.
     - `test_sender_filter_returns_only_sender_members`: same but filtered by sender.
     - `test_channel_and_sender_filter_compose`: both filters together produce the intersection.
     - `test_channel_filter_empty_channel_returns_empty`: filter on a channel with no episodes returns `[]` (no crash).
     - `test_channel_filter_respects_multi_hop_traversal`: channel-filtered retrieval still walks multi-hop through filtered-channel nodes (inherits `retrieve_on_cue(multi_hop=True)` default).
   - **Persistence round-trip tests:**
     - `test_channel_rules_survive_dump_load`: boundary rules themselves are NOT persisted (they're behavior, not state), but the episodes they produced ARE. Dump + load + rerun channel-filtered retrieval must produce byte-identical results. Reuses P3.5 Stage 1 rebuild-from-episodes.
     - `test_channel_membership_filter_reconstructs_after_load`: the filter is rebuilt from `episode_store` state post-load; assert it produces the same results as pre-dump.
   - **Wire discipline regression** (inherited from P3a pattern):
     - `test_p3b_source_has_no_truthy_biosystem_checks`: AST grep over `memory/episode.py` + `memory/hippocampus.py` P3b-added methods. Reuses the `is not None` invariant.

4. **Metric extractor shell** (`tests/substrate/p3b_metrics.py`, ~50 LOC in Stage 1): imports `aggregate_seeds` + `compare_to_baseline` from `p3a_metrics.py` (deferred consolidation to `metrics_common.py` noted in P3a Stage 2 deferred list; may promote now if Stage 1 adds the third consumer). Stage 1 ships `channel_filtered_precision_at_k` / `channel_filtered_recall_at_k` helpers.

**Pass gate (Stage 1):**
- All 12+ synthetic mechanism tests in `TestP3bMechanism` pass.
- `ruff check` + `ruff format` clean on all touched files.
- No truthy bio-system checks in the P3b diff (AST regression guard).
- Fast suite clean (standing exclusions per CLAUDE.md).
- Substrate subset clean: `PYTHONPATH=src python -m pytest tests/substrate/ tests/unit/test_bio_system_snapshot.py -q`.
- P3a Stage 1+2 tests + P3.5 Stage 1 tests all remain green (zero regression).

**Tests (Stage 1):** See above. No fixture YAML, no baseline, no 10-seed sweep — those are Stage 2.

### Stage 2 — fixture-based validation + metadata-grep baseline

**What's built:**

- `scenarios/substrate/channel_episodes.yaml` — ~100 episodes across SMS + narrative channels with labeled sender metadata + ground truth co-occurrence. Generator pattern: hub+chain topology per (channel, sender) pair, with some cross-channel nodes that test channel-filter discrimination. ~1-2 days authoring time per the parent plan.
- **Metadata-grep baseline** in `tests/substrate/metadata_grep_baseline.py` (~80 LOC): searches episodes by channel+sender metadata, returns all nodes appearing in matching episodes. No graph traversal. This is the "what you'd get with just episode metadata indexing" baseline the plan's pass gate compares against.
- **Full metric extractor** in `p3b_metrics.py`: per-seed precision/recall/F1, aggregate mean+std across seeds, baseline comparison by `baseline_mean + 2 × baseline_std`. If `p3b_metrics` ends up with substantial shared logic with `p3a_metrics` + `p2_metrics`, this is the point to extract `tests/substrate/metrics_common.py` (rule of three).
- **Fixture validation test** `tests/substrate/test_p3b_fixture_validation.py::TestP3bFixture` — 10-seed head-to-head on the 100-episode fixture. Asserts: Hebbian channel-filtered retrieval beats metadata-grep by `baseline_mean + 2×std` on F1, and mean precision/recall both exceed 0.70.
- **Variance source:** per-seed episode dropout (10%), matching the P3a Stage 2 pattern. Non-ceremonial `+2σ` gate.
- **Persistence round-trip** on the fixture: dump → load → re-run retrieval → assert byte-exact metrics (reuses P3.5 Stage 1 harness).

**Pass gate (Stage 2):**
- Aggregate precision > 0.70, recall > 0.70 across ≥10 seeds on the 100-episode fixture.
- Hebbian channel-filtered retrieval beats metadata-grep baseline by `baseline_mean + 2 × baseline_std`.
- Persistence round-trip preserves retrieval F1 within ε=0.01.
- Fast suite + substrate subset + `ruff check` all green.

**Budget 2-3 metric pivots.** Per [feedback_three_iteration_metric_pivot.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_three_iteration_metric_pivot.md) and the P2/P3a retrospectives — the first fixture-based run is likely to surface a tie or near-tie between Hebbian and the grep baseline, requiring a fixture structural refinement (maybe cross-channel nodes need to be more aggressive; maybe the grep baseline needs to include multi-episode joins). The response is rebuild-the-metric, not widen-the-gate.

### Stage 3 — real-data sweep + pre-merge review

**What's built:**

- End-to-end sweep across ≥10 seeds × fixture with full numerical report.
- `docs/experiments/p3b_channel_integration_sweep.md` + `docs/experiments/results/p3b_channel_integration_sweep.json`.
- Reproduction runbook: `docs/experiments/protocols/p3b_channel_integration_reproduction.md`.
- **Pre-merge review round** (Executor + Architecture lenses in parallel). Fold critical + important findings into the same branch before PR opens.

**Pass gate (Stage 3):**
- All 10 seeds pass individually.
- Review round completed with zero outstanding critical findings.
- Substrate subset + fast suite + `ruff check` all green.

**Tests (Stage 3):** Existing Stage 1+2 tests + any regression guards surfacing from the review round.

## Pass criteria (maps to version gate)

Stage 1 + 2 + 3 together close P3b's contribution to 0.3-target. 0.3-target closes when P3a ✅ + P3b + P3.5 + P4 are all `Status: COMPLETE`.

## Load-bearing invariants (filled in AFTER shipping Stage 1)

TODO — populate after Stage 1 review round. Expected candidates based on the audit:

- **Channel-specific rules are a WRAPPING pattern, not a data-structure change.** `channel_specific_rule(channel, inner)` wraps any `BoundaryRule` to gate it on `pending.channel == channel`. The detector's `_rules` list stays flat; per-channel rule sets do NOT get their own collection. This keeps the Stage 1 extension minimal and inherits the existing ordering semantics (all rules commute via `any()`).
- **Channel-filtered retrieval uses `node_filter`, not a new retrieval method.** `Hippocampus.channel_membership_filter(channel, sender=None)` builds a `Callable[[str], bool]` passed to `retrieve_on_cue(node_filter=...)`. This preserves the `retrieve_on_cue` contract shipped in P3a Stage 2 and doesn't create a second retrieval surface P4 would then inherit from.
- **Sender-change rule guards on `sender_id is None`.** A `None` sender from either side is a no-op (doesn't fire). Otherwise the rule fires on any sender who hasn't appeared in the pending episode's `sender_ids` set yet. "First message from Bob after Alice" closes the Alice episode; "anonymous system message" doesn't.
- **Channel-filtered retrieval walks multi-hop.** `retrieve_on_cue(cue, node_filter=..., multi_hop=True)` is the default — the channel filter composes cleanly with `spreading_activation`. One-hop is an explicit opt-in, same as P3a Stage 2.
- **Channel-membership filter recomputes from live `_episode_store` state post-load.** No cached membership maps; the filter is a fresh closure over `hippocampus._episode_store` each time. Load doesn't need special handling — the store round-trips via P3.5 Stage 1 and the filter closure re-resolves.
- **`is not None` for bio-system wire checks, never truthy.** Same regression guard pattern as P3a Stage 1.

## Review questions (Stage 3 reviewers — templates for Round 2 code review)

**Executor lens:**
- Does `sms_sender_change_rule` correctly handle the pending-episode-has-no-senders case (initial capture)? What if `event.sender_id` is the empty string vs `None`?
- Does `channel_specific_rule` correctly handle the case where the pending episode's channel changes mid-episode (shouldn't happen with default rules, but could under a future rule that allows it)?
- Does `channel_membership_filter` have perf issues under 10k+ episodes? Each filter call queries `episodes_containing(node_id)` — that's O(|episodes containing node|). For a hot cue with many episodes, this runs once per multi-hop visit.
- Does the filter correctly handle the EMPTY `sender_ids` case on an episode (a new pending episode that never got a sender)?
- Edge: what if an episode has senders from BOTH matching and non-matching for the filter criterion? Current plan: "any matching sender → include." Is that the right semantics?

**Architecture lens:**
- Is wrapping rules (`channel_specific_rule`) the right shape, or should the detector gain per-channel rule sets (`_rules_by_channel: dict[str, list[BoundaryRule]]`) instead? The wrapping pattern is cheaper but creates a "wrap every rule you want channel-gated" footgun. The data-structure pattern is clearer but requires detector-core changes.
- Does `channel_membership_filter` belong on `Hippocampus` (current plan) or on `EpisodeStore` (where the inverted index lives)? The filter is really a query over episode store state; putting it on Hippocampus creates a thin forwarding method.
- When P4 cross-modal ships, does it build on `node_filter` (current P3b seam) or does it need a different seam (e.g., a per-modality subgraph)? If P4 needs a different seam, does that invalidate P3b's design or coexist with it?
- Does the metadata-grep baseline represent a credible adversary, or is it too weak? The plan's premise is that it's the "what you'd get without a binding graph" baseline. Is there a stronger non-Hebbian alternative (e.g., PageRank over channel-filtered subgraphs)?

## Deferred follow-ups

1. **Real-text channel fixtures** (actual SMS transcripts, actual narrative passages via `LinguisticEncoder`). Stage 2 uses synthetic node IDs with hand-labeled channel metadata. Real text + encoder integration is post-P3b.
2. **Thread_id handling** in `Episode` — reserved since Stage 1 but still unused. P3b might wire it up if threading matters for retrieval; otherwise defer to P8 or beyond.
3. **Multi-channel joint retrieval** — "find nodes that co-occurred with cue across BOTH SMS and narrative." Not in Stage 2 scope; add if a behavioral experiment needs it.
4. **Channel-specific Hebbian tuning** (different `hebbian_init/delta` per channel). SMS might want slower reinforcement than narrative. Deferred until evidence suggests it matters.
5. **`metrics_common.py` extraction** — if Stage 1 is the third consumer of `aggregate_seeds` / `compare_to_baseline`, extract the shared helpers now per the rule of three. Otherwise defer.
6. **Channel change rule refinement** — the existing `channel_change_rule` closes on any channel switch. P3b may add a "soft" variant that allows cross-channel continuation for multi-channel conversations (e.g., SMS → phone call → SMS without starting a new episode). Not in Stage 1.

## Not in this plan

- Anything requiring P4 cross-modal, P5 stress, P6 extinction, P8 sleep replay.
- Integration with the production agent loop (`runtime/agent_loop.py` / `memory_hub.py`). P3b tests wire to Hippocampus directly; runtime wiring is a post-0.3-target integration pass.
- Real-text fixtures via `LinguisticEncoder`. Synthetic throughout P3b Stage 1-3.
- Changes to `similarity/encoder.py`, `decisions/nac.py`, `proprioception/pain_bus.py`, or other P2-shipped surfaces.
- Changes to `ATL.graph`. P3a Stage 1 pivoted Hebbian edges to `Hippocampus._binding_graph`; P3b preserves that.
- Changes to `retrieve_on_cue`'s signature. The `node_filter` seam shipped in P3a Stage 2 fold is exactly what P3b needs; no new kwargs.
- Changes to `DependencyGraph.spreading_activation`'s signature. Same.
