# Substrate P3b — Channel integration: boundary rules + filtered retrieval

**Status:** ✅ Stage 1 SHIPPED (PR #116, 2026-04-14). Stages 2 + 3 (real fixture YAML + cue-aware metadata-grep baseline + 10-seed sweep) explicitly DEFERRED — not version-gating for 0.3-target. Stage 1 alone unblocks P4 via the `episode_membership_filter(membership_mode="exclusive", **criteria)` + `retrieve_on_cue(node_filter=...)` seams.
**Blocks:** P4 cross-modal mug test. **Stage 1 unblocks P4 seam consumption** (`retrieve_on_cue(node_filter=...)` + `EpisodeStore.episode_membership_filter`); **Stage 2 unblocks P4 baseline comparison work** (real-text fixture shape + 10-seed harness).
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
| `channel_gap_rule(channel, max_gap_ticks)` — generalized from earlier `sms_gap_rule` per Round 2 Arch important #2 | ~15 LOC | 1 |
| `sender_change_rule()` (channel-agnostic) with cold-start guard — generalized from `sms_sender_change_rule` per Round 2 Arch important #2 | ~25 LOC | 1 |
| `channel_specific_rule(channel, inner_rule)` wrapper — gates on `event.channel`, NOT `pending.channel` | ~20 LOC | 1 |
| `EpisodeStore.episode_membership_filter(*, membership_mode="any", **criteria)` — snapshot-based, lock-free closure | ~85 LOC | 1 |
| `Hippocampus.episode_filter(*, membership_mode="any", **criteria)` — thin general forwarder (replaces narrow `channel_membership_filter` per Round 2 Arch important #1) | ~10 LOC | 1 |
| Synthetic mechanism tests | ~850 LOC test file | 1 |
| `scenarios/substrate/channel_episodes.yaml` + generator | ~200 LOC generator + YAML | 2 |
| Cue-aware metadata-grep baseline | ~80 LOC | 2 |
| Full metric extractor (defers `metrics_common.py` extraction to P4) | ~50 LOC delta | 2 |
| Fixture validation head-to-head test | ~200 LOC | 2 |
| Results writeup + reproduction protocol | ~300 lines docs | 3 |

## Stages

### Stage 1 — mechanism tests on synthetic geometry

**What's built:**

**1. Boundary rule factories** — all land in `memory/episode.py` (no new `channel_rules.py` module; this is two rule factories + one wrapper, which fits the existing file cleanly):

- `sms_gap_rule(max_gap_ticks=500)` — closes the pending episode when the next event is more than `max_gap_ticks` ticks after the previous capture. SMS-specific because conversations can have minute-to-hour gaps that aren't episode boundaries; narrative gaps are tighter. Default 500 is a tuning placeholder; Stage 2 sweep will refine.

- `sms_sender_change_rule()` — closes when the incoming `event.sender_id` differs from all senders in the pending episode's `sender_ids` set. **Cold-start guard (Round 1 Exec critical #1):** rule returns `False` when `pending.sender_ids` is empty — if the first event had `sender_id=None`, a later event with a non-None sender must NOT trivially fire the rule (empty-set-membership would otherwise return True for any non-None string). Full spec:
  ```python
  def _rule(pending, event):
      if event.sender_id is None or not pending.sender_ids:
          return False
      return event.sender_id not in pending.sender_ids
  ```

- `channel_specific_rule(channel: str, rule: BoundaryRule) -> BoundaryRule` — wraps a rule so it only fires when **`event.channel == channel`**, NOT `pending.channel == channel`. **Round 1 Exec important #2 fix:** the earlier draft gated on `pending.channel`, but that only works because the default `channel_change_rule` is always installed (which means `pending.channel` is guaranteed to match `event.channel` when any rule evaluates). The moment someone removes `channel_change_rule` (reasonable for cross-channel threading), `pending.channel` gating silently disables every channel-specific rule. Gating on `event.channel` is the intended semantic — "this rule applies when the incoming event is on channel X" — and it stays correct under any default-rule configuration.

- **`narrative_scene_rule` is NOT shipped.** It would be a one-line alias over `scn_tag_change_rule()` (Round 1 Exec minor #1). Stage 1 P3b call sites use `scn_tag_change_rule()` directly from `episode.py`. If Stage 2 surfaces a genuine need for narrator-change detection on top of scene-tag change, the factory lands then as substantive code, not an alias.

**2. `EpisodeStore.episode_membership_filter(*, membership_mode="any", **criteria) -> Callable[[str], bool]`** — **moved from `Hippocampus` to `EpisodeStore`** (Round 1 Arch important #3). The filter is a pure query over `EpisodeStore._by_node` + episode metadata; putting it on `Hippocampus` made Hippocampus a filter-builder factory that P4/P5 would keep extending with thin forwarding methods.

- **Generalized filter axes** (Round 1 Arch important #4): takes `**criteria` that introspect `Episode` frozen dataclass fields rather than hard-coding `channel` + `sender`. Callers: `episode_membership_filter(channel="sms")`, `episode_membership_filter(channel="sms", sender="alice")`, and — when P4 cross-modal ships — `episode_membership_filter(modality="vision")` without any new code.

- **Membership mode parameter** (Round 1 Arch critical #1): `membership_mode: Literal["any", "exclusive"] = "any"`. Stage 1 ships `"any"` only (node retained if it appears in ≥1 matching episode). `"exclusive"` (node retained ONLY if every episode containing it matches the filter) is **reserved as a parameter value for P4**. P4's cross-modal mug test will need exclusive-modality filters to answer "is this node a bridge between modalities?" — a question ANY-semantics silently erases. Commit the parameter shape now so P4 can add the mode without breaking P3b callers.

- **`sender` criterion matches via `Episode.sender_ids` set membership** — the filter returns True if the sender appears in any matching episode's `sender_ids` tuple. Handles the empty-`sender_ids` case: an episode whose senders were all None has an empty tuple, and the `sender="alice"` filter never matches it (correct — an episode with no known sender can't be attributed to Alice).

- **Implementation detail:** for each candidate node, call `self.episodes_containing(node_id)`, filter by the criteria, count matches. For `"any"` mode: return True if ≥1 episode matches. For `"exclusive"`: return True if EVERY episode in `episodes_containing(node_id)` matches. The inverted index + per-criterion field comparison is O(|episodes containing node| × |criteria|).

**3. `Hippocampus.channel_membership_filter(channel, sender=None, *, membership_mode="any")`** — thin convenience alias that forwards to `self._episode_store.episode_membership_filter(channel=channel, sender=sender, membership_mode=membership_mode)`. Kept on Hippocampus for ergonomics (callers already have a Hippocampus reference in hand) but the real logic lives on `EpisodeStore`. Total LOC: ~5.

**4. Synthetic mechanism tests** — `tests/substrate/test_p3b_channel_integration.py::TestP3bMechanism` (~280 LOC test file, following the P3a Stage 1 pattern):

- **Rule firing tests:**
  - `test_sms_gap_rule_closes_on_long_gap`: captures at ticks 0, 10, 200 (within 500 gap) vs 0, 10, 600 (> 500 gap). First forms one episode, second closes after tick 10.
  - `test_sms_sender_change_rule_closes_on_new_contact`: capture from alice then alice then bob on SMS; rule fires on bob, closing the alice episode.
  - `test_sms_sender_change_rule_no_op_when_event_sender_is_none`: incoming event has `sender_id=None`; rule does not fire.
  - `test_sms_sender_change_rule_no_op_on_cold_start` (Round 1 Exec C1 regression): first event `sender_id=None` → `pending.sender_ids` stays empty → subsequent event with `sender_id="bob"` must NOT trigger the rule (empty-set cold-start guard).
  - `test_channel_specific_rule_gates_on_event_channel` (Round 1 Exec I2 regression): wrap an always-fire rule with `channel_specific_rule("sms", ...)`, then feed a non-SMS event → rule must NOT fire regardless of what `pending.channel` is. Regression-guards the `event.channel` gating invariant.
  - `test_channel_specific_rule_composes_additively_with_defaults`: wrapped rule + default tick_gap + default channel_change all fire via `any()` — explicit commit to the additive composition model (Round 1 Arch minor #1).

- **Channel-filtered retrieval tests:**
  - `test_channel_filter_returns_only_channel_members_any_mode`: build a fixture with cue=X present in both SMS and narrative episodes. `retrieve_on_cue(X, node_filter=channel_membership_filter("sms"))` returns only nodes from SMS-matching episodes.
  - `test_sender_filter_returns_only_sender_members`: same but filtered by sender.
  - `test_channel_and_sender_filter_compose`: both criteria produce the intersection.
  - `test_channel_filter_empty_channel_returns_empty`: filter on a channel with no episodes returns `[]` (no crash).
  - `test_channel_filter_exclusive_mode_parameter_accepted`: call `episode_membership_filter(channel="sms", membership_mode="exclusive")` — Stage 1 accepts the kwarg (even if Stage 1 semantics are identical to `"any"` on the synthetic fixture). P4 will land the real exclusive-mode implementation. Regression guard: the parameter name is the API contract, not the implementation.
  - **`test_channel_filter_mixed_hop_breaks_transitive_path`** (Round 1 Exec C3 regression): fixture has `cue (SMS)` → `intermediate (narrative-only)` → `target (SMS)`. Under `channel=SMS` filter, `spreading_activation` rejects the intermediate → the BFS stops there → target is NOT retrieved. Test explicitly pins this semantic so P4 / future reviewers know that channel-filtered multi-hop is path-strict, not bridge-transparent.

- **Persistence round-trip tests:**
  - `test_channel_rules_contract_re_registered_on_load` (Round 1 Exec I3 regression): explicit contract — boundary rules are NOT persisted. After `Hippocampus.load_state`, `_episode_detector` contains only rules the NEW instance's `__init__` installed (the defaults). A callsite that added P3b rules via `add_boundary_rule` must re-add them post-load. Test verifies: (a) pre-load behavior with SMS rules added, (b) dump, (c) fresh Hippocampus with DEFAULTS ONLY, (d) load_state, (e) `observe_episode_event` now produces DIFFERENT boundary behavior than pre-dump (the SMS rule is gone). This pins the contract so P3.5 integration work doesn't silently inherit a gap.
  - `test_channel_membership_filter_reconstructs_after_load`: the filter is rebuilt from `episode_store` state post-load; retrieval results on the reloaded store match pre-dump byte-exactly (filter closures re-resolve against the fresh store reference).

- **Capture-thread deadlock regression** (Round 1 Exec minor #2, following the P3a Stage 1 pattern):
  - `test_channel_filter_no_deadlock_under_concurrent_capture`: spawn a background worker that repeatedly fires `observe_episode_event` + `finalize_pending_episode`, while the main thread repeatedly calls `retrieve_on_cue(..., node_filter=channel_membership_filter("sms"))`. Assert completion within a 2-second budget. The filter callback runs inside `spreading_activation`'s graph lock and calls `episodes_containing` inside `EpisodeStore._lock` — verifies the acquire order doesn't reverse.

- **Wire discipline regression** (inherited from P3a pattern):
  - `test_p3b_source_has_no_truthy_biosystem_checks`: AST grep over `memory/episode.py` + `memory/hippocampus.py` P3b-added methods. Reuses the `is not None` invariant.

**5. Metric extractor shell** (`tests/substrate/p3b_metrics.py`, ~50 LOC in Stage 1): imports `aggregate_seeds` + `compare_to_baseline` from `p3a_metrics.py`. **`metrics_common.py` extraction is explicitly deferred to P4** (Round 1 Exec minor #3, Round 1 Arch minor #3). Rationale: two plans currently share the helpers (P2, P3a); P3b makes it three but P4 is the natural consolidation point because P4 will ship the 1.0-gating comparison shape and deserves to drive the shared module's API. Folding the extraction into P3b now would create a rush decision on an API that has to serve P4. Flag with a TODO comment in `p3b_metrics.py` pointing at the P4 entry condition.

**Pass gate (Stage 1):**
- All 13+ synthetic mechanism tests in `TestP3bMechanism` pass.
- `ruff check` + `ruff format` clean on all touched files.
- No truthy bio-system checks in the P3b diff (AST regression guard).
- Fast suite clean (standing exclusions per CLAUDE.md).
- Substrate subset clean: `PYTHONPATH=src python -m pytest tests/substrate/ tests/unit/test_bio_system_snapshot.py -q`.
- P3a Stage 1+2 tests + P3.5 Stage 1 tests all remain green (zero regression).

**Tests (Stage 1):** See above. No fixture YAML, no baseline, no 10-seed sweep — those are Stage 2.

### Stage 2 — fixture-based validation + cue-aware metadata baseline

**Baseline-first design discipline (Round 1 Arch I4 + Exec I1 cross-confirmed):** the Stage 2 baseline **contract is specified before the fixture is built**, not after. Both reviewers independently flagged that the earlier "grep matching episodes, return all nodes" draft was strictly weaker than Hebbian one-hop — it would clear the `+2σ` gate by construction because Hebbian one-hop returns cue-co-occurring nodes while the baseline returns arbitrary nodes from matching episodes, guaranteeing a precision gap unrelated to the mechanism's actual claim. Fixing this by rebuilding the fixture around the wrong baseline would be the textbook circular-head-to-head trap from the P3a Stage 2 clique-topology finding.

**What's built:**

- **Cue-aware metadata-grep baseline** in `tests/substrate/metadata_grep_baseline.py` (~80 LOC): the baseline returns **nodes that co-occurred with the cue in episodes matching the channel/sender criteria, one-hop only, no graph walk**. Concretely: look up episodes containing the cue (via `EpisodeStore._by_node[cue]`), filter those episodes by the metadata criteria, union the `activated_nodes` across the filtered episodes, return the union minus the cue itself. This is the correct "what you'd get with just episode metadata indexing + one-hop co-occurrence" baseline — a credible adversary for Hebbian multi-hop filtered retrieval.

- **Two head-to-heads in Stage 2**, not one. The pass gate requires BOTH to clear:
  1. **Hebbian multi-hop filtered vs cue-aware metadata-grep baseline** — measures whether the multi-hop binding-graph walk adds signal beyond one-hop co-occurrence within the same channel filter. This is the real architectural claim P3b is making.
  2. **Hebbian multi-hop filtered vs Hebbian one-hop filtered** — measures whether multi-hop lift is still ≥ 0.20 when both sides use the channel filter. This is the P3a Stage 2 lift invariant applied to the filtered subgraph. If multi-hop filtered == one-hop filtered, the channel filter is preventing the graph walk from reaching the transitive structure the mechanism depends on — and that's a real problem P3b would need to surface, not hide.

- `scenarios/substrate/channel_episodes.yaml` — ~100 episodes across SMS + narrative channels with labeled sender metadata + ground truth co-occurrence. Generator specified AFTER the baseline is written so the topology can't be tuned to flatter the baseline. Hub+chain topology per (channel, sender) pair with enough cross-channel nodes that the channel filter has a meaningful effect but not so many that the filter becomes trivial. ~1-2 days authoring per the parent plan.

- **Full metric extractor** in `p3b_metrics.py`: per-seed precision/recall/F1, aggregate mean+std across seeds, `compare_to_baseline` reused from `p3a_metrics.py`. `metrics_common.py` extraction deferred to P4 (see Stage 1 note).

- **Fixture validation test** `tests/substrate/test_p3b_fixture_validation.py::TestP3bFixture` — 10-seed head-to-head. Asserts BOTH head-to-heads clear.

- **Variance source:** per-seed episode dropout (10%), matching the P3a Stage 2 pattern. Non-ceremonial `+2σ` gate.

- **Persistence round-trip** on the fixture: dump → load + re-register channel rules at new-instance construction + re-run retrieval → assert byte-exact metrics. The rule-re-registration step pins the Round 1 Exec I3 contract from Stage 1.

**Pass gate (Stage 2):**
- Aggregate precision > 0.70, recall > 0.70 across ≥10 seeds on the 100-episode fixture.
- **Hebbian multi-hop filtered beats cue-aware metadata-grep baseline** by `baseline_mean + 2 × baseline_std` on F1.
- **Hebbian multi-hop filtered beats Hebbian one-hop filtered** by ≥ 0.20 absolute F1 lift (the same architectural invariant P3a Stage 2 locked in, now applied to the filtered subgraph).
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

## Load-bearing invariants (post-Round-2-review fold)

Surfaced by the plan-level Round 1 pre-merge review (Executor lens + Architecture lens) before any code landed AND the code-level Round 2 review (Executor lens + Architecture lens) on the Stage 1 implementation. Both rounds folded their critical + important findings into the plan and the code before this commit.

**Rule factories and detector composition:**

- **Channel-specific rules are a WRAPPING pattern, not a data-structure change.** `channel_specific_rule(channel, inner)` wraps any `BoundaryRule` to gate it on **`event.channel == channel`** (NOT `pending.channel` — Round 1 Exec important #2). Gating on `event.channel` is correct under any default-rule configuration; `pending.channel` gating silently disables every wrapped rule the moment `channel_change_rule` is removed from the defaults.
- **Additive rule composition is the committed default.** P3b-registered rules run alongside the P3a defaults (`tick_gap_rule`, `channel_change_rule`, `scn_tag_change_rule`); they do NOT replace them. All rules compose via `EpisodeBoundaryDetector.should_close = any(rule(...) for rule in self._rules)`. Rules commute — insertion order does not affect the close decision. Regression guard: `test_channel_specific_rule_composes_additively_with_defaults`.
- **Sender-change rule has a cold-start guard.** `sms_sender_change_rule` returns `False` when `pending.sender_ids` is empty. Without this guard, a cold-start episode opened by an event with `sender_id=None` would close the moment any non-None sender arrived, because `"bob" not in empty_set` is trivially True. Round 1 Exec critical #1 regression: `test_sms_sender_change_rule_no_op_on_cold_start`.
- **No `narrative_scene_rule` alias.** Callers use the existing `scn_tag_change_rule()` directly. Stage 1 does NOT ship a rename-only alias (Round 1 Exec minor #1). If Stage 2 or P3c needs narrator-change detection, the factory lands then as substantive code.

**Filter shape and placement:**

- **`episode_membership_filter` lives on `EpisodeStore`, not `Hippocampus`** (Round 1 Arch important #3). `EpisodeStore` owns the inverted index the filter queries; `Hippocampus.channel_membership_filter` is a thin convenience alias that forwards to it. When P4 cross-modal adds `episode_membership_filter(modality=...)`, it lands in `EpisodeStore` — NOT as another forwarding method on `Hippocampus`.
- **Filter axes are introspected from `Episode` dataclass fields, not hard-coded.** `episode_membership_filter(*, membership_mode="any", **criteria)` accepts any combination of Episode field matchers (`channel="sms"`, `sender="alice"`, and — after P4 — `modality="vision"`). Adding a new filter axis is a zero-LOC P3b change when P4 ships (Round 1 Arch important #4).
- **`membership_mode: Literal["any", "exclusive"] = "any"`** is a committed parameter from Stage 1 even though Stage 1 only implements `"any"` (Round 1 Arch critical #1). P4 cross-modal will add `"exclusive"` semantics without breaking P3b callers. A Stage 1 test passes `membership_mode="exclusive"` to verify the parameter is accepted by the API even though Stage 1 semantics are identical to `"any"` on the synthetic fixture.
- **Channel-filtered multi-hop is path-strict, not bridge-transparent** (Round 1 Exec critical #3). `spreading_activation(node_filter=...)` rejects filtered nodes as both sources AND hop targets → a BFS path through a filtered-out intermediate node breaks at that intermediate; downstream nodes that would be reachable through the intermediate become unreachable even if they pass the filter themselves. This is the intended semantic for channel-segregated retrieval, not a bug. Regression guard: `test_channel_filter_mixed_hop_breaks_transitive_path`.
- **`node_filter` is the ONLY retrieval filter seam.** P3b commits to using `retrieve_on_cue(cue, node_filter=...)` exclusively — no parallel retrieval methods (`retrieve_on_cue_in_channel`, `retrieve_on_cue_channel_aware`, etc.). **STOP-and-escalate rule**: if Stage 2 fixture validation surfaces a case where `node_filter` is insufficient, do not add a parallel retrieval surface inline — open a discussion + flag it as a P4 blocker so the two plans can pick a unified shape (Round 1 Arch minor #3).

**Persistence and state:**

- **Boundary rules are NOT persisted; callers MUST re-register them at construction time post-load** (Round 1 Exec important #3). `Hippocampus.load_state` restores the `_episode_store` (via P3.5 Stage 1 rebuild-from-episodes) but not `_episode_detector._rules`. Downstream code that adds P3b rules via `add_boundary_rule` must re-add them after each `load_state` call — or, cleaner, add them at `__init__` time via an explicit `boundary_rules` constructor kwarg (consider for Stage 2 if the gap bites in practice). Regression guard: `test_boundary_rules_NOT_persisted_post_load` (uses a baseline-snapshot of the default rule count rather than a hard-coded number, per Round 2 Arch minor #1).
- **`is not None` for bio-system wire checks, never truthy.** Same regression guard pattern as P3a Stage 1.

**Round 2 fold — load-bearing invariants:**

- **`episode_membership_filter` is SNAPSHOT-BASED, not lazy** (Round 2 self-found critical + Exec critical, cross-confirmed). The filter walks `self._episodes` ONCE under `self._lock` at build time and produces a frozen set of allowed node ids. The returned closure is a lock-free `node_id in allowed_set` check. This is **load-bearing for thread safety**: `DependencyGraph.spreading_activation` holds the binding graph's lock for the entire BFS and invokes `node_filter` while it's held. A lazy filter that re-acquired `EpisodeStore._lock` per call would deadlock against `Hippocampus._close_pending_episode_locked` (which acquires store lock first via `store.add(episode)` then binding graph lock via `apply_hebbian_on_close`). Snapshot side effect: episodes added AFTER filter construction are NOT reflected — that's the right retrieval semantics (consistent point-in-time view, not mutation mid-traversal). Regression guards: `test_filter_closure_holds_no_lock_after_construction` (spawns a side thread holding `EpisodeStore._lock` and verifies the filter call returns in <0.5s) + `test_filter_snapshot_does_not_see_episodes_added_after_construction`.
- **Unknown criterion field names raise `ValueError` at filter build time** (Round 2 cross-confirmed critical, Exec important #1 + Arch critical #1). Pre-fold: typos like `channnel="sms"` or unknown axes like `modality="vision"` (before P4 lands `Episode.modality`) silently returned empty filters because `getattr(episode, key, None)` + the None short-circuit conflated typos with missing fields. The consequence cited by Arch was sharp: "P4 cross-modal mug test would pass at 0% collapse with an empty filter and nobody would know why." Fix: enumerate `dataclasses.fields(Episode)` once at filter build and raise on unknown keys.
- **Collection-typed criterion VALUES raise `TypeError`** (Round 2 Exec important #3). Pre-fold: passing `sender_ids=("alice", "bob")` (intent: "any of these senders") silently returned nothing because `tuple in tuple` checks element membership, not subset. Fix: type-check at filter build and raise. Subset / set-intersection semantics are explicitly deferred to a future plan.
- **`field_value is None` is a legitimate scalar match, not a no-match short-circuit** (Round 2 Arch critical #2). Pre-fold: a filter `thread_id=None` (intent: "episodes with no thread") always returned False because the None check bailed before the equality check. Fix: split typo validation from None handling — once the typo check has passed, let `field_value != value` do the work, so `None == None` succeeds. Regression guard: `test_filter_matches_legitimate_none_scalar_field`.
- **`exclusive` mode + zero containing episodes returns False** (Round 2 Exec important #2). Pinned semantic: a node with no containing episodes has "no evidence either way" and is excluded under `exclusive` mode. The vacuous-truth alternative ("every (zero) episode matches") was rejected as practically wrong. Regression guard: `test_exclusive_mode_zero_episodes_returns_false`.
- **`exclusive` mode is exercised with a 2-criteria mug-shape test in Stage 1** (Round 2 Arch important #4). Uses `scn_tag` as a stand-in for the P4 `modality` field that doesn't exist yet, validating the `**criteria` cross-criteria pattern works under `exclusive` mode. Regression guard: `test_exclusive_mode_with_two_criteria_cross_mug_shape`.

**Naming and shape (Round 2 Arch fold):**

- **`Hippocampus.episode_filter(**criteria)` is the general forwarder** (Round 2 Arch important #1). Replaces the earlier `channel_membership_filter(channel, sender)` per-axis alias which would have forced P4 to add `modality_membership_filter`, P5 to add `stress_membership_filter`, etc. — N per-axis aliases each requiring sync with the underlying signature. The general `**criteria` form takes any combination of `Episode` field matchers and inherits future axes for free. Common-case ergonomics are preserved via plain kwargs: `h.episode_filter(channel="sms", sender_ids="alice")`.
- **`channel_gap_rule(channel, max_gap_ticks)` and channel-agnostic `sender_change_rule()` replace the earlier `sms_gap_rule` / `sms_sender_change_rule`** (Round 2 Arch important #2). The earlier `sms_*` factories hard-coded the channel string three layers deep and would have spawned a per-channel rule zoo as P5/P6 added email / slack / voice channels. The renamed factories take the channel as a parameter (or omit it entirely for `sender_change_rule()`, leaving callers to compose with `channel_specific_rule` if they want gating).
- **`channel_specific_rule` correctness is regression-guarded WITHOUT the default `channel_change_rule`** (Round 2 Arch minor #3). The earlier draft test ran with the full default rule set installed, which masked the difference between event-channel and pending-channel gating. The Round 2 fold adds `test_wrapper_correct_in_isolation_without_default_rules` that builds a detector with ONLY the wrapped rule and verifies it still gates correctly on `event.channel`.

**Per-rule O(1) perf invariant (Round 2 Arch important #3, deferred to P5):**

- **`EpisodeBoundaryDetector.should_close` evaluates every installed rule on every event.** A Slack event pays the SMS-rule cost. At Stage 1's 3-5 rules this is irrelevant; at P5's anticipated channel count (~8-10) it's still cheap. P8 sleep replay pumps O(episodes) events through the detector in a batch — that's where quadratic behavior would bite. **Rule authors:** keep boundary rules to O(1) work per event. P5 should consider a per-channel rule index if the rule list crosses ~8-10 entries. Documented in `EpisodeBoundaryDetector` class docstring.

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

## Deferred concerns flagged by Round 1 review (documented for downstream plans)

These are P4/P6/P8 entry conditions surfaced by the Round 1 plan review. Not folded into P3b code — documented here so the next session picking them up inherits the constraint.

- **`exclusive` membership mode is a P4 entry condition** (Round 1 Arch critical #1). Stage 1 ships the parameter shape but implements only `"any"`. P4 cross-modal mug test will need `"exclusive"` semantics to answer "is this node a bridge between modalities?" — the ANY-mode filter retains bridge nodes under either side's filter, which is correct for channel isolation but wrong for identifying cross-modal pivots. P4 entry condition: implement `membership_mode="exclusive"` in `EpisodeStore.episode_membership_filter` before opening the mug-test fixture.
- **P8 sleep replay × channel-rule interaction** (Round 1 Arch important #3). P3b adds channel-aware boundary rules. P8 sleep replay re-runs closed episodes through `apply_hebbian_on_close`. **The open question:** does replay feed events back through the boundary detector (re-firing channel rules and potentially re-segmenting episodes), or does it bypass the detector entirely and replay already-closed `Episode` objects? **P8 entry condition:** pick one explicitly before implementing replay. If going through the detector: channel rules must be replay-idempotent (re-running the same event sequence produces the same boundaries). If bypassing: replay needs its own path into `apply_hebbian_on_close` that doesn't touch `_episode_detector`.
- **Boundary provenance for P6 extinction** (Round 1 Arch critical #2). P3b's rule-wrapping pattern composes via `any()` — the detector can tell that SOME rule fired but NOT which one. If P6 extinction later wants to decay channel-specific boundaries differently from generic ones (e.g., "SMS-sender-change boundaries decay faster than tick-gap boundaries"), the plan needs `Episode.closed_by: str | None` or similar provenance metadata. **P6 entry condition:** either extend `Episode` with a `closed_by` tag and have rules self-identify, OR document that extinction operates exclusively at the binding-graph edge level and never cares about boundary provenance. Either decision is fine; silence is not. P3b does NOT ship the provenance field — P3b's architectural claim is edge-level Hebbian, not boundary-level.

## Previously-deferred items (unchanged from initial draft)

1. **Real-text channel fixtures** (actual SMS transcripts, actual narrative passages via `LinguisticEncoder`). Stage 2 uses synthetic node IDs with hand-labeled channel metadata. Real text + encoder integration is post-P3b.
2. **Thread_id handling** in `Episode` — reserved since Stage 1 but still unused. P3b might wire it up if threading matters for retrieval; otherwise defer to P8 or beyond.
3. **Multi-channel joint retrieval** — "find nodes that co-occurred with cue across BOTH SMS and narrative." Not in Stage 2 scope; add if a behavioral experiment needs it.
4. **Channel-specific Hebbian tuning** (different `hebbian_init/delta` per channel). SMS might want slower reinforcement than narrative. Deferred until evidence suggests it matters.
5. **`metrics_common.py` extraction.** Round 1 Exec minor #3 + Round 1 Arch minor #3 both flagged. Decision: **defer to P4** — P4 is the 1.0-gating comparison shape and deserves to drive the shared module's API. Stage 1 `p3b_metrics.py` imports from `p3a_metrics.py`; a TODO comment in both files points at the P4 entry condition.
6. **Channel change rule refinement** — the existing `channel_change_rule` closes on any channel switch. P3b may add a "soft" variant that allows cross-channel continuation for multi-channel conversations (e.g., SMS → phone call → SMS without starting a new episode). Not in Stage 1.
7. **PageRank-over-channel-filtered-subgraph baseline** — Round 1 Arch important #4 asked whether metadata-grep is a strong enough adversary. Stage 2 ships the cue-aware metadata-grep baseline (folded) as the canonical head-to-head, AND adds a second head-to-head (Hebbian multi-hop filtered vs Hebbian one-hop filtered) to prove the multi-hop lift is real within the filtered subgraph. A third head-to-head against a PageRank baseline is an interesting open question but explicitly deferred to P4 or later — two baselines in Stage 2 is enough to clear the pass gate without becoming a baseline-zoo.

## Not in this plan

- Anything requiring P4 cross-modal, P5 stress, P6 extinction, P8 sleep replay.
- Integration with the production agent loop (`runtime/agent_loop.py` / `memory_hub.py`). P3b tests wire to Hippocampus directly; runtime wiring is a post-0.3-target integration pass.
- Real-text fixtures via `LinguisticEncoder`. Synthetic throughout P3b Stage 1-3.
- Changes to `similarity/encoder.py`, `decisions/nac.py`, `proprioception/pain_bus.py`, or other P2-shipped surfaces.
- Changes to `ATL.graph`. P3a Stage 1 pivoted Hebbian edges to `Hippocampus._binding_graph`; P3b preserves that.
- Changes to `retrieve_on_cue`'s signature. The `node_filter` seam shipped in P3a Stage 2 fold is exactly what P3b needs; no new kwargs.
- Changes to `DependencyGraph.spreading_activation`'s signature. Same.
