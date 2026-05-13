# Cross-modal substrate binding via temporal co-activation

**Target version:** 1.1 (post-1.0)
**Status:** Active. Plan written 2026-05-13.
**Owns:** [similarity/ec.py](../../src/maxim/similarity/ec.py), [time/oscillator.py](../../src/maxim/time/oscillator.py), [time/scn.py](../../src/maxim/time/scn.py), [decisions/nac.py](../../src/maxim/decisions/nac.py) (consumer-side), [decisions/temporal_credit.py](../../src/maxim/decisions/temporal_credit.py)
**Companion plans:** [release_0_9_1.md](release_0_9_1.md) (ships Wire-A as the interim; this plan is the structural fix Wire-A surfaces around), [grounded_language_acquisition.md](grounded_language_acquisition.md) (Phase 1's `token_id → ec_node_id` symbol-binding registry is *populated* by the substrate-binding edges this plan ships; the two plans are complementary, not equivalent — see "Relationship to grounded_language_acquisition" below), [persona_convergence_crucible.md](persona_convergence_crucible.md) (Roy-4 is the experimental prerequisite; Roy-5+ are the validation iterations)

## Why this plan exists

Roy-2c (PR #244, [docs/experiments/20_roy_2c.md](../experiments/20_roy_2c.md)) confirmed **H1: LinguisticEncoder → EC alignment failure** is the load-bearing block on the cluster_reward_bias path. The priming substrate's WMS contents (sensor/drive state + cradle narrator output) embed into one EC region; CLI test percepts ("you sense food nearby") embed into a structurally disjoint EC region — even when humans read the semantic overlap as obvious. The `cluster_reward_bias` map has the right *tool* keys (`sense_food_source`) but the wrong *cluster* keys, so `recommend_action` never finds the bias on engineered-overlap test percepts.

Wire-A (0.9.1 Stage 2) routes around this by surfacing tool-level bias to the LLM prompt — bypassing the EC retrieval path entirely. That ships in 0.9.1 as the interim because it gives us substrate-attributable behavior immediately, but **Wire-A is static signal-surfacing, not dynamic signal-formation.** It cannot teach the substrate new cross-modal associations; it can only render associations the substrate already encoded (via tool-name keys that survive the encoder gap).

This plan ships the structural fix: **EC nodes that activate in the same tick window across modalities acquire a Hebbian binding edge.** When `hunger drive high` + `food entity sensed` + narrator text "food" co-fire during priming, the EC nodes for each modality bind. Later test-time activation of any one node propagates through the binding edges to the others, lighting up the cluster_reward_bias path even when the encoder embeds the test percept into a different region than priming.

This is bio-inspired in the same shape real brains use for cross-modal binding (visual + auditory neurons in superior colliculus, sensory + motor in M1). It composes with existing infrastructure: B2's `OscillatorNetwork` + `_temporal_anchors` (the first SCN-substrate PoC, documented in [CLAUDE.md](../../CLAUDE.md)), NAc's `cluster_reward_bias`, EC's `pattern_complete_or_separate`. The infrastructure is mostly in place — the binding mechanism is what's missing.

## Framing rule

**The binding mechanism ships only if a prior instrumentation experiment (Roy-4) confirms that priming-cluster ↔ test-cluster pairs WOULD have been linked under a proposed Hebbian rule.** Skipping the validation and shipping the mechanism risks landing a binding rule that over-fires or under-fires and only discovering it months later when persona behavior doesn't converge. The architectural shape is the easy part; the binding criterion is where the design lives.

## Relationship to grounded_language_acquisition

The two plans address **adjacent layers** of the same bio-grounded language problem:

- **This plan (substrate-level binding).** Operates inside EC: which EC nodes bind to which other EC nodes via Hebbian co-activation. The output is a graph over EC node IDs with binding-strength edges. Modality-agnostic — a sensor cluster can bind to a drive cluster can bind to a linguistic cluster as long as they co-fire.
- **Grounded language Phase 1 (token-level binding).** Operates on the LLM logit space: which LLM output tokens are allowed to fire based on which EC nodes are bound to them. The `token_id → ec_node_id` symbol-binding registry that Phase 1 needs is **a special case of this plan's binding edges** — specifically, the subset of edges where one endpoint is a linguistic-modality EC node tagged with its source token.

Phase 1's symbol-binding registry can be implemented as a lookup over this plan's `_binding_edges`: for each token, find its linguistic-modality EC node, then enumerate the bound EC nodes across all modalities to determine "is this token grounded in substrate experience?" Without this plan's mechanism, Phase 1 has to invent its own co-occurrence learner (which the plan currently sketches but doesn't depend on a specific mechanism for); with this plan, Phase 1's co-occurrence learner *is* the EC binding update path, just queried at token granularity.

**The plans do not block each other.** This plan ships in 1.1 regardless of Phase 1's status; Phase 1 ships independently in 1.1+ and queries this plan's edges when available. Phase 1 can fall back to its own co-occurrence learner if this plan slips, at the cost of building duplicate machinery — coordinating the two during 1.1 is mostly about ensuring Phase 1's registry consumes this plan's edge format rather than re-implementing co-occurrence tracking.

## Tick-discrete temporal semantics (the neuroscience nit)

Real-brain cross-modal binding happens on ~25-100ms gamma cycles. Maxim's SCN is **circadian phase**, not millisecond-scale binding — it tracks where in the daily cycle an event fired (`_event_phases` in [time/oscillator.py:94](../../src/maxim/time/oscillator.py#L94)). Maxim's equivalent of gamma binding is **same-tick or adjacent-tick co-activation**. SCN contributes longer-horizon phase context (this happened during the "post-meal" phase) but moment-to-moment binding is tick-level.

The existing `_temporal_anchors` dict in [decisions/nac.py:183](../../src/maxim/decisions/nac.py#L183) — `dict[tuple[agent_id, node_id], tuple[activation, TemporalSignature]]` — already does this shape for *credit assignment*: when fast-decay eligibility traces expire, `distribute_reward` falls back to temporal-similarity-based credit. This plan extends the same intuition from credit-assignment-via-coincidence to **edge-formation-via-coincidence**. Same mechanism, different consumer.

## Sizing

| Stage | Item | LOC | Where |
|---|---|---|---|
| 1 | Roy-4 instrumentation experiment (prerequisite — runs in 0.9.1 dev cycle, validates 1.1 design) | ~50 | scenarios/roy/, docs/experiments/, EC instrumentation hooks |
| 2 | EC binding edge data structure + update path | ~250 | similarity/ec.py |
| 3 | Binding rule design + decay + salience-weighting | ~200 | similarity/ec.py |
| 4 | Bound-edge consumption via `pattern_complete_or_separate` | ~150 | similarity/ec.py + decisions/nac.py |
| 5 | NAc `recommend_action` consults bound neighborhood (not just active cluster) | ~100 | decisions/nac.py |
| 6 | Persistence (binding edges survive session restart) | ~80 | similarity/ec.py + integration/snapshot.py |
| 7 | Roy-5+ validation iterations | ~30 each | scenarios/roy/, docs/experiments/ |
| **Total 1.1 implementation (Stages 2-6)** | | **~780** | |

Estimated calendar: 5-8 days for Stages 2-6 implementation including pre-merge two-lens reviews. Roy-4 (Stage 1) is one session and lands during 0.9.1's development.

## Stage 1 — Roy-4 instrumentation experiment (0.9.1-cycle prereq)

**Purpose:** validate that priming-cluster ↔ test-cluster pairs *would* have been linked under a proposed Hebbian rule. Without this validation, the implementation work risks shipping a binding rule that doesn't actually fix the Roy-2c gap.

**Setup:** identical to Roy-2c (multi-arc priming + engineered-overlap fixture + substrate-primary at test), plus one instrumentation addition: emit a per-tick JSONL event `sim_ec_activation` containing the active node IDs from every `pattern_complete_or_separate` call. This is a temporary instrumentation hook — production EC doesn't emit it.

**Analysis:** post-hoc compute a pairwise co-activation matrix across the priming session. For each pair `(node_a, node_b)` where both fired in the same tick at least N times during priming, mark them as "would-have-bound." Then for each test-phase tick, list the active nodes and check whether any of them are in the would-have-bound neighborhood of a priming cluster. **Pass criteria:** for at least one test percept in arms A (primed), the active node has a would-have-bound edge to a priming `sense_food_source` cluster. If yes → Hebbian binding would have closed the Roy-2c gap; Stage 2 implementation is justified. If no → the priming and test percepts genuinely never co-fire even with future binding; a deeper encoder fix is needed instead.

**Owns:** `scenarios/roy/roy_4_iteration.yaml` (instrumented Roy-2c re-run), `docs/experiments/21_roy_4.md` (outcome doc), an EC instrumentation env-var (`MAXIM_EC_TRACE_ACTIVATIONS=1`) to gate the per-tick events behind opt-in.

**Sizing:** ~50 LOC (instrumentation hook + env-var gate + autouse conftest scrub). Single-session experiment after the hook ships.

## Stage 2 — Binding edge data structure

**Implementation:**
- New `EC._binding_edges: dict[tuple[str, str], float]` keyed by `(node_id_a, node_id_b)` (canonically ordered so `(a, b) == (b, a)` collapses). Value is the binding strength in `[0.0, 1.0]`.
- New method `EC.update_binding(active_node_ids: list[str], dt: float, *, agent_id: str) -> None`:
  - Called from every site that ends a tick with active EC nodes (after `pattern_complete_or_separate`).
  - For every pair `(a, b)` in `active_node_ids`, increment `_binding_edges[(a, b)]` by `dt * binding_rate`.
  - Apply per-tick decay to all edges: `_binding_edges[k] *= decay_factor`. Edges below `prune_threshold` get deleted.
- New `EC._binding_rate: float` (default 0.02 per tick) and `_decay_factor: float` (default 0.999 per tick). Stage 3 tunes these based on Roy-4 findings.
- Per-agent keying via the same `agent_id` rule as `_cluster_reward_bias` (CLAUDE.md "Per-agent stash dicts" rule). Forgetting `agent_id` is a `TypeError`, not a silent no-op.

**Frozen contract impact:** `EC._binding_edges` is mutable internal state. No new persisted dataclass. Persistence in Stage 6.

**Test surface:**
- Unit: `update_binding` with synthetic active-node lists produces expected `_binding_edges` deltas after N ticks.
- Unit: decay reduces all edge weights uniformly; edges below threshold are pruned.
- Multi-agent: two agents sharing one EC instance maintain disjoint binding edges per `agent_id`.

## Stage 3 — Binding rule design (decay + salience-weighting)

**The hard part is the binding criterion, not the architecture.** Three failure modes to defend against:

1. **Pavlov-on-steroids (over-binding):** every percept fires every association because spurious co-firings accumulated. Defense: aggressive decay (`decay_factor = 0.999/tick` ≈ 5-minute half-life at 2Hz) + min-co-firing-count threshold before edge is considered "bound" (default 5).
2. **Under-binding:** real associations never accumulate enough weight to cross the consumption threshold. Defense: `binding_rate` tuned against Roy-4 findings (Stage 1 measures the actual co-firing frequency for true positives).
3. **Salience hijacking:** weakly-active nodes (low pattern completion confidence) over-bind because every tick they happen to be active. Defense: weight the binding update by min(activation_a, activation_b) — only bind strongly-co-active nodes.

**Implementation:**
- `update_binding` accepts `activations: dict[str, float]` not just `list[str]`. Increment becomes `dt * binding_rate * min(activations[a], activations[b])`.
- `_binding_edges` values capped at `max_binding_strength` (default 1.0).
- `EC.get_bound_neighbors(node_id: str, *, agent_id: str, min_strength: float = 0.3) -> dict[str, float]` returns nodes bound to `node_id` above threshold, with their strengths.
- Env-var overrides for all four tunables (`MAXIM_EC_BINDING_RATE`, `MAXIM_EC_BINDING_DECAY`, `MAXIM_EC_BINDING_MIN_STRENGTH`, `MAXIM_EC_BINDING_PRUNE_THRESHOLD`) so Roy iterations can sweep without code edits. Pair each with a conftest scrub.

**Test surface:**
- Unit: low-activation co-firing doesn't bind; high-activation does.
- Unit: 5+ co-firing events crosses min-co-firing-count threshold; 4 doesn't.
- Unit: decay over 1000 ticks of no co-firing prunes the edge below threshold.

## Stage 4 — Bound-edge consumption via `pattern_complete_or_separate`

**Implementation:**
- Extend `pattern_complete_or_separate` to return `(active_node_id, bound_neighbors: dict[str, float])` instead of just `active_node_id`.
- Existing callers that read just the node_id continue to work via tuple unpacking; the new field is additive.
- Caller's choice whether to propagate downstream — `NAc.recommend_action` opts in (Stage 5); other consumers (Hippocampus retrieval) remain on the active-node-only path.

**Frozen contract impact:** `PatternCompletionResult` (if it's a dataclass) gets an additive `bound_neighbors: dict[str, float] = field(default_factory=dict)` field per CC3 rules. Audit gate: docstring update.

**Test surface:**
- Unit: callers that don't read bound_neighbors continue to work.
- Unit: bound_neighbors contains only edges above `min_strength`.

## Stage 5 — NAc `recommend_action` consults bound neighborhood

**The Roy-2c-specific fix at the architectural level.** When `recommend_action` looks up `cluster_reward_bias[agent_id][current_cluster_id]` and finds it empty (the H1 failure mode), it now also looks up `cluster_reward_bias` for every node in `bound_neighbors`. The aggregation rule:

- For each tool, take the max bias across the active cluster AND its bound neighborhood, weighted by binding strength: `weighted_bias = max(bias[active] for tools) + sum(strength[n] * bias[n][tool] for n in bound)`.
- Cap the weighted bias at `max_cluster_reward_bias` to prevent runaway under high-binding-strength regimes.

This is the consumer-side mechanism that lets the cluster wire express behaviorally on cross-modal percepts.

**Implementation:**
- Modify `NAc.recommend_action` signature to optionally accept `bound_neighbors: dict[str, float]`.
- The substrate-primary call site at [agent_loop.py::propose_via_substrate](../../src/maxim/runtime/agent_loop.py) passes the bound neighborhood from the `pattern_complete_or_separate` result.
- The llm-primary path doesn't call `recommend_action`, so Wire-A (0.9.1 Stage 2) remains the surface for llm-primary's substrate visibility — bound-edge consumption is a substrate-primary-only fix.

**Test surface:**
- Unit: empty `cluster_reward_bias` on active cluster + non-empty bias on a bound neighbor produces a proposal weighted by binding strength.
- Integration: a sim where priming activates cluster A and test activates cluster B with `_binding_edges[(A, B)] = 0.5` produces a `recommend_action` result that reads A's cluster_reward_bias scaled by 0.5.

## Stage 6 — Persistence

`_binding_edges` survives session restart. Critical for the cross-session learning claim 1.0 gates on.

**Implementation:**
- Extend `EC.dump()` / `load_state()` to round-trip `_binding_edges`.
- `_format_version` bump on EC snapshot. Backward-compat reader: missing field → empty dict.
- Pre-merge review must verify a session restart preserves binding edges and they continue to decay correctly after reload (clock-skew test).

## Stage 7 — Roy-5+ validation iterations

Roy-5 runs the same Roy-2c spec on top of the binding mechanism. Pre-registered diagnostic:
- **Arm A produces `sense_food_source` calls in the test phase** (the result Roy-2c could not produce) → binding mechanism closes the gap.
- **Arm A still produces only `infant_humanoid_pick_up`** → binding rule needs retuning OR the priming co-activation pattern doesn't include the expected pairs OR there's another structural block.

Roy-5 is a single-session ablation: re-run with `MAXIM_EC_BINDING_RATE=0.0` to disable binding entirely and confirm we reproduce Roy-2c's identical-distribution outcome under the same code.

Roy-6+ explore parameter sweeps and longer priming horizons.

## Cross-cutting: env-var inventory

| Env var | Stage | Default | Purpose |
|---|---|---|---|
| `MAXIM_EC_TRACE_ACTIVATIONS` | 1 | unset → disabled | Per-tick `sim_ec_activation` JSONL emission for Roy-4 |
| `MAXIM_EC_BINDING_RATE` | 3 | unset → 0.02 | Per-tick co-firing weight increment |
| `MAXIM_EC_BINDING_DECAY` | 3 | unset → 0.999 | Per-tick decay factor on all edges |
| `MAXIM_EC_BINDING_MIN_STRENGTH` | 3 | unset → 0.3 | Threshold for `get_bound_neighbors` inclusion |
| `MAXIM_EC_BINDING_PRUNE_THRESHOLD` | 3 | unset → 0.05 | Below this, edges are deleted |

All paired with conftest autouse scrubs per [feedback_opt_in_env_in_hot_paths.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_opt_in_env_in_hot_paths.md).

## Cross-cutting: frozen contract impact

- `PatternCompletionResult` (or equivalent) gets additive `bound_neighbors: dict[str, float]` field per CC3 rules. Audit gate: docstring update.
- `EC` snapshot `_format_version` bump for `_binding_edges` round-trip.
- No new frozen dataclasses introduced.

## Definition of done

- Roy-4 instrumentation experiment shipped to main; pass criteria met (priming-cluster ↔ test-cluster pairs would have been linked).
- All Stage 2-6 implementation behind pre-merge two-lens reviews.
- Roy-5 produces non-zero `sense_food_source` calls in arm A's test phase — the Roy-2c-specific behavioral expression the cluster wire was designed to enable.
- Roy-5 ablation with `MAXIM_EC_BINDING_RATE=0.0` reproduces Roy-2c's identical-distribution result, confirming the binding mechanism (not some other code change) is the load-bearing variable.
- Cross-session persistence: `_binding_edges` round-trips through dump/load; binding strengths decay correctly post-reload.
- `grounded_language_acquisition.md` Phase 1's symbol-binding-registry implementation can query `EC._binding_edges` rather than building duplicate co-occurrence machinery. This plan does not *gate* Phase 1 (Phase 1 retains a fallback co-occurrence learner if this plan slips), but coordinating the two during 1.1 dev reduces duplicate work.
- Wire-A from 0.9.1 still active. Bound-edge consumption is the substrate-primary substrate-level fix; Wire-A is the llm-primary prompt-level fix. They coexist; neither supersedes the other.

## What this plan does NOT do

- **No millisecond-scale binding.** Maxim is tick-discrete; cross-modal binding is same-tick or adjacent-tick co-activation, not gamma-cycle phase-locking.
- **No phase-locking via SCN oscillator.** SCN provides circadian phase context for credit assignment (B2 shipped). Whether SCN should *also* mediate cross-modal binding (e.g., phase-aligned co-activation binds stronger than phase-misaligned) is an open question for 1.2+; out of 1.1 scope.
- **No replacement for `LinguisticEncoder`.** Encoder alignment is the underlying problem; binding routes around it. A 1.2+ research direction could replace `LinguisticEncoder` with an aligned multimodal encoder, but that's out of scope here.
- **No automatic learning rate adaptation.** `binding_rate` is a static hyperparameter tuned per-deployment. Roy iterations can sweep it; adaptive learning rate is post-1.1.
- **No multi-agent binding sharing.** Each agent's `_binding_edges` is per-agent. Hivemind shareability (post-1.0 grounded_language_acquisition Phase 2+) could share binding-edge snapshots across agents but that's a separate plan.

## Risk register

| Risk | Mitigation |
|---|---|
| Binding rule over-fires (Pavlov-on-steroids) | Roy-4 validates the binding criterion BEFORE Stage 2 ships. Decay + salience-weighting + min-co-firing-count threshold are three independent over-binding guards. |
| Binding rule under-fires (associations never form) | Same Roy-4 validation. Env-var tunables let Roy iterations sweep without code edits. |
| `_binding_edges` grows unboundedly | `prune_threshold` deletes weak edges. Pre-merge review must verify edge count plateaus over a long-horizon Roy run. |
| Binding works in Roy but not under noisier real-world inputs | Documented as out-of-scope for 1.1; Hivemind track + Minecraft benchmark will surface this empirically. |
| Roy-4 fails (pairs don't co-fire even at instrumentation level) | Cancel Stage 2. The deeper fix is replacing LinguisticEncoder with an aligned multimodal encoder — a 1.2+ research direction. Roy-4 is the cheap gate that prevents this misallocation. |

## 1.0 / 0.9.1 implications

- 0.9.1 Stage 2 (Wire-A) ships as planned — it is the interim that gives us substrate-attributable behavior immediately while the binding mechanism (this plan) builds up associations through experience.
- Roy-4 lands during 0.9.1's development cycle as the experimental prereq for 1.1.
- 1.0 release plan ([v1_refinement.md](v1_refinement.md)) is unaffected. This plan is strictly 1.1+ work.
- `grounded_language_acquisition.md` Phase 1 is now concretely-scoped via this plan, no longer an open-ended "Hebbian binding across modalities" placeholder.

## References

- [docs/experiments/20_roy_2c.md](../experiments/20_roy_2c.md) — H1 confirmation; the empirical floor that motivates this plan.
- [docs/experiments/19_roy_2pc.md](../experiments/19_roy_2pc.md) — Roy-2pc's positive-control negative result; structural-vs-behavioral gap pre-disambiguation.
- [release_0_9_1.md](release_0_9_1.md) — Wire-A interim that ships before this plan.
- [grounded_language_acquisition.md](grounded_language_acquisition.md) — Phase 1 implementation depends on this plan.
- [CLAUDE.md "SCN temporal coupling for eligibility traces"](../../CLAUDE.md) — the first SCN-substrate PoC documenting `_temporal_anchors`; this plan extends the same intuition from credit-assignment to edge-formation.
