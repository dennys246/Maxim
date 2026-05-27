# Behavioral Graduation Candidates — Path to 1.0

**Status:** Active. Pairs alongside benchmarking as one of two 1.0 gates.
**Created:** 2026-05-27.
**Sibling doc:** [behavioral_convergence_practice.md](behavioral_convergence_practice.md) — "does the agent get better" (system-level, ongoing). This doc — "do specific bio-mechanisms carry the load they claim" (mechanism-level, experiment-driven).
**Triggered by:** CLAUDE.md tagging audit (PR #272, 2026-05-26) revealed 67 `[engineering]` vs 4 `[behavioral]` — a 5.6% behavioral ratio. Honest read: many bio-mechanisms in CLAUDE.md claim behavioral weight but haven't yet earned a cited experiment. The Roy harness was the prerequisite for earning them; it is just now mature enough to start producing graduations at scale.

## Why this is a 1.0 gate

1.0 has **two orthogonal gates**, not one:

- **Benchmarking** — agent performance on tasks (does it solve the thing).
- **Behavioral graduation** — mechanism validation (does the bio-pipeline carry the load it claims).

Both gate 1.0. They are siblings, not nested. A passing benchmark with unbacked bio-claims still leaves the bio-inspired-LLM-harness framing exposed. A failing benchmark with strong mechanism validation is not 1.0-shippable either.

This doc tracks the second axis. Each entry below is an engineering invariant (or family of invariants) from CLAUDE.md whose bio-flavored framing makes an implicit behavioral claim. By 1.0 each entry resolves to one of: (a) **Earned** by a cited Roy/equivalent experiment, (b) **Downgraded** to scaffolding-only with no apology, or (c) **Dormant** per [Principle 2](../../CLAUDE.md#working-principles-for-new-mechanisms).

## Three-tier scope

Not all 67 `[engineering]` invariants from CLAUDE.md need to graduate. Don't burn experiments on engineering scaffolding.

### Tier 1 — Thesis-load-bearing (MUST graduate, or 1.0 is dishonest)

These are the bio-claims of the product itself. Shipping 1.0 without cited experiments for these undermines the [framing strategy](../../CLAUDE.md) (bio-inspired LLM harness, not academic cog-sci).

| Claim | Bio-mechanism | Status |
|---|---|---|
| Cross-session learning without fine-tuning | Hippocampus persistence + recall + consolidation | **PARTIAL** — 3 memories/turn on resume per [Exp 10](../experiments/10_cross_session_enrichment.md); predictions + concepts pending |
| EC pattern completion / separation | EC centroid + threshold tracking NAc fallback | **EARNED** — Roy-2c Phase 4 + Roy-5 H1C (CLAUDE.md L35) |
| SEM pain → NAc cascade | ToolPainBridge direct-attribution + record_outcome | **EARNED (de facto)** — Substrate P2 cascade test end-to-end on rusty_sword (CLAUDE.md L75 + L83 borderline-tagged behavioral) |
| Affordance concept transfer | Affordance LinguisticEncoder + EC pattern completion across entities | **PARTIAL** — cross-entity transfer measured at 0.785 cosine via affordance encoder PoC; broader Roy-5+ work continuing |
| Substrate-primary action selection | `NAc.propose_via_substrate` with confidence gate | **PARTIAL** — Phase 0 harness shipped per B5 (PR #228, 2026-05-09); Phase 0 validation pending |

**1.0 commitment:** all five reach `EARNED` status before 1.0 ships. If a claim can't be earned, the bio-framing for that mechanism gets pulled in 1.0 release notes — explicit retraction, not silent omission.

### Tier 2 — Scaffolding (stays `[engineering]` permanently, no apology)

Engineering walls around the bio-substrate. Code-correctness contracts, builder invariants, persistence formats, type freezes. **Do not burn experiments earning these.** They will never graduate, and that's fine.

Representative examples (not exhaustive — see CLAUDE.md "Architectural invariants" section for the full list):

- All canonical-builder rules: `build_executor`, `build_pain_bus`, `build_default_network`, `build_reaction_bus`, `build_memory_hub`, `build_bio_stack`
- `_format_version` contract, `atomic_write_json` persistence, frozen-dataclass freeze (CC3)
- `_MaximPeerBackend` one-HTTP-call + `for_url` instance-key + `health_check` rules
- Memory tier progression direction (FORMING → SHORT_TERM → LONG_TERM)
- `mesh.yml` parser frozen + declarative-vs-mutable split
- `Tool.cancel()` / `Tool.input_schema` API freezes (CC9, CC11)
- `PerceptSource` / `ActionSink` protocol minimality (CC8)
- HTTP errors typed + httpx stream context lifetime
- Role detection ordering in `cli.py::main`
- HTTP call sites must use `maxim/utils/http.py`

**Why these don't graduate:** the failure mode is "code breaks loudly without it." There is no behavioral measurement that would meaningfully validate `atomic_write_json` — its job is to not corrupt files on power loss, which is a correctness property, not a bio-claim. Same for the rest. Surfacing this as a deliberate "no graduation needed" set is itself useful — it bounds the graduation work to a tractable scope.

### Tier 3 — Middle subset (graduate-or-downgrade per checkpoint)

These are the mechanisms where the bio-flavored framing in CLAUDE.md (or in the shipping commit messages / experiments / plans) makes an implicit behavioral claim that isn't yet cited. Each gets a graduation predicate or an explicit "can't predicate yet — claim fuzzy" flag.

**Status conventions:**

- **Pending — predicate written.** Setup + metric + checkpoint named below. Ready to run when fixture lands.
- **Pending — needs fixture design.** Experimental shape is clear but fixture / metric threshold need design work.
- **Pending — can't predicate yet.** Multi-confound or "what counts as the dependent variable is partially the experiment." These are the most honest finding — flagged so they're visible, not hidden.
- **Earned — `<date>`.** Cited experiment ships; CLAUDE.md tag flipped (or scheduled).
- **Dropped.** Couldn't earn it by checkpoint. Either bio-framing pulled (stays `[engineering]` no apology) or marked Dormant per [Principle 2](../../CLAUDE.md#working-principles-for-new-mechanisms).
- **Borderline scaffolding.** On reflection this probably belongs in Tier 2; surface for decision rather than predicate.

#### Initial seed table (v0.1 — expand iteratively as candidates surface)

| # | CLAUDE.md ref | Mechanism | Bio-claim | Graduation predicate | Status |
|---|---|---|---|---|---|
| 1 | L162 (affordance encoder split) | LinguisticEncoder + AffordanceDecompositionStrategy | Affordance names decompose to substrate concepts that transfer across entities via shared EC nodes | **Setup:** cross-entity transfer fixture (e.g., "flame" trained on torch, evaluated on dragon affordances). **Metric:** NAc `reward_bias` propagation to held-out entity ≥ cosine threshold across N=10 entity pairs. **Checkpoint:** 1.0 | Pending — predicate written |
| 2 | Wire-A (across CLAUDE.md + env table) | Cluster reward bias annotation with split tau | Substrate-voice multi-turn annotation modulates action selection at horizons > single-turn reward bias | **Setup:** Roy-3a-retry with engineered priming fixture; arms = annotation-on vs annotation-off with split tau (300 cluster vs 50 reward). **Metric:** post-priming action distribution shift sustained across N≥5 turns. **Checkpoint:** Tier 2 tau-split Phase 1 + Roy-3a-retry → 0.9.2 or 1.0 | Pending — predicate written; depends on [cluster_reward_bias_decay_tau_split.md](cluster_reward_bias_decay_tau_split.md) Phase 1 |
| 3 | Wire 1 (CLAUDE.md L37) | Variance annotation on tool descriptions | Risk-sensitive action selection — agent avoids high-variance affordances when low-variance alternatives exist | **Setup:** Roy-3 ablation arm with two affordances at same mean reward, different variance. **Metric:** preference shift toward low-variance affordance ≥ Δ across N attempts. **Checkpoint:** Roy-3 ablation → 0.9.2 or 1.0 | Pending — predicate written |
| 4 | L161 (SCN oscillator) | OscillatorNetwork anticipatory pre-activation | Diurnal event-type patterns pre-activate relevant percepts/affordances before predicted events | **Setup:** repeated-pattern fixture (e.g., periodic thermal-discomfort event). **Metric:** NAc eligibility-trace pre-activation at predicted event time ≥ baseline by ≥ 0.2x weight. **Checkpoint:** 1.0 if cradle exercises oscillator; 1.1 otherwise | Pending — needs fixture design |
| 5 | L160 (SCN temporal coupling for traces) | `NAc._temporal_anchors` phase-similarity fallback at 0.3x weight | When fast-decay eligibility traces expire, temporal-phase similarity restores credit to nodes activated in same phase | **Setup:** affordance concept transfer fixture with delayed reward beyond fast-decay window. **Metric:** temporal-coupling-on arm credits same-phase nodes at ≥ X cosine vs temporal-coupling-off arm. **Checkpoint:** 1.0 | Pending — needs fixture design |
| 6 | L163 (drive protocol) | HomeostaticDriveSpec / EntropicDriveSpec → behavior | Drive deviation beyond `comfort_band` shifts action selection toward corrective affordances | **Setup:** Cradle scenario with thermal / hunger perturbation; measure action selection on corrective vs distractor affordances. **Metric:** corrective preference rate > random + significance threshold when drive deviation > `comfort_band`. **Checkpoint:** 1.0 — cradle B4 is the natural test bed | Pending — predicate written; covered by [v1_refinement.md](v1_refinement.md) cradle work |
| 7 | L156-157 (valence on edges) | `spreading_activation(propagate_valence=True)` + `apply_hebbian_on_close` + `salience_spike_rule` | Negative-valence-tagged paths attenuate spreading activation; positive amplify | **Setup:** synthetic episode graph with valence-tagged edges; measure recall preference distribution. **Metric:** recall probability for negative-valence path < random baseline; positive > baseline. **Checkpoint:** 1.0 | Pending — needs fixture design |
| 8 | L159 (NAc per-tick decay wired) | `decay_eligibility` + `decay_reward_biases` per agent_loop tick | Decay enables credit to track recency; without decay, traces stay forever and disrupt new learning | **Setup:** two arms — decay-on vs decay-disabled — running affordance learning across N tasks. **Metric:** decay-on arm shows faster convergence on later tasks (less interference from earlier reward biases). **Checkpoint:** 1.0 | Pending — needs fixture design |
| 9 | Reflex system (per `project_percept_reflex_system_shipped` memory) | Innate body reflexes (e.g., infant thermal contact) | Reflexes fire below deliberation; shape learned avoidance over repeated exposure | **Setup:** already validated per [Experiment 09](../experiments/09_percept_reflex.md) — infant thermal contact reflex. **Metric:** reflex fires at contact + learned-avoidance trajectory measured. | **EARNED — Experiment 09** (CLAUDE.md citation update pending) |
| 10 | L164 (entity acquisition) | `entity_acquired` / `entity_released` contact sensation | Acquired entities contribute sensors to body damage model while equipped; behavioral effect of contact persists | **Setup:** cradle / sim scenario with damaging vs benign acquired entity. **Metric:** damage propagation through acquired entity sensors validated end-to-end; behavioral preference shift over N pickups. **Checkpoint:** 1.0 | Pending — predicate written; partial coverage from component damage work |
| 11 | B3 Acting Coach (per `project_07_r0_b3_shipped` memory) | Acting Coach prompt with bio-system modulation (NAc caution, pain anticipation, cerebellum predictions) | Substrate-informed meta-prompt steers affordance exploration toward salient/cautious actions per NAc state | **Setup:** generative sim with Acting Coach on vs off; measure exploration breadth + caution markers. **Metric:** caution rate scales with NAc reward bias magnitude in on-arm. **Checkpoint:** 1.0 | Pending — **can't predicate cleanly yet** (multi-confound: LLM stylistic variation + Acting Coach effect entangled; needs ablation design that isolates substrate-modulation from prompt-tone changes) |
| 12 | Pre-deliberation (per `project_pre_deliberation_shipped`) | ThoughtGate + BioEnrichment Layer 1 pre-LLM | Bio-enrichment of pre-deliberation thought stream improves downstream action selection | **Setup:** comparative arms — pre-deliberation on vs off — on a benchmark task. **Metric:** task success rate or action coherence delta. **Checkpoint:** 1.0 | Pending — **can't predicate cleanly yet** (multi-confound; entangles with prompt construction in non-trivial ways) |
| 13 | Working memory (per `project_working_memory_plan`) | `WorkingMemorySet` Exec-owned active reference | Active-reference items in WMS bias recall + action selection toward relevant context | **Setup:** distractor-task fixture; WMS-on vs WMS-bypassed arm. **Metric:** recall latency / accuracy delta on WMS-held items. **Checkpoint:** 1.0 if predicate sharpens; 1.1 if not | Pending — **can't predicate cleanly yet** (what counts as "relevant" is partially the experiment) |
| 14 | Imagination (per `project_i1_i2_imagination_shipped`) | `ImaginationTrigger` w/ DN arousal gate + ComponentIndex lookup + energy budget | Novel entity mentions trigger LLM-driven entity design only when DN arousal + energy budget permit | **Setup:** sim with novel-entity-rich percepts; measure design call rate vs arousal floor. **Metric:** design rate scales with arousal above floor; below floor → zero rate. **Checkpoint:** 1.0 | Pending — predicate written |
| 15 | L166 (three interaction levels) | Observe / Touch / Pick up sensation layering | All three converge on sensor change → `evaluate_failures` → PainBus → NAc; each layer adds distinct behavioral signal | **Setup:** cradle / sim exercising all three interaction types on damaging entity; measure NAc learning rate per layer. **Metric:** NAc learning observed in all three branches; layering doesn't collapse to single-layer behavior. **Checkpoint:** 1.0 | Pending — predicate written; partial coverage from component damage + entity acquisition work |
| 16 | L152 (`Hippocampus.recall` touch + RECALL reconsolidation) | `memory.touch` + WorkingMemorySet RECALL entries on recall | Recall strengthens accessed memories; reconsolidation pull-into-active improves subsequent recall | **Setup:** repeated recall fixture across sessions; measure access strength growth on touched items. **Metric:** touched-item recall probability > untouched-item probability after N repetitions. **Checkpoint:** 1.0 — partial coverage via V1 cross-session work | **PARTIAL** — V1 cross-session shows 3 memories/turn on resume (per Exp 10), but the isolated reconsolidation effect is not yet ablated against a no-touch baseline |
| 17 | ComponentIndex (per `project_e25_component_index_shipped`) | Two-layer semantic discovery (alias hash O(1) + embedding cosine) | Semantic discovery surfaces relevant components faster than naive scan; alias hits dominate cosine on common queries | **Setup:** discovery benchmark with N entity queries; measure alias-hit rate, cosine-fallback rate, latency. **Metric:** alias hit > 50% on synonym-rich corpus; total discovery latency < baseline scan. **Checkpoint:** could ship before 1.0 | **Borderline scaffolding** — performance property of a discovery mechanism, not a bio-claim. Surface for decision: probably belongs in Tier 2 (engineering, no graduation needed). |
| 18 | L116 (Hippocampus + NAc + ATL separate stores) | Three separate `EpisodicMemory` instances | Coexistence (not merger) is bio-correct AND behaviorally distinct from a merged store | **Setup:** comparative — merged-EpisodicMemory arm vs separate-stores arm. **Metric:** separate-stores arm shows behavior X absent from merged arm. **Checkpoint:** N/A | Pending — **can't predicate cleanly** (architectural / structural; the alternative requires substantial rebuild and isolating "what behavior differs" is the experiment). Strong candidate to drop to Tier 2 if no predicate emerges by 0.9.x checkpoint. |
| 19 | L157 (NAc `_reward_bias` clamping) | `_reward_bias` clamps to `[0, max_reward_bias]`; negative rewards handled via valence-on-edges | Bias only widens EC recognition, never narrows; pain avoidance routed via valence path not bias narrowing | **Setup:** compare arms — negative-clamped-to-zero vs negative-narrows-recognition. **Metric:** clamping arm shows pain-avoidance via valence; non-clamping arm shows narrowing artifacts. **Checkpoint:** N/A | Pending — **architectural choice; predicate fuzzy.** The alternative would require reimplementing the bias formula. Strong candidate to drop to Tier 2 if no predicate emerges. |
| 20 | L85 (PainBus refractory gate) | `(entity, failure_mode)` refractory dedup | Fine-grained refractory prevents within-tick duplicate dispatch while preserving distinct entities | **Setup:** multi-entity pain-burst fixture (two distinct entities fire same-tick). **Metric:** 2 distinct dispatches preserved; same-entity-same-failure collapsed to 1. **Checkpoint:** N/A | **Borderline scaffolding** — correctness property already pinned by regression test, not a behavioral claim. Surface for decision: belongs in Tier 2. |

#### Honest accounting of the v0.1 seed

20 candidates above resolve as:

- **5 with predicates I'd ship today** (#1, #2, #3, #6, #10, #14, #15): clear setup + metric + checkpoint.
- **4 needing fixture design** (#4, #5, #7, #8): experimental shape is clear but fixture / metric threshold need design work before running.
- **1 EARNED** (#9 — reflexes via Exp 09): documentation update pending.
- **1 PARTIAL** (#16 — reconsolidation): V1 broadly covers it, isolated effect not yet ablated.
- **3 borderline scaffolding** (#17, #19, #20): probably belong in Tier 2; surface for decision rather than predicate.
- **3 can't predicate yet — claim fuzzy** (#11, #12, #13 — Acting Coach, pre-deliberation, working memory): multi-confound or "what counts as relevant is partially the experiment."
- **1 architectural / can't predicate cleanly** (#18 — separate EpisodicMemory stores): the alternative requires substantial rebuild.

**Most useful single output of this exercise:** the four "can't predicate yet" entries (#11, #12, #13, #18). These mark places where the codebase's bio-framing outruns the experimental backing. Either the claim sharpens (predicate emerges at a later checkpoint) or the bio-framing gets pulled in 1.0 release notes (claim retracted, mechanism stays as engineering scaffolding with no apology).

## Status check cadence

- **0.9.x checkpoint** (each minor release): walk the table, flip `Earned` tags, drop entries that didn't earn predicates, fold borderline-scaffolding entries down to Tier 2.
- **1.0 readiness review:** Tier 1 must be 100% `Earned`. Tier 3 entries that can't be earned have explicit dispositions: scaffolding-downgrade, dormant per Principle 2, or 1.1+ deferral.
- **Post-1.0:** this becomes a living doc like [behavioral_convergence_practice.md](behavioral_convergence_practice.md) — new bio-mechanisms entering 1.x+ track here from inception, not retroactively.

## Cross-references

- [CLAUDE.md "Working principles for new mechanisms"](../../CLAUDE.md#working-principles-for-new-mechanisms) — Principle 1 (two-tier invariant tracking) is the gate that triggered this doc.
- [behavioral_convergence_practice.md](behavioral_convergence_practice.md) — "does the agent get better" (system-level, ongoing). Sibling doc, different question.
- [v1_refinement.md](v1_refinement.md) — 1.0 unified plan. Tier 1 entries here line up with V1 validation work.
- [release_0_9_1.md](release_0_9_1.md) — current active release. Tier 3 #2 + #3 (Wire-A + Wire 1) hinge on Roy-3 ablation arms shipping there.
- [cluster_reward_bias_decay_tau_split.md](cluster_reward_bias_decay_tau_split.md) — Tier 3 #2 depends on Phase 1 implementation.
- [grounded_language_acquisition.md](grounded_language_acquisition.md) — Tier 1 substrate-primary entry lines up with B5 work.
- [roy/](roy/) — Roy harness directory. Experimental setups for Tier 3 predicates land here.

## What this doc is NOT

- **Not a substitute for individual experiment plans.** Predicates here are seeds; full experimental designs live in `docs/experiments/` once they're being run.
- **Not a wish list.** Entries without predicates after the 0.9.x checkpoint cycle drop or downgrade — they don't sit indefinitely as aspirations.
- **Not exhaustive.** v0.1 seeds 20 candidates; expand iteratively as new bio-mechanisms ship or audits surface more.
- **Not the 1.0 release gate by itself.** Benchmarking is the sibling gate; both must pass.
