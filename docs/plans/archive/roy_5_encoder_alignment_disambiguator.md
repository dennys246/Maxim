# Roy-5 encoder alignment disambiguator (1.1+ research direction)

> **ARCHIVED (2026-07-15 plans audit):** ✅ STAGES 1–4 ALL RESOLVED. Stage 1 analyzer (PRs #249/#251, H1a confirmed — 384-dim vs 768-dim spaces); Stage 3 drive→linguistic co-firing scaffold shipped (PR #295) and adjudicated by Exp 35 + Exp 36 (gap closure attributed to EC drift fix #264); Stage 4a executed (cross-modal binding archived), Stage 4b left unpromoted (JEPA stays a candidate). The open "Stage 5" mechanistic question is new-plan-shaped work, not in-flight.


**Target version:** 1.1+ research direction (the 1.0 plan is unaffected; 0.9.1 ships Wire-A independently).
**Status:** **Stage 3 COMPLETE — Branch A decisive (threshold-driven) via Roy-5b-confound-isolation (2026-05-29). Stage 4a rationale collapsed; `cross_modal_substrate_binding.md` ARCHIVED. Stage 4b stays parked at "candidate" with weakened motivation. Disambiguator plan moves to a new next-question.** Roy-5b ([docs/experiments/35_roy_5b.md](../../experiments/35_roy_5b.md)) produced a Conditional PASS / Ambiguous verdict (1 matched (drive, drive) intra-modal edge at default rule); Roy-5b-confound-isolation ([docs/experiments/36_roy_5b_confound_isolation.md](../../experiments/36_roy_5b_confound_isolation.md)) ran the Roy-4 spec at HEAD without the naming-event scaffold and produced **10/10 arm-A overlap with priming** — identical to Roy-5b's. The recognition gap closure attributed to the scaffold in Roy-5b is entirely explained by the EC drift fix (PR #264, `pattern_complete_threshold` 0.40 → 0.44, 2026-05-24). The scaffold IS doing real work at the binding-rule layer (it produces 1 extra (drive, drive) edge at default) but **not at the load-bearing gap-closure metric**. The cross-modal binding mechanism this plan was designed to evaluate is empirically dead as a Stage 4a resurrection target. **New Stage 5 question:** what explains the threshold-driven recognition gap closure mechanistically? Does the closure produce downstream behavioral convergence (Roy-2c-style persona-inertness was the original problem), or is it visible at the EC layer but not the cluster_reward_bias / behavioral layer? Does the small `Δ=0.04` threshold tuning that produced this closure motivate adaptive thresholds, modality-specific frozen-prototype semantics, or a different 1.1+ research direction entirely? Stage 1 (Roy-5a cosine localization, 2026-05-13) confirmed H1a; the exp 30→32→33→34 sequence proved the substrate→action thesis is bottlenecked at the encoder layer; this plan's Stage 4 disposition is now resolved (4a: dead, 4b: parked, new direction TBD).
**Owns:** `scripts/analyze_roy_5_cosine_localization.py` (new), `scenarios/roy/roy_5a_*.yaml` (post-hoc analysis spec), `scenarios/roy/roy_5b_*.yaml` (conditional cradle-arc redesign + Hebbian retest), `docs/experiments/22_roy_5*.md`, `_data/components/bodies/infant_humanoid_*.yaml` (conditional cradle-arc edits), `embodiment/naming_events.py` (conditional naming-event scaffolding).
**Companion plans:** [persona_convergence_crucible.md](../deferred/persona_convergence_crucible.md) (Roy iteration log) · [release_0_9_1.md](release_0_9_1.md) (Wire-A interim is unchanged by this plan) · [grounded_language_acquisition.md](../grounded_language_acquisition.md) (Phase 1's `token_id → ec_node_id` registry consumes whichever populator this plan validates) · [cross_modal_substrate_binding.md](cross_modal_substrate_binding.md) (cancelled by Roy-4; may resurrect with corrected scaffold per Stage 3)

## Front-gate scope pressure (retroactive)

Added 2026-05-27 per CLAUDE.md Principle 3.

**Question:** does this diagnostic + branched-implementation plan need to be its own mechanism, or can it ride on existing infrastructure?

**Note:** Stage 1 is **diagnostic only** (analyzer script consuming existing logs); no front-gate question at all — pure measurement. The front-gate question bites only for the branched implementation stages (2a/2b/2c → 3 → 4a/4b).

**Existing infrastructure surveyed (Stage 2a — H1c threshold/centroid sweep):**

| Candidate | Why insufficient (or sufficient) |
|---|---|
| `ECConfig.pattern_complete_threshold` + `frozen_centroid_modalities` | **Already the right knobs** — H1c branch is pure parameter tuning, no new code |
| Existing `scripts/diagnose_*` matrix sweep pattern ([scripts/diagnose_roy_paraphrase_collapse.py](../../scripts/diagnose_roy_paraphrase_collapse.py)) | **Already the right sweep harness** — Stage 2a reuses |

**Verdict (Stage 2a):** could-ride-on-existing entirely. No new mechanism.

**Existing infrastructure surveyed (Stage 2b — H1b encoder A/B):**

| Candidate | Why insufficient (or sufficient) |
|---|---|
| `LinguisticEncoder._get_encoder` singleton | **The bug** — process-wide singleton blocks A/B. Stage 2b kills the singleton, adds per-call-site factory accepting explicit model name. Additive — existing zero-arg form still works |
| Existing `LinguisticEncoder` infrastructure | Otherwise unchanged — Stage 2b is one signature change |

**Verdict (Stage 2b):** could-ride-on-existing with one additive API shape change (the existing singleton becomes a default for the new factory).

**Existing infrastructure surveyed (Stages 3-4 — H1a + cradle redesign + binding/encoder branches):**

| Candidate | Why insufficient (or sufficient) |
|---|---|
| Cradle arc YAML ([_data/components/bodies/infant_humanoid.yaml](../../src/maxim/_data/components/bodies/infant_humanoid.yaml)) + narrator scaffold | **Already the right modification targets** — Stage 3 adds deliberate naming events to existing arc structure |
| `cross_modal_substrate_binding.md` Hebbian binding | Resurrected on Stage 4a PASS — the new mechanism is *already-designed elsewhere* |
| `jepa_cross_modal_alignment.md` projection layer | Promoted on Stage 4a-rescue-fails — the new mechanism is *designed elsewhere* |

**Verdict (Stages 3-4):** the implementation branches both **point at separate plans** for their new-mechanism scope. This plan's diagnostic-first reframe is the front-gate discipline itself — defer the new-mechanism decision until measurement names which branch is needed.

**Verdict aggregate:** this plan is **largely diagnostic-only with branched parameter tuning**. The expensive new mechanisms (binding edges, JEPA projection) are deferred to dependent plans. Specific reason: per the two-lens review convergence cited below, "audit before building + diagnose before implementing" prevents committing to a new mechanism before data names which one is required.

## Why this plan exists

Roy-4 (PR #246, [docs/experiments/21_roy_4.md](../../experiments/21_roy_4.md)) cancelled `cross_modal_substrate_binding.md` Stages 2-6 because the proposed Hebbian binding rule found zero priming↔test bound edges across the full reasonable parameter sweep. The empirical finding was specific:

- Zero EC node-ID overlap between priming (37 unique nodes) and any test arm (10/13/9 unique nodes).
- Priming food clusters fire in near-isolation: 61 ticks where any of the 6 food clusters fires, only 1 with a non-food co-firing partner, zero overlap between those 7 non-food partners and arm A's test-phase nodes.

The user's follow-up proposal — a "linguistic funneling + lexicon" system with a factory + per-engine encoders + central dictionary — went through two-lens pre-merge review (architecture + bio-fidelity). **Both reviewers independently rejected the proposal as scoped**, citing distinct mechanisms:

- **Architecture lens** ([review transcript, this PR]): Options A (pre-embedding normalizer) and B (post-embedding canonical-node routing) do not solve Roy-2c — A has no surface tokens to normalize between cradle sensor/drive snapshots and CLI fixture text; B makes the substrate's emergent-clustering thesis unfalsifiable. The reviewer specifically recommended: *"do not write the lexicon plan. Write `roy_5_encoder_alignment_disambiguator.md` (~50 LOC scope, days) and let its outcome name the 1.1 plan."*
- **Bio-fidelity lens** ([review transcript, this PR]): Option A is "engineered patch dressed as bio-inspiration"; Option B converts EC from "bio-inspired pattern-completion network" to "hash table with vector layer for show"; both contaminate the 1.0 thesis. Existing infrastructure (`ComponentIndex`'s alias+embedding pattern, `ATL.find_or_create`, `AffordanceDecompositionStrategy`) already implements per-domain lookup more bio-faithfully than a central module would.

Both reviews converged independently on **"audit before building" + "diagnose before implementing"** — the same framing rule that saved months of misallocated work on the binding plan via Roy-4.

This plan ships that diagnostic, in two sequenced stages, and pre-registers the implementation that gets scoped from each outcome. It does **not** pre-commit to lexicon, encoder replacement, threshold tuning, or scaffold redesign — it commits to running the cheapest experiment that disambiguates which fix the data actually demands, then building only that fix.

## Framing rule

**The diagnostic ships before any 1.1+ implementation plan is written.** If Roy-5a (Stage 1) localizes the gap to threshold/centroid mechanics (H1c), the implementation is a ~50 LOC tuning sweep and no further plan is needed. If Roy-5a localizes it to encoder model fit (H1b), the implementation is an encoder A/B sweep, ~150 LOC. If Roy-5a localizes it to encoder subspace incompatibility (H1a), the diagnostic continues into Stage 3 (Option D Roy-5b — cradle-arc redesign + Hebbian retest) before committing to encoder replacement.

This is the same shape as Roy-4 → cancel binding plan: cheap experimental gate → implementation scoped by outcome → no premature scope-commit.

## The three sub-hypotheses behind Roy-2c

Roy-2c proved priming-cluster ↔ test-cluster pairs are structurally disjoint. It did NOT disambiguate WHY. Three live sub-hypotheses exist; each prescribes a different fix:

| ID | Hypothesis | Fix shape | Cost |
|---|---|---|---|
| **H1c** | Text *is* close in embedding space but EC's `pattern_complete_threshold=0.40` + frozen-prototype + running-mean centroid drift make pattern completion miss | Threshold/centroid sweep | ~50 LOC, days |
| **H1b** | Both *are* text but cradle narrator output ("the infant is in a room with a fire pit nearby") and CLI test fixture ("you sense food nearby") land far in paraphrase-mpnet's embedding space specifically — a different encoder model would close the gap | Encoder model A/B at the `LinguisticEncoder._get_encoder` singleton | ~150 LOC, weeks |
| **H1a** | Sensor-drive embeddings (`SensorEncoder` SHA basis, modality="interoception") and text embeddings (paraphrase-mpnet, modality="text") embed into structurally incomparable subspaces of EC even within the same modality channel | Cross-modal alignment (encoder replacement, Option C in conversation framing) — promoted to 1.2+ if Stage 3 doesn't rescue via cradle-arc redesign | Months |

Roy-2c's "tool keys survive, cluster keys don't" pattern is consistent with all three. Roy-5a measures which actually holds.

## Sizing

| Stage | Item | LOC | Persistence | Frozen impact |
|---|---|---|---|---|
| 1 | Roy-5a — cosine-localization analysis script + outcome doc | ~120 | none | none |
| 2a | H1c branch: threshold/centroid sweep + autouse env scrub | ~80 | none | none |
| 2b | H1b branch: encoder A/B + per-call-site `LinguisticEncoder` factory (kill the process-wide singleton) | ~200 | none | `_get_encoder` API shape (additive — accept explicit model name; existing zero-arg form still works) |
| 2c | H1a branch: redirect to Stage 3 (no implementation yet) | 0 | none | none |
| 3 | Roy-5b — redesigned cradle priming arc (deliberate naming events) + Hebbian retest | ~250 | none | new `bodies/infant_humanoid_naming_v1.yaml` + `embodiment/naming_events.py` co-firing scaffold |
| 4a | H1a + Stage 3 PASS: resurrect `cross_modal_substrate_binding.md` Stages 2-6 with corrected scaffold; lift the cancel | ~780 (from cancelled plan) | EC `_format_version` bump + `PatternCompletionResult.bound_neighbors` field | (see cancelled plan) |
| 4b | H1a + Stage 3 FAIL: promote encoder replacement to 1.2+ research direction; archive this plan + `cross_modal_substrate_binding.md` as definitively superseded | ~30 (note only) | none | none |
| **Total 1.1+ if H1c wins** | | **~200** | none | none |
| **Total 1.1+ if H1b wins** | | **~320** | none | additive |
| **Total 1.1+ if H1a + scaffold rescues** | | **~1150** | (see cancelled plan) | (see cancelled plan) |
| **Total 1.1+ if H1a + scaffold fails** | | **~400** | none | none |

Note: the maximum scope (~1150 LOC) is only reached on the "encoder subspaces are incompatible AND the cradle scaffold can rescue Hebbian binding" branch. Most branches keep total scope well below the cancelled binding plan's ~780 LOC.

## Stage 1 — Roy-5a: cosine-localization analysis on existing Roy-4 data

**Goal:** localize Roy-2c's encoder-alignment gap to one of three sub-hypotheses using **only** the Roy-4 EC traces and priming/test session text logs that already exist. Zero new sim runs. Hours of work.

### Implementation

- New script `scripts/analyze_roy_5_cosine_localization.py`. Mirrors the structure of `scripts/analyze_roy_4_coactivation.py` (same JSONL ingestion, same dual-format detection for sim_log vs MAXIM_LOG_FILE-bridge records).
- Read priming session's persisted EC state — `EC._substrate_nodes` is keyed by `node_id → (centroid_embedding, modality)`. Available from the priming session's persisted EC dump (`~/.maxim/sim_reports/{priming_sid}/aut_*` — verify path during implementation).
- Read each arm's persisted EC state from the same locations.
- For arm A, compute three matrices:
  - **M_tt**: pairwise cosine between every priming `text`-modality centroid and every arm-A `text`-modality centroid.
  - **M_dt**: priming `interoception` centroids vs arm-A `text` centroids (cross-modality).
  - **M_dd**: priming `interoception` centroids vs arm-A `interoception` centroids.
- For each food-bearing priming centroid (the 6 priming `sense_food_source` cluster IDs from `aut_nac.json`'s `cluster_reward_bias` compound keys, identified the same way the Roy-4 analyzer does), report max cosine across the arm-A side per modality pair.
- Emit summary table + JSON bundle. Pre-registered decoding:

| `max(M_tt)` for food-bearing priming centroids | Diagnosis | Fix branch |
|---|---|---|
| ≥ 0.40 (current EC threshold) | **H1c wins** | Stage 2a — threshold/centroid sweep |
| 0.20 – 0.40 | **H1b wins** | Stage 2b — encoder A/B |
| < 0.20 | **H1a wins** | Stage 2c → Stage 3 |

The 0.40 cutoff is the current `ECConfig.pattern_complete_threshold` (P1-tuned); the 0.20 lower band is a conservative gap below which "the encoder doesn't even see these as related" is the most parsimonious explanation.

### Test surface

- Unit: parse Roy-4's actual JSONL into the analyzer's intermediate matrices; verify expected node-count and modality-count outputs.
- Unit: H1c/H1b/H1a decoding correctness against synthetic matrices.
- Unit: cross-modal (M_dt) handling — when priming centroids exist only in `interoception` and arm-A in `text`, the matrix should be non-empty (this is an explicit Roy-4 finding to confirm).

### Pre-registered output

`docs/experiments/22_roy_5a.md` — the outcome doc, mirroring `21_roy_4.md` structure. Includes the M_tt / M_dt / M_dd headline tables, the decoded sub-hypothesis, and a named pointer to the Stage 2 branch the result triggers. Per-stage Definition of Done.

### What this stage does NOT do

- Does NOT run a new sim. The whole point is to extract maximum signal from already-paid-for Roy-4 data.
- Does NOT commit to any implementation. Implementation work is gated on the H1c/H1b/H1a verdict.
- Does NOT touch `cross_modal_substrate_binding.md`'s archival status. That plan stays cancelled until Stage 4a resurrects it.

## Stage 2 — Conditional implementation per Roy-5a verdict

Each branch ships its own ~few-day implementation PR. Two-lens review per branch per [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md).

### Stage 2a — H1c branch: threshold/centroid tuning

- New env vars `MAXIM_EC_PATTERN_THRESHOLD_TEXT` (override `ECConfig.pattern_complete_threshold` for the text modality only) and `MAXIM_EC_FROZEN_TEXT` (toggle whether text becomes frozen-prototype like interoception). Pair each with an autouse conftest scrub.
- Roy-5a-tune sweep: re-run Roy-2c (existing fixture, existing arms) across a grid of `(threshold ∈ {0.20, 0.30, 0.35, 0.40 default}, frozen_text ∈ {True, False})` to find the setting where arm A's `sense_food_source` count > arm C's.
- New outcome doc `docs/experiments/22_roy_5a_tune.md` with the sweep results.

**Pass criteria:** at least one parameter combination produces non-zero `sense_food_source` calls in arm A's test phase (the result Roy-2c could not produce). If no combination passes, falls through to Stage 2b.

### Stage 2b — H1b branch: encoder A/B sweep

- Kill the `LinguisticEncoder._get_encoder` process-wide singleton — accept explicit `model_name` per `LinguisticEncoder` instance. Default zero-arg construction preserves current behavior (per CC3 "additive only" rule).
- Test 2-3 alternative sentence-transformer models (e.g., `all-MiniLM-L12-v2`, `multi-qa-mpnet-base-dot-v1`) against the same Roy-2c spec. Decode pass criteria same as Stage 2a.
- New outcome doc `docs/experiments/22_roy_5a_encoder_ab.md`.

**Frozen contract impact:** `LinguisticEncoder._get_encoder` API gains an optional `model_name` parameter. Existing callers (none currently pass an explicit name) continue to work.

### Stage 2c — H1a branch: redirect to Stage 3

If Roy-5a returns max cosine < 0.20 across all food-bearing priming centroids vs arm A, the embedding subspaces are structurally incompatible at the encoder level. Threshold tuning cannot rescue this; encoder A/B at the same modality has limited upside. The diagnostic continues into Stage 3.

No implementation work in 2c itself — Stage 3 is the next experiment.

## Stage 3 — Roy-5b: drive→linguistic-channel co-firing scaffold + Hebbian retest

**Conditional on Stage 2c (H1a verdict).** Roy-4 refuted the Hebbian binding rule on the **existing** cradle priming arc, which does not produce co-firing between sensor-drive states and the linguistic clusters the test fixture activates. The structural property the binding rule needs is a temporally-aligned (sensor pattern, drive state, linguistic utterance) triple within a shared salience window — text-modality EC firing co-temporally with interoception-modality EC firing on the SAME priming tick. The existing cradle narrator generates prose ("the infant is in a room with a fire pit nearby") that does NOT co-occur structurally with sensor/drive snapshots; it co-occurs with them in time but not in EC encoding. Worse, in `aut_mode: substrate-primary` (which Roy-4 used) the narrator's text injection is suppressed at `SimulationBridge.send_and_wait`, so the linguistic modality stays empty during priming altogether.

The Stage 3 scaffold is a **drive→linguistic-channel co-firing emitter**: when a drive crosses threshold, the body's percept source appends a short, structurally-stable utterance ("hungry", "thirsty", "warm") into the SAME percept text the sensor snapshot rendered. Because the utterance lands inside the body-state text the substrate-primary AUT does consume, the `LinguisticEncoder` fires on the utterance in the same agent-loop tick the `SensorEncoder` fires on the drive state — closing the co-firing gap.

**Framing note (post-2026-05-28 fold).** Earlier drafts of this plan framed the mechanism as "joint attention + caregiver naming events" — temporally aligned (sensor, drive, narrator-utterance) triples mirroring the way infants acquire word-meaning grounding from caregivers. The pre-merge bio-fidelity review on PR #295 caught that the implementation has no caregiver and no joint attention — the body emits its own utterance derived from its own drive state. The cleaner bio-defense is **interoceptive vocalization**: real interoceptive states (hunger, thirst, thermal discomfort) DO produce co-temporal vocalizations in mammals (crying, whimpering) without requiring an external caregiver. The Hebbian retest property is identical under either framing — both routes establish text-modality EC firing co-temporally with interoception-modality EC firing, which is the only property Stage 3 needs to test. The "joint attention" rationale is preserved as a downstream extension (Stage 4a's resurrected `cross_modal_substrate_binding.md` could explore caregiver-driven scaffolds once the basic binding mechanism is validated), but Stage 3 itself is the interoceptive-vocalization variant.

### Implementation

- Drive→linguistic co-firing scaffold: `_data/components/bodies/infant_humanoid_naming_v1.yaml` + new interoceptive-vocalization emitter in `embodiment/naming_events.py` that deliberately co-fires:
  - When entity-level drive `hunger ≥ 0.7`: body emits "hungry" into the next percept in the same tick window as the SensorEncoder firing.
  - When entity-level drive `thirst ≥ 0.6`: body emits "thirsty" co-tick.
  - When modulator sub-sensor `arms.thermal ≥ 0.7`: body emits "warm" co-tick.
  - Hysteresis bands prevent same-utterance re-emission tick-after-tick once a drive sits above threshold; emission re-arms when the drive drops back below `threshold − hysteresis_band`.
  - Each utterance is a short, structurally-stable text — token-level overlap with the eventual test fixture is the load-bearing property.
- Roy-5b spec `scenarios/roy/roy_5b_iteration.yaml` — same shape as Roy-4 (3 arms, EC trace instrumentation on, substrate-primary aut_mode) but uses the redesigned body. Single-variable change from Roy-4.
- Re-run the Roy-4 analyzer (`scripts/analyze_roy_4_coactivation.py`) on Roy-5b's trace. Pre-registered diagnostic:

| Outcome | Diagnosis |
|---|---|
| At least one would-have-bound edge connects a priming food cluster to an arm-A test-phase node under the default Hebbian rule | **Mechanism rescues with corrected scaffold** — Stage 4a resurrects `cross_modal_substrate_binding.md`. |
| Zero would-have-bound edges across the full parameter sweep, same as Roy-4 | **Mechanism is dead** even under deliberately-scaffolded co-firing — Stage 4b promotes encoder replacement to 1.2+. |

### What this stage does NOT do

- Does NOT relax Roy-4's parameter sweep. The Hebbian rule's defaults (min_cofire=5, min_weight=0.5) and the sweep grid stay identical. If they don't catch the corrected-scaffold co-firing, no further relaxation rescues the mechanism.
- Does NOT touch the existing cradle arcs used elsewhere. The naming-event arc is additive (`infant_humanoid_naming_v1.yaml` is a new file), not a replacement of the existing arc. Production sims keep the production arc; only Roy-5b uses the redesign.

## Stage 4 — Resurrect or promote (conditional on Stage 3)

### Stage 4a — H1a + Roy-5b PASS

Resurrect `docs/plans/archive/cross_modal_substrate_binding.md` with:
- An updated front-matter status: *"Cancelled by Roy-4 (2026-05-13); resurrected by Roy-5b (DATE) with corrected priming scaffold."*
- A new "Prerequisites" section noting that Stage 2 onward depends on the agent running with the naming-event cradle arc (or an equivalent scaffold), and that the existing cradle arcs do NOT provide the co-firing pattern the mechanism requires.
- Stages 2-6 as originally written, with the implementation gated on the corrected scaffold being active.

`grounded_language_acquisition.md` Phase 1's `token_id → ec_node_id` registry can now be populated by binding edges, the original design.

### Stage 4b — H1a + Roy-5b FAIL

Both the existing cradle arc AND a deliberately-scaffolded naming-event arc fail to produce the temporal-coincidence signal the binding rule needs. The encoder-alignment gap is structurally too severe for any reasonable Hebbian binding rule to bridge.

Archive `cross_modal_substrate_binding.md` definitively. Promote encoder replacement (Option C in conversation framing — CLIP-style cross-modal aligned encoder trained on (sensor_pattern, narrator_text, CLI_percept) triples) to a 1.2+ research direction. The natural home is `grounded_language_acquisition.md` Phase 2's symbol-binding-layer scope; Phase 1's registry then needs a different populator (lexicon-style table is a defensible answer here only after the bio-faithful options have been ruled out).

Update `MEMORY.md` with the definitive negative result.

## Existing infrastructure this plan extends (audit before building)

Per both reviews' "extend existing patterns" recommendation:

| Existing surface | What it already does | How Stage 2/3 extends it (if at all) |
|---|---|---|
| `similarity/encoder.py::LinguisticEncoder._get_encoder` | Process-wide paraphrase-mpnet singleton | Stage 2b kills the singleton (additive — explicit model_name param). |
| `similarity/encoder.py::SensorEncoder` | Sensor dict → SHA-basis embeddings → EC (interoception modality) | Stage 3 changes WHEN it fires (alongside narrator utterance) but not its mechanism. |
| `similarity/decomposer.py::AffordanceDecompositionStrategy` | Underscore-split affordance names → noun-phrase chunks | Production-scale lexicon-shaped precedent. NOT extended by this plan — stays scoped to affordance names. |
| `embodiment/component_index.py::ComponentIndex` | Two-layer per-entity discovery: alias hash + embedding cosine | Bio-faithful per-domain index pattern. Cited as the "what a lexicon should look like" precedent; NOT replaced by this plan. |
| `memory/atl.py::ATL.find_or_create(name=...)` | Episodic-reinforcement exact-name → concept lookup | Bio-faithful canonical-concept entry point. NOT replaced by this plan. |
| `decisions/nac.py::cluster_reward_bias` | `(agent_id, cluster_id, tool_signature)` keyed map | Unchanged — Stage 2/3 changes which `cluster_id` gets the right tool's bias, not the keying. |

**Explicit non-introductions:** this plan does NOT introduce a central lexicon module, a `LinguisticFactory` class, a per-engine encoder hierarchy, or a `token_id → cluster_id` table separate from grounded_language_acquisition.md Phase 1. Reviews from both lenses flagged each of those as either thesis-eroding (bio-fidelity) or non-solving (architecture).

## Cross-cutting: env-var inventory

| Env var | Stage | Default | Purpose |
|---|---|---|---|
| `MAXIM_EC_PATTERN_THRESHOLD_TEXT` | 2a | unset → 0.40 | Override pattern_complete_threshold for text modality only |
| `MAXIM_EC_FROZEN_TEXT` | 2a | unset → False | Toggle frozen-prototype semantics for text modality (currently only interoception is frozen) |

Both paired with conftest autouse scrubs per [feedback_opt_in_env_in_hot_paths.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_opt_in_env_in_hot_paths.md). Added to CLAUDE.md env-var table at the time Stage 2a actually ships.

## Cross-cutting: frozen contract impact

Per CLAUDE.md CC3 audit rules:

- Stage 2b's `LinguisticEncoder._get_encoder(model_name=None)` API extension is additive (existing zero-arg calls preserved). Class docstring update declares the new parameter.
- `ECConfig.pattern_complete_threshold` is mutable runtime config; Stage 2a's per-modality threshold extension follows the existing `frozen_centroid_modalities` precedent (per-modality scope at the config layer, not a new field on the result type).
- No new persisted dataclasses introduced.
- No new persistence format-version bumps (Stage 2a/2b/3 are all session-scoped).
- Stage 4a resurrects cross_modal_substrate_binding.md's frozen-contract impact (the `bound_neighbors` field on `PatternCompletionResult` + EC snapshot `_format_version` bump) only if H1a + Stage 3 PASS.

## Definition of done

- **Stage 1 (mandatory):** Roy-5a analyzer landed, run end-to-end on Roy-4's data, outcome doc names the winning sub-hypothesis. PR opened against this plan's branch + merged.
- **Stage 2 (conditional, exactly one branch ships):** 2a OR 2b OR 2c-redirects-to-3. Each ships with two-lens pre-merge review. Pass criterion is a non-zero `sense_food_source` count in arm A's test phase against the existing Roy-2c fixture (the result Roy-2c-through-Roy-4 could not produce).
- **Stage 3 (conditional on 2c):** Roy-5b spec + cradle redesign + Hebbian retest. Outcome doc names PASS/FAIL for the corrected scaffold.
- **Stage 4 (conditional on 3):** either resurrect cross_modal_substrate_binding.md with corrected-scaffold prerequisites, OR archive both plans + promote encoder replacement to 1.2+.

## What this plan does NOT do

- **No central lexicon, no LinguisticFactory.** Both reviews independently rejected this shape; existing per-domain patterns (ComponentIndex, ATL.find_or_create, AffordanceDecompositionStrategy) are the architectural precedent.
- **No commitment to Option A (pre-embedding normalizer) or Option B (post-embedding canonical-node routing).** Architecture review showed A doesn't solve Roy-2c; both reviews showed B unfalsifies the 1.0 thesis.
- **No commitment to encoder replacement (Option C / CLIP-for-cradle).** Only Stage 4b promotes it, and only conditional on H1a + Stage 3 FAIL.
- **No 0.9.1 plan changes.** Wire-A in `release_0_9_1.md` Stage 2 remains the operator-visible interim regardless of which Stage 2 branch ships. Roy-3 remains the 0.9.1 validation iteration.
- **No 1.0 plan changes.** `v1_refinement.md` is unaffected; 1.0 ships when its D1-D3 + C4/C5/C6 cycle complete.
- **No pre-commit to which Roy iteration the 5a/5b ID gets.** If 5a returns H1c quickly, "Roy-5b" may end up being the H1c-tuning sweep itself. The numbering follows the experiment, not the stage.

## Risk register

| Risk | Mitigation |
|---|---|
| Roy-5a's existing-data approach gives ambiguous results (e.g., max cosine sits exactly at 0.40) | Decode ambiguous → run the smallest of Stage 2a's sweep first; if even relaxed thresholds don't surface `sense_food_source` in arm A, fall through to Stage 2b. The framing rule (cheap experiment first) survives either way. |
| Stage 2a's threshold relaxation regresses Roy iterations on different fixtures (e.g., lowers cluster discrimination during normal priming) | The env-var override defaults to current behavior; the sweep only runs against the Roy-2c-specific spec. If H1c wins, the production threshold change requires its own pre-merge review covering all Roy iterations, not just Roy-2c. |
| Stage 2b's encoder A/B doesn't surface a better model in the 2-3 candidates tested | Falls through to Stage 2c → Stage 3 (the H1a path). The diagnostic ladder continues. |
| Stage 3's redesigned cradle arc inadvertently breaks something else in the cradle production sims | Arc is additive (new YAML file); production cradle sims keep production arc. Stage 3 is opt-in via spec selection in Roy-5b's iteration YAML. |
| Stage 4a resurrects cross_modal_substrate_binding.md but the original ~780 LOC scope is now stale relative to other 0.9.1/1.0 changes | Audit the resurrected plan against the current state of `decisions/nac.py`, `similarity/ec.py`, `runtime/agent_loop.py` before treating it as in-flight. The cancel-then-resurrect cycle is a known stale-design risk. |
| Cross-confirmed review findings are wrong, and the lexicon-as-sketched would actually work | Sufficiently small probability per the cross-confirmation rule that we don't gate this plan on hedging. If Roy-5a/5b conclusively rule out all bio-faithful paths AND a user-requested lexicon experiment proves a lexicon outperforms encoder replacement, that's a new plan, not a hedge on this one. |
| User decides post-Stage-2a that "this is too slow, ship the lexicon anyway" | Document the architectural-vs-bio-fidelity review trade-offs in the plan body (this section) so the cost of overriding is visible. The user values pushback per `feedback_pushback_is_valued.md`; the review's recommendation stands regardless of urgency. |

## Reviews folded into this plan

Both reviews kicked off in parallel during the conversation that produced this plan. Verbatim findings are quoted inline above; full transcripts available in the conversation history. The convergent recommendations:

- (Architecture) "Do not write the lexicon plan. Write `roy_5_encoder_alignment_disambiguator.md` (~50 LOC scope, days)." → This plan's filename + Stage 1 scoping.
- (Architecture) "The cross-Roy pattern: tool-name survives, EC cluster identity doesn't — that's two identity schemes for the same concept, and Wire-A already exploits the surviving granularity." → This plan's "no central lexicon" stance + the Roy-2/Wire-A reference in Stage 4b.
- (Bio-fidelity) "There is no biological warrant for a central, hand-curated lexicon." → This plan's "audit before building" stance + the existing-infrastructure table.
- (Bio-fidelity) Originally framed as "joint attention + caregiver naming events" — deliberately-aligned (sensor, drive, narrator-utterance) triples mirroring infant word-meaning acquisition. → This plan's Stage 3 (Roy-5b cradle-arc redesign). **Reframed 2026-05-28** during PR #295 fold: the implementation is drive-derived interoceptive vocalization, not caregiver-driven joint attention. Same Hebbian-retest property under either framing; cleaner bio-defense + no caregiver overclaim. Caregiver-driven scaffolds remain a downstream extension for Stage 4a if the binding mechanism resurrects.
- (Bio-fidelity) "Interim contamination: ship Option A as 'interim' and the thesis erodes silently." → This plan's "No commitment to Option A" non-introduction.
- (Both) "Use ComponentIndex's alias+embedding pattern + ATL.find_or_create as the per-domain index precedent." → This plan's existing-infrastructure table + non-introduction list.

Divergent findings (architecture's Roy-5a vs bio-fidelity's Roy-5b cradle-redesign) sequence cleanly: 5a is cheaper and runs first; 5b only runs if 5a localizes the gap to H1a (encoder subspace incompatibility), which is the case where threshold tuning and encoder A/B both can't rescue.

## References

- [docs/experiments/21_roy_4.md](../../experiments/21_roy_4.md) — Roy-4 FAIL; the empirical floor that motivates this plan.
- [docs/experiments/20_roy_2c.md](../../experiments/20_roy_2c.md) — Roy-2c H1 confirmation; the three sub-hypotheses (H1a/b/c) live inside Roy-2c's H1.
- [docs/plans/archive/cross_modal_substrate_binding.md](cross_modal_substrate_binding.md) — cancelled by Roy-4; resurrectable by Stage 4a (conditional on H1a + Stage 3 PASS).
- [docs/plans/grounded_language_acquisition.md](../grounded_language_acquisition.md) — Phase 1's `token_id → ec_node_id` registry consumes whichever populator Stage 2/3 validates.
- [docs/plans/archive/release_0_9_1.md](release_0_9_1.md) — unchanged by this plan; Wire-A remains the operator-visible interim.
- [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md) — two-lens review template; the pre-merge ritual that produced this plan's reframe.
- [feedback_audit_before_building.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_audit_before_building.md) — both reviews cited this as the rule the lexicon plan violated.
- [feedback_bio_inspired_over_engineering.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_bio_inspired_over_engineering.md) — bio-fidelity review cited this as the rule Options A/B violate.
- [feedback_cross_confirmed_review_findings.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_cross_confirmed_review_findings.md) — both reviews independently rejecting Options A/B is the cross-confirmation that pinned this plan's reframe.
