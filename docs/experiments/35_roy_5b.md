# Roy-5b — Cross-modal binding retest with naming-event scaffolding

**Date:** 2026-05-28 (run completed 2026-05-28 22:25 local)
**Plan:** [roy_5_encoder_alignment_disambiguator.md § Stage 3](../plans/roy_5_encoder_alignment_disambiguator.md) · [persona_convergence_crucible.md § "Iteration log"](../plans/persona_convergence_crucible.md)
**Companion:** [21_roy_4.md](21_roy_4.md) (Roy-4 baseline — refuted Hebbian binding on standard cradle arc) · [22_roy_5a.md](22_roy_5a.md) (Stage 1 — H1a confirmed)
**Spec:** [scenarios/roy/roy_5b_iteration.yaml](../../scenarios/roy/roy_5b_iteration.yaml)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../scenarios/roy/roy_2pc_holdout.yaml) (reused unchanged from Roy-2pc / Roy-2c / Roy-4)
**Reproduction:** [protocols/35_roy_5b_reproduction.md](protocols/35_roy_5b_reproduction.md)
**Analyzer:** [scripts/analyze_roy_4_coactivation.py](../../scripts/analyze_roy_4_coactivation.py) (reused as-is — single-variable change vs Roy-4 is the embodiment, not the analysis)

## Status

**CONDITIONAL PASS by literal pre-registration; AMBIGUOUS by structural shape — surfaced to user for explicit classification call.** This verdict has three structural caveats the user needs to weigh before authorizing Stage 4a:

1. **The matched edge at default is `(drive, drive)`, not the cross-modal `(drive, linguistic)` binding the naming-event scaffold was designed to enable.** Both endpoints are priming `sense_food_source` clusters — two near-duplicate food drives co-firing. The scaffold's express bio-mechanism (drive→linguistic-channel co-firing per [docs/plans/roy_5_encoder_alignment_disambiguator.md § Stage 3 lines 164-170](../plans/roy_5_encoder_alignment_disambiguator.md)) did NOT clear the default rule. Cross-modal binding the scaffold designed for IS happening (4 `(drive, linguistic)` edges at `min_cofire ∈ {1, 2, 3}`) but drops out at the default `min_cofire=5` threshold. The disambiguator plan's pre-registration table row 3 explicitly hatches this case as **"Ambiguous: structural shape not matching the binary diagnostic → Pause, surface to user."** The literal pre-registration text in row 1 is satisfied; the structural diagnostic in row 3 is also satisfied. Both reviewers (architecture lens + bio-fidelity lens) independently cross-confirmed the over-claim risk.

2. **The EC drift fix confound is unresolved.** [PR #264 (2026-05-24)](https://github.com/dennys246/Maxim/pull/264) raised `ECConfig.pattern_complete_threshold` from 0.40 to 0.44 between Roy-4 (2026-05-13) and Roy-5b (2026-05-28). The 0% → 100% recognition gap closure observed in Roy-5b (see Node-set overlap section below) could be scaffold-driven, threshold-driven, or both. **A ~25 min wall re-run of the Roy-4 spec at HEAD with `MAXIM_SUBSTRATE_PATH=1` and the standard `infant_humanoid` body would isolate the attribution.** This is a GATE on the verdict, not a phase of an already-authorized resurrection plan — if the gap closes from threshold alone, the binding mechanism was never the active ingredient and the Stage 4a authorization rationale collapses.

3. **The "100% recognition gap closure" headline has a denominator caveat.** Arm A has 10 unique EC nodes in both Roy-4 and Roy-5b, but priming dropped 37 → 18 unique nodes. With a smaller, more attractive priming basin (partly from EC drift fix collapsing the basin, partly from naming-utterance scaffold producing structurally-stable text centroids), arm-A percepts are more likely to pattern-complete onto existing priming nodes simply because there are fewer broader basins to land on. The 100% number is real but the attribution is split between "stronger encoder alignment" and "fewer-broader basins to align to."

**Per the kickoff's "Ambiguous → Pause" rule and the cross-confirmed reviewer convergence, this experiment does NOT authorize Stage 4a.** The Phase 2 deliverable is this surfaced verdict + the proposed confound-isolation re-run as the next experiment. Stage 4a eligibility is contingent on (a) the confound-isolation re-run isolating the scaffold's contribution, AND (b) the user's explicit classification call on whether the (drive, drive) intra-modal edge counts as evidence the cross-modal binding mechanism is rescued.

## Pre-registered diagnostic logic (from [roy_5_encoder_alignment_disambiguator.md § Stage 3](../plans/roy_5_encoder_alignment_disambiguator.md))

| Outcome | Diagnosis | Triggered |
|---|---|---|
| At least one test-phase active node has a would-have-bound edge to a priming `sense_food_source` cluster under DEFAULT rule (`min_cofire=5, min_weight=0.5`) | **PASS** — corrected scaffold rescues the binding mechanism. Stage 4a: resurrect `cross_modal_substrate_binding.md` with the naming-event prerequisite. | **Literally yes** — 1 matched edge: `6e582e62 ↔ e3f59dbe`, both `sense_food_source` clusters, w=4.668, cofire=5, modality=**(drive, drive)** |
| Zero would-have-bound edges between priming and test clusters across the full Roy-4 parameter sweep (`min_cofire ∈ {1, 2, 3, 5}` × `min_weight ∈ {0.01, 0.1, 0.5}`) | **FAIL** — mechanism is dead under deliberately-scaffolded co-firing. Stage 4b: promote `jepa_cross_modal_alignment.md` to 1.2. | NO |
| Ambiguous (edges form only at very permissive thresholds, or **in a structural shape not matching the binary diagnostic**) | **Pause** — document, surface to user. | **YES** — the matched edge's modality_pair is `(drive, drive)`, not the cross-modal `(drive, linguistic)` the scaffold was designed to produce. The cross-modal edges form at `min_cofire ≤ 3` but drop below default. Two-lens reviewers (architecture + bio-fidelity) independently classified as Ambiguous. |

**Both row 1 (literal) and row 3 (structural) are triggered.** The disambiguator plan's row 3 hatch ("Pause — document, surface to user") is the dominant classification per the kickoff's "force into 4a or 4b only when unambiguous" rule. The verdict is **Ambiguous — surfaced** rather than PASS.

## What shipped (Phase 1 prerequisite — see [project_roy_5b_phase_1_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_roy_5b_phase_1_shipped.md))

PR #295 (merged 2026-05-28 at commit 9d9dae6) shipped the Phase 1 infrastructure that Roy-5b needs:

- [src/maxim/embodiment/naming_events.py](../../src/maxim/embodiment/naming_events.py) — deterministic drive→linguistic co-firing emitter. `NamingPattern` parser + `collect_sensor_values` (modulator-walking helper) + `derive_naming_utterances` (hysteresis state machine) + `format_naming_section` (percept-text formatter).
- [src/maxim/_data/components/bodies/infant_humanoid_naming_v1.yaml](../../src/maxim/_data/components/bodies/infant_humanoid_naming_v1.yaml) — opt-in body declaring `naming_events:` metadata (`hunger > 0.7 → "hungry"`; `thirst > 0.6 → "thirsty"`; `arms.thermal > 0.7 → "warm"`). Standard `infant_humanoid.yaml` is unchanged.
- [src/maxim/embodiment/percepts.py](../../src/maxim/embodiment/percepts.py) — `EmbodimentPerceptSource` wired to parse naming patterns at construction and append utterances to body-state text on each tick.
- [scenarios/roy/roy_5b_iteration.yaml](../../scenarios/roy/roy_5b_iteration.yaml) — Roy-5b iteration spec; single-variable change vs Roy-4 is the body.
- [tests/unit/test_naming_events.py](../../tests/unit/test_naming_events.py) — 42 unit tests covering parser, sensor collector (entity + modulator paths), hysteresis state machine, within-tick formatting, edge cases. Regression guard.

Two pre-merge review BLOCKs folded into PR #295:
- **B1 — modulator-walking:** initial draft of `collect_sensor_values` walked only entity-level `vital_metrics`, which silently missed dotted-key modulator drive specs like `arms.thermal`. The "warm" utterance would never have fired across Roy-5b's 50 priming turns. Fix: walk the modulator layer too.
- **B2 — within-tick co-firing test:** initial test surface verified hysteresis state advancement across ticks but did not assert the within-tick property (utterance reaches the SAME percept the sensor snapshot rendered). Fix: add `test_within_tick_naming_section_appended_after_body_state` pinning the structural property the binding retest requires.

## Operational note — first run wasted on missing env var

The first Roy-5b run (started 2026-05-28 20:33 local, 24 min wall) was kicked off with the env command the [21_roy_4_reproduction.md](protocols/21_roy_4_reproduction.md) protocol documents, which omits `MAXIM_SUBSTRATE_PATH=1`. The result was 152 drive-modality EC events but **zero linguistic-modality events** — the diagnostic prerequisite was missing because [MemoryHub._encoder](../../src/maxim/integration/memory_hub.py) requires the env var to wire up. Per the canonical lesson at [feedback_substrate_path_env_var_for_roy.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_substrate_path_env_var_for_roy.md), this gotcha has burned multiple prior Roy iterations and now this one.

The Phase 2 reproduction runbook ([protocols/35_roy_5b_reproduction.md](protocols/35_roy_5b_reproduction.md)) is updated to include `MAXIM_SUBSTRATE_PATH=1` in the command and a new troubleshooting entry. The Roy-4 reproduction protocol has the same gap (it omits the env var) and should be updated separately — orthogonal to this experiment's scope.

## Result (run #2, 2026-05-28 22:25 local)

Wall: **1467.3s (~24.5 min)** — same shape as Roy-4 (1547s). Pre-flight clean (`outcome: ok`, latency 546ms). Priming completed all 5 stages × 10 turns; arms a/b/c each completed 10 turns.

### Per-arm

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260528_221032` | 10 | 273.51 | cancel |
| b | blank | "You are a hungry infant" | `20260528_221505` | 10 | 295.62 | cancel |
| c | blank | neutral | `20260528_222001` | 10 | 295.61 | cancel |

### Headline pairwise diffs

| Pair | `cluster_reward_bias_l2` | (keys) | `causal_link_Δ` | `episodes_Δ` | `valence_KS` (p) |
|---|---|---|---|---|---|
| **a_vs_b** | **0.3521** | 5 | +159 | +683 | 0.997 (0.000) |
| **a_vs_c** | **0.3525** | 5 | +159 | +683 | 0.997 (0.000) |
| b_vs_c | 0.1807 | 4 | 0 | 0 | 0.000 (1.000) |

**Comparison vs Roy-4:** Roy-4's `a_vs_b` cluster_reward_bias_l2 was 2.4678 (10 keys differ). Roy-5b's is 0.3521 (5 keys differ) — the substrate signal magnitude dropped ~7×. This is structurally consistent with the recognition gap closing (test percepts now pattern-complete onto priming nodes, so fewer cluster-key divergences accrue).

### EC activation instrumentation — capture summary

The unified `/tmp/roy_5b_ec_trace.jsonl` (`MAXIM_LOG_FILE` bridge format) was partitioned by absolute timestamp into priming + per-arm slices using session start times from `~/.maxim/roy/roy-5b/result.json`. Session-window mapping:

| Phase | Session ID | Start (unix) | End (unix) |
|---|---|---|---|
| Priming stage 1 (`act1_neonatal_a`) | `20260528_220031` | 1780027231 | (continues) |
| Priming stage 2 (`act1_neonatal_b`) | `20260528_220325` | 1780027405 | (continues) |
| Priming stage 3 (`act2_cradle_a`) | `20260528_220511` | 1780027511 | (continues) |
| Priming stage 4 (`act2_cradle_b`) | `20260528_220656` | 1780027616 | (continues) |
| Priming stage 5 (`act3_consolidation`) | `20260528_220848` | 1780027728 | 1780027832 (= arm A start) |
| Arm A | `20260528_221032` | 1780027832 | 1780028105 |
| Arm B | `20260528_221505` | 1780028105 | 1780028401 |
| Arm C | `20260528_222001` | 1780028401 | 1780028757 (= arm C start + duration + 60s grace) |

Per-phase event counts after partition:

| Phase | Sessions | EC events | Linguistic | Drive |
|---|---|---|---|---|
| Priming (5 stages) | 5 | 168 | 88 | 80 |
| Arm A | 1 | 34 | 10 | 24 |
| Arm B | 1 | 34 | 10 | 24 |
| Arm C | 1 | 34 | 10 | 24 |

**vs Roy-4:** priming 148 events (68 linguistic + 80 drive). Roy-5b adds +20 linguistic events in priming — empirically the "hungry"/"thirsty"/"warm" utterance encodings landing in EC text-modality. The drive-modality count is unchanged (same SensorEncoder path firing on the same drive-tick pattern).

### Node-set overlap — observation + denominator caveat

| Phase | Unique nodes | (linguistic / drive) | Overlap with priming | Overlap with priming food clusters |
|---|---|---|---|---|
| Priming | 18 | (16 / 2) | — | 2 / 2 |
| Arm A | 10 | (8 / 2) | **10 / 10 (100%)** | **2 / 2 (100%)** |
| Arm B | 10 | (8 / 2) | (not analyzed in default-rule edge match) | — |
| Arm C | 10 | (8 / 2) | (not analyzed in default-rule edge match) | — |

**Roy-4 comparison:** Roy-4 had 37 priming unique nodes; arm A had 10 unique nodes with **0/10 priming overlap** and **0/6 priming food cluster overlap**.

**Roy-5b: every arm-A test-phase node is a priming-acquired node, and both priming `sense_food_source` clusters fire in arm A test phase.** This is striking when compared to Roy-4's 0%. **But the attribution is not clean — three competing explanations all fit the data:**

1. **Scaffold-driven:** the naming-event scaffold produces structurally-stable utterances ("hungry", "thirsty", "warm") that pattern-complete onto the same handful of text-modality centroids, and arm-A test percepts then pattern-complete onto those centroids. Bio-defensible: the scaffold's whole purpose was to create stable cross-modal references.
2. **Threshold-driven:** PR #264 (2026-05-24) raised `ECConfig.pattern_complete_threshold` 0.40 → 0.44 between Roy-4 and Roy-5b. The new threshold rejects marginal admissions during priming (priming node count 37 → 18) and the resulting fewer-broader basins are more attractive landing zones for arm-A percepts. A Roy-4-spec re-run at HEAD (same threshold, NO scaffold, with `MAXIM_SUBSTRATE_PATH=1`) isolates this — without that run, we cannot attribute the gap closure to the scaffold vs the threshold.
3. **Denominator artifact:** arm-A unique-node count is 10 in both Roy-4 and Roy-5b, so the numerator side is fair, but priming's 37 → 18 drop means each priming basin is broader. "10/10 priming-acquired" can be partly explained by priming basins being fewer, more attractive landing zones — not stronger cross-source alignment per se.

**Authorizing Stage 4a on this data would commit to "scaffold rescued cross-modal binding" before disambiguating which of the three explanations dominates.** The confound-isolation re-run (Branch A of the post-Phase-2 next-experiment options below) names the dominant cause cleanly before any binding-plan resurrection LOC is written.

### Co-firing analysis (the binding-rule diagnostic)

Default Hebbian rule (`min_cofire=5, min_weight=0.5`) on Roy-5b priming:

| Metric | Roy-5b | Roy-4 |
|---|---|---|
| Priming unique nodes | 18 | 37 |
| Priming co-firing pairs | 98 | (not reported in 21_roy_4.md) |
| Priming would-have-bound edges | 1 | 2 |
| Priming food clusters | 2 | 6 |
| Arm A test events | 34 | 47 |
| Arm A test active nodes | 10 | 10 |
| **Matching priming↔test edges (PASS criterion)** | **1** | **0** |

The single matched edge at default:

```
node_a       node_b       weight   cofire   modality_pair
6e582e62 ↔   e3f59dbe     4.668    5        (drive, drive)
```

Both endpoints are priming `sense_food_source` clusters (verified via `cluster_reward_bias` keys decoded from `aut_nac.json`). Both also fire in arm A's test phase. The analyzer's match condition `a_food and b_food and (a_test or b_test)` is satisfied — this is the strongest of the three matching-edge shapes the analyzer recognizes.

### Parameter sweep — PASS across the entire range

| `min_cofire` | `min_weight` | Priming would-have-bound edges | Matching priming↔test edges | Top cross-modal edge present? |
|---|---|---|---|---|
| 1 | 0.01 | 98 | 9 | Yes — 4 (drive, linguistic) edges + 4 (drive, drive) co-firing once + the default (drive, drive) edge |
| 1 | 0.1 | 98 | 9 | Yes — same |
| 1 | 0.5 | 97 | 9 | Yes — same |
| 2 | 0.01 | 56 | 3 | Yes — 2 (drive, linguistic) edges + 1 (drive, drive) |
| 2 | 0.1 | 56 | 3 | Yes — same |
| 2 | 0.5 | 56 | 3 | Yes — same |
| 3 | 0.01 | 27 | 2 | Yes — 1 (drive, linguistic) + 1 (drive, drive) |
| 3 | 0.1 | 27 | 2 | Yes — same |
| 3 | 0.5 | 27 | 2 | Yes — same |
| 5 (default) | 0.01 | 1 | 1 | No — only (drive, drive) survives |
| 5 (default) | 0.1 | 1 | 1 | No — only (drive, drive) survives |
| **5 (default)** | **0.5 (default)** | **1** | **1** | **No — only (drive, drive) survives** |

**PASS at every sweep point.** The (drive, drive) edge between the two priming food clusters survives at every threshold because both endpoints co-fire 5 ticks during priming with combined activation 4.668, comfortably above all weight thresholds.

**Cross-modal binding edges DO form but at sub-default-threshold strength.** The 4 (drive, linguistic) edges that emerge at `min_cofire ∈ {1, 2, 3}` connect the central drive cluster `6e582e62` (one of the two food clusters) to four distinct text-modality nodes (`8f5ab48a`, `c2925f41`, `ea9372fd`, `6f705165`) — these are the naming-utterance encodings the scaffold was designed to produce. Their co-firing weight tops out at 2.176 (cofire=3) before dropping below default `min_cofire=5`.

## What this proves

### Definitively proved

- **The instrumentation prerequisite + scaffold prerequisite are both met.** `MAXIM_EC_TRACE_ACTIVATIONS=1` emitted 270 EC traces across 4 sessions; `MAXIM_SUBSTRATE_PATH=1` routed text through LinguisticEncoder; the naming-event scaffold added +20 linguistic priming events vs Roy-4's standard cradle arc.
- **The pre-registered PASS criterion's literal text is met at the DEFAULT rule.** 1 matching priming↔test edge at `(min_cofire=5, min_weight=0.5)`. But the matched edge's modality_pair is `(drive, drive)`, not the cross-modal binding the scaffold was designed to enable.
- **The cross-modal binding the scaffold was designed to produce IS happening, but sub-threshold of the default rule.** 4 (drive, linguistic) edges form at `min_cofire ∈ {1, 2, 3}`; the central food drive cluster `6e582e62` co-fires with all four naming-utterance text nodes during priming. Co-firing weight tops at 2.176 (`cofire=3`), below the default rule's `min_cofire=5` requirement. The scaffold IS producing cross-modal co-firing — just not at the strength the default Hebbian rule was tuned for.
- **Same-modality (drive, drive) binding between two near-duplicate priming food clusters DOES clear the default rule.** Roy-4 didn't have this either; the closest analog in Roy-4 was the 6 priming food clusters firing in near-isolation (1 co-firing tick total). Roy-5b's two food clusters co-fire 5 ticks with combined weight 4.668 — the scaffold's pattern of drive-threshold-crossings produces more inter-cluster co-firing than Roy-4's standard arc did.

### Conditionally eligible (NOT cleanly authorized)

- **Stage 4a of [roy_5_encoder_alignment_disambiguator.md](../plans/roy_5_encoder_alignment_disambiguator.md)** is **ELIGIBLE pending two gates**, not authorized:
  - **Gate 1 — confound-isolation re-run:** Roy-4 spec at HEAD with standard `infant_humanoid` body + `MAXIM_SUBSTRATE_PATH=1` ships first. If the recognition gap closes from the EC drift fix alone (no naming-event scaffold), the binding mechanism was never the active ingredient and the resurrection rationale collapses.
  - **Gate 2 — user classification call:** the (drive, drive) matched edge is row 1 of the pre-registration (literal PASS) AND row 3 (structural Ambiguous). The user picks which row dominates the verdict. The kickoff's "force into 4a or 4b only when unambiguous" rule recommends row 3 / Pause.

### Cleanly refuted

- **Roy-4's Hebbian-binding-cannot-fire-at-all conclusion is at least partly scaffold-specific.** Some Hebbian binding fires under the corrected scaffold where none fired under the standard arc. But "the cross-modal binding the scaffold was designed to enable rescues Roy-2c" is NOT cleanly proved here — that's the structural ambiguity the verdict surfaces.

### Still unfalsified

- **Whether the 0% → 100% recognition gap closure is scaffold-driven, threshold-driven, or both.** Confound-isolation re-run (Branch A) names the dominant cause. **This is THE gate on Stage 4a eligibility** — not a Phase 0 prereq of an already-authorized plan.
- **Whether the (drive, drive) intra-modal edge counts as evidence the cross-modal binding mechanism is rescued.** The Hebbian binding plan's bio-defense per [cross_modal_substrate_binding.md](../plans/cross_modal_substrate_binding.md) is about **cross-modal** co-activation (visual+auditory in superior colliculus, sensorimotor in M1). Two food drives co-firing 5 ticks is intra-modality near-duplicate collapse — different mechanism, different bio-defense. Roy-5b shows the Hebbian rule CAN fire at default; it does NOT show the cross-modal flavor that's the plan's actual claim.
- **Whether `(drive, linguistic)` cross-modal edges become the dominant edge type under tunable scaffold variants** (lower hysteresis, finer drive thresholds, multi-tick utterance persistence). Roy-5b's cross-modal edges sit below default `min_cofire=5` because each utterance fires only a handful of times per drive crossing. Out of Roy-5b scope.
- **Whether 100% test-percept pattern-completion onto priming nodes generalizes to non-food, non-cradle fixtures.** Roy-5b reused Roy-2c/Roy-4's `roy_2pc_holdout.yaml`. A Roy-5c on a divergent fixture would confirm. Out of Roy-5b scope.
- **Whether the Hebbian rule generalizes beyond ONE specific co-firing shape.** Roy-5b shows the rule fires at default for two near-duplicate priming food clusters that both co-occur in narrator-driven food scenes. It does NOT show the rule fires across DIFFERENT concepts within a modality, nor across modalities at default thresholds. The generalization claim that motivates the resurrected binding plan rests on this single co-firing pattern.

## What this means for [roy_5_encoder_alignment_disambiguator.md](../plans/roy_5_encoder_alignment_disambiguator.md)

**Stage 3 COMPLETE with Ambiguous verdict — surfaced.** Stage 4a is **eligible**, not authorized. Stage 4b stays parked at "candidate". The plan's front-matter status updates to "Stage 3 COMPLETE 2026-05-28 — Conditional PASS / Ambiguous; Stage 4a eligibility gated on confound-isolation re-run + user classification call".

**[cross_modal_substrate_binding.md](../plans/cross_modal_substrate_binding.md)** front-matter status updates to "CANCELLED by Roy-4 (2026-05-13); CONDITIONALLY ELIGIBLE for resurrection by Roy-5b (2026-05-28) — gated on (a) confound-isolation re-run isolating scaffold contribution from EC drift fix contribution, (b) user classification call on whether the (drive, drive) intra-modal matched edge counts as evidence the cross-modal binding mechanism is rescued. **Resurrection implementation does NOT happen without explicit user authorization following the gates above.**"

**[jepa_cross_modal_alignment.md](../plans/jepa_cross_modal_alignment.md)** front-matter status stays at "Stage 4b candidate" — does NOT promote, does NOT archive. Annotated: "Roy-5b produced Conditional PASS / Ambiguous (not the clean FAIL that would have promoted this plan to 1.2 in-flight); parks pending Stage 4a's eligibility gates resolving."

## What this still does NOT prove

- **Whether the 0% recognition gap closure is scaffold-driven or threshold-driven.** The natural follow-up (re-run Roy-4 spec at HEAD with `MAXIM_SUBSTRATE_PATH=1`) would disambiguate. Add to the Stage 4a-prereq checklist before binding plan implementation begins.
- **Whether the same scaffold + the same binding rule succeeds on a DIFFERENT fixture.** Roy-5b's fixture is identical to Roy-2c/Roy-4. A second fixture (non-food, non-cradle) would confirm the scaffold generalizes vs reflects fixture-specific dynamics.
- **Whether SCN phase-aligned co-activation would tighten the cross-modal edge weights** to clear the default rule's `min_cofire=5` threshold. Current rule uses same-tick co-firing; phase alignment would only further filter — it cannot rescue a sub-threshold edge but COULD strengthen one already above threshold (which the (drive, drive) edge already is).

## Reproduction

See [protocols/35_roy_5b_reproduction.md](protocols/35_roy_5b_reproduction.md).

## Recommendation — next experiment, NOT Stage 4 implementation

**Run the confound-isolation re-run as Roy-5b-confound-A.** Roy-4 iteration spec at HEAD with the standard `infant_humanoid` body (NO naming events) + `MAXIM_SUBSTRATE_PATH=1` + `MAXIM_EC_TRACE_ACTIVATIONS=1`. ~25 min wall, no new code. Produces a clean disambiguation:

- **If recognition gap closes from threshold alone (arm A overlap with priming ≥ ~50%):** the EC drift fix was the dominant cause. The naming-event scaffold contributed marginally or not at all. Stage 4a's resurrection rationale collapses — the binding mechanism was never the active ingredient in Roy-5b's overlap.
- **If recognition gap stays close to Roy-4's 0% (overlap < ~10%):** the naming-event scaffold IS the dominant cause. Stage 4a eligibility moves forward to gate 2 (user classification call on the (drive, drive) edge).
- **If recognition gap is partial (~20-50%):** both factors contribute. The scaffold's marginal contribution doesn't justify ~780 LOC of binding-plan implementation; the right move is scaffold variants (Roy-5c with tightened hysteresis) before any Stage 4a code.

**Do NOT authorize Stage 4a yet.** The confound-isolation re-run is the GATE on the verdict, not a Phase 0 of an already-authorized plan.

**Do NOT promote [jepa_cross_modal_alignment.md](../plans/jepa_cross_modal_alignment.md) to 1.2 in-flight.** Roy-5b did NOT produce the clean FAIL across all parameter sweeps that the JEPA promotion needs. Plan stays parked as Stage 4b candidate.

**Surface the Ambiguous verdict to the user along with the confound-isolation re-run proposal.** Two-lens reviewers (architecture + bio-fidelity) independently confirmed the over-claim risk in the unqualified PASS reading; the user's explicit classification call resolves the (drive, drive)-counts-or-doesn't question that the literal pre-registration text cannot resolve.

## PR

(filled in after PR open)
