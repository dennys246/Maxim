# Experiments — Lab Notebook

This directory is the auditable evidence trail for Maxim's substrate research claims. Every experiment that produces data referenced by a plan, a pass/fail decision, or the 1.0 release criteria gets an entry here.

## Why this exists

The 1.0 claim is "cross-session learning without LLM fine-tuning." That claim requires evidence, and evidence requires reproducibility. A result that can't be regenerated from the repo is not evidence — it's an anecdote.

## Structure

```
docs/experiments/
    README.md                           # this file
    protocols/                          # reproduction runbooks (copy-paste commands)
    results/                            # machine-readable JSON (programmatic comparison)
        p0_baseline_sweep.json          # raw numbers from P0 threshold sweep
        ...
    p0_baseline_sweep.md                # methodology, results, decision, repro commands
    ...
```

See [protocols/](protocols/) for per-experiment reproduction runbooks. Every numbered Roy iteration and most substrate experiments have a matching `protocols/<N>_*_reproduction.md`.

## Entry template

Every experiment entry follows this structure:

```markdown
# [Phase] — [Experiment Name]

**Date:** YYYY-MM-DD
**Phase:** P0 / P1 / P2 / ...
**Status:** recorded / superseded / invalidated

## Hypothesis
## Methodology
## Results
## Reproduction
## Raw data
```

## Rules

1. **Every entry must have a Reproduction section** with copy-paste commands. If it can't be reproduced, it doesn't belong here.
2. **Raw data goes in `results/`** as JSON. Markdown entries reference it. Future phases can load prior results programmatically.
3. **Entries are append-only.** Don't edit a recorded entry — if results change, add a new entry and mark the old one as superseded with a link to the replacement.
4. **Tie to git.** Every entry records the git hash it was run against. Results are only valid for that code version unless explicitly re-validated.
5. **Living practice docs link here.** [behavioral_convergence_practice.md](../plans/deferred/behavioral_convergence_practice.md) and [memory_consolidation_practice.md](../plans/deferred/memory_consolidation_practice.md) reference entries by filename when citing experimental evidence.
6. **Plan decisions link here.** When a plan phase passes or fails, the decision entry in the plan links to the experiment that produced the evidence.

---

## Index

### Substrate P0–P4: mechanism validation

| Entry | Date | Status | Decision |
|---|---|---|---|
| [p0_baseline_sweep.md](p0_baseline_sweep.md) | 2026-04-12 | recorded | Fixtures well-calibrated (78.5% @ best operating point). Proceed to P1. |
| [p1_recognition_sweep.md](p1_recognition_sweep.md) | 2026-04-12 | recorded | All P1 criteria met (91.7% ± 2.9% collapse, 3.1% ± 1.3% cross-cluster, 100% persistence). Proceed to P2. |
| [p2_reward_modulation_sweep.md](p2_reward_modulation_sweep.md) | 2026-04-14 | recorded | All P2 Stage 3 gates met: `paraphrase-mpnet@0.70` + `reward=2.0` → +56.0 ± 29.0 pp gain, 0.0 distractor drift, 94% monotone, 9/10 seeds. Substrate P2 gate closed. |
| [p2_sem_pain_cascade.md](p2_sem_pain_cascade.md) | 2026-04-13 | recorded | End-to-end `rusty_sword` cascade validated; agent prefers `drop_weapon` over `slash` after one pain-learning cycle. Proceed to Stage 3. |
| [p3a_episode_binding_sweep.md](p3a_episode_binding_sweep.md) | 2026-04-14 | recorded | Hebbian multi-hop retrieval clears Stage 2 gate: F1 ≈ 0.9955 ± 0.0055 vs TF-IDF baseline at 0.672 (best single-hop). Multi-hop transitive traversal is the mechanism's load-bearing value. |
| [p4_stage1_mechanism.md](p4_stage1_mechanism.md) | 2026-04-15 | recorded | Cross-modal binding plumbing shipped end-to-end on synthetic embeddings; Stage 2 plug-in with real CLIP deferred. |
| [p4_clip_calibration.md](p4_clip_calibration.md) | 2026-04-16 | recorded | CLIP ViT-B-32 Oxford Flowers-102 calibration; 10-class subset pinned in headroom band [0.50, 0.85]. |
| [p4_mug_test_sweep.md](p4_mug_test_sweep.md) | 2026-04-15 | **WITHDRAWN** | v1 conclusion tautological (no distractors); v2 post-mortem at [p4_stage2_v2_post_mortem.md](p4_stage2_v2_post_mortem.md). |
| [p4_stage2_v2_post_mortem.md](p4_stage2_v2_post_mortem.md) | 2026-04-15 | archived | Post-mortem documenting why two Stage 2 attempts were invalidated; v3 measured Option 2 lift = 0 → DEFER. |
| [p4_option2_measurement.md](p4_option2_measurement.md) | 2026-04-16 | recorded | Option 2 (bridge concepts) lift = 0 across 10 seeds; decision: defer as post-Stage-3 cleanup. |
| [p4_cross_modal_sweep.md](p4_cross_modal_sweep.md) | 2026-04-16 | recorded | PASS: 20-seed three-arm head-to-head at CLIP-text threshold 0.8344; forward/reverse rate and false-binding gates all cleared. |
| [p4_vram_audit.md](p4_vram_audit.md) | 2026-04-16 | recorded | RTX 5080 authoritative VRAM audit: CLIP + paraphrase-mpnet co-residency with Qwen-14B confirmed within headroom. |
| [p4_vram_audit_mac_reference.md](p4_vram_audit_mac_reference.md) | 2026-04-16 | reference only | Mac MPS smoke-test of the audit script; NOT authoritative (see p4_vram_audit.md). |

### Substrate P5–P8: persistence, extinction, replay

| Entry | Date | Status | Decision |
|---|---|---|---|
| [p5_stress_persistence_results.md](p5_stress_persistence_results.md) | 2026-04-21 | recorded | PASS — all gates cleared. SemanticMemory serialization bug found and fixed during run. Final 1.0 gate closed. |
| [p6_extinction_results.md](p6_extinction_results.md) | 2026-04-19 | recorded | PASS — Hebbian edge weights decay correctly without reinforcement; graded decay outperforms LRU eviction. |
| [p8_sleep_replay_results.md](p8_sleep_replay_results.md) | 2026-04-19 | recorded | PASS — offline replay during sleep strengthens rewarded associations; retrieval F1 improves on replayed probes. |

### EC centroid drift (experiments 24–28)

| Entry | Date | Status | Decision |
|---|---|---|---|
| [24_roy_paraphrase_diagnostic.md](24_roy_paraphrase_diagnostic.md) | 2026-05-23 | recorded | Diagnostic confirms `CENTROID_DRIFT_COLLAPSE`: 19/20 paraphrase pairs collapse to one EC node sequentially vs 10/10 isolated. Root cause identified; Phase 1 matrix sweep authorized. |
| [25_ec_centroid_drift_fix_phase_1.md](25_ec_centroid_drift_fix_phase_1.md) | 2026-05-23 | recorded | Phase 1 matrix sweep: threshold 0.40 → 0.44 is the sweet spot; frozen-centroid path rejected for `"text"` modality. PR #261 authorized. |
| [26_ec_drift_phase_2_regression.md](26_ec_drift_phase_2_regression.md) | 2026-05-23 | recorded | Phase 2 P1+P2 regression guard passes at threshold 0.44; NAc parameterization follow-up authorized. |
| [28_ec_drift_phase_3_5_nac_parameterization.md](28_ec_drift_phase_3_5_nac_parameterization.md) | 2026-05-23 | recorded | NAc threshold-override base parameterized to track EC live threshold; SHIP verdict — P2 improved, not regressed. |
| [27_ec_drift_phase_4_behavioral.md](27_ec_drift_phase_4_behavioral.md) | 2026-05-24 | recorded | Roy-2c behavioral re-run post-drift-fix: drift was real (cluster_reward_bias_l2 -79%) but NOT the dominant Roy inertness mechanism. Fix ships as substrate hygiene; behavioral claim deferred. |

### Roy harness iterations (experiments 14–23, 29–36)

Companion reproduction runbooks are in [protocols/](protocols/).

| Entry | Date | Status | Decision |
|---|---|---|---|
| [14_g3_roy_preflight_probe.md](14_g3_roy_preflight_probe.md) | 2026-05-11 | recorded | G3: Roy fail-fast LLM pre-flight probe shipped; peer.yml fallback wired. Prevents 10+ min grind on broken local 14B. |
| [15_g4_cluster_reward_wire.md](15_g4_cluster_reward_wire.md) | 2026-05-11 | recorded | G4: substrate-primary cluster_id reward-feedback wire confirmed live (`cluster_reward_bias_l2 = 2.46`). Roy-0 baseline established. |
| [16_roy_1a.md](16_roy_1a.md) | 2026-05-12 | recorded | Roy-1a (llm-primary test): substrate priming writes readable bio-state (cluster bias preserved) but cluster-wire consumer not invoked by llm-primary proposer; salience divergence (KS=0.879) is the load-bearing positive finding. |
| [17_roy_1b.md](17_roy_1b.md) | 2026-05-12 | recorded | Roy-1b (substrate-primary test): same priming, substrate-primary AUT at test; methodology three-pointer refinement. Roy-2 unblocked. |
| [18_roy_2.md](18_roy_2.md) | 2026-05-12 | recorded | Roy-2: wider priming arc (narrated cradle); partial behavioral divergence observed but prompt-mediated not cluster-bias-mediated. Roy-2b recommended. |
| [19_roy_2pc.md](19_roy_2pc.md) | 2026-05-13 | recorded | Roy-2pc: positive-control on engineered-overlap fixture; A ≈ B ≈ C outcome surfaces sample-asymmetry caveat; `min_confidence` probe authorized. |
| [20_roy_2c.md](20_roy_2c.md) | 2026-05-13 | recorded | Roy-2c (`min_confidence=0.0`): H1 confirmed — substrate-primary proposer consumes the cluster wire when cold-start gate is bypassed. Behavioral delta present at `min_confidence=0.0`. |
| [21_roy_4.md](21_roy_4.md) | 2026-05-13 | recorded | Roy-4: FAIL — Hebbian binding cannot close the Roy-2c gap; priming and test EC clusters structurally disjoint; cross-modal binding plan cancelled. Roy-5 disambiguator authorized. |
| [22_roy_5a.md](22_roy_5a.md) | 2026-05-14 | recorded | Roy-5a (cosine localization): H1a confirmed — food concepts land strictly in interoception-modality EC nodes (384-dim), not text-modality (768-dim); cross-modal binding was structurally impossible. Stage 3 cradle redesign authorized. |
| [23_roy_3.md](23_roy_3.md) | 2026-05-23 | recorded | Roy-3 (0.9.1 annotation validation): PRIMARY FAILED — zero `sense_food_source` calls in any arm; wiring bug (empty cluster-bias section rendered as `""`) identified post-run. Wire-A annotation confirmed absent from LLM prompt. |
| [29_roy_3c_bisect.md](29_roy_3c_bisect.md) | 2026-05-24 | recorded | Roy-3c bisect: wire merges did NOT introduce the Roy-3 priming regression; two outside causes (non-code encoder drift + Wire-A intentional decay) explain the observation. |
| [30_wire_a_tau_validation.md](30_wire_a_tau_validation.md) | 2026-05-25 | recorded | Wire-A tau-split validation (Roy-3a-retry): NULL outcome — PRIMARY failed (`sense_food_source` = 0); three structural findings surfaced (Bug A agent_id mismatch, registry gap, imagination blindness). |
| [31_roy_divergence_audit.md](31_roy_divergence_audit.md) | 2026-05-27 | recorded | Convergence/divergence audit of Roy iterations 1a–30 per CLAUDE.md Principle 4; divergence cycle C identified; bird's-eye bisect to encoder replacement authorized. |
| [32_wire_a_post_w1_w2.md](32_wire_a_post_w1_w2.md) | 2026-05-27 | recorded | Post-W1+W2 integration: AMBIGUOUS-WITH-WIRING-BUG; PRIMARY failed but two upstream wiring gaps (Bug A agent_id mismatch + manifest bias gap) identified. Fix A authorized. |
| [33_wire_a_post_fix_a.md](33_wire_a_post_fix_a.md) | 2026-05-27 | recorded | Post-Fix-A: PRIMARY failed (Arm A=0) but Wire-A demonstrably reaches the LLM for the first time; substrate→action conversion gap narrowed to manifest LLM. Fix B authorized. |
| [34_wire_a_post_fix_a_b.md](34_wire_a_post_fix_a_b.md) | 2026-05-28 | recorded | Post-Fix-B: PRIMARY failed (Arm A=0) but failure mode narrowed to substrate→scene-entity semantic mismatch at manifest LLM; pipeline end-to-end validated. |
| [35_roy_5b.md](35_roy_5b.md) | 2026-05-28 | recorded | Roy-5b (naming-event scaffold): Conditional PASS / Ambiguous — literal pre-reg row 1 PASS but (drive, drive) intra-modal edge; EC drift confound unresolved. Confound-isolation re-run authorized. |
| [36_roy_5b_confound_isolation.md](36_roy_5b_confound_isolation.md) | 2026-05-29 | recorded | Roy-5b confound isolation: **Branch A decisive** — EC drift fix (0.40 → 0.44) is the dominant cause of Roy-5b's gap closure; scaffold contribution is zero on the recognition metric. Stage 4a rationale collapses; naming_events.py marked Dormant. |

### Behavioral convergence

| Entry | Date | Status | Decision |
|---|---|---|---|
| [hippocampal_recall_experiment.md](hippocampal_recall_experiment.md) | 2026-04-06 | plan + run notes | Infrastructure validated; protocol defined; early session data at `~/.maxim/sim_reports/20260406_172631/`. |
| [hippocampal_recall_run_notes.md](hippocampal_recall_run_notes.md) | 2026-04-06 | run notes | Direct-injection run (Qwen2.5-14B orch + Mistral-7B AUT); 7/7 turns delivered; AUT behavior recorded. |
| [behavioral_convergence_exp2.md](behavioral_convergence_exp2.md) | 2026-04-17 | recorded | PASS (13/13 hypotheses) — energy-driven consumable learning: bio-pipeline correctly modulates consumable preference based on energy state. |
| [behavioral_convergence_exp3_tier2.md](behavioral_convergence_exp3_tier2.md) | 2026-04-17 | recorded | PASS (12/12) — Tier 2: LLM acts on bio-system learning; experienced agent chose correct vial 10/10 vs random-baseline fresh agent. No fine-tuning. |
| [behavioral_convergence_exp4_tier3.md](behavioral_convergence_exp4_tier3.md) | 2026-04-17 | recorded | PASS (5/5) — Tier 3: organic LLM learning; agent's own interactions alter session-to-session behavior without scripted training. |
| [b4_replanning_results.md](b4_replanning_results.md) | 2026-04-19 | recorded | PASS (4/4 gates) — B4 replanning Stage 3 blind A/B: treatment arm outperforms control on all replanning quality metrics. |

### Cradle sensorimotor + embodiment

| Entry | Date | Status | Decision |
|---|---|---|---|
| [e0_sim_embodiment_poc.md](e0_sim_embodiment_poc.md) | 2026-04-19 | recorded | E0: `--sim` + `--embodiment` integration validated end-to-end; AUT receives SEM affordance tools + pain cascade. |
| [08_component_damage_poc.md](08_component_damage_poc.md) | 2026-04-25 | recorded | H1–H5 all pass: damage cascade, affordance blocking, pain context, scene manifest pre-trigger, DamageComponentTool usage all validated. |
| [09_percept_reflex_poc.md](09_percept_reflex_poc.md) | 2026-04-25 | recorded | H1–H4 all pass: reflexes fire automatically on keywords, body-part targeting correct, pain carries reflex context, habituation reduces intensity. |
| [11_cradle_sensorimotor_poc.md](11_cradle_sensorimotor_poc.md) | 2026-04-26 | recorded | Infrastructure validated; narrator generating cradle scenes; developmental arc confirmed structurally sound. |
| [13_phase0_harness_smoke.md](13_phase0_harness_smoke.md) | 2026-05-09 | recorded | Phase 0 harness (cradle_prelinguistic + substrate-primary) clears success criterion; no behavioral validation claim made. |

### SEM PoC + tool discovery

| Entry | Date | Status | Decision |
|---|---|---|---|
| [valence_annotation_poc.md](valence_annotation_poc.md) | 2026-04-17 | recorded | PASS — SEM pain → Hebbian edge valence annotation pipeline validated end-to-end. |
| [sem_learning_loop_poc.md](sem_learning_loop_poc.md) | 2026-04-17 | recorded | PASS — full pain/success → NAc learning → retrieval cycle validated; agent avoids previously-punished affordances in retrieval. |
| [sem_tool_discovery_s1.md](sem_tool_discovery_s1.md) | 2026-04-20 | recorded | S1: Universal sense + discover_tools + hybrid prompt mode PoC; top-k affordance visibility on turn 1 confirmed. |
| [concept_decomposition_validation.md](concept_decomposition_validation.md) | 2026-04-17 | recorded | PASS — decomposition yields +63.6 pp improvement in concept-level cross-modal recall (36.4% → 100.0%); Stage 1 validated. |

### Imagination + component index

| Entry | Date | Status | Decision |
|---|---|---|---|
| [07_imagination_wiring.md](07_imagination_wiring.md) | 2026-04-20 | recorded | I1+I2+I3 wiring validated: entity extraction → ComponentIndex lookup → DN arousal gate → EntityDesigner → `register_ephemeral()`; session-end decay confirmed. |
| [e25_component_index_discovery.md](e25_component_index_discovery.md) | 2026-04-19 | recorded | E2.5: two-layer discovery (alias hash O(1) + embedding cosine ≥ 0.65) over 62 components / 543 aliases validated; thread-safe via RLock. |

### Deliberation + PFC cycle

| Entry | Date | Status | Decision |
|---|---|---|---|
| [08_deliberation_system.md](08_deliberation_system.md) | 2026-04-20 | recorded | L1+L2 deliberation: 18 stress tests + 25 unit tests green; BioEnrichmentPipeline enrichment and active deliberation control validated. |
| [09_pfc_deliberation_cycle.md](09_pfc_deliberation_cycle.md) | 2026-04-22 | protocol defined | Stages 1+2 shipped; live LLM run protocol defined but not yet executed. |
| [10_cross_session_enrichment.md](10_cross_session_enrichment.md) | 2026-04-25 | recorded | Cross-session enrichment validated: `--resume-sim` surfaces prior-session memories in BioEnrichmentPipeline prompt section, producing measurably different behavior from fresh start. |

### 0.9.1 annotation wires

| Entry | Date | Status | Decision |
|---|---|---|---|
| [temporal_credit_validation.md](temporal_credit_validation.md) | TBD | **STATUS: PRE-REGISTERED, NOT YET RUN** — the `[behavioral]` tag in CLAUDE.md for SCN temporal coupling is provisional pending execution of this protocol. See [protocols/temporal_credit_validation.md](protocols/temporal_credit_validation.md) for the reproduction runbook. | Four simulation sets defined; cross-session fire-danger transfer is the headline claim. |

### Exp 37 — 1.0 graduation gate

| Entry | Date | Status | Decision |
|---|---|---|---|
| [37_cross_session_graduation.md](37_cross_session_graduation.md) | 2026-05-30 → 2026-06-13 | fired; mixed/partial | Cross-session memory persists, but behavioral shift is Goldilocks-dependent and does not establish general prior override. See the cross-model row below. |

### Post-1.0 substrate, embodiment, and heartbeat experiments (38–50)

This index is generated manually from the experiment records and therefore does
not publish a hard-coded total. Status in the linked record wins if this summary
ever drifts.

| Entry | Date | Status | Decision |
|---|---|---|---|
| [37_cross_model_results.md](37_cross_model_results.md) | 2026-06-13 | in progress; four local-model fires complete | Substrate visibility is Goldilocks-dependent; the R1 variant produced the first clean Wire-A attribution. Deferred cloud comparisons are not results. |
| [38_counter_prior_substrate.md](38_counter_prior_substrate.md) | 2026-06-13 | fired | Across the completed model set, carried substrate did not override a strong wrong prior. |
| [39_substrate_primary_counter_prior.md](39_substrate_primary_counter_prior.md) | 2026-06-13 | pre-registered predecessor | Frozen counter-prior design; the executable line continued through Exp 41/42. |
| [40_counter_prior_goldilocks.md](40_counter_prior_goldilocks.md) | 2026-06-16 | fired | Counter-prior dominance also held on Qwen2.5-32B, where the prior-aligned task had shown headroom. |
| [41_substrate_primary_exploration.md](41_substrate_primary_exploration.md) | 2026-06-19 | void | Mechanism operated, but the experiment could not isolate exploration and the primary metric floored. |
| [42_substrate_primary_preference.md](42_substrate_primary_preference.md) | 2026-06-23 | **graduate; maintained** | Substrate-primary safe-vs-harm discrimination passed; scope is discrimination without an LLM, not override of an LLM prior. |
| [42b_drive_pain_fold_revalidation.md](42b_drive_pain_fold_revalidation.md) | 2026-07-29 | fired + verified | Exp 42's graduation held post-fold; the saturated metric detects breakage but not moderate degradation. |
| [43_gaze_operant_substrate.md](43_gaze_operant_substrate.md) | 2026-06-28 | complete feasibility study | EC category generalization beat the lookup baseline in simulation; no production behavior claim. |
| [44_substrate_counterfactual.md](44_substrate_counterfactual.md) | 2026-07-28 | exploratory | Counterbalanced positive at modest N; not EARNED-clean. |
| [44b_pilot.md](44b_pilot.md) | 2026-08-10 | pilot complete; not a result | Apparatus works, but name-mismatched control and non-independent axes must be resolved before confirmatory freeze. |
| [45_reachy_orient_live.md](45_reachy_orient_live.md) | 2026-07-15 | complete | Live substrate-primary direction learning; current graduation scope and healthy-hardware caveats live in the behavioral ledger. |
| [45b_orient_magnitude.md](45b_orient_magnitude.md) | 2026-07-16 | pass on recorded hardware | Magnitude mechanism result; interpret with the later hardware/DoA re-characterization. |
| [45c_flip_bins.md](45c_flip_bins.md) | 2026-07-16 | pass on recorded hardware | Derived bin boundary removed the measured magnitude ceiling. |
| [45d_magnitude_replication.md](45d_magnitude_replication.md) | 2026-07-23 | complete | Replication and cross-session policy transfer recorded. |
| [45e_orient_s4_population_readout.md](45e_orient_s4_population_readout.md) | 2026-07-27 | complete | Population readout resolved the recorded cell-starvation ceiling. |
| [46_operant_orient_creche.md](46_operant_orient_creche.md) | 2026-07-22 | complete | Scripted operant learning and NAc merge/federation, with no LLM in the action path. |
| [47_habituation_novel_in_noise.md](47_habituation_novel_in_noise.md) | 2026-07-22 | complete | Scripted habituation enabled novelty detection under dense noise. |
| [48_cradle_mother_seam.md](48_cradle_mother_seam.md) | 2026-08-14 → 2026-08-18 | **partial, apparatus-v2 — superseded by Exp 52** | Mother effect re-earned; LEARNED-v2 failed and the sweep indicates credit-tipped phase locking, not graded orienting. Retired v1 numbers are not current evidence. Stays as the apparatus case study; its PARTIAL stands for the constant-credit apparatus (`--credit constant`). |
| [49_two_joint_centering.md](49_two_joint_centering.md) | 2026-08-04 | complete | H1 supported; H2 and H3 passed after corrected credit attribution. |
| [50_readaptation_after_plant_change.md](50_readaptation_after_plant_change.md) | 2026-08-07 | pre-registered | No result yet; runs only after its healthy-hardware preconditions. |
| [52_nurture.md](52_nurture.md) | 2026-08-25 | **complete — EARNED (graduate)** | Caregiver-taught orienting through hunger relief: Phase A scripted taught 0.892 vs satiated/yoked/no_feed 0.496; Phase B embodied (apparatus v3, shuffled order, 12 seeds/arm) taught 0.878 vs satiated 0.441 (fed 35%, credited 0%) vs no_feed 0.413 — gate v3 PASS, L2 apparatus check clean. One session, n = 12/arm; sign-only relief credit. Supersedes Exp 48. |
| [53_cross_context_readout.md](53_cross_context_readout.md) | 2026-08-26 · R1 2026-08-28 | **complete — Exp 53 APPARATUS; Exp 53b EARNED; R1 replication at `v1.1.0` (clean tree) PASS** | Cross-context readout on the physical Reachy Mini: the Exp 52 Phase B infants' persisted NAc+EC loaded unchanged (no credit, SHA-verified) into the production substrate-primary path. Gate I: 120/120 live percepts pattern-complete into the nursery's clusters. Exp 53 (δ 0.55 rad): taught chose the right direction 36/36 but delivered directedness 0.75 (every miss the −0.2 target — overshoot) → pre-registered APPARATUS. Exp 53b (δ 0.30 rad, the robot's own step; the one declared change): taught **1.00/1.00/1.00**, satiated 0.00 (no action), no_feed 0.50 (side-blind) → **PASS**. +0.2 turns the wrong way 18/18, as predicted (three-bin representation). One session, n = 3/arm. **R1 (2026-08-28):** same protocol at tag `v1.1.0` from a clean worktree after the 1.1.0 re-score found the originals' provenance unestablishable — taught **1.00/1.00/1.00**, satiated 0.00, no_feed 0.50, sign rule 1.0; the EARNED status now rests on R1. |
| [protocols/exp53_cross_context_readout_preregistration.md](protocols/exp53_cross_context_readout_preregistration.md) | 2026-08-26 | pre-registration (amendments 1–2 pre-data) + [53b](protocols/exp53b_cross_context_readout_delta_preregistration.md) | Cross-context readout: the Exp 52 Phase B infants' persisted NAc+EC loaded unchanged (no credit, SHA-verified) into the production substrate-primary path on the live Reachy Mini (`bodies/infant_operant`, DoAFeed, explicit δ map). Gate I = instrument (live percepts complete into the nursery's audio clusters; frozen probe correct with margin > 0.11) with a pre-registered STOP; Gate T = delivered directedness ≥ 0.70 and ≥ 0.20 above satiated/no_feed controls. Recorded outcome gates the cut, not a PASS. |
| [54_nurture_reachy_body.md](54_nurture_reachy_body.md) | 2026-08-27 → | **Phase A GRADUATE; Phase B/C pending** | Nurture on the robot's own body (`bodies/reachy_mini_infant`, the four Reachy orient affordances, keys `tool:reachy_mini_turn_*`): taught late **0.858** (at ceiling from act 1), satiated 0.472, no_feed 0.514 — MOTHER-TAUGHT +0.34, HUNGER-NECESSARY +0.39, L2/S3 clean, 12 seeds/arm. Magnitude NOT resolved (taught big-fraction ≈ 0.27 in every stimulus bin vs 0.50 controls — a flat normal-step preference). The learned map mirrors Exp 53's (left bias on the far-left bin, right bias on the centre bin; seed 43 the Exp 53 shape). Sweep-declared Phase B targets {−0.6, −0.5, +0.2, +0.3}, exploratory −0.2. |
| [protocols/exp54_nurture_reachy_body_preregistration.md](protocols/exp54_nurture_reachy_body_preregistration.md) | 2026-08-26 | pre-registration (amendment 1 pre-data) — 1.1.x item 15 | Nurture on the robot's own body: `bodies/reachy_mini_infant` (extends `reachy_mini`, entity name `reachy_mini` so learned keys are the robot's, innate azimuth drive nulled, infant hunger/thirst added). Phase A nursery under gate v3 → Phase B readout through the production factory path (no δ map; step size is the agent's choice) → Phase C consulted under plain `bodies/reachy_mini`. The prerequisite for sharing the taught want (Oasis case study). |
| [h2_loudness_bench_2026-08-25.md](h2_loudness_bench_2026-08-25.md) | 2026-08-25 | bench note; not a result | Sound level is one daemon REST read away (`AEC_SPENERGY_VALUES` pre-AGC + `PP_AGCGAIN`); loudness/onset salience is not part of 1.1 — design deferred to 1.1.1 (roadmap item 18). |

### V1 phased attribution

| Entry | Date | Status | Decision |
|---|---|---|---|
| [12_v1_phased_attribution.md](12_v1_phased_attribution.md) | 2026-04-30 | recorded | **Clean pass.** Phase A (substrate-only) recalls token across sessions; all 7 phases recalled. Confound flags scheduled for removal in 1.0. |
