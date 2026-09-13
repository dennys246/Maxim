# Wiring candidates — harvested from the experiment catalog (2026-09-13)

The catalog-revamp workflow read all 92 `docs/experiments/*.md` and surfaced these reusable *wire-a-subsystem-into-an-environment* lessons. This is a **prioritized backlog** for [the wiring field guide](README.md): each becomes (or extends) a full page as we next touch that subsystem — the field guide's fill-as-we-go ethos, not 11 pages written up front. Ordered as the synthesis ranked them (priority noted inline). Source experiments cited for provenance.


## 1. The substrate→LLM annotation channel is lossy (extends docs/wiring/substrate-learning-channels.md)

**Sources:** 44b_pilot.md, 20_roy_2c.md, 30_wire_a_tau_validation.md, 32_wire_a_post_w1_w2.md, 33_wire_a_post_fix_a.md, 34_wire_a_post_fix_a_b.md, 16_roy_1a.md, 17_roy_1b.md, 23_roy_3.md

HIGHEST PRIORITY — recurs across 9 cards. NAc.get_agent_tool_biases drops cluster_id, max-aggregates per-tool agent-wide, and keys on exact tool-signature strings — so context-dependent value ('good HERE, bad THERE') is structurally inexpressible in the prompt and the affordance-concept/EC transfer machinery is bypassed. An annotation reaching the LLM is necessary but not sufficient: the LLM won't act on an inactive-tool signal, and a named tool whose entity isn't in scene is un-invokable. Consolidate F1–F7 (44b) + the Wire-A fix chain here.


## 2. Cross-agent want transfer: the substrate_merge fold + its coverage-not-volume ceiling

**Sources:** r1_cross_layout.md, 56_four_arm_sharing.md, 57_dose_response_ladder.md, 46_operant_orient_creche.md

HIGH PRIORITY. substrate_merge = ec_merge_aligned + rekey_nac_state (re-keys operant bias 1:1 to the ingested node) + nac_merge. The shipped fold is left-associative convex combination, so pooling scales only by COVERAGE (partial-but-correct biases spanning more situations), never by louder wants — this dilution is the measured merge-cost (57). The shared want is an exact-key (agent,cluster,tool) cache with no cross-layout generalization (r1). nac_merge pools only across agents sharing a perceptual encoder; independent agents need ec_merge alignment first. Read the outcome at the real consumer (NAc_RECOMMEND provenance in recommend_action), never the emitted action (D44).


## 3. Relief-sourced operant credit (score on the recipient, zero relief = zero credit)

**Sources:** 52_nurture.md, 42_substrate_primary_preference.md, r2_drive_premise_check.md, 56_four_arm_sharing.md, behavioral_convergence_exp2.md

HIGH PRIORITY. Make operant credit value = sign(Σ drive_comfort_progress) over the drives a feed/act touched, scored on the recipient — a satiated infant credited like a starving one is the Exp 48 bug. Rides the existing one-turn pending-operant trace; no new mechanism. For live world-owned drives (r2): a corrective-NEED prior in _read_drive_states + _DRIVE_TOOL_AFFINITIES so the prior lands on eat/attack not passive read tools; a measured-relief path (strip modeled self_effect, withhold credit on the live body); and a world that affords the acts. Caveat: cluster credit can form (bias=1.0) yet be a behavioural messenger not a cause — needs state-contingency + choice-space titration to isolate.


## 4. Cross-context readout: load a persisted policy into a different body/world unchanged

**Sources:** 53_cross_context_readout.md, 54_nurture_reachy_body.md, 45_reachy_orient_live.md, 50_readaptation_after_plant_change.md

MED-HIGH. Load the NAc+EC pair unchanged (SHA-verified, apply_decay=False, no credit) into the production substrate-primary path; map the new world's sensor (robot live DoA) to the trained azimuth cluster and its actuator (body yaw) to the turn tool. Train the nursery on the SAME body component the user runs so learned keys are the robot's real tool names (no δ-map adapter). Gotcha: step size must be the target body's own δ; an append-only --out silently pools partial+full runs (fresh path, run-aware verdict).


## 5. Substrate-primary regime + Phase-0 feasibility gate (behavior with no LLM in the action path)

**Sources:** 39_substrate_primary_counter_prior.md, 41_substrate_primary_exploration.md, 13_phase0_harness_smoke.md, 42_substrate_primary_preference.md

MED-HIGH. The substrate-primary regime (actions from drives + SensorEncoder→EC + NAc causal links + confidence gate, prose-silent narrator) borrows NO runtime side-effects from the LLM submit path — drive drift, percept polling, telemetry must be explicitly re-wired (evaluate_failures per tick). Gate every behavioral run behind a Phase-0 triage (EC clusters form, NAc differentiates, proposals actually fire). Entropic (not homeostatic) drive regenerates so the exploitation metric doesn't floor; harm-RATE-by-third is the wrong DV under try-once aversion.


## 6. Embodiment coupling without weak-model tool-calling (reflexes / auto-damage / percept keywords)

**Sources:** 09_percept_reflex_poc.md, 08_component_damage_poc.md, e0_sim_embodiment_poc.md

MED. 14B models can't reliably call damage/sense tools, so drive environment→agent effects directly: percept-keyword reflexes (attack_flinch→torso) apply body-part damage automatically, and an auto-damage fallback routes through the full DamageComponentTool pipeline when the LLM won't call it. Reusable pattern for any environment effect a weak model can't trigger.


## 7. Pre-LLM automatic deliberation (the opt-in-tool egg-before-chicken fix)

**Sources:** 08_deliberation_system.md, 09_pfc_deliberation_cycle.md, sem_learning_loop_poc.md

MED. The LLM never calls an opt-in think tool — it can't know it needs to think unless already thinking. Fix: a ThoughtGate evaluated BEFORE the LLM call, injecting enrichment into the first proposal. Make the bio subsystem non-optional by constructing it inside build_bio_stack() and having the orchestrator READ from BioStack (the 'push the wire into the builder' pattern) rather than a dead manual wire_* helper.


## 8. Hardware sensorimotor policy wiring (DoA→state, body-yaw→action, potential_diff credit) + the head world-frame gotcha

**Sources:** 45_reachy_orient_live.md, 45b_orient_magnitude.md, 45c_flip_bins.md, 45d_magnitude_replication.md, 49_two_joint_centering.md, h2_loudness_bench_2026-08-25.md

MED. state = DoA az_bin, action = body-YAML orient affordances, credit = potential_diff relief, selection = epsilon-greedy over recommend_action. Magnitude needs an expanded ACTION set (not finer bins), with action magnitudes derived from measured hardware gain and the state-bin boundary DERIVED per-robot (a bin straddling the flip point caps performance). LOAD-BEARING: goto_target(body_yaw=X, head=None) COUNTER-ROTATES the head where the mics live — ship an explicit head matrix. Level/loudness rides the existing percept.salience field via PP_AGCGAIN + AEC_SPENERGY reads (no new plumbing).


## 9. Concept-granular cross-modal binding (decompose before encoding; auto-tag modality at episode close)

**Sources:** concept_decomposition_validation.md, p4_stage1_mechanism.md, p3a_episode_binding_sweep.md

MED. Decompose sentences into concept-level noun phrases so each gets its own substrate node with a direct Hebbian binding to its paired vision node (+63.6pp cross-modal recall vs whole-blob). Auto-tag modality at episode close so forgetting is structurally impossible; the binding mechanism only beats bag-of-words when retrieval must traverse structure (hub+chain topology exposes it, clique hides it).


## 10. Substrate observability & falsifiability instrumentation

**Sources:** 21_roy_4.md, 22_roy_5a.md, p4_stage2_v2_post_mortem.md, p4_mug_test_sweep.md, p2_reward_modulation_sweep.md

MED-LOW. Reusable per-tick EC tracing (MAXIM_EC_TRACE_ACTIVATIONS→_emit_ec_activation→sim_log), persist EC.save()+ATL.save() into save_aut_state so post-hoc analyzers read centroids per session, and the anti-vacuity check before trusting any fixture number: 'would this reproduce with a broken substrate / zero Hebbian weights / randomized encoder?' A construction-identity metric deducible from the fixture spec alone is not evidence.


## 11. Persistence round-trip serialization gate

**Sources:** p5_stress_persistence_results.md, p4_stage1_mechanism.md

LOW (belongs as much in persistence-config brief as wiring). Any frozen/persisted record's to_dict()/from_dict() must be tested with NON-default field values across a round-trip — hippocampus-only round-trip tests missed that ATL semantic types silently dropped new consolidation fields on every save/load.
