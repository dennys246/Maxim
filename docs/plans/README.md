# Maxim Plans

Current version: **1.3.3** (`pyproject.toml` + `src/maxim/__init__.py`; PyPI: https://pypi.org/project/pymaxim/ — `main` is ahead of PyPI by policy, CLAUDE.md §Versioning; checked by `scripts/lint_version_sync.py`). 1.3.3 = **"Release pipeline"**: the fixes maxim-pulse found, the Console's live stream from a clean install and the first CI-published release, no new behavioural claim (breaking security changes: every spawned llama-cpp-server keyed, the agent cannot change the operator's internet switch; T1-2 STALE under `release-1.3.3-T1-2`). 1.3.2 = **"Decomposition"**: the agent loop taken apart under characterization tests plus the ratchets and gates that hold it, no new behavioural claim (a breaking security change to leader defaults, the evidence gate, T1-13/14/15 PARTIAL under B1, T1-2 STALE under `release-1.3.2-T1-2`). 1.3.1 = **"Hardening"**: fixes and their guards, no behavioural claim (security, the public format freeze + Oasis release format v2, public-API correctness, release integrity; the ledger's trigger walk with Exp 10 re-run MAINTAINED (narrow)). 1.3.0 = **"Oasis-2"**: the survival world — **Exp 60 EARNED** (anticipatory drowning-avoidance) + **Exp 61 EARNED** (the survival fear transfers between agents) + the **R3 survival benchmark** (instrument + frozen baseline) + Exp 56 re-baselined on 1.20.4. 1.2.1 = **"Spoken-code loop"** (the pairing pieces; composition unwired — corrected in 1.3.0). 1.2.0 = **"Oasis"**: **Exp 56 EARNED** + **Exp 57 PARTIAL** + the **Hivemind P2P exchange**. 1.1.4 = **"The world seam"**, 1.1.3 = **"Reachability"**, 1.1.2 = **"Decomposition"**, 1.1.1 = **"Enforcement"**.
Now (2026-10-16): **1.3.2 "Decomposition" shipped**; the path to 1.4.0 is four patches (owner decisions 2026-10-16, and 2026-10-17 to split 1.3.3): **1.3.3** the release pipeline + pulse's blockers (shipped 2026-10-24), **1.3.4** the robot and recall fixes (owner decision 2026-10-24), the **atonia gate** on its own clock (the next free number when done), the **orchestrator** decomposition ("1.3.5"), **loop phase 2** + the E1 instrument ("1.3.6"), then **1.4.0** (hard-gated by `scripts/lint_release_gates.py`, #1300). Detail: [roadmap_1_3_x.md §Path to 1.4.0](roadmap_1_3_x.md#path-to-140-owner-decision-2026-10-16).
- [roadmap_1_3_x.md](roadmap_1_3_x.md) — **the 1.3.x line** (drafted 2026-09-19 from the v1.3.0 re-score; **re-scoped 2026-09-26**; **1.3.1 SHIPPED 2026-09-27 as "Hardening"**; **1.3.2 SHIPPED 2026-10-16 as "Decomposition"**; carries the path to 1.4.0): 1.3.1 = what users need now — the security cluster, public-API correctness, release integrity (nightly lane, network block), the public format freeze and the two ingest defects it surfaced (#913/#914), plus everything on `main` since 1.3.0, each with its guard and no new behavioural claim; 1.3.2 = the enforcement ratchets carried from 1.3.1, then the `agent_loop` decomposition with behaviour-preservation gates and the ledger triggers discharged. Both before 1.4's experiments.
- [outstanding.md](outstanding.md) — **the standing register of owed work**, cross-cutting and not release-scoped, each entry verified at its stated date. Created 2026-09-20 after an audit found the 1.1.x item-16 block entirely shipped while CLAUDE.md still declared one of them a KNOWN GAP.
- [roadmap_1_4.md](roadmap_1_4.md) — **the 1.4 plan** (working title "Anticipation", re-pointed 2026-09-18, five-lens reviewed): generalization + sequence credit + a trajectory instrument on the survival rig — **Exp 62 EARNED 2026-09-20 (Phase 1 closed; PARTIAL, floor matched, since 2026-10-15)** → E1 conflict → [relief store] → E2 pockets → E3 strategic breathing; primitives on a variant body; release thresholds T1–T4 set, T5/T6 conditional, T7/T8 engineering only (engram integrity; grounding truth), T9 conditional (the first grounding claim); the release name is fixed at the transaction from the highest EARNED rung.
- [archive/roadmap_1_3.md](archive/roadmap_1_3.md) — **the 1.3 plan** ("Oasis-2", the survival world; rescoped 2026-09-09): the phased survival-loop → shared-survival-wants roadmap with per-rung mechanism-dependency markers. · [deferred/second_body_staging.md](deferred/second_body_staging.md) — **DEFERRED 2026-09-18 on a physical trigger (a second body exists); NOT the 1.4 plan** — the perception-fabric sequencing (Stage A duck → Stage B engine seam → Stage C fabric); the 1.4 plan is [roadmap_1_4.md](roadmap_1_4.md), and the fabric's own revive trigger was re-keyed 2026-10-07 to the grounding line's GL3.
- [archive/quality_burndown.md](archive/quality_burndown.md) — **the quality burndown, ARCHIVED 2026-09-20**: merged into [roadmap_1_3_x.md](roadmap_1_3_x.md) §1.3.1, which now carries its re-verified remainder; kept as the record of Batches 0–2.
The current grading baseline is the **v1.3.1 blind re-score**
([../limits/score_cards/2026-09-27-claude.md](../limits/score_cards/2026-09-27-claude.md), three
firewalled evidence agents, grades unadjusted) and its Codex twin
([2026-09-27-codex.md](../limits/score_cards/2026-09-27-codex.md)). **The v1.3.2 re-score is owed** (cut 2026-10-16). Previous: the v1.3.0 re-score
([2026-09-19-claude.md](../limits/score_cards/2026-09-19-claude.md)) and its Codex twin; the owed Codex 1.1.0 and 1.2
cards are superseded. Historical: the Claude 1.1.0 re-score is
[../limits/score_cards/2026-08-27-claude.md](../limits/score_cards/2026-08-27-claude.md) (Codex pairing owed).
1.1 "Sensorimotor" shipped 2026-08-26 on two pre-registered results: Exp 52 (the want is
learned) and Exp 53b (it reads out on the physical robot).

**Directory last deeply audited 2026-10-16** (after the 1.3.2 cut: a six-reader review of `docs/` and `docs/plans/`; one root plan archived, shipped plans that code cites by path marked SHIPPED where they stand, the index's unlisted and misfiled entries fixed, fired deferred triggers re-homed). **Before that, 2026-09-19** (at the 1.3.0 release: 15 plans archived — 10 from the
root, 5 from `deferred/` that carried no revive trigger — and 18 moved from the root to `deferred/`
with a trigger written on each; the root is 16 active plans). Previous deep audit 2026-08-27. Root = active work only. Superseded release checklists belong in
`archive/`; the 1.1 cut exists only in the roadmap above.

**New here?** [glossary.md](glossary.md) decodes the coded IDs (`CC3`, `Wire-A`, `Roy-4`, `P2`, `NAc`, GRADUATE, …) — including the prefixes that mean different things in different plans.

## Version Roadmap

| Version | Theme | Status |
|---------|-------|--------|
| **0.6** | **Generalizable embodiment** — E0 sim wiring + E1 Asset Foundry | **Published** |
| **0.7** | **Self-generating simulations** — Imagination, B3 Acting Coach, E2-E3, F3-F5, SEM discovery, gating, deliberation | **SHIPPED** (2026-04-20) |
| **0.8** | **Cognitive maturity + embodiment** — WM+Exec, PFC cycle, temporal credit, display overhaul, reflex system, proprioceptive discovery, affordance concept transfer, entity ownership, component damage | **SHIPPED** (2026-04-25) |
| **0.9.1** | **Substrate-annotates-LLM-context** — Roy-2c probe, Stage 0 telemetry, Wires A+1+2+3, Roy-3 validation, EC centroid-drift fix | **SHIPPED** (2026-05-25, plan: [release_0_9_1.md](archive/release_0_9_1.md)) |
| **0.9.2** | **Config unification + Hivemind shareability + LLM timeout scalability** — `~/.config/maxim/config.json`, `maxim config/model/substrate` CLIs, `hivemind/` substrate bundle, TTFT keepalive, per-tier timeout, context-overflow admission, stall detector, leader-local harness singleton guard | **SHIPPED** (2026-06-05) |
| **0.9.3** | **Loud optional-dependency failures** — `utils/optional_deps.py` centralises 45+ import sites; missing requested backend raises `OptionalDependencyError` instead of silently returning empty responses | **SHIPPED** (2026-06-06) |
| **1.0** | **Validation + stabilization + grounding** — cross-session proof, bio-system protocol freeze, sensorimotor cradle, SEM world enrichment, SCN feedback loop, cleanup, docs | **SHIPPED** (2026-06-17) |
| **1.1** | **"Sensorimotor" release closure** — merged embodiment work plus D13/D14 liveness, stable Python API truth, hermetic fast tests, atomic NAc+EC invalidation, architecture-audit regression enforcement, S4, remaining heartbeat chapters, release mechanics, and agent-guidance convergence. **Zero new mechanisms.** | **Published 2026-08-26** (tag `v1.1.0`; Exp 52 + Exp 53b recorded, item 18 → 1.1.1) — history in [roadmap](archive/roadmap_1_1_to_1_3.md#11-cut-line--reconciled-2026-08-19-reopened-2026-08-25) |
| **1.2** | **Oasis** — peer substrate sharing + P2P protocol. | **PUBLISHED 2026-09-09** (tag `v1.2.0`, [PyPI](https://pypi.org/project/pymaxim/1.2.0/)) — [Exp 56](../experiments/56_four_arm_sharing.md) EARNED (headline) + [Exp 57](../experiments/57_dose_response_ladder.md) PARTIAL (dose-response scaling); P2P slices A→D merged (#651–#654, [scope](archive/hivemind_p2p_scope.md)); R1/R2 offline nulls; gates 1/3/7/8 closed. |
| **1.3** | **"Oasis-2" — the survival world** (rescoped 2026-09-09) — learning reward FROM the game: the survival loop (R2's 3 breaks), the survival world with marked classrooms, **the survival reflex tier** (innate startle/flee below deliberation), and the headline **shared survival wants**. Plan: [archive/roadmap_1_3.md](archive/roadmap_1_3.md); designs: [survival_world_1_3.md](archive/survival_world_1_3.md), [intrinsic_motivation_1_3.md](deferred/intrinsic_motivation_1_3.md). | **PUBLISHED 2026-09-19** (tag `v1.3.0`, [PyPI](https://pypi.org/project/pymaxim/1.3.0/)) — Exp 60 + Exp 61 EARNED, R3 shipped as an instrument + frozen baseline (nothing graduated), Exp 56 re-baselined on 1.20.4. |
| **1.4** | **Working title "Anticipation"** — re-pointed 2026-09-18 ([roadmap_1_4.md](roadmap_1_4.md)): generalization + sequence credit + a trajectory instrument on the survival rig; the name is fixed at the release transaction from the highest EARNED rung. **+ the grounding line (GL0–GL6, 2026-10-07, [grounding.md](grounding.md))**: concepts made similar by what they do to the body, inside Phase 5's slots; T8 (engineering) gates 1.4.0, while T9 (the first grounding claim) is conditional and never co-headlined with E3. **"Shared perception" DEFERRED 2026-09-18 on a physical trigger** (a second body exists; plan intact in [deferred/second_body_staging.md](deferred/second_body_staging.md); its perception half — the fabric, placement and modality resolution — re-keyed 2026-10-07 to the grounding line's GL3) — it was: richer video/audio (cochlear front-end, vision encoder, binding, three-factor calibration — [cross_modal_perception_fabric.md](deferred/cross_modal_perception_fabric.md), [deferred/three_factor_credit_assignment.md](deferred/three_factor_credit_assignment.md)) **plus sharing at the perception layer**, the **DN-canonical orienting reflex** (perception-coupled; distinct from 1.3's survival reflex), the microduck, breeding (Exp 55 re-pointed 2026-09-27 to [coding_world.md](coding_world.md)). Pivotal may-fail (Stage 0c); shared-perception inherits R1's transfer bound. | Planned |

**Discipline (carried forward from the 1.1 freeze):** existing behavior is not assumed live
because a type or bridge exists — release claims require an executed contract test or a
graduated behavioral row. New mechanism work in 1.1.x is limited to the roadmap's
follow-through list; Oasis/Hivemind mechanisms start only when the 1.2 entry gates are
satisfied.

## 1.0 — SHIPPED (2026-06-17)

The unified plan lives in the archive: [archive/v1_refinement.md](archive/v1_refinement.md) — all hard requirements closed 2026-06-15 (C1–C6 cleanup, CC1–CC13 contract clarifications, B1/B2/B4 bio-system stabilization, P1–P4 pipeline completion, D1–D3 docs).
Release writeup: [The Honest Benchmark](../announcements/maxim_1_0_release.md) (legacy HTML copy in [archive/html-guides/](../archive/html-guides/README.md)) — what shipped + the pre-registered cross-session experiments ([Exp 37](../experiments/37_cross_session_graduation.md) Goldilocks zone · [Exp 38](../experiments/38_counter_prior_substrate.md) counter-prior dominance · [Exp 40](../experiments/40_counter_prior_goldilocks.md) Goldilocks counter-prior) that mapped where the substrate helps vs where the LLM prior dominates. *Re-assessment 2026-10-25 ([#1301 re-assessment](../experiments/narrator_defects_reassessment.md)): bounded.* These fires ran with the AUT told `respond` was NOT AVAILABLE (#1260); Exp 38 Sonnet's interaction and ablation values are withdrawn, so the four-model dominance reads: dominance at GPT-4o, DeepSeek-V3 and R1, with Sonnet's verdict resting on first-contact over 3 of 5 arm-A deceptive trials. The 1.0 benchmark gate scope + disposition: [archive/benchmarking_1_0.md](archive/benchmarking_1_0.md). Post-1.0, the substrate-primary safe-vs-harm claim was EARNED mechanism-level by [Exp 42](../experiments/42_substrate_primary_preference.md) (GRADUATE, PR #380).

## Active

Everything in the root is in-flight, owned by a roadmap item, or a living doc.
Deep-audited 2026-10-16 after the 1.3.2 cut, with a 2026-10-17 follow-up pass (see the stamp above).

### Roadmap + release

- [roadmap_1_3_x.md](roadmap_1_3_x.md) → [roadmap_1_4.md](roadmap_1_4.md) — **the live
  scope authority** (path to 1.4.0 decided 2026-10-16, 1.3.3 split 2026-10-17; 1.3.1 and 1.3.2
  shipped): **1.3.3** the release pipeline and pulse (shipped), **1.3.4** the robot and recall fixes, the
  **atonia gate** (own clock, next free number), the orchestrator decomposition ("1.3.5"), loop phase 2 and the
  B2→B3→B4 floor chain ("1.3.6"), then **1.4.0**
  ([§Path to 1.4.0](roadmap_1_3_x.md#path-to-140-owner-decision-2026-10-16)).
- [archive/roadmap_1_3.md](archive/roadmap_1_3.md) — the 1.3 plan, now a RECORD of what shipped as 1.3.0
  (its Phases 4–6 moved to `roadmap_1_4.md`; Phase 1b is deferred on its trigger).
- [archive/roadmap_1_1_to_1_3.md](archive/roadmap_1_1_to_1_3.md) — **HISTORICAL** (it was the sole scope
  authority through 1.2): the 1.1.x follow-through list (items 1–16) and the pre-1.2 gates.
  Its title no longer describes its contents; it is kept for the item numbering other
  plans cite.
- [../bugs/README.md](../bugs/README.md) — **KNOWN-DEFECTS LEDGER.** Every row carries
  evidence and a disposition (D32/D33 filed 2026-08-26).
- [../limits/score_cards/](../limits/score_cards/) — repo-grade baselines, re-scored at
  each release cut by two assessors grading blind. Current:
  [2026-09-27-claude.md](../limits/score_cards/2026-09-27-claude.md) at tag `v1.3.1`
  (three firewalled evidence agents, grades unadjusted, evidence committed beside it);
  and its Codex twin [2026-09-27-codex.md](../limits/score_cards/2026-09-27-codex.md), graded blind
  to it. **The v1.3.2 re-score is owed** (cut 2026-10-16). Previous: `v1.3.0`'s pair. The owed Codex 1.1.0 and 1.2 cards are
  **superseded** — they would grade states nobody works on.

### Substrate-primary spine (shipped 1.2; living architecture)

- [memory_strength_and_forgetting.md](memory_strength_and_forgetting.md) — **a 1.4 PARALLEL LINE
  opened 2026-09-21**: a hippocampal forgetting model (storage strength from existing signals,
  retrievability on an experience clock, honest activation, sleep with floors, a `maxim config`
  byte budget). Opt-in with defaults pinned. Phases 0 (#844) and 1 (honest activation, #846)
  SHIPPED; **Phase 2 decisions resolved 2026-09-21** (validated in the LLM sim worlds first); **Phase
  2S, the survival gate (#848)**: survival percepts score salience 0 and pattern completion cannot
  cue sensor concepts, so nothing there ever uses a memory yet. **Phase 2's look-back is unblocked** —
  R4's look-back review (2026-09-24) decided tagging looks back over the Hippocampus record.
  **Status 2026-10-17:** 2d-2 (retroactive tagging, #897) and 2S-d (the situation cue, recall only,
  #898) are BUILT, merged 2026-09-25; 2S-e (B) is PARKED (#900, the `time_of_day` wrap #899, not a
  generalization gap). It feeds T4a (2S-d's recall is the consumerless mechanism T4a walks) and engram
  E6 (2S-e, under T7).
- [social_referencing.md](social_referencing.md) — **PROPOSED 2026-09-24, plan only; a parallel line.** Consult a locally held, signature-verified Oasis mirror only when the agent is both ignorant and being hurt; hold the answer as advice in a foreign layer apart from its own experience; promote advice only when it demonstrably worked. Exp C (may-fail): does the gate limit a stale Oasis's harm beyond the same advice given at random? Depends on public_oasis Phase 0 (scheduled); src after the fence lifts (the fence lifts when 1.3.6's decomposition slices land — the point at which T0 can pass); rig after E3. Split from the language line. Stays active as a parallel paper line (owner 2026-10-17).
- [coding_world.md](coding_world.md) — **PROPOSED 2026-09-27, plan only; a parallel line.** Revives [deferred/coding_habits_oasis.md](deferred/coding_habits_oasis.md) by owner request. Does an aversion learned from pain bind to the act or to being watched? Exp W probes Exp 60's EARNED fear with a second player as observer (offline key-geometry replay first); Exp K asks it in a coding world (hand-authored patch variants in Docker, measured pain, harness-only oracle labels, a tamper arm), then Exp 55's transfer. The one missing mechanism is a pain producer for the existing `credit_operant_reward` write (plus the world's backend and body). Stack Overflow is not grounded text (its uses live in social_referencing and grounded_word_binding; no need to pay). Stays active as a parallel paper line (owner 2026-10-17): no src before the 1.4.0 cut; claims are 1.5 candidates.
- [baseline_floors.md](baseline_floors.md) — **Status 2026-10-16: B0 shipped and B1 ran (`FLOOR_MATCHED_Q`, 2026-10-14; T1-13/14/15 now PARTIAL). B2's code prerequisites are 1.3.6 exit items; the chain is B2 ([#1352](https://github.com/dennys246/Maxim/issues/1352)) → B3 ([#1353](https://github.com/dennys246/Maxim/issues/1353), Reflexion first, owner 2026-10-17) → B4 ([#1354](https://github.com/dennys246/Maxim/issues/1354)); B4, T1-2's re-fire, runs before 1.4.0.** **PROPOSED 2026-10-10, a parallel line (owner decisions BF1–BF2).** A bare LLM (`bare_llm`), a generic-memory LLM (`llm_notes`) and a non-LLM learner (`tabular_q`, `random`) beside every claim; until B1, no ledger row had been compared with an agent lacking the bio layers. BF2: a new claim earns only over its floor (M46). B0 (provenance fields) is merged; B1 (the Exp 60 non-LLM floor) RAN 2026-10-14 and returned `FLOOR_MATCHED_Q` (a tabular Q-learner matched FEAR at the first post placement), so T1-13 is `PARTIAL (narrow: floor matched)`; the bare loop (B2) waits for the `start_simulation_mode` decomposition, so no new LLM-path claim can be EARNED before it. Revises the owner's 2026-10-10 draft: the Exp 60 world cannot host a fair bare-LLM arm (priors hit the ceiling on the pre-probe), and Exp 37's narrator is a cross-session memory channel.
- [experience_clock_decay.md](experience_clock_decay.md) — **PROPOSED 2026-10-17, plan only ([#1361](https://github.com/dennys246/Maxim/issues/1361)).** NAc decay and body drift move off loop passes and the wall clock onto the agent's experience clock (`ExperienceClockDriver`, which the hippocampus already rides): owner decisions 2026-10-17 — anchor in-session decay to the driver, and no forgetting while switched off. Slices S0–S6 with strict red gates; supersedes [deferred/scn_decay_anchoring.md](deferred/scn_decay_anchoring.md).
- [maxim_hivemind.md](maxim_hivemind.md) — **Status 2026-10-16: 1.2 shipped (#651–#654); this is now the living architecture source for the Queen tier and the gauntlet that public_oasis and social_referencing consume.** **1.2 = Oasis + Hivemind.** B5 shareability
  SHIPPED (#305–#311); the Oasis ingestion contract, the re-keyed merge (gate 6) and the
  bundle action namespace (gate 7) are the 1.2 work.
- [public_oasis.md](public_oasis.md) — **Status 2026-10-17: Phase 0 items 1, 2, 5, 6 and 7 are done; [oasis_entry_index_v2.md](oasis_entry_index_v2.md) records the owner's item 3/4 decisions (license `CDLA-Permissive-2.0`; the privacy-read exemplar, Exp 56 seed-42 1.20.4). Whether the human privacy read itself has been done is unverified; item 5's server-side read tier remains. No longer "scoping only".** **PROPOSED 2026-09-19; Phase 0 SCHEDULED 2026-09-24** (social_referencing depends on it), scoping only: a public Oasis on own
  hardware at `oasis.pymaxim.bio`. Verdict **publish, do not accept** — Phase 1 (read-only release
  tier over a tunnel) rides entirely on shipped infrastructure and adds no trust decision; Phase 2
  (submissions) stays deferred because V1's operator allowlist IS the trust boundary and the
  promotion gauntlet that would replace it has zero of its four prerequisites. Re-opens decision
  point 4 for publication only; the discovery-only website stance is preserved. Self-hosted sibling
  of [hf_hive_repository.md](hf_hive_repository.md).

- [engram_formation.md](engram_formation.md) — **Status 2026-10-16: E1–E4, the T7 gate, are DONE (E3 2026-10-05, E4 2026-10-06); E5–E7 ride with other owners. T7 still owes the release-PR caller grep.** **ACTIVE 2026-09-25, a 1.4 parallel line
  ("Engram integrity", release threshold T7).** Fix list from a four-path audit of how engrams form:
  E1 Cerebellum never saved (#908, done 2026-10-04), E2 motor-engram docs overclaim + read side
  undeclared-dormant (#909, done 2026-10-04), E3 unreachable `[DANGEROUS]` annotation (#910), E4 text-only widening + drift measurement
  (#911) gate 1.4.0; E5 the `time_of_day` wrap (#899, Phase 5 keying), E6 episodic readout (#848,
  2S-e) and E7 motor-engram resurrection (Phase 5 graded predictor) ride on existing owners. State
  page: [../wiring/engram-formation.md](../wiring/engram-formation.md).

### The grounding line (1.4, owner decisions G1–G8, 2026-10-07; release thresholds T8, T9)

- [grounding.md](grounding.md) — **ACTIVE 2026-10-07, the umbrella plan (the
  [#1120](https://github.com/dennys246/Maxim/issues/1120) audit).** Its thesis is that concepts become
  similar by what they do to the body. #1120 found the EC split in two, with nothing joining the halves
  and no positive producer: a body world (sensor clusters at 0.85, the only path that acts without the
  LLM) and a word world (EC `text` at 0.44, only with `MAXIM_SUBSTRATE_PATH=1`; affordance NAMES only
  from the `--sim` orchestrator). No EARNED row depends on the word path.
  - **Stages:** GL0 truth → GL1 paper, the L0 gate and a name-vs-consequence census → GL2 autonomic
    (a record-only, b regulatory fixes, c the positive producer) → GL3 relay (GL3.B0 census,
    runnable now; GL3.B1 registry+provenance; GL3.B3 the first track slice, a flag-off thermal fan-out;
    GL3.B4 preemption as a declared rung arm) → GL4 latent forward model
    → GL5 the first claim (T9, conditional, never co-headlined with E3) → GL6 a selection consumer as a
    declared arm.
  - **Placement:** it fills roadmap 1.4 Phase 5's slots (the relief store and the graded predictor),
    and T8 (engineering only) gates 1.4.0. Body world first; the word world comes later, as the
    innate-prior tier.
  - **Subsumes** [grounded_language_acquisition.md](grounded_language_acquisition.md) for grounding,
    and [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) (owner decision
    G5). Narrated consequences are stamped `narrated` and used at a declared discount (G6); harness writes
    to a drive (rescue, heal, respawn, teacher and mother feeds) are `apparatus`, recorded and never
    trained on or credited (G16, 2026-10-09, from the GL1 design review). Under TR20 (G24, 2026-10-24)
    the "never credited" half acts only once GL3.B1's credit flag flips; until then every writer,
    `apparatus` included, credits as today and is only recorded.
- [tool_failure_credit.md](tool_failure_credit.md) — **Status 2026-10-16: Stage 1 (#1207) and Stage 2 (#1229) are merged, with #1231 and #1221; Stage 3 waits for `kind` on `Reaction` (autonomic step 2).** **ACTIVE 2026-10-10 ([#1200](https://github.com/dennys246/Maxim/issues/1200)).** Tool failure was never wired into learning; putting it on the PainBus as it stands makes learning worse (the context-similarity NAc subscriber steals the bridge's event). Staged design B (owner TF1–TF3): Stage 1 a stale-success defect (#1207); Stage 2 failure credit through `ToolPainBridge.record_tool_failure` with a real RPE; Stage 3 a felt-only FRUSTRATION PainBus signal once `Reaction` carries kind.
- [autonomic_layer.md](autonomic_layer.md) — **PROPOSED 2026-10-07 (GL2); GL2a BUILT 2026-10-09 and merged (#1194: the record-only tool-path `InteroceptiveOutcome` in `embodiment/sem.py`, `embodiment/body.py`, `tools/base.py::ToolOutput`, `runtime/executor.py`; no reader acts on it). GL2b and GL2c are not built.** The signed
  body-consequence record, `InteroceptiveOutcome` (deviation, pain, relief, urgency), the predictor targets.
  - **The regulatory defects the audit found:** no heat corrective need; an infant touch-burn that
    peaks at 0.2 and is classed as a drive, not nociception; percept valence keyed on the sufferer, not
    the cause; and no positive producer (`EntropicDriveSpec`'s satisfaction Reaction was never built).
  - [deferred/nociception_layer.md](deferred/nociception_layer.md) is REVIVED into it (G5).
  - GL2a is record-only and outside the 1.3.2 fence (an exact exempt file set). GL2c fires T1-11, T1-12
    and T1-13/14/15 under any routing (owner decision G7): it ships off by default and lands only with a
    batched live re-run of Exp 60/61/62 (+ the T1-11 and T1-12 arguments or re-runs); its joint review with Phase 5's relief store is an owner
    decision at GL2c's start.
- [thalamic_relay.md](thalamic_relay.md) — **PROPOSED 2026-10-07 (GL3); GL3.B0 BUILT 2026-10-09 and merged (#1201: tests only, the receptor census, the strict-xfail timing red gates and a frozen latency record). GL3.B1 onward is not built; `runtime/preemption.py`'s `PreemptionCircuit` is pre-GL3 code, Dormant since 2026-10-07 (TR1).** One registration
  contract for percept sources (`Receptor`) and the afferent pathways they emit onto (`AfferentTrack`).
  - One physical event fans out under a shared, deterministic `PhysicalEventId` (agent, seq; since
    G15, 2026-10-09, agent, session id, seq). Since G17 (2026-10-09) GL2a mints no id: the type, the
    per-agent sequencer, its session-id source and the cross-session resume land together at a
    post-fence resume stage, before GL4 S1.
  - Tracks are logical channels on the loop clock, never OS threads.
  - The registry enters with its consumer (owner decision G8): it lands in one stage with provenance
    on the write path (GL3.B1), registering the existing receptors byte-identically. The first track
    slice (GL3.B3) is the thermal dual-track fan-out, flag off, no preemption; nociceptive-fast
    preemption (GL3.B4) is a declared 1.4 rung arm or waits for one. The timing defects
    are filed as defect issues (#1176 L1, #1177 L2, #1178 L5, #1179 L7), fixable without the registry.
  - GL3.B0 (census + red gates, tests only) runs now; GL3.B1 onward waits for the decomposition
    fence, and the plan must answer `archive/percept_testbed_audit.md`'s "not a
    manifest as conceived".
- [latent_forward_model.md](latent_forward_model.md) — **PROPOSED 2026-10-07, plan only (GL4); the name
  roadmap 1.4 Phase 5 reserved.** An action-conditioned predictor of the autonomic consequence from an
  affordance or percept context, seeded from the Cerebellum's training signal.
  - Its S0a is Phase 5's mandated Cerebellum + `anticipatory_pre_activate` audit.
  - It inherits the JEPA plan's four rules, and is not called JEPA while its target is the fixed
    autonomic code.
  - The word embedding is the innate-prior tier; experience is the learned tier.
- [../wiring/body-and-word-worlds.md](../wiring/body-and-word-worlds.md) — **the state page (GL0).**
  The two chains with file::symbol, the live/dormant/dead status of every would-be bridge between them,
  and the names-vs-consequences collisions on shipped affordances.

### The R4 credit map

The decided plans this map listed are SHIPPED and listed once, under §Shipped plans kept at their path below (2026-10-18):
[lookback_primitive.md](lookback_primitive.md) (no new store; credit stays on `NAc._eligibility`),
[memory_2d2_retroactive_tagging.md](memory_2d2_retroactive_tagging.md),
[memory_2s_d_situation_cue.md](memory_2s_d_situation_cue.md) and
[oasis_entry_index_v2.md](oasis_entry_index_v2.md). The credit-routing map,
[deferred/three_factor_credit_assignment.md](deferred/three_factor_credit_assignment.md), is deferred until
the R4 credit build starts (2026-10-18; see §Deferred).

*The shared-perception designs live in [deferred/](deferred/). The reflex runtime and the microduck
are deferred on a second physical body arriving; the fabric was re-keyed 2026-10-07 (owner decision
G4) to revive when GL3's registry+provenance stage ships AND a 1.4 rung needs cross-modal binding.*

### Revived and homed out of `deferred/` (2026-10-18)

Moved to the root where nothing cites the old path. The rest stay under `deferred/` because `src/`, `scripts/` or
`.github/` cite their path; each banner says so.

**Homed (owner decisions 2026-10-17):**
- [hf_hive_repository.md](hf_hive_repository.md) — Hugging Face Hive Repository. Phase 1 (read-only distribution of Queen-signed release bundles + the replay Space): its trigger **fired 2026-09-09** (1.2.0, #670; its prerequisite, the format freeze, #918), and it was **REVIVED 2026-10-17** — the owner is setting up the dataset repo, a launch companion outside any release. Phase 2 (public submissions + gauntlet + main stream) stays deferred until the Slice-D promotion prerequisites land AND "validated bundle" has a defensible meaning AND decision point 4 is deliberately re-opened. Moved to the root 2026-10-18.
- [loop_stack_builder.md](loop_stack_builder.md) — one shared `build_loop_stack()` for the hand-assembled `run_agentic_loop` stacks. Its trigger (the next hand-assembled stack) **fired 2026-09-04** (`simulation/minecraft_harness.py`, #610); **homed in 1.3.6**. Moved to the root 2026-10-18.
- [node_security_simplification.md](node_security_simplification.md) — Phase 1 shipped; Phase 2's role-detector unification shipped in #318; its trigger fired with 0.5.0. The remainder (the legacy `"client"` role and `role_divergence` shims marked "Drops in 1.2", and an auth-model doc) is **homed in 1.3.3, Session H**. Moved to the root 2026-10-18.
- [deferred/console_tunnel_hardening.md](deferred/console_tunnel_hardening.md) — **ACTIVE (homed) 2026-10-18; stays under `deferred/` because src cites it.** The console-facade security ladder: PR 1 trust guard (#609) and PR 2 bearer auth (#613, 1.1.4) merged; **PR 3 admission control homed in 1.3.3**; PR 4 pre-GA authz tiers. Its trigger (the console/UI line scheduled into a release) fired: the Console line shipped in 1.3.2 (contract 0.6.0, #1268/#1279/#1293).
- [deferred/measurement_path_fail_loud.md](deferred/measurement_path_fail_loud.md) — **ACTIVE (homed) 2026-10-18; stays under `deferred/` because src and scripts cite it.** Stages 1 + 2 + 4 SHIPPED (1: #487; 2: 2026-08-30, zero firings; 4: the CI lock). Stage 3 (narrowing) was scheduled into 1.3.2 and did not land there; **homed in 1.3.3, Session H**.
- [deferred/stall_detector_timeout_awareness.md](deferred/stall_detector_timeout_awareness.md) — **ACTIVE (homed) 2026-10-18; stays under `deferred/` because src cites it.** Stage 1 SHIPPED (PR #324); Stage 2, the heartbeat migration, SHIPPED 2026-10-14 (PR #1268). The homed remainder is the stall follow-ups #1272 and #1043, **in 1.3.5**. Stage 3 is deferred in place: blocked on [deferred/llm_timeout_scalability.md](deferred/llm_timeout_scalability.md) Stage 4.

**Revived 2026-10-18 (owner decision); each homing is proposed, owner to confirm:**
- [deferred/maxim_console.md](deferred/maxim_console.md) + [maxim_ui.md](maxim_ui.md) + [deferred/reachy_app_maxim_seams.md](deferred/reachy_app_maxim_seams.md) + [reachy_dm_app.md](reachy_dm_app.md) — the UI line. Phase 0 / MVP seams shipped earlier (`maxim serve` #416, seams #419–#428, `src/maxim/console/`), and the Console shipped in 1.3.2 (contract 0.6.0: run history, health, stall detection). `maxim_ui.md` and `reachy_dm_app.md` moved to the root; the other two stay under `deferred/` because src and scripts cite them. FIT on the Pi is still open. Proposed homing: alongside Session S (the pulse seam) / post-1.3.3 console work.
- [config_extensibility.md](config_extensibility.md) — which config restrictions are the typo check (keep) and which merely require a fork (relax); built-ins ∪ a registry plus an `extensions.*` namespace. Its trigger, the [#856](https://github.com/dennys246/Maxim/issues/856) format-version decision, fired when #856 closed 2026-09-29. Proposed homing: after 1.3.3's single config-format bump (`CONFIG_FORMAT_VERSION` 1.3 → 1.4).
- [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) — multi-peer load distribution; stays under `deferred/` because src cites it. Its 2+-GPU-node trigger is met: the multi-GPU rig exists. Proposed homing: a post-1.4 LLM line.

### Shipped plans kept at their path, and frozen contracts (listed 2026-10-16)

These plans' work has shipped, but `src/`, `scripts/` or `tests/` cite them by path, so they stay put rather than
break the citation. Each is a record now, not a plan.
- [memory_2d2_retroactive_tagging.md](memory_2d2_retroactive_tagging.md): DECIDED + BUILT 2026-09-25 (#897).
- [memory_2s_d_situation_cue.md](memory_2s_d_situation_cue.md): DECIDED + BUILT (#898); 2S-e is parked in `deferred/`.
- [m1b_ledger_evidence_gate.md](m1b_ledger_evidence_gate.md): every PR built (3, 4, 5a–5a-3, 5b-1, 5b-2); the gate is `lint_evidence_gate.py` and `lint_ledger_format.py` (M1).
- [baseline_floors_b0.md](baseline_floors_b0.md), [baseline_floors_b1_build.md](baseline_floors_b1_build.md), [archive/baseline_floors_b1_evidence.md](archive/baseline_floors_b1_evidence.md): B0 merged (#1233); B1 built and run (2026-10-14). The live plan is [baseline_floors.md](baseline_floors.md).
- [oasis_entry_index_v2.md](oasis_entry_index_v2.md): PRs A–C shipped 2026-09-25; the format is frozen under public_format_freeze.
- [lookback_primitive.md](lookback_primitive.md): DECIDED 2026-09-24 (no new store; #891/#892); R4's look-back design record.
- [grounded_language_acquisition.md](grounded_language_acquisition.md): superseded by [grounding.md](grounding.md) 2026-10-07 (owner decision G5); kept at its path because `src/`, the CLI help text and `analysis/roy_log.py` cite it. Its paired-data audit entry condition is answered there (the pair became (context, consequence)).
- [minecraft_benchmark.md](minecraft_benchmark.md): Part I COMPLETE (Exp 56); Part II's R0–R3 shipped (R3 an instrument + frozen baseline in 1.3.0), and R4 moved to `roadmap_1_4.md` Phase 5. The ladder's own record; `src/maxim/simulation/minecraft.py` and the R1/R2 probe scripts cite it.

Frozen contracts (reference, not plans):
- [public_format_freeze.md](public_format_freeze.md): FROZEN 2026-09-26; guarded by `tests/unit/test_public_format_freeze.py`.
- [sharing_threat_model.md](sharing_threat_model.md): FROZEN 2026-09-04 (1.2 gate 4).

Approach notes for 1.3.3 and 1.3.4:
- [advisor_mode.md](advisor_mode.md): **1.3.4, the atonia gate** — Maxim observes and reports while its motor output is inhibited; the gate guards tool dispatch, and the sandbox alone guards disk and egress until item 7 and an egress seam ship (#1315; O1–O9 verified against v1.3.2; owner decisions AD1–AD9 decided 2026-10-17).
- [readiness_and_extras_standards.md](readiness_and_extras_standards.md): four standards R1–R4 from the pulse seam (readiness proves the capability, extras tested against the real library, typed console payloads, a checked vendoring step; #1332–#1336; 1.3.3).
- [upgrade_safety.md](upgrade_safety.md): `maxim doctor`'s advisory check and an upgrade manifest (#1295, approach note merged as #1297).

### Living discipline docs

- [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md) —
  **Graduation ledger + post-1.0 regression discipline.** Earned rows carry re-run
  triggers; `Stale`/`Broken` rows block the next release. (Owes an Exp 54 row when Phase
  B/C record.)
- [bio_faithful_roadmap.md](bio_faithful_roadmap.md) — per-system bio-fidelity status
  and trajectory; the algorithmic companion to the ledger (#493).
- [simulation_apparatus_standards.md](simulation_apparatus_standards.md) — S1–S8, the
  standards every pre-registration declares against (S4 shipped; the S2 canaries are item 2 of the archived 1.1.x list, [archive/roadmap_1_1_to_1_3.md](archive/roadmap_1_1_to_1_3.md)).
- [glossary.md](glossary.md) — decoder for the coded IDs. (Owes 1.1's one admitted
  mechanism — relief-sourced operant credit — and Exp 52/53/54 entries.)
- [../limits/score_cards/](../limits/score_cards/) — repository scorecards.

## Deferred (revive on trigger)


Design work preserved in [deferred/](deferred/). Each has an explicit "revive when" condition (full rationale in the banner at the top of each doc).

**Deferred in place (kept at their root path because src/ cites it; 2026-10-17):** not on the path to 1.4.0 (owner); each file carries the banner at its top.

- [executive_function.md](executive_function.md) — staged executive function from #841's deep dive ([#1073](https://github.com/dennys246/Maxim/issues/1073)); Stage 1 is a gated goal slot riding NAc. **Revive when:** R4 credit routing or E2's relief-store review names a goal-slot consumer (homed after 1.4.0, #1072–#1075).
- [fear_learning.md](fear_learning.md) — relief as a reinforcer and extinction, from #840's deep dive ([#1072](https://github.com/dennys246/Maxim/issues/1072)); two Exp 60-classroom experiment sketches. **Revive when:** the Phase 5 relief-store review adopts Experiment A as its consumer (homed after 1.4.0).

- [deferred/behavior_tiers.md](deferred/behavior_tiers.md) — which automatic behaviours stay code (invariants, innate priors) and which should become learned. The declare-your-tier **rule** is active; the migrations are deferred per item: source trust (M1) revives with the language line's L4 prereg, curiosity (M3) with intrinsic motivation's trigger or L5, learned reflex gain (M4) when a rung names a fixed gain as the limit, adaptive pain (M6) on its own plan's trigger. All wait for memory-strength Phase 2.
- [deferred/adaptive_nociception.md](deferred/adaptive_nociception.md) — pain habituation (harmless, repeated) and sensitisation (damaging, repeated), adapted **upstream in the pain producer**, never inside a consumer. Deferred 2026-09-21 out of the memory line's scope. **Revive when:** a world produces chronic pain that does no damage, or repeated pain measurably over-tags.
- [deferred/reflex_layering.md](deferred/reflex_layering.md) — the narrative reflex path collapses stimulus, felt pain and response into one number (habituation/sensitization/pre-emption scale the DAMAGE inflicted); routes 2 (response habituation: `context_key`, recovery) and 3 (re-layering; bracing as a real anticipatory act). Deferred 2026-09-24; route 1 (firing outcomes) shipped. **Revive when:** anticipation is to be measured in the narrative world, adaptive nociception revives, a non-heartbeat row 9 re-run is scheduled, or a consumer starts acting on `ReflexFiring`; since 2026-10-07 also when the grounding line's GL3 two-track thermal receptor opens (routes 2/3 are its fast/slow split). Pairs with adaptive nociception.
- [deferred/nociception_layer.md](deferred/nociception_layer.md) — pain says what it is (`PainKind` on the type; step 1 in memory Phase 2S-c) and each consumer's rule is written in that vocabulary; ten consumers, several rules, and the right rule differs by consumer (memory excludes air hunger, fear needs it). Records F1 (anticipated pain paid out as negative reward — live on `--sim`, near-inert until R4), F1b, F2 (three NAc writers), F3 (lossy ReactionBus→PainBus rebuild). Four-lens reviewed. **Revive when:** adaptive nociception or reflex-layering route 3 revives, a post-2026-09-24 consumer re-derives meaning, **an R4 credit-routing PR opens**, the strength tag reads pain in a validation run, or F1/F2 is measured. Deferred 2026-09-24. **REVIVED 2026-10-07 into [autonomic_layer.md](autonomic_layer.md) (grounding GL2)**; kept as the record. Step 3 (F1) still lands before or with any R4 routing PR and before GL2c.
- [deferred/retro_tagging_extensions.md](deferred/retro_tagging_extensions.md) — the four limits memory-strength 2d-2 ships with (no tagging in the LLM sims, ~5 s reach, eviction before sleep, `--sim`'s lightweight end), each with what would lift it and its revive trigger.
- [deferred/situation_cue_fallback.md](deferred/situation_cue_fallback.md) — the 2S-d situation cue's lossy links (compressed / evicted / re-created concepts) and the llm-primary + hardware paths it does not cover, each with its revive trigger.
- [deferred/generalization_by_pattern_completion.md](deferred/generalization_by_pattern_completion.md) — memory 2S-e (B), parked 2026-09-25: its measured gap (the Exp 62 "night miss") was the `time_of_day` wrap (#899), a keying defect; revive on a measured generalization gap keying does not own.

**Deferred 2026-10-18:**

- [deferred/three_factor_credit_assignment.md](deferred/three_factor_credit_assignment.md) — the credit-routing map `roadmap_1_4.md` Phase 5 names (the shipped eligibility trace and the Cerebellum), plus one proposed learnable (calibration). Design note, zero code; moved from the root. **Revive when:** the R4 credit build starts. Owner decisions A (the thesis wording) and B (learnable or hand-specified tuning curves) remain open.
- [deferred/vllm_self_hosted_servers.md](deferred/vllm_self_hosted_servers.md) — vLLM (or any self-hosted OpenAI-compatible server) as a first-class lane: no new mechanism, fixes on the existing self-hosted peer path. Drafted 2026-09-27. **Revive when:** any OpenAI-compatible self-hosted server is put behind a lane, or a vLLM host is stood up. Its server-agnostic findings are filed: [#1369](https://github.com/dennys246/Maxim/issues/1369) (D5), [#1370](https://github.com/dennys246/Maxim/issues/1370) (D6), [#1371](https://github.com/dennys246/Maxim/issues/1371) (the stale env key).

**Deferred 2026-09-18 (owner):**

- **Shared perception** — [deferred/second_body_staging.md](deferred/second_body_staging.md) + the fabric, microduck and reflex-runtime plans (indexed below) + breeding — **DEFERRED 2026-09-18 (owner).** (Exp 55 left this deferral 2026-09-27 for [coding_world.md](coding_world.md): a coding body needs no hardware.) Revive when: a second body physically exists (a real backend registered through `maxim.robots`, or the operator records its arrival). Stage A (the baseline measurement) runs unchanged on revival. The files are in `deferred/`. **Split 2026-10-07 (owner decision G4):** the physical trigger now holds for the robot half only (the hardware factory in second_body_staging Stage B, the microduck and the reflex runtime). The perception fabric re-keys (it revives when GL3's registry+provenance stage ships AND a 1.4 rung needs cross-modal binding), because Minecraft already satisfies the perception abstraction.

**Moved down from the root in the 2026-09-19 plans audit** (each file now carries its revive trigger at the top):

- [cradle_mother.md](deferred/cradle_mother.md) — **Revive when:** a second robot body physically exists. The nursery vehicle: the
  mother-scaffolded feeding scenario (`simulation/cradle_mother.py`, the `cradle_mother`
  arc, `benchmark_cradle_mother.py` / `analyze_cradle_mother.py --gate v3`) that Exp 52
  GRADUATED on (relief-sourced credit, apparatus v3) and Exp 54 now runs on the
  Reachy-native body (Phase A GRADUATE 2026-08-27; Phase B/C on the robot next).
- [substrate_native_orienting.md](deferred/substrate_native_orienting.md) — **Revive when:** a second robot body physically exists. The umbrella
  for the whole orient line (Exp 45 → 52 → 53b → 54). Layer 1 = the orient policy
  (hardware-validated, nursery-taught, read out on the robot); Layer 2 = spatial
  co-location as the grounding substrate (gated north star).
- [reachy_orient_live.md](deferred/reachy_orient_live.md) — **Revive when:** a second robot body physically exists. **Live-rig runbook** (Steps 1–3
  PASSED 2026-07-15/16). The rig, preflights and probes that Exp 53/53b and Exp 54
  Phase B/C run on.
- [sem_motor_binding.md](deferred/sem_motor_binding.md) — **Revive when:** a second robot body physically exists. Phases 1–2 SHIPPED (#460/#461), Arm 1
  = Exp 49. Produces `ReachyOrientMotorBackend`, the motor path the readouts go through;
  open: Arm 2 + the Stage-5 pending map (a 1.3 reflex-tier prerequisite).
- [live_audio_orient_wiring.md](deferred/live_audio_orient_wiring.md) — **Revive when:** a second robot body physically exists. The live DoA feed
  (Stages 0b–4b SHIPPED, #447). **Roadmap item 18 → 1.1.1** rides its Stage-2 surface
  (loudness/onset salience from the daemon's REST-served level registers); Stage 5 is the
  1.3 reflex tier.
- [perception_pipeline_placement.md](deferred/perception_pipeline_placement.md) — PARTIALLY
  landed (#382–#385); `runtime/perception_placement.py` is a Dormant type layer whose
  `src` docstring still names "the 1.3 fabric actually placing stages" as its resurrection trigger
  (GL0 re-points it). **Revive trigger re-keyed 2026-10-07 (owner decision G4):** GL3's
  registry+provenance stage ships AND a stage is placed across a wire
  ([thalamic_relay.md](thalamic_relay.md)), a capability trigger rather than a body count.
- [decision_provenance.md](deferred/decision_provenance.md) — **Revive when:** a 1.4 rung names the gap it fills. Stages 1+2 SHIPPED (#504:
  `learned_margin` / `explore_decisive` — the instrument that explained Exp 52's weak
  seed and gated Exp 53's readout). **Stages 3+4 = item 2.**
- [annotation_context_and_provenance.md](deferred/annotation_context_and_provenance.md) — **Revive when:** a 1.4 rung names the gap it fills. S1
  SHIPPED (#501/#503), S4 DONE (1.1 cut item 11). **S2 + Exp 51 + S3 = item 3.**
- [modality_resolution_and_alignment.md](deferred/modality_resolution_and_alignment.md) — the
  discrimination facts (~3 azimuth bins at the production threshold — measured on live
  hardware by Exp 53/53b) and the place-code path. **§7 place-code default-ON = item 5.** **Revive
  trigger re-keyed 2026-10-07 (owner decision G4):** GL3's registry+provenance stage ships
  ([thalamic_relay.md](thalamic_relay.md)).
- [coding_habits_oasis.md](deferred/coding_habits_oasis.md) — **REVIVED 2026-09-27 into [coding_world.md](coding_world.md); kept as the record of the 2026-09-05 design.** ADOPTED 2026-09-05: collective
  coding-safety habits. The coding gauntlet (#3) is a Queen-tier promotion INSTRUMENT
  inside the 1.2 poison-resistance work — no 1.2 claim; the habit-transfer claim
  (Exp 55) is 1.3-line. Defines the inherent bias class (Queen-promoted, decay-exempt,
  tighten-only) and the pain-credit extension decision.
- [cross_modal_perception_fabric.md](deferred/cross_modal_perception_fabric.md) — the 1.3
  perception fabric (rev 4; Stage 0a complete via H1, 0b/0c are the gate). **Revive trigger re-keyed
  2026-10-07 (owner decision G4):** GL3's registry+provenance stage ships AND a 1.4 rung needs
  cross-modal binding ([thalamic_relay.md](thalamic_relay.md)). Until then it is an input to that plan,
  not a plan of record.
- [hybrid_substrate_reflex_runtime.md](deferred/hybrid_substrate_reflex_runtime.md) — the 1.3
  reflex tier (DN-canonical orienting reflex; sole owner of BL-1..BL-5). Physical-robot trigger
  unchanged 2026-10-07 (G4); its reflex tier is not the grounding line's GL3 `reflex` track.
- [microduck_intent_layer.md](deferred/microduck_intent_layer.md) — **Revive when:** a second robot body physically exists. The microduck (rev 2,
  2026-08-31, two-lens reviewed): the operator's design constraints worked through against
  the code. Zero code, no schedule change, no experiment number. Records what the
  constraints ride versus what is new, two inputs to 1.1.3's gate-7 design pass, and the
  finding that `last_clamped_axes` reaches no learner. Its rev 1 recommendations were
  withdrawn in review — read §2.4 and §5.2 for the arguments, not the index.
- [reactive_peer_mesh_roadmap.md](deferred/reactive_peer_mesh_roadmap.md) — **Status 2026-10-17: the old trigger (the console/UI line scheduled into a release) fired with 1.3.2 and was not pursued (owner). Revive when:** a second peer-mesh consumer (beyond the console) is scheduled into a release. Living mesh
  roadmap/index over the deferred mesh plans. §5 is stale: C10's 1.0 prep SHIPPED (#329)
  and its ship slice is 1.2 (it waits on Hivemind), not near-term.
- [intrinsic_motivation_1_3.md](deferred/intrinsic_motivation_1_3.md) — curiosity and competence as
  learned reward (1.3 draft, 2026-09-08); shares the mining world, not the survival claim. **Revive when:** a 1.4 rung names the gap it fills.
- [setpoint_aware_neutral.md](deferred/setpoint_aware_neutral.md) — the set-point-aware neutral for the
  L11 direction problem; the 2026-09-16 four-lens review returned DO-NOT-BUILD as proposed. **Revive when:** a 1.4 rung names the gap it fills. Its own
  header narrows that: a want that structurally needs a small same-side move of one graded sensor to separate,
  and then only as the adaptive (salience-layer) design.


Deferred in the 2026-08-27 post-1.1.0 triage (banner + revive trigger at the top of each doc):
- [deferred/orient_magnitude_learning.md](deferred/orient_magnitude_learning.md) — S0/S1 shipped; S4 completed via Exp 45e. Only S3 open. Revive on a pre-registered held-out-bin generalization claim.
- [deferred/productive_orienting_affordance.md](deferred/productive_orienting_affordance.md) — Phases 1–2 SHIPPED (#403); the 2-axis decision waits for a second-axis body.
- [deferred/substrate_learns_from_experience.md](deferred/substrate_learns_from_experience.md) — Phase 1 SHIPPED (#437). Phases 2–3 revive when imagination may write to the substrate.
- [deferred/passive_sense_discovery.md](deferred/passive_sense_discovery.md) — Phase 0 never authorized; zero code. Revive when an experiment is blocked by the cost of `sense_tools`.
- [deferred/sem_environmental_proximity_sensing.md](deferred/sem_environmental_proximity_sensing.md) — zero code; revive when a drive must be relieved by sensed ambient environment.
- [deferred/tool_refinement_plan.md](deferred/tool_refinement_plan.md) — stalled living doc (missed the whole 2026-07/08 tool wave). Revive at the next tool-surface curation wave.

Deferred 2026-09-24 — **on a frozen condition:**

- [deferred/signed_signer_identity.md](deferred/signed_signer_identity.md) — STEP 1 LANDED 2026-09-25 (release format v2, item 7 PR A): bind `signer_identity` into the bundle signature. Not exploitable today (verification uses the claimed identity's key; one-key-two-identities refused). Scheme v2 lands with public_oasis item 7's format change (v1 accepted with a warning); v1 dropped at the next major. Also revive if any path trusts the label unverified.
- [deferred/transfer_non_situation_nac_rows.md](deferred/transfer_non_situation_nac_rows.md) — DEFERRED 2026-09-25: make a donor's percept valences / outcome stats / node-keyed reward bias transfer (today dropped at ingest — never readable). A behaviour change past what Exp 56/61 earned. Trigger: a design needs a transferred valence or outcome statistic (e.g. social_referencing).
- [deferred/grounded_word_binding.md](deferred/grounded_word_binding.md) — DEFERRED 2026-09-24 on a condition (renamed from `grounded_word_binding_demo.md`): a heard word calls back its situation (substrate-primary). Its L0 offline gate — blind-authored phrasings of five situations cluster by situation through Maxim's own EC formation (criterion frozen in its header) — was its revive condition (pass → candidate 1.5 headline, fail → archive). Even a pass licenses only "labels and their paraphrases bind". **2026-10-07 (owner decision G5):** L0 runs unchanged as the grounding line's GL1 innate-prior measurement, and the "pass → candidate 1.5 headline" consequence is **RETIRED**. A pass no longer revives this plan on its own; it informs [latent_forward_model.md](latent_forward_model.md)'s word head. **Revive when:** the grounding line's word-world (innate-prior) tier opens. Whether a FAIL still archives the plan is an owner decision at GL1's start.

Deferred 2026-07-22 to 2026-08-11 — **indexed 2026-10-17** (missing from this index until then):
- [deferred/credit_on_progress_not_execution.md](deferred/credit_on_progress_not_execution.md) — DOWNGRADED 2026-07-22: credit a tool for drive or goal progress, not for executing. A probe disproved it as the embodied root cause, so it is an optional cleanliness change. States no revive trigger (audit flag).
- [deferred/long_lived_server_config_guards.md](deferred/long_lived_server_config_guards.md) — 2026-07-26: once-per-process config guards (`_lane_env_applied`, the role export) go stale inside a long-lived `maxim serve`. Revive when the console wizard needs run → re-setup → run to re-route lanes in one `maxim serve` process, or Phase 3 talk/rest modes ship.
- [deferred/scn_event_producer_gap.md](deferred/scn_event_producer_gap.md) — 2026-07-29 (updated 2026-08-29, #572): five of the SCN's six declared temporal-event categories have no producer. Revive when the oscillator earns behavioural weight and a second category is needed, or a drive-conditioned anticipatory behaviour is on the critical path.
- [deferred/retrosplenial_spatial_frames.md](deferred/retrosplenial_spatial_frames.md) — 2026-08-11: egocentric ↔ allocentric frame translation, plus the re-validation registry every EC-clustering change must walk (§5). Revive when any §4 trigger fires: T1 place memory across motion, T2 multi-vantage identity, T3 return-to-place (`roadmap_1_4.md`'s E2 question), T4 a second frame-confusion incident.

Deferred in the 2026-07-15 audit — **partially shipped, remainder paused:**
- [deferred/llm_timeout_scalability.md](deferred/llm_timeout_scalability.md) — Stages 1–3.5 SHIPPED (per-tier timeout, admission gate, TTFT keepalive; PRs #320/#321). Stage 4 adaptive throughput model unbuilt. Revive when a big-model soak yields the `(TTFT, tok/sec)` data it needs.
- [deferred/mesh_perception_transport.md](deferred/mesh_perception_transport.md) — 1.0 prep SHIPPED (PR #329); single-cut-point framing superseded by `perception_pipeline_placement.md`. Revive when a perception stage is placed across the wire (1.2+ frame transport).
- [deferred/bio_emergent_persona_foundations.md](deferred/bio_emergent_persona_foundations.md) — Stages 0–3 + Wire-A SHIPPED via 0.9.1. Only Wires 4/5 remain; evidence points at encoder alignment, not these wires, as the persona blocker. Revive if a Roy iteration shows Wire 4 or 5 is load-bearing.
- [deferred/scene_actor_affordances.md](deferred/scene_actor_affordances.md) — Stages 1+2 SHIPPED (PR #213). Stages 3–5 (prompt rule, designer hint, validation experiment) revive when an adversarial sim needs narrated entity actions to hit real AUT body mechanics, or when re-evaluating agent_backed_entities.
- [deferred/sense_tool_registry.md](deferred/sense_tool_registry.md) — 1.0 MVP SHIPPED (PR #287: `auto_fire`, `kind`, grayscale visibility). 1.1+ full plan revives if substrate→action tool visibility is re-prioritized or predictive learning needs `sensory_events.jsonl` / predicate-outcome typing.
- [deferred/imagination_substrate_signals.md](deferred/imagination_substrate_signals.md) — Hookup 1 MVP SHIPPED (PRs #288/#292). Hookups 2+3 revive after encoder replacement, if per-tick substrate-driven imagination is the next lever.

Deferred in the 2026-07-15 audit — **never started / blocked:**
- [deferred/mcp_compatibility.md](deferred/mcp_compatibility.md) — Zero MCP code; CC9 prerequisite shipped. Revive when the MCP spec stabilizes AND a concrete user need appears.
- [deferred/key_drift_detection.md](deferred/key_drift_detection.md) — Never implemented; the reactive 401-hint fold reduced urgency. Revive if stale-key confusion recurs outside the 401 path.
- [deferred/transition_based_drive_pain.md](deferred/transition_based_drive_pain.md) — Root-cause fix for per-tick drive-pain re-firing. **Phases 0, 1 and 4 SHIPPED 2026-07-28** (its trigger fired 2026-07-17; the breach latch is `Embodiment._drive_breach`). Only Phase 3, the B8 disposition, remains; owner decision 2026-09-19: settle it inside 1.4's credit-routing audit, by measurement if reading the code cannot.
- [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) — **SUBSUMED 2026-10-07 by [latent_forward_model.md](latent_forward_model.md)**: the owner reversed roadmap 1.4 decision 3, its four rules are inherited there, and the projection itself is not revived. Before that it was a 1.2+ research direction; motivating 384-vs-768-dim split still true, but Exp 35/36 resolved via threshold tuning. Revive on a structurally cross-modal problem unsolvable by threshold tuning + a passing Stage 0 paired-data audit.
- [deferred/scn_decay_anchoring.md](deferred/scn_decay_anchoring.md) — **SUPERSEDED 2026-10-17 by [experience_clock_decay.md](experience_clock_decay.md)** (experience clock, not SCN). Hardware portability of decay timescales; unstarted. Revive when a second hardware baseline joins benchmarking or decay calibration is greenlit (hard prereq of the next entry).
- [deferred/decay_consolidation_calibration_plan.md](deferred/decay_consolidation_calibration_plan.md) — Calibration-by-simulation framework; blocked on scn_decay_anchoring. Revive when decay timescales are confirmed as the bottleneck or a new tau consumer mis-inherits a default.

Deferred in the 2026-07-15 audit — **stalled practice docs** (were "living," stopped accumulating):
- [deferred/persona_convergence_crucible.md](deferred/persona_convergence_crucible.md) — 8 Roy iterations of real data (May 2026); halted on the encoder-alignment gap its own runs root-caused (768-vs-384-dim). Load-bearing negative evidence — do NOT archive. Revive when encoder replacement / cross-modal alignment ships; Roy-1 Adversarial then becomes runnable.
- [deferred/behavioral_convergence_practice.md](deferred/behavioral_convergence_practice.md) — Exp 1–5 (41/41) logged in April; the validation energy migrated to [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md). Revive on a new system-level "does the agent get better" experiment.

Earlier deferrals:
- [deferred/b5_embodiment_narrative_separation.md](deferred/b5_embodiment_narrative_separation.md) — Formalize SEM/DM prompt boundary. Revive when prompt-bleed bug surfaces.
- [deferred/agent_backed_entities.md](deferred/agent_backed_entities.md) — 3-tier cognition + Cradle-trained cast + mesh-pressure budget. Revive if scene_actor_affordances diagnostic doesn't close the gap.
- [deferred/bio_system_plugin_plan.md](deferred/bio_system_plugin_plan.md) — Plugin discovery for bio-systems. Revive when external contributors appear.
- [deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md) — Two-agent stimulus pattern. Revive when behavioral convergence hits variety ceiling.
- [deferred/pecking_order_graph_plan.md](deferred/pecking_order_graph_plan.md) — Unified hierarchy DAG. Revive when multi-node topology matters.
- [deferred/dungeon_master_extensions.md](deferred/dungeon_master_extensions.md) — DM post-MVP features (a maybe-list). Revive when real DM campaign usage points at one specific extension.
- [deferred/pain_bus_bridge_subscriber_unification.md](deferred/pain_bus_bridge_subscriber_unification.md) — Bridge/subscriber attribution-asymmetry fix. Monitor; open when pending-event context enriched.
- [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md) — Capability-aware routing. Revive when heterogeneous mesh.
- [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md) — Async router. Revive if head-of-line blocking observed.
- [deferred/llm_path_fair_scheduling.md](deferred/llm_path_fair_scheduling.md) — Fair-share scheduling. Revive if rate limiting insufficient.

## Archive

Completed or superseded plans live in [archive/](archive/).

Archived 2026-10-24 (the plans archive after 1.3.3; each file carries an ARCHIVED banner):
- [archive/download_consent_gate.md](archive/download_consent_gate.md): #1343 shipped (PR #1400). The rule lives in docs/agents/llm-routing.md.
- [archive/internet_toggle_session_gate.md](archive/internet_toggle_session_gate.md): #1345 shipped (PR #1401). The rule lives in docs/agents/runtime-tools.md.
- [archive/release_pipeline.md](archive/release_pipeline.md): 1.3.3 published through `release.yml`; its hand-off was applied to docs/publication_guide.md.
- [archive/r4_console_vendor_gate.md](archive/r4_console_vendor_gate.md): built by PR #1383. The rule lives in docs/agents/ci-testing.md, and the cut rule in readiness_and_extras_standards.md §R4.
- [archive/baseline_floors_b1_evidence.md](archive/baseline_floors_b1_evidence.md): built in #1245. The BF2 obligation lives in docs/agents/simulation-experiments.md.

Marked SHIPPED in place (2026-10-24), because code or its docs cite them by path: memory_2d2_retroactive_tagging.md,
memory_2s_d_situation_cue.md, oasis_entry_index_v2.md, lookback_primitive.md, baseline_floors_b1_build.md.

Redirects for plans that left `deferred/` (moved out of the Deferred index 2026-10-17):
- ~~deferred/mother_maxim_plan.md~~ → **SUPERSEDED** (2026-05-09) by [maxim_hivemind.md](maxim_hivemind.md). Old plan moved to [archive/mother_maxim_plan.md](archive/mother_maxim_plan.md). Reframed for the substrate-primary world: peer-to-peer Maxim Oases (sustaining gathering places) instead of central Mother server; Maxim Hivemind (collective cognition layer) instead of Pecking Order Graph; distilled bio-substrate snapshots instead of raw episodes. ~2,100 LOC instead of 3,800. Phasing: shareability infrastructure SHIPPED in 1.0 (B5, PRs #305–#311); Oasis + full Hivemind protocol gated for 1.2.
- ~~deferred/grounded_language_acquisition.md~~ → **PROMOTED** to [grounded_language_acquisition.md](grounded_language_acquisition.md) (2026-05-09). **Phase -1 GATE CLEARED + Phase 0 harness + EC sensor-encoding ALL SHIPPED** (PR #228 + branch `feat/phase0-sensor-encoding`, 2026-05-09): substrate produces non-LLM actions; `cradle_prelinguistic` arc + motor-only AUT prompt + per-tick telemetry; `SensorEncoder` hashes drive snapshots through EC with modality `"interoception"`. Hivemind shareability SHIPPED (PRs #305–#311, 2026-05-31). Full Phase 0 validation + Phase 1 + Phase 2 in 1.1+ (1.1-T7). Trigger: 2026-05-09 audit found 60-70% of recent engineering effort going to LLM-mitigation scaffolding (~845 LOC of band-aids) rather than substrate work; the parallel-mode pivot is the structural fix. D&D campaign survival is the bidirectional kill criterion (substrate failure OR simulation-environment failure). The existing LLM-AUT path stays as the user-facing default.
- ~~deferred/colibri_worldgen_smoke.md~~ → **ARCHIVED** 2026-10-17, at [archive/colibri_worldgen_smoke.md](archive/colibri_worldgen_smoke.md) (listed below).

- [archive/colibri_worldgen_smoke.md](archive/colibri_worldgen_smoke.md) — **ARCHIVED 2026-10-17** (owner): the colibrì GLM-5.2 world-generation smoke. Its weights were deleted, and the July run measured 0.05–1 tok/s, too slow to beat the cloud arm under its pre-registered rule. Reviving it needs a new speed argument.
- [archive/exp44_overnight_runcard.md](archive/exp44_overnight_runcard.md) + [deferred/acting_coach_body_state_ablation.md](deferred/acting_coach_body_state_ablation.md) — **ARCHIVED 2026-10-18** (the runcard moved to `archive/`; the plan is archived in place under `deferred/` because src cites it). Prerequisites shipped (#391); the body-state arms never validly ran. Their revive trigger keys on ledger row T3-11 (B3 Acting Coach), DROPPED 2026-08-30, so it can never fire; the code stays per dormancy.
- [archive/world_channel_weighting.md](archive/world_channel_weighting.md) — **CLOSED by measurement 2026-09-20, archived 2026-10-16**: all four lenses returned DO-NOT-BUILD; #784 stays homed in 1.4 Phase 5.

Archived in the 2026-08-27 post-1.1.0 triage (each verified against code + git history; disposition banner at the top of each doc):
**Archived in the 2026-09-19 plans audit** (shipped, superseded, or deferred without a trigger; each file says which at the top):

- [god_function_decomposition.md](archive/god_function_decomposition.md) — sequenced after
  item 6's Stage 2. **= item 7.**
- [external_critique_response.md](archive/external_critique_response.md) — living scorecard
  against the 2026-08-10 external critique (rows for the diet, docstring pass, fail-loud,
  god-function work). Its table needs a truth pass on rows 2 and 6 (both shipped).
- [oasis_case_study_taught_orient.md](archive/oasis_case_study_taught_orient.md) — ADOPTED
  2026-08-26 as the 1.2 motivating case study: sharing the nursery-taught want; the
  claim that earns it is cross-unit readout on a second Reachy.
- [archive/world_seam_1_1_4.md](archive/world_seam_1_1_4.md) — ARCHIVED 2026-09-19: shipped as 1.1.4 "The world seam" (2026-09-05).
- [archive/persona_cleanup_and_mode_transition.md](archive/persona_cleanup_and_mode_transition.md) — ARCHIVED 2026-09-19: COMPLETE 2026-08-09 (Stage 1 #217; Stages 3–6 in the 1.1 persona hard-remove). `src/` cites this archive path.
- [archive/memory_consolidation_practice.md](archive/memory_consolidation_practice.md) — ARCHIVED 2026-09-19: its own downgrade condition fired (1.1 shipped without touching consolidation).
- [archive/goal_depth_integration.md](archive/goal_depth_integration.md) — ARCHIVED 2026-09-19: no revive trigger.
- [archive/cross_platform_file_lock.md](archive/cross_platform_file_lock.md) — ARCHIVED 2026-09-19: no revive trigger.
- [archive/mesh_doc_transport.md](archive/mesh_doc_transport.md) — ARCHIVED 2026-09-19: no revive trigger.

- [archive/claude_md_diet.md](archive/claude_md_diet.md) — **✅ SHIPPED 2026-08-14 (#509).**
- [archive/bio_docstring_truth_pass.md](archive/bio_docstring_truth_pass.md) — **✅ SHIPPED 2026-08-11 (#492).**
- [archive/exteroception_interoception_seam.md](archive/exteroception_interoception_seam.md) — **✅ SHIPPED 2026-07-22 (#411)** — the seam Exp 52/53/54 credit routes through.
- [archive/embodiment_runtime_wiring.md](archive/embodiment_runtime_wiring.md) — **✅ SHIPPED 2026-07-17 (#400)** Track 1; Layer 2 → live_audio_orient_wiring, Track 2 → the 1.3 reflex tier.
- [archive/exp44_substrate_counterfactual_harness.md](archive/exp44_substrate_counterfactual_harness.md) — **✅ SHIPPED 2026-07-26 (#429).**
- [archive/controlled_llm_primary_embodied_harness.md](archive/controlled_llm_primary_embodied_harness.md) — **✅ SHIPPED 2026-07-15 (G1)**; vehicle role passed to `scripts/exp44/campaign.py` + the 44b pre-registration.
- [archive/console_handle_campaign_injection.md](archive/console_handle_campaign_injection.md) — **✅ SHIPPED 2026-07-26 (#428).**
- [archive/thalamus_relay_design_pass.md](archive/thalamus_relay_design_pass.md) — **✅ SHIPPED 2026-07-17 (#402 + #411).**
- [archive/thalamus_hypothalamus_framing.md](archive/thalamus_hypothalamus_framing.md) — **⛔ SUPERSEDED** by the relay design pass (fork resolved: subsume); kept as the organizing frame.
- [archive/orient_runtime_integration.md](archive/orient_runtime_integration.md) — **⛔ SUPERSEDED** — Landing 1 = live_audio_orient_wiring; Landings 2–3 = the 1.3 reflex tier.
- [archive/percept_testbed_audit.md](archive/percept_testbed_audit.md) — **✅ AUDIT CONCLUDED 2026-07-17 (#401)**: do not build the manifest as conceived.
- [archive/reachy_mini_app.md](archive/reachy_mini_app.md), [archive/maxim_pulse_seed/](archive/maxim_pulse_seed/), [archive/maxim_web_seed/](archive/maxim_web_seed/) — **✅ SEEDS CONSUMED** — the live plans are the `Maxim-pulse` and `Maxim-web` repos; these forks are stale by design.
- `roy/` — **removed 2026-08-27**: an orphan copy; the suite loads `scenarios/cradle/roy_0_smoke.yaml` + `scenarios/roy/roy_0_smoke_iteration.yaml`.

- [archive/release_1_1_checklist.md](archive/release_1_1_checklist.md) —
  **SUPERSEDED 2026-08-19.** Preserved July snapshot; its Oasis/Exp 44 critical path
  was replaced by the current release-closure roadmap.

Archived in the 2026-07-15 deep audit (each verified against code + git history; disposition banner at the top of each doc):
- [archive/v1_refinement.md](archive/v1_refinement.md) — **✅ 1.0 SHIPPED (2026-06-17).** The release master plan; all hard requirements closed 2026-06-15. Residual 1.1 pointers live in sibling plans.
- [archive/benchmarking_1_0.md](archive/benchmarking_1_0.md) — **✅ Gate fired + dispositioned (2026-06-13).** Exp 37/38 executed; performance claim honestly pulled, mechanism/persistence claims stand. Continuation owned by behavioral_graduation_candidates.
- [archive/structural_invariant_tests.md](archive/structural_invariant_tests.md) — **✅ All 3 stages SHIPPED (PRs #279/#280/#281)** — statistic-shape tests, trajectory invariants, multi-agent marker + CI lint.
- [archive/lane_capability_placement_split.md](archive/lane_capability_placement_split.md) — **✅ All phases SHIPPED (PRs #357–#362)** + CLAUDE.md invariant. Hardware tuning per-placement is a documented 1.1+ extension.
- [archive/prompt_caching_for_cloud_backends.md](archive/prompt_caching_for_cloud_backends.md) — **✅ SHIPPED (PR #350).** Byte-stable prefix split; measured ~38% Anthropic ITPM reduction (6× premise honestly falsified in Phase 0).
- [archive/cradle_activation_fixes.md](archive/cradle_activation_fixes.md) — **✅ CONCLUDED.** All P0/P1/P2 fixes shipped (PRs #330/#331); Exp 37 concluded. P3 superseded by `MAXIM_DISABLE_IMAGINATION`.
- [archive/substrate_exploration_policy.md](archive/substrate_exploration_policy.md) — **✅ SHIPPED + VALIDATED.** Exploration policy landed with Exp 41 plumbing (PR #379); Exp 42 GRADUATED using it (PR #380).
- [archive/substrate_primary_cradle_readiness.md](archive/substrate_primary_cradle_readiness.md) — **✅ CONCLUDED.** B1–B5 all resolved; Exp 41 VOID → Exp 42 GRADUATE. Open question A1 handed to the Exp 44 G1 work.
- [archive/sem_world_enrichment.md](archive/sem_world_enrichment.md) — **✅ Phases 1–3 substantially SHIPPED** (`archetype.py`, 7 archetypes, 75 tagged seeds). Only the `maxim_sim_avatar` default-embodiment item was descoped.
- [archive/roy_5_encoder_alignment_disambiguator.md](archive/roy_5_encoder_alignment_disambiguator.md) — **✅ Stages 1–4 ALL RESOLVED.** H1a confirmed (384-dim vs 768-dim); Stage 3 scaffold shipped (PR #295) and adjudicated by Exp 35/36. The open "Stage 5" question is new-plan-shaped.
- [archive/cross_modal_substrate_binding.md](archive/cross_modal_substrate_binding.md) — **❌ CANCELLED, do-not-resurrect.** Roy-4 found zero would-have-bound edges; the resurrection condition was then falsified by Exp 36 (gap closure = EC drift fix, not binding). Cross-modal work continues via jepa_cross_modal_alignment (deferred).
- [archive/cluster_reward_bias_decay_tau_split.md](archive/cluster_reward_bias_decay_tau_split.md) — **✅ All 5 phases COMPLETE** (PR #267 + Exp 30 validation). Downstream gaps handed to sense_tool_registry + imagination_substrate_signals (both since MVP-shipped).

Previously archived (2026-07-15 — plans consolidation + shipped/concluded sweep):
- [archive/audiovisual_orienting.md](archive/audiovisual_orienting.md) — **MERGED UP into [substrate_native_orienting.md](deferred/substrate_native_orienting.md).** Its coordination content (shared orient backbone, two-signal split, Phase 0b credit resolution, phased sequence, fusion endgame) is preserved in the new umbrella plan.
- [archive/release_0_9_1.md](archive/release_0_9_1.md) — **✅ 0.9.1 SHIPPED (2026-05-25).** All stages shipped; Roy-3 outcome recorded; follow-ups redirected to `sense_tool_registry.md` / `imagination_substrate_signals.md` / `cluster_reward_bias_decay_tau_split.md`.
- [archive/config_unification.md](archive/config_unification.md) — **✅ SHIPPED (PR #318, 2026-06-03).** `config.json` operator layer + `resolve_setting` + `maxim config` + role-detector unification.
- [archive/leader_ux_profile_management.md](archive/leader_ux_profile_management.md) — **✅ SHIPPED (PR #314, 2026-05-31).** Bundled + user profiles + `maxim model` verbs.
- [archive/ec_centroid_drift_fix.md](archive/ec_centroid_drift_fix.md) — **✅ SHIPPED (2026-05-24).** Text-modality centroid drift fix (threshold 0.44), PRs #259–#264.
- [archive/auth_format_freeze_audit.md](archive/auth_format_freeze_audit.md) — **✅ SHIPPED (2026-06-15, CC13).** Four security-shaped format-freeze surfaces.
- [archive/doctor_robot_reachable.md](archive/doctor_robot_reachable.md) — **✅ IMPLEMENTED (2026-07-15, PR #392).** `check_robot_reachable` doctor check (resolve → :8000 → daemon status → era-coherence).
- [archive/exp37_metric_pivot.md](archive/exp37_metric_pivot.md) · [archive/exp37_sd_shift.md](archive/exp37_sd_shift.md) · [archive/exp37_cross_model_characterization.md](archive/exp37_cross_model_characterization.md) · [archive/counter_prior_substrate_experiment.md](archive/counter_prior_substrate_experiment.md) — **✅ Exp 37/38/40 CONCLUDED** (Goldilocks zone + counter-prior dominance; 1.0 gates settled PR #371, 2026-06-15). These metric/statistical/pre-reg amendments served their purpose.
- [archive/cloud_dispatch_debug.md](archive/cloud_dispatch_debug.md) — **✅ SERVED PURPOSE.** Cloud-dispatch path validated during the Exp 37 cross-model window.


Recently archived (2026-05-02 — V1 phased re-run wave complete):
- [archive/confound_quarantine.md](archive/confound_quarantine.md) — **✅ PURPOSE SERVED + ARCHIVED.** Flags shipped (PR #214), V1 phased re-run executed against commit `f742527` (2026-04-30, [Experiment 12](../experiments/12_v1_phased_attribution.md)) → all 7 phases recalled `BLUE-7-DAWN`, Phase A clean pass forced flag removal in 1.0 (PR #215). Plan stays in archive as the artifact explaining the disposition; harness `scripts/run_v1_phases.sh` and Experiment 12 stay in-tree for academic-ML reproducibility.

Previously archived (2026-04-29 1.0 audit — verified shipped against code):
- [archive/bio_system_protocol_enrichment.md](archive/bio_system_protocol_enrichment.md) — **✅ COMPLETE + ARCHIVED.** All 5 `*Context` dataclasses live at `src/maxim/models/bio_context.py`.
- [archive/scn_oscillator_feedback.md](archive/scn_oscillator_feedback.md) — **✅ SHIPPED + ARCHIVED.** B2 anticipatory temporal credit. `OscillatorNetwork.predict_event_imminence`, `TemporalCreditDistributor.anticipatory_pre_activate`, `scn.enable_oscillator()` all wired.
- [archive/proprioceptive_discovery.md](archive/proprioceptive_discovery.md) — **✅ Mechanisms A+B SHIPPED + ARCHIVED.** Latent-affordance discovery + entity acquisition.

Recently archived (2026-04-25 plans audit + stall recovery verified shipped):
- [archive/affordance_concept_transfer.md](archive/affordance_concept_transfer.md) — **✅ Stages 0–4 shipped and archived.** Cross-entity transfer is compound-name level only; there is no component path ([#1120](https://github.com/dennys246/Maxim/issues/1120)), and the behavioural claim is pulled (T1-5; owner decision 2026-10-08 to drop it). SCN temporal coupling is live.
- [archive/component_level_damage.md](archive/component_level_damage.md) — **✅ Stages 1-5 SHIPPED + ARCHIVED.** Per-modulator damage, `--deep-embodiment`. Stage 6 deferred.
- [archive/percept_reflex_system.md](archive/percept_reflex_system.md) — **✅ SHIPPED + ARCHIVED.** ReflexRegistry, wired into BioEnrichmentPipeline.
- [archive/sem_entity_ownership.md](archive/sem_entity_ownership.md) — **✅ SHIPPED + ARCHIVED.** Self/scene separation, discovery filter.
- [archive/orchestrator_sem_damage.md](archive/orchestrator_sem_damage.md) — **✅ SHIPPED + ARCHIVED.** Auto-damage + pain cascade. PR #190.
- [archive/deliberative_thought_stream.md](archive/deliberative_thought_stream.md) — **✅ Stages 1+2 SHIPPED + ARCHIVED.** Stages 3/3b/4 absorbed into temporal_credit_integration.
- [archive/temporal_credit_integration.md](archive/temporal_credit_integration.md) — **✅ Phases 1-7 SHIPPED + ARCHIVED.** TemporalEvent, Distributor, ValenceSignal, goal tags.
- [archive/gating_abstraction.md](archive/gating_abstraction.md) — **✅ G0+G1 SHIPPED + ARCHIVED.** SalienceScorer, AdaptiveThresholdController. G2/G3 deferred.
- [archive/agent_factory_canonicalization.md](archive/agent_factory_canonicalization.md) — **✅ F1-F5 SHIPPED + ARCHIVED.** Wave G deferred.
- [archive/llm_path_refinement.md](archive/llm_path_refinement.md) — **✅ Plans 1-4 ALL SHIPPED + ARCHIVED.** Meta-index. All children in archive.
- [archive/substrate_episode_boundary_enrichment.md](archive/substrate_episode_boundary_enrichment.md) — **✅ Stages 1+3 SHIPPED + ARCHIVED.** Stage 2 absorbed into v1_refinement.md.
- [archive/cross_session_sim_validation.md](archive/cross_session_sim_validation.md) — **ABSORBED into v1_refinement.md Section 1.**
- [archive/tool_pain_bridge_temporal_migration.md](archive/tool_pain_bridge_temporal_migration.md) — **ABSORBED into v1_refinement.md Section 2.**
- [archive/orchestrator_stall_recovery.md](archive/orchestrator_stall_recovery.md) — **✅ SHIPPED (PR #181, 2026-04-23).** Content-aware tool cap + feedback + diversity injection. Three-layer defense.

Previously archived (2026-04-23):
- [archive/pfc_deliberation_cycle.md](archive/pfc_deliberation_cycle.md), [archive/interactive_display_overhaul.md](archive/interactive_display_overhaul.md), [archive/working_memory_exec_loop.md](archive/working_memory_exec_loop.md), [archive/substrate_p5_stress_persistence.md](archive/substrate_p5_stress_persistence.md), [archive/concept_exploration.md](archive/concept_exploration.md), [archive/biosystem_unification.md](archive/biosystem_unification.md), [archive/llm_path_operator_visibility.md](archive/llm_path_operator_visibility.md), [archive/deliberation_observability.md](archive/deliberation_observability.md), [archive/circular_import_resolution.md](archive/circular_import_resolution.md), [archive/asset_foundry_plan.md](archive/asset_foundry_plan.md)

Previously archived (2026-04-19):
- [archive/prompt_b4_replanning.md](archive/prompt_b4_replanning.md), [archive/substrate_p6_extinction.md](archive/substrate_p6_extinction.md), [archive/substrate_p8_sleep_replay.md](archive/substrate_p8_sleep_replay.md), [archive/agent_loop_state_repair.md](archive/agent_loop_state_repair.md), [archive/interactive_experience_031.md](archive/interactive_experience_031.md), [archive/peer_update_pip_mode.md](archive/peer_update_pip_mode.md), [archive/substrate_concept_decomposition.md](archive/substrate_concept_decomposition.md), [archive/game_npc_integration.md](archive/game_npc_integration.md), [archive/llm_path_peer_failover.md](archive/llm_path_peer_failover.md)

Previously archived (2026-04-17):
- [archive/sem_learning_loop.md](archive/sem_learning_loop.md), [archive/behavioral_convergence_wiring.md](archive/behavioral_convergence_wiring.md), [archive/cerebellum_activation.md](archive/cerebellum_activation.md), [archive/substrate_valence_annotation.md](archive/substrate_valence_annotation.md), [archive/substrate_binding_persistence.md](archive/substrate_binding_persistence.md), [archive/substrate_binding_split_proposal.md](archive/substrate_binding_split_proposal.md), [archive/bio_stack_unification.md](archive/bio_stack_unification.md), [archive/router_drain_coupling.md](archive/router_drain_coupling.md), [archive/auto_drain_persistent_failure.md](archive/auto_drain_persistent_failure.md)

Previously archived (biosystem unification, 2026-04-14/16):
- [archive/executor_bootstrap_unification.md](archive/executor_bootstrap_unification.md), [archive/pain_bus_unification.md](archive/pain_bus_unification.md), [archive/reaction_bus_unification.md](archive/reaction_bus_unification.md), [archive/memory_hub_unification.md](archive/memory_hub_unification.md), [archive/default_network_unification.md](archive/default_network_unification.md)

Previously archived (LLM path, 2026-04-12/13):
- [archive/llm_path_foundation.md](archive/llm_path_foundation.md), [archive/llm_path_typed_errors.md](archive/llm_path_typed_errors.md), [archive/llm_path_fast_failover.md](archive/llm_path_fast_failover.md), [archive/llm_path_cancellation_hygiene.md](archive/llm_path_cancellation_hygiene.md)

Earlier archives (foundations, cleanup, peer/leader, simulator, substrate):
- [archive/foundations_plan.md](archive/foundations_plan.md), [archive/cleanup_wave.md](archive/cleanup_wave.md), [archive/peer_leader_flexibility_plan.md](archive/peer_leader_flexibility_plan.md), [archive/unified_event_bus_plan.md](archive/unified_event_bus_plan.md), [archive/simulator_upgrades_plan.md](archive/simulator_upgrades_plan.md), [archive/reaction_abstraction_plan.md](archive/reaction_abstraction_plan.md), [archive/substrate_plan.md](archive/substrate_plan.md), [archive/substrate_recognition.md](archive/substrate_recognition.md), [archive/substrate_p0_pilot.md](archive/substrate_p0_pilot.md), [archive/substrate_p3a_episode_binding.md](archive/substrate_p3a_episode_binding.md), [archive/substrate_p3b_channel_integration.md](archive/substrate_p3b_channel_integration.md), [archive/substrate_p3_5_persistence_snapshot.md](archive/substrate_p3_5_persistence_snapshot.md), [archive/substrate_p4_cross_modal_binding.md](archive/substrate_p4_cross_modal_binding.md), [archive/substrate_p4_option2_measurement.md](archive/substrate_p4_option2_measurement.md), [archive/sem_execution_hook.md](archive/sem_execution_hook.md), [archive/07_feature_completion.md](archive/07_feature_completion.md), [archive/sem_tool_discovery.md](archive/sem_tool_discovery.md), [archive/foundational_buildout_plan.md](archive/foundational_buildout_plan.md), [archive/api_surface_hardening_plan.md](archive/api_surface_hardening_plan.md), [archive/embodiment_core_plan.md](archive/embodiment_core_plan.md)

## What 0.3 proved

The 0.3 release demonstrates the core 1.0 claim at prototype scale:

1. **Tier 1 (substrate):** Bio-systems learn affective associations and persist them across sessions (Exp 1: 11/11, Exp 2: 13/13 at fe1124b9: it measured the since-removed `EnergyReactionBridge`)
2. **Tier 2 (LLM reads learning):** The LLM makes different decisions when it sees the agent's learned valence (Exp 3: 12/12)
3. **Tier 3 (organic learning):** The agent learns from its own actions without scripted training (Exp 4: 5/5, teal rate 0% -> 25% -> 100%)

All 41/41 hypotheses confirmed. No fine-tuning. No prompt engineering beyond surfacing the substrate's learned associations.

## Review pattern

Each implementation plan receives two independent pre-merge agent reviews: one
execution/correctness lens and one architecture/maintenance lens. Findings fold into
the implementation branch before the PR opens. The review protocol must be tracked in
the repository; private `.claude/projects/.../memory` files are historical evidence,
not load-bearing instructions.

## Rules for this directory

- **Active plans stay in the root.** Anything in the root is on the critical path or actively shipping.
- **`reviews/` is not a plan.** It holds pre-merge two-lens review artifacts that plans cite as evidence — it stays put. (The former `roy/` fixture copy was an orphan — the suite reads `scenarios/` — and was removed 2026-08-27.)
- **Repo seeds are consumed, not maintained.** `archive/maxim_pulse_seed/` and `archive/maxim_web_seed/` are the frozen seeds of the `Maxim-pulse` and `Maxim-web` repos; edit those repos, not the seeds.
- **Deferred plans must state a revive trigger.** No trigger = archive, not deferred.
- **No ghost plans.** If a plan references a module that doesn't exist, fix or delete the reference.
- **`rationale/` is not a plan either.** It holds design-review lens reports (one subdir per experiment or plan) that preregs and roadmaps cite as evidence — it stays put, like `reviews/`.
- **A plan whose work ships moves in the same PR as the release transaction.** Every one of the 10 root plans archived on 2026-09-19 was "shipped but never moved", which is how the root came to describe work finished three releases earlier.
- **Merge before multiplying.** If two plans overlap by more than a phase, merge them.
- **Automatic behaviours declare their tier** (adopted 2026-09-21). A PR that adds or changes an
  automatic behaviour says in its body whether it is an **invariant** (never learned: safety,
  anything an adversary could exploit), an **innate prior** (coded default, learnable gain) or
  **learned** (substrate, persisted, shareable). If the honest tier is prior or learned but the PR
  hard-codes it anyway, the PR files the "make it learned" follow-up with a revive trigger. Tier map,
  prerequisite and the deferred migrations: [deferred/behavior_tiers.md](deferred/behavior_tiers.md).
