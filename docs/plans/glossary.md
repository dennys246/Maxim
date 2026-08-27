# Maxim glossary — decoding the codes

Maxim's plans, experiments, and agent-guidance invariants use terse coded IDs (`CC3`, `Wire-A`, `Roy-4`, `Exp 42`, `NAc`). They look opaque on day one, but they are **stable anchors**, not throwaway labels: each ties a docstring marker → a CI grep → a tracked invariant → the PR that shipped it → a graduation-candidate row. **They are deliberately not renamed** (renaming severs that traceability — see the reasoning in any "opaque naming" discussion). This page is the decoder ring.

**To resolve any code yourself:** `rg -n "CC3" AGENTS.md CLAUDE.md docs/` — during the 1.1 single-source migration, the invariant, its regression guard, and the plan that introduced it remain findable across both adapters and tracked docs.

---

## 1. Version themes (the timeline anchor)

| Version | Theme |
|---|---|
| 0.6 | Generalizable embodiment (sim wiring + Asset Foundry) |
| 0.7 | Self-generating simulations (Imagination, Acting Coach, SEM discovery) |
| 0.8 | Cognitive maturity + embodiment (WM+Exec, PFC cycle, temporal credit, reflexes, affordance transfer) |
| 0.9.1 | Substrate-annotates-LLM-context (Wires A+1+2+3, Roy-3 validation, EC drift fix) |
| 0.9.2 | Config unification + Hivemind shareability + LLM timeout scalability |
| 0.9.3 | Loud optional-dependency failures (`utils/optional_deps.py`) |
| 1.0 | Validation + stabilization + grounding (cross-session proof, protocol freeze, cradle) |
| 1.1 | Sensorimotor release closure: merged embodiment + correctness, contract, truth, and verification debt; zero new mechanisms |
| 1.2 | Gated Oasis + Hivemind federation |

## 2. Work-item prefixes

Each prefix scopes to a **home plan**; the number is a sub-item. Where a prefix is **reused across plans**, context (the surrounding plan/version) disambiguates.

| Prefix | Expands to | Home / context | Examples |
|---|---|---|---|
| `CC` | **C**ontract **C**larification | 1.0 freeze ([v1_refinement.md](archive/v1_refinement.md)) — frozen-dataclass + API contracts | CC1 `_format_version`, CC3 frozen-dataclass audit, CC8 `PerceptSource`, CC9 dual-schema, CC10 CWD paths, CC11 `Tool.cancel` |
| `C` | **C**leanup | 1.0 cleanup wave; also config_unification (C1–C4) | C1 format-version freeze, C2 config-writer, C3 role detection, C6 hard-error flip |
| `B` | **B**io-system stabilization | v1_refinement / release_0_9_1 / Exp 42 | B2 SCN oscillator, B3 Acting Coach, B5 Hivemind shareability, B8 delta-attribution |
| `G` | **G**ate | Exp 44 harness ([controlled_llm_primary_embodied_harness.md](archive/controlled_llm_primary_embodied_harness.md)) | G1 deterministic scene embodiment, G2 drive-gating, G3 terse narrator |
| `E` | **E**mbodiment | 0.6 sim wiring + Foundry | E0 sim wiring, E1 Asset Foundry, E2–E3 real-LLM + auto-curation |
| `I` | **I**magination | 0.7 | I1 trigger, I2 real-time design, I3 scene-scoped tools |
| `Wave` | biosystem-unification wave | biosystem_unification plans | Wave 1 buses, Wave 2 hubs, Wave 3 `build_bio_stack` |

**⚠ Reused prefixes — disambiguate by context:**

| Prefix | Scheme A | Scheme B | Scheme C |
|---|---|---|---|
| `P` | **Substrate pilot** — P0 baseline / P1 recognition / P2 reward-modulation (substrate_* plans) | **Pipeline completion** P1–P4 (v1_refinement) | **Peer/leader flexibility** P1–P9 (llm_path plans; e.g. P9 decision log) |
| `F` | **F**oundations F0.1–F0.8 (early waves) | **Agent Factory** F3–F5 (0.7 migration) | — |
| `R` | **R**eview rounds R1–R3 (pre-merge two-lens) | Plan-2 sub-stages R2a/R2b/R2c (llm_path) | R0 prerequisites |
| `S` | **S**ubstrate test infra S1–S4 (S1 fixture, S2 backend protocol, S4 seeding) | **S**imulator upgrades S1–S4 | — |

Tell them apart by the plan or version in scope: "P2 Stage 2" + substrate → reward-modulation; "P9" + peer/llm_path → decision log; "R2b" → Plan-2 llm_path, not a review round.

## 3. Bio-system + architecture abbreviations

**Load-bearing names — never renamed** (CLAUDE.md: they carry the mental model; a CI grep even blocks `NucleusAccumbens`).

| Code | Full name | Role |
|---|---|---|
| Hippocampus | — | Episodic memory (capture, recall, consolidation) |
| `NAc` | Nucleus Accumbens (class is `NAc`, not `NucleusAccumbens`) | Reward bias, causal links, eligibility traces |
| `EC` | Entorhinal Cortex | Pattern completion / separation (concept clustering) |
| `ATL` | Anterior Temporal Lobe | Semantic hub |
| `SCN` | Suprachiasmatic Nucleus | Temporal / circadian credit (oscillator) |
| `DN` | Default Network | Reactive behaviors, arousal, novelty |
| AngularGyrus | — | Cross-modal association |
| `SEM` | Sensory–Entity–Modulator | Embodiment spec system (sensors, drives, affordances) |
| `AUT` | Agent Under Test | The learning agent inside a sim |
| `WM` | Working Memory | Exec-owned active-reference layer (not a memory tier) |
| `PFC` | Prefrontal Cortex | Deliberation cycle |
| `DoA` | Direction of Arrival | Audio localization (Reachy mic array) |

## 4. Experiment + mechanism codenames

| Codename | What it is |
|---|---|
| `Roy-N` | The paraphrase-collapse / persona-inertness experiment lineage (Roy-2c … Roy-5b) — substrate probes |
| `Wire-A / Wire 1 / 2 / 3` | Substrate→LLM-context annotation mechanisms (0.9.1): Wire-A cluster-bias, Wire 1 variance, Wire 2 imagination-signal, Wire 3 |
| `Exp N` | Numbered experiments in [docs/experiments/](../experiments/) (e.g. Exp 42 substrate-primary GRADUATE, Exp 44 LLM-primary choice) |
| Goldilocks zone | The Exp 37/38/40 finding: substrate helps only where the LLM prior leaves headroom |
| cradle | Sensorimotor developmental sim (infant body, narrative acts) |
| `cradle_mother` / reactive mother | **The teacher mother** — the operant-conditioning caregiver NPC that feeds + credits the infant AUT to shape orienting (`NAc.credit_operant_reward`). SUPERSEDED/DORMANT; operant claim validated on scripted substrate + Exp 46/48. **Unrelated to "Mother Maxim."** |
| LLM-AUT mode | The default individual agent — an LLM as action selector over the bio-substrate. The perpetual *data source* feeding Oases (1.1→1.2). |
| Oasis | A persistent substrate-primary **instance** — a real bio-agent (own NAc/EC/ATL/Hippocampus, *not a database*; multiple coexist) that absorbs contributions, distills consensus, + broadcasts back. Gated for 1.2. **The current name for the old "Mother Maxim."** |
| Hivemind | The P2P substrate-sharing **protocol / fabric** — a bundle format + exchange convention, **not a hosted service** ("nobody hosts the Hivemind"). Connects Oases + substrate-primary Maxims. `src/maxim/hivemind/` (1.0 bundle/merge; full P2P 1.2). |
| — *axis* | Both federate **substrate** (NAc/EC/reflex/ATL), never **mind** (episodes/dialogue are local-only). Hivemind = the network; an Oasis = a hub-agent on it. |
| Mother Maxim | **DEPRECATED** (superseded 2026-05-09). The old federated/database-backed parent that stored + redistributed *user memories*; renamed → **Oasis** (+ Hivemind). Not the teacher mother — see `cradle_mother`. |

**⚠ Two unrelated "mothers":** **Mother Maxim** = deprecated federation/memory-redistribution plan (now Oasis). **`cradle_mother`** = the operant *teacher* caregiver in the orient sim. Same word, no relationship.

## 5. Status vocabulary

| Term | Meaning |
|---|---|
| SHIPPED | Landed on a released version |
| GRADUATE | A behavioral claim earned mechanism-level (measured, not asserted) — the bar for `[behavioral]` |
| Earned / Stale / Broken | [Graduation-candidate](behavioral_graduation_candidates.md) states; Stale/Broken block the next release |
| Dormant | Mechanism stays wired but nothing new builds on it (dormancy-over-deletion); resurrection needs an earning experiment |
| DEFERRED | Plan paused with a documented revive-on-trigger condition ([deferred/](deferred/)) |
| `[engineering]` / `[behavioral]` | Invariant tiers: code breaks loudly without it vs empirically validated as carrying behavioral weight |

---

*This is a decoder, not a rename. The codes stay stable so `git log --grep`, CI greps, and cross-references keep working. If a code isn't here, `grep -rn "<code>" CLAUDE.md docs/` finds its origin.*
