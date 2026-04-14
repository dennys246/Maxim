# Substrate Binding, Persistence & Consolidation — P3a through P8 + B3-B5

**Status:** **SPLITTING** (2026-04-14). This monolithic plan is being broken into 8 focused per-phase plans following the P2 shipping pattern. See [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md) for the narrative and [sem_execution_hook.md](sem_execution_hook.md) for the companion plan that closes a cross-cutting gap discovered during the split audit. Until all per-phase plan files land, this document remains the canonical reference for scope + pass criteria on phases that haven't been split out yet. Per-phase plan files opening incrementally:

- [substrate_p3_5_persistence_snapshot.md](substrate_p3_5_persistence_snapshot.md) — **OPEN** (Stage 1 in progress, 2026-04-14). Opens first because P3a's round-trip tests need the `BioSystemSnapshot` Protocol shell.
- [substrate_p3a_episode_binding.md](substrate_p3a_episode_binding.md) — **OPEN** (Stage 1 in progress, 2026-04-14).
- `substrate_p3b_channel_integration.md` — not yet opened; will follow P3a Stage 1 ship.
- `substrate_p4_cross_modal_binding.md` — 1.0-gating; opens after P3a+P3b+P3.5 all green.
- `substrate_p5_stress_persistence.md`, `substrate_p6_extinction.md`, `substrate_p8_sleep_replay.md` — 0.5 track.
- `prompt_b3_b5_track.md` — Track B prompt layer.

**substrate_recognition.md predecessor closed (2026-04-14) for 0.3-minimum** — P1+P2 Stages 1+2+3 all shipped. This plan is no longer blocked.

**Scope:** ~4,100 LOC across 10 phases (7 substrate + 3 prompt)
**Target version:** 0.3-target (P3a/P3b/P3.5/P4) through 0.5 (P5/P6/P8), with prompt track at 0.4
**Master reference:** [archive/substrate_plan.md](archive/substrate_plan.md) for full rationale, baselines, statistical hygiene

## Goal

Prove that episode binding, cross-modal retrieval, persistence, extinction, and offline consolidation work as designed — against plausible baselines a skeptic would actually propose. This plan covers the "heavier" substrate phases that build on the recognition layer from `substrate_recognition.md`.

## Dependencies

```
substrate_recognition.md (B1 + P1 + P2)
  └─→ P3a (episode binding)
        └─→ P3b (channel integration)
              └─→ P3.5 (cross-session persistence + BioSystemSnapshot)
                    └─→ P4 (cross-modal binding — 1.0-GATING)  [0.3-target]
                          └─→ P5 (stress persistence)           [0.5]
                                └─→ P6 (extinction vs LRU)      [0.5]
                                      └─→ P8 (sleep replay)     [0.5]

Track B (interleaved):
  B3 Acting Coach        [0.4, after B1]
  B4 Replanning          [0.4, after B1+P3a — gates 1.0]
  B5 Embodiment/narrative separation [0.4, after B1]
```

## Track A — Substrate phases

### P3a — Episode binding produces retrieval on partial cue

**Split from old P3.** Tests the mechanism — Hebbian link formation and partial-cue retrieval — on synthetic episodes. If this fails, the mechanism is wrong. If this passes and P3b fails, only the channel rules are wrong.

**Hypothesis:** Nodes co-occurring in the same hippocampus episode form durable links; presenting a partial cue retrieves the others, by a margin greater than a TF-IDF bag-of-concepts baseline.

**Minimum implementation:**
- `Episode` dataclass: id, start_tick, end_tick, channel, sender_ids, thread_id, activated_nodes, reward_events, scn_tag
- Hippocampus episode store with episode-to-node edges
- Generic episode boundary (tick gap + scene signal)
- Hebbian within-ATL edge updates on episode close
- Retrieval path (cue → episode lookup → reconstruct → return co-occurring nodes)

**Pass criteria:** Precision >0.70, recall >0.70. Beats TF-IDF gate baseline by `baseline_mean + 2×baseline_std`. Persistence round-trip. Mean + std across ≥10 seeds.

**Fixtures:** `scenarios/substrate/synthetic_episodes.yaml` — 100 synthetic episodes with labeled co-occurrence. ~1-2 days authoring.

**Scope:** ~400 LOC + ~100 metric extractor.

### P3b — Channel integration: episode boundary rules + filtered retrieval

Tests per-channel boundary rules (SMS contact+gap, narrative scene change) on realistic data.

**Hypothesis:** Per-channel episode boundary rules produce episodes whose channel-filtered retrieval beats a metadata-only grep baseline.

**Minimum implementation:**
- Per-channel episode boundary rules (SMS + narrative for 0.3)
- SMS and narrative channel adapters
- Retrieval filter by sender / channel

**Pass criteria:** Precision >0.70, recall >0.70 on channel-filtered retrieval. Regression check vs metadata-grep baseline. Persistence round-trip. Mean + std across ≥10 seeds.

**Fixtures:** `scenarios/substrate/channel_episodes.yaml` — 100 realistic SMS + narrative episodes. ~1-2 days authoring.

**Scope:** ~250 LOC + ~100 metric extractor.

### P3.5 — Cross-session persistence + BioSystemSnapshot protocol

**Hypothesis:** ATL nodes, hippocampus episodes, NAc reward biases, and channel episode structure survive serialization. A reloaded system recognizes the same nodes and respects the same reward biases.

Most persistence already exists (ATL, Hippocampus, NAc all have save/load). The main new work is:
- NAc save/load covering per-node reward bias fields (added in P2)
- PerceptTraceBuffer save/load (currently manual-only)
- `BioSystemSnapshot` Protocol for unified schema-versioned snapshots
- `SessionSnapshot` composing all bio-system snapshots
- Cross-layer round-trip harness using S3 subprocess harness

**BioSystemSnapshot Protocol:**
```python
class BioSystemSnapshot(Protocol):
    schema_version: int
    def dump(self) -> dict[str, Any]: ...
    @classmethod
    def load(cls, state: dict[str, Any]) -> Self: ...
```

All five bio-systems (ATL, Hippocampus, NAc, SCN, PerceptTraceBuffer) implement this. `migrate(old_state, from_v, to_v)` handles forward migration.

**Pass criteria:** Node identity round-trips. Edge weights match within tolerance. Episode retrieval identical. NAc biases round-trip. PerceptTraceBuffer state round-trips.

**Known exclusion:** ATL callbacks (`_on_concept_captured`, `_on_concept_deleted`) are live Callables — not persisted, re-registered post-load.

**Scope:** ~500 LOC (protocol + NAc fields + PTB save/load + cross-layer harness + schema versioning).

### P4 — Cross-modal binding via hippocampus (1.0-GATING)

**This is the architecture's central claim.** Nodes of different modalities co-occurring in the same episode can cue each other across modality boundaries through episode reconstruction, by a margin greater than a shared-embedding-space baseline.

**The "mug test":** Session 1: text "mug" + vision mug co-occur in episodes. Save state. Session 2 (subprocess): present text "mug" alone. Vision mug node should be retrieved.

**Minimum implementation:**
- Minimal `VisionEncoder` using CLIP (single-object → embedding → `Percept(modality="vision")`)
- Cross-modal retrieval path: text cue → ATL text node → hippocampus episode → reconstruct → vision nodes
- Symmetric vision-cue path
- CLIP shared-space baseline for head-to-head

**Pass criteria:** Forward/reverse retrieval >80%. False-binding <10%. Rewarded margin >15%. **Beats OpenCLIP baseline by `baseline_mean + 2×baseline_std`** — this is the whole justification for commitment #3. Losing here is plan-ending. Mean + std across ≥20 seeds (doubled from other phases).

**Scope:** ~500 LOC + ~100 metric extractor.

### P5 — Robust cross-session persistence under stress

Long-running mixed-channel sim (10,000+ nodes, 1,000+ episodes). Serialize every 100 episodes. Reload. Verify no degradation.

**Pass criteria:** State size bounded. Retrieval stable across ≥10 save/reload cycles. Load time <5s for 10k nodes.

**Scope:** ~400 LOC + ~100 metric extractor.

### P6 — Extinction without reinforcement

Two groups: A (reinforced) and B (not). Simulated time via SCN phase ticks (not wall clock). Group B retrieval drops below 20%; Group A stays above 80%.

**Must beat LRU gate baseline** by `baseline_mean + 2×baseline_std`. If LRU matches, graded decay is unjustified.

**Scope:** ~300 LOC + ~100 metric extractor.

### P8 — Minimum-viable sleep replay and consolidation

During an explicit sleep phase, replay top-N rewarded episodes with Hebbian link updates. Retrieval F1 improves on replayed probes without new input.

**Deliberately not ambitious.** One strategy, one scheduling rule, one measurable improvement. Everything else goes to [memory_consolidation_practice.md](memory_consolidation_practice.md).

**Scope:** ~350 LOC + ~100 metric extractor.

## Track B — Prompt layer

### B3 — Acting Coach layer (0.4)

Meta-prompt scaffold: role values, speech register, failure modes, continuity contract. Optional per-character.

**Exit:** Blind A/B test: acting-coach NPC measurably more consistent.

**Scope:** ~300 LOC.

### B4 — Replanning with failure diagnosis (0.4, gates 1.0)

Structured replan: failure point, evidence, prior attempts, root cause, alternatives, selection. Persist replan attempts in-session.

**Exit:** Induced failure → plan 2 differs structurally from plan 1 → plan 3 doesn't repeat either.

**Dependencies:** B1, P3a (episode retrieval of prior attempts).

**Scope:** ~400 LOC.

### B5 — Embodiment/narrative separation (0.4)

Formalize SEM → embodiment, DM → narrative, PromptAssembler → composition. Lint-style contract test.

**Scope:** ~150 LOC.

## Version mapping

| Version | Phases | What it proves |
|---|---|---|
| **0.3-target** | P3a, P3b, P3.5, P4 | Episode binding, channel integration, persistence certified, cross-modal binding beats OpenCLIP |
| **0.4** | B3, B4, B5 | NPCs coherent, replanning recovers, embodiment/narrative separated |
| **0.5** | P5, P6, P8 | Persistence under stress, appropriate forgetting, offline consolidation |

## Scope summary

| Phase | LOC | Notes |
|---|---|---|
| P3a Episode binding | ~400 + ~100 metric | Hebbian + retrieval |
| P3b Channel integration | ~250 + ~100 metric | Boundary rules + SMS stub |
| P3.5 Persistence + BioSystemSnapshot | ~500 + ~100 metric | Protocol + cross-layer |
| P4 Cross-modal binding | ~500 + ~100 metric | Vision encoder + mug test. **1.0-GATING** |
| P5 Stress persistence | ~400 + ~100 metric | 10k+ nodes, mixed channels |
| P6 Extinction | ~300 + ~100 metric | Decay + LRU head-to-head |
| P8 Sleep replay | ~350 + ~100 metric | Minimum-viable consolidation |
| B3 Acting Coach | ~300 | Meta-prompt scaffold |
| B4 Replanning | ~400 | Failure diagnosis. **Gates 1.0** |
| B5 Separation | ~150 | Lint-style contract |
| Persistence round-trip per phase | ~50 × 7 = ~350 | Uses S3 harness |
| **Total** | **~4,100** | |

## If things fail

See [archive/substrate_plan.md](archive/substrate_plan.md) "If the whole thing fails" section for per-commitment fallbacks. The critical one: if P4 loses to OpenCLIP, commitment #3 (hippocampus-only cross-modal binding) is revisited — that's a plan-level finding, not a bug fix.
