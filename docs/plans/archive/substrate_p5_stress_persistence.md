# Substrate P5 — Robust Cross-Session Persistence Under Stress

**Status:** ✅ COMPLETE (2026-04-21). All 7 stages PASS. 19 tests, 5680 total suite pass. **1.0 GATE CLOSED.**
**Scope:** ~600 LOC stress tests + SemanticMemory serialization bug fix
**Target version:** 1.0 (the final gate)
**Gates:** **1.0 gate — CLOSED**
**Depends on:** P3.5 (persistence protocol), P4 (cross-modal binding), Working Memory Stage 7 (promotion_pressure)
**Blocks:** 1.0 release
**Parent:** [substrate_binding_persistence.md](substrate_binding_persistence.md)
**Related:** [memory_consolidation_practice.md](../deferred/memory_consolidation_practice.md)
**Results:** [experiments/p5_stress_persistence_results.md](../../experiments/p5_stress_persistence_results.md)

## Goal

Prove that the bio-substrate survives realistic persistence load: 10,000+ nodes, 1,000+ episodes, mixed modalities, repeated serialize/reload cycles with no degradation.

## Hypothesis

A substrate populated with 10k+ nodes and 1k+ episodes across mixed channels, serialized every 100 episodes and reloaded, produces identical retrieval results before and after each cycle, with bounded state size and load time <5s.

## Dependencies (scaffolding audit)

| Surface | Status | Notes |
|---|---|---|
| `BioSystemSnapshot` Protocol (P3.5) | Shipped (S1+S2) | `dump()` / `load()` on all bio-systems |
| `persistence_harness.py` (S3) | Shipped | Subprocess round-trip harness |
| Hippocampus save/load | Shipped | Core persistence |
| NAc save/load (reward bias) | Shipped (P2) | `_reward_bias` fields persist |
| ATL save/load | Shipped | Concept-level persistence |
| PerceptTraceBuffer save/load | Shipped (P3.5 S2) | Buffer state round-trips |
| Episode dataclass + EpisodeStore | Shipped (P3a) | Episode-level persistence |
| Cross-modal nodes (P4) | Shipped | Vision + text nodes coexist |

## Stages

### Stage 1 — mechanism + metric (shipped 0.4)

**What's built:**
- Stress fixture generator: creates 10k+ nodes across text + vision modalities with realistic episode structure
- Metric extractor: measures retrieval stability (F1 delta before/after reload), state size growth, load time
- Mechanism test: 10 serialize/reload cycles on a 1k-node substrate, verify zero F1 degradation

**Pass gate:** F1 delta = 0.0 across 10 reload cycles at 1k nodes. State size growth sub-linear. ✅ PASS

### Stage 2 — mid-scale validation (shipped 0.4)

**What's built:**
- Scale to 10k+ nodes (12k: 2500 text + 500 vision episodes × 4 nodes)
- Mixed-modality nodes (text + vision from P4)
- Load time measurement at checkpoint

**Pass gate:** Load time <5s for 10k nodes. F1 delta = 0.0 across all checkpoints. ✅ PASS (load: 0.06s)

### Stage 3 — hippocampus sweep (shipped 0.4, `@pytest.mark.slow`)

**What's built:**
- Full 10-seed sweep with 10k+ nodes each
- State size bounding verification (no unbounded growth)

**Pass gate:** All Stage 2 criteria hold across 10 seeds. ✅ PASS (F1 delta: +0.000000 ± 0.000000)

### Stage 4 — Stage 7 field fidelity (shipped 2026-04-21)

**What's built:**
- 500 memories with diverse `promotion_pressure` (0.0–3.0), `last_scored_at`, `access_contexts` (0–10 entries)
- Field-level exact-match verification after round-trip
- `deque(maxlen=10)` overflow behavior verification after reload
- Tier model test: FORMING / SHORT_TERM / LONG_TERM memories all survive

**Pass gate:** Zero field mismatches across 500 memories. ✅ PASS

**Bug found:** SemanticMemory, Concept, CompressedSemantic in `memory/semantic_types.py` were missing Stage 7 fields in serialization. Fixed.

### Stage 5 — full bio-system round-trip (shipped 2026-04-21)

**What's built:**
- EC: 10,000 substrate nodes (64-dim embeddings), embedding fidelity verification (100 sampled × 64 dims, abs <1e-10)
- ATL: 1,000 concepts with promotion_pressure, confidence, access_contexts
- NAc: 500 causal links with valence, predicted_value, observation_count + 200 reward biases
- Combined: all four systems populated and round-tripped together

**Pass gate:** Load <5s, zero data loss, zero field degradation. ✅ PASS (EC: 0.13s, ATL: 0.01s, NAc: <0.01s)

### Stage 6 — concurrent access stress (shipped 2026-04-21)

**What's built:**
- Hippocampus: 3 writer threads + 1 saver thread, verify no corruption after final save/load
- NAc: 2 observer threads + 1 saver thread
- EC: 2 registrar threads + 1 saver thread
- Multi-agent isolation: 3 agents × 200 memories on separate paths, verify no cross-contamination

**Pass gate:** No thread errors, no corruption, no cross-contamination. ✅ PASS

### Stage 7 — 10-seed full-stack sweep (shipped 2026-04-21, `@pytest.mark.slow`)

**What's built:**
- 10 seeds × full bio-stack (hippocampus 10k nodes + NAc 200 links + ATL 100 concepts + EC 2k nodes)
- Stage 7 field injection with random values per seed
- Field-level fidelity sampling (50 memories per seed)
- EC embedding fidelity sampling (20 nodes per seed)

**Pass gate:** All criteria across 10 seeds. ✅ PASS
- F1 delta: +0.000000 ± 0.000000
- Load time: 0.07 ± 0.00s (max 0.08s)
- Stage 7 field fidelity: 10/10 seeds
- EC node fidelity: 10/10 seeds

## Pass criteria (maps to 1.0 gate)

All criteria met:
- ✅ State size bounded (sub-linear growth with node count)
- ✅ Retrieval stable across 10+ save/reload cycles (F1 delta = 0.0)
- ✅ Load time <5s for 10k nodes (actual: <0.1s)
- ✅ Stage 7 fields survive round-trip (promotion_pressure, access_contexts, last_scored_at)
- ✅ All four bio-systems (Hippocampus, NAc, ATL, EC) at scale
- ✅ Concurrent multi-thread access: no corruption
- ✅ Multi-agent isolation: no cross-contamination
- ✅ Mean + std across 10 seeds

## Load-bearing invariants (filled in AFTER shipping)

**SemanticMemory, Concept, and CompressedSemantic must serialize Stage 7 fields.** The `to_dict()` / `from_dict()` methods in `memory/semantic_types.py` MUST include `promotion_pressure`, `last_scored_at`, and `access_contexts`. The MemoryRecord ABC defines these fields, but each concrete subclass serializes independently — adding a field to the ABC does NOT automatically propagate to subclass serialization. When adding a new MemoryRecord field: update `to_dict()`/`from_dict()` in ALL subclasses: `EpisodicMemory`, `CompressedMemory` (in `memory/types.py`), `SemanticMemory`, `Concept`, `CompressedSemantic` (in `memory/semantic_types.py`). The P5 Stage 4 test is the regression guard.

**Concurrent save is safe because bio-systems use RWLock or RLock during dump().** Hippocampus uses `_rwlock.read()` + `_episode_lock` during dump; ATL uses `_rwlock.read()`; NAc uses `_lock` (RLock). EC has no locking on save — concurrent registration during save may produce a torn snapshot, but the file is always valid JSON (atomic_write_json). The P5 Stage 6 concurrent tests are the regression guards.

**Deferred follow-ups:**
- Schema migration stress (version upgrade during load) — deferred post-1.0
- Incremental save / checkpoint-based restore — deferred post-1.0
