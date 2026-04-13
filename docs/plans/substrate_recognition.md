# Substrate Recognition — B1 + P1 + P2

**Status:** Active — B1+P1 **SHIPPED** (2026-04-12), P2 core merged, P2 validation remaining
**Scope:** ~2,500 LOC across three phases + text-to-prompt migration
**Target version:** 0.3-pre (B1+P1 combined) through 0.3-minimum (P2)
**Blocks:** substrate_binding_persistence.md (P3a onward)
**Master reference:** [archive/substrate_plan.md](archive/substrate_plan.md) for full rationale, baselines, statistical hygiene
**P1 results:** [experiments/p1_recognition_sweep.md](../experiments/p1_recognition_sweep.md) — 91.7% ± 2.9% collapse, paraphrase-mpnet@0.40 + centroid update

## Goal

Text flows through the substrate and learning modulates it. Specifically: text percepts get encoded, recognized by EC, stored in modality-tagged ATL nodes, and NAc reward bias sharpens recognition of behaviorally relevant stimuli.

## Current state (updated 2026-04-12)

**Prereqs (shipped before this plan):**
- PerceptContext typed schema (F0.4), Percept factories (F0.6), PerceptTraceBuffer (F0.2)
- SensoryTag + SensoryModality enum (F0.8), agent_id threading (F0.5), Tier enforcement (F0.7)
- ReactionBus (Phase 2a), FixtureDrivenOrchestrator (S1), MockLLMBackend (S2)
- Persistence harness (S3), Deterministic seeding (S4)

**B1 + P1 — SHIPPED (2026-04-12):**
- `Percept.embedding` + `substrate_node_id` fields — `agents/bus.py`
- `LinguisticEncoder` — `similarity/encoder.py` (paraphrase-mpnet-base-v2, fallback bag-of-words for tests)
- `EC.pattern_complete_or_separate()` with centroid update — `similarity/ec.py`
- ATL modality-tagged nodes + `activate_substrate_node()` + `get_by_modality()` — `memory/atl.py`
- `SubstrateModality` derivation from `SensoryTag` — `agents/modality.py`
- `PromptAssembler` + `MemorySummary` + `SubstrateNode` — `prompts/assembler.py`
- Dual-write wired behind `MAXIM_SUBSTRATE_PATH=1` (Phase 1 of text-to-prompt migration)
- P1 metric extractor — `tests/substrate/p1_metrics.py`
- conftest autouse scrub for `MAXIM_SUBSTRATE_PATH` env var

**P2 core — MERGED (2026-04-12):**
- NAc per-node reward bias keyed by `(agent_id, node_id)` — `decisions/nac.py`
- NAc eligibility traces + `distribute_reward()` + `decay_reward_biases()`
- EC threshold formula wired: `threshold = base - bias` via `get_threshold_overrides()`
- `CausalLink.percept_refs: tuple[TraceSnapshot, ...]` — Phase 5 of reaction abstraction
- NAc save/load persists reward biases

**P2 remaining (what needs to ship for this plan to close):**
- P2 metric extractor plugin (~100 LOC)
- Reward-annotated fixtures (extend paraphrase_clusters.yaml with reward events)
- SEM pain cascade PoC (~230 LOC)
- P2 validation sweep + lab notebook entry
- Text-to-prompt migration Phases 2-4 (shadow read → cutover → legacy removal)

**P2 validation scheduling — runs as Phase A of the LLM path stress test.** Per the meta-plan + stress test protocol, P2 validation shares infrastructure with the Plan 3 Fast Failover stress test. See [../experiments/protocols/llm_path_stress_test.md](../experiments/protocols/llm_path_stress_test.md) "Phase A — Baseline + substrate P2 validation (single-user, one agent)". Running them together means one setup serves both the "does substrate P2 pass mechanistic targets" question AND the "is the pre-Plan-3 52s retry loop still there" baseline. If you are running P2 validation standalone (no stress test), the test fixture in `tests/substrate/test_p2_reward_modulation.py` is the same one Phase A invokes — they are intentionally overlapping so pre-stress runs don't waste effort.

## Modality taxonomy reconciliation

The existing `SensoryModality` enum (SIGHT, SOUND, TOUCH, etc.) captures biological senses for the SEM layer. The substrate needs a coarser TEXT/VISION distinction for EC routing and ATL tag filtering. These serve different purposes:

- `SensoryModality` → "what biological sense produced this?" (SEM layer)
- Substrate modality → "what EC/ATL processing path should this take?" (substrate layer)

**Resolution:** add a `SubstrateModality` field to `Percept` (or derive it from `SensoryTag.modality`) rather than replacing `SensoryModality`. TEXT covers SOUND(speech), NARRATIVE, ABSTRACT. VISION covers SIGHT. This mapping is a ~20 LOC utility, not a new enum war.

## Phases

### B1 — PromptAssembler (single composition point)

One class that takes structured inputs and produces the final system message. Replaces the four scattered prompt locations.

```python
PromptAssembler.compose(
    identity: Persona,
    sensors: SensorState,
    affordances: list[Action],
    scene: SceneContext,
    memory: MemorySummary,
    coach: ActingCoach | None,  # B3, later
) -> SystemMessage
```

**Files touched:** new `prompts/assembler.py`, refactor `agents/prompt_builder.py` to delegate, deprecate ad-hoc injection in `prompts/prompt_profiles.py`.

**Exit:** All NPC and planning-agent system messages flow through `PromptAssembler.compose`. `grep -r "system_message = f\""` returns nothing outside the assembler. `MemorySummary` consumes P1's ATL output.

**Scope:** ~500 LOC refactor.

### Text-to-prompt migration (B1+P1 combined, highest-risk change)

Text content on `Percept.transcript_chunk` currently bypasses the substrate entirely. The migration adds a parallel substrate path alongside the existing direct-to-prompt path.

**Current consumers (verified 2026-04-12):**
| Consumer | Location | Type |
|---|---|---|
| `prompt_builder.py:380-381` | `if percept.transcript_chunk: lines.append(...)` | Prompt-layer — migrate |
| `context_pool.py:255-256` | `if percept.transcript_chunk: parts.append(...)` | Prompt-layer — migrate |
| `sim_adapter.py:56-63` | Reads for observation dicts | Non-prompt — leave alone |
| `skill_matcher.py:162-163` | Reads for tool-selection heuristics | Non-prompt — leave alone |

**Migration approach:** keep `transcript_chunk` as-is, add substrate path alongside:
1. **Phase 1 — dual path (behind flag).** `LinguisticEncoder` routes text through EC+ATL. Both paths write; only legacy reads. Verify parity.
2. **Phase 2 — shadow read.** `PromptAssembler` reads from `MemorySummary` alongside legacy. Log divergences.
3. **Phase 3 — cutover.** Substrate path authoritative. Legacy still writes for rollback.
4. **Phase 4 — legacy removal.** Delete legacy path after one release cycle.

**Rollback:** flipping the flag fully reverts behavior. Integration test pins this.

**Scope:** ~600 LOC (encoder + dual-write + shadow-read + flag wiring).

### P1 — Stable within-modality recognition under controlled paraphrase

**Hypothesis:** EC + modality-tagged ATL collapses paraphrases of the same referent to a single stable ATL node.

**Minimum implementation:**
- `Percept.embedding: list[float] | None = None` field added
- `LinguisticEncoder` producing `Percept(embedding=..., text_node_id=...)` — lands with B1
- `EntorhinalCortex.pattern_complete_or_separate(percept, modality)` → activated or new node
- ATL with modality-tagged nodes, tag-filtered queries, edge enforcement
- `SubstrateModality` derivation from `SensoryTag`
- P1 metric extractor plugin (~100 LOC): paraphrase collapse rate, cluster distinctness, node count stability

**Pass criteria (all must fire):**
- Paraphrase collapse: ≥90% within-cluster → same node
- Cluster distinctness: ≤5% cross-cluster collapse
- Node stability: <10% growth over final 20% of run
- Modality isolation: no text cluster collapses into non-text node (mixed-modality probe)
- Persistence round-trip: ATL subprocess reload, ≥95% node activations preserved
- Sanity floor: within 5 pp of FAISS baseline from P0 (**pinned: 73.5%** — see [P0 results](../experiments/p0_baseline_sweep.md))
- Beats degenerate control (random node assignment) by >30 pp
- Mean + std across ≥10 seeds

**Fixtures:** `scenarios/substrate/paraphrase_clusters.yaml` — authored in P0, reused here.

**Swap points (if mechanism fails, in order):**
1. Similarity metric (cosine → euclidean → learned metric)
2. Pattern completion threshold (static → adaptive-per-node)
3. Encoding granularity (sentence → noun-phrase → head word)
4. Embedding model (sentence-transformers → syntactically-aware)
5. Add shallow coreference resolution (~1 week)

**Scope:** ~300 LOC (EC extension + ATL modality tags) + ~100 LOC metric extractor.

### P2 — Reward-modulated recognition sharpens rewarded nodes

**Hypothesis:** After a reward event credited to node X, near-miss percepts that previously separated now complete to X. Recognition radius expands for relevant stimuli and decays when reinforcement stops.

**Minimum implementation:**
- NAc per-node reward bias keyed by `(agent_id, node_id)`
- NAc eligibility traces reading from `PerceptTraceBuffer`
- EC threshold formula: `threshold = base - α × nac.reward_bias(agent_id, nearest)`
- Reaction abstraction Phase 5: NAc causal link gains `percept_refs: tuple[TraceSnapshot, ...]`
- P2 metric extractor plugin (~100 LOC)
- SEM-to-SEM pain cascade PoC (~150 LOC fixture + ~80 LOC harness)

**Pass criteria (all must fire):**
- Rewarded-node collapse: ≥30% fewer distinct nodes vs unrewarded baseline
- Non-interference: distractor node count matches unrewarded ±5%
- Decay: recognition radius returns toward baseline after reinforcement stops
- Per-agent isolation: rewarding agent A's target → no change in agent B
- Persistence round-trip: ATL + NAc bias + PerceptTraceBuffer
- Beats degenerate control (α=0) — no recognition-radius change
- Sanity floor: rewarded collapse delta is not negative
- Mean + std across ≥10 seeds

**SEM pain cascade PoC:** Sword component (existing in `_data/components/weapons/`) with durability sensor → failure mode at durability < 0.1 → `Reaction(kind="pain")` through ReactionBus → NAc learns context-conditional avoidance → agent prefers `drop_weapon` over `slash` at low durability. Tests the full Percept→Reaction→Learning loop end-to-end.

**Fixtures:** Reuse P1 paraphrase clusters with reward-event annotations.

**Swap points:**
1. NAc reward bias decay rate (τ)
2. Threshold modulation strength (α)
3. Eligibility trace timescale
4. Per-node vs per-cluster modulation
5. Reward magnitude scaling

**Scope:** ~300 LOC + ~100 LOC metric extractor + ~230 LOC SEM PoC.

## Dependencies

```
P0 pilot (fixture calibration)
  └─→ B1 + P1 combined (text-to-prompt + recognition)  [0.3-pre]
        └─→ P2 (reward modulation)                       [0.3-minimum]
              └─→ substrate_binding_persistence.md
```

## Exit criteria for this plan

The recognition plan is complete when:
1. B1 PromptAssembler is the single composition point for system messages
2. P1 passes all mechanistic criteria + persistence round-trip across ≥10 seeds
3. P2 passes all criteria including SEM pain cascade PoC
4. Text flows through LinguisticEncoder → EC → ATL → MemorySummary → PromptAssembler (substrate path authoritative, legacy path removed or flagged off)

Only then does `substrate_binding_persistence.md` open.

## Scope summary

| Item | LOC | Notes |
|---|---|---|
| B1 PromptAssembler | ~500 | Refactor, not net-new |
| Text-to-prompt migration | ~600 | Highest risk. Dual-write + cutover |
| P1 substrate additions | ~300 + ~100 metric | EC + ATL modality |
| P2 reward modulation | ~300 + ~100 metric | NAc bias + traces |
| P2 SEM pain cascade PoC | ~230 | Fixture + harness |
| Persistence round-trip per phase | ~100 | Uses S3 harness |
| **Total** | **~2,230** | |
