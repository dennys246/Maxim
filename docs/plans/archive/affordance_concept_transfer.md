# Affordance Concept Transfer — substrate-native cross-entity learning

**Status:** Refined plan v3 (2026-04-24)
**Scope:** DecompositionStrategy, LinguisticEncoder on-ramp, SCN temporal coupling, BioEnrichment annotations, per-agent imagination, self-affordance encoding
**Depends on:** [sem_entity_ownership.md](sem_entity_ownership.md) (shipped — self vs scene entity separation)
**Gates:** None; behavioral intelligence improvement + first SCN-substrate PoC
**Branch:** `feat/affordance-concept-transfer`

---

## Problem

When the agent encounters `dragon_fire_breath` and later encounters `mage_fire_breath`, it has zero knowledge transfer. Each entity-specific affordance is a fresh learning target. The agent must re-learn "fire breath is dangerous" from scratch for every entity that has it.

Worse: `flame_jet` and `fire_breath` are semantically similar but would never transfer via exact name matching.

## Bio-plausible framing

The brain operates on at least two levels for action understanding:

- **Motor layer (cerebellum):** Specific to the exact entity and context. "How does this particular sword swing?" Forward models, prediction error, timing. Does NOT transfer across entities.
- **Concept layer (cortex/ATL):** Abstract categories. "Swords are sharp," "fire burns," "ranged attacks need dodging." DOES transfer — seeing one dragon breathe fire teaches you about all fire breath.

Transfer happens through **concept decomposition** — the cortex breaks compound experiences into constituent concepts ("fire" + "breath") and learns valence associations on the components. When "flame jet" is encountered, "flame" activates the same cortical representation as "fire" — transfer is automatic through shared substrate.

## Design: substrate-native concept transfer

Instead of building a parallel exact-match system, route affordance names through the **existing** bio-pipeline (LinguisticEncoder → EC → ATL → NAc reward bias). Transfer happens through EC pattern completion — when "flame" maps to the same substrate node as "fire", all reward learning on that node transfers for free.

### Verified: embedding similarity supports this

Tested on leader with `paraphrase-mpnet-base-v2` (EC threshold: 0.40):

| Pair | Cosine | Above 0.40? |
|------|--------|-------------|
| fire / flame | **0.785** | YES — strong transfer |
| fire / inferno | 0.606 | YES |
| fire / blaze | 0.658 | YES |
| heal / cure | 0.680 | YES |
| slash / cut | 0.551 | YES |
| fire breath / flame jet | **0.601** | YES — compound transfer |
| tail sweep / tail swipe | 0.668 | YES |
| poison cloud / toxic mist | 0.649 | YES |
| fire / sword | 0.339 | no — clean negative |
| fire / heal | 0.178 | no — clean negative |

Single-word action components ("breath" / "jet" = 0.258) don't cluster, which is correct — the semantic core ("fire") transfers, the delivery mechanism ("breath" vs "jet") stays distinct.

### The transfer flow

```
1. dragon registered as scene entity
     → affordances: fire_breath, tail_sweep, circle
     → normalize: "fire breath", "tail sweep", "circle"
     → encode each through LinguisticEncoder.encode_decomposed:
         "fire breath" → EC node X (compound concept)
         "fire"        → EC node A (component)
         "breath"      → EC node B (component)
         "tail sweep"  → EC node Y (compound)
         "tail"        → EC node C
         "sweep"       → EC node D
     → NAc eligibility traces created for nodes A, B, C, D, X, Y
     → SCN temporal signature registered for each node

2. Agent uses dragon_fire_breath → gets burned
     → PainBus → NAc.distribute_reward(agent_id, -1.0)
     → SCN temporal coupling: nodes A["fire"], B["breath"] registered
       in same temporal phase → receive negative credit
     → NAc.reward_bias(agent_id, node_A) = -0.15 (negative = dangerous)

3. Later: mage registered with flame_jet
     → normalize: "flame jet"
     → encode "flame" → EC: cosine("flame", "fire") = 0.785
       → COMPLETES to node A (same node as "fire"!)
     → encode "jet" → EC: new node F (no match to "breath")
     → Node A already has negative reward_bias from dragon experience
     → Transfer is automatic: "flame" = "fire" at the substrate level

4. Agent encounters mage → BioEnrichmentPipeline checks node A bias
     → Annotation: "flame_jet [DANGEROUS — learned from fire_breath]"
     → Agent shows caution without re-learning
```

## Pre-existing bugs discovered during audit

**NAc per-tick maintenance is unwired.** `decay_eligibility()` and `decay_reward_biases()` are defined in nac.py but have **zero production callers** — only experiment scripts call them. This means:
- Eligibility traces never decay → persist at original strength indefinitely
- Reward biases never decay → persist forever once set

This is a pre-existing correctness bug. Today it masks the trace-decay problem (affordance concepts keep their eligibility forever). Once we fix the decay bug (wire both methods into the agent tick loop), the SCN temporal coupling becomes load-bearing — without it, traces decay to zero before reward arrives.

**This plan wires the decay calls AND builds SCN coupling in the same stage.** The decay fix is correct behavior; SCN coupling prevents the fix from breaking the new transfer mechanism.

## Stages

### Stage 0: Wire NAc per-tick maintenance (prerequisite fix)

Wire `nac.decay_eligibility()` and `nac.decay_reward_biases()` into the agent tick loop. This fixes the pre-existing bug where traces and biases never decay.

**Where:** The tick loop already calls bio-system maintenance in `runtime/agent_loop.py` or `integration/memory_hub.py`. Add both calls alongside existing per-tick bio-system operations.

**Files:**
| File | Change | LOC |
|------|--------|-----|
| `runtime/agent_loop.py` or `integration/memory_hub.py` | Wire `nac.decay_eligibility()` + `nac.decay_reward_biases()` per tick | +5 |
| `tests/` | Verify decay is called, verify existing behavior not broken | +15 |

### Stage 1: AffordanceDecompositionStrategy + entity registration on-ramp

**1a. AffordanceDecompositionStrategy** — new `DecompositionStrategy` implementation.

Splits underscore-joined affordance identifiers into constituent concepts. Does NOT use spaCy — the input is domain-specific identifiers, not natural language.

```python
class AffordanceDecompositionStrategy:
    """Split affordance identifiers into constituent concepts.

    "fire_breath" → ["fire breath", "fire", "breath"]
    "tail_sweep"  → ["tail sweep", "tail", "sweep"]
    "circle"      → ["circle"]  (single-word, no split)
    """

    def extract(self, text: str) -> list[ConceptChunk]:
        text = text.strip()
        if not text:
            return [ConceptChunk(text="", span=(0, 0))]

        # Normalize: replace underscores with spaces
        normalized = text.replace("_", " ")
        parts = normalized.split()

        chunks = [ConceptChunk(text=normalized, span=(0, len(normalized)))]
        if len(parts) > 1:
            for part in parts:
                if len(part) >= 2:  # Skip single-char fragments
                    chunks.append(ConceptChunk(text=part, span=None))
        return chunks
```

Returns the compound name PLUS its components. Three-level encoding: compound → component-A → component-B. Single-word affordances pass through unchanged.

Plugs into the existing `ConceptDecomposer` constructor via the `strategy=` parameter — no changes to `ConceptDecomposer` itself.

**1b. Entity registration on-ramp** — when a scene entity is registered, encode its affordance names through the substrate path.

**Where:** After entity registration in `ImaginationTrigger._ensure_entity_live()` (seed entities) and `ImaginationTrigger._resolve_phrase()` (imagined entities).

**How:** ImaginationTrigger gets a new optional parameter: `encoder: LinguisticEncoder | None = None`. After registering the entity, extract bare affordance names from `entity.modulators[mod_name].affordances.keys()` and encode each through a dedicated `AffordanceDecompositionStrategy`-backed encoder:

```python
for mod in entity.modulators.values():
    for aff_name in mod.affordances:
        self._aff_encoder.encode_decomposed(aff_name, "text", agent_id)
```

**Critical:** The `agent_id` must be passed through so eligibility traces are filed under the correct agent. ImaginationTrigger gets `agent_id: str = ""` at construction, set by the orchestrator from the AUT agent_id.

**The encoder for affordance names is SEPARATE from the main percept encoder.** It uses `AffordanceDecompositionStrategy` (splits on underscores), while the main encoder uses `SpaCyNounChunkStrategy` (parses natural language). Both share the same EC and ATL instances — the substrate nodes are shared. The affordance encoder is constructed once at ImaginationTrigger construction time:

```python
self._aff_encoder = LinguisticEncoder(
    ec=encoder.ec, atl=encoder.atl, nac=encoder._nac,
    decomposer=ConceptDecomposer(strategy=AffordanceDecompositionStrategy()),
    config=encoder.config,
)
```

**"Forgot to wire" mitigation:** `encoder` defaults to `None`. When `None`, affordance encoding is silently skipped — existing behavior preserved. This is acceptable (not a required keyword-only) because the feature degrades gracefully (no transfer, but no breakage). The encoder is wired at two production sites: `simulation/orchestrator.py` (AUT path) and `embodied_runtime/agentic_runtime.py` (Reachy path). Both already construct `LinguisticEncoder`.

**Files:**
| File | Change | LOC |
|------|--------|-----|
| `similarity/decomposer.py` | Add `AffordanceDecompositionStrategy` | +25 |
| `imagination/trigger.py` | Add `encoder` param, affordance encoding after entity registration | +30 |
| `simulation/orchestrator.py` | Pass encoder to ImaginationTrigger | +3 |
| `tests/unit/test_decomposer.py` | Tests for AffordanceDecompositionStrategy | +30 |
| `tests/unit/test_imagination_trigger.py` | Tests for affordance encoding on registration | +30 |

### Stage 2: SCN temporal coupling for eligibility credit

**Problem (with Stage 0 fix applied):** Eligibility traces decay at 0.9 per tick. Entity registration and tool use can be 50+ ticks apart. By the time `distribute_reward` fires on pain, the affordance concept nodes have zero eligibility.

**Solution:** Extend NAc eligibility with temporal anchoring via SCN. When a concept node is activated, record its `TemporalSignature` alongside the activation strength. When `distribute_reward` fires, nodes whose fast-decay trace has expired but whose temporal signature is similar to the current time still receive credit (at reduced weight).

**NAc changes:**

```python
# New field alongside _eligibility
_temporal_anchors: dict[tuple[str, str], TemporalSignature]  # (agent_id, node_id) → sig

def update_eligibility(self, agent_id, node_id, activation, temporal_sig=None):
    """Extended: also stores temporal anchor when provided."""
    self._eligibility[(agent_id, node_id)] = activation
    if temporal_sig is not None:
        self._temporal_anchors[(agent_id, node_id)] = temporal_sig

def distribute_reward(self, agent_id, reward):
    """Extended: fast-decay traces + temporal-anchored fallback."""
    # 1. Fast-decay path (existing): credit nodes with active traces
    # 2. Temporal fallback: for nodes with expired traces but valid anchors,
    #    compute TemporalSignature.similarity(anchor, now) and credit
    #    proportionally at REDUCED weight (0.3x fast-decay weight)
    ...
```

**Temporal similarity weighting:** For in-session transfer (the primary use case), wall-clock circadian similarity is high (same hour) and useful. For cross-session transfer (secondary), reward_bias persists and handles it — temporal coupling is session-scoped.

**Weight scheme:**
- Fast-decay trace available (recent activation): full credit proportional to trace strength
- Fast-decay expired, temporal anchor available: `0.3 * temporal_similarity * original_activation`
- Neither available: zero credit

The 0.3x factor prevents temporal coincidence from overwhelming direct causal evidence. `TemporalSignature.similarity()` with default weights (all 1.0) produces high scores for same-hour activations.

**Persistence:** `_temporal_anchors` is session-scoped — NOT persisted across saves. Cross-session transfer happens through `reward_bias` (which IS persisted). This is intentional: temporal anchors reference wall-clock time that becomes stale across sessions.

**SCN thread safety:** SCN's `_signatures` dict has no lock. Add a simple `threading.Lock` for the read/write paths touched by this plan. Minimal change.

**Files:**
| File | Change | LOC |
|------|--------|-----|
| `decisions/nac.py` | Add `_temporal_anchors`, extend `update_eligibility` + `distribute_reward` | +40 |
| `similarity/encoder.py` | Pass `TemporalSignature.now()` to `update_eligibility` in encode_decomposed | +5 |
| `time/scn.py` | Add `threading.Lock` for `_signatures` dict | +8 |
| `tests/unit/test_nac.py` | Tests for temporal eligibility + distribute_reward with temporal fallback | +50 |

### Stage 3: BioEnrichmentPipeline + discovery annotations

Surface the transfer to the LLM prompt so the agent can act on transferred knowledge.

**BioEnrichmentPipeline changes:**
1. Already queries ComponentIndex for affordances at Layer 1.
2. Extend: for each affordance found, decompose with `AffordanceDecompositionStrategy`, look up the component words' substrate node IDs in ATL, check NAc `reward_bias` for each.
3. If negative bias exists on any component node, annotate the affordance: `"fire_breath [DANGEROUS — learned from prior experience]"`.
4. If positive bias exists: `"fire_breath [effective — worked well before]"`.
5. Annotations flow through the existing `EnrichmentResult.affordances` field into `StructuredContext`.

**SensePresenceTool changes:**
1. Gets optional `encoder: LinguisticEncoder | None = None` and `nac: NAc | None = None`.
2. For each scene entity affordance, decompose name → look up component substrate nodes → check reward_bias.
3. Annotate capability lines: `fire_breath (ranged fire attack) [DANGEROUS]`.
4. Degrades gracefully when encoder/nac are None — no annotations, existing behavior.

**SenseToolsTool changes:**
1. Already has `nac` — extend `_nac_annotation` to also check bare affordance component nodes.
2. When entity-specific link doesn't exist, fall back to checking component-level reward_bias.
3. This is the "transfer surfaces in discovery" path.

**Files:**
| File | Change | LOC |
|------|--------|-----|
| `integration/bio_enrichment.py` | Affordance component bias lookup + annotations | +30 |
| `tools/discovery.py` | SensePresenceTool + SenseToolsTool annotations via component nodes | +25 |
| `simulation/orchestrator.py` | Pass encoder/nac to tools at construction | +5 |
| `tests/unit/test_bio_enrichment.py` | Annotation tests | +25 |
| `tests/unit/test_tool_discovery.py` | Extended: component-level annotation tests | +20 |

### Stage 4: Integration validation + tracing

**End-to-end sim test:**
1. Dragon registered → verify affordance concept nodes created in EC/ATL.
2. Agent fights dragon, gets burned → verify NAc `reward_bias` on "fire" node goes negative.
3. Mage registered with `flame_jet` → verify "flame" completes to same node as "fire" in EC.
4. Verify BioEnrichment annotates `flame_jet` with [DANGEROUS].
5. Verify Cerebellum has NO forward model for mage (motor layer stays entity-specific).
6. Agent experiences mage healing → verify specific positive experience overrides abstract negative prior.

**Tracing:** `MAXIM_PROVENANCE_VERBOSITY=2` traces the full pipeline. Add structured log events:
- `affordance_concept_encoded`: entity, affordance, node_ids, temporal_sig
- `temporal_credit_applied`: node_id, temporal_similarity, credit
- `affordance_transfer_detected`: source_entity, target_entity, shared_node, bias

**Files:**
| File | Change | LOC |
|------|--------|-----|
| `tests/integration/test_affordance_transfer.py` (NEW) | End-to-end transfer test | +80 |
| Provenance logging across stages | Structured events | +15 |

## What changes (net)

| Stage | LOC est |
|-------|---------|
| Stage 0: Wire NAc decay | +20 |
| Stage 1: Decomposer + on-ramp | +118 |
| Stage 2: SCN temporal coupling | +103 |
| Stage 3: Enrichment annotations | +105 |
| Stage 4: Integration + tracing | +95 |
| **Net** | **~441** |

## Key constraints

1. **Cerebellum stays entity-specific.** Forward models key on `(entity_path, modulator, affordance, param_bucket)`. No change.
2. **NAc.observe() and NAc.record_outcome() stay unchanged.** No hidden side-effects. Transfer happens through the substrate layer (EC pattern completion + reward_bias), not through parallel abstract links.
3. **No new StructuredContext fields.** Affordance annotations flow through the existing `EnrichmentResult.affordances` field.
4. **Entity-specific tool names unchanged.** Motor/execution layer keeps `{entity}_{affordance}` naming.
5. **Bio-system separation preserved.** EC handles pattern completion, ATL handles concept nodes, NAc handles reward bias, SCN handles temporal anchoring. Each does its job.
6. **Affordance names come from Entity spec, not generated tool names.** `entity.modulators[mod_name].affordances.keys()` — the source of truth before collision resolution.
7. **The affordance encoder is SEPARATE from the percept encoder.** Same EC/ATL/NAc backing, different decomposition strategy. Avoids spaCy parsing affordance identifiers as natural language.
8. **Temporal anchors are session-scoped.** Cross-session transfer uses persisted `reward_bias`. Temporal anchors reference wall-clock time that becomes stale.
9. **SCN temporal coupling weight (0.3x) is conservative.** Prevents temporal coincidence from overwhelming direct causal evidence.

## Review findings (folded — v2)

Two-lens review of v1 plan (13 findings, 4 critical) + review of v2 substrate-native design (9 findings, 2 critical). Key findings that shaped this design:

### From v1 review (led to substrate-native pivot)
- **F1:** Abstract links silently never looked up → eliminated (transfer through EC, not name matching)
- **F2:** Tool name → bare affordance extraction unreliable → still relevant, extract from Entity spec
- **F3:** "Forgot to wire" bug class → reduced (encoder defaults to None, degrades gracefully)
- **F4:** Attribution-asymmetry trap → eliminated (no dual NAc links)

### From v2 review
- **R1 (critical):** `decay_eligibility` never called — verified: pre-existing bug, wired in Stage 0
- **R2 (critical):** encode_decomposed creates traces under agent_id="" → fixed: pass agent_id through ImaginationTrigger
- **R3 (high):** TemporalSignature.similarity() doesn't exist → **reviewer was wrong**, method exists at temporal_signature.py:85
- **R4 (high):** SCN not thread-safe → addressed: add Lock in Stage 2
- **R5 (medium):** "Forgot to wire" for encoder → accepted: graceful degradation (None → no transfer, not crash)
- **R6 (medium):** temporal_anchors won't survive save/load → intentional: session-scoped, cross-session via reward_bias
- **R7 (medium):** False temporal credit → mitigated: 0.3x weight, circadian similarity high within session (correct behavior)
- **R8 (low):** SpaCy won't decompose affordance names → addressed: separate AffordanceDecompositionStrategy
- **R9 (low):** SCN signatures dict grows with node IDs → bounded by EC max_nodes (10000 default)

## Open questions — resolved

### Q1: Why substrate-native instead of explicit dual NAc links?
The original plan v1 built a parallel AffordanceConceptBridge with explicit abstract links. The substrate-native approach is superior because:
- Handles synonyms ("flame" ≈ "fire") through embedding similarity, not just exact names
- Uses existing infrastructure (EC, ATL, NAc reward_bias, eligibility traces)
- Transfer is gradient (proportional to cosine similarity), not binary
- No hidden side-effects in NAc.observe()
- ~50% less new code

### Q2: Where does the on-ramp live?
ImaginationTrigger, not BioEnrichmentPipeline or ThalamicGate. Reasons:
- ThalamicGate is a visual percept filter in DefaultNetwork — numerical gating, no semantic content
- BioEnrichmentPipeline runs per-tick on LLM prompt text — wrong granularity for entity registration
- ImaginationTrigger is the canonical site for entity instantiation — affordance concepts should form at perception time, not at prompt assembly time

### Q3: How does SCN coupling prevent false credit?
Three safeguards:
1. **0.3x weight** — temporal credit is always weaker than fast-decay (direct causal evidence)
2. **Circadian similarity** — within a sim session (typically < 1 hour), all activations are in the same hour bin, so temporal similarity is uniformly high. This is CORRECT: everything in the session is temporally relevant.
3. **Agent_id scoping** — `distribute_reward` only credits nodes for the specific agent, not globally

### Q4: What about imagined entities?
Same as v1: imagined entity affordances get encoded through the substrate path. Ephemeral flag propagates via `NAc.tag_imagined_links()` at session end. `decay_imagined_links(0.5)` reduces confidence on imagined causal links. Substrate nodes themselves persist (an imagined "fire" node is the same node as a real "fire" node).

## Validation scenarios

1. **Dragon → Mage fire transfer:** Agent observes dragon fire_breath, gets burned. Later encounters mage with flame_jet. Verify "flame" completes to "fire" node. Verify BioEnrichment shows [DANGEROUS]. Verify agent shows caution.
2. **No false transfer:** Agent gets burned by fire_breath. Later encounters healing_fountain with water_jet. Verify "water" does NOT complete to "fire" node (cosine("water", "fire") < 0.40). No false danger annotation.
3. **Specific overrides abstract:** Agent learns fire_breath is dangerous (dragon). Later learns flame_jet is healing (mage). Verify specific positive experience on "flame_jet" node outweighs abstract negative on "fire" node.
4. **Cerebellum isolation:** Verify Cerebellum has NO forward model for mage entities from dragon experience. Motor precision stays entity-specific.
5. **SCN temporal credit:** With decay wired, verify affordance concept nodes still receive credit through temporal fallback when fast-decay trace has expired. Trace with MAXIM_PROVENANCE_VERBOSITY=2.
6. **Compound + component encoding:** Verify "fire_breath" produces three EC nodes (compound + two components). Verify "fire" node is shared across "fire_breath" and "flame_jet" encounters.
7. **Self-affordance transfer:** Agent's own `slash` affordance and a scene entity's `slash` share the same substrate node. Verify reward from agent's own `slash` success biases the node, and encountering a new entity with `cut` (cosine 0.551) gets the annotation.
8. **Multi-agent isolation:** Two agents in the same scene. Agent A gets burned by fire_breath, Agent B doesn't. Verify Agent A's reward_bias on "fire" node doesn't leak to Agent B (agent_id scoping).

---

## Addendum: Rough-edge refinements (v3, 2026-04-24)

Five architectural refinements from review discussion. These modify Stages 1-3 and add scope.

### RE1: Dynamic temporal weight (NACConfig, not mesh)

**Decision:** `NACConfig.temporal_credit_weight: float = 0.3` with env-var override `MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT` (clamped 0.05-1.0 via `_safe_float_env` pattern from lane_backends.py).

**Why not mesh.yml:** The mesh is strictly identity/topology (node name, URL, role, cluster key). Bio-system learning constants are a different concern. Audit confirmed no existing pattern for mesh-level behavioral tunables — all bio-system configs are hardcoded dataclass defaults today.

**Why not a dynamic/learned value:** Premature. We don't have enough empirical data to design a learning rule for meta-parameters. Start with a tunable constant, measure transfer quality across different session lengths and temporal gaps, then consider adaptive weights if the constant proves insufficient.

**Where different environments matter:** A Reachy robot running real-time has different temporal characteristics than a sim with compressed time. If this becomes a problem, the right pattern is a `bio_config` section in mesh.yml or per-role NACConfig overrides — not scattering individual constants across the mesh parser. Deferred until the need is demonstrated.

**Pair with `conftest.py` autouse scrub fixture** per the `feedback_opt_in_env_in_hot_paths.md` lesson.

### RE2: Reverse index via ATL's existing `_context_index`

**Decision:** Use ATL's existing `recall(name="fire", category="substrate")` for Stage 3 lookups. No new reverse index.

**Audit finding:** Five systems maintain text→ID mappings (ATL `_context_index`, ComponentIndex `_aliases`, AssociationIndex `_keyword_index`, Hippocampus `_context_index`, EC embedding→node). ATL's composite-key pattern (`f"name:{value}"` → `set[id]`) is the de facto standard and already supports the lookup needed.

`recall(name="fire", category="substrate")` hits the `_context_index["name:fire"]` set and filters on `category=="substrate"` — O(1) index lookup, no embedding forward pass. This is sufficient for Stage 3 annotations.

**Follow-up (not this plan):** Extract a shared `TextIndex` class from ATL's pattern and unify with Hippocampus's `_context_index`. Both use the same `f"{field}:{value}"` → `set[id]` pattern with the same RWLock threading model. This would reduce ~40 lines of duplicate index code and give future systems the same lookup primitive.

### RE3: Self-affordance encoding

**Decision:** Encode BOTH self and scene entity affordances through the substrate path. Added to Stage 1.

**Rationale:** Biologically accurate — humans form abstract concepts about their own capabilities and transfer between similar body parts ("I can grip with both hands, but my right is more dexterous"). The same "slash" substrate node should be shared between the agent's own `base_humanoid_slash` and a scene entity's `crystal_dragon_slash`.

**Implementation:** Self-entity affordances get encoded when the agent's body is registered. This happens in the orchestrator AUT setup path (before the agent loop starts) and in `embodied_runtime/agentic_runtime.py` (Reachy path). Both paths already instantiate Entity objects from SEM specs — add the substrate encoding call alongside entity registration.

**Where (new):** `_encode_entity_affordances(entity, encoder, agent_id)` helper called from:
1. ImaginationTrigger._ensure_entity_live (scene entities — existing)
2. ImaginationTrigger._resolve_phrase (imagined entities — existing)
3. Orchestrator AUT setup after self-entity registration (self entities — new call site)
4. `agentic_runtime.py` after Reachy body registration (self entities — new call site)

The helper is the same for self and scene entities — the substrate doesn't distinguish ownership. Ownership is an EntityMap concern, not a substrate concern.

**Stage 1 LOC impact:** +10 (two new call sites + helper extraction).

### RE4: Per-agent ImaginationTrigger

**Decision:** ImaginationTrigger becomes per-agent. Shared infrastructure (ComponentRegistry, ComponentIndex, ImaginationDesigner) stays orchestrator-level. Per-agent state (cache, mention counts, designing guard, imagined_refs, agent_id) is per-trigger.

**Rationale:** Imagination is perception-driven — each agent's perceptual experience should drive its own imagination pipeline. The current orchestrator-level singleton is an artifact of the single-AUT-agent sim mode, not a deliberate architectural choice. Multi-agent scenes need per-agent triggers so:
- Agent A imagining a "crystal dragon" doesn't interfere with Agent B's mention counting
- Per-agent `_cache` prevents cross-agent cache hits that leak perceptual state
- Per-agent `_imagined_refs` enables correct per-agent provenance tagging
- Per-agent `agent_id` enables correct eligibility trace scoping

**Architecture:**
```
Shared (orchestrator-level):
  ComponentRegistry      — entity specs, ephemeral registration
  ComponentIndex         — semantic lookup (alias + embedding)
  ImaginationDesigner    — LLM-based entity design (expensive, shared)
  LinguisticEncoder      — substrate encoding (EC + ATL shared)

Per-agent (ImaginationTrigger instance):
  _cache                 — session-scoped imagination cache
  _designing             — per-phrase concurrent-design guard
  _imagined_refs         — provenance tracking
  agent_id               — for eligibility trace scoping
  _aff_encoder           — affordance-specific decomposer (shares EC/ATL/NAc)
```

**Changes:**
1. ImaginationTrigger constructor gains `agent_id: str` (required keyword-only).
2. Orchestrator creates per-agent trigger instances using shared infrastructure.
3. AgentFactory learns to create triggers for NPC agents (when embodiment is configured).
4. `run_agentic_loop` already takes `imagination_trigger` as a parameter — no change needed.

**Impact on this plan:** Stage 1 on-ramp passes `self.agent_id` to `encode_decomposed` calls — eligibility traces are correctly scoped per-agent. No contamination across agents.

**Stage 1 LOC impact:** +15 (agent_id parameter, orchestrator per-agent construction).

### RE5: Stage 0 regression testing

**Decision:** Stage 0 (wire NAc decay) ships as its own commit with a full behavioral convergence experiment pass.

**Process:**
1. Wire `nac.decay_eligibility()` and `nac.decay_reward_biases()` into the agent tick loop
2. Run existing experiments: `scripts/behavioral_convergence_exp1.py` through `exp4_tier3.py`
3. Compare results against documented baselines in `docs/experiments/`
4. If regression detected, adjust decay rates (the experiments manually call `decay_eligibility(factor=0.5)` for aggressive pre-clearing — the per-tick `0.9` factor is gentler and may not cause issues)
5. Only proceed to Stage 1 after Stage 0 passes

**Note:** The experiment scripts manually call `decay_eligibility(factor=0.5)` in loops of 5-10 iterations to aggressively clear stale traces between entity interactions. The production per-tick call uses `factor=0.9` (gentler, one call per tick). These are different regimes — the experiments may not be affected by the production wiring at all, since they do their own aggressive clearing regardless.

## Revised LOC estimate (v3)

| Stage | LOC est |
|-------|---------|
| Stage 0: Wire NAc decay + regression tests | +25 |
| Stage 1: Decomposer + on-ramp + self-affordances + per-agent trigger | +150 |
| Stage 2: SCN temporal coupling (with NACConfig field + env-var) | +115 |
| Stage 3: Enrichment annotations (using ATL recall) | +105 |
| Stage 4: Integration + tracing + multi-agent validation | +110 |
| **Net** | **~505** |
