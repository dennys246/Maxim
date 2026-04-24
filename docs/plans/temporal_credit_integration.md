# Temporal Credit Integration — generalized signal-time-reward system

**Status:** Shell plan (2026-04-24)
**Scope:** NAc, SCN, ToolPainBridge, ReactionBus, LinguisticEncoder, bio_stack wiring
**Depends on:** [affordance_concept_transfer.md](affordance_concept_transfer.md) (shipped — first SCN-substrate closed loop)
**Gates:** None; infrastructure generalization + integration testing for affordance transfer
**Branch:** `feat/temporal-credit-integration`

---

## Problem

The affordance concept transfer feature (shipped 2026-04-24) established the first closed SCN→NAc loop: temporal anchors on eligibility traces enable credit attribution after fast-decay expiry. But this is one consumer-specific wiring in one path. The broader signal-time-reward architecture has seven identified paths, and only one has temporal coupling.

**Audit (2026-04-24) found 7 signal→time→reward paths:**

| # | Signal source | Temporal context | Used in attribution? |
|---|---------------|-----------------|---------------------|
| 1 | Tool outcome → `nac.observe()` | `delta_seconds` only | No — stored for delay prediction, not credit |
| 2 | Embodiment failure → ToolPainBridge | `TemporalSignature` → SCN bins | No — stored, never read back |
| 3 | Pain signal → ToolPainBridge | `TemporalSignature` → SCN bins | No — stored, never read back |
| 4 | Percept → LinguisticEncoder → eligibility | `TemporalSignature` → `_temporal_anchors` | **YES** — affordance transfer (shipped) |
| 5 | Reaction → ReactionBus → distribute_reward | None | No — no temporal context on Reactions |
| 6 | Out-of-band pain → context similarity | `temporal_window_seconds` only | Partial — wall-clock delta, no phase |
| 7 | SCN.register (4 call sites in ToolPainBridge) | `TemporalSignature` → SCN bins | No — one-way write, never read back |

The current state is fragmented: each bridge independently creates TemporalSignatures, registers with SCN, then separately calls NAc with no link between them. Path 4 works because we explicitly wired `_temporal_anchors` — but the pattern doesn't generalize without a shared abstraction.

## Goal

A generalized, compartmentalized **temporal credit system** where:
1. **Any signal source** can emit a temporally-anchored event
2. **SCN temporal phase** is automatically captured at emission time
3. **NAc reward attribution** automatically considers temporal similarity when crediting events
4. **The abstraction is reusable** — adding a new signal source requires implementing one protocol, not wiring three systems

Plus: integration testing for the shipped affordance transfer feature, end-to-end sim validation, and rough-edge smoothing.

## Design

### Layer 1: TemporalEvent protocol

A single envelope type that carries event metadata + temporal context:

```python
@dataclass(frozen=True)
class TemporalEvent:
    """A temporally-anchored event for credit attribution.

    Emitted by signal sources (tool outcomes, pain, reactions, percepts).
    Consumed by NAc temporal credit and SCN temporal indexing.
    """
    event_id: str              # unique event identifier
    event_type: str            # "tool", "pain", "reaction", "percept", "affordance"
    event_signature: str       # hashable identifier (tool name, affordance name, etc.)
    agent_id: str              # scoping
    temporal_sig: TemporalSignature  # SCN phase at emission time
    activation: float          # signal strength (1.0 for direct, <1.0 for indirect)
    context: dict[str, Any]    # event-specific metadata
```

Signal sources emit `TemporalEvent` instead of making ad-hoc `TemporalSignature.now()` + `SCN.register()` + `NAc.update_eligibility()` calls separately.

### Layer 2: TemporalCreditDistributor

Unified distributor that replaces the current split paths:

```python
class TemporalCreditDistributor:
    """Centralized temporal credit attribution.

    Receives TemporalEvents, stores anchors, distributes reward with
    temporal similarity fallback. Replaces the ad-hoc _temporal_anchors
    dict and the inline distribute_reward temporal path.
    """

    def __init__(self, nac: NAc, scn: SCN | None = None, *,
                 temporal_credit_weight: float = 0.3):
        ...

    def record_event(self, event: TemporalEvent) -> None:
        """Record a temporal event for future credit attribution.

        Stores in NAc eligibility + temporal anchors, registers in SCN.
        """
        ...

    def distribute(self, agent_id: str, reward: float) -> list[tuple[str, float]]:
        """Distribute reward with temporal credit fallback.

        Two-path: fast-decay traces first, temporal anchors second.
        Delegates to NAc.credit_node for the actual bias update.
        """
        ...

    def query_temporal_patterns(self, event_signature: str) -> dict[str, float]:
        """Query SCN for temporal patterns of an event type.

        Returns phase→frequency histograms. Enables future "does this
        tool fail more at night?" queries.
        """
        ...
```

### Layer 3: Signal source migration

Migrate each signal path to emit `TemporalEvent`:

| Path | Current | After |
|------|---------|-------|
| Tool outcome | `nac.observe()` + `scn.register()` | `distributor.record_event(TemporalEvent(...))` + `nac.observe()` |
| Embodiment failure | `nac.record_outcome()` + `scn.register()` | `distributor.record_event(...)` + `nac.record_outcome()` |
| Pain signal | `nac.record_outcome_full()` + `scn.register()` | `distributor.record_event(...)` + `nac.record_outcome_full()` |
| Percept encoding | `nac.update_eligibility(..., temporal_sig)` | `distributor.record_event(...)` |
| Reaction | `nac.distribute_reward()` (no temporal) | `distributor.distribute()` (with temporal) |

**Key constraint:** NAc's causal link system (`observe`, `record_outcome`, `record_outcome_full`) stays unchanged — it handles event→outcome association. The temporal credit system is **orthogonal** — it handles temporal phase → credit weighting. They coexist, not merge.

### Layer 4: Oscillator feedback (future)

Once Layers 1-3 are stable:
- SCN oscillator `observe(signature)` called on each `TemporalEvent`
- Coupling weights learn co-occurrence patterns (Hebbian on Kuramoto phases)
- `predict_next_occurrence(event_signature)` becomes actionable
- Anticipatory credit: pre-activate eligibility traces for events predicted by the oscillator

This is deferred until Layers 1-3 prove stable in production.

## Integration testing for affordance transfer (shipped feature)

### IT-1: Substrate-level transfer verification

Unit-level tests exist (shipped with Stages 0-3). These integration tests verify the full pipeline end-to-end without mocking.

```
Scenario: Dragon → Mage fire transfer (no LLM)
Setup:
  - Build full bio-stack with real EC + ATL + NAc + SCN
  - Register dragon entity with fire_breath affordance
  - Encode affordances through substrate path
  - Simulate tool outcome: dragon_fire_breath → NEGATIVE valence
  - Register mage entity with flame_jet affordance
  - Encode mage affordances through substrate path

Assertions:
  1. Dragon "fire" and mage "flame" complete to same EC node (cosine > 0.40)
  2. NAc reward_bias on shared node is negative after dragon pain
  3. BioEnrichmentPipeline annotates flame_jet with [DANGEROUS]
  4. SensePresenceTool output contains [DANGEROUS] for mage flame_jet
  5. SenseToolsTool _nac_annotation returns "caution" for fire-containing tools
  6. Cerebellum has NO forward model for mage (entity-specific isolation)
```

### IT-2: No false transfer (negative control)

```
Scenario: Fire transfer does NOT contaminate water
Setup:
  - Same bio-stack as IT-1
  - After dragon pain, register fountain entity with water_jet affordance

Assertions:
  1. "water" does NOT complete to "fire" node (cosine("water","fire") < 0.40)
  2. water_jet has NO [DANGEROUS] annotation
  3. No NAc reward_bias on water node
```

### IT-3: Specific overrides abstract

```
Scenario: Mage fire_heal overrides dragon-learned fire danger
Setup:
  - Same bio-stack, dragon fire_breath → NEGATIVE
  - Then mage fire_heal → POSITIVE (repeated 5x)

Assertions:
  1. Shared "fire" node has mixed bias (positive experience overrides)
  2. Entity-specific mage_fire_heal link is POSITIVE with high confidence
  3. After sufficient positive experience, annotation changes from [DANGEROUS] to [effective]
```

### IT-4: SCN temporal coupling under decay

```
Scenario: Temporal credit survives eligibility decay
Setup:
  - Encode dragon affordances → eligibility traces + temporal anchors
  - Run 200 decay cycles (simulating 200 ticks)
  - Trigger distribute_reward

Assertions:
  1. Fast-decay eligibility traces are gone (pruned)
  2. Temporal anchors survive (recent timestamp)
  3. distribute_reward credits concept nodes via temporal fallback
  4. Credit amount is 0.3x the temporal_credit_weight × temporal similarity
```

### IT-5: Multi-agent isolation

```
Scenario: Agent A learns, Agent B doesn't
Setup:
  - Two agents share same EC/ATL but different agent_ids
  - Agent A: dragon fire_breath → pain → negative bias
  - Agent B: no experience

Assertions:
  1. Agent A's reward_bias on "fire" node is negative
  2. Agent B's reward_bias on "fire" node is zero
  3. Agent B's SensePresenceTool shows NO [DANGEROUS] annotation
```

### IT-6: Self-affordance encoding

```
Scenario: Agent's own body affordances form substrate concepts
Setup:
  - Register agent body (base_humanoid) with slash, move, use affordances
  - Encode self-entity affordances

Assertions:
  1. ATL has substrate concepts for "slash", "move", "use"
  2. NAc eligibility traces exist under agent_id
  3. Shared concepts: agent's "slash" and scene dragon's "slash" use same EC node
```

### IT-7: End-to-end sim validation (requires LLM)

```
Scenario: Full sim — dragon encounter, then mage encounter
Setup:
  - maxim --sim "test fire danger transfer" --embodiment bodies/base_humanoid
    --interactive false --sim-max-turns 30
  - Campaign: turn 1-10 dragon encounter with fire_breath pain,
    turn 11-20 mage encounter with flame_jet

Validation (from JSONL log):
  1. SEM_TRACE shows affordance concept encoding on entity registration
  2. NAc learning events show negative link for dragon_fire_breath
  3. BioEnrichment annotations show [DANGEROUS] on mage flame_jet
  4. Agent behavior: shows caution around mage (doesn't rush in)
  5. MAXIM_PROVENANCE_VERBOSITY=2 trace shows full pipeline

Note: This test requires a running LLM. Use --language-model mistral-7b
for determinism. Expected cost: $0.05-0.10.
```

## Rough-edge smoothing

### RE-1: TemporalSignature on single-percept encode path

`LinguisticEncoder.encode()` (single percept, line 201-206) does NOT pass `TemporalSignature` to `update_eligibility` — only `encode_decomposed()` does. This means percepts encoded through the standard (non-decomposed) path get eligibility traces without temporal anchors. Fix: add `temporal_sig=TemporalSignature.now()` to the single-percept encode path.

### RE-2: ToolPainBridge SCN registrations are one-way

Four `scn.register()` call sites in ToolPainBridge write temporal signatures to SCN bins but never connect to NAc temporal credit. With the generalized `TemporalCreditDistributor`, these should also emit `TemporalEvent`s so tool outcomes get temporal credit attribution.

### RE-3: Reactions have no temporal context

`Reaction` (reactions/types.py) has `scn_tag: CircadianContext | None` but `_distribute_reward_from_reaction` in bio_stack.py doesn't use it. With the generalized system, the reaction's temporal context should feed into the distributor.

### RE-4: _temporal_anchors pruning edge case

Anchors are pruned when fast-decay expires AND timestamp is older than `temporal_window_seconds` (300s default). In a short sim (< 5 minutes), this means ALL anchors survive for the entire session — the pruning never fires. This is correct behavior (everything is temporally relevant in a short session) but should be tested.

### RE-5: AffordanceDecompositionStrategy on full tool names

`SenseToolsTool._nac_annotation` receives full tool names like `rusty_sword_fire_slash` and decomposes ALL words including entity-prefix words ("rusty", "sword"). These won't match substrate concepts unless independently encoded, so it's not a correctness bug — but it's wasteful. The fix is to extract the bare affordance name from the tool before decomposition. This requires the entity→affordance name mapping, which is available from `ModulatorAffordanceTool._affordance_name`.

## Staging

### Phase 1: Integration tests (IT-1 through IT-6)

Write the substrate-level integration tests. No LLM required — pure bio-stack verification. These are the regression guards for the affordance transfer feature.

**Files:** `tests/integration/test_affordance_transfer.py` (NEW)

### Phase 2: Rough-edge smoothing (RE-1 through RE-5)

Fix the five rough edges identified above. Each is a small, targeted change.

### Phase 3: TemporalEvent protocol + TemporalCreditDistributor

Design and implement the generalized abstraction. This is the major refactor — extracting the ad-hoc `_temporal_anchors` dict and inline `distribute_reward` temporal path into a proper compartmentalized system.

**Files:**
- `time/temporal_event.py` (NEW) — `TemporalEvent` dataclass
- `decisions/temporal_credit.py` (NEW) — `TemporalCreditDistributor`
- `decisions/nac.py` — migrate `_temporal_anchors` and `distribute_reward` temporal path to distributor
- `runtime/bio_stack.py` — wire distributor into bio-stack construction

### Phase 4: Signal source migration

Migrate all 7 signal paths to emit `TemporalEvent` through the distributor. Incremental — one path at a time, each with its own test.

### Phase 5: End-to-end sim validation (IT-7)

Run full sim with LLM. Capture JSONL trace. Verify the complete pipeline from entity registration through to agent behavior.

### Phase 6: Oscillator feedback (deferred)

Wire SCN oscillator into the temporal credit system. Anticipatory credit, temporal pattern learning. Only after Phases 1-5 are stable.

## Key constraints

1. **NAc causal link system stays unchanged.** `observe()`, `record_outcome()`, `record_outcome_full()` handle event→outcome association. Temporal credit is orthogonal — it handles phase→weight, not cause→effect.
2. **TemporalCreditDistributor is a consumer of NAc, not a replacement.** It calls `nac.credit_node()` and `nac.update_eligibility()` — it doesn't bypass them.
3. **SCN integration is read+write, not just write.** The distributor both registers events (write) and queries phase similarity (read). This closes the one-way loop.
4. **Signal sources opt in.** Migrating to `TemporalEvent` emission is incremental. Each path migrates independently. Un-migrated paths continue working exactly as before.
5. **Session-scoped anchors, persisted biases.** Temporal anchors in the distributor are session-scoped (wall-clock timestamps go stale). Cross-session transfer uses persisted `reward_bias`.
6. **Cerebellum stays entity-specific.** Motor precision does not participate in temporal credit generalization.
