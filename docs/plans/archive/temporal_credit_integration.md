# Temporal Credit Integration — generalized signal-time-reward system

**Status:** Shell plan (2026-04-24), **merged with deliberative_thought_stream Stages 3/3b/4** (2026-04-24), **refined — 4-lens review complete** (2026-04-24)
**Scope:** NAc, SCN, ToolPainBridge, ReactionBus, LinguisticEncoder, WMS, ThoughtGate, bio_stack wiring
**Depends on:** [affordance_concept_transfer.md](affordance_concept_transfer.md) (shipped — first SCN-substrate closed loop), [deliberative_thought_stream.md](deliberative_thought_stream.md) (Stages 1+2 shipped — transcript + computed salience)
**Absorbs:** deliberative_thought_stream Stages 3 (goal-tagged thoughts + NAc goal-outcome learning), 3b (SCN temporal correlation), and 4 (ValenceSignal)
**Gates:** None; infrastructure generalization + goal-outcome attribution chain
**Branch:** `feat/temporal-credit-integration`

---

## Problem

Two plans converge on the same gap: **temporal credit attribution**.

### From affordance concept transfer (shipped)

The affordance transfer feature established the first closed SCN→NAc loop: temporal anchors on eligibility traces enable credit attribution after fast-decay expiry.  But this is one consumer-specific wiring in one path.

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

### From deliberative thought stream (Stages 3/3b/4)

The PFC deliberation cycle (0.8) runs multi-cycle recurrence with transcript accumulation (Stage 1, shipped) and computed salience (Stage 2, shipped).  Three gaps remain:

1. **Thoughts are disconnected from goals.**  WMS THOUGHT entries carry no goal tag.  NAc learns "tool X produces good outcomes" but never learns "thinking pattern Y under goal Z tends to produce good outcomes."
2. **Deliberation events have no temporal context.**  Thoughts, actions, and outcomes don't happen in lock-step.  Without SCN temporal indexing, there's no correlation signal between a thought at t=0 and an outcome at t=15s.
3. **No abstract signal type for salience modulation.**  Outcome valence (NAc), pain intensity (PainBus), and novelty (EC) all affect thought salience, but there's no common type for WMS entries to receive.

### The convergence

Both problems need the same infrastructure: a generalized way for any signal source to emit temporally-anchored events, and for NAc reward attribution to consider temporal similarity when crediting those events.  The deliberative plan's Stage 3b (SCN thought registration) is literally "register deliberation events with the temporal system" — exactly what the TemporalEvent protocol does generically.  Stage 3's ad-hoc `_last_deliberation_event_id` lifecycle is what the TemporalCreditDistributor replaces.  Stage 4's ValenceSignal is the output interface of temporal credit distribution.

Building them separately would mean implementing ad-hoc deliberation→NAc attribution in Stage 3, then refactoring it into the distributor weeks later.  Merging avoids that throwaway step.

## Goal

A generalized, compartmentalized **temporal credit system** where:
1. **Any signal source** (tool outcomes, pain, reactions, percepts, **deliberation events**) can emit a temporally-anchored event
2. **SCN temporal phase** is automatically captured at emission time
3. **NAc reward attribution** automatically considers temporal similarity when crediting events
4. **Goal-level reward bias** tracks whether deliberation under specific goals produces good outcomes
5. **ValenceSignal** provides a common output type so WMS entries can receive modulation from any bio-system producer
6. **The abstraction is reusable** — adding a new signal source requires implementing one protocol, not wiring three systems

## Design

### Layer 1: TemporalEvent protocol

A single envelope type that carries event metadata + temporal context:

```python
@dataclass(frozen=True)
class TemporalEvent:
    """A temporally-anchored event for credit attribution.

    Emitted by signal sources (tool outcomes, pain, reactions, percepts,
    deliberation events).  Consumed by NAc temporal credit and SCN temporal
    indexing.
    """
    event_id: str              # unique event identifier
    event_type: str            # "tool", "pain", "reaction", "percept",
                               # "affordance", "deliberation"
    event_signature: str       # hashable identifier (tool name,
                               # "deliberation:goal:<tag>", etc.)
    agent_id: str              # scoping
    temporal_sig: TemporalSignature  # SCN phase at emission time
    activation: float          # signal strength (1.0 for direct, <1.0 for
                               # indirect)
    context: dict[str, Any]    # event-specific metadata
```

Signal sources emit `TemporalEvent` instead of making ad-hoc `TemporalSignature.now()` + `SCN.register()` + `NAc.update_eligibility()` calls separately.

For deliberation events, `context` carries `{"goal": goal_tag, "cycle": cycle_num}`.  The `event_signature` is `f"deliberation:goal:{goal_tag}"` so NAc learns at the goal level.

### Layer 2: TemporalCreditDistributor

Unified distributor that **composes** NAc + SCN for temporal credit — it is NOT a bio-system, it is software plumbing.  Lives in `decisions/` (not `time/` or a bio module).  Do not add it to the bio-system naming convention (Hippocampus, NAc, SCN, ATL, etc.).

**Ownership invariant:** `_temporal_anchors` stays on NAc.  The distributor READS NAc's anchors via `nac.get_temporal_anchors(agent_id)` during `distribute()` but does NOT own them.  `LinguisticEncoder.encode_decomposed()` continues calling `nac.update_eligibility(temporal_sig=...)` directly — no circular dependency, no substrate-layer coupling to the decisions layer.  `NAc.decay_eligibility()` continues pruning its own anchors.

**Double-attribution prevention:** The distributor REPLACES `_distribute_reward_from_reaction` in `build_bio_stack` — they must NOT coexist for the same agent.  When a distributor is present, the ReactionBus subscriber calls `distributor.distribute()` instead of `nac.distribute_reward()`.  `NAc.distribute_reward()` stays as a public method (10 test callers + 5 scripts) but production reward flows go through the distributor.

```python
class TemporalCreditDistributor:
    """Centralized temporal credit attribution.

    Composes NAc + SCN for temporal-phase-aware reward distribution.
    NOT a bio-system — software wiring convenience.  NAc owns
    _temporal_anchors and _eligibility; the distributor reads them
    and adds the temporal fallback + goal credit + SCN registration
    layers on top.
    """

    def __init__(self, nac: NAc, scn: SCN, *,
                 temporal_credit_weight: float = 0.3):
        # scn is REQUIRED keyword-only.  Callers that don't want
        # temporal credit should not use a distributor — call
        # nac.distribute_reward() directly.
        self._lock = threading.RLock()  # coarse outer lock
        # Lock ordering: distributor._lock -> nac._lock (never reverse)
        ...

    def record_event(self, event: TemporalEvent) -> None:
        """Record a temporal event for future credit attribution.

        Calls nac.update_eligibility() for the fast-decay trace,
        registers with SCN for bin indexing.  For "deliberation"
        events: stored internally for goal-level attribution.
        """
        ...

    def distribute(self, agent_id: str, reward: float,
                   goal_tag: str | None = None) -> list[tuple[str, float]]:
        """Distribute reward with temporal credit fallback.

        Two-path: fast-decay traces first (via nac._eligibility),
        temporal anchors second (via nac._temporal_anchors).
        Delegates to nac.credit_node() for the actual bias update.

        When goal_tag is not None and a deliberation event with that
        tag exists, also calls nac.credit_goal(goal_tag, reward).
        Guards: goal_tag=None is a no-op for goal credit (no phantom
        None key in _goal_reward_bias).
        """
        ...

    def cleanup_session(self) -> None:
        """Unregister session-scoped events from SCN.

        Called from BioStack.on_session_end() — single centralized
        call site, same pattern as save_cerebellum().  Idempotent.
        """
        ...
```

Three public methods: `record_event`, `distribute`, `cleanup_session`.  Tight API.

~~`get_pending_deliberation_event()`~~ — DROPPED.  Deliberation event tracking for the ad-hoc `_last_deliberation_event_id` pattern is handled by a local variable in the agent loop (5 LOC, see Phase 3 staging notes).  The distributor should not own consumer-specific bookkeeping.

~~`query_temporal_patterns()`~~ — DROPPED.  Zero callers in any phase.  YAGNI.  Add it when Phase 8 (oscillator feedback) creates a consumer.

**Key architectural decisions:**

1. **SCN is required** on the constructor (keyword-only).  The distributor's primary value IS temporal credit — making SCN optional degrades it to a passthrough wrapper with no warning (silent no-op invariant violation).

2. **Thread safety:** The distributor has its own `threading.RLock`.  `distribute()` acquires it around the full sequence (read anchors → credit nodes → credit goal → emit ValenceSignal).  NAc's internal methods acquire their own lock, so lock ordering is distributor → NAc (never reverse).

3. **`cleanup_session()` is called from `BioStack.on_session_end()`** — a new method that centralizes all session-end cleanup (cerebellum save, distributor cleanup, etc.).  This prevents the N-sites-forget-to-call bug class.

### Layer 3: Goal-tagged thoughts + NAc goal-level bias

This is the goal-specific content from deliberative_thought_stream Stage 3.  The **attribution mechanism** flows through the distributor (Layer 2), but the **goal-level storage** lives on NAc directly:

**WMEntry changes:**
- Add `goal_tag: str | None = None` to `WMEntry` (`frozen=True, slots=True`).  The only construction site is `WorkingMemorySet.add()`, which gets a new `goal_tag` kwarg.
- `wms.add(THOUGHT, ..., goal_tag=active_goal)` in the deliberation cycle

**NAc changes — new `_goal_reward_bias` dict:**

The existing `_reward_bias` is keyed by `(agent_id, node_id)` for substrate node recognition modulation.  Goal-level bias is a **different concept** — it modulates ThoughtGate threshold based on whether deliberation under a goal type historically produces good outcomes.  These need separate storage:

- New `_goal_reward_bias: dict[str, float] = {}` on NAc (keyed by goal string, not `(agent_id, node_id)`)
- `credit_goal(goal_tag: str, reward: float)` — analogous to `credit_node` but for goal-level bias.  Called by `TemporalCreditDistributor.distribute()` when `goal_tag` is provided.
- `get_goal_reward_bias(goal_tag: str) -> float` — returns bias for ThoughtGate modulation
- `decay_goal_reward_biases()` — called alongside existing `decay_reward_biases()` in the per-tick decay cycle
- All three methods acquire `self._lock` (RLock)
- **Serialization:** add `goal_reward_bias` to `dump()`/`load_state()` alongside existing `reward_bias`.  Same `{goal_string: float}` format.  Old snapshots missing the field: `state.get("goal_reward_bias", {})`.

**Goal bias range: `[-max_goal_reward_bias, +max_goal_reward_bias]` (NOT clamped to 0).**

The existing `_reward_bias` for substrate nodes clamps to `[0, max_reward_bias]` — it only widens EC recognition, never narrows.  Goal-level bias is **fundamentally different**: it must support BOTH facilitation (positive → lower ThoughtGate threshold → deliberate more) AND suppression (negative → raise threshold → act faster, skip deliberation).

Bio-plausible framing: the basal ganglia have two anatomically separate pathways — the direct pathway (D1 receptors, "go") facilitates actions, and the indirect pathway (D2 receptors, "no-go") suppresses them.  Clamping negative bias to 0 implements only the direct pathway.  The agent can learn "deliberation under goal X helps" but cannot learn "deliberation under goal Y wastes time — just act."  This is half a learning system.

**Implementation:** `credit_goal` clamps to `[-max_goal_reward_bias, +max_goal_reward_bias]` where `max_goal_reward_bias` defaults to `max_reward_bias` (same scale).  `get_goal_reward_bias(None)` returns 0.0 (guard against None goal tags).  `credit_goal(None, reward)` is a no-op (early return, no phantom None key).

**`goal_tag=None` guards (Critical — F2):**
- `credit_goal(None, ...)` → early return, no-op
- `get_goal_reward_bias(None)` → return 0.0
- `distributor.distribute(..., goal_tag=None)` → skips goal credit path entirely
- Without these guards, every goalless turn accumulates bias under a `None` key, and `get_goal_reward_bias(None)` returns a real float that modulates ThoughtGate — the agent thinks more when it has NO goal, which is backwards.

**ThoughtGate changes:**
- `should_think()` gains a `goal_reward_bias: float = 0.0` parameter
- The bias modulates the adaptive threshold: `threshold = clamp(threshold - goal_reward_bias, self._config.min_combined_score, max_threshold)`
- Positive bias → lower threshold → deliberation fires more readily (direct pathway, "go")
- Negative bias → higher threshold → deliberation suppressed (indirect pathway, "no-go")
- The caller (agent loop) passes `nac.get_goal_reward_bias(active_goal)` when NAc is available

**Note on goal_depth_integration absorption:** This plan adds `goal_tag` to THOUGHT entries — the goal is *metadata on thoughts*, not a first-class WMS entry.  [goal_depth_integration.md](../deferred/goal_depth_integration.md) Stage 1 adds `GOAL` to `WorkingMemoryKind` so the goal itself appears in WMS.  These are complementary.  goal_depth Stage 3 (NAc goal-outcome learning) is properly absorbed here.  Stages 1 (GOAL kind), 2 (episode goal tagging), and 4 (goal persistence) remain independent.

### Layer 4: ValenceSignal — abstract output type

The distributor's output side.  When credit is distributed, the result is expressed as a `ValenceSignal` that any WMS entry can receive.

**Naming clarification — three valence concepts coexist:**

1. `Valence` enum in `decisions/causal_link.py`: POSITIVE/NEGATIVE/NEUTRAL/UNKNOWN.  Discrete, on CausalLink.
2. `Episode.valence` float in `memory/episode.py`: Net valence from captured reactions.  Continuous.
3. `ValenceSignal` frozen dataclass (this layer): A **transport type** carrying a float value from any producer to WMS consumers.

```python
@dataclass(frozen=True, slots=True)
class ValenceSignal:
    """Abstract reward/punishment signal from any bio-system.

    Produced by NAc (outcome valence), PainBus (negative),
    EC (novelty as mild positive).  Consumed by WMS entries
    to modulate thought salience over time.

    NOT a replacement for the Valence enum (causal_link.py) or
    Episode.valence float (episode.py). Those are storage types;
    this is a transport type.
    """
    value: float          # -1.0 to +1.0, PRE-NORMALIZED by the producer
    source: str           # "nac", "pain", "ec_novelty", "temporal_credit"
                          # For observability/logging ONLY — consumers
                          # must NOT branch on this field (see below)
    goal_tag: str | None  # which goal was active when this fired
    timestamp: float
```

**Consumer source-blindness invariant:** Consumers (WMS, `top_by_salience()`) use `value` only.  They do NOT branch on `source`.  The `source` field exists for observability: bio_telemetry JSONL, sim_log traces, debugging.  Bio-region-specific weighting belongs at the **producer** side — each producer emits an appropriately-scaled `value` that already encodes its relative importance:

- PainBus emits `value=-intensity` (0.0 to -1.0, magnitude reflects threat level)
- NAc emits `value=±1.0` (strong discrete signal, reward prediction error)
- EC novelty emits `value=+0.3` (mild exploratory boost, already attenuated)

If a future bio-system needs to RECEIVE valence differently (e.g., hippocampus weighs pain higher for episodic encoding), it subscribes to PainBus/ReactionBus directly — those are the source-aware typed dispatch surfaces.  ValenceSignal is the already-normalized output downstream of that dispatch.  Adding a new producer means emitting an appropriately-scaled `value` — no consumer edits.

**Wiring:**
- `TemporalCreditDistributor.distribute()` → emits `ValenceSignal(value=reward_fraction, source="temporal_credit", goal_tag=...)`
- NAc outcome path: after `record_outcome`, emit `ValenceSignal(value=±1.0, source="nac", goal_tag=active_goal)`
- PainBus: `PainSignal` → `ValenceSignal(value=-intensity, source="pain", goal_tag=active_goal)`
- EC novelty: on pattern separation → `ValenceSignal(value=+0.3, source="ec_novelty", goal_tag=active_goal)`

**WMS consumption:**
- Valence signals stored on `WorkingMemorySet` in a side-dict keyed by `WMEntry.tick` (monotonic, always unique, never None).  NOT keyed by `ref` — THOUGHT entries have `ref=None` and are the primary valence target.
- **Accumulation uses exponential moving average, not unbounded list.** `_valence_ema: dict[int, float]` stores a single running aggregate per tick: `ema = 0.9 * ema + 0.1 * signal.value`.  This is O(1) memory, naturally bounded in `[-1, +1]`, and avoids the unbounded list problem where a long-lived entry accumulates thousands of signals.
- When `add()` evicts an entry from the bounded deque (capacity=64), also prune `_valence_ema` for the evicted tick.  The evicted tick is `self._buf[0].tick` before append when `len(self._buf) == self._capacity`.
- `top_by_salience()` incorporates the EMA: `effective_salience = base_salience + valence_ema`.  Since EMA is bounded `[-1, +1]`, effective salience stays in a reasonable range.
- Negative valence from pain **increases** `abs(effective_salience)` (survival — you want to REMEMBER what hurt you).  Implementation: `effective_salience = base_salience + abs(valence_ema)` when `valence_ema < 0`.
- Positive valence from NAc **increases** salience (reinforcement — thoughts that led to success are more valuable)
- **Phase 3 prerequisite:** THOUGHT entries must have non-None `ref` for cross-referencing (e.g., `ref=f"thought:{uuid4().hex[:8]}"`).  Add ref generation in Phase 3 alongside `goal_tag`.

### Layer 5: Signal source migration

Migrate each signal path to emit `TemporalEvent`:

| Path | Current | After |
|------|---------|-------|
| Tool outcome | `nac.observe()` + `scn.register()` | `distributor.record_event(TemporalEvent(...))` + `nac.observe()` |
| Embodiment failure | `nac.record_outcome()` + `scn.register()` | `distributor.record_event(...)` + `nac.record_outcome()` |
| Pain signal | `nac.record_outcome_full()` + `scn.register()` | `distributor.record_event(...)` + `nac.record_outcome_full()` |
| Percept encoding | `nac.update_eligibility(..., temporal_sig)` | `distributor.record_event(...)` |
| Reaction | `nac.distribute_reward()` (no temporal) | `distributor.distribute()` (with temporal) |
| **Deliberation** | *(not wired)* | `distributor.record_event(TemporalEvent(event_type="deliberation", ...))` |

**Key constraint:** NAc's causal link system (`observe`, `record_outcome`, `record_outcome_full`) stays unchanged — it handles event→outcome association.  The temporal credit system is **orthogonal** — it handles temporal phase → credit weighting.  They coexist, not merge.

### Layer 6: Oscillator feedback (future)

Once Layers 1-5 are stable:
- SCN oscillator `observe(signature)` called on each `TemporalEvent`
- Coupling weights learn co-occurrence patterns (Hebbian on Kuramoto phases)
- `predict_next_occurrence(event_signature)` becomes actionable
- Anticipatory credit: pre-activate eligibility traces for events predicted by the oscillator

This is deferred until Layers 1-5 prove stable in production.

## Deliberation cycle integration — how the flow works

**Phase 3 (pre-distributor):**

```
Goal active: "escape the dungeon"
    │
    ▼
Deliberation cycle starts:
  _last_deliberation_event_id = uuid4().hex  (local variable)
  wms.add(THOUGHT, salience=0.7, goal_tag="escape the dungeon",
          ref=f"thought:{uuid4().hex[:8]}")
    │
    ▼
Action: sneak_past_guard → success
  nac.credit_goal("escape the dungeon", +1.0)
  _last_deliberation_event_id = None  (consumed)
    │
    ▼
NAc now has:
  tool:sneak_past_guard → positive  (existing causal link path)
  _goal_reward_bias["escape the dungeon"] > 0  (NEW — goal path)
    │
    ▼
Next time goal "escape" is active:
  nac.get_goal_reward_bias("escape the dungeon") > 0
  → ThoughtGate threshold lowered → deliberation fires more readily
```

**Phase 4+ (with distributor) — final-state architecture:**

```
Goal active: "escape the dungeon"
    │
    ▼
Deliberation cycle starts:
  distributor.record_event(TemporalEvent(
      event_id=uuid4(),
      event_type="deliberation",
      event_signature="deliberation:goal:escape the dungeon",
      agent_id="sim_aut",
      temporal_sig=TemporalSignature.now(),
      activation=1.0,
      context={"goal": "escape the dungeon", "cycle": 1}
  ))
  → calls nac.update_eligibility() for fast-decay trace
  → registers with SCN for bin indexing

  wms.add(THOUGHT, salience=0.7, goal_tag="escape the dungeon",
          ref=f"thought:{uuid4().hex[:8]}")
    │
    ▼
Action: sneak_past_guard → success
  distributor.distribute("sim_aut", reward=+1.0,
                         goal_tag="escape the dungeon")
  → fast-decay path: credits substrate nodes with active traces
  → phase-similarity fallback: credits nodes whose anchors match
  → goal credit: nac.credit_goal("escape the dungeon", +1.0)
  → emits ValenceSignal(value=+1.0, source="temporal_credit",
                         goal_tag="escape the dungeon")
    │
    ▼
NAc now has:
  tool:sneak_past_guard → positive  (existing causal link path)
  _reward_bias[(agent, fire_node)] adjusted  (substrate path)
  _goal_reward_bias["escape the dungeon"] > 0  (NEW — goal path)
    │
    ▼
Next time goal "escape" is active:
  nac.get_goal_reward_bias("escape the dungeon") > 0
  → ThoughtGate threshold lowered → deliberation fires more readily
  Bio-enrichment queries NAc → "deliberation under escape-type goals
  tends to produce positive outcomes" surfaces as prediction
```

## Integration testing for affordance transfer (shipped feature)

### IT-1: Substrate-level transfer verification

Unit-level tests exist (shipped with Stages 0-3).  These integration tests verify the full pipeline end-to-end without mocking.

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

### IT-7: Goal-level credit attribution (deliberation + temporal)

```
Scenario: Deliberation under goal A produces good outcomes, goal B bad
Setup:
  - Build full bio-stack + TemporalCreditDistributor + ThoughtGate
  - Turn 1: goal="escape", deliberation fires, action succeeds
    → distributor.distribute(reward=+1.0, goal_tag="escape")
  - Turn 2: goal="negotiate", deliberation fires, action fails
    → distributor.distribute(reward=-1.0, goal_tag="negotiate")
  - Turn 3: goal="escape" again

Assertions:
  1. nac._goal_reward_bias["escape"] > 0 (positive → "go" direct pathway)
  2. nac._goal_reward_bias["negotiate"] < 0 (negative → "no-go" indirect pathway)
  3. ThoughtGate threshold is LOWER for "escape" (deliberate more) and HIGHER for "negotiate" (skip deliberation)
  4. get_goal_reward_bias(None) returns 0.0 (no phantom None key)
  5. credit_goal(None, 1.0) is a no-op (no entry created)
```

### IT-8: End-to-end sim validation (requires LLM)

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

`LinguisticEncoder.encode()` (single percept, line 201-206) does NOT pass `TemporalSignature` to `update_eligibility` — only `encode_decomposed()` does.  This means percepts encoded through the standard (non-decomposed) path get eligibility traces without temporal anchors.  Fix: add `temporal_sig=TemporalSignature.now()` to the single-percept encode path.

### RE-2: ToolPainBridge SCN registrations are one-way

Four `scn.register()` call sites in ToolPainBridge write temporal signatures to SCN bins but never connect to NAc temporal credit.  With the generalized `TemporalCreditDistributor`, these should also emit `TemporalEvent`s so tool outcomes get temporal credit attribution.

### RE-3: Reactions have no temporal context

`Reaction` (reactions/types.py) has `scn_tag: CircadianContext | None` but `_distribute_reward_from_reaction` in bio_stack.py doesn't use it.  With the generalized system, the reaction's temporal context should feed into the distributor.

### RE-4: _temporal_anchors pruning edge case

Anchors are pruned when fast-decay expires AND timestamp is older than `temporal_window_seconds` (300s default).  In a short sim (< 5 minutes), this means ALL anchors survive for the entire session — the pruning never fires.  This is correct behavior (everything is temporally relevant in a short session) but should be tested.

### RE-5: AffordanceDecompositionStrategy on full tool names

`SenseToolsTool._nac_annotation` receives full tool names like `rusty_sword_fire_slash` and decomposes ALL words including entity-prefix words ("rusty", "sword").  These won't match substrate concepts unless independently encoded, so it's not a correctness bug — but it's wasteful.  The fix is to extract the bare affordance name from the tool before decomposition.  This requires the entity→affordance name mapping, which is available from `ModulatorAffordanceTool._affordance_name`.

## Staging

**Phase reorder rationale:** The original plan ordered distributor (infrastructure) before goals (feature).  Four-lens review cross-confirmed that Phase 3 (goals) is self-contained, immediately testable in a sim, and does NOT depend on the distributor.  The throwaway cost of a local `_last_deliberation_event_id` variable is ~5 LOC.  Shipping the feature first means users get goal-level learning sooner and the distributor can be validated against real goal-attribution data.

### Phase 1: Integration tests (IT-1 through IT-6)

Write the substrate-level integration tests.  No LLM required — pure bio-stack verification.  These are the regression guards for the affordance transfer feature.

**Files:** `tests/integration/test_affordance_transfer.py` (NEW)

### Phase 2: Rough-edge smoothing (RE-1 through RE-5)

Fix the five rough edges identified above.  Each is a small, targeted change.

### Phase 3: Goal-tagged thoughts + goal-level NAc bias (absorbed from deliberative S3+S3b)

The user-visible feature.  Ships BEFORE the distributor — does not depend on it.

**Files:**

| File | Change | LOC |
|------|--------|-----|
| `agents/working_memory.py` | Add `goal_tag: str | None` to `WMEntry`, thread through `add()`.  Add `ref=f"thought:{uuid4().hex[:8]}"` generation for THOUGHT entries (Phase 5 prerequisite). | +10 |
| `decisions/nac.py` | `_goal_reward_bias` dict, `credit_goal()` (with None guard), `get_goal_reward_bias()` (None→0.0), `decay_goal_reward_biases()`, serialization in `dump()`/`load_state()`.  Range: `[-max_goal_reward_bias, +max_goal_reward_bias]`. | +45 |
| `runtime/thought_gate.py` | `goal_reward_bias` parameter on `should_think()`, bidirectional threshold modulation (positive lowers, negative raises) | +10 |
| `runtime/agent_loop.py` | Deliberation cycle stores `_last_deliberation_event_id` (local variable, ~5 LOC); tags WMS entries with `goal_tag=active_goal`; passes `goal_reward_bias` to ThoughtGate; outcome attribution calls `nac.credit_goal(active_goal, reward)` | +35 |

**Ad-hoc `_last_deliberation_event_id`:** Stored as a local variable in the agent loop's turn function.  Set at deliberation cycle end, read at outcome attribution, cleared after use.  Phase 4 replaces this with the distributor's lifecycle management — the ~5 LOC are throwaway.

### Phase 4: TemporalEvent protocol + TemporalCreditDistributor

The infrastructure refactor.  Ships AFTER Phase 3 goals are validated against real sim data.

**Files:**
- `time/temporal_event.py` (NEW) — `TemporalEvent` frozen dataclass
- `decisions/temporal_credit.py` (NEW) — `TemporalCreditDistributor` (3 public methods: `record_event`, `distribute`, `cleanup_session`)
- `decisions/nac.py` — add `get_temporal_anchors(agent_id)` public accessor (distributor reads, does NOT own `_temporal_anchors`)
- `runtime/bio_stack.py` — add `distributor: TemporalCreditDistributor | None` field to `BioStack` dataclass; construct in `build_bio_stack`; rewrite `_distribute_reward_from_reaction` to call `distributor.distribute()` instead of `nac.distribute_reward()` when distributor is present; add `on_session_end()` method that calls `cleanup_session()` + cerebellum save
- `runtime/agent_loop.py` — replace ad-hoc `_last_deliberation_event_id` with `distributor.record_event(TemporalEvent(event_type="deliberation", ...))` + `distributor.distribute(..., goal_tag=active_goal)` at outcome time

**Migration strategy for `NAc.distribute_reward()`:** Keep it as a public method (10 test callers + 5 scripts).  The distributor WRAPS it — calls `nac.distribute_reward()` internally for the fast-decay path, then adds the phase-similarity fallback + goal credit on top.  `NAc.distribute_reward()` keeps its existing temporal fallback code (behavioral equivalence for direct callers).  The distributor adds value by also registering with SCN and handling goal credit.  No tests break.

**Test migration:** 6 tests in `TestTemporalEligibility` assert against `nac._temporal_anchors` — these stay as-is (anchors remain on NAc).  New `TestTemporalCreditDistributor` tests the distributor's compose-on-top behavior.  Write new tests first (TDD), existing tests are unmodified.

### Phase 5: ValenceSignal as distributor output (absorbed from deliberative S4)

The output interface.  Distributor distributes credit and emits ValenceSignals; WMS entries receive them and modulate effective salience.

**Files:**

| File | Change | LOC |
|------|--------|-----|
| `decisions/valence_signal.py` (NEW) | `ValenceSignal` frozen dataclass | +25 |
| `agents/working_memory.py` | `receive_valence(tick, signal)` keyed by tick (not ref) + `_valence_ema: dict[int, float]` EMA accumulator + prune on eviction + incorporate into `top_by_salience()` | +25 |
| `decisions/temporal_credit.py` | `distribute()` emits `ValenceSignal` after credit distribution | +10 |
| `runtime/agent_loop.py` | Wire ValenceSignal reception on WMS entries after outcome attribution | +10 |

### Phase 6: Signal source migration

Migrate the remaining signal paths (tool outcomes, pain, reactions, percepts) to emit `TemporalEvent` through the distributor.  Incremental — one path at a time, each with its own test.  `_distribute_reward_from_reaction` is already migrated in Phase 4.

### Phase 7: End-to-end validation (IT-7 + IT-8)

Run IT-7 (goal-level credit, no LLM) and IT-8 (full sim with LLM).  Capture JSONL trace.  Verify the complete pipeline from entity registration through deliberation attribution to agent behavior.

### Phase 8: Oscillator feedback (deferred)

Wire SCN oscillator into the temporal credit system.  Anticipatory credit, temporal pattern learning.  Only after Phases 1-7 are stable.

## Key constraints

1. **NAc causal link system stays unchanged.** `observe()`, `record_outcome()`, `record_outcome_full()` handle event→outcome association.  Temporal credit is orthogonal — it handles phase→weight, not cause→effect.

2. **TemporalCreditDistributor is a consumer of NAc, not a replacement.** It calls `nac.credit_node()`, `nac.credit_goal()`, and `nac.update_eligibility()` — it doesn't bypass them.

3. **SCN phase similarity is a cross-session generalization signal, NOT within-session precision.**  `TemporalSignature.similarity()` computes circadian/weekly/monthly/annual phase distance.  Two events 15 seconds apart have similarity ~0.999 on ALL phase dimensions — zero discriminative power within a session.  For within-session credit, the fast-decay eligibility traces are the correct mechanism.  The "temporal fallback" path is more accurately called **phase-similarity credit** — it fires when fast traces have decayed AND the phase matches, which means "this event type tends to happen at this time of day/week across sessions."  The SCN registers events (write) and the distributor reads phase similarity (read).  Do NOT rely on SCN for sub-minute correlation — that's the fast-decay traces' job.

4. **Signal sources opt in.** Migrating to `TemporalEvent` emission is incremental.  Each path migrates independently.  Un-migrated paths continue working exactly as before.

5. **Session-scoped anchors, persisted biases.** Temporal anchors in the distributor are session-scoped (wall-clock timestamps go stale).  Cross-session transfer uses persisted `reward_bias` and `goal_reward_bias`.

6. **Cerebellum stays entity-specific.** Motor precision does not participate in temporal credit generalization.

7. **`_goal_reward_bias` is separate from `_reward_bias`.** The existing `_reward_bias` dict is keyed by `(agent_id, node_id)` and modulates EC substrate recognition thresholds.  Goal-level bias is keyed by goal string and modulates ThoughtGate deliberation threshold.  Different keys, different consumers, different semantics.  Do NOT merge them.

8. **Negative valence = higher salience, not lower.** Pain thoughts are salient (survival).  The intuition "negative = drop it" is wrong for bio-plausibility.  Negative valence increases salience (you REMEMBER what hurt you); it's the *goal-level reward bias* that determines whether the agent *pursues* or *avoids* the associated goal.  Unlike `_reward_bias` (clamped to `[0, max]`), `_goal_reward_bias` allows negative values — negative bias raises ThoughtGate threshold (indirect pathway suppression: "don't bother deliberating, just act").

9. **Goal tag is the active goal string, not a goal ID.** Goals don't have stable IDs yet ([goal_depth_integration.md](../deferred/goal_depth_integration.md) Stage 4).  Using the description string is lossy (paraphrase collapse) but sufficient for the PoC.  When goal IDs land, swap the tag type.  Do NOT add normalization heuristics — they break on goals where casing or stopwords are semantically meaningful.  `decay_goal_reward_biases()` cleans up stale entries over time.

10. **NAc `_goal_reward_bias` must be serialized.** Add to `dump()` as `"goal_reward_bias": self._goal_reward_bias` and to `load_state()` as `self._goal_reward_bias = state.get("goal_reward_bias", {})`.  Old snapshots missing the field silently start fresh.

11. **Deliberation event lifecycle has two stages.** Phase 3 (pre-distributor): a local variable `_last_deliberation_event_id` in the agent loop's turn function tracks the pending event.  Set at deliberation cycle end, read at outcome attribution, cleared after use.  ~5 LOC throwaway.  Phase 4 (post-distributor): the distributor stores deliberation events internally; the agent loop calls `distributor.record_event(...)` at cycle start and `distributor.distribute(..., goal_tag=active_goal)` at outcome time.  In both stages, no field on `StructuredContext` — the lifecycle is owned locally, not threaded through the context.

12. **SCN thought registration uses LOW significance (0.1) and cleans up at session end.** The distributor's `record_event()` registers with SCN at significance=0.1 for `"deliberation"` events (thoughts lose eviction battles to real memories).  `cleanup_session()` unregisters all session-scoped event_ids.  The temporal correlation only needs them present for the ~36-second `query_similar_time` window.

13. **No new bus.** PainBus and ReactionBus already exist.  ValenceSignal is a type, not a transport — it flows through existing paths (distributor output, PainBus subscriber, direct WMS injection).

14. **Deliberation decay race is by design, not a bug.** `decay_eligibility(factor=0.9)` runs every tick.  A trace of strength 1.0 reaches the 0.01 prune threshold after ~44 ticks (1.5-22 seconds at 2-30Hz).  A multi-cycle deliberation takes 5-15 seconds.  So: `record_event()` fires at cycle start, the trace decays during the LLM calls, and `distribute()` fires after the action outcome.  The fast-decay trace may be gone; credit falls through to the phase-similarity fallback at 0.3x weight.  **This is correct:** the causal link between "I thought about X" and "X succeeded" IS weaker than "I used tool X" and "X succeeded."  The deliberation→outcome relationship is indirect (thought → action selection → tool execution → outcome), so attenuated credit is the right signal.  Do NOT add special decay rates for deliberation events to keep traces artificially alive.

15. **TemporalCreditDistributor is NOT a bio-system.** It has no neural correlate.  It lives in `decisions/` as software plumbing that composes NAc + SCN.  Do not name it with bio-system conventions (no class rename to match Hippocampus/NAc/SCN/ATL naming).  The closest neural analog would be VTA (ventral tegmental area) dopaminergic projections, but VTA is a source, not a coordinator.  If this ever evolves into a bio-system, VTA is the correct name.

## Relationship to existing plans

- **Absorbs** [deliberative_thought_stream.md](deliberative_thought_stream.md) Stages 3 (goal-tagged thoughts + NAc goal-outcome learning), 3b (SCN temporal correlation), and 4 (ValenceSignal).  Stages 1 (transcript) and 2 (computed salience) are already shipped and untouched.
- **Partially absorbs** [goal_depth_integration.md](../deferred/goal_depth_integration.md) Stage 3 (NAc goal-outcome learning).  Stages 1 (GOAL WMS kind), 2 (episode goal tagging), and 4 (goal persistence) remain independent follow-ons.
- **Extends** [affordance_concept_transfer.md](affordance_concept_transfer.md) — generalizes the shipped `_temporal_anchors` + `distribute_reward` temporal fallback path into a reusable distributor.
- **Enables** ThoughtGate adaptive threshold via NAc goal bias — currently ThoughtGate has no NAc input.  Phase 4 creates the signal.
- **Enables** cross-session goal learning — once NAc has `_goal_reward_bias`, it persists across sessions via existing NAc serialization.

## Validation

1. **Phase 1:** IT-1 through IT-6 pass.  Pure substrate-level verification.
2. **Phase 3:** Run a multi-turn sim where the same goal type recurs.  Confirm NAc has `_goal_reward_bias` entries (positive for good goals, negative for bad goals).  Confirm ThoughtGate fires more readily for positive-bias goals and less readily for negative-bias goals.  Confirm `credit_goal(None, ...)` is a no-op.
3. **Phase 4:** `distribute_reward()` temporal path now routes through distributor.  Existing affordance transfer tests still pass (behavioral equivalence).  `_distribute_reward_from_reaction` calls distributor, not NAc directly.
4. **Phase 5:** Confirm PainBus-originated negative valence increases thought salience (not decreases).  Confirm ValenceSignal EMA is bounded.  Confirm evicted entries don't leak in `_valence_ema`.
5. **Phase 7:** IT-7 (goal-level credit without LLM) + IT-8 (full sim with LLM) pass.

## Pre-implementation review findings

### Round 1: Deliberative thought stream 5-lens review (inherited, still applicable)

| ID | Finding | Severity | Resolution |
|----|---------|----------|------------|
| C1 | WMS `add()` with `goal_tag` is thread-safe (lock serializes) | Info | No action |
| C3 | SCN has no internal lock; thought registration races with capture thread on shared bins | **Important** | GIL-safe today; document as concurrency risk for free-threaded Python.  Distributor's `record_event()` inherits this — defer SCN lock to future audit |
| C4 | `_goal_reward_bias` reads/writes must be under NAc's RLock | Info | Natural — new methods acquire `self._lock` |
| M3 | NAc `dump()/load_state()` needs `_goal_reward_bias` serialization | **Important** | Phase 3 adds it.  `state.get("goal_reward_bias", {})` for backward compat.  Downgrade silently drops the field (acceptable — biases decay anyway) |
| K2 | `_goal_reward_bias` grows by one entry per unique goal string; paraphrase proliferation | **Important** | Accept for PoC.  `decay_goal_reward_biases()` cleans up.  goal_depth Stage 4 introduces stable goal IDs |

### Round 2: Four-lens review (2026-04-24) — folded into design

**Lens 1: Composability + API Surface**

| ID | Finding | Severity | Folded? |
|----|---------|----------|---------|
| CS-1 | Distributor was a god object: owned anchors + deliberation lifecycle + credit distribution + SCN registration + temporal patterns | **Important** | **YES.** Trimmed to 3 methods. Anchor ownership stays on NAc. `get_pending_deliberation_event()` dropped (agent loop owns lifecycle). `query_temporal_patterns()` dropped (YAGNI). |
| CS-2 | `distribute_reward()` migration is NOT incremental — 1 production caller, 10 test callers, 5 scripts | **Important** | **YES.** NAc.distribute_reward() stays as public method. Distributor wraps it, doesn't replace. Tests unmodified. |
| CS-3 | Moving `_temporal_anchors` out of NAc creates circular dep with LinguisticEncoder | **Critical** | **YES.** Anchors stay on NAc. Distributor reads via `get_temporal_anchors()`. No substrate-layer coupling. |
| CS-5 | `cleanup_session()` creates hidden lifecycle contract — N sites forget to call it | **Important** | **YES.** Called from `BioStack.on_session_end()` — single centralized site. |
| CS-7 | ValenceSignal keyed by `WMEntry.ref` breaks for `ref=None` THOUGHT entries + eviction leak | **Critical** | **YES.** Keyed by tick (monotonic, never None). EMA accumulator instead of unbounded list. Prune on eviction. |
| CS-8 | Distributor needs its own lock — can't piggyback on NAc's | **Important** | **YES.** Own `threading.RLock`. Lock ordering: distributor → NAc (documented). |

**Lens 2: Failure Modes + Silent Bugs**

| ID | Finding | Severity | Folded? |
|----|---------|----------|---------|
| F2 | `credit_goal(None, reward)` creates phantom None key in `_goal_reward_bias` — agent deliberates more when it has NO goal | **Critical** | **YES.** Three guards: `credit_goal(None)` → no-op, `get_goal_reward_bias(None)` → 0.0, `distribute(goal_tag=None)` → skip goal credit. |
| F3 | State desync if `_temporal_anchors` exists on both NAc and distributor | **Critical** | **YES.** Single-writer: NAc owns anchors, distributor reads only. |
| F4 | Decay race: 44 ticks to prune, deliberation takes 5-15s | **Important** | **YES.** Documented as by-design (constraint 14). Attenuated credit is the correct signal for indirect deliberation→outcome link. |
| F5 | `cleanup_session()` never called on crash — stale IDs accumulate | Info | Bounded and self-healing via eviction. Significance 0.1 loses to real memories. Try/finally in session-end path. |
| F6 | ValenceSignal accumulation without bound — 2000+ signals on long-lived entries | **Important** | **YES.** Replaced list with EMA: `ema = 0.9 * ema + 0.1 * signal.value`. O(1), bounded `[-1, +1]`. |
| F7 | Double-attribution: `_distribute_reward_from_reaction` AND distributor credit same nodes | **Critical** | **YES.** Distributor REPLACES the reaction subscriber — they must not coexist. |
| F9 | `TemporalSignature.similarity()` provides zero within-session discrimination (15s apart → sim ~0.999) | **Important** | **YES.** Constraint 3 rewritten: phase-similarity is cross-session generalization, not within-session precision. Fast-decay traces handle sub-minute. |
| F11 | `should_think()` has default `goal_reward_bias=0.0` — callers can silently forget | **Important** | Accepted: single production caller, documented. Structural enforcement (ThoughtGate holds NAc ref) considered but rejected as over-coupling for one caller. |

**Lens 3: Bio-Plausibility Integrity**

| ID | Finding | Bio-Plausibility | Folded? |
|----|---------|-----------------|---------|
| B1 | TemporalCreditDistributor has no neural correlate | Stretch | **YES.** Explicitly documented as non-bio-system plumbing (constraint 15). VTA noted as future analog. |
| B2 | "Temporal credit" is a misnomer — fast-decay IS temporal credit, the fallback is phase-similarity | Incorrect naming | Addressed in constraint 3 rewrite. The name `temporal_credit_weight` stays for code compatibility but the plan calls it "phase-similarity credit." |
| B3 | `_goal_reward_bias` is a lookup table, not TD learning | Reasonable compromise | Accepted. Consistent with existing `_reward_bias`. TD refactor would change both together in a future plan. |
| B5 | SCN for 36-second deliberation is a biological category error — SCN is circadian, not sub-minute | Incorrect | **YES.** Constraint 3 rewritten. SCN registration of deliberation events is harmless (populates bins) but NOT relied upon for within-session correlation. |
| B6 | ThoughtGate + goal_reward_bias is accurate PFC-dopamine gating analog | Accurate | Kept as-is. Good mapping. |
| B7 | Clamping negative goal bias to 0 = missing indirect pathway = no avoidance learning | Gap (half a learning system) | **YES.** `_goal_reward_bias` now allows `[-max, +max]`. Negative bias raises ThoughtGate threshold (indirect pathway "no-go" suppression). This is the review's most consequential fix. |

**Lens 4: Incremental Adoption + Testability**

| ID | Finding | Severity | Folded? |
|----|---------|----------|---------|
| F3/F7 | Phase 4 (goals) CAN and SHOULD ship before Phase 3 (distributor) — ~5 LOC throwaway | **Important** | **YES.** Phases reordered: P3=goals, P4=distributor. |
| F1 | `NAc.distribute_reward()` extraction strategy underspecified | **Critical** | **YES.** Explicit: distributor wraps NAc.distribute_reward(), doesn't replace. NAc keeps temporal fallback for direct callers. |
| F8 | `_distribute_reward_from_reaction` must migrate in Phase 4, not Phase 6 — it's the single production caller | **Critical** | **YES.** Phase 4 scope explicitly includes bio_stack.py rewrite. |
| F9 | BioStack needs `distributor` field | **Important** | **YES.** Phase 4 adds `distributor: TemporalCreditDistributor | None` to BioStack. |
| F11 | `decay_eligibility` prunes `_temporal_anchors` as a side effect — coupling to distributor | **Important** | Resolved by keeping anchors on NAc. `decay_eligibility` keeps its pruning lines unchanged. |
| F13 | THOUGHT entries need non-None `ref` for ValenceSignal delivery | **Important** | **YES.** Phase 3 adds `ref=f"thought:{uuid4().hex[:8]}"` alongside `goal_tag`. |
| F5 | 6 temporal tests assert against `nac._temporal_anchors` — anchors stay on NAc so tests are unmodified | Info | Confirmed: no test breakage from Phase 4 (anchors don't move). New distributor tests are additive. |

## Open questions (resolve during implementation)

### OQ-1: Reaction path has no goal context

`_distribute_reward_from_reaction` (bio_stack.py:280) extracts `agent_id` from `Reaction.context.agent_id` but `Reaction`/`ReactionContext` carry no `goal_tag`.  When Phase 4 rewrites this subscriber to call `distributor.distribute(agent_id, reward, goal_tag=?)`, goal_tag must come from somewhere.

**Options:** (a) Pass `goal_tag=None` from the reaction path — goalless credit, accepted for PoC.  Goal-level credit only flows through the deliberation path (agent_loop direct calls), not reactions.  (b) Add `goal_tag: str | None` to `ReactionContext` and thread it from the agent loop.  (c) Capture `active_goal` in the subscriber closure's scope at construction time.

I think the core part of this is that ultimately the reaction outcomes get fed into the goal so I'm leaning toward A, although am I missing something

**Recommended:** Option (a) for Phase 4.  The reaction→reward path is a different attribution surface than deliberation→reward.  Reactions fire from any bio-system event, not just goal-directed deliberation.  Forcing goal context onto reactions conflates two signals.  Revisit if goal_depth_integration Stage 2 (episode goal tagging) creates a natural threading point.

### OQ-2: `BioStack.on_session_end()` vs existing `save_cerebellum()`

The plan creates `on_session_end()` to centralize cleanup.  `save_cerebellum()` already exists as a standalone public method.

**Resolution:** `on_session_end()` calls `save_cerebellum()` + `distributor.cleanup_session()` internally.  `save_cerebellum()` stays public for backward compat (callers that only need the cerebellum save).  `on_session_end()` is the new canonical session-end entry point.  Existing callers that call `save_cerebellum()` directly should migrate to `on_session_end()` — but this is not urgent since calling both is idempotent (cerebellum save is a no-op on second call if state hasn't changed, distributor cleanup is explicitly idempotent).

### OQ-3: ValenceSignal EMA coefficient tuning

The plan uses `ema = 0.9 * ema + 0.1 * signal.value`.  The 0.9/0.1 split means recent signals dominate quickly (a single +1.0 signal on a zero-EMA entry produces 0.1, then 0.19 after a second).  This may be too sluggish or too responsive depending on how frequently signals arrive.  The coefficient should be configurable (class-level constant on WorkingMemorySet) and tuned after Phase 5 ships.  Don't optimize prematurely — ship the default, observe in sims, adjust.
