# Complete the SEM Learning Loop

**Status:** COMPLETE (2026-04-17). All 5 stages shipped. PoC: 11/11 + 13/13 hypotheses confirmed.
**Scope:** ~500-700 LOC across 5 stages. Consolidates 4 shell plans into one coherent deliverable.
**Target version:** 0.3 (pre-publish).
**Absorbs:**
- [cerebellum_activation.md](cerebellum_activation.md) — wire Cerebellum into production
- [substrate_valence_annotation.md](substrate_valence_annotation.md) Stage 4 — positive valence + success reactions
- [substrate_episode_boundary_enrichment.md](substrate_episode_boundary_enrichment.md) Stage 3 — pain/salience spike boundary (Stages 1-2 deferred)
- `distribute_reward` wiring gap (documented in valence annotation plan + concept decomposition companion)

## Motivation

The SEM → bio-pipeline learning loop has all its pieces built but several are not connected in production:

```
SEM Entity interaction
    │ agent executes affordance via tool
    ▼
CerebellumModulator.execute()           ← NOT ACTIVATED in production
    │ fail → _emit_failure_reaction()   ← reaction_bus already wired
    │ success → (nothing)               ← NO success reaction exists
    ▼
ReactionBus.publish()
    │ → hippocampus.capture_reaction()  ← SHIPPED (valence annotation S1-3)
    │ → NAc.distribute_reward()         ← ZERO CALLERS
    ▼
Episode close → apply_hebbian_on_close
    │ → Edge.metadata["valence"]        ← SHIPPED (valence annotation S1-3)
    │ → EC threshold_override           ← WIRED but distribute_reward never called
    ▼
Pain spike → episode boundary           ← NOT IMPLEMENTED
    │ → clean "what went wrong" episode
    ▼
Future retrieval
    │ spreading_activation(propagate_valence=True)  ← SHIPPED (valence annotation S2)
    │ EC.pattern_complete(threshold adjusted by NAc) ← EXISTS but never adjusts
```

After this plan ships, every arrow fires. An agent that swings a rusty sword and gets hurt will:
1. Cerebellum learns the forward model (fewer LLM calls next time)
2. Pain reaction fires → edge valence annotated negative
3. NAc `distribute_reward` adjusts EC threshold for sword concepts
4. Pain spike closes the episode → clean "sword interaction" boundary
5. Success reaction on confident Cerebellum predictions → positive valence
6. Next encounter: spreading activation carries affective memory + EC recognizes sword concepts more aggressively

## Stages

### Stage 1 — Cerebellum activation (~150 LOC)

Wire the existing Cerebellum infrastructure into `build_bio_stack` and production entry points.

1. Add `cerebellum: Any = None` to `BioStack` frozen dataclass
2. Construct `Cerebellum(config=CerebellumConfig(persistence_path=...))` in `build_bio_stack`, gated on `persistence_dir is not None`
3. Wire `memory_hub.cerebellum = cerebellum`
4. Pass `bio.cerebellum` to `build_executor(cerebellum=...)` at all production call sites
5. Verify `generate_tools_for_entity` calls `cerebellum_modulator_factory` with both `cerebellum=` and `reaction_bus=`
6. Load persisted state on startup if `cerebellum.json` exists; save on session end
7. Tests: BioStack construction with/without persistence_dir, cerebellum forwarded to executor

**Key files:** `runtime/bio_stack.py`, `cli.py`, `simulation/orchestrator.py`, `embodied_runtime/agentic_runtime.py`

### Stage 2 — Wire `distribute_reward` (~100 LOC)

Connect reactions to NAc's reward-bias pathway so eligibility traces produce threshold adjustments.

**The gap:** `NAc.update_eligibility()` IS called when percepts complete to substrate nodes (`encoder.py:193`). `NAc.distribute_reward()` exists and correctly distributes reward across eligible nodes via `credit_node()`. But `distribute_reward` has **zero external callers**. Eligibility traces build up and decay unused.

1. Create a ReactionBus subscriber that calls `nac.distribute_reward(agent_id, reward)` where `reward = -intensity` for NEGATIVE valence, `+intensity` for POSITIVE
2. Wire the subscriber in `build_bio_stack` alongside the existing `hippocampus.capture_reaction` subscriber (Step 4b)
3. The subscriber needs `agent_id` from the reaction's context — use `reaction.context.agent_id` (populated by producers since F0.5) with fallback to a default
4. Tests: reaction fires → distribute_reward called → credit_node updates reward_bias → get_threshold_overrides returns non-zero

**Key files:** `runtime/bio_stack.py`, `decisions/nac.py`

**Design note:** The subscriber is a simple closure, not a new class. It lives in `bio_stack.py` as a local function within `build_bio_stack`, capturing `nac` from the enclosing scope. This follows the same pattern as the `hippocampus.capture_reaction` subscriber.

### Stage 3 — Positive valence + success reactions (~120 LOC)

Add `CerebellumModulator._emit_success_reaction()` for confident Cerebellum predictions.

1. Add `_emit_success_reaction(affordance, intensity)` to `CerebellumModulator` — mirrors `_emit_failure_reaction` with `valence=POSITIVE`
2. Call it when `predicted is not None` (Cerebellum is confident) in `execute()`
3. Intensity scales with Cerebellum confidence (higher confidence → stronger positive signal)
4. Tests: confident prediction emits POSITIVE reaction, no-confidence fallback does NOT emit success (success is for cached predictions, not LLM fallback results)

**Key files:** `embodiment/backends/cerebellum_modulator.py`

**Design note:** Success intensity should be lower than failure intensity. The biological asymmetry (negativity bias) means pain signals are stronger than pleasure signals. Suggested: `success_intensity = confidence * 0.3` vs failure `intensity = 0.3-0.5`.

### Stage 4 — Pain/salience spike episode boundary (~130 LOC)

From [substrate_episode_boundary_enrichment.md](substrate_episode_boundary_enrichment.md) Rule 3. Tool execution (Rule 1) and semantic shift (Rule 2) are deferred — they need more calibration work and are less tightly coupled to the SEM loop.

1. Add `salience_spike: float | None = None` to `CaptureEvent`
2. Implement `salience_spike_rule(min_intensity: float = 0.5)` in `episode.py`
3. Wire: PainBus subscriber records latest pain intensity since last capture event; agent loop / MemoryHub populates the field on the next `CaptureEvent`
4. Register the rule in `Hippocampus.__init__` episode detector (alongside tick_gap, channel_change, scn_tag_change)
5. Tests: pain spike above threshold closes episode; below threshold does not; P4 mug regression passes

**Key files:** `memory/episode.py`, `memory/hippocampus.py`, wiring site TBD (agent loop or build_bio_stack)

### Stage 5 — End-to-end PoC + experiment writeup

Deterministic script (no LLM calls) exercising the full loop:

**Scenario A — Pain loop:**
1. Agent encounters rusty sword SEM entity
2. CerebellumModulator executes "swing" affordance → falls back to LLM stub → trains Cerebellum
3. Affordance fails → pain reaction → ReactionBus
4. Pain captured into episode → edges annotated with negative valence
5. NAc `distribute_reward` adjusts EC threshold for sword-related nodes
6. Pain spike closes the episode boundary
7. Second encounter: spreading activation from "sword" cue carries negative valence + EC threshold is adjusted

**Scenario B — Success loop:**
1. Same agent encounters sword again
2. Cerebellum is now confident → returns cached prediction → success reaction
3. Positive valence annotates edges
4. Verify: edges have mixed valence (negative from pain, positive from success — net still negative since pain was stronger)

**Scenario C — Clean control:**
1. Fresh agent, same sword, no pain history
2. Verify: zero valence, zero reward bias, default EC threshold

**Measurements:**
- Edge `metadata["valence"]` values before/after each episode
- NAc `_reward_bias` values before/after distribute_reward
- EC `get_threshold_overrides` output
- Episode count (should be 2 in Scenario A due to pain spike boundary)
- Cerebellum fallback count (should decrease across interactions)

**Deliverables:**
- `scripts/sem_learning_loop_poc.py` — deterministic script
- `docs/experiments/sem_learning_loop_poc.md` — results with reproduction instructions
- `docs/experiments/protocols/sem_learning_loop_reproduction.md` — step-by-step reproduction protocol

## What this plan does NOT do

- **Tool execution boundary (episode enrichment Rule 1):** Needs executor → agent loop wiring that's coupled to the agent loop internals, not the SEM loop. Deferred.
- **Semantic shift boundary (episode enrichment Rule 2):** Needs calibration sweep against real conversation data. Deferred to P5.
- **`distribute_reward` decay integration:** `NAc.decay_reward_biases()` exists but is not called periodically. Adding periodic calls is a tick-loop concern, not an SEM concern. Documented as a follow-up.
- **Agent-level valence consumption:** How the prompt assembler or goal system uses valence. That's a behavioral design question, not a wiring question. Deferred to behavioral_convergence_practice.md.

## Dependency order

```
Stage 1 (Cerebellum activation)
    ↓
Stage 2 (distribute_reward)     Stage 3 (success reactions)
    ↓                               ↓
Stage 4 (pain spike boundary)
    ↓
Stage 5 (PoC + writeup)
```

Stages 2 and 3 are independent of each other. Stage 4 depends on the PainBus wiring being live (which it is post-valence-annotation). Stage 5 exercises everything.

## After this plan

- **Episode boundary enrichment Stages 1-2** (tool execution + semantic shift) — deferred, ship before P5
- **0.3 publish** — concept decomposition shipped, valence shipped, SEM loop complete, boundaries enriched
- **P5 stress persistence** — exercises the substrate under realistic load with all the above active
- **behavioral_convergence_practice.md** — first real experiment entry using the complete SEM loop
