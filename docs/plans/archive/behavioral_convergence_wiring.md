# Behavioral Convergence Wiring — Close the Tier 2 Gaps

**Status:** COMPLETE (2026-04-17). All 4 stages shipped. Experiment 2: 13/13 hypotheses confirmed. All 3 testing tiers now PASS (Tier 3 Exp 4: 5/5, organic LLM learning).
**Scope:** ~400-600 LOC across 4 stages.
**Target version:** 0.3 (pre-publish). Unblocks Experiment 2 in [behavioral_convergence_practice.md](../deferred/behavioral_convergence_practice.md).
**Depends on:** SEM learning loop (SHIPPED), concept decomposition Stage 1 (SHIPPED).

## Motivation

Experiment 1 (2026-04-17) confirmed that the substrate learns affective associations across sessions — rusty swords carry negative valence, healing potions carry positive. But the LLM never sees this information. The agent's memory has opinions about swords; the agent's decisions don't reflect them.

Three gaps prevent the bio-pipeline learning from reaching the LLM:

```
                                        ← Gap 1: valence not in prompt
                                       /
Substrate ──→ Valence on edges ──→ ??? ──→ LLM decision ──→ Tool selection
                                       \
NAc reward bias ──→ EC threshold ──→ ???    ← (this path works mechanically,
                                              but EC results aren't surfaced)

Energy tracker ──→ ??? ──→ Reaction ──→ NAc  ← Gap 2: no depletion → reaction
                                              
Agent loop ──→ ??? ──→ observe_episode_event  ← Gap 3: episodes not in production
```

## Audit of existing infrastructure (2026-04-17)

### What exists and works

| System | Status | Location |
|---|---|---|
| `StructuredContext.causal_context` | EXISTS, populated | `agents/bus.py:549`. `MemoryAgent._build_causal_context()` queries `NAc.predict()` for recent tools. |
| `PromptBuilder` injects causal_context | EXISTS | Causal predictions are included in LLM context. |
| `ReactionKind` includes hunger/fatigue/satiation | EXISTS | `reactions/types.py:58` |
| Energy tracker | EXISTS | `energy/tracker.py` — token/compute/movement tracking |
| `Embodiment.vital_metrics` | EXISTS | `embodiment/body.py` — per-entity vital metric dict |
| `StructuredContext.body_state` | EXISTS | `agents/bus.py:560` — populated by embodiment integration |
| `Hippocampus.observe_episode_event` | EXISTS | `memory/hippocampus.py:1206` — full episode detection with boundary rules |
| `bio_integration.capture_loop_cycle` | EXISTS | `runtime/bio_integration.py:52` — wires agent loop to hippocampus capture |

### What's missing

| Gap | Impact | Severity |
|---|---|---|
| **Gap 1: Valence not in LLM context** | Agent learns but doesn't act on learning | Critical |
| **Gap 2: Energy → Reaction bridge** | Agent can't learn to eat/drink when hungry | Medium |
| **Gap 3: observe_episode_event not in production** | Episode boundaries don't fire in real agent runs | Critical |

## Stages

### Stage 1 — Surface valence in StructuredContext (~100 LOC)

The prompt assembler must show the LLM what the agent has learned about entities and concepts.

**Approach:** Add a `valence_context: list[dict]` field to `StructuredContext`. Populate it from `hippocampus.retrieve_on_cue(cue, include_valence=True)` during context building.

1. Add `valence_context: list[dict] = field(default_factory=list)` to `StructuredContext`
   - Each entry: `{"concept": str, "valence": float, "associations": list[str]}`
2. In `MemoryAgent._build_context()` (or `bio_integration.py`), after substrate encoding:
   - For each recently activated substrate node, call `hippocampus.retrieve_on_cue(node, include_valence=True)`
   - Filter to nodes with non-zero valence
   - Build entries for `valence_context`
3. In `PromptAssembler.compose_memory_section()`, inject valence context:
   ```
   ## Learned associations
   - "rusty sword": negative association (pain from past experience)
   - "healing potion": positive association (beneficial past experience)
   ```
4. Tests: StructuredContext with valence populated, PromptAssembler includes it in output

**Key files:** `agents/bus.py` (StructuredContext), `prompts/assembler.py` (compose), `runtime/bio_integration.py` (wiring)

**Design decision:** Valence context is descriptive ("you have negative associations with X"), not prescriptive ("avoid X"). The LLM decides what to do with the information — we don't override tool selection. This preserves the isolation hygiene rule: bio-systems inform, they don't command.

### Stage 2 — Wire observe_episode_event into production (~80 LOC)

Episode boundary detection only fires in tests. The production agent loop calls `capture_from_loop_async` but never `observe_episode_event`. Both paths need to run — they serve different purposes (memory capture vs episode binding).

**Approach:** Call `observe_episode_event` from the same site that calls `capture_from_loop_async`.

1. In `bio_integration.capture_loop_cycle()`, after the existing `capture_from_loop_async` call:
   ```python
   # Episode detection — P3a binding mechanism
   if hippocampus is not None:
       from maxim.memory.episode import CaptureEvent
       hippocampus.observe_episode_event(CaptureEvent(
           tick=tick_counter,
           channel=channel or "text",
           sender_id=sender_id,
           activated_nodes=tuple(substrate_node_ids),
           salience_spike=latest_pain_intensity,
       ))
   ```
2. Thread `tick_counter` through the agent loop (increment on each cycle)
3. Thread `substrate_node_ids` from the encoder output (already computed for `capture_from_loop_async`)
4. Thread `latest_pain_intensity` from PainBus (record highest since last cycle, reset after)
5. Tests: agent loop cycle produces episode events, boundary rules fire on tick gap / pain spike

**Key files:** `runtime/bio_integration.py`, `runtime/agent_loop.py`

**Risk:** Double-capture of substrate nodes (once via capture_from_loop, once via observe_episode_event). The two systems have different purposes — capture stores EpisodicMemory records, observe builds episode co-activation graphs. They SHOULD see the same data. The risk is performance (two writes per cycle), which is negligible (~1ms combined).

### Stage 3 — Energy depletion → interoceptive Reaction bridge (~150 LOC)

When energy drops below a threshold, emit a Reaction so the bio-pipeline learns about energy states.

1. Create `energy/reactions.py` — bridge between energy tracker and ReactionBus:
   ```python
   def create_energy_reaction_bridge(
       tracker: EnergyTracker,
       reaction_bus: ReactionBus,
       *,
       hunger_threshold: float = 0.3,
       fatigue_threshold: float = 0.2,
   ) -> Callable:
       """Returns a tick callback that checks energy levels and emits
       hunger/fatigue Reactions when thresholds are crossed."""
   ```
2. The bridge checks `tracker.current_level("food")` and `tracker.current_level("stamina")` on each tick
3. Below threshold: emit `Reaction(kind="hunger", intensity=1.0-level, valence=NEGATIVE)`
4. On consume (energy restored): emit `Reaction(kind="satiation", intensity=delta, valence=POSITIVE)`
5. Wire in `build_bio_stack` or at the agent loop level (alongside the existing energy tracking)
6. Tests: energy drops below threshold → hunger reaction emitted → captured into episode → valence annotation

**Key files:** `energy/reactions.py` (new), `runtime/bio_stack.py` or `runtime/agent_loop.py` (wiring)

**Design note:** The energy tracker already has level tracking. The bridge is a simple threshold-crossing detector, not a new system. Refractory period via ReactionBus prevents spam.

### Stage 4 — Food/water SEM entity specs + integration test (~100 LOC)

Create consumable entity specs and run the full Experiment 2.

1. Create `_data/components/items/food_ration.yaml`:
   ```yaml
   entity:
     name: food_ration
     entity_type: item
     sensors:
       portions: {unit: count, range: [0, 5], initial: 5}
       nutrition: {unit: ratio, range: [0, 1], initial: 0.7}
     modulators:
       usage:
         affordances:
           eat: {params: {}, description: "Eat a portion to restore energy"}
   ```
2. Create `_data/components/items/water_flask.yaml` (similar pattern)
3. Create `_data/components/items/poison_vial.yaml` (looks like potion, harmful)
4. Integration test / Experiment 2 script:
   - Agent with depleting energy
   - Three consumables available
   - Session 1: learn outcomes (food restores, poison harms)
   - Session 2: measure tool selection preference

**Key files:** `_data/components/items/` (new specs), `scripts/behavioral_convergence_exp2.py` (new)

## What this plan does NOT do

- **Tool selection override.** The bio-pipeline informs the LLM; it does not force tool selection. The agent can still choose to drink poison if the LLM decides that's the right action (e.g., "I need to test if this is really poison").
- **Real-time energy depletion.** Energy levels are updated per-tick by the existing tracker. This plan adds the reaction bridge, not a new depletion model.
- **Full episode boundary enrichment.** Rules 1-2 (tool execution + semantic shift) from `substrate_episode_boundary_enrichment.md` are still deferred. Only the salience_spike_rule (already shipped) fires in production.
- **Multi-agent energy isolation.** In AgentPool, each agent has its own bio-stack. Energy reactions are per-agent by construction. No new isolation work needed.

## Dependency order

```
Stage 1 (valence in prompt)     Stage 2 (episode wiring)
         ↓                              ↓
Stage 3 (energy → reactions)
         ↓
Stage 4 (food/water specs + Experiment 2)
```

Stages 1 and 2 are independent. Stage 3 needs the reaction pipeline working in production (Stage 2). Stage 4 needs everything.

## Connection to other plans

- **[behavioral_convergence_practice.md](../deferred/behavioral_convergence_practice.md):** Experiment 2 is the first Tier 2 entry.
- **[substrate_episode_boundary_enrichment.md](substrate_episode_boundary_enrichment.md):** Stage 2 here wires `observe_episode_event` into production, which is prerequisite for ALL episode boundary rules to fire in real agent runs — not just the salience spike rule.
- **[sem_learning_loop.md](sem_learning_loop.md):** This plan extends the SEM loop from substrate-only (Tier 1) to behavioral (Tier 2).
- **[agent_factory_canonicalization.md](agent_factory_canonicalization.md):** Stage F5 (headless API bio-learning default) may interact with Stage 1 here — both touch how bio-systems are surfaced to the agent.

## Risks

1. **Prompt bloat.** Valence context adds tokens to every LLM call. Mitigation: limit to top-5 strongest associations, skip zero-valence. PromptAssembler already has token-budget management.
2. **LLM ignoring valence.** The LLM might not use the valence information even when it's in context. Mitigation: measure tool selection rates in Experiment 2; if no effect, try stronger prompt framing ("you have learned from experience that...").
3. **Energy depletion rate tuning.** Too fast and the agent spends all its time eating; too slow and hunger never fires. Mitigation: configurable thresholds in the bridge, calibrated in Experiment 2.
4. **Double-write performance.** `capture_from_loop_async` + `observe_episode_event` both run per cycle. Mitigation: both are ~0.5ms; combined overhead is negligible vs LLM call latency (2-5s).
