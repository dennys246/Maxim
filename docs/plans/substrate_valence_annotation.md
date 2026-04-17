# Substrate — Valence annotation via SEM reactions

**Status:** COMPLETE (2026-04-17). Stages 1-3 shipped. Stage 4 absorbed into sem_learning_loop.md (also shipped).
**Scope:** ~300–500 LOC (two broken-link fixes + episode capture + edge annotation + sim PoC).
**Target version:** post-concept-decomposition. Finer-grained concept nodes make valence targets more precise.
**Parent:** None (standalone). Companion to [substrate_concept_decomposition.md](substrate_concept_decomposition.md).
**Depends on:** Concept decomposition Stage 1 (recommended but not hard-blocked — works on sentence-level nodes too, just less precise).

## Motivation

When an agent interacts with a SEM entity (e.g., picks up a rusty sword) and experiences pain, the agent learns "using tool X caused pain" (NAc causal link) — but the **concept** "rusty sword" acquires no affective association in the substrate. The agent won't recognize rusty swords more cautiously or propagate negative valence through related concepts ("rusty", "sword", "weapon").

Biological brains don't work this way. The amygdala annotates perceptual representations with valence — you don't just learn "that action hurt," you learn "that thing is associated with pain." The substrate should carry this signal.

### The gap (discovered 2026-04-16)

NAc has two sub-systems that are structurally disconnected:

1. **Pain recording** — `ToolPainBridge` + `PainBus` subscriber write to NAc's `_links` using action/tool event signatures. This is causal learning: "this action caused this outcome."

2. **Reward bias** — `NAc._reward_bias` + `credit_node` + `distribute_reward` map substrate node IDs to reward values. EC reads this via `threshold_override` to widen/narrow recognition radius.

3. **Eligibility traces** — `update_eligibility` IS called when percepts complete to nodes (`encoder.py:193`). The substrate knows which nodes were recently active.

4. **Missing bridge** — `distribute_reward` has **zero external callers.** Pain fires, eligibility traces accumulate, but reward never arrives. The traces decay unused.

Additionally, `CerebellumModulator._reaction_bus` is always `None` in production — the `cerebellum_modulator_factory` never passes `reaction_bus=`. So SEM reactions (pain, surprise, etc.) are emitted but silently dropped before reaching any subscriber.

## Design

### Two broken links to fix

**Link 1: Wire `reaction_bus` in `cerebellum_modulator_factory`**

`embodiment/backends/cerebellum_modulator.py` — the factory function constructs `CerebellumModulator` without passing `reaction_bus=`. One-line fix: pass the bus from the caller's bio-pipeline. After this, SEM interactions that produce reactions (pain on failed affordances, etc.) actually emit `Reaction` objects to the bus.

**Link 2: Capture reactions into the pending episode**

`PendingEpisodeState` (in `memory/episode.py`) currently tracks `activated_nodes` and `reward_events` but has no `reactions` field. The fix:

1. Add `reactions: list[Reaction] = field(default_factory=list)` to `PendingEpisodeState`
2. Register a `ReactionBus` subscriber that appends reactions to the current pending episode:
   ```python
   def _capture_reaction_into_episode(reaction: Reaction) -> None:
       if hippocampus._pending_episode is not None:
           hippocampus._pending_episode.reactions.append(reaction)
   ```
3. Wire the subscriber during bio-pipeline construction (same site as `build_pain_bus`)

### Edge valence annotation

When `apply_hebbian_on_close()` runs at episode close, it already iterates all pairs of `activated_nodes` and creates/updates `ASSOCIATES` edges. After this plan, it also reads `episode.reactions` and annotates each edge:

```python
# In apply_hebbian_on_close(), after weight update:
for reaction in episode.reactions:
    sign = -1.0 if reaction.valence == Valence.NEGATIVE else 1.0
    valence_delta = sign * reaction.intensity
    # Accumulate into edge metadata (existing dict[str, Any] field)
    current = edge.metadata.get("valence", 0.0)
    edge.metadata["valence"] = max(-1.0, min(1.0, current + valence_delta))
```

**No schema change to `Edge`.** The `metadata: dict[str, Any]` field is the intended extensibility point. Valence is stored as `metadata["valence"]` — a float in `[-1, 1]`.

### Valence propagation in spreading activation

`DependencyGraph.spreading_activation()` currently propagates `activation = parent_activation * decay * weight`. After this plan, it optionally propagates valence alongside:

```python
# Optional: when caller requests valence-aware retrieval
if propagate_valence:
    node_valence[target] = node_valence.get(target, 0.0) + (
        node_valence[node_id] * decay + edge.metadata.get("valence", 0.0)
    )
```

This is additive — spreading activation still works exactly as before when `propagate_valence=False` (default). The valence signal decays through the graph just like activation does, so distant associations carry weaker affective signals.

### Positive valence (not just pain)

The design is symmetric — `Valence.POSITIVE` reactions (successful affordance execution, goal achievement, pleasure signals) annotate edges with positive valence. The agent learns "this concept is associated with good outcomes" as naturally as "this concept is associated with pain."

This requires extending `CerebellumModulator` to emit positive reactions on successful affordance execution (not just pain on failure). Currently only `_emit_failure_reaction` exists. A parallel `_emit_success_reaction` with `valence=POSITIVE, intensity=<scaled>` would complete the picture.

## Architecture flow

```
SEM Entity (rusty sword)
    │ agent interacts via tool
    ▼
CerebellumModulator.execute()
    │ affordance fails → _emit_failure_reaction()
    │ affordance succeeds → _emit_success_reaction() [NEW]
    ▼
Reaction(kind="pain", intensity=0.8, valence=NEGATIVE)
    │ ← FIX LINK 1: wire reaction_bus in factory
    ▼
ReactionBus.publish()
    │ dispatches to subscribers
    ▼
_capture_reaction_into_episode()  [NEW subscriber]
    │ ← FIX LINK 2: PendingEpisodeState.reactions field
    ▼
PendingEpisodeState.reactions.append(reaction)
    │
    ▼ (episode closes)
apply_hebbian_on_close()
    │ annotates each Hebbian edge with reaction valence
    ▼
Edge(source="text_rusty_sword", target="text_heavy",
     weight=0.7, metadata={"valence": -0.8})
    │
    ▼ (future retrieval)
spreading_activation(propagate_valence=True)
    → "rusty sword" concepts carry negative valence
    → agent behavior informed by affective memory
```

## Stages

### Stage 1 — Wire the broken links + edge annotation

1. Wire `reaction_bus=` in `cerebellum_modulator_factory` (1 line)
2. Add `reactions: list` to `PendingEpisodeState` + reaction capture subscriber
3. Annotate Hebbian edges with `metadata["valence"]` at episode close
4. Unit tests: reaction captured into episode, valence written to edge, valence accumulated across episodes
5. Persistence: `Edge.metadata` already serializes via `DependencyGraph.to_dict()` — verify valence survives dump/load round-trip

### Stage 2 — Spreading activation valence propagation

1. Add `propagate_valence: bool = False` to `spreading_activation()` signature
2. Return `dict[str, tuple[float, float]]` (activation, valence) when enabled
3. `retrieve_on_cue` gains optional `include_valence=True` that passes through
4. Unit tests: valence decays through multi-hop paths, positive and negative accumulate correctly

### Stage 3 — Simulation PoC

1. Sim scenario: agent encounters a rusty sword SEM entity, interacts, gets pain
2. Second encounter (same or next session): spreading activation from "sword" concepts carries negative valence
3. Measure: (a) does `metadata["valence"]` on relevant edges reflect the pain? (b) does `spreading_activation(propagate_valence=True)` from a "sword" cue return negative valence? (c) does a clean agent (no pain history) show zero valence on the same cue?
4. Write up as `docs/experiments/valence_annotation_poc.md`

### Stage 4 — Positive valence + success reactions

1. `CerebellumModulator._emit_success_reaction()` on successful affordance execution
2. Symmetric valence annotation (positive edges for successful interactions)
3. Sim scenario: agent uses a key to open a door (success) → "key" and "door" concepts acquire positive valence
4. Compare: agent prefers interacting with positively-valenced concepts over neutral ones

## Connection to concept decomposition

Without concept decomposition, a sentence like `"I pick up the rusty sword"` is one substrate node. The valence annotation lands on edges connected to that sentence-blob node. With concept decomposition, `"rusty sword"` is its own node — the valence annotation is precise: "rusty sword" carries negative valence, but "pick up" (as an action concept) does not.

**Recommended sequencing:** concept decomposition Stage 1 → valence annotation Stage 1 → valence annotation Stage 3 (PoC). The PoC is more compelling with decomposed concepts.

## Connection to NAc reward bias

This plan and the NAc `distribute_reward` gap (documented in the concept decomposition companion shell) are **complementary, not competing** solutions:

- **This plan** annotates Hebbian edges with valence at the substrate level. The signal propagates through spreading activation. It's a "memory" mechanism — the substrate remembers which concepts were associated with pain.

- **`distribute_reward`** adjusts EC's pattern-completion threshold via `_reward_bias`. It's a "perception" mechanism — EC recognizes painful concepts more/less aggressively.

Both should eventually exist. This plan ships first because it's self-contained (no NAc API changes). The `distribute_reward` wiring can ship independently later, using the same reaction-capture infrastructure (the `_capture_reaction_into_episode` subscriber).

## Risks

1. **Valence saturation.** If an agent repeatedly interacts with the same entity, valence accumulates toward -1.0 or +1.0 and never decays. Mitigation: apply a decay factor per episode (e.g., `valence *= 0.95` before adding new delta), similar to Hebbian weight decay.

2. **Valence bleeding.** All edges in an episode get the same valence annotation, even edges between concepts unrelated to the pain source. E.g., if "rusty sword" and "sunny day" co-occur in the same episode, "sunny day" edges also get negative valence. Mitigation: Stage 2+ could use the reaction's `source` field (which names the SEM entity) to scope annotation to edges involving the entity's concept nodes.

3. **Positive valence path doesn't exist yet.** `CerebellumModulator` only emits failure reactions. Stage 4 adds success reactions, but until then the system only learns "what hurts," not "what works." This is biologically plausible (negativity bias) but may need explicit design attention for balance.

4. **Persistence compatibility.** Adding `metadata["valence"]` to existing edges is additive — old snapshots load with `metadata={}`, which is fine (default valence = 0.0). No migration needed.

## Open design questions

1. **Valence in `retrieve_cross_modal`:** Should cross-modal retrieval surface valence? If `retrieve_cross_modal("sword", target_modality="vision")` returns vision nodes, should each carry the accumulated valence from the text→vision path? This enables "show me things I have negative associations with" queries.

2. **Agent-level behavioral impact:** How does the agent loop consume valence? Options: (a) the prompt assembler includes valence as context ("you have negative associations with X"), (b) the goal system uses valence to bias action selection, (c) the DefaultNetwork uses valence to trigger avoidance reactions. Each is a different integration point with different implications.

3. **Multi-agent valence isolation:** In `AgentPool`, each agent has its own hippocampus. Valence annotations are per-agent by construction. But if agents share a `MeshKnowledge` graph in the future, valence isolation becomes a concern. Document the isolation boundary.
