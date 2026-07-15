# Deliberative Thought Stream — goal-aware, salience-scored inner monologue

**Status:** Stages 1+2 SHIPPED (PR #183, 2026-04-23). **Stages 3, 3b, 4 ABSORBED** into [temporal_credit_integration.md](temporal_credit_integration.md) (2026-04-24).
**Scope:** ~350-450 LOC across existing modules (no new bio-system)
**Target version:** 0.9
**Depends on:** [archive/pfc_deliberation_cycle.md](pfc_deliberation_cycle.md) (shipped)
**Extends:** [goal_depth_integration.md](../deferred/goal_depth_integration.md) (Stage 3 absorbed here; Stage 1 partially absorbed as goal_tag on THOUGHTs — full GOAL WMS kind remains in goal_depth)
**Gates:** none; behavioral improvement + architectural foundation

> **Note:** Stages 3 (goal-tagged thoughts + NAc goal-outcome learning), 3b (SCN temporal correlation), and 4 (ValenceSignal) are now tracked in [temporal_credit_integration.md](temporal_credit_integration.md) Phases 4, 3, and 5 respectively.  The deliberation-specific content (goal_tag on WMEntry, _goal_reward_bias on NAc, ThoughtGate goal_reward_bias parameter) is preserved verbatim.  The attribution mechanism (how deliberation events reach NAc) is upgraded from ad-hoc _last_deliberation_event_id to the TemporalEvent protocol + TemporalCreditDistributor.  Stages 1 and 2 below are shipped and unchanged.

---

## Problem

The PFC deliberation cycle (0.8) runs multi-cycle recurrence: the LLM thinks → bio-systems enrich → the LLM thinks again.  But three gaps prevent the cycle from actually building coherent reasoning:

### 1. Thoughts don't build on each other

Each cycle **replaces** `bio_enrichment_context` with the current cycle's enrichment.  The LLM in cycle 3 sees cycle 3's enrichment but not cycle 1's or 2's.  The bio-system associations that prompted the LLM to say "I need to think more" are gone by the next cycle.

WMS THOUGHT entries accumulate, but the prompt builder renders them as a bullet list: 6 entries max, 200 chars each, 400 token budget total.  A single enrichment round produces 500+ characters.  By cycle 3, the LLM sees fragments of fragments.

**Net effect:** Each deliberation cycle starts semi-fresh instead of building on prior reasoning.

### 2. All thoughts have the same weight

Every THOUGHT entry gets `salience=0.5`, hardcoded.  A thought that triggered 4 memory recalls and 2 causal predictions is weighted identically to one that triggered nothing.  When the prompt builder selects which thoughts to render, it takes the most recent, not the most important.

The signals for computed salience already exist — enrichment section count, Jaccard novelty between cycles — they're just unused.

### 3. Thoughts are disconnected from goals

Goals flow through `StructuredContext.active_goal` and the LLM prompt, but WMS THOUGHT entries carry no goal tag.  NAc learns "tool X produces good outcomes" (direct event-id attribution via ToolPainBridge) but never learns "thinking pattern Y under goal Z tends to produce good outcomes."

The missing link: **thoughts sit between goals and actions**.  A thought happens *in service of* a goal and *produces* an action.  If thoughts carry a goal tag, outcome attribution flows through the full chain: `outcome → action → thought → goal`.  NAc learns the chain, not just the last link.

## Bio-plausible framing

The prefrontal cortex maintains a **persistent workspace** — active representations that accumulate across recurrence cycles, not a FIFO that drops prior iterations.  The basal ganglia (NAc) modulate which representations stay active based on predicted reward value.  Dorsolateral PFC holds goal representations; orbitofrontal PFC evaluates goal value through NAc.  Hippocampus provides episodic context tagged with the goal that was active during encoding.

The current architecture is missing the workspace persistence (gap 1), the salience modulation (gap 2), and the goal tagging on representations (gap 3).  No new bio-system is needed — the existing structures need wiring.

## Design

### Stage 1: Deliberation transcript (thoughts build on each other)

Replace the lossy per-cycle replacement with an accumulating transcript.

**What changes:**
- `_run_deliberation_cycles` builds a `deliberation_transcript: list[str]` where each entry pairs the LLM's reasoning with the bio-system response it triggered
- Each entry: `"Your reasoning: {reasoning}\nBio-system response: {enrichment}"`
- The full transcript is set on `StructuredContext.deliberation_transcript` (new field, `list[str] | None`)
- `bio_enrichment_context` still carries the *current* cycle's enrichment (for the inline section 1.2 one-shot path); the transcript carries the *accumulated* chain (for multi-cycle)

**Prompt builder changes:**
- New `_add_deliberation_transcript_section()` renders the transcript as a single section at `SectionPriority.IMPORTANT`, truncatable
- **Token budget: proportional to n_ctx, not hardcoded.** Formula: `min(2000, int((n_ctx - response_reserve - overhead) * 0.3))`. On 4K local: ~891 tokens. On 8K Claude: ~2000 tokens. On 16K+: capped at 2000. The 2000 flat budget would leave only ~972 tokens for all other sections on 4K models, crowding out identity, tools, conversation, and memories.
- Truncation: drops oldest entries first (most recent reasoning is most valuable)
- **When a transcript is present, suppress the separate `bio_enrichment` section** to avoid rendering the current cycle's enrichment twice (once in the transcript's last entry, once inline). The one-shot path (no transcript) continues to use `bio_enrichment` normally.
- The old `_add_working_memory_section()` still renders non-THOUGHT WMS entries (PERCEPT, OUTCOME, etc.); THOUGHT entries move to the transcript section when a transcript is present

**What this looks like in the prompt:**
```
=== Your deliberation ===

[Cycle 1]
Your reasoning: "The guard is sleeping but I notice keys on his belt.
I could try to take them, but that risks waking him..."
Bio-system response:
- Memory: Last time you reached for something near a sleeping NPC,
  the noise check succeeded (salience=0.71)
- Prediction: stealth actions near sleeping entities have 72% success rate
- Concept: "keys" → unlock, escape, inventory

[Cycle 2]
Your reasoning: "Given the memory of success with stealth near sleeping
NPCs, and the prediction of 72% success, I'll reach for the keys slowly..."
Bio-system response:
- Memory: Slow movements reduce noise check difficulty by one tier
- Prediction: combined stealth + slow movement → 89% success estimate
```

The LLM sees its own chain of thought and how the bio-systems responded at each step.  Cycle 3 genuinely builds on cycles 1 and 2.

### Stage 2: Computed salience on thoughts

Replace hardcoded `salience=0.5` with a score derived from the enrichment result.

**Salience signal:** The enrichment result already tells us how much the bio-systems "care" about this thought.  Score components:

| Component | Source | Weight | Rationale |
|---|---|---|---|
| Section count | `n_sections` (0-5) | 0.3 | More bio-systems activated = more cross-system relevance |
| Recall depth | len(enrich_result.memories) | 0.3 | More memories triggered = stronger associative resonance |
| Novelty | `1.0 - jaccard_similarity_with_previous` | 0.4 | Novel thoughts are more informative than repetitive ones |

```python
def _compute_thought_salience(
    n_sections: int,
    n_memories: int,
    jaccard_with_previous: float,
) -> float:
    """Compute salience for a THOUGHT WMS entry.

    Range: [0.0, 1.0].  Components weighted to favor novelty
    (a thought that says something new) over mere activation
    (a thought that triggers many systems but says the same thing).
    """
    section_score = min(n_sections / 5.0, 1.0)  # Normalize to [0, 1]
    recall_score = min(n_memories / 4.0, 1.0)   # Cap at 4 recalls
    novelty_score = 1.0 - jaccard_with_previous  # 0 = identical, 1 = fully novel
    return 0.3 * section_score + 0.3 * recall_score + 0.4 * novelty_score
```

**Note on novelty bias:** This formula rewards *novel* thoughts, not *refined* ones. Cycle 1 (maximum novelty) will always outscore cycle 3 (convergent). This is acceptable for the PoC — novel contributions are legitimately more informative. Stage 4's ValenceSignal can modulate post-hoc based on outcome attribution (refined thoughts that led to success get boosted retroactively). The `_jaccard_convergence` function already in `agent_loop.py` can be reused for the Jaccard computation.

**WMS changes:**
- `wms.add(THOUGHT, salience=_compute_thought_salience(...), ...)` in both the section 1.2 inline path and `_run_deliberation_cycles`
- New query: `wms.top_by_salience(kinds, limit, min_salience=0.0)` — returns entries sorted by salience descending, then recency as tiebreaker

**Prompt builder changes:**
- `_add_working_memory_section()` uses `top_by_salience({THOUGHT}, limit=6)` instead of `by_kind({THOUGHT}, limit=6)` for the non-transcript path (section 1.2 one-shot)
- When a deliberation transcript is present (multi-cycle), THOUGHT rendering defers to the transcript section

### Stage 3: Goal-tagged thoughts + NAc goal-outcome learning

This is the bridge connecting goals to the reward system through thoughts.

**WMEntry changes:**
- Add `goal_tag: str | None = None` to `WMEntry`. WMEntry is `frozen=True, slots=True`, so this is added as a field with a default. The only construction site is `WorkingMemorySet.add()`, which gets a new `goal_tag` kwarg threaded through. No other construction sites exist.
- `wms.add(THOUGHT, ..., goal_tag=active_goal)` in the deliberation cycle

**Note on goal_depth_integration Stage 1 absorption:** This plan adds `goal_tag` to THOUGHT entries — the goal is *metadata on thoughts*, not a first-class WMS entry. goal_depth Stage 1 adds `GOAL` to `WorkingMemoryKind` so the goal itself appears in WMS. These are complementary. This plan partially absorbs Stage 1's intent (goals become visible to the cycle through the tag), but the full `GOAL` kind (tracking goal status changes, sub-goal completion) remains in goal_depth_integration.md for independent implementation.

**NAc changes — new `_goal_reward_bias` dict:**

The existing `_reward_bias` is keyed by `(agent_id, node_id)` for substrate node recognition modulation. Goal-level bias is a **different concept** — it modulates ThoughtGate threshold based on whether deliberation under a goal type historically produces good outcomes. These need separate storage:

- New `_goal_reward_bias: dict[str, float] = {}` on NAc (keyed by goal string, not `(agent_id, node_id)`)
- `credit_goal(goal_tag: str, reward: float)` — analogous to `credit_node` but for goal-level bias
- `get_goal_reward_bias(goal_tag: str) -> float` — returns bias for ThoughtGate modulation
- `decay_goal_reward_biases()` — called alongside existing `decay_reward_biases()`
- All three methods acquire `self._lock` (RLock)
- **Serialization:** add `goal_reward_bias` to `dump()`/`load_state()` alongside existing `reward_bias`. Same `{goal_string: float}` format.

**Deliberation event lifecycle:**

- Deliberation cycle calls `event_id = nac.record_event("deliberation", f"deliberation:goal:{goal_tag}", context={"goal": goal_tag, "cycle": cycle})` at cycle start
- **The event_id must survive from cycle end to outcome arrival.** Store `_last_deliberation_event_id: str | None` on StructuredContext (ephemeral, not persisted). The agentic loop reads it when attributing outcomes after the action executes.
- When the resulting action produces an outcome (existing outcome attribution path), the deliberation event_id is attributed alongside the tool event_id: `nac.record_outcome("deliberation", deliberation_event_id, outcome_valence)`
- NAc now learns: `deliberation:goal:escape → positive` (goal-level) alongside `tool:sneak_past_guard → positive` (action-level)

**NAc reward bias modulation:**
- `nac.get_goal_reward_bias(goal_tag)` — query accumulated reward bias for deliberation events tagged with this goal
- ThoughtGate can use this: when goal "escape" has high positive bias, ThoughtGate is more likely to fire (the goal context predicts that deliberation leads to good outcomes)
- When goal "escape" has negative bias, ThoughtGate threshold rises (deliberation under this goal type hasn't helped — act faster)

**ThoughtGate changes:**
- `should_think()` gains a `goal_reward_bias: float = 0.0` parameter
- The bias lowers the adaptive threshold: `threshold = max(threshold - goal_reward_bias, self._config.min_combined_score)`
- The caller (agent loop) passes `nac.get_goal_reward_bias(active_goal)` when NAc is available

**Flow:**
```
Goal active: "escape the dungeon"
    │
    ▼
Deliberation cycle:
  event_id = nac.record_event("deliberation",
      "deliberation:goal:escape the dungeon")
  context._last_deliberation_event_id = event_id
  wms.add(THOUGHT, salience=0.7, goal_tag="escape the dungeon")
    │
    ▼
Action: sneak_past_guard → success
  nac.record_outcome("deliberation", event_id, POSITIVE)
  nac.credit_goal("escape the dungeon", +1.0)
    │
    ▼
NAc now has:
  tool:sneak_past_guard → positive  (existing path)
  deliberation:goal:escape → positive  (NEW)
  _goal_reward_bias["escape the dungeon"] > 0
    │
    ▼
Next time goal "escape" is active:
  goal_reward_bias > 0 → ThoughtGate threshold lowered
  Bio-enrichment queries NAc → "deliberation under escape-type goals
  tends to produce positive outcomes" surfaces as a prediction
```

### Stage 3b: SCN temporal correlation — connecting asynchronous events

Thoughts, actions, and outcomes don't happen in lock-step.  The agent thinks about the guard's keys (t=0), takes an action (t=2s), and the outcome arrives (t=15s).  In a multi-agent sim, the orchestrator's probe and the AUT's response are separated by seconds of LLM latency.  Without temporal context, these events are just entries in separate buffers with no correlation signal.

The SCN already indexes events by temporal signature (`scn.register(memory_id, signature, significance)`).  It can answer "what else happened around the same time?" via `query_similar_time()`.  The gap: thoughts and goal-tagged deliberation events aren't registered with SCN, so the temporal index has no entry for "the agent was thinking about escape at 14:23:07."

**What changes:**

- When `_run_deliberation_cycles` records a THOUGHT entry to WMS, also register it with SCN: `scn.register(thought_id, current_signature, significance=salience)`.  The thought's `ref` field (currently None for THOUGHTs) gets set to a generated thought_id so it's cross-referenceable.
- When NAc records a deliberation event (`record_event("deliberation", ...)`), the event carries the SCN temporal signature in its context: `context={"goal": goal_tag, "scn_signature": scn.current_signature()}`.
- When an outcome arrives and NAc attributes it, SCN can answer: "what thoughts were registered near this outcome's timestamp?" via `query_similar_time(outcome_signature, window_hours=0.01)` (~36 seconds).  This is the temporal correlation signal.
- `current_signature()` convenience method on SCN: `return TemporalSignature.from_timestamp(time.time())`. Trivial but avoids importing TemporalSignature at every call site.

**Thread-safety constraint (from review):** SCN has no internal lock. Memory registration happens in the Hippocampus capture thread; thought registration happens in the agent loop thread. These can race on shared defaultdicts. **Resolution:** thought registration in `_run_deliberation_cycles` must go through the agent loop thread (it already does — the function runs synchronously). If SCN grows concurrent callers in the future, a threading.Lock should be added to SCN.register(). For now, the existing single-threaded-registration assumption holds because the capture thread registers *memory* IDs and the agent loop registers *thought* IDs into the same bins — `defaultdict(BoundedBin)` is not thread-safe for concurrent `add()` calls to the same bin. **Mitigation for Stage 3b: document this as a known concurrency risk. If the capture thread and agent loop ever register to the same hourly bin concurrently, SCN needs a lock. Today this race is benign because `BoundedBin.add()` only appends to a list + adds to a set, and CPython's GIL serializes these. But this is a GIL-dependent correctness assumption that breaks under free-threaded Python (3.13+). Track in a TODO for a future SCN lock audit.**

**Session-end cleanup (from M4 review finding):**

SCN persists ALL registered IDs to disk via `scn.save()`. Thought_ids are ephemeral (session-scoped) — they reference WMS entries that don't survive restart. Without cleanup, stale thought_ids accumulate across sessions and consume bin slots meant for real memories.

- `_run_deliberation_cycles` collects registered thought_ids in a session-scoped `set` (passed in or stored on the agent loop's session state)
- At session end (alongside `MemoryHub.on_session_end()`), iterate the set and call `scn.unregister(thought_id)` for each
- Register thoughts with `significance=0.1` (not the computed salience) so they lose eviction battles to real memories even if cleanup is missed

**Concretely:**

| File | Change | LOC |
|------|--------|-----|
| `runtime/agent_loop.py` | Register THOUGHT entries with SCN at cycle time, collect IDs for cleanup | +15 |
| `runtime/agent_loop.py` | Include `scn_signature` in NAc deliberation event context | +5 |
| `runtime/agent_loop.py` | Session-end: unregister thought_ids from SCN | +10 |
| `time/scn.py` | `current_signature()` convenience method | +5 |

### Stage 4: ValenceSignal — abstract signal type for future bio-systems

The goal is NOT a new bus.  It's a common type that existing producers can emit and WMS entries can receive, so future bio-systems plug in without knowing the source.

**Naming clarification — three valence concepts coexist:**

1. `Valence` enum in `decisions/causal_link.py`: POSITIVE/NEGATIVE/NEUTRAL/UNKNOWN. Used by NAc CausalLink to classify outcome quality. Discrete.
2. `Episode.valence` float in `memory/episode.py`: Net valence computed from captured reactions (-1.0 to +1.0). Continuous.
3. `ValenceSignal` frozen dataclass (this stage): An abstract signal carrying a float value from any bio-system producer. Used to modulate thought salience over time.

These are distinct concepts at different layers. `ValenceSignal` does NOT replace or subsume the other two — it's a **transport type** that carries the output of producers (including NAc outcome valence and PainBus intensity) to consumers (WMS entries). The name is intentionally parallel: it's a *signal* about *valence*, not the valence itself.

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
    value: float          # -1.0 to +1.0
    source: str           # "nac", "pain", "ec_novelty", etc.
    goal_tag: str | None  # which goal was active when this fired
    timestamp: float
```

**Wiring (Stage 4 only — not in PoC):**
- NAc outcome path: after `record_outcome`, emit `ValenceSignal(value=±1.0, source="nac", goal_tag=active_goal)`
- PainBus: `PainSignal` → `ValenceSignal(value=-intensity, source="pain", goal_tag=active_goal)`
- EC novelty: on pattern separation → `ValenceSignal(value=+0.3, source="ec_novelty", goal_tag=active_goal)`

**Consumption:**
- WMS entries accumulate valence signals tagged with their goal
- `wms.top_by_salience()` incorporates accumulated valence into the sort:
  `effective_salience = base_salience + sum(signal.value for signal in signals_matching_goal)`
- Positive valence from NAc boosts thoughts that contributed to success
- Negative valence from pain boosts thoughts about the pain source (survival salience — you want to REMEMBER what hurt you)

Stage 4 is the extensibility layer.  It doesn't change behavior — it creates the interface that makes Stages 1-3 composable with future bio-systems.

## What changes

| Stage | File | Change | LOC |
|-------|------|--------|-----|
| 1 | `runtime/agent_loop.py` | Build deliberation transcript in `_run_deliberation_cycles`, set on context | +30 |
| 1 | `agents/bus.py` | Add `deliberation_transcript: list[str] \| None` to `StructuredContext` | +3 |
| 1 | `agents/prompt_builder.py` | `_add_deliberation_transcript_section()`, proportional budget, suppress bio_enrichment when transcript present | +45 |
| 2 | `runtime/agent_loop.py` | `_compute_thought_salience()`, replace hardcoded 0.5 in both paths (reuse `_jaccard_convergence`) | +25 |
| 2 | `agents/working_memory.py` | `top_by_salience(kinds, limit, min_salience)` query method | +15 |
| 2 | `agents/prompt_builder.py` | Use `top_by_salience` in `_add_working_memory_section` | +5, -3 |
| 3 | `agents/working_memory.py` | Add `goal_tag: str \| None` to `WMEntry`, thread through `add()` | +5 |
| 3 | `agents/bus.py` | Add `_last_deliberation_event_id: str \| None` to `StructuredContext` | +2 |
| 3 | `runtime/agent_loop.py` | `nac.record_event("deliberation", goal_tag=...)` + event_id storage + outcome attribution | +25 |
| 3 | `decisions/nac.py` | `_goal_reward_bias` dict, `credit_goal()`, `get_goal_reward_bias()`, `decay_goal_reward_biases()`, serialization | +40 |
| 3 | `runtime/thought_gate.py` | `goal_reward_bias` parameter on `should_think()`, threshold modulation | +10 |
| 3b | `runtime/agent_loop.py` | Register THOUGHTs with SCN (significance=0.1), include `scn_signature` in NAc event context, session-end cleanup | +30 |
| 3b | `time/scn.py` | `current_signature()` convenience method | +5 |
| 4 | `decisions/valence_signal.py` | `ValenceSignal` frozen dataclass (new file — type only, not a module) | +25 |
| 4 | `agents/working_memory.py` | `receive_valence(signal)` + incorporate into `top_by_salience` | +20 |
| **Net** | | | **~+290** |

## Staging and order

**Stage 1 ships first and soaks.** The deliberation transcript is the foundation — without it, salience scoring and goal tagging have nowhere to land in the prompt.  Validate by running a 3-cycle deliberation and confirming the LLM references prior cycle reasoning in its response.

**Stage 2 ships with Stage 1 or immediately after.** Computed salience is a 25-line change with no new interfaces.  The formula can be tuned after shipping.

**Stage 3 ships after Stages 1+2 soak.** Goal tagging is the architectural contribution — it creates the goal ↔ outcome attribution chain.  Needs a multi-turn sim to validate that NAc learns goal-level patterns.

**Stage 3b ships with or after Stage 3.** SCN temporal correlation is a wiring change — the SCN API already exists.  Ships naturally alongside Stage 3 since both touch the deliberation cycle's NAc integration.

**Stage 4 ships after Stage 3+3b validate.** ValenceSignal is the abstraction layer.  Don't build it until the concrete path (Stages 3+3b) proves the pattern works.

## Key constraints

1. **No new bus.** PainBus and ReactionBus already exist.  ValenceSignal is a type, not a transport — it flows through existing paths (NAc outcome callback, PainBus subscriber, direct WMS injection).

2. **Salience is computed, not learned (Stages 1-2).** The formula is static.  NAc goal-level learning (Stage 3) is the first learned modulation.  Don't prematurely optimize the formula — ship the static version, observe in sims, tune.

3. **Goal tag is the active goal string, not a goal ID.** Goals don't have stable IDs yet (goal_depth_integration.md Stage 4).  Using the description string is lossy (paraphrase collapse) but sufficient for the PoC.  When goal IDs land, swap the tag type.

4. **Transcript replaces enrichment context in multi-cycle, coexists in one-shot.** The one-shot path (gate fires, one enrichment round, `ready_to_act: true`) still uses `bio_enrichment_context` directly.  The transcript is only built when cycles 2+ actually run.  No prompt change for the common case.

5. **Negative valence = higher salience, not lower.** Pain thoughts are salient (survival).  The intuition "negative = drop it" is wrong for bio-plausibility.  Negative valence increases salience; it's the *goal-level reward bias* that determines whether the agent *pursues* or *avoids* the associated goal.

6. **`_goal_reward_bias` is separate from `_reward_bias`.** The existing `_reward_bias` dict is keyed by `(agent_id, node_id)` and modulates EC substrate recognition thresholds.  Goal-level bias is keyed by goal string and modulates ThoughtGate deliberation threshold.  Different keys, different consumers, different semantics.  Do NOT merge them.

7. **Deliberation event_id must survive from cycle end to outcome arrival.** Store on `StructuredContext._last_deliberation_event_id` (ephemeral).  The agentic loop reads it after tool execution for outcome attribution.  This field is NOT persisted — deliberation events that span a session restart are lost (acceptable for PoC; goal_depth Stage 4 handles cross-session).

8. **Transcript budget is proportional to n_ctx.** Formula: `min(2000, int((n_ctx - response_reserve - overhead) * 0.3))`.  Never hardcode 2000 — it crowds out critical sections on 4K local models.

9. **Suppress `bio_enrichment` section when transcript is present.** The current cycle's enrichment appears as the last entry in the transcript.  Rendering it twice wastes budget and confuses the LLM about which is "current."

10. **SCN thought registration uses LOW significance and cleans up at session end.** Thoughts registered in SCN bins persist to disk via `scn.save()` at session end.  Without cleanup, stale thought_ids accumulate across sessions, consuming `BoundedBin` slots (cap 200/bin) meant for genuine memory_ids and referencing WMS entries that no longer exist.  **Two mitigations:** (a) register with `significance=0.1` so thoughts lose eviction battles to memories naturally, and (b) collect thought_ids in a session-scoped `set` and call `scn.unregister(thought_id)` for each during the agentic loop's session-end path.  The temporal correlation only needs thoughts present for the ~36-second `query_similar_time` window.

11. **NAc `_goal_reward_bias` must be serialized.** Add to `dump()` as `"goal_reward_bias": self._goal_reward_bias` and to `load_state()` as `self._goal_reward_bias = state.get("goal_reward_bias", {})`.  Old snapshots missing the field silently start fresh — graceful degradation.

12. **Goal string paraphrase proliferation is accepted for PoC.** Natural-language goal tags produce duplicates under paraphrase ("escape the dungeon" vs "get out of the dungeon").  `decay_goal_reward_biases()` cleans up stale entries.  goal_depth Stage 4 introduces stable goal IDs that eliminate this class of problem.  Do NOT add normalization heuristics (lowering, stopword removal) — they break on goals where casing or stopwords are semantically meaningful.

13. **Stage 2 must emit sim_log for salience visibility.** THOUGHT entries in WMS are never persisted to JSONL.  Without a `sim_log("THOUGHT", f"salience={salience:.2f} ...")` at the `wms.add()` call site, the Stage 2 validation step ("confirm varying salience in JSONL") is impossible.  Add the log call alongside the `wms.add()` in both the inline path and `_run_deliberation_cycles`.

## Pre-implementation review findings

### Concurrency/State (Lens 1)

| ID | Finding | Severity | Resolution |
|----|---------|----------|------------|
| C1 | WMS `add()` with `goal_tag` is thread-safe (lock serializes) | Info | No action |
| C2 | Deliberation event_id needs to survive from cycle end to outcome attribution | **Critical** | Store on `StructuredContext._last_deliberation_event_id` |
| C3 | SCN has no internal lock; thought registration races with capture thread on shared bins | **Important** | GIL-safe today; document as concurrency risk for free-threaded Python. Defer SCN lock to a future audit |
| C4 | `_goal_reward_bias` reads/writes must be under NAc's RLock | Info | Natural — new methods acquire `self._lock` |

### Prompt/LLM behavior (Lens 2)

| ID | Finding | Severity | Resolution |
|----|---------|----------|------------|
| P1 | 2000 token flat budget leaves only ~972 tokens for all other sections on 4K models | **Critical** | Proportional budget: `min(2000, int(available * 0.3))` |
| P2 | Salience formula favors novel (cycle 1) over refined (cycle 3) thoughts | Info | Acceptable for PoC; document as known bias. Stage 4 ValenceSignal modulates retroactively |
| P3 | Local 14B models may struggle with structured `[Cycle N]` transcript format | **Nice-to-have** | Defer compact mode to post-soak tuning. The format is standard enough for 14B |
| P4 | Bio_enrichment section duplicates current cycle's enrichment when transcript present | **Important** | Suppress bio_enrichment section when deliberation_transcript is non-None |

### Persistence/Migration (Lens 3)

| ID | Finding | Severity | Resolution |
|----|---------|----------|------------|
| M1 | WMEntry is never persisted — `goal_tag` addition has zero migration risk | Info | No action |
| M2 | StructuredContext is never persisted — `deliberation_transcript` and `_last_deliberation_event_id` have zero migration risk | Info | No action |
| M3 | NAc `dump()/load_state()` needs `_goal_reward_bias` serialization. Old snapshots missing the field need `state.get("goal_reward_bias", {})` default | **Important** | Stage 3 must add serialization. Same `{goal_string: float}` format as existing `reward_bias` |
| M4 | SCN persists ALL registered IDs via `dump()`. Thought_ids registered in SCN bins survive session restart and consume `BoundedBin` slots (cap 200/bin) meant for genuine memory_ids. Old thought_ids are stale — they reference session-scoped WMS entries that no longer exist | **Critical** | **Stage 3b must unregister thought_ids at session end.** Add cleanup to `_run_deliberation_cycles` exit (clear thought_ids from SCN via `scn.unregister(thought_id)`) OR register thoughts with significance=0.1 so they lose eviction battles naturally AND are among the first evicted when real memories arrive. Preferred: explicit cleanup in the agentic loop's session-end path, alongside existing `on_session_end()` calls. Collect thought_ids in a session-scoped set for bulk unregistration |

### Observability/Debugging (Lens 4)

| ID | Finding | Severity | Resolution |
|----|---------|----------|------------|
| O1 | `sim_deliberation_update()` and `sim_deliberation_end()` push to display panel only — they do NOT write JSONL structured events. Transcripts and salience scores won't appear in `MAXIM_LOG_FILE` | **Important** | Stage 1 should add `sim_log("DELIBERATION", ...)` calls alongside the display updates so transcripts appear in JSONL for debugging. The existing `sim_log` function handles JSONL emission |
| O2 | `EVENT_VERBOSITY` in `utils/structured_logging.py` has no deliberation-specific event types | **Nice-to-have** | Defer to post-soak. `sim_log` already works for diagnostic purposes. Adding structured event types is a polish step |
| O3 | Interactive display's thinking panel (`set_thinking()`) already receives full reasoning text cycle-by-cycle. Transcript rendering works without display changes | Info | No display changes needed for Stage 1 |
| O4 | Provenance collector (`PipelineStage` enum) has no DELIBERATION stage. Deliberation events are invisible to provenance tracing | **Nice-to-have** | Defer. Provenance integration can use `PipelineStage.DECISION` as a sub-stage if needed |
| O5 | Stage 2 validation step says "confirm THOUGHT entries in the JSONL log have varying salience values" — but THOUGHT entries are WMS-only (never persisted to JSONL). The salience is only visible in sim_log output or the prompt itself | **Important** | Stage 2 should emit `sim_log("THOUGHT", f"salience={salience:.2f} ...")` when adding THOUGHT entries to WMS, so the validation step actually works |

### Cost/Performance (Lens 5)

| ID | Finding | Severity | Resolution |
|----|---------|----------|------------|
| K1 | Transcript adds ~200-600 tokens/cycle to prompt input. Over 3 cycles: ~$0.005/deliberation at cloud rates. Local models: ~0.5-1.5s extra latency/cycle on 14B | Info | Acceptable. Proportional budget cap limits this |
| K2 | `_goal_reward_bias` grows by one entry per unique goal string. Natural-language goals produce paraphrase variants: "escape the dungeon" vs "get out of the dungeon" create separate entries. No dedup | **Important** | Known limitation for PoC. Mitigations: (1) `decay_goal_reward_biases()` cleans up stale entries over time; (2) goal_depth Stage 4 introduces stable goal IDs that eliminate paraphrase duplication. For PoC: accept proliferation, rely on decay |
| K3 | SCN thought registration: ~20-50 thoughts/session compete with memories for 200-slot bins. Thoughts at typical salience (0.3-0.7) can evict real memories | **Important** | Register thoughts with LOW significance (0.1-0.2) so they lose eviction battles against memory registrations. The temporal correlation only needs them present for the ~36-second query window, not permanently |
| K4 | `_pending_events` buffer: 3 deliberation events per cycle is negligible vs 100-entry cap | Info | No action |
| K5 | `top_by_salience` sorts 64-entry deque: O(n log n) ≈ 384 comparisons, once per prompt build | Info | No action |

### Absorption integrity

| ID | Finding | Resolution |
|----|---------|------------|
| A1 | Plan claimed to absorb goal_depth Stage 1 (GOAL WMS kind) but only adds goal_tag to THOUGHTs | Corrected: Stage 1 absorption is partial. GOAL WMS kind remains in goal_depth_integration.md |
| A2 | goal_depth Stages 2+4 (episode goal tagging, goal persistence) are independent | No breakage. These remain viable follow-ons |
| A3 | goal_depth Stage 3 (NAc goal-outcome learning) is properly absorbed | Confirmed |

## Relationship to existing plans

- **Partially absorbs** [goal_depth_integration.md](../deferred/goal_depth_integration.md) Stage 3 (NAc goal-outcome learning).  Stage 1 (GOAL WMS kind) is *partially* absorbed as goal_tag on THOUGHTs — the full GOAL kind + status tracking remains in goal_depth.  Stages 2 (episode goal tagging) and 4 (goal persistence) remain independent follow-ons.
- **Extends** [archive/pfc_deliberation_cycle.md](pfc_deliberation_cycle.md) — uses the cycle infrastructure shipped in 0.8 without modifying cycle logic.  The cycle produces richer data; the prompt builder consumes it differently.
- **Enables** ThoughtGate adaptive threshold via NAc goal bias — currently ThoughtGate has no NAc input.  Stage 3 creates the signal.
- **Enables** cross-session goal learning — once NAc has goal-tagged causal links, they persist across sessions via existing NAc serialization.  No new persistence code needed.

## Validation

1. **Stage 1:** Run a 3-cycle deliberation sim.  Confirm the LLM's cycle 3 reasoning explicitly references content from cycles 1 and 2 (grep the response for keywords from prior enrichment).
2. **Stage 2:** Run a sim and confirm THOUGHT entries in the JSONL log have varying salience values (not all 0.5).  Check that high-salience thoughts appear in the prompt when the budget is tight.
3. **Stage 3:** Run a multi-turn sim where the same goal type recurs.  Confirm NAc has `deliberation:goal:*` causal links after the first occurrence.  Confirm ThoughtGate fires more readily on the second occurrence (reward bias lowers threshold).
4. **Stage 3b:** Run an embodiment sim where SEM pain fires during deliberation.  Confirm SCN temporal query finds the thought registered nearest to the pain event.  Confirm NAc attributes the pain outcome to the temporally proximate deliberation event when no direct event_id exists.
5. **Stage 4:** Confirm PainBus-originated negative valence increases thought salience (not decreases).  Confirm NAc positive valence for a goal tag boosts ThoughtGate firing for that goal.
