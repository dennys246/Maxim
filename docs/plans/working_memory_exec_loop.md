# Working Memory + Executive Loop — PFC-style thinking as the default

**Status:** Stages 1-7 SHIPPED (2026-04-21). All stages complete.
**Branch:** `feat/working-memory-exec-loop` (each stage gets its own sub-branch per L6)
**Scope:** ~1800-2200 LOC across 7 staged PRs
**Target version:** 0.8 (maturity) — NOT gating 1.0
**Gates:** none; structural refactor + behavioral inversion
**Depends on:** [gating_abstraction.md](gating_abstraction.md) (G0+G1 shipped, G3 folded here), [concept_exploration.md](concept_exploration.md) (L0-L2 shipped), [biosystem_unification.md](biosystem_unification.md) (pattern), [agent_factory_canonicalization.md](agent_factory_canonicalization.md) (soft prereq: Stage 1 needs Exec-before-MemoryAgent construction order, which `create_full_agent` already guarantees)
**Blocks:** nothing immediate; informs future substrate work
**Parent:** [biosystem_unification.md](biosystem_unification.md) — follows the same audit-first, one-unification-per-PR discipline
**Related:** [memory_consolidation_practice.md](memory_consolidation_practice.md) (Stage 7 composes with P8 sleep replay), [substrate_episode_boundary_enrichment.md](substrate_episode_boundary_enrichment.md)

---

## Sequencing Gate: P5 First

**This plan MUST NOT ship before P5 stress persistence passes.** P5 is the final 1.0 gate and validates the substrate at 10k+ nodes. This plan touches the 5 most load-bearing files in the repo (`agents/bus.py`, `agents/exec_agent.py`, `agents/memory_agent.py`, `agents/prompt_builder.py`, `runtime/agent_loop.py`). Shipping this alongside P5 risks masking substrate regressions behind working-memory migration bugs.

**Sequence:** P5 passes → Stage 1 ships → validate for one full sim session → proceed with Stages 2-7.

After Stage 1 ships, run a behavioral convergence sanity check (Exp 1 or Exp 3 from behavioral_convergence_practice.md) to verify the prompt-assembly migration didn't subtly change agent behavior. This is the checkpoint — if it passes, proceed; if it regresses, debug Stage 1 before adding more stages.

---

## Goal

Invert Maxim's cognitive default from **"propose action, emit output every tick"** to **"always deliberating over working memory, emit speech/action when deliberation converges."**

Make working memory a first-class Exec-owned system (not a memory tier). Feed it into the prompt as the authoritative recent-turn context surface. Gate deliberation on composite bio-signals (salience × novelty × goal-relevance × energy). Make speech a converged output of deliberation rather than the default behavior of every LLM call. Close the orphaned WORKING→SHORT_TERM transition by making use-based consolidation automatic on access, using SCN ticks for decay and EC pattern-separation for context-diversity.

## The shape of the problem

Today's recent-turn context is scattered across four parallel surfaces (`StructuredContext` deques in `MemoryAgent`, `ContextPool` 50-entry sliding window, `PerceptTraceBuffer` τ-decay ring buffer, `ReactionBus._history` deque) with three independently-tuned windowing constants (`context_window=10`, `max_turns=5`, `keep_recent=5`). `MemoryTier.WORKING` exists in the enum but is orphaned — `WorkingMemoryEntry.should_promote()` is defined and never called; `WORKING→SHORT_TERM` is declared in `_TIER_FORWARD_TRANSITIONS` and never traversed. The prompt builder assembles 4+ separate sections from these surfaces; none of them reference the WORKING tier.

Separately, deliberation (`ThinkTool` + `BioEnrichmentPipeline`) shipped as an *optional tool* the LLM can invoke, rather than the default cognitive process. Speech/text output is imperatively produced on every LLM turn. The biological PFC-working-memory-executive-function loop has all its components in the codebase but is composed backwards: thinking is opt-in, action is default. Inverting this produces a more biologically-honest, more deliberate agent with natural silence as a valid output.

The `MemoryTier.WORKING` confusion is the root cause of the scattered surfaces: the tier was trying to be both a **consolidation stage** and an **active-reference set**. Biology keeps these separate. Dropping WORKING from the tier enum and introducing a dedicated `WorkingMemorySet` as an Exec-owned active-reference layer resolves the modeling error and collapses the four parallel surfaces into one owner.

## Lessons encoded (carried from biosystem_unification.md)

- **L1 (silent-failure trigger):** four parallel recent-context producers + five parallel readers is exactly the shape that produces silent "this producer forgot to update that surface" bugs. The typed `working_memory.add(kind, ref, ...)` interface is the structural enforcement.
- **L2 (audit before designing):** Stage 1 opens with a full audit of every producer and consumer of recent-turn context.
- **L3 (pre-merge review is non-negotiable):** every stage gets two parallel reviewers before PR merge.
- **L4 (gate construction on the learning subject):** `WorkingMemorySet` is constructed by Exec, not by the signal sources that write to it.
- **L5 (declared fields beat stashes):** every kind of working-memory entry is a typed variant, not an untyped dict stash.
- **L6 (one unification per PR):** seven stages, seven PRs. Do not combine.
- **L7 (doc + memory refinement is part of the work):** each stage updates CLAUDE.md invariants and the `docs/plans/README.md` index.

## Audit

Every current producer and consumer of recent-turn context, in `src/maxim/`:

### Producers (write into the current parallel surfaces)

| # | Site | What it writes to today | Migration under this plan |
|---|---|---|---|
| 1 | `agents/memory_agent.py::_on_percept` | `StructuredContext.recent_percepts`, `detected_speech` deques | `working_memory.add(kind=PERCEPT, ref=episode_id)` via Exec's ref |
| 2 | `agents/memory_agent.py::_complete_forming_memory` | tier=WORKING on WorkingMemoryEntry; `_forming_pool` | tier=SHORT_TERM; pool sweep renamed |
| 3 | `agents/memory_agent.py::record_outcome` | `StructuredContext.recent_outcomes` deque | `working_memory.add(kind=OUTCOME, ...)` |
| 4 | `agents/context_pool.py::add_percept` / `add_outcome` | ContextPool entries | ContextPool deprecated; absorbed into WorkingMemorySet |
| 5 | `memory/percept_trace_buffer.py::record` | internal ring buffer | keeps internal buffer (NAc learning uses it); also emits `working_memory.add(kind=ACTIVATION, ...)` |
| 6 | `reactions/bus.py::publish` | `_history` deque (200) | new subscriber publishes to `working_memory.add(kind=REACTION, ...)` |
| 7 | `proprioception/pain_bus.py::publish` | PainBus subscribers (reaches hippocampus + NAc) | also publishes `working_memory.add(kind=PAIN, ...)` so Exec can see pain in deliberation |
| 8 | `memory/hippocampus_retrieval.py::recall` et al | stats only | calls `memory.touch()`; writes refs to `working_memory.add(kind=RECALL, ...)`; triggers Stage 7 use-based promotion check |
| 9 | `tools/narrative.py::ThinkTool.execute` | ActionFollowup context | `working_memory.add(kind=THOUGHT, ...)` captures deliberation output |

### Consumers (read from the current parallel surfaces)

| # | Site | What it reads today | Migration under this plan |
|---|---|---|---|
| A | `agents/prompt_builder.py::_build_tool_aware_prompt` | `conversation_history_text`, `context_pool_text`, `recent_percepts`, `recent_outcomes`, `relevant_memories` (5+ sections) | 1-2 queries over `exec.working_memory.recent(...)` / `.by_kind(...)` |
| B | `prompts/assembler.py::MemorySummary` | legacy transcript_chunk path (Phase 1 dual-write) | sources from `WorkingMemorySet`; substrate path cutover completes |
| C | `agents/memory_agent.py::build_context` | assembles StructuredContext from local deques | reads via `exec.working_memory`; StructuredContext becomes a view, not a cache |
| D | `agents/exec_agent.py::_contemplate` | local state + StructuredContext | owns `working_memory`; uses it as the deliberation substrate |
| E | `api.py` (headless) | caller must pre-populate StructuredContext | `working_memory` auto-populated from percept subscriptions; headless gap closed |

### Pre-existing bugs the audit surfaces

1. **Headless API has no recent-context subscriptions.** `api.py::maxim.create.agent` requires the caller to pre-populate `StructuredContext` — any `pymaxim` headless agent starts with zero recent-turn memory unless manually wired. This is the same class of silent-no-op the executor_bootstrap audit surfaced for `pain_bus=None`. After Stage 1: WorkingMemorySet is auto-populated because producers write through Exec's owned instance.
2. **MemoryTier.WORKING references the orphan.** 21 sites (grep needed in audit pass) reference `MemoryTier.WORKING`; after Stage 1 the tier does not exist. Each site gets an explicit decision: migrate to `SHORT_TERM` (most common — freshly-formed episodes with outcomes), delete the reference, or replace with `WorkingMemorySet` lookup.
3. **`should_promote()` is dead code.** `WorkingMemoryEntry.should_promote()` in `agents/bus.py:375-380` has zero callers. Stage 7 replaces it with the composite-scored promotion gate.
4. **`access_count` is never incremented by retrieval.** `MemoryRecord.touch()` (types.py:307-311) exists but `Hippocampus.recall()` never calls it. `access_count` stays at 1 forever unless explicitly touched. Stage 7 closes this.

---

## Design — seven stages

### Stage 0 — Pre-validation checkpoint

**Goal:** Ensure P5 stress persistence has passed and the substrate is stable before touching the 5 most load-bearing files.

**Pass criteria:** P5 experiment passes. No open regressions on main.

**LOC:** 0 (gate only)

### Stage 1 — Working Memory Unification (PFC Working Memory)

**Goal:** Drop `MemoryTier.WORKING` from the tier enum. Introduce `WorkingMemorySet` as an Exec-owned active-reference layer. Migrate all recent-turn context producers to write through the typed `add()` interface. Migrate the prompt builder to read from it.

**Post-Stage-1 checkpoint:** Run behavioral convergence sanity check (Exp 1 or Exp 3). If it regresses, debug before proceeding. This catches prompt-assembly migration bugs early.

**New module:** `agents/working_memory.py`

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Literal

class WorkingMemoryKind(str, Enum):
    PERCEPT = "percept"
    OUTCOME = "outcome"
    REACTION = "reaction"
    PAIN = "pain"
    ACTIVATION = "activation"    # PerceptTraceBuffer emission
    THOUGHT = "thought"          # ThinkTool output
    RECALL = "recall"            # Hippocampus.recall() result
    CONVERSATION = "conversation"  # user/assistant turn

@dataclass(frozen=True, slots=True)
class WorkingMemoryEntry:
    kind: WorkingMemoryKind
    ref: str | None              # episode_id, percept_id, reaction_id, or None for synthesized
    content: Any                 # dict or string payload (display-ready)
    timestamp: float             # wall-clock
    scn_tick: int                # SCN tick at add time (for decay)
    salience: float = 0.0
    agent_id: str | None = None  # multi-agent scoping (required post-Stage-F5)

class WorkingMemorySet:
    """Exec-owned active-reference layer over recent cognitive events.
    Bounded deque + SCN-tick-based decay. Agent-scoped.

    Single write interface: add(). All parallel recent-context producers route
    through this method. No code outside this module maintains a parallel buffer.
    """
    def __init__(self, *, agent_id: str, scn: SCN, capacity: int = 64) -> None: ...
    def add(self, kind: WorkingMemoryKind, *, ref: str | None, content: Any,
            salience: float = 0.0) -> WorkingMemoryEntry: ...
    def recent(self, limit: int = 20) -> list[WorkingMemoryEntry]: ...
    def by_kind(self, kinds: set[WorkingMemoryKind], limit: int = 20) -> list[WorkingMemoryEntry]: ...
    def since(self, scn_tick: int) -> list[WorkingMemoryEntry]: ...
    def within_window(self, ticks: int) -> list[WorkingMemoryEntry]: ...
    def for_agent(self, agent_id: str) -> "WorkingMemorySet": ...  # filtered view
    def size(self) -> int: ...
    def clear(self) -> None: ...  # session end
```

**MemoryTier redefinition:**

```python
# agents/bus.py
class MemoryTier(str, Enum):
    FORMING = "forming"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    # WORKING removed.

_TIER_FORWARD_TRANSITIONS = {
    MemoryTier.FORMING: {MemoryTier.SHORT_TERM, MemoryTier.FORMING},
    MemoryTier.SHORT_TERM: {MemoryTier.LONG_TERM, MemoryTier.SHORT_TERM},
    MemoryTier.LONG_TERM: {MemoryTier.LONG_TERM},
}
```

**ExecAgent gains:**

```python
# agents/exec_agent.py
class ExecAgent:
    def __init__(self, *, agent_id: str, scn: SCN, ...) -> None:
        self.working_memory = WorkingMemorySet(agent_id=agent_id, scn=scn)
```

**Write-interface contract:** MemoryAgent, ReactionBus subscribers, PainBus subscribers, PerceptTraceBuffer, Hippocampus.recall, and ThinkTool all write via `exec.working_memory.add(...)`. No local deques. `StructuredContext` becomes a read-only view computed on demand from `exec.working_memory`.

**Construction order invariant:** `ExecAgent` must be constructed before `MemoryAgent` in every entry path. `agent_factory_canonicalization.md::create_full_agent` already guarantees this for migrated sites. Non-migrated sites (if any remain) get explicit fix-ups in this stage.

**LOC estimate:** ~700-800 (new module + 21+ MemoryTier.WORKING site migrations + prompt builder rewrite from 5+ sections to WorkingMemorySet queries + test updates)

### Stage 2 — ThoughtGate (gate-on-think)

**Goal:** Compose existing `SalienceScorer` + extracted `NoveltyScorer` + new `EnergyGate` into a single `ThoughtGate` that decides whether Exec should deliberate on the current working memory contents.

**G3 from [gating_abstraction.md](gating_abstraction.md) is folded here.** Extract `NoveltyScorer` from `default_network/gate.py` (ThalamicGate) and share it with BioEnrichmentPipeline. Both already conceptually compute novelty; today ThalamicGate does it for vision and BioEnrichment does it for text — same shape, duplicated.

**New module:** `runtime/thought_gate.py`

```python
@dataclass(frozen=True, slots=True)
class ThoughtGateConfig:
    min_combined_score: float = 0.4   # adapts via AdaptiveThresholdController
    min_energy_fraction: float = 0.15  # don't think below 15% energy budget
    refractory_ticks: int = 2          # don't re-fire within N SCN ticks

class ThoughtGate:
    """Composes SalienceScorer + NoveltyScorer + EnergyGate for Exec deliberation.

    Decides: should Exec deliberate on the current working memory contents?
    Returns a GateDecision with score breakdown for provenance.
    """
    def __init__(self, *, salience_scorer: SalienceScorer,
                 novelty_scorer: NoveltyScorer, energy_tracker: LLMEnergyTracker,
                 threshold_controller: AdaptiveThresholdController,
                 config: ThoughtGateConfig) -> None: ...

    def should_think(self, *, working_memory: WorkingMemorySet,
                     context: GatingContext) -> GateDecision: ...

    def record_outcome(self, decision: GateDecision, *,
                       was_useful: bool) -> None:
        """Feed back into AdaptiveThresholdController."""
```

**Composition logic:**
1. If refractory window hasn't elapsed → reject (no deliberation spam)
2. If energy < `min_energy_fraction` of budget → reject (conservation)
3. Score the working memory head against `GatingContext` (current goal, recent percepts, arousal)
4. If `score.combined` < adaptive threshold → reject
5. Otherwise → pass; ExecAgent runs deliberation

**LOC estimate:** ~250 (new module + NoveltyScorer extraction from ThalamicGate + wiring into Exec)

### Stage 3 — Executive Deliberation Loop (contemplate-as-default)

**Goal:** Invert the Exec default. `_contemplate` becomes the primary path; the "skip contemplation for simple goals" branch becomes an optimization, not the default. Working memory is the always-injected deliberation substrate. ThoughtGate is the entry predicate.

**Current shape** (`agents/exec_agent.py:500-612`):
- `propose_intent` → `_build_llm_context` → optional `_contemplate` (only for complex plans) → publish goal
- `_contemplate_standard` / `_contemplate_fast` already implement N-pass critique+refine with convergence detection + hard cap

**New shape:**
- `propose_intent` → `ThoughtGate.should_think(working_memory, context)` → if pass, run `_contemplate` (N hops, max 3, NAc-gated) → converged deliberation produces either a `SpeakIntent`, an `ActionIntent`, or a `NoOpIntent` (continue thinking / silent)
- The `NoOpIntent` case is new: deliberation tick produced no externalizable output. Internal state (working memory, Hebbian bindings from deliberation, thought accumulation) is updated regardless.

**Convergence detection:** Complete the EC pattern_separation wiring that [concept_exploration.md](concept_exploration.md) started — compare current thought against `_recent_thought_keywords` via EC centroid cosine. Converged when separation falls below threshold (same idea being re-thought → stop).

**Hard cap:** Existing `max_deliberation_hops = 3`. Reaching the cap forces emission (SpeakIntent or ActionIntent) — the agent commits rather than loops forever.

**LOC estimate:** ~300 (contemplate promotion, EC wiring, intent kinds)

### Stage 4 — Speech as Converged Action

**Goal:** Silence becomes the default. Speech/text is emitted only when deliberation converges on a `SpeakIntent`. Existing `respond` / `speak` / `say` tools stay imperative; the loop invokes them via the intent-to-tool-dispatch path we already have.

**New intent kind:**

```python
# agents/bus.py
@dataclass(frozen=True, slots=True)
class SpeakIntent(Intent):
    """Emitted by Exec when deliberation converges on 'say something now'."""
    channel: Literal["respond", "speak", "say"]  # maps to existing tool
    content: str
    reason: str             # provenance for review
    deliberation_hops: int  # telemetry
```

**Dispatch:** `LoopController` extended to route `SpeakIntent` to the matching tool. No new tool code — just dispatch.

**Display-layer resilience:** `interactive/display.py` handles silent ticks — the UI already supports showing "agent is thinking" states (interactive_experience_031 shipped the display primitives). This stage tunes the display to not assume "LLM call = visible output."

**Important non-change:** Tools that already produce visible output (write_file, search, etc.) keep doing so. This change is about the speech/text output path specifically, not all externalization.

**Rollback plan:** If silent-by-default is UX-hostile in practice (users feel the agent is broken), the rollback is NOT reverting the architecture — it's adding a `silence_tolerance` config parameter on ThoughtGate that biases toward SpeakIntent after N silent ticks. The architecture stays (deliberation-first), but the threshold adapts to user expectations. This preserves the structural win without the UX cost. The interactive display can show "thinking..." with a progress indicator during silent ticks to set expectations.

**LOC estimate:** ~150 (intent kind, dispatch, display tuning)

### Stage 5 — Observability

**Goal:** Distinguish thinking ticks from action ticks in heartbeat and cost tracking. Prevent the "continuous thinking looks healthy while the loop is wedged on the same thought" failure mode.

**Heartbeat extension:** `runtime/heartbeat.py` (or create if absent) emits per-tick samples distinguishing `{thinking, action, speech, noop}`. Stall detection splits: `MAXIM_ACTION_STALL_S` (no action for N seconds — existing semantic) + new `MAXIM_THINKING_STALL_S` (same thought N ticks in a row → convergence failure, wedged loop).

**Cost breakdown:** `cloud_dispatch.py` tracks cost-per-call; add call-type attribution so users can see `"thinking: 42 calls, $0.18"` separately from `"action: 12 calls, $0.09"`. Enables budget tuning without forking router logic.

**Env vars:**
```
MAXIM_THINKING_STALL_S=45           # wedged-deliberation alert
MAXIM_THOUGHT_COST_CAP=1.00        # optional per-session thinking budget (USD)
```

**LOC estimate:** ~100

### Stage 6 — Re-validation

**Goal:** Re-run behavioral convergence experiments against the new deliberation-first system.

This is not optional. Stages 1-5 fundamentally change how the agent forms and emits responses. The 41/41 experiments from 0.3 were run against the old "propose action, emit output" default. The new system may produce:
- Different action selection patterns (deliberation filters impulsive tool calls)
- Different learning trajectories (more deliberation = more bio-enrichment queries = different hippocampal/NAc stimulation)
- Different prompt composition (WorkingMemorySet view vs. 5 scattered sections)

**Minimum re-validation set:**
- Exp 1 (cross-session affective memory) — substrate still learns across sessions
- Exp 3 (LLM reads learning) — LLM still acts on bio-system context
- Exp 4 (organic learning) — agent still learns from own actions without scripted training

If any experiment regresses: debug the specific stage that caused it before shipping the next stage. Do not proceed on "it probably still works."

**LOC:** ~0 (experiment runs, not code)

### Stage 7 — Use-Based Consolidation (SCN + EC + adaptive threshold)

**Goal:** Close the orphaned WORKING→SHORT_TERM transition — but in the new 3-tier model, this becomes the **SHORT_TERM→LONG_TERM** use-based path. Memories that get accessed during deliberation accumulate score-weighted pressure; when pressure crosses the adaptive threshold, they promote. Decay uses SCN ticks. Context-diversity uses EC pattern-separation.

**`MemoryRecord` gains:**

```python
# memory/types.py
@dataclass
class MemoryRecord:
    # ... existing fields ...
    promotion_pressure: float = 0.0        # accumulates on use
    last_scored_at_tick: int = 0            # for SCN-based decay
    access_contexts: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    # ^ EC-separated context signatures; used for context-diversity check
```

**`Hippocampus.recall` gains:**

```python
def recall(self, query: ..., context: GatingContext | None = None) -> list[MemoryRecord]:
    results = self._existing_retrieval_logic(query)
    for memory in results:
        memory.touch()  # increments access_count
        exec_ref.working_memory.add(  # reconsolidation pull
            kind=WorkingMemoryKind.RECALL,
            ref=memory.id,
            content=memory.summary_for_prompt(),
        )
        if context is not None:
            self._score_and_maybe_promote(memory, context)
    return results

def _score_and_maybe_promote(self, memory: MemoryRecord, context: GatingContext) -> None:
    # Decay existing pressure using SCN ticks
    decay = self._scn.decay_factor(memory.last_scored_at_tick)
    memory.promotion_pressure *= decay

    # Score this access via composite scorer
    score = self._scorer.score(memory, context)

    # Context-diversity via EC pattern separation
    context_sig = self._ec.signature(context)
    is_novel_context = all(
        self._ec.pattern_separation(context_sig, prev) > self._ec.context_threshold
        for prev in memory.access_contexts
    )

    # Only accumulate meaningful pressure when context is diverse
    if is_novel_context:
        memory.promotion_pressure += score.combined
        memory.access_contexts.append(context_sig)

    memory.last_scored_at_tick = self._scn.current_tick

    # Promotion gate
    if memory.tier == MemoryTier.SHORT_TERM:
        threshold = self._promotion_threshold_controller.threshold
        if memory.promotion_pressure >= threshold:
            self._promote_to_long_term(memory)
            self._promotion_threshold_controller.record_outcome(promoted=True)
```

**Promotion rule composition:**

- **Always:** `recall()` calls `touch()` (increments `access_count`)
- **Always:** `recall()` adds reference to `WorkingMemorySet` (the reconsolidation pull — memory is now active)
- **Context-diverse access only:** accumulates `promotion_pressure` via GateScore
- **Threshold crossed:** promotes to LONG_TERM; `AdaptiveThresholdController` records outcome for adaptation
- **Decay:** `promotion_pressure` bleeds off over SCN ticks (so a memory accessed once a week doesn't eventually promote from accumulated trickle)

**FORMING → SHORT_TERM transition** stays outcome-triggered (unchanged behavior — an episode with an outcome is consolidated into short-term storage the same way it used to move to the orphan WORKING tier, just now lands at SHORT_TERM directly).

**Interaction with P8 sleep replay:** Composes cleanly. Wake uses `access_count` + `promotion_pressure` (experience-dependent plasticity). Sleep uses Hebbian replay (edge strengthening on top-N episodes). Separate mechanisms, separate counters, same biological intent.

**Interaction with P6 extinction (Hebbian decay):** `promotion_pressure` decays on SCN-tick schedule; Hebbian edges decay on their own schedule. A memory with high `promotion_pressure` but decayed edges is a legitimate candidate for promotion (it's being actively deliberated on) even if its historical co-activations have faded.

**Content mutation on access:** NOT in this stage. Reconsolidation adds a reference to WorkingMemorySet (read path) but does not mutate the stored episode's content. Full reconsolidation-style content update is deferred to a future research direction; keeping recall read-only for content avoids a whole class of silent-drift bugs.

**LOC estimate:** ~300 (MemoryRecord fields, recall path, promotion logic, SCN decay integration, EC context-diversity wiring)

---

## Migration plan

Seven PRs, one per stage, per L6. Each PR:

1. Opens with the audit table for its stage, filled in with current state.
2. Lands the structural change + producer/consumer migration.
3. Runs the required-checks block from CLAUDE.md before merge.
4. Triggers a pre-merge review round (two parallel Claudes, Executor lens + Architecture lens).
5. Folds cross-confirmed findings in a second commit before PR opens.
6. Updates CLAUDE.md invariants + docs/plans/README.md + writes a memory file if the stage surfaced a new lesson.

**Order (hard dependencies enforced):**

```
Stage 0 (P5 gate)
  │
Stage 1 (WorkingMemorySet + MemoryTier simplification)
  │
  ├── Post-Stage-1 checkpoint: behavioral convergence sanity
  │
Stage 2 (ThoughtGate + G3 NoveltyScorer extraction)
  │
Stage 3 (Executive Deliberation Loop — contemplate-as-default)
  │
Stage 4 (Speech as Converged Action — silence-as-valid-output)
  │
Stage 5 (Observability — thinking vs. action metrics)
  │
Stage 6 (Re-validation — behavioral convergence re-run)
  │
Stage 7 (Use-Based Consolidation — SCN + EC + adaptive promotion)
```

**Rationale for this ordering:** The full deliberation loop (Stages 1-5) should be working end-to-end before adding the consolidation refinement (Stage 7). Stage 7 depends on `Hippocampus.recall()` writing to `WorkingMemorySet` (Stage 1's deliverable) and benefits from the deliberation-driven access patterns established by Stage 3. Re-validation (Stage 6) comes BEFORE Stage 7 so we know the base system is behaviorally sound before adding promotion logic.

---

## Doc + memory refinement

Per L7, each stage ships with doc updates. Catalog:

**CLAUDE.md invariants to revise/add:**
- Revise: *"Memory tier progression is one-way: FORMING → WORKING → SHORT_TERM → LONG_TERM"* → *"Memory tier progression is one-way: FORMING → SHORT_TERM → LONG_TERM. WORKING is not a tier — it's an Exec-owned WorkingMemorySet (active reference layer)."*
- Add: *"WorkingMemorySet is owned by ExecAgent. MemoryAgent, Hippocampus.recall(), ReactionBus subscribers, PainBus subscribers, PerceptTraceBuffer, and ThinkTool all write via the typed `exec.working_memory.add(kind=..., ref=..., content=...)` interface. No code outside this contract maintains a parallel recent-context buffer."*
- Add: *"`Hippocampus.recall()` always calls `memory.touch()` and adds a RECALL entry to the caller's WorkingMemorySet. This is the reconsolidation-pull path. Content is not mutated on access."*
- Add: *"SHORT_TERM → LONG_TERM promotion is pressure-based: each context-diverse access accrues `promotion_pressure` via GateScore; SCN ticks decay pressure; the AdaptiveThresholdController adapts the crossing bar. FORMING → SHORT_TERM is outcome-triggered."*
- Add: *"ThoughtGate composes SalienceScorer + NoveltyScorer + energy budget. Exec runs deliberation only when ThoughtGate.should_think() passes. Silence is a valid tick output; speech is a SpeakIntent produced by deliberation convergence."*

**Memory files:**
- `feedback_working_memory_ownership.md` — document the Option-C decision + the write-interface contract
- `feedback_use_based_consolidation.md` — document the SCN+EC+GateScore composition + the context-diversity guard

---

## Pass criteria

- **Stage 0:** P5 passes. No open regressions.
- **Stage 1:** Zero references to `MemoryTier.WORKING` in `src/maxim/`; zero local deques in `StructuredContext`; `PromptAssembler` assembles from ≤2 `WorkingMemorySet` queries instead of 5+ sections; headless `api.py` path has non-empty working memory after one `run_turn()` call without caller pre-population. Behavioral convergence sanity check passes.
- **Stage 2:** `ThoughtGate.should_think()` is called on every Exec proposal entry; `AdaptiveThresholdController` is wired to record deliberation outcomes; `NoveltyScorer` is a shared module used by both ThalamicGate and BioEnrichment (G3 closed).
- **Stage 3:** `_contemplate` runs on every Exec tick where `ThoughtGate` passes; convergence detection uses EC pattern_separation; hard cap = 3 hops enforced.
- **Stage 4:** `SpeakIntent` exists and is the only path that produces user-visible text from a deliberation tick; silence is a valid tick output with no display-layer warnings.
- **Stage 5:** Heartbeat distinguishes thinking/action/noop; `MAXIM_THINKING_STALL_S` trips on wedged deliberation; cost tracking breaks down by call type.
- **Stage 6:** Behavioral convergence Exp 1, 3, and 4 pass on the new system. Any regression is debugged and resolved before Stage 7.
- **Stage 7:** `Hippocampus.recall()` always calls `touch()` and adds RECALL entries to `WorkingMemorySet`; SHORT_TERM→LONG_TERM promotes based on GateScore-weighted pressure with SCN decay and EC context-diversity; a session running the P2 cascade test shows at least one use-based promotion occurring.

---

## Pre-merge review questions

**For Stage 1 reviewers:**
- Are there any producers the audit missed? Grep for `deque`, `.append`, `recent_*` in `agents/` and `memory/`.
- Is `StructuredContext`-as-view actually cheaper than keeping the deques? If view computation cost is visible, consider a cached view that invalidates on `working_memory.add`.
- Construction order: is there any entry path where `MemoryAgent` could exist before `ExecAgent`? (Post-F5 AgentFactory, this shouldn't be possible.)

**For Stage 4 reviewers:**
- How does silent-by-default interact with the orchestrator's turn expectations? The orchestrator waits for AUT responses to advance the narrative. If the AUT stays silent for 3 ticks, does the orchestrator stall?
- Is `silence_tolerance` the right rollback, or should the orchestrator be aware of NoOpIntent and treat it as a valid "the AUT is still thinking" signal?

**For Stage 7 reviewers:**
- Does the attention-loop guard (EC context-diversity) actually fire on the "agent repeatedly thinks about goal X" case? Add a regression test that deliberates 10 times on the same input and verifies `promotion_pressure` saturates rather than monotonically growing.
- Does SCN-tick decay compose correctly with sleep-replay Hebbian strengthening? Verify no double-counting.
- Is there a path where a SHORT_TERM memory can promote based on a single very-high-salience access? If so, is that the desired behavior or a bug? (Probably desired — emotionally intense single events do consolidate strongly biologically.)

---

## Out of scope

- **Full reconsolidation-style content mutation on access.** Future research direction.
- **Timer-driven "always thinking at 1 Hz"** (Variant A). Event-triggered + ThoughtGate-gated is the shipped model.
- **Multi-agent shared working memory.** Each agent owns its own `WorkingMemorySet` via its Exec.
- **Removing `PerceptTraceBuffer`** — its τ-decay activations are load-bearing for NAc credit assignment via `TraceSnapshot` bindings. It keeps its internal buffer and *also* emits ACTIVATION entries to WorkingMemorySet.
- **Changing the 5-agent pipeline.** Perception, Memory, Exec, Goal, Statistician remain.

---

## Estimated scope

| Stage | LOC | PR size | Migration complexity |
|---|---|---|---|
| 0 | 0 | gate | — |
| 1 | ~700-800 | large | high (21+ sites + prompt builder rewrite) |
| 2 | ~250 | small | low (new module + G3 extraction) |
| 3 | ~300 | medium | medium (Exec entrypoint + convergence wiring) |
| 4 | ~150 | small | low (intent kind + dispatch + rollback plan) |
| 5 | ~100 | small | low (instrumentation only) |
| 6 | ~0 | experiment | — (re-validation runs) |
| 7 | ~300 | medium | medium (recall path + SCN/EC wiring) |
| **Total** | **~1800-1900** | 7 PRs | — |

Comparable in scope to `biosystem_unification.md` Waves 1+2 combined.

---

## Pre-design review findings (2026-04-21, folded before implementation)

Parallel architecture + execution review surfaced 12 findings. Cross-confirmed findings (flagged by both reviewers independently) are marked with ✦.

### Blocking — must resolve before Stage 1 PR opens

**F1 ✦ StructuredContext field survival unspecified.** The plan says "StructuredContext becomes a read-only view" but ~20 fields (knowledge_context, causal_context, valence_context, motor_programs, body_state, autonomy_level, exploration_*, statistical_*) are NOT recent-turn context — they come from parallel subsystem queries in `_run_parallel_memory_queries`. WorkingMemorySet replaces only the **deque-backed surfaces** (recent_percepts, recent_outcomes, detected_speech, conversation_history, cli_inputs). The parallel bio-system queries stay. **Action:** Stage 1 audit must include a field-by-field migration table for all 30 StructuredContext fields. Revise "1-2 queries" claim to "1-2 WorkingMemorySet queries + existing parallel bio-system queries."

**F2 ✦ NoOpIntent + orchestrator hang.** Stage 4 introduces silence as valid output, but the sim bridge's `send_and_wait()` counts action records to detect AUT responses. A silent AUT produces no action records → bridge times out on every turn. **Action:** Pre-Stage-4 design decision required: (A) bridge recognizes NoOpIntent as a valid response, (B) action_sink records a null action marker on NoOp, or (C) LoopController emits a heartbeat ping even on NoOp ticks. The `silence_tolerance` rollback plan in Stage 4 doesn't solve this — it's about *when* to force speech, not about *how* the bridge detects silence.

**F3 ✦ WorkingMemorySet thread safety + producer wiring.** 9 producers (MemoryAgent, PainBus, ReactionBus, PTB, Hippocampus.recall, ThinkTool, etc.) call `add()` from different threads. Plan specifies no locking strategy and no wiring diagram (how each producer gets a reference to `exec.working_memory`). **Action:** (a) Add `threading.Lock` to WorkingMemorySet (serializes `add()`, queries snapshot the deque). (b) Define wiring: dependency injection at construction (pass `working_memory` ref to MemoryAgent, subscribe PainBus/ReactionBus listeners at construction). (c) Add concurrency test: 5 threads × 100 adds, verify all 500 entries present.

**F4. MemoryTier.WORKING persistence migration (data corruption).** Episodes persisted with `tier="working"` in hippocampus JSON will crash on `MemoryTier("working")` after the enum value is removed. **Action:** Add load-time migration in `EpisodicMemory.from_dict()`: `if tier == "working": tier = "short_term"`. Add test: persist episode at WORKING, load post-migration, verify SHORT_TERM. Bump persistence schema version.

**F5. PerceptTraceBuffer dual-write coherence.** PTB keeps its internal ring buffer (NAc learning) AND emits to WorkingMemorySet. PTB decays entries internally (`activation_strength *= decay`); the WMS copy does NOT decay. They diverge. **Action:** Clarify in Stage 1: WMS entries are snapshots (point-in-time salience at add time), NOT live mirrors of PTB state. NAc reads PTB directly (unchanged). Exec reads WMS (snapshot). Two audiences, two surfaces with explicitly different semantics — this is intentional, not an oversight. Document this.

**F6. FORMING → SHORT_TERM transition rule change.** Old rule: FORMING → WORKING (mandatory). New rule: FORMING → SHORT_TERM (direct). Plan doesn't justify why the consolidation step is safe to skip. **Action:** Clarify: FORMING → SHORT_TERM is still outcome-triggered (same as FORMING → WORKING was). The intermediate tier was always a naming fiction — outcomes promote regardless.

### Follow-up — address before respective stage PR

**F7. Construction order not factory-guaranteed.** ExecAgent before MemoryAgent is guaranteed by MaximAgent.__init__ ordering, NOT by the factory. If someone bypasses MaximAgent, the invariant breaks silently. **Action:** Add assertion in ExecAgent.__init__ or WorkingMemorySet construction. Low urgency — `create_full_agent` is the only production door.

**F8. ContextPool semantics lost.** ContextPool has LRU eviction, token-counting, summarization on overflow. WorkingMemorySet has bounded deque + SCN decay. Summarization may be load-bearing for long sessions (token budget). **Action:** Audit ContextPool callers in Stage 1. If summarization is needed, add capacity-overflow callback to WorkingMemorySet.

**F9. Stage 3 EC convergence detection underspecified.** Plan says "EC centroid cosine" but doesn't specify: threshold, update rule for recent centroids, interaction with existing confidence gate (0.7). **Action:** Stage 3 design doc must include pseudocode + examples.

**F10. Stage 7 promotion_pressure decay during sleep.** SCN ticks stop during sleep. Memories accessed 10 ticks ago, then forgotten for 8 hours, decay to zero. Sleep replay strengthens Hebbian edges but promotion_pressure is gone. **Action:** Stage 7 must clarify: either freeze decay during sleep (pause SCN) or accept that sleep replay and use-based promotion are independent mechanisms.

**F11. should_promote() deletion timing.** Between Stage 1 (WORKING removed) and Stage 7 (new promotion logic), there's no SHORT_TERM → LONG_TERM promotion at all. **Action:** Keep should_promote() as dead-but-present code until Stage 7 replaces it. Don't delete in Stage 1.

**F12. StructuredContext deque fields should be deprecated-marked.** After Stage 1, `recent_percepts`, `recent_outcomes` etc. are sourced from WorkingMemorySet. Mark old fields with `# DEPRECATED: sourced from WorkingMemorySet` to prevent re-adding parallel surfaces.
