# PFC Deliberation Cycle — unified think-or-act loop

**Status:** Shell plan (2026-04-22)
**Scope:** ~500-600 LOC net (add ~400, delete ~300 dead code)
**Target version:** 0.8 patch
**Priority:** Highest — deliberation is the missing bio-plausible link
**Depends on:** [working_memory_exec_loop.md](working_memory_exec_loop.md) (shipped), [concept_exploration.md](concept_exploration.md) (L0-L2 shipped), [gating_abstraction.md](gating_abstraction.md) (G0+G1 shipped)
**Supersedes:** [deliberation_observability.md](deliberation_observability.md) (sim_log helpers survive, call sites move)
**Enables:** [goal_depth_integration.md](goal_depth_integration.md) (follow-on: GOAL WMS entries, goal-tagged episodes, NAc goal-outcome learning)
**Gates:** none; architectural unification + behavioral improvement

---

## Problem

The deliberation pipeline is split across two code paths that never run together:

1. **ExecAgent** (`agents/exec_agent.py`) has a full 3-layer pipeline — ThoughtGate → BioEnrichment → LLM → Contemplation — running on a background `_worker_loop` thread. But `on_start()` is never called: not in sim, not in CLI, not in the Reachy runtime. The worker thread never spawns. The pipeline is dead code.

2. **`run_agentic_loop`** (`runtime/agent_loop.py`) has a stripped-down inline enrichment block (section 1.2, lines 633-694) that runs BioEnrichmentPipeline without ThoughtGate, without working memory integration, without contemplation. This is the only path that actually fires.

**Consequence:** The orchestrator wires ThoughtGate + BioEnrichmentPipeline to ExecAgent (orchestrator.py:672-677) AND passes the pipeline to `run_agentic_loop` (line 1272). The ExecAgent wiring is orphaned. The agent gets pre-LLM enrichment in sim, but no gating, no contemplation, and no deliberation cycle in any path.

**Additionally:** BioEnrichmentPipeline + ThoughtGate are sim-only. The CLI non-sim path doesn't construct them. Every bio-system should run in unison — deliberation is not optional when the rest of the bio-stack is active.

## Bio-plausible framing

The prefrontal cortex doesn't fire once and commit. It runs **iterative recurrence**: the thalamic relay surfaces context from memory systems → PFC evaluates → if uncertain, PFC requests more context → evaluates again → when confident, basal ganglia (NAc) gates the motor output. This is a tight loop within a single cognitive cycle, not separate actions spread across turns.

The current architecture inverts this: thinking is an optional tool the LLM calls (`ThinkTool`), and action is the default output of every LLM turn. The plan inverts it back: **deliberation is the default, action is the converged output**.

## Design

### Core: thinking IS the cycle

The key insight: thinking is not a tool the LLM calls — it's the cycle itself. Each iteration, the LLM responds with a standard JSON that includes a `ready_to_act` signal. When the LLM isn't ready, its `reasoning` text becomes the input for the next bio-enrichment round. When it is ready, the action executes.

This eliminates the need for a mini-executor inside the cycle (the old plan's friction point #3). No tool-call indirection — the LLM's own reasoning text feeds directly back through hippocampus, NAc, ATL, EC.

### LLM call model: blocking poll within the cycle

`LLMWorker.submit_context()` is non-blocking (submits to WorkerPool, returns immediately). Results are polled via `get_latest_proposal()`. The deliberation cycle needs synchronous results — it can't proceed to cycle 2 without cycle 1's response.

Solution: a blocking poll wrapper inside the cycle:

```python
def _wait_for_proposal(
    llm_worker: LLMWorker,
    stop_event: threading.Event,
    timeout: float = 300.0,
) -> LLMProposal | None:
    """Block until LLM responds, checking stop_event every 100ms."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop_event.is_set():
            return None
        proposal = llm_worker.get_latest_proposal()
        if proposal is not None:
            return proposal
        time.sleep(0.1)
    return None
```

This blocks the main thread but checks `stop_event` every 100ms — preemption and cancellation still work. The agentic loop already effectively blocks (it polls each iteration and doesn't do meaningful work while waiting for the LLM). The cycle makes the blocking explicit. ~10 lines, no new infrastructure.

**Why not peer mesh offload:** Routing deliberation LLM calls through the peer mesh would add network protocol work, error handling, and a new API surface to solve a local control flow issue. Interesting as a future "deliberation offload" mesh capability, but scope creep for PFC launch.

**Why not a state machine across iterations:** Structuring the cycle as states (DELIBERATING_CYCLE_1, DELIBERATING_CYCLE_2...) that span multiple loop iterations is architecturally cleaner (loop never blocks), but significantly more complex: requires buffering percepts during deliberation, managing cycle state across iterations, and handling preemption mid-cycle. The blocking poll is simpler and correct; the state machine is a future optimization if blocking proves problematic in practice.

```python
def _run_deliberation_cycle(
    *,
    percept_text: str,
    thought_gate: ThoughtGate | None,
    bio_enrichment: BioEnrichmentPipeline | None,
    working_memory: WorkingMemorySet,
    context: StructuredContext,
    llm_worker: LLMWorker,
    nac: NAc | None,
    max_cycles: int = 3,
    agent_id: str | None = None,
) -> LLMProposal | None:
    """PFC deliberation cycle — think-or-act loop.

    Each cycle:
      1. ThoughtGate evaluates working memory salience/novelty
      2. If gate rejects: one-shot LLM call (no enrichment)
      3. If gate fires: cycle 1 ALWAYS enriches (gate-fired
         minimum — local models get enriched one-shot even if
         they never produce ready_to_act: false)
      4. Enrichment added to working memory as THOUGHT entry
      5. LLM called with enriched context
      6. If ready_to_act == true (or absent): return proposal
      7. If ready_to_act == false: feed reasoning text back
         through bio-enrichment, loop (PFC recurrence)
      8. Each cycle replaces bio_enrichment_context (not appends);
         WM accumulates naturally via THOUGHT entries
      9. If max_cycles reached or convergence detected
         (Jaccard >= 0.8): force action from best response

    The LLM decides when to act via ready_to_act — not a
    confidence threshold from a separate critique pass. NAc
    learns the optimal cycle count over time.
    """
```

### Cycle flow

```
percept arrives
    │
    ▼
┌─────────────────────────────────────┐
│ ThoughtGate.should_think()          │ ◄── salience × novelty × energy
│   not passed → one-shot LLM call   │     (no enrichment, ready_to_act
│                                     │      defaults true)
│   passed → enter deliberation cycle │
└─────────────┬───────────────────────┘
              │
              ▼  cycle 1 (ALWAYS enriches when gate fires)
┌─────────────────────────────────────────────────┐
│ BioEnrichmentPipeline.enrich(percept_text)      │
│   memories, predictions, concepts,              │
│   affordances, recent_context                   │
│        │                                        │
│        ▼                                        │
│ working_memory.add(kind=THOUGHT, enrichment)    │
│ context.bio_enrichment_context = formatted      │
│        │                                        │
│        ▼                                        │
│ LLM call (sees accumulated WM + enrichment)     │
│        │                                        │
│        ├── ready_to_act == true (or absent)     │──► return proposal
│        │   (local models land here — enriched   │    (minimum viable
│        │    one-shot, the gate already decided   │     deliberation)
│        │    this input was worth enriching)      │
│        │                                        │
│        ├── ready_to_act == false                 │
│        │   reasoning: "the guard might have..." │──► cycle 2+
│        │                                        │
│        └── no response / IDLE                   │──► return None
└─────────────────────────────────────────────────┘
              │
              ▼  cycle 2..N (max 3 sim, max 2 interactive)
┌─────────────────────────────────────────────────┐
│ BioEnrichmentPipeline.enrich(LLM reasoning)     │
│   the LLM's own words drive the next round      │
│   of memory/prediction associations             │
│        │                                        │
│        ▼                                        │
│ working_memory.add(kind=THOUGHT, enrichment)    │
│ context.bio_enrichment_context = replaced       │
│   (each cycle replaces, not appends —           │
│    WM accumulates via THOUGHT entries)          │
│        │                                        │
│        ▼                                        │
│ LLM call (sees prior reasoning + new enrichment)│
│        │                                        │
│        ├── ready_to_act == true                  │──► return proposal
│        │                                        │
│        ├── ready_to_act == false                 │
│        │   convergence check (Jaccard ≥ 0.8)    │──► loop or force
│        │   energy check                         │    action if
│        │   text = reasoning (for next enrich)   │    converged/max
│        │                                        │
│        └── no response / IDLE                   │──► return None
└─────────────────────────────────────────────────┘
              │
              ▼  (after action committed)
┌─────────────────────────────────────┐
│ NAc.observe("deliberation:N_cycles",│
│             outcome_valence)        │
│                                     │
│ Over time, NAc learns whether 1-    │
│ cycle or 3-cycle decisions produce  │
│ better outcomes. ThoughtGate's      │
│ adaptive threshold adjusts.         │
└─────────────────────────────────────┘
```

### The LLM response format

The existing JSON schema (`action: {tool_name, params}`, `reasoning`, `confidence`) gains one field — `ready_to_act`:

```json
{
  "reasoning": "The guard is sleeping but I notice keys on his belt...",
  "ready_to_act": false
}
```

vs. when the LLM has deliberated enough:

```json
{
  "reasoning": "Guard is asleep, door is 3m away, keys aren't needed — the door is unlocked.",
  "ready_to_act": true,
  "action": {"tool_name": "base_humanoid_move", "params": {"direction": "door", "speed": "slow"}},
  "confidence": 0.85
}
```

**Priority rules for edge cases:**
- `ready_to_act` wins over `action` presence. If `ready_to_act: false` but `action` is present, the cycle loops (the LLM is uncertain despite proposing an action — enrichment may change its mind).
- `ready_to_act: true` (or absent) with no `action` → return `None` (IDLE/no-op). The LLM thinks it's ready but has nothing to do.
- `ready_to_act` absent → defaults to `true` (backward compatible one-shot).

**Parsing:** `ready_to_act` is extracted in `_process_request` alongside existing fields. `json_parser.py` preserves all dict keys — no risk of silent field dropping. Add `ready_to_act` to `LLMProposal` as `ready_to_act: bool = True`.

When `ready_to_act` is false, the cycle:
1. Takes the `reasoning` text
2. Feeds it through BioEnrichmentPipeline (the LLM's own words drive the next round of associations)
3. Adds enrichment to working memory
4. Calls the LLM again — it now sees its prior reasoning + the bio-system response

This is PFC recurrence: the LLM thinks → bio-systems respond with memories and predictions about *what the LLM just thought* → the LLM sees those associations → thinks more precisely → commits to action.

**Forced action on max cycles:** If the LLM says `ready_to_act: false` on all cycles with no `action` in any response, the cycle returns `None` (IDLE/no-op). The cycle does NOT synthesize an action — if the LLM couldn't decide after 3 enrichment rounds, forcing an action is worse than waiting. The next percept triggers a fresh cycle.

### Prompt framing: deliberation as identity

The PFC cycle requires a prompt preamble that frames the agent as a deliberative entity. This goes into the system prompt at `SectionPriority.IMPORTANT` (before tool descriptions, after role/goal — supplementary framing, not a hard operational constraint):

```
You are a thoughtful agent. Before acting, you reflect on each situation
using your experience. Your bio-systems will surface relevant memories,
predictions, and associations — use them.

Set "ready_to_act" to true ONLY when your next step requires: calling a
tool, speaking to someone, or moving. If you are still gathering context
from your memories and associations, set "ready_to_act" to false and
explain your reasoning — your reasoning will be enriched with additional
associations from your experience.
```

This goes in `agents/exec_prompts.py` as `PFC_PREAMBLE` and is injected by the prompt builder at `SectionPriority.IMPORTANT`.

**Important for local models:** The preamble uses a concrete trigger checklist ("tool, speak, move") rather than abstract principles. Vague framing ("understand reality through thinking") causes 14B models to philosophize instead of act. The checklist gives the LLM a mechanical decision rule: "does my response contain a tool call or speech act? No → `ready_to_act: false`."

### Working memory THOUGHT rendering in prompt

**Critical gap identified by review:** THOUGHT entries added to WorkingMemorySet during the cycle are invisible to the LLM — `PromptBuilder` has no section that renders WMS entries. Without this, the LLM never sees its prior reasoning or the bio-system associations from earlier cycles. The recurrence loop is broken.

**Fix:** Add a `_add_working_memory_section()` method to `PromptBuilder` that renders recent THOUGHT entries from the `WorkingMemorySet`. This section:

- Renders `working_memory.recent(kind=THOUGHT, limit=6)` — the last 6 THOUGHT entries (covers 2 full 3-cycle deliberations)
- Each entry shows: the enrichment summary (not the full enrichment text — that's in `bio_enrichment_context` for the current cycle)
- Priority: `SectionPriority.IMPORTANT`, truncatable, placed after `bio_enrichment` in the perception group
- **Token budget:** 400 tokens max, enforced by `budgeter.add(..., max_tokens=400, truncatable=True)`. THOUGHT entries are summaries (~50-80 tokens each), so 6 entries fit comfortably. The budgeter drops this section before CRITICAL sections if the context window is tight.
- The bio_enrichment_context section shows the *current* cycle's enrichment; the working_memory_thoughts section shows *prior* cycles' enrichment. No duplication.

**StructuredContext integration:** Add `working_memory_thoughts: list[str] | None = None` to `StructuredContext`. The deliberation cycle populates this from `working_memory.recent(kind=THOUGHT)` before each LLM call. `PromptBuilder._add_working_memory_section()` reads it.

### ThinkTool's role changes

ThinkTool stays in the registry but is **demoted from cycle driver to explicit multi-turn tool:**

- **Inside a deliberation cycle**: ThinkTool is NOT used. The cycle handles deliberation natively via `ready_to_act`. If the LLM happens to call `think` during a cycle, the tool result enrichment still works, but the cycle doesn't depend on it.
- **Outside a cycle (multi-turn)**: ThinkTool remains useful for explicit reasoning between action turns — "let me think about the long-term plan." Its convergence detection, hop counter, and bio-enrichment integration are unchanged.

### One-shot fast path

When ThoughtGate rejects (familiar input, low salience, refractory window, low energy), the cycle degenerates to a single LLM call with no enrichment. `ready_to_act` defaults to `true` (backward compatible). This is the common case for routine inputs — the agent doesn't waste cycles thinking about "hello."

### Gate-fired minimum: one enrichment round guaranteed

When ThoughtGate fires, cycle 1 **always enriches** regardless of the LLM's `ready_to_act` value. The gate already decided the input is novel/salient enough to warrant enrichment — the LLM's readiness signal controls whether cycle 2+ runs, not whether cycle 1 enriches. This is the safety net for local models (7B-14B) that may not reliably produce `ready_to_act: false`: they still get one enrichment round on novel inputs (the minimum viable deliberation), and act with the enriched context. The cycle flow becomes:

1. ThoughtGate fires → enrich percept → LLM call 1 (always enriched)
2. If `ready_to_act == false` → enrich LLM reasoning → LLM call 2
3. If `ready_to_act == false` → enrich again → LLM call 3 (max, forced action)
4. If `ready_to_act == true` (or absent) at any point → return proposal

Local models that never produce `ready_to_act: false` get step 1 (enriched one-shot). Claude and capable models that signal uncertainty get steps 1-3 (full recurrence).

### Latency mitigation

Each deliberation cycle is an LLM call (~10-15s on local 14B). Three strategies prevent this from hurting interactive responsiveness:

1. **ThoughtGate as the primary gate.** Novel/salient inputs get deliberation, routine inputs get one-shot. Most turns are one-shot.

2. **Interactive mode cap.** `max_cycles` defaults to 3 in sim, 2 in interactive mode. The agent still deliberates but commits faster when a human is waiting.

3. **Context thread (future).** A persistent enrichment context that carries bio-system associations across turns, so the agent doesn't need to re-derive the same associations every cycle. If the dungeon scenario was enriched on turn 1, the "guard + keys + stealth" associations persist in working memory and don't need another enrichment cycle on turn 2. This is already partially solved by WorkingMemorySet — THOUGHT entries from prior turns are visible to the LLM. Full context threading (summarizing prior enrichment for the next turn's cycle) is a follow-on optimization, not a launch blocker.

### Where the cycle lives in the agentic loop

```python
# runtime/agent_loop.py — inside the main loop iteration

# 1. PERCEPTION (existing)
observation = sim.next_observation(environment, default_network)
state.update(observation)

# 1.1 IMAGINATION (existing)
# ...

# 1.2 DELIBERATION CYCLE (replaces inline bio-enrichment block)
if _has_percept_text and llm_worker is not None:
    proposal = _run_deliberation_cycle(
        percept_text=_percept_text,
        thought_gate=thought_gate,
        bio_enrichment=bio_enrichment_pipeline,
        working_memory=_working_memory,
        context=context,
        llm_worker=llm_worker,
        nac=nac,
        max_cycles=2 if interactive_mode else 3,
        agent_id=agent_id,
    )
    if proposal is not None:
        ctrl.pending_proposal = proposal

# 2. PROPOSAL EXECUTION (existing)
# ...
```

## What changes

### Stage 1: Wire ThoughtGate + BioEnrichment as non-optional

**Principle:** All bio-systems run in unison. Deliberation is not optional when the bio-stack is active.

| File | Change |
|------|--------|
| `runtime/bio_stack.py` | Add `thought_gate` and `bio_enrichment_pipeline` fields to `BioStack` dataclass. Construct both in `build_bio_stack()` — not in try/except (they depend only on EC + NAc + Hippocampus which are already required). |
| `simulation/orchestrator.py` | Remove the sim-only BioEnrichmentPipeline + ThoughtGate construction (lines 650-682). Read them from `BioStack` instead. Remove dead `wire_thought_gate()` / `wire_bio_enrichment()` calls to ExecAgent. |
| `cli.py` | Pass `thought_gate` and `bio_enrichment_pipeline` from BioStack to `run_agentic_loop`. |
| `runtime/agent_loop.py` | Add `thought_gate` parameter (currently missing). |

**Stage 1 ships and soaks before Stage 2 begins.** Run the full test suite + a sim to confirm all entry points construct and pass ThoughtGate + BioEnrichmentPipeline from BioStack without regressions. Stage 1 is a wiring change — it should not alter runtime behavior (the gate and pipeline already exist in sim, they're just sourced from BioStack now and available in non-sim paths).

### Stage 2: Implement the deliberation cycle

| File | Change |
|------|--------|
| `runtime/agent_loop.py` | Extract `_run_deliberation_cycle()` function. Replace inline bio-enrichment block (section 1.2) with cycle call. Add `nac`, `working_memory`, `thought_gate` parameters. Convergence detection extracted from ThinkTool into shared utility. |
| `agents/exec_prompts.py` | Add `PFC_PREAMBLE` template — deliberation-as-identity framing for the system prompt. |
| `agents/prompt_builder.py` | Inject `PFC_PREAMBLE` at `SectionPriority.IMPORTANT` when bio-stack is active. Add `ready_to_act` to the expected JSON response schema. Add `_add_working_memory_section()` for THOUGHT entries (400 token budget, truncatable). |
| `simulation/sim_logger.py` | `sim_pre_deliberation()` and `sim_contemplation()` already exist — call them from within the cycle function (single call site, not dual). |

### Stage 3: Delete dead ExecAgent deliberation code

| File | Change |
|------|--------|
| `agents/exec_agent.py` | Delete `_run_pre_deliberation()`, `_maybe_contemplate()`, `_contemplate()`, `_contemplate_standard()`, `_contemplate_fast()`, `_critique_plan()`, `_refine_plan()`, `_worker_loop()`, `on_start()` worker thread management, `wire_thought_gate()`, `wire_bio_enrichment()`. **Keep** `propose_intent()` and `deliberate()` — these are intent classification ("is this a speak, act, or no-op?"), a separate concern from the decision-making pipeline. The agentic loop already calls `propose_intent` after getting a response (agent_loop.py:173, 1043); that call chain is unchanged. ExecAgent becomes a thin intent classifier with no deliberation responsibility — "executive function" delegates the thinking-loop to the agentic loop where it belongs. |
| `agents/exec_prompts.py` | Delete `CRITIQUE_SYSTEM_TEMPLATE`, `CRITIQUE_USER_TEMPLATE`, `FAST_CONTEMPLATE_SYSTEM_TEMPLATE`, `FAST_CONTEMPLATE_USER_TEMPLATE`, `REFINE_SYSTEM_TEMPLATE`, `REFINE_USER_TEMPLATE`. |

### Stage 4: NAc outcome learning for cycle count

| File | Change |
|------|--------|
| `runtime/agent_loop.py` | Before LLM call in cycle: `event_id = nac.record_event(event_type="deliberation", event_signature=f"deliberation:{n_cycles}_cycles")`. After action execution (when outcome is known): `nac.record_outcome(event_id=event_id, valence=outcome_valence)`. Uses direct event_id attribution, NOT `record_outcome_full` (context similarity would cross-match tool and deliberation pending events). |

## What stays the same

- **ThoughtGate** — same gating logic, just called from the agentic loop instead of ExecAgent.
- **BioEnrichmentPipeline** — same enrichment, same `EnrichmentResult`, same formatting.
- **ThinkTool** — still in the registry for explicit multi-turn reasoning. Demoted from cycle driver to standalone tool. Bio-enrichment integration unchanged.
- **WorkingMemorySet** — same THOUGHT entries, same accumulation. Deliberation rounds add to it naturally.
- **StructuredContext.bio_enrichment_context** — same injection point into the prompt.
- **sim_pre_deliberation() / sim_contemplation()** — same helpers, just called from one place.
- **`api.py` headless with `learning=False`** — still valid opt-out. When bio-stack is None, no deliberation cycle runs.

## Key constraints

1. **The LLM decides when to act via `ready_to_act`, not a confidence threshold.** ExecAgent's old contemplation used a separate critique LLM call + confidence score to decide whether to refine. The new cycle lets the LLM's own response signal readiness. No tool-call indirection, no mini-executor, no extra critique pass. The LLM's uncertainty drives deliberation naturally.

2. **The LLM's reasoning text IS the enrichment input.** When `ready_to_act` is false, the `reasoning` field is fed back through BioEnrichmentPipeline. This is PFC recurrence: the agent's own thoughts drive the next round of memory/prediction associations. Both percept text (cycle 1) and LLM-generated text (cycles 2+) flow through the same enrichment path.

3. **One-shot is the fast path.** ThoughtGate rejection → single LLM call, no enrichment, `ready_to_act` defaults to `true`. The cycle adds latency only when the gate fires (novel/salient input).

4. **Max cycles: 3 in sim, 2 in interactive.** Hard cap prevents runaway deliberation. Interactive cap is lower because a human is waiting. NAc learns whether the extra cycles are worth it.

5. **Bio-enrichment uses `bypass_gate=True` inside the cycle.** The ThoughtGate already decided the input is worth enriching. The pipeline's internal novelty gate should not re-reject it.

5b. **ThoughtGate refractory resets after the cycle completes, not at gate-pass time.** The current tick-based refractory (`current_tick - last_pass_tick < refractory_ticks`, default 2) creates a problem: the deliberation cycle runs multiple LLM calls within one tick, so `last_pass_tick` is set at cycle start and the next 2 percepts after deliberation are auto-rejected regardless of novelty. Fix: `_run_deliberation_cycle` calls `thought_gate.reset_refractory(current_tick)` after the cycle completes, so the refractory window starts counting from when deliberation *finished*, not when it started. Add `reset_refractory(tick: int)` to ThoughtGate — single-line method that sets `_last_pass_tick = tick` under the existing lock. The refractory still prevents rapid re-triggering (2 ticks after completion), but doesn't penalize percepts that arrived during a long deliberation.

6. **Convergence detection uses a local Jaccard check, not a shared utility.** Jaccard similarity ≥ 0.8 across reasoning texts → force `ready_to_act=true` on the next cycle. The 5-line Jaccard math is copied as a local helper in `_run_deliberation_cycle`, NOT extracted into a shared module with ThinkTool. Reason: ThinkTool's convergence operates on `_recent_thought_keywords` (rolling deque across tool calls, session-lifetime), while the cycle's convergence operates on reasoning texts within a single turn (2-3 iterations, turn-lifetime). Different data structures, lifetimes, and reset semantics. The abstraction would be more complex than the duplicated math.

7. **`ready_to_act` is backward compatible.** If absent from the LLM response, defaults to `true`. Old prompts and models that don't know about the field work as one-shot (no deliberation cycle).

## Observability

The `sim_pre_deliberation()` and `sim_contemplation()` helpers from [deliberation_observability.md](deliberation_observability.md) are wired inside `_run_deliberation_cycle()`:

- **Cycle start**: `sim_pre_deliberation(gate_passed, score, threshold, enrichment_sections)` — shows whether the gate fired and how many bio-system sections contributed.
- **Each think iteration**: `sim_pre_deliberation(gate_passed=True, ..., enrichment_sections=N)` — shows accumulating context.
- **Cycle end (action)**: `sim_contemplation(gate_passed=True, refined=N>1, score)` — shows whether the agent deliberated and how many cycles it took.
- **Cycle end (gate rejected)**: `sim_contemplation(gate_passed=False, ...)` — one-shot path.

Terminal output at `--display bio`:
```
  12.34s [THOUGHT     ] [AUT] pre-deliberation: gate passed (score=0.72 >= 0.40), 3 enrichment section(s)
  12.38s [THOUGHT     ] [AUT] pre-deliberation: cycle 2, 2 enrichment section(s) (convergence=0.45)
  12.42s [DELIBERATION] [AUT] deliberation converged after 2 cycles (score=0.72)
```

## Validation

1. Run the dungeon escape sim and confirm `[THOUGHT]` + `[DELIBERATION]` lines appear on every turn with real scores:
   ```bash
   PYTHONPATH=src python -m maxim --sim "You are trapped in a dungeon with a sleeping guard. Escape quietly." --interactive false --sim-max-turns 6 --display bio
   ```

2. Confirm CLI non-sim path also produces deliberation traces:
   ```bash
   PYTHONPATH=src MAXIM_LOG_FILE=/tmp/maxim.jsonl maxim --llm mistral-7b
   # Send a message, check /tmp/maxim.jsonl for THOUGHT subsystem events
   ```

3. Run the full test suite:
   ```bash
   ruff check src/ tests/ && ruff format src/ tests/
   python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
   ```

4. Run a behavioral convergence check (Exp 1 from behavioral_convergence_practice.md) to verify the migration doesn't regress agent behavior.

## Files to touch

| File | Change | LOC |
|------|--------|-----|
| `runtime/bio_stack.py` | Add `thought_gate`, `bio_enrichment_pipeline` to BioStack | +30 |
| `runtime/agent_loop.py` | `_run_deliberation_cycle()` + `_wait_for_proposal()` functions, replace section 1.2, add parameters, local Jaccard convergence helper, refractory reset | +240, -50 |
| `agents/exec_prompts.py` | Add `PFC_PREAMBLE`; delete dead critique/refine templates | +15, -80 |
| `agents/prompt_builder.py` | Inject PFC preamble at `SectionPriority.IMPORTANT`; add `ready_to_act` to response schema; add `_add_working_memory_section()` (THOUGHT rendering, 400 token budget) | +50 |
| `agents/llm_worker.py` | Extract `ready_to_act` from response JSON into `LLMProposal.ready_to_act: bool` | +5 |
| `agents/bus.py` | Add `working_memory_thoughts: list[str] | None` to `StructuredContext` | +3 |
| `runtime/thought_gate.py` | Add `reset_refractory(tick)` method | +5 |
| `simulation/orchestrator.py` | Remove dead ExecAgent wiring, read from BioStack | +10, -30 |
| `cli.py` | Pass new params to `run_agentic_loop` | +5 |
| `agents/exec_agent.py` | Delete dead deliberation code (keep `propose_intent`, `deliberate`) | -250 |
| `interactive/display.py` | No change (log lines via sim_logger are sufficient for PFC launch; dedicated thinking panel deferred to [interactive_display_overhaul.md](interactive_display_overhaul.md)) | 0 |
| `simulation/sim_logger.py` | No change (helpers already written) | 0 |
| **Net** | | **~+380, -410 = -30** |

## Known friction points

1. **Prompt deduplication.** Cycle 1 enriches the percept; cycles 2+ enrich the LLM's reasoning. Both produce THOUGHT entries in working memory and set `bio_enrichment_context`. Each cycle **replaces** (not appends to) the enrichment context — this is now explicit in the cycle flow diagram. Working memory accumulates naturally via THOUGHT entries; the prompt builder already renders those. The LLM sees: current enrichment (one section) + all prior THOUGHT entries (working memory).

2. **NAc attribution delay.** The cycle records deliberation metadata to NAc after action execution. But the outcome (success/failure) may not arrive until several turns later. Wire this through `nac.record_event(event_type="deliberation", event_signature="deliberation:N_cycles")` / `nac.record_outcome(event_id=...)` with direct event_id attribution — NOT `record_outcome_full` (context similarity), because the shared `_pending_events` list contains both tool and deliberation events, and context-similarity matching could cross-match between types. The cycle count context is stored on the pending event so it's available at outcome time. Note: `record_tool_start` does not exist on NAc; the correct API is `record_event()`.

3. **GoalAgent + MemoryAgent bus subscriptions.** Both subscribe to `ProposedGoal` on the agent bus (agentic_goal_agent.py:74, memory_agent.py:182). ExecAgent's `_worker_loop` was the publisher, but it never ran. Verify these subscriptions are truly dead before deleting the worker. If GoalAgent needs proposals, wire them from the deliberation cycle's return value through the existing `ctrl.pending_proposal` path, not the bus.

4. **`ready_to_act` with local models.** Small local models (7B-14B) may not reliably produce the `ready_to_act` field even when prompted. Solved by the "gate-fired minimum" invariant (see above): when ThoughtGate fires, cycle 1 always enriches regardless of `ready_to_act`. Local models get enriched one-shot behavior on novel inputs. The prompt uses a concrete trigger checklist ("tool, speak, move") rather than abstract principles to maximize the chance of local models producing the field. Models that never produce `ready_to_act: false` still get the minimum viable deliberation.

5. **Interactive latency visibility.** When the cycle runs 2 iterations in interactive mode (~20-30s), the user sees nothing. For PFC launch: deliberation cycle events are logged to the agent log panel at BIO tier via existing `sim_pre_deliberation()` / `sim_contemplation()` helpers — this is sufficient to show progress and prevent the stall detector from firing. A dedicated thinking panel with agent focus switching and dynamic resize is designed in the follow-on [interactive_display_overhaul.md](interactive_display_overhaul.md) plan, which ships after PFC.

## Relationship to existing plans

- **Supersedes** [deliberation_observability.md](deliberation_observability.md) — the sim_log helpers survive but the call sites move into the unified cycle.
- **Completes** [working_memory_exec_loop.md](working_memory_exec_loop.md) — that plan shipped the WorkingMemorySet and ThoughtGate. This plan makes them load-bearing by routing all deliberation through them.
- **Composes with** [concept_exploration.md](concept_exploration.md) L0-L2 — BioEnrichmentPipeline and ThinkTool are unchanged; they're called from the cycle instead of from ExecAgent / inline code.
- **Enables** [goal_depth_integration.md](goal_depth_integration.md) — once the PFC cycle is the canonical deliberation site, goal-depth (GOAL WMS entries, goal-tagged episodes, NAc goal-outcome learning) enriches the data the cycle sees without changing cycle logic.
- **Enables** [interactive_display_overhaul.md](interactive_display_overhaul.md) — dedicated thinking panel, agent focus switching, dynamic resize. Ships after PFC; uses PFC deliberation events as content source.
- **Informs** [cross_session_sim_validation.md](cross_session_sim_validation.md) — once deliberation is always-on, cross-session validation can measure whether enrichment improves across sessions.

## Pre-implementation review findings (2026-04-22)

Two-lens parallel review (concurrency/state + prompt/LLM-behavior). All findings folded into the plan above.

**Critical (resolved):**
- C1: LLMWorker is async submit/poll, cycle assumed sync → added `_wait_for_proposal` blocking poll wrapper
- C2: JSON examples used top-level `tool_name` instead of `action: {tool_name, params}` → fixed to match actual schema
- C3: THOUGHT entries invisible to LLM (PromptBuilder has no WMS section) → added `_add_working_memory_section()` with 400 token budget

**High (resolved):**
- H1: ThoughtGate tick-based refractory penalizes percepts arriving during deliberation → added `reset_refractory(tick)` call after cycle completes
- H2: No priority rule for `ready_to_act` vs `action` presence → specified: `ready_to_act` wins; `true` with no action → IDLE
- H3: "Force action from best response" undefined → specified: return `None` (no-op), don't synthesize
- H4: Plan referenced non-existent `record_tool_start` on NAc → corrected to `record_event()` with direct event_id attribution

**Low (noted, no plan change needed):**
- L1: THOUGHT entries could dominate WMS `recent(limit=20)` window, pushing out PERCEPT/OUTCOME entries
- L2: Acting Coach "Don't ask permission" vs PFC preamble "only act when needed" — different concerns (exploration vs timing), not contradictory. Monitor on local models.
- L3: BioEnrichmentPipeline novelty self-suppression on rapid calls — arguably correct (similar thoughts score lower novelty)
