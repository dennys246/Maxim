# File Splitting Plan

Detailed plans for splitting the largest files in the codebase into focused, maintainable modules. Each split preserves the public API via re-exports from the original module path or a new `__init__.py`.

---

## 1. Split `agents/bus.py` (1,231 LOC → package)

**Current state:** 9 enums, 23 dataclasses, 1 generic class (DependencyGraph, 354 LOC), 1 pub/sub class (AgentBus, 34 LOC) — all in one file.

**Problem:** Unrelated concepts (graph algorithms, memory tiers, tool errors, planning events) are coupled by file proximity. Finding anything requires scrolling through 1200 lines.

### Target structure: `agents/bus/`

```
agents/bus/
  __init__.py          # Re-exports everything (backward compat)
  enums.py             # ~85 LOC — 7 enums
  percept.py           # ~60 LOC — Percept dataclass
  memory_types.py      # ~170 LOC — MemoryItem, WorkingMemoryEntry[T]
  context.py           # ~110 LOC — StructuredContext, PlanProgressContext
  goals.py             # ~140 LOC — SubGoal, ProposedGoal, GoalAccepted, GoalCompleted
  execution.py         # ~50 LOC — ToolCall, ToolResult
  plan_events.py       # ~70 LOC — PlanCreated/Started/Completed/Restored/ReplanRequested
  stream.py            # ~50 LOC — StreamEvent, LoopTerminated, StatisticalInsight/Summary
  graph.py             # ~370 LOC — Edge, DependencyGraph[T]
  agent_bus.py         # ~40 LOC — AgentBus class
```

### Split rationale (by import frequency)

| Module | Key symbols | Import count | Why separate |
|--------|-------------|-------------|--------------|
| `enums.py` | GoalPriority, MemoryTier, SubGoalStatus, FailureStrategy, EdgeType, StopReason, ToolErrorKind | 33 total | Highest import volume; used everywhere; zero dependencies |
| `percept.py` | Percept | 13 files | Core data type, imported by perception, memory, salience, default_network |
| `memory_types.py` | MemoryItem, WorkingMemoryEntry | 10 files | Memory-specific, depends only on enums |
| `graph.py` | Edge, DependencyGraph | 6 files | Self-contained generic data structure; complex (354 LOC) |
| `context.py` | StructuredContext, PlanProgressContext | 6 files | Agent context contract, large dataclass (28 fields) |
| `goals.py` | SubGoal, ProposedGoal, GoalAccepted, GoalCompleted | 7 files | Planning domain types |
| `agent_bus.py` | AgentBus | 6 files | Pub/sub mechanism, tiny but foundational |
| `execution.py` | ToolCall, ToolResult | 2 files | Tool execution events, minimal |
| `plan_events.py` | PlanCreated, PhaseStarted, etc. | 0-1 files each | Internal planning events, rarely imported directly |
| `stream.py` | StreamEvent, LoopTerminated, Statistical* | 1-3 files | Streaming/monitoring events |

### Migration strategy

1. Create `agents/bus/` directory
2. Move symbols to their new files
3. Create `agents/bus/__init__.py` that re-exports everything:
   ```python
   from maxim.agents.bus.enums import *
   from maxim.agents.bus.percept import *
   from maxim.agents.bus.memory_types import *
   # ... etc
   ```
4. All existing `from maxim.agents.bus import X` imports continue to work unchanged
5. Over time, update imports to point at specific submodules (optional, not required for correctness)

### Internal dependency order

```
enums.py          ← no internal deps
percept.py        ← no internal deps (uses stdlib dataclass + fields)
memory_types.py   ← imports MemoryTier from enums
context.py        ← imports MemoryItem from memory_types (for type hints)
goals.py          ← imports SubGoalStatus, FailureStrategy, GoalPriority from enums
execution.py      ← imports ToolErrorKind from enums
plan_events.py    ← no internal deps
stream.py         ← imports StopReason from enums
graph.py          ← imports EdgeType from enums
agent_bus.py      ← no internal deps (pure pub/sub)
```

No circular dependencies. Clean DAG.

---

## 2. Split `runtime/agent_loop.py` (2,629 LOC → focused modules)

**Current state:** Two god functions — `run_agent_loop()` (231 LOC, the simple canonical loop) and `run_agentic_loop()` (2,332 LOC, the full LLM-integrated loop). The agentic loop has 9 major sections, 3 nested helper functions, and handles perception, LLM polling, execution, approval workflows, memory capture, and cleanup.

**Problem:** `run_agentic_loop()` is too large to reason about. Changes to one section (e.g., perception handling) risk breaking unrelated sections (e.g., approval workflows). Testing individual sections in isolation is impossible.

### Target structure: `runtime/`

```
runtime/
  agent_loop.py        # ~300 LOC — run_agent_loop() (simple loop, unchanged)
                       #            + run_agentic_loop() skeleton (delegates to sections)
  agentic_perception.py  # ~500 LOC — Section 1: percept source, CLI/voice input processing
  agentic_proposals.py   # ~200 LOC — Section 2+3: LLM proposal polling + agent fallback
  agentic_execution.py   # ~600 LOC — Section 4+5: execute pending actions, approval paths
  agentic_submission.py  # ~400 LOC — Section 6: build context, submit to LLM worker
  agentic_lifecycle.py   # ~150 LOC — Sections 0+7+8+9+cleanup: stop checks, step callback,
                         #            frequency control, session teardown
  loop_state.py          # (existing, keep as-is — 67 LOC)
  approval.py            # (existing, keep as-is)
```

### How the skeleton works

`run_agentic_loop()` stays in `agent_loop.py` but becomes an orchestrator that calls section handlers:

```python
def run_agentic_loop(agent, environment, state, ...):
    ctx = AgenticLoopContext(agent, environment, state, ...)  # shared state object
    
    # Setup
    _agentic_setup(ctx)
    
    for step in iterator:
        # Section 0: stop checks, DN config
        if _check_stop_conditions(ctx):
            break
        
        # Section 1: perception
        _handle_perception(ctx)
        
        # Section 2+3: check LLM proposals / agent fallback
        _handle_proposals(ctx)
        
        # Section 4+5: execute pending actions
        _handle_execution(ctx)
        
        # Section 6: submit context to LLM
        _handle_submission(ctx)
        
        # Section 7+8+9: callbacks, persist, frequency
        _handle_maintenance(ctx)
    
    # Cleanup
    _agentic_teardown(ctx)
```

### `AgenticLoopContext` dataclass

The key enabler is a shared context object that replaces the ~30 local variables currently scattered through the function:

```python
@dataclass
class AgenticLoopContext:
    """Shared mutable state for the agentic loop sections."""
    # Injected dependencies (from function params)
    agent: Any
    environment: Any
    state: Any
    memory: Any
    executor: Any
    llm_worker: LLMWorker | None
    autonomy_controller: AutonomyController | None
    default_network: Any | None
    hippocampus: Any | None
    memory_hub: Any | None
    context_pool: Any | None
    percept_source: Any | None
    action_sink: Any | None
    on_step: Callable | None
    on_event: Callable | None
    
    # Mutable loop state
    pending_proposal: LLMProposal | None = None
    pending_actions: deque = field(default_factory=deque)
    confirmation_mode: bool = False
    confirmation_action: dict | None = None
    grace_deadline: float | None = None
    last_input_text: str | None = None
    step: int = 0
```

### Section breakdown

| Section | Current lines | New file | Key responsibility |
|---------|--------------|----------|-------------------|
| Stop checks + DN config | 534-589 | `agentic_lifecycle.py` | Check stop_event, shutdown mode, configure DN, grace period |
| Perception | 591-1078 | `agentic_perception.py` | Percept source, CLI input, confirmation mode, modification mode, timeout retry, normal input |
| LLM proposals | 1113-1248 | `agentic_proposals.py` | Poll LLMWorker, staleness guard, queue next_actions |
| Agent fallback | 1250-1476 | `agentic_proposals.py` | Agent propose_intent when no LLM proposal |
| Execute actions | 1479-2070 | `agentic_execution.py` | Parallel actions, planning mode, multi-step queue, normal execution, ADaPT replan |
| Approved proposals | 2071-2151 | `agentic_execution.py` | Planning autonomy approved execution |
| Submit to LLM | 2153-2549 | `agentic_submission.py` | Build context, check triggers, route to LLM worker |
| Callbacks + persist | 2550-2586 | `agentic_lifecycle.py` | Step callback, step counter, state persistence, frequency control |
| Cleanup | 2587-2629 | `agentic_lifecycle.py` | Final persist, hippocampus flush, session end, DN stop |

### Migration strategy

1. Create `AgenticLoopContext` dataclass in `runtime/agentic_context.py`
2. Extract each section into its own file as a function that takes `ctx: AgenticLoopContext`
3. Replace `run_agentic_loop()` body with the skeleton that calls section functions
4. Run full test suite to verify behavior is preserved
5. The 3 nested helper functions (`_get_all_tools`, `configure_dn_for_mode`, `inhibit_dn_for_tool`) move to `agentic_lifecycle.py`

### Risk

**Medium.** The sections share mutable local state (pending_proposal, confirmation_mode, etc.) which the context object must capture correctly. The main risk is missing a state variable during extraction. Mitigation: diff the local variable set before and after extraction to ensure nothing was lost.

---

## 3. Simplify `models/language/router.py` (1,721 LOC)

**Current state:** LLMRouter class (42 methods, ~1,120 LOC) + ~350 LOC of inline data (profiles, pricing, quantization) + `load_llm_config()` (190 LOC).

**Approach:** Extract data, keep class intact. LLMRouter's 42 methods are well-grouped and don't benefit from further splitting — the class is the right abstraction boundary.

### Extractions

| What | Current location | New location | LOC saved |
|------|-----------------|-------------|-----------|
| `QUANTIZATION_LEVELS` + helpers | lines 45-73 | `data/models/quantization.json` + 10-line loader | ~30 |
| `_PROFILE_ALIASES` + `_BUILTIN_PROFILES` | lines 76-226 | `data/models/llm_profiles.json` + 15-line loader | ~150 |
| `_DEFAULT_PRICING` + `_MODEL_DOWNGRADE_MAP` | lines 368-382 | Inline in LLMRouter.__init__ or separate `pricing.py` | ~15 |
| `load_llm_config()` | lines 409-600 | `models/language/config_loader.py` | ~190 |

**Net result:** router.py drops from ~1,721 to ~1,340 LOC. The class itself stays in one file.

---

## 4. Simplify `modes/definitions.py` (1,254 LOC)

**Current state:** ~550 LOC of inline data (prompts, tool descriptions) + ~700 LOC of logic.

### Extractions

| What | Current lines | New location | LOC saved |
|------|--------------|-------------|-----------|
| `TOOL_DESCRIPTIONS` dict | 960-1194 | `data/prompts/tool_descriptions.json` | ~230 |
| Strategy `context_prompt` strings | inside STRATEGIES (106-197) | `data/prompts/strategies/{name}.txt` | ~60 |
| Mode `context_prompt` strings | inside OPERATIONAL_MODES (404-556) | `data/prompts/modes/{name}.txt` | ~80 |

**Net result:** definitions.py drops from ~1,254 to ~880 LOC. The mode/strategy logic stays. `__post_init__` on ModeDefinition already has a pattern for loading prompts from files (line 293-307), so this extends an existing mechanism.

---

## 5. Simplify `agents/llm_worker.py` (1,142 LOC)

**Current state:** ~60 LOC of re-exports (already flagged as #13), ~25 pass-through `@staticmethod` wrappers (~120 LOC, flagged as #23/S7), legacy dual-mode branching (~115 LOC, flagged as #30).

### Extractions (all already flagged in cleanup plan)

| What | LOC removed | Cleanup # |
|------|------------|-----------|
| Remove re-exports (lines 17-84) | ~60 | #13 |
| Remove pass-through statics (lines 1011-1127) | ~120 | #23/S7 |
| Remove legacy queue path (if WorkerPool is standard) | ~115 | #30 |

**Net result:** llm_worker.py drops from ~1,142 to ~850 LOC with no new files needed — just deletions.

---

## Execution Order

1. **bus.py split** (Medium effort, low risk) — pure data type reorganization, no behavior changes. Re-exports preserve backward compat. Do first because it's the simplest structurally.

2. **definitions.py data extraction** (Small effort, low risk) — moving strings to files. The `__post_init__` file-loading pattern already exists.

3. **llm_worker.py cleanup** (Small effort, low risk) — pure deletions of dead code and wrappers.

4. **router.py data extraction** (Small effort, low risk) — similar pattern to definitions.py.

5. **agent_loop.py split** (Large effort, medium risk) — requires the AgenticLoopContext refactor. Do last because it's the most complex and benefits from having the other files cleaned up first.

---

## Validation

After each split:
1. `python -c "import maxim"` — no import errors
2. `pytest tests/unit/ -x -q` — no test regressions
3. Architecture audit (if AST-based validator exists) — layer rules still pass
4. Grep for old import paths — all still resolve via re-exports
