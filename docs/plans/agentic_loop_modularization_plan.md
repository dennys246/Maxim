# Agentic Loop Modularization Plan

Refactor `run_agentic_loop()` from a 2,300-line monolithic function into a testable, maintainable `LoopController` class with discrete phase methods.

**Last updated:** 2026-04-03  
**Status:** Complete (all phases implemented on `refactor/loop-modularization-phase0`)

---

## Motivation

`src/maxim/runtime/agent_loop.py` has grown organically and accumulated significant debt:

- **2,300 lines in one function** with 4-5 levels of nesting and `continue` jumps
- **7x duplication** of outcome recording (recent_outcomes + llm_worker + context_pool + hippocampus)
- **Stringly-typed state** — ~15 `state.data["pending_*"]` keys with no schema
- **Simulation code inline** — ~20 `if percept_source is not None` guards polluting the core path
- **Two loop variants** (`run_agent_loop` and `run_agentic_loop`) duplicating the observe → propose → execute skeleton
- **A correctness bug** — `set.pop()` on `processed_cli_inputs` evicts an arbitrary element, not the oldest

The architecture (non-blocking LLM, pub/sub agents, staged memory, autonomy levels) is sound. This plan restructures the implementation without changing the architecture.

---

## Phase 0: Extract Helpers (No Behavior Change)

**Goal:** Reduce duplication and fix the correctness bug without changing control flow.

### 0.1 — `_record_outcome()` helper

Extract the 7 copy-pasted outcome recording blocks into a single function:

```python
def _record_outcome(
    tool_name: str,
    success: bool,
    result_summary: str | None,
    error: str | None,
    *,
    recent_outcomes: list[dict],
    max_recent: int,
    llm_worker: LLMWorker | None,
    reasoning: str,
    context_pool: ContextPool,
) -> None:
    ...
```

**Locations to consolidate:**
- Confirmed action (~line 733)
- Rejected action (~line 816)
- Agent fallback success/failure (~line 1427)
- LLM action success (~line 1804)
- LLM action failure (~line 1946)
- Plan approval execution (~line 2095)
- Hard rejection (~line 2046)

### 0.2 — Fix `processed_cli_inputs`

Replace `set` with `collections.deque(maxlen=20)` so eviction is FIFO:

```python
processed_cli_inputs: deque[str] = deque(maxlen=20)
```

Adjust membership check to use `in` (deque supports it, O(n) but n=20).

### 0.3 — `_execute_and_record()` helper

Unify the tool execution pipeline shared by section 3 (agent fallback) and section 4 (LLM proposal):

```python
def _execute_and_record(
    action: dict,
    confidence: float,
    source: str,
    reasoning: str,
    *,
    executor: Executor,
    autonomy_controller: AutonomyController,
    state: State,
    hippocampus: Hippocampus | None,
    ...
) -> ToolResult:
    ...
```

### 0.4 — Cache `_get_all_tools()` per iteration

Currently called ~3 times per loop tick. Cache at loop-start:

```python
available_tools_snapshot = _get_all_tools()
```

**Validation:** All existing tests pass. No behavior change.

---

## Phase 1: Type the State Bag

**Goal:** Replace stringly-typed `state.data` keys with a typed dataclass.

### 1.1 — `LoopState` dataclass

```python
@dataclass
class LoopState:
    # User input
    pending_user_input: str | None = None
    pending_user_input_time: float = 0.0
    pending_user_input_source: str = "CLI"

    # Confirmation flow
    pending_confirmation: PendingConfirmation | None = None

    # Planning flow
    pending_plan_text: str | None = None
    pending_plan_tool: str | None = None
    plan_modification_context: PlanModification | None = None

    # Timeout retry
    pending_timeout_retry: TimeoutRetry | None = None

    # Modification
    pending_modification: PendingModification | None = None

    # Replan (ADaPT)
    replan_candidate: LLMProposal | None = None
    replan_goal: str | None = None

    # Exploration
    exploration_mode: bool = False
    exploration_focus: str = ""
    exploration_session_id: str = ""
    exploration_policy: dict = field(default_factory=dict)

    # Mode
    mode: str = "observe"
    processing_state: str = "awake"
    current_strategy: str = ""
```

### 1.2 — Sub-dataclasses for nested state

```python
@dataclass
class PendingConfirmation:
    action: dict
    reasoning: str
    confidence: float
    tool_name: str

@dataclass
class PendingModification:
    original_action: dict
    original_reasoning: str
    original_tool_name: str
    user_modification: str
    timestamp: float

@dataclass
class TimeoutRetry:
    original_request: Any
    timeout_s: float
```

### 1.3 — Migration shim

Keep `state.data` as the backing store during transition. `LoopState` reads/writes through it so existing consumers still work:

```python
@property
def pending_confirmation(self) -> PendingConfirmation | None:
    raw = self._state.data.get("pending_confirmation")
    return PendingConfirmation(**raw) if raw else None
```

Remove shim once all consumers are migrated.

**Validation:** Type checker catches key typos. All tests pass.

---

## Phase 2: Extract Phase Methods

**Goal:** Turn the numbered sections into methods on a `LoopController` class.

### 2.1 — `LoopController` class skeleton

```python
class LoopController:
    def __init__(
        self,
        agent, environment, state, memory,
        executor, autonomy_controller,
        llm_worker, hippocampus, memory_hub,
        context_pool, protocol_registry,
        *,
        target_hz: float = 30.0,
        on_event: Callable | None = None,
        on_step: Callable | None = None,
    ): ...

    # Phase methods (one per numbered section)
    def check_stop_conditions(self) -> bool: ...
    def observe(self) -> dict: ...
    def parse_input(self, observation: dict) -> ParsedInput | None: ...
    def handle_confirmation(self, text: str) -> bool: ...
    def handle_timeout_retry(self, text: str) -> bool: ...
    def handle_plan_approval(self, text: str) -> bool: ...
    def check_proposals(self) -> LLMProposal | None: ...
    def agent_fallback(self) -> dict | None: ...
    def execute_proposal(self, proposal: LLMProposal) -> None: ...
    def submit_to_llm(self) -> bool: ...
    def persist_if_needed(self) -> None: ...
    def maintain_frequency(self, loop_start: float) -> None: ...

    # The main loop becomes ~40 lines
    def run(self) -> None:
        for step in self._step_iter():
            loop_start = time.time()
            if self.check_stop_conditions():
                break
            obs = self.observe()
            parsed = self.parse_input(obs)
            proposal = self.check_proposals()
            if proposal:
                self.execute_proposal(proposal)
            elif not self._has_pending_llm_input:
                self.agent_fallback()
            self.submit_to_llm()
            self.persist_if_needed()
            self.maintain_frequency(loop_start)
        self._shutdown()
```

### 2.2 — Input parsing sub-dispatcher

The confirmation → timeout → plan approval → modification chain becomes:

```python
def parse_input(self, observation: dict) -> ParsedInput | None:
    cli_text = self._extract_cli_input(observation)
    if not cli_text:
        return None

    # Priority-ordered dispatch
    if self._loop_state.pending_confirmation:
        if self.handle_confirmation(cli_text):
            return None  # Consumed
    if self._loop_state.pending_timeout_retry:
        if self.handle_timeout_retry(cli_text):
            return None
    if self._pending_plan_proposal:
        if self.handle_plan_approval(cli_text):
            return None

    # Normal input path
    return ParsedInput(text=cli_text, source=source)
```

### 2.3 — Backward-compatible wrapper

Keep `run_agentic_loop()` as a thin wrapper that creates a `LoopController` and calls `.run()`:

```python
def run_agentic_loop(agent, environment, state, memory, ...) -> None:
    controller = LoopController(agent, environment, state, memory, ...)
    controller.run()
```

**Validation:** Identical behavior. Each phase method independently testable.

---

## Phase 3: Consolidate Loop Variants

**Goal:** Make `run_agent_loop` a configuration of `LoopController` instead of a separate implementation.

### 3.1 — Synchronous execution mode

Add a `sync_mode: bool` flag to `LoopController`:
- When `True`: `execute_proposal()` blocks on LLM, no polling
- When `False`: non-blocking LLM via `LLMWorker` (current behavior)

### 3.2 — Retire `run_agent_loop`

Replace with:

```python
def run_agent_loop(agent, environment, state, memory, decision_engine, executor, **kwargs):
    controller = LoopController(
        agent, environment, state, memory, executor,
        sync_mode=True,
        decision_engine=decision_engine,
        **kwargs,
    )
    controller.run()
```

**Validation:** Existing callers of `run_agent_loop` work unchanged.

---

## Phase 4: Isolate Simulation Concerns

**Goal:** Remove inline simulation guards from the core loop.

### 4.1 — `SimulationAdapter`

```python
class SimulationAdapter:
    """Wraps percept_source, action_sink, and sim_logger."""

    def __init__(self, percept_source, action_sink, pain_bus):
        self.percept_source = percept_source
        self.action_sink = action_sink
        self.pain_bus = pain_bus

    def next_observation(self, environment) -> dict:
        """Returns observation from percept_source or environment."""
        ...

    def is_exhausted(self) -> bool: ...
    def check_grace_period(self) -> bool: ...
    def log(self, category: str, msg: str, data: dict | None = None): ...
```

### 4.2 — Null adapter for production

```python
class NullSimulationAdapter:
    def next_observation(self, environment) -> dict:
        return environment.observe()
    def is_exhausted(self) -> bool:
        return False
    def check_grace_period(self) -> bool:
        return False
    def log(self, *args, **kwargs):
        pass
```

### 4.3 — Replace inline checks

All `if percept_source is not None` and `try: from sim_logger import sim_log` blocks replaced with `self._sim.method()` calls.

**Validation:** Simulation tests pass. Core loop reads cleanly without simulation noise.

---

## Phase 5: Improve Followup and Parallel Execution

**Goal:** Fix misnamed APIs and fragile data passing.

### 5.1 — `ActionFollowup` dataclass

Replace `dict[str, Any]` with:

```python
@dataclass
class ActionFollowup:
    tool: str
    result: str
    original_query: str
    followup_type: str  # "process" | "respond" | "engage"
    mode: str
    timestamp: float
```

### 5.2 — First-class followup in LLM submission

Add `followup: ActionFollowup | None` to `LLMRequest` instead of injecting synthetic `[ACTION_FOLLOWUP ...]` strings into `cli_inputs`. The prompt builder formats it as a dedicated section.

### 5.3 — Rename `parallel_actions` to `batched_actions`

Or actually parallelize with `concurrent.futures.ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=min(len(actions), 4)) as pool:
    futures = {pool.submit(executor.execute, a): a for a in actions}
    for future in as_completed(futures):
        result = future.result()
        ...
```

**Validation:** Followup responses render identically. Batched actions complete faster for I/O tools.

---

## Phase 6: Make StructuredContext Immutable

**Goal:** Prevent accidental mutation of shared context objects.

### 6.1 — Freeze the dataclass

```python
@dataclass(frozen=True)
class StructuredContext:
    ...
```

### 6.2 — Builder pattern for construction

```python
@dataclass
class StructuredContextBuilder:
    """Mutable builder, produces frozen StructuredContext."""
    ...
    def build(self) -> StructuredContext:
        return StructuredContext(**self.__dict__)
```

### 6.3 — Replace mutations with copies

Current mutations like `context.cli_inputs.append(...)` become:

```python
context = replace(context, cli_inputs=[*context.cli_inputs, synthetic_input])
```

**Validation:** Any code that was accidentally mutating shared context will fail loudly at freeze time instead of silently corrupting state.

---

## Phase 7: Improve Error Handling

**Goal:** Replace blanket `except Exception: pass` with structured error handling so failures are diagnosable without losing real-time resilience.

### 7.1 — Error severity tiers

Define which subsystems are critical vs. optional:

| Tier | Subsystem | On failure |
|------|-----------|------------|
| Critical | Tool execution, LLM submission, autonomy check | Log WARNING, record in outcomes, continue |
| Important | Memory storage, hippocampus capture, context pool | Log WARNING, continue |
| Optional | DefaultNetwork, simulation logger, provenance, statistician | Log DEBUG, continue |
| Silent | Step callback, on_event callback | Log DEBUG, continue |

### 7.2 — `@resilient` decorator

Replace inline try/except with a decorator that logs at the appropriate level:

```python
def resilient(tier: str = "important", operation: str = ""):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                level = logging.WARNING if tier in ("critical", "important") else logging.DEBUG
                logger.log(level, "%s failed: %s", operation or fn.__name__, e)
                if tier == "critical":
                    log_agentic("agent_loop", "error",
                                {"context": operation, "error": str(e)[:200]},
                                level="WARNING")
                return None
        return wrapper
    return decorator
```

### 7.3 — Remove bare `pass` in except blocks

Audit all `except Exception: pass` in `agent_loop.py`. Each should either:
- Use the `@resilient` decorator (for extracted phase methods)
- Log at DEBUG minimum (for inline catches that remain)
- Be removed if the exception is impossible (dead code)

**Validation:** Same runtime behavior, but failures now leave a trail in logs.

---

## Phase 8: Bus Handler Safety

**Goal:** Prevent slow bus subscribers from blocking the agent loop.

### Current problem

`AgentBus.publish()` calls handlers synchronously in the caller's thread. A slow `MemoryAgent._on_percept()` or `StatisticianAgent._on_tool_result()` blocks the loop iteration.

### 8.1 — Async handler queue (optional per subscriber)

```python
class AgentBus:
    def subscribe(self, msg_type, handler, *, async_delivery: bool = False):
        if async_delivery:
            # Handler runs in a background thread via ThreadPoolExecutor
            self._async_handlers[msg_type].append(handler)
        else:
            self._sync_handlers[msg_type].append(handler)
```

### 8.2 — Handler timeout guard

For synchronous handlers, add a timeout warning:

```python
def publish(self, message):
    for handler in self._sync_handlers[type(message)]:
        start = time.monotonic()
        handler(message)
        elapsed = time.monotonic() - start
        if elapsed > 0.05:  # 50ms
            logger.warning("Slow bus handler: %s took %.1fms",
                           handler.__qualname__, elapsed * 1000)
```

### 8.3 — Identify candidates for async delivery

- `MemoryAgent._on_percept()` — if it triggers hippocampus capture
- `StatisticianAgent._on_tool_result()` — if AG escalation fires
- `Hippocampus._on_tool_result()` — already has async capture worker

**Validation:** Bus behavior unchanged for sync handlers. Slow handlers logged. Async handlers don't block loop.

---

## Phase 9: Default Network Decoupling

**Goal:** Move DN lifecycle management out of the loop body.

### Current problem

`configure_dn_for_mode()` and `inhibit_dn_for_tool()` are closures defined inside `run_agentic_loop` (~lines 446-490). DN start/stop is mixed into the loop's stop conditions and shutdown sequence.

### 9.1 — `DefaultNetworkController`

```python
class DefaultNetworkController:
    def __init__(self, default_network: Any | None):
        self._dn = default_network
        self._last_mode: str | None = None

    @property
    def enabled(self) -> bool:
        return self._dn is not None

    def configure_for_mode(self, mode_name: str) -> None:
        """Apply mode-specific DN config (behavior priorities, thresholds)."""
        ...

    def inhibit_for_tool(self, mode_name: str) -> bool:
        """Check if DN should pause during tool execution."""
        ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
```

### 9.2 — Wire into LoopController

```python
class LoopController:
    def __init__(self, ..., default_network=None):
        self._dn_ctrl = DefaultNetworkController(default_network)
```

`configure_dn_for_mode` called in `observe()` phase. `inhibit_for_tool` called in `execute_proposal()`. Start/stop in `run()`/`_shutdown()`.

**Validation:** DN behavior unchanged. Loop body shorter. DN testable in isolation.

---

## Dependency Graph

```
Phase 0: Extract Helpers ─────────────────────────────┐
    (no behavior change, fixes bug)                    │
                                                       ▼
Phase 1: Type the State Bag ──────────► Phase 2: Extract Phase Methods
    (can be done independently)              (main refactor)
                                                       │
                                    ┌──────────────────┼──────────────────┐
                                    ▼                  ▼                  ▼
                             Phase 3:           Phase 4:           Phase 5:
                             Consolidate        Isolate Sim        Fix Followup
                             Loop Variants                         & Parallel
                                    │                  │                  │
                                    └──────────────────┼──────────────────┘
                                                       │
                              ┌─────────────┬──────────┼──────────┬──────────────┐
                              ▼             ▼          ▼          ▼              ▼
                        Phase 6:      Phase 7:    Phase 8:   Phase 9:
                        Freeze        Error       Bus        DN
                        Context       Handling    Safety     Decoupling
```

**Phase 0 is safe to do immediately.** It fixes a real bug and reduces duplication with zero behavior change.

**Phase 2 is the critical path.** Everything after it becomes much easier.

**Phases 3-5 are independent** and can be done in any order after Phase 2.

**Phases 6-9 are independent polish** that can be done in any order after Phase 2.

---

## Testing Strategy

Each phase should maintain full backward compatibility:

- **Unit tests:** Each extracted phase method gets its own test file (`test_loop_observe.py`, `test_loop_parse_input.py`, etc.)
- **Integration test:** A single `test_loop_controller.py` that runs the full loop with a mock environment and verifies the same sequence of tool calls
- **Simulation regression:** Run existing simulation scenarios and diff action logs before/after
- **Bus safety:** Benchmark bus handler latency before/after Phase 8
- **Error audit:** Grep for `except Exception` after Phase 7 — count should drop significantly

---

## What This Plan Does NOT Change

- Agent architecture (MaximAgent, pub/sub bus, agent composition)
- Memory system (Hippocampus, ATL, staged formation)
- LLMWorker async model
- Autonomy controller logic
- Tool registry or executor interface
- Any external API or CLI behavior
