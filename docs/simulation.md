# Percept Simulation -- Developer Architecture

Internal documentation for the `src/maxim/simulation/` module.

## Module Structure

```
src/maxim/simulation/
    __init__.py
    sources.py                 # PerceptSource protocol + ConversationalSource
    sinks.py                   # ActionSink protocol + RecordingSink
    scenario_source.py         # ScenarioSource (YAML loader + emitter)
    interactive.py             # Conversational REPL (rewired for multi-turn)
    runner.py                  # ScenarioRunner (standalone executor)
    validation.py              # Expectation checking + ScenarioResult
    instrumented_executor.py   # InstrumentedExecutor wrapper
    simulation_generator.py    # LLM-powered natural language → YAML generation
    sim_logger.py              # Bio-subsystem tracing + JSONL persistence

src/maxim/runtime/
    fear_gate.py               # FearGatedExecutor (independent of DefaultNetwork)
```

## PerceptSource Protocol

Defined in `sources.py`. Any object that produces `Percept` instances implements this protocol:

```python
@runtime_checkable
class PerceptSource(Protocol):
    @property
    def name(self) -> str: ...

    def next_percept(self) -> Percept | None: ...

    def is_exhausted(self) -> bool: ...

    @property
    def capabilities(self) -> set[str]: ...
```

Key semantics:
- `next_percept()` is **non-blocking**. Returns `None` when idle or exhausted.
- `is_exhausted()` is `True` only for finite sources (scenarios, replays). Live sources always return `False`.
- `capabilities` tells the pipeline which subsystems to activate (e.g., `{"vision", "cli"}`).

To implement a new source, create a class satisfying this protocol and pass it as the `percept_source` parameter to `run_agentic_loop()`.

## ConversationalSource

Defined in `sources.py`. A `PerceptSource` implementation for the interactive REPL mode (`maxim --sim` with no arguments). Unlike `ScenarioSource`, which replays a finite YAML file, `ConversationalSource` generates percepts from user input via the LLM:

1. User types a natural-language scenario description.
2. The LLM generates structured percepts from the description.
3. Percepts are fed through the normal pipeline with full bio-subsystem tracing.
4. After the pipeline processes the turn, the user is prompted again: "Simulated, what happens next?"

`is_exhausted()` returns `False` -- conversational sources are infinite until the user types `quit` or `/new`.

## interactive.py

The interactive simulation module (`interactive.py`) has been rewritten around conversational multi-turn interaction rather than per-scenario execution. Key design points:

- **Single boot**: the pipeline boots once and stays running across multiple turns.
- **Contextual continuation**: each turn builds on the conversation history.
- **Commands**: `/new` (reset context), `/save` (persist session), `/status` (show state), `quit` (end).
- **Session consolidation**: memory promotion and hippocampus compaction are deferred to conversation end (`quit` or `/new`), not triggered after each turn. This avoids expensive consolidation work between rapid interactive turns.

## ActionSink Protocol and RecordingSink

Defined in `sinks.py`. Captures every tool execution for post-run validation.

```python
class ActionSink(Protocol):
    def record(self, action: ActionRecord) -> None: ...
    @property
    def actions(self) -> list[ActionRecord]: ...
```

`ActionRecord` is a frozen dataclass with fields: `timestamp`, `tool_name`, `tool_args`, `result_success`, `result_output`, `result_error`, `blocked`, `block_reason`.

`RecordingSink` is the concrete implementation. It is thread-safe (internal lock) and provides two query methods:
- `find_blocked(tool_pattern, reason_contains)` -- finds actions blocked by FearAgent or autonomy.
- `find_actions(tool, output_matches)` -- finds successful actions matching criteria.

Both methods use regex matching.

## InstrumentedExecutor

Defined in `instrumented_executor.py`. Wraps any `Executor` to record all calls into an `ActionSink`.

```python
instrumented = InstrumentedExecutor(real_executor, sink)
result = instrumented.execute(action_dict)  # forwards to real executor, records result
instrumented.record_block("Bash", "SYSTEM_DAMAGE", params)  # record a FearAgent block
```

Uses `__getattr__` forwarding so it is a drop-in replacement -- any attribute not defined on `InstrumentedExecutor` is proxied to the wrapped executor.

## Integration with run_agentic_loop()

In `src/maxim/runtime/agent_loop.py`, `run_agentic_loop()` accepts two optional parameters:

```python
def run_agentic_loop(
    ...,
    percept_source: Any | None = None,   # PerceptSource for simulation
    action_sink: Any | None = None,       # ActionSink for recording
) -> None:
```

When `percept_source` is provided, the loop replaces the normal `environment.observe()` call:

1. **Exhaustion check** (step 0.5): if `percept_source.is_exhausted()` is `True`, the loop breaks. Note: for `ConversationalSource`, `is_exhausted()` is always `False`; termination is driven by `state.is_done()` or `max_steps`.
2. **Percept fetch** (step 1): calls `percept_source.next_percept()` instead of `environment.observe()`.
3. **Pain routing**: if the percept has `source == "proprioception"` and `content == "pain_signal"`, it calls `route_pain_percept(percept, pain_bus)`. Pain routing works in headless mode via the standalone `PainBus`.
4. **Observation dict conversion**: the percept's fields are mapped into the observation dict that the rest of the pipeline consumes (`cli_input`, `transcript_chunk`, `detections`, etc.).
5. **Step advance**: calls `percept_source.advance_step()` (if available) once per iteration.

### Grace Period

After a finite percept source (YAML scenario) is exhausted, a 60-second grace period allows the pipeline to finish processing pending work. Once the LLM produces a response, the grace period tightens to 5 seconds to avoid unnecessary waiting. This prevents premature termination of scenarios where LLM inference takes variable time.

### LLMRouter.wait_ready()

`LLMRouter.wait_ready()` is called at simulation startup to ensure the language model is fully loaded before percepts begin flowing. The companion `LLMRouter.is_ready` property can be polled if non-blocking startup is needed.

## Pain Routing

`route_pain_percept()` in `src/maxim/proprioception/pain_bus.py` bridges the simulation layer into the bio-inspired subsystem:

```
Percept (source=proprioception, content=pain_signal)
    -> route_pain_percept()
        -> PainSignal constructed from metadata (pain_type, joint, intensity, velocity)
            -> PainBus.emit()
                -> Hippocampus subscriber forms episodic memory
```

This is called both in the standalone `ScenarioRunner.run()` and in `run_agentic_loop()` when a simulation percept is detected.

## ScenarioRunner (Standalone)

`ScenarioRunner` in `runner.py` provides a lightweight execution path that does not require an LLM or the full agent pipeline:

1. Creates a `ScenarioSource` from the YAML file.
2. Creates a `RecordingSink`.
3. Iterates up to `max_steps`, calling `next_percept()` and `advance_step()` each iteration.
4. Routes pain percepts through `PainBus` if provided.
5. After the loop, calls `validate_expectations()` with the sink, hippocampus, and emitted tags.
6. Returns a `ScenarioResult`.

The convenience function `run_scenario(path, hippocampus, pain_bus, max_steps)` wraps this.

For full integration testing, pass a `ScenarioSource` as `percept_source` to `run_agentic_loop()` instead.

## Adding New Expectation Types

Expectations are validated in `validation.py`. To add a new type:

1. Add fields to the `Expectation` dataclass in `scenario_source.py`:

```python
@dataclass
class Expectation:
    type: str
    # ... existing fields ...
    # Add new fields:
    my_new_field: str | None = None
```

2. Update `load_scenario()` in `scenario_source.py` to parse the new fields from YAML.

3. Add a checker function in `validation.py`:

```python
def _check_my_new_type(exp: Expectation, sink: RecordingSink, ...) -> ExpectationResult:
    # Inspect sink.actions, hippocampus, or other state
    if condition_met:
        return ExpectationResult(expectation=exp, passed=True, detail="...")
    return ExpectationResult(expectation=exp, passed=False, detail="...")
```

4. Wire it into `validate_expectations()`:

```python
elif exp.type == "my_new_type":
    results.append(_check_my_new_type(exp, sink, ...))
```

5. Add the new parameters to the `validate_expectations()` signature if your checker needs additional state (e.g., a new subsystem reference).

## Hippocampus.search_by_content()

Used by `_check_memory_formed` in validation. Defined in `src/maxim/memory/hippocampus.py`:

```python
def search_by_content(self, query: str, limit: int = 20) -> list[EpisodicMemory | CompressedMemory]:
```

Performs substring search across all text fields of stored memories. Returns up to `limit` matches. This is the mechanism by which `memory_formed` expectations verify that a percept (e.g., a pain signal) was successfully captured as an episodic memory.

## FearGatedExecutor

`src/maxim/runtime/fear_gate.py` wraps any Executor with FearAgent safety review. This operates **independently of DefaultNetwork**, ensuring tool calls are safety-gated in all modes (robot, headless, simulation).

Two-tier review:
1. **Action review**: classifies tool as `shell_exec`, `file_write`, `network_request`, or `tool_call` and runs through `FearAgent.review_action()`.
2. **Code review**: extracts code content from bash commands, file writes, and edit operations, then scans via `FearAgent.review_code()`.

FearGatedExecutor reviews ALL tool calls in ALL modes -- robot, headless, and simulation. This is independent of DefaultNetwork. DefaultNetwork retains its own FearAgent for motor/movement gating (`dn_movement` actions with pain bridge integration). FearGatedExecutor handles tool safety; DN handles motor safety.

## Session Consolidation in Simulation

In simulation mode, session consolidation (memory promotion, hippocampus compaction) is **not** run after each percept turn. Instead, consolidation is deferred to conversation end -- triggered when the user types `quit` or `/new` in interactive mode, or after the last scenario finishes in batch mode. This avoids expensive consolidation overhead between rapid interactive turns while still ensuring memories are properly promoted before the session exits.

## Simulation Generator

`src/maxim/simulation/simulation_generator.py` converts natural language descriptions to YAML scenarios using the local LLM.

Uses `LLMAgent` with a specialized system prompt that teaches the model about percept types, source fields, expectation types, and timing modes. Robust JSON extraction handles LLM output with preamble/postamble text.

CLI: `maxim --generate-simulation "description" -o output.yaml`

## Simulation Logger

`src/maxim/simulation/sim_logger.py` provides bio-inspired subsystem tracing during simulation runs.

Subsystem labels: `PERCEPT`, `HIPPOCAMPUS`, `NAc`, `FEAR`, `PAIN`, `MOTOR`, `SALIENCE`, `SCN`.

Features:
- Color-coded terminal output with elapsed timestamps
- JSONL persistence to `data/sim_sandbox/sim_log_*.jsonl`
- In-memory record accumulation via `get_sim_records()`

Saved logs can be used for:
- System refinement and debugging
- Input to sleep mode's dream function for offline pattern analysis
- Regression comparison between runs

Wired into: ScenarioSource (percept emission), FearGatedExecutor (allow/block/execute), PainBus (pain routing).
