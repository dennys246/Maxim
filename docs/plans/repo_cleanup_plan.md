# Repo Cleanup Plan

Targeted fixes for design smells, redundancies, and inconsistencies discovered during development. None of these are blocking — they're quality-of-life improvements that prevent future confusion.

---

## Partially Addressed

### 2. Simulation Trace Noise Cleanup — PARTIALLY DONE

**Status:** PIPELINE traces silenced from terminal by default, `--sim-debug` flag added. Debug traces always persist to JSONL log.

**Remaining:** WorkerPool `mark_completed` trace still uses `sim_log("PIPELINE", ...)` which is now silenced. The trace in `worker_pool.py` should be removed entirely (it was added for debugging, not for production tracing).

**Files:** `src/maxim/runtime/worker_pool.py:113`

---

## Open Issues (Original)

### 1. Duplicate Step Counter Termination

**Problem:** Two independent mechanisms terminate the agent loop:
- `run_agentic_loop(max_steps=N)` uses `range(N)` or `itertools.count()` as the loop iterator
- `state.is_done()` checks `state.steps_taken >= state.max_steps` at line 2576 of agent_loop.py

These conflict silently. The `--sim` bug (loop dying at step 200 despite `max_steps=0`) was caused by `build_state(max_steps=200)` overriding the loop parameter. Currently worked around by setting `state.max_steps=0` in sim mode.

**Fix:** Remove `state.is_done()` step check from the agent loop. Let `run_agentic_loop(max_steps=)` be the sole step-based termination. Keep `state.done` flag for explicit shutdown signaling.

**Risk:** Low. Check all callers of `state.is_done()` to ensure nothing depends on the step-count behavior.

**Files:** `src/maxim/runtime/agent_loop.py:2576`, `src/maxim/runtime/state.py:74-79`

---

### 3. Double LLM Model Load in Simulation

**Problem:** The simulation generator creates its own `LLMAgent` instance with `n_ctx=4096`, while the main agent pipeline loads Mistral with `n_ctx=8192`. Two separate llama-cpp instances in memory (~4GB each).

**Fix:** Have the generator accept an existing LLMRouter or share the backend. The generator only needs `generate()` — it doesn't need its own context window. The continuation agent in interactive.py has the same issue.

**Files:** `src/maxim/simulation/simulation_generator.py`, `src/maxim/simulation/interactive.py`

---

### 4. llama-cpp Metal Kernel Warnings

**Problem:** Every llama-cpp model load prints ~20 lines of `ggml_metal_init: skipping kernel_*_bf16` warnings.

**Fix options:**
- `GGML_METAL_LOG_LEVEL=0` environment variable (if llama-cpp supports it)
- Redirect stderr during `_ensure()` via `contextlib.redirect_stderr`
- Accept the noise (lowest effort)

**Files:** `src/maxim/models/language/llama_backend.py`

---

### 5. Orphaned `_sim_interactive` Variable Scoping

**Problem:** `_sim_interactive` set inside `if sim_path is not None:` block, referenced later in agentic block. Works because Python lacks block scoping, but confusing.

**Fix:** Use `getattr(args, "sim", None)` checks consistently or store as `args._sim_interactive`.

**Files:** `src/maxim/cli.py`

---

### 6. Batch Scenario Mode Only Runs First File

**Problem:** `--sim scenarios/` iterates files but `break`s after the first. Only one scenario runs per invocation.

**Fix:** Implement sequential batch processing with results aggregation, or redirect to interactive mode for directories.

**Files:** `src/maxim/cli.py:514-525`

---

### 7. CWD Change Not Protected by try/finally

**Problem:** `os.chdir()` to sandbox happens early, cleanup only in post-validation. Exception between them leaves CWD wrong.

**Fix:** Wrap entire sim execution in try/finally that always restores CWD.

**Files:** `src/maxim/cli.py`

---

## New Issues (from audit)

### 8. Unused PerceptSource Protocol Module

**Problem:** `src/maxim/simulation/sources.py` defines the `PerceptSource` Protocol but nothing imports it. The `percept_source` parameter in `run_agentic_loop` uses `Any | None` instead.

**Fix:** Either:
- Use `PerceptSource` as the type annotation in agent_loop.py (preferred — makes the contract explicit)
- Or remove sources.py if the protocol isn't providing value

**Files:** `src/maxim/simulation/sources.py`, `src/maxim/runtime/agent_loop.py:323`

---

### 9. `Any` Type Overuse in Runtime Functions

**Problem:** Core functions use `Any` for parameters that have well-defined types:
- `run_agentic_loop()`: agent, environment, state, memory, decision_engine, executor all typed as `Any`
- `FearGatedExecutor.__init__()`: executor typed as `Any`
- `pain_bus` parameter everywhere: `Any | None`
- `run_interactive_sim()`: most params as `Any`

**Fix:** Define Protocols for the key interfaces (Executor, Environment, State) or import actual types with `TYPE_CHECKING` guards. Start with the most-used: Executor and PainBus.

**Files:** `src/maxim/runtime/agent_loop.py`, `src/maxim/runtime/fear_gate.py`, `src/maxim/simulation/runner.py`, `src/maxim/simulation/interactive.py`

---

### 10. Silent Exception Swallowing in Agent Loop

**Problem:** Multiple `except Exception: pass` blocks in agent_loop.py sleep/retry logic with no logging:

```python
except Exception:
    pass  # Lines ~165, 175, 185, 196, 211
```

These hide failures that could explain unexpected behavior.

**Fix:** Add `logger.debug()` to all bare `except` blocks in the agent loop.

**Files:** `src/maxim/runtime/agent_loop.py`

---

### 11. Missing Unit Tests for Simulation Modules

**Problem:** Only `test_percept_simulation.py` exists. These modules have no dedicated tests:
- `instrumented_executor.py`
- `sim_logger.py`
- `simulation_generator.py` (requires LLM, but can test `_clean_percepts`, `_extract_json`)
- `conversational_source.py`
- `interactive.py`

**Fix:** Add unit tests for the deterministic parts (JSON extraction, percept cleaning, sim_logger state management, ConversationalSource queue behavior). LLM-dependent code can be tested with mocks.

**Files:** `tests/unit/`

---

### 12. Config Value Mismatch: "int4" vs "Q4_K_M"

**Problem:** `data/util/llm.json` uses `"quantization": "int4"` but the codebase standard is `Q4_K_M`. The `int4` shorthand works (transformers_backend handles it) but is inconsistent with all documentation and profiles.

**Fix:** Change llm.json to `"quantization": "Q4_K_M"`.

**Files:** `data/util/llm.json:6`

---

### 13. Stale Re-exports in llm_worker.py

**Problem:** ~60 lines of re-exports from modularized modules (token_counter, prompt_formats, json_parser, llama_backend). Marked `# noqa: F401`. None of these re-exports are actually imported from llm_worker anywhere in the codebase — all callers import from the original modules.

**Fix:** Remove the re-exports or add a deprecation warning. These were a "just in case" backward-compat layer that nobody uses.

**Files:** `src/maxim/agents/llm_worker.py:17-84`

---

### 14. ScenarioRunner Race Condition

**Problem:** `ScenarioRunner.run()` stores mutable `_source` and `_sink` as instance attributes. Concurrent calls to `run()` would clobber each other.

**Fix:** Use local variables instead of `self._source`/`self._sink`, or document as single-threaded.

**Files:** `src/maxim/simulation/runner.py:34-40`

---

### 15. README Prompt Profile Table Inaccurate

**Problem:** README claims minimal/standard/rich prompt profiles with specific depths and worker counts. The actual implementation uses per-mode response configs from llm.json (`mode_response_config`), not these hardcoded profiles.

**Fix:** Update README to accurately describe the per-mode system, or implement the profiles as described.

**Files:** `README.md:165-173`

---

## Priority

| # | Issue | Effort | Impact | Status |
|---|-------|--------|--------|--------|
| 1 | Duplicate step counter | Small | Prevents bugs | Open |
| 2 | Trace noise | Small | UX | **Partially done** |
| 3 | Double model load | Medium | Saves ~4GB RAM | Open |
| 4 | Metal warnings | Small | Cosmetic | Open |
| 5 | Variable scoping | Small | Code clarity | Open |
| 6 | Batch scenario | Medium | Feature completion | Open |
| 7 | CWD protection | Small | Robustness | Open |
| 8 | Unused PerceptSource | Small | Type safety / dead code | New |
| 9 | Any type overuse | Medium | Type safety | New |
| 10 | Silent exceptions | Small | Debuggability | New |
| 11 | Missing sim tests | Medium | Test coverage | New |
| 12 | Config mismatch | Trivial | Consistency | New |
| 13 | Stale re-exports | Small | Code clarity | New |
| 14 | ScenarioRunner race | Small | Thread safety | New |
| 15 | README profiles | Small | Doc accuracy | New |

## Suggested execution order

1. **#12** Config mismatch (trivial, one line)
2. **#10** Silent exceptions (small, add logger.debug)
3. **#1** Duplicate step counter (small, prevents future bugs)
4. **#8** PerceptSource protocol (small, type safety)
5. **#7** CWD protection (small, robustness)
6. **#13** Stale re-exports (small, cleanup)
7. **#14** ScenarioRunner race (small, local vars)
8. **#5** Variable scoping (small, clarity)
9. **#15** README profiles (small, accuracy)
10. **#11** Missing sim tests (medium, coverage)
11. **#9** Any type overuse (medium, incremental)
12. **#3** Double model load (medium, memory savings)
13. **#6** Batch scenario (medium, feature)
14. **#4** Metal warnings (small, cosmetic — may not be fixable)