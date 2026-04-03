# Repo Cleanup Plan

Targeted fixes for design smells, redundancies, and inconsistencies discovered during development. None of these are blocking — they're quality-of-life improvements that prevent future confusion.

---

## 1. Duplicate Step Counter Termination

**Problem:** Two independent mechanisms terminate the agent loop:
- `run_agentic_loop(max_steps=N)` uses `range(N)` or `itertools.count()` as the loop iterator
- `state.is_done()` checks `state.steps_taken >= state.max_steps` at line 2576 of agent_loop.py

These can conflict silently. The `--sim` bug (loop dying at step 200 despite `max_steps=0`) was caused by `build_state(max_steps=200)` overriding the loop parameter.

**Fix:** Remove `state.is_done()` step check from the agent loop. Let `run_agentic_loop(max_steps=)` be the sole step-based termination. Keep `state.done` flag for explicit shutdown signaling.

**Risk:** Low. Check all callers of `state.is_done()` to ensure nothing depends on the step-count behavior.

**Files:** `src/maxim/runtime/agent_loop.py:2576`, `src/maxim/runtime/state.py:74-79`

---

## 2. Simulation Trace Noise Cleanup

**Problem:** During simulation development, many debug traces were added (every 5 steps during grace period, WorkerPool completion, etc.). These are useful for debugging but noisy for normal use.

**Fix:** Gate all sim traces behind a `SIM_DEBUG` verbosity level or `--sim-debug` flag. Default simulation should show key events only (PERCEPT, FEAR, MOTOR, PAIN, HIPPOCAMPUS) not every poll iteration.

**Files:** `src/maxim/runtime/agent_loop.py`, `src/maxim/runtime/worker_pool.py`

---

## 3. Double LLM Model Load in Simulation

**Problem:** The simulation generator creates its own `LLMAgent` instance with `n_ctx=4096`, while the main agent pipeline loads Mistral with `n_ctx=8192`. These are two separate llama-cpp instances in memory.

**Fix:** Have the generator accept an existing LLMRouter or share the backend. The generator only needs `generate()` — it doesn't need its own context window or model instance.

**Files:** `src/maxim/simulation/simulation_generator.py`, `src/maxim/simulation/interactive.py`

---

## 4. llama-cpp Metal Kernel Warnings

**Problem:** Every llama-cpp model load prints ~20 lines of `ggml_metal_init: skipping kernel_*_bf16` warnings. These are benign (bf16 not supported on the current GPU) but noisy.

**Fix:** Redirect stderr during model load or filter these warnings. The llama-cpp library uses C-level stderr output, so Python's `warnings.filterwarnings` doesn't help. Options:
- Redirect `sys.stderr` during `_ensure()` calls
- Set `GGML_METAL_LOG_LEVEL=0` environment variable if supported
- Accept the noise (lowest effort)

**Files:** `src/maxim/models/language/llama_backend.py`

---

## 5. Orphaned `_sim_interactive` Variable Scoping

**Problem:** The `_sim_interactive` variable is set inside the `if sim_path is not None:` block in cli.py but referenced later in the agentic block. Python doesn't have block scoping so it works, but it's fragile and confusing.

**Fix:** Use `getattr(args, "sim", None)` checks consistently instead of a separate variable, or set it as an attribute on `args`.

**Files:** `src/maxim/cli.py`

---

## 6. Batch Scenario Mode Only Runs First File

**Problem:** When `--sim scenarios/` is passed with a directory, the loop iterates scenario files but `break`s after the first one. Only one scenario runs.

**Fix:** Either implement proper batch processing (run each scenario sequentially, aggregate results) or document that batch mode is not yet supported.

**Files:** `src/maxim/cli.py:514-525`

---

## 7. CWD Change Not Protected by try/finally

**Problem:** `os.chdir()` to the simulation sandbox happens early but cleanup (`os.chdir(original_cwd)`) is only in the post-validation block. If an exception occurs between the chdir and cleanup, the working directory is never restored.

**Fix:** Wrap the entire sim execution in a try/finally that always restores CWD.

**Files:** `src/maxim/cli.py`

---

## Priority

| # | Issue | Effort | Impact |
|---|-------|--------|--------|
| 1 | Duplicate step counter | Small | Prevents future bugs like the sim termination issue |
| 2 | Trace noise | Small | Better user experience in sim mode |
| 3 | Double model load | Medium | Saves ~4GB RAM and startup time |
| 4 | Metal warnings | Small | Cosmetic but noisy |
| 5 | Variable scoping | Small | Code clarity |
| 6 | Batch scenario | Medium | Feature completion |
| 7 | CWD protection | Small | Robustness |
