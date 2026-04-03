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

## Open Issues (Audit Round 1)

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

## Open Issues (Audit Round 2)

### 16. No CI/CD Pipeline

**Problem:** 2,066 tests exist but nothing runs them automatically. No GitHub Actions, no pre-commit hooks. Tests only run when someone remembers to run them manually.

**Fix:** Add a minimal GitHub Actions workflow:
- Run `pytest tests/unit/ -m "not slow"` on push/PR
- Run full suite (`pytest tests/`) on merge to main
- Include coverage reporting via pytest-cov

**Effort:** Small (1-2 hours). High leverage — catches regressions before they compound.

**Files:** `.github/workflows/test.yml` (new)

---

### 17. Broken Simulation Test Paths (5 Tests Silently Skipped)

**Problem:** `test_percept_simulation.py` uses relative paths like `Path("scenarios/malware_with_pain.yaml")` which resolve from CWD, not from the test file. The scenario files exist but tests skip with "file not found" because pytest runs from the project root.

**Fix:** Use `Path(__file__).parent.parent / "scenarios/..."` or a conftest fixture that resolves scenario paths.

**Effort:** Trivial (10 minutes). Recovers 5 tests.

**Files:** `tests/unit/test_percept_simulation.py`

---

### 18. Benchmark Functions Not Discoverable by Pytest

**Problem:** `tests/benchmarks/test_memory_performance.py` defines 9 functions named `benchmark_*()` instead of `test_*()`. Pytest won't discover or run them. They're effectively dead code.

**Fix:** Either:
- Rename to `test_benchmark_*()` and mark with `@pytest.mark.slow`
- Or move to a `benchmarks/` directory outside `tests/` with its own runner script

**Files:** `tests/benchmarks/test_memory_performance.py`

---

### 19. Duplicate `_env_flag()` Utility Function

**Problem:** Identical `_env_flag()` function defined in two places:
- `src/maxim/tools/filesystem.py:98-107`
- `src/maxim/data/camera/display.py:23-32`

**Fix:** Move to `src/maxim/utils/` and import from both locations.

**Files:** `src/maxim/tools/filesystem.py`, `src/maxim/data/camera/display.py`

---

### 20. Deprecated `localhost_only` Field Still Present

**Problem:** `ConnectionConfig` in `src/maxim/conscience/connection.py:42` has `localhost_only: bool = False` marked deprecated ("use connection_mode instead") but still defined. No deprecation warning is emitted when it's used.

**Fix:** Remove the field if nothing reads it, or add a `__post_init__` deprecation warning if it's still referenced externally.

**Files:** `src/maxim/conscience/connection.py`

---

### 21. Deprecated `_reachy` Attribute in selfy.py

**Problem:** `selfy.py:354` exposes `_reachy` property marked "DEPRECATED: Use self._robot (RobotController) for new code" but it's still functional and used. Both `self._reachy` and `self._robot` coexist.

**Fix:** Grep for `_reachy` usage, migrate callers to `_robot`, then remove the property.

**Files:** `src/maxim/conscience/selfy.py`

---

### 22. Hardcoded RTSP Port Across 5 Files

**Problem:** `rtsp://localhost:8554/reachy` is hardcoded in at least 5 files:
- `src/maxim/tools/rtsp_bridge.py:28`
- `src/maxim/skills/base.py:63`
- `src/maxim/skills/protocols/shredder_segmenter.py:31,36`
- `src/maxim/skills/rtsp_streaming.py:26,169`

**Fix:** Define `DEFAULT_RTSP_URL` constant in one place (e.g., `src/maxim/hardware/constants.py`) and import everywhere.

**Files:** See above.

---

### 23. Pass-Through Static Methods in LLMWorker

**Problem:** `llm_worker.py:1053+` has multiple `@staticmethod` methods that just forward to `prompt_builder` functions with identical signatures. Pure wrappers adding no value.

**Fix:** Have callers import from `prompt_builder` directly and remove the pass-through methods.

**Files:** `src/maxim/agents/llm_worker.py`

---

### 24. Deprecated duckduckgo_search Fallback Code

**Problem:** `src/maxim/tools/internet_search.py` keeps import fallback logic for the old `duckduckgo_search` package (now renamed to `ddgs`). Since `ddgs>=6.0.0` is a core dependency in pyproject.toml, the fallback path is dead code.

**Fix:** Remove the try/except import fallback. Just `import ddgs`.

**Files:** `src/maxim/tools/internet_search.py`

---

### 25. CommunicationBridge Potentially Unused

**Problem:** `CommunicationBridge` (180 LOC) in `src/maxim/bridges/communication_bridge.py` is only imported in `bridges/__init__.py` as a re-export. No code in `memory_hub.py` or elsewhere appears to instantiate it.

**Fix:** Audit imports. If truly unused, remove it. If it's wired conditionally, add a comment explaining when it activates.

**Files:** `src/maxim/bridges/communication_bridge.py`, `src/maxim/bridges/__init__.py`

---

### 26. Plans Reference Unbuilt Features Without Status Markers

**Problem:** The three unstarted plans (simulation_agent, agent_mesh, multi_llm_scaling) read like specifications for implemented features. Someone reading the docs directory would assume these exist. No clear "NOT IMPLEMENTED" status at the top.

**Fix:** Add a status header to each plan file:
```
**Status:** Not started — this is a design document, not implemented code.
**Dependencies:** [list what must exist first]
```

Also consider a status summary table in `docs/plans/README.md` or the project README.

**Files:** `docs/plans/simulation_agent_plan.md`, `docs/plans/agent_mesh.md`, `docs/plans/multi_llm_scaling.md`

---

## Open Issues (Audit Round 3)

### 27. Duplicate `_env_flag()` / Boolean Env Parsing (15+ instances)

**Problem:** Beyond the two identical `_env_flag()` functions (#19), there are 15+ modules that independently parse boolean env vars with varying patterns: `.lower() in ("1", "true", "yes")` appears inline in `agentic_runtime.py` (8x), `media_loop.py`, `structured_logging.py`. A separate `_is_truthy()` helper exists in `plotting.py`.

**Fix:** Designate `gpu_compat.py:env_flag()` as the canonical utility (it already exists). Replace all inline patterns and duplicates with imports from there.

**Files:** `src/maxim/utils/gpu_compat.py`, `src/maxim/conscience/agentic_runtime.py`, `src/maxim/utils/plotting.py`, `src/maxim/utils/structured_logging.py`, + others

---

### 28. Duplicate Velocity Calculations Across 3 Modules

**Problem:** Angular/translation velocity is computed independently in three places:
- `proprioception/movement_tracker.py:196-242` — canonical implementation
- `energy/movement_tracker.py:60-80` — duplicates the same delta_yaw/delta_pitch/Euclidean math
- `salience/movement_detector.py:84-191` — pixel-based variant of the same concept

**Fix:** Extract shared `compute_velocity()` to `src/maxim/math/velocity.py`. Have all three modules call it with coordinate-system adapters.

**Files:** `src/maxim/proprioception/movement_tracker.py`, `src/maxim/energy/movement_tracker.py`, `src/maxim/salience/movement_detector.py`

---

### 29. Inconsistent Serialization Patterns (4 different styles)

**Problem:** Most subsystems use `to_dict()/from_dict()` (standard), but:
- `similarity/lsh.py` uses `serialize()/deserialize()` — different names, and `deserialize()` mutates self instead of returning a new instance
- `spatial/bounds_learner.py` uses raw `json.dump()` with pathlib (no abstraction)
- `bridges/fear_bridge.py` has custom JSON with timestamp tracking

**Fix:** Standardize on `to_dict()/from_dict()` everywhere. Rename LSHIndex methods. Extract file I/O to persistence helpers where it's mixed in with serialization logic.

**Files:** `src/maxim/similarity/lsh.py`, `src/maxim/spatial/bounds_learner.py`, `src/maxim/bridges/fear_bridge.py`

---

### 30. LLMWorker Legacy Dual-Mode Execution Path

**Problem:** `llm_worker.py` maintains two parallel execution paths — modern WorkerPool mode and legacy internal queue/thread mode. Every lifecycle method (`start()`, `stop()`, `submit_context()`, `get_latest_proposal()`, `get_all_proposals()`) branches on which mode is active. This adds ~115 LOC of branching logic.

**Fix:** If WorkerPool is the standard path, remove the legacy queue/thread code. If both are needed, extract to a Strategy pattern so the branching happens once at init rather than in every method.

**Files:** `src/maxim/agents/llm_worker.py:155-724`

---

### 31. Duplicate Feature Detection in LLMWorker

**Problem:** `cloud_allowed()` check is duplicated at lines 142 and 636. `preview_provider()` check is duplicated at lines 494 and 820 — and the second instance lacks the `isinstance(preview, dict)` safety check the first has. `get_provider_configs()` is called with `hasattr` guards in 3 separate places.

**Fix:** Extract `_has_cloud_providers()` and `_get_provider_hint()` helper methods. Call once and cache or call consistently.

**Files:** `src/maxim/agents/llm_worker.py`

---

### 32. `training/losses.py` — Single Function, Likely Unused

**Problem:** The entire `training/` package contains one function (`euclidian_distance()`, 35 LOC) and an empty `__init__.py`. Grep finds zero imports of this function anywhere in the codebase.

**Fix:** If truly unused, remove the package. If needed, move the function to `src/maxim/math/`.

**Files:** `src/maxim/training/losses.py`, `src/maxim/training/__init__.py`

---

### 33. `data/audio/_file_based_transcription.py` — Dead Re-export Shim

**Problem:** This file (11 LOC) only re-exports `create_task_file` and `watch_and_transcribe` from `maxim.inference.transcribe_audio`. It's a backward-compat shim that nothing imports.

**Fix:** Delete the file.

**Files:** `src/maxim/data/audio/_file_based_transcription.py`

---

### 34. Three Skills Defined But Never Instantiated

**Problem:** `HealthReportingSkill`, `RTSPStreamingSkill`, and `TimedProtocolSkill` are defined but never instantiated anywhere in the codebase. Only `ShredderSegmenterProtocol` is actually wired in `agentic_runtime.py:517`.

**Fix:** If these are planned features, mark them clearly. If abandoned, remove them. Dead skill definitions add maintenance burden when skill interfaces change.

**Files:** `src/maxim/skills/health_reporting.py`, `src/maxim/skills/rtsp_streaming.py`, `src/maxim/skills/timed_protocol.py`

---

### 35. SpatialMemoryBridge and SalienceMemoryBridge Possibly Never Connected

**Problem:** These two bridges are exported from `bridges/__init__.py` but may never be instantiated in `memory_hub.py` or elsewhere. They appear to be lazy-initialized stubs or incomplete Phase 3 work.

**Fix:** Audit instantiation. If they're connected conditionally, document when. If never connected, consider removing or marking as stubs.

**Files:** `src/maxim/bridges/__init__.py`, `src/maxim/integration/memory_hub.py`

---

### 36. Singleton Boilerplate Repeated Across 8 Modules

**Problem:** Eight modules independently implement the same `_get_X_singleton()` + `get_X()` + optional `set_X()` pattern for module-level singletons. Each is 15-20 lines of near-identical code.

**Fix:** Create a lightweight `singleton_factory()` helper that generates the accessor functions, or use a simple module-level registry dict.

**Files:** `src/maxim/utils/structured_logging.py`, `src/maxim/utils/web_cache.py`, `src/maxim/utils/content_safety.py`, `src/maxim/utils/output_watcher.py`, `src/maxim/utils/agent_output.py`, `src/maxim/utils/sandbox_executor.py`, `src/maxim/utils/filesystem_policy.py`

---

### 37. Magic Number Sprawl (Timeouts, Buffer Sizes, Ports)

**Problem:** Hardcoded numeric constants scattered across 10+ modules with no central definition:
- Ports: 7447, 8000, 8443, 5000 (in `reachy_diagnostics.py`, `bootstrap.py`)
- Timeouts: 2s GPU check, 2s poll interval, 30s sandbox timeout
- Buffer sizes: 500 max entries, 1000 max events, 900s cache TTL, 300s DNS TTL
- Cooldowns: `2.0` repeated in 20+ `cooldown_s` defaults

**Fix:** Create `src/maxim/utils/constants.py` with named constants grouped by domain. Import where needed.

**Files:** Multiple (see list in description)

---

### 38. Public Methods That Should Be Private in similarity/signature.py

**Problem:** `hamming_distance()`, `structural_match()`, and `context_match()` (lines 160, 166, 186) are public methods but only called internally within the same class. No external callers.

**Fix:** Prefix with `_` to mark as private.

**Files:** `src/maxim/similarity/signature.py`

---

## Open Issues (Audit Round 4)

### 39. Evaluation Module is Vestigial (~311 LOC, zero decision-making value)

**Problem:** Three evaluators (`AgentEvaluator`, `PlanEvaluator`, `ToolExecutionEvaluator`) are instantiated in `bootstrap.py` and called once per agent step, but they're trivial stubs:
- `AgentEvaluator`: checks `confidence >= 0.5` (hardcoded threshold)
- `PlanEvaluator`: checks if all tools are registered (duplicates `DecisionEngine` constraint logic)
- `ToolExecutionEvaluator`: checks `result.success` (boolean)

Results are passed to `on_step()` callback but **never used for any decision-making**. `metrics.py` (5 LOC) defines two helper functions that are never called. `llm_benchmark.py` (83 LOC) is a standalone CLI tool misplaced in the evaluation package.

**Fix:** Either:
- **Option A:** Remove the module entirely (~300 LOC saved, zero functional loss)
- **Option B:** Keep `base.py` as an extension point, move stubs to `examples/`, move `llm_benchmark.py` to `scripts/`

**Files:** `src/maxim/evaluation/`

---

### 40. `ReachyConnection` Class is Dead Code (336 LOC)

**Problem:** `connection.py` defines `ReachyConnection` (lines 121-456) with full lifecycle management, `FailureTracker`, and `ConnectionState` enum. But **nothing imports `ReachyConnection`** — the actual connection logic uses `ConnectionMixin` (lines 458-708) which is mixed into `Maxim` via `selfy.py`. Two parallel implementations of the same feature; one is dead.

**Fix:** Remove `ReachyConnection` class (lines 121-456). Verify `FailureTracker` and `ConnectionState` aren't used elsewhere; if not, remove them too.

**Files:** `src/maxim/conscience/connection.py`

---

### 41. Movement Step-Clamping Boilerplate (138 LOC)

**Problem:** `movement.py` `move()` method (lines 860-997) has 138 lines of identical per-axis boilerplate for x, y, z, roll, pitch, yaw, body_yaw. Each axis does the same try/float/clamp pattern.

**Fix:** Extract `_clamp_axis(current, target, axis_name, max_step_dict) -> float` helper (~5 LOC). Replace 7 identical blocks with 7 one-line calls.

**Files:** `src/maxim/conscience/movement.py:860-997`

---

### 42. Dead Methods in selfy.py: `learn()` and `journal()`

**Problem:** `learn()` (line 610-611) is a no-op stub (`return`). `journal()` (line 613-618) creates a dict and returns it. Neither is called anywhere in the codebase.

**Fix:** Remove both methods.

**Files:** `src/maxim/conscience/selfy.py`

---

### 43. `TurnAround` Missing from `behaviors/__init__.py` Export

**Problem:** `TurnAround` behavior is defined in `turn_around.py`, registered in `config.py`, and used in `network.py`, but it's missing from `behaviors/__init__.py` imports and `__all__`. External code trying `from maxim.default_network.behaviors import TurnAround` will fail.

**Fix:** Add `from maxim.default_network.behaviors.turn_around import TurnAround` and include in `__all__`.

**Files:** `src/maxim/default_network/behaviors/__init__.py`

---

### 44. `DNActionProposal` Duplicates `ActionProposal` (messages.py)

**Problem:** `DNActionProposal` (21 LOC) and `ActionProposal` (40 LOC) in `default_network/messages.py` have heavily overlapping fields. `DNActionProposal` adds `timestamp` and `was_executed` but is otherwise identical. Only used for bus publishing.

**Fix:** Merge into `ActionProposal` by adding the two extra fields with defaults.

**Files:** `src/maxim/default_network/messages.py`

---

### 45. `Microsaccades.note_movement()` Never Called

**Problem:** `note_movement()` method (line 292-294 of `idle.py`) is defined but never called anywhere in the codebase. The fixation timer that Microsaccades depends on may never reset properly as a result, making Microsaccades potentially non-functional.

**Fix:** Either wire `note_movement()` into the perception loop (if Microsaccades should work) or remove the method and document Microsaccades as inactive.

**Files:** `src/maxim/default_network/behaviors/idle.py`

---

### 46. `InternetEnv` is an Unused Stub (67 LOC)

**Problem:** `environment/internet_env.py` defines `InternetEnv` with URL tracking and response history, but it's never instantiated anywhere. Only `FileSystemEnv` and `ReachyEnv` are used.

**Fix:** Remove the file, or mark as future work with a clear status comment.

**Files:** `src/maxim/environment/internet_env.py`

---

### 47. TensorFlow/Keras Should Be Optional Dependencies

**Problem:** `tensorflow==2.20.0` and `keras==3.13.0` are core dependencies but only used in 4 files: `cli.py` (GPU detection, wrapped in try/except), `selfy.py` (training checkpoint loading), `motor_cortex.py` (model definition), `training/losses.py` (with numpy fallback). All usage is conditional/training-only. This forces ~1.5GB of downloads for inference-only deployments.

**Fix:** Move to a new `training` optional group in pyproject.toml. All existing import sites already have try/except guards, so no code changes needed.

**Files:** `pyproject.toml`

---

### 48. `h5py` Has Zero Usage

**Problem:** `h5py==3.15.1` is a core dependency but grep finds zero direct imports. Keras model loading in `selfy.py` uses `.keras` format, not `.h5`. h5py is only needed for legacy H5 format.

**Fix:** Remove from core dependencies. Add to the `training` optional group if legacy checkpoint support is needed.

**Files:** `pyproject.toml`

---

### 49. `matplotlib` Unused in Runtime

**Problem:** `matplotlib==3.10.8` is a core dependency but only imported in `utils/plotting.py` for a font-preloading utility used during development. Never imported in production runtime code.

**Fix:** Move to an optional `dev` group.

**Files:** `pyproject.toml`

---

### 50. `protobuf` Pin is Unnecessary

**Problem:** `protobuf>=3.20.3,<6.0.0` is explicitly pinned but never directly imported. It's only a transitive dependency of onnxruntime and tensorflow. Pinning it explicitly can cause version conflicts.

**Fix:** Remove the explicit pin. Let onnxruntime's own dependency constraints handle it.

**Files:** `pyproject.toml`

---

### 51. Version Pins Overly Strict

**Problem:** 12+ dependencies use exact `==` pins (`numpy==2.2.5`, `scipy==1.15.3`, `opencv-python==4.12.0.88`, etc.). This prevents security patches and forces manual bumps for every update.

**Fix:** Change to range constraints where the package follows semver: `numpy>=2.2,<3.0`, `scipy>=1.15,<2.0`, `opencv-python>=4.12,<5.0`, etc.

**Files:** `pyproject.toml`

---

### 52. Stale Exports in `prompts/__init__.py`

**Problem:** `InjectedPrompt` and `PromptPrompt` are exported in `prompts/__init__.py` but never imported by any other module. Only `ExecutivePrompt` and `load_prompt_profile` are used externally (in `exec_agent.py`).

**Fix:** Remove from `__all__` and imports.

**Files:** `src/maxim/prompts/__init__.py`

---

### 53. `AdaptivePolicy.explain_score()` is Dead Code

**Problem:** `explain_score()` method (line 60-71 in `adaptive_policy.py`) is defined but never called anywhere. It's a debug helper that was never wired in.

**Fix:** Remove the method (13 LOC).

**Files:** `src/maxim/planning/adaptive_policy.py`

---

### 54. Debug `print()` Statements in CLI Sim Mode

**Problem:** `cli.py` lines ~505, 515, 516 have unconditional `print()` statements in the simulation initialization path. These should be gated behind `--sim-debug`.

**Fix:** Wrap with `if getattr(args, "sim_debug", False):`.

**Files:** `src/maxim/cli.py`

---

## Simplification Opportunities (Optional Refactoring)

These aren't bugs — they're structural improvements for long-term maintainability. Only pursue when actively working in the area.

### S1. Extract Mode Context Prompts & Tool Descriptions from definitions.py

`modes/definitions.py` (1,254 LOC) has ~550 LOC of inline data: context_prompt strings in STRATEGIES (~60 LOC), OPERATIONAL_MODES (~80 LOC), and TOOL_DESCRIPTIONS (~230 LOC). Moving to external files would cut the file nearly in half and make prompts/descriptions editable without touching Python.

### S2. Split bus.py — See [file_splitting_plan.md](file_splitting_plan.md)

### S3. Split agent_loop.py — See [file_splitting_plan.md](file_splitting_plan.md)

### S4. Externalize LLM Profiles from router.py

`models/language/router.py` has ~350 LOC of hardcoded data: `_BUILTIN_PROFILES` (9 model profiles, ~120 LOC), `QUANTIZATION_LEVELS` (~15 LOC), `_PROFILE_ALIASES` (~30 LOC), `_DEFAULT_PRICING` (~8 LOC). Moving to `data/models/llm_profiles.json` would reduce the file and make profiles editable without code changes.

### S5. Merge Pain Bridges

`PainCircuitBridge` (605 LOC, proprioceptive pain) and `ToolPainBridge` (378 LOC, tool execution errors) both register with NAc and implement PromotionSource. Could unify into a single configurable pain bridge.

### S6. Consider Consolidating DefaultNetworkConfig

`DefaultNetworkConfig` has 14 sub-configs plus ~40 individual fields. Grouping into fewer subsystem configs (e.g., `SalienceSystemConfig`, `GazeSystemConfig`) would reduce surface area.

### S7. Remove LLMWorker Pass-Through Wrappers

`llm_worker.py` lines 1011-1127 contain ~25 `@staticmethod` methods that delegate to `prompt_builder` and `llm_fallback` with identical signatures. Callers could import the originals directly. Removing these wrappers cuts ~120 LOC and eliminates dual-maintenance.

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
| 8 | Unused PerceptSource | Small | Type safety / dead code | Open |
| 9 | Any type overuse | Medium | Type safety | Open |
| 10 | Silent exceptions | Small | Debuggability | Open |
| 11 | Missing sim tests | Medium | Test coverage | Open |
| 12 | Config mismatch | Trivial | Consistency | Open |
| 13 | Stale re-exports | Small | Code clarity | Open |
| 14 | ScenarioRunner race | Small | Thread safety | Open |
| 15 | README profiles | Small | Doc accuracy | Open |
| 16 | No CI/CD pipeline | Small | Regression safety | Open |
| 17 | Broken sim test paths | Trivial | Recovers 5 tests | Open |
| 18 | Benchmark discovery | Small | Recovers 9 benchmarks | Open |
| 19 | Duplicate _env_flag | Trivial | DRY | Open |
| 20 | Deprecated localhost_only | Small | API clarity | Open |
| 21 | Deprecated _reachy attr | Small | API clarity | Open |
| 22 | Hardcoded RTSP port | Small | Maintainability | Open |
| 23 | Pass-through statics | Small | Code clarity | Open |
| 24 | Dead ddgs fallback | Trivial | Dead code removal | Open |
| 25 | CommunicationBridge unused | Small | Dead code audit | Open |
| 26 | Plans missing status | Small | Doc clarity | Open |
| 27 | Env bool parsing sprawl | Small | DRY, consistency | Open |
| 28 | Duplicate velocity calc | Medium | DRY, correctness | Open |
| 29 | Inconsistent serialization | Medium | Consistency | Open |
| 30 | LLMWorker dual-mode | Medium | -115 LOC, clarity | Open |
| 31 | LLMWorker feature detect dup | Small | Correctness (missing isinstance) | Open |
| 32 | training/ unused package | Trivial | Dead code removal | Open |
| 33 | Dead transcription shim | Trivial | Dead code removal | Open |
| 34 | Three unused skills | Small | Dead code audit | Open |
| 35 | Two bridges never connected | Small | Dead code audit | Open |
| 36 | Singleton boilerplate 8x | Medium | -120 LOC, DRY | Open |
| 37 | Magic number sprawl | Medium | Maintainability | Open |
| 38 | Public methods should be private | Trivial | API clarity | Open |
| 39 | Evaluation module vestigial | Small | -300 LOC or move to examples | Open |
| 40 | ReachyConnection dead code | Small | -336 LOC | Open |
| 41 | Movement step-clamping boilerplate | Small | -120 LOC | Open |
| 42 | Dead methods: learn(), journal() | Trivial | Dead code removal | Open |
| 43 | TurnAround missing from exports | Trivial | Bug fix | Open |
| 44 | DNActionProposal duplicates ActionProposal | Small | -21 LOC, fewer types | Open |
| 45 | Microsaccades.note_movement() never called | Small | Bug or dead code | Open |
| 46 | InternetEnv unused stub | Trivial | -67 LOC | Open |
| 47 | TensorFlow/Keras should be optional | Small | -1.5GB install size | Open |
| 48 | h5py zero usage | Trivial | Smaller install | Open |
| 49 | matplotlib unused in runtime | Trivial | Smaller install | Open |
| 50 | protobuf pin unnecessary | Trivial | Fewer conflicts | Open |
| 51 | Version pins overly strict | Small | Allow security patches | Open |
| 52 | Stale exports in prompts/__init__ | Trivial | API clarity | Open |
| 53 | AdaptivePolicy.explain_score() dead | Trivial | -13 LOC | Open |
| 54 | Debug print() in CLI sim mode | Trivial | UX | Open |
| S1 | Extract mode prompts/tool descs | Medium | -550 LOC in definitions.py | Optional |
| S2 | Split bus.py | Medium | Clarity | Optional |
| S3 | Split agent_loop.py | Large | Clarity, testability | Optional |
| S4 | Externalize LLM profiles | Medium | -350 LOC in router.py | Optional |
| S5 | Merge pain bridges | Medium | -200 LOC, fewer concepts | Optional |
| S6 | Consolidate DN config | Low | Cleaner config surface | Optional |
| S7 | Remove LLMWorker wrappers | Small | -120 LOC, clarity | Optional |

## Suggested Execution Order

**Phase 1 — Trivial wins (< 30 min total):**
1. **#12** Config mismatch (one line)
2. **#17** Fix sim test paths (recover 5 tests)
3. **#19** Consolidate `_env_flag()`
4. **#24** Remove dead ddgs fallback
5. **#32** Remove `training/` package (unused)
6. **#33** Remove dead transcription shim
7. **#38** Prefix private methods in signature.py
8. **#42** Remove dead `learn()`, `journal()` in selfy.py
9. **#43** Fix TurnAround export (2-line fix)
10. **#46** Remove InternetEnv stub
11. **#52** Remove stale prompts/__init__ exports
12. **#53** Remove dead `explain_score()` method
13. **#54** Gate debug print() behind --sim-debug
14. **#48** Remove h5py from core deps
15. **#49** Move matplotlib to optional dev group
16. **#50** Remove explicit protobuf pin

**Phase 2 — Small stabilization fixes (1-2 hours):**
17. **#10** Silent exceptions → `logger.debug()`
18. **#1** Duplicate step counter removal
19. **#7** CWD try/finally protection
20. **#13** Remove stale re-exports
21. **#14** ScenarioRunner local vars
22. **#23** Remove pass-through statics (S7)
23. **#31** Fix LLMWorker duplicate feature detection
24. **#40** Remove dead ReachyConnection class (-336 LOC)
25. **#41** Extract movement step-clamping helper (-120 LOC)
26. **#44** Merge DNActionProposal into ActionProposal

**Phase 3 — Dead code audit (1-2 hours):**
27. **#25** Audit CommunicationBridge
28. **#34** Audit three unused skills
29. **#35** Audit SpatialMemory/SalienceBridge connection
30. **#39** Decide on evaluation module (remove or move to examples)
31. **#45** Investigate Microsaccades.note_movement() (bug or dead?)
32. **#27** Consolidate env bool parsing to `env_flag()`

**Phase 4 — Documentation & process (1-2 hours):**
33. **#26** Add status headers to unstarted plans
34. **#15** Fix README profiles section
35. **#16** Add GitHub Actions CI workflow
36. **#18** Fix benchmark function naming

**Phase 5 — Dependency cleanup (1 hour):**
37. **#47** Move TensorFlow/Keras to optional `training` group
38. **#51** Relax version pins to range constraints

**Phase 6 — Medium fixes (half day):**
39. **#8** Wire PerceptSource protocol
40. **#5** Variable scoping cleanup
41. **#22** Consolidate RTSP constant
42. **#20** Remove deprecated localhost_only
43. **#21** Migrate _reachy → _robot
44. **#6** Batch scenario processing
45. **#29** Standardize serialization patterns
46. **#37** Extract magic numbers to constants.py

**Phase 7 — Larger improvements (1-2 days):**
47. **#3** Share LLM backend in simulation
48. **#9** Protocol types for runtime interfaces
49. **#11** Simulation module unit tests
50. **#30** Remove LLMWorker legacy dual-mode
51. **#28** Extract shared velocity calculation
52. **#36** Consolidate singleton boilerplate
53. **#4** Metal kernel warning suppression
54. **#2** Remove remaining PIPELINE trace

**Phase 8 — Structural splits (see [file_splitting_plan.md](file_splitting_plan.md)):**
55. **S2** Split bus.py into package (low risk)
56. **S1** Extract prompts/tool descriptions from definitions.py
57. **S7** Delete LLMWorker wrappers (if not done in Phase 2)
58. **S4** Externalize LLM profiles from router.py
59. **S3** Split agent_loop.py (highest risk, do last)
