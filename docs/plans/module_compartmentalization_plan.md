# Module Compartmentalization Plan

> **Status:** Not started
> **Goal:** Break up the 5 largest god-modules into focused, single-responsibility files without changing behavior.
> **Estimated scope:** ~0 net LOC (pure refactor — moves code, doesn't add or remove functionality)
> **Sequence:** Executes AFTER the API Surface Hardening Plan. The integration tests from that plan serve as a safety net for this refactor.
> **Prerequisite:** Integration tests from API Surface Hardening Phase 4 must be passing.
> **Timeframe:** 2-3 focused sessions

---

## Why This Matters

Five modules have grown past the point where they can be understood, debugged, or reviewed as units:

| Module | Lines | Core Problem |
|--------|-------|-------------|
| `runtime/agent_loop.py` | 2,312 | `run_agentic_loop()` is a 1,700-line function mixing loop control, tool dispatch, autonomy gating, bio-system integration, and error recovery |
| `simulation/orchestrator.py` | 1,743 | Campaign loading, simulation lifecycle, research protocol wiring, and report generation in one file |
| `cli.py` | 1,687 | Argument parsing, command dispatch, agent bootstrap, and display logic interleaved |
| `models/language/router.py` | 1,349 | LLM routing, cost tracking, cloud redaction, provider state management, and JSON repair coordination |
| `runtime/lane_backends.py` | 1,352 | Lane model loading, tier detection, server lifecycle, caching, and hot-swap logic |

These modules are where bugs hide. The research protocol's 5 bugs all live in `orchestrator.py`. The `run()` TypeError lived in the interaction between `api.py` and `agent_loop.py`. Silent error swallowing concentrates in `lane_backends.py` (125 handlers) and `orchestrator.py` (114 handlers).

**This is a pure structural refactor.** No behavior changes. No new features. Every public import path continues to work. The goal is smaller files that are easier to read, test, and debug.

---

## Guiding Principles

1. **Extract, don't rewrite.** Move code blocks verbatim into new files. Resist the urge to "improve" while moving.
2. **One module, one responsibility.** Each new file should have a clear single-sentence purpose.
3. **Preserve public imports.** If `from maxim.runtime.agent_loop import run_agentic_loop` works today, it works after. Use re-exports where needed.
4. **Test after every extraction.** Run `python -m pytest tests/ -x -q` after each file split. Don't batch extractions.
5. **Don't change error handling during refactor.** That's the API Surface Hardening Plan's job. Move the `except Exception:` blocks as-is.
6. **Keep diffs reviewable.** Each commit should be one extraction (one new file + one slimmed original). Avoid multi-file reorganizations in a single commit.

---

## Phase 1 — `agent_loop.py` (2,312 → ~800 + 4 new files)

This is the highest-value extraction. `run_agentic_loop()` currently handles:
- Loop lifecycle (start, step, shutdown)
- Tool dispatch and result processing  
- Autonomy level enforcement (planning/supervised/autonomous)
- Bio-system integration (NAc, hippocampus, energy, pain)
- Error recovery and retry logic

### 1a. Extract tool dispatch → `runtime/tool_dispatch.py`

**What moves:** The tool execution pipeline — tool lookup, argument preparation, execution, result processing, alias resolution, usage tracking.

**Boundary:** Takes a tool call from the LLM response, returns a tool result. Does not know about the loop, autonomy, or bio-systems.

**Estimated size:** ~400 LOC

**Current locations to extract from:**
- Tool call parsing and validation
- `TOOL_ALIASES` dict and alias resolution logic
- Tool execution with timeout and error handling
- Result formatting and truncation
- Usage statistics tracking (`tool_usage_stats()`)

### 1b. Extract autonomy gating → `runtime/autonomy.py`

**What moves:** The permission/approval logic that gates tool execution based on the current autonomy level (planning, supervised, autonomous).

**Boundary:** Takes a tool call + current autonomy level, returns allow/deny/ask-user. Does not execute the tool.

**Estimated size:** ~200 LOC

**Current locations:**
- FearAgent integration (pre-execution review)
- User approval flow for supervised mode
- Planning mode constraints
- Autonomy level transitions

### 1c. Extract bio-system integration → `runtime/bio_integration.py`

**What moves:** The per-step bio-system calls — NAc observation, hippocampus capture queueing, energy tracking, pain signal processing, salience updates.

**Boundary:** Called once per loop step with the step's context. Updates bio-systems. Returns bio-signals (pain, energy, salience) that the loop uses for control flow.

**Estimated size:** ~300 LOC

**Current locations:**
- NAc `observe()` calls after tool results
- Hippocampus capture queueing
- Energy token/cost tracking per step
- Pain signal processing and fear gating
- Salience/novelty updates

### 1d. Slim `agent_loop.py` to orchestration only

**What remains:** The main loop skeleton — init, step, shutdown. Each step calls:
1. LLM inference (already delegates to router)
2. Tool dispatch (now in `tool_dispatch.py`)
3. Autonomy check (now in `autonomy.py`)
4. Bio-system update (now in `bio_integration.py`)
5. Loop control (continue/sleep/stop)

**Target:** ~800 LOC — readable in one sitting.

**Verify:** 
```bash
python -m pytest tests/unit/test_agent_loop.py tests/unit/test_executor.py -v
python -m pytest tests/integration/ -v
```

---

## Phase 2 — `orchestrator.py` (1,743 → ~600 + 3 new files)

### 2a. Extract campaign loading → `simulation/campaign_loader.py`

**What moves:** YAML loading, campaign validation, turn extraction, pre-campaign setup.

**Boundary:** Takes a file path, returns a structured campaign object (turns, metadata, expectations). Does not run anything.

**Estimated size:** ~300 LOC

### 2b. Extract report generation → `simulation/report_generator.py`

**What moves:** Post-simulation report building — metrics collection, JSON serialization, file writing, summary formatting.

**Boundary:** Takes simulation results, writes report files, returns report summary. Does not know about the simulation lifecycle.

**Estimated size:** ~250 LOC

### 2c. Extract research protocol wiring → keep in `research_orchestrator.py`

The research wiring is already partially in `research_orchestrator.py`, but `orchestrator.py` still owns the `--research` flag handling, experiment log creation, and Writer/Reviewer invocation.

**What moves:** All research-specific logic from `orchestrator.py` into `research_orchestrator.py`. The orchestrator should just call `research_orchestrator.run()` with the sim results.

**Estimated size:** ~200 LOC moved (net zero — already exists in target file)

### 2d. Slim `orchestrator.py` to lifecycle only

**What remains:** Simulation lifecycle — init, configure, run loop, shutdown. Delegates to:
- `campaign_loader.py` for YAML parsing
- `research_orchestrator.py` for research protocol
- `report_generator.py` for output

**Target:** ~600 LOC

**Verify:**
```bash
python -m pytest tests/unit/test_simulation_agent.py tests/unit/test_orchestrator*.py -v
python -m pytest tests/integration/test_research_pipeline.py -v
```

---

## Phase 3 — `cli.py` (1,687 → ~500 + 3 new files)

### 3a. Extract argument parsing → `cli_args.py`

**What moves:** All `argparse` setup — argument definitions, groups, validation, help text.

**Boundary:** Returns a parsed `Namespace`. Does not import any runtime modules.

**Estimated size:** ~500 LOC (argument definitions are verbose)

### 3b. Extract command dispatch → `cli_commands.py`

**What moves:** The `if args.sim: ... elif args.doctor: ... elif args.tunnel: ...` dispatch tree. Each branch becomes a function.

**Boundary:** Takes parsed args, calls the appropriate runtime entry point. One function per subcommand.

**Estimated size:** ~400 LOC

### 3c. Slim `cli.py` to entry point only

**What remains:** `main()` function that:
1. Parses args (delegates to `cli_args.py`)
2. Configures logging/display
3. Dispatches (delegates to `cli_commands.py`)
4. Handles top-level errors

**Target:** ~500 LOC

**Verify:**
```bash
python -m maxim --help
python -m maxim doctor --json 2>/dev/null
python -m pytest tests/unit/test_cli*.py -v
```

---

## Phase 4 — `router.py` (1,349 → ~600 + 2 new files)

### 4a. Extract cloud dispatch → `models/language/cloud_dispatch.py`

**What moves:** `CloudAuditLogger`, `CloudRedactionFilter`, cloud provider selection, cloud-specific retry logic, cost tracking for cloud calls.

**Boundary:** Takes a prompt + cloud config, returns an LLM response. Handles redaction, audit logging, and cost.

**Estimated size:** ~350 LOC

### 4b. Extract provider state → `models/language/provider_state.py`

**What moves:** Provider initialization, health tracking, failover logic, warm/cold state management.

**Boundary:** Manages which providers are available and healthy. The router queries it before dispatching.

**Estimated size:** ~200 LOC

### 4c. Slim `router.py` to routing logic only

**What remains:** The core routing decision — given a request, pick the best provider/tier, dispatch, return result. Delegates cloud specifics and provider health to extracted modules.

**Target:** ~600 LOC

**Verify:**
```bash
python -m pytest tests/unit/test_router*.py tests/unit/test_llm*.py -v
```

---

## Phase 5 — `lane_backends.py` (1,352 → ~500 + 2 new files)

### 5a. Extract server lifecycle → `runtime/llm_server.py`

**What moves:** llama-cpp-server process management — spawn, health check, shutdown, stale process detection, PID tracking, auto-spawn logic.

**Boundary:** Manages one or more LLM server processes. The lane system queries it for server URLs.

**Estimated size:** ~450 LOC

### 5b. Extract hot-swap logic → `runtime/model_swap.py`

**What moves:** `swap_llm_server()`, model persistence (`_read_persisted_model`, `_write_persisted_model`), the swap coordination protocol.

**Boundary:** Takes a model name, orchestrates the swap (shutdown old → start new → verify → persist). Uses `llm_server.py` for process management.

**Estimated size:** ~250 LOC

### 5c. Slim `lane_backends.py` to tier/lane wiring only

**What remains:** `build_primary_router()`, tier detection integration, lane configuration, backend selection.

**Target:** ~500 LOC

**Verify:**
```bash
python -m pytest tests/unit/test_lane_backends.py tests/unit/test_local_server_spawner.py -v
```

---

## Verification Protocol

After ALL phases complete:

```bash
# 1. Full test suite
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py

# 2. Integration tests (safety net)
python -m pytest tests/integration/ -v

# 3. Import paths preserved
python -c "from maxim.runtime.agent_loop import run_agentic_loop"
python -c "from maxim.simulation.orchestrator import start_simulation_mode"
python -c "from maxim.models.language.router import LLMRouter"
python -c "from maxim.runtime.lane_backends import build_primary_router"

# 4. CLI works
python -m maxim --help
python -m maxim doctor --json 2>/dev/null

# 5. Lint
ruff check src/ tests/
ruff format --check src/ tests/

# 6. No new files over 1000 lines
find src/maxim -name "*.py" -exec wc -l {} + | sort -rn | head -20
```

---

## Risk Mitigation

**Risk: Breaking imports.** Every extraction must re-export from the original module. Example:
```python
# agent_loop.py (after extraction)
from maxim.runtime.tool_dispatch import dispatch_tool, TOOL_ALIASES  # re-export
```
Remove re-exports in a future version (v0.3.0) after downstream code has time to update imports.

**Risk: Circular imports.** The extracted modules may need to import each other. Use late imports (inside function bodies) for any circular dependencies discovered during extraction. Document each one with a `# late import: avoids circular with X` comment.

**Risk: Test fragility.** Some tests patch internal functions by path (e.g., `@mock.patch("maxim.runtime.agent_loop._dispatch_tool")`). After extraction, the patch path changes. Fix each broken patch as you encounter it — this is expected and acceptable.

**Risk: Merge conflicts.** If the API Surface Hardening Plan modifies these files, do that work FIRST. This plan moves code; that plan changes code. Moving changed code is safe. Changing moved code creates conflicts.

---

## What This Does NOT Cover

- **Behavior changes** — no error handling improvements, no bug fixes, no feature additions
- **Test additions** — the API Hardening Plan adds integration tests; this plan just keeps them passing
- **Smaller modules** — files under 1,000 lines are fine. Don't over-decompose.
- **`selfy.py` decomposition** — tracked separately in future_plans.md under "Embodiment Hardware Adapter"
- **`exec_agent.py` / `bus.py`** — deferred to v0.2.1 (listed in refinement plan as god class candidates but under 1,000 lines)
