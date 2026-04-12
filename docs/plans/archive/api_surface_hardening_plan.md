# API Surface Hardening Plan

> **Status:** ALL PHASES COMPLETE (2026-04-08)
> **Goal:** Make every public interface honest — if it's in `__all__`, it works. If it doesn't work, it doesn't exist.
> **Estimated scope:** ~600 LOC of fixes + ~200 LOC of integration tests + ~200 LOC of docs
> **Sequence:** Executes BETWEEN Publication Refinement Plan Phase 0 (done) and Phase 4 (publish). **Replaces Phases 1-3** of the refinement plan for items covered here.
> **Timeframe:** 2-3 focused sessions

---

## Relationship to Publication Refinement Plan

This plan was created after a comprehensive repo review revealed that the refinement plan's Phases 1-3 underestimate the API surface issues. Specifically:

- **Refinement Phase 0** (hard blockers): DONE. Keep as-is.
- **Refinement Phases 1-3**: Items covered here should be **skipped** in the refinement plan (annotated there with `[SKIP — covered by API Surface Hardening Plan]`). Items NOT covered here remain active in the refinement plan.
- **Refinement Phase 4** (publish): Executes AFTER this plan completes.

### What this plan covers (skip in refinement plan)

| Refinement Item | Covered Here As |
|-----------------|-----------------|
| 0b (stub verb warnings) | Phase 1 (wire or remove — warnings aren't enough) |
| 1a (triage except blocks) | Phase 3 (user-facing path triage) |
| 1f (OpenAI rate limits) | Phase 3e |
| 1g (API key validation) | Phase 3f |
| 1j (hippocampus silent chain) | Phase 3 (part of user-facing error audit) |
| 1k (daemon thread pass) | Phase 3 (part of user-facing error audit) |
| 3a (broken links/stale status) | Phase 5a |
| 3b (trim CLAUDE.md) | Phase 5b |
| 3h (getting-started install path) | Phase 5c |
| 3i (@maxim.tool schema inference) | Phase 1d |

### What stays in the refinement plan (NOT covered here)

| Refinement Item | Why it stays |
|-----------------|-------------|
| 1b (hardcoded sandbox paths) | Security, not API surface |
| 1c (shell blocklist gaps) | Security, not API surface |
| 1d (replace critical Any) | Type quality, not functionality |
| 1e (atomic write bypasses) | Data integrity, not API surface |
| 1h (permission error handling) | Already in-flight (uncommitted changes) |
| 1i (env var parsing) | Already in-flight (uncommitted changes) |
| 1m-1n (store protocol wiring) | Mother Maxim on-ramp, separate concern |
| 1.5 (data directory migration) | Infrastructure, not API surface |
| 2a-2c (threading/behavioral/JSON tests) | Test depth, not API surface |
| 3c (packaging fixes) | Packaging, not API surface |
| 3d (htmls-guides README) | Housekeeping |
| 3e-3g (extras, diagnostics, git guard) | Packaging, not API surface |

---

## Phase 1 — Wire or Remove Stub Verbs (~3-4 hours)

The 3 stub verbs (`campaign`, `benchmark`, `research`) currently emit `warnings.warn()` and return empty objects. This is worse than `NotImplementedError` — it's silent failure dressed up as success.

**Principle:** If it's in `__all__`, it works end-to-end. If it can't work yet, raise `NotImplementedError` with a message pointing to the CLI alternative.

### 1a. Wire `campaign()` to PartyDMRuntime

**Current state:** Returns empty `CampaignResult` with a warning.

**CLI equivalent:** `maxim --sim scenarios/campaigns/heist_v1.yaml` works via `orchestrator.py` → `dm_runtime.py`.

**Action:**
1. Read `src/maxim/simulation/orchestrator.py` — find the DM campaign entry path (around the `_start_dm_campaign` section)
2. Read `src/maxim/simulation/dm_runtime.py` — understand `PartyDMRuntime` initialization
3. In `api.py:campaign()`, replace the stub with:
   - Load the YAML via `dm_schema.load_campaign()`
   - Create a `SimulationBridge` (or minimal equivalent)
   - Instantiate `PartyDMRuntime` and call `run()`
   - Return a populated `CampaignResult` with encounter results
4. If full wiring requires the orchestrator's setup (LLM, agent loop, etc.), delegate through `start_simulation_mode()` with campaign-specific args instead of reimplementing

**Files:** `src/maxim/api.py`, `src/maxim/simulation/dm_runtime.py`, `src/maxim/simulation/dm_schema.py`
**Verify:** `python -c "import maxim; r = maxim.campaign('scenarios/campaigns/heist_v1.yaml'); print(type(r))"`

### 1b. Wire `benchmark()` to BenchmarkRunner

**Current state:** Returns empty `BenchmarkResult` with a warning.

**CLI equivalent:** `maxim --sim benchmark --models mistral-7b --campaign scenarios/benchmarks/quick_check.yaml` works via `simulation/benchmark.py`.

**Action:**
1. Read `src/maxim/simulation/benchmark.py` — understand `BenchmarkRunner` initialization and `run()` return value
2. In `api.py:benchmark()`, replace the stub with:
   - Instantiate `BenchmarkRunner` with the provided models and suite path
   - Call `runner.run()` 
   - Return a populated `BenchmarkResult`
3. Handle the case where no models or suite are provided — raise `ConfigurationError` with guidance

**Files:** `src/maxim/api.py`, `src/maxim/simulation/benchmark.py`
**Verify:** Unit test that mocks LLM and verifies `BenchmarkResult` has populated fields

### 1c. Wire `research()` — requires fixing the research protocol first

**Current state:** Returns empty `ResearchResult` with a warning. But even the CLI path is broken (5 bugs, D-0a through D-0e).

**Action:** This is a two-step fix:

**Step 1 — Fix the 5 research protocol bugs:**

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| D-0a | Two separate `ExperimentLog` instances | Pass orchestrator's log to `research_orchestrator` instead of creating a new one |
| D-0b | `campaign_analysis` only built with `--campaign` | Build analysis dict unconditionally from sim results |
| D-0c | WriterAgent early-returns on empty experiments | Guard: if no experiments, log warning and skip (don't silently return None) |
| D-0d | Hardcoded `"memory_recall_verath"` key | Use the campaign's actual name/key from the analysis dict |
| D-0e | Reviewer rejects empty papers | Resolved by D-0c fix (Writer no longer produces empty papers) |

**Step 2 — Wire `api.py:research()`:**
1. Delegate through the orchestrator with `--research` flag equivalent
2. Return populated `ResearchResult` with paper path, metrics, experiment log

**Files:** `src/maxim/simulation/orchestrator.py`, `src/maxim/simulation/research_orchestrator.py`, `src/maxim/simulation/research_agents.py`, `src/maxim/api.py`
**Verify:** `maxim --sim "test recall" --research --campaign scenarios/experiments/hippocampal_recall_short.yaml` generates a `paper.md`

### 1d. `@maxim.tool` schema inference

**Current state:** The decorator registers the tool but doesn't infer `input_schema` from type annotations. The LLM can't discover tool parameters.

**Action:**
1. In `api.py`, update the `tool()` decorator to use `inspect.signature()` + `typing.get_type_hints()`
2. Map Python types to JSON Schema types (str→string, int→integer, float→number, bool→boolean, list→array)
3. Extract parameter descriptions from the docstring (Google/NumPy style) if available
4. Set `input_schema` on the registered tool spec

**Files:** `src/maxim/api.py`
**Scope:** ~30 LOC
**Verify:** Register a tool with typed params, verify the schema appears in tool spec

---

## Phase 2 — Fix Research Protocol (D-0a through D-0e) (~2 hours)

This is separated from Phase 1c because it's a prerequisite — the research verb can't be wired until the underlying protocol works.

### 2a. Single ExperimentLog instance (D-0a — root cause)

**Problem:** `orchestrator.py:~818` creates `ExperimentLog`. `research_orchestrator.py:79` creates a second one. The researcher writes to Log A; the Writer reads from Log B (always empty).

**Fix:** Pass the orchestrator's `ExperimentLog` instance to `research_orchestrator.run_research_protocol()` as a parameter. Remove the internal instantiation.

**Files:** `src/maxim/simulation/orchestrator.py`, `src/maxim/simulation/research_orchestrator.py`

### 2b. Build campaign_analysis unconditionally (D-0b)

**Problem:** Analysis dict built inside `if pre_campaign_turns:` block. Without `--campaign`, it's `{}`.

**Fix:** Build the analysis dict from simulation results regardless of whether a campaign YAML was provided. The sim always produces metrics, memories, and causal links — these should feed the analysis even in generative mode.

**Files:** `src/maxim/simulation/orchestrator.py`

### 2c. WriterAgent empty experiment guard (D-0c)

**Problem:** Writer early-returns on empty experiments without saving `paper.md`. `draft.output_path` stays `None`.

**Fix:** If experiments list is empty, Writer should:
1. Log a warning: "No experiments recorded — generating observational paper from available metrics"
2. Generate a paper from the analysis dict alone (metrics, memory state, causal links)
3. Always save to `paper.md` — even if thin

**Files:** `src/maxim/simulation/research_agents.py`

### 2d. Remove hardcoded key (D-0d)

**Problem:** `research_orchestrator.py:158` uses `analysis.get("memory_recall_verath", {})`.

**Fix:** Use the campaign's actual key. The analysis dict should use the campaign name (from YAML metadata) as the key, or iterate all entries.

**Files:** `src/maxim/simulation/research_orchestrator.py`

### 2e. Verify D-0e resolves

**Problem:** Reviewer auto-rejects empty papers.

**Verify:** After D-0c fix, Writer always produces content → Reviewer gets real input → auto-resolve.

**Files:** `src/maxim/simulation/research_agents.py` (verify only, no code change expected)

---

## Phase 3 — Error Handling on User-Facing Paths (~2-3 hours)

Not all 593 `except Exception:` blocks. Just the ones users will hit.

### 3a. Audit API entry points

**Scope:** `src/maxim/api.py` — every public verb function.

**Action:** For each `except Exception` block:
- If the caller needs to know → re-raise as appropriate `MaximError` subclass with context
- If it's best-effort (telemetry, optional subsystem) → `logger.warning()` with the exception
- If truly inconsequential → add a comment explaining why

**Target:** Zero silent `except Exception: pass` in `api.py`.

### 3b. Audit bootstrap path

**Scope:** `src/maxim/runtime/bootstrap.py`, `src/maxim/cli.py` (the startup path from CLI → agent loop).

**Action:** Trace the path from `maxim` CLI entry point through bootstrap to `run_agentic_loop()`. At each `except Exception`:
- Startup failures (can't find model, can't create data dir, can't parse config) → fail fast with actionable error
- Runtime failures (LLM timeout, tool error) → log and continue (these are expected)

### 3c. Audit LLM router

**Scope:** `src/maxim/models/language/router.py`

**Action:** The router is the most-called internal API. Focus on:
- Provider initialization failures → `ModelError` with provider name and fix hint
- Inference failures → distinguish between transient (retry) and permanent (raise)
- Cloud dispatch failures → `CloudDispatchError` with cost context

### 3d. Audit memory subsystem user-facing paths

**Scope:** `src/maxim/memory/hippocampus.py` (the silent failure chains from refinement plan 1j)

**Action:** Fix the `state.snapshot() → None → AttributeError` chain. Apply same pattern to all maybe-None results from caught exceptions (~5 instances in hippocampus.py).

### 3e. OpenAI rate limit backoff

**Problem:** Linear backoff on 429s. Should be exponential like the Anthropic backend.

**Fix:** Port `_is_rate_limit_error()` check and `backoff * 4` pattern from `anthropic_backend.py`.

**Files:** `src/maxim/models/language/openai_backend.py`

### 3f. API key fail-fast at startup

**Problem:** Missing API key errors surface minutes later during inference.

**Fix:** After model selection in CLI, validate required API key exists. Exit immediately with: `"Error: {model} requires {KEY_NAME}. Set it with: export {KEY_NAME}=..."`

**Files:** `src/maxim/cli.py`

---

## Phase 4 — Integration Tests (~2 hours)

The unit tests mock everything. These integration tests verify real wiring.

### 4a. API verb integration tests

**File:** `tests/integration/test_api_verbs.py` (~100 LOC)

Test each verb with mocked LLM but real wiring:

```python
def test_run_starts_and_stops():
    """maxim.run() with mock LLM completes without error."""
    
def test_imagine_returns_session():
    """maxim.imagine() returns Session with populated fields."""
    
def test_campaign_returns_results():
    """maxim.campaign() with a test YAML returns CampaignResult with encounters."""
    
def test_benchmark_returns_comparison():
    """maxim.benchmark() returns BenchmarkResult with per-model metrics."""

def test_diagnose_returns_report():
    """maxim.diagnose() returns DiagnosticReport with checks."""

def test_observe_returns_state():
    """maxim.observe('memory') returns dict with expected keys."""
```

### 4b. Bootstrap-to-loop integration test

**File:** `tests/integration/test_bootstrap.py` (~50 LOC)

```python
def test_bootstrap_with_mock_llm():
    """Full bootstrap path: CLI args → config → LLM init → loop start → clean shutdown."""
    
def test_bootstrap_missing_api_key_fails_fast():
    """Cloud model without API key exits immediately with actionable message."""
    
def test_bootstrap_invalid_model_suggests_alternatives():
    """Unknown model name produces helpful 'did you mean' suggestion."""
```

### 4c. Research pipeline integration test

**File:** `tests/integration/test_research_pipeline.py` (~50 LOC)

```python
def test_research_produces_paper():
    """Full research pipeline: sim → experiment log → writer → paper.md exists."""
    
def test_research_single_experiment_log():
    """Only one ExperimentLog instance exists throughout the pipeline."""
```

---

## Phase 5 — README & Docs Overhaul (~1-2 hours)

### 5a. Fix broken links and stale status

Same items as refinement plan 3a, plus:
- Update verb status table in `publication_refinement_plan.md` after Phase 1 wiring
- Remove "CLI broken" status for research if Phase 2 fixes land

### 5b. Trim CLAUDE.md

Same as refinement plan 3b. Target: under 250 lines.

### 5c. README overhaul

**Current:** 606 lines, buries getting-started, leads with robot/hardware language.

**Target:** ~200 lines.

**Structure:**
1. **One-liner** — what Maxim is (cognitive architecture for AI agents)
2. **5-minute quickstart** — `pip install pymaxim[llm-anthropic]` → `maxim doctor` → `maxim --sim "test memory"`
3. **What you can do** — 4 bullet points (simulate, benchmark, campaign, connect robot)
4. **Installation** — core + extras table (compact)
5. **Python API** — 5-line example showing `import maxim; maxim.imagine()`
6. **Links** — docs, contributing, license

Move everything else (architecture glossary, full CLI reference, bio-system table) to docs/.

### 5d. Fix mode terminology conflict

**Problem:** README uses old mode names (exploration, live, sleep). `modes-guide.md` uses new names (planning, supervised, autonomous).

**Fix:** Update README to use current terminology. Add a one-line note: "Previously called exploration/live/agentic — now planning/supervised/autonomous."

---

## Verification Checklist (run before declaring done)

```bash
# 1. All 13 API verbs respond (not stub, not error)
python -c "
import maxim
print('configure:', type(maxim.configure))
print('run:', type(maxim.run))
print('imagine:', type(maxim.imagine))
print('connect:', type(maxim.connect))
print('diagnose:', type(maxim.diagnose))
print('observe:', type(maxim.observe))
print('list_models:', type(maxim.list_models))
print('download_model:', type(maxim.download_model))
print('delete_model:', type(maxim.delete_model))
print('campaign:', type(maxim.campaign))
print('benchmark:', type(maxim.benchmark))
print('research:', type(maxim.research))
print('on:', type(maxim.on))
print('register_tool:', type(maxim.register_tool))
print('register_persona:', type(maxim.register_persona))
print('tool:', type(maxim.tool))
"

# 2. No silent except passes in API surface
grep -n "except Exception" src/maxim/api.py | grep -v "logger\|raise\|warn"
# Should return nothing

# 3. Tests pass
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py

# 4. Integration tests pass
python -m pytest tests/integration/test_api_verbs.py tests/integration/test_bootstrap.py -v

# 5. Clean import
python -c "import maxim" && echo "OK"

# 6. Lint
ruff check src/maxim/api.py src/maxim/simulation/orchestrator.py src/maxim/simulation/research_orchestrator.py
```

---

## What comes after this plan

1. **Commit the in-flight changes** (12 modified files — defensive coding, test hygiene)
2. **Complete remaining refinement plan items** (1b-1e security/integrity, 1m-1n store protocols, 1.5 data migration, 2a-2c test depth, 3c-3g packaging)
3. **Module Compartmentalization Plan** (separate plan — break up god-functions)
4. **Refinement Phase 4** (publish to PyPI)
