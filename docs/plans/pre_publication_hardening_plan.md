# Phase 12b: Pre-Publication Hardening Plan

> **Status:** Partially done (12b.0 cv2 imports + error hierarchy shipped, remaining items tracked)
> **Parent plan:** [foundational_buildout_plan.md](foundational_buildout_plan.md) — this is Phase 12b, runs in parallel with Phase 11 (Test PyPI + Publish).
> **Goal:** Fix every issue that would make a `pip install pymaxim` user's first 30 minutes frustrating, broken, or confusing. Ruthless triage — ship quality, not completeness.
> **Estimated scope:** ~2,500 LOC across 7 sub-phases (12b.0 – 12b.6)
> **Sequence:** Blocking bugs (12b.0) → Error honesty (12b.1) → CLI UX (12b.2) → Thread safety (12b.3) → API fixes (12b.4) → Test gaps (12b.5) → Docs (12b.6)

---

## The Brutal Truth

Five agents independently audited this project from a first-time user's perspective. Here's the honest summary:

**What's good:**
- The bio-inspired architecture is genuinely novel and interesting
- Error messages (when they fire) are helpful and actionable
- Test suite is large (3,400+ tests) with strong memory/simulation coverage
- Tool registry pattern is clean
- `__all__` and lazy loading in `__init__.py` are well-done

**What's broken:**
- `maxim.run()` — the flagship API call — throws a TypeError on invocation
- Hardcoded `data/` paths survive in templates and runtime despite Phase 0 cleanup
- 639 bare `except Exception:` clauses silently swallow errors users need to see
- `cv2` imported at module level in 7 files — `import maxim` fails without opencv
- 748 `print()` calls in library code instead of `logging`
- 21 core modules (attention, bridges, comms, decisions, etc.) have zero tests
- The original 6 API verbs (run, imagine, connect, diagnose, configure, introspect) have zero tests

**What's confusing:**
- `--sim` does 5 different things depending on the argument (goal string, YAML path, "agent", "research", "benchmark")
- 39 top-level packages — user has no idea where to look
- Bio-system abbreviations (NAc, SCN, ATL, EC) used without explanation in CLI flags and error messages
- `--language-model` has no discovery command — users can't list available models
- Deprecated legacy modes (`--sim agent`, `--sim research`) still exposed as primary options
- Four separate installation guides with no clear "start here"
- Python API docs don't exist — users can only learn the API by reading source

---

## 12b.0: Fix Breaking Bugs (~200 LOC)

Things that cause crashes or import failures for pip-installed users. Non-negotiable.

### 0a. Fix `maxim.run()` TypeError

**Problem:** `api.py:133` passes `router=router` to `LLMWorker.__init__()` which doesn't accept that kwarg. The flagship API call is broken.

**Fix:** Match the `LLMWorker` constructor signature. Likely needs the lane manager or router wired through differently.

**Test:** Add `test_api_core.py` with basic smoke test for `maxim.run()` (mocked LLM).

### 0b. Fix module-level `cv2` imports

**Problem:** 7 files import `cv2` at module level:
- `models/vision/rtm_engine.py`
- `models/vision/ultralytics_engine.py`
- `models/vision/segmentation.py`
- `conscience/workers.py`
- `conscience/media_loop.py`
- `conscience/connection.py`
- `data/camera/display.py`

A user doing `import maxim` who triggers any code path touching these modules gets `ModuleNotFoundError: No module named 'cv2'` unless they installed the `vision` extra.

**Fix:** Move all `import cv2` inside function/method bodies. Guard with try/except that raises a clear `ImportError("Install pymaxim[vision] for camera support")`.

### 0c. Fix surviving hardcoded paths

**Problem:** Despite Phase 0 cleanup, these still use repo-relative paths:
- `_data/templates/llm.json`: `"model_path": "data/models/LLM/mistral-7b-..."` — should use `~/.maxim/models/`
- `_data/templates/motor_cortex.json`: `"save_dir": "data/models/MotorCortex"` — same
- `models/language/router.py:284`: falls back to `data/util/cost_state.json`
- `hardware/config.py:22`: first search path is `data/util/robots.yaml`
- `mesh/agent_identity.py:29`: hardcoded `data/util/node_id.txt`

**Fix:** All paths must resolve through `utils/paths.py` — `data_home()` for user data, `bundled_data()` for read-only defaults. Templates should use `$MAXIM_HOME` placeholder or be resolved at runtime.

---

## 12b.1: Error Honesty (~400 LOC)

Users need to see errors, not silence. This is the single biggest quality-of-life improvement.

### 1a. Audit critical `except Exception:` blocks

**Problem:** 639 bare `except Exception:` clauses. Not all need fixing — some are legitimate resilience (tool execution, network probes). But many silently eat configuration errors, missing deps, and bad state.

**Strategy — triage, don't fix all 639:**
1. **API surface** (`api.py`): Every `except Exception:` in a public function must log the error and re-raise or return a structured error. Users calling `maxim.run()` must never get silent failure.
2. **Import paths** (any `except ImportError:`): Must give clear install instructions.
3. **Configuration loading** (config.py, paths.py, llm.json loading): Must tell the user what file is missing/broken and how to fix it.
4. **Leave alone:** Tool execution error handling, network probes, resilience decorators — these are correct as-is.

**Estimated:** ~60 of the 639 need fixing. The rest are legitimate.

### 1b. Replace critical `print()` with `logging`

**Problem:** 748 `print()` calls. Library code should never print.

**Strategy — triage again:**
1. **api.py**: All prints → `logging.getLogger("maxim")`
2. **cli.py**: CLI output prints are fine (they're the UI)
3. **Runtime/agents**: Replace with logging at appropriate levels
4. **Leave alone:** CLI output, simulation display, interactive mode prompts

**Estimated:** ~100 of the 748 need replacing. CLI output stays as `print()`.

### 1c. Upfront validation in API functions

**Problem:** `maxim.run()` doesn't validate that the model exists, API keys are set, or dependencies are installed before starting. Failure happens deep in the stack with cryptic traces.

**Fix:** Add validation at the top of each API verb:
```python
def run(model="mistral-7b", **kw):
    _validate_model_available(model)  # checks profile exists, deps installed, key set
    _validate_data_paths()            # checks ~/.maxim/ is writable
    ...
```

Raise `MaximConfigurationError` with actionable message:
```
MaximConfigurationError: Model 'claude-sonnet' requires the Anthropic SDK.
  Fix: pip install pymaxim[llm-anthropic]
  Then: export ANTHROPIC_API_KEY=<your-key>
```

---

## 12b.2: CLI UX Cleanup (~300 LOC)

### 2a. Add `--list-models` flag

**Problem:** Users can't discover available models. They have to read source code or guess.

**Fix:** Add `maxim --list-models` that prints:
```
Local models (requires local GPU):
  mistral-7b          Mistral 7B Instruct (default, requires ~5GB download)
  qwen2.5-14b         Qwen 2.5 14B (requires ~10GB, good orchestrator)
  smollm-1.7b         SmolLM 1.7B (small, fast, limited quality)
  ...

Cloud models (requires API key):
  claude-sonnet       Claude Sonnet (requires ANTHROPIC_API_KEY)
  gpt-4o              GPT-4o (requires OPENAI_API_KEY)
  ...

Currently configured: mistral-7b (from ~/.maxim/config/llm.json)
```

### 2b. Deprecation warnings for legacy `--sim` modes

**Problem:** `--sim agent`, `--sim research`, `--sim benchmark` are deprecated but shown as primary options.

**Fix:**
1. Keep them working (backward compat) but emit `DeprecationWarning` with the new syntax
2. Hide from `--help` using `argparse.SUPPRESS` or move to "Legacy Options" section
3. Add clear examples to help text:
```
Common usage:
  maxim --sim "test safety boundaries"        Run a generative simulation
  maxim --sim scenarios/my_test.yaml          Run a YAML scenario
  maxim --sim "test memory" --research        Run with research protocol
  maxim doctor                                Check your setup
```

### 2c. Better first-run experience

**Problem:** `maxim` with no args gives a parser error. No guidance.

**Fix:** When no action is specified, print a short getting-started message:
```
Maxim — bio-inspired cognitive architecture

Quick start:
  maxim doctor                              Check your environment
  maxim --sim "test the agent's memory"     Run a simulation
  maxim --list-models                       See available LLM models
  maxim --help                              Full option reference

Python API:
  import maxim
  maxim.diagnose()                          Check environment from Python
```

### 2d. Humanize debug subsystem names in CLI

**Problem:** `--debug hippo,nac` means nothing to a new user. `--clear-memory scn` is cryptic.

**Fix:** Accept both bio names and human names:
- `hippo` / `memory` → Hippocampus tracing
- `nac` / `reward` / `causal` → NAc tracing  
- `scn` / `temporal` / `clock` → SCN tracing
- `atl` / `semantic` / `concepts` → ATL tracing

Document the mapping in `--help`. Keep bio names as primary (they're load-bearing for the mental model) but accept aliases.

---

## 12b.3: Thread Safety & Global State (~200 LOC)

### 3a. Lock global mutable state

**Problem:** Module-level globals mutated without locks:
- `api.py`: `_event_subscriptions`, `_pending_tools`, `_next_handle_id`
- `runtime/lane_backends.py`: `_active_routers`
- `utils/prompts.py`: `_PROMPTS_DIR` cache

**Fix:** Wrap with `threading.Lock()` or use `threading.local()` where appropriate. The API module should be thread-safe for concurrent callers.

### 3b. Stop creating directories as side effect

**Problem:** `utils/prompts.py` creates directories in CWD when prompts path isn't found. Pip-installed users get surprise directories.

**Fix:** Only create directories under `~/.maxim/`. If bundled data lookup fails, raise `FileNotFoundError` with clear message, don't create CWD directories.

---

## 12b.4: Public API Completeness (~500 LOC)

### 4a. Wire remaining API stubs

**Problem:** `api.py` has two stub functions that return empty results:
- `research()` — returns `ResearchResult(goal=goal)` with no actual execution
- `campaign()` — returns `CampaignResult(campaign_name=...)` with no actual execution

**Fix:** Wire to existing internals (`research_orchestrator.py`, `dm_runtime.py`). These systems work from CLI — the API just needs the plumbing.

### 4b. Add `maxim.list_models()` API function

**Problem:** No programmatic way to discover models.

**Fix:**
```python
def list_models() -> dict[str, list[ModelInfo]]:
    """Return available models grouped by type (local, cloud)."""
```

### 4c. Structured errors for all API verbs

**Fix:** Define `MaximError` hierarchy:
```python
class MaximError(Exception): ...
class MaximConfigurationError(MaximError): ...  # bad config, missing deps
class MaximRuntimeError(MaximError): ...        # runtime failures
class MaximModelError(MaximError): ...          # model not found, download needed
```

Users can catch `maxim.MaximConfigurationError` instead of random `TypeError`/`KeyError` from deep in the stack.

---

## 12b.5: Test Gaps (~600 LOC)

### 5a. Core API smoke tests

**Problem:** The 6 original API verbs have zero tests. This is the most dangerous gap for PyPI.

**Add `tests/unit/test_api_core.py`:**
```python
def test_run_smoke(mock_llm):
    """maxim.run() completes without crash when LLM is mocked."""

def test_imagine_smoke(mock_llm):
    """maxim.imagine() returns SimulationResult."""

def test_diagnose_returns_report():
    """maxim.diagnose() returns DiagnosticReport with check results."""

def test_configure_sets_verbosity():
    """maxim.configure(verbosity=2) affects logging level."""

def test_observe_returns_subsystems():
    """maxim.observe() returns dict with expected keys."""

def test_connect_missing_robot():
    """maxim.connect('nonexistent') raises clear error."""
```

### 5b. Missing dependency error tests

**Add `tests/unit/test_optional_deps.py`:**
```python
def test_vision_import_without_cv2():
    """Importing vision modules without cv2 gives clear error."""

def test_anthropic_import_without_sdk():
    """Using claude model without anthropic SDK gives install instructions."""

def test_comms_import_without_twilio():
    """Using comms without twilio gives clear error."""
```

### 5c. Configuration error tests

```python
def test_run_without_model_gives_helpful_error():
    """maxim.run() with unavailable model explains what to do."""

def test_invalid_yaml_scenario_gives_helpful_error():
    """maxim.imagine(scenario='bad.yaml') explains the parse failure."""

def test_missing_data_dir_gives_helpful_error():
    """Operations with corrupted ~/.maxim/ explain recovery."""
```

### 5d. Add LLM mock fixture to conftest

**Problem:** No shared LLM mock fixture. Each test file rolls its own.

**Fix:** Add to `tests/conftest.py`:
```python
@pytest.fixture
def mock_llm():
    """Mock LLM router that returns canned responses."""
```

---

## 12b.6: Documentation (~300 LOC of markdown)

### 6a. Create `docs/user/api-quickstart.md`

**Problem:** No Python API docs exist. Users can only learn the API by reading source.

**Content:**
```markdown
# Python API Quick Start

## Installation
pip install pymaxim                        # core only
pip install pymaxim[llm-anthropic]         # + Claude support
pip install pymaxim[vision]                # + camera/vision

## Basic Usage
import maxim

# Check your environment
report = maxim.diagnose()
print(report.summary)

# Run a simulation
result = maxim.imagine(goal="test memory recall", model="claude-sonnet")

# Observe internals
state = maxim.observe("memory")
```

### 6b. Add bio-system glossary to README

**Problem:** README uses Hippocampus, ATL, NAc, SCN without linking to explanations.

**Fix:** Add a 10-line glossary table after the architecture section:

| Bio Name | Plain English | Module |
|----------|--------------|--------|
| Hippocampus | Episodic memory (events, experiences) | `memory/` |
| ATL | Semantic memory (concepts, categories) | `memory/` |
| NAc | Reward/causal learning ("what causes what") | `decisions/` |
| SCN | Internal clock (temporal patterns) | `time/` |
| EC | Memory indexing (entorhinal cortex) | `memory/` |
| Angular Gyrus | Cross-modal memory algebra | `math/` |

### 6c. Consolidate installation into single path

**Problem:** Four separate install guides.

**Fix:** README points to ONE canonical guide (`docs/user/getting-started.md`). Other guides become "advanced setup" references linked from there.

### 6d. Reconcile config path documentation

**Problem:** Some docs say `data/util/llm.json`, others say `~/.maxim/config/llm.json`.

**Fix:** Grep all docs for `data/util/` and `data/models/` — update to `~/.maxim/` paths. The `data/` convention is repo-development only.

---

## Non-Goals (explicitly out of scope)

These are real issues but NOT blockers for publication:

1. **39 top-level packages** — Architectural. Would break all imports. Do post-1.0 if ever.
2. **Class naming overloads (33 "Agent*" classes)** — Annoying but functional. Document, don't rename.
3. **`--sim` dual semantics** — Confusing but working. Add examples to help text (Phase 2b), don't redesign the CLI.
4. **21 untested modules** — Real gap but these are internal. Public API tests (Phase 5a) are the priority.
5. **All 639 `except Exception:` blocks** — Fix the ~60 in public-facing code (Phase 1a), leave internal resilience alone.
6. **All 748 `print()` calls** — Fix ~100 in library code (Phase 1b), leave CLI output alone.
7. **Circular import refactoring** — Works fine with lazy imports. Not user-facing.
8. **Full tool parameter documentation** — Nice-to-have. Ship API quickstart first.

---

## Dependency on Existing Plans

This is **Phase 12b** of the foundational buildout, running in parallel with Phase 11 (Test PyPI dry-run). Publication to PyPI is **manual** — no automated publish. This plan expands on Phase 9's "pre-pub hardening" bullet with concrete, audited findings.

- Phase 11 (Test PyPI) can proceed independently — it validates packaging mechanics
- This plan validates user experience and code quality
- Both must complete before manual `twine upload`

---

## Success Criteria

A new user can:
1. `pip install pymaxim` — no errors
2. `import maxim` — no side effects, no crashes
3. `maxim.diagnose()` — returns structured report
4. `maxim --list-models` — sees available options
5. `maxim --sim "hello world"` — gets a clear error about model setup OR runs successfully
6. `maxim.run(model="claude-sonnet")` — either works or gives `MaximConfigurationError` with exact fix steps
7. `maxim doctor` — tells them exactly what to fix
8. Read API quickstart docs and understand the 6 verbs in under 5 minutes
