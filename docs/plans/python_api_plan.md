# Python API Plan — Verb-Based Public Interface

> **Status:** Not started.
> **Scope:** ~400 LOC new facade + ~100 LOC rename refactor across ~14 files.
> **Depends on:** Nothing (existing internals are sufficient).
> **Blocks:** PyPI publication (Phase 2 of pypi_publication_plan.md).

---

## Problem

Maxim has well-structured internals, but no clean programmatic entry point. The only way to use it is through the 1,500-line CLI. A developer who installs pymaxim and writes `import maxim` gets `__version__` and nothing else.

The internal entry points exist (`run_agentic_loop()`, `start_simulation_mode()`, `AUTIntrospector`, `run_all_checks()`), but discovering them requires reading the CLI source.

## Solution

Six top-level verb functions that map directly to user intent. Each is a thin facade (~30-80 LOC) over existing internals. No refactoring of core systems — just a clean front door.

```python
import maxim

maxim.run()           # agentic cycle
maxim.imagine()       # simulation
maxim.connect()       # robot/peer connection
maxim.diagnose()      # doctor checks
maxim.observe()       # bio-subsystem introspection
maxim.configure()     # verbosity, logging, settings
```

---

## Phase 0: Rename AUTIntrospector to Observer (~100 LOC across ~14 files)

**Goal:** Align internal naming with the public API verb before building the facade.

**Why:** "AUTIntrospector" is simulation jargon ("Agent Under Test"). External users — and LLMs interacting with the codebase — will find `Observer` immediately intuitive. The public verb is `observe`, the class should be `Observer`.

### Rename map

| File | Change |
|------|--------|
| `simulation/introspection.py` | `class AUTIntrospector` -> `class Observer` |
| `simulation/introspection.py` | Module docstring update |
| `simulation/tools.py` | Import + type hints (~4 references) |
| `simulation/orchestrator.py` | Import + instantiation (~4 references) |
| `simulation/experiment.py` | Docstrings + type hints (~3 references) |
| `simulation/validation.py` | Docstring (~1 reference) |
| `tests/unit/test_introspection_and_tools.py` | Import + usage |
| `tests/unit/test_benchmark_phase0.py` | Import + usage |

**Backward compat:** Add a one-liner alias in `simulation/introspection.py`:
```python
AUTIntrospector = Observer  # Deprecated alias — remove in 0.2.0
```

This keeps existing internal code working during transition without blocking the rename.

**Test:** `python -m pytest tests/unit/test_introspection_and_tools.py tests/unit/test_benchmark_phase0.py -v`

---

## Phase 1: `maxim.configure()` (~30 LOC)

**Goal:** Single entry point for all runtime configuration.

**Why first:** Every other verb may need configuration (model, verbosity, log path). Build this before the others so they can call it.

```python
def configure(
    *,
    verbosity: int = 1,
    log_file: str | None = None,
    debug: str | None = None,
) -> None:
    """Configure Maxim runtime settings.

    Args:
        verbosity: 0=quiet, 1=normal, 2=verbose, 3=debug.
        log_file: Path to log file (None = stdout only).
        debug: Subsystem trace filter ("hippo", "nac", "hippo,nac", or None for all).
    """
```

**Wraps:**
- `utils.logging.configure_logging(verbosity, log_file)`
- `utils.structured_logging.configure_agentic_verbosity(verbosity, console_output)`
- Subsystem trace env vars (`MAXIM_HIPPO_TRACE`, etc.) parsed from `debug` string

**Files:** New `src/maxim/api.py`, updated `src/maxim/__init__.py`

---

## Phase 2: `maxim.run()` (~80 LOC)

**Goal:** Start the agentic cycle in one call.

```python
def run(
    model: str = "mistral-7b",
    *,
    goal: str | None = None,
    headless: bool = True,
    robot: str | None = None,
    home_dir: str = "~/.maxim",
    verbosity: int = 1,
) -> None:
    """Run Maxim's agentic cycle.

    Args:
        model: LLM profile name (e.g., "mistral-7b", "claude-sonnet").
        goal: Optional goal string. If provided, agent works toward it
              with a utility prompt. If None, enters interactive loop.
        headless: If True, run without robot hardware (default).
        robot: Robot type to connect (e.g., "reachy_mini"). Requires
               the robot's package to be installed.
        home_dir: Data/persistence directory.
        verbosity: Logging verbosity (0-3).
    """
```

**Wraps:** The bootstrap sequence from `cli.py` lines ~876-1239, distilled:
1. `configure()` with verbosity
2. `build_primary_router()` with model profile
3. Build tool registry, executor, decision engine, state, memory
4. If `robot`: `RobotRegistry().connect_robot()`
5. `run_agentic_loop()` with all objects

**Model gate:** If no LLM is available (no model file, no API key), raise `MaximConfigurationError` with a clear message explaining install options.

**Files:** `src/maxim/api.py`

---

## Phase 3: `maxim.imagine()` (~60 LOC)

**Goal:** Run a simulation in one call, return structured results.

```python
def imagine(
    goal: str = "general exploration",
    *,
    persona: str = "cooperative",
    scenario: str | None = None,
    model: str = "mistral-7b",
    sandbox: str = "tmpdir",
    verbosity: int = 1,
) -> "SimulationResult":
    """Run a Maxim simulation.

    Args:
        goal: What the simulation orchestrator should test.
        persona: Orchestrator persona ("adversarial", "cooperative",
                 "confused", "escalating", "researcher", etc.).
        scenario: Path to YAML scenario file. If provided, overrides
                  goal/persona with scenario-defined percepts.
        model: LLM profile for both AUT and orchestrator.
        sandbox: Sandbox type ("tmpdir" or "docker").
        verbosity: Logging verbosity (0-3).

    Returns:
        SimulationResult with metrics, action log, memory snapshots.
    """
```

**Wraps:** `start_simulation_mode()` from `simulation/orchestrator.py`.

**Files:** `src/maxim/api.py`

---

## Phase 4: `maxim.connect()` (~50 LOC)

**Goal:** Connect to a robot or peer node.

```python
def connect(
    robot_type: str,
    *,
    name: str | None = None,
    config: dict | None = None,
    timeout: float = 30.0,
    set_primary: bool = True,
) -> "RobotController":
    """Connect to a robot.

    Uses the plugin discovery system (entry_points group "maxim.robots")
    to find the controller class for the given robot_type.

    Args:
        robot_type: Registered robot type (e.g., "reachy_mini", "simulated").
        name: Instance name (defaults to robot_type).
        config: Robot-specific configuration dict.
        timeout: Connection timeout in seconds.
        set_primary: Whether to set this as the primary robot.

    Returns:
        Connected RobotController instance.

    Raises:
        MaximConfigurationError: If robot_type is not registered.
        MaximConnectionError: If connection fails.
    """
```

**Wraps:** `RobotRegistry().connect_robot()` with auto-discovery from Phase 3 of pypi_publication_plan.md (entry points). Falls back to manual registration if entry points aren't available yet.

**Files:** `src/maxim/api.py`

---

## Phase 5: `maxim.diagnose()` (~50 LOC)

**Goal:** Run diagnostics and return structured results.

```python
def diagnose(
    *,
    peer: str | None = None,
    api_key: str | None = None,
) -> "DiagnosticReport":
    """Run Maxim diagnostics.

    Without arguments, runs local doctor checks (platform, GPU, models,
    dependencies). With a peer URL, tests remote connectivity.

    Args:
        peer: Remote peer URL to test (e.g., "https://maxim.example.com/v1").
        api_key: API key for peer authentication.

    Returns:
        DiagnosticReport with check results, pass/fail status, and fix hints.
    """
```

**Return type:**
```python
@dataclass
class DiagnosticReport:
    platform: PlatformInfo
    checks: list[CheckResult]
    all_passed: bool
    failures: list[CheckResult]   # convenience filter

    def summary(self) -> str:
        """Human-readable summary."""
```

**Wraps:** `detect_platform()` + `run_all_checks()` from `doctor/`. Pulls the structured data out of the print-based CLI flow.

**Files:** `src/maxim/api.py`, possibly small refactor in `doctor/cli.py` to extract report-building from printing.

---

## Phase 6: `maxim.observe()` (~60 LOC)

**Goal:** Inspect bio-subsystem state.

```python
def observe(
    subsystem: str | None = None,
    *,
    keyword: str | None = None,
    limit: int = 10,
) -> dict:
    """Observe Maxim's cognitive subsystem state.

    Args:
        subsystem: Which subsystem to query. Options:
            - None: summary of all subsystems
            - "memory": Hippocampus episodic memories
            - "causal": NAc causal links and predictions
            - "concepts": ATL semantic concepts
            - "pain": Pain/harm detection history
            - "temporal": SCN temporal patterns
            - "energy": Token/compute/cost tracking
        keyword: Filter results by keyword (for memory/causal queries).
        limit: Max results to return.

    Returns:
        Dict with subsystem-specific data. Structure varies by subsystem.
    """
```

**Wraps:** `Observer` methods (renamed from `AUTIntrospector`):

| `subsystem` arg | `Observer` method |
|-----------------|-------------------|
| `None` | `.system_stats()` |
| `"memory"` | `.memory_recall(keyword=keyword, limit=limit)` |
| `"causal"` | `.causal_links(event_signature=keyword)` |
| `"concepts"` | `.concept_query(name=keyword)` |
| `"pain"` | `.pain_history(limit=limit)` |
| `"temporal"` | `.temporal_patterns()` |
| `"energy"` | `.energy_status()` |

**Alias support:** Register `introspect` as an alias for `observe` in `__init__.py`:
```python
introspect = observe  # Alias — both names work
```

This mirrors the `TOOL_ALIASES` pattern from `runtime/executor.py`. LLMs and developers who think "introspect" find it; those who think "observe" find it. No ambiguity, no cost.

**Requires:** A running agent session (the Observer needs subsystem references). If called without an active session, loads from the most recent persisted state in `home_dir`.

**Files:** `src/maxim/api.py`

---

## Phase 7: Wire into `__init__.py` + Lazy Imports (~30 LOC)

**Goal:** Make all verbs available at `import maxim`.

```python
# src/maxim/__init__.py
from maxim.api import configure, connect, diagnose, imagine, observe, run

introspect = observe  # Alias

__all__ = [
    "__version__",
    "configure",
    "connect",
    "diagnose",
    "imagine",
    "introspect",
    "observe",
    "run",
]
```

**Lazy loading:** Use `__getattr__` so importing `maxim` doesn't trigger heavy dependency chains. The actual imports from `api.py` happen on first call, not on `import maxim`.

```python
def __getattr__(name: str):
    if name in ("configure", "connect", "diagnose", "imagine", "observe", "run", "introspect"):
        from maxim import api
        func = getattr(api, name)
        globals()[name] = func  # Cache for subsequent calls
        return func
    raise AttributeError(f"module 'maxim' has no attribute {name}")
```

---

## Implementation Sequence

| # | Phase | LOC | Depends on |
|---|-------|-----|-----------|
| 0 | Rename AUTIntrospector -> Observer | ~100 | Nothing |
| 1 | `maxim.configure()` | ~30 | Nothing |
| 2 | `maxim.run()` | ~80 | Phase 1 |
| 3 | `maxim.imagine()` | ~60 | Phase 1 |
| 4 | `maxim.connect()` | ~50 | Phase 1 |
| 5 | `maxim.diagnose()` | ~50 | Phase 1 |
| 6 | `maxim.observe()` | ~60 | Phase 0 |
| 7 | Wire `__init__.py` | ~30 | Phases 1-6 |

**Phases 2-6 are independent of each other** and can be built in any order after Phase 1.

---

## Open Questions

1. **`maxim.run()` blocking vs async?** The agentic loop is blocking (runs until Ctrl+C or goal completion). Should we offer `maxim.run_async()` or is blocking fine for v1?
2. **Session state for `observe`.** If called outside a running agent, should it load from persisted state on disk? Or require an active session? Loading from disk is more useful for post-hoc analysis.
3. **Should `connect` auto-discover or require explicit type?** Entry point discovery (from pypi_publication_plan.md Phase 3) would let `maxim.connect()` with no args scan for available robots. But that's a bigger feature. For v1, require explicit `robot_type`.
4. **CLI parity.** Should the CLI be refactored to call `maxim.api.*` internally? This would guarantee the API and CLI always behave identically. But it's a larger refactor of `cli.py`. Candidate for v2.
