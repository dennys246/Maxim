# Repository review findings — 2026-08-19

**Status:** OPEN investigation cluster.  
**Scope:** public API contracts, lifecycle cleanup, architecture enforcement, and
offline test behavior observed during the 1.1 release-readiness review.  
**Method:** source tracing plus direct lint, type, architecture, unit, integration,
and advertised-fast-suite execution. No hardware or paid-model experiments were run.

This document preserves the detailed evidence behind D15–D20 in the
[known-defects ledger](README.md). The score and prioritization live separately in
[the repository scorecards](../limits/score_cards/).

## D15 — stable `maxim.run()` arguments do not fulfill their contracts

### Confirmed behavior

- `goal` is accepted and documented at `src/maxim/api.py::run`, but is not read
  after argument validation and is not passed to `run_agentic_loop`.
- `robot` is connected when requested, but the tool registry was already built
  with `maxim=None`; the connected controller is not attached to the agent,
  executor, registry, or loop invocation.
- `home_dir` controls several API-created persistence paths, but the main loop
  separately writes runtime state beneath CWD-relative `data/agents/...`.

### Required contract

Every stable public argument must produce an observable effect, be rejected as an
invalid combination, or be removed through the documented compatibility process.
Facade tests must exercise the public-to-runtime seam; lower-level controller and
registry tests must pin physical lifecycle and ownership behavior.

### Disposition

- `goal` and `robot`: **FIXED for the 1.1 release gate in v1.0.9**. `goal` uses
  the canonical CLI mailbox; `robot` requires `headless=False`, atomically
  acquires and wakes the selected controller, reaches direct controller motion,
  and attempts to restore prior awake/connection state on exit. A failed sleep
  or disconnect raises `HardwareError` and retains the live registration for
  operator recovery. Controller-bound motion cannot retarget another global
  robot. Facade, registry-lifecycle, and controller guards live in
  `test_api_core.py`, `test_robot_registry.py`, and `test_move_tool_gaze.py`.
- Lifecycle scope is explicit: completing the initial goal does not currently
  stop the service loop, and `goal=None` starts idle because this facade installs
  no terminal-input reader. Those are documented semantics, not implicit
  interactive behavior.
- complete `home_dir` ownership: **OPEN — 1.1.x**, with partial behavior
  documented before the 1.1 cut.

## D16 — API cleanup starts after fallible side effects

`maxim.run()` mutates process environment and starts `LLMWorker` before entering
its main `try/finally`. Failures while constructing the agent, executor, evaluators,
or robot can therefore leave worker and environment state behind. Similar
pre-cleanup parsing/setup windows exist in `imagine()` and `campaign()`.

### Required contract

The cleanup boundary begins before run-owned LLM environment overrides and
runtime-resource acquisition. Restoration of those two overrides, worker stop,
robot lifecycle, and returned bio-system shutdown each run from one structural
cleanup path. Cleanup stages continue after an earlier cleanup failure.

### Disposition

**FIXED for the 1.1 release gate in v1.0.9** for `run()`: the structural cleanup
boundary covers its LLM overrides, worker startup, returned agent/bio instance,
atomic robot lease, and loop execution. Factory executor failure shuts down the
partially built bio instance, and the superseded skeleton MemoryHub worker is
stopped before its replacement is installed; robot connect/wake failure is
transactionally unwound; each later cleanup stage still runs if an earlier one fails. Concurrent
`run()` calls fail loudly rather than racing process-global model state.
Equivalent cleanup for `imagine()`/`campaign()` and broader process-global
configuration isolation remain **OPEN — 1.1.x**.

## D17 — `maxim.load.agent()` does not immediately restore everything promised

The public docstring promises Hippocampus, NAc, and ATL restoration. Factory
construction auto-loads Hippocampus and NAc but constructs ATL without loading it.
ATL is loaded later by `MemoryHub.on_session_start()`, which `load.agent()` does not
call before returning.

Corrupt Hippocampus and SCN loads are also broadly caught and replaced with fresh
state. For an API explicitly named `load`, silent substitution risks overwriting
recoverable data later.

### Required contract

`load.agent()` either returns a fully restored object or fails with structured,
actionable recovery information. A fresh replacement must require explicit caller
choice.

### Disposition

**1.1 release gate** because this is persistence correctness on the stable API.

## D18 — `register_tool()` registration is one-shot

`_inject_pending_tools()` clears the global pending list after creating a registry.
Each subsequent API call constructs a fresh registry, so a registered tool is
available only to the next injection. The public contract says otherwise in two
places: the `register_tool` docstring promises a tool "available to all agents"
(`src/maxim/api.py::register_tool`), and `docs/user/extension_api.md` declares
`register_tool` **stable, part of the 1.0 contract** (§2 stability note).

### Required contract

Choose and test one behavior:

1. persistent process-wide registration, with explicit unregister/reset support; or
2. one-shot registration, renamed and documented as such.

### Disposition

**1.1 release gate** because the current behavior silently contradicts the public
extension API.

## D19 — the architecture audit cannot enforce architectural change

Direct execution of `python -m maxim --audit-architecture` reported **33 violations**
and exited 1. Findings included runtime imports across the documented `agents`,
`tools`, `memory`, and `bridges` boundaries. Some may be typing-only or explicitly
accepted debt, but no reviewed baseline distinguishes them.

`tests/unit/test_architecture_audit.py::TestRealCodebase` asserts only that the audit
returns a list. CI does not run the CLI audit or reject new findings.

### Required contract

Classify the current 33 findings into fixed, accepted with rationale, and false
positive. Store an accepted-debt baseline and fail CI on any unreviewed addition.
Burning the baseline to zero is valuable but not required to cut 1.1.

### Disposition

**Baseline + regression gate in 1.1; debt burn-down in 1.1.x.**

**Resolved 2026-08-24 (gate half):** the 33 findings were classified by import scope —
10 typing-only (`if TYPE_CHECKING:`), 16 function-local lazy imports (none breaks an import cycle — verified by a transitive import-graph check), 7 module-level —
and recorded with per-entry rationale in `src/maxim/utils/architecture_baseline.json`.
`tests/unit/test_architecture_audit.py::TestRealCodebase` fails on additions, stale
entries, and unreviewed entries; `maxim --audit-architecture` reports against the same
baseline. Zero-debt burn-down remains the 1.1.x list's item 11 in the roadmap.

## D20 — the advertised offline fast suite is not hermetic

Direct execution found multiple environment leaks:

- `tests/unit/test_clip_encoder.py` loads a remote model whenever
  `sentence_transformers` is importable, rather than requiring a cached-model or
  network marker;
- `tests/substrate/test_p4_fixture_validation.py` attempts a live Hugging Face
  lookup for `paraphrase-mpnet-base-v2` when `sentence_transformers` is installed
  but the model is not cached (reproduced again during D22 verification);
- `tests/behavioral/test_cradle_mother_pipeline.py` launched a subprocess harness
  whose workdir defaulted under `~/.maxim`; the D22 verification pass now passes
  its existing `--workdir` option under `tmp_path`, closing that one leak;
- cost tracking can write `~/.maxim/util/cost_state.json` during teardown.

CI currently runs only `tests/unit/`, while the required project check is the wider
`tests/ -m "not slow"` suite.

### Required contract

The default fast suite must run without network, hardware, installed model caches,
or writes outside its temporary test root. Tests needing those resources require an
explicit marker and opt-in CI job.

### Disposition

**FIXED for 1.1 in v1.0.9 (2026-08-19).** The default pytest process now uses a
unique temporary HOME/config/cache root inherited by subprocesses, resets Maxim
path caches between tests, and keeps Hugging Face/Transformers offline. Tests
requiring pretrained model or dataset assets carry `requires_model_cache` and
need explicit `MAXIM_RUN_MODEL_TESTS=1` opt-in. CI runs the same wider fast-suite
command as the contributor guide. The exact command completed locally with
9,303 passed, 44 explicit resource/platform skips, and 41 slow deselections in
4m57s.

## Additional hardening findings

These are important but do not need standalone defect IDs yet:

- the 30 Hz loop performs synchronous atomic persistence on the control thread;
- normal-tail session cleanup is not protected by a whole-loop `finally` for all
  callers (UNVERIFIED in the 2026-08-19 claims-check round — kept as a hardening
  lead, not an established finding; verify against `run_agentic_loop`'s exit paths
  before acting on it);
- persistence failures in `runtime/loop_state.py::_persist_state_json` are silently
  swallowed;
- Python support, dependency, API-count, architecture, and decision docs disagree
  with current code;
- the local workspace has a large stale-branch/worktree footprint, increasing
  provenance risk but requiring deliberate cleanup to preserve WIP.

These are assigned in the [1.1→1.3 roadmap](../plans/roadmap_1_1_to_1_3.md) and
[scorecards](../limits/score_cards/) rather than expanded into speculative bugs.
