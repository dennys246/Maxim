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
Black-box tests must exercise the effect rather than only inspect the signature.

### Disposition

- `goal` and `robot`: **1.1 release gate**.
- complete `home_dir` ownership: **1.1 if surgical; otherwise 1.1.x with the
  partial behavior documented explicitly before 1.1 ships**.

## D16 — API cleanup starts after fallible side effects

`maxim.run()` mutates process environment and starts `LLMWorker` before entering
its main `try/finally`. Failures while constructing the agent, executor, evaluators,
or robot can therefore leave worker and environment state behind. Similar
pre-cleanup parsing/setup windows exist in `imagine()` and `campaign()`.

### Required contract

The cleanup boundary begins before the first reversible side effect. Environment
restoration, worker stop, robot disconnect, and bio-system shutdown each run from a
single structural cleanup path. Partial initialization must be safe.

### Disposition

**1.1 release gate** for `run()` and any identical stable-facade path; broader
deduplication can follow in 1.1.x.

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

Direct execution of `python -m maxim --audit-architecture` reported **32 violations**
and exited 1. Findings included runtime imports across the documented `agents`,
`tools`, `memory`, and `bridges` boundaries. Some may be typing-only or explicitly
accepted debt, but no reviewed baseline distinguishes them.

`tests/unit/test_architecture_audit.py::TestRealCodebase` asserts only that the audit
returns a list. CI does not run the CLI audit or reject new findings.

### Required contract

Classify the current 32 findings into fixed, accepted with rationale, and false
positive. Store an accepted-debt baseline and fail CI on any unreviewed addition.
Burning the baseline to zero is valuable but not required to cut 1.1.

### Disposition

**Baseline + regression gate in 1.1; debt burn-down in 1.1.x.**

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

**1.1 release gate.** A release cannot claim the required suite is green if the
documented command is environment-dependent.

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
