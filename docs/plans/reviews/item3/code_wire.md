# #1315 item 3 — Wire-integrity lens (pre-merge code review)

Diff: `git diff origin/main...HEAD` on `feat/1315-item3-tool-policy` (e2f139af + 06af7bca). Charter:
`docs/CODE_REVIEW.md` (wire integrity). Scope searched: `src/`, `scripts/`, `tests/`, `docs/` (user, agents, wiring,
bugs ledger, experiment rationale scripts), `changelog.d/`, `README.md`. Narrow test runs only (no full suite).

## Producer → consumer table

| Contract | Producers (before → after) | Consumers outside the hunks (before → after) | What each gets | Probe |
|---|---|---|---|---|
| `Executor.__init__` (kw-only required `policy`, `permissions=` gone, positional after registry gone) | `bootstrap.build_executor` (only src site); tests | src: none outside bootstrap. tests: ~40 sites migrated; **`tests/unit/test_agent_permissions.py` 8 sites NOT migrated** (F1). docs: `docs/experiments/rationale/e1-conflict/e1_argmax_replay.py::_executor_at` NOT migrated (F6) | F1/F6: `TypeError` | 2, 3 |
| `build_executor(policy=)` | 9 src sites (api, agentic_runtime ×2, foundry, minecraft_harness, sub-sim, agent_factory, pain_bus docstring, pain_interceptor docstring) | all src/scripts sites pass `policy=` (AST scan). `test_refusal_booking_red_gate._build_executor` keeps a `permissions=` branch behind `find_spec("maxim.runtime.tool_policy")` — dead once the module exists, harmless | real policy | 2, 3 |
| `AgentConfig.policy` (required kw-only) | cli, api, console handle, orchestrator AUT + narrator, create, load, create_npc_agent | `create_full_agent` → `build_executor(policy=config.policy)`; `require_policy` refuses `None`. The adopted persistent agent (`orchestrator._adopt_persistent_agent`) carries the console's policy, built earlier. No `asdict`/pickle of `AgentConfig` in src (ToolPolicy `__reduce__` raises) | real policy or `TypeError` | 2, 8 |
| `record_outcome(refused=)` required | `tool_dispatch`: `execute_and_learn` ×2, `execute_parallel_actions`, `book_refusal`, `book_machine_refusal`, `book_hard_rejection`, `book_agent_fallback_failure` | `loop_setup._rec_outcome` partial (passes through); scripts `exp56/common.py`, `orient_substrate/2,9`, `survival_world/dark_danger_probe, r2_learned_bias` migrated; tests migrated (AST scan: every `tool_dispatch.record_outcome` call passes `refused=` or `**kw`). **`docs/experiments/rationale/e1-conflict/learner_sim.py` and `e1_argmax_replay.py` NOT migrated** (F6) | F6: `TypeError` | 3 |
| `ToolOutput.error_kind` gate kinds + `GATE_REFUSAL_KINDS` | only `runtime/executor.py` | `tool_dispatch.gate_refusal_of` (new); `proprioception/pain.py` mapping (no longer reached by gate kinds); `loop_state.py` ReplanContext `failure_type=str(error_kind)` (F9); `memory/types.py` legacy repr parser (legacy only); sinks / `actions.jsonl` / console / pulse: **do not read it** (the §1A.2 hand-off to Session S, still open) | gate kind reaches learning only via `record_outcome`/`_record_plan_outcome` | 1, 4 |
| `metadata["refusal_layer"]` / `["effect"]` | executor `_refused_output` | no reader in src/scripts (tests only) | record only; `effect` is `list[str]` or the string `"unknown"` (F8) | 1, 7 |
| Gate-refusal "books no pain/NAc" claim | executor (inner) | **outer wrappers run BEFORE the inner gate:** `AnticipatoryPainExecutor` (assessor publishes anticipated pain), `FearGatedExecutor` (fear review + its own blocks book NAc negative) (F3) | refused actions can still book pain / NAc | 4, 9 |
| `Executor.check(tool)` | executor | `loop_perception._checked_call` + interoception `executor.check(_sense)`. Loop executor is `InstrumentedExecutor`/`FearGatedExecutor`/`Anticipatory`/`PainInterceptor` — all proxy via `__getattr__`. `ConcurrentExecutor` has no `__getattr__` (no `check`/`permits`/`policy`), not used in prod (NIT, F10). `check()` omits the inactive-scene gate dispatch applies (F7) | typed refusal or None | 2, 9 |
| `Executor.permits` (now includes ceiling + pin) | executor | `agent_loop` roster (`executor.permits(t)`), `_runnable_tools` | consistent with dispatch | — |
| `tools/base.py::current_policy` / `dispatching` | `Executor._run_started` only | `SpawnSubSimulationTool.execute` (synchronous read, refuses on None). `ConcurrentExecutor` submits `Executor.execute` → set inside the pool thread | dispatching policy, narrowed | 8, 10 |
| `agents/permissions.py::tool_permissions_from_settings` removed | — | src/scripts: zero readers. `docs/bugs/README.md` D73 row: location column still names the deleted symbol (F11) | — | 3 |
| `Executor._permissions` removed | — | zero readers outside test docstrings | — | 3 |
| `execute_parallel_actions` new audit entries | `_audit_parallel_action` | `AutonomyController._get_consecutive_failures`, `_get_recent_action_rate`, `get_audit_log` (no external reader) — **the `consecutive_failures` violation is severity `error`, and `approval_blocker` acts only on `critical`** (F4). No double logging: the loop's batch branch writes no entry of its own | entry counted, never acted on | 1, 6 |
| `Tool.run` re-raises `OfflineRefusal` | `tools/base.py` | callers of `tool.run`: only `Executor._run_started` (maps to OFFLINE). Direct `.execute()` callers (bio_enrichment reflexes, `OrchestratorActorTool`, auto_sense) never went through `run`; an `OfflineRefusal` there propagates to their own `except` (logged / reflex `failed`). `OfflineRefusal(RuntimeError)` | fail-closed | 3 |
| `maxim.run` `allowed_dirs_override=[home]` | `api.run` | `build_tool_registry` → all 7 filesystem tools incl. `bash`, `write_file`; workspace scaffold now under home. **Default `home` is `data_home()` = `~/.maxim`** (F2). README / `api-quickstart.md` / `python-api.md` still describe the old reach (F5) | wider on the data home, narrower on CWD | 6 |
| Census `tests/unit/test_tool_policy_census.py` | — | scans `src/maxim` only, receivers named `*tool*` | misses tests/ (F1's cause) and `loop_perception`'s `_sense.execute` (guarded by `check`, but unpinned) (F12) | 2 |

## Findings

### F1 — BLOCKER (fail-closed: loud) — `tests/unit/test_agent_permissions.py::TestExecutorPermissionGate` / `::TestKindSelector`
Eight `Executor(...)` constructions were left on the removed contract (`Executor(reg)` with no policy, and
`Executor(reg, permissions=perms)` ×7, lines ~130–335). On this branch:
`python -m pytest tests/unit/test_agent_permissions.py` → **8 failed, 24 passed** (`TypeError: ... missing 1 required
keyword-only argument: 'policy'` / `unexpected keyword argument 'permissions'`). The file is in the diff (only its
`TestToolPermissionsFromSettings` class was migrated). The census did not catch it because it scans `src/maxim` only.
`TestKindSelector` is also a cited guard in the D73 ledger row, so a guard of a FIXED row is red.
Fix: migrate the eight sites to `policy=ToolPolicy.deny(...)` / `ToolPolicy.only(..., ceiling=Effect.ALL, ...)` /
`TEST_POLICY` (the `kind:` selector cases need `only([...])`/`deny([...])` with the selector string). Consider
extending the census's first test (policy= at every `Executor(`/`build_executor(`/`AgentConfig(`) to `tests/` and
`scripts/` too; it is cheap and would have caught this.

### F2 — SHOULD-FIX (fail-open: security boundary) — `api.py::run` (`allowed_dirs_override=[os.path.realpath(effective_home)]`)
`effective_home = _api_home(home_dir)`, which for the default call (`maxim.run()` with no `home_dir`) is
`utils/paths.py::data_home()` = `~/.maxim`. All seven filesystem tools (`read_file`, `write_file`, `edit_file`,
`execute_file`, `glob`, `bash`, `search_code`) are now confined to — i.e. granted — the whole shared data home:
`config/` (user config), every other agent's stores (`agents/*`, `memory/`, `orchestrator/`, `sessions/`), the api
agent's own NAc/EC/hippocampus files, `sim_reports/`, `provenance/`, `console/`, `util/` (active-model files,
cost state). Before, the reach was CWD + `CWD/.maxim_workspace` (passive mode). The owner criterion in §1A.4 is
literally `allowed_dirs == [home]`, so this matches the letter; the wire consequence of "home" defaulting to the
shared data home is probably not what #1342's "workspace-only" meant. Fix: confine to a workspace UNDER the home,
e.g. `[get_workspace_path(effective_home)]` (the `.maxim_workspace` scaffold `build_tool_registry` already creates
there) or `<home>/workspace`, and pin it in `test_api_core`; or take this back to the owner as a decision.

### F3 — SHOULD-FIX (fail-open: claim vs production stack) — `runtime/pain_interceptor.py::AnticipatoryPainExecutor.execute`, `runtime/fear_gate.py::FearGatedExecutor.execute`
`docs/agents/runtime-tools.md`'s new invariant and the changelog say a gate refusal books "never NAc, pain, goal
credit, cluster reward or a plan outcome". In the shipped sim AUT stack (`orchestrator.start_simulation_mode`:
inner Executor → `PainInterceptorExecutor` → `AnticipatoryPainExecutor` → `FearGatedExecutor`) and the CLI/factory
fear wrap, the outer layers act BEFORE the inner policy/mode/ceiling gate: the `PerceivedPainAssessor.assess`
publishes anticipated pain for an action the executor then refuses, and `FearAgent.review_action` reviews (and may
block with `fear_agent_blocked`, no `error_kind`, so `record_outcome` books NAc negative) a call the policy would have
refused. Pre-existing for mode/permission denials, but the diff's claim is now broader than the code. Fix: have the
outer wrappers consult `self._inner.check(...)`-equivalent (a name-level `refusal_for(action)` on the executor) and
pass a gate refusal straight through, or narrow the invariant text to "the executor books no pain …" and file the
wrapper ordering as a follow-up.

### F4 — SHOULD-FIX (fail-open: a counter that acts on nothing) — `agents/autonomy.py::SafetyConstraints.check_constraints` / `AutonomyController.approval_blocker`; claim in `tool_dispatch.py::_audit_parallel_action`, approach notes §1A.2, `test_refusal_booking_red_gate.py` docstring
The new per-action audit entries in the parallel batch are justified as feeding "`max_consecutive_failures` … the
counter that stops a model that keeps naming a refused tool". `_get_consecutive_failures` does count them, but the
resulting `Violation` has `severity="error"` and `approval_blocker` returns only `critical` violations; no other
reader of `check_constraints` exists in `src/`. So the counter stops nothing (the store is written → keyed → read →
**never acts**). The entries are harmless records; the claim is false. Fix: correct the claim in the notes, the red
gate docstring and the helper docstring, and file an issue for whether `consecutive_failures` should block (that is
an owner decision, not this PR's).

### F5 — SHOULD-FIX (claim surface users copy) — `README.md`, `docs/user/api-quickstart.md`, `docs/user/python-api.md`; `changelog.d/1315-item3-tool-policy.breaking.md`
README and the quickstart show `maxim.run(goal="inspect the workspace")` / `"summarize the workspace"`; after F2's
change the agent inspects `~/.maxim`, not the user's project. `python-api.md` §run still says `home_dir` roots
"memory, data, and session files" only. The breaking fragment says "the stable verbs (`maxim.run`, …) build their own
policy and are unchanged", while the changed fragment (correctly) lists `maxim.run`'s filesystem change. Fix: one
line each in README/quickstart/python-api naming the new reach and `home_dir`; drop "and are unchanged" for
`maxim.run` in the breaking fragment (or point to the changed bullet).

### F6 — SHOULD-FIX (fail-closed: loud) — `docs/experiments/rationale/e1-conflict/e1_argmax_replay.py::_executor_at` / `::_book`, `docs/experiments/rationale/e1-conflict/learner_sim.py` (`outcome`)
Both are committed reproduction scripts cited by `docs/experiments/e1_conflict_prereg.md`. `Executor(tool_registry=
ToolRegistry())` and `record_outcome(...)` without `refused=` are now `TypeError`, so the prereg's evidence can no
longer be re-run from `main`. Fix: `policy=ToolPolicy.unrestricted(reason="e1 replay")` and `refused=None`
(behaviour-identical: neither path produces a gate kind).

### F7 — NIT (fail-open, pre-existing behaviour) — `runtime/executor.py::Executor.check`
Documented as "every dispatch gate on *tool* without dispatching it", but it skips the registry gates dispatch applies
(`INACTIVE_SCENE`, `NOT_REGISTERED`): an `auto_fire` tool scoped to an inactive scene passes `check()` and runs in
`auto_sense`, though dispatch would refuse it. Same as before the diff; the docstring overclaims. Fix: add the
`get_tool_scene`/`is_tool_active` test to `check()` or narrow the docstring.

### F8 — NIT — `runtime/executor.py::_refused_output` (`metadata["effect"]`)
`effect` is `list[str]` for a resolved tool and the string `"unknown"` otherwise; a future consumer (the Session S
`actions.jsonl` hand-off) would have to branch on type. `INACTIVE_SCENE` also reports `"unknown"` though the tool
object exists in the registry. Fix: always a list (`[]` or `["UNKNOWN"]`), or document the union.

### F9 — NIT — `runtime/loop_state.py` (ReplanContext `failure_type=str(getattr(result, "error_kind", "unknown"))`)
Gate kinds now reach the replan prompt as `"ToolErrorKind.POLICY_CEILING"` (`str()` of the enum) and `"None"` for an
untyped failure, contrary to §1A.2's "serialised by `.value` everywhere". Pre-existing format, newly carrying gate
kinds into LLM-facing text. Fix: `.value` when it is a `ToolErrorKind`.

### F10 — NIT — `runtime/concurrent_executor.py::ConcurrentExecutor`
The one executor wrapper without `__getattr__`: no `policy`, `check`, `permits`. No production construction today
(exported only), so nothing breaks; if wired as a loop executor, `auto_sense` would `AttributeError` into its swallow
(fail-closed) and the roster's `executor.permits` would raise. Fix: add the delegating `__getattr__` like the others.

### F11 — NIT — `docs/bugs/README.md` D73 row (location column)
Still lists `src/maxim/agents/permissions.py::tool_permissions_from_settings`, deleted by this diff (the Fix column was
amended, the location column was not). Fix: replace with `runtime/tool_policy.py::policy_from_tools_settings`.

### F12 — NIT — `tests/unit/test_tool_policy_census.py::test_direct_tool_calls_outside_the_executor_are_pinned`
The receiver heuristic (`*tool*` names or `get`/`find` results) misses `runtime/loop_perception.py::auto_sense`'s
`_sense.execute(entity_name=...)` direct call. It is guarded (`elif` after `executor.check(_sense)`), but it is the kind
of site the census exists to pin. Separately, approach notes §1A.5 says `simulation/dm_schema.py::CampaignDef.permissions`
"is censused as dormant", and it is in no census table (its field comment still says "enforced"). Fix: pin the
`_sense` site (rename the receiver or extend the heuristic) and either add the dormant field to the census or drop the
claim.

## Checked and clean
- Every `Executor(` / `build_executor(` / `AgentConfig(` in `src/` and `scripts/` passes `policy=` (AST scan); every
  `tool_dispatch.record_outcome` / `rec_outcome` call in src/scripts/tests passes `refused=` or a `**kw` splat.
- All executor wrappers used in production (`FearGated`, `PainInterceptor`, `AnticipatoryPain`, `Instrumented`) proxy
  `policy` / `check` / `permits` through `__getattr__`; `PainInterceptorExecutor` fires only on success, so gate
  refusals never reach it.
- No double audit entry for parallel batches (the loop's batch branch logs none).
- `Tool.run`'s re-raise has one caller (`Executor._run_started`); no tool body in src produces a gate kind.
- Narrow runs: 323 + 308 tests across the touched and neighbouring files pass; only `test_agent_permissions.py` fails.

**Verdict: NOT READY — one BLOCKER (F1, eight unmigrated test sites, red on the branch) plus F2 (default `maxim.run`
filesystem reach is the whole `~/.maxim`) needs a fold or an owner call before merge; F3–F6 are claim/repro folds.**
