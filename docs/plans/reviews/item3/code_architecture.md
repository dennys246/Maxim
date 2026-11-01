# #1315 item 3 code review: the Architecture lens

Diff: `git diff origin/main...HEAD` on `feat/1315-item3-tool-policy` (e2f139af, 06af7bca). Read against
`docs/CODE_REVIEW.md`, `CLAUDE.md`, `ARCHITECTURE.md` / `utils/audit.py::LAYER_RULES`, the approach notes §1, §1A.2,
§1A.3, §1A.5 and §3 (with its "as built" block), and `docs/agents/runtime-tools.md`.

What I ran (worktree, `PYTHONPATH=$PWD/src`): `test_architecture_audit.py`, `test_tool_policy.py` and
`test_tool_policy_census.py` (82 passed); `lint_function_length.py` (clean); `lint_no_silent_swallows.py` (clean);
`lint_claude_md_invariants.py` (PASS); `lint_orphan_modules.py` (clean); `changelog_fragments.py check` (clean);
`ruff check` (clean); the characterization golden and both red-gate files (101 passed);
`tests/unit/test_stop_reasons.py` (**1 failed**, finding A1). I also ran a one-file probe of a
tool body that returns a gate kind (A4).

## What holds

- **Layering.** `ToolPolicy` is in `runtime/tool_policy.py`, the only layer allowed to see both `agents/`
  (`AgentPermissions`) and `tools/` (`Effect`). `PolicyHandle`, `current_policy()` and the context variable are in
  `tools/base.py`, so `simulation/tools.py::SpawnSubSimulationTool` reads the policy without importing `runtime`.
  `tools/base.py` gains only `maxim._offline`, a leaf. The architecture audit is green with no new baseline entry.
- **The seam is the one the notes name.** The gates run in §1A.2's order: forbid, mode and permissions in
  `_permission_denial`, then the `kind` permissions, the ceiling (`effective_ceiling`, the V7 AND applied once) and
  the pin in `_object_refusal`, all on the single `registry.get`. `permits()` runs both, so the roster and dispatch
  agree. The as-built block states the two-method split.
- **V3 and §1A.5.** `require_policy` checks `type() is`; `Executor` is keyword-only after the registry;
  `AgentConfig.policy` is `kw_only` with no default; `permissions=` and `tool_permissions_from_settings` are gone from
  `src/`, `scripts/` and the live docs. The only remaining `permissions=` hits are unrelated: `DockerSandbox`, and
  `dm_schema.CampaignDef`.
- **Rule T.** Every extraction I compared is a verbatim move: `_check_build_executor_args`, `_build_runtime_executor`,
  `_stash_timeout_retry`, `_observe_energy_cost`, `_parallel_entry`, `_aut_agent_config` and
  `_orchestrator_agent_config`. Every pinned function shrank (`build_executor` 250→223, `_start_agentic_runtime`
  829→800, `start_simulation_mode` 2414→2383, and others), and the lint is clean.
- **The invariant stub.** The new runtime-tools.md entry ends with a `Regression guard:` line that cites the red gates,
  the census and `require_policy`. The "THREE gates" invariant now reads FIVE, and the builder list reads six sites.
- **Test migration.** I compared every `-assert` against its `+` line. The asserted behaviour is kept, apart from the
  NIT in A9. Each red-gate flip only removes the `xfail` decorator and leaves the assertions unchanged. The one
  scenario change (`offline_in_tool_body` now engages the latch) is an owner decision recorded in the notes.

## Findings

### A1. BLOCKER (fail-closed: a red suite). `tests/unit/test_stop_reasons.py::TestToolErrorKind::test_member_count`

- **Evidence:** it fails on the branch with `assert 14 == 7`. §1A.2 says this test "is updated in the same commit",
  but it is not in the diff. `test_all_members_exist` also still lists only the seven old members.
- **Fix:** set the count to 14 and add the seven gate members, with their snake_case values, to `expected`.

### A2. SHOULD-FIX (fail-open in the census). `tests/unit/test_tool_policy_census.py::test_permissive_policy_callers_are_pinned`

- **What it pins:** the census pins where `ToolPolicy.unrestricted` and `deny` are called. But three of those callers
  are themselves permissive factories with no arguments: `runtime/tool_policy.py::cli_tool_policy`,
  `aut_tool_policy` and `orchestrator_tool_policy`. A fourth, `policy_from_tools_settings(None, None)`, is
  unrestricted under another name.
- **The gap:** their own callers (`cli._cli_agent_config`, `orchestrator._aut_agent_config` /
  `_orchestrator_agent_config`, `console/handle.py`) are not pinned. So a new `aut_tool_policy()` call anywhere in
  `src/` gets an allow-all policy and the census stays green. That is exactly what the census exists to catch: "every
  caller of `unrestricted(` or `deny(` is on a pinned list with its reason".
- **Unstated blind spots:** the matcher also misses an alias (`U = ToolPolicy.unrestricted`), `cls.unrestricted`,
  and an import under another name (`from ... import ToolPolicy as TP`). None of these is in the docstring's list of
  blind spots.
- **Fix:** the extracted helpers already exist. Inline `ToolPolicy.unrestricted(reason=...)` into
  `_cli_agent_config`, `_aut_agent_config` and `_orchestrator_agent_config`, then delete the three wrappers. That
  also takes sim and CLI knowledge out of `runtime/`. Failing that, have the census treat every function that
  returns one of the permissive factories as permissive itself, and pin its callers. Either way, add the alias and
  import-name cases to the stated blind spots.

### A3. SHOULD-FIX (fail-open on learning, behaviour wider than its rationale). `runtime/executor.py::Executor.execute` (`_suppress_nac ... or is_offline()`)

- **What it does:** under the latch, the tool-pain bridge books nothing for every tool, for the whole session. That
  includes real failures that have nothing to do with `OfflineRefusal`.
- **Why the rationale does not hold:** the stated rationale is that a started invocation "leaves a pending NAc event
  that later pain could link to". But `execute`'s `finally` already calls
  `ToolPainBridge.finish_invocation`, which pops the pending entry and calls `nac.discard_pending_event` on every
  exit (#851, #1207).
- **Why the scope note is wrong:** the as-built note justifies the breadth with "the atonia observer, the one
  offline consumer". That is not true. `MAXIM_OFFLINE=1` engages the latch at `import maxim`
  (`maxim/__init__.py`) for any CLI, sim, robot or `maxim.run` session, and V7 names the CLI's `llm.offline`.
- **Why it matters:** this is the CLAUDE.md "flag that toggles around broken behaviour" shape. The narrow root cause
  is the `record_tool_start` → `OfflineRefusal` ordering, and it can be fixed in the `except OfflineRefusal` branch
  by retiring or discarding the started event before returning.
- **Fix:** ask the owner, with the strict option first:
  - (a) Narrow the suppression to the `OfflineRefusal` branch, and prove that the red gate's "unchanged NAc state"
    still holds with the discard.
  - (b) Keep the session-wide suppression, but correct the note's "one offline consumer" claim and say in the
    CHANGELOG that offline sessions learn no tool-pain aversion at all.

### A4. SHOULD-FIX (fail-open on learning; the stated invariant is wider than its guard). `runtime/tool_dispatch.py::gate_refusal_of`, `tools/base.py::GATE_REFUSAL_KINDS`

- **The claim:** `tools/base.py` and the new runtime-tools.md invariant say "only `runtime/executor.py` produces a
  gate kind, so a tool body cannot exempt itself from NAc".
- **The guard's reach:** the guard, `test_item3_gate_kinds_constructed_only_in_executor`, scans `src/maxim` only.
- **What still gets through:** `gate_refusal_of` trusts any `result.error_kind`. Probe: a registered tool whose
  `execute` returns `ToolOutput(success=False, error_kind=ToolErrorKind.MODE)` comes out of `Executor.execute`
  still carrying `MODE` and no `refusal_layer`, and `gate_refusal_of` returns `MODE`. So `record_outcome` books no
  NAc, goal credit or cluster for it, and `execute_and_learn` skips the plan outcome.
- **Who is exposed:** any `@maxim.tool` / `register_tool` user tool, or a plugin, that reuses the enum's `MODE` or
  `OFFLINE` for its own failure silently stops teaching. The members are public on a stable surface.
- **Fix:** one of these, in the seam rather than in the guard:
  - in `_run_started`, re-tag a gate kind returned by a body that ran (to `EXTERNAL_FAILURE`, plus one WARNING per
    class);
  - or make `gate_refusal_of` require `metadata["refusal_layer"]`, which only the executor's `_refused_output` sets.

  Then narrow the invariant text to what is enforced.

### A5. SHOULD-FIX (fail-open on scope; owner decision). `api.py::run` (`allowed_dirs_override=[os.path.realpath(effective_home)]`)

- **What it does:** with no `home_dir`, `effective_home` is `utils/paths.py::data_home()`, the whole shared
  `~/.maxim`. The model's `read_file`, `write_file`, `edit_file`, `execute_file` and `bash` now reach every agent's
  memory store (`agents/*`), `config/` (the user overrides), `robots.yaml`, `sim_reports/`, `provenance/` and the
  orchestrator home.
- **Why that matters:** for most users the old reach was a project directory. Writing Maxim's own config and other
  agents' stores is the GOVERN class V2 created. #1342 asked for "workspace-only".
- **Classification:** the CHANGELOG files the confinement under `changed`. But it removes the CWD from a ✅ stable
  verb's reach in a patch, which users will meet as a break.
- **Where it comes from:** §1A.4 literally says `allowed_dirs == [home]`, so the code follows the notes. The finding
  is against the notes as well.
- **Fix:** confine to a dedicated workspace under the home: the `.maxim_workspace` that `build_tool_registry`
  already scaffolds, or `<home>/workspace`. Update the O1 red gate's criterion to match. Move the bullet to the
  `breaking` fragment with a migration line (pass `home_dir=` to give the agent a project). Ask the owner, since it
  amends §1A.4.

### A6. SHOULD-FIX (honesty of claims). Follow-ups said to be "filed" that I cannot find

- **What I searched:** `gh issue list --search` across all states, for each of the four below.
- **The four:**
  - §3: "`can_access_sem` has no caller, which is filed";
  - `tool_dispatch.execute_parallel_actions`: "the autonomy rejection above stays a booked negative (... filed for an
    owner line)";
  - V6: "Census the rest ... with a follow-up issue" (`bio_enrichment._dispatch_reflex_tool` and
    `OrchestratorActorTool.execute` are now pinned as direct calls);
  - the sub-sim child's registry ceiling, marked "Not built".
- **Result:** none has an issue number in the code, the notes or the tracker.
- **Fix:** file them and cite the numbers at each site. Otherwise the "filed" claims are false, and the census pins
  become permanent exemptions with no owner.

### A7. NIT (an unmet claim surface). The CC3 roster and `docs/agents/persistence-config.md`

The notes (§3) say "Its docstring says CC3 does not apply, and the CC3 list records that." The docstring does, but
`ToolPolicy` appears nowhere in `persistence-config.md`. **Fix:** add the one roster line, or drop the claim from the
notes.

### A8. NIT (fail-closed). `runtime/executor.py::Executor.policy` is a plain, rebindable attribute

The policy is sealed, but the executor's slot that holds it is not: `executor.policy = ToolPolicy.unrestricted(...)`
after construction skips `require_policy`. This is in-process code, out of the §0 threat model, but it is the one
place where "unforgeable" depends on convention. **Fix:** store it as `_policy` behind a read-only `policy` property.

### A9. NIT. Two test assertions moved from identity to equality

`tests/unit/test_agent_factory.py` and `tests/unit/test_build_executor.py` used `executor._permissions is perms` and
now use `executor.policy.permissions == perms`. Equality is correct, because the factory copies through
`deny`/`narrowed`. But nothing pins any more that the configured policy object itself is the one the executor holds.
`test_console_tool_allowlist.py` does pin it (`captured["config"].policy is handle.instance.executor.policy`).
**Fix:** add `instance.executor.policy is cfg.policy` to the `create_full_agent` threading test.

### A10. NIT. Small consistency points

- `ToolPolicy.narrowed()` keeps the parent's `reason`, so a sub-sim child's policy reads as its parent's in the logs.
  An optional `reason=` would fix it.
- `effective_ceiling` imports `is_offline` inside the function, while `executor.py` imports it at module level.
- `SpawnSubSimulationTool._run_sub_simulation(goal, policy: Any)` and `PolicyHandle.narrowed(only: Any, deny: Any)`
  could be typed (`PolicyHandle`, `Iterable[str]`).
- The census docstring's blind-spot list should also name a `Subscript` receiver (`tools[x].execute()`) and receivers
  whose name has no "tool" in it. Neither occurs in `src/` today; I grepped for both.
- The `changed` fragment does not mention that `Tool.run` now re-raises `OfflineRefusal` instead of returning a
  failed `ToolOutput`. That is a visible change to the stable `Tool` ABC for anyone who calls `run()` directly.

**Verdict: NOT READY. Fix A1 (a red test) first. A2 to A5 are design-level fail-open gaps, and A3 and A5 each need
an owner line; fold them before the PR opens. The layering, the seam, V3 and §1A.5, and Rule T are sound.**
