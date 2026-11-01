# #1315 item 3: delta round (the folds of the three-lens round)

Scope: `git diff --cached` in `.worktrees/v-toolpolicy` (the staged folds on top of 06af7bca). Charter:
`docs/CODE_REVIEW.md` (delta rounds; executor + architecture + wire on the folded `src/` and `scripts/`). The four
owner decisions of 2026-10-31 (notes §3 "as built") are taken as given and checked only for implementation.
Probes: `/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/248fdc0d-2f2e-471a-b37d-dacb96a7ecce/scratchpad/review3_delta/`
(`p1_reinit.py`, `p2_workspace.py`, `p3_deletion.py`, `p4_census_alias.py`, `conftest_probe.py`).

What I ran (worktree, `PYTHONPATH=$PWD/src`, narrow only):

- every staged test file plus `test_fear_gate`, `test_tool_dispatch`, `test_loop_perception_characterization`,
  `test_execute_and_learn_characterization`: 410 passed;
- every test file naming `FearGatedExecutor`, `AnticipatoryPainExecutor`, `_orchestrator_source`, `maxim.run(` or
  `lint_loop_mode_reads`: 668 passed;
- `test_architecture_audit.py`: 55 passed;
- `ruff check` / `ruff format --check` (clean), `lint_function_length`, `lint_no_silent_swallows`,
  `lint_loop_mode_reads`, `lint_orphan_modules`, `lint_claude_md_invariants`, `changelog_fragments.py check`: all clean;
- `scripts/run_mypy_public.sh`: **1 error** (D1);
- both E1 rationale scripts (`e1_argmax_replay.py`, `learner_sim.py`) run to exit 0;
- deletion probes (monkeypatched, no file edits): removing `_retag_body_gate_kind`, either wrapper's `gates_refuse`,
  or `ToolPainBridge.finish_invocation` turns its guard red (the last: 4 offline cases fail on
  `_pending_events (a started invocation left pending)`).
- I checked #1549–#1552: all four exist, are open, and match the text that cites them.

## Findings

### D1. BLOCKER (fail-closed: CI red). The mypy step fails on `runtime/executor.py::Executor._run_started`

```
src/maxim/runtime/executor.py:608: error: Incompatible types in assignment (expression has type "_Refusal | None",
variable has type "_Refusal")  [assignment]
```

The F8 fold put `refusal = _Refusal(ToolErrorKind.INACTIVE_SCENE, ...)` in the inactive-scene branch. mypy therefore
types `refusal` as `_Refusal`, and the later `refusal = self._object_refusal(tool_name, tool)` (Optional) no longer
type-checks. `executor.py` is on `run_mypy_public.sh`'s list, so CI's mypy step and `prepr.sh` both fail.
**Fix:** inline it (`_refused_output(_Refusal(...), self.registry.find(tool_name))`), or rename the variable in one
branch. This adds no new behaviour.

### D2. SHOULD-FIX (fail-open: the seal regressed). `runtime/tool_policy.py::ToolPolicy.__init__` no longer checks the key

The fold deleted `if _key is not _KEY: raise TypeError(...)  # pragma: no cover - __new__ already refused it`. That
comment was wrong. `__new__` runs only on construction, so re-running `__init__` on an existing policy skips it, and
`__init__` writes through `object.__setattr__`, past the sealed `__setattr__`. Probe `p1_reinit.py`:

```
before: ToolPolicy(ceiling=<Effect.READ: 1>, reason='locked') frozenset()
after re-init: ToolPolicy(ceiling=<Effect.READ|...|GOVERN: 127>, reason='widened') None
```

This widens a locked policy in place, which also widens every executor that holds the same object. It needs no key
and no introspection: `policy.__init__(_key=None, permissions=AgentPermissions(), ceiling=Effect.ALL, pins=None,
reason="x")`. On HEAD the same call raises `TypeError`.

The module docstring still claims "``__new__`` and ``__init__`` check a module-private key". The runtime-tools.md
invariant and §3 also claim "cannot be mutated". Both now overstate the code. No review finding asked for the
removal; it looks like a pragma-count cleanup.

**Fix:** restore the check without the pragma. Add a test that calls `p.__init__(_key=None, ...)`, expects a
`TypeError` and expects the policy to be unchanged. That test also covers the line, so no pragma is needed.

### D3. SHOULD-FIX (fail-open: a claim users act on). `bash` and `execute_file` are not confined to the workspace

The fold's user-facing text says that `maxim.run`'s file tools, `bash` named among them, "reach only
`<home>/.maxim_workspace` ... never the process working directory or the home's own stores and config". It appears in:

- `docs/user/stable_api.md`;
- `docs/user/python-api.md`;
- `README.md` (only "file tools");
- the `breaking` fragment, which names `bash` as no longer reaching the working directory or the home's stores.

The text is accurate for `read_file`, `write_file`, `edit_file`, `glob` and `search_code`, which check paths against
`allowed_dirs`. It is not accurate for the other two:

- **`BashTool`** only starts in `allowed_dirs[0]`. Its own comment, in `tools/filesystem.py::BashTool.execute`, says:
  "Absolute paths inside the command are NOT contained -- this is a host shell, not a sandbox."
- **`execute_file`** runs a file that lives in the workspace, but the code inside it can reach anything.

`bash` is off unless `MAXIM_ALLOW_BASH=1`. An operator who arms it, trusting this text, gets a shell that can read
`~/.maxim/config` and every agent's stores. HEAD's wording ("reach only the run's home") was already loose, and the
fold made the claim stronger.

**Fix:** say exactly what is confined. Path-taking tools are confined to the workspace. `bash` and `execute_file`
start there, but they are host processes and are not contained. `bash` is off by default. Alternatively, drop `bash`
from the examples in all four places.

### D4. NIT (fail-open on the census; already-covered class). The A2 alias fold catches the class alias only

`test_tool_policy_census.py::test_tool_policy_is_never_aliased` catches three shapes:

- `import ToolPolicy as X`;
- `X = ToolPolicy`;
- `.ToolPolicy` attribute access.

It does not catch the shape A2 named, `U = ToolPolicy.unrestricted`, which is an `Assign` of an `Attribute`. It also
misses an import of a permissive helper under another name (`from ...tool_policy import aut_tool_policy as atp`).
Probe `p4_census_alias.py`: the alias test reports `[]`, and neither the factory scan nor the helper scan sees `U(...)`
or `atp()`.

The new blind-spot list in the docstring names neither shape, and the §3 "as built" line ("refuses an aliased
`ToolPolicy`") reads wider than the test. **Fix:** also flag an `Assign` whose value is `ToolPolicy.<factory>` and an
`ImportFrom` that gives a `PERMISSIVE_HELPERS` member an `asname`. Or add both shapes to the stated blind spots.

### D5. NIT (fail-closed). `Executor.refuses` disagrees with dispatch under `SimToolRegistry` at ceiling `ALL`

`refuses()` returns True when `registry.find(resolved) is None`. But `SimToolRegistry.get` returns a
`_FallbackRedirectTool` for an unknown name, and dispatch RUNS it. The redirect is a body that returns an untyped
failure, which books as a ran failure.

So the docstring's "True when `execute` would refuse ... without running anything" is false for the narrator's
registry. A wrapped narrator would skip the fear review and the anticipated-pain assessment of a READ decoy that only
returns an error. The effect is harmless: nothing is wider. **Fix:** a docstring sentence, or make `refuses` mirror
`get` (treat a non-None `get` as found). `permits()` has the same pre-existing gap.

### D6. NIT. "One Hippocampus episode" is stated as true of every gate refusal

The new runtime-tools.md invariant says a gate refusal books "the context pool the autonomy audit log and one
Hippocampus episode". A comma is missing after "context pool". The `changed` fragment says "A refused call is still
remembered as an episode".

`execute_parallel_actions` captures no episode, for refused or ran actions. The booking gate asserts the capture on
`execute_and_learn` only, as §3's as-built line says. **Fix:** scope the sentence to `execute_and_learn` and the
agent-fallback path.

### D7. NIT (fail-closed). `maxim.run` now scaffolds a nested workspace

`api._run_workspace` creates `<home>/.maxim_workspace/{drafts,notes,plans,scratch}`, then passes that path as the
override root. `build_tool_registry` scaffolds under `allowed_dirs_override[0]`, so it also creates
`<home>/.maxim_workspace/.maxim_workspace/{...}` (probe `p2_workspace.py`). The console handle's `<home>/workspace`
override already does the same (D70), so this is pre-existing builder behaviour. It is only clutter, but it is visible
to a user who is told to put files in that workspace. **Fix (optional):** pass the home's realpath to the builder and
let it scaffold, then confine to `get_workspace_path(home)`. Or accept it.

### D8. NIT. Leftovers from the first round's NITs

- `tool_dispatch.py` (the autonomy-rejection comment in `execute_parallel_actions`) still says "filed for an owner
  line" without `#1550` (A6 asked for the number at each site).
- E6 is not folded: `ToolPolicy.pinned`'s docstring still says an unregistered name is refused at layer `"pin"`, but
  it is refused at `"registry"`.
- `_refused_output`'s docstring says "no lookup is made to find one", but the inactive-scene branch now makes one for
  it.
- Removing `# pragma: no cover` from `PolicyHandle.narrowed`'s Protocol stub leaves one never-executed line in
  `tools/base.py`.

## Checked and found correct

- **E1/F1** (eight `test_agent_permissions.py` sites) and **A1** (the `ToolErrorKind` count, 14, with the seven gate
  members) are ported. Every assertion is kept.
- **E2/A4 (re-tag).** `_retag_body_gate_kind` runs on the one ran-and-returned path, after `tool.run` and before any
  bridge booking. The `OfflineRefusal` and raise paths return earlier, so the executor's own `OFFLINE` is never
  re-typed. A non-`ToolOutput` result is untouched. The deletion probe turns its test red.
- **E3/A8.** `policy` is a read-only property over `_policy`. No `src/`, `scripts/` or test writer of
  `executor.policy` remains, and the wrappers still reach it through `__getattr__`.
- **E5.** `permits()` refuses an alias under a pin, the same condition `execute` applies.
- **Owner decision: wrappers defer.**
  - `refuses()` follows `execute`'s order: deny-only on the raw name, alias plus pin, the full name gate, the
    inactive scene, the lookup, then the object gates. It counts nothing.
  - `gates_refuse` accepts only a literal `True`, so `MagicMock` and stand-ins without the method keep the old path.
  - `FearGatedExecutor` and `AnticipatoryPainExecutor` pass the action through unchanged.
  - There is no import cycle: each module imports cleanly first.
  - The forbid and mode sources it calls again are pure: `AutonomyController.forbids_tool` is a set lookup.
  - The deletion probes turn both tests red.
- **Owner decision: the bridge stays on under the latch.**
  - `is_offline()` is gone from `_suppress_nac`.
  - `finish_invocation` in `execute`'s `finally` retracts the NAc pending event that `record_tool_start` queued.
  - The red gate exempts only the offline case's start call, and asserts that the pending events net to zero for
    every case. Probe: a no-op `finish_invocation` fails all four offline cases.
- **Owner decision: one episode.** The red gate patches `tool_dispatch.capture_loop_action` and requires exactly one
  call on `execute_and_learn`.
- **Owner decision: `<home>/.maxim_workspace`.**
  - `_run_workspace` returns a realpath'd path.
  - `test_api_core` and the O1 red gate assert the workspace.
  - The bullet moved to the `breaking` fragment, with a migration line.
  - D3 is about the wording, not the decision.
- **A2.** `PERMISSIVE_HELPER_CALLERS` pins the four helper callers, and `test_every_pin_has_a_reason` covers the new
  table (D4 is the remaining gap).
- **E9/F12.** `_sense.execute` is pinned through `KNOWN_TOOL_RECEIVERS`.
- **E10.** The audit write now follows the `all_succeeded` update.
- **F4.** The `max_consecutive_failures` claim is corrected in the notes, the red-gate docstring and the helper
  docstring, citing #1549.
- **F6.** Both rationale scripts migrated and run.
- **F11, A7, A9, A10.**
  - F11: the D73 location column names `policy_from_tools_settings`.
  - A7: the CC3 roster line was added.
  - A9: the identity asserts were added in both tests.
  - A10: the `narrowed` reason is marked, `PolicyHandle.narrowed` is typed, and the `changed` fragment names
    `Tool.run`'s re-raise.
- **`lint_loop_mode_reads` allow-list entry.** The `"mode"` literal in `_permission_denial` is the refusal-layer label,
  and the lint is clean with the entry, so the entry is not stale.

## Verdict

D1 is a red CI gate. D2 is a fail-open regression the fold introduced (the policy seal), and D3 is a fail-open claim
users act on. Neither D2 nor D3 is in a class the first round's findings cover. D4 to D8 are NITs. Fold D1 to D3. A
delta-scoped round on those `src/` folds is then owed: the executor lens plus wire on `tool_policy.py`, and a claim
check on the four doc sites.

**NOT STOP**
