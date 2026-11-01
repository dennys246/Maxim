# #1315 item 3: micro-round on the D1–D8 folds

Scope: only the folds for D1–D8 of `code_delta.md`, as staged in `.worktrees/v-toolpolicy` (uncommitted, on top of
06af7bca). Charter: `docs/CODE_REVIEW.md` (delta rounds, stop rule). Read-only. New probes:
`/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/248fdc0d-2f2e-471a-b37d-dacb96a7ecce/scratchpad/review3_delta2/`
(`p3_docs.py`, `p4b.py` + `fake/m.py`, `p5_pin_layers.py`, `p6_inactive.py`).

What I ran (worktree, `PYTHONPATH=$PWD/src`, narrow only, no other suite on the box):

- `bash scripts/run_mypy_public.sh`: `Success: no issues found in 55 source files`.
- `ruff check src/ tests/` clean; `ruff format --check src/ tests/` clean (`ruff check scripts/` reports 2 F841 in
  two unrelated experiment scripts this branch does not touch).
- `lint_function_length`, `lint_no_silent_swallows`, `changelog_fragments.py check`: exit 0.
- `test_tool_policy`, `test_tool_policy_census`, `test_build_executor`, `test_atonia_red_gates_effect_policy`,
  `test_refusal_booking_red_gate`, `test_stop_reasons`, `test_api_core`, `test_agent_permissions`: 214 passed.
- `test_tool_pain_pending_leak`, `test_grayscale_tools_annotation`, `integration/test_advisor_gate`,
  `test_execute_and_learn_characterization`, `test_tool_dispatch`, `test_fear_gate`: 117 passed, 61 xfailed (the
  pre-existing advisor red gates).

## Fold by fold

- **D1 (closed).** The inactive-scene branch builds `inactive = _Refusal(...)` and passes
  `self.registry.find(tool_name)`; mypy is clean. `find` is a lock-guarded dict lookup on `ToolRegistry` and a plain
  lookup on `SimToolRegistry`, and it returns a deactivated scene tool, so the refusal now names the real effect. Probe
  `p6_inactive.py`: `INACTIVE_SCENE`, `{'refusal_layer': 'registry', 'effect': ['WRITE_LOCAL']}`, and `refuses()`
  agrees (True). No test asserted `"unknown"` for this branch.
- **D2 (closed).** `ToolPolicy.__init__` checks `_key` again, without the pragma, with a comment that states why.
  The delta reviewer's `p1_reinit.py` now prints `re-init refused: TypeError ToolPolicy has no public constructor`.
  `TestReviewFolds::test_a_reinit_cannot_widen_a_policy` passes and would go red without the check (the call carries
  valid permissions, ceiling and reason, so nothing else raises and `pytest.raises` fails). The module docstring's
  "`__new__` and `__init__` check a module-private key" is true again.
- **D3 (closed).** All five sites now scope the claim to the path-taking tools and say `bash` and `execute_file` are
  not contained. Probe `p3_docs.py` builds `maxim.run`'s registry the way `api.run` does
  (`allowed_dirs_override=[_run_workspace(home)]`): `read_file` (absolute and `../config.json`), `write_file`, `glob`,
  `search_code` all refuse a path in the home outside `.maxim_workspace`, and nothing is written there. `git_diff`
  goes through `contained_path`. `maxim.run` has no mode parameter, the loop default is `observe` (passive), the model
  cannot raise its own mode (`mode_switch._self_grantable`), and the passive `dispatch_refusal` refuses `bash`,
  `execute_file`, `run_tests` and `git_commit` (all `HOST_ACTING_TOOLS`). `BashTool` defaults its cwd to
  `allowed_dirs[0]` and `ExecuteFileTool` runs with `cwd=tool_workdir(allowed_dirs)`, so "start in that workspace"
  holds. The text claims what the code does.
- **D4 (narrowed, see R1).** The alias test now also flags `ImportFrom ... as` for `ToolPolicy` and every
  `PERMISSIVE_HELPERS` member, and `X = ToolPolicy.<attr>`. The old `p4_census_alias.py` re-implements the old walker,
  so it is stale; `p4b.py` runs the real test function against a synthetic module and catches both D4 shapes
  (`import aut_tool_policy as atp`, `U = ToolPolicy.unrestricted`).
- **D5 (closed).** `refuses()`'s docstring now states the `SimToolRegistry` decoy case and that it fails closed.
- **D6 (half closed, see R2).** `runtime-tools.md` now says "the context pool, the autonomy audit log and, on
  `execute_and_learn`, one Hippocampus episode".
- **D8 (three of four closed, see R3).** `tool_dispatch.py` cites `#1550`; `ToolPolicy.pinned` names the
  `"registry"` layer for an unregistered name (probe `p5_pin_layers.py`: `nope` → `not_registered`/`registry`, `b`
  and alias `aa` → `policy_pin`/`pin`); `_refused_output`'s docstring no longer says no lookup is made.

## Findings

### R1. NIT (census-only blind spot; the D4 class). Three alias shapes still pass the census

`p4b.py` shows `test_tool_policy_is_never_aliased` still misses:

- an annotated assignment, `V: object = ToolPolicy.deny` (`ast.AnnAssign`, not `ast.Assign`);
- a tuple assignment, `X, Y = ToolPolicy.unrestricted, 1` (the value is an `ast.Tuple`);
- a helper bound to a plain name, `W = aut_tool_policy` (an `Assign` whose value is a `Name` in
  `PERMISSIVE_HELPERS`).

None occurs in `src/` today (the test passes on the tree). The test's docstring says "`ToolPolicy` or one of its
factories is never assigned to another name", which the first two shapes contradict. **Fix (test-only):** also
match `ast.AnnAssign` and walk a `Tuple` value, and match a `Name` value in `{"ToolPolicy", *PERMISSIVE_HELPERS}`;
or name these three shapes in the module's stated blind spots. Either is a `tests/` change, so folding it owes no
further round.

### R2. NIT (docs). The `changed` fragment still says every refusal is remembered

`changelog.d/1315-item3-tool-policy.changed.md`: "A refused call is still remembered as an episode (no credit)."
`execute_parallel_actions` captures no episode. **Fix:** "A refused call on `execute_and_learn` is still remembered
as an episode (no credit)."

### R3. NIT (fail-closed). The Protocol-stub pragmas were not restored

D8's fourth item is unfolded: `tools/base.py::PolicyHandle`'s `ceiling` and `narrowed` stubs no longer carry
`# pragma: no cover`, and the coverage config excludes only `pragma: no cover`, `if TYPE_CHECKING:`,
`raise NotImplementedError` and `__main__`. At worst a line or two of never-run Protocol body is added to
`maxim/tools`'s denominator, which is too small to move the pinned 71.3%. Restore the pragmas or accept it.

### R4. NIT (cosmetic). Two over-long lines from the folds

`executor.py::Executor.refuses`'s docstring has a 157-character line (the D5 sentence). Line 19 of the `breaking`
fragment runs past 120 columns. E501 is not enforced and `ruff format` does not wrap docstrings, so nothing fails.
In `python-api.md` and the fragment, "a command can name any absolute path" fits `bash`. For `execute_file` the
accurate phrase is "the file's code can reach any path". The text does not overclaim.

## Verdict

D1, D2, D3 and D5 are closed. D4, D6 and D8 are closed except for R1 to R3. Every finding is a NIT. R3 and R4 are
fail-closed. R1 is the census blind-spot class D4 already covers, and its fold is test-only. R2 is docs-only. None of
them touches `src/` behaviour. Fold R1 and R2 (and R3 and R4 if wanted) or file them as a follow-up. No further round
is owed.

**STOP**
