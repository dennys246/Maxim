# Step 0a: second delta round (Executor + Wire integrity), the D1 fold

Reviewer: a different reader (not the first-round reviewers or the D1 reviewer). Scope: `git diff 1f7a4c94 1ee422ac`.
Read-only apart from this file. Narrow tests run in the worktree; a deletion probe was run in a scratch copy.

## Verdict

**STOP: no further round owed.** D1 is closed. Every finding below is a fail-closed NIT in a class already
covered (doc wording, message polish, or a run that stops later than the CLI check). None is a BLOCKER or a
SHOULD-FIX.

## Probes

### 1. Repair commands dispatch above the new call: yes

Traced `cli.py::_main_impl` from the top. Before `require_valid_user_profiles()`, in order: `serve`, `tunnel`,
`doctor`, `bench`, `roy`, `model` (add/remove edit `profiles.yml` through `model_cli`'s own YAML read and call no
profile accessor), `hivemind`, `config`, `peer` (all actions), `--show-last`, `--clear-last`, `--last` (a replay
falls through to the check, as it should), argparse (`--help`/`--version` exit inside `parse_args`),
`_validate_operational_grant`, `_early_leader_boot`, `_normalize_args`, `--list-models`, `--delete-model`,
`--clear-cache`, `--clear-memory`. No onboarding wizard exists. Test `..._stops_cli_runs_but_not_the_repair_commands`
pins doctor and config.

Below the call: foundry, every mode, and the bare-`maxim` menu (N1).

`--list-models`, `--delete-model` and `_normalize_args` with `--llm` read profiles before the call, with nothing
catching the error. They now raise `UserProfilesInvalid`, which `main()` renders as exit 2. That is fail-closed, and
main refused at import in the same way.

### 2. Run modes that bypass both seams: none still degrade

- The CLI-launched agentic, live/robot (`embodied_runtime.selfy.Maxim` is constructed only from `cli.py`, apart from
  a TYPE_CHECKING import in `runtime/capture.py`), foundry, curate and research paths all sit below the CLI call.
  These are D1's four rows.
- `api.run`, `api.imagine` and `api.campaign` reach `_validate_model` / `build_primary_router` / `start_simulation_mode`
  without a swallow, so they raise. The same holds for `create.py` and `console/handle.py`.
- `api.research` (library): `_build_research_llm` still swallows the seam's error (WARNING, heuristics). The
  `start_research_mode` → `start_simulation_mode` call that follows raises without a swallow, so the run stops before
  any sim turn (N2).
- `maxim roy run`, dispatched above: `start_simulation_mode` raises. The runner records `aborted_at="priming"` at
  ERROR with the `UserProfilesInvalid` text and exits 1 (N3).
- `maxim bench` reads no profile and measures the peer path. Unaffected.
- `maxim serve`, dispatched above: its runs build through `console/handle.py` → `build_primary_router`, which
  propagates. A leader's early proxy boot reads no profile. It starts, and the process then exits 2 at the check,
  the same outcome as main's import-time refusal.

### 3. Exception plumbing: clean

- Every `except ConfigurationError` in `src/` matches with `isinstance`. Nothing compares by exact type or by
  message, and `UserProfilesInvalid` subclasses it, so `config_cli` and the others still catch it.
  `doctor/checks.py::check_user_profiles` calls the loader directly with `path=`, so the conversion does not touch it.
- `ConfigDirUnreadable` is re-raised unchanged, before the conversion. `main()` checks `UserProfilesInvalid` first;
  the two types are disjoint, so the order does not matter. `test_unreadable_config_dir_1403.py`: 17 passed.
- `raise ... from exc` keeps the chain. The traceback goes to DEBUG.
- N4: the path appears twice in the message.

### 4. Goldens' `_PREIMPORT` additions: no production behaviour hidden

These modules were loaded only as a side effect of the old eager `maxim.agents` / `maxim.math` init. In production,
`runtime/loop_setup.py` (context_pool) and `runtime/bio_stack.py` (AngularGyrus) import them during setup. At worst,
`agent_loop`'s lazy `working_memory` import lands in the first tick. Measured in a fresh process after
`import maxim.runtime.agent_loop`: importing all six leaf modules costs about 23 ms, once per process. The harness
pre-imports them because of its fake-clock contract (import-time defaults must not read the real clock inside the
window). That is a fidelity guard for the test, not a latency gate, so the list hides no timing regression.

Completeness: every user of `_loop_harness`, `_orch_harness` and `_cradle_loop_driver` was run **in isolation**, one
file per process: orchestrator_characterization_golden 41 passed; gl2a_trio_golden 3; afferent_red_gates_loop 2 + 4
xfail; agent_loop_selection_golden 8; interoceptive_outcome 25 + 2 xfail; experience_clock_decay 4 + 7 xfail;
afferent_red_gates_units 4 xfail; tty_stdin 8; afferent_latency_characterization 2. No `ImportedInsideWindow`.
`test_profile_readers_v9.py`: 37 passed, 1 xfail (#1479).

### 5. Deletion probe

In a scratch copy with the CLI `require_valid_user_profiles()` call replaced by `pass`,
`test_a_malformed_profiles_file_stops_cli_runs_but_not_the_repair_commands` **fails**. The guard guards. It fails by
a 180 s child timeout, though: agentic mode runs with no LLM and never returns. See N5.

## Findings (all NIT, all fail-closed)

- **N1** `cli.py::_main_impl`: bare `maxim` (the menu) now refuses before the menu is drawn, and the menu holds the
  "Run diagnostics" and "Show help" entries. The refusal's fix hint names `maxim doctor`, and main refused at import
  anyway. Nothing to do, unless the owner wants the menu reachable.
- **N2** `simulation/research_orchestrator.py::_build_research_llm`: library `maxim.research()` still logs "Writer/
  Reviewer will use heuristics" before `start_simulation_mode` raises. The run stops, but a misleading WARNING comes
  first. A `except UserProfilesInvalid: raise` beside the `LeaderRefusal` re-raise would make it stop at the first
  build. Optional, and it is the class D1(b) named.
- **N3** Doc wording: the changelog fragment says "every `maxim` run mode exits 2 with `UserProfilesInvalid`".
  `maxim roy run` sits above the check and exits **1** as a recorded priming abort. Either say "refuses (exit 2;
  `maxim roy` records an abort)", or leave it.
- **N4** `models/language/config.py::_ensure_user_profiles`: the message is `cannot load {path}: profiles file at
  {path} is not valid YAML: ...`, so the path appears twice. Every `load_user_profiles` error already names the file.
  The per-profile schema errors do not, which is why the wrapper adds it. Cosmetic.
- **N5** `tests/unit/test_profile_readers_v9.py::_CLI_BODY`: if the check regresses, the test fails only after the
  180 s subprocess timeout, with no assertion message. Consider a shorter timeout for this case, or stubbing the
  agentic loop entry in the child so it fails at once with "reached mode dispatch".
- **N6** (carried over from D2) The re-entry guard still raises `RuntimeError`, which the swallowing handlers would
  eat. Nothing can re-enter today. `require_valid_user_profiles`'s docstring says "or raise `ConfigurationError`"; it
  raises `UserProfilesInvalid`, `ConfigDirUnreadable` or that `RuntimeError`.
- **N7** `cli.py::main`: the `UserProfilesInvalid` branch puts the fix hint into `message` instead of passing
  `fix=e.fix_hint`, as the `BackendError` branch does. The JSONL event then has no separate `fix` field. Cosmetic.

The rule-T pin (1458 → 1419) and its history line match the move. `_handle_foundry` is a verbatim move.
