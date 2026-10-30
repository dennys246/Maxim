# Step 0a (#1315, V9): Executor lens review

Reviewer: Executor lens ([docs/CODE_REVIEW.md](../../../CODE_REVIEW.md)). Diff: `feat/1315-step0a-lazy-inits`. I
reviewed the staged diff. It was committed as 7bfbee71 while I worked, and `git diff HEAD~1 HEAD` matches the reviewed
diff exactly (ignoring `index` lines). Probes ran against scratch copies of `src/` (old = `HEAD~1`, new = the commit).
Nothing in the worktree was modified except this file.

**Verdict: no BLOCKER.** Every migrated reader behaves as it did before, for every input I could construct. The
findings below are two SHOULD-FIX and a few NITs.

## What I verified as identical (with evidence)

- **Lazy tables against the old eager inits.** I compared, with an AST check, each old `__init__`'s bound names
  against the new `_LAZY` and `__all__` for runtime, agents, tools and utils. No name is missing and none is extra. No
  lazy name collides with a submodule name, and every lazy name resolves.
- **No lost import-time side effects.** I scanned every module-level call, decorator and Call-valued assignment in
  `src/maxim`. The only registrations are self-contained: `selfy._robot_registry` is private to its own module, the
  `hivemind/bundle` migrations register into their own module, and `http.on_engage(close_all)` only matters once
  `http` holds pools. The `atexit` registrations in `sim_logger` cover state that exists only once that module is
  loaded. The five formerly eager `utils` modules have no module-level side effects.
- **Cycles.** Every `from maxim.<pkg> import <lazy name>` in `src/` is function-nested: `cli.py` twice and
  `agentic_runtime.py` twice. No code anywhere in src, scripts or tests walks an attribute chain into a submodule it
  never imported (`maxim.<pkg>.<sub>.<attr>`). A fresh-process import of each of the 191 modules under runtime,
  agents, tools, utils and models gave 0 failures.
- **Reader semantics.**
  - `_normalize_profile` and `normalize_llm_profile` are the same function (the public one delegates).
  - `.get(x, {})` → `get_profile(x) or {}` is equivalent. Stored values are always dicts, an empty entry maps to a
    fresh empty dict, and every site only reads the result.
  - `key in _BUILTIN_PROFILES` → `get_profile(key) is not None` is equivalent, because no value is ever None.
  - `iter_profiles()` returns `sorted(items())`, which is a list, exactly as the old sites built it.
  - `profile_aliases()` returns a copy, and `cli` only reads it.
- **Merge atomicity.** `profile_loader.apply_user_profiles` stages everything before it commits anything, so a failed
  attempt leaves the tables unchanged. The guard is set only after the merge succeeds.
- **`load_llm_config`.** Its only table read (`_BUILTIN_PROFILES.get`) comes after `_normalize_profile`, which
  ensures the merge first.
- **End to end, with a valid user profile and alias.**
  - `maxim --list-models` output is byte-identical, old against new.
  - `maxim.models.LLM_MODELS` now holds the user profile in a fresh process (old: `False`), which is the V9 fix.
  - `normalize_llm_profile(alias)`, `list_llm_profiles()` and `api.list_models()` agree, old against new.
- **Malformed `profiles.yml`.** Old: `import maxim.models.language.config` raised, and `maxim --list-models` and
  `maxim doctor --json` both died with a traceback. New: import succeeds, and every accessor raises
  `ConfigurationError` on every call (checked twice). `--list-models` still exits 1. `doctor --json` now completes
  and reports the failing checks, an improvement.
  - The sites whose `try: from config import ... except Exception: <skip>` used to swallow the import failure now
    raise from the call instead, which is louder (fail-closed). These are `_apply_local_llm_override`,
    `_apply_cloud_cli_overrides`, `_apply_cloud_fallback_override`, `_placement_entry_to_provider`,
    `_estimate_swap_n_ctx`, `check_peer_vs_local_conflict` and the benchmark script.
  - Two sites swallow exactly as before: `has_cost_tracking` still returns True, and `_lane_profile_is_cloud` still
    logs at debug and returns False.
  - I found no fail-open regression.
- **The executor's `suppress_nac`.** Every consumer of `suppress_nac` (`_run_started`, `_fail_ran`, `_book_failure`)
  is already gated on `self._tool_pain_bridge is not None`, so pinning it to False without a bridge changes nothing.
  Deletion probe: forcing `_suppress_nac = False` even with a bridge fails
  `test_build_executor.py::TestInteractiveAttributionGate::test_a_broken_interactive_read_does_not_attribute_to_nac`,
  so the computation is guarded.
- **Narrow tests (worktree).** `test_profile_readers_v9` + `test_advisor_gate` (closure/profile/lazy selection):
  19 passed, 3 xfailed. `test_build_executor`, `test_tool_failure_credit_1200`, `test_wire_2_percept_aversion`,
  `test_user_profiles`, `test_model_cli` and `test_doctor`: 359 passed. `lint_function_length.py`: clean.

## Findings

### F1. SHOULD-FIX (fail-open, tests): the suite's one profile merge now runs inside whichever test first touches an accessor, under THAT test's `XDG_CONFIG_HOME`

`models/language/config.py::_ensure_user_profiles`, with `tests/conftest.py` (isolation block).

Before this diff, conftest's first `maxim` import merged `profiles.yml` once, at collection, under the isolated
home. That timing is the whole point of `test_conftest_isolation_order.py`. Now nothing merges at collection. A probe
test showed `_user_profiles_loaded is False` at the first test.

I instrumented `_ensure_user_profiles` to log `PYTEST_CURRENT_TEST` and the resolved path, then ran 10
XDG-redirecting files one at a time. The first merge happened inside, for example,
`test_doctor.py::TestDoctorCLI::test_runs_without_retry`,
`test_cloud_solo_auto_detect.py::TestConfiguredProfile::test_a_configured_local_profile_is_not_replaced_by_a_cloud_one`
and `test_diagnose_matches_doctor.py::test_both_entry_points_run_the_same_checks`. The last two ran under their own
`tmp_path/xdg`.

None of those dirs holds a `profiles.yml` today, so nothing leaks yet. The failure scenario is a future in-process
test that writes `tmp_path/xdg/maxim/profiles.yml` and calls any accessor before anything else has. Its profile and
aliases are then merged permanently into `_BUILTIN_PROFILES`, `_PROFILE_ALIASES` and `LLM_MODELS` for every later
test in the process. That is order-dependent and silent, the same class as the 2026-10-27 isolation-order incident.
A malformed file there instead raises inside an unrelated test.

**Fix:** one line in conftest after the isolation block, `_ensure_user_profiles()`, which restores "merged once,
under the isolated home". Alternatively, an autouse fixture that snapshots and restores the three tables and the
guard. The red gates run in fresh `-I` children, so neither fix masks them. Also update the
`test_conftest_isolation_order.py` docstring, which still says importing reads `profiles.yml`.

### F2. SHOULD-FIX (unguarded mechanism): nothing guards the executor's bridge-gated `sim_logger` import

`runtime/executor.py::Executor` dispatch (`if self._tool_pain_bridge is not None:` around the `sim_logger` import).

Deletion probe: I replaced the gate with `if True:`, which restores the old unconditional import on every dispatch.
`test_build_executor`, `test_advisor_gate` and `test_profile_readers_v9` all still pass (45 passed, 62 xfailed).
`maxim.simulation` is not in `FORBIDDEN_MODULES`/`FORBIDDEN_PREFIXES`, so even the future flip of
`test_import_closure` will not catch the regression. Measured from a fresh process, the gate saves only 2 modules on
top of `maxim.runtime.executor`'s 131, yet the changelog claims it.

**Fix:** either add a fresh-process test that builds an Executor without a bridge, dispatches one tool and asserts
`"maxim.simulation.sim_logger" not in sys.modules`, or add `maxim.simulation` to the forbidden list if it belongs
there. If neither is wanted, drop the changelog bullet.

### F3. NIT (latent, fail-loud): the RLock comment promises safe re-entry, but re-entry recurses into a second merge

`models/language/config.py::_ensure_user_profiles`. The lock is commented "re-entrant: the merge itself may resolve
a name through an accessor". A nested accessor call during the merge finds `_user_profiles_loaded` still False,
re-acquires the RLock and calls `apply_user_profiles` again, without bound.

Demonstrated by wrapping `profile_loader.apply_user_profiles` so it calls `config.get_profile('x')` first: it
re-entered 6 times before my cap stopped it. Without the cap the result is a `RecursionError`. Today's loader never
calls an accessor, so this is latent. It fails loud.

**Fix:** set an in-progress flag before the merge and have a nested call return without merging (it then reads the
pre-merge tables), or use a plain `Lock` and correct the comment.

### F4. NIT: `maxim.models.__getattr__("LLM_MODELS")` can raise a non-AttributeError

`models/__init__.py::__getattr__`. With a malformed `profiles.yml`, `hasattr(maxim.models, "LLM_MODELS")` raises
`ConfigurationError` instead of returning a bool. `from maxim.models import *` raises too, because `LLM_MODELS` is in
`__all__`. Before this diff the whole package import raised, so this is no worse. I found no in-repo caller. Noted
only for the pymaxim public surface.

### F5. NIT: a module-level `__getattr__ -> Any` makes mypy accept any name on the four packages

`runtime/__init__.py`, `agents/__init__.py`, `tools/__init__.py`, `utils/__init__.py`. mypy treats a module that
defines `__getattr__` as having every attribute, so `from maxim.runtime import Executr` type-checks and then fails
only at runtime. The `TYPE_CHECKING` mirror types the real names but cannot reject typos. This is the usual cost of
PEP 562. If wanted, a unit test that checks every `from maxim.<pkg> import X` in src/scripts/tests against `_LAZY`
plus the submodule names would close the gap. Architecture's call.

### F6. NIT: stale "at import" prose outside the hunks (behaviour unaffected)

- `doctor/checks.py::check_user_profiles` docstring: "raises `ConfigurationError` from module import".
- `models/language/profile_loader.py` module docstring: "Reads ... at import time", and "Loading is intentionally
  synchronous + import-time".
- `tests/unit/test_unreadable_config_dir_1403.py` module docstring: "`models.language` loads the user profiles at
  import".
- `scripts/benchmark_cross_session.py::_registered_profile_cloud_flag` docstring: "into `_BUILTIN_PROFILES` at
  import".
- `tests/unit/test_conftest_isolation_order.py`; see F1.

The `doctor` fix hint "maxim will refuse to start" still holds in practice: `--list-models` exits 1, and
`load_llm_config` and `normalize_args` raise. But it now fires at first lookup rather than at import.

### F7. NIT (pre-existing, moved verbatim): `_lane_profile_is_cloud` calls `logger.debug` in its `except` even when `logger is None`

`runtime/lane_backends.py::_lane_profile_is_cloud`. The success branch checks `logger is not None`, but the except
branch does not, so `logger=None` plus a lookup error raises `AttributeError`. The original block in
`_maybe_auto_spawn_server` had the same bug, so this is not a regression. Now that it is its own function, it is a
one-line fix.

## Not probed

- I did not run the full `test_fresh_import_sweep` (slow, about 800 children). I ran the 191-module subset above.
- I did not run the full suite or prepr, because of the one-full-suite-per-box rule.
