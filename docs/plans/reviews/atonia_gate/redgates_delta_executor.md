# Delta round, Executor lens: the red-gate fold (0ac02acb..11c90ab1)

Scope: `git diff 0ac02acb 11c90ab1`, which is the staged fold, now committed. That is seven test files, the
`tests/conftest.py` autouse reset and notes v6.3. The question for each changed gate: once the code lands exactly as
v6.3 says, does the gate flip only when the behaviour is right?

**Verdict.** 0 BLOCKER, 3 SHOULD-FIX (1 fail-open, 2 fail-closed), 4 NIT. **A further round is owed:** one fail-open
SHOULD-FIX (S1) is a fold the previous round introduced. It is small, and a delta-scoped round on S1 alone would do.

## Measured

- Default run: **12 passed, 184 xfailed** (22 s in pytest, 55 s wall).
- `--runxfail` with a junit report: **184 failed, 12 passed**. Every one of the 184 is `failure type=AssertionError`,
  with no `error` elements.
- `python scripts/lint_release_gates.py`: `release gates: clean` (exit 0).
- The cloud-lane census driver (`_cloud_lane`) reaches the lane today. In a scratch child with a blocking hook,
  `LLMRouter(cfg)` builds, and `generate_json` attempts `getaddrinfo` (Anthropic, three retries) plus one
  `subprocess.Popen`. So the site is live, not vacuous.

## SHOULD-FIX

### S1 [fail-open; incomplete fold of B2] `test_no_effect` accepts every relative path
The fold made `_outside()` return `False` for every relative name. Its justification: "the CWD is read-only... and the
HOME/TMPDIR/home_dir diffs catch any other directory." That covers only the four snapshotted trees.

- **The gap.** A `dir_fd`-relative write into any directory outside those four trees is now invisible. Examples: a
  sibling under `tmp_path` such as `harness/`, the repo, or any absolute directory. The directory's own open is an
  `O_RDONLY` open, recorded only under `ROOTS`, and the relative leaf is accepted.
- **Before the fold** a relative name was flagged unless it existed in `report_dir`. That was fail-closed, and it was
  wrong for transient temp names (B2).
- **Inherited, same class:** the FS-mutation check looks only at `e[1][:1]`. So `os.rename`/`os.replace` with a
  relative src and an absolute dst outside `report_dir` also passes.
- **Fix, cheap:**
  - (a) Snapshot all of `tmp_path` except `report_dir`, not four trees.
  - (b) In the child, record every directory open, meaning an `open` event whose flags carry `O_DIRECTORY` or whose
    path is a directory. Fail on any directory open outside `report_dir` and the import roots, or require that the
    only `dir_fd` base is `report_dir`.
  - (c) Check both path args of `os.rename`.

### S2 [fail-closed; notes vs seam contradiction] `_reset_for_tests` is required, but v6.3 says no reset exists in `src/`
Notes §6 (line ~980) say: "No reset exists in `src/`, and the census forbids one there". §1A.9 row 12 says
"conftest-only reset". Yet three places depend on `maxim._offline._reset_for_tests()`:

- the conftest autouse fixture;
- `test_offline_latch.py`'s SEAMS block;
- `test_refusal_booking_red_gate.py::_rig("offline_ceiling")`, which calls `_require_offline("engage",
  "_reset_for_tests")`.

The consequences, if the code follows the notes:
- The four `offline_ceiling-*` booking cases can never flip.
- The conftest fixture is a permanent no-op, so an in-process `open()` leaves the latch up for the rest of the session.

**Fix:** pick one and write it in v6.3.
- Either `_offline.py` carries `_reset_for_tests()` as a sanctioned private test seam, and the census allows exactly
  it.
- Or the conftest/rig reset touches named private state, `_ENGAGED`/`_LEDGER` say, which the notes then name.

### S3 [fail-closed; incomplete fold of three-lens NIT 2] the reset does not restore the environment
v6.3 says `engage()` writes five keys to `os.environ`, `MAXIM_OFFLINE` among them, and that `is_offline()` "engages
on any call that sees `MAXIM_OFFLINE` truthy".

- The autouse fixture restores neither the environment nor those keys.
- No in-process engager uses `monkeypatch` for them. Those engagers are `_rig("offline_ceiling")`, the
  `test_advisor_gate` in-process `open()` gates, and `test_advisor_log_red_gates::test_o7`.

So after the first in-process engage, `MAXIM_OFFLINE=1` stays in the pytest process. The next `is_offline()`
re-latches, and later unrelated tests are refused far from the cause. It is loud, but it would block the item-1/0b
PRs' suites.

**Fix:** have the fixture snapshot the five keys and restore them after the reset (the CLAUDE.md autouse-scrub
lesson).

## NIT

1. **`test_per_layer_deletion_sweep_pin` hangs on an unstated choice.** It demands that non-pinned READ tools refuse
   at layer `"pin"`. But the seam order is permissions(3) before pin(5). If `ToolPolicy.pinned()` also carries the pin
   keys as a name allow-list, which is a natural reading of "authorises objects, not names", those tools refuse at
   `"permissions"` and the gate cannot flip. The impostor case and the booking `policy_pin` case are unaffected,
   because their names match the pin key.
   **Fix:** state in v6.3 that `pinned()` has no name allow-list (a name mismatch refuses at `"pin"`), or accept
   either layer for a different name. This is fail-closed and could be argued up to SHOULD-FIX. It sits here because
   v6.3 does say the pin, not the name, is the authority.
2. **`test_open_builds_all_three_layers` patches `_session._build_dispatch_stack`.** If `open()` lives in
   `maxim/advisor/__init__.py` and does `from ._session import _build_dispatch_stack`, the spy never sees the call.
   The gate then stays red with the behaviour right (fail-closed).
   **Fix:** pin in the SEAMS block that `open()` resolves the builder through `_session`'s globals.
3. **`ENV_READ_ALLOW_CALLERS` is matched with `str.startswith("numpy")`, with no dot boundary.** That is not a
   realistic hole: `_caller()` skips only `os`/`collections.abc`/`__main__` frames, so an advisor read is attributed
   to the advisor module. A callback numpy invokes is the advisor's own frame too. Still, use
   `m == p or m.startswith(p + ".")`.
4. **The §1A.11 gate skips `maxim_command` (two classes: `tools/reachy.py`, `tools/reachy_stubs.py`) and
   `execute_sandbox_script`.** All three are real `Tool` classes, so this is a fail-open gap. `web_search` and
   `list_directory` are only aliases and are rightly out. The process-runner gate likely backstops the sandbox tool,
   but nothing backstops `maxim_command`.
   **Fix:** for an unbuilt name, check the class-level record of every `Tool` subclass whose `name` matches.

## Checked and holds

- **The policy-aware booking rig.**
  - `_build_executor` switches on `find_spec("maxim.runtime.tool_policy")` and matches v6.3's factories:
    `unrestricted`, `deny`, `only(names, *, ceiling, reason)` and `pinned(instances, *, ceiling, reason)`.
  - `_Probe` declares no effect and so records `ALL`, which v6.3 says is consistent rather than a raise.
  - The legacy-text fallback cannot make a wrong-kind refusal pass. A mistyped kind gets `refused=None`, and the
    zero-writes assertion then fails.
  - The `_NAC_STATE` snapshot closes unspied writes, and `ScenarioBroken` guards every precondition.
- **The census.**
  - The not-applicable rule is closed. `NEEDS` maps each library site, a not-applicable site is never counted, it is
    warned and named in the failure message, and an all-not-applicable run is `ScenarioBroken`.
  - On a lean CI cell the guard runs on fewer sites. That is stated, and it is S7's evidence question, not new.
  - `HF_BACKED_SITES` matches §6's fold.
- **The ddgs sentinel.** `_search_with_ddg_package` imports `DDGS` at call time from `sys.modules["ddgs"]`, so the
  patch is live. The sentinel records into `OUT["spy"]` before it raises, so the function's `except Exception` swallow
  cannot turn the site into a pass.
- **`_expect_raises`** catches only the named class, and anything else propagates. Every use names a specific type:
  `AdvisorCompositionError`, `SealedRegistryError`, `OSError`/`ValueError`. Each sits behind an existence assert, so
  a missing API is an AssertionError today.
- **The O4/O6/O8 folds** are now behavioural and narrow:
  - `ConfigurationError` and `ModelError` are typed, and any other exception stays red.
  - `PERMISSION_DENIED` is typed.
  - O6 is driven through the real producers.
- **`test_open_builds_all_three_layers`**, setting aside NIT 2: it makes exactly one call, with all three layers and
  `registry_tools=None`, and checks `session._executor is built` and the identity of the pinned tool. Its positive
  controls are not vacuous.
- **chmod 0555 on the CWD.** It is restored in `finally`. Under a root CI user the chmod does nothing, but the CWD tree
  diff is still the backstop. Gap: S1.
- **The conftest fixture.** One `sys.modules` lookup per test, so its cost is negligible. It is safe for every test,
  and a no-op until 0b ships. Its correctness depends on S2 and S3.
