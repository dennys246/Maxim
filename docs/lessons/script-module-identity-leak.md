# A test that re-registered a script module made another test's patch miss

**Date:** 2026-10-17 (PR #1339, the M22 xdist trial)
**Class:** test-order dependence hidden by serial order: a leaked `sys.modules` replacement.

## What happened

The first CI run of the fast suite under `pytest -n auto --dist loadfile` (run 39050775378, `unit-tests`) failed
`tests/unit/test_m1b_writer_stamps.py::test_an_analyzer_refuses_with_3_when_its_code_cannot_be_established[analyze_exp56.py]`
with `assert 4 == 3`. The test had passed serially on `main` and in a local macOS xdist trial.

## Mechanism

- The test imports `_provenance as P` at collection and patches `P.in_process_code_provenance` to raise.
- `scripts/analyze_exp56.py::main` runs `from _provenance import ...` at CALL time, so it uses whatever
  `sys.modules["_provenance"]` holds then.
- `tests/unit/test_report_provenance_m1.py::_load_script("_provenance")` re-executed the script by path and set
  `sys.modules["_provenance"]` to the new module, in five tests, and never put the old one back.

After that, the patch landed on a stale object. The analyzer ran the real provenance check, which passed on CI's clean
checkout, and the one mock row made the verdict NO-VERDICT: exit 4, not the refusal's 3. Serially, `test_report_...`
sorts after `test_m1b_...`, so the leak never reached its victim; `--dist loadfile` let one worker run them the
other way round. `-q` hid the worker's file order, and `-x` stopped at the first of the two failing parameters. A
two-test reproducer (the leaking test, then the victim) failed on the pre-fix tree. `test_clean_flag_998.py`
re-registered `_provenance` the same way at import time, harmless only because it happened to load first; and
`test_harness_identity_1003.py` carried a fixture whose docstring described the leak and worked around it.

## Fix

- Both leakers load their copy under a private name.
- `tests/script_module_guard.py`, wired into `tests/conftest.py` around every test, every test module and every
  collected module, fails whatever leaves a `scripts/` module replaced or removed under the name `import`
  resolves it to, then restores the original so later tests on that worker do not fail for the wrong reason.
  Its positive control is `tests/unit/test_script_module_guard.py`.

Auto-restoring alone (snapshot and silently restore around each test) was rejected: it would hide the leaking
test and let the next one of its kind ship, and a restore that silently changes what a test sees is the band-aid
shape CLAUDE.md forbids. The guard restores only AFTER it has failed the leaker.

Rule and blind spots: [docs/agents/ci-testing.md](../agents/ci-testing.md) §3.

## Sibling class: reload (#1350)

The identity guard cannot see `importlib.reload`: the object stays the same while its contents change. #1339's next CI run failed in `test_lane_placement_runtime.py::TestCloudPlacementCap::test_cloud_primary_placement_consumes_cap`. `test_lane_routing_via_config.py` had reloaded `lane_backends` to clear a once-per-startup flag, which redefined `BackendGateError`, and the placement test still held the old class. It reproduced serially as well, with the routing file run first. The fix swapped the reload for the module's existing reset helper (`_reset_lane_env_applied`), and the AST tripwire `tests/unit/test_no_module_reload.py` now bans reload across `tests/`.
