# #1645: the fast suite runs in deterministic shards (approach note, v2, switch built)

> **Status: v2, trial passed, switch BUILT 2026-11-13** (pending its three-lens review folds and merge). Written by
> maxim-4d, 2026-11-11. v1 had an adversarial gate pass, verdict DO NOT BUILD as written:
> D1 and D2, S1–S12, N1–N8, in [reviews/1645/adversarial.md](reviews/1645/adversarial.md). All of them are folded
> here. Where the build departs from this note, see "As built" at the end.
> Owner request (2026-11-11): do the real fix now, not a timeout bump.
>
> **v2 in one line:** the TRIAL is data only and can be built now. The SWITCH changes no ratchet rule and grants no
> exception; the statements sharding loses get direct tests first.
>
> **2026-11-13:** trial run 6 (CI run 43533293370) passed T1–T4; the owner approved the switch, built on branch
> `ci/1645-shard-switch` (the strict path, no exception). The switch deletes `.github/workflows/shard-trial.yml`; the
> deletion probes below live in `tests/unit/test_run_fast_suite.py`.

## Problem

- `unit-tests` takes 40–56 min on `main` (15 push runs, 2026-11-07..11; median about 50) against `timeout-minutes: 60`.
- #1643 timed out at 99% (#1645).
- The brief's 38.6-min median dates from 2026-10-16, when the suite had 15,427 tests. It now has about 18,761.
- §1 of the brief is updated in the switch PR (N6).

## Why xdist failed, and what is new here

- #1359's cause: `--dist loadfile` assigns files dynamically, so the order differs on every run, and coverage depends
  on order (up to 16 statements per package).
- Serial runs in the same order differ by about 2 statements per package, against `K_MIN = 5`.
- So the fix is a FIXED order: a committed partition, with each shard running in sorted file order.

Two variables sharding adds (review, claim 1), and how each is handled:
- **(a) Which test modules a process imported.** Serially, every test runs after ALL modules are imported. **Fix:**
  each shard COLLECTS THE WHOLE SUITE (all modules imported, zero-item ones included, so S4 is solved) and deselects
  the items outside its files, using a small plugin (`tests/_shard_plugin.py`, loaded with `-p`). Import-time state
  is then identical to serial; only which tests have already RUN differs.
- **(b) Contention timing and hash seeds.** 4 traced processes share 4 vCPUs. Measured by T1; `PYTHONHASHSEED`
  pinning is measured, not assumed (N7).

## Design

1. **The map.** `scripts/suite_shards.json` maps each test-module path to a shard index 0..N−1.
   - It is balanced from measured per-file times. An unmapped file goes to `sha256(path) mod N`.
   - A stale key is a printed NOTE (S5).
   - A test file renamed under `git diff -M` against the merge-base, whose new path is unmapped while the old key
     exists, FAILS with "carry the entry".
   - Duplicate keys and out-of-range indices fail.
   - A rebalance follows the switch rule below: it lands only when no scope falls by more than k.
2. **The runner.** `scripts/run_fast_suite.py` takes `--shards N` and the step's pytest arguments.
   - **Refuses to start** (S11, S2):
     - if `coverage.json`, `fast-suite.xml` or any `.coverage*` exists (the step's `CI_CLEAN` is extended to
       `.coverage.*`; the lint sees that);
     - if any shard would be empty.
   - **Forwards the step's arguments VERBATIM** (S1). It adds only an allowlist:
     - `-p tests._shard_plugin --shard k/N`;
     - `--junitxml=fast-suite.shard<kk>.xml` in place of `fast-suite.xml`;
     - `--cov-report=` in place of the json report;
     - `-p no:cacheprovider`.

     It refuses any argument it does not know how to forward or replace. It prints each shard's argv.
   - **Environment** (S1): it sets `COVERAGE_FILE=.coverage.shard<kk>` (zero-padded, N1) and scrubs every other
     `COVERAGE_*` and `COV_CORE_*` name. It does NOT override `HOME`/`TMPDIR`: conftest's isolation root already does
     that per process (S8).
   - **Each shard's plugin** writes the shard's selected and deselected nodeid sets plus its visited `Module` set to
     `shard<kk>.collect.json`.
   - **Reconciliation, as SETS** (S3, S4):
     - the shards' visited Module sets are identical (each collected the whole suite);
     - the selected sets are pairwise disjoint;
     - their union equals the whole collection's item set;
     - the merged junit's nodeids equal that union exactly.

     Any mismatch → exit 2.
   - **Results** (N4, S2):
     - any non-zero or negative shard return code fails the step;
     - exit 5 is a failure;
     - a shard stopped by `-x` is reported as "shard k stopped at its first failure (-x)";
     - a duplicate `(classname, name)` in the merge fails.
   - **Combine** (N1, S11):
     - `coverage combine` is given the exact N files;
     - every shard's data must `has_arcs()`;
     - then `coverage json -o coverage.json`;
     - only the combine writes `coverage.json`, and only the merge writes `fast-suite.xml`.
   - **Output** (N2, N3):
     - each shard's output is buffered and printed in shard order, with the failing shard's tail first;
     - per-shard wall time is printed every run;
     - there is a per-shard timeout below the job's.
   - **Modes** (S10): `--no-x`, `--no-cov` and `--cacheprovider` for `suite_at_commit.sh`. `--shards 1` is
     byte-equivalent to serial (same order, no partition), and the rig keeps it until its `RIG_KNOWN_FAILURES` is
     re-calibrated under sharding.
   - Every exception exits non-zero. (As built: the silent-swallow lint scans `src/maxim` only, so the runner has its
     own no-broad-`except` test; see "As built".)
3. **The wiring checks** are re-pointed exactly (S1, S12):
   - `ci_step_rules` pins the measuring step's run text to the exact runner invocation inside `ci_netns.sh`;
   - `config_rules` scans the runner, with one allowlisted `COVERAGE_FILE` assignment;
   - `lint_release_gates` rule 9 and `test_ci_workflow_shape.py` pin the same invocation;
   - a stub-pytest test in `test_lint_coverage.py` proves each shard's argv equals the step's arguments plus the
     allowlist;
   - the M47 test is re-pointed at the runner;
   - `box_quiet.py` counts a sharded suite as N, not 1 and not N + 1 (S10).
4. **No matrix fallback** (D2). If 4 processes exceed memory, N drops to 3 or 2 in the same job. A matrix design needs
   its own adversarial pass.

## The switch changes no rule (D1, the strict option)

The ratchet's `--pins-from-run` pins rises only. CHANGED never lets `missing` rise. LOWERING needs measured
`missing <= base`. So:
1. T2's executed-lines diff (sharded against serial, same commit) names every statement sharding loses.
2. Before the switch, ordinary serially-measured PRs cover those statements with DIRECT tests. Coverage that existed
   only as an order side effect is exactly what should be tested directly. Lines shown to be timing noise (they flip
   in serial's own runs) are named instead.
3. The switch PR: no scope falls by more than k. Rises are pinned with `--pins-from-run` from that PR's own run. No
   `missing` pin rises. Each scope's switch delta is printed in the PR for review (N5).

The exception path, for completeness, is not recommended. It would need a committed one-shot file keyed to the PR,
bounded per scope by the measured same-commit delta, refused if `src/`/`tests/` change, and printed per scope.

## The trial (pre-registered; data only; a separate non-required workflow on a draft PR)

`.github/workflows/shard-trial.yml` runs on one commit:
- **≥ 3 sharded runs and ≥ 3 serial runs** (CI only, never the leader box);
- each run records its merge SHA, `ImageVersion` and CPU model;
- each run uploads its `coverage.json`, `collect.json`, junits, peak RSS and its swap/OOM events.

| | Measure | Go condition |
|---|---|---|
| **T1, reproducibility** (S6) | max pairwise \|Δmissing\| across sharded runs, for every package floor, file floor and `overall` | ≤ 2 AND ≤ serial's own max pairwise + 1 |
| | churn: the symmetric difference of `executed_lines` per file | no-go if the sharded flip set > 2× the serial flip set, or it touches a line serial never flips in a scope with a file floor. Flipping lines are named (#1103's instrument) |
| | `PYTHONHASHSEED` pinned vs unpinned | measured, decided after |
| **T2, loss** (S7) | sharded-minus-serial executed lines, same commit | every lost line is listed. This feeds the strict path. It is not re-pinned |
| **T3, resources** (S8) | max wall time across ALL sharded runs | ≤ 25 min |
| | peak total RSS | ≤ 13 GB, with no swap and no OOM |
| | per-shard times | reported; they feed the balanced map |
| **T4, results** (S3, S8) | (nodeid, outcome) sets | equal to serial on EVERY run |

A pre-trial audit greps for fixed ports, fixed temp paths and file locks on fixed paths.

**Revert trigger** (S9): in the first 20 `unit-tests` runs after the switch, any ratchet failure that a re-run of the
same commit clears means revert to serial. #1103's diff is the instrument. This goes into M22's row too.
`timeout-minutes: 60` stays until that window closes (N3).

## Deletion probes (build)

Each of these must turn the build red:
- a file in no shard;
- an empty shard;
- a shard that exits 5;
- a shard killed by a signal;
- a deleted `.coverage.shard*`;
- a stale `.coverage.x`;
- line-only data in one shard;
- a dropped or duplicated junit case;
- a count-equal but set-different collection;
- a forwarded argument dropped (`--require-extras`, `--cov-branch`);
- an injected `COVERAGE_RCFILE`;
- a renamed file without its map entry.

## Owner decisions

1. **The switch:** the strict path (cover the lost statements first; no rule change; RECOMMENDED), or a one-shot
   committed exception.
2. **N = 4 in one job,** dropping to 3 or 2 on memory. No matrix.
3. **Interim relief while this lands** (days): raise `timeout-minutes` 60 → 75 as an explicit stopgap, reverted by the
   switch PR. Or none: keep 60, re-run timed-out PRs, and read every "cancelled" run before merging.

## As built (the switch, 2026-11-13)

Taken: 1 strict, 2 N = 4, 3 the stopgap (now reverted to 60). Departures from the design above:

- **Modes:** only `--no-cov` was built. `--no-x` and `--cacheprovider` would have had no caller:
  `suite_at_commit.sh` passes its own arguments, which carry no `-x`, and the runner adds `-p no:cacheprovider`
  unless the arguments already carry it.
- **Empty shard:** detected after the shards ran (reconciliation: "kept no tests"), not refused before launch. Each
  shard collects the whole suite, so its kept set is known only after collection.
- **Exit codes:** a reconciliation or merge failure exits 1, not 2. Exit 2 is kept for refusals before anything runs
  (leftover outputs, an argument the runner cannot forward, a selection not starting with `tests/`).
- **The stub-pytest argv test** lives in `tests/unit/test_run_fast_suite.py`, with the deletion probes, not in
  `test_lint_coverage.py`.
- **The silent-swallow lint does not cover the runner**: it scans `src/maxim` only. The runner's own guard is
  `tests/unit/test_run_fast_suite.py::test_the_runner_and_plugin_catch_no_broad_exception` (no bare or broad
  `except` in the runner or the plugin).
- **The rebalance rule** (Design 1: a rebalance lands only when no scope falls by more than k) has no check yet.
  Regression guard: process invariant — mechanization backlog M90.
- **The S9 revert window** is counted by hand. A revert restores the serial step, its pins and `timeout-minutes: 75`.
  Regression guard: process invariant — mechanization backlog M22.
- **The rename rule** runs in the lint job (`run_fast_suite.py --check-map`), not in the measuring step: that step
  runs inside the network namespace, and a push's base needs `gh api`.
- **Diagnostics:** each shard's log, junit and collection record are uploaded as `fast-suite-shards-<attempt>`
  (`if: always()`), separately from `coverage-json-<attempt>`.
- `.github/workflows/shard-trial.yml` is deleted.

Known limits, left as they are:

- the per-shard timeout kills each shard's process, not its grandchildren;
- the runner forwards no SIGTERM to its shards (a cancelled job relies on the runner's process group being killed);
- on a `-x` stop the sibling shards are not killed; they run to their end;
- `SHARD_TIMEOUT_S` (50 min) counts from the step's start, not the job's, so it names a hung shard before the job's
  60-minute limit only while the setup before the step (install, the re-use decision, mypy) takes under 10 min.

Pins (after the delta review): the step text lives in the runner (`CI_STEP_RUN`), but `lint_coverage.py` keeps its
own `CI_PYTEST_ARGS` and `CI_STEP_EXPECTED` literals, and the runner's text must EQUAL them (rule 9 compares against
the same literal). The stub-pytest test drives the runner with the gate's literal, not the runner's constant.
