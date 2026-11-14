# #1645 adversarial gate-design pass: the fast suite in deterministic shards

> Reviewer: adversarial gate lens. The work was read-only. It covers `ci_sharded_suite_1645_approach.md` (DRAFT v1)
> against `scripts/lint_coverage.py` (docstring, `K_MIN`, `ci_step_rules`, `config_rules`, `changed_entry`'s rules
> as documented, `--merge-floors`, `--pins-from-run` / `run_measurement` / `pins_from_measurement`),
> `.github/workflows/test.yml` (`unit-tests`), `scripts/ci_netns.sh`, `scripts/suite_at_commit.sh`,
> `scripts/lint_release_gates.py` (rule 6 `_junit_cases` / `junit_problems`, rule 9 wiring), `scripts/check_lane_roster.py`
> and `scripts/lane_rosters/`, `tests/conftest.py`, `tests/substrate/conftest.py`,
> `tests/unit/test_ci_workflow_shape.py`, `tests/unit/test_lint_coverage.py::test_the_suite_script_measures_with_cis_own_command`,
> `docs/agents/ci-testing.md`, `docs/plans/reviews/m60/adversarial.md`, `docs/plans/coverage_ratchet_m60_approach.md`,
> `docs/lessons/ci-efficiency-2026-10-16.md` and outstanding.md M22/M60. Date: 2026-11-10.

**Verdict: DO NOT BUILD as written.** The trial (T1 to T4 on a draft PR) is data and may run once S6 and S8 are
folded. The switch (design steps 3 and 4) may not be built yet.

The steady-state idea is sound in kind: a fixed file→shard partition, run in fixed order, is deterministic in the
same way serial is. But two parts of the note are fail-open or impossible:

- **D1, the one-time re-pin.** The note promises that "the ratchet's rules do not change". The re-pin it describes
  cannot happen under those rules, and the tools it names do not do what it says. Landing the switch therefore needs
  either a red PR that can never go green, or an unstated mass floor reset. A floor reset is exactly the D2 shape
  that M60 was rejected for.
- **D2, the matrix fallback.** As written, it turns the required check into one that a failing shard can skip,
  and a skipped check counts as passing.

With D1 folded (the strict path below), the fallback dropped, and the SHOULD-FIX items adopted, the verdict becomes
BUILD WITH CONDITIONS.

---

## The claims tested

**(1) "Same files, same order, same process → as reproducible as serial."** This is half true.

- **Order:** yes. A shard is a fixed ordered list in a fresh process, so the order effect that sank #1339 (the
  `loadfile` reshuffle) is gone.
- **Process composition** is a separate variable, and it is not order. Serially, pytest COLLECTS all ~756 test
  modules before the first test runs. So every test runs with every module's import-time side effects in place:
  registries, module-level `register_*` calls (e.g. `tests/integration/test_offline_latch.py`'s
  `http.register_endpoint`), and `src` modules already imported. In a shard, a test sees only its quarter of them.
  That is stable run to run, so it does not break reproducibility. But it can change OUTCOMES (a hidden dependency on
  another file's import), not only coverage, and the map becomes an input to test semantics (S3, S5).
- **Timing:** not established. The note says the cause of #1359 was "order variation, not parallelism", but that is
  inferred from a forward-vs-reversed SERIAL experiment. That experiment shows order matters. It does not show that
  contention does not.
  - Four coverage-traced processes on 4 vCPUs leave no spare core for the background threads that serial runs had.
  - #1103 already records statements that flip between identical serial CI runs because of timing. Contention can
    only enlarge that set.
  - Each process also draws its own `PYTHONHASHSEED`, so there are four independent chances to hit a branch that
    depends on set iteration order.
  - Only T1 can answer this, and with two runs and a net-count metric it is underpowered (S6).

**(2) A new test file only affects its own shard.** The same in kind as serial, and not worse. Serially, a new file's
import-time effects reach the WHOLE suite. In a shard they reach only that shard. The new failure modes are moves,
not adds:
- a rename (old key stale, new path hashed to a possibly different shard);
- a rebalance;
- a deletion that forces a map edit.

Each of these moves a file across a shard boundary. That changes the global state, and so the coverage and possibly
the outcomes, of OTHER packages in two shards at once. A pure test rename can then trip the ratchet (S5).

**(3) Combining branch data.** Correct in principle. `coverage combine` takes the union of the line and arc sets, and
partial branches are computed at report time from that union. That is what one process running the union would
record. Each shard's data also records the source files it never executed (`_post_save_work` touches the files found
under `source`), so unimported files still appear. Conditions: see N1.

**(4) Consumers.** None breaks silently today, because each wiring check is string-based and would go red
(fail-closed). The danger is the fix: re-pointing those checks at a runner they cannot read. See S1 for the runner as
an unscanned gate component, S2 and S3 for `_junit_cases`' silent collapse of duplicates, S10 for local, rig and box
consumers, and D2 for `--pins-from-run` under the fallback.

**(5) `-x` per shard.** Failures are not hidden: any non-zero shard fails the step, and up to four failures are
reported instead of one. The residual risks are diagnostic only (N4), plus S2's empty-shard case.

**(6) Trial conditions.** Insufficient and partly gameable (S6, S7, S8): two runs; net counts that hide churn; no
file floors; no per-line flips for diff coverage; no memory threshold; T4 compares counts, not sets; T2 "does not
gate", so any delta can be re-pinned; and there is no revert trigger after merge (S9).

**(7) M60 D1/D2.** The steady state reopens neither, because no tolerance widens. The one-time re-pin does reopen D2
(this review's D1): a floor's `missing` pin is raised to whatever the new measurement says, so a real fall present at
the switch is absorbed. It is the same pattern as "a stale missing pin lets a floor be lowered past a real fall".

---

## DO-NOT-BUILD

### D1. The one-time re-pin cannot pass the unchanged rules, and the tools named do not perform it

*Fail-closed as literally specified: the switch PR stays red. It turns fail-open as soon as anyone grants the
exception it implicitly needs.*

**Attack.** Design step 4 says every floor is "re-taken as the minimum of two sharded CI runs … via
`--pins-from-run`", while step 3 says the rules do not change. Against `scripts/lint_coverage.py` as it stands:

- **`--pins-from-run` writes only rises.** `pins_from_measurement` pins scopes more than `band` ABOVE their floor. A
  scope below its floor is a note ("cover code; a pin cannot fix a fall"). It reads ONE run and takes no minimum.
- **"Minimum of two runs" is `--merge-floors`, which serves only NEW floors.** It merges only scopes that are `null`
  in the committed file (unless `--all-scopes`). Its input is the floors block CI prints, which appears in the log
  only while some floor is `null`.
- **Floors cannot be re-bootstrapped on a PR.** A `null` package floor fails closed on every run, and a file floor
  set back to `null` fails. So a mass re-bootstrap is not reachable on a PR.
- **CHANGED never lets `missing` rise.** `h.missing <= min(m.missing + K_MIN, b.missing)`. LOWERING needs
  `m.missing <= b.missing`, with no tolerance.
- **The HEAD rule fails any scope more than k under its floor.** k is `max(ceil(0.1% of statements), 5)`.

**Consequence.** Every package or file floor whose sharded measurement is more than k worse than its serial pin (the
note expects up to ~16 statements) fails the switch PR. No pin edit can fix it, because `missing` may not rise. The
only ways through are:
- (a) cover those statements; or
- (b) an exception to CHANGED/LOWERING for this one PR.

The note silently assumes (b). That is a mass floor reset. Every real fall that exists at the switch commit, or that
lands in a PR between the trial and the switch, becomes part of the new pin, invisibly, because the order shift
explains any number up to ~16. Small file floors are hit hardest: k = `K_MIN` = 5 is already 10 pt of a 50-statement
file. Every later rebalance ("takes the re-pin step below") repeats the reset, so the hole recurs on the map's
schedule.

**Fix (the strict option, recommended and listed first):** the switch changes no rule and grants no exception.
1. The trial's executed-lines diff (sharded against serial, same commit; see S7) names every statement that sharding
   loses.
2. Before the switch, land ordinary PRs that cover those statements directly. Those PRs are measured serially, where
   they only help. Coverage that existed only as an order side effect is exactly the coverage that should be tested
   directly, and doing so also shrinks future order noise.
3. On the switch PR, scopes that rose above band are pinned with `--pins-from-run` on that PR's own run, as today. No
   `missing` pin rises.
4. A rebalance follows the same path: it may land only when no scope falls by more than k. Otherwise it waits for the
   tests.

Rewrite owner decision 2 to offer this first.

**If the owner chooses the exception instead** (CLAUDE.md: "a committed machine-readable exception", strict option as
the default recommendation), it must be:
- one-shot, keyed to the switch PR, in a committed file the lint reads;
- bounded per scope by the sharded-minus-serial delta measured on the SAME commit in the trial, never by "up to 16";
- refused if the switch PR changes anything under `src/` or `tests/` other than the runner and map;
- printed loudly, scope by scope, in the CI log.

### D2. The matrix fallback is fail-open as written

*Fail-open.*

**Attack.** "Fallback: a 4-job matrix plus an aggregating job still named `unit-tests`."

- An aggregating job with `needs: [shard jobs]` is SKIPPED when any shard fails. ci-testing.md §3 says it outright:
  "a skipped required check counts as passing".
- The usual cure is `if: always()` plus a `needs.*.result` check, and that is banned twice: `check_repo_settings
  --static` requires a required check to map to a job with no job-level `if:`, and `ci_step_rules` bans a job-level
  `if:` on the gate's job.
- `ci_step_rules` reads ONE job's steps. "Nothing between the measuring suite and the gate except MemoryHub" cannot
  be stated across jobs, because the artifact upload and download in between is exactly a step that can rewrite the
  data.
- `--pins-from-run` binds the step "Run required fast suite" inside the job named `unit-tests`. Under a matrix, no
  such step measures.
- `lint_release_gates` rule 9 and `test_ci_workflow_shape.py` both look for the `--cov` pytest step inside
  `unit-tests`.

**Consequence.** If memory forces the fallback mid-trial, it would be built under time pressure as a different gate
whose obvious form passes a red shard.

**Fix.** Delete the fallback from this note. If 4 processes exceed memory, the in-job fallback is N = 3 (or 2), which
keeps every single-job rule. A matrix design needs its own adversarial pass.

---

## SHOULD-FIX

### S1. The runner becomes an unscanned gate component

*Fail-open.*

**Attack.** Today `ci_step_rules`, `config_rules`, `lint_release_gates` rule 9 and
`test_ci_workflow_shape.py::test_every_test_run_is_inside_the_loopback_only_network_namespace` read the exact pytest
command in the workflow text. After the change they can only see that `run_fast_suite.py` was CALLED with the right
strings. What the runner then passes to the N pytest processes is invisible to all four. A runner edit could:
- drop `--require-extras=console,sign` or `--ignore`, or change `-m "not slow"` to something narrower;
- drop `--cov-branch`;
- add `--cov-append` or `--cov-config`, or set `COVERAGE_RCFILE`.

`_BANNED_OPTS_RE` and `_BANNED_ENV_RE` scan only workflows, composite actions and the pytest ini sources, so they
would not see it. The design itself sets `COVERAGE_FILE`, a name the env ban exists to keep out, from a place the ban
does not scan. Rule 4 of the M60 lens applies here: the runner is now part of the measurement.

**Fix.**
- **Forward verbatim.** The runner forwards the step's pytest arguments byte for byte, and adds only an allowlisted
  set: the file list, `--junitxml=fast-suite.shard<k>.xml` replacing `--junitxml=fast-suite.xml`,
  `--cov-report=json:...` replaced by `--cov-report=` (S11), and `-p no:cacheprovider`.
- **Refuse unknown changes.** It refuses any argument it does not recognise as forward or replace.
- **Print the argv.** It prints each shard's exact argv.
- **Pin it with a test.** A test in `test_lint_coverage.py` drives the runner with a stub pytest and asserts that
  each shard's argv equals the step's arguments plus the allowlist, with nothing else.
- **Scan the runner.** `config_rules` adds `scripts/run_fast_suite.py` to its scanned sources, with an exact
  allowlist for the one `COVERAGE_FILE` assignment.
- **Scrub the environment.** The runner removes every other `COVERAGE_*` / `COV_CORE_*` name from the shards'
  environment.
- **Pin the step.** `ci_step_rules` pins the measuring step's run line to the exact runner invocation.
- **Swallows.** The runner is in the silent-swallow lint's scope (it lives under `scripts/`, so check that
  `lint_no_silent_swallows` covers it), and every exception exits non-zero.

### S2. An empty shard runs the WHOLE suite

*Fail-open for coverage, and duplicates collapse in the junit.*

**Attack.** `pytest <args>` with no file arguments falls back to `testpaths = ["tests"]`. This happens if a shard's
file list is ever empty: N above the file count, a map bug, or every file of shard k deleted. That shard then runs the
entire serial suite. The combined coverage is the union with a full serial run, so it is at least the serial
measurement and hides any shard-level fall inside the band. In the merged junit, every case appears twice.
`lint_release_gates._junit_cases` keys on `(classname, name)` and keeps the LAST, so a failed copy can be overwritten
by a passed one.

**Fix.**
- The runner refuses an empty shard (exit 2) before launching anything.
- Each shard's junit must contain only nodeids from its own files.
- The merge refuses any duplicate `(classname, name)`.
- Add "empty shard" and "duplicate case" to the deletion probes.

### S3. Every reconciliation compares counts where it must compare sets

*Fail-open.*

**Attack.** These checks are all counts:
- "the per-shard collected test counts sum to the whole";
- "merged case count = sum of shard counts = collected count";
- T4's "equal test counts".

Counts can match while the sets differ. Parametrisations built at import time (from a registry another test module
populates, `BACKEND_CLASSES`, or a tool list) can differ between the whole-suite collection and a shard's collection,
per claim (1). One shard gains a case and another loses one, and the sums agree.

**Fix.** Use nodeid-SET equality throughout. The whole-suite collection must equal the union of the per-shard
collected nodeids. Each shard records its own collected set: a ten-line plugin, or a shard's junit plus its
deselected list. The union must also equal the merged junit's nodeids, and T4 compares (nodeid, outcome) sets against
serial.

### S4. Deriving the file set from `--collect-only -q` drops files that collect zero items

*Fail-open, mostly for coverage.*

**Attack.** `-q` lists only selected items. A file whose tests are all deselected by `-m "not slow"` (21 files use
`pytest.mark.slow`, and 2 set a module-level slow `pytestmark`), or that collects nothing, is absent from the
"collected set". So it lands in no shard, and the disjoint-union check still passes, because it checks against that
same set. Serially, that module is still IMPORTED at collection: its module-level `src` imports are covered, and its
registrations are live. Every future "mark the last test of a file slow" then silently moves `src` coverage in the
sharded world in a way it does not serially.

**Fix.** The shard file set is every `Module` collector pytest visits under `testpaths` with the suite's
`--ignore`s (zero-item modules included), recorded by the collection plugin from S3. Every one is mapped and passed.

### S5. Moves across shards change other packages' coverage; stale-entry refusal makes the map a hot conflict file

*Fail-closed, but costly, and the cost pushes toward D1's reset.*

**Attack.**
- A test rename leaves a stale key (exit 2, so the author must edit the map), and the new path hashes to a shard that
  is likely different (3 in 4).
- A deletion forces a map edit too.
- Both make `scripts/suite_shards.json` a file many PRs edit. A CONFLICTING PR fires no Tests (the green-PR lesson).
  Two PRs that delete tests conflict on the map even when the deletions are unrelated.
- A file that changes shard moves the global state of two shards, and with it the coverage of unrelated packages
  (claim (2)).

**Fix.**
- A stale entry is a printed note, not a refusal: a key for a deleted file cannot hide a test, so refusing it buys no
  safety.
- A rename carries its key. The runner reads `git diff -M --name-status` against the merge-base, and a renamed test
  file whose new path is unmapped while the old key exists fails with "carry the entry `<old>` → `<new>`".
- Duplicate keys and out-of-range shard indices stay refusals.

### S6. T1 is underpowered and blind to churn

*Fail-open: a noisy design passes the trial, then flakes on main as #1339 did.*

**Attack.**
- **Two runs.** Two runs estimate a tail badly, and serial's "±2" came from more runs than two.
- **Net counts.** `|Δmissing|` is a net count: two lines gained and two lost reads as Δ0 with four flips.
- **Packages only.** File floors, which have the same `K_MIN`, and `overall` are not named.
- **Diff coverage has no tolerance.** Per-line flips decide whether a small PR's changed lines reach 80%, and
  nothing measures them.
- **A vacuous clause.** "None above K_MIN" adds nothing to "≤ 2".
- **No serial control.** "As reproducible as serial" is a RELATIVE claim, and T1 has no serial control on the same
  commit and runner pool.

**Fix.**
- **Runs.** At least three sharded runs and three serial runs of one commit. Record each run's merge SHA (a PR re-run
  reuses its old merge ref), `ImageVersion` and CPU model.
- **Go condition.** For every package floor, file floor and `overall`, the maximum pairwise |Δmissing| is ≤ 2 AND no
  greater than serial's maximum pairwise value + 1.
- **Churn.** Report the symmetric difference of `executed_lines` per file across the sharded runs. No-go if the
  sharded flip set is more than twice the serial flip set, or touches a line serial never flips in a scope with a file
  floor.
- **Attribution.** Name the flipping lines; they are #1103's instrument.
- **Option worth measuring:** pin `PYTHONHASHSEED` across shards, since it is a noise source sharding multiplies
  (see N7).

### S7. T2 "does not gate", so any delta is re-pinnable

*Fail-open, through D1.*

**Attack.** Every sharded-minus-serial delta is accepted as "order shift" and absorbed. That includes deltas caused by
runner defects (S2, S4) or a dropped conftest effect.

**Fix.**
- T2 produces the executed-lines diff, sharded against serial on the same commit.
- Every scope whose sharded `missing` exceeds serial's by more than the measured serial order bound (16) is
  explained line by line before the switch.
- Under D1's strict path, every lost line is either covered by a direct test or shown to be timing noise.

### S8. T3 and T4 miss the concurrency failure classes

*Fail-open for flakes; fail-closed for timeouts.*

**Attack.**
- **Memory.** Peak RSS is "recorded" but has no go/no-go. A run that swaps or loses a child to the OOM killer can
  still finish under 25 min.
- **Wall time** comes from one run.
- **T4** compares counts (S3) from one pair of runs.
- **"Concurrency already shaken out by #1339"** rests on one green xdist run under a different distribution. New
  shared state appears with four processes in ONE cwd: four `.pytest_cache` writers racing at session end, fixed
  ports, fixed `/tmp` names, file locks on fixed paths. The `HOME`/`TMPDIR` part of the row is already true:
  `tests/conftest.py` builds a per-process isolation root before any maxim import. The runner should not override
  `HOME`; it would only mask what conftest does.

**Fix.**
- **Memory.** No-go if peak total RSS is above 13 GB, or if there is any swap use or OOM event (read `dmesg` or the
  cgroup's `memory.events` after the step).
- **Wall time.** The go condition is the MAXIMUM wall time across all trial runs being ≤ 25 min.
- **T4.** (nodeid, outcome) set equality against serial on EVERY trial run.
- **Pre-trial audit.** Grep for fixed ports and fixed temp paths, and pass `-p no:cacheprovider` to every shard (as
  `suite_at_commit.sh` already does).

### S9. No pre-registered revert trigger after merge

*Fail-closed noise that trains people to re-run red runs.*

**Attack.** #1359 found the noise only on `main` after merge, and the trial cannot see the tail.

**Fix.** Pre-register a rule: in the first 20 `unit-tests` runs after the switch, any ratchet failure that a re-run
of the same commit clears reverts to serial, and #1103's artifact diff is the instrument. Write it into the note and
into M22's row.

### S10. Local, rig and box consumers

*Fail-closed, but it erodes M47 and the box rules.*

**Attack.**
- **`suite_at_commit.sh`** must keep: no `-x`; `-p no:cacheprovider`; `--no-cov`; the `suite stage <name>: exit=N`
  lines `rig_suite_verdict.py` reads; and the M47 shape test (`CI_CLEAN` before `CI_COV_ARGS` before `--local`).
- **The box rule** is "two full suites per box". One sharded suite uses 4 cores, so two of them are 8 traced
  processes, on top of the CPU contention S6 worries about. `box_quiet.py`'s detection is anchored on argv starts: it
  must count the runner as one suite and must not count its four children as four.
- **The rig** has real-time cadence tests that already fail there when run alone (`RIG_KNOWN_FAILURES`). Under 4×
  contention, more will fail, and the calibration would silently stop matching.

**Fix.**
- The runner takes `--no-x`, `--no-cov` and `--cacheprovider` modes, and `suite_at_commit.sh` calls it.
- Re-point the M47 test at the runner invocation.
- `box_quiet.py` weights a sharded suite as its N.
- Re-calibrate `RIG_KNOWN_FAILURES` with a sharded rig run before the rig verdict accepts one. Until then, the rig
  keeps serial with `--shards 1`; a one-shard run is the same partition-free order as serial.

### S11. Output-path hygiene

*Fail-open if the combine path ever falls through.*

**Attack.**
- "Collect ONCE with the suite's exact arguments" would run the collection under `--cov`, and write `coverage.json`
  and `fast-suite.xml`.
- Each shard given `--cov-report=json:coverage.json` writes the same `coverage.json` concurrently, so a later runner
  bug (combine skipped, or an exception caught) leaves one shard's partial file for the gate to read.

**Fix.**
- The collection pass drops every `--cov*` and `--junitxml` argument.
- Shards get `--cov-report=` only.
- After the clean, the runner REFUSES to start if `coverage.json`, `fast-suite.xml` or any `.coverage*` exists. That
  is better than deleting them inside the runner, where the lint cannot see it. Keep the deletion in the step's
  `CI_CLEAN`, extended to `.coverage.*`.
- `coverage.json` and `fast-suite.xml` are written only by the combine and merge.

### S12. Re-pointing the wiring checks must keep them meaningful

*Fail-open if they are re-pointed loosely.*

**Attack.**
- `lint_release_gates` rule 9 requires `--cov`, `pytest` and exactly one `--junitxml=fast-suite.xml` in the run text.
- `test_ci_workflow_shape.py` requires `bash scripts/ci_netns.sh` and `"$PY" -m pytest tests/` in the run text.

A runner invocation can satisfy all of these as dead strings while doing something else.

**Fix.** Each check pins the exact runner invocation as a whole, and S1's runner test proves what the runner then
runs. The netns property still holds, because the children inherit the namespace. The shape test should say so and
assert the runner is launched inside `ci_netns.sh`.

---

## NIT

- **N1. The combine.**
  - Assert that every shard's data `has_arcs()`; combining line and arc data fails, but say so.
  - Pass the N files explicitly to `coverage combine`.
  - Naming: `.coverage.shard1` against `.coverage.shard10` is safe only because pytest-cov combines
    `<name>.*`. Use zero-padded indices.
  - Add a one-time probe: on one shard's data, `coverage json` from the CLI equals pytest-cov's own JSON byte for byte
    (apart from the timestamp).
- **N2.** Four interleaved `-q` streams make the failure log unreadable. Buffer each shard's output and print it in
  shard order, with the failing shard's tail first.
- **N3.**
  - Give the runner its own per-shard timeout, so a hung shard names itself before the job's 60 min kill.
  - Print per-shard wall time every run, so imbalance from hash-assigned files is visible before it times out.
  - Keep `timeout-minutes: 60` until S9's window closes.
- **N4. `-x` messages.**
  - A shard stopped by `-x` fails the junit-count check too. Report it as "shard k stopped at its first failure (-x)",
    not as a merge error.
  - A negative return code (killed by a signal) is a failure.
- **N5.** The band and stale-high `missing` pins (the documented residual) absorb part of the one-time shift without
  any rule firing. Print each scope's switch delta in the switch PR for review.
- **N6.** The problem statement's "median of about 50 min" disagrees with the brief's 38.6 min (2026-10-16). Cite the
  runs, and update §1 of the brief in the same PR.
- **N7.** `PYTHONHASHSEED`: pinning it across shards removes a noise source. It can also hide hash-order bugs.
  `test_stable_hash_two_process.py` sets its own seeds, so it is unaffected. Measure before deciding.
- **N8.** Owner decisions: list the strict option first in decision 2 (D1). Decision 1's "matrix" option goes (D2).

---

## Inputs the gate consults after the change, and how each could lie

| Input | How it could lie | Finding |
|---|---|---|
| The workflow step text | a correct runner call while the runner does something else | S1, S12 |
| `run_fast_suite.py` | dropped selection or extras arguments, banned coverage options or env, a swallowed merge error | S1, S11 |
| The collection | zero-item files omitted; count-equal but set-different parametrisations | S3, S4 |
| `suite_shards.json` | empty shard → whole suite; renames and deletions move files; stale keys as a hot file | S2, S5 |
| Shard exit codes | negative (signal); `-x` stop misreported | N4 |
| `.coverage.shard*` | partial or stale files; mixed line and arc data | S11, N1 |
| `coverage.json` | written by a shard instead of the combine; union with an accidental full run | S2, S11 |
| `fast-suite.xml` | duplicates collapsed by `_junit_cases`; counts not sets | S2, S3 |
| The floors file at the switch | a mass reset absorbs real falls | D1 |
| The runner hardware | contention timing, hash seeds, memory/OOM | S6, S8 |
| The trial records | wrong merge ref; two runs; net counts; a non-gating T2 | S6, S7 |
| Post-merge runs | the noise tail unseen by the trial | S9 |
| Local, rig and box | M47 drift, box over-subscription, rig calibration | S10 |
| A matrix fallback | skipped aggregator = passing; cross-job gap | D2 |
