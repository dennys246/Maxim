# CI and pre-PR efficiency review (2026-10-16)

**What prompted it:** a PR was costing about three hours of wall time, roughly 60 minutes of local snapshot suite
plus around three CI runs of ~38 minutes each, while CI had stopped finding functional defects. Four parallel
readers measured where the time went on 2026-10-16, after the 1.3.2 cut. The owner's decisions are below. The rules
that follow from them live in [docs/agents/ci-testing.md](../agents/ci-testing.md); the work is in
[docs/plans/outstanding.md](../plans/outstanding.md) §Mechanization backlog, rows M22 and M59–M65.

## Measured

**The suite.** 15,427 tests: unit 14,508, substrate 352, integration 336, behavioral 173, other 58. CLAUDE.md's
`## Testing` line still said 9,168 tests in 12:09 (2026-08).

**CI `unit-tests`.** Median 38.6 min per run. The fast-suite step is a median 34.65 min, serial, under
`--cov --cov-branch`. Install takes 3.11 min, of which the `llama-cpp-python` sdist build is 2m50s. There is no pip
cache (the only cache in the workflow is the HuggingFace model cache of the nightly lanes).

**Locally.** `scripts/suite_at_commit.sh` runs ~50–60 min. **Estimate**, not a timed run: it is extrapolated from
the CI step and the coverage overhead below.

**Parallelism.** `pytest-xdist` is in the `[test]` extra and on CI's pip line, and nothing uses it. A serial sample
of 4,544 tests took 642 s; the same sample at `-n 6` took 138 s, with 0 failures. That is a **pre-trial
measurement**, not the M22 trial: one sample, one machine, not the whole suite, and nothing checked for
`~/.maxim`, port or global-state collisions beyond "no failures".

*Update 2026-10-18:* the M22 trial then ran (#1339) and was reverted (#1359). **Coverage depends on test order**:
two serial runs on one commit and one machine, forward vs reversed file order, differed by up to 16 missing
statements in one package (34 in total), so an exact-pin ratchet flakes under `--dist loadfile`, which reshuffles
the order every run. Outcome and follow-ups: [docs/agents/ci-testing.md](../agents/ci-testing.md) §1.

**Coverage overhead.** About 1.5x, measured on a subset only.

**Slow tests.** 11 tests of 5 s or more account for 44% of the sample's serial time. Among them:
`test_r3_pilot` (64 s), `test_scn_thread_safety_1217`'s forced-switching test (47 s), `test_lint_bf2_obligation`'s
real-repo test (43 s), `test_b1_floor_run` (87 s for the file) and `test_water_trial_smoke` (36 s).

**Autouse fixtures.** 54 of them, costing under a minute in total, so negligible.

## What CI caught (last 40 merged PRs, #1240–#1306)

- Median open-to-merge time 1.9 h.
- 81 PR `Tests` runs: 10 failed, 17 cancelled.
- **Zero functional test failures.**
- 9 failures were the coverage ratchet. 7 were exact-pin or rise-above-floor bookkeeping, often right after merging
  `main`, or because macOS and Linux measure coverage differently. Two were real gaps (#1299, #1268).
- 1 failure was the PR-title lint (#1275).
- Docs-only PRs used about 24% of PR CI minutes.
- Every merge re-runs the full workflow on its push to `main`.
- The `edited` trigger (kept for base retargets) also restarts a run on every title or body edit.

## Owner decisions (2026-10-16)

1. **CI's green `unit-tests` run on the head SHA counts as the final full-suite run.** Adopted, but under the
   "Enforced, or on the backlog" principle the rule text in CLAUDE.md ("Faster without lowering the bar" item 2)
   and [development-flow-speed.md](development-flow-speed.md) §2 changes only when M64 lands. M64 makes
   `scripts/pr_merge_readiness.py` assert a green, non-cancelled `unit-tests` run on the PR's `headRefOid`. Until
   then the last local full suite still runs on the exact commit pushed.
2. **Coverage-ratchet redesign.** A rise above a pin becomes a warning or an auto-bump, and pins come from CI's
   Linux artifact. Adopted, pending the adversarial design pass any gate change owes (M60).
3. **The `edited` trigger** gets a per-job guard plus a workflow-shape test that every job carries it (M61). The
   test closes the "silently missed by the next job added" objection in test.yml's comment, so that comment gets
   rewritten in the same change.
4. Unindexed memory files are archived rather than deleted. This is outside the repo.

## Caveats and dead ends

- The local 50–60 min figure is an estimate, and the xdist and coverage-overhead numbers come from samples. None
  of them is a recorded trial. M22 stays open until a full `-n auto` run is recorded with every collision isolated
  or marked serial.
- "Zero functional failures caught by CI" is about these 40 PRs, which all ran the full local suite before pushing.
  It shows the local run is where defects were found. It is **not** evidence that CI could be dropped: CI is still
  the only place coverage is measured on Linux, inside the network namespace, with the diff-scoped lints judged
  against `push_base`.
- The test.yml comments citing "~12 min of real work" and 18:26/26:46 figures are stale. A docs-only change cannot
  edit `.github/`, so they are recorded here and in the brief's §4.

## Backlog rows

M22 (amended: xdist installed but unused, plus the pre-trial sample), M59 (`scripts/prepr.sh` + a sync test),
M60 (coverage-ratchet redesign), M61 (`edited`-trigger guard + shape test), M62 (pip cache / prebuilt
`llama-cpp-python`), M63 (slow-test trim under a durations budget), M64 (CI head-SHA run as the final suite,
asserted by `pr_merge_readiness.py`), M65 (`scripts/run_mypy_public.sh` as the single source for the mypy file list).
