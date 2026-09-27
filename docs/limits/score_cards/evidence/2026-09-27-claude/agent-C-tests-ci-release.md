# Evidence Agent C: Test quantity, Test/CI truthfulness, Release governance (pymaxim v1.3.1 @ 7e695a58)

**Checkout and environment:** `/Users/dennyschaedig/Scripts/Maxim/.worktrees/rescore-v1.3.1`, detached HEAD `7e695a58cb596bafb165dec03286635826c38b59` (`git describe` = v1.3.1). `PYTHONPATH` was set to the worktree `src` and `maxim` imported from the worktree (`maxim.__version__` = 1.3.1). `MAXIM_DATA_HOME` was the scratchpad `C_home`. Live GitHub and PyPI state was read on 2026-09-27. Read-only: nothing was edited, committed or pushed, and no sims or LLMs were run.

---

## Axis 1: Test quantity

**Proposed grade: B+**

### Deciding findings
1. **VERIFIED: 12,213 tests collected.** Ran `python -m pytest --collect-only -q`. With `-m "not slow"` there are 12,169; with `-m slow` there are 44; with `-m requires_model_cache` there are 21. The CI unit-tests job at the release commit (run 36347881731) collected 12,188, deselected 44, and reported **12,089 passed, 55 skipped**. The MemoryHub gate then reported 25 passed.
2. **VERIFIED: the test-to-source ratio is about 0.84.** `tests/` has 196,345 lines in 575 `.py` files. `src/` has 233,361 lines in 524 `.py` files. `scripts/` has another 55,671 lines.
3. **VERIFIED: the largest modules have direct test importers.** Counts of test files that import each module: `agent_loop.py` (5,543 lines) 33, `decisions/nac.py` 89, `simulation/orchestrator.py` 14, `doctor/checks.py` 7, `lane_backends.py` 19, `leader_proxy.py` 6, `hippocampus.py` 52, `console/server.py` 14.
4. **VERIFIED: targeted coverage of the risky modules is uneven.** I ran pytest-cov locally over the 138 test files that mention hivemind, sandbox, survival_world or executor (3,579 passed).
   - `hivemind/`: 83–97% on the core files (bundle 93%, signing 97%, merge 93%, ingest 89%, store 87%). `substrate_client` is 60% and `oasis_cli` 75%.
   - `runtime/executor.py`: 84%.
   - `simulation/sandbox.py`: 76%.
   - `utils/sandbox_executor.py`: 67%.
   - **`tools/sandbox.py`: 27%** (183 of 267 statements missed).
   - `scripts/survival_world`: 0% for `exp58_run.py`, `exp58_offline_gates.py`, `r2_learned_bias.py`, `instrument_check.py` and `loop_tick_probe.py`. `r3_run.py` and `exp60_run.py` are about 30%; `water_trial.py` 83%, `scripts/survival_world/scripted_water.py` 92%.
5. **VERIFIED: CI neither measures nor gates coverage.** `.github/workflows/test.yml` has no `--cov` or `fail_under`. `pyproject.toml:298/307` has `[tool.coverage.*]` config, but nothing runs it.
6. **VERIFIED: slow and model-cache tests run in nightly lanes.** Both lanes (`test.yml:138` model-cache, `:247` slow) passed on 7e695a58 in dispatched run 36347890771.

### What holds
The suite is very large and has a real slow/nightly split. The wire-boundary code (hivemind signing, bundle, merge) is well covered.

### Deciding gap
No coverage number is enforced. The code-execution sandbox tool (`tools/sandbox.py`) is 27% covered by its related tests. Several experiment harnesses that produce gated evidence run with 0% coverage.

### To reach A−
- A coverage measurement in CI with a ratchet or floor that fails the build, at least per-package for `hivemind/`, `tools/sandbox.py`, `utils/sandbox_executor.py` and `runtime/executor.py`.
- `tools/sandbox.py` at 70% or higher under that gate.

---

## Axis 2: Test/CI truthfulness

**Proposed grade: B**

### Deciding findings
1. **VERIFIED: the extras lane has a positive control.** The required unit-tests job installs the `console` and `sign` extras from `pyproject` with no `|| echo` (`test.yml:397-410`). It runs with `--require-extras=console,sign` (`test.yml:428-432`). `tests/conftest.py:108-150` turns a skip whose reason names a required extra into a failure. That hook is tested in `tests/unit/test_require_extras_lane.py`.
2. **VERIFIED: one gap in the extras contract.** The contract matches skip-reason substrings (`"[sign] extra"`). `tests/unit/test_pre_freeze_fixes.py:373` calls `pytest.importorskip("cryptography")` with no reason, so that skip would evade `--require-extras`.
3. **VERIFIED: the required lane swallows one install failure.** `test.yml:395` (and `:266` in the slow lane) installs `faster-whisper … numpy scipy llama-cpp-python …` with `|| echo "Some optional deps unavailable on CI"`. A failed install silently turns those tests into skips. `sentence_transformers` (10 importorskips) is not installed in the required lane at all; it is covered only by the model-cache nightly.
4. **VERIFIED: a hermetic network guard is installed for the whole suite, in-process only.**
   - `tests/conftest.py:51-55` installs `tests/network_guard.py`.
   - `tests/unit/test_network_guard.py` has 9 tests.
   - At the release commit it printed "network guard: blocked 99 outbound call(s) (connect, dns)".
   - Blocked calls are counted, not failed, and the count is not ratcheted.
   - Its docstring states the scope: in-process only, subprocesses unguarded.
5. **VERIFIED: branch protection and the ruleset.**
   - `gh api …/branches/main/protection`: required checks are `unit-tests`, `lint` and `Release build (wheel contents + version)`, with `strict: true` and `enforce_admins: true`. Force-push is off, and 0 approving reviews are required.
   - Ruleset 13705164 ("main-protection") is active with no bypass actors. It enforces deletion, non-fast-forward and a CodeQL `code_scanning` gate.
   - The Python 3.10/3.11/3.13/3.14 compatibility jobs and aarch64 are not required. Those jobs only run install, `compileall`, an import smoke and `--help`; the test suite runs only on Linux 3.12.
6. **VERIFIED: the model-cache nightly was red for 23 consecutive scheduled runs.** `gh run list --event schedule` shows failure every day from 2026-09-04 to 2026-09-26 (32 of the last 38 scheduled runs failed). The failing job was `Model-cache tests (nightly)`: console tests skipped against the lane's skip allow-list (run 36244163200 log). Releases 1.2.1 (09-10) and 1.3.0 (09-19) were cut during that red window. The first green scheduled run is 2026-09-27 (36324374166), after the lane fix.
7. **VERIFIED: the required Release-build check is red on the tagged commit.** On 7e695a58 it failed in both the push run (36347881731) and the dispatch run (36347890771). The cause was `check_nightlies.py --only-when-releasing` refusing from inside the very nightly it waits for: "run 36345871460 tested 2f3c408b…, but main is at 7e695a58". Every other job in both runs passed, including the slow and model-cache nightly jobs in the dispatch run. This is self-reported in open issue #938 ("main reads red until the tag lands … a false alarm, and one that trains people to ignore it").
8. **VERIFIED: anti-vacuity mechanisms are present.**
   - 0 non-strict `xfail(` in `tests/`, and 2 with `strict=True`.
   - 0 `assert True` in `tests/`.
   - `lint_fix_touches_tests.py` runs in CI (`test.yml:1082`).
   - The model-cache lane has a skip allow-list.
   - `scripts/pr_merge_readiness.py` exists with its test.
9. **VERIFIED, cause unexplained: a flake under coverage instrumentation.** My 138-file subset under `--cov` had 6 failures (5 in `test_r3_run.py` with a `TypeError`, 1 in `test_water_trial_smoke.py`). The same subset without coverage passed 3,585/3,585, and the two files alone passed 15/15 with coverage. The cause is UNVERIFIED. CI does not run coverage, so this does not affect CI results, but the tests are sensitive to instrumentation or ordering.

### What holds
- Required checks cannot be bypassed: admins are enforced, the ruleset has no bypass actors, and CodeQL is required through the ruleset.
- The extras lane has a real positive control.
- The network guard is installed and tested.
- The tooling states its own blind spots honestly.

### Deciding gap
A red nightly was tolerated for 23 days across two releases. The tagged commit ships with a red required check. Issue #938 argues that red is a false alarm, but it is still red. There is also one silent-skip path in the required lane (`|| echo` on the optional-dependency install).

### To reach B+ / A−
- #938 fixed so that a nightly on a merged, untagged release commit goes fully green, with a guard test.
- The `|| echo` removed from the required lane's install, or replaced by a `--require-extras`-style check covering those dependencies.
- `test_pre_freeze_fixes.py:373` given the contract reason string.
- The network-guard attempt count ratcheted so it can only go down.
- At least one non-3.12 or macOS job running the fast suite as a required check.

---

## Axis 3: Release governance

**Proposed grade: B+**

### Deciding findings
1. **VERIFIED: the tag is signed and points at the release merge.** `git cat-file -p v1.3.1` shows an annotated tag on object 7e695a58. `git tag -v v1.3.1` gives "Good signature from Denny Schaedig" (EDDSA key 20A2897195567CB09825E294016F8AEDB2795CAF), made Sun Sep 27 16:08:48 2026 MDT, which is 22:08:48 UTC. 7e695a58 is the merge commit of release PR #937 (`gh pr list … head:release/1.3.1`: merged 2026-09-27T20:23:58Z).
2. **VERIFIED: PyPI, GitHub and the downloaded files agree.**
   - PyPI JSON: wheel sha256 `b53dd1c3…13009` and sdist sha256 `c5bd54ac…05f94`, uploaded 2026-09-27T22:06:30Z and 22:06:33Z. PyPI latest is 1.3.1.
   - `gh release view v1.3.1`: the same two assets with identical sha256 digests. Published 22:09:07Z, not a draft, not a prerelease.
   - `gh release download` plus `shasum -a 256` reproduces both hashes.
3. **VERIFIED: the published wheel matches the tag's source.** I unzipped the published wheel and ran `diff -rq` against the tag's `src/maxim`. The only difference is `console/ui_dist`. That directory is vendored at release and gitignored (`.gitignore:271`), as documented in `docs/publication_guide.md:239-262`. Its manifest records pulse commit 64f09e7 with `contract_version` 0.5.0, matching `CONSOLE_CONTRACT_VERSION = "0.5.0"` (`src/maxim/console/ui_bundle.py:80`).
4. **VERIFIED: the published artifacts pass the build audit.** `python scripts/audit_release_build.py --dist-dir <downloaded>` printed "release-build audit: clean" and exited 0.
5. **VERIFIED: dates and ordering match the documented procedure.** The CHANGELOG header is `## [1.3.1] - 2026-09-27` (`CHANGELOG.md:26`), which matches the UTC upload date. The sequence was upload at 22:06, tag at 22:08, Release at 22:09 (publish → tag → Release). The release notes link the CHANGELOG absolutely, but to `blob/main`, not `blob/v1.3.1`.
6. **VERIFIED: the version and tag audits pass.**
   - `scripts/lint_version_sync.py`: "OK — 1.3.1 in pyproject, __init__, CHANGELOG and the three version lines".
   - `scripts/audit_release_tags.py --check-releases`: "clean — 24 PyPI version(s) checked, 14 grandfathered", and "All released CHANGELOG versions have tags". The 14 grandfathered versions are historical: 0.2.1–1.0.0, and 1.0.9 with no assets.
   - The release-audit job runs on non-PR events and passed on 7e695a58.
7. **VERIFIED: the nightly gate was substantively met, but only by hand.** `scripts/check_nightlies.py` now reports "nightlies green at main 7e695a58 (run 36347890771)", and with `--only-when-releasing` it reports "v1.3.1 is already tagged". So the nightlies were green on the released commit before the tag. However, the in-CI form of that gate was red at the tag (Axis 2 finding 7), and the publish-time check is a manual step in the guide (`publication_guide.md:71-74`).
8. **VERIFIED: publishing is not mechanized or protected.**
   - `.github/workflows/` contains only `test.yml`: no publish or trusted-publishing workflow.
   - Upload, tagging and Release creation are manual per `publication_guide.md:306-430`.
   - The only ruleset targets the default branch, so there is no tag protection.
   - The release PR #937 passed every check it ran (`gh pr checks 937`: unit-tests, lint, Release build, CodeQL, compatibility jobs; nightly lanes skipped on PR). It has 0 reviews, and none are required.
9. **VERIFIED: tests cover the release tooling.** `tests/unit/test_check_nightlies.py` (12 tests), `test_audit_release_build.py`, `test_audit_release_tags.py`, `test_lint_version_sync.py`, `test_lint_unreleased_on_src_change.py` and `test_pr_merge_readiness.py` all exist.

### What holds
v1.3.1 verifies end to end:
- a signed tag on the merged release commit;
- byte-identical artifacts on PyPI and GitHub;
- wheel source identical to the tag;
- a clean audit of the published artifacts;
- consistent UTC dates;
- the version and tag audits enforced in CI.

### Deciding gap
The last mile is human-executed and unprotected: a manual twine upload, a manual tag, no tag ruleset, and a manual nightly check at publish time. The tagged commit was tagged while its required check read red (#938). Two earlier releases shipped over red nightlies; that history is before this tag, but it shows what the manual procedure allowed.

### To reach A−
- Publishing moved into a CI workflow triggered by the signed tag: PyPI trusted publishing, a hard-required `check_nightlies`, a build audit on the exact uploaded files, and automatic Release creation with those assets.
- A tag ruleset restricting `v*` creation and deletion.
- #938 closed so the release commit is green on every required context before the tag.

---

## Contacts
- `.github/workflows/test.yml:1083-1085`, in a comment on `lint_fix_touches_tests`: "the subject that squash-merges onto main, and the one the score card counts". This refers to a score card but gives no grade.
- `scripts/audit_release_tags.py --check-releases` output for 1.0.9: "Named by the 2026-08-27 score card". This refers to an earlier card but gives no grade.
- v1.3.1 GitHub Release body, first paragraph: "It fixes what the v1.3.0 re-score found". This refers to the earlier re-score but gives no grade; I did not read further into that section.

None of these stated or implied a grade, and none moved mine.

## Commands run
`git rev-parse`/`describe`/`cat-file -p v1.3.1`/`tag -v v1.3.1`/`log`/`check-ignore`; `grep`/`sed` on `.github/workflows/test.yml`, `tests/conftest.py`, `tests/network_guard.py`, `pyproject.toml`, `docs/publication_guide.md`, the `CHANGELOG.md` headers only (`grep '^## \['`) and the `tests/` tree; `python -m pytest --collect-only -q` (all, not slow, slow, requires_model_cache); `pytest` with `--cov` over 138 targeted test files, the same files without coverage, and the two flaky files alone; `wc`/`find` LOC counts; `gh api` for branch protection, rulesets (list and 13705164), tags and the job log; `gh run list` (all events and schedule); `gh run view` (jobs and `--log-failed` for 36347881731, 36347890771, 36345871460, 36324374166, 36244163200, 35443082104, 34859502576); `gh pr list`/`checks`/`view` for 937; `gh issue list`/`view` for 938; `gh release view`/`download` for v1.3.1; `curl` against the PyPI JSON; `shasum -a 256`; `unzip` plus `diff -rq` of the wheel against `src`; `python scripts/check_nightlies.py` (plain and `--only-when-releasing`); `scripts/lint_version_sync.py`; `scripts/audit_release_tags.py --check-releases`; `scripts/audit_release_build.py --dist-dir <downloaded>`.
