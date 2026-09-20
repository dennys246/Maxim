# Evidence Agent C — Test quantity, Test/CI truthfulness, Release governance

Scope: tag `v1.3.0` @ `ad541dd7` (worktree `.worktrees/rescore-v1.3.0`), plus live PyPI/GitHub/CI state on 2026-09-19.

| Axis | Proposed grade |
|---|---|
| Test quantity | **A−** |
| Test/CI truthfulness | **C+** |
| Release governance | **B+** |

---

## 1. Test quantity: A−

**Findings**

1. **Collection count.** `python -m pytest --collect-only -q -o addopts=""` at the tag collects **10,973 tests**. By directory: unit 9,998 · integration 368 · substrate 367 · behavioral 173 · learning 35 · experiments 18 · benchmarks 9 · performance 5. Markers: `-m slow` gives 42, `-m requires_model_cache` gives 21. There are 491 `test_*.py` files.
2. **Test-to-code ratio.** `tests/**/*.py` has 181,060 lines and `src/**/*.py` has 226,516 lines (520 files), a ratio of about **0.80**. The survival world is not counted in `src/`: it lives in `scripts/survival_world/` (9,643 py lines).
3. **The new 1.3 surfaces are covered in the fast suite.** Collected counts by filename: test_exp60_run 30, test_exp60_water_classroom 33, test_exp61_run 27, test_l11_geometry_probe 23, test_r3_run 10 (one of them `slow`), test_r3_pilot 1, test_water_trial_lethal 5, test_water_trial_smoke 5. That is about 134 survival-world tests. Other areas: hive* 270 tests / 7 files, NAc 143 / 5 files, EC (`test_ec_trace_activations`) 30, console 364+ / 16 files, minecraft 37.
4. **Local run.** `python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py` exited **0 in 615 s**: 10,867 passed, 39 skipped, 42 deselected. My machine has fastapi and cryptography installed; CI does not (see Truthfulness F1).
5. **Gaps that keep this below A.** The survival-world harness (about 9.6k LOC) has one pilot test and 10 r3_run tests. The count that CI actually executes is smaller than the collected count: 10,520 passed in the tag's main push run, versus 10,867 locally.

**Enforced vs documented.** Counts are measured, not taken from docs. Nothing enforces a coverage floor: `[tool.coverage.run]` is configured, but no `--cov-fail-under` appears in CI.

**Raise one step (to A):** get the survival-world harness to unit coverage comparable to its size, and install the console and sign extras in a gating lane, so that the executed count matches the collected count.

---

## 2. Test/CI truthfulness: C+

**Findings**

1. **The console suite and the signed-bundle suite run in NO CI lane.** `.github/workflows/test.yml` installs `pip install --no-deps -e .` plus a fixed list (faster-whisper … pyyaml), with no fastapi, uvicorn or cryptography. `grep -n "cryptography\|fastapi\|\[console\]\|\[sign\]" .github/workflows/test.yml` returns nothing.
   - The `unit-tests` log of main push run 35466702773 (the `ad541dd7` commit) shows `collected 10638 items / … 12 skipped` and `10520 passed, 88 skipped`.
   - Skip reasons include 13 console modules (`test_console_{agent_id,auth,device_handoff,event_seam,handle_workspace,identity,launcher_seams,pairing,sandbox,server,talk,trust_guard,ui_bundle}` → "requires the `console` extra (fastapi/uvicorn)").
   - They also include all of `test_hivemind_signing.py` (14 tests), `test_hive_pull_e2e.py`, `test_oasis_exchange_e2e.py`, `test_oasis_store`, and `test_hive_cli:414` → "signed bundles need the [sign] extra (cryptography)".
   - Both released headlines depend on these paths: 1.2.1 is the spoken-code console pairing release, and 1.2/1.3 claim a "shipped signed-bundle path". Their tests execute only on developer machines. The model-cache lane's allow-list (`scripts/check_model_cache_lane.py` `ALLOWED_MODULE_SKIPS`) records this as "the console extra is not installed in this lane" instead of fixing it.
2. **The nightly model-cache lane has been red for 16 consecutive nights (2026-09-04 → 09-19), and 1.3.0 shipped through it.**
   - `gh run list --workflow Tests --event schedule`: 25 of the last 30 scheduled runs failed. The only green stretch is 08-30 → 09-03.
   - Every failed night has the same single failing job, "Model-cache tests (nightly)": `17 passed, 16 skipped … FAIL: skipped tests not on the explicit allow-list` (the console modules added later, `test_console_agent_id`, `_auth`, `_device_handoff`, … "collection skipped").
   - The lane's vacuity guard works as designed and goes red. Nobody acted on it for 16 days, and the release PR #769 shows the nightly lanes as SKIPPED (`gh pr view 769 --json statusCheckRollup`). No step in the release procedure reads nightly status: `grep -i nightly docs/publication_guide.md` finds only the release-audit mention.
3. **The slow lane is green but mostly vacuous.**
   - Last night (run 35443082104): `slow lane: 54 collected, 42 skipped, 12 executed`.
   - 24 of the skips are "sentence-transformers not installed", covering the substrate P1/P2/P5 validation sweeps. 12 are "collection skipped" and 4 are model-cache.
   - `scripts/check_slow_lane.py` only asserts `executed > 0` and says so in its docstring ("Skips are reported but not themselves fatal … no curated allow-list here yet").
   - Those substrate sweeps are `slow` but not `requires_model_cache`. The model-cache lane (which does install `[semantic]`) never selects them, and the slow lane never installs their dependencies, so in practice they run nowhere in CI.
4. **Required-check presence is enforced by GitHub, but it is thin and bypassable.**
   - Branch protection on main (`gh api …/branches/main/protection`): required contexts are `unit-tests` and `lint` only. `enforce_admins: false`, `required_approving_review_count: 0`, `strict: false`.
   - Ruleset `main-protection` (13705164): no deletion, no non-fast-forward, and a CodeQL code-scanning gate (`alerts_threshold: all`). It has a bypass actor (`RepositoryRole 5`, `bypass_mode: always`).
   - `release-build` is not a required context, although CLAUDE.md says the list must contain it.
   - `scripts/pr_merge_readiness.py` does check presence (`REQUIRED_CHECK_PATTERNS = ("unit-tests","lint","release build","codeql")`), but it is a manual CLI and nothing invokes it in CI. GitHub's native "required context missing = pending" covers presence for the two required contexts only.
5. **Positives that hold (and keep this out of D).**
   - The PR gate runs the documented commands verbatim, including the separate MemoryHub step (`test.yml` "Run required fast suite", "MemoryHub integration gate").
   - mypy is pinned and runs on the CLAUDE.md-listed files.
   - About 20 invariant lints run in the `lint` job with positive controls: silent-swallow ratchet, prereg-precedes-data, `[Unreleased]`-on-src-change, harness provenance, fix-ships-with-test, and others.
   - The PR trigger includes `edited`, so retargeting a PR's base branch fires CI.
   - The main push run for `ad541dd7` is fully green, and the release PR's first commit (2178e004) was red on `unit-tests` and was fixed before merge (68386e5b). The gate bit.
   - xfails: `grep -rn xfail tests/` finds only 3 hits, all in comments or docstrings describing past `strict=True` gates. No live xfail, strict or non-strict, remains.
6. **Hermeticity is partial.**
   - `tests/conftest.py` redirects HOME, XDG and HF_HOME, forces `HF_HUB_OFFLINE=1`, gates model tests behind `MAXIM_RUN_MODEL_TESTS` (`pytest_collection_modifyitems`), and has 48 autouse fixtures, most of them env scrubs.
   - There is no socket or network block: `grep socket tests/conftest.py` finds nothing.
   - My local run leaked background threads. The terminal kept printing "Orchestrator planning next probe… (566s)" spinners after the session summary (39 such frames in the log), so some tests leave live threads behind.
7. **The R3 test move to `slow` was justified, and the test does run.** Commit d1aa70a4 moved `test_offline_campaign_apparatus_and_one_event_per_in_process_arm` to the slow lane. The commit message and the in-file comment (`tests/unit/test_r3_run.py:171-179`) explain why: a real-time 0.15 s stale-sample rule under one GIL is not holdable on shared runners (a 0.348 s stall on PR #761). The move is in the 1.3.0 CHANGELOG, the rule's arithmetic stays in the fast suite, and the nightly slow log shows `tests/unit/test_r3_run.py .` passing. This is not a hidden demotion.

**Enforced vs documented.**
- Enforced: the fast suite plus MemoryHub on every PR; `unit-tests` and `lint` required; CodeQL ruleset; lint ratchets.
- Documented or claimed but not enforced: nightly lanes feeding any gate; the presence of `release-build` as a required check; the readiness script (manual only); console and signing coverage in CI.

**Raise one step (to B−):**
- Install `[console,sign]` (plus sentence-transformers for the slow lane) in CI, so those suites execute somewhere automatically.
- Get the model-cache lane back to green, and make a red nightly block the release, for example via a release-PR check or a preflight in `audit_release_build.py`.
- Replace the slow lane's `executed > 0` assert with an allow-list like the model-cache lane's.
- Add `release-build` to the required contexts.

---

## 3. Release governance: B+

**Findings**

1. **1.3.0 is internally consistent and matches PyPI, and I verified this independently.**
   - `pyproject.toml:7` `version = "1.3.0"` and `src/maxim/__init__.py:16` agree. The CHANGELOG has `## [1.3.0] - 2026-09-19` above an empty `## [Unreleased]`. `python scripts/lint_version_sync.py` gives "OK — 1.3.0 …", exit 0.
   - PyPI uploaded 1.3.0 at 2026-09-19T20:24:26Z (UTC date matches the CHANGELOG).
   - The GitHub Release `v1.3.0` carries the wheel (`sha256:84be7724…`) and the sdist (`7154644e…`). Both digests equal PyPI's.
   - I downloaded the PyPI sdist myself (`sha256 7154644e…`) and byte-compared all 520 `src/**/*.py` against `git show ad541dd7:<path>`: **0 differ**. The tag is on the published commit.
   - `git verify-tag v1.3.0`: "Good signature" (EdDSA key 2089…5CAF), annotated tag, tagged 20:24:53Z, 27 s after upload. v1.1.0 through v1.3.0 are all annotated and signed.
   - The 1.3.0 Release body uses 5 absolute `https://github.com` links and no relative links.
2. **The release audit is mechanized and clean for the modern series.**
   - `python3 scripts/audit_release_tags.py --check-releases` exited 0: "release-object audit: clean — 23 PyPI version(s) checked, **14 grandfathered**".
   - The 14 grandfathered versions are 0.2.1–0.9.1, 1.0.0 and 1.0.9. Each has an explicit reason in `GRANDFATHERED_RELEASES` (`scripts/audit_release_tags.py:59`), and each is re-printed as still failing on every run: missing tags for 0.6.0–0.8.1, missing Release objects, or Releases with 0 assets.
   - Every version from 1.1.0rc1 onward passes all checks.
   - It runs in CI (`release-audit` job) on push to main and nightly, **but not on PRs**, so it detects drift after the fact instead of blocking it.
   - The `release-build` job checks wheel contents and version on every PR, but against a stubbed console bundle, and it is not a required check.
3. **"Main ahead of PyPI" is enforced in CI.** `lint` runs `lint_version_sync.py`, `lint_unreleased_on_src_change.py` ("[Unreleased] entry on a src change") and `lint_unreleased_declared.py` ("A post-tag src/ commit declares itself"), each with positive-control tests. The two unreleased lints overlap, which is minor redundancy rather than a gap.
4. **Not mechanized:**
   - **UTC-date rule.** `audit_release_tags.py` never compares CHANGELOG dates with PyPI `upload_time`. The dates I checked by hand agree (1.1.2–1.3.0).
   - **Tag signature and tag = built commit.** No script checks either; I did it by hand above.
   - **Structure-or-time rule for gating results** (`docs/publication_guide.md:312-328`). The guide itself says clauses (a) and (b) are "mechanically checkable", yet no script checks them. `lint_prereg_precedes_data.py` covers only prereg-before-data. Clause (c), the different reader, is admitted to be unmechanized.
   - **Application to 1.3.0.** Structurally the rule was met: Exp 61 data landed in #748 and the interpretation in #749; the Exp 56 re-baseline data landed in #766 (12:10) and the outcome in #768 (12:45). The release (#769, 14:13 local) followed about 90 minutes later. Whether (c) happened is unverifiable from the repo.
   - **Publishing is manual.** A local `twine upload` is used; there is no trusted-publishing workflow, so artifacts are not CI-built. The sdist-equals-tag check above shows it was done correctly this time.
5. **Post-release correction was only partly visible to readers.**
   - The 1.2.1 overclaim ("completes the spoken-code device-pairing loop end to end") is corrected in the CHANGELOG's 1.3.0 section ("### Correction to 1.2.1") and in the release notes.
   - The **v1.2.1 GitHub Release body still says "end to end"** with no correction or link (`gh release view v1.2.1 --json body | grep -i correction` finds nothing). `docs/announcements/release_1_2_1.md` is likewise uncorrected, and PyPI 1.2.1 cannot be changed.
   - Readers who land on the 1.2.1 Release page do not see the correction.
6. **Historical record oddity (grandfathered, not new).** CHANGELOG sections and tags exist for 0.9.2, 0.9.3 and 1.0.1–1.0.8, but PyPI never served those versions. The publication guide records these tags as "reconstructed … deliberately have none" (`docs/publication_guide.md:388`).

**Enforced vs documented.**
- Enforced: version sync, the `[Unreleased]` discipline, the Release object with sha256-matched assets and absolute links (post-merge and nightly), wheel contents and version (per PR, not required), prereg-before-data.
- Documented only: the UTC date, tag signing, tag = published commit, structure-or-time, the different-reader rule, correcting older Release pages.

**Raise one step (to A−):**
- Mechanize the UTC-date comparison, tag-commit = sdist content, and tag signature checks in `audit_release_tags.py`.
- Mechanize structure-or-time clauses (a) and (b) for any gating result the CHANGELOG cites.
- Append the 1.2.1 correction to the v1.2.1 Release body.
- Make `release-build` required, and run the Release-object preflight for the release PR itself.

---

## Verified vs inferred

**Verified by running commands:**
- Collection counts, the local fast-suite result (exit 0, 615 s, 10,867 passed, 39 skipped).
- CI job lists and logs (main push 35466702773; nightly 35443082104, 35227493666, 34759515545, 34032279820, 33874318826, 33757558573) and the 30-night schedule history.
- Skip reasons in the CI logs, branch protection JSON, and ruleset JSON.
- `lint_version_sync.py` (exit 0) and `audit_release_tags.py --check-releases` (exit 0, 14 grandfathered).
- PyPI JSON, the GitHub Release asset digests, the sdist-to-tag byte compare, `git verify-tag`, the v1.2.1 Release body, and the tag signature status for v1.1.0 through v1.3.0.

**Inferred:**
- That no admin bypass was used on recent merges. Every first-parent main commit since 08-20 I inspected is a PR merge, but I did not audit individual merge events.
- That the leaked spinner threads come from simulation/orchestrator tests. I did not identify the specific tests.
- That clause (c) (a different reader) did or did not happen for the Exp 56 RB-1 interpretation. Not verifiable from the repo.
- That the console and signing tests pass. They pass locally, but nothing in CI demonstrates it.

## Independence

I did not read, grep or recover `docs/limits/score_cards/`, `docs/plans/burndown_1_3.md`, the "Scorecard → roadmap reconciliation" section, or anything under `~/.claude/`. I ran no `git log -p` or `git show` on those paths.

**Incidental contacts:**
1. Two in-repo source comments name an earlier score card's requirements, with no letter grades:
   - `scripts/check_slow_lane.py` docstring: "Score card 2026-08-27, Test/CI-truthfulness 'Upgrade to B': …". The phrase implies an earlier grade below B on this axis.
   - `test.yml` lint step comment: "Maintainability score-card mechanization".

   I saw them while reading CI configuration that is in scope, and I did not use them. My grades rest on the evidence above.
2. The session's injected project context contained auto-memory text about the project's history. I did not open anything under `~/.claude/`, and it contained no letter grades that I relied on.
