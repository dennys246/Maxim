# Roadmap 1.3.x — the hardening line: 1.3.1 (fixes + guards) → 1.3.2 (decomposition)

> **1.3.1 SHIPPED 2026-09-27 as "Hardening"** — every **[1.3.1]** item below built and merged with its
> guard (PR numbers on each row), plus the ledger's 1.3.1 trigger walk (#934) and the Exp 10 re-run
> (#936, MAINTAINED narrow; D13 shortened every run, #935). Items marked **[→ 1.3.2]** carry to 1.3.2.
> Release notes: [release_1_3_1.md](../announcements/release_1_3_1.md).

**Drafted 2026-09-19**, the day 1.3.0 "Oasis-2" published, from the v1.3.0 blind re-score
([docs/limits/score_cards/2026-09-19-claude.md](../limits/score_cards/2026-09-19-claude.md) and its
Codex twin) plus the release-day different-reader pass. **Owner decision the same day:** ship two
patch releases before 1.4's experiments — 1.3.1 for the defects and the enforcement gaps, 1.3.2 for
the `agent_loop` decomposition and the typing scope — rather than carrying either into 1.4.

**The rule this line runs on:** the score card credits only what is ENFORCED. So every item here
ships with its guard in the same PR — a test, a lint, a ratchet, a CI lane or a required check.
An item whose guard cannot be named does not belong in these releases. Both releases are
**infrastructure only, no behavioural claim**; the 1.1.2 "Decomposition" release is the precedent.

**Why before 1.4, not during** (the decomposition half): 1.4's Phase 0 builds the trajectory
instrument and the per-step credit read directly on `agent_loop.py`; decomposing afterwards means
building that instrument twice. And a loop refactor fires the Exp 60 / Exp 61 re-run triggers by
their letter — discharging them is cheapest now, while the rig is set up and the classroom is built,
and before 1.4's campaigns exist. Refactoring *while* may-fail experiments run is what the
divergence rule warns against: a null then confounds mechanism with refactor.

---

> **RE-SCOPED 2026-09-26 (owner).** 1.3.1 stopped being "the hardening release, all of it": by the
> 2026-09-26 audit only 2 of its ~24 items had shipped, while `main` ran 250+ commits ahead of PyPI —
> including security fixes (#796, #821–#825) and the public release format v2 (public_oasis item 7).
> Holding all of that behind every ratchet would ship users nothing for weeks and grow the release
> notes past reviewing. So:
>
> - **1.3.1 = what users need now, each item still with its guard:** the security, public-API
>   correctness and release-integrity items below marked **[1.3.1]**; the public format freeze and the
>   two ingest defects it surfaced (§"Added by the re-scope"); and everything already on `main` since
>   1.3.0. No NEW behavioural claim rides it (Exp 62 stays out until its different-reader pass is
>   recorded); the behaviour changes it carries (security fixes, the v2 format, the scrub) are stated
>   in its notes.
> - **1.3.2 = the enforcement ratchets + the decomposition:** the items marked **[→ 1.3.2]** join the
>   `agent_loop` decomposition (§"Carried from 1.3.1").
>
> Order: #914 → the public format freeze → #913 → the [1.3.1] items (the security cluster and the
> nightly lane first) → the 1.3.1 release PR → 1.3.2.

> **The quality burndown is merged in (2026-09-20).** `quality_burndown.md` was a second,
> mutually-unaware list over this same territory; its live remainder now lives in §1.3.1 "Carried in
> from the quality burndown" below, each item re-verified against `docs/bugs/README.md` that day and
> given the guard this line's rule requires. The old file is archived at
> [archive/quality_burndown.md](archive/quality_burndown.md) as the record of Batches 0–2.

## 1.3.1 — the defects and the enforcement gaps

Grouped by the axis each item lifts; the "to reach" conditions come from the card.

### Test/CI truthfulness (C+ → B−) — the biggest lever

| item | guard that makes it count |
|---|---|
| **[shipped #894]** **A gating lane that installs the `console` extra and the crypto dependency**, so the console, bundle-signing, hive-pull and Oasis-exchange tests run on every PR. Today **no lane installs fastapi or cryptography**, so those tests are skipped everywhere — and the 1.2 and 1.3 headlines both travel the signed-bundle path. | the lane itself, required in branch protection; a positive control asserting the previously-skipped modules now execute (count > 0), so the lane cannot go quietly vacuous. *Built 2026-09-25: `unit-tests` (already required) installs `console` + `sign` from `pyproject.toml`; `--require-extras=console,sign` fails any skip for a missing required extra (`tests/conftest.py`, pinned by `tests/unit/test_require_extras_lane.py`).* |
| **[1.3.1 — SHIPPED (#926); first green scheduled run 2026-09-27]** **The nightly model-cache lane green** (red 16 nights running, 25 of the last 30 scheduled runs; new console modules missing from its skip allow-list). Fix by making a missing module FAIL rather than by extending the allow-list. | the lane's own red/green + a check that the allow-list cannot grow silently *(Built: the lane installs the console + sign extras and runs with `--require-extras`; `ALLOWED_MODULE_SKIPS` is gone, and `ALLOWED_SKIPS` is ratcheted in `tests/unit/test_model_cache_names.py` (since #1117: the model-cache roster's `allowed_skips`, pinned in `tests/unit/test_lane_rosters.py`).)* |
| **[1.3.2 — MERGED (#1122, #1139), see item 7]** **A slow lane that runs**: install `sentence-transformers` so the 24 substrate sweeps execute; replace "executed > 0" with an exact roster (owner decision 2026-10-04; it was a pinned minimum). | `scripts/check_lane_roster.py --lane slow` asserting the exact roster |
| **[1.3.1 — SHIPPED (#928)]** **Network blocked in tests** (hermeticity is HOME/HF isolation + ~48 env scrubs today, with no block). | a conftest socket guard + a test that asserts an outbound call raises |
| **[1.3.2 — MERGED (#1130); settings verified 2026-10-06, drift-checked nightly by `scripts/check_repo_settings.py`, token live]** **`release-build` required**, `enforce_admins` on, and a required-checks-present gate (verified 2026-10-06: all three contexts required, strict, admins enforced, the ruleset's bypass list empty; GitHub itself refuses a merge while a required check is absent). | branch-protection settings — **owner action**; guard: `scripts/check_repo_settings.py` (nightly, pinned to `scripts/repo_settings_expected.json`) + `tests/unit/test_check_repo_settings.py` |
| **[1.3.1 — SHIPPED (#926)]** **The release procedure reads the nightlies**: refuse to publish while a nightly lane is red. | a step in `audit_release_build.py` or the release PR checklist, mechanized *(Built as `scripts/check_nightlies.py`, not inside the offline `audit_release_build.py`: it needs the network. Runs `--only-when-releasing` in the `release-build` job; blocking needs `release-build` required, row above. Guard: `tests/unit/test_check_nightlies.py`.)* |

### Runtime correctness (C+ → B−)

| item | guard |
|---|---|
| **[1.3.1 — SHIPPED (#930)]** **`AgentInstance.export_memories()` always reports 0** — it reads `self.hippocampus.memories`, which does not exist, and an `except Exception` turns the error into `0`; `AgentPool.export_all_memories` propagates it; the documented example in `docs/user/python-api.md` prints "0 memories" beside a hippocampus holding one. | a test asserting the COUNT (both current tests are vacuous: one checks the key exists, the other that it is a dict) |
| **[1.3.1 — SHIPPED (#930)]** **`create.agent`'s docstring example crashes** — `capture(perception="dark cave ahead")` raises `AttributeError`; `capture` does not validate its argument. | argument validation + a doctest-style test that runs the documented example |
| **[1.3.1 — SHIPPED (#930)]** **`maxim.diagnose()` and `maxim doctor --json` disagree** (diagnose reports all-passed while the CLI exits 1 on a probe diagnose never runs). | a test pinning one probe set for both entry points |
| **[1.3.2 — MERGED (#1099)]** **The silent-default swallow shape** — a handler that ASSIGNS a fallback instead of `pass`, which is what hid `export_memories` and which `lint_no_silent_swallows.py` cannot see. | extend the lint to that shape, as a ratchet on today's count (430 bare sites, 1,788 `except Exception` total) *(Built 2026-10-04 in 1.3.2, merged as #1099: check 5 of `scripts/lint_no_silent_swallows.py`, 298 silent-default sites at the build, sharing one per-function pool with check 2's 415; checks 2, 3 and 5 credit verbatim moves and recorded edited moves (`scripts/swallow_moves.json`), so the decomposition slices do not trip them.)* |

### Maintainability (C → C+, the cheap half)

| item | guard |
|---|---|
| **[→ 1.3.2 — SUPERSEDED by item 6]** ~~**Extend the function-length ratchet to every function over 300 lines** (18 today, one of 921; the ratchet covers 3). Pin at current length, shrink-only.~~ Item 6 below replaces it with one ratchet at **200** lines (53 functions pinned, 19 of them over 300, measured 2026-10-04). **Built 2026-10-04, PR pending.** | `scripts/lint_function_length.py` + `src/maxim/utils/function_length_baseline.json` + `tests/unit/test_lint_function_length.py` |
| **[→ 1.3.2]** **A repo-wide mypy error-count ratchet** starting at today's measurement (1,050 errors in 141 files over all of `src/maxim` — a different, earlier invocation, superseded: the ratchet's own invocation is the only count that binds; CI's typed set stays at 18 files). **Built 2026-10-04** as a per-file ratchet against the merge-base (item 4 below; 1,096 cold at the build, 1,075 at `v1.3.1` the same way). | `scripts/lint_mypy_ratchet.py` (lint job) + `tests/unit/test_lint_mypy_ratchet.py` |

### Research integrity + documentation honesty (both B+, cheap items)

- **[1.3.1 — SHIPPED (#931)]** **Extend `lint_prereg_precedes_data.py` to `docs/experiments/*_prereg.md`** — it reads only
  `protocols/*preregistration*.md`, so **none of 1.3.0's own experiments** (Exp 60, Exp 61, R3) are
  covered; their ordering was verified by hand. Guard: the lint, with the three 1.3 experiments in
  its governed set.
- **[shipped #773]** **Point the 1.2.1 surfaces at the correction** — the 1.2.1 CHANGELOG entry, `release_1_2_1.md`
  and the v1.2.1 GitHub Release body still say "end to end" with no pointer to the 1.3.0 correction.
- **[1.3.1 — SHIPPED (#931)]** **Rewrite README.md for what ships** — it is the PyPI description, still calls substrate-driven
  action selection a "post-1.0 research direction", never mentions the 1.2/1.3 results, and says 16
  extras where there are 21.
- **[1.3.1 — SHIPPED (#931)]** **Three smaller errors:** the release notes' `maxim substrate invalidate --drop-geometry`
  invocation is incomplete (needs `--session`, `--modality`, a tag value, `--apply`); the ledger's
  Exp 60 freeze hash names the wrong PR merge; the Exp 56 row calls amendments 3–4 pre-confirmatory
  while their headers say POST-DATA.

### Carried in from the quality burndown (merged 2026-09-20)

Re-verified against [docs/bugs/README.md](../bugs/README.md) on the merge date; anything already
closed was dropped rather than copied (**D19 was FIXED** — the architecture-audit gate exists — and
the burndown still listed it; **D84 → #796** is fixed by PR #804). Same rule as every item above:
it ships with its guard or it does not ship.

| item | guard that makes it count |
|---|---|
| **[1.3.1 — SHIPPED (#930)]** **D40 remainder (was N1)** — thread `prompt_handler` through `start_simulation_mode` (the consumer, `bootstrap.build_tool_registry(prompt_handler=…)`, already exists; only the passthrough is missing). `npc_model` stays a loud `NotImplementedError` until party-mode NPC agents exist — that half is a mechanism, not a defect. | extend `tests/unit/test_api_expansion.py::TestCampaignParametersAreThreadedOrRejected`: a passed handler is the one the run's prompts reach |
| **[1.3.1 — SHIPPED (#930)]** **D32** — load the foundational preamble from `CONSTITUTION.md` as package data (pip users get an empty preamble today) | a drift test: packaged copy == repo-root `CONSTITUTION.md`, and a wheel-install test that the preamble is non-empty |
| **[→ 1.3.2]** **D49** — benchmark honesty: apply-or-delete `weight`, fix the running half-mean, drop-or-ship the missing tier2/tier3 suite files (`simulation/benchmark.py`). **2026-10-07:** `weight`, `metrics` and true means done (Session C, owner: honour the format); dropping `tier2`/`tier3` from `--benchmark`'s choices goes with the `cli.py` decomposition slice (Session A) | a unit test per promise: a weighted suite's aggregate moves with `weight` (or the key is rejected), and every suite file the format names loads |
| **[→ 1.3.2]** **D46 + D50** — delete the dead percept-transport reference (`simulation/sources.py`); warn on the inert `party_mode` / `choice_resolution` keys in `load_campaign` and drop the dead schema field | a test that loading a campaign carrying either key WARNS once |
| **[1.3.2 — MERGED (#1130)]** **D63** — a PR against a non-`main` base runs no required checks (Tests now runs on PRs to any base; GitHub's required checks block the final merge to `main` while a check is absent; the settings that make that so are drift-checked nightly, `scripts/check_repo_settings.py`) | guard: `scripts/check_repo_settings.py` (nightly live half + `--static` in the lint job) + `tests/unit/test_check_repo_settings.py`; the any-base trigger is pinned by `tests/unit/test_ci_workflow_shape.py::test_tests_run_on_prs_to_any_base` |
| **[→ 1.3.2]** **Fail-loud Stage 3** — narrow the measurement-path swallows; green-lit since Stage 2 measured **zero** firings ([deferred/measurement_path_fail_loud.md](deferred/measurement_path_fail_loud.md)). Must not land mid-walk on a branch a graduation run reads from. | `scripts/lint_no_silent_swallows.py`'s zero-total set grows to cover each narrowed file |
| **[1.3.1 — SHIPPED (#920, #923, #925)]** **The security cluster (register O11)** — the sandbox ([#800](https://github.com/dennys246/Maxim/issues/800) Python scripts never run, [#801](https://github.com/dennys246/Maxim/issues/801) raw-prefix containment, [#802](https://github.com/dennys246/Maxim/issues/802) the path runs instead of the approved content) and mode/approval ([#828](https://github.com/dennys246/Maxim/issues/828) any audio can say "maxim singularity" — highest, [#827](https://github.com/dennys246/Maxim/issues/827) autonomy approvals never shown or resolved, [#826](https://github.com/dennys246/Maxim/issues/826) suspected prompt-only tool lists), plus [#824](https://github.com/dennys246/Maxim/issues/824) DNS rebinding. Widened from the sandbox trio on the re-scope: a release that ships the security fixes on `main` does not ship knowing these. | each issue's own red gate |
| **[→ 1.3.2]** **L8 record-stamping** (stamp model / endpoint / n_ctx / quantization on every run record) — Exp 44b's prerequisite; status **not re-verified** on the merge date, check before starting | a test that a run record without those fields is refused by its writer |

Already covered above, so not duplicated: `mypy` scope (the ratchet in this section), god-function
decomposition (§1.3.2). Behavioral-suite thickening for Exp 52/53b/56 has no nameable guard as
stated and is left out by this line's rule, not forgotten.

### Added by the re-scope (2026-09-26)

| item | guard |
|---|---|
| **[shipped #917]** **#914** — `merge.rekey_nac_state` folds colliding donor clusters by OVERWRITE and keeps the inherent (safety-floor) marker when any source was inherent, so a learned bias can overwrite an inherent one and stay decay-exempt. One shared "fold rows + markers" helper for the export scrub (fixed in #915) and ingest; check the Exp 56/61 evidence for collapsed situations. | a test that a learned row never inherits the marker and colliding biases mean-fold, at BOTH seams |
| **[shipped #918]** **The public format freeze** (public_oasis Phase 0 item 2, part 2; record: [public_format_freeze.md](public_format_freeze.md)) — the freeze record (what "public format 1" promises, the change rule, the compatibility horizon) after part 1's pre-freeze hardening (#915). | `tests/unit/test_public_format_freeze.py`: checked-in fixtures every build must verify, ingest AND recompose byte-for-byte |
| **[1.3.1 — SHIPPED (#919)]** **#913** — `merge._merge_link_lists` indexes the receiver's OWN links by outcome signature alone, so context-distinct links overwrite each other on every ingest (a real state: 607 → 443 links with an EMPTY donor). A known data-loss bug does not ship in the release that publishes the format. Includes an audit of whether any earned result ingested through the lossy path. | a test that an empty-donor merge is the identity on the receiver's links |

### Not in 1.3.1

The items marked [→ 1.3.2], the decomposition, full mypy coverage, and any new mechanism.

---

## 1.3.2 — the decomposition

### Carried from 1.3.1 (re-scoped 2026-09-26)

The enforcement ratchets and small defects marked **[→ 1.3.2]** in §1.3.1 — the slow lane, the
required-checks gate, the silent-default swallow lint, the function-length and mypy ratchets, D49,
D46 + D50, D63, fail-loud Stage 3 and L8 — land here beside the decomposition, each with the guard
named in its row. The ratchets go FIRST among the enforcement items (after #951 and #935, which unblock CI truth and live re-runs — §Sequence): they pin the ceilings the decomposition then lowers.

### From the v1.3.1 score cards (added 2026-09-27)

Both blind cards ([Claude](../limits/score_cards/2026-09-27-claude.md), [Codex](../limits/score_cards/2026-09-27-codex.md))
found the same facts and differed on weight; the plan is set against the **lower** readings. Every item
ships with the guard named beside it, proven by deleting its mechanism. Order matters: 1 unblocks every
live re-run, 2–4 are data-safety and silent-failure fixes, 5–8 are the checks the cards credit.

1. **D13 short runs ([#935](https://github.com/dennys246/Maxim/issues/935))** — first, because every live
   heartbeat re-run now stops after 1–3 turns. Guard: an offline replay test of the narrator's follow-up
   input (August commit vs `main`), and a committed Sim-Short record that reaches its turn cap.
2. **Evidence provenance + the typed-abort gate** (mechanization backlog M1). Guard: the ledger lint.
   **Stamps shipped 2026-09-29** (`report.json`'s `provenance` + `ts`), with an owner-approved fence
   exception: `start_simulation_mode` +3 lines over its measured 3323 (both length fences now 3326) to capture the code at sim start. Owner
   decisions for the lint half: one committed exceptions file (owner, reason, date; stale entries fail);
   a `rerun_*` directory needs its own PRE-DATA amendment; the gate checks record-citing status changes;
   a run that ended `error` or `unknown` is no more citable than a typed abort; a dirty-tree allowance lives
   only in the harness record and binds a dirty sim report by commit, run window and the shared
   `code_tree_sha256` (a sim report never carries `allow_dirty`; a hand-run sim needs a clean tree or an
   owner exception), and the prereg lint's echo check must name the entry or harness run. A record whose
   stamp is unestablished (hash `unknown`, dirty unallowed, an empty stamp, no `ts`, or
   `code_changed_during_run`) is not citable.
   **The clean-tree flag ([#998](https://github.com/dennys246/Maxim/issues/998), before M1b):** it no longer
   asks `git status`; a tree is clean when every code path is on disk exactly as HEAD has it (blob,
   executable bit, symlink target), over the same path set the digest hashes, and a refusal names the
   first difference. One stdlib-only module, `utils/code_tree.py`, which the harnesses load by path.
   Then the **complete Exp 10 re-run** that replaces 1.3.1's narrow one — pre-registered, provenance-stamped,
   phases run to their cap — and a **complete Exp 09 re-run**. Since M1b PR 3 (2026-09-30, owner decision)
   both ledger rows are `STALE` (T1-1, T3-9: their narrow re-runs ended in typed aborts), which blocks the cut.
3. **`create.*` overwrites an existing store; `load.*` raises raw errors ([#939](https://github.com/dennys246/Maxim/issues/939)).**
   Guard: a test that fails on today's clobber.
4. **The silent seams #840 and #841, and type-checking the composition layer.** Both are an argument
   mismatch swallowed by a broad `except` — exactly what mypy catches. Extend CI's mypy set to `runtime/executor.py`,
   `runtime/agent_loop.py`, `bridges/`, `planning/`, and start the repo-wide error-count ratchet carried from
   1.3.1 (1,071 at the tag; it may only fall — per file, against each PR's merge-base, not as a repo
   total: errors cross files, so two PRs that each pass can together raise the total; see the residuals in
   `scripts/lint_mypy_ratchet.py`'s docstring). Guard: CI mypy on those modules + the ratchet.
   **Ratchet half BUILT 2026-10-04** (owner decision 2026-10-04: per file, pinned at today's counts, no
   file may rise, new files start at 0): `scripts/lint_mypy_ratchet.py` in the lint job, positive control
   `tests/unit/test_lint_mypy_ratchet.py` (every mechanism deletion-proven). No committed baseline: each
   PR measures its HEAD and its merge-base (`git archive`) in the same job with one fixed invocation
   (empty `--config-file`, `--no-site-packages`, `--check-untyped-defs`, cold cache), and a separate
   per-file suppression counter (`type: ignore`, `# mypy:`, `@no_type_check`, any test mentioning `TYPE_CHECKING`/`MYPY`)
   may not rise either. The count lives in that step's CI output; cold-run numbers at the build, for the
   record: **1,096 errors in 142 files, 87 suppressions** at merge-base `4bf1fe7d` (identical on a dev
   venv and on a bare lint-job venv, because `--no-site-packages` removes the environment; without it a
   full dev venv reads 1,105), and **1,075 in 141 files at `v1.3.1`** measured the same way (the "1,071"
   above came from a different invocation). The +21 since the tag: `cli.py` +17, `orchestrator.py` +4,
   seven files +1, four files falling. About 16 s per run (two mypy runs of ~5 s each plus the scans).
   **CI-set half BUILT 2026-10-04 (PR [#1087](https://github.com/dennys246/Maxim/pull/1087), [#1083](https://github.com/dennys246/Maxim/issues/1083)):** the four
   paths join CI's mypy step at 0 errors (37 before). The extension found a real crash: an approved
   PLANNING proposal had no `cluster_id`/`clusters`, so its outcome record raised after the tool ran
   (#1083, fixed: the queued `Proposal` references its `LLMProposal` and credit keys to proposal time, owner
   decision 2026-10-04; full parity with the autonomous path is #1085). `fear_bridge.py`'s three dead
   calls carry line-level ignores (Dormant, #840); `AdaptivePlanner.set_mesh_context` is marked Dormant
   ([#1084](https://github.com/dennys246/Maxim/issues/1084): no non-test caller). The other 33 were
   behaviour-neutral annotation and narrowing fixes.
5. **Coverage as a ratchet — and a coverage push where the risk is** (widened 2026-09-27, owner). Baseline:
   the Codex card's whole-suite run at `v1.3.1` — **61.9% of 90,416 statements**, 31,797 uncovered
   ([evidence](../limits/score_cards/evidence/2026-09-27-codex/)). Three mechanisms, all in CI:
   - **An overall floor and per-package floors** that only rise, set from that baseline. *(Superseded
     2026-10-04, owner: the floors are measured in CI's own environment on the gate PR, not taken from
     this card baseline; see "Gate half built" below.)*
   - **Changed-line coverage ≥ 80% on every PR** (diff coverage), so new and moved code arrives tested
     whatever the file's history.
   - **A reviewed exclusion list** for code that needs a model or hardware (vision engines,
     `inference/transcribe_audio.py`, `models/language/transformers_backend.py`, camera display): covered
     by the model-cache nightly or named with a reason — never silently omitted (today's
     `pyproject.toml` omits `embodied_runtime/selfy.py` without one).

   **Where the push goes, by risk × uncovered lines** (coverage at `v1.3.1`):
   | Area | Coverage | Why first |
   |---|---|---|
   | the three decomposition targets — `orchestrator.py`, `cli.py`, `agent_loop.py` | 11%, 14%, 51% | pinned by characterization tests before any slice (below) |
   | `bridges/` (`fear_bridge.py` 29%) | 49% | where #840's silent failure lives |
   | `default_network/`, `attention/`, `math/angular_gyrus.py` | 41%, 33%, 45% | bio systems that feed behaviour |
   | `tools/sandbox.py`, `utils/sandbox_executor.py` | 27%, 70% | the code-execution boundary |
   | `leader_proxy.py`, `router.py`, `lane_backends.py`, `peer/cli.py` | 53%, 63%, 66%, 46% | network, auth, routing |
   | `embodied_runtime/` (mockable parts: `agentic_runtime`, `movement`, `workers`) | 32% | the robot runtime |

   **Quality, not just lines:** the push writes behavioural and composition tests (a caller through its
   real callee — the #840/#841 class), each new guard proven by deleting its mechanism; a test that
   raises coverage without asserting behaviour does not count. Guard: the three CI checks above; the
   floors, the changed-line threshold and the exclusion list are committed files, and the lint fails
   on a floor that drops or an exclusion without a reason.
   **Gate half built 2026-10-04, PR pending; floors bootstrapped from CI.** `scripts/lint_coverage.py`
   runs in the `unit-tests` job after the fast suite, which now carries `--cov` in its single run
   (`coverage==7.13.3`, `pytest-cov==7.0.0` pinned exactly; `fetch-depth: 0`; timeout 60). Owner decisions
   2026-10-04: floors measured in CI's own environment on the gate PR and pinned rounded down to 0.1 pt
   (not the v1.3.1 card number), as the minimum of two CI runs (`--merge-floors`); a new or changed floor
   is verified against the measurement: a floor is lowered only to round_down(measured) and only when
   uncovered statements did not grow, and its pinned `missing` never rises, so no PR chain can walk a floor
   down; a scope fails below its floor or more than 1.0 pt above it; diff coverage
   ≥ 80% of changed executable `src/maxim` statements against the merge-base, with moved code counted as
   changed. `scripts/coverage_floors.json` holds an overall floor and one per package (first path
   component; top-level modules as `maxim/<root>`; 200 statements or more, kept while the directory
   exists), each with its pinned `missing` count: a floor drops only when `missing` does not rise (a
   deletion of covered code). It is committed with `null` values: the gate PR's first CI run prints
   the measured floors, and they are committed on the same branch. `scripts/coverage_exclusions.json` is
   the reviewed exclusion list (today only `embodied_runtime/selfy.py`, with its reason) plus an
   append-only ledger: each excluded file's statement count, and every file's count of `exclude_lines`
   matches (`pragma: no cover`, `if TYPE_CHECKING:` and the rest, except an imports-only `TYPE_CHECKING`
   block and a lone `raise NotImplementedError`), seeded at today's counts. A rise needs a new entry with
   a `ref`. `pyproject.toml`'s `omit` is derived from the list exactly, and the whole
   `[tool.coverage]` table, a second coverage config, banned pytest options, `COVERAGE_*` env and the
   coverage API in tests are checked. **Open:** the model/hardware files named above are not omitted
   today, so they are measured and counted; whether any join the list is an owner decision.
   `covered_by` must be null until the model-cache nightly produces coverage data the lint can read. Guard: `tests/unit/test_lint_coverage.py` (each mechanism
   deletion-proven).
6. **One function-length ratchet** — nothing over 200 lines may grow, nothing new may exceed 200 — replacing
   the two mismatched mechanisms ([#940](https://github.com/dennys246/Maxim/issues/940)). Guard: the lint, with
   a per-function baseline.
   **Built 2026-10-04, merged as #1090 (#940 item 1); follow-ups #1089 merged as #1115.** `scripts/lint_function_length.py` is now the only checker
   (`tests/unit/test_function_length_baseline.py` is deleted) and covers all of `src/maxim`: every function
   over 200 lines is pinned in `src/maxim/utils/function_length_baseline.json` (format 2) at its exact span,
   **53 at the build, 19 of them over 300**. Owner decisions 2026-10-04: threshold 200; strict equality (a
   shrink fails until the pin is lowered in the same commit); a raise, or a new entry over 200, only through a
   committed exception record that is new in the same diff. Exceptions are append-only against the
   merge-base, and an unused new one fails. A move is free only when the body's AST is identical; any other
   move carries `moved_from`. A decomposition piece still over 200 carries `split_from`, and the lint checks
   that the source's pin dropped in the same diff, so each slice of the decomposition below records the debt
   it transfers. The prose HISTORY of earlier raises survives as the file's `history` list. Every pinned span
   and the totals print on each run.
7. **CI escape paths ([#940](https://github.com/dennys246/Maxim/issues/940))**: the `|| echo` optional install,
   the reason-less `importorskip`, the slow lane's expected roster, the network guard at the process-tree
   boundary. Guard: each lane fails on the escape.
   **The importorskip half built 2026-10-04, merged as #1097:** `--require-extras` now refuses to start unless each
   requirement's top-level module imports (`tests/conftest.py::_require_extras_importable`), and also reads pytest's
   own "could not import" message for those modules. Owner decisions 2026-10-04 for the rest: the slow lane
   installs and runs its tests (46 selected and 16 ran on the 2026-10-05 nightly) against an exact roster with reasoned
   skips only; the unit-tests pytest step runs in a loopback-only network namespace, with no OS-level exception.
   **The rest built 2026-10-05, merged as #1122** (its first dispatch run found a leaking test, an always-skipping
   guard and two real substrate failures, now strict red gates on [#1120](https://github.com/dennys246/Maxim/issues/1120)):
   - `|| echo` is gone from both install steps.
   - The slow lane installs the semantic/console/sign extras and the model cache, runs with
     `--require-extras`, and is held to the exact roster in `scripts/lane_rosters/slow.json`
     (`scripts/check_lane_roster.py`; one checker and one setup action for both nightly lanes since #1117, which
     also holds the model-cache lane to an exact roster). Its PR-time twin is `tests/unit/test_lane_rosters.py`.
   - The fast suite runs in a loopback-only network namespace; its positive control is
     `tests/unit/test_network_boundary.py`.
   - Structural pins: `tests/unit/test_ci_workflow_shape.py`.
   The namespace wraps the fast suite and both nightly lanes (`scripts/ci_netns.sh`). Stated limit: the MemoryHub step
   keeps the in-process guard only, because the coverage gate pins its exact form.
   **Item 7 DONE 2026-10-06.** Follow-ups merged: #1132 (#1091: the ledger ref and append-only rules shared in
   `scripts/_lint_allowance.py`; the two TYPE_CHECKING detectors deliberately kept apart), #1139 (#1117: one roster
   checker, `scripts/check_lane_roster.py --lane`, and one setup action, `.github/actions/model-cache-setup`, for
   both nightly lanes; the model-cache lane now has an exact roster too). Watch: `ubuntu-latest` moves to Ubuntu 26
   on 2026-10-19; the first nightly after it is the check that `scripts/ci_netns.sh` still works.
8. **One source of truth for claims** (mechanization backlog M2). Guard: the claims-registry lint.
   **Built 2026-10-04, PR pending** (owner decisions 2026-10-04: the ledger is the single source; the surfaces are linted,
   not generated; v1 covers the README results table and the experiments index): `scripts/lint_claims_sync.py`. Each
   README results row and each index row citing a ledger row carries `<!-- claim: T1-n -->` and must show that row's
   status token and date verbatim, its scope word, a SUPERSEDED row's successor, and no other uppercase status token.
   Every Tier 1 row is cited by the index; the one reasoned exemption is T1-5, which has no experiment doc. The README's
   memory row now claims Exp 63 (T1-16, EARNED narrow), with Exp 10 named as superseded. It counts as a guard once its
   CI lint step lands, with Session B's `test.yml` batch after #1092; then it also discharges
   [#940](https://github.com/dennys246/Maxim/issues/940) item 2's guard (the README's Exp 10 row, corrected on 2026-09-27
   with nothing to stop it drifting). Remaining surfaces: backlog M36.
   **Item 8 DONE: merged as #1107; its CI lint step landed with #1122.** Follow-up: #1143 (#1108, owner decisions
   2026-10-06) makes the evidence gate judge ANY qualifier change and require new support for a widened one
   (removed, or its scope word changed); an owner exception supports a widening only with `to_qualifier`.

**Engram integrity (pulled in from 1.4's parallel line, 2026-09-27).** Its four engineering items gate
1.4.0 (release threshold T7), touch no survival rung's path and run off the rig
([engram_formation.md](engram_formation.md)), so they fit the hardening line:
[#908](https://github.com/dennys246/Maxim/issues/908) the Cerebellum is never saved (guard: a round trip
that fails on today's default config; done 2026-10-04); [#909](https://github.com/dennys246/Maxim/issues/909) the
motor-engram docs overclaim and the read side is undeclared-dormant (guard: the Dormant docstring + a
caller-grep test; done 2026-10-04); [#910](https://github.com/dennys246/Maxim/issues/910) the `[DANGEROUS]` annotation is
unreachable (guard: a test that reaches it through the real annotator); [#911](https://github.com/dennys246/Maxim/issues/911)
the text-only reward-widening drift hazard (an offline measurement committed as a record, not a fix).

**Issue burn-down (added 2026-09-27, owner).** 1.3.2 spends a solid share of its time closing open
GitHub issues, because most of them are the class both score cards penalise: something that silently
never worked. Every issue open on 2026-09-27 (39) has a home: the ten already scheduled keep theirs, the rest are placed below.

- **Already scheduled (10).** Nine are in 1.3.2 above: #935, #939, #840, #841, #940 and #908–#911.
  #938 is "before 1.4.0", below this block, and keeps that home.
- **Placed here.** The rest are sorted into four homes: a **commitment** (batches 0–2), **best
  effort** (batches 3–4), **with the slice that owns the file**, and **1.4**.
- **Split issues.** Some are split between two homes, and each split names both halves.

*Rules for every burn-down fix:*

- **Reproduce first.** The fix PR's first commit adds a test that fails on today's code, and the fix
  commit turns it green. A probe that does not reproduce the defect closes the issue only if it asserts
  that the defect's precondition was actually reached, by constructing it directly. #816 needs a
  compressed concept and #819 a store full of long-term memories at its cap. If the probe cannot assert
  its precondition, the issue stays open.
- **Guard by deletion**, and **a caller, not a capability** (M3). Same bar as the rest of 1.3.2.
- **Batch by concern**, one PR per batch. **Exception:** an issue that changes a format contract, owes
  a decision, or is marked ⟲ gets its own PR. Each PR gets a review round (three lenses since 2026-10-05: [CODE_REVIEW.md](../CODE_REVIEW.md)).
- **⟲ marks a fix that changes what an EARNED ledger row's path computes.** Each ⟲ fix:
  - names its rows;
  - adds each row's own `Re-run on:` trigger to the walk;
  - discharges each row **in its own PR**, by that row's re-run or by a dated structural annotation
    on the row (memory plan Phase 0's rule, "each fix PR states which ledger rows it re-ran or
    discharged").

  The decomposition's closing Exp 60 re-run discharges Exp 60 only. **Timing:** ⟲ fixes land before
  the first `agent_loop` slice's characterization commit, so that closing re-run measures the
  decomposition alone. Fixes that fire the Exp 10 row land before the complete Exp 10 re-run (item 2),
  so that re-run covers them rather than going stale on arrival.

*The fence* (roadmap 1.4 Groundwork's boundary, verbatim): "Nothing under `runtime/agent_loop.py`,
`decisions/nac.py`'s selection path or `simulation/orchestrator.py` until the 1.3.2 decomposition slices
touching it have landed." Every batch below stays inside it. A ⟲ fix in best-effort batch 4 that lands
after decomposition work has started is discharged by its own rows' re-runs in its own PR, never by the
closing walk.

**Fence exception, #1042 (owner-approved 2026-10-01):** the simulation stall detector's suppression in
`orchestrator.py::start_simulation_mode` was dead (it queried the lane, the router registered the cost tier), and
O19 Exp 10 attempt 1 aborted on the stale nudges it let through. The touch moves the decision out to
`runtime/stall_threshold.py::stall_suppression` (one call, fed by the bridge's new `turn_in_progress` flag and the
lane timeout), restarts the idle clock while the agent's turn is in progress, reports a registry failure instead of
swallowing it, and exempts ping-pong; the function shrinks (ceiling 3277 → 3265). Follow-ups: #1043, #1044.

**Fence exception, #1042 PR B (owner-approved 2026-10-01, text only):** the narrator's stall-nudge and
diversity-checkpoint strings in `orchestrator.py::start_simulation_mode` now label the agent-under-test's tool names
as not the narrator's own (they were quoted unlabelled, and the narrator echoed them). No logic moves; the function's
length is unchanged.

**Fence exception, narrator reliability (owner-approved 2026-10-02, before O19 campaign 3):**
- `agent_loop.py::run_agentic_loop` holds the next planning submit while the narrator's previous job is in flight
  (`_submit_held`, from the module-level `_planning_submit_in_flight`). It folds inputs held meanwhile into the
  follow-up it submits (`_take_deferred_inputs`, `deferred_inputs=`).
  A refused or raising submit releases them (`_release_unsent_deferred_inputs`, in a `finally`).
- The stale-proposal drop moved to the module-level `_drop_stale_proposal`; ceiling 3391 -> 3384.
- `_handle_planning_failure` (module-level) passes its reason to the retry.
- `orchestrator.py::start_simulation_mode` builds the narrator's openings with `sim_types.build_kickoff_prompt` /
  `narrator_tools_block` from its registry, and takes `min_finish_turns` (the cap, only under the opt-in
  `--sim-run-full-turns`; observe-only exempt) for `FinishSimulationTool`; ceiling 3248 -> 3229. The flag's caller
  is Exp 09's O19 protocol (amended before data, 2026-10-03; the owner dropped an Exp 10 campaign 3, #1060), pinned by
  `test_narrator_reliability.py::test_every_open_o19_campaign_runs_every_turn_it_asks_for`.
- All of it is gated on planning liveness, which only the narrator has, or lives in the narrator's own tools.

**Fence exception, #1052 (owner-approved 2026-10-02):** `submit_context`'s `deliberation_available` is a REQUIRED
keyword, so `agent_loop.py::run_agentic_loop` states it twice. The planning submit passes `bio_enrichment_pipeline is
not None`, and the deliberation-cycle submit passes True (ceiling 3389 -> 3391). The fact "this loop can deliberate"
exists only in the loop, so no unfenced seam can supply it; the decision lives in `prompt_builder` (unfenced). In
`orchestrator.py::start_simulation_mode`, the narrator's three-way kickoff instruction moved to
`sim_types.kickoff_instruction`, which the resume prompt now shares (`observe_only=` passed). The function shrinks:
3265 -> 3248.

*Commitment: batches 0–2.* These close in 1.3.2. Moving one to a later release takes an owner decision
recorded in this plan.

| Batch | Issues | Notes |
|---|---|---|
| 0. CI truth | [#951](https://github.com/dennys246/Maxim/issues/951) the scripted water-trial tests depended on wall time | **First in the chain** (§Sequence). A flaky required check turned `main` red on a docs-only merge, and until it is fixed every red check is ambiguous. **Fix as shipped (owner chose the test-side lockstep, 2026-09-27):** the scripted world runs on a step clock that the harness advances, so what a sample sees is fixed by the harness's own schedule. The donor smoke's `train_cap_s` is only a wall-clock timeout, valid because that scripted world has no damage onset. The lethal death test's "no call before the death" assertion encoded scheduling (3/3 CI failures). It is replaced by "nothing acts before damage exists", with a 2 s escape delay that makes the death certain. It neither retries nor lengthens any window a verdict reads. **Not ⟲ after all:** `WaterTrial` and the live bridge are untouched, and the scripted bridge's default wall mode is behaviourally identical (`clock=None`), so the four verdicts are discharged structurally (no verdict code changed). The Exp 61/62 rows carry a "guard edited, not fired" note. Other scripted tests in the same race class: [#954](https://github.com/dennys246/Maxim/issues/954). |
| 1. Security | [#949](https://github.com/dennys246/Maxim/issues/949) coding tools run in the host cwd with the full environment; [#921](https://github.com/dennys246/Maxim/issues/921) `validate_base_url` / `download_to_file` re-resolve after their check; [#924](https://github.com/dennys246/Maxim/issues/924) a passive agent can self-switch to active; [#829](https://github.com/dennys246/Maxim/issues/829) the mode re-exec passes a `--mode` argparse rejects; [#832](https://github.com/dennys246/Maxim/issues/832) items 1, 3, 4, 5 of the internet-policy follow-ups | **#924 decided STRICT (owner, 2026-09-27) and landed before #829**, because fixing the re-exec makes passive→active work end to end: the model's mode tool never raises capability until #922's approval surface exists (`modes.definitions.raises_capability`). It reverses the 2026-09-26 "keep #821" decision; the pinned `test_passive_to_active_still_allowed` became `test_passive_to_active_is_refused`. **#829 and #832 edit `cli.py::_main_impl`'s body** (the re-exec loop, `_operational_mode`, the `internet_access` state). They land before `_main_impl`'s characterization pass, a deliberate exception to the land-after-the-slice rule (the fence does not cover `cli.py`) because that slice is "if time". #832 item 2 (the approval gate, and the operator reset that rides with it) waits for #922. **#829 as shipped:** a `--operational-mode` launch grant (owner decisions), with a small owner-approved exception to the fence: three `agent_loop` read sites resolve the mode through `_effective_mode`. **#832 as shipped** (items 1, 3, 4, 5; owner decisions 2026-09-28, after a parallel review of the policy's shape): search obeys the operator's lists and page limit; the builder takes a required `internet_launch_enabled` and builds the getter; the recorded state is the effective one at launch; `InternetAccessPolicy` is frozen and operator-only, on/off composed at read time by `EffectiveInternetPolicy`. **Fence touches, both mechanical:** `orchestrator.py::start_simulation_mode` states `internet_launch_enabled=False` because the required argument makes the call a `TypeError` without it (ceiling raised 3322 → 3323, a reviewed exception). `agent_loop.py`'s two `internet_access` defaults went `True` → `False` to match the third (owner-approved; nothing decides on the value). That review's findings go here too: [#966](https://github.com/dennys246/Maxim/issues/966) (the private-address pre-check's classifier and DNS-failure reason) and [#967](https://github.com/dennys246/Maxim/issues/967) (robots.txt fails open; decision owed), [#968](https://github.com/dennys246/Maxim/issues/968) (IDNA 2003 vs 2008 for `ß`-style entries). |
| 2. Data safety | [#950](https://github.com/dennys246/Maxim/issues/950) `load.*` and `persistence_path` do not expand `~` (the store silently loads empty), in the same PR as item 3's #939; [#856](https://github.com/dennys246/Maxim/issues/856) a new config section breaks downgrades (decided 2026-09-28: bump to 1.1 + a pinned schema; own PR); [#816](https://github.com/dennys246/Maxim/issues/816) crash half, reinforcing a compressed concept raises (**moved here from memory Phase 0**); [#819](https://github.com/dennys246/Maxim/issues/819) loud half, the silently exceeded cap made loud (**moved here from memory Phase 4**; the O(N log N) eviction cost stays in Phase 4's heap); [#812](https://github.com/dennys246/Maxim/issues/812) typed ATL relations share one update slot; [#818](https://github.com/dennys246/Maxim/issues/818) ⟲ wall-clock decay ignores the inherent-class exemption (Exp 56/57/61 rows); [#843](https://github.com/dennys246/Maxim/issues/843) ⟲ percept captures default to success, and a double reinforce (treated as firing the Exp 10 row's "hippocampus persistence schema change": it changes stored values, not the schema, so this is a conservative judgment); [#932](https://github.com/dennys246/Maxim/issues/932) its non-orchestrator callers (listed below) | A user's memory silently lost or corrupted is the worst failure a memory system can have. #818 and #812 are free: the memory plan lists them as "filed separately". #818's ⟲ rests on "changes what the path computes" (no row's `Re-run on:` names wall-clock bias decay; Exp 57 is PARTIAL), which its PR states. #816's design half stays in memory Phase 3. **#950 + #939 as shipped** (owner decisions 2026-09-28): a save guard on Hippocampus and ATL (a store never writes over a file it neither read nor created), `create.agent` refuses an existing agent home, `~` expanded everywhere, missing-file `load()` raises, typed `MemoryCorruptionError` from `load.*`. The same guard for NAc, EC, SCN, AngularGyrus and the cross-layer index is [#971](https://github.com/dennys246/Maxim/issues/971), homed here, with [#972](https://github.com/dennys246/Maxim/issues/972) (the write-but-don't-read orchestrator still restores its ATL). An unreadable Hippocampus/ATL file a store starts fresh from is copied to `<name>.corrupt-<UTC>`, then saved over (owner decision at review). **Fence touch:** `decisions/nac.py`'s `save`/`load`/`load_safe` expand `~`; this is persistence only, not the selection path the fence names. **#856 as shipped** (owner decision 2026-09-28): `CONFIG_FORMAT_VERSION` bumped to `1.1` once for the four sections, the writer stamps its own version, and the schema is pinned per version in `tests/fixtures/config_schema_by_version.json`. The write half of a downgrade (an older build drops a newer build's keys on write) is [#974](https://github.com/dennys246/Maxim/issues/974), homed here. **#812 as shipped:** updates scoped to the relation type (`DependencyGraph.update_edge(metadata_match=)`); the reading rows discharged structurally (0 multi-typed pairs in 283 persisted ATLs) with an "ATL typed-relation update path" re-run trigger (owner decision). The grounder's own direction bug found in its review is [#976](https://github.com/dennys246/Maxim/issues/976), homed here. **#816's crash half had already shipped** in memory Phase 0 (`803ea63e`, guard `tests/unit/test_memory_phase0_input_integrity.py`, re-proven by deletion 2026-09-28); its reversible-compression half stays in memory Phase 3. **#819's loud half as shipped:** a WARNING whenever an insert leaves the store over its cap (at the first overage and each doubling, reset under the cap), plus `stats()["over_capacity_inserts_this_process"]`; eviction choice unchanged. **#974 as shipped** (owner decision 2026-09-28): `config set` refuses a newer file; `maxim config downgrade` sets unknown settings aside in `config.preserved.json`; `maxim config restore-preserved` re-validates, diffs and restores on an interactive confirmation (injection review: escaped output, no `--yes`, security flags). **#932's non-orchestrator callers as shipped** (owner decisions 2026-09-28): `load.session` resolves an exact ID or path through `resolve_run_dir`, and a prefix must match one session (several raise); the API home defaults to `data_home()` and its agent moved to `agents/api_agent/` with a one-time, all-or-nothing migration; the research/campaign reports write under `<data home>/sim_reports/`. **#971 as shipped** (owner decisions 2026-09-28): the save guard on NAc, EC, SCN, AngularGyrus and the cross-layer graph; an unreadable file kept as a copy and saved over, uniformly (SCN's pathless case retired); an unreadable EC resets its NAc; every write-but-don't-read overwrite declared; every automatic save logs a refusal at ERROR. **#972 as shipped:** write-but-don't-read holds at session start for the stores in the agent's home (`build_memory_hub(load_persisted=)`, now required, passed by both builders): the ATL, AngularGyrus and cross-layer graph. Follow-ups: [#984](https://github.com/dennys246/Maxim/issues/984) (shared learned state outside any home), [#985](https://github.com/dennys246/Maxim/issues/985) (`create_npc_agent` over an existing home). **#976 as shipped:** the grounder updates each typed relation once, in its stored direction (a symmetric one moved by two deltas per pass; an incoming non-symmetric one was never grounded); its trigger walk discharged the three rows structurally (behavioral_graduation_candidates.md, 2026-09-29). **#818 as shipped:** wall-clock decay-on-load exempts inherent-class cluster biases, as the per-tick decay did; the trigger walk (Exp 45's `Re-run on:` fires by wording) discharged every row that persists NAc: no recorded run can hold an inherent key (behavioral_graduation_candidates.md, 2026-09-29). Follow-ups: [#988](https://github.com/dennys246/Maxim/issues/988), [#989](https://github.com/dennys246/Maxim/issues/989). **#843 as shipped** (owner decisions 2026-09-29): a non-outcome memory's success is unknown (`None`) in both directions (percepts were successes, observations failures); unknown reads as neither in the memory readers (`ImportanceBasedStrategy` 0.65); no migration of files on disk; double reinforce and the forming-pool key collision fixed. Exp 10 discharged structurally **pending the complete Exp 10 re-run (the 1.3.2 plan's item 2; `outstanding.md` O19), which covers it** (owner decision); the Exp 37 prompt-text note is recorded. **#991 as shipped** (#843's follow-up, landed before the Exp 10 re-run as O19 requires): an enrichment memory's valence and icon come from its own outcome; `EpisodicMemory.success` answers like a compressed record's, and every reader of either kind goes through `memory.types.record_success` (tool and goal: #995). Exp 37 fires on prompt text; Exp 10 does not fire (valence neither ranks nor filters) and its re-run covers the prompt change. Every recorded LLM-AUT run since 2026-04-20 showed every surfaced episode with the failure icon. |

*Best effort: batches 3–4.* These may move to a named plan path with a trigger, recorded in this plan
(where M15 sees it), not only on the issue.

| Batch | Issues | Notes |
|---|---|---|
| 3. Silent seams | [#851](https://github.com/dennys246/Maxim/issues/851) ⟲ `ToolPainBridge` pending entries leak and disable embodiment-pain attribution (the "SEM pain → NAc cascade" row: "ToolPainBridge attribution change"); [#845](https://github.com/dennys246/Maxim/issues/845) memory consumers that never deliver: items 2, 3 and 5 (5 verified first), and item 4's `exec_agent.py::recall_deep` site (its `plan_manager` site is #841's); item 1 rides the slice — **per consumer, wire it or mark it Dormant, each with its behaviour tier declared**; [#863](https://github.com/dennys246/Maxim/issues/863) step 2, the telemetry wraps catching caller logic, **outside** `agent_loop.py`, `orchestrator.py` and `_main_impl`; **added 2026-10-06 (owner):** [#1138](https://github.com/dennys246/Maxim/issues/1138) `EpisodicRecallSource.recalled_items` ignores its limit (live since #1129; first, beside #1140) and [#1128](https://github.com/dennys246/Maxim/issues/1128) the Dormant `MemoryAgent` queries bump `access_count` every tick (default retention until memory Phase 5; one owner call, stop vs uncounted; its own ledger walk) | The #840/#841 class, beside item 4's mypy extension, which catches more of them. Session C order (owner, 2026-10-06): #1138 → simulation honesty (D49, D46/D50) → #1128 → [#1124](https://github.com/dennys246/Maxim/issues/1124) → #1077–#1079, #1081. |
| 4. Body defects | [#873](https://github.com/dennys246/Maxim/issues/873) ⟲ the no-silent-fallback half: `damage_component` fails on a missing part instead of reporting success (row 9); [#874](https://github.com/dennys246/Maxim/issues/874) ⟲ cradle heat never reaches the arm (row 9's "Cradle / drive / SEM body change"; probably discharged structurally, since row 9 is the dragon / `base_humanoid` setup, but stated in the PR) | Both are "reports success, did nothing". #873's four design points (sum vs weighted mean, a missing part, partless bodies, archetype reflex sets) stay with [deferred/reflex_layering.md](deferred/reflex_layering.md). **Added 2026-10-06 (owner):** [#1124](https://github.com/dennys246/Maxim/issues/1124) (#874's review) a dotted ROOT key shadows the real modulator sub-sensor in `evaluate_failures` (reloaded pre-#874 orphans; the derived `<mod>.integrity` keys); one design call, moving derived integrity off `vital_metrics`. **Decided 2026-10-07 (owner), fixed with the reload defect it sat on** (a reload emptied every modulator, so integrity froze and a missing trigger field read 0.0): integrity derived on read, Entity format 1.1, dotted keys dropped on load, a trigger with no reading does not fire. Follow-ups #1155, #1156. |

*With the slice that owns the file*, each in its own PR. A fix measured on a loop mid-refactor cannot be
told apart from the refactor.

- **With the `agent_loop` slices:**
  - [#835](https://github.com/dennys246/Maxim/issues/835): tool results travel as synthetic human
    input. It lands after the slice that owns the follow-up channel, which #834 builds on.
  - [#850](https://github.com/dennys246/Maxim/issues/850): `NAc.last_rpe` is sticky. Its per-goal RPE
    reaches `ExecAgent` only through the loop (`runtime/bio_integration.py`); owner decision on the
    binding design.
  - [#845](https://github.com/dennys246/Maxim/issues/845) item 1: the replan prompt from
    `runtime/loop_state.py`.
  - [#965](https://github.com/dennys246/Maxim/issues/965) (#832's review): the internet policy summary
    never reaches the model, and the recorded on/off is a launch snapshot. Wire a true summary through
    the live getter per turn, or delete the dead fields end to end (owner decision).
  - **[MERGED (#1198)]** [#963](https://github.com/dennys246/Maxim/issues/963) (#829's follow-up): the loop resolves the
    operational mode per call site, so the follow-up type ignores the launch grant. The slice that owns
    the loop's mode handling makes one accessor the only capability reader, guarded against new raw
    `state.data["mode"]` reads. #829's source-pin wiring tests become behavioural in the same PR.
  - [#1125](https://github.com/dennys246/Maxim/issues/1125) (#874's review): `Executor._drive_pressure_snapshot` misses modulator-qualified drives
    (`arms.thermal`, `head.thermal`, `arms.pressure`); with the slice that moves `_read_drive_ranges`.
    Record-only. Handed to Session A 2026-10-06.
  - [#863](https://github.com/dennys246/Maxim/issues/863): its `agent_loop.py` sites. Its `cli.py::_main_impl`
    sites go with that slice if it lands; if not, they stay open on #863 with this home recorded.
- **After the orchestrator's characterization tests** (the first slice of its decomposition):
  - [#932](https://github.com/dennys246/Maxim/issues/932): the `--resume-sim` lookup. Its other callers
    shipped in batch 2's PR: `session.py`, `api.py`, `research_orchestrator.py`, `campaign_runner.py` and
    `scripts/check_oscillator_coldstart.py` moved onto `resolve_run_dir` / `data_home()`; the
    `scripts/exp44/` and `scripts/benchmark_*` harness scans stay, listed in the persistence-config
    brief's invariant with their reason (they scan a data home they created, not name a run).
  - #863's `orchestrator.py` sites.
  - [#1123](https://github.com/dennys246/Maxim/issues/1123) (#874's review): the orchestrator's `set_entity_sensor` hint advertises `health` (not writable on
    derived-health bodies; it now fails) and "0.0-1.0" (value mode clamps to declared ranges), and the schema's
    `value` defaults to 1.0. With the `start_simulation_mode` slice. Handed to Session A 2026-10-06.
- **After the decomposition, not inside it:**
  - [#866](https://github.com/dennys246/Maxim/issues/866), then [#865](https://github.com/dennys246/Maxim/issues/865).
    Moving `sim_logger` touches imports in 79 files, `agent_loop.py` among them, and #866 changes
    `sim_log`, the hottest function on the path. That is design work, not a behaviour-preserving
    move, and each gets its own review.
  - If they do not fit in 1.3.2, they keep their home in [outstanding.md](outstanding.md) O12.

*1.4, by design.* Each needs an experiment or a mechanism review, not a hardening fix. Each is linked
from the plan that owns it:

- [#922](https://github.com/dennys246/Maxim/issues/922), [#834](https://github.com/dennys246/Maxim/issues/834)
  and #832 item 2 → [roadmap_1_4.md](roadmap_1_4.md) §Before 1.4.
- [#880](https://github.com/dennys246/Maxim/issues/880) → [deferred/nociception_layer.md](deferred/nociception_layer.md)
  (its F1).
- [#848](https://github.com/dennys246/Maxim/issues/848) → memory 2S-e.
- [#899](https://github.com/dennys246/Maxim/issues/899) → roadmap 1.4 Phase 5 keying.
- [#784](https://github.com/dennys246/Maxim/issues/784) → [world_channel_weighting.md](world_channel_weighting.md).
- [#1137](https://github.com/dennys246/Maxim/issues/1137) (compression keeps only the intent goal; a persisted-shape change) → the memory line
  ([memory_strength_and_forgetting.md](memory_strength_and_forgetting.md)). Trigger: the first path that compresses
  in a sim, or any work on sleep/consolidation in the loop (no committed store holds a compressed record today).
- [#1118](https://github.com/dennys246/Maxim/issues/1118) (the E4 record's `widening_overreach` field is narrower than the prereg's term) →
  [engram_formation.md](engram_formation.md) E4. Trigger: before any E4 re-run.
- **Session B follow-ups (placed 2026-10-07; owner to confirm):**
  - [#1141](https://github.com/dennys246/Maxim/issues/1141) (a widened qualifier's new support is not checked
    against the new scope) — **MERGED as #1158 (2026-10-08)**: pass-table entries declare `scopes`, and all new
    support must match the row's scope ([m1b_ledger_evidence_gate.md](m1b_ledger_evidence_gate.md) §New support).
  - **The Session B follow-up cleanup (owner, 2026-10-08: leave nothing hanging):** [#1012](https://github.com/dennys246/Maxim/issues/1012),
    [#1014](https://github.com/dennys246/Maxim/issues/1014), [#1037](https://github.com/dennys246/Maxim/issues/1037),
    [#1111](https://github.com/dennys246/Maxim/issues/1111) and [#1010](https://github.com/dennys246/Maxim/issues/1010)'s
    lint-side items (lint hygiene); [#1077](https://github.com/dennys246/Maxim/issues/1077), [#1078](https://github.com/dennys246/Maxim/issues/1078)
    and [#1081](https://github.com/dennys246/Maxim/issues/1081) items 1–6 (O19 hardening); [#1079](https://github.com/dennys246/Maxim/issues/1079)
    (the within-campaign leaked-gate bar: refuse, owner); #1081 item 7 (a redaction retires the campaign) and the data-PR secret
    scan, taken over from Session C; [#1103](https://github.com/dennys246/Maxim/issues/1103) (coverage nondeterminism). Owner
    decisions 2026-10-08: the O19 campaign cap is read from the merge-base (a raise is its own PR); "consider"-grade NITs are
    fixed when cheap and fail-closed, the rest closed with a recorded reason; #1010 items 8–9 (orchestrator) go to the
    orchestrator characterization slice. Each gate change gets an approach note and an adversarial design pass first.
    **All merged 2026-10-08/10** (#1169, #1170, #1171, #1182 with #1183, #1184, #1188, #1196 for [#1175](https://github.com/dennys246/Maxim/issues/1175),
    #1199 for [#1166](https://github.com/dennys246/Maxim/issues/1166), #1205). Its follow-ups wait on triggers, each linked from the doc read at that moment:
    [#1168](https://github.com/dennys246/Maxim/issues/1168) and [#1173](https://github.com/dennys246/Maxim/issues/1173) before the next O19 campaign (reproduction.md checklist step 0);
    [#1172](https://github.com/dennys246/Maxim/issues/1172) at the first non-O19 redaction and [#1174](https://github.com/dennys246/Maxim/issues/1174) at the first same-experiment earn-back
    (m1b §Redaction); [#1197](https://github.com/dennys246/Maxim/issues/1197) at the next `cradle_mother` or `exp49` run (simulation-experiments brief).
  - [#1120](https://github.com/dennys246/Maxim/issues/1120) (with the real encoder, the affordance-transfer
    negative controls form no water concept; two strict red gates) → Session C, beside the substrate work.
    Trigger: before any claim rests on IT-1's positive transfer (`tests/integration/test_affordance_transfer.py`).
  - [#1100](https://github.com/dennys246/Maxim/issues/1100) (flaky: `test_store_guard_971` NAc round trip across
    wall-clock decay) → batch 0, CI truth, the #951/#954 timing class. Trigger: its next CI firing.

**Done when:**
- Batches 0–2 are closed.
- Each of batches 3–4 is closed, or moved as recorded above.
- The slice-bound items have landed with their slices, or stay open with them.
- **Breaking changes in a patch release (owner, at the cut).** The release PR asks the owner whether the
  `[Unreleased]` breaking entries fit a patch: #939/#950 (`create.*`/`load.*` store guards), #932 (run-directory
  resolution, PR #980), #971 (the save guard on NAc, EC, SCN, AngularGyrus and the cross-layer graph) and #1071
  (`create.*` refuses an existing store at construction); #972 (`build_memory_hub(load_persisted=)`) breaks only an
  internal builder. Recorded here when #939 closed (2026-10-04 UTC) so the question is not lost.
- The release PR lists the open-issue count at `v1.3.1` and at the cut, and names each open issue with
  its home.

The scheduled items keep their own "ships with its guard" rule, and #938 keeps its home.

**Guards:**
- M15: every open issue is linked from the plan it is homed to.
- M16: a PR that touches a ledger row's `Re-run on:` path records a walk line.
- M17: a PR closing a bug issue carries a test that fails at its merge-base.

Until those exist, this block is the check, by attention.

**Before 1.4.0, not in 1.3.2:** the release pipeline in CI — [#938](https://github.com/dennys246/Maxim/issues/938)
fixed, then a tag-triggered workflow that builds once, audits those bytes, publishes them with PyPI trusted
publishing and creates the Release from them, plus a `v*` tag ruleset and required signatures. Both cards'
Release-governance gap; it also takes GPG off the release path.

**Not a process item:** Ambition moves only with new science — an EARNED result beyond one binary cue at
ceiling (a non-binary discriminator, or R4's delayed credit), and a production consumer for one
recorded-but-unused memory system. That is 1.4's work ([roadmap_1_4.md](roadmap_1_4.md)).

### The decomposition

**Scope widened 2026-09-27 (owner): two targets in order, a third if time allows.**

1. **`agent_loop.py`** (5,543 lines; `run_agentic_loop` 3,389 at `v1.3.1`) — first, because 1.4's Phase 0
   instrument is built on it.
   *(Slice 0, the gates, merged as #1114. Slice 1, the setup → `runtime/loop_setup.py::build_loop_run`,
   built 2026-10-05, merged as #1127: `run_agentic_loop` 3,381 → 3,248 lines. Slice 2, §0–0.6 the pre-tick
   gate → `runtime/loop_gates.py::pre_tick_gate -> GateOutcome`, built 2026-10-06, merged as #1136: 3,248 →
   3,125 lines; the helpers the body shares with the gate moved to `loop_state.py` and `loop_controller.py`.
   Slice 3, §6b the substrate tick → `runtime/loop_substrate.py::substrate_tick` and the proposer family →
   the leaf `runtime/substrate_proposal.py`, built 2026-10-07, merged as #1157: 2,819 → 2,783 lines (the 3,125 → 2,819 step between slices 2 and 3 was #1133's move of §4 into `tool_dispatch.execute_and_learn`).
   Slice 4, §5 the PLANNING approved path → `runtime/loop_planning.py::drain_approved`, which runs each approved
   action through `execute_and_learn(human_involved=True)` (#1085 PR-a: learning parity, a one-at-a-time drain,
   `approved_action_blocker` at drain time (pause, safety forbids, the policy's hard denials), no NAc for machine refusals, no overwrite retry for a write confirmed or approved by a person or a policy; it
   opens no approval route, the surface is #1185), built 2026-10-08, merged as #1187: 2,783 → 2,720 lines (then
   2,717 with #963, merged as #1198). Slice 5, §1.1–§1.16 perception (imagination, auto-sense, audio orientation)
   → `runtime/loop_perception.py::perceive -> PerceptionOutcome` (`imagine`, `auto_sense`, `orient_to_audio`, each
   verbatim and under 200 lines; `next_observation` and `state.update` stay at the call site), built 2026-10-10,
   PR pending: 2,717 → 2,429 lines; the same PR applies rule (c) below.)*
2. **`start_simulation_mode`** (`simulation/orchestrator.py`, 3,322). Its tests cover **11%** of its lines
   (the Codex card's measurement at `v1.3.1`), so it is NOT decomposed blind: characterization tests
   first, then an orchestrator coverage floor set from them (item 5 above), then slices under the same
   gates below. The coverage ratchet is what makes this decomposition safe.
3. **`_main_impl`** (`cli.py`, 1,696) — only if 1 and 2 land; it is CLI glue, the least valuable of the three.

The "kicked down the road" worry that kept this to one target is answered by the rule in **Sizing**: if a
slice stalls, 1.3.2 ships the slices that landed and the ratchet records the new ceiling.

**Baseline edits per slice (the function-length ratchet, item 6).** In the same commit as the extraction:
lower the source function's pin to the span the lint prints (or remove its entry if it is now 200 lines or
fewer); for each extracted piece over 200, add an entry at its printed span plus a new exception
`{file, qualname, from: null, to: <span>, split_from: {file, qualname: <source>}, date, ref, reason}`;
pieces of 200 or fewer need nothing. A bare `from: null` is refused in a diff that lowers or removes a pin. Full rule:
`scripts/lint_function_length.py`'s docstring.

**Import direction (slice 1 review, 2026-10-05).** Extracted modules must not keep reaching back into
`agent_loop`. (a) When a slice extracts code that calls an `agent_loop` module-level helper, the helper
moves to a leaf module in the same slice if only the extracted code calls it, and to a leaf both import
if both do. (b) A patch seam stays readable on `agent_loop` only for tests that already patch it there;
new tests patch the new location. (c) The tests that patch `agent_loop._record_outcome` and
`agent_loop.resolve_llm_loop_overrides` are retargeted in the slice that removes the last `_al.`
back-reference (slice 5 at the latest), which also deletes `loop_setup`'s lazy `agent_loop` import.
**(c) done in slice 5 (2026-10-10):** `resolve_llm_loop_overrides` moved to `loop_setup` (rule (a): its only
caller), `loop_setup` binds the run's outcome recorder as `tool_dispatch.record_outcome` through a module
reference, and the `agent_loop._record_outcome` re-binding and `loop_setup`'s lazy `agent_loop` import are gone;
the tests patch `loop_setup.resolve_llm_loop_overrides` and `tool_dispatch.record_outcome` (rule (b), no
re-exports). `tests/unit/test_loop_setup.py::test_no_loop_module_or_leaf_imports_agent_loop` now holds the
direction for the loop_* modules and every runtime module they import, a set the test derives as the
transitive closure of their `maxim.runtime` imports (an AST check of every import form, with a negative control
per form), and the runtime-tools brief carries it as an `[engineering]` invariant.
Slice 1 applied (a): `_prepare_executor`, `_loop_bio_handles`, `_build_loop_sensor_encoder`,
`_resolve_situation_cue` and `_planning_liveness_enabled_via_env` moved into `loop_setup.py`, so its only
`_al.` reads are the two seams. Slice 2 applied (a): `tick_embodiment_drift`, `_loop_live_tick`,
`_maybe_auto_revert_display` and `_loop_is_idle` (only the gate called them) moved into `loop_gates.py`; the
helpers the gate shares with the loop body went to existing leaves, `_effective_mode`, `_substrate_tick_due`
and `_planning_attempt_is_active` to `loop_state.py` and the D13 handlers (`_handle_planning_failure`,
`_handle_planning_transport_failure`, `_report_planning_exhaustion`) to `loop_controller.py`, beside the
counters they drive; `loop_gates` reads nothing through `agent_loop`. Slice 3 moved the whole substrate-proposer
family to the leaf `substrate_proposal.py` (no re-exports; every importer, `executor.py`, `loop_setup.py`, the
`scripts/` harnesses and the tests, retargeted), and applied (b): the three tests that patched
`agent_loop.propose_via_substrate` now patch `substrate_proposal`, which `loop_substrate` reads through its module
reference. (d) Extracted functions take
individual fields (or `ctrl`) as explicit keyword arguments, never the whole `LoopRun`.

**Coverage first, then extract (2026-09-27).** No slice moves code its tests do not pin. Each slice
adds characterization tests for the code it will move, in its own commit BEFORE the extraction, and
the extraction commit must keep them green unchanged. Extracted modules arrive at ≥ 80% line coverage
(the changed-line gate enforces it), and the target file's per-module floor rises to its new measured
value in the same PR. The orchestrator (11%) and `cli.py` (14%) get their characterization pass as the
first slice of their decomposition, not after.

**Behaviour preservation is the gate, not an aspiration.** Every slice must keep these green,
UNCHANGED, in the same PR (built in slice 0, owner decisions 2026-10-05):
- `tests/unit/test_agent_loop_selection_golden.py`, **the loop-level selection gate**: the real
  `run_agentic_loop` on a step clock (substrate-primary, a plain arm and a Wire-4 fear-on-water arm that
  must select `escape_water`), pinning every tick's decision, every executor call and the loop's
  lifecycle (session start/end, captures, persists, the 2S-d cue) against
  `tests/fixtures/agent_loop_selection_golden_v1.json`. Driver: `tests/unit/_loop_harness.py`.
- `tests/unit/test_decision_provenance.py`, the NAc-level byte-identical selection test (kept).
- `tests/unit/test_encoder_golden_v1.py`, the encoder golden pin.
- The offline verdict reproductions from committed data:
  `test_exp60_run.py::test_verdict_over_the_committed_exp60_record_matches_the_committed_verdict`,
  `test_exp61_run.py::test_verdict_over_the_committed_exp61_record_matches_the_committed_verdict` and
  `test_r3_run.py::test_report_over_the_committed_r3_bench_matches_the_committed_amended_report`.

A slice that cannot show all of them does not merge. **A golden is regenerated only from the
pre-slice commit, in its own commit** with the diff justified (the `--regen` entry point refuses a
dirty `src/`), never by pasting a slice's output. A source pin that says "the loop's code contains
X" reads `tests/unit/_loop_source.py` (`agent_loop.py` plus every `runtime/loop_*.py`), so it survives
an extraction. **Function-specific pins** (an ordering or region inside one function) are listed by
owning slice in that module's docstring and are updated consciously by that slice: slice 1
(`test_planning_liveness` gate definition and raise-after-teardown, `test_experience_clock` AST),
slice 2 (`test_planning_liveness` idle-gate pins), slice 3 (`test_substrate_action_budget` §6b
ordering), and the slice that moves §2 (`test_planning_liveness` proposal-time stamp; slice 4 moved §5, not §2).

**Characterization owed before slices 4 and 5** (no gate pins this code today; the executor lens,
slice 0): **slice 4** needs a PLANNING arm, or an extended `test_approved_proposal_situation_1083.py`,
pinning §5's `pending_action_followup`, `log_action` and `_reset_deliberation` (skipped for `think`) —
**done** (2026-10-08): the driver's `level="planning"` arm, `tests/unit/test_approved_path_characterization.py`.
Slice 4 is an exception to "Coverage first, then extract": by owner decision 1 (#1085, 2026-10-08) it is a fix as
well as a move, so its fix commit CHANGES those characterization pins (each changed test says so) instead of
keeping them green unchanged.
**Slice 5** needs a transcript-bearing or scripted-percept arm that runs §1.1 imagination, §1.15
auto-sense, §1.16 audio and `state.update(observation)`; only text pins touch them now — **done** (2026-10-10):
`tests/unit/test_loop_perception_characterization.py`, through the real loop on the LLM-primary path (a scripted
non-sim adapter, a sim percept source for the reflex tier, a capturing worker, the real
`bodies/reachy_mini_infant`), the block's statement coverage 27% → 100%. It pins, unchanged, the percept-text
divergence filed as #1202.

**Typing rides along, scoped:** every module the decomposition creates enters CI's mypy set. The
repo-wide ratchet from 1.3.1 holds the rest. Full coverage is not promised.

**Close it honestly:** the complete Exp 10 re-run (item 2 above), a trigger walk over the ledger (Exp 60 and Exp 61 both name
`run_agentic_loop`'s idle-gate and autonomy handling) and a live re-run of Exp 60 on the rig
(5 seeds per arm, ~1 h) to discharge them with a dated annotation, rather than an argument that a
pure extraction changes nothing.

**Sizing, honestly:** this is the largest item on the list — several sittings of work in reviewed
slices, plus the rig hour. If a slice stalls, the release ships the slices that landed; the ratchet
records the new ceiling either way.

---

## Sequence

```
1.3.0 (published 2026-09-19)
  → 1.3.1  what users need now (re-scoped 2026-09-26): #914 → public format freeze → #913 →
           security cluster (#800-802, #824, #826-828), nightly lane + release-reads-nightlies,
           network block, public-API
           fixes, D40, D32, doc honesty; plus everything on main since 1.3.0
  → 1.3.2  #951 (main's flaky red) → #935 + provenance + burn-down fixes that fire Exp 10 → the complete
           Exp 10 re-run → data-safety + silent seams (#939, #840/#841, mypy) → the ratchets (FIRST among the
           enforcement items; #951/#935 precede them because they unblock CI truth and re-runs) → engram
           integrity (#908–#911) → ⟲ burn-down fixes, each discharged in its own PR → the decomposition
           (agent_loop, then start_simulation_mode behind characterization tests; _main_impl if time)
           → trigger walk + live Exp 60 re-run
           ‖ in parallel, inside the fence: the issue burn-down, batches 1–4 (security first)
           ‖ in parallel, no loop code: 1.4 groundwork (roadmap_1_4.md §Groundwork in parallel with 1.3.2)
  → 1.4    Phase 0 instrument on the decomposed loop, then Exp 62 → E1 → …
             ([roadmap_1_4.md](roadmap_1_4.md))
```

Exp 62 RAN and is EARNED on the ledger (2026-09-20) — it never depended on either patch release. **It is NOT a 1.3.1 claim** (owner decision 2026-09-27): it rides a release only once its different-reader pass is recorded. Any further rung likewise runs on the rig in parallel
(`docs/experiments/exp62_pressure_interoception_prereg.md`, decisions D1–D4 taken).

## Cadence

**1.3.1 (owner decision 2026-09-27): a blind re-score is taken at the `v1.3.1` tag**, the same firewalled procedure as v1.3.0. After that: re-score at the 1.4 cut, or when an axis's "to reach" condition is claimed complete — and the claim
is that the guard exists, not that the work was done.
