# Roadmap 1.3.x — the hardening line: 1.3.1 (fixes + guards) → 1.3.2 (decomposition) → 1.3.3–1.3.6 (patch releases to 1.4.0)

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
| **[→ 1.3.2 — SUPERSEDED by item 6]** ~~**Extend the function-length ratchet to every function over 300 lines** (18 today, one of 921; the ratchet covers 3). Pin at current length, shrink-only.~~ Item 6 below replaces it with one ratchet at **200** lines (53 functions pinned, 19 of them over 300, measured 2026-10-04). **Built 2026-10-04, merged as #1090.** | `scripts/lint_function_length.py` + `src/maxim/utils/function_length_baseline.json` + `tests/unit/test_lint_function_length.py` |
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
| **[→ 1.3.2]** **Fail-loud Stage 3** — narrow the measurement-path swallows; green-lit since Stage 2 measured **zero** firings ([deferred/measurement_path_fail_loud.md](deferred/measurement_path_fail_loud.md)). Must not land mid-walk on a branch a graduation run reads from. *(2026-10-11: #1134 instrumented `loop_controller.py` and `loop_state.py` in the Stage-1 form and added them to the lock (Stage 1 + 4); their narrowing, like the rest, is still owed.)* | `scripts/lint_no_silent_swallows.py`'s zero-total set grows to cover each narrowed file |
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

> **Released as v1.3.2 "Decomposition" (2026-10-16 UTC).** The release ships: the agent_loop decomposition
> (setup, pre-tick gate, perception, planning, substrate tick); the ratchets and the evidence gate; the batch
> fixes; and a breaking leader security change. What did not finish moved with its home:
> - the orchestrator slices and agent_loop phase 2 are hard-gated for 1.4.0 (#1300);
> - the 1.3.3 moves are recorded in the release-readiness block below.
> T1-2 ships STALE under `release-1.3.2-T1-2`.

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
   **Gate half built 2026-10-04, merged as #1092 (2026-10-05); floors bootstrapped from CI. Per-file floors (format v2) merged as #1266 (2026-10-13).** `scripts/lint_coverage.py`
   runs in the `unit-tests` job after the fast suite, which now carries `--cov` in its single run
   (`coverage==7.13.3`, `pytest-cov==7.0.0` pinned exactly; `fetch-depth: 0`; timeout 60). Owner decisions
   2026-10-04: floors measured in CI's own environment on the gate PR and pinned rounded down to 0.1 pt
   (not the v1.3.1 card number), as the minimum of two CI runs (`--merge-floors`); a new or changed floor
   is verified against the measurement: a floor is lowered only to round_down(measured) and only when
   uncovered statements did not grow, and its pinned `missing` never rises, so no PR chain can walk a floor
   down; a scope fails below its floor or more than 1.0 pt above it; diff coverage
   ≥ 80% of changed executable `src/maxim` statements against the merge-base, with moved code counted as
   changed. `scripts/coverage_floors.json` holds an overall floor, one per package (first path
   component; top-level modules as `maxim/<root>`; 200 statements or more, kept while the directory
   exists) and, since format v2 (2026-10-13), opt-in per-file floors (`files`, with the split rule and
   `retired_files`; section 1b of the lint's docstring), each with its pinned `missing` count: a floor drops only when `missing` does not rise (a
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
   both nightly lanes; the model-cache lane now has an exact roster too). The Ubuntu 26 watch fired: on 2026-10-20
   `ubuntu-latest` became Ubuntu 26.04, whose sudo-rs ignores `sudo -E`, so `scripts/ci_netns.sh` stopped carrying
   the job's environment into the namespace (the tts extras cells and both nightlies red, `unit-tests` green without
   it). Fixed by passing the environment by name with an in-namespace arrival check (O20 in
   [outstanding.md](outstanding.md) §Closed).
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
`decisions/nac.py`'s selection path or `simulation/orchestrator.py` until the decomposition slices
touching it have landed." (It read "the 1.3.2 decomposition slices" until 2026-10-18.) Every batch below stays inside it. A ⟲ fix in best-effort batch 4 that lands
after decomposition work has started is discharged by its own rows' re-runs in its own PR, never by the
closing walk. The remaining slices are not in 1.3.2 (owner decision 2026-10-15), so the fence stays up until they
land: the fence lifts when 1.3.6's decomposition slices land — the point at which T0 can pass (owner decision
2026-10-18; T0 is the 1.4.0 gate, [roadmap_1_4.md](roadmap_1_4.md)).

**Fence touches, the NAc audit, #1361 (recorded 2026-10-17, owner decision):** [#1363](https://github.com/dennys246/Maxim/pull/1363) (merged) edited `decisions/nac.py` outside the selection path, by the #971 precedent (persistence and bookkeeping): orphan `_cluster_reward_source` pruning in the decay and load paths, the `_event_outcome_welford` bound in `_enforce_limits` (it feeds the Wire-1 risk annotation, so it is recorded here), and `SCN.register(observe_oscillator=)`. Its correctness sibling [#1365](https://github.com/dennys246/Maxim/pull/1365) (merged) carried one `runtime/agent_loop.py` hunk, recorded here as a fence touch after merge: section 8.5 runs each per-tick NAc decay in its own `try` with a Stage-1 report, so one raising decay no longer skips the rest. No behaviour change on the success path; Session A (which owns `agent_loop.py`) rebases over it. The rest of #1365 (credit-entry guards, the session observer, the locked eligibility snapshot, the `:` refusal) is outside the fence. The experience-clock decay plan ([experience_clock_decay.md](experience_clock_decay.md)) runs S0 (measurements, strict red gates, doc fixes; tests and docs only) and, by the same precedent, S1 (offline decay opt-in on NAc load) now; S2 coordinates with Session V's `simulation/` files; S3–S6 are fenced and home in 1.3.6. S1's follow-up [#1102](https://github.com/dennys246/Maxim/issues/1102) (owner decision 2026-10-18) edits `decisions/nac.py` by the same precedent, persistence only: `load_state` clamps each restored bias map to its live writer's range (`_load_bounds`); the credit writers, the selection path, `agent_loop.py` and `orchestrator.py` are untouched.

**Fence exception, the atonia gate, #1315 (owner-approved 2026-10-17, AD8):** item 3 of
[advisor_mode.md](advisor_mode.md) may touch `orchestrator.py` (the AUT and narrator executor builds) and
`agent_loop.py` (`run_agent_loop`; and the PLANNING proposal path, for item 1's advisor composition). The scope is
`ToolPolicy` call sites and the advisor composition only. Rule T (#1327) applies to every line touched inside a
pinned function. Session A grants the hunks.

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
| 1. Security | [#949](https://github.com/dennys246/Maxim/issues/949) coding tools run in the host cwd with the full environment; [#921](https://github.com/dennys246/Maxim/issues/921) `validate_base_url` / `download_to_file` re-resolve after their check; [#924](https://github.com/dennys246/Maxim/issues/924) a passive agent can self-switch to active; [#829](https://github.com/dennys246/Maxim/issues/829) the mode re-exec passes a `--mode` argparse rejects; [#832](https://github.com/dennys246/Maxim/issues/832) items 1, 3, 4, 5 of the internet-policy follow-ups | **#924 decided STRICT (owner, 2026-09-27) and landed before #829**, because fixing the re-exec makes passive→active work end to end: the model's mode tool never raises capability until #922's approval surface exists (`modes.definitions.raises_capability`). It reverses the 2026-09-26 "keep #821" decision; the pinned `test_passive_to_active_still_allowed` became `test_passive_to_active_is_refused`. **#829 and #832 edit `cli.py::_main_impl`'s body** (the re-exec loop, `_operational_mode`, the `internet_access` state). They land before `_main_impl`'s characterization pass, a deliberate exception to the land-after-the-slice rule (the fence does not cover `cli.py`) because that slice is "if time". #832 item 2 (the approval gate, and the operator reset that rides with it) waits for #922. **#829 as shipped:** a `--operational-mode` launch grant (owner decisions), with a small owner-approved exception to the fence: three `agent_loop` read sites resolve the mode through `_effective_mode`. **#832 as shipped** (items 1, 3, 4, 5; owner decisions 2026-09-28, after a parallel review of the policy's shape): search obeys the operator's lists and page limit; the builder takes a required `internet_launch_enabled` and builds the getter; the recorded state is the effective one at launch; `InternetAccessPolicy` is frozen and operator-only, on/off composed at read time by `EffectiveInternetPolicy`. **Fence touches, both mechanical:** `orchestrator.py::start_simulation_mode` states `internet_launch_enabled=False` because the required argument makes the call a `TypeError` without it (ceiling raised 3322 → 3323, a reviewed exception). `agent_loop.py`'s two `internet_access` defaults went `True` → `False` to match the third (owner-approved; nothing decides on the value). That review's findings go here too: [#966](https://github.com/dennys246/Maxim/issues/966) (the private-address pre-check's classifier and DNS-failure reason) and [#967](https://github.com/dennys246/Maxim/issues/967) (robots.txt fails open; decision owed), [#968](https://github.com/dennys246/Maxim/issues/968) (IDNA 2003 vs 2008 for `ß`-style entries). All three fixed by [#1282](https://github.com/dennys246/Maxim/pull/1282) (2026-10-14; release-readiness block below). |
| 2. Data safety | [#950](https://github.com/dennys246/Maxim/issues/950) `load.*` and `persistence_path` do not expand `~` (the store silently loads empty), in the same PR as item 3's #939; [#856](https://github.com/dennys246/Maxim/issues/856) a new config section breaks downgrades (decided 2026-09-28: bump to 1.1 + a pinned schema; own PR); [#816](https://github.com/dennys246/Maxim/issues/816) crash half, reinforcing a compressed concept raises (**moved here from memory Phase 0**); [#819](https://github.com/dennys246/Maxim/issues/819) loud half, the silently exceeded cap made loud (**moved here from memory Phase 4**; the O(N log N) eviction cost stays in Phase 4's heap); [#812](https://github.com/dennys246/Maxim/issues/812) typed ATL relations share one update slot; [#818](https://github.com/dennys246/Maxim/issues/818) ⟲ wall-clock decay ignores the inherent-class exemption (Exp 56/57/61 rows); [#843](https://github.com/dennys246/Maxim/issues/843) ⟲ percept captures default to success, and a double reinforce (treated as firing the Exp 10 row's "hippocampus persistence schema change": it changes stored values, not the schema, so this is a conservative judgment); [#932](https://github.com/dennys246/Maxim/issues/932) its non-orchestrator callers (listed below) | A user's memory silently lost or corrupted is the worst failure a memory system can have. #818 and #812 are free: the memory plan lists them as "filed separately". #818's ⟲ rests on "changes what the path computes" (no row's `Re-run on:` names wall-clock bias decay; Exp 57 is PARTIAL), which its PR states. #816's design half stays in memory Phase 3. **#950 + #939 as shipped** (owner decisions 2026-09-28): a save guard on Hippocampus and ATL (a store never writes over a file it neither read nor created), `create.agent` refuses an existing agent home, `~` expanded everywhere, missing-file `load()` raises, typed `MemoryCorruptionError` from `load.*`. The same guard for NAc, EC, SCN, AngularGyrus and the cross-layer index is [#971](https://github.com/dennys246/Maxim/issues/971), homed here, with [#972](https://github.com/dennys246/Maxim/issues/972) (the write-but-don't-read orchestrator still restores its ATL). An unreadable Hippocampus/ATL file a store starts fresh from is copied to `<name>.corrupt-<UTC>`, then saved over (owner decision at review). **Fence touch:** `decisions/nac.py`'s `save`/`load`/`load_safe` expand `~`; this is persistence only, not the selection path the fence names. **#856 as shipped** (owner decision 2026-09-28): `CONFIG_FORMAT_VERSION` bumped to `1.1` once for the four sections, the writer stamps its own version, and the schema is pinned per version in `tests/fixtures/config_schema_by_version.json`. The write half of a downgrade (an older build drops a newer build's keys on write) is [#974](https://github.com/dennys246/Maxim/issues/974), homed here. **#812 as shipped:** updates scoped to the relation type (`DependencyGraph.update_edge(metadata_match=)`); the reading rows discharged structurally (0 multi-typed pairs in 283 persisted ATLs) with an "ATL typed-relation update path" re-run trigger (owner decision). The grounder's own direction bug found in its review is [#976](https://github.com/dennys246/Maxim/issues/976), homed here. **#816's crash half had already shipped** in memory Phase 0 (`803ea63e`, guard `tests/unit/test_memory_phase0_input_integrity.py`, re-proven by deletion 2026-09-28); its reversible-compression half stays in memory Phase 3. **#819's loud half as shipped:** a WARNING whenever an insert leaves the store over its cap (at the first overage and each doubling, reset under the cap), plus `stats()["over_capacity_inserts_this_process"]`; eviction choice unchanged. **#974 as shipped** (owner decision 2026-09-28): `config set` refuses a newer file; `maxim config downgrade` sets unknown settings aside in `config.preserved.json`; `maxim config restore-preserved` re-validates, diffs and restores on an interactive confirmation (injection review: escaped output, no `--yes`, security flags). **#932's non-orchestrator callers as shipped** (owner decisions 2026-09-28): `load.session` resolves an exact ID or path through `resolve_run_dir`, and a prefix must match one session (several raise); the API home defaults to `data_home()` and its agent moved to `agents/api_agent/` with a one-time, all-or-nothing migration; the research/campaign reports write under `<data home>/sim_reports/`. **#971 as shipped** (owner decisions 2026-09-28): the save guard on NAc, EC, SCN, AngularGyrus and the cross-layer graph; an unreadable file kept as a copy and saved over, uniformly (SCN's pathless case retired); an unreadable EC resets its NAc; every write-but-don't-read overwrite declared; every automatic save logs a refusal at ERROR. **#972 as shipped:** write-but-don't-read holds at session start for the stores in the agent's home (`build_memory_hub(load_persisted=)`, now required, passed by both builders): the ATL, AngularGyrus and cross-layer graph. Follow-ups: [#984](https://github.com/dennys246/Maxim/issues/984) (shared learned state outside any home), [#985](https://github.com/dennys246/Maxim/issues/985) (`create_npc_agent` over an existing home). **#976 as shipped:** the grounder updates each typed relation once, in its stored direction (a symmetric one moved by two deltas per pass; an incoming non-symmetric one was never grounded); its trigger walk discharged the three rows structurally (behavioral_graduation_candidates.md, 2026-09-29). **#818 as shipped:** wall-clock decay-on-load exempts inherent-class cluster biases, as the per-tick decay did; the trigger walk (Exp 45's `Re-run on:` fires by wording) discharged every row that persists NAc: no recorded run can hold an inherent key (behavioral_graduation_candidates.md, 2026-09-29). Follow-ups: [#988](https://github.com/dennys246/Maxim/issues/988), [#989](https://github.com/dennys246/Maxim/issues/989). **#843 as shipped** (owner decisions 2026-09-29): a non-outcome memory's success is unknown (`None`) in both directions (percepts were successes, observations failures); unknown reads as neither in the memory readers (`ImportanceBasedStrategy` 0.65); no migration of files on disk; double reinforce and the forming-pool key collision fixed. Exp 10 discharged structurally **pending the complete Exp 10 re-run (the 1.3.2 plan's item 2; `outstanding.md` O19), which covers it** (owner decision); the Exp 37 prompt-text note is recorded. **#991 as shipped** (#843's follow-up, landed before the Exp 10 re-run as O19 requires): an enrichment memory's valence and icon come from its own outcome; `EpisodicMemory.success` answers like a compressed record's, and every reader of either kind goes through `memory.types.record_success` (tool and goal: #995). Exp 37 fires on prompt text; Exp 10 does not fire (valence neither ranks nor filters) and its re-run covers the prompt change. Every recorded LLM-AUT run since 2026-04-20 showed every surfaced episode with the failure icon. |

*Best effort: batches 3–4.* These may move to a named plan path with a trigger, recorded in this plan
(where M15 sees it), not only on the issue.

| Batch | Issues | Notes |
|---|---|---|
| 3. Silent seams | [#851](https://github.com/dennys246/Maxim/issues/851) ⟲ `ToolPainBridge` pending entries leak and disable embodiment-pain attribution (the "SEM pain → NAc cascade" row: "ToolPainBridge attribution change"); [#845](https://github.com/dennys246/Maxim/issues/845) memory consumers that never deliver: items 2, 3 and 5 (5 verified first), and item 4's `exec_agent.py::recall_deep` site (its `plan_manager` site is #841's); item 1 rides the slice — **per consumer, wire it or mark it Dormant, each with its behaviour tier declared**; [#863](https://github.com/dennys246/Maxim/issues/863) step 2, the telemetry wraps catching caller logic, **outside** `agent_loop.py`, `orchestrator.py` and `_main_impl`; **added 2026-10-06 (owner):** [#1138](https://github.com/dennys246/Maxim/issues/1138) `EpisodicRecallSource.recalled_items` ignores its limit (live since #1129; first, beside #1140) and [#1128](https://github.com/dennys246/Maxim/issues/1128) the Dormant `MemoryAgent` queries bump `access_count` every tick (default retention until memory Phase 5; one owner call, stop vs uncounted; its own ledger walk) | The #840/#841 class, beside item 4's mypy extension, which catches more of them. Session C order (owner, 2026-10-06): #1138 → simulation honesty (D49, D46/D50) → #1128 → [#1124](https://github.com/dennys246/Maxim/issues/1124) → #1077–#1079, #1081. |
| 4. Body defects | [#873](https://github.com/dennys246/Maxim/issues/873) ⟲ the no-silent-fallback half: `damage_component` fails on a missing part instead of reporting success (row 9); [#874](https://github.com/dennys246/Maxim/issues/874) ⟲ cradle heat never reaches the arm (row 9's "Cradle / drive / SEM body change"; probably discharged structurally, since row 9 is the dragon / `base_humanoid` setup, but stated in the PR) | Both are "reports success, did nothing". #873's four design points (sum vs weighted mean, a missing part, partless bodies, archetype reflex sets) stay with [deferred/reflex_layering.md](deferred/reflex_layering.md). *(2026-10-18: #873's no-silent-fallback half shipped in 1.3.2 (2026-10-04; ledger T3-9's note); the issue stays open for those design points, unhomed until reflex_layering.md revives. Proposed 2026-10-18, owner to confirm.)* **Added 2026-10-06 (owner):** [#1124](https://github.com/dennys246/Maxim/issues/1124) (#874's review) a dotted ROOT key shadows the real modulator sub-sensor in `evaluate_failures` (reloaded pre-#874 orphans; the derived `<mod>.integrity` keys); one design call, moving derived integrity off `vital_metrics`. **Decided 2026-10-07 (owner), fixed with the reload defect it sat on** (a reload emptied every modulator, so integrity froze and a missing trigger field read 0.0): integrity derived on read, Entity format 1.1, dotted keys dropped on load, a trigger with no reading does not fire. Follow-ups #1155, #1156. |

*With the slice that owns the file*, each in its own PR. A fix measured on a loop mid-refactor cannot be
told apart from the refactor.

- **With the `agent_loop` slices:** phase 1 (slices 0–5) closed 2026-10-10 with #1210 (§The decomposition).
  #963 and #1125 landed with it (below); #835, #850, #845 item 1, #965 and #863's `agent_loop.py` sites did not, and
  stay open with the loop's later slices, which are not in 1.3.2, per "Done when" below. **By owner decision
  (2026-10-15) each is a 1.4.0 release-gate item** ([roadmap_1_4.md](roadmap_1_4.md) T0,
  `scripts/release_gates.json`): #835, #850, #845 item 1, #965, and #863's four innermost `loop_perception` wraps
  (slice 5 moved them out of `agent_loop.py`).
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
  - **[MERGED (#1164)]** [#1125](https://github.com/dennys246/Maxim/issues/1125) (#874's review): `Executor._drive_pressure_snapshot` misses modulator-qualified drives
    (`arms.thermal`, `head.thermal`, `arms.pressure`); with the slice that moves `_read_drive_ranges`.
    Record-only. Handed to Session A 2026-10-06.
  - [#863](https://github.com/dennys246/Maxim/issues/863): its `agent_loop.py` sites. Its `cli.py::_main_impl`
    sites go with that slice if it lands; if not, they stay open on #863 with this home recorded.
- **After the orchestrator's characterization tests** (the first slice of its decomposition; **landed as #1269,
  2026-10-14**). None of the three below has landed. The remaining orchestrator slices are not in 1.3.2, and by owner decision
  (2026-10-15) they hard-gate 1.4.0 (§The decomposition), so each stays open on its issue, lands with or after the
  slice that moves its code, and is a 1.4.0 gate item: #932's lookup (`_load_resume_context_at`) sits in §9, the
  campaign dispatch slice; #1123's three parts (the `health` hint, the range text, the `value` default) sit in §5,
  the `build_orchestrator` slice; and #863's three DROP-class orchestrator SEM_TRACE wraps widen #863 beyond its
  own scan by the same decision.
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
  - If they do not fit in 1.3.2, they keep their home in [outstanding.md](outstanding.md) O12. They did not fit;
    re-homed to 1.3.5 with the orchestrator slices (§1.3.5; owner decision 2026-10-18).

*1.4, by design.* Each needs an experiment or a mechanism review, not a hardening fix. Each is linked
from the plan that owns it:

- [#922](https://github.com/dennys246/Maxim/issues/922), [#834](https://github.com/dennys246/Maxim/issues/834)
  and #832 item 2 → [roadmap_1_4.md](roadmap_1_4.md) §Before 1.4.
- [#880](https://github.com/dennys246/Maxim/issues/880) → [deferred/nociception_layer.md](deferred/nociception_layer.md)
  (its F1).
- [#848](https://github.com/dennys246/Maxim/issues/848) → memory 2S-e.
- [#899](https://github.com/dennys246/Maxim/issues/899) → roadmap 1.4 Phase 5 keying.
- [#784](https://github.com/dennys246/Maxim/issues/784) → [archive/world_channel_weighting.md](archive/world_channel_weighting.md).
- [#1137](https://github.com/dennys246/Maxim/issues/1137) (compression keeps only the intent goal; a persisted-shape change) → the memory line
  ([memory_strength_and_forgetting.md](memory_strength_and_forgetting.md)). Trigger: the first path that compresses
  in a sim, or any work on sleep/consolidation in the loop (no committed store holds a compressed record today).
  Release home: §After 1.4.0, beside #816 (proposed 2026-10-18, owner to confirm).
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

**Release-readiness audit, 2026-10-14: owner decisions and recorded moves.** A read-only audit of `main` @ 26fcaf1b
found 1.3.2 not ready. The owner decided:
- **T1-2 (Exp 37) stays STALE, and 1.3.2 releases without its re-fire (owner, 2026-10-14; this replaces the
  same day's first decision, "the release WAITS for its re-run").** The exception is committed and
  machine-readable: `release-1.3.2-T1-2` in
  [`evidence_exceptions.json`](../experiments/evidence_exceptions.json), a `release` clause that names the row,
  its STALE status and 1.3.2 only (M25's form; `lint_evidence_gate.py` checks its shape). It moves no status:
  only a run can. Why the strict default was not kept:
  - The row carries no EARNED claim: its cross-session override claim was pulled in June. The 1.3.1 walk
    released over a fired trigger on that ground, but the row was PARTIAL then. This is the first release over a
    STALE row, which is why it takes a committed exception.
  - A re-fire now would most likely be INCONCLUSIVE. Arm A's baseline moved from 0.42 to 0.71–0.75 on the same
    commit ([L8](../limits/README.md), BINDING), so the June gate has no headroom left to measure.
  - The prerequisite [#1226](https://github.com/dennys246/Maxim/issues/1226) is parked. Exp 37's cradle goals
    run the synchronous generative campaign, which never consumes the narrator resume prompt, so the narrator
    already starts fresh there.

  **The owed re-fire becomes L8's dose-response design in 1.4**: baseline-floors Stage B4
  ([baseline_floors.md](baseline_floors.md)), with a four-lens design review before the harness. The
  cross-model Goldilocks map stays cited only with L8's caveat (unverifiable, not refuted). The 1.3.2 release
  notes name the exception.
- **Batch 1's review findings [#966](https://github.com/dennys246/Maxim/issues/966),
  [#967](https://github.com/dennys246/Maxim/issues/967) and [#968](https://github.com/dennys246/Maxim/issues/968)
  are fixed in 1.3.2** (Session C, one PR; MERGED as [#1282](https://github.com/dennys246/Maxim/pull/1282),
  2026-10-14).
  - #967 FAILS CLOSED: a robots.txt that cannot be read (5xx, 429, timeout, network) refuses the fetch.
  - Following RFC 9309: any other 4xx, 401/403 included, means no restrictions; redirects are followed for up to
    five hops; the robots.txt is the URL's own origin's.
- **Moved, so batches 0–3 read true:**
  - [#954](https://github.com/dennys246/Maxim/issues/954), the scripted water-trial tests still on wall time
    (batch 0's timing class): moved to 1.3.3, Session B, CI truth.
  - [#984](https://github.com/dennys246/Maxim/issues/984), [#985](https://github.com/dennys246/Maxim/issues/985),
    [#988](https://github.com/dennys246/Maxim/issues/988) and [#989](https://github.com/dennys246/Maxim/issues/989),
    batch 2's follow-ups: moved to 1.3.3, Session C.
  - [#1291](https://github.com/dennys246/Maxim/issues/1291), a sandbox app's /ws client can opt in to another
    app's health stream in one process (found by F3's review, predates it): 1.3.3, Session C.
  - [#1285](https://github.com/dennys246/Maxim/issues/1285), [#1286](https://github.com/dennys246/Maxim/issues/1286)
    and [#1287](https://github.com/dennys246/Maxim/issues/1287), the #1268 review's NITs (the heartbeat
    sampler's ownership, the scrub on the failing thread, one `HealthLog` shared across console tests): filed
    2026-10-14, 1.3.3, Session C.
  - [#1292](https://github.com/dennys246/Maxim/issues/1292), F4: a `console.device.build_device_app` facade that
    owns the device-app composition pulse assembles today (owner decision 2026-10-14: build in 1.3.3, after 1.3.2
    and pulse v0.4.1 ship, design review first), Session C. *Moved 2026-10-24: F4 is the embodied-device plan's
    `run_device` ([embodied_device.md](embodied_device.md)).* F2 and F3, the other #1268 follow-ups, shipped in
    1.3.2 ([#1290](https://github.com/dennys246/Maxim/pull/1290), [#1293](https://github.com/dennys246/Maxim/pull/1293)).
  - [#863](https://github.com/dennys246/Maxim/issues/863) step 2, the sites outside the loop (`orchestrator.py` and
    `cli.py::_main_impl`): moved with those slices. It stays open on #863, home recorded.

**Done when:**
- Batches 0–2 are closed.
- Each of batches 3–4 is closed, or moved as recorded above.
- The slice-bound items have landed with their slices, or stay open with them (each still open is a 1.4.0
  release-gate item, owner decision 2026-10-15, [roadmap_1_4.md](roadmap_1_4.md) T0).
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
Release-governance gap; it also takes GPG off the release path. *(2026-10-18: the tag-triggered trusted-publishing
workflow and the `v*` ruleset are [#1372](https://github.com/dennys246/Maxim/issues/1372), the 1.3.3 exit; #938 is only
the nightly-red window between a release merge and its tag. Both are 1.3.3 work, §1.3.3.)*

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
   merged as #1210: 2,717 → 2,429 lines; the same PR applied rule (c) below.)*
   **Phase 1 COMPLETE 2026-10-10:** slices 0–5 merged as #1114, #1127, #1136, #1157, #1187 and #1210, with
   #1133 (#1151), #963 (#1198) and #1125 (#1164) landed alongside. `run_agentic_loop` is pinned at 2,434 lines on
   `main` today (the function-length baseline; later fixes moved it from slice 5's 2,429). Further `agent_loop`
   slices (phase 2) are not in 1.3.2. **By owner decision (2026-10-15) they hard-gate 1.4.0**, with the
   loop-bound issues still open (§Issue burn-down): `run_agentic_loop` and `execute_and_learn` (392 lines today,
   split out by #1133; its own split is #1150) must each reach 1,000 lines or fewer (owner decision 2026-10-16, #1323; it was 200), as must everything split out
   of them (§The 1.4.0 gate, below).
2. **`start_simulation_mode`** (`simulation/orchestrator.py`, 3,322). Its tests cover **11%** of its lines
   (the Codex card's measurement at `v1.3.1`), so it is NOT decomposed blind: characterization tests
   first, then an orchestrator coverage floor set from them (item 5 above), then slices under the same
   gates below. The coverage ratchet is what makes this decomposition safe.
   *(Slice 0 merged as #1269, 2026-10-14: the characterization golden
   (`tests/unit/test_orchestrator_characterization_golden.py`, driver `tests/unit/_orch_harness.py`, three arms),
   then one verbatim move, the TTY loops (`stdin_reader_raw`, `read_post_sim_goal_raw`) to
   `simulation/tty_stdin.py` (whole-file coverage exclusion, ref #947): 3,227 → 3,003 lines (3,004 after #1289's one line; **2,889 on `main` today**, after #1299 moved §5's 14
   narrator-only descriptions to `simulation/tools.py::NARRATOR_TOOL_DESCRIPTIONS`). `orchestrator.py`'s per-file coverage floor (#1266, merged 2026-10-13) is pinned at **54.0%**
   from two distinct CI runs of the slice branch (`4e1d403c`; spread 0), taken directly from the runs because
   `--merge-floors` falsely refuses two byte-identical deterministic runs as one run given twice
   ([#1271](https://github.com/dennys246/Maxim/issues/1271), open; the provenance gap behind it is M52).
   The defects slice 0 surfaced are pinned as current behaviour, each fixed as a reviewed golden change:
   #1261 merged as #1289: the narrator reads the AUT's words through the typed `ActionRecord.utterance`, and
   speech is declared on the tool (`Tool.speech_params`, or an affordance's YAML `speech_param`). #1260 merged as
   #1299 with #1263's DefaultNetwork part: the narrator's tool descriptions are a registry-scoped overlay
   (`SimToolRegistry.tool_descriptions()`), no longer written into the process-global `TOOL_DESCRIPTIONS`; the loop
   stops the DefaultNetwork it started, in sim mode too (the leak behind the CI flake #1281); and the golden was
   regenerated in its own commit (`feb7dd5a`). #1259, #1262, #1263's remainder, #1264, #1265 and #1267 stay open,
   each a 1.4.0 gate item.)*
   **The remaining orchestrator slices are not in 1.3.2, and they hard-gate 1.4.0 (owner decision
   2026-10-15).** They do not gate the 1.3.2 cut, where the function-length ratchet records the ceiling (Sizing,
   below); 1.4.0 cannot pass CI until `start_simulation_mode` (2,889 lines) reaches 1,000 or fewer, with
   everything split out of it (the bound was 200 until #1323, owner decision 2026-10-16) (§The 1.4.0 gate, below). Order, tail first: report/result (§14) → teardown (§13) →
   stall detector (§11) → stdin (§10) → campaign dispatch (§9) → `build_aut` → `build_orchestrator`. Each owes
   the characterization arms listed below before it moves.
3. **`_main_impl`** (`cli.py`, 1,696) — only if 1 and 2 land; it is CLI glue, the least valuable of the three.

The "kicked down the road" worry that kept this to one target is answered by the rule in **Sizing**: if a
slice stalls, 1.3.2 ships the slices that landed and the ratchet records the new ceiling. Since 2026-10-15 the
remainder is not an open-ended deferral: it gates 1.4.0 (below).

**The 1.4.0 gate (owner decision 2026-10-15).** The owner decided that the remaining decomposition is not in
1.3.2 and that it HARD-GATES 1.4.0. It is [roadmap_1_4.md](roadmap_1_4.md) T0, enforced by
`scripts/lint_release_gates.py` over `scripts/release_gates.json` (PR #1300; CI's lint job plus a runtime twin in
unit-tests), which fails every version at or above 1.4.0 until three things hold:
- `start_simulation_mode` (2,889 lines today), `run_agentic_loop` (2,434) and `execute_and_learn` (392) are each
  1,000 lines or fewer, and so is everything split or moved out of them and every def in their three files (item
  `1323.functions-1000`; owner decision 2026-10-16, #1323, retiring the earlier 200-line bound; 200 or ~500 is
  reassessed for 1.5 at the 1.4 release; their function-length pins stay, so none may grow).
- The five orchestrator characterization arms (`persistent_agent`, `imagination`, `new_goal`, `stall`,
  `generative`; §"Characterization owed" below) are complete in the golden, with direct-assertion evidence.
- 33 items (32, plus #1323's bound) are each satisfied by an evidence test that was once a strict red gate, or retired on the record: the
  slice-0 defects (#1259, #1262, #1263's six parts including the DefaultNetwork stop #1299 shipped, #1264, #1265,
  #1267); the folded issues (#1123's three parts, #932's resume lookup, #863's seven DROP-class sites: four
  `loop_perception` wraps and three orchestrator SEM_TRACE wraps); #1288's four name lists
  (`modes/definitions.py::CORE_TOOLS` is exempt as a mode-membership list); and the loop-bound #835, #850, #845
  item 1 and #965; and #1281's remainder after #1299 fixed its root cause (`1305.owned-rng`, owned generators in
  `default_network/`, `attention/`, `spatial/` and `decisions/significance.py`; `1263.thread-guard`, a conftest
  guard with an empty allowlist; added 2026-10-15 by owner decision).

What only review can catch around it, including that an arm lands before the slice that moves its code, is
mechanization backlog M56 ([outstanding.md](outstanding.md)).

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
dirty `src/`), never by pasting a slice's output; the check that would enforce it is mechanization backlog M53 ([outstanding.md](outstanding.md)). A source pin that says "the loop's code contains
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

**Characterization owed before the orchestrator's report, teardown, stall and campaign-dispatch slices** (the
review of orchestrator slice 0, 2026-10-13; the campaign-dispatch arm added 2026-10-15). All five arms
below are 1.4.0 gate arms (§The 1.4.0 gate): the lint checks that each is complete, but that each lands BEFORE the
slice that moves its code is enforced by that slice's review and this paragraph, not mechanically (M56).
`tests/unit/test_orchestrator_characterization_golden.py` (driver:
`tests/unit/_orch_harness.py`) ran three arms at slice 0, all on a fresh throwaway AUT. **Status 2026-10-24: all five arms below have landed** (Session A, branch `test/orch-characterization-arms`; `lint_release_gates`: arms 5/5), each with its deletion probes. Three follow-ups, owed before the slices they name: (1) the **stall arm** reaches the idle threshold only through the post-finish nudge, the #1259 path, because since D13 a live narrator never idles (#1452); #1259's fix removes that path and must re-home the arm in the same PR (the remaining offline route is a wedged narrator call; the harness docstring, arm 5, names it), because the arm's scope is immutable; (2) before the **stdin slice**, turn on the opt-in `events` / `held_inputs` trace keys for the slice-0 `stdin` and `narrator` arms in their own regeneration commit: today they do not record which thread and source each operator percept goes through, and #1262 keeps those percepts out of `turns` and `narrator_calls`. (3) **#1488**: the stall arm runs at a 15 s idle floor (was the 5 s clamp floor) because a nudge that landed in the narrator's `observe_actions` → `finish` hand-off on a loaded CI runner once sat unplanned until the next nudge woke the loop; that strand's cause is open (unloaded probes never strand it; the golden test prints the agent loops' log with a mismatch). The **§11 slice** injects the detector's clock so the arm reaches its nudge by construction, and either answers #1488 or pins the strand as a strict red gate. Before the slice that moves §13
teardown or §14 report/save, it needs an **adopted `persistent_agent` arm** (the HANDLE seam's branches 3 and 4:
no session AUT snapshot, the tool-lease restore, the DefaultNetwork pain-bridge unsubscribe), an
**imagination-enabled arm** (§7) and a **`/new` arm** (the post-sim reader faked to return a goal, so the run
re-enters `start_simulation_mode`). Before the slice that moves §11, it needs a **stall-detector arm** that reaches
the detector's injections; today's arms stay under its first diversity checkpoint and its 30 s idle threshold on
purpose. Before the slice that moves §9 (campaign dispatch), it needs a **generative-campaign arm**
(`generative=True`). Session C found that the generative campaign runs synchronously, inside §9, and its
`_finish_runner` sets `stop_event` BEFORE §9's resume/kickoff injection (`orchestrator_source.inject_cli`), so on
that path the narrator never consumes the resume prompt (the reason #1226 is parked, §Issue burn-down). The arm
pins that ordering, so the §9 slice cannot reorder it silently; changing it is a fix with its own reviewed golden
change. Already pinned by slice 0's review fold: the report's provenance by value (only the keys that name the
checkout and the interpreter are masked, by name), the whole `MAXIM_DATA_HOME` file set (the orchestrator's own
`shutdown()` stores included; `agents/*/runtime/` excluded for #1264), the threads a run leaves alive (#1263,
pinned as current behaviour; #1299 regenerated the golden as a reviewed change in its own commit, `feb7dd5a`:
`DefaultNetwork` and `DN-BackgroundTasks` left the leaked-thread set, and `6238109d` stopped pinning the
order-dependent memctx pool),
the saved AUT NAc's causal links and the `actions.jsonl` records, and what each
report writer is handed.
**The `orchestrator.py` per-file coverage floor** was pinned from CI's measurement on the slice-0 branch with
#1266 (per-file floors with the split rule; merged to main 2026-10-13) on it, not from a local run: **54.0%**
(`4e1d403c`, two distinct CI runs; see item 2 above for #1271). Later slices move it under #1266's split
rule.

**Typing rides along, scoped:** every module the decomposition creates enters CI's mypy set. The
repo-wide ratchet from 1.3.1 holds the rest. Full coverage is not promised.

**Close it honestly:** the complete Exp 10 re-run (item 2 above), a trigger walk over the ledger (Exp 60 and Exp 61 both name
`run_agentic_loop`'s idle-gate and autonomy handling) and a live re-run of Exp 60 on the rig
(5 seeds per arm, ~1 h) to discharge them with a dated annotation, rather than an argument that a
pure extraction changes nothing.
**The live Exp 60 re-run is DONE:** T1-13 MAINTAINED 2026-10-12 (Exp 60 Amendments 8–9, run at the code pin
`baf0e15f`, all five frozen gates PASS; data #1251, ledger move #1253), re-validating the row AT THE PIN only. The
post-pin trigger walk (#1255) covers every PR merged after the pin through #1253: no re-run owed. Session A's PRs
after #1253 (#1269, #1266, #1289, #1299 and #1300) are walked on T1-13's row (2026-10-15): no re-run owed. The
other lanes' PRs after #1253 are walked at the cut. (T1-13 has since moved to PARTIAL 2026-10-14, narrow: floor matched, through the
baseline-floors line's B1 run, #1283; the FEAR-vs-ABLATED contrast this re-run re-validated stands.)

**Sizing, honestly:** this is the largest item on the list — several sittings of work in reviewed
slices, plus the rig hour. If a slice stalls, the release ships the slices that landed; the ratchet
records the new ceiling either way. For 1.3.2 that is what happened, and by owner decision (2026-10-15) the
remainder hard-gates 1.4.0 (§The 1.4.0 gate).

---

## Path to 1.4.0 (owner decision 2026-10-16)

1.3.2 "Decomposition" shipped 2026-10-16. 1.4.0 is hard-gated by `scripts/release_gates.json` and
`scripts/lint_release_gates.py` (#1300). At the cut that gate had 31 of 32 items open (33 items, 32 open, as of 2026-10-18), none of its five
characterization arms recorded, and three functions far over its then 200-line limit (1,000 since #1323): `start_simulation_mode` (2,889
lines), `run_agentic_loop` (2,434) and `tool_dispatch.execute_and_learn` (392). The owner chose **patch releases, then 1.4.0**
(three on 2026-10-16; four since 2026-10-17, when 1.3.3 was split so the atonia gate ships as its own 1.3.4), so the
decomposition ships in steps. Each step is a release users get, and the repo keeps
improving on the way. A six-reader review of `docs/` and `docs/plans/` (2026-10-16) produced the scopes below. Every
open issue that had no home is placed here (M15).

Owner decisions taken with this path (2026-10-16):
- **E1 is redesigned before its harness.** As drafted (roadmap_1_4.md §Phase 2), its argmax is fixed by protocol
  constants, so a same-prior floor learner matches it and it cannot EARN under BF2 (M46). E1's four-lens design
  review starts from one question: what contingency can a reasonable no-bio floor lose (delayed or multi-step
  credit, a learning curve rather than a fixed argmax)?
- **1.3.3 is split (2026-10-17).** 1.3.3 ships the release pipeline and pulse's blockers fast. The atonia gate becomes
  1.3.4, on its own clock. The orchestrator becomes 1.3.5, and loop phase 2 becomes 1.3.6.
- **1.3.4 becomes the robot and recall fixes (2026-10-24).** The atonia gate keeps its own clock and takes the next
  free patch number when it is done. The labels "1.3.5" (the orchestrator) and "1.3.6" (loop phase 2) stay as the
  names of those lines, because other docs cite them; each release's actual number is assigned at its cut, in the
  order releases ship.
- **B3 runs Reflexion first** (2026-10-17), as one end-of-session reflection call. Episodic retrieval may follow as a
  second arm.
- **T1-2 (Exp 37) is resolved by B4 before 1.4.0.** The B4 re-fire (L8's dose-response design) runs after the
  orchestrator slices unblock B2. If it has not run, 1.4.0 waits. No second release exception is taken for 1.4.0.
  - **Amended 2026-10-18 (owner decision, M25 vs T1-2):** each patch release before it (1.3.3, 1.3.4, 1.3.5, 1.3.6)
    carries its own explicit, committed, machine-readable release exception for T1-2 in
    [`evidence_exceptions.json`](../experiments/evidence_exceptions.json) (the `release-1.3.2-T1-2` form; the
    weak-evidence rule: an override is a committed machine-readable exception), so M25 can refuse a STALE row without
    blocking the path. Writing that exception is each release session's job. "No second release exception" stays a
    1.4.0-only rule.

### 1.3.3, "The release pipeline and the pulse seam" (no `nac.py` selection or loop-body moves; split 2026-10-17)

> **Released as v1.3.3 "Release pipeline" (2026-10-24 UTC;
> [release_1_3_3.md](../announcements/release_1_3_3.md)).** The release ships: the release pipeline (signed `v*` tag,
> nightly-at-the-commit gate, exact-wheel smoke, owner approval, PyPI trusted publishing; #1372, #938, #1318); the
> pulse seam (#1332 + R1, #1347 + R2's tts and console cells, R4 with pulse v0.4.2 at contract 0.6.0, #1411–#1414);
> the security follow-ups #1311, #1312, #1314, #957, #997, #1345; Session H's #1337, #1310, #1343; the NAc audit
> (#1102, #1361 S1, A1, A2); and the agent-id fix from the pre-release review. Breaking in places, a patch release
> by owner decision (2026-10-23). T1-2 ships STALE under `release-1.3.3-T1-2`. What did not finish **slipped to
> 1.3.4** (owner decision 2026-10-23), each still open with its home:
> - R3 typed payloads [#1336](https://github.com/dennys246/Maxim/issues/1336) and F4
>   [#1292](https://github.com/dennys246/Maxim/issues/1292): §1.3.4, Session S's second slice (F4 moved 2026-10-24 to
>   [embodied_device.md](embodied_device.md) as its `run_device`);
> - doctor advisories [#1295](https://github.com/dennys246/Maxim/issues/1295) (owner decision 2026-10-18; recorded
>   here, not only in memory);
> - CI and evidence truth: #1313, #1203, #1121, #1218, #1220, #1246, #1247, #1038, #1024, #1031, #1215, #1023;
> - the 1.4.0 gate items outside the large functions: #1305, #1263 (with #1016), #1264, #1265, #1267, and #1123's
>   and #1288's lists. They still hold 1.4.0;
> - small correctness fixes #982, #1153, #1155, #1163, #1225, and #1369, #1370, #1371;
> - the recorded Session C moves #954, #984, #985, #988, #989, #1285, #1286, #1287, #1291, and
>   console_tunnel_hardening's PR 3 (admission control);
> - the node_security remainder, fail-loud Stage 3 and M25 (the STALE/BROKEN-row release check; until it is
>   built, the release session checks the ledger by hand and writes the `kind: "release"` clause).
- **The release pipeline:** [#1372](https://github.com/dennys246/Maxim/issues/1372) (a tag-triggered,
  trusted-publishing workflow and a `v*` tag ruleset), [#938](https://github.com/dennys246/Maxim/issues/938) (the
  release gate refuses inside the nightly it waits for, so `main` reads red between a release merge and its tag), CI caching [#1318](https://github.com/dennys246/Maxim/issues/1318). Owner standard 2026-10-16: no Test PyPI;
  the exact built wheel is smoke-tested in a clean venv, then published.
- **Security follow-ups:** [#1311](https://github.com/dennys246/Maxim/issues/1311), [#1312](https://github.com/dennys246/Maxim/issues/1312), [#1314](https://github.com/dennys246/Maxim/issues/1314), [#957](https://github.com/dennys246/Maxim/issues/957), [#997](https://github.com/dennys246/Maxim/issues/997), [#1295](https://github.com/dennys246/Maxim/issues/1295). The design work [#1315](https://github.com/dennys246/Maxim/issues/1315) ships as 1.3.4, the atonia gate (advisor mode, below; §1.3.4).
- **CI and evidence truth:** [#1313](https://github.com/dennys246/Maxim/issues/1313), [#1203](https://github.com/dennys246/Maxim/issues/1203), [#1121](https://github.com/dennys246/Maxim/issues/1121), [#1218](https://github.com/dennys246/Maxim/issues/1218), [#1220](https://github.com/dennys246/Maxim/issues/1220), [#1246](https://github.com/dennys246/Maxim/issues/1246), [#1247](https://github.com/dennys246/Maxim/issues/1247), [#1038](https://github.com/dennys246/Maxim/issues/1038), [#1024](https://github.com/dennys246/Maxim/issues/1024), [#1031](https://github.com/dennys246/Maxim/issues/1031), [#1215](https://github.com/dennys246/Maxim/issues/1215), [#1023](https://github.com/dennys246/Maxim/issues/1023).
- **1.4.0 gate items outside the large functions:** [#1305](https://github.com/dennys246/Maxim/issues/1305) (owned RNGs), [#1263](https://github.com/dennys246/Maxim/issues/1263) (the thread guard and its
  parts outside `start_simulation_mode`; [#1016](https://github.com/dennys246/Maxim/issues/1016) folds into it), [#1264](https://github.com/dennys246/Maxim/issues/1264), [#1265](https://github.com/dennys246/Maxim/issues/1265), [#1267](https://github.com/dennys246/Maxim/issues/1267); plus #1123's three items and #1288's console and
  exec_agent lists (homed in §1.3.2). `932.resume-sim` is NOT here: it lives in the fenced `orchestrator.py`, so it
  lands in 1.3.5 with the orchestrator slices (owner decision 2026-10-18; §1.3.5).
- **Small correctness fixes:** [#982](https://github.com/dennys246/Maxim/issues/982), [#1153](https://github.com/dennys246/Maxim/issues/1153), [#1155](https://github.com/dennys246/Maxim/issues/1155), [#1163](https://github.com/dennys246/Maxim/issues/1163), [#1225](https://github.com/dennys246/Maxim/issues/1225); the benchmark label [#1310](https://github.com/dennys246/Maxim/issues/1310) (its open question,
  whether `passed` should require the expectations, is the owner's).
- **Recorded 1.3.3 moves** (release-readiness block above): #954, #984, #985, #988, #989, #1285–#1287, #1291,
  #1292 (F4, the device-app facade, after pulse v0.4.1), and console_tunnel_hardening's PR 3 (admission control),
  whose revive trigger fired.
- **Backlog:** M25 (the STALE/BROKEN-row release check that reads `kind: "release"` exceptions). While T1-2 stays
  STALE, M25 refuses each patch release unless it carries its own committed `kind: "release"` exception for T1-2
  (owner decision 2026-10-18, §Path to 1.4.0); writing that exception is each release session's job.
- **Deferred plans whose triggers fired** (owner decisions 2026-10-17):
  - fail-loud Stage 3 ([deferred/measurement_path_fail_loud.md](deferred/measurement_path_fail_loud.md)), scheduled for
    1.3.2 and not shipped there;
  - the [node_security_simplification.md](node_security_simplification.md) remainder: drop the
    `leader_mode` "client" role and the `role.py` `role_divergence` shim (both marked "drops in 1.2" in the code), and
    write the missing auth-model doc.
- **Plans:** E1's redesigned prereg goes to its four-lens design review; it is paper only, because the fence still
  holds: the fence lifts when 1.3.6's decomposition slices land — the point at which T0 can pass.
- **The pulse seam** (maxim-pulse's test of the 1.3.2 wheel, verified 2026-10-17;
  [readiness_and_extras_standards.md](readiness_and_extras_standards.md), standards R1–R4). These block pulse #42,
  v0.4.2:
  - **[#1332](https://github.com/dennys246/Maxim/issues/1332): spoken-code pairing is silent** on every installable
    piper-tts. *Shipped in [#1381](https://github.com/dennys246/Maxim/pull/1381) (merged 2026-10-19), with R1's first
    instance and R2's tts cells.*
  - **[#1347](https://github.com/dennys246/Maxim/issues/1347): the `console` extra cannot serve `/ws`.** Plain
    `uvicorn` has no WebSocket library, so a clean `pymaxim[console]` install gives the Console no live stream. The
    fix declares `websockets` in the extra (owner decision 2026-10-18: not `uvicorn[standard]`, whose uvloop and
    httptools uvicorn would auto-select), guarded by an R2 test: the extra alone upgrades `/ws` against a real server.
    *In [#1388](https://github.com/dennys246/Maxim/pull/1388), with R2's console cells and the `/ws` shutdown fix
    (`maxim serve` hung up to 15 s after a Console tab closed).*
  - **[#1333](https://github.com/dennys246/Maxim/issues/1333): the wheel vendored console v0.4.0.** Vendor v0.4.1, and
    make the release check the vendored tag (R4, a gate: design pass first).
  - **The standards, first slice each:**
    - [#1334](https://github.com/dennys246/Maxim/issues/1334) (R1): readiness proves the capability, starting with
      `pairing_ready`;
    - [#1335](https://github.com/dennys246/Maxim/issues/1335) (R2): extras tested against the real library, tts
      first;
    - [#1336](https://github.com/dennys246/Maxim/issues/1336) (R3): typed console payloads. **Slipped to 1.3.4**
      (owner decision 2026-10-18): 1.3.3 vendors pulse v0.4.1 at contract 0.6.0. *Amended 2026-10-22 (owner
      decision): 1.3.3 vendors pulse **v0.4.2** (signed tag, commit bcd0d1e; pulse #52 hides the Console's inert
      Models button), same contract 0.6.0; pulse tags nothing further until 1.3.3 is on PyPI.*
  - **Cut rule:** R3's contract 0.7.0 lands only together with a pulse tag built for 0.7.0. Otherwise it slips past
    the cut, and the release vendors v0.4.2 (amended 2026-10-22, was v0.4.1) at contract 0.6.0. Since R4 the bump itself merges, and `main` is then unreleasable until the console pin moves to a
    0.7.0 pulse tag; at the cut, revert the bump or wait ([archive/r4_console_vendor_gate.md](archive/r4_console_vendor_gate.md)).
  - **The 1.3.3 release checklist for the pulse seam** (Session S's code is done once #1388 merges; these are what
    is left, in order):
    - [x] **#1388 merges** once `scripts/pr_merge_readiness.py 1388` exits 0.
    - [x] **The first nightly after it is green in all four `extras-matrix` cells** (console and tts, each at floor and
      latest). They are `(nightly)` jobs, so `scripts/check_nightlies.py` refuses the cut while one is red. A PR runs
      only the cells its diff touches; the nightly runs every one, so its green is the release evidence. A red cell is
      an upstream break (cap the requirement or fix the code) or a lane bug; never an override.
    - [x] **Pulse's two checks against the REAL 1.3.3 candidate wheel**, which also carries #1383's vendored bundle.
      Both passed 2026-10-19 against a wheel built from #1388's branch (E2E 59/59 in Chrome; SMOKE 7/7 in an aarch64
      container, with PR #42's Reachy app); re-run them on the cut:
      - pulse's `scripts/smoke-reachy-release.sh` on Linux prints `SMOKE PASSED`;
      - pulse's browser suite `scripts/e2e.sh` prints `E2E PASSED (59/59)`.
      Run them from a fresh clone of pulse with `REACHY_APP` set to PR #42's app until #42 merges (pulse `main` does
      not wire the voice provisioner yet). *Done 2026-10-23: candidate 2 (rehearsal run 39902958693 on `a5e1030b`)
      passed pulse's release acceptance 5/5 and its live check of #1411, #1412 and #1414; pulse: "1.3.3 is good to cut".*
    - [ ] **The note to pulse at the release** (with the M70 rule): the vendored tag (v0.4.2, contract 0.6.0; amended 2026-10-22); no
      untyped `data` field changed from Session S; `pairing_ready` now means the voice has SPOKEN audible audio once,
      not that it loaded (R1); pulse's `apps/reachy/maxim_reachy_app/app.py` comment "until the voice loads" should
      say "speaks"; the `console` extra now installs `websockets`; and #42 can pin `>=1.3.3` and ship as v0.4.2.
- **Advisor mode** ([#1315](https://github.com/dennys246/Maxim/issues/1315); [advisor_mode.md](advisor_mode.md),
  verified against v1.3.2 and reviewed 2026-10-17). It moved here from §After 1.4.0 at an external request; it is
  planned here and ships as 1.3.4 (below).
  - Owner decisions AD1–AD9 were made 2026-10-17, all as recommended. AD2 is the **atonia gate**: motor output
    is inhibited below cognition and fixed at construction, while the report stays open. AD8 grants the fence
    exception below. AD9: `maxim.advisor` ships Experimental, with its sandbox-only channels stated.
  - Items:
    - 2: `Tool.effect` and the registry ceiling;
    - 3: a required `ToolPolicy` on every executor;
    - 6: fail loud on readiness, [#1338](https://github.com/dennys246/Maxim/issues/1338);
    - 1: the closed advisor builder with a sealed registry, plus a minimal proposal log (5).
  - **AvServ plan 50** (2026-10-17) adds:
    - item 9, a bounded yield on SIGTERM ([#1344](https://github.com/dennys246/Maxim/issues/1344));
    - item 10, offline startup and runtime;
    - item 11, `maxim doctor --advisor`;
    - item 12, a cursor the caller owns;
    - items 13 and 14, the SAR lead schema and cell sweeps (Profile A).
  - **These items ship as 1.3.4, not in 1.3.3** (owner decision 2026-10-17; §1.3.4 below). Item 9's process-level
    yield may land in 1.3.3 if Session H finishes it first.
  - Profile A's Phase 1 runs on 1.3.2 through a verified harness, with conditions (advisor_mode.md §Phase 1 on 1.3.2).
- **Found while verifying plan 50:**
  - [#1342](https://github.com/dennys246/Maxim/issues/1342): `run()` writes under the CWD, and prefetch reads it
    outside the registry (V);
  - [#1343](https://github.com/dennys246/Maxim/issues/1343): the CLI agentic path downloads with no consent gate (H);
  - [#1345](https://github.com/dennys246/Maxim/issues/1345): an agent can overwrite the operator's persisted
    internet-off (H).
- **Found while verifying:** [#1337](https://github.com/dennys246/Maxim/issues/1337): `--clear-memory` resolves
  against `./data` and misses `sessions/`.
- **Exit:** [#1372](https://github.com/dennys246/Maxim/issues/1372) closed and 1.3.3 published through the new
  pipeline (re-pointed 2026-10-18 from #938, which is only the nightly-red window); the gate's open-item count down by
  the items above.

- **From the 2026-10-17 CI and NAc audits** (owner decisions): [#1362](https://github.com/dennys246/Maxim/issues/1362) (refuse `:` in agent ids at `create`/`load`) goes to Session V, which owns `create.py`; it ships with 1.3.4 if V's work does (V's atonia items are 1.3.4), or in 1.3.3 if it lands before that cut. The pytest-xdist re-land (#1339, reverted by #1359) is **not** in 1.3.3: it waits until after the 1.3.3 release, on M60 (a coverage ratchet that tolerates the measured test-order noise, up to 16 statements per package from file order alone) and the determinism fixes it surfaced (#1349, #1356, #1360, all merged). The CI-tooling issues ride with it after the 1.3.3 release (proposed 2026-10-18, owner to confirm): [#1325](https://github.com/dennys246/Maxim/issues/1325) (prepr resolves SSH host-alias remotes), [#1340](https://github.com/dennys246/Maxim/issues/1340) (`lint_coverage` rejects chained commands after `CI_CLEAN`), [#1341](https://github.com/dennys246/Maxim/issues/1341) (`lint_coverage` banned env vars via `$GITHUB_ENV`) and [#1271](https://github.com/dennys246/Maxim/issues/1271) (`lint_coverage --merge-floors` refuses two distinct deterministic runs).
- **Orphans homed 2026-10-18** (proposed 2026-10-18, owner to confirm):
  - [#1369](https://github.com/dennys246/Maxim/issues/1369) (a request-shaped 400 cools down a healthy provider),
    [#1370](https://github.com/dennys246/Maxim/issues/1370) (a mid-stream error chunk returned as a successful
    completion) and [#1371](https://github.com/dennys246/Maxim/issues/1371) (a stale env API key beats a rotated
    config key), from the vLLM draft ([deferred/vllm_self_hosted_servers.md](deferred/vllm_self_hosted_servers.md)):
    Session H in 1.3.3 if ready, otherwise the next patch;
  - [#1366](https://github.com/dennys246/Maxim/issues/1366) (a learns-only agent silently loses its MemoryHub):
    Session V, which owns `runtime/agent_factory.py`;
  - [#1377](https://github.com/dennys246/Maxim/issues/1377) (`bootstrap.py::build_default_network` types `dn_config` as
    `DNConfig` then assigns `None`, a latent mypy error): Session V, which owns `runtime/bootstrap.py`; once it lands,
    `NAc._offline_decay_enabled` switches its `importlib` version read to a static import (filed 2026-10-18 from S1);
  - [#1009](https://github.com/dennys246/Maxim/issues/1009) (`--resume-sim` resolves twice; from
    [m1b_ledger_evidence_gate.md](m1b_ledger_evidence_gate.md)): in the same PR as `932.resume-sim` (1.3.5, with the
    orchestrator slices; owner decision 2026-10-18), since its fix is one resolution through `resolve_run_dir`;
  - [#1102](https://github.com/dennys246/Maxim/issues/1102) (`NAc.load_state` skips the reward-bias clamp): with
    the experience-clock decay plan's NAc load work (S1, by the #971 persistence precedent above), with its trigger
    walk over the rows that persist NAc.

**1.3.3 in parallel sessions** (owner request 2026-10-17). Each session works in its own worktree, and the "Owns"
column is its file fence. A file under "Shares" is edited in small, separate hunks, merged with main locally in a
signed commit.

| Session | Line | Owns | Shares | Starts with |
|---|---|---|---|---|
| **P (pipeline)** | #1372 trusted publishing and the `v*` ruleset (re-pointed 2026-10-18); #938 the nightly-red window between a release merge and its tag; #1333 vendoring with R4; #1318 CI caching; #1295 upgrade safety | `scripts/vendor_console_ui.py`, `scripts/audit_release_build.py`, `tests/unit/test_vendor_console_freshness.py`, `tests/unit/test_audit_release_build.py`; vendoring into `src/maxim/console/ui_dist/` (a release step, not committed) | `test.yml`: the `release-build` job hunk (its stub manifest) and the caching hunk. `console/ui_bundle.py` and `tests/unit/test_console_ui_bundle.py`: S owns them, P reads `ui_bundle.py` and edits the vendor-script test hunk. `pyproject.toml`: the release bump. `docs/publication_guide.md`: the process session's; hand over the steps | #1333, which has to be in the next wheel; then R4's design pass; then #1372 and #938 |
| **S (pulse seam)** | #1332 the piper fix; #1347 the console extra's `/ws`; R1 #1334; R2 #1335 (tts first); R3 #1336 (stall and health kinds); #1292 F4 (moved 2026-10-24 to embodied_device.md) | `models/audio/`, `console/` except `handle.py` (schemas, announcer, server, health, `ui_bundle.py`, `openapi.json`, `contract_surface.json`), the stall-dict hunk of `runtime/heartbeat.py`, `utils/audio.py`, `utils/response_output.py`, `.github/constraints/console.txt`, `tests/unit/test_console_ui_bundle.py`, the extras in `pyproject.toml` | `console/handle.py` with V; `test.yml` (the matrix lane) | #1332's red gate against the real library |
| **V (atonia gate → 1.3.4)** | advisor_mode.md items 2, 3, 6, then 1 with 5, and 10 (then 4, 12, 7, 11, and Profile A's 13, 8, 14); #1338, #1342; #1362; #1366 (proposed 2026-10-18); #1377 | `tools/` except the `InternetAccessTool` hunk; `runtime/executor.py`, `runtime/bootstrap.py`, `runtime/agent_factory.py`, `runtime/prefetch.py`, `runtime/tool_dispatch.py`, `runtime/concurrent_executor.py`, `runtime/agent_pool.py`; `api.py`, `create.py`, `modes/`, `agents/autonomy.py`; `embodied_runtime/agentic_runtime.py`, `embodiment/tool_bridge.py`, `simulation/{foundry,minecraft_harness,tools,tools_dm,research_tools,campaign_runner,dm_runtime}.py`. `utils/internet_access.py` (H shares the #1345 hunk). For item 10: `utils/http.py` (the offline refusal), `models/download.py`, `models/language/{llama_backend,transformers_backend}.py`, `models/vision/`, `similarity/{encoder,semantic}.py`, `inference/transcribe_audio.py`, `simulation/container_runner.py`, `runtime/decision_log.py`; `doctor/` (item 11) | `console/handle.py` with S. `models/audio/tts.py` (the `ensure_tts_voice` hunk) with S. `tools/internet_search.py` and `tools/comms.py` with H. `runtime/config_loader.py`: the one shared 1.4 config-format bump. **`orchestrator.py` and `agent_loop.py` under the AD8 fence exception.** `cli.py` is Session A's file. Rule T applies to the pinned functions in advisor_mode.md §Order and the fence | the approach notes and design passes for items 2, 3, 1, 5 and 10, together |
| **H (hardening)** | security #1311 #1312 #1314 #957 #997; LLM routing #1369 #1370 #1371 (if ready; proposed 2026-10-18); the `node_security` remainder; fail-loud Stage 3; #1337; #1310 (the owner's strict `passed`); the comms recipient allowlist (O4, non-advisor); #1343; #1345; advisor_mode.md item 9's process-level yield, converging with #1267 on one SIGTERM installer (#1344); CI and evidence truth (#1313 #1203 #1121 #1218 …); the 1.4.0 gate items #1305 #1263 #1264 #1265 #1267; M25; the small fixes | `runtime/role.py`, `runtime/leader_mode.py`, `runtime/local_server_spawner.py` (including R1's decision on `_health_check`), `simulation/benchmark.py`, `cli_utils.py`, `comms/` and the scripts each issue names | `tools/comms.py` (the allowlist hunk) and `tools/internet_search.py` plus `utils/internet_access.py` (#1345) with V; `cli.py` (#1310's exit code, #1337, #1343's consent hunk, item 9's installer) with Session A; `test.yml` (the CI-truth hunks) | #1337 and #1310 (small, decided), then security |
| **E (E1, B4 and T8 (b), paper only)** | *E1 and B4 DONE 2026-10-18/19 (#1382: floor decision, replays, both preregs DRAFT v2 through review; B4's four-lens design review ran there, v1 2026-10-18, folded into DRAFT v2; the delta at freeze gets its own round). T8 (b) DONE 2026-10-24: the four-lens reviews of both plans, folded (owner decisions G21–G27; follow-ups #1449, #1450).* The BF2 floor decision for E1; the argmax and hunger-state replays (offline, committed); E1's prereg draft through the four-lens design review; **B4's prereg through its four-lens design review** ([#1354](https://github.com/dennys246/Maxim/issues/1354); a new claim, L8's dose-response design); **T8 (b): the four-lens design reviews of [thalamic_relay.md](thalamic_relay.md) and [latent_forward_model.md](latent_forward_model.md)**, paper only, before 1.4.0 (owner decision 2026-10-18) | `docs/experiments/`, the replay scripts under `scripts/` | nothing in `src/` (the fence lifts when 1.3.6's decomposition slices land — the point at which T0 can pass) | the floor decision, presented as options to the owner |
| **A (decomposition)** | 1.3.5 can start now: the five orchestrator characterization arms are tests only | `orchestrator.py`, `agent_loop.py` (the fence), `cli.py` (Session A's file) | grants V's AD8 hunks; V's `cli.py` hunks (the agentic-path call site, #1338's server-mode readiness); H's `cli.py` hunks (#1310, #1337, #1343, item 9's installer) | the characterization arms |

**At most four code sessions run at once:** P, S, V and H, or A in place of one of them. E is paper only and does not
count. Three constraints set the limit:
- the full suite takes about 38 minutes per session, and concurrent suites on one box distort timing assertions;
- `~/.maxim/` is shared across worktrees, so concurrent sims and red gates collide;
- `test.yml`, `CHANGELOG.md` `[Unreleased]` and `outstanding.md` are hotspots every session edits.

**One config-format bump.** `CONFIG_FORMAT_VERSION` moves from 1.3 to 1.4 once in 1.3.3. Every session adding a key
(V's `llm.offline` is the first) appends its fields to that one entry: `tests/fixtures/config_schema_by_version.json`
plus the downgrade transition (#974).

The 1.3.3 cut waits for P (the new pipeline publishes it) and for S's #1332, #1333 and #1347, because pulse is blocked on
them. Session V's atonia items do not hold the 1.3.3 cut; they are 1.3.4 (owner decision 2026-10-17). Session H's items ship in
1.3.3 if they are ready, and otherwise slip to the next patch without holding the cut.

### 1.3.4, "The robot and recall fixes" (owner decision 2026-10-24)

**Why it comes first:** maxim-pulse's real-Reachy sessions on 1.3.3 found user-facing defects. The robot's speech
is silent or slowed, its voice input is off, Talk forgets what the user said across sessions, and a stop can kill
the memory save. The atonia gate still has five design passes ahead, so these would otherwise sit unreleased on
`main` for weeks. 1.3.4 ships them when they are ready. The gate takes the next free number (§The atonia gate,
below). Each line below rides 1.3.4 only if it is ready, and none holds the cut for another.

**The backlog triage of 2026-10-24** (all 166 open issues read against main by three agents):
- **Closed** with evidence: #1362, #873, #989, #1028 (fixed); #1340, #1341 (stale since the xdist revert); #1292
  (superseded by the embodied-device plan); #1416 (duplicate of #1431); #962 (folded into #1193).
- **Retitled**, because their silent halves had shipped: #819, #1200, #832.
- **Fenced**, so they wait for Session A's decomposition (1.3.5 and 1.3.6), about 30 issues:
  - the orchestrator lifecycle (#1259, #1262–#1267, #1005–#1009, #932, #981, #1228, #1393);
  - the agent-loop gate items (#835, #965, #1054, #1057, #1176, #1178, #1147, #863).
- **Parked as design work or owner decisions**, about 20: the encoding pair #784/#899, authority and approval
  (#834, #922, #1185, #960), fear and extinction (#1072, #1373), executive function (#1073), #1144, #848.
- **The rest are bounded and unfenced.** They are batched by file below and in each session's line.

**Sessions** (at most four code sessions at once; paper work doesn't count). The "Owns" column is the file fence.
A file under "Shares" is edited in small separate hunks, each merged with `main` locally in a signed commit.

| Session | Line | Owns | Shares | Starts with |
|---|---|---|---|---|
| **S (pulse seam, robot)** | #1435 → #1436 → #1437; **#1496 → #1499 (HIGH, exit items, owner 2026-10-26)**; R3 #1336 with the 0.7.0 tag plan sent to pulse up front, plus #1428 and #1430/#1431; R2's reachy cell; the `console/health.py` fixes #1285 #1286 #1287 #1291; R1's remaining audit | as in 1.3.3 (`console/` except `handle.py`, `models/audio/`, `utils/audio.py`, `utils/response_output.py`, the extras), plus named exceptions for `embodied_runtime/media_loop.py` and `hardware/reachy/streams.py`, and (#1435's review round, owner 2026-10-24) `embodied_runtime/connection.py` and `embodied_runtime/selfy.py`, and (#1436's review round, owner 2026-10-25) `embodied_runtime/workers.py`, `utils/optional_deps.py` and two Exp 2 record notes in `docs/experiments/` | `console/handle.py` with V; `tools/response.py` (#1435's honesty hunk) with V; the one-line `cli.py` hunk for #1437 with A | #1435's design pass |
| **R (recall), DONE 2026-10-28** (#1465, #1484, #1515; §Session R's status below) | **#1444 + #1163 + #1064 + #1447, as ONE end-to-end change**: the user's words rendered, framed untrusted (#1444); the `ToolOutput` repr kept out of the record (#1163, at the persistence seam in `memory/types.py`); one shared tokenizer that ranks on what captures write (#1064); the query cues on the user's input (#1447). Then #1137, #1153, and #845 item 1. | `integration/bio_enrichment.py`, `memory/` (hippocampus, retrieval, types), `math/angular_gyrus.py` | `runtime/loop_state.py` (#845 item 1) with A; its tests build through V's `runtime/agent_factory.py` (tests only) | the strict red gate (below) |
| **H (hardening)** | the bounded stop #1438 with #1344, as one mechanism whose deadline the caller sets ([bounded_stop.md](bounded_stop.md): Session A's verbatim `run_agentic_loop` extraction first, then PR 0 session ownership, PR 1 #1438, PR 2 #1344, PR 3 servers; the orchestrator `/new` and shutdown-tail hunks and `run_agentic_loop`'s tail are granted, 2026-10-25/26); the addendum of 2026-10-25, after it and before the CI tail: benchmark verdict honesty #1391 #1395 #1394 #1392 (one PR), scripts #1038 #1024 #1031 #1121, the evidence-gate belt #1218 #1220 (one PR), then if slack #982 #988 #1123 #1093 #1455 #954, plus #1246 (stamp `ts`) and #984 (a write-but-don't-read agent neither reads nor writes shared learned state); LLM routing #1369 #1370 #1371 #1408, #1417 (which closes #1415), #1396; keys and config #1402 #1403; CI #1442 #1271 #1313 #1426 #1325, and #1384 with Session P's design pass; sim-record scrub #1404 | as in 1.3.3 (`runtime/role.py`, `leader_mode.py`, `local_server_spawner.py`, `comms/`, the scripts each issue names), plus `runtime/lane_backends.py`, `tunnel/keys.py`, `scripts/pr_merge_readiness.py` | `console/server.py::_drain_and_stop_handle`, the lifespan and `build_app(stop_budget_s=)`, plus `console/handle.py`'s stop (#1438) with S; `cli.py` (item 9's installer, the #1403 renderer), `orchestrator.py` (`/new`, the shutdown tail, `stop_event=`) and `agent_loop.py` (the wrapper and tail after A's extraction) with A; `runtime/agent_factory.py` with V; the `memory/` and `math/angular_gyrus.py` loaders (prune and count) with R; the unclaimed `bio_integration.py`, `loop_setup.py`, `loop_controller.py`, `simulation/interactive.py` (owner grants 2026-10-25/26) | #1442 and #1402/#1403 (small, decided), then #1438 + #1344's approach note |
| **V (atonia gate)** | its own section, below | as in 1.3.3 | as in 1.3.3 | paper first: the approach notes and design passes |
| **A (decomposition)** | 1.3.5's characterization arms, then the orchestrator slices; first, the verbatim `run_agentic_loop` → `_run_agentic_loop_body` extraction (`split_from`, zero body edits) that H's bounded-stop PR 0 waits on (owner decision 2026-10-26) | `orchestrator.py`, `agent_loop.py`, `cli.py` | grants the hunks above | the five arms |
| **E (paper only)** | B4's design review #1354; T8 (b); the #1301 evidence re-assessment; #1226 once B4's dry run confirms it | `docs/experiments/` | nothing in `src/` | B4 |

**Order** (the box takes four concurrent full suites):
- **Wave 1:** S, R and A, plus V on paper and E on paper. S and R are the user-facing fixes; A is on 1.4.0's critical
  path and unblocks the fenced 30.
- **H runs while V is on paper.** When V starts code, H finishes the PR it has open, then waits for the next slot. That
  slot opens when a session is waiting on a merge.

**Session R's red gate** (lands first, `xfail(strict=True)`, flipped in place). It runs through the shipped
composition, the way `MaximHandle` builds it, in a tmp home:
1. Capture "My name is Denny and my favourite colour is green." through `MemoryAgent`.
2. Add 30 or more newer, unrelated captures, so recency cannot rescue the fact.
3. `shutdown()`, then rebuild from the same home.
4. Enrich "What is Denny's favorite color?" under Talk's goal.
5. The text the LLM receives contains "green", inside the untrusted frame, and no `ToolOutput(` repr.

Also, an utterance shaped like an instruction appears only inside the frame. Ideally the gate drives
`MaximHandle.talk` with a mocked worker, so the fix ships with its caller.

**Ledger:**
- R's fixes to the ranker and to the record shape (#1064, #1163, #1447) fire T1-16's named re-run trigger. The row
  is bounded by its own text ("on the goal-keyword ranker", "not about cue-dependent retrieval"), not at risk.
- So R does the trigger walk and re-points `tests/unit/test_o19_rerun.py::test_the_frozen_ranker_is_the_shipped_ranker`
  at the executed commit's ranker. It never deletes it.
- On the robot, the stop fix (#1438, H) is also needed. The embodied-device plan's step 1 must carry `console_agent`'s
  home across.


**Session R's status (2026-10-28): the code is DONE; the claim waits on #1476.**
- **Merged:**
  - #1465 (a0a3e571) ships #1444, #1163, #1064 and #1447 as one change, plus the first-turn ThoughtGate pass and the
    user's words published at arrival. The red gate flipped in place.
  - #1484 (7b82002b) ships #1153: a blank AngularGyrus category means no filter, and store and recall share one contract.
  - #1515 (f177ab45) ships #1137: the user's words survive compression (`hippocampus.json` 1.2).
  - The design record is [recall_1_3_4.md](recall_1_3_4.md), SHIPPED.
- **Blocking the recall CLAIM, not the code:** [#1476](https://github.com/dennys246/Maxim/issues/1476), T1-16's
  judge rebuild and Exp 63 re-run. T1-16 is STALE since 2026-10-25. 1.3.4 may publish the fixes, but its notes claim no
  cross-session recall until a stamped re-run lands (weak evidence never gates). It is a pure re-run, so it needs no
  design review; it needs the harness provenance assert and a box with no leader beside it.
- **Deferred, tracked:** [#1516](https://github.com/dennys246/Maxim/issues/1516) (the boundary, activation and
  tokenizer follow-ups); #1470–#1475 (side findings); #845 item 1 (1.3.6, below); #1422 (loop work, retitle suggested).
- **1.4:** the experience stream (roadmap_1_4.md §Groundwork, Track B item 7).

- **Session S: the pulse seam's second slice and the robot defects.** These follow 1.3.3's first slice:
  - **R3 ([#1336](https://github.com/dennys246/Maxim/issues/1336)), the stall and health kinds first** (slipped from
    1.3.3, owner decision 2026-10-18). Contract 0.7.0 lands only together with a pulse tag built for 0.7.0 (the cut rule
    above; since R4 the bump makes `main` unreleasable until the console pin moves). Ask pulse for that tag at the
    start, not at the cut.
  - **[#1292](https://github.com/dennys246/Maxim/issues/1292) F4** **moved to the embodied-device plan**
    (2026-10-24): F4 is that plan's `run_device`, in `console/device.py`, and its design review is that plan's step 6
    review (§The embodied device, below).
  - **R2's next extras: reachy, then audio** (readiness_and_extras_standards.md §R2 "Order"). Each adds its cells to
    `scripts/extras_matrix.py::EXTRAS` and leaves `_NOT_YET_IN_MATRIX` in `tests/unit/test_extras_matrix.py`, whose
    ceiling may only fall (M69, partly mechanized).
  - **R1's remaining audit** (readiness_and_extras_standards.md §R1: `TTSEngine.is_available`,
    `LocalSpeaker.is_available`, `SeamStatus.live`, `HealthResponse.monitoring`/`stall_detection`,
    `ModelInfoWire.ready`; `_health_check` is Session H's): each flag exercises its capability or is renamed. Then
    M68's lint.
  - **A startup signal when `maxim serve` has no WebSocket library** (found in #1347's review). An install that gets
    uvicorn some other way still answers `/ws` with 404 silently; `run_serve` should say so once, R1-shaped.
  - **Probe the SAVED leader or cloud configuration server-side** ([#1428](https://github.com/dennys246/Maxim/issues/1428),
    requested by maxim-pulse on candidate 2). A stale saved setup (a rotated key, a moved or downed leader) is silent
    until a chat fails, and `/api/diagnose` reports the remote lane "ok" on presence alone. Proposed: a third
    `ProbeRequest` shape, `{"saved": "<tier>"}`, that resolves the lane's saved URL and key ref on the server (the key
    never leaves it) and returns the existing `ProbeResult` through the same classifier `maxim doctor` uses, cached
    with a timestamp so Console startup and every tab focus stay cheap. Declared as a `SeamStatus` seam so older
    servers show nothing; a contract change, so it rides on R3's 0.7.0 bump. Cloud placements need a real cheap
    authenticated call (today's `provider` probe only checks the key's shape). It is R1 applied to the leader
    connection.
  - **The Reachy speaker sink** ([#1435](https://github.com/dennys246/Maxim/issues/1435), found by maxim-pulse on a
    real Reachy Mini with 1.3.3). `utils/audio.py::push_audio_to_device` returns True on the SDK's silent drop before
    `start_playing()` (and nothing in maxim starts the player, so the runtime's own voice drops too), and never
    resamples Piper's 22,050 Hz to the device's 16,000 Hz (speech ~27% slow and pitched down). The runtime voice's
    own resample also hard-clips (it stays on the int16 scale; measured), and `ReachyAudioStream.push_sample` tells
    the same lie, so the fix covers both sinks. Owner decisions 2026-10-24:
    - whoever holds the SDK handle starts the player once, never the sink (`MediaLoopMixin` at media start, plus a
      documented call for bare-SDK embedders);
    - the SDK's opt-in head wobbler is explicitly disabled at that start, so speech never moves the head
      (docs/agents/embodiment.md §1);
    - the sink refuses loudly when the player is not started, normalizes the scale and resamples to
      `get_output_audio_samplerate()`, and `response_output`'s pre-resample goes in the same PR.

    Design pass first (detecting "started" has no public SDK query); done, with the owner decisions, in
    [speaker_sink_1435.md](speaker_sink_1435.md). **Release gate (owner decision 2026-10-24): 1.3.4 does not publish
    before maxim-pulse releases the build that calls `start_device_player`/`stop_device_player`** (released pulse
    starts the SDK player itself, so its pairing code would go silent on 1.3.4; mechanization backlog M79). **Met 2026-10-25:**
    pulse v0.4.4 calls the pair (vendored by #1462). Rides with R2's reachy cell, whose device smoke
    asserts audio reached a STARTED player at the device rate. R1's in-memory "has spoken" stays the readiness signal
    until the embodied-device plan's step 3 makes it PLAYS.
  - **The device defects from pulse's review** (Maxim-pulse #58, 2026-10-24). Each one is bounded, none waits for
    the embodied-device plan, and each ships in 1.3.4 only if ready, never holding the atonia gate's cut. Owners
    (owner decision 2026-10-24): Session S takes #1435, #1436 (a named fence exception for
    `embodied_runtime/media_loop.py`, which no session owns) and #1437 (a named fence exception for its one-line
    `cli.py` wiring). #1435 also needs named exceptions: `media_loop.py` (its interim player start) and
    `hardware/reachy/streams.py` (the second lying sink; no session owns it); its `tools/response.py` honesty hunk
    is shared with Session V. Session H takes #1438 with #1344, one mechanism whose deadline the caller sets:
    [#1436](https://github.com/dennys246/Maxim/issues/1436), robot voice input off since 2026-04-03 (a lazy import
    of a deleted module behind one warning; fail loud, plus an audit or lint for lazy imports of missing modules);
    [#1437](https://github.com/dennys246/Maxim/issues/1437), `maxim --tts` never speaks (`speaker_fn=None`);
    [#1438](https://github.com/dennys246/Maxim/issues/1438), a Reachy stop can kill the session-end save (the daemon
    kills at ~20 s; `_drain_and_stop_handle`'s defaults allow up to ~110 s; Session H, with #1344).
    [#1439](https://github.com/dennys246/Maxim/issues/1439) (Talk reaches the body-less `console_agent`) is not
    bounded: its fix is the plan's one-agent-per-device contract.
    [#1444](https://github.com/dennys246/Maxim/issues/1444), cross-session recall drops what the user said
    (`integration/bio_enrichment.py::_summarize_episode` renders a tool-less episode as "past experience"). Every
    Talk user hits it, not only the robot. The fix renders the user's words bounded and **framed as untrusted
    content**, because recalled text replayed into later prompts is a stored-injection channel. It gets a strict
    red gate across two sessions on the shipped path. The owner is the bio-memory area (read
    docs/agents/bio-memory.md first). The plan's step 1 must carry `console_agent`'s home across, or these facts
    are lost in the fold.
  - **Talk states and chooses its real toolset** ([#1430](https://github.com/dennys246/Maxim/issues/1430), then
    [#1431](https://github.com/dennys246/Maxim/issues/1431), requested by maxim-pulse). #1430: the Talk prompt names
    the tools it actually has (from `TALK_CONVERSATIONAL_TOOLS`), so it stops claiming capabilities it lacks. #1431:
    an MCP-shaped `GET /api/tools?mode=talk` and `POST /api/setup/talk-tools` let the operator add read-only tools
    (web search, page fetch first), default off. The server decides read-only from a declared eligible set, never
    from the tool's own annotations, and enabling a tool never raises the operator's internet switch. A declared
    seam and a contract change, so it rides R3's 0.7.0 bump with #1428. Owner decisions owed at its start: the first
    set, the MCP spec revision, per install or per conversation, and whether `ddgs` joins the `console` extra.
  - **Reachy motion safety: EXIT ITEMS. 1.3.4 does not publish without both** (owner decisions 2026-10-26, Session S).
    - [#1496](https://github.com/dennys246/Maxim/issues/1496): no non-finite, out-of-range-duration or instant
      command reaches a motor (a NaN `head_yaw` in a body YAML swung the body 160° in 1 s).
      Design: [motion_finite_1496.md](motion_finite_1496.md).
    - [#1499](https://github.com/dennys246/Maxim/issues/1499), split out of #1496 after five design rounds: the
      mid-move neck sweep (the head interpolates on the world geodesic and the body in joint space, so the neck
      passes through about 180° whenever |Δbody + Δrel_yaw| > 180°; live on `main`), the rate cap, `look_at` turning
      the body to 0, and readback-as-reference. Design (v8, after the v6 three-lens review, the v7 delta and the v8 scoped rounds; owner decisions O1–O7, 2026-10-29): [motion_path_model.md](motion_path_model.md). Three PRs; 1.3.4 does not publish until all three merge.
    - Until #1499 ships, pulse holds an operational caution: body turns stay within ±90° on the real robot.
    - Fence (owner, 2026-10-26): Session S, with named exceptions for `embodied_runtime/movement.py`, `motion/`,
      `hardware/controller.py`, `hardware/reachy/`, `default_network/`, `embodiment/spec.py`, the CI greps, and
      shared hunks with V in `tools/reachy.py` and `embodied_runtime/agentic_runtime.py::_resolve_body_wiring`.

### The atonia gate (own clock; it takes the next free patch number when it ships)

*It was "1.3.4" until 2026-10-24. Owner decision 2026-10-24: 1.3.4 ships the robot, recall and console fixes when
they are ready, and the gate takes the next free number when it is done, still on its own clock.*
- **[advisor_mode.md](advisor_mode.md) items 2, 3, 6, then 1 with a minimal 5, and 10** (Session V), plus item 9 if Session H
  has not already shipped it in 1.3.3. Items 4 and 12 join it if they are ready, which starts AvServ's Phase 2 drill;
  items 5 (full), 7, 8, 11, 13 and 14 follow in a later patch.
- **It ships when it is done, on its own clock.** It is a security-critical gate with five design passes, so it does
  not hold 1.3.3. The orchestrator and loop phase 2 do not wait for it, and if they are ready first they ship first.
- **Session V's other issues ride with it**: #1338, #1342 and #1366, plus #1377. (#1362 is closed: fixed in 1.3.3.)
- **Exit:** the registration and dispatch sweeps, the builder-refusal tests and item 10's census test pass.

### The embodied device (own clock, in parallel; not 1.3.4 scope)
- **[embodied_device.md](embodied_device.md)** (DRAFT v2, review round 1 folded; owner decisions 2026-10-24):
  - **The approach:** one owner per device contract, with maxim-pulse's #58 as input. Every piece answers the
    front-gate question against what exists. The audio port is the existing `hardware/streams.py::AudioStream`, and
    there are no new top-level packages. F4 (#1292) becomes its `run_device` in `console/device.py`.
  - **Home:** its own session and its own clock, like the atonia gate. It ships as a patch when ready and holds
    nothing, and it touches no decomposition-fenced file, so 1.3.5, 1.3.6 and 1.4.0 keep their order.
  - **Atonia:** the gate ships as designed. Its closed builder refuses any embodiment, so it and this plan do not
    gate each other. The plan's embodied census is its own acceptance criterion, and the test for a later "atonia
    within embodiment" extension if one is ever taken.
  - **The SDK's head wobbler** is explicitly disabled.
  - **Pulse's four questions are decided:** STT on the leader first; listening after "Maxim"; a private `-dev` Space
    first; the 1.4 memory merge rides `migrate_agent_state` (whole or nothing, refuse on conflict).

### 1.3.5, "The orchestrator"
- **First the five characterization arms** (`persistent_agent`, `imagination`, `new_goal`, `stall`, `generative`) (landed 2026-10-24; their findings #1451–#1457; the two follow-ups owed before the #1259 fix and the stdin slice are in §"Characterization owed").
  **Then the `start_simulation_mode` slices** down to 1,000 lines or fewer, descendants included (gate rule 4;
  item `1323.functions-1000`).
- **Riding with the slices:** [#1259](https://github.com/dennys246/Maxim/issues/1259), [#1262](https://github.com/dennys246/Maxim/issues/1262), #863's orchestrator wraps, #835, #965, #1288's stall nudge, and
  the sim-lifecycle and narrator issues [#981](https://github.com/dennys246/Maxim/issues/981), [#1005](https://github.com/dennys246/Maxim/issues/1005), [#1006](https://github.com/dennys246/Maxim/issues/1006), [#1007](https://github.com/dennys246/Maxim/issues/1007), [#1028](https://github.com/dennys246/Maxim/issues/1028), [#1033](https://github.com/dennys246/Maxim/issues/1033), [#1034](https://github.com/dennys246/Maxim/issues/1034), [#1043](https://github.com/dennys246/Maxim/issues/1043), [#1046](https://github.com/dennys246/Maxim/issues/1046), [#1272](https://github.com/dennys246/Maxim/issues/1272), [#1276](https://github.com/dennys246/Maxim/issues/1276), [#1241](https://github.com/dennys246/Maxim/issues/1241).
  Also [#1301](https://github.com/dennys246/Maxim/issues/1301): re-assess the narrator-driven evidence recorded before #1260 and #1261.
  Also `932.resume-sim` (the `--resume-sim` lookup, a 1.4.0 gate item) with [#1009](https://github.com/dennys246/Maxim/issues/1009) in the same PR
  (owner decision 2026-10-18: moved from 1.3.3, since the code is in the fenced `orchestrator.py`).
  After the slices, not inside them: [#866](https://github.com/dennys246/Maxim/issues/866), then
  [#865](https://github.com/dennys246/Maxim/issues/865), the `sim_logger` move out of `simulation/` (from §1.3.2's
  "After the decomposition"; each its own reviewed PR; owner decision 2026-10-18). #866's import rewrite in
  `runtime/agent_loop.py` needs Session A to grant that one hunk (record it as a fence touch).
  **Homed 2026-10-26** (filed by the characterization arms, #1459; addendum from the open-issue deep dive):
  [#1451](https://github.com/dennys246/Maxim/issues/1451) rides with #1262 (strict red gate first: a percept that arrives while a planning job is in flight is
  stranded); [#1452](https://github.com/dennys246/Maxim/issues/1452) rides with #1259 and #1288 (the idle nudge is unreachable for a live narrator); [#1453](https://github.com/dennys246/Maxim/issues/1453),
  [#1454](https://github.com/dennys246/Maxim/issues/1454) and [#1457](https://github.com/dennys246/Maxim/issues/1457) are ONE `/new` slice, with the owner's decisions of 2026-10-26 recorded on each issue (run
  settings carry and campaign-scoped inputs do not, by a test that classifies every parameter; the learning carries;
  `--report-json` describes the returned run; the slice also makes the post-sim prompt ONE constant used by both the panel and the raw reader, pinned by equality and tied to the `new_goal` characterization arm, and takes the whole interactive post-sim block, which returns `_new_goal`, as its unit, leaving the recursion in the caller); [#1456](https://github.com/dennys246/Maxim/issues/1456) (the golden flake) stays with #1459's tests and is never
  closed on non-recurrence.
  **Fence touches (2026-10-26):** Session H takes the one-line `cli.py` hunk for [#1392](https://github.com/dennys246/Maxim/issues/1392) (the CLI scenario path
  calls the new `validation.py` helper, so the bio expectations get the real snapshot), recorded like #1437's.
  Session A took two small 1.3.4 hunks of its own, each with strict red gates and paid for under rule T:
  [#1468](https://github.com/dennys246/Maxim/issues/1468) (one exit-code table for a run, owner decisions 2026-10-26: a crash 1, a typed backend error or leader
  refusal 2 via `cli.main`, Ctrl+C 130; the run loop's teardown and error mapping moved to `cli.py`'s
  `_end_robot_run` and `_run_stopped_on`) and #1454's interim prompt fix (the post-sim prompt drops "(memory carries over)"; the
  report view and prompt moved to `orchestrator.py::_show_report_and_post_sim_prompt`, registered in
  `tests/unit/_orchestrator_source.py`'s `PARTS`). The `/new` slice restores the promise with the behaviour.
- **Meets B2's entry condition** ([#1352](https://github.com/dennys246/Maxim/issues/1352), the bare LLM loop). B2's code
  prerequisites land in 1.3.6.
- **Exit:** `start_simulation_mode` and its descendants pass gate rule 4, and all five arms are satisfied.

### 1.3.6, "The loop, phase 2"
- **`run_agentic_loop` and `execute_and_learn`** ([#1150](https://github.com/dennys246/Maxim/issues/1150)) to 1,000 lines or fewer
  (item `1323.functions-1000`; it was 200 until #1323).
- **[loop_stack_builder.md](loop_stack_builder.md)** (owner decision 2026-10-17): its trigger fired
  2026-09-04 (#610's ninth hand-assembled loop stack), and the loop rebuild is where one builder replaces the nine.
- **The experience-clock NAc decay, S3–S6** ([experience_clock_decay.md](experience_clock_decay.md), #1361; owner decisions 2026-10-17): decay and body drift on the `ExperienceClockDriver`. Fenced, so it lands with this loop work.
- **Riding with them:**
  - #863's loop wraps, #850, #845 item 1, and the rest of #1263;
  - the dispatch and credit issues [#1256](https://github.com/dennys246/Maxim/issues/1256), [#1145](https://github.com/dennys246/Maxim/issues/1145), [#1146](https://github.com/dennys246/Maxim/issues/1146), [#1147](https://github.com/dennys246/Maxim/issues/1147), [#1149](https://github.com/dennys246/Maxim/issues/1149), [#1054](https://github.com/dennys246/Maxim/issues/1054), [#1057](https://github.com/dennys246/Maxim/issues/1057), [#1202](https://github.com/dennys246/Maxim/issues/1202), [#1176](https://github.com/dennys246/Maxim/issues/1176), [#1177](https://github.com/dennys246/Maxim/issues/1177), [#1178](https://github.com/dennys246/Maxim/issues/1178), [#1211](https://github.com/dennys246/Maxim/issues/1211),
    and [#1200](https://github.com/dennys246/Maxim/issues/1200) (no `build_executor` caller passes `pain_detector=`,
    so tool-failure pain is de-wired; [tool_failure_credit.md](tool_failure_credit.md); proposed 2026-10-18, owner to
    confirm). *(2026-10-18: the old note "#1200, #1227 and #1228 are already homed" was false for #1200; #1227 and
    #1228 are homed in the B0b bullet below.)*
  - the mode readers [#960](https://github.com/dennys246/Maxim/issues/960), [#962](https://github.com/dennys246/Maxim/issues/962), [#1192](https://github.com/dennys246/Maxim/issues/1192), [#1193](https://github.com/dennys246/Maxim/issues/1193);
  - Entity save/load [#1159](https://github.com/dennys246/Maxim/issues/1159) and the sensor resolver [#1156](https://github.com/dennys246/Maxim/issues/1156).
- **B2's code prerequisites are exit items** (owner decision 2026-10-17). The loop slices already make these cuts:
  - the `_llm_unavailable` fallback leaves `run_agentic_loop` for a module with no bio imports;
  - `tool_dispatch.execute_and_learn` becomes importable without bio modules (#1150), and callable with `nac`,
    `hippocampus` and `memory_hub` set to `None`;
  - the `runtime` and `agents` package `__init__`s become lazy.
- **B0b** ([#1227](https://github.com/dennys246/Maxim/issues/1227), [#1228](https://github.com/dennys246/Maxim/issues/1228);
  [baseline_floors_b0.md](baseline_floors_b0.md)): a reliable attribution of LLM calls to the agent under test
  (#1227, `RequestContext.agent`; #1228, dual-LLM goal proposals on the narrator's model) and the `arm_effective`
  stamp built with it, before B4, so M46's floor check (model parity between arms) can pass (owner decision
  2026-10-18).
- **When the fence lifts, file by file** (the fence lifts when 1.3.6's decomposition slices land — the point at
  which T0 can pass; owner decision 2026-10-18):
  - roadmap_1_4 Phase 0 items 1a, 2 and 4, built and run offline;
  - the LLM-floor chain, which is serial:
    1. **B2** ([#1352](https://github.com/dennys246/Maxim/issues/1352)), the bare loop;
    2. **B3** ([#1353](https://github.com/dennys246/Maxim/issues/1353)), `llm_notes`, **Reflexion first** (owner decision
       2026-10-17);
    3. **B4** ([#1354](https://github.com/dennys246/Maxim/issues/1354)), T1-2's re-fire, whose arms need both.
- **Exit:**
  - `lint_release_gates.py` reports nothing outstanding for 1.4.0;
  - B2's three prerequisites are in.

### 1.4.0
- **T1** on the rig (pilots, primitives, the R3 A+C variant).
- **T3:** E1, run under the redesigned and reviewed prereg.
- **T4:** the trigger walk, with T1-2 resolved by B4.
- **T4a, T7** (the release-PR caller grep) **and T8** (b) and (c). T8 (b), the four-lens design reviews of
  thalamic_relay.md and latent_forward_model.md (grounding.md and autonomic_layer.md had theirs 2026-10-09), is
  **DONE 2026-10-24** in Session E, paper only (owner decision 2026-10-18): both plans reviewed and folded, owner
  decisions G21–G27, follow-ups #1449 and #1450; (c), the Dormancy markers and re-points, is checked at 1.4.0.
- **The name** follows the release rule: the highest EARNED rung.

### After 1.4.0, or when a 1.4 rung names them
- **Design work:** [#1185](https://github.com/dennys246/Maxim/issues/1185), #922, #834 and #832 item 2 ([#1315](https://github.com/dennys246/Maxim/issues/1315) moved to 1.3.4 (the atonia gate), 2026-10-17).
- **Mechanisms that enter through Phase 5 or the grounding line:** [#1026](https://github.com/dennys246/Maxim/issues/1026), [#1064](https://github.com/dennys246/Maxim/issues/1064), plus #784, #848, #880, #899,
  #1072–#1075, #1144, #1161, #1180, #1181, #816 (the reversible-compression half) and #819 (the eviction half).
  Beside them (proposed 2026-10-18): [#1137](https://github.com/dennys246/Maxim/issues/1137) (compression keeps only
  the intent goal) **moved to 1.3.4, Session R's next PR (owner decision 2026-10-25)**: it also keeps the user's
  words, a release condition for the recall claim; and
  [#1373](https://github.com/dennys246/Maxim/issues/1373) (`cluster_fear` is permanent until extinction exists) with
  #1072 ([fear_learning.md](fear_learning.md)), on its trigger: extinction exists, or fear_learning revives.
  [#1064](https://github.com/dennys246/Maxim/issues/1064) (the recall ranker) is also a re-run trigger for T1-16.
- **The evidence-gate tail** (trigger-gated): #1168, #1172–#1174 and #1197; also #1118 and #1226 (#1226 is parked;
  its re-fire becomes B4).

### Risks
1. **The decomposition can hold 1.4 for a long time.** About 3,300 lines must leave `start_simulation_mode` and
   `run_agentic_loop` to bring them to 1,000 lines or fewer (#1323; it was about 5,300 at the old 200-line bound), and the gate is shrink-only: leaving it needs an append-only `retired` record. Neither score card asks for
   that much.
2. **The fence serializes E1's instrument behind the whole decomposition** (the fence lifts when 1.3.6's
   decomposition slices land — the point at which T0 can pass), so rig time sits idle unless B1-style floor work
   fills it.
3. **E1 may still fail BF2** after its redesign, leaving 1.4 without an Ambition-axis result.
4. **T1-2 waits on B4 (#1354).** B4 waits on B2 (#1352) and B3 (#1353), which wait on 1.3.5 and 1.3.6. This is the
   longest chain to 1.4.0, longer than E1's. If B4 slips, 1.4.0 slips (owner decision).
   - B4's four-lens design review ran in Session E's lane (v1 2026-10-18, folded into DRAFT v2 by #1382).
     The delta at freeze gets its own round. Only the run waits.
5. **Plan text drifts without a lint** (M2, M36). Each patch release re-reads its plans as part of its release
   transaction.

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
           (agent_loop, then start_simulation_mode behind characterization tests; _main_impl if time;
           agent_loop phase 1 DONE #1210, start_simulation_mode slice 0 DONE #1269; the rest is not in
           1.3.2 and hard-gates 1.4.0, owner decision 2026-10-15, roadmap_1_4.md T0)
           → trigger walk + live Exp 60 re-run (re-run DONE 2026-10-12, #1251/#1253; post-pin walk #1255;
           Session A's post-#1253 walk 2026-10-15)
           ‖ in parallel, inside the fence: the issue burn-down, batches 1–4 (security first)
           ‖ in parallel, no loop code: 1.4 groundwork (roadmap_1_4.md §Groundwork in parallel with 1.3.2)
  → 1.3.3  the release pipeline (#1372) + the pulse seam (#1332, #1333, #1347)
  → 1.3.4  the robot and recall fixes (owner decision 2026-10-24; recall code DONE 2026-10-28,
           its claim waits on #1476)
  ‖ the atonia gate (advisor mode, #1315) on its own clock; it takes the next free number
  → 1.3.5  the orchestrator (characterization arms, then the start_simulation_mode slices)
  → 1.3.6  loop phase 2 (run_agentic_loop, execute_and_learn); the fence lifts when 1.3.6's
           decomposition slices land — the point at which T0 can pass; then B2 → B3 → B4
  → 1.4    Phase 0 instrument on the decomposed loop, then Exp 62 → E1 → …
             ([roadmap_1_4.md](roadmap_1_4.md))
```

Exp 62 RAN and is EARNED on the ledger (2026-09-20) — it never depended on either patch release. **It is NOT a 1.3.1 claim** (owner decision 2026-09-27): it rides a release only once its different-reader pass is recorded. Any further rung likewise runs on the rig in parallel
(`docs/experiments/exp62_pressure_interoception_prereg.md`, decisions D1–D4 taken).

## Cadence

**1.3.1 (owner decision 2026-09-27): a blind re-score is taken at the `v1.3.1` tag**, the same firewalled procedure as v1.3.0. After that: re-score at the 1.4 cut, or when an axis's "to reach" condition is claimed complete — and the claim
is that the guard exists, not that the work was done.
