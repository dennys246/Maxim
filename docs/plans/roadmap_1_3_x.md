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
| **[1.3.1 — SHIPPED (#926); first green scheduled run 2026-09-27]** **The nightly model-cache lane green** (red 16 nights running, 25 of the last 30 scheduled runs; new console modules missing from its skip allow-list). Fix by making a missing module FAIL rather than by extending the allow-list. | the lane's own red/green + a check that the allow-list cannot grow silently *(Built: the lane installs the console + sign extras and runs with `--require-extras`; `ALLOWED_MODULE_SKIPS` is gone, and `ALLOWED_SKIPS` is ratcheted in `tests/unit/test_model_cache_names.py`.)* |
| **[→ 1.3.2]** **A slow lane that runs**: install `sentence-transformers` so the 24 substrate sweeps execute; replace "executed > 0" with a pinned minimum. | `scripts/check_slow_lane.py` asserting the minimum |
| **[1.3.1 — SHIPPED (#928)]** **Network blocked in tests** (hermeticity is HOME/HF isolation + ~48 env scrubs today, with no block). | a conftest socket guard + a test that asserts an outbound call raises |
| **[→ 1.3.2; owner settings partly done]** **`release-build` required**, `enforce_admins` on, and a required-checks-present gate (`pr_merge_readiness.py` is manual today; the ruleset grants an always-bypass admin role). | branch-protection settings — **owner action**, not a PR |
| **[1.3.1 — SHIPPED (#926)]** **The release procedure reads the nightlies**: refuse to publish while a nightly lane is red. | a step in `audit_release_build.py` or the release PR checklist, mechanized *(Built as `scripts/check_nightlies.py`, not inside the offline `audit_release_build.py`: it needs the network. Runs `--only-when-releasing` in the `release-build` job; blocking needs `release-build` required, row above. Guard: `tests/unit/test_check_nightlies.py`.)* |

### Runtime correctness (C+ → B−)

| item | guard |
|---|---|
| **[1.3.1 — SHIPPED (#930)]** **`AgentInstance.export_memories()` always reports 0** — it reads `self.hippocampus.memories`, which does not exist, and an `except Exception` turns the error into `0`; `AgentPool.export_all_memories` propagates it; the documented example in `docs/user/python-api.md` prints "0 memories" beside a hippocampus holding one. | a test asserting the COUNT (both current tests are vacuous: one checks the key exists, the other that it is a dict) |
| **[1.3.1 — SHIPPED (#930)]** **`create.agent`'s docstring example crashes** — `capture(perception="dark cave ahead")` raises `AttributeError`; `capture` does not validate its argument. | argument validation + a doctest-style test that runs the documented example |
| **[1.3.1 — SHIPPED (#930)]** **`maxim.diagnose()` and `maxim doctor --json` disagree** (diagnose reports all-passed while the CLI exits 1 on a probe diagnose never runs). | a test pinning one probe set for both entry points |
| **[→ 1.3.2]** **The silent-default swallow shape** — a handler that ASSIGNS a fallback instead of `pass`, which is what hid `export_memories` and which `lint_no_silent_swallows.py` cannot see. | extend the lint to that shape, as a ratchet on today's count (430 bare sites, 1,788 `except Exception` total) |

### Maintainability (C → C+, the cheap half)

| item | guard |
|---|---|
| **[→ 1.3.2]** **Extend the function-length ratchet to every function over 300 lines** (18 today, one of 921; the ratchet covers 3). Pin at current length, shrink-only. | `scripts/lint_function_length.py` + the baseline file |
| **[→ 1.3.2]** **A repo-wide mypy error-count ratchet** starting at today's measurement (1,050 errors in 141 files over all of `src/maxim`; CI's typed set stays at 18 files). | a new lint in CI, shrink-only |

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
| **[→ 1.3.2]** **D49** — benchmark honesty: apply-or-delete `weight`, fix the running half-mean, drop-or-ship the missing tier2/tier3 suite files (`simulation/benchmark.py`) | a unit test per promise: a weighted suite's aggregate moves with `weight` (or the key is rejected), and every suite file the format names loads |
| **[→ 1.3.2]** **D46 + D50** — delete the dead percept-transport reference (`simulation/sources.py`); warn on the inert `party_mode` / `choice_resolution` keys in `load_campaign` and drop the dead schema field | a test that loading a campaign carrying either key WARNS once |
| **[→ 1.3.2]** **D63** — a PR against a non-`main` base runs no required checks | a ruleset/branch-protection change (owner action) + `scripts/pr_merge_readiness.py` reporting it; the guard is the gate existing |
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
   1.3.1 (1,071 at the tag; it may only fall). Guard: CI mypy on those modules + the ratchet.
5. **Coverage as a ratchet — and a coverage push where the risk is** (widened 2026-09-27, owner). Baseline:
   the Codex card's whole-suite run at `v1.3.1` — **61.9% of 90,416 statements**, 31,797 uncovered
   ([evidence](../limits/score_cards/evidence/2026-09-27-codex/)). Three mechanisms, all in CI:
   - **An overall floor and per-package floors** that only rise, set from that baseline.
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
6. **One function-length ratchet** — nothing over 200 lines may grow, nothing new may exceed 200 — replacing
   the two mismatched mechanisms ([#940](https://github.com/dennys246/Maxim/issues/940)). Guard: the lint, with
   a per-function baseline.
7. **CI escape paths ([#940](https://github.com/dennys246/Maxim/issues/940))**: the `|| echo` optional install,
   the reason-less `importorskip`, the slow lane's expected roster, the network guard at the process-tree
   boundary. Guard: each lane fails on the escape.
8. **One source of truth for claims** (mechanization backlog M2). Guard: the claims-registry lint.

**Engram integrity (pulled in from 1.4's parallel line, 2026-09-27).** Its four engineering items gate
1.4.0 (release threshold T7), touch no survival rung's path and run off the rig
([engram_formation.md](engram_formation.md)), so they fit the hardening line:
[#908](https://github.com/dennys246/Maxim/issues/908) the Cerebellum is never saved (guard: a round trip
that fails on today's default config); [#909](https://github.com/dennys246/Maxim/issues/909) the
motor-engram docs overclaim and the read side is undeclared-dormant (guard: the Dormant docstring + a
caller-grep test); [#910](https://github.com/dennys246/Maxim/issues/910) the `[DANGEROUS]` annotation is
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
  a decision, or is marked ⟲ gets its own PR. Each PR gets a two-lens review round.
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
| 3. Silent seams | [#851](https://github.com/dennys246/Maxim/issues/851) ⟲ `ToolPainBridge` pending entries leak and disable embodiment-pain attribution (the "SEM pain → NAc cascade" row: "ToolPainBridge attribution change"); [#845](https://github.com/dennys246/Maxim/issues/845) memory consumers that never deliver: items 2, 3 and 5 (5 verified first), and item 4's `exec_agent.py::recall_deep` site (its `plan_manager` site is #841's); item 1 rides the slice — **per consumer, wire it or mark it Dormant, each with its behaviour tier declared**; [#863](https://github.com/dennys246/Maxim/issues/863) step 2, the telemetry wraps catching caller logic, **outside** `agent_loop.py`, `orchestrator.py` and `_main_impl` | The #840/#841 class, beside item 4's mypy extension, which catches more of them. |
| 4. Body defects | [#873](https://github.com/dennys246/Maxim/issues/873) ⟲ the no-silent-fallback half: `damage_component` fails on a missing part instead of reporting success (row 9); [#874](https://github.com/dennys246/Maxim/issues/874) ⟲ cradle heat never reaches the arm (row 9's "Cradle / drive / SEM body change"; probably discharged structurally, since row 9 is the dragon / `base_humanoid` setup, but stated in the PR) | Both are "reports success, did nothing". #873's four design points (sum vs weighted mean, a missing part, partless bodies, archetype reflex sets) stay with [deferred/reflex_layering.md](deferred/reflex_layering.md). |

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
  - [#963](https://github.com/dennys246/Maxim/issues/963) (#829's follow-up): the loop resolves the
    operational mode per call site, so the follow-up type ignores the launch grant. The slice that owns
    the loop's mode handling makes one accessor the only capability reader, guarded against new raw
    `state.data["mode"]` reads. #829's source-pin wiring tests become behavioural in the same PR.
  - [#863](https://github.com/dennys246/Maxim/issues/863): its `agent_loop.py` sites. Its `cli.py::_main_impl`
    sites go with that slice if it lands; if not, they stay open on #863 with this home recorded.
- **After the orchestrator's characterization tests** (the first slice of its decomposition):
  - [#932](https://github.com/dennys246/Maxim/issues/932): the `--resume-sim` lookup. Its other callers
    shipped in batch 2's PR: `session.py`, `api.py`, `research_orchestrator.py`, `campaign_runner.py` and
    `scripts/check_oscillator_coldstart.py` moved onto `resolve_run_dir` / `data_home()`; the
    `scripts/exp44/` and `scripts/benchmark_*` harness scans stay, listed in the persistence-config
    brief's invariant with their reason (they scan a data home they created, not name a run).
  - #863's `orchestrator.py` sites.
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

**Done when:**
- Batches 0–2 are closed.
- Each of batches 3–4 is closed, or moved as recorded above.
- The slice-bound items have landed with their slices, or stay open with them.
- **Breaking changes in a patch release (owner, at the cut).** The release PR asks the owner whether the
  `[Unreleased]` breaking entries fit a patch: #939/#950 (`create.*`/`load.*` store guards), #971/#972/#980
  (`load_persisted=`, run-directory resolution) and #1071 (`create.*` refuses an existing store at construction).
  Recorded here when #939 closed (2026-10-04) so the question is not lost.
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
2. **`start_simulation_mode`** (`simulation/orchestrator.py`, 3,322). Its tests cover **11%** of its lines
   (the Codex card's measurement at `v1.3.1`), so it is NOT decomposed blind: characterization tests
   first, then an orchestrator coverage floor set from them (item 5 above), then slices under the same
   gates below. The coverage ratchet is what makes this decomposition safe.
3. **`_main_impl`** (`cli.py`, 1,696) — only if 1 and 2 land; it is CLI glue, the least valuable of the three.

The "kicked down the road" worry that kept this to one target is answered by the rule in **Sizing**: if a
slice stalls, 1.3.2 ships the slices that landed and the ratchet records the new ceiling.

**Coverage first, then extract (2026-09-27).** No slice moves code its tests do not pin. Each slice
adds characterization tests for the code it will move, in its own commit BEFORE the extraction, and
the extraction commit must keep them green unchanged. Extracted modules arrive at ≥ 80% line coverage
(the changed-line gate enforces it), and the target file's per-module floor rises to its new measured
value in the same PR. The orchestrator (11%) and `cli.py` (14%) get their characterization pass as the
first slice of their decomposition, not after.

**Behaviour preservation is the gate, not an aspiration.** Every slice must keep green, in the same
PR: the byte-identical-selection provenance test, the encoder golden pin, and an offline
reproduction of the committed Exp 60, Exp 61 and R3 verdicts from their data. A slice that cannot
show all three does not merge.

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
