# Outstanding — the standing register of owed work

> **What this is.** Cross-cutting work that is genuinely owed, each entry **verified at the date
> shown** rather than copied forward. It is deliberately NOT release-scoped: these outlive any one
> version, which is why they kept ending up in roadmaps named after shipped releases and going stale
> there. Release-scoped work lives in its roadmap (`roadmap_1_3_x.md` for the hardening line,
> `roadmap_1_4.md` for the experiment ladder); experiment graduations live in
> `behavioral_graduation_candidates.md`.
>
> **The rule that makes this useful:** an entry states *what would close it*, and anyone adding one
> checks first that it is not already done. That check is not theoretical — see §Closed below.

## Why this file exists (2026-09-20)

Asked why the 1.1–1.3 roadmaps were still in `docs/plans/` rather than archived, the answer looked
like "they hold the open-item ledger — CLAUDE.md cites items 16.1 / 16.4 / 16.10 as owed." Auditing
them found **all of 16.1–16.10 already shipped**, and CLAUDE.md declaring a `KNOWN GAP` for one of
them that had been enforced in CI for weeks and had blocked a PR that same day (#793).

The failure mode is worth naming because it is the inverse of the one this repo usually guards
against: not a mechanism that does not run, but **a mechanism that runs while the docs say it does
not**. A guard documented as absent gets worked around. Owed work kept in a document named for a
shipped release is not read as live, so it is never re-checked, and its status rots in whichever
direction nobody is looking.

## Open

| # | Item | Why it is owed | What would close it |
|---|---|---|---|
| O2 | **The A4 gain inverts a place code** | Latent (the one place-coded sensor is on the ungained audio channel), and a trap for whoever first place-codes a world sensor. | **[#784](https://github.com/dennys246/Maxim/issues/784)** — authoritative for the measurements and the options. |
| O3 | **SUPPORT for Rung B's entry condition** — **CLOSED 2026-09-25, answered offline: no SUPPORT at the fear place; the 0.799 is the time wrap (#899); rig run withdrawn on design review ([rationale](../experiments/rationale/rungb-support/)).** | `world_channel_landscape.py` established the similarity landscape has a MIDDLE (shape). Nothing establishes the world ever visits it. The only committed open-world trace has `light_level` 0.0 in 1193/1193 and `time_of_day` pinned in 1193/1193. A graded read is worth building only where continuous shape AND real support overlap. | A trace with `doDaylightCycle` **on**, analysed the way `l11_real_trace_remeasure.py` analyses its own. Only run it if a rung wants Rung B. |
| O4 | **Reviewed-diff vs merged-diff comparison** | CLAUDE.md's review-round discipline says a round covers the diff as it existed when it ran, and names this comparison as mechanically checkable and tracked follow-up. Today it is author attention. It is the guard for the 2026-07-29 incident where a PR was squash-merged with only its first commit, shipping a design its own review had refuted, with green CI. | A check that compares a merge commit's diff against the last-reviewed diff, or refuses a squash-merge on a branch that gained commits after its review. |
| O6 | **`world_channel_weighting.md`'s provenance gap** | Unlike `setpoint-neutral`, its four lens reports are not preserved verbatim under `docs/experiments/rationale/`. Every load-bearing finding was re-verified before folding, but the reports' reasoning lives only in a session transcript. Stated in the file, so it is disclosed rather than hidden. | Write the four reports to `docs/experiments/rationale/world-channel-weighting/`, or accept the file as a decision record that is never cited as evidence. |
| O7 | **Rig housekeeping (big-mac-mini)** | `~/RMSrv/scripts/Maxim` carries a pile of uncommitted experiment output — cradle runs, Exp 38/42/52 leftovers, `cohort0_artifacts/`, `pair0_artifacts/`, and a file named `2c9f1579`. Some may be evidence nobody committed; some is certainly scratch. It does not affect provenance (`DIRTY_SCOPE` is `src`+`scripts`), but it makes `git status` unreadable on the box where experiments run. | Triage: commit what is evidence, delete what is scratch, gitignore what recurs. |
| O9 | **`CLAUDE.md` diet — partially done, 11973 → 11460** | First pass 2026-09-20 moved two entries (the Reachy motion pair, the harness-provenance lesson) into briefs the routing table already makes mandatory. ~540 tokens of headroom now. The remaining bulk is genuinely cross-cutting: `atomic_write_json`, `stable_hash_32`, `_format_version` and CC3 all read as persistence-subsystem rules but apply to ANY code that persists, so demoting them to `persistence-config.md` would hide them from someone editing `decisions/`. | **The rule the first pass produced:** move an entry only where the owning brief ALREADY carries its substance, so the move is a de-duplication and the stub is a pointer to something real. The four-lens design-review lesson failed that test — the brief has zero coverage, so relocating it would let anyone who skips the brief skip the gate. Further cuts need either brief-enrichment first, or a decision to accept that CLAUDE.md is near its useful size. |
| O10 | **`CLAUDE.md` is near its 12k ceiling regardless** | `lint_claude_md_invariants.py` holds the ceiling, so the next invariant anyone adds fails CI. A diet pass is owed — but it is a judgment call about **what must stay always-loaded**, not a mechanical move: a rule demoted to a satellite doc stops being in context by default, and a guard nobody loads is a guard nobody follows. | A stated principle FIRST (e.g. "a rule stays only if violating it would be invisible without the rule in context"), then apply it. A session opener, not a closer. |
| O12 | **`sim_logger` out of `simulation/` — schedule soon** (2026-09-23) | The layering decision is made (keep the function names; make the terminal a sink first, then move only the trace core). Order: [#866](https://github.com/dennys246/Maxim/issues/866) → [#865](https://github.com/dennys246/Maxim/issues/865). | Both issues closed. |
| O13 | **#863 step 2 — the redundant emitter wraps** (2026-09-23) | Classification and plan: [#863 comment](https://github.com/dennys246/Maxim/issues/863#issuecomment-5808338889). Lowest value on this list; batch it or hand it to a parallel session. | [#863](https://github.com/dennys246/Maxim/issues/863) closed. |
| O15 | **Potential experiment re-runs — flagged 2026-09-26, none scheduled** | Surfaced while re-checking the pymaxim.bio experiments pages against the records. Each is a result whose evidence the record itself now marks incomplete; none is an EARNED headline, and each may instead be closed by an owner decision not to re-run. **(a) Exp 37 NAc-bias-off arm** — never a valid ablation ([#889](https://github.com/dennys246/Maxim/issues/889), fixed); whether NAc reward bias mediates the delta is unanswered ([record](../experiments/37_cross_model_results.md), correction 2026-09-25). **(b) Exp 42 C2** — UNESTABLISHED in the 2026-09-01 re-validation, live records lost to an operator error ([ledger](behavioral_graduation_candidates.md), substrate-primary row). **(c) Exp 45 arm 3 (merge)** — downgraded 2026-09-01 (D62); the guard was repaired 2026-09-02 but the original evidence was not re-earned (ledger, real-hardware sensorimotor row). **(d) `temporal_credit_validation`** — never run ([record](../experiments/temporal_credit_validation.md), audit 2026-09-13: STALE), yet cited as a regression guard on a Tier-1 ledger row and behind the `[behavioral]` SCN temporal-coupling tag in `docs/agents/bio-memory.md`; per #889 the `distribute_reward` path that tag describes has had no production caller since 2026-04-24. Already owed elsewhere, listed so it is not re-audited: the Exp 53b hardware block (roll/pitch recalibration, L11 re-measure, two-robot replication — ledger, cross-context readout row). | Per item: a re-run recorded in its experiment file, or an owner decision not to re-run recorded beside the claim it bounds. **(d) first** — a `[behavioral]` tag resting on a never-run protocol is the only one where a current claim outruns its evidence; re-tag it or run it. |
| O16 | **Sim-Short re-runs stop after 1–3 turns on D13 — verified 2026-09-27** | Every Exp 10 re-run session on 2026-09-27 ended `planning_failed` (the narrator's model proposing an AUT tool, `sense_tools`), against 8 clean turns per phase in August on the same model and settings; Exp 09 (2026-09-24) stopped at 5. It is why Exp 10 is MAINTAINED only narrowly, and it limits every Sim-Short heartbeat. Regression (e.g. #823's follow-up framing) or model noise is undecided. [#935](https://github.com/dennys246/Maxim/issues/935). | The cause named; if a regression, fixed, and a Sim-Short phase reaching its turn cap again on mistral-7b. |
| O17 | **Run-directory lookups outside `resolve_run_dir` — verified 2026-09-27** | `utils/paths.py::resolve_run_dir` answers "which directory is run X" for `maxim substrate`, `hive pull` and `roy diff` (#933); the orchestrator's `--resume-sim`, `Session.from_disk`, a script and two writers that ignore `MAXIM_DATA_HOME` still resolve on their own. [#932](https://github.com/dennys246/Maxim/issues/932). | Each caller on `resolve_run_dir`, or listed in the persistence-config brief's invariant with its reason. |
| O18 | **Security follow-ups from the 1.3.1 cluster — verified 2026-09-27** | [#921](https://github.com/dennys246/Maxim/issues/921): `validate_base_url` and `download_to_file` re-resolve after their address check (the #824 class, operator-supplied URLs). [#922](https://github.com/dennys246/Maxim/issues/922): the in-session human approval surface for autonomy/mode requests (#827 made them fail closed meanwhile). [#924](https://github.com/dennys246/Maxim/issues/924): a passive agent can self-switch to active (latent; decide with #922). | Each issue closes itself; this row goes when the last does. |

## Mechanization backlog (added 2026-09-27)

Rules the repo follows **by attention** — each a `Regression guard: process invariant` in CLAUDE.md or a
brief, citing its row here (CLAUDE.md §Working principles, "Enforced, or on the backlog"). The v1.3.1
score cards credit only enforcement, and the release's slips all landed on rules like these. **Close a
row by shipping the check** (a lint, test or required CI job, proven by deleting its mechanism), then
point the rule's guard line at it and move the row to §Closed. Ranked by the axis it moves; both v1.3.1
cards' deciding gaps are named where they apply.

| # | Rule (where) | The check that would enforce it | Axis |
|---|---|---|---|
| M1 | Weak evidence never gates; typed aborts are not data (CLAUDE.md) | `maxim --sim` stamps commit, clean tree, model, `n_ctx` and interpreter path into `report.json`; a lint refuses a ledger status change citing a record that lacks them or whose runs ended in a typed abort, unless a committed exception names owner + reason; `rerun_*` data governed by the prereg lint, keyed to the row it re-runs. **Both cards' Research-integrity gap.** | Research integrity |
| M2 | One source of truth for claims (new) | A claims registry (status, scope, evidence per claim); the README results table, experiments index and release-note claim lines generated from it or linted against it. **Would have caught all four 2026-09-27 statement errors.** | Documentation honesty |
| M3 | A fix ships with a caller (CLAUDE.md) | Diff-scoped CI check: a PR that adds a public symbol or says it fixes an issue shows a non-test caller of the new symbols. | Runtime correctness |
| M4 | The merged diff is the reviewed diff (CLAUDE.md) | The review round records the head SHA it read; a CI check fails a merge whose diff differs from the last-reviewed SHA's, unless the delta is docs-only. | Test/CI truthfulness |
| M5 | Four-lens design review before a harness (CLAUDE.md) | The prereg lint requires `docs/experiments/rationale/<slug>/` with the required lenses before a prereg's freeze commit. | Research integrity |
| M6 | Run the readiness check before merging (CLAUDE.md) | `scripts/pr_merge_readiness.py` as a required status check, so a merge cannot happen with an expected context absent. | Test/CI truthfulness |
| M7 | Tool results flow through the agent bus (CLAUDE.md) | An architecture-audit rule: a tool may not call into an agent class directly (allowlisting `agents.autonomy`, which `tools/mode_switch.py` and `tools/sandbox.py` import legitimately today). | Maintainability |
| M8 | Module extraction never re-imports a mutable global by name (runtime-tools brief) | A lint flagging `from X import _lowercase_global` where `X` assigns that name at module level. | Maintainability |
| M9 | A failure message naming a config fix is checked against what else is running (simulation-experiments brief) | A cadence/timing assertion first asserts the box is quiet (no other `maxim` process), and snapshot tests assert the canonical environment (FastAPI version) before telling anyone to regenerate. | Test/CI truthfulness |

## Where a thing goes — issue, plan, or here

Adopted 2026-09-20, after this register was created and immediately duplicated three GitHub issues
in prose. Two descriptions of one defect drift, and the one nobody reads rots in whichever direction
nobody is looking — which is the 16.10 story in §Closed, reproduced on day one.

| the work is… | it lives in |
|---|---|
| a discrete defect, actionable **now**, with a definite done state | a **GitHub issue** |
| gated on a trigger (a second body exists; a rung names the mechanism) | a **deferred plan**, `deferred/` |
| reasoning, measurements, or why something was rejected | a **doc** (plan, brief, `docs/wiring/`) |
| "what is owed" at a glance | **here**, as an INDEX — one line and a link, never a second description |

**Why issues for defects specifically, in a repo that is otherwise docs-driven:** `Closes #N` in a
PR body closes the issue *mechanically*. A register entry needs someone to remember to delete it.
This repo's standing principle is to push invariants into mechanisms rather than convention, and a
tracker that closes itself is that principle applied to its own bookkeeping — the 16.10 failure
(a doc claiming a gap that had been enforced in CI for weeks) is structurally impossible for an
issue closed by the PR that fixed it.

**Do NOT bulk-backfill issues from existing docs.** Most owed work is trigger-gated, and an open
issue that cannot be worked is noise — the same rot in a different place. File an issue when an
audit VERIFIES a live defect, which is where #783, #784 and #796 came from.

## Editing rule — frozen records are not relinked

When a doc moves, **dated records of what someone observed or declared are NOT rewritten**, even to
fix a link: score cards and their evidence, four-lens rationale reports (the discipline preserves
them *verbatim*), published announcements, and `lessons/claude-md-2026-08-13-pre-diet.md` (CLAUDE.md
calls it frozen). A dangling link inside a dated record is correct; editing the record to fix it is
falsifying what was declared. Learned the hard way on 2026-09-20: a bulk link-rewrite silently
edited 11 such files, including a blind-grading agent's independence declaration listing the paths
it did *not* read. Maintained references (plans, briefs, lessons) do get relinked.

## Closed — recorded so they are not re-audited

- **O11 — the security cluster**, closed 2026-09-27: #800, #801, #802, #824, #826, #827 and #828 all
  closed, shipped in 1.3.1 (#920, #923, #925). The follow-ups it filed are O18.
- **O14 — the two ingest defects the format freeze surfaced**, closed 2026-09-27: #914 (#917) and #913
  (#919) closed, shipped in 1.3.1.
- **O5 — archive the 1.1–1.3 roadmaps**, closed 2026-09-26 (done in #797, the row was stale):
  `archive/roadmap_1_1_to_1_3.md` and `archive/roadmap_1_3.md` exist, and `roadmap_1_3_path.md` was
  renamed to `deferred/second_body_staging.md`; the inbound links left are in frozen records, which
  this file's editing rule leaves unrelinked.

- **O8 — two overlapping hardening plans**, closed 2026-09-20: owner decision to MERGE.
  `quality_burndown.md` → [archive/](archive/quality_burndown.md); its re-verified remainder is in
  [roadmap_1_3_x.md](roadmap_1_3_x.md) §1.3.1 "Carried in from the quality burndown", each item with a
  named guard. Re-verification dropped D19 (already FIXED) rather than copying it forward.
- **Items 16.1–16.10** (the 1.1.x release-governance block), verified 2026-09-20: 16.1/16.5/16.6
  shipped in #571, 16.2–16.4 in #570, 16.7–16.9 in #569, and **16.10** as
  `scripts/lint_unreleased_on_src_change.py` — in the CI lint job, with a unit test, and it blocked
  PR #788 the same day it was still being described as a gap. Artifacts spot-checked for 16.4
  (`lint_function_length.py` + `test_function_length_baseline.py` + its CI step) and 16.10.
  CLAUDE.md's two stale `KNOWN GAP` sentences corrected in #793.

## Not in this file

- Release-scoped hardening → [roadmap_1_3_x.md](roadmap_1_3_x.md)
- The experiment ladder → [roadmap_1_4.md](roadmap_1_4.md)
- Experiment graduations and their re-run triggers → [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md)
- Mechanisms awaiting a rung that names them → `roadmap_1_4.md` §Phase 5 and [deferred/](deferred/)
- Process invariants CLAUDE.md declares unenforced **by design** (review-round discipline, dormancy,
  design-review discipline). Those are not owed work; they are convention with a stated reason.
