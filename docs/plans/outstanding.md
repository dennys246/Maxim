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
| O1 | **`live_fingerprint` does not read the live encoder** | Every frozen apparatus depends on it, and its encoder leg is a tautology against a source default. | **[#783](https://github.com/dennys246/Maxim/issues/783)** — authoritative for the detail and the red gate. |
| O2 | **The A4 gain inverts a place code** | Latent (the one place-coded sensor is on the ungained audio channel), and a trap for whoever first place-codes a world sensor. | **[#784](https://github.com/dennys246/Maxim/issues/784)** — authoritative for the measurements and the options. |
| O3 | **SUPPORT for Rung B's entry condition** | `world_channel_landscape.py` established the similarity landscape has a MIDDLE (shape). Nothing establishes the world ever visits it. The only committed open-world trace has `light_level` 0.0 in 1193/1193 and `time_of_day` pinned in 1193/1193. A graded read is worth building only where continuous shape AND real support overlap. | A trace with `doDaylightCycle` **on**, analysed the way `l11_real_trace_remeasure.py` analyses its own. Only run it if a rung wants Rung B. |
| O4 | **Reviewed-diff vs merged-diff comparison** | CLAUDE.md's review-round discipline says a round covers the diff as it existed when it ran, and names this comparison as mechanically checkable and tracked follow-up. Today it is author attention. It is the guard for the 2026-07-29 incident where a PR was squash-merged with only its first commit, shipping a design its own review had refuted, with green CI. | A check that compares a merge commit's diff against the last-reviewed diff, or refuses a squash-merge on a branch that gained commits after its review. |
| O5 | **Archive the 1.1–1.3 roadmaps** | Their stated reason for staying — the open-item ledger — is gone (§Closed). `roadmap_1_1_to_1_3.md` and `roadmap_1_3.md` are historical records of shipped releases with ~43 inbound references that would need rewriting to `archive/`. `roadmap_1_3_path.md` is misnamed: it is the 1.4 sequencing plan, not a 1.3 doc. | Rewrite inbound links, move both to `archive/`, rename `roadmap_1_3_path.md` to what it is. Mechanical but not trivial; its own task. |
| O6 | **`world_channel_weighting.md`'s provenance gap** | Unlike `setpoint-neutral`, its four lens reports are not preserved verbatim under `docs/experiments/rationale/`. Every load-bearing finding was re-verified before folding, but the reports' reasoning lives only in a session transcript. Stated in the file, so it is disclosed rather than hidden. | Write the four reports to `docs/experiments/rationale/world-channel-weighting/`, or accept the file as a decision record that is never cited as evidence. |
| O7 | **Rig housekeeping (big-mac-mini)** | `~/RMSrv/scripts/Maxim` carries a pile of uncommitted experiment output — cradle runs, Exp 38/42/52 leftovers, `cohort0_artifacts/`, `pair0_artifacts/`, and a file named `2c9f1579`. Some may be evidence nobody committed; some is certainly scratch. It does not affect provenance (`DIRTY_SCOPE` is `src`+`scripts`), but it makes `git status` unreadable on the box where experiments run. | Triage: commit what is evidence, delete what is scratch, gitignore what recurs. |

| O8 | **Two overlapping, mutually-unaware hardening plans** | [quality_burndown.md](quality_burndown.md) (the standing track) and [roadmap_1_3_x.md](roadmap_1_3_x.md) (the 1.3.x line) cover the same territory. As of 2026-09-20 the 1.3.x line names **none** of the burndown's items — N1, D32, D84, D49, fail-loud Stage 3, D19; only `mypy` is in both — and neither document referenced the other. The burndown's remainder was deferred "to post-1.3", which **expired** when 1.3.0 published. Surfacing it produced [#796](https://github.com/dennys246/Maxim/issues/796) (see below), which had been sitting unread. | Decide: merge them, or keep both with a stated split. Cross-links added 2026-09-20 so the overlap is at least visible. |
| O9 | **`CLAUDE.md` diet — partially done, 11973 → 11460** | First pass 2026-09-20 moved two entries (the Reachy motion pair, the harness-provenance lesson) into briefs the routing table already makes mandatory. ~540 tokens of headroom now. The remaining bulk is genuinely cross-cutting: `atomic_write_json`, `stable_hash_32`, `_format_version` and CC3 all read as persistence-subsystem rules but apply to ANY code that persists, so demoting them to `persistence-config.md` would hide them from someone editing `decisions/`. | **The rule the first pass produced:** move an entry only where the owning brief ALREADY carries its substance, so the move is a de-duplication and the stub is a pointer to something real. The four-lens design-review lesson failed that test — the brief has zero coverage, so relocating it would let anyone who skips the brief skip the gate. Further cuts need either brief-enrichment first, or a decision to accept that CLAUDE.md is near its useful size. |
| O10 | **`CLAUDE.md` is near its 12k ceiling regardless** | `lint_claude_md_invariants.py` holds the ceiling, so the next invariant anyone adds fails CI. A diet pass is owed — but it is a judgment call about **what must stay always-loaded**, not a mechanical move: a rule demoted to a satellite doc stops being in context by default, and a guard nobody loads is a guard nobody follows. | A stated principle FIRST (e.g. "a rule stays only if violating it would be invisible without the rule in context"), then apply it. A session opener, not a closer. |

### Verified defects found while auditing this list

- **[#800](https://github.com/dennys246/Maxim/issues/800) — sandbox Python scripts have never run.** The
  restricted wrapper blocks its own `import os`, and the obvious reorder would leak `os` and the unrestricted
  originals into the script's namespace. Found while writing #796's guard tests (#796 itself is fixed: SUPERVISED
  now fails closed). The issue is authoritative.

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
