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
| O1 | **`WaterTrial.live_fingerprint` does not read the live encoder** ([#783](https://github.com/dennys246/Maxim/issues/783)) | It reads `SensorEncoderConfig().pattern_threshold` — a *freshly default-constructed* config — so the encoder leg is a tautology against a source default git already guards. It omits `gain_exponent`/`gain_modalities` (the encoding equation) entirely and covers 3 of 17 sensor ranges, excluding `light_level` and `time_of_day`. Every frozen apparatus depends on it. | Read `self.encoder.config`; add the two gain keys to the live fingerprint AND `FROZEN["exp60"]["fingerprint"]` in the same commit; widen `sensor_ranges`. Red gate: mutate a live config and assert `check_fingerprint` raises — passes today. |
| O2 | **The A4 gain inverts a place code** ([#784](https://github.com/dennys246/Maxim/issues/784)) | Measured: at a between-centre value the two most informative cells draw weight 0.0006 each and the population totals 0.0012 — a near-zero vector `encode_sensors` returns as `None`. Latent only because the one place-coded sensor (azimuth) is on the ungained audio channel. Place coding is the correct encoding for a rest-less cyclic variable like `time_of_day`, so it is a trap waiting for whoever reaches for it. | A declared `place_coded: true` that excludes those cells from the gain (or gain on the source scalar). Do not build until a rung needs a place-coded world sensor. Red gate: a between-centre reading must not encode to `None`. |
| O3 | **SUPPORT for Rung B's entry condition** | `world_channel_landscape.py` established the similarity landscape has a MIDDLE (shape). Nothing establishes the world ever visits it. The only committed open-world trace has `light_level` 0.0 in 1193/1193 and `time_of_day` pinned in 1193/1193. A graded read is worth building only where continuous shape AND real support overlap. | A trace with `doDaylightCycle` **on**, analysed the way `l11_real_trace_remeasure.py` analyses its own. Only run it if a rung wants Rung B. |
| O4 | **Reviewed-diff vs merged-diff comparison** | CLAUDE.md's review-round discipline says a round covers the diff as it existed when it ran, and names this comparison as mechanically checkable and tracked follow-up. Today it is author attention. It is the guard for the 2026-07-29 incident where a PR was squash-merged with only its first commit, shipping a design its own review had refuted, with green CI. | A check that compares a merge commit's diff against the last-reviewed diff, or refuses a squash-merge on a branch that gained commits after its review. |
| O5 | **Archive the 1.1–1.3 roadmaps** | Their stated reason for staying — the open-item ledger — is gone (§Closed). `roadmap_1_1_to_1_3.md` and `roadmap_1_3.md` are historical records of shipped releases with ~43 inbound references that would need rewriting to `archive/`. `roadmap_1_3_path.md` is misnamed: it is the 1.4 sequencing plan, not a 1.3 doc. | Rewrite inbound links, move both to `archive/`, rename `roadmap_1_3_path.md` to what it is. Mechanical but not trivial; its own task. |
| O6 | **`world_channel_weighting.md`'s provenance gap** | Unlike `setpoint-neutral`, its four lens reports are not preserved verbatim under `docs/experiments/rationale/`. Every load-bearing finding was re-verified before folding, but the reports' reasoning lives only in a session transcript. Stated in the file, so it is disclosed rather than hidden. | Write the four reports to `docs/experiments/rationale/world-channel-weighting/`, or accept the file as a decision record that is never cited as evidence. |
| O7 | **Rig housekeeping (big-mac-mini)** | `~/RMSrv/scripts/Maxim` carries a pile of uncommitted experiment output — cradle runs, Exp 38/42/52 leftovers, `cohort0_artifacts/`, `pair0_artifacts/`, and a file named `2c9f1579`. Some may be evidence nobody committed; some is certainly scratch. It does not affect provenance (`DIRTY_SCOPE` is `src`+`scripts`), but it makes `git status` unreadable on the box where experiments run. | Triage: commit what is evidence, delete what is scratch, gitignore what recurs. |

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
