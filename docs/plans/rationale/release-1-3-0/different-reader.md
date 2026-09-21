# Release 1.3.0: different-reader pass (publication_guide.md §"Tag the released commit", condition (c))

Reader: a Claude session that wrote none of the interpretations under review. Date: 2026-09-19.
Branch read: `release/1.3.0` (working tree: `docs/announcements/release_1_3_0.md` modified, with relative
links changed to absolute. That is the only uncommitted diff). No file other than this one was edited.

**Verdict: DO NOT PUBLISH AS-IS. One BLOCK-RELEASE finding is a wrong directional claim (R3 recount "against
the claim"). A second BLOCK is the "first time it is submerged" / "FIRST submersion" wording for Exp 61, which is
literally false because of the loop-OFF submersion in lifecycle step 4. Both fixes are single sentences.
Every number I recomputed matched its source, apart from the items listed below.**

---

## What I recomputed, and what matched

### 1. Exp 56 RB-1 (data #766, interpretation #768)

Recomputed in Python from `docs/experiments/data/exp56_rebaseline_1204/56_four_arm.jsonl`:

- 201 rows. Arm counts: isolated 51, taught 50, satiated 50, dangling 50. There is exactly one duplicate key,
  `(isolated, pair 42)`: rows 0 and 1, written 9.6 s apart (13:28:05 and 13:28:15 UTC). Both copies have
  `chose_target=false`, `bias_decisive=false` and chose `aff_c` against target `aff_a`. **MATCHES** the disclosure.
- Every row: `server_version` = "Paper version git-Paper-499 (MC: 1.20.4) …" (one distinct value);
  `mc_version_expected` 1.20.4; `harness_git_hash` = `executed_git_hash` = `8f8191e5`;
  `working_tree_dirty_src_scripts` false; `mock` false; `gated` true; `allow_dirty` false. `8f8191e5` is
  reachable from `origin/main` (it is the #765 merge). **MATCHES.**
- Raw file under GATES_V1: isolated 11/51 = 0.2157, taught raw 42/50 = 0.84, decisive 40/50 = 0.80,
  satiated 6/50 = 0.12, dangling 6/50 = 0.12. TRANSFERRED 0.80 ≥ 0.70; ABOVE-FLOOR 0.6243 ≥ 0.20;
  WANT-NOT-FILE 0.72 ≥ 0.20; BOTH-HALVES −0.0957 < 0.10. PASS. Matches `56_four_arm_verdict.json`.
- First-row-stands: isolated 11/50 = 0.22. Gates 0.62 / 0.72 / −0.10, all PASS. Matches
  `56_four_arm_verdict_first_row_stands.json`. Last-row-stands gives the same result, because the two copies
  are identical.
- Against the EARNED 1.16.5 `56_four_arm.jsonl` (200 rows): **every one of the 200 (arm, pair) rows is
  identical** on `chose_target`, `bias_decisive`, first-contact `chosen` and `target_aff`. The result
  therefore reproduces row for row, which is stronger than "every rate identical".
- `56_phase0.json` compared with the 1.16.5 file: they differ only in `ts`, `provenance`, `server_version` and
  `mc_version_expected`. Every check reading is identical (separation 1.0, stability 1.0, margin 0.9, dangling
  causal 0.8873, floor concentration 0.3, K = 96). **MATCHES.**
- The RB-1 plan text was on main (d308b8ae, 2026-09-18 17:39 UTC) before the first data timestamp
  (Phase 0, 2026-09-19 05:44 UTC). Campaign rows run 13:28–14:19 UTC (51 min).
- Not verifiable from committed data: "`verify`'s surface probe measured grass at y=63" (prereg RB-1
  outcome, exp56 prereg line 573). No committed artifact records the probe reading. See NIT N1.

### 2. R3 (data #760, interpretation #761)

Recomputed in Python from `docs/experiments/data/r3_bench.jsonl`: 75 rows, made up of 2 apparatus, 12 donor
and 61 event rows (E 388 appears twice: the refused original and the superseding re-run).

- Refusals: 7 tick-band rows (C 342/349/351, D 360/364/366/368) plus one stale-sample row (E 388). Both
  apparatus rows are clean. **MATCHES** §Amendments and §Outcome.
- Per-arm `t_surface` medians. Amended set: A 27.9955, B 8.5755, C 3.180, D 3.131, E 28.0845. Frozen set:
  C 3.277 (n = 9), D 3.131 (n = 8), other arms unchanged. **MATCHES** the table. Bootstrap CIs match
  `r3_report_amended.json`.
- Oxygen pain, health pain and health lost (medians): A 22.467 / 2.126 / 10.667; B 3.033 / 0 / 0;
  C, D 0 / 0 / 0; E 22.508 / 2.246 / 10.833. Survived 12/12 in every arm. `escaped_before_damage` is 0 for
  A and E and 12 for B, C and D. Drive-decisive on 60/60, and on every executed escape
  `score_components` has drive > 0 with causal = learned_bias = 0 (0 exceptions). No deaths, no guard
  breaches, max state age ≤ 0.124 s. **MATCHES.**
- Contrasts (scipy `mannwhitneyu`, two-sided, default method): C−A −24.816, B−A −19.42, C−B −5.396,
  C−E −24.904, all at p 3.66e-5. D−C −0.049, p 0.840. **MATCHES** the table.
- The recount's effect: refused C rows have median 2.997 and clean C rows 3.277. Refused D rows have median
  3.2025 and clean D rows 3.131. C moves 3.277 → 3.180 and D stays at 3.131. Refused rows surfaced between
  2.587 and 3.395 s, with 0 pain publishes, 0 health lost, decisive, and max state age ≤ 0.102 s.
  In-window periods: each of the 7 refused rows has exactly 1, and exactly 4 clean rows (C 350,
  D 365/369/370) have 1 period below 0.768. **MATCHES** every Amendment 2 fact.
- "C's 3.18 s decomposes" (from each event's `calls`, amended set, medians): first call 0.989, `flee`
  duration 0.001 s, flee→escape gap 0.765, escape→surface 1.429. D: 0.949 / 0.755 / 1.437. B: 6.36 / 0.759 /
  1.472. 0.989 + 0.765 + 1.429 = 3.18, and 0.77/3.18 = 24 %. **MATCHES.**
- Amendment 2's tie-break figures (A 0.745, B 0.765, C 0.786, D 0.767) reproduce as the median
  flee-tick→escape-tick interval on the `ticks` telemetry. E 0.736 reproduces only when the superseded,
  refused E 388 row is included (the counted set gives 0.719). The pooled idle cadence (A 0.581 / B 0.579 /
  C 0.584 / D 0.566 / E 0.580, C vs A p 0.767) reproduces. See NIT N4.
- Code compared against the amendment text. `reclassify_under_amendments` implements exactly the stated
  conditions: the band is the only refusal (`";" not in core`); the median is numeric and above the band's
  quoted upper edge; the event is complete (`end in {surface, death}`); and no later clean row exists for the
  same (arm, seed). `harness_unchanged_between` performs `merge-base --is-ancestor` plus a
  `git diff --name-only` filtered to the stated six path prefixes, and fails closed. Running it myself:
  `6b16bbe9` is an ancestor of `4cca5524`, the diff touches only `r3_cal.jsonl`, `r3_gauntlet.json` and the
  prereg, and `4cca5524` is on origin/main. **MATCHES.**

### 3. Release notes and CHANGELOG

- Exp 60 numbers (5 seeds/arm, 6 placements, 1.0 vs 0.0, p = 1/252, water −1.0 / shore 0.0, median 1.72 s,
  4.34 s window, "3.4 s before the pain") all trace to exp60 §Outcome. See F3 for a missing caveat.
- Exp 61 numbers (12/12 v 0/24, 0/12, 0/24, p 8.0e-10 and 3.7e-7, ×0.75, six gates, one hash, zero refusals,
  escape median 1.72 s, first air 3.14 s) all trace to exp61 §Outcome.
- R3 table (28.0 / 8.6 / 3.2 / 3.1 / 28.1) and "≈ 25 s, ≈ 11 hp, ≈ 22 s of oxygen pain" trace to the
  amended report.
- "No language model in the action path": the harnesses run `run_minecraft_aut` → `_loop_kwargs` with
  `aut_mode: "substrate-primary"` (`src/maxim/simulation/minecraft_harness.py::_loop_kwargs`).
  **SUPPORTED** for Exp 60, Exp 61 and R3.
- "Not claimed" items: Exp 58 is BLOCKED at the instrument (exp58 prereg §Outcome 2026-09-14,
  NULL-WITH-CAUSE); Exp 62 is DRAFT v2.1 with no Outcome section, so "designed, not run" is correct; the
  Phase 1b deferral and its trigger match `archive/roadmap_1_3.md` line 52; extinction, scaling and hive-side promotion
  are listed in exp61 "does NOT claim". **All correctly described.**
- 1.2.1 correction checked against the code. Non-test references to `make_pairing_announcer` are its
  definition, its re-export in `console/__init__.py` and a docstring in `utils/audio.py`: **zero callers**.
  `run_serve_subcommand` (cli.py:659) reaches `run_serve`, which calls `build_app(ui_dist, ui_source)` with no
  `pairing_announcer` (server.py:2173). `post_pair_request` and `post_pair_claim` then raise 409 "Pairing is
  not available on this deployment." (server.py:615/654). `make_device_speak_sink` has zero non-test callers.
  **The correction is ACCURATE**: pieces without the composition, token sign-in, 409 on `/api/pair/*`.

---

## BLOCK-RELEASE

**B1: the R3 recount moved C's median in the claim's favour, not against it.**
`docs/announcements/release_1_3_0.md:55-56`: "the recount moved the carried-fear median *down*, against the
claim, and no contrast changed."
Source: a lower `t_surface` means a faster escape. The amended report makes every C contrast larger in the
claim's direction: C−A −24.718 → −24.816, C−B −5.298 → −5.396, C−E −24.807 → −24.904 (`r3_report.json` and
`r3_report_amended.json`). The recount also brings D closer to C (−0.146 → −0.049). The prereg's own
Amendment 2 says it correctly: the FROZEN rule "runs AGAINST the claim" because it "selectively refused C's
FAST tail" (prereg ~536-538). The recount restores that tail, so it moves toward the claim. The effect is
0.10 s against contrasts of 5.4–24.9 s, which changes no conclusion. The stated direction is still wrong.
Correction (release): "the frozen band had selectively refused C's fastest rows; restoring them moved the
carried-fear median 0.10 s faster (3.28 → 3.18 s), immaterial beside 5–25 s contrasts, and no contrast's
conclusion changed."
The same inversion appears in the merged interpretation: `r3_survival_benchmark_prereg.md:556-557`, Honesty
note, "neither touches an outcome in the claim's favour: the recount lowers C's median". This should be
corrected in the prereg too, as a dated erratum line under the Honesty note (see F1). The release must not
repeat the inverted form.

**B2: Exp 61's receiver is submerged once before first contact. "The first time it is submerged" is false.**
- `release_1_3_0.md:8-9`: "…inherits the fear and acts on it the first time it is submerged."
- `CHANGELOG.md:32-33`: "…out of the water on its FIRST submersion".
- `CHANGELOG.md:53`: "12/12 transferred receivers executed `escape_water` on their first submersion".
Source: exp61 §Receiver lifecycle step 4 is "one loop-OFF submersion, US-free, ≈ 2 s", run before step 5.
Step 5, the DV, is "the first teleport into water B ever receives **with the loop live**". exp61 §Outcome
says "first loop-live submersion". The release body (line 27) and the CHANGELOG bullet heading (line 53) say
it correctly. Only these three phrasings drop "loop-live".
Correction: "…the first time it is submerged with its loop running" or "…on its first loop-live
submersion". In all three places.

---

## FIX-BEFORE-PUBLISH

**F1: prereg Honesty note (merged interpretation #761), `r3_survival_benchmark_prereg.md:556-557`.** This is
the source of B1. The two admissibility arguments are "the recount lowers C's median" and "the hash rule was
unsatisfiable". The first points the wrong way (see B1). The amendment remains admissible on the
instrument argument the section already makes ("the refusal never measured the agent"), not on direction.
Correction: replace the clause with "the recount moves C's median 0.10 s in the claim's direction, restoring
the fast tail the frozen band had selectively refused; admissible because the band measured tick phase, not
the agent, and the move changes no contrast's conclusion". Record it as a dated post-review erratum rather
than a silent edit.

**F2: E was never "trained like C, then detached". Its subscriber was detached during training too.**
- `release_1_3_0.md:44`: "E | trained like C, fear detached".
- `r3_survival_benchmark_prereg.md:635`: "Trained exactly as C and then detached".
Source: the prereg Arms table (line 208) reads "Exp 60 ABLATED training (same exposure, subscriber
detached) … DETACHED throughout". The rows confirm it: every E row has `cluster_fear_dump: {}`,
`water_fear: 0.0` and `fear_before: {}`. E never booked a fear, so there was no fear to detach. The
conclusion ("the exposure without the subscriber buys nothing") is correct. The description is not.
Correction: "E | exposed like C, fear subscriber detached throughout (no fear ever formed)". In the prereg:
"Given C's exposure with the subscriber detached throughout, E surfaces …".

**F3: Exp 60's "median 1.72 s" mixes fear-only reads with reads helped by a positive link.**
`release_1_3_0.md:20-21`: "Latency to air after training: median 1.72 s, all inside the 4.34 s window".
Source: exp60 §Outcome, Caveat: "Placements 2–6 of each post probe are therefore read through fear PLUS a
positive link … The fear-only read is the FIRST post placement per seed: 5/5 surfaced, latency 2.9–3.3 s".
The 1.72 s median is over all 30 placements. The CHANGELOG (line 60) carries the caveat and the release does
not. Correction: append "(the fear-only first placement of each seed: 2.9–3.3 s; later placements also ride
a positive `escape_water` link)".

**F4: "The reward now comes from the world itself" / "moves the reward to the GAME".**
`release_1_3_0.md:5-6`, `CHANGELOG.md:28`. What earned in 1.3 is an aversive signal: the game's
air-hunger PAIN writes a fear. The positive-reward half of the theme ("eat when hungry") is R2
PREMISE-NULL, prior-driven, as the release's own "not claimed" list says. A reader will take "reward" as
appetitive learning from the game, and that was not shown. The training episodes are also harness-scheduled
(held under, then rescued), so the harness has not fully left. Correction: "The learning signal now comes
from the world itself — the game's own pain" (and in the CHANGELOG: "moves the learning signal to the GAME
(its pain)").

---

## NIT

**N1: RB-1's y=63 surface claim has no committed artifact.** `exp56_four_arm_sharing_preregistration.md:573`
reads "`verify`'s surface probe measured grass at y=63". Nothing in `exp56_rebaseline_1204/` records the
probe reading (only the operator's console saw it). Either commit the verify output or write "`verify`
passed its surface probe (which refuses unless grass is at y=63)". The second form is true by construction
of `setup_world.py`'s refusal.

**N2: "every rate identical" (ledger line 194, `archive/roadmap_1_3.md:35`, `release_1_3_0.md:62`, `CHANGELOG.md:51`)
holds under the first-row-stands rule. The raw file's isolated rate is 0.2157.** The ledger and prereg
disclose both verdicts, so this is not wrong. It is stronger, and more informative, to say "row-for-row
identical: all 200 (arm, pair) first contacts reproduce the 1.16.5 campaign exactly", and to add that a
same-seed re-run on a deterministic substrate-primary stack is a platform-port reproduction, not an
independent replication. No interpretation claims replication. This forestalls a reader inferring it.

**N3: "a complete separation floors p at 3.7e-5"** (`r3_survival_benchmark_prereg.md:620`). That is the
floor of scipy's asymptotic approximation at n = 12 v 12. The exact two-sided p for complete separation is
2/C(24,12) = 7.4e-7. Correction: "(asymptotic Mann–Whitney; a complete separation gives p 3.7e-5 under the
approximation, exact 7.4e-7)".

**N4: two statistics share the name "idle cadence".** Amendment 2 (`prereg:531`) gives POOLED idle periods
(A 0.581 … D 0.566), and those include the superseded, refused E 388 row. The §Outcome table and "E ticks
like A (0.579 v 0.579)" use the report's MEDIAN OF PER-ROW MEDIANS (A 0.579, B 0.583, C 0.581, D 0.555,
E 0.579). Label them. Also, C's pooled idle figure rests on 23 periods (≈ 2 per row), against 516 for A, so
the "loop does not tick slower with a carried fear" check has little power on the arms where it matters.
Worth one clause. Relatedly, `CHANGELOG.md:55` says "arm-invariant, ± 15 ms". On the reported covariate D
sits 24 ms below A (0.555 v 0.579). Say "within ≈ 25 ms", or name the pooled statistic. The E 0.736
tie-break figure at `prereg:526` is computed over the superseded row; the counted set gives 0.719.

**N5: "The pilot's 3.15 ± 0.20 s is met"** (`prereg:645`). By the prereg's own line 278 and line 485, 3.15 ±
0.20 is Exp 61's n = 12 figure, not the pilot's. Correction: "Exp 61's 3.15 ± 0.20 s is met".

**N6: `prereg:510-511`, "`docs/` cannot alter behaviour".** The bench reads its gauntlet
(`docs/experiments/data/r3_gauntlet.json`) and apparatus record from `docs/`, and the gauntlet changed
between the two hashes by design. Those files are guarded by per-row pinning and gauntlet validation, not by
the git rule. Correction: "`docs/` holds no code; the two data files the bench reads from it are pinned per
row and validated".

**N7: `release_1_3_0.md:51`, "its one post-data change"** is followed by "Both were amended". There are two
amendments. Correction: "its post-data amendments".

**N8: `release_1_3_0.md:48`, "D ≡ C (the received fear acts like the learned one)".** This is a p = 0.84
non-difference at n = 12, not an equivalence test, and the prereg calls it structural (the argmax: both clear
θ, same tie-break). Suggest "D ≡ C (structurally, by the argmax — the received fear clears the same threshold
and acts like the learned one; the discount is not distinguished)".

**N9: `release_1_3_0.md:3`, "Released 2026-09-19 (UTC — PyPI `upload_time`)".** Confirm this against the
actual `upload_time` at publish. The date appears in three places (memory: release-dates-track-utc).
