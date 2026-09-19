# Evidence Agent A: blind grading of pymaxim v1.3.0 @ ad541dd7

Scope: the worktree `.worktrees/rescore-v1.3.0` (HEAD ad541dd7 = tag v1.3.0, confirmed by `git rev-list -n1 v1.3.0`), plus live PyPI, GitHub and CI state as of 2026-09-19.

External state (verified):
- PyPI `pymaxim` latest is 1.3.0. The wheel and sdist were uploaded 2026-09-19T20:24Z.
- GitHub Release v1.3.0 is published (20:25Z) with the same two artifacts.
- The `Tests` workflow on main passed at ad541dd7 (`gh run list`).
- One earlier `Tests` run on release/1.3.0 failed (18:50Z). The later run on the same PR passed.

---

## 1. Research integrity: proposed grade **B+**

### Deciding findings

**R1. The verdicts can be reproduced from the committed data (verified).** I ran each pure analysis at the tag against the committed data. Each result matches the committed verdict file, except for its `provenance` block.
- `exp60_run.py verdict --run-id 301eb2edff6d --run-id eeb92752ee2b` gives EARNED. The only key that differs from `exp60_verdict.json` is `provenance`.
- `exp61_run.py verdict --campaign-id exp61-campaign-1` gives EARNED, with checks {transferred, above_floor, cluster_not_fear, both_halves, specificity, anti_vacuity} all true.
  - Rates: transferred 12/12, isolated 0/24, cluster_not_fear 0/12, dangling 0/24.
  - Fisher one-sided p = 7.99e-10.
  - Only `provenance` differs from the committed file.
- `r3_run.py report --amended --gauntlet ...` gives COMPLETE and is **identical** to `r3_report_amended.json`. Arm medians: A 27.995, B 8.575, C 3.18, D 3.131, E 28.084.
- The frozen `r3_run.py report` gives INCOMPLETE, with the same `incomplete_cause` string. One discrepancy:
  - The committed `r3_report.json` has an `arms.*.tick_period_median_s` field.
  - The tag's code emits `tick_period_median_s_in_window` + `idle_tick_period_median_s` instead. The values are equal, but the field was renamed.
  - So the committed "frozen" report cannot be byte-reproduced by the tag's code. The amendment PR changed the report function after the frozen file was written.
- `analyze_exp56.py --in docs/experiments/data/56_four_arm.jsonl` gives PASS with all four gates true. The re-baseline file `exp56_rebaseline_1204/56_four_arm.jsonl` also gives PASS.

**R2. Provenance of the 1.3 data is clean and consistent with the preregistrations (checked by hand).** I parsed every row's `provenance` block.
- Exp 60: 20 rows, all with `working_tree_dirty_src_scripts: false`. Two code hashes:
  - be038305: run 1, 10 rows, 02:47–03:30Z on 9-16. Declared INCOMPLETE.
  - 2708e208: run 2, 10 rows, 20:36–21:31Z on 9-16. This is the verdict input.
  - The prereg's first commit on main was 2026-09-15T16:42Z (#718), and the freeze was be038305 (#729, 20:44 −06).
- Exp 61: 121 rows at **one** hash, 4e25b475, which is the freeze commit (#747, 17:03Z). The first row is at 17:05Z. All rows are clean.
- R3:
  - cal: 18 rows at 6b16bbe9 (the harness merge), from 00:53Z on 9-18. Prereg v3.1 was on main at 22:40Z on 9-17.
  - bench: 75 rows at 4cca5524 (the freeze-and-cal merge), from 02:16Z on 9-18.
  - All rows are clean.
- Exp 56 re-baseline: rows at 8f8191e5, each stamped with the server's `version` reply (per the ledger; I did not parse the rows independently).

**R3. The prereg-before-data lint does not cover any of 1.3.0's own experiments (verified; the most material finding).**
- `scripts/lint_prereg_precedes_data.py` runs in CI (`.github/workflows/test.yml:1107-1123`), has a positive-control test (20 tests pass), keeps an explicit grandfather list with reasons, and refuses to pass vacuously.
- But its prereg map is only `protocols/*preregistration*.md` plus result-doc links matching `protocols/…preregistration….md` (lines 18, 153, 191).
- The 1.3 preregs are named differently and live elsewhere:
  - `docs/experiments/exp60_drowning_avoidance_prereg.md`
  - `docs/experiments/exp61_shared_fear_prereg.md`
  - `docs/experiments/r3_survival_benchmark_prereg.md`
  - Exp 58/59 as well.
- As a result, `exp60_trials.jsonl`, `exp61_pairs.jsonl`, `r3_cal.jsonl` and `r3_bench.jsonl` are out of scope. Running the lint against `origin/main` prints "clean — 33 governed data entries", with zero lines mentioning exp60, exp61 or r3.
- So the lint's "clean" says nothing about the headline results of the release it gates. This is the "a mechanism that does not run looks like one that ran and found nothing" pattern the project's own CLAUDE.md warns about.
- The ordering does hold when checked by hand (R2). But for 1.3 it is self-attested, not enforced.
- Related gaps:
  - `scripts/lint_harness_provenance.py` requires the dirty-tree gate only for maxim-spawning scripts and `scripts/orient_*/`. The survival_world harnesses call `executed_code_provenance(..., out_path=, allow_dirty=)` (e.g. `exp61_run.py:1095-1194`) by convention, not because the lint requires it.
  - The rule-(5) data-PR-before-interpretation-PR split is documented as not mechanized (`docs/agents/simulation-experiments.md:46`).

**R4. Post-data changes are handled openly and conservatively, and are dated.**
- **Exp 60:** Amendments 3–7 are all POST-DATA (`exp60_drowning_avoidance_prereg.md:310-386`).
  - Each names a measured instrument cause: head-block `is_in_water`, bridge cadence, the idle-gate wake defect, PLANNING autonomy never executing.
  - Each names a fix PR and a regression guard verified RED before the fix.
  - Run 1 is kept in the same JSONL as "the instrument's null, not the mechanism's".
  - Caveat: the code under test changed between the freeze (be038305) and the verdict run (2708e208). This is disclosed, and the gates and DVs are unchanged. It is a legitimate but real departure from "one frozen hash".
- **Exp 60 caveat, stated in §Outcome:** placements 2–6 are confounded by a positive `escape_water` link. The fear-only read is restricted to the first placement (5/5, 2.9–3.3 s).
- **R3:**
  - Two instrument-only rules were loosened post-data (§Amendments, lines 495–574).
  - Both frozen and amended reports are committed, and the amended analysis is a tested pure function rather than a hand recount.
  - A release-day **erratum** (lines 558–566) corrects the amendment's own claim about which way the recount moved the result (it moved toward the claim, not against it).
  - This is the right behaviour, but R3 is also a case of rules changed after seeing the data. The owner accepted it; there is no external check.
- **Exp 56 re-baseline:**
  - The duplicated row (pair 42 isolated) is left in the file, and both the raw and the first-row-stands verdicts are committed.
  - It is honestly labelled "a same-seed reproduction, not an independent replication".
  - Inconsistency: the ledger row says "amendments 1–4 all pre-confirmatory-data". The prereg headers label 3 and 4 `POST-DATA (Phase-0 … only)` / `POST-DATA (harness robustness fix; no confirmatory campaign…)`, and the lint reports them as POST-DATA. The substance is consistent (they precede the confirmatory rows), but the ledger's wording conflicts with the headers.

**R5. Older rows: losses and retractions are disclosed on the ledger row itself (verified by reading `docs/plans/behavioral_graduation_candidates.md:183-197`).**
- Exp 10 (cross-session memory): the original raw records are LOST (`/tmp`), disclosed. A heartbeat re-run is committed.
- EC pattern completion (Roy-2c/Roy-5): "Data lost" is annotated. The row rests on write-ups and tests.
- Exp 37 Qwen32B +1.43 SD: flagged "unverifiable, not refuted" after a same-commit re-run moved the baseline.
- Exp 45: rests partly on `h1_partc_big_block.jsonl`, which carries a dirty-tree stamp. It is grandfathered in the lint and disclosed on the row.
- Exp 53/53b: the original prereg-after-data failure is grandfathered and disclosed. The row rests on the R1 replication, which passes the lint.
- Exp 52: one sentence is explicitly RETRACTED (per-seed reproducibility).
- `lint_claude_md_invariants.py` enforces that every EARNED row cites committed data or a dated data-lost annotation (it prints PASS).
- Downside: several EARNED rows (Exp 10, EC completion, the SEM cascade) rest on lost data or tests only. They stay EARNED with the caveat attached rather than being downgraded.

**R6. Statistical scope is small but stated.**
- Exp 60 is 5 vs 5 seeds, and p = 1/252 is the floor for that design. It uses one pool, one geometry and one seed set.
- Exp 61 is n = 12/24 in one campaign.
- R3's survival is at ceiling by design.
- All of this is stated under "Not claimed". It is honest, but the evidence base per claim is thin, and nothing has been replicated independently on another rig or world.

### Enforced vs documented
- **Enforced:**
  - The prereg-before-data lint (for protocol-named preregs only).
  - The EARNED-row data-citation lint.
  - `lint_harness_provenance` (for spawners and orient harnesses).
  - Per-row provenance stamping, and dirty-tree refusal in the survival harnesses (in code, not lint-required).
  - Pure, tested verdict functions (117 tests pass across exp60/exp61/r3/cluster_fear/fear_transport/substrate_wake/prereg-lint).
- **Documented only:**
  - Prereg-before-data for the 1.3 experiments (docs/experiments/*_prereg.md naming).
  - The data-PR/interpretation-PR split.
  - The design-review discipline.
  - That POST-DATA amendments are truly instrument-only (owner-attested).

### To raise the grade one step (to A−)
1. Extend `lint_prereg_precedes_data.py`'s map to `docs/experiments/*_prereg.md` (or rename the preregs into `protocols/`), and show it governing exp60/exp61/r3 data with a clean result.
2. Make the survival_world harnesses lint-required for the dirty-tree gate.
3. Keep committed report files reproducible by the tagged code, or version the report schema.

---

## 2. Ambition and originality: proposed grade **B+**

### Deciding findings

**O1. Substrate-primary action selection with no LLM in the action path is real and wired in production code.**
- The chain for 1.3 is: `recommend_action` / `propose_via_substrate` → `runtime/agent_loop.py`, with `_substrate_tick_due` as the wake source (fixed during Exp 60 with a RED-verified guard).
- The 1.3 results run through the production `run_agentic_loop` at AUTONOMOUS autonomy, not a hand-composed sequence.
- Wire-4 fear is auto-wired in the canonical PainBus builder: `src/maxim/proprioception/pain_bus.py:756`, `bus.subscribe(create_pain_cluster_fear_subscriber(nac))`, next to Wires 1 and 2. It is therefore not a harness-only attachment.

**O2. Cross-agent transfer of learned wants and fears through a signed-bundle pipeline is unusual, and the earned results demonstrate it.**
- The fear path uses code in `src/maxim/hivemind/{bundle,ingest,merge}.py`:
  - `record_cluster_fear` export with scrub and allowlist, then ed25519-signed bundles.
  - Ingest validation, then `FOREIGN_FEAR_DISCOUNT` 0.75, then re-key, then MIN-fold.
- Exp 61 (12/12 vs 0/24, 0/12, 0/24, with ablation arms separating "cluster shipped" from "fear shipped") and Exp 56 (a taught want, n = 50/arm, four gates) show it working.
- The CLI surface exists: `maxim substrate {export,import,ingest,inspect,keygen,merge-nac,invalidate}` and `maxim hive {add,remove,list,trust,pull,contribute}`.

**O3. The survival world is a real external instrument, not a toy mock.**
- Paper 1.20.4 runs with game-native pain (oxygen, health) and a frozen, depth-calibrated gauntlet (R3).
- The R3 ordering matches the design's prediction: A 28.0, B 8.6, C 3.2, D 3.1, E 28.1 s, with E ≡ A and D ≡ C.
- The bio-inspired memory stack (Hippocampus, NAc, EC, ATL, SCN) also exists as code with persistence and tests.

**O4. The demonstrated capability is narrow relative to the ambition.**
- Each earned 1.3 contingency is one binary cue (`is_in_water`), one action (`escape_water`), one pool and one world.
- "Dark = danger" is BLOCKED at the instrument (Exp 58). Eat-when-hungry is prior-driven (R2), not learned. Cross-pool generalization (Exp 62) has not been run.
- The learning mechanism is cosine-clustered situation keys plus scalar biases and fear values. It is essentially one-step Pavlovian and operant credit, and the project itself names delayed credit as "the real gap".
- Under the LLM-harness framing, the cross-session behavioural-override claim is PARTIAL or pulled (ledger row 2).
- Exp 57 (scaling via pooling) is PARTIAL: pooling costs more total experience.
- In short, the architecture is broad and unusual, but the behaviour it has earned is a small set of toy-scale contingencies.

### To raise the grade one step (to A−)
- An earned result in which the substrate learns something non-trivial: a multi-step or delayed-credit contingency, or generalization to an unseen situation (e.g. Exp 62 cross-pool).
- Or an independent replication on a second world or body, showing the mechanism is not specific to one binary cue.

---

## 3. Documentation honesty: proposed grade **B+**

### Deciding findings

**D1. The release notes' numbers match the experiments (verified).**
- `docs/announcements/release_1_3_0.md`, Exp 60: FEAR 1.0 v ABLATED 0.0, p = 1/252, water −1.0 / shore 0.0, median latency 1.72 s, first placement 2.9–3.3 s. These match §Outcome and `exp60_verdict.json`.
- Exp 61: 12/12, 0/24, 0/12, 0/24, p = 8.0e-10, "all six frozen gates". Six checks are true in the verdict, so this matches.
- R3 table: 28.0 / 8.6 / 3.2 / 3.1 / 28.1. These match `r3_report_amended.json` (27.995 / 8.575 / 3.18 / 3.131 / 28.084).
- The notes carry an explicit "What is not claimed" section, and the positive-link confound for placements 2–6 is stated in the release notes themselves.

**D2. The correction to 1.2.1 exists but does not sit where 1.2.1 readers will see it.**
- The correction is in `CHANGELOG.md:41-49` (1.3.0 section) and in `release_1_3_0.md` ("A correction to 1.2.1").
- But the 1.2.1 CHANGELOG entry (`CHANGELOG.md:173`, "Completes the spoken-code device-pairing loop end to end"), `docs/announcements/release_1_2_1.md:5,9` and the GitHub Release v1.2.1 body (checked with `gh release view v1.2.1`) all still carry the overclaim, with no forward pointer.
- "Left as published" is a defensible policy for a changelog. But a one-line forward note, or an edit to the GitHub release body, costs nothing, and a reader who lands on 1.2.1 still reads the false claim.

**D3. The README (which is also the PyPI long description) is stale on the project's central positioning.**
- `README.md:7` says "Substrate-driven action selection independent of the LLM is post-1.0 research direction via Exp 38 substrate-primary work."
- At the tag, that capability has earned ledger rows (Exp 42, Exp 60, Exp 61), and every 1.2/1.3 headline is substrate-primary.
- The README does not mention Minecraft, survival, Exp 56/60/61 or the fear transfer at all (grep over README returns nothing for these).
- It understates rather than overclaims, but the front door does not describe what the code now does.
- Smaller inaccuracies:
  - "See getting-started.md for the full list of 16 extras" (`README.md:117`), while `pyproject.toml` defines 21.
  - "21 verb-based functions": `maxim.__all__` exposes 23 verb functions if `create` and `load` are counted. This is within the margin of interpretation.

**D4. Small precision errors in the release-critical docs.**
- `release_1_3_0.md` "Upgrading" says "`maxim substrate invalidate --drop-geometry` removes them".
  - The real CLI (`python -m maxim substrate invalidate --help`) requires `--session`, `--drop-geometry TAG` takes a value, it needs `--modality`, and the default is a dry-run without `--apply`.
  - `docs/user/cli-reference.md:278` has the correct form.
- The ledger's Exp 60 row says "frozen at `db7749f3` #729". Git shows db7749f3 is #728 (the harness) and #729 is be038305.
  - The prereg STATUS line also says "FROZEN … at main db7749f3b75f (chunk iv)", while its §Outcome calls be038305 "the freeze commit".
  - Readable as "the code hash frozen at", but it is internally inconsistent.
- The Exp 56 ledger wording "amendments 1–4 all pre-confirmatory-data" conflicts with the prereg's own POST-DATA headers (see R4).

**D5. Version and claim consistency is enforced.**
- `lint_version_sync.py` runs in CI and pins pyproject == `__init__` == the CHANGELOG header == three sync lines.
- `python -c "import maxim; print(maxim.__version__)"` prints 1.3.0.
- CLAUDE.md, `docs/index.md` and `docs/plans/README.md` all say 1.3.0 with PyPI links.
- `lint_claude_md_invariants.py` passes: repo links resolve, and every EARNED row cites data.
- The CLI flags cited in CLAUDE.md/README appear in `python -m maxim --help`: `--sim-max-turns`, `--embodiment`, `--sandbox`, `--aut-mode`, `--interactive`, `--research`, `--foundry`, `--auto-curate`, `--seed`, `--list-models`, `--delete-model`, `--llm`.
- The release notes' "Scope, stated plainly" (the bridge is not in the wheel; reproduction needs a checkout plus a Paper server) is accurate and useful.
- One more staleness item: `docs/index.md:7` still leads with a 2026-05-09 "architectural pivot" banner, and its "Quick Links" advertise a scorecards directory (I did not follow it, per the blind).

### Enforced vs documented
- **Enforced:** version sync, link resolution, EARNED-row data citations, the CLAUDE.md invariant format.
- **Not enforced:** whether README and user-guide prose matches current capability; whether corrections are propagated back to the claims they correct; whether commands quoted in release notes are correct.

### To raise the grade one step (to A−)
- Update the README positioning and "What you can do" to the 1.2/1.3 substrate-primary reality, and fix the extras count.
- Add a forward-pointing correction note to the 1.2.1 CHANGELOG entry, `release_1_2_1.md` and the GitHub release body.
- Fix the invalidate command in the release notes and the Exp 60 freeze-hash and Exp 56 amendment wording in the ledger.

---

## Verified vs inferred

**Verified (commands run or files read at the tag):**
- The tag hash, PyPI version and upload times, GitHub releases and assets, and CI status on ad541dd7.
- Recomputing the Exp 60, Exp 61, R3 (frozen and amended) and Exp 56 (original and re-baseline) verdicts from committed data. Match, apart from provenance and the R3 frozen-report field rename.
- The per-row provenance hashes, clean-tree flags and first/last `ts` for the exp60/exp61/r3 data.
- First-parent commit times for the three 1.3 preregs and their freeze commits.
- That `lint_prereg_precedes_data.py` reports clean but does not govern the exp60/exp61/r3 data (zero mentions; glob at lines 18/153/191).
- That the lint invariants pass.
- 117 targeted tests passing.
- That Wire-4 is auto-wired in `pain_bus.py:756`, and the non-test callers of the fear/transfer symbols.
- The CLI help surfaces.
- The README, CHANGELOG, release-notes and ledger passages quoted above.

**Inferred or not checked:**
- That POST-DATA amendments were genuinely instrument-only. I checked their text and the unchanged gates, but not diffs of every fix PR.
- The Exp 56 re-baseline row stamps (read from the ledger text; not parsed).
- The full test suite and the full CI job list for the tag (only the `Tests` conclusion was checked).
- Older experiments' raw data beyond the ledger annotations and lint output.
- The docs/user guides beyond `cli-reference.md` (spot-checked, not audited).

## Independence

- I did not read or open anything under `docs/limits/score_cards/`, `docs/plans/burndown_1_3.md`, the "Scorecard → roadmap reconciliation" section of `docs/plans/roadmap_1_1_to_1_3.md`, or `~/.claude/`. I did not use `git show` or `git log -p` on any of them.
- My git log calls were limited to specific experiment prereg paths and specific commits.
- One incidental contact: `docs/index.md`'s Quick Links line mentions "Repository scorecards … dual-assessor: Codex + Claude". It shows no grade, and I did not follow it.
- No earlier letter grade was encountered.
- Note: the harness auto-loaded a user memory file into my context, and it contains project-state notes (no grades). I did not rely on it for any finding. Every claim above was verified against repository files, git or live services.
