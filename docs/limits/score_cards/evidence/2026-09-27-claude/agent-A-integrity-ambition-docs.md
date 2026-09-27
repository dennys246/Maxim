# Evidence Agent A: Research integrity, Ambition and originality, Documentation honesty (pymaxim v1.3.1 @ 7e695a58)

**Checkout and environment.** I worked in `/Users/dennyschaedig/Scripts/Maxim/.worktrees/rescore-v1.3.1`, detached at `7e695a58cb596bafb165dec03286635826c38b59` (the `v1.3.1` tag). `git status` shows only the firewall's edits: the score cards deleted, the archive/critique files removed, and `docs/lessons/experiment-prereg-precedes-data.md` modified. `MAXIM_DATA_HOME=…/scratchpad/rescore/A_home` and `PYTHONPATH=<worktree>/src` were each exported on their own line. `python -c "import maxim;print(maxim.__file__)"` resolved to the worktree's `src/`. I ran no sims and touched no LLM or rig. Live external state is as of 2026-09-27.

---

## Research integrity

**Proposed grade: A−**

### Deciding findings

1. **VERIFIED: Exp 60 reproduces.**
   - Command: `exp60_run.py verdict --data exp60_trials.jsonl --run-id 301eb2edff6d --run-id eeb92752ee2b`
   - Result: `VERDICT: EARNED`. n_clean fear 5, ablated 5; fear post-median 1.0, pre-median 0.0; ablated post-median 0.0. Permutation p = 0.00397 over 252 relabellings, which is the floor at n = 5 per arm. This matches `exp60_verdict.json`.
   - Run without `--run-id`, the verdict is INCOMPLETE because the JSONL also holds run 1.
   - Run 1 (`0e5f7ee98b2d` / `7439a7a969e8`) re-computes as INCOMPLETE. It is disclosed as the instrument's null, with four causes fixed in #730–#733 (`exp60_drowning_avoidance_prereg.md:537-551`).
   - The EARNED rows all stamp hash `2708e2082fdd`, a clean tree, and that hash is an ancestor of the tag.
2. **VERIFIED: Exp 61 reproduces.**
   - Command: `exp61_run.py verdict --campaign-id exp61-campaign-1`
   - Result: `EARNED`. Transferred 12/12; isolated 0/24, cluster_not_fear 0/12, dangling 0/24.
   - Fisher p = 7.99e-10 (= 1/C(36,12)) and 3.70e-7 (= 1/C(24,12)); I checked both by hand.
   - All 121 rows sit at one clean hash, `4e25b475`.
3. **VERIFIED: Exp 62 rung A reproduces.**
   - Command: `exp62_run.py verdict --campaign-id exp62-rungA-1`
   - Result: `EARNED`. `n_clean`, `rates`, `replay`, `fisher_cross_vs_ablated`, `checks` and `verdict` are key-for-key equal to `exp62_verdict.json`.
   - Cross 12/12 (Wilson [0.758, 1.0]) against cross-ablated 0/3, Fisher p = 0.0022. The ablated arm's n = 3 was pre-registered (`exp62_…_prereg.md:90`).
   - All 31 rows sit at one clean hash, `acf568b8`.
4. **VERIFIED: R3 reproduces, including its post-data flip from INCOMPLETE to COMPLETE.**
   - `r3_run.py report … --amended`: output is identical to `r3_report_amended.json` on every key, `STATUS: COMPLETE`.
   - The plain report comes out `INCOMPLETE` (C has 9 clean rows, D has 8, both below 12, and the rows are at a hash other than the gauntlet's). Its values equal `r3_report.json`, but three field names differ, because the post-data amendment renamed `tick_period_median_s` and added `idle_…`.
   - Both amendments are headed POST-DATA (`r3_…_prereg.md:504,521`). R3 claims "nothing graduated" (lines 6, 66, 663).
   - I checked the README's "~25 s, 11 hp, 22 s" against `C_minus_A`: −24.8 s, −22.5 s oxygen pain, and A's median `health_lost` of 10.667 against C's 0. It matches.
5. **VERIFIED, with one gap: Exp 56 and its 1.20.4 re-baseline reproduce.**
   - `analyze_exp56.py` gives PASS on both files; `stats` and `gates` are equal to both committed verdicts. The re-baseline with `--assert-noop-fails` returns rc 0 and `kit_pass: true`.
   - The re-baseline's taught, satiated and dangling rates are identical to the original's (isolated has n = 51). The ledger calls this "a same-seed reproduction, not an independent replication", which is accurate.
   - **Gap:** the original campaign's no-op kit cannot be re-run from committed files. `--assert-noop-fails` on `56_four_arm.jsonl` exits rc 4 with "no artifacts meta at docs/experiments/data/pair0_artifacts/meta.json".
6. **VERIFIED: the prereg lint is real and runs in CI.**
   - `lint_prereg_precedes_data.py --ref v1.3.1` prints "clean — 52 governed data entries checked…, 7 grandfathered by explicit list, 1 not governed".
   - It is wired at `.github/workflows/test.yml:1162-1178` with `--ref origin/main` and a full-history fetch.
   - Its positive-control tests pass, together with `test_exp60_run.py`: 58 passed.
   - The 7 grandfathered entries (the Exp 53/53b originals, the 44b pilot, 54 targets, the H1 `_big` block) are reported as "still failing" with reasons, not silently passed.
   - Exp 53b's EARNED status rests on a clean replication (R1, 2026-08-28, at `v1.1.0`) that is committed.
7. **VERIFIED: the lint's limits.**
   - POST-DATA amendments are "reported, not judged".
   - Prereg edits after data arrived produce only NOTEs, e.g. "r3_…_prereg.md was touched by 8 commit(s) … after the data's first ts — … not judged".
   - `rerun_exp10_2026-09-27/` and `rerun_exp09_…` are out of scope entirely, because the token `rerun` has no prereg and nothing is printed for them. The ledger-flipping re-run therefore sits outside the one CI provenance check.
8. **VERIFIED: the Exp 10 re-run's numbers hold, but its provenance falls short of the repo's own rule.**
   - Numbers:
     - Store sizes: baseline 100 memories; resumes `115820` and `121056` open at `hippocampus_size: 100` and close at 144 and 136.
     - Experience clock: 15,000,000 µs at baseline, 20,000,000 µs after each resume.
     - One trace carries `"memories":3`.
     - The `run_log.jsonl` SHA-256 values match `SHA256SUMS.uncompressed`.
   - The executed commit `a1ba1e5d` is operator-attested only: `report.json` stamps no commit. The CLAUDE.md rule says "a result whose code-under-test cannot be established is not a validation".
   - Every run was a typed `planning_failed` abort. D22 says such runs are not data, and the owner decided to override that.
   - The row still went MAINTAINED (narrow). All of this is disclosed in the record's README.
9. **VERIFIED: the ledger's data-citation lint works.**
   - `lint_claude_md_invariants.py` check 5 requires every row starting `**EARNED` to cite a data link or a dated data-lost note. It runs in CI (`test.yml:~1043`).
   - Locally it exits 1, but only on a CLAUDE.md link to a file the firewall removed. That is an artifact of this checkout, not of the tag.
   - Tier-1 rows 240 (EC, "Data lost (2026-08-29)") and 241 carry data-lost annotations.
   - Rows whose status changed to `RE-VALIDATED` / `TRIGGER FIRED…` / `RE-BASELINED` escape the `**EARNED` prefix match; the ones I looked at still carry data links.
   - No row reads Stale today. The 1.3.1 trigger walk (`behavioral_graduation_candidates.md:205-231`) annotates 13 rows.
   - "Discharged from structure + offline evidence" for Exp 42/53b/56/60/61/62 is the owner's judgment, not a check.
10. **VERIFIED: corrections and nulls are recorded.**
    - Exp 61's "signed-bundle path" is corrected: `exp61_run.py` has no `--sign`. The correction is on ledger row 252 and at `release_1_3_1.md:103`.
    - Exp 37/38's NAc-bias-off arm is declared void (#889).
    - Exp 60 run 1 is kept as an instrument null.
    - Exp 57 is PARTIAL, and the Exp 10/37 LLM-side claims are pulled or reframed.

### What holds
- Every headline verdict reproduces from committed data with a pure offline script.
- Every headline row stamps one clean-tree code hash that is an ancestor of the tag.
- Prereg-before-data is enforced in CI with a positive control.
- Nulls, voids and overclaims are corrected in place, with dates.

### The deciding gap
Validation beyond the pre-registered gate is not enforced.
- The one status change in 1.3.1 (Exp 10 → MAINTAINED) rests on a commit nothing records, and on runs the repo's own ledger (D22) says are not data.
- That record falls outside the prereg lint.
- POST-DATA amendments and in-place prereg edits are listed but not judged.
- The sample sizes are small (5 per arm; 12; 3 ablated), the outcomes are at ceiling, and none of Exp 60/61/62 has an independent replication.

### To reach A
- Stamp the executed commit, clean-tree status and `n_ctx` into sim `report.json`. Then have a lint refuse any ledger status change citing a `docs/experiments/data/` record that lacks them.
- Bring `rerun_*` data under the prereg/provenance lint, keyed to the row it re-runs.
- Make "in-place prereg edit after data" a failure (or require an amendment header), not a NOTE.
- Commit the original Exp 56 `pair0_artifacts`, or mark that kit claim non-reproducible in the verdict file.

---

## Ambition and originality

**Proposed grade: B+**

### Deciding findings

1. **VERIFIED: the earned mechanisms are wired through production paths, not only through harness code.**
   - `NAc.record_cluster_fear` is called at `src/maxim/proprioception/pain_bus.py:700`.
   - `propose_via_substrate(situation_cue=…)` is called from `runtime/agent_loop.py:4518`.
   - Exp 61 uses the real CLI export and ingest.
   - Exp 60/61/62/R3 each run on the tag's history at a clean hash (finding 1 above).
2. **VERIFIED: the scope is broad and original.** It combines a substrate-primary agent with no LLM in the action path, learning from game-native pain in live Minecraft; cross-agent transfer of learned want and fear through signed and unsigned bundles; hardware orienting on a Reachy robot (Exp 45, ledger row 244); and a 5-agent LLM harness.
3. **VERIFIED: the earned results are narrow, by the repo's own account.**
   - The engram tracker (`docs/wiring/engram-formation.md:19-26`) says only the situation engram changes behaviour without the LLM.
   - Motor engrams have no production caller (#909).
   - The Cerebellum is never saved (#908).
   - Episodic situation recall is "built — result discarded".
   - A grep finds no non-test consumer of `retro_tag` beyond storage and strength-floor reads (`strategies.py:687`). `recall_by_situation` does not exist. The notes say "situation recall is wired but nothing consumes it yet".
   - `PerceptTraceBuffer` has 0 call sites.
4. **VERIFIED: the effects are ceiling-shaped on a single discriminator.**
   - Exp 60/61/62 are all 1.0 against 0.0 on one binary water/air situation, in a sealed-shell pool on a frozen day.
   - In R3, survival is 12/12 in every arm, including the ablated one.
   - The README says so: "survival itself was at ceiling… generalization to an unseen situation is untested".
   - Exp 57 dose-response is PARTIAL. On the LLM-harness side, Exp 10 is narrow and Exp 37 is PARTIAL.
5. **VERIFIED: the headline harnesses ship only in the repository checkout.** `scripts/survival_world/` and `scripts/exp56/` are not in the wheel, and the README says so (line ~42).

### What holds
A genuinely original architecture, with several pre-registered live-world results that go through the shipped code paths, including agent-to-agent transfer of a learned fear.

### The deciding gap
What has earned weight is one family of mechanism: a situation-keyed NAc bias or fear on a coarse sensor channel. It has been shown on one discriminator, at ceiling. Much of the bio stack (motor engrams, retro-tagging, situation recall, semantic engrams) records state that nothing uses.

### To reach A−
- A pre-registered EARNED result, with its own verdict script and committed data, on a second non-binary discriminator, or on a multi-step or delayed-credit task (R4), in which an arm does not hit the ceiling.
- A production caller that consumes one of the "recorded, not used" systems (retro_tag, situation recall, motor engrams), with a strict red→green gate.

---

## Documentation honesty

**Proposed grade: B+**

### Deciding findings

1. **VERIFIED: the published artifacts match the repo.**
   - The README is byte-identical to the PyPI 1.3.1 description (empty `diff` against `info.description`).
   - The `gh release view v1.3.1` body equals `docs/announcements/release_1_3_1.md`, apart from one trailing newline.
   - The GitHub asset sha256 values (`b53dd1c3…`, `c5bd54ac…`) match the PyPI file digests.
   - The PyPI upload was 2026-09-27T22:06Z, matching "Released 2026-09-27 (UTC)".
   - `lint_version_sync.py` passes ("1.3.1 in pyproject, __init__, CHANGELOG and the three version lines"). CLAUDE.md:225 reads 1.3.1.
2. **VERIFIED: the release-note numbers match their sources.**
   - "607 → 443": `tests/unit/test_link_merge_identity.py:5`.
   - "52 outbound attempts": `tests/network_guard.py:4`. The guard is installed in `tests/conftest.py:51-55`; a local run printed "network guard: blocked 46 outbound call(s) (dns)".
   - "52 records, all passing": the lint's 52 governed entries. The 7 grandfathered failures are not mentioned in the notes.
   - "1–3 turns across five sessions": the rerun README table.
   - "12/12 against 0 of 60": `exp61_verdict.json` (24 + 12 + 24).
   - "25 s / 11 hp / 22 s": R3 (finding 4 under Research integrity).
3. **VERIFIED: corrections of earlier overclaims are made prominently** (`release_1_3_1.md:101-113`). The notes also carry a recorded two-reader pass: `docs/experiments/rationale/release-1-3-1/different-reader.md`, 22 findings with a disposition for each.
4. **VERIFIED, deciding: the README / PyPI page is stale on a headline row at the tag.**
   - `README.md:27` says Exp 10 "re-run pending before 1.3.1". The re-run was done and recorded MAINTAINED (narrow) in #936, before the tag.
   - The same row asserts "NAc causal links persist and accumulate". The 1.3.1 re-run explicitly did not show accumulation (rerun README, "Link accumulation … is not shown").
   - `docs/experiments/10_cross_session_enrichment.md` does not mention the re-run.
   - No check keeps the README in sync with the ledger.
5. **VERIFIED: the notes do not mention that the tag's own "Release build" check refused, and I found no record of it.** Both CI runs on `7e695a58` failed only at Release build. The push run is 36347881731. The dispatch run is 36347890771, and every other job in it passed, both nightly lanes included. The gate's message was: "REFUSED: … run 36345871460 tested 2f3c408b9186, but main is at 7e695a58cb59". The in-CI gate reads the previous run, so it refuses the release commit's own run by design. PyPI publish followed about 1h40m later.
   - Run plainly now, `check_nightlies.py` passes ("nightlies green at main 7e695a58cb59 (run 36347890771)").
   - So the claim "the release build refuses unless a nightly passed on the exact commit" holds in substance, but the tag carries a red required check.
6. **VERIFIED: one doc still gives the wrong path.** `docs/user/upgrading.md:183` still tells users to back up `~/.maxim/sessions/` before a possible downgrade, and does not mention `sim_reports/`. The different-reader record says both documents "now say `sim_reports/`"; the table row at line 27 was fixed, the backup advice was not.
7. **VERIFIED: the commands I spot-ran work.**
   - `maxim substrate export|inspect|ingest`, `hive add|pull`, `config list`, `model list` and `doctor --help` all return rc 0.
   - Every flag the README uses exists: `--contributor-id`, `--body-ref`, `--trust`, `--receiver-body`, `--apply`, `--queen-key`, `--from`, `--receiver-agent-id`, `--sim-max-turns`, `--embodiment`, `--list-models`.
   - `bodies/infant_humanoid*` exists.
   - The doc-example tests pass: `test_create_agent_example`, `test_readme_pypi_rendering` and `test_diagnose_matches_doctor`, 16 passed.
8. **VERIFIED: the Exp 62 wording is consistent.** It is EARNED on the ledger but "not a 1.3.1 claim until its different-reader pass is recorded". The README omits it, as it should.

### What holds
The release notes, CHANGELOG, GitHub Release and version lines agree with each other and with their sources, number for number. Overclaims are corrected prominently. User-facing commands and the documented API example are exercised.

### The deciding gap
The page the release actually ships (README = PyPI description) contradicts the release's own ledger on the Exp 10 row, which says "pending" and asserts accumulation that was not re-shown. The tag's red Release-build check is not mentioned in the notes, and I found no record of it. Consistency with the ledger depends on reader attention; nothing checks it.

### To reach A−
- Add a lint that cross-checks each README results-table row's status wording against the ledger row it links, and fails on "pending" or unsupported qualifiers.
- Make the release-build nightly gate accept the in-flight run on the release commit, or record the red check in the release record, so the tag's CI is green.
- Add a CI grep that forbids `~/.maxim/sessions/` as a report location in `docs/user/`.

---

## Contacts
- `docs/announcements/release_1_3_1.md` (and so the GitHub Release body) says 1.3.1 "fixes what the v1.3.0 re-score found" and that the new card "lands … beside the 1.3.0 cards". It names no grade.
- `CLAUDE.md:225` links `docs/plans/archive/roadmap_1_1_to_1_3.md`, which the firewall removed. `lint_claude_md_invariants.py` exits 1 on that link here. I did not read it.
- `git status` lists the firewall's deletions under `docs/limits/score_cards/` and an edited `docs/lessons/experiment-prereg-precedes-data.md`. I opened neither.
- The session's injected memory context mentions "scorecard-reconciliation" and "Codex 1.1.0 card OWED", with no grades.
- No earlier grade was seen anywhere, and none influenced these grades.

## Commands run
`git rev-parse/status/log/merge-base/diff --stat`; `exp60_run.py verdict` (with and without `--run-id`, both run pairs); `exp61_run.py verdict --campaign-id exp61-campaign-1`; `exp62_run.py verdict --campaign-id exp62-rungA-1`; `r3_run.py report` (plain and `--amended`, with `--gauntlet`); `analyze_exp56.py --in` (original and re-baseline, with and without `--assert-noop-fails`); `lint_prereg_precedes_data.py --ref v1.3.1`; `lint_claude_md_invariants.py`; `lint_version_sync.py`; `check_nightlies.py`; `pytest test_lint_prereg_precedes_data.py test_exp60_run.py` (58 passed); `pytest test_create_agent_example.py test_readme_pypi_rendering.py test_diagnose_matches_doctor.py` (16 passed); `python -m maxim … --help` (8 subcommands plus the top level); `curl pypi.org/pypi/pymaxim/json`; `gh release view v1.3.1`; `gh run list/view` (scheduled, dispatched and per-commit runs; failed logs of 36347881731 and 36347890771); `gh pr view 937`; Python/grep/sed inspections of ledger rows, the rerun_exp10 records (hippocampus sizes, clocks, SHA256SUMS), provenance stamps and caller greps.
