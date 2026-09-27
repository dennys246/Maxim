# Release 1.3.1 "Hardening" — the different-reader pass

**Date:** 2026-09-27, before the tag. **Readers:** two independent Claude readers, read-only, on the
uncommitted release branch. Reader A checked the claims: the CHANGELOG headline, the "Correction to
1.3.0" block and the announcement except Upgrading. Reader B checked Upgrading plus consistency (version
sync, anchors, roadmap, outstanding register, bugs ledger). A first combined reader stalled and was
re-run as these two. The owner's gating calls (Exp 10 "MAINTAINED, narrow"; Exp 37 not re-fired; Exp
62 not a 1.3.1 claim; a blind re-score at the tag) were read separately: Exp 10's by its own
independent reader, recorded in [the Exp 10 re-run record](../../data/rerun_exp10_2026-09-27/README.md).

**Verdicts:** A — SHIP-WITH-CHANGES. B — not ready to tag until #936 (the Exp 10 data PR) is on the
release branch. Both read against the evidence and cited it; every PR and issue number they checked
matches what it is cited for.

## Findings and what was done

| # | Reader | Finding | Disposition |
|---|---|---|---|
| 1 | A | **Overclaim:** "the agent can't give itself code execution". Only singularity is refused; passive → active is unchanged, and active runs shell/sandbox tools under approval, which non-interactive runs auto-approve, so a passive agent can leave passive on its own. | Reworded in the announcement and CHANGELOG to "can't switch itself into singularity", naming the passive → active gap and its tracking issue (#924, with the approval surface #922). Not a code change in this release; the owner's decision is #924. |
| 2 | A, B | **Blocker:** the Exp 10 MAINTAINED claim rests on PR #936, not yet merged; the release branch's ledger still read STALE. | #936 merged (merge commit) and merged into the release branch before the tag; the ledger reads MAINTAINED (narrow) there. |
| 3 | A | The correction block said "the 1.3.0 section is left as published" while the next bullet marks a correction in that section. | "The 1.3.0 section's Exp 61 wording is left as published." |
| 4 | A | The Exp 61 correction said the export/ingest path is "unchanged", but 1.3.1 changes it (#913, #914, the scrub). | "…rests on the shipped export and ingest path, not on signing." |
| 5 | A | #823 overclaim: fencing cannot stop a model from following injected text. | "Fenced as untrusted data … does not stop a model from choosing to follow injected text." |
| 6 | A | The situation fold is #914, credited to #913. | "(#914)" added. |
| 7 | A | "Every other fired row is discharged" contradicts Exp 37. | "Every other fired row except Exp 37 …", and the walk window stated (`v1.3.0..042b7d90`; #933 fires no trigger). |
| 8 | A | "one turn per phase": the baseline ran 3. | "one turn per resumed phase (1–3 across all five sessions)". |
| 9 | A | CHANGELOG "no behavioural claim" vs the announcement's "no *new*". | "no new behavioural claim" everywhere, including CLAUDE.md. |
| 10 | A | "The test suite cannot reach the network" is broader than the guard. | "The test process …", with the subprocess caveat. |
| 11 | A | Memory strength "what it cost" is vague. | "the body's drive pressure and relief at the time". |
| 12 | A | CLAUDE.md placed Exp 62 inside the 1.3.0 sentence. | Marked "after 1.3.0; not claimed by 1.3.1 until its different-reader pass is recorded". |
| 13 | B | **Upgrading missed that `capture()` / `capture_from_loop()` / `store()` now REQUIRE `encoding=`**, so every 1.3.0 call raises; `docs/memory.md` showed the old call twice. | Upgrading leads with it and names the import; `docs/memory.md` examples fixed. |
| 14 | B | The plain CLI agent and `maxim.run()` default to passive, which is now enforced. | Stated, with "maxim active" as the way back. |
| 15 | B | A 1.3.0 signing key needs `--release-sequence N` once. | Stated. |
| 16 | B | `hive pull` / `contribute` to a remote Oasis need `--api-key`. | Stated. |
| 17 | B | The internet policy now takes effect (#822); proxies no longer apply to model-chosen fetches (#824). | Stated. |
| 18 | B | `doctor --as peer` / `diagnose(peer=)` send the key only to its own URL. | Stated. |
| 19 | B | Sandbox refusal also applies with no autonomy controller; `working_dir` outside and >120 KiB scripts refused. | Stated. |
| 20 | B | Input validation (intensities, absolute-value sensor reflexes, negative damage, unknown strategy). | Stated. |
| 21 | B | Export/ingest/config changes (own-learning-only export, identifier pattern, `hive add` refusals, empty `--session`, config downgrade hazard). | Stated. |
| 22 | B | Removed/now-required internals (`Executor.get_last_rpe`, `allowed_mode_transitions`, `situation_cue=`). | Stated. |

Found while folding, beyond the readers: CLAUDE.md and `docs/user/upgrading.md` said simulations save to
`~/.maxim/sessions/`, the same wrong path behind #933. Both now say `sim_reports/`.

## What the readers confirmed

- **Name.** "Hardening" is an instrument name, as DECISIONS 2026-09-19 requires for a release with no
  new EARNED result.
- **Consistency.** Version sync is clean in all six places. The CHANGELOG anchor matches GitHub's
  slug. Every roadmap PR number is merged with a matching title. O11/O14's issues are all closed, and
  O16–O18 match their issues.
- **Corrections.** The Exp 61 unsigned correction (the harness exports without `--sign`), the
  `invalidate` correction (on the v1.3.0 GitHub Release, `release_1_3_0.md` and the 1.3.0 CHANGELOG
  section) and the Exp 37/38 void arm (#889) are each carried where the notes say they are.
- **Numbers.** 607 → 443 links, 52 outbound attempts, 52 prereg records and "red for 16 nights" all
  match their source entries.

**Owed at publish:** the release date assumes a PyPI upload on 2026-09-27 UTC. If it slips, the date
moves in three places (CHANGELOG header, the announcement's Released line, the CHANGELOG anchor in the
announcement).
