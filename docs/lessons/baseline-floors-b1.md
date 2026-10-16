# B1, the first floor: what it took to compare a bio mechanism with a no-bio learner, and what it found

**Date:** 2026-10-10 → 2026-10-15 (PRs #1224, #1233, #1236, #1238, #1245, #1258, #1274, #1278, #1283, #1294, #1296)
**Class:** research integrity. A result is compared with its own ablations only, and a design that looks
fair loses (or wins) by construction.

## What happened

Owner rule BF2 (2026-10-10, [baseline_floors.md](../plans/baseline_floors.md)): a claim counts as earned only over
a floor, a plain agent without the bio layers. Until then no ledger row had ever met one. "Earned" meant beating
Maxim's own ablation, for example Exp 60's FEAR vs ABLATED (the fear subscriber detached).

B1 ran the non-LLM floors (`tabular_q`, `random`) on Exp 60's drowning task (campaign `48ce40118c20`, 2026-10-14,
40/40 arm-seeds clean on their first attempt). The verdict was `FLOOR_MATCHED_Q`:
- **First post placement surfaced:** FEAR 8/8, Q_EXPLORE 8/8, RANDOM 2/8.
- **Tests:** FEAR > RANDOM p = 0.0035. FEAR > Q_EXPLORE was not shown, p = 1.0, and no equivalence is claimed.
- **The status change:** T1-13 moved to `PARTIAL (narrow: floor matched)` in the data PR itself (#1283).
- **Two rows followed analytically:** T1-14 (Exp 61) and T1-15 (Exp 62), after an owner decision (#1294, #1296).

The FEAR-vs-ABLATED contrasts stand. The rows no longer show behaviour beyond a no-bio learner.

## The lessons

1. **Model the floor learner offline before rig time.** The v3 design's Q constants (γ < 1, optimistic Q₀) lost to
   RANDOM by construction in a toy model (`docs/experiments/rationale/b1-exp60-floor/learner_sim.py`: about 0.10 vs
   RANDOM's 0.15). The run would have "beaten the floor" against a strawman. Owner decision OD5 (γ = 1, Q₀ = −2)
   made the floor a real learner. The same toy model predicted the tie the rig then measured.
2. **A floor can be unfair in either direction. Read the prereg's rejected-arms table before proposing one.**
   - A passive action-value floor (Q_YOKED) cannot learn from pain that arrives whatever it proposes, so it can't
     fail to lose (B1 bio-faithful D1).
   - A floor that the harness arms in water gets WHEN for free. With FEAR's prior it surfaced on its first-ever
     submersion with zero experience, 8/8.
   - The fair floor for learning WHEN from passive pain is a TD state-value learner. With the same prior, that IS
     Wire-4 (the "mechanism twin"), so a tie there is the expected result, not an embarrassment.
   - On 2026-10-15 a "passive WHEN floor that can lose" was proposed. It was Q_YOKED again, already rejected.
3. **Run the mechanism twin; don't only name it.** B1's confounding lens said that naming a tie without running it
   is not the honest answer. B1 left the twin analytical. E1–E3 run it as an arm (baseline_floors.md §Why).
4. **A floor-can-lose contingency has to be designed for.** Exp 60/61/62's generalization lives in the EC key, which
   the floors share. A rung that wants to earn over a floor needs a contingency a REASONABLE floor can lose; the
   candidate is generalization through EC similarity. Making a floor's key over-fine so that it fails is a strawman
   in our favour.
5. **A gate fixture that adds a field the real writer lacks hides a dead path.** `attempt_start` carried no
   provenance in the harness, but the gate tests' fixtures supplied it. So no real campaign could be ESTABLISHED,
   and every real resume was refused. Feed gate tests with the harness's own writers.
6. **Prove a rig script through its own command line.** `b1_floor_run.py` crashed at argparse construction (a help
   string called a function with a placeholder; #1258). The tests exercised the functions, never the parser.
7. **A demotion is a re-date, and a re-date needs new support.** The evidence gate demands a newly cited verdict, or
   an owner exception already ON MAIN, for any date change (`needs_support` in `scripts/lint_evidence_gate.py`).
   An analytical demotion has no record to cite, so it is two PRs: the `kind: "ledger"` exceptions first (#1294),
   then the ledger move (#1296). Prove the clause works in a simulated base before opening the second PR.
8. **Exemption sets in invariant tests are asserted EXACTLY.** A status move silently made one exemption in
   `test_the_real_pass_table_scopes_match_each_rows_current_head` stale (#1283), so the test now asserts equality.
   It caught the next two stranded entries on its own (#1296).

## Operator notes

- **The rig can't sign commits.** Its data commit is re-signed on the dev box with `git commit-tree -S` on the same
  tree, parent, author and message, then pushed with `--force-with-lease`. Verify the bytes against the verdict's
  `data_sha256` first.
- **GitHub's "Update branch" adds an unsigned merge to a PR.** Prefer merging main locally in a signed commit. If
  one appears, check that it is lossless (both parents in your merge, and its tree equal to the automatic merge of
  its parents) before replacing it with `--force-with-lease=<branch>:<that sha>`.

Regression guards: [scripts/lint_bf2_obligation.py](../../scripts/lint_bf2_obligation.py) (a floor-matched B1 verdict
holds T1-13 down) and [tests/unit/test_lint_evidence_gate.py](../../tests/unit/test_lint_evidence_gate.py)
(`test_the_real_pass_table_scopes_match_each_rows_current_head`, exact exemption set). Lessons 2–4 are design-review
practice; no check enforces them.
