# Re-run pre-registration template (M1b PR 4)

Copy everything below the `---` line into `docs/experiments/protocols/exp<NN>_rerun_<date>_preregistration.md`
(the `_rerun_` filename marks a re-run pre-registration and carries the experiment token), fill every field, and merge it as its own PR **before** the run. This is for an
experiment whose original run had no pre-registration (Exp 09, Exp 10, most rows before Exp 44). An experiment
that has one declares its re-run with a scoped PRE-DATA amendment instead:
`**Amendment N — <date>, PRE-DATA, for \`<entry>\`, <why>.**`

A re-run pre-registration governs ONLY the entries its Scope line names. It does not reach back over the
original run's data. The prereg lint (`scripts/lint_prereg_precedes_data.py`) checks that the Scope line
reached `main` before each named entry's first record.

---

# Exp <NN> re-run, <YYYY-MM-DD>

**Scope:** `rerun_exp<NN>_<YYYY-MM-DD>`

- **Ledger row:** `<T1-n / T3-n>` (docs/plans/behavioral_graduation_candidates.md), and the trigger that fired.
- **Claim re-tested:** the row's claim, verbatim.
- **Gate and threshold:** copied from the original result doc, with its link. No change: a changed gate is a
  new experiment, not a re-run.
- **Apparatus:** the box, the model profile and `n_ctx` (set through `maxim config`), and the commit the run
  will execute (on `main`, clean tree).
- **Command:** the exact harness invocation, writing to `docs/experiments/data/rerun_exp<NN>_<YYYY-MM-DD>/`.
- **Pass / fail / abort:** what each outcome does to the row. A typed abort (`planning_failed`, …) is not data,
  and the row cannot move on it.
