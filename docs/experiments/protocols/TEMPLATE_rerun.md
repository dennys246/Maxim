# Re-run pre-registration template (M1b PR 4)

Worked examples: [exp10_rerun_2026-09-30_preregistration.md](exp10_rerun_2026-09-30_preregistration.md),
[exp09_rerun_2026-09-30_preregistration.md](exp09_rerun_2026-09-30_preregistration.md).

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

**Scope:** `rerun_exp<NN>_<YYYY-MM-DD>` (or a campaign name, e.g. `rerun_exp10_o19`, when the run date is not yet known)

- **Ledger row:** `<T1-n / T3-n>` (docs/plans/behavioral_graduation_candidates.md), and the trigger that fired.
- **Claim re-tested:** the row's claim, verbatim.
- **Gate and threshold:** copied from the original result doc, with its link. No change: a changed gate is a
  new experiment, not a re-run. **Written as numbers** over the run's committed bytes (which record, which
  field, which count or comparison), so the verdict is computed, not read. A criterion the run cannot satisfy as
  written is NOT MET, never reinterpreted; one the bytes cannot measure is stated NOT MEASURED.
- **Complete-run condition:** what makes an attempt count (e.g. every session `finish_reason == "max_turns"`, the
  resume chain loaded, one known clean `code_tree_sha256`). Every attempt is recorded, aborts included.
- **Apparatus:** the box, the model profile and `n_ctx` (set through `maxim config`), and the commit the run
  will execute (on `main`, clean tree).
- **Command:** the exact harness invocation, writing to `docs/experiments/data/rerun_exp<NN>_<YYYY-MM-DD>/`. The
  harness hands each sim the harness run id and records every attempt; a hand-run sim cannot back the row.
- **Attempts:** a stop rule fixed in advance (e.g. at most 3 attempts; the first complete attempt decides; the
  harness refuses a new attempt once one is complete), so a run is never retried until it passes.
- **Verdict and row mapping:** each verdict value (e.g. `PASS` / `FAIL` / `ABORT`) and the status it gives the row,
  pre-registered. A typed abort (`planning_failed`, …) is not data, and the row cannot move on it. Name the
  verdict kind and its pass set for the M1b evidence gate (only a stamped verdict supplies new support).
