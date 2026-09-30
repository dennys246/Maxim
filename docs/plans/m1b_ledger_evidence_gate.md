# M1b — the ledger evidence gate (design, after the adversarial pass)

**Status:** design agreed 2026-09-29 (owner), after the adversarial design pass CLAUDE.md requires for a
gate ("Faster without lowering the bar" (1)). The first draft was red-teamed before any code and was
DO-NOT-BUILD; this is the revised shape. Build follows the PR sequence at the end.
**Closes:** the second half of mechanization backlog M1 (`docs/plans/outstanding.md`).
**Builds on:** M1a (#999: sim `report.json` `provenance` + `ts`), #998 (#1001: `utils/code_tree.py`).

## What it protects

A row of `docs/plans/behavioral_graduation_candidates.md` (the ledger) moving to, or staying at, a positive
status on evidence that cannot be established: a run whose code, model or context size is not stamped,
whose tree was dirty without a human's allowance, or that ended in a typed abort or an error. CLAUDE.md's
"weak evidence never gates" is enforced by attention today. It catches forgetting, not evasion: an author
can still cite a clean but irrelevant record, and review is the check for that.

## Why the first draft failed (red-team, 2026-09-29)

- **Status cells are prose.** Replayed over the ledger's history, "the status token changed and a
  citation was added" fired on 4 events and missed every re-validation that kept its leading word --
  including the Exp 09 re-run whose session ended `planning_failed` (#875), the case M1 exists for.
- **Claim text is not an identity.** Keying rows by claim text turned honest corrections (#934) into
  "new rows".
- **Judging every link penalises disclosure.** Rows link aborted and discarded data on purpose.
- **`cancel` is not an abort in practice.** The orchestrator stamps `cancel` when a runner simply finishes
  (27 of 31 committed `report.json`, all of `37_results.jsonl`).
- **The prereg `rerun_*` rule governed nothing** (Exp 09/10 have no prereg) and an appended PRE-DATA
  amendment retroactively fails an experiment's original data.
- **The dirty-sim binding could not work:** hash lengths differ between harness and sim stamps, no
  harness records a run window, and two harnesses (Exp 37, Exp 41) do not stamp their allowance at all.

## Owner decisions (2026-09-29)

1. Exceptions live in ONE committed file, **merged on `main` before** the change they excuse (squash
   erases commit order), pinned to the record's sha256, bound to a row ID and a status transition, and
   append-only. Stale = the path is missing, its sha256 no longer matches, the row ID is gone, or the
   record now passes without it. An entry for a transition that is no longer current is inert history.
2. A re-run's data needs a governing prereg and its own PRE-DATA amendment, scoped to the entry.
3. Not citable: `finish_reason` in the typed aborts, `error`, or `unknown`.
4. A dirty-tree allowance lives only in the harness record a human granted; a sim report never carries
   `allow_dirty`.
5. **Fix the `cancel` conflation in the orchestrator first** (a small fenced touch with its own tests):
   runner completion and an operator stop get different codes.
6. **Normalise the ledger first:** stable row IDs, a dated status from a closed vocabulary, an explicit
   `Evidence:` field. The owner reviews every row's assigned status.
7. **Legacy records** (committed before M1a, unstamped) are frozen by hash into a snapshot that can only
   shrink; they report as a NOTE, never a failure, and can never be the sole support for a raise.
8. **Tests-only re-validation** gets its own status token (`RE-VALIDATED-BY-TESTS`), outside the set the
   gate treats as run-evidence-backed, and named as unchecked by this gate.

## Shape

### The ledger (format lint, every run, not diff-scoped)

- Every table row carries a stable ID in its first cell (`T1-1`, `T3-9`); IDs are unique and never vanish.
- The status cell opens with `**Status: <TOKEN> <YYYY-MM-DD>**` from a closed vocabulary, e.g. `EARNED`,
  `MAINTAINED`, `RE-VALIDATED`, `RE-VALIDATED-BY-TESTS`, `PARTIAL`, `STALE`, `BROKEN`, `DORMANT`,
  `DROPPED`, `SETUP`, `BORDERLINE`. Annotations (trigger walks, history) follow it. The table is
  authoritative; walk prose is not.
- A positive status (`EARNED`, `MAINTAINED`, `RE-VALIDATED`) requires a non-empty `**Evidence:**` field
  of files or session directories under `docs/experiments/data/`. A README is context, not evidence. Other
  data links in the row are context.
- `scripts/lint_claude_md_invariants.py::_ledger_earned_violations` folds into this lint.

### The evidence gate (diff-scoped against `origin/main`)

- **Triggers:** a row's status class is raised; a positive row gains an Evidence entry; a positive row's
  claim cell changes; a file under a positive row's Evidence path changes in the diff.
- **Judged:** every Evidence record of a triggered row. A raise also needs at least one NEWLY cited,
  fully stamped record whose `ts` is after the previous status date.
- **Records** are classified by an explicit `record_kind` stamp -- `sim_report` (written by
  `simulation/report.py`), `harness_row`, `verdict` -- and a closed per-file classification; an unknown
  file fails closed. Inside a session directory the `report.json` is the record (logs are not parsed; a
  session directory without one fails). A `.gz` outside a session directory is decompressed and judged.
  Symlinks are refused. A verdict is judged through its `data` source.
- **A sim report is established when:** `record_kind` and `provenance` present; `executed_git_hash` not
  `unknown`; clean, or dirty and bound (below); `code_changed_during_run` false; `ts` present; each role
  that ran names its profile and context (`{role}_profile`, `{role}_router_n_ctx`, `configured_n_ctx`);
  `finish_reason` citable (decision 3, after decision 5's fix).
- **A dirty sim is bound by identity, not time:** the harness exports a run id (env var, stamped as
  `provenance.harness_run_id`; the conftest scrub in the same commit), its rows list the `session_id`s it
  spawned, and the binding needs the session named, equal run ids, and equal non-`unknown`
  `code_tree_sha256`. Hashes are compared at full length (the stamps are normalised to one length).
- **Citations** are extracted from the whole row line (links and backticked paths); a GitHub URL form
  fails; `|` inside code spans no longer breaks a row.

### The prereg lint

- `token_of` strips a `rerun_` prefix; `NNdMM` resolves to its parent `NN`.
- A `rerun_*` entry with no governing prereg FAILS unless listed with a reason.
- Amendments may be scoped: `**Amendment N — <date>, PRE-DATA, for <entry>**` is judged only against the
  named entry; an unscoped one against all.
- The echo check requires the result doc to name the entry path or the harness run id.

## Out of scope

Other status surfaces (the CLAUDE.md active-initiatives line, CHANGELOG, release notes): backlog M2.

## PR sequence

1. **`cancel` vs completion** ([#1002](https://github.com/dennys246/Maxim/issues/1002); src; fenced orchestrator touch).
2. **Harness stamping** ([#1003](https://github.com/dennys246/Maxim/issues/1003): Exp 37 and Exp 41 harnesses stamp provenance and their allowance),
   with `record_kind` and `harness_run_id` stamps across sim reports and harnesses, and one hash length.
3. **Ledger normalisation + format lint** (owner reviews every row's status).
4. **Prereg lint** (`rerun_`, scoped amendments, `NNdMM`).
5. **The evidence gate** + the legacy snapshot + the exceptions file.
