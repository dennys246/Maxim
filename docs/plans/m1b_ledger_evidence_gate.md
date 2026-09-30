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
- **Citations** are the row's `**Evidence:**` field only (as built in PR 3; the rest of the row is context), parsed by `scripts/_ledger.py`; a GitHub URL form
  fails; `|` inside code spans no longer breaks a row.

### The prereg lint

- `token_of` strips a `rerun_` prefix; `NNdMM` resolves to its parent `NN`.
- A `rerun_*` entry with no governing prereg FAILS unless listed with a reason.
- Amendments may be scoped: `**Amendment N — <date>, PRE-DATA, for <entry>**` is judged only against the
  named entry; an unscoped one against all.
- The echo check requires the result doc to name the entry path or the harness run id.
- *(As built in PR 4: re-runs are recognised by name or declaration, and the exception lists are frozen; see
  "PR 4 as built" below.)*

## Out of scope

Other status surfaces (the CLAUDE.md active-initiatives line, CHANGELOG, release notes): backlog M2.

## PR sequence

1. **`cancel` vs completion** ([#1002](https://github.com/dennys246/Maxim/issues/1002); src; fenced orchestrator touch).
2. **Harness stamping** ([#1003](https://github.com/dennys246/Maxim/issues/1003): Exp 37 and Exp 41 harnesses stamp provenance and their allowance),
   with `record_kind` and `harness_run_id` stamps across sim reports and harnesses, and one hash length.
3. **Ledger normalisation + format lint** (owner reviews every row's status).
4. **Prereg lint** (`rerun_`, scoped amendments, `NNdMM`).
5. **The evidence gate** + the legacy snapshot + the exceptions file, split in two: **5a** stamps the writers
   the gate reads, **5b** is the gate.

## PR 2 as built (#1003, 2026-09-29)

The first approach note was DO-NOT-BUILD in its own design pass: the sim reports a harness spawns live
under the gitignored `data/`, so a row listing `session_id`s names files the gate can never read, and three
harnesses dropped failed runs. Owner decisions (2026-09-29), all the recommended options:

- **E1 echo:** a harness row carries `sims`, the evidence of each report its spawn wrote
  (`_provenance.sim_evidence`), and `depends_on`, the evidence of every session its home already held. The
  gate judges the row alone.
- **E2 failed runs are rows** (`status: "failed"`); readers exclude them, and a row with no `status` (every
  legacy row) is a trial.
- **E3 `record_kind` on evidence writers only:** `sim_report`, `harness_row` (every harness the lint's family
  1 covers, plus Exp 56/57), `verdict` (Exp 56, 57, 60, 61, 62). Any other file fails closed at the gate.
- **E4 lineage:** the report's `provenance.resume` says whether a `--resume-sim` loaded (both resolutions
  agree and every store the prior session wrote was restored); the harness's `depends_on` is the lineage
  authority. Exp 41 gets a fresh home. Exp 37 Arm A and the Arm C prior do too: the same contamination,
  found while building this PR by the new `depends_on` stamp, approved by the owner at review (2026-09-29).
- **Fence:** the resume stamp is an owner-approved fenced touch of `orchestrator.py`, paid for by extracting
  the restore to `_restore_aut_from_session` (`start_simulation_mode` 3313 -> 3277).

What PR 5 can rely on, and must apply (from the PR 2 review round):
- **A row is judged only if it says how it ended.** `record_kind` present and `status` absent means the
  writer does not stamp an ending (Exp 44's `campaign_start`, Exp 49/56/57 rows, the instrument checks):
  fail closed as sim-run evidence (since PR 5a, the Exp 49/56/57 rows and the instrument checks do stamp one;
  Exp 44's `campaign_start` still does not). `status: "failed"` is never evidence. A row with no `record_kind` and no
  `status` is legacy (the frozen snapshot).
- **An ok row with no `sims`, or an empty list, carries no sim evidence** (an in-process record, or a mock
  run; mock rows also say `mock: true`, Exp 37 included). Judge the sims, not the row's claim about them.
- **Bind each sim to its row by code, not by name:** `sims[].code_tree_sha256` equals the row's
  `provenance.code_tree_sha256`, and read `sims[].working_tree_dirty_src_scripts` (the prereg lint reads
  only the row's own block, so a sim that went dirty mid-fire shows only there). The key is `record_kind`;
  `_record_kind` is the `actions.jsonl` header marker.

Also: the run id is a JOIN KEY, not evidence (a report binds to a harness only when a harness
row names its session); every hash is the full commit id; a harness row's own `provenance` block is the one
`executed_code_provenance` returned (with `allow_dirty` exactly when granted), nested under `provenance` in
every family-1 harness (the legacy per-row `git_hash` field says where the harness lives and is never read). The design pass's round-2 and round-3 folds are in
the PR. Follow-up: [#1009](https://github.com/dennys246/Maxim/issues/1009) (one resolution for
`--resume-sim`).

## PR 3 as built (2026-09-30)

The ledger's two status tables now carry stable row IDs (`T1-1`–`T1-15`, `T3-1`–`T3-20`). Each row opens with
`**Status: <TOKEN> <date>**`, from the closed vocabulary with ranks written in the ledger's own section. A
positive row carries an `**Evidence:**` field of tracked records. The rules are enforced by
`scripts/lint_ledger_format.py` through the shared parser `scripts/_ledger.py`, which PR 5 imports; the lint
replaces `lint_claude_md_invariants.py`'s check 5. The design pass ran two rounds: v1 was DO-NOT-BUILD (the
Evidence field had no defined shape, and dates could be laundered). Its folds are the rules below.

- **Evidence grammar:**
  - one field per row, straight after the Status line, holding only links or code spans;
  - every entry tracked by git and not a symlink;
  - a directory counts only as a session directory holding a `report.json`;
  - refused: scripts, markdown, READMEs, the data root, and aborted, invalid or non-gated names;
  - link text that looks like a path must name its own target.
- **Dates against the merge-base:** never later than today; never moving back; a raise needs a later date;
  a new row is not backdated before the base.
- **LEGACY** can be kept, never entered.

Owner decisions (2026-09-30, all the recommended options): T1-1 (Exp 10) and T3-9 (Exp 09) are `STALE`, because
their re-runs were typed aborts, and this blocks the 1.3.2 cut until they are re-run (`outstanding.md` O19). EC
pattern completion is `LEGACY`, since its data is lost. Two tokens are new: `SUPERSEDED` (T1-8, by T1-9) and
`TIER-2` (T3-17–T3-20). T1-6, T1-7 and T1-15 keep their status with the caveat written in the row.

**For PR 5:**
- A raise is a rank increase (`_ledger.is_raise`); a move between positive tokens also needs a new date (`_ledger.needs_new_date`). A new ID arriving at rank ≥ 1 is a raise from rank 0, and its previous date is
  the merge-base commit's date.
- A date change on a positive row is a trigger, like a raise.
- The claim cells are Tier 1's `Claim`, and Tier 3's `Bio-claim` and `Graduation predicate`.

## PR 4 as built (2026-09-30)

The prereg lint now governs re-runs. Before this, `rerun_exp09_…` and `rerun_exp10_…` had the token `rerun`,
and `42d53`, `52d53` and `53d53` had no parent mapping, so all of them were silently out of scope.

- **Recognising a re-run:** by its name (`rerun_…`, `NNd<MM>`, or a replication / re-baseline / rerun word on
  a governed entry), or because a pre-registration declares it.
- **Declaring a re-run:** a re-run needs a scoped PRE-DATA declaration before its data. That is an amendment
  `for \`<entry>\``, or a re-run pre-registration's `**Scope:**` line (`protocols/TEMPLATE_rerun.md`). A
  re-run pre-registration governs only the entries it names.
- **Timing:** declarations are timed by walking the pre-registration's history. This closes re-scoping, the
  POST→PRE relabel and the header-needle collision.
- **Frozen exception lists:** the lists only shrink. Exceptions after this PR go through PR 5's exceptions
  file.
- **For PR 5:** `classify()` / `--json`.

The design pass ran two rounds; v1 was DO-NOT-BUILD, because of a dangling scope, the `-S` timing and
in-script self-excuse.

Owner decisions (2026-09-30), all the recommended options:
- The existing re-runs are listed by path with reasons: 52d53, 53d53 ×2, the 53b R1 replication and the Exp 56
  re-baseline as grandfathered; 42d53 ×2 and the Exp 09 / Exp 10 re-runs as ungoverned re-runs.
- A re-run is recognised by name or by declaration.
- A minimal re-run pre-registration template.

For PR 5:
- A GRANDFATHERED or UNGOVERNED_RERUN entry is not the sole support for a raise (decision 7's legacy rule).
  T1-6 (42d53) and T1-9 / T1-10 (52d53, 53d53) rest on such entries.
- Read `classify_all()` or `--json` (a versioned envelope: `entries` with each entry's status, re-run flag,
  data time and its source, first commit, run ids, dirty/allowance flags, governing pre-registrations and what
  declared it; plus `failures`). Fail closed on any status outside `STATUSES`, and on a non-empty `failures`
  list: a failure no single entry carries, such as a foreign declaration or a changed listed path, appears only
  there, while every entry can read PASS.
- Wire the exceptions file into this lint too (an `EXCEPTED` status). Otherwise, now that the in-script lists
  are frozen, a future legitimate exception has no path.

## PR 5a as built (2026-09-30)

Every record the gate will read from the writers below now says what it is, stamped where it is written:

- **Harness rows** go through `scripts/_provenance.py::stamp_harness_row`: `record_kind: "harness_row"`, `status`
  and an explicit `mock`. `status` is `failed` when the row carries a `refusal` (`is not None`, the verdicts' own
  reading) or already said so, else `ok`; any other existing `status` is refused rather than rewritten. Users: the
  survival writers (Exp 60, 61, 62, R3), the Exp 56/57 campaigns and instrument checks, and the Exp 49 trials (a
  trial whose `maxim` exited non-zero is `failed`; the scripted arm is `mock`). Exp 37's failed row (`mock` now a
  required argument) and the Exp 44 manifest (`mock` = `--dry-run`) carry `mock` too.
- **Verdicts** go through `stamp_verdict`: `record_kind: "verdict"`, a `kind` (`exp56_verdict` … `exp62_verdict`;
  the gate owns each kind's pass values), `data` repo-relative, `data_sha256` over the bytes the verdict parsed
  (the caller reads the file once), and a `scope` naming every row the verdict read: its selectors (`run_ids`,
  `campaign_id`, which covers all of a campaign's kinds, since the Exp 61/62 verdicts read apparatus and donor rows
  for drift and the one-code-hash check), or `{"all_rows": true}`. An empty scope or a `None` selector is refused.
  The Exp 56/57 analyzers stamp their own `provenance`.
- **`harness_family`** is stamped inside the provenance block by `_provenance` (`in_process` / `spawning`). Exp
  56/57 import `maxim` and never spawn it, so they moved to `in_process_code_provenance(repo, maxim.__file__)`
  (the console-script probe described a different package). A provenance failure there, or in their analyzers,
  exits 3 (a refusal), never 1 (FAIL); their `--mock` runs are no longer exempt. `lint_harness_provenance.py`
  refuses `in_process_code_provenance` in any file that spawns `maxim`, and the `"harness_family"` literal in any
  file under `scripts/` but `_provenance.py`: presence checks, which catch forgetting, not evasion.
- The harness edits fire no ledger `Re-run on:` trigger and match no pre-registration's pinned hash.

For PR 5b:
- **Not stamped yet (fail closed at the gate):** `orient_backbone/exp53_cross_context_readout.py` (T1-10's
  `53d53_*`), the h1 DoA sweep / part-c writers (T1-7), the cradle phase-A scripted output (T1-9), `exp58_run.py`,
  `exp58_offline_gates.py`, `exp60_water_check.py` (the gate-(ii) records Exp 60/62 cite), `survival_world/
  instrument_check.py`, `exp62_precheck.py`, and R3's report and gauntlet.
- The Exp 49 rows carry no `sims` (they spawn `--mode live`, not `--sim`), so under the spawning-family rule
  they can never be ESTABLISHED.
- R3's report re-admits rows whose only refusal is the tick band (Amendment 2). Those rows are stamped `failed`,
  so a gate that skips failed rows never judges rows the report counts: less strict, not more.
- An instrument check is stamped `ok` whether or not `all_pass` holds; `ok` says how the run ended, not what it
  measured.
- `kind` means the row type on survival rows ("receiver") and the verdict kind on verdicts; key on `record_kind`
  first. `stamp_verdict` resolves `data` through a symlink, so the gate's no-symlink rule reads the verdict's
  own path.
