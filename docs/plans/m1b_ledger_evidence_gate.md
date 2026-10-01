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
  since PR 5a-2 Exp 44's `campaign_start` is a `harness_header`, never a run). `status: "failed"` is never evidence. A row with no `record_kind` and no
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

Owner decisions (2026-09-30), after the review round and a design pass on widening 5a:
- **Widen, as PR 5a-2 before 5b:** stamp every writer below. Its design pass found per-line status in event
  logs fails open (data lines of an aborted run read `ok`), so 5a-2 gets a run-level terminal status per
  `harness_run_id`, verdicts out of the exp53 log, a `diagnosis` kind, and its own design round.
- **Instrument checks** get `record_kind: "instrument_check"` with a `pass` field the gate requires true (5a-2
  moves the Exp 56/57 checks off `stamp_harness_row`); the Exp 44 `campaign_start` row becomes
  `record_kind: "harness_header"`.
- **Event logs are judged per run group**, not per file; row files and verdicts stay per file. (Superseded in
  detail by PR 5a-2: the group key is the per-log `log_run_id`, never the process-wide `harness_run_id`.)
- **`JsonlLog` refuses everywhere** (exit 3) when the imported `maxim` is not this repo's src, scratch logs too.
- **For 5b:** only `harness_row`, `verdict` and `sim_report` count as new support; an instrument check, diagnosis
  or header never does. (Extended by the owner decision of 2026-09-30 recorded in PR 5a-2: `harness_event` too.)
- R3's re-admitted rows stay `failed` (noted below).

For PR 5b (**Superseded for 5b by "PR 5b design: owner decisions (2026-09-30)" below** (only a stamped verdict supplies new support; R3's report and gauntlet are stamped by 5a-3). Kept as the record of what was decided at the time.):
- **Not stamped yet (fail closed at the gate; PR 5a-2 — see its section for what it stamped):** `orient_backbone/exp53_cross_context_readout.py` (T1-10's
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

## PR 5a-2 as built (2026-09-30)

The writers PR 5a left unstamped now stamp too. The design took five adversarial rounds: rounds 1 to 4 each found a
fail-open hole, and round 5 closed. The code review round (executor + architecture) then folded its findings below.

Owner decisions (2026-09-30), in order:
- **Event logs are support, per run** (design round 4): `harness_event` lines count at 5b, judged per `log_run_id`
  group — exactly one terminal `ok`, the group's own provenance, no mock line.
- **Evidence vs non-support** is declared per `JsonlLog` (round 4).
- **Review round:** `settle_s` is EXEMPT for the Exp 56/57 checks, not frozen (a blind wait before a read can only
  lower a pass; no pre-registration names a value); provenance rides ONCE per run (first and terminal line) with a
  `provenance_sha256` on every line; the live harnesses (Exp 58/60/61, R3) authorize only on a stamped, real,
  passing instrument check (`_provenance.instrument_check_authorizes`), no longer on `all_pass` — so a pre-M1b
  apparatus record is refused and the check must be re-run before the next live campaign (R3 reads the DATED
  record `exp60_water_apparatus_2026-09-17.json`, Exp 60/61 `exp60_water_apparatus.json`: the re-check's `--out`
  must be the path each harness reads, #1019); a verdict's `mock` is
  judged over the WHOLE file for every verdict writer (smokes go in their own files).

- **Event logs** (`orient_backbone/live_common.py::JsonlLog`, 10 constructions): `mock` and `evidence` are
  required keywords. Every line carries `record_kind`, `mock`, a `log_run_id` minted fresh per log (never the
  process-wide harness run id: gate6 runs several Exp 53 phases in one process) and `provenance_sha256`; the full
  `provenance` block (taken once) rides on the run's first line and its terminal line. Stamps win over caller
  fields, and a caller's `provenance` must equal the log's own. The log refuses
  (exit 3) for any path when the imported `maxim` is not this repo's src.
  - **Non-support** (`evidence=False`; `orient_demo`, `exp53_demo_readout`, `ear_map`, `loudness_bench_poll`,
    `live_2_reactive`, `doa_settle`): `record_kind: "harness_demo"`, never support, no terminal status. The gated
    dirty-tree refusal still applies.
  - **Evidence** (`evidence=True`; `doa_sweep`, `delivered_shift_block`, `live_3_learn`,
    `exp53_cross_context_readout`): lines are `harness_event`; each run ends in exactly ONE `harness_run_end` line
    carrying `status` and `end_code_tree_sha256`. It is `ok` only through an explicit `finish("ok")`, written as
    the last statement of the `with JsonlLog(...)` body. An exception, an early return, a `close()`, a clean exit
    without `finish`, or an abort-class event (`abort`, `*_aborted`, `mark_aborted`) ends it `failed`, so a
    caught-and-continued abort (`live_3_learn`'s lost robot, `delivered_shift_block`'s Ctrl-C) can never end `ok`.
    Exp 53 finishes `ok` on rc 0 or 6: a Gate-I or Gate-C FAIL is a computed result, not a refusal.
  - **Where the run's code lives:** each evidence caller's body moved into a helper that receives the log
    (`doa_sweep::_sweep`, `delivered_shift_block::_block`, `live_3_learn::_learn`, exp53 `_run_logged`), so the
    design's handler rule is applied there: every `except` in a function taking a `JsonlLog` re-raises, writes an
    abort-class event or calls `mark_aborted`, or sits inside a handler that does (pinned by an AST test). A
    handler in a function the helper CALLS is outside it (stated below).
  - **Deviation from design N-c:** the helpers keep their early `log.close()` calls. It is sound — closing an
    unfinished evidence log can only write `failed` — but the terminal reason then reads "closed without finish"
    rather than the specific refusal. Early refusals now append a lone `failed` terminal line (no `run_id`).
- **The Exp 53 verdict** is its own record (`<records stem>_verdict.json`, `--verdict-out`, refusing an existing
  file unless `--overwrite`; `_format_version` and `ts` stamped). gate6 passes `--overwrite` (it owns those records)
  and reads a gate-T verdict only when its `data_sha256` matches the records' current bytes, so a verdict left by
  an earlier run is never read. It is no longer appended to the records it judged, which would change the bytes
  its hash names. Its scope is `{"run_ids": [...]}` over the exp53 run ids it used; it is `mock` when any line of
  the file is mock or does not say, and `scoped_lines_stamped` says whether every scoped line carries `log_run_id`,
  `mock` and `provenance_sha256`, and its run group has a line carrying the block that digest names
  (`exp53::_lines_stamped`; fixed in #1019 — the first cut required the block on every line, which provenance-once
  made false for every real run). Its in-run `gate_I` line stays in the records; gate6 reads gate T from the verdict file.
- **Verdicts** (`stamp_verdict`) take a required `mock`: true when any row of the file is mock or unstamped, or
  there are no rows (`any_not_stamped_real`); an empty selector (`[]`, `""`) is refused like a `None` one. A re-verdict over the committed, pre-stamp records therefore reads `mock: true`: unknown
  is mock. Read it that way, not as "the run was a smoke".
- **Instrument checks** (`stamp_instrument_check`): `record_kind: "instrument_check"`, `status` (how it ended),
  and `pass` (what it measured, only at frozen parameters, and never on a failed run). The survival check
  `CYCLES = 20` and the water check `CYCLES = 3` are frozen (`--cycles 1` passes easier); the Exp 56/57 checks
  and the Exp 58 offline gates have no pass-relevant flag (`settle_s` exempt, above). The live harnesses read
  `pass` through `instrument_check_authorizes`.
- **Diagnoses** (`stamp_diagnosis`, never support; `stamp_diagnosis` and `stamp_instrument_check` refuse an
  unknown existing `status` like `stamp_harness_row`): the L11 geometry probe (its code provenance moves to
  `code_provenance`) and the Exp 62 precheck. The precheck now takes its provenance, including the gated
  dirty-tree refusal, before any world time, gains `--allow-dirty`, and records an `InstrumentError` as a
  failed diagnosis instead of dropping it (refusing an existing `--out` unless `--overwrite`, so a failed
  attempt never erases an earlier record). Pre-5a-2 L11 records keep code provenance under `provenance.code`:
  legacy; 5b reads only `code_provenance`.
- **Exp 52 Phase A** (`9_hunger_relief_orient.py`): `FROZEN_PARAMS` from its pre-registration. A run at any other
  value reads `NOT_FROZEN`, never `PASS`. A VOID run is `failed`, and the report gains `ts`, `verdict` and
  `frozen_params`.
- **Headers** (`stamp_harness_header`, no status, never support): the Exp 44 `campaign_start` row. **Exp 58 rows**
  are stamped with `stamp_harness_row`.

For PR 5b (**Superseded for 5b by "PR 5b design: owner decisions (2026-09-30)" below** (only a stamped verdict supplies new support; R3's report and gauntlet are stamped by 5a-3). Kept as the record of what was decided at the time.):
- Support kinds: `harness_row`, `harness_event` (judged per `log_run_id` group: exactly one terminal `ok`, no mock
  line, every line's `provenance_sha256` equal to the digest of the group's block), `verdict` and `sim_report`.
  The digest is `live_common.provenance_digest`: sha256 of `json.dumps(block, sort_keys=True)` (default
  separators, `ensure_ascii`) over the block as stamped, `allow_dirty` included; 5b imports it rather than
  re-serialising. A code-tree digest reading `unknown` (git failed, at the start or the end) never matches
  anything, itself included.
  `instrument_check`, `diagnosis`, `harness_header` and `harness_demo` never count. Pass sets: `exp53_verdict`
  `{PASS}`; Exp 52 Phase A rows (`experiment: "exp52_phaseA_scripted"`) `{PASS}` read from `verdict`.
- **A cited instrument check must hold** (owner decision, 5a): `status: ok`, `pass: true`, `mock: false`. It never
  supports a row by itself, but a cited failing one sinks the row. T1-11 cites `56_phase0.json`.
- **Unjudgeable records, stated:** `h1_partc_summary.json` (T1-7) is hand-authored and has no writer, so it stays
  legacy forever; T1-7 can be re-earned only through a new `doa_sweep` (evidence) run. R3's report and gauntlet
  (`r3_run.py::report` / `write_gauntlet`) stay unstamped: uncited, fail closed. Every committed record today is
  legacy until re-run.
- An Exp 53 verdict joins its `run_ids` to every `log_run_id` group holding them; each group must be ok, and
  each scoped `run_id` must have at least one line.
- **Inputs that could lie:**
  - The abort latch is a naming convention (`abort` / `*_aborted`). The handler rule covers functions that take
    the log; a handler in a function they CALL that swallows a failure and lets the helper return 0 would still
    reach `finish("ok")`.
  - `finish` itself is caller-declared.
  - Each writer's `passed` rule and `FROZEN_PARAMS` are the writer's own.
- gate6 appends to its records; a second gauntlet into the same records file (e.g. with
  `--write-experiment-results`) holds two complete runs per phase and its verdict refuses (`VerdictError`). That
  fails closed and gate6 is uncited, so it stays (#1019 item 7); give each gauntlet a fresh records file.
- **Owner note:** `doa_settle` (the 0.23 s convergence figure cited in T1-7 and Exp 45) and `loudness_bench_poll`
  (`h2_loudness_bench.jsonl`) produce cited measurements but are non-support logs, so those findings can never be
  gate support. **Owner decision (2026-09-30): both stay non-support.** `doa_settle` measures the instrument (how
  long a DoA reading takes to converge), not behaviour, so it must never be the new record that advances a row;
  the kind that fits is an instrument check, which needs a frozen pass bound no pre-registration names yet.
  **Revisit when** a prereg freezes a settle wait or a claim depends on a convergence bound: then `doa_settle`
  becomes an instrument check with that frozen pass rule, never evidence. Neither record is in any ledger
  Evidence field today, so nothing is blocked.

## PR 5b design: owner decisions (2026-09-30)

The 5b design pass ran four rounds on an addendum to the closed PR 5 design. After three rounds in a row found
new failure modes in making RAW files count as support (raw rows of a failed run, an unidentifiable experiment
family, a family binding that shared writers get wrong), a bird's-eye audit led to a scope reset:

- **Only a stamped verdict supplies NEW support.** Raw rows files, event logs and sim reports are judged when
  cited (they must be established) but never count by themselves; a verdict's `kind` names its experiment, and
  the gate-owned pass table (`docs/experiments/evidence_pass_table.json`, read from the merge-base) decides.
  No writer-declared family machinery.
- **Stated losses until their analyzers stamp a verdict:** T1-7 (doa_sweep), T1-9 (cradle Phase B — Exp 52
  Phase A is judged, never support), Exp 37/41/42/44 rows, Exp 49 (unsupported: live mode stamps no start
  provenance, #1023).
- **T1-1 (Exp 10) and T3-9 (Exp 09)**, STALE and blocking 1.3.2, re-validate through re-run verdicts built in
  **O19's own PR before 5b**: re-run preregs that write each prose gate as numbers (Exp 09 = the ORIGINAL
  2026-04-25 H1–H7; halves that cannot be measured or pass are stated NOT MET), spawning harnesses that record
  every attempt and hash the measured files, and the gate's source hash stamped.
- **Strict defaults chosen by the owner:** a mock, unstamped or foreign-kind line sinks the whole cited file
  (event logs too; failed/aborted runs are only excluded); one `code_tree_sha256` per cited file and per verdict
  scope; cited non-support kinds are judged and a failure sinks the row. **Strict default adopted in the design
  pass (not separately asked):** a verdict that leaves out a complete run of an arm it scopes is not established.
- **What 5b must also apply** (from the design notes):
  - Pass table entries: `exp53_verdict`, `exp54_verdict`, `exp56_verdict` (only with `noop_kit.kit_pass` true —
    the gate's own check, beside the analyzer's guard), `exp57_verdict`, `exp60/61/62_verdict`, and O19's
    `exp10_verdict` / `exp09_verdict`. An entry added in the same PR counts nothing.
  - "Complete run" per experiment: Exp 60 — a row for every seed in the row's `frozen.seeds` under one
    `(run_id, arm)`, refused rows INCLUDED (so "re-run until no refusals" cannot escape); Exp 53 — re-derive
    `runs_of` / `Run.status` from the bound bytes (never trust `runs_excluded`), filtered to the verdict's
    `experiment` and the phases/conditions its gate uses; Exp 61/62 — one campaign per file (5a-3), in-campaign
    duplicates and gaps are `compute_verdict`'s INCOMPLETE. **Stated limit:** a retry under a NEW campaign id
    (a fresh file) is invisible to the gate, and one campaign per file makes that the easy way to drop a failed
    campaign quietly — review is the check (the evasion class).
  - Sim-spawning rows: the gate checks `sims[].code_tree_sha256` (equal across the scope, equal to the row's
    harness provenance, not unknown/dirty), not only the harness block.
  - Exp 56/57 verdicts go to stdout only: the committed verdict is the operator's redirect.
  - A record's time is the min `ts` over its counted units (a row's `ts`, else its `sims[].ts`; never a terminal's
    or a verdict's own); a digest reading `unknown` (or `unknown:`) never matches anything.
- **O19's re-run requirements:** a re-run prereg per experiment writing the gate as numbers (Exp 09 = the
  original H1–H7; halves that cannot be measured or pass stated NOT MET), stamped with the prereg blob sha; a
  spawning harness that sets `MAXIM_HARNESS_RUN_ID`, records every attempt (failed rows included), copies AND
  hashes the measured files (`run_log.jsonl`, `aut_hippocampus.json`), asserts `finish_reason == "max_turns"`,
  one code tree and the resume chain; a verdict writer stamping the gate function's source hash.
- **Still owed:** a delta design round on the 5b notes' v6 folds before 5b is built (after O19).

## PR 5a-3 as built (2026-09-30)

The last writer fixes before 5b.

- **One campaign per file (#1022):** Exp 61/62 and R3 default `--out` to a file named by the campaign
  (`_provenance.campaign_out_path`: `exp61_pairs_<id>.jsonl`, `exp62_rows_<id>.jsonl`, `r3_cal_<id>.jsonl` /
  `r3_bench_<id>.jsonl`); `--resume` needs `--campaign-id` and refuses a missing file. Exp 58/60 require `--out`
  (Exp 60's two arms go into the operator's one file). Exp 62 `run` requires `--campaign-id` and a replay row for
  it in the file. R3's gauntlet is keyed by campaign (`r3_gauntlet_<id>.json`), and bench needs `--campaign-id`
  to read its cal campaign's gauntlet.
- **The append refusal** (`_provenance.append_refusal`, every survival row writer): a file whose existing lines
  are not JSON, unstamped (pre-M1b), or from another `code_tree_sha256` takes no append (exit 2). A resume on the
  same tree (docs-only commits included, since the digest covers `src/` + `scripts/`) appends freely; the
  verdicts' own one-`executed_git_hash` check stays stricter.
- **`stamp_harness_row` requires an epoch `ts`** (the run's time, set by its writer); Exp 49 rows gain theirs.
- **The Exp 53/54 verdict kind** comes from the scoped runs' `start` lines (`experiment`), via
  `VERDICT_KIND_BY_EXPERIMENT`: `exp53_verdict`, `exp54_verdict`, `gate6_exp53_verdict` (uncited). A mix or an
  unknown experiment refuses; runs that predate the field (the 53b R1 replication) get
  `exp53_unlabelled_verdict`, which no pass table names.
- **Companions stamped:** the Exp 53 manifest and targets declaration are `harness_header`s (with code
  provenance); R3's gauntlet and `report --json` are `diagnosis` records (R3's COMPLETE/INCOMPLETE moves to
  `r3_status`, since a stamped `status` says how the run ended); the gate6 payload is a mock `diagnosis`.
- **Exp 56's analyzer** turns a PASS without the no-op kit into NO-VERDICT (Exp 57's guard).

Deviations from the closed design, and consequences, stated:
- **`exp53_unlabelled_verdict`** (design said "absent → refuse"): runs that predate the `experiment` field still
  get a computable verdict for analysis, under a kind no pass table names. T1-10 cites `53d53`, whose runs carry
  the field.
- **gate6:** its in-process Exp 53 verdicts are `gate6_exp53_verdict` and its payload a MOCK `diagnosis` (it runs
  on the dry rig; uncited).
- **R3 bench shares its cal campaign id:** one bench per cal; bench rows carry the cal id (the 1.3 campaign used
  `r3-bench-1`, so its prereg runbook command no longer finds a gauntlet — refused with a message); re-benching on
  a new tree needs an explicit `--out` under the same id; `--campaign-id` is required even with `--gauntlet`.
  `_cal` still writes the gauntlet straight under `docs/experiments/data/` without the gated-write check
  (pre-existing; it no longer overwrites a committed file).
- **R3 report:** R3's COMPLETE/INCOMPLETE moves to `r3_status` and the file is `_format_version: "1.1"` (owner
  decision); a pre-1.1 report reads `status` as R3's own value.
- **Exp 62:** `run` refuses unless the file holds an UNREFUSED replay row for its campaign. Without
  `--write-experiment-results` each step is redirected to a fresh temp file, so offline use needs the same
  explicit `--out` on every step (the refusals say so).
- `stamp_diagnosis` / `stamp_instrument_check` still default `ts` to write time (neither is support);
  `stamp_harness_row` requires the writer's.
- Exp 56/57 and cradle `--resume` into an existing `--out` get no append refusal yet; at 5b the one-code-per-file
  rule fails such a file closed.
- The frozen preregs (`r3_survival_benchmark_prereg.md`, `exp60…prereg.md`, `exp61…prereg.md`) keep their
  original commands; a re-run prereg must use the new ones.


## PR 5b build spec (consolidated 2026-10-01; design closed after nine rounds)

The base design (PR 5 approach, three rounds) plus the 5b addendum (rounds 1–6 and two delta checks), merged
into one text. **5b ships in two PRs** (owner decision 2026-10-01): **5b-1**, the gate core, below; **5b-2**,
the per-experiment "complete run" rules for Exp 53/60/61/62, whose pass-table entries stay out until it lands
(those kinds fail closed meanwhile).

### Files
- `scripts/lint_evidence_gate.py` — the gate (CI lint job; diff-scoped against the merge-base).
- `docs/experiments/evidence_pass_table.json` — `{kind: {"rows": [IDs], "targets": {TOKEN: [verdict values]},
  "require": {dotted.field: scalar}}}`, read from the MERGE-BASE (absent there = empty: no support, never
  unrestricted). 5b-1 entries: `exp10_verdict` (T1-1; MAINTAINED: PASS; require `apparatus_checked: true`),
  `exp09_verdict` (T3-9; MAINTAINED: PASS, PARTIAL: PARTIAL; require `apparatus_checked: true`), `exp56_verdict`
  (T1-11; RE-VALIDATED: PASS; require `noop_kit.kit_pass: true`), `exp57_verdict` (T1-12; EARNED: PASS, PARTIAL: PARTIAL).
- `docs/experiments/evidence_legacy.json` — `{path: sha256}` of every tracked file under `docs/experiments/data/`
  first committed before the M1a cutoff (merge of #999, 2026-09-29T22:41:16Z) whose bytes (decompressed for
  `.gz`) carry no `"record_kind"`. Shrink-only against the base; a key whose file is gone or changed FAILS.
- `docs/experiments/evidence_exceptions.json` — owner-named overrides (empty at landing), read from the merge-base,
  append-only.

### Triggers (per row ID, base ledger vs HEAD ledger)
A row is JUDGED when its HEAD token is positive (EARNED/MAINTAINED/RE-VALIDATED), or its change is a raise into
PARTIAL, AND any of: a raise (`is_raise`, a new row included); a move between positive tokens; a date change; an
Evidence entry added or removed; a claim cell changed; a qualifier removed or rewritten (old text not contained in
the new; adding text is free); a changed file under one of its Evidence paths, under a cited verdict's `data`, or
under a cited O19 verdict's session directories (prefix `dirname(data)/<session_id>/`). RE-VALIDATED-BY-TESTS is
named unchecked. Raises into rank ≤ 1 claim nothing and are not gated. (No structured `**Caveat:**` field exists
yet; the caveat half of F2 has nothing to read until one does.)

### Judging a cited record (each Evidence entry of a judged row)
Read from git objects at HEAD (the working tree must match for cited paths). Classified by `record_kind`, never by
name; unknown → NOT-ESTABLISHED. Outcomes: ESTABLISHED / LEGACY (in the snapshot) / EXCEPTED / NOT-ESTABLISHED.
- **Prereg:** the entry's top-level data entry (`classify_all`) FAIL / NON_GATED / NOT_GOVERNED → NOT-ESTABLISHED;
  GRANDFATHERED / UNGOVERNED_RERUN count like LEGACY; any `classify_all` failure fails the gate.
- **sim_report** (a session directory's `report.json`): `record_kind`, provenance with a 40-hex
  `executed_git_hash` and a known `code_tree_sha256`, clean (dirty only when bound by a cited harness row with
  `allow_dirty`, equal `harness_run_id` and tree), `code_changed_during_run` false, `ts`, each role's
  `{role}_profile` / `{role}_router_n_ctx` plus `configured_n_ctx`, `finish_reason` in the allowlist
  (`completed`, `max_turns`, `complete`, `all_encounters_complete`, `campaign_end:*`; pinned against
  `sim_types.SIMULATION_FAILURE_FINISH_REASONS`), `resume_loaded` true when resumed.
- **harness_row file:** only `harness_row` + `harness_header` lines; any `mock` not `false` sinks the file; every
  non-`failed` row established (full hash, clean or `allow_dirty`); a `spawning` row's `sims[]` judged by the sim
  rules on the echoed fields; one `code_tree_sha256` per file.
- **event log** (`harness_event` + `harness_run_end` only): judged per `log_run_id` group — exactly one terminal
  `ok`, one in-group provenance block (`harness_family: in_process`) whose `provenance_digest` equals every line's
  `provenance_sha256`, terminal `end_code_tree_sha256` equal to the block's tree; a failed group is excluded, a
  mock line sinks the file; one tree per file.
- **verdict:** `data` repo-relative, tracked, not a symlink, sha256 = `data_sha256`; scope selects ≥ 1 counted
  unit; one tree per scope; `mock` re-derived false from the bytes; its own provenance clean, known and an
  ancestor of the merge-base (never compared with the scope's tree).
- **Non-support kinds** cited (instrument_check, diagnosis, harness_header, harness_demo): provenance clean or
  allowed, known, executed hash an ancestor of the merge-base, `mock` false; else the row fails. Never support.
- `unknown` / `unknown:…` digests never match anything; every executed hash must be an ancestor of the merge-base.

### O19 verdicts (`exp10_verdict`, `exp09_verdict`)
- The merge-base blob of `scripts/o19_verdict.py` must equal `bound_files["scripts/o19_verdict.py"]` and its
  sha256 `verdict_source_sha256`; the merge-base copy is loaded from a temporary file (its `sys.path` entry
  popped), and its pure `judge()` re-run on the bound bytes with attempts built by its own `attempts_from_rows`,
  ordered by a stable sort of `apparatus.markers` on k; the markers must cover every row's `harness_run_id`; the
  re-judged `verdict`, `deciding_attempt` and `[(run_id, k, complete)]` must equal the verdict's. Any load or
  judge exception → NOT-ESTABLISHED.
- Each scoped `status: ok` row's `files` re-hashed under `dirname(data)/<session_id>/` (plain or `.gz`,
  uncompressed bytes; both forms present → refused); `session_id` one plain path component inside
  `dirname(data)`; file names without `/` or `..`; no symlinks. The `sims[]` check runs over ok rows only
  (`code_changed_during_run` false, end tree == start tree == the row's harness tree).
- T1-1 / T3-9 cite `rerun_exp*_o19/verdict.json` (+ optionally `rows.jsonl`), never the session directories.

### New support
A judged row's raise, move between positives, or date change needs ≥ 1 NEWLY cited (absent from the base
Evidence) record that is a verdict, in the merge-base pass table with this row in `rows`, whose `verdict` is in
`targets[<new token>]`, whose `require` predicates hold (dotted from the verdict's top level; missing key or
non-dict on the way = unmet; type-strict equality), ESTABLISHED, prereg-PASS, clean, and whose time (min `ts` over
its scoped units; never a terminal's or its own) is after the commit time of the base commit that set the row's
previous status line (a new row: the merge-base's). LEGACY / GRANDFATHERED / UNGOVERNED / allowed-dirty never
count; an ACTIVE exception (first clause: this PR performs its exact from→to with HEAD's date == to_date) may.
Removing Evidence: a row with ≥ 1 ESTABLISHED entry at base keeps one.

### Bootstrap
5b-1's own PR moves no row (the pass table is not on main yet). T1-1 / T3-9 move in a later PR, once the table
and O19's verdicts are on main. A PR that tightens an entry and moves a row in one diff is judged by the
merge-base (looser) entry.

### 5b-2 (next)
Pass entries + "complete run" rules for `exp53_verdict` / `exp54_verdict` (re-derive `runs_of` / `Run.status`
from the bound bytes, gate-owned), `exp60_verdict` (every seed of `frozen.seeds` under one `(run_id, arm)`,
refused rows included), `exp61/62_verdict` (one `campaign_id` per data file and scope). Also: `kind: "prereg"` exceptions read by the prereg lint as an `EXCEPTED` status for that data entry (the
successor of its frozen in-script lists).
