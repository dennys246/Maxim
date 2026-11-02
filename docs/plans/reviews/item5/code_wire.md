# #1315 item 5: wire-integrity review (pre-merge)

Diff: `git diff origin/main...HEAD`, commit 53580dc5 (`feat/1315-item5-advisor-log`). Lens: Wire integrity
(docs/CODE_REVIEW.md). Read: docs/CODE_REVIEW.md, docs/wiring/, notes §5, §1A.8 and §4.2.

Run on the clean tree, after the lead said another reviewer's in-place probe was restored (`cmp` against HEAD matched):
- `tests/unit/test_advisor_log_red_gates.py`, `tests/unit/test_advisor_log.py` and
  `tests/integration/test_advisor_gate.py`: 40 passed, 62 xfailed. Every item-1 gate still xfails for an item-1 reason.
- orphan, no-silent-swallows, function-length and mypy ratchets, plus ruff: clean.
- `scripts/prepr.sh --allow-dirty`: 30/30 passed. It ran while the probe was in place, so it was TAINTED; the lints that
  matter were re-run clean afterwards.
- Census and scanner tests: `test_report_redaction_997`, `test_tool_effect`, `test_tool_policy_census`,
  `test_profile_readers_v9`, `test_download_consent_1343`, `test_record_fields_995`, `test_architecture_audit`,
  `test_lint_orphan_modules`, `test_persistence_compat` and `test_optional_deps`: 270 passed, 1 known xfail (#1479).

Context: V18 ships item 5 before item 1, so the module has **zero production callers** by owner decision. Every
consumer below is test-only or pre-written for item 1, and is marked as such.

## Producer → consumer table

| Contract | Producers (before → after) | Consumers (before → after) | What each consumer gets, before → after | Probe |
|---|---|---|---|---|
| Call: `AdvisorLog.create(report_dir_fd, session_id, *, mode=0o600)` | none → `advisor/log.py::AdvisorLog.create` | **test-only:** red gates (`test_create_*`, 7 sites), `test_advisor_log.py`. **Item 1 (planned):** `open(log_mode=0o600)` → `create(mode=)` (§5, §1A.8; `test_builder_refusals_signature` pins `log_mode`) | `find_spec` AssertionError → a working create. The signature matches the red gates' seam line and §5 exactly | 1, 3 (match), 10 (see SF-2) |
| Call: `append(record) -> int` (log stamps `seq`, `ts`) | none → `AdvisorLog.append` | **test-only:** red gates and unit tests. **Item 1:** "the turn" writes each proposal and refusal (§4.2). The report's `proposals[].seq`, `refusals[].seq` and `log_last_seq` come from the return value and `last_seq` | int seq, monotonic from 1. `test_report_schema_and_status` joins report seq ∈ log seqs, which matches | 1, 7 |
| Call: `close()` | none → `AdvisorLog.close` | **Item 1:** `session.close()` writes the close record, also after a stop (§1A.8; `test_session_stop_abandons_at_the_next_token` asserts `read_log(...)["closed"] is True`) | Idempotent. A broken log writes no close record (`closed: False`), which is the right signal | 10 |
| Payload: proposal record `{_record_kind, report_id, arguments}` + `seq`, `ts`, `arguments_truncated?` | none → `_shape` + `append` | O7 gate (`records[*].report_id == rep["report_id"]`, `_record_kind == "proposal"`); `test_report_schema_and_status` (seq join); AvServ (via the owner's hand-off) | Field names match §5 and §1A.8 (`seq`, `report_id`, `ts`). Over-bound `arguments` become a **string** prefix, so a consumer must not assume a dict (documented in the §5 status block) | 7 |
| Payload: refusal record `{_record_kind, report_id, name, error_kind, layer, effect, reason}` + `seq`, `ts`, `name_truncated?`, `reason_truncated?` | none → `_shape` + `append` | O7 gate (`name == "bash"`); item 1's report `refusals[]` (the same fields, §1A.8) | Key names match §1A.8's `refusals[]` example 1:1. **Values are not validated** (N-1) | 4, 9 |
| Vocabulary: `error_kind` = `ToolErrorKind.value` | `tools/base.py::ToolErrorKind`, `GATE_REFUSAL_KINDS` (item 3, shipped) | log records (passed through, never checked); gate `GATE_KIND_VALUES` (test) | The 7 gate kinds match `GATE_KIND_VALUES` exactly. `ToolErrorKind` is a plain `Enum`, so passing the member instead of `.value` raises `TypeError` before the write, which is loud. `str(kind)` would persist `"ToolErrorKind.POLICY_PIN"` silently (N-1) | 4 |
| Vocabulary: `layer` = `metadata["refusal_layer"]` | `runtime/executor.py::_Refusal` sites: `forbid`, `mode`, `permissions`, `ceiling`, `pin`, `registry` (NOT_REGISTERED and INACTIVE_SCENE), `offline` | log records (passed through); gate `REFUSAL_LAYERS` (test) | The shipped producer set equals `REFUSAL_LAYERS`. A `None` from `metadata.get(...)` would persist silently (N-1) | 4, 8 |
| Vocabulary: `effect` = sorted flag names or `"unknown"` | `executor.py::_refused_output` (`effect_names(...)` or `"unknown"`); `tools/base.py::effect_names` | log (passed through); gate `isinstance(effect, (list, str))` | `Effect` is a plain `Flag`, not an `IntFlag`, so a raw member raises `TypeError` in `_encode` and the integer can never be persisted silently. Matches §1A.8's "never the Flag integer" | 4 |
| Persisted / wire: `advisor_log_<session_id>.jsonl`, `advisor_log/1`, header `_format_version` | none → `create` (header), `append`, `close` | `read_log` (in-diff); `test_no_effect` (filters `advisor_log_*.jsonl`); `_log_path` in both gate files (hard-coded spelling, independent of `log_name`) | The names agree. `read_log` calls `check_format_version(header, "advisor_log")`, which warns only on a missing version and does not refuse "2.0"; `schema` is the real check. No other loader exists | 5, 8 |
| Return shape: `read_log` → `{header, records, closed, last_seq}` | none → `read_log` | O7 gate (`records`, `last_seq`, `closed`); `test_report_schema_and_status`, `test_report_status_rows[*]` (`last_seq == log_last_seq`), stop gate (`closed`) | The keys match every pre-written consumer. `records` excludes header and close, as the gate docstring pins | 7 |
| Identity: `session_id` → file name | none → `log_name(session_id)` (caller-supplied, unvalidated) | item 1 `open()` (planned uuid); `_log_path(report_dir, report["session_id"])` in the gates | Joins as long as item 1 passes the same id it reports. Not validated (SF-1) | 8 |
| Exceptions: `AdvisorLogBroken(MaximError, RuntimeError)`, `AdvisorLogMalformed(MaximError, ValueError)` + `__all__` | none → `exceptions.py` | red gates (catch `OSError`/`RuntimeError`, `ValueError`); `test_advisor_log.py`; item 1's turn → report `error` | The gates' "subclass of either" contract is met. No `__all__` or roster test on `maxim.exceptions` exists; `maxim/__init__` lazily re-exports only category-level errors, which is unaffected | 3 |
| Gate (test precondition): `_require_advisor()` | `find_spec("maxim.advisor")` → plus `hasattr(mod, "open")` | every item-1 gate, directly or via `_session_mod`/`_child_session`; the child's `advisor_missing` branch | Before: AssertionError "no maxim.advisor". After: the package exists, so the new `hasattr(open)` keeps an AssertionError. All 62 still xfail strict. The child's `find_spec(...) is None` branch is now unreachable, but `_child_session` calls `_require_advisor()` first, so nothing reaches the child before item 1 | 6 |
| Gate: O7 `_log_mod()` in the red-gates file | `find_spec("maxim.advisor")` passes now | `test_o7_*` | Still xfails, on `find_spec("maxim.advisor._session")`. The message "item 1 not built: no maxim.advisor" on line 1 of `_log_mod` is stale, but that line can no longer fire | 6 |
| Package `maxim.advisor` (enumerators) | none → `src/maxim/advisor/__init__.py` (empty) | wheel `packages.find include=["maxim*"]` (shipped); orphan lint; coverage `source=src/maxim`; `utils/audit.py::LAYER_RULES`; function-length and mypy ratchets | Ships in the wheel. Coverage: about 140 statements, under `min_package_statements` 200, so no floor is owed yet; item 1 will cross 200 and the lint then fails closed until a floor is added (N-3). The architecture audit does not police `advisor` (N-4). The orphan lint passes only through test references (N-2) | 1 |
| Import closure of `maxim.advisor.log` | — | §5: "imports nothing from runtime/ or tools/ … the import-closure test checks that module too" | Measured in a fresh `-I` child: 6 modules (`maxim`, `maxim.advisor`, `.log`, `maxim.exceptions`, `maxim.utils`, `maxim.utils.format_version`). True today, but **no test pins it** in this diff (SF-3) | claims |
| Claims | `changelog.d/1315-item5-advisor-log.added.md`; notes §5 status block; `advisor/__init__.py` docstring | `docs/plans/advisor_mode.md` lines ~150–161 and ~252; `docs/agents/persistence-config.md` §1 table; O7 row | The changelog is accurate ("Item 1's open() will be its caller"). advisor_mode.md says "AdvisorLog with strict `atomic_io` writers" and "the writer mirrors save_action_log"; both contradict §5 (N-5). The persistence brief has no row for this sanctioned non-`atomic_io` writer (SF-4) | claims |

## Findings

**SF-1 (SHOULD-FIX): `advisor/log.py::log_name` / `AdvisorLog.create` take an unvalidated `session_id` as a path
component.**
- `O_NOFOLLOW` covers only the final component, and `dir_fd` resolves relative names, so a `session_id` such as
  `"../x"` or `"a/b"` creates the file outside, or under a subdirectory of, `report_dir`. Intermediate symlinks are
  followed.
- Today the only producer is the tests, and item 1 is planned to mint a uuid4.
- But the identity contract is "any str". `test_no_effect` asserts the final state of `report_dir`, and the
  docstrings claim the log lives "under `report_dir_fd`"; nothing in the call contract enforces either.
- Failure scenario: item 1, or a future embedder path, lets the caller name the session (AvServ correlating its feed),
  and an id containing `/` writes the log outside the sandboxed directory with no refusal.
- Fix: reject a `session_id` containing `os.sep`, `os.altsep` or NUL, or equal to `.`/`..`/empty (or require a
  `[A-Za-z0-9_-]+` match) in `create`, with a test.

**SF-2 (SHOULD-FIX, forward wire note for item 1; record it in the notes now): the lifecycle around `create` has no
owner on `open()`'s failure path.**
- `AdvisorLog` has no context manager and no finalizer. The fd is closed only by `close()`, and by `create`'s own
  header-sync failure.
- Per §4.2 "the backend failed to load at `open()` → `open()` raises; there is no report".
- Failure scenario: item 1 creates the log before loading the backend. A load failure then leaks the fd and leaves a
  header-only `advisor_log_*.jsonl` in `report_dir`, with no report, which is the state `test_no_effect` forbids for an
  ok run.
- Item 1 must either create the log after every `open()` refusal and the backend load, or close it on every exception
  path.
- Fix now: add one line to §5's status block, so item 1 inherits the ordering rule. Optionally add `__enter__`/`__exit__`.

**SF-3 (SHOULD-FIX): §5's structural claim "imports nothing from runtime/ or tools/ … the import-closure test checks
that module too" has no guard in this diff.**
- The only closure gate, `test_import_closure`, is item-1 xfail and checks `maxim.advisor` after `open()`. It does not
  check `maxim.advisor.log` alone.
- The claim is true today (6 modules, measured), but a future `from maxim.tools.base import effect_names` in `log.py`,
  for example to normalise `effect`, would pass every check in this PR.
- Fix: one test in `test_advisor_log.py` runs `import maxim.advisor.log` in a `-I` child and asserts that no
  `maxim.runtime*`, `maxim.tools*` or `maxim.agents*` module is in `sys.modules`. Prove it by deletion: add a
  `maxim.tools` import and watch it go red.

**SF-4 (SHOULD-FIX, claims): `docs/agents/persistence-config.md` §1 ("Which file may I write from code?") has no row
for the advisor log.**
- CLAUDE.md's cross-cutting invariant says persistence uses `atomic_io`, and this module deliberately does not (the
  `atomic_io` helpers swallow fsync errors, per §5).
- `console/health.jsonl` has exactly such a row.
- Failure scenario: a later "consistency" cleanup routes `append` through `atomic_io`, and a failed fsync then stops
  failing the turn, with no doc to stop it.
- Fix: add a row, "`<report_dir>/advisor_log_<session_id>.jsonl` → YES, observer only → `advisor/log.py::AdvisorLog`
  (`O_APPEND`, fsync per record, not `atomic_io` on purpose)", with `test_advisor_log_red_gates.py` as its guard.

**N-1 (NIT): `advisor/log.py::_shape` checks that the refusal keys are present, not what their values are.**
- `error_kind`, `layer` and `effect` pass through unchecked.
- Failure scenario: item 1 builds the record with `str(output.error_kind)` or `output.metadata.get("refusal_layer")`
  and persists `"ToolErrorKind.POLICY_PIN"` or `null`. The gates check only the REPORT's `refusals[]` against
  `GATE_KIND_VALUES`/`REFUSAL_LAYERS`, never the log's records, so the log and the report could disagree with every
  gate green.
- Fix (probe 9, one seam): have item 1 build ONE refusal dict and use it for both the report and the log. Or validate
  `error_kind ∈ {k.value for k in ToolErrorKind}`, `layer` as a non-empty str and `effect` as a list[str] or
  `"unknown"` here, as an Architecture call.
- Related: the log bounds `name` to 128 characters (`name_truncated`). §1A.8's report `refusals[].name` has no stated
  bound, so the same refusal can carry two different names. Item 1 should reuse `MAX_NAME_CHARS`.

**N-2 (NIT, guard blind spot): `scripts/lint_orphan_modules.py::_candidate_modules` keys modules by stem.**
- The stem `log` is satisfied by any `log` token or import part anywhere (an import of `maxim.advisor.log` in tests,
  among others), so the lint could never flag this module.
- A second `log.py` would overwrite the first in the dict.
- Not this diff's defect, and V18 accepts zero production callers. It does mean the orphan lint is no evidence that
  the module is wired: item 1's caller-grep (the CLAUDE.md "ships with a caller" rule) is the real check.

**N-3 (NIT): coverage.** `maxim/advisor` is about 140 statements, so `scripts/coverage_floors.json` owes no floor now.
Item 1 will cross `min_package_statements` 200, and `lint_coverage.py` then fails closed (`floor missing`) until a
`maxim/advisor` floor is pinned from the CI measurement. Flag it in item 1's plan so it is not a surprise.

**N-4 (NIT): `src/maxim/utils/audit.py::LAYER_RULES` has no `advisor` entry.** The §1A.8 closure ("never imports
`maxim.agents` directly"; no loop, worker or autonomy modules) is enforced only by the item-1 child-process gates, not
by the architecture audit. Adding `advisor: must_not_import [agents, planning, environment]` would make it static.
That is Architecture's call.

**N-5 (NIT, claims): `docs/plans/advisor_mode.md` says the opposite of what was built.**
- About line 159 says "`AdvisorLog` with strict `atomic_io` writers", and about line 151 says "the writer mirrors
  `simulation/report.py::save_action_log`". Both contradict §5 and the built module.
- About line 252 says swallowing-seam ceiling refusals are "written to item 5's log". §5 says non-observer seams use
  structured WARNINGs in this release.
- Notes §8 tracks only `save_action_log` for item 5. Add the other two to §8, or strike them in advisor_mode.md.
- The O7 row correctly stays PARTIAL: proposals are still in memory only until item 1 calls `append`.

**N-6 (NIT): `tests/unit/test_advisor_log_red_gates.py::_log_mod`'s first assertion message ("item 1 not built: no
maxim.advisor") can no longer fire. The child's `advisor_missing` branch in `test_advisor_gate.py` is likewise dead
until it is removed with item 1.** Harmless. Leave both for item 1's PR, which already owns those files.

## What matched (no finding)

- The field names across §5, §1A.8, the red-gate seams and the code agree: `seq`, `ts`, `report_id`, `log_last_seq`
  (via `last_seq`), `layer`, `error_kind`, `effect`, `arguments_truncated`, `name_truncated`, `closed`, `last_seq`.
- Every item-3 `refusal_layer` value and gate `ToolErrorKind` value equals the gates' pinned sets.
- `Effect` (a plain `Flag`) and `ToolErrorKind` (a plain `Enum`) cannot be persisted silently as an int or a member:
  `_encode` raises `TypeError` before writing.
- The `_require_advisor` change keeps all item-1 gates red for an item-1 reason (62 xfailed, 0 XPASS).
- The package ships in the wheel. No package-data entry is needed, since the package has no data files.

**Verdict: SHIP WITH FOLDS. There is no BLOCKER. Fold SF-1 and SF-3 (code and test, cheap) and SF-4 (one brief row),
and record SF-2 in §5 for item 1. The NITs are optional or belong to item 1.**
