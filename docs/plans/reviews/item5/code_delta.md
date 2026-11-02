# #1315 item 5: delta review round (53580dc5..3f5c7096)

Scope: the fold commit `3f5c7096` only, read against the three reports it folds (`code_executor.md`,
`code_architecture.md`, `code_wire.md`) and against notes §1A.3, §1A.8, §4.2 and §5. Fresh reader; nothing in the
worktree was modified except this file. Every probe ran in a scratch copy
(`.../scratchpad/delta1`, `PYTHONPATH` on its own line; a child test confirmed `maxim.advisor.log.__file__` resolved
to the scratch copy, not the worktree).

## What was run

- Narrow tests (scratch copy): `test_advisor_log.py`, `test_advisor_log_red_gates.py`, `test_atomic_io.py`,
  `test_architecture_audit.py`, `tests/integration/test_advisor_gate.py`: **145 passed, 62 xfailed**.
- `lint_no_silent_swallows`, `lint_orphan_modules`, `lint_atomic_io_ratchet` (also in the worktree, read-only, with
  the diff-scoped ratchet: "no file's count rose"): all rc=0. `ruff check` / `ruff format --check` on the touched
  files: clean.
- mypy 1.20.0 on `run_mypy_public.sh`'s list plus `advisor/log.py`, `utils/atomic_io.py`, `utils/audit.py`
  (neither utils file is on the list; checked anyway): no issues in 59 files.
- `compare_to_baseline(audit_architecture(src/maxim), load_baseline())`: `added=() stale=() unreviewed=()`. The new
  `advisor` rule adds no finding against the committed baseline; `maxim --audit-architecture` rc=0.

## Deletion probes (question 1)

Twenty probes, each a one-anchor replacement in the scratch copy, restored after the run. **Every one turns a test
red:**

| Fold removed | Test that fails |
|---|---|
| `_write_synced` `except BaseException` -> `except OSError` (Executor B1) | `test_an_interrupt_mid_append_breaks_the_log_and_it_stays_readable` |
| bisection result halved (Executor SF2) | `test_truncated_arguments_full_of_escapes_still_fit` |
| `session_id` regex disabled (Executor SF3, Wire SF-1) | `test_a_session_id_that_is_not_one_safe_component_is_refused[x/../../escaped]` |
| mode bound removed (Arch NIT4) | `test_a_mode_beyond_0640_is_refused[438]` |
| closed schema -> superset allowed (Executor NIT6) | `test_a_refused_record_writes_nothing_and_the_log_stays_usable[caller_seq]` |
| text-field type check removed (Wire N-1) | `test_the_record_schema_is_closed_and_typed[layer_none]` |
| `effect` type check removed | `test_the_record_schema_is_closed_and_typed[effect_int]` |
| `_format_version` gate removed (Executor NIT5, Arch NIT1) | `test_read_log_refuses_a_header_it_cannot_read[future_version]` |
| header built without `with_format_version` | `test_a_long_reason_is_bounded_with_a_flag` (first to read) |
| `RecursionError` catch removed (Executor NIT8) | `test_read_log_refuses_deep_nesting_as_malformed` |
| `S_ISREG` check removed | `test_read_log_refuses_a_fifo_without_blocking` |
| `O_NONBLOCK` removed | same test, by its 10 s timeout |
| `__exit__` does not close (Executor NIT10) | `test_the_log_is_a_context_manager_that_closes` |
| `F_FULLFSYNC` fallback taken on every errno | `TestStrictFsync::test_unsupported_full_fsync_falls_back_to_fsync_and_others_raise` |
| `F_FULLFSYNC` fallback never taken | same test |
| zero-byte write guard removed (Executor NIT4) | `test_a_write_that_makes_no_progress_raises_instead_of_spinning` (by its timeout, no hang) |
| directory fsync in `create()` removed | `test_create_closes_its_fd_when_the_header_cannot_be_synced` |
| `LAYER_RULES["advisor"]` removed (Arch SF3, Wire N-4) | `test_the_advisor_is_fenced_from_the_agent_memory_and_network_layers` |
| `import maxim.tools.base` added to `log.py` (Arch SF3, Wire SF-3) | `test_the_log_imports_no_runtime_tools_or_agents` |
| `AdvisorLogBroken` off the `AdvisorError` base (Arch NIT2) | `test_malformed_is_a_value_error_and_broken_a_runtime_error` |

## The specific checks (question 2)

- **The strict-helper move vs the red gates' monkeypatches: still reached.** `_SyncRecorder`,
  `_inject_write_failure`, the short-write test and `test_full_fsync_branch` all patch attributes on the global `os`
  and `fcntl` module objects (`monkeypatch.setattr(os, "write", ...)`, `setattr(fcntl, "fcntl", ...)`). `atomic_io`
  calls `os.write`/`os.fsync` through the same module object, and `fsync_strict` does `import fcntl` and reads
  `F_FULLFSYNC` at call time, so every patch lands. The unit tests that patch `atomic_io.fsync_strict` reach
  `append_line_fsync`'s module-global lookup; the one that patches `advisor_log.fsync_strict` reaches only
  `create()`'s directory fsync, which is what it intends.
- **The fallback errno set cannot downgrade a real I/O failure to success.** EIO (and anything outside
  `{EINVAL, ENOTSUP, EOPNOTSUPP, ENOTTY}`) re-raises, which the new test pins, and the fallback `os.fsync` itself
  raises on failure. What it can do is downgrade *durability* (cache flush instead of a full flush) on a volume
  that refuses `F_FULLFSYNC`, logged at DEBUG only. See NIT 1.
- **The closed schema vs §1A.8 and §4.2: too strict in one field.** See SHOULD-FIX 1 (`effect: "unknown"`). The
  key sets themselves match §1A.8's `refusals[]` 1:1 and §5's proposal record; proposal `arguments` are untyped,
  so any JSON value passes.
- **`session_id` regex vs item 1's ids: compatible.** `str(uuid.uuid4())` (36 chars, hex and `-`) and
  `uuid4().hex` both `fullmatch`; I created a log with a uuid4 string. `fullmatch` refuses a trailing newline.
- **`read_log` on a regular file opened `O_NONBLOCK`: harmless.** POSIX gives `O_NONBLOCK` no effect on regular-file
  reads; the round-trip tests all go through it. A FIFO opens immediately and is refused by `fstat`.
- **`LAYER_RULES` vs the committed baseline: clean** (see above). The comment's claim that `runtime.agent_loop`
  cannot be fenced (first-segment matching) is accurate; item 1's `test_import_closure` owns that.
- **Module-level `from maxim import __version__`: fine.** The parent package is fully imported before any
  submodule body runs, and the closure child test passes with it.

## Findings

### SHOULD-FIX 1: `src/maxim/advisor/log.py::_shape` refuses `effect: "unknown"`, the value the shipped executor produces for every unregistered tool

The fold added `if not isinstance(effect, list) or not all(isinstance(flag, str) ...): raise TypeError`. The
pre-fold code passed `effect` through.

- The producer, `runtime/executor.py::_refused_output`, sets `effect = "unknown"` whenever no tool object resolved,
  which is the `NOT_REGISTERED` path (`executor.py`, the `_Refusal(ToolErrorKind.NOT_REGISTERED, "registry", ...)`
  site).
- §1A.8's own example refusal is exactly that case: `{"name": "bash", "error_kind": "not_registered", "layer":
  "registry", "effect": "unknown"}`. The pre-written consumer
  `tests/integration/test_advisor_gate.py::test_report_schema_and_status` asserts `isinstance(effect, (list, str))`,
  "flag names (sorted list) or 'unknown'". The Wire lens's table recorded this vocabulary.
- Reproduced: `append({... "effect": "unknown" ...})` raises `TypeError: a refusal's effect is a list of flag
  names (sorted), not 'unknown'`; the log stays usable.

Failure scenario: item 1 lands; the observer's commonest refusal (the model calls `bash`, which the sealed
registry does not hold) cannot be logged. Either the turn becomes `error` on every such batch, or item 1's author
coerces it (`["unknown"]`, `[]`) so the log's `effect` disagrees with the report's `refusals[].effect`, which is
the log-vs-report split Wire N-1 warned of, with every gate green (the gates check the report's `effect`, never the
log's). It fails loudly today, but it is a fold-introduced contradiction of §1A.8 in shipped code, and the
as-built block repeats it ("`effect` a list of flag names").

Fix: accept `effect == "unknown"` exactly, beside a list of str; add `_refusal(effect="unknown")` as a passing
case and `effect="other"` as a refused one; amend the as-built line to "a list of flag names or `\"unknown\"`".

### SHOULD-FIX 2: `src/maxim/advisor/log.py::AdvisorLog.append`: B1's window is narrowed, not closed

`self._seq = seq` runs in `append`, outside `_write_synced`'s `except BaseException`. On CPython 3.12 the `CALL`
instruction checks the eval breaker after the callee returns, so a SIGINT can raise `KeyboardInterrupt` in
`append` after `append_line_fsync` has written and fsynced the line, but before `_seq` advances. That is B1's state
exactly: the line is on disk with `seq = N+1`, `broken` is False, `last_seq` is N.

Reproduced (scratch copy, Python 3.12.12; a `sys.settrace` hook raises `KeyboardInterrupt` at the `self._seq = seq`
line, which stands in for the signal at that check): `broken = False, last_seq = 1`. The next `append` writes seq 2
again, and after `close()` `read_log` raises `AdvisorLogMalformed: line 4 has seq 2 after 2`. The whole log is
unreadable, the consequence B1 was rated BLOCKER for. The window is now a single eval-breaker check, not the
milliseconds of `F_FULLFSYNC`, so this is SHOULD-FIX, not BLOCKER. But the fold commit and the as-built block
present B1 as closed.

Fix: advance `_seq` inside the guarded frame, e.g. `_write_synced(line)` becomes `_commit(line, seq)`, which does
`append_line_fsync(...)` and then `self._seq = seq` inside the `try`. An interrupt after the store is harmless,
since the state is consistent by then. Pin it with the settrace injection above: assert `broken` is True, or that
`last_seq` advanced, and that `read_log` parses after `close()`.

### NIT 1: `src/maxim/utils/atomic_io.py::fsync_strict` logs the `F_FULLFSYNC` downgrade at DEBUG, per call

Executor NIT 7 asked for a fallback with a WARNING. As folded, an observer whose `report_dir` is on SMB, exFAT or
FUSE silently gets plain-`fsync` durability on every record, and nothing above DEBUG says so. That is documented
policy, not a swallow, but the operator cannot see it. Use a deduped WARNING (once per process, or once per fd),
or record why DEBUG was chosen.

### NIT 2: `src/maxim/advisor/log.py::AdvisorLog.create`: an interrupt between `os.open` and the `try` leaks the fd and leaves a zero-byte file

This is the same eval-breaker point as SHOULD-FIX 2: the `CALL` to `os.open` returns, then the interrupt fires
before the `try:` is entered. `read_log` refuses the empty file loudly, so it fails closed. The structure is
pre-existing. Folding SHOULD-FIX 2 is a good moment to move `os.open` under a guard, for example an outer
`try/except BaseException` that closes the fd when one was assigned.

### NIT 3: `src/maxim/advisor/log.py::AdvisorLog.create` accepts `mode=0`

`0 & ~0o640 == 0`, so the log is created owner-unreadable. The writer is unaffected, since its fd was opened at
creation, but the operator's `read_log` raises `PermissionError`, not `AdvisorLogMalformed`. Consider requiring
`mode & 0o600 == 0o600`.

### NIT 4: findings left unfolded with no recorded reason (question 3)

The two deliberate leaves are fine:
- unbounded file size: `read_log`'s docstring records "read whole: … bounded per record";
- the child's `advisor_missing` branch: Wire N-6 itself says to leave it for item 1, though nothing in the delta
  records that.

These are not folded and have no recorded reason:
- Executor NIT 5, the `os.get_inheritable(fd) is False` pin for `O_CLOEXEC`;
- Executor NIT 7's WARNING (NIT 1 above);
- Wire N-1's value-level vocabulary checks (`error_kind` against `ToolErrorKind` values, `layer` against the
  refusal layers). Only the types are checked; deferring that is defensible for item 1's one-refusal-dict seam,
  but it should say so;
- Wire N-1's note that item 1's report `refusals[].name` reuses `MAX_NAME_CHARS`;
- Wire N-3, the coverage floor owed when `maxim/advisor` crosses 200 statements;
- Wire N-5's "add the other two to §8", which was resolved instead by in-place "Superseded" notes in
  advisor_mode.md. That is acceptable, but §8 still lists only `save_action_log`.

A short "Not folded (owner: item 1 / reason)" list in §5's as-built block would close this.

### NIT 5: docs vs code (question 4)

The as-built block, the persistence-config row and the gotcha, and the advisor_mode.md notes match the code, except
where they inherit SHOULD-FIX 1 ("`effect` a list of flag names") and SHOULD-FIX 2 (the block's B1 line).

Minor items:
- §5's body sentence "No fsync'd append helper exists" is now false. The block's "where it differs" framing covers
  it, but a one-word strike would help.
- `changelog.d/1315-item5-advisor-log.added.md` does not mention the two new public `atomic_io` helpers, the
  `session_id`/mode refusals, or `AdvisorError`.
- "`F_FULLFSYNC` on macOS" is now "where the volume supports it".

## Verdict

The folds fix their findings; every one is proved by deletion. The new mechanisms (closed schema, strict helpers,
fence, closure test) introduce no regression in the gates, the baseline, the lints or mypy. Two items need a fold:
- SHOULD-FIX 1 was introduced by this fold and contradicts §1A.8 and the shipped executor producer;
- SHOULD-FIX 2 is B1's own failure through a narrower window.

Both are a few lines of code plus a test each, and both change `src/`, so a delta-scoped round on that fold is owed.

**NOT STOP**
- SHOULD-FIX 1: `advisor/log.py::_shape` refuses `effect: "unknown"` (§1A.8, `executor.py::_refused_output`).
- SHOULD-FIX 2: `advisor/log.py::AdvisorLog.append` advances `_seq` outside the `BaseException` guard (B1 residual).
