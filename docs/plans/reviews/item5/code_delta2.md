# #1315 item 5: delta round 2 (3f5c7096..6150333d)

Scope: the fold commit `6150333d` only, read against delta round 1 (`code_delta.md`: SF1, SF2, NITs 1-5). Fresh
reader. Nothing in the worktree was modified except this file. Every probe ran in a scratch copy
(`.../scratchpad/delta2`, with `PYTHONPATH` on its own line and `__pycache__` removed). `maxim.advisor.log.__file__`
resolved to the copy, and the deletion probes below turned the copy's tests red, which confirms the tests ran the
copy's code.

## What was run

- **Narrow tests** (scratch copy): `test_advisor_log.py`, `test_advisor_log_red_gates.py`, `test_atomic_io.py`,
  `test_architecture_audit.py`, `tests/integration/test_advisor_gate.py`: **151 passed, 62 xfailed**.
- **Ruff:** `ruff check` and `ruff format --check` on `src/maxim/advisor/log.py` and `tests/unit/test_advisor_log.py`
  are clean. The longest added `src/` line is 120 characters, which is the configured limit.
- **Lints** (worktree, read-only):
  - `scripts/changelog_fragments.py check` is clean.
  - `lint_no_silent_swallows` is clean. The `except BaseException` in `_write_synced` sets `broken` and re-raises.

## Deletion probes (question 1)

Each probe is a one-anchor replacement in the scratch copy, restored after the run.

| Probe | Result |
|---|---|
| SF2 reverted: `self._seq = seq` moved back into `append`, after `_write_synced` | RED: `test_an_interrupt_after_the_fsync_before_seq_is_counted_breaks_the_log` ("an interrupt at the seq count left the log usable") |
| SF2 variant: guard keeps a `pass`, count in `append` (the pre-fold shape) | RED, same test |
| SF1 reverted: `effect != UNKNOWN_EFFECT and` removed | RED: `test_an_unregistered_names_unknown_effect_is_logged` |
| SF1 widened: any `str` accepted | RED: `test_the_record_schema_is_closed_and_typed[effect_str]` |
| Mode rule: `or mode & 0o600 != 0o600` removed | RED: `test_a_mode_beyond_0640_is_refused[0]`, `[256]` (0400), `[32]` (0040) |
| Close record passes `self._seq + 1` | GREEN. This is not a defect, but the parameter is unpinned for close (NIT 2) |

### The `sys.settrace` test

**The injection is sound.** A `line` event fires before its line executes. Raising from a trace function propagates
the exception into the traced frame at that line, so the `KeyboardInterrupt` lands exactly where SF2's eval-breaker
interrupt would: after `append_line_fsync` has returned, before the `STORE_ATTR`.

**It targets the right line.** The tracer filters on `co_filename == advisor_log.__file__` and on the stripped
source text `self._seq = seq`, which occurs once in `log.py`. The other assignment is `self._seq = 0`.

**It fails without the fix.** If the count is moved back to `append`, the same text matches there, outside the
guard. The test then fails on `assert log.broken`, not vacuously. The `assert fired` check separately guards
against the injection point disappearing.

**It restores the previous tracer.** `previous = sys.gettrace()` is restored in `finally`. CPython also unsets a
trace function that raises, so nothing of the test's tracer outlives the `with` block either way. I verified the
restore three ways:

- A sentinel outer tracer, installed around a direct call of the test function, is still `sys.gettrace()`
  afterwards.
- Under `COVERAGE_CORE=ctrace`, a later test sees `<coverage.CTracer>`, and coverage keeps measuring the later
  test's lines.
- Under `COVERAGE_CORE=pytrace`, a later test sees `PyTracer._trace`.

Under the default core here (coverage 7.13.3, Python 3.12, sysmon), `gettrace()` is `None` before and after.

**It cannot leak into later tests.** `settrace` is per-thread. The tracer returns `None` for every frame outside
`log.py`, so it adds no cost elsewhere. A test failure inside the `try` still runs the `finally`. The only cost is
that coverage under ctrace or pytrace misses `log.py` lines executed during the injected `append`, which is a few
lines that other tests cover.

## The specific checks (question 2)

- **`_write_synced(line, self._seq)` for the close record is a no-op re-count.** `self._seq = self._seq`. An
  interrupt after the close record's fsync sets `broken`, then `finally` closes. The close record is on disk and
  `read_log` reads `closed: True` with a consistent `last_seq`. Nothing new is wrong. NIT 2 covers the design smell.
- **The mode rule is exactly {0600, 0640}.** Python binds `&` tighter than `!=`, so `mode & 0o600 != 0o600` parses
  as `(mode & 0o600) != 0o600`. Within the 0640 mask, the only values that carry 0600 are 0600 and 0640. Every
  other value I tried is refused with a `ValueError`: 0620, 0610, 0, 0400, 0040, `-1` and `True`. A non-int raises
  `TypeError` before `os.open`.
- **`effect` can be no string other than `"unknown"`.** For a `str`, the guard reduces to `effect == "unknown"`.
  Two values only `==`-equal `"unknown"`:
  - a `str` subclass, which encodes identically;
  - an object with a custom `__eq__`, which reaches `json.dumps` and raises `TypeError` before anything is written,
    so the log stays usable.

  Both fail closed. Sortedness is still not checked (NIT 3).
- **The remaining SF2 window is consistent.** With the count inside the guard, an interrupt after the store (at
  `return seq` in `append`) leaves a counted, consistent log. The caller loses the returned `seq`, but `last_seq`
  carries it. Neither `STORE_ATTR` nor the handler's `self.broken = True` is an eval-breaker check point, so SF2 is
  closed, not narrowed.
- **The "Not folded" list in §5 is honest, with one misattribution and one omission.**
  - **Whole-file read:** honest.
  - **`O_NOFOLLOW` and `O_CLOEXEC`:** honest. POSIX `O_CREAT|O_EXCL` fails with `EEXIST` on a symlink at the final
    component, and `os.open` fds are non-inheritable by PEP 446. Round 1 asked for an `os.get_inheritable` pin, not
    a deletion test, so the reason answers a slightly different question. The pin remains cheap.
  - **DEBUG, not WARNING:** the stated reason is true (the plain fsync still raises). It does not address round 1's
    point, which is that the downgrade of durability is invisible to the operator. It is still a recorded choice,
    which is all NIT 4 asked for.
  - **Vocabulary checks: misattributed.** The list says "would import `runtime/` into the log", but `ToolErrorKind`
    lives in `src/maxim/tools/base.py`. The layer strings are literals in `runtime/executor.py`. The conclusion
    holds, since the closure test bans `tools` as well. A literal copy plus a parity test was also an option the
    reason does not weigh. NIT 1.
  - **The fd leak between `os.open` and the guard:** honest. `read_log` refuses an empty file.
  - **The item 1 notes:** honest. §8 does still list only `save_action_log`.
  - **Omission:** round 1 NIT 5's third sub-item, "`F_FULLFSYNC` on macOS" vs "where the volume supports it", was
    neither folded nor listed (see the changelog bullet below and NIT 4).
- **The changelog fragment** is valid and passes `check`: three `- ` bullets, which the README allows. It now names
  the strict helpers, the `session_id` and mode refusals, and `AdvisorError`. Two wording points (NIT 4):
  - It still says "(`F_FULLFSYNC` on macOS)".
  - "Any failure or interrupt marks the log broken" reads broader than the code. A record refused by `_shape` (a
    `TypeError` or `ValueError` before the write) leaves the log usable. The sentence's context is the write, but a
    reader could take it as any `append` failure.
- **Format and line length:** clean, as above. The `effect` guard line is exactly 120 characters.

## Findings

No BLOCKER. No SHOULD-FIX.

### NIT 1: `docs/plans/atonia_gate_approach_notes.md` §5 "Not folded" names the wrong layer

The bullet says checking `error_kind` against the vocabulary "would import `runtime/` into the log". `ToolErrorKind`
is `src/maxim/tools/base.py::ToolErrorKind`. Failure scenario: item 1's author reads "`runtime/`", concludes that a
`tools`-side vocabulary is importable, and adds `from maxim.tools.base import ToolErrorKind` to the log. That fails
loudly at `test_the_log_imports_no_runtime_tools_or_agents`, so this is fail-closed. Fix: say "`tools/` (and
`runtime/` for the layer strings)".

### NIT 2: `src/maxim/advisor/log.py::AdvisorLog.close` passes `self._seq` as the close record's "seq"

The close record has no `seq`. Passing `self._seq` makes `_write_synced` re-store the same value, and no test pins
it: a probe passing `self._seq + 1` stays green. Failure scenario: a later edit passes a different value, and
`last_seq` read after `close()` drifts by one from the close record and from `read_log`. Item 1 reads `last_seq` for
the report's `log_last_seq`. Fix: `seq: int | None` (`None` means do not count), or assert `last_seq` after
`close()` in a test.

### NIT 3: `src/maxim/advisor/log.py::_shape` does not check that `effect` is sorted, but its message and the as-built block say "sorted"

The TypeError text, "a list of flag names (sorted)", and §5's "a sorted list of flag names" read as enforced. Item
1's producer, `Effect` serialisation, sorts, so nothing breaks today. Failure scenario: a hand-built refusal with
`["WRITE", "READ"]` persists unsorted, and a consumer comparing the log's `effect` to the report's by `==` sees a
mismatch. This is pre-existing, not introduced by this fold. Fix: check `effect == sorted(effect)`, or drop
"sorted" from the message.

### NIT 4: stale docstrings and wording

- `AdvisorLog.create`'s docstring still says "``mode`` within 0640". The rule is now exactly 0600 or 0640.
- `_shape`'s docstring still says "an ``effect`` that is not a list of flag names", without "or `\"unknown\"`".
- `tests/unit/test_advisor_log.py::test_a_mode_beyond_0640_is_refused` now also covers modes inside 0640 (0, 0400,
  0040). The name undersells it.
- The changelog's "(`F_FULLFSYNC` on macOS)" and "any failure or interrupt" (see question 2). This is round 1 NIT
  5's unlisted sub-item.

## Coverage of delta round 1 (question 3)

| Round 1 finding | Disposition |
|---|---|
| SF1 `effect: "unknown"` | Folded, and proved by deletion |
| SF2 seq outside the guard | Folded, and proved by deletion. The window is closed, not narrowed |
| NIT 1 DEBUG fallback | Listed in "Not folded", with a reason |
| NIT 2 fd leak between `os.open` and the guard | Listed, with a reason |
| NIT 3 `mode=0` | Folded (0600 or 0640 exactly), and proved by deletion |
| NIT 4 unrecorded leaves (`O_CLOEXEC` pin, WARNING, vocabulary, `MAX_NAME_CHARS`, the coverage floor, §8, `advisor_missing`) | All listed. The vocabulary reason is misattributed (NIT 1) |
| NIT 5 docs: "No fsync'd append helper exists" | Folded (as-built note) |
| NIT 5 docs: changelog omissions | Folded |
| NIT 5 docs: "`F_FULLFSYNC` on macOS" wording | **Neither folded nor listed** (NIT 4) |

## Verdict

Both SHOULD-FIXes are fixed at the root and pinned by tests that turn red when the fix is deleted. The settrace
injection is sound, restores coverage's tracer, cannot leak, and fails without the fix. The fold introduced no
defect. The four remaining findings are NITs: docs wording, docstrings, and one unpinned no-op parameter. All are
fail-closed or cosmetic, and none changes behaviour.

**STOP**
