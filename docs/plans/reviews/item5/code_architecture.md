# #1315 item 5: Architecture lens, code review

Diff reviewed: `git diff origin/main...HEAD` = commit 53580dc5 (`feat/1315-item5-advisor-log`).
Charter: `docs/CODE_REVIEW.md` (Architecture: right layer and owner, true to the invariants, honest docs).
Read against: CLAUDE.md invariants, `docs/agents/persistence-config.md` §1/§3/§4, `docs/agents/runtime-tools.md`,
`docs/plans/atonia_gate_approach_notes.md` §1A.3, §1A.8, §4.2, §5 (with its "Status: BUILT" block).

What I ran:
- `tests/unit/test_advisor_log.py` + `tests/unit/test_advisor_log_red_gates.py`: 37 passed, 1 xfailed (O7). I
  re-ran this after the coordinator said another reviewer had probed log.py in place. The tree was clean against
  HEAD at that point.
- `lint_no_silent_swallows`, `lint_atomic_io_ratchet`, `lint_orphan_modules` and `maxim --audit-architecture`:
  all rc=0.
- The import closure of `import maxim.advisor.log` in a fresh `-I -B` process: exactly `maxim`, `maxim.advisor`,
  `maxim.advisor.log`, `maxim.exceptions`, `maxim.utils`, `maxim.utils.format_version`. No `runtime`, `tools`,
  `httpx` or `numpy`.
- A deletion probe in a scratch copy (`git archive HEAD`, not the worktree): I removed the new `hasattr(mod, "open")`
  line from `_require_advisor` and ran the item-1 gates. Result below, under the red-gate question.

## Findings

### SHOULD-FIX 1: The strict-fsync primitive lives in the wrong layer, and the as-built block does not record the move

Symbols: `src/maxim/advisor/log.py::_sync` / `::_write_all`, against §1A.3's table.

§1A.3's table is the plan's layer rule, and it says:
- `AdvisorLog`/`read_log` live in `advisor/log.py`;
- "The writer helpers it uses live in `utils/atomic_io.py`" as `append_line_fsync(fd, line)`, beside
  `atomic_write_json_at(dir_fd, name, payload)`;
- both helpers are fsync-strict;
- "The module docstring names its two fsync policies (lenient: the existing helpers; strict: these) and who uses
  each. persistence-config.md §4 gets the line."

§5's sentence "the new one must not borrow theirs" means a new strict helper, not an exemption from `atomic_io`.

As built, the strict policy (`F_FULLFSYNC` where present, else `os.fsync`; an `OSError` propagates) is private to
`advisor/log.py`. `utils/atomic_io.py` is untouched, and the "as built" block does not list this deviation, although
it lists smaller ones.

Failure scenario: item 1 has to write the report through `atomic_write_json_at`, which is strict and fd-anchored.
- It will either re-implement `_sync` a second time, or import a private `_sync` from `advisor/log.py` into
  `utils/` (an upward import).
- One of the two copies will then drift. For example, one gains `F_FULLFSYNC` on the directory fd and the other
  does not, or one starts tolerating `EINVAL` the way `atomic_io._fsync_path` does.
- After that, the log and the report from one turn have different durability guarantees, and the report's
  `log_last_seq` can point past a tail that was never made durable.

This is the "two homes for one policy" shape the front-gate principle exists to stop.

Fix, either one:
- (preferred, matches §1A.3) Move `_sync`/`_write_all` into `utils/atomic_io.py` as `append_line_fsync(fd, line)`
  (plus a public strict `fsync_strict(fd)`), add the two-policy paragraph to the module docstring, and have
  `log.py` call them.
- Record the deviation in the "as built" block, with item 1 named as the owner of the move, and file the follow-up
  in the same commit.

### SHOULD-FIX 2: The persistence brief does not list the new write surface

Location: `docs/agents/persistence-config.md` §1, the "Which file may I write from code?" table, and §4.

The table already carries a row for a comparable JSONL log (`console/health.jsonl` → `HealthLog`: fsync and lock
policy, `_format_version` per record, file type). The advisor log is a new runtime write with a different policy:
- the caller's `report_dir`, not the data home;
- fd-anchored and `O_EXCL|O_NOFOLLOW`;
- strict fsync;
- `_format_version` on the header only, file type `advisor_log`;
- no `atomic_io`.

§1A.3 also promised the §4 line. Without either, the CLAUDE.md invariant "Persistence uses `atomic_write_json`" has
an unrecorded exception.

Failure scenario: a later reader sees an `os.open`/`os.write` persistence path outside `atomic_io` and "fixes" it
onto `atomic_write_text`, which swallows `EINVAL`/`ENOTSUP` fsync errors (see `atomic_io._fsync_path`). That
silently breaks the "a failed fsync fails the turn" contract. No lint catches it: `lint_atomic_io_ratchet` counts
renames only.

Fix: add one table row and one §4 line naming `advisor/log.py::AdvisorLog` and why it is strict. Together with
SHOULD-FIX 1, that is the recorded rationale the invariant needs. Today the rationale exists only in the module
docstring and in §5.

### SHOULD-FIX 3: The "imports nothing from `runtime/` or `tools/`" claim has no guard until item 1

Symbols: `src/maxim/advisor/log.py` (module docstring) and `src/maxim/advisor/__init__.py` ("imports nothing at
import time"), against `src/maxim/utils/audit.py::LAYER_RULES`.

§5 calls "nothing reads the log back into a dispatch" structural. It relies on `read_log` sitting in a module that
imports nothing from `runtime/`/`tools/`, and says "the import-closure test checks that module too". That test
(`test_advisor_gate.py::test_import_closure`) is strict-xfail until item 1, and `LAYER_RULES` has no `advisor`
entry yet (§1A.3 assigns it). The claim is true today (I verified the closure above), but nothing enforces it in the
window V18 opens.

Failure scenario: a convenience edit before item 1 lands, such as `from maxim.tools.base import Effect` to type the
refusal's `effect`, passes CI. Item 1's closure gate then flips on a module that is already dirty, or someone
re-points the gate.

Fix: add the `advisor` entry to `LAYER_RULES` now. It is a static fence, cheap, already designed in §1A.3, and lists
`agents`, `memory`, `default_network`, `comms`, `hivemind`, `peer`, `mesh`, `tunnel`; add `tools` and `runtime` for
`log.py`, or a module-level rule. Alternatively, add a three-line fresh-process closure assertion for
`maxim.advisor.log` to `test_advisor_log.py`. If neither is done, the as-built block should say the claim is
unenforced until item 1.

### NIT 1: The header bypasses `with_format_version`, and the version constant is a second copy

Symbols: `src/maxim/advisor/log.py::FORMAT_VERSION`, `::AdvisorLog.create`, `::read_log`.

- The writer contract is "Writers wrap via `with_format_version(payload)`". The header hand-writes
  `"_format_version": FORMAT_VERSION`, with a module-local `"1.0"` that duplicates
  `utils/format_version.FORMAT_VERSION`.
- `read_log` calls `check_format_version` and discards the return value. A header with no `_format_version` is
  accepted after one deduped WARNING, and a `"2.0"` header is read as 1.0.
- The real version gate is `schema == "advisor_log/1"`. That is defensible, and it is the dual convention §3
  describes, but no docstring says which of the two fields gates.

Scenario: a future shape bump that moves `_format_version` but not `schema` is read silently as the old shape.

Fix: build the header with `with_format_version({...})`, drop the local constant, and state in `read_log`'s
docstring that `schema` is the gate and `_format_version` is the repo-wide stamp.

### NIT 2: The log errors share no base with item 1's

Symbols: `src/maxim/exceptions.py::AdvisorLogBroken`, `::AdvisorLogMalformed`.

The placement is right. `maxim/exceptions.py` holds the `MaximError, RuntimeError` refusals (`LeaderRefusal`,
`StopStillPending`, and `ToolEffectRefused` per §2's as-built block). The mixins are right too: an append to a
broken log is a `RuntimeError`, and a malformed file is a `ValueError` (the red gates pin both).

Item 1 will add `AdvisorCompositionError`, `AdvisorInputError` and `AdvisorStopped`, exposed on `maxim.advisor`
(`test_advisor_gate.py` reads `advisor.AdvisorCompositionError`). With no common `AdvisorError(MaximError)` base, an
embedder who wants "any advisor failure" must list five types. Adding a base later is non-breaking, so this is a NIT.
Item 1 should either define them beside these or re-export them, but should not create a second home such as
`advisor/errors.py` for half of them.

Also note for item 1: the first failure surfaces as a raw `OSError`, and later appends raise `AdvisorLogBroken`. The
red gate pins this. The turn must map both to `error`. The class docstring says so; item 1's code must catch both.

### NIT 3: `create()`'s function-local `from maxim import __version__` buys nothing

Symbol: `src/maxim/advisor/log.py::AdvisorLog.create`.

`maxim` is already in `sys.modules` as the parent package by the time this module runs, so the local import neither
narrows the closure nor breaks a cycle. Hoist it, or comment why it is local. The §1A.8 rule ("`maxim.__version__`,
never `get_version_info()`") is honoured.

### NIT 4: `mode` is applied exactly, with no upper bound

Symbol: `src/maxim/advisor/log.py::AdvisorLog.create`.

`fchmod(fd, mode)` deliberately overrides the umask, and the as-built block records that honestly. As a result,
`mode=0o666` makes a model-text log world-writable with no refusal at this layer. §4.2 puts the input checks in
`open()`, so this is item 1's to bound (for example, refuse any `o+w`/`g+w` bit). It belongs on item 1's list, not
here.

### NIT 5: A stale message and an unrecorded test change

- `tests/unit/test_advisor_log_red_gates.py::_log_mod` still says "item 1 not built: no maxim.advisor" for the
  package check. The package is now item 5's, so the message would mislead if it ever fired.
- The `_require_advisor` change (below) is not mentioned in the as-built block. Given how sensitive CLAUDE.md is
  about red gates, one line there would help.

## The questions asked

**Right layer and owner?**
- `maxim/advisor/log.py` is the owner §1A.3 names.
- `maxim/exceptions.py` is the right home for the two errors.
- The one layer miss is the strict-fsync primitive (SHOULD-FIX 1).

**Are the as-built deviations justified and honest?** Mostly, yes.
- Over-bound `arguments` becoming a string prefix of their own encoding: §5 says only "bounded to 4 KiB with a
  truncation flag". A string prefix keeps the record valid JSON under the bound, is never re-parsed, and is flagged
  by `arguments_truncated`. The type change (dict to str) is the price, and it is stated. A consumer must branch on
  the flag; item 1's report builder should not assume a dict. The loop terminates, and the escape-heavy case is
  tested (`test_truncated_arguments_full_of_escapes_still_fit`).
- The `reason` bound (1,024) fills a value §5 left open.
- `fchmod`, `read_log`'s extra refusals and `check_format_version` are all honest additions.
- The missing entry is the `atomic_io` relocation (SHOULD-FIX 1).

**Does it honour the invariants?**
- No silent swallows: `except BaseException: os.close(fd); raise` re-raises.
- No optional-dependency `try`/`except`: `fcntl` is stdlib on POSIX and imported per call by design, and the
  repo's other modules already assume POSIX.
- No `hash()`.
- No env reads.
- `_format_version` is present (NIT 1).
- The `atomic_io` exemption needs a recorded rationale in the persistence brief (SHOULD-FIX 2). No lint expects
  one: the ratchet counts renames and this module has none.

**Exception placement and hierarchy:** right (NIT 2 for item 1).

**Does the empty package harm item 1's gates?** No, by construction.
- The package imports nothing.
- `lint_orphan_modules` passes, because it judges references permissively and tests count.
- `lint_coverage`: `advisor` has fewer than 200 statements, so it needs no floor today. Item 1 will push it over N
  and owes a NEW floor then.
- `--audit-architecture` passes.
- The import-closure and env-read-set gates run `advisor.open()` in the child, and both stay xfail on the
  in-process `_require_advisor` before the child spawns.
- The child script's `find_spec("maxim.advisor") is None` branch (`advisor_missing`) is now effectively
  unreachable. That is harmless: a missing `open` surfaces first in the parent, as an `AssertionError`.

**Are the docs and the changelog fragment truthful?** Yes.
- Every claim in `changelog.d/1315-item5-advisor-log.added.md` matches the code: fd-anchored, fsync before return,
  `F_FULLFSYNC` on macOS, write or fsync error → broken, torn final line only.
- It names item 1 as the future caller, which is the honest statement of a capability without a caller under V18.
- The as-built claim "all of item 5's gates are flipped except O7" checks out: 37 passed, 1 xfailed.

**Is the `_require_advisor` change a precondition fix or a re-pointed gate?** It is a legitimate precondition fix.
- The precondition encoded "item 1 is built" as "the package exists", and item 5 made that proxy false.
- My scratch-copy probe removed the new line. `test_builder_refusals_signature` then FAILED with
  `AttributeError: module 'maxim.advisor' has no attribute 'open'`. That is not an `AssertionError`, so the strict
  `raises=AssertionError` xfail reports red.
- So without the change, item 5 would turn the suite red for the wrong reason. With it, every item-1 gate stays red
  for the same reason as before (item 1 not built).
- No gate body, no assertion and no flip condition changed. A real item 1 must define `open`, so the extra
  precondition cannot make a gate pass that would otherwise fail.
- The removed xfail decorators in `test_advisor_log_red_gates.py` are flips, not removals: each body now passes, and
  O7, which needs item 1, keeps its decorator.

Verdict: SOUND WITH SHOULD-FIXES. There are no blockers. Fold SHOULD-FIX 1–3 (or record 1 and 3 as deviations owned
by item 1, with a follow-up issue) before the PR opens.
