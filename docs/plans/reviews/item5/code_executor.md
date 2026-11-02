# #1315 item 5: Executor lens (commit 53580dc5)

Scope: `src/maxim/advisor/{__init__,log}.py`, the two exceptions in `src/maxim/exceptions.py`,
`tests/unit/test_advisor_log.py`, the flipped gates in `tests/unit/test_advisor_log_red_gates.py`, and the
`_require_advisor` precondition in `tests/integration/test_advisor_gate.py`. Checked against
`atonia_gate_approach_notes.md` §5 (with its BUILT block), §1A.8 and §4.2.

## What was run

- `pytest tests/unit/test_advisor_log.py tests/unit/test_advisor_log_red_gates.py --cov=maxim.advisor --cov-branch`
  on macOS (APFS): 37 passed, 1 xfailed (O7, as intended). Coverage of `log.py` is 99% branch. The one line missed is
  `log.py::_sync`'s `os.fsync` fallback. It cannot run on darwin and it runs on Linux CI. The real `F_FULLFSYNC` call
  on a file fd and on a directory fd ran against APFS in `create()` and passed, so `F_FULLFSYNC` on a directory fd
  works here.
- 27 deletion probes, each run on a scratch copy of `src/` + `tests/`, restored with `cp` and checked with `cmp`. See
  the table below.
- Scenario scripts for the edge cases you named. Each ran against the unmodified module, imported from the scratch
  copy.

**Process note.** I ran the first seven probes (A to G) in place in the shared worktree. That was the same tree another
lens was testing in: pid 83915 was running `pytest ... tests/integration/test_advisor_gate.py --no-cov` there at the
same time. For about 3 minutes, `src/maxim/advisor/log.py` was a mutated copy, and probe G (the zero-byte guard
removed) was in place for the second half of that. Any result the sibling lens got in that window may be contaminated
and should be re-run. I restored the file and checked it with `cmp` against `git show HEAD:`; `git status` is clean. I
ran the rest of the probes in a scratch tree.

## Deletion probes

| Probe | Tests fail? |
|---|---|
| drop `O_EXCL` | yes (1) |
| drop `O_NOFOLLOW` | **no**: redundant, because `O_CREAT|O_EXCL` already refuses a final-component symlink (POSIX) |
| drop `O_CLOEXEC` | **no**: redundant, because `os.open` fds are non-inheritable by default (PEP 446) |
| drop `fchmod` | yes |
| drop the directory fsync | yes |
| drop `broken = True` | yes (3) |
| drop the `written <= 0` guard | yes, but only by **hanging**: the test has no timeout (NIT 4) |
| a single write instead of the loop | yes |
| drop the after-close check, the close `last_seq` check, the torn-tail drop, the `seq` bool check, the schema check, the kind check, the name bound, the missing-fields check, the not-a-dict check, `ensure_ascii`, the caller-stamped check, close-when-broken, create's fd close, the `F_FULLFSYNC` branch, the closed refusal | yes, each one |
| drop `check_format_version` | **no** |
| drop the header `_record_kind == "header"` check | **no**: the schema check alone covers the fixtures |
| drop the lock | **no** (an 8-thread × 50 run with the lock in place gives 400 records with seq rising) |
| `allow_nan=True` in `_encode` only, or in `_bound_arguments` only | **no** for each one alone: the two are redundant for proposal `arguments`, and only `_encode` guards a NaN in a refusal or extra field, which nothing tests |

## Findings

### BLOCKER 1: `log.py::AdvisorLog._write_synced`: a non-`OSError` exception after the write makes the log unreadable once `close()` runs

`_write_synced` sets `broken` only on `except OSError`, and `append` advances `self._seq` only after the write and the
sync. The window that matters is between `os.write` returning and `_sync` finishing; `F_FULLFSYNC` takes
milliseconds. A `KeyboardInterrupt` landing there (the embedder's default SIGINT, since the library installs no
handler, per §1A.8 D), or any other `BaseException`, leaves the line on disk with `seq = N+1` while the log believes
`N` and is not broken. Reproduced:

- interrupt the second append, then `close()`: `read_log` raises
  `AdvisorLogMalformed: the close record says last_seq 1, the records 2`;
- interrupt one append, then append again: `read_log` raises `line 3 has seq 1 after 1`.

So an operator's Ctrl-C, followed by the `finally: session.close()` item 1 will have, destroys the whole log's
readability. That contradicts §5: "a torn line can only be the final one", and only that one is dropped. The same
holds for a Ctrl-C between two `os.write` calls in `_write_all`'s loop: a torn line mid-file, followed by the close
record.

Fix: `except BaseException: self.broken = True; raise` in `_write_synced`, which matches what `create()` already
does. Add a test that injects a `KeyboardInterrupt` from `_sync` and asserts that `broken` is set and that `read_log`
still parses after `close()`.

### SHOULD-FIX 2: `log.py::_bound_arguments`: escape-heavy arguments truncate to `""` or `"{"`

`cut -= over` overshoots: when more than about half the characters are `"` or `\`, `over` exceeds `cut`. `cut` then
goes negative, `encoded[:cut]` becomes a long prefix, `over` grows, and `cut` runs down to `-len`, which yields an
empty prefix. Measured:

| `arguments` | Result |
|---|---|
| `'\\' * 5000` | `""` |
| `'"' * 5000` | `""` |
| `{"s": '"\\' * 5000}` | `"{"` (one character) |
| `"\x01" * 5000` or `"é" * 5000` | 3,410 characters, encoding to 3,982 bytes, not the 4,096 available |

The loop does terminate: `cut` strictly decreases while `over > 0`, and `''` fits. But the proposal's content is lost
for exactly the model-controlled hostile class (Windows paths, quoted shell). `test_truncated_arguments_full_of_escapes_still_fit`
asserts only the bound, so a 1-character result passes it.

Fix: never let `cut` go below 0, and step by `max(1, over // 2)`, or better, build the prefix by summing each
character's escaped length. Then assert that the escaped prefix is at least about 4,090 bytes.

### SHOULD-FIX 3: `log.py::log_name` / `AdvisorLog.create`: `session_id` is not validated as a single path component

`O_NOFOLLOW` guards only the final component. `create(dfd, "x/../../escaped")` with a directory `advisor_log_x`
present under `report_dir` wrote `escaped.jsonl` outside `report_dir`. I verified this. Item 1 will probably mint a
uuid, so this is defence in depth. But `create()` is the module's public surface, and the whole design rests on
fd-anchored writes. Fix: refuse a `session_id` that is empty, or that contains `/`, `\0`, or `..`, or allow-list
`[A-Za-z0-9._-]`.

### NIT 4: `tests/unit/test_advisor_log.py::test_a_zero_byte_write_raises_instead_of_spinning` has no `@pytest.mark.timeout`

There is no global timeout in pyproject. With the guard deleted, the test spins forever: it hung more than 2 minutes
in-tree, and failed only because I passed `--timeout=30`. Add `@pytest.mark.timeout(5)`.

### NIT 5: three mechanisms have no test that fails without them

- **`check_format_version` in `read_log`.** A header with no `_format_version` is accepted as "pre-1.0, assuming
  0.x", but this format has no pre-1.0 history. `"_format_version": "99.0"` reads as OK too.
- **The header `_record_kind` check.**
- **`_encode`'s `allow_nan=False` for refusal and extra fields.** Add a `float("nan")` in a refusal's `effect` or
  `reason`. `reason` is `str()`'d, so use `effect`.

Also consider a test that pins `os.get_inheritable(fd) is False`, so that `O_CLOEXEC` is pinned in spirit.

### NIT 6: `log.py::_shape` passes unknown keys through unbounded, and only `arguments`, `name` and `reason` are bounded

`report_id`, `error_kind`, `layer` and `effect`, and any extra key, are written at any size. Those fields come from
policy today, not from the model. But an allow-list of each kind's keys would stop item 1 from accidentally logging a
model-derived field unbounded.

### NIT 7: `log.py::_sync`: `F_FULLFSYNC` has no fallback when the filesystem refuses it

Some macOS filesystems (SMB, some FUSE and exFAT mounts) answer `F_FULLFSYNC` with `ENOTSUP` or `ENOTTY`, where
`fsync` would work. Today `create()` then fails, so the observer cannot run with a `report_dir` on such a volume.
That fails closed, so it is acceptable, but SQLite's practice is to fall back to `fsync`. Either document it or fall
back, logging a WARNING.

Similarly, `create()` needs `report_dir_fd` to be readable for `os.fsync` on Linux: an `O_PATH` fd gives `EBADF`. §4.2
says `O_DIRECTORY|O_NOFOLLOW`, which implies `O_RDONLY`. State that in the docstring.

### NIT 8: `log.py::read_log` on hostile files

- Deep nesting (`[` × 100k) escapes as `RecursionError`, not `AdvisorLogMalformed`.
- A FIFO at the path blocks forever in `read_bytes`.
- A huge file is read whole into memory.

Invalid UTF-8, duplicate keys, `1e400` and CRLF are all handled correctly. These are reader-side, and the reader is an
operator or test tool, so NIT. Catch `RecursionError` as well, and `fstat` the file to be a regular file.

### NIT 9: the log can be ahead of the report after a failed fsync

When the write succeeds and the fsync raises, the record is on disk with `seq = N`, but `append` raised, so the
turn's report says `log_last_seq = N-1`. The test pins this, but `read_log`'s docstring talks only of a lost tail. Say
that `last_seq > log_last_seq` is possible after an `error` turn.

### NIT 10: no context manager and no fd finalizer on `AdvisorLog`

A caller that forgets `close()` leaks the fd until the process exits. Item 1's `close()` covers the planned path;
`__enter__` and `__exit__` would be cheap.

## The API against item 1's needs (§1A.8, §4.2)

The API fits item 1:

- `create(report_dir_fd, session_id, *, mode)` takes the fd that `open()` already checked, and `log_mode` passes to
  it as `mode=`.
- `append()` returns `seq`, which feeds `proposals[].seq`, `refusals[].seq` and `log_last_seq`.
- `last_seq` is there for the report.
- After a stop, `close()` writes the close record.
- `AdvisorLogBroken` is a `RuntimeError`, so item 1 can map it to `error`.

`ts` is UTC ISO-8601 with milliseconds and a `+00:00` suffix. Neither §5 nor §1A.8 fixes the format, so item 1's
`created_at` should use the same one. The `_require_advisor` tightening (it now needs `hasattr(mod, "open")`) is
correct. Without it, the empty package would turn every O-gate's precondition true and the gates would fail for the
wrong reason.

**Verdict: REQUEST CHANGES.** Fold BLOCKER 1, which is a one-line fix plus a test, along with SHOULD-FIX 2 and 3. The
NITs are optional, though NIT 4 and NIT 5 are cheap test additions.
