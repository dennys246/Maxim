# Step 0b (offline latch) — Executor lens

Diff: `git diff origin/main...HEAD` (63f196bb + merge 3b7f775a). Reviewer ran probes in `python -I` children from a
scratch dir, deletion probes on a scratch copy of the tree (with a scratch `git init` for the provenance gates), and
`lint_coverage.diff_coverage` on the touched test files. Nothing in the worktree was edited except this file.

## Channel probe (engaged child, `engage("probe")`)

| Channel | Result | Ledger |
|---|---|---|
| connect / connect_ex loopback, connect `::1` | OfflineRefusal | +1 each |
| sendto / sendmsg UDP to 127.0.0.1 | OfflineRefusal | +1 each |
| getaddrinfo, gethostbyname(_ex), gethostbyaddr, getnameinfo | OfflineRefusal | +1 each |
| AF_UNIX connect, AF_ROUTE socket, `socket.socket(fileno=fd)` | OfflineRefusal | +1 each |
| Popen, `run(shell=True)`, os.system, os.popen, posix_spawn(p), os.execv | OfflineRefusal | +1 each |
| **AF_UNIX SOCK_DGRAM `sendto(path)`** | **passes** | 0 |
| **`socket.fromfd(fd, AF_INET, ...)`, `socket.socket(AF_INET, SOCK_STREAM, 0, fd)`** | **passes, and sent bytes** | 0 |
| **`os.spawnv(P_WAIT, ...)`** | **returns 127, no exception in the parent** | 0 |
| bind/listen inbound, socketpair | passes (as specified) | 0 |

`OfflineRefusal` is a RuntimeError, not an OSError. Ledger is booked before the raise (deleting `_book` in `_hook`
fails 12 channel tests).

## BLOCKER

**B1. The diff-coverage gate fails: 109/159 = 68.6% on the touched test files (the gate is 80%).**
`src/maxim/_offline.py::_hook` and `::_refused_detail` (lines 185–216) show as uncovered even though in-process tests
engage the latch. The cause is CPython itself: audit hooks run untraced unless they set `__cantrace__`, so no test can
cover the hook body *through* an audit event. I verified this with a `settrace` probe. The other uncovered lines run
only in child processes: `maxim/__init__.py::_self_engage_offline` (lines 28–31), `get_version_info`'s offline return,
`simulation/report.py::_code_stamp`'s offline branch, `utils/code_tree.py::git_out`'s probe, and
`utils/internet_access.py::session_internet_lowered`. Nothing else in the suite engages in-process, so the full suite is
unlikely to close the gap.
Failure: CI's coverage job fails the PR.
Fix: add in-process unit tests:
- call `_offline._refused_detail` and `_offline._hook` directly; direct calls are traced, so there is no need for
  `__cantrace__` on a security hook;
- test `_self_engage_offline` with monkeypatched env;
- test `get_version_info`, `_code_stamp`, `git_out` and `session_internet_lowered` after `engage()`, using the autouse
  reset.

## SHOULD-FIX

**S1. `_offline.py::_SPAWN_EVENTS` claims `os.spawn*`, but on POSIX a fork followed by exec is refused only in the child.**
The `"os.spawn"` audit event exists only on Windows. On POSIX, `os.spawnv` is a Python fork+exec, so:
- the exec refusal is raised in the child, which runs `_exit(127)`;
- the parent sees rc 127, no `OfflineRefusal`, and an empty ledger.

That breaks the docstring's typed-and-booked claim. Any hand-rolled `os.fork(); os.exec*()` without a try is worse, and
so is `pty.spawn`. I probed it: the child raises `OfflineRefusal` and unwinds the CALLER's stack as a duplicate process.
It runs the caller's `except`/`finally` blocks, and may write files and run atexit handlers.

The network still fails closed. Fix: choose one, as an owner decision:
- refuse `os.fork`/`os.forkpty` while engaged, which costs fork-mode multiprocessing;
- or state this residual in the docstring and in notes §6.

Either way, add `os_exec` and `os_spawn` to `REFUSED_CHANNELS` with the honest expectation. Today neither is tested,
and neither are `connect_ex`, `sendmsg`, `gethostbyname` or `getnameinfo`, although all four pass my probe.

**S2. `_offline.py::_refused_detail` misses fd adoption that passes an explicit family.**
The `socket.__new__` audit event carries no `fileno`, so the latch refuses only `socket(fileno=fd)` without a family.
`socket.fromfd(fd, AF_INET, SOCK_STREAM)` is the stdlib's own adoption API, and it passes. So does
`socket(AF_INET, …, fileno)`. Both sent bytes on a connection opened before the latch went up.
This is the same class as the pre-latch residual the notes already state. But the module docstring and the CHANGELOG
say "fd adoption … is refused" without qualification.
Fix: reword to "adoption without a family". Name `fromfd` and an explicit family in the boundary paragraph, and in the
notes' residual list.

**S3. `_offline.py::_refused_detail` lets `sendto`/`sendmsg` to an AF_UNIX datagram path pass.**
A local daemon on an AF_UNIX datagram socket can still be reached. Examples are syslogd, which may forward off-box, or
any local datagram service, and `sendmsg` can also pass fds with SCM_RIGHTS. V8 says "no socket endpoint". The notes'
list names only the INET case, so this is a spec gap that the implementation copied.
Fix: one line. Refuse any `sendto`/`sendmsg` with a non-None destination, whatever the socket's family. A send with a
`None` destination goes over an already-connected socket, and that connect was already refused.

## NIT

**N1. No test guards the env restore in `_offline.py::_reset_for_tests`.**
Deleting the restore loop leaves every test green, because the conftest fixture restores the same five keys.
Nothing breaks today. But the docstring's guarantee is unguarded: one assertion on `os.environ` after `_reset_for_tests()`
would cover it.

**N2. `annotation_disabled_via_env` no longer accepts `t`/`y`.**
The CHANGELOG states this behaviour change, so the Wire-integrity lens should grep scripts, configs and docs for
`MAXIM_DISABLE_*=y|t`.

## Verified OK

- **Inert until engaged:**
  - `import maxim` without the variable loads no `_offline` and adds no hook;
  - `MAXIM_OFFLINE` set to `1`, `yes`, ` On ` or `TRUE` engages; `y`, `t`, `0` and empty do not;
  - `truthy` is exactly 1/true/yes/on, case-insensitive and whitespace-tolerant.
- **`_reset_for_tests`** restores the five keys: a prior `HF_HUB_OFFLINE=0` comes back, and absent keys are popped.
- **`on_engage`** runs at once when the latch is already up.
- **`utils/http.py`:**
  - `_offline.on_engage(close_all)` is registered;
  - the request event hook is on both client constructors, so it covers the registry clients and `_checked_client`,
    which in turn covers `_download_client` and `address_checked_client`;
  - `strict` and `trust_env` are forced only under the latch;
  - offline, the #921 proxy-mount precondition holds;
  - the registry lock is an RLock, so engaging inside `get_client` → `close_all` cannot deadlock.
- **`code_tree`** uses a `sys.modules` probe and imports nothing.
- **`"unknown: offline"`** is refused by `scripts/_evidence_records.py::unknown`.
- **The conftest pop** runs before the first `maxim` import, and no pytest11 plugin imports `maxim` earlier.
- **Deletion probes:** every one of these turned a test red:
  - `on_engage(close_all)`;
  - the request hook;
  - `strict`;
  - `trust_env`;
  - the version-info return;
  - the `_code_stamp` branch;
  - the `git_out` return;
  - internet-lowered;
  - self-engage;
  - `_book`;
  - the lazy hook install;
  - the library flags.

  Gates that need a `.git` raise `ScenarioBroken` when it is missing, so they do not skip and cannot pass vacuously.
