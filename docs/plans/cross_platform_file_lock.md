# Cross-platform file lock — unify ``process_lock`` and third-party ``filelock``

**Status:** Shell plan — tracking only, no active work. Filed during Plan 4 C2 pre-merge review (2026-04-14, finding I1).

**Scope:** TBD. Estimated 100-200 LOC consolidation + broader test suite.

**Target version:** Not on any current release gate.

**Gates:** None. This is tech-debt cleanup, not a feature.

**Depends on:** Plan 4 C3 (admin API) may add more in-house locking users; wait for C3 to land before unifying.

## The problem

Plan 4 Stage C2 shipped drain state with `filelock.FileLock` (third-party package, added as core dep). The Maxim codebase already had an in-house minimal lock at `maxim/utils/filelock.py` (renamed in C2 to `maxim/utils/process_lock.py` to avoid name collision with the third-party import). Both provide process-level advisory file locking; both support POSIX + Windows; both are used by real code paths today:

- **`maxim.utils.process_lock.file_lock` + `LockContended`** — context manager style, raises `LockContended` on contention. Used by `models/download.py::ensure_available` to serialize concurrent model downloads.
- **`filelock.FileLock` + `Timeout`** — class-based, `with FileLock(path, timeout=N):` style, raises `filelock.Timeout` on contention. Used by `peer/drain_state.py` for drain RMW serialization.

The APIs are different enough that callers can't trivially swap one for the other. The third-party library is more mature (cross-platform Windows handling, timeout support, recursive locks, multiple lock types). The in-house wrapper is simpler and has zero external deps but less battle-tested on Windows.

## Why it's tech debt

1. **Two locking abstractions in one codebase is a footgun.** Future contributors have to pick one without clear guidance, and the decision matrix is non-obvious (third-party for new code? in-house for existing? what about hybrid code paths?).
2. **Adding `filelock` as a core dep bought the dep cost anyway.** The in-house module is no longer "savings" — it's duplicated infrastructure.
3. **The in-house Windows path is less tested.** `test_process_lock.py` covers POSIX well but the Windows branch has minimal coverage. The third-party library has a much larger battle-tested user base on Windows.
4. **Name collision (pre-rename) was confusing.** Even after renaming the in-house module to `process_lock`, someone reading `from filelock import FileLock` and `from maxim.utils.process_lock import file_lock` in adjacent files has to mentally context-switch between two lock APIs.

## The fix (when it's time)

**Option A — migrate in-house callers to `filelock`.** Rewrite `models/download.py::ensure_available` to use `filelock.FileLock` directly. Delete `maxim/utils/process_lock.py` + `tests/unit/test_process_lock.py`. All locking in the codebase goes through one library.

- **Pros:** Single API, mature cross-platform, zero in-house code to maintain.
- **Cons:** Breaking change for the download path's `LockContended` exception semantics — callers that catch it would need to catch `filelock.Timeout` instead (check `ensure_available` for try/except handling).
- **Estimated scope:** ~50 LOC migration + test rewrite + ensure `ensure_available`'s retry semantics still match.

**Option B — migrate `drain_state.py` to `process_lock`.** Rewrite `peer/drain_state.py` to use `file_lock` + `LockContended`. Remove `filelock>=3.0,<4.0` from pyproject. Keep the in-house module as the sole locking API.

- **Pros:** No new external dep, fewer moving parts.
- **Cons:** In-house module needs a `timeout` argument (current API is non-blocking only — raises immediately on contention). Adding timeout requires a polling loop or a platform-specific blocking acquire, which re-implements what `filelock` already does well. Windows path in the in-house module is less tested and this change would depend on it more heavily.
- **Estimated scope:** ~80 LOC for the timeout-aware in-house API + drain_state migration + expanded Windows test coverage.

**Option C — make `process_lock` a thin wrapper around `filelock`.** Keep the in-house API shape (`file_lock` context manager + `LockContended` exception) but delegate implementation to `filelock.FileLock`. Callers don't change; the deduplication is invisible.

- **Pros:** Zero breaking changes, single implementation underneath, in-house API stays as an adapter layer.
- **Cons:** Still two API shapes in the codebase — just with one underneath. Defers the real decision.
- **Estimated scope:** ~30 LOC adapter + test.

**Recommendation (for the future session that picks this up):** Option A. The third-party library is better for the longest-term maintenance story, and the breaking change surface is small (one exception-class rename in one call site). Wait until Plan 4 C3 lands to minimize churn — C3 will likely add more locking call sites and we want to make the decision once.

## Migration checklist (for when this plan activates)

- [ ] Audit all uses of `maxim.utils.process_lock` + `filelock` across `src/` and `tests/`
- [ ] Pick Option A/B/C based on the audit (current plan: A)
- [ ] Update `models/download.py::ensure_available` to catch `filelock.Timeout` instead of `LockContended`
- [ ] Delete `maxim/utils/process_lock.py` + `tests/unit/test_process_lock.py`
- [ ] Update `tests/unit/test_ensure_available.py` mock target
- [ ] Update CLAUDE.md "mesh.yml is declarative" lesson — drop the `filelock.FileLock` mention and replace with the generalized locking pattern
- [ ] Pre-merge review round (one lens is fine for this — scope is small)
- [ ] Post-merge smoke test: run `maxim --auto-download` to confirm the download lock path still serializes correctly on two parallel invocations

## Open questions

1. **Is there any call path that relies on `LockContended` vs `Timeout` distinction?** E.g., a caller that catches one but not the other to distinguish "lock held by me" from "lock held by another process." Check before migration.
2. **Does the third-party `filelock` handle NFS better than the in-house version?** The in-house module's docstring explicitly says "NFS and network filesystems are **not** officially supported." If the third-party library has better NFS semantics, the migration is a strict improvement; if it has the same caveats, the migration is neutral.
3. **Windows coverage gap in `test_process_lock.py`.** Before deleting the in-house module, add a regression test for the Windows branch via `sys.platform` monkeypatch — the review should ensure no Windows-specific edge case was relying on the in-house implementation's quirks.

## Non-goals

- This plan does NOT touch `maxim.utils.atomic_io` — that's a separate concern (atomic file writes, not advisory locks). Atomic writes continue to use `os.replace` + fsync; locks continue to be advisory.
- This plan does NOT change the drain state design. Drain state stays in `~/.maxim/util/drained_nodes.{role}.txt` with the filelock-around-RMW serialization; only the locking library underneath changes.
- This plan does NOT change any user-facing behavior. Zero CLI, log output, or exit code changes.

## Why this is a shell plan, not active work

Plan 4 C2's implementer filed this plan because the name collision was the most visible symptom of a larger deduplication opportunity, but:

1. C2 shipped on a tight schedule after a pre-design review pivot and there was no budget for the audit + migration
2. The rename alone (`filelock.py` → `process_lock.py`) resolved the immediate footgun
3. C3 is about to add admin API write paths that may need their own locking, so waiting to consolidate until C3's surface is visible avoids a double-migration
4. The in-house module works today and has no active bugs

Revisit after C3 ships. If C3 doesn't add new lock users, this plan can ship standalone as a ~1-2 hour cleanup.
