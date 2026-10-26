# Atonia red gates: second delta round, Executor lens

Scope: only the staged fold answering `redgates_delta_executor.md` S1 and `redgates_delta_architecture.md` D1, D2 and
D3. The fold covers `test_no_effect`'s snapshot, directory-open and rename checks, `_REGISTRY_READERS` and its
classification guard, the two new offline-latch gates, and the conftest `_reset_offline_latch` fixture. Read-only. All
probes ran from the scratch dir against a copy of `src/` with stub implementations.

## Measured

- **Counts.** The PR's seven test files give **15 passed, 186 xfailed**. With `--runxfail` and junit they give **186
  failed, 15 passed**. All 186 are `<failure>` with an `AssertionError` message. There are no `<error>` elements.
- `python scripts/lint_release_gates.py` prints `release gates: clean` and exits 0.
- **(a) `test_no_effect`.** I ran it against a stub `maxim.advisor` in a `src/` copy, with `m.SRC` re-pointed and the
  fake backend. The stub imports numpy 2.2.5 and threadpoolctl 3.6.0 at import time. It opens `home_dir` and
  `report_dir` with `O_RDONLY|O_DIRECTORY|O_NOFOLLOW` and writes the log `O_EXCL` through the report_dir fd. It writes
  the report as a temp file plus `os.replace(..., src_dir_fd=, dst_dir_fd=)`.

  | Stub variant | Result |
  |---|---|
  | Correct (as above) | **PASS**. Neither Python's import machinery nor numpy or threadpoolctl produced an `open_dir` event. Directory walks go through `os.listdir`/`os.scandir`, which raise no `open` event. |
  | `os.replace` of a relative temp to an absolute destination outside `tmp_path` | **FAIL**, at the both-paths rename check. |
  | Write through the dir fd of an outside absolute directory (opened without `O_DIRECTORY`) | **FAIL**, at the `open_dir` check. |
  | `os.open("../harness")` relative, used as a `dir_fd` base | **FAIL**, at `open_dir`. |
  | `os.open("..", dir_fd=report_fd)` used as a base | **FAIL**, at `open_dir`. |

- **(c) Offline latch gates.** I wrote a stub `maxim/_offline.py` that follows §1A.1's three steps. It has a stdlib
  audit hook that refuses `AF_INET` connects, loopback included (V8), and getaddrinfo. It has `on_engage` and
  `engage_callbacks`, and `_reset_for_tests`.

  | Stub variant | Result |
  |---|---|
  | `utils/http.py` copy that does not register `close_all` | Pooled gate **FAILS** (`did not register close_all`). |
  | `close_all` registered with `on_engage` | Pooled gate **PASSES**. The fresh connect after `close_all` is refused by the latch's hook, which runs before the spy, so the spy sees nothing. |
  | `close_all` registered, but `engage()` skips the callbacks | Pooled gate **FAILS**: the listener saw a third request on the same port. |
  | Correct flags | Flags gate **PASSES**, with ultralytics 8.3.248 installed and exercised. |
  | `ultralytics.utils.ONLINE` not set | Flags gate **FAILS**. |
  | `huggingface_hub.constants.HF_HUB_OFFLINE` not set | Flags gate **FAILS**. |

  Neither gate can pass vacuously:
  - `require(pre_ports two and equal)` proves the connection was pooled.
  - "Listener saw nothing" is paired with `refusal or ledger_delta > 0` and with `_assert_spy_saw_nothing`.

  Both are safe:
  - The listener is loopback only.
  - The pooled body's pre-engage requests go only to loopback.
  - ultralytics is imported under `YOLO_OFFLINE=true` with the spy installed first. Any probe is blocked and logged as
    `spy_preimport`.
- **(d) Fixture probe.** In a toy conftest, `pytest.fail` in an autouse fixture's teardown reports `ERROR at teardown`
  on the test that loaded the module, and on every later test. The xfail in the loading test does not mask it. This
  fails closed and is loud. `-x` stops at the culprit.

  Restoring the five keys after `_reset_for_tests()` is safe:
  - `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` go back to the conftest-forced `"1"`, which keeps the hermetic invariant.
  - `monkeypatch` teardown in either order converges on the same values.
  - `finally` restores the env even if the reset raises.

## BLOCKER

None.

## SHOULD-FIX

None.

## NIT

- **N1 [fail-closed] (a).** `open_dir` resolves a relative directory open against the CWD, not its `dir_fd`. It is
  right today only because `cwd` and `report_dir` are siblings at the same depth under `tmp_path`.
  - A correct build that opens `"."` relative to the report_dir fd would be flagged.
  - An adversarial relative path that is a directory from `report_dir` but not from `cwd` would go unrecorded. The
    `root` tree diff still catches any landing inside `tmp_path`.
  - Worth one comment line: keep `cwd` and `report_dir` siblings.
- **N2 [fail-open, latent] (b).** `_registry_public_methods` filters on `callable(v)`. A future public *property* on
  `ToolRegistry` that hands back mutable internals would escape both the classification guard and the sweep.
  - None exists today: 20 public methods, all callable, exactly `_REGISTRY_READERS` (14) plus `_REGISTRY_WRITERS` (6).
    The MRO is `ToolRegistry, object`.
  - No reader body mutates `self`, checked by AST/regex.
  - Cheap fix: classify every public name, not only callables.
- **N3 [fail-closed] (c).** The `close_all_registered` assertion pins identity (`cb is http.close_all`). A lambda or
  wrapper registration fails the gate even though it works. The docstring pins this seam, so that is acceptable.
- **N4 [fail-closed] (c).** `_FLAGS_BODY` imports `huggingface_hub` unconditionally. It is not a core dependency; CI
  gets it only transitively, through faster-whisper. On a box without it, the child crashes. The result is
  `ScenarioBroken`, which reports FAILED rather than xfail. This is loud, and it matches the existing `_hf_direct`
  site.
- **Not exercised:** the `real` backend leg of `test_no_effect`, meaning llama_cpp's Python-level opens.

## Verdict

The S1, D1, D2 and D3 folds are complete. A correct v6.4 build passes them, and every misbehaving stub I tried fails
them. No BLOCKER and no fail-open SHOULD-FIX remain.

**STOP: no further round owed.**
