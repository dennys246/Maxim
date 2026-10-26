# #1338 (advisor item 6): Executor lens review of d362b481

Reviewed `git show d362b481` against docs/CODE_REVIEW.md (Executor). Tests run with
`PYTHONPATH=<worktree>/src`: the touched test files give 158 passed, 4 xfailed (O4/O6/O9, owned elsewhere).

## Deletion probes (on a scratch copy of src/ + tests/; the worktree was not touched)

| Probe | Mechanism removed | Result |
|---|---|---|
| P1 | `create.py::pool`, the `llm_router is None` raise | `test_o8_create_pool_wires_an_llm_or_raises` + `test_create_pool_without_an_llm_raises` FAIL |
| P2 | `agent_pool.py::_generate_npc_response`, the LLM call wrapped back in `except: pass` (stub fallback) | `test_o8_run_turn_llm_failure_is_typed` + `test_run_turn_empty_llm_text_is_an_error_not_a_stub` FAIL |
| P3 | `_generate_llm_response`, the empty-text raise | `test_run_turn_empty_llm_text_is_an_error_not_a_stub` FAILS. The O8 gate does not cover this case, and the unit test does. |
| P4 | `cli.py::_start_server_llm`, `wait_ready`'s result ignored | `test_o8_server_mode_not_ready_is_reported` FAILS |

All three flipped O8 gates fail when their mechanism is deleted.

## BLOCKER

**B1. Diff coverage is 36.1% (22/61), below the 80% gate. Location: `cli.py::_start_server_llm`, `cli.py::_run_server_mode`.**
I measured this with `scripts/lint_coverage.py::diff_coverage` against `d362b481~1`. The coverage data came from the
4 touched test files plus every `tests/unit/test_cli*.py`. cli.py has 2 of 41 changed statements covered; create.py 5/5
and agent_pool.py 15/15 are fully covered. The only test that executes server mode is the O8 gate, and it runs in an
`-I` subprocess that coverage does not record. `test_leader_fail_closed` only reads the source text. A grep finds no
other test that reaches `_run_server_mode` or `_start_server_llm`, so the full suite should give the same result.
Failure scenario: the CI coverage job (and `suite_at_commit.sh`) fail the push.
The following paths run under no test in-process:
- router `None` → return 1
- `LeaderRefusal` → return 2
- init exception → return 1
- the ready path → idle wait → `KeyboardInterrupt` → return 0
- `stop_active_spawner` in the `finally`

Fix: add in-process unit tests that monkeypatch `lane_backends.build_primary_router`, `threading.Event` and
`stop_active_spawner`. Assert the return code and the stderr text for each path, and assert that the spawner is
stopped on every exit.

## SHOULD-FIX

**S1. Server mode now exits 1 on more than a failed `wait_ready`. Location: `cli.py::_start_server_llm`.**
The owner decided that a failed `wait_ready` exits 1. The commit also exits 1 when the router is `None` and when init
raises (the CHANGELOG states this). Before, both of those cases warned and kept serving. A leader with no action
enters server mode by default (`_main_impl`: `if not _has_action and (_is_leader or _has_llm)`). On that exit the
early-booted LeaderProxy dies along with the LLM.
Failure scenario: a leader's model fails to build, so the leader exits. Peers then lose
`maxim peer logs`, `peer llm <model>` and `peer update`, which are the remote recovery path. A supervisor that
restarts the leader will loop on the same failure.
This may be intended. Confirm with the owner that the decision covers the router-`None` and init-exception cases,
not only `wait_ready`.

## NIT

- **N1. `_run_server_mode` compared line by line with the old block.** The following match the old code:
  - the signal handling: the SIGINT handler is unchanged, and a second Ctrl+C still calls `os._exit(1)`
  - the `LeaderRefusal` path still returns 2, and its except clause still comes before `except Exception`
  - the "Proxy running", usage and "Shutting down." messages are unchanged

  The `finally` now wraps the LLM start as well. As a result, the spawner is now also stopped on a refusal (exit 2),
  on not-ready (exit 1), and on a Ctrl+C during the 180 s `wait_ready`. The old code skipped all three, so this is an
  improvement. The warning text moved from stdout (`WARNING: ...`) to stderr (`✗ ... not ready`); no scraper of the
  old text was found in src/, scripts/ or docs/.
- **N2. `run_round` and `broadcast_percept` are unaffected. Location: `agent_pool.py::run_turn`.**
  `ModelInferenceError` is caught by `run_turn`'s outer `except Exception` and becomes `TurnResult(error=...)`.
  `run_turn` does not raise, so the sequential and concurrent paths both get the error result. The NAc observation is
  skipped on a failed turn, which is correct.
- **N3. The empty-text change is in scope.** #1338 says "the fallback becomes a typed error", and the old fallback
  also fired on empty text. `src/` and `scripts/` contain no production `AgentPool` caller other than `create.pool`.
  A pool built directly with `AgentPool()` and no router still returns the stub silently. The docstring documents
  this, and `AgentPool` is reached through `create.pool`.
- **N4.** The rule-T baseline (1588 → 1548) and the swallow lint are clean. The moved `except Exception: pass` in
  `_run_server_mode` was credited as a move.
