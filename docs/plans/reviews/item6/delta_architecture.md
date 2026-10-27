# #1338 (advisor item 6) — Architecture lens, DELTA round

Scope: the staged fold (`git diff --cached` on top of `d362b481`), answering `architecture.md`, `executor.md`, `wire.md`.
Read: docs/CODE_REVIEW.md, docs/agents/llm-routing.md, docs/agents/runtime-tools.md, CLAUDE.md invariants.
Checks run: `lint_no_silent_swallows.py` clean; `lint_function_length.py` clean (touch rule included).

## Answers to the asks

- **`LLMRouter.generate()`: right layer.** It lives in `models/language/router.py` and rides `_complete_text`, so it gets
  the same cost ceiling, inference lock, `_try_provider` failover and accounting as `generate_json`. No backend import
  leaks out, and no parallel dispatch path appears.
  - Error contract: `ModelInferenceError` is an existing `ModelError` subclass, not a parallel type, and `LeaderRefusal`
    passes through as itself.
  - `_complete_text` already absorbs typed `BackendError`s into `""` inside `_try_provider`, so the broad
    `except Exception` only wraps residual faults, and it keeps the cause with `from e`.
  - `max_tokens=256` is a sensible default for plain text (`generate_json` uses 1024 for structured output).
  - Documentation: stable_api.md records it. The llm-routing brief does not (N1).
  - Callers in `src/`: `agent_pool.py::_generate_llm_response` and `bridges/tool_pain_bridge.py::_generate_reflection`.
    The second has no production `llm=` (see N4).
- **leader_proxy state.** It follows the mutable-globals lesson: `cli.py` imports the module (`from maxim.runtime import
  leader_proxy`) and calls accessors, never binding `_server_llm_status` by name. `server_llm_status()` returns a copy,
  and the assignment is a single atomic rebind.
  - `/v1/debug/ping` is the right surface. Peers and doctor already poll it, it already carries `llm_ready`, and no
    separate health endpoint exists.
  - It is authenticated: `do_GET` routes every `_is_debug_path` through `_check_auth`, which uses ProxyAuth.
- **`create.router` raising.** Documented honestly: it sits under `### Changed` with the "Breaking, to stable surfaces"
  label and a migration line, and the stable_api row is marked breaking with the owner decision cited. That matches the
  #939 precedent.
- **`TurnResult.error_type`.** `TurnResult` is a plain `@dataclass`, not frozen, so CC3 does not apply. The new field has
  a default (N3).
- **Swallow move credit.** The old `finally: except Exception: pass` is gone. `_stop_server_spawner` now logs at
  WARNING, and the lint is clean with no `swallow_moves.json` record needed. `_start_server_llm` logs, then returns a
  reason (handle-and-log), and `_generate_llm_response` re-raises typed errors.
- **Rule T.** The leader_proxy edits are a 2-line docstring/payload change in `_handle_debug_ping` (not pinned) plus new
  module-level functions. The touch-rule lint is clean.

## BLOCKER

None.

## SHOULD-FIX

**S1 (fail-closed). `cli.py::_report_server_not_ready` uses `leader_proxy.leader_proxy_running()`, which is true only
for the DEFAULT-port proxy.** `start_leader_proxy` records `_leader_proxy_server` only when
`proxy_port == DEFAULT_PROXY_PORT`.
- **Failure scenario:** a leader started with `MAXIM_LEADER_PROXY_PORT=9099` and an LLM that never becomes ready. A
  proxy is running, yet the process prints "No LeaderProxy runs in this process: exiting (1)" and exits, which loses the
  remote recovery path.
- **Why fail-closed:** the outcome is a loud exit, the same as the pre-fold behaviour.
- **Two problems:** the stderr text is false in this case, and cli-reference.md's "Behind a LeaderProxy … does not exit"
  overclaims.
- **Fix:** have the proxy start record "a proxy is serving in this process" independently of the idempotency singleton,
  or reword both texts to "default-port LeaderProxy".

## NIT

- **N1 (fail-closed).** docs/agents/llm-routing.md's Router row lists the router's public entry points by implication
  only. Add `generate` (plain text, raises `ModelInferenceError`) beside `generate_json`, so the next reader does not
  re-add a shim.
- **N2 (fail-closed). `server_llm` is a startup verdict and is never updated.**
  - After a successful `maxim peer llm <model>` swap without a restart, ping shows `llm_ready: true` beside
    `server_llm.ready: false`. That is conservative and documented as "startup verdict", and nothing machine-reads it.
  - Consider clearing or updating it on a successful llm-swap, or stating in leader_proxy_debug.md that it is
    meaningful only until the next restart.
  - The `reason` carries `str(exception)` from init failures. That is acceptable behind auth, but under `--insecure` it
    is served unauthenticated (as `/v1/debug/logs` already is).
- **N3 (fail-closed).** `error_type` is inserted between `error` and `duration_ms`, which shifts the position of every
  later field for any external positional `TurnResult(...)` construction. In-repo constructions are all keyword.
  Appending the field last costs nothing.
- **N4 (overclaim, fail-closed).** CHANGELOG: "the Reflexion path of `ToolPainBridge` called a `.generate` no router
  had".
  - True, but `runtime/bootstrap.py` builds `ToolPainBridge` with no `llm=`, so Reflexion has no production LLM and this
    fold does not activate it.
  - Say "(not wired to an LLM in production)" so no reader infers that Reflexion now runs, which matters for the B4
    Reflexion line.
- **N5 (fail-closed). Stale or inconsistent text.**
  - The `function_length_baseline.json` note for `_main_impl 1588 -> 1548` says server mode "exits non-zero". Behind a
    proxy it now stays up.
  - cli-reference.md's server-mode paragraph says "since the next release after 1.3.3", where the adjacent #1468
    paragraph says "since 1.3.4". Pick one.
  - Server mode's Ctrl+C returning `0`, against #1468's "130 wherever it lands", is pre-existing (round 1 N1). The fold
    now documents the `0`, which is honest, but the two tables in the same file disagree. Either align the code or add
    one clause explaining why server mode differs.

STOP: no further round owed. Nothing blocks, and the one SHOULD-FIX (S1) fails closed.
