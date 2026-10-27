# Item 6 (#1338) — wire-integrity lens

Diff: `d362b481` (fix(#1338): fail loud on LLM readiness). Read-only review; searched `src/`, `scripts/`, `tests/`,
`docs/` (user, troubleshooting, agents, plans), `CHANGELOG.md`, `ARCHITECTURE.md`, `.github/workflows/`. No
`examples/` or `notebooks/` directory exists; `src/maxim/ui_dist` and the console have no pool reference.

## Producer → consumer tables

### C1. Call: `maxim.create.pool(*, max_workers=4, llm_router=None) -> AgentPool`, raises `ConfigurationError`

| Contract | Producers (before → after) | Consumers (before → after) | What each consumer gets, before → after | Probe |
|---|---|---|---|---|
| `create.pool()` signature + raise | `create.py::pool` (only constructor of a pool in `src/`; no other `AgentPool(` in `src/` or `scripts/`) | **Production: none.** `src/`/`scripts/` have zero callers. Test-only: `test_composable_api.py::TestCreatePool` (4, all pass `object()` or no router), `test_atonia_open_items_other.py::test_o8_create_pool_wires_an_llm_or_raises`. Claims: `create.py` module + function docstrings, `docs/user/python-api.md` §pool, `docs/user/getting-started.md`, `docs/user/stable_api.md` row, `CHANGELOG.md` | no-arg call: pool with `_llm_router=None` → `ConfigurationError`. `llm_router=create.router(...)`: pool → pool **only if `create.router` returned non-None** (it returns `build_primary_router()[0]`, which is `None` when the large tier resolves to nothing) → then `ConfigurationError` saying "pass llm_router=" to a caller who did | 3, 6 |
| `AgentPool(llm_router=None)` direct construction | unchanged | `test_agent_factory.py` (8 `AgentPool()` sites, test-only); public via `stable_api.md` "AgentPool … returned by factory functions" | still the persona-stub path; the raise lives only in `create.pool`, not the type | 2, 9 |

### C2. Call/payload: `AgentPool.run_turn` → `TurnResult{response, error, percept}`; the router call inside it

| Contract | Producers (before → after) | Consumers (before → after) | What each consumer gets, before → after | Probe |
|---|---|---|---|---|
| `_generate_llm_response` calls `self._llm_router.generate(prompt, function="npc_dialogue", max_tokens=100)` | **no producer of `generate` on `LLMRouter`** (verified: `'generate' in dir(LLMRouter)` is False; public surface is `route`, `generate_json`, `warmup`, `wait_ready`…; `create.router` docstring promises `.generate()` falsely). `"npc_dialogue"` is in no registry | `agent_pool.py::_generate_npc_response` | before: `AttributeError` swallowed at DEBUG → persona stub. after: `ModelInferenceError("… 'LLMRouter' object has no attribute 'generate'")` → **every turn errors** (reproduced, see B1) | 1, 3 |
| `TurnResult.response` / `.error` | `run_turn` (ok / except branch), `run_round` (future exception) | **Production: none in `src/` or `scripts/`.** `console/handle.py` and `simulation/minecraft_harness.py` explicitly do not ride `run_turn`; Party Mode / campaigns / DM runtime do not use `AgentPool`; nothing serialises `TurnResult` (no report/console payload). `run_round`, `broadcast_percept` pass results through unchanged. Test-only: `test_agent_factory.py`, `test_atonia_open_items_other.py::test_o8_run_turn_llm_failure_is_typed`. External API users via `python-api.md` | LLM failure: `response="*persona*", error=None` → `response=None, error=str(ModelInferenceError)`; type and `percept` dropped on the error branch | 4, 7 |
| `response=None, error=None` | `_generate_npc_response` when `instance.personality` is empty (router configured or not) | same | unchanged, but now indistinguishable from the docstring's "failed turn = error set, response None" only by `error` | 4 |

### C3. `ModelInferenceError` vs existing handling

| Contract | Producers | Consumers | Gets | Probe |
|---|---|---|---|---|
| `ModelInferenceError` (subclass of `ModelError` → `MaximError`, takes `context=`) | `agent_pool.py::_generate_llm_response` (first raiser in `src/`) | `run_turn`'s `except Exception` only; flattened to `str(e)`. Never reaches `cli.py::main`'s typed-error surfacing or `BackendError` handling (`ModelError` is not a `BackendError`) | a string; class and `context` discarded | 4, 7 |

### C4. Operator surface: server mode's exit code (`cli.py::_run_server_mode` / `_start_server_llm`)

| Contract | Producers (before → after) | Consumers (before → after) | What each consumer gets, before → after | Probe |
|---|---|---|---|---|
| exit code + stdout/stderr text | `_main_impl` inline block → `_run_server_mode` | Entry: implicit — `MAXIM_ROLE=leader` or `MAXIM_LLM_ENABLED=1` with no action sets `args.mode="server"` (`cli.py` ~1986). Re-entry: `leader_proxy.py::_handle_admin_restart` (`os.execv` same argv) via `peer/admin_core.py::restart_on_target` (`maxim peer restart`) and `mesh_cli.py::_run_node_restart`. Supervisors: no unit file in repo; `docs/experiments/protocols/llm_path_stress_test.md` cites `systemctl restart maxim-leader`. Text greps: none in `src/`/`scripts/` (only the O8 test reads "not ready") | not-ready / router None / init exception: `0` + proxy stays up idle → `1` + **process exits, taking the in-process LeaderProxy with it**. Refusal `2` unchanged. Ctrl+C in idle wait `0` (unchanged; contradicts CHANGELOG #1468 "130 wherever it lands"). `docs/user/cli-reference.md` exit-code table does not list server mode | 6, 10 |

### C5. Persisted lint data: `function_length_baseline.json` pin `_main_impl` 1588 → 1548

| Contract | Producers | Consumers | Gets | Probe |
|---|---|---|---|---|
| pin + note | this diff | `scripts/lint_function_length.py` (clean, run here), `tests/unit/test_lint_release_gates.py` (reads file), `_evidence_records.py` / `o19_verdict.py` (`SUBJECT_EXCLUDED`, path only) | tighter pin; lint "clean"; no other reader keys on the number | 5 |

## Findings

### BLOCKER

**B1. `runtime/agent_pool.py::AgentPool._generate_llm_response` calls a method no real router has.**
`LLMRouter` has no `generate` (only `generate_json`, `route`, private `_complete_text`). Failure scenario, reproduced
with a real `LLMRouter` instance: `pool = maxim.create.pool(llm_router=<LLMRouter>)`, `pool.run_turn("guard", …)` →
`response=None, error="NPC 'guard' LLM inference failed: 'LLMRouter' object has no attribute 'generate'"`, every turn,
every agent. The documented happy path (`python-api.md`, `getting-started.md`, `create.pool` docstring,
`stable_api.md`) now fails 100% of turns; before, the same `AttributeError` was the silent stub. The diff makes it
loud (correct direction) but ships a "fix" whose only working caller is a `MagicMock` without `spec`
(`test_agent_factory.py::test_run_turn_uses_the_llm_text`, `test_composable_api.py` passes `object()`). Probe 1/3:
zero real producers of the called symbol. Fix needs an owner call on the surface (public text-completion method on
`LLMRouter`, or route through an existing one), plus a test with `MagicMock(spec=LLMRouter)` or a real router over a
stub backend that goes red when the call is wrong. Same latent bug: `bridges/tool_pain_bridge.py` (`self._llm.generate`)
and `create.py::router`'s docstring — one seam (probe 9).

### SHOULD-FIX

**S1. `create.py::pool` — `create.router()` can return `None`.** `build_primary_router` returns `(None, mgr)` when
the large tier resolves to nothing; `create.router` passes that through. `create.pool(llm_router=create.router("x"))`
then raises "needs an LLM router: pass llm_router=…" to a caller who passed one. Distinguish "router build returned
None" (or make `create.router` raise) so the message names the real cause.

**S2. `cli.py::_start_server_llm` exit 1 now kills the leader proxy, including after a remote restart.** Server
mode is the leader's default; `maxim peer restart` re-execs into it. A model that fails warmup (bad profile after
`peer update`, a transient cloud/peer 429 during warmup's `backend.complete("Hello")`, a >180 s cold load) now exits
the process, so `/v1/admin/llm-swap`, `peer logs`, `peer llm --status` become unreachable and recovery needs shell
access. `peer/admin_core.py::restart_on_target` then prints "Leader did not respond … Check with: maxim peer
version", which cannot help. Owner chose exit 1; the wiring consequence needs either an operator note
(`docs/troubleshooting/remote_update.md`, the `restart_on_target` hint) or a decision recorded that the remote
recovery path is accepted lost. Also `docs/user/cli-reference.md`'s exit-code table omits server mode (`1` not ready,
`2` refusal).

**S3. `runtime/agent_pool.py::AgentPool.run_turn` flattens the typed error.** `ModelInferenceError` becomes
`error=str(e)`; a consumer cannot tell an LLM failure from a hippocampus/percept failure without substring matching,
and `percept` is dropped on the error branch. The CHANGELOG says "a `ModelInferenceError` inside", but nothing outside
can observe it. Consider an `error_type`/exception field (TurnResult is a non-frozen dataclass; adding a defaulted
field is compatible).

### NIT

**N1. `cli.py::_run_server_mode`** — Ctrl+C in the idle wait returns `0`, against CHANGELOG #1468 "Ctrl+C exits 130
wherever it lands". Pre-existing; the function is new, so it is the cheap place to align.

**N2. `agent_pool.py::_generate_npc_response`** — empty personality returns `response=None, error=None` even with a
router; the new `run_turn` docstring ("a failed turn … error set and response None") leaves this third outcome
undocumented.

**N3. `create.pool` raise lives in the factory, not the type** — `AgentPool(llm_router=None)` remains the stub path
and `AgentPool` is listed public in `stable_api.md`. Fine if intended; say so in the `AgentPool` docstring.

C5 (baseline pin): no finding; `lint_function_length.py` clean, swallow lint clean (verbatim `finally` move),
`[Unreleased]` lint clean.
