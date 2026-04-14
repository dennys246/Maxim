# LLM Path Refinement — Plan 3.5: Cancellation Hygiene

**Status:** Draft v1 — 2026-04-13
**Scope:** ~150 LOC modified + ~80 LOC new tests
**Target version:** 0.4 (blocker for Phase D stress test)
**Part of:** [llm_path_refinement.md](llm_path_refinement.md)
**Depends on:** [llm_path_fast_failover.md](llm_path_fast_failover.md) (Plan 3 R2.5 + R2.6) — `_MaximPeerBackend`, router typed dispatch, `_inference_lock`
**Blocks:** [llm_path_operator_visibility.md](llm_path_operator_visibility.md) (Plan 4) — Phase D stress test validates cancellation before operator-visibility work can rely on it

## Goal

**Fix the cancellation leak in `LLMWorker._call_llm_with_timeout` that makes the agent-level timeout a lie.**

When the agent-level 60s timeout fires, `future.cancel()` sets a flag but the underlying thread continues executing inside `LLMRouter._inference_lock`. The next LLM call blocks on the lock, hits its OWN 60s timeout, and the cycle repeats until the upstream HTTP request eventually errors (observed: Cloudflare 524 at ~125s). Provider state gets polluted by the orphaned thread's eventual `_note_provider_failure` call.

This plan fixes the cancellation path so:
1. The agent-level timeout is either real (actually aborts the call) or explicitly a safety net above a shorter HTTP-level timeout that fires first.
2. `_inference_lock` is never held by an orphaned thread across a timeout boundary.
3. `_provider_states` is not polluted by cancelled calls.
4. The mesh-era hardcoded `60.0` default is replaced with a value aligned to the HTTP layer's authoritative timeout.

## Evidence

**Observed in stress test trace2 (`/tmp/maxim_trace2.jsonl`), 2026-04-13:**

```
T+0.0s    probe_completed ok (liveness_ms=389.5)      leader is fine
T+62.0s   llm_worker WARN  "LLM call timed out after 60.0s"   first timeout
T+122.5s  llm_worker WARN  "LLM call timed out after 60.0s"   second timeout
T+125.4s  peer_backend_failed status=524 latency_ms=125401.2  Cloudflare 524
T+125.4s  dispatch_exhausted total_elapsed_ms=125401.5
```

Two 60s timeouts stacked back-to-back for a single logical LLM request. A single HTTP call's true latency was 125s. Pattern repeats on every AUT turn in the heist_v1 campaign run.

**Review agent findings (both independently, 2026-04-13):**

1. [llm_worker.py:400](../../src/maxim/agents/llm_worker.py#L400) — `future.cancel()` on a running future only sets a flag; the thread continues executing inside `_inference_lock`.
2. [router.py:521-528](../../src/maxim/models/language/router.py#L521-L528) — the orphaned thread's eventual error path calls `_note_provider_failure()`, polluting `_provider_states[provider_key]`. Instance-scoped, never auto-resets.
3. Mesh-era 60s defaults scattered across [llm_worker.py:155](../../src/maxim/agents/llm_worker.py#L155), [loop_controller.py:160](../../src/maxim/runtime/loop_controller.py#L160), [agent_loop.py:1307](../../src/maxim/runtime/agent_loop.py#L1307) — all default to the same magic number, consistent with a refactor artifact from the pre-reactive-mesh architecture.

## Non-goals

- **Not switching to async httpx.** Full async rewrite is out of scope. We'll keep the thread-per-call pattern and fix the cancellation contract around it.
- **Not investigating why the leader needed 125s for "say hi".** That's a separate perf bug (probably AUT prompt-size explosion from memory/ATL/causal context). Tracked as a follow-up, not gated by this plan.
- **Not removing the agent-level timeout entirely.** Runaway token generation (streaming infinite output) still needs a safety net — the HTTP layer's per-chunk `read_s` doesn't catch it. The agent-level timeout stays, but becomes a real cancel-or-guaranteed-release contract.
- **Not deleting `src/maxim/mesh/`.** Still imported by simulation code. Dead-code cleanup is tracked separately.
- **Not touching the cloud backend 60s client-init timeouts.** Cloud providers have their own SDK timeout conventions; if a follow-up plan decides to align them, it can.

## Architecture — the contract after this plan

**Single rule:** the HTTP layer is the authoritative timeout. The agent layer is a safety net strictly above it.

```
User request → LLMWorker._call_llm_with_timeout (agent-level safety net, 300s)
  → router.generate_json (acquires _inference_lock)
    → router._try_provider (typed exception dispatch)
      → _MaximPeerBackend.complete_with_usage
        → _http.post (authoritative HTTP timeout, 300s — from _INFERENCE_PROXY_TIMEOUT_S)
```

- **HTTP layer fires first (300s).** Raises `HTTPTimeout` → `BackendTimeout` → router catches, records attempt, releases lock cleanly. Normal path.
- **Agent layer fires second (300s + buffer, e.g., 330s).** ONLY if HTTP layer is wedged (deadlock, stuck thread, bug). When it fires, it's a LOUD signal something is wrong upstream — not a routine event.
- **Lock release is guaranteed** by `try/finally` around every lock acquisition in the router. Verified by a reproducer test.
- **Provider state is clean** even if the orphaned thread eventually errors — cancellation check before `_note_provider_failure`.

## Scope — the R-stages

### R1 — Reproducer test (do first, no code changes)

Write a unit test that:
1. Stubs `LLMBackend` with a `complete_with_usage` that `time.sleep(120)` before returning.
2. Calls `LLMWorker._call_llm_with_timeout` with `llm_timeout_s=1.0`.
3. Asserts the timeout fires in ~1s and returns the fallback.
4. **Asserts `router._inference_lock` is not held after the call returns.** (Currently fails.)
5. **Asserts `router._provider_states[provider_key]` is not polluted.** (Currently fails.)

This test lives in `tests/unit/test_llm_worker_cancellation.py`. Must be skipped if `_inference_lock` isn't accessible (e.g., in mocked router tests) — use a real router instance with a mocked backend.

### R2 — Align timeouts

- Raise [llm_worker.py:155](../../src/maxim/agents/llm_worker.py#L155) default `llm_timeout_s: float = 60.0` → `300.0` (matching `_INFERENCE_PROXY_TIMEOUT_S` conceptually, but a safety net).
- Add env var `MAXIM_LLM_CALL_TIMEOUT_S` with range clamping, same pattern as the probe timeouts in lane_backends.
- Update [loop_controller.py:160](../../src/maxim/runtime/loop_controller.py#L160) and [agent_loop.py:1307](../../src/maxim/runtime/agent_loop.py#L1307) fallback defaults to read the same env var.
- Document the new "HTTP layer fires first" contract in `docs/architecture/llm_routing.md`.

### R3 — Guaranteed lock release

- Audit [router.py](../../src/maxim/models/language/router.py) `_complete_text_locked`, `_try_provider`, and any other method that acquires `_inference_lock`. Every acquisition must be in a `try/finally` that releases on any exception.
- Add a defensive `assert not self._inference_lock.locked()` at the end of the LLMWorker submission path (the repro test's main assertion).
- No new threading primitives — just ensure exception paths don't skip the release.

### R4 — Provider state hygiene on cancellation (ContextVar + Event hybrid)

**Design:** store a `threading.Event` as the *value* of a `ContextVar`. The `ContextVar` handles propagation (automatic capture at `ThreadPoolExecutor.submit` time — no parameter plumbing through router/backend call chains), and the `Event` handles cross-thread signalling (main thread calls `event.set()`, orphaned background thread sees it on next check). Matches the existing codebase convention — `utils/http.py` already uses `ContextVar` for `RequestContext` / X-Maxim-* header propagation, so reviewers find the cancellation pattern in the same place.

**Why not plain `threading.Event` as a parameter?** Requires plumbing a new argument through every call site (`LLMWorker` → router → backend → http). Easy to forget on a future code path; breaks silently on refactor.

**Why not plain `ContextVar[bool]`?** Doesn't work. `ThreadPoolExecutor` captures the context at submit time, so the background thread gets its own snapshot. Setting the var in the main thread AFTER submission has no effect on the orphaned thread. **The `ContextVar`'s value MUST be a mutable reference (the Event), not the boolean.** If a future "simplification" changes it to `ContextVar[bool]` with `.set(True)`, it breaks silently. Load-bearing; deserves a comment + a two-thread regression test asserting cross-thread visibility.

**Implementation:**

1. Create `src/maxim/agents/cancellation.py` (new tiny module):
   ```python
   from __future__ import annotations
   import threading
   from contextvars import ContextVar

   # Value MUST be an Event (mutable shared reference), NOT a bool.
   # ThreadPoolExecutor captures the ContextVar at submit time — setting
   # a bool in the parent thread would not propagate to the background
   # thread's snapshot. The Event is shared by reference; both threads
   # see .set() calls. This is load-bearing.
   _cancel_event_var: ContextVar[threading.Event | None] = ContextVar(
       "maxim_cancel_event", default=None
   )

   def current_cancel_event() -> threading.Event | None:
       return _cancel_event_var.get()

   def set_cancel_event(event: threading.Event | None) -> object:
       return _cancel_event_var.set(event)

   def reset_cancel_event(token: object) -> None:
       _cancel_event_var.reset(token)
   ```

2. In [llm_worker.py](../../src/maxim/agents/llm_worker.py) `_call_llm_with_timeout`:
   - Before `executor.submit(...)`, create an `Event` and set it via `set_cancel_event(event)`.
   - When the 60s/300s timeout fires, call `event.set()` BEFORE logging the warning or returning the fallback.
   - `reset_cancel_event(token)` in a `finally`.

3. In [router.py](../../src/maxim/models/language/router.py) `_note_provider_failure` and anywhere else that writes to `_provider_states`:
   - Check `current_cancel_event()` at the top. If set, log `provider_failure_skipped_cancelled` (structured) and return without mutating state.
   - Same check before appending to `_dispatch_attempts`.

4. In [maxim_peer_backend.py](../../src/maxim/models/language/maxim_peer_backend.py) `complete_with_usage` and `_stream_response`:
   - At natural checkpoint boundaries (before `_http.post`, between `iter_lines` chunks), check `current_cancel_event()` and raise `BackendDown(fix_hint="cancelled")` if set. This gives the orphaned thread a clean unwind path instead of waiting for HTTP to error.

5. Add a two-thread regression test in `tests/unit/test_cancellation.py`:
   - Thread A sets the ContextVar to an Event.
   - Thread A submits a function to `ThreadPoolExecutor` that waits on `current_cancel_event()`.
   - Thread A calls `event.set()` from outside the executor.
   - Thread B (the submitted function) must observe `event.is_set() == True`.
   - This test fails immediately if someone ever changes the ContextVar value to `bool`.

### R5 — Validation

- Run the R1 reproducer test. All 5 assertions must pass.
- Run the full fast suite: `python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py`
- Run a real sim with the fix: `maxim --sim scenarios/campaigns/heist_v1.yaml --seed 42`. Verify:
  - No `LLM call timed out after 60.0s` warnings within a single turn
  - No paired `peer_backend_failed` → `dispatch_exhausted` cascades
  - AUT turns either complete normally OR error cleanly with a single typed exception
- Capture a new `MAXIM_BACKEND_TRACE=1 MAXIM_LOG_FILE=/tmp/maxim_trace3.jsonl` run and compare `peer_backend_call` latencies to pre-fix baseline.

### R6 — Pre-merge review round (per the ship-pattern from Plan 3)

Spawn two parallel review Claudes (Executor + Architecture lens). Fold findings into the branch BEFORE opening the PR. The Plan 3 R3 session history is the template.

## Success criteria

**Must-have gates (blocking merge):**

1. Reproducer test passes (lock released, provider state clean).
2. Fast suite green.
3. Real heist_v1 sim: zero `peer_backend_failed` cascades within 3 turns.
4. `_inference_lock.locked() == False` asserted at every LLMWorker return path.
5. One agent-level timeout warning per logically-wedged call, not two-per-turn stacking.

**Nice-to-have:**

1. JSONL trace shows `peer_backend_call` events landing with real `elapsed_ms` instead of hitting the timeout fallback.
2. Phase A sim cost drops significantly (AUT turns actually using their LLM responses instead of shipping `llm_timeout` fallbacks).

## Risks

**High:**
- **The 125s leader-side latency might be a separate bug that reappears under load.** This plan doesn't fix the "why was the leader grinding for 125s" question, only the "why did it look like two stacked 60s timeouts" part. Phase D may surface the same latency under a different failure mode. Mitigation: instrument the AUT prompt assembly in R5 to see if prompts are 8000+ tokens; if yes, open a follow-up plan.

**Medium:**
- **Cancellation-event approach has a race window.** Between `_cancelled.set()` and the router's check, state can still be polluted. Mitigation: keep the cancellation check inside the router's locked section so the orphaned thread sees it atomically.
- **Raising `llm_timeout_s` to 300s on a misbehaving backend turns 60s stalls into 300s stalls.** The fast-failover work in Plan 3 is the mitigation — typed exceptions should fire within seconds, not minutes. If they don't, THAT's a Plan 3 regression and needs a separate fix.

**Low:**
- `loop_controller` and `agent_loop` fallback defaults might have other callers that expect 60s. Mitigation: grep for `timeout_s=60` and `_timeout_s` callers and update as a batch.

## Test protocol

Ran interactively by the author on the Mac peer + RTX 5080 leader setup:

1. **Reproducer test** (offline, fast): `python -m pytest tests/unit/test_llm_worker_cancellation.py -v` — must pass.
2. **Fast suite**: `python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py`
3. **Real sim** (requires leader up): `MAXIM_BACKEND_TRACE=1 MAXIM_LOG_FILE=/tmp/trace_post_fix.jsonl maxim --sim scenarios/campaigns/heist_v1.yaml --seed 42` — cap at 3 turns, Ctrl+C. Validate against R5 checklist.
4. **Phase D dry run**: Once the above are clean, proceed to Phase D stress test in the main LLM path plan.

## Related

- [llm_path_fast_failover.md](llm_path_fast_failover.md) — Plan 3 R2.5 + R2.6 (already shipped)
- [llm_path_operator_visibility.md](llm_path_operator_visibility.md) — Plan 4 (blocked on this)
- [llm_path_refinement.md](llm_path_refinement.md) — meta plan
- [project_llm_path_r3_shipped.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_llm_path_r3_shipped.md) — Plan 3 load-bearing invariants
- Stress test trace data: `/tmp/maxim_trace.jsonl`, `/tmp/maxim_trace2.jsonl` (ephemeral, regenerate as needed)
- Motivating sim: PR #94 merged, session memory at `project_llm_path_r3_shipped.md`
