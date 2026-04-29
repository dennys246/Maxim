# Public API stability — 1.0 contract

This page tracks 1.0 stability commitments for the public Maxim API. The fuller classification (every verb, every result type, every event payload) is being assembled alongside CC2; this page currently documents the **async wrappability** contract that ships with CC10.

---

## Async wrappability (CC10)

The Maxim public verbs in [`api.py`](../../src/maxim/api.py) are **synchronous**. To call them from async code (FastAPI, Pydantic AI, LangGraph, etc.), wrap them with `asyncio.to_thread`:

```python
import asyncio
import maxim

async def handler():
    # Run a sync verb on a worker thread; the event loop stays free.
    session = await asyncio.to_thread(
        maxim.imagine,
        goal="test memory recall",
        max_turns=20,
    )
    return {"session_id": session.session_id}
```

This pattern is **stable in 1.0** for every public verb. Each verb satisfies three constraints:

- **No internal `asyncio` usage that needs a running event loop.** Verbs are pure-sync internally; they don't `await` anything.
- **No required stdin input.** Verbs don't call `input()` on the calling thread. (`imagine()` and `campaign()` spawn a daemon stdin-reader thread for slash-command support — that thread blocks on `input()` but handles `EOFError` gracefully and is not load-bearing for the verb's success.)
- **No CWD assumption** — with three documented exceptions:
  - `maxim.benchmark(suite=...)` accepts a bare suite name like `"cognitive"` and looks for `./scenarios/benchmarks/cognitive.yaml` *relative to the current working directory*. This is a developer-checkout convenience. **From async / pip-install / arbitrary-CWD callers, pass an absolute path** — e.g. `suite="/abs/path/to/cognitive.yaml"`. The verb raises `ConfigurationError` (with the active CWD in the message) when the bare-name lookup fails so the failure mode is obvious.
  - `maxim.imagine(scenario=...)` accepts a YAML path; relative paths resolve against the CWD via `pathlib.Path(scenario)`. Same guidance — pass an absolute path from async or arbitrary-CWD contexts.
  - `maxim.campaign(path=...)` accepts a campaign YAML path; relative paths resolve against the CWD. Same guidance — pass an absolute path from async or arbitrary-CWD contexts.

### Cancellation from async

Cancelling the wrapping `asyncio.Task` does **not** stop the synchronous verb mid-execution — `asyncio.to_thread` cannot interrupt a running thread. Treat the verb as fire-and-forget once it starts. For long-running sims, use the `max_turns=` parameter as the upper bound, or run the verb in a subprocess that you can terminate.

For finer-grained cancellation **inside** a verb (e.g. aborting an in-flight tool's HTTP request), the Tool ABC exposes `Tool.cancel()` (CC11) — the agent loop can call it to ask a tool to abort cleanly. Built-in tools that override the default no-op:

| Tool | Cancel semantics |
|---|---|
| `HttpFetchTool` | Sets a flag; the next `execute()` safety-check returns a "cancelled" error before issuing the network call. Does not interrupt a thread already inside the underlying HTTP request — the `TimeoutPolicy` is the load-bearing hard interruption. |
| `InternetSearchTool` | Same shape as `HttpFetchTool`. |

Other tools inherit the default no-op `cancel()` and require their `timeout` setting to bound runtime.

---

## Adapter contract (CC8)

`PerceptSource` ([`simulation/sources.py`](../../src/maxim/simulation/sources.py)) and `ActionSink` ([`simulation/sinks.py`](../../src/maxim/simulation/sinks.py)) are the two protocols an external integration implements to drive the agent loop with non-sim percepts and capture its actions. The protocols are deliberately minimal — see the module docstrings for the full contract:

- `PerceptSource`: `name`, `next_percept()`, `is_exhausted()`, `capabilities`. Optional duck-typed extensions: `has_pending()`, `advance_step()`.
- `ActionSink`: `record(action)`, `actions` property.

Neither protocol assumes the simulation orchestrator, narrative phases, conversational turns, or the tool registry. A future Mineflayer / Minecraft adapter (or any other game/world client) implementing these can drive `run_agentic_loop` directly.

The `is_sim_mode` flag inside `runtime/sim_adapter.py` is more accurately read as "an external adapter is driving percepts" — it gates behaviors that fire when a `PerceptSource` is supplied, not behaviors specific to the simulation orchestrator. The field is retained as `is_sim_mode` for back-compat.

**Caveat for long-running adapters:** when `is_sim_mode=True`, the agent loop ends the session via `MemoryHub.on_session_end_lightweight()` rather than the full `on_session_end()` (the full path runs blocking sleep/replay consolidation). This is correct for short-lived sim runs but may surprise an external adapter (e.g. a Minecraft session running for hours) that expects full consolidation at session end. Adapter integrations that need the full path should arrange for `end_bio_session(is_sim_mode=False, ...)` to be called directly. This trade-off is preserved in 1.0 to avoid changing existing sim behavior.

---

## Stability rules

1. **Adding** a new keyword-only argument with a default to a stable verb is non-breaking and may happen in any minor release.
2. **Removing** or **renaming** a stable symbol or argument is a breaking change and only happens at a major-version bump (1.x → 2.0).
3. **Behavior** of stable verbs may evolve (smarter defaults, better error messages, additional internal validation) without a major-version bump as long as the documented input/output contract holds.
4. **Adding** a new method to the `Tool` ABC with a default body is non-breaking (CC11 `cancel()` is the canonical example). Adding an `@abstractmethod` is a breaking change for third-party tool authors.

The fuller classification of stable vs experimental verbs, result types, event payloads, env vars, and CLI flags will be added here as part of CC2.
