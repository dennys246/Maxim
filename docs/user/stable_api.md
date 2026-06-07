# Public API stability — 1.0 contract

This page lists what is **stable** in pymaxim 1.0 and what is **experimental**. The contract is:

- **Stable** symbols: removal or rename is a breaking change. Argument names + types do not shift without a major-version bump. New keyword-only arguments with defaults are allowed.
- **Experimental** symbols: marked in the symbol's docstring header. Shape may shift in 1.x without a major-version bump. Use at your own risk; pin to a version if you build hard against them.

**Anything not listed on this page is not part of the public 1.0 contract** — it is fair game for changes in any future release. If you want a non-listed symbol elevated to stable, open an issue.

---

## Stable verbs (`maxim.*`)

| Symbol | Stable | Notes |
|---|---|---|
| `maxim.configure(...)` | ✅ | Logging + display settings. Keyword-only. |
| `maxim.run(...)` | ✅ | Agentic cycle entry point. |
| `maxim.imagine(...) -> Session` | ✅ | Generative simulation. Returns persistent Session. |
| `maxim.connect(robot_type, ...) -> RobotController` | ✅ | Robot registry connection. |
| `maxim.diagnose(...) -> DiagnosticReport` | ✅ | Local + peer diagnostics. |
| `maxim.observe(subsystem, ...)` | ✅ | Cognitive subsystem inspection. |
| `maxim.introspect(...)` | ✅ | Alias for `observe`. |
| `maxim.campaign(path, ...) -> CampaignResult` | ✅ | DM campaign runner. |
| `maxim.benchmark(models, ...) -> BenchmarkResult` | ✅ | Multi-model benchmark. |
| `maxim.list_models() -> dict[str, list[ModelInfo]]` | ✅ | Profile discovery. |
| `maxim.download_model(name) -> bool` | ✅ | Local LLM download. |
| `maxim.delete_model(name) -> bool` | ✅ | Local LLM cleanup. |
| `maxim.register_tool(tool)` | ✅ | Custom tool registration. |
| `maxim.tool` (decorator) | ✅ | Function-as-tool decorator. |
| `maxim.get_version_info() -> dict` | ✅ | Version + git hash. |
| `maxim.__version__` | ✅ | Package version string. |

## Experimental verbs

| Symbol | Stable | Notes |
|---|---|---|
| `maxim.research(...) -> ResearchResult` | ⚠️ Experimental | Research orchestrator surface still evolving. Prompt templates, paper structure, and reviewer logic may change. |
| `maxim.on(event_name, callback) -> EventHandle` | ⚠️ Experimental | Event names + payload fields may grow. Subscription mechanism may evolve to support filters or async callbacks. |
| `maxim.register_persona(...)` | ⛔ Deprecated in 0.9 — removed in 1.1 | Emits `DeprecationWarning` in 0.9 / 1.0; raises in 1.1. Persona system is being replaced by `--sim-mode` (orchestrator flow-shape) plus bio-emergent disposition mechanics. See [`docs/plans/persona_cleanup_and_mode_transition.md`](../plans/persona_cleanup_and_mode_transition.md). |

---

## Stable result types (importable from `maxim.*`)

| Symbol | Stable | Notes |
|---|---|---|
| `DiagnosticReport` | ✅ | Returned by `diagnose()`. |
| `CampaignResult` | ✅ | Returned by `campaign()`. |
| `BenchmarkResult` | ✅ | Returned by `benchmark()`. |
| `ModelInfo` | ✅ | Element of `list_models()` output. |
| `Session` | ✅ | Returned by `imagine()`; load via `maxim.load.session(...)`. |
| `Report` | ✅ | Final-form simulation report. |
| `Entity` | ✅ | SEM entity composition root. |
| `RobotController` | ✅ | Returned by `connect()`. |

## Experimental result + event types

| Symbol | Stable | Notes |
|---|---|---|
| `ResearchResult` | ⚠️ Experimental | Returned by `research()`. Field set may grow. |
| `EventHandle` | ⚠️ Experimental | Returned by `on()`. |
| `ToolCallEvent` | ⚠️ Experimental | Payload type for `on("tool_call", ...)`. |
| `MemoryCaptureEvent` | ⚠️ Experimental | Payload type for `on("memory_capture", ...)`. |
| `PainSignalEvent` | ⚠️ Experimental | Payload type for `on("pain_signal", ...)`. |
| `PromptEvent` | ⚠️ Experimental | Payload type for `on("prompt", ...)`. |

---

## Stable error hierarchy

All exceptions are importable from `maxim.*`:

| Symbol | Stable | Notes |
|---|---|---|
| `MaximError` | ✅ | Base class for catch-all. |
| `ConfigurationError` | ✅ | Config / model unavailable / SDK missing. |
| `MaximConnectionError` | ✅ | Robot / peer connect failures. |
| `ModelError` | ✅ | LLM-level errors (base). |
| `ModelLoadError` | ✅ | Backend / GGUF load failures. |
| `ToolExecutionError` | ✅ | Tool dispatch / execution failures. |
| `ToolNotFoundError` | ✅ | Tool registry lookup failures. |
| `MaximMemoryError` | ✅ | Persistence / load failures. |
| `PlanningError` | ✅ | Goal / plan failures. |
| `HardwareError` | ✅ | Robot hardware faults. |
| `MaximRuntimeError` | ✅ | Generic runtime fallback. |

---

## Stable namespaces

### `maxim.create.*` — factory namespace

| Symbol | Stable | Notes |
|---|---|---|
| `create.hippocampus(...)` | ✅ | |
| `create.nac(...)` | ✅ | |
| `create.atl(...)` | ✅ | |
| `create.scn()` | ✅ | |
| `create.angular_gyrus(...)` | ✅ | |
| `create.agent(name, ...) -> AgentInstance` | ✅ | |
| `create.pool(...) -> AgentPool` | ✅ | |
| `create.entity(template_ref, ...) -> Entity` | ✅ | |
| `create.embodiment(root_entity, ...) -> Embodiment` | ✅ | |
| `create.templates() -> dict[str, list[str]]` | ✅ | |
| `create.router(model) -> LLMRouter` | ✅ | |

### `maxim.load.*` — deserialization namespace

| Symbol | Stable | Notes |
|---|---|---|
| `load.hippocampus(path) -> Hippocampus` | ✅ | |
| `load.nac(path) -> NAc` | ✅ | |
| `load.atl(path) -> ATL` | ✅ | |
| `load.session(session_id) -> Session` | ✅ | |
| `load.sessions(*, limit=20) -> list[Session]` | ✅ | |
| `load.agent(name, *, base_dir=None) -> AgentInstance` | ✅ | |
| `load.entity(path) -> Entity` | ✅ | |

`AgentInstance`, `AgentPool`, `LLMRouter`, `Hippocampus`, `NAc`, `ATL`, and `Embodiment` are returned by these factory/loader functions; their **public** methods (the ones documented in [python-api.md](python-api.md)) are stable. Internal helper methods (leading underscore) are not.

---

## Token telemetry contract (CC12)

Per-call token telemetry is exposed under these field names — frozen at 1.0:

| Field | Meaning |
|---|---|
| `input_tokens` | Total prompt tokens (cached + uncached). |
| `output_tokens` | Generated tokens. |
| `cached_tokens` | Cached portion of the input. Read from prompt cache, charged at the cached rate (or free, depending on provider). |

These names appear in:

- `LLMResponse.input_tokens`, `LLMResponse.output_tokens`, `LLMResponse.cached_tokens` (property alias for `cached_input_tokens`)
- The `usage` dict returned alongside `LLMRouter.generate(...)` results
- JSONL log records emitted under `MAXIM_LOG_FILE` for `peer_backend_call`, `peer_stream_complete`, and the leader proxy's per-request log entry
- `CostTracker.get_session_tokens()`

Legacy field names (`cached_input_tokens`, `uncached_input_tokens`, `prompt_tokens`, `completion_tokens`) are kept as **permanent legacy aliases** — internal cost-calculation paths still reference `cached_input_tokens`/`uncached_input_tokens`, and the peer wire format mirrors OpenAI's `prompt_tokens`/`completion_tokens`. Removing any of these is a major-version-bump change. **External callers should prefer the standard names** (`input_tokens`, `output_tokens`, `cached_tokens`).

See [configuration.md](configuration.md) for the full token telemetry surface table.

---

## Stable env var + CLI flag contracts

See [configuration.md](configuration.md) for the public env-var classification and CLI flag `[experimental]` tags.

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
    return {"session_id": session.id}
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
5. **Experimental** symbols may shift in a minor release (1.0 → 1.1). When they stabilize, the experimental note is removed and the row in this table moves to the stable section.
6. **Internal** symbols (anything starting with `_`, anything not on this page) are not part of the contract regardless of how someone is using them today.
