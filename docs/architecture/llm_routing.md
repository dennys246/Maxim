# LLM Routing Architecture

**Status:** Draft — written ahead of [llm_path_refinement.md](../plans/llm_path_refinement.md) implementation as a design reference
**Last updated:** 2026-04-12
**Target state:** post-0.4 LLM path refinement (four sub-plans complete)
**Maintenance contract:** this doc is the **single authoritative reference** for how LLM requests flow through Maxim. Any refinement of the routing path must update this doc in the same PR. If the doc and code disagree, the doc is wrong and must be fixed.

## How to read this doc — Present vs Target markers

**This document describes the POST-0.4 target state.** About 50% of the named artifacts (`_MaximPeerBackend`, `BACKEND_CLASSES`, `BackendError` taxonomy, two-stage probe, per-outcome cache TTLs, `cluster_key`, `runtime/role.py`) do not yet exist in the codebase. They will exist once Plans 2-4 ship.

**Plan 1 R1 has SHIPPED** (PRs #88, #90): `maxim/utils/http.py`, `RequestContext`, `contextvars` propagation, header sanitization, typed `HTTPError` hierarchy, endpoint registry, JSONL dual-format logging, and `http_*` metrics all exist today on main. Layer 7 below is [Present].

To keep orientation clear, major subsections are tagged:
- **[Present]** — describes code that exists today on main
- **[Target]** — describes code that will exist after the specified sub-plan ships
- **[Mixed]** — describes a layer where some features exist and others don't; per-feature markers are inline

**If you're reading this doc to understand the current codebase state, treat anything unmarked or marked `[Target]` as a future specification, not a description of today.** Cross-reference against `src/maxim/` before acting on any claim.

**If you're reading this doc to understand where we're going, treat it as the contract that Plans 1-4 will satisfy.** The plans cite this doc; this doc should not drift from the plans as they land.

## Purpose

When a caller (agent loop, simulation orchestrator, API verb) asks Maxim to run inference, the request flows through eight layers before the bytes leave the process. Each layer has a specific responsibility. Each layer can refuse, retry, rewrite, or reclassify. When something goes wrong, knowing which layer owned the decision is the difference between a ten-minute fix and a two-hour rabbit hole.

This document walks the full path top-to-bottom, names each layer's single responsibility, and documents the contracts between layers. It is the first doc to read when working on anything LLM-adjacent.

## Request flow overview

```
Caller
  │
  ▼
RequestContext (multi-agent context bundle: agent_id, session_id, request_id, lane)
  │
  ▼
FunctionRouter                 ← "which tier for this function?"
  │
  ▼
LaneBackendManager             ← "which backend for this tier? cache it."
  │
  ▼
LLMRouter                      ← "which provider? fall over on failure."
  │
  ▼
Backend dispatch               ← BACKEND_CLASSES table: maxim_peer | openai | llama | transformers
  │
  ▼
_MaximPeerBackend              ← "one HTTP call, typed errors, no retry"
      (or _OpenAIBackend for cloud providers, _LlamaCppBackend for local, etc.)
  │
  ▼
utils/http.py                  ← "connection pool, X-Maxim-* headers, metrics"
  │
  ▼
HTTPS wire → Cloudflare tunnel → Leader proxy → llama-cpp-server
```

Each arrow crosses a responsibility boundary. Each boundary has a contract documented below.

## Layer 1: Caller

**Who:** `agents/exec_agent.py`, `agents/llm_worker.py`, `simulation/orchestrator.py`, `api.py` verbs, DM runtime, etc.

**Responsibility:** express intent. "I need a completion for this prompt with this function name (e.g. `agent_inference`, `memory_summarize`)."

**Contract upward:** call a `FunctionRouter` method with a `RequestContext`.

**Contract downward:** *none* — the caller doesn't know about tiers, lanes, providers, backends. It names a function, provides a context, passes prompt + parameters.

**RequestContext is the multi-agent contract:**

```python
@dataclass(frozen=True)
class RequestContext:
    request_id: str              # auto-generated if caller doesn't provide
    agent_id: str | None = None  # canonical key — "agent" is legacy alias
    session_id: str | None = None
    lane: str | None = None      # set by FunctionRouter, not caller
    parent_request_id: str | None = None  # for fan-out tracing
```

**Why this matters:** the context flows through every subsequent layer via a `contextvars.ContextVar`. The logging formatter pulls from the contextvar so every log line related to this request automatically carries `agent_id`, `session_id`, `request_id`. Callers don't have to remember to pass them to log calls.

**If you add a new caller:** construct a `RequestContext` at the entry point. Don't pass `None` unless you're making a truly agent-agnostic call (probe, health check).

## Layer 2: FunctionRouter

**Who:** `src/maxim/runtime/function_router.py::FunctionRouter`

**Responsibility:** map function name → tier. "An `agent_inference` call goes to the `large` tier. A `memory_summarize` call goes to `small`."

**Why it exists:** different work needs different compute. Summarizing memory fragments is cheap (smollm). Deep reasoning wants a 14B model. Without a function router, every call would pay for the biggest available lane.

**Contract upward:** receives `(function_name, request_context, prompt_kwargs)` from the caller.

**Contract downward:** looks up the tier in its registry (defaults in `DEFAULT_FUNCTIONS`, overrides from `llm.json::functions`), asks `LaneBackendManager` for that tier's backend, delegates the call.

**Tier fallback:** if a tier's backend is unavailable (no local GPU + no remote URL + no cloud fallback), the function router walks a fallback chain: large → medium → small → None. This happens during lane construction, not per-request.

**What it does NOT do:**
- Does NOT pick providers (that's `LLMRouter`'s job)
- Does NOT retry (that's `LLMRouter`'s job)
- Does NOT know about HTTP (that's layers below)
- Does NOT know about agents or sessions (it just propagates the `RequestContext`)

## Layer 3: LaneBackendManager

**Who:** `src/maxim/runtime/lane_backends.py::LaneBackendManager`

**Responsibility:** own the per-lane backend cache. "The large tier's backend is an instance of `LLMRouter` wrapping either a local `_LlamaCppBackend` or a remote `_MaximPeerBackend`, constructed lazily on first request."

**Why it exists:** backends are expensive to construct (model load = seconds; HTTP client init = less, but still). Lanes share construction across agents. The manager also enforces the `MAXIM_MAX_CONCURRENT_BACKENDS` gate and the `MAXIM_MAX_CLOUD_LANES` gate.

**Contract upward:** `get_backend(lane: str) -> Backend | None`.

**Contract downward:** calls `_build_remote_backend` or `_build_local_backend` on first access, caches the result. Backends are keyed by lane name, not by `agent_id`.

**Construction inputs:**
- `LaneConfig` from `detect_tiers()` (hardware-driven defaults)
- Environment overrides from `apply_lane_env_overrides()`
- Cloud CLI overrides from `_apply_cloud_cli_overrides()`
- Local CLI overrides from `_apply_local_llm_override()`
- Peer config auto-load from `peer/config.py::apply_peer_config_to_env()`
- Remote URL probe via `_validate_remote_urls()`
- Auto-spawn llama-cpp-server via `_maybe_auto_spawn_server()`

**The peer-vs-cloud classification** (`_classify` method) decides whether a lane with a `remote_url` is "self-hosted" (private IP, peer tunnel) or "cloud" (public provider). Cloud lanes are gated by `MAXIM_MAX_CLOUD_LANES`; self-hosted are not.

**Concurrency:** `LaneBackendManager._lock` serializes **both backend construction and teardown**. Specifically: `_build_backend` holds the lock during lazy first-construction, and `unload_all` holds the lock while releasing cached backends ([lane_backends.py:461-472](../../src/maxim/runtime/lane_backends.py#L461-L472)). Runtime inference calls do NOT hold this lock — only cold-start construction and shutdown. A future refactor that adds any hot-path holder of `_lock` would regress latency and should be rejected.

## Layer 4: LLMRouter

**Who:** `src/maxim/models/language/router.py::LLMRouter`

**Responsibility:** sequential provider fallback with backoff. "Try provider A. If it fails, try provider B. If that fails, try provider C. If all fail, return empty and let the caller handle it."

**Why it exists:** the same lane can have multiple providers — a cloud fallback for a local primary, or multiple self-hosted peers (per `llm_path_multi_peer_dispatch.md` if shipped). The router iterates `provider_priority` in order, catching failures and advancing.

**Contract upward:** `complete_text(system, user, ...)` or `complete_with_usage(...)`.

**Contract downward:** `_try_provider(provider_key, ...)` dispatches to the backend for that provider. On failure (empty content OR typed exception), the router records the failure in `ProviderState`, applies backoff, and moves to the next provider.

**Empty content → failure is non-obvious from the loop shape.** For reference, the current code path ([router.py:648-654](../../src/maxim/models/language/router.py#L648-L654)): `_try_provider` catches exceptions inside a `try:` block, then unconditionally checks `if text:` — if the string is empty, it skips the success return and falls through to `self._note_provider_failure(provider_key, "call_failed")`. This is how the existing `_OpenAIBackend` (which swallows errors and returns empty `LLMResponse` instead of raising) correctly triggers fallback. Plan 3's `_MaximPeerBackend` takes the cleaner typed-exception path, but the legacy empty-content signal stays intact so `_OpenAIBackend` continues to work for cloud providers.

**Key datastructures:**
- `self._providers: dict[str, dict]` — provider config, keyed by provider_key
- `self._backends: dict[str, Backend]` — backend instances, lazy
- `self._provider_states: dict[str, ProviderState]` — backoff + consecutive_errors per provider
- `self._inference_lock: threading.Lock` — **serializes ALL calls within this router instance** (one call per lane at a time)

**The concurrency wall:** `_inference_lock` means two agents hitting the same lane queue behind each other. Under Plan 3's `_MaximPeerBackend`, the typical hold time is ~100ms-5s (actual call duration). Under the pre-plan `_OpenAIBackend`, the worst case was ~50s (gateway retries) + ~1.5s (normal retries) = ~52s total (the retry loop). Fixing this bottleneck for truly concurrent agents is tracked in [llm_path_async_router.md (deferred)](../plans/deferred/llm_path_async_router.md).

**Per-provider backoff** (in `ProviderState`):
- `consecutive_errors` increments on each failure
- `backoff_until` is set to `now + min(60.0, 1.0 * (2 ** max(consecutive_errors - 1, 0)))` on failure — sequence is 1s, 2s, 4s, 8s, 16s, 32s, 60s (capped). See [router.py:437](../../src/maxim/models/language/router.py#L437) for the live formula.
- `_candidate_providers` skips any provider whose `backoff_until > now`
- Successful call resets `ProviderState` to healthy defaults: `consecutive_errors = 0`, `backoff_until = 0.0`, `last_error = ""`, and `last_success = time.time()` (see [router.py::_note_provider_success](../../src/maxim/models/language/router.py#L419)). Any future refactor adding fields to `ProviderState` must update `_note_provider_success` to clear them on the success path — or document why not.

**Typed exception handling** (post Plan 3):
- `BackendOverloaded` → short backoff, try next
- `BackendDown` → exponential backoff
- `BackendTimeout` → exponential like Down
- `BackendAuthFailed` → **300s hard backoff**
- `BackendModelMissing` → 60s backoff
- `BackendInferenceBroken` → 15s backoff (**should match** the probe cache `inference_broken` TTL — these two constants are linked via `INFERENCE_BROKEN_BACKOFF_S` in `models/language/types.py` (added in Plan 2 R2b) to prevent drift)
- `BackendError` (generic) → normal failure
- `Exception` (safety net) → increments `backend_unclassified_errors_total`; flag to investigate

## Layer 5: Backend dispatch — `BACKEND_CLASSES` table [Target — Plan 3 R2.5]

*Current [Present] state: `lane_backends.py::_build_remote_backend` constructs `_OpenAIBackend` unconditionally for any remote URL. There is no dispatch table. Plan 3 R2.5 introduces `BACKEND_CLASSES` as a module-level dict + a `"backend_class"` config field.*

**Who:** `src/maxim/runtime/lane_backends.py` — a module-level dict

**Responsibility:** pick the right backend class for a provider config. "If the provider has `backend_class: maxim_peer`, instantiate `_MaximPeerBackend`. If `backend_class: openai`, instantiate `_OpenAIBackend`."

```python
BACKEND_CLASSES: dict[str, type] = {
    "maxim_peer": _MaximPeerBackend,  # self-hosted peers (default for self-hosted URLs)
    "openai": _OpenAIBackend,          # cloud providers (Anthropic, OpenAI, Groq, etc.)
    "llama_cpp": _LlamaCppBackend,     # local llama.cpp
    "transformers": _PyTorchTransformersBackend,  # PyTorch/HuggingFace
}
```

**Why two OpenAI-compatible backends:** `_OpenAIBackend` has an internal retry loop + cost tracking + PII redaction — correct for cloud providers, wrong for self-hosted peers. `_MaximPeerBackend` is purpose-built for peer tunnels: single HTTP call, typed exceptions, no retry, no cost tracking. See [plans/llm_path_refinement.md](../plans/llm_path_refinement.md) for the full justification of why the backends are split.

**The classification** (self-hosted vs cloud) happens earlier in `lane_backends._classify`. By the time we reach the `BACKEND_CLASSES` lookup, `backend_class` is already set on the provider config.

## Layer 6: The backends [Mixed]

### `_MaximPeerBackend` — self-hosted peer tunnels [Target — Plan 3 R2.5]

**Does not exist yet.** The file `src/maxim/models/language/maxim_peer_backend.py` will be created in Plan 3 R2.5. This entire subsection describes the target design; cross-reference against Plan 3's spec for the exact shape. Until Plan 3 ships, self-hosted peer URLs flow through `_OpenAIBackend` (with its ~50s internal retry loop — the problem Plan 3 fixes).

**Who:** `src/maxim/models/language/maxim_peer_backend.py` (created in Plan 3)

**Responsibility:** make exactly one HTTP call per `complete_with_usage()`. Raise a typed exception on failure. Do not retry.

**Why this shape:** the router does fallback. Retries at the backend level defeat fallback — they hold `_inference_lock` for minutes while the router could be trying a different provider in milliseconds.

**What it does:**
- Build the OpenAI-compatible chat completions payload
- Call `http.post("peer-{provider_key}", path="/chat/completions", json=payload, context=context)`
- Classify the HTTP response: 200 → parse → return `LLMResponse`; 429 → `BackendOverloaded`; 5xx → `BackendDown`; etc.
- Support streaming via `http.stream_post(...)` — still one HTTP call, just with streaming body

**What it does NOT do:**
- NO retry loop
- NO cost tracking
- NO PII redaction (self-infrastructure)
- NO OpenAI SDK dependency (direct httpx via `utils/http.py`)

**Streaming:** one HTTP call, streaming body via httpx. Collects chunks into an `LLMResponse`. Mid-stream errors raise `BackendDown` and the router tries the next provider.

**Health check:** the backend exposes `health_check() -> ProbeResult` implementing the two-stage probe (Plan 2 R2c). This is THE canonical probe implementation — `probe_llm_server` and `llm_server_responding_at` are deleted in Plan 3 R2.6 in favor of this method.

### `_OpenAIBackend` — cloud providers [Present, unchanged by refinement]

**Who:** `src/maxim/models/language/openai_backend.py` (existing, unchanged by the refinement)

**Responsibility:** talk to cloud LLM providers (Anthropic, OpenAI, Groq, Together, etc.) via the OpenAI SDK.

**Why it keeps the retry loop:** cloud providers have transient rate limits and gateway errors that self-heal. Retrying on the same provider is correct. Cost tracking matters (billing). PII redaction matters (compliance). None of these apply to self-hosted peers, which is why `_MaximPeerBackend` is a separate class.

**Invariant:** `_OpenAIBackend` is NEVER used for `backend_class: maxim_peer` providers. The two backends are strictly separated by `BACKEND_CLASSES`.

### `_LlamaCppBackend`, `_PyTorchTransformersBackend`

In-process local backends. Not relevant to the peer routing path. Documented in `reference.md`.

## Layer 7: `maxim/utils/http.py` — unified HTTP client [Present — shipped in Plan 1 R1]

**Shipped in PRs #88 + #90 (2026-04-12).** Before R1, HTTP calls were scattered across ~11 files using raw `urllib.request` with inconsistent headers — the fragmentation that caused the 2026-04-12 Cloudflare User-Agent incident (commit `8b52cbd`). All eleven call sites are now routed through `maxim/utils/http.py`; the CI grep `grep -r "urllib.request.urlopen" src/maxim/` returns zero matches.

**Who:** [src/maxim/utils/http.py](../../src/maxim/utils/http.py)

**Responsibility:** single place where outbound HTTP happens in the codebase. One endpoint registry, one connection pool per endpoint, one header contract, one metrics surface.

**Key abstractions (as shipped):**

```python
@dataclass(frozen=True)
class HTTPEndpoint:
    name: str                          # "leader", "_external", etc.
    base_url: str | None
    default_headers: Mapping[str, str] # set ONCE at registration
    auth_provider: Callable[[], str | None] | None  # late-bound bearer token
    timeouts: TimeoutPolicy            # connect/read/total
    max_pool_connections: int = DEFAULT_POOL_PER_ENDPOINT  # 10
    internal: bool = True              # gates X-Maxim-* propagation

def register_endpoint(endpoint: HTTPEndpoint) -> None: ...

# Registry-based surface (internal endpoints, referenced by name):
def get(endpoint_name, path, *, context=None, **kwargs) -> Response: ...
def post(endpoint_name, path, json=None, *, context=None, **kwargs) -> Response: ...

# Ad-hoc URL surface (user tools, HF downloads, peer-cli remote admin):
def fetch_url(url, *, method="GET", context=None, **kwargs) -> Response: ...
def download_to_file(url, dest_path, *, progress_hook=None, ...) -> int: ...

# Escape hatch for leader_proxy reverse-proxy forwarding only:
def raw_proxy_forward(url, method, *, headers, body, timeout) -> Response: ...
```

**Two registered endpoints today:**

- `"leader"` — registered in `peer/config.apply_peer_config_to_env` when a peer config loads. Late-bound auth via closure over `MAXIM_LANE_LARGE_REMOTE_API_KEY` so cluster-key rotation doesn't touch call sites. `internal=True`.
- `"_external"` — lazily registered on first use by `fetch_url`/`download_to_file`. Used for HuggingFace model downloads, tool-surface HTTP (`http_fetch`, `internet_search`), and peer-cli remote admin calls (`peer update/restart/llm/version/logs/install/deps`). `internal=False` — X-Maxim-* headers do NOT leak to third parties.

**Automatic header propagation** from `RequestContext` (only for `internal=True` endpoints):
- `X-Maxim-Request-Id: <request_id>` (always)
- `X-Maxim-Agent-Id: <agent_id>` (when set)
- `X-Maxim-Session-Id: <session_id>` (when set)
- `X-Maxim-Lane: <lane>` (when set)
- `X-Maxim-Parent-Request-Id: <parent_request_id>` (when set, for fan-out)
- `X-Maxim-Protocol-Version: 1` (always)

These headers are the **wire protocol between nodes**. Changing a header name is a breaking protocol change requiring a version bump. Adding new headers is non-breaking.

**Input sanitization at boundary:** header values pass through `_sanitize_header_value` which rejects control chars, CR/LF, non-ASCII bytes, and lengths > `MAX_HEADER_VALUE_LEN` (256) — a module-level named constant, not an inline magic number. The limit is enforced in `maxim/utils/http.py::_sanitize_header_value`; rejections increment `http_header_rejected_total{field}` and raise `HTTPClientError` with a `fix_hint` identifying the offending field. Prevents log injection from user-controlled values.

**Typed HTTP errors (all shipped):**
- `HTTPTimeout` — caller (Plan 3) maps to `BackendTimeout`
- `HTTPConnectionError` — caller maps to `BackendDown` (wraps DNS, TLS, refused)
- `HTTPServerError` (5xx) — caller maps to `BackendDown`
- `HTTPAuthError` (401/403) — caller maps to `BackendAuthFailed`
- `HTTPRateLimited` (429) — caller maps to `BackendOverloaded` with `retry_after_s` + `suggested_peer` from response headers
- `HTTPClientError` (other 4xx) — caller maps to `BackendError` or `BackendModelMissing` (404)

The router-side backend taxonomy is still Plan 3 work; Plan 1 just ships the HTTP-side classes so callers (probes, doctor, tools, peer-cli) can branch on them.

**Connection pool per endpoint:** `DEFAULT_POOL_PER_ENDPOINT = 10`. Under multi-agent load, the pool is the concurrency ceiling for outbound calls to that endpoint. Metric `http_pool_exhausted_total{endpoint}` fires when a call waits for a pool slot — non-zero means raise the constant.

**Dual-format logging:** when `MAXIM_LOG_FILE=/path/to/log.jsonl` is set in the environment, `utils/logging.configure_logging()` attaches a JSONL `FileHandler` backed by `StructuredFormatter` from `maxim.utils.structured_logging`. stdout stays human-readable; the JSONL file captures structured events (`event=http_request`, `event=http_request_failed`, `event=startup_phase`) and every regular `logger.info/warning` call as `event=log` fallbacks.

**Metrics exposed via `http.metrics_snapshot()`:**
- `requests_total{endpoint/status}` — counter
- `latency{endpoint}` — `{count, p50_ms, p99_ms}` reservoir (last 200 samples)
- `pool_exhausted_total{endpoint}` — counter
- `header_rejected_total{field}` — counter
- `startup_phases{phase}` — duration_ms map, populated via `http.record_startup_phase(phase, ms)`

Bounded cardinality: no `agent_id` labels on metrics. Per-agent debugging goes through JSONL logs with `jq 'select(.agent_id=="npc-X")'`.

## Layer 8: Wire transport

Outside Maxim. HTTPS to the leader's Cloudflare tunnel (or direct LAN URL for private-IP peers). The leader proxy (`runtime/leader_proxy.py`) forwards to the upstream llama-cpp-server.

**Proxy forwarding bypasses the HTTP client safety net — intentionally.** `leader_proxy._proxy_request` uses `http.raw_proxy_forward(url, method, headers=..., body=..., timeout=...)` instead of the regular `http.post` / `http.fetch_url` surface. The raw variant bypasses:

- Header sanitization (`_sanitize_header_value`) — because peers send arbitrary headers that must be forwarded verbatim to upstream
- `X-Maxim-*` injection — the proxy must preserve the peer's original request_id / agent_id / session_id, not overwrite them with its own
- `auth_provider` late-binding — the proxy forwards whatever `Authorization` header the peer sent; it's not the auth-injection point

`raw_proxy_forward` has an intentionally scary docstring: "Do NOT use this for anything else — it bypasses the registry's safety net." It reuses the shared `_external` httpx client for connection pooling, so the hot path still benefits from keepalive. The `Content-Length` header is stripped from forwarded headers because httpx sets it automatically from the body; duplicating it would cause "Transfer-Encoding: chunked + Content-Length" double-framing errors on strict HTTP servers (a latent bug that existed pre-R1 but never fired because urllib quietly ignored the duplicate).

**What crosses this boundary** — see [data sovereignty](#data-sovereignty-what-crosses-node-boundaries).

## Error taxonomy [Target — Plan 2 R2b]

**Does not exist yet.** The `BackendError` hierarchy is introduced in Plan 2 R2b. Until then, the router uses string-matching on exception messages (`"429" in str(e).lower()`) which has a documented fragility against Cloudflare HTML bodies and wrapped exceptions. The section below describes the post-Plan-2 design.

The `BackendError` hierarchy is the router's language for classifying failures. Every class has a `.fix_hint` attribute so log lines are actionable without the operator needing to look up exception codes.

```
BackendError
├── BackendOverloaded          (429 from peer, retry elsewhere)
├── BackendDown                (connection refused, 5xx, gateway error)
├── BackendTimeout             (exceeded TimeoutPolicy)
├── BackendAuthFailed          (401/403 — 300s hard backoff)
├── BackendModelMissing        (404 — operator needs to install the model)
└── BackendInferenceBroken     (stage-2 probe failed — 15s short backoff)
```

**Per-class backoff policy** (applied in `LLMRouter._try_provider`):

| Exception | Backoff | Rationale |
|---|---|---|
| `BackendOverloaded` | Honor `retry_after_s` if set, else short | Leader is alive, just busy — retry fast |
| `BackendDown` | Exponential: 1s, 2s, 4s, 8s, 16s, 32s, 60s (capped) | Network / peer dead — don't hammer |
| `BackendTimeout` | Exponential like Down | Symptom of slow peer or network |
| `BackendAuthFailed` | **300s hard backoff** | Doesn't self-heal — operator must rotate key |
| `BackendModelMissing` | 60s | Operator must install — waste no retries |
| `BackendInferenceBroken` | 15s | Might be mid-load; matches probe cache TTL |
| Generic `BackendError` | Normal router default | Unknown — try next anyway |
| Safety net (`Exception`) | Normal default + metric | Missing taxonomy entry — file a bug |

**Contract:** any new backend class that raises typed exceptions MUST use these same classes. Backends cannot invent new exception types without the router knowing about them (or they hit the safety net and the `backend_unclassified_errors_total` counter fires).

## Multi-agent context flow [Present — shipped in Plan 1 R1]

**Shipped.** `RequestContext` + `contextvars.ContextVar` propagation exist today in `maxim/utils/http.py`. A grep for `ContextVar` in `src/maxim/` now returns one match, in `utils/http.py`. The `_invoke_backend` path still takes `request_context: dict[str, Any]` as a regular kwarg (legacy shim — Plan 2 R2b's `_normalize_request_context` will bridge the dict to `RequestContext`).

Maxim runs multiple agents concurrently under one user's API key (AgentPool, NPC campaigns). Every LLM call carries `agent_id`, `session_id`, `request_id` through every layer.

**Propagation mechanism:** `contextvars.ContextVar` — Python's stdlib mechanism for request-scoped context. Set once at the caller (Layer 1), read at every layer down to the HTTP request body. The HTTP client reads the contextvar automatically in `_build_headers()`, so outbound requests carry these fields even if the call site didn't pass them explicitly.

**Why contextvars instead of threading the context through every function signature:** passing through function signatures means every intermediate layer has to know about context. That pollutes function signatures and breaks when callers forget to pass it. Contextvars are transparent: set at the boundary, read at the leaves.

**Internal vs external endpoints — data sovereignty boundary.** `HTTPEndpoint.internal: bool` gates whether `X-Maxim-*` headers propagate. This is a load-bearing design decision that diverged from the initial plan:

| Endpoint | `internal` | X-Maxim-* headers propagated? |
|---|---|---|
| `"leader"` | `True` | ✓ Request-Id, Agent-Id, Session-Id, Lane, Parent-Request-Id, Protocol-Version |
| `"_external"` | `False` | ✗ Only User-Agent |

`_external` is the shared endpoint used by HuggingFace model downloads, third-party tool surface (`tools/http_fetch`, `tools/internet_search`), and peer-cli remote admin calls. Suppressing X-Maxim-* on these calls prevents request IDs / agent IDs / session IDs from leaking to arbitrary web services. Future endpoints that cross node boundaries within the cluster must set `internal=True`; endpoints that talk to third parties must set `internal=False`.

**What uses the context:**
- `utils/http.py::_build_headers` reads from it to set X-Maxim-* on outbound requests to internal endpoints
- `log_structured(logger, level, event, data)` emits structured events that callers can enrich with agent_id from their own context
- Plan 2 R2b's `_normalize_request_context` will bridge legacy dict contexts to `RequestContext`
- Plan 3 R2.5's `_MaximPeerBackend.complete_with_usage` will use it directly (does not exist yet)
- Plan 4's admin API will read from it for per-agent stats (does not exist yet)

**What does NOT use the context:**
- `_OpenAIBackend` — pre-existing cloud path, uses its own request ID scheme
- Simulation generators / test fixtures — may run without a context; contextvar is None; logging falls back to null values
- Probe calls — use null or synthetic context since they're not agent-scoped. This is why the doctor smoke test's JSONL events show `agent_id: null` — doctor probes are cluster-scoped diagnostics, not agent-scoped inference calls. Context binding is the caller's responsibility and happens in the agent loop / sim orchestrator, not in `utils/http.py`.
- `raw_proxy_forward` — explicitly bypasses context to preserve caller-supplied headers (see below)

**If you add a new layer that handles requests:** read from the contextvar via `http.current_context()`; don't require it as a function parameter.

## Data sovereignty — what crosses node boundaries

Single-tenant deployment assumption: one user's API key controls one cluster. Multi-tenant user isolation is out of scope.

**What travels over HTTPS from a peer to the leader:**
- Full system + user prompts (chat completion request body)
- Tool definitions (when tool calling)
- Thinking mode config (Anthropic extended thinking)
- `X-Maxim-*` headers (agent_id, session_id, request_id, lane, protocol version)
- Cluster key in `Authorization` header

**What gets logged on each side:**

| Data | Sender side | Leader side |
|---|---|---|
| Request ID | ✓ (structured log) | ✓ (structured log) |
| Agent ID | ✓ | ✓ |
| Session ID | ✓ | ✓ |
| Lane | ✓ | ✓ |
| HTTP status + latency | ✓ | ✓ |
| Full request body | ✗ unless `MAXIM_HTTP_TRACE=1` | ✗ unless `MAXIM_HTTP_TRACE=1` |
| Full response body | ✗ | ✗ |
| Cluster key | ✗ (never logged) | ✗ (never logged) |

**Retention:** the JSONL log file (`MAXIM_LOG_FILE=...`) rotates per the operator's log rotation config. Default Python logging: no rotation unless operator configures it. Recommendation: use `logging.handlers.RotatingFileHandler` with 10MB rotation, 5 backups.

**Operator implication:** for single-user deployments, this is fine — all data is yours anyway. For a hypothetical future multi-tenant deployment, the contract would need rework (per-tenant keys, tenant-scoped log partitioning, cross-tenant audit). **Multi-tenant is out of scope; see the note at the top.**

## Probe lifecycle [Target — Plan 2 R2c]

**Stage 2 does not exist yet.** Current [Present] state: [`probe_llm_server`](../../src/maxim/runtime/llm_server.py#L216) does a single-stage `GET /v1/models` with a 2-attempt retry. Both attempts hit stage 1; there is no micro-completion readiness probe yet. Plan 2 R2c adds stage 2 and the per-outcome cache TTL table below.

Probes determine whether a peer is reachable and usable. Two stages:

**Stage 1 — liveness** (`GET /v1/models`):
- 1.5s timeout
- Classifies into: `ok`, `auth_rejected` (401 — listener alive but key rejected), `http_5xx`, `timeout`, `connection_refused`, `dns_fail`, `tls_error`, `other`
- Fast, cheap, no model compute

**Stage 2 — readiness** (`POST /v1/chat/completions` with `max_tokens=1, temperature=0`):
- Only runs if stage 1 returned `ok`
- 3s timeout
- Classifies into: `ok` (inference works), `inference_broken` (listener alive but chat endpoint broken), or falls through to generic outcome
- Non-HTTP exceptions (JSON parse, library crash) treat stage 1's `ok` as final + log warning

**Cache TTLs by outcome** (in `runtime/probe_cache.py`):

| Outcome | TTL | Rationale |
|---|---|---|
| `ok` | 60s | Stable state, don't re-probe aggressively |
| `auth_rejected` | 60s | User must fix key; don't hammer |
| `inference_broken` | **15s** | Might be mid-load; retry sooner |
| `http_5xx`, `timeout`, `connection_refused` | 60s | Local fallback kicks in |
| `dns_fail`, `tls_error` | 60s | Network issue; unlikely to resolve in 15s |

**Corruption handling:** [Present] `probe_cache.load_cache` catches `JSONDecodeError` / `OSError` and returns an empty dict. Plan 2 R2c upgrades the log level on this path: current code logs at `logger.debug` (see [probe_cache.py:72](../../src/maxim/runtime/probe_cache.py#L72)); Plan 2 promotes it to `logger.warning` since a corrupted cache file is operationally significant enough to surface without needing `-v`.

**Probe metrics:**
- `probe_outcome_total{endpoint, outcome, stage}` — counter
- `probe_latency_seconds{endpoint, stage}` — histogram
- `probe_cache_hits_total{endpoint}` — counter

## Role detection [Target — Plan 2 R2a]

**Does not exist yet.** `runtime/role.py::detect_role()` is introduced in Plan 2 R2a. Current [Present] state: role is inferred implicitly in three different places (`peer/config.py::read_peer_config` existence check, `lane_backends._apply_local_llm_override`, `cli.py::main`) which is exactly the pattern that caused the 2026-04-12 persisted-profile incident (commit `d875fb9`). The section below describes the post-R2a state.

Role (`leader | peer | solo`) is detected **once** at process startup by `runtime/role.py::detect_role()`, called as the first action in `cli.py::main()`. Result is exported to `os.environ["MAXIM_ROLE"]` for downstream reads.

**Decision tree:**

```
1. MAXIM_ROLE env var set?                 → use that
2. mesh.yml exists?                         → peer (or leader if self matches a role:leader entry)
3. peer.yml exists?                         → peer (legacy)
4. --llm <local> flag + no peer config?     → solo
5. default                                   → leader
```

**Why explicit:** the 2026-04-12 persisted-profile incident was caused by implicit role inference in three different code paths. Having a single `detect_role()` function called once at startup makes "what role am I?" unambiguous.

**Role-scoped state:** `~/.maxim/util/active_llm_model.{role}.txt`. Only the matching role reads/writes it. Prevents a stale leader profile from clobbering peer config (the original incident).

## Observability surfaces

Summary of the metrics + logs + traces that Plans 1-4 introduce. Two parallel metric registries exist today, and future metrics should be added to whichever fits better — don't try to unify them.

### Two parallel metric singletons

1. **`lane_metrics.get_metrics_registry()` → `MetricsRegistry.snapshot()`**: LLM-lane scoped (per large/medium/small tier). Tracks jobs submitted/completed/failed, in-flight count, per-lane latency reservoir, token + cost accumulators. Owned by `LaneBackendManager` + `LeaderProxy` admission control.

2. **`http.metrics_snapshot()`**: HTTP-endpoint scoped (per registered endpoint name). Tracks request counts by `(endpoint, status)`, per-endpoint latency reservoir, pool exhaustion, header sanitizer rejections, startup phase durations. Owned by `utils/http.py` module-level `_metrics` singleton.

**Why separate:** lane metrics and HTTP metrics don't share a key space. Lane = `"large"` / `"medium"` / `"small"`; endpoint = `"leader"` / `"_external"` / etc. Forcing them into the same registry would either flatten the shape or introduce a fake composite key. Both snapshots are consumed by `maxim doctor` and will be consumed by Plan 4's `/v1/mesh/state` admin endpoint.

### Metrics — Plan 1 R1 (shipped)

Exposed via `http.metrics_snapshot()`. Field names use the dict-of-dicts shape, not the Prometheus `name{label=value}` serialization:

- `requests_total` — `{"endpoint/status": count}` dict (e.g. `"leader/200": 42`)
- `latency` — `{endpoint: {count, p50_ms, p99_ms}}` reservoir (last 200 samples per endpoint)
- `pool_exhausted_total` — `{endpoint: count}` dict
- `header_rejected_total` — `{field: count}` dict (sanitizer rejections)
- `startup_phases` — `{phase: duration_ms}` dict

**Reserved but not yet emitted:** `startup_phases` is populated via `http.record_startup_phase(phase, ms)`. The helper exists; no call sites invoke it yet. Plan 2's `detect_role()` or a future `cli.py` startup timer is the natural first caller. A future session that adds the first `record_startup_phase` call should also update this sentence.

**Planned (Plans 2-4, not yet shipped):**
- `http_pool_in_use{endpoint}` gauge — Plan 1 spec mentioned it; not implemented (httpx's pool doesn't expose a cheap gauge read)

**Plan 2 (Typed Errors):**
- `probe_outcome_total{endpoint, outcome, stage}` counter
- `probe_latency_seconds{endpoint, stage}` histogram
- `probe_cache_hits_total{endpoint}` counter

**Plan 3 (Fast Failover):**
- `backend_call_duration_seconds{provider, status}` histogram ← **the key metric for verifying the ~50s retry loop is gone** (code comment at [openai_backend.py:291](../../src/maxim/models/language/openai_backend.py#L291) says "~50s")
- `backend_unclassified_errors_total{provider}` counter ← safety net gap detector
- `backend_stream_chunks{provider}` histogram

**Plan 4 (Operator Visibility):**
- `dispatch_requests_total{lane, outcome}` counter
- `dispatch_latency_seconds{lane}` histogram
- `dispatch_attempts{lane}` histogram
- `dispatch_selected_node{lane, node}` counter
- `request_trace_entries_total` counter
- `request_trace_dropped_total` counter
- `mesh_admin_requests_total{endpoint, status}` counter
- `mesh_admin_rate_limited_total{endpoint}` counter
- `agent_rate_limited_total{agent_id}` counter (bounded cardinality — only configured agents)
- `agent_in_flight_requests{agent_id}` gauge (bounded cardinality)

**Metric label cardinality rule — enumerated:** "hot-path" is defined by this list. The following metrics are labeled ONLY with the listed labels, never extended with `agent_id` / `session_id` / `request_id`:

| Metric | Allowed labels | Forbidden |
|---|---|---|
| `http_requests_total` | `endpoint`, `status` | everything else |
| `http_latency_seconds` | `endpoint` | everything else |
| `http_pool_in_use` | `endpoint` | everything else |
| `http_pool_exhausted_total` | `endpoint` | everything else |
| `startup_phase_duration_seconds` | `phase` | everything else |
| `probe_outcome_total` | `endpoint`, `outcome`, `stage` | everything else |
| `probe_latency_seconds` | `endpoint`, `stage` | everything else |
| `backend_call_duration_seconds` | `provider`, `status` | everything else |
| `backend_unclassified_errors_total` | `provider` | everything else |
| `dispatch_requests_total` | `lane`, `outcome` | everything else |
| `dispatch_latency_seconds` | `lane` | everything else |

**Two exceptions** where `agent_id` IS a label, with bounded cardinality enforced:
- `agent_rate_limited_total{agent_id}` — bounded by `mesh.yml::agent_rate_limits` config (only configured agents appear)
- `agent_in_flight_requests{agent_id}` — bounded by top-N observed agents; all others aggregate as `agent_id="__other__"` with N=20 by default

**Adding a new metric** means adding it to this table + justifying the labels. Metrics that don't fit this pattern go in logs, not Prometheus. Per-agent debugging uses JSONL logs (grep by `agent_id`) + Plan 4's `/v1/mesh/agents/<id>/stats` endpoint, never metric-label-explosion.

### Structured log events (via `log_structured()`)

**Plan 1:**
- `startup_phase` — per-phase duration at startup
- `http_request` / `http_request_failed` — every outbound call

**Plan 2:**
- `role_detected` — first log line of every startup
- `persisted_model_migrated` — migration on upgrade
- `backend_error_classified` — any typed exception raised
- `probe_started` / `probe_completed` — each probe lifecycle
- `probe_cache_corrupt` — when cache file unreadable

**Plan 3:**
- `peer_backend_call` — every `_MaximPeerBackend` call
- `peer_backend_failed` — every failure with full context
- `peer_stream_start` / `peer_stream_complete` — streaming events
- `dispatch_exhausted` — aggregated failure when all providers exhausted

**Plan 4:**
- `request_trace` — every dispatch decision
- `request_trace_overflow` — ring buffer overflow
- `mesh_admin` — admin API access
- `agent_rate_limited` — when per-agent rate limit fires
- `cluster_key_rotation_step` / `cluster_key_rotation_complete` — rotation audit trail

### Traces (Plan 4 ring buffer)

`GET /v1/mesh/request-trace/<req_id>` returns the dispatch decision history for a specific request. Per-agent filtering via `?agent_id=X`. In-memory only, last 100 entries by default (configurable via `MAXIM_REQUEST_TRACE_SIZE`).

## Concurrency model

Three concurrency boundaries:

1. **`LaneBackendManager._lock`** serializes backend construction. Not held during runtime calls.
2. **`LLMRouter._inference_lock`** serializes runtime calls within a router instance. **This is the per-lane head-of-line blocker under multi-agent load.** Plan 3 shortens the typical hold time from ~50s (gateway retries) + ~1.5s (normal retries) = ~52s total to ~100ms-5s. Full async routing is [deferred/llm_path_async_router.md](../plans/deferred/llm_path_async_router.md).
3. **`httpx.Client` connection pool** in `utils/http.py` — bounded per endpoint, N concurrent calls max. Under multi-agent load, this is the cluster-level outbound cap.

**What's thread-safe:**
- `ProviderState` mutations under `_inference_lock`
- `probe_cache` reads/writes (internal lock)
- `ENDPOINT_REGISTRY` reads (immutable post-registration)
- `lane_metrics.metrics_snapshot()` reads (internal lock)

**What's NOT thread-safe:**
- Multiple routers sharing state (none today — each lane has its own router instance)
- Direct writes to `ENDPOINT_REGISTRY` after startup (register endpoints once at boot)

## Decision tables for the most common questions

### "Which backend will serve this request?"

| Provider config field | Result |
|---|---|
| `backend_class: "maxim_peer"` | `_MaximPeerBackend` |
| `backend_class: "openai"` + cloud URL | `_OpenAIBackend` with cloud features |
| `backend_class: "openai"` + self-hosted URL | ⚠ pre-refinement path; should be `maxim_peer` after Plan 3 |
| `backend_class: "llama_cpp"` | `_LlamaCppBackend` |
| `backend_class: "transformers"` | `_PyTorchTransformersBackend` |
| No `backend_class` field + self-hosted URL | Default to `maxim_peer` |
| No `backend_class` field + cloud URL | Default to `openai` |

### "Why did my request fail?"

| Log event | Root cause | Fix |
|---|---|---|
| `peer_backend_failed error=BackendOverloaded` | Leader is at capacity | Add peers, enable batching, wait |
| `peer_backend_failed error=BackendDown http_status=502` | Leader process down or upstream crashed | Check leader logs, `maxim peer --node X status` |
| `peer_backend_failed error=BackendTimeout` | Network slow or leader overloaded | Check RTT, increase timeout, check leader GPU load |
| `peer_backend_failed error=BackendAuthFailed` | Cluster key out of sync | `maxim peer rotate-cluster-key` or verify `mesh.yml::cluster_key` |
| `peer_backend_failed error=BackendModelMissing` | Model not loaded on peer | `maxim peer --node X install <model>` |
| `peer_backend_failed error=BackendInferenceBroken` | Leader up, chat endpoint broken | Check llama-cpp-server logs, chat template, VRAM |
| `backend_unclassified_errors_total > 0` | **Typed exception gap** | File a bug, add a typed class |
| `dispatch_exhausted` | All providers failed | See individual attempt outcomes in the event |

### "Where does this metric come from?"

Grep for the metric name in `lane_metrics.py` and the plans. Every metric has a named emitter.

## Behaviors not obvious from the layer walkthrough

A grab-bag of behaviors that span layers or aren't visible from the happy path. Claude 2's architecture review flagged these as missing from the earlier draft — including them here makes future refactors safer.

### Mid-stream failure in `_OpenAIBackend._stream_response` [Present]

Current [Present] state: the streaming path at [openai_backend.py:421-478](../../src/maxim/models/language/openai_backend.py#L421-L478) collects text chunks into a list. If a chunk iteration raises mid-stream, the method doesn't re-raise — it returns whatever text was collected so far with whatever `stop_reason` was last seen. This is a **silent partial-response path** that the router's empty-content detection will *not* catch if any text was received.

Plan 3's `_MaximPeerBackend` takes a stricter approach: mid-stream HTTP errors raise `BackendDown` so the router falls over to a new provider. The two backends have intentionally different contracts here — cloud providers (`_OpenAIBackend`) tolerate partial responses because cloud streaming is commonly used for first-token-latency UX where "got some tokens" is better than "nothing." Peer tunnels (`_MaximPeerBackend`) prefer hard failure because the router has fallback options.

### `LaneBackendManager.unload_all` shutdown semantics [Present]

`unload_all` ([lane_backends.py:461-472](../../src/maxim/runtime/lane_backends.py#L461-L472)) acquires `_lock`, iterates cached backends, calls `unload()` on each that has one, swallows exceptions, then clears the backend dict. It's called from `LLMWorker.stop()` during process shutdown. Key properties:

- **Blocking.** Holds `_lock` until all backends have been released. If a backend's `unload()` hangs, the shutdown hangs.
- **Best-effort.** Any exception from a backend's `unload()` is swallowed. A broken unload doesn't prevent other backends from being released.
- **One-shot.** After `unload_all` returns, the manager's cache is empty. A subsequent `get_backend` would re-construct from scratch.

**Startup-ordering implication** (CLAUDE.md "startup ordering in cli.py" lesson): because `_lock` is held during unload, a request arriving mid-shutdown that tries to construct a new backend will block on `_lock`. This is acceptable for graceful shutdown but breaks if shutdown is also trying to drain in-flight requests that need new backends. The LLMWorker shutdown sequence MUST drain in-flight requests before calling `unload_all`, not after.

### `_validate_remote_urls` probe caching + invalidation [Present, extended in Plan 2]

Current state: `_validate_remote_urls` ([lane_backends.py:1185](../../src/maxim/runtime/lane_backends.py#L1185)) runs during router construction, probes each lane's `remote_url`, and drops unreachable ones from the lane config. Probe results flow through `runtime/probe_cache.py`, which persists outcomes to `~/.maxim/util/probe_cache.json` with a TTL.

**Operational subtlety:** `maxim peer restart` relies on `probe_cache.clear_cache_for_url(url)` to invalidate the cached result for the restarted leader. Without the invalidation, a peer that probed the leader 30 seconds ago would still consider it alive for another 30 seconds and burn requests against a dead URL. The invalidation happens in `runtime/leader_proxy.py` as part of the restart lifecycle.

**Plan 2 extension:** R2c adds per-outcome TTLs (15s for `inference_broken`, 60s for others). The invalidation path is unchanged; only the freshness check is.

### Shutdown interaction of retry loops [Present — load-bearing]

`_OpenAIBackend` (current) has an internal retry loop. Each iteration checks `is_shutdown_requested()` before the HTTP call and during backoff sleeps via `shutdown_wait(backoff_s)`. This is why `Ctrl+C` during a cloud provider outage aborts the retry within seconds instead of burning the full ~50s.

**Load-bearing:** any future refactor that replaces the retry loop (or adds a new one) must preserve these cooperative cancellation points. Plan 3's `_MaximPeerBackend` has no retry loop so the problem doesn't exist; but if Plan 3's deferred multi-peer dispatch ever adds a client-side retry, it MUST also integrate `is_shutdown_requested()` or regress the Ctrl+C-responsiveness property.

See [openai_backend.py:294-297](../../src/maxim/models/language/openai_backend.py#L294-L297) and [openai_backend.py:355](../../src/maxim/models/language/openai_backend.py#L355) for the existing integration.

### `_candidate_providers` context-window filtering [Present]

The router's provider candidate list is filtered by MORE than just backoff state. [`_candidate_providers`](../../src/maxim/models/language/router.py#L398-L401) also rejects providers whose advertised context window is too small for the estimated `prompt_tokens + max_tokens` of the current request.

**Implication:** a provider with a 4k-context model will be silently skipped for a 16k-context request, even if it's healthy. This is correct routing but can look like "the provider is broken" in logs. The architecture's error-flow assumes backoff-state is the only filter; context-window filtering is a parallel filter that's just as real.

**Plan 3 consideration:** when router logs `dispatch_exhausted`, the JSONL event should include WHY each provider was skipped (backoff vs. context-window). Otherwise operators will misdiagnose large-prompt routing as peer-down.

### `allow_local_endpoints` parallel classification [Present]

`lane_backends.py::_classify` decides whether a lane with a `remote_url` is "self-hosted" (private IP or peer-owned) or "cloud" (public host, cloud provider). This is the PRIMARY classification — it determines which `BACKEND_CLASSES` entry is used (Plan 3 R2.5) and whether the lane counts against `MAXIM_MAX_CLOUD_LANES`.

**But there's a parallel check** at [router.py:391-394](../../src/maxim/models/language/router.py#L391-L394): `_candidate_providers` independently checks `allow_local_endpoints` on the provider dict to bypass the cloud gate. This is a second source of truth for "is this self-hosted?" and it can disagree with `_classify` if the config is inconsistent.

**Invariant**: `_classify` is authoritative. The router's `allow_local_endpoints` check should match. If you add a new classification path, it MUST consult `_classify`, not re-implement the private-IP check. The 2026-04-12 Cloudflare incident had one of its roots in this divergence — the probe classified a public CF-tunnel URL as "self-hosted" based on one heuristic while the HTTP client treated it differently based on another.

## Versioning + stability

**Protocol version header:** `X-Maxim-Protocol-Version: 1` on every outbound mesh call. Receivers log WARN on unknown versions but process requests normally (forward compat). Breaking the wire protocol requires bumping this.

**`mesh.yml` schema stability:** adding fields is non-breaking. Removing or renaming fields requires a version bump.

**`RequestContext` field stability:** the five canonical fields (`request_id`, `agent_id`, `session_id`, `lane`, `parent_request_id`) are stable. Adding fields is non-breaking.

**`BackendError` class stability:** the existing six subclasses are stable. Adding new subclasses is non-breaking as long as the router's safety-net catches `BackendError` base.

**Log event names:** stable post-Plan-4. Renaming or removing an event name is a breaking change for log consumers.

## Things this architecture explicitly does NOT do

- **Multi-tenant user isolation** — one cluster = one user
- **Proactive load-aware routing** — heartbeats, capability broadcast, ranking formulas are all deferred
- **Async router** — concurrent per-lane agent calls still serialize under `_inference_lock`
- **Fair-share scheduling** — per-agent rate limiting is a hard cap, not work-conserving fairness
- **Peer-to-peer GGUF streaming** — every node pulls from the canonical source (HuggingFace)
- **Request-triggered auto-download** — explicit `install` only
- **Mesh service discovery** — nodes declared in `mesh.yml`, no mDNS/gossip
- **Per-node JWT / mTLS** — shared cluster key

Each of these has a deferred plan documenting when to revive it.

## Related docs

- **Plans:** [llm_path_refinement.md](../plans/llm_path_refinement.md) — meta-plan
- **Sub-plans:** [llm_path_foundation.md](../plans/llm_path_foundation.md), [llm_path_typed_errors.md](../plans/llm_path_typed_errors.md), [llm_path_fast_failover.md](../plans/llm_path_fast_failover.md), [llm_path_operator_visibility.md](../plans/llm_path_operator_visibility.md)
- **Deferred plans:** [../plans/deferred/llm_path_multi_peer_dispatch.md](../plans/deferred/llm_path_multi_peer_dispatch.md), [../plans/deferred/llm_path_async_router.md](../plans/deferred/llm_path_async_router.md), [../plans/deferred/llm_path_fair_scheduling.md](../plans/deferred/llm_path_fair_scheduling.md)
- **Operator runbook** (created by Plan 4): [mesh_operations.md](mesh_operations.md)
- **Troubleshooting:** [../troubleshooting/http_debugging.md](../troubleshooting/http_debugging.md), [../troubleshooting/peer_backend_debug.md](../troubleshooting/peer_backend_debug.md), [../troubleshooting/mesh_debug.md](../troubleshooting/mesh_debug.md)
- **Codebase reference:** [../reference.md](../reference.md)
- **Project guide:** [../../CLAUDE.md](../../CLAUDE.md)
