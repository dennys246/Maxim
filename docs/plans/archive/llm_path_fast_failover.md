# LLM Path Refinement — Plan 3: Fast Failover

**Status:** Draft v3 — renumbered 2026-04-12 after Plan 2 (Typed Errors) split out of former Plan 1
**Scope:** ~420 LOC new + ~-80 LOC deleted (includes streaming support)
**Target version:** 0.4 (single stability version)
**Part of:** [llm_path_refinement.md](llm_path_refinement.md)
**Depends on:**
- [llm_path_foundation.md](llm_path_foundation.md) (Plan 1) — `utils/http.py`, `RequestContext`, `X-Maxim-*` headers
- [llm_path_typed_errors.md](llm_path_typed_errors.md) (Plan 2) — `BackendError` hierarchy, two-stage probe shape, SSRF in `utils/net.py`
**Enables:** [llm_path_operator_visibility.md](llm_path_operator_visibility.md) (Plan 4) — formerly "reactive mesh", now scope-reduced per user decision

## Goal

**Kill the 52-second retry loop** in a multi-agent-aware way.

`_OpenAIBackend` has an internal retry loop that waits up to ~52 seconds on 502/503/504 before giving up. When the leader restarts, every in-flight peer request hits this loop, making `maxim peer restart` feel glacial. This plan replaces `_OpenAIBackend` for self-hosted peer traffic with a purpose-built `_MaximPeerBackend` that raises typed exceptions on first failure and lets the router's existing fallback loop do the retrying.

**Multi-agent enrichment:** every backend call carries full `RequestContext` (from Plan 1) to the leader as `X-Maxim-*` headers. The leader can log, account, and trace by agent without guessing.

Four concrete outcomes:

1. **Sub-5-second failover.** `backend_call_duration_seconds` p99 under mocked-dead-peer fixture < 5s (vs. ~52s pre-plan).
2. **One probe implementation.** Three scattered probe functions collapse into `_MaximPeerBackend.health_check()`.
3. **Streaming works on self-hosted peers.** (User decision: ship streaming in v1.)
4. **Agent context flows end-to-end.** Every request's `agent_id` is in the leader's logs, the peer's logs, metrics, and the Plan 4 dispatch trace.

## Non-goals

- **Not building distribution.** Single peer configured via `peer.yml` or `mesh.yml` (Plan 4). Multi-peer overflow is **deferred** in [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md).
- **Not touching `_OpenAIBackend`.** Cloud providers keep existing behavior.
- **Not changing router's fallback loop.** Already correct.
- **Not adding heartbeats.** See deferred capability-aware mesh.

## Why a custom backend — the decision preserved

This is the most important architectural decision in the plan. Preserved from v1 because it's still the correct call.

**`_OpenAIBackend` was designed for cloud providers.** Features that make sense there are actively harmful for self-hosted peers:

| `_OpenAIBackend` feature | Cloud context | Self-hosted peer context |
|---|---|---|
| Retry loop (429 + 502/503/504) | Correct for transient cloud errors | **Blocks fallover ~52s** |
| `max_retries=2`, `max_gateway_retries=4` | Reasonable for billed retries | Wrong for peer tunnel |
| Returns empty `LLMResponse` on failure | Matches router expectations | **Loses error taxonomy** |
| Cost tracking | Required for billing | Zero cost — overhead |
| PII redaction | Required for compliance | Self-infrastructure — N/A |
| String-matching error classification | Fragile but functional | Breaks on CF error bodies |
| OpenAI SDK dependency | Already using it | Unnecessary — httpx is simpler |

**Patching would require gutting half its logic conditionally.** Worse than having two backends: a fragile conditional is harder to reason about than two clear-purpose classes.

**Key invariant:** `_MaximPeerBackend.complete_with_usage()` makes **exactly one HTTP call**. No retry. No backoff. If the call fails, a typed exception raises. The router decides what to do next.

## Context

The audit math ([openai_backend.py:278-374](../../src/maxim/models/language/openai_backend.py#L278-L374)):
- `max_gateway_retries = 4` × backoffs (`min(5s × n, 20s)`) → up to **50s** on 502/503/504
- `max_retries = 2` × backoffs (`0.5s × n`, × 4 for rate limits up to 30s) → up to **60s** on rate limits

Peer startup currently feels broken because the retry loop is invisible. The peer looks healthy, the log says "retrying," and 50+ seconds later the request finally fails.

**Multi-agent lens finding (new in v2):** under concurrent agents, the retry loop is worse than the numbers suggest. Agent A's 52-second retry holds `LLMRouter._inference_lock` (per-lane). Agent B's next call to the same lane blocks waiting for A to finish. **One slow peer under the old backend serializes the entire lane's traffic.** With the new backend, agent A fails in ~2s, agent B's call attempts immediately after, no serialization-amplified delay.

## Phases

### R2.5 — `_MaximPeerBackend` + router integration + streaming — ~420 LOC new

**The most important phase in the entire LLM path refinement.**

**New file: `src/maxim/models/language/maxim_peer_backend.py` (~300 LOC)**

```python
class _MaximPeerBackend:
    """OpenAI-compatible backend purpose-built for self-hosted Maxim peers.
    
    Design principles:
    - Single HTTP call per complete_with_usage(). No internal retry loop.
      Router is the single point of retry policy.
    - Raises typed exceptions (BackendError hierarchy from Plan 2) so
      the router classifies failures without string matching.
    - Parses Maxim-specific response headers (X-Maxim-Queue-Depth,
      X-Maxim-Suggested-Peer, X-Maxim-Node-Id, Retry-After).
    - Uses Plan 1's shared http client — shared pool, automatic request
      ID + agent_id header propagation via RequestContext.
    - Direct JSON parse. No openai SDK dependency.
    - Supports streaming via httpx native streaming (user decision).
    
    NOT this backend's job:
    - Cost tracking (zero cost on self-hosted)
    - PII redaction (peer tunnel is self-infrastructure)
    - Retry logic (router does this)
    - Internal backoff (existing ProviderState tracks it at the router level)
    """
    requires_prompt_formatting = False
    supports_model_override = True
    supports_streaming = True
    supports_tool_use = True

    def __init__(self, cfg: LLMConfig, provider_key: str):
        self._cfg = cfg
        self._provider_key = provider_key
        self._endpoint_name = f"peer-{provider_key}"
        self._register_endpoint_if_needed()

    def warmup(self) -> bool:
        """Return True if auth + URL validation pass. No billable call."""
        return bool(self._get_api_key())

    def health_check(self) -> ProbeResult:
        """Two-stage probe. Replaces probe_llm_server + llm_server_responding_at.
        
        Stage 1 (liveness): GET /v1/models, 1.5s timeout
        Stage 2 (readiness): micro-completion, 3s timeout
        Failure on stage 2 → BackendInferenceBroken, 15s cache TTL
        """
        stage1 = self._probe_liveness()
        if stage1.outcome != "ok":
            return stage1
        return self._probe_readiness_with_fallback(stage1)

    def complete_with_usage(
        self, *, system, user, max_tokens, temperature, 
        stop=(), model_override=None, tools=None, thinking=None, stream=False,
        **kwargs,
    ) -> LLMResponse:
        """ONE HTTP call. Raises typed exceptions on failure."""
        context = self._build_request_context(kwargs)
        
        # Shutdown check — honor cancellation even though we don't retry
        if is_shutdown_requested():
            raise BackendDown(self._provider_key, http_status=None)

        payload = self._build_payload(
            system, user, max_tokens, temperature,
            stop=stop, model_override=model_override, tools=tools,
            thinking=thinking, stream=stream,
        )

        try:
            if stream:
                return self._stream_response(payload, context)
            resp = http.post(
                self._endpoint_name,
                path="/chat/completions",
                json=payload,
                context=context,
            )
        except HTTPTimeout as e:
            raise BackendTimeout(self._provider_key, elapsed_s=e.elapsed_s) from e
        except HTTPConnectionError as e:
            raise BackendDown(self._provider_key, http_status=None) from e
        except HTTPServerError as e:
            raise BackendDown(self._provider_key, http_status=e.status) from e
        except HTTPRateLimited as e:
            raise BackendOverloaded(
                self._provider_key,
                retry_after_s=e.retry_after_s,
                suggested_peer=e.suggested_peer,
                queue_depth=self._parse_queue_depth(e.response_headers),
            ) from e
        except HTTPAuthError as e:
            raise BackendAuthFailed(self._provider_key) from e
        except HTTPClientError as e:
            if e.status == 404:
                raise BackendModelMissing(self._provider_key, self._model_name()) from e
            raise BackendError(self._provider_key, fix_hint=str(e)) from e
        
        return self._parse_llm_response(resp.json(), context=context)

    def complete(self, prompt, *, max_tokens, temperature, stop, system=None) -> str:
        """Legacy path-A entry. Delegates to complete_with_usage."""
        resp = self.complete_with_usage(
            system=system or "",
            user=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return resp.content

    def unload(self) -> None:
        """Shared http client is registry-owned. Nothing to unload."""
        pass

    # ─── Private helpers ────────────────────────────────────────────────────
    def _build_request_context(self, kwargs) -> RequestContext: ...
    def _build_payload(self, system, user, max_tokens, temperature, **kw) -> dict: ...
    def _parse_llm_response(self, raw: dict, *, context: RequestContext) -> LLMResponse: ...
    def _stream_response(self, payload: dict, context: RequestContext) -> LLMResponse: ...
    def _probe_liveness(self) -> ProbeResult: ...
    def _probe_readiness_with_fallback(self, stage1: ProbeResult) -> ProbeResult: ...
    def _parse_queue_depth(self, headers: Mapping[str, str]) -> int: ...
    def _register_endpoint_if_needed(self) -> None: ...
    def _get_api_key(self) -> str: ...
    def _model_name(self) -> str: ...
```

**`_build_request_context` is a thin wrapper around Plan 2 R2b's `_normalize_request_context`** (canonical shim living in `agents/llm_worker.py`). This backend only builds a `RequestContext` from the `request_context` kwarg — it reuses the shim; it does not introduce a parallel normalization path. The shim owns the `"agent"` → `"agent_id"` migration; the backend just calls into it.

**Streaming support (user decision — ship in v1):**

```python
def _stream_response(self, payload: dict, context: RequestContext) -> LLMResponse:
    """Streamed completion via httpx native streaming.
    
    Collects the full response into an LLMResponse. Caller gets the same
    shape as non-streaming. If the stream errors mid-flight, raises
    BackendDown — the router falls over to a new provider.
    
    NOTE: this is a single HTTP call with streaming response body. Still
    one call per complete_with_usage(). No retry loop here either.
    """
    payload_with_stream = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    text_parts: list[str] = []
    usage = {}
    stop_reason = ""
    
    try:
        with http.stream_post(
            self._endpoint_name,
            path="/chat/completions",
            json=payload_with_stream,
            context=context,
        ) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                # ... parse chunks, accumulate text, extract usage
    except HTTPError as e:
        # Convert HTTPError subclasses to BackendError subclasses
        # (same mapping as complete_with_usage)
        raise self._map_http_error_to_backend_error(e)
    
    return LLMResponse(
        content="".join(text_parts),
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        model=self._model_name(),
        provider=self._provider_key,
        stop_reason=stop_reason,
    )
```

~80 LOC extra for streaming. Worth it for feature parity with `_OpenAIBackend` so callers don't silently downgrade.

**Shutdown handling:** the shared httpx client closes on process shutdown, which cancels in-flight streams. The `is_shutdown_requested()` check at the top of `complete_with_usage` prevents new calls from starting during shutdown. Streams that are mid-flight when shutdown fires raise `BackendDown`.

**Router integration (~100 LOC) — modify `router.py::_try_provider`:**

Update the catch-all at line 650 to catch typed exceptions first:

```python
try:
    text, usage = self._invoke_backend(...)
    if text:
        return text, usage, "success"
except BackendOverloaded as e:
    self._note_provider_overload(provider_key, retry_after_s=e.retry_after_s)
    self._record_suggested_peer_hint(e.suggested_peer)  # used by deferred multi-peer
    return "", None, "failed"
except BackendDown as e:
    self._note_provider_failure(provider_key, f"down_{e.http_status}")
    return "", None, "failed"
except BackendTimeout as e:
    self._note_provider_failure(provider_key, f"timeout_{e.elapsed_s:.1f}s")
    return "", None, "failed"
except BackendAuthFailed as e:
    self._note_provider_failure(provider_key, "auth_failed")
    self._set_long_backoff(provider_key, 300.0)  # don't hammer
    return "", None, "failed"
except BackendModelMissing as e:
    self._note_provider_failure(provider_key, f"model_missing:{e.requested_model}")
    self._set_long_backoff(provider_key, 60.0)
    return "", None, "failed"
except BackendInferenceBroken as e:
    self._note_provider_failure(provider_key, "inference_broken")
    self._set_short_backoff(provider_key, 15.0)  # match probe cache TTL
    return "", None, "failed"
except BackendError as e:
    self._note_provider_failure(provider_key, "generic_backend_error")
    return "", None, "failed"
except Exception as e:
    # Safety net — typed-exception gap fires this
    self._note_provider_failure(provider_key, "unclassified")
    self._log_provider_error(provider_key, e)
    return "", None, "failed"
```

**Per-exception backoff policy:**
- `BackendOverloaded` → honor `Retry-After`, else short backoff
- `BackendDown` → exponential `min(60s, 2^errors)`
- `BackendTimeout` → exponential like Down
- `BackendAuthFailed` → **300s hard backoff** (doesn't self-heal)
- `BackendModelMissing` → **60s** (operator needs to install)
- `BackendInferenceBroken` → **15s** (match probe cache)

**Aggregated failure logging** — when `_complete_text_locked` exits with all providers failed, log ONE structured JSONL line:

```json
{"ts":"...","level":"WARN","event":"dispatch_exhausted","request_id":"abc123","lane":"large","total_elapsed_ms":2340,"agent_id":"npc-mother","session_id":"sim-42","attempts":[{"provider":"lane-large-rtx-leader","outcome":"overloaded","retry_after_s":5},{"provider":"lane-large-mac-peer","outcome":"timeout","elapsed_s":2.1}]}
```

One WARN per failed dispatch. Grep-friendly. Actionable. Preserves multi-agent context.

**Metric `backend_unclassified_errors_total{provider}`** — fires when the safety-net `except Exception` path is taken. Non-zero means we missed an exception type — file a bug, add a typed class. User decision: warning for now, CI failure after bake-in.

**`_MaximPeerBackend` selection via dispatch table:**

`lane_backends.py::_build_remote_backend` reads a `"backend_class"` config field. Dispatch table is a module-level dict in `lane_backends.py`:

```python
BACKEND_CLASSES: dict[str, type] = {
    "openai": _OpenAIBackend,       # cloud providers
    "maxim_peer": _MaximPeerBackend, # self-hosted peers (default for self-hosted)
}
```

~10 LOC dispatch function. Extensible for future backend types.

### R2.6 — Probe consolidation — ~-80 LOC net

Delete three parallel probe implementations.

**Delete:**
- `runtime/llm_server.py::probe_llm_server`
- `runtime/llm_server.py::llm_server_responding_at`
- `runtime/llm_server.py::_probe_once`

**Update callers:**
- `lane_backends._validate_remote_urls` → `_MaximPeerBackend(cfg).health_check()` per-lane
- `local_server_spawner.py` readiness loop → same
- `doctor/checks.py` leader probe → same
- `peer/cli.py test` command → same

**Keep unchanged:**
- `runtime/probe_cache.py` — interface is probe-source-agnostic

**One probe implementation** means the 2026-04-12 Cloudflare incident (missing User-Agent in one of three probes) becomes structurally impossible.

## Logging & verbosity requirements

Plan 1 established JSONL format. Plan 3 adds:

**Every `_MaximPeerBackend` call** emits a DEBUG line (INFO at `MAXIM_BACKEND_TRACE=1`):
```json
{"ts":"...","level":"DEBUG","event":"peer_backend_call","provider":"lane-large-rtx-leader","model":"qwen2.5-14b-instruct","status":200,"latency_ms":340,"input_tokens":524,"output_tokens":89,"request_id":"abc123","agent_id":"npc-mother","session_id":"sim-42","lane":"large"}
```

**Backend failures** at WARN with full typed exception context:
```json
{"ts":"...","level":"WARN","event":"peer_backend_failed","provider":"lane-large-mac-peer","error":"BackendOverloaded","retry_after_s":5.0,"queue_depth":7,"suggested_peer":"rtx-leader","request_id":"abc123","agent_id":"npc-mother","session_id":"sim-42","fix_hint":"Peer is at capacity. Try a different peer or wait."}
```

**Streaming events** at DEBUG:
```json
{"ts":"...","level":"DEBUG","event":"peer_stream_start","provider":"lane-large-rtx-leader","request_id":"abc123","agent_id":"npc-mother"}
{"ts":"...","level":"DEBUG","event":"peer_stream_complete","provider":"lane-large-rtx-leader","request_id":"abc123","chunks":47,"latency_ms":1240}
```

**Health check outcomes** at INFO on first probe, DEBUG on cache hits. The log line identifies which backend instance ran the probe, because a peer might have multiple backends (one per cached provider).

**Aggregated dispatch failures** at WARN — one line per exhausted dispatch (format shown above).

**New metrics:**
- `backend_call_duration_seconds{provider, status}` — **the key metric** for verifying R2.5's win. Pre-R2.5 p99 includes the retry loop; post-R2.5 should be tight to actual call duration.
- `backend_unclassified_errors_total{provider}` — safety net for missing exception types
- `backend_stream_chunks{provider}` — histogram of streaming response chunk counts

**Multi-agent observation:** all metric labels stay aggregate per Plan 1 decision. Per-agent debugging uses JSONL logs filtered by `agent_id`.

**New env var:** `MAXIM_BACKEND_TRACE=1` — bumps all `_MaximPeerBackend` calls to INFO and includes payload sizes, header dumps, streaming chunk breakdowns. Off by default.

## Multi-agent / multi-user lens findings (new in v2)

Applied during v2 rewrite. Key findings:

**1. Agent context propagation through `RequestContext`.** Plan 1 introduced `RequestContext`. Plan 2 introduced the `_normalize_request_context` shim that canonicalizes legacy `"agent"` keys. Plan 3 threads both through every `_MaximPeerBackend` entry point. The contract: if a caller passes `kwargs["request_context"]`, we hand it to the Plan 2 shim; the shim returns a typed `RequestContext`. If not, we generate a synthetic one with null `agent_id`. Either way, the HTTP call carries `X-Maxim-*` headers.

**2. Serialization under `_inference_lock` is not worse than today.** Existing behavior: one inference call per lane at a time (protected by `LLMRouter._inference_lock`). Under the old backend, slow failures hold the lock for ~52s, blocking all other agents. Under `_MaximPeerBackend`, slow failures hold the lock for ~2-5s. **The lock isn't new; the duration shortens dramatically.** Full async router (multi-agent parallelism within a lane) is a much bigger refactor, flagged for a deferred plan.

**3. Streaming under multi-agent.** A streaming response holds `_inference_lock` for the full duration. Agent A streaming a 2000-token response for 10 seconds blocks Agent B's lane call for 10s. This is the existing constraint of the router, not new. The fix is async routing, deferred.

**4. Header injection via model names.** If an agent's prompt somehow injects a malicious model name into the payload, that value goes into the outbound request body (not headers — bodies are opaque to log injection). Bodies are sanitized at the router level (existing behavior). Headers are sanitized in Plan 1. No new attack surface from multi-agent.

**5. Leader-side per-agent accounting.** Plan 3 propagates `X-Maxim-Agent-Id` to the leader. The leader's log now includes `agent_id` on every inbound call. Plan 4 exposes this via admin API for per-agent query. Future fair-share scheduling is a natural extension point (deferred).

**6. Concurrent `_MaximPeerBackend` instance creation.** If two agents trigger `_build_remote_backend` simultaneously (first cold-start dispatch), `LaneBackendManager._lock` serializes. Existing behavior. OK.

**7. Shutdown under concurrent agents.** SIGTERM signals `is_shutdown_requested()`. Every backend call at the top checks this and raises `BackendDown` if set. In-flight HTTP calls get cancelled when the shared httpx client closes (handled by R1's registry shutdown hook). Streaming calls iterate lines; each iteration can check the shutdown flag. ~5 LOC addition.

## Stress test protocol (runs after R2.5 + R2.6 pass unit tests)

**User decision — integrated with substrate P2 validation + llama.cpp batching PoC.**

This protocol serves triple duty:
1. Validates Plan 3's 52-second-retry-loop fix
2. Stress-tests substrate P2 reward modulation under realistic load
3. Measures whether `llama.cpp --parallel` batching obviates the need for Plan 4's multi-peer dispatch (deferred)

**Setup:**
- **Leader:** RTX 5080, Qwen 14B loaded, `llama.cpp --parallel N` (N varies in sweep)
- **Peer:** Mac (24GB unified memory), connected via `peer.yml`
- Both running 0.4-foundation + 0.4-fast-failover

**Phase A — Substrate P2 validation + Plan 3 baseline (single-user):**

Run substrate P2 fixture sweep from both leader and peer. Expected results:
- Substrate P2 metrics meet their targets (defined in `substrate_recognition.md`)
- `backend_call_duration_seconds` p99 stays under 5s on the peer path
- Zero `backend_unclassified_errors_total` increments
- JSONL logs show consistent `agent_id` propagation

Record baseline: leader p50/p99 latency per lane, peer overflow rate, token throughput.

**Phase B — Multi-agent fan-out (stress Plan 3):**

Run `maxim --sim scenarios/campaigns/heist_v1.yaml --seed 42` on the peer with AgentPool sized at 3-5 concurrent NPCs. Each NPC makes independent LLM calls.

Observe:
- Leader saturation: does `X-Maxim-Queue-Depth` header ever exceed `max_concurrent`?
- Peer behavior on leader 429: do peer requests failover gracefully?
- Log hygiene: can you reconstruct each NPC's full request history via `jq 'select(.agent_id=="npc-X")'`?

**Phase C — `llama.cpp --parallel` batching PoC:**

On the leader, sweep `--parallel` from 1 → 8 with Qwen-14B. For each value:
1. Run the Phase B multi-agent fan-out scenario
2. Measure leader throughput (tokens/sec across all lanes)
3. Measure leader p99 latency
4. Measure peer's effective throughput (via dispatch-trace-like logs)

**Decision criteria** after Phase C:
- **If `--parallel 4` doubles leader throughput and peer never saturates:** batching solves the problem cheaper than distribution. **Deferred multi-peer dispatch can stay deferred indefinitely.** Plan 4 still ships as scoped (operator visibility). Document the finding. Commit the batching config.
- **If batching saturates at some `--parallel` N but peer still needs overflow:** ship Plan 4 as scoped (R3.0 + R3.5-lite + R3.6-lite — operator visibility). Revive deferred multi-peer dispatch from `deferred/llm_path_multi_peer_dispatch.md`.
- **If batching has no measurable effect:** leader's bottleneck is not concurrency. Investigate VRAM pressure, context length, or other constraints before Plan 4 scope decision.

**Phase D — Leader restart mid-workload (the big Plan 3 test):**

Start Phase B on the peer. At turn 20, SSH to leader and `systemctl restart maxim-leader`. Measure:
- Pre-Plan-2 baseline (if available): how many seconds until peer recovers? Expected ~60+
- Post-Plan-2: how many seconds until peer recovers? **Target: < 10s**
- Agent context during recovery: are log lines correctly attributing failed requests to the right agent?

**Phase E — Fault injection (chaos):**

Scripted fault injection against the peer:
1. Kill leader process entirely (not restart — kill)
2. Network partition (iptables drop leader IP)
3. Leader returning 429 for 30 seconds (mock)
4. Leader returning 502 for 60 seconds (mock)
5. Auth rejection (rotate cluster key on leader without telling peer)

For each: verify the peer's logs emit the correct typed exception, `fix_hint` is actionable, `backend_unclassified_errors_total` stays at zero.

**Output — stress test report:**

File: `docs/experiments/results/llm_path_stress_<date>.md`

Required sections:
- **Phase A baseline** — substrate P2 results + Plan 3 baseline latencies
- **Phase B fan-out** — multi-agent behavior, log hygiene findings
- **Phase C batching sweep** — the data that drives the Plan 4 scope decision
- **Phase D restart recovery** — the definitive "52-second retry loop is dead" proof
- **Phase E chaos** — all 5 scenarios, per-scenario verdict
- **Decision** — Plan 4 scope (scoped-only / scoped + multi-peer revival / defer) + rationale
- **Follow-up actions** — anything surprising that needs its own plan

Link this report from the meta-plan + from Plan 4's intro.

## Success criteria — R2.5 + R2.6

**R2.5 — the 52-second retry loop kill:**

- **Performance gate:** `backend_call_duration_seconds` p99 against mocked "dead" peer < 5s (vs. ~52s pre-R2.5)
- **Manual benchmark:** `maxim peer restart` on leader → peer recovers within 10s end-to-end
- **Code invariant:** `grep -E "retry|backoff|gateway" src/maxim/models/language/maxim_peer_backend.py` returns zero
- **Typed exceptions work:** integration test fires one request per exception class, asserts router backoff policy matches class
- **Safety net is instrumented:** `backend_unclassified_errors_total` starts at zero
- **Streaming parity:** streaming test against mocked peer returns identical `LLMResponse` shape to non-streaming
- **Multi-agent propagation:** outbound HTTP request includes `X-Maxim-Agent-Id` header when caller passes agent context

**R2.6 — probe consolidation:**

- `grep "def probe_llm_server\|def llm_server_responding_at" src/maxim/` returns zero
- All previous callers updated and tested
- `maxim doctor` probe results unchanged

**Regression gates:**
- Fast suite green
- `_OpenAIBackend` cloud path unchanged (separate test with mocked Anthropic)
- JSONL log format holds under concurrent agents

## Test fixture specifications

Claude 1's executor review flagged that the performance gate referenced `tests/performance/test_fast_failover.py` with a "mocked dead peer" fixture that was never specified. Here's the design. An executor can implement directly from this section.

### Fixture 1: `tests/performance/test_fast_failover.py`

**Purpose:** verify `_MaximPeerBackend` fails fast when the upstream peer is unresponsive. Gates the p99 < 5s performance claim that this plan's entire motivation rests on.

**Structure:** pytest file with three tests, each exercising a different failure mode. All use `pytest.monkeypatch` to swap the shared `httpx.Client` with a mock; no real network required.

**Test A — mocked `httpx.ConnectError` (TCP refused / DNS fail):**
```python
import time
import pytest
from unittest.mock import MagicMock
import httpx
from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
from maxim.models.language.types import BackendDown

def test_connect_error_fails_under_5s(mock_peer_backend_cfg):
    """Connection refused → BackendDown within total_timeout, not retry loop."""
    backend = _MaximPeerBackend(cfg=mock_peer_backend_cfg, provider_key="test")
    
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.ConnectError("connection refused")
    # Monkeypatch the shared registry so backend uses our mock
    # (implementation detail: _MaximPeerBackend looks up its endpoint from
    # maxim.utils.http.ENDPOINT_REGISTRY at call time)
    
    start = time.monotonic()
    with pytest.raises(BackendDown):
        backend.complete_with_usage(
            system="", user="test", max_tokens=10, temperature=0.0
        )
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"Fast failover broken: took {elapsed:.1f}s (target < 5s)"
```

**Test B — mocked 5xx chain (leader process dead, gateway error):**
```python
def test_502_chain_fails_under_5s(mock_peer_backend_cfg):
    """HTTP 502/503/504 chain → BackendDown fast, NOT 50s of retry."""
    backend = _MaximPeerBackend(cfg=mock_peer_backend_cfg, provider_key="test")
    
    mock_response = MagicMock(status_code=502, headers={}, content=b"")
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    
    start = time.monotonic()
    with pytest.raises(BackendDown):
        backend.complete_with_usage(
            system="", user="test", max_tokens=10, temperature=0.0
        )
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"Gateway retry loop returned: took {elapsed:.1f}s"
```

**Test C — mocked read timeout (peer accepted but hung):**
```python
def test_read_timeout_fails_under_5s(mock_peer_backend_cfg):
    """httpx.ReadTimeout → BackendTimeout at TimeoutPolicy.read_s, no retry."""
    backend = _MaximPeerBackend(cfg=mock_peer_backend_cfg, provider_key="test")
    
    def hang(*args, **kwargs):
        raise httpx.ReadTimeout("peer hung")
    
    mock_client = MagicMock()
    mock_client.post.side_effect = hang
    
    start = time.monotonic()
    with pytest.raises(BackendTimeout):
        backend.complete_with_usage(
            system="", user="test", max_tokens=10, temperature=0.0
        )
    elapsed = time.monotonic() - start
    # Timeout policy read_s = 30s default, but we override for test
    assert elapsed < 5.0
```

**Shared fixture** (conftest.py or top of file):
```python
@pytest.fixture
def mock_peer_backend_cfg():
    """LLMConfig with a short-timeout peer endpoint pre-registered."""
    from maxim.utils.http import register_endpoint, HTTPEndpoint, TimeoutPolicy
    from maxim.models.language.config import LLMConfig
    
    register_endpoint(HTTPEndpoint(
        name="peer-test",
        base_url="http://127.0.0.1:9999/v1",  # unused — http call is mocked
        default_headers={"User-Agent": "maxim-peer/1.0"},
        auth_provider=lambda: "test-key",
        timeouts=TimeoutPolicy(connect_s=1.0, read_s=2.0, total_s=3.0),
    ))
    return LLMConfig(providers={"test": {"base_url": "http://127.0.0.1:9999/v1", "model": "mock"}})
```

**Why mocking httpx directly instead of spinning up a real server:** real servers add port-allocation flakiness, race conditions on startup, and ~100ms-1s of spin-up overhead per test. Mocks are deterministic and complete in <100ms. The claim we're testing is "no internal retry loop" — that's fully testable at the HTTP client boundary.

**What this doesn't test:** actual network timing, real Cloudflare tunnel latency, real llama-cpp-server response shape. Those are covered by the stress test protocol (Phase D).

### Fixture 2: `tests/stress/multi_agent_fan_out.py`

**Purpose:** exercise `LLMRouter._inference_lock` under concurrent agent load. Used by the stress test protocol Phase B. Not part of the fast-failover gate — this is a slow test tagged `@pytest.mark.slow`.

**Structure:**
```python
import threading
import time
import pytest
from maxim.runtime.lane_backends import build_primary_router
from maxim.models.language.router import LLMRouter

@pytest.mark.slow
def test_multi_agent_fan_out_no_starvation(mock_leader_endpoint):
    """5 concurrent agents × 20 requests each = 100 calls.
    
    Each call takes ~200ms on the mocked peer. Under _inference_lock
    serialization, total wall-clock should be roughly 20 seconds
    (100 × 200ms). Test asserts no agent is starved: each agent's
    max wait-time is within 2x the mean.
    """
    router, _mgr = build_primary_router()
    results: dict[str, list[float]] = {}
    lock = threading.Lock()
    
    def agent_worker(agent_id: str, n: int):
        times = []
        for _ in range(n):
            start = time.monotonic()
            router.complete_text(
                system="", user="test", 
                request_context={"agent_id": agent_id, "lane": "large"},
            )
            times.append(time.monotonic() - start)
        with lock:
            results[agent_id] = times
    
    threads = [
        threading.Thread(target=agent_worker, args=(f"agent_{i}", 20))
        for i in range(5)
    ]
    t0 = time.monotonic()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.monotonic() - t0
    
    # Per-agent max latency within 2x of mean → no starvation
    all_times = [t for times in results.values() for t in times]
    mean_latency = sum(all_times) / len(all_times)
    per_agent_max = {aid: max(ts) for aid, ts in results.items()}
    worst = max(per_agent_max.values())
    
    assert worst < mean_latency * 2.0, (
        f"Starvation detected: worst per-agent max={worst:.2f}s vs "
        f"mean={mean_latency:.2f}s (threshold 2x)"
    )
```

**`mock_leader_endpoint` fixture:** similar to fixture 1 but simulates a responsive-but-slow peer (200ms sleep + 200 OK with valid JSON payload). Lives in `tests/stress/conftest.py`.

**Observability during the test:** the fixture enables `MAXIM_LOG_FILE=/tmp/multi_agent_fan_out.jsonl` so wait times can be grep'd out of the JSONL log after the test. That's the data that gets folded into the stress test report Phase B.

**What the test gates:** the decision to ship [deferred/llm_path_async_router.md](deferred/llm_path_async_router.md). If this test fails under Plan 3's `_MaximPeerBackend` (typical latency ~200ms, 5 concurrent agents), the async router deferred plan revives. If it passes, the lock is acceptable and async stays deferred.

### Why these specs live in Plan 3, not earlier

Both fixtures exercise `_MaximPeerBackend` which only exists after Plan 3 R2.5. They can be scaffolded with placeholder imports during Plan 1/2, but the real tests can't run until Plan 3 ships. Document now, implement in Plan 3.

## Hard testing requirement (checkpoint before Plan 4 decision point)

**Non-negotiable.**

**Automated:**
```bash
# Full fast suite
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# Targeted peer backend tests
python -m pytest tests/unit/test_maxim_peer_backend.py -v
python -m pytest tests/unit/test_maxim_peer_backend_streaming.py -v
python -m pytest tests/unit/test_router_exception_integration.py -v

# Performance gate
python -m pytest tests/performance/test_fast_failover.py -v
# Expected: backend_call_duration_seconds p99 < 5s against mocked dead peer

# Mypy + lint + format
mypy src/maxim/models/language/maxim_peer_backend.py --ignore-missing-imports
ruff check src/ tests/
ruff format --check src/ tests/

# Invariant: no retry loop in the new backend
! grep -E "retry|backoff|gateway" src/maxim/models/language/maxim_peer_backend.py

# Invariant: safety-net counter is zero after a clean test run
python -c "
from maxim.models.language.lane_metrics import metrics_snapshot
m = metrics_snapshot()
c = m.get('backend_unclassified_errors_total', {}).get('count', 0)
assert c == 0, f'Safety-net counter non-zero — missed an exception type: {c}'
"

# Multi-agent header propagation
MAXIM_BACKEND_TRACE=1 maxim --sim "quick test" 2>&1 | \
  jq -c 'select(.event=="peer_backend_call")' | \
  python -c "
import sys, json
for line in sys.stdin:
    obj = json.loads(line)
    assert 'agent_id' in obj and 'session_id' in obj and 'request_id' in obj, \
           f'Missing multi-agent fields: {obj}'
"
```

**Then run the stress test protocol (Phases A-E above).**

**Then decide Plan 4 scope.**

## Documentation & memory update (runs after testing passes)

**Load-bearing.** The `_MaximPeerBackend` is a new concept future sessions will wonder about.

**1. Update [../reference.md](../reference.md):**

Expand the "LLM backends" section with:

- **`_MaximPeerBackend`** — purpose-built for self-hosted peers. Lives in `models/language/maxim_peer_backend.py`. Single HTTP call per `complete_with_usage`, raises typed exceptions, no retry loop, uses `utils/http.py`. Supports streaming. Dispatched via `BACKEND_CLASSES` map when `backend_class: "maxim_peer"`.
- **`_OpenAIBackend`** — cloud providers. Internal retry loop + cost tracking + PII redaction. Unchanged by this plan.
- **Decision table:** self-hosted URL → `_MaximPeerBackend`; cloud URL → `_OpenAIBackend`. Configurable via `"backend_class"` field.
- **`health_check()` method** — the consolidated two-stage probe.

**2. Extend [../architecture/llm_routing.md](../architecture/llm_routing.md):**

The architecture doc already exists — it was drafted upfront on 2026-04-12 before any implementation began. Plan 3 extends the following sections with what actually shipped:
- Layer 6 ("The backends"): flip the `_MaximPeerBackend` subsection from `[Target]` → `[Present]`
- "Error taxonomy" section: flip to `[Present]` with final per-class backoff numbers
- "Observability surfaces": append the `backend_call_duration_seconds` + `backend_unclassified_errors_total` metrics to the Plan 3 column
- "Probe lifecycle": flip to `[Present]` once `health_check()` is wired in R2.6

For reference, the doc's full layer structure covers:

```
Caller (agent loop, sim, API) with RequestContext(agent_id, session_id, request_id, lane)
  → FunctionRouter (pick tier: large/medium/small)
    → LaneBackendManager (per-lane backend cache)
      → LLMRouter (provider selection + fallback loop; serializes under _inference_lock)
        → _MaximPeerBackend (self-hosted) OR _OpenAIBackend (cloud)
          → utils/http.py (connection pool, X-Maxim-* headers, metrics)
            → leader_proxy (another node) OR llama-cpp-server (local)
```

For each layer: what it does, what it owns, what it delegates, what typed errors bubble up through it, **what multi-agent context it propagates**.

**Dedicated section: "What crosses node boundaries"** (data sovereignty lens):
- Full request payloads (system + user messages, tools, thinking) travel over HTTPS
- `X-Maxim-*` header contract (full list with stability guarantees)
- What gets logged on the sender side: request_id, agent_id, session_id, lane, status, latency
- What gets logged on the leader side: same + full request body if `MAXIM_HTTP_TRACE=1`
- Retention: logs rotate per the project's log rotation policy (document the rotation config)
- Operator implication: **full prompt content reaches the leader and is logged there.** For multi-tenant deployments, this is a data sharing contract. Document it.

**Dedicated section: "Error taxonomy"** — the `BackendError` hierarchy diagram with per-class backoff policies.

**Dedicated section: "Multi-agent contract"** — `RequestContext` dataclass, contextvar propagation, header protocol stability, logging format.

**This document is the single source of truth.** Future refinements start by updating this doc.

**3. Update [../../CLAUDE.md](../../CLAUDE.md):**

- **Env var table:** add `MAXIM_BACKEND_TRACE`
- **Lessons learned:**
  > **LLM backends are split by upstream type:** `_MaximPeerBackend` for self-hosted peers, `_OpenAIBackend` for cloud providers. Don't merge them. Peer backend is intentionally simple (single HTTP call, no retry) because the router handles failover. Cloud backend intentionally has retry + cost tracking. Patching one to serve the other's use case is fragile — adding a third backend is the right answer if a new upstream type appears.
  > 
  > **Backend failures raise typed exceptions, not return empty content.** The `BackendError` hierarchy lets the router classify failures without string matching. Every new exception class needs a `.fix_hint`. Catching `BackendError` broadly is fine; catching `Exception` is forbidden (see `backend_unclassified_errors_total`).
- **Architectural invariants:**
  > **`_MaximPeerBackend.complete_with_usage()` makes exactly one HTTP call.** No retry loop. No backoff. Failover is the router's job. Do not add `try: ... except: retry` to this backend — it will re-introduce the 52-second retry loop incident from 2026-04-12.
  > 
  > **Multi-agent context (`agent_id`, `session_id`, `request_id`) propagates via `RequestContext` through backends to outbound HTTP headers.** If you add a new backend, it must accept `RequestContext` and set `X-Maxim-*` headers. If you add a new log event in backend code, it must include these fields. The logging formatter pulls from contextvar automatically — if your log line lacks them, you're logging from the wrong context.
- **Quick reference table:** add `_MaximPeerBackend` row

**4. Create [../troubleshooting/peer_backend_debug.md](../troubleshooting/peer_backend_debug.md):**

Runbook (~150 lines):
- `MAXIM_BACKEND_TRACE=1` and what JSONL events to expect
- Reading each typed exception's `fix_hint` → action mapping
- Debugging "peer request takes N seconds" via `backend_call_duration_seconds`
- Checking `backend_unclassified_errors_total` for gaps
- Rolling back to `_OpenAIBackend` via `BACKEND_CLASSES` dispatch
- Per-agent debugging with `jq 'select(.agent_id=="X")'` patterns
- Full typed-exception class → default backoff → root cause table

**5. Update project memory:**

Add `project_llm_path_fast_failover_shipped.md` covering the two-backend decision, the 52s retry loop incident, the streaming design, the multi-agent header contract, and the stress test decision for Plan 4.

**6. Update meta-plan status table.**

**7. Link the stress test report** (`docs/experiments/results/llm_path_stress_<date>.md`) from the meta-plan and Plan 4.

## Migration notes

- `_OpenAIBackend` unchanged; cloud users see no difference
- Self-hosted peer users get automatic migration via `BACKEND_CLASSES` dispatch
- `MAXIM_BACKEND_TRACE=1` is the new debug env var for peer-specific issues
- Rollback is a one-line config change: set `"backend_class": "openai"` to re-enable the old path
- Streaming callers (if any): same API, now works on self-hosted peers too

## Open questions — resolved

1. **Streaming in v1:** **yes, include.** User decision. ~80 LOC extra.
2. **Safety-net counter — warning or CI fail:** **warning now, CI fail after bake-in.** User agreed.
3. **Dispatch table naming:** `BACKEND_CLASSES` dict in `lane_backends.py`. User agreed.

## Related docs

- **Previous plan:** [llm_path_foundation.md](llm_path_foundation.md) — prerequisite
- **Meta plan:** [llm_path_refinement.md](llm_path_refinement.md)
- **Previous plan:** [llm_path_typed_errors.md](llm_path_typed_errors.md) — Plan 2 (prerequisite)
- **Next plan:** [llm_path_operator_visibility.md](llm_path_operator_visibility.md) — Plan 4 (formerly reactive mesh, scope-reduced)
- **Deferred:** [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) — the multi-peer overflow we moved out of Plan 4
- **Architecture:** [../architecture/llm_routing.md](../architecture/llm_routing.md) — created by this plan
- **Project guide:** [../../CLAUDE.md](../../CLAUDE.md)
- **Related incident:** `_OpenAIBackend` retry-loop math from [openai_backend.py:278-374](../../src/maxim/models/language/openai_backend.py#L278-L374)
