# LLM Timeout Scalability — bigger models on slower hardware through tunnels

**Status:** Shell plan, drafted 2026-06-03. Triggered by a real Mac Mini incident: qwen2.5-32b on growing context hit a 149s TTFT, broke the proxy pipe, and surfaced three independent timeout layers (client SDK, httpx read_s, cloudflared tunnel idle).
**Scope:** Stage 1 ~30 LOC audit-fix; Stage 2 ~80 LOC config schema extension; Stage 3 ~120 LOC proxy heartbeat; Stage 4 ~250 LOC adaptive throughput model + persistence.
**Target versions:** Stage 1 → 1.0 (bug fix). Stage 2 → 1.0 (extends shipped config-unification surface). Stage 3 → 1.0 or 1.0.x (proxy-side, tunnel-defense). Stage 4 → 1.1 (adaptive, research-y).
**Gates:** Stages 1+2+3 gate themselves on "does this break self-hosted leaders running models >24B?" — yes today. Stage 4 is not a 1.0 release gate.
**Driving incident:** 2026-06-02. Mac Mini leader serving qwen2.5-32b. Four consecutive `/v1/chat/completions` requests with growing latency (13s → 44s → 50s → 149s broken-pipe). Final request: `chunks=1 elapsed_ms=149182` followed by `[Errno 32] Broken pipe` on the proxy write-back. Upstream returned 200 OK at 149s but the client was already gone.

**Depends on:**
- [`utils/http.py::TimeoutPolicy`](../../src/maxim/utils/http.py) — existing per-endpoint timeout dataclass; `read_s` controls httpx idle window
- [`models/language/maxim_peer_backend.py::_get_timeout_policy`](../../src/maxim/models/language/maxim_peer_backend.py) — already reads `cfg.get("timeout_s", 300.0)`; default is fine, plumbing isn't
- [`models/language/openai_backend.py::_get_timeout`](../../src/maxim/models/language/openai_backend.py) — defaults to 60s blanket SDK timeout; needs per-tier override
- [`runtime/lane_backends.py::_classify_backend`](../../src/maxim/runtime/lane_backends.py) — routes `"self-hosted"` → `_MaximPeerBackend`, `"cloud"` → `_OpenAIBackend`; correct logic, may have a misclassification bug surfacing the incident
- [`runtime/config_loader.py::LaneTierConfig`](../../src/maxim/runtime/config_loader.py) — frozen dataclass shipped in PR #318; `timeout_s` would be an additive field via the `extra` escape hatch or a new declared field
- [`runtime/leader_proxy.py`](../../src/maxim/runtime/leader_proxy.py) — the proxy that needs TTFT heartbeats; `_INFERENCE_PROXY_TIMEOUT_S = 300.0` is the upstream-side ceiling
- [`utils/atomic_io.py::atomic_write_json`](../../src/maxim/utils/atomic_io.py) — for Stage 4 persistence to `~/.maxim/util/`
- [`peer/drain_state.py`](../../src/maxim/peer/drain_state.py) — `filelock.FileLock` RMW pattern for the throughput-stats persistence

**Enables:**
- Self-hosted leaders running 30B+ models (qwen2.5-32b, mixtral-8x7b, llama-3.1-70b) on edge hardware (Mac Mini, Jetson, etc.)
- Reachy Mini app + similar embodiment peers tunneling through cloudflared (Stage 3 specifically — tunnel-idle defense is non-negotiable for tunneled deployments)
- Operator-explicit per-tier timeout tuning via `config.json::lanes.<tier>.timeout_s` (Stage 2)
- Self-calibrating timeouts per `(machine, model)` (Stage 4) — operator doesn't need to know what TTFT or tok/sec to expect; system learns

---

## Front-gate scope pressure (Principle 3)

**Question:** does this need to be its own mechanism, or can it ride on existing infrastructure?

| Candidate | Why insufficient (or sufficient) |
|---|---|
| Just bump `TimeoutPolicy.long().read_s` from 120 → 300 | **Insufficient.** Solves the median case but (a) doesn't address `_OpenAIBackend`'s 60s SDK-level cap that the failing path actually hit, (b) doesn't address cloudflared's ~100s tunnel idle window, (c) wastes resource on small-tier completions that should fail fast. The constant exists; the per-tier overhead doesn't. |
| Per-tier `timeout_s` in `config.json` | **Sufficient as Stage 2.** `LaneTierConfig` is already shipping in PR #318 with an `extra` escape hatch (CC3 path-a). Adding `timeout_s: float \| None = None` as a declared field is non-breaking. Both backends already read from `_provider_cfg()`, so the plumbing is a one-line lookup at the read site. |
| TTFT heartbeat in proxy | **Yes-needs-own.** No existing mechanism. The proxy currently relays exactly what upstream sends; if upstream sends nothing for 100s, the tunnel closes. Sending a periodic SSE comment (`: keepalive\n\n`) during the silent window is a small but distinct mechanism. The "single HTTP call" invariant on `_MaximPeerBackend` is about client-side retry, not server-side keepalive — those are orthogonal. |
| Adaptive throughput model | **Yes-needs-own, deferred to Stage 4.** Welford or EWMA running stats per `(tier, model)` persisted to `~/.maxim/util/lane_throughput.{tier}.{model}.json`. Reuses the `~/.maxim/util/` mutable-state-layer + `filelock` pattern from Plan 4 C2 + the `atomic_write_json` writer. No new infrastructure; just a new data type using existing patterns. |

**Verdict:** Stage 1 is an audit-fix; Stage 2 is a small additive field; Stage 3 is a small but distinct new mechanism (proxy keepalive); Stage 4 is a research-y addition that reuses existing persistence patterns. None of the four warrant a "new bus" or "new abstraction" — all four ride on infrastructure that's already shipped or being shipped.

---

## Why land Stages 1+2 in 1.0 (vs. 1.0.x)

The 1.0 release ships a config-unification layer (`config.json`) whose stated motivation includes "absorbing operator-visible knobs onto one source of truth." Per-tier `timeout_s` is exactly that — operators with self-hosted leaders running big models need to tune it, and right now they can't. Shipping 1.0 with `MAXIM_LANE_LARGE_REMOTE_URL` honored but no per-tier timeout knob means the first big-model operator hits the wall on day one (we know this; the Mac Mini incident just did).

Stage 1 is a bug fix — `_OpenAIBackend` getting picked for self-hosted leaders is a misclassification that violates the Plan 3 R2.5 invariant. That's a 1.0 blocker, not a 1.0.x cleanup.

Stage 3 is the only stage with genuine ambiguity. The argument for 1.0: cloudflared-tunneled peers are a documented setup pattern, and a 100s TTFT cap silently breaks them. The argument for 1.0.x: it's a proxy-internal change that doesn't affect the public API or storage format, so deferring doesn't lock in any contract. Lean 1.0 if the implementation lands clean; 1.0.x if Stage 1+2 are tight on time.

Stage 4 is post-1.0 because it requires real-world observation data to validate the adaptive model (cold-start behavior, convergence rate, multi-model bucket collisions). Premature shipment would force a wire-format on the throughput-stats file that we'd then have to migrate.

---

## Failure-mode anatomy (the three-layer timeout problem)

The 2026-06-02 incident exposes that "LLM call timeout" is actually **three orthogonal layers**, each with its own failure mode:

| Layer | Where it lives | Default | What fires it | Recovery |
|---|---|---|---|---|
| Client SDK timeout | `openai` SDK or `httpx.Timeout` | 60s (OpenAI) / 120s (httpx via `TimeoutPolicy.long`) | Total time from request start to any new byte received exceeds the configured budget | Client closes the socket → BrokenPipe at proxy |
| Tunnel idle timeout | cloudflared | ~100s | No bytes flow on the tunnel for the idle window | Cloudflared closes the request → 502 at peer |
| Upstream-side timeout | `_INFERENCE_PROXY_TIMEOUT_S` (300s) | 300s | Proxy's read from upstream llama-cpp-server exceeds budget | Proxy returns 502 with typed error |

**The Mac Mini incident hit Layer 1** (`_OpenAIBackend` 60s × retry stacked to ~150s) **and would have hit Layer 2 next** (cloudflared 100s) **with any TTFT >100s on a non-streaming or slow-TTFT response.** Layer 3 was fine — `_INFERENCE_PROXY_TIMEOUT_S = 300s` is generous.

Each stage targets a specific layer:
- Stage 1 — fixes the misclassification routing Layer 1 to `_OpenAIBackend` (60s) instead of `_MaximPeerBackend` (300s)
- Stage 2 — makes Layer 1 operator-configurable per-tier
- Stage 3 — defeats Layer 2 by emitting heartbeats during the TTFT window
- Stage 4 — auto-tunes Layer 1 based on observed `(TTFT, tok/sec)` per `(tier, model)`

---

## Stage 1 — Backend misclassification audit  *(COMPLETE — no code change needed)*

**Original hypothesis:** the Mac Mini's `large` lane was classified as `cloud`, routing `_classify_backend("cloud") → "openai"` and instantiating `_OpenAIBackend` with its 60s default. The 149s observation would then match `60s × (1 + 2 SDK retries) ≈ 150s`.

**Audit run (2026-06-03):** simulated the peer's CLI startup state (`detect_and_apply_role` + `_apply_lane_config_to_env`) on the same Mac that produced the failing 149s request. Result:

| tier | url | source | classification | backend |
|---|---|---|---|---|
| `large` | `https://maxim.big-mac-mini.org/v1` | env (from peer.yml compat-read) | self-hosted | `_MaximPeerBackend` |
| `medium` | None | default | cloud | `_OpenAIBackend` |
| `small` | None | default | cloud | `_OpenAIBackend` |

`large` resolves correctly to `_MaximPeerBackend`. The hypothesis is **falsified**. `_MaximPeerBackend._get_timeout_policy()` reads `cfg.get("timeout_s", 300.0)` → 300s read timeout. A 149s broken-pipe is below that ceiling, so it's not a client SDK timeout firing.

**What this means for the diagnosis:** the failure was at **Layer 2 (cloudflared tunnel idle)**, not Layer 1 (client SDK). Cloudflared's idle window closed the tunnel mid-TTFT; the proxy didn't notice until it tried to write the response body and got EPIPE (TCP write returns EPIPE only when there's actual data to push). The 100-150s observed range is consistent with cloudflared's free-tier idle timeout + TLS close drain.

**Code change required:** none. The audit closes Stage 1 without a fix.

**Promoted to most urgent fix:** Stage 3 (TTFT heartbeats) — it's the only stage that addresses Layer 2 directly.

---

## Stage 2 — Per-tier `timeout_s` in `config.json`  *(SHIPPED 2026-06-04)*

**Schema extension:** `LaneTierConfig` gains a declared field `timeout_s: float | None = None`. `None` → backend default (`_MaximPeerBackend` 300s, `_OpenAIBackend` 60s). Operators set it explicitly:

```json
{
  "lanes": {
    "large":  { "remote_url": "http://127.0.0.1:8100", "timeout_s": 600 },
    "medium": { "timeout_s": 180 },
    "small":  { "timeout_s": 60 }
  }
}
```

**Backend plumbing — completed:** the lane → provider config bridge in `lane_backends.py:985-996` now threads `cfg.remote_timeout_s` into `providers[provider_key]["timeout_s"]` when set. Both `_MaximPeerBackend._get_timeout_policy()` and `_OpenAIBackend._get_timeout()` already read this key via `_provider_cfg().get("timeout_s", <default>)` per the Plan 3 R2.5 contract — no backend-side wiring change needed.

**Env-var counterpart for back-compat:** `MAXIM_LANE_<TIER>_TIMEOUT_S`. Wired into `_FIELD_TO_ENV` with float coercion + strict-positive validation in `_coerce_for_field`. `_apply_lane_config_to_env` populates the env var from `config.json`; `apply_lane_env_overrides` in `lane_models.py` reads it into `LaneConfig.remote_timeout_s` with defensive silent-ignore on malformed/non-positive values (loader-side raises `ConfigurationError`; this is the bypass-the-loader defence).

**`maxim doctor` integration:** the "Resolved Config" section automatically grows a row per tier (`lanes.large.timeout_s`, `lanes.medium.timeout_s`, `lanes.small.timeout_s`) showing the effective value + source. Free from PR #318's `_FIELD_TO_ENV` walk.

**Regression guard:** [tests/unit/test_config_loader.py::TestLaneTierTimeoutField](../../tests/unit/test_config_loader.py) — 13 tests pinning field validation (zero/negative/non-numeric rejected), JSON parser (string/bool rejected, null accepted), env coercion (positive pass-through, malformed/zero/negative raise), `resolve_setting` precedence (env vs default). [tests/unit/test_leader_proxy.py::TestLaneTimeoutFieldFlow](../../tests/unit/test_leader_proxy.py) — 4 tests pinning env passthrough into `LaneConfig.remote_timeout_s`.

---

## Stage 3 — TTFT heartbeat in leader proxy

**Problem:** cloudflared's ~100s idle timeout closes any request that has no bytes flowing for the idle window. Streaming responses are immune **once tokens start flowing** (~40ms inter-token latency), but during TTFT (prompt evaluation) no bytes flow. For qwen2.5-32b on Mac Mini with a 4K-token prompt, TTFT can be 100s+.

**Mechanism:** while the proxy is waiting on upstream's first chunk, emit an SSE comment frame every ~30s to the client. Per [SSE spec](https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream), lines beginning with `:` are comments — clients ignore them but the bytes count as activity for tunnel-idle purposes. Format: `: keepalive\n\n`.

**Where:** [`runtime/leader_proxy.py::_proxy_request`](../../src/maxim/runtime/leader_proxy.py) wraps the upstream call via [`utils/http.py::raw_proxy_forward_streaming`](../../src/maxim/utils/http.py). The keepalive logic lives in the streaming-forward path:
1. Start a timer at request start
2. While reading from upstream: if no bytes received in the last 30s AND no chunks have been forwarded yet, write `: keepalive\n\n` to the client
3. Stop emitting heartbeats as soon as the first real chunk forwards (streaming generation has its own activity)

**Gate:** only for `/v1/chat/completions` and `/v1/completions` with `Accept: text/event-stream` or `stream: true` in the request body. Non-streaming requests (`stream: false`) are buffered end-to-end and there's nothing to interleave heartbeats into — those are explicitly out of scope for this stage; operators using `stream: false` against tunneled leaders need to either (a) set `stream: true`, (b) keep TTFT under cloudflared's idle window, or (c) raise the cloudflared idle timeout server-side.

**Knob:** `MAXIM_PROXY_KEEPALIVE_INTERVAL_S` (default 30, clamped 5-90).

**Regression guard:** new test in `tests/unit/test_leader_proxy.py` using a mock upstream that delays its first chunk by 60s — assert that the client receives at least 1 keepalive frame before the real chunk arrives. Second test pinning that buffered (`stream: false`) requests do NOT receive keepalives.

**Out-of-scope alternatives considered + rejected:**
- Switch transport to gRPC / WebSocket — too disruptive; would invalidate the OpenAI-API-compatible wire shape that everything else uses.
- Disable cloudflared's idle timeout — server-side config; not all operators have access; doesn't generalize.
- Use Cloudflare Workers / Cloudflare Tunnel WebSocket mode — vendor lock-in; not all peers tunnel via Cloudflare.

---

## Stage 3.5 — Proxy context-overflow admission  *(SHIPPED 2026-06-04)*

**Surfaced by the Stage 3 heavy-stress validation** (2026-06-03 evening). A 9534-token prompt against an 8192-token `n_ctx` was held alive by 5 keepalive frames over 150s — Stage 3 did its job — but llama-cpp-server eventually aborted generation with `INTERNAL_ERROR`. The agent got no actionable error; the operator had to read llama-cpp logs to diagnose context overflow.

**Problem class:** oversize prompts that exceed the upstream model's context window. The upstream silently aborts mid-generation, the stream limps along with keepalives until something cuts the connection, and the failure surfaces as a generic stream interruption rather than a typed "you're asking for more than I can hold" error.

**Fix:** admission control at the proxy. Before forwarding to upstream, estimate input tokens and compare against the resolved context window. Reject overflowing requests with HTTP 413 + a typed JSON error carrying the numbers the operator needs (`estimated_prompt_tokens`, `max_tokens`, `context_window`, `safety_overhead_tokens`).

**Implementation:**
- `_PROXY_INFERENCE_PATHS` = `{/v1/chat/completions, /v1/completions}` — admission only gates inference endpoints; `/v1/models`, `/debug/ping`, etc. pass through unchecked.
- `_resolve_proxy_context_window()` reads through `resolve_setting("llm.n_ctx", config=cfg)` — env (`MAXIM_LLM_N_CTX`) > `config.json::llm.n_ctx` > default-unset. Returns `None` when unresolvable.
- `_get_proxy_context_window()` wraps the resolver in a thread-safe `(resolved, value)` tuple cache + `threading.Lock`. Logs INFO on first resolved value, WARNING on first None (admission stays off).
- `_estimate_inference_input_tokens(body)` parses JSON, sums string `content` (and text parts of multipart content) across messages, adds 16 chars per message for chat-template overhead, divides by `_PROXY_CHAR_TO_TOKEN_RATIO = 3.5`. Also handles legacy `/v1/completions` with string-or-list `prompt`.
- `_check_context_admission(body, n_ctx, overhead)` compares `prompt_tokens + max_tokens (or 1024 default) + overhead` against `n_ctx`. Returns `None` (admit) or an error dict (reject). Malformed bodies return None (let upstream return a cleaner 400).
- `_proxy_request` calls the gate after reading body, before forwarding. Rejection sends 413 directly and returns.

**Knobs:**
- `MAXIM_PROXY_CONTEXT_ADMISSION` (default ON when `n_ctx` resolves; explicit OFF via `0`/`false`/`no`/`off`)
- `MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS` (default 256, clamped 0-4096)

**Graceful default-on:** existing operators with no `MAXIM_LLM_N_CTX` set get a startup WARNING and admission stays off (existing behaviour preserved). Operators who set `MAXIM_LLM_N_CTX` get the gate automatically.

**Why character-based (not tiktoken / not llama-cpp `/tokenize`):**
- tiktoken adds a dependency and is inaccurate for Qwen/Llama tokenizers (~10-30% off)
- llama-cpp `/tokenize` is precise but adds an HTTP round-trip per request
- The admission gate is a safety net, not a precise predictor — character-based with a generous overhead margin catches 10×-over prompts cleanly without complexity

**`maxim doctor` integration:** `_check_proxy_context_admission` renders the gate's effective state in the "Resolved Config" section: `enabled [context_window=8192, overhead=256 tokens, char_to_token_ratio=3.5]` or `disabled [reason=llm.n_ctx not configured]` with a `fix:` line directing the operator to set `MAXIM_LLM_N_CTX`.

**Regression guard:** [tests/unit/test_leader_proxy.py::TestContextOverheadResolver / TestAdmissionEnableGate / TestInputTokenEstimator / TestAdmissionCheck / TestContextWindowResolver](../../tests/unit/test_leader_proxy.py) — 28 tests pinning env parsing/clamping, gate logic, estimator across body shapes (chat completion / multipart / legacy completions / empty / malformed / large), OpenAI-compatible error envelope, and cache thread-safety.

---

## Stage 4 — Adaptive throughput model (per-machine, per-model)

**Concept (user's question, 2026-06-03):** measure `(TTFT, tok/sec)` per `(tier, model)` during normal operation, persist running stats, use them to compute an expected total-time-to-completion that drives the read timeout. Operator doesn't need to know what timeout to set; system learns.

**Three design refinements (user, 2026-06-03):**
1. **Error-proportional convergence** — accelerate the running average toward the true value when the prediction is far from observed reality. Slow updates when predictions are accurate, fast updates when they're way off.
2. **Conservative cold-start** — start with a high initial timeout for new `(tier, model)` buckets so the first few calls cannot false-timeout. The cost of an over-generous prediction is delayed errors (small); the cost of a too-tight prediction is user-visible failures (large). Asymmetric cost → asymmetric prior.
3. **Parameter-size bootstrap** — derive a sane cold-start prediction analytically from `model_param_count_B × hardware_class_constant`, so the very first call already has a reasonable budget. The adaptive loop refines from there.

### Two measurements, not one

- **TTFT** is prompt-eval-bound, scales roughly linearly with prompt token count: `expected_TTFT_s = ttft_intercept + ttft_slope * prompt_tokens`. Linear regression on observed pairs.
- **Generation rate** is near-constant per `(model, hardware)`: `expected_gen_s = max_output_tokens / observed_tok_per_sec`. Adaptive EWMA over recent completions.
- **Total expected completion:** `expected_TTFT + expected_gen + k * residual_std_dev` where `k=3` covers ~99% of completions assuming roughly-normal residuals.

### Hardware-class generalisation (covers refinement 3)

The cold-start prior is dimensionally:

```
expected_TTFT_s   = prompt_tokens / prompt_eval_tok_per_sec(hardware, model_B)
expected_gen_s    = output_tokens / generation_tok_per_sec(hardware, model_B)

where:
  prompt_eval_tok_per_sec = HW_PROMPT_EVAL_RATE[hardware_class] / model_B
  generation_tok_per_sec  = HW_GEN_RATE[hardware_class] / model_B
```

`hardware_class` is one of `{cpu, apple_silicon_unified, consumer_gpu, datacenter_gpu}`, detected via `psutil.cpu_count` + `torch.cuda.is_available()` + Apple Silicon check. `model_B` is parameter count in billions, extracted from the profile name regex (`r"(\d+(?:\.\d+)?)\s*[Bb]"` matches `mistral-7b`, `qwen2.5-32b`, etc.) with a GGUF metadata fallback for ambiguous cases.

`HW_PROMPT_EVAL_RATE[hardware_class]` and `HW_GEN_RATE[hardware_class]` are seeded with literature/benchmark values (rough order: `apple_silicon_unified` ~400 tok/sec/B for generation, ~5000 tok/sec/B for prompt-eval; `consumer_gpu` ~3-5× faster; `cpu` ~10× slower) and **then refined by every observation across all models on the same hardware class**. The first qwen2.5-32b run on Mac Mini informs the cold-start prediction for the next mistral-small-24b run on the same Mac Mini, because they share the hardware-class constant.

This is the load-bearing piece for refinement 3: predictions improve *across* models, not just *within* a single model bucket.

### Conservative cold-start (covers refinement 2)

For a brand-new `(tier, model)` bucket with zero samples:

```
cold_start_timeout_s = max(
    backend_default,                    # 300s for _MaximPeerBackend
    2 * parameter_based_estimate,       # 2× the analytical prediction
    _COLD_START_FLOOR_S,                # absolute floor (e.g. 120s)
)
```

The `2×` multiplier on the analytical prediction is the "guaranteed lock" the operator asked for — even if the hardware-class constants are pessimistic by a factor of 2, the first call won't false-timeout.

The cold-start budget shrinks as samples accumulate:

```
n = sample_count(tier, model)
trust = 1 - exp(-n / 5)                 # ranges from 0 (cold) → ~1 (warm at n=20)
effective_timeout = trust * adaptive_prediction + (1 - trust) * cold_start_timeout
```

At `n=0`, `trust=0`, full cold-start budget. At `n=20`, `trust≈0.98`, almost-pure adaptive prediction. Smooth crossover.

### Error-proportional convergence (covers refinement 1)

Standard EWMA: `new = (1 - α) * old + α * obs`, fixed `α`. Problem: a single bad prior + 20 normal observations leaves the estimate dragged toward the bad prior for a long time.

Adaptive EWMA: scale `α` by the relative error of the current prediction:

```
relative_error = abs(observed - predicted) / max(predicted, 1.0)
α_effective = clamp(
    base_α * (1 + error_gain * relative_error),
    min_α,
    max_α,
)
new_estimate = (1 - α_effective) * old_estimate + α_effective * observed
```

With `base_α = 0.1`, `error_gain = 2.0`, `min_α = 0.05`, `max_α = 0.5`:
- Observation matches prediction (`error ≈ 0`) → `α ≈ 0.1`, slow update
- Observation 50% off (`error = 0.5`) → `α ≈ 0.2`, faster update
- Observation 2× off (`error = 1.0`) → `α ≈ 0.3`, much faster
- Observation 5× off (`error = 4.0`) → `α = 0.5` (clamped), maximum trust-shift

This gives the "accelerate to the actual number the further it is" property without going fully Bayesian. The clamp prevents a single catastrophic outlier from fully replacing the estimate.

**Reset trigger** as a safety net: if 3 consecutive observations are all >2σ above current mean, treat as model/hardware change and snap the estimate to the median of the 3.

### Where the data lives

`~/.maxim/util/lane_throughput.{tier}.{model}.json` — one file per `(tier, model)` bucket. Schema:

```json
{
  "_format_version": "1.0",
  "tier": "large",
  "model": "qwen2.5-32b",
  "hardware_class": "apple_silicon_unified",
  "model_params_b": 32.0,
  "ttft": {
    "intercept_s": 12.0,
    "slope_s_per_tok": 0.035,
    "residual_std_s": 2.1,
    "n": 47
  },
  "generation": {
    "ewma_tok_per_sec": 21.3,
    "residual_std": 1.8,
    "n": 47
  },
  "completed_at": "2026-06-03T14:23:00Z"
}
```

Hardware-class constants live in a separate `~/.maxim/util/hardware_throughput.json` shared across all buckets on the same machine (this is the cross-model generalisation surface). Writes go through `atomic_write_json` + `with_format_version` under a `filelock.FileLock` — same pattern as `drain_state.py` from Plan 4 C2. Reads tolerate file absence (cold-start) and unknown future fields.

### Where measurement happens

The `_MaximPeerBackend.complete_with_usage` streaming path already knows:
- Request start timestamp
- First-chunk-received timestamp (→ TTFT)
- Last-chunk-received timestamp + total output tokens (→ tok/sec)
- Prompt token count (from usage block)

Recording is one function call at request end: `record_throughput_observation(tier, model, prompt_tokens, ttft_s, output_tokens, generation_s)`. ~50 LOC for the recorder; ~120 LOC for the adaptive-EWMA + linear regression + hardware-class update math; ~60 LOC for the persistence layer; ~80 LOC of tests covering cold-start, warm-up, error-proportional convergence, and reset trigger.

### Where the prediction is consumed

Stage 2's `timeout_s` resolution gets a new precedence rank inserted between "config.json" and "backend default":
1. CLI flag (if added)
2. Env var (`MAXIM_LANE_<TIER>_TIMEOUT_S`)
3. `config.json::lanes.<tier>.timeout_s`
4. **Adaptive prediction** (if not disabled — operator gets analytical cold-start at n=0, blended cold-start + adaptive at small n, pure adaptive at large n)
5. Backend default

The adaptive prediction is the smart floor, not a ceiling — operator's explicit `timeout_s` always wins.

### Disable knob

`MAXIM_ADAPTIVE_TIMEOUT_DISABLED=1`. Useful for debugging timeout issues — operator wants to know if it's the adaptive model misbehaving or the actual upstream.

### Open questions (need answering before Stage 4 starts)

- **Hardware-class auto-detection accuracy:** does `psutil` + `torch.cuda.is_available()` + Apple Silicon check produce stable hardware-class labels across Linux/macOS/Windows? Need a smoke test on each platform.
- **Hardware-class constants:** what are the right initial values for `HW_PROMPT_EVAL_RATE` and `HW_GEN_RATE` per class? Plan: seed with conservative values from the literature, let the first 50-100 observations on each hardware class refine them. Document the cold-start values in the module docstring so operators understand the prior.
- **Bucket cap:** the per-`(tier, model)` bucket directory grows unbounded over time as operators try different models. Add an LRU cap (~20 buckets) with on-disk rotation. The hardware-class file is unbounded by design but it's a single file.
- **Cross-machine pollution:** if `~/.maxim/util/` is on a network share (rare but possible), two machines writing the hardware-class file could corrupt each other's estimates. Add a `machine_id` field (MAC address hash) to the hardware-class entries and partition.
- **Cloudflared interaction:** Stage 4 fixes Layer 1 (client SDK timeout) but doesn't touch Layer 2 (tunnel idle). Operators running through tunnels still need Stage 3. The two stages are complementary, not alternative.

### Regression guard

New test in `tests/unit/test_adaptive_timeout.py` pinning:
- (a) Cold-start with zero samples returns `max(backend_default, 2 * analytical_estimate, floor)`
- (b) Hardware-class constants are seeded from defaults when the shared file is absent
- (c) Adaptive EWMA accelerates `α` proportionally to relative error
- (d) Reset trigger fires after 3 consecutive >2σ observations
- (e) Operator's explicit `timeout_s` overrides prediction
- (f) `MAXIM_ADAPTIVE_TIMEOUT_DISABLED=1` disables the prediction layer
- (g) Parameter-size regex correctly extracts `B` count from `mistral-7b`, `qwen2.5-32b`, `llama-3.1-70b-instruct`, etc.
- (h) Hardware-class file persists + reloads losslessly across runs

---

## Roll-out order  *(updated 2026-06-04)*

1. **Stage 1 (DONE 2026-06-03):** Audit complete. `_MaximPeerBackend` was correctly bound on the failing path; no misclassification. Closed without code change.
2. **Stage 3 (DONE 2026-06-03, PR #320):** Proxy TTFT keepalive emitter. The fix for the Mac Mini incident's Layer 2 (cloudflared tunnel idle) failure mode. Validated end-to-end with a 150s silent TTFT carried by 5 keepalive frames.
3. **Stage 3.5 (DONE 2026-06-04, this PR):** Proxy context-overflow admission. Closes the failure-mode gap surfaced by the Stage 3 validation — oversize prompts now get a clean 413 instead of silently hanging upstream.
4. **Stage 2 (DONE 2026-06-04, this PR):** `LaneTierConfig.timeout_s` field + env var + backend plumbing + doctor row. Operators tuning custom timeouts now have an explicit per-tier knob.
5. **Stage 4 (1.1):** Adaptive throughput model with error-proportional convergence, conservative cold-start, and parameter-size bootstrap. ~1-2 weeks including the open-questions resolution + on-disk format design + soak validation.

Stages 1-3.5 unblock self-hosted leaders running 30B+ models through tunnels with clean error paths for oversize requests. Stage 4 is the calibrated-by-simulation flavor of the same problem and pairs naturally with [`decay_consolidation_calibration_plan.md`](decay_consolidation_calibration_plan.md)'s calibration discipline (different domain, same "measure the system, don't hand-pick" philosophy).
