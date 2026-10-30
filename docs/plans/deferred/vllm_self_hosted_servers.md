# vLLM (and any self-hosted OpenAI-compatible server) — making it a first-class lane

> **DEFERRED 2026-10-18** (filed into `deferred/` from an untracked 2026-09-27 draft). **Revive when:** any OpenAI-compatible self-hosted server is put behind a lane, or a vLLM host is stood up. Three findings are not vLLM-specific and are filed now: D5 → [#1369](https://github.com/dennys246/Maxim/issues/1369), D6 → [#1370](https://github.com/dennys246/Maxim/issues/1370), the stale env-key item (§Not in scope) → [#1371](https://github.com/dennys246/Maxim/issues/1371).

**Status:** Drafted 2026-09-27 from a four-lens parallel review (dispatch path · context/config/doctor ·
leader/peer lifecycle · verification/guardrails). Not scheduled. Blockers were re-verified against
`main` @ `93a5daab` by the author after the review.
**Motivating question:** can an operator point Maxim at a `vllm serve` endpoint and get correct,
observable behaviour — right model, right context budget, typed failures — without Maxim taking a
dependency on vLLM?
**Adjacent:** [docs/agents/llm-routing.md](../../agents/llm-routing.md) §6 (n_ctx drift, cross-process
reuse leg) · [docs/lessons/sim-n-ctx-config-drift.md](../../lessons/sim-n-ctx-config-drift.md) ·
[docs/lessons/no-harness-on-leader-machine.md](../../lessons/no-harness-on-leader-machine.md).

## Motivation

vLLM gives continuous batching (several concurrent requests share one forward pass) and faster
serving of larger models on CUDA. Maxim's workload — multiple lanes, multiple sims, peers behind a
leader — is the shape batching helps. vLLM is Linux/CUDA-only and pins its own torch, so it must be
an **operator-run server Maxim connects to, never a pip extra**.

The router already names vLLM as an intended self-hosted target (`router.py` cloud-gate comments,
`lane_backends.py` self-hosted classification). But today, a default vLLM server behind a lane
**silently does not get used**: the startup probe drops the lane and Maxim falls back to a local
llama server with only a WARNING. That is the thing this plan fixes.

## What exists (verified against main)

- **Routing is already right.** A private-IP `lanes.<tier>.remote_url` → `derive_placement` origin
  PEER → `_classify` "self-hosted" → `_classify_backend` → `_MaximPeerBackend` (single HTTP call,
  300 s read timeout, typed errors, router-level failover, `$0` cost, no cloud gate/redaction). This
  is the correct backend for vLLM; `_OpenAIBackend`'s 60 s timeout and in-backend retries are not.
- **No llama.cpp-only request features are used on the peer path**: no GBNF, `response_format`,
  logprobs, `cache_prompt`, slots or native `tools`. Tool calls are prompt-described JSON parsed by
  `json_parser.py`, so vLLM needs no `--enable-auto-tool-choice`. Chat templating is server-side.
- **Config already has the slots**: `LaneTierConfig.remote_url/remote_model/remote_api_key_ref/
  timeout_s`, `LaneTierPlacement` (+ CC3 `extra`). No new provider-type key is needed.

### Defects found (each verified in source)

| # | Sev | Where | What happens with vLLM |
|---|---|---|---|
| D1 | BLOCKER | `lane_backends.py::_validate_remote_urls` → `_MaximPeerBackend.for_url(url, api_key=…)` (no `model`) → `maxim_peer_backend.py::_probe_stage2_readiness` sends `"model": "default"` | vLLM 404s → classified `inference_broken` → `remote_url` set to `None` → lane silently falls back / auto-spawns llama.cpp |
| D2 | BLOCKER | `lane_backends.py::_build_remote_backend` model pick `primary.model or cfg.remote_model or cfg.model_profile or base.model` | Maxim profile name (e.g. `qwen2.5-14b`) is sent; vLLM 404 → `BackendModelMissing` → 60 s cooldown. `_try_discover_served_model` knows the real id but uses it only for display/cost |
| D3 | BLOCKER (drift class) | `_build_remote_backend` uses `base.n_ctx` unless `served_n_ctx` is stamped, which only auto-spawn does | Remote lane budgeted at the **global profile's** n_ctx, unrelated to the lane's model. Budget > `--max-model-len` → every large prompt 400s |
| D4 | SHOULD-FIX | `leader_proxy.py::_current_llama_server_n_ctx`, `doctor/checks.py::check_context_window` (two copies of the parse; raw sockets to `127.0.0.1`, bypassing `utils/http.py`) | Read `context_length/n_ctx/max_context_length`, never `max_model_len`; vLLM's window is invisible |
| D5 | SHOULD-FIX | 400 context overflow → `http.py::_classify_status` (body discarded) → generic `BackendError(400)` → router backoff | A request-shaped fault **cools down a healthy provider**; the next, smaller prompt is refused too. Looks like the old 500 cascade. Filed as [#1369](https://github.com/dennys246/Maxim/issues/1369) |
| D6 | SHOULD-FIX | `maxim_peer_backend.py::_stream_response` reads only `choices`/`usage` | vLLM's mid-stream `data: {"error": …}` → partial text returned as success; breaks the brief's strict mid-stream-fail contract (affects any OpenAI-compatible server, not just vLLM). Filed as [#1370](https://github.com/dennys246/Maxim/issues/1370) |
| D7 | SHOULD-FIX | `leader_proxy.py::start_leader_proxy` `upstream_url = http://127.0.0.1:{port}` | A leader cannot front a vLLM on another (Linux) box |
| D8 | SHOULD-FIX | `lane_backends.py::swap_llm_server` with `_active_spawner is None` | `peer llm <model>` starts llama.cpp on :8100, colliding with an operator-run vLLM |
| D9 | SHOULD-FIX | `llm_server.py::_served_model_matches` | HF id `Qwen/Qwen2.5-14B-Instruct-AWQ` ≠ profile name → `check_existing_llm_server` raises |
| D10 | SHOULD-FIX | doctor: no check iterates self-hosted `lanes.<tier>.remote_url` on leader/solo; `check_llama_cpp_server_installed` FAILs "no inference path" on a vLLM-only leader | No preflight for the one thing most likely to be misconfigured |
| D11 | NIT | `cost_tracker.py::get_pricing` exact-match, ignores `treat_as_cloud` | A `--served-model-name` that collides with a priced key (`gpt-4o-mini`) bills self-hosted calls |
| D12 | NIT | `leader_proxy` debug status / VRAM projection | `llm_model` "unknown", GGUF VRAM math meaningless under vLLM's preallocation |

## Scope pressure

**No new mechanism.** vLLM rides the existing self-hosted peer path; no `BACKEND_CLASSES` entry, no
profile `backend: vllm`, no new bus. Every fix below is either a correctness fix on that path that
also benefits llama-cpp-server and Ollama (D1, D5, D6, D4) or a removal of a llama.cpp assumption
(D7–D9, D12). A dedicated backend class is justified only if we later want vLLM-only payload fields
(`guided_json`, `chat_template_kwargs`) — see Not in scope.

## Phases

Each phase is one PR, lands its guard tests in the same commit as its fix, and goes through the
two-lens (Executor + Architecture) review round before merge.

### Phase 0 — shared test fixture (lands with Phase 1)

`fake_openai_server` fixture built on the existing `httpx.MockTransport` injection pattern
(`tests/unit/test_http_client.py::test_stream_post_stores_stream_ctx`), **parametrized over two
shapes** — llama-cpp-server and vLLM — so one contract pins both:

- `/v1/models` (vLLM: `max_model_len`, `owned_by: "vllm"`, id = `--served-model-name`; llama: `context_length`)
- chat completion, SSE stream with final usage chunk
- 404 for an unknown model id
- vLLM 400 overflow body: `{"object":"error","message":"This model's maximum context length is N tokens. However, you requested M tokens …","type":"BadRequestError","code":400}`
- mid-stream `data: {"error": …}` chunk

### Phase 1 — it works: model identity + typed failures (D1, D2, D5, D6, D11)

1. **Probe sends the lane's model** — `_validate_remote_urls` passes `model=cfg.remote_model`
   into `for_url`. A stage-2 404 classifies as `model_missing` (with the served ids in the detail),
   not `inference_broken`.
2. **Dispatch model id** — resolution: explicit `remote_model`/`placement.model` wins; if unset and
   the server serves exactly one model, adopt it (logged to `lane_decisions.jsonl`); if unset and it
   serves several, fail the lane loudly naming them. Never send a Maxim profile name to a server that
   did not list it. *(Owner decision O1.)*
3. **Stream error chunk** — any chunk with an `error` key raises `BackendDown` (strict mid-stream).
4. **`BackendContextOverflow(requested_tokens, context_window)`** — new `BackendError` subclass,
   parsed from the 400 body (vLLM + llama.cpp shapes) with the matching router branch per the
   `types.py::BackendError` docstring: **no provider cooldown**, logged loudly, surfaced to the
   budgeter; triggers the Phase 2 re-probe.
5. **Cost** — `get_pricing` returns 0 when not `treat_as_cloud`.

Guardrail notes: `maxim_peer_backend.py` is under the Plan-3 CI grep — no `retry|backoff|gateway`
tokens, even in comments. `lint_no_silent_swallows` — no new swallows.

### Phase 2 — one source of truth for context length (D3, D4)

1. **One `/v1/models` reader** in `models/language/` (URL-based, via `utils/http.py`), returning
   `(served_ids, context_window, trusted: bool)`. `max_model_len` is **trusted** (it is the runtime
   limit); llama.cpp's `context_length` is **not** (it is the training context — per the
   `_resolve_proxy_context_window` docstring). Replace both raw-socket copies (D4) with it —
   without de-instrumenting their existing swallow sites.
2. **`_build_remote_backend` probes once per URL** (cached, short timeout) and stamps a trusted
   value into `served_n_ctx` — the slot and min-clamp that leg 3 of the n_ctx-drift lesson already
   trusts. Precedence: `lanes.<tier>.n_ctx` (new, additive, default `None`; clamped to the server
   value with a WARN) → probed trusted value → `llm.n_ctx`/profile. Adopted value + source logged to
   `lane_decisions.jsonl`. Probe failure → today's behaviour + doctor FAIL.
3. **Re-probe on `BackendContextOverflow`** (server restarted with a different `--max-model-len`).
4. **Caller check (fix-ships-with-a-caller):** grep non-test callers — the value must reach
   `LLMWorker`'s n_ctx clamp, `router.py::_provider_n_ctx`, and the proxy admission gate, not stop
   at a helper. This also closes the llm-routing §6 cross-process-reuse leg for any server that
   reports a trusted window.

### Phase 3 — doctor (D10)

`check_openai_compat_lane(tier)` for every lane whose primary placement is self-hosted, in all
roles, via `http.fetch_url` with the resolved key:

- reachability (401/403 → FAIL naming `api_key_ref`)
- model id present in `data[].id` (FAIL listing what the server offers)
- context: budget > server → **FAIL** with `maxim config set lanes.<tier>.n_ctx <max_model_len>`;
  budget < server → INFO; `max_model_len < max_response_tokens + minimal prompt` → FAIL
- opt-in `retry_id` probe for tool-calling flags (INFO; we don't use native tools today)

Plus: `check_llama_cpp_server_installed` accepts "reachable local OpenAI-compatible upstream" as an
inference path.

### Phase 4 — vLLM behind the leader (D7, D8, D9, D12)

Only needed when vLLM is the leader's upstream (e.g. the RTX 5080 box) so peers keep talking to the
leader unchanged. *(Owner decision O2 — schedule now or on first use.)*

1. `llm.upstream_url` config key (prefer config over env; if an env mirror is added, autouse scrub in
   `tests/conftest.py` in the same commit). Replaces the loopback literal in `start_leader_proxy`,
   `_probe_upstream_ready`, `swap_llm_server`.
2. **Operator-managed only** (option B): `swap_llm_server` refuses with 409 "upstream is externally
   managed" when Maxim did not spawn it. Reasons: minutes-long vLLM load vs
   `READINESS_TIMEOUT_S = 120`; ~90 % VRAM preallocation makes stop/start hot-swap costly; separate
   Python env means no `sys.executable -m` spawn.
3. `_served_model_matches` accepts any id in `/v1/models`; debug status reports both `llm_model`
   (profile) and `upstream_model_id` + `upstream_kind`. **Session reports record `upstream_model_id`**
   — a GGUF Q4 and an AWQ build of "the same" model are different code-under-test for provenance.
4. Proxy concurrency cap (`MAXIM_PROXY_MAX_CONCURRENT`, default 4, instant 429) becomes a config key
   sized per upstream (match vLLM `--max-num-seqs`); suppress GGUF VRAM projection for external
   upstreams. The router's per-lane `_inference_lock` is **not** touched here (see Not in scope).

### Phase 5 — docs

- `docs/user/llm-setup.md`: "Self-hosted OpenAI-compatible servers (vLLM)" — the recipe:
  `vllm serve <hf-id> --served-model-name <name> --max-model-len <N> --port 8100 --api-key <K>`
  (8100, not vLLM's default 8000 — that is the Reachy daemon port); `--reasoning-parser <p>` for
  Qwen3/R1-class models (otherwise `<think>` lands in `content` and `json_parser` can grab the first
  `{` inside it); `maxim config set lanes.<tier>.remote_url …`; tunnelled/public hostnames need an
  explicit placement with origin `peer`, or they take the cloud path.
- `configuration.md`, `cli-reference.md`, `peer-setup.md`, troubleshooting (`leader_proxy_debug.md`:
  vLLM overflow is a 400, not llama's 500; `maxim_doctor.md`: the new check).
- `docs/agents/llm-routing.md`: §1 mental model (not llama-only), §3 new `[engineering]` invariant
  ("a self-hosted lane's context window resolves from the server's trusted report; a request-shaped
  400 never cools down a provider" — Regression guard: the Phase 0 fixture tests), §6 drift gotcha
  update, §7 any new keys.
- A `changelog.d/` fragment per `src/` PR (an `[Unreleased]` line before M83 step 3).

### Phase 6 — measure before claiming (optional, `[engineering]` evidence only)

Pre-registered, harness on a **different machine** than the server: same GPU, same model +
quantization class, same prompts, N concurrent lanes; vLLM vs llama-cpp-server (state whether
`--parallel` was set) — tokens/s and p95 TTFT, code provenance asserted. This is a performance
claim, **not** a `[behavioral]` graduation; no claim that agent behaviour improved without its own
prereg.

**Minimum live smoke (every phase that touches dispatch):**
`MAXIM_LOG_FILE=/tmp/maxim.jsonl MAXIM_BACKEND_TRACE=1 maxim --sim "test basic recall" --interactive false --sim-max-turns 3`
— pass iff `peer_backend_call` succeeds against the vLLM URL, zero `_llm_unavailable`, tool calls
execute, and `lane_decisions.jsonl` shows the adopted model id + n_ctx with their source. Then force
one oversize prompt and see `BackendContextOverflow`, not a cooldown.

## Owner decisions

- **O1 — dispatch model id when `remote_model` is unset:** adopt the single served id (recommended),
  or require `remote_model` for every self-hosted lane (stricter, more config).
- **O2 — Phase 4 timing:** now (the 5080 leader is the natural vLLM host), or on first real use.
- **O3 — roadmap slot:** "Parallel lines — outside the ladder" in `roadmap_1_4.md`, with the entry
  condition "never switched on mid-arm in an E1–E3 run" (changing the served model mid-experiment is a
  confound). *(2026-10-18: the slot is not taken while this plan is deferred. The condition stands for
  any revival: the server is never switched mid-arm in E1–E3.)*

## Not in scope

- A `vllm` pip extra, or any import of vLLM.
- Maxim spawning/managing vLLM (option A). If ever wanted: a separate `maxim vllm serve` unit, never
  the auto-spawn path.
- Relaxing `LLMRouter._inference_lock` (one in-flight call per lane per process; its
  "llama-cpp not thread-safe" comment is stale for HTTP backends, but `_record_attempt_outcome`
  asserts the lock — its own refactor). Until then batching helps across processes/peers/sims, not
  within one sim.
- Native tool calling, `guided_json`/`response_format`, vLLM `/tokenize` for exact budgeting —
  opportunities, each its own opt-in.
- `os.environ.setdefault(api_key_env, …)` in `_build_remote_backend` letting a stale env key beat a
  rotated config key — real, unrelated to vLLM; filed as [#1371](https://github.com/dennys246/Maxim/issues/1371) (2026-10-18).

## Review record

- 2026-09-27 — four parallel read-only lenses (dispatch path; context/config/doctor; leader/peer
  lifecycle; verification/guardrails/docs). D1, D2, D3/D4 key list, D6 and D7 re-verified in source
  by the author. Cross-confirmed by ≥2 lenses independently: D1/D2 (model field), D3/D4
  (`max_model_len` unread), D5 (untyped overflow), option B (operator-managed).
