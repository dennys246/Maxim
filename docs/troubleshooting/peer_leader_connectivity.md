# Peer → Leader LLM Communication: Debug & Observability Plan

## Why this plan exists

Mid-way through the multi-LLM scaling rollout, remote peers appear unable to
successfully trigger LLM jobs against the leader's GPU. A deep code trace
(2026-04-05) found a mix of **actual routing gaps** and **missing observability**
that makes it nearly impossible to tell *where* in the path a peer request is
dying. This plan captures both the architectural gaps and a staged debugging
strategy, including temporary security relaxations we can use to bisect the
failure fast.

Companion doc: the multi-LLM scaling work (now complete).

**Status (2026-04-05)**: Stage D ("real leader-side routing") is now folded into the multi-LLM plan as Phases 7a-7d. This debug plan's Stages A-C remain the sequenced entry points — Stage A observability is a prerequisite for Phase 7a, and the `LeaderProxy` described in §4.4 is what Phase 7a builds.
>
> **Operational runbook:** [peer_diagnosis_runbook.md](peer_diagnosis_runbook.md) — step-by-step bisection ladder for diagnosing a peer that can't reach the leader. Built from real failure modes found during 2026-04-05 debugging. Safe for a Claude on the peer machine to follow autonomously.

---

## 1. What the code actually does today

Peer → leader inference currently flows like this:

```
peer's large tier  →  openai-python client  →  https://<tunnel-url>/v1/chat/completions
                                                    │ (Cloudflare tunnel = dumb HTTP proxy)
                                                    ▼
                                              localhost:8100  (llama-cpp-server, spawned by leader)
                                                    │
                                                    ▼
                                              returns directly to peer
```

**The leader's Maxim runtime never sees the peer request.** No `LLMRouter`, no
`WorkerPool`, no lane manager, no logging. Peers and the leader are two
independent clients of the same `llama-cpp-server` process, contending for GPU
time with no coordination.

### Concrete code references

| # | Where | What it does | Gap |
|---|-------|--------------|-----|
| 1 | [src/maxim/runtime/lane_backends.py:349-352](../../src/maxim/runtime/lane_backends.py#L349-L352), [L286](../../src/maxim/runtime/lane_backends.py#L286) | Peer large lane is wired directly to `{remote_url}/v1/chat/completions` via OpenAI client | No Maxim-level endpoint; bypasses leader runtime entirely |
| 2 | [src/maxim/runtime/lane_backends.py:612-621](../../src/maxim/runtime/lane_backends.py#L612-L621) | Leader rewires its own large lane to `localhost:8100` | Leader + peer compete for GPU with no priority/queue awareness |
| 3 | [src/maxim/runtime/local_server_spawner.py:160-172](../../src/maxim/runtime/local_server_spawner.py#L160-L172) | llama-cpp-server gets `--api_key` only if one is set | Solo→leader transitions can leave server unauthenticated while tunnel is exposed |
| 4 | [src/maxim/tunnel/config.py:23-36](../../src/maxim/tunnel/config.py#L23-L36) | Cloudflare tunnel is a pure proxy | No upstream auth enforcement; relies entirely on llama-cpp-server |
| 5 | [src/maxim/peer/cli.py](../../src/maxim/peer/cli.py), [src/maxim/doctor/cli.py](../../src/maxim/doctor/cli.py) | `maxim peer test` probes `/v1/models` + a chat completion | Doesn't verify GPU served it, doesn't confirm leader runtime saw it |
| 6 | [src/maxim/peer/config.py:81-91](../../src/maxim/peer/config.py#L81-L91) | API key stored plaintext (0600), shared across peers | No per-peer keys, no rotation without manual redistribution |

---

## 2. Likely root cause of "peers can't trigger jobs right now"

In order of likelihood:

**(a) Lane init races peer.yml load.** If the peer's `LaneBackendManager`
initializes before `peer.yml` is read, `MAXIM_LANE_LARGE_REMOTE_URL` is unset
and the large lane silently falls back to `local-llama`. The peer will *appear*
to work — it just never hits the leader.

**(b) API key mismatch / absent key.** Leader's llama-cpp-server may have
been spawned in solo mode (no `--api_key`) and never restarted after becoming
a leader. Or the peer holds a stale key after `maxim tunnel key rotate`.
Symptom: 401/403 from llama-cpp-server.

**(c) Cloudflare WAF / Bot Fight Mode.** Commit a74a5e2 patched the UA, but
zone-level Bot Fight Mode can re-block the openai-python client.

**(d) Design gap (not a regression).** A Maxim-level leader HTTP receiver that
routes peer jobs through the `WorkerPool` simply doesn't exist yet. If that's
what "peers trigger LLM jobs" means to us, it's unimplemented.

---

## 3. Debug strategy — bisect with temporary security relaxations

The goal is a decision tree that lets us pinpoint the failing hop in under
5 minutes. Each step **temporarily** disables one layer; we re-enable
afterwards.

### Phase 0: Baseline instrumentation (do first, keep permanently)

Before touching security, add logging that will survive:

1. **Peer lane resolution log.** In `lane_backends.py` around L349, log the
   resolved `remote_url`, `remote_api_key[:8]+"…"`, and `remote_model` at
   startup. If this prints `None` or `local-llama`, root cause (a) confirmed.
2. **Per-request tag log.** Wrap the OpenAI client call in a logger that emits
   `peer_infer_request: model=X url=Y status=Z latency=Nms` for every call.
3. **Leader-side tail helper.** Add `maxim tunnel tail` that streams
   `cloudflared` logs + llama-cpp-server stdout in a single pane, filtered by
   request ID if present.

### Phase 1: Can the peer reach the tunnel at all?

```bash
# On peer:
curl -v https://<tunnel-url>/v1/models     # no auth header
curl -v -H "Authorization: Bearer $KEY" https://<tunnel-url>/v1/models
```

- 403 with "Just a moment" / Cloudflare HTML → **WAF/Bot Fight Mode blocking**.
  Fix: disable Bot Fight Mode on the zone, or add a WAF skip rule for the
  tunnel hostname.
- 401 → auth problem; jump to Phase 2.
- Connection refused / 502 → tunnel down or llama-cpp-server not bound.
- 200 with models list → tunnel & auth both fine; jump to Phase 3.

### Phase 2: Temporarily strip auth to isolate tunnel vs. auth

**Temporary, debug-only.** Do this on a non-public tunnel or behind a
Cloudflare Access policy if possible.

1. Restart leader with `MAXIM_LEADER_DISABLE_AUTH=1` (new flag — see §4).
   This spawns llama-cpp-server **without** `--api_key`.
2. From peer: `curl -v https://<tunnel-url>/v1/chat/completions -d '{...}'`
   with **no** Authorization header.
3. Interpretation:
   - Works → auth was the failure. Re-enable auth, verify peer.yml has the
     right key (`maxim tunnel key show` on leader vs peer's `peer.yml`).
   - Still fails → tunnel or llama-cpp-server; check `cloudflared` logs.

**Re-enable auth immediately after.** Do not leave a public tunnel
unauthenticated.

### Phase 3: Is the peer's Maxim runtime actually calling the remote URL?

If `curl` from the peer works but the peer's Maxim process doesn't trigger the
leader's GPU:

1. Start peer with `MAXIM_LANE_TRACE=1` (new flag, see §4). Logs every lane
   dispatch: `lane=infer backend=remote-openai url=https://...`.
2. If `backend=local-llama` → root cause (a): peer.yml loaded after lane init.
   Fix: move peer config load before `LaneBackendManager.__init__`.
3. If `backend=remote-openai` but GPU on leader idles → request is going
   somewhere else. Check DNS, tunnel routing, or a stray local proxy.

### Phase 4: Is the leader's GPU actually serving peer requests?

- On leader: `nvidia-smi dmon -s u` while peer sends a request. Util spike
  should correlate.
- Add a request-id header (`X-Maxim-Request-Id`) in peer's lane client, log it
  both sides. Grep llama-cpp-server logs for the ID.

---

## 4. New observability tooling to build

These are small, bounded additions that pay for themselves on the first real
debug session:

### 4.1 Env flags (debug-only, documented, warn loudly when set)

| Flag | Effect | Default |
|------|--------|---------|
| `MAXIM_LEADER_DISABLE_AUTH=1` | Spawn llama-cpp-server without `--api_key`; leader logs a WARNING every 30s | off |
| `MAXIM_LANE_TRACE=1` | Log every lane dispatch with backend, URL, model, request-id | off |
| `MAXIM_PEER_LOG_REQUESTS=1` | Peer logs each remote call with status, latency, bytes | off |
| `MAXIM_TUNNEL_ECHO=1` | Leader logs inbound request headers (for debugging WAF/UA issues) | off |

When any of these are set, the process should print a boxed WARNING at startup
so they're never left on accidentally.

### 4.2 New CLI commands

- **`maxim peer trace <url>`** — extended `maxim peer test` that:
  - Sends a known-payload inference request with a unique request-id
  - Polls the leader (via a new `/debug/last-requests` endpoint, auth-gated)
    to confirm the leader saw it
  - Reports full timing breakdown: DNS → TCP → TLS → tunnel → server → GPU
  - Flags UA/WAF issues from response headers

- **`maxim tunnel tail`** — streams cloudflared + llama-cpp-server logs
  side-by-side with timestamps aligned. Optional `--request-id <id>` filter.

- **`maxim doctor peer-flow`** — end-to-end check runnable from either side:
  validates peer.yml freshness, API key match, tunnel reachability, WAF
  behavior, llama-cpp-server readiness, GPU availability.

### 4.3 Request-id propagation

Add `X-Maxim-Request-Id` to every outbound peer inference request (UUID4).
Log it on peer send, on leader receipt (via a lightweight sidecar — see 4.4),
and in any error path. This is the single most useful thing for correlating
events across machines.

### 4.4 Lightweight leader sidecar (thin, debug-first)

Add a minimal HTTP sidecar on the leader that sits **in front of**
llama-cpp-server (reverse-proxy on a new port, e.g. `:8099`). Responsibilities:

1. Log every inbound request with headers, request-id, source IP, auth status
2. Enforce API key check **before** llama-cpp-server (closes the Phase 3 gap)
3. Expose `/debug/last-requests` (auth-gated, ring buffer of last 100)
4. Future hook point for routing peer jobs through the leader's WorkerPool

Tunnel ingress swaps from `localhost:8100` → `localhost:8099`. llama-cpp-server
stays on 8100 but is no longer directly exposed. This is the smallest change
that closes the architectural gap from §1 and gives us real logs.

### 4.5 Structured log channel

Route all peer/leader/tunnel log lines through a named logger
(`maxim.mesh.trace`) with JSON output available via `MAXIM_MESH_LOG_JSON=1`.
Makes post-hoc analysis with `jq` trivial.

---

## 5. Staged execution

**Stage A — observability first (no behavior change):**
1. Lane resolution log (§4.1 `MAXIM_LANE_TRACE`)
2. Peer request log (§4.1 `MAXIM_PEER_LOG_REQUESTS`)
3. Request-id propagation (§4.3)
4. `maxim tunnel tail` (§4.2)

**Stage B — diagnostic flags:**
5. `MAXIM_LEADER_DISABLE_AUTH` + startup WARNING banner (§4.1)
6. `MAXIM_TUNNEL_ECHO` (§4.1)
7. `maxim peer trace` command (§4.2)

**Stage C — close the architectural gap:**
8. Leader sidecar reverse-proxy (§4.4) — starts as pure passthrough+logging
9. Move auth enforcement from llama-cpp-server into the sidecar
10. `/debug/last-requests` endpoint
11. `maxim doctor peer-flow` end-to-end check (§4.2)

**Stage D — real leader-side routing (future, feeds multi_llm_scaling):**
12. Sidecar parses inference requests, enqueues them in leader's WorkerPool
13. Per-peer request accounting + rate limits
14. Per-peer API keys with rotation (replaces shared-key model)

---

## 6. Non-goals

- This plan is **not** a rewrite of the tunnel or auth architecture.
- It does **not** replace the multi-LLM scaling plan — it unblocks it by
  making the system observable.
- Stage D items are explicit TODOs for the main multi-LLM plan, not this one.

---

## 7. Success criteria

- Given a peer that can't reach the leader, we can identify the broken hop
  (DNS / tunnel / WAF / auth / lane config / GPU) in under 5 minutes using
  only `maxim peer trace` output.
- Every peer inference request has a request-id visible in both peer and
  leader logs.
- No permanent security regressions: all debug flags are opt-in, warn at
  startup, and are off by default.
