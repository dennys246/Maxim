# Peer → Leader Diagnosis Runbook

> **Type:** Operational runbook (not a build plan).
> **Audience:** Humans or AI assistants debugging a peer that can't reach the leader's GPU.
> **Companion docs:** [peer_leader_connectivity.md](peer_leader_connectivity.md) (architectural analysis), the multi-LLM scaling work (now complete) (Phase 7+ context).

This runbook was built from real debugging sessions (2026-04-05) that surfaced every failure mode listed here. Commands are copy-pasteable. Expected outputs are from a known-working state.

---

## Prerequisites

Before starting this runbook, verify on the **leader** machine:

```bash
maxim doctor
```

You need all of these green before peer diagnosis makes sense:

| Check | Required |
|---|---|
| GPU / CUDA | ✓ (any GPU visible) |
| Auto-spawn server responding | ✓ (port 8100 alive) |
| Role: leader | ✓ (bind=0.0.0.0) |
| cloudflared installed | ✓ |
| Tunnel config present | ✓ |
| API key set | ✓ |

If any are red/yellow, fix them first — the peer path can't work if the leader itself isn't healthy. See `maxim doctor --retry` for guided fixes.

**Also required:** the leader must be running `maxim` (auto-spawn keeps port 8100 alive). If maxim isn't running, start it:

```bash
maxim
```

Wait for the banner showing `large   self-hosted http://127.0.0.1:8100/v1` before proceeding.

---

## The bisection ladder

Run these in order. Each rung isolates one hop. Stop at the first failure — that's your blocker.

### Rung 0 — Is the peer configured for remote inference?

**Run on: peer**

```bash
maxim doctor
```

Look for:
- `Role: client` or `Role: solo` — both are fine for a peer
- `Auto-spawn server: nothing responding` — expected if the peer has no GPU; it'll use the remote

Then check the peer config:

```bash
maxim peer show
```

**Expected (good):**
```
Peer config: /home/user/.config/maxim/peer.yml
  url:      https://maxim.yourdomain.com/v1
  api_key:  jXzgjz…3LwzD4
```

**If "No peer config" →** the peer isn't configured at all:
```bash
maxim peer connect https://maxim.yourdomain.com/v1
# paste the leader's API key when prompted
```

**If the URL or key looks wrong →** re-do the setup:
```bash
maxim peer forget
maxim peer connect https://maxim.yourdomain.com/v1
```

**Alternative check — env vars:**
```bash
echo $MAXIM_LANE_LARGE_REMOTE_URL
echo $MAXIM_LANE_LARGE_REMOTE_API_KEY
echo $MAXIM_MAX_CLOUD_LANES
```

If set, these override peer.yml. Make sure they point at the right URL + key. If stale, unset them:
```bash
unset MAXIM_LANE_LARGE_REMOTE_URL MAXIM_LANE_LARGE_REMOTE_MODEL MAXIM_LANE_LARGE_REMOTE_API_KEY MAXIM_MAX_CLOUD_LANES
```

### Rung 1 — Does DNS resolve on the peer?

**Run on: peer**

```bash
dig maxim.yourdomain.com
```

**Expected (good):**
```
;; ANSWER SECTION:
maxim.yourdomain.com.  300  IN  A  104.21.76.26
maxim.yourdomain.com.  300  IN  A  172.67.186.25
```

Those IPs are Cloudflare's proxy edge — correct for a tunnel-routed domain.

**If "NXDOMAIN" or "SERVFAIL" →** the DNS record doesn't exist or hasn't propagated:
- Verify on the leader: `cloudflared tunnel route dns maxim-llm maxim.yourdomain.com`
- Check Cloudflare dashboard → DNS for the record (should be orange-clouded CNAME)
- Try `dig @1.1.1.1 maxim.yourdomain.com` to bypass local DNS cache

**If stale cache (previously gray-clouded, now orange again) →** flush:
- macOS: `sudo dscacheutil -flushcache && sudo killall -HUP mDNSResponder`
- Linux: `sudo systemd-resolve --flush-caches` or restart NetworkManager
- Windows: `ipconfig /flushdns`

### Rung 2 — Does Cloudflare edge respond?

**Run on: peer**

```bash
curl -I https://maxim.yourdomain.com/v1/models
```

**Expected responses (any of these means the edge is alive):**

| HTTP code | Meaning | Next step |
|---|---|---|
| `401 Unauthorized` | Edge forwarded to origin, auth rejected | Good — edge works. Jump to Rung 4 (auth) |
| `405 Method Not Allowed` + `allow: GET` | Edge forwarded to origin, uvicorn responded | Edge + origin alive. Jump to Rung 4 (auth with GET) |
| `502 Bad Gateway` | Edge forwarded, but origin is down | Leader's port 8100 isn't listening. Start `maxim` on leader |
| `403 Forbidden` | Cloudflare blocked before reaching tunnel | See §Common failures: "403 from Cloudflare" below |
| `Connection refused` / DNS error | Can't reach Cloudflare | Back to Rung 1 (DNS) or check network/firewall on peer |

### Rung 3 — Does cloudflared forward the request?

**Run on: leader (simultaneously with peer's curl)**

```bash
maxim tunnel tail --since 5m
```

**While running that**, re-send the curl from the peer:

```bash
# On peer:
curl -v -H "Authorization: Bearer <key>" https://maxim.yourdomain.com/v1/models
```

**Expected on leader's tail:**
```
DBG GET https://maxim.yourdomain.com/v1/models HTTP/1.1 connIndex=2
    headers={"Authorization":["Bearer ..."],"Cf-Connecting-Ip":["<peer-ip>"],...}
    originService=http://localhost:8100
DBG 200 OK connIndex=2 content-length=138
```

**If nothing appears in the tail →** cloudflared didn't receive the request:
- The 403 is from Cloudflare's edge layer (WAF / Access policy / Bot Fight Mode)
- Check Cloudflare dashboard → Security → WAF / Bot Fight Mode / Access Applications

**If you see the request + `ERR connection refused` →** port 8100 isn't listening:
- Start `maxim` on the leader
- Or start the server manually: `python -m llama_cpp.server --model ~/.maxim/models/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf --n_gpu_layers -1 --host 0.0.0.0 --port 8100 --n_ctx 8192 --api_key "$(cat ~/.config/maxim/api_key)"`

**If you see the request + `ERR 401` →** auth mismatch. Jump to Rung 4.

> **Note:** `maxim tunnel tail` requires cloudflared's `loglevel: debug` in the config file to show per-request lines. If you only see connection events, add `loglevel: debug` to `/etc/cloudflared/config.yml` and `sudo systemctl restart cloudflared`.

### Rung 4 — Does the auth handshake succeed?

**Run on: peer (or leader for local comparison)**

```bash
# From peer — full GET with auth:
curl -i -H "Authorization: Bearer <key>" https://maxim.yourdomain.com/v1/models

# Compare: same call on leader via loopback (bypasses tunnel):
curl -i -H "Authorization: Bearer $(cat ~/.config/maxim/api_key)" http://127.0.0.1:8100/v1/models
```

**Expected (good):**
```
HTTP/2 200
content-type: application/json
{"object":"list","data":[{"id":"~/.maxim/models/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf",...}]}
```

**If loopback 200 but tunnel 401 →** the key the peer is sending doesn't match. Common causes:
- Peer has a stale key (leader rotated since peer last connected)
- Key pasted with extra whitespace or decorative characters
- Fix: on leader `maxim tunnel key show`, on peer `maxim peer forget && maxim peer connect <url>`

**If loopback 200 but tunnel 403 →** Cloudflare's edge is blocking (not the server). See §Common failures.

**If both 401 →** the key file on the leader doesn't match what the server was spawned with. Restart `maxim` on the leader (re-reads the key file, passes to --api_key).

### Rung 5 — Is the peer's Maxim making outbound LLM calls?

**Run on: peer**

```bash
MAXIM_LANE_TRACE=1 maxim --sim "say hi"
```

**Expected in peer's terminal (within ~10s of sim start):**
```
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  DEBUG FLAGS ACTIVE: MAXIM_LANE_TRACE
  ...
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  Maxim LLM lanes
  large   self-hosted https://maxim.yourdomain.com/v1     ← MUST say "self-hosted" not "local"
  ...

peer_large req=abc12345 provider=lane-large model=mistral-7b status=ok latency=312ms http=200 tokens=50+10
```

**If the banner shows `large local ...` →** the peer is using its own local model, NOT the tunnel. Causes:
- `peer.yml` wasn't loaded (check Rung 0)
- `MAXIM_LANE_LARGE_REMOTE_URL` isn't set (check Rung 0)
- Auto-spawn found a GPU on the peer and spawned a local server (the remote_url env var was empty, so auto-spawn took over). Fix: set the URL explicitly or `maxim peer connect`.
- **Pre-loading race (debug plan root cause (a))**: peer.yml was read AFTER `LaneBackendManager` initialized. This would be a bug — report with logs.

**If the banner shows `self-hosted` but no `peer_infer` lines →** the LLM call isn't reaching the outbound client:
- The sim might be stalling before making any LLM calls (check for `LLM submit` lines)
- The orchestrator might be hallucinating non-existent tools (pre-existing orchestrator-prompt issue)

**If you see `peer_infer req=... status=error http=403` →** confirmed Cloudflare edge blocking. See §Common failures.

**If you see `peer_infer req=... status=error http=401` →** auth mismatch. Jump to Rung 4.

**If you see `peer_infer req=... status=error http=502` →** origin down. Start `maxim` on leader.

### Rung 6 — Is the leader's GPU actually firing?

**Run on: leader (simultaneously with peer's sim)**

```bash
nvidia-smi dmon -s u -c 60
```

This samples GPU utilization every second for 60 seconds. You should see utilization spike to 50-100% when an LLM request arrives (each inference takes ~44ms for short completions, longer for complex prompts).

**If GPU spikes correlate with peer's `peer_infer` lines →** everything works. The system is end-to-end healthy.

**If peer logs `status=ok latency=312ms` but GPU stays at 0% →** something else is handling the request:
- Is there another server running on port 8100? Check `ss -ltnp | grep 8100`
- Is the tunnel pointed at a cached response? (Unlikely for POST /chat/completions but possible for GET /models)

**Continuous monitoring alternative (stays running):**
```bash
watch -n 0.5 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

---

## Decision tree

```
Peer runs `maxim peer test <url>`
│
├── DNS failed ──────────────────────────── Rung 1
├── HTTP timeout / connection refused ───── Rung 1 or network/firewall issue
├── HTTP 401 ────────────────────────────── Rung 4 (auth mismatch)
├── HTTP 403 ────────────────────────────── §Common failures: Cloudflare blocking
├── HTTP 502 ────────────────────────────── Rung 3 (origin down, start `maxim` on leader)
├── HTTP 200 + models listed ────────────── Path works! Check Rung 5 (is peer's Maxim using it?)
│
Peer banner shows `large   local` ──────────── Rung 0 (peer config missing/overridden)
Peer banner shows `large   self-hosted` ────── Rung 5 (MAXIM_LANE_TRACE to verify calls)
GPU stays idle despite peer traffic ──────── Rung 6 (something else is serving)
```

---

## Common failure modes

### Stale DNS cache on peer

**Symptom:** `dig` works but `curl` / Python gets `[Errno 8] nodename not found`.

**Cause:** DNS was looked up during an earlier failure state (e.g., when the record was temporarily gray-clouded) and the resolver cached the NXDOMAIN / SERVFAIL.

**Fix:**
```bash
# macOS
sudo dscacheutil -flushcache && sudo killall -HUP mDNSResponder
# Linux
sudo systemd-resolve --flush-caches
# Restart the terminal / Python process that's holding the stale cache
```

### Cloudflare edge returns 403

**Symptom:** curl to the tunnel URL returns `HTTP/2 403` with a `cf-ray` header but no uvicorn `x-request-id`.

**Cause (in order of likelihood):**
1. **Bot Fight Mode** enabled on the Cloudflare zone — blocks non-browser User-Agents (curl, openai-python)
2. **Cloudflare Access policy** on the tunnel hostname — enforces SSO/email auth before requests reach the tunnel
3. **WAF Custom Rule** — blocking requests by path, header, or IP
4. **Super Bot Fight Mode** — free tier has limited toggle options

**Diagnosis:** Check Cloudflare dashboard → Security → WAF / Bot Fight Mode / Access → Applications. Look for rules applied to the tunnel hostname.

**Fix:** Disable Bot Fight Mode for the tunnel hostname, OR create a WAF Exception rule (`Skip` action) that matches `hostname = maxim.yourdomain.com`.

**Verification:** after changing the rule, retry `curl -I https://maxim.yourdomain.com/v1/models` — expect 401 or 405 (not 403).

### Cached responses masking a dead origin

**Symptom:** curl returns 200/405 with `server: cloudflare` headers, but the leader's port 8100 is down. GPU shows no activity.

**Cause:** Cloudflare's CDN cached a previous successful response. Subsequent GETs to `/v1/models` hit the cache, never reaching the tunnel. POST requests to `/v1/chat/completions` are NOT cached (POST is a non-cacheable method).

**Diagnosis:** Check `cf-cache-status` header in the curl response:
- `DYNAMIC` = cache considered, not stored (normal for tunnel)
- `HIT` = served from cache without reaching origin — stale data

**Fix:** Start `maxim` on leader so the origin is alive. The cache will naturally expire (TTL based on origin headers). Or purge: Cloudflare dashboard → Caching → Purge Cache → Custom Purge → `maxim.yourdomain.com/v1/models`.

### --api_key causes health-check timeouts

**Symptom:** auto-spawn on the leader times out after 120s even though the subprocess is alive. Falls back to in-process inference. Tunnel peers get 502.

**Cause:** llama-cpp-server requires Bearer auth on ALL endpoints when `--api_key` is set, including `/v1/models`. The health-check probe wasn't sending the auth header.

**Fix (already applied in `bug/auto-spawn-auth`):** `_health_check()` sends `Authorization: Bearer <key>`. Also treats 401 as "server is up" (HTTP listener alive, just rejected the key).

### peer.yml / env var precedence confusion

**Symptom:** peer's Maxim connects to the wrong URL or uses no auth, despite `maxim peer show` looking correct.

**Cause:** env vars (`MAXIM_LANE_LARGE_REMOTE_URL`, etc.) take precedence over `peer.yml`. If the shell has a stale export from a previous session, it overrides the file.

**Diagnosis:**
```bash
env | grep MAXIM_LANE_LARGE
```

**Fix:**
```bash
unset MAXIM_LANE_LARGE_REMOTE_URL MAXIM_LANE_LARGE_REMOTE_MODEL MAXIM_LANE_LARGE_REMOTE_API_KEY MAXIM_MAX_CLOUD_LANES
```

Then restart `maxim`. peer.yml values will take effect.

### Auto-spawn on the peer overrides the remote URL

**Symptom:** peer has a GPU → auto-spawn fires → spawns a LOCAL llama-cpp-server → peer talks to its own server instead of the leader.

**Cause:** If the peer has a GPU and `MAXIM_LANE_LARGE_REMOTE_URL` wasn't set before auto-spawn's check runs, auto-spawn takes over.

**Diagnosis:** peer startup banner shows `large self-hosted http://127.0.0.1:8100/v1` instead of the tunnel URL.

**Fix:** ensure peer.yml is loaded OR env vars are set BEFORE auto-spawn. `maxim peer connect <url>` handles this. Alternatively, force: `MAXIM_AUTO_SPAWN_LLM_SERVER=0`.

---

## Commands a peer-side Claude can run autonomously

These are safe, read-only, no side effects. A Claude instance on the peer machine can run them without asking for permission:

```bash
# Configuration state
maxim doctor
maxim peer show
env | grep MAXIM_LANE_LARGE
env | grep MAXIM_MAX_CLOUD

# Connectivity probes (read-only)
dig maxim.yourdomain.com
curl -I https://maxim.yourdomain.com/v1/models
curl -i -H "Authorization: Bearer $MAXIM_LANE_LARGE_REMOTE_API_KEY" https://maxim.yourdomain.com/v1/models
maxim peer test https://maxim.yourdomain.com/v1

# Trace-level diagnosis (generates log output but no state mutation)
MAXIM_LANE_TRACE=1 maxim peer test https://maxim.yourdomain.com/v1
```

## What requires the leader-side human (or Claude) to check

These can't be verified from the peer:

| Check | Why it needs leader access |
|---|---|
| `maxim tunnel tail` — did cloudflared receive the request? | Journal logs are local to the leader |
| `ss -ltnp \| grep 8100` — is port 8100 listening? | Local socket state |
| `nvidia-smi dmon` — is the GPU firing? | Local GPU monitoring |
| `maxim tunnel key show` — does the leader's key match the peer's? | Key file on leader's filesystem |
| Cloudflare dashboard → WAF / Bot Fight Mode | Requires Cloudflare account access |
| `sudo systemctl status cloudflared` — is the service alive? | Systemd on the leader |

A peer-side Claude should report the peer's findings from the "autonomous" commands above, then ask the human to check these leader-side items.

---

## Success criteria

The diagnosis is complete when:

1. `maxim peer test <url>` returns 4/4 green on the peer
2. Peer's startup banner shows `large   self-hosted https://maxim.yourdomain.com/v1`
3. `MAXIM_LANE_TRACE=1` shows `peer_infer req=... status=ok` for every LLM call
4. Leader's `maxim tunnel tail` shows `200 OK` for forwarded requests
5. Leader's `nvidia-smi dmon` shows GPU utilization spikes correlating with peer traffic

All five simultaneously = fully working peer → leader GPU offload.
