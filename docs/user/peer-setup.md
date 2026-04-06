# Peer Setup Guide — Connecting to a Remote Leader

How to configure a Maxim instance to offload inference to another machine running Maxim as a **leader** — typically a home server with a GPU, exposed to the internet via a Cloudflare tunnel.

This guide walks through the full path from an empty machine to a working peer → leader connection, including the failure modes you'll hit along the way and how to diagnose each one.

> **Prerequisites on the leader:** `maxim tunnel setup` has been run, a cloudflared tunnel is live, and `MAXIM_ROLE=leader maxim` (or plain `maxim` with `~/.cloudflared/config.yml` present) is serving a model. See [LLM Setup — Tunnels](llm-setup.md#maxim-tunnel--guided-cloudflare-tunnel-setup) for leader-side setup.

---

## The short version

On the **leader** machine:

```bash
maxim tunnel status          # confirm the hostname + tunnel config
maxim tunnel key export      # print the API key (copy the value after `=`)
maxim                        # start the leader (auto-detects role)
```

On the **peer** machine:

```bash
maxim peer connect https://<hostname-from-leader>
# paste the key at the hidden prompt
# test runs automatically — must pass or config is not saved

maxim                        # routes inference to the leader
```

That's it when everything works. The rest of this guide is for when it doesn't.

---

## Verify the leader is healthy FIRST

Before touching peer config, confirm the leader is actually serving. From any machine (including the peer you're about to set up):

```bash
curl -sI https://maxim.yourdomain.com/v1/models
```

Expected: `HTTP/2 200` with JSON body listing the model.

If you get anything else, stop — the peer can't connect to a broken leader. See [Diagnosing leader-side failures](#diagnosing-leader-side-failures) below.

---

## The `peer connect` command

```bash
maxim peer connect <url> [--key KEY] [--model MODEL] [--skip-test]
```

**What it does:**

1. Normalizes the URL (appends `/v1` if missing, strips trailing `/`)
2. Prompts for the API key (hidden input) unless `--key` was passed
3. Runs a four-step connectivity test against the leader
4. Saves `{url, api_key, model?, is_cloud}` to `~/.config/maxim/peer.yml` (mode 0600) only if the test passes
5. Auto-detects `is_cloud: true` for public hostnames and enables the cloud-lane gate

**On subsequent `maxim` runs**, the config is read from disk and populates `MAXIM_LANE_INFER_*` env vars via `setdefault` — env vars still win for per-session overrides.

**Companion commands:**

```bash
maxim peer show      # print current peer config (key truncated)
maxim peer forget    # delete the stored peer config file
maxim peer test <url>  # run the connectivity test ad-hoc (doesn't save)
```

---

## Interpreting peer-test failures

The four checks and what each failure means:

| Stage | Failure message | Cause | Fix |
|---|---|---|---|
| **URL parse** | `URL has no host` | Missing `https://` scheme or pasted placeholder literally (e.g. `<hostname>` — zsh parses `<`/`>` as redirection) | Prepend `https://`; type the real hostname, not example placeholders |
| **DNS** | `DNS failed` | Hostname doesn't resolve | `dig <hostname>` — check Cloudflare DNS record exists and points at tunnel |
| **DNS** | DNS resolves but the A record points at origin IPs (not Cloudflare anycast) | DNS record is an A record, not a tunnel CNAME | On leader: `cloudflared tunnel route dns <tunnel-name> <hostname>` |
| **TLS** | `SSL: CERTIFICATE_VERIFY_FAILED` | You're reaching a different server than intended — parked domain, expired cert, wrong hostname | Re-verify the hostname with `maxim tunnel status` on the leader |
| **`/v1/models`** | `HTTP 401` | Bearer token rejected by leader's auth layer | Re-export the key from the leader and paste only the value after `=` |
| **`/v1/models`** | `HTTP 403` | Cloudflare WAF, Access policy, or bot-protection rule blocking the request | See [Diagnosing 403](#diagnosing-403) below |
| **`/v1/models`** | `HTTP 404` | Wrong base URL — `/v1` might already be in the URL twice, or the server doesn't serve OpenAI-compatible paths | Check the URL; try `curl -sI <url>/models` directly |
| **`/v1/models`** | `HTTP 502` | Cloudflare reached the tunnel edge but the origin isn't responding | See [Diagnosing 502](#diagnosing-502) below |
| **`/v1/models`** | `HTTP 521` / `522` | Origin server down or connection timeout | Leader process isn't running or has crashed — start it |
| **chat completion** | `Chat completion failed: <error>` | Auth passed but the model call failed | Check leader logs; might be OOM, model-loading failure, or chat-template issue |

---

## Diagnosing leader-side failures

### Diagnosing 502

Cloudflare reached the tunnel edge, but nothing responded at the origin. On the **leader**:

```bash
# 1. Is the leader process running?
ps aux | grep -E "(maxim|llama-cpp-server)" | grep -v grep

# 2. Is cloudflared running?
ps aux | grep cloudflared | grep -v grep

# 3. Does the local server respond directly?
curl -sI http://127.0.0.1:8100/v1/models

# 4. What does doctor say?
maxim doctor
```

Common causes:
- Maxim not started on the leader — run `maxim` (auto-detects leader role if `~/.cloudflared/config.yml` exists)
- llama-cpp-server auto-spawn failed — check logs at `data/logs/` or bump `MAXIM_AUTO_SPAWN_TIMEOUT` for slow model loads
- cloudflared pointing at the wrong local port — check `~/.cloudflared/config.yml` matches the server's actual port (default 8100)

### Diagnosing 403

A 403 from `peer test` alongside a 200 from `curl` usually means **User-Agent filtering**. Cloudflare's default Bot Fight Mode blocks Python's default `Python-urllib/*` user agent. Maxim's `peer test` sends a neutral UA to work around this; if you're still seeing 403:

```bash
# Compare headers — is the 403 coming from Cloudflare or your backend?
curl -sI https://maxim.yourdomain.com/v1/models
curl -sI -H "Authorization: Bearer <key>" https://maxim.yourdomain.com/v1/models
```

Look at the `server:` header:

- **Only `server: cloudflare`** (no backend header) → Cloudflare edge is blocking. Check:
  - Cloudflare Dashboard → Security → WAF → Custom Rules
  - Cloudflare Dashboard → Zero Trust → Access → Applications (any app covering this hostname?)
  - Security → Bots → disable "Bot Fight Mode" or add an exception for your tunnel hostname
- **Backend header present** (`server: uvicorn`, `llama-cpp-server`, etc.) → your server is rejecting the key. Re-export from leader.

If you have a Cloudflare Access application on the tunnel hostname, it enforces browser-based SSO and will block any non-browser API client. For a single-user API tunnel, remove the Access app and rely on the Bearer key alone.

### Diagnosing DNS pointing at the wrong thing

If `dig <hostname>` returns Cloudflare anycast IPs (the `104.21.*` / `172.67.*` ranges) but you're still getting 502, the DNS record is probably a **proxied A record** pointing at nothing (or at the wrong origin), not a **tunnel CNAME**.

Fix on the leader:

```bash
cloudflared tunnel list                                      # find your tunnel name
cloudflared tunnel route dns <tunnel-name> <hostname>        # re-point DNS at the tunnel
```

Or in the Cloudflare Dashboard: DNS → delete the A record for `<hostname>` → create a CNAME: `<hostname>` → `<tunnel-id>.cfargotunnel.com` (proxied).

---

## Pitfalls to avoid

**Don't paste angle-bracketed placeholders.** If a guide says `maxim peer connect https://<leader-hostname>`, the `<leader-hostname>` is a fill-in, not literal syntax. zsh will try to parse `<` as input redirection and fail with `parse error near '\n'`.

**Don't paste multi-line shell blocks with `#` comments.** Pasting a code block with `# comment` lines into zsh executes each one separately. Comments will print `zsh: command not found: #` and any line fragments get re-executed. If Maxim's interactive agent loop is already running, every stray paste gets forwarded to the agent as user input — which triggers real LLM calls (real cost). If you land in this state, Ctrl+C out and `clear` the terminal before continuing.

**Don't use `--skip-test` to silence auth failures.** The test is what catches wrong keys, wrong hostnames, and broken tunnels before they get written into a persistent config. `--skip-test` has exactly one legitimate use: pre-staging a config for a leader you know is offline (e.g., before taking a laptop on a trip). If a test is failing, fix the underlying problem instead of bypassing it.

**Don't disable TLS verification.** A cert error almost always means you're reaching the wrong host. Fix the hostname.

**Watch what you paste into the hidden key prompt.** The prompt echoes nothing, so a mis-paste (your shell prompt, the URL, the `export MAXIM_LANE_INFER_API_KEY=` prefix, a `✓` decoration character) gets silently stored as your key. If `peer show` displays something that doesn't look like a random string, that's what happened — `peer forget` and try again.

**Copy only the key value, not the surrounding text.** `maxim tunnel key export` on the leader prints something like `export MAXIM_LANE_INFER_API_KEY=jXzgjz...3LwzD4`. The key is **only** the part after `=`, without quotes, without `export`, without the variable name, without any leading checkmark or prompt decoration.

---

## Verification checklist

Once `peer connect` succeeds:

```bash
maxim peer show
# url:      https://maxim.yourdomain.com/v1
# api_key:  jXzgjz…3LwzD4        ← looks like a random string, not your shell prompt
# is_cloud: true
```

Then run Maxim:

```bash
maxim
```

On startup the logs should show the peer URL being picked up as the infer-lane backend. Inference calls will then flow through the tunnel → leader's llama-cpp-server → back.

To temporarily override for a session (without editing the saved config):

```bash
MAXIM_LANE_INFER_REMOTE_URL=http://other-host:8100/v1 maxim
```

To stop using the peer config entirely:

```bash
maxim peer forget
```

---

## Common end-to-end gotchas

**The leader's chat template leaks tokens** (e.g., output contains `<|im_start|>` / `<|im_end|>` fragments). Not fatal — the OpenAI SDK that Maxim uses in normal operation handles message structure correctly, so this only surfaces in `peer test`. If you want clean output anyway, tune the llama-cpp-server's `--chat_format` or stop-token list on the leader.

**Latency over tunnel is ~65-94 ms per short completion** (Cloudflare tunnel hop + inference). Acceptable for planning/review lanes, tight for real-time motor control — keep a local backend for motion lanes if the peer is driving actuators. See the [latency baseline in the multi-LLM plan](../plans/multi_llm_scaling.md#latency-baseline-2026-04-04) for measured numbers.

**If the tunnel keeps flapping** (502 and 403 alternating on repeated requests), cloudflared is reconnecting. Check the leader's cloudflared logs — typically a local network flap or the leader process restarting the llama-cpp-server. Fix the leader's stability before troubleshooting the peer.

**Remote GPU isn't being used even though inference works.** Requests reach the leader (you see completions come back at reasonable latency), but `nvidia-smi` on the leader shows 0% GPU utilization. Most common cause: `llama-cpp-python` was installed without CUDA support, or the server was spawned with `n_gpu_layers=0`. On the leader:

```bash
# Check the running server's flags
ps auxww | grep llama-cpp-server | grep -v grep
# → look for --n_gpu_layers; if missing or 0, it's CPU-only

# Check if llama-cpp-python was built with CUDA
python -c "import llama_cpp; print(llama_cpp.llama_cpp._lib)"
```

Fix — rebuild llama-cpp-python with CUDA:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

And force GPU offload on auto-spawn:

```bash
export MAXIM_AUTO_SPAWN_N_GPU_LAYERS=-1   # -1 = offload all layers to GPU
maxim
```

Then re-run `nvidia-smi` on the leader during a sim and confirm GPU utilization spikes. A 7B-Q4 model at ~5 GB VRAM should peg a modern GPU briefly on each inference call.

> **TODO (leader-side doctor check):** `maxim doctor` on the leader should detect CPU-only llama-cpp-python and warn — currently it just checks that the server responds on port 8100. See [docs/plans/doctor_upgrade_plan.md](../plans/doctor_upgrade_plan.md#1-deeper-gpu-health) for the planned GPU health checks.

## Remote Updates

Leaders automatically enable remote updates — peers can trigger `git pull + pip install` without SSH:

```bash
# From any peer:
maxim peer update              # preview pending commits (dry run)
maxim peer update --apply      # pull + install
maxim peer update --branch dev # target a specific branch
```

The leader must restart `maxim` after an update to load new code. Disable with `MAXIM_ALLOW_REMOTE_UPDATE=0` if you don't want peers to be able to trigger updates.
