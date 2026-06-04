# Peer Setup Guide — Connecting to a Remote Leader

How to configure a Maxim instance to offload inference to another machine running Maxim as a **leader** — typically a home server with a GPU, exposed to the internet via a Cloudflare tunnel.

This guide walks through the full path from an empty machine to a working peer → leader connection, including the failure modes you'll hit along the way and how to diagnose each one.

> **Prerequisites on the leader:** `maxim tunnel setup` has been run, a cloudflared tunnel is live, and `MAXIM_ROLE=leader maxim` (or plain `maxim` with `~/.cloudflared/config.yml` present) is serving a model. See [LLM Setup — Tunnels](llm-setup.md#maxim-tunnel--guided-cloudflare-tunnel-setup) for leader-side setup.

> **Picking what model the leader serves:** bundled profiles go up to `llama-3.1-70b` for capable hardware. For any GGUF beyond the bundled set (custom quantizations, smaller specialty models, gated repos), register a profile on the leader with `maxim model add` — see [Adding Custom Profiles](llm-setup.md#adding-custom-profiles).

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

> **As of 1.0**, `maxim peer connect` writes both `~/.config/maxim/config.json::lanes.large.*` (canonical) and `~/.config/maxim/peer.yml` (deprecated compat through 1.x, retired in 2.0). Inspect either source via `maxim config get` or `maxim peer show`. See [Configuration → Quick start](configuration.md#quick-start-maxim-config).

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
6. Clears the remote-probe cache (P6) so the next `maxim` startup re-probes the freshly-configured leader instead of trusting any stale entry from a previous URL

**On subsequent `maxim` runs**, the config is read from disk and populates `MAXIM_LANE_LARGE_*` env vars via `setdefault` — env vars still win for per-session overrides.

> **Local override (`--llm`) takes precedence over peer config (P1).** If you pass `--llm mistral-7b` (or any local profile name) on a peer that has a remote leader configured, the peer runs that model **locally** instead of forwarding to the leader. The remote URL is cleared on the large lane for that session, auto-spawn (or in-process inference) takes over, and the rest of the runtime is unaffected. This lets you override the peer config for one session without editing files. Pass a cloud profile (e.g. `--llm claude-sonnet`) and that wins over both peer and local. To swap the **leader's** model instead, run `maxim peer llm <profile>`.

> **Probe-cache clearing.** `maxim peer connect`, `forget`, `restart`, `update`, and `llm <model>` all clear (or invalidate) the on-disk probe cache at `~/.maxim/util/last_probe_status.json`. The next `maxim` startup re-probes the leader rather than trusting an entry from before the change. Useful to know when debugging "I changed the leader URL but the peer is still using the old one."

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

# 3. Does the LeaderProxy respond?
curl -sI http://127.0.0.1:8099/v1/debug/status

# 4. Does the inference server respond directly?
curl -sI http://127.0.0.1:8100/v1/models

# 4. What does doctor say?
maxim doctor
```

Common causes:
- Maxim not started on the leader — run `maxim` (auto-detects leader role if `~/.cloudflared/config.yml` exists)
- llama-cpp-server auto-spawn failed — check logs at `data/logs/` or bump `MAXIM_AUTO_SPAWN_TIMEOUT` for slow model loads
- cloudflared pointing at the wrong local port — check `~/.cloudflared/config.yml` has `service: http://localhost:8099` (LeaderProxy, not 8100)
- Tunnel needs restart after config changes:
  - **Linux/WSL2 (systemd):** `sudo systemctl restart cloudflared`
  - **WSL2 (no systemd):** kill and relaunch: `pkill cloudflared && cloudflared --config ~/.cloudflared/config.yml tunnel run &`
  - **Manual/foreground:** Ctrl+C the running `cloudflared` process, relaunch with `cloudflared --config ~/.cloudflared/config.yml tunnel run`

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

**Watch what you paste into the hidden key prompt.** The prompt echoes nothing, so a mis-paste (your shell prompt, the URL, the `export MAXIM_LANE_LARGE_API_KEY=` prefix, a `✓` decoration character) gets silently stored as your key. If `peer show` displays something that doesn't look like a random string, that's what happened — `peer forget` and try again.

**Copy only the key value, not the surrounding text.** `maxim tunnel key export` on the leader prints something like `export MAXIM_LANE_LARGE_API_KEY=jXzgjz...3LwzD4`. The key is **only** the part after `=`, without quotes, without `export`, without the variable name, without any leading checkmark or prompt decoration.

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

On startup the logs should show the peer URL being picked up as the large-lane backend. Inference calls will then flow through the tunnel → leader's llama-cpp-server → back.

To temporarily override for a session (without editing the saved config):

```bash
MAXIM_LANE_LARGE_REMOTE_URL=http://other-host:8100/v1 maxim
```

To stop using the peer config entirely:

```bash
maxim peer forget
```

---

## Mesh management (Plan 4 Stage C)

`peer connect` gives you a single peer → leader pairing via `peer.yml`. Plan 4 Stage C ships a richer mesh management surface backed by `~/.config/maxim/mesh.yml` for operators who want **multiple nodes**, **graceful traffic shaping** via drain/resume, or **read-only health probes** without dropping into `curl`.

### Quick reference

| Verb | What it does | Stage |
|---|---|---|
| `maxim peer list-nodes [--json]` | Table or JSON of all configured nodes + live status (single network probe per node, classified into `ok`/`fail`/`warn`/`info`). Reads `mesh.yml`; falls back to `peer.yml` as a synthesized one-node mesh so existing installs see zero behavior change. | C1 |
| `maxim peer --node <name> status` | Probe a single node and print its current status + latency. Alias: `health`. Drained nodes report `drained (not probed)` without making a network call. | C1 |
| `maxim peer --node <name> drain [--force-self]` | Add `<name>` to the role-scoped drain set so future routing decisions skip it. Drain state persists at `~/.maxim/util/drained_nodes.{role}.txt` under a `filelock.FileLock` so concurrent drain calls don't race. Refuses to drain `mesh.yml::self` without `--force-self` — draining yourself strands in-flight requests. | C2 |
| `maxim peer --node <name> resume` | Remove `<name>` from the drain set. Idempotent. | C2 |
| `maxim peer list-drained` | Dump the current drain set, separated into "active" (drained name matches a real `mesh.yml` node) and "orphans" (drained name no longer matches any node — usually because the operator edited mesh.yml mid-flight). Orphans get a `resume` cleanup hint. | C2 |
| `maxim peer init-mesh [--force]` | Synthesize `~/.config/maxim/mesh.yml` from the existing `peer.yml`. Use this when you already ran `peer connect` and want to start using the mesh management verbs above. `peer.yml` is left in place by design — `runtime/role.py` reads its existence as part of role detection. `--force` overwrites an existing `mesh.yml` after backing it up to `mesh.yml.bak`. Refuses if `.bak` already exists (prevents losing your original on a double `--force`). | C3.1 |
| `maxim peer add-node <name> --url <url> [--role peer\|leader] [--force]` | Append a new node to `mesh.yml::nodes`. URL validation is **syntax-only** at add time — reachability is the next `list-nodes` / doctor probe's job. `--force` replaces an existing node in place, preserving operator-typed node order. | C3.2 |
| `maxim peer remove-node <name>` | Drop a node from `mesh.yml::nodes`. **Side effect:** auto-clears any drain state for the node with a visible "also cleared from drain state" message. Refuses on `self` — you can't delete the running daemon's own identity (the error documents the 3-step workaround). | C3.2 |
| `maxim peer --node <name> install <extras>` | Mesh-aware install. Composes drain → install → resume. Resolves URL + key from `mesh.yml::nodes`. Refuses self (use `pip install` locally). Exit code 3 = install ok but resume failed. | C3.3 |
| `maxim peer --node <name> update [--dry-run] [--force] [--branch <b>]` | Mesh-aware update. Composes drain → update → resume. `--dry-run` previews without draining. Refuses self (use `maxim peer update` locally). Same exit codes as install. | C3.5 |
| `maxim peer --node <name> restart` | Mesh-aware restart. Composes drain → restart → resume. Two-phase recovery poll (proxy up, then LLM ready). Refuses self. Same exit codes as install. | C3.5 |
| `maxim peer --node <name> llm <model>` | Mesh-aware LLM swap. Composes drain → swap → resume. Enables per-node model assignment (prerequisite for C5 capacity-aware routing). Refuses self. Same exit codes as install. | C3.6 |

### The two config files: `peer.yml` vs `mesh.yml`

The two files coexist by design. Each has a distinct purpose:

- **`~/.config/maxim/peer.yml`** is the **simple-single-leader** config from the `peer connect` flow. Maxim's `runtime/role.py` reads its **existence** as part of the leader-vs-peer role detection decision order (per Plan 2 R2a). Every Plan 4 Stage C verb leaves `peer.yml` untouched — even `init-mesh`, which copies values out of it but never modifies the source. **Do not delete `peer.yml` after running `init-mesh`** — role detection breaks silently on the next `maxim` invocation.
- **`~/.config/maxim/mesh.yml`** is the **multi-node topology** the C1/C2/C3 verbs read and (for C3.1+) write. It's deliberately declarative — the only sanctioned writers are `init-mesh`, `add-node`, and `remove-node` (all in `src/maxim/peer/mesh_setup.py`, enforced by a CI grep allow-list). Operators can hand-edit it; runtime code paths cannot.

### The two-layer split: declarative topology vs runtime mutable state

`mesh.yml` holds **declarative topology** (cluster_key, self, nodes). It is never mutated by automatic / runtime code paths.

`~/.maxim/util/drained_nodes.{role}.txt` holds **runtime mutable state** (which nodes are currently drained). It's role-scoped via `MAXIM_ROLE` (so leader and peer machines on the same host don't share drain state) and serialized via `filelock.FileLock` so concurrent `drain`/`resume` calls don't race. This is the Kubernetes-style "spec vs status" split — the two layers serve strictly disjoint purposes and need no reconciliation contract.

**Why this matters operationally:** if you back up your config, copy `mesh.yml` (and `peer.yml`). The drain state in `~/.maxim/util/` is per-machine and ephemeral — restoring it from a backup of a different machine isn't useful and could collide with that machine's own drain decisions.

### Walkthrough: peer.yml-only install → drained mesh in 4 commands

```bash
# Starting state: you already ran `maxim peer connect <url>` so peer.yml exists.
$ maxim peer list-nodes
━━━ Mesh: 1 node(s), self=leader ━━━
  ✓ leader  leader  https://maxim.yourdomain.com/v1 (self)
      → reachable (stage2 HTTP 200) [392ms]

# Step 1: convert peer.yml into a real mesh.yml so add-node + remove-node work
$ maxim peer init-mesh
✓ Synthesized ~/.config/maxim/mesh.yml from peer.yml
  → 1 node (leader, https://maxim.yourdomain.com/v1)
  → cluster_key copied from peer.yml::api_key
  → self set to 'leader'
  → peer.yml left in place (still used for role detection)

# Step 2: add a second node
$ maxim peer add-node tablet --url https://tablet.yourdomain.com/v1
✓ Added node 'tablet' to ~/.config/maxim/mesh.yml
  → peer, https://tablet.yourdomain.com/v1

# Step 3: drain it (e.g., before a tablet restart)
$ maxim peer --node tablet drain
✓ Drained 'tablet'. Drain set: ['tablet']

# Step 4: confirm the state is what you think it is
$ maxim peer list-drained
Drained nodes (1):
  ⊝ tablet
```

To bring tablet back: `maxim peer --node tablet resume`. To remove it permanently: `maxim peer remove-node tablet` (this auto-clears the drain state with a visible warning, so you don't end up with an orphan).

### Exit code contract

All Plan 4 Stage C verbs use a consistent exit code shape:

- **0** — success, including idempotent no-ops (e.g., draining an already-drained node prints "already drained" and exits 0)
- **1** — environmental failure (file missing, network unreachable, can't write to `~/.maxim/util/`)
- **2** — operator error (typo, missing required arg, refuse-without-force, attempting to drain self without `--force-self`, attempting to remove self)

Scripts wrapping these verbs should distinguish 1 vs 2 — exit 1 means "fix the environment and retry," exit 2 means "fix the command line."

---

## Common end-to-end gotchas

**The leader's chat template leaks tokens** (e.g., output contains `<|im_start|>` / `<|im_end|>` fragments). Not fatal — the OpenAI SDK that Maxim uses in normal operation handles message structure correctly, so this only surfaces in `peer test`. If you want clean output anyway, tune the llama-cpp-server's `--chat_format` or stop-token list on the leader.

**Latency over tunnel is ~65-94 ms per short completion** (Cloudflare tunnel hop + inference). Acceptable for planning/review lanes, tight for real-time motor control — keep a local backend for motion lanes if the peer is driving actuators. Historical latency baseline from 2026-04-04 measurements is preserved in the archived [agent_mesh.md](../archive/agent_mesh.md#latency-baseline-2026-04-04).

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

> **Note:** `maxim doctor` now includes an inference coherence check that sends a test prompt and verifies the response. It also checks GPU/CUDA, tier detection, and disk/RAM. If the leader is running CPU-only, the tier detection check will warn. Deeper GPU health checks (tokens/sec, latency jitter) are tracked inline as TODOs in [src/maxim/doctor/](../../src/maxim/doctor/).

## Remote Updates

Leaders automatically enable remote updates — peers can trigger `git pull + pip install` without SSH:

```bash
# From any peer:
maxim peer update              # preview pending commits (dry run)
maxim peer update --apply      # pull + install
maxim peer update --branch dev # target a specific branch
```

The leader must restart `maxim` after an update to load new code. Disable with `MAXIM_ALLOW_REMOTE_UPDATE=0` if you don't want peers to be able to trigger updates.

## Remote Package Installation

Install optional extras on the leader without SSH:

```bash
# Install a pymaxim optional extra
maxim peer install semantic           # pip install pymaxim[semantic]
maxim peer install llm-torch,vision   # multiple extras

# Install arbitrary pip packages
maxim peer install sentence-transformers

# Check what's installed on the leader
maxim peer deps
```

Available extras: `semantic`, `llm-llama`, `llm-server`, `llm-torch`, `llm-anthropic`, `llm-openai`, `vision`, `audio`, `reachy`, `comms`, `search`, `temporal`, `training`, `tts`, `yolo`, `database`.

Requires `MAXIM_ALLOW_REMOTE_UPDATE=1` on the leader (same as `peer update`).
