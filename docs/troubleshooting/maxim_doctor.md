# Troubleshooting with `maxim doctor`

`maxim doctor` runs platform-aware diagnostics and prints actionable fix hints. Run it first whenever something isn't working.

```bash
maxim doctor           # one-shot check
maxim doctor --retry   # walk through failures, retest after each fix
```

## Understanding the output

Each check shows one of:

| Symbol | Meaning |
|--------|---------|
| `✓` | Check passed |
| `⚠` | Warning — works but could be better |
| `✗` | Failed — this is likely your problem |

Lines starting with `→` are fix instructions. They're copy-pasteable and use your actual IPs/paths (not placeholders).

## Sections

### Environment

| Check | What it tests | Common fixes |
|-------|--------------|--------------|
| Platform | OS + runtime detection | Informational only |
| Architecture | CPU architecture | Informational only |
| GPU / CUDA | CUDA device visibility | Install CUDA drivers; on Blackwell (RTX 50xx), needs CUDA 12.8+ and torch>=2.7 |

**"No CUDA device available"** on a machine with a GPU:
- Check `nvidia-smi` — if it fails, GPU drivers aren't installed
- On WSL2: `nvidia-smi` must work inside WSL, not just Windows
- On Docker: run with `--gpus all`

### Local LLM

| Check | What it tests | Common fixes |
|-------|--------------|--------------|
| llama-cpp-server installed | `llama_cpp.server` importable | `pip install -e '.[llm-server]'` |
| Auto-spawn server | Port 8100 responding | Start `maxim` in another terminal (auto-spawn fires on startup) |

**"Nothing responding at port 8100":**
- Expected if you haven't started `maxim` yet — auto-spawn only fires during `maxim` startup
- If `maxim` is running but port is dead: check for port conflicts (`lsof -i :8100`)
- Manual start: `python -m llama_cpp.server --model <path.gguf> --port 8100`

### Role & Access

| Check | What it tests | Common fixes |
|-------|--------------|--------------|
| Role | leader / client / solo detection | Set `MAXIM_ROLE=leader` or add cloudflared config |
| LAN access | Peers can reach this machine | Platform-specific firewall hints |

**Roles explained:**
- **solo** (default): binds to `127.0.0.1`, only local access
- **leader**: binds to `0.0.0.0`, accepts peer connections + starts tunnel
- **client**: peer that uses a remote leader for inference

### Tunnel (Cloudflare)

| Check | What it tests | Common fixes |
|-------|--------------|--------------|
| cloudflared | Binary installed and in PATH | `brew install cloudflared` (macOS) or see [cloudflare docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) |
| Tunnel config | `~/.cloudflared/config.yml` exists | Run `maxim tunnel setup` for guided setup |

**"No config at ~/.cloudflared/config.yml":**
```bash
maxim tunnel setup    # interactive guided setup
```

### API key

| Check | What it tests | Common fixes |
|-------|--------------|--------------|
| API key | Key set for auth-gated access | `maxim tunnel key rotate` to generate one |

**"No key set":**
- Fine for localhost-only use (solo mode)
- Required before exposing via LAN or tunnel:
```bash
maxim tunnel key rotate     # generate a key
maxim tunnel key export     # print export commands for peers
```

### Lane Metrics (Phase 8)

Shown only after Maxim has processed LLM requests. Reports per-lane performance:

| Check | What it tests | Status thresholds |
|-------|--------------|-------------------|
| Lane: infer | Inference lane health | ok: <20% failures, warn: 20-50%, fail: >50% |
| Lane: review | Review lane health | Same thresholds |

**High failure rate on infer lane:**
- Check leader connectivity: `maxim peer test <url>`
- Check model is loaded: `curl -H "Authorization: Bearer $KEY" <url>/v1/models`
- Check logs for specific errors: `MAXIM_LANE_TRACE=1 maxim`

## Peer-specific diagnostics

If you're a peer trying to reach a leader, also run:

```bash
maxim peer show              # verify peer config
maxim peer test <leader-url> # end-to-end connectivity check
```

For deeper diagnosis, see [peer_diagnosis_runbook.md](peer_diagnosis_runbook.md).

## Debug flags

These env vars produce additional diagnostic output:

| Flag | Effect |
|------|--------|
| `MAXIM_HEARTBEAT=1` | System health heartbeat every 10s: GPU/CPU/RAM/disk/WiFi + stall detection |
| `MAXIM_LANE_TRACE=1` | Log every LLM dispatch with provider, latency, tokens, GPU metrics (also enables heartbeat) |
| `MAXIM_PEER_LOG_REQUESTS=1` | JSON log per outbound peer call |
| `MAXIM_PROVENANCE_VERBOSITY=2` | Verbose decision tracing in agent loop |

All print a loud startup banner so you don't leave them on accidentally.

## Heartbeat monitor

The heartbeat monitor samples system + LLM metrics every 10 seconds and logs a compact health line:

```
[heartbeat] gpu=87% vram=6.2/16G 72C 180W | cpu=23% | ram=8.1/16G | disk=42G free | infer: 5calls p50=280ms fail=0% | loop: idle=2.1s state=waiting
```

**Stall detection:** if the agent loop has been idle for 30+ seconds (configurable via `MAXIM_HEARTBEAT_STALL_S`), the heartbeat emits a WARNING:
```
[heartbeat] STALL DETECTED — agent loop idle for 62s (state=waiting_followup, threshold=30s)
```

**Enable:** `MAXIM_HEARTBEAT=1` or `MAXIM_LANE_TRACE=1` (both start the monitor). Always on in leader mode.

**Leader endpoint:** `GET /v1/debug/heartbeat` returns the full system snapshot as JSON (auth-gated). Peers can poll this for leader health visibility.

## Getting more help

- Troubleshooting guides: [docs/troubleshooting/](.)
- File an issue: https://github.com/dennys246/Maxim/issues
