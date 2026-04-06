# Doctor Upgrade Plan

> **Status:** `maxim doctor` v1 shipped (platform detection + environment checks + platform-specific fix hints + retry loop + `maxim peer test`). This doc sketches where to take it next.
>
> **Current surface:** GPU/CUDA, llama-cpp-server install, auto-spawn reachability, leader role, LAN access (WSL2/Linux/macOS/Windows), cloudflared install, tunnel config, API key presence.

Everything here is optional expansion — `maxim doctor` is already useful at v1. Use this as a menu; pick what matters when the need shows up.

---

## Peer-side diagnostics (~100–200 LOC)

Today `maxim doctor` assumes the invoking machine *is* or *wants to be* the leader — every check (LAN bind, tunnel config, API key) advises how to expose this box. When a user runs Maxim as a **peer** pointed at a remote leader, doctor's output is misleading: it flags missing tunnel config and an absent API key as warnings, when really the peer just needs to consume a remote URL.

Add a peer-mode path so `maxim doctor` diagnoses *either* role correctly.

**Detection — what role is this box playing?**
- `MAXIM_ROLE=peer` env var (explicit)
- `MAXIM_LANE_INFER_REMOTE_URL` set and pointing at a non-local host (implicit)
- Otherwise fall through to current behavior (solo / leader)

**Peer-mode checks** (replace the "Role & Access" + "Tunnel" + "API key" sections when role=peer):
- **Remote URL reachability** — resolve DNS, TCP-connect, hit `/v1/models`, report latency. Reuse `_peer_test` logic from [src/maxim/doctor/cli.py:153](src/maxim/doctor/cli.py#L153).
- **API key set on peer** — `MAXIM_LANE_INFER_API_KEY` (or equivalent) present? If leader requires auth and peer has no key, fail loud with the fix: "Run `maxim tunnel key export` on the leader and paste the snippet here."
- **Auth smoke** — send a real completion with the configured key, confirm 200. If 401, key mismatch.
- **Model availability** — does the remote `/v1/models` advertise the model the peer expects to use? Catches the case where the leader swapped models.
- **Clock skew** (optional) — if auth is time-sensitive (future HMAC keys), flag > 30s drift.
- **Latency budget warning** — if round-trip p50 > 200ms, nudge that real-time lanes may struggle (reference the [multi_llm_scaling_ARCHIVED.md latency baseline](multi_llm_scaling_ARCHIVED.md#latency-baseline-2026-04-04)).

**Fix hints** should point at the **leader** machine, not this one:
- "Ask the leader to run `maxim tunnel key rotate` then `maxim tunnel key export`"
- "On the leader, verify `MAXIM_ROLE=leader maxim` is running and `maxim doctor` passes"

**`maxim doctor --as peer <url>`** — one-shot peer check from a machine that isn't configured yet (expands today's `maxim peer test` with the full check-list formatting + retry loop).

**`maxim doctor --as leader`** / **`--as solo`** — explicit role override for the ambiguous cases (e.g., a machine that's *both* a leader and runs its own agent loop).

Keep the existing `maxim peer test <url>` as the minimal one-command path — the new `--as peer` mode is a superset that runs in the full doctor formatting with retry support.

---

## Near-term quick wins (each ~50–150 LOC)

Small, self-contained checks that add immediate diagnostic value.

### 1. Deeper GPU health
- **VRAM free vs total** — `torch.cuda.memory_reserved()` / `torch.cuda.memory_allocated()` to show headroom during a sim run, not just capacity.
- **GPU driver + CUDA runtime version** — warn when PyTorch's CUDA version doesn't match the installed driver (the exact pattern that hit us during the Blackwell unblock).
- **Compute-capability match** — `torch.cuda.get_arch_list()` vs `torch.cuda.get_device_properties(0).major/minor` — flag mismatches so users know their torch build doesn't target their actual GPU.
- **Multi-GPU enumeration** — list all devices when `device_count() > 1` with per-device VRAM, useful before Phase 6 selects a GPU per lane.
- **Temperature + power state** — `nvidia-smi --query-gpu=temperature.gpu,power.draw` once, as a baseline. Useful in thermal-throttling debugging.

### 2. Disk + memory sanity
- **Model directory write permissions** — can Maxim actually download a model here?
- **Free disk space** — `data/sim_reports/` shouldn't silently fill the disk.
- **RAM headroom at startup** — warn when free RAM is less than 2× the infer profile's expected size.
- **GGUF file integrity** — quick SHA-256 spot-check against known hashes (or at least size check) to catch truncated downloads.

### 3. Key hygiene
- **Key file age warning** — nudge rotation when `api_key` file is older than 90 days.
- **Permissions check** — fail loud if `~/.config/maxim/api_key` is mode 644/666 (not 0600).
- **Auth smoke test** — with the key set, hit the local server's `/v1/models` using the actual Bearer token to verify auth is wired correctly.
- **Unauth smoke test** — send a request with a bogus key, expect 401. If it succeeds, server isn't enforcing auth.
- **Cloudflared loglevel warning** — parse `~/.cloudflared/config.yml` (or `/etc/cloudflared/config.yml`); if `loglevel: debug` is set, warn that Bearer tokens will be logged in plaintext to systemd journal. Suggest `loglevel: info` for production, with `journalctl --vacuum-time=1d` to purge the history. Discovered during Stage A peer-leader debugging.

### 4. JSON output
- **`maxim doctor --json`** — machine-readable output for CI scripts, log pipelines, and support-bundle tooling.
- **`maxim doctor --diff <snapshot>`** — compare against a saved-state snapshot to flag regressions ("last week this worked, now it doesn't").

---

## Mid-term expansions (each ~200–500 LOC)

Bigger checks that need more infrastructure but unlock meaningful capabilities.

### 5. Inference behavior probes
Go beyond "server responds to /v1/models" into "server generates sensible output."

- **Coherence check** — send a fixed prompt, verify response contains expected tokens (e.g., "What is 2+2? Answer in one word." → must contain "four" or "4"). Catches gibberish models.
- **Tokens/sec benchmark** — standard prompt, measure throughput. Track over time to catch silent regressions (thermal throttle, model swap, quantization change).
- **Latency jitter measurement** — 20 sequential short completions, report mean/p50/p95/p99. High p99 = GPU contention or thermal issues.
- **Concurrency ceiling** — ramp up parallel requests, find where queuing dominates. Informs mesh load-balancing later.
- **Context window sweep** — binary search to find the actual max working ctx vs what's configured. Sometimes the spawned server lies / runs out of VRAM partway up.
- **Cold-start vs warm-cache timing** — first inference after restart vs steady-state.

These compose naturally into a `maxim doctor benchmark` subcommand that produces a performance baseline document (complementing the [multi_llm_scaling_ARCHIVED.md observability section](multi_llm_scaling_ARCHIVED.md) numbers).

### 6. Network / connectivity depth
- **Latency to Cloudflare edge** — ping the user's nearest POP to size tunnel-hop overhead.
- **TLS cert validation** — for tunnel URLs, verify cert chain + expiry.
- **DNS resolver health** — measure resolution latency for the tunnel hostname.
- **UPnP / NAT-PMP detection** — can we auto-open the LAN port for peer access on supported routers?
- **mDNS broadcast + listen** — for Phase 7 peer discovery, verify mDNS works on this LAN (firewalls often block it).
- **`maxim doctor peer-flow`** — codifies the [peer_diagnosis_runbook.md](peer_diagnosis_runbook.md) as an automated end-to-end check: peer config → DNS → edge → tunnel → auth → origin → GPU. Returns a structured report per rung. Runnable from either side.
- **WebSocket test** — if/when multi-front input ships, verify WebSocket upgrades survive the tunnel.

### 7. Sim-based behavioral tests
Run actual short simulations and assert properties of the output. Bridges `maxim doctor` and the existing sim infrastructure.

- **`maxim doctor sim-check`** — run a 30-second cooperative sim with fixed seed, assert:
  - LLM calls complete without connection errors
  - Orchestrator + AUT each make ≥1 action
  - No "unknown tool" failures (catches persona prompt drift)
  - Roundup report generates successfully
  - Session cost stays under a threshold
- **Adversarial persona regression** — catch the hallucinated-tool pattern by scanning action history for names not in TOOL_DESCRIPTIONS.
- **Backend swap test** — run a sim, swap infer lane from local → remote mid-run, verify no interruption.
- **Scenario validation** — YAML-scenario lint (every scenario in `scenarios/` can be loaded + completes its first turn without schema errors).

### 8. Fix automation (explicit opt-in only)
Take the printed commands and actually run them on the user's behalf when they ask.

- **`maxim doctor --fix lan`** — calls PowerShell from WSL via `powershell.exe -Command "Start-Process ..."` with `RunAsAdmin` elevation prompt. Runs netsh + firewall rules. Tests connectivity after.
- **`maxim doctor --fix install-cloudflared`** — detects distro, runs the install command (with user confirmation), verifies binary works.
- **`maxim doctor --fix all`** — applies every auto-fixable issue. Prints plan, waits for confirm.
- **Undo log** — every auto-applied fix writes to `data/util/doctor_fixes.log` so users can reverse them (`maxim doctor --undo-last`).

Risk: automated fixes that go wrong are worse than fix hints that fail. Gate behind explicit flags, not default.

---

## Long-term bets (tied to other phases)

### 9. Agent mesh health (Phase 7)
Once peers exist, `maxim doctor` becomes a mesh-topology diagnostic:

- **Peer discovery** — list all mDNS-visible peers + their advertised models.
- **Peer latency matrix** — measure round-trip to each peer, show table.
- **Key validity across peers** — verify each peer's stored API key still matches what the leader's issued (for per-device keys from the [security model](multi_llm_scaling_ARCHIVED.md#phase-10-observability--verbose-tracing)).
- **Peer capability audit** — compare `RuntimeCapabilities` across the mesh, flag weak links (e.g., "Reachy has 4GB RAM, can't run the review lane — sharing with home PC").
- **Topology visualizer** — ASCII graph of peers + roles + load.

### 10. Observability integration (Phase 10)
Doctor becomes the UI for the verbose-tracing output:

- **`maxim doctor trace`** — tail the structured LLM-call trace log, show recent routing decisions.
- **Pressure history** — replay recent memory/compute-pressure snapshots to see when a spawned server thrashed.
- **Failure-pattern dashboard** — "over the last N sims, X% of LLM calls failed with reason Y."

### 11. Diagnostic bundle upload
For when users report an issue:

- **`maxim doctor bundle`** — zip up: platform info, recent logs, last sim report, doctor JSON output, router config (redacted of secrets).
- Bundle is purely **local-file output** — user can review and send wherever they want. Maxim doesn't auto-upload.
- Useful for reproducing user issues without needing a screen-share.

### 12. Learning loop
Lightweight opt-in telemetry that makes diagnostics smarter over time:

- Record which platforms most commonly need which fixes.
- Track fix-success rate per platform (did `ufw allow` actually unblock the port? or did it stay broken?).
- On a new user's first run, suggest fixes that worked for similar setups.
- All opt-in, all local-first (telemetry stays on-device until user explicitly shares).

---

## Cross-cutting: doctor invoked FROM other code paths

Today `maxim doctor` is a user-invoked subcommand. Future use cases that want automatic diagnostics:

- **Startup sanity** — on every `maxim` launch, run a subset of cheap checks silently, print only if something's wrong ("noisy-default" → "quiet-success" UX).
- **Sim pre-flight** — before launching a sim, run `check_server_reachable` + `check_gpu` and abort with a clear message if anything's broken.
- **Test fixture** — pytest fixture that asserts environment readiness before slow integration tests.
- **CI guardrail** — `maxim doctor --json --strict` fails CI if any check is non-ok, keeping the dev environment healthy.

These all reuse the existing checks — no new infrastructure needed, just exposing them as library calls alongside the subcommand.

---

## Sequencing suggestion

If I had to pick a next-three from this list to build:

1. **Inference coherence + tokens/sec benchmark** (#5 subset) — gives users a quick "is my LLM actually working well?" answer without reading the sim output. ~200 LOC.
2. **`maxim doctor --json`** (#4) — tiny, unlocks scripting + support bundles. ~30 LOC.
3. **Key hygiene checks** (#3 — age warning, permissions, auth smoke test) — rounds out the security story. ~100 LOC.

Together ~330 LOC + tests + docs. A focused session's worth of work, high per-LOC value.

The sim-based checks (#7) and mesh health (#9) are bigger and should wait for Phase 7 so they're built against a real mesh substrate.

---

## Non-goals

Things that sound doctor-adjacent but belong elsewhere:

- **Model downloading** — `scripts/download_models.sh` already owns this. Doctor just verifies files exist.
- **Configuration editing** — doctor reports state, doesn't mutate config files (except via `--fix` which is its own opt-in).
- **Process supervision** — `LocalServerSpawner` owns subprocess lifecycle. Doctor observes it.
- **Full benchmark suite** — if benchmarks grow beyond a few checks, they belong in a dedicated `maxim benchmark` subcommand, not in doctor.
