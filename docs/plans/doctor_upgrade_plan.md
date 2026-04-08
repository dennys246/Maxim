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
- **Latency budget warning** — if round-trip p50 > 200ms, nudge that real-time lanes may struggle (reference the [agent_mesh.md latency baseline](agent_mesh.md#latency-baseline-2026-04-04)).

**Fix hints** should point at the **leader** machine, not this one:
- "Ask the leader to run `maxim tunnel key rotate` then `maxim tunnel key export`"
- "On the leader, verify `MAXIM_ROLE=leader maxim` is running and `maxim doctor` passes"

**`maxim doctor --as peer <url>`** — one-shot peer check from a machine that isn't configured yet (expands today's `maxim peer test` with the full check-list formatting + retry loop).

**`maxim doctor --as leader`** / **`--as solo`** — explicit role override for the ambiguous cases (e.g., a machine that's *both* a leader and runs its own agent loop).

Keep the existing `maxim peer test <url>` as the minimal one-command path — the new `--as peer` mode is a superset that runs in the full doctor formatting with retry support.

---

## Near-term quick wins (each ~50–150 LOC)

Small, self-contained checks that add immediate diagnostic value.

### 0. Tier detection check (ships with Lane Tier Architecture)

Added as part of the [lane tier plan](../archive/lane_tier_plan.md) Phase 7. `check_tier_detection()` reports which tiers (large/medium/small) are available based on `RuntimeCapabilities` + `detect_tiers()`. Warns when only `small` is detected and provides fix hints (`--language-model`, `--cloud-fallback`, `--tier-model`). Fits after `check_gpu()` in `run_all_checks()`. ~40 LOC.

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

These compose naturally into a `maxim doctor benchmark` subcommand that produces a performance baseline document (complementing the [agent_mesh.md observability section](agent_mesh.md) numbers).

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
- **Key validity across peers** — verify each peer's stored API key still matches what the leader's issued (for per-device keys from the [security model](agent_mesh.md#phase-10-observability--verbose-tracing)).
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

## Capability Agent — continuous runtime awareness (~300–500 LOC)

Beyond one-shot diagnostics: a **CapabilityAgent** that maintains a live picture of what this system (and its peers) can do, and gates actions that exceed those capabilities. Runs as a lightweight background service alongside the agent loop.

### Problem

Today, capability awareness is fragmented:
- `detect_tiers()` runs once at startup and never updates
- The benchmark runner blindly attempts models the hardware can't run
- The sim orchestrator loads a 14B model on a Mac that should route to the leader
- No system knows what models are available across the mesh at any given moment
- A peer going offline mid-sim causes a hard failure instead of graceful rerouting

### CapabilityAgent design

```python
class CapabilityAgent:
    """Continuous awareness of what this system can do.

    Maintains a live CapabilitySnapshot that other subsystems query
    before attempting actions. Updated on events (peer join/leave,
    GPU thermal throttle, model swap) and periodically (heartbeat).
    """

    def __init__(self, caps: RuntimeCapabilities, peer_registry=None):
        self._local_caps = caps
        self._peer_registry = peer_registry
        self._snapshot = CapabilitySnapshot(...)
        self._lock = threading.RLock()

    # ── Queries (called by other subsystems) ──

    def can_run_model(self, profile: str) -> ModelAvailability:
        """Check if a model can run locally, remotely, or not at all.
        Returns: ModelAvailability(where="local"|"remote"|"cloud"|"unavailable",
                                   node=..., estimated_latency_ms=..., reason=...)
        """

    def available_models(self) -> list[ModelInfo]:
        """All models runnable right now (local + peer + cloud)."""

    def recommended_tier(self, function: str) -> TierRecommendation:
        """Best tier for a function given current load + capabilities.
        Considers: local hardware, peer availability, queue depth, cost.
        """

    def gate_action(self, action: str, requirements: dict) -> GateResult:
        """Should this action proceed? Returns allow/deny with reason.
        Examples:
          gate_action("benchmark", {"models": ["qwen2.5-14b"]})
            → deny: "qwen2.5-14b requires 10GB+ VRAM, local has 0GB, leader unreachable"
          gate_action("sim", {"model": "mistral-7b"})
            → allow: "routing to leader (RTX 5080), estimated 44ms latency"
        """

    # ── Updates (called by events) ──

    def on_peer_joined(self, peer_info): ...
    def on_peer_left(self, peer_id): ...
    def on_model_swapped(self, tier, new_profile): ...
    def on_heartbeat(self, metrics_snapshot): ...
    def on_thermal_throttle(self, gpu_temp): ...
```

### CapabilitySnapshot

A frozen-at-a-point-in-time view of the full system:

```python
@dataclass
class CapabilitySnapshot:
    timestamp: float

    # Local hardware
    local_tiers: dict[str, LaneConfig]      # From detect_tiers()
    local_gpu: GPUState | None              # VRAM free/total, temp, utilization
    local_ram_free_gb: float
    local_disk_free_gb: float

    # Models available right now
    local_models: list[str]                  # GGUF files present + loaded
    remote_models: dict[str, str]            # peer_id → model currently loaded
    cloud_models: list[str]                  # Cloud profiles with valid API keys

    # Peer state (from PeerRegistry / heartbeat)
    peers: list[PeerCapability]              # Each peer's tiers + load + latency
    leader_available: bool
    leader_queue_depth: int
    leader_model: str | None

    # Aggregate
    total_gpu_vram_gb: float                 # Sum across local + peers
    total_cpu_ram_gb: float
    strongest_tier: str                      # "large" if any node has it
```

### Integration points

| Consumer | How it uses CapabilityAgent |
|----------|---------------------------|
| **Benchmark runner** | `available_models()` to filter `--models` list before running. Skip models that can't run anywhere with clear message: "Skipping qwen2.5-14b: requires 10GB VRAM, best available is 0GB (Mac peer). Add --cloud-fallback or run on leader." |
| **Sim orchestrator** | `can_run_model(aut_model)` before building AUT router. Routes to leader if local can't handle it. |
| **FunctionRouter** | `recommended_tier(function)` as the dynamic `health_check` callback. Considers real-time load, not just static tier existence. |
| **`maxim doctor`** | `snapshot()` powers a rich capability report: what you can run, where, estimated performance. Replaces the static `check_tier_detection()`. |
| **Agent mesh** | `on_peer_joined/left` keeps the capability picture current as the mesh topology changes. |
| **Default Network** | Gate exploration actions: "don't attempt tool X, it requires a model we can't run right now." |
| **CLI pre-flight** | Before any `--sim` or `--language-model` command, quick gate check: "this model needs leader, leader is unreachable → fail fast with fix hint." |

### Model availability check (for benchmarks)

```python
def check_model_availability(models: list[str]) -> dict[str, ModelAvailability]:
    """Pre-flight check for benchmark runner.

    For each model:
      1. Local VRAM sufficient? (detect_tiers + _INFER_VRAM_TIERS)
      2. GGUF downloaded? (_profile_has_local_file)
      3. Leader can run it? (peer config + /v1/models probe)
      4. Cloud profile exists? (_BUILTIN_PROFILES with cloud: True)

    Returns dict of model → availability with routing recommendation.
    """
```

The benchmark runner calls this before its run loop and either:
- Routes each model to the right node (local vs leader vs cloud)
- Skips unavailable models with an actionable message
- Warns when a model will run on slow hardware (Mac CPU vs leader GPU)

### Gating + proactive suggestions

The CapabilityAgent doesn't just block — it **advises and suggests**. When a requested action exceeds local capabilities, it proactively recommends alternatives:

- "qwen2.5-14b can't run locally (no GPU). Leader has an RTX 5080 — route there? Or use --cloud-fallback claude-sonnet."
- "mistral-7b will run locally but expect ~60s/turn on CPU. Leader can do it in ~2s. Routing to leader."
- "Leader is under load (queue depth 3). Running concept extraction locally on small tier instead."
- "You haven't tried llama-3-8b yet — it fits your VRAM and benchmarks show 2x better tool compliance than mistral-7b on this task."

The agent should be **proactive** — not just answering "can I?" but volunteering "here's what I'd recommend given what I know about this system." This makes it feel like a teammate that knows the hardware, not just a gatekeeper.

### Relationship to existing systems

Wraps existing infrastructure — no replacement:
- `detect_tiers()` → startup detection; CapabilityAgent adds runtime updates
- `FunctionRouter` → function→tier mapping; CapabilityAgent feeds dynamic health via `health_check` callback
- `LaneMetrics` → per-tier counters; CapabilityAgent reads for load awareness
- `RuntimeCapabilities` → hardware dataclass; CapabilityAgent enriches with peer + cloud
- `maxim doctor` → user-facing diagnostic; CapabilityAgent provides live data

### Implementation phases

| Phase | What | LOC | Depends on |
|-------|------|-----|------------|
| CA-1 | `CapabilitySnapshot` + `check_model_availability()` | ~100 | Lane tiers (done) |
| CA-2 | `CapabilityAgent` with local + leader awareness | ~150 | CA-1 |
| CA-3 | Benchmark + sim pre-flight gates | ~100 | CA-2 |
| CA-4 | Peer awareness + mesh integration | ~100 | Agent Mesh Phase 0a |
| CA-5 | FunctionRouter health_check + proactive suggestions | ~50 | CA-2 |
| **Total** | | **~500** | |

CA-1 through CA-3 ship before the agent mesh. CA-4 and CA-5 integrate with mesh discovery. Full design deferred to its own plan when implementation starts.

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

## Mother Maxim Diagnostics (post-publication)

When Mother Maxim ships, `maxim doctor` needs new check categories. These are additive — same `CheckResult` pattern, new check functions.

### Contributor diagnostics (`maxim doctor --as contributor`)

Checks for users who want to contribute memories to Mother:

| Check | What it verifies | Fix hint |
|-------|-----------------|----------|
| Mother reachability | DNS + HTTPS + `/v1/stats` responds | "Mother is at <url>. Check your network." |
| API key valid | Auth against `/v1/session` | "Get a contributor key at <url>/register" |
| Client version | `client_version >= MINIMUM_CLIENT_VERSION` | "Run `pip install --upgrade pymaxim`" |
| Model tier sufficient | Session's LLM profile meets `MINIMUM_DEIDENTIFICATION_TIER` | "Your model (smollm-1.7b) is below minimum. Use mistral-7b or higher." |
| Deidentification pipeline | Run `ContributionPreparer.prepare()` on a synthetic memory, verify it works | "Install pymaxim[semantic] for embedding support" |
| Bio-system health | Hippocampus/ATL/NAc can save/load (store protocols working) | "Your memory state may be corrupted. Run maxim doctor --fix memory" |

### Mother operator diagnostics (`maxim doctor --as mother`)

Checks for the operator running Mother (you):

| Check | What it verifies | Fix hint |
|-------|-----------------|----------|
| PostgreSQL reachable | Connect to configured DB, verify schema | "PostgreSQL not running. `docker compose up -d postgres`" |
| pgvector extension | `CREATE EXTENSION IF NOT EXISTS vector` works | "Install pgvector: `apt install postgresql-16-pgvector`" |
| Mother agent running | Process alive, agent loop cycling | "Start with `maxim mother start`" |
| Memory stats | Total memories, growth rate, last contribution time | Informational — no fix needed |
| Deidentification stats | Rejection rate, quarantine depth, flagged tenants | "High rejection rate (>30%) — check deidentification model quality" |
| Coalescence health | Merge rate, consensus convergence, contradiction count | Informational |
| Sleep/circadian | Last sleep time, consolidation stats, SCN accuracy | "Mother hasn't slept in 48h — check SCN configuration" |
| Cognitive health | Recall precision trend, NAc confidence trend, ATL concept count | "Recall precision declining — investigate memory quality" |
| Security | No host paths in API responses, auth enforced, rate limits active | "Security hardening incomplete. Run stress test." |
| Disk/DB size | Database size, growth projection, backup recency | "Database at 80% of disk. Last backup: 7 days ago." |

### Federation diagnostics (`maxim doctor --as federation`) — future

| Check | What it verifies |
|-------|-----------------|
| Peer Mothers reachable | Each known Mother responds to `/v1/stats` |
| Domain coverage | Which domains are covered, which have gaps |
| Consensus health | Are peer Mothers converging or diverging? |
| Clock sync | SCN drift between Mothers |
| Cross-Mother latency | Round-trip p50 for knowledge sharing |

### CapabilityAgent integration

The CapabilityAgent (already designed above) absorbs Mother-awareness:

```python
# New methods on CapabilityAgent
def can_contribute(self) -> ContributionReadiness:
    """Pre-flight for contribution: model tier, deidentification, Mother reachability."""

def mother_health(self) -> MotherHealthSnapshot:
    """Live cognitive health metrics from Mother's bio-systems."""
```

`maxim doctor` calls these for the contributor/operator checks. The CapabilityAgent provides the data; doctor provides the formatting + fix hints + retry loop.

---

## Non-goals

Things that sound doctor-adjacent but belong elsewhere:

- **Model downloading** — `scripts/download_models.sh` already owns this. Doctor just verifies files exist.
- **Configuration editing** — doctor reports state, doesn't mutate config files (except via `--fix` which is its own opt-in).
- **Process supervision** — `LocalServerSpawner` owns subprocess lifecycle. Doctor observes it.
- **Full benchmark suite** — if benchmarks grow beyond a few checks, they belong in a dedicated `maxim benchmark` subcommand, not in doctor.
