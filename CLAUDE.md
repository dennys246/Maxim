# CLAUDE.md

## Project Overview

Maxim is a bio-inspired cognitive architecture for robotic agents. It combines a 5-agent pipeline (Perception, Memory, Exec, Goal, Statistician) with biological memory systems (Hippocampus, ATL, Angular Gyrus, SCN, NAc) and a reactive Default Network.

## When making changes — required checks

Run these before considering any non-trivial task done:

```bash
# Lint + format
ruff check src/ tests/
ruff format src/ tests/

# Tests (fast suite)
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py

# If touching memory/, decisions/, integration/memory_hub.py:
python -m pytest tests/integration/test_memory_hub.py -q
```

Additional guardrails:
- Prefer editing existing modules over creating new ones — this codebase favors many small files already
- Don't rename bio-system classes (Hippocampus, ATL, NAc, SCN, EC, AngularGyrus) — names are load-bearing for the mental model
- If you touch provenance, run a sim with `MAXIM_PROVENANCE_VERBOSITY=2` and eyeball the trace

## Running simulations — keep them small

Simulations call a live LLM for every turn and can burn cost + time quickly. When running sims from this CLI (for diagnostics, verification, or debugging):

- **Set a narrow goal.** `--goal "test X specifically"` beats `--goal "test safety"` — specific goals converge faster.
- **Cap duration.** Hit Ctrl+C after 30–90 seconds when you've seen what you need. Sims report partial results on cancel.
- **Prefer --sandbox tmpdir for debugging** unless you're specifically testing Docker — tmpdir has no pull/startup cost.
- **Use --debug sparingly.** The verbose-trace output is great for diagnosing stalls but floods the terminal for routine runs.
- **Don't invoke sims from test suites** unless the test is specifically for sim machinery. The sim runner spins up real LLM calls and can 2-3x test-suite runtime.
- **Re-use sessions with --resume-sim SESSION_ID** to avoid re-running setup + warm-up costs when iterating on a specific run.
- **Local models > Claude for loop-testing.** Use `--language-model mistral-7b` for sanity checks; save Claude for verifying final behavior.
- **Watch for Cost:** in the final report. $0.05–$0.15 per short run is normal; $0.50+ for a single debug session suggests the sim is too broad or too long.

## Architectural invariants (do not break without discussion)

- **Memory tier progression is one-way**: FORMING → WORKING → SHORT_TERM → LONG_TERM. Don't skip or reverse.
- **Hippocampus, NAc, and ATL maintain SEPARATE EpisodicMemory instances** — this is intentional coexistence, not tech debt. Don't merge.
- **Tool results flow through the agent bus**; don't call agents directly from tools.
- **Persistence uses `maxim.utils.atomic_io.atomic_write_json`** (fsync + tmp cleanup). Don't hand-roll `open().write()` + `os.replace()`.
- **LLM access goes through `models/language/router.py`**; backends (anthropic/llama/openai/transformers) should not be imported directly from outside `models/language/`.
- **The WorkerPool is owned by LLMWorker**, which shuts it down on `stop()`. Don't create a parallel pool.
- **`@resilient` decorator (runtime/resilient.py) wraps any callback that can fail** — use it instead of bare `except Exception: pass`.

## `maxim doctor` — environment diagnostics

Runs platform-aware checks + prints fix hints with the user's actual IPs filled in.
Lives in [src/maxim/doctor/](src/maxim/doctor/) — three modules:

- `platform_detect.py` — OS + runtime (native/WSL1/WSL2/docker) + Linux distro
- `checks.py` — individual check functions, each returns a `CheckResult`
- `cli.py` — `maxim doctor` and `maxim peer test` subcommands

Companion: `maxim tunnel` subcommand in [src/maxim/tunnel/](src/maxim/tunnel/) (cloudflared wrapping + API key management).

### Maintaining this over time

**Adding a new check:**
1. Write a pure function in `doctor/checks.py` that takes `PlatformInfo` (if platform-aware) and returns a `CheckResult`.
2. Add it to the correct section in `run_all_checks()`.
3. If the fix differs per platform, branch on `info.runtime` / `info.os` / `info.distro` inside the check and produce platform-specific `fix` strings with user-visible commands (users copy-paste, so make them runnable as-is).
4. Use actual detected values (IPs, paths) in fix strings — call `detect_wsl_ip()` / `detect_lan_ip()` rather than `<your-ip>` placeholders when possible.
5. Add a unit test in `tests/unit/test_doctor.py`; mock out network/process calls so tests run offline.

**When a check references another module's function** (e.g., `find_cloudflared`, `_llm_server_responding_at`), import inside the function body (not module-level) to keep `maxim doctor` fast when unused features aren't installed. Tests must patch the **original** module path (`maxim.tunnel.cloudflared.find_cloudflared`), not `maxim.doctor.checks.find_cloudflared`.

**Retry loop** (`maxim doctor --retry`): add `retry_id` on any `CheckResult` the user can fix iteratively, then register the retry callable in `cli._retry_loop.retryable`.

**Adding a new platform:** extend `PlatformInfo`'s `OSName` / `Runtime` / `Distro` Literal types + the detection branches in `platform_detect.py`, and add fix-hint branches in every platform-aware check.

**`maxim peer test`** should stay self-contained — no imports from the agent runtime. It's run from peer machines that may not have the full dependency set installed.

**Don't:**
- Don't auto-execute fixes without the user asking (`--fix` flag is explicit opt-in; see doctor_upgrade_plan.md).
- Don't make checks slow (> 1s). Network probes use short timeouts (1.5–2s). Long-running benchmarks belong in a future `maxim benchmark` subcommand.
- Don't silently drop failures — any failing check needs a user-actionable `fix` string.

Upgrade roadmap in [docs/plans/doctor_upgrade_plan.md](docs/plans/doctor_upgrade_plan.md).

## Key Commands

```bash
# Run with local LLM
maxim --language-model mistral-7b

# Run with Claude (requires ANTHROPIC_API_KEY)
maxim --language-model claude-sonnet

# Simulation agent mode (recommended for testing)
maxim --sim agent --goal "test safety" --persona adversarial --language-model claude-sonnet

# Resume a previous simulation
maxim --sim agent --goal "continue" --resume-sim 20260403 --language-model claude-sonnet

# Run YAML scenario
maxim --sim scenarios/malware_with_pain.yaml

# Run tests
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py

# Environment diagnostics (platform-aware, with fix hints)
maxim doctor
maxim doctor --retry          # walk through failures, retest after each fix

# Cloudflare tunnel for remote access
maxim tunnel setup            # one-time guided setup
maxim tunnel status           # show what's configured
maxim tunnel key rotate       # generate/replace peer API key
maxim tunnel key export       # print shell-specific export snippets

# Verify peer connectivity (run from peer machines)
maxim peer test https://maxim.yourdomain.com/v1

# Remote update (push code to leader without SSH)
maxim peer update              # pull + pip install on leader
maxim peer update --dry-run    # preview pending commits only
maxim peer update --branch dev # target a specific branch
```

## Remote Update Workflow

After pushing code to origin, update the leader remotely:

```bash
git push origin main
maxim peer update          # leader pulls + installs automatically
```

**Best practices:**
- Always `git push` before `maxim peer update` — the leader pulls from origin, not from your local machine
- Use `--dry-run` first if you're unsure what will be pulled
- The leader must restart `maxim` after an update to load new code (soft restart is planned)
- If update fails with "dirty working tree", the leader has uncommitted files — commit or stash them on the leader
- If update fails with "git pull failed", the leader has divergent branches — run `git pull --rebase origin main` on the leader
- Leader mode auto-enables remote update; disable with `MAXIM_ALLOW_REMOTE_UPDATE=0` if needed
- Troubleshooting: [docs/troubleshooting/remote_update.md](docs/troubleshooting/remote_update.md)

**Important for Claude agents:** `maxim peer update --dry-run` is safe and read-only. `maxim peer update` (without `--dry-run`) modifies leader state — only run when explicitly asked by the user.

## Project Structure

```
src/maxim/
  agents/           # 5-agent pipeline (perception, memory, exec, goal, statistician)
  conscience/       # Main Maxim class (selfy.py) + 6 mixins + agentic runtime
  runtime/          # Agent loop, LoopController, SimulationAdapter, executor, worker pool
  memory/           # Hippocampus, ATL semantic memory, layer protocol
  math/             # IPS (fast stats) + Angular Gyrus (algebraic memory)
  decisions/        # NAc causal learning, adaptive planner
  models/language/  # LLM router, backends (llama-cpp, anthropic, openai, transformers)
  simulation/       # SimulationBridge, orchestrator, tools, personas, report system
  mesh/             # Agent mesh primitives (AgentProfile, UMR naming, MeshMessage, LocalMessageBus)
  tools/            # 40+ tools (filesystem, introspection, math, communication)
  default_network/  # Reactive behavior layer (thalamic gate, arbiter, behaviors)
  modes/            # Operating modes, strategies, exploration policy
  skills/           # Protocol/skill system for operational composition
  provenance/       # Decision tracing (2-tier: cycle traces + activity log)
  proprioception/   # Pain detection, movement tracking, focus learning
  attention/        # Spatial attention grid, gaze control
  salience/         # Novelty tracking, interest matching
  energy/           # Token/cost/compute tracking
  harm/             # Predictive harm detection
  time/             # SCN temporal rhythm indexing
  comms/            # Communication gateway (Twilio SMS/voice)
  integration/      # MemoryHub cross-system coordinator
  bridges/          # 8 cross-system integration bridges
  doctor/           # `maxim doctor` — platform-aware diagnostics + peer test
  tunnel/           # `maxim tunnel` — Cloudflare tunnel + API key management

docs/               # Internal architecture docs
docs/user/          # User-facing guides
docs/plans/         # Development roadmap
htmls-guides/       # Jinja2 HTML templates for dennyschaedig.com
tests/              # Unit + integration + benchmark tests
scenarios/          # YAML simulation scenarios
data/util/          # Runtime config (llm.json, cost_state.json)
```

## Architecture Essentials

- **Agent loop** lives in `runtime/agent_loop.py` with `LoopController` in `runtime/loop_controller.py`
- **LLM routing** lives in `models/language/router.py` (config in `models/language/config.py`)
- **Simulation** orchestrator in `simulation/orchestrator.py`, bridge in `simulation/bridge.py`
- **Mode system**: ProcessingState (awake/sleep) x OperationalMode (passive/active/singularity) x Strategy (6 types)
- **Memory tiers**: FORMING -> WORKING -> SHORT_TERM -> LONG_TERM
- **Thread model**: Main loop at 2-30Hz + WorkerPool (3 lanes: infer/review/record, owned by LLMWorker) + Hippocampus capture thread (owned + shut down by MemoryHub.on_session_end)

## Quick reference — where to look

| I'm touching... | Start here |
|---|---|
| The agent loop / step pipeline | `runtime/agent_loop.py`, `runtime/loop_controller.py` |
| Adding a new tool | `tools/` + register in the tool registry; see `tools/introspection.py` for a clean example |
| LLM prompts / routing | `models/language/router.py`, `models/language/prompt_formats.py` |
| Sim personas | `simulation/personas.py` (8 today: adversarial, cooperative, confused, escalating, campaign, refinement, researcher, sweep) |
| Memory capture → consolidation | `memory/hippocampus.py`, `memory/concept_extractor.py`, `memory/semantic_promoter.py` |
| Causal learning | `decisions/nac.py` |
| Cross-layer wiring | `integration/memory_hub.py` (the single coordinator) |
| Adding an env var | Put it here in the env table + touch whatever reads it |
| Atomic JSON persistence | `utils/atomic_io.py` |
| Mesh primitives (identity, messaging) | `mesh/` (AgentProfile, UMR, MeshMessage, LocalMessageBus) |
| Research experiment tracking | `simulation/research_tools.py` (ExperimentLog, record/query tools) |

## Environment Variables

```bash
ANTHROPIC_API_KEY          # Required for Claude backend
OPENAI_API_KEY             # Required for OpenAI backend
MAXIM_LLM_ENABLED=1        # Enable LLM inference
MAXIM_LLM_PROFILE=claude-sonnet  # Default model profile
MAXIM_PROVENANCE_VERBOSITY=1     # 0=off, 1=compact, 2=verbose

# Heartbeat + trace (debug/diagnostics)
MAXIM_HEARTBEAT=1                # System health heartbeat every 10s (GPU/CPU/RAM/disk/WiFi + stall detection)
MAXIM_HEARTBEAT_INTERVAL_S=10    # Heartbeat sample interval
MAXIM_HEARTBEAT_STALL_S=30       # Warn after this many seconds with no LLM calls
MAXIM_LANE_TRACE=1               # Per-request LLM trace logs (also enables heartbeat)
MAXIM_PEER_LOG_REQUESTS=1        # JSON log per outbound peer call

# Leader proxy admission control
MAXIM_PROXY_MAX_CONCURRENT=4     # Max in-flight requests to upstream (0=unlimited)
MAXIM_PROXY_RATE_LIMIT_RPM=0     # Per-peer requests/minute (0=unlimited)
```

## Testing

```bash
# Full suite (one pre-existing logic failure in test_record_plan_outcome)
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py

# Specific test file
python -m pytest tests/unit/test_simulation_agent.py -v

# Just the module you changed (fast feedback)
python -m pytest tests/unit/test_lane_metrics.py -v
```

Known pre-existing failure: `tests/integration/test_memory_hub.py::TestPlanningBridge::test_record_plan_outcome` — `record_plan_outcome` doesn't currently drive NAc's observation counter (the assertion `nac.stats()["total_observations"] > 0` fails). Not a blocker for the rest of the suite. (The NAc circular import that previously masked this was fixed in Wave A stabilization.)

### Testing efficiently

**Run narrow first, then wide.** Test the specific module you changed before running the full suite (~3 min). The full suite has 2500+ tests; don't wait for all of them on every edit.

**Kill stale sims before running tests.** A running `maxim --sim agent` process holds GPU + port resources and can cause test hangs:
```bash
pkill -f "maxim.*sim" 2>/dev/null; sleep 2
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py
```

**Threading pitfalls (learned the hard way):**
- Use `threading.RLock` (not `Lock`) if a method acquires the lock and then calls another method that also acquires it (e.g. `snapshot()` calling `self.failure_rate`). Regular `Lock` deadlocks on re-entry.
- Thread-safety tests with many workers (8+ threads × 100 calls) can appear to hang if a deadlock exists — they're not slow, they're stuck.

**Don't run sims from tests.** Sims call real LLMs and can 2-3x test-suite runtime. The sim runner is for manual/CLI testing only (`maxim --sim agent`). Tests should mock LLM calls.

**Peer/tunnel testing requires the leader.** Tests that exercise peer→leader inference need the leader machine running `maxim` with the tunnel up. Use `curl` probes first (fast, no Python overhead), then test through the LLMRouter:
```bash
# Quick connectivity check (no Maxim runtime needed)
curl -si -H "Authorization: Bearer $KEY" https://maxim.yourdomain.com/v1/models

# Full pipeline check (exercises lane wiring + provider routing)
MAXIM_LANE_TRACE=1 python -c "
from maxim.peer.config import read_peer_config, apply_peer_config_to_env
cfg = read_peer_config(); apply_peer_config_to_env(cfg)
from maxim.runtime.lane_backends import build_primary_router
router, mgr = build_primary_router()
print(router.generate_json('Reply: {\"ok\": true}', max_tokens=10))
"
```

**Troubleshooting docs**: [docs/troubleshooting/](docs/troubleshooting/) has in-depth guides for peer connectivity issues.

## Simulation Reports

Every sim run saves to `data/sim_reports/{session_id}/`:
- `report.json` -- Metrics + LLM analysis
- `actions.jsonl` -- Action records
- `aut_hippocampus.json` -- AUT memories
- `aut_nac.json` -- AUT causal links

## Active initiatives

See `docs/plans/future_plans.md` for the full roadmap. Current state:

- **Multi-LLM scaling** — COMPLETE. All phases done (LeaderProxy, admission control, LaneMetrics, heartbeat, remote update). Archived at `docs/plans/agent_mesh.md`. mDNS + InferenceRouter moved to Agent Mesh.
- **Agent Mesh** (`docs/plans/agent_mesh.md`) — Phase 1a-1b foundations in place (AgentProfile, UMR in `src/maxim/mesh/`). Phases 0a-0b (mDNS discovery + InferenceRouter) are next, then full protocol.
- **Research Protocol** (`docs/plans/research_protocol_plan.md`) — Phase 0 complete (mesh primitives: AgentProfile, UMR, MeshMessage, LocalMessageBus in `src/maxim/mesh/`). Phase 1 complete (record_experiment, query_experiments in `src/maxim/simulation/research_tools.py`). Phases 2-4 (Writer, Reviewer, Orchestrator) next.
- **Realtime Refinement** (`docs/plans/realtime_refinement_plan.md`) — ~90% done; remaining ~50 LOC (6th persona + metric expectation types).
- **Docker Sandbox** (`docs/plans/docker_sandbox_plan.md`) — Phase A (TmpdirSandbox + pain triggers) + Phase B (DockerSandbox + ContainerRunner protocol + image catalog + autonomy-scaled resource limits + unprivileged `maxim` user) DONE.
