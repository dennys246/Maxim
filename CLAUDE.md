# CLAUDE.md

## Project Overview

Maxim is a bio-inspired cognitive architecture for AI agents. It combines a 5-agent pipeline (Perception, Memory, Exec, Goal, Statistician) with biological memory systems (Hippocampus, ATL, Angular Gyrus, SCN, NAc) and a reactive Default Network. Works headless, in simulation, or connected to a robot.

## When making changes — required checks

Run these before considering any non-trivial task done:

```bash
# Lint + format
ruff check src/ tests/
ruff format src/ tests/

# Tests (fast suite)
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# If touching memory/, decisions/, integration/memory_hub.py:
python -m pytest tests/integration/test_memory_hub.py -q
```

Additional guardrails:
- Prefer editing existing modules over creating new ones — this codebase favors many small files already
- Don't rename bio-system classes (Hippocampus, ATL, NAc, SCN, EC, AngularGyrus) — names are load-bearing for the mental model
- If you touch provenance, run a sim with `MAXIM_PROVENANCE_VERBOSITY=2` and eyeball the trace
- **Run `mypy` on public API files** after changing api.py, session.py, create.py, load.py, or __init__.py: `mypy src/maxim/__init__.py src/maxim/api.py src/maxim/session.py src/maxim/create.py src/maxim/load.py --ignore-missing-imports`
- **Run `ruff format`** after any changes: `ruff format src/ tests/`

## Lessons learned (bugs that bit us)

**Mutable globals + module extraction:** When extracting module-level mutable globals (like `_active_spawner`) into a new file, do NOT re-import them by name (`from new_module import _active_spawner`). Python binds by value at import time — assignments in the importing module diverge from the source. Use module reference instead: `import new_module as _mod; _mod._active_spawner = value`. Functions are safe to re-import (they close over their own module's namespace).

**Auth in health probes:** Any HTTP health check that probes an endpoint behind API key auth MUST include the auth header. The leader's `_probe_upstream_ready()` was silently getting 401s from an auth-gated llama-cpp-server, causing `llm_ready` to be permanently false. Always send auth in probes, and treat 401 as "server is up" (auth-gated but alive).

**NAc class name:** The class is `NAc` (in `decisions/nac.py`), NOT `NucleusAccumbens`. Old code may reference the wrong name — always grep for `NucleusAccumbens` after touching NAc-related code.

**Lane tier names:** The canonical tier names are `"large"`, `"medium"`, `"small"`. The old names `"infer"`, `"review"`, `"record"` have been fully removed. Do not re-introduce them.

**Startup ordering in cli.py:** The LeaderProxy MUST start BEFORE `_normalize_args()` because arg normalization can trigger heavy CUDA imports (5-15s on GPU systems). Peers polling for the proxy during restart will time out if the proxy starts after these imports.

**Dead code accumulates silently:** Before publishing or after major refactors, grep for orphan modules: `.py` files whose basename doesn't appear in any `import` statement. We found 15 dead modules (~8,500 LOC) shipping in the wheel.

**Opt-in env vars in hot startup paths need autouse scrubs:** When you wire a new `if os.environ.get("MAXIM_FOO"): do_side_effect()` branch into anything reachable from `build_primary_router` (auto-spawn, tier detection, ensure_available, ...), pair it in the same commit with a `@pytest.fixture(autouse=True)` env-scrub fixture in [tests/conftest.py](tests/conftest.py). Without it, ANY test that sets the env var (e.g., a `normalize_args` unit test asserting `--auto-download` populates the var) leaks into every later test that constructs the runtime — and the leaked side effect runs for real. P5 cost a 9-minute pytest hang on a real 1 GB GGUF download to `~/.maxim/` before this was caught. The two existing scrubs (`_isolate_maxim_llm_profile_env`, `_isolate_maxim_auto_download_env`) are the template.

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
- `checks.py` — individual check functions, each returns a `CheckResult` (status: ok/warn/fail/info)
- `cli.py` — `maxim doctor` and `maxim peer test` subcommands

**Check surface (v2):** GPU/CUDA, tier detection, llama-cpp-server, auto-spawn reachability, inference coherence, leader role, LAN access, cloudflared, tunnel config/sync, API key (presence + age + permissions + auth smoke), disk space, RAM headroom, lane metrics. Peer mode: URL reachability, key check, auth, model availability, latency.

**CLI flags:** `--retry` (interactive fix loop), `--json` (machine-readable output), `--as peer <url>` / `--as leader` / `--as solo` (role override).

Companion: `maxim tunnel` subcommand in [src/maxim/tunnel/](src/maxim/tunnel/) (cloudflared wrapping + API key management).

### Maintaining this over time

**Adding a new check:**
1. Write a pure function in `doctor/checks.py` that takes `PlatformInfo` (if platform-aware) and returns a `CheckResult`.
2. Add it to the correct section in `run_all_checks()`. For peer-only checks, add to the `detected_role == "peer"` branch; for leader/solo checks, add to the `else` branch.
3. If the fix differs per platform, branch on `info.runtime` / `info.os` / `info.distro` inside the check and produce platform-specific `fix` strings with user-visible commands (users copy-paste, so make them runnable as-is).
4. Use actual detected values (IPs, paths) in fix strings — call `detect_wsl_ip()` / `detect_lan_ip()` rather than `<your-ip>` placeholders when possible.
5. Add a unit test in `tests/unit/test_doctor.py`; mock out network/process calls so tests run offline.

**When a check references another module's function** (e.g., `find_cloudflared`, `_llm_server_responding_at`), import inside the function body (not module-level) to keep `maxim doctor` fast when unused features aren't installed. Tests must patch the **original** module path (`maxim.tunnel.cloudflared.find_cloudflared`), not `maxim.doctor.checks.find_cloudflared`.

**Retry loop** (`maxim doctor --retry`): the loop is data-driven — any `CheckResult` with a `retry_id` and non-ok status is automatically included. Add the retry_id to the `retryable_fns` dict in `cli._retry_loop` with a callable that re-runs the check.

**Role detection** (`_detect_doctor_role()`): auto-detects from `MAXIM_LANE_LARGE_REMOTE_URL`. A non-localhost URL triggers peer mode. Override with `--as peer/leader/solo`.

**Adding a new platform:** extend `PlatformInfo`'s `OSName` / `Runtime` / `Distro` Literal types + the detection branches in `platform_detect.py`, and add fix-hint branches in every platform-aware check.

**`maxim peer test`** should stay self-contained — no imports from the agent runtime. It's run from peer machines that may not have the full dependency set installed.

**Don't:**
- Don't auto-execute fixes without the user asking (`--fix` flag is explicit opt-in).
- Don't make checks slow (> 1s). Network probes use short timeouts (1.5–2s). Long-running benchmarks belong in a future `maxim benchmark` subcommand.
- Don't silently drop failures — any failing check needs a user-actionable `fix` string.

Remaining doctor enhancements are tracked inline as TODO comments in [src/maxim/doctor/](src/maxim/doctor/); no standalone plan doc.

## Key Commands

```bash
# Agent runtime
maxim --llm mistral-7b                       # local LLM
maxim --llm claude-sonnet                    # Claude (needs ANTHROPIC_API_KEY)

# Model management
maxim --list-models                          # show models + download status
maxim --delete-model llama-2-13b-chat        # free disk space

# Simulation
maxim --sim "test memory recall"             # generative campaign
maxim --sim scenarios/campaigns/heist_v1.yaml  # DM campaign
maxim --sim "test safety" --persona adversarial --research  # with research report
maxim --sim benchmark --models mistral-7b,qwen2.5-14b      # benchmark

# Diagnostics + networking
maxim doctor                                 # environment check
maxim doctor --retry                         # interactive fix loop
maxim tunnel setup                           # Cloudflare tunnel
maxim peer update && maxim peer restart      # remote update

# Tests
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
```

Full CLI reference: [docs/user/cli-reference.md](docs/user/cli-reference.md)

## Remote Update Workflow

```bash
git push origin main && maxim peer update && maxim peer restart
```

Use `--dry-run` first if unsure. Use `--force` if the leader has untracked runtime files blocking the pull. Troubleshooting: [docs/troubleshooting/remote_update.md](docs/troubleshooting/remote_update.md).

**Important for Claude agents:** `maxim peer update --dry-run`, `maxim peer version`, `maxim peer logs`, and `maxim peer llm --status` are safe and read-only. `maxim peer update`, `maxim peer restart`, and `maxim peer llm <model>` modify leader state — only run when explicitly asked by the user.

## Versioning

Version is defined in two places that **must stay in sync**: `pyproject.toml` and `src/maxim/__init__.py`.

**When to bump:** Any change that affects runtime behavior, CLI interface, or peer/leader protocol. Docs-only or test-only changes do not require a bump.

**Check versions:** `python -c "from maxim import get_version_info; print(get_version_info())"` or `maxim peer version` to compare local vs leader. Version mismatch means the leader needs `maxim peer update && maxim peer restart`.

## Architecture Essentials

Project structure is documented in [docs/reference.md](docs/reference.md).

- **Agent loop** lives in `runtime/agent_loop.py` with `LoopController` in `runtime/loop_controller.py`
- **Multi-agent runtime**: `AgentFactory` in `runtime/agent_factory.py` creates independent agent instances (NPC agents with isolated Hippocampus, NAc, ATL). `AgentPool` in `runtime/agent_pool.py` orchestrates concurrent multi-agent execution with `LocalMessageBus`.
- **LLM routing** lives in `models/language/router.py` (config in `models/language/config.py`). 8 cloud providers (Anthropic, OpenAI, Google Gemini, Groq, Together, Fireworks, Mistral, DeepSeek) across 15 cloud profiles, plus 15 local profiles (llama-cpp and PyTorch/Transformers backends).
- **Simulation** orchestrator in `simulation/orchestrator.py`, bridge in `simulation/bridge.py`. Campaign runners in `simulation/campaign_runner.py`. Types in `simulation/sim_types.py`.
- **Interactive runtime** in `interactive/` — universal prompt protocol (`PromptRequest`/`PromptHandler`), rich terminal display with split panels, DM display extensions.
- **Mode system**: ProcessingState (awake/sleep) x OperationalMode (planning/supervised/autonomous). Sleep is a tool the agent calls; it wakes automatically on user input.
- **Memory tiers**: FORMING -> WORKING -> SHORT_TERM -> LONG_TERM (enforced by `TierTransitionError` in `agents/bus.py` — see F0.7)
- **Memory store protocols**: `EpisodicStore`, `CausalStore`, `SemanticStore` in `memory/store.py` — split persistence protocols with `File*Store` defaults and database implementations for Mother Maxim.
- **Percept/Reaction dual surface** (reaction_abstraction_plan, Phases 1–4 shipped):
  - **Percept** = sensory/environmental input. Typed `PerceptContext` (channel, sender, agent_id, scn_tag) in `agents/percept_context.py`. Named factories in `agents/percept_factory.py`: `make_text_percept`, `make_scene_percept`, `make_intero_percept`. `SensoryTag` populated at all producers via `agents/modality.py`.
  - **Reaction** = evaluative signal driving learning. Types in `reactions/types.py` (`Reaction`, `ReactionContext`, `TraceSnapshot`, `ReactionKind`). `ReactionBus` in `reactions/bus.py` (generalized from PainBus, per-kind dispatch). `PerceptProducer`/`ReactionProducer` protocols in `reactions/protocols.py`.
  - **SEM integration**: sensors → PerceptProducer (via `EmbodimentPerceptSource`), modulators → ReactionProducer (via `CerebellumModulator` mediation). SEM specs don't import Reaction types.
  - **Runtime unification**: both MaximAgent and AgentPool produce typed Percepts. AgentPool.run_turn wraps string input via `make_text_percept`.
  - **Isolation hygiene**: PerceptContext and ReactionContext must NOT carry cross-agent intent, private state, scenario oracles, or learned-policy hints. Rules documented in module docstrings of `percept_context.py` and `reactions/types.py`.
- **Lane tier system**: Functions route to capability tiers (large/medium/small) via `FunctionRouter` in `runtime/function_router.py`. `detect_tiers()` in `lane_models.py` auto-detects from hardware.
- **Data paths**: Bundled seed data in `src/maxim/_data/` (components, encounters, prompts, templates). User data at `~/.maxim/` (memory, sessions, benchmarks, config). Resolution via `utils/paths.py`.
- **SEM Component Registry**: `embodiment/component_registry.py` discovers SEM entity templates from campaign-local, `~/.maxim/components/`, and `_data/components/`. 54 seed components across 7 categories (bodies, creatures, environments, items, npcs, vehicles, weapons). Genre-gated: fantasy, cyberpunk, scifi, horror, historical, modern, devops.
- **Thread model**: Main loop at 2-30Hz + WorkerPool (tier-based lanes: large/medium/small, owned by LLMWorker) + Hippocampus capture thread (owned + shut down by MemoryHub.on_session_end)

## Quick reference — where to look

| Area | Key files |
|---|---|
| Agent loop | `runtime/agent_loop.py`, `runtime/loop_controller.py` |
| Tools | `tools/` (register in registry), `runtime/executor.py` (aliases) |
| LLM routing | `models/language/router.py`, `models/language/config.py` (profiles), `models/language/json_parser.py` (JSON repair) |
| Memory | `memory/hippocampus.py`, `memory/concept_extractor.py`, `memory/store.py` (protocols), `memory/percept_trace_buffer.py` (τ-decay ring buffer) |
| Causal learning | `decisions/nac.py` |
| Percept schema | `agents/percept_context.py` (PerceptContext), `agents/percept_factory.py` (factories), `agents/modality.py` (SensoryTag) |
| Reactions | `reactions/types.py` (Reaction, ReactionContext, TraceSnapshot), `reactions/bus.py` (ReactionBus), `reactions/protocols.py` (PerceptProducer, ReactionProducer) |
| Cross-layer wiring | `integration/memory_hub.py` (single coordinator) |
| Persistence | `utils/atomic_io.py`, `utils/paths.py` (data path resolution) |
| Simulation | `simulation/orchestrator.py`, `simulation/bridge.py`, `simulation/personas.py` |
| Generative campaigns | `simulation/arcs.py`, `simulation/narrator.py`, `simulation/generative_runner.py` |
| DM campaigns | `simulation/dm_schema.py`, `simulation/dm_runtime.py` |
| Benchmarks | `simulation/benchmark.py`, `simulation/validation.py` |
| Research | `simulation/research_agents.py`, `simulation/research_orchestrator.py` |
| Embodiment | `embodiment/sem.py`, `embodiment/body.py`, `embodiment/cerebellum.py`, `embodiment/motor.py` |
| Mesh | `mesh/identity.py`, `mesh/knowledge.py`, `mesh/task_delegation.py`, `mesh/clock.py` |
| Lane tiers | `runtime/function_router.py`, `runtime/lane_models.py`, `runtime/lane_backends.py` |
| Multi-agent | `runtime/agent_factory.py`, `runtime/agent_pool.py` |
| Interactive UI | `interactive/prompts.py`, `interactive/display.py` |
| Seed data | `_data/components/`, `_data/encounters/` |
| Adding env vars | Add to the env table below + touch whatever reads it |

## Environment Variables

```bash
ANTHROPIC_API_KEY          # Required for Claude backend
OPENAI_API_KEY             # Required for OpenAI backend
GOOGLE_API_KEY             # Required for Gemini backend
GROQ_API_KEY               # Required for Groq backend
TOGETHER_API_KEY           # Required for Together backend
FIREWORKS_API_KEY          # Required for Fireworks backend
MISTRAL_API_KEY            # Required for Mistral API backend
DEEPSEEK_API_KEY           # Required for DeepSeek backend
MAXIM_LLM_ENABLED=1        # Enable LLM inference
MAXIM_LLM_PROFILE=claude-sonnet  # Default model profile
MAXIM_LLM_N_CTX=4096       # Override auto-computed llama.cpp n_ctx (P4c). Same as --llm-n-ctx.
MAXIM_AUTO_DOWNLOAD_MODELS=1     # Skip the auto-download prompt (P5). Same as --auto-download.
MAXIM_DATA_BUDGET_GB=50          # Optional soft cap on ~/.maxim disk usage; refuses downloads over budget.
MAXIM_SKIP_REMOTE_PROBE=1        # Bypass the remote-URL probe (P6) — CI escape hatch.
MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S=0.8   # First probe attempt timeout (clamped 0.2-5.0)
MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S=2.5   # Retry probe timeout (clamped 0.5-10.0)
MAXIM_REMOTE_PROBE_CACHE_TTL_S=60        # Probe cache freshness window (clamped 0-600)

# Decision log (P9). Append-only JSONL at ~/.maxim/util/lane_decisions.jsonl.
# Inspect with: maxim doctor --last-decision   (or tail/jq the file directly)
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

# Cloud provider integration
MAXIM_LLM_CLOUD_ENABLED=1       # Enable cloud dispatch (required for --cloud-* flags)
MAXIM_MAX_CLOUD_LANES=1          # Max lanes using cloud providers (default: 0)
MAXIM_LLM_REDACTION_POLICY=standard  # Redaction policy for cloud dispatch (standard/relaxed/strict)
MAXIM_CLOUD_SESSION_BUDGET=5.00  # Hard ceiling on cloud spending per session

# Peer/tier remote configuration (tier names: large, medium, small)
MAXIM_LANE_LARGE_REMOTE_URL=     # Override large tier to use remote server
MAXIM_LANE_LARGE_REMOTE_MODEL=   # Model name to request from remote server
MAXIM_LANE_LARGE_REMOTE_API_KEY= # Auth token for remote server
# Tier names only: large, medium, small (legacy infer/review/record removed in v1.0)
```

## Testing

```bash
# Full suite
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# Specific test file
python -m pytest tests/unit/test_simulation_agent.py -v

# Just the module you changed (fast feedback)
python -m pytest tests/unit/test_lane_metrics.py -v
```

### Testing efficiently

**Run narrow first, then wide.** Test the specific module you changed before running the full suite (~3 min). The full suite has ~4,000 tests; don't wait for all of them on every edit.

**Kill stale sims before running tests.** A running `maxim --sim agent` process holds GPU + port resources and can cause test hangs:
```bash
pkill -f "maxim.*sim" 2>/dev/null; sleep 2
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
```

**Threading pitfalls (learned the hard way):**
- Use `threading.RLock` (not `Lock`) if a method acquires the lock and then calls another method that also acquires it (e.g. `snapshot()` calling `self.failure_rate`). Regular `Lock` deadlocks on re-entry.
- Thread-safety tests with many workers (8+ threads × 100 calls) can appear to hang if a deadlock exists — they're not slow, they're stuck.

**Don't run sims from tests.** Sims call real LLMs and can 2-3x test-suite runtime. The sim runner is for manual/CLI testing only (`maxim --sim agent`). Tests should mock LLM calls.

**Peer/tunnel testing requires the leader.** Use `curl -si -H "Authorization: Bearer $KEY" https://maxim.yourdomain.com/v1/models` for quick checks. See [docs/troubleshooting/](docs/troubleshooting/) for in-depth peer connectivity guides.

## Simulation Reports

Sim runs save to `~/.maxim/sessions/{session_id}/` (report.json, actions.jsonl, aut_hippocampus.json, aut_nac.json). Research protocol details and campaign execution flow are documented in `docs/simulation.md` and `docs/experiments/`.

## Python API (pymaxim)

Published to PyPI as `pymaxim` (import name stays `maxim`). 17 verb-based functions, all lazy-loaded from `src/maxim/api.py`. Key files: `api.py` (facades), `__init__.py` (lazy wiring), `simulation/introspection.py` (Observer).

**Rules for maintaining the API:**
- **Verbs are facades, not logic.** Delegate to existing internals. Don't put business logic in `api.py`.
- **Lazy imports only.** `import maxim` must not trigger loading of optional dependencies.
- **Return structured data, not prints.** `diagnose()` returns `DiagnosticReport`, `imagine()` returns `SimulationResult`, etc.
- **`introspect` is an alias for `observe`.** Don't add behavior to one without the other.
- **`Observer`** is the canonical name (no aliases).

**Package management:**
- **Package name:** `pymaxim` on PyPI, `maxim` as import
- **Core deps:** `numpy`, `scipy`, `pyyaml`, `json-repair` only. Everything else is optional extras.
- **Optional extras:** `llm-llama`, `llm-server`, `llm-torch`, `llm-anthropic`, `llm-openai`, `vision`, `audio`, `reachy`, `comms`, `search`, `temporal`, `training`, `tts`, `yolo`, `semantic`, `database`
- **Robot plugins:** Auto-discovered via `maxim.robots` entry-point group.
- **Build validation:** `python -m build && twine check dist/*` before any publish
- Publication guide: [publication_guide.md](docs/publication_guide.md)

## Active initiatives

See [docs/plans/README.md](docs/plans/README.md) for the roadmap index. Current version: v0.2.1 on PyPI as `pymaxim` ([publication guide](docs/publication_guide.md)).

**Recently shipped (2026-04-11):**
- Foundations wave F0.1–F0.8 — all landed. Archived to [docs/plans/archive/foundations_plan.md](docs/plans/archive/foundations_plan.md).
- Reaction abstraction Phases 1–4 — Percept/Reaction dual-surface architecture, ReactionBus, producer protocols, factories, runtime unification. See [docs/plans/reaction_abstraction_plan.md](docs/plans/reaction_abstraction_plan.md).
- Cleanup wave C1–C4 — shipped in 0.2.2. Archived.
- Peer/leader flexibility P1–P9 — dynamic n_ctx, auto-download, remote probes, lane decision log. Archived.

**Gating 1.0** (next steps):
- [simulator_upgrades_plan.md](docs/plans/simulator_upgrades_plan.md) — S1–S4 test-harness infrastructure. Blocks substrate P0. ~850 LOC.
- [substrate_plan.md](docs/plans/substrate_plan.md) — bio-stack convergence (Track A: P0, P1–P6, P8) and prompt layer (Track B: B1–B5). Reaction abstraction Phase 5 (NAc structured percept-context access) folds into P2.

**Living practice docs (paired with substrate_plan):**
- [behavioral_convergence_practice.md](docs/plans/behavioral_convergence_practice.md) — does the agent actually get better across sessions? Living doc, not a gate.
- [memory_consolidation_practice.md](docs/plans/memory_consolidation_practice.md) — refines the P8 sleep-replay mechanism. Kicks in when P8 ships in 0.5.

**Parallel:**
- [tool_refinement_plan.md](docs/plans/tool_refinement_plan.md) — living doc for agent tool curation.

**Deferred (post-1.0, revive on trigger):** Bio-System Plugin Discovery, Unified Event Bus, Mother NPC Stimulus, Mother Maxim, Pecking Order Graph, Asset Foundry, DM Extensions. See [docs/plans/deferred/](docs/plans/deferred/).
