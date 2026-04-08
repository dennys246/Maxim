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

**Role detection** (`_detect_doctor_role()`): auto-detects from `MAXIM_LANE_INFER_REMOTE_URL` / `MAXIM_LANE_LARGE_REMOTE_URL`. Non-localhost URLs trigger peer mode. Override with `--as peer/leader/solo`.

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

# Generative campaign (default — goal string triggers narrative arc)
maxim --sim "test memory recall under interference"
maxim --sim "test safety boundaries" --persona adversarial
maxim --sim "test skill learning" --arc scenarios/arcs/herbalism_skill.yaml

# With research report (Writer + Reviewer after sim)
maxim --sim "test memory recall" --research

# YAML campaign (direct injection — pass a .yaml path)
maxim --sim scenarios/experiments/hippocampal_recall_short.yaml

# Legacy agent mode (still works, deprecated)
maxim --sim agent --goal "test safety" --persona adversarial --language-model claude-sonnet

# Resume a previous simulation
maxim --sim agent --goal "continue" --resume-sim 20260403 --language-model claude-sonnet

# Research protocol (legacy — use --research flag instead)
maxim --sim research --goal "hippocampal recall under interference" --campaign scenarios/experiments/hippocampal_recall_short.yaml

# Dual-LLM research (Claude orchestrates, Mistral experiences)
maxim --sim "hippocampal recall" --research --language-model claude-sonnet --aut-model mistral-7b --campaign scenarios/experiments/hippocampal_recall_*.yaml

# Debug with subsystem tracing (hippo=memory, nac=causal, all=everything)
maxim --sim agent --goal "test" --debug hippo
maxim --sim agent --goal "test" --debug hippo,nac
maxim --sim agent --goal "test" --debug        # all subsystems

# Run YAML scenario
maxim --sim scenarios/malware_with_pain.yaml

# DM campaigns (auto-detected from YAML metadata)
maxim --sim scenarios/campaigns/heist_v1.yaml
maxim --sim scenarios/campaigns/poisoned_crown_v1.yaml
maxim --sim scenarios/campaigns/arena_v1.yaml
maxim --sim scenarios/campaigns/darkened_cavern_v1.yaml

# Benchmark (multi-model comparison)
maxim --sim benchmark --models mistral-7b,qwen2.5-14b --campaign scenarios/benchmarks/cognitive_suite.yaml
maxim --sim benchmark --models mistral-7b --campaign scenarios/benchmarks/quick_check.yaml --runs 3

# Run tests
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py

# Environment diagnostics (platform-aware, with fix hints)
maxim doctor
maxim doctor --retry          # walk through failures, retest after each fix
maxim doctor --json           # machine-readable output for CI/scripts
maxim doctor --as peer https://maxim.yourdomain.com/v1  # peer-mode checks
maxim doctor --as leader      # force leader-mode checks

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
maxim peer update --force      # stash dirty tree, pull, restore (handles runtime state files)
maxim peer restart             # soft-restart leader (reloads code after update)
maxim peer version             # compare local vs leader version + git hash
maxim peer logs                # show recent leader logs
maxim peer logs -f             # follow leader logs in real time (Ctrl+C to stop)

# Remote LLM hot-swap (change model without restarting Maxim)
maxim peer llm qwen2.5-14b    # swap leader's llama-cpp-server to Qwen2.5-14B
maxim peer llm mistral-7b     # swap to Mistral-7B
maxim peer llm --status        # show active model, uptime, GPU, lane metrics

# Cloud provider integration (optional fallback/dedicated tiers)
maxim --cloud-fallback claude-sonnet     # cloud fallback when self-hosted fails
maxim --cloud-lane small claude-haiku    # dedicated cloud model for small tier
maxim --cloud-budget 2.00               # set max session cost for cloud providers
```

## Remote Update Workflow

After pushing code to origin, update the leader remotely:

```bash
git push origin main
maxim peer update          # leader pulls + installs automatically
maxim peer restart         # soft-restart to load new code
```

**Best practices:**
- Always `git push` before `maxim peer update` — the leader pulls from origin, not from your local machine
- Use `--dry-run` first if you're unsure what will be pulled
- After `maxim peer update`, run `maxim peer restart` to reload the new code
- Use `--force` if the leader has untracked runtime files (e.g., `active_llm_model.txt`) blocking the pull
- If update fails with "dirty working tree", the leader has uncommitted files — commit or stash them on the leader
- If update fails with "git pull failed", the leader has divergent branches — run `git pull --rebase origin main` on the leader
- Leader mode auto-enables remote update + restart; disable with `MAXIM_ALLOW_REMOTE_UPDATE=0` if needed
- Troubleshooting: [docs/troubleshooting/remote_update.md](docs/troubleshooting/remote_update.md)

**Important for Claude agents:** `maxim peer update --dry-run`, `maxim peer version`, `maxim peer logs`, and `maxim peer llm --status` are safe and read-only. `maxim peer update`, `maxim peer restart`, and `maxim peer llm <model>` modify leader state — only run when explicitly asked by the user.

## Versioning

Version is defined in two places that **must stay in sync**:
- `pyproject.toml` line 7: `version = "X.Y.Z"`
- `src/maxim/__init__.py`: `__version__ = "X.Y.Z"`

**When to bump:** Any change that affects runtime behavior, CLI interface, or peer/leader protocol. Pure docs-only or test-only changes do not require a bump.

**Version is the source of truth on reboot.** After a restart or deploy, always verify the running version matches the expected git hash before assuming new code is live. The `get_version_info()` function reads the current git hash at runtime — if it doesn't match what was pushed, the code hasn't been reloaded.

**How to check versions:**
```bash
# Local version + git hash:
python -c "from maxim import get_version_info; print(get_version_info())"

# Compare local vs leader:
maxim peer version

# Query leader only (no auth needed for debug endpoints):
curl -s -H "User-Agent: maxim-peer/1.0" https://maxim.yourdomain.com/v1/debug/version
```

**Version mismatch between leader and peer** means the leader needs `maxim peer update && maxim peer restart` to sync.

**Startup priority:** On boot, the leader must fully initialize the LLM engine (CUDA binaries, model load, warmup) before serving inference requests. The `/v1/debug/version` endpoint is available immediately (served by LeaderProxy before LLM is ready), but `/v1/chat/completions` will 502 until the model is warm. CUDA binary loading can take 30-60s on first boot; subsequent restarts reuse the existing llama-cpp-server process and are faster.

## Project Structure

```
src/maxim/
  agents/           # 5-agent pipeline (perception, memory, exec, goal, statistician)
  conscience/       # Main Maxim class (selfy.py) + 6 mixins + agentic runtime
  runtime/          # Agent loop, LoopController, SimulationAdapter, executor, worker pool, FunctionRouter, AgentFactory, AgentPool
  memory/           # Hippocampus, ATL semantic memory, layer protocol, store protocols (EpisodicStore, CausalStore, SemanticStore)
  math/             # IPS (fast stats) + Angular Gyrus (algebraic memory)
  decisions/        # NAc causal learning, adaptive planner
  models/language/  # LLM router, backends (llama-cpp, anthropic, openai, transformers), 10 cloud provider profiles
  simulation/       # SimulationBridge, orchestrator, tools, personas, report system, DM party runtime, encounter library, entity designer
  mesh/             # Agent mesh: identity, protocol, transport, admission, knowledge sharing, delegation, clock sync
  tools/            # 40+ tools (filesystem, introspection, math, communication)
  interactive/      # Prompt protocol (PromptRequest/PromptHandler), rich terminal display, DM display panels
  default_network/  # Reactive behavior layer (thalamic gate, arbiter, behaviors)
  modes/            # Operating modes (passive/active/singularity), sleep tool
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
  embodiment/       # SEM protocol, body runtime, auto-tools, ComponentRegistry (template catalog)
  doctor/           # `maxim doctor` — platform-aware diagnostics + peer test
  tunnel/           # `maxim tunnel` — Cloudflare tunnel + API key management
  _data/            # Bundled seed data (components/, encounters/, prompts/, templates/)

docs/               # Internal architecture docs
docs/user/          # User-facing guides
docs/plans/         # Development roadmap
docs/experiments/   # Experiment designs + run notes
htmls-guides/       # Jinja2 HTML templates for dennyschaedig.com
tests/              # Unit + integration + benchmark tests
scenarios/          # YAML simulation scenarios
~/.maxim/           # User data home (memory, sessions, benchmarks, components, encounters, config)
```

## Architecture Essentials

- **Agent loop** lives in `runtime/agent_loop.py` with `LoopController` in `runtime/loop_controller.py`
- **Multi-agent runtime**: `AgentFactory` in `runtime/agent_factory.py` creates independent agent instances (NPC agents with isolated Hippocampus, NAc, ATL). `AgentPool` in `runtime/agent_pool.py` orchestrates concurrent multi-agent execution with `LocalMessageBus`.
- **LLM routing** lives in `models/language/router.py` (config in `models/language/config.py`). 10 cloud provider profiles: Gemini, Groq, Together, Fireworks, Mistral, DeepSeek (plus Anthropic, OpenAI, local llama-cpp, transformers).
- **Simulation** orchestrator in `simulation/orchestrator.py`, bridge in `simulation/bridge.py`. Party DM runtime in `simulation/dm_party.py` for multi-agent campaigns. Encounter library in `simulation/encounter_library.py` with seed templates.
- **Interactive runtime** in `interactive/` — universal prompt protocol (`PromptRequest`/`PromptHandler`), rich terminal display with split panels, DM display extensions.
- **Mode system**: ProcessingState (awake/sleep) x OperationalMode (passive/active/singularity). Sleep is a tool the agent calls; it wakes automatically on user input.
- **Memory tiers**: FORMING -> WORKING -> SHORT_TERM -> LONG_TERM
- **Memory store protocols**: `EpisodicStore`, `CausalStore`, `SemanticStore` in `memory/store.py` — split persistence protocols with `File*Store` defaults and database implementations for Mother Maxim.
- **Lane tier system**: Functions route to capability tiers (large/medium/small) via `FunctionRouter` in `runtime/function_router.py`. Legacy lane names (infer/review/record) are aliased to tier names. `detect_tiers()` in `lane_models.py` auto-detects from hardware.
- **Data paths**: Bundled seed data in `src/maxim/_data/` (components, encounters, prompts, templates). User data at `~/.maxim/` (memory, sessions, benchmarks, config). Resolution via `utils/paths.py`.
- **SEM Component Registry**: `embodiment/component_registry.py` discovers SEM entity templates from campaign-local, `~/.maxim/components/`, and `_data/components/`. 9 seed components across 5 categories (bodies, creatures, environments, npcs, weapons).
- **Thread model**: Main loop at 2-30Hz + WorkerPool (tier-based lanes: large/medium/small, owned by LLMWorker) + Hippocampus capture thread (owned + shut down by MemoryHub.on_session_end)

## Quick reference — where to look

| I'm touching... | Start here |
|---|---|
| The agent loop / step pipeline | `runtime/agent_loop.py`, `runtime/loop_controller.py` |
| Adding a new tool | `tools/` + register in the tool registry; see `tools/narrative.py` for sim tools or `tools/introspection.py` for subsystem tools |
| AUT introspection (programmatic) | `simulation/introspection.py` (AUTIntrospector — clean API, no tool dispatch) |
| Tool aliases (hallucination redirect) | `runtime/executor.py` (TOOL_ALIASES dict + tool_usage_stats()) |
| LLM prompts / routing | `models/language/router.py`, `models/language/prompt_formats.py` |
| Sim personas | `simulation/personas.py` (8 today: adversarial, cooperative, confused, escalating, campaign, refinement, researcher, sweep) |
| Research protocol (Writer/Reviewer) | `simulation/research_agents.py`, `simulation/research_orchestrator.py` |
| Memory capture → consolidation | `memory/hippocampus.py`, `memory/concept_extractor.py`, `memory/semantic_promoter.py` |
| Causal learning | `decisions/nac.py` |
| Cross-layer wiring | `integration/memory_hub.py` (the single coordinator) |
| Adding an env var | Put it here in the env table + touch whatever reads it |
| Atomic JSON persistence | `utils/atomic_io.py` |
| Mesh identity + protocol | `mesh/identity.py` (AgentProfile), `mesh/agent_identity.py` (AgentIdentity), `mesh/message.py` (MeshMessage, 24 types) |
| Mesh transport + admission | `mesh/peer_channel.py` (PeerChannel), `mesh/peer_registry.py` (PeerRegistry), `mesh/admission.py` (MeshAdmissionControl) |
| Mesh knowledge sharing | `mesh/knowledge.py` (ExperienceBroker, KnowledgeProvider/Receiver protocol, CausalLink + Reflection + MotorProgram adapters) |
| Mesh task delegation | `mesh/task_delegation.py` (TaskDelegator, TaskReceiver) |
| Mesh distributed planning | `planning/adaptive_planner.py` (`set_mesh_context()`, `_tag_delegatable_subgoals()`) |
| Mesh clock synchronization | `mesh/clock.py` (PeerClockEstimator), `time/scn.py` (`register_external()`) |
| Research experiment tracking | `simulation/research_tools.py` (ExperimentLog, record/query tools) |
| Function → tier routing | `runtime/function_router.py` (FunctionRouter, FunctionSpec, DEFAULT_FUNCTIONS) |
| Tier auto-detection | `runtime/lane_models.py` (detect_tiers, _INFER_VRAM_TIERS) |
| LLM hot-swap + persistence | `runtime/lane_backends.py` (swap_llm_server, _active_spawner) |
| Cloud provider profiles | `models/language/config.py` (_BUILTIN_PROFILES, cloud: True marker, 10 cloud profiles: Gemini, Groq, Together, Fireworks, Mistral, DeepSeek) |
| JSON repair pipeline | `models/language/json_parser.py` (4-stage + compliance counters via `json_parse_stats()`) |
| Scenario expectations (validation) | `simulation/validation.py` (15 types: behavioral, metric, bio-system) |
| Scenario YAML loading + metadata | `simulation/scenario_source.py` (ScenarioDefinition with tags, benchmark, suite sections) |
| Standalone experiment runner | `simulation/experiment.py` (`run_campaign()` → `ExperimentResult`) |
| Benchmark runner | `simulation/benchmark.py` (`BenchmarkRunner` — multi-model comparison, tiered metrics) |
| Experiment run notes | `docs/experiments/` (per-run findings + methodology) |
| Embodiment SEM protocol | `embodiment/sem.py` (Entity, Sensor, Modulator, FailureMode) |
| Embodiment YAML loading | `embodiment/spec.py` (load_spec, SpecSensor, SpecModulator, attach_backends) |
| Embodiment auto-tool generation | `embodiment/tool_bridge.py` (generate_tools_for_entity, collision detection) |
| Embodiment runtime | `embodiment/body.py` (Embodiment, failure eval, vital drift, prompt state) |
| Embodiment LLM/narrative backends | `embodiment/llm_backend.py` (LLMSensor, LLMModulator, NarrativeSensor, NarrativeModulator) |
| Embodiment YAML reference | `docs/embodiment_yaml_reference.md` |
| Cerebellum forward models | `embodiment/cerebellum.py` (Cerebellum, ForwardModel, ModelKey, bucket_params, ProgramRegistry) |
| Motor programs + registry | `embodiment/motor.py` (MotorProgram, MotorStep, ProgramRegistry, entity_state_similarity) |
| Motor engrams | `embodiment/engrams.py` (MotorEngram, formation thresholds, graph node naming) |
| Program executor | `embodiment/program_executor.py` (step-by-step runner, pain gates, PainBus interrupt) |
| CerebellumModulator | `embodiment/backends/cerebellum_modulator.py` (predict/fallback/train + factory) |
| Generative campaign arcs | `simulation/arcs.py` (NarrativeArc, NarrativePhase, BUILTIN_ARCS, load_arc_yaml) |
| Generative campaign narrator | `simulation/narrator.py` (two-call + single-call, system prompts, story compression) |
| Generative campaign runner | `simulation/generative_runner.py` (run_generative_campaign, YAML export, SEM entity loading) |
| Plan-to-arc bridge | `simulation/plan_arc_bridge.py` (translate_plan_to_arc, enrich_narrator_context, bridge_and_compress) |
| ask_user tool (interactive) | `simulation/tools_user.py` (AskUserTool, JSONL audit, replay, timeout escalation) |
| Generative campaign guide | `docs/generative_campaigns_guide.md` |
| DM campaign schema | `simulation/dm_schema.py` (campaign YAML schema, encounter/character/entity definitions) |
| DM campaign runtime | `simulation/dm_runtime.py` (encounter executor, entity transfer/visibility) |
| DM encounter tools | `simulation/tools_dm.py` (ChooseTool + alias system for encounter choices) |
| Multi-agent factory | `runtime/agent_factory.py` (AgentFactory — creates NPC agents with isolated subsystems) |
| Multi-agent pool | `runtime/agent_pool.py` (AgentPool — concurrent agent execution, LocalMessageBus) |
| Party DM runtime | `simulation/dm_party.py` (PartyDMRuntime — multi-agent campaign execution with NPC memory) |
| SEM component registry | `embodiment/component_registry.py` (ComponentRegistry — template catalog, multi-path discovery) |
| Encounter library | `simulation/encounter_library.py` (EncounterLibrary — reusable encounter templates, tag queries) |
| Entity designer (LLM) | `simulation/entity_designer.py` (EntityDesigner — natural language → SEM spec generation) |
| Memory store protocols | `memory/store.py` (EpisodicStore, CausalStore, SemanticStore protocols + File* defaults) |
| Interactive prompt protocol | `interactive/prompts.py` (PromptRequest, PromptHandler ABC, prompt types) |
| Rich terminal display | `interactive/display.py` (split-panel UI, DisplayExtension, graceful degradation) |
| DM display panels | `interactive/dm_display.py` (encounter info, character sheet panels) |
| Bundled seed data | `_data/components/` (bodies, creatures, environments, npcs, weapons), `_data/encounters/` (combat, exploration, puzzle, social) |
| Data paths + user home | `utils/paths.py` (bundled `_data/` vs user `~/.maxim/` resolution) |

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
# Legacy names (infer/review/record) are aliased to tier names automatically
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

## Research Protocol — Campaign Execution

When `--campaign <yaml>` is passed, campaign turns are **injected directly through the bridge** — the orchestrator LLM never touches the narrative text. This avoids JSON escaping issues with dialogue-heavy content.

Flow:
1. Campaign YAML loaded → turns extracted with salience/novelty
2. Each turn sent via `bridge.send_and_wait()` with progress output
3. AUT processes each turn (LLM inference → tool execution → hippocampus capture)
4. After all turns complete, orchestrator LLM starts with analysis-only goal
5. Orchestrator runs `inspect_aut`, `record_experiment`, `finish_simulation`

Without `--campaign`, the orchestrator LLM drives the full simulation (probes, observations, analysis).

**JSON robustness**: LLM JSON output goes through a 4-stage repair pipeline (`json_parser.py`): direct parse → control-char sanitize → `json_repair` library → structural repair. System prompts include explicit quote-escaping guidance (`_JSON_RULES` in `router.py`).

**Experiment notes**: Run findings go in `docs/experiments/`. Current experiments:
- `hippocampal_recall_experiment.md` — experiment design (seed/interference/recall)
- `hippocampal_recall_run_notes.md` — per-run observations and findings

## Python API (pymaxim)

The package is published to PyPI as `pymaxim` (import name stays `maxim`). Users interact through 13 verb-based functions, all lazy-loaded from `src/maxim/api.py`:

```python
import maxim

# Core 6 (original)
maxim.configure(verbosity=2)                                    # logging/debug setup
maxim.run(model="mistral-7b")                                   # agentic cycle
maxim.imagine(goal="test safety", persona="adversarial")        # simulation
maxim.connect("reachy_mini")                                    # robot connection
report = maxim.diagnose()                                       # doctor checks
state = maxim.observe("memory")                                 # bio-subsystem introspection
maxim.introspect("causal")                                      # alias for observe

# Expanded verbs (buildout)
maxim.campaign("scenarios/campaigns/heist_v1.yaml")             # DM campaign runner
maxim.benchmark(models=["mistral-7b", "qwen2.5-14b"])          # multi-model comparison
maxim.research(goal="hippocampal recall", campaign="...")        # research protocol
maxim.on("memory_capture", callback)                            # event subscription
maxim.register_tool(my_tool)                                    # runtime tool registration
maxim.register_persona(name="scout", system_prompt="...")       # custom persona registration

@maxim.tool                                                     # decorator for tool functions
def my_tool(query: str) -> str: ...
```

### Key files

| File | Purpose |
|------|---------|
| `src/maxim/api.py` | All 13 verb implementations (thin facades over existing internals) |
| `src/maxim/__init__.py` | Lazy `__getattr__` wiring — keeps `import maxim` fast |
| `src/maxim/simulation/introspection.py` | `Observer` class (renamed from `AUTIntrospector`) — powers `observe()` |

### Rules for maintaining the API

- **Verbs are facades, not logic.** Each function in `api.py` bootstraps objects and delegates to existing internals (`run_agentic_loop`, `start_simulation_mode`, `RobotRegistry`, `run_all_checks`, `Observer`, `BenchmarkRunner`, `PartyDMRuntime`, etc.). Don't put business logic in `api.py`.
- **Lazy imports only.** All heavy imports happen inside function bodies. `import maxim` must not trigger loading of optional dependencies.
- **Return structured data, not prints.** `diagnose()` returns `DiagnosticReport`, `imagine()` returns `SimulationResult`, `observe()` returns dicts. Don't print to stdout from API functions.
- **`introspect` is an alias for `observe`.** Both work. Don't add behavior to one without the other.
- **`Observer`** is the canonical name for what was `AUTIntrospector`. The deprecated alias `AUTIntrospector = Observer` exists in `introspection.py` for backward compat — remove it in 0.2.0.

### Package management

- **Package name:** `pymaxim` on PyPI, `maxim` as import
- **Version:** Defined in both `pyproject.toml` and `src/maxim/__init__.py` — keep in sync
- **Core deps:** `numpy`, `scipy`, `pyyaml`, `json-repair` only. Everything else is optional extras.
- **Optional extras:** `llm-local`, `llm-anthropic`, `llm-openai`, `vision`, `audio`, `reachy`, `comms`, `search`, `temporal`, `training`, `tts`, `yolo`, `semantic`
- **Robot plugins:** Auto-discovered via `maxim.robots` entry-point group. Third-party packages register controllers by declaring entry points.
- **Build validation:** `python -m build && twine check dist/*` before any publish
- Plans: [pypi_publication_plan.md](docs/plans/pypi_publication_plan.md)

## Active initiatives

See `docs/plans/future_plans.md` for the full roadmap.

**Current:**
- **Foundational Buildout Phase 11** — Test PyPI validation (dry-run). Phases 0-10 done. Publication delayed pending 12a (security hardening) + 12b (pre-pub hardening). [Plan](docs/plans/foundational_buildout_plan.md).
- **Phase 12a: Security Hardening** — shell injection, path traversal, auth bypass, CORS, error sanitization. ~200 LOC. [Plan](docs/plans/foundational_buildout_plan.md).
- **Phase 12b: Pre-Publication Hardening** — broken APIs, error honesty, CLI UX, test gaps, docs. ~2,500 LOC. [Plan](docs/plans/pre_publication_hardening_plan.md).
- **Mother Maxim** — Post-publication. Persistent shared cognitive instance with bio-system-aware deidentification, MCP server, federation protocol. [Plan](docs/plans/mother_maxim_plan.md).

**Recently completed (buildout Phases 0-10):**
- Package Hygiene, SEM Component Registry, Encounter Library, Agent Factory + Pool, Party DM Runtime, Hippocampus Recall Refinement, Interactive Runtime + Rich Display, Generative Architect + Entity Designer, API Surface Expansion, Deps + Docs + Cloud Profiles, Publication Prep.

**Previously completed (all archived):**
- Multi-LLM Scaling, Agent Mesh (Pre-7), Embodiment Core, Generative Campaigns, Bio-System Wiring Hardening, Mode Refactor, DM MVP, Research Protocol, Docker Sandbox, Python API, Tool Refactoring, Lane Tier Architecture, Simulation Benchmark (0-6), Realtime Refinement. See `docs/archive/` for plans.
