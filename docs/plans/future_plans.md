# Future Plans

Master roadmap for Maxim development. Individual plan files remain as detailed design references.

**Last updated:** 2026-04-07

---

## Status Overview

| Plan | Status | Next step |
|------|--------|-----------|
| Tool Refactoring | **Complete** | All 10 phases done: say, think, examine, introspection, aliases, tracking, proactive list. [Plan](tool_refactoring_plan.md) |
| Introspection API | **Phases 1-4 done** | `Observer` (renamed from `AUTIntrospector`) + standalone `run_campaign()` shipped. Remaining: Ph5 self-introspection (needs discussion). |
| Lane Tier Architecture | **Complete (archived)** | Size-based model routing (large/medium/small). FunctionRouter, detect_tiers, doctor check, config loader, LaneMetrics aliases. `infer_net` absorbed. [Plan](../archive/lane_tier_plan.md) |
| Simulation Benchmark | **Complete (Phases 0-6)** | BenchmarkRunner, `--sim benchmark` CLI, bio-system expectations, scenario suite, baseline comparison, live progress. Phases 7-9 deferred (paper gen, narrative transcriber, embodiment hooks). Pending: promote to `maxim --benchmark` top-level flag with tiered benchmarks (Tier 1 cognitive, Tier 2 bio-system, Tier 3 embodiment). Part of Generative Campaign CLI simplification. [Plan](../archive/benchmark_plan.md) |
| Docker Sandbox | **Complete** | Phase A (TmpdirSandbox + pain) + Phase B (DockerSandbox + ContainerRunner + CLI) both shipped |
| Research Protocol | **Complete** | All phases: mesh primitives, research tools, Writer + Reviewer agents, Research Orchestrator. CLI: `maxim --sim research`. |
| Multi-LLM Scaling | **Complete** | All phases done. mDNS + InferenceRouter moved to Agent Mesh as Phases 0a-0b. |
| Agent Mesh | **Phases Pre-7 COMPLETE** | All core mesh shipped: identity, protocol, transport, admission, knowledge sharing (provider/receiver protocol), task delegation, distributed planning, SCN temporal coordination. Remaining: Phase 0a (mDNS) + 0b (InferenceRouter) deferred until multiple LAN machines. |
| Realtime Refinement | **Core done** | InspectAUTTool, 8 personas, 3 metric expectations, baseline scenario. Per-lane LLM metrics deferred to Multi-LLM Phase 8 |
| Embodiment Core | **Complete** | All phases shipped: SEM protocol, PainBus, Cerebellum, motor programs, NarrativeModulator, auto-tool generation. 164 tests. Hardware adapter deferred. |
| Embodiment Hardware Adapter | **Not started** | Deferred to future when deploying to real hardware. |
| Generative Campaign Mode | **Complete** | All stages shipped: arc system, narrator, planner integration, `ask_user` tool, benchmark tiers, YAML export. 71 tests. [Plan](generative_campaign_plan.md) |
| Bio-System Wiring Hardening | **Complete** | All phases shipped: 7 disconnected bio-systems wired, pipeline correctness, percept abstraction (SensoryGate), cascade surfacing, ~1,100 LOC dead code removed, audit 14/14. [Plan](../archive/biosystem_wiring_hardening.md) |
| Mode System Refactor | **Complete** | Autonomy levels only, strategies/exploration/LiveModeIntent removed, sleep is a tool, ~1,800 LOC removed. [Plan](mode_refactor_plan.md) |
| Dungeon Master Persona (MVP) | **Ready to start** | Bundled SEM characters + cascade DAG + entity transfer/visibility + 4 showcase campaigns (~1,100 LOC). All prerequisites shipped. [Plan](dungeon_master_persona.md) |
| DM Choice Classifier Spike | **Not started** | ~0.5 day: ATL+NAc classification accuracy. Merged into DM plan. |
| Dungeon Master Extensions | **Deferred** | Optional follow-ons layered onto DM MVP. Each extension gated on MVP usage pain. Needs update for SEM character model. |
| Interactive Sim Prompts | **Complete (folded)** | `ask_user` tool shipped as part of Generative Campaign Mode. |
| Hippocampus AUT Memory Refinement | **Not started** | Improve how the AUT queries its own memories. Current `memory_recall` tool returns raw episodic records — needs semantic filtering, relevance ranking, and modality-aware recall (e.g., "what did I hear?" vs "what did I see?"). Also: reduce observation capture spam, improve recall precision for campaign-specific queries, and enable decision_rationale search. |
| Capability Agent | **Not started** | Continuous runtime awareness — live model availability, gate actions by hardware, proactive routing suggestions. ~500 LOC across 5 phases. Wraps detect_tiers + FunctionRouter + LaneMetrics + peer registry. [Design notes in doctor_upgrade_plan.md](doctor_upgrade_plan.md#capability-agent--continuous-runtime-awareness-300500-loc) |
| Peer Inference Retry on Leader Restart | **Not started** | Retry with backoff on 502/503 during leader restart. ~30 LOC in openai_backend.py. |
| Python API | **Complete** | Verb-based public interface (`run`, `imagine`, `connect`, `diagnose`, `observe`, `configure`). Observer rename done. `src/maxim/api.py` + lazy `__init__.py`. [Plan](python_api_plan.md) |
| PyPI Publication | **Phase 0-2 done** | Name (`pymaxim`), metadata, dep restructuring, Python API all done. Remaining: multi-robot plugins (Ph3), CI/CD (Ph4), README rewrite (Ph5), Test PyPI (Ph6). [Plan](pypi_publication_plan.md) |

### Completed Plans

| Plan | What it delivered |
|------|-------------------|
| Tool Refactoring | say, think, examine tools + introspection on AUT + "did you mean?" + alias map + tool usage tracking + proactive tool list |
| Research Protocol | AgentProfile, UMR, MeshMessage, LocalMessageBus, research tools, Writer + Reviewer agents, Research Orchestrator |
| Multi-LLM Scaling (Phases 1-8) | Remote backend, tunnel, auto-spawn, leader mode, peer discovery, inference routing, per-lane metrics, cloud providers |
| Remote Update + Soft Restart | `maxim peer update/restart/version/logs/llm` — all working through tunnel |
| Wave A Stabilization | NAc circular import fix, bounded queues, atomic_io, silent-except cleanup |
| Wave B Refinement Harness | YAML params, response_latency_ms, refinement_baseline.yaml, 9 tests |
| Simulation Decomposition | spawn/extend/continuous, 8 personas, stall detector, SimToolRegistry |
| Router Modularization | router.py split into config.py, types.py, token_counter.py, prompt_formats.py, json_parser.py |
| Agentic Loop Modularization | LoopController, SimulationAdapter, DefaultNetworkController, @resilient |
| Repo Cleanup (~90%) | Dead code removed, CI added, deps slimmed |
| Doctor Upgrade (v1) | Platform detection, GPU/server/LAN/tunnel checks, `maxim peer test` |
| Lane Tier Architecture | FunctionRouter (13 functions, fallback chains, priority boost), DEFAULT_TIERS, detect_tiers(), `infer_net` absorbed, config loader, MetricsRegistry aliases, doctor `check_tier_detection` |
| Simulation Benchmark (Phases 0-6) | BenchmarkRunner + `--sim benchmark` CLI, 8 bio-system expectation types, scenario metadata, JSON compliance counter, Observer.benchmark_snapshot(), unified YAML loader, 6 benchmark scenarios (cognitive_suite), tiered terminal output + JSON/MD persistence, baseline comparison with deltas, live progress display, DefaultNetwork wired in sim. Phases 7-9 deferred: paper gen (research Writer already exists), narrative transcriber (new feature), Tier 3 embodiment hooks (blocked on Embodiment Core). |
| Python API | Verb-based `maxim.run/imagine/connect/diagnose/observe/configure` in `src/maxim/api.py`. Lazy `__getattr__` in `__init__.py`. Observer rename (AUTIntrospector→Observer). `introspect` alias. DiagnosticReport dataclass. |
| PyPI Publication (Phases 0-2) | Package name `pymaxim`, authors/URLs/keywords metadata, core deps slimmed to 4 (numpy, scipy, pyyaml, json-repair), new optional extras (vision, audio, reachy, search), build validated with twine check. |

---

## Dependency Graph

```
     ┌─────────────────────────────────────────────────────────┐
     │  COMPLETED FOUNDATION                                    │
     │                                                          │
     │  Research Protocol ✅    Multi-LLM (all phases) ✅       │
     │  Lane Tier Arch ✅       Sim Benchmark (0-6) ✅          │
     │  Embodiment Core ✅      Generative Campaigns ✅         │
     │  Tool Refactoring ✅     Python API ✅                   │
     └──────────────────────────┬──────────────────────────────┘
                                │
          ┌─────────────────────┼──────────────────────┐
          ↓                     ↓                      ↓
  ┌───────────────────┐  ┌─────────────────┐  ┌──────────────────┐
  │  Agent Mesh P2+   │  │  DM Classifier  │  │  PyPI Pub Ph3-6  │
  │  (network transp) │  │  Spike (~2 days)│  │  (independent)   │
  └───────┬───────────┘  └────────┬────────┘  └──────────────────┘
          │                       ↓
          │              ┌────────┴────────┐
          │              │  DM MVP (~730)  │
          │              │  Bundled SEM +  │
          │              │  cascade DAG    │
          │              └────────┬────────┘
          │                       ↓
          │              ┌────────┴────────┐
          ├─────────────►│  DM Extensions  │
          │              │  (demand-driven)│
          │              └─────────────────┘
          ↓
  ┌───────────────────┐
  │  Multi-AUT party  │  (requires Mesh P2+ AND DM MVP)
  └───────────────────┘

Optional / independent (ship when demand surfaces):
  test_record_plan_outcome fix
  Stdlib OpenAI-Compat Client
  Capability Agent
  Hardware Adapter (when deploying to real robots)
```

---

## Implementation Sequence (solo-work ordering)

Reassess after each phase — this is a recommended order, not a rigid commitment.

| # | Work | LOC | Rationale |
|---|------|-----|-----------|
| ~~1~~ | ~~Lane Tier Architecture~~ | ~~820~~ | ✅ Complete |
| ~~2-4~~ | ~~Simulation Benchmark (Phases 0-6)~~ | ~~950~~ | ✅ Complete |
| ~~5~~ | ~~Embodiment Core (all phases)~~ | ~~per plan~~ | ✅ Complete (SEM, PainBus, Cerebellum, motor programs, 164 tests) |
| ~~6~~ | ~~Generative Campaign Mode~~ | ~~1,210~~ | ✅ Complete (arcs, narrator, ask_user, benchmark tiers, 71 tests) |
| ~~7~~ | ~~Bio-System Wiring Hardening~~ | ~~1,010~~ | ✅ Complete. All phases shipped: wiring, cascade surfacing, pipeline correctness, percept abstraction, audit script, consolidation, dead code cleanup. [Plan](../archive/biosystem_wiring_hardening.md) |
| ~~7b~~ | ~~Mode System Refactor~~ | ~~-1,800~~ | ✅ Complete. Autonomy levels only, strategies/exploration/LiveModeIntent removed, sleep is a tool. [Plan](mode_refactor_plan.md) |
| 8 | **DM MVP** | ~1,100 + YAMLs | Bundled SEM characters, cascade DAG, entity transfer/visibility, 4 showcase campaigns with pipeline health checks + ablation. [Plan](dungeon_master_persona.md) |
| ~~9~~ | ~~Agent Mesh Phases Pre-7~~ | ~~per plan~~ | ✅ Complete (identity, protocol, transport, admission, knowledge sharing, delegation, planning, SCN temporal). Phase 0a-0b (mDNS + InferenceRouter) deferred. |
| 10 | **DM Extensions** | per-extension | Demand-driven, never speculative. Needs update for SEM character model. |
| 11 | **Multi-AUT Party Mode** | per plan | Requires Agent Mesh P2+ AND DM MVP. Civilization-scale stress test. |

**Why this order:**
- DM spike first: validates that bio-systems actually influence AUT behavior (foundational assumption for DM as stress test)
- DM MVP next: exercises the full bio-stack with embodiment, generative campaigns, and SEM protocol working together
- Agent Mesh is independent — can be worked in parallel with DM
- Multi-AUT party mode is the ultimate capstone, combining mesh + DM

**Parallelism opportunities:**
- Agent Mesh Phase 2+ is fully independent from DM track
- PyPI Publication (Ph3-6) is independent from everything

---

## 1. Router Modularization — DONE

> **Status:** Complete. router.py reduced to 1,268 LOC.
> **Design:** [router_modularization_plan.md](router_modularization_plan.md)

`src/maxim/models/language/` now contains:
- `config.py` — LLMConfig, load_llm_config, profiles, quantization
- `types.py` — LLMResponse, RoutingPolicy, ProviderState
- `token_counter.py` — token counting helpers
- `prompt_formats.py` — format-specific prompt building
- `json_parser.py` — response JSON extraction
- `router.py` — LLMRouter class
- `cost_tracker.py`, backends (anthropic/llama/openai/transformers)

---

## 2. Multi-LLM Scaling

> **Status:** Phases 1–6 live. Phase 7 (peer mesh + multi-front input) next.
> **Effort:** ~2,000 LOC across 10 phases
> **Design:** the multi-LLM scaling work (now complete)

### Phases 1-3: Local Multi-Model (✅ done)

- Phase 1: `LaneConfig` gains `model_profile`, `device`, `n_gpu_layers`
- Phase 2: `LaneModelConfig` + capability-driven assignment based on hardware
- Phase 3: `LaneBackendManager` with safety gates (concurrent backend + cloud-lane caps)

**Target achieved:** Run mistral-7b (GPU) + SmolLM-1.7B (CPU) simultaneously on RTX 5080 via auto-spawn.

### Phases 4-6: Remote + Auto-Spawn + Leader Mode (✅ done)

- Phase 4-5: llama-cpp-server remote backend + Cloudflare tunnel docs
- Phase 6: `LocalBackendSpawner` + leader-mode detection (`~/.cloudflared/config.yml` or `MAXIM_ROLE=leader`)

### Doctor Upgrades (companion effort)

> **Design:** [doctor_upgrade_plan.md](doctor_upgrade_plan.md)

`maxim doctor` v1 ships with the multi-LLM work (platform detection, GPU/server/LAN/tunnel checks, platform-specific fix hints, retry loop, `maxim peer test`). Future expansions: deeper GPU health probes, inference coherence + tokens/sec benchmarks, sim-based behavior regression tests, JSON output for CI, fix automation, and agent-mesh health diagnostics.

### Phases 4-6: Remote + Auto-Spawn

- Phase 4: Home server with vLLM/llama-cpp exposing OpenAI-compatible API
- Phase 5: Cloudflare tunnel for zero-config WAN access
- Phase 6: `LocalBackendSpawner` — auto-detect hardware, spawn model servers

### Phase 7: Peer Mesh

- `PeerRegistry` — mDNS discovery, `InferenceRouter` — local → peer → remote fallback

### Phases 8-9: Metrics + Config

- Per-lane performance counters (feeds into Realtime Refinement)
- Environment variable / config file support

### Prerequisites

- Router Modularization (so Phase 3 adds `lane_manager.py` to a clean module structure)
- Cleanup #8 (wire PerceptSource protocol) and #9 (Any type overuse) — recommended before Phase 3

---

## 3. Research Protocol (Agent Mesh proving ground)

> **Status:** Complete. All phases shipped.
> **Effort:** ~1,300 LOC across 5 phases
> **Design:** [research_protocol_plan.md](research_protocol_plan.md)

Three specialized agents collaborating on a research question:
- **Researcher** — runs experiments via simulation tools, records structured results
- **Writer** — produces a structured paper (Methods → Results → Intro → Discussion → Conclusions)
- **Peer Reviewer** — validates claims by re-running experiments, flags issues, demands revisions

**Shipped:**
- `src/maxim/mesh/` — AgentProfile, UMR naming, MeshMessage, LocalMessageBus (~200 LOC)
- `src/maxim/simulation/research_tools.py` — ExperimentLog, RecordExperimentTool, QueryExperimentsTool (~150 LOC)
- `src/maxim/simulation/research_agents.py` — WriterAgent, ReviewerAgent, PaperDraft, ReviewResult (~300 LOC)
- `src/maxim/simulation/research_orchestrator.py` — start_research_mode, ResearchResult (~200 LOC)
- Dual-LLM wiring: `--aut-model` flag in orchestrator for separate AUT model
- 82 unit tests across mesh, research tools, and research agents

CLI: `maxim --sim research --goal "does the agent block code execution?" [--campaign <yaml>] [--aut-model mistral-7b]`

---

## 4. Agent Mesh

> **Status:** Phases Pre-7 COMPLETE (2026-04-07). All core mesh infrastructure shipped. Phase 0a (mDNS) + 0b (InferenceRouter) deferred.
> **Effort:** ~1,800 LOC shipped across Pre + Phases 1-7, plus ~600 LOC tests
> **Design:** [agent_mesh.md](../archive/agent_mesh.md)

Cooperative peer-to-peer network of sovereign Maxim instances. Phases 1a-1b (AgentProfile + UMR) and Phase 2-3 foundations (MeshMessage + LocalMessageBus) are implemented via Research Protocol. Remaining phases add network discovery (mDNS), PeerChannel, knowledge sharing with transfer discount, and distributed planning.

---

## 5. Realtime Refinement

> **Status:** Not started. Ongoing practice after sim agent + multi-LLM are live.
> **Design:** [realtime_refinement_plan.md](realtime_refinement_plan.md)

Observation-driven tuning across all subsystems:

- **Simulation agent tuning:** Persona prompt iteration, tool usage patterns, settle detection
- **Intelligent context refinement:** Edit disambiguation metrics, LLM-driven turn pinning (v2-v4)
- **Per-lane LLM metrics:** From Multi-LLM Phase 8
- **NAc causal learning observation:** Using existing introspection tools
- **Provenance & tracing:** Using existing ExplainTool and session logs

Not a build phase — a practice that starts once there's data to observe.

---

## 6. Tool Refinement (ongoing)

> **Status:** Living document — see [tool_refinement_plan.md](tool_refinement_plan.md).

Ongoing curation of the tool surface the agent can call: introspection tools (agent → its own state), action tools (agent → world), composite tools. Tracks what's shipped, what's proposed, what's deprecated, and the design principles (read-only by default, secrets opaque, limits self-enforce, size-capped outputs, context-gated registration).

**Currently proposed** (organized by subsystem in the plan doc):
- **Mesh introspection** (Phase 8 landed, buildable now) — `lane_status`, `inference_trace`, `compute_budget`, `peer_list`, `cluster_status`
- **System health** (buildable now, heartbeat infra landed) — `system_heartbeat` (GPU/CPU/RAM/disk/WiFi snapshot), `stall_check` (detect idle agent loop), `resource_pressure` (unified view of compute budget vs demand). These let the agent reason about its own resource state — e.g. "GPU is thermal-throttling, switch to CPU model" or "disk is 95% full, skip sim report archival".
- **Runtime introspection** (buildable today) — `loop_stats`, `recent_actions`, `mode_status`, `worker_pool_status`
- **Memory dynamics** (buildable today) — `memory_pressure`, `consolidation_status`, `bridge_activity`
- **Decision + learning** (buildable today) — `nac_stats`, `plan_history`, `confidence_calibration`
- **Pain + safety awareness** (buildable today) — `pain_triggers_active`, `fear_review_history`
- **Sim-mode introspection** (buildable today, sim-gated) — `sim_status`, `sim_action_history`, `sim_observe_self`
- **Provenance + explainability** (buildable today) — `session_overview`, `cycle_trace`

Full catalog, design principles, lifecycle policy, and deprecation log live in the plan doc.

---

## 7. Remaining Cleanup (opportunistic)

> **Status:** ~90% complete. Pick up when touching the file.

| # | Item | When |
|---|------|------|
| 6 | Fix batch scenario break | Before Simulation Agent Phase 4 |
| 8 | Wire PerceptSource protocol | Before Multi-LLM Phase 3 |
| 9 | Any type overuse (Protocols) | Before Multi-LLM Phase 3 |
| 13 | Stale re-exports in llm_worker | Next time touching llm_worker |
| 27 | Consolidate env bool parsing | Anytime (12 files) |
| ~~29~~ | ~~Standardize serialization~~ | ~~Done (Agent Mesh Pre-work)~~ |
| 41 | Movement step-clamping helper | Next time touching movement.py |
| 44 | Merge DNActionProposal | Next time touching default_network |

### Security hardening (post-Stage A)

Items surfaced while debugging peer-leader tunneling. Each is small and bounded; belongs in a later `bug/` or `feature/` branch once the current debug cycle settles.

| Item | Where | Fix |
|------|-------|-----|
| Bearer tokens are logged in plaintext by `cloudflared` at `loglevel: debug` (found in journalctl after tunnel debugging) | `/etc/cloudflared/config.yml` | Document "switch loglevel back to `info` after debugging" in [llm-setup.md](../user/llm-setup.md); optionally have `maxim tunnel status` warn when loglevel is verbose |
| `MAXIM_TUNNEL_ECHO=1` streams uvicorn access logs which include `x-request-id` but also any full URL/query strings | `runtime/local_server_spawner.py` | Already warns at startup; document that echo mode is debug-only, never leave on in production |
| `maxim tunnel key show` prints the full API key to stdout (deliberate) — can end up in shell history + terminal scrollback | `tunnel/cli.py` `_cmd_key_show` | Optional: add `--copy` flag that pipes to `pbcopy`/`xclip`/`clip.exe` without printing; default still prints for scriptability |
| Per-device keys still a parked discussion; shared-key model limits revocation granularity | Phase 7b/7c mesh work | Covered in the multi-LLM scaling work (now complete) Phase 7 security notes |
| `cloudflared` debug log rotation: journal holds Bearer tokens until rotation policy trims them | systemd/journald | Document `journalctl --vacuum-time=1d` as a cleanup step when downgrading loglevel |

**Stage A specific**: the Stage A trace flags (`MAXIM_LANE_TRACE`, `MAXIM_PEER_LOG_REQUESTS`, `MAXIM_TUNNEL_ECHO`) all produce a loud startup banner. That's intentional for debug visibility, but the flags' output contains request URLs + provider names. Not secrets, but a privacy consideration worth noting in docs when Stage A ships for wider use.

---

## Recommended Execution Order

### Wave 1: Simulation Depth + Multi-LLM

| Step | What | Why |
|------|------|-----|
| 1 | Simulation Decomposition | Merge PR: spawn + extend + continuous |
| 2 | Multi-LLM Phases 1-3 | Local dual-model for faster sim + stronger adversary |

### Wave 2: Research Protocol (local mesh)

| Step | What | Why |
|------|------|-----|
| 3 | ~~Research Protocol Phase 0-1~~ | ✅ Mesh primitives + research tools shipped |
| 4 | Research Protocol Phases 2-3 | Writer + Peer Reviewer agents |
| 5 | Research Protocol Phases 4-5 | Orchestration + validation suite |

### Wave 3: Infrastructure + Network Mesh

| Step | What | Why |
|------|------|-----|
| 6 | Multi-LLM Phases 4-7 | Remote server, tunnel, peer discovery |
| 7 | Agent Mesh Phases 2+ | Network primitives (local primitives proven by Wave 2) |
| 8 | Realtime Refinement | Tune everything with accumulated data |

### Wave 4: Embodiment

| Step | What | Why |
|------|------|-----|
| 9 | Embodiment Core Phase 0 (MVP gate) | ATL-grounded LLM percepts; validate σ reduction + NAc convergence |
| 10 | Embodiment Core Phases 1-2 | Cerebellum forward models + structured composable failures |
| 11 | Embodiment Hardware Adapter | HardwareBackend wrapping RobotController (1 sprint, ~300 LOC) |

---

## Stdlib OpenAI-Compatible Client

> **Status:** Not started. Ship when peer dependency weight becomes painful.
> **Effort:** ~40-60 LOC in `models/language/openai_backend.py`

The `openai` pip package (`openai` → `httpx` → `pydantic` → `anyio` → `sniffio` → `jiter` → `distro`) is the only non-stdlib dep required on peer machines for remote inference. Our usage is minimal:
- POST JSON to `/v1/chat/completions`, parse JSON response
- GET `/v1/models` (health check)
- Bearer token auth header

**Plan:**
1. Add a `_StdlibOpenAIClient` class in `openai_backend.py` using `urllib.request` (~40 LOC)
2. `_OpenAIBackend._get_client()` tries `from openai import OpenAI` first, falls back to `_StdlibOpenAIClient`
3. Stdlib client implements only `client.chat.completions.create()` — same interface, minimal surface
4. Streaming support deferred (not used in peer→leader path today)

**Testing requirements:**
- Both client paths must be exercised against a real llama-cpp-server (leader + peer)
- Verify error handling (401, 403, 502, timeout) matches `openai` package behavior
- Confirm no regressions in existing local + cloud provider paths
- Test on both leader (has `openai` installed) and peer (stdlib-only)

**Why not now:** The `openai` package works and is already installed on the leader. This becomes valuable when onboarding new peers that should stay lightweight, or when dep conflicts arise.

---

## Simulation Test Bed

> **Status:** Not started. Builds on existing scenario YAML + refinement harness infrastructure.
> **Effort:** ~400-600 LOC across 3 phases

Automated regression suite that runs a battery of simulation scenarios, assesses results against defined expectations, and produces a structured pass/fail report with bio-system health metrics. Catches regressions in agent behavior, tool safety, memory systems, and LLM response quality without manual observation.

### What exists today

- 3 YAML scenarios in `scenarios/` (malware_with_pain, long_horizon_coding, refinement_baseline)
- Refinement harness with metric expectations (action_count_range, tool_success_rate, response_latency_ms)
- Sim reports saved to `data/sim_reports/{session_id}/` with metrics + LLM analysis
- 8 personas (adversarial, cooperative, confused, escalating, campaign, refinement, researcher, sweep)

### What's missing

A runner that orchestrates multiple scenarios end-to-end and produces a single pass/fail verdict.

### Phase 1: Test bed runner (~200 LOC)

**CLI**: `maxim sim test` or `maxim sim test scenarios/tests/safety_basic.yaml`

- Runs all `.yaml` files in `scenarios/tests/` (or a specified subset)
- Each scenario gets a capped runtime (default 90s, configurable per-scenario)
- Collects sim reports and evaluates against YAML-defined expectations
- Produces a structured summary: scenario name, pass/fail, metrics, cost, issues found
- Exit code 0 if all pass, 1 if any fail (CI-friendly)
- `--json` flag for machine-readable output

**New expectation types:**
- `bio_system_active`: verify specific bio systems fired (Hippocampus capture, Pain detection, NAc observation)
- `memory_count_range`: min/max episodic memories formed
- `pain_triggered`: expect at least N pain events (for safety scenarios)
- `cost_range`: max acceptable cost per scenario run

### Phase 2: Scenario library in `scenarios/tests/` (~150 LOC in YAML)

Test-specific scenarios live in `scenarios/tests/`, separate from general-purpose scenarios in `scenarios/` (malware_with_pain, long_horizon_coding, etc.). This keeps the test bed self-contained and avoids polluting the user-facing scenario library with assertion-heavy test configs.

Curated scenarios covering key behavioral dimensions:

| Scenario | Tests | Persona | Expected |
|----------|-------|---------|----------|
| `safety_basic.yaml` | Refuses to delete system files | adversarial | pain_triggered >= 1, blocked actions > 0 |
| `tool_usage.yaml` | Uses correct tools for file ops | cooperative | tool_success_rate >= 0.8 |
| `memory_formation.yaml` | Forms and recalls memories | cooperative | memory_count_range [3, 20], bio_system_active: hippocampus |
| `cost_ceiling.yaml` | Stays within cost budget | cooperative | cost_range [0, 0.15] |
| `stall_recovery.yaml` | Recovers from tool failures | confused | action_count_range [5, 50] |
| `peer_inference.yaml` | Works over peer→leader tunnel | cooperative | tool_success_rate >= 0.9, latency checks |

Each scenario is a standalone YAML file in `scenarios/tests/` with goal, persona, expectations, and optional params.

### Phase 3: CI integration + trend tracking (~100 LOC)

- `maxim sim test --baseline` saves results as the reference baseline
- `maxim sim test --compare` diffs current run against baseline, flags regressions
- JSON output consumable by CI (GitHub Actions, etc.)
- Optional: publish results to `data/sim_test_history/` for trend analysis over time
- Integrate with `maxim doctor`: "last sim test bed run: 6/6 passed (2h ago)"

### Relationship to other plans

- **Realtime Refinement**: test bed validates that refinement tuning didn't regress other behaviors
- **Multi-LLM Phase 8 (metrics)**: per-lane metrics feed into `peer_inference.yaml` latency checks
- **Remote self-update (7a-ext)**: run `maxim sim test` automatically after `POST /v1/admin/update` to validate the update before confirming success
- **DM MVP**: DM campaigns become the ultimate stress-test scenarios in the library

### Design constraints

- **Scenarios must not require specific hardware** — use `--sandbox tmpdir` and `--language-model` flags to keep them portable across peer and leader machines
- **Cost-capped**: each scenario declares max acceptable cost; runner aborts if exceeded
- **Deterministic where possible**: use fixed seeds, specific goals, and bounded turn counts to reduce flakiness
- **No test-suite dependency**: `maxim sim test` is a CLI command, not a pytest fixture. It calls real LLMs and should never run in `python -m pytest` (per CLAUDE.md guidance)

---

## Research Paper Writer Refactor

> Formerly standalone plan `research_paper_writer_plan.md`. Folded here as a quality improvement for the existing Writer agent.

The Writer agent produces correct structure and data but has prose quality issues:
- Section generation sometimes produces JSON instead of prose (root cause: `_JSON_RULES` in router.py system prompts)
- Hallucinated references (smollm-1.7b invents fake papers — fix: only cite experiment UMRs)
- Duplicate headings (LLM generates `## Heading` when `to_markdown()` also adds one)
- Metrics partially incorrect (stat key mismatch — fix: use `AUTIntrospector` directly)
- Prose quality limited by small models (recommend `--cloud-lane review claude-haiku` for Writer)

**Components:**

| Component | LOC | Priority |
|-----------|-----|----------|
| Writer prose prompts (per-section templates) | ~150 | High |
| Paper template with markdown formatting | ~50 | High |
| Multi-experiment aggregation | ~100 | Medium |
| Reviewer prose quality checks | ~80 | Medium |
| LaTeX output option | ~100 | Low |
| Visualization generation | ~150 | Low |
| **Total** | **~630** | |

Ship incrementally: prose prompts + template first (~200 LOC), then aggregation + reviewer checks, then optional LaTeX/viz.

---

## Research Directions (Not Scheduled)

Tracked for future consideration. Not committed to any timeline.

- **ATL Self-Extension through Mechanism Discovery.** LLM proposes new concept categories or mechanisms, simulation exercises them, NAc learns whether they produce useful predictions, EC/Hippocampus recalls successful mechanisms. Genuinely novel, but requires separating signal from simulation noise. Deserves its own plan if pursued.
- **Federated Embodiments.** Multiple agents contribute components to one logical body (arm from A, cameras from B, voice from C). Naturally fault-tolerant distributed embodiment.
- **Cross-Agent Affordance Delegation.** Sovereign delegation of affordance invocations between mesh peers, with embodiment-gated FearAgent review.
- **NAc Causal Link Transfer.** Transfer learned causal links between agents, gated by embodiment-spec similarity.
- **Uncertainty-as-Pain.** High-variance Cerebellum models could fire pain from prediction uncertainty itself (biologically plausible — unfamiliar motion feels risky). Deferred because it risks suppressing exploration.
- **Curriculum Embodiment Learning.** Graduate an agent through progressively complex bodies; measure cross-embodiment transfer.
- **Bio-Multimodal Sensors.** Olfaction, taste, audition, vestibular, interoception beyond basic proprioception/vision/nociception.
- **Distributed Embodiment Construction.** LLM-driven composition tools that fan out across mesh peers for parallel spec generation.
