# Future Plans

Master roadmap for Maxim development. Individual plan files remain as detailed design references.

**Last updated:** 2026-04-04

---

## Status Overview

| Plan | Status | Next step |
|------|--------|-----------|
| Docker Sandbox | **Phase A done** | TmpdirSandbox + pain triggers implemented; Phase B (Docker backend) optional |
| Hippocampal Recall Experiment | **Not started** | Merges Research Protocol + DM into a single deliverable: D&D campaign as Hippocampus experiment. See [hippocampal_recall_experiment.md](hippocampal_recall_experiment.md) |
| Research Protocol | **Not started** | Phase 0 mesh primitives (AgentProfile, UMR, MeshMessage, LocalMessageBus ~200 LOC). First experiment: hippocampal recall. |
| Multi-LLM Scaling | **Complete** | All phases done. mDNS + InferenceRouter moved to Agent Mesh as Phases 0a-0b. [Archived](multi_llm_scaling_ARCHIVED.md) |
| Agent Mesh | **Not started** | Multi-LLM infra complete. Phases 0a-0b (mDNS + InferenceRouter) are first, then identity + protocol. Research Protocol Phase 0 unblocks full mesh. |
| Realtime Refinement | **Core done** | InspectAUTTool, 8 personas, 3 metric expectations, baseline scenario. Per-lane LLM metrics deferred to Multi-LLM Phase 8 |
| Embodiment Core | **Not started** | Phase 0 MVP + ATL grounding (~400 LOC) is the gate; Cerebellum + structured failures follow. Designed and scoped. |
| Embodiment Hardware Adapter | **Not started** | Blocked on Embodiment Core MVP. 1-sprint adapter (~300 LOC) wrapping RobotController. |
| Wave A Stabilization | **Done** | Circular import + bounded queues + atomic-write hardening + silent-except cleanup |
| Wave B Refinement Harness | **Done** | YAML `params` loader + `response_latency_ms` expectation + refinement_baseline.yaml + 9 new tests |
| Dungeon Master Persona (MVP) | **Deferred** | Hand-authored D&D campaigns as ultimate bio-system stress test (~840 LOC). Held until Multi-LLM + Agent Mesh + Embodiment Core land. `CharacterState` mirrors Embodiment body-state patterns; narrative damage flows through shared `PainDetector` pathway. Gated on choice-classifier spike. |
| DM Choice Classifier Spike | **Not started** | Half-day spike validating ATL concept similarity + NAc causal scoring can classify AUT free-text responses against campaign choices. Runs before DM MVP commits. |
| Dungeon Master Extensions | **Deferred** | Optional follow-ons layered onto DM MVP: architect persona, encounter library, adaptive difficulty, true RNG, etc. Each extension gated on MVP usage pain. |
| Interactive Sim Prompts | **Not started** | `ask_user` tool with timeout + replay (~180 LOC). Needed for DM architect extension; useful to any authoring persona. |
| Sim Entity Naming | **Not started** | Per-entity name prefix in sim logs (AUT/orchestrator only, ~120 LOC). Optional readability win. |
| Stdlib OpenAI-Compat Client | **Not started** | Replace `openai` pip dep with ~40 LOC urllib fallback for peer→leader inference. Zero extra deps on peer machines. |
| Simulation Test Bed | **Not started** | Automated sim regression suite: run scenario battery, assess against expectations, report pass/fail with bio-system health metrics. |
| Remote Update Soft Restart | **Not started** | Auto-restart maxim process after `maxim peer update` via `os.execv`. Currently requires manual restart on leader after code pull. ~30 LOC in leader_proxy.py. |

### Completed Plans

| Plan | What it delivered |
|------|-------------------|
| Simulation Decomposition | spawn_sub_simulation, extend_simulation, --continuous, 8 personas, approach param, stall detector, SimToolRegistry, bio system wiring |
| Repo Cleanup (~90%) | Dead code removed, CI added, deps slimmed, version pins relaxed |
| Agentic Loop Modularization | LoopController, SimulationAdapter, DefaultNetworkController, @resilient, typed state |
| Simulation Agent (Phases 1-3) | SimulationBridge, 10 tools, 8 personas, orchestrator lifecycle, CLI wiring |
| Intelligent Context Upgrade (~90%) | Edit disambiguation, turn pinning v1, dropped context notice |
| LLMWorker Cleanup (Track B) | Removed legacy dual-mode, pass-through statics, fixed feature detection |
| Router Modularization | router.py split into config.py, types.py, token_counter.py, prompt_formats.py, json_parser.py (router down to 1,268 LOC) |
| Wave A Stabilization | NAc circular import fix, bounded `_consolidation_candidates` + `_pending_events`, `atomic_io` util with fsync, silent-except audit in agent_loop, defensive shutdown for concept subsystems |
| Wave B Refinement Harness | YAML `params` loader, `response_latency_ms` expectation (p50/p95 inter-action gaps), `scenarios/refinement_baseline.yaml`, 9 expectation tests |

---

## Dependency Graph

```
                    ┌─────────────────────────────────┐
                    │    Research Protocol Phase 0    │ (~200 LOC, unblocks half of Agent Mesh)
                    └──────────────┬──────────────────┘
                                   ↓
     ┌─────────────────────┐    ┌──┴──────────────────┐
     │  Multi-LLM P1-3     │    │  Research Protocol  │
     │  (local dual-model) │    │  Phases 1-3         │
     └──────────┬──────────┘    └─────────────────────┘
                ↓
     ┌──────────┴──────────┐
     │  Multi-LLM P4-6     │
     │  (remote/tunnel)    │
     └──────────┬──────────┘
                ↓
     ┌──────────┴──────────┐      ┌──────────────────────┐
     │  Multi-LLM P7       │──┬──►│  Agent Mesh P1+      │
     │  (PeerRegistry)     │  │   │                      │
     └──────────┬──────────┘  │   └──────────────────────┘
                ↓             │
     ┌──────────┴──────────┐  │   ┌──────────────────────┐
     │  Multi-LLM P8       │  │   │  Embodiment Core     │ (parallel track, independent)
     │  (per-lane metrics) │  │   │  Phase 0 MVP         │
     └──────────┬──────────┘  │   └──────────┬───────────┘
                ↓             │              ↓
     [Refinement closure]     │   ┌──────────┴───────────┐
                              │   │  Embodiment Core     │
                              │   │  (further phases)    │
                              │   └──────────┬───────────┘
                              │              ↓
                              │   ┌──────────┴───────────┐
                              │   │  Hardware Adapter    │
                              │   └──────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  DM prerequisites   │
                    │  all satisfied      │
                    └──────────┬──────────┘
                               ↓
              ┌────────────────┴──────────────────┐
              │  DM Choice Classifier Spike       │
              └────────────────┬──────────────────┘
                               ↓
              ┌────────────────┴──────────────────┐
              │  DM MVP → DM Extensions (demand)  │
              └───────────────────────────────────┘

Optional / independent (ship when demand surfaces):
  Interactive Sim Prompts, Sim Entity Naming, test_record_plan_outcome fix
```

---

## Implementation Sequence (solo-work ordering)

Reassess after each phase — this is a recommended order, not a rigid commitment.

| # | Work | LOC | Rationale |
|---|------|-----|-----------|
| 1 | **Embodiment Core Phase 0 MVP** | ~400 | No upstream deps, standalone-valuable, establishes body-state primitives that DM/Mesh inherit |
| 2 | **Research Protocol Phase 0** | ~200 | Tiny, unblocks Agent Mesh; shared mesh primitives |
| 3 | **Multi-LLM Phases 1-3** | ~500 | Local dual-model routing; bottleneck for compute scaling |
| 4 | **Embodiment Core remaining phases** | per plan | Cerebellum forward models, structured failures |
| 5 | **Multi-LLM Phases 4-6** | per plan | Remote LLM, tunnel, auto-spawn |
| 6 | **Research Protocol Phases 1-3** | per plan | Researcher/Writer/Reviewer agents |
| 7 | **Multi-LLM Phase 7 + Agent Mesh Phase 1+** | per plans | Mesh lands (consumes RP + Multi-LLM P7) |
| 8 | **Multi-LLM Phase 8** | per plan | Per-lane metrics, closes Refinement |
| 9 | **Embodiment Hardware Adapter** | ~300 | Wraps RobotController for hardware |
| 10 | **Interactive Sim Prompts** | ~180 | Ship when DM architect or other consumer surfaces |
| 11 | **Sim Entity Naming** | ~120 | Ship when multi-entity log output becomes painful |
| 12 | **DM Choice Classifier Spike** | ~150 scratch | Validates ATL+NAc classification path |
| 13 | **DM MVP** | ~840 | Capstone bio-system stress test |
| 14 | **DM Extensions** | per-extension | Demand-driven, never speculative |

**Why this order:**
- Finishes architectural foundations before layering features
- Embodiment Core before DM so DM's `CharacterState` inherits established body-state patterns
- Research Protocol Phase 0 early because it's tiny and unblocks mesh
- Multi-LLM drives the critical path for compute scaling
- DM comes last as the capstone that validates everything below it

**Parallelism opportunities (if capacity allows):**
- Embodiment track (1, 4, 9) is fully independent from scaling/coordination tracks
- Research Protocol (2, 6) can run in parallel to Multi-LLM (3, 5, 7, 8)
- Optional plans (10, 11) ship opportunistically whenever pain surfaces

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
> **Design:** [multi_llm_scaling_ARCHIVED.md](multi_llm_scaling_ARCHIVED.md)

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

> **Status:** Not started. First local mesh use case.
> **Effort:** ~1,300 LOC across 5 phases
> **Design:** [research_protocol_plan.md](research_protocol_plan.md)

Three specialized agents collaborating on a research question:
- **Researcher** — runs experiments via simulation tools, records structured results
- **Writer** — produces a structured paper (Methods → Results → Intro → Discussion → Conclusions)
- **Peer Reviewer** — validates claims by re-running experiments, flags issues, demands revisions

Builds the agent mesh primitives locally first (AgentProfile, UMR naming, MeshMessage, LocalMessageBus) as Phase 0, proving them before adding network code. Includes a validation suite with known-flawed scenarios to test reviewer effectiveness.

CLI: `maxim --sim research --goal "does the agent block code execution?"`

---

## 4. Agent Mesh

> **Status:** Not started. Phases 1a-1b built as part of Research Protocol.
> **Effort:** ~4,500 LOC across 10 phases
> **Design:** [agent_mesh.md](agent_mesh.md)

Cooperative peer-to-peer network of sovereign Maxim instances. Phases 1a-1b (AgentProfile + UMR) are built by the Research Protocol. Remaining phases add network discovery (mDNS), PeerChannel, knowledge sharing with transfer discount, and distributed planning.

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
| 29 | Standardize serialization | Before Agent Mesh Phase 8 |
| 41 | Movement step-clamping helper | Next time touching movement.py |
| 44 | Merge DNActionProposal | Next time touching default_network |

### Security hardening (post-Stage A)

Items surfaced while debugging peer-leader tunneling. Each is small and bounded; belongs in a later `bug/` or `feature/` branch once the current debug cycle settles.

| Item | Where | Fix |
|------|-------|-----|
| Bearer tokens are logged in plaintext by `cloudflared` at `loglevel: debug` (found in journalctl after tunnel debugging) | `/etc/cloudflared/config.yml` | Document "switch loglevel back to `info` after debugging" in [llm-setup.md](../user/llm-setup.md); optionally have `maxim tunnel status` warn when loglevel is verbose |
| `MAXIM_TUNNEL_ECHO=1` streams uvicorn access logs which include `x-request-id` but also any full URL/query strings | `runtime/local_server_spawner.py` | Already warns at startup; document that echo mode is debug-only, never leave on in production |
| `maxim tunnel key show` prints the full API key to stdout (deliberate) — can end up in shell history + terminal scrollback | `tunnel/cli.py` `_cmd_key_show` | Optional: add `--copy` flag that pipes to `pbcopy`/`xclip`/`clip.exe` without printing; default still prints for scriptability |
| Per-device keys still a parked discussion; shared-key model limits revocation granularity | Phase 7b/7c mesh work | Covered in [multi_llm_scaling_ARCHIVED.md](multi_llm_scaling_ARCHIVED.md) Phase 7 security notes |
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
| 3 | Research Protocol Phase 0 | AgentProfile + UMR + MeshMessage + LocalMessageBus |
| 4 | Research Protocol Phases 1-3 | Researcher + Writer + Peer Reviewer agents |
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
