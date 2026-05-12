# Maxim Reference

Detailed reference material for Maxim's architecture, bio-system mappings, and configuration. For getting started, see [README.md](../README.md).

---

## Architecture

Maxim follows a strict layered architecture with one-way dependencies:

```
Agents -> Planning -> Decision Engine -> Runtime -> Executor -> Tools -> Environment -> State -> Memory
```

### Core Modules

| Module | Responsibility |
|--------|---------------|
| `src/maxim/agents/` | Goal reasoning, intent generation (no side effects) |
| `src/maxim/planning/` | Plan generation, refinement, and decision engine |
| `src/maxim/tools/` | Tool implementations (side effects) |
| `src/maxim/environment/` | World observation (no side effects) |
| `src/maxim/memory/` | Storage and retrieval |
| `src/maxim/runtime/` | Agent orchestration loop |
| `src/maxim/embodied_runtime/` | Robot capture/inference/control loop (Reachy hardware mixin stack) |
| `src/maxim/modes/` | Operating mode definitions |
| `src/maxim/proprioception/` | Movement tracking and pain detection |
| `src/maxim/harm/` | Predictive harm detection (velocity, joint limits) |
| `src/maxim/energy/` | Resource expenditure tracking (LLM tokens, cost, latency). `LLMEnergyTracker` + `EnergyRegistry` are live. Interoceptive drive signals (hunger, fatigue, stamina recovery) are handled by the drive protocol in `embodiment/sem.py` (`HomeostaticDriveSpec` / `EntropicDriveSpec`), not the energy module |
| `src/maxim/bridges/` | Cross-system integration (pain, energy, memory) |
| `src/maxim/embodiment/` | SEM protocol (Sensor-Entity-Modulator), Cerebellum, motor programs, ComponentIndex 3-layer semantic discovery (alias + head-noun fallback + embedding), ComponentRegistry ephemeral overlay, component-level damage model (per-modulator sensors, integrity aggregation, damage affinities, affordance `requires` gating), resolution LOD (`resolution.py`: level 2 collapsed / level 3 deep via `--deep-embodiment`) |
| `src/maxim/imagination/` | Real-time entity design from novel percept mentions: entity extraction, ComponentIndex lookup, LLM-driven design, session-scoped registration, DN arousal gating. Scene manifest pre-trigger (`process_manifest()`) pre-instantiates entities before first sim turn. Head-noun mention accumulation for threshold convergence |
| `src/maxim/mesh/` | Simulation-only: `bus`, `identity`, `message`, `naming` (R0 deleted the dead agent-mesh subsystem; see "Removed in R0" below) |
| `src/maxim/simulation/` | Simulation modes, generative campaigns, research protocol, benchmarks, **Roy iteration runner** (`roy_runner.py`) + **Roy curriculum runner** (`curriculum_runner.py`) for substrate priming |
| `src/maxim/analysis/` | Cross-session substrate analysis: `substrate_diff.py` (NAc / EC / Hippocampus / ATL pairwise divergence; **`cluster_reward_bias_l2`** Roy headline metric) + `roy_log.py` (idempotent protocol/iteration-log generation from `result.json`) |
| `src/maxim/roy/` | `maxim roy` CLI subcommand dispatch (`run` / `diff` / `log`) |
| `src/maxim/integration/` | MemoryHub cross-system coordinator (11 bio-systems) |
| `src/maxim/reactions/` | Reaction types, ReactionBus (per-kind dispatch), PerceptProducer/ReactionProducer protocols |
| `src/maxim/decisions/` | NAc causal learning, adaptive planner, reward distribution, **TemporalCreditDistributor** (temporal-phase-aware credit via NAc+SCN), **ValenceSignal** (abstract reward/punishment transport type for WMS salience modulation), goal-level reward bias (`_goal_reward_bias` `[-max,+max]` for ThoughtGate "go"/"no-go" modulation) |
| `src/maxim/time/` | SCN temporal rhythm indexing, **TemporalEvent** (frozen dataclass envelope for signal sources — tool, pain, reaction, percept, deliberation events) |
| `src/maxim/similarity/` | Entorhinal Cortex (pattern completion, centroid update) + LinguisticEncoder (P1, affordance name encoding) + ConceptDecomposer (noun-phrase extraction before EC) + AffordanceDecompositionStrategy (underscore-split for SEM affordance identifiers) |
| `src/maxim/prompts/` | PromptAssembler (B1), MemorySummary, prompt profiles, **Acting Coach (B3)** — meta-prompt scaffolding for affordance exploration with bio-system modulation (NAc valence, pain anticipation, cerebellum predictions) |
| `src/maxim/math/` | Angular Gyrus mathematical cognition, IPS fast stats |
| `src/maxim/default_network/` | Reactive behavior layer (thalamic gate, arbiter) |
| `src/maxim/salience/` | Novelty tracking, interest matching |
| `src/maxim/interactive/` | Interactive runtime: universal prompt protocol, rich terminal display, DM extensions |
| `src/maxim/cli_utils.py` | CLI helper utilities (extracted from main CLI module) |
| `src/maxim/runtime/tool_dispatch.py` | Tool execution dispatch (extracted from executor) |
| `src/maxim/runtime/bio_integration.py` | Bio-system wiring for the agent loop |
| `src/maxim/runtime/llm_server.py` | Local llama-cpp server lifecycle management |
| `src/maxim/runtime/rate_limit.py` | `KeyedRateLimiter` — per-key sliding-window rate limiter, cherry-picked from `mesh/admission.py` in R0. Consumed by Plan 4's admin API + per-agent router entry. |
| `src/maxim/utils/http.py` | **Plan 1 R1 (2026-04-12).** Unified HTTP client: endpoint registry, `HTTPEndpoint`/`TimeoutPolicy`/`RequestContext` dataclasses, typed `HTTPError` hierarchy, header sanitizer (`MAX_HEADER_VALUE_LEN=256`), `contextvars.ContextVar` propagation of X-Maxim-* headers, dual-format logging (`MAXIM_LOG_FILE`). Every outbound HTTP call in `src/maxim/` goes through this module; `urllib.request` is banned by CI grep. |
| `src/maxim/simulation/sim_types.py` | Shared type definitions for simulation subsystem |
| `src/maxim/simulation/campaign_runner.py` | Campaign execution engine (generative + DM + fixture) |
| `src/maxim/simulation/fixture_orchestrator.py` | FixtureDrivenOrchestrator — YAML fixtures without narrator LLM (S1) |
| `src/maxim/simulation/curriculum_runner.py` | Roy R1: chained curriculum runner — primes substrate across N stages, threading `resume_session` from stage to stage so the final session inherits everything. Returns a `CurriculumResult` with per-stage outcomes + the final session id for arm-A inheritance. |
| `src/maxim/simulation/roy_runner.py` | Roy R3: three-arm iteration orchestrator. Drives priming via `curriculum_runner`, runs the same held-out test scenario across arms (A = substrate-primed neutral, B = blank persona-injected, C = blank neutral), computes pairwise `substrate_diff`, persists `result.json` + `summary.md`. G3 LLM pre-flight probe aborts in <3s on unreachable leader (env or `peer.yml`). |
| `src/maxim/simulation/substrate_telemetry.py` | Phase 0 per-tick JSONL telemetry writer: EC node count, NAc reward_bias snapshot, drive states, active proposal. Fail-soft (never crashes the loop). |
| `src/maxim/analysis/substrate_diff.py` | Roy R2: substrate divergence library — reads `aut_nac.json`, `aut_hippocampus.json`, `ec.json`, `atl.json` from two session dirs and reports `NacDiff` (reward_bias / **cluster_reward_bias** / goal_reward_bias / causal-link delta), `EcDiff` (node count, modality histogram, top-N cross-side cosine pairs), `HippocampusDiff` (episode count + valence KS), `AtlDiff` (concept count + Jaccard). All sides surface `available=false` honestly when snapshots are missing. |
| `src/maxim/analysis/roy_log.py` | Roy R4: idempotent generator for protocol runbooks + iteration-log entries from a persisted `result.json`. HTML-comment markers preserve hand edits across regenerations (`--keep-edits`). |
| `src/maxim/roy/cli.py` | `maxim roy` subcommand dispatch — `run` / `diff` / `log`. Entry point in `cli.py::main` short-circuits to this before subcommand dispatch logging gap. |
| `src/maxim/models/language/backend_protocol.py` | LLMBackend Protocol — formal contract for all backends (S2) |
| `src/maxim/utils/seeding.py` | Global deterministic seeding + per-agent RNG streams (S4) |
| `src/maxim/models/language/cloud_dispatch.py` | Cloud provider request routing and redaction |
| `src/maxim/memory/store.py` | Split persistence protocols: EpisodicStore, CausalStore, SemanticStore |
| `src/maxim/memory/atl.py` | ATL semantic concept memory (concepts, relationships, grounding) |
| `src/maxim/memory/concept_extractor.py` | Concept extraction from episodic memories |
| `src/maxim/memory/concept_grounder.py` | Concept grounding against perceptual evidence |
| `src/maxim/memory/pattern_completer.py` | Pattern completion from partial cues |
| `src/maxim/memory/semantic_promoter.py` | Promotion of recurring patterns to semantic concepts |
| `src/maxim/memory/cross_layer.py` | CrossLayerGraph: associative edges between memory systems |
| `src/maxim/memory/consolidation.py` | ConsolidationOrchestrator: wave-based sleep consolidation |
| `src/maxim/memory/sleep_replay.py` | P8 sleep replay: episode ranking by NAc reward_bias + valence, re-fire Hebbian with consolidation multiplier (0.5) |
| `src/maxim/planning/structural_diff.py` | B4 Jaccard distance on action sequences for plan comparison. Pure utility, no agent/memory/runtime imports (0.5) |
| `src/maxim/planning/adaptive_planner.py` | ADaPT-style lazy planner with deep memory integration. B4 anti-repetition constraint in decomposition prompts (0.5) |
| `src/maxim/runtime/agent_factory.py` | F3-F5 `AgentFactory.create_full_agent()` — canonical agent construction composing `build_bio_stack` + `build_executor` + `FearGatedExecutor`. Used by CLI, sim orchestrator (AUT + orch), and headless API (0.7) |
| `src/maxim/runtime/bio_stack.py` | Wave 3 canonical bio-pipeline construction: `build_bio_stack` returns frozen `BioStack` (0.3) |

See [ARCHITECTURE.md](../ARCHITECTURE.md) for detailed design rules.

### HTTP client — `maxim/utils/http.py` (Plan 1 R1)

The single place outbound HTTP happens in `src/maxim/`. Adding new HTTP calls outside this module is blocked by a CI grep invariant:

```bash
grep -r "urllib.request.urlopen" src/maxim/ | grep -v "utils/http.py"
# Expected: empty
```

**Three public surfaces** — pick the one that fits your call site:

| Function | When to use | Endpoint |
|---|---|---|
| `http.get(name, path, **kw)` / `http.post(name, path, **kw)` | Calling a registered endpoint by name (leader admin calls after R1 lands them, Plan 3's `_MaximPeerBackend`) | Registry-based |
| `http.fetch_url(url, method="GET", **kw)` | Calling an arbitrary URL where the target isn't known at registration time (tool fetches, HF model downloads, peer CLI remote admin) | `_external` (lazily registered, `internal=False`) |
| `http.download_to_file(url, dest_path, progress_hook=..., **kw)` | Streaming a large file to disk with progress callbacks (GGUF downloads) | `_external` |
| `http.raw_proxy_forward(url, method, headers, body, timeout)` | **Reserved for `leader_proxy._proxy_request` only.** Bypasses sanitization + X-Maxim-* injection to forward client headers verbatim. Do not use elsewhere. | `_external` (raw) |

**Two registered endpoints** in R1:

- `"leader"` — registered by `peer/config.apply_peer_config_to_env`. `internal=True`, late-bound auth via `MAXIM_LANE_LARGE_REMOTE_API_KEY`, 3/60/120 timeout policy.
- `"_external"` — registered lazily by `_ensure_external_endpoint()`. `internal=False` (no X-Maxim-* leak to third parties), User-Agent set once, 5/120/600 long-timeout policy.

**Typed HTTP errors:** `HTTPTimeout`, `HTTPConnectionError`, `HTTPAuthError`, `HTTPServerError`, `HTTPClientError`, `HTTPRateLimited` — all inherit from `HTTPError` and carry `.status` + `.fix_hint`. Callers branch on these instead of string-matching exception messages.

**Env vars:**

- `MAXIM_HTTP_TRACE=1` — bumps `http_request` events from DEBUG to INFO
- `MAXIM_LOG_FILE=/path/to/x.jsonl` — attaches a JSONL file handler via `StructuredFormatter`

See [docs/architecture/llm_routing.md](architecture/llm_routing.md) (Layer 7) for the full design and [docs/troubleshooting/http_debugging.md](troubleshooting/http_debugging.md) for operational use.

### Removed in R0 (2026-04-11)

R0 of the LLM Path Refinement plan deleted ~1,250 LOC of dead mesh scaffolding from `src/maxim/mesh/`. These modules had zero production imports — they were swept along during the task→size lane refactor and never re-wired. Historical note: [docs/troubleshooting/mesh.md](troubleshooting/mesh.md).

| Deleted | What it was | Replacement |
|---|---|---|
| `mesh/peer_registry.py` | Speculative peer registry | Plan 4's `mesh.yml` + admin API (not yet shipped) |
| `mesh/peer_info.py` | Peer metadata (trust level, capabilities) | Plan 4 node entries in `mesh.yml` |
| `mesh/peer_channel.py` | Bi-directional peer message channel | None — direct HTTP via `utils/http.py` |
| `mesh/task_delegation.py` | Cross-peer goal delegation | None — out of scope for current cluster |
| `mesh/knowledge.py` | `ExperienceBroker` for shared CausalLinks | Deferred; see substrate plan cross-agent transfer |
| `mesh/clock.py` | `PeerClockEstimator` (NTP-free) | Operators use real NTP |
| `mesh/agent_identity.py` | Tool-list advertisement | Plan 4 admin API |
| `mesh/admission.py` | Rate limiter — **cherry-picked** to `runtime/rate_limit.py` | Lives in `runtime/rate_limit.py` now |

**Still present in `src/maxim/mesh/`:** `bus.py`, `identity.py`, `message.py`, `naming.py` — simulation-only consumers in `simulation/` and `create.py`.

---

## Bio-System Glossary

Maxim uses neuroscience-inspired names. Here is the translation:

| Bio Name | Plain English | Module | What It Does |
|----------|--------------|--------|--------------|
| Hippocampus | Episodic memory | `memory/` | Stores and recalls experiences (events, conversations) |
| ATL | Semantic memory | `memory/` | Extracts concepts, categories, and generalizations |
| NAc | Reward / causal learning | `decisions/` | Learns cause-and-effect relationships ("what leads to what"). `distribute_reward` now wired via ReactionBus subscriber in `build_bio_stack` |
| SCN | Internal clock | `time/` | Tracks circadian-like temporal patterns and rhythms. Kuramoto oscillator learns event-type co-occurrence for anticipatory temporal credit (B2) |
| EC | Memory indexing + substrate recognition | `similarity/` | Routes queries via similarity; pattern_complete_or_separate for substrate nodes (P1) |
| Angular Gyrus | Cross-modal algebra | `math/` | Combines memories across different modalities |
| Cerebellum | Motor prediction | `embodiment/` | Predicts outcomes of physical actions, learns motor programs. Now activated in production via `BioStack.cerebellum` and `build_executor(cerebellum=...)` |
| Amygdala / Fear | Threat detection | `proprioception/` | Detects harm, triggers pain signals, gates risky actions |
| Default Network | Reactive behavior | `default_network/` | Background processing, idle behaviors, spontaneous thoughts |
| Valence | Affective edge signal | `memory/episode.py` | Affective signal on Hebbian edges (`Edge.metadata["valence"]`), computed from Reactions at episode close via `apply_hebbian_on_close`. Propagated by `spreading_activation(propagate_valence=True)` |
| Episode Boundary Rules | Pluggable boundary detection | `memory/episode.py` | `BoundaryRule` callables on `EpisodeBoundaryDetector`. Defaults: tick gap, channel change, scn_tag change. New: `salience_spike_rule(min_intensity=0.5)` triggers boundary on pain/salience spikes via `CaptureEvent.salience_spike` |

### Behavioral Convergence (v0.3.0)

All 3 testing tiers PASS with 41/41 hypotheses confirmed across 4 experiments:

| Tier | Experiment | Result | What it proves |
|------|-----------|--------|----------------|
| Tier 1 | Exp 1: Cross-session affective memory | 11/11 | Bio-systems learn and persist |
| Tier 1 | Exp 2: Energy-driven consumable learning | 13/13 | Energy bridge feeds learning loop |
| Tier 2 | Exp 3: LLM acts on bio-system learning | 12/12 (10/10 vs 0/10) | LLM decisions change with valence context |
| Tier 3 | Exp 4: Organic LLM learning | 5/5 (teal 0%->25%->100%) | Agent learns AND acts organically; fresh control dies |

Tier 3 is the headline result: fully organic LLM-driven training with no scripted reactions. The experienced agent escapes on turn 1 in Session 3; the fresh control agent dies. See [experiments/protocols/](experiments/protocols/) for reproduction protocols and [plans/behavioral_convergence_practice.md](plans/behavioral_convergence_practice.md) for detailed writeups.

---

## Operating Modes

Maxim's mode system combines two dimensions:

1. **ProcessingState** -- `awake` or `sleep`. Determines whether the agent loop is running.
2. **OperationalMode** -- `planning`, `supervised`, or `autonomous`. Controls permissions and initiative level.

Sleep is not a mode -- it is a processing state the agent enters by calling the `sleep` tool and exits automatically when user input arrives.

See [docs/user/modes-guide.md](user/modes-guide.md) for the full guide.

---

## Planning System

| Component | Location | Description |
|-----------|----------|-------------|
| `PlanManager` | `planning/plan_manager.py` | Plan lifecycle management |
| `DecisionEngine` | `planning/decision_engine.py` | Single point of action selection |
| `Policy` | `planning/policy.py` | Constraints and guardrails |
| `PlanDocument` | `planning/plan_document.py` | Structured plan representation |

Decision flow:
```
Observe state -> Agents propose intents -> Planners propose plans
    -> Policies constrain plans -> Decision engine selects action
    -> Runtime executes
```

---

## Feature Reference

| Feature | Description |
|---------|-------------|
| Adaptive Planning | ADaPT + Reflexion hybrid: direct execution first, recursive decomposition on failure |
| Multi-Signal Scoring | Plans scored across 6 dimensions: NAc value, EC familiarity, concept relevance, delay efficiency, depth penalty, action cost |
| Parallel Execution | Independent sub-goals run concurrently with thread pools |
| Reflection Loops | Surprising failures (RPE > 0.3) generate verbal self-critiques stored as episodic memories |
| Prompt Profiles | `minimal`, `standard`, `rich` profiles for different hardware |
| Checkpointing | Persist and resume goal trees across restarts |
| FearAgent | Safety review for sensitive operations before execution |
| Pain Detection | Proprioceptive monitoring for aversive movement patterns |
| Harm Prediction | Zero-latency prediction of harmful outcomes before execution |
| Contemplation | Local chain-of-thought: multi-pass critique+refine for complex plans |
| Energy Tracking | Resource expenditure monitoring for tokens, compute, and movement |
| Introspection Tools | 10 read-only tools exposing biological subsystems to the LLM |
| Learned Tool Index | Keyword-weighted hashtable learns which tools match which goals; saves ~74% of tool-context tokens |
| Coding Tools | Edit files, search code, run tests, git diff/commit with structured error reporting |
| Skills & Protocols | Composable capabilities with lifecycle states and workspace constraints |
| SMS/Voice Comms | Send and receive texts/calls via Twilio |
| Generative Campaigns | LLM-driven narrative arcs, bridge-and-compress for long campaigns |
| Research Protocol | Multi-agent research: Researcher + Writer + Reviewer agents, dual-LLM, experiment tracking |
| Embodiment | SEM protocol (Sensor-Entity-Modulator) for body definition, Cerebellum forward models, motor programs with engrams, ComponentIndex semantic discovery |
| Imagination | Real-time entity design from novel percept mentions: entity extraction → ComponentIndex lookup → LLM design → ephemeral registration. DN arousal-gated, energy-budgeted. Imagined entities carry provenance tags on Episodes and CausalLinks |
| Agent Mesh | Cooperative peer-to-peer network: knowledge sharing, task delegation, distributed planning, SCN clock sync |
| Multi-LLM Scaling | Local + remote + cloud LLM backends, Cloudflare tunnel, per-tier model routing, hot-swap |
| Benchmarks | Multi-model comparative testing with bio-system expectations and scenario suites |

---

## LLM Models

### Local Models

| Profile | Model | Size | Context |
|---------|-------|------|---------|
| `smollm-1.7b` | SmolLM 1.7B Instruct | ~1.1 GB | 4096 |
| `mistral-7b` | Mistral 7B Instruct v0.2 | ~4.4 GB | 4096 |
| `llama3-8b` | Llama 3 8B Instruct | ~4.9 GB | 8192 |
| `phi3-mini` | Microsoft Phi-3 Mini | ~2.3 GB | 4096 |
| `qwen2-7b` | Qwen2 7B Instruct | ~4.4 GB | 8192 |
| `qwen2.5-14b-instruct` | Qwen2.5 14B Instruct | ~9.0 GB | 8192 |

### Cloud Providers

8 cloud providers are supported across 15 profiles: Anthropic (3 Claude models), OpenAI (2 GPT-4o models), Google Gemini (2), Groq (2), Together (1), Fireworks (1), Mistral (2), DeepSeek (2). Cloud dispatch is opt-in via `MAXIM_LLM_CLOUD_ENABLED=1`.

### Quantization

| Level | Quality | Use Case |
|-------|---------|----------|
| `Q3_K_M` | Fair | Memory constrained |
| `Q4_K_M` | Good (default) | Recommended |
| `Q5_K_M` | Better | Quality priority |
| `Q8_0` | Excellent | Maximum quality |

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for Claude backend |
| `OPENAI_API_KEY` | Required for OpenAI backend |
| `GOOGLE_API_KEY` | Required for Gemini backend |
| `GROQ_API_KEY` | Required for Groq backend |
| `TOGETHER_API_KEY` | Required for Together backend |
| `FIREWORKS_API_KEY` | Required for Fireworks backend |
| `MISTRAL_API_KEY` | Required for Mistral API backend |
| `DEEPSEEK_API_KEY` | Required for DeepSeek backend |
| `MAXIM_LLM_ENABLED` | Enable LLM (1/true) |
| `MAXIM_LLM_PROFILE` | Model profile name |
| `MAXIM_LLM_QUANTIZATION` | Quantization level |
| `MAXIM_COMMS_ENABLED` | Enable SMS/Voice comms (1/true) |
| `MAXIM_PROVENANCE_VERBOSITY` | 0=off, 1=compact, 2=verbose |
| `CUDA_VISIBLE_DEVICES` | GPU selection (empty for CPU) |

See [CLAUDE.md](../CLAUDE.md) for the full environment variable reference including heartbeat, cloud, and peer variables.

---

## Persistence

Components that persist state to `~/.maxim/`:

| Component | CLI Clear |
|-----------|-----------|
| FocusLearner | `--clear-memory focus` |
| WorkspaceBoundsLearner | `--clear-memory bounds` |
| EscalationLearningBridge | `--clear-memory escalation` |
| FearCircuitBridge | `--clear-memory fear` |
| AdaptiveThresholdController | `--clear-memory threshold` |
| NAc | `--clear-memory nac` |
| SCN | `--clear-memory scn` |
| Hippocampus | `--clear-memory hippo` |
| ATL | `--clear-memory atl` |
| AngularGyrus | `--clear-memory angular` |
| Cerebellum | `--clear-memory cerebellum` |
| PainDetector | `--clear-memory pain` |
| SemanticEmbeddings | `--clear-memory semantic` |

Clear all: `maxim --clear-memory all`

---

## Reachy Mini Robot Setup

### Prerequisites

1. Reachy Mini on the same LAN/Wi-Fi
2. Pollen Robotics SDK installed (see [SDK guide](https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md))

### Starting the Daemon

```bash
ssh pollen@<REACHY_IP>
sudo systemctl stop reachy-mini-daemon
source /venvs/mini_daemon/bin/activate
python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only
```

### Troubleshooting

```bash
python scripts/check_reachy_connection.py --host <REACHY_IP>
```

| Issue | Solution |
|-------|----------|
| Port 8443 refused | Restart Reachy or run `reachyminios_check` |
| Port 7447 refused | Check daemon: `systemctl status reachy-mini-daemon` |
| Matplotlib crash | `rm -rf ~/.cache/matplotlib && fc-cache -f` |
| Whisper segfaults | `MAXIM_WHISPER_COMPUTE_TYPE=float32 maxim` |
| OpenCV Qt warnings | `MAXIM_DISABLE_IMSHOW=1 maxim` |

---

## GPU Acceleration

Maxim auto-detects GPUs for vision, motor cortex, and LLM inference.

```bash
# Force CPU-only mode
CUDA_VISIBLE_DEVICES="" maxim
```

---

## Outputs

Each run creates timestamped artifacts under `~/.maxim/`:

| Path | Content |
|------|---------|
| `sessions/` | Session data, sim reports |
| `memory/` | Persisted memory state |
| `benchmarks/` | Benchmark results |

---

## See Also

- [README.md](../README.md) -- Getting started
- [ARCHITECTURE.md](../ARCHITECTURE.md) -- Design rules
- [CLI Reference](user/cli-reference.md) -- All CLI flags
- [Modes Guide](user/modes-guide.md) -- Operating modes
- [Python API](user/python-api.md) -- Programmatic usage
- [Simulation Guide](user/simulation.md) -- Simulation framework
