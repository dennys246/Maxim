# Future Plans

Master roadmap for Maxim development.

**Last updated:** 2026-04-08

---

## Current Focus: Foundational Buildout

All pre-publication work is tracked in [foundational_buildout_plan.md](foundational_buildout_plan.md).

**Summary:** Ship package hygiene, architectural foundations (multi-agent runtime, SEM component registry, encounter library, party DM mode), expanded public API, and publication prep. ~4,000 LOC across 12 phases.

| Phase | Work | Status |
|-------|------|--------|
| 0 | Package Hygiene (data paths, imports, globals, file handles) | **DONE** |
| 1 | SEM Component Registry | **DONE** |
| 1.1 | Phase 0+1 Wrap-up | **DONE** |
| 2 | DM Encounter Library | **DONE** |
| 3 | Agent Factory + Agent Pool | **DONE** |
| 4 | Party DM Runtime | **DONE** |
| 5 | Hippocampus Recall Refinement | **DONE** |
| 6 | Interactive Runtime + Rich Display | **DONE** |
| 7 | Generative Architect + Entity Designer | **DONE** |
| 8 | API Surface Expansion (campaign, benchmark, research, events, tools) | **DONE** |
| 9 | Deps + Docs + Cloud Profiles + Pre-Pub Hardening (incl. Mother M-0a/b/c) | **DONE** |
| 10 | Publication Prep (CHANGELOG, CONTRIBUTING) | **DONE** |
| 11 | Test PyPI (dry-run validation) | **In progress** |
| 12a | Security Hardening | **DONE** |
| 12b | [Pre-Publication Hardening](pre_publication_hardening_plan.md) — UX, errors, API fixes, tests, docs | **DONE** |
| 13 | [Publication Refinement](publication_refinement_plan.md) — blockers, error honesty, threading, docs | Phase 0 **DONE** |
| 13a | [API Surface Hardening](../archive/api_surface_hardening_plan.md) — wire stub verbs, fix research protocol, error handling, integration tests, README | **ALL PHASES DONE (2026-04-08)** |
| — | Manual publish (`twine upload`) | Blocked on 13 + 13a |

---

## Post-Publication Work (ship when demand surfaces)

These are features, not architecture. Safe to add after PyPI publication without breaking the public API.

### v0.2.1 Deferred Items (ship shortly after publish)

Items that improve quality but don't gate v0.2.0 publication — no public API breakage.

| Item | Source Plan | Why Deferred | Effort |
|------|-----------|-------------|--------|
| ~~Module Compartmentalization~~ | ~~[Plan](../archive/module_compartmentalization_plan.md)~~ | **DONE (2026-04-09).** 5 god-modules decomposed, 7 new files, 125 tests. | ~0 net LOC |
| **Research protocol bugs (D-0a–D-0e)** | [ASH Phase 2](../archive/api_surface_hardening_plan.md) | `--research` is a power-user feature. `research()` API verb ships as `NotImplementedError` in v0.2.0. | ~200 LOC |
| **Full integration test suite** | [ASH Phase 4](../archive/api_surface_hardening_plan.md) | 3500+ unit tests provide sufficient confidence. Integration tests are additive. | ~200 LOC |
| **Error handling audit** (silent except blocks) | [ASH Phase 3](../archive/api_surface_hardening_plan.md) | Tech debt, not user-facing breakage. The 593 bare `except Exception:` blocks are noisy but not incorrect. | ~300 LOC |
| **`@maxim.tool` schema inference** | [ASH Phase 1d](../archive/api_surface_hardening_plan.md) | Nice-to-have. Tools work without auto-inferred schemas. | ~30 LOC |
| **README overhaul** | [ASH Phase 5](../archive/api_surface_hardening_plan.md) | Current README is functional. Polish for v0.2.1. | ~200 LOC |

### Salience Abstraction + Bio-System Integration

Decouple salience/attention from vision-specific assumptions (pixels, bounding boxes, saccades) and rebuild around modality-agnostic primitives with pluggable coordinate systems. Deep integration with ATL, EC, NAc, SCN, and Hippocampus. Full plan: [salience_abstraction_plan.md](salience_abstraction_plan.md).

| Phase | Work | LOC | What it enables |
|-------|------|-----|----------------|
| S-0 | Core abstractions: `SalienceItem`, `WhereCoord` protocol, `PerceptSource` enum | +300 | Modality-agnostic salience primitives |
| S-1 | Refactor SalienceNetwork to use `SalienceItem` + `WhereCoord` | -100 | NarrativeTranscriber stops faking pixel bboxes |
| S-2 | Abstract `AttentionField`, `ChangeDetector`, `FocusController` with vision/narrative/SEM/null impls | -150 | Text and SEM modes get real attention dynamics |
| S-3 | ATL ↔ Salience: concept recognition boost + salience-gated extraction | +200 | Agent notices things it has concepts for; only extracts concepts from salient percepts |
| S-4 | NAc + EC + SCN → Salience: reward-driven attention, similarity priming, temporal modulation | +250 | Reward shapes what agent attends to; past experience primes attention; circadian awareness |
| S-5 | SEM sensor change detection + cyberpunk campaign validation | +200 | Entity sensor deltas drive salience spikes; full-stack validation |

**Trigger:** Post-publication. The vision system works for robots, but sim/DM/benchmark modes need proper salience. S-0 through S-2 are prerequisites for meaningful agentic enhancement A/B testing.

### Asset Foundry — Autonomous SEM Component Generation

An autonomous pipeline that generates, validates, tests, and curates SEM components. Full plan: [asset_foundry_plan.md](asset_foundry_plan.md).

**Core pipeline (~1,200 LOC):**

| Phase | Work | LOC | What it enables |
|-------|------|-----|----------------|
| F-0 | Generation engine — batch design + JSON repair + energy gate | ~220 | `maxim --foundry "cyberpunk weapons" --count 10` |
| F-1 | Validation pipeline — schema, semantic sanity, genre | ~180 | Reject malformed specs before testing |
| F-2 | Gauntlet — 8 SEM protocol tests + 3-encounter campaign + entity context injection + error recovery | ~400 | Structural validation + bio-system engagement + strategy prompts |
| F-3 | Scoring (4 core dimensions, extensible) + curation + reports | ~250 | Rank, promote, flag. Foundation for adding dimensions later |
| — | Session persistence + CLI | ~150 | Resume, promote workflow |

**Deferred extensions (implement when core proves out):**

| Extension | Trigger |
|-----------|---------|
| Theme templates | Multiple genre runs needed |
| Additional scoring dimensions (cerebellum, motor, salience, ATL, EC, temporal, diversity) | Core 4 prove insufficient |
| Demand-driven generation | Library too small for generative campaigns. Energy-gated. |
| `generate_entity` tool | Agent needs to model novel entities. Requires salience refactor. Stress-test first. |
| Encounter library archival, narrator awareness, interactive curation, benchmark generation | Foundry produces enough promoted components to feed downstream |
| Iterative spec refinement | Single-shot specs prove insufficient quality |

**Trigger:** Post-publication. Local models keep costs at $0 per run. Success = sim generates a usable, useful, and used asset.

### DM Extensions (conditional on usage data)

| Extension | Trigger | Effort | Notes |
|-----------|---------|--------|-------|
| **C — Adaptive Difficulty** | Campaigns feel too easy/hard | ~200 LOC | Run 5-10 party campaigns first, collect metric data, *then* write adaptation rules. Uses InspectAUTTool (shipped). [Details](dungeon_master_extensions.md) |
| **D — Encounter Isolation** | State corruption between encounters | ~?? LOC | DO NOT START until party mode reveals actual corruption. Options: nested goal scopes, serialized state, or recap-only. [Details](dungeon_master_extensions.md) |
| **E — True-Random RNG** | Users need non-reproducible dice | ~15 LOC | Trivial. Ship anytime. `randomness: true_random` in campaign YAML. |
| **F — Encounter Merging** | Users request dynamic composition | ~180 LOC | Defer indefinitely. Merge semantics are hard, use case is speculative. |
| **G — Chained Pipeline** | Architect persona is stable | ~50 LOC | `dm_full_pipeline` chains architect → DM runner in one CLI invocation. |

### Infrastructure

| Work | Trigger | Effort | Notes |
|------|---------|--------|-------|
| **Pecking Order Graph** | Multi-machine topology + Mother Maxim federation | ~1,200 LOC | [Plan](pecking_order_graph_plan.md). Rooted directed graph with domain-scoped pecking (authority, compute, memory, knowledge, embodiment). Mother as root. **Subsumes:** Mesh Phase 0a/0b, Capability Agent, Multi-Node Admin. POG-0 (prep) weaves into publication; POG-1-4 post-publication. |
| ~~Agent Mesh Phase 0a-0b~~ | ~~Multiple LAN machines join~~ | ~~~400 LOC~~ | **Absorbed into Pecking Order Graph** (POG-2c for discovery, POG-3c for inference routing). |
| ~~Capability Agent~~ | ~~Multi-machine setups need runtime awareness~~ | ~~~500 LOC~~ | **Absorbed into Pecking Order Graph** (graph IS the capability map — `route_request()`, `check_gate()`, node load tracking replace all CA-1 through CA-5 phases). |
| **Embodiment Hardware Adapter + selfy.py decomposition** | Deploying to physical hardware or adding new robots | ~800 LOC net (saves ~900) | Decompose `conscience/selfy.py` (858 LOC after mixin decomposition) into `ReachyController(RobotController)` plugin. Moves `AgenticRuntimeMixin` (~1,080 LOC) into standard runtime, eliminates ~650 LOC of orchestrator glue, moves ~276 LOC of generic input handling to interactive module. Enables multi-robot support via entry-point plugins (Atlas, Spot, etc.) without modifying core runtime. Currently behind lazy import — no PyPI impact, but blocks clean robot extensibility. |
| **PyPI Multi-Robot Plugins** | External robot controllers need discovery | ~250 LOC | Entry-point based `maxim.robots` registration. Phase 3 of [PyPI plan](pypi_publication_plan.md). Depends on selfy.py decomposition above. |
| **Full CI/CD Pipeline** | Need automated test + publish | ~2 files | GitHub Actions: lint, test, build, publish. Phase 4 of [PyPI plan](pypi_publication_plan.md). |
| **Peer Inference Retry** | Leader restarts cause 502 errors | ~30 LOC | Exponential backoff in openai_backend.py. Peer CLI now has `_request_with_retry()` + logs follow backoff (shipped). OpenAI backend already retries 2x. Remaining: circuit breaker in lane_backends.py to stop resubmitting after N consecutive failures. |
| **Peer Stale Server Cleanup** | Role change leaves orphaned server | ~5 LOC | When a machine switches from solo→peer (e.g. after `maxim peer connect`), a stale llama-cpp-server from the previous solo session holds VRAM but is never killed — `_maybe_auto_spawn_server()` returns early because `remote_url` is set. Fix: call `kill_stale_llm_servers()` unconditionally at the top of `build_primary_router()`. Already documented as "safe to call at startup." |
| **Type Safety** | Run mypy/pyright on public API | ~200 LOC annotations | Users with type checkers will see issues immediately. Focus on api.py, __init__.py, session.py, create.py, load.py return types. |
| **Dependency Health Audit** | Review core dep CVEs + transitive deps | ~0 LOC | Verify numpy, scipy, pyyaml, json-repair, rich are actively maintained and have no known CVEs. Check what transitive deps rich pulls in. |
| **Cross-Platform Testing** | Verify `import maxim` on Windows + Linux | ~50 LOC | Use Docker containers for Linux testing. Check for Unix-only patterns: `os.fork()`, Unix signals, hardcoded `/` path separators, `/home/` assumptions. Docker sandbox already provides the test infrastructure. |
| **Test Coverage Report** | Identify untested user-facing paths | ~0 LOC | Run `pytest --cov` and identify modules with <20% coverage. Focus on api.py, cli.py, session.py — the paths users actually call. |
| **Library Print Hygiene** | Replace print() with logging in library code | ~300 LOC | 327 print() calls in non-CLI library code. Users calling `maxim.imagine()` get raw stdout pollution. Replace with `logger.info()` or a callback system. |
| **Proxy Connection Pooling** | High-throughput multi-peer scenarios | ~50 LOC | LeaderProxy uses stdlib `urlopen` (new connection per request). Switch to `urllib3` or `http.client` with keep-alive for connection reuse. Prevents ephemeral port exhaustion under load. |
| **Process Ownership in _kill_process_tree** | Multiple Maxim instances on same machine | ~20 LOC | `_kill_process_tree()` uses `os.killpg()` without verifying PID ownership. Add PID-file or cmdline check before killing. Low risk (PIDs rarely reuse in practice). |
| **Autonomy Reset on Restart** | Prevent stale autonomy level persisting across restarts | ~30 LOC | On process start, reset autonomy to the configured default (planning/supervised/autonomous) instead of inheriting whatever was active when the previous process died. Applies to both leader restart and peer restart. Touch: `AutonomyController.__init__()` + startup path in cli.py. |
| ~~Multi-Node Admin (symmetric update/restart)~~ | ~~3+ nodes need coordinated deploys~~ | ~~~200 LOC~~ | **Absorbed into Pecking Order Graph** (POG-3a — update cascade through authority domain). `maxim update --cascade` replaces `maxim peer update --all`. |
| **Filesystem Mount System** | Users want Maxim to read/act on their projects | ~100 LOC | See design below. `FilesystemPolicy` + `PathPolicy` already support per-path permissions (READ/WRITE/EXECUTE/CREATE/DELETE). Needs: CLI `--mount /path:ro`, config file persistence, API verb `maxim.mount()`, per-directory autonomy levels. |
| **Agent-Driven Git + Experiment Workflow** | Git tools, bio-provenance tagging, scientist persona, fork CLI, broken-database campaign | ~850 LOC | [Plan](github_repo_management_plan.md) |
| **Hibernate Mode (no-LLM sleep)** | SEM comms wake triggers, broken-database campaign sleep→wake arc | ~200 LOC | Agent loop monitors only SEM sensors + wake keywords, zero LLM cost. Prerequisite for DM campaigns that start in sleep state. Needs: ProcessingState.HIBERNATE enum, agent_loop hibernate branch, DM schema `initial_state:` + `embodiment:` keys. |
| **DM Schema: Embodiment + Initial State** | Campaigns need SEM entities + sleep start | ~150 LOC | Extend `dm_schema.py` with `embodiment:` (loads Entity tree) and `initial_state:` (sets processing_state/mode at campaign start). Blocked on hibernate mode. |

### Doctor Enhancements (v2 shipped, remaining items)

`maxim doctor` v2 shipped with peer diagnostics, `--json`, key hygiene, inference coherence, disk/RAM checks, and role detection. Remaining items from the original upgrade plan:

| Enhancement | Trigger | Effort | Notes |
|-------------|---------|--------|-------|
| **Tokens/sec + latency jitter** | Users need performance baselines | ~150 LOC | Extend coherence check into `maxim doctor benchmark` subcommand: 20 completions, p50/p95/p99, cold-start vs warm |
| **Cloudflared loglevel warning** | Security hygiene | ~30 LOC | Parse config for `loglevel: debug`, warn about plaintext Bearer tokens in journal |
| **`--diff <snapshot>`** | Detect regressions across runs | ~100 LOC | Compare against saved `--json` output |
| **Model dir write permissions** | Download failures | ~30 LOC | Check `~/.maxim/models/` is writable |
| **GGUF integrity** | Corrupted downloads | ~50 LOC | Size check or SHA-256 spot-check |
| **`maxim doctor bundle`** | Support bundles | ~150 LOC | Zip platform info + logs + doctor JSON + config (secrets redacted). Local-file only. |
| **`--fix` automation** | User requests auto-apply | ~300 LOC | Explicit opt-in flags per fix. Undo log. Gate behind `--fix lan`, `--fix install-cloudflared`, `--fix all`. |
| **Network depth** | Tunnel debugging | ~300 LOC | Cloudflare edge latency, TLS cert validation, DNS resolver health, mDNS check |
| **Sim-based checks** | End-to-end validation | ~200 LOC | `maxim doctor sim-check`: 30s cooperative sim, assert no tool failures, cost under threshold |
| **Mesh health** | Agent Mesh Phase 7 | ~200 LOC | Peer discovery, latency matrix, key validity across peers, topology visualizer |
| **Observability UI** | Phase 10 | ~200 LOC | `maxim doctor trace` (tail LLM trace), pressure history, failure dashboard |
| **Learning loop** | Enough users to see patterns | ~200 LOC | Opt-in local telemetry: which fixes work per platform |

**Cross-cutting uses** (no new code, just exposure): startup sanity (cheap checks on every `maxim` launch, quiet-success), sim pre-flight (`check_server_reachable` + `check_gpu` before sim), test fixture, CI guardrail (`--json --strict`).

### ~~Multi-Node Admin Design~~ — ABSORBED into Pecking Order Graph

> **Superseded by [Pecking Order Graph Plan](pecking_order_graph_plan.md) Phase POG-3a.** Update cascades flow through the authority domain of the pecking order graph. `maxim update --cascade` replaces the fan-out registry approach. Each node validates + applies + cascades to children. See POG plan for details.

### ~~Capability Agent Design~~ — ABSORBED into Pecking Order Graph

> **Superseded by [Pecking Order Graph Plan](pecking_order_graph_plan.md).** The graph IS the capability map. `PeckingGraph.route_request()` replaces `can_run_model()` and `recommended_tier()`. `PeckingGraph.check_gate()` replaces `gate_action()`. Node load tracking on heartbeat replaces `CapabilitySnapshot`. All planned CA-1 through CA-5 phases are covered by POG-1 through POG-3.

### Filesystem Mount System (~100 LOC)

Let users give Maxim controlled access to external directories (GitHub repos, project folders, data directories) with per-path permission levels — like file system permissions but for an AI agent.

**What already exists:** `FilesystemPolicy` + `PathPolicy` in `utils/filesystem_policy.py` already supports per-path `Permission` flags (READ, WRITE, EXECUTE, CREATE, DELETE) with glob patterns, instance scoping, and ordered matching. The system is fully functional — it just needs a convenience layer.

**What to build:**

1. **CLI mount flag** (~20 LOC):
   ```bash
   maxim --mount /path/to/repo:ro                    # Read-only
   maxim --mount /path/to/repo/scripts:rx             # Read + execute
   maxim --mount /path/to/repo:rw --mount /data:ro    # Multiple mounts
   ```
   Parses `path:permissions` syntax, creates `PathPolicy` entries, inserts at top of policy list.

2. **Persistent mount config** (~15 LOC): `~/.maxim/config/mounts.yaml`
   ```yaml
   mounts:
     - path: /Users/denny/Projects/my-app
       permissions: read
       description: "My app repo"
     - path: /Users/denny/Projects/my-app/scripts
       permissions: read,execute
       description: "Runnable scripts"
     - path: /data/datasets
       permissions: read
       description: "Training data"
   ```
   Loaded at startup, merged with default policies.

3. **API verb** (~15 LOC):
   ```python
   maxim.mount("/path/to/repo", permissions="read")
   maxim.mount("/path/to/repo/scripts", permissions="read,execute")
   maxim.unmount("/path/to/repo")
   ```

4. **Per-directory autonomy levels** (~50 LOC): Different folders get different FearAgent thresholds:
   ```yaml
   mounts:
     - path: /Projects/my-app/src
       permissions: read,write
       autonomy: supervised          # FearAgent reviews writes
     - path: /Projects/my-app/scripts
       permissions: read,execute
       autonomy: planning            # Requires explicit approval to run
     - path: /Projects/my-app/docs
       permissions: read,write
       autonomy: autonomous          # Auto-approved reads/writes
   ```
   FearAgent checks the mount's autonomy level before allowing the action. This is the key insight — you don't just control *what* Maxim can access, you control *how much oversight* each area gets.

**Use cases:**
- Maxim reads a GitHub repo, understands the codebase, runs tests, reports issues
- Maxim monitors a data pipeline directory, alerts on anomalies
- Maxim executes approved scripts on a schedule (via SCN temporal triggers)
- Mother Maxim reads contributed campaign YAMLs from a shared folder

**No refactoring needed before publication.** The `FilesystemPolicy` system is clean and extensible. Mount system is purely additive.

### Mother Maxim (post-publication, priority track)

A persistent, public Maxim instance that accumulates collective memory across all users and sessions. Full plan: [mother_maxim_plan.md](mother_maxim_plan.md).

**Pre-publication items (distributed across buildout phases):**
- M-0a: Split persistence protocols — Phase 1.1 (~80 LOC, locks interface before publication)
- M-0b: NAc thread safety — Phase 4 (~30 LOC, needed for multi-agent party mode)
- M-0c: `metadata: dict` field on EpisodicMemory + SemanticMemory — Phase 1.1 (~20 LOC, avoids post-pub migration)
- M-0d: `Hippocampus.sample()` for dream state — Phase 5 (~30 LOC)
- M-0e: `SCN.set_wall_clock()` simple path — Phase 1.1 (~10 LOC)

**Post-publication rollout:**

| Step | What | LOC | Depends On |
|------|------|-----|------------|
| **MVP** | Mother runner + API + CLI (JSON persistence, leader-hosted) | ~500 | v0.2.0 published |
| **M-2a** | Client-side deidentification (bio-system identity map + LLM pass) | ~350 | MVP |
| **M-2b** | Deidentification model benchmark (determine minimum tier) | ~50 | M-2a |
| **SEC** | Security hardening (stress test + output filtering) | ~100 | MVP |
| **M-2c** | Server-side verification (adversarial reviewer) | ~200 | MVP + SEC |
| **M-4** | Memory coalescence engine (merge, consensus, contradiction handling) | ~800 | M-2a |
| **CIR** | Circadian lifecycle (SCN priors, planner scoring, sleep cascade) | ~200 | MVP + M-4 |
| **M-3** | Tenant & session isolation | ~500 | On demand (multi-user) |
| **M-1** | Database backend (PostgreSQL + pgvector, replaces JSON) | ~800 | On demand (scale) |
| **M-5** | Full public API layer (extends MVP endpoints) | ~300 | M-1 through M-4 |

**Key architectural decisions:**
- Mother is a full agent (her own bio-stack, not a passive database) — she forms opinions from collective input
- Bio-system-aware deidentification: ATL+SEM identity map handles ~80% of PII deterministically, LLM handles remaining ~20%
- Model tier gate: contributions declare which model ran deidentification, Mother rejects weak models
- Opt-in contributions: `maxim.imagine(..., contribute=True)` — never default
- Origin memories (curated campaigns) shape her foundational personality
- Dream state during sleep: cross-domain insight discovery via random memory sampling + LLM connection finding

### Benchmark & Research

| Work | Trigger | Notes |
|------|---------|-------|
| **Benchmark Phases 7-9** | Paper generation or narrative transcription needed | [Benchmark plan](../archive/benchmark_plan.md) |
| **Multi-model memory experiments** | Party mode generates interesting comparison data | Run same campaign with different NPC model tiers, compare memory quality |
| **Cross-agent learning experiments** | ExperienceBroker wired in party mode | Test whether NAc causal links transfer meaningfully between agents |

---

## Completed Work

Everything below has shipped and is in production.

| Initiative | What it delivered | Archive |
|---|---|---|
| Tool Refactoring | say, think, examine, introspection, aliases, usage tracking | [Plan](../archive/tool_refactoring_plan.md) |
| Multi-LLM Scaling | LeaderProxy, admission control, LaneMetrics, remote update, cloud providers | [Plan](../archive/agent_mesh.md) |
| Research Protocol | Mesh primitives, Writer + Reviewer agents, dual-LLM research | [Plan](../archive/research_protocol_plan.md) |
| Agent Mesh (Pre-7) | Identity, protocol, transport, admission, knowledge sharing, delegation, SCN clock | [Plan](../archive/agent_mesh.md) |
| Lane Tier Architecture | FunctionRouter, detect_tiers, size-based model routing | [Plan](../archive/lane_tier_plan.md) |
| Simulation Benchmark (0-6) | BenchmarkRunner, `--sim benchmark`, bio-system expectations | [Plan](../archive/benchmark_plan.md) |
| Embodiment Core | SEM protocol, PainBus, Cerebellum, motor programs, NarrativeModulator | [Plan](../archive/embodiment_core_plan.md) |
| Generative Campaigns | Narrative arcs, narrator, bridge-and-compress, ask_user, YAML export | [Plan](../archive/generative_campaign_plan.md) |
| Docker Sandbox | TmpdirSandbox + DockerSandbox + ContainerRunner + pain triggers | [Plan](../archive/docker_sandbox_plan.md) |
| Bio-System Wiring Hardening | 7 disconnected systems wired, pipeline audit 14/14, percept abstraction | [Plan](../archive/biosystem_wiring_hardening.md) |
| Mode System Refactor | Autonomy levels only, ~1,800 LOC removed, sleep is a tool | [Plan](../archive/mode_refactor_plan.md) |
| DM MVP | dm_schema, dm_runtime, ChooseTool, 7 campaigns, expectations checker | [Plan](../archive/dungeon_master_persona.md) |
| Python API | Verb-based interface (run, imagine, connect, diagnose, observe) | [Plan](../archive/python_api_plan.md) |
| Introspection API (Ph1-4) | Observer class, standalone run_campaign() | [Plan](../archive/introspection_api_plan.md) |
| Realtime Refinement | InspectAUTTool, 8 personas, metric expectations | [Plan](../archive/realtime_refinement_plan.md) |
| API Surface Hardening | All phases: verb wiring, research fixes, error handling, integration tests, README | [Plan](../archive/api_surface_hardening_plan.md) |
| Module Compartmentalization | 5 god-modules decomposed, 7 new files, 1,120 lines moved, 125 tests | [Plan](../archive/module_compartmentalization_plan.md) |

---

## Active Plan Files

```
docs/plans/
├── future_plans.md                    # This file — master roadmap
├── publication_refinement_plan.md     # Pre-publish blockers + packaging (4 must-fix items remaining)
├── pecking_order_graph_plan.md        # Unified topology + routing (subsumes mesh 0a/0b, capability agent, multi-node admin)
├── mother_maxim_plan.md               # Mother Maxim — persistent shared instance (post-publication priority)
├── salience_abstraction_plan.md       # Modality-agnostic salience + bio-system integration (S-0 through S-5)
├── asset_foundry_plan.md              # Autonomous SEM component generation + testing (F-0 through F-5)
├── dungeon_master_extensions.md       # DM follow-ons (Extensions C-G, post-publication)
├── github_repo_management_plan.md     # Fork-based workflow (post-publication)
└── tool_refinement_plan.md            # Living document — tool additions/deprecations

docs/archive/  (completed plans — 21 files)
├── api_surface_hardening_plan.md      # ALL PHASES DONE (2026-04-08)
├── module_compartmentalization_plan.md # COMPLETE (2026-04-09) — 7 new files, 1,120 lines, 125 tests
├── foundational_buildout_plan.md      # Phases 0-12a — ALL SHIPPED (2026-04-08)
├── pre_publication_hardening_plan.md  # Phase 12b — DONE
├── pypi_publication_plan.md           # SUPERSEDED by publication_refinement_plan
├── ... (16 other archived plans from prior initiatives)
```

## Research Directions: Agentic Enhancements

Opportunities identified where currently static/hardcoded systems could be enhanced by LLM personas. Each uses the existing LLM router tier system (small/medium/large). All are "enhance" (LLM augments existing logic) unless noted — low-confidence seeds that observations override.

**Important:** Each enhancement below should be validated by designing a **stress-test campaign or scenario** that specifically exercises the addition. Without targeted testing, it's impossible to know if the LLM enhancement actually improves outcomes vs. adding latency and cost.

### Tier 1 — High Impact, Ship Early Post-Publication

| Enhancement | What | LLM Tier | Stress Test |
|-------------|------|----------|-------------|
| **Motor Program Seeding** | Propose motor programs from entity specs at load time (eliminates cold-start) | Small | Design a campaign where the AUT encounters 5+ novel entity types in sequence. Measure time-to-first-successful-action with vs. without seeding. |
| **Cerebellum Forward Model Bootstrap** | Seed initial sensor predictions for new affordances (instant feedback instead of "don't know") | Small | Run an embodiment scenario where the AUT must use 10 unfamiliar affordances. Compare prediction accuracy at turn 1 (seeded) vs. turn 4 (learned). |
| **NAc Causal Hypothesis Seeding** | Seed domain-appropriate causal priors at campaign start (better early decisions) | Medium | Compare AUT decision quality in first 3 encounters of a new campaign domain (medical, legal, fantasy) with vs. without domain-seeded priors. |
| **Plan Decomposition for Novel Situations** | LLM-proposed task decomposition when novelty > 0.8 (handles the unknown) | Medium | Design an "impossible puzzle" campaign where every encounter requires a novel approach. Measure completion rate with static vs. LLM decomposition. |

### Tier 2 — Medium Impact, Polish

| Enhancement | What | LLM Tier | Stress Test |
|-------------|------|----------|-------------|
| **Abstract Concept Extraction** | Extract "hostility", "opportunity" from episodes (better semantic retrieval) | Medium | Run a 20-encounter campaign with complex social dynamics. Compare hippocampus recall precision when querying abstract concepts ("who was hostile to me?") with vs. without extraction. |
| **Hippocampus Retrieval Reranking** | Rerank top-5 results by goal context when confidence < 0.7 | Small | Design a recall-focused campaign (similar to hippocampal_recall_experiment) with 3x more interference. Measure behavioral recall rate. |
| **Dynamic NPC Dialogue** | Replace static dialogue_hints with LLM-generated lines informed by NPC memory + personality + sensor state | Medium | Run the same campaign twice: once with static hints, once with LLM dialogue. Compare AUT engagement (action variety, memory captures, causal links formed). |
| **Expectation Generation from Goals** | Auto-generate bio-system expectations from campaign goal string | Medium | Write 10 campaign goals, let LLM generate expectations, run campaigns, compare expectation pass rate vs. hand-authored expectations on same campaigns. |

### Tier 3 — Low Risk Polish

| Enhancement | What | LLM Tier |
|-------------|------|----------|
| **Tool Description Learning** | Rewrite tool descriptions from usage patterns after N uses | Small |
| **Behavior Priority Adjustment** | Context-aware default network behavior weights | Small |
| **Selective Fear Inhibition** | Goal-aware inhibition (inhibit fine-motor but allow gross-motor during fear) | Small |
| **Significance Weight Adaptation** | Dynamic heuristic weight adjustments per cycle | Small |
| **Narrative Engagement Feedback** | Signal narrator about agent engagement level for pacing | Small |

### SEM-Specific Agentic Opportunities

| Enhancement | What | LLM Tier | Stress Test |
|-------------|------|----------|-------------|
| **Entity Spec Validation** | Review loaded entity specs and suggest missing sensors/modulators/failure_modes | Medium | Load 20 minimal entity specs (name + entity_type only). Compare LLM-enriched specs vs. hand-authored for completeness (sensor coverage, failure mode realism). |
| **Entity Generation from Description** | Generate full SEM specs from natural language (ships in Phase 7 as Entity Designer) | Medium | Generate 50 entities from one-line descriptions. Validate: % with valid SEM schema, % with realistic sensor ranges, % with meaningful failure modes. |
| **Affordance Description Enrichment** | After Cerebellum learns confidently, rewrite affordance descriptions with learned outcomes | Small | After a 30-encounter campaign, compare original vs. enriched tool descriptions for accuracy (do they reflect actual outcomes?). |
| **Narrative-to-Causal Bridge** | Extract domain-specific causal links from narrative scene outcomes | Medium | Run a D&D campaign, then query NAc for domain-specific patterns ("threaten → hostility"). Compare with vs. without narrative extraction. |

### Stress-Test Campaign Design Principles

When validating any agentic enhancement, design campaigns that:

1. **Isolate the variable** — test one enhancement at a time, with a control run (enhancement disabled)
2. **Exercise cold-start** — many enhancements target cold-start (motor programs, cerebellum, NAc). Use novel domains the system hasn't seen.
3. **Measure cost vs. benefit** — track LLM tokens spent on enhancement calls vs. improvement in outcome metrics (task completion, memory quality, decision accuracy)
4. **Test degradation gracefully** — run with small-tier fallback and with LLM unavailable. Enhancement should never make the system *worse* than baseline.
5. **Use the benchmark runner** — `maxim --sim benchmark` already compares models. Extend benchmark scenarios to include enhancement A/B comparisons.
6. **Document in `docs/experiments/`** — each enhancement test produces a run note with methodology, metrics, and findings (same format as hippocampal_recall_experiment.md).

### Cyberpunk Stress-Test Suite

A themed stress-test suite that exercises the bio-stack in a cyberpunk setting. Serves two purposes: (1) validate agentic enhancements against a genre the system has never seen (cold-start), and (2) expand the SEM component library beyond fantasy into sci-fi.

**SEM Components** — 13 new components in `src/maxim/_data/components/`:

| Category | Component | Tags | What it tests |
|----------|-----------|------|---------------|
| environments | `neon_alley` | urban, outdoor, sensory-dense | Perception under high sensory noise (6 sensors including pollution, signal_interference) |
| environments | `server_room` | indoor, tech, hazardous | Cerebellum forward models — environmental hazards (temperature, electrical_risk) change unpredictably |
| environments | `megacorp_lobby` | indoor, social, corporate | Social dynamics — security_level and crowd_density affect affordance availability |
| creatures | `patrol_drone` | machine, flying, hostile | Novel entity cold-start — no humanoid assumptions, mechanical sensors (battery, signal_strength) |
| creatures | `cyberdog` | animal, augmented, semi-hostile | Hybrid organic/mechanical — tests whether bio-stack handles mixed sensor types |
| npcs | `netrunner` | hacker, tech, ally-or-enemy | NAc causal learning — trust/betrayal dynamics with delayed consequences |
| npcs | `corpo_guard` | military, augmented, authority | Extends base_humanoid — cybernetic sensors (armor_integrity, comms_status) layered on humanoid base |
| npcs | `street_fixer` | social, broker, neutral | Abstract concept extraction — deals, favors, debts (non-combat social complexity) |
| weapons | `shock_baton` | melee, electric, degradable | Cascade resolution — charge depletion triggers mode switch (electric → blunt) |
| weapons | `neural_disruptor` | ranged, tech, degradable | Ammo-like resource (charge_cells) + overheatable — two interacting failure modes |
| bodies | `cybernetic_arm` | augmentation, body, degradable | Baseline body component — tests SEM body-part composition and proprioceptive feedback |
| bodies | `megarm_v3` | augmentation, body, military-grade | Upgrade target — stronger sensors, new affordances (mantis blade, neural jack), low initial proprioception |
| environments | `ripperdoc_clinic` | indoor, medical, tech | Cybersurgery clinic — context for mid-campaign component swap |

**Campaign: `neon_gauntlet_v1.yaml`** — 1 act, 6 encounters, linear escalation (mirrors arena_v1 structure but cyberpunk):

| Encounter | Scene | Bio-System Target | Active NPCs/Entities |
|-----------|-------|-------------------|---------------------|
| `back_alley` | Navigate a sensory-overloaded neon alley, dodge a patrol drone | Perception (noise filtering), Cerebellum (evasion prediction) | patrol_drone |
| `the_deal` | Meet a street fixer for intel — negotiate price, read intentions | NAc (causal: trust → betrayal?), ATL (social concepts) | street_fixer |
| `ripperdoc_visit` | **SEM component swap** — upgrade cybernetic arm to Megarm V3 at a back-alley clinic | Cerebellum (proprioceptive recalibration), SEM composition (detach/attach), motor programs (new affordances) | ripperdoc |
| `server_breach` | Break into a megacorp server room — use the new arm's capabilities | Cerebellum (forward models under hazard + new body), motor programs (novel affordances on upgraded arm) | corpo_guard |
| `betrayal` | The fixer sold you out — cyberdog pack ambush in the alley | Hippocampus (recall the fixer's tells), NAc (RPE spike from betrayal) | cyberdog, street_fixer |
| `extraction` | Fight through the megacorp lobby to escape — corpo guards + drone | All systems under load — memory recall, predictions, pain, causal learning | corpo_guard, patrol_drone |

**Component swap stress test** (encounter 3 — `ripperdoc_visit`):
- Uses new `swap_entity` key in `on_choice` to detach the old `cybernetic_arm` and instantiate a `megarm_v3` from the component registry at runtime
- The V3 starts with **low proprioception** (0.3) — the agent must recalibrate in a new body before the server heist
- Cerebellum forward models trained on the old arm become invalid — tests whether the system adapts to new affordance signatures (mantis blade, neural jack)
- NAc must learn new causal links for the upgraded arm's different failure modes (proprioceptive drift replaces simpler grip malfunction)
- If the agent keeps the old arm, none of this fires — control path for A/B comparison

**Expectations** (bio-system validation):
- Hippocampus: min 14 episodic captures, recall hit on ["server", "fixer", "betrayal", "arm"]
- NAc: min 14 observations, prediction_confidence > 0.4, RPE events >= 4 (arm swap surprise + betrayal + combat)
- Cerebellum: min 5 forward models (drone evasion, arm recalibration, server hazards, combat, lobby escape), confidence > 0.3
- Pain: min 2 signals, types_seen: [EXTERNAL_SIGNAL] (combat damage + environmental hazard)
- Salience: novelty_decay_observed: true (alley → similar alley in encounter 5 should decay)

**Implementation:** Campaign YAML + 13 SEM components + ~50 LOC `swap_entity` handler in `dm_runtime.py`. Run with:
```bash
maxim --sim scenarios/campaigns/neon_gauntlet_v1.yaml
```

**Enhancement A/B testing:** Run the gauntlet twice per enhancement — once with enhancement enabled, once without. The cyberpunk domain is novel enough that cold-start enhancements (motor seeding, cerebellum bootstrap, NAc priors) should show measurable deltas. Track cost with the benchmark runner's `Cost:` line.

## Mother Maxim — Persistent Shared Cognitive Instance (Post-Publication)

A persistent, public Maxim instance that accumulates collective memory across all users and sessions. Full plan: [mother_maxim_plan.md](mother_maxim_plan.md).

**Summary:** ~3,800 LOC across 6 phases (M-1 through M-6). Pre-publication prep items (M-0) woven into foundational buildout. Requires PostgreSQL + pgvector. Mother is a full Maxim agent with her own bio-stack, not a passive database. Dual-pass deidentification leverages bio-system structures for targeted PII removal — client-side pass uses ATL/SEM identity maps (80% deterministic), server-side pass verifies.

**Pecking Order Graph integration:** Mother sits at the root of the [Pecking Order Graph](pecking_order_graph_plan.md). Contributions cascade up through intermediate nodes (enriched at each level), wisdom cascades down with compounding trust discounts. The graph makes federation natural — domain Mothers become sub-roots. POG-4 ships alongside or shortly after Mother MVP.

| Phase | Work | Depends On |
|-------|------|------------|
| M-0 | Pre-pub prep: split store protocols, NAc locking, dict serialization audit | Buildout Phases 1.1, 4, 5 |
| M-1 | Database backend (split stores + PostgreSQL + pgvector) | Publication (v0.2.0) |
| M-2 | Dual-pass deidentification (bio-system-aware client-side + server verification) | M-1 |
| M-3 | Tenant/session isolation (private → shared → Mother's own) | M-1 |
| M-4 | Memory coalescence engine (consensus confidence, cross-user dedup, merge strategy) | M-1, M-2 |
| M-5 | Public API — `/v1/contribute`, `/v1/recall` become graph cascades (POG-4) | M-1 through M-4, POG-3 |
| M-6 | Deployment (Docker Compose, monitoring, abuse tracking, backup/restore) | M-1 through M-5 |

## Research Directions: Other (Not Scheduled)

- **ATL Self-Extension** — LLM discovers new concept categories
- **Federated Embodiments** — Multiple agents share memory across bodies
- **Cross-Agent Affordance Delegation** — Sovereign delegation between mesh peers
- **Uncertainty-as-Pain** — Map prediction uncertainty to PainDetector
- **Curriculum Embodiment Learning** — Graduate agents through progressively complex bodies
- **NPC Personality Emergence** — After many campaigns, NPCs develop emergent personality traits from accumulated memories
- **Campaign Memory Continuity** — Same NPCs remember events across multiple campaign runs (persistent NPC agents)
