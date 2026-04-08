# Future Plans

Master roadmap for Maxim development.

**Last updated:** 2026-04-08

---

## Current Focus: Foundational Buildout

All pre-publication work is tracked in [foundational_buildout_plan.md](foundational_buildout_plan.md).

**Summary:** Ship package hygiene, architectural foundations (multi-agent runtime, SEM component registry, encounter library, party DM mode), expanded public API, and publication prep. ~4,000 LOC across 12 phases.

| Phase | Work | Status |
|-------|------|--------|
| 0 | Package Hygiene (data paths, imports, globals, file handles) | Not started |
| 1 | SEM Component Registry | Not started |
| 2 | DM Encounter Library | Not started |
| 3 | Agent Factory + Agent Pool | Not started |
| 4 | Party DM Runtime | Not started |
| 5 | Hippocampus Recall Refinement | Not started |
| 6 | ask_user Tool | Not started |
| 7 | Generative Architect Persona | Not started |
| 8 | API Surface Expansion (campaign, benchmark, research, events, tools) | Not started |
| 9 | PyPI Deps + User Docs + Examples | Not started |
| 10 | Publication Prep (CHANGELOG, CONTRIBUTING, SECURITY) | Not started |
| 11 | Test PyPI + Publish | Not started |

---

## Post-Publication Work (ship when demand surfaces)

These are features, not architecture. Safe to add after PyPI publication without breaking the public API.

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
| **Agent Mesh Phase 0a-0b** | Multiple LAN machines join | ~400 LOC | mDNS discovery + InferenceRouter. Current `LocalMessageBus` is sufficient for single-machine multi-agent. |
| **Capability Agent** | Multi-machine setups need runtime awareness | ~500 LOC | [Design notes](doctor_upgrade_plan.md). Depends on lane tiers (done) + mesh Phase 0a. |
| **Embodiment Hardware Adapter + selfy.py decomposition** | Deploying to physical hardware or adding new robots | ~800 LOC net (saves ~900) | Decompose `conscience/selfy.py` (5,189 LOC monolith) into `ReachyController(RobotController)` plugin. Moves `AgenticRuntimeMixin` (~1,080 LOC) into standard runtime, eliminates ~650 LOC of orchestrator glue, moves ~276 LOC of generic input handling to interactive module. Enables multi-robot support via entry-point plugins (Atlas, Spot, etc.) without modifying core runtime. Currently behind lazy import — no PyPI impact, but blocks clean robot extensibility. |
| **PyPI Multi-Robot Plugins** | External robot controllers need discovery | ~250 LOC | Entry-point based `maxim.robots` registration. Phase 3 of [PyPI plan](pypi_publication_plan.md). Depends on selfy.py decomposition above. |
| **Full CI/CD Pipeline** | Need automated test + publish | ~2 files | GitHub Actions: lint, test, build, publish. Phase 4 of [PyPI plan](pypi_publication_plan.md). |
| **Peer Inference Retry** | Leader restarts cause 502 errors | ~30 LOC | Exponential backoff in openai_backend.py |
| **GitHub Fork Workflow** | Contributors need fork-based PRs | ~550 LOC | [Plan](github_repo_management_plan.md) |

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
| DM MVP | dm_schema, dm_runtime, ChooseTool, 4 campaigns, expectations checker | [Plan](../archive/dungeon_master_persona.md) |
| Python API | Verb-based interface (run, imagine, connect, diagnose, observe) | [Plan](../archive/python_api_plan.md) |
| Introspection API (Ph1-4) | Observer class, standalone run_campaign() | [Plan](../archive/introspection_api_plan.md) |
| Realtime Refinement | InspectAUTTool, 8 personas, metric expectations | [Plan](../archive/realtime_refinement_plan.md) |

---

## Active Plan Files

```
docs/plans/
├── future_plans.md                 # This file — master roadmap
├── foundational_buildout_plan.md   # Pre-publication buildout (current focus)
├── dungeon_master_extensions.md    # DM follow-ons (Extensions C-G, post-publication)
├── pypi_publication_plan.md        # PyPI publication (reference; phases absorbed into buildout)
├── doctor_upgrade_plan.md          # Doctor expansions + Capability Agent design
├── github_repo_management_plan.md  # Fork-based workflow
└── tool_refinement_plan.md         # Living document — tool additions/deprecations
```

## Research Directions (Not Scheduled)

- **ATL Self-Extension** — LLM discovers new concept categories
- **Federated Embodiments** — Multiple agents share memory across bodies
- **Cross-Agent Affordance Delegation** — Sovereign delegation between mesh peers
- **Uncertainty-as-Pain** — Map prediction uncertainty to PainDetector
- **Curriculum Embodiment Learning** — Graduate agents through progressively complex bodies
- **NPC Personality Emergence** — After many campaigns, NPCs develop emergent personality traits from accumulated memories
- **Campaign Memory Continuity** — Same NPCs remember events across multiple campaign runs (persistent NPC agents)
