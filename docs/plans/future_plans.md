# Future Plans

Master roadmap for Maxim development.

**Last updated:** 2026-04-08

---

## Priorities

### 1. Hippocampus AUT Memory Refinement

**Status:** Not started
**Effort:** ~300-500 LOC
**Why now:** DM campaigns exposed that recall precision is the weakest link. The AUT repeats early memories and fails to recall cross-encounter details.

Improve how the AUT queries its own memories:
- Semantic filtering and relevance ranking for `memory_recall` tool
- Modality-aware recall ("what did I hear?" vs "what did I see?") using SensoryTag metadata
- `decision_rationale` search (why was this action chosen?)
- Reduce observation capture spam further
- Improve recall precision for campaign-specific queries

### 2. DM Extensions: Generative Architect Persona

**Status:** Not started. [Plan](dungeon_master_extensions.md)
**Effort:** ~500 LOC
**Why now:** Hand-authoring campaign YAMLs is the primary friction. The `--dm` boolean flag is reserved.

An architect persona interviews the user via `ask_user` tool and generates campaign YAML:
- `maxim --sim "run a heist adventure" --dm` → architect generates campaign → DM runs it
- Character creation sub-flow (PC + NPCs)
- Composes from SEM component templates
- Ship gate: architect produces a runnable campaign in < 8 minutes

### 3. PyPI Publication (Phases 3-6)

**Status:** Phase 0-2 done. [Plan](pypi_publication_plan.md)
**Effort:** ~200 LOC + config
**Why now:** Package is ready (`pymaxim`), API shipped, deps clean. Gets the project into the ecosystem.

Remaining: multi-robot plugins (Ph3), CI/CD (Ph4), README rewrite (Ph5), Test PyPI (Ph6).

---

## Future Work (ship when demand surfaces)

| Work | Trigger | Plan |
|------|---------|------|
| **SEM Component Database** | Campaign authoring becomes repetitive | Designed in [DM plan](../archive/dungeon_master_persona.md) |
| **Embodiment Hardware Adapter** | Deploying to physical hardware (Reachy Mini) | ~300 LOC, wraps SDKs as SEM backends |
| **Capability Agent** | Multi-machine setups need runtime awareness | [Design notes](doctor_upgrade_plan.md) |
| **DM: Encounter Library** | Hand-authored encounters become repetitive | [Extensions plan](dungeon_master_extensions.md) |
| **DM: Adaptive Difficulty** | Campaigns feel too easy/hard | [Extensions plan](dungeon_master_extensions.md) |
| **Multi-AUT Party Mode** | Agent Mesh P2+ ships | Requires mesh network transport + DM |
| **Agent Mesh Phase 0a-0b** | Multiple LAN machines join | mDNS discovery + InferenceRouter |
| **Benchmark Phases 7-9** | Paper generation or narrative transcription needed | [Benchmark plan](../archive/benchmark_plan.md) |
| **Peer Inference Retry** | Leader restarts cause 502 errors | ~30 LOC in openai_backend.py |
| **GitHub Fork Workflow** | Contributors need fork-based PRs | [Plan](github_repo_management_plan.md) |

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
├── future_plans.md              # This file — master roadmap
├── dungeon_master_extensions.md # DM follow-ons (architect, library, adaptation)
├── pypi_publication_plan.md     # PyPI publication (Ph3-6 remaining)
├── doctor_upgrade_plan.md       # Doctor expansions + Capability Agent design
├── github_repo_management_plan.md # Fork-based workflow
└── tool_refinement_plan.md      # Living document — tool additions/deprecations
```

## Research Directions (Not Scheduled)

- **ATL Self-Extension** — LLM discovers new concept categories
- **Federated Embodiments** — Multiple agents share memory across bodies
- **Cross-Agent Affordance Delegation** — Sovereign delegation between mesh peers
- **Uncertainty-as-Pain** — Map prediction uncertainty to PainDetector
- **Curriculum Embodiment Learning** — Graduate agents through progressively complex bodies
