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
| 1 | SEM Component Registry | **In progress** |
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

## Research Directions: Other (Not Scheduled)

- **ATL Self-Extension** — LLM discovers new concept categories
- **Federated Embodiments** — Multiple agents share memory across bodies
- **Cross-Agent Affordance Delegation** — Sovereign delegation between mesh peers
- **Uncertainty-as-Pain** — Map prediction uncertainty to PainDetector
- **Curriculum Embodiment Learning** — Graduate agents through progressively complex bodies
- **NPC Personality Emergence** — After many campaigns, NPCs develop emergent personality traits from accumulated memories
- **Campaign Memory Continuity** — Same NPCs remember events across multiple campaign runs (persistent NPC agents)
