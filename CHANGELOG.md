# Changelog

All notable changes to pymaxim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note (2026-05-11):** versions 0.6.0 → 0.8.1 shipped to PyPI without
> matching CHANGELOG entries. The summaries for that window live in
> [docs/plans/v1_refinement.md](docs/plans/v1_refinement.md) and the
> [HTML roadmap](html-guides/maxim-roadmap.html). The 0.9.0 entry below
> picks up the convention again; backfilling 0.6/0.7/0.8 from the
> roadmap is tracked as documentation debt and can land in a follow-up.

## [0.9.0] - 2026-05-11

### Added

- **Roy three-arm iteration runner (R1–R5).** Long-horizon persona-convergence harness. `maxim roy run <spec.yaml>` primes substrate via a multi-stage curriculum, runs the same held-out test across three arms (substrate-primed neutral / blank persona-injected / blank neutral), and reports pairwise substrate divergence (`reward_bias L2`, `cluster_reward_bias L2`, hippocampus episode + valence deltas, ATL concept Jaccard). Sister subcommands `maxim roy diff <a> <b>` and `maxim roy log <iter>` provide ad-hoc diffs and idempotent protocol/iteration-log regeneration. New modules: `src/maxim/simulation/roy_runner.py`, `src/maxim/simulation/curriculum_runner.py`, `src/maxim/analysis/substrate_diff.py`, `src/maxim/analysis/roy_log.py`, `src/maxim/roy/cli.py`. Methodology: [docs/plans/persona_convergence_crucible.md](docs/plans/persona_convergence_crucible.md).
- **G3 — fail-fast LLM pre-flight probe** in `run_roy_iteration` (PRs #235, #238). Roy iterations chain ~5 sims back-to-back; if the configured `large` lane is unreachable the iteration grinds out ~10 min of static-fallback narration with `dispatch_exhausted` on every call. Probe resolves `MAXIM_LANE_LARGE_REMOTE_URL` env var first, then `~/.config/maxim/peer.yml`, then skips for local/cloud setups. One HTTP call only (Plan 3 R2.5 invariant); aborts with `result.aborted_at = "preflight"` + persisted `result.json` + `summary.md` in ≤3.3s when the leader is dead. 10 regression tests.
- **G4 — substrate-primary cluster_id reward-feedback wire** (PRs #236, #237). Closes the wire Track 2 (commit `6d0e4a7`) deliberately deferred: `LLMProposal.cluster_id` field carries the active EC interoception cluster from `propose_via_substrate` to outcome recording; `record_outcome(..., cluster_id=...)` calls `NAc.update_cluster_reward(agent_id, cluster_id, sig, ±1.0)`; all 6 `_record_outcome` call sites + `execute_parallel_actions` thread it through; `NAc.dump`/`load_state` serialise `_cluster_reward_bias` under a new JSON key; `substrate_diff.NacDiff` surfaces `cluster_reward_bias_{available,l2,top_deltas}` so Roy `result.json` carries the headline metric. Empirically validated: live Roy-0 re-run produced `cluster_reward_bias_l2 = 2.4587` on A-vs-blank pairs (≈11.6× blank-vs-blank noise floor). 6 regression tests.
- **Roy summary rendering of `cluster_reward_bias`.** `roy_runner._format_summary` now emits `NAc cluster_reward_bias L2=… (N keys differ)` on its own line so `summary.md` carries the headline metric, not just the (necessarily 0) `reward_bias_l2`.
- **CLI documentation refresh** (PR #239). `docs/user/cli-reference.md` gains `--aut-mode` flag entry plus a full "Roy Harness" section documenting `run`/`diff`/`log` subcommands, spec shape, preflight semantics, and examples. `docs/reference.md` adds `src/maxim/analysis/` and `src/maxim/roy/` module entries. `docs/index.md` adds a Roy Harness link to the Architecture & Modes table.
- **HTML guide refresh** (PR #239). `html-guides/maxim-overview.html` nav now links the 10 previously-orphaned guides (semantic-memory, component-library, deliberation, concept-decomposition, tools, prompt-system, agent-mesh, dm-campaigns, experiments, benchmarks) in a three-row layout. `html-guides/maxim-roadmap.html` "Path to 1.0" ASCII updated through v0.8.0 + Roy + G3/G4 with empirical numbers. `html-guides/maxim-substrate-primary.html` gains a new "Roy harness — how we measure substrate convergence" section with the three-arm methodology table and the Roy-0 empirical results.

### Backward compatibility

- **`LLMProposal.cluster_id: str | None = None`** is an optional dataclass field added at the end of the frozen `LLMProposal` — CC3-compatible non-breaking. LLM-primary proposals leave it `None`.
- **`aut_nac.json::cluster_reward_bias`** is a new optional JSON key. Pre-G4 snapshots (no field) load to an empty dict; the loader emits no warning. Field key format joins `(agent_id, cluster_id, tool_signature)` with `\x1f` (ASCII unit separator) so tool signatures containing `:` round-trip cleanly.
- **`record_outcome(..., cluster_id=None)`** is a no-op for the cluster-update path — the LLM-primary tool-outcome path stays bit-identical.
- **No breaking changes to public API surface.** No deprecation warnings added in this release; C4/C5/C6 deprecation cycle (per `docs/plans/v1_refinement.md`) remains scoped to 1.0.

### Why bump minor

This is a feature release (Roy harness + substrate-primary reward feedback) that's strictly additive to a working v0.8.x install. Per semver and the project's "any change that affects runtime behavior, CLI interface, or peer/leader protocol" guidance, this earns a minor bump rather than a patch.

## [0.5.0] - 2026-04-19

### Added

- **B4 Replanning — all 3 stages shipped (1.0 gate closed).** Failure diagnosis with prior-attempt retrieval via hippocampus episodes, Jaccard distance metric for structural novelty, anti-repetition prompt constraint. Blind A/B validation: treatment (replanning) 100% vs control (no replanning) 0%, mean Jaccard 0.894. 48 tests across 3 test files.
- **P6 Extinction.** `DependencyGraph.decay_edges()` — multiplicative Hebbian decay with pruning. Beats LRU baseline across 10 seeds. 9 tests.
- **P8 Sleep Replay.** `memory/sleep_replay.py` — offline memory consolidation. Episode ranking by NAc reward_bias + valence. Replay re-fires `apply_hebbian_on_close` with consolidation multiplier. F1 improves vs no-replay control across 10 seeds. 13 tests.
- **F2 AgentFactory CLI migration.** `AgentFactory.create_full_agent()` composes `build_bio_stack` + `build_executor` + `FearGatedExecutor`. CLI non-sim bootstrap (~100 lines) replaced with one factory call. `AgentConfig` extended with `with_bio_stack`, `with_executor`, `with_pain_bridge`, `with_fear_gate`, `embodiment_ref`. `AgentInstance` extended with `bio_stack`, `pain_bus`, `embodiment`. 10 new tests.
- **`planning/structural_diff.py`** — Jaccard distance on action sequences for plan comparison. Pure utility, no agent/memory/runtime imports.
- **`AgentInstance.shutdown()` saves cerebellum** — learned forward models no longer lost on session end.
- **Experiment results:** `b4_replanning_results.md`, `p6_extinction_results.md`, `p8_sleep_replay_results.md`.

### Fixed

- `executor.embodiment` attribute lookup was using `_embodiment` (wrong name) — always returned None. Fixed in both factory and CLI.
- Sim path was building a second PainBus on the same hippocampus/nac, causing double-subscription of learning callbacks. Fixed to reuse bio-stack's bus.
- Bio-stack construction failure now propagates instead of silently degrading to a partial agent.

## [0.4.0] - 2026-04-19

### Added

- **Input standardization.** Unified input handling across all simulation modes (generative, DM, interactive, fixture). `PerceptSource` protocol with 4 implementations.
- **DM interactive mode.** Free-text roleplay between choices. Campaign runs on thread so stdin reader accepts input.
- **Rich menu system.** `maxim` (no args) launches interactive menu with campaigns, chat, doctor, help.
- **NAc suppression in interactive mode.** Tool-outcome learning gated on `get_interactive_mode()` to prevent human-directed actions from corrupting causal models.
- **Scale validation.** 20/20 seeds, p = 3.87e-6. Cross-session learning is not a fluke.

### Fixed

- Display/interactive globals now reset between menu sims (`reset_sim_display_state()`).
- DM campaign thread ordering: campaign runs AFTER stdin reader starts.
- Stall detector disabled in interactive mode (nudge prompts contained adversarial probes).

## [0.3.2] - 2026-04-18

### Added

- **Bidirectional interactive mode.** Raw terminal input with in-panel rendering, `request_interaction` agent-to-user prompting, `set_scene` dynamic scene header, `/pause` `/resume` `/display` commands, scrollable log with bio trace dimming, end-of-sim review prompt.

### Fixed

- Display corruption from `print()` calls during Rich Live panels.
- Stdin contention between display thread and input reader.
- Tool schema validation for JSON schema vs flat tool formats.
- LLM prompt context truncation for long conversations.

## [0.3.1] - 2026-04-18

### Added

- **Interactive UX fixes.** `RequestInteractionTool` honest reporting, narrator fallback immersion, handler logging, story context truncation, `MaximDisplay` → `sim_logger` wiring, prompt cleanup.
- **4 introspection tools.** `nac_stats`, `memory_pressure`, `loop_stats`, `pain_triggers_active` — agent can reason about its own learning state.

## [0.3.0] - 2026-04-17

### Added

- **Cross-session learning without fine-tuning — demonstrated across all 3 tiers.** 41/41 hypotheses confirmed across 4 experiments.
- **SEM Learning Loop (5 stages).** Cerebellum activation in BioStack, distribute_reward wiring, success reactions, pain spike episode boundary.
- **Valence Annotation (Stages 1-3).** Episode.valence, Edge.metadata["valence"], spreading_activation(propagate_valence), retrieve_on_cue(include_valence).
- **Behavioral Convergence Wiring (4 stages).** Valence in PromptAssembler, observe_episode_event in agent loop, energy→Reaction bridge, food/water/poison SEM specs.
- **Bio-Stack Unification (Waves 0-3).** `build_bio_stack`, `build_pain_bus`, `build_memory_hub`, `build_default_network`, `build_executor` — all canonical construction sites with structural enforcement.
- **Substrate P0-P4 complete.** Recognition, reward modulation, episode binding, channel integration, persistence/snapshot, cross-modal binding — all shipped.
- **LLM Path Refinement (Plans 1-4).** Typed errors, fast failover, `_MaximPeerBackend`, reactive peer mesh with auto-drain.

## [0.2.1] - 2026-04-10

### Changed

- **Re-publish of 0.2.0 contents.** No functional changes from the 0.2.0 draft. The 0.2.0 version slot on PyPI was burned by an earlier upload+delete cycle (PyPI version numbers are immutable even after deletion), so this patch bump is the smallest version that could be published.

## [0.2.0] - 2026-04-10 — Research Preview

**Versioning note:** This release was originally drafted as 1.0.0 and the entries below describe that work. After a deep architectural review on 2026-04-10, the 1.0 label was pulled and reissued as a 0.2.0 research preview. The reasoning: the bio-inspired stack is currently half-earned (NAc and Cerebellum implement genuine analogs of their brain namesakes; ATL, Angular Gyrus, SCN, and Default Network use the vocabulary without the cross-region mechanisms), and shipping 1.0 would lock in stability promises before the percept-substrate refactor that closes that gap. The 1.0 label is now reserved for the version that demonstrably improves on a task across sessions without fine-tuning the underlying LLM. See [docs/plans/archive/substrate_plan.md](docs/plans/archive/substrate_plan.md) for the original 0.3 → 0.4 → 0.5 → 1.0 narrative (the plan has since been split into focused sub-plans under [docs/plans/](docs/plans/README.md)).

The work documented below is real and shipping in 0.2.0 — only the label changed.

## [Original 1.0.0 draft — shipped as 0.2.0] - 2026-04-09

### Added

- **Expanded SEM component library** — 54 seed components across 7 categories (bodies, creatures, environments, items, npcs, vehicles, weapons). Covers fantasy, cyberpunk, sci-fi, horror, historical, modern, and devops genres.
- **SEM entity wiring in DM campaigns** — campaigns now instantiate live SEM entities at startup. NPCs and world objects have real sensors, affordances, and failure modes. Registry refs (`ref: npcs/guard`) with optional overrides.
- **Event subscription system** — `maxim.on("tool_call", callback)` with typed payloads (`ToolCallEvent`, `PainSignalEvent`, `MemoryCaptureEvent`, `PromptEvent`). Bridged to internal AgentBus.
- **Custom tool registration** — `@maxim.tool` decorator and `maxim.register_tool()` now wired into runtime. Tools injected into all `run()`/`imagine()`/`campaign()` calls.
- **4 new DM campaigns** — Wizard's Tower (fantasy), Server Breach (devops), Haunted Manor (horror), Space Station Crisis (sci-fi).
- **Context manager protocol** — `AgentPool` and `Session` support `with` statements for automatic cleanup.
- **Return type annotations** — all `maxim.create.*` and `maxim.load.*` functions return typed objects instead of `Any`.
- **`ComponentNotFoundError`** — custom exception for missing SEM templates (was generic `KeyError`).
- **Exception renames** — `MaximConnectionError`, `MaximMemoryError`, `MaximRuntimeError` no longer shadow Python builtins. Old aliases removed.

### Changed

- **CLI restructured** — 70+ flags organized into 11 argparse groups (core, cloud, autonomy, memory, hardware, agentic, exploration, simulation, debug, benchmark, utilities). `--internet-access`/`--no-internet` now mutually exclusive.
- **`configure()` validates inputs** — warns on out-of-range verbosity, unknown show channels, unknown debug subsystems.
- **`model=` parameter now overrides env vars** — `setdefault` replaced with direct assignment in `run()`/`imagine()`/`campaign()`.
- **Observe dispatch deduplicated** — shared `query_observer()` between `Session.observe()` and `maxim.observe()`.
- **Atomic writes** — model persistence and markdown report saves now use tmp+fsync+replace.
- **Deferred numpy/scipy imports** — `response_output.py` no longer eagerly loads scipy on `import maxim.utils`.
- **`load.nac()`/`load.atl()` standardized** — both now check file existence before loading (was inconsistent).

### Fixed

- `maxim.on()` and `maxim.register_tool()` were no-ops — callbacks/tools never reached the runtime.
- `@maxim.tool` decorator missing thread lock on `_pending_tools` append.
- `_inject_pending_tools()` didn't clear the list — tools double-registered on subsequent calls.
- Silent `except Exception: pass` in `session.research()` — now logs warning.
- `--sim-report` silently ignored without `--sim` — now validated.
- All `llm-local` references updated to `llm-llama` (7 locations across docs and source).
- Stale CLI defaults in docs (wrong `--mode` default, wrong `--persona` default, non-existent `--prompt-profile`).
- Deprecated `--sim agent` syntax updated to `--sim "goal"` across all active docs.
- Broken `</span>` tags in maxim-operating-modes.html.

## [0.2.0] - 2026-04-08

### Added

- **SEM Component Registry** — reusable entity templates with inheritance and deep merge. Components stored as YAML specs in `~/.maxim/components/` (user) and bundled in the package. 9 seed components across NPCs, weapons, creatures, and environments.
- **DM Encounter Library** — reusable scene + choice templates for campaigns. 8 seed encounters across combat, social, exploration, and puzzle categories. Campaigns reference templates via `template:` key with campaign-specific wiring.
- **Agent Factory + Pool** — multi-agent runtime infrastructure. Independent agents with isolated Hippocampus, NAc, ATL, MemoryHub, and ToolRegistry. Concurrent execution via ThreadPoolExecutor. Thread-safe ToolRegistry.
- **Party DM Runtime** — NPC agents with real memory and learning in DM campaigns. NPCs react to scenes, their dialogue is folded into PC's stimulus, and outcomes are broadcast for hippocampus capture. Per-NPC memory export.
- **Hippocampus recall improvements** — relevance ranking via keyword overlap, observation dedup window (30s default), lightweight `store_observation()` for NPC agents.
- **Interactive Runtime** — universal prompt protocol (8 prompt types, 5 handlers) + Rich terminal display with structured panels. DM campaign extension with character sheet, inventory, encounter info, NPC relationships, and user notes.
- **Generative Architect** — LLM-driven campaign creation with Entity Designer (generates valid SEM specs from natural language), character templates (5 PC archetypes, 6 NPC roles), and architect tools (browse components/encounters, design entities, emit campaigns).
- **Expanded Python API** — 7 new verbs: `campaign()`, `benchmark()`, `research()`, `on()`, `register_tool()`, `register_persona()`, `@tool` decorator. Structured result types: `CampaignResult`, `BenchmarkResult`, `ResearchResult`, `EventHandle`.
- **Split persistence protocols** — `EpisodicStore`, `CausalStore`, `SemanticStore` protocol classes with `File*Store` defaults. Foundation for Mother Maxim's database backend.
- **Cloud provider profiles** — 10 new builtin LLM profiles: Gemini (2), Groq (2), Together (1), Fireworks (1), Mistral (2), DeepSeek (2). Zero new backend code — all use existing OpenAI-compatible endpoint.
- **`metadata: dict` field** on `EpisodicMemory` for extensible per-memory metadata (domain tags, contribution source, tenant ID).
- **Package infrastructure** — `py.typed` (PEP 561), `__main__.py` (`python -m maxim`), bundled data in `src/maxim/_data/`.

### Changed

- **Data paths** — all user data now writes to `~/.maxim/` (override via `$MAXIM_DATA_HOME`). Bundled defaults ship in `src/maxim/_data/`. 28+ source files migrated from CWD-relative `data/` paths.
- **GPU detection** — moved from import-time subprocess to lazy `gpu_detect.py` called from `cli.main()`. `import maxim` no longer has side effects.
- **Import hygiene** — `selfy.py` import-time side effects (`mp.set_start_method`, `PYOPENGL_PLATFORM`, GPU detection) moved to lazy `_setup_hardware_env()`.
- **Persistence safety** — 13 hand-rolled JSON persistence patterns replaced with `atomic_write_json()`.
- **Thread safety** — locks added to `ToolRegistry`, `narrative_transcriber._class_registry`, `lane_backends._active_routers`, `gpu_detect._detected`.
- **Version constraints relaxed** — `requires-python >=3.10` (was 3.12), `numpy >=1.26` (was 2.2), `scipy >=1.11` (was 1.15).
- **`rich` moved to core dependency** (was optional `[ui]` extra).

### Fixed

- Unclosed file handles in `provenance/store.py` and `sim_logger.py` (added `__del__`/atexit cleanup).
- `print()` in library code (`mesh_trace.py`) replaced with `logger.warning()`.
- Duplicate `llm-local` / `llm-llama` dependency removed.
- Pre-existing persona count test updated for `dungeon_master` + `adventure_architect`.

## [0.1.0] - 2026-04-06

### Added

- Initial release with bio-inspired cognitive architecture.
- 5-agent pipeline (Perception, Memory, Exec, Goal, Statistician).
- Biological memory systems (Hippocampus, ATL, NAc, SCN, Angular Gyrus).
- SEM protocol for embodiment (Entity, Sensor, Modulator).
- DM campaign system with 4 hand-authored campaigns.
- Simulation benchmarking with multi-model comparison.
- Verb-based Python API (run, imagine, connect, diagnose, observe, configure).
- Remote peer management (update, restart, LLM hot-swap).
- Doctor diagnostics with platform detection.
