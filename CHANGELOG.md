# Changelog

All notable changes to pymaxim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
