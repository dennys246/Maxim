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

## Key Commands

```bash
# Run with local LLM
maxim --language-model mistral-7b

# Run with Claude (requires ANTHROPIC_API_KEY)
maxim --language-model claude-sonnet

# Simulation agent mode (recommended for testing)
maxim --sim agent --goal "test safety" --persona adversarial --language-model claude-sonnet

# Resume a previous simulation
maxim --sim agent --goal "continue" --resume-sim 20260403 --language-model claude-sonnet

# Run YAML scenario
maxim --sim scenarios/malware_with_pain.yaml

# Run tests
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py
```

## Project Structure

```
src/maxim/
  agents/           # 5-agent pipeline (perception, memory, exec, goal, statistician)
  conscience/       # Main Maxim class (selfy.py) + 6 mixins + agentic runtime
  runtime/          # Agent loop, LoopController, SimulationAdapter, executor, worker pool
  memory/           # Hippocampus, ATL semantic memory, layer protocol
  math/             # IPS (fast stats) + Angular Gyrus (algebraic memory)
  decisions/        # NAc causal learning, adaptive planner
  models/language/  # LLM router, backends (llama-cpp, anthropic, openai, transformers)
  simulation/       # SimulationBridge, orchestrator, tools, personas, report system
  tools/            # 40+ tools (filesystem, introspection, math, communication)
  default_network/  # Reactive behavior layer (thalamic gate, arbiter, behaviors)
  modes/            # Operating modes, strategies, exploration policy
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

docs/               # Internal architecture docs
docs/user/          # User-facing guides
docs/plans/         # Development roadmap
htmls-guides/       # Jinja2 HTML templates for dennyschaedig.com
tests/              # Unit + integration + benchmark tests
scenarios/          # YAML simulation scenarios
data/util/          # Runtime config (llm.json, cost_state.json)
```

## Architecture Essentials

- **Agent loop** lives in `runtime/agent_loop.py` with `LoopController` in `runtime/loop_controller.py`
- **LLM routing** lives in `models/language/router.py` (config in `models/language/config.py`)
- **Simulation** orchestrator in `simulation/orchestrator.py`, bridge in `simulation/bridge.py`
- **Mode system**: ProcessingState (awake/sleep) x OperationalMode (passive/active/singularity) x Strategy (6 types)
- **Memory tiers**: FORMING -> WORKING -> SHORT_TERM -> LONG_TERM
- **Thread model**: Main loop at 2-30Hz + WorkerPool (3 lanes: infer/review/record, owned by LLMWorker) + Hippocampus capture thread (owned + shut down by MemoryHub.on_session_end)

## Quick reference — where to look

| I'm touching... | Start here |
|---|---|
| The agent loop / step pipeline | `runtime/agent_loop.py`, `runtime/loop_controller.py` |
| Adding a new tool | `tools/` + register in the tool registry; see `tools/introspection.py` for a clean example |
| LLM prompts / routing | `models/language/router.py`, `models/language/prompt_formats.py` |
| Sim personas | `simulation/personas.py` (5 today: adversarial, cooperative, confused, escalating, campaign) |
| Memory capture → consolidation | `memory/hippocampus.py`, `memory/concept_extractor.py`, `memory/semantic_promoter.py` |
| Causal learning | `decisions/nac.py` |
| Cross-layer wiring | `integration/memory_hub.py` (the single coordinator) |
| Adding an env var | Put it here in the env table + touch whatever reads it |
| Atomic JSON persistence | `utils/atomic_io.py` |

## Environment Variables

```bash
ANTHROPIC_API_KEY          # Required for Claude backend
OPENAI_API_KEY             # Required for OpenAI backend
MAXIM_LLM_ENABLED=1        # Enable LLM inference
MAXIM_LLM_PROFILE=claude-sonnet  # Default model profile
MAXIM_PROVENANCE_VERBOSITY=1     # 0=off, 1=compact, 2=verbose
```

## Testing

```bash
# Full suite (one pre-existing logic failure in test_record_plan_outcome)
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py

# Specific test file
python -m pytest tests/unit/test_simulation_agent.py -v
```

Known pre-existing failure: `tests/integration/test_memory_hub.py::TestPlanningBridge::test_record_plan_outcome` — `record_plan_outcome` doesn't currently drive NAc's observation counter (the assertion `nac.stats()["total_observations"] > 0` fails). Not a blocker for the rest of the suite. (The NAc circular import that previously masked this was fixed in Wave A stabilization.)

## Simulation Reports

Every sim run saves to `data/sim_reports/{session_id}/`:
- `report.json` -- Metrics + LLM analysis
- `actions.jsonl` -- Action records
- `aut_hippocampus.json` -- AUT memories
- `aut_nac.json` -- AUT causal links

## Active initiatives

See `docs/plans/future_plans.md` for the full roadmap. Current state:

- **Multi-LLM scaling** (`docs/plans/multi_llm_scaling.md`) — all prereqs done (router modularization, WorkerPool lanes, RuntimeCapabilities). Phase 1 ready to start.
- **Research Protocol** (`docs/plans/research_protocol_plan.md`) — not started, self-contained. Builds mesh primitives reused by agent-mesh later.
- **Realtime Refinement** (`docs/plans/realtime_refinement_plan.md`) — ~90% done; remaining ~50 LOC (6th persona + metric expectation types).
- **Agent Mesh** (`docs/plans/agent_mesh.md`) — blocked on Multi-LLM Phase 7 + Research Protocol Phase 0.
- **Docker Sandbox** (`docs/plans/docker_sandbox_plan.md`) — Phase A (TmpdirSandbox + pain triggers) + Phase B (DockerSandbox + ContainerRunner protocol + image catalog + autonomy-scaled resource limits + unprivileged `maxim` user) DONE.
