# CLAUDE.md

## Project Overview

Maxim is a bio-inspired cognitive architecture for robotic agents. It combines a 5-agent pipeline (Perception, Memory, Exec, Goal, Statistician) with biological memory systems (Hippocampus, ATL, Angular Gyrus, SCN, NAc) and a reactive Default Network.

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
- **Thread model**: Main loop at 2-30Hz + WorkerPool (3 lanes: infer/review/record) + Hippocampus capture thread

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
# Full suite (skip pre-existing circular import failure)
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py

# Specific test file
python -m pytest tests/unit/test_simulation_agent.py -v
```

Known pre-existing failure: `tests/integration/test_memory_hub.py::TestPlanningBridge::test_record_plan_outcome` (circular import in NAc, not related to recent changes).

## Simulation Reports

Every sim run saves to `data/sim_reports/{session_id}/`:
- `report.json` -- Metrics + LLM analysis
- `actions.jsonl` -- Action records
- `aut_hippocampus.json` -- AUT memories
- `aut_nac.json` -- AUT causal links
