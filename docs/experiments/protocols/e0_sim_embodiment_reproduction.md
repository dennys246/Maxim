# E0 Sim Embodiment — Reproduction Protocol

**Plan:** [asset_foundry_plan.md](../../plans/deferred/asset_foundry_plan.md) Stage 0
**PoC results:** [e0_sim_embodiment_poc.md](../e0_sim_embodiment_poc.md)

## Quick verification (~1.5s, no deps beyond core)

```bash
python -m pytest tests/integration/test_sim_embodiment.py -v
```

Expected: 10 tests PASS. Covers:
- Entity loading via ComponentRegistry
- Affordance tool registration (slash, parry, sense tools)
- Pain cascade: tool failure -> negative NAc link, success -> positive NAc link
- Precondition validation (missing nac, missing pain_bus, nonexistent entity)
- Default path (no entity_ref) unchanged

## Live simulation (~30-60s, requires LLM backend)

```bash
# Short generative sim with embodiment (non-interactive for CI/scripts)
maxim --sim "Test the rusty sword at low durability" \
  --embodiment weapons/rusty_sword \
  --interactive false \
  --sim-max-turns 3

# With full trace logging
MAXIM_LOG_FILE=/tmp/e0_poc.jsonl \
MAXIM_BACKEND_TRACE=1 \
  maxim --sim "Test sword combat" \
  --embodiment weapons/rusty_sword \
  --interactive false \
  --sim-max-turns 5
```

Look for in stdout:
- `AUT ComponentRegistry created for entity_ref='weapons/rusty_sword'`
- `LLM raw response parsed: tool=rusty_sword_slash`
- `Causal link: tool:rusty_sword_slash -> positive` or `-> negative`
- `Captured: rusty_sword_slash (salience=...)`

## DM campaign with embodiment

```bash
maxim --sim scenarios/campaigns/heist_v1.yaml \
  --embodiment weapons/rusty_sword \
  --interactive false \
  --sim-max-turns 5
```

## Regression suite

```bash
# Embodiment + bootstrap + cascade tests (~90 tests)
python -m pytest tests/integration/test_sim_embodiment.py \
  tests/integration/test_bootstrap.py \
  tests/substrate/test_sem_execution_production.py \
  tests/unit/test_embodiment_failures.py \
  tests/unit/test_embodiment_sem.py -v

# Full suite (~5200 tests)
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
```

## What to check if tests fail

1. **`test_entity_ref_produces_embodiment` fails:** Check `build_executor` in `runtime/bootstrap.py` — the entity_ref -> ComponentRegistry -> Embodiment path around line 438-463.
2. **`test_tool_failure_produces_nac_link` fails:** Check `ToolPainBridge.record_tool_embodiment_failure` and the `Executor.execute` side_effects plumbing.
3. **`test_nonexistent_entity_ref_raises` fails:** Check that `ComponentNotFoundError` is raised by `ComponentRegistry.instantiate` for bad refs.
4. **Live sim runs but no affordance tools appear:** Check that `cli.py` threads `entity_ref` to `start_simulation_mode` (the `_sim_entity_ref` variable at line ~969).
