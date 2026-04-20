# 0.7 Imagination Wiring — Reproduction Protocol

**Experiment:** `07_imagination_wiring.md`
**Date:** 2026-04-20
**Version:** 0.7.0

## Prerequisites

- Python 3.10+
- `pymaxim` installed with `semantic` extra (`pip install pymaxim[semantic]`)
- For live sim PoC: LLM available (local or cloud)

## Step 1: Integration tests (no LLM required)

```bash
# Run the 17 imagination wiring integration tests
PYTHONPATH=src python -m pytest tests/integration/test_imagination_wiring.py -v

# Run the 63 imagination unit tests
PYTHONPATH=src python -m pytest tests/unit/test_imagination.py -v

# Run the 40 component index tests
PYTHONPATH=src python -m pytest tests/unit/test_component_index.py -v
```

Expected: all pass. Total: 120 tests.

## Step 2: Construction path verification

Verify the orchestrator constructs ImaginationTrigger correctly:

```bash
# Run sim embodiment integration tests (verifies entity_ref → executor path)
PYTHONPATH=src python -m pytest tests/integration/test_sim_embodiment.py -v
```

## Step 3: Live simulation PoC (requires LLM)

```bash
# Option A: Cloud LLM (fastest, ~$0.05-0.15)
MAXIM_LOG_FILE=/tmp/maxim_07_poc.jsonl \
MAXIM_BACKEND_TRACE=1 \
ANTHROPIC_API_KEY=<your-key> \
maxim --sim "explore a dungeon" \
  --embodiment bodies/base_humanoid \
  --auto-curate \
  --interactive false \
  --sim-max-turns 10

# Option B: Local LLM (no cost)
MAXIM_LOG_FILE=/tmp/maxim_07_poc.jsonl \
MAXIM_BACKEND_TRACE=1 \
maxim --sim "explore a dungeon" \
  --embodiment bodies/base_humanoid \
  --auto-curate \
  --interactive false \
  --sim-max-turns 10 \
  --language-model mistral-7b
```

## Step 4: Verify JSONL trace

```bash
# Check for imagination trigger activity
grep "ImaginationTrigger" /tmp/maxim_07_poc.jsonl | head -5

# Check for entity extraction
grep "phrases_extracted\|imagination" /tmp/maxim_07_poc.jsonl | head -10

# Check for ComponentIndex usage
grep "ComponentIndex\|component_index" /tmp/maxim_07_poc.jsonl | head -5

# Check for ephemeral registration
grep "register_ephemeral\|ephemeral" /tmp/maxim_07_poc.jsonl | head -5

# Check for session cleanup
grep "decay_imagined\|clear_ephemeral\|Imagination:" /tmp/maxim_07_poc.jsonl | head -5
```

## Step 5: Full test suite (regression check)

```bash
PYTHONPATH=src python -m pytest tests/ -x -q -m "not slow" \
  --ignore=tests/integration/test_memory_hub.py

# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Type check public API
mypy src/maxim/__init__.py src/maxim/api.py src/maxim/session.py \
  src/maxim/create.py src/maxim/load.py --ignore-missing-imports
```

## Verification checklist

- [ ] Entity extraction identifies SEM-relevant noun phrases
- [ ] ComponentIndex two-layer lookup (alias + embedding) resolves known entities
- [ ] DN arousal gate blocks imagination during high-arousal states
- [ ] Cache deduplicates repeated phrases
- [ ] Thread safety under concurrent access
- [ ] ImaginationTrigger only constructed when entity_ref is set
- [ ] Session cleanup: NAc imagined link decay + ephemeral entity clearing
- [ ] Episode.imagined provenance field defaults to False
- [ ] CausalLink.imagined provenance field defaults to False
- [ ] No regressions in full test suite
