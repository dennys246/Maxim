# SEM Learning Loop — Reproduction Protocol

**Plan:** [sem_learning_loop.md](../../plans/archive/sem_learning_loop.md)
**PoC results:** [sem_learning_loop_poc.md](../sem_learning_loop_poc.md)

## Quick verification (~0.5s, no deps beyond core)

```bash
PYTHONPATH=src python scripts/sem_learning_loop_poc.py
```

Expected: `ALL ASSERTIONS PASSED`. Script exercises:
- Pain → negative edge valence
- Pain spike → episode boundary
- Success → positive valence + NAc reward bias
- Clean control → zero valence
- Persistence round-trip

## Unit tests (26 tests, ~0.4s)

```bash
python -m pytest tests/substrate/test_valence_annotation.py -v
```

Covers: reaction capture, valence computation (7 edge cases), edge annotation with decay, Episode/Edge persistence round-trip, spreading activation valence propagation, retrieve_on_cue include_valence.

## Regression suite

```bash
# Substrate tests (P3a + P3b + P4 + valence = ~110 tests)
python -m pytest tests/substrate/ -v

# Full suite (~4977 tests)
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
```

## What to check if tests fail

1. **Episode valence tests fail:** Check `PendingEpisodeState.finalize()` in `memory/episode.py` — valence computation uses `getattr` on reaction objects. If the Reaction type changed fields, the computation silently returns 0.0 (with a log warning).

2. **Edge valence tests fail:** Check `apply_hebbian_on_close` in `memory/episode.py` — valence annotation uses `update_edge(metadata_updates={"valence": ...})` which acquires the graph lock. If `update_edge` semantics changed, metadata may not be written.

3. **Spreading activation valence fails:** Check `DependencyGraph.spreading_activation` in `agents/bus.py` — the `propagate_valence=True` path is a separate code path (`_spreading_activation_with_valence`). If the BFS traversal order changed, valence propagation may differ.

4. **distribute_reward produces no bias:** NAc `credit_node` clamps to `[0, max_reward_bias]`. Negative rewards (pain) correctly produce 0.0 bias — this is by design, not a bug. Only positive rewards produce non-zero bias.

5. **Salience spike boundary doesn't fire:** Check that `salience_spike_rule` is registered in the episode detector AND that `CaptureEvent.salience_spike` is populated (not None) above the threshold.

## Invariants to preserve

- `Episode.valence` defaults to 0.0 on old data (backward compat)
- `spreading_activation(propagate_valence=False)` returns `dict[str, float]` exactly as before
- `retrieve_on_cue(include_valence=False)` returns `list[tuple[str, float]]` exactly as before
- `CaptureEvent.salience_spike` defaults to `None` — existing callers unaffected
- `HebbianConfig.valence_decay` defaults to 0.95 — no behavior change without explicit tuning
