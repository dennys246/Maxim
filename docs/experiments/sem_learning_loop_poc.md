# SEM Learning Loop PoC — Full Pain/Success → Learning → Retrieval Cycle

**Date:** 2026-04-17
**Status:** PASS
**Plan:** [sem_learning_loop.md](../plans/archive/sem_learning_loop.md) Stage 5

## What this proves

All four stages of the SEM learning loop fire end-to-end:
1. **Cerebellum activation** — forward model prediction/training infrastructure wired into BioStack
2. **distribute_reward** — reactions → NAc reward bias → EC threshold adjustment
3. **Positive valence** — `_emit_success_reaction` on confident Cerebellum predictions
4. **Pain spike boundary** — salience_spike_rule closes episode on high-intensity pain

## Scenarios

### Scenario A — Pain loop

Agent encounters `text_rusty_sword`, `text_heavy`, `text_sharp`. CerebellumModulator failure simulated → pain reaction (intensity 0.8, NEGATIVE).

| Measurement | Value | Expected |
|---|---|---|
| Episode valence | **-0.800** | Negative |
| Edge valence (sword→heavy) | **-0.800** | Negative |
| NAc bias from pain | **None (0.0)** | Correct — bias clamps to [0, max], pain can't narrow |
| EC threshold overrides | **{}** | Correct — no widening from pain |

**Design insight:** NAc reward bias only *widens* EC recognition (makes concepts easier to recognize), never narrows. Pain → avoidance is handled by valence annotation on Hebbian edges. This is a deliberate asymmetry: pain makes you *remember* bad things (valence), but doesn't make you *fail to recognize* them (bias). Both pathways are biologically motivated.

### Scenario A2 — Pain spike episode boundary

Pain spike (intensity 0.7, above 0.5 threshold) fired between two events.

| Measurement | Value | Expected |
|---|---|---|
| Episodes after spike | **2** | Boundary triggered |

### Scenario B — Success loop

Positive reaction (intensity 0.3, POSITIVE) simulating confident Cerebellum prediction.

| Measurement | Value | Expected |
|---|---|---|
| Episode valence | **+0.300** | Positive |
| Edge valence after success | **-0.460** | Less negative (decay + positive offset) |
| NAc has positive bias | **True** | Success widens EC recognition |
| NAc biases | **0.015 per node** | Proportional to eligibility |

**Key finding:** After pain (-0.800) then success (+0.300), the net edge valence is -0.460 (= -0.800 * 0.95 + 0.300). The agent still associates the sword with pain, but the success interaction is beginning to offset it. With more successful interactions, valence would approach zero and eventually go positive.

### Scenario C — Clean control

Fresh hippocampus, same sword concepts, no reactions.

| Measurement | Value | Expected |
|---|---|---|
| All zero valence | **True** | Confirmed |

### Persistence

Dump/load round-trip preserves edge valence: **True**.

## The full loop (what fires in production)

```
SEM Entity interaction (e.g., swing rusty sword)
    │
    ▼
CerebellumModulator.execute()
    ├─ confident → cached prediction → _emit_success_reaction (POSITIVE)
    └─ not confident → LLM fallback → train Cerebellum
         └─ fails → _emit_failure_reaction (NEGATIVE)
    │
    ▼
ReactionBus.publish(reaction)
    ├─ hippocampus.capture_reaction → PendingEpisodeState.reactions
    └─ _distribute_reward_from_reaction → NAc.distribute_reward
         └─ credit_node → _reward_bias (positive only, clamps at 0)
    │
    ▼
Episode close (finalize)
    ├─ net valence computed from reactions
    ├─ apply_hebbian_on_close → Edge.metadata["valence"]
    └─ EC.pattern_complete(threshold adjusted by NAc bias)
    │
    ▼
Pain spike → salience_spike_rule → episode boundary
    │
    ▼
Future retrieval
    ├─ spreading_activation(propagate_valence=True) → affective memory
    └─ EC threshold_override from NAc → adjusted recognition radius
```

## Reproduction

```bash
# Quick (deterministic, no LLM, ~0.5s)
PYTHONPATH=src python scripts/sem_learning_loop_poc.py

# Full test suite (26 valence tests + 4977 total)
python -m pytest tests/substrate/test_valence_annotation.py -v
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
```

## Files changed

| File | What |
|---|---|
| `memory/episode.py` | Episode.valence, PendingEpisodeState.reactions, salience_spike on CaptureEvent, salience_spike_rule, valence annotation in apply_hebbian_on_close |
| `memory/hippocampus.py` | capture_reaction, retrieve_on_cue(include_valence=True), HebbianConfig.valence_decay |
| `memory/hippocampus_persistence.py` | Thread valence_decay through rebuild |
| `agents/bus.py` | spreading_activation(propagate_valence=True) with @overload |
| `runtime/bio_stack.py` | BioStack.cerebellum, reaction capture subscriber, Cerebellum construction, distribute_reward subscriber |
| `cli.py` | cerebellum= forwarded to build_executor |
| `simulation/orchestrator.py` | cerebellum= forwarded to build_executor |
| `embodiment/backends/cerebellum_modulator.py` | _emit_success_reaction on confident predictions |
