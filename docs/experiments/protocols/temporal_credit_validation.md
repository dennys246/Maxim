# Temporal Credit Validation — Reproduction Protocol

**Experiment:** [temporal_credit_validation.md](../temporal_credit_validation.md) (results)
**Plan:** [temporal_credit_integration.md](../../plans/archive/temporal_credit_integration.md)

## Overview

Four simulation sets validating the temporal credit integration (Phases 1-7):

1. **Cross-session affordance transfer** — the headline 1.0 test. Fire danger learned from dragon transfers to mage's fire affordances across sessions.
2. **Goal-level deliberation learning** — `_goal_reward_bias` bidirectional pathway modulates ThoughtGate across a DM campaign.
3. **Multi-entity imagination + temporal credit** — diverse entity imagination, affordance encoding, temporal credit distribution in a single rich session.
4. **Sensory deprivation stress test** — bio-system adaptation under progressive sensor degradation.

## Requirements

- Local LLM (tested with `qwen2.5-14b-instruct` on RTX 5080)
- ~2 hours total runtime
- ~500MB disk for session data + JSONL logs

## Quick run (all 4 sets)

```bash
PYTHONPATH=src bash scripts/temporal_credit_validation.sh
```

Results are saved to `~/.maxim/experiments/temporal_credit_YYYYMMDD_HHMMSS/`.

## Individual runs

### Sim Set 1: Cross-session affordance transfer (~30 min)

The most important test. Two sessions sharing a persistence directory.

```bash
RESULTS_DIR=~/.maxim/experiments/temporal_credit_manual
mkdir -p "$RESULTS_DIR"

# Session 1: Dragon encounter — learn fire = dangerous
MAXIM_LOG_FILE="$RESULTS_DIR/sim1_dragon.jsonl" \
MAXIM_BACKEND_TRACE=1 \
maxim --sim "explore a dungeon, encounter a fire-breathing dragon, try to fight it" \
  --embodiment bodies/base_humanoid \
  --sim-max-turns 20 \
  --interactive false \
  --language-model qwen2.5-14b-instruct

# Note the session ID from the report output, then:
SESSION_ID=<session_id_from_sim1>

# Session 2: Mage encounter — does fire transfer?
MAXIM_LOG_FILE="$RESULTS_DIR/sim2_mage.jsonl" \
MAXIM_BACKEND_TRACE=1 \
maxim --sim "explore a wizard's tower, encounter a fire mage with flame attacks" \
  --embodiment bodies/base_humanoid \
  --sim-max-turns 20 \
  --interactive false \
  --resume-sim "$SESSION_ID" \
  --language-model qwen2.5-14b-instruct
```

**What to check:**
- `grep "DANGEROUS" "$RESULTS_DIR/sim2_mage.jsonl"` — fire affordances annotated without direct experience
- `grep "reward_bias" "$RESULTS_DIR/sim2_mage.jsonl"` — negative bias on fire substrate nodes
- `grep "goal_reward_bias" "$RESULTS_DIR/sim2_mage.jsonl"` — goal-level learning carried over
- Compare ThoughtGate fire rates between session 1 (baseline) and session 2 (biased)

### Sim Set 2: Goal-level deliberation learning (~30 min)

```bash
MAXIM_LOG_FILE="$RESULTS_DIR/sim2_arena.jsonl" \
maxim --sim scenarios/campaigns/arena_v1.yaml \
  --embodiment weapons/rusty_sword \
  --sim-max-turns 40 \
  --interactive false \
  --language-model qwen2.5-14b-instruct
```

**What to check:**
- `grep "credit_goal" "$RESULTS_DIR/sim2_arena.jsonl"` — goal credits accumulate
- `grep "goal_reward_bias" "$RESULTS_DIR/sim2_arena.jsonl"` — bias entries for recurring goals
- Compare ThoughtGate threshold in early vs late encounters

### Sim Set 3: Multi-entity imagination (~45 min)

```bash
MAXIM_LOG_FILE="$RESULTS_DIR/sim3_marketplace.jsonl" \
maxim --sim "explore a fantasy marketplace, then venture into monster-infested ruins, encountering merchants, thieves, creatures, and magical artifacts" \
  --embodiment bodies/base_humanoid \
  --auto-curate \
  --sim-max-turns 30 \
  --interactive false \
  --language-model qwen2.5-14b-instruct
```

**What to check:**
- `grep "imagination" "$RESULTS_DIR/sim3_marketplace.jsonl"` — entity extraction + instantiation
- `grep "index_hit\|cache_hit\|design" "$RESULTS_DIR/sim3_marketplace.jsonl"` — hit types
- `grep "encode_decomposed\|substrate" "$RESULTS_DIR/sim3_marketplace.jsonl"` — affordance encoding

### Sim Set 4: Sensory deprivation stress test (~30 min)

```bash
MAXIM_LOG_FILE="$RESULTS_DIR/sim4_cavern.jsonl" \
maxim --sim scenarios/campaigns/darkened_cavern_v1.yaml \
  --embodiment bodies/base_humanoid \
  --sim-max-turns 30 \
  --interactive false \
  --language-model qwen2.5-14b-instruct
```

**What to check:**
- `grep "PAIN\|pain" "$RESULTS_DIR/sim4_cavern.jsonl"` — pain fires on sensory degradation
- `grep "cerebellum\|forward_model" "$RESULTS_DIR/sim4_cavern.jsonl"` — model adaptation
- Compare enrichment section counts early vs late (fewer sensors → fewer sections)

## Validation criteria

| Set | Pass condition |
|-----|---------------|
| 1 | Session 2 shows `[DANGEROUS]` on mage fire affordances without direct experience |
| 2 | `_goal_reward_bias` has non-zero entries for recurring goals by end of campaign |
| 3 | Imagination instantiates 3+ diverse entity types, affordance substrate nodes form |
| 4 | PainBus fires on sensory threshold crossings, bio-systems continue functioning under degradation |

## What to check if validation fails

1. **No `[DANGEROUS]` in session 2:** Check NAc persistence — is `nac.json` in the session directory? Does it contain `reward_bias` entries for fire-related nodes? If empty, the session wasn't saved properly.
2. **`_goal_reward_bias` stays at zero:** Check that `active_goal` is populated in `state.data` during the sim. If None, the orchestrator isn't setting goals. Check `tool_dispatch.py::record_outcome` — verify `active_goal` parameter is threaded.
3. **Imagination doesn't fire:** Check `ImaginationTrigger ACTIVE` trace at sim start. If missing, the trigger wasn't constructed. Check orchestrator wiring around line 1219.
4. **PainBus silent:** Check that `--embodiment` is passed (pain requires SEM entities). Check ToolPainBridge construction in `build_executor`.
