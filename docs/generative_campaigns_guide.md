# Generative Campaign Guide

## Overview

Generative campaigns are LLM-driven narrative simulations where a narrator generates scenes dynamically, adapting to the AUT's responses while following structured narrative arcs. This is the default mode when you pass a goal string to `--sim`.

## Quick Start

```bash
# Basic generative campaign
maxim --sim "test memory recall under interference"

# With research report after sim
maxim --sim "test causal learning" --research

# Interactive mode (human-in-the-loop)
maxim --sim "adventure in the forest" --sim-interactive

# YAML campaign (direct injection, bypass generative)
maxim --sim scenarios/experiments/hippocampal_recall_short.yaml

# Tiered benchmarks
maxim --benchmark tier1 --models mistral-7b,qwen2.5-14b
maxim --benchmark all --models mistral-7b
```

## How It Works

```
Goal → Arc Selection → Narrator → Bridge → AUT → Response → Next Turn
        ↕                  ↕                         ↓
   AdaptivePlanner    Two-call approach         Memory systems
   (memory context)   (decision + generation)   (NAc, Hippocampus, ATL)
```

1. **Arc selection**: goal keywords match a builtin arc, or AdaptivePlanner decomposes the goal into phases
2. **Narrator**: generates narrative scenes (two-call approach: decision JSON + plain text generation)
3. **Bridge**: sends narrative to AUT via `SimulationBridge.send_and_wait()`
4. **AUT responds**: cognitive architecture processes the narrative, selects tools, generates response
5. **Loop**: narrator adapts based on response, advances through arc phases

## Narrative Arcs

### Builtin Arcs

| Arc | Tests | Phases | Min Turns |
|-----|-------|--------|-----------|
| `memory_recall` | Episodic memory retention | seed → reinforcement → interference → recall → epilogue | 7 |
| `causal_learning` | Cause-effect learning | establish → variation → reversal | 5 |
| `safety_boundary` | Safety boundary maintenance | trust_building → escalation → boundary_test | 6 |
| `skill_learning` | Bio-skill acquisition | introduction → guided_practice → independent_practice → interference → indirect_recall → transfer → reflection | 15 |

### Custom Arcs (YAML)

```yaml
name: "emotional_memory"
description: "Test if emotionally charged events are recalled better"
phases:
  - name: neutral_seed
    turns: [2, 2]
    instruction: "Describe a mundane, forgettable scene"
  - name: emotional_seed
    turns: [1, 1]
    instruction: "Describe a highly emotional event with a specific detail"
  - name: interference
    turns: [5, 5]
    instruction: "Neutral encounters"
  - name: recall_neutral
    turns: [1, 1]
    instruction: "Cue recall of the neutral scene's detail"
  - name: recall_emotional
    turns: [1, 1]
    instruction: "Cue recall of the emotional scene's detail"
```

## Narrator Approaches

### Two-Call (Option C) — Default for Medium/Large Models

1. **Decision call** (JSON): `{"phase": "interference", "scene_type": "encounter", "done": false}`
2. **Generation call** (text): "Past the marsh, the road narrows. Three bandits drop from the trees..."

The decision call is simple JSON (no narrative text), avoiding JSON escaping issues. The generation call outputs raw text.

### Single-Call (Option B) — Fallback for Small Models

Combined decision + narrative in one JSON call: `{"phase": "seed", "narrative": "The forest...", "done": false}`.

## AdaptivePlanner Integration

When available, the AdaptivePlanner decomposes goals with memory context:
- **NAc predictions**: "the AUT has seen this before — outcome was positive"
- **Situation novelty**: "this is highly novel — introduce gradually"
- **Ranked skills**: "the AUT knows herbalism — test at higher difficulty"
- **Hippocampal associations**: "related to prior campfire scene"

The planner's decomposition is translated into a `NarrativeArc` via `translate_plan_to_arc()`. Memory context is formatted as narrator hints via `enrich_narrator_context()`.

### Bridge-and-Compress (Large Models)

Large models can run multi-arc campaigns. When one arc completes:
1. Story is compressed (~200 token summary)
2. AdaptivePlanner re-decomposes with updated memory state
3. Bridge scene connects arcs narratively
4. Next arc begins seamlessly

## Interactive Mode

With `--sim-interactive`, the narrator can pause for human input:

```bash
maxim --sim "adventure in the dungeon" --sim-interactive
```

The `ask_user` tool prompts via stdin with randomized timeouts (~10s). Timeout escalation:

| Consecutive Timeouts | Behavior |
|---------------------|----------|
| 1st | Gentle nudge — NPC shifts impatiently |
| 2nd-3rd | World reacts — opportunities close, new paths open |
| 4th+ | Passive protagonist — story happens TO the AUT |

All interactions are recorded to `user_interactions.jsonl`.

## YAML Export

Every generative run exports turns to `data/sim_reports/{session_id}/generated_campaign.yaml`. This enables:
- **Replay**: `maxim --sim generated_campaign.yaml`
- **A/B testing**: same narrative, different AUT models
- **Debugging**: inspect what the narrator generated vs. arc template

## SEM World Entities

Generative campaigns can declare `world_entities` in arc YAML for structured items/NPCs:

```yaml
world_entities:
  - name: rusty_sword
    entity_type: weapon
    sensors:
      durability: {unit: ratio, range: [0, 1], initial: 0.3}
    modulators:
      combat:
        affordances:
          slash: {params: {target: str}, description: "Slash at target"}
    failure_modes:
      - name: shatter
        trigger: {field: durability, op: "<", value: 0.1, pain: 0.6}
```

Auto-generates tools, pain triggers, and Cerebellum forward models. See the [Embodiment Guide](embodiment_guide.md) for details.

## Benchmark Tiers

```bash
maxim --benchmark tier1 --models mistral-7b         # Cognitive (fast)
maxim --benchmark tier2 --models mistral-7b         # Bio-system (moderate)
maxim --benchmark tier3 --models mistral-7b         # Embodiment (expensive)
maxim --benchmark all --models mistral-7b,qwen2.5   # All tiers
```

| Tier | What | Cost |
|------|------|------|
| Tier 1: Cognitive | Memory, causal learning, safety | ~$0.05, ~2min |
| Tier 2: Bio-system | Hippocampus, NAc, ATL, pain | ~$0.15, ~5min |
| Tier 3: Embodiment | SEM tools, Cerebellum, motor programs | ~$0.30, ~10min |

## Module Map

| File | Purpose |
|------|---------|
| `simulation/arcs.py` | NarrativeArc, builtin arcs, YAML loader, goal-to-arc matching |
| `simulation/narrator.py` | Narrator (two-call + single-call), system prompts |
| `simulation/generative_runner.py` | Main runner, YAML export, SEM entity loading |
| `simulation/plan_arc_bridge.py` | Plan-to-arc translation, narrator context enrichment, bridge-and-compress |
| `simulation/tools_user.py` | ask_user tool, JSONL audit, replay, timeout escalation |
| `mesh/identity.py` | AgentProfile with entity_type + log_prefix |

## CLI Reference

| Flag | Description |
|------|-------------|
| `--sim <goal>` | Goal string → generative campaign |
| `--sim <path.yaml>` | YAML path → direct injection campaign |
| `--research` | Generate research report after sim |
| `--sim-interactive` | Enable human-in-the-loop |
| `--benchmark [tiers]` | Top-level benchmark command |
| `--models <list>` | Models for benchmark |
| `--continuous` | Never auto-complete |
| `--campaign <yaml>` | Force direct injection with custom goal |
