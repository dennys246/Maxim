# Experiment 11: Cradle Sensorimotor PoC

**Date:** 2026-04-26
**Branch:** feat/cradle-sensorimotor
**Status:** Infrastructure validated, narrator tuning needed

## Setup

```bash
# Run 1: Local 14B (qwen2.5-14b-instruct), 8 turns
maxim --sim cradle --embodiment bodies/infant_humanoid --interactive false --sim-max-turns 8

# Run 2: Claude Sonnet, 12 turns (before generative mode fix)
maxim --sim cradle --embodiment bodies/infant_humanoid --interactive false --sim-max-turns 12 --language-model claude-sonnet

# Run 3: Claude Sonnet, 15 turns (with generative mode fix)
MAXIM_LOG_FILE=/tmp/cradle_test4.jsonl maxim --sim cradle --embodiment bodies/infant_humanoid --interactive false --sim-max-turns 15 --language-model claude-sonnet
```

## Results

### Run 1 (local 14B)
- **Issue:** Standard orchestrator path, not generative runner. "cradle" treated as goal word.
- **Lesson:** Need `generative=True` for goal strings matching builtin arcs.

### Run 2 (Claude, pre-fix)
- **Issue:** Same — standard orchestrator, not generative runner.
- **Fix applied:** Auto-detect generative mode when `select_arc_for_goal()` matches.

### Run 3 (Claude, with fix) — PRIMARY RESULTS
- **Duration:** 183.9s, 15 turns, $0.44
- **Finish:** max_turns (generative runner completed its arc)
- **Tool usage:**
  - infant_humanoid_look: 8 calls (100% success) — exploration
  - infant_humanoid_use: 3 calls (100% success) — interaction
  - sense_tools: 2 calls (100% success) — affordance discovery
  - infant_humanoid_pick_up: 1 call (100% success) — entity acquisition
  - sense: 1 call (100% success) — body sensing
- **Bio-pipeline:**
  - 20 enrichment traces (all with hippocampus + NAc wired)
  - 30 causal links formed (infant_humanoid_pick_up, look, use, sense_tools)
  - 15 episodic memories captured
  - SCN registered all tool events with circadian signatures
- **Issues:**
  - Narrator did not follow cradle arc phase instructions closely (generic scenes, no fire pit mention)
  - No thermal sensor writes observed (orchestrator didn't use set_entity_sensor)
  - Drive drift did not produce visible hunger/temperature signals in this run
  - `GenerativeCampaignResult.turns_completed` attribute error on session save

## Infrastructure Validation

| Component | Status | Evidence |
|---|---|---|
| DriveSpec parsing | PASS | infant_humanoid loads with 6 drive_specs |
| Entity acquisition | PASS | pick_up tool called, NAc link formed |
| Self-effect | NOT TESTED | No food entity interaction in this run |
| Drive prompt visibility | NOT TESTED | No drives crossed thresholds in 15 turns |
| Act tags on NarrativePhase | PASS | Cradle arc loaded, generative runner activated |
| Cradle YAML templates | PASS | All templates load and instantiate |
| Generative mode auto-detect | PASS (after fix) | select_arc_for_goal → generative=True |

## Fixes Applied During PoC

1. **cli.py:** Auto-detect generative mode for goal strings matching builtin arcs
2. **orchestrator.py:** Import `sim_reports` at generative runner call site (UnboundLocalError fix)

## Next Steps

1. **Narrator tuning:** Phase instructions need to be stronger — the narrator generates generic fantasy scenes instead of the cradle-specific developmental stimuli. Consider: (a) shorter more directive instructions, (b) explicit sensor-write commands in narrator system prompt, (c) cradle-specific persona that understands developmental stages.
2. **Longer runs:** 25+ turns needed for hunger to cross deprivation threshold (at 0.002/s drift rate, ~350s = ~6 min).
3. **Cross-session test:** Resume with `--resume-sim` to validate enrichment surfaces prior-session memories.
4. **Fix `GenerativeCampaignResult.turns_completed`** attribute error.

## Reproduction

```bash
git checkout feat/cradle-sensorimotor
source ~/.zshrc  # for ANTHROPIC_API_KEY
PYTHONPATH=src MAXIM_LOG_FILE=/tmp/cradle.jsonl \
  maxim --sim cradle --embodiment bodies/infant_humanoid \
  --interactive false --sim-max-turns 15 --language-model claude-sonnet
```
