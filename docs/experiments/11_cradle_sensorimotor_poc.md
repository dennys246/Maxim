# Experiment 11: Cradle Sensorimotor PoC

**Date:** 2026-04-26
**Branch:** feat/cradle-sensorimotor
**Status:** Infrastructure validated, narrator generating cradle scenes

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

## Run 6 (Claude Sonnet, with all fixes) — VALIDATED

```bash
MAXIM_LOG_FILE=/tmp/cradle_test6.jsonl maxim --sim cradle \
  --embodiment bodies/infant_humanoid --interactive false \
  --sim-max-turns 10 --language-model claude-sonnet
```

- **Duration:** 390.3s, 10 turns, $0.21
- **Narrator:** ALL 10 scenes generated (0 fallbacks!). Fire pit, warm room, objects all described correctly.
- **Phase progression:** exploration → pain_consequence → object_introduction → discrimination
- **Tool usage:** sense_tools (4x), infant_humanoid_pick_up (2x!), examine (1x), infant_humanoid_use (1x)
- **Bio-pipeline:** 355 episodic memories, 23 causal links
- **Narrator content:** "You find yourself in a cozy, warm room where the scent of wood smoke fills the air. The fire pit cra[ckles]..." → "You sit in the cozy room... In front of [you, two objects]..."

### Bugs fixed during PoC

1. **cli.py:** Auto-detect generative mode for goal strings matching builtin arcs
2. **orchestrator.py:** Import `sim_reports` at generative runner call site (UnboundLocalError)
3. **narrator.py:** Phase instructions now flow to generate() call (were only in decide())
4. **narrator.py:** Use `generate_json()` instead of `generate_text()` (LLMRouter has no generate_text)
5. **infant_humanoid.yaml:** Tuned hunger/thirst drift rates 3x faster for shorter PoC runs

## Next Steps

1. **Cross-session test:** Resume with `--resume-sim` to validate enrichment surfaces prior-session memories.
2. **Fix `GenerativeCampaignResult.turns_completed`** attribute error (non-blocking — sim completes before this).
3. **Orchestrator sensor writes:** The narrator describes fire proximity but doesn't write thermal sensors yet. This requires the orchestrator (not narrator) to have `set_entity_sensor` tool access during generative campaigns.
4. **25-turn runs:** Test full 4-act arc including Act 3 (secondary circular) and Act 4 (consolidation).

## Reproduction

```bash
git checkout feat/cradle-sensorimotor
source ~/.zshrc  # for ANTHROPIC_API_KEY
PYTHONPATH=src MAXIM_LOG_FILE=/tmp/cradle.jsonl \
  maxim --sim cradle --embodiment bodies/infant_humanoid \
  --interactive false --sim-max-turns 15 --language-model claude-sonnet
```
