# Experiment 08: Component-Level Damage PoC

**Date:** 2026-04-25
**Branch:** `feat/sem-damage-autosense`
**Version:** 0.8.0+ (component damage infrastructure)

## Hypothesis

When the AUT (agent-under-test) has a composable body with per-component integrity sensors and affordance `requires` gating:

1. **H1: Damage cascades correctly.** Component damage (e.g., wing.integrity drops) propagates to entity health via derived aggregation.
2. **H2: Affordance blocking fires.** When a component's integrity drops below the `requires` threshold, the affordance tool returns a failure result (not a silent no-op).
3. **H3: Pain signals carry component context.** PainBus publishes include `component` field in the context dict.
4. **H4: Scene manifest pre-triggers entities.** The dragon and other scene entities are live before the first AUT turn.
5. **H5: The orchestrator uses DamageComponentTool.** With the updated TOOL_DESCRIPTIONS, the orchestrator targets specific body parts (not flat health).

## Protocol

### Prerequisites

- Leader running with `feat/sem-damage-autosense` branch
- Substrate path enabled: `MAXIM_SUBSTRATE_PATH=1`
- Log file for analysis: `MAXIM_LOG_FILE=/tmp/component_damage_poc.jsonl`
- Backend trace for token tracking: `MAXIM_BACKEND_TRACE=1`

### Run command

```bash
MAXIM_SUBSTRATE_PATH=1 \
MAXIM_LOG_FILE=/tmp/component_damage_poc.jsonl \
MAXIM_BACKEND_TRACE=1 \
maxim --sim "You are an adventurer in a cave. A dragon attacks you with claws and fire breath. Fight back with your sword. The dragon targets your arms and legs. You take heavy wing damage when the dragon grabs you and flies." \
  --embodiment bodies/base_humanoid \
  --interactive false \
  --sim-max-turns 10
```

### Validation checks

After the run, analyze the JSONL:

```bash
# H1: Check entity health derives from components
cat /tmp/component_damage_poc.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    if e.get('event') == 'SEM_DAMAGE':
        print(e.get('message', ''))
"

# H2: Check for affordance blocking
cat /tmp/component_damage_poc.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    msg = e.get('message', '')
    if 'too damaged' in msg or 'affordance_blocked' in str(e):
        print(msg)
"

# H3: Check pain signals have component context
cat /tmp/component_damage_poc.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    if e.get('event') == 'pain_published':
        ctx = e.get('data', {}).get('context', {})
        if 'component' in ctx:
            print(f\"component={ctx['component']}, integrity={ctx.get('component_integrity', '?')}\")
"

# H4: Check scene manifest pre-trigger
grep 'Scene manifest' /tmp/component_damage_poc.jsonl | head -5

# H5: Check orchestrator uses damage_component (not damage_entity)
grep 'damage_component\|damage_entity' /tmp/component_damage_poc.jsonl | head -10
```

### Success criteria

| Hypothesis | Pass condition |
|-----------|---------------|
| H1 | At least one `SEM_DAMAGE` log with `component damage:` (not `fallback to entity health`) |
| H2 | At least one `too damaged for` message in logs (affordance blocked) |
| H3 | At least one pain signal with `component` key in context |
| H4 | `Scene manifest pre-trigger: N entities resolved` with N > 0 |
| H5 | At least one `damage_component` tool call in logs |

### Expected results

The sim should show:
1. Scene manifest resolving dragon + sword + cave features before turn 1
2. Dragon attacks targeting specific humanoid body parts (arms, legs, torso)
3. Component integrity degrading per body part
4. Eventually, some affordances blocking (e.g., move blocked if leg integrity < 0.1)
5. Pain signals with body-part context flowing to NAc for causal learning

### Notes

- The orchestrator LLM must understand `damage_component` from TOOL_DESCRIPTIONS. If it calls `damage_entity` instead, the deprecated shim handles it (targets torso). This is expected for the first run — the LLM needs to see the new tool description.
- H2 (affordance blocking) depends on enough sustained damage to a single component. If the orchestrator spreads damage across many parts, no single part may cross the threshold in 10 turns. Increase `--sim-max-turns 15` if needed.
- The PoC tests the infrastructure end-to-end. Cross-session learning (does the agent avoid attacks to its damaged parts next time?) is tested separately in the affordance concept transfer experiments.

## Results

_(To be filled after running the experiment.)_
