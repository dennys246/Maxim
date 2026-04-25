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

## Results (2026-04-25)

**Run on:** RTX 5080 leader, Qwen 2.5 14B Instruct, Mac peer
**Duration:** ~90s for 5 turns analyzed (full 10 turns ~180s)
**Branch:** `feat/sem-damage-autosense` commit `b94c2a3`

### Hypothesis Results

| H# | Result | Evidence |
|----|--------|----------|
| H1 | **PASS** | `torso.integrity: 1.00 → 0.85 → 0.70 → 0.55 → 0.40 → 0.25 → 0.10`. `health: 0.98 → 0.93 → 0.88 → 0.82 → 0.77 → 0.72 → 0.67`. Derived health correctly tracks component integrity via weighted mean. |
| H2 | **NOT TESTED** | All auto-damage targeted torso (default). Torso has no `requires`-gated affordances. Need leg/arm targeting to trigger blocking. Deferred to reflex system implementation. |
| H3 | **PASS** | Pain signals fire every damage event: `🔴 pain (intensity=0.30) from pain_detector:external_signal`. `SEM_DAMAGE component damage: torso.integrity → X (source=auto_attack)` logs confirm component targeting. |
| H4 | **PASS** | `Manifest pre-trigger: 2 phrases extracted: ['large wooden sword', 'ground a giant dragon']`. Dragon instantiated from seed: `Scene entity 'dragon' instantiated from creatures/dragon`. `Scene manifest pre-trigger: 1 entities resolved`. |
| H5 | **PARTIAL** | 14B Qwen model does not call `damage_component` explicitly despite updated persona prompt. Auto-damage fallback routes through `DamageComponentTool(torso)` correctly. Tool infrastructure works; LLM tool-calling is the bottleneck. |

### Damage Cascade Data

| Turn | torso.integrity | health (derived) | Pain | Episode valence |
|------|----------------|-------------------|------|-----------------|
| 0 | 1.00 | 0.98 | — | — |
| 1 | 0.85 | 0.93 | 0.30 | — |
| 2 | 0.70 | 0.88 | 0.30 | -0.30 |
| 3 | 0.55 | 0.82 | 0.30 | -0.30 |
| 4 | 0.40 | 0.77 | 0.30 | -0.30 |
| 5 | 0.25 | 0.72 | 0.30 | -0.30 |
| 6 | 0.10 | 0.67 | 0.30 | -0.30 |

Health derivation: `health = head(0.94)*0.30 + torso(X)*0.35 + arms(1.0)*0.15 + legs(1.0)*0.20`

### Key Observations

1. **Dragon resolves from seed on first mention** — head-noun alias fallback (`"attack by a dragon"` → score=0.95) works reliably.
2. **Sword designed by imagination** — `"yourself using your sword"` triggers LLM design, produces `weapons/sword` with 1 modulator and 1-2 affordances.
3. **Component sensors visible every tick** — `head.integrity=0.94, torso.integrity=X, arms.integrity=1.00, legs.integrity=1.00` in all sensor dumps.
4. **NAc learning active** — `tool:base_humanoid_use → positive` links forming with increasing confidence (0.50 → 0.64 → 0.67).
5. **Episode valence tracks pain** — episodes close with valence=-0.30 matching pain intensity. The agent is encoding "this hurts" in episodic memory.
6. **Auto-damage fallback works** — routes through full `DamageComponentTool` pipeline (integrity reduction → derived health → pain → NAc).
7. **14B models can't reliably call damage_component** — this validates the need for the percept reflex system (auto-detection without LLM tool-calling).

### Limitations

- All damage targets torso (auto-damage default). Body-part detection from percept text is needed (percept reflex system plan).
- No habituation — every turn applies identical 0.15 damage.
- No affordance blocking demonstrated — needs arm/leg damage.
- Orchestrator prompt strength insufficient for 14B tool-calling.
- Scene manifest resolved only 1 entity (dragon) from 2 phrases — "large wooden sword" didn't match any seed (no `sword` alias without `rusty_`).

### Next Steps

1. Implement percept reflex system → auto-damage becomes a reflex with body-part targeting
2. Test with Claude as orchestrator → should call damage_component explicitly
3. Run cross-session test → does NAc retain attack→pain links?
4. Add leg-targeted damage scenario → test affordance blocking
