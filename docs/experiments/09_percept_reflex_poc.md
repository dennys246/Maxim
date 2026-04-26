# Experiment 09: Percept Reflex System PoC

**Date:** 2026-04-25
**Branch:** `feat/percept-reflex-system`
**Version:** 0.8.0+ (percept reflex system)
**Depends on:** Experiment 08 (component damage PoC, validated)

## Hypothesis

When the AUT has a composable body with the percept reflex system active:

1. **H1: Reflexes fire automatically.** Attack keywords in orchestrator narration trigger the `attack_flinch` reflex, applying damage to the torso component WITHOUT the orchestrator calling `damage_component` explicitly.
2. **H2: Body-part targeting works.** The `fire_burn` reflex targets torso, `impact_brace` targets legs, `startle` adjusts awareness — different reflexes route to different body parts/sensors.
3. **H3: Pain signals carry reflex context.** Pain published from reflex-triggered damage includes `source: reflex_attack` (not `auto_attack`).
4. **H4: Habituation reduces intensity.** Repeated attack narration in the same context produces decreasing damage amounts across turns.
5. **H5: Sensitization amplifies on damaged parts.** As torso integrity drops, subsequent reflex damage increases (sensitization factor > 1.0).
6. **H6: SendMessageTool no longer applies auto-damage.** The old `_detect_attack` code is gone — all damage routes through reflexes.
7. **H7: Telemetry visible.** `sim_enrichment("reflex", ...)` logs appear in JSONL with reflex names and effective intensities.

## Protocol

### Prerequisites

- Leader running with `feat/percept-reflex-system` branch
- Substrate path enabled: `MAXIM_SUBSTRATE_PATH=1`
- Log file for analysis: `MAXIM_LOG_FILE=/tmp/reflex_poc.jsonl`
- Backend trace for token tracking: `MAXIM_BACKEND_TRACE=1`

### Run command

```bash
MAXIM_SUBSTRATE_PATH=1 \
MAXIM_LOG_FILE=/tmp/reflex_poc.jsonl \
MAXIM_BACKEND_TRACE=1 \
maxim --sim "You are an adventurer in a dark cave. A dragon attacks you repeatedly with claws and fire breath. The dragon roars deafeningly. It slams you against the wall. A freezing wind blows through the cave." \
  --embodiment bodies/base_humanoid \
  --interactive false \
  --sim-max-turns 8
```

### Validation checks

```bash
# H1: Check reflex firings in enrichment logs
cat /tmp/reflex_poc.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    msg = e.get('message', '')
    if 'reflex' in msg.lower():
        print(msg[:200])
"

# H2: Check body-part targeting (different components)
cat /tmp/reflex_poc.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    if e.get('event') == 'SEM_DAMAGE':
        print(e.get('message', '')[:200])
"

# H3: Check pain signals have reflex source
cat /tmp/reflex_poc.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    if e.get('event') == 'pain_published':
        ctx = e.get('data', {}).get('context', {})
        src = ctx.get('source', ctx.get('failure_mode', ''))
        if 'reflex' in str(src):
            print(f'source={src}, component={ctx.get(\"component\", \"?\")}, intensity={e.get(\"data\", {}).get(\"intensity\", \"?\")}')
"

# H4: Check for habituation (decreasing damage amounts)
cat /tmp/reflex_poc.jsonl | python3 -c "
import json, sys
amounts = []
for line in sys.stdin:
    e = json.loads(line)
    if e.get('event') == 'SEM_DAMAGE':
        msg = e.get('message', '')
        if 'amount=' in msg:
            import re
            m = re.search(r'amount=([0-9.]+)', msg)
            if m:
                amounts.append(float(m.group(1)))
print(f'Damage amounts across turns: {amounts}')
if len(amounts) >= 3:
    print(f'Habituation detected: {amounts[0] > amounts[-1]}')
"

# H5: Check sensitization (integrity × damage)
cat /tmp/reflex_poc.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    msg = e.get('message', '')
    if 'reflex' in msg.lower() and 'intensity' in msg:
        print(msg[:200])
"

# H6: Verify no auto_attack source (old SendMessageTool path)
grep 'auto_attack\|auto_damage' /tmp/reflex_poc.jsonl

# H7: Check sim_enrichment reflex telemetry
cat /tmp/reflex_poc.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    if e.get('event') == 'sim_enrichment' and 'reflex' in str(e.get('data', '')):
        print(e.get('data', ''))
"
```

### Success criteria

| Hypothesis | Pass condition |
|-----------|---------------|
| H1 | At least one `sim_enrichment("reflex", ...)` log with `attack_flinch` |
| H2 | Damage logs targeting at least 2 different components (torso + legs) |
| H3 | Pain signal with `source` containing `reflex` |
| H4 | Damage amounts show decreasing trend (3+ data points) |
| H5 | At least one reflex firing with sensitization_factor > 1.0 |
| H6 | Zero matches for `auto_attack` or `auto_damage` in JSONL |
| H7 | `sim_enrichment` logs with `reflex` substring present |

### Expected behavior

1. Scene manifest resolves dragon + cave before turn 1
2. Each turn's narration triggers reflexes based on keywords (attack → torso, fire → torso, slam → legs, roar → awareness)
3. Component integrity degrades per body part
4. Damage per reflex firing decreases as habituation builds
5. Damage per reflex firing increases as sensitization kicks in (on already-damaged parts)
6. No `auto_attack` logs — all damage routes through reflex system

### Notes

- Habituation and sensitization work in opposite directions. Net effect depends on which dominates: early turns see habituation reducing damage; late turns (with low integrity) may see sensitization overtaking habituation.
- The `startle` reflex adjusts awareness sensor, not component integrity. Check for `set_entity_sensor` in logs, not `SEM_DAMAGE`.
- `environment_cold` targets stamina and has a longer cooldown (10s). May not fire more than once in 8 turns.

## Results

*Pending — to be filled after running the experiment.*
