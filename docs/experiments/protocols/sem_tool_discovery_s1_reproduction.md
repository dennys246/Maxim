# SEM Tool Discovery S1 — Reproduction Protocol

**Experiment:** [sem_tool_discovery_s1.md](../sem_tool_discovery_s1.md)

## Prerequisites

- pymaxim 0.7.0 with S1 commit applied
- Leader with an LLM running (any model — local or Claude)
- `ANTHROPIC_API_KEY` or peer config set up

## Steps

### 1. Run the S1 sim

```bash
MAXIM_LOG_FILE=/tmp/sem_discovery_s1.jsonl \
  maxim --sim "test sword combat" \
    --embodiment weapons/rusty_sword \
    --interactive false \
    --sim-max-turns 5
```

### 2. Verify tool count reduction

Count active tools in the first LLM request:
```bash
python3 -c "
import json
with open('/tmp/sem_discovery_s1.jsonl') as f:
    for line in f:
        ev = json.loads(line)
        if 'available_tools' in ev.get('data', {}):
            tools = ev['data']['available_tools']
            print(f'Active tools: {len(tools)}')
            for t in sorted(tools):
                print(f'  {t}')
            break
"
```

**Expected:** ~12 tools (6 core + sense + discover_tools + 3-5 top-k affordances)

### 3. Check turn-1 affordance use

```bash
python3 -c "
import json
with open('/tmp/sem_discovery_s1.jsonl') as f:
    for line in f:
        ev = json.loads(line)
        if ev.get('event') == 'tool_call':
            print(f'Turn {ev.get(\"turn\", \"?\")}: {ev[\"data\"][\"tool_name\"]}')
" | head -5
```

**Expected:** First tool call should be an affordance tool (slash, parry, etc.), not discover_tools.

### 4. Check for discover_tools and sense usage

```bash
grep -o '"tool_name": "[^"]*"' /tmp/sem_discovery_s1.jsonl | sort | uniq -c | sort -rn
```

**Expected:** `sense` and/or `discover_tools` appear in the tool call distribution.

### 5. Compare baseline (optional A/B)

Check out the pre-S1 commit and run the same sim:
```bash
git stash  # or checkout pre-S1
MAXIM_LOG_FILE=/tmp/sem_discovery_baseline.jsonl \
  maxim --sim "test sword combat" \
    --embodiment weapons/rusty_sword \
    --interactive false \
    --sim-max-turns 5
git stash pop  # restore S1
```

Compare tool counts between the two JSONL files.

## Success Criteria

1. Active tool count < 15 on turn 1 (down from ~36)
2. Agent uses an affordance tool on turn 1 (no cold start regression)
3. `sense` and `discover_tools` are both present and callable
4. No errors related to missing tools or deactivated tool calls
