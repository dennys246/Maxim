# Reproduction — Roy-3 (Roy-3a + Roy-3b: 0.9.1 annotation-pattern validation)

**Companion:** [23_roy_3.md](../23_roy_3.md)
**Iteration specs:** [scenarios/roy/roy_3a_iteration.yaml](../../../scenarios/roy/roy_3a_iteration.yaml) · [scenarios/roy/roy_3b_iteration.yaml](../../../scenarios/roy/roy_3b_iteration.yaml)
**Fixtures:** [scenarios/roy/roy_1_holdout.yaml](../../../scenarios/roy/roy_1_holdout.yaml) (Roy-3a) · [scenarios/roy/roy_2pc_holdout.yaml](../../../scenarios/roy/roy_2pc_holdout.yaml) (Roy-3b — reused from Roy-2pc / Roy-2c / Roy-4 / Roy-5a)
**A/B partners:** [18_roy_2.md](../18_roy_2.md), [19_roy_2pc.md](../19_roy_2pc.md), [20_roy_2c.md](../20_roy_2c.md)
**Owning plan:** [release_0_9_1.md § Stage 5](../../plans/release_0_9_1.md)

Roy-3 is the 0.9.1 Stage 5 validation iteration. Two sub-iterations measure whether Wires A+1+2+3 (PRs #253 / #254 / #255 / #256 / #257, merged 2026-05-13 → 2026-05-22) produced cross-arm behavioral divergence on the Roy harness that the bare `cluster_reward_bias` path could not under Roy-1a / Roy-2 / Roy-2pc / Roy-2c.

## Pre-registered diagnostic outcomes

| Outcome | Diagnosis |
|---|---|
| Arm A `sense_food_source` count > B AND > C in Roy-3b | Wire-A reached LLM proposer's decision pathway on engineered overlap. |
| Arm A > C on tool-family divergence in Roy-3a (richer than Roy-2's 17/3/2 vs 21/5/1/1) | Wire 1 + Wire 2 compound the salience-mediated signal Roy-2 surfaced. |
| **A ≈ B ≈ C across both fixtures** | **Annotation pattern alone is insufficient; investigate prompt rendering / priming-side regressions before escalating to post-1.0 pre-filter ranker.** |

Roy-3 produced the A ≈ B ≈ C diagnostic; see [23_roy_3.md § What this proves](../23_roy_3.md) for the disambiguation.

## Prerequisites

- All four wire PRs merged to main: #253 (Wire-A), #254 (Stage 0b/0c telemetry), #255 (Wire 3), #256 (Wire 2), #257 (Wire 1). Verify via `git log --oneline main | head -20`.
- Leader healthy at the URL in `~/.config/maxim/peer.yml` and reachable from the runner host.
- Local checkout has the `bodies/infant_humanoid` component (built-in seed; ships in `src/maxim/_data/components/bodies/`).
- `MAXIM_SUBSTRATE_PATH=1` set in the runner environment (load-bearing per [feedback_substrate_path_env_var_for_roy.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_substrate_path_env_var_for_roy.md)). Without it, the LinguisticEncoder → EC text-modality routing short-circuits silently and Wire-A's annotation reads against a stale agent-wide bias map.

## A. Pre-flight (3 min)

```bash
# 1. Specs parse?
maxim roy run scenarios/roy/roy_3a_iteration.yaml --dry-run
maxim roy run scenarios/roy/roy_3b_iteration.yaml --dry-run

# 2. All four wires landed in current checkout?
git log --oneline main | grep -E "wire-(a|1|2|3)|stage-0b" | head -8
# Expect 5 commits/merges:
#   feat(0.9.1): Wire-A — cluster-bias annotation
#   feat(0.9.1): Stages 0b + 0c — action JSONL telemetry + recommend_action emission
#   feat(0.9.1): Wire 3 — embodiment-state → tool filter
#   feat(0.9.1): Wire 2 — Pavlovian percept aversion
#   feat(0.9.1): Wire 1 — risk-sensitive action annotation

# 3. Leader healthy?
curl -si --max-time 10 \
  -H "Authorization: Bearer $(awk '/api_key:/ {print $2}' ~/.config/maxim/peer.yml)" \
  "$(awk '/url:/ {print $2}' ~/.config/maxim/peer.yml)/models" | head -3
# Expect HTTP/2 200.
```

## B. Roy-3a (≈15-17 min wall)

```bash
# Kill any stale sims first.
pkill -f "maxim.*sim" 2>/dev/null

# LOAD-BEARING runner env (per feedback_substrate_path_env_var_for_roy):
#
# MAXIM_SUBSTRATE_PATH=1   gates MemoryHub._encoder wiring. Without
#   this set, BioEnrichmentPipeline's text-modality EC fire
#   (bio_enrichment.py:576) short-circuits on encoder=None and
#   Wire-A's annotation reads against a stale agent-wide bias map.
#
# MAXIM_EC_TRACE_ACTIVATIONS=1   gates per-tick sim_ec_activation
#   JSONL emission from similarity/ec.py::pattern_complete_or_separate.
#   Roy-3 reads this for the cross-modality split (linguistic vs drive)
#   that the priming-side regression analysis uses as a cross-check.
#
# MAXIM_LOG_FILE=/tmp/roy_3a_ec_trace.jsonl   the JSONL sink that
#   makes Wire-A render checks, Wire 3 WIRE_3_FILTER events, and
#   Stage 0c sim_recommend_action emissions inspectable post-hoc.

MAXIM_SUBSTRATE_PATH=1 \
MAXIM_EC_TRACE_ACTIVATIONS=1 \
MAXIM_LOG_FILE=/tmp/roy_3a_ec_trace.jsonl \
  maxim roy run scenarios/roy/roy_3a_iteration.yaml > /tmp/roy_3a_run.log 2>&1
```

The runner orchestrates: 5 priming stages (`act1_neonatal_a` → `act1_neonatal_b` → `act2_cradle_a` → `act2_cradle_b` → `act3_consolidation`) at 10 turns each under substrate-primary AUT, then 3 arms (a/b/c) at 10 turns each from `roy_1_holdout.yaml` under llm-primary AUT. Per-stage session_ids land in `~/.maxim/roy/roy-3a/result.json`.

## C. Roy-3b (≈14-16 min wall)

```bash
pkill -f "maxim.*sim" 2>/dev/null

MAXIM_SUBSTRATE_PATH=1 \
MAXIM_EC_TRACE_ACTIVATIONS=1 \
MAXIM_LOG_FILE=/tmp/roy_3b_ec_trace.jsonl \
  maxim roy run scenarios/roy/roy_3b_iteration.yaml > /tmp/roy_3b_run.log 2>&1
```

Same priming + arm shape as Roy-3a; fixture is `roy_2pc_holdout.yaml` (engineered food/hunger/eating overlap).

## D. Post-run analysis (5 min)

The Roy-3 disambiguation reads from FOUR data sources. Replay these for each iteration:

### D.1 Per-arm tool counts (the headline behavioral signal)

```bash
ITER=roy-3a  # repeat for roy-3b
for arm in a b c; do
  sid=$(python3 -c "import json; print(json.load(open('$HOME/.maxim/$ITER/result.json'))['arms']['$arm']['session_id'])")
  echo "=== Arm $arm: $sid ==="
  python3 -c "
import json
from collections import Counter
counts = Counter()
for line in open('$HOME/.maxim/sim_reports/$sid/actions.jsonl'):
    d = json.loads(line)
    tool = d.get('tool_name') or d.get('action') or d.get('tool') or '<no-tool>'
    counts[tool] += 1
for k,v in counts.most_common():
    print(f'  {v:3d}× {k}')
"
done
```

**Diagnostic outcome:** count `sense_food_source` occurrences per arm.

### D.2 Wire-A renderable annotation per arm (the prompt-side check)

```bash
ITER=roy-3a  # repeat for roy-3b
for arm in a b c; do
  sid=$(python3 -c "import json; print(json.load(open('$HOME/.maxim/$ITER/result.json'))['arms']['$arm']['session_id'])")
  echo "=== Arm $arm renderable annotation ==="
  python3 -c "
import json
try:
    with open('$HOME/.maxim/sim_reports/$sid/aut_nac.json') as f:
        d = json.load(f)
    crb = d.get('cluster_reward_bias', {})
    if not crb:
        print('  <empty — Wire-A annotation section skipped>')
    else:
        per_tool = {}
        for k,v in crb.items():
            # key shape: agent_id\\x1fcluster_id\\x1ftool:NAME
            parts = k.split(chr(0x1f))
            if len(parts) == 3 and parts[0] == 'default_agent':
                tool = parts[2]
                existing = per_tool.get(tool)
                if existing is None or abs(v) > abs(existing):
                    per_tool[tool] = v
        for tool, bias in sorted(per_tool.items(), key=lambda kv: -abs(kv[1])):
            ab = abs(bias)
            if ab >= 0.5: band = 'strongly rewarding' if bias > 0 else 'strongly aversive'
            elif ab >= 0.1: band = 'mildly rewarding' if bias > 0 else 'mildly aversive'
            else: band = 'neutral / mixed'
            print(f'  {tool:35s}  [{band}]  (bias={bias:+.4f})')
except FileNotFoundError:
    print('  <no aut_nac.json>')
"
done
```

**Diagnostic outcome:** is arm A's annotation in the "strongly rewarding" band? If max(|bias|) < 0.1, the wire renders "neutral / mixed" and conveys no signal — this is the Roy-3 finding.

### D.3 Stage 0c `sim_recommend_action` gate-pass distribution (priming-side health check)

```bash
ITER=roy-3a  # repeat for roy-3b; uses /tmp/roy_3{a,b}_ec_trace.jsonl
JSONL=/tmp/${ITER//-/_}_ec_trace.jsonl
python3 -c "
import json
from collections import Counter
counts = {'total': 0, 'passed': 0, 'gated': 0}
passed_tools = Counter()
for line in open('$JSONL'):
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    if d.get('subsystem') == 'NAc_RECOMMEND':
        counts['total'] += 1
        if d.get('passed_gate'):
            counts['passed'] += 1
            passed_tools[d.get('best_tool') or '<none>'] += 1
        else:
            counts['gated'] += 1
print(f'NAc_RECOMMEND total={counts[chr(34)+\"total\"+chr(34)]} passed={counts[chr(34)+\"passed\"+chr(34)]} gated={counts[chr(34)+\"gated\"+chr(34)]}')
print('  passed_gate=True best_tool distribution:')
for tool, c in passed_tools.most_common(8):
    print(f'    {c:4d}× {tool}')
"
```

**Diagnostic outcome:** does the substrate-primary priming proposer fire `sense_food_source` proposals that pass the gate? Roy-3 shows 800-811 passes per iteration, all `sense_food_source` — the wire is healthy at the priming-side consumer level.

### D.4 Priming-side cluster_reward_bias count (the structural anomaly check)

```bash
ITER=roy-3a  # repeat for roy-3b
priming_sid=$(python3 -c "import json; print(json.load(open('$HOME/.maxim/$ITER/result.json'))['priming']['final_session_id'])")
echo "Roy-${ITER#roy-} priming NAc ($priming_sid):"
python3 -c "
import json
with open('$HOME/.maxim/sim_reports/$priming_sid/aut_nac.json') as f:
    d = json.load(f)
crb = d.get('cluster_reward_bias', {})
print(f'  cluster_reward_bias entries: {len(crb)}')
food = sorted([(k.split(chr(0x1f))[1][:8], v) for k,v in crb.items() if 'sense_food_source' in k], key=lambda x: -x[1])
for cid, v in food:
    print(f'    {cid}…  sense_food_source: {v:+.4f}')
"
```

**Diagnostic outcome:** how many distinct EC cluster IDs did substrate-primary priming fan out across? Roy-2 / Roy-2c / Roy-4 / Roy-5a all wrote 6 keys at +1.0; Roy-3a / Roy-3b wrote 2 keys at non-saturated values. This is the priming-side regression Roy-3 surfaces — the candidate cause is one of the wire merges between 5/13 and 5/22.

## Expected output (Roy-3 actual)

| Iteration | Wall | Arms | sense_food_source per arm | Arm A annotation band |
|---|---|---|---|---|
| Roy-3a | 952.5s | 3 × 10 turns | 0 / 0 / 0 | `neutral / mixed` (max bias 0.036) |
| Roy-3b | 879.9s | 3 × 10 turns | 0 / 0 / 0 | `neutral / mixed` (max bias 0.098) |

All six arms × two fixtures produced zero `sense_food_source` calls. Arm A's Wire-A annotation rendered under the "neutral / mixed" band in both iterations — a structurally null signal.

## Cross-iteration notes

Roy-3 uses the **same** priming arc structure as Roy-2 / Roy-2c / Roy-4 / Roy-5a. The single structural change vs Roy-2 is the four annotation wires shipped between 5/13 and 5/22. The priming-side cluster_reward_bias regression (6 saturated keys → 2 partially-saturated keys) is the load-bearing finding; rooting-cause it requires a follow-up Roy-2-shaped iteration on each intermediate commit (Wire-A → Stage 0b/c → Wire 3 → Wire 2 → Wire 1).

## Troubleshooting

- **`aut_nac.json` missing from an arm session_dir** — sim aborted mid-run. Check `/tmp/roy_3*_run.log` for tracebacks. Re-run after fixing.
- **Arm A's renderable annotation lists tools other than `sense_food_source`** — priming arc didn't fire `sense_food_source` enough during the substrate-primary phase. Re-run; if reproducible, inspect the priming arc YAML. The `cradle` and `cradle_prelinguistic` arcs are tuned to surface food affordances when the hunger drive is elevated.
- **Different priming `cluster_reward_bias` count than 2 (Roy-3 era) or 6 (pre-5/22 era)** — likely indicates either (a) priming was longer / shorter than the 5-stage × 10-turn budget the YAML specifies, or (b) the substrate-primary AUT path has changed since this protocol was written. The 6-key pre-5/22 baseline is stable across four iterations; the 2-key Roy-3 baseline is stable across two iterations.
- **NAc_RECOMMEND emissions show passed_gate=True with best_tool=<none>** — would indicate a Stage 0c emission bug. Roy-3 should have all passed_gate=True emissions key on `sense_food_source` (the only tool the substrate-primary AUT proposes during priming).
- **WIRE_3_FILTER count > 0 on Roy-3** — would indicate a damaged component on `bodies/infant_humanoid`. Roy-3 should have 0 WIRE_3_FILTER events; the body is intact throughout.
