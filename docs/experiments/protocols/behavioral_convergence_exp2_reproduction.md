# Experiment 2 — Reproduction Protocol

**Experiment:** [behavioral_convergence_exp2.md](../behavioral_convergence_exp2.md)
**Plan:** [behavioral_convergence_wiring.md](../../plans/archive/behavioral_convergence_wiring.md)

## Quick verification (~0.5s, no LLM)

```bash
PYTHONPATH=src python scripts/behavioral_convergence_exp2.py
```

Expected: `ALL HYPOTHESES CONFIRMED` (13/13).

## With persistence directory

```bash
PYTHONPATH=src python scripts/behavioral_convergence_exp2.py --persist /tmp/exp2_run1
# Inspect saved state:
cat /tmp/exp2_run1/hippocampus.json | python3 -m json.tool | head -50
cat /tmp/exp2_run1/nac.json | python3 -m json.tool | head -50
```

## JSON output for analysis

```bash
PYTHONPATH=src python scripts/behavioral_convergence_exp2.py --json > exp2_results.json
python3 -c "import json; d=json.load(open('exp2_results.json')); print(json.dumps(d['session_2'], indent=2))"
```

## What to check if hypotheses fail

1. **Energy bridge reactions don't fire:** Check `energy/reactions.py::EnergyReactionBridge.check()` — verify threshold logic and that ReactionBus is wired.
2. **Food/water valence is zero:** Check that `hippocampus.capture_reaction` is subscribed to the ReactionBus AND that `finalize()` computes valence from the reactions list.
3. **Poison has positive valence:** Check whether satiation reactions from the energy bridge are bleeding into poison's episode. The eligibility decay between entities should isolate them.
4. **NAc biases are all zero:** Check `distribute_reward` subscriber — requires `agent_id` on reaction context. Energy bridge reactions use the configured `agent_id`.
5. **Control is non-zero:** Check that the control agent has no reactions (only concept nodes, no interactions).
