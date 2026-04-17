# Experiment 3 (Tier 2) — Reproduction Protocol

**Experiment:** [behavioral_convergence_exp3_tier2.md](../behavioral_convergence_exp3_tier2.md)

## Quick verification

```bash
# Tier 1 only (~0.5s, no LLM):
PYTHONPATH=src python scripts/behavioral_convergence_exp3_tier2.py --tier1-only

# Full Tier 2 (~60-120s, requires leader with qwen2.5-14b or similar):
PYTHONPATH=src python scripts/behavioral_convergence_exp3_tier2.py --model qwen2.5-14b
```

## Prerequisites for Tier 2

- Leader online with LLM loaded (`maxim peer llm --status`)
- Peer config exists (`~/.maxim/peer.yml` or env vars)
- Network connectivity to leader (`maxim peer version`)

## Interpreting results

**Tier 1 (10 substrate hypotheses):**
- Teal vial should have strongest positive valence (>0.3)
- Purple vial positive but less than teal
- Orange vial negative (<-0.1)
- Poisoned cue: teal valence > purple valence
- Control: all zero

**Tier 2 (2 LLM hypotheses):**
- Experienced agent should prefer teal vial MORE than fresh agent
- Experienced agent should rarely/never pick orange vial

**Expected:** 12/12 PASS. Experienced: 8-10/10 teal. Fresh: ~uniform or biased toward first item.

## If Tier 2 hypotheses fail

1. **LLM always picks same vial regardless of valence:** Check that valence context is actually in the prompt (run with `--json` and inspect `tier2_prompt_experienced`). The strength differentiation ("VERY GOOD" vs "good") is critical — flat labels don't work.

2. **LLM picks randomly even with valence:** Try a stronger model or lower temperature. qwen2.5-14b at 0.3 temperature is the validated config.

3. **Experienced = fresh:** Check persistence — the hippocampus/NAc save/load may have failed. Inspect the persist dir files.

## Experimental controls

- **Positional bias control:** Vial order shuffled per trial (seed: `turn * 7 + 13`)
- **Language prior control:** Vial names are arbitrary visual attributes (purple/hexagonal/glass, teal/cylindrical/ceramic, orange/triangular/crystal) — no LLM pretraining about what these do
- **Deterministic training:** Same 5 episodes, same reactions, same valence every run
- **Temperature:** 0.3 (low enough for consistency, high enough for variation)
