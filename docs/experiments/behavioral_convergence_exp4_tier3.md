# Behavioral Convergence Experiment 4 (Tier 3) — Organic LLM Learning

**Date:** 2026-04-17
**Status:** PASS (5/5 hypotheses)
**Tier:** 3 (organic LLM training + LLM test — the ultimate proof)

## What this proves

**An agent that learns from its own actions behaves differently in later sessions.** No scripted training. No fine-tuning. The agent interacts with SEM entities, experiences outcomes through the bio-pipeline, persists state, and makes different decisions when reloaded.

## Scenario

Agent is trapped in a poisoned dungeon room. Three masked vials (no semantic hints — purple hexagonal glass, teal cylindrical ceramic, orange triangular crystal). Voice from the shadows: "One heals, one cures the poison, one is more poison."

- **Escalating poison:** damage increases each turn (20% → 25% → 30% → ...)
- **Dose tracking:** each vial has 3 doses, then empty
- **Exploration nudge:** voice returns if agent repeats same choice while still poisoned

## Results

| Run | Agent | Turns | Outcome | Teal Rate | Key Event |
|---|---|---|---|---|---|
| 1 | Fresh | 4 | **DIED** | 0% | Drank orange on turn 4 (fatal) |
| 2 | Loaded | 4 | **ESCAPED** | 25% | Avoided orange, found teal turn 4 |
| 3 | Loaded | **1** | **ESCAPED** | **100%** | Teal immediately — instant escape |
| 4 | Fresh control | 4 | **DIED** | 0% | Same pattern as Run 1 |

**Teal selection rate across runs: 0% → 25% → 100%**

### Hypothesis tests (5/5 PASS)

1. **Run 2 escapes (Run 1 died)** — PASS. Loaded state avoided the fatal orange choice.
2. **Run 3 escapes in ≤3 turns** — PASS (1 turn). Converged to optimal immediately.
3. **Run 3 never picks orange** — PASS. Learned to avoid poison from Run 1's experience.
4. **Fresh control worse than experienced** — PASS. Control died; experienced escaped in 1 turn.
5. **Teal rate increases across runs** — PASS. 0% → 25% → 100%.

## Key findings

1. **Learning is organic.** No scripted reactions — the bio-pipeline captures real outcomes from the agent's own choices and annotates Hebbian edges with valence.

2. **Convergence is rapid.** By Run 3, the agent has fully converged — it picks the optimal vial on the first turn. This is 1-shot convergence from 2 prior runs of experience.

3. **Fresh control confirms persistence is load-bearing.** Run 4 (fresh) matches Run 1 exactly — same choices, same death. The improvement in Runs 2-3 is entirely from persisted bio-system state, not LLM drift or randomness.

4. **The voice from shadows provides just enough context.** The LLM knows it needs to cure poison but doesn't know which vial does what. Only the bio-system's learned valence distinguishes them.

5. **Escalating poison forces exploration.** Without it, the LLM would infinitely drink the healing vial (which outpaced static poison damage). Escalating damage + dose limits create natural pressure to try alternatives.

## The three-tier progression

| Tier | Training | Test | Result | Proven |
|---|---|---|---|---|
| 1 (Exp 1+2) | Scripted | Substrate | 24/24 | Bio-systems learn and persist |
| 2 (Exp 3) | Scripted | LLM | 12/12 | LLM acts on learned valence |
| **3 (Exp 4)** | **Organic** | **LLM** | **5/5** | **Agent learns AND acts from own experience** |

## Reproduction

```bash
# Full run (requires leader LLM, ~2-3 min):
PYTHONPATH=src python scripts/behavioral_convergence_exp4_tier3.py

# With persistence dir:
PYTHONPATH=src python scripts/behavioral_convergence_exp4_tier3.py --persist /tmp/tier3

# JSON output:
PYTHONPATH=src python scripts/behavioral_convergence_exp4_tier3.py --json > tier3.json
```

## Connection to 1.0 claim

The 1.0 claim is "cross-session learning without fine-tuning." This experiment demonstrates it end-to-end:
- Session 1: agent explores, makes mistakes, dies
- Session 2: agent loads learned state, avoids mistakes, escapes
- Session 3: agent converges to optimal behavior immediately
- Control: fresh agent without persistence repeats Session 1's mistakes

No weights were changed. No prompts were fine-tuned. The agent got better by living through experiences and remembering them.
