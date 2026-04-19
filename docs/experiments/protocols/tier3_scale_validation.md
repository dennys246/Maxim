# Tier 3 Scale Validation — Experiment Protocol

**Experiment:** Run organic LLM learning (Exp 4, Tier 3) at 20+ independent seeds to prove the learning effect is statistically robust.

## Quick run

```bash
# Full scale run (~60-100 min, requires leader with qwen2.5-14b):
PYTHONPATH=src python scripts/behavioral_convergence_exp4_scale.py --seeds 20

# Resume from a partial run:
PYTHONPATH=src python scripts/behavioral_convergence_exp4_scale.py --resume-dir /path/to/output
```

## Prerequisites

- Leader online with LLM loaded (`maxim peer llm --status`)
- Network connectivity to leader (`maxim peer version`)
- scipy installed (for statistical tests)

## What each seed does

Each of the 20+ seeds runs the full Exp 4 pipeline independently:

1. **Session 1 (exploration):** Fresh agent, no bio-state. Random exploration.
2. **Session 2 (early learning):** Reloaded from Session 1. Should show preference shift.
3. **Session 3 (convergence):** Reloaded from Session 2. Should converge to teal (antidote).
4. **Fresh control:** No bio-state. Baseline comparison.

Bio-state (hippocampus + NAc) is saved/loaded between sessions within each seed, but seeds are fully isolated from each other.

## Primary metrics

- **Teal rate per session:** fraction of choices that select the antidote vial
- **S3-S1 improvement:** paired difference in teal rate between Session 3 and Session 1
- **Control comparison:** experienced S3 vs fresh control

## Pass gates

| Gate | Criterion | Rationale |
|------|-----------|-----------|
| Mean S3 teal rate | >= 70% | Strong convergence on average |
| Mean S3-S1 improvement | > 0% | Learning occurred |
| Wilcoxon signed-rank (S3 > S1) | p < 0.05 | Statistically significant |
| S3 escape rate | >= 80% | Most seeds converge to escape |
| Control death rate | >= 60% | Fresh agents reliably fail |
| S3 teal > control teal | mean comparison | Experienced beats fresh |

## Statistical tests

- **Wilcoxon signed-rank test:** paired, one-sided (S3 > S1). Non-parametric — no normality assumption on teal rates (they're bounded [0, 1]).
- **Mann-Whitney U:** unpaired, one-sided (S3 > control). Different sample sizes possible if some controls have different choice counts.

## Sources of variance

The only source of per-seed variance is **LLM sampling noise** (temperature 0.4). Vial order shuffling is deterministic per turn (`random.Random(turn * 7 + 13)`). This is intentional — we're measuring whether the bio-system's learned valence is strong enough to overcome LLM sampling noise across 20+ independent trials.

## Resume support

Results are saved incrementally to `partial_results.jsonl` in the output directory. If a run is interrupted, use `--resume-dir` to continue from where it stopped. Completed seeds are skipped automatically.

## Output

- `partial_results.jsonl` — one line per completed seed (incremental)
- `scale_results.json` — full aggregate results with statistics
- `seed_NNN/` — per-seed persist directories (hippocampus.json, nac.json)

## If gates fail

1. **Mean S3 teal rate < 70%:** Check if valence context is being surfaced in the prompt. Inspect a failing seed's persist dir for hippocampus/NAc state. The bio-pipeline may not be producing strong enough valence signal.

2. **Wilcoxon p >= 0.05:** May need more seeds (try 30). Or the effect size is smaller than expected — check the improvement distribution for outlier seeds that regress.

3. **Control death rate < 60%:** The LLM may have a language prior about teal/antidote despite masked names. Check the control choices — if teal is selected by name preference (not learning), the masking is insufficient.

4. **S3 escape rate < 80%:** Some seeds may get stuck in a local minimum (always picking purple/heal but never finding the antidote). Check if those seeds' Session 2 ever tried teal — if not, the exploration in Session 1 may have been too narrow.
