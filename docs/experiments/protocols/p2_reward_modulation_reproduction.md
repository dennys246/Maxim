# Substrate P2 Reward-Modulation Sweep — Reproduction Protocol

**Status:** Active protocol, minimal scope
**Purpose:** Reproduce the Stage 3 real-embedding sweep result on fresh hardware or after any change that might affect recognition geometry (encoder swap, EC threshold logic, NAc bias logic, fixture edits, sentence-transformer model upgrade).
**Expected runtime:** ~30s on an M3 Mac (including model cold-load); ~2 minutes including setup.

## Background

Stage 3 closed substrate P2 for the 0.3-minimum gate with:
- mean target gain **+56.0 ± 29.0 pp**
- mean distractor drift **0.0 ± 0.0 pp**
- mean target monotone fraction **94%**
- **9 of 10 seeds** pass individually

Full methodology + result archive: [../p2_reward_modulation_sweep.md](../p2_reward_modulation_sweep.md). Raw numbers: [../results/p2_reward_modulation_sweep.json](../results/p2_reward_modulation_sweep.json).

This protocol exists so a future session can verify the result hasn't drifted without re-discovering the operating point or the metric design decisions. If the reproduction fails, start by reading the Stage 3 sweep doc — the "Iterations" table and pivot history explain every decision that was made to reach the passing configuration.

## Prerequisites

**Code:**
- `main` at or after the Substrate P2 Stage 3 merge
- `pymaxim` importable from the checkout (`PYTHONPATH=src` or editable install)

**Dependencies:**
- `sentence-transformers` (the `pymaxim[semantic]` extra)
- ~500 MB disk for the `paraphrase-mpnet-base-v2` model cache

**Hardware:**
- Any CPU is sufficient. M3 Mac: ~27s full sweep wall clock. x86 laptops typically 30–60s.
- No GPU required. No network required after the initial model download.

**Fixtures:**
- `scenarios/substrate/p2_reward_modulation.yaml` (v3; 10 clusters, 50 sentences)
- Expected file hash is tracked implicitly by the Stage 3 commit — any fixture edit will change the numbers, which is the point.

## Setup (one-time)

```bash
pip install 'pymaxim[semantic]'
```

First run downloads `paraphrase-mpnet-base-v2` (~420 MB) into the HuggingFace cache. Subsequent runs reuse it.

## Running the sweep

### Full 10-seed gate

```bash
python -m pytest tests/substrate/test_p2_reward_modulation.py::TestP2ValidationSweep::test_sweep_10_seeds -xvs
```

**Pass criteria** (enforced by the test):
- Mean target gain ≥ +30 pp
- Mean distractor drift ≤ 5 pp
- Mean target monotone fraction ≥ 50%

**Expected output** (exact numbers may vary ±2 pp due to shuffled-order interactions with the mpnet tokenizer, but the aggregate gate should clear with >20 pp margin):

```
P2 Reward Modulation Sweep — PASS
  Model: paraphrase-mpnet-base-v2 @ 0.7
  Target gain:       +56.0% pp ± 29.0% (need >=+30 pp)
  Distractor drift:  0.0% pp ± 0.0% (need <=5 pp)
  Target monotone:   94% of clusters (need >=50%)
  Seeds passing individually: 9/10
```

**Result JSON** written to `docs/experiments/results/p2_reward_modulation_sweep.json` — per-seed breakdown + all summary stats, always overwritten with the latest run. If you re-run after a code change, `git diff docs/experiments/results/p2_reward_modulation_sweep.json` is a compact way to see what moved.

### Fast single-seed smoke (for iteration during development)

```bash
python -m pytest tests/substrate/test_p2_reward_modulation.py::TestP2ValidationSweep::test_single_seed -xvs
```

~5s. Loads the model once, runs both baseline and rewarded passes at seed 0, prints per-cluster table. Does NOT enforce the pass gate — use this for "is the mechanism alive at all" checks.

### Mechanism tests only (no sentence-transformers required)

```bash
python -m pytest tests/substrate/test_p2_reward_modulation.py::TestP2Mechanism -xvs
```

~4s. Validates the metric + reward-bias wiring on synthetic embeddings. Good first check if you're on a machine without `sentence-transformers` installed.

## What to do if it fails

**Aggregate gate clears but 2–3 seeds fail individually:** this is the known Stage 3 residual — mpnet embedding adjacencies at threshold 0.70 occasionally let a hard paraphrase leak onto a neighboring reward-widened node. Seed 3 and seed 5 were the known failures at Stage 3 commit time. If you see the same seeds failing with similar gains, the result is stable. If NEW seeds fail below +30 pp, that's a drift signal — something changed in the encoding path or the fixture.

**Distractor drift > 5 pp:** the plurality-ownership metric should keep this at 0.0 pp in all normal cases. Non-zero distractor drift means either (a) the fixture was edited to introduce a pairwise-close target/distractor pair, or (b) `NAc.reward_bias_alpha` / `max_reward_bias` was increased and the widened radius now engulfs neighbor clusters. Check `git log -- src/maxim/decisions/nac.py tests/substrate/p2_metrics.py scenarios/substrate/p2_reward_modulation.yaml` before touching anything else.

**Aggregate target gain < +30 pp:** three root causes seen during Stage 3 development, in order of likelihood:

1. **RC2 regression — `if self._nac` truthy check in `similarity/encoder.py`**. `NAc` has `__len__` over causal links, so an empty NAc evaluates as falsy and the reward-override branch silently skips. Fix lives in [src/maxim/similarity/encoder.py](../../../src/maxim/similarity/encoder.py) — the line must be `if self._nac is not None else None`, NOT `if self._nac else None`. This regressed once during Stage 3 when the test harness picked up a shadowed pre-Stage-2 install of the package; the symptom was +0 pp gain across every threshold.
2. **Metric regression — raw pair-collapse.** If `tests/substrate/p2_metrics.py::_cluster_self_collapse_rates` has been replaced with `_collapse_rate` (raw), the metric will spuriously credit cross-cluster contamination as "collapse" and produce unstable seed-to-seed numbers. Plurality-ownership is the invariant — see the `p2_metrics.py` module docstring and the `load_bearing_invariants` list in [../p2_reward_modulation_sweep.md](../p2_reward_modulation_sweep.md).
3. **Fixture edit in a semantically-close target.** Any new target cluster added without a solo-target probe (see below) risks cross-contamination. Stage 3 ships with 10 clusters pre-probed for pairwise semantic distance.

## Threshold exploration (for methodology changes)

If you're changing the encoder model, NAc alpha, or adding targets to the fixture, re-run the threshold sweep to find the operating point:

```bash
PYTHONPATH=src python <<'EOF'
from tests.substrate.p2_metrics import compute_p2_metrics, load_clusters_with_targets
from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder
from maxim.memory.atl import ATL
from maxim.decisions.nac import NAc

clusters, targets = load_clusters_with_targets('scenarios/substrate/p2_reward_modulation.yaml')

def make(thresh):
    ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=thresh))
    atl = ATL()
    nac = NAc()
    enc = LinguisticEncoder(ec=ec, atl=atl, nac=nac,
                            config=EncoderConfig(model_name='paraphrase-mpnet-base-v2'))
    return enc, ec, atl, nac

print(f'{"thresh":>7} {"base":>7} {"rew":>7} {"gain":>7} {"drift":>6} {"mono":>5} pass')
print('-' * 60)
for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    m = compute_p2_metrics(
        clusters=clusters, target_clusters=targets, make_encoder_nac=make,
        threshold=threshold, reward=2.0, shuffle=True, seed=0,
    )
    p = 'PASS' if m.passes_p2() else 'FAIL'
    print(f'{threshold:>7.2f} {m.baseline_target_mean:>6.1%} {m.rewarded_target_mean:>6.1%} '
          f'{m.target_gain:>+6.1%} {m.distractor_drift:>5.1%} {m.target_monotone_fraction:>4.0%}  {p}')
EOF
```

Pick the threshold in the middle of the pass band. Too low leaves residual baseline collapse that dilutes the gain signal; too high drives baseline to zero and leaves no operating margin.

## Solo-target probe (for fixture changes)

If you're adding a new target cluster, verify it passes a solo-target probe before shipping. A cluster that fails this probe will cross-contaminate when rewarded alongside other targets:

```bash
PYTHONPATH=src python <<'EOF'
from tests.substrate.p2_metrics import compute_p2_metrics, load_clusters_with_targets
from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder
from maxim.memory.atl import ATL
from maxim.decisions.nac import NAc

clusters, _ = load_clusters_with_targets('scenarios/substrate/p2_reward_modulation.yaml')

def make(thresh):
    ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=thresh))
    atl = ATL()
    nac = NAc()
    enc = LinguisticEncoder(ec=ec, atl=atl, nac=nac,
                            config=EncoderConfig(model_name='paraphrase-mpnet-base-v2'))
    return enc, ec, atl, nac

print(f'{"target":<20} {"base":>7} {"rew":>7} {"gain":>7} {"drift":>6}')
print('-' * 55)
for c in clusters:
    t = c['name']
    m = compute_p2_metrics(
        clusters=clusters, target_clusters=[t], make_encoder_nac=make,
        threshold=0.70, reward=2.0, shuffle=True, seed=0,
    )
    base = m.baseline_rates[t]; rew = m.rewarded_rates[t]
    print(f'{t:<20} {base:>6.1%} {rew:>6.1%} {rew-base:>+6.1%} {m.distractor_drift:>5.1%}')
EOF
```

A cluster passes the solo-target probe if its solo `target_gain >= +30 pp` AND `distractor_drift <= 5 pp`. Clusters that fail either check must be rejected from the fixture before shipping — adding them will introduce flaky runs at the 10-seed aggregate level.

## Related docs

- [../p2_reward_modulation_sweep.md](../p2_reward_modulation_sweep.md) — Stage 3 lab notebook with full methodology + pivot history + results table
- [../p2_sem_pain_cascade.md](../p2_sem_pain_cascade.md) — Stage 2 SEM pain cascade integration PoC (sibling protocol)
- [../p1_recognition_sweep.md](../p1_recognition_sweep.md) — P1 methodology, which P2's collapse-rate metric mirrors
- [../../plans/substrate_recognition.md](../../plans/substrate_recognition.md) — the plan this closes
- `tests/substrate/p2_metrics.py` module docstring — the three-iteration metric history (node count → raw pair-collapse → plurality self-collapse)
