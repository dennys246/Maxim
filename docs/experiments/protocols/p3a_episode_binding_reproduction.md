# P3a Stage 2 reproduction runbook

**Plan:** [substrate_p3a_episode_binding.md](../../plans/archive/substrate_p3a_episode_binding.md)
**Results:** [p3a_episode_binding_sweep.md](../p3a_episode_binding_sweep.md) + [results/p3a_episode_binding_sweep.json](../results/p3a_episode_binding_sweep.json)

This runbook reproduces the 10-seed Hebbian multi-hop vs TF-IDF head-to-head and the persistence round-trip checks from scratch. Every step is deterministic — a faithful re-run should produce **byte-identical** numbers to the results JSON.

## Prerequisites

- Clean Maxim checkout (main at or after PR #109 substrate-p3-kickoff merged).
- Python 3.12 in a worktree / venv with `pip install -e .` run once.
- No extras required — Stage 2 uses synthetic node IDs and does not touch `LinguisticEncoder`, sentence-transformers, or any optional extras.

## Step 1 — Regenerate the fixture (optional)

The fixture is deterministic given the generator source, so re-running the generator should produce the same YAML byte-for-byte. Skip this step unless you're verifying the generator itself.

```bash
PYTHONPATH=src python tests/substrate/p3a_fixture_gen.py \
  --out scenarios/substrate/synthetic_episodes.yaml \
  --seed 0
```

Expected output:

```
wrote fixture to scenarios/substrate/synthetic_episodes.yaml (seed=0)
```

The YAML should be ≈800 lines. A quick sanity check:

```bash
grep -c "^  - id:" scenarios/substrate/synthetic_episodes.yaml   # → 170
grep -c "kind: hub$" scenarios/substrate/synthetic_episodes.yaml # → 80
grep -c "kind: chain$" scenarios/substrate/synthetic_episodes.yaml # → 60
grep -c "kind: peripheral$" scenarios/substrate/synthetic_episodes.yaml # → 30
```

`80 + 60 + 30 = 170` matches the expected 17 episodes/topic × 10 topics.

## Step 2 — Run the Stage 2 validation test suite

```bash
PYTHONPATH=src python -m pytest tests/substrate/test_p3a_fixture_validation.py -v
```

Expected: **12 tests pass** (0.5–2s wall-clock). All test classes:

- `TestStage2PassGate` — the plan's precision/recall + 2σ head-to-head gate
- `TestOneHopArchitecturalFinding` — locks in the one-hop ≈ TF-IDF parity
- `TestRankingStability` — tie-free stratification + shuffle invariance
- `TestFixturePersistenceRoundTrip` — dump/load + binding-graph rebuild
- `TestFixtureShape` — fixture sanity
- `TestEpisodeConfigRetrievalDefaults` — config plumbing

## Step 3 — Regenerate the results JSON

The results artifact is produced by a standalone generator script that uses the same code paths as the tests:

```bash
PYTHONPATH=src python <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.episode import CaptureEvent
from tests.substrate.p3a_fixture_gen import build_fixture
from tests.substrate.tfidf_baseline import TfidfBaseline
from tests.substrate.p3a_metrics import run_probes, aggregate_seeds, compare_to_baseline


def ingest(h, eps):
    tick = 0
    for ep in eps:
        h.observe_episode_event(CaptureEvent(
            tick=tick, channel="text",
            activated_nodes=tuple(ep["activated_nodes"]),
        ))
        tick += 1000
    h.finalize_pending_episode()


seeds = list(range(10))
multihop, onehop, tfidf_runs = [], [], []

for seed in seeds:
    f = build_fixture(seed=seed)
    h = Hippocampus(HippocampusConfig())
    ingest(h, f["episodes"])
    tfidf = TfidfBaseline.from_episodes(f["episodes"])
    multihop.append(run_probes(
        lambda c, k: h.retrieve_on_cue(c, k, multi_hop=True),
        f["probes"], seed=seed,
    ))
    onehop.append(run_probes(
        lambda c, k: h.retrieve_on_cue(c, k, multi_hop=False),
        f["probes"], seed=seed,
    ))
    tfidf_runs.append(run_probes(tfidf.retrieve, f["probes"], seed=seed))

hebbian_agg = aggregate_seeds(multihop)
tfidf_agg = aggregate_seeds(tfidf_runs)
cmp = compare_to_baseline(hebbian_agg, tfidf_agg)

print(f"Hebbian multi-hop: F1 = {hebbian_agg.mean_f1:.4f} ± {hebbian_agg.std_f1:.4f}")
print(f"TF-IDF baseline:   F1 = {tfidf_agg.mean_f1:.4f} ± {tfidf_agg.std_f1:.4f}")
print(f"Margin: {cmp.margin:.4f}  beats_baseline: {cmp.beats_baseline}")
PY
```

Expected output:

```
Hebbian multi-hop: F1 = 1.0000 ± 0.0000
TF-IDF baseline:   F1 = 0.7000 ± 0.0000
Margin: 0.3000  beats_baseline: True
```

If the numbers diverge, something has drifted. Likely suspects:

1. **`EpisodeConfig` retrieval defaults changed** — check `hippocampus.py::EpisodeConfig` for `retrieval_decay=0.7, retrieval_threshold=0.001, retrieval_max_depth=5`. A lower decay or higher threshold will cut off deep-chain targets.
2. **`hebbian_delta` changed** — if delta drops below ~0.05, the core-vs-peripheral weight gap closes and ranking starts tying.
3. **Pair enumeration order** — `apply_hebbian_on_close` uses `itertools.combinations` (unordered). A regression to ordered pairs would double-apply delta.
4. **`DependencyGraph.spreading_activation` semantics** — the multi-hop retrieval uses the `max()` path aggregation in `agents/bus.py`. If that changes to sum, weights shift.
5. **Fixture generator drift** — run `grep -c "kind: hub$"` etc. above; if the episode counts differ, the reinforcement invariant is broken.

## Step 4 — Spot-check the sample probe

For a quick sanity read without running the full sweep:

```bash
PYTHONPATH=src python -c "
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.episode import CaptureEvent
from tests.substrate.p3a_fixture_gen import build_fixture

f = build_fixture(seed=0)
h = Hippocampus(HippocampusConfig())
tick = 0
for ep in f['episodes']:
    h.observe_episode_event(CaptureEvent(tick=tick, channel='text', activated_nodes=tuple(ep['activated_nodes'])))
    tick += 1000
h.finalize_pending_episode()

# Chain-head probe: should get 4 chain targets strictly above 3 peripherals
print('cue = cooking.prep')
for node, weight in h.retrieve_on_cue('cooking.prep', limit=10, multi_hop=True):
    print(f'  {weight:.4f}  {node}')
"
```

Expected output:

```
cue = cooking.prep
  0.2800  cooking.stove
  0.2800  cooking.saute
  0.0784  cooking.simmer
  0.0784  cooking.plate
  0.0588  cooking.garlic
  0.0588  cooking.onion
  0.0588  cooking.tomato
```

**The load-bearing invariant:** all 4 chain targets (stove, saute, simmer, plate) must strictly outrank all 3 peripherals. Any tie or reversal is a regression.

## Step 5 — Verify persistence round-trip

```bash
PYTHONPATH=src python -c "
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.episode import CaptureEvent
from tests.substrate.p3a_fixture_gen import build_fixture

f = build_fixture(seed=0)

# Pre-dump
h1 = Hippocampus(HippocampusConfig())
tick = 0
for ep in f['episodes']:
    h1.observe_episode_event(CaptureEvent(tick=tick, channel='text', activated_nodes=tuple(ep['activated_nodes'])))
    tick += 1000
h1.finalize_pending_episode()
pre = h1.retrieve_on_cue('cooking.prep', limit=10, multi_hop=True)

# Dump → fresh instance → load_state
dumped = h1.dump()
h2 = Hippocampus(HippocampusConfig())
h2.load_state(dumped)
post = h2.retrieve_on_cue('cooking.prep', limit=10, multi_hop=True)

print('pre-dump:', pre)
print('post-load:', post)
assert pre == post, 'ROUND-TRIP DRIFT'
print('OK — byte-identical round-trip')
"
```

## Step 6 — Full substrate subset check

To verify Stage 2 did not regress anything Stage 1 or P3.5 ships:

```bash
PYTHONPATH=src python -m pytest \
  tests/substrate/test_p3a_episode_binding.py \
  tests/substrate/test_p3a_fixture_validation.py \
  tests/unit/test_bio_system_snapshot.py \
  -q
```

Expected: **88 tests pass** (24 P3a Stage 1 + 12 P3a Stage 2 + 52 P3.5 Stage 1).

## When this protocol will need updating

- **When P3b ships channel integration rules:** the fixture will get per-channel variants and the probe count will grow. Re-baseline the results JSON.
- **When P4 ships cross-modal:** the fixture grows a vision-modality dimension. Cross-modal retrieval gets its own sweep; this runbook stays as-is for the same-modality regression baseline.
- **If `retrieve_on_cue` signature changes** (e.g., cross-modal retrieval adds a `modality` filter): update the inline reproduction snippets in Steps 3–5.
- **If `EpisodeConfig` retrieval defaults are re-tuned:** re-run Step 3 and update the results JSON. Document the tuning change in the sweep writeup.

## Known flake surface

- None observed. All runs so far are byte-exact. If you see seed-to-seed variance, check whether you accidentally enabled a source of nondeterminism (e.g., a thread racing the capture worker, or hash randomization leaking into dict ordering — Python 3.7+ dict preserves insertion order, but `set` iteration is insertion-ordered only in CPython and should not be relied on elsewhere).
