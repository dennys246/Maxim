# P4 Stage 3 — Cross-Modal Head-to-Head Reproduction Protocol

## Prerequisites

- Python 3.12+
- `pymaxim` with `semantic` extra installed (`pip install -e ".[semantic]"`)
- Flowers-102 dataset cached at `~/.cache/maxim/p4_flowers/` (download via `scripts/p4_clip_calibration_sweep.py` on first run)
- No GPU required — paraphrase-mpnet and clip-ViT-B-32 run on CPU/MPS

## Step 1: Verify fixture integrity

```bash
PYTHONPATH=src python -c "
from tests.substrate.p4_fixture_loader import compute_fixture_sha256, FIXTURE_SHA256
actual = compute_fixture_sha256()
assert actual == FIXTURE_SHA256, f'SHA mismatch: {actual} != {FIXTURE_SHA256}'
print(f'Fixture SHA OK: {actual}')
"
```

Expected: `Fixture SHA OK: 74a8201c...`

## Step 2: Run the sweep

```bash
PYTHONPATH=src python scripts/p4_cross_modal_sweep.py
```

Expected runtime: ~60-90s on Apple Silicon (MPS), ~2-3 min on CPU.

Expected output:
```
VERDICT: PASS
Arm B F1: 1.0000, Arm C F1: 0.8200, B-C delta: +0.1800
```

## Step 3: Verify outputs

```bash
# Check results files exist
ls docs/experiments/p4_cross_modal_sweep.md
ls docs/experiments/results/p4_cross_modal_sweep.json

# Verify JSON schema
python -c "
import json
with open('docs/experiments/results/p4_cross_modal_sweep.json') as f:
    d = json.load(f)
assert 'seed_results' in d
assert 'pass_result' in d
assert len(d['seed_results']) == 20
print(f'Verdict: {d[\"pass_result\"][\"verdict\"]}')
print(f'Seeds: {len(d[\"seed_results\"])}')
"
```

## Step 4: Run substrate tests

```bash
python -m pytest tests/substrate/test_p4_cross_modal_mechanism.py -x -q
```

Expected: 21 passed in ~1s.

## What this protocol covers

- Three-arm comparison: Arm A (mpnet+CLIP+hippo), Arm B (CLIP+CLIP+hippo), Arm C (CLIP+CLIP+cosine)
- 20 seeds with EC threshold jitter (±0.02)
- Pooled per-probe metrics (n=50, NOT mean-of-per-class)
- Two pass criteria: margin floor + paired bootstrap 95% CI
- CLIP-text EC threshold auto-calibration

## What this protocol does NOT cover

- Stage 1 mechanism tests (see `protocols/p4_stage1_reproduction.md`)
- Option 2 measurement (see `docs/experiments/p4_option2_measurement.md`)
- Persistence round-trip (P3.5 harness)

## Determinism guarantees

- Arm C (cosine baseline) is fully deterministic — no seed effect.
- Arms A and B are deterministic per seed via fixed EC threshold jitter from `numpy.random.default_rng(seed)`.
- CLIP-text EC threshold is auto-calibrated from the fixture class names — same fixture always produces the same threshold.
- Bootstrap CI uses fixed `rng_seed=42` with 10,000 resamples.

## Single-shot rerun rule

Per the plan, the 20-seed sweep runs ONCE per (fixture, encoder, mechanism) config. Re-running with the same config requires a written failure report at `docs/experiments/p4_cross_modal_failure_<date>.md` explaining why the rerun is necessary.
