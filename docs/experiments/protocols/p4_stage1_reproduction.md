# P4 Stage 1 reproduction runbook

**Plan:** [substrate_p4_cross_modal_binding.md](../../plans/substrate_p4_cross_modal_binding.md)
**Milestone summary:** [p4_stage1_mechanism.md](../p4_stage1_mechanism.md)

This runbook reproduces the P4 Stage 1 cross-modal binding mechanism tests from scratch. Stage 1 is **mechanism unit tests** — there is no sweep, no LLM, no real images, no statistical sampling. A faithful re-run of every test produces byte-identical assertion outcomes (all green) on any machine running Python 3.12.

## Prerequisites

- Clean Maxim checkout (main at or after the P4 Stage 1 PR merges, OR the `feat/substrate-p4` branch directly).
- Python 3.12 in a worktree / venv with `pip install -e .` run once.
- **No optional extras required.** Stage 1 uses pure-numpy synthetic embeddings and does NOT touch `sentence-transformers`, CLIP, `LinguisticEncoder`, or the `semantic` extra. (Stage 2 will need `semantic`; Stage 3 will need it for the OpenCLIP baseline.)
- No GPU required.
- No network access required.

## Step 1 — Run the Stage 1.5 vacuous-pass guard FIRST

The vacuous-pass guard validates the cluster-aware fixture's mathematical properties + asserts that real `EntorhinalCortex` clusters paired same-modality samples to one node id. If this file fails, every other Stage 1 test result is uninterpretable — fix the fixture before proceeding.

```bash
PYTHONPATH=src python -m pytest tests/substrate/test_p4_00_vacuous_pass_guard.py -v
```

Expected: 8 passed in ~0.2s.

```
TestFixtureGeometry::test_within_pair_text_text_above_ec_threshold PASSED
TestFixtureGeometry::test_within_pair_vision_vision_above_ec_threshold PASSED
TestFixtureGeometry::test_cross_pair_below_ec_threshold PASSED
TestFixtureGeometry::test_orthogonal_centroids_are_actually_orthogonal PASSED
TestECClusteringVacuousPassGuard::test_paired_text_samples_collapse_to_one_node PASSED
TestECClusteringVacuousPassGuard::test_paired_vision_samples_collapse_to_one_node PASSED
TestECClusteringVacuousPassGuard::test_cross_pair_samples_land_in_distinct_nodes PASSED
TestECClusteringVacuousPassGuard::test_text_and_vision_for_same_pair_get_distinct_nodes PASSED
```

If you instead see failures in `TestFixtureGeometry`, the fixture math has drifted (likely `noise_std` / `dim` / number of pairs combination produced cosine similarities outside the expected band). If you see failures in `TestECClusteringVacuousPassGuard`, EC's `pattern_complete_or_separate` is misbehaving with the synthetic embeddings — investigate `src/maxim/similarity/ec.py` directly, NOT the fixture.

## Step 2 — Run the mechanism tests

```bash
PYTHONPATH=src python -m pytest tests/substrate/test_p4_cross_modal_mechanism.py -v
```

Expected: 17 passed in ~0.2s.

The four test classes:

- `TestAutoTagAtEpisodeClose` (4 tests) — auto-tag at episode close, legacy events, mixed-modality episodes, drain-before-Hebbian-close consistency
- `TestRetrieveCrossModal` (6 tests) — forward / reverse retrieval, defensive same-modality cue ValueError, untagged cue, no-cross-modal-episodes, multi-pair routing isolation
- `TestSnapshotPatternFilter` (2 tests) — frozenset closure cell + lock-free closure under contention
- `TestPersistence` (5 tests) — dump/load round trip, unknown modality literal rejection, non-dict payload rejection, legacy snapshot, clear-then-load semantics, atomic-rollback regression guard

## Step 3 — Confirm zero regressions in the broader episode + persistence + hippocampus surface

```bash
PYTHONPATH=src python -m pytest \
  tests/substrate/test_p3a_episode_binding.py \
  tests/substrate/test_p3b_channel_integration.py \
  tests/substrate/test_p3a_fixture_validation.py \
  tests/substrate/test_persistence_harness.py \
  tests/substrate/test_snapshot_subprocess_round_trip.py \
  tests/substrate/test_p4_00_vacuous_pass_guard.py \
  tests/substrate/test_p4_cross_modal_mechanism.py \
  tests/unit/test_hippocampus.py \
  tests/unit/test_hippocampus_atomic_load.py \
  tests/unit/test_hippocampus_threading.py \
  -q
```

Expected: 176 passed in ~10s.

## Step 4 — Confirm the lint + format invariants are clean

```bash
ruff check src/maxim/memory/ tests/substrate/test_p4_*.py tests/substrate/p4_fixture_gen.py
ruff format --check src/maxim/memory/ tests/substrate/test_p4_*.py tests/substrate/p4_fixture_gen.py
```

Expected: `All checks passed!` and `X files already formatted`.

## What this runbook does NOT cover

- Real CLIP encoder runs (Stage 2)
- Oxford Flowers-102 fixture loading (Stage 2)
- 20-seed three-arm head-to-head sweep (Stage 3)
- VRAM audit (Stage 2)
- Subprocess mug test on real images (Stage 2)

These are reproducible after Stage 2 / Stage 3 land via separate runbooks — `protocols/p4_stage2_reproduction.md` and `protocols/p4_stage3_reproduction.md`.

## Determinism guarantees

- Synthetic fixture is seeded via `numpy.random.default_rng(seed)` at every call site; default seed `42`. Same seed → identical embeddings.
- All tests use deterministic node IDs (`text_mug`, `vision_mug`, etc.) — no UUIDs in the test surface.
- The atomic-rollback test uses pytest's `monkeypatch` fixture for adapter-level monkeypatching; no global state leakage between tests.
- Lock-inversion test uses `threading.Event` synchronization, NOT sleep — robust under heavy CI load.

A faithful re-run on any reasonable machine produces 25/25 green for the P4 Stage 1 surface and 176/176 green for the broader regression surface.
