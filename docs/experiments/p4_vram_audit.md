# Substrate P4 Stage 2 — VRAM audit

**Total wall clock:** 53.1s
**Backend:** cuda

## Summary

Measurements taken at each stage of bringing up the P4 Stage 2
encoder stack: baseline (whatever was already loaded) → add CLIP
→ add paraphrase-mpnet → run full mug test encoding → drop GPU
cache. On a machine with a running local LLM (RTX 5080 leader +
Qwen-14B-Q4_K_M), the baseline should show ~8-10 GB used by the
LLM; the deltas tell us how much additional VRAM the P4 Stage 2
encoders consume co-resident with inference.

## Measurement table

| step | wall | torch alloc MB | torch reserved MB | nvidia-smi used MB | nvidia-smi total MB |
|---|---|---|---|---|---|
| baseline | 7.3s | 0 | 0 | 11810 | 16303 |
| after_clip_load | 48.6s | 577 | 630 | 12699 | 16303 |
| after_mpnet_load | 51.0s | 996 | 1080 | 13149 | 16303 |
| after_mug_test_encode | 53.1s | 1005 | 1084 | 13217 | 16303 |
| after_cuda_empty_cache | 53.1s | 1005 | 1080 | 13213 | 16303 |

## Deltas (successive steps)

- **baseline → after_clip_load**: torch alloc +577 MB (nvidia-smi: +889 MB)
- **after_clip_load → after_mpnet_load**: torch alloc +418 MB (nvidia-smi: +450 MB)
- **after_mpnet_load → after_mug_test_encode**: torch alloc +9 MB (nvidia-smi: +68 MB)
- **after_mug_test_encode → after_cuda_empty_cache**: torch alloc +0 MB (nvidia-smi: +-4 MB)

## Headroom check

- **GPU:** 16303 MB total, 13213 MB used, 3090 MB free (3.02 GB)
- **Spillover-detection headroom rule** (Plan 3.6): recommended minimum headroom is `max(1.5 GB, 0.55 * weights_gb)`. For Qwen-14B Q4_K_M (~8 GB weights) that's `max(1.5, 4.4) = 4.4 GB`.
- **VERDICT: WARN** — 3.02 GB free is above the 1.5 GB minimum but below the 4.4 GB recommended for Qwen-14B. Consider smaller LLM (Qwen-7B or mistral-7b) OR lower n_ctx OR run Phase 2D/Stage 3 sweeps on a dedicated worktree with MAXIM_LLM_ENABLED=0.

## How to reproduce

```bash
# Run ON the RTX 5080 leader (or any machine where you want
# to measure co-residency). Requires the 'semantic' extra and
# the Flowers102 torchvision cache populated by Phase 2B.
PYTHONPATH=src python scripts/p4_vram_audit.py
```

The script is idempotent — re-running gives the same delta
values as long as the baseline GPU state is unchanged (i.e.,
the LLM is still loaded at the same ctx length). The absolute
baseline MB number will vary depending on what else is loaded.
