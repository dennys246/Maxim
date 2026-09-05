# Substrate P4 Stage 2 — VRAM audit (Mac peer reference, NOT authoritative)

> **This report was produced on a Mac peer (MPS backend) as a smoke test
> of the audit script.** The authoritative RTX 5080 + Qwen-14B
> co-residency measurement lives in `p4_vram_audit.md` (run on the
> leader). These numbers are useful only as order-of-magnitude reference
> for the CLIP + paraphrase-mpnet standalone footprint — they do NOT
> answer the headroom-check question.

**Total wall clock:** 16.3s
**Backend:** mps

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
| baseline | 0.9s | 0 | 0 | n/a | n/a |
| after_clip_load | 10.6s | 577 | 577 | n/a | n/a |
| after_mpnet_load | 14.7s | 995 | 995 | n/a | n/a |
| after_mug_test_encode | 16.3s | 995 | 995 | n/a | n/a |

## Deltas (successive steps)

- **baseline → after_clip_load**: torch alloc +577 MB
- **after_clip_load → after_mpnet_load**: torch alloc +418 MB
- **after_mpnet_load → after_mug_test_encode**: torch alloc +0 MB

## Headroom check

- No CUDA backend detected — headroom verdict not applicable. This run provides reproducibility data only.

## How to reproduce

```bash
# Run ON the RTX 5080 leader (or any machine where you want
# to measure co-residency). Requires the 'semantic' extra and
# the Flowers102 torchvision cache populated by Phase 2B.
PYTHONPATH=src python scripts/p4_vram_audit.py  # D27: add --write-experiment-results to update the committed record
```

The script is idempotent — re-running gives the same delta
values as long as the baseline GPU state is unchanged (i.e.,
the LLM is still loaded at the same ctx length). The absolute
baseline MB number will vary depending on what else is loaded.
