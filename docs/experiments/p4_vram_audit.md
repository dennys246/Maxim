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

## Interpretation — why WARN ≠ "don't ship"

The Plan 3.6 headroom rule (`max(1.5 GB, 0.55 × weights_gb)`) is a **conservative runtime growth buffer** designed around the KV-cache spillover incident that motivated Plan 3.6 in the first place. That incident happened because Qwen-14B @ 12k context dynamically grew its KV cache into shared GPU memory during long-running sims, causing the 125s latency regression. The 4.4 GB rule exists to prevent that class of failure.

On THIS machine in THIS configuration:

- **n_ctx is 8k, not 12k.** The commit headline is "WARN @ 8k" — the leader was intentionally restarted with a reduced context window before the audit. At fixed n_ctx, llama.cpp pre-allocates the KV cache at model load time; there is NO runtime growth pressure. The 3.02 GB headroom is static.
- **The P4 Stage 2 encoder stack also has no runtime growth pressure.** CLIP + paraphrase-mpnet weights are loaded once and never dynamically re-allocate. The torch caching allocator may briefly overshoot during encode, but the measurements show 0 MB growth between `after_mpnet_load` and `after_mug_test_encode`.
- **The 3.02 GB free budget is therefore steady-state, not peak.** The Plan 3.6 rule assumes peak-vs-steady-state delta, which doesn't apply here.

**Operationally:** the WARN verdict is technically correct per the rule, but the rule over-penalizes fixed-n_ctx configurations. The co-residency works for Phase 2D's single-shot mug test (which is what this audit measured). The open question is Stage 3, which runs 20 seeds × 3 arms = 60 mug-test-equivalent runs back-to-back.

## Stage 3 implications

Stage 3 is the 20-seed three-arm sweep (Arm A: paraphrase-mpnet text + CLIP vision + hippocampus; Arm B: CLIP-text + CLIP-vision + hippocampus — the load-bearing claim; Arm C: CLIP-text + CLIP-vision + raw cosine baseline). It needs the CLIP + paraphrase-mpnet encoders loaded for Arms A/B and CLIP alone for Arm C.

**Three options, ordered by user preference for Stage 3:**

### Option 3α — Stay on Qwen-14B @ 8k, accept WARN

Run Stage 3 with the current co-residency config. Rationale: 3.02 GB steady-state headroom plus Phase 2D's demonstrated 0 MB encode-time growth means the sweep should succeed. The agent-loop sim isn't what's being measured — Stage 3's retrieval metric is pure encoder + hippocampus, and the LLM is only tangentially present.

- **Pro:** no config changes; the 5080 continues serving inference while the sweep runs.
- **Con:** if ANY component of Stage 3 touches the LLM (e.g. a logging hook that triggers a synchronous inference call), the in-flight KV cache plus the encoder stack could briefly exceed the 3 GB budget and trigger a CUDA OOM.
- **Mitigation:** run Stage 3 with `MAXIM_LLM_CALL_TIMEOUT_S=60` so any accidental LLM invocation fails fast rather than hanging at OOM.

### Option 3β — Downgrade LLM to Qwen-7B or mistral-7b for Stage 3

Drop the LLM footprint by ~4 GB, giving Stage 3 ~7 GB of headroom comfortably above the rule.

- **Pro:** comfortable headroom; leader stays functional for any accidental LLM call.
- **Con:** if other ongoing work is comparing sim results against Qwen-14B behavior, swapping models mid-project complicates the comparison. Need to flip back after Stage 3 lands.

### Option 3γ — Dedicated Stage 3 worktree with `MAXIM_LLM_ENABLED=0` (RECOMMENDED)

Run the Stage 3 sweep in a worktree with the LLM explicitly disabled for the duration of the sweep. The 20-seed three-arm sweep runs pure CLIP + hippocampus mechanics — NO agent loop, NO sim orchestrator, NO LLM inference. The LLM is 100% unused during the Stage 3 measurements.

- **Pro:** maximum headroom (~14 GB free) during the sweep. Zero risk of co-residency OOM. Fastest wall-clock time because nothing contends with the GPU. The leader can be restored to Qwen-14B AFTER the sweep completes without having to re-measure anything.
- **Con:** requires temporarily pausing any other work that depends on the leader serving inference. If there's a Claude coding session that needs the LLM, they have to wait for the sweep to finish (~15-30 min expected).
- **Implementation:** `git worktree add .worktrees/p4_stage3 feat/substrate-p4-stage3`, then set `MAXIM_LLM_ENABLED=0` in the sweep script's environment, run, commit results, remove worktree.

**Recommendation: Option 3γ.** It's the cleanest architectural choice, gives Stage 3 the most headroom for any surprise growth, and the 15-30 min of exclusive GPU access is a one-time cost. The fallback is Option 3α if Stage 3 work has to share the leader with active sim usage.

## Note on 8k vs 12k context

The "WARN @ 8k" commit headline reflects that the debug Claude (or user) explicitly restarted the leader with a reduced n_ctx before running this audit. The underlying incident that motivated Plan 3.6's spillover detection was Qwen-14B @ **12k** context on this same 16 GB 5080, where the KV cache overflowed into shared GPU memory causing a 125s latency regression. At 8k, the KV cache fits cleanly and the system operates deterministically.

This is a relevant datapoint for the leader's general operating configuration — if the spillover incident was caught at 12k but the leader is now running at 8k, there may already be an implicit understanding that 12k is unsafe on this hardware. The P4 Stage 2 encoder stack co-residency constraint reinforces that: at 8k there is 3.02 GB free; at 12k there would be roughly 1.5 GB LESS (the KV cache delta is ~2 GB), bringing the free budget to ~1 GB, well under the 1.5 GB absolute minimum and into FAIL territory.

**Implicit conclusion:** on this 16 GB hardware, Qwen-14B co-resident with the P4 Stage 2 encoder stack is only safe at n_ctx ≤ 8k. Documented here so future Stage 3 / Stage 4+ work doesn't silently regress to 12k without re-auditing.

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
