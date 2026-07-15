# Deferred: LLM Mesh — Capability-Aware Routing

**Status:** Shell plan — not yet active (expanded 2026-04-13 with Stage 5 spillover detection after the 125s leader latency was root-caused to VRAM spillover into shared memory)
**Revive when:** the mesh has ≥2 GPU nodes with **different capabilities** (different models loaded, different VRAM, different speeds) and operators want the router to pick based on what each node can actually serve, not just static priority order. Stage 5 (runtime tok/s baseline) revives independently when the static VRAM ratio check from Plan 3.6 R5 proves insufficient — i.e., when an operator hits a slowdown that the static check missed.
**Estimated scope:** ~450 LOC new (300 capability ranking + 150 spillover detection)
**Depends on:**
- [llm_path_operator_visibility.md](../archive/llm_path_operator_visibility.md) (Plan 4) — `mesh.yml` schema, `/v1/mesh/state` admin endpoint, request-trace ring buffer
- [llm_path_peer_failover.md](../archive/llm_path_peer_failover.md) (Plan 3.6) — multi-leader provider list in router (the strict-priority precursor)
- [llm_path_multi_peer_dispatch.md](llm_path_multi_peer_dispatch.md) — rendezvous hash + drain state (the load-distribution layer)
**Related deferred:** [llm_path_async_router.md](llm_path_async_router.md) — if lane serialization is the bottleneck instead of capability matching

## Why this is deferred

Plan 3.6 (Peer Failover) and the multi-peer dispatch deferred plan together cover the case where **every node serves the same model**. They route by priority (3.6) or by load-distribution hash (multi-peer dispatch). Neither handles the case where **node A has Qwen-14B loaded and node B has Mistral-7B loaded** — a request for Qwen-14B should NEVER go to node B, even if node B is healthy and underloaded.

This is a real concern once the mesh is heterogeneous. But:

1. **The current mesh is homogeneous.** RTX 5080 + RTX 3070 will likely both serve Qwen-14B (or both serve a 7B model the 3070 can handle). Capability matching only matters when nodes diverge.
2. **Plan 3.6 + multi-peer dispatch get the user 90% of the value** for a fraction of the LOC. Capability matching is the last 10% — important when needed, but not blocking the immediate roadmap.
3. **The data needed for capability-aware routing isn't expensive to collect** — Plan 4 R3.6-lite's `/v1/mesh/state` already exposes `vram_total_gb` and the loaded model. The expensive part is the routing decision logic + the ranking formula.

## What this plan would add

### Stage 1 — Capability advertisement (~80 LOC)

Each node's `/v1/mesh/state` endpoint (introduced in Plan 4) reports:

```json
{
  "node": "rtx-5080",
  "role": "leader",
  "capabilities": {
    "models": ["qwen2.5-14b-instruct", "mistral-7b-instruct-v0.2"],
    "loaded_model": "qwen2.5-14b-instruct",
    "vram_total_gb": 16,
    "vram_free_gb": 0.7,
    "tier": "large",
    "max_context_tokens": 8192,
    "approx_tokens_per_sec": 32.5
  },
  "queue": {
    "in_flight": 1,
    "max_concurrent": 4,
    "queue_depth": 0
  },
  "uptime_s": 84620,
  "last_request_ts": 1776125555.0
}
```

The peer's `LLMRouter` polls this endpoint at startup and every N seconds (config: `MAXIM_MESH_REFRESH_S=30`) and caches the result in `_provider_capabilities[provider_key]`.

### Stage 2 — Capability-aware provider selection (~120 LOC)

Replace the static priority-order loop in `_try_provider` with a ranking function that walks the `_provider_capabilities` cache:

```python
def _rank_providers_for_request(self, lane: str, model_hint: str | None) -> list[str]:
    """Return providers in preferred order for this lane + model.
    
    Ranking factors:
    1. Required model loaded (HARD filter — drop providers that can't serve)
    2. VRAM headroom (soft factor — prefer nodes with more free VRAM)
    3. Queue depth (soft factor — prefer less-loaded nodes)
    4. Health score (soft factor — penalize nodes with recent failures)
    5. Static priority (tiebreaker — operator's mesh.yml ordering)
    """
    candidates = [pk for pk, cap in self._provider_capabilities.items() 
                  if self._can_serve(cap, lane, model_hint)]
    return sorted(candidates, key=lambda pk: self._provider_score(pk, lane, model_hint))
```

The `_can_serve` filter is the hard gate — model must be loaded, VRAM must be sufficient. Nodes that fail the filter are NEVER tried, regardless of priority.

The `_provider_score` soft ranking combines queue depth + VRAM + health + priority. Tunable weights via `mesh.yml::routing_weights:`.

### Stage 3 — Stale capability handling (~50 LOC)

Capabilities can change underfoot — operator might `install qwen-72b` on a node, blowing past its VRAM budget. The cache must invalidate quickly enough that routing decisions reflect reality.

- TTL on the cache: re-poll every `MAXIM_MESH_REFRESH_S` (default 30s).
- Invalidate on `BackendError`: a `BackendInferenceBroken` from a node forces a re-poll within 5s.
- `mesh.yml::self`-aware: don't poll yourself (loopback would race with the local state).

### Stage 4 — Doctor + admin API integration (~50 LOC)

- `maxim doctor --capabilities` lists each node's advertised capabilities + flags mismatches.
- `GET /v1/mesh/capabilities` returns the local node's capability snapshot.
- `POST /v1/mesh/refresh-capabilities/<peer>` forces a fresh poll.

### Stage 5 — Runtime spillover detection via tok/s baseline (~150 LOC)

**Motivating evidence:** the 2026-04-13 125s leader latency was VRAM spillover into shared memory. Static `vram_used / vram_total > 0.93` (added in Plan 3.6 R5) catches the case at startup or via `maxim doctor`, but doesn't catch:
- Models that load fine but spill once KV cache grows during a long prompt
- Multi-process VRAM contention (another process eats VRAM after Maxim starts)
- Driver-level overcommit on cards that allow it transparently

The robust signal is **measured tok/s vs expected tok/s for this (model, GPU class) pair**. A node generating at 5 tok/s when the baseline says 32 tok/s is degraded — the cause is usually spillover, but it could also be thermal throttling, another GPU consumer, or a model-specific perf bug. The router doesn't need to know WHY; it just needs to know "this node is currently 5x slower than its baseline" and route accordingly.

**Implementation:**

1. **Baseline lookup table (`runtime/lane_models.py::_TOKENS_PER_SEC_BASELINE`).** Static table of measured tok/s per (model_profile, gpu_class) pair. Populated from real-world measurements on the user's hardware tier — extends the existing `_INFER_VRAM_TIERS` table that already maps GPU classes to VRAM-appropriate models. Format: `{("qwen2.5-14b-instruct", "rtx-50-class"): 32.5, ("qwen2.5-7b-instruct", "rtx-30-class"): 28.0, ...}`. Defaults conservative; specific entries override.

2. **Per-call tok/s measurement.** `_MaximPeerBackend.complete_with_usage` already records `output_tokens` and `elapsed_ms` in the `peer_backend_call` JSONL event. Compute `tokens_per_sec = output_tokens / (elapsed_ms / 1000)` and compare against the baseline. Note: this is GENERATION speed, excluding prefill — prefill latency is mostly prompt-size-dependent and orthogonal.

3. **Sliding-window degradation detector.** The router maintains a per-provider EMA of recent tok/s (window: last N=10 calls). When EMA drops below `0.5 * baseline`, mark the provider as `degraded` and emit a structured `provider_degraded` WARN with `provider`, `observed_tps`, `baseline_tps`, `degradation_ratio`. The capability ranking formula (Stage 2's `_provider_score`) penalizes degraded providers heavily — they fall to the bottom of the priority list but are NOT removed entirely (graceful degradation, not hard removal).

4. **Recovery detection.** When EMA recovers above `0.8 * baseline` for N=5 consecutive calls, clear the degraded flag and emit `provider_recovered`. Same EMA window, hysteresis prevents flapping.

5. **Per-call exposure.** Each `peer_backend_call` event gains a `degradation_ratio` field (0.0-1.0) so operators can grep the JSONL for slow nodes without waiting for the WARN. The aggregated `dispatch_exhausted` event (already present from Plan 3) gains a `degraded_providers` field listing any providers that were skipped due to degradation.

**Why this lives in Stage 5 of the deferred plan, not Plan 3.6:**

- The baseline table needs real measurements across the user's actual GPU classes. RTX 5080, RTX 3070, and Apple Silicon all have different baselines. Plan 3.6 only has the RTX 5080 + RTX 3070 setup; expanding the table needs more hardware coverage.
- The router needs to be capability-aware (Stages 1-2 of this plan) before degradation-aware ranking is meaningful. Without capability data, a slow node is just "a node" — the router has nowhere to fall back to.
- The 5x-slowdown threshold is heuristic. Tuning needs evidence from multiple workloads (sim, real chat, agent loop) before it's stable.

**Cross-platform notes:**

- **NVIDIA Linux:** the simplest signal is tok/s. `nvidia-smi --query-gpu=memory.used` does NOT include shared memory on Linux as of driver 535. Don't rely on it.
- **NVIDIA Windows 11+:** `nvidia-smi --query-gpu=memory.used,memory.shared` reports shared memory directly on driver 545+. If available, use as a confirmation signal alongside tok/s.
- **Apple Silicon (UMA):** there is no spillover by definition — system RAM IS GPU RAM. Tok/s baseline still useful for thermal throttling detection.
- **AMD ROCm:** untested. The tok/s baseline approach works regardless of vendor; only the optional vendor-specific telemetry differs.

## What this plan does NOT add

- **Predictive routing** ("node A averages 150 tok/s, this prompt is 500 tokens, ETA 3.3s") — too noisy for v1, requires per-prompt latency modeling.
- **Heterogeneous quantization handling** ("the 7080 has Q4_K_M, the 3070 has Q5_0, prefer the higher quality if both can serve") — out of scope.
- **Cross-lane fallback** (request a `large` model, fall back to `medium` if no large nodes) — separate concern, would live in `lane_backends.py`.
- **Spot-bidding / preemption** — out of scope permanently.

## Revive trigger checklist

Before reviving, all must be true:

- [ ] [llm_path_peer_failover.md](../archive/llm_path_peer_failover.md) (Plan 3.6) shipped — strict-priority failover works
- [ ] [llm_path_operator_visibility.md](../archive/llm_path_operator_visibility.md) (Plan 4) shipped — `mesh.yml` + admin API exist
- [ ] User has ≥2 GPU nodes with **different** loaded models (e.g., RTX 5080 with Qwen-14B + RTX 3070 with Mistral-7B)
- [ ] There's a measurable operator pain point: requests are being routed to nodes that can't serve them, OR operator is manually editing `peer.yml` priority on every model change
- [ ] [deferred/llm_path_multi_peer_dispatch.md](llm_path_multi_peer_dispatch.md) is shipped or shipping (rendezvous-hash distribution provides the routing primitive this plan extends)

## Architectural impact (when revived)

This is **the layer that turns the system into a true reactive mesh.** Until this ships:
- Plan 3.6 = strict-priority failover (cluster has redundancy but not load awareness)
- Multi-peer dispatch = load-distribution hash (cluster spreads load but assumes homogeneity)
- This plan = capability-aware ranking (cluster is genuinely smart about where to send each request)

After this ships, the term "reactive mesh" is accurate, not aspirational.

## Related

- [llm_path_refinement.md](../archive/llm_path_refinement.md) — meta plan
- [llm_path_operator_visibility.md](../archive/llm_path_operator_visibility.md) — Plan 4
- [llm_path_peer_failover.md](../archive/llm_path_peer_failover.md) — Plan 3.6 (cheap precursor)
- [llm_path_multi_peer_dispatch.md](llm_path_multi_peer_dispatch.md) — load-distribution layer
- [llm_path_async_router.md](llm_path_async_router.md) — concurrency layer (orthogonal but related)
