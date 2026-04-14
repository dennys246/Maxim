# Deferred: LLM Mesh — Capability-Aware Routing

**Status:** Shell plan — not yet active
**Revive when:** the mesh has ≥2 GPU nodes with **different capabilities** (different models loaded, different VRAM, different speeds) and operators want the router to pick based on what each node can actually serve, not just static priority order.
**Estimated scope:** ~300 LOC new
**Depends on:**
- [llm_path_operator_visibility.md](../llm_path_operator_visibility.md) (Plan 4) — `mesh.yml` schema, `/v1/mesh/state` admin endpoint, request-trace ring buffer
- [llm_path_peer_failover.md](../llm_path_peer_failover.md) (Plan 3.6) — multi-leader provider list in router (the strict-priority precursor)
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

## What this plan does NOT add

- **Predictive routing** ("node A averages 150 tok/s, this prompt is 500 tokens, ETA 3.3s") — too noisy for v1, requires per-prompt latency modeling.
- **Heterogeneous quantization handling** ("the 7080 has Q4_K_M, the 3070 has Q5_0, prefer the higher quality if both can serve") — out of scope.
- **Cross-lane fallback** (request a `large` model, fall back to `medium` if no large nodes) — separate concern, would live in `lane_backends.py`.
- **Spot-bidding / preemption** — out of scope permanently.

## Revive trigger checklist

Before reviving, all must be true:

- [ ] [llm_path_peer_failover.md](../llm_path_peer_failover.md) (Plan 3.6) shipped — strict-priority failover works
- [ ] [llm_path_operator_visibility.md](../llm_path_operator_visibility.md) (Plan 4) shipped — `mesh.yml` + admin API exist
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

- [llm_path_refinement.md](../llm_path_refinement.md) — meta plan
- [llm_path_operator_visibility.md](../llm_path_operator_visibility.md) — Plan 4
- [llm_path_peer_failover.md](../llm_path_peer_failover.md) — Plan 3.6 (cheap precursor)
- [llm_path_multi_peer_dispatch.md](llm_path_multi_peer_dispatch.md) — load-distribution layer
- [llm_path_async_router.md](llm_path_async_router.md) — concurrency layer (orthogonal but related)
