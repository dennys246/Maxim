# Deferred: LLM Path — Multi-peer Reactive Dispatch

**Status:** Deferred shell plan — partially triggered as of 2026-04-13 (user has RTX 3070 as a second GPU node)
**Revive when:** ANY of the following:
1. **Phase D stress test shows leader saturation** that `llama.cpp --parallel` batching does not resolve, OR
2. **User has ≥2 GPU nodes that need to be reachable as a single cluster** (RTX 5080 + RTX 3070 setup is the current concrete trigger), OR
3. **Plan 3.6 (multi-URL `peer.yml` failover) ships and the operator wants load distribution beyond strict-priority failover.**
**Estimated scope:** ~250 LOC new
**Depends on:**
- Plan 3 (`llm_path_fast_failover.md`) shipped — `_MaximPeerBackend` + typed exceptions + router fallback loop
- Plan 3.6 (`llm_path_peer_failover.md`) shipped — provides the multi-provider router config that this plan extends with rendezvous-hash distribution
- Plan 4 (`llm_path_operator_visibility.md`) shipped — `mesh.yml` schema + drain state + admin API
**Related deferred:**
- [llm_path_async_router.md](llm_path_async_router.md) — if lane serialization is the bottleneck instead of leader compute
- [llm_mesh_capability_aware.md](llm_mesh_capability_aware.md) — the next layer up (capability-aware ranking instead of homogeneous distribution)

## Relationship to Plan 3.6 (peer failover)

Plan 3.6 ships the **strict-priority multi-leader** primitive: `peer.yml` lists N leaders, the router tries them in order, falls through on `BackendDown`. That's enough for hot-standby setups (RTX 5080 primary + RTX 3070 fallback).

This plan (multi-peer dispatch) is the **load-distribution layer on top**. Instead of strict priority, it uses rendezvous hashing keyed on `agent_id` or `session_id` so concurrent agents naturally spread across all healthy peers. Hot-standby becomes hot-active; both nodes serve real load.

**Migration path:** Plan 3.6's `peer.yml::leaders:` and Plan 4's `mesh.yml::nodes:` both feed the same provider-list shape that this plan's rendezvous hash consumes. When this plan ships, the change is in the SELECTION strategy (`_rank_providers_for_request`), not in the config schema.

## Why this was deferred

The architecture audit ([llm_path_refinement.md](../llm_path_refinement.md)) revealed that `LLMRouter._complete_text_locked` already implements reactive provider fallback natively. Adding multi-peer dispatch is a ~100 LOC change to `_build_remote_backend` that feeds the router a multi-peer provider list. **But:**

1. **Stress test may show it isn't needed.** Plan 2's Phase C batching PoC measures whether `llama.cpp --parallel` at the leader doubles/triples throughput. If yes, we don't need distribution.
2. **Multi-peer adds failure modes that don't exist today.** Self-dispatch loops, uneven distribution, peer-to-peer trust. Better to ship without it and add only if the data demands.
3. **Plan 3's operator visibility layer is useful regardless.** `mesh.yml` + admin API + per-agent stats have value even when routing is single-peer. This plan builds on top of that foundation.

## What this plan would add

### R3.3-lite — Multi-peer provider registration (~100 LOC)

`lane_backends.py::_build_remote_backend` iterates `mesh.yml::nodes` and registers one provider per peer.

```python
def _build_remote_backend(self, cfg: LaneConfig) -> Any | None:
    mesh = load_mesh_config()
    if not mesh or len(mesh.nodes) <= 1:
        return self._build_single_remote_backend(cfg)
    
    # Self-dispatch protection (Plan 3 R3.0 sets mesh.self)
    routable = [n for n in mesh.nodes if n.name != mesh.self]
    # Drain state (Plan 3 R3.0)
    active = [n for n in routable if n.name not in mesh.drain]
    
    if not active:
        logger.warning({"event": "mesh_no_routable_nodes", "self": mesh.self, "drained": len(mesh.drain)})
        return self._build_local_backend(cfg)
    
    # Leader first, peers in rendezvous-hash order
    # Agent-scoped seed so concurrent agents distribute across peers
    seed = os.environ.get("MAXIM_SESSION_ID", str(time.monotonic()))
    sorted_nodes = rendezvous_hash(active, seed, leader_first=True)
    
    providers = dict(base.providers or {})
    priority = []
    for node in sorted_nodes:
        provider_key = f"lane-{cfg.name}-{node.name}"
        providers[provider_key] = {
            "base_url": node.url,
            "api_key_env": "MAXIM_CLUSTER_KEY",
            "model": cfg.remote_model or cfg.model_profile,
            "backend_class": "maxim_peer",
            "pricing_required": False,
        }
        priority.append(provider_key)
    routing["provider_priority"] = priority
    return LLMRouter(remote_cfg)
```

**Drain state invalidates router** (from Plan 3 R3.6-lite): admin `drain` endpoint signals `LaneBackendManager` to rebuild affected routers on next access.

**New utility:** `maxim/utils/hashing.py::rendezvous_hash(items, seed, leader_first=False)` — ~20 LOC pure function.

### R3.4 — 429 response headers in `leader_proxy` (~100 LOC)

Extend `runtime/leader_proxy.py`'s 429 path to set headers that `_MaximPeerBackend` (Plan 2) already parses into `BackendOverloaded`:

- `Retry-After: <seconds>` — calculated from `avg_wait + slot_position × avg_latency`
- `X-Maxim-Queue-Depth: <int>` — current in-flight count
- `X-Maxim-Node-Id: <name>` — from `mesh.yml::self`
- `X-Maxim-Suggested-Peer: <name>` — empty initially (populate in a future iteration if needed)

**Successful responses get `X-Maxim-Backpressure: 1`** when node is above 80% of `max_concurrent`. Observation signal only; `_MaximPeerBackend` logs it but doesn't yet drive routing. Future enhancement.

### R3.4a — Multi-peer integration tests (~50 LOC)

- 3-node mock fixture (1 leader + 2 peers)
- Chaos scenarios: kill peer mid-request, partition simulation, lying peer (advertises model it doesn't have), slow peer EMA decay
- SLO checks: saturated-leader failover latency, dispatch overhead, rendezvous hash distribution

## What this plan does NOT add

- Proactive capability-aware routing (heartbeats, ranking formulas) — separate deferred plan
- Async router (concurrent per-lane dispatching) — separate deferred plan
- Fair-share scheduling — separate deferred plan
- Peer-to-peer GGUF streaming — out of scope permanently

## Revive trigger checklist (updated 2026-04-13)

Before reviving, ALL of the following must be true:

- [x] Plan 3 (`llm_path_fast_failover.md`) shipped with typed-exception router loop ✅ (PR #94)
- [x] Plan 3.5 (`llm_path_cancellation_hygiene.md`) shipped with the "HTTP fires first" contract ✅ (PR #96)
- [ ] Plan 3.6 (`llm_path_peer_failover.md`) shipped — provides the multi-provider config primitive this plan extends with rendezvous hashing
- [ ] Plan 4 (`llm_path_operator_visibility.md`) shipped — provides `mesh.yml` + drain state + admin API
- [ ] User has ≥2 GPU nodes in the cluster (the RTX 5080 + RTX 3070 setup is **partially met** as of 2026-04-13 — the 3070 box exists but isn't yet wired into a `peer.yml` or `mesh.yml`)
- [ ] At least one of:
  - Phase D stress test shows leader saturation that `llama.cpp --parallel` does NOT solve
  - Operator has a measurable workload that needs both nodes serving concurrent traffic
  - Plan 3.6 strict-priority failover is shipping but operator wants load distribution beyond hot-standby

**Current status (2026-04-13):** the hardware trigger is partially met but the dependency chain is not complete. The natural progression is Plan 3.6 → Plan 4 → this plan. Each unlocks the next.

## Related docs

- **Current plan:** [../llm_path_operator_visibility.md](../llm_path_operator_visibility.md)
- **Previous plan:** [../llm_path_fast_failover.md](../archive/llm_path_fast_failover.md)
- **Meta plan:** [../llm_path_refinement.md](../llm_path_refinement.md)
