# LLM Path Refinement — Plan 3.6: Peer Failover (Multi-URL `peer.yml`)

**Status:** Draft v1 — 2026-04-13
**Scope:** ~150 LOC new
**Target version:** 0.4 (post-Plan-3.5)
**Part of:** [llm_path_refinement.md](llm_path_refinement.md)
**Depends on:** Plan 3 R2.5 (`_MaximPeerBackend` + router typed exceptions) — already shipped
**Enables:** Multi-GPU mesh testing without waiting for full Plan 4 mesh infrastructure
**Sister plan:** [llm_path_operator_visibility.md](llm_path_operator_visibility.md) (Plan 4) — Plan 4's `mesh.yml` supersedes the multi-URL `peer.yml` shape introduced here

## Goal

**Allow `peer.yml` to list multiple leader URLs in priority order**, so the peer can fall through to a hot standby leader when the primary returns persistent `BackendDown`. This is the smallest concrete step toward true multi-GPU mesh that doesn't require Plan 4's full operator-visibility infrastructure to ship first.

The user has two GPU-capable nodes (RTX 5080 + RTX 3070) on the roadmap. This plan unblocks testing the second node as a fallback target without waiting for Plan 4's `mesh.yml` + admin API + per-agent rate limiting work.

**Three concrete outcomes:**

1. **`peer.yml` accepts a list under `leaders:`** in addition to the legacy single `url:` field. Backwards compatible — existing single-URL configs work unchanged.
2. **`PeerConfig` exposes an ordered list of leader URLs.** `apply_peer_config_to_env` registers ALL leaders as separate providers in the router's provider config, with priority order matching the `peer.yml` ordering.
3. **Router's existing typed-exception fallback handles failover.** No new dispatch code — Plan 3's `_try_provider` loop already iterates providers in priority order. We just feed it more than one provider for the `large` lane.

## Non-goals

- **Not capability advertisement.** Both leaders are assumed to serve the same model. Nodes don't tell the peer "I have Qwen-14B loaded" — operator declares it in `peer.yml`. (Capability advertisement is [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md).)
- **Not discovery.** No mDNS, no gossip. Static config in `peer.yml`. (Discovery is a separate future shell plan.)
- **Not load balancing.** Strict priority order. Leader 1 is tried first; only on `BackendDown` does the router try Leader 2. Round-robin and capacity-aware routing are [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md).
- **Not fan-out concurrency.** A single agent request still goes to ONE leader at a time. Fan-out would require async router work.
- **Not hot reload.** `peer.yml` changes still require a peer restart.
- **Not the full `mesh.yml` schema from Plan 4.** This plan is the cheap pre-cursor that uses `peer.yml`. Plan 4 introduces `mesh.yml` as the more general successor.

## The `peer.yml` shape

**Current (single-leader, unchanged):**
```yaml
url: https://maxim.dennyschaedig.com/v1
api_key: <cluster-key>
model: qwen2.5-14b-instruct
is_cloud: false
```

**New (multi-leader, additive):**
```yaml
leaders:
  - url: https://maxim.dennyschaedig.com/v1     # primary — RTX 5080
    name: rtx-5080
    model: qwen2.5-14b-instruct
  - url: https://maxim2.dennyschaedig.com/v1    # standby — RTX 3070
    name: rtx-3070
    model: qwen2.5-7b-instruct                  # smaller model, faster on 3070
api_key: <cluster-key>                           # shared key, same for all leaders
is_cloud: false
```

**Backwards compatibility rules:**
- If `leaders:` is present, it wins. The legacy `url:` and `model:` top-level fields are ignored (with a one-line WARN at startup if both are set).
- If `leaders:` is absent, fall back to the legacy single-`url:` shape unchanged.
- Each leader entry MUST have `url:`. `name:` and `model:` are optional (defaults: name = `leader-N`, model = config-level `model:` or unset).
- Single shared `api_key:` for all leaders. Per-leader keys are out of scope (cluster-wide trust model).
- The order of `leaders:` is the priority order. First entry is primary, subsequent entries are fallbacks.

## Router integration (the small change)

`peer/config.py::apply_peer_config_to_env` currently registers exactly one `leader` endpoint via `register_leader_endpoint`. Post-plan, it iterates the list and registers `leader-rtx-5080`, `leader-rtx-3070`, etc.

`runtime/lane_backends.py::_build_remote_backend` currently constructs a router with one `large` provider. Post-plan, it constructs the router with one provider per leader in priority order:

```python
# Pseudocode for _build_remote_backend post-plan
def _build_remote_backend(self, cfg: LaneConfig) -> Any | None:
    peer_cfg = read_peer_config()
    leaders = peer_cfg.leaders or [Leader(url=peer_cfg.url, name="leader", model=peer_cfg.model)]
    
    providers: dict[str, dict[str, Any]] = {}
    priority: list[str] = []
    for leader in leaders:
        provider_key = f"lane-{cfg.name}-{leader.name}"
        providers[provider_key] = {
            "type": "maxim_peer",
            "base_url": leader.url,
            "api_key_env": "MAXIM_LANE_LARGE_REMOTE_API_KEY",  # shared
            "model": leader.model or cfg.remote_model,
            "pricing_required": False,
        }
        priority.append(provider_key)
    
    return LLMRouter(replace(self.cfg, providers=providers, routing={"provider_priority": priority}))
```

**That's the entire dispatch change.** The router's existing `_try_provider` loop already walks the priority list, catches `BackendDown`/`BackendOverloaded`/`BackendTimeout` and moves to the next provider. Plan 3 R2.5 made this work for typed exceptions. Plan 3.5 R4 made cancellation hygiene work across the loop. There is no new fallback code to write — we just feed the existing loop a longer list.

## Phases

### R1 — Schema + parsing (~50 LOC)

- Add `Leader` dataclass to `peer/config.py` with `url`, `name`, `model` fields (frozen).
- Extend `PeerConfig` with `leaders: list[Leader] | None` field. Default `None` for backwards compat.
- Update `read_peer_config` parser to recognize the `leaders:` key. Use the existing minimal-YAML parser (no PyYAML dep).
- Update `to_yaml` writer to emit `leaders:` when populated, otherwise fall back to single `url:`.
- Add validation: every leader entry MUST have `url:`; reject duplicates; reject cloud URLs in self-hosted-only mode.
- Add startup WARN if both `url:` and `leaders:` are present.

### R2 — Router integration (~50 LOC)

- Update `apply_peer_config_to_env` to set `MAXIM_LANE_LARGE_REMOTE_URL` to the FIRST leader's URL (for backwards compat with code that reads the env var) AND register all leaders as separate HTTP endpoints in `utils/http.py` with names `leader-<leader-name>`.
- Update `runtime/lane_backends.py::_build_remote_backend` to feed the multi-leader provider list to `LLMRouter` per the pseudocode above.
- Verify provider_priority is honored — the `_try_provider` loop in `router.py` already iterates in this order; just confirm with a test.

### R3 — Tests (~30 LOC + fixtures)

- Unit test: `peer.yml` with two leaders parses to a `PeerConfig` with `leaders=[...]`.
- Unit test: backwards compat — single-`url:` `peer.yml` still works.
- Unit test: provider priority order matches `peer.yml` order.
- Integration test: stub backends where leader-1 raises `BackendDown` and leader-2 succeeds; assert the router picks leader-2 on the first call's retry.
- Integration test: leader-1 healthy → all calls go to leader-1, leader-2 untouched.
- Integration test: leader-1 dies mid-stream → next call falls through to leader-2 within a few hundred ms.

### R4 — `maxim doctor` integration (~10 LOC)

- Doctor's existing peer-mode probe checks `MAXIM_LANE_LARGE_REMOTE_URL` (one URL). Extend to check ALL configured leaders, report per-leader status with the typed exception class.

### R5 — Pre-merge review round (per the Plan 3 ship pattern)

Two parallel review Claudes (Executor + Architecture lens). Fold findings into the same branch before opening the PR.

## Success criteria

**Must-have gates:**

1. Single-URL `peer.yml` (existing user configs) work unchanged. Zero regression.
2. Multi-leader `peer.yml` parses correctly + registers N HTTP endpoints + constructs router with N providers.
3. Failover test: with leader-1 stubbed to raise `BackendDown`, a single LLM call succeeds via leader-2 within < 2s.
4. `maxim doctor` reports per-leader status when multi-leader config is detected.
5. `maxim peer version` works against the FIRST leader by default; new `--leader <name>` flag selects a specific one (out of scope or stretch goal).

**Nice-to-have:**

1. Real-world test: user runs sim against `peer.yml` with the RTX 5080 as primary + RTX 3070 as standby. Kills the 5080 mid-sim. Sim continues on the 3070 within a few seconds.

## Why this is "Plan 3.6" not "Plan 4"

Plan 4 (operator visibility) introduces `mesh.yml` as the more general successor to `peer.yml`. `mesh.yml` has `nodes:` with `role:`, `drain:`, `agent_rate_limits:`, and a full admin API for cluster operations. This plan (Plan 3.6) is **the cheap pre-cursor** that uses the existing `peer.yml` shape and the existing router fallback loop to get multi-GPU failover working in 1 day instead of waiting for Plan 4's full ~650 LOC.

**Migration path:** when Plan 4 ships, `mesh.yml` becomes the canonical multi-node config. Plan 3.6's `leaders:` field in `peer.yml` is documented as "still supported for single-cluster setups" but operators with multi-node deployments are nudged to use `mesh.yml` for the additional features (drain/resume, per-agent rate limits, admin API).

**The two configs CAN coexist temporarily.** When both are present, `mesh.yml` wins. Documented in Plan 4's R3.0 schema.

## Risks

**High:**
- **Provider key naming collision.** `_try_provider` indexes providers by key. If two leaders are named `leader-rtx-5080` (e.g., user config error), the second silently overwrites the first. Mitigation: validate uniqueness at parse time, fail-loud at startup.

**Medium:**
- **Backwards compat for env var consumers.** Code that reads `MAXIM_LANE_LARGE_REMOTE_URL` directly assumes one URL. Plan 3.6 sets it to the FIRST leader for compat, but downstream code that builds URLs from the env var won't see the fallback. Mitigation: audit env var consumers; the `_external` endpoint and the leader proxy use the registered httpx client (which has all N endpoints), so this only affects code that builds URLs by hand. Should be ~3-5 call sites max.

**Low:**
- **Provider state pollution between calls to different leaders.** `_provider_states` is keyed by provider_key, so leader-1's failures don't pollute leader-2's state. Plan 3 R2.5 + Plan 3.5 R4 already enforce this. No new risk.

## Test protocol

1. **Unit suite:** `python -m pytest tests/unit/test_peer_config.py tests/unit/test_lane_backends.py tests/unit/test_router_typed_exceptions.py -v`
2. **Fast suite:** `python -m pytest tests/ -q -m "not slow" --ignore=tests/integration/test_memory_hub.py`
3. **Manual smoke (single leader):** existing `maxim --sim "say hi"` works unchanged.
4. **Manual smoke (two leaders):**
   - Configure `peer.yml` with two URLs (point both at the same Cloudflare tunnel for the test).
   - Run `maxim peer version` — should report success against the first leader.
   - Run `maxim doctor` — should report two leaders with per-leader status.
5. **Failover smoke (requires real second leader, e.g., RTX 3070):**
   - Configure leader-1 = RTX 5080, leader-2 = RTX 3070.
   - Start a sim.
   - Mid-sim, run `maxim peer restart` (kills leader-1's tunnel briefly).
   - Verify the sim continues without interruption (router falls through to leader-2).
   - Time-to-recovery: should be < 5 seconds (Plan 3's typed-exception failover speed, not the pre-Plan-3 ~52s).

## Related

- [llm_path_refinement.md](llm_path_refinement.md) — meta plan
- [llm_path_fast_failover.md](llm_path_fast_failover.md) — Plan 3 (the typed-exception router loop this plan reuses)
- [llm_path_cancellation_hygiene.md](llm_path_cancellation_hygiene.md) — Plan 3.5 (cancellation contract that makes the failover loop safe)
- [llm_path_operator_visibility.md](llm_path_operator_visibility.md) — Plan 4 (the canonical successor)
- [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) — capability-aware multi-peer dispatch (the next step beyond strict-priority failover)
- [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md) — capability advertisement shell plan
