# Plan 4 Stage C4: Wire the Router to Drain State

**Status:** PLAN (2026-04-17). Pre-design review pending.
**Parent:** [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) Stage C4.
**Predecessor:** C3.4 (PR #142) shipped `/v1/debug/vram`. C4 is the first stage that changes **runtime routing behavior**, not just operator tooling.
**Gating:** C4.5 (auto-drain), C5 (capacity-aware routing), and C6 (admin API) all depend on C4.

---

## 1. The problem

`grep -r "read_drained_nodes\|drained_nodes\|drain_state" src/maxim/models/language/` returns **zero matches**. The router does not know that drain exists. An operator who runs `maxim peer --node leader-desk drain` today sees the node disappear from `list-nodes` and `doctor`, but the next inference call from a co-resident agent still hits it.

This is the single biggest gap between "we have a mesh" and "the mesh is reactive."

---

## 2. Design decisions (user-confirmed 2026-04-17)

### Q1: Where does drain logic live?

**Decision: `drain_constraint` callback injected into `LLMRouter`.**

Add an optional `drain_constraint: Callable[[str], bool] | None = None` parameter to `LLMRouter.__init__`. The callback receives a provider key and returns `True` if the provider should be skipped (drained). Called inside `_candidate_providers()` alongside the existing backoff/budget/context filters.

**Why not a wrapper class:** A `MeshAwareRouter` decorator would silently remove providers before the router sees them, making `dispatch_exhausted` misleading ("all providers failed" when the truth is "2 of 4 were drained"). The callback approach lets drained providers participate natively in attempt recording with `"outcome": "drained"`.

**Why not per-attempt (`_try_provider`):** Filtering inside `_try_provider` means we've already entered the try/except block for nothing. Candidate filtering is the right layer.

### Q2: Provider-name to mesh-node-name mapping

**Decision: URL lookup table derived from `mesh.yml` at router construction.**

At router init, if `mesh.yml` exists, build a `canonical_url -> node_name` lookup table from `MeshConfig.nodes`. The `drain_constraint` callback receives a provider key, looks up its URL from the provider config, canonicalizes it, and checks if the corresponding node name is in the drained set.

URL canonicalization reuses `probe_cache._canonical_url` (strips trailing `/` and `/v1`). Cloud providers (Anthropic, OpenAI, etc.) have no mesh.yml entry, so they're never in the table -- no special-casing needed. If mesh.yml doesn't exist, the table is empty and drain is a no-op -- zero behavior change for non-mesh installs.

**Why not explicit `mesh_node` config field:** Would require operators to maintain the mapping in two places (`llm.json` + `mesh.yml`). Drift risk. Single source of truth from `mesh.yml` is better.

### Q3: In-memory cache strategy

**Decision: mtime polling cache.**

Cache the drain set as a `(mtime: float, drained: frozenset[str])` tuple on the constraint object. Each dispatch does one `os.stat()` call (~1us Linux, ~5us macOS). If mtime matches, return cached set. If changed, read the file under filelock and update the cache.

Optional `MAXIM_DRAIN_CACHE_TTL_S` env var (default 1.0) skips the stat call entirely if the last check was within TTL seconds. Useful for high-throughput benchmarks where 5us/call matters.

**Why not inotify/kqueue:** Three platform-specific code paths + daemon thread lifecycle for a signal that changes once per operator action. Overkill.

**Why not process-internal pub/sub:** Dead on arrival. `maxim peer --node X drain` runs in a CLI process; the router runs in the sim/agent process. Two different PIDs. The CLI writes the file; the router needs to notice.

### Q4: Check frequency

**Decision: per-dispatch (every `_complete_text_locked` call).**

The mtime cache makes this cheap. If an operator drains a node mid-conversation, the next inference call respects it. Per-worker-binding would leave drained nodes in the dispatch path until process restart -- defeats the purpose.

### Q5: All-drained contract

**Decision: `dispatch_exhausted_all_drained` event for all-drained, regular `dispatch_exhausted` with `"outcome": "drained"` entries for mixed.**

When the drain filter eliminates ALL candidates and there WERE providers before filtering, emit a distinct `dispatch_exhausted_all_drained` WARNING with fix hint: `"all N providers are drained -- run 'maxim peer list-drained' to inspect"`. When some candidates survive drain filtering but then all fail for other reasons, use the regular `dispatch_exhausted` with `"outcome": "drained"` entries in the attempt list. Operators get the right signal in both scenarios.

---

## 3. Implementation plan

### Stage 1: `DrainConstraint` factory + mtime cache (~100 LOC)

**New file: `src/maxim/peer/drain_routing.py`**

```python
class DrainConstraint:
    """Stateful callback that checks provider drain status.
    
    Built from mesh.yml topology + drain state file. Caches drain set
    in memory, refreshing on file mtime change. Injected into LLMRouter
    as drain_constraint parameter.
    """
    
    def __init__(
        self,
        url_to_node: dict[str, str],    # canonical_url -> node_name
        provider_urls: dict[str, str],   # provider_key -> canonical_url
        drain_path: Path,
        cache_ttl_s: float = 1.0,
    ) -> None: ...
    
    def is_drained(self, provider_key: str) -> bool:
        """Return True if provider_key maps to a drained mesh node."""
        ...
    
    def drained_providers(self) -> frozenset[str]:
        """Return set of currently drained provider keys (for logging)."""
        ...


def build_drain_constraint(
    mesh_cfg: MeshConfig,
    provider_cfgs: dict[str, dict],
) -> DrainConstraint | None:
    """Factory. Returns None if no provider maps to any mesh node.
    
    Builds the url_to_node table from mesh_cfg.nodes and the
    provider_urls table from provider configs with a 'url' field.
    Uses probe_cache._canonical_url for normalization.
    """
    ...
```

**Key invariants:**
- `DrainConstraint` never writes to drain state. Read-only.
- `build_drain_constraint` returns `None` when no provider URL matches any mesh node URL. The router skips drain checking entirely -- zero overhead for cloud-only or local-only setups.
- The mtime cache reads the drain file under filelock (10s timeout, same as `read_drained_nodes`). File parse reuses `drain_state._parse_drain_file` if we extract it, or duplicates the trivial line-split logic (it's 5 lines).
- `MAXIM_DRAIN_CACHE_TTL_S` env var (clamped 0.0-60.0, default 1.0).

### Stage 2: Router integration (~80 LOC diff in router.py)

**`LLMRouter.__init__` gains one parameter:**
```python
def __init__(
    self, 
    cfg: LLMConfig | None = None,
    *,
    drain_constraint: Callable[[str], bool] | None = None,
) -> None:
    ...
    self._drain_constraint = drain_constraint
```

**`_candidate_providers` gains one filter:**
```python
# In the candidate filtering loop, after backoff check:
if self._drain_constraint is not None and self._drain_constraint(provider_key):
    self._record_attempt_outcome(provider_key, outcome="drained")
    continue
```

Wait -- `_record_attempt_outcome` requires `_inference_lock` and is called inside the dispatch loop, not the candidate filter. The drain skip needs to record the outcome at the right layer. Two sub-options:

**Sub-option 2a:** Record "drained" in the candidate filter by appending to a pre-dispatch drain list, then merge into `_dispatch_attempts` at dispatch start. Cleaner separation.

**Sub-option 2b:** Don't record drained providers in `_dispatch_attempts` at all. Instead, pass the drain count to `_emit_dispatch_exhausted` as a separate field. Simpler.

**Chosen: 2b.** The drained providers didn't fail -- they were never tried. They belong in the exhausted event metadata, not the attempt list. This keeps `_dispatch_attempts` as "things we actually tried."

**`_emit_dispatch_exhausted` gains drain context:**
```python
def _emit_dispatch_exhausted(self, *, request_context, total_elapsed_ms, drained_keys=None):
    ...
    data = {
        "request_id": ...,
        "attempts": attempts,
        "drained_providers": sorted(drained_keys) if drained_keys else [],
    }
```

**New `_emit_dispatch_exhausted_all_drained`:**
```python
def _emit_dispatch_exhausted_all_drained(self, *, request_context, drained_keys):
    """All candidates were drained. Distinct event with fix hint."""
    ctx = _normalize_request_context(request_context)
    log_structured(
        logger,
        logging.WARNING,
        event="dispatch_exhausted_all_drained",
        data={
            "request_id": ctx.request_id,
            "agent_id": ctx.agent_id,
            "drained_providers": sorted(drained_keys),
            "fix": "all providers are drained -- run 'maxim peer list-drained'",
        },
    )
```

### Stage 3: Production wiring (~30 LOC across call sites)

Wire `build_drain_constraint` into the router construction sites. The router is constructed in:
- `runtime/lane_backends.py` or wherever `LLMRouter()` is instantiated
- Need to audit exact call sites

Pattern at each site:
```python
mesh_cfg = read_or_synthesize_mesh_config()
drain_constraint = None
if mesh_cfg is not None:
    dc = build_drain_constraint(mesh_cfg, router_cfg.providers)
    if dc is not None:
        drain_constraint = dc.is_drained

router = LLMRouter(cfg, drain_constraint=drain_constraint)
```

### Stage 4: Tests (~150 LOC)

**`tests/unit/test_drain_routing.py`** (new file):
- `TestDrainConstraint`: mtime cache hit/miss, file absent returns empty, file with nodes returns correct set, TTL skip, provider key not in URL table returns False
- `TestBuildDrainConstraint`: returns None for cloud-only providers, builds correct table from mesh.yml + provider URLs, canonical URL normalization
- `TestDrainConstraintIntegration`: end-to-end with real drain_state file write + constraint read

**`tests/unit/test_llm_fallback.py`** (additions to existing):
- `test_drained_provider_skipped_in_dispatch`: inject drain_constraint that returns True for one provider, verify it's skipped and the other is tried
- `test_all_drained_emits_distinct_event`: all providers drained, verify `dispatch_exhausted_all_drained` log event
- `test_mixed_drain_and_failure`: 1 drained + 1 fails, verify regular `dispatch_exhausted` with `drained_providers` field
- `test_no_drain_constraint_is_noop`: drain_constraint=None, verify zero behavior change
- `test_drain_constraint_not_called_for_cloud`: cloud provider not in URL table, constraint never consulted

---

## 4. Files touched

| File | Change | Risk |
|---|---|---|
| `src/maxim/peer/drain_routing.py` | **NEW.** DrainConstraint + factory. | Low -- leaf module, no imports from runtime. |
| `src/maxim/models/language/router.py` | Add `drain_constraint` param, filter in `_candidate_providers`, drain context in exhausted events. | Medium -- core dispatch path. Pre-merge review mandatory. |
| `src/maxim/runtime/lane_backends.py` or router construction site | Wire `build_drain_constraint` into router init. | Low -- one-time construction. |
| `tests/unit/test_drain_routing.py` | **NEW.** DrainConstraint unit tests. | None. |
| `tests/unit/test_llm_fallback.py` | Drain integration tests. | Low -- additive. |

**NOT touched:** `drain_state.py`, `mesh_config.py`, `leader_proxy.py`, any substrate path.

---

## 5. Invariants (do not break)

1. **`DrainConstraint` is read-only.** It never writes to drain state. It never calls `drain_node()` or `resume_node()`. Writing is the CLI's job (C2) or the admin API's job (C6).
2. **`drain_constraint=None` is zero behavior change.** No mesh.yml = no constraint = no drain filtering. Non-mesh installs must be completely unaffected.
3. **`build_drain_constraint` returns `None` when no URL matches.** Cloud-only setups skip drain entirely. No wasted stat() calls.
4. **The mtime cache never holds a stale set for longer than `MAXIM_DRAIN_CACHE_TTL_S`.** The stat() call is the freshness check. If the file doesn't exist, the drained set is empty.
5. **`dispatch_exhausted_all_drained` fires IFF all candidates were eliminated by drain AND there were candidates before drain.** If there are zero providers configured, that's a different bug -- the existing `dispatch_exhausted` handles it.
6. **Drained providers do NOT appear in `_dispatch_attempts`.** They were never tried. They appear in the `drained_providers` metadata field of the exhausted event.
7. **The URL lookup table is built once at router construction.** It does NOT update when mesh.yml changes. Router restart picks up topology changes. This is consistent with the existing behavior: the router's provider list is static for its lifetime.
8. **Long-backoff branches (auth 300s, model_missing 60s, inference_broken 15s) do NOT call `_note_provider_failure`** -- this existing invariant from Plan 3 is unmodified. Drain is orthogonal to backoff.

---

## 6. What this does NOT do (deferred)

- **Auto-drain on persistent failure** -- C4.5. Requires drain-entry tagging (`# auto` vs `# operator`) and an auto-undrain story.
- **Capacity-aware routing** -- C5. Requires `/v1/debug/vram` data flowing into routing decisions.
- **Hot mesh.yml reload** -- the URL table is built at router construction. Topology changes require restart. Hot reload is a future optimization.
- **Admin API drain/resume** -- C6. Today drain/resume is CLI-only.

---

## 7. Estimated effort

~350 LOC + tests across 2-3 files. 1 session for implementation + 1 session for pre-merge review fold. The design is straightforward once the 5 questions are settled (which they now are).

---

## 8. Pre-merge review checklist

- [ ] Executor lens: backoff/drain interaction (drain skip must NOT reset consecutive_errors), thread safety of mtime cache under concurrent dispatch
- [ ] Architecture lens: import hygiene (router.py must not import drain_state directly -- only through the injected callback), no circular imports, URL normalization edge cases
- [ ] Regression: full test suite + existing llm_fallback tests unmodified
