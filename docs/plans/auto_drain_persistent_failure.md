# Plan 4 Stage C4.5: Auto-Drain on Persistent Failure

**Status:** PLAN (2026-04-17). User-confirmed design decisions.
**Parent:** [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) Stage C4.5.
**Predecessor:** C4 (PR #148) wired the router to drain state. C4.5 is the first *automatic* reactive behavior -- the mesh drains nodes itself.
**Deferred:** Auto-undrain via periodic health probe (C4.6). Ship C4.5 as auto-drain only; operator resumes manually.

---

## 1. The problem

After C4, the router skips drained nodes. But drain is still operator-initiated. When a mesh node goes down, the router backs off exponentially (max 60s), tries again, fails, backs off again -- forever. Each retry wastes a dispatch cycle and adds latency to the fallback path. The operator has to notice and manually `maxim peer --node X drain`.

C4.5 makes the mesh self-healing: after a failure threshold, the router auto-drains the node and stops trying until the operator resumes it (or, in future C4.6, until a health probe confirms recovery).

---

## 2. Design decisions (user-confirmed 2026-04-17)

### Q1: Trigger threshold -- type-aware, two tiers

Different failure types have different recovery profiles. Auth failures are permanent until the operator fixes the key; network failures often self-heal after a tunnel reconnect.

**Permanent failures** (auto-drain after 1 long-backoff occurrence):
- `BackendAuthFailed` (401/403) -- key won't fix itself
- `BackendModelMissing` (404) -- model won't self-load

**Transient failures** (auto-drain after 5 consecutive):
- `BackendDown` -- node unreachable, may recover
- `BackendTimeout` -- may be transient load
- `BackendInferenceBroken` -- GPU crash, needs restart
- Generic `BackendError` / unclassified

The threshold for transient failures is configurable via `MAXIM_AUTO_DRAIN_THRESHOLD` env var (clamped 2-20, default 5). Permanent thresholds are not configurable -- 1 is always correct.

The existing `ProviderState.consecutive_errors` counter is the input signal. `_note_provider_success()` already resets it to 0, so a successful call before the threshold is hit prevents auto-drain.

### Q2: Write mechanism -- post-dispatch callback

The C4 invariant says `DrainConstraint` (the read path) is read-only. Auto-drain writes through a separate `auto_drain_callback: Callable[[str, str], None] | None` injected into `LLMRouter` alongside `drain_constraint`.

The callback signature is `(provider_key: str, reason: str) -> None`. It's invoked **after** `_inference_lock` is released, at the end of `_complete_text_locked`, to avoid nesting the filelock inside the inference lock.

The callback implementation lives in `drain_routing.py::AutoDrainWriter`:
1. Maps provider_key -> mesh node name (reusing the same URL table from `DrainConstraint`)
2. Writes `node_name  # auto:<ISO-timestamp> reason:<failure_type>` to the drain file
3. Logs a structured `auto_drain` WARNING event with the node name and reason

### Q3: Entry tagging -- inline `# auto:` comments

Auto-drain entries are tagged in the existing drain file:

```
leader-desk  # auto:2026-04-17T12:34:56 reason:auth_failed
```

`drain_state._load_names` already strips `#` comments, so existing readers see plain `leader-desk`. A new `_load_tagged_entries(path) -> dict[str, str | None]` function reads raw lines and returns `{name: tag_or_none}`. Used by:
- `maxim peer list-drained` to show origin (future CLI enhancement)
- C4.6 auto-undrain to identify which entries are auto-clearable

Fail-safe: missing or malformed tag = treated as operator drain (sticky, never auto-cleared). Erring on the side of not auto-clearing is correct.

### Q4: Auto-undrain -- deferred to C4.6

C4.5 ships auto-drain only. The operator runs `maxim peer --node X resume` to un-drain. The auto-undrain probe loop (background thread, periodic health checks against auto-drained nodes) is a separate design problem that doubles the scope. Ship, see it work, iterate.

---

## 3. Implementation plan

### Stage 1: `AutoDrainWriter` + tagged entries (~100 LOC)

**Extend `src/maxim/peer/drain_routing.py`:**

```python
# ── auto-drain thresholds ──────────────────────────────────────────────

_PERMANENT_FAILURE_THRESHOLD = 1   # auth_failed, model_missing
_DEFAULT_TRANSIENT_THRESHOLD = 5   # down, timeout, inference_broken
_MIN_TRANSIENT_THRESHOLD = 2
_MAX_TRANSIENT_THRESHOLD = 20

PERMANENT_FAILURE_TYPES = frozenset({"auth_failed", "model_missing"})


class AutoDrainWriter:
    """Writes auto-drain entries to the drain state file.

    Injected into LLMRouter as auto_drain_callback. Maps provider keys
    to mesh node names (same URL table as DrainConstraint) and writes
    tagged entries.
    """

    def __init__(
        self,
        provider_to_node: dict[str, str],
        drain_path: Path,
    ) -> None: ...

    def maybe_auto_drain(self, provider_key: str, reason: str) -> None:
        """Write an auto-drain entry if the provider maps to a mesh node.

        Idempotent: if the node is already drained (operator or auto),
        this is a no-op. Does NOT overwrite operator drains.
        """
        ...


def _load_tagged_entries(path: Path) -> dict[str, str | None]:
    """Read drain file, returning {name: raw_tag_or_none}.

    For 'leader-desk  # auto:2026-04-17 reason:auth' returns
    {'leader-desk': 'auto:2026-04-17 reason:auth'}.
    For 'mac-studio' (no tag) returns {'mac-studio': None}.
    """
    ...
```

### Stage 2: Router integration (~50 LOC diff in router.py)

**`LLMRouter.__init__` gains one more callback:**
```python
def __init__(
    self,
    cfg: LLMConfig | None = None,
    *,
    drain_constraint: Callable[[str], bool] | None = None,
    auto_drain_callback: Callable[[str, str], None] | None = None,
) -> None:
    ...
    self._auto_drain_callback = auto_drain_callback
```

**Post-dispatch threshold check in `_complete_text_locked`:**

After the dispatch loop (whether success or exhausted), check each provider that failed in this dispatch. If `consecutive_errors >= threshold` for the failure type, invoke the callback. The check happens inside `_inference_lock` but the callback itself can release the lock first.

Actually, simpler: the check happens inside `_try_provider` at the point where `_note_provider_failure` / `_set_long_backoff` is called. Right after incrementing `consecutive_errors`, check the threshold. If crossed, schedule the auto-drain by appending to a `_pending_auto_drains: list[tuple[str, str]]` buffer. After `_inference_lock` is released, process the buffer.

```python
# In _try_provider, after each _note_provider_failure / _set_long_backoff:
if self._auto_drain_callback is not None:
    state = self._provider_states[provider_key]
    threshold = (
        _PERMANENT_FAILURE_THRESHOLD
        if outcome in PERMANENT_FAILURE_TYPES
        else _transient_threshold()
    )
    if state.consecutive_errors >= threshold:
        self._pending_auto_drains.append((provider_key, outcome))
```

```python
# In complete_text (the public method), after releasing _inference_lock:
if self._pending_auto_drains:
    for key, reason in self._pending_auto_drains:
        self._auto_drain_callback(key, reason)
    self._pending_auto_drains.clear()
```

### Stage 3: Production wiring (~15 LOC in lane_backends.py)

Extend the existing C4 wiring block to also build and inject the `AutoDrainWriter`:

```python
if dc is not None:
    drain_constraint = dc.is_drained
    writer = AutoDrainWriter(dc._provider_to_node, drain_path)
    auto_drain_callback = writer.maybe_auto_drain
```

### Stage 4: Tests (~120 LOC)

**`tests/unit/test_drain_routing.py` additions:**
- `TestAutoDrainWriter`: writes tagged entry, idempotent on existing drain, skips unknown provider, tagged entry parseable by `_load_tagged_entries`
- `TestLoadTaggedEntries`: parses plain names, tagged names, mixed, comments
- `TestRouterAutoDrain`: permanent failure type triggers after 1, transient after 5, success resets counter (no drain), callback invoked outside inference lock

---

## 4. Files touched

| File | Change | Risk |
|---|---|---|
| `src/maxim/peer/drain_routing.py` | AutoDrainWriter, thresholds, `_load_tagged_entries` | Low -- additive to existing module |
| `src/maxim/models/language/router.py` | `auto_drain_callback` param, threshold check in `_try_provider`, pending buffer processing | Medium -- dispatch path. Review mandatory. |
| `src/maxim/runtime/lane_backends.py` | Wire AutoDrainWriter alongside DrainConstraint | Low -- extends existing wiring block |
| `tests/unit/test_drain_routing.py` | Auto-drain tests | None |

**NOT touched:** `drain_state.py` (we write to the file directly via `atomic_write_text`, same pattern as `drain_state._save_names`), `mesh_config.py`, `leader_proxy.py`.

---

## 5. Invariants

1. **`DrainConstraint` remains read-only.** Auto-drain writes go through `AutoDrainWriter`, a separate object.
2. **Auto-drain writes happen OUTSIDE `_inference_lock`.** The pending buffer is populated inside the lock; the writes are flushed after release.
3. **Auto-drain entries are tagged `# auto:<timestamp> reason:<type>`.** Missing tag = operator drain = sticky.
4. **Auto-drain is idempotent.** If a node is already drained (operator or auto), the write is a no-op.
5. **`_note_provider_success` resets `consecutive_errors` to 0.** A successful call before the threshold prevents auto-drain.
6. **Permanent failures (auth, model_missing) auto-drain after 1.** Transient failures after `MAXIM_AUTO_DRAIN_THRESHOLD` (default 5, clamped 2-20).
7. **`auto_drain_callback=None` is zero behavior change.** Non-mesh installs are completely unaffected.
8. **Auto-drain does NOT auto-undrain.** Operator must `resume` manually. Auto-undrain deferred to C4.6.

---

## 6. Env vars

```bash
MAXIM_AUTO_DRAIN_THRESHOLD=5    # Transient failure threshold (clamped 2-20, default 5)
```

Add to CLAUDE.md env var table and `conftest.py` autouse scrub fixture.

---

## 7. What this does NOT do (deferred)

- **Auto-undrain** (C4.6) -- periodic health probe against auto-drained nodes
- **Drain entry origin in CLI output** -- `list-drained` showing `[auto]` vs `[operator]` tags
- **C3.3 install-drain tagging** -- retrofit `# install:<timestamp>` tags
- **Admin API auto-drain control** -- `/v1/admin/auto-drain-config` (C6)

---

## 8. Estimated effort

~285 LOC across 3 source files + tests. Single session. Pre-merge review mandatory (dispatch path changes).
