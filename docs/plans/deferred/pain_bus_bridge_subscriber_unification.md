# PainBus bridge × subscriber unification — close the latent attribution-asymmetry trap

**Status:** SHELL ONLY. Surfaced by the `pain_bus_unification.md` pre-merge architecture review (2026-04-14). NOT scheduled. Open this when (a) `record_tool_start`'s pending-event context is enriched OR (b) the latent risk is observed firing in production.
**Parent index:** [biosystem_unification.md](biosystem_unification.md)
**Wave:** Sibling to Wave 1 / Wave 2 — exact slot TBD when the plan opens.
**Depends on:** `pain_bus_unification.md` (this is the follow-up to it). Probably also wants `executor_bootstrap_unification.md` (which already shipped) and arguably `memory_hub_unification.md`.

## What this plan exists to fix

The pain_bus_unification audit closed Gap A (three CLI sites silently skipping NAc bus subscription) by introducing `build_pain_bus(*, hippocampus, nac, ...)` as the canonical PainBus construction door. Both `create_pain_memory_subscriber(hippocampus)` and `create_pain_nac_subscriber(nac)` are auto-wired when their subjects are non-None.

Post-migration, the CLI fast paths now have BOTH the NAc bus subscriber AND `ToolPainBridge` wired to the same `PainBus` instance. When `body.py::_publish_pain` fires mid-tool:

1. `PainBus.publish` dispatches to all direct subscribers
2. `ToolPainBridge._on_embodiment_pain` checks `bool(self._pending_tools)` → returns early (the guard introduced by the SEM execution hook Stage 1 fix)
3. `create_pain_nac_subscriber._on_pain` does NOT have this guard. It calls `nac.record_outcome_full(...)` with the rich 7-key signal context.
4. NAc walks `_pending_events`, finds the pending tool event recorded at `record_tool_start` time with `{"params": ...}` context (1 key)
5. `_context_similarity({"params":...}, {7 keys})` = `0 / 1` = `0.0`, below the 0.5 threshold
6. **No link forms via this path.** Benign no-op.

Then after `tool.run()` returns:

7. Executor calls `record_tool_embodiment_failure` → pops pending event by `(tool_name, invocation_id)` direct lookup → calls `nac.record_outcome` (NOT `record_outcome_full`) → ONE link forms via direct attribution

**Result today: correct behavior. No double-counting.** But the correctness is **load-bearing on the context-similarity mismatch** between the pending tool event's `{"params": ...}` context and the rich outcome context. It is *not* load-bearing on any guard, threshold, or ordering — just on the bare arithmetic that two contexts with zero key overlap produce a similarity of 0.0.

## The latent trap

`bridges/tool_pain_bridge.py::_on_embodiment_pain` lines 367-371 explicitly contemplate enriching `record_tool_start`'s context to include `entity` for narrowing the broad-guard semantics in a future concurrent-executor refactor. **The moment that enrichment lands, the subscriber's `record_outcome_full` will start matching the pending tool event** (because `entity` is one of the 7 keys in the body's outcome context, so similarity becomes `1/2 = 0.5`, exactly at threshold). At that point:

- Mid-tool publish: subscriber links the pending tool event with valence NEGATIVE
- After tool returns: bridge's `record_tool_embodiment_failure` pops the SAME pending tool event by direct lookup and links it AGAIN with valence NEGATIVE

Two `CausalLink` entries on the same `(event_signature, outcome)` pair, two Rescorla-Wagner updates, double-counted RPE. The bug shape is: **silent drift from one update to two updates as soon as anyone touches `record_tool_start`'s context dict**.

The substrate P2 cascade test would not catch this because it operates at the link-existence level, not the link-count or RPE-magnitude level.

## Why the obvious fixes are band-aids

**Band-aid A: Add `_pending_tools` guard to `create_pain_nac_subscriber`.** This couples the subscriber (lives in `proprioception/pain_bus.py`) to bridge state (lives in `bridges/tool_pain_bridge.py`). The subscriber is per-bus, the bridge is per-agent — there's no clean ownership relationship. To make this work, the subscriber would need a reference to "the bridge for this bus," which means `create_pain_nac_subscriber` grows a `bridge=` parameter, which means every test that uses it grows a bridge mock, which means the bridge becomes a leaked dependency for every NAc consumer. **Coupling band-aid that violates layer boundaries.**

**Band-aid B: Add a global "tool in flight" flag.** Re-introduces the ContextVar-based signal-stash pattern that Substrate P2 Stage 2 explicitly forbade for re-entrancy hazard reasons. **Pre-merge-review-rejected pattern.**

**Band-aid C: Make `create_pain_nac_subscriber` skip events in `_pending_events` whose type is `"tool"`.** Closer, but introduces type-string knowledge into the subscriber that doesn't belong there, and breaks for any future "direct-attribution" event type. **Type-coupling band-aid that doesn't generalize.**

**Band-aid D: Ship `pain_bus_unification` without wiring `create_pain_nac_subscriber` on the CLI paths.** Re-introduces Gap A. **Reverts the structural fix.**

## The non-band-aid fix (Option B from the architecture review)

The clean answer is: **`create_pain_nac_subscriber` and `ToolPainBridge` are not complements. They are two different attribution mechanisms that should not coexist on the same bus.** The bridge handles direct-attribution for in-flight tools; the subscriber handles context-similarity attribution for out-of-band events (autonomous SEM ticks, ambient sensor decay, sandbox events with no tool in flight). When both are wired to the same bus, the bridge should be the canonical NAc-attribution mediator and the subscriber should be subordinate to it.

The structural fix is to extend `build_pain_bus` to accept a `tool_pain_bridge=` parameter and skip wiring the NAc subscriber when a bridge is present:

```python
def build_pain_bus(
    *,
    hippocampus: Hippocampus | None,
    nac: NAc | None,
    tool_pain_bridge: ToolPainBridge | None = None,
    additional_subscribers: tuple[Callable[[PainSignal], None], ...] = (),
    ...,
) -> PainBus:
    bus = PainBus(...)
    if hippocampus is not None:
        bus.subscribe(create_pain_memory_subscriber(hippocampus))
    if nac is not None:
        if tool_pain_bridge is not None:
            # Bridge mediates NAc attribution for in-flight tools.
            # Subscribe a guarded variant that defers to the bridge
            # when a tool is pending and falls through to context
            # similarity only when it's not.
            bus.subscribe(create_bridge_aware_pain_nac_subscriber(nac, tool_pain_bridge))
        else:
            bus.subscribe(create_pain_nac_subscriber(nac))
    ...
```

OR, even cleaner, **invert the wiring**: `ToolPainBridge.__init__` subscribes itself to the bus AND ALSO becomes the NAc attribution mediator (the bridge's `_on_embodiment_pain` becomes the only path to NAc, with the context-similarity fallback moved into the bridge). Then `create_pain_nac_subscriber` becomes a fallback for buses that don't have a bridge wired. The bus consumer chooses one or the other, never both.

Both shapes need the same audit + design discipline as Wave 1: walk every PainBus + ToolPainBridge construction site, map who currently wires what, confirm the bridge ownership story is consistent, write tests that pin both the in-flight and out-of-band paths.

## Why we are NOT fixing it in this PR

1. **The current behavior is correct.** Zero double-counting today.
2. **The risk is latent on a specific future change** (`record_tool_start` context enrichment). That change has not happened.
3. **The structural fix to Gap A still ships cleanly.** Closing this latent risk is its own audit + design pass — fits the "one unification per PR" rule (L6).
4. **A regression test pins the current behavior** so the trap fires loudly the moment anyone enriches the context, forcing the deeper plan to be opened then.
5. **No band-aid is acceptable** — every shortcut listed above is a coupling violation, a type leak, or a Substrate P2 Stage 2 forbidden pattern. The non-band-aid fix is genuinely a separate plan.

## Trigger conditions to open this plan

- Anyone adds `entity`, `failure_mode`, or any context key to the dict passed to `record_tool_start` from the executor or modulator paths.
- Any new test fails because `create_pain_nac_subscriber` linked a pending tool event via context similarity (the regression test in `tests/unit/test_pain_bus.py::TestBuildPainBus::test_subscriber_does_not_link_pending_tool_event` is designed to catch exactly this).
- The bridge guard is narrowed from `bool(self._pending_tools)` to a per-`(tool_name, entity)` match (per the broad-guard-semantics docstring at `tool_pain_bridge.py:367-371`) — that change inherently enriches the pending-event context and trips the trap.
- A future plan introduces concurrent in-flight tool executions sharing a single `ToolPainBridge` (the broad-guard contract assumes serialized executors).
- The substrate P2 cascade test or any new bio-pipeline integration test starts failing with "duplicate causal link" or "RPE double-counted" symptoms.

## Estimated scope

~200 LOC for the bridge-aware subscriber variant + ~150 LOC of new tests + ~50 LOC of audit + ~50 LOC of doc + memory refinement. Single PR. ~3-4 days of focused work. Pre-merge review round mandatory (the bridge × subscriber boundary is exactly the kind of subtle cross-module interaction that the executor unification + pain_bus unification reviews caught; expect ≥3 cross-confirmed findings).

## Out of scope (for THIS shell)

- The actual implementation. Open the plan first, do the audit, surface the existing PainBus × ToolPainBridge construction sites, then design.
- Any change to `create_pain_nac_subscriber` itself — the function is correct as a standalone subscriber for buses without a bridge.
- Any change to `_context_similarity` — the directional denominator is load-bearing per the Substrate P2 Stage 2 lesson.

## Cross-references

- [pain_bus_unification.md](pain_bus_unification.md) — the parent plan that surfaced this trap during pre-merge review
- [biosystem_unification.md](biosystem_unification.md) — central index
- [executor_bootstrap_unification.md](executor_bootstrap_unification.md) — `build_executor` precedent + bridge wiring rules
- `CLAUDE.md` "Tool-invoked embodiment pain attributes directly, not via context similarity" invariant — the rule the bridge guard enforces
- `CLAUDE.md` "Context-similarity attribution is the wrong mechanism when a direct lookup key exists" lesson — the SEM execution hook Stage 1 root-cause writeup
