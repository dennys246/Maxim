# Unified Typed Event Bus — Deferred Post-1.0 Refactor

**Status:** Deferred. Not on the critical path to 1.0. Not a blocker for the research claim.
**Revive when:** (a) a cross-layer debugging bug surfaces that a unified bus would have caught immediately, OR (b) a second developer joins and needs to trace cross-layer event flow without reading seven files, OR (c) an external contributor is writing a bio-system against the `BioSystem` Protocol and needs a predictable subscription model for percept/memory/reward events, OR (d) observability tooling (tracing, replay, offline analysis) needs a single place to hook into.

## Current state (why this isn't urgent)

The Maxim codebase has **five separate event transports** that together form an informal coordination layer:

1. **`LocalMessageBus`** ([agent_pool.py:76-81](../../src/maxim/runtime/agent_pool.py#L76-L81)) — pool-wide message transport, intentionally shared across agents so they can coordinate. One bus per `AgentPool`, not per agent.
2. **`AgentBus`** (per-`MaximAgent` instance, [agents/bus.py](../../src/maxim/agents/bus.py)) — internal agent coordination. Separate from the pool-wide bus.
3. **`ConversationalSource`** ([simulation/conversational_source.py](../../src/maxim/simulation/conversational_source.py)) — percept injection, ad-hoc per-sensor methods (`inject_cli`, `inject_pain`, plus `inject_sensor` after F0.8).
4. **Direct cross-layer callbacks** — `hippocampus.register_deletion_callback(nac.remove_memory)` at [memory_hub.py:169](../../src/maxim/integration/memory_hub.py#L169) and similar. These bypass `MemoryHub`'s intended coordinator role.
5. **`MemoryHub`** ([integration/memory_hub.py](../../src/maxim/integration/memory_hub.py)) — nominally the cross-layer coordinator, but the direct callbacks above bypass it. Today it's a partial mediator, not a real one.

**Why this is tolerable today:** the substrate plan's research claim holds at the memory layer, not the message layer. ATL/Hippocampus/NAc/PerceptTraceBuffer are all per-agent and per-instance (verified by the iceberg sweep in foundations_plan's F0.5). The fragmented bus situation doesn't introduce correctness bugs for the 1.0 claim — it introduces *cognitive* cost for anyone trying to understand cross-layer event flow, which is tolerable for a single-developer project.

**Why it stops being tolerable later:** platform ambition. When external contributors start writing against the `BioSystem` Protocol (see [bio_system_plugin_plan.md](bio_system_plugin_plan.md)), they'll need a predictable way to subscribe to events — "tell me when any agent captures a new ATL concept" should be one line, not a tour of five files to figure out which transport carries that event. When observability tooling is added (tracing, replay, offline analysis), having one place to hook into is worth orders of magnitude more than having five.

## What the unified bus would look like

One typed-topic bus with structural subscription:

```python
class EventBus(Protocol):
    def publish(self, topic: str, event: Event) -> None: ...
    def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], None],
        *,
        filter: Callable[[Event], bool] | None = None,
        agent_id: str | None = None,  # default: subscribe to all agents
    ) -> SubscriptionHandle: ...
    def unsubscribe(self, handle: SubscriptionHandle) -> None: ...
```

Typed event classes per topic:

```python
@dataclass(frozen=True)
class PerceptEvent:
    agent_id: str
    percept: Percept
    tick: int

@dataclass(frozen=True)
class MemoryDeletedEvent:
    agent_id: str
    node_id: NodeId
    reason: Literal["decay", "pruning", "explicit"]

@dataclass(frozen=True)
class RewardEvent:
    agent_id: str
    target_node_id: NodeId
    magnitude: float
    source: str  # "plan_outcome", "user_feedback", etc.
```

Every bio-system subscribes to the topics it cares about. `MemoryHub` becomes a real coordinator because it owns the bus, not a vestigial class that hints at coordination without doing it.

## Phased refactor plan when revived

### Phase 1 — Define `EventBus` Protocol and typed events (~200 LOC)

Lives in `src/maxim/contracts/event_bus.py` and `src/maxim/contracts/events.py`. Part of the incremental contracts layer from substrate_plan, but pulled forward for this refactor. No implementation yet — just the types.

### Phase 2 — Implement `InProcessEventBus` as the first backend (~300 LOC)

Synchronous dispatch, typed topics, filter-based subscription, per-agent scoping. Single implementation, no remote transport. Drop-in replacement for `LocalMessageBus` at the `AgentPool` level.

### Phase 3 — Migrate direct callbacks to bus subscriptions (~400 LOC)

The cross-layer callbacks in `MemoryHub` become subscriptions:

```python
# Before
hippocampus.register_deletion_callback(nac.remove_memory)

# After
bus.subscribe(
    "memory_deleted",
    lambda evt: nac.remove_memory(evt.agent_id, evt.node_id),
    agent_id=self.agent_id,
)
```

This is the riskiest phase — it touches every cross-layer interaction the codebase currently has. Land it behind a feature flag, run the full fast test suite + sim suite, compare behavior before and after. Only remove the flag once there's confidence.

### Phase 4 — Migrate `ConversationalSource` to publish percept events (~150 LOC)

`inject_sensor(modality, **fields)` becomes `bus.publish("percept", PerceptEvent(...))`. The agent loop's `percept_source.next_percept()` becomes a bus subscription on the `"percept"` topic.

### Phase 5 — Deprecate `AgentBus` and `LocalMessageBus` (~100 LOC deletion + migration)

Once Phase 3 and 4 land, the old buses have no unique callers. Delete them, or keep them as thin shims over `EventBus` for a release, then delete.

### Phase 6 — Observability hooks (~200 LOC)

Add a `BusObserver` interface that can subscribe to all topics for tracing, replay, offline analysis. This is where the deferred value shows up — a unified bus makes observability trivial, where today it would require hooking into five different transports.

**Total scope when revived:** ~1,350 LOC across six phases. 3–5 weeks of focused work. Most of the risk is in Phase 3 (migrating cross-layer callbacks), which is why this plan is deferred — the cost is high, the benefit doesn't show up until external contributors or observability needs create demand.

## Why not incremental like the contracts layer?

The contracts layer can accrete during substrate phase work because each Protocol is small and touches one extension point. The event bus refactor is different — it touches **every cross-layer interaction** simultaneously. Doing it incrementally means running two transport systems in parallel for months, which is worse than doing it as one focused refactor when there's clear demand.

Wait for the demand. Don't do this speculatively.

## Non-goals (when revived)

- **No remote/networked bus.** Single-process only. Distributed coordination is out of scope forever (or until Mother Maxim needs it, which is post-1.0 by several versions).
- **No message persistence.** Events are fire-and-forget. Durability is the persistence layer's job, not the bus's.
- **No at-least-once delivery guarantees.** Synchronous dispatch, in-process, handlers are called exactly once. If a handler throws, that's the handler's problem.
- **No backpressure or rate limiting.** Too much infrastructure for a single-process bus.
- **No typed event schema registry.** The event dataclasses are the schema. If a future version adds a field, use the snapshot protocol's migration pattern (see substrate_plan P3.5.1) for in-flight events — but this is almost never needed because events are ephemeral.

## Relationship to other plans

- **[../substrate_plan.md](../substrate_plan.md) contracts layer** — defines `EventBus` Protocol and typed events. Protocol definition can happen incrementally (during substrate work); the *implementation* is what this plan defers.
- **[../substrate_plan.md](../substrate_plan.md) P3.5.1 snapshot protocol** — same pattern (Protocol defined during phase work, implementation is its own thing) but at a smaller scale. The snapshot protocol is in-plan because P3.5 needs it; the event bus is deferred because no current phase needs it.
- **[bio_system_plugin_plan.md](bio_system_plugin_plan.md)** — plugin discovery for bio-systems. Has a soft dependency on the unified bus because plugin bio-systems need a predictable way to subscribe to events. If that plan revives before this one, the plugin bio-systems subscribe to `AgentBus` / direct callbacks until this plan lands too. Messy but workable.
- **[../foundations_plan.md](../foundations_plan.md) F0.5** — clarified that multi-agent isolation means memory state, not message transport. That clarification is what makes the current fragmented bus situation tolerable — the bus fragmentation is not a correctness bug.

## If you're reading this cold

You found this plan because you're considering unifying the event bus. Before you start:

1. **Confirm the trigger.** Is there a real cross-layer debugging pain, observability need, or external-contributor friction? If no concrete signal, close this file. Speculative refactor of cross-layer coordination is how you spend six weeks and break everything subtly.
2. **Check the `EventBus` Protocol state.** Does `src/maxim/contracts/event_bus.py` exist? If yes, start at Phase 2. If no, start at Phase 1.
3. **Read [memory_hub.py](../../src/maxim/integration/memory_hub.py) cold.** Understand what the direct callbacks currently do before you migrate them. The `hippocampus → nac` deletion callback exists for a reason; find the reason before rewriting it.
4. **Plan the feature-flag cutover carefully.** Phase 3 is the highest-risk step in the plan. Running the old and new transports in parallel should be possible for at least a week, with the ability to diff behavior between them.
5. **Don't delete `AgentBus` or `LocalMessageBus` in the same commit that migrates their callers.** Phase 5 is strictly after Phase 3 and 4 have been in production long enough to trust.

## Revive trigger, stated concretely

Any of these, not "it would be nice":

- A debugging session takes more than four hours because a cross-layer event wasn't traceable
- An external contributor opens a PR asking "how do I subscribe to memory deletion events" and there's no clean answer
- Observability / tracing / replay tooling is being added and needs a single hook point
- A multi-agent bug is traced to "one of the five transports did the wrong thing" and the root cause requires reading all five
- Mother Maxim revives and needs networked coordination (at which point this plan and the Mother Maxim plan merge, because you're doing both at once)

Until one of these fires, this plan stays here.
