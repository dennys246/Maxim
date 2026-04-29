# Tool side_effects Registry

`ToolOutput.side_effects` is a typed channel for bio-pipeline signals that flow from a tool's `execute()` to downstream consumers (the executor, bio-system bridges, learning subscribers). It is separate from `metadata` (caller-facing extras) and `output` (the main result the LLM sees) — collapsing them silently couples the tools layer to bio concepts.

This page is the canonical, append-only registry of well-known `side_effects` keys. Third-party tool authors reading this page can produce these keys to interoperate with Maxim's bio pipeline; third parties consuming `ToolOutput.side_effects` can rely on the listed keys remaining stable in 1.x.

## Contract

- **Append-only.** Once a key is documented here at a given `Since` version, its name and shape do not change. New keys may be added.
- **Optional.** A consumer of `ToolOutput.side_effects` MUST handle a `None` value (the field is `dict[str, Any] | None`). Missing keys MUST be tolerated (use `.get(key)`).
- **Typed at the value level, untyped at the dict level.** The dict itself is `dict[str, Any]`. The value shape per key is documented below and is part of the contract.
- **Non-breaking with success/failure.** A tool returning `success=True` may emit side_effects (e.g., a successful action with embodiment failures). A tool returning `success=False` may also emit side_effects (e.g., a blocked affordance with `affordance_blocked` metadata).

## Registry

| Key | Value type | Producer | Consumer | Since |
|---|---|---|---|---|
| `embodiment_failures` | `list[dict]` — see schema below | [`ModulatorAffordanceTool.execute`](../../src/maxim/embodiment/tool_bridge.py) when `embodiment.evaluate_failures()` fires post-action | [`runtime/executor.py`](../../src/maxim/runtime/executor.py) routes to [`ToolPainBridge.record_tool_embodiment_failure`](../../src/maxim/bridges/tool_pain_bridge.py) for direct NAc attribution | 0.6 |
| `entity_acquired` | `str` — entity name (matches `EntityMap.resolve(name)`) | [`ModulatorAffordanceTool.execute`](../../src/maxim/embodiment/tool_bridge.py) on a successful `pick_up` affordance against an entity whose `metadata["acquirable"]` is True | [`runtime/executor.py::_handle_entity_acquisition`](../../src/maxim/runtime/executor.py) reparents the entity to the agent body and registers its tools | 0.7 |
| `entity_released` | `str` — entity name | [`ModulatorAffordanceTool.execute`](../../src/maxim/embodiment/tool_bridge.py) on a successful `drop` affordance | [`runtime/executor.py::_handle_entity_acquisition`](../../src/maxim/runtime/executor.py) deregisters the entity's tools and returns it to the scene | 0.7 |
| `affordance_blocked` | `dict` — see schema below | [`ModulatorAffordanceTool.execute`](../../src/maxim/embodiment/tool_bridge.py) when a `requires` precondition fails (e.g., damaged body part) | informational — currently no automated consumer; logged for telemetry and replay | 0.6 |

## Value schemas

### `embodiment_failures`

```python
list[dict]  # one dict per active SEM failure
```

Each dict has:

```python
{
    "name": str,        # failure mode name, e.g., "overheat", "snap"
    "entity": str,      # entity path that failed, e.g., "rusty_sword"
    "pain": float,      # signal intensity in [0.0, 1.0]
}
```

A tool that succeeded at its action but produced embodiment failures still returns `success=True` — the tool did what was asked; the side-effect reports what the body felt.

### `entity_acquired`

```python
str  # entity name resolvable via EntityMap.resolve(name)
```

The entity must be present in the agent's `EntityMap` at the time the tool returns. The executor calls `entity_map.resolve(name)` and reparents the resulting entity onto the agent body.

### `entity_released`

```python
str  # entity name; matches a previously-acquired entity
```

Releasing an entity that was never acquired is a no-op at the executor layer (logged at debug level).

### `affordance_blocked`

```python
{
    "affordance": str,   # blocked affordance name, e.g., "fly"
    "modulator": str,    # modulator that owns the affordance, e.g., "wings"
    "entity": str,       # entity name
    "reason": str,       # human-readable block reason, e.g., "wing integrity 0.0 < 0.3"
}
```

Currently informational only — no automated learning hook reads this key. Useful for telemetry, replay, and prompt construction (the LLM also sees the reason in `ToolOutput.error`).

## Adding a new key

When you want to add a new well-known `side_effects` key:

1. **Decide whether it belongs in `side_effects` at all.** If it's caller-facing extras (timestamps, latency, structured replay metadata), use `metadata`. If it's the main result the LLM should see, that goes in `output`. `side_effects` is reserved for **bio-pipeline signals** the executor / bridge / NAc layer routes on, OR purely informational telemetry signals consumed by external observers (replay tooling, dashboards).
2. **Pick a stable name.** Once shipped, the name is frozen by the append-only contract.
3. **Define the value shape** — keep it JSON-serializable (str, int, float, bool, None, list, dict only). Numpy arrays and datetimes break persistence.
4. **Wire the consumer first** — *for keys with bio-pipeline semantics* (a Maxim-internal consumer routes on this key). A producer with no internal consumer of those keys is dead weight; document the producer/consumer pair atomically. **Informational keys** are exempt — they exist so external telemetry / replay / prompt-composition subscribers can read them, and may ship without any internal consumer wired. `affordance_blocked` is the canonical example: no learning hook reads it today, but external observers rely on the key being present and stable.
5. **Add a row to the registry table above** in the same PR. Set `Since` to the version you're shipping in. For informational keys, set the Consumer column to `informational` and document the intended external consumer in the value-schema section.

The append-only invariant is the load-bearing rule: any tool author can read this page and rely on the keys remaining present and shaped as documented. Removing or reshaping a key requires a major-version bump.

## Why this is centralized here

Pre-1.0, this registry lived in `ToolOutput`'s class docstring. That made it invisible to third-party authors who weren't already reading source. Centralizing the registry as a user-facing page lets external tool packages produce or consume these keys without subclassing into Maxim internals.

The `ToolOutput` class docstring now references this page; the page is authoritative.

## Reference

- `ToolOutput` definition: [`src/maxim/tools/base.py`](../../src/maxim/tools/base.py)
- Executor consumption sites: [`src/maxim/runtime/executor.py`](../../src/maxim/runtime/executor.py) (`_handle_entity_acquisition`, post-execute embodiment-failure routing)
- Direct-attribution bridge: [`src/maxim/bridges/tool_pain_bridge.py`](../../src/maxim/bridges/tool_pain_bridge.py) (`record_tool_embodiment_failure`)
- Producer surface: [`src/maxim/embodiment/tool_bridge.py`](../../src/maxim/embodiment/tool_bridge.py) (`ModulatorAffordanceTool.execute`)
