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
| `drive_potential_diff` | `float` — signed value-progress toward comfort (consumer uses its SIGN), see schema below | [`ModulatorAffordanceTool.execute`](../../src/maxim/embodiment/tool_bridge.py) when a `self_effect` touches an entity-level drive sensor (orient→azimuth, eat→hunger) | [`runtime/tool_dispatch.py`](../../src/maxim/runtime/tool_dispatch.py) uses its sign as the ±1 cluster reward for substrate-primary action selection in place of the flat tool-success signal | 1.0.1 |
| `drive_credit_withheld` | `bool` — the affordance's declared effect targets a drive sensor a LIVE measurement stream owns; modeled credit was filtered and no measured credit exists (turn in a silent room / measurement timed out) | [`ModulatorAffordanceTool.execute`](../../src/maxim/embodiment/tool_bridge.py) when `self_effect ∩ live_world_set_sensors ∩ drive_specs ≠ ∅` and no `drive_potential_diff` was emitted (modeled or measured) | [`runtime/tool_dispatch.py`](../../src/maxim/runtime/tool_dispatch.py) suppresses the flat +1 tool-success cluster floor for this action (a real motor-bound turn in a silent room must not mint direction-blind cluster credit) | 1.0.5 |
| `drive_relief_channel` | `str` — `"exteroceptive"`: the accompanying `drive_potential_diff` is a MEASURED exteroceptive transition (post-motion sensor re-read), not a modeled delta (sem_motor_binding.md Phase 2) | [`ModulatorAffordanceTool.execute`](../../src/maxim/embodiment/tool_bridge.py) when a motor backend reports `metadata["measured_drive_transitions"]` for a live-owned drive sensor | [`runtime/tool_dispatch.py`](../../src/maxim/runtime/tool_dispatch.py) routes the ±1 cluster credit to the direction-bearing (audio/operant) cluster instead of interoception — measured exteroceptive relief is source-attributable, so live experience compounds the trained policy keys | 1.0.6 |

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

**Delta-attribution filter (Exp 42, 1.0.1):** for **drive-spec** failures (names of the form `drive:<sensor>:discomfort` / `drive:<sensor>:deprived`), `ModulatorAffordanceTool.execute` only includes the failure in `embodiment_failures` when the affordance's *own* `self_effect`/`target_effect` delta is intrinsically harmful to that sensor (i.e. that delta alone would breach a healthy sensor). This stops a *bystander* affordance from inheriting blame for a drive breach a different (harmful) affordance caused while the breach lingers. **Standard `failure_mode` failures pass through unchanged.** Consumers that previously saw a drive-spec breach attributed to every action executing during it will now see it only on the affordance that caused it. Note this filter applies to the `side_effects` channel only — the parallel `PainBus`/`_publish_drive_pain` channel is not delta-attributed (transition-based drive-pain is a tracked follow-up).

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

### `drive_potential_diff`

```python
float  # signed VALUE-progress toward comfort; positive = moved toward comfort. Consumer uses its SIGN. Key ABSENT (None) = no signal
```

The net **value-progress** the affordance's *own* `self_effect` moved the body's drives toward comfort, summed over the entity-level drive sensors it touched:
`Σ drive_comfort_progress(spec, before, after)` — **value-based, not pain-based** (homeostatic = reduction in `|value − set_point|`; entropic = value moved toward comfort per drift direction).

**Why value-based:** `drive_pain_for_value` is a *step function* for entropic drives (`deprivation_pain` past threshold, `0` below), so a pain-based signal books **zero** relief for a `warm`/`eat` that reduces cold/hunger without crossing the threshold — which starved entropic-relief actions of credit and floored substrate-primary engagement (the #405 Exp-42 regression). Value-progress is graded and nonzero for any real movement.

**The consumer ([`runtime/tool_dispatch.py`](../../src/maxim/runtime/tool_dispatch.py)) takes the SIGN, not the magnitude** (`+1` progress / `−1` regress), so drive-relief actions sit on the same `±1` scale as the tool-success signal non-drive actions get (a small graded magnitude would lose the argmax to a flat `+1`).

- **Positive** — moved drives toward comfort (`turn_left` reduced `|azimuth|`; `eat` reduced `hunger`) → cluster reward `+1`.
- **Negative** — moved drives away (turned away from the sound) → cluster reward `−1`.
- **`0.0` (present)** — a drive sensor was touched but net progress was exactly zero (e.g. turning into a `±1.0` azimuth wall) → the consumer falls back to the tool-success signal.
- **Key ABSENT** (the producer emits `None`) — either the affordance touched no entity-level drive sensor, **or** it caused *collateral harm* (see below). The consumer falls back to the `±1` tool-execution-success signal.

**Collateral-harm gate:** the signal is suppressed (key absent) when the action caused a failure on a sensor its relief did **not** account for — a non-drive `failure_mode` (e.g. `arms.thermal` thermal shock) or an untouched drive sensor. This stops an *attractive-but-harmful* action (relieves `cold` but breaches `arms.thermal`) from being credited positively, which would defeat the safe-vs-harm discrimination. A *same-sensor* drive discomfort (azimuth still off-center after a relieving turn) is **not** collateral — the relief already reflects that sensor's net change — so it does not suppress the signal.

**`target_effect` is intentionally NOT scored.** Motor-credit attributes only to the *acting* body's own `self_effect`; relief a caregiver's `target_effect` produces on another body (e.g. a mother feeding an infant) is that other body's state, not the actor's learned policy. Only entity-level drive sensors (`body.drive_specs`: azimuth-centeredness, hunger, thirst, energy) are scored; qualified modulator sub-sensors (`arms.thermal`) carry no drive spec today and are skipped. Single source of truth for the pain term is [`drive_pain_for_value`](../../src/maxim/embodiment/sem.py) (the homeostatic + entropic pain formula, shared with `Embodiment.evaluate_failures`' homeostatic branch).

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
