# DefaultNetwork Unification — required deps at construction

**Status:** SHELL ONLY. Audit + design pending.
**Parent index:** [biosystem_unification.md](biosystem_unification.md)
**Wave:** 2 of 4 (parallel-safe with `memory_hub_unification.md`)
**Depends on:** [pain_bus_unification.md](pain_bus_unification.md) — DefaultNetwork consumes the bus.
**Parallel-safe with:** [memory_hub_unification.md](memory_hub_unification.md) — different files.
**Blocks:** [bio_stack_unification.md](bio_stack_unification.md).

## Goal

Make `DefaultNetwork` construction structurally enforce the (maxim, bus, nac) dependency triple. Today it's constructed inconsistently across entry points — some include it, some don't, some pass a stub `_sim_maxim_stub = object()`. Forgetting to construct it skips the reactive fear-gate silently.

## The repeating shape

```python
default_network = None
try:
    from maxim.default_network.network import DefaultNetwork, DefaultNetworkConfig
    default_network = DefaultNetwork(
        maxim=...,
        bus=...,
        config=DefaultNetworkConfig(...),
        nac=...,
    )
    logger.info("DefaultNetwork active")
except Exception as e:
    logger.debug("DefaultNetwork creation failed: %s", e)  # ← silent skip
```

The `except Exception: logger.debug(...)` pattern means construction failures fall through silently. The default-network-driven fear-gate doesn't fire on this path; nothing tells the user. **Silent failure** with the additional smell of broad-except swallowing.

## Audit (2026-04-16)

### Key finding: a `build_default_network` helper already exists at Layer 4

Unlike PainBus (which had no helper pre-Wave-1), DefaultNetwork already has `runtime/bootstrap.py::build_default_network(*, maxim, bus, fear_agent, nac, config_path, frame_size)` (line 610). But it's Layer 4 — all parameters have defaults, construction failures are swallowed (`try/except Exception` returns `None`), and the sim orchestrator bypasses it entirely. The plan is to **upgrade this to Layer 5** (required keyword-only deps, narrower exception handling) rather than writing a new builder from scratch.

### Construction sites in `src/maxim/` (production code)

| # | Site | File:line | How constructed | nac | bus | Exception handling | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Reachy runtime | [agentic_runtime.py:671](src/maxim/embodied_runtime/agentic_runtime.py#L671) | Via `build_default_network(...)` | ✅ `nac` | ✅ `agent_bus` | Caller swallows → `None` + warning | **The only caller of `build_default_network`.** Correctly passes all available deps. Helper returns `None` when `maxim is None` (headless). Hippocampus subscriber wired EXTERNALLY at agentic_runtime.py:719 (Gap B split ownership). |
| 2 | Sim orchestrator | [orchestrator.py:909](src/maxim/simulation/orchestrator.py#L909) | **Direct `DefaultNetwork(...)` — BYPASSES helper** | ✅ `aut_nac` | ❌ `None` | Swallows → `None` + debug | **Gap C.** Uses `_sim_maxim_stub = object()` (undocumented). No behaviors, no YAML config. Manual `DefaultNetworkConfig(publish_actions=False, fear_gate_enabled=False)`. |
| 3 | bootstrap.py helper | [bootstrap.py:610](src/maxim/runtime/bootstrap.py#L610) | (definition) | Optional, default `None` | Optional, default `None` | Swallows → `None` + warning | **Layer 4:** all optional with defaults, returns `None` on failure. Needs Layer 5 upgrade. |

### Sites that do NOT construct DefaultNetwork (silent omissions)

| # | Site | Notes |
|---|---|---|
| 4 | CLI non-sim (`maxim --llm X`) | No DN at all. Reactive fear-gate, PainCircuitBridge, focus learner, novelty tracking all absent. Two comment refs at cli.py:1231, 1419. **Legitimate headless opt-out** — needs explicit documentation, not a fix. |
| 5 | api.py headless (`maxim.create.agent`) | No DN at all. Same as #4. Belongs to `agent_factory_canonicalization.md` Stage F5. |

### Pre-existing gaps surfaced

**Gap A (Layer 4 → Layer 5 upgrade) — `build_default_network` is too permissive.**

The helper at [bootstrap.py:610](src/maxim/runtime/bootstrap.py#L610) accepts all optional params and returns `None` on any failure. The caller wraps it in another `try/except`. Two layers of exception swallowing. If DN construction fails for a non-trivial reason, the agent runs without reactive behaviors and the user sees only a debug/warning-level log line. **Silent failure.** L1 applies: pain detection, fear gating, novelty tracking all disappear with no user-visible indication.

**In-scope fix:** upgrade `build_default_network` to require `nac` as keyword-only with no default (it drives pain detection — the core learning subsystem). `maxim` stays optional with `None` as the headless opt-out (headless agents don't have motor control). `bus` stays optional. Replace the broad `except Exception → None` with narrow handling: import failures → warning (optional dep), config/type errors → user-visible with fix hint.

**Gap B (inherited from Wave 1) — DN constructs its own PainBus internally, split subscriber ownership.**

[network.py:360](src/maxim/default_network/network.py#L360) constructs `PainBus()` inside `_init_pain_circuit`. NAc wired via `PainCircuitBridge` internally. Hippocampus subscription happens EXTERNALLY at [agentic_runtime.py:719](src/maxim/embodied_runtime/agentic_runtime.py#L719). Two-file split. The Wave 1 pain_bus audit committed to inverting this in Wave 2: DN becomes a bus **consumer** (accepts injected PainBus) rather than bus **constructor**.

**In-scope fix:** add `pain_bus: PainBus | None = None` parameter to `build_default_network`. When provided, DN uses the injected bus instead of constructing its own. The external hippocampus subscription at agentic_runtime.py:719 becomes unnecessary when the injected bus already has both subscribers from `build_pain_bus`. When `None`, DN constructs its own (backward compat for tests). This closes the split-ownership gap.

**Gap C — sim orchestrator bypasses `build_default_network`.**

[orchestrator.py:909](src/maxim/simulation/orchestrator.py#L909) constructs `DefaultNetwork(...)` directly. Different config path (manual vs YAML), no behavior wiring, undocumented `object()` stub. **In-scope fix:** migrate to call `build_default_network(maxim=None, nac=aut_nac, ...)` — the helper needs to handle `maxim=None` gracefully for sim mode (today it returns `None` early, which is wrong for sim — sim wants DN for pain detection but not for motor control).

**Gaps D + E — cli.py + api.py headless opt-outs.** Legitimate — no DN for non-robot paths. Add explicit `# DefaultNetwork: explicit headless opt-out` comments. The user-facing decision of "should headless agents get DN?" belongs to `agent_factory_canonicalization.md` Stage F5.

## Design sketch

```python
def build_default_network(
    *,
    nac: NAc,
    pain_bus: PainBus | None = None,
    bus: AgentBus | None = None,
    maxim: Any | None = None,
    config: DefaultNetworkConfig | None = None,
) -> DefaultNetwork:
    """Construct a DefaultNetwork with explicit bio-system deps.

    `nac` is required (the network drives NAc-prediction-based fear);
    `bus` and `maxim` are optional with sensible defaults for headless
    use. `config` defaults to a reasonable DefaultNetworkConfig.
    """
```

**Open design questions:**
- Should the `_sim_maxim_stub = object()` pattern be replaced with an explicit `HeadlessMaximStub` class with documented null-method behavior?
- Should the broad-except wrapping disappear (DefaultNetwork construction failures become user-visible) or become typed (`DefaultNetworkConstructionError`)?
- Is there a `DefaultNetwork` opt-out for sandbox/test use, or should every test pass a real one?

## Pre-merge review round

**Executor lens:**
- Does the new builder correctly handle the sim-mode stub-maxim pattern without re-introducing the `object()` smell?
- Are all DefaultNetwork construction sites migrated?
- Does any test depend on the silent-skip behavior (assumes construction can fail) and need updating?

**Architecture lens:**
- Should DefaultNetwork construction failure be a runtime warning, a runtime error, or a TypeError at signature level?
- Is `Any | None` the right type for `maxim`, or should we declare a `MaximStub` Protocol?
- Cross-check: does Plan 5 (bio_stack) need a specific construction order between PainBus, NAc, and DefaultNetwork?

## Estimated scope

~100 LOC + ~150 LOC tests. Single PR. ~1-2 days.

## Out of scope

- The DefaultNetworkConfig schema itself.
- The fear-gate routing inside DefaultNetwork (separate concern).
- Sleep-replay timing (lives in the sleep-replay plan).
