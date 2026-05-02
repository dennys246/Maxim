# Maxim Extension API

Stable extension points for third-party packages and end-user code. Anything on this page is part of the 1.0 contract — surfaces marked **stable** will not break in 1.x. Surfaces marked **experimental** may change without a major-version bump; the experimental tag is also documented in the relevant docstring.

If you're extending Maxim against a surface **not** on this page, expect breakage on minor-version upgrades.

This page documents **extension surfaces** — what third parties plug INTO Maxim. For the **stability contract on the verbs Maxim exposes outward** (what callers consume), see [stable_api.md](stable_api.md). The two pages are siblings: this one for plugin/extension authors, that one for callers using `maxim.run()` / `maxim.imagine()` / `maxim.campaign()` etc.

## Index

| # | Extension point | Stability | Where to plug in |
|---|---|---|---|
| 1 | Robot drivers | stable | [`maxim.robots` entry-point group](#1-robot-drivers) |
| 2 | Custom tools | stable | [`Tool` ABC + `register_tool` / `@tool`](#2-custom-tools) |
| 3 | Custom LLM backends | stable | [`runtime/lane_backends.py::BACKEND_CLASSES`](#3-custom-llm-backends) |
| 4 | Custom percept sources | stable | [`PerceptSource` protocol](#4-custom-percept-sources) |
| 5 | Custom action sinks | stable | [`ActionSink` protocol](#5-custom-action-sinks) |
| 6 | Bio-system bridges | experimental | [`PainBus.subscribe` / `ReactionBus.subscribe`](#6-bio-system-bridges) |
| 7 | Event subscriptions | experimental | [`maxim.on(event_name, callback)`](#7-event-subscriptions) |
| 8 | Custom personas | ⛔ deprecated in 0.9 — removed in 1.1 | [`maxim.register_persona(...)`](#8-custom-personas) |

---

## 1. Robot drivers

**Stability:** stable. Plugin discovery and the `RobotController` ABC are part of the 1.0 contract.

The `maxim.robots` Python entry-point group is the canonical way to add support for a new robot. Any package that registers an entry point in this group is auto-discovered by `RobotRegistry` at startup — no core code changes needed.

### Minimal example

```toml
# In your plugin package's pyproject.toml
[project.entry-points."maxim.robots"]
atlas = "maxim_atlas.controller:AtlasController"
```

```python
# maxim_atlas/controller.py
from maxim.hardware import (
    RobotController,
    MotionTarget,
    PixelTarget,
    RobotCapabilities,
    MotionCapability,
    StreamCapability,
)


class AtlasController(RobotController):
    @property
    def robot_type(self) -> str:
        return "atlas"

    # Lifecycle
    def connect(self, timeout: float = 30.0) -> bool: ...
    def disconnect(self) -> None: ...

    # Motion
    def goto_target(self, target: MotionTarget) -> bool: ...
    def look_at_pixel(self, target: PixelTarget) -> bool: ...
    def get_current_pose(self) -> dict[str, float]: ...

    # Sleep / wake
    def wake_up(self) -> bool: ...
    def goto_sleep(self) -> bool: ...

    # Streams + recording (required by the ABC even when capabilities
    # don't advertise them — return None / no-op when unsupported)
    def get_audio_stream(self): ...
    def get_video_stream(self): ...
    def start_recording(self, path: str) -> bool: ...
    def stop_recording(self) -> bool: ...
```

`RobotController` declares 12 abstract methods total (lifecycle, motion, sleep/wake, streams, recording). All must be implemented for the subclass to instantiate; advertise the actual supported subset via the `RobotCapabilities` flag set you return from `connect()`.

### Reference

- ABC: [`src/maxim/hardware/controller.py`](../../src/maxim/hardware/controller.py)
- Registry: [`src/maxim/hardware/registry.py`](../../src/maxim/hardware/registry.py)
- Capabilities: [`src/maxim/hardware/capabilities.py`](../../src/maxim/hardware/capabilities.py)
- Full guide: [robot-setup.md](robot-setup.md)

---

## 2. Custom tools

**Stability:** stable. The `Tool` ABC, `register_tool`, and the `@tool` decorator are part of the 1.0 contract.

Tools are the action surface — anything the agent can do is exposed as a `Tool` subclass. `Tool.input_schema` accepts both the legacy custom format and JSONSchema (CC9 dual-format support); JSONSchema is the canonical format going forward.

### Minimal example — class-based

```python
from maxim.tools.base import Tool, ToolOutput
import maxim


class WeatherTool(Tool):
    name = "get_weather"
    description = "Get current weather for a city"
    # JSONSchema (preferred for new tools)
    input_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    }

    def execute(self, **kwargs) -> ToolOutput:
        city = kwargs["city"]
        # ... call your weather API ...
        return ToolOutput(success=True, output=f"Sunny in {city}, 72°F")


maxim.register_tool(WeatherTool())
maxim.run(model="mistral-7b")  # WeatherTool is now available to the agent
```

### Minimal example — decorator

```python
import maxim


@maxim.tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}, 72°F"


maxim.run(model="mistral-7b")
```

The decorator infers `input_schema` from type hints and exports it as JSONSchema.

### Reference

- ABC: [`src/maxim/tools/base.py`](../../src/maxim/tools/base.py)
- Registration verbs: [`src/maxim/api.py::register_tool`](../../src/maxim/api.py), [`src/maxim/api.py::tool`](../../src/maxim/api.py)
- Schema export: `Tool.to_json_schema()` returns JSONSchema 2020-12 / MCP-compatible output
- Cancellation hook: `Tool.cancel()` is a non-abstract no-op on the ABC, reserved for 1.1+ MCP-subprocess and async-cancel work. No 1.0 dispatch path calls it; heavy tools (HTTP fetch, web search) override it to set a `threading.Event` for cooperative cancellation. See CLAUDE.md "Tool.cancel" invariant.
- Side-effects channel: [tool_side_effects.md](tool_side_effects.md)
- Full guide: [tools.md](tools.md)

---

## 3. Custom LLM backends

**Stability:** stable. The dispatch table and the backend-class shape are part of the 1.0 contract.

Backends bridge Maxim's lane router to a wire protocol (OpenAI-compatible REST, Anthropic SDK, self-hosted peer, etc.). Adding a new backend type is a one-line entry in `runtime/lane_backends.py::BACKEND_CLASSES`, plus a class that implements the same `complete_with_usage()` / `_stream_response` / `health_check()` surface as the two existing backends.

### Minimal example

```python
# my_package/my_backend.py
from maxim.models.language.types import (
    BackendError,
    BackendDown,
    BackendTimeout,
    BackendOverloaded,
)


class _MyBackend:
    """One HTTP call per complete_with_usage. No retry loops — the
    router handles failover via the typed BackendError subclasses.
    """

    def __init__(self, *, url: str, api_key: str | None, model: str) -> None:
        self._url = url
        self._api_key = api_key
        self._model = model

    def complete_with_usage(self, messages, **kwargs):
        # ... one HTTP call, raise typed BackendError on failure ...
        ...

    def health_check(self, *, enable_stage2: bool = False):
        ...

    @classmethod
    def for_url(cls, url: str, *, api_key: str | None = None, model: str = ""):
        # Concurrency-safe factory — store overrides on the instance,
        # never mutate os.environ.
        return cls(url=url, api_key=api_key, model=model)
```

```python
# In your package's __init__.py or a setup hook the user runs once.
# IMPORTANT: register BEFORE the first call to maxim.run() / maxim.imagine() /
# maxim.campaign() — backends are resolved during lane bootstrap, and a
# registration that runs after bootstrap won't affect the active session.
from maxim.runtime.lane_backends import BACKEND_CLASSES

BACKEND_CLASSES["my_backend"] = ("my_package.my_backend", "_MyBackend")
```

### Reference

- Dispatch table: [`src/maxim/runtime/lane_backends.py`](../../src/maxim/runtime/lane_backends.py) (`BACKEND_CLASSES`, `resolve_backend_class`)
- Reference implementations: [`src/maxim/models/language/openai_backend.py`](../../src/maxim/models/language/openai_backend.py), [`src/maxim/models/language/maxim_peer_backend.py`](../../src/maxim/models/language/maxim_peer_backend.py)
- Typed exceptions: [`src/maxim/models/language/types.py`](../../src/maxim/models/language/types.py) (`BackendError` and subclasses — extend rather than introduce parallel hierarchies)
- Router: [`src/maxim/models/language/router.py`](../../src/maxim/models/language/router.py)

**Invariants for new backends** (frozen at 1.0):

- `complete_with_usage()` makes **exactly one HTTP call**. No internal retry / backoff / gateway loops. Failover is the router's job.
- Failure raises a typed `BackendError` subclass — never bare `Exception`. The router branches on type in specific-before-general order.
- `for_url()` factories store overrides on the returned instance, never on `os.environ`.

---

## 4. Custom percept sources

**Stability:** stable. The `PerceptSource` protocol is part of the 1.0 contract.

A `PerceptSource` is anything that produces input to the agent — a robot's vision pipeline, a CLI prompt reader, a Mineflayer adapter, a recorded scenario file. The agent loop consumes percepts without knowing their origin.

### Minimal example

```python
from typing import Protocol  # for documentation
from maxim.simulation.sources import PerceptSource
from maxim.agents.bus import Percept
from maxim.agents.percept_factory import make_text_percept


class StdinPerceptSource:
    """A PerceptSource that reads one line at a time from stdin."""

    @property
    def name(self) -> str:
        return "stdin"

    @property
    def capabilities(self) -> set[str]:
        return {"transcript"}

    def next_percept(self) -> Percept | None:
        try:
            line = input()
        except EOFError:
            self._exhausted = True
            return None
        return make_text_percept(text=line, sender="user")

    def is_exhausted(self) -> bool:
        return getattr(self, "_exhausted", False)
```

`PerceptSource` is `@runtime_checkable`, so Python's `isinstance()` works against it — duck typing is fine, no inheritance required.

### Reference

- Protocol: [`src/maxim/simulation/sources.py`](../../src/maxim/simulation/sources.py)
- `Percept` shape: [`src/maxim/agents/bus.py`](../../src/maxim/agents/bus.py)
- Factories: [`src/maxim/agents/percept_factory.py`](../../src/maxim/agents/percept_factory.py) (`make_text_percept`, `make_scene_percept`, `make_intero_percept`)
- Isolation rules: [`src/maxim/agents/percept_context.py`](../../src/maxim/agents/percept_context.py) docstring (no cross-agent intent, no oracle leakage)

---

## 5. Custom action sinks

**Stability:** stable. The `ActionSink` protocol is part of the 1.0 contract.

An `ActionSink` captures every tool execution (including FearAgent blocks). Used by the simulation runner for post-run validation and replay; can also be used by external integrations that want to observe what the agent did.

### Minimal example

```python
from maxim.simulation.sinks import ActionRecord, ActionSink


class JSONLogSink:
    """Append every action to a JSONL file."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._actions: list[ActionRecord] = []

    def record(self, action: ActionRecord) -> None:
        import json
        self._actions.append(action)
        with open(self._path, "a") as f:
            f.write(json.dumps({
                "timestamp": action.timestamp,
                "tool_name": action.tool_name,
                "success": action.result_success,
                "blocked": action.blocked,
            }) + "\n")

    @property
    def actions(self) -> list[ActionRecord]:
        return list(self._actions)


# isinstance check confirms the protocol fit:
assert isinstance(JSONLogSink("/tmp/x.jsonl"), ActionSink)
```

### Reference

- Protocol + `ActionRecord` dataclass: [`src/maxim/simulation/sinks.py`](../../src/maxim/simulation/sinks.py)
- Built-in implementation: `RecordingSink` in the same module (bounded, lock-protected, supports compression of old records)

---

## 6. Bio-system bridges

**Stability:** experimental. The bus surfaces are stable; the well-known subscriber factories (`create_pain_memory_subscriber`, `create_pain_nac_subscriber`) and the `PainSignal.context` key set may grow.

Bio-system bridges subscribe to `PainBus` or `ReactionBus` and translate signals into learning updates, telemetry, or external side-effects (alerts, dashboards). Two buses coexist by design:

- **`PainBus`** — rich free-form context (`PainSignal.context: dict[str, Any]`) carrying cause-description metadata for NAc causal learning. Bio-internal publishers and subscribers.
- **`ReactionBus`** — strict typed isolation surface (`Reaction`, `ReactionContext`, `TraceSnapshot`). Generic typed pub/sub keyed by `ReactionKind = Literal["pain", "fear", "hunger", "surprise", "fatigue", "satiation"]`.

Anything subscribing to `PainBus` directly receives the full `signal.context`. Anything subscribing through `ReactionBus.subscribe(kind, callback)` sees only the typed view.

### Minimal example — telemetry subscriber

```python
from maxim.proprioception.pain_bus import PainBus
from maxim.proprioception.pain import PainSignal


def my_telemetry(signal: PainSignal) -> None:
    """Forward every PainSignal to your dashboard."""
    payload = {
        "type": signal.pain_type,
        "intensity": signal.intensity,
        "source": signal.context.get("source"),
        "entity": signal.context.get("entity"),
    }
    # ... POST payload to your collector ...


# Subscribe to a PainBus you've constructed (e.g., obtained from
# build_bio_stack(...).pain_bus):
pain_bus.subscribe(my_telemetry)
```

### Minimal example — typed reaction subscriber

```python
from maxim.reactions.bus import ReactionBus
from maxim.reactions.types import Reaction


def on_fear(reaction: Reaction) -> None:
    print(f"fear from {reaction.context.source}: intensity={reaction.intensity}")


reaction_bus.subscribe("fear", on_fear)
```

### Reference

- `PainBus`: [`src/maxim/proprioception/pain_bus.py`](../../src/maxim/proprioception/pain_bus.py)
- Canonical PainBus constructor: `build_pain_bus(*, hippocampus, nac, ...)` in the same module
- `ReactionBus`: [`src/maxim/reactions/bus.py`](../../src/maxim/reactions/bus.py) (`build_reaction_bus`)
- `Reaction` types and isolation rules: [`src/maxim/reactions/types.py`](../../src/maxim/reactions/types.py) docstring
- Built-in subscriber factories: `create_pain_memory_subscriber`, `create_pain_nac_subscriber` in `pain_bus.py`

**Why experimental:** the rich-context key set on `PainSignal.context` is bio-pipeline-internal and may grow as new failure modes are wired. Subscribers should defensively use `signal.context.get(key)` rather than indexing — keys present today are not guaranteed to be present forever, and new keys may appear.

---

## 7. Event subscriptions

**Stability:** experimental. The event names are stable; the payload dataclass field sets may grow.

`maxim.on(event_name, callback)` is a high-level subscription API for users who want to observe agent activity without touching bus internals. Returns an `EventHandle` with `unsubscribe()`.

### Supported events

| Event name | Payload type | Stability |
|---|---|---|
| `"tool_call"` | `ToolCallEvent` | experimental |
| `"memory_capture"` | `MemoryCaptureEvent` | experimental |
| `"pain_signal"` | `PainSignalEvent` | experimental |
| `"prompt"` | `PromptEvent` | experimental |

### Minimal example

```python
import maxim


def log_tool(event):
    print(f"{event.tool_name}({event.params}) -> success={event.success}")


handle = maxim.on("tool_call", log_tool)
maxim.run(model="mistral-7b")
handle.unsubscribe()
```

### Reference

- API: [`src/maxim/api.py::on`](../../src/maxim/api.py)
- Payload dataclasses: `ToolCallEvent`, `MemoryCaptureEvent`, `PainSignalEvent`, `PromptEvent` in the same file

---

## 8. Custom personas

**Stability:** ⛔ **deprecated in 0.9 — removed in 1.1.** `maxim.register_persona()` emits `DeprecationWarning` in 0.9 / 1.0 and will raise in 1.1. The persona system is being replaced by `--sim-mode` (an orchestrator flow-shape selector) plus the bio-emergent disposition mechanics tracked in `docs/plans/bio_emergent_persona_foundations.md`. The CLI flag `--persona` (and its `--sim-persona` alias) is also deprecated in 0.9; use `--sim-mode` instead. See [`docs/plans/persona_cleanup_and_mode_transition.md`](../plans/persona_cleanup_and_mode_transition.md) for the migration timeline and rationale.

> **Note on flag naming:** the persona-cleanup plan originally proposed the short alias `--mode`, but that token is already owned by the core run-mode flag (`--mode {live,train,reflection,sleep,agentic,exploration}`). Stage 1 ships `--sim-mode` only; freeing `--mode` for sim use is a separate breaking change with its own deprecation cycle.

> **Note (audit finding):** registered personas currently flow through to reports and logs as a label; the orchestrator does not inject the supplied `context_prompt` into the agent prompt today. The rich prompt strings shipped in `simulation/personas.py` exist as scaffolding for behavioural shaping that is being moved to `--sim-mode` and the bio-emergent disposition mechanics. Stage 5 of the cleanup plan removes the unused field.

Personas shape how the simulation orchestrator framing affects an agent — adversarial probing, cooperative coaching, etc. Register a persona once and reference it by name in `--persona <name>` or the `persona=` argument to `imagine()`/`run()`.

### Minimal example

```python
import maxim


maxim.register_persona(
    name="cautious_explorer",
    description="Probes carefully, prefers reversible actions",
    focus="risk-averse exploration of unfamiliar SEM affordances",
    context_prompt=(
        "You are a cautious explorer. Before every action, ask: "
        "if this fails, can I recover? Prefer affordances you have "
        "already seen succeed."
    ),
    max_initiative=0.3,
)


maxim.imagine("explore the cradle", persona="cautious_explorer")
```

### Reference

- API: [`src/maxim/api.py::register_persona`](../../src/maxim/api.py)
- Built-in personas: [`src/maxim/simulation/personas.py`](../../src/maxim/simulation/personas.py)

---

## What is **not** an extension point

These are explicitly **internal** in 1.0 — extending against them is unsupported and will break:

- Direct subclassing of `Hippocampus`, `NAc`, `ATL`, `SCN`, `EC`, `AngularGyrus`, `EpisodicMemory`. Wire your behavior through `register_tool` + bus subscribers instead.
- Direct calls into `runtime/agent_loop.py` or `runtime/loop_controller.py` internals. Use `maxim.run()` / `maxim.imagine()` / `maxim.campaign()` from `api.py`.
- The `_*` prefix on a module or attribute. Underscore = internal.
- `runtime/bootstrap.py::build_executor` and the other `build_*` constructors. Use the public `api.py` verbs unless you're contributing to Maxim itself.

If you have a use case that needs one of these, open an issue — we'd rather promote a surface to "stable" than have third parties pin against internals.

## Stability promise summary

- **Stable** surfaces will not break in 1.x without a deprecation cycle.
- **Experimental** surfaces may break in 1.x. The experimental tag also appears in the docstring header of the relevant API.
- Anything not on this page may break at any time.
