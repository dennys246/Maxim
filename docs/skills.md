# Skills & Protocols

Composable, reusable capabilities (**Skills**) orchestrated into named operational profiles (**Protocols**).

## Conceptual Model

```
ProtocolRegistry
  register(protocol)
  activate(name) → starts skills, applies constraints
  deactivate(name) → stops skills, restores constraints
       │
       ▼
  Protocol (ABC)          Skill (ABC)
    name               ──▶  name
    skills()                 tools() → [Tool]
    workspace_bounds()       activate(maxim, context)
    phrases()                deactivate()
    context_for_llm()        can_activate(maxim) → (bool, str)
    on_skill_failed()        bio_dependencies() → [str]
                             state → SkillState
```

Skills are atomic, independently testable. Protocols compose skills with workspace bounds, LLM context, and voice/CLI activation phrases.

## Skill Lifecycle

```
IDLE → ACTIVATING → ACTIVE → DEACTIVATING → IDLE
                  ↘ FAILED
```

- **IDLE** — not running, ready to activate
- **ACTIVATING** — `activate()` in progress
- **ACTIVE** — running normally
- **FAILED** — runtime error detected
- **DEACTIVATING** — `deactivate()` in progress

## Protocol Activation Sequence

1. **Resolve bio dependencies** — map system names ("hippocampus", "nac") to live instances
2. **Check preconditions** — `can_activate()` on every skill (fail-fast before starting anything)
3. **Activate skills in order** — each receives a shared context dict (blackboard)
4. **On failure** — `on_skill_failed()` returns "abort" (rollback) or "continue" (skip)
5. **Register tools** — skill tools added to the tool registry
6. **Apply workspace bounds** — tighten movement limits (min composition across protocols)
7. **Inject LLM context** — protocol context added to every LLM prompt

## Activation Methods

Protocols can be activated via:
- **Voice**: "Maxim run shredder segmenter protocol"
- **CLI**: typing the same phrase
- **Tool**: LLM calls `run_protocol(name="shredder_segmenter")`
- **Code**: `protocol_registry.activate("shredder_segmenter")`

## Design Decisions

### Why Skill != Tool?

Tools are the LLM-facing interface: name, schema, execute. Skills are the capability layer: lifecycle, state, health, LLM context, and they *contain* tools. A single skill may expose 0-N tools. This separation means:

- Skills can be passive (e.g., workspace constraints) with no tools
- Skills can have internal state (bridge process, threads) that tools don't
- Multiple protocols can share a skill class with different configs
- Skills are testable without the LLM loop

### Why SkillState enum instead of a boolean is_active?

ROS2 actions use explicit goal states (ACCEPTED -> EXECUTING -> SUCCEEDED/ABORTED/CANCELED) and BehaviorTree.CPP nodes return SUCCESS/FAILURE/RUNNING. A boolean loses critical information: you can't tell if a skill was never started, failed during activation, is actively shutting down, or succeeded and was deactivated. The enum gives the LLM better context ("RTSP skill is in FAILED state: ffmpeg process exited" vs. "RTSP skill is not active").

### Why can_activate() precondition checks?

Inspired by SayCan's affordance scoring and ProgPrompt's runtime assertions. Without preconditions, `on_activate()` partially starts skills, discovers the third skill is missing ffmpeg, then has to roll back the first two. With `can_activate()`, all preconditions are checked *before* any skill starts — fail-fast, no partial state to clean up. The `(bool, str)` return gives the LLM a human-readable reason it can relay to the user.

### Why SkillResult structured feedback?

Inner Monologue (Google, 2022) showed that structured feedback after each action is critical for LLM-based closed-loop control. Without it, the LLM only knows "it worked" or "it didn't." With `SkillResult`, the LLM gets "streaming at 720p 20fps to rtsp://..." or "failed: ffmpeg not found in PATH." The `metadata` dict allows skills to attach arbitrary structured data without changing the interface.

### Why a shared context dict (blackboard)?

BehaviorTree.CPP's blackboard lets nodes share typed data without coupling. RTSPStreamingSkill writes `context["rtsp_url"]` after starting, and a future HealthReportingSkill reads it to know what to monitor. The API cost is near zero (just an optional dict parameter on `activate()`), and it avoids skills needing to reach into each other's internals.

### Why on_skill_failed() instead of behavior tree Fallback nodes?

Full behavior trees (Sequence/Fallback/Parallel/Decorator) are powerful but overkill when protocols have 1-3 skills. The `on_skill_failed()` hook gives protocols the key benefit — deciding whether to abort or continue — without the complexity of a tree runtime. If composition needs grow, the hook can be extended without breaking existing protocols.

### Why workspace bounds can only tighten?

`_apply_workspace_bounds` uses `min(base, override)` — a protocol can never widen the robot's safety limits, only constrain further. This is a safety invariant: the hardcoded/learned limits represent physical boundaries.

### Why phrases are registered permanently?

Rather than requiring static `phrase_responses.json` entries, the registry registers phrases once at startup. New protocols work immediately without config changes, and deactivating a protocol does not remove the activation phrases needed to re-activate it via voice.

### Why not use Strategy/Mode for protocols?

Strategies (observe, explore, assist...) and Modes (passive, active, singularity) are Maxim's *general behavioral posture*. Protocols are *specific operational missions* layered on top. You can run ShredderSegmenter in active+assist or active+observe mode. They're orthogonal.

### Why attribute override for workspace bounds?

1. **Survives restart** — `_workspace_limit_override` is a simple attribute, not a closure capturing stale references
2. **Debuggable** — `print(maxim._workspace_limit_override)` shows the override; a monkey-patched method is opaque
3. **Multiple protocols** — attribute can be composed (intersect overrides); monkey-patching replaces the whole method

### Why copy-on-write for phrase registration?

The transcript listener thread iterates `phrase_responses` in three separate `for` loops without holding any lock. In-place dict mutation during iteration crashes with `RuntimeError`. Copy-on-write (new dict -> atomic reference swap under GIL) lets the old iteration finish on the old dict while the new one picks up changes on the next call.

### Why protocol_context on LLMRequest instead of ContextPool?

`ContextPool.add_raw()` entries participate in summarization and eviction (max 50 entries, 2000 tokens). Protocol context would get summarized away after a few observations. By putting it on `LLMRequest.protocol_context`, it's re-injected fresh each LLM submission and rendered at `IMPORTANT` priority — always present, never evicted.

## Built-in Skills

### RTSPStreamingSkill

Streams Reachy camera frames as RTSP via ffmpeg + MediaMTX.

| Property | Value |
|----------|-------|
| Preconditions | ffmpeg in PATH, robot connected |
| Tools | `start_rtsp_stream`, `stop_rtsp_stream` |
| Config | `RTSPStreamingConfig` (url, fps, preset, tune, gop, bitrate) |
| Runtime detection | Detects ffmpeg crash and transitions to FAILED |

Standalone usage (no agentic loop):
```bash
python scripts/rtsp_bridge.py --url rtsp://localhost:8554/reachy --fps 20
```

### TimedProtocolSkill

Auto-deactivate the enclosing protocol after a set duration. Passive skill (no tools) — runs a background timer that calls `registry.deactivate()` when it expires.

| Property | Value |
|----------|-------|
| Preconditions | duration_minutes > 0 |
| Tools | None (passive) |
| Config | `TimedProtocolConfig` (duration_minutes, default 60) |
| Context | Reads `_protocol_name` from shared blackboard |

### HealthReportingSkill

Push periodic health reports to an external HTTP endpoint. Reads shared context (blackboard) to discover active stream URLs and collects health from sibling skills.

| Property | Value |
|----------|-------|
| Preconditions | endpoint_url configured |
| Tools | None (passive) |
| Config | `HealthReportingConfig` (endpoint_url, interval_seconds, timeout_seconds, headers) |
| Failure detection | Transitions to FAILED after 10 consecutive HTTP failures |

Health payload includes: status, timestamp, rtsp_url, protocol name, and sibling skill health.

## Built-in Protocols

### ShredderSegmenterProtocol

Streams Reachy camera as RTSP for ShredderSegmenter ski recording with constrained gaze.

| Property | Value |
|----------|-------|
| Skills | RTSPStreamingSkill + optional HealthReportingSkill + optional TimedProtocolSkill |
| Bounds | yaw=30deg, pitch=20deg (default) |
| API integration | Optional camera registration with ShredderSegmenter server |
| On failure | Aborts if RTSP fails; continues if health/timer fail |

Configuration via environment variables:
```
SHREDDER_API_URL              — ShredderSegmenter server URL
SHREDDER_LICENSE_ID           — License for camera registration
SHREDDER_API_KEY              — Bearer token for API auth
SHREDDER_SITE_ID              — Required for remote recording jobs
SHREDDER_DURATION_MINUTES     — Auto-stop after N minutes (0 = disabled)
SHREDDER_HEALTH_URL           — Health report endpoint (empty = disabled)
SHREDDER_HEALTH_INTERVAL      — Health report interval in seconds (default 30)
```

## Writing a Custom Skill

```python
from maxim.skills.base import Skill, SkillResult, SkillState

class MySkill(Skill):
    @property
    def name(self) -> str:
        return "my_skill"

    @property
    def description(self) -> str:
        return "Does something useful"

    def tools(self) -> list:
        return []  # or return Tool instances

    def can_activate(self, maxim) -> tuple[bool, str]:
        # Check prerequisites
        return True, ""

    def activate(self, maxim, context=None) -> SkillResult:
        self._state = SkillState.ACTIVE
        return SkillResult(state=SkillState.ACTIVE, message="Running")

    def deactivate(self) -> SkillResult:
        self._state = SkillState.IDLE
        return SkillResult(state=SkillState.IDLE)
```

## Writing a Custom Protocol

```python
from maxim.skills.protocol import Protocol, WorkspaceBounds

class MyProtocol(Protocol):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "my_protocol"

    @property
    def description(self) -> str:
        return "Combines skills for a specific task"

    def skills(self) -> list:
        return [MySkill()]

    def workspace_bounds(self) -> WorkspaceBounds:
        return WorkspaceBounds(yaw=45.0)  # constrain gaze

    def on_skill_failed(self, skill, result) -> str:
        return "abort"  # or "continue" to skip failed skills
```

Register in `agentic_runtime.py`:
```python
self._protocol_registry.register(MyProtocol())
```

## Bio-Skill Integration

Skills can declare biological system dependencies:

```python
def bio_dependencies(self) -> list[str]:
    return ["hippocampus", "nac"]  # resolved to live instances
```

Available systems: `hippocampus`, `atl`, `ips`, `angular_gyrus`, `nac`, `scn`.

Deeper integration (markdown skill definitions, skills-as-memories, concept-driven sequences) is planned in ATL concept memory Phase A7.

## Future Skills (Brainstorming)

| Skill | Description | Protocols it could join |
|-------|-------------|------------------------|
| `FixedGazeSkill` | Lock head to a specific yaw/pitch | SecurityPatrol, Timelapse |
| `ScanningSkill` | Sweep head across an arc at fixed speed | SecurityPatrol, Panorama |
| `ObjectTrackingSkill` | Follow a specific class/person | FollowMe, SecurityPatrol |
| `AudioRecordingSkill` | Record audio to file | Interview, Dictation |
| `WebhookSkill` | Listen for HTTP webhooks from external systems | ShredderSegmenter (active tracking) |

## MediaMTX Setup

MediaMTX is the RTSP relay between Maxim and consumers like ShredderSegmenter. Reachy, Maxim, ffmpeg, and MediaMTX all run on the same host. ShredderSegmenter connects from its own network by pulling the RTSP stream.

```
Reachy + Maxim + ffmpeg + MediaMTX (Network A) ← pull ← ShredderSegmenter (Network B)
```

Maxim auto-starts MediaMTX when the RTSP port isn't in use — just put `mediamtx` on your PATH. For manual/remote setups, see [mediaMTX.md](mediaMTX.md).

## File Structure

```
src/maxim/skills/
├── __init__.py          # Exports: Skill, Protocol, SkillState, SkillResult, WorkspaceBounds
├── base.py              # Skill ABC, SkillConfig, SkillState, SkillResult
├── protocol.py          # Protocol ABC, WorkspaceBounds
├── registry.py          # ProtocolRegistry
├── tools.py             # RunProtocolTool, StopProtocolTool, ListProtocolsTool
├── rtsp_streaming.py    # RTSPStreamingSkill
├── timed_protocol.py    # TimedProtocolSkill (auto-stop after duration)
├── health_reporting.py  # HealthReportingSkill (periodic HTTP health pings)
└── protocols/
    └── shredder_segmenter.py  # ShredderSegmenterProtocol

src/maxim/tools/
└── rtsp_bridge.py       # RTSPBridge (ffmpeg pipe to MediaMTX)

scripts/
└── rtsp_bridge.py       # Standalone CLI for RTSP without agentic loop
```
