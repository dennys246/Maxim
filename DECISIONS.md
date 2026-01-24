# Design Decisions (Maxim)

This file tracks decisions that affect public behavior, repo structure, and long-term maintenance.

## 2026-01-18: Agentic MaximAgent naming + GPU-gated runtime
Decision:
- The composite agentic implementation is now `MaximAgent` (alias `AgenticMaximAgent` preserved for compatibility).
- Removed `ReachyMiniAgent`; agentic control now relies on tools for Reachy SDK access.
- `--mode agentic` requires GPU availability before starting.
- Vision events are streamed to `data/vision/vision_events_<run_id>.jsonl` and surfaced as `latest_vision_event`.
- `execute_file` tool execution is opt-in via `MAXIM_ALLOW_EXECUTE_FILE=1`.

Reason:
- Align the primary agent name with the agentic implementation.
- Keep agents action-free and centralize SDK control in tools.
- Avoid running the agentic loop without accelerator support.
- Feed YOLO detections into the agentic perception loop without blocking control.
- Reduce the risk of transcript-triggered arbitrary file execution.

Tradeoffs:
- Existing imports referencing the old class names should update (aliases remain).
- Agentic runs now exit/skip on GPU-less machines.

## 2026-01-07: Add interactive terminal input for keyword actions
Decision:
- `maxim` starts a line-based terminal prompt (`maxim>`) when `--interactive true` (default).
- Prompted input is matched against `phrase_responses.json` (same as voice triggers).
- Single-key shortcut mode is disabled while interactive prompt input is enabled to avoid stdin conflicts.
- Interactive CLI input is recorded under `data/cli/cli_input_<run_id>.jsonl`.
- CLI and vision-overlay inputs bypass phrase cooldowns and `requires_agentic` gating.
- When the OpenCV display is active, the vision overlay includes a text input box with a Send button that routes input through the same phrase responses.

Reason:
- Provide a reliable non-audio control path for keyword actions without adding new dependencies.

Tradeoffs:
- Keypress-only shortcuts require `--interactive false` or Enter in the prompt.
- The overlay input requires a direct OpenCV display backend; process-based display modes cannot capture input.

## 2026-01-02: Queue-based capture + writer pipeline
Reason:
- Avoid blocking perception/motor control on disk I/O.
- Enable “record everything” semantics by applying backpressure (blocking queues) instead of dropping samples.

Tradeoffs:
- More moving parts (threads/process + shutdown signaling).
- When disk/CPU can’t keep up, capture blocks and effective FPS may decrease.

## 2026-01-02: Single-run artifacts (MP4 + WAV + JSONL transcript)
Reason:
- A single `videos/*.mp4` is more efficient and simpler than thousands of PNGs.
- A single `audio/*.wav` preserves a continuous audio stream; JSONL allows streaming transcript append.

Tradeoffs:
- Requires codecs/backends for MP4 writing (environment dependent).
- Large files require log/cleanup discipline.

## 2026-01-04: Store transcripts under `data/transcript/`
Decision:
- JSONL transcripts are written under `data/transcript/` (previously `data/text/`).

Reason:
- Avoid confusion with generic “text” outputs and make transcripts easier to locate.

## 2026-01-02: Whisper transcription runs in a separate process
Reason:
- Whisper inference is heavy and should not stall the control loop.
- Process isolation avoids GIL contention and keeps the rest of the system responsive.

Tradeoffs:
- Whisper dependency/model availability may be missing; transcription must degrade gracefully.
- Requires chunking audio and coordinating handoff via a queue.

## 2026-01-02: Optional audio pipeline via CLI flags
Decision:
- `--audio True/False` controls audio capture/transcription.
- `--audio_len <seconds>` controls chunk size for efficient streaming transcription.

Reason:
- Some runs are vision-only; audio should be skippable.
- Chunking balances latency (short chunks) vs throughput (long chunks).

## 2026-01-02: `--mode sleep` skips `wake_up()`
Decision:
- `--mode sleep` records/transcribes audio without running the camera/ML loop and does not call `ReachyMini.wake_up()`.

Reason:
- Support “leave motors asleep” debugging and audio-only dataset capture.

Tradeoffs:
- The run won’t auto-stop based on frame epochs; it runs until interrupted.

## 2026-01-02: Default mode is `exploration`
Decision:
- Default `--mode` is `exploration`.
- `Maxim(mode=...)` defaults to `exploration`.

Reason:
- Exploration mode actively discovers and learns about the environment.
- Maxim is immediately curious and engaged on startup.
- Aligns with the goal of building understanding through active observation.

Tradeoffs:
- Higher resource usage than passive modes like `sleep` or `reflection`.
- Users who want minimal activity can pass `--mode sleep` or `--mode reflection`.

## 2026-01-02: Per-run logs saved under `data/logs/`
Decision:
- Each CLI run writes logs to `data/logs/reachy_log_<run_id>.log`.

Reason:
- Makes runs debuggable after the fact without copying terminal output.
- Keeps artifacts grouped per session alongside video/audio/transcripts.

Tradeoffs:
- Produces additional files; users may need periodic cleanup.

## 2026-01-02: Inference code lives under `src/maxim/inference/`
Reason:
- Keep “runtime inference/control” separate from “robot orchestration” (`src/maxim/conscience/`) and “model definitions” (`src/maxim/models/`).

Tradeoffs:
- Requires stable re-export modules to preserve import paths during refactors.

## 2026-01-02: Vision via YOLOv8 (segmentation + pose)
Reason:
- Fast, general-purpose perception for “person/object of interest” detection.
- Pose keypoints enable eye/face target refinement when available.

Tradeoffs:
- Heavier runtime dependency; performance depends on hardware.
- Model weights and backends vary by environment.

## 2026-01-02: MotorCortex uses ConvNeXt-Tiny backbone
Decision:
- MotorCortex predicts head movement deltas: `[x, y, z, roll, pitch, yaw, duration]`.

Reason:
- Strong image feature extractor that trains well for regression with minimal custom code.

Tradeoffs:
- Requires TensorFlow/Keras for training/inference in this repo’s implementation.

## 2026-01-02: Add `maxim` CLI entrypoint
Decision:
- `pip install -e .` installs a `maxim` console script (entrypoint: `maxim.cli:main`).
- The importable package is `maxim` (code lives under `src/maxim/`); `src.*` imports are removed.
- `python scripts/main.py` remains supported as a compatibility entrypoint.

Reason:
- Reduce friction for new users (no need to remember the module/file path).
- Avoid confusion from a top-level package named `src`.

## 2026-01-02: JSON-configured key responses
Decision:
- Maxim loads `data/util/key_responses.json` on startup and listens for terminal key presses while running (override via `$MAXIM_KEY_RESPONSES`).

Reason:
- Allow quick, extensible runtime actions (e.g., recenter vision) without impacting the control loop.

## 2026-01-04: Training sample log under `data/training/`
Decision:
- When vision-driven movement is initiated, Maxim appends a JSONL record to `data/training/motor_training_set.jsonl` via a background writer.
- The `u` key writes a marked record (`user_marked=true`) for the most recent sample.

Reason:
- Keep an always-on stream of “trainable moments” for MotorCortex without blocking the control loop.
- Make it easy to curate a subset of samples for training by marking moments during a run.

Tradeoffs:
- Samples reference run artifacts (video/audio/transcript paths + timestamps); extracting frames is a post-processing step.

## 2026-01-04: Phrase-triggered actions from transcripts + event labels
Decision:
- Maxim can trigger actions from transcribed speech using `data/util/phrase_responses.json` (override via `$MAXIM_PHRASE_RESPONSES`).
- The default wake words are `Maxim` and `Reachy`, which call `wake_up()`, start the agentic runtime loop, and enable voice-triggered actions.
- Voice commands `Maxim shutdown`, `Maxim sleep`/`sleep maxim`, and `Maxim observe`/`observe maxim` request clean shutdown / mode switches (the CLI restarts Maxim into the requested mode).
- When a non-wake command phrase matches, wake-word triggers are suppressed for that transcript line to avoid double actions.
- Transcript text is normalized before matching (punctuation/possessives stripped; common alias `maximum` → `maxim`).
- When `maxim` is present in a transcript line, Maxim also attempts to infer the best matching non-wake command from the remaining words before falling back to the wake action (and does not re-fire the wake action once enabled).
- Runtime events (voice/key actions + user outcome labels) are appended to `data/training/action_events.jsonl` via the same background writer used for training samples.
- Keys `0`–`9` are reserved for simple outcome labels (`0` = no errors; `1`–`9` = generic error/odd behavior codes).

Reason:
- Tie transcripts, actions, and “trainable moments” together via time-aligned JSONL logs.
- Support lightweight human-in-the-loop labeling during runs without blocking the control loop.

## 2026-01-05: Optional local LLM routing for wake-word transcripts
Decision:
- When the agentic runtime is running, transcript lines that contain the wake word (`maxim` + common variants like `maximum`) may be routed through an optional local LLM to produce a single agentic action (`{"tool_name": ..., "params": ...}`).
- Hard keyword commands for mode switching (`sleep/observe/shutdown` with `maxim`) always override LLM routing.
- LLM configuration is stored in `data/util/llm.json` (override via `$MAXIM_LLM_CONFIG`) and is disabled by default.
- Initial reference backend uses `llama-cpp-python` (local GGUF) with built-in profiles for Mistral 7B and SmolLM 1.7B.
- LLM backends live under `src/maxim/models/language/` to keep them swappable.

Reason:
- Keep voice control deterministic for critical mode switches while enabling richer, optional transcript-driven behaviors when compute is available.

## 2026-01-06: Configure Whisper compute type via env var
Decision:
- `MAXIM_WHISPER_COMPUTE_TYPE` controls the `faster-whisper` compute type for transcription (default: `int8`).

Reason:
- Provide a safe fallback for Linux/WSL segfaults in CTranslate2/Whisper without code edits.

Tradeoffs:
- `float32` is slower and may increase CPU usage; `int8` is faster but less stable on some systems.

## 2026-01-06: Disable OpenCV display in headless/non-main thread runs
Decision:
- `MAXIM_DISABLE_IMSHOW=1` or `MAXIM_HEADLESS=1` skips `cv2.imshow` calls to avoid Qt/GTK thread crashes on WSL/headless setups.
- `MAXIM_IMSHOW_MODE=process` runs `cv2.imshow` in a dedicated process to keep GUI calls on that process's main thread.
- On Linux/WSL, default to the display process; set `MAXIM_IMSHOW_MODE=direct` to force main-thread imshow.

Reason:
- OpenCV GUI backends often crash when invoked from non-main threads or without a display server.

Tradeoffs:
- No on-screen visualization during runs; rely on saved videos/logs instead.
- Display process adds IPC overhead and may drop frames under load.

## 2026-01-05: CLI model selection flags
Decision:
- `--language-model <profile>` overrides the LLM profile for the run (prints available profiles on unknown).
- `--segmentation-model <name>` selects the vision segmenter (default: `YOLO8`; prints available models on unknown).

Reason:
- Make per-run experimentation easier without editing JSON/env vars.

## 2026-01-04: Agentic decision flow + single point of decision
Decision:
- Action selection happens in exactly one place: `src/maxim/planning/decision_engine.py`.
- Canonical flow (no skipping): Observe state → Agents propose intents → Planners propose candidate plans → Policies constrain plans → Decision engine selects one next action → Runtime executes.
- Planners generate plans but do not select final actions or mutate state.
- Policies are deterministic/auditable guardrails and do not plan or execute.
- “Hidden decisions” are forbidden: if a component chooses between alternatives, prioritizes options, or suppresses actions, that logic belongs in the decision engine.

Reason:
- Keep behavior predictable, testable, and debuggable as the codebase grows.
- Prevent side effects and control-flow decisions from leaking into the wrong layers.

Tradeoffs:
- Requires discipline and occasional refactors to keep boundaries intact.
- Some features may need more explicit state representation and dependency injection to remain testable.

## 2026-01-04: Standardize agentic plan/action schema
Decision:
- Canonical action schema: `{"tool_name": <str>, "params": <dict>}`.
- Canonical plan schema: `list[action]`.
- `DecisionEngine.decide()` returns a dict containing the selected `action` and its `plan` context.
- Agentic orchestration lives under `src/maxim/runtime/` and executes actions via `Executor` + `ToolRegistry`.

Reason:
- Keep planner outputs, policy checks, evaluators, and runtime execution interoperable.
- Reduce “stringly-typed” ambiguity and make plans serializable/debuggable.

## 2026-01-04: Persist agentic runtime state under `data/agents/`
Decision:
- Agentic runtime state snapshots are persisted to `data/agents/<STATE_NAME>/runtime/state_<run_id>.json`.
- `STATE_NAME` defaults to `Agent.agent_name`, but agents may override it via `state_name`.

Reason:
- Support resuming/debugging agent runs with a durable, per-agent state artifact outside the installed package.

## 2026-01-04: Add `--mode agentic` to the CLI
Decision:
- `maxim --mode agentic` runs the agentic runtime loop (`src/maxim/runtime/`) instead of the Reachy orchestration loop.

Reason:
- Provide a first-class entrypoint for agentic development/testing without requiring robot connectivity.

## 2026-01-04: Agentic runtime defaults
Decision:
- `--mode agentic` runs the composite `MaximAgent` (agentic architecture).
- Alternate agent selection is not exposed via the CLI at the moment.

Reason:
- Keep the agentic entrypoint focused on the primary architecture.

Tradeoffs:
- Switching agents requires code changes rather than a CLI flag.

## 2026-01-04: Keep agents in independent files
Decision:
- Each agent implementation should live in its own file under `src/maxim/agents/` (e.g., `maxim_agent.py`, `goal_agent.py`).
- `src/maxim/agents/base.py` should only contain shared interfaces/helpers (`Agent`, `AgentList`, utilities).
- Exception: agents that share nearly all logic via inheritance (or are tightly coupled variants) may be co-located.

Reason:
- Improves discoverability and reduces unrelated coupling as the agent set grows.

## 2026-01-03: Store motion presets under `data/`
Decision:
- Default motion actions load from `data/motion/default_actions.json`.

Reason:
- Keep editable JSON configs separate from code and easy to find.

## 2026-01-05: Preflight Matplotlib font cache before loading vision models
Decision:
- Before loading vision models, Maxim runs a Matplotlib font-cache preflight in a subprocess.
- The preflight uses `MPLCONFIGDIR` under the run's home directory when not already set and forces `MPLBACKEND=Agg`.
- `MAXIM_SKIP_MPL_PREFLIGHT=1` bypasses the preflight when needed.
- Maxim also preloads Matplotlib in-process early (before Reachy/GStreamer/Ultralytics init) to stabilize native font libs.
- `MAXIM_SKIP_MPL_PRELOAD=1` bypasses the early preload when needed.

Reason:
- On Linux/WSL, Matplotlib + FreeType can abort while scanning fonts; preflighting isolates failures and surfaces actionable errors.

Tradeoffs:
- Adds a small startup cost to vision initialization.
- Users may need to clean or repair system fonts if preflight fails.

## 2026-01-06: Allow disabling VAD filter for faster-whisper transcription
Decision:
- `MAXIM_VAD_FILTER=0` disables the faster-whisper VAD filter when running the transcription worker.

Reason:
- VAD uses onnxruntime (Silero ONNX) and can segfault on some Linux builds; the toggle lets users isolate or bypass that path.

Tradeoffs:
- Without VAD, transcription may be slower and include more silence.

## 2026-01-06: Default epochs to unlimited
Decision:
- CLI `--epochs` defaults to `0` (unlimited).
- `Maxim` treats epochs `<= 0` as unlimited; `Maxim.live(epochs=...)` overrides.
- Agentic `max_steps <= 0` runs without a step cap.

Reason:
- Prevent unexpected stops in long-running sessions unless the user explicitly sets a limit.

Tradeoffs:
- Users must pass `--epochs` to cap runtime by default.

## 2026-01-04: Store head poses under `data/motion/default_poses.json`
Decision:
- Default head poses (including the `centered` pose used by the `c` key) load from `data/motion/default_poses.json`.

Reason:
- Allow robot-specific calibration of “centered” without changing code.

## 2026-01-05: Clamp head movement step size
Decision:
- Head movement commands are clamped per call using `data/motion/movement_thresholds.json` to avoid large, sudden jumps.

Reason:
- Improve stability/safety and make movement behavior tunable without changing code.

## 2026-01-03: Store trained models under `data/models/`
Decision:
- Default model artifacts (MotorCortex checkpoints/history, YOLO weights) live under `data/models/`.

Reason:
- Keep model artifacts separate from run outputs under `data/`.

## 2026-01-03: Extract reusable helpers from nested defs
Decision:
- Avoid defining reusable helper functions inside other functions/methods.
- Put cross-cutting helpers under `src/maxim/utils/` (or at module scope) and import them where needed.

Reason:
- Improve reuse and reduce duplicated logic while keeping runtime loops readable.

## 2026-01-04: Keep Python code under the `maxim` namespace
Decision:
- Importable code lives under `src/maxim/` (packaged as `maxim*`).
- Avoid creating new top-level packages under `src/` (e.g., `src/agents/`) unless `pyproject.toml` explicitly includes them.

Reason:
- Ensures `pip install -e .` installs everything needed for imports and avoids collisions with overly-generic package names.

## 2026-01-19: Architecture migration - `live()` as hardware I/O layer

### Current State

The system has two parallel control paths:

1. **`live()` loop** (`src/maxim/conscience/selfy.py`):
   - Hardware I/O: frame capture, audio capture, video/audio writing
   - Media recording: saves MP4/WAV files for training data
   - Transcription pipeline: spawns Whisper process for speech-to-text
   - CLI/keyboard listeners: user input handling
   - Observation functions: `passive_observation()` / `motor_cortex_control()`
   - Display: shows annotated frames via OpenCV

2. **Agentic runtime** (`src/maxim/runtime/`, `src/maxim/agents/`):
   - PerceptionAgent: processes frames/audio into Percepts
   - MemoryAgent: builds StructuredContext from percepts
   - AgenticGoalAgent: proposes goals based on context
   - ExecAgent: executes goals via tool calls
   - AutonomyController: gates tool execution by autonomy level
   - LLMWorker: non-blocking LLM inference

### Intended Migration Path

**Phase 1 (Current):** Keep both paths, document boundaries.
- `live()` remains the hardware interface layer
- `passive_observation()` / `motor_cortex_control()` are fallbacks when agentic runtime is inactive
- Agentic runtime consumes data from `live()` via shared state (`_last_frame`, transcripts)

**Phase 2:** Make agentic runtime the primary decision-maker.
- `live()` becomes a pure capture/recording layer (no observation logic)
- All perception → decision → action flows through the agentic system
- `passive_observation()` becomes a simple "display frame + detections" helper
- Remove `motor_cortex_control()` training logic (training moves to offline pipeline)

**Phase 3:** Merge capture threads into agentic runtime.
- Move frame/audio capture workers into `_start_agentic_runtime()`
- `live()` becomes a thin wrapper that starts the agentic runtime
- Single entry point for all modes (sleep/observe/agentic)

### Key Boundaries (Current)

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| `live()` | Hardware I/O, recording, display | Decision-making, goal selection |
| `passive_observation()` | Legacy fallback: segment + display + simple tracking | Goal proposal, LLM reasoning |
| PerceptionAgent | Convert raw data → Percepts | Movement commands, tool calls |
| MemoryAgent | Build context, manage memories | Propose goals, execute actions |
| AgenticGoalAgent | Propose goals from context | Execute tools directly |
| ExecAgent | Execute approved actions via tools | Propose goals, bypass autonomy |
| AutonomyController | Gate tool execution | Make decisions, propose goals |

### Migration Checklist

**Phase 2 (Completed 2026-01-19):**
- [x] Display logic extracted from `passive_observation()` into `display_detections()` standalone helper
- [x] `passive_observation()` simplified to display-only (returns target info, no movement)
- [x] `motor_cortex_control()` removed from `live()` observation loop
- [x] `live()` now auto-starts agentic runtime when not in sleep mode
- [x] Target info stored in `_last_detection_target` for agentic system access

**Phase 3 (Completed 2026-01-19):**
- [x] Created `CaptureManager` class (`src/maxim/runtime/capture.py`) for unified frame/audio capture
- [x] PerceptionAgent directly receives frames via CaptureManager callbacks (bypasses JSONL polling)
- [x] MaximAgent accepts `capture_manager` parameter and passes to PerceptionAgent
- [x] `live()` observation loop uses CaptureManager's pre-segmented frames when available
- [x] `display_detections()` updated to handle both tuple and dict detection formats
- [x] Single entry point: `live()` is the unified entry (sleep/observe/agentic modes)
- [x] `sleep()` calls `live(vision=False, motor=False, wake_up=False)`

### Current Data Flow (Phase 3)

```
CaptureManager (agentic runtime)
    ↓ direct frame capture + YOLO segmentation
    ↓ callback notification
PerceptionAgent._on_captured_frame()
    ↓
Percept published to AgentBus
    ↓
MemoryAgent → StructuredContext
    ↓
AgenticGoalAgent → Goals
    ↓
ExecAgent → Tool calls (via AutonomyController)

live() display loop (parallel)
    ↓ polls CaptureManager.get_latest_frame()
    ↓ display_detections() for visualization
```

### Key Changes in Phase 3

1. **CaptureManager** (`src/maxim/runtime/capture.py`):
   - Unified capture for frame and audio data
   - Direct YOLO segmentation in capture thread
   - Callback-based notification to PerceptionAgent
   - Bypasses JSONL intermediary for lower latency

2. **PerceptionAgent updates**:
   - Accepts optional `capture_manager` in constructor
   - Registers `_on_captured_frame()` callback for direct frame processing
   - `process_captured_frame()` public API for manual frame processing

3. **Detection format normalization**:
   - `_normalize_detection()` helper handles both tuple and dict formats
   - `display_detections()` works with CaptureManager's dict output

### Tradeoffs

- **Keeping `live()` for recording:** Still needed for video/audio file writing; CaptureManager focuses on agentic perception
- **Dual capture paths:** CaptureManager captures for agentic system; live()'s threads still write to disk
- **Fallback observation:** `passive_observation()` still available when CaptureManager unavailable
- **Direct callbacks:** Lower latency but tighter coupling between CaptureManager and PerceptionAgent

## 2026-01-19: Active visual tracking via TrackTargetTool

### Decision

Added `TrackTargetTool` to enable the agentic system to actively move the head to center detected objects of interest.

### Implementation

**New Tool:** `track_target` (`src/maxim/tools/reachy.py`)
- Reads detection targets from CaptureManager (Phase 3) or `_last_detection_target` (fallback)
- Computes if target is outside configurable deadzone from frame center
- Calls `look_at_image()` to move head and center the target
- Parameters:
  - `deadzone_px`: Minimum offset from center to trigger movement (default: 40)
  - `duration_s`: Movement duration in seconds (default: 0.3)
  - `prefer_people`: Prioritize people over other objects (default: true)

**ExecAgent Changes:**
- Added `track_target` to available tools in system prompt
- Updated default behavior: when detections present, propose `track_target` instead of just `focus_interests`
- Added guidelines encouraging tracking behavior for people/objects

**Flow:**
```
CaptureManager → YOLO detections
    ↓
ExecAgent sees detected_objects/detected_people in StructuredContext
    ↓
Proposes track_target goal (MEDIUM priority)
    ↓
TrackTargetTool reads latest detections
    ↓
If target outside deadzone: look_at_image(u, v, duration)
    ↓
Head centers on target
```

### Reason

- Enables proactive visual engagement without explicit voice commands
- Makes Maxim appear more attentive and aware of surroundings
- Leverages existing detection pipeline for active tracking
- Respects deadzone to avoid jitter from small movements

### Tradeoffs

- **Continuous movement:** May be distracting; deadzone helps mitigate
- **Rate limiting:** 10 Hz cap prevents excessive motor commands
- **LLM-optional:** Default tracking works without LLM; LLM can propose higher-priority goals to override
- **People preference:** May miss interesting non-person objects when people are present

## 2026-01-19: Enhanced Verbosity System for Agentic Information Flow

### Decision

Updated the verbosity system to provide granular control over agentic logging, making it easier to debug and observe the perception-memory-goal-action pipeline.

### Implementation

**New Verbosity Levels** (`src/maxim/utils/structured_logging.py`):
- **Level 0 (QUIET):** Errors and critical warnings only
- **Level 1 (NORMAL):** Key events - goal proposals, tool executions, mode changes
- **Level 2 (VERBOSE):** + Perception events, memory updates, autonomy decisions
- **Level 3 (DEBUG):** + Loop iterations, rate limiting, internal state changes

**Event Categories:**
Events are categorized with minimum verbosity levels:
```python
EVENT_VERBOSITY = {
    # Level 0: Always shown
    "error": 0, "critical": 0, "hard_stop": 0,

    # Level 1: Key events
    "goal_proposed": 1, "tool_called": 1, "tool_result": 1,
    "mode_change": 1, "action_executed": 1, "action_rejected": 1,

    # Level 2: Detailed events
    "percept": 2, "detection": 2, "memory_store": 2,
    "autonomy_check": 2, "intent_proposed": 2,

    # Level 3: Debug events
    "loop_iteration": 3, "rate_limited": 3, "idle": 3,
}
```

**CLI Arguments:**
- `--agentic-verbosity {0,1,2,3}`: Set agentic logging verbosity (default: 1)
- `--agentic-console`: Print agentic events to console in real-time

**Environment Variables:**
- `MAXIM_AGENTIC_VERBOSITY`: Default verbosity level (0-3)
- `MAXIM_AGENTIC_CONSOLE`: Enable console output ("1", "true", "yes")

**New API Functions:**
```python
# Configure globally
configure_agentic_verbosity(verbosity=2, console_output=True)

# Log directly to abstraction stream
log_agentic("track_target", "detection", {"target_u": 500, "is_person": True})

# Get buffer for inspection
buf = get_abstraction_buffer()
print(buf.get_recent_human(10))  # Human-readable output
print(buf.get_summary())  # Event/source counts
```

**LogRecord Formats:**
- `to_compact()`: Minimal JSON for LLM context (`{"t":1234.5,"s":"exec_agent","e":"goal_proposed"}`)
- `to_verbose()`: Full field names for debugging
- `to_human()`: Human-readable console format (`12:34:56.789 [exec_agent] goal_proposed | tool=track_target`)

**Agent Loop Integration:**
The `run_agentic_loop()` now logs throughout the pipeline:
- Loop iterations (level 3)
- Intent proposals from agent fallback (level 2)
- Autonomy checks (level 2)
- Tool calls and results (level 1)
- Errors and hard stops (level 0)

### Reason

- **Debugging:** Easier to trace why actions happen or don't happen
- **Observability:** Clear visibility into the perception→goal→action pipeline
- **Flexibility:** Different verbosity for development vs production
- **Non-intrusive:** Default level 1 shows key events without flooding logs

### Tradeoffs

- **Performance:** Level 3 logging adds overhead; use level 1-2 in production
- **Storage:** AbstractionBuffer is limited to 500 entries by default
- **Complexity:** Multiple output formats (compact/verbose/human) to maintain
