# Design Decisions (Maxim)

This file tracks decisions that affect public behavior, repo structure, and long-term maintenance.

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

## 2026-01-02: Default mode is `passive-interaction`
Decision:
- Default `--mode` is `passive-interaction`.
- `Maxim(mode=...)` defaults to `passive-interaction`.

Reason:
- Safer default behavior (no MotorCortex training) while still tracking targets.
- Reduces surprise ML/compute costs when running `python scripts/main.py` with no flags.

Tradeoffs:
- Users who want the previous behavior must pass `--mode live` (or `--mode train`).

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
- Agentic runtime state snapshots are persisted to `data/agents/<AGENT_NAME>/runtime/state_<run_id>.json`.

Reason:
- Support resuming/debugging agent runs with a durable, per-agent state artifact outside the installed package.

## 2026-01-04: Add `--mode agentic` to the CLI
Decision:
- `maxim --mode agentic` runs the agentic runtime loop (`src/maxim/runtime/`) instead of the Reachy orchestration loop.

Reason:
- Provide a first-class entrypoint for agentic development/testing without requiring robot connectivity.

## 2026-01-04: Select agentic agents by `--agent` name
Decision:
- `--mode agentic` accepts `--agent <agent_name>` and selects from a small built-in registry using `Agent.agent_name`.
- Built-in names: `goal` (`GoalAgent`) and `reachy_mini` (`ReachyAgent`).
- Default agent is `reachy_mini`.
- `reachy_mini` uses `ReachyEnv` to observe artifacts under `data/` and can act on `latest_*` paths in state.

Reason:
- Support multiple agents without passing Python classes through the CLI.
- Keep per-agent runtime state organized under `data/agents/<AGENT_NAME>/`.

Tradeoffs:
- New agents must be added to the registry so they are discoverable via `--agent`.

## 2026-01-04: Keep agents in independent files
Decision:
- Each agent implementation should live in its own file under `src/maxim/agents/` (e.g., `reachy_agent.py`, `goal_agent.py`).
- `src/maxim/agents/base.py` should only contain shared interfaces/helpers (`Agent`, `AgentList`, utilities).
- Exception: agents that share nearly all logic via inheritance (or are tightly coupled variants) may be co-located.

Reason:
- Improves discoverability and reduces unrelated coupling as the agent set grows.

## 2026-01-03: Store motion presets under `data/`
Decision:
- Default motion actions load from `data/motion/default_actions.json`.

Reason:
- Keep editable JSON configs separate from code and easy to find.

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
